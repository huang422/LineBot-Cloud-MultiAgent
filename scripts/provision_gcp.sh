#!/usr/bin/env bash
# One-time GCP hardening for LineBot-CloudAgent (ADR-0008).
#
# Dry run by default: every gcloud command that would change something is
# printed, never run. Read-only checks (describe, list, get-iam-policy) do
# run, so the plan shows what already exists; every step describes before it
# creates, so re-running a step is safe. Only NOT_FOUND counts as missing
# (and, in a dry run, an API the apis step enables being off): any other
# read error stops the script. Stays compatible with macOS
# /bin/bash 3.2: no mapfile, no ${var,,}, and no "${array[@]}" expansion of
# an array that can be empty under `set -u`.
set -euo pipefail

# gcloud never prompts: a lookup's stderr goes to a file, where a question
# such as "API [x] not enabled ... enable and retry? (y/N)" would wait unseen.
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
PROJECT_ID=""
REGION=""
APPLY="false"
YES_PURGE="false"
STEP=""
LIST="false"

# Names shared with scripts/deploy_cloud_run.sh (it detects them).
RUNTIME_SA_NAME="linebot-runtime"
BUILDER_SA_NAME="linebot-builder"
DOTENV_SECRET="linebot-dotenv"
# The Colab tier's credential (ADR-0010). scripts/mint_colab_adc.sh creates
# it (the token needs a person at a browser); the colab-secret step grants
# the runtime SA read access.
COLAB_SECRET="linebot-colab-adc"
AR_REPOSITORY="linebot-cloud-agent"
REQUIRED_APIS="cloudtasks.googleapis.com secretmanager.googleapis.com"
AR_CLEANUP_POLICIES="keep-2-most-recent delete-older-than-1d"

LIFECYCLE_3D="$ROOT_DIR/infra/gcp/lifecycle-delete-3d.json"
LIFECYCLE_1D="$ROOT_DIR/infra/gcp/lifecycle-delete-1d.json"
AR_CLEANUP_FILE="$ROOT_DIR/infra/gcp/ar-cleanup.json"

# `all` runs these in order; ar-cleanup-enforce and drop-editor only run on
# their own.
ALL_STEPS="apis runtime-sa staging-bucket builder-sa purge-cloudbuild secrets colab-secret ar-cleanup budget summary"
KNOWN_STEPS="$ALL_STEPS ar-cleanup-enforce drop-editor"

usage() {
  cat <<'USAGE'
Usage: ./scripts/provision_gcp.sh [--apply] [--step NAME] [options]

One-time GCP setup for LineBot-CloudAgent: least-privilege service accounts,
a us-west1 Cloud Build staging bucket, the Secret Manager dotenv secret, read
access to the Colab credential secret, the Artifact Registry cleanup policy
and the purge of the old Cloud Build tarballs. See DEPLOY.md,
docs/adr/0008-secrets-and-identity.md and docs/adr/0010-colab-ollama-tier.md.

Without --apply nothing changes: every command that would change something
is printed ([dry-run]); read-only checks still run.

Options:
  --apply            Run the printed commands
  --step NAME        Run one step only (see --list); recommended with --apply
  --list             List the steps and exit
  --yes-purge        Allow the purge-cloudbuild step to delete objects
  --project-id ID    GCP project (default: GCP_PROJECT_ID in the env file, which it
                     must match; never gcloud's default project)
  --region REGION    Region (default: CLOUD_RUN_REGION in .env, then us-west1)
  --env-file PATH    Env file (default: .env); the secrets step uploads it. Missing,
                     unreadable or unparsable, it stops every step (missing is
                     allowed with --project-id)
  -h, --help         Show this help message
USAGE
}

list_steps() {
  cat <<'STEPS'
apis                enable cloudtasks and secretmanager (never container scanning)
runtime-sa          linebot-runtime: datastore.user, objectUser on the media bucket,
                    tokenCreator on itself, enqueuer on the queues, accessor per secret
staging-bucket      gs://<project>-cb-src-usw1: us-west1, STANDARD, uniform access,
                    public access prevention, no soft delete, delete after 3 days
builder-sa          linebot-builder: artifactregistry.writer on the repo,
                    logging.logWriter, objectViewer on the staging bucket
purge-cloudbuild    clear soft delete on gs://<project>_cloudbuild, delete source/**,
                    then a 1-day lifecycle rule (needs --yes-purge with --apply)
secrets             create linebot-dotenv from .env (automatic replication, via stdin)
colab-secret        let linebot-runtime read linebot-colab-adc (the Colab token, which
                    ./scripts/mint_colab_adc.sh --apply creates; skipped until it exists)
ar-cleanup          set the Artifact Registry cleanup policy in dry-run mode
ar-cleanup-enforce  switch the cleanup policy to deleting (only with --step)
drop-editor         remove roles/editor from the Compute Engine default service account
                    (never the account) once linebot-runtime runs the service, builds run
                    as linebot-builder and nothing else acts as it (only with --step)
budget              print the console steps for the ~30 TWD budget alert
summary             what still needs a person
STEPS
}

fail() {
  echo "[fail] $*" >&2
  exit 1
}

warn() {
  echo "[warn] $*" >&2
}

note() {
  echo "       $*"
}

format_cmd() {
  local out="" arg
  for arg in "$@"; do
    out="$out $(printf '%q' "$arg")"
  done
  printf '%s' "${out# }"
}

# Print a command that changes something; run it only with --apply.
run() {
  if [[ "$APPLY" == "true" ]]; then
    echo "  [run] $(format_cmd "$@")"
    "$@"
  else
    echo "  [dry-run] $(format_cmd "$@")"
  fi
}

# Like run, for IAM bindings: a service account created a moment ago can
# take a few seconds to become visible to IAM.
run_binding() {
  local attempt
  if [[ "$APPLY" != "true" ]]; then
    run "$@"
    return
  fi
  echo "  [run] $(format_cmd "$@")"
  for attempt in 1 2 3; do
    if "$@" >/dev/null; then
      return 0
    fi
    [[ "$attempt" -lt 3 ]] && sleep 5
  done
  fail "IAM binding failed three times: $(format_cmd "$@")"
}

# probe ARGS...: a read-only gcloud call. Returns 0 with the output in
# LOOKUP_OUT when it succeeds, 1 when gcloud says the resource was not
# found, and 2 on any other error (its message in LOOKUP_ERR).
LOOKUP_OUT=""
probe() {
  local rc=0
  LOOKUP_OUT="$(gcloud "$@" 2>"$LOOKUP_ERR")" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi
  # NOT_FOUND / "was not found" / "gs://x not found: 404" / "Cannot find".
  if grep -Eqi 'not[ _]found|cannot find|HTTPError 404' "$LOOKUP_ERR"; then
    return 1
  fi
  return 2
}

# Prints the API of REQUIRED_APIS that the last probe failed on for being
# off (PERMISSION_DENIED, reason SERVICE_DISABLED); nothing otherwise.
disabled_required_api() {
  local api
  grep -q 'SERVICE_DISABLED' "$LOOKUP_ERR" || return 0
  for api in $REQUIRED_APIS; do
    if grep -qF "$api" "$LOOKUP_ERR"; then
      printf '%s' "$api"
      return 0
    fi
  done
}

# Succeeds when the last probe failed because API $1 is off in the
# project: PERMISSION_DENIED with reason SERVICE_DISABLED, or, as gcloud
# compute words it, "... has not been used in project ... or it is disabled".
api_is_off() {
  grep -qF "$1" "$LOOKUP_ERR" \
    && grep -Eq 'SERVICE_DISABLED|has not been used in project|it is disabled' "$LOOKUP_ERR"
}

# lookup ARGS...: like probe, but any error other than NOT_FOUND
# (permission denied, API disabled, expired login, network) stops the
# script, so a failed read is never reported as "already gone" or
# "nothing to purge". Two exceptions read as not there: a dry run reading
# an API that the apis step enables while it is still off, and a read of
# the API named in OFF_API_MEANS_NONE (set it for that one call) while that
# API is off, as a project without the API has none of its resources.
# Never call it inside $(...), where the stop would only end the subshell.
lookup() {
  local rc=0 api=""
  probe "$@" || rc=$?
  if [[ "$rc" -eq 2 ]]; then
    if [[ -n "${OFF_API_MEANS_NONE:-}" ]] && api_is_off "$OFF_API_MEANS_NONE"; then
      note "$OFF_API_MEANS_NONE is not enabled in project $PROJECT_ID, so this reads as: none"
      return 1
    fi
    api="$(disabled_required_api)"
    if [[ -n "$api" ]] && [[ "$APPLY" != "true" ]]; then
      note "$api is not enabled yet, so this reads as: does not exist yet (the apis step enables the API)"
      return 1
    fi
    cat "$LOOKUP_ERR" >&2
    if [[ -n "$api" ]]; then
      fail "$api is not enabled in project $PROJECT_ID; run --step apis --apply first (if it just ran, wait a few minutes), then run the step again"
    fi
    fail "gcloud $1 $2 $3 failed, and not with NOT_FOUND; fix the access (gcloud auth login, the account's roles, the API), then run the step again"
  fi
  return "$rc"
}

# Succeeds when a lookup finds the resource and prints something.
exists() {
  lookup "$@" && [[ -n "$LOOKUP_OUT" ]]
}

# $ENV_FILE as `envfile.py dump` reads it, loaded once by load_env_file (a
# python3 start per key read adds up). Never exported. The file must pass
# the secrets step's check (readable UTF-8, every line KEY=VALUE, nothing
# python-dotenv would read differently) before anything reads it: a key it
# silently dropped would take its default, GCP_PROJECT_ID included. Only an
# explicit --project-id runs without an env file.
ENV_KEYS=()
ENV_VALUES=()
load_env_file() {
  local key value
  if [[ ! -e "$ENV_FILE" ]]; then
    [[ -n "$PROJECT_ID" ]] || fail "Env file not found: $ENV_FILE. Pass --env-file, or --project-id to run without one"
    warn "Env file not found: $ENV_FILE; every setting but the project takes its default"
    return 0
  fi
  [[ -r "$ENV_FILE" ]] || fail "Cannot read env file: $ENV_FILE"
  if ! python3 "$ROOT_DIR/scripts/envfile.py" to-dotenv --file "$ENV_FILE" --sha256 >/dev/null; then
    fail "Fix the keys and lines listed above in $ENV_FILE (it must be UTF-8), then run this again"
  fi
  # Each key comes with its value and a placeholder flag, unused here.
  while IFS= read -r -d '' key && IFS= read -r -d '' value && IFS= read -r -d '' _; do
    ENV_KEYS+=("$key")
    ENV_VALUES+=("$value")
  done < <(python3 "$ROOT_DIR/scripts/envfile.py" dump --file "$ENV_FILE" 2>/dev/null)
}

# Succeeds when $ENV_FILE sets KEY (even to an empty value); ENV_INDEX is
# then its position in ENV_KEYS.
ENV_INDEX=0
env_has_key() {
  ENV_INDEX=0
  while [[ "$ENV_INDEX" -lt "${#ENV_KEYS[@]}" ]]; do
    [[ "${ENV_KEYS[$ENV_INDEX]}" == "$1" ]] && return 0
    ENV_INDEX=$((ENV_INDEX + 1))
  done
  return 1
}

# Prints KEY's value like `envfile.py get`; nothing when it is not set.
read_env_key() {
  if env_has_key "$1"; then
    printf '%s\n' "${ENV_VALUES[$ENV_INDEX]}"
  fi
}

# stdin: an IAM policy as JSON. Succeeds when MEMBER holds ROLE unconditionally.
policy_has_binding() {
  python3 -c '
import json, sys
role, member = sys.argv[1], sys.argv[2]
try:
    policy = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
for binding in (policy or {}).get("bindings") or []:
    if binding.get("role") == role and member in (binding.get("members") or []) and not binding.get("condition"):
        sys.exit(0)
sys.exit(1)
' "$1" "$2"
}

# grant SCOPE RESOURCE MEMBER ROLE: add the binding unless it is there.
grant() {
  local scope="$1" resource="$2" member="$3" role="$4" policy="" found=0
  # NOT_FOUND: an earlier step of this dry run creates the resource.
  case "$scope" in
    project)
      lookup projects get-iam-policy "$PROJECT_ID" --format=json || found=$?
      ;;
    bucket)
      lookup storage buckets get-iam-policy "$resource" --project "$PROJECT_ID" --format=json || found=$?
      ;;
    sa)
      lookup iam service-accounts get-iam-policy "$resource" --project "$PROJECT_ID" --format=json || found=$?
      ;;
    queue)
      lookup tasks queues get-iam-policy "$resource" --project "$PROJECT_ID" --location "$TASKS_LOCATION" --format=json || found=$?
      ;;
    secret)
      lookup secrets get-iam-policy "$resource" --project "$PROJECT_ID" --format=json || found=$?
      ;;
    repo)
      lookup artifacts repositories get-iam-policy "$resource" --project "$PROJECT_ID" --location "$REGION" --format=json || found=$?
      ;;
    *)
      fail "unknown IAM scope: $scope"
      ;;
  esac
  [[ "$found" -eq 0 ]] && policy="$LOOKUP_OUT"
  if [[ -n "$policy" ]] && printf '%s' "$policy" | policy_has_binding "$role" "$member"; then
    note "ok: $role for $member on ${resource:-project $PROJECT_ID}"
    return
  fi
  case "$scope" in
    project)
      run_binding gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$member" --role="$role" --condition=None
      ;;
    bucket)
      run_binding gcloud storage buckets add-iam-policy-binding "$resource" \
        --project "$PROJECT_ID" --member="$member" --role="$role"
      ;;
    sa)
      run_binding gcloud iam service-accounts add-iam-policy-binding "$resource" \
        --project "$PROJECT_ID" --member="$member" --role="$role"
      ;;
    queue)
      run_binding gcloud tasks queues add-iam-policy-binding "$resource" \
        --project "$PROJECT_ID" --location "$TASKS_LOCATION" --member="$member" --role="$role"
      ;;
    secret)
      run_binding gcloud secrets add-iam-policy-binding "$resource" \
        --project "$PROJECT_ID" --member="$member" --role="$role"
      ;;
    repo)
      run_binding gcloud artifacts repositories add-iam-policy-binding "$resource" \
        --project "$PROJECT_ID" --location "$REGION" --member="$member" --role="$role"
      ;;
  esac
}

sa_exists() {
  exists iam service-accounts describe "$1" --project "$PROJECT_ID" --format='value(email)'
}

# Bucket names are global, so only a bucket this project lists is ours.
bucket_in_project() {
  lookup storage buckets list --project "$PROJECT_ID" --format='value(name)' \
    && printf '%s\n' "$LOOKUP_OUT" | grep -Fxq "${1#gs://}"
}

secret_exists() {
  exists secrets describe "$1" --project "$PROJECT_ID" --format='value(name)'
}

# stdin: `gcloud storage buckets describe --format=json`. Prints what differs
# from AGE-day Delete lifecycle + no soft delete: "lifecycle", "soft-delete".
bucket_drift() {
  python3 -c '
import json, sys
age = int(sys.argv[1])
try:
    data = json.load(sys.stdin)
except ValueError:
    data = {}
lifecycle = data.get("lifecycle_config") or data.get("lifecycle") or {}
rules = lifecycle.get("rule") or []
if not any(
    isinstance(rule, dict)
    and (rule.get("action") or {}).get("type") == "Delete"
    and (rule.get("condition") or {}).get("age") == age
    for rule in rules
):
    print("lifecycle")
soft = data.get("soft_delete_policy") or data.get("softDeletePolicy") or {}
if str(soft.get("retentionDurationSeconds", "0")) not in ("0", ""):
    print("soft-delete")
' "$1"
}

# ── Steps ────────────────────────────────────────────────────

step_apis() {
  local enabled="" missing="" api
  if lookup services list --enabled --project "$PROJECT_ID" --format='value(config.name)'; then
    enabled="$LOOKUP_OUT"
  fi
  for api in $REQUIRED_APIS; do
    if printf '%s\n' "$enabled" | grep -qx "$api"; then
      note "ok: $api is enabled"
    else
      missing="$missing $api"
    fi
  done
  if [[ -n "$missing" ]]; then
    # shellcheck disable=SC2086 # one word per API
    run gcloud services enable $missing --project "$PROJECT_ID"
  fi
  note "container scanning (containerscanning / ondemandscanning) stays off: it is billed per image"
}

ensure_service_account() {
  local name="$1" display="$2" email="$1@$PROJECT_ID.iam.gserviceaccount.com"
  if sa_exists "$email"; then
    note "ok: $email exists"
  else
    run gcloud iam service-accounts create "$name" --project "$PROJECT_ID" --display-name="$display"
  fi
}

step_runtime_sa() {
  local member="serviceAccount:$RUNTIME_SA" secret
  ensure_service_account "$RUNTIME_SA_NAME" "LineBot Cloud Run runtime"
  grant project "" "$member" roles/datastore.user
  if [[ -n "$GCS_BUCKET_NAME_VALUE" ]]; then
    grant bucket "gs://$GCS_BUCKET_NAME_VALUE" "$member" roles/storage.objectUser
  else
    note "skipped: GCS_BUCKET_NAME is empty, so there is no media bucket to grant"
  fi
  # V4 signed media URLs: the SA signs through IAM Credentials as itself.
  grant sa "$RUNTIME_SA" "$member" roles/iam.serviceAccountTokenCreator
  local queue
  for queue in "$TASKS_QUEUE" "$TASKS_COLAB_QUEUE"; do
    if exists tasks queues describe "$queue" --project "$PROJECT_ID" --location "$TASKS_LOCATION" --format='value(name)'; then
      grant queue "$queue" "$member" roles/cloudtasks.enqueuer
    elif [[ "$queue" == "$TASKS_COLAB_QUEUE" ]]; then
      note "skipped: queue $queue ($TASKS_LOCATION) does not exist yet; a deploy with COLAB_ENABLED=true creates it and grants the enqueuer role (Colab, optional)"
    else
      note "skipped: queue $queue ($TASKS_LOCATION) does not exist yet; the deploy creates it and grants the enqueuer role"
    fi
  done
  if secret_exists "$DOTENV_SECRET"; then
    grant secret "$DOTENV_SECRET" "$member" roles/secretmanager.secretAccessor
  else
    note "skipped: secret $DOTENV_SECRET does not exist yet; the secrets step grants it"
  fi
  if secret_exists "$COLAB_SECRET"; then
    grant secret "$COLAB_SECRET" "$member" roles/secretmanager.secretAccessor
  else
    note "skipped: secret $COLAB_SECRET does not exist yet; ./scripts/mint_colab_adc.sh --apply creates it (Colab, optional)"
  fi
}

step_staging_bucket() {
  local drift rc=0
  if bucket_in_project "$STAGING_BUCKET"; then
    note "ok: $STAGING_BUCKET exists"
    lookup storage buckets describe "$STAGING_BUCKET" --project "$PROJECT_ID" --format=json \
      || fail "$STAGING_BUCKET disappeared while it was being checked; run the step again"
    drift="$(printf '%s' "$LOOKUP_OUT" | bucket_drift 3)"
    if [[ "$drift" == *lifecycle* ]]; then
      run gcloud storage buckets update "$STAGING_BUCKET" --project "$PROJECT_ID" --lifecycle-file="$LIFECYCLE_3D"
    fi
    if [[ "$drift" == *soft-delete* ]]; then
      run gcloud storage buckets update "$STAGING_BUCKET" --project "$PROJECT_ID" --clear-soft-delete
    fi
    return
  fi
  # The name is predictable; if someone else's project holds it, builds must
  # never stage their source there (the deploy then keeps the default bucket).
  probe storage buckets describe "$STAGING_BUCKET" --project "$PROJECT_ID" --format='value(name)' || rc=$?
  if [[ "$rc" -ne 1 ]]; then
    [[ "$rc" -eq 2 ]] && cat "$LOOKUP_ERR" >&2
    fail "$STAGING_BUCKET exists outside project $PROJECT_ID (bucket names are global) or cannot be read, so it is not used. If it is yours, move it into $PROJECT_ID or delete it; otherwise report it and keep deploying without it (the deploy then stages in gs://${PROJECT_ID}_cloudbuild)"
  fi
  run gcloud storage buckets create "$STAGING_BUCKET" \
    --project "$PROJECT_ID" \
    --location=us-west1 \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --soft-delete-duration=0 \
    --lifecycle-file="$LIFECYCLE_3D"
}

step_builder_sa() {
  local member="serviceAccount:$BUILDER_SA"
  ensure_service_account "$BUILDER_SA_NAME" "LineBot Cloud Build"
  if exists artifacts repositories describe "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" --format='value(name)'; then
    note "ok: Artifact Registry repository $AR_REPOSITORY exists"
  else
    run gcloud artifacts repositories create "$AR_REPOSITORY" \
      --project "$PROJECT_ID" \
      --repository-format=docker \
      --location "$REGION" \
      --description "LineBot-CloudAgent container images"
  fi
  grant repo "$AR_REPOSITORY" "$member" roles/artifactregistry.writer
  grant project "" "$member" roles/logging.logWriter
  if [[ "$APPLY" != "true" ]] || bucket_in_project "$STAGING_BUCKET"; then
    grant bucket "$STAGING_BUCKET" "$member" roles/storage.objectViewer
  else
    fail "$STAGING_BUCKET is not a bucket of project $PROJECT_ID; run --step staging-bucket --apply first"
  fi
}

step_purge_cloudbuild() {
  local json drift listing="" count rc=0
  cat >&2 <<WARNING
[warn] ================================================================
[warn]  purge-cloudbuild PERMANENTLY deletes gs://${PROJECT_ID}_cloudbuild/source/**.
[warn]  Older deploys uploaded every .env secret into those tarballs.
[warn]  Soft delete is cleared first, so nothing can be restored afterwards.
[warn]  Rotate the leaked secrets as well (see the summary step).
[warn] ================================================================
WARNING
  if ! lookup storage buckets describe "$LEGACY_CB_BUCKET" --project "$PROJECT_ID" --format=json; then
    note "ok: $LEGACY_CB_BUCKET does not exist; nothing to purge"
    return
  fi
  json="$LOOKUP_OUT"
  if [[ "$APPLY" == "true" ]] && [[ "$YES_PURGE" != "true" ]]; then
    if [[ "$STEP" == "purge-cloudbuild" ]]; then
      fail "purge-cloudbuild deletes objects for good; rerun with --yes-purge"
    fi
    warn "skipped purge-cloudbuild: rerun with --step purge-cloudbuild --apply --yes-purge"
    SKIPPED_PURGE="true"
    return
  fi
  drift="$(printf '%s' "$json" | bucket_drift 1)"
  # Soft delete must go first: it only applies to objects deleted later.
  if [[ "$drift" == *soft-delete* ]]; then
    run gcloud storage buckets update "$LEGACY_CB_BUCKET" --project "$PROJECT_ID" --clear-soft-delete
  else
    note "ok: soft delete is already off on $LEGACY_CB_BUCKET"
  fi
  # Only "matched no objects" means empty; any other error stops the purge.
  listing="$(gcloud storage ls "$LEGACY_CB_BUCKET/source/**" --project "$PROJECT_ID" 2>"$LOOKUP_ERR")" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    if ! grep -qi 'matched no objects' "$LOOKUP_ERR"; then
      cat "$LOOKUP_ERR" >&2
      fail "Could not list $LEGACY_CB_BUCKET/source/; nothing was deleted from it"
    fi
    listing=""
  fi
  count="$(printf '%s\n' "$listing" | grep -c . || true)"
  if [[ "$count" -gt 0 ]]; then
    note "$count object(s) under $LEGACY_CB_BUCKET/source/"
    run gcloud storage rm "$LEGACY_CB_BUCKET/source/**" --project "$PROJECT_ID"
  else
    note "ok: no objects under $LEGACY_CB_BUCKET/source/"
  fi
  if [[ "$drift" == *lifecycle* ]]; then
    run gcloud storage buckets update "$LEGACY_CB_BUCKET" --project "$PROJECT_ID" --lifecycle-file="$LIFECYCLE_1D"
  else
    note "ok: $LEGACY_CB_BUCKET already deletes objects after 1 day"
  fi
}

step_secrets() {
  if secret_exists "$DOTENV_SECRET"; then
    note "ok: secret $DOTENV_SECRET exists; scripts/deploy_cloud_run.sh adds a version whenever .env changes"
  else
    [[ -f "$ENV_FILE" ]] || fail "Env file not found: $ENV_FILE"
    # Refuse (by key name) before anything is created.
    if ! python3 "$ROOT_DIR/scripts/envfile.py" to-dotenv --file "$ENV_FILE" --sha256 >/dev/null; then
      fail "Fix the keys and lines listed above in $ENV_FILE, then run this step again"
    fi
    # The blob goes through a pipe, never a file: no copy of the secrets
    # lands in the repository or in a temp directory.
    if [[ "$APPLY" == "true" ]]; then
      echo "  [run] python3 scripts/envfile.py to-dotenv --file $(printf '%q' "$ENV_FILE") | gcloud secrets create $DOTENV_SECRET --project $PROJECT_ID --replication-policy=automatic --data-file=-"
      python3 "$ROOT_DIR/scripts/envfile.py" to-dotenv --file "$ENV_FILE" \
        | gcloud secrets create "$DOTENV_SECRET" \
          --project "$PROJECT_ID" \
          --replication-policy=automatic \
          --data-file=-
    else
      echo "  [dry-run] python3 scripts/envfile.py to-dotenv --file $(printf '%q' "$ENV_FILE") | gcloud secrets create $DOTENV_SECRET --project $PROJECT_ID --replication-policy=automatic --data-file=-"
    fi
  fi
  if [[ "$APPLY" != "true" ]] || sa_exists "$RUNTIME_SA"; then
    grant secret "$DOTENV_SECRET" "serviceAccount:$RUNTIME_SA" roles/secretmanager.secretAccessor
  else
    warn "$RUNTIME_SA does not exist yet; run --step runtime-sa --apply to grant it $DOTENV_SECRET"
  fi
}

# The Colab credential (ADR-0010): only a person can mint it, so this step
# never creates the secret; it lets the runtime SA read it. The deploy
# mounts it only while COLAB_ENABLED=true, and refuses to when the runtime
# SA lacks this binding (Cloud Run would reject the revision).
step_colab_secret() {
  if ! secret_exists "$COLAB_SECRET"; then
    note "skipped: secret $COLAB_SECRET does not exist yet; ./scripts/mint_colab_adc.sh --apply creates it"
    note "(sign in with the dedicated Colab account) and runs this step; see DEPLOY.md, Colab 層"
    return
  fi
  if [[ "$APPLY" != "true" ]] || sa_exists "$RUNTIME_SA"; then
    grant secret "$COLAB_SECRET" "serviceAccount:$RUNTIME_SA" roles/secretmanager.secretAccessor
  else
    warn "$RUNTIME_SA does not exist yet; run --step runtime-sa --apply (it grants every existing secret, $COLAB_SECRET included)"
  fi
  note "the deploy mounts $COLAB_SECRET at /secrets/colab/adc.json while COLAB_ENABLED=true, pinned to its newest enabled version"
}

# Sets AR_STATE to "missing", "dry-run" or "enforced" for the repository's
# cleanup policy.
AR_STATE=""
read_ar_cleanup_state() {
  local json=""
  if lookup artifacts repositories describe "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" --format=json; then
    json="$LOOKUP_OUT"
  fi
  AR_STATE="$(printf '%s' "$json" | python3 -c '
import json, sys
names = sys.argv[1].split()
try:
    repo = json.load(sys.stdin)
except ValueError:
    repo = {}
policies = repo.get("cleanupPolicies") or {}
if not all(name in policies for name in names):
    print("missing")
elif repo.get("cleanupPolicyDryRun", False):
    print("dry-run")
else:
    print("enforced")
' "$AR_CLEANUP_POLICIES")"
}

step_ar_cleanup() {
  read_ar_cleanup_state
  case "$AR_STATE" in
    missing)
      run gcloud artifacts repositories set-cleanup-policies "$AR_REPOSITORY" \
        --project "$PROJECT_ID" \
        --location "$REGION" \
        --policy="$AR_CLEANUP_FILE" \
        --dry-run
      note "dry run: nothing is deleted. After about a day (the Artifact Registry Data Access"
      note "audit logs show what would go), run: --step ar-cleanup-enforce --apply"
      ;;
    dry-run)
      note "ok: cleanup policy is set in dry-run mode; enforce it with --step ar-cleanup-enforce --apply"
      ;;
    enforced)
      note "ok: cleanup policy is enforced (keep the 2 newest images, delete older ones after 1 day)"
      ;;
  esac
  note_ar_rollback_cap
}

# An enforced policy deletes, after a day, every image but the 2 newest,
# including images of kept revisions; Cloud Run keeps its own copy only
# while a revision serves traffic.
note_ar_rollback_cap() {
  local keep_revisions keep_images
  note "once enforced, only the 2 newest images stay: rollback reaches back 2 images, and an"
  note "image pushed by a failed deploy or a build-only submission takes one of those slots"
  keep_revisions="$(read_env_key DEPLOY_KEEP_REVISIONS)"
  keep_images="$(read_env_key DEPLOY_KEEP_IMAGES)"
  if [[ "${keep_revisions:-2}" =~ ^[0-9]+$ && "${keep_revisions:-2}" -gt 2 ]] \
    || [[ "${keep_images:-2}" =~ ^[0-9]+$ && "${keep_images:-2}" -gt 2 ]]; then
    warn "DEPLOY_KEEP_REVISIONS / DEPLOY_KEEP_IMAGES is above 2: the enforced cleanup policy still deletes the older revisions' images after a day"
  fi
}

step_ar_cleanup_enforce() {
  read_ar_cleanup_state
  case "$AR_STATE" in
    missing)
      fail "no cleanup policy on $AR_REPOSITORY yet; run --step ar-cleanup --apply first and review its dry run"
      ;;
    enforced)
      note "ok: cleanup policy is already enforced"
      ;;
    dry-run)
      run gcloud artifacts repositories set-cleanup-policies "$AR_REPOSITORY" \
        --project "$PROJECT_ID" \
        --location "$REGION" \
        --policy="$AR_CLEANUP_FILE" \
        --no-dry-run
      ;;
  esac
  note_ar_rollback_cap
}

# ── Phase 9: the Compute Engine default service account ─────

# stdin: a JSON list of Cloud Run services, Cloud Scheduler jobs or VMs
# (KIND run, scheduler or compute). Prints each one that acts as ACCOUNT,
# except the Cloud Run service SKIP (NAME/REGION).
users_of_account() {
  python3 -c '
import json, sys
kind, account, skip = sys.argv[1:4]
raw = sys.stdin.read().strip()
for item in json.loads(raw) if raw else []:
    item = item or {}
    if kind == "run":
        meta = item.get("metadata") or {}
        name = meta.get("name") or "?"
        region = (meta.get("labels") or {}).get("cloud.googleapis.com/location") or "?"
        if name + "/" + region == skip:
            continue
        label = "Cloud Run service " + name + " (" + region + ")"
        spec = ((item.get("spec") or {}).get("template") or {}).get("spec") or {}
        # A service that names no account runs as the default one.
        used = [spec.get("serviceAccountName") or account]
    elif kind == "scheduler":
        label = "Cloud Scheduler job " + str(item.get("name") or "?").rsplit("/", 1)[-1]
        target = item.get("httpTarget") or {}
        used = [(target.get(token) or {}).get("serviceAccountEmail") for token in ("oidcToken", "oauthToken")]
    else:
        zone = str(item.get("zone") or "?").rsplit("/", 1)[-1]
        label = "VM " + str(item.get("name") or "?") + " (" + zone + ", " + str(item.get("status") or "?") + ")"
        used = [(sa or {}).get("email") for sa in item.get("serviceAccounts") or []]
    if account in used:
        print(label)
' "$1" "$2" "${3:-}"
}

# Every project gets PROJECT_NUMBER-compute@ with roles/editor on the whole
# project. Once linebot-runtime runs the service, builds run as
# linebot-builder, and no other Cloud Run service, Cloud Scheduler job or
# VM acts as that account, the role goes. The account stays: whatever was
# created with the defaults still expects it, and a deleted one is gone for
# good after 30 days.
step_drop_editor() {
  local number compute_sa member service_sa users user problems=""
  lookup projects describe "$PROJECT_ID" --format='value(projectNumber)' \
    || fail "Project $PROJECT_ID was not found"
  number="$LOOKUP_OUT"
  [[ "$number" =~ ^[0-9]+$ ]] || fail "Could not read the project number of $PROJECT_ID"
  compute_sa="$number-compute@developer.gserviceaccount.com"
  member="serviceAccount:$compute_sa"
  lookup projects get-iam-policy "$PROJECT_ID" --format=json \
    || fail "Could not read the IAM policy of project $PROJECT_ID"
  if ! printf '%s' "$LOOKUP_OUT" | policy_has_binding roles/editor "$member"; then
    note "ok: $compute_sa does not hold roles/editor; nothing to do"
    return
  fi

  # Its replacements, as the deploy uses them.
  sa_exists "$RUNTIME_SA" \
    || problems="$problems"$'\n'"       - $RUNTIME_SA does not exist: run --step runtime-sa --apply"
  sa_exists "$BUILDER_SA" \
    || problems="$problems"$'\n'"       - $BUILDER_SA does not exist: run --step builder-sa --apply"
  bucket_in_project "$STAGING_BUCKET" \
    || problems="$problems"$'\n'"       - $STAGING_BUCKET is not a bucket of $PROJECT_ID, so the deploy's Cloud Build runs as the default service account: run --step staging-bucket --apply"
  if lookup run services describe "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" \
    --format='value(spec.template.spec.serviceAccountName)'; then
    service_sa="$LOOKUP_OUT"
    [[ "$service_sa" == "$RUNTIME_SA" ]] \
      || problems="$problems"$'\n'"       - Cloud Run service $SERVICE_NAME runs as ${service_sa:-the default service account}, not $RUNTIME_SA: deploy with ./scripts/deploy_cloud_run.sh"
  else
    problems="$problems"$'\n'"       - Cloud Run service $SERVICE_NAME does not exist in $REGION: deploy with ./scripts/deploy_cloud_run.sh first"
  fi

  # Anything else that acts as the account (Cloud Run: every region).
  users=""
  if lookup run services list --project "$PROJECT_ID" \
    --format='json(metadata.name,metadata.labels,spec.template.spec.serviceAccountName)'; then
    users="$users$(printf '%s' "$LOOKUP_OUT" | users_of_account run "$compute_sa" "$SERVICE_NAME/$REGION")"$'\n'
  fi
  # A job's headers can carry a secret (the cron token): no gcloud file log.
  if CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true lookup scheduler jobs list --project "$PROJECT_ID" --location "$REGION" \
    --format='json(name,httpTarget.oidcToken.serviceAccountEmail,httpTarget.oauthToken.serviceAccountEmail)'; then
    users="$users$(printf '%s' "$LOOKUP_OUT" | users_of_account scheduler "$compute_sa")"$'\n'
  fi
  # A project with the Compute Engine API off (it is not in REQUIRED_APIS)
  # has no VM.
  if OFF_API_MEANS_NONE=compute.googleapis.com lookup compute instances list --project "$PROJECT_ID" \
    --format='json(name,zone,status,serviceAccounts[].email)'; then
    users="$users$(printf '%s' "$LOOKUP_OUT" | users_of_account compute "$compute_sa")"
  fi
  while IFS= read -r user; do
    [[ -z "$user" ]] && continue
    problems="$problems"$'\n'"       - $user runs as $compute_sa: give it its own service account, or delete it"
  done <<< "$users"

  if [[ -n "$problems" ]]; then
    fail "Not removing roles/editor from $compute_sa:$problems"
  fi
  note "ok: $SERVICE_NAME runs as $RUNTIME_SA; the deploy's Cloud Build runs as $BUILDER_SA"
  note "ok: no other Cloud Run service (any region), Cloud Scheduler job ($REGION) or VM runs as $compute_sa"
  note "Check it in the Policy Simulator first: https://console.cloud.google.com/iam-admin/iam?project=$PROJECT_ID"
  note "(edit $compute_sa, remove Editor, click Test changes instead of Save)"
  note "Roll back: gcloud projects add-iam-policy-binding $PROJECT_ID --member=$member --role=roles/editor --condition=None"
  run gcloud projects remove-iam-policy-binding "$PROJECT_ID" --member="$member" --role=roles/editor --condition=None
  note "The account itself stays: never delete it."
}

step_budget() {
  cat <<BUDGET
       No command: budgets are created in the console (free; only Pub/Sub
       notifications would cost money). Billing currency is TWD.
       1. https://console.cloud.google.com/billing/budgets?project=$PROJECT_ID -> Create budget
       2. Name: linebot-monthly. Scope: project $PROJECT_ID, all services.
          Keep "include credits" on: Always Free usage is applied as credits.
       3. Amount: Specified amount, 30 TWD (about US\$1), monthly.
       4. Thresholds: 50%, 90%, 100% of actual spend, and 100% of forecasted spend.
       5. Notifications: email to billing admins and users. No Pub/Sub, no spend cap
          (a Cloud Run spend cap pauses the bot and counts gross cost before credits).
BUDGET
}

step_summary() {
  cat <<SUMMARY
       Still needs a person:
       - Rotate every secret the old Cloud Build tarballs held: LINE channel secret and
         access token, NVIDIA / OpenRouter / Tavily keys, INTERNAL_CRON_SECRET and
         INTERNAL_TASKS_SECRET. Update .env, then ./scripts/deploy_cloud_run.sh
         (it adds a $DOTENV_SECRET version and destroys the old ones).
       - Budget alert: see the budget step (console only).
       - LINE Developers Console: turn on Webhook redelivery.
       - purge-cloudbuild, if it has not run: --step purge-cloudbuild --apply --yes-purge
       - ar-cleanup-enforce, a day after ar-cleanup: --step ar-cleanup-enforce --apply
       - Colab (optional, ADR-0010): ./scripts/mint_colab_adc.sh --apply signed in as a
         dedicated Colab account, then the first-time steps in DEPLOY.md (Colab 層).
       - Phase 9: remove roles/editor from the compute default service account after a
         Policy Simulator check: --step drop-editor --apply (it never deletes that account).
SUMMARY
  if [[ "$SKIPPED_PURGE" == "true" ]]; then
    warn "purge-cloudbuild was skipped in this run (no --yes-purge)"
  fi
}

# ── Main ─────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step|--project-id|--region|--env-file)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      ;;
  esac
  case "$1" in
    --apply)
      APPLY="true"
      shift
      ;;
    --yes-purge)
      YES_PURGE="true"
      shift
      ;;
    --list)
      LIST="true"
      shift
      ;;
    --step)
      # Empty would mean every step.
      [[ -n "$2" ]] || fail "--step needs a step name (see --list)"
      STEP="$2"
      shift 2
      ;;
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

if [[ "$LIST" == "true" ]]; then
  list_steps
  exit 0
fi

is_known_step() {
  local known
  for known in $KNOWN_STEPS; do
    [[ "$1" == "$known" ]] && return 0
  done
  return 1
}
if [[ -n "$STEP" ]] && ! is_known_step "$STEP"; then
  fail "Unknown step: '$STEP' (see --list)"
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}
require_cmd gcloud
require_cmd python3
load_env_file

# Never gcloud's default project: --apply changes IAM and deletes objects.
ENV_PROJECT_ID="$(read_env_key GCP_PROJECT_ID)"
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$ENV_PROJECT_ID"
  [[ -n "$PROJECT_ID" ]] || fail "GCP_PROJECT_ID is empty in $ENV_FILE: set it, or pass --project-id"
elif [[ -n "$ENV_PROJECT_ID" ]] && [[ "$ENV_PROJECT_ID" != "$PROJECT_ID" ]]; then
  fail "--project-id $PROJECT_ID does not match GCP_PROJECT_ID=$ENV_PROJECT_ID in $ENV_FILE, whose settings the steps use"
fi
if [[ ! "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
  fail "Not a GCP project id: $PROJECT_ID"
fi
[[ -z "$REGION" ]] && REGION="$(read_env_key CLOUD_RUN_REGION)"
[[ -z "$REGION" ]] && REGION="us-west1"

# Scratch space outside the repository for gcloud error messages.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/linebot-provision.XXXXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
LOOKUP_ERR="$WORK_DIR/gcloud.err"

GCS_BUCKET_NAME_VALUE="$(read_env_key GCS_BUCKET_NAME)"
TASKS_QUEUE="$(read_env_key CLOUD_TASKS_QUEUE)"
[[ -z "$TASKS_QUEUE" ]] && TASKS_QUEUE="linebot-events"
# The Colab controller's queue; the deploy creates it while COLAB_ENABLED=true.
TASKS_COLAB_QUEUE="$(read_env_key CLOUD_TASKS_COLAB_QUEUE)"
[[ -z "$TASKS_COLAB_QUEUE" ]] && TASKS_COLAB_QUEUE="linebot-colab"
# The deploy's Cloud Run service; drop-editor checks which account it runs as.
SERVICE_NAME="$(read_env_key CLOUD_RUN_SERVICE_NAME)"
[[ -z "$SERVICE_NAME" ]] && SERVICE_NAME="linebot-cloud-agent"
# Same rule as the app and the deploy: a missing key means us-west1, an
# empty value means the Cloud Run region.
if env_has_key CLOUD_TASKS_LOCATION; then
  TASKS_LOCATION="$(read_env_key CLOUD_TASKS_LOCATION)"
  [[ -z "$TASKS_LOCATION" ]] && TASKS_LOCATION="$REGION"
else
  TASKS_LOCATION="us-west1"
fi

RUNTIME_SA="$RUNTIME_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
BUILDER_SA="$BUILDER_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
STAGING_BUCKET="gs://$PROJECT_ID-cb-src-usw1"
LEGACY_CB_BUCKET="gs://${PROJECT_ID}_cloudbuild"
SKIPPED_PURGE="false"

if [[ "$APPLY" == "true" ]]; then
  echo "[mode] APPLY: commands below change project $PROJECT_ID"
else
  echo "[mode] dry run: nothing changes in project $PROJECT_ID (add --apply to run the [dry-run] lines)"
fi

steps="$ALL_STEPS"
[[ -n "$STEP" ]] && steps="$STEP"
for step in $steps; do
  echo "[step] $step"
  case "$step" in
    apis) step_apis ;;
    runtime-sa) step_runtime_sa ;;
    staging-bucket) step_staging_bucket ;;
    builder-sa) step_builder_sa ;;
    purge-cloudbuild) step_purge_cloudbuild ;;
    secrets) step_secrets ;;
    colab-secret) step_colab_secret ;;
    ar-cleanup) step_ar_cleanup ;;
    ar-cleanup-enforce) step_ar_cleanup_enforce ;;
    drop-editor) step_drop_editor ;;
    budget) step_budget ;;
    summary) step_summary ;;
  esac
done

echo "[done] $([[ "$APPLY" == "true" ]] && echo "applied" || echo "dry run finished; nothing was changed")"
