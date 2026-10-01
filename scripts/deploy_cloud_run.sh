#!/usr/bin/env bash
# Stays compatible with macOS /bin/bash 3.2: no mapfile, no ${var,,}, and no
# "${array[@]}" expansion of an array that can be empty under `set -u`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ID=""
SERVICE_NAME=""
REGION=""
SERVICE_ACCOUNT=""
ENV_FILE="$ROOT_DIR/.env"
IMAGE_TAG=""
SKIP_TESTS="false"
SKIP_SMOKE="false"
KEEP_REVISION_COUNT=""
KEEP_IMAGE_COUNT=""
ENABLE_APIS=""
ALLOW_ENV_FALLBACK="false"
DOTENV_VERSION=""

# cloudbuild.yaml pushes ${_REGION}-docker.pkg.dev/$PROJECT_ID/${_AR_REPOSITORY}/app;
# this script passes _AR_REPOSITORY so both sides use the same repository.
AR_REPOSITORY="linebot-cloud-agent"
IMAGE_NAME="app"

# Created by scripts/provision_gcp.sh (ADR-0008); the deploy uses each one
# once it exists and falls back to the pre-provisioning path until then.
RUNTIME_SA_NAME="linebot-runtime"
BUILDER_SA_NAME="linebot-builder"
DOTENV_SECRET="linebot-dotenv"
# src/config.py SECRET_ENV_FILE: Settings reads it after .env.
SECRET_MOUNT_PATH="/secrets/app/app.env"
# The Colab tier's credential (ADR-0010), minted by scripts/mint_colab_adc.sh
# and mounted only while COLAB_ENABLED=true, at src/config.py's
# COLAB_ADC_PATH default. Its own directory: Cloud Run mounts each secret
# as a volume.
COLAB_SECRET="linebot-colab-adc"
COLAB_MOUNT_PATH="/secrets/colab/adc.json"
COLAB_ADC_VERSION=""
# Always Free Cloud Storage (5 GB) covers Standard buckets in these regions.
FREE_BUCKET_LOCATIONS="US-WEST1 US-CENTRAL1 US-EAST1"

# Always scale to zero: Cloud Scheduler wakes the service for /internal/cron,
# so nothing needs a warm instance. The service-level --max caps scaling
# across revisions and removes any maxScale drift left on the service.
MIN_INSTANCES="0"
MAX_INSTANCES="1"

usage() {
  cat <<'USAGE'
Usage: ./scripts/deploy_cloud_run.sh [options]

One-command cloud deployment for LineBot-CloudAgent.
Default deploy settings are loaded from .env, so the normal workflow is:
  ./scripts/deploy_cloud_run.sh

The script will:
  1. read deploy settings from .env and verify required runtime env keys,
  2. refuse to deploy while .env still selects retired v2 models or legacy
     model keys (fix: python3 scripts/envfile.py migrate-v3 --file .env),
  3. refuse a media bucket (GCS_BUCKET_NAME) outside the Always Free
     locations (US-WEST1, US-CENTRAL1, US-EAST1) or not STANDARD,
  4. refuse to submit if a secret file (.env*, .cloudrun-env*, .claude/,
     credentials.json, *.bak, or the --env-file itself) would be uploaded
     to Cloud Build, or if an --env-file inside the repository is not
     ignored by git,
  5. ensure required GCP APIs, the Artifact Registry repository and the
     Cloud Tasks queues (unless TASKS_MODE=inline) exist: CLOUD_TASKS_QUEUE,
     and CLOUD_TASKS_COLAB_QUEUE while COLAB_ENABLED=true,
  6. submit Cloud Build to run ruff + pytest and build/push the image
     (no secrets are uploaded and Cloud Build does not deploy) as
     linebot-builder, staging the source in gs://<project>-cb-src-usw1;
     while either is missing (scripts/provision_gcp.sh creates them) the
     deploy stops before the build, as the default bucket
     gs://<project>_cloudbuild is outside the free tier,
  7. deploy the pushed image to Cloud Run from this machine, with exactly
     the volumes it mounts. Provisioned: as linebot-runtime, with .env as
     the Secret Manager secret linebot-dotenv (a new version only when
     .env changed; the one checked before the build) mounted at
     /secrets/app/app.env, and APP_VERSION as the only plain variable.
     Not provisioned yet: with an env-vars file created in $TMPDIR before
     the build and deleted right after the deploy. A service that already
     reads linebot-dotenv is never moved back to plain variables unless
     --allow-env-fallback is given, and a lookup that fails for any
     reason but NOT_FOUND stops the deploy.
     With COLAB_ENABLED=true, the newest enabled version of
     linebot-colab-adc is pinned and mounted at /secrets/colab/adc.json
     (never read by the deploy); a missing secret leaves the Colab tier
     off with a warning. A runtime account with no secretAccessor binding
     on a mounted secret gets a warning (a project or group grant also
     works; Cloud Run rejects a revision that cannot read it),
  8. apply the GCS lifecycle, signing role (a runtime account other than
     linebot-runtime), Cloud Tasks enqueuer, public access and Cloud
     Scheduler settings, then run smoke checks,
  9. after a passing smoke check, delete the revisions behind the newest
     --keep-revisions Ready ones (a revision that never became Ready takes
     no slot), images beyond --keep-images that no revision in the region
     uses, and linebot-dotenv (and, when mounted, linebot-colab-adc)
     versions no revision in any region still mounts.

Env keys are read case-insensitively, as the app reads them.

Deploy settings supported in .env:
  GCP_PROJECT_ID
  CLOUD_RUN_SERVICE_NAME
  CLOUD_RUN_REGION
  CLOUD_RUN_SERVICE_ACCOUNT
  DEPLOY_KEEP_REVISIONS
  DEPLOY_KEEP_IMAGES
  DEPLOY_ENABLE_APIS

Options:
  --project-id ID            Override GCP project ID
  --service-name NAME        Override Cloud Run service name
  --region REGION            Override Cloud Run region
  --service-account EMAIL    Runtime service account (default: linebot-runtime once
                             provisioned, else keep the service's current one)
  --env-file PATH            Env file to deploy (default: .env)
  --image-tag TAG            Override Docker image tag (default: <git-sha>-<utc-timestamp>)
  --keep-revisions N         How many revisions to keep (default from .env or 2)
  --keep-images N            How many image digests to keep (default from .env or 2)
  --skip-tests               Skip the Cloud Build lint, unit tests and compile checks
  --skip-smoke               Skip the post-deploy cloud smoke checks (old revisions,
                             images and secret versions are then kept)
  --skip-enable-apis         Do not auto-enable required GCP APIs
  --allow-env-fallback       Deploy with plain environment variables even though the
                             service reads linebot-dotenv (the runtime account or the
                             secret is gone); every .env value becomes readable to
                             anyone who can view the service
  -h, --help                 Show this help message
USAGE
}

fail() {
  echo "[fail] $*" >&2
  exit 1
}

warn() {
  echo "[warn] $*" >&2
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

wait_for_public_health() {
  local url="$1"
  local status=""
  for _ in $(seq 1 24); do
    status="$(curl -sS -o /dev/null -w '%{http_code}' "$url/health" || true)"
    if [[ "$status" == "200" ]]; then
      return 0
    fi
    sleep 5
  done
  return 1
}

validate_positive_int() {
  local value="$1"
  local name="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
    fail "$name must be an integer greater than or equal to 1"
  fi
}

normalize_bool() {
  local value
  local name="$2"
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    true|1|yes|y|on)
      echo true
      ;;
    false|0|no|n|off)
      echo false
      ;;
    *)
      fail "$name must be one of: true, false, 1, 0, yes, no"
      ;;
  esac
}

# $ENV_FILE as `envfile.py dump` reads it, loaded once by load_env_file: a
# python3 start per key read was most of the deploy's local time. Never
# exported. Keys are in upper case, so KEY and key read alike, as the app
# reads them (pydantic-settings, case-insensitive). ENV_PLACEHOLDER_KEYS
# lists the keys whose value is an example placeholder, space-separated.
ENV_KEYS=()
ENV_VALUES=()
ENV_PLACEHOLDER_KEYS=" "
load_env_file() {
  local key value placeholder
  while IFS= read -r -d '' key && IFS= read -r -d '' value && IFS= read -r -d '' placeholder; do
    ENV_KEYS+=("$key")
    ENV_VALUES+=("$value")
    if [[ "$placeholder" == "1" ]]; then
      ENV_PLACEHOLDER_KEYS="$ENV_PLACEHOLDER_KEYS$key "
    fi
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

env_key_is_placeholder() {
  [[ "$ENV_PLACEHOLDER_KEYS" == *" $1 "* ]]
}

is_placeholder_value() {
  python3 "$ROOT_DIR/scripts/envfile.py" is-placeholder "$1" >/dev/null 2>&1
}

require_env_key() {
  local key="$1"
  local value
  value="$(read_env_key "$key")"
  if [[ -z "$value" ]]; then
    fail "Required env key missing or empty in $ENV_FILE: $key"
  fi
  if env_key_is_placeholder "$key"; then
    fail "Required env key still uses an example placeholder in $ENV_FILE: $key"
  fi
}

# A placeholder value is never empty, so an empty key passes.
reject_placeholder_env_key() {
  if env_key_is_placeholder "$1"; then
    fail "Env key still uses an example placeholder in $ENV_FILE: $1. Set a real value or leave it empty."
  fi
}

gcloud_beta_installed() {
  # --only-local-state never contacts the network and never installs anything.
  [[ "$(gcloud components list --only-local-state --filter='id=beta' \
    --format='value(id)' 2>/dev/null || true)" == "beta" ]]
}

# gcloud_lookup ARGS...: a read-only gcloud describe or list. Returns 0 with
# the output in LOOKUP_OUT when gcloud succeeds, and 1 when gcloud says the
# resource was not found. Any other failure (permission denied, API
# disabled, expired login, network) stops the deploy: it must never read as
# "not provisioned", or the deploy would quietly fall back to plain-text
# settings. Never call it inside $(...), where the stop would only end the
# subshell. File logging is off: gcloud copies everything it prints into
# ~/.config/gcloud/logs (kept about 30 days), and some lookups print secret
# fields (a Cloud Scheduler job's headers). Prompts are off: stderr goes to
# a file, where gcloud's "API [x] not enabled ... enable and retry? (y/N)"
# would wait unseen.
LOOKUP_OUT=""
gcloud_lookup() {
  local rc=0
  LOOKUP_OUT="$(CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
    gcloud "$@" 2>"$LOOKUP_ERR")" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi
  # NOT_FOUND / "was not found" / "gs://x not found: 404" / Cloud Run's
  # "Cannot find service [x]".
  if grep -Eqi 'not[ _]found|cannot find|HTTPError 404' "$LOOKUP_ERR"; then
    return 1
  fi
  cat "$LOOKUP_ERR" >&2
  fail "gcloud $1 $2 $3 failed, and not with NOT_FOUND; fix the access (gcloud auth login, the account's roles, the API), then deploy again"
}

sa_exists() {
  gcloud_lookup iam service-accounts describe "$1" \
    --project "$PROJECT_ID" \
    --format='value(email)' \
    && [[ -n "$LOOKUP_OUT" ]]
}

# Bucket names are global, so only a bucket this project lists is ours: a
# same-named bucket in another project must never receive the source.
bucket_in_project() {
  gcloud_lookup storage buckets list \
    --project "$PROJECT_ID" \
    --format='value(name)' \
    && printf '%s\n' "$LOOKUP_OUT" | grep -Fxq "${1#gs://}"
}

secret_exists() {
  gcloud_lookup secrets describe "$1" \
    --project "$PROJECT_ID" \
    --format='value(name)' \
    && [[ -n "$LOOKUP_OUT" ]]
}

# Succeeds when the service's current template mounts the dotenv secret.
# Unreadable output counts as mounted, so the fallback stays refused.
service_mounts_dotenv() {
  gcloud_lookup run services describe "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --format='json(spec.template.spec.volumes)' || return 1
  printf '%s' "$LOOKUP_OUT" | python3 -c '
import json, sys
secret = sys.argv[1]
raw = sys.stdin.read().strip()
try:
    data = json.loads(raw) if raw else {}
except ValueError:
    sys.exit(0)
spec = ((data.get("spec") or {}).get("template") or {}).get("spec") or {}
for volume in spec.get("volumes") or []:
    name = ((volume or {}).get("secret") or {}).get("secretName") or ""
    if name == secret or name.endswith("/secrets/" + secret):
        sys.exit(0)
sys.exit(1)
' "$DOTENV_SECRET"
}

# stdin: `gcloud run services list --format='json(metadata.labels)'`, which
# lists every region. Prints argv[1] and every region with a service.
SERVICE_REGIONS_PY='
import json, sys
regions = {sys.argv[1]}
raw = sys.stdin.read().strip()
for service in json.loads(raw) if raw else []:
    labels = ((service or {}).get("metadata") or {}).get("labels") or {}
    if labels.get("cloud.googleapis.com/location"):
        regions.add(labels["cloud.googleapis.com/location"])
print("\n".join(sorted(regions)))
'

# stdin: `gcloud run revisions list --format='json(metadata.name,spec.volumes,
# spec.containers[].volumeMounts)'` of every service in a region:
# linebot-dotenv and linebot-colab-adc are one per project, so another
# service (a staging copy) may pin them too. Prints the versions of secret
# argv[1] the revisions mount, one per line. A volume no container mounts
# pins nothing: gcloud --set-secrets alone leaves the previous deploy's
# volumes in the template, unmounted (seen 2026-10-01), and the revisions
# made from it carry them. A revision that lists no mount at all keeps
# every volume (a listing that left the mounts out must not free a version).
PINNED_SECRET_VERSIONS_PY='
import json, sys
secret = sys.argv[1]
raw = sys.stdin.read().strip()
for revision in json.loads(raw) if raw else []:
    spec = (revision or {}).get("spec") or {}
    mounted = {
        (mount or {}).get("name")
        for container in spec.get("containers") or []
        for mount in (container or {}).get("volumeMounts") or []
    }
    for volume in spec.get("volumes") or []:
        if mounted and (volume or {}).get("name") not in mounted:
            continue
        source = (volume or {}).get("secret") or {}
        name = source.get("secretName") or ""
        if name != secret and not name.endswith("/secrets/" + secret):
            continue
        for item in source.get("items") or []:
            pinned = str((item or {}).get("key") or (item or {}).get("version") or "")
            if pinned:
                print(pinned)
'

# pinned_secret_versions SECRET: prints the versions of SECRET that the
# revisions of every Cloud Run service pin, in every region with one (the
# secret is one per project; revisions are listed per region).
pinned_secret_versions() {
  local regions region
  regions="$(gcloud run services list \
    --project "$PROJECT_ID" \
    --format='json(metadata.labels)' \
    | python3 -c "$SERVICE_REGIONS_PY" "$REGION")" || return 1
  for region in $regions; do
    gcloud run revisions list \
      --project "$PROJECT_ID" \
      --region "$region" \
      --format='json(metadata.name,spec.volumes,spec.containers[].volumeMounts)' \
      | python3 -c "$PINNED_SECRET_VERSIONS_PY" "$1" || return 1
  done
}

# cleanup_old_secret_versions SECRET DEPLOYED: destroys the versions of
# SECRET no kept revision can use. Kept: DEPLOYED (the version just
# deployed), every version a remaining revision of any service in any
# region pins (read from the revisions, so a version added by a deploy that
# never made a revision cannot push a rollback target out) and, as a floor,
# the newest max(2, keep-revisions). Disabled versions are billed like enabled ones;
# only destroyed ones are free.
cleanup_old_secret_versions() {
  local secret="$1" deployed="$2"
  local keep="$KEEP_REVISION_COUNT" versions pinned old_versions version
  local failures=0 live=0
  [[ "$keep" -lt 2 ]] && keep=2
  if ! pinned="$(pinned_secret_versions "$secret")"; then
    warn "Could not read which $secret versions the revisions use; skipping its version cleanup"
    return 0
  fi
  if ! versions="$(gcloud secrets versions list "$secret" \
    --project "$PROJECT_ID" \
    --filter='state!=DESTROYED' \
    --sort-by='~createTime' \
    --format='value(name)')"; then
    warn "Could not list $secret versions; skipping its version cleanup"
    return 0
  fi
  live="$(printf '%s\n' "$versions" | grep -c . || true)"
  old_versions="$(printf '%s\n' "$versions" | awk -v keep="$keep" 'NF && ++n > keep')"
  if [[ -z "$old_versions" ]]; then
    echo "       No old $secret versions to destroy"
  fi
  while IFS= read -r version; do
    [[ -z "$version" ]] && continue
    version="${version##*/}"
    [[ "$version" == "$deployed" ]] && continue
    if printf '%s\n' "$pinned" | grep -Fxq "$version"; then
      echo "       Keeping $secret version $version (a kept revision uses it)"
      continue
    fi
    if gcloud secrets versions destroy "$version" \
      --secret "$secret" \
      --project "$PROJECT_ID" \
      --quiet >/dev/null 2>&1; then
      echo "       Destroyed $secret version $version"
      live=$((live - 1))
    else
      warn "Failed to destroy $secret version $version"
      failures=$((failures + 1))
    fi
  done <<< "$old_versions"
  if [[ "$failures" -gt 0 ]]; then
    warn "Deploy completed, but $failures $secret version(s) need manual cleanup (disabled versions are still billed)"
  fi
  warn_many_secret_versions "$secret" "$live"
}

# warn_many_secret_versions SECRET COUNT. Secret Manager's free tier is 6
# active versions per billing account, shared by every secret; a deploy
# normally keeps 2 of linebot-dotenv and at most 2 of linebot-colab-adc.
warn_many_secret_versions() {
  if [[ "$2" -gt 3 ]]; then
    warn "$1 has $2 active versions; beyond 6 across all secrets Secret Manager bills each one. A deploy with passing smoke checks destroys the ones no kept revision uses"
  fi
}

# keep_secret_versions SECRET: the --skip-smoke path; nothing is destroyed.
keep_secret_versions() {
  echo "[step] Keeping every $1 version (smoke checks were skipped)"
  warn_many_secret_versions "$1" "$(gcloud secrets versions list "$1" \
    --project "$PROJECT_ID" \
    --filter='state!=DESTROYED' \
    --format='value(name)' 2>/dev/null | grep -c . || true)"
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

# warn_unless_secret_reader SECRET STEP: Cloud Run rejects a revision whose
# account cannot read a secret it mounts. The secret's own policy shows only
# direct grants, not a project-level or group one, so this only warns (an
# unreadable policy too); STEP is the provision_gcp.sh step that grants it.
warn_unless_secret_reader() {
  local policy
  policy="$(CLOUDSDK_CORE_DISABLE_PROMPTS=1 gcloud secrets get-iam-policy "$1" \
    --project "$PROJECT_ID" \
    --format=json 2>/dev/null || true)"
  if ! printf '%s' "$policy" | policy_has_binding roles/secretmanager.secretAccessor "serviceAccount:$DEPLOY_RUNTIME_SA"; then
    warn "Could not confirm that $DEPLOY_RUNTIME_SA can read secret $1: its policy grants it no secretAccessor role (a project-level or group grant does not show there). If it cannot read it, Cloud Run rejects the revision after the build; ./scripts/provision_gcp.sh --step $2 --apply grants it to ${RUNTIME_SA_NAME}"
  fi
}

# stdin: `gcloud run revisions list --format='json(metadata.name,status.conditions)'`,
# newest first. Prints the revisions that argv[1] Ready revisions are newer
# than. A revision that never became Ready (a failed deploy) takes no slot,
# so it cannot push the last good one out; it goes with the ones behind it.
# The newest Ready revision, which serves all traffic after --to-latest,
# is always kept (Cloud Run refuses to delete a revision serving traffic).
OLD_REVISIONS_PY='
import json, sys
keep = int(sys.argv[1])
raw = sys.stdin.read().strip()
ready = 0
for revision in json.loads(raw) if raw else []:
    if ready >= keep:
        print(revision["metadata"]["name"])
    conditions = ((revision or {}).get("status") or {}).get("conditions") or []
    if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions):
        ready += 1
'

cleanup_old_revisions() {
  local old_revisions revision
  local failures=0
  if ! old_revisions="$(gcloud run revisions list \
    --service "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --sort-by='~metadata.creationTimestamp' \
    --format='json(metadata.name,status.conditions)' \
    | python3 -c "$OLD_REVISIONS_PY" "$KEEP_REVISION_COUNT")"; then
    warn "Could not list revisions; skipping revision cleanup"
    return 0
  fi

  if [[ -z "$old_revisions" ]]; then
    echo "       No old revisions to delete"
    return 0
  fi

  while IFS= read -r revision; do
    [[ -z "$revision" ]] && continue
    if gcloud run revisions delete "$revision" \
      --project "$PROJECT_ID" \
      --region "$REGION" \
      --quiet >/dev/null 2>&1; then
      echo "       Deleted revision: $revision"
    else
      warn "Failed to delete revision: $revision"
      failures=$((failures + 1))
    fi
  done <<< "$old_revisions"

  if [[ "$failures" -gt 0 ]]; then
    warn "Deploy completed, but $failures revision(s) need manual cleanup"
  fi
}

cleanup_old_images() {
  local images old_images digest revision_digests deleted=""
  local failures=0 position=0 at_risk=0
  # Digests still referenced by the remaining revisions (after the revision
  # cleanup) are never deleted, so every kept revision can still roll back;
  # every service in the region counts, as they share the repository.
  if ! revision_digests="$(gcloud run revisions list \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --format='value(status.imageDigest)')"; then
    warn "Could not list revision images; skipping image cleanup"
    return 0
  fi
  if ! images="$(gcloud artifacts docker images list "$IMAGE_REPO_URI" \
    --project "$PROJECT_ID" \
    --sort-by='~UPDATE_TIME' \
    --format='get(version)')"; then
    warn "Could not list images; skipping image cleanup"
    return 0
  fi

  old_images="$(printf '%s\n' "$images" | awk -v keep="$KEEP_IMAGE_COUNT" 'NF && ++n > keep')"
  if [[ -z "$old_images" ]]; then
    echo "       No old images to delete"
  fi

  while IFS= read -r digest; do
    [[ -z "$digest" ]] && continue
    if [[ "$revision_digests" == *"@$digest"* ]]; then
      echo "       Keeping image still used by a revision: $digest"
      continue
    fi
    if gcloud artifacts docker images delete "${IMAGE_REPO_URI}@${digest}" \
      --project "$PROJECT_ID" \
      --quiet \
      --delete-tags >/dev/null 2>&1; then
      echo "       Deleted image digest: $digest"
      deleted="$deleted $digest"
    else
      warn "Failed to delete image digest: $digest"
      failures=$((failures + 1))
    fi
  done <<< "$old_images"

  if [[ "$failures" -gt 0 ]]; then
    warn "Deploy completed, but $failures image(s) need manual cleanup"
  fi

  # An enforced Artifact Registry cleanup policy keeps only its newest
  # AR_POLICY_KEEP images, whatever the revisions use, and Cloud Run keeps
  # its own copy of an image only while a revision serves traffic.
  [[ -z "$AR_POLICY_KEEP" ]] && return 0
  while IFS= read -r digest; do
    [[ -z "$digest" ]] && continue
    [[ " $deleted " == *" $digest "* ]] && continue
    position=$((position + 1))
    if [[ "$position" -gt "$AR_POLICY_KEEP" ]] && [[ "$revision_digests" == *"@$digest"* ]]; then
      at_risk=$((at_risk + 1))
    fi
  done <<< "$images"
  if [[ "$at_risk" -gt 0 ]]; then
    warn "$at_risk kept revision(s) use an image outside the $AR_POLICY_KEEP newest, which the enforced Artifact Registry cleanup policy deletes after a day; such a revision can no longer be rolled back to. Keep DEPLOY_KEEP_REVISIONS / DEPLOY_KEEP_IMAGES at or below $AR_POLICY_KEEP (an image from a failed deploy or a build-only submission also takes a slot; see DEPLOY.md, 回滾)"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id|--service-name|--region|--service-account|--env-file|--image-tag|--keep-revisions|--keep-images)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      ;;
  esac
  case "$1" in
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --service-name)
      SERVICE_NAME="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --service-account)
      SERVICE_ACCOUNT="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --keep-revisions)
      KEEP_REVISION_COUNT="$2"
      shift 2
      ;;
    --keep-images)
      KEEP_IMAGE_COUNT="$2"
      shift 2
      ;;
    --skip-tests)
      SKIP_TESTS="true"
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE="true"
      shift
      ;;
    --skip-enable-apis)
      ENABLE_APIS="false"
      shift
      ;;
    --allow-env-fallback)
      ALLOW_ENV_FALLBACK="true"
      shift
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

require_cmd gcloud
require_cmd python3
require_cmd curl

if [[ ! -f "$ENV_FILE" ]]; then
  fail "Env file not found: $ENV_FILE"
fi
# Its checksum before any read: the settings are checked here and taken
# again before the build, and must be the same file both times.
ENV_FILE_CKSUM="$(cksum < "$ENV_FILE")"
load_env_file

if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(read_env_key GCP_PROJECT_ID)"
fi
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-${GCLOUD_PROJECT:-}}"
fi
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
  PROJECT_ID="${PROJECT_ID//$'\n'/}"
  if [[ "$PROJECT_ID" == "(unset)" ]]; then
    PROJECT_ID=""
  fi
fi

if [[ -z "$SERVICE_NAME" ]]; then
  SERVICE_NAME="$(read_env_key CLOUD_RUN_SERVICE_NAME)"
fi
if [[ -z "$REGION" ]]; then
  REGION="$(read_env_key CLOUD_RUN_REGION)"
fi
if [[ -z "$SERVICE_ACCOUNT" ]]; then
  SERVICE_ACCOUNT="$(read_env_key CLOUD_RUN_SERVICE_ACCOUNT)"
fi
if [[ -z "$KEEP_REVISION_COUNT" ]]; then
  KEEP_REVISION_COUNT="$(read_env_key DEPLOY_KEEP_REVISIONS)"
fi
if [[ -z "$KEEP_IMAGE_COUNT" ]]; then
  KEEP_IMAGE_COUNT="$(read_env_key DEPLOY_KEEP_IMAGES)"
fi
if [[ -z "$ENABLE_APIS" ]]; then
  ENABLE_APIS="$(read_env_key DEPLOY_ENABLE_APIS)"
fi

[[ -z "$SERVICE_NAME" ]] && SERVICE_NAME="linebot-cloud-agent"
[[ -z "$REGION" ]] && REGION="us-west1"
[[ -z "$KEEP_REVISION_COUNT" ]] && KEEP_REVISION_COUNT="2"
[[ -z "$KEEP_IMAGE_COUNT" ]] && KEEP_IMAGE_COUNT="2"
[[ -z "$ENABLE_APIS" ]] && ENABLE_APIS="true"
ENABLE_APIS="$(normalize_bool "$ENABLE_APIS" 'DEPLOY_ENABLE_APIS / --skip-enable-apis')"

if [[ -n "$PROJECT_ID" ]] && is_placeholder_value "$PROJECT_ID"; then
  fail "GCP project is still an example placeholder. Set GCP_PROJECT_ID in $ENV_FILE, use --project-id, or run: gcloud config set project YOUR_PROJECT_ID"
fi

if [[ -z "$PROJECT_ID" ]]; then
  usage >&2
  fail "No GCP project configured. Set GCP_PROJECT_ID in $ENV_FILE, use --project-id, or run: gcloud config set project YOUR_PROJECT_ID"
fi

if [[ ! "$SERVICE_NAME" =~ ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
  fail "Cloud Run service name must be 1-63 chars, lowercase, digits, or hyphen"
fi

if [[ -n "$SERVICE_ACCOUNT" ]] && [[ ! "$SERVICE_ACCOUNT" =~ ^[^@[:space:]]+@[^@[:space:]]+$ ]]; then
  fail "CLOUD_RUN_SERVICE_ACCOUNT / --service-account must be a service account email"
fi

validate_positive_int "$KEEP_REVISION_COUNT" "DEPLOY_KEEP_REVISIONS / --keep-revisions"
validate_positive_int "$KEEP_IMAGE_COUNT" "DEPLOY_KEEP_IMAGES / --keep-images"

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1)"
if [[ -z "$ACTIVE_ACCOUNT" ]]; then
  fail "No active gcloud account found. Run: gcloud auth login"
fi

# Model guard: a v2 .env pins retired NVIDIA ids (HTTP 410) or legacy router
# keys, and a quoted value python-dotenv cannot parse is skipped by the app.
# check-v3 prints key names only (a line that is no KEY=VALUE, whose text may
# be a secret, by its number) and the fix. It runs first, so a broken quote
# is reported as such, not as a missing key.
echo "[step] Checking $ENV_FILE for retired v2 model settings and unreadable values"
if ! python3 "$ROOT_DIR/scripts/envfile.py" check-v3 --file "$ENV_FILE"; then
  fail "Env guard failed: fix the keys and lines listed above, then deploy again"
fi

require_env_key LINE_CHANNEL_SECRET
require_env_key LINE_CHANNEL_ACCESS_TOKEN
require_env_key OPENROUTER_API_KEY
reject_placeholder_env_key NVIDIA_API_KEY
reject_placeholder_env_key TAVILY_API_KEY

SCHEDULED_MESSAGES_ENABLED_VALUE="$(read_env_key SCHEDULED_MESSAGES_ENABLED)"
[[ -z "$SCHEDULED_MESSAGES_ENABLED_VALUE" ]] && SCHEDULED_MESSAGES_ENABLED_VALUE="false"
SCHEDULED_MESSAGES_ENABLED_VALUE="$(normalize_bool "$SCHEDULED_MESSAGES_ENABLED_VALUE" 'SCHEDULED_MESSAGES_ENABLED')"

SCHEDULED_GROUP_ID_VALUE="$(read_env_key SCHEDULED_GROUP_ID)"
SCHEDULED_WEEKLY_MESSAGES_VALUE="$(read_env_key SCHEDULED_WEEKLY_MESSAGES)"
SCHEDULED_YEARLY_MESSAGES_VALUE="$(read_env_key SCHEDULED_YEARLY_MESSAGES)"
GCS_BUCKET_NAME_VALUE="$(read_env_key GCS_BUCKET_NAME)"

HAS_SCHEDULED_JOBS_CONFIG="false"
for raw_value in "$SCHEDULED_WEEKLY_MESSAGES_VALUE" "$SCHEDULED_YEARLY_MESSAGES_VALUE"; do
  normalized_value="$(printf '%s' "$raw_value" | tr -d '[:space:]')"
  if [[ -n "$normalized_value" ]] && [[ "$normalized_value" != "[]" ]]; then
    HAS_SCHEDULED_JOBS_CONFIG="true"
    break
  fi
done

# The Cloud Scheduler job exists whenever INTERNAL_CRON_SECRET is set; it is
# mandatory once scheduled messages are fully configured.
INTERNAL_CRON_SECRET_VALUE="$(read_env_key INTERNAL_CRON_SECRET)"
if [[ "$SCHEDULED_MESSAGES_ENABLED_VALUE" == "true" ]] \
  && [[ -n "$SCHEDULED_GROUP_ID_VALUE" ]] \
  && [[ "$HAS_SCHEDULED_JOBS_CONFIG" == "true" ]]; then
  if [[ -z "$INTERNAL_CRON_SECRET_VALUE" ]]; then
    fail "SCHEDULED_MESSAGES_ENABLED=true requires INTERNAL_CRON_SECRET in $ENV_FILE (generate one with: openssl rand -hex 32)"
  fi
fi
reject_placeholder_env_key INTERNAL_CRON_SECRET

# Post-webhook work runs as Cloud Tasks calling back /internal/tasks/{kind}
# (ADR-0006). The app uses Cloud Tasks only with INTERNAL_TASKS_SECRET set;
# without it every webhook does its work inline and can blow LINE's 2 s
# limit, so anything but TASKS_MODE=inline requires the secret.
TASKS_MODE_VALUE="$(read_env_key TASKS_MODE | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
[[ -z "$TASKS_MODE_VALUE" ]] && TASKS_MODE_VALUE="auto"
case "$TASKS_MODE_VALUE" in
  auto|cloud|inline)
    ;;
  *)
    fail "TASKS_MODE must be one of: auto, cloud, inline"
    ;;
esac
reject_placeholder_env_key INTERNAL_TASKS_SECRET
INTERNAL_TASKS_SECRET_VALUE="$(read_env_key INTERNAL_TASKS_SECRET)"
if [[ "$TASKS_MODE_VALUE" != "inline" ]] && [[ -z "$INTERNAL_TASKS_SECRET_VALUE" ]]; then
  fail "TASKS_MODE=$TASKS_MODE_VALUE requires INTERNAL_TASKS_SECRET in $ENV_FILE (generate one with: openssl rand -hex 32), or set TASKS_MODE=inline"
fi
# A task request must end before both Cloud Tasks and Cloud Run give up on
# it: Cloud Run answers 504 at --timeout but leaves the handler running with
# no CPU. So --timeout follows the dispatch deadline (never below 300 s, the
# inline fallback runs inside the webhook) and the handler budget must stay
# below the deadline.
TASK_DISPATCH_DEADLINE_VALUE="$(read_env_key TASK_DISPATCH_DEADLINE_SECONDS | tr -d '[:space:]')"
[[ -z "$TASK_DISPATCH_DEADLINE_VALUE" ]] && TASK_DISPATCH_DEADLINE_VALUE="300"
validate_positive_int "$TASK_DISPATCH_DEADLINE_VALUE" "TASK_DISPATCH_DEADLINE_SECONDS"
if [[ "$TASK_DISPATCH_DEADLINE_VALUE" -lt 15 ]] || [[ "$TASK_DISPATCH_DEADLINE_VALUE" -gt 1800 ]]; then
  fail "TASK_DISPATCH_DEADLINE_SECONDS must be between 15 and 1800 (the Cloud Tasks limits)"
fi
TASK_HANDLER_BUDGET_VALUE="$(read_env_key TASK_HANDLER_BUDGET_SECONDS | tr -d '[:space:]')"
[[ -z "$TASK_HANDLER_BUDGET_VALUE" ]] && TASK_HANDLER_BUDGET_VALUE="280"
validate_positive_int "$TASK_HANDLER_BUDGET_VALUE" "TASK_HANDLER_BUDGET_SECONDS"
if [[ "$TASK_HANDLER_BUDGET_VALUE" -ge "$TASK_DISPATCH_DEADLINE_VALUE" ]]; then
  fail "TASK_HANDLER_BUDGET_SECONDS ($TASK_HANDLER_BUDGET_VALUE) must be below TASK_DISPATCH_DEADLINE_SECONDS ($TASK_DISPATCH_DEADLINE_VALUE), or the request is cut off while the handler still runs"
fi
CLOUD_RUN_TIMEOUT_SECONDS="300"
if [[ "$TASK_DISPATCH_DEADLINE_VALUE" -gt "$CLOUD_RUN_TIMEOUT_SECONDS" ]]; then
  CLOUD_RUN_TIMEOUT_SECONDS="$TASK_DISPATCH_DEADLINE_VALUE"
fi

CLOUD_TASKS_QUEUE_VALUE="$(read_env_key CLOUD_TASKS_QUEUE)"
[[ -z "$CLOUD_TASKS_QUEUE_VALUE" ]] && CLOUD_TASKS_QUEUE_VALUE="linebot-events"
# The Colab controller's `colab` tasks (minutes each) have their own queue,
# so they never hold the event queue's dispatch slots (ADR-0010).
CLOUD_TASKS_COLAB_QUEUE_VALUE="$(read_env_key CLOUD_TASKS_COLAB_QUEUE)"
[[ -z "$CLOUD_TASKS_COLAB_QUEUE_VALUE" ]] && CLOUD_TASKS_COLAB_QUEUE_VALUE="linebot-colab"
if [[ "$CLOUD_TASKS_COLAB_QUEUE_VALUE" == "$CLOUD_TASKS_QUEUE_VALUE" ]]; then
  fail "CLOUD_TASKS_COLAB_QUEUE must differ from CLOUD_TASKS_QUEUE ($CLOUD_TASKS_QUEUE_VALUE): Colab tasks would take the dispatch slots of LINE events"
fi
# Match the app: a missing key means the Settings default (us-west1), an
# empty value means the Cloud Run region.
if env_has_key CLOUD_TASKS_LOCATION; then
  CLOUD_TASKS_LOCATION_VALUE="$(read_env_key CLOUD_TASKS_LOCATION)"
  [[ -z "$CLOUD_TASKS_LOCATION_VALUE" ]] && CLOUD_TASKS_LOCATION_VALUE="$REGION"
else
  CLOUD_TASKS_LOCATION_VALUE="us-west1"
fi

# The per-minute cron also flushes passive group buffers of chats that went
# quiet below the threshold; without it those wait for the chat's next
# message, a trigger, or the instance stopping.
PASSIVE_RECORDING_VALUE="$(read_env_key PASSIVE_GROUP_RECORDING_ENABLED)"
[[ -z "$PASSIVE_RECORDING_VALUE" ]] && PASSIVE_RECORDING_VALUE="true"
PASSIVE_RECORDING_VALUE="$(normalize_bool "$PASSIVE_RECORDING_VALUE" 'PASSIVE_GROUP_RECORDING_ENABLED')"
if [[ "$PASSIVE_RECORDING_VALUE" == "true" ]] && [[ -z "$INTERNAL_CRON_SECRET_VALUE" ]]; then
  warn "INTERNAL_CRON_SECRET is empty, so no cron job will flush passive group messages; a chat that goes quiet below the threshold is written only on its next message, a trigger, or when the instance stops"
fi

if [[ -z "$IMAGE_TAG" ]]; then
  GIT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo manual)"
  TIMESTAMP="$(date -u +%Y%m%d%H%M%S)"
  IMAGE_TAG="${GIT_SHA}-${TIMESTAMP}"
fi
if [[ ! "$IMAGE_TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  fail "Image tag must match [A-Za-z0-9_][A-Za-z0-9_.-]{0,127}: $IMAGE_TAG"
fi
IMAGE_REPO_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}"
IMAGE_URI="${IMAGE_REPO_URI}:${IMAGE_TAG}"

# Scratch space for the generated env-vars file (it holds every secret from
# .env) and the GCS lifecycle JSON. It lives outside the repo, so gcloud can
# never upload it, and it is removed on any exit.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/linebot-deploy.XXXXXXXX")"
chmod 700 "$WORK_DIR"
cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
case "$WORK_DIR/" in
  "$ROOT_DIR"/*)
    fail "TMPDIR points inside the repository ($WORK_DIR); set TMPDIR elsewhere"
    ;;
esac
TEMP_ENV_YAML="$WORK_DIR/cloudrun-env.yaml"
TEMP_GCS_LIFECYCLE_JSON="$WORK_DIR/gcs-lifecycle.json"
LOOKUP_ERR="$WORK_DIR/gcloud-lookup.err"

if compgen -G "$ROOT_DIR/.cloudrun-env.generated.*" >/dev/null; then
  warn "Stale generated env files (they contain secrets) are still in the repo root; delete them: rm $ROOT_DIR/.cloudrun-env.generated.*"
fi

# Leak guard: list exactly what `gcloud builds submit` would upload and stop
# if any secret file is in it, the env file being deployed included (it may
# have any name; inside the repository git must ignore it too). Fails closed
# if the listing is unavailable.
echo "[step] Checking which files Cloud Build would upload"
if ! UPLOAD_FILE_LIST="$(gcloud meta list-files-for-upload "$ROOT_DIR")"; then
  fail "Could not list the files gcloud would upload (gcloud meta list-files-for-upload); refusing to submit"
fi
if [[ -z "$UPLOAD_FILE_LIST" ]]; then
  fail "gcloud reported no files to upload from $ROOT_DIR; refusing to submit"
fi
if ! printf '%s\n' "$UPLOAD_FILE_LIST" \
  | python3 "$ROOT_DIR/scripts/envfile.py" check-upload --root "$ROOT_DIR" --env-file "$ENV_FILE"; then
  fail "Refusing to submit: secret files would be uploaded or committed. Move them out of the repository, or add them to .gitignore and .gcloudignore."
fi

# The media bucket must stay inside Always Free Cloud Storage; checked before
# any build minute is spent.
if [[ -n "$GCS_BUCKET_NAME_VALUE" ]]; then
  echo "[step] Checking media bucket gs://$GCS_BUCKET_NAME_VALUE"
  MEDIA_BUCKET_JSON="$(gcloud storage buckets describe "gs://$GCS_BUCKET_NAME_VALUE" \
    --project "$PROJECT_ID" \
    --format=json 2>/dev/null || true)"
  if ! MEDIA_BUCKET_PROBLEM="$(MEDIA_BUCKET_JSON="$MEDIA_BUCKET_JSON" \
    FREE_BUCKET_LOCATIONS="$FREE_BUCKET_LOCATIONS" python3 - <<'PY'
from __future__ import annotations

import json
import os
import sys

raw = os.environ["MEDIA_BUCKET_JSON"]
try:
    data = json.loads(raw) if raw.strip() else None
except ValueError:
    data = None
if not isinstance(data, dict):
    print("could not be described (it does not exist, or this account cannot read it)")
    sys.exit(1)
location = str(data.get("location") or "").upper()
storage_class = str(data.get("default_storage_class") or data.get("storageClass") or "").upper()
problems = []
if location not in os.environ["FREE_BUCKET_LOCATIONS"].split():
    problems.append(f"is in {location or 'an unknown location'}")
if storage_class != "STANDARD":
    problems.append(f"has default storage class {storage_class or 'unknown'}")
if problems:
    print(" and ".join(problems))
    sys.exit(1)
PY
  )"; then
    fail "Media bucket gs://$GCS_BUCKET_NAME_VALUE $MEDIA_BUCKET_PROBLEM. Always Free Cloud Storage (5 GB) covers only STANDARD buckets in US-WEST1, US-CENTRAL1 or US-EAST1. Create one: gcloud storage buckets create gs://NEW_BUCKET --project $PROJECT_ID --location=us-west1 --default-storage-class=STANDARD --uniform-bucket-level-access, set GCS_BUCKET_NAME=NEW_BUCKET in $ENV_FILE, grant it with ./scripts/provision_gcp.sh --step runtime-sa --apply, then deploy again"
  fi
  echo "       Bucket is STANDARD in an Always Free location"
fi

# Provisioned resources (scripts/provision_gcp.sh, ADR-0008). Only a
# NOT_FOUND answer counts as "not provisioned"; any other lookup error stops
# the deploy.
echo "[step] Checking provisioned service accounts, staging bucket and secret"
BUILDER_SA="${BUILDER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
STAGING_BUCKET="gs://${PROJECT_ID}-cb-src-usw1"
DEPLOY_RUNTIME_SA="${SERVICE_ACCOUNT:-${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"

# Without both, Cloud Build would stage the source in its default bucket,
# which is US multi-region and outside the free tier.
if ! { sa_exists "$BUILDER_SA" && bucket_in_project "$STAGING_BUCKET"; }; then
  fail "$BUILDER_SA or $STAGING_BUCKET does not exist in project $PROJECT_ID, and without them Cloud Build would stage the source in gs://${PROJECT_ID}_cloudbuild (US multi-region, outside the free tier), so nothing was built. Create them with ./scripts/provision_gcp.sh --step staging-bucket --apply, then --step builder-sa --apply, and deploy again"
fi
echo "       Cloud Build runs as $BUILDER_SA and stages the source in $STAGING_BUCKET"

# The settings are taken from $ENV_FILE once, here, before the build, and
# only if it is still the file checked above: an edit after the checks or
# during the build cannot ship settings nobody checked.
DEPLOY_MODE="env-file"
if sa_exists "$DEPLOY_RUNTIME_SA" && secret_exists "$DOTENV_SECRET"; then
  DEPLOY_MODE="secrets"
  echo "       Cloud Run runs as $DEPLOY_RUNTIME_SA and reads $ENV_FILE from secret $DOTENV_SECRET"
  # The blob this deploy uploads, kept in memory, never in a file. to-dotenv
  # refuses, by key name, values python-dotenv would not read back as-is.
  # The blob ends with a quote and a newline: $(...) drops the newline, and
  # printf '%s\n' puts it back.
  if ! DOTENV_BLOB="$(python3 "$ROOT_DIR/scripts/envfile.py" to-dotenv --file "$ENV_FILE")"; then
    fail "Fix the keys and lines listed above in $ENV_FILE, then deploy again"
  fi
  LOCAL_DOTENV_SHA="$(printf '%s\n' "$DOTENV_BLOB" | python3 "$ROOT_DIR/scripts/envfile.py" sha256)"
  warn_unless_secret_reader "$DOTENV_SECRET" secrets
else
  # Moving a service that reads the secret back to plain variables would
  # expose every secret to anyone who can view the service (ADR-0008).
  if service_mounts_dotenv; then
    if [[ "$ALLOW_ENV_FALLBACK" != "true" ]]; then
      fail "Cloud Run service '$SERVICE_NAME' reads its settings from secret $DOTENV_SECRET, but $DEPLOY_RUNTIME_SA or $DOTENV_SECRET was not found, so this deploy would put every .env value in plain environment variables. Restore them with ./scripts/provision_gcp.sh --step runtime-sa --apply and --step secrets --apply (or fix CLOUD_RUN_SERVICE_ACCOUNT), then deploy again; --allow-env-fallback deploys with plain variables anyway"
    fi
    warn "--allow-env-fallback: service '$SERVICE_NAME' leaves secret $DOTENV_SECRET for plain environment variables"
  fi
  warn "$DEPLOY_RUNTIME_SA or secret $DOTENV_SECRET does not exist, so the deploy puts every .env value in plain Cloud Run environment variables and keeps the service's current service account. Run ./scripts/provision_gcp.sh first"
  if ! (
    umask 077
    python3 "$ROOT_DIR/scripts/envfile.py" to-cloudrun-yaml \
      --file "$ENV_FILE" \
      --set "APP_VERSION=$IMAGE_TAG" \
      "$TEMP_ENV_YAML"
  ); then
    fail "Fix the keys and lines listed above in $ENV_FILE, then deploy again"
  fi
fi
if [[ "$(cksum < "$ENV_FILE")" != "$ENV_FILE_CKSUM" ]]; then
  fail "$ENV_FILE changed while the deploy checked it, so nothing was built or uploaded; deploy again"
fi

# The Colab credential (ADR-0010). The app keeps the tier off, with a
# warning, whenever the file is missing, so a missing secret only warns.
# The newest enabled version is pinned like the dotenv secret, so a
# rollback keeps its token; the deploy never reads the payload.
COLAB_ENABLED_VALUE="$(read_env_key COLAB_ENABLED)"
[[ -z "$COLAB_ENABLED_VALUE" ]] && COLAB_ENABLED_VALUE="false"
COLAB_ENABLED_VALUE="$(normalize_bool "$COLAB_ENABLED_VALUE" 'COLAB_ENABLED')"
if [[ "$COLAB_ENABLED_VALUE" == "true" ]]; then
  # The controller runs only from /internal/cron, which refuses every call
  # without the secret; the app then keeps the tier off (colab_config_problems).
  if [[ -z "$INTERNAL_CRON_SECRET_VALUE" ]]; then
    warn "COLAB_ENABLED=true needs INTERNAL_CRON_SECRET in $ENV_FILE (the Colab controller runs from /internal/cron), so the Colab tier stays off; generate one with: openssl rand -hex 32"
  fi
  echo "[step] Checking the Colab credential secret $COLAB_SECRET"
  if [[ "$DEPLOY_MODE" != "secrets" ]]; then
    warn "COLAB_ENABLED=true, but only a provisioned deploy ($DEPLOY_RUNTIME_SA and $DOTENV_SECRET) mounts $COLAB_SECRET, so the Colab tier stays off (/health/detail colab.reason). Run ./scripts/provision_gcp.sh first"
  elif ! secret_exists "$COLAB_SECRET"; then
    warn "COLAB_ENABLED=true, but secret $COLAB_SECRET does not exist, so the Colab tier stays off (/health/detail colab.reason). Mint it with ./scripts/mint_colab_adc.sh --apply (DEPLOY.md, Colab 層), then deploy again"
  else
    gcloud_lookup secrets versions list "$COLAB_SECRET" \
      --project "$PROJECT_ID" \
      --filter='state=ENABLED' \
      --sort-by='~createTime' \
      --limit=1 \
      --format='value(name)' || true
    COLAB_ADC_VERSION="$(printf '%s\n' "$LOOKUP_OUT" | head -n 1)"
    COLAB_ADC_VERSION="${COLAB_ADC_VERSION##*/}"
    COLAB_ADC_VERSION="${COLAB_ADC_VERSION//[[:space:]]/}"
    if [[ ! "$COLAB_ADC_VERSION" =~ ^[0-9]+$ ]]; then
      COLAB_ADC_VERSION=""
      warn "COLAB_ENABLED=true, but $COLAB_SECRET has no enabled version, so the Colab tier stays off. Mint a new token with ./scripts/mint_colab_adc.sh --apply, then deploy again"
    else
      warn_unless_secret_reader "$COLAB_SECRET" colab-secret
      echo "       Mounting $COLAB_SECRET version $COLAB_ADC_VERSION at $COLAB_MOUNT_PATH"
      COLAB_ADC_PATH_VALUE="$(read_env_key COLAB_ADC_PATH)"
      if [[ -n "$COLAB_ADC_PATH_VALUE" ]] && [[ "$COLAB_ADC_PATH_VALUE" != "$COLAB_MOUNT_PATH" ]]; then
        warn "COLAB_ADC_PATH is $COLAB_ADC_PATH_VALUE, but the deploy mounts the Colab credential at $COLAB_MOUNT_PATH, so the Colab tier stays off; remove COLAB_ADC_PATH from $ENV_FILE or set it to $COLAB_MOUNT_PATH"
      fi
    fi
  fi
fi

if [[ "$ENABLE_APIS" == "true" ]]; then
  echo "[step] Ensuring required GCP APIs are enabled"
  gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    iamcredentials.googleapis.com \
    cloudresourcemanager.googleapis.com \
    cloudscheduler.googleapis.com \
    cloudtasks.googleapis.com \
    --project "$PROJECT_ID" >/dev/null
fi

echo "[step] Ensuring Artifact Registry repository '$AR_REPOSITORY' in $REGION"
# The keepCount of an enforced cleanup policy (provision_gcp.sh
# ar-cleanup-enforce); empty while it is missing or a dry run.
AR_POLICY_KEEP=""
if gcloud_lookup artifacts repositories describe "$AR_REPOSITORY" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --format=json; then
  echo "       Repository already exists"
  AR_POLICY_KEEP="$(printf '%s' "$LOOKUP_OUT" | python3 -c '
import json, sys
raw = sys.stdin.read().strip()
repo = json.loads(raw) if raw else {}
policies = [p for p in (repo.get("cleanupPolicies") or {}).values() if isinstance(p, dict)]
def action(policy):
    value = policy.get("action")
    if isinstance(value, dict):
        value = value.get("type")
    return str(value or "").upper()
if repo.get("cleanupPolicyDryRun") or not any(action(p) == "DELETE" for p in policies):
    sys.exit(0)
keeps = [
    int((p.get("mostRecentVersions") or {}).get("keepCount") or 0)
    for p in policies
    if action(p) == "KEEP"
]
print(max(keeps, default=0))
' 2>/dev/null || true)"
else
  gcloud artifacts repositories create "$AR_REPOSITORY" \
    --project "$PROJECT_ID" \
    --repository-format=docker \
    --location "$REGION" \
    --description "LineBot-CloudAgent container images" >/dev/null
  echo "       Repository created"
fi

# ensure_tasks_queue NAME WORK RATE_FLAGS...: create or update one queue.
ensure_tasks_queue() {
  local queue="$1" work="$2" retry_duration state
  shift 2
  echo "[step] Ensuring Cloud Tasks queue '$queue' in $CLOUD_TASKS_LOCATION_VALUE"
  # --max-attempts=1 and never --max-retry-duration: Cloud Tasks stops
  # retrying only once both limits are met, so a retry duration would add
  # retries back. The rate limits protect the single Cloud Run instance.
  local queue_args=(
    --project "$PROJECT_ID"
    --location "$CLOUD_TASKS_LOCATION_VALUE"
    --max-attempts=1
    "$@"
  )
  if gcloud tasks queues describe "$queue" \
    --project "$PROJECT_ID" \
    --location "$CLOUD_TASKS_LOCATION_VALUE" >/dev/null 2>&1; then
    gcloud tasks queues update "$queue" "${queue_args[@]}" >/dev/null
    echo "       Queue updated"
  else
    gcloud tasks queues create "$queue" "${queue_args[@]}" >/dev/null
    echo "       Queue created"
  fi
  retry_duration="$(gcloud tasks queues describe "$queue" \
    --project "$PROJECT_ID" \
    --location "$CLOUD_TASKS_LOCATION_VALUE" \
    --format='value(retryConfig.maxRetryDuration)' 2>/dev/null || true)"
  if [[ -n "$retry_duration" ]] && [[ "$retry_duration" != "0s" ]]; then
    warn "Cloud Tasks queue '$queue' has maxRetryDuration=$retry_duration, so tasks may run more than once; clear it in the Cloud Console"
  fi
  # A PAUSED queue still accepts tasks (so the app never falls back inline)
  # but dispatches none of them. Resuming is left to a person: the pause
  # may be deliberate.
  state="$(gcloud tasks queues describe "$queue" \
    --project "$PROJECT_ID" \
    --location "$CLOUD_TASKS_LOCATION_VALUE" \
    --format='value(state)' 2>/dev/null || true)"
  if [[ -n "$state" ]] && [[ "$state" != "RUNNING" ]]; then
    fail "Cloud Tasks queue '$queue' is $state, so queued $work would never run. Resume it with: gcloud tasks queues resume $queue --project $PROJECT_ID --location $CLOUD_TASKS_LOCATION_VALUE"
  fi
}

# The queues the app enqueues to: `colab` tasks only while COLAB_ENABLED=true.
TASKS_QUEUES=()
if [[ "$TASKS_MODE_VALUE" != "inline" ]]; then
  ensure_tasks_queue "$CLOUD_TASKS_QUEUE_VALUE" "webhook work" \
    --max-concurrent-dispatches=4 --max-dispatches-per-second=5
  TASKS_QUEUES+=("$CLOUD_TASKS_QUEUE_VALUE")
  if [[ "$COLAB_ENABLED_VALUE" == "true" ]]; then
    # Two slots: at most an assign or launch plus a heartbeat or reconcile.
    ensure_tasks_queue "$CLOUD_TASKS_COLAB_QUEUE_VALUE" "Colab work" \
      --max-concurrent-dispatches=2 --max-dispatches-per-second=1
    TASKS_QUEUES+=("$CLOUD_TASKS_COLAB_QUEUE_VALUE")
  fi
elif [[ "$COLAB_ENABLED_VALUE" == "true" ]]; then
  warn "COLAB_ENABLED=true needs Cloud Tasks, but TASKS_MODE=inline, so the Colab tier stays off (/health/detail colab.reason)"
fi

substitutions=(
  "_REGION=$REGION"
  "_AR_REPOSITORY=$AR_REPOSITORY"
  "_IMAGE_TAG=$IMAGE_TAG"
  "_SKIP_TESTS=$SKIP_TESTS"
)
SUBSTITUTIONS="$(IFS=,; echo "${substitutions[*]}")"

# cloudbuild.yaml logs to Cloud Logging only; the GA command cannot stream
# those logs, the beta one can.
build_submit=(gcloud builds submit)
if gcloud_beta_installed; then
  build_submit=(gcloud beta builds submit)
fi

# cloudbuild.yaml logs to Cloud Logging only, which a user-specified build
# service account requires.
build_args=(
  "$ROOT_DIR"
  --project "$PROJECT_ID"
  --config "$ROOT_DIR/cloudbuild.yaml"
  --substitutions "$SUBSTITUTIONS"
  "--service-account=projects/${PROJECT_ID}/serviceAccounts/${BUILDER_SA}"
  "--gcs-source-staging-dir=${STAGING_BUCKET}/source"
)

echo "[step] Submitting Cloud Build (ruff + pytest, then build and push the image)"
echo "       project=$PROJECT_ID region=$REGION image=$IMAGE_URI"
"${build_submit[@]}" "${build_args[@]}"

deploy_args=(
  run deploy "$SERVICE_NAME"
  --project "$PROJECT_ID"
  --region "$REGION"
  --platform managed
  --image "$IMAGE_URI"
  --allow-unauthenticated
  --cpu-throttling
  --min-instances "$MIN_INSTANCES"
  --max-instances "$MAX_INSTANCES"
  --max "$MAX_INSTANCES"
  --timeout "$CLOUD_RUN_TIMEOUT_SECONDS"
  --memory 512Mi
  --cpu 1
  # The deploy owns every volume. --set-secrets / --clear-secrets alone
  # leave the volumes the container mounted until now in the template,
  # unmounted (seen 2026-10-01: four secret volumes, two mounted), and every
  # later revision carries them. gcloud clears these first, then adds back
  # only what the secret flags mount.
  --clear-volume-mounts
  --clear-volumes
  --quiet
)
if [[ "$DEPLOY_MODE" == "secrets" ]]; then
  # A new version only when the blob checked before the build changed; the
  # hashes are compared, never printed. The payload travels through a pipe,
  # never a file, and the read runs with file logging off: gcloud copies
  # whatever it prints into ~/.config/gcloud/logs (kept about 30 days).
  echo "[step] Syncing $ENV_FILE to Secret Manager secret '$DOTENV_SECRET'"
  if REMOTE_DOTENV_SHA="$(CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true gcloud secrets versions access latest \
    --secret "$DOTENV_SECRET" \
    --project "$PROJECT_ID" \
    --format='get(payload.data)' 2>/dev/null \
    | python3 "$ROOT_DIR/scripts/envfile.py" sha256 --base64 2>/dev/null)" \
    && [[ "$REMOTE_DOTENV_SHA" == "$LOCAL_DOTENV_SHA" ]]; then
    DOTENV_VERSION="$(gcloud secrets versions describe latest \
      --secret "$DOTENV_SECRET" \
      --project "$PROJECT_ID" \
      --format='value(name)')"
    echo "       Unchanged since the latest version"
  else
    DOTENV_VERSION="$(printf '%s\n' "$DOTENV_BLOB" \
      | gcloud secrets versions add "$DOTENV_SECRET" \
        --project "$PROJECT_ID" \
        --data-file=- \
        --format='value(name)')"
    echo "       Added a new version"
  fi
  DOTENV_VERSION="${DOTENV_VERSION##*/}"
  DOTENV_VERSION="${DOTENV_VERSION//[[:space:]]/}"
  if [[ ! "$DOTENV_VERSION" =~ ^[0-9]+$ ]]; then
    fail "Could not determine which $DOTENV_SECRET version to deploy"
  fi
  echo "       Pinning $DOTENV_SECRET version $DOTENV_VERSION"
  # --set-secrets replaces every mount (all of them in one flag), so a
  # deploy without COLAB_ENABLED drops the Colab token.
  # --set-env-vars replaces every plain variable, so the old plain-text
  # secrets disappear and cannot shadow the mounted file. Pinned, the
  # versions roll back with their revision.
  SECRET_MOUNTS="${SECRET_MOUNT_PATH}=${DOTENV_SECRET}:${DOTENV_VERSION}"
  if [[ -n "$COLAB_ADC_VERSION" ]]; then
    SECRET_MOUNTS="${SECRET_MOUNTS},${COLAB_MOUNT_PATH}=${COLAB_SECRET}:${COLAB_ADC_VERSION}"
  fi
  deploy_args+=(
    --service-account "$DEPLOY_RUNTIME_SA"
    "--set-secrets=${SECRET_MOUNTS}"
    "--set-env-vars=APP_VERSION=${IMAGE_TAG}"
  )
else
  # The env-vars file was made before the build. The revision must not
  # depend on a secret the env file replaces.
  deploy_args+=(--env-vars-file "$TEMP_ENV_YAML" --clear-secrets)
  if [[ -n "$SERVICE_ACCOUNT" ]]; then
    deploy_args+=(--service-account "$SERVICE_ACCOUNT")
  fi
fi

echo "[step] Deploying $IMAGE_URI to Cloud Run service '$SERVICE_NAME'"
echo "       min_instances=$MIN_INSTANCES max_instances=$MAX_INSTANCES (service max=$MAX_INSTANCES)"
# File logging off: gcloud writes every parsed flag to its DEBUG log under
# ~/.config/gcloud/logs, and the env-file fallback's --env-vars-file parses
# into every .env value.
CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true gcloud "${deploy_args[@]}"
rm -f "$TEMP_ENV_YAML"

gcloud run services update-traffic "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --to-latest >/dev/null

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')"

if [[ -z "$SERVICE_URL" ]]; then
  fail "Could not determine deployed Cloud Run URL"
fi

LATEST_REVISION="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.latestReadyRevisionName)')"

RUNTIME_SERVICE_ACCOUNT="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(spec.template.spec.serviceAccountName)')"

# Signing media URLs needs roles/iam.serviceAccountTokenCreator on the
# runtime service account itself, or image and voice replies fall back to
# text. scripts/provision_gcp.sh grants it to linebot-runtime; any other
# account (CLOUD_RUN_SERVICE_ACCOUNT, or the service's own before
# provisioning) gets it here when it lacks it.
if [[ -n "$GCS_BUCKET_NAME_VALUE" ]] \
  && [[ "$RUNTIME_SERVICE_ACCOUNT" != "${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" ]]; then
  if [[ -z "$RUNTIME_SERVICE_ACCOUNT" ]]; then
    warn "Runtime service account unknown; grant it roles/iam.serviceAccountTokenCreator on itself, or image and voice replies fall back to text"
  elif ! gcloud iam service-accounts get-iam-policy "$RUNTIME_SERVICE_ACCOUNT" \
    --project "$PROJECT_ID" \
    --format=json 2>/dev/null \
    | policy_has_binding roles/iam.serviceAccountTokenCreator "serviceAccount:$RUNTIME_SERVICE_ACCOUNT"; then
    echo "[step] Ensuring runtime service account can sign GCS URLs"
    if ! gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SERVICE_ACCOUNT" \
      --project "$PROJECT_ID" \
      --member="serviceAccount:$RUNTIME_SERVICE_ACCOUNT" \
      --role="roles/iam.serviceAccountTokenCreator" >/dev/null; then
      warn "Could not grant $RUNTIME_SERVICE_ACCOUNT roles/iam.serviceAccountTokenCreator on itself; image and voice replies fall back to text until it is granted"
    fi
  fi
fi

if [[ -n "$GCS_BUCKET_NAME_VALUE" ]]; then
  echo "[step] Ensuring GCS lifecycle rule (auto-delete after 3 days)"
  BUCKET_URL="gs://$GCS_BUCKET_NAME_VALUE"
  CURRENT_BUCKET_JSON="$(gcloud storage buckets describe "$BUCKET_URL" \
    --project "$PROJECT_ID" \
    --format=json)"
  DESIRED_LIFECYCLE_JSON="$(CURRENT_BUCKET_JSON="$CURRENT_BUCKET_JSON" python3 - <<'PY'
from __future__ import annotations

import json
import os

data = json.loads(os.environ["CURRENT_BUCKET_JSON"])
lifecycle_config = data.get("lifecycle_config") or data.get("lifecycle") or {}
rules = (lifecycle_config.get("rule") or [])
target_rule = {"action": {"type": "Delete"}, "condition": {"age": 3}}

def normalize(rule: dict) -> str:
    return json.dumps(rule, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

merged_rules: list[dict] = []
seen: set[str] = set()
for rule in [*rules, target_rule]:
    if not isinstance(rule, dict):
        continue
    marker = normalize(rule)
    if marker in seen:
        continue
    seen.add(marker)
    merged_rules.append(rule)

print(json.dumps({"rule": merged_rules}, ensure_ascii=False, separators=(",", ":")))
PY
)"
  printf '%s' "$DESIRED_LIFECYCLE_JSON" > "$TEMP_GCS_LIFECYCLE_JSON"
  if ! gcloud storage buckets update "$BUCKET_URL" \
    --lifecycle-file="$TEMP_GCS_LIFECYCLE_JSON" \
    --project "$PROJECT_ID" >/dev/null; then
    fail "Could not set GCS lifecycle rule on $BUCKET_URL"
  fi

  VERIFIED_BUCKET_JSON="$(gcloud storage buckets describe "$BUCKET_URL" \
    --project "$PROJECT_ID" \
    --format=json)"
  if ! VERIFIED_BUCKET_JSON="$VERIFIED_BUCKET_JSON" python3 - <<'PY'
from __future__ import annotations

import json
import os
import sys

data = json.loads(os.environ["VERIFIED_BUCKET_JSON"])
lifecycle_config = data.get("lifecycle_config") or data.get("lifecycle") or {}
rules = (lifecycle_config.get("rule") or [])
has_target = any(
    isinstance(rule, dict)
    and (rule.get("action") or {}).get("type") == "Delete"
    and (rule.get("condition") or {}).get("age") == 3
    for rule in rules
)
sys.exit(0 if has_target else 1)
PY
  then
    fail "GCS lifecycle verification failed: expected a 3-day Delete rule on $BUCKET_URL"
  fi
  echo "       GCS lifecycle rule verified: delete objects after 3 days"
fi

if [[ "${#TASKS_QUEUES[@]}" -gt 0 ]]; then
  # Without this the app cannot create tasks: post-webhook work runs
  # inline, and Colab work is not queued at all.
  if [[ -n "$RUNTIME_SERVICE_ACCOUNT" ]]; then
    echo "[step] Ensuring runtime service account can enqueue Cloud Tasks"
    for queue in "${TASKS_QUEUES[@]}"; do
      if ! gcloud tasks queues add-iam-policy-binding "$queue" \
        --project "$PROJECT_ID" \
        --location "$CLOUD_TASKS_LOCATION_VALUE" \
        --member="serviceAccount:$RUNTIME_SERVICE_ACCOUNT" \
        --role="roles/cloudtasks.enqueuer" >/dev/null; then
        warn "Could not grant roles/cloudtasks.enqueuer on queue '$queue'; its tasks cannot be queued until it is granted"
      fi
    done
  else
    warn "Runtime service account unknown; grant it roles/cloudtasks.enqueuer on queue(s) ${TASKS_QUEUES[*]} manually"
  fi
fi

echo "[step] Ensuring public Cloud Run access for LINE webhook"
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member="allUsers" \
  --role="roles/run.invoker" >/dev/null

# The cron job exists whenever INTERNAL_CRON_SECRET is set, even with
# scheduled messages off: /internal/cron answers 200 without a scheduler
# and still flushes stale passive group buffers (ADR-0006).
# --max-retry-attempts=0 is load-bearing for dedupe (ADR-0001).
SCHEDULER_JOB_NAME="${SERVICE_NAME}-cron"
if [[ -n "$INTERNAL_CRON_SECRET_VALUE" ]]; then
  echo "[step] Ensuring Cloud Scheduler job '$SCHEDULER_JOB_NAME' for /internal/cron"
  scheduler_args=(
    --project "$PROJECT_ID"
    --location "$REGION"
    --schedule="* * * * *"
    --time-zone="Asia/Taipei"
    --uri="$SERVICE_URL/internal/cron"
    --http-method=POST
    --attempt-deadline=60s
    --max-retry-attempts=0
    --description="LineBot /internal/cron trigger (scheduled messages and maintenance)"
  )
  # The token has to travel as a flag value, and gcloud writes every parsed
  # flag to its DEBUG log under ~/.config/gcloud/logs (kept ~30 days), and
  # every line it prints (a described job's headers) as well. File logging
  # is off for these calls (gcloud_lookup included) so the secret is not
  # copied to disk; it is still briefly visible in the local process list.
  if gcloud_lookup scheduler jobs describe "$SCHEDULER_JOB_NAME" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --format='value(name)'; then
    CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true \
      gcloud scheduler jobs update http "$SCHEDULER_JOB_NAME" \
      "${scheduler_args[@]}" \
      --update-headers="X-Cron-Token=$INTERNAL_CRON_SECRET_VALUE" >/dev/null
    echo "       Cloud Scheduler job updated"
  else
    CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true \
      gcloud scheduler jobs create http "$SCHEDULER_JOB_NAME" \
      "${scheduler_args[@]}" \
      --headers="X-Cron-Token=$INTERNAL_CRON_SECRET_VALUE" >/dev/null
    echo "       Cloud Scheduler job created"
  fi
else
  if gcloud_lookup scheduler jobs describe "$SCHEDULER_JOB_NAME" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --format='value(name)'; then
    echo "[step] Removing Cloud Scheduler job '$SCHEDULER_JOB_NAME' (INTERNAL_CRON_SECRET is empty)"
    CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true gcloud scheduler jobs delete "$SCHEDULER_JOB_NAME" \
      --project "$PROJECT_ID" \
      --location "$REGION" \
      --quiet >/dev/null
  fi
fi

if [[ "$SKIP_SMOKE" != "true" ]]; then
  echo "[step] Running post-deploy smoke checks"
  if ! wait_for_public_health "$SERVICE_URL"; then
    fail "Service did not become publicly reachable in time: $SERVICE_URL/health"
  fi

  # The secret goes through the environment, not argv, so `ps` cannot show it.
  SMOKE_WEBHOOK_SECRET="$(read_env_key LINE_CHANNEL_SECRET)" \
    python3 "$ROOT_DIR/scripts/smoke_test_cloud.py" \
    --base-url "$SERVICE_URL" \
    --expect-ready
fi

# Cleanup runs only after a passing smoke check (a failed one stops the
# deploy above), so a failed or skipped check keeps the previous revisions,
# images and secret versions around for a rollback.
if [[ "$SKIP_SMOKE" == "true" ]]; then
  echo "[step] Keeping every old revision and image (smoke checks were skipped)"
else
  echo "[step] Cleaning up old revisions (keeping the ${KEEP_REVISION_COUNT} newest Ready ones)"
  cleanup_old_revisions
  echo "[step] Cleaning up old images (keeping ${KEEP_IMAGE_COUNT})"
  cleanup_old_images
fi
if [[ "$DEPLOY_MODE" == "secrets" ]]; then
  # Destroying is permanent, so it waits for a smoke check that passed. The
  # Colab secret is cleaned only by a deploy that mounts it.
  if [[ "$SKIP_SMOKE" == "true" ]]; then
    keep_secret_versions "$DOTENV_SECRET"
    if [[ -n "$COLAB_ADC_VERSION" ]]; then
      keep_secret_versions "$COLAB_SECRET"
    fi
  else
    echo "[step] Destroying $DOTENV_SECRET versions no kept revision uses"
    cleanup_old_secret_versions "$DOTENV_SECRET" "$DOTENV_VERSION"
    if [[ -n "$COLAB_ADC_VERSION" ]]; then
      echo "[step] Destroying $COLAB_SECRET versions no kept revision uses"
      cleanup_old_secret_versions "$COLAB_SECRET" "$COLAB_ADC_VERSION"
    fi
  fi
fi

echo "[done] Cloud deployment complete"
echo "       Active gcloud account: ${ACTIVE_ACCOUNT}"
echo "       Image: ${IMAGE_URI}"
echo "       Service URL: ${SERVICE_URL}"
if [[ -n "$LATEST_REVISION" ]]; then
  echo "       Latest revision: ${LATEST_REVISION}"
fi
if [[ "$DEPLOY_MODE" == "secrets" ]]; then
  echo "       Settings: secret ${DOTENV_SECRET} version ${DOTENV_VERSION} at ${SECRET_MOUNT_PATH}"
fi
if [[ -n "$COLAB_ADC_VERSION" ]]; then
  echo "       Colab credential: secret ${COLAB_SECRET} version ${COLAB_ADC_VERSION} at ${COLAB_MOUNT_PATH}"
fi
echo "       Webhook URL: ${SERVICE_URL}/webhook"
echo "       Roll back: gcloud run services update-traffic ${SERVICE_NAME} --project ${PROJECT_ID} --region ${REGION} --to-revisions=PREVIOUS_REVISION=100"
if [[ -n "$INTERNAL_CRON_SECRET_VALUE" ]]; then
  echo "       (job ${SCHEDULER_JOB_NAME}'s X-Cron-Token stays this deploy's INTERNAL_CRON_SECRET: a revision from before a change of it answers /internal/cron with 403; see DEPLOY.md, 回滾)"
fi
