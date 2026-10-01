#!/usr/bin/env bash
# Mints the Colab tier's credential (ADR-0010) and stores it in Secret
# Manager. A person runs this on their own machine: Google's consent screen
# needs a browser and the Colab account's password, which no script has.
#
# Dry run by default: the read-only checks run and every step that would
# mint, store or grant something is printed, never run. With --apply:
#
#   1. `gcloud auth application-default login` under a scratch
#      CLOUDSDK_CONFIG (mktemp -d, mode 700), so the machine's own ADC and
#      gcloud logins stay untouched. Sign in with the Colab account.
#   2. The file must be an authorized_user credential with a refresh token
#      (the same rule the bot's control plane applies); nothing is stored
#      otherwise.
#   3. It becomes a new version of linebot-colab-adc (the secret is created
#      on first use), read from stdin; the payload is never printed.
#   4. linebot-runtime gets read access through
#      `scripts/provision_gcp.sh --step colab-secret --apply`.
#   5. The scratch directory is overwritten where possible and deleted, on
#      any exit.
#
# Stays compatible with macOS /bin/bash 3.2: no mapfile, no ${var,,}.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
PROJECT_ID=""
APPLY="false"

# Shared with scripts/provision_gcp.sh and scripts/deploy_cloud_run.sh.
COLAB_SECRET="linebot-colab-adc"
RUNTIME_SA_NAME="linebot-runtime"
COLAB_MOUNT_PATH="/secrets/colab/adc.json"
# openid and cloud-platform only because gcloud insists on them; Colab's
# session backend needs userinfo.email, colaboratory is kept for the future.
SCOPES="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"

usage() {
  cat <<'USAGE'
Usage: ./scripts/mint_colab_adc.sh [--apply] [options]

Mint the Colab account's refresh token and store it as a new version of the
Secret Manager secret linebot-colab-adc, readable by linebot-runtime. The
deploy mounts it at /secrets/colab/adc.json while COLAB_ENABLED=true.
See DEPLOY.md (Colab 層) and colab/README.md.

Without --apply nothing is minted, stored or granted: the plan is printed
and only read-only checks run.

Options:
  --apply            Sign in (browser consent), store the token, grant access
  --project-id ID    GCP project (default: GCP_PROJECT_ID in the env file, which it
                     must match; never gcloud's default project)
  --env-file PATH    Env file to read GCP_PROJECT_ID from (default: .env); missing,
                     unreadable or unparsable, it stops the script (missing is
                     allowed with --project-id)
  -h, --help         Show this help message
USAGE
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

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}


# lookup ARGS...: a read-only gcloud call. Returns 0 with the output in
# LOOKUP_OUT when it succeeds and 1 when gcloud says the resource was not
# found; any other error stops the script before anything is minted.
# Prompts are off: stderr goes to a file, where gcloud's "API [x] not
# enabled ... enable and retry? (y/N)" would wait unseen.
LOOKUP_OUT=""
lookup() {
  local rc=0
  LOOKUP_OUT="$(CLOUDSDK_CORE_DISABLE_PROMPTS=1 gcloud "$@" 2>"$LOOKUP_ERR")" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi
  if grep -Eqi 'not[ _]found|cannot find' "$LOOKUP_ERR"; then
    return 1
  fi
  cat "$LOOKUP_ERR" >&2
  fail "gcloud $1 $2 $3 failed, and not with NOT_FOUND; fix the access (gcloud auth login, the account's roles, the Secret Manager API), then run this again. Nothing was minted"
}

print_account_advice() {
  cat <<ADVICE
[note] Sign in with a DEDICATED Google account that holds the Colab plan and has
       no role in project $PROJECT_ID. gcloud always adds the cloud-platform scope,
       so a token minted by an owner of the project could manage all of it from
       Secret Manager. Use a private browser window signed in to that account only,
       and use the account for nothing else with gcloud or colab-cli: each login
       mints another token on the same Cloud SDK OAuth client, and past 100 live
       tokens Google silently revokes the oldest, possibly the bot's.
ADVICE
}

print_revocation_advice() {
  cat <<REVOKE
[note] The token lasts until revoked, or until it goes unused for 6 months (while
       the tier is on, the bot uses it every day). To cut it off (it leaked, or the tier is
       retired): sign in to https://myaccount.google.com/connections as the Colab
       account and remove "Google Cloud SDK". That revokes every token the account
       minted through gcloud, this one included, which is one more reason for a
       dedicated account. The Colab tier then turns DISABLED with invalid_grant and
       replies go on through NVIDIA and OpenRouter. Mint a new token with
       ./scripts/mint_colab_adc.sh --apply and deploy, then send !colab on.
       A version replaced by a newer one needs nothing: after a passing smoke check
       the deploy destroys the $COLAB_SECRET versions no kept revision pins.
REVOKE
}

# Deletes the scratch config (the token and gcloud's own files) on any exit.
# Overwriting first is best effort: SSDs and APFS may keep old blocks.
WORK_DIR=""
cleanup() {
  [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]] || return 0
  if command -v shred >/dev/null 2>&1; then
    find "$WORK_DIR" -type f -exec shred -u {} + 2>/dev/null || true
  elif command -v gshred >/dev/null 2>&1; then
    find "$WORK_DIR" -type f -exec gshred -u {} + 2>/dev/null || true
  fi
  rm -rf "$WORK_DIR"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id|--env-file)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      ;;
  esac
  case "$1" in
    --apply)
      APPLY="true"
      shift
      ;;
    --project-id)
      PROJECT_ID="$2"
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

require_cmd gcloud
require_cmd python3

# The project comes from $ENV_FILE, never gcloud's default project, and the
# file must pass the check scripts/provision_gcp.sh (run at the end) applies,
# so a token is never minted for a grant that would then be refused. Only
# an explicit --project-id runs without an env file.
ENV_PROJECT_ID=""
if [[ -e "$ENV_FILE" ]]; then
  [[ -r "$ENV_FILE" ]] || fail "Cannot read env file: $ENV_FILE"
  if ! python3 "$ROOT_DIR/scripts/envfile.py" to-dotenv --file "$ENV_FILE" --sha256 >/dev/null; then
    fail "Fix the keys and lines listed above in $ENV_FILE (it must be UTF-8), then run this again"
  fi
  ENV_PROJECT_ID="$(python3 "$ROOT_DIR/scripts/envfile.py" get --file "$ENV_FILE" GCP_PROJECT_ID || true)"
elif [[ -z "$PROJECT_ID" ]]; then
  fail "Env file not found: $ENV_FILE. Pass --env-file, or --project-id to run without one"
fi
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$ENV_PROJECT_ID"
  [[ -n "$PROJECT_ID" ]] || fail "GCP_PROJECT_ID is empty in $ENV_FILE: set it, or pass --project-id"
elif [[ -n "$ENV_PROJECT_ID" ]] && [[ "$ENV_PROJECT_ID" != "$PROJECT_ID" ]]; then
  fail "--project-id $PROJECT_ID does not match GCP_PROJECT_ID=$ENV_PROJECT_ID in $ENV_FILE"
fi
if [[ ! "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
  fail "Not a GCP project id: $PROJECT_ID"
fi
RUNTIME_SA="$RUNTIME_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Scratch space outside the repository: gcloud's error messages, and with
# --apply the scratch CLOUDSDK_CONFIG that receives the token.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/linebot-colab-adc.XXXXXXXX")"
chmod 700 "$WORK_DIR"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
case "$WORK_DIR/" in
  "$ROOT_DIR"/*)
    fail "TMPDIR points inside the repository ($WORK_DIR); set TMPDIR elsewhere"
    ;;
esac
LOOKUP_ERR="$WORK_DIR/gcloud.err"
ADC_FILE="$WORK_DIR/application_default_credentials.json"

if [[ "$APPLY" == "true" ]]; then
  echo "[mode] APPLY: mints a Colab token and stores it in project $PROJECT_ID"
else
  echo "[mode] dry run: nothing is minted, stored or granted in project $PROJECT_ID (add --apply to run the [dry-run] lines)"
fi

# Everything that could stop the store runs before the token exists, so a
# failure never leaves a minted token unused.
echo "[step] checks"
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1 || true)"
if [[ -z "$ACTIVE_ACCOUNT" ]]; then
  fail "No active gcloud account. Run gcloud auth login with the account that deploys (not the Colab account)"
fi
note "Secret Manager calls run as $ACTIVE_ACCOUNT (your usual gcloud login; the Colab account is only used in the browser)"
SECRET_EXISTS="false"
if lookup secrets describe "$COLAB_SECRET" --project "$PROJECT_ID" --format='value(name)'; then
  SECRET_EXISTS="true"
  note "secret $COLAB_SECRET exists: the token becomes a new version"
else
  note "secret $COLAB_SECRET does not exist: it is created (automatic replication)"
fi

echo "[step] mint"
print_account_advice
LOGIN_CMD="gcloud auth application-default login --no-launch-browser --disable-quota-project --scopes=$SCOPES"
if [[ "$APPLY" != "true" ]]; then
  echo "  [dry-run] CLOUDSDK_CONFIG=<scratch dir> $LOGIN_CMD"
else
  echo "  [run] CLOUDSDK_CONFIG=$WORK_DIR $LOGIN_CMD"
  note "Open the URL it prints in that private window, allow access, and paste the code back here."
  if ! CLOUDSDK_CONFIG="$WORK_DIR" gcloud auth application-default login \
    --no-launch-browser \
    --disable-quota-project \
    --scopes="$SCOPES"; then
    fail "The sign-in did not finish; nothing was stored"
  fi
  [[ -f "$ADC_FILE" ]] || fail "gcloud wrote no $ADC_FILE; nothing was stored"
  chmod 600 "$ADC_FILE"
  # The rule of src/services/colab/control_plane.py (adc_file_problem);
  # names fields only, never values.
  if ! python3 - "$ADC_FILE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        info = json.load(handle)
except (OSError, ValueError) as exc:
    print(f"the credential file cannot be read ({type(exc).__name__})")
    sys.exit(1)
if not isinstance(info, dict) or info.get("type") != "authorized_user":
    kind = info.get("type") if isinstance(info, dict) else type(info).__name__
    print(f"the credential is {kind!r}, not an authorized_user refresh token")
    sys.exit(1)
missing = [name for name in ("client_id", "client_secret", "refresh_token") if not info.get(name)]
if missing:
    print(f"the credential lacks {', '.join(missing)}")
    sys.exit(1)
PY
  then
    fail "The minted credential cannot drive Colab (see above); nothing was stored"
  fi
  note "ok: an authorized_user credential with a refresh token"
fi

echo "[step] store"
if [[ "$SECRET_EXISTS" == "true" ]]; then
  STORE_CMD="gcloud secrets versions add $COLAB_SECRET --project $PROJECT_ID --data-file=-"
else
  STORE_CMD="gcloud secrets create $COLAB_SECRET --project $PROJECT_ID --replication-policy=automatic --data-file=-"
fi
STORED_VERSION=""
if [[ "$APPLY" != "true" ]]; then
  echo "  [dry-run] $STORE_CMD < <the minted credential file>"
else
  echo "  [run] $STORE_CMD < <the minted credential file>"
  # stdin, never an argument; gcloud's file log stays off.
  if [[ "$SECRET_EXISTS" == "true" ]]; then
    STORED_VERSION="$(CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true gcloud secrets versions add "$COLAB_SECRET" \
      --project "$PROJECT_ID" \
      --data-file=- \
      --format='value(name)' < "$ADC_FILE")" \
      || fail "Could not add a version to $COLAB_SECRET; nothing was stored"
    STORED_VERSION="${STORED_VERSION##*/}"
  else
    CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true gcloud secrets create "$COLAB_SECRET" \
      --project "$PROJECT_ID" \
      --replication-policy=automatic \
      --data-file=- < "$ADC_FILE" >/dev/null \
      || fail "Could not create $COLAB_SECRET; nothing was stored"
    STORED_VERSION="1"
  fi
  STORED_VERSION="${STORED_VERSION//[[:space:]]/}"
  note "stored as $COLAB_SECRET version $STORED_VERSION"
fi

echo "[step] grant"
# One place decides the binding: the provisioning step (idempotent).
if [[ "$APPLY" != "true" ]]; then
  echo "  [dry-run] ./scripts/provision_gcp.sh --step colab-secret --apply --project-id $PROJECT_ID  (roles/secretmanager.secretAccessor for $RUNTIME_SA)"
else
  echo "  [run] ./scripts/provision_gcp.sh --step colab-secret --apply --project-id $PROJECT_ID"
  bash "$ROOT_DIR/scripts/provision_gcp.sh" --step colab-secret --apply --project-id "$PROJECT_ID" --env-file "$ENV_FILE" \
    || fail "$COLAB_SECRET version $STORED_VERSION is stored, but granting $RUNTIME_SA failed; rerun: ./scripts/provision_gcp.sh --step colab-secret --apply"
fi

echo "[step] next"
print_revocation_advice
if [[ "$APPLY" == "true" ]]; then
  cat <<NEXT
[next] 1. In .env: FIRESTORE_ENABLED=true, COLAB_SHARED_SECRET (32+ random characters),
          LINE_ADMIN_USER_IDS, COLAB_ENABLED=true; for the first runs also
          COLAB_DRY_RUN=true and MODEL_PATH_PREFERENCE=api (DEPLOY.md, Colab 層).
       2. ./scripts/deploy_cloud_run.sh mounts $COLAB_SECRET version $STORED_VERSION at
          $COLAB_MOUNT_PATH (the newest enabled version, pinned per revision).
       3. After a re-mint: deploy, then send !colab on if the tier was DISABLED.
[done] minted and stored; the scratch config is deleted on exit
NEXT
else
  echo "[done] dry run finished; nothing was minted, stored or granted"
fi
