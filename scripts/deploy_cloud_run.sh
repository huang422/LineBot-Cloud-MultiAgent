#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ID=""
SERVICE_NAME=""
REGION=""
MIN_INSTANCE_COUNT=""
ENV_FILE="$ROOT_DIR/.env"
IMAGE_TAG=""
SKIP_TESTS="false"
SKIP_SMOKE="false"
KEEP_REVISION_COUNT=""
KEEP_IMAGE_COUNT=""
ENABLE_APIS=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/deploy_cloud_run.sh [options]

One-command cloud deployment for LineBot-CloudAgent.
Default deploy settings are loaded from .env, so the normal workflow is:
  ./scripts/deploy_cloud_run.sh

The script will:
  1. read deploy settings from .env,
  2. verify required runtime env keys,
  3. ensure required GCP APIs are enabled,
  4. generate a temporary Cloud Run env-vars file,
  5. submit Cloud Build to run tests, build, deploy, smoke check, and cleanup,
  6. print the deployed Cloud Run URL and latest revision.

Deploy settings supported in .env:
  GCP_PROJECT_ID
  CLOUD_RUN_SERVICE_NAME
  CLOUD_RUN_REGION
  DEPLOY_KEEP_REVISIONS
  DEPLOY_KEEP_IMAGES
  DEPLOY_ENABLE_APIS

Options:
  --project-id ID            Override GCP project ID
  --service-name NAME        Override Cloud Run service name
  --region REGION            Override Cloud Run region
  --env-file PATH            Env file to deploy (default: .env)
  --image-tag TAG            Override Docker image tag (default: <git-sha>-<utc-timestamp>)
  --keep-revisions N         Override how many revisions to keep (default from .env or 1)
  --keep-images N            Override how many image digests to keep (default from .env or 3)
  --skip-tests               Skip Cloud Build unit tests and compile checks
  --skip-smoke               Skip the post-deploy cloud smoke checks
  --skip-enable-apis         Do not auto-enable required GCP APIs
  -h, --help                 Show this help message
USAGE
}

fail() {
  echo "[fail] $*" >&2
  exit 1
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

validate_non_negative_int() {
  local value="$1"
  local name="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 0 ]]; then
    fail "$name must be an integer greater than or equal to 0"
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

read_env_key() {
  python3 "$ROOT_DIR/scripts/envfile.py" get --file "$ENV_FILE" "$1" 2>/dev/null || true
}

require_env_key() {
  local key="$1"
  local value
  value="$(read_env_key "$key")"
  if [[ -z "$value" ]]; then
    fail "Required env key missing or empty in $ENV_FILE: $key"
  fi
}

while [[ $# -gt 0 ]]; do
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
[[ -z "$KEEP_REVISION_COUNT" ]] && KEEP_REVISION_COUNT="1"
[[ -z "$KEEP_IMAGE_COUNT" ]] && KEEP_IMAGE_COUNT="3"
[[ -z "$ENABLE_APIS" ]] && ENABLE_APIS="true"
ENABLE_APIS="$(normalize_bool "$ENABLE_APIS" 'DEPLOY_ENABLE_APIS / --skip-enable-apis')"

if [[ -z "$PROJECT_ID" ]]; then
  usage >&2
  fail "No GCP project configured. Set GCP_PROJECT_ID in $ENV_FILE, use --project-id, or run: gcloud config set project YOUR_PROJECT_ID"
fi

if [[ ! "$SERVICE_NAME" =~ ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
  fail "Cloud Run service name must be 1-63 chars, lowercase, digits, or hyphen"
fi

validate_positive_int "$KEEP_REVISION_COUNT" "DEPLOY_KEEP_REVISIONS / --keep-revisions"
validate_positive_int "$KEEP_IMAGE_COUNT" "DEPLOY_KEEP_IMAGES / --keep-images"

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1)"
if [[ -z "$ACTIVE_ACCOUNT" ]]; then
  fail "No active gcloud account found. Run: gcloud auth login"
fi

require_env_key LINE_CHANNEL_SECRET
require_env_key LINE_CHANNEL_ACCESS_TOKEN
require_env_key OPENROUTER_API_KEY

SCHEDULED_MESSAGES_ENABLED_VALUE="$(read_env_key SCHEDULED_MESSAGES_ENABLED)"
[[ -z "$SCHEDULED_MESSAGES_ENABLED_VALUE" ]] && SCHEDULED_MESSAGES_ENABLED_VALUE="false"
SCHEDULED_MESSAGES_ENABLED_VALUE="$(normalize_bool "$SCHEDULED_MESSAGES_ENABLED_VALUE" 'SCHEDULED_MESSAGES_ENABLED')"

LINE_PUSH_FALLBACK_ENABLED_VALUE="$(read_env_key LINE_PUSH_FALLBACK_ENABLED)"
[[ -z "$LINE_PUSH_FALLBACK_ENABLED_VALUE" ]] && LINE_PUSH_FALLBACK_ENABLED_VALUE="true"
LINE_PUSH_FALLBACK_ENABLED_VALUE="$(normalize_bool "$LINE_PUSH_FALLBACK_ENABLED_VALUE" 'LINE_PUSH_FALLBACK_ENABLED')"

LINE_PUSH_MONTHLY_LIMIT_VALUE="$(read_env_key LINE_PUSH_MONTHLY_LIMIT)"
[[ -z "$LINE_PUSH_MONTHLY_LIMIT_VALUE" ]] && LINE_PUSH_MONTHLY_LIMIT_VALUE="0"
validate_non_negative_int "$LINE_PUSH_MONTHLY_LIMIT_VALUE" "LINE_PUSH_MONTHLY_LIMIT"

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

if [[ "$SCHEDULED_MESSAGES_ENABLED_VALUE" == "true" ]] && [[ -n "$SCHEDULED_GROUP_ID_VALUE" ]]; then
  if [[ "$LINE_PUSH_FALLBACK_ENABLED_VALUE" != "true" ]]; then
    fail "Scheduled messages are enabled but LINE push is disabled. Set LINE_PUSH_FALLBACK_ENABLED=true in $ENV_FILE."
  fi
fi

if [[ -z "$MIN_INSTANCE_COUNT" ]]; then
  if [[ "$SCHEDULED_MESSAGES_ENABLED_VALUE" == "true" ]] \
    && [[ -n "$SCHEDULED_GROUP_ID_VALUE" ]] \
    && [[ "$HAS_SCHEDULED_JOBS_CONFIG" == "true" ]]; then
    MIN_INSTANCE_COUNT="1"
  else
    MIN_INSTANCE_COUNT="0"
  fi
fi
validate_non_negative_int "$MIN_INSTANCE_COUNT" "auto-calculated Cloud Run min instances"

gcloud config set project "$PROJECT_ID" >/dev/null

if [[ -z "$IMAGE_TAG" ]]; then
  GIT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo manual)"
  TIMESTAMP="$(date -u +%Y%m%d%H%M%S)"
  IMAGE_TAG="${GIT_SHA}-${TIMESTAMP}"
fi

TEMP_ENV_YAML="$(ROOT_DIR="$ROOT_DIR" python3 - <<'PY'
from __future__ import annotations

import os
import tempfile

root_dir = os.environ["ROOT_DIR"]
fd, path = tempfile.mkstemp(
    prefix=".cloudrun-env.generated.",
    suffix=".yaml",
    dir=root_dir,
)
os.close(fd)
print(path)
PY
)"
TEMP_GCS_LIFECYCLE_JSON=""
cleanup() {
  rm -f "$TEMP_ENV_YAML"
  rm -f "$TEMP_GCS_LIFECYCLE_JSON"
}
trap cleanup EXIT

cloudrun_excludes=(
  GOOGLE_APPLICATION_CREDENTIALS
  PORT
  HOST
  GCP_PROJECT_ID
  CLOUD_RUN_SERVICE_NAME
  CLOUD_RUN_REGION
  CLOUD_RUN_MIN_INSTANCES
  CLOUD_RUN_SERVICE_ACCOUNT
  DEPLOY_KEEP_REVISIONS
  DEPLOY_KEEP_IMAGES
  DEPLOY_ENABLE_APIS
)

cloudrun_exclude_args=()
for key in "${cloudrun_excludes[@]}"; do
  cloudrun_exclude_args+=(--exclude "$key")
done

python3 "$ROOT_DIR/scripts/envfile.py" to-cloudrun-yaml \
  --file "$ENV_FILE" \
  "${cloudrun_exclude_args[@]}" \
  "$TEMP_ENV_YAML"

if [[ "$ENABLE_APIS" == "true" ]]; then
  echo "[step] Ensuring required GCP APIs are enabled"
  gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    iamcredentials.googleapis.com \
    cloudresourcemanager.googleapis.com \
    --project "$PROJECT_ID" >/dev/null
fi

substitutions=(
  "_SERVICE_NAME=$SERVICE_NAME"
  "_REGION=$REGION"
  "_MIN_INSTANCES=$MIN_INSTANCE_COUNT"
  "_IMAGE_TAG=$IMAGE_TAG"
  "_ENV_VARS_FILE=$(basename "$TEMP_ENV_YAML")"
  "_SKIP_TESTS=$SKIP_TESTS"
  "_SKIP_SMOKE=$SKIP_SMOKE"
  "_KEEP_REVISION_COUNT=$KEEP_REVISION_COUNT"
  "_KEEP_IMAGE_COUNT=$KEEP_IMAGE_COUNT"
)

SUBSTITUTIONS="$(IFS=,; echo "${substitutions[*]}")"

echo "[step] Submitting Cloud Build"
echo "       project=$PROJECT_ID service=$SERVICE_NAME region=$REGION image_tag=$IMAGE_TAG"
echo "       min_instances=$MIN_INSTANCE_COUNT"
echo "       cleanup=keep ${KEEP_REVISION_COUNT} revision(s), ${KEEP_IMAGE_COUNT} image digest(s)"
gcloud builds submit "$ROOT_DIR" \
  --project "$PROJECT_ID" \
  --config "$ROOT_DIR/cloudbuild.yaml" \
  --substitutions "$SUBSTITUTIONS"

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

if [[ -n "$GCS_BUCKET_NAME_VALUE" ]]; then
  if [[ -z "$RUNTIME_SERVICE_ACCOUNT" ]]; then
    fail "GCS_BUCKET_NAME is set but the Cloud Run runtime service account could not be determined"
  fi

  echo "[step] Ensuring runtime service account can sign GCS URLs"
  gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SERVICE_ACCOUNT" \
    --project "$PROJECT_ID" \
    --member="serviceAccount:$RUNTIME_SERVICE_ACCOUNT" \
    --role="roles/iam.serviceAccountTokenCreator" >/dev/null

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
  TEMP_GCS_LIFECYCLE_JSON="$(mktemp "${TMPDIR:-/tmp}/linebot-gcs-lifecycle.XXXXXX.json")"
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

echo "[step] Ensuring public Cloud Run access for LINE webhook"
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member="allUsers" \
  --role="roles/run.invoker" >/dev/null

if [[ "$SKIP_SMOKE" != "true" ]]; then
  echo "[step] Running post-deploy smoke checks"
  if ! wait_for_public_health "$SERVICE_URL"; then
    fail "Service did not become publicly reachable in time: $SERVICE_URL/health"
  fi

  WEBHOOK_SECRET="$(read_env_key LINE_CHANNEL_SECRET)"
  python3 "$ROOT_DIR/tests/smoke_test_cloud.py" \
    --base-url "$SERVICE_URL" \
    --expect-ready \
    --webhook-secret "$WEBHOOK_SECRET"
fi

echo "[done] Cloud deployment complete"
echo "       Active gcloud account: ${ACTIVE_ACCOUNT}"
echo "       Service URL: ${SERVICE_URL}"
if [[ -n "$LATEST_REVISION" ]]; then
  echo "       Latest revision: ${LATEST_REVISION}"
fi
echo "       Webhook URL: ${SERVICE_URL}/webhook"
