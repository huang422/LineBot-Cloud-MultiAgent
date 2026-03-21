# LineBot-CloudAgent

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-GCP-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![LINE](https://img.shields.io/badge/LINE-Messaging%20API-00C300?logo=line&logoColor=white)](https://developers.line.biz/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Qwen3.5%20397B-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Free%20Tier-6366F1)](https://openrouter.ai/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/huang422)

**A cloud-first, multi-agent LINE bot for GCP Cloud Run.**

LineBot-CloudAgent receives LINE webhook events, routes them through a smart orchestrator, and dispatches them to specialized chat, vision, web-search, image-generation, and voice workflows while staying inside free-tier-friendly guardrails.

---

## Why this project stands out

- **Multi-agent orchestration** — Fast regex rules handle obvious requests without an LLM call; harder cases fall back to model-based routing.
- **Task-specific model routing** — The orchestrator defaults to OpenRouter first with an NVIDIA fallback, while text and vision agents keep NVIDIA Qwen3.5 as the primary model when available.
- **Multi-modal I/O** — Text, quoted context, images, generated images, and voice replies all move through the same webhook pipeline.
- **Cloud-only deploy path** — The included deploy wrapper submits Cloud Build; you do not need a local Docker build step.
- **Repeatable redeploys** — The pipeline updates the same Cloud Run service, routes 100% traffic to the latest revision, runs smoke checks, prunes old revisions on a best-effort basis, and keeps the latest 3 container images.
- **Config-first behavior** — Prompts live in `prompts/*.md`; runtime behavior lives in `.env` rather than hard-coded constants.

---

## What it does

| Feature | How it works |
| --- | --- |
| Chat | General conversation, coding help, translation, and creative replies via the fallback chain |
| Vision | Accepts images, screenshots, and photos for analysis |
| Web search | Tavily search results are injected into the prompt before synthesis |
| Webpage reading | If the user posts a URL, Tavily Extract fetches the page body and injects the webpage content into the prompt before synthesis |
| Image generation | Stage 1 prompt refinement through the text-agent reasoning chain, then NVIDIA Stable Diffusion 3 image generation via the NVIDIA API |
| Voice reply | `edge-tts` produces audio, uploads it to GCS, and sends a LINE audio reply |
| Quoted context | Reply to an earlier text or image and recover the original content from local cache |
| Scheduled messages | APScheduler can send recurring group reminders and yearly birthday messages |
| Simplified-to-traditional conversion | All LLM output is normalized through OpenCC before being sent back to LINE |
| Prompt-injection blocking | Regex-based screening rejects common jailbreak patterns before dispatch |
| Health dashboard | `/health` exposes readiness, provider status, quotas, and agent call counters |

---

## Architecture at a glance

1. `POST /webhook` receives a LINE event and validates the HMAC signature.
2. The API returns `200 OK` immediately, then continues processing in background tasks.
3. Input processing downloads image payloads, sanitizes text, and recovers quoted context.
4. Rate limiting and conversation history enrichment happen before routing.
5. The orchestrator chooses `chat`, `vision`, `web_search`, or `image_gen`.
6. The selected agent runs through the fallback chain and returns a normalized response.
7. The output processor sends text, voice, or image messages back to LINE.

### Provider fallback order

- **Orchestrator**: OpenRouter `ORCHESTRATOR_MODEL`, then NVIDIA `ORCHESTRATOR_FALLBACK_MODEL` by default
- **Text-centric agents (`chat`, `web_search`, image prompt refinement)**: NVIDIA `NVIDIA_MODEL`, then OpenRouter `AGENT_FALLBACK_MODEL`, then `openrouter/free`
- **Vision**: NVIDIA `NVIDIA_MODEL`, then `VISION_FALLBACK_MODEL`

The rate tracker skips exhausted models before sending a request, so a `429` can move to the next provider without waiting for a failed full response cycle.

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Runtime | GCP Cloud Run |
| Framework | FastAPI + Uvicorn |
| Text reasoning LLM | OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` |
| Vision LLM | NVIDIA Qwen3.5 397B |
| Fallback LLMs | Trinity / Gemma / `openrouter/free` (task-dependent) |
| Image generation | NVIDIA Stable Diffusion 3 (`stabilityai/stable-diffusion-3-medium`) |
| Web search | Tavily |
| Voice | `edge-tts` |
| Object storage | Google Cloud Storage |
| Scheduling | APScheduler |
| Text conversion | OpenCC |
| Image processing | Pillow |
| Delivery pipeline | Cloud Build -> Artifact Registry -> Cloud Run |

---

## Repository layout

```text
LineBot-CloudAgent/
├── main.py
├── Dockerfile
├── cloudbuild.yaml
├── .gcloudignore
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── prompts/
├── scripts/
│   ├── deploy_cloud_run.sh
│   └── envfile.py
├── src/
└── tests/
```

---

## Cloud deployment

### 1. Prepare configuration

```bash
cp .env.example .env
```

Minimum required values:

- `LINE_CHANNEL_SECRET`
- `LINE_CHANNEL_ACCESS_TOKEN`
- `OPENROUTER_API_KEY`
- `GCP_PROJECT_ID`

Recommended deploy values:

- `CLOUD_RUN_SERVICE_NAME=linebot-cloud-agent`
- `CLOUD_RUN_REGION=us-west1`
- `CLOUD_RUN_MIN_INSTANCES=` (leave blank for auto: `1` when scheduled jobs are configured, otherwise `0`)
- `DEPLOY_KEEP_REVISIONS=1`
- `DEPLOY_KEEP_IMAGES=3`
- `DEPLOY_ENABLE_APIS=true`

Common runtime optional values:

- `NVIDIA_API_KEY` (required for image generation; recommended for chat/vision quality)
- `TAVILY_API_KEY`
- `GCS_BUCKET_NAME`
- `GCS_SIGNED_URL_EXPIRY_HOURS` (default: `48`)
- `GCS_MEDIA_CLEANUP_DELAY_SECONDS` (default: `172800`, i.e. 2 days)
- `LINE_PUSH_MONTHLY_LIMIT`
- `SCHEDULED_MESSAGES_ENABLED`
- `SCHEDULED_GROUP_ID`
- `SCHEDULED_WEEKLY_MESSAGES`
- `SCHEDULED_YEARLY_MESSAGES`

If `SCHEDULED_MESSAGES_ENABLED=true`, keep `LINE_PUSH_FALLBACK_ENABLED=true`, set `SCHEDULED_GROUP_ID`, and configure at least one entry in `SCHEDULED_WEEKLY_MESSAGES` or `SCHEDULED_YEARLY_MESSAGES`. Set `LINE_PUSH_MONTHLY_LIMIT=0` for uncapped scheduled pushes, or use a value greater than `0` if you want a monthly direct-push cap.

### 2. One-time GCP setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com
```

If you want Cloud Run to access other GCP services through a dedicated runtime identity, set `CLOUD_RUN_SERVICE_ACCOUNT` in `.env` or pass `--service-account` during deployment.

### 3. Deploy

```bash
./scripts/deploy_cloud_run.sh
```

The script reads `GCP_PROJECT_ID`, `CLOUD_RUN_SERVICE_NAME`, and `CLOUD_RUN_REGION` from `.env`. You can still override them with CLI flags when needed.

What the wrapper does:

1. Reads `.env` and validates the required keys.
2. Ensures the required GCP APIs are enabled.
3. Generates a temporary Cloud Run env-vars file.
4. Submits `cloudbuild.yaml` to Cloud Build.
5. Runs `pytest` and `compileall` in Cloud Build before deployment.
6. Builds and pushes a uniquely tagged container image.
7. Deploys to the same Cloud Run service and routes 100% traffic to the latest revision.
8. If `GCS_BUCKET_NAME` is configured, the deploy flow ensures the runtime service account has `roles/iam.serviceAccountTokenCreator` on itself so voice/image signed URLs work on Cloud Run, and it verifies a 3-day bucket lifecycle delete rule as a safety net.
9. Runs smoke checks against `/health` and `/webhook`.
10. Keeps the newest 1 revision by default and the newest 3 image digests by default.

### 4. Set the LINE webhook URL

```text
https://YOUR_CLOUD_RUN_URL/webhook
```

---

## Redeploy after code changes

Use the same command again:

```bash
./scripts/deploy_cloud_run.sh
```

Because deployment settings live in `.env`, the same zero-argument command works for both first deploys and later redeploys after code changes.

If you need a one-off override, you can still pass flags such as `--project-id`, `--region`, or `--service-account`.

### What gets replaced or cleaned

- The same Cloud Run **service** is updated on every deploy.
- The included pipeline routes **100% traffic** to the newest ready revision.
- Old **container images** are trimmed by policy; the script keeps the latest 3 digests by default.
- Old **revisions** are trimmed by policy; the script keeps the latest ready deployment footprint small by default.

### What is not removed automatically

- Cloud Build history
- Cloud Logging entries
- Any manual resources you created outside the included deployment flow

If you deploy manually through the Cloud Console or plain `gcloud run deploy`, Cloud Run will keep revision history until you delete it yourself or the platform lifecycle removes older revisions.

---

## Direct Cloud Build submission

If you do not want to use the wrapper script, you can submit Cloud Build directly:

```bash
gcloud builds submit . \
  --project YOUR_PROJECT_ID \
  --config cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=linebot-cloud-agent,_REGION=us-west1,_IMAGE_TAG=manual-$(date -u +%Y%m%d%H%M%S)
```

Use this mode only when the target Cloud Run service already has its environment variables configured, or when you provide a generated env-vars file through `_ENV_VARS_FILE`.

---

## Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Readiness, provider availability, quotas, and agent call counters |
| `/webhook` | `POST` | LINE webhook receiver with HMAC-SHA256 validation |

---

## Cost guardrails

| Resource | Control mechanism |
| --- | --- |
| LLM calls | Per-model RPM/RPD tracking and automatic fallback on `429` |
| LINE push | Monthly push counter controlled by `LINE_PUSH_MONTHLY_LIMIT` |
| Web search | Monthly quota counter controlled by `WEB_SEARCH_MONTHLY_QUOTA` |
| GCS media | Signed URLs expire after 48 hours, app cleanup runs after 2 days by default, and deploy verifies a 3-day bucket lifecycle safety net |
| User requests | Per-user sliding-window rate limit |
| Cloud Run | `max-instances=1`, `min-instances` auto-derived (`0` normally, `1` for in-process scheduled jobs unless overridden) |

---

## Operational notes

- `/health` can return `200` even when `ready_for_webhook=false`; check the payload, not only the HTTP status.
- Voice and generated-image delivery require `GCS_BUCKET_NAME`. Without it, text replies still work. With the defaults, signed URLs expire after 48 hours, app cleanup waits 2 days, and the deploy flow verifies a 3-day bucket lifecycle fallback.
- Scheduled messages require `SCHEDULED_MESSAGES_ENABLED=true`, `LINE_PUSH_FALLBACK_ENABLED=true`, a `SCHEDULED_GROUP_ID`, and at least one configured scheduled job.
- Prompts are loaded from `prompts/*.md`, so you can change routing or tone without editing Python source.

---

## License

GNU General Public License v3 (GPLv3) — see [LICENSE](LICENSE).

## Contact

- Developer: Tom Huang
- Email: huang1473690@gmail.com
