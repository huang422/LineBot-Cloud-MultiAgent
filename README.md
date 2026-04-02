# LineBot-Cloud-MultiAgent

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-GCP-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![LINE](https://img.shields.io/badge/LINE-Messaging%20API-00C300?logo=line&logoColor=white)](https://developers.line.biz/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Qwen3.5%20397B-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Nemotron%20120B-6366F1)](https://openrouter.ai/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-orange.svg?logo=buymeacoffee&logoColor=white)](https://www.buymeacoffee.com/huang422)


**A cloud-first, multi-agent LINE bot for GCP Cloud Run.**

LineBot-Cloud-MultiAgent receives LINE webhook events, routes them through a smart orchestrator, and dispatches them to specialized chat, vision, web-search, image-generation, and voice workflows while staying inside free-tier-friendly guardrails.

---

## Key Features

### Multi-Agent Orchestration

Fast regex rules handle obvious requests without an LLM call; harder cases fall back to model-based routing. The orchestrator dispatches to four specialist agents — `chat`, `vision`, `web_search`, and `image_gen` — each with its own model configuration and system prompt.

### Adaptive Thinking Control

The orchestrator does more than choose an agent: it also decides whether the downstream model should use provider-native reasoning / thinking mode for that specific request. **By default, requests run on the larger Qwen3.5 397B model without thinking for fast, high-quality responses.** Only clearly reasoning-heavy requests — for example step-by-step debugging, traceback / exception troubleshooting, proofs / derivations, deep planning, or full-length writing / translation — switch to the dedicated Qwen3.5 122B thinking model. If a thinking attempt runs too long, the fallback chain retries once with thinking disabled.

### Multi-Modal Input & Output

Text, images, quoted context, and voice replies all flow through the same webhook pipeline. Users can send photos for vision analysis, request image generation, receive voice replies, and quote earlier messages. Incoming audio messages are currently kept only as lightweight conversation context — speech-to-text is not implemented yet.

### Intelligent Provider Fallback

Every agent call goes through a rate-limit-aware fallback chain. Exhausted models are skipped before sending a request, so a `429` moves to the next provider without waiting for a failed full response cycle. Text agents try NVIDIA Qwen3.5 397B (or 122B when thinking) → OpenRouter Nemotron → `openrouter/free`; vision tries NVIDIA → Gemma 3.

### Image Generation with Description

User requests are first refined into optimized English prompts by the text LLM, then passed to NVIDIA Stable Diffusion 3. Prompt refinement and user-facing description generation run in parallel for lower latency. The image is uploaded to GCS and delivered as a LINE image message alongside a text description.

### Voice Replies

`edge-tts` synthesizes speech at zero cost. Audio is uploaded to GCS with signed URLs and sent as a LINE audio message. Voice and language are configurable.

### Cloud-Only, One-Command Deploy

`./scripts/deploy_cloud_run.sh` handles everything — API enablement, Cloud Build submission, smoke checks, revision routing, and old image/revision pruning. No local Docker build step required.

### Config-First Behavior

System prompts live in `prompts/*.md`, and most runtime tuning lives in `.env`. Models, temperatures, token limits, rate limits, cost controls, and scheduling are configurable without touching code, while some fast-routing, safety, and output heuristics intentionally stay in Python for deterministic behavior.

---

## What it does

| Feature | How it works |
| --- | --- |
| Chat | General conversation, coding help, translation, and creative replies via the fallback chain |
| Adaptive thinking control | The orchestrator decides per request whether to switch to the dedicated thinking model; most requests stay on non-thinking 397B, while only clearly reasoning-heavy tasks activate the 122B model |
| Vision | Accepts images, screenshots, and photos for analysis |
| Web search | User queries are optimized into search keywords, then Tavily search results are fetched with topic/time/country hints and injected into the prompt before synthesis |
| Webpage reading | If the user posts a URL, Tavily Extract fetches the page body and injects the webpage content into the prompt before synthesis |
| Image generation | Prompt refinement and description generation run in parallel, then NVIDIA Stable Diffusion 3 generates the image; both image and text description are delivered together |
| Voice reply | `edge-tts` produces audio, uploads it to GCS, and sends a LINE audio reply |
| Quoted context | Reply to an earlier text or image and recover the original content from local cache |
| Scheduled messages | APScheduler can send recurring group reminders and yearly birthday messages |
| Simplified-to-traditional conversion | All LLM output is normalized through OpenCC before being sent back to LINE |
| Prompt-injection blocking | Regex-based screening rejects common jailbreak patterns before dispatch |
| Health dashboard | `/health` exposes readiness, provider status, cost-control state, and agent call counters |

---

## From local GPU to cloud-native

This project is the next-generation successor to [LineBot-VLM-GroupAgent](https://github.com/huang422/LineBot-VLM-GroupAgent), which ran a single Ollama model on a local NVIDIA GPU. LineBot-Cloud-MultiAgent has been redesigned from scratch for zero-hardware, cloud-only deployment.

| | [LineBot-VLM-GroupAgent](https://github.com/huang422/LineBot-VLM-GroupAgent) (v1) | LineBot-Cloud-MultiAgent (v2) |
| --- | --- | --- |
| Deployment | Local server + Cloudflare Tunnel | GCP Cloud Run (serverless) |
| Hardware | NVIDIA GPU + 32 GB RAM required | No GPU, no local hardware |
| LLM Provider | Ollama (single local model) | NVIDIA + OpenRouter (multi-provider fallback) |
| Architecture | Single model, serial queue | Multi-agent orchestrator with parallel dispatch |
| Vision Model | Qwen3.5 9B/35B (local) | Qwen3.5 397B VLM (NVIDIA API) |
| Image Generation | Not supported | NVIDIA Stable Diffusion 3 (two-stage pipeline) |
| Voice Reply | Not supported | edge-tts + GCS signed URL |
| Reasoning | Ollama thinking mode (may OOM) | Provider-native thinking with token budget control |
| Resilience | Single point of failure | Auto-fallback across providers and models |
| Cost | Electricity + hardware | Free-tier APIs + pay-per-use Cloud Run |
| Deploy Command | Manual setup | One-command `./scripts/deploy_cloud_run.sh` |

---

## Architecture at a glance

1. `POST /webhook` receives a LINE event and validates the HMAC signature.
2. The API returns `200 OK` immediately, then continues processing in background tasks.
3. Input processing downloads image payloads, sanitizes text, and recovers quoted context.
4. Rate limiting and conversation history enrichment happen before routing.
5. The orchestrator chooses `chat`, `vision`, `web_search`, or `image_gen`, and also decides whether the request needs deep thinking.
6. The selected agent runs through the fallback chain and returns a normalized response.
7. The output processor sends text, voice, or image messages back to LINE.

### Provider fallback order

- **Orchestrator**: OpenRouter `ORCHESTRATOR_MODEL`, then NVIDIA `ORCHESTRATOR_FALLBACK_MODEL` by default
- **Text-centric agents (`chat`, `web_search`, image prompt refinement)**: NVIDIA `NVIDIA_MODEL` (or `NVIDIA_THINKING_MODEL` when thinking), then OpenRouter `AGENT_FALLBACK_MODEL`, then `openrouter/free`
- **Vision**: NVIDIA `NVIDIA_MODEL`, then `VISION_FALLBACK_MODEL`

The rate tracker skips exhausted models before sending a request, so a `429` can move to the next provider without waiting for a failed full response cycle.

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Runtime | GCP Cloud Run |
| Framework | FastAPI + Uvicorn |
| Text LLM (default) | NVIDIA Qwen3.5 397B (non-thinking) |
| Text reasoning LLM | NVIDIA Qwen3.5 122B (thinking) / OpenRouter Nemotron 120B |
| Vision LLM | NVIDIA Qwen3.5 397B (thinking: 122B) |
| Fallback LLMs | Nemotron 120B / Gemma 3 27B / `openrouter/free` (task-dependent) |
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
LineBot-Cloud-MultiAgent/
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

## Quick start

```bash
# 1. Copy and fill in your credentials
cp .env.example .env

# 2. First-time GCP login
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Deploy (one command does everything)
./scripts/deploy_cloud_run.sh

# 4. Set your LINE webhook URL to:
#    https://YOUR_CLOUD_RUN_URL/webhook
```

Minimum `.env` values: `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN`, `OPENROUTER_API_KEY`, `GCP_PROJECT_ID`

If you want the bot to **deliver generated images or voice replies back to LINE**, also set `GCS_BUCKET_NAME`. Without GCS, text chat still works, but media delivery falls back to text-only responses.

Redeploy after code changes — same command: `./scripts/deploy_cloud_run.sh`

For detailed deployment options, GCS/scheduler setup, log monitoring, and troubleshooting, see **[DEPLOY.md](DEPLOY.md)**.

---

## Current runtime notes

- `OPENROUTER_API_KEY` is part of the minimum setup because `/health` readiness and orchestrator primary routing depend on it.
- Incoming audio is **not** transcribed yet. The bot currently supports **voice replies only**, not speech-to-text input.
- Conversation history, quoted-message cache, LINE push counters, and model rate-tracker state are **in-memory per instance**. They reset on redeploy and are not shared across multiple Cloud Run instances.
- `WEB_SEARCH_MONTHLY_QUOTA` is kept only for backward compatibility. The app no longer enforces a local search quota; actual availability depends on Tavily credentials and provider-side limits.
- Relative search phrases like `today`, `latest` and `recent` are handled through keyword extraction plus Tavily topic/time-range targeting rather than by prefixing the raw query with a timestamp.

---

## Configuration reference

All settings are loaded from `.env`. See [.env.example](.env.example) for a ready-to-copy template.

### Required

| Variable | Description |
| --- | --- |
| `LINE_CHANNEL_SECRET` | LINE Messaging API channel secret |
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Messaging API access token |
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/keys) API key (free tier available) |
| `GCP_PROJECT_ID` | GCP project ID for Cloud Run deployment |

### Recommended

| Variable | Default | Description |
| --- | --- | --- |
| `NVIDIA_API_KEY` | — | [NVIDIA](https://build.nvidia.com) API key; enables Qwen3.5 397B/122B + image generation |
| `NVIDIA_MODEL` | `qwen/qwen3.5-397b-a17b` | NVIDIA primary model (non-thinking) |
| `NVIDIA_THINKING_MODEL` | `qwen/qwen3.5-122b-a10b` | NVIDIA thinking model (auto-switched when deep reasoning needed) |
| `TAVILY_API_KEY` | — | [Tavily](https://tavily.com) key for web search + URL extraction |
| `GCS_BUCKET_NAME` | — | GCS bucket for voice/image delivery; required if you want generated images or TTS audio to reach LINE |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Optional local-only GCP SDK credential JSON path; leave empty on Cloud Run |

### Reasoning / Thinking

| Variable | Default | Description |
| --- | --- | --- |
| `OPENROUTER_REASONING_ENABLED` | `true` | Enable reasoning for OpenRouter models |
| `OPENROUTER_REASONING_EFFORT` | `high` | Reasoning effort: `xhigh` / `high` / `medium` / `low` / `minimal` / `none` |
| `OPENROUTER_REASONING_EXCLUDE` | `false` | Exclude reasoning from response (free models may ignore this) |
| `OPENROUTER_THINKING_BUDGET` | `4096` | Extra completion tokens reserved for OpenRouter reasoning |
| `NVIDIA_THINKING_ENABLED` | `true` | Enable thinking for NVIDIA models |
| `NVIDIA_THINKING_BUDGET` | `4096` | Extra tokens reserved for model thinking |
| `THINKING_TIMEOUT_SECONDS` | `150` | Abort a reasoning attempt after N seconds and retry once with thinking disabled; set `0` to disable |
| `REQUIRE_REASONING_MODELS` | `true` | Only route to reasoning-capable models |
| `REQUIRE_REASONING_TOKENS` | `true` | Warn when a reasoning model returns no reasoning content |

These flags control provider-side reasoning support, but the actual on/off decision is made per request by the orchestrator. **By default, all requests use the larger 397B model without thinking.** Only clearly reasoning-heavy tasks — such as step-by-step debugging, traceback / exception troubleshooting, proofs / derivations, deep planning, or full-length writing / translation — switch to the dedicated 122B thinking model. When thinking mode activates, the provider automatically swaps from `NVIDIA_MODEL` to `NVIDIA_THINKING_MODEL`.

### Model routing

| Variable | Default | Description |
| --- | --- | --- |
| `ORCHESTRATOR_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Primary orchestrator model (OpenRouter) |
| `ORCHESTRATOR_FALLBACK_MODEL` | `qwen/qwen3.5-122b-a10b` | Orchestrator NVIDIA fallback target (with default settings, this runs in thinking mode) |
| `ORCHESTRATOR_TEMPERATURE` | `0` | Sampling temperature for LLM-based routing |
| `ORCHESTRATOR_MAX_TOKENS` | `384` | Max tokens for orchestrator JSON routing output |
| `AGENT_FALLBACK_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Shared fallback for text agents |
| `VISION_FALLBACK_MODEL` | `google/gemma-3-27b-it:free` | Vision agent fallback |

### Per-agent tuning

| Variable | Default | Description |
| --- | --- | --- |
| `CHAT_TEMPERATURE` / `CHAT_MAX_TOKENS` | `0.7` / `2048` | Chat agent |
| `VISION_TEMPERATURE` / `VISION_MAX_TOKENS` | `0.5` / `1024` | Vision agent |
| `WEB_SEARCH_TEMPERATURE` / `WEB_SEARCH_MAX_TOKENS` | `0.2` / `3072` | Web search agent |
| `IMAGE_GEN_TEMPERATURE` / `IMAGE_GEN_MAX_TOKENS` | `0.7` / `1024` | Image gen prompt refinement |

### Image generation (Stage 2)

| Variable | Default | Description |
| --- | --- | --- |
| `IMAGE_GEN_PRIMARY_MODEL` | `stabilityai/stable-diffusion-3-medium` | Primary NVIDIA image model |
| `IMAGE_GEN_FALLBACK_MODEL` | — | Fallback image model |
| `IMAGE_GEN_STEPS` | `50` | Diffusion steps |
| `IMAGE_GEN_CFG_SCALE` | `5` | Classifier-free guidance scale |

### Cost controls

| Variable | Default | Description |
| --- | --- | --- |
| `LINE_PUSH_FALLBACK_ENABLED` | `true` | Allow push fallback when reply fails |
| `LINE_PUSH_MONTHLY_LIMIT` | `0` | Monthly push cap (`0` = unlimited) |
| `WEB_SEARCH_MONTHLY_QUOTA` | `1000` | Legacy compatibility setting; app-side web-search quota is not enforced, actual limits come from Tavily |
| `GCS_SIGNED_URL_EXPIRY_HOURS` | `48` | Signed URL expiry |
| `GCS_MEDIA_CLEANUP_DELAY_SECONDS` | `172800` | App-level media cleanup delay (2 days) |
| `RATE_LIMIT_MAX_REQUESTS` | `30` | Per-user request limit |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit sliding window |

### Bot & conversation

| Variable | Default | Description |
| --- | --- | --- |
| `BOT_NAME` | `Assistant` | Bot display name in prompts |
| `LINE_BOT_USER_ID` | — | Bot's LINE userId for precise @mention detection |
| `MAX_CONVERSATION_HISTORY` | `10` | In-memory messages kept per conversation |
| `CONVERSATION_TTL_SECONDS` | `3600` | In-memory conversation expiry (1 hour, resets on redeploy) |
| `TTS_ENABLED` | `true` | Enable voice replies (incoming audio transcription is not included) |
| `TTS_VOICE` | `zh-TW-HsiaoChenNeural` | [edge-tts](https://github.com/rany2/edge-tts) voice |

### Scheduled messages

| Variable | Default | Description |
| --- | --- | --- |
| `SCHEDULED_MESSAGES_ENABLED` | `false` | Enable scheduled message delivery |
| `SCHEDULED_GROUP_ID` | — | Target LINE group ID |
| `SCHEDULED_WEEKLY_MESSAGES` | `[]` | Weekly jobs (JSON array) |
| `SCHEDULED_YEARLY_MESSAGES` | `[]` | Yearly jobs (JSON array) |

### Deployment

| Variable | Default | Description |
| --- | --- | --- |
| `CLOUD_RUN_SERVICE_NAME` | `linebot-cloud-agent` | Cloud Run service name |
| `CLOUD_RUN_REGION` | `us-west1` | Cloud Run region |
| `DEPLOY_KEEP_REVISIONS` | `1` | Revisions to keep after deploy |
| `DEPLOY_KEEP_IMAGES` | `3` | Container images to keep |
| `DEPLOY_ENABLE_APIS` | `true` | Auto-enable required GCP APIs |

---

## License

GNU General Public License v3 (GPLv3) — see [LICENSE](LICENSE).

## Contact

- Developer: Tom Huang
- Email: huang1473690@gmail.com
