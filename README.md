# LineBot-Cloud-MultiAgent

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-GCP-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![LINE](https://img.shields.io/badge/LINE-Messaging%20API-00C300?logo=line&logoColor=white)](https://developers.line.biz/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Qwen3.5%20397B-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Gemma%204%2031B-6366F1)](https://openrouter.ai/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-orange.svg?logo=buymeacoffee&logoColor=white)](https://www.buymeacoffee.com/huang422)


**A cloud-first, multi-agent LINE bot for GCP Cloud Run.**

LineBot-Cloud-MultiAgent receives LINE webhook events, routes them through a smart orchestrator, and dispatches them to specialized chat, vision, web-search, image-generation, and voice workflows while staying inside free-tier-friendly guardrails.

---

## Key Features

### Multi-Agent Orchestration

Fast regex rules handle obvious requests without an LLM call; harder cases fall back to model-based routing. The orchestrator dispatches to four specialist agents — `chat`, `vision`, `web_search`, and `image_gen` — each with its own model configuration and system prompt. For short follow-up turns, the router now also carries forward the previous routed agent / task state so continuation messages stay anchored to the prior context instead of relying on raw chat history alone.

### Adaptive Thinking Control

The orchestrator does more than choose an agent: it also decides whether the downstream model should use provider-native reasoning / thinking mode for that specific request. **The orchestrator's own JSON routing stays in non-thinking mode for stability.** By default, downstream requests run on the larger Qwen3.5 397B model without thinking for fast, high-quality responses. Only clearly reasoning-heavy requests — for example step-by-step debugging, traceback / exception troubleshooting, proofs / derivations, deep planning, or full-length writing / translation — switch to the dedicated NVIDIA Gemma 4 31B thinking model. If a downstream thinking attempt runs too long, the fallback chain retries once with thinking disabled.

### Multi-Modal Input & Output

Text, images, quoted context, and voice replies all flow through the same webhook pipeline. Users can send photos for vision analysis, request image generation, receive voice replies, and quote earlier messages. Incoming audio messages are currently kept only as lightweight conversation context — speech-to-text is not implemented yet.

### Intelligent Provider Fallback

Every agent call goes through a rate-limit-aware fallback chain. Exhausted models are skipped before sending a request, so a `429` moves to the next provider without waiting for a failed full response cycle. Text agents try NVIDIA Qwen3.5 397B (or NVIDIA Gemma 4 31B when thinking) → OpenRouter Nemotron → `openrouter/free`; vision uses NVIDIA Qwen3.5 397B and falls back to OpenRouter Gemma 4 31B.

### Image Generation with Description

User requests are first refined into optimized English prompts by the text LLM, then passed to NVIDIA FLUX.1-dev through the existing NVIDIA Visual GenAI NIM API. Prompt refinement and user-facing description generation run in parallel for lower latency. The image is uploaded to GCS and delivered as a LINE image message alongside a text description.

### Voice Replies

`edge-tts` synthesizes speech at zero cost. Audio is uploaded to GCS with signed URLs and sent as a LINE audio message. Voice and language are configurable.

### Layered Chat Memory

Each `user` / `group` / `room` chat keeps a rolling **6-message recent text window** plus **one long-term summary**. When the recent window fills up, the app uses NVIDIA Qwen3.5 397B to compact `existing summary + latest 6 messages` into the next summary in the background, with a bias toward preserving still-valid long-term facts and preferences unless newer dialogue clearly invalidates them. Sending `!new` clears the current chat scope's recent window, summary, and episodic recall records, then replies with `Let's start a new chat!`.

For groups and rooms, non-triggered user text can also be buffered and batch-persisted into memory so a later `!hej` / @mention turn can still see the surrounding thread without forcing a write for every single message.

Memory is stored through a pluggable backend layer:

- **`InMemoryBackend`** (default) — fast, no external dependencies, but resets on Cloud Run scale-to-zero. Suitable for local dev and tests.
- **`FirestoreBackend`** (opt-in) — persists `chat_memory/{source_type}::{chat_id}` and `user_profile/{user_id}` documents in Firestore Native, surviving cold starts and multi-instance deployments. Stays inside the Firestore free tier (50K reads / 20K writes / 1 GiB per day) when the database is co-located with Cloud Run (e.g. `us-west1`). Enable by setting `FIRESTORE_ENABLED=true` and `FIRESTORE_PROJECT_ID=<gcp-project>` (see `.env.example`).

In addition to the chat-scoped memory, the bot maintains a **per-user profile** (`user_profile/{user_id}`) that is shared across DMs and groups. Phase A populates it with the LINE `displayName` (resolved via `/v2/bot/profile/{userId}` or `/v2/bot/{group|room}/{id}/member/{userId}` depending on context) and `last_seen` metadata. Phase B lets the LLM actively extend the profile via the `update_user_profile` tool when the user volunteers durable facts (nickname, hobbies, location). `!new` intentionally does not delete this cross-chat profile; remove that document manually from Firestore if a user needs a full profile reset.

### Native Tool Calling for Chat Agent

The chat agent now ships with an OpenAI-style **agentic tool loop**. While a normal reply is being generated, the LLM can decide to invoke tools and re-enter the loop with the results — no orchestrator round-trip required:

- **`recall_memory(query, k)`** — vector-search older episodic snapshots stored in Firestore (`{prefix}_episodes/...`). Episodes are written automatically on every summary rotation using free NVIDIA `nv-embedqa-e5-v5` embeddings (1024-dim). When the user references something older than the recent window, the model can pull just the relevant snippet on demand instead of carrying everything in the prompt.
- **`update_user_profile(facts, confidence)`** — persists durable user facts into `user_profile/{user_id}`. Confidence-gated (>= 0.7) and capped at 20 facts.
- **`web_search(query)`** — fallback web search via Tavily when the orchestrator's fast-rules did not pre-route to the dedicated `web_search` agent (dual-track design). Fast-rule routing for news / URLs / finance still wins by default, but the LLM can also pull live data on its own.

The tool loop falls back to a plain `fallback_chain.generate()` call on any error so users always get a reply, and the iteration count is hard-capped by `tool_loop_max_iterations` (default 4) to keep request budgets bounded.

### Cloud-Only, One-Command Deploy

`./scripts/deploy_cloud_run.sh` handles everything — API enablement, Cloud Build submission, smoke checks, revision routing, and old image/revision pruning. No local Docker build step required.

### Config-First Behavior

System prompts live in `prompts/*.md`, and most runtime tuning lives in `.env`. Models, temperatures, token limits, rate limits, cost controls, and scheduling are configurable without touching code, while some fast-routing, safety, and output heuristics intentionally stay in Python for deterministic behavior.

---

## What it does

| Feature | How it works |
| --- | --- |
| Chat | General conversation, coding help, translation, and creative replies via the fallback chain |
| Adaptive thinking control | The orchestrator decides per request whether to switch to the dedicated thinking model; most requests stay on non-thinking 397B, while only clearly reasoning-heavy tasks activate NVIDIA Gemma 4 31B |
| Vision | Accepts images, screenshots, and photos for analysis |
| Web search | User queries are optimized into search keywords, then Tavily search results are fetched with topic/time/country hints and injected into the prompt before synthesis |
| Webpage reading | If the user posts a URL, Tavily Extract fetches the page body and injects the webpage content into the prompt before synthesis |
| Image generation | Prompt refinement and description generation run in parallel, then NVIDIA FLUX.1-dev generates the image; both image and text description are delivered together |
| Voice reply | `edge-tts` produces audio, uploads it to GCS, and sends a LINE audio reply |
| Quoted context | Reply to an earlier text or image and recover the original content from local cache |
| Scheduled messages | Cloud Scheduler pings `/internal/cron` every minute; the app fires any weekly / yearly LINE push reminders whose cron expression matches the current Asia/Taipei minute |
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
| Image Generation | Not supported | NVIDIA FLUX.1-dev (two-stage pipeline) |
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
4. Rate limiting and chat-memory enrichment happen before routing.
5. The orchestrator chooses `chat`, `vision`, `web_search`, or `image_gen`, and also decides whether the request needs deep thinking.
6. The selected agent runs through the fallback chain and returns a normalized response.
7. The output processor sends text, voice, or image messages back to LINE.

### Provider fallback order

- **Orchestrator**: OpenRouter `ORCHESTRATOR_MODEL`, then NVIDIA `ORCHESTRATOR_FALLBACK_MODEL` by default
- **Text-centric agents (`chat`, `web_search`, image prompt refinement)**: NVIDIA `NVIDIA_MODEL` (or `NVIDIA_THINKING_MODEL` when thinking), then OpenRouter `AGENT_FALLBACK_MODEL`, then `openrouter/free`
- **Vision**: NVIDIA `NVIDIA_MODEL` (or `NVIDIA_THINKING_MODEL` when the primary NVIDIA leg needs thinking), then `VISION_FALLBACK_MODEL`

The rate tracker skips exhausted models before sending a request, so a `429` can move to the next provider without waiting for a failed full response cycle.

---

## System design

This section is the deep-dive companion to the quick overview above. It is intended to be read top-to-bottom as a walkthrough of how a single LINE message becomes a reply, including every model, fallback hop, and storage touchpoint.

### 1. System architecture overview

A serverless, event-driven design. Cloud Run hosts a single FastAPI app that fans out to external AI providers and stateless storage. Nothing runs when no traffic arrives (scale-to-zero), which keeps the system inside free-tier budgets.

```text
┌────────────────────────────────────────────────────────────────┐
│ CLIENT  -  LINE user · group · room                            │
└────────────────────────────────────────────────────────────────┘
                                 ↓  text / image / quoted message / voice request
┌────────────────────────────────────────────────────────────────┐
│ LINE MESSAGING API                                             │
│ webhook delivery  +  reply (free) / push (fallback)            │
└────────────────────────────────────────────────────────────────┘
                                 ↓  POST /webhook   (+ X-Line-Signature)
┌────────────────────────────────────────────────────────────────┐
│ GCP CLOUD RUN  -  FastAPI app (main.py)   [scale-to-zero]      │
├────────────────────────────────────────────────────────────────┤
│ Endpoints:  /webhook    /health    /internal/cron              │
│                                                                │
│ Async pipeline (HTTP returns 200 OK before any model call):    │
│   Input -> Enrich -> Orchestrator -> Agent x4 -> Output        │
│   proc.   rate+mem   (router)                   proc.          │
│                                                                │
│ Core engine:  FallbackChain + RateTracker                      │
└────────────────────────────────────────────────────────────────┘
                                 ↓  fan-out to external services
┌────────────────────────────────────────────────────────────────┐
│ EXTERNAL AI PROVIDERS                                          │
├────────────────────────────────────────────────────────────────┤
│ NVIDIA API      LLM · vision · image gen · embeddings          │
│ OpenRouter API  LLM + vision fallback                          │
│ Tavily          web search + URL extract                       │
│ edge-tts        voice synthesis (TTS)                          │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ STATE & STORAGE                                                │
├────────────────────────────────────────────────────────────────┤
│ Firestore Native  chat memory · profile · episodes             │
│ Cloud Storage     images · audio (signed URLs)                 │
│ In-memory/inst.   quoted msgs · rate state · push count        │
└────────────────────────────────────────────────────────────────┘

Cloud Scheduler ──(1 call / min · X-Cron-Token)──→ POST /internal/cron
```

| Component | Responsibility | Key implementation |
| --- | --- | --- |
| FastAPI endpoints | Webhook intake, health, cron trigger | [main.py](main.py) |
| Input processor | Decode event, download image, sanitize, anti-injection, recover quoted context | [src/processors/input_processor.py](src/processors/input_processor.py) |
| Webhook handler | Trigger detection (`!hej` / @mention / DM), memory enrichment, rate limit | [src/handlers/webhook_handler.py](src/handlers/webhook_handler.py) |
| Orchestrator | Choose agent + output format + thinking flag | [src/agents/orchestrator.py](src/agents/orchestrator.py) |
| Specialist agents | `chat`, `vision`, `web_search`, `image_gen` | [src/agents/](src/agents/) |
| Fallback chain | Rate-limit-aware multi-provider retry | [src/providers/fallback_chain.py](src/providers/fallback_chain.py) |
| Providers | NVIDIA + OpenRouter OpenAI-compatible clients | [src/providers/](src/providers/) |
| Output processor | Text/voice/image delivery + OpenCC s2t | [src/processors/output_processor.py](src/processors/output_processor.py) |
| Memory service | Layered chat memory, episodic recall, user profile | [src/services/memory_service.py](src/services/memory_service.py) |

### 2. End-to-end request lifecycle

The webhook returns `200 OK` **before** any model is called. All heavy work happens in a fire-and-forget background task, so LINE never sees a slow response and never retries.

```text
┌────────────────────────────────────────────────────────────────┐
│ 1-2  User  ->  LINE platform  ->  POST /webhook                │
│              (+ X-Line-Signature)                              │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 3  FastAPI /webhook                                            │
├────────────────────────────────────────────────────────────────┤
│ validate HMAC-SHA256 signature                                 │
│ return 200 OK immediately  ->  LINE   (never retries)          │
│ create_task(_process_event)   (fire-and-forget)                │
└────────────────────────────────────────────────────────────────┘
                                 ↓  background task
┌────────────────────────────────────────────────────────────────┐
│ 4  Input processor                                             │
├────────────────────────────────────────────────────────────────┤
│ download image · sanitize · block injection · recover quote    │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 5  Enrich                                                      │
├────────────────────────────────────────────────────────────────┤
│ rate-limit check  +  load memory                               │
│ -> history + summary + previous routing state                  │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 6  Orchestrator.route()                                        │
├────────────────────────────────────────────────────────────────┤
│ fast rule hit          ->  RouterDecision                      │
│ else LLM classify      ->  Nemotron 3 Super 120B (JSON)        │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 7  Agent.process()  ->  FallbackChain.generate()               │
├────────────────────────────────────────────────────────────────┤
│ try model 1 -> 2 -> 3  (skip rate-limited; thinking on/off)    │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 8  Output processor   (OpenCC simplified -> traditional)       │
├────────────────────────────────────────────────────────────────┤
│ reply_token (free)  ->  push (fallback)                        │
│ text · voice (TTS -> GCS) · image (GCS signed URL)             │
└────────────────────────────────────────────────────────────────┘
                                 ↓  deliver
┌────────────────────────────────────────────────────────────────┐
│ 9-10  LINE delivers reply -> user                              │
├────────────────────────────────────────────────────────────────┤
│ record_interaction; rotate summary if window full (async)      │
└────────────────────────────────────────────────────────────────┘
```

### 3. Orchestrator routing & adaptive thinking

Routing is a two-tier funnel. Cheap deterministic regex rules resolve the obvious cases with **zero LLM calls**; only genuinely ambiguous text reaches the LLM classifier. Independently, the orchestrator decides whether the downstream model should reason ("thinking"). The orchestrator's own JSON classification always stays non-thinking for output stability.

```text
┌────────────────────────────────────────────────────────────────┐
│ AgentRequest                                                   │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ FAST RULES   (regex · zero LLM call)                           │
├────────────────────────────────────────────────────────────────┤
│ image + gen keywords      ->  image_gen                        │
│ image (only / question)   ->  vision                           │
│ URL detected              ->  web_search                       │
│ time-sensitive / lookup   ->  web_search                       │
│ draw / generate keywords  ->  image_gen                        │
│ follow-up to last turn    ->  continue previous agent          │
│ no rule matched           ->  LLM classifier (below)           │
└────────────────────────────────────────────────────────────────┘
                                 ↓  only ambiguous text
┌────────────────────────────────────────────────────────────────┐
│ LLM CLASSIFIER   Nemotron 3 Super 120B · temp 0 · 384 tok      │
├────────────────────────────────────────────────────────────────┤
│ outputs single-line JSON:                                      │
│   agent · output_format · needs_thinking · task · reason       │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ ROUTER DECISION  ->  adaptive thinking control                 │
├────────────────────────────────────────────────────────────────┤
│ reasoning-heavy   (debug · traceback · proof · planning ·      │
│ long-form)        ->  THINKING ON   · Gemma 4 31B              │
│                                                                │
│ default / short   (greeting · simple Q)                        │
│                   ->  THINKING OFF  · Qwen3.5 397B             │
└────────────────────────────────────────────────────────────────┘
```

- **Fast-rule signals:** image-generation verbs, attached images, URLs, time-sensitive / lookup keywords (news, weather, prices, addresses), and follow-up indicators that anchor to the previous routed agent.
- **Output format** (`text` / `voice` / `image`) is decided in the same pass — voice keywords trigger TTS, negation keywords force text, image-gen always returns an image.
- **Thinking is opt-in:** the default path runs the larger Qwen3.5 397B without reasoning for speed; only clearly reasoning-heavy requests switch to the dedicated thinking model.

### 4. Multi-agent dispatch & model matrix

Four specialist agents share one rate-limit-aware fallback chain but carry their own system prompt, temperature, token budget, and model order.

```text
┌────────────────────────────────────────────────────────────────┐
│ SPECIALIST AGENTS   (own prompt · temperature · max tokens)    │
├────────────────────────────────────────────────────────────────┤
│ chat        0.7 / 2048   (+ agentic tool loop)                 │
│ vision      0.5 / 1024                                         │
│ web_search  0.2 / 3072                                         │
│ image_gen   0.7 / 1024   (two-stage pipeline)                  │
└────────────────────────────────────────────────────────────────┘
                                 ↓  each agent picks a fallback chain
┌────────────────────────────────────────────────────────────────┐
│ TEXT FALLBACK CHAIN   (chat · web_search · img stage 1)        │
├────────────────────────────────────────────────────────────────┤
│ 1  NVIDIA Qwen3.5 397B   (Gemma 4 31B when thinking)           │
│ 2  OpenRouter Nemotron 3 Super 120B  (free)                    │
│ 3  openrouter/free       (auto router)                         │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ VISION FALLBACK CHAIN                                          │
├────────────────────────────────────────────────────────────────┤
│ 1  NVIDIA Qwen3.5 397B VLM  (Gemma 4 31B when thinking)        │
│ 2  OpenRouter Gemma 4 31B   (free)                             │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ IMAGE GENERATION   (NVIDIA Visual GenAI NIM)                   │
├────────────────────────────────────────────────────────────────┤
│ Stage 1 (text):  refine prompt + describe   (parallel)         │
│ Stage 2 (NIM) :  FLUX.1-dev · 50 steps · cfg 5                 │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ chat AGENTIC TOOL LOOP   (max 4 iterations)                    │
├────────────────────────────────────────────────────────────────┤
│ recall_memory(query, k)     vector recall of episodes          │
│ update_user_profile(facts)  durable cross-chat facts           │
│ web_search(query)           Tavily fallback                    │
└────────────────────────────────────────────────────────────────┘
```

**Model inventory (all roles & specs):**

| Role | Model | Provider | Params / context | Capabilities |
| --- | --- | --- | --- | --- |
| Text default (non-thinking) | Qwen3.5 397B (`qwen/qwen3.5-397b-a17b`) | NVIDIA | 397B MoE / 17B active · 262K | vision, reasoning |
| Text reasoning (thinking) | Gemma 4 31B (`google/gemma-4-31b-it`) | NVIDIA | 31B · 262K | vision, tools, reasoning |
| Text fallback #1 | Nemotron 3 Super 120B (`nvidia/nemotron-3-super-120b-a12b:free`) | OpenRouter | 120B MoE · 262K | tools, reasoning |
| Text fallback #2 | Free auto router (`openrouter/free`) | OpenRouter | auto · 200K | vision, tools |
| Orchestrator primary | Nemotron 3 Super 120B (`:free`) | OpenRouter | 120B MoE · 262K | tools, reasoning |
| Orchestrator fallback | Nemotron 3 Super 120B (`nvidia/nemotron-3-super-120b-a12b`) | NVIDIA | 120B MoE · 1M | tools, reasoning |
| Vision primary | Qwen3.5 397B | NVIDIA | 397B MoE · 262K | vision, reasoning |
| Vision fallback | Gemma 4 31B (`google/gemma-4-31b-it:free`) | OpenRouter | 31B · 262K | vision, tools |
| Image generation | FLUX.1-dev (`black-forest-labs/flux.1-dev`) | NVIDIA Visual GenAI NIM | diffusion · 50 steps | image_gen |
| Episodic embeddings | `nvidia/nv-embedqa-e5-v5` | NVIDIA | 1024-dim | multilingual embeddings |
| Memory compaction | Qwen3.5 397B | NVIDIA | 397B MoE · 262K | background summary |

### 5. Provider fallback chain & thinking timeout

The fallback chain is where resilience lives. It pre-skips rate-limited models, falls through on `429` / provider errors, and protects latency with a thinking-mode timeout that retries once with reasoning disabled.

```text
┌────────────────────────────────────────────────────────────────┐
│ agent.generate(targets, messages)                              │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ thinking_timeout set AND thinking ON ?                         │
├────────────────────────────────────────────────────────────────┤
│ yes ->  run chain in reasoning mode (deadline 180s);           │
│         on timeout, retry whole chain with thinking OFF        │
│ no  ->  go straight to the target loop below                   │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ FOR EACH (provider, model) target — in priority order          │
├────────────────────────────────────────────────────────────────┤
│ RateTracker: is model available ?                              │
│   no   ->  skip (pre-empt 429)  ->  next target                │
│   yes  ->  provider.generate()                                 │
│              429            ->  next target                    │
│              ProviderError  ->  next target                    │
│              success        ->  return ProviderResponse        │
│                                                                │
│ all targets exhausted  ->  AllModelsRateLimitedError /         │
│                            AllProvidersFailedError             │
└────────────────────────────────────────────────────────────────┘
```

NVIDIA reasoning activation is model-family specific and handled inside `NvidiaProvider`: Qwen3.5 uses the `/think` soft switch plus `chat_template_kwargs`; Gemma 4 and Nemotron 3 use `chat_template_kwargs.enable_thinking`; the provider's `resolve_model()` transparently swaps the primary `NVIDIA_MODEL` to `NVIDIA_THINKING_MODEL` when reasoning is requested.

### 6. Layered memory & episodic recall

Each chat scope keeps a small rolling window plus one long-term summary. When the window fills, a background job compacts it into the next summary and writes an embedded episode for later vector recall. A cross-chat user profile lives independently and survives `!new`.

```text
┌────────────────────────────────────────────────────────────────┐
│ New interaction recorded                                       │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ PER-CHAT MEMORY   (user / group / room scope)                  │
├────────────────────────────────────────────────────────────────┤
│ Recent window (6 msgs)  ──full──>  Long-term summary           │
│                                    (1 paragraph)               │
│ window full  ->  background compaction (Qwen3.5 397B)          │
│              ->  rewrites summary + writes one episode         │
└────────────────────────────────────────────────────────────────┘
                                 ↓  embed snapshot
┌────────────────────────────────────────────────────────────────┐
│ EPISODIC MEMORY                                                │
├────────────────────────────────────────────────────────────────┤
│ episode snapshots + nv-embedqa-e5-v5 (1024-dim vectors)        │
│ queried on demand by chat tool: recall_memory(query, k)        │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ CROSS-CHAT PROFILE   user_profile/{user_id}                    │
├────────────────────────────────────────────────────────────────┤
│ durable facts shared across DMs & groups                       │
│ written by tool: update_user_profile() · survives !new         │
└────────────────────────────────────────────────────────────────┘

Backend:  InMemoryBackend (default, resets on cold start)  |  FirestoreBackend (opt-in)
```

- **Recent window:** last `MEMORY_RECENT_MESSAGE_LIMIT` (default 6) text turns, fed verbatim into the prompt.
- **Long-term summary:** one compact paragraph biased toward preserving still-valid facts; regenerated in the background so it never blocks a reply.
- **Episodic recall:** every summary rotation writes an embedded snapshot; the chat agent can vector-search older context on demand via `recall_memory`.
- **User profile:** durable facts (nickname, hobbies, location) shared across all chats; intentionally **not** cleared by `!new`.
- **Backends:** `InMemoryBackend` for free/local runs, `FirestoreBackend` for persistence across Cloud Run cold starts and multiple instances.

### 7. Deployment & delivery pipeline

```text
┌────────────────────────────────────────────────────────────────┐
│ Developer   -   ./scripts/deploy_cloud_run.sh                  │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ Cloud Build   (no local Docker build needed)                   │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ Artifact Registry   (container image)                          │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ Cloud Run revision                                             │
├────────────────────────────────────────────────────────────────┤
│ smoke check  ->  route traffic  ->  prune old images           │
│ signed URLs -> Cloud Storage  ·  optional -> Firestore         │
└────────────────────────────────────────────────────────────────┘

Cloud Scheduler ──(1 / min · X-Cron-Token)──→ POST /internal/cron
```

One command (`./scripts/deploy_cloud_run.sh`) enables APIs, submits the build, runs smoke checks, routes traffic to the new revision, and prunes old revisions/images — no local Docker build required.

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Runtime | GCP Cloud Run |
| Framework | FastAPI + Uvicorn |
| Text LLM (default) | NVIDIA Qwen3.5 397B (non-thinking) |
| Text reasoning LLM | NVIDIA Gemma 4 31B (thinking) / OpenRouter Nemotron 120B |
| Vision LLM | NVIDIA Qwen3.5 397B (primary) / NVIDIA Gemma 4 31B when thinking |
| Fallback LLMs | NVIDIA Nemotron 3 Super 120B / OpenRouter Nemotron 120B / OpenRouter Gemma 4 31B / `openrouter/free` (task-dependent) |
| Image generation | NVIDIA FLUX.1-dev (`black-forest-labs/flux.1-dev`) |
| Web search | Tavily |
| Voice | `edge-tts` |
| Object storage | Google Cloud Storage |
| Scheduling | Cloud Scheduler → `/internal/cron` |
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
- The main prompt memory is `1 long-term summary + up to 6 recent text messages` per chat. It defaults to in-memory per instance; set `FIRESTORE_ENABLED=true` to persist chat memory, user profile, and episodic recall across redeploys / cold starts.
- Quoted-message cache, LINE push counters, and model rate-tracker state are still **in-memory per instance**.
- `WEB_SEARCH_MONTHLY_QUOTA` is kept only for backward compatibility. The app no longer enforces a local search quota; actual availability depends on Tavily credentials and provider-side limits.
- Relative search phrases like `today`, `latest` and `recent` are handled through keyword extraction plus Tavily topic/time-range targeting rather than by prefixing the raw query with a timestamp.
- Users can send `!new` in DM, group, or room chats to clear the current chat memory scope.

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
| `NVIDIA_API_KEY` | — | [NVIDIA](https://build.nvidia.com) API key; enables Qwen3.5 397B / Gemma 4 31B, embeddings, and image generation |
| `NVIDIA_MODEL` | `qwen/qwen3.5-397b-a17b` | NVIDIA primary model (non-thinking) |
| `NVIDIA_THINKING_MODEL` | `google/gemma-4-31b-it` | Dedicated NVIDIA thinking model (Gemma 4 31B; auto-switched when deep reasoning is needed) |
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
| `THINKING_TIMEOUT_SECONDS` | `180` | Abort a reasoning attempt after N seconds and retry once with thinking disabled; set `0` to disable |
| `REQUIRE_REASONING_MODELS` | `true` | Only route to reasoning-capable models |
| `REQUIRE_REASONING_TOKENS` | `true` | Warn when a reasoning model returns no reasoning content |

These flags control provider-side reasoning support, but the actual on/off decision is made per request by the orchestrator for downstream agents. **The orchestrator's own routing classification stays non-thinking to keep JSON output stable.** By default, downstream requests use the larger 397B model without thinking. Only clearly reasoning-heavy tasks — such as step-by-step debugging, traceback / exception troubleshooting, proofs / derivations, deep planning, or full-length writing / translation — switch to the dedicated NVIDIA Gemma 4 31B thinking model. When thinking mode activates on the primary NVIDIA leg, the provider automatically swaps from `NVIDIA_MODEL` to `NVIDIA_THINKING_MODEL`. Explicit alternate NVIDIA targets such as `ORCHESTRATOR_FALLBACK_MODEL` keep their configured model ID and receive their native thinking flag instead. OpenRouter-hosted reasoning models continue to use OpenRouter's `reasoning` payload, while direct NVIDIA Gemma 4 and Nemotron 3 use `chat_template_kwargs.enable_thinking` automatically.

### Model routing

| Variable | Default | Description |
| --- | --- | --- |
| `ORCHESTRATOR_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Primary orchestrator model (OpenRouter) |
| `ORCHESTRATOR_FALLBACK_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Orchestrator NVIDIA fallback target; keeps Nemotron and uses NVIDIA's native thinking flag when needed |
| `ORCHESTRATOR_TEMPERATURE` | `0` | Sampling temperature for LLM-based routing |
| `ORCHESTRATOR_MAX_TOKENS` | `384` | Max tokens for orchestrator JSON routing output |
| `AGENT_FALLBACK_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Shared fallback for text agents |
| `VISION_FALLBACK_MODEL` | `google/gemma-4-31b-it:free` | Vision agent fallback |

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
| `IMAGE_GEN_PRIMARY_MODEL` | `black-forest-labs/flux.1-dev` | Primary NVIDIA Visual GenAI NIM image model |
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

### Conversation memory

| Variable | Default | Description |
| --- | --- | --- |
| `MEMORY_RECENT_MESSAGE_LIMIT` | `6` | Recent text-message window size before compaction |
| `MEMORY_SUMMARY_TIMEOUT_SECONDS` | `180` | Background summary timeout; on timeout the memory keeps the latest 6-message recent window |
| `MEMORY_SUMMARY_TEMPERATURE` | `0.2` | Summary compaction temperature |
| `MEMORY_SUMMARY_MAX_TOKENS` | `384` | Max tokens for long-term summary updates |

### Persistent memory / episodic recall (optional)

| Variable | Default | Description |
| --- | --- | --- |
| `FIRESTORE_ENABLED` | `false` | Keep `false` for free/local in-memory mode; set `true` to persist memory in Firestore Native |
| `FIRESTORE_PROJECT_ID` | — | Firestore project; leave empty on Cloud Run to use the runtime project |
| `FIRESTORE_DATABASE` | `(default)` | Firestore database name |
| `FIRESTORE_COLLECTION_PREFIX` | `linebot` | Prefix for chat memory, user profile, and episode collections |
| `MEMORY_CACHE_TTL_SECONDS` | `60` | Local read-through cache TTL to reduce Firestore reads |
| `LINE_PROFILE_CACHE_TTL_SECONDS` | `86400` | LINE displayName cache TTL |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | Embedding model for episodic recall; only used when `NVIDIA_API_KEY` is set |
| `NVIDIA_EMBEDDING_ENDPOINT` | NVIDIA embeddings API | Optional custom embeddings endpoint |

### Bot & conversation

| Variable | Default | Description |
| --- | --- | --- |
| `BOT_NAME` | `Assistant` | Bot display name in prompts |
| `LINE_BOT_USER_ID` | — | Bot's LINE userId for precise @mention detection |
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
