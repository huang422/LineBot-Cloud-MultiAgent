# LineBot-Cloud-MultiAgent

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-GCP-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![LINE](https://img.shields.io/badge/LINE-Messaging%20API-00C300?logo=line&logoColor=white)](https://developers.line.biz/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron%203%20Super%20%C2%B7%20Ultra-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-dots--3%20%C2%B7%20Nemotron%203-6366F1)](https://openrouter.ai/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-orange.svg?logo=buymeacoffee&logoColor=white)](https://www.buymeacoffee.com/huang422)


**A cloud-first, multi-agent LINE bot for GCP Cloud Run.**

LineBot-Cloud-MultiAgent receives LINE webhook events, routes them through a smart orchestrator, and dispatches them to specialized chat, vision, web-search, image-generation, and voice workflows while staying inside free-tier-friendly guardrails.

---

## Key Features

### Multi-Agent Orchestration

Fast regex rules handle obvious requests without an LLM call; harder cases fall back to model-based routing. The orchestrator dispatches to four specialist agents — `chat`, `vision`, `web_search`, and `image_gen` — each with its own model configuration and system prompt. A short continuation of the previous turn (at most 12 characters starting with a follow-up word such as 那…呢, 然後, 還有, 繼續, 再 or 更多, or asking to search or check again: 再搜尋一次, 重新查, 確認一下, and the common typo 在搜尋確認一次) keeps the previous route; a quoted message, or anything longer, goes to the LLM router together with the previous route, task and output format, so a new question after a search is not forced back into search.

### Adaptive Thinking Control

The orchestrator does more than choose an agent: it also decides whether the downstream model should use provider-native reasoning / thinking mode for that specific request. **The orchestrator's own JSON routing stays in non-thinking mode for stability.** By default, downstream requests run on NVIDIA Nemotron 3 Super 120B without thinking for fast responses (after the Colab qwen3.8 tier when a Colab session is READY, see below). Only clearly reasoning-heavy requests — for example step-by-step debugging, traceback / exception troubleshooting, proofs / derivations, deep planning, or full-length writing / translation — go to the **thinking role**: Colab qwen3.8 at `COLAB_DEEP_THINK` when READY, then Nemotron 3 Super with thinking on (with a reasoning budget), then Nemotron 3 Ultra with thinking on at medium effort (Ultra takes no reasoning budget), then OpenRouter `dots-3-note-preview:free` and `nemotron-3-ultra-550b-a55b:free` at medium effort. The thinking tiers together may use at most half of the time left; after them the chat tiers answer with thinking off, minus any model that already timed out in the same request. A request with an image uses the vision tiers instead, with thinking on, then off, when the request needs it.

### Multi-Modal Input & Output

Text, images, quoted context, and voice replies all flow through the same webhook pipeline. Users can send photos for vision analysis, request image generation, receive voice replies, and quote earlier messages. Incoming audio messages are currently kept only as lightweight conversation context — speech-to-text is not implemented yet.

### Intelligent Provider Fallback

Every agent call goes through a fallback chain of model tiers. Each tier has its own time limit inside the request's deadline (the 90 s reply window, or the 240 s turn deadline for thinking and images; see [Reply window rules](#5-provider-fallback-chain--deadlines)). A tier that runs out of time after at least `MIN_TIER_SECONDS` counts as a transient failure, and two in a row cool the model down for two minutes, so one hung endpoint cannot keep eating the window. Models that are cooling down, over their provider's request budget (NVIDIA 40/minute; OpenRouter free 20/minute and 50/day per account) or retired are skipped before sending a request, and an empty completion counts as a failure. A `200` whose body is an `error` object (OpenRouter's way of reporting an upstream failure after routing) counts as the error its `code` names, so two calls that end in such a 503 (an overloaded or unavailable upstream is first retried once, below) cool the model down like real ones. Within one walk (in chat, the whole tool loop) a model that timed out after a fair try is not called again by a later tier when it had its tier's whole time limit or the later tier would give it hardly longer (a thinking request whose Ultra thinking tier hung skips Ultra with thinking off). After a READY Colab session's qwen3.8 tier (if any), text agents try NVIDIA Nemotron 3 Super → NVIDIA Nemotron 3 Ultra → OpenRouter `dots-studio/dots-3-note-preview:free` → `nvidia/nemotron-3-ultra-550b-a55b:free` (Colab, then two NVIDIA models, then OpenRouter); vision uses NVIDIA Nemotron 3 Nano Omni, then muse-glimmer 30B, then the free OpenRouter dots-3 (Ultra and Super are text only). An OpenRouter 429 passed on from one model's upstream ("temporarily rate-limited upstream") cools only that model; only the account's own free-model cap cools every `:free` id. A model the API reports as retired (HTTP 410) is skipped for the rest of the process; an HTTP 404 takes it out for six hours on NVIDIA and for ten minutes on OpenRouter, whose "No endpoints found" clears once an upstream comes back. Only `:free` OpenRouter ids are called unless `OPENROUTER_ALLOW_PAID=true`. Once a day a lifecycle check asks the public model catalogs whether any configured model is retired, deprecated or expiring, before live traffic finds out.

### Colab Ollama tier

With `COLAB_ENABLED`, the bot also runs its own models on the owner's Colab A100 under Ollama (the latest release at or above 0.40.4, which the bot picks at each launch, else a pinned fallback): `qwen3.8:27b-mtp-q4_K_M` answers chat, vision and thinking requests, and `clef-flash:9b` (Cloudflare's decision model) routes on Ollama's `/v1/systemone` with its top choice (argmax by default). Both models follow their tags; their digests are logged, never enforced. Every user is served from Colab first (`MODEL_PATH_PREFERENCE=colab`), but **no message ever waits for it**: a session starts on demand in 4–8 minutes, and until it is READY every message is answered at once by NVIDIA → OpenRouter; once READY, a Colab tier stops 20 s before the reply deadline so an API tier can still answer. The per-minute cron drives the session (start on demand within a compute-unit floor and daily cap, keep-alive, release after 15 idle minutes), the slow Colab calls run as `colab` Cloud Tasks, the VM registers its Cloudflare tunnel through a signed `POST /internal/colab/register`, and owners (`LINE_ADMIN_USER_IDS`) control it with `!colab on` / `off` / `status`. The Colab FAQ forbids serving a web service and third-party access from a runtime; the owner accepted that risk.

- Why, and the guards: [ADR-0010](docs/adr/0010-colab-ollama-tier.md)
- First-time setup, the CPU dry run, the first A100 run and token re-minting: [DEPLOY.md](DEPLOY.md), "Colab 層"
- The VM side (bootstrap, auth proxy, protocol, the Ollama release and model tags, VRAM, troubleshooting): [colab/README.md](colab/README.md)

### Image Generation with Description

User requests are first refined into optimized English prompts by the utility tiers (Nemotron 3 Super, then Nemotron 3 Ultra on NVIDIA, then dots-3 and Nemotron 3 Ultra free on OpenRouter; text only, so a reference image is not sent), then passed to NVIDIA FLUX.2 klein 4B (FLUX.1-dev as fallback) through the NVIDIA Visual GenAI NIM API, each within its own time limit. A prompt klein's safety filter blocks is tried once more on FLUX.1-dev; if that blocks it too, the user is told (in zh-TW) that the image model's safety filter blocked it and to rephrase. Prompt refinement and user-facing description generation run in parallel for lower latency. The description is optional: the image never waits for it, and a description not ready when the image is goes out without it. The image is uploaded to GCS and delivered as a LINE image message alongside a text description.

### Voice Replies

`edge-tts` synthesizes speech at zero cost. Audio is uploaded to GCS with signed URLs and sent as a LINE audio message. Voice and language are configurable.

### Layered Chat Memory

Each `user` / `group` / `room` chat keeps a rolling **6-message recent text window** plus **one long-term summary**. When the recent window fills up, the app queues a compaction task that uses the utility chain (NVIDIA Nemotron 3 Super 120B, then Nemotron 3 Ultra on NVIDIA, then OpenRouter dots-3 and Ultra's free id, all four inside the summary's 180 s, which the one or two ranking calls of a summary over 400 characters share) to compact `existing summary + the window` into the next summary, with a bias toward preserving still-valid long-term facts and preferences unless newer dialogue clearly invalidates them. The window is kept in the order LINE received the messages, and the commit removes exactly the messages it summarized (by message id), so messages that arrive during a compaction are never lost or re-summarized ([ADR-0003](docs/adr/0003-recent-window-is-a-trigger.md)). A chat compacted within the last `COMPACTION_DEBOUNCE_SECONDS` (120) waits for a later message, unless its window reached half the safety cap. Sending `!new` clears the current chat scope's recent window, summary, and episodic recall records (messages sent after the `!new` are kept; a question still being answered when `!new` runs is not recorded into the new conversation), then replies with `Let's start a new chat!`; a compaction still summarizing the cleared conversation discards its result instead of writing it into the new one.

For groups and rooms, non-triggered user text can also be buffered and batch-persisted into memory so a later `!hej` / @mention turn can still see the surrounding thread without forcing a write for every single message.

Memory is stored through a pluggable backend layer:

- **`InMemoryBackend`** (default) — fast, no external dependencies, but resets on Cloud Run scale-to-zero. Suitable for local dev and tests.
- **`FirestoreBackend`** (opt-in) — persists `{prefix}_chat_memory/{source_type}::{chat_id}` and `{prefix}_user_profile/{user_id}` documents (`FIRESTORE_COLLECTION_PREFIX`, default `linebot`) in Firestore Native, surviving cold starts and multi-instance deployments. Stays inside the Firestore free tier (50K reads / 20K writes / 1 GiB per day) when the database is co-located with Cloud Run (e.g. `us-west1`). Enable by setting `FIRESTORE_ENABLED=true` and `FIRESTORE_PROJECT_ID=<gcp-project>` (see `.env.example`). Stored data, not reads or writes, is the free-tier limit most likely to run out: DEPLOY.md lists the large fields (`recent_messages`, episode `summary` / `snippet`, …) whose single-field indexes to exempt, with the `gcloud` commands.

In addition to the chat-scoped memory, the bot maintains a **per-user profile** (`{prefix}_user_profile/{user_id}`) that is shared across DMs and groups. It holds the LINE `displayName` (resolved via `/v2/bot/profile/{userId}` or `/v2/bot/{group|room}/{id}/member/{userId}` depending on context) and `last_seen` metadata, updated after the reply is sent and written only when the name changed or the last write is a day old. The LLM extends it via the `update_user_profile` tool when the user volunteers durable facts (nickname, hobbies, location), and the sender's newest facts (at most 10, 500 characters) are shown to the chat, vision and web-search agents as a fenced `[使用者 profile，僅供參考]` block (in a group or room, with a system-prompt rule not to bring them up unless the sender does, since every member reads the reply); the router and the image prompt refiner do not see them. Each stored fact is one line of at most 200 characters, with prompt-injection markers stripped. Profiles are read through a 10-minute in-process cache. `!new` intentionally does not delete this cross-chat profile; remove that document manually from Firestore if a user needs a full profile reset.

### Native Tool Calling for Chat Agent

The chat agent now ships with an OpenAI-style **agentic tool loop**. While a normal reply is being generated, the LLM can decide to invoke tools and re-enter the loop with the results — no orchestrator round-trip required:

- **`recall_memory(query, k)`** — vector-search older episodes stored in Firestore (`{prefix}_episodes/...`). A chat gets at most one episode per `EPISODE_MIN_INTERVAL_SECONDS` (3600): the conversation compacted since the previous episode (the newest 2000 characters, each message clipped to 300) plus the summary at that time, embedded together with free NVIDIA `nemotron-3-embed-1b` embeddings (the API returns 2048 dims; the first 1024 are kept and re-normalized) and tagged with the embedding model, so recall only compares vectors from the same model ([ADR-0011](docs/adr/0011-embedding-migration.md)). Each match returns up to 600 characters of that conversation (around the query's words, else the newest part) and up to 300 of the summary, so the model gets the raw exchange, not just a copy of the summary it already has.
- **`update_user_profile(facts, confidence)`** — persists durable user facts into `{prefix}_user_profile/{user_id}`, shown in later conversations (see above). Confidence-gated (>= 0.7) and capped at 20 facts.
- **`web_search(query)`** — fallback web search through the same backend chain as the `web_search` agent (Tavily; snippets only, no page reads) when the orchestrator's fast-rules did not pre-route to the dedicated `web_search` agent (dual-track design), at most 2 searches per reply, every tool round counted (`WEB_SEARCH_TOOL_MAX_CALLS_PER_TURN` in [src/agents/tools.py](src/agents/tools.py); the tool loop answers a third call `limit` without running it). It is offered only when Tavily is configured (a key, or keyless mode). Fast-rule routing for news / URLs / finance still wins by default, but the LLM can also pull live data on its own.

Tool rounds are capped by `tool_loop_max_iterations` (default 4); after the last round one more request with `tool_choice="none"` asks for the answer, so the user gets text and the last tool results are used. The first round has the request's whole deadline, like a reply without tools (most turns end there with a plain answer). Once the model asks for tools, every tool call and every later tool-offering round must end 20 s (`ANSWER_RESERVE_SECONDS`) before the deadline, which only that final request uses; a tool that runs out of time returns a short error result, tool calls that arrive inside those 20 s get it without being run, and a later round that would not fit (or is cut off, which is not held against the model) goes straight to that final request. The loop never changes the caller's messages, and it sends each assistant tool-call turn back rebuilt in OpenAI format, so a round can move to another tier. When the loop cannot answer (a tool-loop error, or every tier failing the tools-bearing requests, e.g. HTTP 400 for tools), one plain `fallback_chain.generate()` without tools retries with the original messages within the same deadline; no time left, or every tier rate limited, reaches the user as the error message.

### Cloud-Only, One-Command Deploy

`./scripts/deploy_cloud_run.sh` handles everything — env and retired-model checks, a free-tier check of the media bucket, an upload leak guard, API enablement, a Cloud Build that lints, tests, builds and pushes the image (no secrets uploaded), a deploy from your machine with `.env` stored as one Secret Manager secret (plus the Colab tier's token mounted from its own secret when `COLAB_ENABLED=true`), smoke checks, revision routing, and old image/revision/secret-version pruning. No local Docker build step required. `./scripts/provision_gcp.sh` (a dry run unless you pass `--apply`) sets the project up once: least-privilege runtime and build service accounts, a short-lived us-west1 build staging bucket, the secret and an Artifact Registry cleanup policy ([ADR-0008](docs/adr/0008-secrets-and-identity.md)). The project stays within the GCP free tier: search runs on Tavily ([ADR-0009](docs/adr/0009-search-backends-and-ssrf-safe-fetching.md)), with no VM or VPC of its own.

### Config-First Behavior

System prompts live in `prompts/*.md`, and most runtime tuning lives in `.env` (every key is explained in [.env.example](.env.example)). Models, temperatures, token limits, rate limits, cost controls, and scheduling are configurable without touching code, while some fast-routing, safety, and output heuristics intentionally stay in Python for deterministic behavior.

---

## What it does

| Feature | How it works |
| --- | --- |
| Chat | General conversation, coding help, translation, and creative replies via the fallback chain |
| Adaptive thinking control | The orchestrator decides per request whether the model should think; most requests run Nemotron 3 Super with thinking off, while only clearly reasoning-heavy tasks turn thinking on |
| Vision | Accepts images, screenshots, and photos for analysis |
| Web search | A query planner on the utility tiers ([src/services/search/query.py](src/services/search/query.py)) turns the message into one search: a follow-up such as 那高雄呢 or 再搜尋確認一次 is rewritten from the recent turns and the router's task summary, a long question is condensed into keywords, and only a short standalone query is searched as typed (so is the message when the rewrite changes a name or version it spells with letters and digits, as when qwen3.8 came back as Qwen3-8B on 2026-10-16). The search goes to **Tavily** (basic depth, with a key or keyless), with at most one relaxed retry when a search found nothing (none when Tavily failed); the top 3 results' pages are read (SSRF-safe fetch + trafilatura) so the model sees page text, not just snippets. If no backend answers, the model answers from its own knowledge, prefixed with （目前無法使用搜尋服務，以下依模型既有知識回答，可能不是最新資訊）. See [ADR-0009](docs/adr/0009-search-backends-and-ssrf-safe-fetching.md) |
| Webpage reading | If the user posts a URL, the bot fetches it itself (http/https on 80/443 only; every resolved address must be public, so private ranges, the metadata server and link-local addresses are refused; redirects re-checked; HTML or text only, 2 MB cap) and extracts the main text with trafilatura; pages it cannot read go to Tavily Extract, and the text is injected before synthesis |
| Image generation | Prompt refinement and description generation run in parallel, then NVIDIA FLUX.2 klein 4B (FLUX.1-dev fallback) generates the image; the text description goes with the image when it is ready in time (the image never waits for it) |
| Voice reply | `edge-tts` produces audio, uploads it to GCS, and sends a LINE audio reply |
| Quoted context | Reply to an earlier text or image and recover the original content from local cache |
| Scheduled messages | Cloud Scheduler pings `/internal/cron` every minute; the app pushes any weekly / yearly reminder whose day and time match the current minute in `SCHEDULED_MESSAGES_TIMEZONE` (Asia/Taipei) |
| Plain text for LINE | LINE shows Markdown and LaTeX raw, so every outgoing text (reply, caption, text read aloud) first loses its markup (`**bold**`, `##`–`######` headings, code fence lines, inline code, `\( … \)` / `\[ … \]` / `$$…$$`, `[text](http…)` link syntax, table separator rows; URLs, emoji and code lines stay, and fenced code stays verbatim, since the reply prompts ask for code in fences) in [src/utils/line_text.py](src/utils/line_text.py) |
| Simplified-to-traditional conversion | All LLM output is then normalized through OpenCC before being sent back to LINE |
| Prompt-injection markers | Chat-template control tokens (`<\|im_start\|>`, `<\|system\|>`, `[INST]`, `<\|channel>`), `system prompt:` and 「忽略之前的指令」-style phrasing are stripped from every incoming text, group chatter included; a message that had one is still answered, with a guard line in the system prompt (one that was nothing but a marker gets the 請附上想問的內容 reply). Router hints (task summary, reasoning) come from the user's text, so they travel as a fenced user turn, never in the system prompt |
| Health | Public `/health` answers only status, readiness and version; `/health/detail` (behind `X-Cron-Token`) shows every model tier with its health and last error, provider budgets, config warnings, the daily end-of-life results, tasks mode, memory, storage and scheduler state, web search (Tavily's mode, credits used, cooldown, last error), and the Colab session (status and since when, readiness, last error, compute-unit balance and rate, today's use; never its URL or bearer) |
| Owner commands | `!colab on` / `off` / `status` for the ids in `LINE_ADMIN_USER_IDS` (others get 此指令僅限管理員), in 1:1 chats and, without a mention, in groups; `!whoami` answers your userId in a 1:1 chat only |

---

## From local GPU to cloud-native

This project is the next-generation successor to [LineBot-VLM-GroupAgent](https://github.com/huang422/LineBot-VLM-GroupAgent), which ran a single Ollama model on a local NVIDIA GPU. LineBot-Cloud-MultiAgent has been redesigned from scratch for zero-hardware, cloud-only deployment.

| | [LineBot-VLM-GroupAgent](https://github.com/huang422/LineBot-VLM-GroupAgent) (v1) | LineBot-Cloud-MultiAgent (v3) |
| --- | --- | --- |
| Deployment | Local server + Cloudflare Tunnel | GCP Cloud Run (serverless) |
| Hardware | NVIDIA GPU + 32 GB RAM required | No local hardware (an optional Colab A100, started on demand) |
| LLM Provider | Ollama (single local model) | NVIDIA + OpenRouter (multi-provider fallback), optional Colab Ollama tier first |
| Architecture | Single model, serial queue | Multi-agent orchestrator with parallel dispatch |
| Vision Model | Qwen3.5 9B/35B (local) | Nemotron 3 Nano Omni, then muse-glimmer 30B (NVIDIA API), then dots-3 (OpenRouter free); Colab qwen3.8 first when READY |
| Image Generation | Not supported | NVIDIA FLUX.2 klein 4B / FLUX.1-dev (two-stage pipeline) |
| Voice Reply | Not supported | edge-tts + GCS signed URL |
| Reasoning | Ollama thinking mode (may OOM) | Provider-native thinking, switched on per request by the router; a token budget on top of `max_tokens` |
| Resilience | Single point of failure | Auto-fallback across providers and models |
| Cost | Electricity + hardware | Free-tier APIs; GCP kept inside the Always Free tier (scale-to-zero Cloud Run, no VM or VPC) |
| Deploy Command | Manual setup | One-command `./scripts/deploy_cloud_run.sh` |

---

## Architecture at a glance

1. `POST /webhook` receives a LINE event and validates the HMAC signature.
2. The webhook queues the event as a Cloud Task and returns `200 OK`; Cloud Tasks calls back `POST /internal/tasks/event`, a request of its own, so the work gets CPU under request-based billing (inline when Cloud Tasks is not configured).
3. Input processing downloads image payloads, sanitizes text (newlines kept, injection markers stripped), and recovers quoted context.
4. Rate limiting and chat-memory enrichment happen before routing.
5. The orchestrator chooses `chat`, `vision`, `web_search`, or `image_gen`, and also decides whether the request needs deep thinking: fast rules first, then clef-flash on a READY Colab session (its top choice; a non-zero COLAB_ROUTER_CONFIDENCE can defer), then the router role's API tiers.
6. The selected agent runs through the fallback chain and returns a normalized response.
7. The output processor sends text, voice, or image messages back to LINE.

Steps 3–7 are one plain function, `process_event(event, deps)` ([src/pipeline/event_pipeline.py](src/pipeline/event_pipeline.py)), not a graph framework: the flow is a single turn inside the LINE reply window, the fallback chain carries the deadlines, and Cloud Tasks plus Firestore carry the durability. Why LangGraph is not used, the seam kept for it, and what would make it worth adopting: [ADR-0012](docs/adr/0012-stay-custom-langgraph-seam.md).

### Model role table

Which models answer what, and in what order, is one declarative table ([src/runtime/model_roles.py](src/runtime/model_roles.py)), checked at startup against exact-model-id profiles ([src/providers/model_profiles.py](src/providers/model_profiles.py)). The defaults (the seconds are each tier's own `timeout_s`, always clipped to what is left of the request's deadline):

| Role | Used for | Tiers, in order |
| --- | --- | --- |
| `router` | LLM routing (thinking off) | Colab `clef-flash:9b` (6 s) → NVIDIA Nemotron 3 Super (`json_schema`, 256 tokens, t=0.1, 12 s) → NVIDIA Nemotron 3 Ultra (no structured output: JSON asked for in the prompt, 256 tokens, t=0.1, 6 s: it routes in about 2 s when healthy, and on 2026-10-16 it hung on every call) → OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` (`json_object`, effort none, 12 s) |
| `chat` | chat and web-search answers | Colab `qwen3.8:27b-mtp-q4_K_M` (think `COLAB_CHAT_THINK`, 70 s) → NVIDIA Nemotron 3 Super (thinking off, the card's sampling, 45 s) → NVIDIA Nemotron 3 Ultra (thinking off, 25 s) → OpenRouter `dots-studio/dots-3-note-preview:free` (effort none, 40 s) → `nvidia/nemotron-3-ultra-550b-a55b:free` (effort none, 40 s). Super before Ultra since 2026-10-16, when Ultra hung on every call (5/5 production timeouts, 0/4 in an A/B) while Super answered 4/4 in 12–52 s and short answers in 1–3 s (an answer over about 1000 tokens runs past 45 s and counts as a timeout). When both NVIDIA tiers run out, dots-3 has about 20 s of the 90 s window minus routing (and search) time, none when routing itself stalls on NIM |
| `thinking` | requests the router marks as needing thinking; the chat tiers follow with thinking off | Colab qwen3.8 (think `COLAB_DEEP_THINK`, 110 s) → NVIDIA Super (thinking on, `NVIDIA_THINKING_BUDGET` as its `reasoning_budget`, 60 s) → NVIDIA Ultra (thinking on at `medium_effort`; no `reasoning_budget`, which Ultra rejects, so `NVIDIA_THINKING_BUDGET` only widens `max_tokens`; 30 s) → OpenRouter `dots-studio/dots-3-note-preview:free` (effort medium, 120 s) → `nvidia/nemotron-3-ultra-550b-a55b:free` (effort medium, 30 s); together at most half of the time left (under 120 s of the 240 s turn deadline). dots-3's 120 s lets that half, not its own limit, end it: the whole half when NIM fails fast, the rest of it when both NVIDIA tiers hang |
| `vision` | any request with an image, whichever agent answers | Colab qwen3.8 (70 s; skipped by design for an image that reaches the model as a URL) → NVIDIA `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (thinking off, t=0.2, 1024 tokens, 15 s) → NVIDIA `meta/muse-glimmer-30b` (thinking off, 35 s) → OpenRouter `dots-studio/dots-3-note-preview:free` (effort none, 20 s). The API tiers never think; an image prompt carries at most 3000 characters of history, summary and profile facts (`IMAGE_CONTEXT_CHARS`), about 4k tokens in all, because muse-glimmer took about 40 s at 6.2k |
| `utility` | memory summaries, search query planning, image-prompt refinement and captions | NVIDIA Nemotron 3 Super (thinking off, t=0.2, 30 s) → NVIDIA Nemotron 3 Ultra (thinking off, 30 s) → OpenRouter `dots-studio/dots-3-note-preview:free` (effort none, 30 s) → `nvidia/nemotron-3-ultra-550b-a55b:free` (effort none, 30 s); all four fit a memory summary's one 180 s deadline (`MEMORY_SUMMARY_TIMEOUT_SECONDS`), which the summary's ranking calls (one before and one after the update, for a summary over 400 characters) share, a model that hung in one of them being skipped in the rest; no Colab tier (its one slot belongs to replies) |
| `embed` | episodic recall | NVIDIA `nvidia/nemotron-3-embed-1b` (first `EMBEDDING_DIMENSIONS` of 2048 kept, 10 s) |
| `image` | image generation | NVIDIA FLUX.2 klein 4B (60 s) → FLUX.1-dev (50 steps, 90 s) |

- **Colab tiers** are on with `COLAB_ENABLED` (see [Colab Ollama tier](#colab-ollama-tier)); otherwise they stay disabled and `/health/detail` shows why. `MODEL_PATH_PREFERENCE=colab` (default) puts Colab tiers first in every role, `api` puts them last. Until a session is READY its tiers are skipped at once, with no network call. Once READY, Colab works to the same deadline as every plain answer, event time + `REPLY_WINDOW_SECONDS` (90 s); a Colab tier with a usable API tier after it must stop 20 s before the deadline (`COLAB_API_RESERVE_SECONDS`), or halfway to it when that is later, whatever its think level, and a request that needs thinking tries it only once, so a slow session cannot use up the window before the API tiers get their turn.
- **Sampling:** a tier's own `temperature` wins, then the model card's sampling from the model profile (Nemotron 3 Ultra and Super on NVIDIA: temperature 1.0, top_p 0.95; muse-glimmer: 0.6), then the agent's setting (`CHAT_TEMPERATURE`, `VISION_TEMPERATURE`, …), which therefore reaches only the models without card sampling (OpenRouter). Colab's qwen3.8 always samples as Qwen recommends (0.7 / top_p 0.8 / top_k 20 / presence penalty 1.5 without thinking, 1.0 / 0.95 / 20 with it) unless its tier sets a `temperature`. The router tiers on Super and Ultra and the utility tier on Super keep a deliberately low temperature (0.1 and 0.2): routing JSON, search queries and memory summaries should come out the same for the same input, and the card's 1.0 is untested for them.
- **Override a role** with `MODEL_ROLES`, a JSON object that replaces only the roles it names, for example `MODEL_ROLES='{"utility": [{"provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b", "thinking": false, "temperature": 0.2, "timeout_s": 30}, {"provider": "openrouter", "model": "dots-studio/dots-3-note-preview:free", "effort": "none"}]}'` (keep the Super tier's own `temperature`: without it the card's 1.0 applies, not `MEMORY_SUMMARY_TEMPERATURE`). Tier fields: `provider`, `model`, and optionally `timeout_s`, `thinking`, `think` (Colab), `effort` (OpenRouter), `reasoning_budget`, `json_mode`, `max_tokens`, `temperature`, `dim` (embed), `steps` (image).
- **Validation never drops a tier silently:** invalid JSON, an unknown role or provider, a paid OpenRouter id (unless `OPENROUTER_ALLOW_PAID=true`), a retired id, or a model that cannot do its role (a text-only vision tier, an unsupported `json_mode`, an embed width other than `EMBEDDING_DIMENSIONS`) fails startup. A tier whose provider has no key is disabled with its reason; an id without a profile gets the provider's generic protocol and a config warning (an unprofiled Colab id fails startup).
- `FORCE_TIER=nvidia` (or `openrouter`, `colab`, `provider:model`) is **for testing only**: it disables every other tier so one path can be exercised end to end, and fails startup if it would leave the router, chat, vision or utility role without a tier.
- Model health (`src/services/model_health.py`) skips dead, cooling-down and over-budget models before sending a request, so a `429` can move to the next provider without waiting for a failed full response cycle. A daily check (`src/services/model_lifecycle.py`) marks a retired model dead before live traffic hits its 410, and records an OpenRouter id's `expiration_date` (read from `GET /api/v1/models`, the only place OpenRouter lists it): `dots-studio/dots-3-note-preview:free` is a preview that expires on 2026-12-31. A model that becomes unusable (retired, deprecated, not in a catalog, no endpoints) is logged at ERROR; `expiring` (a date alone, the model still answers) is logged at INFO and shown in `/health/detail`, with no alert; results live in memory, so each new instance logs every not-ok model at its first check, and an instance that lives on stays quiet until something changes.

Why one table, and how retired models are caught before users find them: [ADR-0007](docs/adr/0007-model-role-table-and-lifecycle.md). The per-model keys of earlier releases are no longer read (see [Configuration](#configuration)).

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
│ Endpoints: /webhook /health(/detail) /internal/cron            │
│            /internal/tasks/{kind} /internal/colab/register     │
│                                                                │
│ Task pipeline (webhook queues, Cloud Tasks calls back):        │
│   Input -> Enrich -> Orchestrator -> Agent x4 -> Output        │
│   proc.   rate+mem   (router)                   proc.          │
│                                                                │
│ Core engine:  FallbackChain + ModelHealth                      │
└────────────────────────────────────────────────────────────────┘
                                 ↓  fan-out to external services
┌────────────────────────────────────────────────────────────────┐
│ EXTERNAL AI PROVIDERS                                          │
├────────────────────────────────────────────────────────────────┤
│ NVIDIA API      LLM · vision · image gen · embeddings          │
│ Colab (opt.)    Ollama on A100: qwen3.8 + clef-flash router    │
│ OpenRouter API  LLM + vision fallback                          │
│ Tavily          web search + URL extract                       │
│ edge-tts        voice synthesis (TTS)                          │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ STATE & STORAGE                                                │
├────────────────────────────────────────────────────────────────┤
│ Firestore Native  memory · profile · episodes · Colab state    │
│ Cloud Storage     images · audio (signed URLs)                 │
│ Secret Manager    linebot-dotenv (.env, mounted as a file)     │
│                   linebot-colab-adc (only with COLAB_ENABLED)  │
│ In-memory/inst.   quoted msgs · rate state · model health      │
└────────────────────────────────────────────────────────────────┘

Cloud Tasks  ──(per event · X-Tasks-Token)──→ POST /internal/tasks/{kind}
Cloud Scheduler ──(1 call / min · X-Cron-Token)──→ POST /internal/cron
Colab VM ──(HMAC-signed, optional)──→ POST /internal/colab/register
```

| Component | Responsibility | Key implementation |
| --- | --- | --- |
| FastAPI endpoints | Webhook intake, health, cron trigger, Cloud Tasks target, Colab registration | [main.py](main.py) |
| Event pipeline | `process_event(event, deps)`: one event → one reply, collaborators injected through `PipelineDeps` ([ADR-0005](docs/adr/0005-composition-root-and-injected-pipeline.md), [ADR-0012](docs/adr/0012-stay-custom-langgraph-seam.md)) | [src/pipeline/event_pipeline.py](src/pipeline/event_pipeline.py) |
| Task handlers | One handler per task kind (`event`, `passive_flush`, `cache_image`, `compact`, `eol`, `colab`) | [src/pipeline/task_handlers.py](src/pipeline/task_handlers.py) |
| Task queue | Cloud Tasks REST client or inline runner; `colab` tasks go to their own queue | [src/services/task_queue.py](src/services/task_queue.py) |
| Input processor | Decode event, download image, sanitize (strip injection markers), recover quoted context | [src/processors/input_processor.py](src/processors/input_processor.py) |
| Webhook handler | Trigger detection (`!hej` / @mention / DM), memory enrichment, rate limit | [src/handlers/webhook_handler.py](src/handlers/webhook_handler.py) |
| Orchestrator | Choose agent + output format + thinking flag | [src/agents/orchestrator.py](src/agents/orchestrator.py) |
| Specialist agents | `chat`, `vision`, `web_search`, `image_gen` | [src/agents/](src/agents/) |
| Model role table | Which models serve each role, in order; validated at startup | [src/runtime/model_roles.py](src/runtime/model_roles.py) |
| Fallback chain | Per-tier deadlines, model health and budgets, multi-provider retry | [src/providers/fallback_chain.py](src/providers/fallback_chain.py) |
| Providers | NVIDIA + OpenRouter OpenAI-compatible clients, Colab Ollama (`/api/chat`) and clef-flash (`/v1/systemone`) | [src/providers/](src/providers/) |
| Colab tier | Session state, registration, busy gates, the control plane and its lifecycle controller | [src/services/colab/](src/services/colab/), [colab/](colab/) |
| Output processor | Text/voice/image delivery; plain text for LINE, then OpenCC s2t | [src/processors/output_processor.py](src/processors/output_processor.py), [src/utils/line_text.py](src/utils/line_text.py) |
| Memory service | Layered chat memory, episodic recall, user profile | [src/services/memory_service.py](src/services/memory_service.py) |

### 2. End-to-end request lifecycle

The webhook returns `200 OK` **before** any model is called, well inside LINE's 2-second limit. Cloud Run only gives CPU while a request is running, so the heavy work is not left running after the response: it is queued as a Cloud Task, which calls back `/internal/tasks/event` as a request of its own. If queueing fails, the webhook does the work inline instead of dropping it. See [ADR-0006](docs/adr/0006-cloud-tasks-post-webhook-work.md).

```text
┌────────────────────────────────────────────────────────────────┐
│ 1-2  User  ->  LINE platform  ->  POST /webhook                │
│              (+ X-Line-Signature)                              │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 3  FastAPI /webhook                                            │
├────────────────────────────────────────────────────────────────┤
│ validate HMAC-SHA256 signature · drop duplicate webhookEventId │
│ queue Cloud Task "event"  ->  return 200 OK to LINE            │
└────────────────────────────────────────────────────────────────┘
                                 ↓  POST /internal/tasks/event (X-Tasks-Token)
┌────────────────────────────────────────────────────────────────┐
│ 4  Input processor                                             │
├────────────────────────────────────────────────────────────────┤
│ download image · sanitize · strip injection · recover quote    │
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
│ else clef-flash (Colab READY)  ->  RouterDecision              │
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
│ 8  Output processor   (plain text, then OpenCC s2t)            │
├────────────────────────────────────────────────────────────────┤
│ reply_token (free)  ->  push (only if the reply failed)        │
│ text · voice (TTS -> GCS) · image (GCS signed URL)             │
└────────────────────────────────────────────────────────────────┘
                                 ↓  deliver
┌────────────────────────────────────────────────────────────────┐
│ 9-10  LINE delivers reply -> user                              │
├────────────────────────────────────────────────────────────────┤
│ record_interaction; window full -> queue "compact" task        │
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
│ draw / generate request   ->  image_gen                        │
│ URL detected              ->  web_search                       │
│ lookup topic / time words ->  web_search                       │
│ short follow-up (≤12 ch.) ->  previous agent (vision -> chat)  │
│ explicit voice / text     ->  chat + voice / text              │
│ bare greeting             ->  chat                             │
│ no rule matched           ->  clef-flash, then LLM classifier  │
└────────────────────────────────────────────────────────────────┘
                                 ↓  only ambiguous text
┌────────────────────────────────────────────────────────────────┐
│ ROUTER Colab clef-flash:9b on /v1/systemone · only when READY  │
├────────────────────────────────────────────────────────────────┤
│ scores agent · output_format · needs_thinking (one call)       │
│ top choice of each (argmax)       ->  RouterDecision           │
│ busy · not READY · error          ->  LLM classifier           │
│ image_gen without image output    ->  LLM classifier           │
│ (MODEL_PATH_PREFERENCE=api: asked only if the classifier fails)│
└────────────────────────────────────────────────────────────────┘
                                 ↓  no clef-flash decision
┌────────────────────────────────────────────────────────────────┐
│ LLM CLASSIFIER   Nemotron 3 Super 120B · temp 0.1 · 256 tok    │
├────────────────────────────────────────────────────────────────┤
│ outputs one JSON object (response_format json_schema):         │
│   agent · output_format · needs_thinking · task · reason       │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ ROUTER DECISION  ->  adaptive thinking control                 │
├────────────────────────────────────────────────────────────────┤
│ reasoning-heavy   (debug · traceback · proof · planning ·      │
│ long-form)        ->  thinking role, then chat (thinking off)  │
│                                                                │
│ default / short   (greeting · simple Q)                        │
│                   ->  chat role · Nemotron 3 Super, no thinking│
└────────────────────────────────────────────────────────────────┘
```

- **Fast-rule signals** are kept high-precision, because a match skips the router: image-generation verbs with an image noun (計畫 / 企畫 / 規畫 / 策畫, 漫畫推薦, withdraw and "draw a conclusion" do not count), attached or quoted images, URLs (a URL's host ends at the first CJK character; its path keeps CJK letters and ends at whitespace or CJK / full-width punctuation), strong lookup topics (weather, news, stock prices, exchange rates, fares, opening hours, flights, typhoons, 哪裡買, 怎麼去) or a time word next to one, and short follow-ups when the previous route is known. Generic words such as 推薦, 評價, 結果, 在哪, review, where or price are left to the router. Every verified misroute is a case in [tests/test_routing_rules.py](tests/test_routing_rules.py).
- **Output format** (`text` / `voice` / `image`) is decided in the same pass — only an explicit request (用語音回覆, 用語音告訴我, 唸出來, 念給我聽 / 念…給我聽, 朗讀, 說給我聽, read aloud, voice reply, tts) triggers TTS, negation keywords force text, image-gen always returns an image.
- **clef-flash on Colab:** with a READY Colab session, `clef-flash:9b` decides the agent, the output format and whether to think jointly, in one `/v1/systemone` call (English field and option descriptions that speak of "the state", the word its prompt shows; the state is the message clipped to 300 characters, with any `<` that opens a tag (`<|im_end|>`, `</think>`) and any `|>` broken so no special token can form, whether an image is attached or quoted, whether a message is quoted, and the previous agent: no memory). It takes the highest-probability choice of each (argmax; an exact tie goes to the alphabetically first option): `COLAB_ROUTER_CONFIDENCE` defaults to 0.0, because the uncalibrated probabilities put real messages at 0.39–0.43 on (p_max − 1/N) / (1 − 1/N) and a gate of 0.90 deferred every one; a non-zero value hands choices below it to the classifier. Thinking is on at p ≥ `COLAB_ROUTER_THINK_ON` (0.85), off at ≤ `COLAB_ROUTER_THINK_OFF` (0.15), and left to the rulebook in between. These gates are provisional; `scripts/evals/routing_eval.py` re-tunes them. Anything else goes to the classifier at once.
- **Router input:** the router reads text only (an image becomes a `[圖片]` marker, so the text-only router tiers stay usable) and gets the previous route as a fenced context block right before the message. If it picks `vision` with no image attached or quoted, the request goes to `chat`.
- **Thinking is opt-in:** the default path runs Nemotron 3 Super without reasoning for speed (after a READY Colab session's qwen3.8 at `COLAB_CHAT_THINK`); only clearly reasoning-heavy requests go to the thinking role (Colab qwen3.8 at `COLAB_DEEP_THINK` when READY, Super with thinking on, Ultra with thinking on at `medium_effort`, then OpenRouter dots-3 and Nemotron Ultra `:free` at medium effort, then the chat tiers with thinking off, skipping a model that already timed out).

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
│ (temperatures only for models without card sampling)           │
└────────────────────────────────────────────────────────────────┘
                                 ↓  each agent picks a fallback chain
┌────────────────────────────────────────────────────────────────┐
│ CHAT ROLE   (chat · web_search answers)                        │
├────────────────────────────────────────────────────────────────┤
│ 0  Colab qwen3.8 27B       (only when READY, 70 s)             │
│ 1  NVIDIA Nemotron 3 Super   (thinking off, 45 s)              │
│ 2  NVIDIA Nemotron 3 Ultra   (thinking off, 25 s)              │
│ 3  OpenRouter dots-3         (free, preview until 2026-12-31)  │
│ 4  OpenRouter Nemotron Ultra (free, served by NVIDIA)          │
│ (thinking requests run the thinking role first)                │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ UTILITY ROLE   (summaries · search queries · img stage 1)      │
├────────────────────────────────────────────────────────────────┤
│ 1  NVIDIA Nemotron 3 Super   (thinking off, temp 0.2)          │
│ 2  NVIDIA Nemotron 3 Ultra   (thinking off)                    │
│ 3  OpenRouter dots-3         (free, effort none)               │
│ 4  OpenRouter Nemotron Ultra (free, served by NVIDIA)          │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ VISION ROLE   (any request with an image)                      │
├────────────────────────────────────────────────────────────────┤
│ 0  Colab qwen3.8 27B     (READY, inline images, 70 s)          │
│ 1  NVIDIA Nemotron 3 Nano Omni (thinking off, 15 s)            │
│ 2  NVIDIA muse-glimmer 30B  (thinking off, 35 s)               │
│ 3  OpenRouter dots-3        (free, 20 s)                       │
│ (prompt context ≤ 3000 chars, about 4k tokens in all)          │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ IMAGE GENERATION   (NVIDIA Visual GenAI NIM)                   │
├────────────────────────────────────────────────────────────────┤
│ Stage 1 (text):  refine prompt (+ caption, never waited for)   │
│ Stage 2 (NIM) :  FLUX.2 klein 4B (4 steps) → FLUX.1-dev        │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ chat AGENTIC TOOL LOOP   (max 4 rounds, then tool_choice=none) │
├────────────────────────────────────────────────────────────────┤
│ recall_memory(query, k)     vector recall of episodes          │
│ update_user_profile(facts)  durable cross-chat facts           │
│ web_search(query)           Tavily, ≤2 per reply               │
└────────────────────────────────────────────────────────────────┘
```

**Model inventory (all roles & specs):**

| Role | Model | Provider | Params / context | Capabilities |
| --- | --- | --- | --- | --- |
| Chat default and thinking | Nemotron 3 Super 120B (`nvidia/nemotron-3-super-120b-a12b`) | NVIDIA | 120B MoE / 12B active · 1M | text only, tools, reasoning |
| Text fallback #1, thinking fallback | Nemotron 3 Ultra 550B (`nvidia/nemotron-3-ultra-550b-a55b`) | NVIDIA | 550B MoE / 55B active · 1M | text, tools, reasoning |
| Text fallback #2, thinking and utility fallback | dots-3 note preview (`dots-studio/dots-3-note-preview:free`) | OpenRouter (upstream AtlasCloud) | — | vision, tools, reasoning; a preview, free until 2026-12-31 |
| Text fallback #3, thinking and utility fallback | Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free`) | OpenRouter (served by NVIDIA) | 550B MoE / 55B active | text only, tools, reasoning |
| Router, first API tier (after Colab clef-flash when READY) | Nemotron 3 Super 120B (`nvidia/nemotron-3-super-120b-a12b`) | NVIDIA | 120B MoE / 12B active · 1M | tools, reasoning |
| Router fallbacks | Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`, JSON by prompt) → Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b:free`, `json_object`) | NVIDIA → OpenRouter | 550B MoE / 55B active; 120B MoE / 12B active | tools, reasoning |
| Vision primary | Nemotron 3 Nano Omni (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) | NVIDIA | 30B MoE / 3B active | vision, reasoning (off here) |
| Vision fallbacks | muse-glimmer 30B (`meta/muse-glimmer-30b`) → dots-3 note preview (`dots-studio/dots-3-note-preview:free`) | NVIDIA → OpenRouter | 30B; — | vision (thinking off) |
| Utility (summaries, search queries, image prompts) | Nemotron 3 Super 120B → Nemotron 3 Ultra → dots-3 note preview (`:free`) → Nemotron 3 Ultra (`:free`) | NVIDIA → OpenRouter | — | background work, thinking off |
| Image generation | FLUX.2 klein 4B (`black-forest-labs/flux.2-klein-4b`) → FLUX.1-dev (`black-forest-labs/flux.1-dev`) | NVIDIA Visual GenAI NIM | 4 steps / 50 steps | image_gen |
| Episodic embeddings | `nvidia/nemotron-3-embed-1b` | NVIDIA | 2048-dim, first 1024 kept | multilingual embeddings |
| Colab tier (optional; chat, thinking, vision) | Qwen 3.8 27B MTP (`qwen3.8:27b-mtp-q4_K_M`) | Colab A100, Ollama ≥ 0.40.4 | 27B, q4_K_M | vision, tools, think levels |
| Colab router (optional) | Cloudflare clef-flash (`clef-flash:9b`, Q8_0, `/v1/systemone`) | Colab A100, Ollama ≥ 0.40.4 | 9.1B (from Qwen3.5-9B) | joint routing decisions |

### 5. Provider fallback chain & deadlines

The fallback chain is where resilience lives. It walks model tiers in order, gives each one its own time limit inside the request's deadline, skips models that cannot take a request, and falls through on every error, timeout and empty reply.

```text
┌────────────────────────────────────────────────────────────────┐
│ agent: tiers + deadline                                        │
├────────────────────────────────────────────────────────────────┤
│ deadline = reply window (event time + 90 s), every request     │
│   thinking or image (any chat), or < MIN_TIER_SECONDS left     │
│   ->  turn deadline (work start + 240 s)                       │
│ every answer: reply first; LINE refuses it  ->  push           │
│ thinking ON ->  [thinking role] + [chat role, thinking off]    │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ FOR EACH tier — in priority order                              │
├────────────────────────────────────────────────────────────────┤
│ deadline passed      ->  stop (DeadlineExceededError)          │
│ image but text-only  ->  skip (not attempted)                  │
│ ModelHealth: dead / cooling down / over budget ->  skip        │
│ model timed out earlier in this walk           ->  skip        │
│ run under min(tier timeout, time left)                         │
│   (thinking tiers together: at most half; a Colab tier:        │
│    until 20 s before the deadline, or half if later)           │
│              429 / error / timeout / empty ->  next tier       │
│              success        ->  return ProviderResponse        │
│                                                                │
│ all tiers exhausted  ->  AllModelsRateLimitedError /           │
│                          AllProvidersFailedError               │
└────────────────────────────────────────────────────────────────┘
```

**Reply window rules** ([src/agents/base_agent.py](src/agents/base_agent.py) `_deadline_for`):

- Every plain answer's deadline is the LINE event time + `REPLY_WINDOW_SECONDS` (90 s), in 1:1 chats and groups alike, whichever tier answers it: a READY Colab tier (which stops 20 s before it, see below) or the API tiers. It must stay below `TURN_DEADLINE_SECONDS` (startup refuses otherwise).
- A thinking answer or an image, in any chat, may run to the turn deadline, `TURN_DEADLINE_SECONDS` (240 s) counted from when work on the event starts. So does a plain answer that starts with less than `MIN_TIER_SECONDS` of the window left: after a late or redelivered event (the task logs that the window had passed), or after slow routing used the window up (the Colab router, then the API router tiers and a repair prompt, each held only to the window).
- Delivery is always the free reply first, however long the answer took: LINE only guarantees a reply token for about a minute, but only LINE knows whether it still works. Only when the reply certainly failed (a 4xx such as an expired or invalid token, or a request that never left) does the answer go out by push. No push is counted, capped or refused: LINE's own monthly quota is the only limit (a push to a group bills every member). A send whose outcome is unknown counts as sent and is never pushed again.
- Inside the deadline each tier gets min(its `timeout_s`, the time left). The thinking tiers together get at most half of it (they must be done halfway to the deadline, so the non-thinking tiers after them keep the other half), and a Colab tier with a usable tier after it must stop 20 s before the deadline (`COLAB_API_RESERVE_SECONDS`), or halfway to it when that is later. An image that needs thinking lets Colab think (the API vision tiers never do), then runs every vision tier with thinking off; a Colab tier the thinking pass already tried is not tried again with thinking off. Web search, and in chat's tool loop every tool call and every tool-offering round after the first, must finish 20 s before the deadline, so the answer keeps its time (a tool that runs out of time returns an error and the model answers with what it has; the first round, usually the whole answer, has the whole deadline); Colab never makes a reply wait (its tiers are skipped until the session is READY).

Providers map every HTTP status onto one error class (`src/providers/base.py`): 429 cools the model down for its Retry-After or 60 s (an OpenRouter 429 that names the account's free-model cap, `free-models-per-day` / `-per-min`, cools every `:free` id for 10 minutes; one passed on from a model's upstream cools only that model), 401, a 403 whose body names the key, and 402 cool the provider down, while any other 403 (a function the key may not call, a moderation refusal) cools only that model for two minutes (a refused Colab bearer is transient instead: the session's health check re-validates it, [ADR-0010](docs/adr/0010-colab-ollama-tier.md)), 410 takes the model out for the process and 404 for six hours (ten minutes on OpenRouter; its 404 "No endpoints found that ...", which names a capability such as tool use or image input, only moves that request to the next tier), 400/422 are never resent, and 5xx, timeouts and bad JSON are transient; two transient failures in a row cool a model down for two minutes. A `200` whose JSON body holds an `error` object, at the top or on its first choice, is mapped by that object's `code` exactly as the status would be (a code that is missing or not a number is transient, as is `finish_reason` "error" with no error object), even beside partial output: OpenRouter can report an upstream failure only that way once it has routed a request, and on 2026-10-16 its Ultra `:free` answered `200` with `{"error": {"message": "Upstream error from Nvidia: Service temporarily overloaded", "code": 503}}`, which the chain had taken for an empty completion, one that never cools a model. A timeout counts when the tier had its whole own time limit, or at least `MIN_TIER_SECONDS` before the deadline cut it off; the timeout of a tier thinking for a request that needs thinking never counts, nor does one cut short by the search query planner's own 8 s budget (a slow rewrite says nothing about the router and summaries that share those utility tiers). Separately, a model that timed out after at least `MIN_TIER_SECONDS` is skipped by a later tier of the same walk (`skip(timed out earlier)`), thinking or not, when it had its tier's whole time limit or the later tier would give it less than `MIN_TIER_SECONDS` more; so the thinking half cutting Ultra's thinking tier a little short still spares its thinking-off tier, while dots-3 cut to 15 s of the half (a05be403) still gets its 40 s thinking-off try. Chat's tool loop shares one such record across its rounds, its last request and the plain retry after it; the next request calls the model again. NVIDIA's 202 answers are polled until the request's time runs out. OpenRouter sends a request once more, after 1 s, when it answers 502 or 503 (as a status or as the `code` of an error in a `200` body) saying the model's upstream was unavailable or overloaded: dots-3 has a single upstream, and on 2026-10-16 it answered a `200` whose error object had code 502 (`{"error": {"message": "JSON error injected into SSE stream", "code": 502, "metadata": {"error_type": "provider_unavailable"}}}`) where a retry a few seconds later answered. One retry only, only while the model is not cooling down and the free budget has a request left (the retry counts as one), and within the tier's time limit, which cancels it like any slow call; 4xx, 429, timeouts and a blocked paid id are never retried.

Request shapes come from the exact-model-id profiles (`src/providers/model_profiles.py`) plus each tier's options: every profiled NVIDIA model says explicitly whether to think (they all think by default), either with `chat_template_kwargs.enable_thinking` true/false (Ultra adds `medium_effort` when thinking) or, on muse-glimmer, a top-level `reasoning_effort` none/medium; Nano Omni (vision) always gets `enable_thinking` false, `max_tokens` 1024 and no `top_p`, and its busy HTTP 503 (`ResourceExhausted`, under a second) moves on to muse-glimmer at once. Thinking adds the budget (the tier's `reasoning_budget`, else `NVIDIA_THINKING_BUDGET`) to `max_tokens`, and Super also gets it as a top-level `reasoning_budget`; Ultra answers any `reasoning_budget` with HTTP 400, so it never gets one. Temperature follows the sampling order above, and temperature / `max_tokens` are clamped to each model's range. Every response logs its prompt, completion and reasoning token counts. JSON output follows the tier's `json_mode`: `response_format` json_schema on Super, json_object on the Super `:free` router fallback, the prompt alone elsewhere (Nemotron 3 Ultra has no structured output on either provider, so its router tier sends no `response_format` and the orchestrator's tolerant parser, with one repair prompt, reads the JSON). OpenRouter calls carry an explicit `reasoning` object where the model needs one (`{"effort": "none"}` when thinking is off, because dots-3 and the Nemotron `:free` ids reason by default: without it dots-3 spent 280 reasoning tokens on a short answer, with it none, live 2026-10-16).

### 6. Layered memory & episodic recall

Each chat scope keeps a small rolling window plus one long-term summary. When the window fills, a queued `compact` task compacts it into the next summary and, at most once an hour per chat, writes an embedded episode for later vector recall. A cross-chat user profile lives independently and survives `!new`.

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
│ window full  ->  "compact" task (utility chain, debounced)     │
│              ->  rewrites summary, removes summarized msg ids  │
│              ->  episode (at most 1/hour; else queued text)    │
└────────────────────────────────────────────────────────────────┘
                                 ↓  embed conversation + summary
┌────────────────────────────────────────────────────────────────┐
│ EPISODIC MEMORY                                                │
├────────────────────────────────────────────────────────────────┤
│ episode snapshots + nemotron-3-embed-1b (1024 of 2048 dims)    │
│ queried on demand by chat tool: recall_memory(query, k)        │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ CROSS-CHAT PROFILE   user_profile/{user_id}                    │
├────────────────────────────────────────────────────────────────┤
│ durable facts shared across DMs & groups                       │
│ written by tool: update_user_profile() · survives !new         │
│ newest 10 facts shown to chat / vision / web_search            │
└────────────────────────────────────────────────────────────────┘

Backend:  InMemoryBackend (default, resets on cold start)  |  FirestoreBackend (opt-in)
```

- **Recent window:** the text messages since the last compaction (compaction starts at `MEMORY_RECENT_MESSAGE_LIMIT`, default 6), in the order LINE received them, fed verbatim into the prompt.
- **Long-term summary:** one paragraph of short facts, at most 400 characters, biased toward preserving still-valid ones: the utility tier answers only with the old items to remove and the new ones, the code applies that (an answer that ignores the format and would lose old items moves on to the next tier), and a longer result keeps the pass's new items and the old items a utility tier ranks most important, as many as fit ([ADR-0003](docs/adr/0003-recent-window-is-a-trigger.md)); regenerated in its own Cloud Task so it never blocks a reply.
- **Episodic recall:** compactions are archived as embedded episodes (conversation + summary), at most one per chat per hour; the chat agent can vector-search older context on demand via `recall_memory`. Each vector is tagged with its embedding model and recall ignores other models' vectors until `scripts/migrate_embeddings.py` re-embeds them ([ADR-0011](docs/adr/0011-embedding-migration.md)).
- **User profile:** durable facts (nickname, hobbies, location) shared across all chats and shown to the answering agents; intentionally **not** cleared by `!new`.
- **Backends:** `InMemoryBackend` for free/local runs, `FirestoreBackend` for persistence across Cloud Run cold starts and multiple instances.

### 7. Deployment & delivery pipeline

```text
┌────────────────────────────────────────────────────────────────┐
│ Once   -   ./scripts/provision_gcp.sh (dry run, then --apply)  │
│ service accounts · staging bucket · secret · AR cleanup        │
│ Colab (optional) - ./scripts/mint_colab_adc.sh --apply         │
│ the Colab token -> secret linebot-colab-adc, runtime can read  │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ Developer   -   ./scripts/deploy_cloud_run.sh                  │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ Local guards   check-v3 · media bucket · upload leak guard     │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ Cloud Build (linebot-builder, us-west1 staging bucket)         │
│ ruff + pytest -> build -> push (no secrets)                    │
└────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ Artifact Registry   (image; cleanup policy keeps 2)            │
└────────────────────────────────────────────────────────────────┘
                                 ↓  .env -> Secret Manager version (if changed)
                                 ↓  gcloud run deploy (local, pinned version)
┌────────────────────────────────────────────────────────────────┐
│ Cloud Run revision (linebot-runtime)                           │
├────────────────────────────────────────────────────────────────┤
│ route traffic -> smoke check -> prune revisions / images /     │
│ secret versions                                                │
│ signed URLs -> Cloud Storage  ·  optional -> Firestore         │
└────────────────────────────────────────────────────────────────┘

Cloud Tasks queue linebot-events ──→ POST /internal/tasks/{kind}
Cloud Tasks queue linebot-colab (COLAB_ENABLED) ──→ POST /internal/tasks/colab
Cloud Scheduler ──(1 / min · X-Cron-Token)──→ POST /internal/cron
```

One command (`./scripts/deploy_cloud_run.sh`) refuses retired models, a media bucket outside the Always Free locations and secret uploads, enables APIs, has Cloud Build test (on the same digest-pinned `python:3.12-slim` base as the image), build and push the image, deploys it from your machine, routes traffic to the new revision, runs smoke checks, and prunes old revisions/images — no local Docker build required.

Settings reach Cloud Run as one Secret Manager secret, `linebot-dotenv` (`.env` without the deploy-only keys), mounted at `/secrets/app/app.env` and pinned to its version number, so each revision rolls back with its own settings; the only plain variable is `APP_VERSION`. With `COLAB_ENABLED=true` the Colab account's token (`linebot-colab-adc`) is mounted the same way at `/secrets/colab/adc.json`. The service runs as `linebot-runtime` and builds run as `linebot-builder`, each with only the roles it uses, and Cloud Build stages the source in a regional bucket inside the free tier. How the deploy uploads, mounts, prunes and falls back: [DEPLOY.md](DEPLOY.md), "標準部署"; why: [ADR-0008](docs/adr/0008-secrets-and-identity.md).

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Runtime | GCP Cloud Run |
| Framework | FastAPI + Uvicorn |
| Text LLM (default) | NVIDIA Nemotron 3 Super 120B (non-thinking) |
| Text reasoning LLM | NVIDIA Nemotron 3 Super, then Ultra, with thinking on |
| Router / utility LLM | NVIDIA Nemotron 3 Super 120B |
| Vision LLM | NVIDIA Nemotron 3 Nano Omni, then muse-glimmer 30B, then OpenRouter dots-3 (free) |
| Fallback LLMs | NVIDIA Nemotron 3 Ultra (chat, thinking, router, utility); OpenRouter `dots-studio/dots-3-note-preview:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`, `nvidia/nemotron-3-super-120b-a12b:free` (role-dependent) |
| Image generation | NVIDIA FLUX.2 klein 4B, FLUX.1-dev fallback |
| Embeddings | NVIDIA `nemotron-3-embed-1b` |
| Self-hosted LLM (optional) | Ollama (latest release ≥ 0.40.4) on a Colab A100 (`qwen3.8:27b-mtp-q4_K_M`, `clef-flash:9b`), reached through a Cloudflare tunnel |
| Web search | Tavily (basic depth; key or keyless); page text via trafilatura |
| Voice | `edge-tts` |
| Object storage | Google Cloud Storage |
| Scheduling | Cloud Scheduler → `/internal/cron` |
| Post-webhook work | Cloud Tasks (`linebot-events`; `linebot-colab` for the Colab controller) → `/internal/tasks/{kind}` |
| Text conversion | OpenCC |
| Image processing | Pillow |
| Delivery pipeline | Cloud Build -> Artifact Registry -> Cloud Run |
| Dependencies | `pyproject.toml` + `uv.lock` (uv; every file sha256-pinned) |

---

## Quick start

```bash
# 1. Copy the settings template and fill in the keys listed under Configuration below
cp .env.example .env

# 2. First-time GCP login
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. One-time GCP setup: read the dry run, then apply step by step (DEPLOY.md)
./scripts/provision_gcp.sh
./scripts/provision_gcp.sh --step apis --apply   # ... and so on, see --list

# 4. Deploy (one command does everything; run it again after every change)
./scripts/deploy_cloud_run.sh

# 5. Set your LINE webhook URL to:
#    https://YOUR_CLOUD_RUN_URL/webhook
```

Going live for the first time, or coming from v2: follow the checklist "上線前檢查清單（需要你本人操作）" at the top of [DEPLOY.md](DEPLOY.md). It includes the steps only the owner can do (rotating the leaked keys, the budget alert, LINE webhook redelivery, the optional Colab runs, removing `roles/editor` last), and its section "從 v2 升級到 v3" has the details of the upgrade. DEPLOY.md also covers the deploy options, logs, rollback, Firestore and troubleshooting.

Local checks run on the image's Python (3.12), from the same `uv.lock` Cloud Build installs (editing and upgrading dependencies: DEPLOY.md, "依賴鎖定與本機驗證"):

```bash
UV_PROJECT_ENVIRONMENT=.venv312 uv sync --locked -p 3.12
.venv312/bin/python -m ruff check . && .venv312/bin/python -m pytest -q
```

Optional but recommended before a deploy: `.venv312/bin/python scripts/live_gate.py` sends a few requests through the bot's own clients with your keys and prints PASS/FAIL per model path (never keys or full replies); `--path tiers` calls every tier of the configured role table once and prints a role / tier / model / result / latency table.

---

## Configuration

Every setting lives in `.env`, and [.env.example](.env.example) is the reference for all of them: each key with its default and what it does (the comments are in zh-TW). The keys to know first; everything else has a working default:

| Key | Why |
| --- | --- |
| `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN` | The LINE Messaging API channel; the deploy stops without them |
| `OPENROUTER_API_KEY` | The free fallback tiers of every text role; the deploy stops without it |
| `NVIDIA_API_KEY` | Recommended: the first API tier of every role, and the only path to image generation and episodic-recall embeddings. Readiness needs this key or the OpenRouter one |
| `GCP_PROJECT_ID` | The deploy's project; stays on your machine |
| `INTERNAL_TASKS_SECRET` | `openssl rand -hex 32`; authenticates the Cloud Tasks callbacks. The deploy stops without it unless `TASKS_MODE=inline` |
| `INTERNAL_CRON_SECRET` | `openssl rand -hex 32`; without it there is no per-minute cron (scheduled messages, stale group-buffer flushes, the daily end-of-life check, the Colab controller) and `/health/detail` answers 403 |
| `GCS_BUCKET_NAME` | Needed for image and voice replies (text works without it): a STANDARD bucket in `us-west1`, `us-central1` or `us-east1`, the Always Free locations; the deploy refuses any other |
| `FIRESTORE_ENABLED` | `true` keeps chat memory, profiles and episodes across cold starts (setup in DEPLOY.md); the default is in-memory per instance |
| `TAVILY_API_KEY` / `TAVILY_KEYLESS_ENABLED` | Web search; without a key, Tavily's keyless mode runs only when enabled (the template enables it) |
| `COLAB_ENABLED` | The optional [Colab Ollama tier](#colab-ollama-tier); needs Firestore, `COLAB_SHARED_SECRET`, `INTERNAL_CRON_SECRET`, Cloud Tasks and the minted Colab token |

Models are not set key by key: they come from the [model role table](#model-role-table), overridden with `MODEL_ROLES`; the per-model keys of earlier releases (`NVIDIA_MODEL`, `ROUTER_MODEL`, `*_FALLBACK_MODEL`, …) are no longer read, and startup names any still set. Every deploy first runs `python3 scripts/envfile.py check-v3 --file .env`, which refuses retired model ids and the keys that would break (it prints key names, never values); `python3 scripts/envfile.py migrate-v3 --file .env` deletes those keys after backing `.env` up (a retired id inside `MODEL_ROLES` is yours to fix; the full list is at the end of `.env.example`). The deploy uploads `.env`, minus its deploy-only keys, as one Secret Manager secret ([ADR-0008](docs/adr/0008-secrets-and-identity.md)).

---

## Architecture decisions

The reasons behind the non-obvious choices are in [docs/adr/](docs/adr/):

| ADR | Decision |
| --- | --- |
| [0001](docs/adr/0001-pull-style-scheduling.md) | Scheduling is pulled by Cloud Scheduler hitting `/internal/cron` every minute; no in-process timers |
| [0002](docs/adr/0002-leap-day-yearly-jobs.md) | Yearly `02-29` jobs fire on `02-28` in other years |
| [0003](docs/adr/0003-recent-window-is-a-trigger.md) | The recent window triggers compaction; commits remove exactly the summarized messages, by id |
| [0004](docs/adr/0004-routing-excludes-long-term-memory.md) | Routing never sees the long-term summary or profile facts |
| [0005](docs/adr/0005-composition-root-and-injected-pipeline.md) | A composition root, and `process_event(event, deps)` with injected `PipelineDeps` |
| [0006](docs/adr/0006-cloud-tasks-post-webhook-work.md) | All work after the webhook's 200 runs as Cloud Tasks |
| [0007](docs/adr/0007-model-role-table-and-lifecycle.md) | One validated model role table; retired models are caught before users find them |
| [0008](docs/adr/0008-secrets-and-identity.md) | `.env` as one Secret Manager secret; least-privilege service accounts |
| [0009](docs/adr/0009-search-backends-and-ssrf-safe-fetching.md) | Tavily, then no search; SSRF-safe page fetching |
| [0010](docs/adr/0010-colab-ollama-tier.md) | A Colab Ollama tier, first for everyone, never waited on |
| [0011](docs/adr/0011-embedding-migration.md) | nemotron-3-embed-1b sliced to the 1024-d index; vectors tagged by model and migrated by script |
| [0012](docs/adr/0012-stay-custom-langgraph-seam.md) | No LangGraph for now; the seam kept for it and the triggers to adopt it |

---

## License

GNU General Public License v3 (GPLv3) — see [LICENSE](LICENSE).

## Contact

- Developer: Tom Huang
- Email: huang1473690@gmail.com
