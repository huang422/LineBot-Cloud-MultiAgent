# CONTEXT

Mental model for the LineBot multi-agent service. Read this before changing
`src/`. Decisions with non-obvious rationale live in [`docs/adr/`](docs/adr/).

## What this is

A LINE bot on Cloud Run that routes each inbound message to one of four
specialist agents, then replies as text, voice, or image. Free-tier
constraints shape most of the design: Cloud Run scales to zero, LINE push
messages are metered, and Firestore writes cost money.

## Request path

```
LINE webhook
  → main.webhook              signature check, returns 200 immediately
  → pipeline.process_event    background task (never blocks the webhook)
      → process_input         LINE event  → AgentRequest
      → enrich_request        rate limit + memory context
      → Orchestrator.route    fast regex rules, else LLM classification
      → <agent>.process       chat | vision | web_search | image_gen
      → send_response         reply-first, push only as fallback
      → record_interaction    persist the turn to memory
```

The webhook returns 200 before any work happens: LINE retries aggressively on
non-200, and a retry storm is worse than a dropped reply.

## Module map

| Area | Module | Notes |
|---|---|---|
| HTTP transport | `main.py` | lifespan, `/webhook`, `/health`, `/internal/cron` — and little else |
| Assembly | `src/runtime/composition.py` | builds providers, agents, targets, tool registry |
| Built app | `src/runtime/container.py` | `AppContainer`: what composition produced |
| Request handling | `src/pipeline/event_pipeline.py` | one event → one reply; deps injected via `PipelineDeps` |
| Routing (LLM) | `src/agents/orchestrator.py` | LLM classification + repair retry |
| Routing (rules) | `src/agents/routing_rules.py` | the regex rulebook; pure, no I/O |
| Agents | `src/agents/` | all extend `BaseAgent`, prompts load from `prompts/*.md` |
| Providers | `src/providers/` | NVIDIA + OpenRouter behind `FallbackChain` |
| Memory | `src/services/memory_service.py` | recent window + long-term summary + episodes |
| Write batching | `src/services/passive_buffer.py` | debounced coalescing; owns *when* to flush |
| Persistence | `src/services/memory_backends.py` | in-memory (tests) or Firestore |
| Scheduling | `src/services/scheduler_service.py` | pull-style, driven by Cloud Scheduler |
| JSON recovery | `src/utils/jsonish.py` | pulls a dict out of LLM prose |

## Invariants worth knowing

These are load-bearing and easy to break by accident:

1. **No in-process timers.** Cloud Run hibernates when idle, so APScheduler /
   `asyncio.sleep` timers cannot fire. Scheduled messages are *pulled* by Cloud
   Scheduler hitting `/internal/cron` every minute. See ADR-0001.

2. **The recent-message window is a compaction trigger, not a retention bound.**
   Crossing `memory_recent_message_limit` *schedules* summarization; it must not
   truncate, or messages are lost before they reach long-term memory. Only the
   hard safety cap trims, and it logs when it does. See ADR-0003.

3. **The orchestrator deliberately drops the long-term memory summary** from
   routing messages. It is useful for answering, but it distracts the classifier
   away from the required single-line JSON. See ADR-0004.

4. **Yearly `02-29` jobs clamp to `02-28` on non-leap years** so annual
   reminders fire every year. The clamp applies to both `next_run` and
   `matches`. See ADR-0002.

5. **Passive group messages are buffered and coalesced** before being written,
   purely to bound Firestore write volume — not for latency.

6. **`reply_token` is single-use and short-lived.** `send_response` tries reply
   first and only falls back to a (metered) push, gated by a monthly budget.

7. **The pipeline takes its collaborators as arguments**, never module globals.
   That is what lets tests drive the real code path without patching, and why
   production code needs no `hasattr` guards. See ADR-0005.

## Testing

`python3 -m pytest -q` — no network, no GCP credentials required. The
`InMemoryBackend` stands in for Firestore; seed it through
`service.backend_instance.store`.

`python3 -m ruff check .` must be clean. Config in `pyproject.toml`; the rule
set is deliberately narrow (correctness over style) and each `ignore` carries
its reason.
