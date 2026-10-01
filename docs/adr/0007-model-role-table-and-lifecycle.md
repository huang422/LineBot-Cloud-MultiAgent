# ADR-0007: A model role table, and protection against retired models

**Status**: Accepted

## Context

v2 picked models with one env key per slot (`NVIDIA_MODEL`,
`AGENT_FALLBACK_MODEL`, `ROUTER_FALLBACK_MODEL`, `IMAGE_GEN_STEPS`, ...), a
registry of known ids, and a filter (`REQUIRE_REASONING_MODELS`) that dropped
any target the registry did not flag as reasoning-capable. Three things went
wrong with that:

- **Silent drops.** The filter removed fallbacks without a word, so a
  deployment could run with one model left and look healthy. Request shapes
  came from substring matching (`"nemotron" in model`), so a similar-looking
  id got the wrong parameters.
- **No shape for new needs.** Thinking was "the first tier with thinking on",
  tied to whatever the chat list started with; there was nowhere to say
  "think on Ultra with a 4096-token budget, then qwen3.8 at medium effort".
  Background work (search keywords, image prompts) borrowed the chat list, so
  it spent the reply model's quota and time.
- **Retirement found by users.** NVIDIA retired `qwen/qwen3.5-397b-a17b` and
  `nvidia/nv-embedqa-e5-v5` with HTTP 410. The bot learned about it only when
  a user's request hit the 410, and memory recall stayed broken for weeks.

## Decision

### One declarative table

`src/runtime/model_roles.py` holds a `RoleTable`: for each **role** (router,
chat, thinking, vision, utility, embed, image) an ordered list of **tiers**
(provider, exact model id, time limit, options such as `thinking`, `think`,
`effort`, `reasoning_budget`, `json_mode`, `max_tokens`, `temperature`,
`dim`, `steps`).
The code defaults are the v3 decisions; `MODEL_ROLES` (JSON) replaces any
role wholesale. Agents never name a model: composition hands each one its
role's tiers, and `BaseAgent.targets_for(request)` picks per request (an
image: the vision tiers, and when it needs thinking first the ones not set
to thinking off (only Colab: the API vision tiers never think), then all of
them with thinking off; a thinking request: the
thinking tiers, then the chat tiers with thinking off (without a Colab tier
the thinking tiers already had); otherwise the chat
tiers). Background work (memory summaries, search query planning,
image-prompt refinement and captions) runs on the utility role.

Facts about each exact id (protocol, input modalities, whether it thinks by
default, which JSON modes and efforts it takes, token and temperature limits,
the model card's default sampling) live in `src/providers/model_profiles.py`.
The providers build payloads from the profile plus the tier's options;
nothing matches substrings. Some facts only a live call finds: on
2026-09-30 Nemotron Ultra on NIM answered any `reasoning_budget` with HTTP
400, so its profile thinks with `medium_effort` instead. On 2026-10-01 Gemma
left the defaults: NIM Gemma 4 31B hung on every live call and the
OpenRouter Gemma `:free` ids answered HTTP 429 on every one. Nemotron 3
Ultra took its places in router, chat and utility. Ultra is text only and
has no structured output on either provider (on NIM only Super takes
`json_schema`), so the router's NIM Ultra tier sets `json_mode` "prompt":
no `response_format` is sent, the routing prompt asks for the JSON object,
and the orchestrator's tolerant parser (with its one repair prompt) reads
it. Vision, which Ultra cannot do, runs (after Colab) Nemotron 3 Nano Omni
(`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, thinking off, 0.2,
`max_tokens` 1024, 15 s), then `meta/muse-glimmer-30b` (thinking off, 35 s),
then qwen3.8 on OpenRouter (15 s). A 39-call live probe on 2026-10-01 chose
that order: Nano Omni answered correct zh-TW in 1.8-6.3 s, or a busy HTTP
503 in under a second, which fails over at once; muse-glimmer took 8.5-19 s
at a 4k-token prompt but about 40 s at 6.2k. So an image prompt, whichever
agent answers, carries at most 3000 characters of history, summary and
profile facts (`IMAGE_CONTEXT_CHARS` in `src/agents/base_agent.py`), about
4k tokens in all.

Sampling is per model too: a tier's `temperature` wins, then the profile's
card values (Nemotron 3 on NIM: 1.0 / top_p 0.95), then the agent's
setting. Otherwise one `CHAT_TEMPERATURE` would reach every model of a
mixed role. The router tiers on Super and Ultra and the utility tier on
Super keep a low temperature on purpose (classification, search queries and
summaries should be repeatable).

### Validate at startup, never drop silently

`RoleTable.from_settings` fails startup (`RoleConfigError`) on invalid JSON,
an unknown role or provider, a paid OpenRouter id without
`OPENROUTER_ALLOW_PAID`, a retired id, or a model that cannot do its role (a
text-only vision tier, an unsupported `json_mode`, an embedding width other
than the index's). A tier whose provider is not configured (no key; Colab
without `COLAB_ENABLED`) stays in the table, disabled, with the reason
shown on `/health/detail`. An id without a profile still runs on the
provider's generic protocol, with a config warning, except on Colab: its VM
pulls only the pinned (profiled) models, and another id would answer 404,
which marks the whole Colab endpoint unhealthy, so it fails startup. `/health` is ready only
when some chat tier that does not wait on Colab is enabled.

### MODEL_PATH_PREFERENCE

`colab` (the default) puts every role's Colab tiers first, `api` puts them
last (used only when Colab happens to be up). The order within the Colab and
the API tiers is the configured one. The router, chat, vision and utility
roles must keep at least one non-Colab tier, because a cold Colab must never
be what a reply waits on. Colab starts on demand and is released after 15
idle minutes (the upgrade plan's decision, confirmed 2026-09-30); until its
session is READY a Colab tier is skipped at once (`SkipTierError` costs no
time), so every message is answered straight away by the API tiers. The
router's Colab tier (nimble) is asked directly by the orchestrator, not
through the chain, before the API router tiers (colab) or only after they
fail (api); `COLAB_ROUTER_ENABLED=false` disables it. See ADR-0010.

### FORCE_TIER, for testing only

`FORCE_TIER=nvidia` (or `openrouter`, `colab`, `provider:model`) disables
every tier that does not match, so one path can be exercised end to end
(`scripts/live_gate.py` reports the rest as SKIP). Because it is so easy to
leave on by mistake, a value that matches nothing, or that leaves the router,
chat, vision or utility role with no usable tier, fails startup.

### Retirement handling

- A 410 from any call marks the model dead for the process (with its
  end-of-life date); a 404 takes it out for six hours (ten minutes on
  OpenRouter), except OpenRouter's "No endpoints found that support ..."
  (tool use, image input), which is the request's problem and only moves
  on to the next tier. Chat, image and embedding calls all go through the same
  `ModelHealth`, which also enforces the provider budgets.
- Once per Asia/Taipei day, the first `/internal/cron` tick queues an `eol`
  task (`src/services/model_lifecycle.py`). It checks every configured id
  against public catalogs, without an API key: NVIDIA chat and embedding ids
  via `GET /v1/models/{id}` (200 live, and clears a dead mark; 410 retired,
  marked dead before traffic hits it; 404 not in the catalog), FLUX ids via
  their build.nvidia.com page (`isDeprecated` / `deprecationNotice`, since
  they are not in that catalog), and OpenRouter ids via
  `/api/v1/models/{id}/endpoints` (no endpoints or an expiration date is a
  warning). State changes are logged at ERROR; results are on
  `/health/detail`. The day guard is in memory, so a restart may check twice,
  which is harmless.
- `scripts/envfile.py check-v3` refuses a deploy whose `.env` still names a
  retired id, in a plain key or inside `MODEL_ROLES`.

## Consequences

- The per-model settings are gone (listed in `REMOVED_ENV_KEYS`): startup
  names any that are still set, and `envfile.py migrate-v3` deletes them.
  Changing a model means editing one JSON value, and a mistake fails the
  revision instead of degrading it quietly.
- The table is the one place to read which model answers what. It costs a
  profile entry per new id; an unprofiled NVIDIA or OpenRouter id works but
  only with the plain protocol (no thinking controls) until it gets one.
- The daily check spends about ten public GETs a day and no quota.

## Alternatives considered

- **Keep the registry and filter, add more keys.** Every new need (a thinking
  tier with its own budget, JSON modes per endpoint, utility work) would have
  been another key per slot, and the silent-drop problem stays.
- **LangGraph or another orchestration framework.** Rejected for now (see
  ADR-0012): the problem was model selection and failure handling, which
  a table and the fallback chain solve without a new runtime.
