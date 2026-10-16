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
then qwen3.8 on OpenRouter (15 s; dots-3 `:free` at 20 s since 2026-10-16,
see the entry below). A 39-call live probe on 2026-10-01 chose
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
pulls only the models versions.json lists (all profiled), and another id
would answer 404, which marks the whole Colab endpoint unhealthy, so it
fails startup. `/health` is ready only when some chat tier that does not
wait on Colab is enabled.

### MODEL_PATH_PREFERENCE

`colab` (the default) puts every role's Colab tiers first, `api` puts them
last (used only when Colab happens to be up). The order within the Colab and
the API tiers is the configured one. The router, chat, vision and utility
roles must keep at least one non-Colab tier, because a cold Colab must never
be what a reply waits on. Colab starts on demand and is released after 15
idle minutes (the upgrade plan's decision, confirmed 2026-09-30); until its
session is READY a Colab tier is skipped at once (`SkipTierError` costs no
time), so every message is answered straight away by the API tiers. The
router's Colab tier (clef-flash since 2026-10-16) is asked directly by the orchestrator, not
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
  `/api/v1/models/{id}/endpoints` plus their entry in `/api/v1/models`,
  read once per run (no endpoints, or an expiration date, is reported;
  OpenRouter lists `expiration_date` only in the model list, see the
  2026-10-16 entry below). State changes are logged at ERROR (a change to
  `expiring` or `ok` at INFO since a later entry); results are on
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
- The daily check spends about a dozen public GETs a day (one per id, plus
  OpenRouter's model list) and no quota.

## Alternatives considered

- **Keep the registry and filter, add more keys.** Every new need (a thinking
  tier with its own budget, JSON modes per endpoint, utility work) would have
  been another key per slot, and the silent-drop problem stays.
- **LangGraph or another orchestration framework.** Rejected for now (see
  ADR-0012): the problem was model selection and failure handling, which
  a table and the fallback chain solve without a new runtime.

## 2026-10-16: dots-3 replaces qwen3.8 `:free`, Super joins chat, errors in a 200 body

- **qwen3.8 `:free` is gone.** OpenRouter has listed no endpoint for
  `qwen/qwen3.8-27b:free` since 2026-10-13; a call answers HTTP 404 "This
  model is unavailable for free". It was the one OpenRouter tier not served
  by NVIDIA in chat, thinking and vision.
- **Gemma `:free` is still not an option.** `google/gemma-4-31b-it:free`
  and `google/gemma-4-26b-a4b-it:free` answered every probe of the day with
  HTTP 429 "temporarily rate-limited upstream" (`limit_source`
  `upstream_provider_shared_pool`), as on 2026-10-01.
- **dots-3 takes its place**: `dots-studio/dots-3-note-preview:free`
  (`OR_DOTS`), upstream AtlasCloud, not NVIDIA. Live probe: a tool call in
  2.0 s; the red-circle image as a data URI with reasoning effort "none" in
  1.7 s, 0 reasoning tokens, answered in Traditional Chinese; without a
  `reasoning` object it reasons (280 tokens), so "off" is sent explicitly
  (`thinking_default`); effort "medium" works. It takes `tools`,
  `tool_choice` "none" and "auto" only (the only two the tool loop sends),
  `response_format` and `structured_outputs`, text and image input. It is a
  preview: `GET /api/v1/models` lists `expiration_date` 2026-12-31. That
  date was invisible to the daily check, which read only
  `/models/{id}/endpoints` (no such key there), so the check now also reads
  the model list once per run and reports the id as `expiring` ahead of
  time. No endpoints is reported ahead of that date, so a withdrawn preview
  does not keep reading as merely expiring. A model list that cannot be
  read fails only the OpenRouter checks that would otherwise read ok (they
  keep their last state rather than lose an expiry); no endpoints, or a
  date in the endpoints answer, is still reported.
- **NVIDIA is two models in chat.** The order is Colab, then NVIDIA, then
  OpenRouter: Colab qwen3.8 → NIM Ultra (40 s) → NIM Super (thinking off,
  30 s, the card's sampling) → dots-3 (effort none, 40 s) → Ultra `:free`.
  The same day NIM Ultra answered HTTP 503 in 0.4 s while Super took
  19–51 s for one sentence; 30 s keeps time for the OpenRouter tiers inside
  the 90 s reply window even when Ultra hangs its whole 40 s. Super is not a
  thinking tier: callers append the chat tiers, thinking off, after the
  thinking tiers, so a thinking request reaches it anyway. Vision is
  unchanged except its last tier, dots-3 (effort none, 20 s). (Reordered
  after the deploy the same day; see the next entry.)
- **An error in a 200 body is that error.** Once it has routed a request,
  OpenRouter can report an upstream failure only in the body, as a
  top-level `error` object (alone, or beside partial output with
  `finish_reason` "error"), or as the same object on the choice itself
  (`NonStreamingChoice.error` in its schema). Ultra `:free` answered so
  during the outage (`"code": 503`, "Upstream error from Nvidia: Service
  temporarily overloaded"); taken for an empty completion, it earned no
  strike and every reply tried it again. `post_chat_completion` now raises
  the error the `code` maps to, with the same mapping as an HTTP status (a
  missing or non-numeric code is transient with status 200, as is
  `finish_reason` "error" with no error object), so two in a row cool the
  model for two minutes, and a body 429 still tells the account cap from an
  upstream limit by its `message` and `metadata.raw`.

## 2026-10-16, after the deploy: Super before Ultra, dots-3 in utility, no second call to a timed-out model

- **Ultra hangs; Super answers.** In production NIM Ultra timed out on
  5 of 5 calls (chat 40 s, thinking 100 s, utility 30 s); the day before
  it had answered 500/503. A live A/B through the bot's `NvidiaProvider`
  with a ~5K-token prompt: Ultra 0/4 with thinking off (60 s timeouts) and
  0/2 with it on (120 s); Super 4/4 with thinking off in 12-52 s
  (450-1121 output tokens) and 3/3 with it on in 22-50 s (809-1666
  tokens), as its profile builds the request (`enable_thinking`, a
  top-level `reasoning_budget`). In production Super's short answers took
  1-3 s and an 810-token answer 26.7 s; one 8K-token prompt at 3072 max
  tokens timed out at 30 s. dots-3 answered short questions in 4-5 s.
- **The worst request** (a05be403, web search with thinking, 215.6 s):
  the thinking tier on Ultra timed out at 100 s (not held against it, as
  thinking timeouts never are), dots-3 thinking got 15 s of the thinking
  half and timed out, then the chat tiers with thinking off called Ultra
  again (40 s), Super timed out at 30 s, and dots-3 answered in 20 s.
- **NVIDIA order, Super first.** The order rule stays Colab, then NVIDIA,
  then OpenRouter, and NVIDIA keeps both models. chat: Colab qwen3.8
  (70 s) → Super (thinking off, 45 s) → Ultra (thinking off, 25 s) →
  dots-3 (effort none, 40 s) → Ultra `:free` (40 s). When both NVIDIA
  tiers run out, dots-3 has about 20 s of the 90 s reply window less the
  routing (and search) time, and none when routing itself stalls on NIM
  (its Super and Ultra tiers, 12 s each; Ultra 6 s since the next
  entry); after two such timeouts
  `ModelHealth` cools the model down and dots-3 gets its 40 s. A Super
  answer of over about 1000 tokens (1121 tokens took 52 s in the A/B)
  runs past 45 s and counts as a timeout. thinking: Colab qwen3.8
  (`COLAB_DEEP_THINK`, still xhigh) → Super (thinking on, 60 s) → Ultra
  (thinking on, 30 s) → dots-3 (effort medium, 120 s) → Ultra `:free`
  (effort medium, 30 s). The thinking tiers share at most half of the
  240 s turn deadline (under 120 s once routing is done): the NIM tiers
  use 90 s of it. dots-3 is the one tier NVIDIA does not serve, and its
  120 s is the largest half, so the half rather than its own limit ends
  it: when NIM fails fast (as on 10-15) it may think for the whole half
  (an 80 s limit would cut off a 90 s answer, and that timeout would then
  skip its thinking-off tier too, see below), and when both NIM tiers hang
  it gets what is left, under 30 s. Ultra `:free` gets 30 s like NIM
  Ultra.
- **utility gains dots-3.** Two memory summaries failed that day: Super
  hit `MEMORY_SUMMARY_MAX_TOKENS` (768, which the memory service rejects),
  then Ultra and Ultra `:free` both hung, and nothing outside NVIDIA was
  left. utility is now Super (t=0.2, 30 s) → Ultra (30 s) → dots-3 (effort
  none, 30 s) → Ultra `:free` (30 s): 120 s, inside the summary's one
  180 s deadline (`MEMORY_SUMMARY_TIMEOUT_SECONDS`). Ultra keeps 30 s: the
  work runs in the background, and two timeouts in a row cool it down.
- **A timed-out model is not called again in the same walk.** A timeout
  after at least `MIN_TIER_SECONDS` marks the model (provider and id) with
  the seconds it had. `FallbackChain` skips a later tier of the walk on
  that model, thinking or not, and records `skip(timed out earlier)` in
  the request's summary line, when the model had its tier's whole
  `timeout_s`, or when the later tier's `timeout_s` would give it less
  than `MIN_TIER_SECONDS` more than it had. So the thinking tier on Ultra
  timing out spares the chat tier on Ultra, also when a search stage let
  the thinking half cut it to 20-30 s; in a05be403 the 15 s of the
  half dots-3 got was not a fair try for its 40 s chat tier, which then
  answered. A single "mark after `MIN_TIER_SECONDS`" rule would have
  skipped that answer; a "mark only a whole `timeout_s`" rule would have
  waited on Ultra twice whenever the half cut it short. The marks last for
  the walk, and chat's tool loop passes one record to all its walks (its
  rounds, its last request, and ChatAgent's plain retry), so a round does
  not wait again on a model that hung in an earlier one; the next request
  calls the model again. `ModelHealth` (strikes, cooldown, thinking
  timeouts never held) is unchanged. A Colab tier was already tried only
  once per request (`Tier.shares_time`).
- **Accepted costs.** Thinking requests no longer cool a hung model down:
  their thinking timeouts are not held, and the thinking-off tier that
  used to earn the strike is now skipped; plain requests and utility work
  that reach the model still do. When NIM fails fast and dots-3 thinks
  through the whole half without answering, its thinking-off tier is
  skipped too (it had far longer than 40 s), and Ultra `:free` with
  thinking off is the last try. When the half leaves Ultra's thinking
  tier 20 s or less (routing and search took over about 80 s, which the
  search stage's own deadline allows only for a late event), Ultra with
  thinking off is still tried for its 25 s.

## 2026-10-16, evening: Ultra's router tier at 6 s, one OpenRouter retry, expiry at INFO

- **Router: Ultra 12 s → 6 s.** Ultra routes in about 2 s when healthy.
  That afternoon a short follow-up (a mistyped "search and check again",
  7,887-token prompt) waited 12 s on Super (it normally routes in 2-4 s),
  then 12 s on the hung Ultra, and OpenRouter Super `:free` answered in
  1.2 s. Super keeps 12 s and the order is unchanged (clef-flash → Super →
  Ultra → Super `:free`); a 6 s timeout is still a whole `timeout_s`, so
  two in a row cool Ultra down as before.
- **OpenRouter: one retry of an unavailable upstream.** dots-3 has one
  upstream (AtlasCloud). It answered a memory summary, after 15 s, with a
  200 whose error object had code 502 (`{"error": {"message": "JSON error
  injected into SSE stream", "code": 502, "metadata": {"error_type":
  "provider_unavailable"}}}`, as `error_for_body` logs it), and
  failed once so in a live gate run, where retries a few seconds later
  passed. `OpenRouterProvider` now sends the same request once more after
  1 s when a 502 or 503 (a status, or the `code` of an error in a 200
  body) says the upstream was unavailable or overloaded, if `ModelHealth`
  still lets the model take a request; the retry counts against the free
  daily budget like any request. A request and its retry earn at most one
  strike (only the retry's error reaches the chain), so a cooldown takes
  two such requests, four failed answers. 4xx, 429, timeouts and a blocked
  paid id are never retried. The provider does not know the tier's time
  left: the chain's limit cancels a retry that runs past it, which then
  reads as a timeout of the tier rather than as its 502.
- **Lifecycle: an expiry date is information, not an alert.** The check
  logged dots-3's 2026-12-31 as an ERROR on every new instance, 2.5 months
  ahead. dots-3 is a backup and the operator wants no expiry reminder, so
  a change to `expiring` (a date alone: the model still answers) is now
  logged at INFO, whatever the date says, and stays on `/health/detail`; a
  change back to `ok` is INFO too. A change to a state in which the model
  is unusable (retired, deprecated, not in a catalog, no endpoints) keeps
  ERROR, so the alert comes when the model is gone, not ahead of it (a
  WARNING with one ERROR 14 days ahead was tried first and dropped).
  Results and the day guard live in memory, so the check runs at each
  instance's first cron tick and logs every not-ok id then (a deploy, or
  Cloud Run replacing the instance: instances lived from under a day to
  five days in the 10-08..10-16 logs), while an instance that lives on
  logs only changes.
