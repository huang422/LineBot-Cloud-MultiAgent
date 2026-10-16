# ADR-0010: A Colab Ollama tier, first for everyone, never waited on

**Status**: Accepted. Implemented in Phase 8: the request path and
registration (8a), the control plane, lifecycle, credential handling and
live gate (8b), the VM package under `colab/` (8c). In service since
2026-10-01: on-demand sessions reach READY on a 40 GB A100
(`NVIDIA A100-SXM4-40GB`) and answer first; the checks still open after the
first runs are listed under Consequences ("Unverified until the Colab
runs") and in colab/README.md (首次實跑時確認). Amended 2026-09-30: the router model is
`nimble:9b`, with per-model gates and a VRAM check ("The router model",
"VRAM"). Amended 2026-10-16: the router is Cloudflare's `clef-flash:9b`,
the VM runs the latest Ollama release at or above 0.40.4 (picked by the
bot at each launch), and both models follow their tags ("Amendment
2026-10-16" at the end, which the sections below already describe). Amended
the same day: the router takes its top choice, with no confidence gate by
default ("Amendment 2026-10-16: argmax routing").

## Context

The owner's Colab plan buys A100 time the bot can use: Ollama on a 40 GB
A100 runs `qwen3.8:27b-mtp-q4_K_M` (Qwen 3.8 27B, multi-token prediction
on) for answers and a decision model on Ollama's `/v1/systemone` for
routing (Bespoke Labs' `nimble:9b` from 2026-09-30, Cloudflare's
`clef-flash:9b` since 2026-10-16), with no per-request quota. But:

- **Colab's terms.** The Colab FAQ forbids offering a web service and
  giving third parties access to a runtime. The owner accepted this risk
  on 2026-09-29/30 (upgrade plan, decision 3) and chose to serve **every
  user** from Colab. The mitigations are below; the account can still be
  warned or blocked.
- **Availability.** A session takes 4–8 minutes to boot, is reclaimed after
  about 25 minutes without kernel activity, lives at most 12 hours, costs
  about 5.4 compute units an hour and may be refused (412/503).
- **No official API.** The only programmatic door is
  `google-colab-cli`, whose library (`colab_cli.client`, `.runtime`,
  `.contents`) talks to Colab's private `/tun/m` endpoints. The maintainers
  call it experimental; Google's own VS Code extension has moved to an
  allowlisted public API.
- **New software.** Ollama 0.35.0 was a pre-release when chosen (stable
  since 2026-09-28; 0.34.4 has no `/v1/systemone`), and decision models
  moved under it within weeks: since 0.40.4 llama.cpp scores them, and the
  registry refuses an older client the models that need it (HTTP 412).
  The router models are days old and uncalibrated (see "The router model"
  and the 2026-10-16 amendment).
- **Ollama's concurrency.** The qwen3.5 family (both models) serves one
  request at a time, and a loaded, busy model never answers 503: a second
  request waits behind the first.

## Decision

### Never make a user wait for Colab

The Colab tiers are enabled with `COLAB_ENABLED` (which needs
`FIRESTORE_ENABLED`, a `COLAB_SHARED_SECRET` of 32+ characters, a readable
`COLAB_ADC_PATH` and `INTERNAL_CRON_SECRET`, since only the authenticated
cron tick runs the controller; without them startup logs a warning,
`/health/detail` shows `colab.reason` and the tiers stay off, never a
crash). A Colab
tier is called only when the session is **READY**: the state document says
READY with a tunnel URL, and a health check, the readiness probe or a
successful call passed in the last 120 s. Otherwise
`OllamaChatProvider` raises `SkipTierError` before any network call and the
chain moves straight to NVIDIA → OpenRouter. The state is read from
Firestore once per process (3 s bound; a failed read is retried after 30 s
at most); afterwards every writer in the process goes through the same
store, whose cache is what the request path reads. While a session boots,
every message is answered by the API tiers. Once READY, Colab answers
within the same window as every request, event time +
`REPLY_WINDOW_SECONDS` (90 s), in 1:1 chats and groups alike (user
decision, 2026-10-01: one window for every request, which replaced a 50 s
window with a 90 s `COLAB_REPLY_WINDOW_SECONDS` when Colab answered
first); the reply is still tried first, and an answer whose token LINE
refuses goes out by push. A Colab tier with a
usable tier after it must stop `COLAB_API_RESERVE_SECONDS` (20 s) before
the deadline, or halfway to it when that is later (`Tier.shares_time`), so
an API tier can still answer; it gets all of the time when every later
tier is dead, cooling down or cannot take the input. The Colab answer
tiers' own limit is 70 s to match. A request that needs thinking tries it only
once: the thinking-off pass leaves out a Colab tier the thinking pass
already had.
Recording demand is a Firestore write bounded to 3 s that the pipeline
joins only after the reply went out.

- **Busy gate.** One `asyncio.Semaphore(1)` per Colab model
  (`ColabEndpoint.semaphore`). A model already serving a request is
  skipped, not queued. This is correct only because Cloud Run runs one
  instance (`--max-instances=1`).
- **Breaker.** Three failed chat calls in a row (transport errors, the
  client's own 115 s timeout, 5xx, 404, a refused bearer) skip Colab for
  120 s. A call the chain's time limit (or the turn's deadline, or a
  shutdown) cancels is not one of them: the chain judges its timeouts,
  never counting those of a tier thinking for a request that needs it
  (Colab chat thinking at `COLAB_CHAT_THINK=low` for a plain request
  counts), ModelHealth cools qwen3.8 down after two it counts, and a hung
  tunnel fails the per-minute health check. A 404 means
  "not pulled or not this session", never the NVIDIA end-of-life breaker;
  401/403 (or Ollama's own empty-bodied Host 403) also mark the endpoint
  unhealthy until the next passing health check, which re-validates the
  bearer (three failed checks drain the session). To the chain they are
  transient, never an `AuthError`: the bearer is per session, not an
  account key, so one refusal (a named tunnel's old connector, say)
  never takes every Colab model out for ModelHealth's 10-minute provider
  cooldown, and a new session is never held back by it. The router's failures
  are its own: ModelHealth cools `colab:clef-flash:9b` down after two in a
  row, and they never touch this breaker, so a slow router (a 6 s timeout
  while qwen3.8 generates) never skips Colab chat. Only its 401, 403 or
  404, which say the session itself is wrong, mark the endpoint unhealthy.
- **Timeouts.** The client gives up after 115 s, under Cloudflare's 125 s
  edge limit (HTTP 524, handled as transient).

### Path preference

`MODEL_PATH_PREFERENCE=colab` (default): every role's Colab tiers come
first, the Colab router routes before the API router, and every triggered message that
passes the rate limit and the block checks records demand
(`last_demand_at`, at most one write a minute), which starts a session.
`api`: the original API path; Colab tiers come last and are used only when
already READY, the Colab router only when the API router fails, no demand is recorded,
and only `!colab on` starts a session.

### Chat on Ollama's native `/api/chat`

`{model, messages, stream: false, think, keep_alive: -1, options}` plus
`tools` (and `format` for a json_schema tier). `think` is `false` or a level:
`COLAB_CHAT_THINK` (default `low`) for chat and vision, `COLAB_DEEP_THINK`
(default `medium`) for the thinking role; never `true`, `high` or `max`,
which Ollama silently runs at medium. `options` are sampling only (Qwen's:
0.7 / 0.8 / 20 / presence 1.5 without thinking, 1.0 / 0.95 / 20 with it)
and `num_predict`, which thinking and the answer share: the answer's
`max_tokens` (the tier's, else `CHAT_MAX_TOKENS` / `VISION_MAX_TOKENS`)
plus `NVIDIA_THINKING_BUDGET` while thinking, as the API tiers get it; a
thinking answer cut there (`done_reason: "length"`) is no answer and the
next tier gets the request. `num_ctx`,
`draft_num_predict` and `shift` are never sent: they are load-time options
and a change reloads the model. Images go as raw base64 in `images` (a
quoted image URL skips the tier); tool calls keep Ollama's `call_…` ids;
`thinking` becomes reasoning that is never replayed.

### Colab routing (clef-flash), its top choice

After the fast rules, `SystemOneRouter` asks three questions in one
`/v1/systemone` call: `agent` (chat, vision, web_search, image_gen),
`output_format` (text, voice, image) and `needs_thinking` (noul). Since
Ollama 0.40.4 llama.cpp answers it for a GGUF decision model, rendering
one joint prompt for all fields from the GGUF's own `systemone` template:
a fixed system line ("Read the complete state and schema. Decide every
field jointly…"), `STATE:` and our state as compact JSON with sorted keys,
then per field `FIELD n`, `ID`, `TYPE`, `INSTRUCTION` and `ALLOWED
OPTIONS` (`{"description", "option_id"}` each, choice keys sorted, so an
exact tie goes to the alphabetically first key). So the instructions are
written as field descriptions and the criteria as option descriptions, in
English like the scaffold, and they speak of the state, the word the
model sees. The state is the message clipped to 300 characters, whether
an image is attached or quoted, whether a message is quoted, and the
previous agent: no memory or history (ADR-0004). The state is tokenized
with special-token parsing and nothing in it is escaped, so a space first
breaks every `<` in the message that opens a tag (before a letter, `_`,
`/` or `|`) and every `|>`: every special token in clef-flash's GGUF (27
control, 6 user-defined) starts with `<|`, `<` and a letter, or `</`, so
a user's `<|im_end|>` cannot end the prompt's turn, nor `</think>` or
`<tts_pad>` become a control token. The whole prompt is about 1,000 tokens, far under
clef-flash's `num_ctx` 16384.

Three gates decide how the answer is used, all per router model and
all settings: both choices need `confidence ≥ COLAB_ROUTER_CONFIDENCE`,
and `needs_thinking` (P(true)) turns thinking on at
`≥ COLAB_ROUTER_THINK_ON` and off at `≤ COLAB_ROUTER_THINK_OFF`, the
rulebook deciding in between. A choice's confidence is TypeSafe's
(p_max − 1/N) / (1 − 1/N) since 0.40.4 (Ollama 0.35's Go scorer used the
normalised entropy 1 − H/ln N). The confidence default is 0.0: the top
choice of each field is always used (argmax; an exact tie goes to the
alphabetically first option), because a gate of 0.90 deferred every real
message ("Amendment 2026-10-16: argmax routing" at the end). A non-zero
value still defers a choice below it. These values, and THINK_ON 0.85 and
THINK_OFF 0.15, are **provisional**: the softmax is uncalibrated at T=1
(clef-flash's GGUF sets no decision temperature).
`scripts/evals/routing_eval.py` re-tunes all three on labelled zh-TW
messages, and again whenever the router model changes or its tag moves.
Validation keeps the confidence within 0.0–0.99, THINK_ON within
0.5–0.99, THINK_OFF within 0.01–0.5, and THINK_ON above THINK_OFF.

The orchestrator's invariants are re-applied: image_gen needs both answers
to agree (agent image_gen and output image; otherwise the API router
decides), image output without image_gen becomes text, and vision needs
an image. Anything else (not READY, a session whose router is not loaded
or not wholly in VRAM, busy, breaker, cooling down, 6 s tier timeout, an
error, a malformed answer, a confidence below a non-zero gate) returns
None at once and the API router decides. The router's digest is never a
reason (the models follow their tags). `COLAB_ROUTER_ENABLED=false` disables the router tier
on the bot alone: the VM still pulls and loads it (see "VRAM" below).

### The router model: nimble:9b (decided 2026-09-30, superseded 2026-10-16)

This section records the 2026-09-30 decision; the router has been
`clef-flash:9b` since 2026-10-16 (the amendment at the end). The router
was first planned as Together's `tev1:4b`. On 2026-09-30 the
owner replaced it with `nimble:9b` (Q8_0) before any A/B run, as one
atomic change of the pin, the model profile, the role table, the VM
bootstrap, the tests and the docs.

Why nimble:

- **Licence.** Apache-2.0 for the LoRA adapter and its Qwen3.5-9B base
  (Ollama's licence layer and the Hugging Face card); the earlier
  choice's fine-tuned weights licence was still pending.
- **Native prompt format.** `/v1/systemone`'s prompt is Nimble's training
  format, SYSTEM text included (Ollama's own PR calls it
  Nimble-specific), and Nimble was trained on exactly the candidate-letter
  scoring Ollama does; the earlier choice was trained on another format.
- **Accuracy.** +2.4 points on the one same-runtime comparison: 75.7% vs
  73.3% mean accuracy over Bespoke Labs' 13 public datasets on Ollama.

Risks, and what bounds them:

- **No zh-TW evaluation.** Nimble's training data has no CJK text and
  German already costs it 3.5 points. The zh-TW message stays data inside
  English questions, the gates are strict, and the routing eval decides.
- **Uncalibrated probabilities** (above): provisional strict gates until
  the eval re-tunes them.
- **A known prompt-injection routing weakness** (Ollama's merge notes).
  A message written to steer nimble can at worst choose a wrong route:
  the invariants still hold, image_gen needs both the agent and the
  output_format answer (steering one field alone makes the router abstain)
  and vision an image, and the answer goes through the same agents and
  guards as any other route.
- **Checkpoint churn.** Three checkpoints in about six days. The digest
  pin and the router's digest policy (below) keep an unreviewed
  checkpoint out; re-pinning means re-running the eval.
- **VRAM margin.** Ollama predicts about 9.87 GiB for nimble Q8_0 and
  loads it beside qwen3.8 only with about 12.3–13.3 GiB free; nobody has
  measured the A100 yet (see "VRAM" below).

### The wire protocol (`src/services/colab/protocol.py`, `colab/register.py`)

Nothing secret is stored. Per session (`nbh`, the UUID the controller picks
before assigning), both keys are derived from `COLAB_SHARED_SECRET`, so the
bot can recompute them at any time and a leaked session key is worth one
session:

```
reg_key     = hex(HMAC_SHA256(COLAB_SHARED_SECRET, "reg:" + nbh))
proxy_token = base64url(HMAC_SHA256(COLAB_SHARED_SECRET, "proxy:" + nbh)) without "="
X-Colab-Signature: t=<unix seconds>,v1=hex(HMAC_SHA256(reg_key as ASCII, "<t>." + raw body))
```

The VM's auth proxy (127.0.0.1:8081) checks the bearer in constant time
and forwards only the probe and inference paths, never pull, push, create
or delete, so a leaked tunnel URL cannot change what the VM serves; the
allowlist, header rules and limits are in colab/README.md (協定摘要).

`POST /internal/colab/register` is public; the signature authenticates it.
Refused: a timestamp more than 300 s off, an `nbh` other than the current
session's, a bad or reused signature (403); a `ready` after `!colab off`
(409, and the session drains); a `tunnel_url` that is not `https://` + host
alone, whose host is not `<name>.trycloudflare.com` or exactly
`COLAB_TUNNEL_HOST`, or that resolves to any non-public address (422; the
page fetcher's checks, ADR-0009). All Colab traffic uses the page
fetcher's pinned, public-only transport on port 443. After `ready`, a
readiness probe (≤ 10 s, with the bearer) needs an Ollama of at least
`ollama.min_version` (0.40.4) that is exactly the release the launcher
picked for the session, qwen3.8 loaded, listed and wholly in VRAM, and
its think levels; no digest is compared, and the router need not be there
at all. A failure answers 503 so the VM can retry. The session keeps the
digests and sizes the probe saw; the VM's own `gpu` report is shown, never
trusted, and its `error` on a `ready` (why the router is missing) is
logged. `failed` and `stopping` move the
session to DRAINING; a `failed` counts as a boot failure only while
ASSIGNING or BOOTING.

### Model tags: followed, digests recorded (since 2026-10-16)

Until 2026-10-16 both models' manifest digests were pinned and a mismatch
kept qwen3.8 from booting and the router out of the session. Now the VM
pulls each tag and uses whatever it points at; `verified_digest` (on the
model profiles and in `colab/versions.json`, kept equal by the tests) only
records the last digest someone checked by hand (colab/README.md, 確認新的
digest). Ollama pulls tags and the registry serves no manifest by digest
(`/v2/library/<model>/manifests/sha256:…` is 404 even for the current
one), so a digest could only ever be checked after the pull anyway.

- The bootstrap logs each model's digest after the pull, and "tag moved
  since it was verified" when it differs; READY logs the same at INFO on
  the bot, `/health/detail` shows `colab.digests` and `colab.tags_moved`,
  and `!colab status` names a moved tag. None of it refuses anything.
- What still matters is that a model is pulled, loaded and wholly in VRAM
  (below), and the `/api/show` think values of qwen3.8, the guard against a
  release that changes how it thinks.
- A model swap must still be atomic: a router tier whose model the VM never
  pulled answers 404, and a 404 marks the whole endpoint, qwen3.8 chat
  included, unhealthy until the next passing health check.

### VRAM: whole for chat, optional for the router

`OLLAMA_MAX_LOADED_MODELS=2` keeps both models resident (keep-alive -1),
and no request sends `num_ctx` (a change would reload the model; clef-flash
keeps its own 16384). Ollama loads a model beside the loaded ones only when
its predicted size fits the free VRAM, and otherwise evicts an idle model;
since 0.40.4 llama-server may also load one partly on the CPU when the
prediction passes but memory runs short. So the bootstrap loads qwen3.8
first and the router second (the other order would need about 28 GiB free
for qwen3.8), and the two models are treated differently:

- **qwen3.8** not wholly in VRAM, or not loaded, fails the probe. If
  loading the router evicts qwen3.8, the bootstrap unloads the router,
  loads qwen3.8 again alone and boots on without the router, with an
  ERROR: chat stays up whatever the router does, and three boots in a row
  can never fail on the router's account.
- **The router** not wholly in VRAM, not loaded or not pulled: the session
  goes READY and the router sits it out. READY logs one ERROR with what
  the probe saw (the MiB in VRAM, or not pulled / not loaded) and the
  VM's own reason from the `ready` event (a pull that failed, with its
  HTTP status, a load that failed, the eviction), and `/health/detail`
  shows `colab.off_gpu`. The health check and the router add no WARNING
  or ERROR of their own for it (the router notes it at INFO).
- **Mid-session**, the per-minute health check needs only qwen3.8 loaded.
  A qwen3.8 reload with the router resident (after a runner crash) evicts
  the idle router, which sits the rest of the session out (one WARNING
  from the health check, which therefore always means a mid-session
  eviction, and one ERROR when the router first sits a message out)
  without marking the endpoint unhealthy; nothing calls it again, since loading it could evict
  qwen3.8 in turn.
- `COLAB_ROUTER_ENABLED=false` does not reach the VM: the router is still
  pulled and loaded. Given the guards above that costs boot time and VRAM,
  never chat; to take it off the VM, drop the router from `versions.json`
  and the model profiles together.

The predicted sizes and the margin (nobody has measured the A100 yet) are
in colab/README.md (VRAM).

### The control plane: the colab_cli library behind a facade

`src/services/colab/control_plane.py` wraps `google-colab-cli==0.7.4` as a
library, never the `colab` binary (which needs a writable HOME, checks PyPI
on every call, logs bearer tokens at DEBUG and writes `--env` values into
its history). Only `colab_cli.client`, `.runtime` and `.contents` are
imported, lazily, and the image installs them without their CLI and
pyarrow dependencies (`tests/test_packaging.py`). `assign` asks for
`Variant.GPU` + `Accelerator.A100`, never the high-mem shape (80 GB, about
6.8 CU/h); the same `nbh` returns the runtime already assigned to it, so a
timed-out assign is retried with it, and `find_assignment` (the assign's
GET half only) names the runtime an `nbh` given up without an answer may
still have (a timed-out POST can leave one). `list_assignments` needs
only each runtime's endpoint and proxy (an accelerator, variant or shape
0.7.4 does not know keeps its raw value, so one new kind of runtime on a
shared account cannot fail every listing). `unassign` treats
404 as released; `ccu_info` keeps the raw fields the library drops
(`eligibleGpus`); `run_cell` bounds a cell with a request-scoped
`threading.Timer` that closes the kernel websocket (the kernel client's own
timeout never fires while a cell prints) and hands back the kernel and
session ids to reuse. Every call runs in a worker thread under an explicit
budget (list and ccu-info 15 s, assign and unassign 120 s, upload 60 s),
and only from the cron tick's `colab` tasks, never from `/webhook`. The
library loggers are held at WARNING and scrubbed of tokens. Errors map to
`NoCapacityError` (412, 503), `CredentialOrPolicyError` (dead refresh
token, 400 on the A100 assign, 401/403, DENYLISTED; a 400 on a dry run's
CPU assign is not an entitlement refusal and is retried),
`TransientControlPlaneError` (a token refresh counts as dead only when the
token endpoint says so, `invalid_grant`, `invalid_client` or reauth:
google-auth also calls a 502, an HTML page or a 200 without a token
non-retryable) and a base
`ControlPlaneError` for answers the client does not understand; contract
tests pin the 0.7.4 wire shapes (`tests/test_colab_control_plane_contract.py`).

### Lifecycle and cost guards (`src/services/colab/controller.py`)

One Firestore document, `{prefix}_colab/state`, written with optimistic
concurrency and a version counter, holds the session. Its decisions are
pure functions of the document, what was just observed and the settings;
the per-minute cron applies them in at most 5 s of state reads and writes
and never calls Colab itself: it queues `colab` Cloud Tasks (`assign`,
`launch`, `heartbeat`, `unassign`, `reconcile`), and a queued step that has
not reported back within its lease (180 s, launch 240 s) is queued again.
The controller only queues (`enqueue_only`): a task Cloud Tasks does not
take is logged and queued again by its lease or the next tick, never run
inline inside the cron request (60 s Scheduler deadline) or the task that
queued it, and on Cloud Run `COLAB_ENABLED` needs Cloud Tasks (with the
inline queue the tier stays off). `colab` tasks have their own queue,
`CLOUD_TASKS_COLAB_QUEUE` (`linebot-colab`, two concurrent dispatches), so
an assign or a launch never takes a dispatch slot a LINE event needs.

- **IDLE → ASSIGNING** when a session is wanted (`desired` auto, and
  demand or `!colab on` within the last 5 minutes) and nothing holds it:
  a ccu-info read no older than 5 minutes (else a `reconcile` reads it
  first), room for one start nobody uses (a boot as long as
  `COLAB_BOOT_TIMEOUT_SECONDS` plus one idle TTL at the expected rate,
  about 2.7 CU on the A100) above `COLAB_MIN_CU_BALANCE` (10) and under
  `COLAB_DAILY_CU_CAP` (30, Asia/Taipei days), and the A100 eligible when
  ccu-info says; otherwise the session would be drained the moment it is
  READY. The new `nbh` is written **before** the assign is queued.
- **ASSIGNING**: success → BOOTING and `launch`; a timeout or transient
  error retries the same `nbh`, three attempts at most; 412/503 → COOLDOWN
  for 5, 10, 20, then 60 minutes; a credential or policy error →
  DISABLED. A session given up while its assign had no answer (that
  COOLDOWN, a drain that waited in vain), or refused only on a retry (the
  try that timed out may still have made a runtime), keeps its `nbh` in
  `unanswered_nbhs` (at most 5); a reconcile a minute later looks each one
  up and releases the runtime it finds. A lookup that fails is retried
  every minute for 30 minutes, then only by the reconciles that run anyway
  (daily, or every tick of a session).
- **BOOTING → READY** by the signed registration and the probe. A failed
  launch, a `failed` from the VM, a runtime that vanishes (the VM's
  `failed` is sent once and may never arrive) or no READY within
  `COLAB_BOOT_TIMEOUT_SECONDS` (900) → DRAINING (or IDLE) and one more boot
  failure; the third in a row ends in DISABLED. A balance under the floor
  or the daily cap reached drains a booting session too.
- **READY → DRAINING** on `!colab off`, three failed health checks in a
  row, `COLAB_IDLE_TTL_SECONDS` (900) without use (counted from the latest
  of the last demand, the last `!colab on` and READY),
  `COLAB_MAX_SESSION_SECONDS` (39600), a balance under the floor or the
  daily cap reached. **DRAINING → IDLE** after `unassign` (404 counts as
  released). An unassign that fails, 401/403 included (Colab may refuse to
  release a runtime the VM already released), never disables the tier: the
  next tick queues it again with a reconcile, whose listing ends the drain
  once the runtime is gone and disables the tier when it is refused too.
- **Reconcile** (every tick while BOOTING or READY, and while DRAINING
  after an unassign that did not end it; once a day otherwise):
  `list_assignments`, ccu-info, the health check and the due
  `unanswered_nbhs` lookups. A runtime of ours that is gone (reclaimed, or
  released by the VM) → IDLE, counted as a boot failure before READY.
  Runtimes that
  are not the current session are released when they are ours
  (`known_endpoints`, the last 20) or `COLAB_ACCOUNT_DEDICATED=true`, and
  otherwise only logged at ERROR once per process (they may be the owner's
  browser sessions).
- **Keep-alive.** Colab keeps a runtime only while its kernel is active
  (tunnel traffic does not count), so while BOOTING or READY and in use a
  `heartbeat` every `COLAB_HEARTBEAT_SECONDS` (300) runs a tiny cell in
  the saved kernel that prints `/content/state.json` (≤ 300 characters,
  never a secret).
- **Compute units.** While a runtime is ours, `cu_used_today` grows by
  ccu-info's hourly rate read after the assign (5.37 CU/h estimated for
  the A100 until then, 0.1 for a dry run) times the time held.

Owner commands, for `LINE_ADMIN_USER_IDS` only (anyone else gets
「此指令僅限管理員」): `!colab on` (auto again, out of DISABLED/COOLDOWN,
demand now under either preference), `!colab off` (stay down; a session
being assigned, booting or ready drains at once), `!colab status`.
`!whoami` answers the sender's userId in a 1:1 chat only. In groups,
`!colab` triggers without a mention, like `!new`.

### Launch and the VM (`src/services/colab/launcher.py`, `colab/`)

The `launch` task picks the Ollama release (the latest GitHub release at
or above `ollama.min_version`, else versions.json's fallback; the
amendment below), uploads `colab.VM_FILES` to `/content` with a
versions.json naming that release, and starts `bootstrap.py` detached; the
steps and their order are in colab/README.md (端到端流程). What the design
depends on:

- **Secrets never touch argv, history or a log.** They reach the VM only
  in `/content/.bootstrap.env`, written with a silent, history-free kernel
  execute (mode 0600 before any byte) and deleted by the bootstrap as it
  reads it; never `colab exec --env`, which logs values.
- **A launch counts only when `BOOTSTRAP_LAUNCHED <pid>` comes back**: a
  failing cell still "succeeds".
- **Only checked files run on the VM.** Ollama and cloudflared are size-
  and sha256-checked (Ollama against GitHub's own asset size and digest
  for the release the bot picked, or versions.json's pinned fallback),
  and the auth proxy's aiohttp stack installs from
  `proxy-requirements.lock` with `--require-hashes --no-deps
  --only-binary=:all:` (the proxy holds the session's keys; its versions
  are the bot image's).
- **The VM releases itself.** Any bootstrap failure registers `failed` and
  releases the runtime (`POST $TBE_RUNTIME_ADDR/unassign`). The proxy's
  watchdog is the second line of the cost guards: idle or age past the
  limits plus 120 s, or Ollama down after one restart, and it registers
  `stopping` and releases the runtime; no error ends that guard (three
  failed checks in a row, or a release that fails, send the bare
  self-unassign and exit).

### Credential model

The control plane runs as the owner's Colab account: a Google
`authorized_user` refresh token that only a person can mint (browser
consent). `scripts/mint_colab_adc.sh` (dry run by default) mints it under a
scratch `CLOUDSDK_CONFIG`, so the machine's own ADC is untouched, stores it
as a new version of the Secret Manager secret `linebot-colab-adc` and lets
`linebot-runtime` read it; the steps are in DEPLOY.md (Colab 層).

- **A dedicated account.** gcloud always adds the cloud-platform scope, so
  the token of an account with roles in the project could manage the
  project from Secret Manager. The Colab plan belongs on a Google account
  with no GCP roles that is used for nothing else: every gcloud or
  colab-cli login mints another token on the same Cloud SDK OAuth client,
  and past 100 live tokens Google silently revokes the oldest.
- **Mounted, never exported.** `scripts/deploy_cloud_run.sh` mounts the
  newest enabled version, pinned per revision like the dotenv secret, at
  `/secrets/colab/adc.json` (`COLAB_ADC_PATH`) only while
  `COLAB_ENABLED=true`, and never reads it. A missing secret leaves the
  tier off with a warning, and so does a runtime account the secret's
  policy grants no `secretAccessor` (a project-level or group grant does
  not show there; Cloud Run rejects a revision that cannot read it).
  After a passing smoke check the deploy destroys the versions no kept
  revision mounts. `GOOGLE_APPLICATION_CREDENTIALS` is never set: it would
  turn every Google client (Firestore, Cloud Storage, Cloud Tasks) into
  the owner's personal account. The control plane reads the file once
  into an `AuthorizedSession`.
- **Lifetime.** The token dies when revoked, after 6 months unused, or
  when pushed out by the 100-token cap. The controller then sees
  `invalid_grant` or "Reauthentication is needed" and moves to DISABLED
  with an ERROR log; replies go on through the API tiers. Any move to
  DISABLED first queues the unassign of the runtime the session holds (the
  one Colab call DISABLED still makes); when that fails too (a dead token),
  the VM's watchdog releases it. Re-minting is a person's job (DEPLOY.md,
  Colab 層, 重新產生 token).

### The dry run and the live gate

`COLAB_DRY_RUN=true` assigns a CPU runtime and starts the VM with
`DRYRUN=1`, whose stub upstream passes the readiness probe and answers
inference with Ollama's busy 503: the whole control path, registration and
READY included, runs for a fraction of a compute unit while users still get
API answers. `scripts/live_gate.py --path colab` checks a READY session
from the operator's machine with the bot's own probe and clients (a dry-run
session gets only the probe). How to run both: DEPLOY.md (Colab 層).

### What cannot be automated

These stay with a person (the steps are in DEPLOY.md, Colab 層):

- Buying the Colab plan, accepting Colab's terms and holding the
  dedicated account.
- Minting the refresh token (Google's consent screen) and re-minting it
  after it dies.
- A named tunnel's Cloudflare setup (optional; quick tunnels need none).
- The first live runs: the CPU dry run, then one A100 run that measures
  the assign latency, the cold start, the burn rate and VRAM, and the
  routing eval that re-tunes the `COLAB_ROUTER_*` gates.
- Reviewing a moved tag (colab/README.md, 確認新的 digest): the boots go on
  with it meanwhile, so a bad republish shows up in the answers or the
  routing eval first. Checking a new Ollama release's notes and the models'
  `requires` against `ollama.min_version`.
- Turning the tier on for everyone (`MODEL_PATH_PREFERENCE=colab`), and
  acting on DENYLISTED or a policy warning.

## Consequences

- Terms risk, accepted by the owner: a dedicated Google account for the
  Colab plan (with no GCP roles) is strongly recommended; `COLAB_ENABLED`
  and `!colab off` are the kill switches; the controller disables the tier
  on a refused credential or account; compute-unit floor and daily cap
  bound the spend, and the VM's watchdog releases a runtime the controller
  lost track of.
- Replies never wait for Colab, so a cold or broken session costs nothing
  but the API tiers' quota. The price is that the first messages of a
  burst go to NVIDIA/OpenRouter, and a READY session's slow answer may
  miss the reply token and go out by push (the reply window is 90 s; the
  token lasts about a minute).
- The busy gate and the demand throttle are process-local: they assume one
  instance. Raising `--max-instances` needs a shared gate first.
- The `/tun/m` endpoints are private and may change without notice: the
  exact library pin, the contract tests and the automatic DISABLED on
  refusals contain the damage, and `COLAB_ENABLED=false` ends it.
- The Colab router may route differently from the API router; by default
  it answers with its top choice (no confidence gate), and the routing eval
  decides the provisional gates (and whether to keep it on). Following
  the latest Ollama and the models' tags means a session can run something nobody tested (see the
  amendment). Unverified until the Colab runs: NDJSON streaming
  through the quick tunnel, whether Cloudflare and cloudflared pass the bot's
  hang-up on to the proxy (the proxy itself aborts the Ollama call when its
  client goes away and logs `cancelled after N s`, tested through `serve()`;
  a proxy change takes effect at the next Colab boot, when the VM files are
  uploaded), Colab's driver version, UDP/QUIC egress, and whether the
  heartbeat alone keeps a runtime.
- Tests drive all of it without Colab or Google:
  `tests/test_ollama_provider.py`, `test_systemone_router.py`,
  `test_colab_endpoint.py`, `test_colab_registration.py`,
  `test_colab_state.py`, `test_colab_admin.py`,
  `test_colab_controller.py` (every transition on a fake clock),
  `test_colab_control_plane_contract.py`, `test_colab_launcher.py`,
  `test_colab_vm_scripts.py` (the VM package, cross-checked with the bot),
  the end-to-end `test_colab_pipeline.py` (a fake tunnel behind
  `httpx.MockTransport`), `test_live_gate.py` (the colab path), and the
  mint, provisioning and deploy scripts against the fake `gcloud`
  (`test_mint_colab_adc.py`, `test_provision_gcp.py`,
  `test_deploy_script.py`).

## Amendment 2026-10-16: clef-flash, the latest Ollama, models that follow their tags

### What broke

On 2026-10-14 Bespoke republished `nimble:9b` (new manifest list, llamacpp
child `6aa3fde9…`, no SYSTEM layer) with a config that says
`"requires": "0.40.4"`. The registry answers HTTP 412 to an older Ollama's
pull, and the VM pinned Ollama 0.35.0, so on 2026-10-15 the router never
pulled. The bootstrap logged that on the VM only; the bot logged nothing
but, after READY, `health check found nimble:9b not loaded`, and the real
reason was never in Cloud Logging. A pinned digest could not have helped:
the pinned manifest `24e550a1…` is no longer served at all.

### What the owner decided, and why

1. **The router is Cloudflare's `clef-flash:9b`** (fine-tuned from
   Qwen3.5-9B, Apache-2.0; a Q8_0 GGUF of 9,657,260,192 bytes plus a
   921,704,928-byte vision projector, params `{"num_ctx":16384}`, config
   `"requires": "0.40.4"`; on 0.40.5 `/api/tags` reports the llamacpp child
   `73e026ff389b…` of the manifest list as of 2026-10-16). The
   `/v1/systemone` request body is unchanged; the prompt (one joint prompt
   from the GGUF's template, "STATE:"), the confidence formula and the
   special-token handling did change, as "Colab routing" above describes,
   so the questions speak of the state, a `<` opening a tag and `|>` in
   the message are broken, and the gate default moved from 0.75 to 0.90 to keep the old
   strictness (the deployed `.env` sets it explicitly and must move too;
   the same day it became 0.0, "Amendment 2026-10-16: argmax routing").
   The trace label is `router` (`router_p=n/a(router …)`), the live gate's
   check `colab.router`.
2. **Ollama follows the latest release.** The launcher asks GitHub for the
   latest release (never a draft or pre-release) and uses its
   `ollama-linux-amd64.tar.zst` when it is at least `ollama.min_version`
   (0.40.4, clef-flash's `requires`) and the asset has a GitHub download
   URL, a `sha256:` digest and a size; anything else (an HTTP error, a
   10 s timeout, a rate limit, a missing digest) uses versions.json's
   pinned fallback (0.40.5), with a WARNING naming why. The bot decides,
   not the VM, so the choice is in Cloud Logging and survives the VM. It
   is kept per session, in-process: every launch retry of a session
   uploads the same release (a retry naming another one than the
   bootstrap already running would fail a healthy probe), and every new
   session asks GitHub again (GitHub rebuilt v0.40.5's assets a day after
   publishing it, so a stale digest costs at most one session). The VM
   still checks the tarball against the size and sha256 it is given (it
   stops a longer body at once and retries a short one). The bot then
   wants exactly that release from the session's probe and health checks
   (`ColabState.ollama_version`), or only the minimum for a session
   launched before this change.
3. **Both models follow their tags.** No digest gates anything; the last
   verified one is kept as information (`verified_digest`) and a moved tag
   is logged at INFO ("tag moved since it was verified").
4. **Observability.** The VM sends why the router is missing in the
   `ready` event's `error`; READY logs it in one ERROR beside what the
   probe saw, and the health check and the router stay quiet about a
   router READY already reported (the router notes it at INFO), so the
   health check's WARNING, and the router's ERROR after it, only mean a
   mid-session eviction.
5. **VRAM.** clef-flash keeps its own `num_ctx` 16384 (no derived model):
   Ollama's preflight predicts about 13.0 GiB, so about 16.2 GiB must be
   free after qwen3.8 (nimble needed about 13.3). It may not fit, or
   llama-server may load it partly on the CPU; either way the router sits
   the session out and chat goes on. The first A100 run must measure it
   (`size_vram` in `/health/detail`).

### The trade-off of following latest

Pins made every boot reproducible and turned a republish into a refused
boot (qwen3.8) or a router left out (nimble). That failed safe but not
usefully: an upstream change that needs a newer Ollama makes a pin
unpullable, and the bot then routes without its router until someone
re-pins by hand. Following the latest release and the tags keeps the
session working through such changes, at the price of running releases
and checkpoints nobody has tested here: a new Ollama could change the
think values (the probe's `/api/show` check refuses that session), the
API (the probe and the warm-up would fail the boot, and three failed boots
disable the tier until `!colab on`), or VRAM use (the router sits out); a
moved model tag can route or answer differently with no failure at all,
which only the logged digest, the answers and the routing eval reveal.
The pinned fallback, the minimum version and the probe bound how far that
can go. There is no setting that holds the VM on the fallback while a newer
release exists: a bad release is skipped once a fixed one is out, by
raising `ollama.min_version` and the fallback entry together to it
(`validate_versions` wants the fallback at least the minimum, so a
minimum raised alone past the latest release fails it), and before that
only by a code change. So the release notes are worth a look whenever the logged
version changes.

## Amendment 2026-10-16: argmax routing

On its first day clef-flash scored the real messages it was asked about at
a confidence of 0.43 and 0.39 (a top probability of about 0.57 and 0.54
out of four options), so the 0.90 gate deferred every one and the API
router decided each, once only after 24 s. A gate on an uncalibrated
softmax, set before any evaluation, measured nothing useful. The owner
decided that clef-flash takes its highest-probability choice:
`COLAB_ROUTER_CONFIDENCE` defaults to 0.0, and validation allows it from
0.0 to 0.99 (THINK_ON keeps 0.5–0.99). An exact tie goes to the
alphabetically first option: the scorer reads the options in sorted order,
and `router_p` lists ties in that order too. Thinking keeps its gates
(0.85 and 0.15, the rulebook in between), and the invariants still hold,
so image_gen still needs both answers to agree. The cost: a wrong top
choice now routes the message where the API router might have been right.
`router_p` keeps every probability in the summary line, and
`scripts/evals/routing_eval.py`, whose recommendation may now be anything
from 0.0, can bring a gate back once labelled zh-TW messages show one
worth its lost coverage.
A deployed `.env` that sets 0.75 or 0.90 must change to 0.0.
