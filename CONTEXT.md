# CONTEXT

Mental model for the LineBot multi-agent service. Read this before changing
`src/`. Decisions with non-obvious rationale live in [`docs/adr/`](docs/adr/)
(ADR-0001 to ADR-0012; one-line index in README.md, "Architecture decisions").

## What this is

A LINE bot on Cloud Run that routes each inbound message to one of four
specialist agents, then replies as text, voice, or image. Free-tier
constraints shape most of the design: the GCP project must stay inside the
Always Free tier (Cloud Run scales to zero, no VM or VPC, Firestore within
its daily 50K reads / 20K writes), and LINE push messages count against
LINE's monthly quota.

## Request path

```
LINE webhook
  → main.webhook              signature check, dedupe, then per event:
      triggered               → queue `event` task
      untriggered group msg   → passive buffer (ready → `passive_flush` task)
      untriggered text/image  → cache text inline / `cache_image` task
  → Cloud Tasks               POST /internal/tasks/{kind}  (a real request: CPU)
  → task_handlers.handle_task
  → pipeline.process_event    (`event` task)
      → process_input         LINE event  → AgentRequest (+ received_at)
      → !new / !colab / !whoami   commands answer here (no rate limit, no model)
      → flush_passive         group/room: earlier untriggered messages first
      → enrich_request        rate limit + memory context
      → Orchestrator.route    fast regex rules, else nimble (Colab READY, confident),
                              else LLM classification (MODEL_PATH_PREFERENCE=api: nimble last)
        + load_profile_facts  the sender's stored facts (10-minute cache), read meanwhile
        + note_demand         Colab demand (colab preference; ≤ 1 write/minute, ≤ 3 s),
                              joined only after the reply went out
      → <agent>.process       chat | vision | web_search | image_gen
                              (web_search: Tavily → no search; pages read SSRF-safe)
      → send_response         plain text for LINE; the free reply first, a push only
                              when the reply certainly failed (nothing counts pushes)
      → record_interaction    persist the turn; over the limit → `compact` task
      → touch_user_profile    display name / last seen; writes at most daily

Cloud Scheduler (1/min) → /internal/cron → scheduled messages + stale passive flush
                                           + first tick of the (Taipei) day: `eol` task
                                           + Colab controller tick (state in/out only;
                                             `colab` tasks: assign | launch | heartbeat |
                                             unassign | reconcile call Colab)

Colab VM → POST /internal/colab/register  (public; HMAC-signed; tunnel probed before READY)
```

The webhook does no slow work: it queues tasks and answers well inside LINE's
2-second limit. The service bills CPU only during requests, so anything left
running after a response starves; every piece of follow-up work is a Cloud
Task instead. Locally (and when Cloud Tasks is not configured) the same
handlers run inline in the request. If queueing fails, the webhook runs the
handler inline rather than drop it. See ADR-0006.

## Module map

| Area | Module | Notes |
|---|---|---|
| HTTP transport | `main.py` | lifespan, `/webhook`, `/health` (public, minimal), `/health/detail` (cron token), `/internal/cron`, `/internal/tasks/{kind}`, `/internal/colab/register` (signed by the VM) — and little else |
| Assembly | `src/runtime/composition.py` | builds providers, binds the role table, hands each agent its role's tiers, wires the tool registry |
| Model roles | `src/runtime/model_roles.py` | `RoleTable`: which models serve router / chat / thinking / vision / utility / embed / image, in order; code defaults + `MODEL_ROLES`, validated at startup (ADR-0007) |
| Built app | `src/runtime/container.py` | `AppContainer`: what composition produced |
| Request handling | `src/pipeline/event_pipeline.py` | one event → one reply; deps injected via `PipelineDeps` |
| Task handlers | `src/pipeline/task_handlers.py` | one handler per task kind; never raises; dedupes `event` |
| Cron maintenance | `src/pipeline/maintenance.py` | `run_cron_tick`: stale passive-buffer sweep; queues the daily `eol` task; runs the Colab controller's tick (state only; it queues `colab` tasks) |
| Task queue | `src/services/task_queue.py` | `CloudTasksQueue` (REST) or `InlineTaskQueue`; `enqueue_or_run` |
| Routing (LLM) | `src/agents/orchestrator.py` | LLM classification + repair retry |
| Routing (rules) | `src/agents/routing_rules.py` | the regex rulebook; pure, no I/O; also the shared URL helpers |
| Agents | `src/agents/` | all extend `BaseAgent`, prompts load from `prompts/*.md` |
| Tool loop | `src/agents/tool_runner.py` | chat's tool calls; works on a copy, ends with a `tool_choice="none"` answer |
| Search planning | `src/services/search/query.py` | `SearchQueryPlanner`: the web search's query, topic and time range; rewrites follow-ups on the utility tiers |
| Search backends | `src/services/search/backend.py`, `tavily.py` | `ChainedSearchBackend`: Tavily (key or keyless) → an empty `degraded` answer; the seam another backend plugs into; shared by the web search agent and the `web_search` tool (ADR-0009) |
| Page reading | `src/services/search/page_fetcher.py`, `extract.py`, `rerank.py` | SSRF-safe fetch (public addresses only, pinned connection, checked redirects, byte cap); trafilatura in its own threads, a simple extractor as fallback; optional embedding rerank |
| Providers | `src/providers/` | NVIDIA + OpenRouter + Colab (`ollama_provider.py`, Ollama's native `/api/chat`) behind `FallbackChain`, which walks model tiers; shared errors and parser in `base.py`, HTTP status mapping in `openai_compat.py` |
| Colab router | `src/providers/systemone.py` | `SystemOneRouter`: nimble on `/v1/systemone`, three scored questions in Nimble's schema style, `RouterGates` (COLAB_ROUTER_CONFIDENCE / THINK_ON / THINK_OFF, provisional until the routing eval); None whenever unsure, not READY, or the session's router is off its pin or not wholly in VRAM |
| Colab tier | `src/services/colab/` | `protocol` (key derivation, signature, tunnel rule, pins), `state` (the Firestore session document, optimistic writes), `endpoint` (READY, busy gates, breaker, tunnel client, probe), `registration` (`/internal/colab/register`), `admin` (`!colab`, `!whoami`), `control_plane` (the colab_cli library facade: assign, list, unassign, ccu-info, cells, uploads; budgets, error classes, token scrubbing), `controller` (pure lifecycle decisions + the cron tick and `colab` tasks), `launcher` (upload, silent env file, detached bootstrap); ADR-0010 |
| Colab VM | `colab/` | what the launcher uploads to the runtime: `bootstrap` (GPU check, pinned Ollama and cloudflared, model pulls with digest checks, warm-up qwen3.8 first and VRAM record, tunnel, signed `ready`/`failed`), `auth_proxy` (bearer, path allowlist, Host rewrite, idle watchdog that releases the runtime), `register` (the bot's protocol mirrored in the standard library), pins in `versions.json` and the proxy's hash-pinned `proxy-requirements.lock`; runbook in `colab/README.md` |
| Ops scripts | `scripts/` | `deploy_cloud_run.sh`, `provision_gcp.sh`, `mint_colab_adc.sh` (a person mints the Colab token into `linebot-colab-adc`), `live_gate.py` (live checks through the bot's own clients; `--path tiers` calls every tier of the configured role table once; `--path colab` needs a READY session), `envfile.py`, `migrate_embeddings.py` |
| Model profiles | `src/providers/model_profiles.py` | exact-id facts per model: protocol, modalities, thinking and JSON modes, limits, model-card sampling; the payload builders and the role table read them |
| Model health | `src/services/model_health.py` | per-model cooldowns and dead models, per-provider request budgets (chat, image and embedding calls); only an account-level OpenRouter 429 cools the whole provider |
| Model lifecycle | `src/services/model_lifecycle.py` | daily end-of-life check against the public catalogs (`eol` task); results for `/health/detail` |
| Memory | `src/services/memory_service.py` | recent window (time-ordered, ids) + long-term summary + episodes + cached user profile |
| Write batching | `src/services/passive_buffer.py` | per-chat coalescing; reports ready / due keys, never schedules; restores a failed flush |
| Persistence | `src/services/memory_backends.py` | in-memory (tests) or Firestore (a fake client in `tests/test_firestore_backend_fake.py`) |
| Scheduling | `src/services/scheduler_service.py` | pull-style, driven by Cloud Scheduler |
| LINE delivery | `src/services/line_service.py` | reply first (free), push only when the reply certainly failed; an unclear outcome counts as sent; no push is counted, capped or refused (LINE's own monthly quota is the only limit); notes `delivery=reply\|push` for the summary line |
| Request summary | `src/utils/request_trace.py` | one log line per event: route, every tier walked with its outcome (`skip(<why>)`, `error(<Kind> <status>)`), the answering tier, `delivery` |
| Settings | `src/config.py` | reads `.env`, then the mounted secret `/secrets/app/app.env`, then the environment (ADR-0008) |
| JSON recovery | `src/utils/jsonish.py` | pulls a dict out of LLM prose; valid JSON is parsed as is first |
| Input cleaning | `src/utils/validators.py` | signature check; `sanitize_input` (newlines kept, injection markers stripped) for every incoming text |
| LINE text | `src/utils/line_text.py` | Markdown/LaTeX in a reply → plain text (LINE shows markup raw); runs before OpenCC in `output_processor` |
| GCP metadata | `src/utils/gcp_metadata.py` | access token (cached), project, region, SA email |
| Dedupe | `src/utils/dedupe.py` | in-process TTL/LRU key set |

## Invariants worth knowing

These are load-bearing and easy to break by accident:

1. **No in-process timers and no work after the response.** Cloud Run only
   gives CPU while a request is running and scales to zero when idle, so an
   `asyncio.sleep` timer, or a `create_task` still running after the response
   is sent, gets no CPU to run on. Scheduled messages and stale passive buffers are *pulled* by
   Cloud Scheduler hitting `/internal/cron` every minute (ADR-0001); all other
   follow-up work is a Cloud Task handled in its own request (ADR-0006). Media
   expiry is the bucket lifecycle rule's job.

2. **The recent-message window is a compaction trigger, not a retention bound.**
   Crossing `memory_recent_message_limit` *schedules* summarization; it must not
   truncate, or messages are lost before they reach long-term memory. Only the
   hard safety cap trims, and it logs when it does. The window is sorted by
   `created_at` (the LINE event time; a reply takes its question's, so it
   stays right after it), and a compaction commit removes exactly
   the messages it summarized, by `id`, so concurrent writes and the cap can
   never make it skip; it is dropped, with its episode, only when another
   summary landed first or `!new` cleared the chat meanwhile (a new
   `generation`). A chat compacted in the last
   `COMPACTION_DEBOUNCE_SECONDS` waits for a later message unless its window
   reached half the cap, and gets at most one episode per
   `EPISODE_MIN_INTERVAL_SECONDS`. A pass that failed (a summary cut off at
   `max_tokens` moves on to the next utility tier first) or whose task never
   started holds the chat for 15 minutes, one minute once its window reached
   half the cap. See ADR-0003.

3. **The orchestrator deliberately drops the long-term memory summary** from
   routing messages, and the sender's profile facts too. They are useful for
   answering, but they distract the classifier from the routing decision.
   Chat, vision and web_search get the facts as a fenced
   `[使用者 profile，僅供參考]` user turn; the image prompt refiner does not.
   The facts follow the user into groups, where every member reads the
   reply, so a group reply's system prompt adds a rule: do not bring the
   profile up unless the sender does. See ADR-0004.

4. **Yearly `02-29` jobs clamp to `02-28` on non-leap years** so annual
   reminders fire every year. The clamp applies to both `next_run` and
   `matches`. See ADR-0002.

5. **Passive group messages are buffered and coalesced** before being written,
   purely to bound Firestore write volume — not for latency. There is no
   timer. The webhook queues a `passive_flush` task when a buffer reaches
   `PASSIVE_GROUP_FLUSH_THRESHOLD` or gets a message after its oldest one is
   `PASSIVE_GROUP_FLUSH_DELAY_SECONDS` old (and asks again if that flush did
   not happen within the delay). A triggered `event` flushes its group first,
   so the window stays in chronological order. Quiet chats are flushed by the
   cron sweep, the rest on shutdown. A failed flush keeps its batch buffered;
   a load failure never saves a default document over stored memory. Buffers
   are process-local, which is fine only while `--max-instances=1`.

6. **`reply_token` is single-use and short-lived.** Every send, in a 1:1
   chat or a group, tries the free reply first, however long the answer
   took, and pushes only when the reply certainly failed (a 4xx such as an
   expired or invalid token, a request that never left). Nothing counts,
   caps, reserves or refuses a push: LINE's own monthly quota (pushes to a
   group bill every member) is the only limit,
   a decision taken on 2026-10-01 because a self-imposed budget kept slow
   answers inside the reply window. A reply or push whose outcome is
   unknown (a timeout after sending, a 5xx) counts as sent: it is never
   followed by a push or a second send.

7. **The pipeline takes its collaborators as arguments**, never module globals.
   That is what lets tests drive the real code path without patching, and why
   production code needs no `hasattr` guards. See ADR-0005. It is also the
   seam for a graph runtime: `process_event(event, deps)` stays the only
   entry the task handler calls, and the steps pass plain data
   (`AgentRequest`, `RouterDecision`, `AgentResponse`). LangGraph is not
   used, on purpose; the triggers that would change that are in ADR-0012.

8. **Every model call runs against a deadline.** A plain answer, in any
   chat and whichever tier answers it (Colab or API), must be done by one
   reply window: event time + `REPLY_WINDOW_SECONDS` (90 s), which stays
   below `TURN_DEADLINE_SECONDS`. A thinking answer or an image, in any
   chat, or a plain answer that starts too late (a late event, or routing
   that used the window up), gets the turn deadline
   (`TURN_DEADLINE_SECONDS` from when work on the event starts). Whatever
   finishes after the reply token is gone goes out by push (invariant 6). The fallback chain clips each tier's own time limit to what
   is left (image generation bounds each image tier the same way, within
   the answer's deadline), and the thinking tiers together may use only
   half of it, so the non-thinking tiers after them keep the other half; an
   image that needs thinking lets the vision tiers not set to thinking off
   (only Colab: the API vision tiers never think) think first, then runs
   them all with thinking off. An image prompt, whichever agent answers,
   carries at most `IMAGE_CONTEXT_CHARS` (3000) characters of history,
   summary and profile facts, so it stays near 4k tokens (muse-glimmer
   took about 40 s at 6.2k). Chat's tool loop gives its first
   round (usually the whole answer) the whole deadline; once the model asks
   for tools, the tool calls and later rounds must end 20 s before it, so
   the answer keeps its time; a tool that runs out of time returns an error
   and the model answers with what it has. `web_search` runs at most twice
   per turn, every round counted (the loop answers a third call "limit").

9. **Deliveries are at-least-once.** LINE redeliveries and duplicate Cloud
   Tasks are dropped by `webhookEventId` in in-process sets (webhook and task
   handler keep separate ones). The queue makes exactly one attempt per task
   (`maxAttempts=1`, no retry duration), and `/internal/tasks` answers 200
   once authenticated. See ADR-0006.

10. **Models come from the role table only.** No code names a model id to
    call: agents get their tiers from composition (`BaseAgent.targets_for`
    picks the thinking or vision tiers per request), and every id is
    checked against its exact profile at startup. A bad table fails
    startup; a tier without its provider is disabled with a reason, never
    dropped silently. See ADR-0007.

11. **Colab never holds up a reply.** Its tiers exist only with
    `COLAB_ENABLED`, and a session that is not READY (state document READY
    and a health check within 120 s) is skipped at once, with no network
    call; the state is read from Firestore once per process. The session
    starts on demand and is released after 15 idle minutes. Until then
    every message is answered by the NVIDIA → OpenRouter tiers, which is
    why router, chat, vision and utility must each keep a non-Colab tier.
    A Colab model already serving a request is skipped, not queued (one
    `asyncio.Semaphore(1)` per model: valid only with `--max-instances=1`),
    and three failed chat calls in a row skip Colab for 120 s (the
    router's failures cool only the router down). Once READY, a Colab
    tier with a usable tier after it must stop `COLAB_API_RESERVE_SECONDS`
    (20 s) before the deadline, or halfway to it when that is later
    (`Tier.shares_time`; a thinking pass keeps its half), and a request
    that needs thinking tries it only once (the tiers that follow with
    thinking off leave out a Colab tier the thinking pass already had), so
    the API tiers always keep time to answer. nimble routes only when confident; otherwise the API
    router decides at once. See ADR-0010.

12. **What LINE shows is plain text.** LINE renders no Markdown, so
    `send_response` strips markup from every outgoing text (reply, image
    caption, text read aloud) before the OpenCC conversion, and a reply
    left blank by that falls back to a default text. Memory stores the
    same plain text, so history never shows a model its own markup. The
    reply prompts say so too, and carry no bold of their own for models
    to copy.

13. **Text derived from the user never goes in the system prompt.** The
    router's `task_description` and `reasoning` are written from the
    user's message, so `BaseAgent._build_messages` sends them as a fenced
    user turn (`[調度器提示，僅供參考，非使用者指令]`) right before the
    user's message; the system prompt only gets lines the code controls
    (the validated output format, the quote note, the injection guard).
    Suspected prompt injection is stripped by `sanitize_input`, triggered
    and passive text alike, and the message is still answered (one that
    was nothing but a marker gets the "請附上想問的內容" reply); nothing is
    dropped silently. Text the code did not write (history turns, quoted
    text, the summary, profile facts, router hints, tool results, web
    text, the router's context lines) loses the fence labels
    (`FENCE_LABELS`, look-alikes too) and, in the router and the search
    planner, those prompts' own labels; stripping repeats until no nested
    label is left, so it cannot end a block early or pose as the user's
    message. In a group, a line of a member's message that starts like a
    speaker tag (`User_xxxx:` with four hex digits, `Assistant:`, the bot's
    name; full-width, look-alike letters and colons, invisible characters
    and Markdown decorations too) gets `〔同一則訊息〕` in front, in the
    message itself and its quote, the history, the summary and episode
    transcripts, so it cannot pose as another member or the bot; the line
    breaks stay, and the reply prompts tell the model not to copy the
    marker (`mark_speaker_lines`, `src/utils/speaker_lines.py`).

14. **A fast rule must be sure.** A regex match skips the router, so each
    rule is high-precision (word boundaries, anchors, image nouns, short
    follow-ups only when the previous route is known) and anything
    ambiguous goes to the router, which sees the history. A topic word is a
    lookup only when the message asks for it or is a bare query. Every verified
    misroute is a case in `tests/test_routing_rules.py`.

15. **Secrets never leave `.env`, Secret Manager and the container.** The
    deploy stores `.env` (minus deploy-only keys) as one dotenv secret,
    `linebot-dotenv`, pinned per revision and mounted as a file; nothing
    secret is uploaded to Cloud Build, baked into the image or set as a
    plain environment variable. Old versions are destroyed, not disabled
    (disabled ones are billed), except the ones kept revisions pin. Only a
    NOT_FOUND lookup lets a deploy fall back to plain variables, and never
    for a service that already mounts the secret. The runtime and build
    service accounts hold only the roles they use. The Colab token is the
    one other secret (`linebot-colab-adc`, invariant 18). See ADR-0008.

16. **A fetched URL never reaches inside.** Users and search results pick
    the URLs, and the service can reach the metadata server (and any
    private network it is ever connected to). `PageFetcher` accepts http/https on 80/443 only,
    resolves the host itself, refuses the host unless every address is
    public (IPv4-mapped and 6to4 unwrapped; private, loopback, link-local,
    site-local, CGNAT, multicast, reserved refused), connects to the address it
    checked, re-checks every redirect, and reads at most 2 MB of HTML or
    text, with `trust_env` off. Search never fails a reply: when no
    backend answers, the answer says it is from the model's own knowledge.
    See ADR-0009.

17. **The Colab VM is trusted only through its signature.** Nothing secret
    is stored: each session's registration key and Ollama bearer are
    derived from `COLAB_SHARED_SECRET` and the session id. A registration
    needs a fresh, unused signature for the current session; its tunnel
    URL must be `https://` plus a quick-tunnel host or `COLAB_TUNNEL_HOST`,
    resolve to public addresses only, and pass the readiness probe (Ollama
    version, the chat model wholly in VRAM at its pinned digest, think
    values) before the session is READY. A router digest other than its pin
    (or none recorded), or a router not wholly in VRAM, only turns Colab
    routing off for that session.
    All Colab traffic uses the page fetcher's pinned public-only transport.

18. **Colab's control plane stays off the request path, and its credential
    stays a file.** Assign, launch, keep-alive, release and reconcile take
    seconds to minutes, so they run only as `colab` Cloud Tasks; the cron
    tick only reads and writes the state document (at most 5 s) and queues
    them, and `/webhook` never calls Colab. The state is written before
    Colab is: a session's `nbh` is stored before its assign is queued, a
    retry reuses it, and an `nbh` given up without an answer is looked up
    later, so a timed-out assign never leaves a runtime the bot cannot
    name. The controller only queues its tasks, never runs them inline,
    on a queue of their own (`CLOUD_TASKS_COLAB_QUEUE`), so they never take
    a LINE event's dispatch slot. The Colab account's refresh token is
    minted by a person (`scripts/mint_colab_adc.sh`, ideally a dedicated
    account with no project roles), kept in `linebot-colab-adc` and mounted at
    `COLAB_ADC_PATH`, pinned, only while `COLAB_ENABLED=true`;
    `GOOGLE_APPLICATION_CREDENTIALS` is never set, or every Google client
    would run as that account. Secrets reach the VM only through the
    silent-execute env file (never `colab exec --env`, argv or a log). A
    refused credential or account turns the tier DISABLED until
    `!colab on`. See ADR-0010.

19. **Vectors are compared only within one embedding model.** The embed
    role has exactly one tier, and its width equals `EMBEDDING_DIMENSIONS`,
    the Firestore vector index's (the model's 2048 dims are cut to the first
    1024 and re-normalised). Every stored vector is tagged with
    `embedding_model` and `embedding_dim`, and `recall_episodes` drops
    candidates of another model after the vector query, so vectors from
    another model are never ranked against the current one.
    `scripts/migrate_embeddings.py` (dry run unless `--apply`) re-embeds
    them from the episode's own text, or deletes an episode that has none.
    See ADR-0011.

## Testing

`python3 -m pytest -q` — no network, no GCP credentials required;
`tests/conftest.py` enforces it (sockets reach loopback only, and Settings
environment variables are dropped at start). The
`InMemoryBackend` stands in for Firestore; seed it through
`service.backend_instance.store`. A `MemoryService` built without a
`compaction_scheduler` compacts inline, inside the call that crossed the
limit; inject one to model compaction running in its own task, and a
`clock` to step through the debounce, the episode interval and the profile
cache (`tests/test_memory_compaction.py`). `FirestoreBackend` takes a
`client`; `tests/test_firestore_backend_fake.py` runs it on a fake one. The Colab
tier runs against `tests/colab_fakes.FakeTunnel` (a scripted tunnel, auth
proxy and Ollama behind `httpx.MockTransport`) and an in-memory state store;
`tests/test_colab_pipeline.py` drives whole events through it. Cloud Tasks,
the metadata server and the LINE API (`tests/test_cost_controls.py`)
are faked with `httpx.MockTransport`, and so is Tavily
(`tests/test_search_backends.py`); the page fetcher's tests run a local HTTP
server with 127.0.0.1 allowed for the test only. The deploy and provisioning
scripts, and the Colab mint script (whose fake sign-in writes into the
scratch `CLOUDSDK_CONFIG`), run against a stateful fake `gcloud`
(`tests/fake_gcloud.py`) and, for the smoke check, a local stub of the
service.
The live gate's Colab path runs against a scripted tunnel, and the control
plane's contract tests against Colab answers shaped from the pinned
library's source. None of them reach Google.

`python3 -m ruff check .` must be clean. Config in `pyproject.toml`; the rule
set is deliberately narrow (correctness over style) and each `ignore` carries
its reason.
