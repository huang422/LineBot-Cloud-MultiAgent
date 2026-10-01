# ADR-0006: Cloud Tasks for all post-webhook work

**Status**: Accepted

## Context

The service runs on Cloud Run with request-based billing
(`--cpu-throttling`): CPU is allocated only while a request is being
processed. The webhook used to answer LINE with 200 and then continue in
`asyncio.create_task` — routing, the model calls, the reply, memory writes.
All of that ran *after* the response, so it got close to no CPU. Replies were
slow or late, and the reply token (valid for about a minute) often expired,
forcing a metered push.

The same flaw sat in three other places: the passive group buffer flushed on
an `asyncio.sleep(30)` timer, memory compaction ran as a background task, and
generated media was deleted by an in-process 2-day `sleep`.

LINE expects the webhook to answer within 2 seconds, so the work cannot
simply move into the webhook request either.

## Decision

Every piece of work that used to run after the response is a **task**: a
`kind` plus a small JSON payload.

- `event` — process one triggered LINE event (`process_event`)
- `passive_flush` — persist one chat's passive buffer
- `cache_image` — download and cache an untriggered image for quoted replies
- `compact` — run one memory compaction pass
- `eol` — the daily model end-of-life check, queued by the first
  `/internal/cron` tick of each Asia/Taipei day (added later; ADR-0007)

In production `CloudTasksQueue` creates an HTTP task on the `linebot-events`
queue (REST API, token from the metadata server). Cloud Tasks then POSTs to
`/internal/tasks/{kind}` on this service. That is a real request, so the work
gets CPU for its whole duration. Locally, and whenever Cloud Tasks is not
configured, `InlineTaskQueue` awaits the same handler directly.
`TASKS_MODE=auto` picks Cloud Tasks only on Cloud Run (`K_SERVICE` set) with
`INTERNAL_TASKS_SECRET` set.

The webhook now only verifies the signature, drops duplicates, buffers
untriggered group messages, caches text messages inline, and queues the rest
(concurrently). If queueing fails, the webhook runs that handler inline and
logs an ERROR, so nothing is dropped. Each task is created under a random
name, so an unclear answer from Cloud Tasks (a timeout or 5xx after the
request went out) is retried once under the same name; 409 ALREADY_EXISTS
means the first try created it. If the retry is unclear too, the task may
already exist: only an `event` still runs inline (its handler drops a second
delivery), and the other kinds are left to the queue rather than run twice.

Nothing sleeps in-process any more:

- the passive buffer tells the webhook when a chat is ready (threshold
  reached, or a new message after the oldest one is
  `PASSIVE_GROUP_FLUSH_DELAY_SECONDS` old) and the webhook queues a
  `passive_flush`. If the buffer is still there one delay after that request
  (the task was lost), the next message asks again. An `event` task flushes
  its group's buffer before building the reply, so earlier untriggered
  messages are in the context and stored before the turn. Chats that go
  quiet are flushed by the per-minute `/internal/cron` sweep
  (`run_cron_tick`), which checks its 30 s budget between chats and never
  cancels a flush half-way; shutdown flushes the rest. A failed or cancelled
  flush puts its batch back, and each buffer is capped at the recent-window
  hard cap;
- compaction goes through an injected scheduler that queues a `compact` task;
- media expires through the bucket's 3-day lifecycle Delete rule, which the
  deploy script enforces. An upload LINE never received is deleted inline.

## Alternatives rejected

- **Instance-based billing** (`--no-cpu-throttling`) would give background
  tasks CPU, but CPU is then billed for the instance's whole lifetime. The
  per-minute Cloud Scheduler job keeps the instance from ever going idle, so
  that is about 30 × 86,400 ≈ 2.6M vCPU-seconds a month against a free tier
  of 240K. Cloud Tasks costs 2 billable operations per event (create +
  dispatch) against a free tier of 1M operations a month. Task requests do
  use Cloud Run's request-based tier, though; see Consequences.
- **Doing the work inside the webhook request.** Correct for CPU, but a model
  call takes far longer than LINE's 2-second limit. It remains the fallback
  only.
- **OIDC-authenticated tasks.** Needs `iam.serviceAccountUser` for both the
  runtime account and the Cloud Tasks service agent, plus token verification
  in the app. The service is public (`allUsers` invoker) anyway, so a shared
  secret checked with `hmac.compare_digest` gives the same protection for
  less setup.

## Consequences

- **At-least-once delivery.** Both LINE (with webhook redelivery enabled) and
  Cloud Tasks can deliver twice. Events are deduped by `webhookEventId`
  (falling back to the message id) in an in-process TTL/LRU set
  (`src/utils/dedupe.py`: 15 minutes, 4,096 keys). The webhook and the task
  handler keep separate sets: the webhook marks an event seen before queueing
  it, so a shared set would drop every task. Dedupe is process-local, which
  is sufficient only while `--max-instances=1`.
- **The queue makes one attempt per task.** It is created with
  `--max-attempts=1` and **no** `--max-retry-duration`: Cloud Tasks stops
  retrying only when both limits are satisfied, so a retry duration would
  bring retries back. Once authenticated, `/internal/tasks/{kind}` always
  answers 200 with `{"ok", "kind", "ms"}`; a 5xx would only be logged by
  Cloud Tasks and dropped.
- **The inline fallback may exceed LINE's 2-second limit.** LINE then counts
  a webhook error (and redelivers if redelivery is on; dedupe drops the copy).
  Replies still go out; this is the degraded path, not the normal one.
- **Payloads stay under 32 KB** (one billable operation). A larger payload is
  not queued; it runs inline.
- **Task requests are billed Cloud Run time.** Work that used to run
  CPU-starved after the response was not billed; now each task is a request
  billed for its whole wall-clock time (1 vCPU, 512 MiB), including time spent
  waiting on a model, and each triggered event is two requests (webhook +
  task). The request-based free tier is 180,000 vCPU-s and 360,000 GiB-s a
  month. For example, 300 triggered events a day at 30 s each is about
  270,000 vCPU-s a month, over the CPU tier by about US$2 at $0.000024 per
  vCPU-s (memory stays within its tier). Set a billing budget alert; lowering
  `TASK_HANDLER_BUDGET_SECONDS` caps the worst case per task.
- **Auth is a separate secret.** Cloud Tasks sends `X-Tasks-Token`
  (`INTERNAL_TASKS_SECRET`). It is deliberately not the cron secret; an empty
  secret makes the endpoint refuse everything, and the deploy script refuses
  to deploy without it unless `TASKS_MODE=inline`.
- **The task target URL** is `PUBLIC_BASE_URL`, or the `Host` of the first
  authenticated request (webhook, task or cron), so a forged `Host` header
  cannot redirect tasks. It is always https: tasks carry the shared secret,
  and an http target would hit run.app's redirect, which counts as the one
  failed attempt.
- **Deadlines.** `dispatchDeadline` is `TASK_DISPATCH_DEADLINE_SECONDS` (300 s),
  and the deploy sets Cloud Run's `--timeout` to max(300 s, that deadline):
  at its timeout Cloud Run answers 504 but leaves the handler running with no
  CPU. Each handler runs inside `asyncio.timeout(TASK_HANDLER_BUDGET_SECONDS)`
  (280 s) so it gives up first; the deploy refuses a budget at or above the
  deadline. An event turn is cut off 10 s before the budget
  (`EVENT_NOTICE_SECONDS`), so a turn that has not answered still gets a
  failure notice.
- **One queue, four slots, all kinds.** `event`, `compact`, `cache_image` and
  `passive_flush` share `--max-concurrent-dispatches=4`. Background tasks are
  normally short (one compaction per chat at a time), but when the summary
  models stall a few `compact` tasks can hold every slot for up to
  `MEMORY_SUMMARY_TIMEOUT_SECONDS`, and `event` tasks then wait past the
  reply window and fall back to push. A second queue for background kinds
  is the fix if this shows up in practice.
- **A paused queue drops work silently.** It still accepts tasks, so the app
  never falls back inline, but it dispatches none. The deploy refuses to
  continue unless the queue is `RUNNING`. Enqueue failures log Google's
  `error.status` and a short `error.message` (a queue re-activating after 30
  idle days also answers 404, like a missing one).
- **No ordering across tasks.** Two messages from one user can be processed
  out of order. The window is kept in event-time order and compaction commits
  by message id (ADR-0003), so memory is safe; replies may occasionally
  arrive out of order.
- **Passive buffers are process-local.** A `passive_flush` task that lands on
  another instance flushes nothing; the owning instance asks again on the
  chat's next message, and its cron sweep or shutdown picks the buffer up.
  Without `INTERNAL_CRON_SECRET` there is no sweep: a chat that goes quiet
  below the threshold waits for its next message, a trigger, or shutdown.
- **Compaction is one task per chat at a time.** A chat stays marked pending
  (in-process, 15-minute TTL so a lost task cannot block it) until its pass
  finishes. A pass that leaves the window at or over the limit queues the
  next one.
- **Reply window groundwork.** `AgentRequest.received_at` comes from the LINE
  event timestamp and `deadline = received_at + REPLY_WINDOW_SECONDS`. At
  first the only consumer was a warning when an `event` task starts after
  the window: the reply token has likely expired and delivery will fall back
  to push. The deadline has since become the answer deadline the fallback
  chain works to (README, "Reply window rules": one 90 s window for every
  plain answer, Colab or API, since 2026-10-01, and `TURN_DEADLINE_SECONDS`
  for a thinking answer or an image in any chat; the reply is always tried
  first and an answer whose token LINE refuses goes out by push, which
  nothing counts or refuses since 2026-10-01); the warning stays.
- **Deploy.** The deploy script enables `cloudtasks.googleapis.com`, creates
  or updates the queue (`--max-concurrent-dispatches=4`,
  `--max-dispatches-per-second=5` to protect the single instance) and grants
  the runtime service account `roles/cloudtasks.enqueuer` on it. LINE webhook
  redelivery is a console setting and has to be turned on by hand.
