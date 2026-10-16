# ADR-0012: Stay on the custom pipeline; keep a seam for LangGraph

**Status**: Accepted (upgrade plan decision 1, 2026-09-29). Revisit when one
of the adoption triggers below appears. This is the reasoning ADR-0007's
"Alternatives considered" refers to.

## Context

The v3 upgrade reopened the question of running the bot's orchestration on
LangGraph (typed graph state, conditional edges, checkpointers, interrupts)
instead of the hand-written pipeline. What the flow is today:

- One LINE event becomes one Cloud Task, whose handler calls
  `process_event(event, deps)` (`src/pipeline/event_pipeline.py`, about 130
  lines): commands (`!new`, `!colab`, `!whoami`) answer at once; otherwise
  flush the group's passive buffer, rate-limit and load memory, route (fast
  rules, then the Colab router on a READY Colab session, then the router role) while the
  sender's profile facts load, run one of four agents, `send_response`,
  `record_interaction`, `touch_user_profile`. One branch (the agent), a few
  early exits, and no cycle outside the chat agent's tool loop (at most 4
  rounds, then one `tool_choice="none"` answer). No step waits on a person.
- Every answer races a deadline: event time + `REPLY_WINDOW_SECONDS`
  (90 s, whoever answers), or the turn deadline (240 s from the start of
  work) for a thinking answer or an image, in any chat. What outlives the reply token
  goes out by push (reply first, push only when the reply certainly
  failed). After the deadline, nothing of the turn is worth resuming: the
  reply token is gone and a push is metered per recipient.

## Decision

Do not adopt LangGraph (or LangChain) now. Keep the custom pipeline, and keep
the seams that make a later move mechanical.

### Why not now

1. **Single-turn reply window.** Checkpoint and resume, the main thing a
   graph runtime adds, has nothing to resume here. A turn that crashed or
   ran out of time can no longer reply (the token has expired), and
   replaying it by push would bill every member of a group. The durable
   unit is the event, and it is durable enough already (point 3).
2. **The hard part is the model calls, and they are custom.**
   `FallbackChain` v2 (`src/providers/fallback_chain.py`) gives each tier its
   own time limit clipped to what is left of the request's deadline, gives the
   thinking tiers together at most half of it and a Colab tier all but a
   20 s reserve for the API tiers (half in a short window), skips text-only tiers for
   images, and asks `ModelHealth` before sending anything: dead models
   (410), cooldowns (429, repeated timeouts), provider budgets (NVIDIA 40 a
   minute and 10,000 a day, OpenRouter's free 20 a minute and the account's
   daily cap). `SkipTierError` makes a cold Colab session cost no time.
   LangChain's `with_fallbacks` and LangGraph's node retry policies know
   nothing of a deadline shared by a whole chain or of a provider budget, so
   the chain would stay as it is inside a node. The graph would only wrap
   it.
3. **Durability comes from Cloud Tasks and Firestore, not checkpointers.**
   The webhook queues each event on `linebot-events` with one attempt per
   task (ADR-0006), `webhookEventId` dedupes redeliveries, and whatever must
   outlive a request is a Firestore document written at a known point: the
   memory window with message ids and a generation (ADR-0003), the Colab
   session document with its version counter (ADR-0010). A checkpointer
   saves state after every step of a run; on Firestore that is several more
   writes per message against the 20K-a-day free tier, for resumes the reply
   window makes useless. The one long-running workflow, the Colab control
   plane, is already a state machine in one document, driven by the
   per-minute cron and leased `colab` tasks on their own queue
   (`linebot-colab`); a checkpointer would duplicate that document.
4. **Cost and footprint.** Cloud Run runs with `--memory 512Mi`,
   `--max-instances=1` and scale to zero; the measured peak RSS is about
   124 MB (DEPLOY.md, 記憶體預算). Resolving `langgraph` on 2026-09-30 pulls 38
   packages, about 20 of them not in `uv.lock` (langchain-core,
   langsmith, langgraph-checkpoint, langgraph-prebuilt, langgraph-sdk,
   orjson, ormsgpack, tenacity, xxhash, zstandard, ...): more to hash-pin,
   audit and import on a cold start, and LangSmith tracing to keep switched
   off, for no feature the bot uses.
5. **The graph is small.** Seven steps in a line with one four-way branch
   (the agents) and a handful of early exits (not for the bot, command
   answered, blocked, not delivered). Its order reads top to bottom in one
   function, and tests drive it end to end with injected deps.

### The seam kept for later

- `process_event(event, deps)` is the only entry the `event` task handler
  calls, and `PipelineDeps` carries every collaborator (ADR-0005). A graph can
  replace the function's body without touching the webhook, the queue or the
  tests that build deps.
- The steps are separate callables with plain data between them:
  `process_input` → `AgentRequest`, `Orchestrator.route` → `RouterDecision`,
  `agent.process(request)` → `AgentResponse`, then `send_response` and
  `record_interaction`. Those types are the graph state.
- Model choice is data, not control flow: the `RoleTable` (ADR-0007) and
  `BaseAgent.targets_for` pick the tiers, so a node calls the same agent with
  the same tiers.
- Tools are OpenAI-style schemas in a registry (`src/agents/tools.py`) run by
  `ToolRunner`; their executors are plain async functions, easy to wrap as
  graph tools.
- Deadlines live on the request (`received_at`, `reply_deadline()`,
  `turn_deadline()`), not in the call stack, so they cross node boundaries
  unchanged.

### When to adopt it

Any one of these makes a graph runtime pay for itself:

- **Multi-step plans with a person in the loop.** The bot proposes a plan,
  the user confirms or edits it in a later LINE message, and the run carries
  on from where it stopped (interrupt and resume across turns).
- **Long-running tool workflows** that outlive one task's budget
  (`TASK_HANDLER_BUDGET_SECONDS`, 280 s) or must survive an instance restart
  half-way, and need checkpoint/resume instead of starting over: multi-page
  research, batch image jobs, agentic coding through the Colab/opencode
  path.
- **More than about six agents with conditional edges.** Agents handing off
  to each other (search, then vision, then image), loops between agents, or
  parallel sub-agents that join, where `process_event`'s single branch would
  turn into a hand-written state machine.

Not triggers on their own: wanting traces (the request id on every log line
and `/health/detail` cover today's needs), or a new agent that still fits
"route once, answer once".

### Migration sketch

State, a `TypedDict`: `event`, `request` (`AgentRequest`), `decision`
(`RouterDecision`), `profile_facts`, `response` (`AgentResponse`),
`delivered`. `PipelineDeps` travels in the run's configuration or runtime
context, never in the state: it holds clients, which a checkpointer would try
to serialise.

| Node | Today | Leaves to |
| --- | --- | --- |
| `ingest` | `process_input`, `apply_cleaned_text` | END when the event is not for the bot |
| `command` | `is_new_chat_command`, `parse_command` + `run_command` | END once a command answered |
| `prepare` | loading animation, `message_cache.cache_processed_request`, `_flush_passive_before_reply`, `enrich_request`, `get_request_block_message` | END when blocked (rate limit, empty message) |
| `route` | `Orchestrator.route` (rules → Colab router → router role), with `_load_profile_facts` and `_note_colab_demand` alongside | a conditional edge on `decision.agent` |
| `chat`, `vision`, `web_search`, `image_gen` | `agent.process(request)`; each keeps `FallbackChain`, and chat's `ToolRunner` may become a subgraph with a tool node | `deliver` |
| `deliver` | `send_response` (reply first, push only when the reply certainly failed) | END when not delivered |
| `remember` | `record_interaction`, `touch_user_profile` | END |
| error handling | the three `except` branches of `process_event` and their messages | END |

In order:

1. Split `process_event`'s body into node functions (`state, deps` in, a
   partial state out) with no behaviour change, still called in sequence by
   `process_event`. The suite stays green.
2. Build the graph from the same functions behind the same signature
   (`process_event(event, deps)` invokes the compiled graph). Run the suite
   and the live gate on both.
3. Keep Cloud Tasks as the trigger and the dedupe as it is. Add a
   checkpointer only for the part that made the move worth it (the
   human-in-the-loop plan or the long workflow), keyed by chat or workflow,
   stored where its writes fit the free tier, and give it its own task kind
   and queue as `colab` has.
4. Deadlines stay on the request and model calls stay in the existing chain;
   the graph's recursion limit is a loop guard, not a timer.
5. Measure RSS and cold start on the 512Mi instance again before deploying.

## Consequences

- The order of steps is code in one function, not graph configuration, and
  every change to it is a code change with its tests.
- No graph visualisation and no LangSmith traces; the logs carry the request
  id through every step.
- Every new feature that spans turns or outlives a task is checked against
  the triggers first. Hand-rolling resume logic on the custom pipeline is
  the signal to do the migration instead.
