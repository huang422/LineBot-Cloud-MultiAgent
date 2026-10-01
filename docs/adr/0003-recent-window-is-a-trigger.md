# ADR-0003: The recent window is a compaction trigger, not a retention bound

**Status**: Accepted (amended in v3 Phase 5b: commit by message id, time
order, debounce, chat generation; then replies stamped with their
question's time, retry holds, tier-by-tier summaries)

## Context

Both write paths — `record_interaction` and `_flush_passive_buffer` — used to
append incoming messages, immediately truncate to
`memory_recent_message_limit`, and only then schedule summary compaction. The
compaction task snapshots the *stored* window, so anything truncated was gone
before it could ever reach the long-term summary.

The shipped defaults made this certain rather than theoretical:
`passive_group_flush_threshold` is 8 against a `memory_recent_message_limit`
of 6, with passive recording enabled by default. Every single flush discarded
at least two messages. `record_interaction` lost turns too whenever compaction
could not run — no NVIDIA provider configured, or any provider outage.

This directly contradicted the documented purpose of passive recording:
recalling the full conversation thread, not just turns where the bot was
mentioned.

The first fix committed a summary only when its snapshot was still a
*prefix* of the stored window. That broke at the hard cap (below): once the
window was full, every passive flush trimmed its head, the snapshot was no
longer a prefix, the commit was skipped and another pass scheduled. In a busy
group with a slow summarizer this livelocked — a summary call per flush,
nothing ever committed, and the cap kept discarding messages.

## Decision

Crossing `memory_recent_message_limit` **schedules** compaction and nothing
else. The write paths no longer truncate there. The summary task summarizes
whatever it snapshots, however long.

Every stored message has an `id` (12 hex characters) and a `created_at`: the
LINE event timestamp for user and group messages, and the question's for the
bot's reply. The window is kept sorted by `created_at` (a stable sort), so a
group message flushed around a turn lands where it was sent, and a reply
stays right after the question it answers. Stamping the reply when it was
recorded put two quick messages' turns in the order question, question,
reply, reply: two assistant turns in a row, which strict chat templates
reject (the history also joins any consecutive replies left in a stored
window).

The commit removes **exactly the snapshot's messages, by id**. A message
stored before ids existed is matched by `(created_at, role, content)`,
because a write since the snapshot may have given the stored copy an id.
Whatever was written meanwhile stays, wherever it sorts; a snapshot message
the hard cap already dropped is simply gone, and the new summary covers it.
The commit is skipped in exactly two cases, which is how a genuinely stale
commit is detected:

- another summary was committed since the snapshot (`summary_version`);
- the chat was cleared by `!new` since the snapshot. `!new` deletes the
  document (keeping, in a new one, only messages sent after the `!new`),
  so the version alone cannot tell: the next document starts at
  version 0 again, like the one a first compaction snapshotted. Every new
  chat document therefore gets a random `generation`, and a commit whose
  snapshot carried another one drops its summary and its episode, which
  belong to the deleted conversation; the episode is embedded before the
  commit and saved inside it, so it is dropped with the summary. Documents
  stored before the field existed have none, which matches until their
  next `!new`. A turn or passive message sent before the `!new` but written
  after it is dropped too (a process-local mark, like the chat locks).

A separate hard safety cap (`_recent_hard_cap`, derived from the limit and the
passive flush threshold) exists solely so a stuck or unavailable summarizer
cannot grow a Firestore document without bound. Reaching that cap *does* lose
data, so it logs a warning naming the drop count.

Compaction is **debounced**: a chat that crosses the limit within
`COMPACTION_DEBOUNCE_SECONDS` (120) of its last commit (`last_summarized_at`)
is not compacted until a later message finds the debounce over, unless its
window has reached half the hard cap. A pass that leaves a full window
schedules the next one on the same rule. There is no timer (ADR-0001): a
chat that goes quiet inside the debounce keeps its window, which the next
message compacts.

One pass per chat is queued or running at a time. A pass that fails, or
whose task never started (the queue makes one attempt), holds the chat for
15 minutes, so a failing summarizer is not called on every message; a window
at half the hard cap waits only a minute, so a lost task cannot leave the
cap to drop what no summary has seen. A running pass holds every trigger and
checks the window again when it ends. The summary walks the utility tiers
one at a time: the chain accepts an answer cut off at `max_tokens`, which
would become the next pass's base and lose its tail every time, so such an
answer, or a blank one, moves on to the next tier, and a pass with none left
fails. A `MEMORY_SUMMARY_MAX_TOKENS` below 768 (v2's 384) is warned about at
startup.

## Consequences

- The window routinely sits above `memory_recent_message_limit` between the
  trigger and the compaction commit (up to half the hard cap inside the
  debounce). Tests asserting a hard cap at the limit were encoding the bug
  and have been updated.
- Concurrent writes during compaction never cost the summary: the first pass
  commits, and the livelock cannot recur. The regression test is
  `test_the_hard_cap_trimming_mid_compaction_no_longer_livelocks` in
  `tests/test_memory_compaction.py`.
- Every write of a chat document runs under the chat's lock, and a write
  whose caller is cancelled (the task handler's timeout) is waited for
  before the cancellation goes on (`FirestoreBackend._write`): cancelling
  does not stop the executor thread, so the write could otherwise land
  after the next writer loaded the document, and one of the two would be
  lost (`test_a_save_cancelled_before_it_landed_is_not_lost` in
  `tests/test_firestore_backend_fake.py`).
- `!new` sent while a compaction is summarizing leaves the new conversation
  clean: no old summary, no episode of the deleted one
  (`NewDuringCompactionTests` in the same file).
- Busy groups pay for at most one summary per chat per debounce period
  instead of one per flush.
- Each commit also decides the chat's episode: at most one per
  `EPISODE_MIN_INTERVAL_SECONDS` (3600). Commits in between append their
  conversation to `episode_pending_text`, which the next episode archives.
  That text keeps only the newest 2000 characters, so in a busy hour the
  episode holds the latest conversation, and what was compacted earlier in
  the hour reaches only the rolling summary. Writing an extra episode
  whenever the text overflows would have kept it all, at the cost of the
  hourly bound.
