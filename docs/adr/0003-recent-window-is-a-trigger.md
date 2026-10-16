# ADR-0003: The recent window is a compaction trigger, not a retention bound

**Status**: Accepted (amended in v3 Phase 5b: commit by message id, time
order, debounce, chat generation; then replies stamped with their
question's time, retry holds, tier-by-tier summaries; 2026-10-16: the
summary is asked for as a change and held to 400 characters)

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

The tier is asked for **the change, not the summary** (2026-10-16). The
request lists the old summary's items (its sentences and `；` clauses) by
number; the answer's first line names the items to remove (`移除：2,5`, or
`移除：無`) and each further line is a new item. The code keeps the other
old items word for word and adds the new ones it does not already have.
The removal line is also read as `移除項目：`, `**移除**：`, `1. 移除：` or
`移除 無`, and only the numbers its value starts with count (`移除：無，第2項
仍有效` removes nothing); a bare `移除：` takes the lines of numbers under
it. Markdown, code fences, the `[2] ` an old item was listed with,
headings (`新增：`, any line ending in `：`) and lines that only say there
is nothing new (`- 無`, `（無新增項目）`) are not items. An
answer without the removal line whose lines are all bullets, or that has
such a heading, is new items too, and one that only says nothing changed
(`無需更新`) keeps the old summary, like `移除：無`. Any other answer is a
whole summary from a tier that ignored the format: it replaces a blank
summary; else it must repeat at least half the old items word for word,
and only its other items are added, after all the old ones. A partial one
(`丁養了柴犬。`) would lose the old items, so that tier is rejected like
one cut off at `max_tokens` and the next one tries.
The summary is held to `SUMMARY_MAX_CHARS` (400 characters): one over it,
the old one before the request or the new one after it, is trimmed. A
utility tier ranks its items (item numbers, best first); the code keeps
them in rank order while they fit, then the unranked ones that still fit,
in their original order, so only the cap decides what is lost. A ranking
that is not just numbers, or was cut off at its `max_tokens`, is no
ranking. After the update, the items it added are kept when they take at
most half the cap and only the old ones are ranked, for what is left: a
fact just learned is not dropped by the same pass. An item longer than
what is left (an old one-paragraph summary) is ranked in pieces cut at its
commas. When no tier ranks the items the summary stays whole and the next
pass tries again. Sentences a model wrote about the update itself
("以上內容…予以保留", "（其餘項目保留）", "目前無衝突內容需要移除") are
dropped from both, and so are notes about something left out ("天氣查詢屬
即時資訊，不納入長期記憶"): with the clauses before it when the note leads
a later clause (they are what was left out; the clauses after it stay),
only its own clause when it has its own subject ("使用者關注台積電，股價查
詢屬即時資訊不納入長期記憶" keeps "使用者關注台積電"). A summary left blank is rejected
unless the old one was blank too (nothing worth keeping yet). The trims and
the update share the pass's one deadline and its record of models that
timed out, so a model that hung in one of them is not waited on again.

Why: on 2026-10-16 one group's compaction failed on every pass. Asked for
the whole summary, NIM Super (thinking off, the first utility tier) copied
the old one word for word whatever length the prompt named (the old "about
300 characters", or the current length, a cap and an order to condense),
and appended 150-560 characters of new detail; asked only to condense, it
dropped a few items and stopped near 560 characters. At about a token per
character of Chinese (0.95), the summaries had grown pass by pass to 794
characters, which Super could no longer copy within
`MEMORY_SUMMARY_MAX_TOKENS` (768) with anything appended: each pass was cut
off and rejected, then Ultra hung and dots-3 answered a 200 whose error
object had code 502. Asked for the change, Super answered that summary in
2-10 s with 70-320 tokens and ranked its items in under a second; 10
passes of 10 finished at 344-390 characters, at or under the cap, and a
review's 3 more at 388-395. `MEMORY_SUMMARY_MAX_TOKENS` keeps 768: the answer no longer
repeats the summary, and the trim before the request keeps even a tier
that copies the old items anyway inside it.

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
  instead of one per flush. A summary over the cap adds a ranking call
  before the request, after it, or both; each walks the utility tiers like
  the update, under the same 180 s deadline. With Super answering a
  ranking takes about a second; a model that hangs costs its tier timeout
  once per pass, not once per walk.
- Old items are never reworded unless the tier removes one and writes it
  again, so the summary reads as a list of facts in the order they were
  learned. A long item (a transcribed list can take 250 characters) leaves
  room for few others: the first pass over the 794-character summary of
  2026-10-16 (613 after its meta sentences) keeps 2 of its 6 items. A new
  item survives its own pass and competes with the old ones from the next.
- A tier that ignores the format now fails the pass instead of replacing
  the memory; the live gate's utility check sends a non-empty summary and
  fails such a tier.
- Each commit also decides the chat's episode: at most one per
  `EPISODE_MIN_INTERVAL_SECONDS` (3600). Commits in between append their
  conversation to `episode_pending_text`, which the next episode archives.
  That text keeps only the newest 2000 characters, so in a busy hour the
  episode holds the latest conversation, and what was compacted earlier in
  the hour reaches only the rolling summary. Writing an extra episode
  whenever the text overflows would have kept it all, at the cost of the
  hourly bound.
