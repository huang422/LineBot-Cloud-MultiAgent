# ADR-0003: The recent window is a compaction trigger, not a retention bound

**Status**: Accepted

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

## Decision

Crossing `memory_recent_message_limit` **schedules** compaction and nothing
else. The write paths no longer truncate there. The summary task summarizes
whatever it snapshots, however long, and then clears exactly that prefix.

A separate hard safety cap (`_recent_hard_cap`, derived from the limit and the
passive flush threshold) exists solely so a stuck or unavailable summarizer
cannot grow a Firestore document without bound. Reaching that cap *does* lose
data, so it logs a warning naming the drop count.

## Consequences

- The window routinely sits above `memory_recent_message_limit` between the
  trigger and the compaction commit. Tests asserting a hard cap at the limit
  were encoding the bug and have been updated.
- Concurrent appends during compaction now commit on the first pass: the
  snapshot stays a prefix of the current window, so the existing
  `current_recent[:snapshot_len]` guard finally matches as designed. This
  removes a redundant second LLM call per contended compaction.
- The retry path is still exercised via the `summary_version` guard, which is
  how a genuinely stale commit is now detected.
