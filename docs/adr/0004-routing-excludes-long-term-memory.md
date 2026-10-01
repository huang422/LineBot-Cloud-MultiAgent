# ADR-0004: Routing messages exclude the long-term memory summary

**Status**: Accepted

## Context

`Orchestrator._llm_classify` must return a JSON object. Including
the long-term memory summary in the routing prompt measurably destabilised
that: the classifier started answering the remembered topic instead of
classifying the current message.

## Decision

`_build_routing_messages` builds from `replace(request, memory_summary="")`.
Quoted context and recent conversation history are kept — they are what
follow-up detection needs — but the summary layer is dropped for
classification only. Downstream agents still receive the full summary.
The sender's profile facts are long-term memory too: the `Orchestrator`
sets `sees_user_profile = False`, so its messages never carry the
`[使用者 profile，僅供參考]` block that chat, vision and web_search get.
Because routing never reads them, the pipeline loads them while the router
classifies, not before.

The profile is per user, not per chat, so facts given in a 1:1 chat also
reach that user's group replies, which every member reads. They stay useful
there (a nickname, a city for "near me"), so they are not withheld; instead
a group or room reply's system prompt carries `PROFILE_GROUP_PRIVACY_LINE`
(do not bring the profile up unless the sender does in this message), and
the three reply prompts say the same. Storing each fact's source chat and
filtering by it was considered and left out: it needs a profile schema
change and still guesses what a user considers private.

The Colab router, nimble on `/v1/systemone` (ADR-0010), follows the same
rule with less: its state is the message clipped to 300 characters, whether
an image is attached or quoted, whether a message is quoted, and the
previous agent. No summary, profile facts or history reach it.

The routing messages are text only: an image, sent or quoted, becomes a
`[圖片]` marker (the router has to know an image is there, not see it), so
an image never makes the chain skip the text-only router tiers. The
structured routing context (previous route, task and output format, the
last turns, whether the message quotes something) is a fenced user turn
right before the user's message, which is labelled as the current message.

## Consequences

- Routing and answering see deliberately different context. This looks like an
  oversight; it is not. There is a regression test named
  `test_routing_messages_exclude_long_term_memory_summary`.
- Unparseable classifier output is re-asked once via `_build_repair_messages`
  (the same system prompt, the bad reply as the assistant turn, then a user
  turn asking for one complete JSON object with short fields; no assistant
  prefill, since NVIDIA NIM needs the last turn to be the user's), then falls
  back to `chat` + `text` + thinking disabled.
