# ADR-0004: Routing messages exclude the long-term memory summary

**Status**: Accepted

## Context

`Orchestrator._llm_classify` must return a single-line JSON object. Including
the long-term memory summary in the routing prompt measurably destabilised
that: the classifier started answering the remembered topic instead of
classifying the current message.

## Decision

`_build_routing_messages` builds from `replace(request, memory_summary="")`.
Quoted context and recent conversation history are kept — they are what
follow-up detection needs — but the summary layer is dropped for
classification only. Downstream agents still receive the full summary.

## Consequences

- Routing and answering see deliberately different context. This looks like an
  oversight; it is not. There is a regression test named
  `test_routing_messages_exclude_long_term_memory_summary`.
- Unparseable classifier output is repaired once via `_build_repair_messages`
  (strict re-prompt plus an assistant prefill of `{"agent":"`), then falls back
  to `chat` + `text` + thinking disabled.
