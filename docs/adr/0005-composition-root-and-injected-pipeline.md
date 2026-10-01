# ADR-0005: Composition root, and a pipeline that takes its dependencies

**Status**: Accepted

## Context

`main.py` had grown to 942 lines holding six mutable module-level globals, the
lifespan wiring, three tool-executor closures, target-list construction,
reasoning filtering, health aggregation, two HTTP endpoints, and the entire
event-processing pipeline.

Because the pipeline read its collaborators from module globals
(`get_line_service()`, `get_memory_service()`, the `orchestrator` global), there
was no way to drive it in a test except by monkeypatching eight names on the
`main` module. Production code had absorbed the cost of that: `hasattr` guards
before calling `set_embedding_service` / `set_tool_registry`, and
`except AttributeError` blocks around `fetch_display_name` and
`touch_user_profile` whose comments said, in as many words, that the test
doubles might not implement the method. `MemoryService` also carried a
`_memory_store` property that existed only so fixtures could reach the backing
dict.

Error handling that exists to accommodate test doubles is not error handling.

## Decision

Three seams:

- **`src/runtime/composition.py`** — the composition root. Everything about
  *which model, in what order, with what fallback* lives here.
- **`src/runtime/container.py`** — `AppContainer`, the built result. One object
  to pass around instead of six globals.
- **`src/pipeline/event_pipeline.py`** — `process_event(event, deps)`, where
  `deps` is a `PipelineDeps` dataclass. The four function-valued fields default
  to the real implementations, so production calls stay short while tests
  substitute freely.

`main.py` keeps the HTTP surface and the process lifecycle, and is now 384
lines.

## Consequences

- The `hasattr` guards, the `except AttributeError` blocks, and the
  `_memory_store` back door are all deleted. Test doubles now declare the
  methods they stand in for, which is also how a missing one gets noticed.
- The two `process_event` tests build a `PipelineDeps` instead of stacking
  eight `patch.object` calls, and read considerably better for it.
- Tests that patch provider/agent classes must target
  `src.runtime.composition.*`, not `main.*`.
- A handful of names are re-exported from `main` under their old private
  aliases so existing import paths keep working; these are pure functions, not
  a re-established seam.
