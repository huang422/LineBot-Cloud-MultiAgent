# ADR-0011: Embeddings on nemotron-3-embed-1b, sliced to the 1024-d index and tagged by model

**Status**: Accepted (v3 Phase 1). Running the migration is a manual step
(DEPLOY.md, 上線前檢查清單 and 從 v2 升級到 v3).

## Context

Episodic recall (the chat agent's `recall_memory` tool) stores one vector per
episode in `{prefix}_episodes/{chat_key}/items` and searches it with
Firestore `find_nearest` (cosine) over a flat vector index: collection group
`items`, field `embedding`, **1024 dimensions** (DEPLOY.md, 啟用 Episodic
Memory).

- v2 embedded with `nvidia/nv-embedqa-e5-v5` (1024-d). NVIDIA retired it on
  2026-08-25; every call answered HTTP 410. Episodes were still written, but
  without a vector, and every recall came back empty. The failure was only
  a warning, so it went unnoticed for weeks (ADR-0007).
- The replacement on NVIDIA's free API, `nvidia/nemotron-3-embed-1b`
  (multilingual, zh-TW included), always returns **2048** dimensions, and the
  hosted endpoint refuses a reduced `dimensions` request.
- A Firestore vector index has one fixed width. A 2048-d index means a new
  index (minutes to build, a second set of index entries against the 1 GiB
  free storage) and every stored vector rewritten before recall works again.
- Vectors from two models live in different spaces. A cosine score between
  an e5 passage and an embed-1b query means nothing, yet same-width vectors
  fit the same index, so the index would rank them side by side without
  complaint.

## Decision

### Keep the 1024-d index: slice and re-normalise

- `EmbeddingService` (`src/services/embedding_service.py`) never sends
  `dimensions`. It keeps the first `EMBEDDING_DIMENSIONS` (1024) values of
  each 2048-d vector (the Matryoshka prefix) and L2-normalises them again, as
  the model card requires. Results are ordered by their `index`; a response
  that does not hold exactly one vector per input is discarded whole, so
  batch callers can zip results with inputs. A zero vector is returned as
  is; a vector narrower than 1024 fails the whole response with a warning
  (never padded, never stored or queried), like any other embedding failure.
- Stored episodes are embedded with `input_type=passage`, recall queries
  with `query`, both with `truncate=END`. The passage is
  `episode_passage_text`: the summary first, then the archived conversation
  (`episode_snippet`), so an over-long input loses the end of the
  conversation, not the summary.
- The `embed` role has **exactly one tier**: two models would mean two
  vector spaces in one index, so there is no fallback. The tier's `dim` must
  equal `EMBEDDING_DIMENSIONS` (the index width) and may not exceed the
  model's native 2048 (`model_profiles.native_dim`); startup refuses
  anything else (`RoleConfigError`). `nvidia/nv-embedqa-e5-v5` is in
  `RETIRED_MODEL_IDS`, so the role table, `scripts/envfile.py check-v3` and
  the migration script all refuse it.
- Embedding calls go through `ModelHealth` like chat calls: they count
  against NVIDIA's shared budget (40 a minute, 10,000 a day), a 410 marks the
  model dead for the process (one ERROR per process, not one per recall),
  and the daily `eol` task checks the id with `GET /v1/models/{id}`. The live
  gate's `nvidia.embed.dim` check confirms 2048 from the API, 1024 kept,
  norm 1.

### Tag every vector; recall compares only its own model

- An episode written with a vector also gets `embedding_model` (the embed
  tier's model id) and `embedding_dim` (`len(vector)`). An episode whose
  embedding failed or was skipped (budget, cooldown, dead model) is not
  saved: its conversation stays in `episode_pending_text` (newest 2000
  chars) and joins the next compaction's episode (WARNING; skip reason at
  INFO). `find_nearest` never returns a document without the vector field.
- `MemoryService.recall_episodes` embeds the query, asks the backend for
  `2 × k` nearest neighbours (k is at most 5), drops every candidate whose
  `embedding_model` is not the current embed model, and returns the first k.
  The width is fixed by the index, so recall checks the model tag only.
  Skipped candidates are logged at INFO as a count with a pointer to the
  script, never their text.
- The filter runs after the vector query. Firestore can pre-filter a vector
  query only through a composite vector index that includes the filtered
  field; one plain index is kept instead, and the over-fetch leaves room for
  the few old vectors a chat still has.

### Migrate with `scripts/migrate_embeddings.py`

A person runs it once per model change, from the repo root, with their
local ADC (`gcloud auth application-default login`) and `.env`
(`--env-file`). **Dry run by default** (counts only); `--apply` writes.

- The model is resolved exactly as the app resolves it
  (`build_embedding_service(settings)`: the embed role's code default or its
  `MODEL_ROLES` override), so migrated vectors carry exactly the tag recall
  filters on. A role table the app would refuse, or an embed role that
  resolves to a retired id, stops the script with exit code 2 before
  anything is read or written.
- It walks `{prefix}_episodes/*/items` (through `list_documents`, because the
  parent documents exist only as paths) and sorts every episode into one of
  three groups:
  - **keep**: tagged with the current model and dimension, vector present;
  - **re-embed**: anything else with a non-blank `summary`: untagged e5
    vectors, episodes saved without a vector during the outage, another
    model, another width, a tag without a vector. It embeds
    `episode_passage_text` in passage mode (older episodes' raw `messages`
    are read through `episode_snippet`, as the app reads them), in batches
    of up to 16 at most `--max-rpm` requests a minute (default and maximum
    40), and updates `embedding`, `embedding_model` and `embedding_dim` in
    place. Nothing else changes: `expires_at` keeps each episode's original
    90-day TTL;
  - **delete**: no summary, so no text to embed from. The **whole episode
    document** is deleted, not just its vector: an episode without a summary
    has nothing for recall to return.
- Deletions run first, then the batches. A batch whose request fails is
  counted and left untouched, so a re-run picks it up (exit code 1). With
  episodes to re-embed and no `NVIDIA_API_KEY`, it exits 2 without writing.
- Output is counts and categories only (`stale because untagged (legacy)`,
  `other model`, `missing vector`, `other dimension`), never text, chat ids
  or keys; the app logger is held at ERROR because an HTTP error body can
  quote the input.
- The script has no `ModelHealth`. Its throttle keeps it under NVIDIA's
  40 a minute, but that limit belongs to the key, which the live bot shares:
  run it at a quiet hour, or with a lower `--max-rpm`.

## Consequences

- No new index and no rebuild: the existing 1024-d index, its storage and
  the TTL policy stay as they are. Slicing gives up some of the 2048-d
  model's quality; zh-TW recall at 1024 has not been measured. Moving to the
  full width is a dimension change (below).
- **Untagged vectors before the migration** stay in the index and in
  storage. `find_nearest` still ranks them, and recall drops them. A chat
  with many of them can fill all `2 × k` candidates with old vectors, so its
  recall returns fewer than k, or nothing, until the script runs. They are
  never compared with a new vector. With the TTL policy on `expires_at`
  enabled they expire 90 days after they were written: the last e5 vector
  predates 2026-08-25, so without a migration they, and their episodes, are
  gone by about 2026-11-23.
- The script is idempotent: a second run after a clean one reports every
  episode as keep.
- **Rollback.**
  - Rolling Cloud Run back to another v3 revision needs nothing: same model,
    same tags.
  - Rolling back to a v2 revision does not bring recall back: e5 answers
    410, so recall stays empty. The migrated vectors are left untouched.
  - Changing the embed model later (the `embed` tier in `MODEL_ROLES`): recall
    ignores every stored episode until the script runs with the new model,
    and switching back needs another run with the old one. Vectors are
    overwritten in place and no copy of the old vector is kept, but the
    source text stays, so any episode with a summary can be re-embedded for
    either model at any time. Deleted episodes without a summary are gone
    for good.
  - Changing `EMBEDDING_DIMENSIONS` needs a vector index of the new width
    first (the old one no longer matches the query vectors), then the script
    (every episode is stale by dimension), then deleting the old index.
- Search rerank (`SEARCH_RERANK_ENABLED`, ADR-0009) uses the same embed role
  but compares only vectors computed in the same call, so tags do not
  matter there.
- Tests: `tests/test_migrate_embeddings.py` (classification, planning,
  batching, throttling, apply and exit codes, with fake Firestore and
  embedder), `tests/test_phase_b_episodic_memory.py` (tags on write, the
  recall filter), `tests/test_model_roles.py` (one embed tier, `dim` checks).

## Alternatives considered

- **A 2048-d index (or a second vector field).** Full quality, but a new
  index and a double set of index entries during the change, and every
  vector rewritten before recall works. Kept as the path if the 1024 prefix
  proves too weak for zh-TW.
- **Re-embed lazily on recall.** Puts embedding calls and Firestore writes on
  the reply path, inside the reply window. The batch script keeps them off
  the request path and under the rate limit.
- **Delete every legacy episode.** Simplest, but throws away recall whose
  text is still there.
- **Compare old and new vectors anyway.** The scores are meaningless.
- **A fallback embed tier** (another provider or model). Two vector spaces
  in one index again; the role table refuses a second embed tier.
