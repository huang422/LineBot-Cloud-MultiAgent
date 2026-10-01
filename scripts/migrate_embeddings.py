"""Re-embed episodic-memory vectors after an embedding-model change.

A human runs this once per model switch, from the repo root, with the
owner's local Application Default Credentials (``gcloud auth
application-default login``) and the settings in ``.env``:

    python3 scripts/envfile.py migrate-v3 --file .env   # first: drop retired ids
    python3 scripts/migrate_embeddings.py               # dry run: counts only
    python3 scripts/migrate_embeddings.py --apply       # re-embed / delete

The embedding model is resolved exactly as the app resolves it: the embed
role of the model role table (its code default, or a ``MODEL_ROLES``
override). A role table the app would refuse (e.g. a MODEL_ROLES embed
tier still naming the retired nv-embedqa-e5-v5) stops the script with exit
code 2 before anything is read or written.

It walks ``{prefix}_episodes/*/items`` and classifies every episode:

- keep: tagged with the embed role's model and dimension;
- re-embed: untagged (legacy nv-embedqa-e5-v5), another model or another
  dimension, and the ``summary`` text is still there. The episode is
  embedded again in ``passage`` mode from the same text the app embeds a
  new one from (``episode_passage_text``: the summary, then the archived
  conversation), in batches of 16 at most 40 requests a minute to stay
  inside NVIDIA's free limits, and the ``embedding`` Vector,
  ``embedding_model`` and ``embedding_dim`` fields are updated;
- delete: no summary text to re-embed, so the stale vector can only be
  dropped.

Old vectors share the 1024-dim Firestore index with new ones but live in a
different vector space, so recall ignores them until they are migrated
(``MemoryService.recall_episodes`` filters on ``embedding_model``).

Output is counts only: never document text, chat ids or keys. Tests drive
the planning and apply helpers with fake clients; they never touch
Firestore, NVIDIA or ``.env``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

KEEP = "keep"
REEMBED = "reembed"
DELETE = "delete"

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_RPM = 40


def _import_src() -> None:
    """Make ``src`` importable when run as ``python3 scripts/...``."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def retired_model_ids() -> frozenset[str]:
    _import_src()
    from src.config import RETIRED_MODEL_IDS

    return frozenset(RETIRED_MODEL_IDS)


def classify_episode(data: Mapping[str, Any], *, model: str, dimensions: int) -> str:
    """Return KEEP, REEMBED or DELETE for one stored episode."""
    if (
        data.get("embedding_model") == model
        and data.get("embedding_dim") == dimensions
        and data.get("embedding") is not None
    ):
        return KEEP
    summary = data.get("summary")
    if isinstance(summary, str) and summary.strip():
        return REEMBED
    return DELETE


def _stale_reason(data: Mapping[str, Any], *, model: str) -> str:
    """Why a non-KEEP episode is stale; categories only, never values."""
    tagged = data.get("embedding_model")
    if not tagged:
        return "untagged (legacy)"
    if tagged != model:
        return "other model"
    if data.get("embedding") is None:
        return "missing vector"
    return "other dimension"


@dataclass
class ReembedItem:
    ref: Any
    # What is embedded: episode_passage_text of the stored episode.
    text: str


@dataclass
class MigrationPlan:
    model: str
    dimensions: int
    chats: int = 0
    keep: int = 0
    reembed: list[ReembedItem] = field(default_factory=list)
    delete: list[Any] = field(default_factory=list)
    stale_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def scanned(self) -> int:
        return self.keep + len(self.reembed) + len(self.delete)


@dataclass
class ApplyResult:
    reembedded: int = 0
    deleted: int = 0
    failed: int = 0
    requests: int = 0


def iter_episode_documents(
    client: Any, episodes_collection: str
) -> Iterator[tuple[str, Any, dict[str, Any]]]:
    """Yield ``(chat_key, doc_ref, data)`` for every stored episode.

    ``list_documents`` also returns parent documents that only exist as a
    path (``items`` is added with ``.add`` and the parent never gets
    fields), which ``stream`` on the parent collection would miss.
    """
    for parent in client.collection(episodes_collection).list_documents():
        for snapshot in parent.collection("items").stream():
            yield parent.id, snapshot.reference, snapshot.to_dict() or {}


def plan_migration(
    documents: Iterable[tuple[str, Any, Mapping[str, Any]]],
    *,
    model: str,
    dimensions: int,
) -> MigrationPlan:
    _import_src()
    from src.services.memory_service import episode_passage_text

    plan = MigrationPlan(model=model, dimensions=dimensions)
    chats: set[str] = set()
    for chat_key, ref, data in documents:
        chats.add(chat_key)
        action = classify_episode(data, model=model, dimensions=dimensions)
        if action == KEEP:
            plan.keep += 1
            continue
        reason = _stale_reason(data, model=model)
        plan.stale_reasons[reason] = plan.stale_reasons.get(reason, 0) + 1
        if action == REEMBED:
            plan.reembed.append(ReembedItem(ref=ref, text=episode_passage_text(data)))
        else:
            plan.delete.append(ref)
    plan.chats = len(chats)
    return plan


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Throttle:
    """Space calls at least ``60 / max_rpm`` seconds apart."""

    def __init__(
        self,
        max_rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if max_rpm <= 0:
            raise ValueError("max_rpm must be positive")
        self._interval = 60.0 / max_rpm
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    async def wait(self) -> None:
        if self._last is not None:
            remaining = self._interval - (self._clock() - self._last)
            if remaining > 0:
                await self._sleep(remaining)
        self._last = self._clock()


async def apply_plan(
    plan: MigrationPlan,
    embedder: Any,
    *,
    vector_factory: Callable[[list[float]], Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
    throttle: Throttle | None = None,
) -> ApplyResult:
    """Delete text-less episodes, then re-embed the rest batch by batch.

    A batch whose embedding request fails is counted as failed and left
    untouched, so a re-run picks it up again.
    """
    result = ApplyResult()
    throttle = throttle or Throttle(DEFAULT_MAX_RPM)

    for ref in plan.delete:
        try:
            ref.delete()
            result.deleted += 1
        except Exception:
            result.failed += 1

    for batch in chunked(plan.reembed, batch_size):
        await throttle.wait()
        result.requests += 1
        vectors = await embedder.embed_batch(
            [item.text for item in batch], input_type="passage"
        )
        if len(vectors) != len(batch):
            result.failed += len(batch)
            continue
        for item, vector in zip(batch, vectors, strict=True):
            try:
                item.ref.update(
                    {
                        "embedding": vector_factory(vector),
                        "embedding_model": embedder.model,
                        "embedding_dim": len(vector),
                    }
                )
                result.reembedded += 1
            except Exception:
                result.failed += 1
    return result


def format_plan(plan: MigrationPlan, *, collection: str) -> str:
    lines = [
        f"Collection: {collection}/*/items",
        f"Target embedding: {plan.model} (dim={plan.dimensions})",
        f"Chats: {plan.chats}  Episodes scanned: {plan.scanned}",
        f"  keep (already current): {plan.keep}",
        f"  re-embed:               {len(plan.reembed)}",
        f"  delete (no summary):    {len(plan.delete)}",
    ]
    for reason, count in sorted(plan.stale_reasons.items()):
        lines.append(f"  stale because {reason}: {count}")
    return "\n".join(lines)


def format_result(result: ApplyResult) -> str:
    return (
        f"Applied: re-embedded {result.reembedded}, deleted {result.deleted}, "
        f"failed {result.failed} ({result.requests} embedding request(s))"
    )


async def run(
    *,
    client: Any,
    collection: str,
    embedder: Any,
    apply: bool,
    vector_factory: Callable[[list[float]], Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
    throttle: Throttle | None = None,
    out: Callable[[str], None] = print,
) -> int:
    if embedder.model in retired_model_ids():
        # Re-embedding "to" a retired model would fail every batch with
        # HTTP 410 while the deletions still ran.
        out(
            f"The embed role resolves to {embedder.model}, which is retired. "
            "Fix the embed tier in MODEL_ROLES first. Nothing was read or written."
        )
        return 2
    plan = plan_migration(
        iter_episode_documents(client, collection),
        model=embedder.model,
        dimensions=embedder.dimensions,
    )
    out(format_plan(plan, collection=collection))
    if not apply:
        out("Dry run: nothing was written. Re-run with --apply to migrate.")
        return 0
    if not plan.reembed and not plan.delete:
        out("Nothing to migrate.")
        return 0
    if plan.reembed and not getattr(embedder, "is_configured", False):
        out("NVIDIA_API_KEY is not set; cannot re-embed. Nothing was written.")
        return 2
    result = await apply_plan(
        plan,
        embedder,
        vector_factory=vector_factory,
        batch_size=batch_size,
        throttle=throttle,
    )
    out(format_result(result))
    return 1 if result.failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-embed Firestore episodic-memory vectors with the embed role's "
            "model (dry run unless --apply)."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=_REPO_ROOT / ".env",
        help="Dotenv file with the Firestore and NVIDIA settings (default: repo .env).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Texts per embedding request (1-{DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=DEFAULT_MAX_RPM,
        help=f"Embedding requests per minute (1-{DEFAULT_MAX_RPM}).",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:  # pragma: no cover - live path
    _import_src()
    from google.cloud import firestore  # type: ignore
    from google.cloud.firestore_v1.vector import Vector  # type: ignore

    from src.config import Settings
    from src.runtime.model_roles import RoleConfigError
    from src.services.embedding_service import build_embedding_service
    from src.utils.logger import setup_logger

    # The app logger prints HTTP error bodies at WARNING, and a 4xx body can
    # quote the input text; this script promises counts only.
    setup_logger(level="ERROR")
    settings = Settings(_env_file=str(args.env_file))
    client_kwargs: dict[str, Any] = {}
    if settings.firestore_project_id:
        client_kwargs["project"] = settings.firestore_project_id
    database = (settings.firestore_database or "(default)").strip() or "(default)"
    if database != "(default)":
        client_kwargs["database"] = database
    prefix = (settings.firestore_collection_prefix or "linebot").strip().rstrip("_") or "linebot"

    # The app's own embed role, so vectors are tagged with the model the app
    # recalls with; a role table the app would refuse stops here.
    try:
        embedder = build_embedding_service(settings)
    except RoleConfigError as exc:
        print(f"{exc}. Nothing was read or written.")
        return 2
    client = firestore.Client(**client_kwargs)
    try:
        return await run(
            client=client,
            collection=f"{prefix}_episodes",
            embedder=embedder,
            apply=args.apply,
            vector_factory=lambda values: Vector([float(v) for v in values]),
            batch_size=args.batch_size,
            throttle=Throttle(args.max_rpm),
        )
    finally:
        await embedder.close()
        client.close()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - live path
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= DEFAULT_BATCH_SIZE:
        parser.error(f"--batch-size must be 1-{DEFAULT_BATCH_SIZE}")
    if not 1 <= args.max_rpm <= DEFAULT_MAX_RPM:
        parser.error(f"--max-rpm must be 1-{DEFAULT_MAX_RPM}")
    if not args.env_file.is_file():
        parser.error(f"env file not found: {args.env_file.name}")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
