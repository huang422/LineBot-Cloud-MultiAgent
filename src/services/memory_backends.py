"""Storage backends for the conversation memory service.

Two backends are provided:

- :class:`InMemoryBackend` — process-local dictionary. Data is lost on
  restart, redeploy, or Cloud Run scale-to-zero. This is the historical
  behaviour and is used by the test suite (which directly inspects
  ``backend.store``).
- :class:`FirestoreBackend` — persists to Google Cloud Firestore Native.
  Survives Cloud Run cold starts and can be shared across instances.

The backend interface is intentionally narrow: load/save/list/delete the
chat-level memory document, plus optional helpers for the per-user
profile collection introduced in Phase A. Episodic vector storage will
extend the same interface in Phase B.

All Firestore access is wrapped in ``run_in_executor`` because the
official ``google-cloud-firestore`` client is sync-only at the time of
writing. The backend keeps that detail isolated from ``MemoryService``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from src.config import Settings
from src.utils.logger import logger


class MemoryBackendError(Exception):
    """Raised when a backend operation fails irrecoverably."""


class MemoryBackend(ABC):
    """Persistence interface used by ``MemoryService``."""

    name: str = "abstract"
    persistent: bool = False

    @abstractmethod
    async def load_chat(self, key: str) -> dict[str, Any] | None:
        """Return the stored chat memory document or None if missing."""

    @abstractmethod
    async def save_chat(self, key: str, document: dict[str, Any]) -> None:
        """Persist a chat memory document."""

    @abstractmethod
    async def load_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Return the stored user profile document or None if missing."""

    @abstractmethod
    async def save_user_profile(self, user_id: str, document: dict[str, Any]) -> None:
        """Persist a user profile document (per-user, cross-chat scope)."""

    async def delete_chat(self, key: str) -> None:
        """Delete a chat memory document. Default: best-effort no-op.

        Subclasses with persistent storage should override this to
        actually remove the record so ``!new`` evicts user / group
        memory rather than leaving a zombie cleared document.
        """

    async def save_episode(self, key: str, episode: dict[str, Any]) -> None:
        """Persist a single episodic-memory snapshot for a chat.

        ``episode`` should contain at minimum ``summary`` and ``ts``.
        ``embedding`` (list[float]) is optional but required for vector
        recall. Default: best-effort no-op so backends without vector
        support degrade gracefully.
        """

    async def search_episodes(
        self,
        key: str,
        query_embedding: list[float],
        *,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return up to ``k`` best-matching episodes for a chat.

        Default returns an empty list. Backends with persistent storage
        should override.
        """
        return []

    async def close(self) -> None:  # pragma: no cover - default no-op
        """Release any resources held by the backend."""


class InMemoryBackend(MemoryBackend):
    """Process-local backend kept for backward compatibility and tests.

    Tests in ``tests/test_runtime_behaviors.py`` directly poke
    ``backend.store`` (and historically ``MemoryService._memory_store``)
    to seed fixtures. The ``store`` attribute is intentionally a plain
    dict to keep that contract.
    """

    name = "memory"
    persistent = False

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}
        self.user_profiles: dict[str, dict[str, Any]] = {}
        self.episodes: dict[str, list[dict[str, Any]]] = {}

    async def load_chat(self, key: str) -> dict[str, Any] | None:
        stored = self.store.get(key)
        return deepcopy(stored) if stored is not None else None

    async def save_chat(self, key: str, document: dict[str, Any]) -> None:
        self.store[key] = deepcopy(document)

    async def load_user_profile(self, user_id: str) -> dict[str, Any] | None:
        stored = self.user_profiles.get(user_id)
        return deepcopy(stored) if stored is not None else None

    async def save_user_profile(self, user_id: str, document: dict[str, Any]) -> None:
        self.user_profiles[user_id] = deepcopy(document)

    async def delete_chat(self, key: str) -> None:
        self.store.pop(key, None)
        self.episodes.pop(key, None)

    async def save_episode(self, key: str, episode: dict[str, Any]) -> None:
        if not key:
            return
        bucket = self.episodes.setdefault(key, [])
        bucket.append(deepcopy(episode))
        # Cap the in-memory ring buffer so tests do not balloon.
        if len(bucket) > 50:
            del bucket[: len(bucket) - 50]

    async def search_episodes(
        self,
        key: str,
        query_embedding: list[float],
        *,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        bucket = self.episodes.get(key) or []
        if not bucket or not query_embedding:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for ep in bucket:
            emb = ep.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue
            score = _cosine_similarity(query_embedding, emb)
            scored.append((score, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, ep in scored[: max(1, k)]:
            item = deepcopy(ep)
            item.pop("embedding", None)
            item["score"] = float(score)
            out.append(item)
        return out


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


class FirestoreBackend(MemoryBackend):
    """Google Cloud Firestore Native backend.

    Layout (collection names use the configured prefix):

    - ``{prefix}_chat_memory/{key}`` — keyed as ``"{source_type}::{chat_id}"``
    - ``{prefix}_user_profile/{user_id}`` — per-user profile facts
    - ``{prefix}_episodes/{key}/items/{auto_id}`` — Phase B episodic memory

    The client library is synchronous, so all RPCs run inside
    ``loop.run_in_executor`` to keep the event loop responsive.
    """

    name = "firestore"
    persistent = True

    def __init__(
        self,
        *,
        project_id: str,
        database: str = "(default)",
        collection_prefix: str = "linebot",
    ) -> None:
        try:
            from google.cloud import firestore  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise MemoryBackendError(
                "google-cloud-firestore is not installed; cannot use Firestore backend"
            ) from exc

        self._firestore_module = firestore
        client_kwargs: dict[str, Any] = {}
        if project_id:
            client_kwargs["project"] = project_id
        if database and database != "(default)":
            client_kwargs["database"] = database

        try:
            self._client = firestore.Client(**client_kwargs)
        except Exception as exc:  # pragma: no cover - depends on env
            raise MemoryBackendError(
                f"Failed to initialise Firestore client: {exc}"
            ) from exc

        prefix = (collection_prefix or "linebot").strip().rstrip("_") or "linebot"
        self._chat_collection = f"{prefix}_chat_memory"
        self._user_collection = f"{prefix}_user_profile"
        self._episodes_collection = f"{prefix}_episodes"
        logger.info(
            "FirestoreBackend ready "
            f"(project={project_id or 'default'}, database={database}, "
            f"prefix={prefix})"
        )

    @property
    def episodes_collection(self) -> str:
        return self._episodes_collection

    @property
    def client(self):
        return self._client

    async def load_chat(self, key: str) -> dict[str, Any] | None:
        return await self._load(self._chat_collection, key)

    async def save_chat(self, key: str, document: dict[str, Any]) -> None:
        await self._save(self._chat_collection, key, document)

    async def load_user_profile(self, user_id: str) -> dict[str, Any] | None:
        if not user_id:
            return None
        return await self._load(self._user_collection, user_id)

    async def save_user_profile(self, user_id: str, document: dict[str, Any]) -> None:
        if not user_id:
            return
        await self._save(self._user_collection, user_id, document)

    async def delete_chat(self, key: str) -> None:
        if not key:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.collection(self._chat_collection)
                .document(key)
                .delete(),
            )
        except Exception as exc:
            raise MemoryBackendError(
                f"Firestore delete failed for {self._chat_collection}/{key}: {exc}"
            ) from exc

        # Best-effort cleanup of the chat's episode subcollection so
        # ``!new`` does not leave dangling vector docs around. Failures
        # here are logged but do not propagate.
        try:
            await loop.run_in_executor(None, self._purge_episodes_sync, key)
        except Exception as exc:
            logger.warning(
                f"Firestore episode purge failed for {key}: {exc}"
            )

    async def save_episode(self, key: str, episode: dict[str, Any]) -> None:
        if not key or not episode:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, self._save_episode_sync, key, episode
            )
        except Exception as exc:
            logger.warning(
                f"Firestore episode save failed for {key}: {exc}"
            )

    async def search_episodes(
        self,
        key: str,
        query_embedding: list[float],
        *,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        if not key or not query_embedding:
            return []
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, self._search_episodes_sync, key, query_embedding, k
            )
        except Exception as exc:
            logger.warning(
                f"Firestore episode search failed for {key}: {exc}"
            )
            return []

    def _episode_collection(self, key: str):
        return (
            self._client.collection(self._episodes_collection)
            .document(key)
            .collection("items")
        )

    def _save_episode_sync(self, key: str, episode: dict[str, Any]) -> None:
        from google.cloud.firestore_v1.vector import Vector  # type: ignore

        doc = deepcopy(episode)
        emb = doc.pop("embedding", None)
        if isinstance(emb, list) and emb:
            doc["embedding"] = Vector([float(v) for v in emb])
        self._episode_collection(key).add(doc)

    def _search_episodes_sync(
        self, key: str, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        from google.cloud.firestore_v1.base_vector_query import (  # type: ignore
            DistanceMeasure,
        )
        from google.cloud.firestore_v1.vector import Vector  # type: ignore

        query = self._episode_collection(key).find_nearest(
            vector_field="embedding",
            query_vector=Vector([float(v) for v in query_embedding]),
            distance_measure=DistanceMeasure.COSINE,
            limit=max(1, k),
        )
        results: list[dict[str, Any]] = []
        for snapshot in query.stream():
            data = snapshot.to_dict() or {}
            data.pop("embedding", None)
            results.append(data)
        return results

    def _purge_episodes_sync(self, key: str) -> None:
        coll = self._episode_collection(key)
        for snapshot in coll.list_documents(page_size=200):
            try:
                snapshot.delete()
            except Exception:
                continue

    async def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best effort
            pass

    # ── Internal helpers ─────────────────────────────────────

    async def _load(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        loop = asyncio.get_running_loop()
        try:
            snapshot = await loop.run_in_executor(
                None,
                lambda: self._client.collection(collection).document(doc_id).get(),
            )
        except Exception as exc:
            raise MemoryBackendError(
                f"Firestore load failed for {collection}/{doc_id}: {exc}"
            ) from exc

        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        return deepcopy(data)

    async def _save(self, collection: str, doc_id: str, document: dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.collection(collection).document(doc_id).set(
                    deepcopy(document), merge=False
                ),
            )
        except Exception as exc:
            raise MemoryBackendError(
                f"Firestore save failed for {collection}/{doc_id}: {exc}"
            ) from exc


def build_backend(settings: Settings) -> MemoryBackend:
    """Pick a backend based on settings, with graceful fallback.

    If ``firestore_enabled`` is true but the Firestore client cannot be
    initialised (missing credentials / library), we log a warning and
    fall back to the in-memory backend so the bot keeps running. Cloud
    Run cold starts will still wipe memory in that case, but text replies
    keep working.
    """
    if not settings.firestore_enabled:
        logger.info(
            "MemoryService backend = in-memory "
            "(set FIRESTORE_ENABLED=true to persist long-term memory)"
        )
        return InMemoryBackend()

    project_id = (settings.firestore_project_id or "").strip()
    database = (settings.firestore_database or "(default)").strip() or "(default)"
    prefix = (settings.firestore_collection_prefix or "linebot").strip() or "linebot"

    try:
        return FirestoreBackend(
            project_id=project_id,
            database=database,
            collection_prefix=prefix,
        )
    except MemoryBackendError as exc:
        logger.warning(
            f"Firestore backend unavailable, falling back to in-memory: {exc}"
        )
        return InMemoryBackend()
