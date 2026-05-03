"""NVIDIA NIM embedding client for episodic memory recall.

Uses NVIDIA's free inference API ``nvidia/nv-embedqa-e5-v5`` (1024 dims,
multilingual including Traditional Chinese). All calls are best-effort:
on any failure we log and return ``None`` so callers can degrade
gracefully (memory still works, just without vector recall).

Singleton helpers ``configure_embedding_service`` / ``get_embedding_service``
mirror the pattern used by the other long-lived services in this app.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

import httpx

from src.config import Settings
from src.utils.logger import logger


_DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
_DEFAULT_MODEL = "nvidia/nv-embedqa-e5-v5"


class EmbeddingServiceError(Exception):
    """Raised when the embedding service cannot fulfil a request."""


class EmbeddingService:
    """Async embedding client with shared httpx pool and basic retry."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = _DEFAULT_ENDPOINT,
        model: str = _DEFAULT_MODEL,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._model = model
        self._configured = bool(api_key)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        if self._configured:
            logger.info(
                f"EmbeddingService ready (model={model}, endpoint={endpoint})"
            )
        else:
            logger.warning(
                "EmbeddingService unconfigured (no NVIDIA API key); "
                "vector recall will be disabled"
            )

    @property
    def is_configured(self) -> bool:
        return self._configured

    @property
    def model(self) -> str:
        return self._model

    async def close(self) -> None:
        await self._client.aclose()

    async def embed_text(self, text: str) -> list[float] | None:
        """Embed a single text snippet. Returns None on failure."""
        text = (text or "").strip()
        if not text or not self._configured:
            return None
        vectors = await self._embed_batch_internal([text], input_type="query")
        if not vectors:
            return None
        return vectors[0]

    async def embed_passage(self, text: str) -> list[float] | None:
        """Embed text using ``passage`` mode (used for stored episodes)."""
        text = (text or "").strip()
        if not text or not self._configured:
            return None
        vectors = await self._embed_batch_internal([text], input_type="passage")
        if not vectors:
            return None
        return vectors[0]

    async def embed_batch(
        self,
        texts: Iterable[str],
        *,
        input_type: str = "passage",
    ) -> list[list[float]]:
        """Embed multiple texts. Skips empty entries; preserves order."""
        prepared = [(t or "").strip() for t in texts]
        if not any(prepared) or not self._configured:
            return []
        return await self._embed_batch_internal(prepared, input_type=input_type)

    async def _embed_batch_internal(
        self,
        texts: list[str],
        *,
        input_type: str,
    ) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        try:
            resp = await self._client.post(self._endpoint, json=payload)
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            logger.warning(f"Embedding request failed (transport): {exc}")
            return []

        if resp.status_code >= 400:
            body = resp.text[:240]
            logger.warning(
                f"Embedding request returned HTTP {resp.status_code}: {body}"
            )
            return []

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning(f"Embedding response not JSON: {exc}")
            return []

        items = data.get("data") or []
        vectors: list[list[float]] = []
        for item in items:
            emb = item.get("embedding")
            if isinstance(emb, list) and emb:
                vectors.append([float(v) for v in emb])
        return vectors


_embedding_service: EmbeddingService | None = None


def configure_embedding_service(settings: Settings) -> EmbeddingService:
    """Initialise (or replace) the shared :class:`EmbeddingService`.

    If a previous instance exists we schedule its ``close()`` and keep
    a reference to the task so the httpx client really gets shut down
    instead of being silently dropped (which would leak the connection
    pool over time).
    """
    global _embedding_service, _pending_close_tasks
    if _embedding_service is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            task = loop.create_task(_embedding_service.close())
            _pending_close_tasks.add(task)
            task.add_done_callback(_pending_close_tasks.discard)
        else:
            # No running loop (e.g. unit-test sync path): close
            # synchronously by running a fresh loop just for the
            # teardown so we don't leak the httpx pool.
            try:
                asyncio.run(_embedding_service.close())
            except RuntimeError:
                pass
    endpoint = (
        getattr(settings, "nvidia_embedding_endpoint", "") or _DEFAULT_ENDPOINT
    )
    model = getattr(settings, "nvidia_embedding_model", "") or _DEFAULT_MODEL
    _embedding_service = EmbeddingService(
        api_key=settings.nvidia_api_key,
        endpoint=endpoint,
        model=model,
    )
    return _embedding_service


_pending_close_tasks: set[asyncio.Task] = set()


def get_embedding_service() -> EmbeddingService | None:
    return _embedding_service


async def close_embedding_service() -> None:
    global _embedding_service
    if _embedding_service is not None:
        await _embedding_service.close()
        _embedding_service = None
    if _pending_close_tasks:
        await asyncio.gather(*_pending_close_tasks, return_exceptions=True)
        _pending_close_tasks.clear()
