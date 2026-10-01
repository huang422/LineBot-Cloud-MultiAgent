"""NVIDIA NIM embedding client for episodic memory recall.

Uses NVIDIA's free inference API ``nvidia/nemotron-3-embed-1b``
(multilingual, including Traditional Chinese). The hosted model always
returns 2048-dim vectors and rejects a reduced ``dimensions`` request, so
we never send one: the client keeps the first ``dimensions`` values
(Matryoshka prefix, 1024 by default so the existing Firestore vector index
still fits) and L2-normalizes them again, as the model card requires.

All calls are best-effort: on any failure we log and return ``None`` / an
empty list so callers can degrade gracefully (memory still works, just
without vector recall). With a :class:`~src.services.model_health.
ModelHealth` every request counts against NVIDIA's shared budget, outcomes
are recorded (a 404/410 takes the model out, as for chat models) and no
request is sent while the model is dead, cooling down or over budget.

Singleton helpers ``configure_embedding_service`` / ``get_embedding_service``
mirror the pattern used by the other long-lived services in this app.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import httpx

from src.config import Settings
from src.providers.base import TransientError
from src.providers.openai_compat import error_for_status
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.providers.fallback_chain import Tier
    from src.services.model_health import ModelHealth

_DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
# ModelHealth's key for this client's budget and model state.
_PROVIDER = "nvidia"

# Models whose endpoint answered 404/410 (retired or unknown). The ERROR is
# logged once per process so a dead embedding model does not flood the log
# on every recall; later failures for the same model log at debug level.
_eol_logged_models: set[str] = set()


class EmbeddingServiceError(Exception):
    """Raised when the embedding service cannot fulfil a request."""


def _slice_and_normalize(vector: list[float], dimensions: int) -> list[float]:
    """Keep the first ``dimensions`` values and L2-normalize them.

    A zero vector is returned as-is (it has no direction to normalize).
    """
    head = vector[:dimensions]
    norm = math.sqrt(sum(v * v for v in head))
    if norm <= 0.0:
        return head
    return [v / norm for v in head]


class EmbeddingService:
    """Async embedding client with a shared httpx pool."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        timeout: float,
        endpoint: str = _DEFAULT_ENDPOINT,
        health: ModelHealth | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """*model*, *dimensions* and *timeout* are the embed role's tier."""
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self._api_key = api_key
        self._endpoint = endpoint
        self._model = model
        self._dimensions = int(dimensions)
        self._health = health
        self._configured = bool(api_key)
        # ``transport`` lets scripts/live_gate.py carry the exact requests
        # this class builds; production always uses httpx's default.
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            transport=transport,
        )
        if self._configured:
            logger.info(
                f"EmbeddingService ready (model={model}, dimensions={self._dimensions}, "
                f"endpoint={endpoint})"
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

    @property
    def dimensions(self) -> int:
        return self._dimensions

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
        """Embed multiple texts in one request.

        Returns exactly one vector per input, in input order, or ``[]`` when
        the request fails. Blank entries are not sent (the API rejects them),
        so a batch containing one returns ``[]``: callers filter first.
        """
        prepared = [(t or "").strip() for t in texts]
        if not prepared or not self._configured:
            return []
        if not all(prepared):
            logger.warning("Embedding batch contains blank text; nothing was sent")
            return []
        return await self._embed_batch_internal(prepared, input_type=input_type)

    async def _embed_batch_internal(
        self,
        texts: list[str],
        *,
        input_type: str,
    ) -> list[list[float]]:
        # ``dimensions`` is deliberately absent: the hosted model only
        # accepts 2048 (or no value), so the prefix is cut client-side.
        payload = {
            "model": self._model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        if self._health is not None:
            blocked = self._health.unavailable(_PROVIDER, self._model)
            if blocked is not None:
                # Not a debug line: a skipped episode embedding is why the
                # episode waits for the next compaction (MemoryService).
                logger.info(f"Embedding skipped ({blocked.kind}: {blocked.reason})")
                return []
            self._health.record_request(_PROVIDER, self._model)
        try:
            resp = await self._client.post(self._endpoint, json=payload)
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.warning(f"Embedding request failed (transport): {exc}")
            self._record_failure(
                TransientError(self._model, 0, str(exc) or type(exc).__name__, _PROVIDER)
            )
            return []

        if resp.status_code != 200:
            self._record_failure(error_for_status(resp, provider=_PROVIDER, model=self._model))

        if resp.status_code in (404, 410):
            body = resp.text[:300]
            if self._model not in _eol_logged_models:
                _eol_logged_models.add(self._model)
                logger.error(
                    f"Embedding model {self._model} is unavailable "
                    f"(HTTP {resp.status_code}, retired or unknown); vector "
                    f"recall is disabled until the embed role's model is fixed: {body}"
                )
            else:
                logger.debug(f"Embedding model {self._model} still HTTP {resp.status_code}")
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

        vectors = self._parse_vectors(data, expected=len(texts))
        if vectors and self._health is not None:
            self._health.record_success(_PROVIDER, self._model)
        return vectors

    def _record_failure(self, error: Exception) -> None:
        if self._health is not None:
            self._health.record_failure(_PROVIDER, self._model, error)

    def _parse_vectors(self, data: Any, *, expected: int) -> list[list[float]]:
        """Order by ``index``, then slice and re-normalize each vector.

        All-or-nothing: a response that does not hold exactly one usable
        vector per input returns ``[]``, so batch callers can zip results
        with their inputs safely. A vector narrower than ``dimensions`` is
        not usable: it would not fit the vector index.
        """
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            logger.warning("Embedding response has no data list")
            return []

        indexed: list[tuple[int, list]] = []
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            emb = item.get("embedding")
            if isinstance(emb, list) and emb:
                indexed.append((index if isinstance(index, int) else position, emb))
        indexed.sort(key=lambda pair: pair[0])

        if len(indexed) != expected:
            logger.warning(
                f"Embedding response returned {len(indexed)} vector(s) for "
                f"{expected} input(s); discarding"
            )
            return []

        vectors: list[list[float]] = []
        for _, emb in indexed:
            try:
                values = [float(v) for v in emb]
            except (TypeError, ValueError):
                logger.warning("Embedding response contains non-numeric values; discarding")
                return []
            if len(values) < self._dimensions:
                # Unexpected for embed-1b (2048 wide). It would not fit the
                # vector index, and padding would invent a tail: discard.
                logger.warning(
                    f"Embedding from {self._model} has {len(values)} dims, fewer "
                    f"than the configured {self._dimensions}; discarding"
                )
                return []
            vectors.append(_slice_and_normalize(values, self._dimensions))
        return vectors


_embedding_service: EmbeddingService | None = None


def build_embedding_service(
    settings: Settings,
    *,
    tier: Tier | None = None,
    health: ModelHealth | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> EmbeddingService:
    """An :class:`EmbeddingService` configured the way the app runs it.

    Model, dimensions and timeout come from the embed role: *tier* when the
    caller has the bound role table, else the role's spec read from
    *settings* (the role table fills in dim and timeout_s). scripts/
    migrate_embeddings.py uses this too, so migrated vectors are tagged
    with exactly the model the app recalls with.

    Raises:
        RoleConfigError: Without *tier*, when the role table is invalid.
    """
    if tier is None:
        from src.runtime.model_roles import role_specs

        spec = role_specs(settings)["embed"][0]
        model, dimensions, timeout = spec.model, spec.dim, spec.timeout_s
    else:
        model, dimensions, timeout = tier.model, tier.options["dim"], tier.timeout_s
    return EmbeddingService(
        api_key=settings.nvidia_api_key,
        endpoint=settings.nvidia_embedding_endpoint or _DEFAULT_ENDPOINT,
        model=model,
        dimensions=dimensions,
        timeout=timeout,
        health=health,
        transport=transport,
    )


def configure_embedding_service(
    settings: Settings,
    *,
    tier: Tier | None = None,
    health: ModelHealth | None = None,
) -> EmbeddingService:
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
    _embedding_service = build_embedding_service(settings, tier=tier, health=health)
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
