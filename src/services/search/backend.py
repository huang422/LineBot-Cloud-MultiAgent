"""Search backends and the chain that tries them in order (ADR-0009).

A backend either answers or raises: :class:`~src.services.search.types.
SearchBackendError` when it cannot (unreachable, refused, rate limited) and
:class:`~src.services.search.types.NoResultsError` when it found nothing.
:class:`ChainedSearchBackend` asks each configured backend in turn (today
Tavily alone; composition decides the order) and moves on after any
exception or an empty answer. When none answers it returns an empty
response marked ``degraded``: the caller answers without search (and says
so) instead of failing. ``all_failed`` on it tells a search nobody could
run (every backend failed) from one that ran and found nothing.

The chain is the seam for another backend: one that satisfies
:class:`SearchBackend` is added in ``build_search_backend`` and needs no
change here or in its callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.services.search.types import (
    ExtractResponse,
    NoResultsError,
    SearchBackendError,
    WebSearchResponse,
)
from src.utils.logger import logger


@runtime_checkable
class SearchBackend(Protocol):
    """One place to search the web, and optionally to read pages."""

    name: str
    # Whether extract() can read pages (a search-only backend cannot).
    supports_extract: bool

    @property
    def is_configured(self) -> bool: ...

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        topic: str | None = None,
        time_range: str | None = None,
        country: str | None = None,
    ) -> WebSearchResponse:
        """Results for *query*. A single backend raises SearchBackendError
        (NoResultsError when it found nothing); the chain never raises and
        marks an empty answer ``degraded`` (and ``all_failed`` when every
        backend failed, rather than one finding nothing).

        ``topic`` is "news", "finance" or None (general); ``time_range`` is
        "day", "week", "month", "year" or None; ``country`` (e.g. "taiwan")
        is a hint a backend may ignore.
        """
        ...

    async def extract(self, urls: list[str]) -> ExtractResponse: ...

    def get_status(self) -> dict: ...

    async def close(self) -> None: ...


class ChainedSearchBackend:
    """Tries its backends in order; see the module docstring."""

    name = "chain"

    def __init__(self, backends: Sequence[SearchBackend]) -> None:
        self.backends: list[SearchBackend] = list(backends)
        self._searches = 0
        self._degraded = 0
        self._answered_by: dict[str, int] = {}

    @property
    def is_configured(self) -> bool:
        return any(backend.is_configured for backend in self.backends)

    @property
    def supports_extract(self) -> bool:
        return any(b.is_configured and b.supports_extract for b in self.backends)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        topic: str | None = None,
        time_range: str | None = None,
        country: str | None = None,
    ) -> WebSearchResponse:
        """The first backend's non-empty answer, else an empty degraded one."""
        self._searches += 1
        found_nothing = False
        for backend in self.backends:
            if not backend.is_configured:
                continue
            try:
                response = await backend.search(
                    query,
                    max_results=max_results,
                    topic=topic,
                    time_range=time_range,
                    country=country,
                )
            except NoResultsError:
                logger.info(f"Search: {backend.name} found nothing; trying the next backend")
                found_nothing = True
                continue
            except Exception as exc:
                logger.warning(
                    f"Search: {backend.name} failed ({_describe(exc)}); trying the next backend"
                )
                continue
            if response.is_empty:
                logger.info(f"Search: {backend.name} found nothing; trying the next backend")
                found_nothing = True
                continue
            response.backend = response.backend or backend.name
            self._answered_by[response.backend] = self._answered_by.get(response.backend, 0) + 1
            return response
        self._degraded += 1
        logger.warning("Search: no backend could answer; the reply goes without search")
        return WebSearchResponse(query=query, results=[], degraded=True, all_failed=not found_nothing)

    async def extract(self, urls: list[str]) -> ExtractResponse:
        """Pages from the first backend that reads any of *urls*.

        Every URL no backend could read is in ``failed_urls``.
        """
        for backend in self.backends:
            if not (backend.is_configured and backend.supports_extract):
                continue
            try:
                response = await backend.extract(list(urls))
            except Exception as exc:
                logger.warning(f"Page extract: {backend.name} failed ({_describe(exc)})")
                continue
            if response.has_results:
                return response
        return ExtractResponse(results=[], failed_urls=list(urls))

    def get_status(self) -> dict:
        """For /health/detail: each backend's own status plus the counts."""
        return {
            "configured": self.is_configured,
            "backends": [backend.get_status() for backend in self.backends],
            "searches": self._searches,
            "degraded": self._degraded,
            "answered_by": dict(self._answered_by),
        }

    async def close(self) -> None:
        for backend in self.backends:
            try:
                await backend.close()
            except Exception as exc:  # pragma: no cover - shutdown best effort
                logger.debug(f"Search: closing {backend.name} failed: {exc}")


def _describe(exc: Exception) -> str:
    """A short reason for the log. Only our own errors carry a message: an
    httpx error's text can hold a URL, and a URL can carry a credential."""
    message = str(exc) if isinstance(exc, SearchBackendError) else ""
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
