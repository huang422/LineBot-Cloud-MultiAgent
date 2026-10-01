"""Tavily: the search backend and the fallback page reader (ADR-0009).

Plain ``httpx`` against the REST API (``POST /search`` and ``/extract``); the
SDK is gone, and tiktoken with it. With ``TAVILY_API_KEY`` requests carry
``Authorization: Bearer``; without one, and with ``TAVILY_KEYLESS_ENABLED``
(an opt-in: a v2 .env left the key empty to turn search off), they carry
``X-Tavily-Access-Mode: keyless`` (free, rate-limited, same answers).
Neither set: the backend is off.

Costs stay at the floor: ``search_depth`` basic (1 credit; advanced is 2)
with ``auto_parameters`` off (it can switch to advanced on its own), and
extract at basic depth (1 credit per 5 pages read). ``include_usage`` makes
Tavily report the credits each call used; ``/health/detail`` shows their
sum for this process. ``country`` is sent only with the general topic, the
only one Tavily accepts it for.

A 429 skips Tavily for its ``Retry-After`` (else a minute, at most an
hour); a plan limit (432/433) or a refused key (401/403) for an hour.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from email.utils import parsedate_to_datetime

import httpx

from src.services.search.types import (
    AI_SUMMARY_LABEL,
    ExtractResponse,
    ExtractResult,
    NoResultsError,
    SearchBackendError,
    SearchResult,
    WebSearchResponse,
)
from src.utils.logger import logger

API_URL = "https://api.tavily.com"
TIMEOUT_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 3.0
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
MAX_COOLDOWN_SECONDS = 3600.0
MAX_EXTRACT_URLS = 5
_MAX_RESULTS = 20


def _published(value) -> str | None:
    """Tavily's ``published_date`` (RFC 2822 or ISO) as YYYY-MM-DD."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
    return parsed.date().isoformat()


def _score(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


class TavilyBackend:
    """Tavily's REST API; see the module docstring."""

    name = "tavily"
    supports_extract = True

    def __init__(
        self,
        api_key: str = "",
        *,
        keyless_enabled: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self.mode = "key" if self._api_key else ("keyless" if keyless_enabled else "off")
        self._clock = clock
        self._wall_clock = wall_clock
        self._cooldown_until = 0.0
        self._last_error: str | None = None
        self._credits = 0.0
        self._client: httpx.AsyncClient | None = None
        if self.mode != "off":
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self.mode == "key":
                headers["Authorization"] = f"Bearer {self._api_key}"
            else:
                headers["X-Tavily-Access-Mode"] = "keyless"
            self._client = httpx.AsyncClient(
                base_url=API_URL,
                timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
                headers=headers,
                transport=transport,
            )

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def search_body(
        self,
        query: str,
        *,
        max_results: int,
        topic: str | None,
        time_range: str | None,
        country: str | None,
    ) -> dict:
        """The JSON body of one search."""
        body: dict = {
            "query": query,
            "topic": topic or "general",
            "search_depth": "basic",
            "max_results": max(1, min(_MAX_RESULTS, max_results)),
            "include_answer": True,
            "auto_parameters": False,
            "include_usage": True,
        }
        if time_range:
            body["time_range"] = time_range
        if country and body["topic"] == "general":
            body["country"] = country
        return body

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        topic: str | None = None,
        time_range: str | None = None,
        country: str | None = None,
    ) -> WebSearchResponse:
        data = await self._post(
            "/search",
            self.search_body(
                query,
                max_results=max_results,
                topic=topic,
                time_range=time_range,
                country=country,
            ),
        )
        results = [
            SearchResult(
                title=str(item.get("title") or item["url"]).strip(),
                url=item["url"],
                content=str(item.get("content") or "").strip(),
                score=_score(item.get("score")),
                published=_published(item.get("published_date")),
            )
            for item in data.get("results") or []
            if isinstance(item, dict) and isinstance(item.get("url"), str)
        ]
        answer = data.get("answer") if isinstance(data.get("answer"), str) else None
        response = WebSearchResponse(
            query=query,
            results=results,
            answer=(answer or "").strip() or None,
            answer_label=AI_SUMMARY_LABEL,
            backend=self.name,
        )
        if response.is_empty:
            raise NoResultsError("no results")
        return response

    async def extract(self, urls: list[str]) -> ExtractResponse:
        """Basic-depth extract of up to MAX_EXTRACT_URLS pages, as text."""
        wanted = list(urls)[:MAX_EXTRACT_URLS]
        data = await self._post(
            "/extract",
            {"urls": wanted, "extract_depth": "basic", "format": "text", "include_usage": True},
        )
        results = [
            ExtractResult(url=item["url"], content=str(item["raw_content"]))
            for item in data.get("results") or []
            if isinstance(item, dict)
            and isinstance(item.get("url"), str)
            and str(item.get("raw_content") or "").strip()
        ]
        read = {result.url for result in results}
        failed = [url for url in urls if url not in read]
        logger.info(f"Tavily extract: {len(results)} read, {len(failed)} failed")
        return ExtractResponse(results=results, failed_urls=failed)

    async def _post(self, path: str, body: dict) -> dict:
        if self._client is None:
            raise SearchBackendError("no TAVILY_API_KEY and keyless mode is off")
        wait = self._cooldown_until - self._clock()
        if wait > 0:
            raise SearchBackendError(f"skipped for {wait:.0f}s more ({self._last_error})")
        try:
            response = await self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise self._failed(type(exc).__name__) from exc
        status = response.status_code
        if status == 429:
            self._cool_down(self._retry_after(response) or RATE_LIMIT_COOLDOWN_SECONDS)
            raise self._failed(f"HTTP 429 ({self.mode}): rate limited")
        if status in (432, 433):
            self._cool_down(MAX_COOLDOWN_SECONDS)
            raise self._failed(f"HTTP {status}: plan or pay-as-you-go limit reached")
        if status in (401, 403):
            self._cool_down(MAX_COOLDOWN_SECONDS)
            raise self._failed(f"HTTP {status} ({self.mode}): refused")
        if status != 200:
            raise self._failed(f"HTTP {status}")
        try:
            data = response.json()
        except ValueError as exc:
            raise self._failed("the answer is not JSON") from exc
        if not isinstance(data, dict):
            raise self._failed("the answer is not a JSON object")
        usage = data.get("usage")
        if isinstance(usage, dict):
            try:
                self._credits += float(usage.get("credits") or 0)
            except (TypeError, ValueError):
                pass
        self._last_error = None
        return data

    def _retry_after(self, response: httpx.Response) -> float | None:
        raw = (response.headers.get("retry-after") or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            pass
        try:
            return parsedate_to_datetime(raw).timestamp() - self._wall_clock()
        except (TypeError, ValueError, IndexError):
            return None

    def _cool_down(self, seconds: float) -> None:
        seconds = max(1.0, min(MAX_COOLDOWN_SECONDS, seconds))
        self._cooldown_until = self._clock() + seconds
        logger.warning(f"Tavily skipped for {seconds:.0f}s")

    def _failed(self, reason: str) -> SearchBackendError:
        self._last_error = reason
        return SearchBackendError(reason)

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "configured": self.is_configured,
            "mode": self.mode,
            "cooldown_seconds": max(0, round(self._cooldown_until - self._clock())),
            "last_error": self._last_error,
            # Credits Tavily reported for this process's calls.
            "credits_used": self._credits,
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
