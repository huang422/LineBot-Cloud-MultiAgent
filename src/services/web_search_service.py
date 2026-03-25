"""Web Search Service using Tavily AI Search API.

Provides web search functionality to augment LLM responses with
real-time search results from the internet.

Tavily is optimized for LLMs and RAG applications.
API Documentation: https://docs.tavily.com/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from src.config import get_settings
from src.utils.logger import logger


class WebSearchError(Exception):
    """Base exception for web search errors."""


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    content: str
    score: float = 0.0

    def to_text(self, index: int) -> str:
        content = self.content[:2000] if len(self.content) > 2000 else self.content
        return f"[{index}] {self.title}\nURL: {self.url}\n{content}"


@dataclass
class WebSearchResponse:
    """Response from web search."""
    query: str
    results: list[SearchResult]
    answer: Optional[str] = None

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    def to_context_text(self) -> str:
        if not self.results:
            return ""
        parts = []
        if self.answer:
            parts.append(f"AI Summary: {self.answer}")
        for i, result in enumerate(self.results, 1):
            parts.append(result.to_text(i))
        return "\n\n".join(parts)


@dataclass
class ExtractResult:
    """A single extracted webpage result."""

    url: str
    content: str

    def to_text(self) -> str:
        content = self.content[:10000] if len(self.content) > 10000 else self.content
        return f"URL: {self.url}\n{content}"


@dataclass
class ExtractResponse:
    """Response from webpage content extraction."""

    results: list[ExtractResult]
    failed_urls: list[str]

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    def to_context_text(self) -> str:
        if not self.results:
            return ""
        return "\n\n---\n\n".join(result.to_text() for result in self.results)


class WebSearchService:
    """Service for web search using Tavily AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 3,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key
        self.max_results = max(1, min(10, max_results))
        self._client = None
        self._configured_monthly_quota = settings.web_search_monthly_quota

        if self.api_key:
            logger.info(
                "WebSearchService initialized "
                f"(app quota disabled, configured_quota={self._configured_monthly_quota})"
            )
        else:
            logger.warning("WebSearchService: TAVILY_API_KEY not configured")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def quota_remaining(self) -> int | None:
        return None

    @property
    def is_quota_available(self) -> bool:
        return self.is_configured

    def get_quota_stats(self) -> dict:
        return {
            "configured": self.is_configured,
            "used": None,
            "quota": None,
            "remaining": None,
            "scope": "provider-side",
            "enforced": False,
            "configured_app_quota": self._configured_monthly_quota,
        }

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise WebSearchError("tavily-python package not installed")
        return self._client

    async def close(self) -> None:
        self._client = None

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_answer: bool | str = True,
        search_depth: str = "advanced",
        topic: Optional[str] = None,
        time_range: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> WebSearchResponse:
        """Execute a Tavily search.

        Args:
            query: Search query string.
            max_results: 1-10, defaults to ``self.max_results``.
            include_answer: ``True`` for basic AI answer, ``"advanced"``
                for a more detailed LLM-generated answer (costs 2 credits).
            search_depth: ``"basic"`` (fast, 1 credit), ``"advanced"``
                (deeper, 2 credits).
            topic: ``"general"`` (default), ``"news"`` or ``"finance"``.
            time_range: ``"day"``, ``"week"``, ``"month"`` or ``"year"``.
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        if not self.is_configured:
            raise WebSearchError("Tavily API key not configured")

        num_results = max(1, min(10, max_results or self.max_results))

        kwargs: dict = {
            "query": query.strip(),
            "max_results": num_results,
            "include_answer": include_answer,
            "search_depth": search_depth,
        }
        if topic:
            kwargs["topic"] = topic
        if time_range:
            kwargs["time_range"] = time_range
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains

        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: client.search(**kwargs)
            )

            results = [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                )
                for item in response.get("results", [])
            ]

            depth_tag = search_depth
            logger.info(
                f"Search complete: '{query[:50]}…' depth={depth_tag}, "
                f"{len(results)} results"
            )

            return WebSearchResponse(
                query=query,
                results=results,
                answer=response.get("answer") if include_answer else None,
            )

        except WebSearchError:
            raise
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "limit" in error_str.lower():
                raise WebSearchError(f"搜尋 API 額度已用完: {error_str}")
            logger.error(f"Tavily search error: {e}", exc_info=True)
            raise WebSearchError(f"Search failed: {error_str}")

    async def extract(self, urls: list[str]) -> ExtractResponse:
        """Extract raw webpage content from URLs using Tavily."""
        if not urls:
            raise ValueError("URL list cannot be empty")
        if not self.is_configured:
            raise WebSearchError("Tavily API key not configured")

        urls = urls[:5]

        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.extract(urls=urls),
            )

            results = [
                ExtractResult(
                    url=item.get("url", ""),
                    content=item.get("raw_content", ""),
                )
                for item in response.get("results", [])
                if item.get("raw_content")
            ]
            failed_urls = [
                item.get("url", "")
                for item in response.get("failed_results", [])
                if item.get("url")
            ]

            logger.info(
                f"Extract complete: success={len(results)}, failed={len(failed_urls)}"
            )

            return ExtractResponse(results=results, failed_urls=failed_urls)

        except WebSearchError:
            raise
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "432" in error_str or "limit" in error_str.lower():
                raise WebSearchError(f"搜尋 API 額度已用完: {error_str}")
            logger.error(f"Tavily extract error: {e}", exc_info=True)
            raise WebSearchError(f"Extract failed: {error_str}")


# Global singleton
_web_search_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service


async def close_web_search_service() -> None:
    global _web_search_service
    if _web_search_service:
        await _web_search_service.close()
        _web_search_service = None
