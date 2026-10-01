"""Optional reordering of search results by meaning (SEARCH_RERANK_ENABLED).

A search backend's own order is its relevance, not necessarily how well a
result answers the question. With the flag on, the embed role
(nemotron-3-embed-1b) embeds the query once (``input_type=query``) and the
results' titles and snippets in one batch (``passage``), and the results
are sorted by cosine similarity before the top ones are read. Both calls
share one RERANK_BUDGET_SECONDS budget; a timeout, a failure or embeddings
being off leave the order as it was. Off by default: two more embed calls
per search against NVIDIA's shared budget, for a gain not yet measured on
zh-TW queries.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from src.services.search.types import SearchResult
from src.utils.logger import logger

RERANK_BUDGET_SECONDS = 3.0
_PASSAGE_CHARS = 500


def _passage(result: SearchResult) -> str:
    text = f"{result.title}\n{result.content}".strip()
    return text[:_PASSAGE_CHARS] or result.url


class SearchReranker:
    """See the module docstring."""

    def __init__(
        self,
        get_embedder: Callable[[], Any],
        *,
        budget_seconds: float = RERANK_BUDGET_SECONDS,
    ) -> None:
        """``get_embedder`` returns the embedding service, or None while
        embeddings are off (it is configured after the container is built)."""
        self._get_embedder = get_embedder
        self.budget_seconds = budget_seconds

    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if len(results) < 2:
            return results
        embedder = self._get_embedder()
        if embedder is None or not embedder.is_configured:
            return results
        try:
            query_vector, passage_vectors = await asyncio.wait_for(
                asyncio.gather(
                    embedder.embed_text(query),
                    embedder.embed_batch([_passage(r) for r in results], input_type="passage"),
                ),
                self.budget_seconds,
            )
        except Exception as exc:
            logger.info(f"Search rerank skipped ({type(exc).__name__})")
            return results
        if not query_vector or len(passage_vectors) != len(results):
            logger.info("Search rerank skipped (no embeddings)")
            return results
        scores = [sum(q * p for q, p in zip(query_vector, vector, strict=False)) for vector in passage_vectors]
        order = sorted(range(len(results)), key=lambda index: scores[index], reverse=True)
        return [results[index] for index in order]
