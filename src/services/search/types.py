"""What a web search or a page read returns, whichever backend produced it.

The shapes started as Tavily's; the agent and the chat tool read every
backend through them. The fields added since all have defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How much of one result's text reaches the model.
RESULT_TEXT_CHARS = 3000
# How much of one page read from a URL the user sent reaches the model.
PAGE_TEXT_CHARS = 10000
# Labels the model sees in search context; prompts/web_search.md explains them.
AI_SUMMARY_LABEL = "AI Summary"  # an LLM-written summary (Tavily)
PUBLISHED_LABEL = "Published"


class SearchBackendError(Exception):
    """A backend could not answer: unreachable, refused, rate limited, or
    it found nothing (:class:`NoResultsError`). The chain moves on."""


class NoResultsError(SearchBackendError):
    """The backend answered, but with no result and no answer."""


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    content: str
    # The page's own text, when it was read (it replaces the snippet).
    raw_content: str | None = None
    score: float = 0.0
    # When the page was published (ISO date), if the backend knows.
    published: str | None = None

    def to_text(self, index: int) -> str:
        text = self.raw_content or self.content
        if len(text) > RESULT_TEXT_CHARS:
            # End at a sentence or a line break in the last fifth, else at
            # the limit: a timetable or a price list may have its only 。 in
            # its first line, and page text must not shrink below a snippet.
            head = text[:RESULT_TEXT_CHARS]
            floor = RESULT_TEXT_CHARS * 4 // 5
            cut = max(head.rfind("。", floor), head.rfind("\n", floor))
            text = (head[: cut + 1] if cut >= 0 else head).rstrip()
        lines = [f"[{index}] {self.title}", f"URL: {self.url}"]
        if self.published:
            lines.append(f"{PUBLISHED_LABEL}: {self.published}")
        lines.append(text)
        return "\n".join(lines)


@dataclass
class WebSearchResponse:
    """Response from web search."""

    query: str
    results: list[SearchResult]
    answer: str | None = None
    # How the answer is labelled for the model: Tavily's is an LLM summary.
    answer_label: str = "Answer"
    # The backend that answered (e.g. "tavily"); empty when none did.
    backend: str = ""
    # True when no backend could answer: every one failed or found nothing.
    degraded: bool = False
    # True when every backend failed (unreachable, refused, rate limited),
    # rather than one of them finding nothing: a retry would only fail again.
    all_failed: bool = False

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    @property
    def is_empty(self) -> bool:
        return not self.results and not self.answer

    def to_context_text(self) -> str:
        parts = []
        if self.answer:
            parts.append(f"{self.answer_label}: {self.answer}")
        for i, result in enumerate(self.results, 1):
            parts.append(result.to_text(i))
        return "\n\n".join(parts)


@dataclass
class ExtractResult:
    """A single extracted webpage result."""

    url: str
    content: str

    def to_text(self) -> str:
        content = self.content[:PAGE_TEXT_CHARS]
        return f"URL: {self.url}\n{content}"


@dataclass
class ExtractResponse:
    """Response from webpage content extraction."""

    results: list[ExtractResult]
    failed_urls: list[str] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    def to_context_text(self) -> str:
        if not self.results:
            return ""
        return "\n\n---\n\n".join(result.to_text() for result in self.results)
