"""Turning a fetched page into the text the model reads.

:class:`TextExtractor` decodes the body (the Content-Type charset, then a
``<meta>`` charset, then UTF-8: Big5 pages are common in Taiwan and often
only say so in ``<meta>``) and extracts the main text with trafilatura
2.2.0. When none of those charsets decodes every byte (a stray byte, a
character cut in two at the byte cap), the one that loses fewest
characters is kept; only when even that one loses many is the charset
guessed, and a page still mostly lost is a failed read
(:class:`UndecodablePageError`). trafilatura is imported lazily, inside
the extractor's own two-thread pool: its import alone takes about a second
and some 50 MB, which a cold start must not pay, and it must never block
the event loop.
When it is missing, fails or finds nothing, a small readability-style
extractor (:func:`simple_extract`) takes over.

:class:`PageReader` fetches (:mod:`~src.services.search.page_fetcher`) and
extracts several pages at once within one time budget, for the URLs a user
sends and for the top search results. The budget never runs past the
caller's deadline, so what was read reaches the caller before its own
timeout does; and no fetch or extraction outlives the call, even one that
is cancelled.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any

from src.services.search.page_fetcher import PageFetcher, describe_failure
from src.services.search.types import ExtractResponse, ExtractResult, WebSearchResponse
from src.utils.logger import logger

EXTRACT_WORKERS = 2
# Extracted text shorter than this is taken for a failed read (a page
# that needs JavaScript, a consent wall), so another reader gets a turn.
MIN_PAGE_CHARS = 80
MAX_TEXT_CHARS = 20000
# How long before the caller's deadline PageReader returns: time for the
# caller to use the pages before its own timeout fires.
DEADLINE_MARGIN_SECONDS = 0.5
# How much of a body a charset guess looks at.
_GUESS_BYTES = 65536
# A decoding that loses (turns into U+FFFD) more than this many characters,
# and more than this share of the non-ASCII ones, is the wrong charset.
_STRAY_CHARS = 3
_MAX_LOST_SHARE = 0.1

_META_CHARSET = re.compile(rb"""<meta[^>]+?charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)
# Labels pages use for encodings Python knows by a wider name.
_CHARSET_ALIASES = {"big5": "cp950", "x-big5": "cp950", "gb2312": "gb18030", "gbk": "gb18030"}

_SKIP_TAGS = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "nav", "header", "footer",
        "aside", "form", "iframe", "button", "select", "textarea", "head",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
        "table", "section", "article", "main", "blockquote", "pre", "dd", "dt", "dl",
        "figcaption", "hr",
    }
)
_MAIN_TAGS = frozenset({"article", "main"})
# Below this, <article>/<main> text is not trusted to be the page's body.
_MIN_MAIN_CHARS = 200


class UndecodablePageError(ValueError):
    """No charset turns the page into text: a failed read."""


def decode_html(body: bytes, charset: str | None = None) -> str:
    """*body* as text; see the module docstring for the order tried."""
    names: list[str] = []
    if charset:
        names.append(charset)
    match = _META_CHARSET.search(body[:4096])
    if match:
        names.append(match.group(1).decode("ascii", "ignore"))
    names.append("utf-8")
    known: list[str] = []
    for name in names:
        codec = _CHARSET_ALIASES.get(name.strip().lower(), name.strip())
        try:
            return body.decode(codec)
        except LookupError:
            continue
        except UnicodeDecodeError:
            known.append(codec)
    # The declared charsets go first, a character at a time: a guess reads
    # only the first _GUESS_BYTES and may name a single-byte charset (cp852)
    # that turns a whole UTF-8 page with one stray byte into garbage.
    text = min((body.decode(codec, errors="replace") for codec in known), key=_lost)
    if _mostly_lost(text):
        text = _guess_decode(body) or text
    if _mostly_lost(text):
        raise UndecodablePageError(f"{_lost(text)} characters do not decode")
    return text


def _lost(text: str) -> int:
    return text.count("\ufffd")


def _mostly_lost(text: str) -> bool:
    lost = _lost(text)
    non_ascii = len(text) - len(text.encode("ascii", "ignore"))
    return lost > _STRAY_CHARS and lost > non_ascii * _MAX_LOST_SHARE


def _guess_decode(body: bytes) -> str | None:
    try:
        from charset_normalizer import from_bytes  # a trafilatura dependency
    except ImportError:
        return None
    best = from_bytes(body[:_GUESS_BYTES]).best()
    if best is None:
        return None
    try:
        return body.decode(best.encoding, errors="replace")
    except LookupError:
        return None


class _TextCollector(HTMLParser):
    """Text of block elements, outside navigation, scripts and forms."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: list[str] = []
        # (text, inside <article>/<main>); a None flag is a line break.
        self.pieces: list[tuple[str, bool | None]] = []
        self._skip = 0
        self._main = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP_TAGS:
            self._skip += 1
        if tag in _MAIN_TAGS:
            self._main += 1
        if tag in _BLOCK_TAGS:
            self.pieces.append(("\n", None))

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in _BLOCK_TAGS:
            self.pieces.append(("\n", None))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        if tag in _MAIN_TAGS:
            self._main = max(0, self._main - 1)
        if tag in _BLOCK_TAGS:
            self.pieces.append(("\n", None))

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.pieces.append((data, self._main > 0))


def _join_lines(pieces: list[str]) -> str:
    lines: list[str] = []
    for raw in "".join(pieces).split("\n"):
        line = " ".join(raw.split())
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return "\n".join(lines)


def simple_extract(html: str) -> str:
    """Readability-style fallback: the text of ``<article>``/``<main>`` when
    they hold enough of it, else of the whole page, without scripts,
    styles, navigation, headers, footers, asides and forms; the title
    first."""
    collector = _TextCollector()
    try:
        collector.feed(html)
        collector.close()
    except Exception as exc:  # pragma: no cover - html.parser is lenient
        logger.debug(f"simple_extract: parser stopped early ({type(exc).__name__})")
    # Line breaks (flag None) belong to both texts.
    main = _join_lines([text for text, in_main in collector.pieces if in_main is not False])
    everything = _join_lines([text for text, _ in collector.pieces])
    body = main if len(main.replace("\n", "")) >= _MIN_MAIN_CHARS else everything
    title = " ".join("".join(collector.title).split())
    if title and not body.startswith(title):
        body = f"{title}\n{body}" if body else title
    return body[:MAX_TEXT_CHARS]


def _import_trafilatura():
    # Lazy on purpose: about a second and ~50 MB, paid by the first
    # extraction in its worker thread, never by a cold start (ADR-0009).
    import trafilatura

    return trafilatura


class TextExtractor:
    """Page bytes → main text, in a dedicated pool of EXTRACT_WORKERS threads.

    ``asyncio.wait_for`` cannot stop a thread, so a slow extraction keeps its
    thread until it finishes; the small pool bounds how many can pile up.
    """

    def __init__(
        self,
        *,
        max_workers: int = EXTRACT_WORKERS,
        load_trafilatura: Callable[[], Any] = _import_trafilatura,
    ) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="page-extract")
        self._load_trafilatura = load_trafilatura
        self._trafilatura_missing = False

    async def extract(
        self,
        body: bytes,
        *,
        url: str = "",
        charset: str | None = None,
        content_type: str = "text/html",
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, self.extract_sync, body, url, charset, content_type
        )

    def extract_sync(self, body: bytes, url: str, charset: str | None, content_type: str) -> str:
        """The work itself; runs in the pool."""
        text = decode_html(body, charset)
        if content_type == "text/plain":
            return _join_lines([text])[:MAX_TEXT_CHARS]
        extracted = self._with_trafilatura(text, url)
        if extracted:
            return extracted[:MAX_TEXT_CHARS]
        return simple_extract(text)

    def _with_trafilatura(self, html: str, url: str) -> str | None:
        if self._trafilatura_missing:
            return None
        try:
            trafilatura = self._load_trafilatura()
        except ImportError:
            self._trafilatura_missing = True
            logger.warning("trafilatura is not installed; pages use the simple extractor")
            return None
        try:
            # No deduplicate: its cache is process-wide, so a warm instance
            # dropped the long paragraphs of a page read a fourth time.
            text = trafilatura.extract(
                html,
                url=url or None,
                fast=True,
                favor_precision=True,
                include_comments=False,
                include_tables=False,
                include_links=False,
                deduplicate=False,
                output_format="txt",
            )
        except Exception as exc:
            logger.warning(f"trafilatura failed ({type(exc).__name__}); using the simple extractor")
            return None
        return text.strip() if isinstance(text, str) and text.strip() else None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class PageReader:
    """Fetches and extracts pages concurrently within ``budget_seconds``."""

    def __init__(
        self,
        fetcher: PageFetcher,
        extractor: TextExtractor,
        *,
        budget_seconds: float = 6.0,
    ) -> None:
        self.fetcher = fetcher
        self.extractor = extractor
        self.budget_seconds = budget_seconds

    async def read(self, urls: Sequence[str], *, deadline: float | None = None) -> ExtractResponse:
        """The text of each URL; one not read within the budget (or at all,
        or with less than MIN_PAGE_CHARS of text) is in ``failed_urls``.

        ``deadline`` (``time.monotonic()``) is the caller's: the budget ends
        DEADLINE_MARGIN_SECONDS before it when that comes first.
        """
        unique = list(dict.fromkeys(url for url in urls if url))
        if not unique:
            return ExtractResponse(results=[], failed_urls=[])
        budget = self.budget_seconds
        if deadline is not None:
            budget = min(budget, deadline - DEADLINE_MARGIN_SECONDS - time.monotonic())
        if budget <= 0:
            logger.info(f"Page read skipped: no time left for {len(unique)} page(s)")
            return ExtractResponse(results=[], failed_urls=unique)
        tasks = {url: asyncio.create_task(self._read_one(url)) for url in unique}
        try:
            done, _ = await asyncio.wait(tasks.values(), timeout=budget)
        finally:
            # asyncio.wait never cancels what it waits on, not even when
            # read() itself is cancelled: stop the rest here.
            pending = [task for task in tasks.values() if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        results: list[ExtractResult] = []
        failed: list[str] = []
        for url, task in tasks.items():
            reason = self._failure(task, done)
            if reason is None:
                results.append(ExtractResult(url=url, content=task.result()))
            else:
                failed.append(url)
                logger.info(f"Page read failed ({reason})")
        logger.info(f"Page read: {len(results)} read, {len(failed)} failed")
        return ExtractResponse(results=results, failed_urls=failed)

    @staticmethod
    def _failure(task: asyncio.Task, done: set) -> str | None:
        if task not in done or task.cancelled():
            return "out of time"
        exc = task.exception()
        if exc is not None:
            return describe_failure(exc)
        if len(task.result()) < MIN_PAGE_CHARS:
            return "too little text"
        return None

    async def _read_one(self, url: str) -> str:
        page = await self.fetcher.fetch(url)
        text = await self.extractor.extract(
            page.body, url=page.url, charset=page.charset, content_type=page.content_type
        )
        return text.strip()

    async def enrich(
        self, response: WebSearchResponse, top_n: int, *, deadline: float | None = None
    ) -> WebSearchResponse:
        """Put the page text of the first *top_n* results into their
        ``raw_content`` (where it is longer than the snippet); ``deadline``
        as for :meth:`read`."""
        top = response.results[: max(0, top_n)]
        if not top:
            return response
        pages = await self.read([result.url for result in top], deadline=deadline)
        texts = {page.url: page.content for page in pages.results}
        for result in top:
            text = texts.get(result.url)
            if text and len(text) > len(result.content):
                result.raw_content = text
        return response

    async def close(self) -> None:
        await self.fetcher.close()
        self.extractor.close()
