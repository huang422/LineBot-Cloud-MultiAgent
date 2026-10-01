"""Page text extraction and the page reader (src/services/search/extract.py).

trafilatura is replaced by a fake module in most tests, so they run the
same whether or not it is installed; one test runs the real one when it is.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.services.search.extract import (
    DEADLINE_MARGIN_SECONDS,
    MIN_PAGE_CHARS,
    PageReader,
    TextExtractor,
    UndecodablePageError,
    decode_html,
    simple_extract,
)
from src.services.search.page_fetcher import BlockedURLError, FetchedPage, FetchError
from src.services.search.types import SearchResult, WebSearchResponse

_ARTICLE = (
    "<html><head><title>颱風動態</title><script>var x = '不該出現';</script>"
    "<style>p { color: red }</style></head><body>"
    "<header><nav>首頁 | 新聞 | 登入</nav></header>"
    "<aside>熱門文章：別的新聞</aside>"
    "<article><h1>颱風今晚接近台灣</h1>"
    "<p>中央氣象署表示，颱風預計今晚接近台灣東部海面，各地將有強風豪雨。</p>"
    "<p>民眾應注意防颱準備，避免前往山區與海邊。氣象署將持續更新最新動態與警報範圍。</p>"
    "<p>交通部也提醒，部分航班與鐵路可能因天候影響而調整班次，出門前請先查詢。</p>"
    "</article>"
    "<form><input name='q'><button>搜尋</button></form>"
    "<footer>版權所有 &copy; 2026</footer></body></html>"
)


class _FakeTrafilatura:
    def __init__(self, result="trafilatura 的正文", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict]] = []
        self.threads: list[str] = []

    def extract(self, html, **kwargs):
        self.calls.append((html, kwargs))
        self.threads.append(threading.current_thread().name)
        if self.error is not None:
            raise self.error
        return self.result


class DecodeTests(unittest.TestCase):
    def test_header_charset_meta_charset_then_utf8(self) -> None:
        text = "今天台北多雲時晴，恒春半島有雨"
        self.assertEqual(decode_html(text.encode("utf-8")), text)
        self.assertEqual(decode_html(text.encode("cp950"), "big5"), text)
        meta = f"<meta charset=\"big5\"><p>{text}</p>".encode("cp950")
        self.assertIn(text, decode_html(meta))
        http_equiv = f"<meta http-equiv='Content-Type' content='text/html; charset=Big5'><p>{text}</p>"
        self.assertIn(text, decode_html(http_equiv.encode("cp950")))

    def test_big5_label_decodes_the_cp950_extensions_pages_use(self) -> None:
        # 碁 and 銹 are outside strict Big5; Taiwanese pages labelled big5 use them.
        text = "圍碁與生銹"
        self.assertEqual(decode_html(text.encode("cp950"), "big5"), text)

    def test_an_unknown_or_wrong_label_falls_through(self) -> None:
        text = "繁體中文內容"
        self.assertEqual(decode_html(text.encode("utf-8"), "no-such-charset"), text)
        # The header says UTF-8 but the bytes are Big5: the <meta> label is next.
        mislabelled = f"<meta charset=big5><p>{text}</p>".encode("cp950")
        self.assertIn(text, decode_html(mislabelled, "utf-8"))

    def test_undeclared_non_utf8_is_guessed_or_refused(self) -> None:
        body = ("這是一段沒有宣告編碼的繁體中文網頁內容，" * 20).encode("cp950")
        if importlib.util.find_spec("charset_normalizer") is not None:
            self.assertIn("繁體中文", decode_html(body))
        with patch("src.services.search.extract._guess_decode", return_value=None):
            with self.assertRaises(UndecodablePageError):
                decode_html(body)

    def test_a_stray_byte_keeps_the_declared_charset(self) -> None:
        # Review: one bad byte failed the strict decode, and the guess, which
        # reads only the first 64 KB, named cp852: the whole page was garbage.
        text = "中央氣象署表示，颱風預計今晚接近台灣東部海面。" * 50
        page = f"<meta charset='utf-8'><title>颱風</title><p>{text}</p>".encode()
        stray = page.replace("海面".encode(), "海面".encode() + b"\x92", 1)
        cut = page[: page.rindex("海".encode()) + 1]  # the byte cap cut 海 in two
        with patch(
            "src.services.search.extract._guess_decode", side_effect=lambda body: body.decode("cp852")
        ) as guess:
            for body in (stray, cut):
                with self.subTest(body=body[-12:]):
                    decoded = decode_html(body, "utf-8")
                    self.assertIn(text[:200], decoded)
                    self.assertEqual(decoded.count("\ufffd"), 1)
        guess.assert_not_called()

    def test_a_page_still_mostly_lost_after_the_guess_is_refused(self) -> None:
        # Declared UTF-8, really Big5, and a guess that loses as much.
        body = ("<p>" + "繁體中文網頁內容" * 20 + "</p>").encode("cp950")
        with patch(
            "src.services.search.extract._guess_decode",
            side_effect=lambda body: body.decode("ascii", errors="replace"),
        ):
            with self.assertRaises(UndecodablePageError):
                decode_html(body, "utf-8")


class SimpleExtractTests(unittest.TestCase):
    def test_keeps_the_article_and_drops_the_page_furniture(self) -> None:
        text = simple_extract(_ARTICLE)

        self.assertTrue(text.startswith("颱風動態\n"))
        self.assertIn("颱風今晚接近台灣", text)
        self.assertIn("避免前往山區與海邊", text)
        for noise in ("不該出現", "color: red", "首頁 | 新聞", "熱門文章", "搜尋", "版權所有"):
            self.assertNotIn(noise, text)

    def test_without_a_long_article_the_whole_body_is_used(self) -> None:
        html = "<html><body><article><p>短</p></article><div><p>第一段內容</p><p>第二段內容</p></div></body></html>"
        text = simple_extract(html)

        self.assertIn("第一段內容\n第二段內容", text)
        self.assertIn("短", text)

    def test_entities_whitespace_and_repeated_lines(self) -> None:
        text = simple_extract("<p>A &amp; B&nbsp; C</p><p>A &amp; B  C</p><br/><p>\n  D </p>")
        self.assertEqual(text, "A & B C\nD")


class TextExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_trafilatura_runs_in_the_extractor_threads_with_the_fast_precise_options(self) -> None:
        fake = _FakeTrafilatura()
        extractor = TextExtractor(load_trafilatura=lambda: fake)
        try:
            text = await extractor.extract(
                _ARTICLE.encode("cp950"), url="https://news.example/a", charset="big5"
            )
        finally:
            extractor.close()

        self.assertEqual(text, "trafilatura 的正文")
        [(html, kwargs)] = fake.calls
        # Decoded here with the page's charset, not left to trafilatura's guess.
        self.assertIn("颱風今晚接近台灣", html)
        self.assertEqual(
            kwargs,
            {
                "url": "https://news.example/a",
                "fast": True,
                "favor_precision": True,
                "include_comments": False,
                "include_tables": False,
                "include_links": False,
                "deduplicate": False,
                "output_format": "txt",
            },
        )
        self.assertTrue(fake.threads[0].startswith("page-extract"))

    async def test_the_simple_extractor_takes_over_when_trafilatura_fails_or_finds_nothing(self) -> None:
        for fake in (_FakeTrafilatura(error=ValueError("lxml")), _FakeTrafilatura(result=None), _FakeTrafilatura(result="  ")):
            with self.subTest(result=fake.result, error=fake.error):
                extractor = TextExtractor(load_trafilatura=lambda fake=fake: fake)
                try:
                    text = await extractor.extract(_ARTICLE.encode(), url="https://news.example/a")
                finally:
                    extractor.close()
                self.assertEqual(len(fake.calls), 1)
                self.assertIn("避免前往山區與海邊", text)
                self.assertNotIn("版權所有", text)

    async def test_a_missing_trafilatura_is_tried_once(self) -> None:
        attempts: list[int] = []

        def missing():
            attempts.append(1)
            raise ImportError("No module named 'trafilatura'")

        extractor = TextExtractor(load_trafilatura=missing)
        try:
            with self.assertLogs("linebot", level="WARNING") as logs:
                first = await extractor.extract(_ARTICLE.encode())
            second = await extractor.extract(_ARTICLE.encode())
        finally:
            extractor.close()

        self.assertEqual(first, second)
        self.assertIn("颱風今晚接近台灣", first)
        self.assertEqual(len(attempts), 1)
        self.assertIn("trafilatura is not installed", "\n".join(logs.output))

    async def test_plain_text_is_not_parsed_as_html(self) -> None:
        fake = _FakeTrafilatura()
        extractor = TextExtractor(load_trafilatura=lambda: fake)
        try:
            text = await extractor.extract(b"line one\n\n  <b>line two</b>  ", content_type="text/plain")
        finally:
            extractor.close()

        self.assertEqual(text, "line one\n<b>line two</b>")
        self.assertEqual(fake.calls, [])

    @unittest.skipUnless(importlib.util.find_spec("trafilatura"), "trafilatura is not installed")
    async def test_the_real_trafilatura_reads_a_zh_tw_article(self) -> None:
        extractor = TextExtractor()
        try:
            text = await extractor.extract(_ARTICLE.encode("cp950"), url="https://news.example/a", charset="big5")
        finally:
            extractor.close()

        self.assertIn("颱風預計今晚接近台灣東部海面", text)
        self.assertNotIn("不該出現", text)

    @unittest.skipUnless(importlib.util.find_spec("trafilatura"), "trafilatura is not installed")
    async def test_the_same_page_read_again_gives_the_same_text(self) -> None:
        # Review: deduplicate=True fills a process-wide cache, so from the
        # fourth read of a page on, its long paragraphs were dropped.
        paragraph = "中央氣象署表示，颱風預計今晚接近台灣東部海面，各地將有強風豪雨，請民眾注意防颱準備。" * 3
        page = f"<html><body><article><h1>颱風動態</h1><p>{paragraph}</p><p>短句</p></article></body></html>"
        extractor = TextExtractor()
        try:
            texts = [await extractor.extract(page.encode(), url="https://news.example/same") for _ in range(5)]
        finally:
            extractor.close()

        self.assertIn(paragraph, texts[0])
        self.assertEqual(texts, [texts[0]] * 5)


class _FakeFetcher:
    def __init__(self, pages: dict, delays: dict | None = None) -> None:
        self.pages = pages
        self.delays = delays or {}
        self.closed = False
        self.started: list[str] = []
        self.cancelled: list[str] = []

    async def fetch(self, url: str) -> FetchedPage:
        self.started.append(url)
        try:
            await asyncio.sleep(self.delays.get(url, 0))
        except asyncio.CancelledError:
            self.cancelled.append(url)
            raise
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        body = page if isinstance(page, bytes) else page.encode()
        return FetchedPage(url=url, body=body, content_type="text/html", charset="utf-8")

    async def close(self) -> None:
        self.closed = True


class _EchoExtractor:
    """Returns the body as the page text."""

    closed = False

    async def extract(self, body, *, url="", charset=None, content_type="text/html"):
        return body.decode()

    def close(self) -> None:
        self.closed = True


_LONG = "長" * MIN_PAGE_CHARS


class PageReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_pages_concurrently_and_names_the_failures(self) -> None:
        fetcher = _FakeFetcher(
            {
                "https://a.example/": f"A{_LONG}",
                "https://b.example/": BlockedURLError("private"),
                "https://c.example/": FetchError("HTTP 404"),
                "https://d.example/": "短",  # too little text: a failed read
            }
        )
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=1.0)

        pages = await reader.read(
            ["https://a.example/", "https://b.example/", "https://a.example/", "https://c.example/", "https://d.example/"]
        )

        self.assertEqual([page.url for page in pages.results], ["https://a.example/"])
        self.assertEqual(pages.results[0].content, f"A{_LONG}")
        self.assertEqual(
            pages.failed_urls, ["https://b.example/", "https://c.example/", "https://d.example/"]
        )

    async def test_the_budget_keeps_what_finished(self) -> None:
        fetcher = _FakeFetcher(
            {"https://fast.example/": f"F{_LONG}", "https://slow.example/": f"S{_LONG}"},
            delays={"https://slow.example/": 5.0},
        )
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=0.2)

        started = asyncio.get_running_loop().time()
        pages = await reader.read(["https://slow.example/", "https://fast.example/"])

        self.assertLess(asyncio.get_running_loop().time() - started, 2.0)
        self.assertEqual([page.url for page in pages.results], ["https://fast.example/"])
        self.assertEqual(pages.failed_urls, ["https://slow.example/"])

    async def test_the_callers_deadline_cuts_the_budget_short(self) -> None:
        # The stage has 1 s left of a 6 s budget: read() returns before
        # the caller's own timeout, with the page that finished, and stops
        # the one that did not.
        fetcher = _FakeFetcher(
            {"https://fast.example/": f"F{_LONG}", "https://slow.example/": f"S{_LONG}"},
            delays={"https://fast.example/": 0.05, "https://slow.example/": 30.0},
        )
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=6.0)
        deadline = time.monotonic() + 1.0

        async with asyncio.timeout_at(asyncio.get_running_loop().time() + 1.0):
            pages = await reader.read(
                ["https://fast.example/", "https://slow.example/"], deadline=deadline
            )

        self.assertLessEqual(time.monotonic(), deadline - DEADLINE_MARGIN_SECONDS + 0.2)
        self.assertEqual([page.url for page in pages.results], ["https://fast.example/"])
        self.assertEqual(pages.failed_urls, ["https://slow.example/"])
        self.assertEqual(fetcher.cancelled, ["https://slow.example/"])

    async def test_no_time_left_reads_nothing(self) -> None:
        fetcher = _FakeFetcher({"https://a.example/": f"A{_LONG}"})
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=6.0)

        pages = await reader.read(
            ["https://a.example/"], deadline=time.monotonic() + DEADLINE_MARGIN_SECONDS / 2
        )

        self.assertEqual((pages.results, pages.failed_urls), ([], ["https://a.example/"]))
        self.assertEqual(fetcher.started, [])

    async def test_a_cancelled_read_leaves_no_fetch_behind(self) -> None:
        fetcher = _FakeFetcher(
            {"https://slow.example/": f"S{_LONG}", "https://slower.example/": f"T{_LONG}"},
            delays={"https://slow.example/": 30.0, "https://slower.example/": 60.0},
        )
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=6.0)
        before = asyncio.all_tasks()

        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(0.2):
                await reader.read(["https://slow.example/", "https://slower.example/"])

        self.assertEqual(sorted(fetcher.cancelled), ["https://slow.example/", "https://slower.example/"])
        self.assertEqual(asyncio.all_tasks() - before, set())

    async def test_enrich_passes_the_deadline_on(self) -> None:
        fetcher = _FakeFetcher(
            {"https://1.example/": f"page one {_LONG}", "https://2.example/": f"page two {_LONG}"},
            delays={"https://2.example/": 30.0},
        )
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=6.0)
        response = WebSearchResponse(
            query="q",
            results=[
                SearchResult(title=f"t{n}", url=f"https://{n}.example/", content=f"snippet {n}")
                for n in (1, 2)
            ],
        )
        started = time.monotonic()

        await reader.enrich(response, 2, deadline=started + DEADLINE_MARGIN_SECONDS + 0.3)

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(response.results[0].raw_content, f"page one {_LONG}")
        self.assertIsNone(response.results[1].raw_content)
        self.assertEqual(fetcher.cancelled, ["https://2.example/"])

    async def test_nothing_to_read(self) -> None:
        pages = await PageReader(_FakeFetcher({}), _EchoExtractor()).read([])
        self.assertEqual((pages.results, pages.failed_urls), ([], []))

    async def test_enrich_puts_page_text_into_the_top_results_only(self) -> None:
        fetcher = _FakeFetcher(
            {
                "https://1.example/": f"page one {_LONG}",
                "https://2.example/": FetchError("timeout"),
                "https://3.example/": f"page three {_LONG}",
            }
        )
        reader = PageReader(fetcher, _EchoExtractor())
        response = WebSearchResponse(
            query="q",
            results=[
                SearchResult(title=f"t{n}", url=f"https://{n}.example/", content=f"snippet {n}")
                for n in (1, 2, 3, 4)
            ],
        )

        enriched = await reader.enrich(response, 2)

        one, two, three, four = enriched.results
        self.assertEqual(one.raw_content, f"page one {_LONG}")
        self.assertIsNone(two.raw_content)  # failed: the snippet stays
        self.assertIsNone(three.raw_content)  # beyond top_n: never fetched
        self.assertIsNone(four.raw_content)
        self.assertIn("page one", enriched.to_context_text())
        self.assertIn("snippet 2", enriched.to_context_text())

    async def test_a_page_shorter_than_its_snippet_keeps_the_snippet(self) -> None:
        reader = PageReader(_FakeFetcher({"https://1.example/": f"x{_LONG}"}), _EchoExtractor())
        response = WebSearchResponse(
            query="q", results=[SearchResult(title="t", url="https://1.example/", content="s" * 500)]
        )

        await reader.enrich(response, 3)

        self.assertIsNone(response.results[0].raw_content)
        self.assertEqual(await reader.enrich(WebSearchResponse(query="q", results=[]), 3), WebSearchResponse(query="q", results=[]))

    async def test_an_undecodable_page_is_a_failed_read(self) -> None:
        # Its URL goes on to the next reader, and a search result keeps its snippet.
        undecodable = ("<p>" + "繁體中文網頁內容" * 20 + "</p>").encode("cp950")
        fetcher = _FakeFetcher({"https://a.example/": f"A{_LONG}", "https://b.example/": undecodable})
        extractor = TextExtractor(load_trafilatura=lambda: SimpleNamespace(extract=lambda html, **_: html))
        reader = PageReader(fetcher, extractor, budget_seconds=5.0)
        try:
            with patch("src.services.search.extract._guess_decode", return_value=None):
                pages = await reader.read(["https://a.example/", "https://b.example/"])
        finally:
            extractor.close()

        self.assertEqual([page.url for page in pages.results], ["https://a.example/"])
        self.assertEqual(pages.failed_urls, ["https://b.example/"])

    async def test_close_closes_both_parts(self) -> None:
        fetcher, extractor = _FakeFetcher({}), _EchoExtractor()
        await PageReader(fetcher, extractor).close()
        self.assertTrue(fetcher.closed and extractor.closed)


class RealPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_extractor_close_stops_the_pool(self) -> None:
        extractor = TextExtractor(load_trafilatura=lambda: SimpleNamespace(extract=lambda html, **_: "x"))
        self.assertEqual(await extractor.extract(b"<p>x</p>"), "x")
        extractor.close()
        with self.assertRaises(RuntimeError):
            await extractor.extract(b"<p>x</p>")


if __name__ == "__main__":
    unittest.main()
