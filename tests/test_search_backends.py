"""Search backends (ADR-0009): the chain that tries them in order, Tavily
behind it, and the web search agent over both.

Every HTTP call goes to an ``httpx.MockTransport``; nothing leaves the
machine. The chain's ordering and fallback rules are tested with a scripted
backend in front of Tavily: the chain is the seam another backend plugs into.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import time
import unittest
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from src.agents.web_search_agent import (
    SEARCH_UNAVAILABLE_CAVEAT,
    SOME_URLS_UNREADABLE_CAVEAT,
    URL_UNREADABLE_CAVEAT,
    WebSearchAgent,
)
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.providers.base import ProviderResponse
from src.services.search.backend import ChainedSearchBackend, SearchBackend
from src.services.search.extract import PageReader
from src.services.search.page_fetcher import FetchedPage
from src.services.search.rerank import SearchReranker
from src.services.search.tavily import MAX_COOLDOWN_SECONDS, TavilyBackend
from src.services.search.types import (
    RESULT_TEXT_CHARS,
    ExtractResponse,
    ExtractResult,
    NoResultsError,
    SearchBackendError,
    SearchResult,
    WebSearchResponse,
)

# Tavily's results for a Taipei weather search, in its rank order.
_TAIPEI = [
    {
        "title": "臺北市 - 維基百科",
        "url": "https://zh.wikipedia.org/wiki/臺北市",
        "content": "臺北市是中華民國的首都。",
        "score": 0.9,
        "published_date": None,
    },
    {
        "title": "臺北市 天氣預報",
        "url": "https://www.cwa.gov.tw/V8/C/W/County/County.html?CID=63",
        "content": "今天多雲時晴，午後雷陣雨。",
        "score": 0.8,
    },
    {
        "title": "台北午後雷陣雨",
        "url": "https://news.example/tpe?id=7",
        "content": "氣象署提醒午後雷陣雨。",
        "score": 0.7,
        "published_date": "2026-09-29T08:00:00",
    },
]


class _Recorder:
    """A MockTransport handler that answers from a queue and keeps the requests."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _json(payload, status: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers)


def _tavily_json(results=None, answer="Tavily 的摘要", credits=1):
    return {
        "query": "q",
        "answer": answer,
        "results": results
        if results is not None
        else [
            {
                "title": "Tavily 結果",
                "url": "https://tavily.example/a",
                "content": "tavily snippet",
                "score": 0.81,
                "published_date": "Mon, 29 Sep 2026 06:00:00 GMT",
            }
        ],
        "usage": {"credits": credits},
    }


class TavilyTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_key_is_sent_as_bearer(self) -> None:
        recorder = _Recorder(_json(_tavily_json()))
        backend = TavilyBackend("tvly-key", transport=recorder.transport)
        try:
            response = await backend.search("台北 天氣")
        finally:
            await backend.close()

        request = recorder.requests[0]
        self.assertEqual(str(request.url), "https://api.tavily.com/search")
        self.assertEqual(request.headers["Authorization"], "Bearer tvly-key")
        self.assertNotIn("X-Tavily-Access-Mode", request.headers)
        self.assertEqual(backend.mode, "key")
        self.assertEqual((response.backend, response.answer, response.answer_label), ("tavily", "Tavily 的摘要", "AI Summary"))
        self.assertEqual(response.results[0].published, "2026-09-29")

    async def test_published_dates_are_read_as_iso_or_rfc_2822(self) -> None:
        dates = (
            "Mon, 29 Sep 2026 06:00:00 GMT",
            "2026-09-29T08:00:00",
            "2026-09-30T08:00:00+08:00",
            "2026-09-28",
            None,
            "",
            "not a date",
        )
        results = [
            {"title": f"r{i}", "url": f"https://a.example/{i}", "content": "c", "published_date": value}
            for i, value in enumerate(dates)
        ]
        recorder = _Recorder(_json(_tavily_json(results=results)))
        backend = TavilyBackend("k", transport=recorder.transport)
        try:
            response = await backend.search("q", max_results=10)
        finally:
            await backend.close()

        self.assertEqual(
            [result.published for result in response.results],
            ["2026-09-29", "2026-09-29", "2026-09-30", "2026-09-28", None, None, None],
        )

    async def test_without_a_key_it_goes_keyless(self) -> None:
        recorder = _Recorder(_json(_tavily_json()))
        backend = TavilyBackend("", keyless_enabled=True, transport=recorder.transport)
        try:
            await backend.search("台北 天氣")
        finally:
            await backend.close()

        request = recorder.requests[0]
        self.assertEqual(request.headers["X-Tavily-Access-Mode"], "keyless")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(backend.get_status()["mode"], "keyless")
        off = TavilyBackend("", keyless_enabled=False)
        self.assertFalse(off.is_configured)
        with self.assertRaises(SearchBackendError):
            await off.search("q")
        # Keyless is an opt-in: without it and without a key, Tavily is off.
        self.assertEqual(TavilyBackend("").mode, "off")

    async def test_every_search_is_basic_without_auto_parameters(self) -> None:
        recorder = _Recorder(_json(_tavily_json()))
        backend = TavilyBackend("k", transport=recorder.transport)
        try:
            await backend.search("台北 天氣", max_results=5, time_range="day", country="taiwan")
        finally:
            await backend.close()

        self.assertEqual(
            recorder.body(),
            {
                "query": "台北 天氣",
                "topic": "general",
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
                "auto_parameters": False,
                "include_usage": True,
                "time_range": "day",
                "country": "taiwan",
            },
        )

    async def test_country_goes_only_with_the_general_topic(self) -> None:
        recorder = _Recorder(_json(_tavily_json()))
        backend = TavilyBackend("k", transport=recorder.transport)
        try:
            for topic, sent in ((None, "taiwan"), ("general", "taiwan"), ("news", None), ("finance", None)):
                with self.subTest(topic=topic):
                    await backend.search("台北", topic=topic, country="taiwan")
                    body = recorder.body()
                    self.assertEqual(body["topic"], topic or "general")
                    self.assertEqual(body.get("country"), sent)
        finally:
            await backend.close()

    async def test_429_honours_retry_after(self) -> None:
        clock, wall = _Clock(), 1_790_000_000.0
        later = datetime.fromtimestamp(wall + 300, UTC)
        for headers, expected in (
            ({"Retry-After": "120"}, 120),
            ({"Retry-After": format_datetime(later, usegmt=True)}, 300),
            ({}, 60),
            ({"Retry-After": "86400"}, MAX_COOLDOWN_SECONDS),
        ):
            with self.subTest(headers=headers):
                recorder = _Recorder(_json({"detail": {"error": "rate"}}, 429, headers), _json(_tavily_json()))
                backend = TavilyBackend(
                    "", keyless_enabled=True, transport=recorder.transport, clock=clock, wall_clock=lambda: wall
                )
                try:
                    with self.assertRaises(SearchBackendError):
                        await backend.search("q")
                    self.assertEqual(backend.get_status()["cooldown_seconds"], expected)
                    with self.assertRaises(SearchBackendError):
                        await backend.search("q")
                    self.assertEqual(len(recorder.requests), 1)
                    clock.now += expected + 1
                    self.assertTrue((await backend.search("q")).has_results)
                finally:
                    await backend.close()

    async def test_plan_limits_and_refused_keys_cool_down_for_an_hour(self) -> None:
        for status in (432, 433, 401, 403):
            with self.subTest(status=status):
                recorder = _Recorder(_json({"detail": {"error": "limit"}}, status))
                backend = TavilyBackend("k", transport=recorder.transport, clock=_Clock())
                try:
                    with self.assertRaises(SearchBackendError):
                        await backend.search("q")
                    self.assertEqual(backend.get_status()["cooldown_seconds"], MAX_COOLDOWN_SECONDS)
                finally:
                    await backend.close()

    async def test_other_failures_raise_without_a_cooldown(self) -> None:
        for answer in (httpx.Response(500, text="x"), httpx.ConnectTimeout("t"), httpx.Response(200, text="nope")):
            with self.subTest(answer=answer):
                recorder = _Recorder(answer)
                backend = TavilyBackend("k", transport=recorder.transport)
                try:
                    with self.assertRaises(SearchBackendError):
                        await backend.search("q")
                    self.assertEqual(backend.get_status()["cooldown_seconds"], 0)
                finally:
                    await backend.close()

    async def test_empty_answers_raise_no_results_and_credits_add_up(self) -> None:
        recorder = _Recorder(_json(_tavily_json(results=[], answer=None, credits=1)), _json(_tavily_json(credits=2)))
        backend = TavilyBackend("k", transport=recorder.transport)
        try:
            with self.assertRaises(NoResultsError):
                await backend.search("q")
            await backend.search("q")
        finally:
            await backend.close()
        self.assertEqual(backend.get_status()["credits_used"], 3)

    async def test_extract_reads_pages_at_basic_depth(self) -> None:
        recorder = _Recorder(
            _json(
                {
                    "results": [{"url": "https://a.example/post", "raw_content": "全文內容"}],
                    "failed_results": [{"url": "https://b.example/", "error": "403"}],
                    "usage": {"credits": 1},
                }
            )
        )
        backend = TavilyBackend("k", transport=recorder.transport)
        try:
            pages = await backend.extract(["https://a.example/post", "https://b.example/", "https://c.example/"])
        finally:
            await backend.close()

        self.assertEqual(str(recorder.requests[0].url), "https://api.tavily.com/extract")
        self.assertEqual(
            recorder.body(),
            {
                "urls": ["https://a.example/post", "https://b.example/", "https://c.example/"],
                "extract_depth": "basic",
                "format": "text",
                "include_usage": True,
            },
        )
        self.assertEqual(pages.results, [ExtractResult(url="https://a.example/post", content="全文內容")])
        self.assertEqual(pages.failed_urls, ["https://b.example/", "https://c.example/"])


class _ScriptedBackend:
    """A search-only backend that answers from a script, one entry per
    search (the last one repeats): a response, or an exception to raise."""

    supports_extract = False

    def __init__(self, *answers, name: str = "first", configured: bool = True) -> None:
        self.name = name
        self.is_configured = configured
        self.answers = list(answers)
        self.queries: list[str] = []

    async def search(self, query: str, **hints) -> WebSearchResponse:
        self.queries.append(query)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def extract(self, urls):
        raise AssertionError("a search-only backend is never asked to read pages")

    def get_status(self) -> dict:
        return {"name": self.name}

    async def close(self) -> None:
        pass


def _found(query: str = "台北 天氣") -> WebSearchResponse:
    return WebSearchResponse(
        query=query, results=[SearchResult(title="臺北市 天氣預報", url="https://www.cwa.gov.tw/", content="多雲")]
    )


def _nothing(query: str = "台北 天氣") -> WebSearchResponse:
    return WebSearchResponse(query=query, results=[])


class ChainTests(unittest.IsolatedAsyncioTestCase):
    """The chain's rules, with a scripted backend in front of Tavily."""

    async def asyncSetUp(self) -> None:
        self.closables: list = []

    async def asyncTearDown(self) -> None:
        for backend in self.closables:
            await backend.close()

    def chain(self, first_answers, tavily_answers, *, key: str = "k") -> tuple[ChainedSearchBackend, _ScriptedBackend, _Recorder]:
        first = _ScriptedBackend(*first_answers)
        tavily = _Recorder(*tavily_answers)
        chain = ChainedSearchBackend(
            [first, TavilyBackend(key, keyless_enabled=True, transport=tavily.transport, clock=_Clock())]
        )
        self.closables.append(chain)
        return chain, first, tavily

    async def test_the_first_backend_that_answers_wins(self) -> None:
        chain, first, tavily = self.chain([_found()], [_json(_tavily_json())])

        response = await chain.search("台北 天氣")

        self.assertEqual(response.backend, "first")
        self.assertFalse(response.degraded)
        self.assertEqual((len(first.queries), len(tavily.requests)), (1, 0))
        self.assertEqual(chain.get_status()["answered_by"], {"first": 1})

    async def test_nothing_found_falls_back_to_tavily_with_the_key_and_the_hints(self) -> None:
        # Found nothing either way: raised NoResultsError or answered empty.
        for nothing in (NoResultsError("no results"), _nothing()):
            with self.subTest(nothing=type(nothing).__name__):
                chain, _, tavily = self.chain([nothing], [_json(_tavily_json())])

                response = await chain.search("台北 天氣", topic="news", time_range="day", country="taiwan")

                self.assertEqual(response.backend, "tavily")
                self.assertEqual(tavily.requests[0].headers["Authorization"], "Bearer k")
                body = tavily.body()
                self.assertEqual((body["topic"], body["time_range"]), ("news", "day"))
                self.assertNotIn("country", body)
                self.assertEqual(chain.get_status()["answered_by"], {"tavily": 1})

    async def test_a_failing_backend_falls_back_to_keyless_tavily(self) -> None:
        chain, _, tavily = self.chain([SearchBackendError("HTTP 403")], [_json(_tavily_json())], key="")

        response = await chain.search("台北 天氣")

        self.assertEqual(response.backend, "tavily")
        self.assertEqual(tavily.requests[0].headers["X-Tavily-Access-Mode"], "keyless")

    async def test_all_failing_is_an_empty_degraded_answer(self) -> None:
        chain, first, tavily = self.chain(
            [SearchBackendError("unreachable")], [_json({"detail": {"error": "rate"}}, 429)]
        )

        response = await chain.search("台北 天氣")

        self.assertTrue(response.degraded)
        self.assertTrue(response.all_failed)
        self.assertTrue(response.is_empty)
        self.assertEqual(response.backend, "")
        self.assertEqual((len(first.queries), len(tavily.requests)), (1, 1))
        status = chain.get_status()
        self.assertEqual((status["searches"], status["degraded"]), (1, 1))
        self.assertEqual([b["name"] for b in status["backends"]], ["first", "tavily"])
        # Tavily cooled down: the next search does not ask it again.
        await chain.search("台北 天氣")
        self.assertEqual((len(first.queries), len(tavily.requests)), (2, 1))

    async def test_a_backend_that_found_nothing_is_not_a_failed_search(self) -> None:
        # Degraded either way; only "every backend failed" rules out a retry.
        for name, first_answer, tavily_answer in (
            ("first empty, tavily down", _nothing(), httpx.Response(500)),
            ("first down, tavily empty", SearchBackendError("down"), _json(_tavily_json(results=[], answer=None))),
            ("both empty", NoResultsError("no results"), _json(_tavily_json(results=[], answer=None))),
        ):
            with self.subTest(name):
                chain, _, _ = self.chain([first_answer], [tavily_answer])

                response = await chain.search("台北 天氣")

                self.assertTrue(response.degraded)
                self.assertFalse(response.all_failed)

    async def test_an_unexpected_exception_also_moves_on(self) -> None:
        broken = SimpleNamespace(
            name="broken",
            supports_extract=True,
            is_configured=True,
            search=AsyncMock(side_effect=RuntimeError("bug")),
            extract=AsyncMock(side_effect=RuntimeError("bug")),
            get_status=lambda: {"name": "broken"},
            close=AsyncMock(),
        )
        tavily = _Recorder(_json(_tavily_json()))
        chain = ChainedSearchBackend([broken, TavilyBackend("k", transport=tavily.transport)])
        self.closables.append(chain)
        self.assertIsInstance(chain.backends[1], SearchBackend)
        self.assertIsInstance(_ScriptedBackend(_found()), SearchBackend)

        self.assertEqual((await chain.search("q")).backend, "tavily")

    async def test_unconfigured_backends_are_skipped(self) -> None:
        first = _ScriptedBackend(_found(), configured=False)
        chain = ChainedSearchBackend([first, TavilyBackend("", keyless_enabled=False)])
        self.assertFalse(chain.is_configured)
        self.assertFalse(chain.supports_extract)
        nobody = await chain.search("q")
        self.assertTrue(nobody.degraded and nobody.all_failed)
        self.assertEqual(first.queries, [])
        pages = await chain.extract(["https://a.example/"])
        self.assertEqual((pages.results, pages.failed_urls), ([], ["https://a.example/"]))

    async def test_extract_skips_a_search_only_backend_and_reports_what_failed(self) -> None:
        tavily_pages = {"results": [{"url": "https://a.example/", "raw_content": "內容"}], "failed_results": []}
        chain, _, _ = self.chain([_found()], [_json(tavily_pages)])
        self.assertTrue(chain.supports_extract)

        pages = await chain.extract(["https://a.example/"])
        self.assertEqual(pages.results[0].content, "內容")

        chain, _, _ = self.chain([_found()], [httpx.Response(500)])
        failed = await chain.extract(["https://a.example/"])
        self.assertEqual((failed.results, failed.failed_urls), ([], ["https://a.example/"]))


def _answer_chain(text: str = "回答") -> SimpleNamespace:
    return SimpleNamespace(generate=AsyncMock(return_value=ProviderResponse(text=text, model="m")))


class _FakeReader:
    def __init__(self, read=None, enrich_text: str | None = None) -> None:
        self.read_calls: list[list[str]] = []
        self.enrich_calls: list[int] = []
        self.deadlines: list[float | None] = []
        self._read = read or ExtractResponse(results=[], failed_urls=[])
        self._enrich_text = enrich_text

    async def read(self, urls, *, deadline=None):
        self.read_calls.append(list(urls))
        self.deadlines.append(deadline)
        return self._read(urls) if callable(self._read) else self._read

    async def enrich(self, response, top_n, *, deadline=None):
        self.enrich_calls.append(top_n)
        self.deadlines.append(deadline)
        if self._enrich_text:
            response.results[0].raw_content = self._enrich_text
        return response


class _TimedFetcher:
    """Pages that take their time; records the fetches that were stopped."""

    def __init__(self, delays: dict[str, float], default_delay: float = 0.0) -> None:
        self.delays = delays
        self.default_delay = default_delay
        self.cancelled: list[str] = []

    async def fetch(self, url: str) -> FetchedPage:
        try:
            await asyncio.sleep(self.delays.get(url, self.default_delay))
        except asyncio.CancelledError:
            self.cancelled.append(url)
            raise
        body = f"page text of {url} " + "內文" * 100
        return FetchedPage(url=url, body=body.encode(), content_type="text/html", charset="utf-8")


class _EchoExtractor:
    async def extract(self, body, *, url="", charset=None, content_type="text/html"):
        return body.decode()


class AgentWithBackendsTests(unittest.IsolatedAsyncioTestCase):
    """WebSearchAgent over the real chain and Tavily (HTTP mocked), as
    composition builds it."""

    async def asyncSetUp(self) -> None:
        self.chains: list = []

    async def asyncTearDown(self) -> None:
        for chain in self.chains:
            await chain.close()

    def backends(self, *tavily_answers, key: str = "") -> tuple[ChainedSearchBackend, _Recorder]:
        tavily = _Recorder(*tavily_answers)
        chain = ChainedSearchBackend(
            [TavilyBackend(key, keyless_enabled=True, transport=tavily.transport, clock=_Clock())]
        )
        self.chains.append(chain)
        return chain, tavily

    def agent(self, answer, search, reader=None, **settings) -> WebSearchAgent:
        return WebSearchAgent(
            Settings(_env_file=None, **settings),
            answer,
            targets=[],
            search=search,
            page_reader=reader,
        )

    async def test_everything_failing_answers_from_the_model_with_the_caveat(self) -> None:
        answer = _answer_chain("依既有知識的回答")
        chain, tavily = self.backends(_json({"detail": {"error": "rate"}}, 429))

        response = await self.agent(answer, chain).process(AgentRequest(text="台北今天新聞"))

        self.assertEqual(response.text, f"{SEARCH_UNAVAILABLE_CAVEAT}\n依既有知識的回答")
        messages = answer.generate.await_args.kwargs["messages"]
        self.assertIn("沒有網頁內容或搜尋結果可以參考", messages[-2]["content"])
        # Every backend failed, so there is no relaxed retry: it would only
        # fail again.
        self.assertEqual(len(tavily.requests), 1)
        self.assertEqual(chain.get_status()["degraded"], 1)

    async def test_a_search_that_found_nothing_is_retried_without_filters(self) -> None:
        answer = _answer_chain("台北新聞整理")
        chain, tavily = self.backends(
            _json(_tavily_json(results=[], answer=None)), _json(_tavily_json(results=_TAIPEI))
        )

        response = await self.agent(answer, chain).process(
            AgentRequest(text="台北今天新聞", disable_thinking=True)
        )

        self.assertEqual(response.text, "台北新聞整理")
        # Nothing among today's news: one retry without the filters.
        self.assertEqual(len(tavily.requests), 2)
        self.assertEqual((tavily.body(0)["topic"], tavily.body(0)["time_range"]), ("news", "day"))
        self.assertEqual(tavily.body(1)["topic"], "general")
        for hint in ("time_range", "country"):
            self.assertNotIn(hint, tavily.body(1))
        self.assertIn("臺北市 天氣預報", answer.generate.await_args.kwargs["messages"][-2]["content"])

    async def test_results_and_page_text_reach_the_answer(self) -> None:
        answer = _answer_chain("台北今天多雲")
        chain, tavily = self.backends(_json(_tavily_json(results=_TAIPEI, answer="28°C")))
        reader = _FakeReader(enrich_text="氣象署網頁全文")

        response = await self.agent(answer, chain, reader).process(AgentRequest(text="台北今天天氣"))

        self.assertEqual(response.text, "台北今天多雲")
        body = tavily.body()
        self.assertEqual((body["query"], body["time_range"], body["topic"]), ("台北今天天氣", "day", "general"))
        self.assertEqual(len(tavily.requests), 1)
        self.assertEqual(reader.enrich_calls, [3])  # SEARCH_FETCH_TOP_N
        context = answer.generate.await_args.kwargs["messages"][-2]["content"]
        self.assertIn("AI Summary: 28°C", context)
        self.assertIn("氣象署網頁全文", context)

    async def test_the_country_rule_through_the_agent(self) -> None:
        # Tavily gets each search with its hints.
        for text, topic, country in (
            ("台北今天天氣", "general", "taiwan"),
            ("台北今天新聞", "news", None),
            ("台股今天走勢", "finance", None),
        ):
            with self.subTest(text=text):
                chain, tavily = self.backends(_json(_tavily_json()), key="k")
                await self.agent(_answer_chain(), chain).process(AgentRequest(text=text))
                body = tavily.body(0)
                self.assertEqual((body["topic"], body.get("country"), body["time_range"]), (topic, country, "day"))

    async def test_no_page_reads_when_fetch_top_n_is_zero(self) -> None:
        chain, _ = self.backends(_json(_tavily_json(results=_TAIPEI)))
        reader = _FakeReader(enrich_text="不該出現")
        answer = _answer_chain()

        await self.agent(answer, chain, reader, search_fetch_top_n=0).process(AgentRequest(text="台北天氣"))

        self.assertEqual(reader.enrich_calls, [])
        self.assertNotIn("不該出現", answer.generate.await_args.kwargs["messages"][-2]["content"])

    async def test_a_url_is_read_here_first_then_by_tavily_extract(self) -> None:
        answer = _answer_chain("重點如下")
        read_here = ExtractResponse(
            results=[ExtractResult(url="https://a.example/1", content="本機讀到的內容")],
            failed_urls=["https://b.example/2"],
        )
        reader = _FakeReader(read=read_here)
        chain, tavily = self.backends(
            _json({"results": [{"url": "https://b.example/2", "raw_content": "Tavily 讀到的內容"}], "failed_results": []})
        )

        response = await self.agent(answer, chain, reader).process(
            AgentRequest(text="比較 https://a.example/1 和 https://b.example/2")
        )

        self.assertEqual(response.text, "重點如下")
        self.assertEqual(reader.read_calls, [["https://a.example/1", "https://b.example/2"]])
        self.assertEqual(tavily.body()["urls"], ["https://b.example/2"])
        self.assertEqual([request.url.path for request in tavily.requests], ["/extract"])  # never searched
        context = answer.generate.await_args.kwargs["messages"][-2]["content"]
        self.assertIn("本機讀到的內容", context)
        self.assertIn("Tavily 讀到的內容", context)

    async def test_a_page_nobody_can_read_is_named_and_never_summarised(self) -> None:
        # Code-review round 4 #11: the model got the generic no-results note
        # ("answer from your own knowledge") and the reply blamed the search
        # service, which invited a made-up summary of the unread page.
        answer = _answer_chain("連結打不開")
        reader = _FakeReader(read=lambda urls: ExtractResponse(results=[], failed_urls=list(urls)))
        chain, tavily = self.backends(httpx.Response(500))

        response = await self.agent(answer, chain, reader).process(AgentRequest(text="https://a.example/1這篇在講什麼"))

        self.assertEqual(response.text, f"{URL_UNREADABLE_CAVEAT}\n連結打不開")
        self.assertNotIn("搜尋服務", response.text)
        self.assertEqual(len(tavily.requests), 1)
        messages = answer.generate.await_args.kwargs["messages"]
        note = messages[-2]["content"]
        self.assertIn("無法開啟", note)
        self.assertIn("https://a.example/1", note)
        self.assertIn("不要猜測、摘要或評論這些網頁的內容", note)
        self.assertNotIn("依你既有的知識回答", note)
        self.assertEqual(messages[-1]["content"], "【使用者這次的訊息】\nhttps://a.example/1這篇在講什麼")

    async def test_a_link_that_could_not_be_read_is_named_beside_the_pages_that_were(self) -> None:
        answer = _answer_chain("比較如下")
        reader = _FakeReader(
            read=ExtractResponse(
                results=[ExtractResult(url="https://a.example/1", content="本機讀到的內容")],
                failed_urls=["https://b.example/2"],
            )
        )
        search = SimpleNamespace(is_configured=True, supports_extract=False)
        agent = WebSearchAgent(Settings(_env_file=None), answer, targets=[], search=search, page_reader=reader)

        response = await agent.process(AgentRequest(text="比較 https://a.example/1 和 https://b.example/2"))

        self.assertEqual(response.text, f"{SOME_URLS_UNREADABLE_CAVEAT}\n比較如下")
        messages = answer.generate.await_args.kwargs["messages"]
        self.assertIn("本機讀到的內容", messages[-2]["content"])
        note = messages[-3]["content"]
        self.assertIn("https://b.example/2", note)
        self.assertNotIn("https://a.example/1", note)
        self.assertIn("無法開啟", note)

    async def test_links_never_read_for_lack_of_time_are_named_too(self) -> None:
        answer = _answer_chain("重點")
        reader = _FakeReader()
        agent = WebSearchAgent(Settings(_env_file=None), answer, targets=[], search=None, page_reader=reader)
        # The reply window is gone and the turn too short to read in.
        request = AgentRequest(text="https://a.example/1", received_at=time.time() - 3600, turn_deadline_seconds=10)

        response = await agent.process(request)

        self.assertEqual(reader.read_calls, [])
        self.assertTrue(response.text.startswith(URL_UNREADABLE_CAVEAT))
        self.assertIn("https://a.example/1", answer.generate.await_args.kwargs["messages"][-2]["content"])

    async def test_pages_read_here_stay_when_the_extract_runs_out_of_time(self) -> None:
        answer = _answer_chain("重點")
        reader = _FakeReader(
            read=ExtractResponse(
                results=[ExtractResult(url="https://a.example/1", content="本機讀到的內容")],
                failed_urls=["https://b.example/2"],
            )
        )

        async def slow_extract(urls):
            await asyncio.sleep(30)

        search = SimpleNamespace(is_configured=True, supports_extract=True, extract=slow_extract)
        agent = WebSearchAgent(Settings(_env_file=None), answer, targets=[], search=search, page_reader=reader)
        # 3.5 s for the whole stage once the answer's reserve is kept.
        request = AgentRequest(
            text="https://a.example/1 https://b.example/2", received_at=time.time() - 90 + 20 + 3.5
        )

        response = await agent.process(request)

        # The page not read is named, not passed over.
        self.assertEqual(response.text, f"{SOME_URLS_UNREADABLE_CAVEAT}\n重點")
        self.assertIn("本機讀到的內容", answer.generate.await_args.kwargs["messages"][-2]["content"])

    async def test_pages_read_before_the_stage_ends_reach_the_answer(self) -> None:
        # 3.5 s of stage against a 6 s page budget: the reader stops before
        # the stage does, keeps the fast page and stops the slow fetch.
        fetcher = _TimedFetcher({"https://fast.example/": 0.05, "https://slow.example/": 30.0})
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=6.0)
        answer = _answer_chain("重點")
        search = SimpleNamespace(is_configured=True, supports_extract=False)
        agent = WebSearchAgent(Settings(_env_file=None), answer, targets=[], search=search, page_reader=reader)
        request = AgentRequest(
            text="https://fast.example/ https://slow.example/",
            received_at=time.time() - 90 + 20 + 3.5,
            disable_thinking=True,
        )
        before = asyncio.all_tasks()
        started = time.monotonic()

        response = await agent.process(request)

        self.assertLess(time.monotonic() - started, 3.5)
        # The page not read is named, not passed over.
        self.assertEqual(response.text, f"{SOME_URLS_UNREADABLE_CAVEAT}\n重點")
        context = answer.generate.await_args.kwargs["messages"][-2]["content"]
        self.assertIn("page text of https://fast.example/", context)
        self.assertNotIn("https://slow.example/", context)
        self.assertEqual(fetcher.cancelled, ["https://slow.example/"])
        self.assertEqual(asyncio.all_tasks() - before, set())

    async def test_top_pages_read_before_the_stage_ends_reach_the_answer(self) -> None:
        cwa = _TAIPEI[1]["url"]
        fetcher = _TimedFetcher({cwa: 0.05}, default_delay=30.0)
        reader = PageReader(fetcher, _EchoExtractor(), budget_seconds=6.0)
        chain, _ = self.backends(_json(_tavily_json(results=_TAIPEI)))
        answer = _answer_chain("台北今天多雲")
        request = AgentRequest(
            text="台北今天天氣", received_at=time.time() - 90 + 20 + 3.5, disable_thinking=True
        )
        started = time.monotonic()

        response = await self.agent(answer, chain, reader).process(request)

        self.assertLess(time.monotonic() - started, 3.5)
        self.assertEqual(response.text, "台北今天多雲")
        context = answer.generate.await_args.kwargs["messages"][-2]["content"]
        self.assertIn(f"page text of {cwa}", context)
        self.assertIn("氣象署提醒午後雷陣雨。", context)  # a snippet whose page was not read
        self.assertEqual(len(fetcher.cancelled), 2)

    async def test_the_reader_gets_the_stage_deadline(self) -> None:
        reader = _FakeReader(enrich_text="全文")
        chain, _ = self.backends(_json(_tavily_json(results=_TAIPEI)))
        request = AgentRequest(text="台北今天天氣", disable_thinking=True)

        await self.agent(_answer_chain(), chain, reader).process(request)

        [deadline] = reader.deadlines
        self.assertAlmostEqual(deadline, request.reply_deadline() - 20, delta=0.5)

    async def test_without_a_page_reader_urls_go_straight_to_extract(self) -> None:
        chain, tavily = self.backends(_json({"results": [{"url": "https://a.example/1", "raw_content": "內容"}]}))
        answer = _answer_chain()
        await self.agent(answer, chain).process(AgentRequest(text="https://a.example/1"))

        self.assertEqual(tavily.body()["urls"], ["https://a.example/1"])
        self.assertEqual(
            answer.generate.await_args.kwargs["messages"][-1]["content"],
            "【使用者這次的訊息】\n請閱讀我提供的網址內容，整理重點後回覆。",
        )


class _FakeEmbedder:
    is_configured = True

    def __init__(self, query=None, passages=None, delay: float = 0.0) -> None:
        self.query = query
        self.passages = passages
        self.delay = delay
        self.calls: list[tuple] = []

    async def embed_text(self, text):
        self.calls.append(("query", text))
        await asyncio.sleep(self.delay)
        return self.query

    async def embed_batch(self, texts, *, input_type="passage"):
        self.calls.append((input_type, list(texts)))
        await asyncio.sleep(self.delay)
        return self.passages


def _results(*titles: str) -> list[SearchResult]:
    return [SearchResult(title=t, url=f"https://{i}.example/", content=f"{t} snippet") for i, t in enumerate(titles)]


class RerankTests(unittest.IsolatedAsyncioTestCase):
    async def test_results_are_sorted_by_similarity_with_one_query_and_one_passage_call(self) -> None:
        embedder = _FakeEmbedder(query=[1.0, 0.0], passages=[[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
        results = _results("A", "B", "C")

        ranked = await SearchReranker(lambda: embedder).rerank("台北 天氣", results)

        self.assertEqual([r.title for r in ranked], ["B", "C", "A"])
        self.assertEqual(
            embedder.calls,
            [("query", "台北 天氣"), ("passage", ["A\nA snippet", "B\nB snippet", "C\nC snippet"])],
        )

    async def test_any_trouble_keeps_the_order(self) -> None:
        results = _results("A", "B")
        for name, embedder in (
            ("off", None),
            ("unconfigured", SimpleNamespace(is_configured=False)),
            ("no query vector", _FakeEmbedder(query=None, passages=[[1.0], [0.0]])),
            ("short batch", _FakeEmbedder(query=[1.0], passages=[[1.0]])),
            ("slow", _FakeEmbedder(query=[1.0], passages=[[0.0], [1.0]], delay=1.0)),
        ):
            with self.subTest(name):
                reranker = SearchReranker(lambda embedder=embedder: embedder, budget_seconds=0.05)
                self.assertEqual(await reranker.rerank("q", results), results)

        class _Broken(_FakeEmbedder):
            async def embed_batch(self, texts, *, input_type="passage"):
                raise RuntimeError("NIM down")

        self.assertEqual(await SearchReranker(lambda: _Broken(query=[1.0])).rerank("q", results), results)

    async def test_a_single_result_needs_no_embedding(self) -> None:
        embedder = _FakeEmbedder(query=[1.0], passages=[[1.0]])
        results = _results("A")
        self.assertEqual(await SearchReranker(lambda: embedder).rerank("q", results), results)
        self.assertEqual(embedder.calls, [])

    async def test_the_agent_reranks_before_reading_pages(self) -> None:
        order: list[str] = []

        class _Reranker:
            async def rerank(self, query, results):
                order.append("rerank")
                return list(reversed(results))

        class _Reader(_FakeReader):
            async def enrich(self, response, top_n, *, deadline=None):
                order.append("enrich")
                return response

        tavily = _Recorder(_json(_tavily_json(results=_TAIPEI)))
        chain = ChainedSearchBackend([TavilyBackend("k", transport=tavily.transport)])
        answer = _answer_chain()
        agent = WebSearchAgent(
            Settings(_env_file=None), answer, targets=[], search=chain, page_reader=_Reader(), reranker=_Reranker()
        )
        try:
            await agent.process(AgentRequest(text="台北天氣"))
        finally:
            await chain.close()

        self.assertEqual(order, ["rerank", "enrich"])
        context = answer.generate.await_args.kwargs["messages"][-2]["content"]
        self.assertLess(context.index("news.example"), context.index("zh.wikipedia.org"))


class SearchCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_chain_follows_the_settings(self) -> None:
        from src.runtime.composition import build_search_backend

        cases = (
            # Nothing set: no live backend (the tests' default Settings too).
            ({}, []),
            ({"tavily_api_key": "tvly-k"}, [("tavily", "key")]),
            ({"tavily_keyless_enabled": True}, [("tavily", "keyless")]),
            # A key wins over keyless.
            ({"tavily_api_key": "tvly-k", "tavily_keyless_enabled": True}, [("tavily", "key")]),
        )
        for settings, expected in cases:
            with self.subTest(settings=settings):
                chain = build_search_backend(Settings(_env_file=None, **settings))
                try:
                    self.assertEqual(
                        [(b.name, getattr(b, "mode", None)) for b in chain.backends], expected
                    )
                    self.assertEqual(chain.is_configured, bool(expected))
                finally:
                    await chain.close()

    async def test_a_v2_env_with_an_empty_tavily_key_keeps_search_off(self) -> None:
        # v2's .env.example said "keep TAVILY_API_KEY empty to disable web
        # search"; keyless Tavily must not switch it back on after the upgrade.
        from src.runtime.composition import build_search_backend

        with tempfile.TemporaryDirectory() as tmp:
            v2 = Path(tmp) / "v2.env"
            v2.write_text("TAVILY_API_KEY=\nWEB_SEARCH_MONTHLY_QUOTA=1000\n", encoding="utf-8")
            v3 = Path(tmp) / "v3.env"
            v3.write_text("TAVILY_API_KEY=\nTAVILY_KEYLESS_ENABLED=true\n", encoding="utf-8")
            cases = ((v2, []), (v3, [("tavily", "keyless")]))
            for env_file, expected in cases:
                with self.subTest(env_file=env_file.name), patch.dict(os.environ, {}, clear=True):
                    chain = build_search_backend(Settings(_env_file=env_file))
                    try:
                        self.assertEqual([(b.name, b.mode) for b in chain.backends], expected)
                    finally:
                        await chain.close()

    async def test_the_container_shares_one_chain_and_reader(self) -> None:
        from src.runtime.composition import build_container, wire_tool_registry
        from src.services.embedding_service import get_embedding_service

        settings = Settings(
            _env_file=None,
            openrouter_api_key="k",
            tavily_api_key="tvly-k",
            search_fetch_max_bytes=12345,
            search_fetch_budget_seconds=4.0,
        )
        container = build_container(settings)
        try:
            agent = container.web_search_agent
            self.assertIsInstance(container.search, ChainedSearchBackend)
            self.assertIs(agent.search, container.search)
            self.assertIs(agent.page_reader, container.page_reader)
            self.assertIsNone(agent.reranker)
            self.assertEqual(container.page_reader.fetcher.max_bytes, 12345)
            self.assertEqual(container.page_reader.budget_seconds, 4.0)

            # The chat agent's web_search tool searches through the same chain.
            wire_tool_registry(container, SimpleNamespace())
            container.search.search = AsyncMock(
                return_value=WebSearchResponse(
                    query="q", results=[SearchResult(title="t", url="https://a.example/", content="c")]
                )
            )
            tool = container.chat_agent.tool_registry.get("web_search")
            from src.agents.tools import ToolContext

            result = await tool.executor({"query": "台北天氣"}, ToolContext())
            self.assertEqual(result["status"], "ok")
            container.search.search.assert_awaited_once_with("台北天氣", max_results=5)
        finally:
            await container.aclose()

        reranked = build_container(Settings(_env_file=None, search_rerank_enabled=True))
        try:
            reranker = reranked.web_search_agent.reranker
            self.assertIsInstance(reranker, SearchReranker)
            self.assertIs(reranker._get_embedder, get_embedding_service)
        finally:
            await reranked.aclose()

    async def test_the_web_search_tool_is_offered_only_with_a_search_backend(self) -> None:
        """Regression (review finding 5): without Tavily the chat agent was
        still offered web_search, which could only answer unavailable."""
        from src.runtime.composition import build_container, wire_tool_registry

        for search, offered in (
            ({"tavily_api_key": "tvly-k"}, True),
            ({"tavily_keyless_enabled": True}, True),
            ({"tavily_api_key": "", "tavily_keyless_enabled": False}, False),
        ):
            with self.subTest(search=search), patch.dict(os.environ, {}, clear=True):
                container = build_container(Settings(_env_file=None, openrouter_api_key="k", **search))
                try:
                    wire_tool_registry(container, SimpleNamespace())
                    names = container.chat_agent.tool_registry.names()
                    self.assertEqual("web_search" in names, offered)
                    self.assertIn("recall_memory", names)
                    self.assertIn("update_user_profile", names)
                finally:
                    await container.aclose()

    async def test_closing_the_container_closes_search_and_reader(self) -> None:
        from src.runtime.container import AppContainer

        search, reader = SimpleNamespace(close=AsyncMock()), SimpleNamespace(close=AsyncMock())
        await AppContainer(model_health=None, fallback_chain=None, search=search, page_reader=reader).aclose()
        search.close.assert_awaited_once()
        reader.close.assert_awaited_once()

    def test_settings_validate_the_search_values(self) -> None:
        from pydantic import ValidationError

        for bad in (
            {"search_fetch_top_n": -1},
            {"search_fetch_max_bytes": 0},
            {"search_fetch_budget_seconds": 0},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    Settings(_env_file=None, **bad)
        defaults = Settings(_env_file=None)
        self.assertEqual(
            (
                defaults.search_fetch_top_n,
                defaults.search_fetch_max_bytes,
                defaults.search_fetch_budget_seconds,
                defaults.search_rerank_enabled,
                defaults.tavily_keyless_enabled,
            ),
            (3, 2_000_000, 6.0, False, False),
        )
        self.assertNotIn("web_search_monthly_quota", Settings.model_fields)
        from src.config import legacy_config_warnings

        [note] = legacy_config_warnings(environ={"WEB_SEARCH_MONTHLY_QUOTA": "1000"}, env_file=None)
        self.assertIn("WEB_SEARCH_MONTHLY_QUOTA in the environment is no longer read", note)

    def test_an_old_env_with_the_removed_search_keys_gets_a_notice_not_a_crash(self) -> None:
        # Values the removed self-hosted backend's validators refused.
        from src.config import legacy_config_warnings

        removed = {
            "SEARXNG_BASE_URL": "10.138.0.10:8080",
            "SEARXNG_TIMEOUT_SECONDS": "3.0",
            "SEARXNG_TIMEOUT_LIMIT_SECONDS": "3.0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "old.env"
            env_file.write_text("".join(f"{key}={value}\n" for key, value in removed.items()), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=env_file)
                notes = legacy_config_warnings(environ={}, env_file=env_file)

        self.assertFalse([field for field in Settings.model_fields if field.upper() in removed])
        self.assertEqual(settings.search_fetch_top_n, 3)
        self.assertEqual(len(notes), len(removed))
        for key, note in zip(sorted(removed), notes, strict=True):
            self.assertIn(f"{key} in {env_file} is no longer read", note)
            self.assertIn("Tavily only", note)


class ResponseShapeTests(unittest.TestCase):
    def test_the_shapes_the_agent_and_tool_read(self) -> None:
        response = WebSearchResponse(query="today news", results=[], answer="summary only")
        self.assertEqual(response.to_context_text(), "Answer: summary only")
        self.assertFalse(response.is_empty)
        self.assertFalse(response.has_results)
        long = SearchResult(title="t", url="u", content="x", raw_content="句子。" * 2000)
        text = long.to_text(1)
        self.assertTrue(text.startswith("[1] t\nURL: u\n"))
        self.assertTrue(text.endswith("。"))
        self.assertLessEqual(len(text), 3000 + len("[1] t\nURL: u\n"))
        pages = ExtractResponse(results=[ExtractResult(url="u", content="y" * 20000)])
        self.assertEqual(pages.failed_urls, [])
        self.assertEqual(len(pages.to_context_text()), len("URL: u\n") + 10000)
        self.assertEqual(copy.copy(response).answer, "summary only")

    def test_long_page_text_keeps_most_of_the_limit(self) -> None:
        # Review: the cut went to the last 。 anywhere in the first 3000
        # characters, so a timetable whose only 。 ends its first line
        # (4717 characters read) reached the model as that one line.
        header = "[1] 高鐵時刻表\nURL: u\n"
        timetable = "以下是本週高鐵時刻表。\n" + "\n".join(
            f"車次 {100 + i}  台北 {6 + i // 12:02d}:{i * 5 % 60:02d} 開  左營 {8 + i // 12:02d}:{i * 5 % 60:02d} 到"
            for i in range(300)
        )
        snippet = "高鐵本週時刻表，台北到左營每日班次一覽，首班車 06:00 ..."
        text = SearchResult(title="高鐵時刻表", url="u", content=snippet, raw_content=timetable).to_text(1)
        body = text.removeprefix(header)
        self.assertGreaterEqual(len(body), RESULT_TEXT_CHARS * 4 // 5)
        self.assertLessEqual(len(body), RESULT_TEXT_CHARS)
        self.assertTrue(timetable.startswith(body) and timetable[len(body)] == "\n")  # whole lines
        # Nowhere to break near the limit: cut at the limit itself.
        prose = "Intro line\n" + "word " * 1000
        body = SearchResult(title="t", url="u", content="s", raw_content=prose).to_text(1).split("\n", 2)[2]
        self.assertEqual(body, prose[:RESULT_TEXT_CHARS].rstrip())


if __name__ == "__main__":
    unittest.main()
