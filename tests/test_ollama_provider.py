"""OllamaChatProvider: the /api/chat body, message shape, answers and errors."""

from __future__ import annotations

import asyncio
import copy
import json
import unittest
from unittest.mock import patch

import httpx

from src.models.agent_request import AgentRequest
from src.providers.base import (
    EmptyCompletionError,
    ModelGoneError,
    PayloadError,
    RateLimitError,
    SkipTierError,
    TransientError,
)
from src.providers.fallback_chain import FallbackChain, Tier
from src.providers.ollama_provider import (
    DEFAULT_MAX_TOKENS,
    URL_IMAGE_SKIP,
    OllamaChatProvider,
    build_chat_body,
    parse_chat_response,
    think_value,
    to_ollama_messages,
)
from src.providers.systemone import RouterGates, SystemOneRouter
from src.services.colab.endpoint import BREAKER_FAILURES, REQUEST_TIMEOUT
from src.services.colab.state import IDLE
from src.services.model_health import COOLDOWN, ModelHealth
from src.utils import request_trace
from tests.colab_fakes import (
    QWEN,
    ROUTER,
    TUNNEL,
    Clock,
    FakeTunnel,
    bearer,
    make_endpoint,
    ready_state,
)

_USER = [{"role": "user", "content": "你好"}]
# A quoted image from the message cache or GCS reaches the agents as a URL.
_URL_IMAGE = [
    {"role": "user", "content": [{"type": "text", "text": "看"}, {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}}]}
]
_RELOAD_OPTIONS = {"num_ctx", "draft_num_predict", "shift", "num_batch", "num_gpu"}


def _chat(content: str = "答", **extra) -> dict:
    return {"model": QWEN, "message": {"role": "assistant", "content": content, **extra}, "done": True, "done_reason": "stop"}


class ThinkValueTests(unittest.TestCase):
    def test_levels_pass_and_everything_else_is_false(self) -> None:
        cases = [
            ({"think": "low"}, False, "low"),
            ({"think": "medium"}, False, "medium"),
            ({"think": "xhigh"}, False, "xhigh"),
            ({"think": "false"}, False, False),
            ({"think": "low"}, True, False),  # the chain turned thinking off
            ({}, False, "low"),  # the tier sets none: COLAB_CHAT_THINK
            ({"think": "high"}, False, False),  # Ollama would run "high" as medium
            ({"think": True}, False, False),
        ]
        for options, disabled, expected in cases:
            with self.subTest(options=options, disabled=disabled):
                self.assertEqual(think_value(options, disable_thinking=disabled, default="low"), expected)


class ChatBodyTests(unittest.TestCase):
    def test_plain_body(self) -> None:
        body = build_chat_body(QWEN, _USER, think=False)

        self.assertEqual(
            body,
            {
                "model": QWEN,
                "messages": [{"role": "user", "content": "你好"}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "num_predict": DEFAULT_MAX_TOKENS,
                },
            },
        )

    def test_thinking_body_uses_thinking_sampling_and_adds_the_thinking_budget(self) -> None:
        # Thinking and the answer share num_predict, so the answer keeps
        # its max_tokens after the thinking budget, as on the API tiers.
        for level in ("low", "medium", "xhigh"):
            with self.subTest(level=level):
                body = build_chat_body(QWEN, _USER, think=level, max_tokens=1024, thinking_budget=2048)
                self.assertEqual(body["think"], level)
                self.assertEqual(
                    body["options"],
                    {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "num_predict": 1024 + 2048},
                )
        self.assertEqual(build_chat_body(QWEN, _USER, think=False, max_tokens=1024)["options"]["num_predict"], 1024)

    def test_tier_max_tokens_and_temperature_win(self) -> None:
        body = build_chat_body(
            QWEN, _USER, think="low", options={"max_tokens": 1000, "temperature": 0.3}, max_tokens=2048, thinking_budget=512
        )
        self.assertEqual((body["options"]["num_predict"], body["options"]["temperature"]), (1000 + 512, 0.3))

    def test_no_load_time_option_is_ever_sent(self) -> None:
        for think in (False, "low", "xhigh"):
            body = build_chat_body(
                QWEN, _USER, think=think, options={"max_tokens": 10, "json_mode": "json_schema"},
                tools=[{"type": "function", "function": {"name": "t"}}], json_schema={"type": "object"},
            )
            with self.subTest(think=think):
                self.assertFalse(_RELOAD_OPTIONS & set(body["options"]))
                self.assertFalse(_RELOAD_OPTIONS & set(body))
                self.assertNotIn(True, (body["think"],))

    def test_tools_are_offered_unless_tool_choice_is_none(self) -> None:
        tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]

        self.assertEqual(build_chat_body(QWEN, _USER, think=False, tools=tools)["tools"], tools)
        self.assertEqual(build_chat_body(QWEN, _USER, think=False, tools=tools, tool_choice="auto")["tools"], tools)
        self.assertNotIn("tools", build_chat_body(QWEN, _USER, think=False, tools=tools, tool_choice="none"))

    def test_format_only_for_a_json_schema_tier(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}

        held = build_chat_body(QWEN, _USER, think=False, options={"json_mode": "json_schema"}, json_schema=schema)
        prompt = build_chat_body(QWEN, _USER, think=False, options={"json_mode": "prompt"}, json_schema=schema)

        self.assertEqual(held["format"], schema)
        self.assertNotIn("format", prompt)
        self.assertNotIn("format", build_chat_body(QWEN, _USER, think=False, options={"json_mode": "json_schema"}))


class MessageShapeTests(unittest.TestCase):
    def test_one_leading_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "你是助理"},
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "好"},
            {"role": "system", "content": "補充規則"},
            {"role": "user", "content": "第二句"},
        ]

        converted = to_ollama_messages(QWEN, messages)

        self.assertEqual(converted[0], {"role": "system", "content": "你是助理\n\n補充規則"})
        self.assertEqual([m["role"] for m in converted], ["system", "user", "assistant", "user"])

    def test_data_uri_images_become_raw_base64_on_the_user_turn(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "這是什麼？"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/AAAA"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}},
                ],
            }
        ]
        original = copy.deepcopy(messages)

        [user] = to_ollama_messages(QWEN, messages)

        self.assertEqual(user, {"role": "user", "content": "這是什麼？", "images": ["/9j/AAAA", "iVBORw0K"]})
        self.assertEqual(messages, original)

    def test_an_image_that_is_not_inline_data_skips_the_tier(self) -> None:
        # A URL is skipped by design (the API vision tiers fetch it), and
        # says so in the summary line; anything else is not inline data.
        for url, detail in (
            ("https://storage.example/quoted.jpg", URL_IMAGE_SKIP),
            ("http://storage.example/quoted.jpg", URL_IMAGE_SKIP),
            ("data:image/png,rawbytes", "image is not inline base64 data"),
            ("", "image is not inline base64 data"),
        ):
            messages = [{"role": "user", "content": [{"type": "text", "text": "看"}, {"type": "image_url", "image_url": {"url": url}}]}]
            with self.subTest(url=url), self.assertRaises(SkipTierError) as caught:
                to_ollama_messages(QWEN, messages)
            self.assertEqual(caught.exception.detail, detail)
        self.assertEqual(URL_IMAGE_SKIP, "by design: url image")

    def test_tool_calls_and_results(self) -> None:
        messages = [
            {"role": "user", "content": "現在幾點"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "never replayed",
                "tool_calls": [
                    {"id": "call_ab12cd34", "type": "function", "function": {"name": "get_time", "arguments": '{"tz": "Asia/Taipei"}'}},
                    {"id": "call_x", "type": "function", "function": {"name": "broken", "arguments": "{not json"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_ab12cd34", "name": "get_time", "content": '{"now": "15:04"}'},
            {"role": "tool", "tool_call_id": "call_x", "content": {"status": "error"}},
        ]

        converted = to_ollama_messages(QWEN, messages)

        self.assertEqual(
            converted[1],
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_ab12cd34", "function": {"name": "get_time", "arguments": {"tz": "Asia/Taipei"}}},
                    {"id": "call_x", "function": {"name": "broken", "arguments": {}}},
                ],
            },
        )
        self.assertEqual(
            converted[2],
            {"role": "tool", "tool_name": "get_time", "tool_call_id": "call_ab12cd34", "content": '{"now": "15:04"}'},
        )
        # The name comes from the assistant's call when the tool turn has none.
        self.assertEqual(converted[3]["tool_name"], "broken")
        self.assertEqual(json.loads(converted[3]["content"]), {"status": "error"})
        self.assertNotIn("reasoning_content", json.dumps(converted))

    def test_consecutive_user_turns_merge(self) -> None:
        messages = [{"role": "user", "content": "【引用】"}, {"role": "user", "content": "問題"}]
        self.assertEqual(to_ollama_messages(QWEN, messages), [{"role": "user", "content": "【引用】\n\n問題"}])

    def test_a_conversation_without_a_user_turn_skips_the_tier(self) -> None:
        with self.assertRaises(SkipTierError):
            to_ollama_messages(QWEN, [{"role": "system", "content": "只有系統"}])


class ChatAnswerTests(unittest.TestCase):
    def test_content_thinking_and_usage(self) -> None:
        data = _chat("答案", thinking="先想一下")
        data.update({"prompt_eval_count": 120, "eval_count": 30, "prompt_eval_cached_count": 100})

        response = parse_chat_response(data, model=QWEN)

        self.assertEqual((response.text, response.reasoning_content), ("答案", "先想一下"))
        self.assertEqual((response.finish_reason, response.provider, response.model), ("stop", "colab", QWEN))
        self.assertEqual(response.usage, {"prompt_tokens": 120, "completion_tokens": 30, "prompt_cached_tokens": 100})

    def test_tool_calls_keep_their_ids_and_get_json_arguments(self) -> None:
        data = _chat(
            "",
            tool_calls=[
                {"id": "call_q1w2e3r4", "function": {"index": 0, "name": "web_search", "arguments": {"query": "台北 天氣"}}},
                {"function": {"name": "noop", "arguments": {}}},
            ],
        )

        response = parse_chat_response(data, model=QWEN)

        self.assertEqual(
            response.tool_calls,
            [
                {"id": "call_q1w2e3r4", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "台北 天氣"}'}},
                {"id": "call_1", "type": "function", "function": {"name": "noop", "arguments": "{}"}},
            ],
        )

    def test_inline_think_markup_never_reaches_the_text(self) -> None:
        response = parse_chat_response(_chat("<think>內部</think>答案"), model=QWEN)
        self.assertEqual((response.text, response.reasoning_content), ("答案", "內部"))


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def _generate(self, tunnel: FakeTunnel, **kwargs):
        endpoint, _ = make_endpoint(tunnel=tunnel)
        provider = OllamaChatProvider(endpoint, default_think="low")
        try:
            return await provider.generate(QWEN, _USER, **kwargs), endpoint
        finally:
            await endpoint.close()

    async def test_a_ready_session_answers_through_the_tunnel_with_the_bearer(self) -> None:
        tunnel = FakeTunnel()

        response, endpoint = await self._generate(tunnel, options={"think": "low", "thinking": True})

        self.assertEqual(response.text, "Colab 的回答")
        [request] = tunnel.requests
        self.assertEqual(str(request.url), f"{TUNNEL}/api/chat")
        self.assertEqual(request.headers["authorization"], f"Bearer {bearer()}")
        body = json.loads(request.content)
        self.assertEqual((body["think"], body["stream"], body["keep_alive"]), ("low", False, -1))
        self.assertTrue(endpoint.is_ready())

    async def test_num_predict_is_the_calls_max_tokens_plus_the_thinking_budget(self) -> None:
        tunnel = FakeTunnel()
        endpoint, _ = make_endpoint(tunnel=tunnel)
        provider = OllamaChatProvider(endpoint, default_think="low", thinking_budget=3000)

        await provider.generate(QWEN, _USER, max_tokens=1024, options={"think": "medium", "thinking": True})
        await provider.generate(QWEN, _USER, max_tokens=1024, disable_thinking=True)

        self.assertEqual([body["options"]["num_predict"] for body in tunnel.bodies("POST /api/chat")], [4024, 1024])
        await endpoint.close()

    async def test_a_thinking_answer_cut_at_num_predict_is_no_answer(self) -> None:
        # The thinking used up most of num_predict: a few words of answer
        # must not reach the user as the reply; the next tier answers.
        cut = _chat("所以根號2是無理數。接下來推廣到", thinking="嗯" * 50)
        cut["done_reason"] = "length"
        tunnel = FakeTunnel(**{"POST /api/chat": (200, cut)})

        with self.assertRaises(EmptyCompletionError) as caught:
            await self._generate(tunnel, options={"think": "medium", "thinking": True})
        self.assertEqual(caught.exception.finish_reason, "length")

        # Without thinking the answer had its whole max_tokens, as on the
        # API tiers, and is kept.
        response, _ = await self._generate(tunnel, disable_thinking=True)
        self.assertEqual((response.text, response.finish_reason), ("所以根號2是無理數。接下來推廣到", "length"))

    async def test_the_chain_turning_thinking_off_sends_think_false(self) -> None:
        tunnel = FakeTunnel()

        await self._generate(tunnel, options={"think": "low", "thinking": False}, disable_thinking=True)

        self.assertIs(tunnel.bodies("POST /api/chat")[0]["think"], False)

    async def test_not_ready_skips_without_a_request(self) -> None:
        clock = Clock()
        for state in (
            ready_state(clock(), status=IDLE),
            ready_state(clock(), tunnel_url=""),
            ready_state(clock(), last_health_at=clock() - 121),
        ):
            tunnel = FakeTunnel()
            endpoint, _ = make_endpoint(state, tunnel, clock=clock)
            with self.subTest(state=state.status), self.assertRaisesRegex(SkipTierError, "not READY"):
                await OllamaChatProvider(endpoint).generate(QWEN, _USER)
            self.assertEqual(tunnel.requests, [])
            await endpoint.close()

    async def test_errors_map_onto_provider_errors(self) -> None:
        cases = [
            # A refused bearer is this session's problem, not an API key's.
            (401, "unauthorized", TransientError, True),
            (403, "bad token", TransientError, True),
            (404, '{"error":"model not found"}', TransientError, True),
            (410, "gone", TransientError, True),
            (499, "client closed", TransientError, False),
            (503, "server busy, please try again.  maximum pending requests exceeded", TransientError, False),
            (524, "timeout", TransientError, False),
            (400, '{"error":"bad"}', PayloadError, False),
            (429, "too many", RateLimitError, False),
        ]
        for status, body, error, unhealthy in cases:
            tunnel = FakeTunnel(**{"POST /api/chat": (status, body)})
            endpoint, _ = make_endpoint(tunnel=tunnel)
            with self.subTest(status=status):
                with self.assertRaises(error) as caught:
                    await OllamaChatProvider(endpoint).generate(QWEN, _USER)
                self.assertNotIsInstance(caught.exception, ModelGoneError)
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(endpoint.is_ready(), not unhealthy)
                self.assertFalse(endpoint.semaphore(QWEN).locked())
            await endpoint.close()

    async def test_an_empty_403_is_ollamas_host_check_and_logged_as_an_error(self) -> None:
        tunnel = FakeTunnel(**{"POST /api/chat": httpx.Response(403)})
        endpoint, _ = make_endpoint(tunnel=tunnel)

        with patch("src.providers.ollama_provider.logger.error") as log_error, self.assertRaises(TransientError):
            await OllamaChatProvider(endpoint).generate(QWEN, _USER)

        self.assertIn("Host", log_error.call_args.args[0])
        self.assertFalse(endpoint.is_ready())
        await endpoint.close()

    async def test_transport_failures_and_bad_bodies_are_transient(self) -> None:
        cases = {
            "connect": httpx.ConnectError("refused"),
            "read timeout": httpx.ReadTimeout("slow"),
            "not json": (200, b"<html>"),
            "not an object": (200, b"[1, 2]"),
        }
        for name, answer in cases.items():
            endpoint, _ = make_endpoint(tunnel=FakeTunnel(**{"POST /api/chat": answer}))
            with self.subTest(name), self.assertRaises(TransientError):
                await OllamaChatProvider(endpoint).generate(QWEN, _USER)
            self.assertFalse(endpoint.semaphore(QWEN).locked())
            await endpoint.close()

    async def test_length_with_nothing_said_is_an_empty_completion(self) -> None:
        data = _chat("", thinking="想太久")
        data["done_reason"] = "length"
        endpoint, _ = make_endpoint(tunnel=FakeTunnel(**{"POST /api/chat": (200, data)}))

        with self.assertRaises(EmptyCompletionError) as caught:
            await OllamaChatProvider(endpoint).generate(QWEN, _USER)

        self.assertEqual(caught.exception.finish_reason, "length")
        await endpoint.close()

    async def test_a_quoted_image_url_skips_by_design_before_any_request(self) -> None:
        # Checked before the session: with Colab down the reason is still
        # the URL, so the summary line never reads it as Colab's failure.
        clock = Clock()
        for state in (ready_state(clock()), ready_state(clock(), status=IDLE)):
            tunnel = FakeTunnel()
            endpoint, _ = make_endpoint(state, tunnel, clock=clock)
            with self.subTest(state=state.status):
                with self.assertRaises(SkipTierError) as caught:
                    await OllamaChatProvider(endpoint).generate(QWEN, _URL_IMAGE)
                self.assertEqual(caught.exception.detail, URL_IMAGE_SKIP)
                self.assertEqual(tunnel.requests, [])
                self.assertFalse(endpoint.semaphore(QWEN).locked())
                self.assertEqual(endpoint.last_error, "")
            await endpoint.close()

    async def test_a_cancelled_call_frees_the_model_and_is_not_counted(self) -> None:
        # A time limit, the turn's deadline or a shutdown cancelling the call
        # says nothing about the tunnel: the chain judges its own timeouts.
        tunnel = FakeTunnel(**{"POST /api/chat": ("hang", 5)})
        endpoint, _ = make_endpoint(tunnel=tunnel)
        provider = OllamaChatProvider(endpoint)

        for _ in range(BREAKER_FAILURES + 1):
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await provider.generate(QWEN, _USER)
            self.assertFalse(endpoint.semaphore(QWEN).locked())

        self.assertFalse(endpoint.breaker_open())
        self.assertEqual(endpoint.last_error, "")
        self.assertEqual(len(tunnel.requests), BREAKER_FAILURES + 1)
        await endpoint.close()

    def test_the_client_gives_up_before_cloudflares_edge_timeout(self) -> None:
        self.assertLess(REQUEST_TIMEOUT.read, 125)
        self.assertLessEqual(REQUEST_TIMEOUT.connect, 10)


class ChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_colab_404_is_not_the_end_of_life_breaker(self) -> None:
        health = ModelHealth()
        endpoint, _ = make_endpoint(tunnel=FakeTunnel(**{"POST /api/chat": (404, "model not found")}))

        class Api:
            name = "nvidia"

            async def generate(self, model, messages, **kwargs):
                from src.providers.base import ProviderResponse

                return ProviderResponse(text="API 答案", model=model)

        response = await FallbackChain(health).generate(
            [Tier(OllamaChatProvider(endpoint), QWEN, shares_time=True), Tier(Api(), "ultra")], _USER
        )

        self.assertEqual(response.text, "API 答案")
        self.assertFalse(health.is_dead("colab", QWEN))
        await endpoint.close()

    async def test_a_url_image_goes_to_the_api_tiers_as_a_skip_not_a_failure(self) -> None:
        health = ModelHealth()
        tunnel = FakeTunnel()
        endpoint, _ = make_endpoint(tunnel=tunnel)
        trace = request_trace.RequestTrace()
        token = request_trace.activate(trace)
        try:
            with request_trace.phase(request_trace.ANSWER):
                response = await FallbackChain(health).generate(
                    [Tier(OllamaChatProvider(endpoint), QWEN, shares_time=True), Tier(_Api(), "muse")], _URL_IMAGE
                )
        finally:
            request_trace.deactivate(token)

        self.assertEqual(response.text, "API 答案")
        [walk] = trace.walks
        self.assertEqual(
            [(attempt.label, attempt.outcome) for attempt in walk.attempts],
            [(f"colab:{QWEN}", "skip(by design: url image)"), ("nvidia:muse", "ok")],
        )
        self.assertIn(f"colab:{QWEN} skip(by design: url image)", request_trace.summary_line(trace))
        self.assertEqual(tunnel.requests, [])
        self.assertIsNone(health.unavailable("colab", QWEN))
        self.assertFalse(endpoint.breaker_open())
        await endpoint.close()

    async def test_a_refused_bearer_waits_for_a_health_check_not_a_provider_cooldown(self) -> None:
        # One 401 (say a named tunnel whose old connector still answers) must
        # not take every Colab model out for ModelHealth's 10-minute
        # API-key cooldown: the endpoint is unhealthy until the next
        # passing health check re-validates the bearer, then Colab chat
        # and the router answer again.
        for answer in ((401, "unauthorized"), (403, "bad token"), httpx.Response(403)):
            clock = Clock()
            health = ModelHealth(clock=clock)
            tunnel = FakeTunnel(**{"POST /api/chat": answer})
            endpoint, _ = make_endpoint(tunnel=tunnel, clock=clock)
            tiers = [Tier(OllamaChatProvider(endpoint), QWEN, shares_time=True), Tier(_Api(), "ultra")]
            router = SystemOneRouter(
                endpoint,
                Tier(OllamaChatProvider(endpoint), ROUTER, timeout_s=6),
                gates=RouterGates(confidence=0.75, think_on=0.85, think_off=0.15),
                health=health,
            )
            with self.subTest(answer=answer):
                self.assertEqual((await FallbackChain(health).generate(tiers, _USER)).text, "API 答案")
                self.assertIsNone(health.unavailable("colab", QWEN))
                self.assertIsNone(health.unavailable("colab", ROUTER))
                self.assertFalse(endpoint.is_ready())
                self.assertIsNone(await router.route(AgentRequest(text="你好")))

                clock.now += 60
                self.assertTrue(await endpoint.check_health())
                tunnel.routes["POST /api/chat"] = (200, _chat("Colab 的回答"))
                self.assertEqual((await FallbackChain(health).generate(tiers, _USER)).text, "Colab 的回答")
                self.assertIsNotNone(await router.route(AgentRequest(text="你好")))
                self.assertEqual(tunnel.paths().count("POST /v1/systemone"), 1)
            await endpoint.close()


    async def test_thinking_tier_timeouts_never_open_the_breaker(self) -> None:
        # A request that needs thinking (disable_thinking=False): a long
        # answer that runs out of time is slow by design, not a broken
        # tunnel, so neither ModelHealth nor the breaker the router shares
        # holds it against Colab.
        health = ModelHealth()
        tunnel = FakeTunnel(**{"POST /api/chat": ("hang", 5)})
        endpoint, _ = make_endpoint(tunnel=tunnel)
        thinking = {"think": "low", "thinking": True}
        tiers = [
            Tier(OllamaChatProvider(endpoint), QWEN, timeout_s=0.05, options=thinking, shares_time=True),
            Tier(_Api(), "ultra"),
        ]

        for _ in range(BREAKER_FAILURES + 1):
            self.assertEqual((await FallbackChain(health).generate(tiers, _USER)).text, "API 答案")

        self.assertEqual(len(tunnel.requests), BREAKER_FAILURES + 1)
        self.assertFalse(endpoint.breaker_open())
        self.assertIsNone(health.unavailable("colab", QWEN))
        await endpoint.close()

    async def test_countable_timeouts_cool_the_model_down(self) -> None:
        # Thinking off and its full time limit spent: the chain counts it,
        # and two in a row cool qwen3.8 down (ModelHealth), not the router.
        health = ModelHealth()
        tunnel = FakeTunnel(**{"POST /api/chat": ("hang", 5)})
        endpoint, _ = make_endpoint(tunnel=tunnel)
        tiers = [
            Tier(OllamaChatProvider(endpoint), QWEN, timeout_s=0.05, options={"thinking": False}, shares_time=True),
            Tier(_Api(), "ultra"),
        ]

        for _ in range(3):
            self.assertEqual((await FallbackChain(health).generate(tiers, _USER)).text, "API 答案")

        self.assertEqual(len(tunnel.requests), 2)
        self.assertEqual(health.unavailable("colab", QWEN).kind, COOLDOWN)
        self.assertIsNone(health.unavailable("colab", ROUTER))
        self.assertFalse(endpoint.breaker_open())
        await endpoint.close()


class _Api:
    name = "nvidia"

    async def generate(self, model, messages, **kwargs):
        from src.providers.base import ProviderResponse

        return ProviderResponse(text="API 答案", model=model)


if __name__ == "__main__":
    unittest.main()
