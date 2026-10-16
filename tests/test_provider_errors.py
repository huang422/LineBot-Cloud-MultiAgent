"""HTTP status -> provider error mapping, NVCF 202 polling and reply parsing.

Also a 200 whose body is an ``error`` object, mapped by its ``code``.

Every request here goes through ``httpx.MockTransport``; nothing is sent.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from src.providers import openrouter_provider
from src.providers.base import (
    AuthError,
    CreditsError,
    EmptyCompletionError,
    ModelGoneError,
    PayloadError,
    ProviderError,
    ProviderResponse,
    RateLimitError,
    SkipTierError,
    TransientError,
    is_empty_completion,
    parse_openai_response,
    require_output,
)
from src.providers.fallback_chain import AllProvidersFailedError, FallbackChain, Tier
from src.providers.nvidia_provider import NvidiaProvider
from src.providers.openai_compat import (
    NVCF_STATUS_URL,
    error_for_status,
    parse_retry_after,
    post_chat_completion,
)
from src.providers.openrouter_provider import OpenRouterProvider
from src.services.model_health import COOLDOWN, ModelHealth

_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
_MUSE = "meta/muse-glimmer-30b"
_EOL_BODY = (
    '{"type":"about:blank","title":"Gone","status":410,"detail":"The model '
    "'qwen/qwen3.5-397b-a17b' has reached its end of life on 2026-07-27T00:00:00Z "
    'and is no longer available."}'
)
_OR_URL = "https://openrouter.ai/api/v1/chat/completions"
_OR_ULTRA = "nvidia/nemotron-3-ultra-550b-a55b:free"
# What OpenRouter Ultra :free answered with HTTP 200 on 2026-10-16.
_INCIDENT_BODY = {
    "id": "gen-1792113316-6IrPtyqon7MXPdBZ9Yt3",
    "error": {
        "message": "Upstream error from Nvidia: Service temporarily overloaded",
        "code": 503,
        "metadata": {"error_type": "provider_overloaded"},
    },
}


def _chat(content: str | None = "ok", *, finish_reason: str | None = "stop", **message) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": finish_reason}],
        "model": _ULTRA,
    }


class _Recorder:
    """MockTransport handler that replays scripted responses in order."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response


async def _post(
    *responses: httpx.Response | Exception,
    timeout: float = 5.0,
    provider: str = "nvidia",
    model: str = _ULTRA,
) -> tuple[dict, _Recorder]:
    recorder = _Recorder(*responses)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as client:
        data = await post_chat_completion(
            client,
            _URL if provider == "nvidia" else _OR_URL,
            {"model": model},
            provider=provider,
            model=model,
            timeout=timeout,
            poll_interval=0.0,
        )
    return data, recorder


async def _body_error(error: dict, **body) -> ProviderError:
    """The error an OpenRouter 200 whose body holds *error* raises."""
    try:
        await _post(httpx.Response(200, json={"error": error, **body}), provider="openrouter", model=_OR_ULTRA)
    except ProviderError as exc:
        return exc
    raise AssertionError("no error raised")


class ErrorHierarchyTests(unittest.TestCase):
    def test_every_error_is_a_provider_error_with_model_status_and_provider(self) -> None:
        errors = [
            RateLimitError("m", 30, provider="nvidia"),
            PayloadError("m", 422, "bad field", "nvidia"),
            AuthError("m", 403, "Authorization failed", "nvidia"),
            CreditsError("m", 402, "no credits", "openrouter"),
            ModelGoneError("m", 410, "gone", "nvidia", eol="2026-07-27T00:00:00Z"),
            TransientError("m", 503, "ResourceExhausted", "nvidia"),
            EmptyCompletionError("m", provider="nvidia"),
            SkipTierError("m", "busy", "colab"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.assertIsInstance(error, ProviderError)
                self.assertEqual(error.model, "m")
                self.assertTrue(error.provider)
                self.assertIn("m", str(error))
        self.assertEqual((errors[0].status, errors[0].retry_after), (429, 30))
        self.assertEqual(errors[4].eol, "2026-07-27T00:00:00Z")
        self.assertEqual(errors[6].status, 200)
        self.assertEqual(errors[7].status, 0)


class StatusMappingTests(unittest.TestCase):
    def _error(self, status: int, text: str = "detail", headers: dict | None = None) -> ProviderError:
        response = httpx.Response(status, text=text, headers=headers or {})
        return error_for_status(response, provider="nvidia", model=_ULTRA)

    def test_every_status_maps_to_its_error_class(self) -> None:
        cases = {
            400: PayloadError,
            413: PayloadError,
            422: PayloadError,
            401: AuthError,
            403: AuthError,
            402: CreditsError,
            404: ModelGoneError,
            410: ModelGoneError,
            429: RateLimitError,
            408: TransientError,
            500: TransientError,
            502: TransientError,
            503: TransientError,
            524: TransientError,  # Cloudflare: origin did not answer in time
        }
        for status, error_class in cases.items():
            with self.subTest(status=status):
                error = self._error(status)
                self.assertIs(type(error), error_class)
                self.assertEqual((error.status, error.provider, error.model), (status, "nvidia", _ULTRA))

    def test_410_parses_the_end_of_life_date(self) -> None:
        error = self._error(410, _EOL_BODY)
        self.assertEqual(error.eol, "2026-07-27T00:00:00Z")
        self.assertIsNone(self._error(404, "Model does not exist by the name of x").eol)

    def test_a_404_for_what_this_request_needs_is_the_requests_problem(self) -> None:
        # OpenRouter: no endpoint of the model takes tools (or images) is not
        # the model gone; the same model still serves requests without them.
        for text in (
            '{"error":{"code":404,"message":"No endpoints found that support tool use. To learn more ..."}}',
            '{"error":{"code":404,"message":"No endpoints found that support image input"}}',
        ):
            with self.subTest(text=text[30:70]):
                self.assertIs(type(self._error(404, text)), PayloadError)
        gone = self._error(404, '{"error":{"message":"No endpoints found for qwen/qwen3.8-27b:free."}}')
        self.assertIs(type(gone), ModelGoneError)

    def test_429_reads_a_numeric_retry_after(self) -> None:
        self.assertEqual(self._error(429, headers={"Retry-After": "12"}).retry_after, 12)
        self.assertIsNone(self._error(429).retry_after)
        self.assertIsNone(
            self._error(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}).retry_after
        )

    def test_parse_retry_after(self) -> None:
        self.assertEqual(parse_retry_after("2.5"), 2)
        self.assertEqual(parse_retry_after("0"), 1)
        self.assertIsNone(parse_retry_after(""))
        self.assertIsNone(parse_retry_after(None))

    def test_detail_is_truncated(self) -> None:
        self.assertEqual(len(self._error(500, "x" * 5000).detail), 500)


class PostChatCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_200_returns_the_json_body(self) -> None:
        data, recorder = await _post(httpx.Response(200, json=_chat("你好")))

        self.assertEqual(data["choices"][0]["message"]["content"], "你好")
        self.assertEqual(json.loads(recorder.requests[0].content), {"model": _ULTRA})

    async def test_bad_json_is_transient(self) -> None:
        for response in (
            httpx.Response(200, text="<html>gateway</html>"),
            httpx.Response(200, json=["not", "an", "object"]),
        ):
            with self.subTest(body=response.text[:10]), self.assertRaises(TransientError) as ctx:
                await _post(response)
            self.assertEqual(ctx.exception.status, 200)

    async def test_transport_errors_and_timeouts_are_transient(self) -> None:
        for exc, text in (
            (httpx.ConnectError("connection refused"), "connection refused"),
            (httpx.ReadTimeout("read timed out"), "timed out"),
        ):
            with self.subTest(exc=type(exc).__name__), self.assertRaises(TransientError) as ctx:
                await _post(exc)
            self.assertEqual(ctx.exception.status, 0)
            self.assertIn(text, ctx.exception.detail)

    async def test_error_statuses_raise_the_mapped_error(self) -> None:
        with self.assertRaises(ModelGoneError) as ctx:
            await _post(httpx.Response(410, text=_EOL_BODY))
        self.assertEqual(ctx.exception.eol, "2026-07-27T00:00:00Z")

        with self.assertRaises(TransientError) as ctx:
            await _post(httpx.Response(503, text='{"detail":"ResourceExhausted"}'))
        self.assertEqual(ctx.exception.status, 503)

    async def test_202_is_polled_until_the_result_arrives(self) -> None:
        data, recorder = await _post(
            httpx.Response(202, headers={"NVCF-REQID": "req-1", "NVCF-STATUS": "pending-evaluation"}),
            httpx.Response(202, headers={"NVCF-REQID": "req-1"}),
            httpx.Response(200, json=_chat("done")),
        )

        self.assertEqual(data["choices"][0]["message"]["content"], "done")
        self.assertEqual([request.method for request in recorder.requests], ["POST", "GET", "GET"])
        for request in recorder.requests[1:]:
            self.assertEqual(str(request.url), NVCF_STATUS_URL.format(request_id="req-1"))

    async def test_a_polled_error_is_mapped_too(self) -> None:
        with self.assertRaises(RateLimitError):
            await _post(
                httpx.Response(202, headers={"NVCF-REQID": "req-2"}),
                httpx.Response(429, headers={"Retry-After": "5"}),
            )

    async def test_a_status_url_that_does_not_know_the_job_is_transient(self) -> None:
        # The job was accepted (202), so a 404 from the status URL is about
        # the job id, never the model: no ModelGoneError and its dead mark.
        with self.assertRaises(TransientError) as ctx:
            await _post(
                httpx.Response(202, headers={"NVCF-REQID": "req-4"}),
                httpx.Response(404, json={"status": 404, "title": "Not Found", "detail": "Not Found"}),
            )

        self.assertEqual(ctx.exception.status, 404)
        self.assertIn("req-4", ctx.exception.detail)

    async def test_202_without_a_request_id_is_transient(self) -> None:
        with self.assertRaises(TransientError) as ctx:
            await _post(httpx.Response(202))
        self.assertIn("NVCF-REQID", ctx.exception.detail)

    async def test_polling_stops_at_the_timeout(self) -> None:
        with self.assertRaises(TransientError) as ctx:
            await _post(httpx.Response(202, headers={"NVCF-REQID": "req-3"}), timeout=0.05)

        self.assertEqual(ctx.exception.status, 202)
        self.assertIn("req-3 still pending", ctx.exception.detail)

    # ── A 200 whose body is an error object (OpenRouter after routing) ──

    async def test_the_incident_body_is_the_503_it_names(self) -> None:
        # 2026-10-16: read as an empty completion, it earned no strike.
        with self.assertRaises(TransientError) as ctx:
            await _post(httpx.Response(200, json=_INCIDENT_BODY), provider="openrouter", model=_OR_ULTRA)

        error = ctx.exception
        self.assertIs(type(error), TransientError)
        self.assertEqual((error.status, error.provider, error.model), (503, "openrouter", _OR_ULTRA))
        self.assertEqual(
            json.loads(error.detail),
            {"error": _INCIDENT_BODY["error"]},
        )

    async def test_a_body_429_from_one_upstream_cools_only_that_model(self) -> None:
        # metadata.raw carries the "upstream" that tells it from the account
        # cap; it leads the detail even behind a long field, past 500 chars.
        error = await _body_error(
            {
                "message": "Provider returned error",
                "code": 429,
                "metadata": {
                    "headers": {"X-Padding": "x" * 600},
                    "raw": f"{_OR_ULTRA} is temporarily rate-limited upstream. Please retry shortly",
                    "provider_name": "Nvidia",
                },
            }
        )

        self.assertIs(type(error), RateLimitError)
        self.assertEqual((error.status, error.retry_after), (429, None))
        self.assertEqual(len(error.detail), 500)
        self.assertIn("rate-limited upstream", error.detail)
        health = ModelHealth(clock=lambda: 1_000.0)
        health.record_failure("openrouter", _OR_ULTRA, error)
        self.assertFalse(health.is_available("openrouter", _OR_ULTRA))
        self.assertTrue(health.is_available("openrouter", "dots-studio/dots-3-note-preview:free"))

    async def test_a_body_429_for_the_accounts_free_cap_cools_every_free_id(self) -> None:
        error = await _body_error(
            {
                "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
                "code": 429,
                "metadata": {"headers": {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0"}},
            }
        )

        self.assertIs(type(error), RateLimitError)
        health = ModelHealth(clock=lambda: 1_000.0)
        health.record_failure("openrouter", _OR_ULTRA, error)
        self.assertFalse(health.is_available("openrouter", "dots-studio/dots-3-note-preview:free"))

    async def test_body_codes_map_like_the_statuses(self) -> None:
        cases = (
            ({"code": 404, "message": "No endpoints found that support tool use. To learn more ..."}, PayloadError, 404),
            ({"code": 404, "message": f"No endpoints found for {_OR_ULTRA}."}, ModelGoneError, 404),
            ({"code": 410, "message": "The model has reached its end of life on 2026-07-27T00:00:00Z"}, ModelGoneError, 410),
            ({"code": 400, "message": "Invalid request"}, PayloadError, 400),
            ({"code": 401, "message": "User not found."}, AuthError, 401),
            ({"code": 402, "message": "Insufficient credits"}, CreditsError, 402),
            ({"code": 502, "message": "Upstream error"}, TransientError, 502),
            ({"code": "503", "message": "a numeric string is still a status"}, TransientError, 503),
        )
        for error, error_class, status in cases:
            with self.subTest(code=error["code"], message=error["message"][:30]):
                raised = await _body_error(error)
                self.assertIs(type(raised), error_class)
                self.assertEqual(raised.status, status)
                self.assertEqual(raised.provider, "openrouter")
        gone = await _body_error(cases[2][0])
        self.assertEqual(gone.eol, "2026-07-27T00:00:00Z")

    async def test_a_body_error_without_a_usable_code_is_transient_200(self) -> None:
        for error in (
            {"message": "Internal error"},
            {"message": "OpenAI style", "code": "rate_limit_exceeded"},
            {"message": "null code", "code": None},
            {"message": "not an error status", "code": 200},
            {"message": "a bool is no status", "code": True},
            # str.isdigit() passes it, int() does not.
            {"message": "a superscript is no number", "code": "²"},
            {"message": "past int()'s digit limit", "code": "5" * 5000},
        ):
            with self.subTest(error=error):
                raised = await _body_error(error)
                self.assertIs(type(raised), TransientError)
                self.assertEqual(raised.status, 200)
                self.assertIn(error["message"], raised.detail)

    async def test_a_body_error_beside_partial_output_still_raises(self) -> None:
        # OpenRouter's errors doc: a stream cut short ends with the error
        # and finish_reason "error"; the partial text is no answer.
        raised = await _body_error(
            {"message": "Upstream stream ended", "code": 502},
            choices=[{"message": {"role": "assistant", "content": "半句"}, "finish_reason": "error"}],
        )

        self.assertIs(type(raised), TransientError)
        self.assertEqual(raised.status, 502)

    async def test_an_error_on_the_choice_beside_partial_output_raises(self) -> None:
        # OpenRouter's schema: NonStreamingChoice.error, no top-level error.
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "台北今天的天氣"},
                    "finish_reason": "error",
                    "error": {"code": 502, "message": "Upstream stream ended"},
                }
            ]
        }
        with self.assertRaises(TransientError) as ctx:
            await _post(httpx.Response(200, json=body), provider="openrouter", model=_OR_ULTRA)

        self.assertEqual((ctx.exception.status, ctx.exception.provider), (502, "openrouter"))
        self.assertIn("Upstream stream ended", ctx.exception.detail)

        rate_limited = {"choices": [{**body["choices"][0], "error": {"code": 429, "message": "slow down"}}]}
        with self.assertRaises(RateLimitError):
            await _post(httpx.Response(200, json=rate_limited), provider="openrouter", model=_OR_ULTRA)

    async def test_finish_reason_error_without_an_error_object_is_transient_200(self) -> None:
        with self.assertRaises(TransientError) as ctx:
            await _post(httpx.Response(200, json=_chat("半句", finish_reason="error")))

        self.assertIs(type(ctx.exception), TransientError)
        self.assertEqual(ctx.exception.status, 200)
        self.assertIn('finish_reason "error"', ctx.exception.detail)

    async def test_the_detail_keeps_the_body_text_readable(self) -> None:
        raised = await _body_error({"message": "上游錯誤：服務暫時過載", "code": 503})

        self.assertIn("上游錯誤：服務暫時過載", raised.detail)
        self.assertTrue(raised.detail.startswith('{"error": {"message": '))

    async def test_a_null_error_is_no_error(self) -> None:
        body = {**_chat("你好"), "error": None}

        data, _ = await _post(httpx.Response(200, json=body), provider="openrouter", model=_OR_ULTRA)

        self.assertEqual(data, body)


class ParseOpenAIResponseTests(unittest.TestCase):
    def test_finish_reason_usage_and_provider_are_kept(self) -> None:
        response = parse_openai_response(
            {
                **_chat("答案", finish_reason="stop"),
                "usage": {"completion_tokens_details": {"reasoning_tokens": 37}},
            },
            model="fallback-id",
            provider="nvidia",
        )

        self.assertEqual(response.text, "答案")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.reasoning_tokens, 37)
        self.assertEqual(response.provider, "nvidia")
        # The body names the model that answered; the argument is a fallback.
        self.assertEqual(response.model, _ULTRA)
        self.assertEqual(parse_openai_response({"choices": []}, model="fallback-id").model, "fallback-id")
        self.assertEqual(
            parse_openai_response({"choices": [], "usage": {"reasoning_tokens": 5}}).reasoning_tokens, 5
        )

    def test_reasoning_sources_in_priority_order(self) -> None:
        cases = {
            "reasoning_content": (_chat("答案", reasoning_content="NIM"), "答案", "NIM"),
            "reasoning": (_chat("答案", reasoning="OpenRouter"), "答案", "OpenRouter"),
            "both fields": (_chat("答案", reasoning_content="NIM", reasoning="OR"), "答案", "NIM"),
            "think pair": (_chat("<think>想一想</think>答案"), "答案", "想一想"),
            "lone closing tag": (_chat("想一想\n</think>\n\n答案"), "答案", "想一想"),
            "channel block": (_chat("<|channel>thought\n看圖<channel|>紅色的圓形"), "紅色的圓形", "看圖"),
            "empty channel block": (_chat("<|channel>thought\n<channel|>你好"), "你好", None),
        }
        for name, (data, text, reasoning) in cases.items():
            with self.subTest(name):
                response = parse_openai_response(data)
                self.assertEqual(response.text, text)
                self.assertEqual(response.reasoning_content, reasoning)

    def test_think_tags_after_the_start_are_the_answers_own_text(self) -> None:
        # Review: every <think>…</think> pair was stripped and the answer cut
        # at any </think>, so an answer explaining the tags lost its words,
        # and one starting with "<think>" became empty (EmptyCompletionError).
        explains = "Qwen 會先輸出 <think>…</think> 再回答。"
        cases = {
            "pair in the middle": (_chat(explains), explains, None),
            "closer after a leading block": (
                _chat("<think>想一想</think>結尾標籤是 </think>。"),
                "結尾標籤是 </think>。",
                "想一想",
            ),
            "closer with reasoning in its own field": (
                _chat("Qwen 用 </think> 結束推理。", reasoning_content="NIM"),
                "Qwen 用 </think> 結束推理。",
                "NIM",
            ),
            "finished answer starting with <think>": (
                _chat("<think> 是 Qwen 的推理標籤。"),
                "<think> 是 Qwen 的推理標籤。",
                None,
            ),
            "stray closer with reasoning in its own field": (
                _chat("\n</think>\n\n答案", reasoning="OR"),
                "答案",
                "OR",
            ),
        }
        for name, (data, text, reasoning) in cases.items():
            with self.subTest(name):
                response = parse_openai_response(data)
                self.assertEqual(response.text, text)
                self.assertEqual(response.reasoning_content, reasoning)

    def test_reply_cut_off_mid_reasoning_is_empty(self) -> None:
        for content in ("<think>still thinking", "<|channel>thought\nstill thinking"):
            with self.subTest(content=content):
                response = parse_openai_response(_chat(content, finish_reason="length"))
                self.assertIsNone(response.text)
                self.assertTrue(is_empty_completion(response))
                with self.assertRaises(EmptyCompletionError) as ctx:
                    require_output(response)
                self.assertIn("finish_reason=length", ctx.exception.detail)

    def test_tool_calls_or_images_are_not_empty(self) -> None:
        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        for response in (
            ProviderResponse(tool_calls=[call]),
            ProviderResponse(images=["data:image/png;base64,abc"]),
            ProviderResponse(text="x"),
        ):
            with self.subTest(response=response):
                self.assertIs(require_output(response), response)
        self.assertTrue(is_empty_completion(ProviderResponse(text="")))


def _nvidia(handler, health: ModelHealth | None = None, **kwargs) -> NvidiaProvider:
    return NvidiaProvider("key", health or ModelHealth(), transport=httpx.MockTransport(handler), **kwargs)


def _openrouter(handler, health: ModelHealth | None = None) -> OpenRouterProvider:
    return OpenRouterProvider("key", health or ModelHealth(), transport=httpx.MockTransport(handler))


def _no_retry_pause():
    """OpenRouter's one retry of an unavailable upstream, without the pause."""
    return patch.object(openrouter_provider, "UPSTREAM_RETRY_PAUSE_SECONDS", 0.0)


class ProviderErrorContractTests(unittest.IsolatedAsyncioTestCase):
    """Both providers surface the same errors through the shared POST."""

    async def test_statuses_reach_callers_as_typed_errors_from_both_providers(self) -> None:
        cases = (
            (401, AuthError),
            (402, CreditsError),
            (404, ModelGoneError),
            (422, PayloadError),
            (429, RateLimitError),
            (503, TransientError),
            (524, TransientError),
        )
        for status, error_class in cases:
            for name, build, model in (
                ("nvidia", _nvidia, _ULTRA),
                ("openrouter", _openrouter, "qwen/qwen3.8-27b:free"),
            ):
                with self.subTest(status=status, provider=name):
                    provider = build(lambda request, status=status: httpx.Response(status, text="nope"))
                    try:
                        with self.assertRaises(error_class) as ctx:
                            await provider.generate(model, [{"role": "user", "content": "hi"}])
                    finally:
                        await provider.close()
                    self.assertEqual(ctx.exception.provider, name)
                    self.assertEqual(ctx.exception.model, model)

    async def test_an_error_in_a_200_body_is_that_error_not_an_empty_completion(self) -> None:
        # 2026-10-16: OpenRouter Ultra :free answered the incident body with
        # HTTP 200; the shared POST raises the 503 it names on either provider
        # (OpenRouter's one retry of an overloaded upstream got it again).
        for name, build, model in (("openrouter", _openrouter, _OR_ULTRA), ("nvidia", _nvidia, _ULTRA)):
            with self.subTest(provider=name):
                provider = build(lambda request: httpx.Response(200, json=_INCIDENT_BODY))
                try:
                    with self.assertRaises(TransientError) as ctx, _no_retry_pause():
                        await provider.generate(model, [{"role": "user", "content": "hi"}])
                finally:
                    await provider.close()
                self.assertNotIsInstance(ctx.exception, EmptyCompletionError)
                self.assertEqual((ctx.exception.status, ctx.exception.provider), (503, name))
                self.assertIn("Service temporarily overloaded", ctx.exception.detail)

    async def test_two_error_bodies_in_a_row_cool_the_model_for_two_minutes(self) -> None:
        now = [1_000.0]
        health = ModelHealth(clock=lambda: now[0])
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=_INCIDENT_BODY)

        provider = _openrouter(handler, health)
        chain = FallbackChain(health)
        try:
            for _ in range(2):
                with self.assertRaises(AllProvidersFailedError) as ctx, _no_retry_pause():
                    await chain.generate([Tier(provider, _OR_ULTRA)], [{"role": "user", "content": "hi"}])
                self.assertIsInstance(ctx.exception.last_error, TransientError)

            blocked = health.unavailable("openrouter", _OR_ULTRA)
            self.assertEqual((blocked.kind, blocked.reason), (COOLDOWN, "2 transient failures in a row"))
            # Later walks skip it without a request ...
            with self.assertRaises(AllProvidersFailedError):
                await chain.generate([Tier(provider, _OR_ULTRA)], [{"role": "user", "content": "hi"}])
            # (each walk: the request and its one retry, a strike only for the pair)
            self.assertEqual(len(requests), 4)
            # ... for 120 s.
            now[0] += 119
            self.assertFalse(health.is_available("openrouter", _OR_ULTRA))
            now[0] += 1
            self.assertTrue(health.is_available("openrouter", _OR_ULTRA))
        finally:
            await provider.close()

    async def test_nvidia_polls_a_202_through_its_own_client(self) -> None:
        responses = iter(
            [
                httpx.Response(202, headers={"NVCF-REQID": "req-7"}),
                httpx.Response(200, json=_chat("好了")),
            ]
        )
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(f"{request.method} {request.url}")
            return next(responses)

        provider = _nvidia(handler)
        with patch("src.providers.openai_compat.NVCF_POLL_INTERVAL_SECONDS", 0.0):
            response = await provider.generate(_ULTRA, [{"role": "user", "content": "hi"}])
        await provider.close()

        self.assertEqual(response.text, "好了")
        self.assertEqual(
            seen, [f"POST {_URL}", "GET https://integrate.api.nvidia.com/v1/status/req-7"]
        )

    async def test_empty_or_reasoning_only_replies_raise_empty_completion(self) -> None:
        for content in ("", "<think>ran out of budget", "  "):
            with self.subTest(content=content):
                provider = _nvidia(
                    lambda request, content=content: httpx.Response(
                        200, json=_chat(content, finish_reason="length")
                    ),
                )
                with self.assertRaises(EmptyCompletionError):
                    await provider.generate(_ULTRA, [{"role": "user", "content": "hi"}])
                await provider.close()

    async def test_lone_think_close_and_channel_markers_never_reach_the_user(self) -> None:
        for model, content, expected in (
            (_ULTRA, "The user greets me.\n</think>\n\n你好！", "你好！"),
            (_ULTRA, "<|channel>thought\n<channel|>台北 101", "台北 101"),
            (_MUSE, "<|channel>thought\n<channel|>一隻橘貓", "一隻橘貓"),
        ):
            with self.subTest(model=model):
                provider = _nvidia(lambda request, content=content: httpx.Response(200, json=_chat(content)))
                response = await provider.generate(model, [{"role": "user", "content": "hi"}])
                await provider.close()
                self.assertEqual(response.text, expected)
                self.assertEqual(response.provider, "nvidia")
                self.assertEqual(response.finish_reason, "stop")


_OR_DOTS = "dots-studio/dots-3-note-preview:free"
# What OpenRouter answered for dots-3 (one upstream, AtlasCloud) with HTTP
# 502 on 2026-10-16, to a memory summary after 15 s and in a live gate run.
_DROPPED_STREAM = {
    "error": {
        "message": "JSON error injected into SSE stream",
        "code": 502,
        "metadata": {"error_type": "provider_unavailable"},
    }
}


class OpenRouterUpstreamRetryTests(unittest.IsolatedAsyncioTestCase):
    """One more try when the model's upstream was unavailable, and only then."""

    @staticmethod
    def _answers(*responses: httpx.Response | Exception):
        """A handler giving *responses* in order, and the requests it saw."""
        pending = iter(responses)
        sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            answer = next(pending)
            if isinstance(answer, Exception):
                raise answer
            return answer

        return handler, sent

    async def _generate(self, *responses, health: ModelHealth | None = None):
        """The answer or the error, and the requests sent."""
        handler, sent = self._answers(*responses)
        provider = _openrouter(handler, health)
        try:
            with _no_retry_pause():
                return await provider.generate(_OR_DOTS, [{"role": "user", "content": "hi"}]), sent
        except ProviderError as exc:
            return exc, sent
        finally:
            await provider.close()

    async def test_a_dropped_upstream_stream_is_sent_once_more_and_answers(self) -> None:
        health = ModelHealth()
        handler, sent = self._answers(
            httpx.Response(502, json=_DROPPED_STREAM), httpx.Response(200, json=_chat("收盤價 1,450 元"))
        )
        provider = _openrouter(handler, health)
        try:
            with _no_retry_pause(), self.assertLogs("linebot", level="WARNING") as logs:
                result = await FallbackChain(health).generate(
                    [Tier(provider, _OR_DOTS)], [{"role": "user", "content": "hi"}]
                )
        finally:
            await provider.close()

        self.assertEqual(result.text, "收盤價 1,450 元")
        # The same request again.
        self.assertEqual(json.loads(sent[0].content), json.loads(sent[1].content))
        self.assertIn(f"OpenRouter: {_OR_DOTS} answered 502 (upstream unavailable); retrying once", logs.output[0])
        status = health.get_status()
        # Both requests count against the free daily budget; the answer
        # leaves the model without a strike.
        self.assertEqual(status["providers"]["openrouter"]["rpd_used"], 2)
        self.assertIsNone(status["models"][f"openrouter:{_OR_DOTS}"]["last_error_at"])

    async def test_the_same_failure_in_a_200_body_is_sent_once_more(self) -> None:
        for body in (_DROPPED_STREAM, _INCIDENT_BODY):
            with self.subTest(code=body["error"]["code"]):
                result, sent = await self._generate(
                    httpx.Response(200, json=body), httpx.Response(200, json=_chat("ok"))
                )
                self.assertEqual((result.text, len(sent)), ("ok", 2))

    async def test_the_error_type_survives_a_long_upstream_body(self) -> None:
        # The detail is cut to 500 characters; error_type goes before raw.
        error = {**_DROPPED_STREAM["error"]}
        error["metadata"] = {"raw": "x" * 600, "provider_name": "AtlasCloud", **error["metadata"]}

        result, sent = await self._generate(
            httpx.Response(200, json={"error": error}), httpx.Response(200, json=_chat("ok"))
        )

        self.assertEqual((result.text, len(sent)), ("ok", 2))

    async def test_a_second_failure_is_the_error(self) -> None:
        error, sent = await self._generate(
            httpx.Response(502, json=_DROPPED_STREAM), httpx.Response(502, json=_DROPPED_STREAM)
        )

        self.assertIsInstance(error, TransientError)
        self.assertEqual((error.status, len(sent)), (502, 2))
        self.assertIn("provider_unavailable", error.detail)

    async def test_other_failures_are_not_sent_again(self) -> None:
        for answer, error_class in (
            (httpx.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day"}}),
             RateLimitError),
            (httpx.Response(400, json={"error": {"message": "bad request", "code": 400}}), PayloadError),
            (httpx.Response(404, json={"error": {"message": "No endpoints found for x"}}), ModelGoneError),
            # A 5xx that does not say the upstream was unavailable.
            (httpx.Response(502, text="<html>502 Bad Gateway</html>"), TransientError),
            (httpx.Response(503, text="nope"), TransientError),
            (httpx.Response(500, json=_DROPPED_STREAM), TransientError),
            (httpx.ReadTimeout("read timed out"), TransientError),
        ):
            with self.subTest(answer=answer):
                error, sent = await self._generate(answer)
                self.assertIsInstance(error, error_class)
                self.assertEqual(len(sent), 1)

    async def test_no_retry_once_the_free_budget_is_spent(self) -> None:
        health = ModelHealth(openrouter_free_daily_limit=1)

        error, sent = await self._generate(httpx.Response(502, json=_DROPPED_STREAM), health=health)

        self.assertEqual((error.status, len(sent)), (502, 1))
        self.assertEqual(health.get_status()["providers"]["openrouter"]["rpd_used"], 1)

    async def test_a_paid_model_is_still_refused_without_a_request(self) -> None:
        handler, sent = self._answers()
        provider = _openrouter(handler)
        try:
            with self.assertRaises(SkipTierError), self.assertLogs("linebot", level="WARNING"):
                await provider.generate("qwen/qwen3.8-27b", [{"role": "user", "content": "hi"}])
        finally:
            await provider.close()
        self.assertEqual(sent, [])

    async def test_the_tier_limit_cancels_the_retry(self) -> None:
        # The pause is real here: a tier with 0.2 s left ends during it, and
        # the retry is never sent.
        health = ModelHealth()
        handler, sent = self._answers(httpx.Response(502, json=_DROPPED_STREAM))
        provider = _openrouter(handler, health)
        try:
            with self.assertLogs("linebot", level="WARNING"), self.assertRaises(AllProvidersFailedError) as ctx:
                await FallbackChain(health).generate(
                    [Tier(provider, _OR_DOTS, timeout_s=0.2)], [{"role": "user", "content": "hi"}]
                )
        finally:
            await provider.close()

        self.assertIn("no answer within 0.2s", str(ctx.exception))
        self.assertEqual(len(sent), 1)


if __name__ == "__main__":
    unittest.main()
