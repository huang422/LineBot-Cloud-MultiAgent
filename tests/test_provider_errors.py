"""HTTP status -> provider error mapping, NVCF 202 polling and reply parsing.

Every request here goes through ``httpx.MockTransport``; nothing is sent.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

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
from src.providers.nvidia_provider import NvidiaProvider
from src.providers.openai_compat import (
    NVCF_STATUS_URL,
    error_for_status,
    parse_retry_after,
    post_chat_completion,
)
from src.providers.openrouter_provider import OpenRouterProvider
from src.services.model_health import ModelHealth

_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
_MUSE = "meta/muse-glimmer-30b"
_EOL_BODY = (
    '{"type":"about:blank","title":"Gone","status":410,"detail":"The model '
    "'qwen/qwen3.5-397b-a17b' has reached its end of life on 2026-07-27T00:00:00Z "
    'and is no longer available."}'
)


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


async def _post(*responses: httpx.Response | Exception, timeout: float = 5.0) -> tuple[dict, _Recorder]:
    recorder = _Recorder(*responses)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as client:
        data = await post_chat_completion(
            client,
            _URL,
            {"model": _ULTRA},
            provider="nvidia",
            model=_ULTRA,
            timeout=timeout,
            poll_interval=0.0,
        )
    return data, recorder


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


if __name__ == "__main__":
    unittest.main()
