"""The per-event summary log line (src/utils/request_trace.py).

The Colab, NIM-router and failure cases run the whole pipeline in
tests/test_colab_pipeline.py (``test_summary_*``); these cover the line's
format, the image and voice paths, the tool loop's several walks (and who
answered when its last call failed), and that concurrent events keep their
own traces.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src.agents.chat_agent import ChatAgent
from src.agents.image_gen_agent import ImageGenAgent
from src.agents.routing_rules import RouterDecision
from src.agents.tools import ToolDefinition, ToolRegistry
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.pipeline.event_pipeline import PipelineDeps, process_event
from src.processors.output_processor import send_response
from src.providers.base import ProviderResponse, RateLimitError, SkipTierError, TransientError
from src.providers.fallback_chain import (
    AllProvidersFailedError,
    DeadlineExceededError,
    FallbackChain,
    Tier,
)
from src.runtime.model_roles import FLUX_DEV, FLUX_KLEIN, NIM_SUPER
from src.services.model_health import ModelHealth
from src.utils import request_trace
from src.utils.request_trace import (
    ANSWER,
    ROUTE,
    SEARCH_PLAN,
    Attempt,
    RequestTrace,
    RouterScores,
    Walk,
    summary_line,
)

QWEN = "colab:qwen3.8"
ULTRA = "nvidia:nemotron-ultra"


def _trace(**changes) -> RequestTrace:
    values = dict(
        preference="colab", request_id="abc12345", started=100.0, agent="chat", output="text", thinking="off"
    )
    values.update(changes)
    return RequestTrace(**values)


class SummaryFormatTests(unittest.TestCase):
    def test_a_tool_loop_aggregates_its_walks_and_the_last_answer_wins(self) -> None:
        not_ready = Attempt(QWEN, "skip(Colab session not READY)")
        trace = _trace(
            route=request_trace.ROUTE_LLM,
            walks=[
                Walk(ROUTE, (Attempt("nvidia:super", "ok", 0.4),)),
                Walk(ANSWER, (not_ready, Attempt(ULTRA, "ok", 1.0, think="off"))),
                Walk(SEARCH_PLAN, (Attempt("nvidia:super", "ok", 0.3),)),
                Walk(ANSWER, (not_ready, Attempt(ULTRA, "timeout", 3.0), Attempt("openrouter:qwen", "ok", 0.5))),
                Walk(ANSWER, (not_ready, Attempt(ULTRA, "ok", 0.7, think="off"))),
            ],
        )

        self.assertEqual(
            summary_line(trace, now=105.25),
            "[abc12345] summary pref=colab path=api route=nvidia:super agent=chat out=text think=off "
            f"answer={ULTRA} 0.7s answer_think=off total=5.2s router_p=n/a "
            f"tiers=[{QWEN} skip(Colab session not READY) x3, {ULTRA} ok 1.0s,timeout 3.0s,ok 0.7s, "
            "openrouter:qwen ok 0.5s] calls=3",
        )

    def test_a_failed_last_answer_walk_is_no_answer_whatever_answered_before(self) -> None:
        # A tool round, or a whole earlier answer, then a last call that failed.
        for earlier in (Attempt(QWEN, "tools", 2.0, think="low"), Attempt(QWEN, "ok", 2.0, think="low")):
            trace = _trace(
                error="AllProvidersFailedError(TransientError 503)",
                walks=[Walk(ANSWER, (earlier,)), Walk(ANSWER, (Attempt(QWEN, "error(TransientError 503)", 0.1),))],
            )
            with self.subTest(earlier.outcome):
                line = summary_line(trace, now=103)
                self.assertIn(" path=none ", line)
                self.assertIn(" answer=none error=AllProvidersFailedError(TransientError 503) ", line)
                self.assertNotIn("answer_think", line)

    def test_tiers_are_grouped_by_tier_in_the_order_first_reached(self) -> None:
        # Thinking pass (Colab, then NIM), then Colab again with thinking off:
        # Colab's two outcomes stay together, before the tier that ran between them.
        trace = _trace(
            walks=[
                Walk(
                    ANSWER,
                    (
                        Attempt(QWEN, "timeout", 25.0),
                        Attempt(ULTRA, "timeout", 5.0),
                        Attempt(QWEN, "ok", 3.0, think="low"),
                    ),
                )
            ]
        )

        line = summary_line(trace, now=134)
        self.assertIn(f" answer={QWEN} 3.0s answer_think=low ", line)
        self.assertTrue(line.endswith(f"tiers=[{QWEN} timeout 25.0s,ok 3.0s, {ULTRA} timeout 5.0s]"), line)

    def test_repeats_of_one_outcome_add_up(self) -> None:
        trace = _trace(walks=[Walk(ANSWER, (Attempt(ULTRA, "ok", 1.0),)), Walk(ANSWER, (Attempt(ULTRA, "ok", 0.5),))])

        self.assertIn(f"tiers=[{ULTRA} ok x2 1.5s] calls=2", summary_line(trace, now=101))

    def test_colab_router_probabilities_are_shown_whether_used_or_not(self) -> None:
        scores = RouterScores(
            model="clef-flash:9b",
            agent={"chat": 0.91, "web_search": 0.05},
            agent_confidence=0.94,
            output={"text": 0.96, "voice": float("nan")},
            output_confidence=0.95,
            think_p=0.5,
            gate=0.90,
            think_on=0.85,
            think_off=0.15,
            thinking="rules->on",
        )
        used = summary_line(_trace(route=request_trace.ROUTE_ROUTER, router_scores=scores), now=100)
        refused_scores = replace(scores, agent_confidence=0.6, verdict="gate fail", thinking="")
        refused = summary_line(_trace(route=request_trace.ROUTE_FALLBACK, router_scores=refused_scores), now=100)

        self.assertIn(" route=colab:clef-flash:9b ", used)
        self.assertIn(
            "router_p=[agent={chat:0.91,web_search:0.05} conf=0.94 gate=0.90 ok | format={text:0.96,voice:?} "
            "conf=0.95 ok | think p=0.50 (on>=0.85 off<=0.15) -> rules->on]",
            used,
        )
        self.assertIn(" route=fallback ", refused)
        self.assertIn("conf=0.60 gate=0.90 fail", refused)
        self.assertIn("think p=0.50 (on>=0.85 off<=0.15) | gate fail -> API router]", refused)

    def test_probabilities_keep_only_the_defined_options_highest_first(self) -> None:
        self.assertEqual(
            request_trace.probabilities({"b": 0.104, "a": 0.896, "x": 0.9, "c": True}, {"a", "b", "c"}),
            {"a": 0.9, "b": 0.1},
        )
        self.assertEqual(request_trace.probabilities(None, {"a"}), {})

    def test_an_unrouted_trace_and_a_failed_delivery(self) -> None:
        line = summary_line(RequestTrace(started=100.0, error="ValueError", delivered=False), now=100)

        self.assertEqual(
            line,
            "[-] summary pref=- path=none route=none agent=- out=- think=- answer=none error=ValueError "
            "sent=no total=0.0s router_p=n/a tiers=[]",
        )

    def test_a_skip_reason_is_one_short_line(self) -> None:
        attempt = request_trace.skipped("nvidia:x", "rate_limit:\n" + "y" * 100)

        self.assertEqual(len(attempt.outcome), len("skip()") + 60)
        self.assertNotIn("\n", attempt.outcome)

    def test_nothing_is_recorded_without_an_active_trace(self) -> None:
        request_trace.record_walk([Attempt(ULTRA, "ok", 1.0)])
        request_trace.note_tts("edge-tts")
        self.assertIsNone(request_trace.current())


class _TextProvider:
    def __init__(self, name: str, text: str = "a red apple, studio light") -> None:
        self.name = name
        self.text = text

    async def generate(self, model: str, messages: list[dict], **kwargs) -> ProviderResponse:
        return ProviderResponse(text=self.text, model=model)


class _ImageProvider:
    name = "nvidia"

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)

    async def generate_image(self, model: str, prompt: str, **kwargs) -> ProviderResponse:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ProviderResponse(images=[outcome], model=model)


class ImageAndVoiceSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_gen_answers_with_the_image_tier_and_names_the_prompt_tier(self) -> None:
        settings = Settings(_env_file=None)
        images = _ImageProvider(RateLimitError(FLUX_KLEIN, provider="nvidia"), "data:image/png;base64,AAAA")
        agent = ImageGenAgent(
            settings,
            FallbackChain(ModelHealth()),
            targets=[Tier(_TextProvider("nvidia"), NIM_SUPER)],
            image_tiers=[Tier(images, FLUX_KLEIN), Tier(images, FLUX_DEV)],
        )
        request = AgentRequest(text="畫一顆蘋果", received_at=time.time(), output_format="image")
        trace = _trace(agent="image_gen", output="image", route=request_trace.ROUTE_RULES)
        token = request_trace.activate(trace)
        try:
            with request_trace.phase(ANSWER):
                response = await agent.process(request)
        finally:
            request_trace.deactivate(token)

        self.assertEqual(response.image_base64, "data:image/png;base64,AAAA")
        line = summary_line(trace)
        self.assertIn(f" path=api route=rules agent=image_gen out=image think=off answer=nvidia:{FLUX_DEV} ", line)
        self.assertIn(f" prompt=nvidia:{NIM_SUPER} ", line)
        # The image tiers are the answer; the caption's walk is not.
        self.assertRegex(line, rf"tiers=\[nvidia:{FLUX_KLEIN} rate_limited \d\.\ds, nvidia:{FLUX_DEV} ok \d\.\ds\]$")
        self.assertNotIn("answer_think", line)

    async def test_voice_names_the_tts_engine(self) -> None:
        line_service = SimpleNamespace(send_audio=AsyncMock(return_value=True), send_text=AsyncMock(return_value=True))
        request = AgentRequest(reply_token="r", user_id="U1", output_format="voice")
        response = AgentResponse(text="你好", output_format="voice")
        for uploaded, tts in ((SimpleNamespace(public_url="u", duration_ms=1000, blob_name=""), "edge-tts"),
                              (None, "edge-tts(failed->text)")):
            trace = _trace(output="voice")
            token = request_trace.activate(trace)
            try:
                with (
                    patch("src.processors.output_processor.get_line_service", return_value=line_service),
                    patch("src.processors.output_processor.get_storage_service"),
                    patch("src.processors.tts_processor.text_to_speech", AsyncMock(return_value=uploaded)),
                ):
                    self.assertTrue(await send_response(request, response))
            finally:
                request_trace.deactivate(token)
            with self.subTest(tts):
                self.assertIn(f" out=voice think=off answer=none tts={tts} ", summary_line(trace))

    async def test_voice_that_could_not_be_sent_says_the_text_went_instead(self) -> None:
        line_service = SimpleNamespace(send_audio=AsyncMock(return_value=False), send_text=AsyncMock(return_value=True))
        request = AgentRequest(reply_token="r", user_id="U1", output_format="voice")
        uploaded = SimpleNamespace(public_url="u", duration_ms=1000, blob_name="")
        trace = _trace(output="voice")
        token = request_trace.activate(trace)
        try:
            with (
                patch("src.processors.output_processor.get_line_service", return_value=line_service),
                patch("src.processors.output_processor.get_storage_service"),
                patch("src.processors.tts_processor.text_to_speech", AsyncMock(return_value=uploaded)),
            ):
                self.assertTrue(await send_response(request, AgentResponse(text="你好", output_format="voice")))
        finally:
            request_trace.deactivate(token)

        line_service.send_text.assert_awaited_once()
        self.assertIn(" tts=edge-tts(send failed->text) ", summary_line(trace))

    async def test_a_flux_tier_the_provider_skips_is_a_skip(self) -> None:
        settings = Settings(_env_file=None)
        cooling = SkipTierError(FLUX_KLEIN, "cooldown: HTTP 403", "nvidia")
        cases = {
            "then FLUX.1-dev": ((cooling, "data:image/png;base64,AAAA"), "data:image/png;base64,AAAA"),
            # Unchanged: every tier skipped still gets the generic failure text.
            "all skipped": ((cooling, SkipTierError(FLUX_DEV, "dead: HTTP 410", "nvidia")), None),
        }
        for name, (outcomes, image) in cases.items():
            agent = ImageGenAgent(
                settings,
                FallbackChain(ModelHealth()),
                targets=[Tier(_TextProvider("nvidia"), NIM_SUPER)],
                image_tiers=[Tier(_ImageProvider(*outcomes), FLUX_KLEIN), Tier(_ImageProvider(*outcomes[1:]), FLUX_DEV)],
            )
            trace = _trace(agent="image_gen", output="image", route=request_trace.ROUTE_RULES)
            token = request_trace.activate(trace)
            try:
                with request_trace.phase(ANSWER):
                    response = await agent.process(
                        AgentRequest(text="畫一顆蘋果", received_at=time.time(), output_format="image")
                    )
            finally:
                request_trace.deactivate(token)

            line = summary_line(trace)
            with self.subTest(name):
                self.assertEqual(response.image_base64, image)
                self.assertIn(f"tiers=[nvidia:{FLUX_KLEIN} skip(cooldown: HTTP 403), nvidia:{FLUX_DEV} ", line)
                self.assertNotIn("SkipTierError", line)
                if image is None:
                    self.assertEqual(response.text, "圖片生成暫時失敗，請稍後再試或換個描述。")
                    self.assertIn(" path=none ", line)
                    self.assertTrue(line.endswith(f"nvidia:{FLUX_DEV} skip(dead: HTTP 410)]"), line)


class _ToolThenProvider:
    """A Colab tier: asks for a tool on the first call, then answers *later*
    (a ProviderResponse, or an exception to raise)."""

    name = "colab"

    def __init__(self, later) -> None:
        self.later = later
        self.calls = 0

    async def generate(self, model: str, messages: list[dict], **kwargs) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            call = {"id": "c1", "type": "function", "function": {"name": "recall_memory", "arguments": "{}"}}
            return ProviderResponse(tool_calls=[call], model=model)
        if isinstance(self.later, Exception):
            raise self.later
        return self.later


class ToolLoopSummaryTests(unittest.IsolatedAsyncioTestCase):
    """A tool round is not the answer: the last call decides who answered."""

    async def _run(self, later, *, tool_uses_up_the_time: bool = False) -> tuple[str, Exception | None]:
        # Seconds the chain's clock runs ahead of time.monotonic().
        offset = [0.0]

        async def recall(args, context):
            if tool_uses_up_the_time:
                offset[0] = 10_000.0
            return {"matches": []}

        registry = ToolRegistry()
        registry.register(ToolDefinition("recall_memory", "Recall.", {"type": "object", "properties": {}}, recall))
        chain = FallbackChain(ModelHealth(), clock=lambda: time.monotonic() + offset[0])
        agent = ChatAgent(
            Settings(_env_file=None),
            chain,
            targets=[Tier(_ToolThenProvider(later), "qwen3.8", options={"think": "low"})],
            tool_registry=registry,
        )
        request = AgentRequest(
            user_id="U1", group_id="U1", reply_token="r", text="q", received_at=time.time(), disable_thinking=True
        )
        trace = _trace()
        token = request_trace.activate(trace)
        error = None
        try:
            with request_trace.phase(ANSWER):
                await agent.process(request)
        except Exception as exc:
            error = exc
            trace.error = type(exc).__name__
        finally:
            request_trace.deactivate(token)
        return summary_line(trace), error

    async def test_a_tool_round_then_the_answer(self) -> None:
        line, error = await self._run(ProviderResponse(text="answer", model="qwen3.8"))

        self.assertIsNone(error)
        self.assertIn(f" path=colab route=none agent=chat out=text think=off answer={QWEN} ", line)
        self.assertRegex(line, rf"tiers=\[{QWEN} tools \d\.\ds,ok \d\.\ds\] calls=2$")

    async def test_a_tool_round_then_every_call_failing_is_no_answer(self) -> None:
        # The next round and chat's plain fallback both fail.
        line, error = await self._run(TransientError("qwen3.8", 503, "busy", "colab"))

        self.assertIsInstance(error, AllProvidersFailedError)
        self.assertIn(" path=none ", line)
        self.assertIn(" answer=none error=AllProvidersFailedError ", line)
        self.assertNotIn("answer_think", line)
        self.assertRegex(line, rf"tiers=\[{QWEN} tools \d\.\ds,error\(TransientError 503\) x2 \d\.\ds\] calls=3$")

    async def test_a_tool_round_then_the_deadline_is_no_answer(self) -> None:
        line, error = await self._run(ProviderResponse(text="too late", model="qwen3.8"), tool_uses_up_the_time=True)

        self.assertIsInstance(error, DeadlineExceededError)
        self.assertIn(" path=none ", line)
        self.assertIn(" answer=none error=DeadlineExceededError ", line)
        self.assertRegex(line, rf"tiers=\[{QWEN} tools \d\.\ds,skip\(deadline\) x2\] calls=3$")


class _SlowProvider:
    """Answers after *seconds*; fails a message that holds *fail_on*."""

    def __init__(self, name: str, seconds: float, fail_on: str = "") -> None:
        self.name = name
        self.seconds = seconds
        self.fail_on = fail_on

    async def generate(self, model: str, messages: list[dict], **kwargs) -> ProviderResponse:
        await asyncio.sleep(self.seconds)
        if self.fail_on and self.fail_on in str(messages[-1]["content"]):
            raise TransientError(model, 503, "busy", self.name)
        return ProviderResponse(text=f"{self.name} answer", model=model)


class ConcurrentTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_events_never_mix_their_traces(self) -> None:
        settings = Settings(_env_file=None)
        chain = FallbackChain(ModelHealth())
        tiers = [
            Tier(_SlowProvider("nvidia", 0.05, fail_on="second"), "alpha", options={"thinking": False}),
            Tier(_SlowProvider("openrouter", 0.01), "beta", options={"thinking": False}),
        ]
        agent = ChatAgent(settings, chain, targets=tiers)
        decision = RouterDecision("chat", "text", "", "", disable_thinking=True)
        requests = {
            text: AgentRequest(user_id="U1", group_id="U1", reply_token="r", text=text, received_at=time.time())
            for text in ("first question", "second question")
        }

        async def route(request):
            await asyncio.sleep(0.01)
            request_trace.note_route(request_trace.ROUTE_RULES)
            return decision

        def deps(text: str) -> PipelineDeps:
            return PipelineDeps(
                line=SimpleNamespace(
                    send_loading_animation=AsyncMock(),
                    send_text=AsyncMock(),
                    fetch_display_name=AsyncMock(return_value=""),
                ),
                memory=SimpleNamespace(
                    record_interaction=AsyncMock(),
                    touch_user_profile=AsyncMock(),
                    load_profile_facts=AsyncMock(return_value=[]),
                ),
                message_cache=SimpleNamespace(cache_processed_request=Mock()),
                orchestrator=SimpleNamespace(route=route),
                get_agent=lambda name: agent,
                process_input=AsyncMock(return_value=requests[text]),
                enrich_request=AsyncMock(side_effect=lambda request: request),
                extract_text=Mock(return_value=text),
                send_response=AsyncMock(return_value=True),
                model_path_preference="api",
            )

        event = {"type": "message", "replyToken": "r", "source": {"type": "user", "userId": "U1"}}
        with self.assertLogs("linebot", "INFO") as logs:
            await asyncio.gather(*(process_event(event, deps(text)) for text in requests))

        summaries = {
            line.split("]", 1)[0].lstrip("["): line
            for line in (record.getMessage() for record in logs.records)
            if " summary " in line
        }
        first = summaries[requests["first question"].request_id]
        second = summaries[requests["second question"].request_id]
        self.assertIn(" pref=api-only path=api route=rules ", first)
        self.assertIn(" answer=nvidia:alpha ", first)
        self.assertRegex(first, r"tiers=\[nvidia:alpha ok \d\.\ds\]$")
        self.assertIn(" answer=openrouter:beta ", second)
        self.assertRegex(second, r"tiers=\[nvidia:alpha error\(TransientError 503\) \d\.\ds, openrouter:beta ok \d\.\ds\]$")
        self.assertIsNone(request_trace.current())

    async def test_a_command_logs_no_summary(self) -> None:
        request = AgentRequest(user_id="U1", group_id="U1", reply_token="r", text="!new", received_at=time.time())
        deps = PipelineDeps(
            line=SimpleNamespace(send_text=AsyncMock()),
            memory=SimpleNamespace(clear_chat=AsyncMock()),
            message_cache=SimpleNamespace(),
            orchestrator=None,
            get_agent=lambda name: None,
            process_input=AsyncMock(return_value=request),
            extract_text=Mock(return_value="!new"),
        )

        with patch("src.pipeline.event_pipeline.logger") as logger:
            await process_event({"type": "message", "replyToken": "r", "source": {"userId": "U1"}}, deps)

        deps.line.send_text.assert_awaited_once()
        self.assertFalse([call for call in logger.info.call_args_list if " summary " in call.args[0]])


if __name__ == "__main__":
    unittest.main()
