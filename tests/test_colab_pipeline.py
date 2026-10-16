"""The request path with Colab: a real container, real providers, a fake tunnel.

Each test runs ``process_event`` end to end: the orchestrator (fast rules,
the Colab router, the API router), the chat agent and the fallback chain, with the
Colab endpoint over :class:`tests.colab_fakes.FakeTunnel` and the NVIDIA
provider replaced by a scripted fake. It checks who answered, what reached
the tunnel, and whether demand was recorded.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from src.config import Settings
from src.models.agent_request import AgentRequest
from src.pipeline.event_pipeline import PipelineDeps, process_event
from src.providers.base import ProviderResponse, TransientError
from src.runtime.composition import build_container
from src.runtime.model_roles import NIM_SUPER, NIM_ULTRA
from src.services.colab.state import IDLE, ColabState, InMemoryColabStateStore
from tests.colab_fakes import (
    QWEN,
    ROUTER,
    SECRET,
    FakeTunnel,
    healthy_routes,
    make_endpoint,
    ready_state,
    systemone_answer,
)

# No fast rule matches it, so a router decides.
_TEXT = "你覺得這個計畫如何"
_API_ROUTE = (
    '{"agent":"chat","output_format":"text","needs_thinking":false,'
    '"task_description":"評估計畫","reasoning":"API 判斷"}'
)


class FakeApi:
    """The NVIDIA provider: router JSON for the router, a text answer otherwise."""

    name = "nvidia"

    def __init__(self, *, chat_error: Exception | None = None, router_error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._chat_error = chat_error
        self._router_error = router_error

    async def generate(self, model: str, messages: list[dict], **kwargs) -> ProviderResponse:
        routing = kwargs.get("json_schema") is not None
        self.calls.append("router" if routing else "chat")
        error = self._router_error if routing else self._chat_error
        if error is not None:
            raise error
        return ProviderResponse(text=_API_ROUTE if routing else "API 的回答", model=model, provider=self.name)

    async def close(self) -> None:
        return None


class ColabPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        preference: str = "colab",
        state: ColabState | None = None,
        tunnel: FakeTunnel | None = None,
        api: FakeApi | None = None,
        min_tier_seconds: int = 5,
        store: InMemoryColabStateStore | None = None,
        text: str = _TEXT,
        source_type: str = "user",
        **settings_overrides,
    ):
        settings = Settings(
            _env_file=None,
            colab_enabled=True,
            firestore_enabled=True,
            colab_shared_secret=SECRET,
            model_path_preference=preference,
            tavily_keyless_enabled=False,
            min_tier_seconds=min_tier_seconds,
            **settings_overrides,
        )
        store = store or InMemoryColabStateStore(state if state is not None else ready_state())
        endpoint, tunnel = make_endpoint(tunnel=tunnel, store=store, auto_start=preference == "colab")
        api = api or FakeApi()
        container = build_container(settings, extra_providers={"nvidia": api}, colab=endpoint)
        self.addAsyncCleanup(container.aclose)
        # Every walk's deadline; the answer's is the last.
        chain = patch.object(container.fallback_chain, "generate", wraps=container.fallback_chain.generate)
        walks = chain.start()
        self.addCleanup(chain.stop)

        chat_id = "U1" if source_type == "user" else "C1"
        request = AgentRequest(
            user_id="U1",
            group_id=chat_id,
            reply_token="r",
            text=text,
            received_at=time.time(),
            source_type=source_type,
            chat_scope="user" if source_type == "user" else "multi",
        )
        sent: list = []
        sent_after: list[float] = []

        async def send_response(req, response) -> bool:
            sent.append(response)
            sent_after.append(time.monotonic() - started)
            return True

        line = SimpleNamespace(
            send_loading_animation=AsyncMock(),
            send_text=AsyncMock(),
            fetch_display_name=AsyncMock(return_value=""),
        )
        deps = PipelineDeps(
            line=line,
            memory=SimpleNamespace(
                record_interaction=AsyncMock(),
                touch_user_profile=AsyncMock(),
                load_profile_facts=AsyncMock(return_value=[]),
                flush_passive=AsyncMock(),
            ),
            message_cache=SimpleNamespace(cache_processed_request=Mock()),
            orchestrator=container.orchestrator,
            get_agent=container.get_agent,
            process_input=AsyncMock(return_value=request),
            enrich_request=AsyncMock(return_value=request),
            extract_text=Mock(return_value=text),
            send_response=send_response,
            reply_window_seconds=settings.reply_window_seconds,
            colab=endpoint,
            model_path_preference=preference,
        )
        source = {"type": source_type, "userId": "U1"}
        if source_type != "user":
            source[f"{source_type}Id"] = chat_id
        started = time.monotonic()
        await process_event({"type": "message", "replyToken": "r", "source": source}, deps)
        return SimpleNamespace(
            sent=sent,
            sent_after=sent_after,
            request=request,
            line=line,
            tunnel=tunnel,
            api=api,
            store=store,
            elapsed=time.monotonic() - started,
            answer_deadline=walks.await_args_list[-1].kwargs["deadline"] if walks.await_args_list else None,
        )

    # (a) ─────────────────────────────────────────────────────

    async def test_colab_preference_and_ready_colab_answers(self) -> None:
        run = await self._run()

        [response] = run.sent
        self.assertEqual(response.text, "Colab 的回答")
        self.assertEqual(run.tunnel.paths(), ["POST /v1/systemone", "POST /api/chat"])
        self.assertTrue(run.request.routing_reasoning.startswith(f"{ROUTER}:"))
        self.assertEqual(run.api.calls, [])
        # Thinking on by default (COLAB_CHAT_THINK=low) for chat.
        self.assertEqual(run.tunnel.bodies("POST /api/chat")[0]["think"], "low")
        self.assertIsNotNone(run.store.cached.last_demand_at)

    async def test_the_colab_tier_gets_the_thinking_budget(self) -> None:
        settings = Settings(
            _env_file=None,
            colab_enabled=True,
            firestore_enabled=True,
            colab_shared_secret=SECRET,
            nvidia_thinking_budget=1234,
        )
        endpoint, _ = make_endpoint(store=InMemoryColabStateStore(ready_state()))
        container = build_container(settings, extra_providers={"nvidia": FakeApi()}, colab=endpoint)
        self.addAsyncCleanup(container.aclose)

        self.assertEqual(container.providers["colab"]._thinking_budget, 1234)

    # (b) ─────────────────────────────────────────────────────

    async def test_colab_preference_not_ready_the_api_answers_at_once_and_demand_is_recorded(self) -> None:
        for state in (ColabState(status=IDLE), ColabState(status="BOOTING", nbh="n"), ready_state(last_health_at=0.0)):
            run = await self._run(state=state)
            with self.subTest(status=state.status):
                [response] = run.sent
                self.assertEqual(response.text, "API 的回答")
                self.assertEqual(run.tunnel.requests, [])
                self.assertEqual(run.api.calls, ["router", "chat"])
                self.assertEqual(run.request.routing_reasoning, "API 判斷")
                self.assertLess(run.elapsed, 1.0)
                self.assertAlmostEqual(run.store.cached.last_demand_at, time.time(), delta=5)

    async def test_a_slow_state_store_never_delays_the_reply(self) -> None:
        class SlowStore(InMemoryColabStateStore):
            async def update(self, change):
                await asyncio.sleep(30)
                return await super().update(change)

        store = SlowStore(ColabState(status=IDLE))
        with patch("src.services.colab.endpoint.DEMAND_WRITE_TIMEOUT_SECONDS", 0.5):
            run = await self._run(store=store)

        [response] = run.sent
        self.assertEqual(response.text, "API 的回答")
        # The answer went out at once; the demand write was joined (and
        # given up on) only afterwards, within its bound.
        self.assertLess(run.sent_after[0], 0.4)
        self.assertLess(run.elapsed, 1.5)
        self.assertEqual(store.writes, 0)

    # (c) ─────────────────────────────────────────────────────

    async def test_api_preference_api_first_no_colab_no_demand(self) -> None:
        run = await self._run(preference="api")

        [response] = run.sent
        self.assertEqual(response.text, "API 的回答")
        self.assertEqual(run.api.calls, ["router", "chat"])
        self.assertEqual(run.tunnel.requests, [])
        self.assertIsNone((await run.store.load()).last_demand_at)
        self.assertEqual(run.store.writes, 0)

    async def test_api_preference_uses_a_ready_colab_last(self) -> None:
        down = TransientError(NIM_ULTRA, 503, "down", "nvidia")
        run = await self._run(preference="api", api=FakeApi(chat_error=down))

        [response] = run.sent
        self.assertEqual(response.text, "Colab 的回答")
        self.assertEqual(run.tunnel.paths(), ["POST /api/chat"])
        self.assertEqual(run.store.writes, 0)

    async def test_api_preference_asks_the_colab_router_only_when_the_api_router_fails(self) -> None:
        down = TransientError(NIM_SUPER, 503, "down", "nvidia")
        run = await self._run(preference="api", api=FakeApi(router_error=down))

        self.assertEqual(run.tunnel.paths()[0], "POST /v1/systemone")
        self.assertTrue(run.request.routing_reasoning.startswith(f"{ROUTER}:"))
        self.assertEqual(run.sent[0].text, "API 的回答")

    async def test_api_preference_never_waits_for_a_colab_that_is_not_ready(self) -> None:
        down = TransientError(NIM_ULTRA, 503, "down", "nvidia")
        run = await self._run(preference="api", state=ColabState(status=IDLE), api=FakeApi(chat_error=down))

        self.assertEqual(run.sent, [])
        self.assertEqual(run.tunnel.requests, [])
        self.assertIn("AI 服務暫時不可用", run.line.send_text.await_args.args[2])
        self.assertEqual(run.store.writes, 0)

    # (d) ─────────────────────────────────────────────────────

    async def test_a_colab_failure_mid_request_falls_through_to_the_api(self) -> None:
        for name, answer in (
            ("503", (503, {"error": "server busy"})),
            ("524", (524, "timeout")),
            ("tunnel gone", httpx.ConnectError("refused")),
            ("model missing", (404, {"error": "model not found"})),
        ):
            run = await self._run(tunnel=FakeTunnel(**{"POST /api/chat": answer}))
            with self.subTest(name):
                [response] = run.sent
                self.assertEqual(response.text, "API 的回答")
                self.assertEqual(run.api.calls, ["chat"])
                self.assertLess(run.elapsed, 1.0)

    async def test_a_hung_colab_leaves_the_api_the_rest_of_the_reply_window(self) -> None:
        run = await self._run(
            tunnel=FakeTunnel(**{"POST /api/chat": ("hang", 60)}),
            min_tier_seconds=1,
            reply_window_seconds=3,
        )

        [response] = run.sent
        self.assertEqual(response.text, "API 的回答")
        # Colab got at most half the time left; the API answered inside the window.
        self.assertGreater(run.elapsed, 1.0)
        self.assertLess(run.elapsed, 3.0)
        self.assertEqual(run.tunnel.paths(), ["POST /v1/systemone", "POST /api/chat"])

    async def test_a_late_colab_answer_is_cancelled_never_sent(self) -> None:
        # Colab would answer after its share of the window (a slow thinking
        # qwen3.8): its request is cancelled when the share ends, so nothing
        # is left to deliver that answer; the API's is the only reply.
        colab: list[str] = []

        async def late_answer(request):
            colab.append("started")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                colab.append("cancelled")
                raise
            colab.append("answered")
            return httpx.Response(200, json=healthy_routes()["POST /api/chat"][1])

        run = await self._run(
            tunnel=FakeTunnel(**{"POST /api/chat": late_answer}),
            min_tier_seconds=1,
            reply_window_seconds=3,
        )

        [response] = run.sent
        self.assertEqual(response.text, "API 的回答")
        self.assertEqual(colab, ["started", "cancelled"])
        run.line.send_text.assert_not_called()

    # (d2) one reply window ──────────────────────────────────

    async def test_every_plain_answer_has_the_same_reply_window(self) -> None:
        # REPLY_WINDOW_SECONDS (90 s) after the event whoever answers first, a
        # READY Colab or the API tiers, in a 1:1 chat and a group alike; past
        # the reply token it goes by push.
        for name, kwargs, answer in (
            ("Colab first", {}, "Colab 的回答"),
            ("Colab first, group", {"source_type": "group"}, "Colab 的回答"),
            ("not READY", {"state": ColabState(status=IDLE)}, "API 的回答"),
            ("health check too old", {"state": ready_state(last_health_at=0.0)}, "API 的回答"),
            ("api preference", {"preference": "api"}, "API 的回答"),
        ):
            with self.subTest(name):
                run = await self._run(**kwargs)

                self.assertEqual(run.sent[0].text, answer)
                self.assertTrue(run.request.disable_thinking)
                self.assertEqual(run.request.reply_window_seconds, 90)
                self.assertAlmostEqual(run.answer_deadline, run.request.reply_deadline(), delta=0.01)

    async def test_thinking_keeps_the_turn_deadline_in_every_chat(self) -> None:
        tunnel = FakeTunnel(**{"POST /v1/systemone": (200, systemone_answer(noul=0.95))})
        for source_type in ("user", "group"):
            with self.subTest(source_type=source_type):
                run = await self._run(tunnel=tunnel, source_type=source_type)

                self.assertFalse(run.request.disable_thinking)
                self.assertAlmostEqual(run.answer_deadline, run.request.turn_deadline(), delta=0.01)

    async def test_an_image_has_the_reply_window_whether_colab_can_see_it_or_not(self) -> None:
        # A quoted image from the cache is a URL; Colab takes inline images only.
        settings = Settings(
            _env_file=None, colab_enabled=True, firestore_enabled=True, colab_shared_secret=SECRET
        )
        endpoint, _ = make_endpoint(store=InMemoryColabStateStore(ready_state()))
        await endpoint.ensure_loaded()
        container = build_container(settings, extra_providers={"nvidia": FakeApi()}, colab=endpoint)
        self.addAsyncCleanup(container.aclose)
        agent = container.vision_agent

        for name, image in (
            ("inline", {"image_base64": "data:image/jpeg;base64,abc"}),
            ("url", {"quoted_image_url": "https://storage.googleapis.com/b/i.jpg"}),
        ):
            with self.subTest(name):
                request = AgentRequest(text="這是什麼", disable_thinking=True, **image)

                deadline = agent._deadline_for(request, thinking=False)

                self.assertAlmostEqual(deadline, request.reply_deadline(), delta=0.01)

    # (e) ─────────────────────────────────────────────────────

    async def test_at_the_default_gate_the_colab_router_takes_its_top_choice_however_unsure(self) -> None:
        # COLAB_ROUTER_CONFIDENCE 0.0 (argmax); 0.39 is what real messages scored.
        tunnel = FakeTunnel(**{"POST /v1/systemone": (200, systemone_answer(agent_confidence=0.39))})

        run = await self._run(tunnel=tunnel)

        self.assertEqual(run.api.calls, [])
        self.assertEqual(run.request.target_agent, "chat")
        self.assertTrue(run.request.routing_reasoning.startswith(f"{ROUTER}: agent 0.39"))
        self.assertEqual(run.tunnel.paths(), ["POST /v1/systemone", "POST /api/chat"])

    async def test_an_unsure_colab_router_hands_routing_to_the_api_router(self) -> None:
        # Only under a gate set above its confidence (the default 0.0 never is).
        tunnel = FakeTunnel(**{"POST /v1/systemone": (200, systemone_answer(agent="web_search", agent_confidence=0.7))})

        run = await self._run(tunnel=tunnel, colab_router_confidence=0.90)

        self.assertEqual(run.api.calls, ["router"])
        self.assertEqual((run.request.target_agent, run.request.routing_reasoning), ("chat", "API 判斷"))
        self.assertEqual(run.tunnel.paths(), ["POST /v1/systemone", "POST /api/chat"])
        self.assertEqual(run.sent[0].text, "Colab 的回答")

    async def test_a_busy_colab_router_hands_routing_to_the_api_router_without_waiting(self) -> None:
        tunnel = FakeTunnel()
        store = InMemoryColabStateStore(ready_state())
        endpoint, _ = make_endpoint(tunnel=tunnel, store=store)
        settings = Settings(
            _env_file=None, colab_enabled=True, firestore_enabled=True, colab_shared_secret=SECRET
        )
        api = FakeApi()
        container = build_container(settings, extra_providers={"nvidia": api}, colab=endpoint)
        self.addAsyncCleanup(container.aclose)

        async with endpoint.semaphore(ROUTER):
            decision = await container.orchestrator.route(AgentRequest(text=_TEXT))

        self.assertEqual((decision.reasoning, api.calls), ("API 判斷", ["router"]))
        self.assertEqual(tunnel.requests, [])

    # (f) the summary line ────────────────────────────────────

    async def _summary(self, **kwargs) -> str:
        """The one summary line a run logs."""
        with self.assertLogs("linebot", "INFO") as logs:
            run = await self._run(**kwargs)
        [line] = [record.getMessage() for record in logs.records if " summary " in record.getMessage()]
        self.assertTrue(line.startswith(f"[{run.request.request_id}] summary "))
        self.assertNotIn(run.request.text, line)
        return line

    async def test_summary_colab_answer_routed_by_the_colab_router_with_its_probabilities(self) -> None:
        answer = systemone_answer(agent_confidence=0.94, output_confidence=0.95, noul=0.05)
        answer["answers"]["agent"]["probabilities"] = {
            "web_search": 0.041, "chat": 0.934, "vision": 0.02, "image_gen": 0.005, "other": 0.5,
        }
        answer["answers"]["output_format"]["probabilities"] = {"text": 0.97, "voice": 0.02, "image": 0.01}

        line = await self._summary(tunnel=FakeTunnel(**{"POST /v1/systemone": (200, answer)}))

        self.assertIn(
            f"summary pref=colab path=colab route=colab:{ROUTER} agent=chat out=text think=off "
            f"answer=colab:{QWEN} ",
            line,
        )
        # Colab chat thinks at COLAB_CHAT_THINK=low although the route says off.
        self.assertIn(" answer_think=low ", line)
        self.assertIn(
            " router_p=[agent={chat:0.93,web_search:0.04,vision:0.02,image_gen:0.01} conf=0.94 gate=0.00 ok"
            " | format={text:0.97,voice:0.02,image:0.01} conf=0.95 ok"
            " | think p=0.05 (on>=0.85 off<=0.15) -> off] ",
            line,
        )
        self.assertRegex(line, rf"tiers=\[colab:{QWEN} ok \d+\.\ds\]$")

    async def test_summary_api_answer_routed_by_the_nim_super_router(self) -> None:
        line = await self._summary(preference="api")

        self.assertIn(
            f"summary pref=api path=api route=nvidia:{NIM_SUPER} agent=chat out=text think=off "
            f"answer=nvidia:{NIM_SUPER} ",
            line,
        )
        self.assertIn(" answer_think=off ", line)
        self.assertIn(" router_p=n/a ", line)
        self.assertRegex(line, rf"tiers=\[nvidia:{NIM_SUPER} ok \d+\.\ds\]$")

    async def test_summary_an_unsure_colab_router_shows_its_probabilities_and_the_api_router(self) -> None:
        tunnel = FakeTunnel(**{"POST /v1/systemone": (200, systemone_answer(agent="web_search", agent_confidence=0.7))})

        line = await self._summary(tunnel=tunnel, colab_router_confidence=0.90)

        self.assertIn(f" route=nvidia:{NIM_SUPER} ", line)
        self.assertIn(
            "router_p=[agent={web_search:0.97} conf=0.70 gate=0.90 fail | format={text:0.98} conf=0.95 ok"
            " | think p=0.05 (on>=0.85 off<=0.15) | gate fail -> API router]",
            line,
        )

    async def test_summary_a_fast_rule_route_and_why_the_colab_router_was_not_asked(self) -> None:
        line = await self._summary(text="你好", state=ColabState(status=IDLE))

        self.assertIn(
            f"summary pref=colab path=api route=rules agent=chat out=text think=off answer=nvidia:{NIM_SUPER} ",
            line,
        )
        self.assertIn(" router_p=n/a ", line)
        self.assertIn(f"tiers=[colab:{QWEN} skip(Colab session not READY), nvidia:{NIM_SUPER} ok ", line)

        line = await self._summary(state=ColabState(status=IDLE))
        self.assertIn(f" route=nvidia:{NIM_SUPER} ", line)
        self.assertIn(" router_p=n/a(router not READY) ", line)

    async def test_summary_every_tier_failed(self) -> None:
        down = TransientError(NIM_ULTRA, 503, "down", "nvidia")

        line = await self._summary(preference="api", state=ColabState(status=IDLE), api=FakeApi(chat_error=down))

        self.assertIn(" path=none ", line)
        self.assertIn(" answer=none error=AllProvidersFailedError(TransientError 503) ", line)
        self.assertNotIn("answer_think", line)
        self.assertIn(f"nvidia:{NIM_ULTRA} error(TransientError 503) ", line)
        self.assertIn(f"colab:{QWEN} skip(Colab session not READY)", line)


if __name__ == "__main__":
    unittest.main()
