"""SystemOneRouter (nimble on /v1/systemone) and its place in the orchestrator."""

from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from src.agents.orchestrator import ERROR_REASON, Orchestrator
from src.agents.routing_rules import VALID_AGENTS, VALID_OUTPUTS, RouterDecision
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.providers.base import ProviderResponse, TransientError
from src.providers.fallback_chain import Tier
from src.providers.ollama_provider import OllamaChatProvider
from src.providers.systemone import (
    QUESTIONS,
    STATE_MESSAGE_CHARS,
    RouterGates,
    SystemOneRouter,
    build_request,
    build_state,
)
from src.services.colab import protocol
from src.services.colab.endpoint import BREAKER_FAILURES
from src.services.colab.state import IDLE
from src.services.model_health import ModelHealth
from tests.colab_fakes import (
    NIMBLE,
    QWEN,
    Clock,
    FakeTunnel,
    bearer,
    healthy_routes,
    make_endpoint,
    on_gpu,
    ready_state,
    systemone_answer,
)

# The provisional gates (the COLAB_ROUTER_* defaults).
GATES = RouterGates(confidence=0.75, think_on=0.85, think_off=0.15)


def _router(tunnel: FakeTunnel | None = None, *, state=None, clock=None, health=None, timeout_s: float = 6):
    endpoint, tunnel = make_endpoint(state, tunnel, clock=clock or time.time)
    tier = Tier(SimpleNamespace(name="colab"), NIMBLE, timeout_s=timeout_s)
    return SystemOneRouter(endpoint, tier, gates=GATES, health=health), endpoint, tunnel


def _answer(**kwargs) -> FakeTunnel:
    return FakeTunnel(**{"POST /v1/systemone": (200, systemone_answer(**kwargs))})


# A digest other than the pin, as after Bespoke republishes the tag (this
# one is nimble:9b-bf16's manifest).
_REPUBLISHED = "adfcdbb4070d303fb43fc11fc06a14269ec2bc7136a55369e86d412b86ba2831"


def _session(nimble: str, nbh: str = "", **changes):
    """A READY session whose probe saw *nimble* as nimble's digest."""
    pins = protocol.pinned_models()
    return ready_state(digests={QWEN: pins[QWEN], NIMBLE: nimble}, **({"nbh": nbh} if nbh else {}), **changes)


class RequestContractTests(unittest.TestCase):
    def test_request_shape(self) -> None:
        request = AgentRequest(text="幫我查一下明天台北會不會下雨", previous_agent="chat")

        body = build_request(NIMBLE, request)

        self.assertEqual(list(body), ["model", "keep_alive", "state", "questions"])
        self.assertEqual((body["model"], body["keep_alive"]), (NIMBLE, -1))
        self.assertEqual(
            body["state"],
            {"message": "幫我查一下明天台北會不會下雨", "has_image": False, "quoted": False, "previous_agent": "chat"},
        )
        self.assertEqual(list(body["questions"]), ["agent", "output_format", "needs_thinking"])
        agent, output, thinking = body["questions"].values()
        self.assertEqual((agent["type"], set(agent["criteria"])), ("choice", VALID_AGENTS))
        self.assertEqual((output["type"], set(output["criteria"])), ("choice", VALID_OUTPUTS))
        self.assertEqual((thinking["type"], set(thinking["criteria"])), ("noul", {"false", "true"}))
        json.dumps(body)  # serialisable as is

    def test_questions_are_english_one_sentence_each(self) -> None:
        for name, question in QUESTIONS.items():
            with self.subTest(question=name):
                self.assertTrue(question["instructions"].isascii())
                for option, description in question["criteria"].items():
                    self.assertTrue(description.isascii(), option)
                    self.assertEqual(description.count(". "), 0, option)
                    self.assertTrue(description.endswith("."), option)

    def test_questions_speak_of_the_context_nimble_sees(self) -> None:
        # Ollama compiles the state into {"context": ..., "schema": ...}:
        # nimble never sees the word "state", and each field says what it means.
        for name, question in QUESTIONS.items():
            with self.subTest(question=name):
                self.assertIn("context", question["instructions"])
                texts = [question["instructions"], *question["criteria"].values()]
                self.assertFalse(any("state" in text.lower() for text in texts))
                self.assertFalse(question["instructions"].startswith(("Pick", "Does", "Is ")))
        agent = QUESTIONS["agent"]["instructions"]
        for key in build_state(AgentRequest(text="嗨")):
            self.assertIn(key if key != "message" else "the message", agent)

    def test_the_criteria_agree_with_the_rulebook_and_the_orchestrator_prompt(self) -> None:
        # The image model cannot see the user's picture: restyling it is
        # vision's (tests/test_routing_rules.py), a new picture image_gen's.
        agent = QUESTIONS["agent"]["criteria"]
        self.assertIn("restyle", agent["vision"])
        self.assertIn("new picture", agent["image_gen"])
        self.assertIn("not only a change to an image the user sent or quoted", agent["image_gen"])
        # needs_thinking is asked about what prompts/orchestrator.md counts.
        thinking = QUESTIONS["needs_thinking"]["instructions"]
        for case in ("math", "writing, debugging or refactoring code", "weighing several options",
                     "multi-step planning", "long writing", "translating a long passage"):
            self.assertIn(case, thinking)
        self.assertIn("short translation", QUESTIONS["needs_thinking"]["criteria"]["false"])

    def test_the_state_carries_no_memory_and_a_clipped_message(self) -> None:
        request = AgentRequest(
            text="長" * 1000,
            memory_summary="使用者住在高雄",
            conversation_history=[{"role": "user", "content": "以前的訊息"}],
            user_profile_facts=["喜歡貓"],
            previous_agent="not-an-agent",
            quoted_message_id="M1",
            quoted_image_url="https://x/y.png",
        )

        state = build_state(request)

        self.assertEqual(len(state["message"]), STATE_MESSAGE_CHARS)
        self.assertEqual((state["has_image"], state["quoted"], state["previous_agent"]), (True, True, "none"))
        dumped = json.dumps(build_request(NIMBLE, request), ensure_ascii=False)
        for private in ("高雄", "以前的訊息", "喜歡貓"):
            self.assertNotIn(private, dumped)

    def test_each_scored_row_stays_inside_what_nimble_was_trained_on(self) -> None:
        # Every question is scored with the whole context and all fields.
        # Ollama allows 8,192 tokens a row, but nimble was trained on at
        # most 2,048: stay well under that even for a full zh-TW message,
        # with room for the chat template and nimble's SYSTEM prompt (385
        # bytes, before every row). A CJK character is at most one token;
        # English is ~4 chars a token.
        request = AgentRequest(text="測" * 2000)
        dumped = json.dumps(build_request(NIMBLE, request), ensure_ascii=False)
        cjk = sum(1 for char in dumped if not char.isascii())
        ascii_chars = len(dumped) - cjk
        self.assertLess(cjk + (ascii_chars + 385) / 3, 1200)


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    async def _route(self, text: str = "你覺得這個計畫如何", **answer):
        router, endpoint, tunnel = _router(_answer(**answer))
        try:
            return await router.route(AgentRequest(text=text)), tunnel
        finally:
            await endpoint.close()

    async def test_a_confident_answer_routes(self) -> None:
        decision, tunnel = await self._route(agent="web_search", agent_confidence=0.82, output_confidence=0.86)

        self.assertEqual(
            (decision.agent, decision.output_format, decision.disable_thinking),
            ("web_search", "text", True),
        )
        self.assertEqual(decision.task_description, "你覺得這個計畫如何")
        self.assertTrue(decision.reasoning.startswith(f"{NIMBLE}: agent 0.82"), decision.reasoning)
        [request] = tunnel.requests
        self.assertEqual(request.headers["authorization"], f"Bearer {bearer()}")
        self.assertEqual(request.url.path, "/v1/systemone")

    async def test_low_confidence_on_either_choice_defers_to_the_api_router(self) -> None:
        for kwargs in ({"agent_confidence": 0.74}, {"output_confidence": 0.6}):
            with self.subTest(**kwargs):
                decision, _ = await self._route(**kwargs)
                self.assertIsNone(decision)
        decision, _ = await self._route(agent_confidence=GATES.confidence, output_confidence=GATES.confidence)
        self.assertIsNotNone(decision)

    async def test_the_noul_gate(self) -> None:
        complex_text = "請一步一步推導這個數學證明"
        cases = [
            (GATES.think_on, "你好嗎", False),
            (0.99, "你好嗎", False),
            (GATES.think_off, complex_text, True),
            (0.0, complex_text, True),
            # Uncertain: the rulebook's heuristic decides.
            (0.8, complex_text, False),
            (0.2, "今天過得如何", True),
            (0.5, complex_text, False),
            (0.5, "今天過得如何", True),
        ]
        for noul, text, disabled in cases:
            with self.subTest(noul=noul, text=text):
                decision, _ = await self._route(text, noul=noul)
                self.assertEqual(decision.disable_thinking, disabled)

    async def test_the_orchestrator_invariants_hold(self) -> None:
        decision, _ = await self._route(agent="image_gen", output="image", noul=0.9)
        self.assertEqual((decision.agent, decision.output_format, decision.disable_thinking), ("image_gen", "image", True))

        # Steering the agent field alone never starts an image generation:
        # both answers must say so, else the API router decides.
        for output in ("text", "voice"):
            with self.subTest(output=output):
                decision, _ = await self._route(agent="image_gen", output=output)
                self.assertIsNone(decision)

        decision, _ = await self._route(agent="chat", output="image")
        self.assertEqual((decision.agent, decision.output_format), ("chat", "text"))

        decision, _ = await self._route(agent="vision")
        self.assertEqual(decision.agent, "chat")

        decision, _ = await self._route(output="voice")
        self.assertEqual(decision.output_format, "voice")

    async def test_malformed_answers_defer(self) -> None:
        bad = systemone_answer()
        cases = {
            "no answers": {"model": NIMBLE},
            "answers not a dict": {"answers": []},
            "unknown agent": {**bad, "answers": {**bad["answers"], "agent": {"choice": "sql", "confidence": 0.9}}},
            "confidence text": {**bad, "answers": {**bad["answers"], "agent": {"choice": "chat", "confidence": "high"}}},
            "confidence bool": {**bad, "answers": {**bad["answers"], "agent": {"choice": "chat", "confidence": True}}},
            "confidence > 1": {**bad, "answers": {**bad["answers"], "output_format": {"choice": "text", "confidence": 1.5}}},
            "no noul": {**bad, "answers": {**bad["answers"], "needs_thinking": {"type": "noul"}}},
            "not json": b"<html>",
            "a list": [1, 2],
        }
        for name, body in cases.items():
            router, endpoint, _ = _router(FakeTunnel(**{"POST /v1/systemone": (200, body)}))
            with self.subTest(name):
                self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
            await endpoint.close()


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_errors_and_timeouts_defer_and_count(self) -> None:
        cases = {
            500: (500, {"error": "OOM"}),
            503: (503, {"error": "server busy"}),
            524: (524, "timeout"),
            499: (499, {"error": "canceled"}),
            "connect": httpx.ConnectError("refused"),
        }
        for name, answer in cases.items():
            health = ModelHealth()
            router, endpoint, _ = _router(FakeTunnel(**{"POST /v1/systemone": answer}), health=health)
            with self.subTest(name):
                self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
                self.assertIsNotNone(health.get_status(include_errors=True)["models"][f"colab:{NIMBLE}"]["last_error"])
                # The router's own failure: nothing held against the session.
                self.assertEqual((endpoint.last_error, endpoint.breaker_open()), ("", False))
                self.assertTrue(endpoint.is_ready())
            await endpoint.close()

    async def test_a_slow_router_is_cut_at_its_time_limit(self) -> None:
        router, endpoint, _ = _router(FakeTunnel(**{"POST /v1/systemone": ("hang", 5)}), timeout_s=0.05)
        started = time.monotonic()

        self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))

        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(endpoint.semaphore(NIMBLE).locked())
        await endpoint.close()

    async def test_a_request_the_model_refuses_is_not_held_against_the_tunnel(self) -> None:
        for status in (400, 413):
            router, endpoint, _ = _router(FakeTunnel(**{"POST /v1/systemone": (status, {"error": "prompt too long"})}))
            with self.subTest(status=status):
                self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
                self.assertEqual(endpoint.last_error, "")
                self.assertTrue(endpoint.is_ready())
            await endpoint.close()

    async def test_auth_and_not_found_mark_the_endpoint_unhealthy(self) -> None:
        for status in (401, 403, 404):
            router, endpoint, _ = _router(FakeTunnel(**{"POST /v1/systemone": (status, {"error": "no"})}))
            with self.subTest(status=status):
                self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
                self.assertFalse(endpoint.is_ready())
            await endpoint.close()

    async def test_repeated_failures_cool_the_router_down_not_colab_chat(self) -> None:
        health = ModelHealth()
        router, endpoint, tunnel = _router(FakeTunnel(**{"POST /v1/systemone": (502, "bad gateway")}), health=health)
        for _ in range(BREAKER_FAILURES + 1):
            self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))

        # Two failures in a row cool nimble down (ModelHealth); the Colab
        # breaker qwen3.8 chat shares stays shut.
        self.assertEqual(len(tunnel.requests), 2)
        self.assertIsNotNone(health.unavailable("colab", NIMBLE))
        self.assertEqual((endpoint.breaker_open(), endpoint.last_error), (False, ""))
        chat = OllamaChatProvider(endpoint)
        tunnel.routes["POST /api/chat"] = healthy_routes()["POST /api/chat"]
        reply = await chat.generate(QWEN, [{"role": "user", "content": "你好"}])
        self.assertEqual(reply.text, "Colab 的回答")
        await endpoint.close()

    async def test_a_failing_router_without_health_never_opens_the_breaker(self) -> None:
        router, endpoint, tunnel = _router(FakeTunnel(**{"POST /v1/systemone": (502, "bad gateway")}))
        for _ in range(BREAKER_FAILURES + 1):
            self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))

        self.assertEqual(len(tunnel.requests), BREAKER_FAILURES + 1)
        self.assertFalse(endpoint.breaker_open())
        await endpoint.close()

    async def test_an_open_breaker_skips_the_router_too(self) -> None:
        router, endpoint, tunnel = _router(_answer())
        await endpoint.ready()
        for _ in range(BREAKER_FAILURES):
            endpoint.record_failure("qwen3.8 HTTP 500")

        self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()


class SkipTests(unittest.IsolatedAsyncioTestCase):
    async def test_not_ready_busy_cooling_or_out_of_time_skip_without_a_request(self) -> None:
        clock = Clock()

        router, endpoint, tunnel = _router(state=ready_state(clock(), status=IDLE), clock=clock)
        self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()

        router, endpoint, tunnel = _router()
        async with endpoint.semaphore(NIMBLE):
            self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()

        health = ModelHealth()
        health.cool_down("colab", NIMBLE, 60, reason="test")
        router, endpoint, tunnel = _router(health=health)
        self.assertIsNone(await router.route(AgentRequest(text="嗨囉")))
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()

        router, endpoint, tunnel = _router()
        self.assertIsNone(await router.route(AgentRequest(text="嗨囉"), deadline=time.monotonic() + 0.1))
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()

    async def test_a_republished_router_is_never_asked_and_reported_once_per_session(self) -> None:
        clock = Clock()
        router, endpoint, tunnel = _router(_answer(), state=_session(_REPUBLISHED, now=clock()), clock=clock)
        pinned = protocol.pinned_models()[NIMBLE]

        with self.assertLogs("linebot", "ERROR") as logs:
            for _ in range(3):
                self.assertIsNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))

        self.assertEqual(tunnel.requests, [])
        (error,) = [record for record in logs.records if record.levelname == "ERROR"]
        self.assertIn(f"{NIMBLE} has digest {_REPUBLISHED}, not the pinned {pinned}", error.getMessage())
        # Nothing held against the session: chat on Colab goes on.
        self.assertTrue(endpoint.is_ready())
        self.assertEqual((endpoint.last_error, endpoint.breaker_open()), ("", False))

        # The next session with the same tag is reported again.
        await endpoint.store.update(lambda state: _session(_REPUBLISHED, nbh="another-session", now=clock()))
        with self.assertLogs("linebot", "ERROR") as logs:
            self.assertIsNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))
        self.assertEqual(len([r for r in logs.records if r.levelname == "ERROR"]), 1)
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()

    async def test_a_session_with_no_router_digest_counts_as_unpinned(self) -> None:
        clock = Clock()
        pins = protocol.pinned_models()
        for digests in ({}, {QWEN: pins[QWEN]}, {QWEN: pins[QWEN], NIMBLE: ""}):
            router, endpoint, tunnel = _router(_answer(), state=ready_state(clock(), digests=digests), clock=clock)
            with self.subTest(digests=digests), self.assertLogs("linebot", "ERROR") as logs:
                self.assertIsNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))
                self.assertEqual(tunnel.requests, [])
                self.assertIn(f"{NIMBLE} has digest (none recorded), not the pinned", logs.output[0])
            await endpoint.close()

    async def test_a_router_not_wholly_in_vram_sits_the_session_out(self) -> None:
        clock = Clock()
        cases = {
            "not fully on GPU (6144 of 10110 MiB in VRAM)": {QWEN: on_gpu(QWEN), NIMBLE: on_gpu(NIMBLE, size_vram=6 << 30)},
            # Not loaded (no sizes recorded for it): not trusted either.
            "not loaded (/api/ps)": {QWEN: on_gpu(QWEN)},
        }
        for shares, vram in cases.items():
            state = _session(protocol.pinned_models()[NIMBLE], now=clock(), vram=vram)
            router, endpoint, tunnel = _router(_answer(), state=state, clock=clock)
            with self.subTest(shares), self.assertLogs("linebot", "ERROR") as logs:
                for _ in range(2):
                    self.assertIsNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))
                self.assertEqual(tunnel.requests, [])
                (error,) = logs.output
                self.assertIn(f"{NIMBLE} off for this Colab session: router {shares}", error)
                # Chat on Colab goes on.
                self.assertTrue(endpoint.is_ready())
                self.assertEqual(endpoint.last_error, "")
            await endpoint.close()

    async def test_a_router_a_health_check_finds_gone_is_not_asked_again(self) -> None:
        # Asking it would load it again, and that can evict qwen3.8.
        clock = Clock()
        tunnel = FakeTunnel(**{"GET /api/ps": (200, {"models": [{"name": QWEN, **on_gpu(QWEN)}]})})
        router, endpoint, tunnel = _router(tunnel, state=_session(protocol.pinned_models()[NIMBLE], now=clock()), clock=clock)
        self.assertTrue(await endpoint.ready())
        with self.assertLogs("linebot", "WARNING"):
            self.assertTrue(await endpoint.check_health())

        with self.assertLogs("linebot", "ERROR") as logs:
            self.assertIsNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))

        self.assertNotIn("POST /v1/systemone", tunnel.paths())
        self.assertIn("router not loaded (/api/ps)", logs.output[0])
        self.assertTrue(endpoint.is_ready())
        await endpoint.close()

    async def test_the_pinned_router_wholly_in_vram_is_asked(self) -> None:
        clock = Clock()
        state = _session(protocol.pinned_models()[NIMBLE], now=clock())
        router, endpoint, tunnel = _router(_answer(), state=state, clock=clock)

        self.assertIsNotNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))
        self.assertEqual(tunnel.paths(), ["POST /v1/systemone"])
        await endpoint.close()

    async def test_a_busy_chat_model_does_not_block_the_router(self) -> None:
        router, endpoint, _ = _router(_answer())
        async with endpoint.semaphore("qwen3.8:27b-mtp-q4_K_M"):
            self.assertIsNotNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))
        await endpoint.close()


def _api_router(reply: str | Exception):
    """A fallback chain whose router tiers answer *reply* (or raise it)."""
    if isinstance(reply, Exception):
        return SimpleNamespace(generate=AsyncMock(side_effect=reply))
    return SimpleNamespace(generate=AsyncMock(return_value=ProviderResponse(text=reply, model="super")))


_API_JSON = '{"agent":"web_search","output_format":"text","needs_thinking":false,"task_description":"查","reasoning":"API"}'


class OrchestratorOrderTests(unittest.IsolatedAsyncioTestCase):
    def _orchestrator(self, chain, system_one, *, first: bool) -> Orchestrator:
        return Orchestrator(Settings(_env_file=None), chain, targets=[], system_one=system_one, system_one_first=first)

    async def test_colab_preference_asks_the_colab_router_first(self) -> None:
        chain = _api_router(_API_JSON)
        colab_router = SimpleNamespace(route=AsyncMock(return_value=RouterDecision("chat", "text", "x", "nimble")))

        decision = await self._orchestrator(chain, colab_router, first=True).route(AgentRequest(text="你覺得這個計畫如何"))

        self.assertEqual(decision.reasoning, "nimble")
        chain.generate.assert_not_called()
        self.assertIn("deadline", colab_router.route.await_args.kwargs)

    async def test_an_unsure_or_failing_colab_router_hands_over_to_the_api_router(self) -> None:
        for route in (AsyncMock(return_value=None), AsyncMock(side_effect=RuntimeError("bug"))):
            chain = _api_router(_API_JSON)
            orchestrator = self._orchestrator(chain, SimpleNamespace(route=route), first=True)
            with self.subTest(route=route):
                decision = await orchestrator.route(AgentRequest(text="你覺得這個計畫如何"))
                self.assertEqual((decision.agent, decision.reasoning), ("web_search", "API"))
                chain.generate.assert_awaited_once()

    async def test_api_preference_asks_the_colab_router_only_when_the_api_router_fails(self) -> None:
        colab_router = SimpleNamespace(route=AsyncMock(return_value=RouterDecision("chat", "voice", "x", "nimble")))
        decision = await self._orchestrator(_api_router(_API_JSON), colab_router, first=False).route(
            AgentRequest(text="你覺得這個計畫如何")
        )
        self.assertEqual(decision.reasoning, "API")
        colab_router.route.assert_not_called()

        decision = await self._orchestrator(
            _api_router(TransientError("super", 503, "down", "nvidia")), colab_router, first=False
        ).route(AgentRequest(text="你覺得這個計畫如何"))
        self.assertEqual((decision.reasoning, decision.output_format), ("nimble", "voice"))

    async def test_nobody_sure_is_the_chat_fallback(self) -> None:
        colab_router = SimpleNamespace(route=AsyncMock(return_value=None))
        decision = await self._orchestrator(
            _api_router(TransientError("super", 503, "down", "nvidia")), colab_router, first=False
        ).route(AgentRequest(text="你覺得這個計畫如何"))

        self.assertEqual((decision.agent, decision.reasoning, decision.disable_thinking), ("chat", ERROR_REASON, True))
        colab_router.route.assert_awaited_once()

    async def test_a_republished_router_leaves_routing_to_the_rules_and_the_api_router(self) -> None:
        clock = Clock()
        router, endpoint, tunnel = _router(_answer(agent="chat"), state=_session(_REPUBLISHED, now=clock()), clock=clock)
        chain = _api_router(_API_JSON)
        orchestrator = self._orchestrator(chain, router, first=True)

        with self.assertLogs("linebot", "ERROR"):
            decision = await orchestrator.route(AgentRequest(text="你覺得這個計畫如何"))
        drawn = await orchestrator.route(AgentRequest(text="幫我畫一隻貓"))

        self.assertEqual((decision.agent, decision.reasoning), ("web_search", "API"))
        chain.generate.assert_awaited_once()
        self.assertEqual(drawn.agent, "image_gen")
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()

    async def test_the_pinned_router_routes_before_the_api_router(self) -> None:
        clock = Clock()
        state = _session(protocol.pinned_models()[NIMBLE], now=clock())
        router, endpoint, tunnel = _router(_answer(agent="chat"), state=state, clock=clock)
        chain = _api_router(_API_JSON)

        decision = await self._orchestrator(chain, router, first=True).route(AgentRequest(text="你覺得這個計畫如何"))

        self.assertEqual(decision.agent, "chat")
        self.assertTrue(decision.reasoning.startswith(f"{NIMBLE}:"))
        chain.generate.assert_not_called()
        self.assertEqual(tunnel.paths(), ["POST /v1/systemone"])
        await endpoint.close()

    async def test_fast_rules_never_ask_the_colab_router(self) -> None:
        colab_router = SimpleNamespace(route=AsyncMock())
        decision = await self._orchestrator(_api_router(_API_JSON), colab_router, first=True).route(
            AgentRequest(text="幫我畫一隻貓")
        )
        self.assertEqual(decision.agent, "image_gen")
        colab_router.route.assert_not_called()


if __name__ == "__main__":
    unittest.main()
