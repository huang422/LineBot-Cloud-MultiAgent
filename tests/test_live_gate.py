"""Offline tests for scripts/live_gate.py.

Every client here uses httpx.MockTransport: the gate's live requests are
only ever sent by a human running the script. The chat, embedding and image
checks run the app's own clients over the gate's bridge, so these tests
also pin that the gate sends exactly the bot's payloads.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import io
import json
import logging
import os
import re
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import httpx

from scripts import live_gate
from src.config import Settings
from src.providers import model_profiles
from src.providers.fallback_chain import AllModelsRateLimitedError
from src.services import memory_service
from src.services.colab.state import ColabState, ColabStateError
from tests import colab_fakes

_NVIDIA_KEY = "nvapi-test-key-0123456789"
_OPENROUTER_KEY = "sk-or-v1-test-key-0123456789"

Responder = Callable[[httpx.Request, dict], httpx.Response]

_ROUTE = {
    "agent": "web_search",
    "output_format": "text",
    "needs_thinking": False,
    "task_description": "查詢台北天氣",
    "reasoning": "即時資訊",
}
_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_current_time", "arguments": '{"timezone":"Asia/Taipei"}'},
}


def _chat(content: str | None = "", **message: object) -> dict:
    finish = message.pop("finish_reason", "stop")
    usage = message.pop("usage", None)
    body: dict = {
        "choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": finish}]
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _image_part(payload: dict) -> dict | None:
    for message in payload["messages"]:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    return part
    return None


def _png_base64() -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048).decode()


class _Recorder:
    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        payload = json.loads(request.content) if request.content else {}
        return self.responder(request, payload)

    def payloads(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests if request.content]


def _context(
    responder: Responder,
    *,
    nvidia_key: str = _NVIDIA_KEY,
    openrouter_key: str = _OPENROUTER_KEY,
    settings: Settings | None = None,
) -> tuple[live_gate.GateContext, _Recorder]:
    recorder = _Recorder(responder)
    ctx = live_gate.GateContext(
        client=httpx.Client(transport=httpx.MockTransport(recorder)),
        nvidia_key=nvidia_key,
        openrouter_key=openrouter_key,
        timeout=5.0,
        sleep=lambda _seconds: None,
        settings=settings or Settings(_env_file=None),
    )
    return ctx, recorder


def _check(check_id: str) -> live_gate.Check:
    return next(check for check in live_gate.CHECKS if check.check_id == check_id)


class _FakeProvider:
    """Stands in for a provider when only the role table's binding matters."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def close(self) -> None:
        pass


class HelperTests(unittest.TestCase):
    def test_read_chat_output_separates_every_reasoning_format(self) -> None:
        cases = {
            "reasoning_content": (_chat("答案", reasoning_content="想一想"), "答案", "想一想"),
            "think pair": (_chat("<think>想一想</think>答案"), "答案", "想一想"),
            "lone closing tag": (_chat("想一想</think>答案"), "答案", "想一想"),
            "channel block": (_chat("<|channel>thought\n<channel|>紅色的圓形"), "紅色的圓形", ""),
        }
        for name, (data, answer, reasoning) in cases.items():
            with self.subTest(name):
                out = live_gate.read_chat_output(data)
                self.assertEqual(out.answer, answer)
                self.assertEqual(out.reasoning, reasoning)

        self.assertTrue(live_gate.read_chat_output(_chat("<|channel>thought x<channel|>ok")).channel_markers)
        self.assertEqual(live_gate.read_chat_output({}).answer, "")

    def test_parse_route_holds_replies_to_the_bots_router_schema(self) -> None:
        self.assertEqual(live_gate.parse_route("```json\n" + json.dumps(_ROUTE) + "\n```"), _ROUTE)
        for agent in ("image_gen", "vision"):
            with self.subTest(agent=agent):
                self.assertEqual(live_gate.parse_route(json.dumps({**_ROUTE, "agent": agent}))["agent"], agent)
        with self.assertRaisesRegex(live_gate.GateError, "is not a router agent"):
            live_gate.parse_route(json.dumps({**_ROUTE, "agent": "calendar"}))
        with self.assertRaisesRegex(live_gate.GateError, "missing needs_thinking"):
            live_gate.parse_route(json.dumps({k: v for k, v in _ROUTE.items() if k != "needs_thinking"}))
        with self.assertRaisesRegex(live_gate.GateError, "is not a router output"):
            live_gate.parse_route(json.dumps({**_ROUTE, "output_format": "pdf"}))

    def test_bot_route_holds_the_object_the_orchestrator_reads_to_the_schema(self) -> None:
        route = json.dumps(_ROUTE, ensure_ascii=False)
        # Prose and a fence around it are fine: the bot reads past them.
        self.assertEqual(live_gate.bot_route(f"路由結果：\n```json\n{route}\n```"), _ROUTE)
        # The orchestrator routes each of these (to chat); the gate does not pass them.
        template = {**_ROUTE, "agent": "chat", "task_description": "...", "reasoning": "..."}
        cases = {
            "{}": "missing agent,output_format,needs_thinking,task_description,reasoning",
            '{"foo": 1}': "missing agent,",
            json.dumps({**_ROUTE, "agent": "foo"}): "agent='foo' is not a router agent",
            json.dumps({**_ROUTE, "agent": ["chat"]}): "is not a router agent",
            json.dumps({**_ROUTE, "output_format": "pdf"}): "output_format='pdf' is not a router output",
            json.dumps(template): "task_description='...' describes no task",
            json.dumps({**_ROUTE, "task_description": " "}): "describes no task",
            "應該查網路": "no JSON object in the reply",
        }
        for reply, message in cases.items():
            with self.subTest(reply=reply):
                with self.assertRaisesRegex(live_gate.GateError, re.escape(message)):
                    live_gate.bot_route(reply)

    def test_the_chains_429_still_names_the_daily_cap(self) -> None:
        # run_tool_loop's chain raises its own error, with the last 429's text in it.
        capped = AllModelsRateLimitedError(
            "All targets exhausted. Last error: openrouter error 429 for qwen: "
            '{"error":{"message":"Rate limit exceeded: free-models-per-day."}}'
        )
        self.assertEqual(
            str(live_gate._gate_error(capped)), "HTTP 429 (rate limited: OpenRouter's free-models-per-day cap)"
        )
        upstream = AllModelsRateLimitedError("All targets exhausted. Last error: qwen is rate-limited upstream")
        self.assertEqual(str(live_gate._gate_error(upstream)), "HTTP 429 (rate limited upstream; retry later)")
        cooled = AllModelsRateLimitedError("All targets exhausted. Last error: None")
        self.assertEqual(str(live_gate._gate_error(cooled)), "HTTP 429 (rate limited)")

    def test_sniff_image_mime(self) -> None:
        self.assertEqual(live_gate.sniff_image_mime(b"\xff\xd8\xff\xe0rest"), "image/jpeg")
        self.assertEqual(live_gate.sniff_image_mime(b"\x89PNG\r\n\x1a\n"), "image/png")
        self.assertEqual(live_gate.sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertEqual(live_gate.sniff_image_mime(b"GIF89a"), "unknown")

    def test_cjk_ratio_and_traditional_detection(self) -> None:
        self.assertEqual(live_gate.cjk_ratio("台北 101"), 0.4)
        self.assertEqual(live_gate.cjk_ratio("..."), 0.0)
        self.assertTrue(live_gate.has_traditional_chars("這是繁體中文"))
        self.assertFalse(live_gate.has_traditional_chars("这是简体中文"))

    def test_redact_removes_keys_and_key_shaped_tokens(self) -> None:
        text = f"bad key {_NVIDIA_KEY} or nvapi-otherKEY12345678 / sk-or-v1-abcdefgh1234"

        redacted = live_gate.redact(text, [_NVIDIA_KEY])

        self.assertNotIn("nvapi-", redacted)
        self.assertNotIn("sk-or-", redacted)

    def test_red_circle_png_is_a_png(self) -> None:
        data = base64.b64decode(live_gate.red_circle_png_base64(64))
        self.assertEqual(live_gate.sniff_image_mime(data), "image/png")

    def test_select_checks_by_path_and_prefix(self) -> None:
        nvidia = live_gate.select_checks("nvidia")
        self.assertTrue(nvidia)
        self.assertTrue(all(check.path == "nvidia" for check in nvidia))
        self.assertEqual(
            [check.check_id for check in live_gate.select_checks("all", ["nvidia.ultra"])],
            ["nvidia.ultra.plain", "nvidia.ultra.tools", "nvidia.ultra.thinking", "nvidia.ultra.router"],
        )
        self.assertEqual(
            [check.check_id for check in live_gate.select_checks("all", ["nvidia.super.router"])],
            ["nvidia.super.router"],
        )
        self.assertEqual(live_gate.select_checks("openrouter", ["nvidia.ultra"]), [])

    def test_gate_models_are_the_role_table_defaults(self) -> None:
        from src.runtime.model_roles import default_role_specs

        default_models = {
            spec.model for specs in default_role_specs(Settings(_env_file=None)).values() for spec in specs
        }
        for model in (
            live_gate.ULTRA,
            live_gate.SUPER,
            live_gate.MUSE,
            live_gate.OMNI,
            live_gate.EMBED,
            live_gate.FLUX_KLEIN,
            live_gate.FLUX_DEV,
            live_gate.OR_QWEN,
            live_gate.OR_ULTRA,
            live_gate.OR_SUPER,
        ):
            with self.subTest(model=model):
                self.assertIn(model, default_models)

    def test_role_tiers_come_from_the_bound_role_table(self) -> None:
        providers = {"nvidia": _FakeProvider("nvidia"), "openrouter": _FakeProvider("openrouter")}
        table = live_gate.role_table(Settings(_env_file=None), providers)

        router = live_gate.role_tier(table, "router", "openrouter", live_gate.OR_SUPER)
        self.assertIs(router.provider, providers["openrouter"])
        self.assertEqual(dict(router.options), {"effort": "none", "json_mode": "json_object", "thinking": False})
        ultra = live_gate.role_tier(table, "router", "nvidia", live_gate.ULTRA)
        self.assertIs(ultra.provider, providers["nvidia"])
        self.assertEqual(
            dict(ultra.options), {"json_mode": "prompt", "max_tokens": 256, "temperature": 0.1, "thinking": False}
        )
        self.assertEqual(
            dict(live_gate.role_tier(table, "image", "nvidia", live_gate.FLUX_DEV).options), {"steps": 50}
        )
        with self.assertRaisesRegex(live_gate.GateSkip, "not in the configured chat role"):
            live_gate.role_tier(table, "chat", "nvidia", live_gate.FLUX_DEV)

    def test_role_tier_skips_what_force_tier_turned_off_and_fails_a_missing_key(self) -> None:
        forced = live_gate.role_table(
            Settings(_env_file=None, force_tier="openrouter"),
            {"nvidia": _FakeProvider("nvidia"), "openrouter": _FakeProvider("openrouter")},
        )
        with self.assertRaisesRegex(live_gate.GateSkip, "FORCE_TIER=openrouter"):
            live_gate.role_tier(forced, "chat", "nvidia", live_gate.ULTRA)

        keyless = live_gate.role_table(Settings(_env_file=None), {"openrouter": _FakeProvider("openrouter")})
        with self.assertRaisesRegex(live_gate.GateError, "NVIDIA_API_KEY not set"):
            live_gate.role_tier(keyless, "chat", "nvidia", live_gate.ULTRA)

    def test_first_api_tier_passes_over_colab_and_disabled_tiers(self) -> None:
        colab = _FakeProvider("colab")
        table = live_gate.role_table(
            Settings(_env_file=None), {"colab": colab, "openrouter": _FakeProvider("openrouter")}
        )

        row = live_gate.first_api_tier(table, "chat")

        self.assertEqual(row.label, f"openrouter:{live_gate.OR_QWEN}")
        with self.assertRaisesRegex(live_gate.GateError, "no enabled non-Colab tier.*NVIDIA_API_KEY not set"):
            live_gate.first_api_tier(table, "embed")

    def test_an_invalid_role_table_is_a_gate_error(self) -> None:
        with self.assertRaisesRegex(live_gate.GateError, "role table: .*FORCE_TIER=colab"):
            live_gate.role_table(Settings(_env_file=None, force_tier="colab"), {})

    def test_tier_kwargs_fold_the_tier_thinking_option_like_the_chain(self) -> None:
        provider = _FakeProvider("nvidia")
        self.assertEqual(
            live_gate.tier_kwargs(live_gate.Tier(provider, "m", options={"thinking": True}), disable_thinking=True),
            {"options": {"thinking": True}, "disable_thinking": False},
        )
        self.assertEqual(
            live_gate.tier_kwargs(live_gate.Tier(provider, "m"), disable_thinking=True),
            {"options": {}, "disable_thinking": True},
        )

    def test_untested_overrides_names_keys_only(self) -> None:
        settings = Settings(
            _env_file=None,
            model_roles='{"chat": [{"provider": "nvidia", "model": "example/other-model"}]}',
            force_tier="nvidia",
        )

        self.assertEqual(live_gate.untested_overrides(settings), ["MODEL_ROLES", "FORCE_TIER"])
        self.assertEqual(live_gate.untested_overrides(Settings(_env_file=None)), [])

    def test_tier_timeout_reads_the_configured_table(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(live_gate.tier_timeout(settings, "vision", "nvidia", live_gate.OMNI), 15)
        self.assertEqual(live_gate.tier_timeout(settings, "vision", "nvidia", live_gate.MUSE), 35)
        self.assertEqual(live_gate.tier_timeout(settings, "vision", "openrouter", live_gate.OR_QWEN), 15)
        self.assertEqual(live_gate.tier_timeout(settings, "router", "nvidia", live_gate.ULTRA), 12)
        self.assertIsNone(live_gate.tier_timeout(settings, "chat", "nvidia", live_gate.MUSE))
        broken = Settings(_env_file=None, model_roles="{not json")
        self.assertIsNone(live_gate.tier_timeout(broken, "vision", "nvidia", live_gate.MUSE))


class NvidiaCheckTests(unittest.TestCase):
    def test_ultra_plain_sends_the_bot_payload_and_passes_without_reasoning(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(200, json=_chat("台北 101 是地標。")))

        evidence = live_gate.check_ultra_plain(ctx)

        payload = recorder.payloads()[0]
        self.assertEqual(payload["model"], live_gate.ULTRA)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_budget", payload)
        # Ultra's model-card sampling, as the bot sends it (not CHAT_TEMPERATURE).
        self.assertEqual((payload["temperature"], payload["top_p"]), (1.0, 0.95))
        self.assertEqual(payload["max_tokens"], 2048)
        # ChatAgent's system prompt, then the user's text.
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "用一句話介紹台北 101。"})
        self.assertEqual(
            recorder.requests[0].headers["Authorization"], f"Bearer {_NVIDIA_KEY}"
        )
        self.assertIn("no reasoning", evidence)
        self.assertIn("traditional: yes", evidence)
        self.assertIn("markup: no", evidence)

    def test_ultra_plain_reports_markup_that_line_would_show_raw(self) -> None:
        ctx, _ = _context(
            lambda request, payload: httpx.Response(200, json=_chat("**台北 101** 高 \\( 508 \\) 公尺。"))
        )

        evidence = live_gate.check_ultra_plain(ctx)

        self.assertIn("markup: yes (stripped before LINE)", evidence)

    def test_ultra_plain_fails_when_reasoning_leaks(self) -> None:
        ctx, _ = _context(
            lambda request, payload: httpx.Response(200, json=_chat("答案", reasoning_content="想了很久"))
        )

        with self.assertRaisesRegex(live_gate.GateError, "reasoning present"):
            live_gate.check_ultra_plain(ctx)

    def test_ultra_plain_reports_http_errors_from_the_provider(self) -> None:
        ctx, _ = _context(
            lambda request, payload: httpx.Response(
                410, json={"detail": "end of life on 2026-07-27T00:00:00Z"}
            )
        )

        with self.assertRaisesRegex(live_gate.GateError, "^HTTP 410"):
            live_gate.check_ultra_plain(ctx)

    def test_ultra_tools_runs_the_bot_tool_loop(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if payload["messages"][-1]["role"] == "tool":
                return httpx.Response(200, json=_chat("現在台北時間是 15:04。"))
            return httpx.Response(200, json=_chat("", tool_calls=[_TOOL_CALL], finish_reason="tool_calls"))

        ctx, recorder = _context(responder)

        evidence = live_gate.check_ultra_tools(ctx)

        first, followup = recorder.payloads()
        tool_names = [tool["function"]["name"] for tool in first["tools"]]
        # The bot's real tool schemas ride along with the clock tool.
        self.assertEqual(
            sorted(tool_names),
            ["get_current_time", "recall_memory", "update_user_profile", "web_search"],
        )
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(first["chat_template_kwargs"], {"enable_thinking": False})
        # run_tool_loop replays the raw assistant turn, then the tool result.
        self.assertEqual(followup["messages"][-2]["role"], "assistant")
        self.assertEqual(followup["messages"][-2]["tool_calls"], [_TOOL_CALL])
        self.assertEqual(followup["messages"][-1]["tool_call_id"], "call_1")
        self.assertEqual(followup["messages"][-1]["name"], "get_current_time")
        self.assertIn("15:04", followup["messages"][-1]["content"])
        self.assertIn("uses tool result: yes", evidence)
        self.assertIn("4 tool schemas accepted", evidence)

    def test_ultra_tools_fails_when_the_model_never_calls_the_tool(self) -> None:
        ctx, _ = _context(lambda request, payload: httpx.Response(200, json=_chat("我不知道現在幾點。")))

        with self.assertRaisesRegex(live_gate.GateError, "no get_current_time call"):
            live_gate.check_ultra_tools(ctx)

    def test_super_router_runs_the_orchestrator_with_the_bot_json_schema(self) -> None:
        from src.agents.routing_rules import ROUTER_JSON_SCHEMA

        ctx, recorder = _context(
            lambda request, payload: httpx.Response(200, json=_chat(json.dumps(_ROUTE, ensure_ascii=False)))
        )

        evidence = live_gate.check_super_router(ctx)

        [payload] = recorder.payloads()
        self.assertEqual(payload["model"], live_gate.SUPER)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["temperature"], 0.1)
        # The router tier's own cap wins over ORCHESTRATOR_MAX_TOKENS.
        self.assertEqual(payload["max_tokens"], 256)
        # The live gate found json_schema works on Super and guided_json 400s.
        self.assertNotIn("nvext", payload)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], ROUTER_JSON_SCHEMA)
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": live_gate.ROUTER_USER_MESSAGE})
        self.assertEqual(evidence, "agent=web_search, output=text, thinking=off; parsed first try")

    def test_super_router_reports_recovery_by_the_repair_prompt(self) -> None:
        replies = iter(["我覺得應該查網路", json.dumps(_ROUTE)])
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(200, json=_chat(next(replies)))
        )

        evidence = live_gate.check_super_router(ctx)

        repair = recorder.payloads()[1]
        # NIM needs the last turn to be the user's: no assistant prefill.
        self.assertEqual(repair["messages"][-1]["role"], "user")
        self.assertIn("recovered by the repair prompt (2 requests)", evidence)

    def test_super_router_fails_when_nothing_parses_or_the_call_fails(self) -> None:
        ctx, _ = _context(lambda request, payload: httpx.Response(200, json=_chat("無法判斷")))
        with self.assertRaisesRegex(
            live_gate.GateError, r"could not parse LLM response after 2 request\(s\)"
        ):
            live_gate.check_super_router(ctx)

        ctx, _ = _context(lambda request, payload: httpx.Response(400, text="temperature must be > 0"))
        with self.assertRaisesRegex(live_gate.GateError, "^HTTP 400: temperature"):
            live_gate.check_super_router(ctx)

    def test_a_model_slower_than_its_tier_limit_fails_as_the_bots_chain_gives_up(self) -> None:
        # The bridge's client blocks; the fallback chain's asyncio.timeout
        # (the tier's timeout_s) must still fire while it waits.
        roles = {
            "router": [{"provider": "nvidia", "model": live_gate.SUPER, "json_mode": "json_schema", "timeout_s": 0.1}],
            "chat": [{"provider": "nvidia", "model": live_gate.ULTRA, "thinking": False, "timeout_s": 0.1}],
        }
        for check, reply in (
            (live_gate.check_super_router, _chat(json.dumps(_ROUTE))),
            (live_gate.check_ultra_tools, _chat("", tool_calls=[_TOOL_CALL], finish_reason="tool_calls")),
        ):
            with self.subTest(check=check.__name__):
                seen: list[float | None] = []

                def slow(request: httpx.Request, payload: dict, seen=seen, reply=reply) -> httpx.Response:
                    seen.append(request.extensions.get("timeout", {}).get("read"))
                    time.sleep(0.4)
                    return httpx.Response(200, json=reply)

                ctx, _ = _context(slow, settings=Settings(_env_file=None, model_roles=json.dumps(roles)))

                with self.assertRaisesRegex(live_gate.GateError, r"^request failed: .*0\.1s"):
                    check(ctx)
                # Each request gets the tier's limit too, as a vision check's does.
                self.assertEqual(seen, [0.1])

    def test_super_router_json_probe_adds_each_shape_to_the_bot_payload(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if "nvext" in payload:
                return httpx.Response(400, json={"detail": "unknown field nvext"})
            return httpx.Response(200, json=_chat(json.dumps(_ROUTE, ensure_ascii=False)))

        ctx, recorder = _context(responder)

        evidence = live_gate.check_super_router_json(ctx)

        self.assertIn("guided_json: FAIL HTTP 400", evidence)
        self.assertIn("json_schema: ok agent=web_search", evidence)
        guided, schema = recorder.payloads()
        # Each probe sends one JSON shape, never the bot's on top of it.
        self.assertNotIn("response_format", guided)
        self.assertNotIn("nvext", schema)
        # The probe checks the very schema the bot's router sends.
        from src.agents.routing_rules import ROUTER_JSON_SCHEMA

        self.assertEqual(guided["nvext"]["guided_json"], ROUTER_JSON_SCHEMA)
        self.assertEqual(schema["response_format"]["json_schema"]["schema"], ROUTER_JSON_SCHEMA)
        for payload in (guided, schema):
            # The router tier's own low temperature, with the card's top_p.
            self.assertEqual((payload["temperature"], payload["top_p"]), (0.1, 0.95))
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(payload["messages"][-1]["content"], live_gate.ROUTER_USER_MESSAGE)

    def test_ultra_thinking_runs_ultra_itself_with_thinking_on(self) -> None:
        answers = iter(["需要 2 小時。", "小明現在 12 歲。", "午後熱對流旺盛，所以常下雷陣雨。"])
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200, json=_chat(next(answers), reasoning_content="推理")
            )
        )

        evidence = live_gate.check_ultra_thinking(ctx)

        payloads = recorder.payloads()
        self.assertEqual(len(payloads), 3)
        for payload in payloads:
            # No model swap any more: Ultra thinks for itself, at medium
            # effort; a reasoning_budget is HTTP 400 there, so max_tokens
            # alone makes room for the thoughts.
            self.assertEqual(payload["model"], live_gate.ULTRA)
            self.assertNotIn("reasoning_budget", payload)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True, "medium_effort": True})
            self.assertEqual(payload["max_tokens"], 2048 + 2048)
        self.assertIn(f"served by {live_gate.ULTRA}", evidence)
        self.assertIn('chat_template_kwargs={"enable_thinking": true, "medium_effort": true}', evidence)
        self.assertIn("answers 3/3", evidence)
        self.assertIn("with reasoning 3/3", evidence)
        self.assertIn("expected 2/2", evidence)
        self.assertIn("(traditional) 3/3", evidence)

    def test_ultra_thinking_fails_without_reasoning(self) -> None:
        ctx, _ = _context(lambda request, payload: httpx.Response(200, json=_chat("需要 2 小時。")))

        with self.assertRaisesRegex(live_gate.GateError, "with reasoning 0/3"):
            live_gate.check_ultra_thinking(ctx)

    def test_ultra_thinking_fails_when_medium_effort_is_not_sent(self) -> None:
        # DEPLOY.md promises the check confirms medium_effort (5a review).
        profile = model_profiles.PROFILES[("nvidia", live_gate.ULTRA)]
        ctx, _ = _context(
            lambda request, payload: httpx.Response(200, json=_chat("需要 2 小時。", reasoning_content="推理"))
        )

        with patch.dict(
            model_profiles.PROFILES,
            {("nvidia", live_gate.ULTRA): dataclasses.replace(profile, thinking_template_kwargs=())},
        ):
            with self.assertRaisesRegex(live_gate.GateError, "medium_effort"):
                live_gate.check_ultra_thinking(ctx)

    def test_ultra_router_asks_for_the_json_in_the_prompt_only(self) -> None:
        reply = "路由結果如下：\n```json\n" + json.dumps(_ROUTE, ensure_ascii=False) + "\n```"
        ctx, recorder = _context(lambda request, payload: httpx.Response(200, json=_chat(reply)))

        evidence = live_gate.check_ultra_router(ctx)

        [payload] = recorder.payloads()
        self.assertEqual(payload["model"], live_gate.ULTRA)
        # No structured output on Ultra: neither response_format nor nvext.
        self.assertNotIn("response_format", payload)
        self.assertNotIn("nvext", payload)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((payload["temperature"], payload["max_tokens"]), (0.1, 256))
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": live_gate.ROUTER_USER_MESSAGE})
        self.assertEqual(evidence, "agent=web_search, output=text, thinking=off; parsed first try")

    def test_a_router_reply_that_ignores_the_schema_fails(self) -> None:
        # The orchestrator routes each to chat on the first try; the check must not pass them.
        template = {**_ROUTE, "agent": "chat", "task_description": "...", "reasoning": "..."}
        cases = {
            "{}": "missing agent,output_format",
            json.dumps({"agent": "foo"}): "missing output_format",
            json.dumps({**_ROUTE, "agent": "foo"}): "agent='foo' is not a router agent",
            json.dumps(template): "task_description='...' describes no task",
        }
        for check in (live_gate.check_ultra_router, live_gate.check_super_router):
            for reply, message in cases.items():
                with self.subTest(check=check.__name__, reply=reply):
                    ctx, recorder = _context(lambda request, payload, reply=reply: httpx.Response(200, json=_chat(reply)))
                    with self.assertRaisesRegex(
                        live_gate.GateError, "^route JSON breaks the schema: " + re.escape(message)
                    ) as caught:
                        check(ctx)
                    self.assertIn("(the bot would send it to chat)", str(caught.exception))
                    self.assertEqual(len(recorder.requests), 1)

    def test_ultra_router_reports_recovery_by_the_repair_prompt(self) -> None:
        replies = iter(["應該要查網路。", json.dumps(_ROUTE)])
        ctx, recorder = _context(lambda request, payload: httpx.Response(200, json=_chat(next(replies))))

        evidence = live_gate.check_ultra_router(ctx)

        repair = recorder.payloads()[1]
        self.assertEqual(repair["messages"][-1]["role"], "user")
        self.assertNotIn("response_format", repair)
        self.assertIn("recovered by the repair prompt (2 requests)", evidence)

    def test_muse_vision_sends_the_red_circle_with_reasoning_effort_none(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(200, json=_chat("圖片中是紅色的圓形。")))

        evidence = live_gate.check_muse_vision(ctx)

        payload = recorder.payloads()[0]
        self.assertEqual(payload["model"], live_gate.MUSE)
        # Thinking off is a top-level reasoning_effort, never chat_template_kwargs.
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("chat_template_kwargs", payload)
        # Its model default wins over VISION_TEMPERATURE.
        self.assertEqual(payload["temperature"], 0.6)
        image_url = _image_part(payload)["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(
            live_gate.sniff_image_mime(base64.b64decode(image_url.split(",", 1)[1])), "image/png"
        )
        self.assertIn("mentions red: yes", evidence)

    def test_omni_vision_sends_the_probes_body_with_the_red_circle(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(200, json=_chat("紅色的圓形。")))

        evidence = live_gate.check_omni_vision(ctx)

        [payload] = recorder.payloads()
        self.assertEqual(payload["model"], live_gate.OMNI)
        # Thinking off whatever the request, the tier's 0.2 and 1024, no top_p.
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((payload["temperature"], payload["max_tokens"]), (0.2, 1024))
        self.assertNotIn("top_p", payload)
        self.assertNotIn("reasoning_budget", payload)
        self.assertTrue(_image_part(payload)["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn("mentions red: yes", evidence)

    def test_a_busy_omni_worker_fails_its_check_with_the_503(self) -> None:
        busy = {"error": {"message": "ResourceExhausted: Worker local total request limit reached"}}
        ctx, recorder = _context(lambda request, payload: httpx.Response(503, json=busy))

        with self.assertRaisesRegex(live_gate.GateError, "HTTP 503: .*ResourceExhausted"):
            live_gate.check_omni_vision(ctx)
        self.assertEqual(len(recorder.requests), 1)

    def test_vision_checks_give_the_model_only_its_tiers_time(self) -> None:
        for check_id, limit in (
            ("nvidia.omni.vision", 15),
            ("nvidia.muse.vision", 35),
            # A tier check gets its tier's limit plus the gate's slack.
            (f"tier.vision.2.nvidia:{live_gate.OMNI}", 15 + live_gate.TIER_SLACK_SECONDS),
            (f"tier.vision.3.nvidia:{live_gate.MUSE}", 35 + live_gate.TIER_SLACK_SECONDS),
        ):
            with self.subTest(check=check_id):
                seen: list[float | None] = []

                def hung(request: httpx.Request, payload: dict, seen=seen) -> httpx.Response:
                    seen.append(request.extensions.get("timeout", {}).get("read"))
                    raise httpx.ReadTimeout("hung", request=request)

                ctx, _ = _context(hung)
                [result] = live_gate.run_checks(ctx, live_gate.select_checks("all", [check_id]))

                self.assertEqual(seen, [limit])
                self.assertFalse(result.passed)
                self.assertIn("timed out", result.evidence)

    def test_vision_checks_hold_the_whole_call_to_the_tier_limit(self) -> None:
        # Each request is within its own timeout, but the call as a whole
        # (a 202 poll sequence, say) must not run past the tier's limit.
        ctx, _ = _context(lambda request, payload: httpx.Response(200, json=_chat("紅色的圓形。")))
        clock = iter([100.0, 100.0 + 36.0])
        ctx.clock = lambda: next(clock)

        with self.assertRaisesRegex(live_gate.GateError, "past the tier's 35s limit"):
            live_gate.check_muse_vision(ctx)

    def test_a_poll_sequence_is_stopped_at_the_tier_limit(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(202, headers={"NVCF-REQID": "req-7"})
            return httpx.Response(202, headers={"NVCF-REQID": "req-7"})

        ctx, recorder = _context(responder)

        with patch.object(live_gate, "tier_timeout", lambda *args: 0.05), patch(
            "src.providers.openai_compat.NVCF_POLL_INTERVAL_SECONDS", 0.01
        ):
            with self.assertRaisesRegex(live_gate.GateError, "no answer within the tier's 0.05s limit"):
                live_gate.check_muse_vision(ctx)
        self.assertGreater(len(recorder.requests), 1)

    def test_muse_vision_fails_when_the_answer_misses_the_colour(self) -> None:
        ctx, _ = _context(lambda request, payload: httpx.Response(200, json=_chat("一個圓形。")))

        with self.assertRaisesRegex(live_gate.GateError, "mentions red: no"):
            live_gate.check_muse_vision(ctx)

    def test_embed_goes_through_the_app_service(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.01] * 2048}]}
            )
        )

        evidence = live_gate.check_embed(ctx)

        payload = recorder.payloads()[0]
        self.assertEqual(payload["model"], live_gate.EMBED)
        self.assertEqual(payload["input_type"], "query")
        self.assertNotIn("dimensions", payload)
        self.assertEqual(evidence, "api dim=2048; app vector dim=1024, norm=1.00")

        ctx, _ = _context(
            lambda request, payload: httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.01] * 1024}]}
            )
        )
        with self.assertRaisesRegex(live_gate.GateError, "expected 2048"):
            live_gate.check_embed(ctx)

    def test_bot_path_does_not_hide_a_pending_202(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(202, headers={"NVCF-REQID": "req-1"})
        )

        [result] = live_gate.run_checks(ctx, [_check("nvidia.embed.dim")])

        self.assertFalse(result.passed)
        # EmbeddingService does not poll, so the pending job is a failure.
        self.assertIn("HTTP 202 (NVCF job still pending)", result.evidence)
        self.assertEqual(len(recorder.requests), 1)

    def test_bot_chat_path_polls_a_pending_202_like_the_app(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(202, headers={"NVCF-REQID": "req-9"})
            return httpx.Response(200, json=_chat("台北 101 是地標。"))

        ctx, recorder = _context(responder)

        with patch("src.providers.openai_compat.NVCF_POLL_INTERVAL_SECONDS", 0.0):
            evidence = live_gate.check_ultra_plain(ctx)

        self.assertIn("no reasoning", evidence)
        self.assertEqual([request.method for request in recorder.requests], ["POST", "GET"])
        self.assertEqual(
            str(recorder.requests[1].url), "https://integrate.api.nvidia.com/v1/status/req-9"
        )

    def test_flux_klein_uses_its_own_schema_and_reports_mime(self) -> None:
        jpeg = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 2048).decode()
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200, json={"artifacts": [{"base64": jpeg, "finishReason": "SUCCESS"}]}
            )
        )

        evidence = live_gate.check_flux_klein(ctx)

        self.assertEqual(
            str(recorder.requests[0].url),
            "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b",
        )
        payload = recorder.payloads()[0]
        # IMAGE_GEN_STEPS=50 is clamped to klein's 1-4.
        self.assertEqual(payload["steps"], 4)
        for field in ("cfg_scale", "mode", "samples", "negative_prompt"):
            self.assertNotIn(field, payload)
        self.assertIn("image/jpeg", evidence)

    def test_flux_dev_success_sends_the_dev_schema(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200, json={"artifacts": [{"base64": _png_base64(), "finishReason": "SUCCESS"}]}
            )
        )

        evidence = live_gate.check_flux_dev(ctx)

        payload = recorder.payloads()[0]
        self.assertEqual(payload["mode"], "base")
        self.assertEqual(payload["cfg_scale"], 5.0)
        self.assertEqual(payload["samples"], 1)
        self.assertEqual(payload["steps"], 50)
        self.assertEqual((payload["width"], payload["height"]), (1024, 1024))
        self.assertEqual(evidence, "image/png, 2 KB, finishReason=SUCCESS")

    def test_flux_error_finish_reason_fails(self) -> None:
        ctx, _ = _context(
            lambda request, payload: httpx.Response(
                200, json={"artifacts": [{"base64": "AAAA", "finishReason": "ERROR"}]}
            )
        )

        with self.assertRaisesRegex(live_gate.GateError, "finishReason=ERROR"):
            live_gate.check_flux_dev(ctx)

    def test_probe_polls_a_pending_202_until_the_result_arrives(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(202, headers={"NVCF-REQID": "req-1"})
            return httpx.Response(200, json=_chat(json.dumps(_ROUTE)))

        ctx, recorder = _context(responder)

        [result] = live_gate.run_checks(ctx, [_check("nvidia.super.router_json")])

        self.assertTrue(result.passed, result.evidence)
        self.assertIn("polled 2x", result.evidence)
        self.assertEqual(
            str(recorder.requests[1].url), "https://integrate.api.nvidia.com/v1/status/req-1"
        )


class OpenRouterCheckTests(unittest.TestCase):
    def test_qwen_passes_when_effort_none_yields_no_reasoning_tokens(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200,
                json=_chat(
                    "颱風是強烈的熱帶氣旋。",
                    usage={"completion_tokens_details": {"reasoning_tokens": 0}},
                ),
            )
        )

        evidence = live_gate.check_openrouter_qwen_no_reasoning(ctx)

        payload = recorder.payloads()[0]
        self.assertEqual(payload["model"], live_gate.OR_QWEN)
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(recorder.requests[0].headers["Authorization"], f"Bearer {_OPENROUTER_KEY}")
        self.assertIn("HTTP-Referer", recorder.requests[0].headers)
        self.assertIn("reasoning_tokens=0", evidence)

    def test_qwen_failure_also_reports_the_enabled_false_alternative(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            tokens = 0 if payload["reasoning"] == {"enabled": False} else 150
            return httpx.Response(
                200,
                json=_chat("答案", usage={"completion_tokens_details": {"reasoning_tokens": tokens}}),
            )

        ctx, _ = _context(responder)

        with self.assertRaisesRegex(
            live_gate.GateError, r"reasoning_tokens=150.*enabled:false -> reasoning_tokens=0"
        ):
            live_gate.check_openrouter_qwen_no_reasoning(ctx)

    def test_qwen_thinking_sends_the_bot_reasoning_request(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200,
                json=_chat(
                    "兩管同時打開需要 2 小時。",
                    reasoning="1/6 + 1/3 = 1/2",
                    usage={"completion_tokens_details": {"reasoning_tokens": 321}},
                ),
            )
        )

        evidence = live_gate.check_openrouter_qwen_thinking(ctx)

        payload = recorder.payloads()[0]
        # The thinking role's effort, one qwen3.8-27b:free supports (xhigh/medium/low).
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertNotIn("include_reasoning", payload)
        self.assertEqual(payload["max_tokens"], 4096 + 2048)
        self.assertIn("effort=medium", evidence)
        self.assertIn("reasoning_tokens=321", evidence)
        self.assertIn("expected answer: yes", evidence)

    def test_qwen_thinking_uses_the_configured_budget(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200,
                json=_chat("2 小時", usage={"completion_tokens_details": {"reasoning_tokens": 9}}),
            ),
            settings=Settings(_env_file=None, nvidia_thinking_budget=1024),
        )

        live_gate.check_openrouter_qwen_thinking(ctx)

        self.assertEqual(recorder.payloads()[0]["max_tokens"], 1024 + 2048)

    def test_qwen_thinking_fails_without_reasoning_or_on_a_rejected_effort(self) -> None:
        ctx, _ = _context(
            lambda request, payload: httpx.Response(
                200,
                json=_chat("2 小時", usage={"completion_tokens_details": {"reasoning_tokens": 0}}),
            )
        )
        with self.assertRaisesRegex(live_gate.GateError, "no reasoning despite thinking on"):
            live_gate.check_openrouter_qwen_thinking(ctx)

        ctx, _ = _context(
            lambda request, payload: httpx.Response(400, text="effort 'high' is not supported")
        )
        with self.assertRaisesRegex(live_gate.GateError, "^HTTP 400: effort"):
            live_gate.check_openrouter_qwen_thinking(ctx)

    def test_ultra_free_chat_and_thinking_send_their_efforts(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(
                200,
                json=_chat(
                    "兩管同時打開需要 2 小時。",
                    reasoning="1/6 + 1/3 = 1/2" if payload["reasoning"]["effort"] != "none" else None,
                    usage={
                        "completion_tokens_details": {
                            "reasoning_tokens": 0 if payload["reasoning"]["effort"] == "none" else 40
                        }
                    },
                ),
            )
        )

        plain = live_gate.check_openrouter_ultra_no_reasoning(ctx)
        thinking = live_gate.check_openrouter_ultra_thinking(ctx)

        chat, deep = recorder.payloads()
        self.assertEqual((chat["model"], chat["reasoning"]), (live_gate.OR_ULTRA, {"effort": "none"}))
        self.assertEqual((deep["model"], deep["reasoning"]), (live_gate.OR_ULTRA, {"effort": "medium"}))
        self.assertIn("reasoning_tokens=0", plain)
        self.assertIn("effort=medium", thinking)

    def test_super_free_router_runs_the_orchestrator_with_json_object(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(200, json=_chat(json.dumps(_ROUTE))))

        evidence = live_gate.check_openrouter_super_router(ctx)

        [payload] = recorder.payloads()
        self.assertEqual(payload["model"], live_gate.OR_SUPER)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        # It reasons by default, so the router's thinking-off is explicit.
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertIn("agent=web_search", evidence)

    def test_an_upstream_429_is_reported_as_such(self) -> None:
        body = {"error": {"code": 429, "metadata": {"raw": "qwen is temporarily rate-limited upstream."}}}
        ctx, _ = _context(lambda request, payload: httpx.Response(429, json=body))

        with self.assertRaisesRegex(live_gate.GateError, r"^HTTP 429 \(rate limited upstream"):
            live_gate.check_openrouter_qwen_no_reasoning(ctx)

        ctx, _ = _context(
            lambda request, payload: httpx.Response(429, text="Rate limit exceeded: free-models-per-day")
        )
        with self.assertRaisesRegex(
            live_gate.GateError, r"^HTTP 429 \(rate limited: OpenRouter's free-models-per-day cap\)$"
        ):
            live_gate.check_openrouter_qwen_no_reasoning(ctx)

        ctx, _ = _context(
            lambda request, payload: httpx.Response(429, text="Rate limit exceeded: free-models-per-min")
        )
        with self.assertRaisesRegex(live_gate.GateError, r"^HTTP 429 \(rate limited\)$"):
            live_gate.check_openrouter_qwen_no_reasoning(ctx)

    def test_qwen_tools_runs_the_bot_tool_loop_on_openrouter(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if payload["messages"][-1]["role"] == "tool":
                return httpx.Response(200, json=_chat("台北現在 15:04。"))
            return httpx.Response(200, json=_chat(None, tool_calls=[_TOOL_CALL]))

        ctx, recorder = _context(responder)

        evidence = live_gate.check_openrouter_qwen_tools(ctx)

        first, followup = recorder.payloads()
        self.assertEqual(first["model"], live_gate.OR_QWEN)
        self.assertEqual(first["reasoning"], {"effort": "none"})
        self.assertEqual(followup["messages"][-1]["role"], "tool")
        self.assertIn("uses tool result: yes", evidence)

    def test_a_daily_cap_met_inside_the_tool_loop_stops_the_later_openrouter_checks(self) -> None:
        cap = '{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 credits","code":429}}'
        ctx, recorder = _context(lambda request, payload: httpx.Response(429, text=cap))

        tools, later = live_gate.run_checks(
            ctx, [_check("openrouter.qwen.tools"), _check("openrouter.ultra.no_reasoning")]
        )

        self.assertFalse(tools.passed or tools.skipped)
        self.assertEqual(tools.evidence, "HTTP 429 (rate limited: OpenRouter's free-models-per-day cap)")
        self.assertTrue(later.skipped)
        self.assertEqual(later.evidence, live_gate.DAILY_CAP_NOTE)
        self.assertEqual(len(recorder.requests), 1)

    def test_qwen_vision_sends_the_vision_agent_messages_with_effort_none(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(200, json=_chat("紅色的圓形")))

        evidence = live_gate.check_openrouter_qwen_vision(ctx)

        payload = recorder.payloads()[0]
        self.assertEqual(payload["model"], live_gate.OR_QWEN)
        # qwen3.8 reasons at xhigh unless told otherwise; the vision tier says none.
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertTrue(_image_part(payload)["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn("mentions red: yes", evidence)

        ctx, _ = _context(lambda request, payload: httpx.Response(200, json=_chat("A blue square.")))
        with self.assertRaisesRegex(live_gate.GateError, "mentions red: no"):
            live_gate.check_openrouter_qwen_vision(ctx)


class RoleCheckTests(unittest.TestCase):
    def test_every_role_has_a_check_on_the_roles_path(self) -> None:
        from src.runtime.model_roles import ROLES

        self.assertEqual(
            [check.check_id for check in live_gate.select_checks("roles")],
            [f"roles.{role}" for role in ROLES],
        )

    def test_role_checks_call_each_roles_first_enabled_non_colab_tier(self) -> None:
        jpeg = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 64).decode()

        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 2048}]})
            if "/genai/" in request.url.path:
                return httpx.Response(200, json={"artifacts": [{"base64": jpeg, "finishReason": "SUCCESS"}]})
            if payload.get("response_format"):
                return httpx.Response(200, json=_chat(json.dumps(_ROUTE)))
            return httpx.Response(200, json=_chat("台北 101 是地標。", reasoning_content="想"))

        ctx, recorder = _context(responder)

        results = live_gate.run_checks(ctx, live_gate.select_checks("roles"))

        self.assertTrue(all(result.passed for result in results), [r.evidence for r in results])
        evidence = {result.check_id: result.evidence for result in results}
        self.assertTrue(evidence["roles.router"].startswith(f"nvidia:{live_gate.SUPER}: agent=web_search"))
        self.assertTrue(evidence["roles.chat"].startswith(f"nvidia:{live_gate.ULTRA}"))
        self.assertTrue(evidence["roles.thinking"].startswith(f"nvidia:{live_gate.ULTRA}"))
        self.assertTrue(evidence["roles.vision"].startswith(f"nvidia:{live_gate.OMNI}"))
        self.assertTrue(evidence["roles.utility"].startswith(f"nvidia:{live_gate.SUPER}"))
        self.assertEqual(evidence["roles.embed"], f"nvidia:{live_gate.EMBED}: vector dim=1024")
        self.assertEqual(evidence["roles.image"], f"nvidia:{live_gate.FLUX_KLEIN}: image returned")
        chat_payloads = [
            payload for payload in recorder.payloads() if payload.get("model") == live_gate.ULTRA
        ]
        # chat with thinking off, then the thinking role with it on.
        self.assertEqual(
            [payload["chat_template_kwargs"]["enable_thinking"] for payload in chat_payloads],
            [False, True],
        )
        router = next(payload for payload in recorder.payloads() if payload.get("response_format"))
        self.assertEqual(router["response_format"]["type"], "json_schema")

    def test_role_checks_follow_force_tier(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(200, json=_chat("好的。")),
            settings=Settings(_env_file=None, force_tier="openrouter"),
        )

        [result] = live_gate.run_checks(ctx, [_check("roles.chat")])

        self.assertTrue(result.passed, result.evidence)
        self.assertTrue(result.evidence.startswith(f"openrouter:{live_gate.OR_QWEN}"))
        self.assertEqual(recorder.payloads()[0]["model"], live_gate.OR_QWEN)

    def test_role_checks_resolve_their_api_for_the_cap_and_the_pacing(self) -> None:
        with_keys, _ = _context(_tier_reply)
        without_nvidia, _ = _context(_tier_reply, nvidia_key="")
        self.assertEqual(_check("roles.chat").provider_in(with_keys), "nvidia")
        self.assertEqual(_check("roles.chat").provider_in(without_nvidia), "openrouter")
        # No usable tier: no API to pace; the check itself says why.
        self.assertEqual(_check("roles.embed").provider_in(without_nvidia), "")

        # Without the NVIDIA key every role check calls OpenRouter: spaced
        # within its per-minute limit, and stopped by its daily cap.
        clock = _FakeClock()
        sent: list[float] = []

        def paced(request: httpx.Request, payload: dict) -> httpx.Response:
            sent.append(clock.now)
            return _tier_reply(request, payload)

        ctx, _ = _context(paced, nvidia_key="")
        ctx.clock, ctx.sleep = clock, clock.sleep
        results = live_gate.run_checks(ctx, [_check("roles.router"), _check("roles.chat")])
        self.assertTrue(all(result.passed for result in results), [r.evidence for r in results])
        self.assertEqual(len(sent), 2)
        self.assertGreaterEqual(sent[1] - sent[0], 60 / 20 - 1e-9)

        cap = '{"error":{"message":"Rate limit exceeded: free-models-per-day","code":429}}'
        ctx, recorder = _context(lambda request, payload: httpx.Response(429, text=cap), nvidia_key="")
        router, chat, embed = live_gate.run_checks(
            ctx, [_check("roles.router"), _check("roles.chat"), _check("roles.embed")]
        )
        self.assertIn("free-models-per-day", router.evidence)
        self.assertTrue(chat.skipped)
        self.assertEqual(chat.evidence, live_gate.DAILY_CAP_NOTE)
        self.assertFalse(embed.passed or embed.skipped)
        self.assertIn("no enabled non-Colab tier", embed.evidence)
        self.assertEqual(len(recorder.requests), 1)

    def test_a_role_without_a_usable_tier_fails(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(500), nvidia_key="")

        [result] = live_gate.run_checks(ctx, [_check("roles.embed")])

        self.assertFalse(result.passed)
        self.assertIn("no enabled non-Colab tier", result.evidence)
        self.assertEqual(recorder.requests, [])

    def test_the_embed_role_check_reports_the_http_status(self) -> None:
        ctx, _ = _context(lambda request, payload: httpx.Response(410, text="end of life"))

        [result] = live_gate.run_checks(ctx, [_check("roles.embed")])

        self.assertFalse(result.passed)
        self.assertTrue(result.evidence.startswith("HTTP 410"), result.evidence)


_COLAB_CHECKS = ["colab.probe", "colab.qwen.chat", "colab.qwen.vision", "colab.nimble.router"]
_COLAB_ON = {
    "colab_enabled": True,
    "firestore_enabled": True,
    "colab_shared_secret": colab_fakes.SECRET,
}


def _colab_reply(request: httpx.Request, payload: dict) -> httpx.Response:
    """qwen3.8 on /api/chat: a red circle for an image, a zh-TW sentence otherwise."""
    looked = any(message.get("images") for message in payload["messages"])
    content = "圖片中央是一個紅色的圓形。" if looked else "台北 101 是台北最有名的摩天大樓。"
    return httpx.Response(
        200,
        json={
            "model": colab_fakes.QWEN,
            "message": {"role": "assistant", "content": content, "thinking": "先確認問題。"},
            "done": True,
            "done_reason": "stop",
        },
    )


def _tunnel(**overrides: object) -> Responder:
    """The session's tunnel (auth proxy + Ollama), as tests/colab_fakes scripts it."""
    routes: dict = colab_fakes.healthy_routes()
    routes["POST /api/chat"] = _colab_reply
    routes["POST /v1/systemone"] = (200, colab_fakes.systemone_answer(agent="web_search"))
    routes.update(overrides)

    def responder(request: httpx.Request, payload: dict) -> httpx.Response:
        answer = routes.get(f"{request.method} {request.url.path}", (404, {"error": "not found"}))
        if callable(answer):
            return answer(request, payload)
        status, body = answer
        return httpx.Response(status, json=body)

    return responder


class _Loader:
    """Stands in for the Firestore read of the session document."""

    def __init__(self, answer: ColabState | Exception) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, settings: Settings) -> ColabState:
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _colab_run(
    responder: Responder, state: ColabState | Exception, **settings: object
) -> tuple[list[live_gate.CheckResult], _Recorder, _Loader]:
    ctx, recorder = _context(responder, settings=Settings(_env_file=None, **{**_COLAB_ON, **settings}))
    loader = _Loader(state)
    ctx.colab_loader = loader
    return live_gate.run_checks(ctx, live_gate.select_checks("colab")), recorder, loader


class ColabPathTests(unittest.TestCase):
    def test_every_colab_check_skips_while_colab_is_off(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(500))
        loader = _Loader(colab_fakes.ready_state())
        ctx.colab_loader = loader

        results = live_gate.run_checks(ctx, live_gate.select_checks("colab"))

        self.assertEqual([result.check_id for result in results], _COLAB_CHECKS)
        for result in results:
            self.assertTrue(result.skipped)
            self.assertFalse(result.passed)
            self.assertEqual(result.evidence, "COLAB_ENABLED is off")
        self.assertEqual(recorder.requests, [])
        self.assertEqual(loader.calls, 0)
        self.assertIn("0/0 checks passed, 4 skipped", live_gate.format_table(results))

    def test_without_a_ready_session_every_check_skips_after_one_read(self) -> None:
        state = ColabState(status="BOOTING", nbh=colab_fakes.NBH, last_error="pull: digest mismatch")

        results, recorder, loader = _colab_run(_tunnel(), state)

        self.assertTrue(all(result.skipped for result in results))
        self.assertIn(
            "no READY Colab session (status BOOTING; last error: pull: digest mismatch); send !colab on",
            results[0].evidence,
        )
        self.assertEqual(loader.calls, 1)
        self.assertEqual(recorder.requests, [])

    def test_an_unreadable_state_skips_and_names_the_fix(self) -> None:
        error = ColabStateError("reading linebot_colab/state failed: DefaultCredentialsError: none found")

        results, recorder, loader = _colab_run(_tunnel(), error)

        self.assertTrue(all(result.skipped for result in results))
        self.assertIn("ColabStateError", results[0].evidence)
        self.assertIn("gcloud auth application-default login", results[0].evidence)
        self.assertEqual(loader.calls, 1)
        self.assertEqual(recorder.requests, [])

    def test_a_ready_session_passes_every_check_as_the_bot_calls_it(self) -> None:
        results, recorder, _ = _colab_run(_tunnel(), colab_fakes.ready_state())

        self.assertEqual([result.check_id for result in results], _COLAB_CHECKS)
        self.assertTrue(all(result.passed for result in results), [r.evidence for r in results])
        probe, chat, vision, router = (result.evidence for result in results)
        self.assertIn(f"Ollama {live_gate.protocol.OLLAMA_VERSION}", probe)
        self.assertIn(
            f"loaded wholly in VRAM ({colab_fakes.QWEN} 24128 MiB, {colab_fakes.NIMBLE} 10110 MiB)", probe
        )
        self.assertIn("think=low", chat)
        self.assertIn("CJK", chat)
        self.assertIn("mentions red: yes", vision)
        self.assertIn("1 image(s) as raw base64", vision)
        self.assertIn("agent=web_search (0.90)", router)
        self.assertIn("decides web_search", router)
        # Everything went to the session's tunnel with its derived bearer,
        # and neither shows up in the report.
        bearer = colab_fakes.bearer()
        for request in recorder.requests:
            self.assertEqual(request.url.host, colab_fakes.TUNNEL_HOST)
            self.assertEqual(request.headers["authorization"], f"Bearer {bearer}")
        table = live_gate.format_table(results)
        for secret in (bearer, colab_fakes.SECRET, colab_fakes.TUNNEL_HOST):
            self.assertNotIn(secret, table)
        # The bodies are the bot's own: think from COLAB_CHAT_THINK, no
        # load-time options, the image as raw base64.
        chats = [json.loads(r.content) for r in recorder.requests if r.url.path == "/api/chat"]
        self.assertEqual([body["think"] for body in chats], ["low", "low"])
        for body in chats:
            self.assertEqual((body["stream"], body["keep_alive"]), (False, -1))
            self.assertFalse({"num_ctx", "draft_num_predict"} & set(body["options"]))
        [image] = [m["images"][0] for m in chats[1]["messages"] if m.get("images")]
        self.assertFalse(image.startswith("data:"))
        [routed] = [json.loads(r.content) for r in recorder.requests if r.url.path == "/v1/systemone"]
        self.assertEqual(routed["model"], colab_fakes.NIMBLE)
        self.assertEqual(routed["state"]["message"], live_gate.ROUTER_USER_MESSAGE)

    def test_a_tunnel_url_that_breaks_the_rule_is_never_called(self) -> None:
        state = colab_fakes.ready_state(tunnel_url="https://evil.example.com")

        results, recorder, _ = _colab_run(_tunnel(), state)

        for result in results:
            self.assertFalse(result.passed or result.skipped)
            self.assertIn("breaks the tunnel rule", result.evidence)
        self.assertEqual(recorder.requests, [])

    def test_failures_are_reported_and_redacted(self) -> None:
        bearer = colab_fakes.bearer()
        wrong_tags = (200, {"models": [{"name": colab_fakes.QWEN, "model": colab_fakes.QWEN, "digest": "0" * 64}]})
        refused = (401, {"error": f"bad bearer {bearer} for {colab_fakes.TUNNEL_HOST}"})

        results, _, _ = _colab_run(_tunnel(**{"GET /api/tags": wrong_tags, "POST /api/chat": refused}), colab_fakes.ready_state())

        probe, chat, vision, _ = results
        self.assertFalse(probe.passed)
        self.assertIn("digest", probe.evidence.lower())
        for result in (chat, vision):
            self.assertFalse(result.passed)
            self.assertTrue(result.evidence.startswith("HTTP 401"), result.evidence)
            self.assertNotIn(bearer, result.evidence)
            self.assertNotIn(colab_fakes.TUNNEL_HOST, result.evidence)
            self.assertIn("***", result.evidence)

    def test_an_unsure_colab_router_passes_and_says_the_api_router_decides(self) -> None:
        unsure = (200, colab_fakes.systemone_answer(agent="chat", agent_confidence=0.41))

        results, _, _ = _colab_run(_tunnel(**{"POST /v1/systemone": unsure}), colab_fakes.ready_state())

        router = results[-1]
        self.assertTrue(router.passed, router.evidence)
        self.assertIn("agent=chat (expected web_search) (0.41)", router.evidence)
        self.assertIn("below the 0.75 gate, the API router decides", router.evidence)

    def test_only_the_routing_call_counts_against_nimbles_limit(self) -> None:
        clock = _FakeClock()
        real_bot_roles = live_gate.bot_roles

        @contextlib.asynccontextmanager
        async def slow_setup(*args: object, **kwargs: object):
            clock.now += 10.0  # building the clients, the endpoint and the table
            async with real_bot_roles(*args, **kwargs) as table:
                yield table

        def slow_route(request: httpx.Request, payload: dict) -> httpx.Response:
            clock.now += 7.0
            return httpx.Response(200, json=colab_fakes.systemone_answer(agent="web_search"))

        nimble_ids = ["colab.nimble.router", f"tier.router.1.colab:{live_gate.COLAB_NIMBLE}"]
        for route, passes in ((_tunnel(), True), (_tunnel(**{"POST /v1/systemone": slow_route}), False)):
            settings = Settings(_env_file=None, **_COLAB_ON)
            ctx, _ = _context(route, settings=settings)
            ctx.colab_loader = _Loader(colab_fakes.ready_state())
            ctx.clock = clock
            with patch.object(live_gate, "bot_roles", slow_setup):
                results = live_gate.run_checks(ctx, live_gate.select_checks("all", nimble_ids, settings))
            self.assertEqual([result.check_id for result in results], nimble_ids)
            for result in results:
                with self.subTest(check=result.check_id, passes=passes):
                    self.assertEqual(result.passed, passes, result.evidence)
                    if not passes:
                        self.assertIn(f"answered in 7.0s, past {live_gate.COLAB_NIMBLE}'s 6s limit", result.evidence)

    def test_a_malformed_colab_router_answer_fails(self) -> None:
        broken = (200, {"model": colab_fakes.NIMBLE, "answers": {"agent": {"choice": "dance", "confidence": 1}}})

        results, _, _ = _colab_run(_tunnel(**{"POST /v1/systemone": broken}), colab_fakes.ready_state())

        self.assertFalse(results[-1].passed)
        self.assertIn("malformed agent answer", results[-1].evidence)

    def test_a_router_not_wholly_in_vram_fails_the_probe(self) -> None:
        pins = live_gate.protocol.pinned_models()
        running = {
            "models": [
                {"name": model, "model": model, **colab_fakes.on_gpu(model, 5 << 30 if model == colab_fakes.NIMBLE else None)}
                for model in pins
            ]
        }

        results, _, _ = _colab_run(_tunnel(**{"GET /api/ps": (200, running)}), colab_fakes.ready_state())

        self.assertFalse(results[0].passed)
        self.assertIn(f"{colab_fakes.NIMBLE} not fully on the GPU (5120 of 10110 MiB in VRAM)", results[0].evidence)

    def test_a_dry_run_session_runs_only_the_probe(self) -> None:
        results, recorder, _ = _colab_run(_tunnel(), colab_fakes.ready_state(), colab_dry_run=True)

        self.assertTrue(results[0].passed, results[0].evidence)
        for result in results[1:]:
            self.assertTrue(result.skipped)
            self.assertIn("COLAB_DRY_RUN=true", result.evidence)
        self.assertEqual(
            {request.url.path for request in recorder.requests},
            {"/api/version", "/api/ps", "/api/tags", "/api/show"},
        )

    def test_a_model_check_whose_tier_is_forced_off_skips(self) -> None:
        ctx, recorder = _context(
            lambda request, payload: httpx.Response(500),
            settings=Settings(_env_file=None, force_tier="openrouter"),
        )

        [result] = live_gate.run_checks(ctx, [_check("nvidia.ultra.plain")])

        self.assertTrue(result.skipped)
        self.assertIn("FORCE_TIER=openrouter", result.evidence)
        self.assertEqual(recorder.requests, [])


class RunnerTests(unittest.TestCase):
    def test_missing_key_fails_without_sending_a_request(self) -> None:
        ctx, recorder = _context(lambda request, payload: httpx.Response(500), nvidia_key="")

        [result] = live_gate.run_checks(ctx, [_check("nvidia.ultra.plain")])

        self.assertFalse(result.passed)
        self.assertEqual(result.evidence, "NVIDIA_API_KEY is not set")
        self.assertEqual(recorder.requests, [])

    def test_error_bodies_are_redacted_and_shortened(self) -> None:
        body = f"invalid key {_NVIDIA_KEY} " + "x" * 500
        ctx, _ = _context(lambda request, payload: httpx.Response(401, text=body))

        [result] = live_gate.run_checks(ctx, [_check("nvidia.embed.dim")])

        self.assertFalse(result.passed)
        self.assertTrue(result.evidence.startswith("HTTP 401"))
        self.assertNotIn(_NVIDIA_KEY, result.evidence)
        self.assertLessEqual(len(result.evidence), 200)

    def _run_main(self, env_text: str, argv: list[str], responder: Responder) -> tuple[int, str, str, _Recorder]:
        recorder = _Recorder(responder)
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "gate.env"
            env_file.write_text(env_text, encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with (
                patch.dict(os.environ),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                for name in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_REASONING_EFFORT", "LOG_LEVEL"):
                    os.environ.pop(name, None)
                code = live_gate.main(
                    ["--env-file", str(env_file), *argv],
                    client=httpx.Client(transport=httpx.MockTransport(recorder)),
                )
        return code, out.getvalue(), err.getvalue(), recorder

    def test_main_prints_a_table_and_exits_1_on_any_failure(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 2048}]})
            return httpx.Response(410, json={"detail": "end of life on 2026-07-27T00:00:00Z"})

        code, table, err, _ = self._run_main(
            f"NVIDIA_API_KEY={_NVIDIA_KEY}\nOPENROUTER_API_KEY={_OPENROUTER_KEY}\n",
            ["--only", "nvidia.embed,nvidia.ultra.plain"],
            responder,
        )

        self.assertEqual(code, 1)
        self.assertIn("nvidia.embed.dim", table)
        self.assertIn("PASS", table)
        self.assertIn("FAIL", table)
        self.assertIn("1/2 checks passed", table)
        self.assertNotIn(_NVIDIA_KEY, table + err)

    def test_main_applies_the_env_file_settings_to_the_bot_clients(self) -> None:
        code, _, err, recorder = self._run_main(
            f"OPENROUTER_API_KEY={_OPENROUTER_KEY}\nNVIDIA_THINKING_BUDGET=1024\n"
            'MODEL_ROLES=\'{"chat": [{"provider": "nvidia", "model": "example/pinned-model"}]}\'\n',
            ["--only", "openrouter.qwen.thinking"],
            lambda request, payload: httpx.Response(
                200, json=_chat("2", usage={"completion_tokens_details": {"reasoning_tokens": 5}})
            ),
        )

        self.assertEqual(code, 0)
        self.assertEqual(recorder.payloads()[0]["max_tokens"], 1024 + 2048)
        # Overrides are named, never echoed.
        self.assertIn("MODEL_ROLES", err)
        self.assertNotIn("example/pinned-model", err)

    def test_main_exits_0_when_everything_selected_is_skipped(self) -> None:
        code, table, _, recorder = self._run_main(
            "", ["--path", "colab"], lambda request, payload: httpx.Response(500)
        )

        self.assertEqual(code, 0)
        for check_id in _COLAB_CHECKS:
            self.assertIn(check_id, table)
        self.assertIn("0/0 checks passed, 4 skipped", table)
        self.assertEqual(recorder.requests, [])

    def test_main_runs_the_colab_checks_against_a_ready_session(self) -> None:
        recorder = _Recorder(_tunnel())
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "gate.env"
            env_file.write_text(
                f"COLAB_ENABLED=true\nFIRESTORE_ENABLED=true\nCOLAB_SHARED_SECRET={colab_fakes.SECRET}\n",
                encoding="utf-8",
            )
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = live_gate.main(
                    ["--env-file", str(env_file), "--path", "colab"],
                    client=httpx.Client(transport=httpx.MockTransport(recorder)),
                    colab_loader=_Loader(colab_fakes.ready_state()),
                )

        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("4/4 checks passed", out.getvalue())
        self.assertNotIn(colab_fakes.SECRET, out.getvalue() + err.getvalue())

    def test_main_refuses_invalid_settings_naming_fields_only(self) -> None:
        code, table, err, recorder = self._run_main(
            f"NVIDIA_API_KEY={_NVIDIA_KEY}\nLOG_LEVEL=CANARY_VALUE\n",
            ["--only", "nvidia.embed"],
            lambda request, payload: httpx.Response(500),
        )

        self.assertEqual(code, 2)
        self.assertIn("log_level", err)
        self.assertNotIn("CANARY_VALUE", err + table)
        self.assertEqual(recorder.requests, [])

    def test_main_restores_the_app_log_level(self) -> None:
        app_logger = logging.getLogger("linebot")
        before = app_logger.level

        self._run_main(
            f"NVIDIA_API_KEY={_NVIDIA_KEY}\n",
            ["--only", "nvidia.embed"],
            lambda request, payload: httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.1] * 2048}]}
            ),
        )

        self.assertEqual(app_logger.level, before)

    def test_real_environment_variables_win_over_the_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "gate.env"
            env_file.write_text(
                "NVIDIA_API_KEY=from-file\nOPENROUTER_API_KEY=your_openrouter_api_key\n",
                encoding="utf-8",
            )

            keys = live_gate.load_keys(env_file, environ={"NVIDIA_API_KEY": "from-env"})

        self.assertEqual(keys, {"nvidia": "from-env", "openrouter": ""})


# ── --path tiers: one call per tier of the configured role table ──

_JPEG = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 4096).decode()
_TIER_ANSWERS = {
    "vision": "圖片中央是一個紅色的圓形。",
    "utility": "使用者正在學鋼琴（每週六上課），並計畫年底前去京都賞楓，預算約五萬元。",
    "thinking": "兩管同時打開需要 2 小時。",
    "chat": "台北 101 是台北信義區的地標摩天大樓。",
}


def _tier_kind(payload: dict) -> str:
    """Which role's request *payload* is (the gate's own messages tell them apart)."""
    messages = payload.get("messages") or []
    if any(isinstance(m.get("content"), list) for m in messages):
        return "vision"
    if messages and messages[0].get("content") == memory_service._SUMMARY_SYSTEM_PROMPT:
        return "utility"
    last = messages[-1].get("content") if messages else ""
    if last == live_gate.ROUTER_USER_MESSAGE:
        return "router"
    if "水池" in str(last):
        return "thinking"
    return "chat"


def _tier_reply(request: httpx.Request, payload: dict) -> httpx.Response:
    """Every API role answered well: what a healthy line-up returns."""
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 2048}]})
    if "/genai/" in request.url.path:
        return httpx.Response(200, json={"artifacts": [{"base64": _JPEG, "finishReason": "SUCCESS"}]})
    kind = _tier_kind(payload)
    if kind == "router":
        route = json.dumps(_ROUTE, ensure_ascii=False)
        # Without a response_format (Ultra) a model wraps its JSON in prose.
        return httpx.Response(
            200, json=_chat(route if "response_format" in payload else f"路由結果：\n```json\n{route}\n```")
        )
    if kind == "thinking":
        return httpx.Response(200, json=_chat(_TIER_ANSWERS[kind], reasoning_content="1/6 + 1/3 = 1/2"))
    return httpx.Response(200, json=_chat(_TIER_ANSWERS[kind]))


class _FakeClock:
    """A clock that only moves when the gate sleeps (the pacing)."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _tier_ids(settings: Settings | None = None) -> list[str]:
    from src.runtime.model_roles import ROLES, role_specs

    specs = role_specs(settings or Settings(_env_file=None))
    return [f"tier.{role}.{index + 1}.{spec.label}" for role in ROLES for index, spec in enumerate(specs[role])]


def _tier_run(
    responder: Responder, *, settings: Settings | None = None, only: list[str] | None = None, **context: object
) -> tuple[dict[str, live_gate.CheckResult], _Recorder, _FakeClock]:
    settings = settings or Settings(_env_file=None)
    ctx, recorder = _context(responder, settings=settings, **context)
    clock = _FakeClock()
    ctx.clock, ctx.sleep = clock, clock.sleep
    results = live_gate.run_checks(ctx, live_gate.select_checks("tiers", only or [], settings))
    return {result.check_id: result for result in results}, recorder, clock


class TierCheckTests(unittest.TestCase):
    def test_one_check_per_tier_of_the_configured_table_in_call_order(self) -> None:
        checks = live_gate.select_checks("tiers")

        self.assertEqual([check.check_id for check in checks], _tier_ids())
        self.assertEqual(
            [check.check_id for check in checks[:4]],
            [
                f"tier.router.1.colab:{live_gate.COLAB_NIMBLE}",
                f"tier.router.2.nvidia:{live_gate.SUPER}",
                f"tier.router.3.nvidia:{live_gate.ULTRA}",
                f"tier.router.4.openrouter:{live_gate.OR_SUPER}",
            ],
        )
        self.assertEqual([check.provider for check in checks[:4]], ["colab", "nvidia", "nvidia", "openrouter"])
        # "all" runs them too, after the other checks.
        everything = [check.check_id for check in live_gate.select_checks("all")]
        self.assertEqual(everything[-len(checks):], _tier_ids())
        self.assertEqual(
            [check.check_id for check in live_gate.select_checks("tiers", ["tier.utility"])],
            [
                f"tier.utility.1.nvidia:{live_gate.SUPER}",
                f"tier.utility.2.nvidia:{live_gate.ULTRA}",
                f"tier.utility.3.openrouter:{live_gate.OR_ULTRA}",
            ],
        )
        # The env file's table, not the code defaults: MODEL_ROLES and the
        # path preference are applied as the bot applies them.
        settings = Settings(
            _env_file=None,
            model_path_preference="api",
            model_roles=json.dumps({"utility": [{"provider": "openrouter", "model": live_gate.OR_QWEN, "effort": "none"}]}),
        )
        tiers = [check.check_id for check in live_gate.select_checks("tiers", [], settings)]
        self.assertEqual(tiers, _tier_ids(settings))
        self.assertIn(f"tier.utility.1.openrouter:{live_gate.OR_QWEN}", tiers)
        self.assertIn(f"tier.chat.4.colab:{live_gate.COLAB_QWEN}", tiers)

    def test_an_invalid_role_table_is_one_failing_check(self) -> None:
        settings = Settings(_env_file=None, model_roles="{not json")
        ctx, recorder = _context(_tier_reply, settings=settings)

        [result] = live_gate.run_checks(ctx, live_gate.select_checks("tiers", [], settings))

        self.assertEqual(result.check_id, "tier.table")
        self.assertFalse(result.passed or result.skipped)
        self.assertIn("role table: MODEL_ROLES is not valid JSON", result.evidence)
        self.assertEqual(recorder.requests, [])

    def test_every_api_tier_answers_one_request_as_the_bot_sends_it(self) -> None:
        results, recorder, _ = _tier_run(_tier_reply)

        for check_id, result in results.items():
            with self.subTest(check=check_id):
                if ".colab:" in check_id:
                    # No READY session to call (the colab path's discovery).
                    self.assertTrue(result.skipped)
                    self.assertEqual(result.evidence, "COLAB_ENABLED is off")
                else:
                    self.assertTrue(result.passed, result.evidence)
        # Exactly one request per API tier: 11 NVIDIA, 7 OpenRouter.
        hosts = [request.url.host for request in recorder.requests]
        self.assertEqual(hosts.count("openrouter.ai"), 7)
        self.assertEqual(
            hosts.count("integrate.api.nvidia.com") + hosts.count("ai.api.nvidia.com"), 11
        )
        evidence = {check_id: result.evidence for check_id, result in results.items()}
        for tier in (f"2.nvidia:{live_gate.SUPER}", f"3.nvidia:{live_gate.ULTRA}", f"4.openrouter:{live_gate.OR_SUPER}"):
            self.assertEqual(evidence[f"tier.router.{tier}"], "agent=web_search, output=text, thinking=off")
        self.assertEqual(evidence[f"tier.chat.2.nvidia:{live_gate.ULTRA}"], "21 chars, CJK 83%, traditional: yes")
        self.assertIn("expected answer: yes", evidence[f"tier.thinking.2.nvidia:{live_gate.ULTRA}"])
        self.assertTrue(evidence[f"tier.vision.2.nvidia:{live_gate.OMNI}"].startswith("red circle:"))
        self.assertTrue(evidence[f"tier.vision.3.nvidia:{live_gate.MUSE}"].startswith("red circle:"))
        self.assertIn("traditional: yes", evidence[f"tier.utility.3.openrouter:{live_gate.OR_ULTRA}"])
        self.assertEqual(evidence[f"tier.embed.1.nvidia:{live_gate.EMBED}"], "dim=1024 = EMBEDDING_DIMENSIONS")
        self.assertEqual(evidence[f"tier.image.2.nvidia:{live_gate.FLUX_DEV}"], "image/jpeg, 4 KB")

        # The bot's own payloads, tier options included.
        def sent(kind: str, model: str) -> list[dict]:
            return [p for p in recorder.payloads() if p.get("model") == model and _tier_kind(p) == kind]

        [super_router] = sent("router", live_gate.SUPER)
        self.assertEqual(super_router["response_format"]["type"], "json_schema")
        [ultra_router] = sent("router", live_gate.ULTRA)
        self.assertNotIn("response_format", ultra_router)
        self.assertEqual((ultra_router["temperature"], ultra_router["max_tokens"]), (0.1, 256))
        [or_router] = sent("router", live_gate.OR_SUPER)
        self.assertEqual(
            (or_router["response_format"], or_router["reasoning"]), ({"type": "json_object"}, {"effort": "none"})
        )
        [chat] = sent("chat", live_gate.ULTRA)
        self.assertEqual(chat["chat_template_kwargs"], {"enable_thinking": False})
        # Thinking as the tier says, with the budget it gets.
        [thinking] = sent("thinking", live_gate.ULTRA)
        self.assertEqual(thinking["chat_template_kwargs"], {"enable_thinking": True, "medium_effort": True})
        self.assertNotIn("reasoning_budget", thinking)
        self.assertEqual(thinking["max_tokens"], 2048 + 4096)
        for model in (live_gate.OR_QWEN, live_gate.OR_ULTRA):
            [deep] = sent("thinking", model)
            self.assertEqual(deep["reasoning"], {"effort": "medium"})
        [omni] = sent("vision", live_gate.OMNI)
        self.assertEqual(omni["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((omni["temperature"], omni["max_tokens"]), (0.2, 1024))
        self.assertIsNotNone(_image_part(omni))
        [muse] = sent("vision", live_gate.MUSE)
        self.assertEqual(muse["reasoning_effort"], "none")
        self.assertIsNotNone(_image_part(muse))
        [qwen_vision] = sent("vision", live_gate.OR_QWEN)
        self.assertEqual(qwen_vision["reasoning"], {"effort": "none"})
        [summary] = sent("utility", live_gate.SUPER)
        self.assertEqual((summary["temperature"], summary["max_tokens"]), (0.2, 768))
        self.assertEqual(summary["chat_template_kwargs"], {"enable_thinking": False})
        [embedding] = [p for p in recorder.payloads() if p.get("model") == live_gate.EMBED]
        self.assertEqual(embedding["input"], ["台北今天會下雨嗎？"])

    def test_requests_are_spaced_within_each_providers_free_per_minute_limit(self) -> None:
        clock = _FakeClock()
        sent: list[tuple[str, float]] = []

        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            sent.append(("openrouter" if request.url.host == "openrouter.ai" else "nvidia", clock.now))
            return _tier_reply(request, payload)

        ctx, _ = _context(responder)
        ctx.clock, ctx.sleep = clock, clock.sleep
        live_gate.run_checks(ctx, live_gate.select_checks("tiers"))

        self.assertEqual(len(sent), 18)
        self.assertTrue(clock.sleeps)
        for provider, interval in (("nvidia", 60 / 40), ("openrouter", 60 / 20)):
            times = [at for name, at in sent if name == provider]
            with self.subTest(provider=provider):
                self.assertGreater(len(times), 1)
                self.assertTrue(
                    all(later - earlier >= interval - 1e-9 for earlier, later in zip(times, times[1:], strict=False)),
                    times,
                )

    def test_openrouter_is_not_called_again_after_its_daily_cap(self) -> None:
        cap = '{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 credits","code":429}}'

        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.url.host == "openrouter.ai":
                return httpx.Response(429, text=cap)
            return _tier_reply(request, payload)

        ctx, recorder = _context(responder)
        checks = [_check("openrouter.qwen.no_reasoning"), *live_gate.select_checks("tiers")]
        results = {result.check_id: result for result in live_gate.run_checks(ctx, checks)}

        first = results["openrouter.qwen.no_reasoning"]
        self.assertFalse(first.passed or first.skipped)
        self.assertEqual(first.evidence, "HTTP 429 (rate limited: OpenRouter's free-models-per-day cap)")
        # One request met the cap; every OpenRouter tier after it SKIPs unsent.
        self.assertEqual([r.url.host for r in recorder.requests].count("openrouter.ai"), 1)
        for check_id, result in results.items():
            if check_id.startswith("tier.") and ".openrouter:" in check_id:
                with self.subTest(check=check_id):
                    self.assertTrue(result.skipped)
                    self.assertEqual(result.evidence, live_gate.DAILY_CAP_NOTE)
                    self.assertTrue(result.evidence.startswith("daily cap"))
            elif check_id.startswith("tier.") and ".nvidia:" in check_id:
                with self.subTest(check=check_id):
                    self.assertTrue(result.passed, result.evidence)

        # The tier check that meets the cap stops the rest the same way.
        ctx, recorder = _context(responder)
        results = {r.check_id: r for r in live_gate.run_checks(ctx, live_gate.select_checks("tiers"))}
        router = results[f"tier.router.4.openrouter:{live_gate.OR_SUPER}"]
        self.assertIn("free-models-per-day", router.evidence)
        self.assertFalse(router.passed or router.skipped)
        self.assertEqual([r.url.host for r in recorder.requests].count("openrouter.ai"), 1)
        self.assertTrue(results[f"tier.chat.3.openrouter:{live_gate.OR_QWEN}"].skipped)

    def test_force_tier_skips_and_a_missing_key_fails_without_a_request(self) -> None:
        results, recorder, _ = _tier_run(_tier_reply, settings=Settings(_env_file=None, force_tier="openrouter"))

        for check_id, result in results.items():
            if ".nvidia:" in check_id:
                with self.subTest(check=check_id):
                    self.assertTrue(result.skipped)
                    self.assertIn("FORCE_TIER=openrouter", result.evidence)
        self.assertEqual({request.url.host for request in recorder.requests}, {"openrouter.ai"})

        results, recorder, _ = _tier_run(_tier_reply, nvidia_key="")
        for check_id, result in results.items():
            if ".nvidia:" in check_id:
                with self.subTest(check=check_id):
                    self.assertFalse(result.passed or result.skipped)
                    self.assertIn("NVIDIA_API_KEY not set", result.evidence)
        self.assertEqual({request.url.host for request in recorder.requests}, {"openrouter.ai"})

    def test_colab_tiers_run_through_a_ready_session_and_skip_without_one(self) -> None:
        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.url.host == colab_fakes.TUNNEL_HOST:
                return _tunnel()(request, payload)
            return _tier_reply(request, payload)

        colab_ids = [check_id for check_id in _tier_ids() if ".colab:" in check_id]
        self.assertEqual(len(colab_ids), 4)
        settings = Settings(_env_file=None, **_COLAB_ON)

        ctx, recorder = _context(responder, settings=settings)
        ctx.colab_loader = _Loader(colab_fakes.ready_state())
        results = {r.check_id: r for r in live_gate.run_checks(ctx, live_gate.select_checks("tiers", colab_ids, settings))}

        self.assertTrue(all(result.passed for result in results.values()), [r.evidence for r in results.values()])
        self.assertIn("decides web_search", results[colab_ids[0]].evidence)
        self.assertIn("red circle", results[f"tier.vision.1.colab:{live_gate.COLAB_QWEN}"].evidence)
        bodies = [json.loads(r.content) for r in recorder.requests if r.url.path == "/api/chat"]
        # chat at COLAB_CHAT_THINK, thinking at COLAB_DEEP_THINK, vision at COLAB_CHAT_THINK.
        self.assertEqual(
            [body["think"] for body in bodies], [settings.colab_chat_think, settings.colab_deep_think, settings.colab_chat_think]
        )
        for request in recorder.requests:
            self.assertEqual(request.url.host, colab_fakes.TUNNEL_HOST)

        # The router switched off, and no READY session: SKIP, nothing sent.
        ctx, recorder = _context(responder, settings=settings.model_copy(update={"colab_router_enabled": False}))
        ctx.colab_loader = _Loader(colab_fakes.ready_state())
        [router] = live_gate.run_checks(ctx, live_gate.select_checks("tiers", colab_ids[:1], settings))
        self.assertTrue(router.skipped)
        self.assertIn("COLAB_ROUTER_ENABLED=false", router.evidence)

        ctx, recorder = _context(responder, settings=settings)
        ctx.colab_loader = _Loader(ColabState(status="BOOTING", nbh=colab_fakes.NBH))
        results = live_gate.run_checks(ctx, live_gate.select_checks("tiers", colab_ids, settings))
        self.assertTrue(all(result.skipped for result in results))
        self.assertIn("no READY Colab session", results[0].evidence)
        self.assertEqual(recorder.requests, [])

    def test_each_role_fails_on_a_bad_answer(self) -> None:
        cases = {
            f"tier.router.3.nvidia:{live_gate.ULTRA}": (_chat("我覺得應該查一下網路"), "no routing decision in the reply"),
            f"tier.chat.2.nvidia:{live_gate.ULTRA}": (_chat("Taipei 101 is a skyscraper in Taipei."), "answer is not zh-TW"),
            f"tier.thinking.2.nvidia:{live_gate.ULTRA}": (_chat("需要 2 小時。"), "no reasoning despite thinking on"),
            f"tier.vision.2.nvidia:{live_gate.OMNI}": (_chat("一個藍色的正方形。"), "red: no, circle: no"),
            f"tier.utility.1.nvidia:{live_gate.SUPER}": (_chat("使用者在學", finish_reason="length"), "cut off at max_tokens"),
            f"tier.embed.1.nvidia:{live_gate.EMBED}": ({"data": [{"index": 0, "embedding": [0.1] * 512}]}, "no vector from the app (api dim=512, EMBEDDING_DIMENSIONS=1024)"),
            # The provider raises ContentFilteredError for it (D5).
            f"tier.image.1.nvidia:{live_gate.FLUX_KLEIN}": ({"artifacts": [{"finishReason": "CONTENT_FILTERED"}]}, "content filtered"),
        }
        for check_id, (body, message) in cases.items():
            with self.subTest(check=check_id):
                results, _, _ = _tier_run(lambda request, payload, body=body: httpx.Response(200, json=body), only=[check_id])
                result = results[check_id]
                self.assertFalse(result.passed or result.skipped)
                self.assertIn(message, result.evidence)

    def test_a_router_tier_that_ignores_the_schema_fails(self) -> None:
        router_tiers = [check_id for check_id in _tier_ids() if check_id.startswith("tier.router.") and ".colab:" not in check_id]
        self.assertEqual(len(router_tiers), 3)
        for reply in ("{}", json.dumps({"agent": "foo"}), "好的：\n```json\n{\"agent\": \"web_search\"}\n```"):
            results, _, _ = _tier_run(
                lambda request, payload, reply=reply: httpx.Response(200, json=_chat(reply)), only=router_tiers
            )
            for check_id, result in results.items():
                with self.subTest(check=check_id, reply=reply):
                    self.assertFalse(result.passed or result.skipped)
                    self.assertIn("route JSON breaks the schema: missing", result.evidence)

    def test_the_utility_tiers_get_the_memory_services_own_request(self) -> None:
        messages = live_gate._utility_messages()

        self.assertEqual(messages, memory_service.summary_messages("", live_gate.UTILITY_DIALOGUE))
        request = messages[1]["content"]
        # What the bot sends for a first summary, its update rules included.
        self.assertIn("現有長期記憶摘要（預設保留，除非新對話明確推翻、過期或確認不再重要）：\n(empty)\n\n", request)
        self.assertIn("更新要求：\n1. 保留舊摘要中仍有效的資訊。", request)
        self.assertTrue(request.endswith(f"最近對話：\n{live_gate.UTILITY_DIALOGUE}"))

    def test_a_tier_slower_than_its_limit_and_slack_fails(self) -> None:
        settings = Settings(
            _env_file=None,
            model_roles=json.dumps(
                {"chat": [{"provider": "nvidia", "model": live_gate.ULTRA, "thinking": False, "timeout_s": 0.1}]}
            ),
        )

        def slow(request: httpx.Request, payload: dict) -> httpx.Response:
            time.sleep(0.4)
            return _tier_reply(request, payload)

        with patch.object(live_gate, "TIER_SLACK_SECONDS", 0.05):
            results, _, _ = _tier_run(slow, settings=settings, only=[f"tier.chat.1.nvidia:{live_gate.ULTRA}"])

        [result] = results.values()
        self.assertFalse(result.passed)
        self.assertEqual(result.evidence, "no answer within 0.15s (the tier's 0.1s + 0.05s)")

    def test_http_failures_name_the_status(self) -> None:
        results, _, _ = _tier_run(
            lambda request, payload: httpx.Response(410, json={"detail": "end of life on 2026-07-27T00:00:00Z"}),
            only=[f"tier.chat.2.nvidia:{live_gate.ULTRA}", f"tier.embed.1.nvidia:{live_gate.EMBED}"],
        )

        for result in results.values():
            with self.subTest(check=result.check_id):
                self.assertTrue(result.evidence.startswith("HTTP 410"), result.evidence)

    def test_the_tier_table_is_short_and_never_shows_keys_or_whole_answers(self) -> None:
        long_answer = "這是一段很長的回答，" * 40

        def responder(request: httpx.Request, payload: dict) -> httpx.Response:
            if request.url.host == "openrouter.ai":
                # A router that answers in prose: its FAIL quotes only a snippet.
                return httpx.Response(200, json=_chat(long_answer))
            return httpx.Response(401, text=f"invalid key {_NVIDIA_KEY} for this account")

        code, out, err = self._main(responder, ["--path", "tiers"])

        self.assertEqual(code, 1)
        self.assertNotIn(_NVIDIA_KEY, out + err)
        self.assertNotIn(_OPENROUTER_KEY, out + err)
        self.assertNotIn(long_answer, out + err)
        lines = out.strip().splitlines()
        self.assertEqual(lines[0].split(), ["ROLE", "#", "PROVIDER:MODEL", "RESULT", "LATENCY", "NOTE"])
        self.assertEqual(len(lines), len(_tier_ids()) + 2)
        self.assertRegex(lines[-1], r"^\d+/\d+ tiers passed, 4 skipped$")
        note_at = lines[0].index("NOTE")
        for line in lines[1:-1]:
            with self.subTest(line=line[:40]):
                self.assertLessEqual(len(line[note_at:]), live_gate.TIER_NOTE_LIMIT)
        router = next(line for line in lines if f"openrouter:{live_gate.OR_SUPER}" in line and line.startswith("router"))
        self.assertEqual(router.split()[:4], ["router", "4", f"openrouter:{live_gate.OR_SUPER}", "FAIL"])
        self.assertIn("no routing decision", router)
        chat = next(line for line in lines if line.startswith("chat") and f"nvidia:{live_gate.ULTRA}" in line)
        self.assertIn("HTTP 401: invalid key *** for this account", chat)
        self.assertRegex(chat, r"FAIL\s+\d+\.\ds\s+HTTP 401")

    def test_main_prints_the_model_and_tier_tables_apart(self) -> None:
        code, out, _ = self._main(
            _tier_reply, ["--only", f"nvidia.embed.dim,tier.embed.1.nvidia:{live_gate.EMBED}"]
        )

        self.assertEqual(code, 0)
        model_table, tier_table = out.strip().split("\n\n")
        self.assertIn("nvidia.embed.dim", model_table)
        self.assertIn("1/1 checks passed", model_table)
        self.assertTrue(tier_table.startswith("ROLE"))
        self.assertIn(f"embed  1  nvidia:{live_gate.EMBED}  PASS", tier_table)
        self.assertIn("1/1 tiers passed", tier_table)

    def test_the_tier_table_reads_ids_whose_model_has_dots_and_colons(self) -> None:
        table = live_gate.format_tier_table(
            [
                live_gate.CheckResult(f"tier.chat.1.colab:{live_gate.COLAB_QWEN}", False, "COLAB_ENABLED is off", 0.0, skipped=True),
                live_gate.CheckResult("tier.table", False, "role table: bad", 0.0),
            ]
        )

        first, second = table.splitlines()[1:3]
        self.assertEqual(first.split()[:4], ["chat", "1", f"colab:{live_gate.COLAB_QWEN}", "SKIP"])
        self.assertEqual(second.split()[:4], ["-", "-", "table", "FAIL"])
        self.assertTrue(table.endswith("0/1 tiers passed, 1 skipped"))

    def _main(self, responder: Responder, argv: list[str]) -> tuple[int, str, str]:
        recorder = _Recorder(responder)
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "gate.env"
            env_file.write_text(
                f"NVIDIA_API_KEY={_NVIDIA_KEY}\nOPENROUTER_API_KEY={_OPENROUTER_KEY}\n", encoding="utf-8"
            )
            out, err = io.StringIO(), io.StringIO()
            with (
                patch.dict(os.environ),
                # No real waiting between requests (test_requests_are_spaced... covers it).
                patch.dict(live_gate.TIER_RPM, clear=True),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                for name in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "LOG_LEVEL", "COLAB_ENABLED"):
                    os.environ.pop(name, None)
                code = live_gate.main(
                    ["--env-file", str(env_file), *argv],
                    client=httpx.Client(transport=httpx.MockTransport(recorder)),
                )
        return code, out.getvalue(), err.getvalue()
