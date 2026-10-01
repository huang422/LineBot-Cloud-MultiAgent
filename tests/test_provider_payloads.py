"""Request bodies per model profile, driven by the tier options.

``build_chat_payload`` / ``build_image_payload`` are pure, so each case is a
direct call: profile x thinking on/off x JSON mode, OpenRouter reasoning
efforts, and the FLUX payloads. The providers' HTTP behaviour is covered in
tests/test_runtime_behaviors.py and tests/test_provider_errors.py.
"""

from __future__ import annotations

import json
import time
import unittest

import httpx

from src.config import Settings
from src.providers import nvidia_provider, openrouter_provider
from src.providers.base import TransientError
from src.providers.model_profiles import (
    generic_profile,
    get_profile,
    response_format_for,
    tier_sampling,
)
from src.runtime.model_roles import (
    FLUX_DEV,
    FLUX_KLEIN,
    NIM_MUSE,
    NIM_OMNI,
    NIM_SUPER,
    NIM_ULTRA,
    OR_QWEN,
    OR_SUPER,
    OR_ULTRA,
    role_specs,
)
from src.services.model_health import ModelHealth

_MESSAGES = [{"role": "user", "content": "hi"}]
_SCHEMA = {"type": "object", "properties": {"agent": {"type": "string"}}, "required": ["agent"]}
_STRICT = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": _SCHEMA}}


def _nim(model: str, *, thinking: bool, options=None, **kwargs) -> dict:
    kwargs.setdefault("temperature", 0.7)
    kwargs.setdefault("max_tokens", 1000)
    return nvidia_provider.build_chat_payload(
        model, _MESSAGES, thinking=thinking, options=options, thinking_budget=2000, **kwargs
    )


def _openrouter(model: str, *, thinking: bool, options=None, **kwargs) -> dict:
    kwargs.setdefault("temperature", 0.7)
    kwargs.setdefault("max_tokens", 1000)
    return openrouter_provider.build_chat_payload(
        model, _MESSAGES, thinking=thinking, options=options, thinking_budget=2000, **kwargs
    )


class NimChatPayloadTests(unittest.TestCase):
    def test_thinking_on_and_off_per_profile(self) -> None:
        # (model, thinking, chat_template_kwargs, reasoning_budget, max_tokens)
        cases = (
            (NIM_ULTRA, False, {"enable_thinking": False}, None, 1000),
            # Ultra: any reasoning_budget is HTTP 400 (live gate 2026-09-30);
            # it thinks at medium_effort and max_tokens caps the length.
            (NIM_ULTRA, True, {"enable_thinking": True, "medium_effort": True}, None, 3000),
            (NIM_SUPER, False, {"enable_thinking": False}, None, 1000),
            (NIM_SUPER, True, {"enable_thinking": True}, 2000, 3000),
            (NIM_OMNI, False, {"enable_thinking": False}, None, 1000),
            (NIM_OMNI, True, {"enable_thinking": True}, 2000, 3000),
        )
        for model, thinking, kwargs, budget, max_tokens in cases:
            with self.subTest(model=model, thinking=thinking):
                payload = _nim(model, thinking=thinking)
                self.assertIs(payload["stream"], False)
                self.assertEqual(payload["chat_template_kwargs"], kwargs)
                self.assertEqual(payload.get("reasoning_budget"), budget)
                self.assertEqual(payload["max_tokens"], max_tokens)
                self.assertNotIn("reasoning_effort", payload)

    def test_the_tiers_reasoning_budget_wins_and_is_capped(self) -> None:
        payload = _nim(NIM_SUPER, thinking=True, options={"reasoning_budget": 4096})
        self.assertEqual((payload["reasoning_budget"], payload["max_tokens"]), (4096, 5096))

        payload = _nim(NIM_SUPER, thinking=True, options={"reasoning_budget": 40_000})
        self.assertEqual(payload["reasoning_budget"], 32_768)
        self.assertEqual(payload["max_tokens"], 32_768)  # the profile's cap

    def test_ultra_never_sends_a_reasoning_budget(self) -> None:
        payload = _nim(NIM_ULTRA, thinking=True, options={"reasoning_budget": 4096})

        self.assertNotIn("reasoning_budget", payload)
        self.assertNotIn("reasoning_budget", payload["chat_template_kwargs"])
        self.assertNotIn("thinking_token_budget", payload["chat_template_kwargs"])
        # The budget still leaves the thoughts room before the answer.
        self.assertEqual(payload["max_tokens"], 5096)

    def test_muse_glimmer_switches_reasoning_with_reasoning_effort(self) -> None:
        off = _nim(NIM_MUSE, thinking=False)
        on = _nim(NIM_MUSE, thinking=True)

        # It reasons by default, so "none" is always sent when thinking is off.
        self.assertEqual((off["reasoning_effort"], off["max_tokens"]), ("none", 1000))
        self.assertEqual((on["reasoning_effort"], on["max_tokens"]), ("medium", 3000))
        for payload in (off, on):
            self.assertNotIn("chat_template_kwargs", payload)
            self.assertNotIn("reasoning_budget", payload)
            self.assertNotIn("top_p", payload)
            # Its model default, 0.6, instead of the call's 0.7.
            self.assertEqual(payload["temperature"], 0.6)
        self.assertEqual(_nim(NIM_MUSE, thinking=False, options={"temperature": 0.0})["temperature"], 0.01)

    def test_muse_glimmer_max_tokens_is_capped_at_16384(self) -> None:
        self.assertEqual(_nim(NIM_MUSE, thinking=True, max_tokens=16_000)["max_tokens"], 16_384)
        self.assertEqual(_nim(NIM_MUSE, thinking=False, max_tokens=1024)["max_tokens"], 1024)

    def test_the_nano_omni_vision_tier_sends_the_body_the_probe_accepted(self) -> None:
        # Vision probe 2026-10-01: exactly these fields answered (thinking
        # off, 0.2, 1024 tokens, no top_p), whatever the call's sampling.
        [omni] = [spec for spec in role_specs(Settings(_env_file=None))["vision"] if spec.model == NIM_OMNI]

        payload = _nim(NIM_OMNI, thinking=False, options=omni.options(), temperature=0.5, max_tokens=2048)

        self.assertEqual(
            payload,
            {
                "model": NIM_OMNI,
                "messages": _MESSAGES,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 1024,
                "temperature": 0.2,
            },
        )
        self.assertIs(omni.options()["thinking"], False)
        self.assertIn("image", get_profile("nvidia", NIM_OMNI).modalities)

    def test_muse_glimmer_takes_images_and_no_tools(self) -> None:
        profile = get_profile("nvidia", NIM_MUSE)
        self.assertIn("image", profile.modalities)
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

        payload = _nim(NIM_MUSE, thinking=False, tools=tools, tool_choice="auto")

        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_json_modes_follow_the_tier(self) -> None:
        # (json_mode, response_format); only a call with a schema is held to JSON.
        for json_mode, expected in (
            ("json_schema", _STRICT),
            ("json_object", {"type": "json_object"}),
            ("prompt", None),
            (None, None),
        ):
            with self.subTest(json_mode=json_mode):
                options = {"json_mode": json_mode} if json_mode else {}
                payload = _nim(NIM_SUPER, thinking=False, options=options, json_schema=_SCHEMA)
                self.assertEqual(payload.get("response_format"), expected)
                self.assertNotIn("nvext", payload)
        payload = _nim(NIM_SUPER, thinking=False, options={"json_mode": "json_schema"})
        self.assertNotIn("response_format", payload)

    def test_the_ultra_router_tier_asks_for_json_in_the_prompt_only(self) -> None:
        # Ultra has no structured output on NIM: the default router tier
        # sends no response_format (nor nvext) even with the router schema.
        [ultra] = [
            spec for spec in role_specs(Settings(_env_file=None))["router"]
            if spec.provider == "nvidia" and spec.model == NIM_ULTRA
        ]
        options = ultra.options()
        self.assertEqual(options["json_mode"], "prompt")

        payload = _nim(NIM_ULTRA, thinking=False, options=options, json_schema=_SCHEMA, temperature=0.7, max_tokens=384)

        self.assertNotIn("response_format", payload)
        self.assertNotIn("nvext", payload)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        # The tier's low temperature and cap win over the card's 1.0 and the call.
        self.assertEqual((payload["temperature"], payload["max_tokens"], payload["top_p"]), (0.1, 256, 0.95))

    def test_the_router_tier_sampling_wins_over_the_call(self) -> None:
        options = {"thinking": False, "json_mode": "json_schema", "max_tokens": 256, "temperature": 0.1}

        payload = _nim(NIM_SUPER, thinking=False, options=options, temperature=0.7, max_tokens=384)

        # The tier's low temperature also wins over the model card's 1.0.
        self.assertEqual((payload["temperature"], payload["max_tokens"]), (0.1, 256))
        self.assertEqual(payload["top_p"], 0.95)

    def test_nemotron_uses_its_model_card_sampling_over_the_call(self) -> None:
        # Nemotron 3 cards: temperature 1.0, top_p 0.95 for every task.
        for model in (NIM_ULTRA, NIM_SUPER):
            for thinking in (False, True):
                with self.subTest(model=model, thinking=thinking):
                    payload = _nim(model, thinking=thinking, temperature=0.2)
                    self.assertEqual((payload["temperature"], payload["top_p"]), (1.0, 0.95))
        # A model without card sampling keeps the call's and sends no top_p.
        plain = _nim("example/new-model", thinking=False, temperature=0.5)
        self.assertEqual(plain["temperature"], 0.5)
        self.assertNotIn("top_p", plain)

    def test_temperature_is_clamped_into_the_nim_range(self) -> None:
        for temperature, expected in ((0.0, 0.01), (1.7, 1.0), (0.5, 0.5)):
            for model in (NIM_ULTRA, NIM_SUPER, NIM_MUSE):
                with self.subTest(temperature=temperature, model=model):
                    # A tier temperature is clamped into the profile's range.
                    options = {"temperature": temperature}
                    self.assertEqual(_nim(model, thinking=False, options=options)["temperature"], expected)
        # NIM rejects 0 for an unprofiled id too.
        self.assertEqual(_nim("example/new-model", thinking=False, temperature=0.0)["temperature"], 0.01)

    def test_tools_go_only_to_models_that_take_them(self) -> None:
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

        payload = _nim(NIM_ULTRA, thinking=False, tools=tools, tool_choice="auto")

        self.assertEqual((payload["tools"], payload["tool_choice"]), (tools, "auto"))

    def test_an_unprofiled_id_gets_a_plain_openai_body(self) -> None:
        payload = _nim("example/new-model", thinking=True, json_schema=_SCHEMA, options={"json_mode": "json_schema"})

        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("reasoning_budget", payload)
        self.assertEqual(payload["max_tokens"], 1000)
        self.assertEqual(payload["response_format"], _STRICT)


class NanoOmniCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_image_goes_out_and_a_busy_worker_fails_fast_as_transient(self) -> None:
        sent: list[dict] = []
        busy = '{"error":{"message":"ResourceExhausted: Worker local total request limit reached"}}'

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(503, text=busy)

        provider = nvidia_provider.NvidiaProvider("test-key", ModelHealth(), transport=httpx.MockTransport(handler))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "這是什麼？"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}},
                ],
            }
        ]
        started = time.monotonic()
        try:
            with self.assertRaises(TransientError) as caught:
                await provider.generate(NIM_OMNI, messages, disable_thinking=True, options={"thinking": False})
        finally:
            await provider.close()

        self.assertEqual(caught.exception.status, 503)
        self.assertLess(time.monotonic() - started, 1.0)
        [payload] = sent
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")


class OpenRouterPayloadTests(unittest.TestCase):
    def test_reasoning_per_profile_and_tier_effort(self) -> None:
        # (model, thinking, options, reasoning object, max_tokens)
        cases = (
            # qwen3.8 thinks at xhigh unless told "none".
            (OR_QWEN, False, {"effort": "none"}, {"effort": "none"}, 1000),
            (OR_QWEN, False, {}, {"effort": "none"}, 1000),
            (OR_QWEN, True, {"effort": "medium"}, {"effort": "medium"}, 3000),
            # effort "none" is the chat tier's off switch; a thinking request
            # (the thinking flag) still thinks, at medium.
            (OR_QWEN, True, {"effort": "none"}, {"effort": "medium"}, 3000),
            (OR_QWEN, True, {"effort": "low", "reasoning_budget": 500}, {"effort": "low"}, 1500),
            # openrouter/free cannot be asked to think, and some of the models
            # it picks reason by default: a thinking request still says off.
            ("openrouter/free", True, {}, {"effort": "none"}, 1000),
            # Nemotron :free ids reason by default: off is said explicitly.
            (OR_ULTRA, False, {"effort": "none"}, {"effort": "none"}, 1000),
            (OR_ULTRA, False, {}, {"effort": "none"}, 1000),
            (OR_ULTRA, True, {"effort": "medium"}, {"effort": "medium"}, 3000),
            (OR_SUPER, False, {"effort": "none"}, {"effort": "none"}, 1000),
        )
        for model, thinking, options, reasoning, max_tokens in cases:
            with self.subTest(model=model, thinking=thinking, options=options):
                payload = _openrouter(model, thinking=thinking, options=options)
                self.assertEqual(payload.get("reasoning"), reasoning)
                self.assertEqual(payload["max_tokens"], max_tokens)
                self.assertNotIn("modalities", payload)
                self.assertNotIn("stream", payload)

    def test_json_modes_follow_the_tier(self) -> None:
        self.assertNotIn(
            "response_format",
            _openrouter(OR_QWEN, thinking=False, options={"json_mode": "prompt"}, json_schema=_SCHEMA),
        )
        self.assertEqual(
            _openrouter(OR_QWEN, thinking=False, options={"json_mode": "json_schema"}, json_schema=_SCHEMA)[
                "response_format"
            ],
            _STRICT,
        )
        # Ultra :free takes no response_format: its tiers set no json_mode.
        self.assertNotIn(
            "response_format", _openrouter(OR_ULTRA, thinking=False, options={"effort": "none"}, json_schema=_SCHEMA)
        )
        # The router's last tier: Super :free with json_object.
        self.assertEqual(
            _openrouter(OR_SUPER, thinking=False, options={"json_mode": "json_object"}, json_schema=_SCHEMA)[
                "response_format"
            ],
            {"type": "json_object"},
        )

    def test_the_tier_sampling_wins_over_the_call(self) -> None:
        payload = _openrouter(OR_QWEN, thinking=False, options={"max_tokens": 64, "temperature": 0.2})

        self.assertEqual((payload["temperature"], payload["max_tokens"]), (0.2, 64))

    def test_openrouter_ids_keep_the_calls_sampling(self) -> None:
        # Card sampling is set only on the NIM Nemotron profiles.
        for model in (OR_QWEN, OR_ULTRA):
            with self.subTest(model=model):
                payload = _openrouter(model, thinking=False, temperature=0.7)
                self.assertEqual(payload["temperature"], 0.7)
                self.assertNotIn("top_p", payload)


class SharedOptionTests(unittest.TestCase):
    def test_tier_sampling(self) -> None:
        self.assertEqual(tier_sampling({}, temperature=0.7, max_tokens=100), (0.7, 100))
        self.assertEqual(
            tier_sampling({"temperature": 0, "max_tokens": 5}, temperature=0.7, max_tokens=100), (0.0, 5)
        )
        # Tier first, then the profile's card default, then the call.
        ultra = get_profile("nvidia", NIM_ULTRA)
        self.assertEqual(tier_sampling({}, temperature=0.7, max_tokens=100, profile=ultra), (1.0, 100))
        self.assertEqual(
            tier_sampling({"temperature": 0.2}, temperature=0.7, max_tokens=100, profile=ultra), (0.2, 100)
        )
        plain = generic_profile("nvidia", "example/new-model")
        self.assertEqual(tier_sampling({}, temperature=0.7, max_tokens=100, profile=plain), (0.7, 100))

    def test_response_format_for(self) -> None:
        self.assertEqual(response_format_for("json_schema", _SCHEMA), _STRICT)
        self.assertEqual(response_format_for("json_object", _SCHEMA), {"type": "json_object"})
        self.assertIsNone(response_format_for("prompt", _SCHEMA))
        self.assertIsNone(response_format_for("json_schema", None))


class ImagePayloadTests(unittest.TestCase):
    def test_flux2_klein(self) -> None:
        # klein: steps 1-4, cfg_scale pinned, unknown fields rejected.
        for options, steps in (({}, 4), ({"steps": 2}, 2), ({"steps": 50}, 4), ({"steps": 1}, 1)):
            with self.subTest(options=options):
                self.assertEqual(
                    nvidia_provider.build_image_payload(FLUX_KLEIN, "a fox", options),
                    {"prompt": "a fox", "width": 1024, "height": 1024, "seed": 0, "steps": steps},
                )

    def test_flux1_dev(self) -> None:
        for options, steps in (({}, 50), ({"steps": 50}, 50), ({"steps": 2}, 5), ({"steps": 500}, 100)):
            with self.subTest(options=options):
                self.assertEqual(
                    nvidia_provider.build_image_payload(FLUX_DEV, "a fox", options),
                    {
                        "prompt": "a fox",
                        "mode": "base",
                        "width": 1024,
                        "height": 1024,
                        "cfg_scale": 5.0,
                        "steps": steps,
                        "samples": 1,
                        "seed": 0,
                    },
                )

    def test_ids_are_exact(self) -> None:
        # No alias or case folding: another spelling has no profile.
        for model in ("black-forest-labs/FLUX.1-dev", "flux.1-dev", "black-forest-labs/flux.2-klein-9b"):
            with self.subTest(model=model):
                self.assertIsNone(nvidia_provider.build_image_payload(model, "a fox"))


if __name__ == "__main__":
    unittest.main()
