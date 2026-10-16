"""FallbackChain v2: per-tier time limits, deadlines, skips and error classes."""

from __future__ import annotations

import asyncio
import math
import time
import unittest
from typing import Any
from unittest.mock import patch

from src.providers import fallback_chain
from src.providers.base import (
    AuthError,
    EmptyCompletionError,
    ModelGoneError,
    ProviderError,
    ProviderResponse,
    RateLimitError,
    SkipTierError,
    TransientError,
)
from src.providers.fallback_chain import (
    COLAB_API_RESERVE_SECONDS,
    DEFAULT_TIER_TIMEOUT_SECONDS,
    AllModelsRateLimitedError,
    AllProvidersFailedError,
    DeadlineExceededError,
    FallbackChain,
    Tier,
)
from src.services.model_health import ModelHealth
from src.utils import request_trace

_TEXT = [{"role": "user", "content": "hello"}]
_IMAGE = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "這是什麼？"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }
]


class FakeProvider:
    """Scripted provider: each call pops the next behaviour (the last one repeats).

    A behaviour is a ProviderResponse or text to return, an exception to
    raise, or ``("sleep", seconds)`` to hang before answering.
    """

    def __init__(self, name: str, *behaviours: Any) -> None:
        self.name = name
        self._behaviours = list(behaviours) or ["ok"]
        self.calls: list[dict] = []

    async def generate(self, model: str, messages: list[dict], **kwargs: Any) -> ProviderResponse:
        self.calls.append({"model": model, **kwargs})
        behaviour = self._behaviours.pop(0) if len(self._behaviours) > 1 else self._behaviours[0]
        if isinstance(behaviour, tuple) and behaviour[0] == "sleep":
            await asyncio.sleep(behaviour[1])
            return ProviderResponse(text="late", model=model)
        if isinstance(behaviour, BaseException):
            raise behaviour
        if isinstance(behaviour, ProviderResponse):
            return behaviour
        return ProviderResponse(text=behaviour, model=model)


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _thinking_then_plain(*tiers: Tier) -> list[Tier]:
    """A thinking request's shape: the first tier thinking, then all of them without."""
    return [tiers[0].with_options(thinking=True), *(tier.with_options(thinking=False) for tier in tiers)]


class TierTests(unittest.TestCase):
    def test_a_hand_built_tier_gets_the_defaults(self) -> None:
        provider = FakeProvider("nvidia")
        tier = Tier(provider, "m")
        self.assertEqual(
            (tier.provider, tier.model, tier.timeout_s, dict(tier.options), tier.modalities),
            (provider, "m", DEFAULT_TIER_TIMEOUT_SECONDS, {}, frozenset({"text", "image"})),
        )
        self.assertEqual(tier.label, "nvidia:m")

    def test_with_options_returns_a_new_tier(self) -> None:
        tier = Tier(FakeProvider("nvidia"), "m", options={"json_mode": "json_schema"})

        thinking = tier.with_options(thinking=True)

        self.assertEqual(dict(thinking.options), {"json_mode": "json_schema", "thinking": True})
        self.assertEqual(dict(tier.options), {"json_mode": "json_schema"})

    def test_provider_name_falls_back_to_the_class_name(self) -> None:
        class Anonymous:
            async def generate(self, model, messages, **kwargs):
                return ProviderResponse(text="ok")

        self.assertEqual(Tier(Anonymous(), "m").label, "anonymous:m")


class TimeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_tier_that_hangs_past_its_own_timeout_moves_on(self) -> None:
        health = ModelHealth()
        slow = FakeProvider("nvidia", ("sleep", 5))
        fast = FakeProvider("openrouter", "fast")
        started = time.monotonic()

        response = await FallbackChain(health).generate(
            [Tier(slow, "muse", timeout_s=0.05), Tier(fast, "qwen:free")], _TEXT
        )

        self.assertEqual(response.text, "fast")
        self.assertLess(time.monotonic() - started, 1)
        # The model ran for its whole own limit: that counts against it.
        status = health.get_status(include_errors=True)["models"]["nvidia:muse"]
        self.assertIn("no answer within 0.05s", status["last_error"])

    async def test_the_deadline_clips_a_tier_and_stops_the_walk(self) -> None:
        health = ModelHealth()
        slow = FakeProvider("nvidia", ("sleep", 5))
        never = FakeProvider("openrouter", "unused")
        started = time.monotonic()

        with self.assertRaises(AllProvidersFailedError) as ctx:
            await FallbackChain(health).generate(
                [Tier(slow, "ultra"), Tier(never, "qwen:free")], _TEXT, deadline=time.monotonic() + 0.05
            )

        self.assertLess(time.monotonic() - started, 1)
        self.assertIsInstance(ctx.exception, DeadlineExceededError)
        self.assertIn("deadline", str(ctx.exception))
        self.assertIsInstance(ctx.exception.last_error, TransientError)
        self.assertEqual(never.calls, [])
        # The deadline left the tier less than min_tier_seconds (5 s by
        # default): that says nothing about the model, so no strike.
        self.assertNotIn("nvidia:ultra", health.get_status()["models"])

    async def test_a_passed_deadline_stops_before_calling_any_tier(self) -> None:
        clock = _Clock()
        provider = FakeProvider("nvidia")

        with self.assertRaises(AllProvidersFailedError) as ctx:
            await FallbackChain(ModelHealth(), clock=clock).generate(
                [Tier(provider, "ultra")], _TEXT, deadline=clock.now
            )

        self.assertIn("deadline reached before nvidia:ultra", str(ctx.exception))
        self.assertEqual(provider.calls, [])

    async def test_the_deadline_is_checked_before_every_tier(self) -> None:
        clock = _Clock()

        class Failing(FakeProvider):
            async def generate(self, model, messages, **kwargs):
                clock.now += 10  # the failed call took 10 s
                return await super().generate(model, messages, **kwargs)

        first = Failing("nvidia", TransientError("ultra", 503, "busy", "nvidia"))
        second = FakeProvider("openrouter")

        with self.assertRaises(AllProvidersFailedError) as ctx:
            await FallbackChain(ModelHealth(), clock=clock).generate(
                [Tier(first, "ultra"), Tier(second, "qwen:free")], _TEXT, deadline=clock.now + 5
            )

        self.assertIn("deadline reached before openrouter:qwen:free", str(ctx.exception))
        self.assertEqual(ctx.exception.last_error.status, 503)
        self.assertEqual(second.calls, [])

    async def test_a_thinking_tier_leaves_half_the_time_to_the_fallback(self) -> None:
        health = ModelHealth()
        thinker = FakeProvider("nvidia", ("sleep", 5), "quick answer")
        started = time.monotonic()

        response = await FallbackChain(health).generate(
            _thinking_then_plain(Tier(thinker, "ultra")), _TEXT, deadline=time.monotonic() + 0.4
        )

        elapsed = time.monotonic() - started
        self.assertEqual(response.text, "quick answer")
        self.assertGreater(elapsed, 0.15)
        self.assertLess(elapsed, 0.38)
        self.assertEqual(
            [call["disable_thinking"] for call in thinker.calls], [False, True]
        )
        # Running out of the thinking share is not the model's fault.
        status = health.get_status(include_errors=True)["models"]["nvidia:ultra"]
        self.assertIsNone(status["last_error"])
        self.assertIsNotNone(status["last_ok_at"])

    async def test_the_thinking_tiers_together_leave_half_the_time_to_the_plain_tiers(self) -> None:
        # Three slow thinking tiers, as a thinking request puts first: each
        # halving the time left would leave the plain tier an eighth of it.
        thinker = FakeProvider("nvidia", ("sleep", 5))
        plain = FakeProvider("openrouter", ("sleep", 0.6))
        tiers = [
            *(Tier(thinker, f"think-{n}", options={"thinking": True}) for n in range(3)),
            Tier(plain, "qwen:free", options={"thinking": False}),
        ]
        started = time.monotonic()

        response = await FallbackChain(ModelHealth()).generate(tiers, _TEXT, deadline=started + 2.0)

        self.assertEqual(response.text, "late")
        self.assertLess(time.monotonic() - started, 1.9)

    async def test_a_thinking_tier_past_the_thinking_half_is_skipped(self) -> None:
        clock = _Clock()

        class SlowFailing(FakeProvider):
            async def generate(self, model, messages, **kwargs):
                clock.now += 10  # each failed call took 10 s
                return await super().generate(model, messages, **kwargs)

        thinker = SlowFailing("nvidia", TransientError("m", 503, "busy", "nvidia"))
        plain = FakeProvider("openrouter", "plain answer")
        tiers = [
            *(Tier(thinker, f"think-{n}", options={"thinking": True}) for n in range(4)),
            Tier(plain, "qwen:free", options={"thinking": False}),
        ]

        response = await FallbackChain(ModelHealth(), clock=clock).generate(
            tiers, _TEXT, deadline=clock.now + 50
        )

        self.assertEqual(response.text, "plain answer")
        # The thinking half ends 25 s in: a thinking tier still starts at
        # 20 s, none at 30 s; the plain tier keeps the rest.
        self.assertEqual([call["model"] for call in thinker.calls], ["think-0", "think-1", "think-2"])

    async def test_a_thinking_tier_with_nothing_after_it_gets_all_the_time(self) -> None:
        thinker = FakeProvider("nvidia", ("sleep", 0.15))

        response = await FallbackChain(ModelHealth()).generate(
            [Tier(thinker, "ultra", options={"thinking": True})], _TEXT, deadline=time.monotonic() + 0.4
        )

        self.assertEqual(response.text, "late")

    async def test_thinking_timeouts_never_cool_the_model_down(self) -> None:
        health = ModelHealth()
        thinker = FakeProvider("nvidia", ("sleep", 5), "plain", ("sleep", 5), "plain", ("sleep", 5), "plain")
        chain = FallbackChain(health)
        tiers = [
            Tier(thinker, "ultra", timeout_s=0.02, options={"thinking": True}),
            Tier(thinker, "ultra", options={"thinking": False}),
        ]

        for _ in range(3):
            response = await chain.generate(tiers, _TEXT)
            self.assertEqual(response.text, "plain")

        # Every request still tried thinking first: no cooldown kicked in.
        self.assertEqual([call["disable_thinking"] for call in thinker.calls], [False, True] * 3)
        self.assertTrue(health.is_available("nvidia", "ultra"))

    async def test_without_a_deadline_only_the_tier_timeout_applies(self) -> None:
        thinker = FakeProvider("nvidia", ("sleep", 0.1))

        response = await FallbackChain(ModelHealth()).generate(
            _thinking_then_plain(Tier(thinker, "ultra")), _TEXT
        )

        self.assertEqual(response.text, "late")
        self.assertEqual(len(thinker.calls), 1)


class ThinkingOptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_thinking_tier_then_the_non_thinking_fallback(self) -> None:
        nvidia = FakeProvider("nvidia", TransientError("ultra", 503, "ResourceExhausted", "nvidia"), "nvidia plain")
        openrouter = FakeProvider("openrouter", "unused")

        response = await FallbackChain(ModelHealth()).generate(
            _thinking_then_plain(Tier(nvidia, "ultra"), Tier(openrouter, "qwen:free")), _TEXT
        )

        self.assertEqual(response.text, "nvidia plain")
        self.assertEqual([call["disable_thinking"] for call in nvidia.calls], [False, True])
        self.assertEqual(openrouter.calls, [])

    async def test_tier_options_override_the_call_level_flag(self) -> None:
        provider = FakeProvider("nvidia", TransientError("a", 500, "x", "nvidia"), TransientError("b", 500, "x", "nvidia"), "ok")

        await FallbackChain(ModelHealth()).generate(
            [
                Tier(provider, "a", options={"thinking": True}),
                Tier(provider, "b", options={"thinking": False}),
                Tier(provider, "c"),
            ],
            _TEXT,
            disable_thinking=True,
        )

        self.assertEqual([call["disable_thinking"] for call in provider.calls], [False, True, True])


class SkipTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_only_tiers_are_skipped_for_images(self) -> None:
        text_only = FakeProvider("nvidia", "never")
        vision = FakeProvider("nvidia", "紅色圓形")
        tiers = [Tier(text_only, "ultra", modalities=frozenset({"text"})), Tier(vision, "muse")]

        response = await FallbackChain(ModelHealth()).generate(tiers, _IMAGE)
        self.assertEqual(response.text, "紅色圓形")
        self.assertEqual(text_only.calls, [])

        # The same text-only tier still serves text.
        response = await FallbackChain(ModelHealth()).generate(tiers, _TEXT)
        self.assertEqual(response.text, "never")

    async def test_all_tiers_skipped_for_images_is_a_failure_not_busy(self) -> None:
        provider = FakeProvider("nvidia")
        with self.assertRaises(AllProvidersFailedError):
            await FallbackChain(ModelHealth()).generate(
                [Tier(provider, "ultra", modalities=frozenset({"text"}))], _IMAGE
            )
        self.assertEqual(provider.calls, [])

    async def test_dead_cooling_and_over_budget_models_are_skipped_without_a_call(self) -> None:
        health = ModelHealth()
        health.mark_dead("nvidia", "dead", "HTTP 410")
        health.cool_down("nvidia", "cooling", 60, reason="HTTP 429")
        for _ in range(20):
            health.record_request("openrouter", "busy:free")
        skipped = FakeProvider("nvidia", AssertionError("must not be called"))
        over_budget = FakeProvider("openrouter", AssertionError("must not be called"))
        working = FakeProvider("colab", "colab answer")

        response = await FallbackChain(health).generate(
            [Tier(skipped, "dead"), Tier(skipped, "cooling"), Tier(over_budget, "busy:free"), Tier(working, "qwen")], _TEXT
        )

        self.assertEqual(response.text, "colab answer")
        self.assertEqual((skipped.calls, over_budget.calls), ([], []))

    async def test_skip_tier_error_counts_as_not_attempted(self) -> None:
        health = ModelHealth()
        busy = FakeProvider("colab", SkipTierError("qwen", "busy", "colab"))
        working = FakeProvider("nvidia", "ok")
        chain = FallbackChain(health)

        response = await chain.generate([Tier(busy, "qwen"), Tier(working, "ultra")], _TEXT)

        self.assertEqual(response.text, "ok")
        self.assertEqual(chain.fallback_count, 0)
        self.assertNotIn("colab:qwen", health.get_status()["models"])

    async def test_a_colab_tier_that_is_not_ready_never_holds_up_the_reply(self) -> None:
        # Until the on-demand Colab session is READY its provider refuses at
        # once; the API tier answers inside the same deadline with its own
        # full time limit (a thinking Colab tier's half share is not taken).
        # The API tier needs more than half the window (0.6 of 1.0 s): it
        # answers only if the skipped Colab tier took no share of the time.
        not_ready = FakeProvider("colab", SkipTierError("qwen", "colab not ready", "colab"))
        api = FakeProvider("nvidia", ("sleep", 0.6))
        deadline = time.monotonic() + 1.0
        started = time.monotonic()

        response = await FallbackChain(ModelHealth()).generate(
            [
                Tier(not_ready, "qwen", timeout_s=110, options={"thinking": True}, shares_time=True),
                Tier(api, "ultra", timeout_s=0.9),
            ],
            _TEXT,
            deadline=deadline,
        )

        self.assertEqual(response.text, "late")
        self.assertLess(time.monotonic() - started, 0.85)
        self.assertEqual(len(not_ready.calls), 1)

    async def test_a_slow_ready_colab_tier_leaves_the_api_tier_half_the_window(self) -> None:
        # Colab READY but slow, thinking off (COLAB_CHAT_THINK=false): its
        # own 45 s limit would take nearly the whole reply window; the API
        # tier after it must still get the other half.
        slow_colab = FakeProvider("colab", ("sleep", 5))
        api = FakeProvider("nvidia", ("sleep", 0.3))
        started = time.monotonic()

        response = await FallbackChain(ModelHealth()).generate(
            [
                Tier(slow_colab, "qwen", timeout_s=45, options={"thinking": False}, shares_time=True),
                Tier(api, "ultra", timeout_s=40),
            ],
            _TEXT,
            deadline=time.monotonic() + 1.0,
        )

        self.assertEqual(response.text, "late")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual((len(slow_colab.calls), len(api.calls)), (1, 1))

    async def test_a_ready_colab_tier_runs_until_the_api_reserve_of_a_long_window(self) -> None:
        # The reply window (REPLY_WINDOW_SECONDS, 90 s): Colab may run
        # until COLAB_API_RESERVE_SECONDS before the deadline, not just half
        # of it; in a short window it still gets half.
        for left, limit in ((88.0, 88.0 - COLAB_API_RESERVE_SECONDS), (30.0, 15.0)):
            with self.subTest(left=left):
                clock = _Clock()
                limits: list[float] = []
                real_timeout = asyncio.timeout

                def recording_timeout(delay: float, limits=limits, real_timeout=real_timeout):
                    limits.append(delay)
                    return real_timeout(delay)

                with patch.object(fallback_chain.asyncio, "timeout", recording_timeout):
                    response = await FallbackChain(ModelHealth(), clock=clock).generate(
                        [
                            Tier(FakeProvider("colab", "colab"), "qwen", timeout_s=70, options={"thinking": False}, shares_time=True),
                            Tier(FakeProvider("nvidia"), "ultra", timeout_s=40),
                        ],
                        _TEXT,
                        deadline=clock.now + left,
                    )

                self.assertEqual(response.text, "colab")
                self.assertEqual(limits, [limit])

    async def test_a_colab_tier_thinking_at_low_for_a_plain_request_keeps_its_colab_share(self) -> None:
        # COLAB_CHAT_THINK=low (the default) makes the Colab chat and vision
        # tiers think on every request: a plain one still gets the Colab
        # share (until the API reserve), only a thinking request is held to
        # the thinking tiers' half.
        for needs_thinking, limit in ((False, 88.0 - COLAB_API_RESERVE_SECONDS), (True, 44.0)):
            with self.subTest(needs_thinking=needs_thinking):
                clock = _Clock()
                limits: list[float] = []
                real_timeout = asyncio.timeout

                def recording_timeout(delay: float, limits=limits, real_timeout=real_timeout):
                    limits.append(delay)
                    return real_timeout(delay)

                with patch.object(fallback_chain.asyncio, "timeout", recording_timeout):
                    response = await FallbackChain(ModelHealth(), clock=clock).generate(
                        [
                            Tier(
                                FakeProvider("colab", "colab"),
                                "qwen",
                                timeout_s=70,
                                options={"think": "low", "thinking": True},
                                shares_time=True,
                            ),
                            Tier(FakeProvider("nvidia"), "ultra", timeout_s=40),
                        ],
                        _TEXT,
                        deadline=clock.now + 88.0,
                        disable_thinking=not needs_thinking,
                    )

                self.assertEqual(response.text, "colab")
                self.assertEqual(limits, [limit])

    async def test_a_colab_tier_with_nothing_usable_after_it_gets_all_the_time(self) -> None:
        # The API tier after Colab is cooling down, dead, or cannot take the
        # image: capping Colab at half the window would fail the reply with
        # the other half unused. Colab needs 0.3 s of a 0.5 s window.
        for case in ("cooling", "dead", "text-only"):
            with self.subTest(case=case):
                health = ModelHealth()
                if case == "cooling":
                    health.cool_down("nvidia", "ultra", 60, reason="HTTP 429")
                elif case == "dead":
                    health.mark_dead("nvidia", "ultra", "HTTP 410")
                slow_colab = FakeProvider("colab", ("sleep", 0.3))
                api = FakeProvider("nvidia", "api")
                modalities = frozenset({"text"}) if case == "text-only" else frozenset({"text", "image"})

                response = await FallbackChain(health).generate(
                    [
                        Tier(slow_colab, "qwen", timeout_s=45, options={"thinking": False}, shares_time=True),
                        Tier(api, "ultra", modalities=modalities),
                    ],
                    _IMAGE,
                    deadline=time.monotonic() + 0.5,
                )

                self.assertEqual(response.text, "late")
                self.assertEqual(api.calls, [])

    async def test_a_thinking_tier_with_nothing_usable_after_it_gets_all_the_time(self) -> None:
        health = ModelHealth()
        health.mark_dead("openrouter", "qwen:free", "HTTP 404", retry_after=600)
        thinker = FakeProvider("nvidia", ("sleep", 0.3))

        response = await FallbackChain(health).generate(
            [Tier(thinker, "ultra", options={"thinking": True}), Tier(FakeProvider("openrouter"), "qwen:free")],
            _TEXT,
            deadline=time.monotonic() + 0.5,
        )

        self.assertEqual(response.text, "late")

    async def test_only_skips_is_a_failure(self) -> None:
        busy = FakeProvider("colab", SkipTierError("qwen", "busy", "colab"))

        with self.assertRaises(AllProvidersFailedError):
            await FallbackChain(ModelHealth()).generate([Tier(busy, "qwen")], _TEXT)


class ErrorClassTests(unittest.IsolatedAsyncioTestCase):
    async def test_next_tier_after_an_error_counts_a_fallback(self) -> None:
        broken = FakeProvider("nvidia", ProviderError("m1", 500, "boom"))
        working = FakeProvider("openrouter", "ok")
        chain = FallbackChain(ModelHealth())

        response = await chain.generate([Tier(broken, "m1"), Tier(working, "m2")], _TEXT)

        self.assertEqual(response.text, "ok")
        self.assertEqual(chain.fallback_count, 1)

    async def test_only_rate_limits_mean_busy(self) -> None:
        limited = FakeProvider("nvidia", RateLimitError("m", 60, provider="nvidia"))
        with self.assertRaises(AllModelsRateLimitedError):
            await FallbackChain(ModelHealth()).generate([Tier(limited, "m1"), Tier(limited, "m2")], _TEXT)

    async def test_rate_limit_plus_any_failure_is_a_failure(self) -> None:
        for failure in (
            TransientError("m", 503, "busy", "nvidia"),
            ProviderError("m", 500, "boom"),
            ValueError("provider broke its contract"),
            ProviderResponse(text=None),  # empty completion
        ):
            with self.subTest(failure=type(failure).__name__):
                limited = FakeProvider("openrouter", RateLimitError("m", 60, provider="openrouter"))
                failing = FakeProvider("nvidia", failure)
                with self.assertRaises(AllProvidersFailedError):
                    await FallbackChain(ModelHealth()).generate(
                        [Tier(failing, "m1"), Tier(limited, "m2:free")], _TEXT
                    )

    async def test_dead_plus_rate_limited_is_still_busy(self) -> None:
        # The rate-limited model will come back, so "busy, try later" is true.
        health = ModelHealth()
        health.mark_dead("nvidia", "m1", "HTTP 410")
        health.cool_down("nvidia", "m2", 60, reason="HTTP 429")
        provider = FakeProvider("nvidia")

        with self.assertRaises(AllModelsRateLimitedError):
            await FallbackChain(health).generate([Tier(provider, "m1"), Tier(provider, "m2")], _TEXT)

    async def test_all_dead_is_a_failure_naming_them(self) -> None:
        health = ModelHealth()
        health.mark_dead("nvidia", "m1", "HTTP 410")
        health.mark_dead("nvidia", "m2", "HTTP 404", retry_after=60)
        provider = FakeProvider("nvidia")

        with self.assertRaises(AllProvidersFailedError) as ctx:
            await FallbackChain(health).generate([Tier(provider, "m1"), Tier(provider, "m2")], _TEXT)

        self.assertIn("Dead: nvidia:m1, nvidia:m2", str(ctx.exception))
        self.assertEqual(provider.calls, [])

    async def test_auth_cooldown_is_a_failure_not_busy(self) -> None:
        health = ModelHealth()
        health.record_failure("nvidia", "m1", AuthError("m1", 401, "bad key", "nvidia"))
        provider = FakeProvider("nvidia")

        with self.assertRaises(AllProvidersFailedError):
            await FallbackChain(health).generate([Tier(provider, "m2")], _TEXT)
        self.assertEqual(provider.calls, [])

    async def test_empty_completions_fall_through_and_fail_as_empty(self) -> None:
        empty = FakeProvider("nvidia", ProviderResponse(text=None, finish_reason="length"))
        chain = FallbackChain(ModelHealth())

        with self.assertRaises(AllProvidersFailedError) as ctx:
            await chain.generate([Tier(empty, "m1"), Tier(empty, "m2")], _TEXT)

        self.assertIsInstance(ctx.exception.last_error, EmptyCompletionError)
        self.assertIn("empty completion", str(ctx.exception))
        self.assertIn("finish_reason=length", str(ctx.exception))
        self.assertEqual(len(empty.calls), 2)

    async def test_tool_calls_or_images_without_text_are_answers(self) -> None:
        call = {"id": "c1", "type": "function", "function": {"name": "recall_memory", "arguments": "{}"}}
        for response in (
            ProviderResponse(text=None, tool_calls=[call]),
            ProviderResponse(text=None, images=["data:image/png;base64,abc"]),
        ):
            with self.subTest(response=response):
                provider = FakeProvider("nvidia", response)
                result = await FallbackChain(ModelHealth()).generate([Tier(provider, "m")], _TEXT)
                self.assertIs(result, response)

    async def test_no_targets(self) -> None:
        with self.assertRaisesRegex(AllProvidersFailedError, "No fallback targets configured"):
            await FallbackChain(ModelHealth()).generate([], _TEXT)


class HealthRecordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_outcomes_update_model_health(self) -> None:
        health = ModelHealth()
        gone = FakeProvider("nvidia", ModelGoneError("m1", 410, "gone", "nvidia", eol="2026-07-27T00:00:00Z"))
        limited = FakeProvider(
            "openrouter",
            RateLimitError(
                "m2:free", 5, provider="openrouter", detail="Rate limit exceeded: free-models-per-min."
            ),
        )
        working = FakeProvider("colab", "ok")

        await FallbackChain(health).generate([Tier(gone, "m1"), Tier(limited, "m2:free"), Tier(working, "m3")], _TEXT)

        self.assertTrue(health.is_dead("nvidia", "m1"))
        # An account-level free-model 429 takes out every :free id.
        self.assertFalse(health.is_available("openrouter", "other:free"))
        self.assertIsNotNone(health.get_status()["models"]["colab:m3"]["last_ok_at"])

    async def test_an_upstream_429_moves_on_to_the_next_free_model(self) -> None:
        health = ModelHealth()
        upstream = RateLimitError(
            "qwen:free", None, provider="openrouter", detail="qwen:free is temporarily rate-limited upstream."
        )
        limited = FakeProvider("openrouter", upstream)
        other = FakeProvider("openrouter", "from ultra")

        response = await FallbackChain(health).generate(
            [Tier(limited, "qwen:free"), Tier(other, "ultra:free")], _TEXT
        )

        self.assertEqual(response.text, "from ultra")
        # Only the rate-limited model cools down; the account stays usable.
        self.assertFalse(health.is_available("openrouter", "qwen:free"))
        self.assertTrue(health.is_available("openrouter", "ultra:free"))

    async def test_a_hung_model_cools_down_after_two_timeouts(self) -> None:
        health = ModelHealth()
        hung = FakeProvider("nvidia", ("sleep", 5))
        backup = FakeProvider("openrouter", "backup")
        chain = FallbackChain(health)
        tiers = [Tier(hung, "muse", timeout_s=0.02), Tier(backup, "q:free")]

        for _ in range(2):
            await chain.generate(tiers, _TEXT)
        self.assertEqual(len(hung.calls), 2)

        await chain.generate(tiers, _TEXT)
        # Cooling down: the third request goes straight to the backup.
        self.assertEqual(len(hung.calls), 2)
        self.assertEqual(len(backup.calls), 3)

    async def test_a_hung_model_the_deadline_cuts_off_still_cools_down(self) -> None:
        # Reply-window requests: the deadline, not the tier's own (default,
        # longer) limit, stops a hung first tier. Once it had at least
        # min_tier_seconds, that timeout is held against the model too.
        health = ModelHealth()
        hung = FakeProvider("nvidia", ("sleep", 5))
        backup = FakeProvider("openrouter", "backup")
        chain = FallbackChain(health, min_tier_seconds=0.05)
        tiers = [Tier(hung, "muse"), Tier(backup, "q:free")]

        for _ in range(2):
            with self.assertRaises(AllProvidersFailedError) as ctx:
                await chain.generate(tiers, _TEXT, deadline=time.monotonic() + 0.1)
            self.assertIn("deadline", str(ctx.exception))
        self.assertFalse(health.is_available("nvidia", "muse"))

        response = await chain.generate(tiers, _TEXT, deadline=time.monotonic() + 0.1)

        self.assertEqual(response.text, "backup")
        self.assertEqual((len(hung.calls), len(backup.calls)), (2, 1))

    async def test_a_callers_own_budget_is_not_held_against_the_model(self) -> None:
        # The search planner's short budget, not the reply window, cut the
        # tier off: however long the slice, no strike (audit finding 5a-F1).
        health = ModelHealth()
        slow = FakeProvider("nvidia", ("sleep", 5))
        chain = FallbackChain(health, min_tier_seconds=0.05)
        tiers = [Tier(slow, "super", timeout_s=30)]

        for _ in range(3):
            with self.assertRaises(DeadlineExceededError):
                await chain.generate(
                    tiers, _TEXT, deadline=time.monotonic() + 0.1, hold_deadline_timeouts=False
                )

        self.assertTrue(health.is_available("nvidia", "super"))
        self.assertEqual(len(slow.calls), 3)

    async def test_a_full_tier_timeout_counts_even_under_a_callers_budget(self) -> None:
        health = ModelHealth()
        hung = FakeProvider("nvidia", ("sleep", 5))
        chain = FallbackChain(health)
        tiers = [Tier(hung, "super", timeout_s=0.02)]

        for _ in range(2):
            with self.assertRaises(AllProvidersFailedError):
                await chain.generate(
                    tiers, _TEXT, deadline=time.monotonic() + 1.0, hold_deadline_timeouts=False
                )

        self.assertFalse(health.is_available("nvidia", "super"))

    async def test_a_colab_tiers_short_half_share_is_not_held_against_it(self) -> None:
        # A slow but healthy Colab tier (thinking off) cut off by its half of
        # the time left is cut off by the deadline: under MIN_TIER_SECONDS,
        # or under a caller's own budget, that is no strike.
        for min_tier_seconds, hold in ((5.0, True), (0.05, False)):
            with self.subTest(min_tier_seconds=min_tier_seconds, hold_deadline_timeouts=hold):
                health = ModelHealth()
                slow_colab = FakeProvider("colab", ("sleep", 5))
                api = FakeProvider("nvidia", "api")
                chain = FallbackChain(health, min_tier_seconds=min_tier_seconds)
                tiers = [
                    Tier(slow_colab, "qwen", timeout_s=45, options={"thinking": False}, shares_time=True),
                    Tier(api, "ultra"),
                ]

                for _ in range(2):
                    response = await chain.generate(
                        tiers, _TEXT, deadline=time.monotonic() + 0.2, hold_deadline_timeouts=hold
                    )
                    self.assertEqual(response.text, "api")

                self.assertEqual(len(slow_colab.calls), 2)
                self.assertNotIn("colab:qwen", health.get_status()["models"])

    async def test_a_colab_tier_thinking_at_low_for_a_plain_request_is_held_to_its_timeouts(self) -> None:
        # COLAB_CHAT_THINK=low makes the Colab chat tier think on every
        # request; thinking is only exempt when the request needs it
        # (disable_thinking=False, the thinking role's tiers).
        for needs_thinking in (False, True):
            with self.subTest(needs_thinking=needs_thinking):
                health = ModelHealth()
                slow_colab = FakeProvider("colab", ("sleep", 5))
                api = FakeProvider("nvidia", "api")
                chain = FallbackChain(health, min_tier_seconds=0.05)
                tiers = [
                    Tier(slow_colab, "qwen", timeout_s=45, options={"think": "low", "thinking": True}, shares_time=True),
                    Tier(api, "ultra"),
                ]

                for _ in range(3):
                    response = await chain.generate(
                        tiers, _TEXT, deadline=time.monotonic() + 0.2, disable_thinking=not needs_thinking
                    )
                    self.assertEqual(response.text, "api")

                # A plain request: two timeouts cool qwen down, the third
                # goes straight to the API tier. A thinking one: never.
                self.assertEqual(len(slow_colab.calls), 3 if needs_thinking else 2)
                self.assertEqual(health.is_available("colab", "qwen"), needs_thinking)

    async def test_a_hung_colab_tier_with_a_long_enough_half_share_still_cools_down(self) -> None:
        health = ModelHealth()
        hung_colab = FakeProvider("colab", ("sleep", 5))
        api = FakeProvider("nvidia", "api")
        chain = FallbackChain(health, min_tier_seconds=0.05)
        tiers = [
            Tier(hung_colab, "qwen", timeout_s=45, options={"thinking": False}, shares_time=True),
            Tier(api, "ultra"),
        ]

        for _ in range(2):
            # The half share ran out, not the deadline: the API tier still answers.
            response = await chain.generate(tiers, _TEXT, deadline=time.monotonic() + 0.2)
            self.assertEqual(response.text, "api")
        self.assertFalse(health.is_available("colab", "qwen"))

        await chain.generate(tiers, _TEXT, deadline=time.monotonic() + 0.2)
        self.assertEqual((len(hung_colab.calls), len(api.calls)), (2, 3))


class SameWalkTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """A model that timed out after a fair try is not called again in the walk."""

    async def _walk(self, chain: FallbackChain, tiers: list[Tier], **kwargs: Any):
        """``(answer text or the error, outcomes, summary line)`` of one walk."""
        trace = request_trace.RequestTrace(request_id="r1")
        token = request_trace.activate(trace)
        try:
            with request_trace.phase(request_trace.ANSWER):
                try:
                    result: Any = (await chain.generate(tiers, _TEXT, **kwargs)).text
                except AllProvidersFailedError as exc:
                    result = exc
        finally:
            request_trace.deactivate(token)
        [walk] = trace.walks
        return result, [attempt.outcome for attempt in walk.attempts], request_trace.summary_line(trace)

    async def test_a_model_that_timed_out_is_skipped_by_its_later_tiers(self) -> None:
        # A thinking request: the thinking tier on Ultra hung its whole 0.05 s,
        # so the chat tier on Ultra (thinking off) is skipped, as on
        # 2026-10-16 it was not (request a05be403 waited on Ultra twice).
        # A plain timeout marks the model the same way.
        for thinking in (True, False):
            with self.subTest(first_tier_thinking=thinking):
                health = ModelHealth()
                hung = FakeProvider("nvidia", ("sleep", 5))
                dots = FakeProvider("openrouter", "dots answer")
                tiers = [
                    Tier(hung, "ultra", timeout_s=0.05, options={"thinking": thinking}),
                    Tier(hung, "ultra", timeout_s=0.05, options={"thinking": False}),
                    Tier(dots, "dots:free", options={"thinking": False}),
                ]

                with self.assertLogs("linebot", "INFO") as logs:
                    text, outcomes, line = await self._walk(FallbackChain(health, min_tier_seconds=0.01), tiers)

                self.assertEqual(text, "dots answer")
                self.assertEqual(len(hung.calls), 1)
                self.assertEqual(outcomes, ["timeout", "skip(timed out earlier)", "ok"])
                self.assertRegex(
                    line, r"tiers=\[nvidia:ultra timeout \d+\.\ds,skip\(timed out earlier\), openrouter:dots:free ok "
                )
                self.assertIn("Skipping nvidia:ultra (timed out earlier)", "\n".join(logs.output))
                # ModelHealth is untouched: a thinking timeout earns no strike,
                # a plain one its one strike (two in a row cool it down).
                status = health.get_status(include_errors=True)["models"].get("nvidia:ultra")
                if thinking:
                    self.assertIsNone(status)
                else:
                    self.assertIn("no answer within 0.05s", status["last_error"])
                self.assertTrue(health.is_available("nvidia", "ultra"))

    async def test_a_sliver_or_a_try_shorter_than_the_later_tier_does_not_skip_it(self) -> None:
        # (min_tier_seconds, deadline in s or None, thinking tier timeout_s):
        # the thinking half (0.15 s of 0.3) is a sliver under 0.2 s; over
        # 0.01 s it is a try, but shorter than the later tier's 60 s (as
        # dots-3 got 15 s of the half in a05be403, then answered with thinking
        # off); a tier's own timeout_s under min_tier_seconds is no try either.
        for min_tier_seconds, deadline, timeout_s in ((0.2, 0.3, 60.0), (0.01, 0.3, 60.0), (0.2, None, 0.05)):
            with self.subTest(min_tier_seconds=min_tier_seconds, deadline=deadline, timeout_s=timeout_s):
                thinker = FakeProvider("nvidia", ("sleep", 5), "plain answer")
                tiers = [
                    Tier(thinker, "ultra", timeout_s=timeout_s, options={"thinking": True}),
                    Tier(thinker, "ultra", options={"thinking": False}),
                ]

                text, outcomes, _ = await self._walk(
                    FallbackChain(ModelHealth(), min_tier_seconds=min_tier_seconds),
                    tiers,
                    deadline=None if deadline is None else time.monotonic() + deadline,
                )

                self.assertEqual(text, "plain answer")
                self.assertEqual(outcomes, ["timeout", "ok"])
                self.assertEqual([call["disable_thinking"] for call in thinker.calls], [False, True])

    async def test_a_try_the_thinking_half_cut_short_skips_a_later_tier_that_gives_hardly_longer(self) -> None:
        # The half (0.25 s of 0.5) cuts the 0.4 s thinking tier short, as it
        # cuts NIM Ultra's 30 s thinking tier to 25 s or less when a search
        # stage took 60 s or more. A chat tier on the model that would give it
        # no longer (0.2 s, as Ultra's 25 s) or less than min_tier_seconds
        # (0.1 s) longer is skipped; one that gives it clearly longer still
        # runs (dots-3's 40 s after 15 s of the half in a05be403).
        for later_timeout_s, expected in (
            (0.2, ("dots answer", ["timeout", "skip(timed out earlier)", "ok"], 1)),
            (0.3, ("dots answer", ["timeout", "skip(timed out earlier)", "ok"], 1)),
            (0.4, ("plain answer", ["timeout", "ok"], 2)),
        ):
            with self.subTest(later_timeout_s=later_timeout_s):
                nim = FakeProvider("nvidia", ("sleep", 5), "plain answer")
                dots = FakeProvider("openrouter", "dots answer")
                tiers = [
                    Tier(nim, "ultra", timeout_s=0.4, options={"thinking": True}),
                    Tier(nim, "ultra", timeout_s=later_timeout_s, options={"thinking": False}),
                    Tier(dots, "dots:free", options={"thinking": False}),
                ]

                text, outcomes, _ = await self._walk(
                    FallbackChain(ModelHealth(), min_tier_seconds=0.1), tiers, deadline=time.monotonic() + 0.5
                )

                self.assertEqual((text, outcomes, len(nim.calls)), expected)

    async def test_walks_that_share_timed_out_skip_the_model_in_each(self) -> None:
        # A tool loop's rounds: the second round does not wait on Ultra again.
        ultra = FakeProvider("nvidia", ("sleep", 5), "ultra answer")
        dots = FakeProvider("openrouter", "dots answer")
        chain = FallbackChain(ModelHealth(), min_tier_seconds=0.01)
        tiers = [
            Tier(ultra, "ultra", timeout_s=0.05, options={"thinking": False}),
            Tier(dots, "dots:free", options={"thinking": False}),
        ]
        timed_out: fallback_chain.TimedOutModels = {}

        first = await self._walk(chain, tiers, timed_out=timed_out)
        second = await self._walk(chain, tiers, timed_out=timed_out)

        self.assertEqual(first[:2], ("dots answer", ["timeout", "ok"]))
        self.assertEqual(second[:2], ("dots answer", ["skip(timed out earlier)", "ok"]))
        self.assertEqual(len(ultra.calls), 1)
        self.assertEqual(timed_out, {("nvidia", "ultra"): math.inf})

    async def test_the_next_walk_calls_the_model_again(self) -> None:
        health = ModelHealth()
        ultra = FakeProvider("nvidia", ("sleep", 5), "ultra answer")
        dots = FakeProvider("openrouter", "dots answer")
        chain = FallbackChain(health, min_tier_seconds=0.01)
        tiers = [
            Tier(ultra, "ultra", timeout_s=0.05, options={"thinking": True}),
            Tier(ultra, "ultra", timeout_s=0.05, options={"thinking": False}),
            Tier(dots, "dots:free", options={"thinking": False}),
        ]

        first = await self._walk(chain, tiers)
        second = await self._walk(chain, tiers)

        self.assertEqual(first[:2], ("dots answer", ["timeout", "skip(timed out earlier)", "ok"]))
        self.assertEqual(second[:2], ("ultra answer", ["ok"]))
        self.assertEqual([call["disable_thinking"] for call in ultra.calls], [False, False])

    async def test_an_error_other_than_a_timeout_does_not_mark_the_model(self) -> None:
        # ModelHealth decides: the first 503 is one strike, so the chat tier
        # still calls Ultra; the second cools it down for the next request.
        health = ModelHealth()
        busy = FakeProvider("nvidia", TransientError("ultra", 503, "busy", "nvidia"))
        dots = FakeProvider("openrouter", "dots answer")
        tiers = [
            Tier(busy, "ultra", options={"thinking": True}),
            Tier(busy, "ultra", options={"thinking": False}),
            Tier(dots, "dots:free", options={"thinking": False}),
        ]

        text, outcomes, _ = await self._walk(FallbackChain(health, min_tier_seconds=0.01), tiers)

        self.assertEqual(text, "dots answer")
        self.assertEqual(outcomes, ["error(TransientError 503)", "error(TransientError 503)", "ok"])
        self.assertEqual(len(busy.calls), 2)
        self.assertFalse(health.is_available("nvidia", "ultra"))

    async def test_the_same_model_id_on_another_provider_is_still_called(self) -> None:
        # NIM Ultra hanging says nothing about Ultra :free behind OpenRouter
        # (they share an id here to show the provider is part of the key).
        nim = FakeProvider("nvidia", ("sleep", 5))
        openrouter = FakeProvider("openrouter", "openrouter answer")
        tiers = [
            Tier(nim, "ultra", timeout_s=0.05, options={"thinking": True}),
            Tier(openrouter, "ultra", options={"thinking": False}),
        ]

        text, outcomes, _ = await self._walk(FallbackChain(ModelHealth(), min_tier_seconds=0.01), tiers)

        self.assertEqual(text, "openrouter answer")
        self.assertEqual(outcomes, ["timeout", "ok"])

    async def test_a_thinking_tier_with_only_timed_out_models_after_it_gets_all_the_time(self) -> None:
        # Super thinking needs 0.3 s of a 0.5 s window; the only tier after
        # it is Ultra, which already timed out, so Super is not held to the
        # thinking half (0.25 s) that would leave Ultra nothing to use.
        ultra = FakeProvider("nvidia", ("sleep", 5))
        super_ = FakeProvider("nvidia", ("sleep", 0.3))
        tiers = [
            Tier(ultra, "ultra", timeout_s=0.05, options={"thinking": True}),
            Tier(super_, "super", options={"thinking": True}),
            Tier(ultra, "ultra", options={"thinking": False}),
        ]

        text, outcomes, _ = await self._walk(
            FallbackChain(ModelHealth(), min_tier_seconds=0.01), tiers, deadline=time.monotonic() + 0.5
        )

        self.assertEqual(text, "late")
        self.assertEqual(outcomes, ["timeout", "ok"])
        self.assertEqual(len(ultra.calls), 1)


class ForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_kwargs_tools_and_json_schema_reach_the_provider(self) -> None:
        provider = FakeProvider("nvidia")
        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        schema = {"type": "object"}

        await FallbackChain(ModelHealth()).generate(
            [Tier(provider, "m")],
            _TEXT,
            tools=tools,
            tool_choice="auto",
            json_schema=schema,
            temperature=0.1,
            max_tokens=384,
        )

        call = provider.calls[0]
        self.assertEqual(
            (call["tools"], call["tool_choice"], call["json_schema"], call["temperature"], call["max_tokens"]),
            (tools, "auto", schema, 0.1, 384),
        )

    async def test_unset_optional_arguments_are_not_sent(self) -> None:
        provider = FakeProvider("nvidia")

        await FallbackChain(ModelHealth()).generate([Tier(provider, "m")], _TEXT)

        self.assertEqual(provider.calls[0], {"model": "m", "disable_thinking": False, "options": {}})

    async def test_each_tier_gets_its_own_options(self) -> None:
        # The role table's per-tier options (json_mode, effort, ...) travel
        # with the tier; "thinking" also decides disable_thinking.
        nvidia = FakeProvider("nvidia", TransientError("super", 503, "busy", "nvidia"))
        openrouter = FakeProvider("openrouter", "ok")
        tiers = [
            Tier(nvidia, "super", options={"json_mode": "json_schema", "thinking": False}),
            Tier(openrouter, "super:free", options={"json_mode": "json_object"}),
        ]

        await FallbackChain(ModelHealth()).generate(tiers, _TEXT, json_schema={"type": "object"})

        self.assertEqual(
            (nvidia.calls[0]["options"], nvidia.calls[0]["disable_thinking"]),
            ({"json_mode": "json_schema", "thinking": False}, True),
        )
        self.assertEqual(
            (openrouter.calls[0]["options"], openrouter.calls[0]["disable_thinking"]),
            ({"json_mode": "json_object"}, False),
        )


if __name__ == "__main__":
    unittest.main()
