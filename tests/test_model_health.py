"""ModelHealth: dead models, cooldowns and provider budgets on a fake clock."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.providers.base import (
    AuthError,
    CreditsError,
    EmptyCompletionError,
    ModelGoneError,
    PayloadError,
    RateLimitError,
    SkipTierError,
    TransientError,
)
from src.services import model_health as model_health_module
from src.services.model_health import COOLDOWN, DEAD, RATE_LIMIT, ModelHealth

_ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
_SUPER = "nvidia/nemotron-3-super-120b-a12b"
_QWEN = "qwen/qwen3.8-27b:free"
_ULTRA_FREE = "nvidia/nemotron-3-ultra-550b-a55b:free"
# OpenRouter 429 bodies (trimmed): the account's free-model cap, and a limit
# one model's upstream provider passed on (seen live on 2026-09-30).
_ACCOUNT_429 = (
    '{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 credits to '
    'unlock 1000 free model requests per day","code":429}}'
)
_UPSTREAM_429 = (
    '{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":'
    '"qwen/qwen3.8-27b:free is temporarily rate-limited upstream. Please retry shortly"}}}'
)


def _openrouter_429(retry_after: int | None, detail: str) -> RateLimitError:
    return RateLimitError(_QWEN, retry_after, provider="openrouter", detail=detail)


class _Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _HealthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.health = ModelHealth(clock=self.clock)


class DeadModelTests(_HealthTest):
    def test_mark_dead_makes_a_model_permanently_unavailable(self) -> None:
        with patch.object(model_health_module.logger, "error") as log_error:
            self.health.mark_dead("nvidia", "m", "HTTP 410, end of life 2026-07-27T00:00:00Z")
            self.health.mark_dead("nvidia", "m", "HTTP 410, end of life 2026-07-27T00:00:00Z")

        self.assertTrue(self.health.is_dead("nvidia", "m"))
        self.assertFalse(self.health.is_available("nvidia", "m"))
        # Keys are provider:model; the same id elsewhere is unaffected.
        self.assertTrue(self.health.is_available("openrouter", "m"))
        status = self.health.get_status()["models"]["nvidia:m"]
        self.assertEqual(
            (status["available"], status["state"], status["dead_until"]), (False, DEAD, None)
        )
        self.assertEqual(status["reason"], "HTTP 410, end of life 2026-07-27T00:00:00Z")
        log_error.assert_called_once()
        self.clock.now += 365 * 86400
        self.assertTrue(self.health.is_dead("nvidia", "m"))

    def test_mark_dead_with_retry_after_expires(self) -> None:
        self.health.mark_dead("nvidia", "m", "HTTP 404, retry in 6h", retry_after=6 * 3600)
        self.assertTrue(self.health.is_dead("nvidia", "m"))
        self.assertEqual(
            self.health.get_status()["models"]["nvidia:m"]["dead_until"], self.clock.now + 6 * 3600
        )

        self.clock.now += 6 * 3600
        self.assertFalse(self.health.is_dead("nvidia", "m"))
        self.assertTrue(self.health.is_available("nvidia", "m"))
        status = self.health.get_status()["models"]["nvidia:m"]
        self.assertEqual((status["state"], status["dead_until"]), ("ok", None))

    def test_410_is_dead_for_the_process_and_keeps_the_eol_date(self) -> None:
        error = ModelGoneError(_ULTRA, 410, "gone", "nvidia", eol="2026-07-27T00:00:00Z")
        self.health.record_failure("nvidia", _ULTRA, error)

        blocked = self.health.unavailable("nvidia", _ULTRA)
        self.assertEqual(blocked.kind, DEAD)
        self.assertEqual(blocked.reason, "HTTP 410, end of life 2026-07-27T00:00:00Z")
        self.assertEqual(self.health.get_status()["models"][f"nvidia:{_ULTRA}"]["eol"], "2026-07-27T00:00:00Z")
        self.clock.now += 30 * 86400
        self.assertTrue(self.health.is_dead("nvidia", _ULTRA))

    def test_404_is_dead_for_six_hours_on_nvidia_and_ten_minutes_on_openrouter(self) -> None:
        self.health.record_failure("nvidia", _ULTRA, ModelGoneError(_ULTRA, 404, "no such model", "nvidia"))
        self.health.record_failure("openrouter", _QWEN, ModelGoneError(_QWEN, 404, "No endpoints", "openrouter"))

        self.assertEqual(self.health.unavailable("nvidia", _ULTRA).reason, "HTTP 404, retry in 6h")
        self.assertEqual(self.health.unavailable("openrouter", _QWEN).reason, "HTTP 404, retry in 10min")
        self.clock.now += 600
        self.assertTrue(self.health.is_available("openrouter", _QWEN))
        self.assertTrue(self.health.is_dead("nvidia", _ULTRA))
        self.clock.now += 6 * 3600
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))

    def test_revive_ends_a_dead_mark_early(self) -> None:
        # The daily lifecycle check found the model listed again.
        self.health.mark_dead("nvidia", "m", "HTTP 410", eol="2026-07-27T00:00:00Z")

        self.assertTrue(self.health.revive("nvidia", "m"))

        self.assertTrue(self.health.is_available("nvidia", "m"))
        self.assertEqual(self.health.get_status()["models"]["nvidia:m"]["eol"], "2026-07-27T00:00:00Z")
        self.assertFalse(self.health.revive("nvidia", "m"))
        self.assertFalse(self.health.revive("nvidia", "never-seen"))

    def test_unknown_models_are_alive_and_not_tracked(self) -> None:
        self.assertFalse(self.health.is_dead("nvidia", "never-seen"))
        self.assertTrue(self.health.is_available("nvidia", "never-seen"))
        self.assertEqual(self.health.get_status()["models"], {})


class CooldownTests(_HealthTest):
    def test_nvidia_429_cools_only_that_model(self) -> None:
        self.health.record_failure("nvidia", _ULTRA, RateLimitError(_ULTRA, 30, provider="nvidia"))

        blocked = self.health.unavailable("nvidia", _ULTRA)
        self.assertEqual((blocked.kind, blocked.reason), (RATE_LIMIT, "HTTP 429"))
        self.assertTrue(self.health.is_available("nvidia", _SUPER))
        self.assertFalse(self.health.is_dead("nvidia", _ULTRA))
        self.clock.now += 30
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))

    def test_429_without_retry_after_cools_for_a_minute(self) -> None:
        self.health.record_failure("nvidia", _ULTRA, RateLimitError(_ULTRA, None, provider="nvidia"))
        self.clock.now += 59
        self.assertFalse(self.health.is_available("nvidia", _ULTRA))
        self.clock.now += 1
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))

    def test_openrouter_account_cap_429_cools_the_whole_account(self) -> None:
        self.health.record_failure("openrouter", _QWEN, _openrouter_429(5, _ACCOUNT_429))

        # The free-model caps are per account: every :free id is out.
        blocked = self.health.unavailable("openrouter", _ULTRA_FREE)
        self.assertEqual(blocked.kind, RATE_LIMIT)
        self.assertIn("account limit", blocked.reason)
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))
        self.clock.now += 599
        self.assertFalse(self.health.is_available("openrouter", _QWEN))
        self.clock.now += 1
        self.assertTrue(self.health.is_available("openrouter", _QWEN))

        self.health.record_failure("openrouter", _QWEN, _openrouter_429(3600, _ACCOUNT_429))
        self.clock.now += 3599
        self.assertFalse(self.health.is_available("openrouter", _ULTRA_FREE))

    def test_openrouter_upstream_429_cools_only_that_model(self) -> None:
        self.health.record_failure("openrouter", _QWEN, _openrouter_429(None, _UPSTREAM_429))

        blocked = self.health.unavailable("openrouter", _QWEN)
        self.assertEqual((blocked.kind, blocked.reason), (RATE_LIMIT, "HTTP 429"))
        # The other free ids stay in rotation, and no provider cooldown is set.
        self.assertTrue(self.health.is_available("openrouter", _ULTRA_FREE))
        self.assertIsNone(self.health.get_status()["providers"]["openrouter"]["cooldown_until"])
        self.clock.now += 59
        self.assertFalse(self.health.is_available("openrouter", _QWEN))
        self.clock.now += 1
        self.assertTrue(self.health.is_available("openrouter", _QWEN))

        # Its Retry-After wins over the default minute.
        self.health.record_failure("openrouter", _QWEN, _openrouter_429(90, _UPSTREAM_429))
        self.clock.now += 89
        self.assertFalse(self.health.is_available("openrouter", _QWEN))
        self.clock.now += 1
        self.assertTrue(self.health.is_available("openrouter", _QWEN))

    def test_an_openrouter_429_that_names_no_account_cap_cools_only_that_model(self) -> None:
        self.health.record_failure("openrouter", _QWEN, RateLimitError(_QWEN, 5, provider="openrouter"))

        self.assertFalse(self.health.is_available("openrouter", _QWEN))
        self.assertTrue(self.health.is_available("openrouter", _ULTRA_FREE))

    def test_paid_openrouter_429_cools_only_that_model(self) -> None:
        self.health.record_failure(
            "openrouter", "qwen/qwen3.8-27b", RateLimitError("qwen/qwen3.8-27b", 20, provider="openrouter")
        )
        self.assertFalse(self.health.is_available("openrouter", "qwen/qwen3.8-27b"))
        self.assertTrue(self.health.is_available("openrouter", _QWEN))

    def test_a_refused_key_and_credit_errors_cool_the_provider_down(self) -> None:
        for error, seconds in (
            (AuthError(_ULTRA, 401, "Authentication failed", "nvidia"), 600),
            (AuthError(_ULTRA, 403, "Key limit exceeded (daily limit)", "nvidia"), 600),
            (CreditsError(_ULTRA, 402, "no credits", "nvidia"), 3600),
        ):
            with self.subTest(error=type(error).__name__):
                health = ModelHealth(clock=self.clock)
                health.record_failure("nvidia", _ULTRA, error)

                blocked = health.unavailable("nvidia", _SUPER)
                self.assertEqual(blocked.kind, COOLDOWN)
                self.assertTrue(blocked.reason.startswith("nvidia: HTTP"))
                status = health.get_status()["providers"]["nvidia"]
                self.assertEqual(status["cooldown_until"], self.clock.now + seconds)
                self.assertTrue(health.is_available("openrouter", _QWEN))

    def test_a_403_that_does_not_name_the_key_cools_only_that_model_briefly(self) -> None:
        # One function the key may not call (FLUX, the embedder), or an
        # OpenRouter moderation refusal: every other model stays in rotation.
        for provider, model, detail in (
            ("nvidia", "black-forest-labs/flux.1-dev", '{"status":403,"title":"Forbidden","detail":"Authorization failed"}'),
            ("openrouter", _QWEN, '{"error":{"code":403,"message":"qwen requires moderation. Your input was flagged"}}'),
        ):
            with self.subTest(provider=provider):
                health = ModelHealth(clock=self.clock)
                health.record_failure(provider, model, AuthError(model, 403, detail, provider))

                blocked = health.unavailable(provider, model)
                self.assertEqual((blocked.kind, blocked.reason), (COOLDOWN, "HTTP 403"))
                self.assertTrue(health.is_available(provider, _SUPER if provider == "nvidia" else _ULTRA_FREE))
                self.assertIsNone(health.get_status()["providers"][provider]["cooldown_until"])
                self.clock.now += 120
                self.assertTrue(health.is_available(provider, model))

    def test_two_transient_failures_in_a_row_cool_a_model_down(self) -> None:
        self.health.record_failure("nvidia", _ULTRA, TransientError(_ULTRA, 503, "ResourceExhausted", "nvidia"))
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))
        self.health.record_success("nvidia", _ULTRA)  # a success resets the count
        self.health.record_failure("nvidia", _ULTRA, TransientError(_ULTRA, 0, "no answer within 60s", "nvidia"))
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))

        self.health.record_failure("nvidia", _ULTRA, TransientError(_ULTRA, 524, "timeout", "nvidia"))

        blocked = self.health.unavailable("nvidia", _ULTRA)
        self.assertEqual(blocked.kind, COOLDOWN)
        self.assertEqual(blocked.reason, "2 transient failures in a row")
        self.clock.now += 120
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))

    def test_payload_and_empty_replies_only_record_the_error(self) -> None:
        for error in (
            PayloadError(_ULTRA, 422, "unknown field guided_json", "nvidia"),
            EmptyCompletionError(_ULTRA, provider="nvidia"),
            ValueError("provider broke its contract"),
        ):
            with self.subTest(error=type(error).__name__):
                self.health.record_failure("nvidia", _ULTRA, error)
                self.assertTrue(self.health.is_available("nvidia", _ULTRA))
        status = self.health.get_status(include_errors=True)["models"][f"nvidia:{_ULTRA}"]
        self.assertEqual(status["last_error"], "ValueError: provider broke its contract")
        self.assertEqual(status["last_error_at"], self.clock.now)

    def test_status_leaves_upstream_error_text_out_by_default(self) -> None:
        # Upstream bodies can quote account details; the public /health
        # must not echo them. Only an explicit detail view gets them.
        body = "Authorization failed: account acme-org lacks access to this model"
        self.health.record_failure("nvidia", _ULTRA, AuthError(_ULTRA, 403, body, "nvidia"))

        public = self.health.get_status()["models"][f"nvidia:{_ULTRA}"]
        self.assertNotIn("last_error", public)
        self.assertNotIn("acme-org", repr(self.health.get_status()))
        self.assertEqual(public["last_error_at"], self.clock.now)
        detail = self.health.get_status(include_errors=True)["models"][f"nvidia:{_ULTRA}"]
        self.assertIn("acme-org", detail["last_error"])

    def test_skip_tier_errors_are_not_recorded(self) -> None:
        self.health.record_failure("colab", "qwen3.8:27b", SkipTierError("qwen3.8:27b", "busy", "colab"))
        self.assertEqual(self.health.get_status()["models"], {})

    def test_success_is_recorded(self) -> None:
        self.health.record_success("nvidia", _ULTRA)
        self.assertEqual(self.health.get_status()["models"][f"nvidia:{_ULTRA}"]["last_ok_at"], self.clock.now)

    def test_cool_down_logs_the_reason(self) -> None:
        with patch.object(model_health_module.logger, "warning") as log_warning:
            self.health.cool_down("openrouter", _QWEN, 600, reason="HTTP 404 (no endpoints)")

        self.assertFalse(self.health.is_available("openrouter", _QWEN))
        log_warning.assert_called_once_with(
            f"HTTP 404 (no endpoints) for openrouter:{_QWEN}, cooling down for 10min"
        )


class BudgetTests(_HealthTest):
    def test_nvidia_budget_is_40_a_minute_shared_by_every_model(self) -> None:
        for index in range(40):
            self.assertTrue(self.health.is_available("nvidia", _ULTRA), index)
            self.health.record_request("nvidia", _ULTRA if index % 2 else _SUPER)

        for model in (_ULTRA, _SUPER, "meta/muse-glimmer-30b"):
            blocked = self.health.unavailable("nvidia", model)
            self.assertEqual(blocked.kind, RATE_LIMIT)
            self.assertIn("40 requests/minute", blocked.reason)
        self.clock.now += 60
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))

    def test_nvidia_daily_budget(self) -> None:
        for _ in range(10_000):
            self.health.record_request("nvidia", _ULTRA)
            self.clock.now += 2  # under 40 a minute
        self.assertIn("10000 requests/day", self.health.unavailable("nvidia", _ULTRA).reason)
        self.clock.now += 86_400 - 20_000 + 2
        self.assertTrue(self.health.is_available("nvidia", _ULTRA))

    def test_openrouter_free_budget_is_20_a_minute_and_the_daily_cap(self) -> None:
        health = ModelHealth(openrouter_free_daily_limit=25, clock=self.clock)
        for _ in range(20):
            health.record_request("openrouter", _QWEN)
        self.assertIn("20 requests/minute", health.unavailable("openrouter", _ULTRA_FREE).reason)

        self.clock.now += 60
        for _ in range(5):
            health.record_request("openrouter", _ULTRA_FREE)
        self.assertIn("25 requests/day", health.unavailable("openrouter", _QWEN).reason)
        status = health.get_status()["providers"]["openrouter"]
        self.assertEqual((status["rpd_used"], status["rpd_limit"], status["rpm_limit"]), (25, 25, 20))

    def test_paid_openrouter_and_colab_have_no_budget(self) -> None:
        for _ in range(100):
            self.health.record_request("openrouter", "qwen/qwen3.8-27b")
            self.health.record_request("colab", "qwen3.8:27b-mtp-q4_K_M")
        self.assertTrue(self.health.is_available("openrouter", _QWEN))
        self.assertTrue(self.health.is_available("colab", "qwen3.8:27b-mtp-q4_K_M"))
        status = self.health.get_status()["providers"]
        self.assertEqual(status["openrouter"]["rpd_used"], 0)
        self.assertNotIn("colab", status)

    def test_daily_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ModelHealth(openrouter_free_daily_limit=0)

    def test_status_lists_both_budgeted_providers(self) -> None:
        status = ModelHealth(clock=self.clock).get_status()
        self.assertEqual(sorted(status["providers"]), ["nvidia", "openrouter"])
        self.assertEqual(
            status["providers"]["nvidia"],
            {
                "rpm_used": 0,
                "rpm_limit": 40,
                "rpd_used": 0,
                "rpd_limit": 10_000,
                "cooldown_until": None,
                "cooldown_reason": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
