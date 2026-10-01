from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx

from src.config import (
    LEGACY_ENV_KEYS,
    REMOVED_ENV_KEYS,
    RETIRED_MODEL_IDS,
    Settings,
    legacy_config_warnings,
)
from src.services import line_service as line_service_module
from src.services import storage_service as storage_service_module
from src.utils import request_trace

# Settings the model role table replaced (Phase 4b).
_REMOVED_MODEL_FIELDS = (
    "nvidia_model",
    "nvidia_vision_model",
    "nvidia_utility_model",
    "nvidia_embedding_model",
    "router_model",
    "router_fallback_model",
    "agent_fallback_model",
    "agent_fallback_model_2",
    "vision_fallback_model",
    "vision_fallback_model_2",
    "utility_fallback_model",
    "image_gen_primary_model",
    "image_gen_fallback_model",
    "image_gen_steps",
    "image_gen_cfg_scale",
    "require_reasoning_models",
    "require_reasoning_tokens",
    "nvidia_thinking_enabled",
    "openrouter_reasoning_enabled",
    "openrouter_reasoning_effort",
    "openrouter_reasoning_exclude",
    "openrouter_thinking_budget",
)


class SettingsValidationTests(unittest.TestCase):
    def test_media_cleanup_delay_setting_is_gone(self) -> None:
        # Media expiry is the bucket lifecycle rule's job now; a stale key
        # is ignored (extra="ignore") rather than failing startup.
        self.assertNotIn("gcs_media_cleanup_delay_seconds", Settings.model_fields)
        settings = Settings(_env_file=None, gcs_media_cleanup_delay_seconds=0)
        self.assertFalse(hasattr(settings, "gcs_media_cleanup_delay_seconds"))

    def test_settings_default_task_settings(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.tasks_mode, "auto")
        self.assertEqual(settings.cloud_tasks_queue, "linebot-events")
        self.assertEqual(settings.cloud_tasks_location, "us-west1")
        self.assertEqual(settings.public_base_url, "")
        self.assertEqual(settings.internal_tasks_secret, "")
        self.assertEqual(settings.task_dispatch_deadline_seconds, 300)
        self.assertEqual(settings.task_handler_budget_seconds, 280)
        self.assertEqual(settings.reply_window_seconds, 90)

    def test_settings_normalise_tasks_mode_and_reject_unknown(self) -> None:
        self.assertEqual(Settings(_env_file=None, tasks_mode=" Cloud ").tasks_mode, "cloud")
        with self.assertRaises(ValueError):
            Settings(_env_file=None, tasks_mode="celery")

    def test_settings_bound_dispatch_deadline_to_cloud_tasks_limits(self) -> None:
        for value in (15, 1800):
            with self.subTest(value=value):
                settings = Settings(_env_file=None, task_dispatch_deadline_seconds=value)
                self.assertEqual(settings.task_dispatch_deadline_seconds, value)
        for value in (14, 1801):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(_env_file=None, task_dispatch_deadline_seconds=value)

    def test_settings_reject_non_positive_task_budgets(self) -> None:
        for field_name in (
            "task_handler_budget_seconds",
            "reply_window_seconds",
            "turn_deadline_seconds",
            "min_tier_seconds",
            "openrouter_free_daily_limit",
        ):
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                Settings(_env_file=None, **{field_name: 0})

    def test_settings_default_model_tier_limits(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.turn_deadline_seconds, 240)
        self.assertEqual(settings.min_tier_seconds, 5)
        self.assertEqual(settings.openrouter_free_daily_limit, 50)
        self.assertEqual(settings.force_tier, "")
        # The whole-chain thinking timeout is gone; tiers have their own.
        self.assertNotIn("thinking_timeout_seconds", Settings.model_fields)
        self.assertLess(settings.turn_deadline_seconds, settings.task_handler_budget_seconds)

    def test_settings_validate_force_tier(self) -> None:
        for value in ("nvidia", " openrouter ", "nvidia:nvidia/nemotron-3-ultra-550b-a55b", "colab"):
            with self.subTest(value=value):
                self.assertEqual(Settings(_env_file=None, force_tier=value).force_tier, value.strip())
        for value in ("nvida", "gemini:x", ":model"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(_env_file=None, force_tier=value)

    def test_settings_validate_public_base_url(self) -> None:
        settings = Settings(_env_file=None, public_base_url=" https://bot.example.com/ ")
        self.assertEqual(settings.public_base_url, "https://bot.example.com")
        # http is refused too: tasks carry X-Tasks-Token to this URL.
        for value in ("bot.example.com", "http://bot.example.com"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(_env_file=None, public_base_url=value)

    def test_settings_default_memory_recent_limit_is_six(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.memory_recent_message_limit, 6)

    def test_settings_ignore_legacy_orchestrator_keys(self) -> None:
        # The fields are gone; pydantic's extra="ignore" drops the old keys
        # (legacy_config_warnings reports them at startup instead).
        settings = Settings(
            _env_file=None,
            orchestrator_model="nvidia/nemotron-3-super-120b-a12b",
            orchestrator_fallback_model="   ",
        )
        self.assertFalse(hasattr(settings, "orchestrator_model"))
        self.assertFalse(hasattr(settings, "orchestrator_fallback_model"))

    def test_per_model_settings_are_gone(self) -> None:
        # The model role table (MODEL_ROLES) replaced them; a stale key is
        # ignored (extra="ignore") and reported by legacy_config_warnings.
        for field_name in _REMOVED_MODEL_FIELDS:
            with self.subTest(field=field_name):
                self.assertNotIn(field_name, Settings.model_fields)
                self.assertIn(field_name.upper(), REMOVED_ENV_KEYS)
        settings = Settings(_env_file=None, nvidia_model="custom/model", image_gen_steps=4)
        self.assertFalse(hasattr(settings, "nvidia_model"))

    def test_settings_default_role_table_switches(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.model_roles, "")
        self.assertEqual(settings.model_path_preference, "colab")
        self.assertEqual((settings.colab_chat_think, settings.colab_deep_think), ("low", "medium"))
        self.assertEqual(settings.nvidia_thinking_budget, 4096)

    def test_settings_validate_model_path_preference(self) -> None:
        self.assertEqual(Settings(_env_file=None, model_path_preference=" API ").model_path_preference, "api")
        for value in ("nvidia", "colab,api", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(_env_file=None, model_path_preference=value)

    def test_settings_validate_colab_think_levels(self) -> None:
        for value in ("false", "LOW", "medium", "xhigh"):
            with self.subTest(value=value):
                settings = Settings(_env_file=None, colab_chat_think=value, colab_deep_think=value)
                self.assertEqual(settings.colab_chat_think, value.lower())
        # Ollama silently reads true / high / max as "medium": refuse them.
        for field_name in ("colab_chat_think", "colab_deep_think"):
            for value in ("true", "high", "max"):
                with self.subTest(field=field_name, value=value), self.assertRaises(ValueError):
                    Settings(_env_file=None, **{field_name: value})

    def test_settings_bound_the_thinking_budget_to_nim(self) -> None:
        for value in (0, 32_769):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(_env_file=None, nvidia_thinking_budget=value)

    def test_settings_default_cost_switches(self) -> None:
        settings = Settings(_env_file=None)
        self.assertFalse(settings.openrouter_allow_paid)
        # NVIDIA NIM rejects temperature 0.
        self.assertGreater(settings.orchestrator_temperature, 0)
        self.assertEqual(settings.orchestrator_temperature, 0.1)

    def test_settings_reject_non_positive_embedding_dimensions(self) -> None:
        for value in (0, -1024):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(_env_file=None, embedding_dimensions=value)

    def test_push_budget_settings_are_gone_and_reported_as_removed(self) -> None:
        for key in (
            "LINE_PUSH_MONTHLY_LIMIT",
            "LINE_PUSH_RESERVE",
            "LINE_PUSH_FALLBACK_ENABLED",
            "LINE_REPLY_FALLBACK_PUSH_ENABLED",
        ):
            with self.subTest(key=key):
                self.assertNotIn(key.lower(), Settings.model_fields)
                [warning] = legacy_config_warnings(environ={key: "0"}, env_file=None)
                self.assertIn(f"{key} in the environment is no longer read", warning)

    def test_one_reply_window_for_every_request(self) -> None:
        # 2026-10-01: every request gets about 90 s from the LINE event,
        # whoever answers; the Colab-only window is gone.
        self.assertEqual(Settings(_env_file=None).reply_window_seconds, 90)
        self.assertNotIn("colab_reply_window_seconds", Settings.model_fields)
        [warning] = legacy_config_warnings(environ={"COLAB_REPLY_WINDOW_SECONDS": "90"}, env_file=None)
        self.assertIn("COLAB_REPLY_WINDOW_SECONDS in the environment is no longer read", warning)

    def test_the_reply_window_stays_below_the_turn_deadline(self) -> None:
        # The turn deadline stays below the task handler budget, so a plain
        # answer working to the window is never cut off by the handler.
        for window in (240, 300):
            with self.subTest(window=window), self.assertRaises(ValueError):
                Settings(_env_file=None, reply_window_seconds=window, turn_deadline_seconds=240)
        self.assertEqual(Settings(_env_file=None, reply_window_seconds=239).reply_window_seconds, 239)

    def test_settings_reject_non_positive_expiry(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, gcs_signed_url_expiry_hours=0)

    def test_settings_default_memory_summary_timeout_is_180_seconds(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.memory_summary_timeout_seconds, 180)

    def test_settings_default_embedding_endpoint_and_width(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.embedding_dimensions, 1024)
        self.assertEqual(
            settings.nvidia_embedding_endpoint,
            "https://integrate.api.nvidia.com/v1/embeddings",
        )


class LegacyConfigWarningTests(unittest.TestCase):
    def test_shared_retired_and_legacy_lists(self) -> None:
        self.assertEqual(
            RETIRED_MODEL_IDS,
            {
                "qwen/qwen3.5-397b-a17b",
                "qwen/qwen3.5-122b-a10b",
                "nvidia/nv-embedqa-e5-v5",
            },
        )
        self.assertEqual(
            LEGACY_ENV_KEYS, {"ORCHESTRATOR_MODEL", "ORCHESTRATOR_FALLBACK_MODEL"}
        )

    def test_clean_config_has_no_warnings(self) -> None:
        self.assertEqual(
            legacy_config_warnings(environ={}, env_file=None),
            [],
        )

    def test_legacy_keys_in_environment_are_reported_by_name_only(self) -> None:
        warnings = legacy_config_warnings(
            environ={
                "ORCHESTRATOR_MODEL": "secret-looking-value",
                "orchestrator_fallback_model": "another-value",
                "PATH": "/usr/bin",
            },
            env_file=None,
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("ORCHESTRATOR_FALLBACK_MODEL, ORCHESTRATOR_MODEL", warnings[0])
        self.assertIn("the environment", warnings[0])
        self.assertNotIn("secret-looking-value", warnings[0])
        self.assertNotIn("another-value", warnings[0])

    def test_process_environment_is_checked_by_default(self) -> None:
        with patch.dict("os.environ", {"ORCHESTRATOR_MODEL": "x"}):
            warnings = legacy_config_warnings(env_file=None)

        self.assertEqual(len(warnings), 1)
        self.assertIn("ORCHESTRATOR_MODEL in the environment", warnings[0])

    def test_legacy_keys_in_env_file_are_reported_by_name_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "LINE_CHANNEL_SECRET=do-not-print\n"
                "ORCHESTRATOR_MODEL=nvidia/nemotron-3-super-120b-a12b:free\n",
                encoding="utf-8",
            )

            warnings = legacy_config_warnings(environ={}, env_file=env_path)

        self.assertEqual(len(warnings), 1)
        self.assertIn("ORCHESTRATOR_MODEL", warnings[0])
        self.assertIn(str(env_path), warnings[0])
        self.assertNotIn("do-not-print", warnings[0])
        self.assertNotIn("nemotron", warnings[0])

    def test_missing_env_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            warnings = legacy_config_warnings(
                environ={},
                env_file=Path(tmp) / "absent.env",
            )
        self.assertEqual(warnings, [])

    def test_removed_model_keys_are_reported_by_name_only(self) -> None:
        # Even a retired id in a removed key is harmless now: nothing reads
        # it. The warning names the key and points at the role table.
        warnings = legacy_config_warnings(
            environ={"NVIDIA_MODEL": "qwen/qwen3.5-397b-a17b", "image_gen_steps": "50"},
            env_file=None,
        )

        self.assertEqual(len(warnings), 2)
        self.assertTrue(any(w.startswith("NVIDIA_MODEL in the environment is no longer read") for w in warnings))
        self.assertTrue(any(w.startswith("IMAGE_GEN_STEPS in the environment") for w in warnings))
        for warning in warnings:
            self.assertIn("MODEL_ROLES", warning)
            self.assertNotIn("qwen3.5", warning)

    def test_removed_media_cleanup_key_is_reported_by_name_only(self) -> None:
        self.assertIn("GCS_MEDIA_CLEANUP_DELAY_SECONDS", REMOVED_ENV_KEYS)
        # Informational only: the deploy guard fails on LEGACY_ENV_KEYS.
        self.assertNotIn("GCS_MEDIA_CLEANUP_DELAY_SECONDS", LEGACY_ENV_KEYS)

        warnings = legacy_config_warnings(
            environ={"gcs_media_cleanup_delay_seconds": "172800"},
            env_file=None,
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("GCS_MEDIA_CLEANUP_DELAY_SECONDS in the environment", warnings[0])
        self.assertIn("bucket lifecycle", warnings[0])
        self.assertNotIn("172800", warnings[0])

    def test_removed_thinking_keys_are_reported_by_name_only(self) -> None:
        for key in ("THINKING_TIMEOUT_SECONDS", "NVIDIA_THINKING_MODEL"):
            with self.subTest(key=key):
                self.assertIn(key, REMOVED_ENV_KEYS)
                self.assertNotIn(key, LEGACY_ENV_KEYS)
                warnings = legacy_config_warnings(
                    environ={key.lower(): "example/old-thinking-model"},
                    env_file=None,
                )
                self.assertEqual(len(warnings), 1)
                self.assertIn(f"{key} in the environment is no longer read", warnings[0])
                self.assertNotIn("old-thinking-model", warnings[0])

    def test_require_reasoning_models_is_reported(self) -> None:
        warnings = legacy_config_warnings(
            environ={"REQUIRE_REASONING_MODELS": "true"},
            env_file=None,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("REQUIRE_REASONING_MODELS in the environment is no longer read", warnings[0])


_DM = "U" + "1" * 32
_GROUP = "C" + "2" * 32


def _line_api(*, push_status: int = 200):
    """A LINE API on httpx.MockTransport: pushes answer ``push_status``;
    every other call (a reply with an expired token, a quota or member
    count read) is recorded and refused."""
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v2/bot/message/push":
            return httpx.Response(push_status, json={"sentMessages": []})
        return httpx.Response(400, json={"message": "Invalid reply token"})

    return httpx.MockTransport(handler), calls


def _pushes(calls: list[tuple[str, str]]) -> int:
    return calls.count(("POST", "/v2/bot/message/push"))


class LineServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def _service(self, transport):
        settings = Settings(_env_file=None, line_channel_access_token="token")
        service = line_service_module.LineService(settings, transport=transport)
        self.addAsyncCleanup(service.close)
        return service


class LinePushTests(LineServiceTestCase):
    """Reply first (free); push only when the reply certainly failed. No
    push is counted, capped or refused: LINE's own quota is the only limit."""

    async def test_a_failed_reply_falls_back_to_a_push_in_every_chat_without_quota_reads(self) -> None:
        transport, calls = _line_api()
        service = self._service(transport)

        for to in (_DM, _GROUP):
            self.assertTrue(await service.send_text("expired", to, "答案"))

        reply_then_push = [("POST", "/v2/bot/message/reply"), ("POST", "/v2/bot/message/push")]
        self.assertEqual(calls, reply_then_push * 2)

    async def test_pushes_are_never_counted_or_capped(self) -> None:
        transport, calls = _line_api()
        service = self._service(transport)

        for _ in range(30):
            self.assertTrue(await service.push_text(_GROUP, "提醒"))
            self.assertTrue(await service.send_text("", _DM, "答案"))

        self.assertEqual(calls, [("POST", "/v2/bot/message/push")] * 60)

    async def test_a_refused_push_is_reported_unsent(self) -> None:
        transport, calls = _line_api(push_status=429)
        service = self._service(transport)

        self.assertFalse(await service.push_text(_DM, "提醒"))
        self.assertFalse(await service.send_text("expired", _DM, "答案"))
        self.assertEqual(_pushes(calls), 2)

    async def test_send_text_caches_successful_bot_reply(self) -> None:
        settings = Settings(_env_file=None, line_channel_access_token="token")

        with patch.object(line_service_module, "get_settings", return_value=settings):
            service = line_service_module.LineService()

        class FakeResponse:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"sentMessages": [{"id": "BOT_TEXT_1"}]}

        cache_service = SimpleNamespace(cache_bot_message=Mock())
        with patch.object(service.client, "post", AsyncMock(return_value=FakeResponse())), patch(
            "src.services.message_cache_service.get_message_cache_service",
            return_value=cache_service,
        ):
            try:
                self.assertTrue(await service.send_text("reply-token", "U1234567890", "bot hello"))
            finally:
                await service.close()

        cache_service.cache_bot_message.assert_called_once_with(
            "BOT_TEXT_1",
            "text",
            text="bot hello",
            image_url="",
        )

    async def test_send_image_caches_successful_bot_reply_url(self) -> None:
        settings = Settings(_env_file=None, line_channel_access_token="token")

        with patch.object(line_service_module, "get_settings", return_value=settings):
            service = line_service_module.LineService()

        class FakeResponse:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"sentMessages": [{"id": "BOT_IMG_1"}]}

        cache_service = SimpleNamespace(cache_bot_message=Mock())
        with patch.object(service.client, "post", AsyncMock(return_value=FakeResponse())), patch(
            "src.services.message_cache_service.get_message_cache_service",
            return_value=cache_service,
        ):
            try:
                self.assertTrue(
                    await service.send_image(
                        "reply-token",
                        "U1234567890",
                        "https://example.com/generated.png",
                    )
                )
            finally:
                await service.close()

        cache_service.cache_bot_message.assert_called_once_with(
            "BOT_IMG_1",
            "image",
            text="",
            image_url="https://example.com/generated.png",
        )


def _line_answering(*, reply, push=200):
    """A LINE API whose reply and push each answer a status or raise an
    httpx error class. A raised error comes after LINE took the request
    unless it is one raised before sending, as the real ones are."""
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        outcome = {"/v2/bot/message/reply": reply, "/v2/bot/message/push": push}.get(path)
        if outcome is None:
            return httpx.Response(500)
        if isinstance(outcome, int):
            calls.append(("POST", path))
            return httpx.Response(outcome, json={"sentMessages": []})
        if outcome not in (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            calls.append(("POST", path))  # LINE took it; the answer is lost
        raise outcome("simulated", request=request)

    return httpx.MockTransport(handler), calls


class LineServiceUnclearOutcomeTests(LineServiceTestCase):
    """A send LINE may have delivered is never followed by a second one."""

    async def test_a_reply_that_may_have_gone_out_is_not_pushed_again(self) -> None:
        for reply in (httpx.ReadTimeout, httpx.RemoteProtocolError, 500, 503):
            with self.subTest(reply=reply):
                transport, calls = _line_answering(reply=reply)
                service = self._service(transport)

                self.assertTrue(await service.send_text("token-1", "U1234567890", "答案"))
                self.assertTrue(
                    await service.send_messages(
                        "token-2", "U1234567890", [{"type": "text", "text": "說明"}]
                    )
                )
                self.assertEqual(_pushes(calls), 0)
                self.assertEqual(calls.count(("POST", "/v2/bot/message/reply")), 2)

    async def test_a_reply_that_certainly_failed_falls_back_to_a_push(self) -> None:
        for reply in (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, 400):
            with self.subTest(reply=reply):
                transport, calls = _line_answering(reply=reply)
                service = self._service(transport)

                self.assertTrue(await service.send_text("token-1", "U1234567890", "答案"))
                self.assertEqual(_pushes(calls), 1)

    async def test_a_push_that_may_have_gone_out_counts_as_sent(self) -> None:
        for push in (httpx.ReadTimeout, 500):
            with self.subTest(push=push):
                transport, calls = _line_answering(reply=400, push=push)
                service = self._service(transport)

                # Counted as sent: no text fallback after it.
                self.assertTrue(await service.send_text("expired", "U1234567890", "答案"))
                self.assertEqual(_pushes(calls), 1)

    async def test_a_push_that_never_left_is_reported_unsent(self) -> None:
        transport, calls = _line_answering(reply=400, push=httpx.ConnectError)
        service = self._service(transport)

        self.assertFalse(await service.send_text("expired", "U1234567890", "答案"))
        self.assertEqual(calls, [("POST", "/v2/bot/message/reply")])


class DeliveryTraceTests(LineServiceTestCase):
    """The summary line says how the answer went out: delivery=reply|push."""

    async def _summary(self, *sends) -> str:
        trace = request_trace.RequestTrace(request_id="r1")
        token = request_trace.activate(trace)
        try:
            for send in sends:
                self.assertTrue(await send)
        finally:
            request_trace.deactivate(token)
        return request_trace.summary_line(trace, now=trace.started)

    async def test_a_reply_and_a_push_fallback_are_told_apart(self) -> None:
        for reply, via in ((200, "reply"), (400, "push")):
            with self.subTest(via=via):
                transport, _ = _line_answering(reply=reply)
                service = self._service(transport)

                line = await self._summary(service.send_text("token-1", _DM, "答案"))

                self.assertIn(f" delivery={via} ", line)

    async def test_a_push_after_a_reply_in_the_same_event_shows_push(self) -> None:
        transport, _ = _line_answering(reply=200)
        service = self._service(transport)

        line = await self._summary(
            service.send_text("token-1", _DM, "圖片"), service.send_text("", _DM, "說明")
        )

        self.assertIn(" delivery=push ", line)

    async def test_nothing_sent_means_no_delivery_field(self) -> None:
        self.assertNotIn("delivery=", await self._summary())


class StorageStatsTests(unittest.TestCase):
    def test_storage_usage_stats_track_uploads_and_deletes(self) -> None:
        settings = Settings(_env_file=None)

        with patch.object(storage_service_module, "get_settings", return_value=settings):
            service = storage_service_module.StorageService()

        service.record_upload(512 * 1024)
        service.record_delete()

        stats = service.get_usage_stats()
        self.assertEqual(stats["uploads"], 1)
        self.assertEqual(stats["upload_bytes"], 512 * 1024)
        self.assertEqual(stats["deleted_objects"], 1)
        # No in-process cleanup timer any more: the bucket lifecycle expires media.
        self.assertEqual(stats["media_expiry"], "bucket lifecycle")
        self.assertNotIn("cleanup_tasks", stats)
        self.assertEqual(stats["scope"], "per-instance")

    def test_storage_service_resolves_metadata_service_account_email(self) -> None:
        settings = Settings(
            _env_file=None,
            gcs_bucket_name="gcp-banana",
        )
        credentials = SimpleNamespace(service_account_email="default")
        fake_client = SimpleNamespace(
            bucket=Mock(return_value=Mock()),
            _credentials=credentials,
        )
        metadata_response = MagicMock()
        metadata_response.getcode.return_value = 200
        metadata_response.read.return_value = b"runtime@example.com\n"
        metadata_request = MagicMock()
        metadata_request.__enter__.return_value = metadata_response
        metadata_request.__exit__.return_value = False

        with patch.object(
            storage_service_module,
            "get_settings",
            return_value=settings,
        ), patch(
            "google.cloud.storage.Client",
            return_value=fake_client,
        ), patch(
            "src.utils.gcp_metadata.urlopen",
            return_value=metadata_request,
        ) as urlopen:
            service = storage_service_module.StorageService()

        self.assertEqual(service._service_account_email, "runtime@example.com")
        metadata_url = urlopen.call_args.args[0]
        self.assertEqual(
            metadata_url.full_url,
            "http://metadata.google.internal/computeMetadata/v1/"
            "instance/service-accounts/default/email",
        )
        self.assertEqual(metadata_url.get_header("Metadata-flavor"), "Google")

    def test_storage_service_uses_iam_signer_credentials_for_unsigned_credentials(self) -> None:
        settings = Settings(
            _env_file=None,
            gcs_bucket_name="gcp-banana",
        )

        class FakeUnsignedCredentials:
            def __init__(self) -> None:
                self.service_account_email = "bot@example.com"
                self.token = None
                self.valid = False
                self.refresh_count = 0

            def refresh(self, _request) -> None:
                self.refresh_count += 1
                self.token = "ya29.test-token"
                self.valid = True

        credentials = FakeUnsignedCredentials()
        iam_credentials = FakeUnsignedCredentials()
        fake_client = SimpleNamespace(
            bucket=Mock(return_value=Mock()),
            _credentials=credentials,
        )
        fake_signer = object()
        fake_signing_credentials = object()

        with patch.object(
            storage_service_module, "get_settings", return_value=settings
        ), patch("google.cloud.storage.Client", return_value=fake_client), patch(
            "google.auth.default", return_value=(iam_credentials, "test-project")
        ), patch.object(
            storage_service_module.iam,
            "Signer",
            return_value=fake_signer,
        ) as signer_ctor, patch.object(
            storage_service_module.google_service_account,
            "Credentials",
            return_value=fake_signing_credentials,
        ) as creds_ctor:
            service = storage_service_module.StorageService()
            kwargs = service._signed_url_kwargs()

        self.assertEqual(kwargs["version"], "v4")
        self.assertIs(kwargs["credentials"], fake_signing_credentials)
        signer_ctor.assert_called_once()
        # IAM signer should use cloud-platform scoped credentials, not storage credentials
        signer_call_args = signer_ctor.call_args
        self.assertIs(signer_call_args[0][1], iam_credentials)
        creds_ctor.assert_called_once_with(
            signer=fake_signer,
            service_account_email="bot@example.com",
            token_uri="https://oauth2.googleapis.com/token",
        )
        self.assertEqual(credentials.refresh_count, 0)
