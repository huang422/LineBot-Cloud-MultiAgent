from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.config import Settings
from src.services import line_service as line_service_module
from src.services import storage_service as storage_service_module


class SettingsValidationTests(unittest.TestCase):
    def test_settings_allow_unlisted_model_ids(self) -> None:
        settings = Settings(
            _env_file=None,
            orchestrator_model="custom/orchestrator",
            orchestrator_fallback_model="custom/fallback",
            agent_fallback_model="custom/agent",
            vision_fallback_model="custom/vision",
            nvidia_model="custom/nvidia",
            image_gen_primary_model="custom/image-model",
        )
        self.assertEqual(settings.orchestrator_model, "custom/orchestrator")
        self.assertEqual(settings.image_gen_primary_model, "custom/image-model")

    def test_settings_reject_blank_model_id(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, orchestrator_model="   ")

    def test_settings_reject_negative_cost_limit(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, line_push_monthly_limit=-1)

    def test_settings_reject_non_positive_expiry(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, gcs_signed_url_expiry_hours=0)

    def test_settings_reject_non_positive_cleanup_delay(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, gcs_media_cleanup_delay_seconds=0)

    def test_settings_reject_invalid_reasoning_effort(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, openrouter_reasoning_effort="ultra")


class LineServiceBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_text_blocks_push_when_disabled(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_access_token="token",
            line_push_fallback_enabled=False,
            line_push_monthly_limit=0,
        )

        with patch.object(line_service_module, "get_settings", return_value=settings):
            service = line_service_module.LineService()

        fake_post = AsyncMock(side_effect=AssertionError("push should stay blocked"))
        with patch.object(service.client, "post", fake_post):
            try:
                self.assertFalse(
                    await service.send_text("", "U1234567890", "hello")
                )
                self.assertFalse(fake_post.called)
                stats = service.get_push_stats()
                self.assertEqual(stats["reply_fallback_used"], 0)
                self.assertEqual(stats["direct_push_used"], 0)
            finally:
                await service.close()

    async def test_reply_fallback_push_works_when_enabled(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_access_token="token",
            line_push_fallback_enabled=True,
            line_push_monthly_limit=0,
        )

        with patch.object(line_service_module, "get_settings", return_value=settings):
            service = line_service_module.LineService()

        class FakeResponse:
            status_code = 200
            text = "ok"

        fake_post = AsyncMock(return_value=FakeResponse())
        with patch.object(service.client, "post", fake_post):
            try:
                self.assertTrue(
                    await service.send_text("", "U1234567890", "fallback")
                )
                self.assertEqual(fake_post.await_count, 1)
                stats = service.get_push_stats()
                self.assertEqual(stats["reply_fallback_used"], 1)
                self.assertEqual(stats["direct_push_used"], 0)
                self.assertIsNone(stats["direct_push_remaining"])
                self.assertTrue(stats["direct_push_unlimited"])
            finally:
                await service.close()

    async def test_direct_push_budget_stops_after_limit(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_access_token="token",
            line_push_fallback_enabled=True,
            line_push_monthly_limit=1,
        )

        with patch.object(line_service_module, "get_settings", return_value=settings):
            service = line_service_module.LineService()

        class FakeResponse:
            status_code = 200
            text = "ok"

        fake_post = AsyncMock(return_value=FakeResponse())
        with patch.object(service.client, "post", fake_post):
            try:
                self.assertTrue(await service.push_text("U1234567890", "first"))
                self.assertFalse(await service.push_text("U1234567890", "second"))
                self.assertEqual(fake_post.await_count, 1)
                stats = service.get_push_stats()
                self.assertEqual(stats["reply_fallback_used"], 0)
                self.assertEqual(stats["direct_push_used"], 1)
                self.assertEqual(stats["direct_push_remaining"], 0)
            finally:
                await service.close()

    async def test_direct_push_is_unlimited_when_limit_is_zero(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_access_token="token",
            line_push_fallback_enabled=True,
            line_push_monthly_limit=0,
        )

        with patch.object(line_service_module, "get_settings", return_value=settings):
            service = line_service_module.LineService()

        class FakeResponse:
            status_code = 200
            text = "ok"

        fake_post = AsyncMock(return_value=FakeResponse())
        with patch.object(service.client, "post", fake_post):
            try:
                self.assertTrue(await service.push_text("U1234567890", "first"))
                self.assertTrue(await service.push_text("U1234567890", "second"))
                self.assertEqual(fake_post.await_count, 2)
                stats = service.get_push_stats()
                self.assertEqual(stats["direct_push_used"], 2)
                self.assertIsNone(stats["direct_push_limit"])
                self.assertIsNone(stats["direct_push_remaining"])
                self.assertTrue(stats["direct_push_unlimited"])
            finally:
                await service.close()

    async def test_send_text_caches_successful_bot_reply(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_access_token="token",
            line_push_fallback_enabled=True,
            line_push_monthly_limit=0,
        )

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
        settings = Settings(
            _env_file=None,
            line_channel_access_token="token",
            line_push_fallback_enabled=True,
            line_push_monthly_limit=0,
        )

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


class StorageStatsTests(unittest.TestCase):
    def test_storage_usage_stats_track_uploads_and_deletes(self) -> None:
        settings = Settings(_env_file=None, gcs_media_cleanup_delay_seconds=90)

        with patch.object(storage_service_module, "get_settings", return_value=settings):
            service = storage_service_module.StorageService()

        service.record_upload(512 * 1024)
        service.record_delete()

        stats = service.get_usage_stats()
        self.assertEqual(stats["uploads"], 1)
        self.assertEqual(stats["upload_bytes"], 512 * 1024)
        self.assertEqual(stats["deleted_objects"], 1)
        self.assertEqual(stats["cleanup_delay_seconds"], 90)
        self.assertEqual(stats["scope"], "per-instance")

    def test_storage_service_uses_iam_signer_credentials_for_unsigned_credentials(self) -> None:
        settings = Settings(
            _env_file=None,
            gcs_bucket_name="gcp-banana",
            gcs_media_cleanup_delay_seconds=90,
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
