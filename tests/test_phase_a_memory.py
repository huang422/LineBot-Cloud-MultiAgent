"""Phase A regression tests: backend abstraction, displayName, user profile.

These tests are intentionally backend-agnostic where possible: the
``InMemoryBackend`` round-trip exercises the same public surface that
``FirestoreBackend`` is expected to satisfy. The Firestore backend is
covered with a mocked async wrapper so we can validate the document
shape without requiring a live GCP project.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings
from src.services.line_service import LineService
from src.services.memory_backends import (
    InMemoryBackend,
    MemoryBackendError,
    build_backend,
)
from src.services.memory_service import MemoryService


def _make_settings() -> Settings:
    return Settings(_env_file=None)


def _make_memory_service(backend: InMemoryBackend) -> MemoryService:
    return MemoryService(_make_settings(), backend=backend)


class InMemoryBackendTests(unittest.TestCase):
    def test_chat_document_round_trip(self) -> None:
        backend = InMemoryBackend()
        doc = {"summary_text": "hi", "recent_messages": []}
        asyncio.run(
            backend.save_chat("user::U1", doc)
        )
        loaded = asyncio.run(
            backend.load_chat("user::U1")
        )
        self.assertEqual(loaded["summary_text"], "hi")
        # Mutating the returned dict must not corrupt the store
        loaded["summary_text"] = "tampered"
        again = asyncio.run(
            backend.load_chat("user::U1")
        )
        self.assertEqual(again["summary_text"], "hi")

    def test_user_profile_round_trip(self) -> None:
        backend = InMemoryBackend()
        asyncio.run(
            backend.save_user_profile("U1", {"display_name": "Alice"})
        )
        loaded = asyncio.run(backend.load_user_profile("U1"))
        self.assertEqual(loaded["display_name"], "Alice")
        miss = asyncio.run(backend.load_user_profile("U2"))
        self.assertIsNone(miss)


class BuildBackendFallbackTests(unittest.TestCase):
    def test_disabled_settings_yield_in_memory(self) -> None:
        settings = SimpleNamespace(
            firestore_enabled=False,
            firestore_project_id=None,
            firestore_database="(default)",
            firestore_collection_prefix="linebot",
            firestore_location="us-west1",
        )
        backend = build_backend(settings)
        self.assertIsInstance(backend, InMemoryBackend)

    def test_firestore_initialization_failure_falls_back_to_in_memory(self) -> None:
        settings = SimpleNamespace(
            firestore_enabled=True,
            firestore_project_id=None,
            firestore_database="(default)",
            firestore_collection_prefix="linebot",
            firestore_location="us-west1",
        )
        with patch(
            "src.services.memory_backends.FirestoreBackend",
            side_effect=MemoryBackendError("no credentials"),
        ):
            backend = build_backend(settings)
        self.assertIsInstance(backend, InMemoryBackend)


class TouchUserProfileTests(unittest.TestCase):
    def test_touch_creates_then_preserves_existing_display_name(self) -> None:
        backend = InMemoryBackend()
        service = _make_memory_service(backend)

        asyncio.run(
            service.touch_user_profile(
                user_id="U1",
                display_name="Alice",
                source_type="user",
                chat_id="U1",
            )
        )
        profile = asyncio.run(service.load_user_profile("U1"))
        self.assertEqual(profile["display_name"], "Alice")
        self.assertEqual(profile["last_seen_source_type"], "user")

        # Subsequent call with empty display_name must NOT overwrite
        asyncio.run(
            service.touch_user_profile(
                user_id="U1",
                display_name="",
                source_type="group",
                chat_id="G42",
            )
        )
        profile = asyncio.run(service.load_user_profile("U1"))
        self.assertEqual(profile["display_name"], "Alice")
        self.assertEqual(profile["last_seen_source_type"], "group")
        self.assertEqual(profile["last_seen_chat_id"], "G42")

    def test_touch_with_empty_user_id_is_noop(self) -> None:
        backend = InMemoryBackend()
        service = _make_memory_service(backend)
        asyncio.run(
            service.touch_user_profile(user_id="", display_name="X")
        )
        self.assertEqual(backend.user_profiles, {})

    def test_touch_swallows_backend_error(self) -> None:
        backend = InMemoryBackend()
        service = _make_memory_service(backend)
        with patch.object(
            backend,
            "save_user_profile",
            side_effect=MemoryBackendError("boom"),
        ):
            # Must not raise — best-effort upsert
            asyncio.run(
                service.touch_user_profile(user_id="U1", display_name="A")
            )


class FetchDisplayNameTests(unittest.TestCase):
    def _service(self) -> LineService:
        with patch("src.services.line_service.get_settings") as mock_settings:
            settings = _make_settings()
            settings.line_channel_access_token = "test-token"
            mock_settings.return_value = settings
            return LineService()

    def _mock_response(self, status: int, payload: dict | None = None):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload or {}
        return resp

    def test_dm_uses_profile_endpoint(self) -> None:
        service = self._service()
        service.client = MagicMock()
        service.client.get = AsyncMock(
            return_value=self._mock_response(200, {"displayName": "Bob"})
        )
        name = asyncio.run(
            service.fetch_display_name(
                source_type="user", chat_id="U1", user_id="U1"
            )
        )
        self.assertEqual(name, "Bob")
        called_url = service.client.get.call_args[0][0]
        self.assertIn("/bot/profile/U1", called_url)
        self.assertNotIn("/group/", called_url)

    def test_group_uses_member_endpoint(self) -> None:
        service = self._service()
        service.client = MagicMock()
        service.client.get = AsyncMock(
            return_value=self._mock_response(200, {"displayName": "Carol"})
        )
        name = asyncio.run(
            service.fetch_display_name(
                source_type="group", chat_id="G42", user_id="U9"
            )
        )
        self.assertEqual(name, "Carol")
        called_url = service.client.get.call_args[0][0]
        self.assertIn("/bot/group/G42/member/U9", called_url)

    def test_404_returns_empty_string(self) -> None:
        service = self._service()
        service.client = MagicMock()
        service.client.get = AsyncMock(return_value=self._mock_response(404))
        name = asyncio.run(
            service.fetch_display_name(
                source_type="user", chat_id="U1", user_id="U1"
            )
        )
        self.assertEqual(name, "")

    def test_cached_response_avoids_second_http_call(self) -> None:
        service = self._service()
        service.client = MagicMock()
        service.client.get = AsyncMock(
            return_value=self._mock_response(200, {"displayName": "Dan"})
        )
        for _ in range(3):
            asyncio.run(
                service.fetch_display_name(
                    source_type="user", chat_id="U1", user_id="U1"
                )
            )
        self.assertEqual(service.client.get.await_count, 1)

    def test_missing_token_returns_empty(self) -> None:
        service = self._service()
        service.channel_access_token = ""
        name = asyncio.run(
            service.fetch_display_name(
                source_type="user", chat_id="U1", user_id="U1"
            )
        )
        self.assertEqual(name, "")


class MemoryServiceBackendIntegrationTests(unittest.TestCase):
    def test_get_stats_reports_backend_name(self) -> None:
        backend = InMemoryBackend()
        service = _make_memory_service(backend)
        stats = service.get_stats()
        self.assertEqual(stats["backend"], "memory")
        self.assertFalse(stats.get("persistent", True))

    def test_legacy_memory_store_property_proxies_in_memory(self) -> None:
        backend = InMemoryBackend()
        service = _make_memory_service(backend)
        # Legacy tests poke `_memory_store` directly; ensure it round-trips
        service._memory_store["user::U1"] = {"summary_text": "legacy"}
        self.assertEqual(
            backend.store["user::U1"]["summary_text"], "legacy"
        )


if __name__ == "__main__":
    unittest.main()
