"""/health (public, minimal) and /health/detail (X-Cron-Token)."""

from __future__ import annotations

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main as main_module
from src.config import Settings
from src.providers.base import ProviderError
from src.runtime.model_roles import RoleTable
from src.services.model_health import ModelHealth
from src.services.model_lifecycle import LifecycleResult, ModelLifecycle
from src.services.scheduler_service import SchedulerService

_CRON = "cron-secret"


def _settings(**overrides) -> Settings:
    values = {
        "line_channel_secret": "secret",
        "line_channel_access_token": "token",
        "openrouter_api_key": "key",
        "internal_cron_secret": _CRON,
        "log_level": "ERROR",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _table(settings: Settings, *providers: str) -> RoleTable:
    return RoleTable.from_settings(settings, {name: SimpleNamespace(name=name) for name in providers})


class ReadinessTests(unittest.TestCase):
    def test_ready_needs_line_credentials_and_an_api_chat_tier(self) -> None:
        settings = _settings()

        status, ready, warnings = main_module._get_health_status(settings, _table(settings, "openrouter"))

        self.assertEqual((status, ready, warnings), ("healthy", True, []))

    def test_an_nvidia_only_deployment_is_ready(self) -> None:
        settings = _settings(openrouter_api_key="", nvidia_api_key="nv")

        status, ready, warnings = main_module._get_health_status(settings, _table(settings, "nvidia"))

        self.assertEqual((status, ready, warnings), ("healthy", True, []))

    def test_without_line_credentials_it_is_degraded(self) -> None:
        settings = _settings(line_channel_secret="")

        status, ready, warnings = main_module._get_health_status(settings, _table(settings, "openrouter"))

        self.assertEqual((status, ready), ("degraded", False))
        self.assertIn("LINE channel credentials are incomplete", warnings)

    def test_without_a_usable_chat_tier_it_is_degraded_and_says_why(self) -> None:
        settings = _settings(openrouter_api_key="")

        status, ready, warnings = main_module._get_health_status(settings, _table(settings))

        self.assertEqual((status, ready), ("degraded", False))
        self.assertEqual(
            warnings,
            [
                "no usable chat model tier (NVIDIA_API_KEY not set; OPENROUTER_API_KEY not set; "
                "colab tier not enabled)"
            ],
        )

    def test_a_colab_only_chat_role_is_not_ready(self) -> None:
        # A cold Colab must never be what a reply waits on.
        settings = _settings(openrouter_api_key="")

        _, ready, _ = main_module._get_health_status(settings, _table(settings, "colab"))

        self.assertFalse(ready)

    def test_before_startup_it_is_degraded(self) -> None:
        status, ready, warnings = main_module._get_health_status(_settings(), None)

        self.assertEqual((status, ready), ("degraded", False))
        self.assertIn("model role table not built yet", warnings)

    def test_scheduled_messages_need_no_push_switch(self) -> None:
        # Pushes have no switch or budget any more (LINE's quota is the limit).
        settings = _settings(scheduled_messages_enabled=True, scheduled_group_id="C1234567890abcdef")

        status, ready, warnings = main_module._get_health_status(settings, _table(settings, "openrouter"))

        self.assertEqual((status, ready, warnings), ("healthy", True, []))


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _settings()
        self.health = ModelHealth()
        self.table = _table(self.settings, "openrouter", "nvidia")
        self.lifecycle = ModelLifecycle([], self.health)
        self.scheduler = SchedulerService(timezone_name="Asia/Taipei")
        self.scheduler.add_weekly_message("weekly-1", "mon", 21, 0, "C1", "明天早上八點開會，記得帶筆電")

    @contextmanager
    def _client(self, settings: Settings | None = None, colab=None):
        container = SimpleNamespace(
            colab=colab,
            colab_controller=None,
            colab_disabled_reason="",
            role_table=self.table,
            model_health=self.health,
            model_lifecycle=self.lifecycle,
            config_warnings=["NVIDIA_MODEL in .env is no longer read"],
            agents_stats=lambda: {"chat": {"calls": 2}},
            fallback_chain=SimpleNamespace(fallback_count=1),
            task_queue=SimpleNamespace(mode="cloud"),
            search=SimpleNamespace(
                get_status=lambda: {"configured": True, "backends": [{"name": "tavily"}], "degraded": 2}
            ),
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(main_module, "container", container))
            stack.enter_context(
                patch.object(main_module, "get_settings", return_value=settings or self.settings)
            )
            stack.enter_context(
                patch.object(
                    main_module,
                    "get_memory_service",
                    return_value=SimpleNamespace(get_stats=lambda: {"backend": "memory"}),
                )
            )
            stack.enter_context(
                patch("src.services.scheduler_service.peek_scheduler_service", return_value=self.scheduler)
            )
            stack.enter_context(
                patch(
                    "src.services.storage_service.get_storage_service",
                    return_value=SimpleNamespace(get_usage_stats=lambda: {"uploads": 4}),
                )
            )
            # Not entered as a context manager: no lifespan, so the container is ours.
            yield TestClient(main_module.app)

    def test_public_health_is_minimal(self) -> None:
        self.health.record_failure(
            "nvidia", "example/x", ProviderError("example/x", 500, "org acme-org is suspended", "nvidia")
        )
        with self._client() as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy", "ready_for_webhook": True, "version": "dev"})
        self.assertNotIn("acme-org", response.text)

    def test_detail_needs_the_cron_token(self) -> None:
        with self._client() as client:
            for headers in ({}, {"X-Cron-Token": "wrong"}, {"X-Tasks-Token": _CRON}):
                with self.subTest(headers=headers):
                    self.assertEqual(client.get("/health/detail", headers=headers).status_code, 403)

        with self._client(_settings(internal_cron_secret="")) as client:
            response = client.get("/health/detail", headers={"X-Cron-Token": ""})
        self.assertEqual(response.status_code, 403)

    def test_detail_reports_models_budgets_and_services(self) -> None:
        self.health.mark_dead(
            "nvidia",
            "nvidia/nemotron-3-ultra-550b-a55b",
            "HTTP 410, end of life 2027-01-01T00:00:00Z",
            eol="2027-01-01T00:00:00Z",
        )
        self.health.record_failure(
            "openrouter",
            "dots-studio/dots-3-note-preview:free",
            ProviderError("dots-studio/dots-3-note-preview:free", 502, "upstream timeout", "openrouter"),
        )
        self.health.record_request("openrouter", "dots-studio/dots-3-note-preview:free")
        self.lifecycle._results["nvidia:nvidia/nemotron-3-super-120b-a12b"] = LifecycleResult("ok", "listed")

        with self._client() as client:
            response = client.get("/health/detail", headers={"X-Cron-Token": _CRON})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["status"], body["ready_for_webhook"], body["version"]), ("healthy", True, "dev"))
        self.assertEqual(body["config_warnings"], ["NVIDIA_MODEL in .env is no longer read"])
        colab, _super, ultra, dots = body["roles"]["chat"][:4]
        self.assertEqual((colab["enabled"], colab["disabled_reason"]), (False, "colab tier not enabled"))
        self.assertEqual((ultra["health"], ultra["dead"], ultra["eol"]), ("dead", True, "2027-01-01T00:00:00Z"))
        self.assertIn("upstream timeout", dots["last_error"])
        self.assertEqual(body["roles"]["router"][1]["lifecycle"]["state"], "ok")
        self.assertEqual(body["budgets"]["openrouter"]["rpd_used"], 1)
        self.assertEqual(body["lifecycle"]["results"]["nvidia:nvidia/nemotron-3-super-120b-a12b"]["state"], "ok")
        self.assertEqual(body["tasks_mode"], "cloud")
        self.assertEqual(body["memory"], {"backend": "memory"})
        self.assertEqual(body["storage"], {"uploads": 4})
        # No push budget: nothing about pushes to report.
        self.assertNotIn("push", body)
        self.assertEqual(
            body["web_search"], {"configured": True, "backends": [{"name": "tavily"}], "degraded": 2}
        )
        self.assertEqual(body["agents_stats"], {"chat": {"calls": 2}})
        self.assertEqual(body["model_path_preference"], "colab")
        self.assertEqual(body["colab"], {"enabled": False, "reason": "COLAB_ENABLED=false"})

    def test_detail_lists_scheduled_jobs_without_their_text(self) -> None:
        with self._client() as client:
            response = client.get("/health/detail", headers={"X-Cron-Token": _CRON})

        [job] = response.json()["scheduler"]["jobs"]
        self.assertEqual(set(job), {"id", "next_run"})
        self.assertEqual(job["id"], "weekly-1")
        self.assertNotIn("開會", response.text)


if __name__ == "__main__":
    unittest.main()
