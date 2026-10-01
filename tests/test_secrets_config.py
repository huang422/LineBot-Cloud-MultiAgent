"""Settings read .env, then the Secret Manager dotenv blob, then the environment.

On Cloud Run the deploy mounts the linebot-dotenv secret at
/secrets/app/app.env (ADR-0008); locally only .env exists. Later files win,
missing files are skipped, and a real environment variable beats both.
"""

from __future__ import annotations

import inspect
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import envfile
from src import config
from src.config import Settings, legacy_config_warnings
from src.runtime.composition import (
    build_colab_endpoint,
    build_model_health,
    build_providers,
    build_search_backend,
)
from src.services.search.backend import ChainedSearchBackend
from src.services.search.tavily import TavilyBackend
from src.services.task_queue import select_tasks_mode
from tests import conftest


class EnvFileOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.dir = Path(tmpdir.name)
        self.local = self.dir / ".env"
        self.secret = self.dir / "app.env"
        # Nothing from the real environment leaks into these Settings.
        environ = patch.dict(os.environ, {}, clear=True)
        environ.start()
        self.addCleanup(environ.stop)

    def _settings(self, *files: Path) -> Settings:
        return Settings(_env_file=files)

    def test_the_app_reads_dot_env_then_the_mounted_secret(self) -> None:
        self.assertEqual(config.ENV_FILES, (".env", "/secrets/app/app.env"))
        self.assertEqual(config.SECRET_ENV_FILE, "/secrets/app/app.env")
        self.assertEqual(Settings.model_config["env_file"], config.ENV_FILES)

    def test_the_later_file_wins_and_keys_only_in_the_first_survive(self) -> None:
        self.local.write_text("BOT_NAME=Local\nTTS_VOICE=zh-TW-YunJheNeural\n", encoding="utf-8")
        self.secret.write_text("BOT_NAME='Deployed'\nRATE_LIMIT_MAX_REQUESTS='9'\n", encoding="utf-8")

        settings = self._settings(self.local, self.secret)

        self.assertEqual(settings.bot_name, "Deployed")
        self.assertEqual(settings.tts_voice, "zh-TW-YunJheNeural")
        self.assertEqual(settings.rate_limit_max_requests, 9)

    def test_missing_files_are_skipped(self) -> None:
        self.secret.write_text("BOT_NAME='Deployed'\n", encoding="utf-8")

        only_secret = self._settings(self.local, self.secret)
        self.secret.unlink()
        self.local.write_text("BOT_NAME=Local\n", encoding="utf-8")
        only_local = self._settings(self.local, self.secret)
        self.local.unlink()
        neither = self._settings(self.local, self.secret)

        self.assertEqual(only_secret.bot_name, "Deployed")
        self.assertEqual(only_local.bot_name, "Local")
        self.assertEqual(neither.bot_name, Settings.model_fields["bot_name"].default)

    def test_a_real_environment_variable_beats_both_files(self) -> None:
        self.local.write_text("BOT_NAME=Local\n", encoding="utf-8")
        self.secret.write_text("BOT_NAME='Deployed'\n", encoding="utf-8")

        with patch.dict(os.environ, {"BOT_NAME": "FromEnv", "APP_VERSION": "abc-1"}):
            settings = self._settings(self.local, self.secret)

        self.assertEqual(settings.bot_name, "FromEnv")

    def test_the_blob_made_from_env_overrides_a_stale_local_file(self) -> None:
        # What Cloud Run sees: the image has no .env, and even if a stale one
        # existed the deployed blob would win.
        self.local.write_text("LINE_CHANNEL_SECRET=old\nBOT_NAME=Old\n", encoding="utf-8")
        source = self.dir / "source.env"
        source.write_text(
            "LINE_CHANNEL_SECRET=new#secret\nGCP_PROJECT_ID=deploy-only\n"
            "SCHEDULED_WEEKLY_MESSAGES='[\n  {\"id\":\"w\",\"day_of_week\":\"mon\","
            "\"hour\":9,\"minute\":0,\"message\":\"早安 #1\"}\n]'\n",
            encoding="utf-8",
        )
        self.secret.write_text(
            envfile.to_dotenv(envfile.parse_env_file(source)), encoding="utf-8"
        )

        settings = self._settings(self.local, self.secret)

        self.assertEqual(settings.line_channel_secret, "new#secret")
        self.assertEqual(settings.bot_name, "Old")
        self.assertEqual(settings.scheduled_weekly_messages[0].message, "早安 #1")


class LegacyWarningsScanBothFilesTests(unittest.TestCase):
    def test_the_default_scans_the_files_settings_reads(self) -> None:
        default = inspect.signature(legacy_config_warnings).parameters["env_file"].default
        self.assertEqual(default, config.ENV_FILES)

    def test_a_stale_key_in_the_mounted_secret_is_reported_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / ".env"
            secret = Path(tmpdir) / "app.env"
            local.write_text("BOT_NAME=x\n", encoding="utf-8")
            secret.write_text("ORCHESTRATOR_MODEL='old/model'\n", encoding="utf-8")

            warnings = legacy_config_warnings(
                environ={}, env_file=(local, secret, Path(tmpdir) / "missing.env")
            )

        self.assertEqual(len(warnings), 1)
        self.assertIn(f"ORCHESTRATOR_MODEL in {secret} is no longer read", warnings[0])
        self.assertNotIn("old/model", warnings[0])

    def test_a_single_path_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / ".env"
            local.write_text("NVIDIA_MODEL=x\n", encoding="utf-8")

            warnings = legacy_config_warnings(environ={}, env_file=local)

        self.assertEqual(len(warnings), 1)
        self.assertIn("NVIDIA_MODEL", warnings[0])


class SuiteIsolationTests(unittest.IsolatedAsyncioTestCase):
    """tests/conftest.py: no real network, and no Settings keys from the shell."""

    def setUp(self) -> None:
        # Without the guard (python -m unittest, --noconftest) the tests
        # below would really try the network: stop before they do.
        if not conftest.network_guard_installed():
            self.fail("tests/conftest.py's network guard is not installed; run with pytest")
        # The refusals below are on purpose; the conftest would fail the test.
        self.addCleanup(conftest.BLOCKED_ATTEMPTS.clear)

    def test_outbound_connections_and_lookups_are_refused(self) -> None:
        with self.assertRaises(conftest.NetworkBlockedError):
            socket.create_connection(("192.0.2.1", 443), timeout=1)  # TEST-NET-1
        with self.assertRaises(conftest.NetworkBlockedError):
            socket.getaddrinfo("example.com", 443)
        self.assertEqual(len(conftest.BLOCKED_ATTEMPTS), 2)
        self.assertIn("192.0.2.1", conftest.BLOCKED_ATTEMPTS[0])

    def test_loopback_and_local_sockets_still_work(self) -> None:
        socket.getaddrinfo("localhost", 80)
        socket.getaddrinfo("0x7f000001", 80)  # numeric: no DNS
        with socket.create_server(("127.0.0.1", 0)) as server:
            with socket.create_connection(server.getsockname(), timeout=2):
                pass
        left, right = socket.socketpair()
        left.close()
        right.close()
        self.assertEqual(conftest.BLOCKED_ATTEMPTS, [])

    async def test_an_attempt_the_code_swallows_is_still_recorded(self) -> None:
        # The chain catches every exception and answers "degraded".
        # No proxy from the shell: the attempt must reach the guard itself.
        no_proxy = {key: "" for key in os.environ if key.lower().endswith("_proxy")}
        with patch.dict(os.environ, no_proxy):
            chain = ChainedSearchBackend([TavilyBackend("tvly-test")])
        try:
            response = await chain.search("q")
        finally:
            await chain.close()
        self.assertTrue(response.degraded)
        self.assertTrue(any("api.tavily.com" in attempt for attempt in conftest.BLOCKED_ATTEMPTS))

    def test_the_shell_cannot_configure_settings(self) -> None:
        environ = {"TAVILY_API_KEY": "tvly-k", "gcs_bucket_name": "b", "PATH": "/bin"}
        self.assertEqual(
            conftest.scrub_settings_env(environ), ["TAVILY_API_KEY", "gcs_bucket_name"]
        )
        self.assertEqual(environ, {"PATH": "/bin"})
        self.assertEqual(
            [key for key in os.environ if key.lower() in Settings.model_fields], []
        )


class DefaultSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_settings_enable_no_live_backend(self) -> None:
        # What every test starts from (the conftest drops Settings keys from
        # the environment): no search backend, model API, Colab or Cloud Tasks.
        settings = Settings(_env_file=None)
        chain = build_search_backend(settings)
        try:
            self.assertEqual(chain.backends, [])
        finally:
            await chain.close()
        self.assertEqual(build_providers(settings, build_model_health(settings)), {})
        self.assertIsNone(build_colab_endpoint(settings))
        self.assertEqual(select_tasks_mode(settings, {"K_SERVICE": "svc"}), "inline")
        self.assertFalse(settings.firestore_enabled)
        self.assertEqual(
            (settings.line_channel_access_token, settings.gcs_bucket_name, settings.tavily_api_key),
            ("", "", ""),
        )


if __name__ == "__main__":
    unittest.main()
