"""An empty value means the default wherever the deploy script says so.

scripts/deploy_cloud_run.sh fills in the default for an empty value
(``[[ -z "$X_VALUE" ]] && X_VALUE="default"``) and validates that instead,
so the app must read ``KEY=`` the same way: a Settings that refused it would
leave the new revision never ready after a build and a new secret version.
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src import config
from src.config import Settings

_DEPLOY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_cloud_run.sh"
_READ_RE = re.compile(r'^\s*(\w+_VALUE)="\$\(read_env_key (\w+)\b', re.MULTILINE)
_DEFAULT_RE = re.compile(r'^\s*\[\[ -z "\$(\w+_VALUE)" \]\] && \1="([^"$]*)"$', re.MULTILINE)

# Keys whose empty value keeps a meaning of its own (or whose default is
# empty anyway): the rule must leave them "".
_EMPTY_IS_KEPT = (
    "CLOUD_TASKS_LOCATION",  # empty = the Cloud Run region
    "TAVILY_API_KEY",
    "INTERNAL_CRON_SECRET",
    "INTERNAL_TASKS_SECRET",
    "PUBLIC_BASE_URL",
    "SCHEDULED_GROUP_ID",
    "GCS_BUCKET_NAME",
    "COLAB_SHARED_SECRET",
    "FORCE_TIER",
    "MODEL_ROLES",
)


def deploy_defaults() -> dict[str, str]:
    """{env key: the default the deploy uses for an empty value}."""
    text = _DEPLOY_SCRIPT.read_text(encoding="utf-8")
    keys = dict(_READ_RE.findall(text))
    return {keys[var]: default for var, default in _DEFAULT_RE.findall(text)}


def as_deploy_text(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


class EmptyMeansDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.env_file = Path(tmpdir.name) / "app.env"
        environ = patch.dict(os.environ, {}, clear=True)
        environ.start()
        self.addCleanup(environ.stop)

    def _settings(self, line: str) -> Settings:
        self.env_file.write_text(f"{line}\n", encoding="utf-8")
        return Settings(_env_file=self.env_file)

    def test_the_deploy_script_defaults_are_found(self) -> None:
        # Guards the parse: an empty result would make the tests below vacuous.
        self.assertLessEqual(
            {
                "TASKS_MODE",
                "TASK_DISPATCH_DEADLINE_SECONDS",
                "TASK_HANDLER_BUDGET_SECONDS",
                "SCHEDULED_MESSAGES_ENABLED",
                "PASSIVE_GROUP_RECORDING_ENABLED",
                "COLAB_ENABLED",
                "CLOUD_TASKS_QUEUE",
                "CLOUD_TASKS_COLAB_QUEUE",
            },
            set(deploy_defaults()),
        )

    def test_the_deploy_script_reads_no_removed_key(self) -> None:
        # The push budget's keys (LINE_PUSH_*) went with it on 2026-10-01.
        read = {key for _, key in _READ_RE.findall(_DEPLOY_SCRIPT.read_text(encoding="utf-8"))}
        self.assertEqual(read & set(config.REMOVED_ENV_KEYS), set())

    def test_the_app_default_is_the_deploy_default(self) -> None:
        for key, deploy_default in deploy_defaults().items():
            with self.subTest(key=key):
                field = Settings.model_fields[key.lower()]
                self.assertEqual(as_deploy_text(field.default), deploy_default)

    def test_every_key_the_deploy_defaults_reads_empty_as_the_default(self) -> None:
        for key in deploy_defaults():
            default = Settings.model_fields[key.lower()].default
            # A plain .env line and the single-quoted line of the mounted blob.
            for line in (f"{key}=", f"{key}=''", f"{key}='  '"):
                with self.subTest(line=line):
                    self.assertEqual(getattr(self._settings(line), key.lower()), default)

    def test_an_empty_environment_variable_means_the_default_too(self) -> None:
        with patch.dict(os.environ, {"TASKS_MODE": "", "COLAB_ENABLED": " "}):
            settings = Settings(_env_file=None)
        self.assertEqual((settings.tasks_mode, settings.colab_enabled), ("auto", False))

    def test_empty_colab_adc_path_is_the_mounted_path(self) -> None:
        # The deploy warns only about a non-empty path other than its mount.
        self.assertEqual(
            self._settings("COLAB_ADC_PATH=").colab_adc_path,
            Settings.model_fields["colab_adc_path"].default,
        )

    def test_empty_scheduled_job_lists_mean_no_jobs(self) -> None:
        # The deploy reads an empty list as "no jobs configured".
        for key in ("SCHEDULED_WEEKLY_MESSAGES", "SCHEDULED_YEARLY_MESSAGES"):
            for line in (f"{key}=", f"{key}=''", f"{key}=[]"):
                with self.subTest(line=line):
                    self.assertEqual(getattr(self._settings(line), key.lower()), [])

    def test_scheduled_job_lists_are_still_json(self) -> None:
        settings = self._settings(
            'SCHEDULED_WEEKLY_MESSAGES=\'[{"id":"w","day_of_week":"MON","hour":9,'
            '"minute":0,"message":"hi"}]\''
        )
        self.assertEqual(settings.scheduled_weekly_messages[0].day_of_week, "mon")
        with self.assertRaises(ValidationError):
            self._settings("SCHEDULED_YEARLY_MESSAGES=[{")

    def test_keys_where_empty_has_a_meaning_stay_empty(self) -> None:
        for key in _EMPTY_IS_KEPT:
            with self.subTest(key=key):
                self.assertNotIn(key.lower(), config.EMPTY_MEANS_DEFAULT)
                self.assertEqual(getattr(self._settings(f"{key}=''"), key.lower()), "")
        self.assertEqual(Settings.model_fields["cloud_tasks_location"].default, "us-west1")

    def test_an_invalid_value_is_still_refused(self) -> None:
        for line in (
            "TASKS_MODE=sometimes",
            "TASK_HANDLER_BUDGET_SECONDS=x",
            "COLAB_ENABLED=maybe",
        ):
            with self.subTest(line=line), self.assertRaises(ValidationError):
                self._settings(line)


if __name__ == "__main__":
    unittest.main()
