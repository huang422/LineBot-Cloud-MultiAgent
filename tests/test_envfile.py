from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import Settings
from scripts.envfile import parse_env_file


class EnvfileParsingTests(unittest.TestCase):
    def test_parse_env_file_supports_multiline_single_quoted_values(self) -> None:
        content = """SCHEDULED_WEEKLY_MESSAGES='[
  {"id":"monday_workout_reminder","day_of_week":"mon","hour":21,"minute":0,"message":"明天操一下嗎"},
  {"id":"pineapple_workout_reminder","day_of_week":"mon","hour":21,"minute":30,"message":"啊哈！@鳳梨 還沒回覆督促一下"}
]'
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(content, encoding="utf-8")

            values = parse_env_file(env_path)

        self.assertIn("SCHEDULED_WEEKLY_MESSAGES", values)
        self.assertTrue(values["SCHEDULED_WEEKLY_MESSAGES"].startswith("[\n"))
        self.assertIn("pineapple_workout_reminder", values["SCHEDULED_WEEKLY_MESSAGES"])

    def test_settings_accept_multiline_scheduler_json(self) -> None:
        content = """SCHEDULED_MESSAGES_ENABLED=true
SCHEDULED_GROUP_ID=C1234567890abcdef1234567890abcdef
SCHEDULED_WEEKLY_MESSAGES='[
  {"id":"monday_workout_reminder","day_of_week":"mon","hour":21,"minute":0,"message":"明天操一下嗎"},
  {"id":"pineapple_workout_reminder","day_of_week":"mon","hour":21,"minute":30,"message":"啊哈！@鳳梨 還沒回覆督促一下"}
]'
SCHEDULED_YEARLY_MESSAGES='[
  {"id":"birthday_reminder_0312","month":3,"day":12,"hour":9,"minute":0,"message":"啊哈！@保羅 生日快樂！"}
]'
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(content, encoding="utf-8")

            settings = Settings(_env_file=env_path)

        self.assertTrue(settings.scheduled_messages_enabled)
        self.assertEqual(
            [job.id for job in settings.scheduled_weekly_messages],
            ["monday_workout_reminder", "pineapple_workout_reminder"],
        )
        self.assertEqual(
            [job.id for job in settings.scheduled_yearly_messages],
            ["birthday_reminder_0312"],
        )
