"""SCHEDULED_MESSAGES_TIMEZONE never stops startup.

A blank value is the default zone (EMPTY_MEANS_DEFAULT); a value that is no
zone at all falls back to UTC with a warning, as an unknown zone does.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.services import scheduler_service
from src.services.scheduler_service import SchedulerService


class ScheduledMessagesTimezoneTests(unittest.TestCase):
    def test_a_blank_timezone_is_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp) / "app.env"
            for line in ("SCHEDULED_MESSAGES_TIMEZONE=", "SCHEDULED_MESSAGES_TIMEZONE='  '"):
                with self.subTest(line=line):
                    env_file.write_text(f"{line}\n", encoding="utf-8")
                    settings = Settings(_env_file=env_file)
                    self.assertEqual(settings.scheduled_messages_timezone, "Asia/Taipei")
        with patch.dict(os.environ, {"SCHEDULED_MESSAGES_TIMEZONE": ""}, clear=True):
            self.assertEqual(Settings(_env_file=None).scheduled_messages_timezone, "Asia/Taipei")

    def test_a_value_that_is_no_zone_falls_back_to_utc_with_a_warning(self) -> None:
        for name in ("", "/etc/localtime", "../Asia/Taipei", "Asia/Nowhere"):
            with self.subTest(name=name), patch.object(scheduler_service.logger, "warning") as warn:
                scheduler = SchedulerService(timezone_name=name)
                self.assertIs(scheduler.timezone, UTC)
                self.assertIn(repr(name), warn.call_args.args[0])

    def test_a_real_zone_is_used(self) -> None:
        self.assertEqual(str(SchedulerService("Asia/Taipei").timezone), "Asia/Taipei")


if __name__ == "__main__":
    unittest.main()
