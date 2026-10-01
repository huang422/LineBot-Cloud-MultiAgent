"""Each app log line goes out once, in the app's format (src/utils/logger.py)."""

from __future__ import annotations

import io
import logging
import unittest

from src.utils.logger import logger


class SingleLogLineTests(unittest.TestCase):
    def test_a_root_handler_does_not_print_app_lines_a_second_time(self) -> None:
        # A library's root-level logging call (colab_cli's logging.debug)
        # installs logging.basicConfig's handler on the root logger; every
        # app line then also came out as "INFO:linebot:...".
        root_stream = io.StringIO()
        root_handler = logging.StreamHandler(root_stream)
        logging.getLogger().addHandler(root_handler)
        self.addCleanup(logging.getLogger().removeHandler, root_handler)
        # pytest adds its own capture handlers (subclasses) to a non-propagating logger.
        [app_handler] = [h for h in logger.handlers if type(h) is logging.StreamHandler]
        app_stream = io.StringIO()
        self.addCleanup(app_handler.setStream, app_handler.setStream(app_stream))
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel(logging.INFO)

        logger.info("只出現一次")

        self.assertEqual(root_stream.getvalue(), "")
        self.assertRegex(
            app_stream.getvalue(),
            r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \| INFO    \| linebot \| 只出現一次\n$",
        )


if __name__ == "__main__":
    unittest.main()
