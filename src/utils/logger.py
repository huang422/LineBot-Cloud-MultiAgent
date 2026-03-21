"""Structured logging setup."""

from __future__ import annotations

import logging
import sys

_configured = False


def setup_logger(name: str = "linebot", level: str = "INFO") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)

    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        _configured = True

    # Always update level so lifespan reconfiguration takes effect
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logger


logger = setup_logger()
