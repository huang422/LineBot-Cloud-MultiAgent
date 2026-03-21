"""Security validators: HMAC signature + prompt injection detection."""

from __future__ import annotations

import hashlib
import hmac
import base64
import re

from src.utils.logger import logger

# Prompt injection patterns
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|above|all)\s+instructions", re.I),
    re.compile(r"disregard\s+(previous|above|all)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"act\s+as\s+if\s+", re.I),
    re.compile(r"pretend\s+(you|that)\s+", re.I),
    re.compile(r"\[INST\]", re.I),
    re.compile(r"<\|system\|>", re.I),
    re.compile(r"system\s*prompt\s*:", re.I),
    re.compile(r"override\s+.*instructions", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
    re.compile(r"forget\s+(everything|all|previous)", re.I),
    re.compile(r"do\s+not\s+follow\s+", re.I),
]

MAX_INPUT_LENGTH = 4000


def validate_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Validate LINE webhook signature (HMAC-SHA256)."""
    mac = hmac.new(
        channel_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    )
    expected = base64.b64encode(mac.digest()).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def sanitize_input(text: str) -> str:
    """Sanitize user input: truncate + normalize whitespace."""
    text = text[:MAX_INPUT_LENGTH]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_prompt_injection(text: str) -> bool:
    """Return True if prompt injection is suspected."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(f"Prompt injection detected: {pattern.pattern}")
            return True
    return False
