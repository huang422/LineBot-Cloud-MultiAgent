"""Security validators: HMAC signature, input cleaning, prompt-injection markers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import textwrap

from src.utils.logger import logger

# High-precision prompt-injection markers only: chat-template control
# tokens that no one types by accident, and explicit "ignore the rules"
# phrasing. Everyday wording ("you are now", "pretend that") is not
# listed: it caught ordinary messages, and no filter is a real defence.
# A match is stripped and the message still answered (with a guard line
# in the system prompt), never dropped.
_INJECTION_MARKERS = re.compile(
    r"<\|im_(?:start|end)\|>"
    r"|<\|(?:system|user|assistant)\|>"
    r"|\[/?INST\]"
    r"|<\|channel\|?>|<channel\|>"
    r"|system\s*prompt\s*[:：]"
    r"|忽略(?:之前|以上|所有|前面)(?:所有)?的?(?:指令|指示|規則)"
    r"|ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions",
    re.IGNORECASE,
)
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_LEADING_SPACE = re.compile(r"^[ \t]*")
_EXTRA_BLANK_LINES = re.compile(r"\n{4,}")

MAX_INPUT_LENGTH = 4000


def validate_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Validate LINE webhook signature (HMAC-SHA256).

    Fails closed and never raises: an empty channel secret would make the
    HMAC key public knowledge (anyone could sign), and a header that is
    empty or not ASCII cannot be a base64 digest.
    """
    if not channel_secret or not signature:
        return False
    try:
        provided = signature.encode("ascii")
        mac = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256)
    except (UnicodeEncodeError, TypeError):
        return False
    expected = base64.b64encode(mac.digest())
    return hmac.compare_digest(expected, provided)


def sanitize_input(text: str) -> str:
    """Clean user text before any model sees it.

    - Prompt-injection markers are removed (:func:`check_prompt_injection`
      tells whether there were any), until none is left: removing one
      must not join the halves of another ("ignore previous[INST]
      instructions").
    - Newlines survive, so pasted code, lists and tracebacks keep their
      shape: CRLF and CR become LF, the lines lose only the indentation
      they all share (a method copied out of a class keeps its relative
      indent), other runs of spaces or tabs become one space, at most two
      blank lines stay in a row, and the ends are trimmed.
    - Text longer than MAX_INPUT_LENGTH is cut, and the cut is logged.
    """
    text, stripped = _INJECTION_MARKERS.subn(" ", text or "")
    # Each pass shortens the text (every marker is longer than its space).
    while stripped:
        text, stripped = _INJECTION_MARKERS.subn(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        indent = _LEADING_SPACE.match(line).group(0)
        body = _HORIZONTAL_SPACE.sub(" ", line[len(indent):]).rstrip()
        lines.append(f"{indent}{body}" if body else "")
    text = _EXTRA_BLANK_LINES.sub("\n\n\n", textwrap.dedent("\n".join(lines))).strip()
    if len(text) > MAX_INPUT_LENGTH:
        logger.info(f"Input truncated from {len(text)} to {MAX_INPUT_LENGTH} characters")
        text = text[:MAX_INPUT_LENGTH].rstrip()
    return text


def check_prompt_injection(text: str) -> bool:
    """True when *text* carries a marker :func:`sanitize_input` strips."""
    return _INJECTION_MARKERS.search(text or "") is not None
