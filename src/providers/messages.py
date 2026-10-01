"""Provider-safe shaping of OpenAI-format message lists.

Agents build conversations freely: a memory summary, quoted text, search
results and the user's own message often arrive as consecutive ``user``
turns. Strict chat templates (NVIDIA NIM requires alternating roles and
string-only system/assistant content; other templates reject repeated
roles) refuse that shape, so both providers pass every request through
:func:`normalize_messages` first.
"""

from __future__ import annotations

from typing import Any

_PART_SEPARATOR = "\n\n"


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Return a normalised copy of *messages*; the input is never mutated.

    - Consecutive ``user`` turns merge into one. Two strings join with a
      blank line; if either side is a content-part list, strings become
      ``{"type": "text"}`` parts and the lists concatenate in order.
    - Consecutive *leading* ``system`` turns merge into one string.
    - ``assistant`` turns (which may carry ``tool_calls``) and ``tool``
      turns are never merged.
    - ``system`` and ``assistant`` content is always a string: a part list
      collapses to its text parts.
    """
    normalized: list[dict] = []
    for message in messages:
        role = message.get("role")
        current = dict(message)
        if role in ("system", "assistant") and isinstance(current.get("content"), list):
            current["content"] = _text_of_parts(current["content"])

        previous = normalized[-1] if normalized else None
        if (
            role == "system"
            and previous is not None
            and len(normalized) == 1
            and previous.get("role") == "system"
        ):
            previous["content"] = _join_text(previous.get("content"), current.get("content"))
            continue
        if role == "user" and previous is not None and previous.get("role") == "user":
            previous["content"] = _merge_user_content(
                previous.get("content"), current.get("content")
            )
            continue
        normalized.append(current)
    return normalized


def has_image_parts(messages: list[dict]) -> bool:
    """True when any message carries an ``image_url`` content part."""
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


def strip_image_parts(messages: list[dict]) -> list[dict]:
    """A copy of *messages* without ``image_url`` parts, for text-only tiers.

    A part list keeps its other parts (a user turn left with text only
    becomes a plain string); a turn left with nothing is dropped. The input
    is never mutated.
    """
    stripped: list[dict] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue
        parts = [
            part
            for part in content
            if not (isinstance(part, dict) and part.get("type") == "image_url")
        ]
        if not parts:
            continue
        if all(isinstance(part, dict) and part.get("type") == "text" for part in parts):
            text = _PART_SEPARATOR.join(
                part["text"] for part in parts if isinstance(part.get("text"), str) and part["text"]
            )
            stripped.append({**message, "content": text})
        else:
            stripped.append({**message, "content": parts})
    return stripped


def _text_of_parts(parts: list[Any]) -> str:
    """Concatenate the text parts of a content list, dropping everything else."""
    return "".join(
        part["text"]
        for part in parts
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    )


def _join_text(first: Any, second: Any) -> str:
    return _PART_SEPARATOR.join(
        text for text in (first, second) if isinstance(text, str) and text
    )


def _merge_user_content(first: Any, second: Any) -> str | list[dict]:
    if isinstance(first, list) or isinstance(second, list):
        return _as_parts(first) + _as_parts(second)
    return _join_text(first, second)


def _as_parts(content: Any) -> list[dict]:
    """Content as a fresh part list (the parts themselves are shared, unchanged)."""
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []
