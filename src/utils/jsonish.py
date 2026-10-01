"""Tolerant extraction of a JSON object from LLM prose.

Models routinely wrap their JSON in code fences, prepend reasoning, use
full-width punctuation, or emit Python-style ``True``/``None``. These
helpers recover a dict from all of those without the caller having to
care which flavour arrived.
"""

from __future__ import annotations

import ast
import json
import re

_JSONISH_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "：": ":",
        "，": ",",
    }
)


def iter_json_candidates(text: str):
    seen: set[str] = set()

    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}") and candidate not in seen:
            seen.add(candidate)
            yield candidate

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}") and stripped not in seen:
        seen.add(stripped)
        yield stripped

    for start_match in re.finditer(r"\{", text):
        depth = 0
        in_string = False
        escaped = False

        for index in range(start_match.start(), len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_match.start(): index + 1].strip()
                    if candidate not in seen:
                        seen.add(candidate)
                        yield candidate
                    break


def load_jsonish_dict(candidate: str) -> dict | None:
    """The dict in *candidate*, or None.

    The candidate is parsed as it is first: valid JSON whose Chinese values
    contain curly quotes or full-width commas must come back unchanged.
    Only when that fails are curly and full-width punctuation replaced by
    their ASCII forms, and then Python literals (``True``/``None``, single
    quotes) accepted.
    """
    raw = candidate.strip()
    parsed = _loads(raw)
    if parsed is not None:
        return parsed

    normalized = raw.translate(_JSONISH_TRANSLATION)
    if normalized != raw:
        parsed = _loads(normalized)
        if parsed is not None:
            return parsed

    pythonish = re.sub(r"\btrue\b", "True", normalized, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)

    try:
        parsed = ast.literal_eval(pythonish)
        return parsed if isinstance(parsed, dict) else None
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def _loads(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
