"""Labels that untrusted text must not carry into a fenced prompt block.

Reference text (page text, search results, stored facts) goes into the
same merged user turn as the user's message, between labels the code
writes. A label inside that text could end its block early or pose as the
user's message (【使用者這次的訊息】), so :func:`strip_labels` removes the
given labels and their look-alikes: any bracket or none around them,
spaces or zero-width characters inside, fullwidth letters and punctuation
for ASCII ones and the reverse. Taking a label out can join the text
around it into another (【使用者這次【使用者這次的訊息】的訊息】), so it
repeats until nothing is left to take out. It is a filter, not the
defence: the blocks still say where they end and that their text is not an
instruction.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

# What may sit between a label's characters: spaces and invisible ones.
_GAP = r"[\s­​-‏⁠-⁤﻿]*"
_OPEN = r"[\[【〖〔〘「『《〈<(（［]"
_CLOSE = r"[\]】〗〕〙」』》〉>)）］]"
# Passes over the text; one more label nesting level needs one more pass.
# No real text nests labels, so one still changing on the last pass was
# built to outlast the filter.
_MAX_PASSES = 8


def strip_labels(text: str, labels: Iterable[str]) -> str:
    """*text* with each of *labels*, look-alikes included, replaced by a
    space until none is left; text that nests them deeper than
    _MAX_PASSES allows becomes a single space."""
    patterns = [_label_pattern(label) for label in labels]
    for _ in range(_MAX_PASSES):
        stripped = text
        for pattern in patterns:
            stripped = pattern.sub(" ", stripped)
        if stripped == text:
            return text
        text = stripped
    return " "


@lru_cache(maxsize=64)
def _label_pattern(label: str) -> re.Pattern[str]:
    core = _GAP.join(_variants(char) for char in label.strip("[]【】") if not char.isspace())
    return re.compile(f"(?:{_OPEN}{_GAP})?{core}(?:{_GAP}{_CLOSE})?", re.IGNORECASE)


def _variants(char: str) -> str:
    """A character class for *char* and its fullwidth or NFKC twin."""
    forms = {char}
    folded = unicodedata.normalize("NFKC", char)
    if len(folded) == 1:
        forms.add(folded)
    if "!" <= char <= "~":
        forms.add(chr(ord(char) + 0xFEE0))
    return "[" + "".join(re.escape(form) for form in sorted(forms)) + "]"
