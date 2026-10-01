"""Marks the lines of a group message that start like a speaker tag.

In a group the agents read each member's message under a ``User_xxxx:``
tag (``User_`` and the last four characters of the LINE user id, which
are hex) and the bot's turns as ``Assistant`` or under its display name.
A line of a message that starts like one of those tags would pose as
another member or the bot, so :func:`mark_speaker_lines` puts
:data:`SPEAKER_LINE_MARKER` before it.

The check runs on a folded copy of the line, so what only looks different
does not hide a tag: NFKC (full-width letters and colons), invisible
characters dropped (format characters, variation selectors, the combining
grapheme joiner, Hangul fillers, the blank Braille pattern), Cyrillic,
Greek and Armenian letters and Latin small capitals that look like Latin
letters read as those, colon
and dash look-alikes read as ``:`` and ``-``. Markdown and list
decorations around the tag (``**Assistant:**``, ``> Assistant:``,
``[Assistant]:``, ``- User_0001:``) and spaces between the letters of a
name (``小 幫手:``) do not hide it either. A tag needs the real shape:
``user_id: str`` in pasted code is not one.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Put before a line of a group message that starts like a speaker tag, so
# the line reads as part of the message it is in, not as another speaker.
SPEAKER_LINE_MARKER = "〔同一則訊息〕"

# Characters that show as nothing or as a blank but are not format
# characters (category Cf, dropped too): the combining grapheme joiner,
# Hangul fillers, Khmer inherent vowels, Mongolian free variation
# selectors, variation selectors and the blank Braille pattern.
_INVISIBLE = frozenset(
    [0x034F, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x2800, 0x3164, 0xFFA0]
    + list(range(0x180B, 0x1810))
    + list(range(0xFE00, 0xFE10))
    + list(range(0xE0100, 0xE01F0))
)

# Letters that look like Latin ones (Cyrillic, Greek, Armenian, Latin small
# capitals), and colon and dash look-alikes that NFKC leaves alone.
_LOOK_ALIKES = str.maketrans(
    {
        **dict(zip("АВЕЁІЇЈКМНОРСТУХЅԚԜҮҺӀ", "ABEEIIJKMHOPCTYXSQWYHI", strict=True)),
        **dict(zip("аеёіїјкорсухѕԁԛԝһӏ", "aeeiijkopcyxsdqwhl", strict=True)),
        **dict(zip("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧϹ", "ABEZHIKMNOPTYXC", strict=True)),
        **dict(zip("αεικνορυχϲϳ", "aeikvopuxcj", strict=True)),
        **dict(zip("օսոհ", "ounh", strict=True)),
        **dict(zip("ᴀʙᴄᴅᴇꜰɪɴʀꜱᴛᴜ", "abcdefinrstu", strict=True)),
        **dict.fromkeys("∶꞉ːˑ˸։׃܃܄፥᛬᠄⁚⦂", ":"),
        **dict.fromkeys("‐‑‒–—―−⁃˗﹘﹣", "-"),
    }
)

# Decorations that may come before a tag and between it and its colon.
_BEFORE = r"[\s*_>#~+\-•·・@\[(【「『《〈]*"
_AFTER = r"[\s*_~\])】」』》〉]*"


def _invisible(char: str) -> bool:
    return ord(char) in _INVISIBLE or unicodedata.category(char) == "Cf"


def _folded(text: str) -> str:
    """*text* as the tag check reads it (see the module docstring)."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(char for char in normalized if not _invisible(char)).translate(_LOOK_ALIKES)


def _spaced(name: str) -> str:
    """A pattern for *name* with any spaces between its characters."""
    return r"\s*".join(re.escape(char) for char in name if not char.isspace())


@lru_cache(maxsize=8)
def _speaker_tag(bot_name: str) -> re.Pattern[str]:
    """A speaker tag at the start of a folded line: ``User``, ``User_xxxx``
    (four hex digits; a space or ``-`` in place of the ``_`` too),
    ``Assistant`` or *bot_name*, then a colon."""
    names = [_spaced("user") + r"(?:[\s_\-]*[0-9a-f]{4})?", _spaced("assistant")]
    bot = _spaced(_folded(bot_name))
    if bot:
        names.append(bot)
    return re.compile(rf"{_BEFORE}(?:{'|'.join(names)}){_AFTER}:", re.IGNORECASE)


def mark_speaker_lines(text: str, *, bot_name: str = "") -> str:
    """*text* with :data:`SPEAKER_LINE_MARKER` before each line that starts
    like a speaker tag (:func:`_speaker_tag`; *bot_name* is the bot's
    display name).

    The line keeps its text and its indentation (the marker goes after the
    leading spaces); every other line, and every line break, is left as
    it was.
    """
    pattern = _speaker_tag(bot_name)
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if pattern.match(_folded(line)):
            indent = len(line) - len(line.lstrip())
            lines[index] = f"{line[:indent]}{SPEAKER_LINE_MARKER}{line[indent:]}"
    return "".join(lines)
