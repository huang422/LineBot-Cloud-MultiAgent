"""Turn a model's Markdown/LaTeX reply into plain text for a LINE message.

LINE shows text messages verbatim, and the reply models (Nemotron Ultra in
particular) write Markdown and LaTeX even for simple questions, so users
saw ``**粗體**`` and ``\\( 1/6 + 1/3 \\)`` literally. :func:`to_line_text`
keeps the words and drops the markup:

- ``**x**``, ``***x***`` and ``__x__`` become ``x``, unless an ASCII letter
  or digit (or a ``/``) touches a marker from outside (``2**3``,
  ``**kwargs``, ``src/**/*.ts`` and ``snake__case`` stay; ``這是**重點**。``
  still loses its markers); ``__x__`` also needs non-ASCII-word text right
  inside its markers, so ``__init__`` stays and ``__重點__`` goes;
- a leading ``##`` to ``######`` heading marker is removed (``## 標題``
  becomes ``標題``); a ``# `` line stays, as it is as likely a code comment;
- a fenced code block keeps its lines, verbatim, without the fences (the
  reply prompts ask for code in fences);
- inline code keeps its text without the backticks;
- LaTeX ``\\( x \\)``, ``\\[ x \\]`` and ``$$x$$`` become ``x``, with a few
  common commands inside them spelled out (``\\frac{1}{2}`` is ``1/2``,
  ``\\frac{a+b}{2}`` is ``(a+b)/2``, ``\\times`` is ``×``, ``\\text{x}``
  and ``\\boxed{x}`` are ``x``, ``\\sqrt{2}`` is ``√2``, ``\\%`` is
  ``%``; ``\\left``/``\\right`` and ``aligned``-style environments lose
  their markup, one line per row); a span with ASCII text right against
  it (bar punctuation after it) or with quotes inside is a regex or shell
  escape and stays
  (``r"\\((\\d+)\\)"``, ``find . \\( -name '*.py' \\)``);
- a Markdown link ``[text](http…)`` becomes ``text url`` (``a[b](c)``
  stays);
- table rows stay as they are, except the ``|---|---|`` separator row.

URLs, emoji, list markers and everything inside code are left untouched.
Unfenced code still goes through these rules, which are kept narrow so
that code survives them too.
"""

from __future__ import annotations

import re

# A fence opens with 3+ backticks or tildes after any indent (a fence
# under a list item is indented 4+ spaces). A backtick fence's info
# string has no backtick, so "```echo hi``` 然後" is inline code.
_FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}(?=[^`]*$)|~{3,})")
# Whole backtick runs only, so a long run costs linear time, not cubic.
_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
# Web links only: handlers[kind](event) is code.
_LINK_RE = re.compile(r"!?\[([^\]\n]+)\]\(\s*(https?://\S+?)\s*\)")
# A URL ends at whitespace, at "*" or "`" so markup around it still goes,
# and at CJK text or fullwidth punctuation, which replies put right after
# a link ("https://a.com，用`x`"). Private-use characters (the hold
# markers below) end it too.
_URL_RE = re.compile(r"https?://[^\s*`<>\u2e80-\u9fff\ue000-\ufaff\ufe30-\ufe4f\uff00-\uffef]+")
# ASCII guards only: 2**3, x**2 and **kwargs stay, while CJK may touch
# the markers ("這是**重點**。"). "*" is guarded too, so a "***" run is
# never split into "**" plus a stray "*".
_BOLD_ITALIC_RE = re.compile(r"(?<![0-9A-Za-z*])\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*(?![0-9A-Za-z*])")
# "/" too: a glob pair ("src/**/*.ts", "tests/**/*.ts") is not bold.
_BOLD_RE = re.compile(r"(?<![0-9A-Za-z*/])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![0-9A-Za-z*/])")
# Same guards (snake__case stays), and no ASCII word character right inside
# the markers either: __init__ and "__main__" are identifiers.
_UNDERSCORE_BOLD_RE = re.compile(
    r"(?<![0-9A-Za-z_])__(?=[^\s0-9A-Za-z_])(.+?)(?<=[^\s0-9A-Za-z_])__(?![0-9A-Za-z_])"
)
_LATEX_INLINE_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_LATEX_DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_LATEX_DOLLARS_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Math stands apart from other ASCII text; a regex or shell escape touches
# it (r"\((\d+)\)", s/\(x\)/, a\[0\]) or holds quotes (find \( -name '*.py'
# \), echo "$$" ... kill $$). A prime (f'(x), y'') is not a quote.
_MATH_BEFORE_RE = re.compile(r"\s|[^\x00-\x7f]")
_MATH_AFTER_RE = re.compile(r"[\s.,;:!?]|[^\x00-\x7f]")
_QUOTE_RE = re.compile(r"\"|(?<![\w)}'])'")
# "##" and deeper, at the line start: "# x" is as likely a code comment.
_HEADING_RE = re.compile(r"^#{2,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$", re.MULTILINE)
# A table separator row: only pipes, dashes, colons and spaces, with a pipe.
_TABLE_SEPARATOR_RE = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+(?:[ \t]*:?-+:?[ \t]*)?\|?[ \t]*$")
# Inside math only: commands with a plain-text spelling.
_FRAC_RE = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
_TEXT_RE = re.compile(r"\\(?:text|mathrm|textbf|mathbf|operatorname|boxed)\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\{([^{}]*)\}")
# A fraction part or radicand that needs no parentheses: one number, one
# word or one command.
_ATOM_RE = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+|\\[A-Za-z]+")
_ENVIRONMENT_RE = re.compile(r"\\(?:begin|end)\{[A-Za-z*]+\}")
# An alignment tab (a bare "&"; "\\&" is a literal ampersand).
_ALIGN_TAB_RE = re.compile(r"(?<!\\)&")
_MATH_SYMBOLS = {
    "\\times": "×",
    "\\cdot": "·",
    "\\div": "÷",
    "\\pm": "±",
    "\\leq": "≤",
    "\\geq": "≥",
    "\\le": "≤",
    "\\ge": "≥",
    "\\neq": "≠",
    "\\approx": "≈",
    "\\quad": " ",
    "\\,": " ",
    "\\;": " ",
    "\\!": "",
    "\\%": "%",
    "\\&": "&",
    "\\left": "",
    "\\right": "",
}
_MATH_SYMBOL_RE = re.compile(
    "|".join(re.escape(command) + (r"(?![A-Za-z])" if command[-1].isalpha() else "") for command in _MATH_SYMBOLS)
)
# Private-use characters mark the spans that must come back verbatim.
_HOLD_OPEN, _HOLD_CLOSE = "\ue000", "\ue001"
_HOLD_RE = re.compile(f"{_HOLD_OPEN}(\\d+){_HOLD_CLOSE}")


def to_line_text(text: str) -> str:
    """*text* with Markdown and LaTeX markup removed, for a LINE text message."""
    if not text:
        return text
    lines: list[str] = []
    for is_code, segment in _split_fences(text.split("\n")):
        lines.extend(segment if is_code else _plain("\n".join(segment)).split("\n"))
    return "\n".join(lines)


def _split_fences(lines: list[str]) -> list[tuple[bool, list[str]]]:
    """``(is_code, lines)`` runs, with the fence lines themselves dropped.

    An unclosed fence (a reply cut off at max_tokens) runs to the end.
    """
    segments: list[tuple[bool, list[str]]] = []
    current: list[str] = []
    fence = ""
    for line in lines:
        if not fence:
            match = _FENCE_OPEN_RE.match(line)
            if match:
                segments.append((False, current))
                current, fence = [], match.group(1)
                continue
        elif _closes(line, fence):
            segments.append((True, current))
            current, fence = [], ""
            continue
        current.append(line)
    segments.append((bool(fence), current))
    return [(is_code, segment) for is_code, segment in segments if segment]


def _closes(line: str, fence: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(fence) and set(stripped) == {fence[0]}


def _plain(text: str) -> str:
    """Prose without markup; inline code and URLs come back verbatim."""
    held: list[str] = []
    # The markers must be ours alone: a stray one would restore the wrong
    # held text, or none at all.
    text = text.replace(_HOLD_OPEN, "").replace(_HOLD_CLOSE, "")

    def hold(value: str) -> str:
        held.append(value)
        return f"{_HOLD_OPEN}{len(held) - 1}{_HOLD_CLOSE}"

    text = _INLINE_CODE_RE.sub(lambda m: hold(m.group(2).strip() or m.group(2)), text)
    text = _LINK_RE.sub(lambda m: _link(m.group(1), m.group(2)), text)
    text = _URL_RE.sub(lambda m: hold(m.group(0)), text)

    text = _BOLD_ITALIC_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _UNDERSCORE_BOLD_RE.sub(r"\1", text)
    for pattern in (_LATEX_DISPLAY_RE, _LATEX_INLINE_RE, _LATEX_DOLLARS_RE):
        text = pattern.sub(_math_span, text)
    text = _HEADING_RE.sub(r"\1", text)
    text = "\n".join(line for line in text.split("\n") if not _TABLE_SEPARATOR_RE.match(line))

    return _HOLD_RE.sub(lambda m: held[int(m.group(1))], text)


def _math_span(match: re.Match[str]) -> str:
    """A LaTeX span as plain text, or verbatim when it is code, not math."""
    text, start, end = match.string, match.start(), match.end()
    before = text[start - 1] if start else " "
    after = text[end] if end < len(text) else " "
    if (
        _MATH_BEFORE_RE.match(before)
        and _MATH_AFTER_RE.match(after)
        and not _QUOTE_RE.search(match.group(1))
    ):
        return _math(match.group(1))
    return match.group(0)


def _math(expression: str) -> str:
    """A LaTeX math span in plain text, as far as a few commands allow."""
    expression = _ENVIRONMENT_RE.sub("", expression)
    # Innermost first: \boxed{\frac{1}{2}} needs the fraction gone first.
    for _ in range(3):
        unwrapped = _TEXT_RE.sub(r"\1", _FRAC_RE.sub(_frac, _SQRT_RE.sub(_sqrt, expression)))
        if unwrapped == expression:
            break
        expression = unwrapped
    expression = _ALIGN_TAB_RE.sub("", expression.replace("\\\\", "\n"))
    expression = _MATH_SYMBOL_RE.sub(lambda m: _MATH_SYMBOLS[m.group(0)], expression)
    rows = (" ".join(row.split()) for row in expression.split("\n"))
    return "\n".join(row for row in rows if row)


def _frac(match: re.Match[str]) -> str:
    return f"{_grouped(match.group(1))}/{_grouped(match.group(2))}"


def _sqrt(match: re.Match[str]) -> str:
    return f"√{_grouped(match.group(1))}"


def _grouped(part: str) -> str:
    """*part* in parentheses unless it is one atom (``_ATOM_RE``)."""
    part = part.strip()
    return part if _ATOM_RE.fullmatch(part) else f"({part})"


def _link(label: str, url: str) -> str:
    label = label.strip()
    return url if label == url else f"{label} {url}"
