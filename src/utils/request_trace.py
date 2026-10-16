"""What one handled LINE event did, for its one-line summary log.

The event pipeline opens a :class:`RequestTrace` per event (:func:`activate`).
It lives in a :class:`~contextvars.ContextVar`, so concurrent events never
share one, while the tasks an event starts (the image caption) inherit it.
The code that knows what happened adds to it; every ``record``/``note``
function is a no-op when no trace is active (scripts, tests, background
work):

- :class:`~src.providers.fallback_chain.FallbackChain` records each walk
  (:func:`record_walk`): every tier with its outcome and seconds;
- the orchestrator records who chose the agent (:func:`note_route`), and the
  Colab router (clef-flash) its scores, accepted or not
  (:func:`note_router_scores`), or why it was not used
  (:func:`note_router_skip`);
- the image agent records its image tiers as a walk, and the output
  processor the TTS engine (:func:`note_tts`);
- the LINE service records how each send went out (:func:`note_delivery`).

:func:`phase` labels the walks: ``route`` (the orchestrator), ``answer``
(the pipeline wraps the agent) and, inside the answer, ``search_plan``,
``image_prompt`` and ``image_caption``, so the answer's own walks can be
told apart. The answering tier is the tier that answered the last
``answer`` walk: that walk's response is the one the agent used (a tool
loop makes several walks, and a round that only asks for tools is
``tools``, not ``ok``). A failed last walk means no answer, even when an
earlier walk succeeded.

:func:`summary_line` writes the line (DEPLOY.md, 〈實際測試時怎麼看 log〉).
It holds tier labels, fixed outcome words and numbers only: never the
user's text, a prompt, a URL or an error message.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

# Walk phases.
ROUTE = "route"
ANSWER = "answer"
SEARCH_PLAN = "search_plan"
IMAGE_PROMPT = "image_prompt"
IMAGE_CAPTION = "image_caption"
OTHER = "other"

# Attempt outcomes (a skip is "skip(<why>)", an error "error(<Kind> <status>)").
OK = "ok"
# The call asked for tools and wrote no text: a tool-loop round, not an answer.
TOOLS = "tools"
TIMEOUT = "timeout"
RATE_LIMITED = "rate_limited"
EMPTY = "empty"

# Who chose the agent (RequestTrace.route).
ROUTE_RULES = "rules"
ROUTE_ROUTER = "router"
ROUTE_LLM = "llm"
ROUTE_FALLBACK = "fallback"

# RouterScores.verdict when the Colab router's decision was used.
ACCEPTED = "ok"

# How a send went out (RequestTrace.delivery): the free reply, or a push.
REPLY = "reply"
PUSH = "push"

_SKIP_DETAIL_CHARS = 60
_COLAB_PROVIDER = "colab"

_TRACE: ContextVar[RequestTrace | None] = ContextVar("request_trace", default=None)
_PHASE: ContextVar[str] = ContextVar("request_trace_phase", default=OTHER)


@dataclass(frozen=True)
class Attempt:
    """One tier of one walk: ``ok``, ``tools``, ``skip(<why>)``,
    ``timeout``, ``rate_limited``, ``empty`` or ``error(<Kind> <status>)``;
    ``seconds`` is None for a skip. ``think`` is how a text tier thought:
    ``off``, ``on`` or its level (Colab chat thinks at COLAB_CHAT_THINK even
    when the route says thinking off)."""

    label: str
    outcome: str
    seconds: float | None = None
    think: str = ""


def skipped(label: str, why: str) -> Attempt:
    why = " ".join(why.split())
    if len(why) > _SKIP_DETAIL_CHARS:
        why = why[: _SKIP_DETAIL_CHARS - 1] + "…"
    return Attempt(label, f"skip({why})")


def error_outcome(error: BaseException) -> str:
    """``error(<Kind> <status>)``, or ``error(<Kind>)`` without a status:
    the class name only, never the message."""
    status = getattr(error, "status", None)
    kind = type(error).__name__
    return f"error({kind} {status})" if isinstance(status, int) and status else f"error({kind})"


@dataclass(frozen=True)
class Walk:
    phase: str
    attempts: tuple[Attempt, ...]

    @property
    def answered_by(self) -> Attempt | None:
        return next((attempt for attempt in reversed(self.attempts) if attempt.outcome == OK), None)


@dataclass(frozen=True)
class RouterScores:
    """The Colab router's answer: probabilities per option (sorted, rounded, only the
    defined options), the confidences against the gate, ``needs_thinking``
    against its thresholds, and what came of it."""

    model: str
    agent: Mapping[str, float]
    agent_confidence: float
    output: Mapping[str, float]
    output_confidence: float
    think_p: float
    gate: float
    think_on: float
    think_off: float
    # ACCEPTED, or why the API router decides instead ("gate fail", ...).
    verdict: str = ACCEPTED
    # How thinking was set when accepted: "on", "off", "rules->on", ...
    thinking: str = ""


def probabilities(values: object, allowed: Iterable[str]) -> dict[str, float]:
    """*values* (``{option: p}``) limited to *allowed*, highest first, to 2 decimals."""
    if not isinstance(values, Mapping):
        return {}
    allowed = set(allowed)
    kept = [
        (option, float(p))
        for option, p in values.items()
        if option in allowed and isinstance(p, (int, float)) and not isinstance(p, bool)
    ]
    kept.sort(key=lambda item: (-item[1], item[0]))
    return {option: round(p, 2) for option, p in kept}


@dataclass
class RequestTrace:
    """One event's trace; see the module docstring."""

    # MODEL_PATH_PREFERENCE: "colab" or "api", "api-only" without Colab.
    preference: str = ""
    request_id: str = ""
    started: float = field(default_factory=time.monotonic)
    walks: list[Walk] = field(default_factory=list)
    route: str = ""
    router_scores: RouterScores | None = None
    router_skip: str = ""
    agent: str = ""
    output: str = ""
    thinking: str = ""
    tts: str = ""
    error: str = ""
    delivered: bool | None = None
    # REPLY or PUSH; PUSH once any send of the event was a push.
    delivery: str = ""
    # Set once routing started or an error reply went out: only then is
    # the summary logged (not for !new, !colab or a blocked message).
    handled: bool = False
    logged: bool = False

    def note_decision(self, decision) -> None:
        self.agent = decision.agent
        self.output = decision.output_format
        self.thinking = "off" if decision.disable_thinking else "on"

    def walks_in(self, phase_name: str) -> list[Walk]:
        return [walk for walk in self.walks if walk.phase == phase_name]


# ── The active trace ─────────────────────────────────────────


def activate(trace: RequestTrace) -> Token:
    return _TRACE.set(trace)


def deactivate(token: Token) -> None:
    _TRACE.reset(token)


def current() -> RequestTrace | None:
    return _TRACE.get()


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Label the walks recorded inside (in this task and the tasks it starts)."""
    token = _PHASE.set(name)
    try:
        yield
    finally:
        _PHASE.reset(token)


def record_walk(attempts: Iterable[Attempt]) -> None:
    trace = _TRACE.get()
    attempts = tuple(attempts)
    if trace is not None and attempts:
        trace.walks.append(Walk(_PHASE.get(), attempts))


def note_route(source: str) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.route = source


def note_router_scores(scores: RouterScores) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.router_scores, trace.router_skip = scores, ""


def note_router_skip(why: str) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.router_skip = why


def note_tts(engine: str) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.tts = engine


def note_delivery(via: str) -> None:
    """A send went out by *via* (REPLY or PUSH); a push is never overwritten."""
    trace = _TRACE.get()
    if trace is not None and trace.delivery != PUSH:
        trace.delivery = via


# ── The summary line ─────────────────────────────────────────


def summary_line(trace: RequestTrace, *, now: float | None = None) -> str:
    """``[<req>] summary pref=… path=… route=… agent=… out=… think=… answer=… …``

    - ``path``: what kind of tier wrote the answer: ``colab``, ``api`` or
      ``none`` (no answer);
    - ``route``: who chose the agent: ``rules``, ``colab:<router model>``,
      the API router tier that answered, or ``fallback``;
    - ``think``: the route's decision; ``answer_think``: how the answering
      tier really thought (``off``, ``on`` or a Colab/OpenRouter level);
    - ``answer``: the tier that answered the last answer walk and its
      seconds (``none`` and ``error=<Kind>`` without one);
    - ``tiers=[…]``: every tier the answer's walks reached, grouped by
      tier: the tiers in the order first reached, each with its outcomes
      in the order they happened (a repeat of the same outcome as
      ``x<n>``, seconds added up), so two passes over the same tiers
      (thinking, then not) do not show which ran between them;
      ``calls=<n>`` when there were several walks (a tool loop);
    - ``router_p``: the Colab router's probabilities, or ``n/a`` (with why
      the router was not used, when it was asked);
    - ``delivery``: ``reply`` or ``push`` (any send of the event that was a
      push), absent when nothing went out.
    """
    now = time.monotonic() if now is None else now
    answer_walks = trace.walks_in(ANSWER)
    answered = _answered(answer_walks)
    parts = [
        f"[{trace.request_id or '-'}] summary",
        f"pref={trace.preference or '-'}",
        f"path={_path(answered)}",
        f"route={_route(trace)}",
        f"agent={trace.agent or '-'}",
        f"out={trace.output or '-'}",
        f"think={trace.thinking or '-'}",
        f"answer={answered.label} {_seconds(answered.seconds)}" if answered else "answer=none",
    ]
    if answered is not None and answered.think:
        parts.append(f"answer_think={answered.think}")
    if trace.error:
        parts.append(f"error={trace.error}")
    prompt = _image_prompt(trace)
    if prompt:
        parts.append(f"prompt={prompt}")
    if trace.tts:
        parts.append(f"tts={trace.tts}")
    if trace.delivery:
        parts.append(f"delivery={trace.delivery}")
    if trace.delivered is False:
        parts.append("sent=no")
    parts.append(f"total={_seconds(max(now - trace.started, 0.0))}")
    parts.append(f"router_p={_router_p(trace)}")
    tiers = f"tiers=[{_tiers(answer_walks)}]"
    if len(answer_walks) > 1:
        tiers += f" calls={len(answer_walks)}"
    parts.append(tiers)
    return " ".join(parts)


def _answered(walks: list[Walk]) -> Attempt | None:
    """The tier that answered the last of *walks*: the one whose response
    the caller used (an earlier walk's was a tool round, an unparseable
    route the repair walk replaced, ...). None when that walk failed."""
    return walks[-1].answered_by if walks else None


def _path(answered: Attempt | None) -> str:
    if answered is None:
        return "none"
    return "colab" if answered.label.split(":", 1)[0] == _COLAB_PROVIDER else "api"


def _route(trace: RequestTrace) -> str:
    if trace.route == ROUTE_ROUTER:
        scores = trace.router_scores
        return f"{_COLAB_PROVIDER}:{scores.model}" if scores else _COLAB_PROVIDER
    if trace.route == ROUTE_LLM:
        answered = _answered(trace.walks_in(ROUTE))
        return answered.label if answered else ROUTE_LLM
    return trace.route or "none"


def _image_prompt(trace: RequestTrace) -> str:
    """The utility tier that wrote the image prompt, ``raw`` when none did."""
    walks = trace.walks_in(IMAGE_PROMPT)
    if not walks:
        return ""
    answered = _answered(walks)
    return answered.label if answered else "raw"


def _router_p(trace: RequestTrace) -> str:
    scores = trace.router_scores
    if scores is None:
        return f"n/a(router {trace.router_skip})" if trace.router_skip else "n/a"
    agent_ok = "ok" if scores.agent_confidence >= scores.gate else "fail"
    output_ok = "ok" if scores.output_confidence >= scores.gate else "fail"
    think = f"think p={scores.think_p:.2f} (on>={scores.think_on:.2f} off<={scores.think_off:.2f})"
    if scores.thinking:
        think += f" -> {scores.thinking}"
    segments = [
        f"agent={_options(scores.agent)} conf={scores.agent_confidence:.2f} gate={scores.gate:.2f} {agent_ok}",
        f"format={_options(scores.output)} conf={scores.output_confidence:.2f} {output_ok}",
        think,
    ]
    if scores.verdict != ACCEPTED:
        segments.append(f"{scores.verdict} -> API router")
    return f"[{' | '.join(segments)}]"


def _options(values: Mapping[str, float]) -> str:
    """``{chat:0.91,web_search:0.05}``; an unscored option (NaN) as ``chat:?``."""
    return "{" + ",".join(f"{option}:{'?' if math.isnan(p) else f'{p:.2f}'}" for option, p in values.items()) + "}"


def _tiers(walks: list[Walk]) -> str:
    # label (first reached first) -> its runs of [outcome, count, seconds
    # or None], in the order they happened.
    runs: dict[str, list[list]] = {}
    for walk in walks:
        for attempt in walk.attempts:
            label_runs = runs.setdefault(attempt.label, [])
            last = label_runs[-1] if label_runs else None
            if last is not None and last[0] == attempt.outcome:
                last[1] += 1
                if attempt.seconds is not None:
                    last[2] = (last[2] or 0.0) + attempt.seconds
            else:
                label_runs.append([attempt.outcome, 1, attempt.seconds])
    return ", ".join(
        f"{label} " + ",".join(_run(*run) for run in label_runs) for label, label_runs in runs.items()
    )


def _run(outcome: str, count: int, seconds: float | None) -> str:
    text = outcome if count == 1 else f"{outcome} x{count}"
    return text if seconds is None else f"{text} {_seconds(seconds)}"


def _seconds(seconds: float | None) -> str:
    return f"{seconds or 0.0:.1f}s"
