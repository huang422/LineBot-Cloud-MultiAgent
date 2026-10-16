"""Routing eval: labelled zh-TW LINE messages through each of the bot's routers.

A human runs this to compare the routers and to tune the Colab router's
gates (plan: Verification 2, routing). From the repo root, with the
.venv312 built from uv.lock (DEPLOY.md, 路由評測):

    .venv312/bin/python scripts/evals/routing_eval.py --dry-run     # the plan; sends nothing
    .venv312/bin/python scripts/evals/routing_eval.py               # the rules only (offline)
    .venv312/bin/python scripts/evals/routing_eval.py --env-file .env --candidates rules,colab
    .venv312/bin/python scripts/evals/routing_eval.py --env-file .env --candidates rules,nvidia,openrouter

Candidates:

- ``rules``: the orchestrator's fast rules (src/agents/routing_rules.py),
  offline. They abstain on what they cannot tell; an abstention goes on to
  the model routers, so it costs coverage, not accuracy.
- ``colab``: the Colab router (SystemOneRouter, /v1/systemone) of a READY
  session, found as scripts/live_gate.py finds it: the session document is
  read from Firestore with this machine's ADC and the bearer is derived
  from COLAB_SHARED_SECRET in the env file. Without COLAB_ENABLED, a
  readable document, a READY session or an inference-capable one it is
  skipped with the reason. Rows go one at a time (the session serves one
  routing request at a time) and every answer's probabilities are kept;
  BREAKER_FAILURES failed rows in a row stop it. The router's answers do
  not keep the session READY (in the bot the controller checks its health
  every minute), so the eval checks it again once READY_HEALTH_TTL_SECONDS
  have passed; a failed check stops it.
- ``nvidia`` / ``openrouter``: the router role's first tier of that
  provider (``nvidia:<model>`` names one), called as the bot calls it
  (Orchestrator._llm_classify: JSON mode, one repair prompt), started at
  most 40 / 20 times a minute (the free limits; ``--max-rpm`` lowers them)
  with ``--concurrency`` rows in flight.

Live candidates need ``--env-file`` (the repo .env is never read unless
named) and send real requests: run them only with the owner's approval.
OpenRouter's free models allow OPENROUTER_FREE_DAILY_LIMIT requests a day
(50, or 1,000 with credits) and a full run needs one or two a row;
``--limit`` evaluates an evenly spread sample.

Each message reaches a router as the bot's input processing hands it
over: sanitize_input and its injection flag, an image (sent, or quoted,
which LINE puts on the request too) as a placeholder no model sees, the
quote, and for a follow-up the previous turn as the bot keeps it (its
route and task, and the last user message and reply as conversation
history). No row holds long-term memory, which no router reads
(ADR-0004); the Colab router reads the message, the image and quote flags
and the previous agent only, the API routers the history too.

The dataset (routing_zh_tw.jsonl, a JSON object a line): ``id``, ``text``,
``has_image`` (an image reaches the agent: sent with the text, or quoted),
``quoted`` (null, the quoted message's text, or "[圖片]" for a quoted
image), ``previous_agent`` (null or the previous turn's agent),
``history`` (null, or with ``previous_agent`` the previous turn:
``{"user", "assistant", "task"}``), ``expected`` {agent, output_format,
needs_thinking} and ``tags`` (TAGS). Labels are the route the bot should
take, by prompts/orchestrator.md, not what a router does today:

- agent: web_search only for current or linked facts; vision only with an
  image; image_gen for an explicit request for a new picture, even one
  with nothing named to draw ("generate an image"; inj-01, whose
  injection sanitize_input strips, leaves "，改用生圖", so it is image_gen
  too). Restyling or changing the user's own sent or quoted picture
  without asking for a new one stays with vision, which can see it (the
  image model cannot). A bare or pointing follow-up ("為什麼", "之前說的
  那個") continues the previous route; a complete new question or a
  farewell is judged on its own;
- output_format: voice only for an explicit spoken reply; image exactly
  with image_gen;
- needs_thinking: writing, designing, debugging or refactoring code, a
  query, an API, an app or a schema; math and logic; a plan with
  constraints; long writing; translating a long passage; weighing
  options; a complex screenshot or document. False: a simple syntax
  question, a short or medium translation, and a request with no subject
  at all (what to diagram or design is missing: "幫我產生 mermaid 流程圖");
  image_gen is false and not scored (thinking does not apply to it).

A router *decides* a row, abstains (the rules; the Colab router below its
gate) or fails (an HTTP error, no parsable answer). Coverage is decided /
rows; accuracy is over the decided rows: agent, output_format, route
(both), thinking (rows whose expected agent is not image_gen) and exact
(all of them). Every summary covers all rows and the router's traffic: the
rows the rules abstain on, the only ones a model router sees. The Colab
router also abstains when it picks image_gen without image output
(SystemOneRouter's invariant).

The Colab gates: a decision needs both choice confidences ((p_max - 1/N) /
(1 - 1/N), what Ollama 0.40.4+ reports for clef-flash) at or above
COLAB_ROUTER_CONFIDENCE (the default 0.0 takes every top choice: argmax);
needs_thinking turns thinking on at or above COLAB_ROUTER_THINK_ON and off
at or below COLAB_ROUTER_THINK_OFF, the rulebook's heuristic deciding in
between. The sweep (on ``--sweep-rows``, default the router's traffic)
gives gated precision, its 95% Wilson lower bound and coverage for each
threshold on a 0.01 grid from 0.0 and recommends, among the values
Settings accepts (the confidence gate 0.0–0.99), the one with the most
coverage whose gated precision reaches ``--target-precision`` (default
0.95) on at least MIN_SUPPORT rows; for the confidence gate that may be
0.0. The confidence gate is chosen first; the two thinking gates are then
swept on the rows it lets through, the only ones whose noul the bot reads.
Each side of needs_thinking is held to the target on its own: THINK_ON by
the rows it turns on, THINK_OFF by the rows it turns off. A few hundred
rows leave wide bounds: the values are provisional until the dataset grows.

A candidate that stops leaves the rest of its rows not run: it is
``partial``, and the run exits 1 and recommends no Colab gates from it
unless ``--allow-partial``.

Output: a Markdown and a JSON report in ``--out-dir`` (default
tmp/routing-eval, which git and gcloud ignore). They hold the dataset's
messages and the routers' choices and probabilities; never keys, bearers,
tunnel hosts or model text. tests/test_routing_eval.py runs everything
over fakes; nothing here runs in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    # Run as ``python scripts/evals/routing_eval.py``: make ``src`` importable.
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import ValidationError  # noqa: E402

from scripts import live_gate  # noqa: E402
from src.agents import routing_rules as rules  # noqa: E402
from src.agents.orchestrator import ERROR_REASON, UNPARSEABLE_REASON, Orchestrator  # noqa: E402
from src.config import Settings  # noqa: E402
from src.models.agent_request import AgentRequest, InputType  # noqa: E402
from src.providers.fallback_chain import FallbackChain, Tier  # noqa: E402
from src.providers.ollama_provider import OllamaChatProvider  # noqa: E402
from src.providers.systemone import RouterGates, SystemOneRouter  # noqa: E402
from src.runtime.composition import build_providers, split_router_tiers  # noqa: E402
from src.runtime.model_roles import (  # noqa: E402
    COLAB,
    RoleConfigError,
    RoleTable,
    TierSpec,
    role_specs,
)
from src.services.colab.endpoint import BREAKER_FAILURES, TUNNEL_PORTS  # noqa: E402
from src.services.colab.state import ColabState  # noqa: E402
from src.services.model_health import DEAD, ModelHealth  # noqa: E402
from src.services.search.page_fetcher import ssrf_safe_transport  # noqa: E402
from src.utils.validators import check_prompt_injection, sanitize_input  # noqa: E402

DEFAULT_DATASET = Path(__file__).with_name("routing_zh_tw.jsonl")
DEFAULT_OUT_DIR = _REPO_ROOT / "tmp" / "routing-eval"

RULES = "rules"
API_PROVIDERS = ("nvidia", "openrouter")
CANDIDATES = (RULES, COLAB, *API_PROVIDERS)

QUOTED_IMAGE = "[圖片]"
TAGS: dict[str, str] = {
    "misroute": "a verified fast-rule misroute (tests/test_routing_rules.py)",
    "rulebook": "a message tests/test_routing_rules.py pins the rules on",
    "prompt-example": "an example in prompts/orchestrator.md",
    "follow-up": "continues the previous turn",
    "quote": "quotes an earlier message",
    "english": "English",
    "mixed": "Chinese mixed with English, code or JSON",
    "injection": "tries to steer the router",
    "url": "carries a URL",
    "multi-line": "spans several lines",
}

# The providers' free limits (ModelHealth budgets the bot keeps too).
FREE_RPM = {"nvidia": 40, "openrouter": 20}
# Starts a little under the limit: ModelHealth counts a rolling minute.
_PACE_MARGIN = 1.05
DEFAULT_CONCURRENCY = 3
MAX_CONCURRENCY = 8
DEFAULT_TARGET_PRECISION = 0.95
# A threshold is recommended only when it lets at least this many rows
# through: 20 right of 20 has a 95% Wilson lower bound of 0.84 (10 of 10:
# 0.72). The report shows the bound of every point.
MIN_SUPPORT = 20
# The normal quantile of the Wilson bound (95%, two-sided).
_WILSON_Z = 1.96
GRID: tuple[float, ...] = tuple(round(step / 100, 2) for step in range(101))
# How long a row waits for its provider to come back (a 429 cool-down or a
# spent budget) before the candidate stops, and how often it asks.
MAX_RATE_WAIT_S = 180.0
_RATE_POLL_S = 5.0
MAX_ATTEMPTS = 3
# The Colab router's gates: env name -> (Settings field, RouterGates field).
GATES: dict[str, tuple[str, str]] = {
    "COLAB_ROUTER_CONFIDENCE": ("colab_router_confidence", "confidence"),
    "COLAB_ROUTER_THINK_ON": ("colab_router_think_on", "think_on"),
    "COLAB_ROUTER_THINK_OFF": ("colab_router_think_off", "think_off"),
}
# The routers only read that an image is there; this one is never sent.
_PLACEHOLDER_IMAGE = "data:image/png;base64,iVBORw0KGgo="
_MISSES_SHOWN = 40
_PROGRESS_EVERY = 20

DECIDED = "decided"
ABSTAINED = "abstained"
FAILED = "failed"
NOT_RUN = "not_run"
# A candidate that left rows not run (its CandidateResult.status).
PARTIAL = "partial"

# The row a request belongs to, for the recording transport.
_ROW: contextvars.ContextVar[str] = contextvars.ContextVar("routing_eval_row", default="")


class DatasetError(ValueError):
    """The dataset cannot be used; the message lists every problem."""


class CandidateSkip(Exception):
    """The candidate cannot run with this configuration; the message says why."""


# ── Dataset ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Expected:
    agent: str
    output_format: str
    needs_thinking: bool


@dataclass(frozen=True)
class Turn:
    """The previous turn as the bot keeps it: the user's message, the reply
    and the task the router gave it."""

    user: str
    assistant: str
    task: str


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    has_image: bool
    quoted: str | None
    previous_agent: str | None
    expected: Expected
    tags: tuple[str, ...]
    history: Turn | None = None

    @property
    def thinking_applies(self) -> bool:
        """Thinking has no effect on image generation, so it is not scored there."""
        return self.expected.agent != "image_gen"


_ROW_KEYS = {"id", "text", "has_image", "quoted", "previous_agent", "history", "expected", "tags"}
_EXPECTED_KEYS = {"agent", "output_format", "needs_thinking"}
_TURN_KEYS = {"user", "assistant", "task"}


def parse_row(data: Any) -> tuple[Row | None, list[str]]:
    """``(row, [])``, or ``(None, problems)`` when *data* is not a well-formed row."""
    if not isinstance(data, dict):
        return None, ["not a JSON object"]
    problems: list[str] = []
    if set(data) != _ROW_KEYS:
        problems.append(f"keys must be {sorted(_ROW_KEYS)}")
        return None, problems
    expected = data["expected"]
    if not isinstance(expected, dict) or set(expected) != _EXPECTED_KEYS:
        return None, [f"expected must have exactly {sorted(_EXPECTED_KEYS)}"]
    if not isinstance(data["id"], str) or not data["id"].strip():
        problems.append("id must be a non-empty string")
    if not isinstance(data["text"], str):
        problems.append("text must be a string")
    if not isinstance(data["has_image"], bool):
        problems.append("has_image must be true or false")
    if data["quoted"] is not None and (not isinstance(data["quoted"], str) or not data["quoted"].strip()):
        problems.append("quoted must be null or a non-empty string")
    if data["previous_agent"] is not None and data["previous_agent"] not in rules.VALID_AGENTS:
        problems.append(f"previous_agent must be null or one of {sorted(rules.VALID_AGENTS)}")
    history = data["history"]
    if history is not None and (
        not isinstance(history, dict)
        or set(history) != _TURN_KEYS
        or not all(isinstance(value, str) and value.strip() for value in history.values())
    ):
        problems.append(f"history must be null or an object of non-empty {sorted(_TURN_KEYS)}")
    elif (history is None) != (data["previous_agent"] is None):
        problems.append("history goes with previous_agent: the bot keeps the previous turn with its route")
    if expected["agent"] not in rules.VALID_AGENTS:
        problems.append(f"expected.agent must be one of {sorted(rules.VALID_AGENTS)}")
    if expected["output_format"] not in rules.VALID_OUTPUTS:
        problems.append(f"expected.output_format must be one of {sorted(rules.VALID_OUTPUTS)}")
    if not isinstance(expected["needs_thinking"], bool):
        problems.append("expected.needs_thinking must be true or false")
    tags = data["tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        problems.append("tags must be a list of strings")
    if problems:
        return None, problems
    row = Row(
        id=data["id"],
        text=data["text"],
        has_image=data["has_image"],
        quoted=data["quoted"],
        previous_agent=data["previous_agent"],
        expected=Expected(**expected),
        tags=tuple(tags),
        history=Turn(**history) if history is not None else None,
    )
    return row, label_problems(row)


def label_problems(row: Row) -> list[str]:
    """What makes *row*'s labels contradict the bot's own invariants."""
    problems: list[str] = []
    expected = row.expected
    if not row.text.strip() and not row.has_image:
        problems.append("an empty text needs an image")
    if (expected.agent == "image_gen") != (expected.output_format == "image"):
        problems.append("image output goes with image_gen, and image_gen always answers with an image")
    if expected.agent == "vision" and not row.has_image:
        problems.append("vision needs an image (has_image)")
    if expected.agent == "image_gen" and expected.needs_thinking:
        problems.append("image_gen never thinks")
    if row.quoted == QUOTED_IMAGE and not row.has_image:
        problems.append("a quoted image reaches the agent: has_image must be true")
    unknown = sorted(set(row.tags) - set(TAGS))
    if unknown:
        problems.append(f"unknown tags {unknown} (known: {', '.join(TAGS)})")
    if len(set(row.tags)) != len(row.tags):
        problems.append("repeated tags")
    return problems


def load_dataset(path: Path) -> list[Row]:
    """Every row of *path*, checked.

    Raises:
        DatasetError: Every problem found, by line.
    """
    rows: list[Row] = []
    problems: list[str] = []
    ids: dict[str, int] = {}
    inputs: dict[tuple, int] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except ValueError as exc:
            problems.append(f"line {number}: not JSON ({exc.msg})")
            continue
        row, row_problems = parse_row(data)
        problems.extend(f"line {number}: {problem}" for problem in row_problems)
        if row is None:
            continue
        if row.id in ids:
            problems.append(f"line {number}: id {row.id!r} repeats line {ids[row.id]}")
        ids.setdefault(row.id, number)
        key = (row.text, row.has_image, row.quoted, row.previous_agent)
        if key in inputs:
            problems.append(f"line {number}: the same message and context as line {inputs[key]}")
        inputs.setdefault(key, number)
        rows.append(row)
    if not rows and not problems:
        problems.append("no rows")
    if problems:
        raise DatasetError(f"{path.name}: " + "; ".join(problems))
    return rows


def select_rows(rows: Sequence[Row], tags: Sequence[str] = (), limit: int | None = None) -> list[Row]:
    """The rows with any of *tags* (all without), then an evenly spread *limit* of them."""
    chosen = [row for row in rows if not tags or set(tags) & set(row.tags)]
    if limit is not None and limit < len(chosen):
        step = len(chosen) / limit
        chosen = [chosen[int(index * step)] for index in range(limit)]
    return chosen


def build_agent_request(row: Row) -> AgentRequest:
    """*row* as the bot's input processing hands it to the orchestrator."""
    text = sanitize_input(row.text)
    request = AgentRequest(text=text, prompt_injection_stripped=check_prompt_injection(row.text))
    if row.has_image:
        request.image_base64 = _PLACEHOLDER_IMAGE
        request.input_type = InputType.IMAGE_TEXT if text else InputType.IMAGE
    if row.quoted is not None:
        request.quoted_message_id = f"quoted-{row.id}"
        if row.quoted == QUOTED_IMAGE:
            request.quoted_message_type = "image"
            request.quoted_image_base64 = _PLACEHOLDER_IMAGE
        else:
            request.quoted_message_type = "text"
            request.quoted_text = row.quoted
    if row.previous_agent:
        request.previous_agent = row.previous_agent
        request.previous_output_format = "image" if row.previous_agent == "image_gen" else "text"
    if row.history is not None:
        request.conversation_history = [
            {"role": "user", "content": row.history.user},
            {"role": "assistant", "content": row.history.assistant},
        ]
        request.previous_task_description = row.history.task
    return request


# ── Outcomes ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Outcome:
    """What one candidate did with one row."""

    row_id: str
    status: str  # decided | abstained | failed | not_run
    agent: str = ""
    output_format: str = ""
    needs_thinking: bool | None = None
    latency_s: float | None = None
    requests: int = 0
    detail: str = ""
    # The Colab router's answer: the choice, confidence and probabilities
    # of each question and needs_thinking's noul; None for the others.
    scores: dict[str, Any] | None = None


@dataclass
class CandidateResult:
    name: str
    label: str = ""
    status: str = "ok"  # ok | partial | skipped | failed
    reason: str = ""
    outcomes: list[Outcome] = field(default_factory=list)


def _ran(name: str, label: str, outcomes: list[Outcome], reason: str) -> CandidateResult:
    """A candidate that ran: failed when it decided no row, partial when it
    left rows not run (*reason*, or the first row's, says why)."""
    not_run = [outcome for outcome in outcomes if outcome.status == NOT_RUN]
    status = "ok"
    if outcomes and all(outcome.status != DECIDED for outcome in outcomes):
        status = "failed"
    elif not_run:
        status = PARTIAL
        reason = f"{len(not_run)} of {len(outcomes)} rows not run: {reason or not_run[0].detail}"
    return CandidateResult(name, label, status=status, reason=reason, outcomes=outcomes)


def route_correct(row: Row, outcome: Outcome) -> bool:
    return (outcome.agent, outcome.output_format) == (row.expected.agent, row.expected.output_format)


def thinking_correct(row: Row, outcome: Outcome) -> bool:
    return not row.thinking_applies or outcome.needs_thinking == row.expected.needs_thinking


def _decided(agent: str, output: str, disable_thinking: bool, row: Row) -> tuple[str, str, bool]:
    """A router's decision after the orchestrator's own invariant: vision
    without an image to look at becomes chat."""
    if agent == "vision" and not row.has_image:
        agent = "chat"
    return agent, output, not disable_thinking


# ── Recording transport and pacing ───────────────────────────


@dataclass(frozen=True)
class Exchange:
    """One request a router's client sent for a row, and what came back."""

    row_id: str
    status: int  # 0: no answer (a transport error)
    data: Any
    text: str


_DROPPED_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


class RecordingTransport(httpx.AsyncBaseTransport):
    """Passes the bot's requests on unchanged and keeps each answer by row."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner
        self.exchanges: list[Exchange] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        row_id = _ROW.get()
        try:
            response = await self._inner.handle_async_request(request)
            try:
                content = await response.aread()
            finally:
                await response.aclose()
        except Exception as exc:
            # Recorded as status 0; the bot's code sees the same exception.
            self.exchanges.append(Exchange(row_id, 0, None, f"{type(exc).__name__}: {exc}"))
            raise
        try:
            data = json.loads(content) if content else None
        except ValueError:
            data = None
        self.exchanges.append(
            Exchange(row_id, response.status_code, data, content[:300].decode("utf-8", "replace"))
        )
        return httpx.Response(
            response.status_code,
            headers=[
                (name, value)
                for name, value in response.headers.items()
                if name.lower() not in _DROPPED_RESPONSE_HEADERS
            ],
            content=content,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()

    def for_row(self, row_id: str) -> list[Exchange]:
        return [exchange for exchange in self.exchanges if exchange.row_id == row_id]


class Pacer:
    """Starts calls at most *rpm* a minute, whatever the concurrency."""

    def __init__(
        self,
        rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.interval = 60.0 / rpm * _PACE_MARGIN
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._next: float | None = None

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock()
            start = now if self._next is None else max(now, self._next)
            if start > now:
                await self._sleep(start - now)
            self._next = start + self.interval


class PacedChain:
    """The router's FallbackChain, each call (a request) started by *pacer*."""

    def __init__(self, chain: FallbackChain, pacer: Pacer) -> None:
        self._chain = chain
        self._pacer = pacer

    async def generate(self, **kwargs: Any) -> Any:
        await self._pacer.wait()
        return await self._chain.generate(**kwargs)


@dataclass
class Runtime:
    """What a run talks to and waits with; tests pass fakes."""

    api_transport: httpx.AsyncBaseTransport | None = None
    colab_transport: httpx.AsyncBaseTransport | None = None
    colab_loader: Callable[[Settings], ColabState] | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    progress: Callable[[str], None] = lambda line: None


@dataclass
class _Stop:
    """Set once a candidate cannot go on; the rows left are not run."""

    reason: str = ""


def _progress(runtime: Runtime, name: str, done: int, total: int) -> None:
    if done == total or done % _PROGRESS_EVERY == 0:
        runtime.progress(f"[routing-eval] {name}: {done}/{total} rows")


# ── Candidates ───────────────────────────────────────────────


def run_rules(rows: Sequence[Row]) -> CandidateResult:
    """The orchestrator's fast rules: offline, a decision or an abstention per row."""
    orchestrator = Orchestrator(Settings(_env_file=None), None, targets=[])
    outcomes = []
    for row in rows:
        decision = orchestrator._apply_fast_rules(build_agent_request(row))
        if decision is None:
            outcomes.append(Outcome(row.id, ABSTAINED))
            continue
        agent, output, thinking = _decided(
            decision.agent, decision.output_format, decision.disable_thinking, row
        )
        outcomes.append(Outcome(row.id, DECIDED, agent, output, thinking))
    return CandidateResult(RULES, "fast rules (src/agents/routing_rules.py)", outcomes=outcomes)


def parse_candidates(value: str) -> list[tuple[str, str]]:
    """``[(name, model)]`` from ``rules,colab,nvidia:<model>,...``.

    Raises:
        ValueError: An unknown or repeated candidate.
    """
    chosen: list[tuple[str, str]] = []
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        name, _, model = item.partition(":")
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name!r} (choose from {', '.join(CANDIDATES)})")
        if model and name not in API_PROVIDERS:
            raise ValueError(f"only {' and '.join(API_PROVIDERS)} take a model ({item!r})")
        if any(existing == (name, model) for existing in chosen):
            raise ValueError(f"{item!r} is listed twice")
        chosen.append((name, model))
    if not chosen:
        raise ValueError("no candidates")
    return chosen


def candidate_id(name: str, model: str) -> str:
    return f"{name}:{model}" if model else name


def _is_candidate_tier(spec: TierSpec, provider: str, model: str) -> bool:
    """*spec* is the router tier a candidate names: its provider's first, or *model*."""
    return spec.provider == provider and (not model or spec.model == model)


def _no_tier(provider: str, model: str) -> CandidateSkip:
    wanted = f"{provider}:{model}" if model else f"a {provider} tier"
    return CandidateSkip(f"the router role has no {wanted} (MODEL_ROLES)")


def router_spec(settings: Settings, provider: str, model: str = "") -> TierSpec:
    """The configured router tier a candidate names (no providers needed)."""
    for spec in role_specs(settings)["router"]:
        if _is_candidate_tier(spec, provider, model):
            return spec
    raise _no_tier(provider, model)


def _api_tier(table: RoleTable, provider: str, model: str) -> tuple[str, Tier]:
    """The bound router tier a candidate names; CandidateSkip when it is off."""
    for row in table.rows("router"):
        if _is_candidate_tier(row.spec, provider, model):
            if row.tier is None:
                raise CandidateSkip(f"{row.label} is off in the router role ({row.disabled_reason})")
            return row.label, row.tier
    raise _no_tier(provider, model)


def _failure_detail(exchanges: Sequence[Exchange], reason: str, blocked: Any) -> str:
    if not exchanges:
        return f"no request sent ({blocked.reason})" if blocked is not None else "no request sent"
    last = exchanges[-1]
    if last.status == 0:
        return f"request failed: {last.text}"
    if last.status != 200:
        return f"HTTP {last.status}: {last.text}"
    return f"{reason} after {len(exchanges)} request(s)"


async def run_api(
    rows: Sequence[Row],
    provider: str,
    model: str,
    settings: Settings,
    runtime: Runtime,
    *,
    rpm: int,
    concurrency: int,
    redact: Callable[[str], str],
) -> CandidateResult:
    """The router role's *provider* tier on every row, as Orchestrator._llm_classify calls it."""
    name = candidate_id(provider, model)
    health = ModelHealth(openrouter_free_daily_limit=settings.openrouter_free_daily_limit, clock=runtime.clock)
    recorder = RecordingTransport(runtime.api_transport or httpx.AsyncHTTPTransport())
    providers = build_providers(settings, health, transport=recorder)
    try:
        try:
            table = RoleTable.from_settings(settings, providers)
        except RoleConfigError as exc:
            return CandidateResult(name, status="failed", reason=redact(f"role table: {exc}"))
        try:
            label, tier = _api_tier(table, provider, model)
        except CandidateSkip as exc:
            return CandidateResult(name, status="skipped", reason=str(exc))
        pacer = Pacer(rpm, clock=runtime.clock, sleep=runtime.sleep)
        orchestrator = Orchestrator(settings, PacedChain(FallbackChain(health), pacer), targets=[tier])
        gate = asyncio.Semaphore(concurrency)
        stop = _Stop()
        done = 0

        async def one(row: Row) -> Outcome:
            nonlocal done
            async with gate:
                outcome = await _api_row(orchestrator, row, recorder, health, tier, runtime, stop, redact)
            done += 1
            _progress(runtime, name, done, len(rows))
            return outcome

        outcomes = list(await asyncio.gather(*(one(row) for row in rows)))
    finally:
        for client in providers.values():
            await client.close()
    return _ran(name, f"{label} (router role)", outcomes, stop.reason)


async def _api_row(
    orchestrator: Orchestrator,
    row: Row,
    recorder: RecordingTransport,
    health: ModelHealth,
    tier: Tier,
    runtime: Runtime,
    stop: _Stop,
    redact: Callable[[str], str],
) -> Outcome:
    provider = tier.provider.name
    attempt = 1
    while True:
        if stop.reason:
            return Outcome(row.id, NOT_RUN, detail=stop.reason)
        sent = len(recorder.for_row(row.id))
        token = _ROW.set(row.id)
        started = runtime.clock()
        try:
            decision = await orchestrator._llm_classify(build_agent_request(row))
        finally:
            _ROW.reset(token)
        elapsed = runtime.clock() - started
        exchanges = recorder.for_row(row.id)
        if decision.reasoning not in (UNPARSEABLE_REASON, ERROR_REASON):
            agent, output, thinking = _decided(
                decision.agent, decision.output_format, decision.disable_thinking, row
            )
            return Outcome(row.id, DECIDED, agent, output, thinking, elapsed, len(exchanges))
        blocked = health.unavailable(provider, tier.model)
        failure = Outcome(
            row.id,
            FAILED,
            latency_s=elapsed,
            requests=len(exchanges),
            detail=redact(_failure_detail(exchanges[sent:], decision.reasoning, blocked)),
        )
        if blocked is None or attempt == MAX_ATTEMPTS:
            return failure
        if blocked.kind == DEAD:
            stop.reason = redact(f"{provider}:{tier.model} is out of rotation ({blocked.reason})")
            return failure
        # A 429 cool-down, a spent budget or a run of transient failures:
        # wait for it to pass, then try the row again.
        blocked = await _wait_until_available(health, provider, tier.model, runtime)
        if blocked is not None:
            stop.reason = redact(
                f"{provider}:{tier.model} still unavailable after {MAX_RATE_WAIT_S:g}s ({blocked.reason})"
            )
            return failure
        attempt += 1


async def _wait_until_available(health: ModelHealth, provider: str, model: str, runtime: Runtime) -> Any:
    """None once *model* can take a request again; why not after MAX_RATE_WAIT_S."""
    deadline = runtime.clock() + MAX_RATE_WAIT_S
    while (blocked := health.unavailable(provider, model)) is not None:
        if runtime.clock() >= deadline:
            return blocked
        await runtime.sleep(_RATE_POLL_S)
    return None


def colab_scores(data: Any) -> dict[str, Any] | None:
    """The raw /v1/systemone answer, per question; None when it is not one."""
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        return None
    scores: dict[str, Any] = {}
    for name, allowed in (("agent", rules.VALID_AGENTS), ("output_format", rules.VALID_OUTPUTS)):
        answer = answers.get(name)
        if not isinstance(answer, dict) or answer.get("choice") not in allowed:
            return None
        confidence, probabilities = answer.get("confidence"), answer.get("probabilities")
        if not _is_probability(confidence) or not isinstance(probabilities, dict):
            return None
        scores[name] = answer["choice"]
        scores[f"{name}_confidence"] = float(confidence)
        scores[f"{name}_probabilities"] = {
            str(key): float(value) for key, value in probabilities.items() if _is_probability(value)
        }
    thinking = answers.get("needs_thinking")
    noul = thinking.get("noul") if isinstance(thinking, dict) else None
    if not _is_probability(noul):
        return None
    scores["noul"] = float(noul)
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    scores["input_tokens"] = usage.get("input_tokens")
    return scores


def _is_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def gate_value(scores: Mapping[str, Any]) -> float:
    """What COLAB_ROUTER_CONFIDENCE is compared with: the lower choice confidence."""
    return min(scores["agent_confidence"], scores["output_format_confidence"])


async def run_colab(
    rows: Sequence[Row],
    ctx: live_gate.GateContext,
    runtime: Runtime,
    *,
    redact: Callable[[str], str],
) -> CandidateResult:
    """The Colab router of the READY session on every row, one at a time.

    The router is asked with its confidence gate at zero, so it decides
    every row it gets an answer for; the gates are applied afterwards
    (:func:`gated`). It gets no ModelHealth: a cool-down would skip rows,
    and the eval asks every row. BREAKER_FAILURES failed rows in a row
    stop it, as they would stop the bot's Colab chat.
    """
    try:
        # A worker thread: the session document is read with its own event
        # loop (live_gate.load_colab_state), which cannot run inside this one.
        state = await asyncio.to_thread(live_gate.colab_session, ctx)
    except live_gate.GateSkip as exc:
        return CandidateResult(COLAB, status="skipped", reason=redact(str(exc)))
    except live_gate.GateError as exc:
        return CandidateResult(COLAB, status="failed", reason=redact(str(exc)))
    settings = ctx.bot_settings()
    if settings.colab_dry_run:
        return CandidateResult(
            COLAB, status="skipped", reason="COLAB_DRY_RUN=true: the VM's stub Ollama answers no inference"
        )
    recorder = RecordingTransport(runtime.colab_transport or ssrf_safe_transport(allowed_ports=TUNNEL_PORTS))
    endpoint = live_gate.colab_endpoint(settings, state, recorder)
    try:
        try:
            table = RoleTable.from_settings(
                settings,
                {
                    COLAB: OllamaChatProvider(
                        endpoint,
                        default_think=settings.colab_chat_think,
                        thinking_budget=settings.nvidia_thinking_budget,
                    )
                },
            )
        except RoleConfigError as exc:
            return CandidateResult(COLAB, status="failed", reason=redact(f"role table: {exc}"))
        _, tier, _ = split_router_tiers(table)
        if tier is None:
            reasons = "; ".join(
                f"{row.label}: {row.disabled_reason}"
                for row in table.rows("router")
                if row.spec.provider == COLAB and not row.enabled
            )
            return CandidateResult(
                COLAB, status="skipped", reason=f"the router role has no enabled Colab tier ({reasons or 'MODEL_ROLES'})"
            )
        label = f"{COLAB}:{tier.model} (router role)"
        if not await endpoint.ready():
            return CandidateResult(COLAB, label, status="skipped", reason="the session is not READY")
        off = _session_off_reason(endpoint, tier.model)
        if off:
            return CandidateResult(COLAB, label, status="skipped", reason=redact(off))
        router = SystemOneRouter(
            endpoint, tier, gates=replace(RouterGates.from_settings(settings), confidence=0.0)
        )
        outcomes: list[Outcome] = []
        stop = _Stop()
        failed_in_a_row = 0
        for done, row in enumerate(rows, start=1):
            if not stop.reason:
                stop.reason = redact(await _still_serving(endpoint, tier.model))
            if stop.reason:
                outcomes.append(Outcome(row.id, NOT_RUN, detail=stop.reason))
                continue
            outcome = await _colab_row(router, row, recorder, runtime, redact)
            outcomes.append(outcome)
            failed_in_a_row = failed_in_a_row + 1 if outcome.status == FAILED else 0
            if failed_in_a_row >= BREAKER_FAILURES:
                stop.reason = f"{failed_in_a_row} rows in a row failed (last: {outcome.detail})"
            _progress(runtime, COLAB, done, len(rows))
    finally:
        await endpoint.close()
    return _ran(COLAB, label, outcomes, stop.reason)


async def _still_serving(endpoint: Any, model: str) -> str:
    """"" while the session serves its router; why not otherwise.

    READY lasts READY_HEALTH_TTL_SECONDS past the last health check, and the
    router's answers do not extend it (in the bot the controller checks every
    minute), so once it has passed the session's health is checked again.
    """
    if not endpoint.is_ready() and not await endpoint.check_health():
        return f"the session failed its health check ({endpoint.last_error})"
    return _session_off_reason(endpoint, model)


def _session_off_reason(endpoint: Any, model: str) -> str:
    """Why the session would never call its router, or ""."""
    spilled = endpoint.off_gpu().get(model)
    if spilled is not None:
        where = "was not wholly in VRAM" if spilled[1] else "was not loaded"
        return f"the session's {model} {where}, so the bot does not use it; see colab/README.md (VRAM)"
    return ""


async def _colab_row(
    router: SystemOneRouter,
    row: Row,
    recorder: RecordingTransport,
    runtime: Runtime,
    redact: Callable[[str], str],
) -> Outcome:
    token = _ROW.set(row.id)
    started = runtime.clock()
    try:
        decision = await router.route(build_agent_request(row))
    finally:
        _ROW.reset(token)
    elapsed = runtime.clock() - started
    exchanges = recorder.for_row(row.id)
    if not exchanges:
        return Outcome(row.id, NOT_RUN, detail="the router sent no request (the session is no longer READY)")
    last = exchanges[-1]
    scores = colab_scores(last.data) if last.status == 200 else None
    if decision is None and scores is not None:
        # A well-formed answer the router would not use at any gate: image_gen
        # without image output (SystemOneRouter's invariant).
        return Outcome(
            row.id, ABSTAINED, latency_s=elapsed, requests=len(exchanges),
            detail="image_gen without image output", scores=scores,
        )
    if decision is None or scores is None:
        if last.status == 0:
            detail = f"request failed: {last.text}"
        elif last.status != 200:
            detail = f"HTTP {last.status}: {last.text}"
        else:
            detail = "malformed answer"
        return Outcome(row.id, FAILED, latency_s=elapsed, requests=len(exchanges), detail=redact(detail))
    agent, output, _ = _decided(decision.agent, decision.output_format, decision.disable_thinking, row)
    return Outcome(row.id, DECIDED, agent, output, None, elapsed, len(exchanges), scores=scores)


# ── Colab gates and the sweep ────────────────────────────────


def noul_thinking(row: Row, outcome: Outcome, gates: RouterGates) -> bool:
    """needs_thinking as the Colab router decides it at *gates*."""
    if outcome.agent == "image_gen":
        return False
    noul = outcome.scores["noul"]
    if noul >= gates.think_on:
        return True
    if noul <= gates.think_off:
        return False
    flat = " ".join(sanitize_input(row.text).split())
    return not rules.should_disable_thinking(flat, has_image=row.has_image)


def gated(rows: Sequence[Row], outcomes: Sequence[Outcome], gates: RouterGates) -> list[Outcome]:
    """The Colab router's outcomes for *rows* as the bot would act on them at *gates*."""
    by_id = {row.id: row for row in rows}
    result = []
    for outcome in outcomes:
        row = by_id.get(outcome.row_id)
        if row is None:
            continue
        if outcome.status != DECIDED or outcome.scores is None:
            result.append(outcome)
        elif gate_value(outcome.scores) < gates.confidence:
            result.append(replace(outcome, status=ABSTAINED, needs_thinking=None))
        else:
            result.append(replace(outcome, needs_thinking=noul_thinking(row, outcome, gates)))
    return result


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    passed: int  # rows the threshold lets through
    correct: int  # of those, the ones it got right
    total: int  # rows swept

    @property
    def coverage(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def precision(self) -> float | None:
        return self.correct / self.passed if self.passed else None

    @property
    def precision_lower(self) -> float | None:
        """The 95% Wilson lower bound of :attr:`precision`."""
        return wilson_lower(self.correct, self.passed)


def wilson_lower(correct: int, total: int, z: float = _WILSON_Z) -> float | None:
    """The Wilson score interval's lower bound for *correct* of *total*; None for no rows."""
    if not total:
        return None
    share = correct / total
    centre = share + z * z / (2 * total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    return max(0.0, (centre - spread) / (1 + z * z / total))


def sweep(
    points: Sequence[tuple[float, bool]],
    passes: Callable[[float, float], bool],
    grid: Sequence[float] = GRID,
) -> list[CurvePoint]:
    """For each threshold on *grid*: the (value, right) *points* it lets through."""
    curve = []
    for threshold in grid:
        through = [right for value, right in points if passes(value, threshold)]
        curve.append(CurvePoint(threshold, len(through), sum(through), len(points)))
    return curve


def confidence_curve(points: Sequence[tuple[float, bool]]) -> list[CurvePoint]:
    """*points*: (gate value, route right); a row passes at gate value >= threshold."""
    return sweep(points, lambda value, threshold: value >= threshold)


def think_on_curve(points: Sequence[tuple[float, bool]]) -> list[CurvePoint]:
    """*points*: (noul, expected thinking); thinking turns on at noul >= threshold."""
    return sweep(points, lambda value, threshold: value >= threshold)


def think_off_curve(points: Sequence[tuple[float, bool]]) -> list[CurvePoint]:
    """*points*: (noul, expected thinking); thinking turns off at noul <= threshold."""
    return sweep([(noul, not thinks) for noul, thinks in points], lambda value, threshold: value <= threshold)


def pick(
    curve: Sequence[CurvePoint],
    target: float,
    allowed: Callable[[float], bool],
    *,
    stricter: Callable[[float], float],
) -> CurvePoint | None:
    """The allowed threshold with the most coverage whose precision reaches
    *target* on at least MIN_SUPPORT rows; ties go to the more precise, then
    the *stricter* one. None when no threshold qualifies."""
    good = [
        point
        for point in curve
        if allowed(point.threshold) and point.passed >= MIN_SUPPORT and point.precision >= target
    ]
    if not good:
        return None
    return max(good, key=lambda point: (point.passed, point.precision, stricter(point.threshold)))


def settings_accept(gates: RouterGates) -> bool:
    """True when Settings takes *gates* as COLAB_ROUTER_* values."""
    values = {settings_field: getattr(gates, attr) for settings_field, attr in GATES.values()}
    try:
        Settings(_env_file=None, **values)
    except ValidationError:
        return False
    return True


def _allowed(base: RouterGates, attr: str) -> Callable[[float], bool]:
    return lambda value: settings_accept(replace(base, **{attr: value}))


def recommend(
    confidence_points: Sequence[tuple[float, bool]],
    thinking_points: Sequence[tuple[float, float, bool]],
    current: RouterGates,
    target: float,
) -> dict[str, Any]:
    """The gates with the most coverage at *target* gated precision, and the curves.

    *confidence_points* are (gate value, route right); *thinking_points*
    (gate value, noul, expected thinking). The confidence gate is chosen
    first, and the thinking gates are swept on the rows it lets through:
    the bot reads the noul of no other row. A gate no threshold qualifies
    for gets its strictest allowed value (the router then rarely decides,
    or the rulebook decides thinking).
    """
    chosen: dict[str, CurvePoint | None] = {}
    confidence_curve_points = confidence_curve(confidence_points)
    chosen["COLAB_ROUTER_CONFIDENCE"] = pick(
        confidence_curve_points, target, _allowed(current, "confidence"), stricter=lambda t: t
    )
    confidence = _value_or_strictest(chosen["COLAB_ROUTER_CONFIDENCE"], current, "confidence", max)
    noul_points = [(noul, thinks) for gate, noul, thinks in thinking_points if gate >= confidence]
    curves = {
        "COLAB_ROUTER_CONFIDENCE": confidence_curve_points,
        "COLAB_ROUTER_THINK_ON": think_on_curve(noul_points),
        "COLAB_ROUTER_THINK_OFF": think_off_curve(noul_points),
    }
    chosen["COLAB_ROUTER_THINK_ON"] = pick(
        curves["COLAB_ROUTER_THINK_ON"], target, _allowed(current, "think_on"), stricter=lambda t: t
    )
    think_on = _value_or_strictest(chosen["COLAB_ROUTER_THINK_ON"], current, "think_on", max)
    with_on = replace(current, confidence=confidence, think_on=think_on)
    chosen["COLAB_ROUTER_THINK_OFF"] = pick(
        curves["COLAB_ROUTER_THINK_OFF"], target, _allowed(with_on, "think_off"), stricter=lambda t: -t
    )
    think_off = _value_or_strictest(chosen["COLAB_ROUTER_THINK_OFF"], with_on, "think_off", min)
    return {
        "gates": RouterGates(confidence=confidence, think_on=think_on, think_off=think_off),
        "meets_target": {name: point is not None for name, point in chosen.items()},
        "points": chosen,
        "curves": curves,
        "noul_points": noul_points,
    }


def _value_or_strictest(
    point: CurvePoint | None, base: RouterGates, attr: str, strictest: Callable[..., float]
) -> float:
    if point is not None:
        return point.threshold
    allowed = [value for value in GRID if settings_accept(replace(base, **{attr: value}))]
    return strictest(allowed) if allowed else getattr(base, attr)


# ── Report ───────────────────────────────────────────────────


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: Sequence[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(share * len(ordered)))], 3)


def summarize(rows: Sequence[Row], outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Coverage and accuracy of *outcomes* over *rows* (see the module docstring)."""
    by_id = {outcome.row_id: outcome for outcome in outcomes}
    pairs = [(row, by_id[row.id]) for row in rows if row.id in by_id]
    counts = Counter(outcome.status for _, outcome in pairs)
    decided = [(row, outcome) for row, outcome in pairs if outcome.status == DECIDED]
    thinking = [(row, outcome) for row, outcome in decided if row.thinking_applies]
    latencies = [outcome.latency_s for _, outcome in pairs if outcome.latency_s is not None]
    return {
        "rows": len(pairs),
        DECIDED: counts[DECIDED],
        ABSTAINED: counts[ABSTAINED],
        FAILED: counts[FAILED],
        NOT_RUN: counts[NOT_RUN],
        "coverage": _ratio(len(decided), len(pairs)),
        "agent": _ratio(sum(o.agent == r.expected.agent for r, o in decided), len(decided)),
        "output_format": _ratio(
            sum(o.output_format == r.expected.output_format for r, o in decided), len(decided)
        ),
        "route": _ratio(sum(route_correct(r, o) for r, o in decided), len(decided)),
        "thinking": _ratio(sum(thinking_correct(r, o) for r, o in thinking), len(thinking)),
        "exact": _ratio(
            sum(route_correct(r, o) and thinking_correct(r, o) for r, o in decided), len(decided)
        ),
        "latency_p50_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
    }


def confusion(rows: Sequence[Row], outcomes: Sequence[Outcome], field: str = "agent") -> dict[str, dict[str, int]]:
    """Expected *field* (agent or output_format) -> what the candidate did
    (a value, or its status)."""
    values = rules.VALID_AGENTS if field == "agent" else rules.VALID_OUTPUTS
    by_id = {outcome.row_id: outcome for outcome in outcomes}
    table: dict[str, Counter] = {value: Counter() for value in sorted(values)}
    for row in rows:
        outcome = by_id.get(row.id)
        if outcome is not None:
            got = getattr(outcome, field) if outcome.status == DECIDED else outcome.status
            table[getattr(row.expected, field)][got] += 1
    return {value: dict(counter) for value, counter in table.items()}


def by_tag(rows: Sequence[Row], outcomes: Sequence[Outcome]) -> dict[str, dict[str, Any]]:
    by_id = {outcome.row_id: outcome for outcome in outcomes}
    result = {}
    for tag in TAGS:
        tagged = [row for row in rows if tag in row.tags and row.id in by_id]
        decided = [row for row in tagged if by_id[row.id].status == DECIDED]
        result[tag] = {
            "rows": len(tagged),
            DECIDED: len(decided),
            "route": _ratio(sum(route_correct(row, by_id[row.id]) for row in decided), len(decided)),
        }
    return result


def misses(rows: Sequence[Row], outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    """The decided rows a candidate got wrong (route or thinking), and its failures."""
    by_id = {outcome.row_id: outcome for outcome in outcomes}
    found = []
    for row in rows:
        outcome = by_id.get(row.id)
        if outcome is None or outcome.status in (ABSTAINED, NOT_RUN):
            continue
        if outcome.status == DECIDED and route_correct(row, outcome) and thinking_correct(row, outcome):
            continue
        found.append(
            {
                "id": row.id,
                "text": rules.clip_text(row.text, 40),
                "expected": _label(row.expected.agent, row.expected.output_format, row.expected.needs_thinking),
                "got": (
                    _label(outcome.agent, outcome.output_format, outcome.needs_thinking)
                    if outcome.status == DECIDED
                    else f"{outcome.status}: {outcome.detail}"
                ),
            }
        )
    return found


def _label(agent: str, output: str, thinking: bool | None) -> str:
    think = "" if thinking is None else f"/{'think' if thinking else 'no-think'}"
    return f"{agent}/{output}{think}"


def _gates_view(gates: RouterGates) -> dict[str, float]:
    return {name: getattr(gates, attr) for name, (_, attr) in GATES.items()}


def colab_section(
    rows: Sequence[Row],
    outcomes: Sequence[Outcome],
    current: RouterGates,
    target: float,
    *,
    withheld: str = "",
) -> dict[str, Any]:
    """The Colab router at the current gates, the sweep over *rows* and the
    recommendation; with *withheld* (why not) neither the sweep nor a
    recommendation."""
    by_id = {outcome.row_id: outcome for outcome in outcomes}
    answered = [
        (row, by_id[row.id])
        for row in rows
        if row.id in by_id and by_id[row.id].status == DECIDED and by_id[row.id].scores is not None
    ]
    confidence_points = [(gate_value(o.scores), route_correct(r, o)) for r, o in answered]
    thinking_points = [
        (gate_value(o.scores), o.scores["noul"], r.expected.needs_thinking)
        for r, o in answered
        if r.thinking_applies and o.agent != "image_gen"
    ]
    section = {
        "rows": len(rows),
        "answered": len(answered),
        "ungated": summarize(rows, outcomes_with_thinking(rows, outcomes, current)),
        "current": {"gates": _gates_view(current), "summary": summarize(rows, gated(rows, outcomes, current))},
    }
    if withheld:
        return {**section, "withheld": withheld}
    result = recommend(confidence_points, thinking_points, current, target)
    recommended = result["gates"]
    # The rows past the recommended confidence gate, whose noul the bot reads.
    noul_points = result["noul_points"]
    return {
        **section,
        "recommended": {
            "gates": _gates_view(recommended),
            "meets_target": result["meets_target"],
            "chosen": {
                name: (None if point is None else _point_view(point)) for name, point in result["points"].items()
            },
            "summary": summarize(rows, gated(rows, outcomes, recommended)),
            "noul_rows": len(noul_points),
            "noul_decides": _ratio(
                sum(not recommended.think_off < noul < recommended.think_on for noul, _ in noul_points),
                len(noul_points),
            ),
        },
        "curves": {name: [_point_view(point) for point in curve] for name, curve in result["curves"].items()},
    }


def outcomes_with_thinking(rows: Sequence[Row], outcomes: Sequence[Outcome], gates: RouterGates) -> list[Outcome]:
    """Every answer used (no confidence gate), thinking decided at *gates*."""
    return gated(rows, outcomes, replace(gates, confidence=0.0))


def _point_view(point: CurvePoint) -> dict[str, Any]:
    precision, lower = point.precision, point.precision_lower
    return {
        "threshold": point.threshold,
        "passed": point.passed,
        "total": point.total,
        "coverage": round(point.coverage, 4),
        "precision": None if precision is None else round(precision, 4),
        "precision_lower_95": None if lower is None else round(lower, 4),
    }


def dataset_view(path: Path, rows: Sequence[Row], router_ids: set[str]) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""
    return {
        "path": _display_path(path),
        "sha256": digest,
        "rows": len(rows),
        "router_rows": len(router_ids),
        "agent": dict(Counter(row.expected.agent for row in rows)),
        "output_format": dict(Counter(row.expected.output_format for row in rows)),
        "needs_thinking": dict(Counter(str(row.expected.needs_thinking).lower() for row in rows)),
        "has_image": sum(row.has_image for row in rows),
        "quoted": sum(row.quoted is not None for row in rows),
        "previous_agent": sum(row.previous_agent is not None for row in rows),
        "tags": {tag: sum(tag in row.tags for row in rows) for tag in TAGS},
    }


def _display_path(path: Path) -> str:
    with contextlib.suppress(ValueError):
        return str(path.resolve().relative_to(_REPO_ROOT))
    return path.name


def build_report(
    rows: Sequence[Row],
    dataset_path: Path,
    results: Sequence[CandidateResult],
    router_ids: set[str],
    *,
    current: RouterGates,
    target: float,
    sweep_rows: str,
    generated_at: str,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Everything the run found, as JSON-ready data (the Markdown is rendered from it).

    A partial Colab run recommends no gates unless *allow_partial*.
    """
    router_rows = [row for row in rows if row.id in router_ids]
    sweep_set = router_rows if sweep_rows == "router" else list(rows)
    candidates = []
    for result in results:
        entry: dict[str, Any] = {
            "name": result.name,
            "label": result.label,
            "status": result.status,
            "reason": result.reason,
        }
        if result.outcomes:
            outcomes = result.outcomes
            if result.name == COLAB:
                withheld = ""
                if result.status == PARTIAL and not allow_partial:
                    withheld = f"{result.reason}. Rerun it, or pass --allow-partial to sweep the rows that ran"
                entry["colab"] = colab_section(sweep_set, outcomes, current, target, withheld=withheld)
                outcomes = gated(rows, outcomes, current)
            entry["summary"] = {"all": summarize(rows, outcomes), "router": summarize(router_rows, outcomes)}
            entry["confusion"] = confusion(rows, outcomes)
            entry["format_confusion"] = confusion(rows, outcomes, "output_format")
            entry["tags"] = by_tag(rows, outcomes)
            entry["misses"] = misses(rows, outcomes)
            entry["outcomes"] = [asdict(outcome) for outcome in result.outcomes]
        candidates.append(entry)
    return {
        "generated_at": generated_at,
        "dataset": dataset_view(dataset_path, rows, router_ids),
        "target_precision": target,
        "min_support": MIN_SUPPORT,
        "sweep_rows": sweep_rows,
        "current_gates": _gates_view(current),
        "candidates": candidates,
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


_SUMMARY_HEADER = (
    "| candidate | rows | decided | abstained | failed | coverage | agent | format | route | thinking | exact | p50 | p95 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def _summary_line(name: str, summary: Mapping[str, Any]) -> str:
    failed = summary[FAILED] + summary[NOT_RUN]
    return (
        f"| {name} | {summary['rows']} | {summary[DECIDED]} | {summary[ABSTAINED]} | {failed} | "
        f"{_pct(summary['coverage'])} | {_pct(summary['agent'])} | {_pct(summary['output_format'])} | "
        f"{_pct(summary['route'])} | {_pct(summary['thinking'])} | {_pct(summary['exact'])} | "
        f"{_seconds(summary['latency_p50_s'])} | {_seconds(summary['latency_p95_s'])} |"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report for a human: summaries, the Colab gates, per-agent and per-tag views, misses."""
    dataset = report["dataset"]
    lines = [
        "# Routing eval",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Dataset: `{dataset['path']}` (sha256 {dataset['sha256']}), {dataset['rows']} rows, "
        f"{dataset['router_rows']} of them reach a model router (the rules abstain)",
        f"- Expected agents: {_counts(dataset['agent'])}; formats: {_counts(dataset['output_format'])}; "
        f"thinking: {_counts(dataset['needs_thinking'])}",
        f"- Target gated precision {report['target_precision']:.2f} on at least {report['min_support']} rows; "
        f"the Colab sweep runs on {'the router traffic' if report['sweep_rows'] == 'router' else 'all rows'}",
        "",
        "Accuracy is over the decided rows; coverage is decided / rows. The Colab router is shown at "
        "the current gates. Thinking is not scored on image_gen rows.",
    ]
    ran = [candidate for candidate in report["candidates"] if "summary" in candidate]
    for subset, title in (("all", "All rows"), ("router", "Router traffic (rows the rules abstain on)")):
        lines += ["", f"## {title}", "", _SUMMARY_HEADER]
        lines += [_summary_line(candidate["name"], candidate["summary"][subset]) for candidate in ran]
    lines += ["", "## Candidates", ""]
    for candidate in report["candidates"]:
        note = f": {candidate['reason']}" if candidate["reason"] else ""
        label = f" ({candidate['label']})" if candidate["label"] else ""
        lines.append(f"- **{candidate['name']}**{label}: {candidate['status']}{note}")
    for candidate in ran:
        if "colab" in candidate:
            lines += _colab_markdown(candidate["colab"], report["current_gates"])
    for candidate in ran:
        lines += ["", f"## {candidate['name']}: expected agent vs decision", ""]
        lines += _confusion_markdown(candidate["confusion"], rules.VALID_AGENTS)
        lines += ["", f"## {candidate['name']}: expected output_format vs decision", ""]
        lines += _confusion_markdown(candidate["format_confusion"], rules.VALID_OUTPUTS)
    lines += ["", "## Route accuracy by tag", ""]
    lines.append("| tag | rows | " + " | ".join(candidate["name"] for candidate in ran) + " |")
    lines.append("|---|---|" + "---|" * len(ran))
    for tag in TAGS:
        total = max((candidate["tags"][tag]["rows"] for candidate in ran), default=0)
        cells = [
            f"{_pct(candidate['tags'][tag]['route'])} ({candidate['tags'][tag][DECIDED]})" for candidate in ran
        ]
        lines.append(f"| {tag} | {total} | " + " | ".join(cells) + " |")
    for candidate in ran:
        found = candidate["misses"]
        lines += ["", f"## {candidate['name']}: misses ({len(found)})", ""]
        if not found:
            lines.append("None.")
            continue
        lines += ["| id | text | expected | got |", "|---|---|---|---|"]
        for miss in found[:_MISSES_SHOWN]:
            lines.append(f"| {miss['id']} | {_cell(miss['text'])} | {miss['expected']} | {_cell(miss['got'])} |")
        if len(found) > _MISSES_SHOWN:
            lines.append(f"\n{len(found) - _MISSES_SHOWN} more in the JSON report.")
    return "\n".join(lines) + "\n"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))


def _colab_markdown(section: Mapping[str, Any], current: Mapping[str, float]) -> list[str]:
    if "withheld" in section:
        return [
            "",
            "## Colab router gates",
            "",
            f"No recommended gates from a partial run: {section['withheld']}.",
            "",
            _SUMMARY_HEADER,
            _summary_line("ungated", section["ungated"]),
            _summary_line("current gates", section["current"]["summary"]),
        ]
    recommended = section["recommended"]
    lines = [
        "",
        "## Colab router gates",
        "",
        f"Swept on {section['rows']} rows, {section['answered']} answered. Recommended values maximise "
        "coverage at the target gated precision (each within what Settings accepts); a gate that "
        "reaches no target gets its strictest value. The thinking gates are swept on the "
        f"{recommended['noul_rows']} scored rows past the recommended confidence gate. The 95% "
        "lower bound shows how little a few rows prove: treat the values as provisional.",
        "",
        "| setting | current | recommended | meets target | through | gated precision | 95% lower bound |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, value in recommended["gates"].items():
        chosen = recommended["chosen"][name]
        through = "-" if chosen is None else f"{chosen['passed']}/{chosen['total']}"
        precision = "-" if chosen is None else _pct(chosen["precision"])
        lower = "-" if chosen is None else _pct(chosen["precision_lower_95"])
        met = "yes" if recommended["meets_target"][name] else "no"
        lines.append(f"| {name} | {current[name]:.2f} | {value:.2f} | {met} | {through} | {precision} | {lower} |")
    lines += [
        "",
        f"The noul decides thinking on {_pct(recommended['noul_decides'])} of those rows at the "
        "recommended values; the rulebook's heuristic decides the rest.",
        "",
        _SUMMARY_HEADER,
        _summary_line("ungated", section["ungated"]),
        _summary_line("current gates", section["current"]["summary"]),
        _summary_line("recommended", recommended["summary"]),
    ]
    for name, curve in section["curves"].items():
        lines += [
            "",
            f"### {name}",
            "",
            "| threshold | through | coverage | gated precision | 95% lower bound |",
            "|---|---|---|---|---|",
        ]
        for point in curve:
            if round(point["threshold"] * 100) % 5 == 0 or point["threshold"] == recommended["gates"][name]:
                lines.append(
                    f"| {point['threshold']:.2f} | {point['passed']}/{point['total']} | "
                    f"{_pct(point['coverage'])} | {_pct(point['precision'])} | {_pct(point['precision_lower_95'])} |"
                )
    return lines


def _confusion_markdown(table: Mapping[str, Mapping[str, int]], values: set[str]) -> list[str]:
    columns = sorted(values) + [ABSTAINED, FAILED, NOT_RUN]
    lines = ["| expected | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for value, counts in table.items():
        lines.append(f"| {value} | " + " | ".join(str(counts.get(column, 0)) for column in columns) + " |")
    return lines


def write_reports(report: Mapping[str, Any], out_dir: Path, stamp: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = out_dir / f"routing-eval-{stamp}.md"
    data = out_dir / f"routing-eval-{stamp}.json"
    markdown.write_text(render_markdown(report), encoding="utf-8")
    data.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return markdown, data


# ── Plan and runner ──────────────────────────────────────────


def plan_lines(
    rows: Sequence[Row],
    router_ids: set[str],
    candidates: Sequence[tuple[str, str]],
    settings: Settings | None,
    keys: Mapping[str, str],
    args: argparse.Namespace,
) -> list[str]:
    """What a run would do, read from the files alone (no network)."""
    lines = [
        f"dataset: {len(rows)} rows, {len(router_ids)} reach a model router (the rules abstain on them)",
    ]
    for name, model in candidates:
        label = candidate_id(name, model)
        if name == RULES:
            lines.append(f"{label}: offline, {len(rows)} rows")
            continue
        if settings is None:
            lines.append(f"{label}: skipped (a live candidate needs --env-file)")
            continue
        try:
            spec = router_spec(settings, name, model)
        except (CandidateSkip, RoleConfigError) as exc:
            lines.append(f"{label}: skipped ({exc})")
            continue
        if name == COLAB:
            state = "" if settings.colab_enabled else " (COLAB_ENABLED is off: it will be skipped)"
            lines.append(
                f"{label}: {spec.label} from the router role; reads the Colab session document from "
                f"Firestore with this machine's ADC, then sends {len(rows)} /v1/systemone requests one "
                f"at a time to a READY session{state}; current gates "
                f"{_gates_view(RouterGates.from_settings(settings))}"
            )
            continue
        rpm = min(FREE_RPM[name], args.max_rpm or FREE_RPM[name])
        key = live_gate.KEY_NAMES[name]
        lines.append(
            f"{label}: {spec.label} from the router role, {len(rows)}-{2 * len(rows)} requests at most "
            f"{rpm}/min (at least {len(rows) / rpm:.1f} min), {args.concurrency} in flight; "
            f"{key} {'set' if keys.get(name) else 'NOT set'}"
        )
        if name == "openrouter" and len(rows) > settings.openrouter_free_daily_limit:
            lines.append(
                f"  note: OPENROUTER_FREE_DAILY_LIMIT is {settings.openrouter_free_daily_limit} a day; "
                "use --limit, or an account with credits (1,000 a day)"
            )
    lines.append(f"reports: {_display_path(args.out_dir)}/routing-eval-<time>.md and .json")
    return lines


async def run_candidates(
    rows: Sequence[Row],
    candidates: Sequence[tuple[str, str]],
    rules_result: CandidateResult,
    ctx: live_gate.GateContext | None,
    runtime: Runtime,
    args: argparse.Namespace,
) -> list[CandidateResult]:
    results: list[CandidateResult] = []

    def redact(text: str) -> str:
        return live_gate.short(ctx.redact(text), 160)

    for name, model in candidates:
        label = candidate_id(name, model)
        if name == RULES:
            results.append(rules_result)
            continue
        if ctx is None:
            results.append(CandidateResult(label, status="skipped", reason="a live candidate needs --env-file"))
            continue
        runtime.progress(f"[routing-eval] {label}: running")
        if name == COLAB:
            results.append(await run_colab(rows, ctx, runtime, redact=redact))
            continue
        settings = ctx.bot_settings()
        if not ctx.key_for(name):
            results.append(
                CandidateResult(label, status="skipped", reason=f"{live_gate.KEY_NAMES[name]} is not set")
            )
            continue
        results.append(
            await run_api(
                rows,
                name,
                model,
                settings,
                runtime,
                rpm=min(FREE_RPM[name], args.max_rpm or FREE_RPM[name]),
                concurrency=args.concurrency,
                redact=redact,
            )
        )
    return results


@contextlib.contextmanager
def _quiet_app_logs() -> Iterator[None]:
    """Keep the app's own log lines (which may quote model output) off the console."""
    app_logger = logging.getLogger("linebot")
    previous = app_logger.level
    app_logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        app_logger.setLevel(previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Routing eval: labelled zh-TW messages through the bot's routers (live ones need approval).",
    )
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATES),
        help="Comma-separated: rules, colab, nvidia[:model], openrouter[:model] (default: all four).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Dotenv file with the keys and settings; live candidates are skipped without one.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL dataset (default: routing_zh_tw.jsonl).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Where the reports go (default: tmp/routing-eval).")
    parser.add_argument(
        "--target-precision",
        type=float,
        default=DEFAULT_TARGET_PRECISION,
        help="Gated precision the recommended Colab gates must reach (default: 0.95).",
    )
    parser.add_argument(
        "--sweep-rows",
        choices=("router", "all"),
        default="router",
        help="Rows the Colab gates are swept on: the router's traffic (default) or all.",
    )
    parser.add_argument("--tag", action="append", default=[], help="Only rows with this tag (repeatable).")
    parser.add_argument("--limit", type=int, default=None, help="An evenly spread sample of this many rows.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"API rows in flight (1-{MAX_CONCURRENCY}; the Colab router always takes one at a time).",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=None,
        help="Requests a minute per API provider, below the free limits (NVIDIA 40, OpenRouter 20).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 and recommend Colab gates even when a candidate left rows not run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and send nothing.")
    return parser


def main(argv: list[str] | None = None, *, runtime: Runtime | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        candidates = parse_candidates(args.candidates)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 < args.target_precision <= 1:
        parser.error("--target-precision must be in (0, 1]")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"--concurrency must be 1-{MAX_CONCURRENCY}")
    if args.max_rpm is not None and args.max_rpm < 1:
        parser.error("--max-rpm must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    unknown = sorted(set(args.tag) - set(TAGS))
    if unknown:
        parser.error(f"unknown tags {unknown} (known: {', '.join(TAGS)})")
    if args.env_file is not None and not args.env_file.is_file():
        parser.error(f"env file not found: {args.env_file.name}")
    runtime = runtime or Runtime(progress=lambda line: print(line, file=sys.stderr, flush=True))

    try:
        rows = select_rows(load_dataset(args.dataset), args.tag, args.limit)
    except (DatasetError, OSError) as exc:
        print(f"[routing-eval] {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("[routing-eval] no rows match --tag", file=sys.stderr)
        return 2
    settings: Settings | None = None
    keys: dict[str, str] = {}
    if args.env_file is not None:
        try:
            settings = live_gate.load_settings(args.env_file)
        except live_gate.GateError as exc:
            print(f"[routing-eval] {exc}", file=sys.stderr)
            return 2
        keys = live_gate.load_keys(args.env_file)

    with _quiet_app_logs():
        rules_result = run_rules(rows)
    router_ids = {outcome.row_id for outcome in rules_result.outcomes if outcome.status == ABSTAINED}
    if args.dry_run:
        for line in plan_lines(rows, router_ids, candidates, settings, keys, args):
            print(f"[routing-eval] {line}")
        print("[routing-eval] dry run: nothing was sent")
        return 0

    ctx = None
    if settings is not None:
        # The gate's own (sync) client is not used: the routers send through
        # the bot's clients over this script's recording transports.
        ctx = live_gate.GateContext(
            client=None,
            nvidia_key=keys["nvidia"],
            openrouter_key=keys["openrouter"],
            settings=settings,
            colab_loader=runtime.colab_loader,
        )
    with _quiet_app_logs():
        results = asyncio.run(run_candidates(rows, candidates, rules_result, ctx, runtime, args))
    current = RouterGates.from_settings(ctx.bot_settings() if ctx is not None else Settings(_env_file=None))
    now = datetime.now().astimezone()
    report = build_report(
        rows,
        args.dataset,
        results,
        router_ids,
        current=current,
        target=args.target_precision,
        sweep_rows=args.sweep_rows,
        generated_at=now.isoformat(timespec="seconds"),
        allow_partial=args.allow_partial,
    )
    markdown, data = write_reports(report, args.out_dir, now.strftime("%Y%m%d-%H%M%S"))
    for candidate in report["candidates"]:
        summary = candidate.get("summary", {}).get("all")
        detail = (
            f"coverage {_pct(summary['coverage'])}, route {_pct(summary['route'])}, exact {_pct(summary['exact'])}"
            if summary
            else candidate["reason"]
        )
        if summary and candidate["reason"]:
            detail += f"; {candidate['reason']}"
        print(f"[routing-eval] {candidate['name']}: {candidate['status']}; {detail}")
    print(f"[routing-eval] reports: {_display_path(markdown)}, {_display_path(data)}")
    partial = [result.name for result in results if result.status == PARTIAL]
    if partial and not args.allow_partial:
        print(
            f"[routing-eval] {', '.join(partial)} left rows not run: rerun, or pass --allow-partial to accept it",
            file=sys.stderr,
        )
    return 1 if any(result.status == "failed" for result in results) or (partial and not args.allow_partial) else 0


if __name__ == "__main__":
    raise SystemExit(main())
