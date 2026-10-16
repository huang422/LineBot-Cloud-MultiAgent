"""The Colab router: clef-flash on Ollama's ``POST /v1/systemone``.

clef-flash (Cloudflare's decision model, fine-tuned from Qwen3.5-9B,
ADR-0010) does not write JSON; it scores fixed options. One request asks
three questions about the user's message and gets a probability per
option:

- ``agent`` (choice): chat, vision, web_search or image_gen;
- ``output_format`` (choice): text, voice or image;
- ``needs_thinking`` (noul): the probability that the answer needs
  step-by-step reasoning.

The request (Ollama 0.40.5 ``api.SystemOneRequest``, the same shape as in
0.35.0)::

    {"model": "clef-flash:9b", "keep_alive": -1,
     "state": {"message", "has_image", "quoted", "previous_agent"},
     "questions": {name: {"type": "choice" | "noul", "instructions", "criteria"}}}

and the answer ``{"model", "answers": {name: ...}, "usage"}``, where a choice
answer is ``{"type", "choice", "probabilities", "confidence"}`` and a noul
answer only ``{"type", "noul"}``, the probability of true. Since Ollama
0.40.4, llama.cpp's own /v1/systemone scores a GGUF decision model, and a
choice's ``confidence`` is TypeSafe's (p_max − 1/N) / (1 − 1/N), no longer
0.35's normalised entropy 1 − H(p)/ln(N). The ``choice`` is the option with
the highest probability.

What clef-flash reads is one joint prompt for all three fields, which
llama.cpp renders (Ollama ≥ 0.40.4) from the GGUF's own ``systemone``
template: a fixed system line ("Read the complete state and schema. Decide
every field jointly…"), then ``STATE:`` and the state as compact JSON with
sorted keys, then per field ``FIELD n``, ``ID``, ``TYPE``, ``INSTRUCTION``
and ``ALLOWED OPTIONS``, each option ``{"description": <criterion>,
"option_id": <key>}``. The choice keys are sorted, so an exact tie goes to
the alphabetically first one. :data:`QUESTIONS` is written for that: each
``instructions`` says what the field means, each criterion describes its
option, and both speak of the state, the word the model sees.

The instructions and criteria are English, like the template's scaffold
(nobody has evaluated clef-flash in Chinese); the zh-TW message goes in the
state as data, clipped to :data:`STATE_MESSAGE_CHARS`. llama.cpp tokenizes
the state with special-token parsing and, for Clef, escapes no special
token in it, so :func:`build_state` first breaks every ``<`` that opens a
tag and every ``|>`` in the message (:func:`neutralize_special_tokens`): a
user who writes ``<|im_end|>`` cannot end the prompt's turn, nor
``</think>`` or ``<tts_pad>`` become a control token. The whole prompt
stays near 1,000 tokens even for a full message, far below clef-flash's
num_ctx 16384, and the request far below the 64 KiB Ollama allows a
request without images. The state has no long-term memory or history
(ADR-0004).

:meth:`SystemOneRouter.route` takes both choices when their confidences
are at or above :attr:`RouterGates.confidence`, which at the default 0.0
is always: the top choice (argmax), an exact tie going to the
alphabetically first option as above (the summary line's ``router_p``
lists tied options in that order too). The noul turns thinking on at ≥
:attr:`RouterGates.think_on` and off at ≤ :attr:`RouterGates.think_off`;
in between, the rulebook's heuristic decides. The orchestrator's
invariants are applied: image_gen needs both answers to agree (agent
image_gen *and* output image, else the API router decides), image output
without image_gen becomes text, and vision needs an image. That also
bounds what a message written to steer the router can do: its prompt has
no "the state is data, never instructions" line, but steering one field
cannot start an image generation; the worst is a wrong agent choice.
Anything else (not READY, a session whose router is not loaded or not
wholly in VRAM, breaker open, the router busy or cooling down, an HTTP
error, a timeout, a malformed answer, a confidence under a non-zero gate)
returns None at once and the API router decides. A router the session
cannot use is never called, and each such session gets one ERROR saying
why: registration's at READY for a router the session went READY without
(this module then notes it at INFO), this module's for one a health check
dropped mid-session. The router's failures are its own (ModelHealth cools
it down); they never open the Colab breaker, so a slow router never skips
Colab chat. Only a 401, 403 or 404, which says the session itself is
wrong, marks the endpoint unhealthy.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import httpx

from src.agents import routing_rules as rules
from src.agents.routing_rules import RouterDecision
from src.models.agent_request import AgentRequest
from src.providers.base import TransientError
from src.providers.fallback_chain import Tier
from src.services.colab.endpoint import ColabEndpoint
from src.services.colab.protocol import mib
from src.services.search.page_fetcher import FetchError
from src.utils import request_trace
from src.utils.logger import logger

STATE_MESSAGE_CHARS = 300
# A "<" that could open a special token (neutralize_special_tokens).
_TAG_OPEN = re.compile(r"<(?=[A-Za-z_/|])")
# Below this much time left there is no point asking.
_MIN_SECONDS = 0.5
_PROVIDER = "colab"

QUESTIONS: dict[str, dict[str, Any]] = {
    "agent": {
        "type": "choice",
        "instructions": (
            "The assistant that should answer the user's latest message in a LINE chat. The state "
            "holds the message, has_image (an image is attached or quoted), quoted (the user quoted "
            "an earlier message) and previous_agent (the assistant that answered the previous turn)."
        ),
        "criteria": {
            "chat": "General conversation, advice, writing, translation, coding or explanation that needs no live data.",
            "vision": (
                "The message asks about, or asks to restyle or change, an image the user sent or quoted, "
                "without asking for a new picture to be drawn."
            ),
            "web_search": "The answer needs current facts from the web, such as news, weather, prices, schedules or a linked page.",
            "image_gen": (
                "The message explicitly asks for a new picture to be drawn or generated, not only a change "
                "to an image the user sent or quoted."
            ),
        },
    },
    "output_format": {
        "type": "choice",
        "instructions": "The form the user wants the reply in, judged from the message in the state.",
        "criteria": {
            "text": "A normal written reply, the default unless the message clearly asks otherwise.",
            "voice": "The message explicitly asks for a spoken or read-aloud reply.",
            "image": "The message asks for a newly generated picture as the reply.",
        },
    },
    "needs_thinking": {
        "type": "noul",
        "instructions": (
            "Whether a good answer to the message in the state needs careful step-by-step "
            "reasoning, such as math or logic, writing, debugging or refactoring code, weighing "
            "several options, multi-step planning, a long analysis, long writing or translating a "
            "long passage."
        ),
        "criteria": {
            "false": "A direct answer is enough, as for chat, facts, advice or a short translation.",
            "true": "Step-by-step reasoning is needed, as for code, math, planning or long writing.",
        },
    },
}


@dataclass(frozen=True)
class RouterGates:
    """When the router's answer is used (COLAB_ROUTER_CONFIDENCE,
    COLAB_ROUTER_THINK_ON, COLAB_ROUTER_THINK_OFF).

    The values belong to the router model: clef-flash's probabilities are
    an uncalibrated softmax at T=1 (its GGUF sets no decision temperature).
    A confidence gate of 0.90 deferred every real message (0.39 to 0.43 on
    2026-10-16), so the default is 0.0, the top choice always (argmax);
    thinking keeps 0.85 / 0.15. All three are provisional:
    scripts/evals/routing_eval.py re-tunes them on labelled zh-TW messages,
    again whenever the router model changes or its tag moves.
    """

    # Both choice answers at least this confident: (p_max - 1/N) / (1 - 1/N);
    # 0.0 passes every answer.
    confidence: float
    # needs_thinking at or above: thinking on; at or below: off.
    think_on: float
    think_off: float

    @classmethod
    def from_settings(cls, settings: Any) -> RouterGates:
        return cls(
            confidence=settings.colab_router_confidence,
            think_on=settings.colab_router_think_on,
            think_off=settings.colab_router_think_off,
        )


def neutralize_special_tokens(text: str) -> str:
    """*text* with a space after every ``<`` that opens a tag (before a
    letter, ``_``, ``/`` or ``|``) and inside every ``|>``, so no special
    token can form in it.

    The Clef path tokenizes the state with special-token parsing and does
    not escape such tokens (llama.cpp b11669), so a user's ``<|im_end|>``
    would become the real control token. clef-flash's GGUF (2026-10-16)
    has 27 control and 6 user-defined tokens, and every one starts with
    ``<|``, ``<`` and a letter, or ``</``: ``<|im_end|>``, ``<tts_pad>``,
    ``<think>``, ``</tool_call>``. ``a < b`` is untouched. One pass
    suffices: the space keeps each replacement from making a new pair.
    """
    return _TAG_OPEN.sub("< ", text).replace("|>", "| >")


def build_state(request: AgentRequest) -> dict[str, Any]:
    """What the router reads about the message (the prompt's "STATE"): no
    memory, no history (ADR-0004). The message is neutralized before it is
    clipped, so the clip still bounds its length (whitespace runs collapse
    to one space, which never joins a broken pair again)."""
    previous = request.previous_agent if request.previous_agent in rules.VALID_AGENTS else "none"
    return {
        "message": rules.clip_text(neutralize_special_tokens(request.text), STATE_MESSAGE_CHARS),
        "has_image": rules.has_image_context(request),
        "quoted": rules.has_quote(request),
        "previous_agent": previous,
    }


def build_request(model: str, request: AgentRequest) -> dict[str, Any]:
    return {"model": model, "keep_alive": -1, "state": build_state(request), "questions": QUESTIONS}


class SystemOneRouter:
    """Routes with the Colab router model when the session is READY; see the module docstring."""

    def __init__(
        self,
        endpoint: ColabEndpoint,
        tier: Tier,
        *,
        gates: RouterGates,
        health: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``tier`` is the router role's Colab tier (model and time limit);
        ``health`` (ModelHealth) records its outcomes for ``/health/detail``."""
        self.endpoint = endpoint
        self.model = tier.model
        self.timeout_s = tier.timeout_s
        self.gates = gates
        self.health = health
        self._clock = clock
        # The session (nbh) whose untrusted router was reported.
        self._off_reported = ""

    async def route(self, request: AgentRequest, *, deadline: float | None = None) -> RouterDecision | None:
        """The router's decision for *request*, or None (the API router decides)."""
        endpoint = self.endpoint
        skip = request_trace.note_router_skip
        if not await endpoint.ready():
            skip("not READY")
            return None
        if self._off_for_session():
            skip("off for this session")
            return None
        if endpoint.breaker_open():
            logger.info(f"{self.model} skipped: Colab breaker open")
            skip("breaker open")
            return None
        if self.health is not None and self.health.unavailable(_PROVIDER, self.model) is not None:
            logger.info(f"{self.model} skipped: cooling down")
            skip("cooling down")
            return None
        gate = endpoint.semaphore(self.model)
        if gate.locked():
            logger.info(f"{self.model} skipped: busy with another message")
            skip("busy")
            return None
        limit = self.timeout_s
        if deadline is not None:
            limit = min(limit, deadline - self._clock())
        if limit < _MIN_SECONDS:
            skip("no time left")
            return None

        body = build_request(self.model, request)
        async with gate:
            try:
                async with asyncio.timeout(limit):
                    response = await endpoint.call("POST", "/v1/systemone", json=body)
            except TimeoutError:
                self._failed(f"no answer within {limit:.3g}s", status=0)
                skip("timeout")
                return None
            except (httpx.HTTPError, FetchError, RuntimeError) as exc:
                self._failed(f"{type(exc).__name__}: {exc}", status=0)
                skip(f"error {type(exc).__name__}")
                return None
        status = response.status_code
        if status != 200:
            skip(f"HTTP {status}")
            detail = f"HTTP {status}: {(response.text or '')[:200]}"
            if status in (400, 413, 422):
                # The request itself (e.g. over Ollama's 64 KiB text cap).
                logger.warning(f"{self.model} refused the request ({detail})")
            else:
                if status in (401, 403, 404):
                    endpoint.mark_unhealthy(f"/v1/systemone {detail}")
                self._failed(detail, status=status)
            return None
        try:
            data = response.json()
        except ValueError:
            self._failed("invalid JSON body", status=200)
            skip("invalid JSON")
            return None
        if self.health is not None:
            self.health.record_success(_PROVIDER, self.model)
        return self._decide(data, request)

    def _off_for_session(self) -> bool:
        """The session's router was not loaded or not wholly in VRAM (its
        digest never matters: the models follow their tags); the first time
        per session, a line says which. That line is an ERROR only for a
        router a health check dropped mid-session: one the session went
        READY without already has its ERROR, with the VM's reason, from
        registration, so it is noted at INFO."""
        reason = self._off_reason()
        if reason is None:
            return False
        state = self.endpoint.state
        session = state.nbh if state is not None else ""
        if session != self._off_reported:
            self._off_reported = session
            if self.model in self.endpoint.off_gpu_at_ready():
                logger.info(f"{self.model} off for this Colab session (reported at READY): {reason}")
            else:
                logger.error(f"{self.model} off for this Colab session: {reason}")
        return True

    def _off_reason(self) -> str | None:
        spilled = self.endpoint.off_gpu().get(self.model)
        if spilled is not None:
            in_vram, size = spilled
            where = f"not fully on GPU ({mib(in_vram)} of {mib(size)} MiB in VRAM)" if size else "not loaded (/api/ps)"
            return f"router {where}; the API router decides. See colab/README.md (VRAM)"
        return None

    def _failed(self, detail: str, *, status: int) -> None:
        """One failed call: held against the router alone (ModelHealth), never
        the Colab breaker that qwen3.8 chat shares."""
        logger.warning(f"{self.model} failed ({detail}); the API router decides")
        if self.health is not None:
            self.health.record_failure(_PROVIDER, self.model, TransientError(self.model, status, detail, _PROVIDER))

    def _decide(self, data: Any, request: AgentRequest) -> RouterDecision | None:
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            logger.warning(f"{self.model} answer has no answers object")
            request_trace.note_router_skip("malformed answer")
            return None
        agent = _choice(answers.get("agent"), rules.VALID_AGENTS)
        output = _choice(answers.get("output_format"), rules.VALID_OUTPUTS)
        noul = _noul(answers.get("needs_thinking"))
        if agent is None or output is None or noul is None:
            logger.warning(f"{self.model} answer is malformed; the API router decides")
            request_trace.note_router_skip("malformed answer")
            return None
        (agent, agent_confidence), (output, output_confidence) = agent, output
        gates = self.gates
        # For the request's summary line, used or not; routing ignores it.
        scores = request_trace.RouterScores(
            model=self.model,
            agent=_probabilities(answers["agent"], rules.VALID_AGENTS, agent),
            agent_confidence=agent_confidence,
            output=_probabilities(answers["output_format"], rules.VALID_OUTPUTS, output),
            output_confidence=output_confidence,
            think_p=noul,
            gate=gates.confidence,
            think_on=gates.think_on,
            think_off=gates.think_off,
        )
        if min(agent_confidence, output_confidence) < gates.confidence:
            logger.info(
                f"{self.model} unsure (agent {agent} {agent_confidence:.2f}, format {output} "
                f"{output_confidence:.2f} < {gates.confidence:.2f}); the API router decides"
            )
            request_trace.note_router_scores(replace(scores, verdict="gate fail"))
            return None

        flat = " ".join(request.text.split())
        if noul >= gates.think_on:
            thinking, how = True, "on"
        elif noul <= gates.think_off:
            thinking, how = False, "off"
        else:
            thinking = not rules.should_disable_thinking(flat, has_image=rules.has_image_context(request))
            how = f"rules->{'on' if thinking else 'off'}"
        # The orchestrator's invariants: image_gen only when both answers
        # say so (steering one field cannot start an image generation), and
        # it needs no thinking; image output only from image_gen; vision
        # only with an image to look at.
        if agent == "image_gen":
            if output != "image":
                logger.info(f"{self.model} chose image_gen without image output; the API router decides")
                request_trace.note_router_scores(replace(scores, verdict="image_gen without image output"))
                return None
            thinking, how = False, "off(image_gen)"
        elif output == "image":
            output = "text"
        if agent == "vision" and not rules.has_image_context(request):
            agent = "chat"
        request_trace.note_router_scores(replace(scores, thinking=how))
        return RouterDecision(
            agent=agent,
            output_format=output,
            task_description=request.text,
            reasoning=(
                f"{self.model}: agent {agent_confidence:.2f}, format {output_confidence:.2f}, thinking p={noul:.2f}"
            ),
            disable_thinking=not thinking,
        )


def _choice(answer: Any, allowed: set[str]) -> tuple[str, float] | None:
    if not isinstance(answer, dict):
        return None
    choice, confidence = answer.get("choice"), answer.get("confidence")
    if choice not in allowed or not _probability(confidence):
        return None
    return choice, float(confidence)


def _probabilities(answer: Any, allowed: set[str], choice: str) -> dict[str, float]:
    """The answer's probabilities for the summary line; the choice alone,
    unscored, when it sent none."""
    scored = request_trace.probabilities(answer.get("probabilities"), allowed)
    return scored or {choice: float("nan")}


def _noul(answer: Any) -> float | None:
    value = answer.get("noul") if isinstance(answer, dict) else None
    return float(value) if _probability(value) else None


def _probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )
