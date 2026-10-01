"""Walks an ordered list of model tiers until one answers.

A *tier* (:class:`Tier`) is one provider/model pair with its own time limit,
per-request options and input modalities; the role table
(:mod:`src.runtime.model_roles`) builds them. For each tier, in order:

1. once the caller's ``deadline`` (``time.monotonic()`` seconds) has
   passed, stop with :class:`DeadlineExceededError`;
2. skip a tier whose modalities lack ``"image"`` when the request carries
   images, and a tier :class:`ModelHealth` reports dead, cooling down or
   over its provider budget;
3. call it under ``asyncio.timeout(min(tier.timeout_s, time left))``. A
   thinking tier (``options["thinking"] is True``) that has a tier after it
   that could run (step 2) must be done by the *thinking cutoff*, set
   once, at the first such tier, halfway to the deadline: the thinking
   tiers together use at most half the time left and the tiers after them
   keep the other half (one that finds the cutoff passed is skipped). A
   Colab tier (``Tier.shares_time``) with such a tier after it must stop
   :data:`COLAB_API_RESERVE_SECONDS` before the deadline, or halfway to it
   when that is later (a short window), so a slow Colab session cannot
   take the whole window from the API tiers; one thinking for a plain
   request (``COLAB_CHAT_THINK``) is held to that share alone, not the
   thinking cutoff. With nothing usable after it, a tier gets all the
   time left;
4. an error, a timeout or an empty completion moves on to the next tier;
   :class:`SkipTierError` counts as not attempted.

Outcomes are recorded in :class:`ModelHealth`. A timeout counts against
the model (as a :class:`TransientError`) when the model had its full
``timeout_s``, or at least ``min_tier_seconds`` before the deadline (for a
Colab tier, its share of the time left) cut it off, so a hung endpoint
cools down instead of eating every request's reply window. It does not
count when the deadline left the tier only a moment, or for a thinking
tier serving a request that needs thinking (``disable_thinking=False``):
thinking is slow by design (the same model is usually tried again right
after with thinking off, which does tell). A tier that thinks although the
request does not need it (Colab chat at ``COLAB_CHAT_THINK=low``) is held
to its timeouts like any other. A caller whose deadline
is its own short budget rather than the reply window (the search query
planner) passes ``hold_deadline_timeouts=False``: a timeout that budget
cut short then never counts, since the model may simply need longer than
the caller chose to wait. Every walk, ending in an answer or an error, is
also recorded for the request's summary log line: each tier reached, its
outcome and seconds (:func:`src.utils.request_trace.record_walk`); a call
that only asks for tools is ``tools`` there, not ``ok``.

``options["thinking"]`` (True/False) overrides the call's
``disable_thinking`` for that tier, so a thinking request is simply the
thinking role's tiers followed by the chat tiers with thinking off
(``BaseAgent.targets_for``). The provider gets the tier's ``options`` too
(json_mode, reasoning_budget, effort, ...), see
:mod:`src.runtime.model_roles`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from src.providers.base import (
    EmptyCompletionError,
    ProviderError,
    ProviderResponse,
    RateLimitError,
    SkipTierError,
    TransientError,
    is_empty_completion,
)
from src.providers.messages import has_image_parts
from src.services.model_health import DEAD, RATE_LIMIT, ModelHealth
from src.utils import request_trace
from src.utils.logger import logger
from src.utils.request_trace import Attempt

# Defaults for a Tier built by hand (tests, scripts); the role table always
# sets both from the role and the model profile.
DEFAULT_TIER_TIMEOUT_SECONDS = 60.0
# Settings.min_tier_seconds default: the least time a tier must have had
# before a deadline timeout is held against its model.
DEFAULT_MIN_TIER_SECONDS = 5.0
TEXT_AND_IMAGE: frozenset[str] = frozenset({"text", "image"})
# Share of the time left that the thinking tiers together, or one Colab
# tier, may use when tiers come after them.
_THINKING_SHARE = 0.5
# Seconds a Colab tier leaves the API tiers after it when that is more than
# half the time left: in the reply window (REPLY_WINDOW_SECONDS, 90 s) Colab
# runs about 70 s and an API backup still has time to answer.
COLAB_API_RESERVE_SECONDS = 20.0


class LLMProvider(Protocol):
    """Structural typing for any provider that has a generate() method."""

    async def generate(self, model: str, messages: list[dict], **kwargs: Any) -> ProviderResponse: ...


def provider_name(provider: Any) -> str:
    """The provider's health key prefix (``NvidiaProvider.name`` is "nvidia")."""
    name = getattr(provider, "name", None)
    return name if isinstance(name, str) and name else type(provider).__name__.lower()


@dataclass(frozen=True)
class Tier:
    """One step of a fallback walk."""

    provider: LLMProvider
    model: str
    timeout_s: float = DEFAULT_TIER_TIMEOUT_SECONDS
    # Per-tier request options, passed to provider.generate(options=...);
    # the chain itself reads "thinking" (True/False).
    options: Mapping[str, Any] = field(default_factory=dict, hash=False)
    # Input the model accepts; image-bearing requests skip text-only tiers.
    modalities: frozenset[str] = TEXT_AND_IMAGE
    # With a usable tier after it, stop COLAB_API_RESERVE_SECONDS before the
    # deadline, or at half the time left when that is later (the role table
    # sets it on Colab tiers).
    shares_time: bool = False

    @property
    def provider_name(self) -> str:
        return provider_name(self.provider)

    @property
    def label(self) -> str:
        return f"{self.provider_name}:{self.model}"

    def with_options(self, **options: Any) -> Tier:
        return replace(self, options={**self.options, **options})


class AllModelsRateLimitedError(Exception):
    """Raised when every usable target is rate limited or over budget."""


class AllProvidersFailedError(Exception):
    """Raised when every target fails for reasons other than pure rate limiting."""

    def __init__(self, message: str, last_error: Exception | None = None) -> None:
        self.last_error = last_error
        super().__init__(message)


class DeadlineExceededError(AllProvidersFailedError):
    """The caller's deadline passed before a tier could answer."""


@dataclass
class _Walk:
    """What happened to the tiers so far, for the final error."""

    last_error: Exception | None = None
    attempted: int = 0
    saw_rate_limit: bool = False
    saw_failure: bool = False
    dead: list[str] = field(default_factory=list)
    # Each tier's outcome, for the request's summary line (request_trace).
    attempts: list[Attempt] = field(default_factory=list)

    def deadline_error(self, label: str) -> DeadlineExceededError:
        return DeadlineExceededError(
            f"deadline reached before {label} could answer. Last error: {self.last_error}",
            last_error=self.last_error,
        )


class FallbackChain:
    """Tries tiers in order with per-tier time limits and health checks.

    ``min_tier_seconds`` (MIN_TIER_SECONDS) is the least time a tier must
    have had before a deadline timeout counts against its model.
    """

    def __init__(
        self,
        health: ModelHealth,
        *,
        min_tier_seconds: float = DEFAULT_MIN_TIER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.health = health
        self.min_tier_seconds = min_tier_seconds
        self._clock = clock
        self.fallback_count: int = 0

    async def generate(
        self,
        targets: Sequence[Tier],
        messages: list[dict],
        *,
        deadline: float | None = None,
        disable_thinking: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        json_schema: dict | None = None,
        hold_deadline_timeouts: bool = True,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Answer from the first tier that can.

        Args:
            targets: Ordered tiers.
            messages: OpenAI-format messages.
            deadline: ``time.monotonic()`` time after which no tier starts
                and none keeps running; None means only per-tier limits.
            disable_thinking: For tiers without ``options["thinking"]``.
            tools, tool_choice: OpenAI tool calling, forwarded as given.
            json_schema: Constrains the reply where the endpoint supports it.
            hold_deadline_timeouts: False when ``deadline`` is the caller's
                own budget: a timeout it cut short is not held against the
                model (one that ran the tier's full ``timeout_s`` still is).
            **kwargs: Forwarded to ``provider.generate()`` (temperature, ...).

        Raises:
            AllModelsRateLimitedError: Everything usable was rate limited.
            DeadlineExceededError: The deadline passed first.
            AllProvidersFailedError: Anything else.
        """
        tiers = list(targets)
        if not tiers:
            raise AllProvidersFailedError("No fallback targets configured")
        has_images = has_image_parts(messages)
        walk = _Walk()
        try:
            # When the thinking tiers must be done, set at the first of them.
            thinking_cutoff: float | None = None

            for index, tier in enumerate(tiers):
                now = self._clock()
                if deadline is not None and now >= deadline:
                    walk.attempts.append(request_trace.skipped(tier.label, "deadline"))
                    raise walk.deadline_error(tier.label)
                name, model = tier.provider_name, tier.model

                if has_images and "image" not in tier.modalities:
                    logger.info(f"Skipping {tier.label} (no image input)")
                    walk.attempts.append(request_trace.skipped(tier.label, "no image input"))
                    continue
                blocked = self.health.unavailable(name, model)
                if blocked is not None:
                    logger.info(f"Skipping {tier.label} ({blocked.kind}: {blocked.reason})")
                    walk.attempts.append(request_trace.skipped(tier.label, blocked.kind))
                    if blocked.kind == DEAD:
                        walk.dead.append(tier.label)
                    elif blocked.kind == RATE_LIMIT:
                        walk.saw_rate_limit = True
                    else:
                        walk.saw_failure = True
                    continue

                thinking = tier.options.get("thinking") is True
                # Where this tier must stop: the deadline, or the end of its
                # share when a tier after it could still answer.
                until = deadline
                if (
                    deadline is not None
                    and (thinking or tier.shares_time)
                    and any(self._can_run(later, has_images) for later in tiers[index + 1 :])
                ):
                    # A Colab tier thinking for a plain request (COLAB_CHAT_THINK)
                    # keeps its Colab share, not the thinking tiers' half.
                    if thinking and not (tier.shares_time and disable_thinking):
                        if thinking_cutoff is None:
                            thinking_cutoff = now + (deadline - now) * _THINKING_SHARE
                        if thinking_cutoff <= now:
                            logger.info(f"Skipping {tier.label} (the thinking tiers' half of the time is used up)")
                            walk.attempts.append(request_trace.skipped(tier.label, "thinking half used"))
                            continue
                        until = thinking_cutoff
                    if tier.shares_time:
                        share = max(now + (deadline - now) * _THINKING_SHARE, deadline - COLAB_API_RESERVE_SECONDS)
                        until = min(until, share)
                shared = until is not None and until < deadline
                limit, deadline_bound = self._time_limit(tier, until, now)
                call_kwargs = dict(
                    kwargs,
                    disable_thinking=_disable_thinking(tier, disable_thinking),
                    options=dict(tier.options),
                )
                if tools:
                    call_kwargs["tools"] = tools
                    if tool_choice is not None:
                        call_kwargs["tool_choice"] = tool_choice
                if json_schema is not None:
                    call_kwargs["json_schema"] = json_schema

                walk.attempted += 1
                try:
                    async with asyncio.timeout(limit):
                        result = await tier.provider.generate(model=model, messages=messages, **call_kwargs)
                except TimeoutError:
                    walk.attempts.append(Attempt(tier.label, request_trace.TIMEOUT, self._clock() - now))
                    timeout = TransientError(model, 0, f"no answer within {limit:.3g}s", name)
                    cut_short = deadline_bound and (
                        not hold_deadline_timeouts or limit < self.min_tier_seconds
                    )
                    if (thinking and not disable_thinking) or cut_short:
                        # Thinking a request needs is slow by design, and a
                        # moment's slice of the deadline (or a caller's own
                        # short budget) says nothing: not held against the model.
                        logger.warning(f"Fallback: {tier.label} gave no answer within {limit:.3g}s")
                        walk.last_error = timeout
                        walk.saw_failure = True
                    else:
                        self._failed(walk, tier, timeout)
                    # A tier cut off at the end of its share leaves the rest to
                    # the tiers after it.
                    if (deadline_bound and not shared) or (deadline is not None and self._clock() >= deadline):
                        raise walk.deadline_error(tier.label) from None
                    continue
                except SkipTierError as exc:
                    walk.attempted -= 1
                    logger.info(f"Skipping {tier.label} ({exc.detail})")
                    walk.attempts.append(request_trace.skipped(tier.label, exc.detail))
                    continue
                except Exception as exc:
                    # Defence in depth: a provider that breaks the error contract
                    # must not take the whole chain down with it. CancelledError
                    # derives from BaseException and is not caught.
                    walk.attempts.append(Attempt(tier.label, _outcome(exc), self._clock() - now))
                    self._failed(walk, tier, exc)
                    continue

                if is_empty_completion(result):
                    walk.attempts.append(Attempt(tier.label, request_trace.EMPTY, self._clock() - now))
                    finish_reason = getattr(result, "finish_reason", None)
                    self._failed(walk, tier, EmptyCompletionError(model, name, finish_reason=finish_reason))
                    continue

                walk.attempts.append(
                    Attempt(
                        tier.label,
                        _answered_outcome(result),
                        self._clock() - now,
                        think=_think_label(tier, call_kwargs["disable_thinking"]),
                    )
                )
                self.health.record_success(name, model)
                if walk.attempted > 1:
                    self.fallback_count += 1
                return result

            # A dead model stays out for hours or for good, so "busy, try
            # later" would be a lie unless something was merely rate limited.
            if walk.saw_failure or not walk.saw_rate_limit:
                dead_note = f" Dead: {', '.join(walk.dead)}." if walk.dead else ""
                raise AllProvidersFailedError(
                    f"All targets failed.{dead_note} Last error: {walk.last_error}",
                    last_error=walk.last_error,
                )
            raise AllModelsRateLimitedError(f"All targets exhausted. Last error: {walk.last_error}")
        finally:
            request_trace.record_walk(walk.attempts)

    def _can_run(self, tier: Tier, has_images: bool) -> bool:
        """Whether the walk would call *tier* now (step 2 would not skip it)."""
        if has_images and "image" not in tier.modalities:
            return False
        return self.health.unavailable(tier.provider_name, tier.model) is None

    @staticmethod
    def _time_limit(tier: Tier, until: float | None, now: float) -> tuple[float, bool]:
        """Seconds this tier may run, and whether *until* (the deadline, or
        the end of the tier's share of it) set them rather than ``timeout_s``."""
        if until is None:
            return tier.timeout_s, False
        allowed = until - now
        if allowed < tier.timeout_s:
            return allowed, True
        return tier.timeout_s, False

    def _failed(self, walk: _Walk, tier: Tier, error: Exception) -> None:
        if isinstance(error, RateLimitError):
            logger.warning(f"Fallback: {tier.label} rate limited, trying next")
            walk.saw_rate_limit = True
        else:
            status = f" {error.status}" if isinstance(error, ProviderError) else ""
            logger.warning(
                f"Fallback: {tier.label} failed ({type(error).__name__}{status}: {error}), trying next"
            )
            walk.saw_failure = True
        walk.last_error = error
        self.health.record_failure(tier.provider_name, tier.model, error)


def _outcome(error: Exception) -> str:
    """The summary line's word for a failed call (never the error's message)."""
    if isinstance(error, RateLimitError):
        return request_trace.RATE_LIMITED
    return request_trace.error_outcome(error)


def _answered_outcome(result: ProviderResponse) -> str:
    """``tools`` for a call that only asked for tools (a tool-loop round,
    not the answer), else ``ok``."""
    if result.tool_calls and not (result.text or result.images):
        return request_trace.TOOLS
    return request_trace.OK


def _think_label(tier: Tier, disabled: bool) -> str:
    """How the call thought, for the summary line: ``off``, the tier's
    level (Colab ``think``, OpenRouter ``effort``) or ``on``."""
    if disabled:
        return "off"
    return str(tier.options.get("think") or tier.options.get("effort") or "on")


def _disable_thinking(tier: Tier, default: bool) -> bool:
    thinking = tier.options.get("thinking")
    return default if thinking is None else not thinking
