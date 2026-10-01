"""Which models and providers can take a request right now.

State is keyed by ``"provider:model"`` (e.g. ``"nvidia:meta/muse-glimmer-30b"``)
and comes in two layers:

- **Per model:** a cooldown (HTTP 429 for its Retry-After or 60 s, two
  transient failures in a row, or two minutes after an HTTP 403 that does
  not name the key: one function the key may not call, a moderation
  refusal), a *dead* flag, the last error and the last success. HTTP 410
  is a documented end of life, so the model stays dead for the rest of
  the process; HTTP 404 can be transient or account-scoped, so it ends on
  its own (six hours on NVIDIA, ten minutes on OpenRouter, whose "No
  endpoints found for <id>" clears when an upstream comes back). An
  OpenRouter 404 saying no endpoint supports this request (tool use,
  image input) is the request's problem: a ``PayloadError``, which changes
  nothing here.
- **Per provider:** a request budget shared by that provider's models
  (NVIDIA: 40/minute and 10,000/day; OpenRouter ``:free`` ids: 20/minute
  and ``OPENROUTER_FREE_DAILY_LIMIT``/day; Colab: none) and a provider-wide
  cooldown: 10 minutes after a refused key (HTTP 401, or a 403 whose body
  names the key, like OpenRouter's "Key limit exceeded") or an
  account-level OpenRouter 429 (the ``free-models-per-day`` / ``-per-min``
  caps), an hour after HTTP 402. A 429 that OpenRouter passes on from one
  model's upstream ("temporarily rate-limited upstream") cools only that
  model, so the other ``:free`` ids stay in rotation.

In-memory only: a redeploy or instance restart clears it. The per-minute
Cloud Scheduler job can keep one instance alive for days, so everything
that might come back expires on its own.
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from src.providers.base import (
    AuthError,
    CreditsError,
    ModelGoneError,
    ProviderError,
    RateLimitError,
    SkipTierError,
    TransientError,
)
from src.providers.openrouter_provider import is_free_model_id
from src.utils.logger import logger

NVIDIA = "nvidia"
OPENROUTER = "openrouter"

DEFAULT_OPENROUTER_FREE_DAILY_LIMIT = 50

# Why a model cannot take a request (Unavailable.kind). The chain reports
# "busy, try later" only when everything it could not use was RATE_LIMIT.
DEAD = "dead"
RATE_LIMIT = "rate_limit"
COOLDOWN = "cooldown"

_NVIDIA_RPM = 40
_NVIDIA_RPD = 10_000
_OPENROUTER_FREE_RPM = 20

_DEFAULT_429_SECONDS = 60
_OPENROUTER_ACCOUNT_429_SECONDS = 600
# OpenRouter names its account caps in the 429 body ("Rate limit exceeded:
# free-models-per-day. ..."); an upstream 429 says "rate-limited upstream".
_OPENROUTER_ACCOUNT_429_MARKER = "free-models-per-"
_UPSTREAM_429_MARKER = "upstream"
_AUTH_COOLDOWN_SECONDS = 600
# A 403 refuses the key itself only when its body says so; otherwise it
# refuses one model (NVIDIA's "Authorization failed" for a function the key
# may not call, OpenRouter's moderation refusal) and cools only that model.
_KEY_RE = re.compile(r"\bkey\b", re.IGNORECASE)
_FORBIDDEN_COOLDOWN_SECONDS = 120
_CREDITS_COOLDOWN_SECONDS = 3600
_NOT_FOUND_DEAD_SECONDS = 6 * 3600
_NOT_FOUND_DEAD_SECONDS_BY_PROVIDER = {OPENROUTER: 600}
# Two transient failures in a row (timeouts, 5xx) cool a model down, so a
# hung endpoint stops eating every request's reply window.
_TRANSIENT_STRIKES = 2
_TRANSIENT_COOLDOWN_SECONDS = 120
_ERROR_TEXT_LIMIT = 200


@dataclass(frozen=True)
class Unavailable:
    """Why a model cannot take a request: ``kind`` is DEAD, RATE_LIMIT or COOLDOWN."""

    kind: str
    reason: str


@dataclass(frozen=True)
class _Budget:
    rpm: int
    rpd: int


@dataclass
class _ModelState:
    cooldown_until: float = 0.0
    cooldown_kind: str = RATE_LIMIT
    cooldown_reason: str = ""
    # Why the model is out of rotation; None while it is alive.
    dead_reason: str | None = None
    # Clock time at which a temporary death ends; None = process lifetime.
    dead_until: float | None = None
    eol: str | None = None
    last_error: str | None = None
    last_error_at: float | None = None
    last_ok_at: float | None = None
    transient_strikes: int = 0


@dataclass
class _ProviderState:
    requests_this_minute: deque[float] = field(default_factory=deque)
    requests_today: deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0
    cooldown_kind: str = COOLDOWN
    cooldown_reason: str = ""


def _span(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def _is_account_429(error: RateLimitError) -> bool:
    """True when an OpenRouter 429 is the account's free-model cap.

    Anything else (an upstream provider's limit on one model, or a body
    that names neither) cools only that model; the local request budget
    already keeps the account under its caps.
    """
    detail = (error.detail or "").lower()
    return _OPENROUTER_ACCOUNT_429_MARKER in detail and _UPSTREAM_429_MARKER not in detail


def _describe(error: BaseException) -> str:
    if isinstance(error, ProviderError):
        text = f"HTTP {error.status}: {error.detail}" if error.status else error.detail
    else:
        text = f"{type(error).__name__}: {error}"
    return text[:_ERROR_TEXT_LIMIT]


class ModelHealth:
    """In-memory availability of every model and provider this process calls."""

    def __init__(
        self,
        *,
        openrouter_free_daily_limit: int = DEFAULT_OPENROUTER_FREE_DAILY_LIMIT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if openrouter_free_daily_limit <= 0:
            raise ValueError("openrouter_free_daily_limit must be greater than 0")
        self._clock = clock
        self._models: dict[str, _ModelState] = {}
        self._providers: dict[str, _ProviderState] = {}
        self._budgets: dict[str, _Budget] = {
            NVIDIA: _Budget(rpm=_NVIDIA_RPM, rpd=_NVIDIA_RPD),
            OPENROUTER: _Budget(rpm=_OPENROUTER_FREE_RPM, rpd=openrouter_free_daily_limit),
        }

    @staticmethod
    def key(provider: str, model: str) -> str:
        return f"{provider}:{model}"

    # ── Queries ──────────────────────────────────────────────

    def unavailable(self, provider: str, model: str) -> Unavailable | None:
        """Why *model* cannot take a request now, or None when it can."""
        now = self._clock()
        state = self._models.get(self.key(provider, model))
        if state is not None:
            self._revive(provider, model, state, now)
            if state.dead_reason:
                return Unavailable(DEAD, state.dead_reason)
            if now < state.cooldown_until:
                return Unavailable(state.cooldown_kind, state.cooldown_reason)
        provider_state = self._providers.get(provider)
        if provider_state is not None and now < provider_state.cooldown_until:
            return Unavailable(provider_state.cooldown_kind, provider_state.cooldown_reason)
        if self._budgeted(provider, model):
            budget = self._budgets[provider]
            provider_state = self._provider(provider)
            self._trim(provider_state, now)
            if len(provider_state.requests_this_minute) >= budget.rpm:
                return Unavailable(RATE_LIMIT, f"{provider} budget: {budget.rpm} requests/minute")
            if len(provider_state.requests_today) >= budget.rpd:
                return Unavailable(RATE_LIMIT, f"{provider} budget: {budget.rpd} requests/day")
        return None

    def is_available(self, provider: str, model: str) -> bool:
        return self.unavailable(provider, model) is None

    def is_dead(self, provider: str, model: str) -> bool:
        blocked = self.unavailable(provider, model)
        return blocked is not None and blocked.kind == DEAD

    # ── Recording ────────────────────────────────────────────

    def record_request(self, provider: str, model: str) -> None:
        """Count one request that reached *provider* against its budget."""
        if not self._budgeted(provider, model):
            return
        now = self._clock()
        state = self._provider(provider)
        state.requests_this_minute.append(now)
        state.requests_today.append(now)

    def record_success(self, provider: str, model: str) -> None:
        state = self._model(provider, model)
        state.last_ok_at = self._clock()
        state.transient_strikes = 0

    def record_failure(self, provider: str, model: str, error: BaseException) -> None:
        """Apply what *error* says about *model* (and possibly its provider)."""
        if isinstance(error, SkipTierError):
            return  # never attempted
        state = self._model(provider, model)
        state.last_error = _describe(error)
        state.last_error_at = self._clock()

        if isinstance(error, RateLimitError):
            if provider == OPENROUTER and self._budgeted(provider, model) and _is_account_429(error):
                # The free-model caps are per account, not per model.
                self.cool_down_provider(
                    provider,
                    max(_OPENROUTER_ACCOUNT_429_SECONDS, error.retry_after or 0),
                    kind=RATE_LIMIT,
                    reason="HTTP 429 on a free model (account limit)",
                )
            else:
                self.cool_down(
                    provider, model, error.retry_after or _DEFAULT_429_SECONDS, reason="HTTP 429"
                )
        elif isinstance(error, AuthError):
            if error.status == 403 and not _KEY_RE.search(error.detail or ""):
                self.cool_down(
                    provider, model, _FORBIDDEN_COOLDOWN_SECONDS, kind=COOLDOWN, reason="HTTP 403"
                )
            else:
                self.cool_down_provider(
                    provider,
                    _AUTH_COOLDOWN_SECONDS,
                    kind=COOLDOWN,
                    reason=f"HTTP {error.status}: check the API key and its permissions",
                )
        elif isinstance(error, CreditsError):
            self.cool_down_provider(
                provider, _CREDITS_COOLDOWN_SECONDS, kind=COOLDOWN, reason="HTTP 402: out of credits"
            )
        elif isinstance(error, ModelGoneError):
            reason = f"HTTP {error.status}"
            if error.eol:
                state.eol = error.eol
                reason += f", end of life {error.eol}"
            if error.status == 410:
                self.mark_dead(provider, model, reason)
            else:
                seconds = _NOT_FOUND_DEAD_SECONDS_BY_PROVIDER.get(provider, _NOT_FOUND_DEAD_SECONDS)
                self.mark_dead(
                    provider, model, f"{reason}, retry in {_span(seconds)}", retry_after=seconds
                )
        elif isinstance(error, (TransientError, TimeoutError)):
            state.transient_strikes += 1
            if state.transient_strikes >= _TRANSIENT_STRIKES:
                state.transient_strikes = 0
                self.cool_down(
                    provider,
                    model,
                    _TRANSIENT_COOLDOWN_SECONDS,
                    kind=COOLDOWN,
                    reason=f"{_TRANSIENT_STRIKES} transient failures in a row",
                )

    def mark_dead(
        self,
        provider: str,
        model: str,
        reason: str,
        *,
        retry_after: float | None = None,
        eol: str | None = None,
    ) -> None:
        """Take *model* out of rotation, for the process or for *retry_after* seconds."""
        state = self._model(provider, model)
        if eol:
            state.eol = eol
        if state.dead_reason is None:
            scope = f"for {_span(retry_after)}" if retry_after else "for this process"
            logger.error(f"Model {self.key(provider, model)} marked dead {scope}: {reason}")
        state.dead_reason = reason or "unavailable"
        state.dead_until = self._clock() + retry_after if retry_after else None

    def revive(self, provider: str, model: str) -> bool:
        """Put a dead *model* back in rotation (the daily lifecycle check saw it live).

        Returns True when it was dead.
        """
        state = self._models.get(self.key(provider, model))
        if state is None or not state.dead_reason:
            return False
        logger.info(f"Model {self.key(provider, model)} is back in rotation after: {state.dead_reason}")
        state.dead_reason = None
        state.dead_until = None
        return True

    def cool_down(
        self,
        provider: str,
        model: str,
        seconds: float,
        *,
        kind: str = RATE_LIMIT,
        reason: str,
    ) -> None:
        state = self._model(provider, model)
        state.cooldown_until = self._clock() + seconds
        state.cooldown_kind = kind
        state.cooldown_reason = reason
        logger.warning(f"{reason} for {self.key(provider, model)}, cooling down for {_span(seconds)}")

    def cool_down_provider(
        self,
        provider: str,
        seconds: float,
        *,
        kind: str = COOLDOWN,
        reason: str,
    ) -> None:
        state = self._provider(provider)
        state.cooldown_until = self._clock() + seconds
        state.cooldown_kind = kind
        state.cooldown_reason = f"{provider}: {reason}"
        logger.warning(f"{reason}: every {provider} model is out for {_span(seconds)}")

    # ── /health ──────────────────────────────────────────────

    def get_status(self, *, include_errors: bool = False) -> dict:
        """Budgets and per-model state for /health.

        ``last_error`` quotes the upstream HTTP error body, so it is left
        out unless *include_errors*: the public /health must not echo
        provider or account messages. Every ``reason`` is this module's own
        wording (status codes, budgets, a parsed end-of-life date).
        """
        now = self._clock()
        providers: dict[str, dict] = {}
        for name in sorted({*self._budgets, *self._providers}):
            state = self._provider(name)
            self._trim(state, now)
            budget = self._budgets.get(name)
            cooling = now < state.cooldown_until
            providers[name] = {
                "rpm_used": len(state.requests_this_minute),
                "rpm_limit": budget.rpm if budget else None,
                "rpd_used": len(state.requests_today),
                "rpd_limit": budget.rpd if budget else None,
                "cooldown_until": state.cooldown_until if cooling else None,
                "cooldown_reason": state.cooldown_reason if cooling else None,
            }
        models: dict[str, dict] = {}
        for key, state in self._models.items():
            provider, _, model = key.partition(":")
            blocked = self.unavailable(provider, model)
            models[key] = {
                "available": blocked is None,
                "state": blocked.kind if blocked else "ok",
                "reason": blocked.reason if blocked else None,
                "dead_until": state.dead_until,
                "eol": state.eol,
                "cooldown_until": state.cooldown_until if now < state.cooldown_until else None,
                "last_error_at": state.last_error_at,
                "last_ok_at": state.last_ok_at,
            }
            if include_errors:
                models[key]["last_error"] = state.last_error
        return {"providers": providers, "models": models}

    # ── Internals ────────────────────────────────────────────

    def _budgeted(self, provider: str, model: str) -> bool:
        if provider not in self._budgets:
            return False
        # Paid OpenRouter ids (OPENROUTER_ALLOW_PAID) bill credits instead.
        return provider != OPENROUTER or is_free_model_id(model)

    def _model(self, provider: str, model: str) -> _ModelState:
        return self._models.setdefault(self.key(provider, model), _ModelState())

    def _provider(self, provider: str) -> _ProviderState:
        return self._providers.setdefault(provider, _ProviderState())

    def _revive(self, provider: str, model: str, state: _ModelState, now: float) -> None:
        """Clear a temporary death whose time is up."""
        if state.dead_reason and state.dead_until is not None and now >= state.dead_until:
            logger.info(
                f"Model {self.key(provider, model)} is back in rotation after: {state.dead_reason}"
            )
            state.dead_reason = None
            state.dead_until = None

    @staticmethod
    def _trim(state: _ProviderState, now: float) -> None:
        while state.requests_this_minute and state.requests_this_minute[0] <= now - 60:
            state.requests_this_minute.popleft()
        while state.requests_today and state.requests_today[0] <= now - 86_400:
            state.requests_today.popleft()
