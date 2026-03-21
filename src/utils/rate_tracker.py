"""Per-model rate limit tracking for OpenRouter free models.

Free models have limits of ~20 RPM and ~200 RPD.  This tracker monitors
usage via response headers and prevents wasting calls on exhausted models.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from src.utils.logger import logger


@dataclass
class _ModelState:
    """Tracks rate-limit state for a single model."""

    requests_this_minute: deque[float] = field(default_factory=deque)
    requests_today: deque[float] = field(default_factory=deque)
    blocked_until: float = 0.0
    rpm_limit: int = 20
    rpd_limit: int = 200


class RateTracker:
    """In-memory per-model rate-limit tracker."""

    def __init__(self) -> None:
        self._models: dict[str, _ModelState] = {}

    def _state(self, model: str) -> _ModelState:
        if model not in self._models:
            self._models[model] = _ModelState()
        return self._models[model]

    def _cleanup(self, state: _ModelState, now: float) -> None:
        """Remove expired timestamps from deques."""
        one_min_ago = now - 60
        while state.requests_this_minute and state.requests_this_minute[0] <= one_min_ago:
            state.requests_this_minute.popleft()

        one_day_ago = now - 86400
        while state.requests_today and state.requests_today[0] <= one_day_ago:
            state.requests_today.popleft()

    def is_available(self, model: str) -> bool:
        """Check if a model is likely under its rate limit."""
        state = self._state(model)
        now = time.time()

        if now < state.blocked_until:
            return False

        self._cleanup(state, now)

        return (
            len(state.requests_this_minute) < state.rpm_limit
            and len(state.requests_today) < state.rpd_limit
        )

    def record_request(self, model: str) -> None:
        """Record a successful request."""
        state = self._state(model)
        now = time.time()
        state.requests_this_minute.append(now)
        state.requests_today.append(now)

    def record_limit_hit(self, model: str, retry_after: int | None = None) -> None:
        """Record a 429 rate-limit hit."""
        state = self._state(model)
        cooldown = retry_after if retry_after else 60
        state.blocked_until = time.time() + cooldown
        logger.warning(f"Rate limit hit for {model}, blocked for {cooldown}s")

    def update_from_headers(self, model: str, headers: dict) -> None:
        """Update state from OpenRouter response headers."""
        state = self._state(model)

        remaining = headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                remaining_int = int(remaining)
                if remaining_int <= 0:
                    retry = headers.get("retry-after")
                    reset = headers.get("x-ratelimit-reset")

                    if retry:
                        # retry-after is seconds to wait
                        cooldown = int(retry)
                    elif reset:
                        # x-ratelimit-reset is a Unix timestamp; compute delta
                        reset_ts = float(reset)
                        now = time.time()
                        if reset_ts > now:
                            cooldown = int(reset_ts - now)
                        else:
                            cooldown = 60
                    else:
                        cooldown = 60

                    cooldown = min(cooldown, 600)  # cap at 10 minutes
                    state.blocked_until = time.time() + cooldown
            except (ValueError, TypeError):
                pass

    def get_status(self) -> dict:
        """Return status for health endpoint."""
        now = time.time()
        result = {}
        for model, state in self._models.items():
            self._cleanup(state, now)
            result[model] = {
                "rpm_used": len(state.requests_this_minute),
                "rpd_used": len(state.requests_today),
                "available": self.is_available(model),
                "blocked_until": state.blocked_until if now < state.blocked_until else None,
            }
        return result
