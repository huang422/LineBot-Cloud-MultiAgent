"""Per-user rate limiting with sliding window."""

from __future__ import annotations

from collections import defaultdict, deque
from time import time

from src.config import get_settings


class RateLimitService:
    def __init__(self) -> None:
        settings = get_settings()
        self._max = settings.rate_limit_max_requests
        self._window = settings.rate_limit_window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._last_cleanup = time()
        self._cleanup_interval = 300  # cleanup every 5 minutes

    def check(self, user_id: str) -> tuple[bool, int]:
        """Check if user is within rate limit.

        Returns:
            (allowed: bool, remaining: int)
        """
        now = time()
        cutoff = now - self._window
        timestamps = self._requests[user_id]

        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        if len(timestamps) >= self._max:
            return False, 0

        timestamps.append(now)
        remaining = self._max - len(timestamps)

        # Periodic cleanup of stale user entries
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_stale(now)

        return True, remaining

    def _cleanup_stale(self, now: float) -> None:
        """Remove user entries with no recent requests."""
        cutoff = now - self._window
        stale_keys = [
            uid for uid, ts in self._requests.items()
            if not ts or ts[-1] <= cutoff
        ]
        for uid in stale_keys:
            del self._requests[uid]
        self._last_cleanup = now


_instance: RateLimitService | None = None


def get_rate_limit_service() -> RateLimitService:
    global _instance
    if _instance is None:
        _instance = RateLimitService()
    return _instance
