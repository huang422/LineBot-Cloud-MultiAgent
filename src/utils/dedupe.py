"""A small in-process TTL + LRU set for "have I seen this key recently?".

Used to drop LINE webhook redeliveries and Cloud Tasks duplicate
deliveries (both are at-least-once), and Colab registration nonces
already used. It is process-local on purpose: the
service runs with ``max-instances=1``, so one process sees every
delivery. It does not dedupe across instances. See ADR-0006.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from time import monotonic


class RecentKeys:
    """Remembers keys for ``ttl_seconds``, holding at most ``max_size``.

    The oldest key is evicted first when the set is full, so memory stays
    bounded however many distinct keys arrive.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 900.0,
        max_size: int = 4096,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._ttl = max(0.0, float(ttl_seconds))
        self._max_size = max(1, int(max_size))
        self._clock = clock
        self._seen: OrderedDict[str, float] = OrderedDict()

    def add(self, key: str) -> bool:
        """Record ``key``; False when it was already recorded and unexpired."""
        now = self._clock()
        self._expire(now)
        if key in self._seen:
            return False
        self._seen[key] = now
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)
        return True

    def discard(self, key: str) -> None:
        self._seen.pop(key, None)

    def clear(self) -> None:
        self._seen.clear()

    def __contains__(self, key: object) -> bool:
        self._expire(self._clock())
        return key in self._seen

    def __len__(self) -> int:
        self._expire(self._clock())
        return len(self._seen)

    def _expire(self, now: float) -> None:
        # Insertion order is age order, so expiry stops at the first fresh key.
        while self._seen:
            key, added_at = next(iter(self._seen.items()))
            if now - added_at < self._ttl:
                break
            self._seen.pop(key, None)
