"""Per-key coalescing buffer with no timers of its own.

Owns one decision: *when* a key is ready to flush. It is deliberately
generic over what is being buffered — items are opaque, and a
caller-supplied coroutine does the actual persisting. That keeps the flush
policy testable on its own and stops write-batching concerns from leaking
into the memory document logic.

Nothing here schedules work. Cloud Run throttles CPU outside requests, so a
timer would not fire reliably (CONTEXT.md invariant 1). Instead:

- :meth:`PassiveMessageBuffer.enqueue` returns True when a key is ready (it
  reached the threshold, or its oldest item is older than the delay), and
  the caller queues a flush task for it. The request is repeated if no
  flush happened within ``delay`` seconds, so one lost task cannot strand
  a buffer;
- :meth:`PassiveMessageBuffer.due_keys` lists keys whose oldest item has
  waited longer than the delay, which the per-minute ``/internal/cron``
  sweep flushes.

A flush that fails or is cancelled puts its items back, so nothing is lost
before it is written. Each buffer is capped (``max_items``) so it stays
bounded while writes keep failing; past the cap the oldest items go.

Used by ``MemoryService`` to coalesce untriggered group messages into a
single Firestore write. See ADR-0003 for why passive recording exists and
ADR-0006 for how flushes are driven.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import time
from typing import Any

from src.utils.logger import logger

FlushCallback = Callable[[str, dict[str, Any], list[Any]], Awaitable[Any]]


class PassiveMessageBuffer:
    """Buffers items per key until a caller flushes them."""

    def __init__(
        self,
        *,
        enabled: bool,
        threshold: int,
        delay: int,
        flush: FlushCallback,
        clock: Callable[[], float] = time,
        max_items: int | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._threshold = max(1, int(threshold))
        self._delay = max(1, int(delay))
        self._flush_cb = flush
        self._clock = clock
        # Never below the threshold, or a buffer could not reach it.
        self._max_items = max(
            self._threshold,
            int(max_items) if max_items is not None else self._threshold * 4,
        )
        self._buffers: dict[str, dict[str, Any]] = {}
        # Buffers handed to the callback and not finished yet, so a discard
        # during the write also stops a failed flush from restoring what it
        # dropped.
        self._in_flight: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def max_items(self) -> int:
        return self._max_items

    def enqueue(self, key: str, item: Any, *, context: dict[str, Any] | None = None) -> bool:
        """Buffer one item; True when the caller should request a flush.

        A key is ready once it holds ``threshold`` items or its oldest item
        is ``delay`` seconds old. One True covers a burst: items arriving
        after it ride along with that flush. If the buffer is still here
        ``delay`` seconds after the last True (the flush task was lost, or
        ran on another instance), the next item asks again.
        """
        if not self._enabled or not key:
            return False

        now = self._clock()
        buf = self._buffers.get(key)
        if buf is None:
            buf = {"context": context or {}, "items": [], "first_at": now, "requested_at": None}
            self._buffers[key] = buf
        buf["items"].append(item)
        self._trim(key, buf)

        ready = len(buf["items"]) >= self._threshold or now - buf["first_at"] >= self._delay
        requested_at = buf["requested_at"]
        if not ready or (requested_at is not None and now - requested_at < self._delay):
            return False
        buf["requested_at"] = now
        return True

    def due_keys(self, now: float | None = None) -> list[str]:
        """Keys whose first pending item is at least ``delay`` seconds old."""
        now = self._clock() if now is None else now
        return [
            key
            for key, buf in self._buffers.items()
            if now - buf["first_at"] >= self._delay
        ]

    def context(self, key: str) -> dict[str, Any] | None:
        """The context stored with ``key``'s first item, or None if empty."""
        buf = self._buffers.get(key)
        return buf["context"] if buf and buf["items"] else None

    async def flush(self, key: str) -> Any:
        """Hand every pending item for ``key`` to the flush callback.

        Returns the callback's result, or None when nothing was pending. If
        the callback raises or is cancelled, the items go back in front of
        anything buffered meanwhile and the error propagates; a write that
        was cancelled after it reached the backend may then be repeated,
        which beats losing it (MemoryService skips the messages whose ids
        are already stored).
        """
        buf = self._buffers.pop(key, None)
        if not buf or not buf["items"]:
            return None
        self._in_flight[key] = buf
        try:
            return await self._flush_cb(key, buf["context"], buf["items"])
        except BaseException:
            keep = buf.pop("keep", None)
            if keep is not None:
                buf["items"] = [item for item in buf["items"] if keep(item)]
            if buf["items"]:
                self._restore(key, buf)
            raise
        finally:
            if self._in_flight.get(key) is buf:
                del self._in_flight[key]

    def _restore(self, key: str, buf: dict[str, Any]) -> None:
        newer = self._buffers.get(key)
        if newer is not None:
            buf["items"].extend(newer["items"])
        # Not requested any more: the next item may ask for a flush again.
        buf["requested_at"] = None
        self._buffers[key] = buf
        self._trim(key, buf)

    def _trim(self, key: str, buf: dict[str, Any]) -> None:
        overflow = len(buf["items"]) - self._max_items
        if overflow > 0:
            del buf["items"][:overflow]
            logger.warning(
                f"Passive buffer {key} is over its cap ({self._max_items}); dropped the "
                f"{overflow} oldest item(s) because flushes are not keeping up"
            )

    def discard(self, key: str, *, keep: Callable[[Any], bool] | None = None) -> None:
        """Drop pending items without persisting them (used by ``!new``).

        With *keep*, the items it accepts stay buffered. A flush in progress
        that then fails restores only those.
        """
        keep = keep or (lambda item: False)
        buf = self._buffers.pop(key, None)
        if buf is not None:
            buf["items"] = [item for item in buf["items"] if keep(item)]
            if buf["items"]:
                self._buffers[key] = buf
        in_flight = self._in_flight.get(key)
        if in_flight is not None:
            in_flight["keep"] = keep

    def pending_keys(self) -> list[str]:
        return list(self._buffers.keys())

    def pending_count(self, key: str) -> int:
        buf = self._buffers.get(key)
        return len(buf["items"]) if buf else 0

    async def close(self) -> None:
        """Flush everything still buffered, so shutdown does not lose items."""
        for key in self.pending_keys():
            try:
                await self.flush(key)
            except Exception as exc:
                logger.warning(f"Passive flush during close failed for {key}: {exc}")
