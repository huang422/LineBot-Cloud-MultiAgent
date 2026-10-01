"""Debounced, per-key coalescing buffer.

Owns one decision: *when* to flush. It is deliberately generic over what
is being buffered — items are opaque, and a caller-supplied coroutine does
the actual persisting. That keeps the flush policy (threshold reached, or
delay elapsed) testable on its own and stops write-batching concerns from
leaking into the memory document logic.

Used by ``MemoryService`` to coalesce untriggered group messages into a
single Firestore write. See ADR-0003 for why passive recording exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from src.utils.logger import logger

FlushCallback = Callable[[str, dict[str, Any], list[Any]], Awaitable[None]]


class PassiveMessageBuffer:
    """Buffers items per key, flushing on threshold or after a delay."""

    def __init__(
        self,
        *,
        enabled: bool,
        threshold: int,
        delay: int,
        flush: FlushCallback,
    ) -> None:
        self._enabled = bool(enabled)
        self._threshold = max(1, int(threshold))
        self._delay = max(1, int(delay))
        self._flush_cb = flush
        self._buffers: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def threshold(self) -> int:
        return self._threshold

    def enqueue(self, key: str, item: Any, *, context: dict[str, Any] | None = None) -> None:
        """Buffer one item, scheduling a flush if this is the first pending one.

        Safe to call from a synchronous context: with no running event loop
        the item is buffered but no timer is scheduled, so a later enqueue
        (or an explicit flush) picks it up.
        """
        if not self._enabled or not key:
            return

        buf = self._buffers.setdefault(key, {"context": context or {}, "items": []})
        buf["items"].append(item)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # caller is sync; nothing to schedule against

        if len(buf["items"]) >= self._threshold:
            existing = self._tasks.pop(key, None)
            if existing and not existing.done():
                existing.cancel()
            self._tasks[key] = loop.create_task(self.flush(key))
            return

        if key in self._tasks:
            return  # debounce timer already running

        async def _delayed_flush() -> None:
            try:
                await asyncio.sleep(self._delay)
                await self.flush(key)
            except asyncio.CancelledError:
                pass

        self._tasks[key] = loop.create_task(_delayed_flush())

    async def flush(self, key: str) -> None:
        """Hand every pending item for ``key`` to the flush callback."""
        buf = self._buffers.pop(key, None)
        self._tasks.pop(key, None)
        if not buf or not buf["items"]:
            return
        await self._flush_cb(key, buf["context"], buf["items"])

    def discard(self, key: str) -> None:
        """Drop pending items without persisting them (used by ``!new``)."""
        self._buffers.pop(key, None)
        task = self._tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    def pending_keys(self) -> list[str]:
        return list(self._buffers.keys())

    def pending_count(self, key: str) -> int:
        buf = self._buffers.get(key)
        return len(buf["items"]) if buf else 0

    async def close(self) -> None:
        """Flush everything still buffered, so shutdown does not lose items."""
        for key in self.pending_keys():
            task = self._tasks.pop(key, None)
            if task and not task.done():
                task.cancel()
            try:
                await self.flush(key)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Passive flush during close failed for {key}: {exc}")
        self._tasks.clear()
