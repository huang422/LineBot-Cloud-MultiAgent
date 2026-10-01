"""Handlers for the work that runs after the webhook, one per task kind.

``handle_task`` is what ``POST /internal/tasks/{kind}`` calls, and what the
inline queue (and the inline fallback) calls directly, so a task behaves the
same wherever it runs. It never raises: the result is a bool, because Cloud
Tasks delivers each task once (``maxAttempts=1``) and a 5xx would only be
dropped anyway. See ADR-0006. An event turn must end
:data:`EVENT_NOTICE_SECONDS` before the handler budget: one that has not
delivered its answer by then is cut off, and the user gets
:data:`EVENT_TIMEOUT_TEXT` in the time left.

Kinds and payloads:

- ``event``         ``{"event": <LINE event>}`` → ``process_event``
- ``passive_flush`` ``{"key": "group::C…"}`` → ``memory.flush_passive``
- ``cache_image``   ``{"event": <LINE event>}`` → ``message_cache.cache_event_message``
- ``compact``       ``{"source_type", "chat_scope", "chat_id"}`` → ``memory.run_compaction``
- ``eol``           ``{}`` → ``lifecycle.run`` (the daily model end-of-life check)
- ``colab``         ``{"action": assign|launch|heartbeat|unassign|reconcile, …}``
                    → ``colab.run_action`` (the Colab controller, ADR-0010)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from src.pipeline.event_pipeline import PipelineDeps
from src.pipeline.event_pipeline import process_event as _default_process_event
from src.utils.dedupe import RecentKeys
from src.utils.logger import logger

# LINE redelivers and Cloud Tasks occasionally duplicates; both arrive
# within minutes, so 15 minutes of memory is plenty.
EVENT_DEDUPE_TTL_SECONDS = 15 * 60
EVENT_DEDUPE_MAX_KEYS = 4096
# The end of the handler budget kept for telling the user that their turn
# ran out of time (a reply that fails fast, then one push).
EVENT_NOTICE_SECONDS = 10.0
EVENT_TIMEOUT_TEXT = "⚠️ 這次處理時間過長，請稍後再試。"


def new_event_dedupe() -> RecentKeys:
    return RecentKeys(ttl_seconds=EVENT_DEDUPE_TTL_SECONDS, max_size=EVENT_DEDUPE_MAX_KEYS)


def event_dedupe_key(event: dict) -> str:
    """``webhookEventId``, else the message id, else "" (no dedupe)."""
    event_id = str(event.get("webhookEventId") or "").strip()
    if event_id:
        return event_id
    message = event.get("message")
    if isinstance(message, dict):
        message_id = str(message.get("id") or "").strip()
        if message_id:
            return f"message:{message_id}"
    return ""


@dataclass
class TaskDeps:
    """What the task handlers need; ``pipeline`` carries memory and cache."""

    pipeline: PipelineDeps
    budget_seconds: float = 280.0
    # Separate from the webhook's dedupe set: the webhook marks an event
    # seen before queueing it, so sharing one set would drop every task.
    seen_events: RecentKeys = field(default_factory=new_event_dedupe)
    process_event: Callable[[dict, PipelineDeps], Awaitable[None]] = _default_process_event
    # The model lifecycle checker (src/services/model_lifecycle.py).
    lifecycle: Any = None
    # The Colab controller (src/services/colab/controller.py).
    colab: Any = None


def _event_from(payload: dict[str, Any]) -> dict | None:
    event = payload.get("event")
    return event if isinstance(event, dict) else None


async def _handle_event(payload: dict[str, Any], deps: TaskDeps) -> bool:
    event = _event_from(payload)
    if event is None:
        logger.warning("Task event: payload has no event object")
        return False
    key = event_dedupe_key(event)
    if key and not deps.seen_events.add(key):
        logger.info(f"Task event: duplicate delivery of {key} skipped")
        return True

    delivered = False
    send_response = deps.pipeline.send_response

    async def tracked_send_response(request, response) -> bool:
        nonlocal delivered
        delivered = await send_response(request, response)
        return delivered

    pipeline = replace(deps.pipeline, send_response=tracked_send_response)
    try:
        async with asyncio.timeout(max(deps.budget_seconds - EVENT_NOTICE_SECONDS, 0)):
            await deps.process_event(event, pipeline)
    except TimeoutError:
        if not delivered:
            await _send_timeout_notice(event, deps.pipeline.line)
        raise
    return True


async def _send_timeout_notice(event: dict, line: Any) -> None:
    """Best effort: tell the sender of *event* that the turn ran out of time."""
    source = event.get("source") or {}
    chat_id = source.get("groupId") or source.get("roomId") or source.get("userId", "")
    try:
        await line.send_text(event.get("replyToken", ""), chat_id, EVENT_TIMEOUT_TEXT)
    except Exception as exc:
        logger.error(f"Task event: the timeout notice was not delivered: {exc}")


async def _handle_passive_flush(payload: dict[str, Any], deps: TaskDeps) -> bool:
    key = str(payload.get("key") or "").strip()
    if not key:
        logger.warning("Task passive_flush: payload has no key")
        return False
    await deps.pipeline.memory.flush_passive(key)
    return True


async def _handle_cache_image(payload: dict[str, Any], deps: TaskDeps) -> bool:
    event = _event_from(payload)
    if event is None:
        logger.warning("Task cache_image: payload has no event object")
        return False
    await deps.pipeline.message_cache.cache_event_message(event)
    return True


async def _handle_compact(payload: dict[str, Any], deps: TaskDeps) -> bool:
    chat_id = str(payload.get("chat_id") or "").strip()
    if not chat_id:
        logger.warning("Task compact: payload has no chat_id")
        return False
    await deps.pipeline.memory.run_compaction(
        source_type=str(payload.get("source_type") or "user"),
        chat_scope=str(payload.get("chat_scope") or ""),
        chat_id=chat_id,
    )
    return True


async def _handle_eol(payload: dict[str, Any], deps: TaskDeps) -> bool:
    if deps.lifecycle is None:
        logger.warning("Task eol: no model lifecycle checker is configured")
        return False
    await deps.lifecycle.run()
    return True


async def _handle_colab(payload: dict[str, Any], deps: TaskDeps) -> bool:
    if deps.colab is None:
        logger.warning("Task colab: Colab is not enabled here")
        return False
    action = payload.get("action")
    if not isinstance(action, str) or not action:
        logger.warning("Task colab: payload has no action")
        return False
    return await deps.colab.run_action(action, payload)


_HANDLERS: dict[str, Callable[[dict[str, Any], TaskDeps], Awaitable[bool]]] = {
    "event": _handle_event,
    "passive_flush": _handle_passive_flush,
    "cache_image": _handle_cache_image,
    "compact": _handle_compact,
    "eol": _handle_eol,
    "colab": _handle_colab,
}

TASK_KINDS: frozenset[str] = frozenset(_HANDLERS)


async def handle_task(kind: str, payload: dict[str, Any], deps: TaskDeps) -> bool:
    """Run one task within the handler budget; True when it completed."""
    handler = _HANDLERS.get(kind)
    if handler is None:
        logger.warning(f"Task kind {kind!r} is unknown; ignored")
        return False
    try:
        async with asyncio.timeout(deps.budget_seconds):
            return await handler(payload, deps)
    except TimeoutError:
        logger.error(f"Task {kind} exceeded its {deps.budget_seconds:.0f}s budget; cancelled")
    except Exception as exc:
        logger.error(f"Task {kind} failed: {exc}", exc_info=True)
    return False
