"""Turning one LINE event into one delivered reply.

The pipeline is pure orchestration: it owns the *order* of steps and the
error handling around them, and nothing else. Every collaborator arrives
through :class:`PipelineDeps` rather than a module-level singleton, so the
whole path can be driven in a test without patching global state — and so
production code does not need defensive ``hasattr`` guards to tolerate
half-built test doubles.

See CONTEXT.md for where this sits in the request path.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.agents.routing_rules import has_image_context
from src.handlers.webhook_handler import (
    enrich_request as _default_enrich_request,
)
from src.handlers.webhook_handler import (
    extract_text as _default_extract_text,
)
from src.models.agent_request import (
    DEFAULT_REPLY_WINDOW_SECONDS,
    DEFAULT_TURN_DEADLINE_SECONDS,
    InputType,
)
from src.processors.input_processor import event_received_at
from src.processors.input_processor import process_input as _default_process_input
from src.processors.output_processor import send_response as _default_send_response
from src.providers.fallback_chain import (
    AllModelsRateLimitedError,
    AllProvidersFailedError,
)
from src.services.colab.admin import parse_command, run_command
from src.services.memory_service import MemoryServiceError, chat_key
from src.utils import request_trace
from src.utils.line_text import to_line_text
from src.utils.logger import logger
from src.utils.validators import sanitize_input


@dataclass
class PipelineDeps:
    """Collaborators the event pipeline needs, injected explicitly.

    The function fields (and ``clock``) default to the real implementations;
    they are parameters so tests can substitute them without monkeypatching
    module globals. ``reply_window_seconds`` and ``turn_deadline_seconds``
    are the Settings fields of the same names.
    """

    line: Any
    memory: Any
    message_cache: Any
    orchestrator: Any
    get_agent: Callable[[str], Any]
    process_input: Callable[[dict], Awaitable[Any]] = _default_process_input
    enrich_request: Callable[[Any], Awaitable[Any]] = _default_enrich_request
    extract_text: Callable[[dict], str] = _default_extract_text
    send_response: Callable[[Any, Any], Awaitable[bool]] = _default_send_response
    reply_window_seconds: float = DEFAULT_REPLY_WINDOW_SECONDS
    turn_deadline_seconds: float = DEFAULT_TURN_DEADLINE_SECONDS
    clock: Callable[[], float] = time.time
    # The Colab session endpoint (None unless COLAB_ENABLED) and who may
    # run !colab (LINE_ADMIN_USER_IDS). ADR-0010.
    colab: Any = None
    admin_user_ids: frozenset[str] = frozenset()
    # MODEL_PATH_PREFERENCE, for the summary line ("api-only" without colab).
    model_path_preference: str = "colab"


# ── Small pure helpers ───────────────────────────────────────


def is_new_chat_command(text: str) -> bool:
    return text.strip().lower() == "!new"


def get_request_block_message(request) -> str | None:
    if request.rate_limited:
        return "⚠️ 請求太頻繁，請稍後再試。"
    if not request.text and not has_image_context(request):
        return "請附上想問的內容，或直接引用訊息／圖片再提問。"
    return None


def build_user_memory_text(request) -> str:
    text = request.text.strip()
    if text:
        return text
    if request.quoted_image_base64 or request.quoted_image_url:
        return "[使用者引用圖片]"
    if request.input_type in (InputType.IMAGE, InputType.IMAGE_TEXT) or request.image_base64:
        return "[使用者傳送圖片]"
    return ""


def build_assistant_memory_text(response) -> str:
    """The reply as the user saw it: plain text, without Markdown or LaTeX.

    History is fed back to the models, so storing the raw reply would
    show them their own ``**`` and ``\\( \\)`` to copy.
    """
    text = to_line_text(response.text or "").strip()
    if text:
        return text
    if response.output_format == "image":
        return "[已傳送圖片]"
    if response.output_format == "voice":
        return "[已傳送語音]"
    return ""


def apply_cleaned_text(request, event: dict, extract_text) -> None:
    """Replace raw text with the cleaned form (drops the ``!hej`` prefix)."""
    if request.text or event.get("message", {}).get("type") == "text":
        request.text = extract_text(event)
        if has_image_context(request):
            request.input_type = InputType.IMAGE_TEXT if request.text else InputType.IMAGE
        else:
            request.input_type = InputType.TEXT


def record_group_message(event: dict, memory) -> str | None:
    """Buffer a non-triggered group/room message into the passive queue.

    Triggered messages skip this: ``record_interaction`` already persists
    the user's text to ``memory.recent_messages``.

    Returns the chat's buffer key when this message brought the buffer to
    its flush threshold (the caller queues a ``passive_flush`` task for
    it), otherwise None.
    """
    if event.get("type") != "message":
        return None

    message = event.get("message", {})
    msg_type = message.get("type", "")
    source = event.get("source", {})
    user_id = source.get("userId", "")
    src_type = source.get("type", "")
    group_id = source.get("groupId") or source.get("roomId") or user_id

    if not user_id or not group_id:
        return None
    if src_type not in {"group", "room"}:
        return None

    text = ""
    if msg_type == "text":
        # Group chatter becomes history and long-term memory for later
        # turns, so it gets the same cleaning as a triggered message.
        text = sanitize_input(message.get("text", ""))
    elif msg_type == "image":
        text = "[圖片]"
    elif msg_type == "sticker":
        text = "[貼圖]"
    elif msg_type == "audio":
        text = "[語音]"

    if not text:
        return None

    try:
        reached_threshold = memory.enqueue_passive_message(
            source_type=src_type,
            chat_scope="multi",
            chat_id=group_id,
            user_id=user_id,
            text=text,
            created_at=event_received_at(event),
        )
    except Exception as exc:
        logger.debug(f"enqueue_passive_message failed: {exc}")
        return None
    return chat_key(src_type, group_id) if reached_threshold else None


# ── The pipeline ─────────────────────────────────────────────


async def process_event(event: dict, deps: PipelineDeps) -> None:
    """Process a single LINE event end to end."""
    source = event.get("source", {})
    chat_id = source.get("groupId") or source.get("roomId") or source.get("userId", "")
    reply_token = event.get("replyToken", "")
    line = deps.line
    demand: asyncio.Task | None = None
    # One summary line per handled event (request_trace); a ContextVar, so
    # concurrent events never share a trace.
    trace = request_trace.RequestTrace(
        preference=deps.model_path_preference if deps.colab is not None else "api-only"
    )
    trace_token = request_trace.activate(trace)

    _warn_if_reply_window_passed(event, deps)

    try:
        request = await deps.process_input(event)
        if request is None:
            return
        trace.request_id = request.request_id
        request.reply_window_seconds = deps.reply_window_seconds
        request.turn_deadline_seconds = deps.turn_deadline_seconds

        apply_cleaned_text(request, event, deps.extract_text)

        if is_new_chat_command(request.text):
            await deps.memory.clear_chat(
                source_type=request.source_type,
                chat_scope=request.chat_scope,
                chat_id=request.group_id,
                up_to=request.received_at,
            )
            await line.send_text(
                request.reply_token, chat_id, "Let's start a new chat!"
            )
            return

        command = parse_command(request.text)
        if command is not None:
            reply = await run_command(
                command,
                request,
                colab=deps.colab,
                admin_user_ids=deps.admin_user_ids,
                clock=deps.clock,
            )
            await line.send_text(request.reply_token, chat_id, reply)
            return

        # Loading animation is free; send it before any slow work.
        await line.send_loading_animation(chat_id)

        deps.message_cache.cache_processed_request(event, request)

        await _flush_passive_before_reply(deps, request)
        request = await deps.enrich_request(request)
        block_message = get_request_block_message(request)
        if block_message:
            await line.send_text(request.reply_token, chat_id, block_message)
            return

        # Colab demand (a message that passed the rate limit and the block
        # checks) is written alongside the answer and joined only once the
        # answer is out: nothing waits for it.
        demand = asyncio.create_task(_note_colab_demand(deps))
        # The router never reads the facts (ADR-0004), so they load while it
        # classifies; only the agent waits for them.
        facts = asyncio.create_task(_load_profile_facts(deps, request))
        trace.handled = True
        try:
            decision = await deps.orchestrator.route(request)
        except BaseException:
            facts.cancel()
            raise
        await facts
        trace.note_decision(decision)
        request.target_agent = decision.agent
        request.output_format = decision.output_format
        request.task_description = decision.task_description
        request.routing_reasoning = decision.reasoning
        request.disable_thinking = decision.disable_thinking

        logger.info(
            f"[{request.request_id}] → {decision.agent} "
            f"(output={decision.output_format}, "
            f"thinking={'off' if decision.disable_thinking else 'on'})"
        )

        agent = deps.get_agent(decision.agent)
        if agent is None:
            await line.send_text(
                request.reply_token, chat_id, f"Agent '{decision.agent}' 尚未實作。"
            )
            return

        with request_trace.phase(request_trace.ANSWER):
            response = await agent.process(request)

        sent = await deps.send_response(request, response)
        trace.delivered = sent
        _log_summary(trace)
        if not sent:
            logger.warning(
                f"[{request.request_id}] Response could not be delivered "
                "(the reply and the push both failed)"
            )
            return

        try:
            await deps.memory.record_interaction(
                source_type=request.source_type,
                chat_scope=request.chat_scope,
                chat_id=request.group_id,
                user_id=request.user_id,
                user_text=build_user_memory_text(request),
                assistant_text=build_assistant_memory_text(response),
                agent_name=request.target_agent,
                output_format=request.output_format,
                task_description=request.task_description,
                routing_reasoning=request.routing_reasoning,
                disable_thinking=request.disable_thinking,
                user_created_at=request.received_at,
            )
        except MemoryServiceError as e:
            logger.error(f"Memory interaction record failed: {e}")

        # Metadata only, so it waits until the user has the answer.
        await _touch_user_profile_safe(deps, request)

    except AllModelsRateLimitedError as e:
        logger.error("All models rate limited!")
        await _send_error_message(line, reply_token, chat_id, "⚠️ AI 模型暫時忙碌中，請稍後再試。")
        _log_summary(trace, error=e)

    except AllProvidersFailedError as e:
        logger.error(f"All provider targets failed: {e}", exc_info=True)
        await _send_error_message(line, reply_token, chat_id, "⚠️ AI 服務暫時不可用，請稍後再試。")
        _log_summary(trace, error=e)

    except Exception as e:
        logger.error(f"Event processing error: {e}", exc_info=True)
        # Once the answer is out, an error message would be a second reply.
        if not trace.delivered:
            await _send_error_message(line, reply_token, chat_id, "❌ 處理請求時發生錯誤，請稍後再試。")
        _log_summary(trace, error=e)

    except BaseException as e:  # cancelled (the task's time ran out)
        _log_summary(trace, error=e)
        raise

    finally:
        # A routed event that ended without a reply (no such agent).
        if trace.handled:
            _log_summary(trace)
        request_trace.deactivate(trace_token)
        if demand is not None:
            # Bounded by ColabEndpoint.note_demand; a cancelled turn cancels it.
            await asyncio.gather(demand, return_exceptions=True)


def _log_summary(trace: request_trace.RequestTrace, *, error: BaseException | None = None) -> None:
    """Log the event's summary line once (request_trace.summary_line).

    An error is named by its class (and its last provider error's), never
    its message; one before routing is still logged.
    """
    if trace.logged:
        return
    if error is not None:
        trace.handled = True
        trace.error = _error_label(error)
    trace.logged = True
    logger.info(request_trace.summary_line(trace))


def _error_label(error: BaseException) -> str:
    """``AllProvidersFailedError(TransientError 503)``: the class names and status only."""
    label = type(error).__name__
    last = getattr(error, "last_error", None)
    if isinstance(last, BaseException):
        status = getattr(last, "status", None)
        label += f"({type(last).__name__}{f' {status}' if isinstance(status, int) and status else ''})"
    return label


async def _flush_passive_before_reply(deps: PipelineDeps, request) -> None:
    """Write the group's buffered untriggered messages before the reply.

    They came before this trigger, so they belong in the context the reply
    is built from, and in the window *before* the turn ``record_interaction``
    appends; flushing later would store them after the bot's answer.
    Compaction is left to ``record_interaction``, so the reply does not wait
    on it. Best effort: ``flush_passive`` keeps a failed batch buffered.
    """
    if request.source_type not in {"group", "room"} or not request.group_id:
        return
    key = chat_key(request.source_type, request.group_id)
    try:
        await deps.memory.flush_passive(key, compact=False)
    except Exception as exc:
        logger.warning(f"[{request.request_id}] Passive flush before the reply failed: {exc}")


def _warn_if_reply_window_passed(event: dict, deps: PipelineDeps) -> None:
    """Log when work starts after the reply window has passed.

    Nothing acts on this: ``send_response`` still tries the reply first
    and pushes only when LINE refuses it.
    """
    now = deps.clock()
    age = now - event_received_at(event, now=now)
    if age > deps.reply_window_seconds:
        logger.warning(
            f"Event processing started {age:.0f}s after the message "
            f"(reply window {deps.reply_window_seconds:.0f}s); the reply token has "
            "likely expired, so delivery will likely fall back to push"
        )


async def _send_error_message(line, reply_token: str, chat_id: str, text: str) -> None:
    """Best-effort error message to the user."""
    try:
        await line.send_text(reply_token, chat_id, text)
    except Exception as e:
        logger.error(f"Failed to deliver error message: {e}", exc_info=True)


async def _note_colab_demand(deps: PipelineDeps) -> None:
    """Tell the Colab controller a message wanted it (MODEL_PATH_PREFERENCE=colab).

    At most one write a minute (``ColabEndpoint.note_demand``), never with
    MODEL_PATH_PREFERENCE=api (Colab then starts only on ``!colab on``),
    and never an error: the reply does not depend on it.
    """
    colab = deps.colab
    if colab is None or not colab.auto_start:
        return
    try:
        await colab.note_demand()
    except Exception as exc:
        logger.warning(f"Colab demand not recorded: {exc}")


async def _load_profile_facts(deps: PipelineDeps, request) -> None:
    """Attach the sender's stored profile facts for the answering agents.

    Read through the memory service's profile cache, while the router
    classifies: a cache miss delays the reply only by whatever of the read
    outlasts routing. Best effort: a failed read is logged and the reply
    goes ahead without facts.
    """
    user_id = (request.user_id or "").strip()
    if not user_id:
        return
    try:
        request.user_profile_facts = await deps.memory.load_profile_facts(user_id)
    except Exception as exc:
        logger.warning(f"[{request.request_id}] Profile facts not loaded: {exc}")


async def _touch_user_profile_safe(deps: PipelineDeps, request) -> None:
    """Resolve the LINE display name and upsert the per-user profile.

    Runs after the reply went out; the memory service writes only when
    the name changed or the last write is a day old.

    Failures are swallowed: the profile is metadata, not on the critical
    path for replying. Logged at debug because a user who has not added
    the bot as a friend cannot be resolved, which is normal.
    """
    user_id = (request.user_id or "").strip()
    if not user_id:
        return

    display_name = ""
    try:
        display_name = await deps.line.fetch_display_name(
            source_type=request.source_type,
            chat_id=request.group_id,
            user_id=user_id,
        )
    except Exception as exc:
        logger.debug(f"Display name lookup skipped for {user_id[:8]}…: {exc}")

    try:
        await deps.memory.touch_user_profile(
            user_id=user_id,
            display_name=display_name,
            source_type=request.source_type,
            chat_id=request.group_id,
        )
    except Exception as exc:
        logger.debug(f"Profile touch skipped for {user_id[:8]}…: {exc}")
