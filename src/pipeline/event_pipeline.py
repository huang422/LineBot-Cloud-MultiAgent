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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.handlers.webhook_handler import (
    enrich_request as _default_enrich_request,
)
from src.handlers.webhook_handler import (
    extract_text as _default_extract_text,
)
from src.models.agent_request import InputType
from src.processors.input_processor import process_input as _default_process_input
from src.processors.output_processor import send_response as _default_send_response
from src.providers.fallback_chain import (
    AllModelsRateLimitedError,
    AllProvidersFailedError,
)
from src.services.memory_service import MemoryServiceError
from src.utils.logger import logger


@dataclass
class PipelineDeps:
    """Collaborators the event pipeline needs, injected explicitly.

    The four ``*_fn`` fields default to the real implementations; they are
    parameters so tests can substitute them without monkeypatching module
    globals.
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


# ── Small pure helpers ───────────────────────────────────────


def is_new_chat_command(text: str) -> bool:
    return text.strip().lower() == "!new"


def request_has_image_context(request) -> bool:
    return bool(
        request.image_base64
        or request.quoted_image_base64
        or request.quoted_image_url
    )


def get_request_block_message(request) -> str | None:
    if request.rate_limited:
        return "⚠️ 請求太頻繁，請稍後再試。"
    if not request.text and not request_has_image_context(request):
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
    text = (response.text or "").strip()
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
        if request_has_image_context(request):
            request.input_type = InputType.IMAGE_TEXT if request.text else InputType.IMAGE
        else:
            request.input_type = InputType.TEXT


def record_group_message(event: dict, memory) -> None:
    """Buffer a non-triggered group/room message into the passive queue.

    Triggered messages skip this: ``record_interaction`` already persists
    the user's text to ``memory.recent_messages``.
    """
    if event.get("type") != "message":
        return

    message = event.get("message", {})
    msg_type = message.get("type", "")
    source = event.get("source", {})
    user_id = source.get("userId", "")
    src_type = source.get("type", "")
    group_id = source.get("groupId") or source.get("roomId") or user_id

    if not user_id or not group_id:
        return
    if src_type not in {"group", "room"}:
        return

    text = ""
    if msg_type == "text":
        text = message.get("text", "").strip()
    elif msg_type == "image":
        text = "[圖片]"
    elif msg_type == "sticker":
        text = "[貼圖]"
    elif msg_type == "audio":
        text = "[語音]"

    if not text:
        return

    try:
        memory.enqueue_passive_message(
            source_type=src_type,
            chat_scope="multi",
            chat_id=group_id,
            user_id=user_id,
            text=text,
        )
    except Exception as exc:
        logger.debug(f"enqueue_passive_message failed: {exc}")


# ── The pipeline ─────────────────────────────────────────────


async def process_event(event: dict, deps: PipelineDeps) -> None:
    """Process a single LINE event end to end."""
    source = event.get("source", {})
    chat_id = source.get("groupId") or source.get("roomId") or source.get("userId", "")
    reply_token = event.get("replyToken", "")
    line = deps.line

    try:
        request = await deps.process_input(event)
        if request is None:
            return

        apply_cleaned_text(request, event, deps.extract_text)

        if is_new_chat_command(request.text):
            await deps.memory.clear_chat(
                source_type=request.source_type,
                chat_scope=request.chat_scope,
                chat_id=request.group_id,
            )
            await line.send_text(
                request.reply_token, chat_id, "Let's start a new chat!"
            )
            return

        # Loading animation is free; send it before any slow work.
        await line.send_loading_animation(chat_id)

        deps.message_cache.cache_processed_request(event, request)

        request = await deps.enrich_request(request)
        block_message = get_request_block_message(request)
        if block_message:
            await line.send_text(request.reply_token, chat_id, block_message)
            return

        await _touch_user_profile_safe(deps, request)

        decision = await deps.orchestrator.route(request)
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

        response = await agent.process(request)

        sent = await deps.send_response(request, response)
        if not sent:
            logger.warning(
                f"[{request.request_id}] Response could not be delivered "
                "(reply failed or push budget unavailable)"
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
            )
        except MemoryServiceError as e:
            logger.error(f"Memory interaction record failed: {e}")

    except AllModelsRateLimitedError:
        logger.error("All models rate limited!")
        await _send_error_message(line, reply_token, chat_id, "⚠️ AI 模型暫時忙碌中，請稍後再試。")

    except AllProvidersFailedError as e:
        logger.error(f"All provider targets failed: {e}", exc_info=True)
        await _send_error_message(line, reply_token, chat_id, "⚠️ AI 服務暫時不可用，請稍後再試。")

    except Exception as e:
        logger.error(f"Event processing error: {e}", exc_info=True)
        await _send_error_message(line, reply_token, chat_id, "❌ 處理請求時發生錯誤，請稍後再試。")


async def _send_error_message(line, reply_token: str, chat_id: str, text: str) -> None:
    """Best-effort error message to the user."""
    try:
        await line.send_text(reply_token, chat_id, text)
    except Exception as e:
        logger.error(f"Failed to deliver error message: {e}", exc_info=True)


async def _touch_user_profile_safe(deps: PipelineDeps, request) -> None:
    """Resolve the LINE display name and upsert the per-user profile.

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
