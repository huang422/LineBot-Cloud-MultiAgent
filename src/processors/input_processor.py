"""Input processor: detect input type, download images, build AgentRequest."""

from __future__ import annotations

import json
import time

from src.models.agent_request import AgentRequest, InputType
from src.services.image_service import process_image
from src.services.line_service import get_line_service
from src.services.message_cache_service import get_message_cache_service
from src.utils.logger import logger
from src.utils.validators import check_prompt_injection, sanitize_input


def event_received_at(event: dict, now: float | None = None) -> float:
    """Epoch seconds from the LINE event's ``timestamp`` (ms), else now."""
    raw = event.get("timestamp")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return float(raw) / 1000.0
    return time.time() if now is None else now


async def process_input(event: dict) -> AgentRequest | None:
    """Convert a LINE webhook event into an AgentRequest.

    Returns None if the event should be ignored.
    """
    event_type = event.get("type")
    if event_type != "message":
        return None

    message = event.get("message", {})
    msg_type = message.get("type")
    source = event.get("source", {})
    reply_token = event.get("replyToken", "")

    user_id = source.get("userId", "")
    source_type = str(source.get("type", "user")).strip().lower() or "user"
    group_id = source.get("groupId", "") or source.get("roomId", "") or user_id
    chat_scope = "user" if source_type == "user" else "multi"

    request = AgentRequest(
        user_id=user_id,
        group_id=group_id,
        reply_token=reply_token,
        source_type=source_type,
        chat_scope=chat_scope,
        received_at=event_received_at(event),
    )

    if msg_type == "text":
        raw_text = message.get("text", "")
        text = sanitize_input(raw_text)
        request.prompt_injection_stripped = _injection_stripped(raw_text, user_id)
        # A message that was nothing but an injection marker is kept, empty:
        # the pipeline answers it with its "nothing to answer" reply rather
        # than ignoring the user.
        if not text and not request.prompt_injection_stripped:
            return None
        request.input_type = InputType.TEXT
        request.text = text

    elif msg_type == "image":
        # Download and process image
        message_id = message.get("id", "")
        if not message_id:
            return None

        line = get_line_service()
        content, content_type = await line.get_message_content(message_id)
        if content is None:
            logger.error(f"Failed to download image {message_id}")
            return None

        image_b64 = await process_image(content)
        request.input_type = InputType.IMAGE
        request.image_base64 = image_b64

        # Check if there's accompanying text (from caption or quote)
        raw_text = message.get("text", "")
        text = sanitize_input(raw_text) if raw_text else ""
        if text:
            request.prompt_injection_stripped = _injection_stripped(raw_text, user_id)
            request.input_type = InputType.IMAGE_TEXT
            request.text = text

    else:
        # Unsupported message type
        return None

    _apply_quoted_context(request, message)

    logger.info(_build_input_log_summary(request))
    return request


def _injection_stripped(raw_text: str, user_id: str) -> bool:
    """True (and logged) when sanitizing removed an injection marker.

    The message is still answered: the markers are gone from its text and
    the agents add a guard line (src/agents/base_agent.py).
    """
    if not check_prompt_injection(raw_text):
        return False
    logger.warning(f"Prompt-injection marker stripped from a message by {user_id[:8]}")
    return True


def _apply_quoted_context(request: AgentRequest, message: dict) -> None:
    quoted_message_id = str(message.get("quotedMessageId", "")).strip()
    if not quoted_message_id:
        return

    request.quoted_message_id = quoted_message_id
    cached_message = get_message_cache_service().get(quoted_message_id)

    if cached_message is None:
        logger.info(
            f"[{request.request_id}] Quoted message {quoted_message_id} not found in local cache"
        )
        return

    request.quoted_message_type = cached_message.message_type
    request.quoted_text = cached_message.text
    request.quoted_image_base64 = cached_message.image_base64
    request.quoted_image_url = cached_message.image_url

    if cached_message.image_base64 and not request.image_base64:
        request.image_base64 = cached_message.image_base64
        request.input_type = InputType.IMAGE_TEXT if request.text else InputType.IMAGE
        return

    if cached_message.image_url:
        request.input_type = InputType.IMAGE_TEXT if request.text else InputType.IMAGE


def _build_input_log_summary(request: AgentRequest) -> str:
    return (
        f"[{request.request_id}] Input: {request.input_type.value}, "
        f"text={bool(request.text)}, image={bool(request.image_base64)}, "
        f"quoted={bool(request.quoted_message_id)}, "
        f"user_text={_json_log_text(request.text)}"
    )


def _json_log_text(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)
