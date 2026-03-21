"""Input processor: detect input type, download images, build AgentRequest."""

from __future__ import annotations

import json

from src.models.agent_request import AgentRequest, InputType
from src.services.image_service import process_image
from src.services.line_service import get_line_service
from src.services.message_cache_service import get_message_cache_service
from src.utils.logger import logger
from src.utils.validators import sanitize_input, check_prompt_injection


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
    group_id = source.get("groupId", "") or source.get("roomId", "") or user_id

    request = AgentRequest(
        user_id=user_id,
        group_id=group_id,
        reply_token=reply_token,
    )

    if msg_type == "text":
        text = sanitize_input(message.get("text", ""))
        if not text:
            return None
        if check_prompt_injection(text):
            logger.warning(
                f"Prompt injection detected from {user_id[:8]}, "
                f"text={_json_log_text(text)}, ignoring"
            )
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
        text = message.get("text", "")
        if text:
            text = sanitize_input(text)
            if check_prompt_injection(text):
                logger.warning(
                    f"Prompt injection detected from {user_id[:8]}, "
                    f"text={_json_log_text(text)}, ignoring"
                )
                return None
            request.input_type = InputType.IMAGE_TEXT
            request.text = text

    else:
        # Unsupported message type
        return None

    _apply_quoted_context(request, message)

    logger.info(_build_input_log_summary(request))
    return request


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
