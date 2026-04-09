"""Webhook handler: trigger detection + agent dispatch pipeline.

Trigger conditions:
1. !hej [message] — explicit command
2. @mention bot name — natural trigger in groups
3. Image sent in DM — auto-trigger vision
"""

from __future__ import annotations

import re

from src.config import get_settings
from src.models.agent_request import AgentRequest
from src.models.conversation import ConversationMessage
from src.services.conversation_service import get_conversation_service
from src.services.memory_service import get_memory_service, MemoryServiceError
from src.services.rate_limit_service import get_rate_limit_service
from src.utils.logger import logger
from src.utils.validators import sanitize_input

_HEJ_PREFIX = re.compile(r"^!hej\s*", re.IGNORECASE)
_NEW_CHAT_PATTERN = re.compile(r"^(?:!hej\s*)?!new\s*$", re.IGNORECASE)


def _is_line_bot_mentioned(event: dict, settings) -> bool:
    """Return True only when the mention metadata points to this bot."""
    mention = event.get("message", {}).get("mention", {})
    mentionees = mention.get("mentionees", [])
    if not mentionees:
        return False

    bot_user_id = settings.line_bot_user_id.strip()
    bot_name = settings.bot_name.strip()

    for mentionee in mentionees:
        if mentionee.get("isSelf") is True:
            return True
        if bot_user_id and mentionee.get("userId") == bot_user_id:
            return True
        mention_text = mentionee.get("text", "").strip()
        if bot_name and mention_text in {bot_name, f"@{bot_name}"}:
            return True

    return False


def should_handle(event: dict) -> bool:
    """Check if this event should be processed (trigger detection)."""
    if event.get("type") != "message":
        return False

    message = event.get("message", {})
    msg_type = message.get("type", "")
    source = event.get("source", {})
    source_type = source.get("type", "")
    text = message.get("text", "")

    settings = get_settings()

    # DM: always handle
    if source_type == "user":
        if msg_type == "image":
            return True
        if msg_type == "text" and text.strip():
            return True
        return False

    # Group/room: need trigger
    if msg_type == "text":
        if _NEW_CHAT_PATTERN.match(text.strip()):
            return True
        # !hej command
        if _HEJ_PREFIX.match(text):
            return True
        # @mention bot via plain text
        if settings.bot_name and f"@{settings.bot_name}" in text:
            return True
        # Check LINE mention metadata for this bot only
        if _is_line_bot_mentioned(event, settings):
            return True

    # Group images are still not auto-triggered. They may be cached so that a
    # later quoted text message can reference them.
    return False


def extract_text(event: dict) -> str:
    """Extract clean user text, removing trigger prefixes."""
    text = event.get("message", {}).get("text", "")
    settings = get_settings()

    # Remove !hej prefix
    text = _HEJ_PREFIX.sub("", text)

    # Remove @mention
    if settings.bot_name:
        text = text.replace(f"@{settings.bot_name}", "").strip()

    # Remove LINE mention markers
    mention = event.get("message", {}).get("mention", {})
    if mention:
        for m in mention.get("mentionees", []):
            mention_text = m.get("text", "")
            if mention_text:
                text = text.replace(mention_text, "").strip()

    return sanitize_input(text)


async def enrich_request(request: AgentRequest) -> AgentRequest:
    """Add conversation history and check rate limit."""
    # Rate limit
    rate_svc = get_rate_limit_service()
    allowed, remaining = rate_svc.check(request.user_id)
    if not allowed:
        logger.warning(f"Rate limited user {request.user_id[:8]}...")
        request.rate_limited = True
        request.text = ""
        request.image_base64 = None  # Clear both to trigger rate-limit path in main
        return request

    # Persistent/rolling memory context
    try:
        memory = await get_memory_service().load_context(
            source_type=request.source_type,
            chat_scope=request.chat_scope,
            chat_id=request.group_id,
        )
        request.conversation_history = memory.to_openai_messages()
        request.memory_summary = memory.summary_text
        request.previous_agent = memory.last_agent
        request.previous_output_format = memory.last_output_format
        request.previous_task_description = memory.last_task_description
        request.previous_routing_reasoning = memory.last_routing_reasoning
        request.previous_disable_thinking = memory.last_disable_thinking
    except MemoryServiceError as e:
        logger.error(f"Memory context load failed: {e}")
        request.conversation_history = []
        request.memory_summary = ""
        request.previous_agent = ""
        request.previous_output_format = "text"
        request.previous_task_description = ""
        request.previous_routing_reasoning = ""
        request.previous_disable_thinking = True

    return request


def record_conversation(
    request: AgentRequest,
    response_text: str,
    *,
    assistant_delivered: bool = True,
) -> None:
    """Record the bot's assistant response in the legacy conversation cache.

    User messages are recorded separately by ``main.py`` once each event has
    been handled, so we only add the assistant reply here.
    """
    if assistant_delivered and response_text:
        conv_svc = get_conversation_service()
        conv_svc.add_message(
            request.group_id,
            ConversationMessage(role="assistant", content=response_text),
        )
