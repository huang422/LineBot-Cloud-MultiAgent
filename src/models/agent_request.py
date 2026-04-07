"""Unified request model flowing through the multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_TEXT = "image_text"


@dataclass
class AgentRequest:
    """Represents a processed user message ready for agent routing."""

    request_id: str = field(default_factory=lambda: str(uuid4())[:8])

    # LINE context
    user_id: str = ""
    group_id: str = ""
    reply_token: str = ""
    source_type: str = "user"  # user | group | room
    chat_scope: str = "user"  # user | multi

    # Input content
    input_type: InputType = InputType.TEXT
    text: str = ""
    image_base64: str | None = None  # base64-encoded JPEG
    quoted_message_id: str = ""
    quoted_message_type: str = ""
    quoted_text: str = ""
    quoted_image_base64: str | None = None
    quoted_image_url: str = ""

    # Conversation context
    conversation_history: list[dict] | None = None
    memory_summary: str = ""
    rate_limited: bool = False

    # Orchestrator decision
    target_agent: str = ""
    output_format: str = "text"  # text | voice | image
    task_description: str = ""
    routing_reasoning: str = ""
    disable_thinking: bool = False  # True = skip reasoning for simple queries
