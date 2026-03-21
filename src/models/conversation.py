"""Conversation history model."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass
class ConversationMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time)
    user_id: str = ""  # LINE user ID (tracks who said what in groups)
    message_type: str = "text"  # "text" | "image" | "sticker" | "audio"
