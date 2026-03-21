"""Conversation history model."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass
class ConversationMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time)
