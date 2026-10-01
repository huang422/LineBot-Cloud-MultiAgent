"""Unified request model flowing through the multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic, time
from uuid import uuid4

# Settings.reply_window_seconds default: how long after the event any plain
# answer may take. LINE only guarantees a reply token for about a minute;
# the reply is still tried first, and a refused one is pushed.
DEFAULT_REPLY_WINDOW_SECONDS = 90.0
# Settings.turn_deadline_seconds default: the hard limit for one turn.
DEFAULT_TURN_DEADLINE_SECONDS = 240.0


def _monotonic_offset() -> float:
    return monotonic() - time()


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
    # When LINE received the message (epoch seconds, from the event's
    # ``timestamp``), how long after that a plain answer may take
    # (``deadline``), and how long the whole turn may take once work on it
    # has started.
    received_at: float = field(default_factory=time)
    reply_window_seconds: float = DEFAULT_REPLY_WINDOW_SECONDS
    turn_deadline_seconds: float = DEFAULT_TURN_DEADLINE_SECONDS
    # ``time.monotonic() - time.time()``, captured once when the request is
    # built, so both deadlines convert to the clock the model tiers use.
    monotonic_offset: float = field(default_factory=_monotonic_offset, repr=False, compare=False)
    # ``time.monotonic()`` when the request was built, i.e. when work on
    # the event started; the turn deadline counts from here.
    started_at: float = field(default_factory=monotonic, repr=False, compare=False)

    # Input content
    input_type: InputType = InputType.TEXT
    text: str = ""
    image_base64: str | None = None  # base64-encoded JPEG
    quoted_message_id: str = ""
    quoted_message_type: str = ""
    quoted_text: str = ""
    quoted_image_base64: str | None = None
    quoted_image_url: str = ""
    # Input validation removed a prompt-injection marker from ``text``
    # (src/utils/validators.py); the agents then add a guard line.
    prompt_injection_stripped: bool = False

    # Conversation context
    conversation_history: list[dict] | None = None
    memory_summary: str = ""
    # The sender's stored profile facts, already bounded
    # (MemoryService.load_profile_facts); shown to the answering agents.
    user_profile_facts: list[str] = field(default_factory=list)
    previous_agent: str = ""
    previous_output_format: str = "text"
    previous_task_description: str = ""
    previous_routing_reasoning: str = ""
    previous_disable_thinking: bool = True
    rate_limited: bool = False

    # Orchestrator decision
    target_agent: str = ""
    output_format: str = "text"  # text | voice | image
    task_description: str = ""
    routing_reasoning: str = ""
    disable_thinking: bool = False  # True = skip reasoning for simple queries

    @property
    def deadline(self) -> float:
        """Epoch seconds by which a plain answer must be done (the reply window)."""
        return self.received_at + self.reply_window_seconds

    def reply_deadline(self) -> float:
        """``deadline`` on the ``time.monotonic()`` clock."""
        return self.deadline + self.monotonic_offset

    def turn_deadline(self) -> float:
        """``time.monotonic()`` time by which the whole turn must be done.

        Counted from when work started, not from the event: the limit
        exists to finish inside the task handler's budget, and an event
        that was queued or redelivered late (reply token long gone) is
        still answered, by push.
        """
        return self.started_at + self.turn_deadline_seconds
