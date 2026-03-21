"""In-memory conversation history per group/user."""

from __future__ import annotations

from collections import defaultdict, deque
from time import time

from src.config import get_settings
from src.models.conversation import ConversationMessage
from src.utils.logger import logger


class ConversationService:
    def __init__(self) -> None:
        settings = get_settings()
        self._max = settings.max_conversation_history
        self._ttl = settings.conversation_ttl_seconds
        self._history: dict[str, deque[ConversationMessage]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )
        self._last_cleanup = time()
        self._cleanup_interval = 600  # cleanup every 10 minutes

    def _prune_expired_messages(
        self, chat_id: str, now: float
    ) -> deque[ConversationMessage] | None:
        messages = self._history.get(chat_id)
        if messages is None:
            return None

        while messages and now - messages[0].timestamp >= self._ttl:
            messages.popleft()

        if not messages:
            del self._history[chat_id]
            return None

        return messages

    def add_message(self, chat_id: str, msg: ConversationMessage) -> None:
        self._history[chat_id].append(msg)

    def get_history(self, chat_id: str) -> list[dict]:
        """Return OpenAI-format message list for context."""
        now = time()
        messages = self._prune_expired_messages(chat_id, now)
        if messages is None:
            return []

        messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        # Periodic cleanup of all expired entries
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()
            self._last_cleanup = now

        return messages

    def cleanup_expired(self) -> int:
        """Remove chat entries where all messages have expired. Returns count removed."""
        now = time()
        removed = 0
        for chat_id in list(self._history.keys()):
            if self._prune_expired_messages(chat_id, now) is None:
                removed += 1
        return removed

    def get_stats(self) -> dict:
        now = time()
        total_messages = 0
        for chat_id in list(self._history.keys()):
            messages = self._prune_expired_messages(chat_id, now)
            if messages is not None:
                total_messages += len(messages)
        return {
            "groups_tracked": len(self._history),
            "total_messages": total_messages,
        }


_instance: ConversationService | None = None


def get_conversation_service() -> ConversationService:
    global _instance
    if _instance is None:
        _instance = ConversationService()
    return _instance
