"""Cache recent incoming LINE messages so quoted replies can recover context.

LINE webhooks only include `quotedMessageId`; they don't let bots fetch the
quoted content later. To support "reply to a photo and ask about that photo",
we keep a short-lived in-memory cache of recent incoming text and image
messages keyed by LINE message ID.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from time import time

from src.models.agent_request import AgentRequest, InputType
from src.services.image_service import fit_image_data_url, process_image
from src.services.line_service import get_line_service
from src.utils.logger import logger
from src.utils.validators import sanitize_input

_TTL_SECONDS = 3600
_MAX_ENTRIES = 100
_MAX_IMAGE_BASE64_LEN = 200_000  # ~150KB decoded, limit per cached image


@dataclass
class CachedMessage:
    message_id: str
    message_type: str
    text: str = ""
    image_base64: str | None = None
    image_url: str = ""
    timestamp: float = field(default_factory=time)


class MessageCacheService:
    def __init__(self) -> None:
        self._messages: OrderedDict[str, CachedMessage] = OrderedDict()

    def _fit_cached_image(self, image_base64: str | None, *, message_id: str) -> str | None:
        if not image_base64:
            return None

        fitted_image = fit_image_data_url(image_base64, _MAX_IMAGE_BASE64_LEN)
        if fitted_image is None:
            logger.warning(f"Skipping oversized cached image for {message_id}")
        return fitted_image

    def _cleanup(self) -> None:
        now = time()
        expired_ids = [
            message_id
            for message_id, msg in self._messages.items()
            if now - msg.timestamp >= _TTL_SECONDS
        ]
        for message_id in expired_ids:
            self._messages.pop(message_id, None)

        while len(self._messages) > _MAX_ENTRIES:
            self._messages.popitem(last=False)

    def remember(self, message: CachedMessage) -> None:
        self._cleanup()
        self._messages.pop(message.message_id, None)
        self._messages[message.message_id] = message
        while len(self._messages) > _MAX_ENTRIES:
            self._messages.popitem(last=False)

    def get(self, message_id: str) -> CachedMessage | None:
        self._cleanup()
        message = self._messages.get(message_id)
        if message:
            self._messages.move_to_end(message_id)
        return message

    def cache_bot_message(
        self,
        message_id: str,
        message_type: str,
        *,
        text: str = "",
        image_url: str = "",
    ) -> None:
        if not message_id or message_type not in {"text", "image"}:
            return

        self.remember(
            CachedMessage(
                message_id=message_id,
                message_type=message_type,
                text=sanitize_input(text) if text else "",
                image_url=image_url,
            )
        )

    async def cache_event_message(self, event: dict) -> None:
        """Cache a message event that won't go through the normal processing path."""
        message = event.get("message", {})
        message_id = message.get("id", "")
        msg_type = message.get("type", "")

        if not message_id or msg_type not in {"text", "image"}:
            return

        if msg_type == "text":
            self.remember(
                CachedMessage(
                    message_id=message_id,
                    message_type="text",
                    text=sanitize_input(message.get("text", "")),
                )
            )
            return

        try:
            line = get_line_service()
            content, _ = await line.get_message_content(message_id)
            if content is None:
                logger.warning(f"Could not cache quoted image source {message_id}")
                return

            image_b64 = await process_image(content)
            cached_image = self._fit_cached_image(
                image_b64,
                message_id=message_id,
            )
            self.remember(
                CachedMessage(
                    message_id=message_id,
                    message_type="image",
                    text=sanitize_input(message.get("text", "")),
                    image_base64=cached_image,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to cache message {message_id}: {e}")

    def cache_processed_request(self, event: dict, request: AgentRequest) -> None:
        """Cache the current handled message without re-downloading the content."""
        message = event.get("message", {})
        message_id = message.get("id", "")
        msg_type = message.get("type", "")

        if not message_id or msg_type not in {"text", "image"}:
            return

        if msg_type == "text":
            cached_text = request.text or sanitize_input(message.get("text", ""))
            self.remember(
                CachedMessage(
                    message_id=message_id,
                    message_type="text",
                    text=cached_text,
                )
            )
            return

        if request.input_type in (InputType.IMAGE, InputType.IMAGE_TEXT) and request.image_base64:
            cached_text = request.text or sanitize_input(message.get("text", ""))
            cached_image = self._fit_cached_image(
                request.image_base64,
                message_id=message_id,
            )
            self.remember(
                CachedMessage(
                    message_id=message_id,
                    message_type="image",
                    text=cached_text,
                    image_base64=cached_image,
                )
            )


_instance: MessageCacheService | None = None


def get_message_cache_service() -> MessageCacheService:
    global _instance
    if _instance is None:
        _instance = MessageCacheService()
    return _instance
