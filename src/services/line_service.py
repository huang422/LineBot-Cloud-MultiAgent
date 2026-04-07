"""LINE Messaging Service.

Core strategy: ALWAYS try reply_token first (free), then fall back to
push_message when reply fails and push fallback is enabled.

Normal reply-fallback push is separate from proactive direct push
(for example scheduled messages), which stays budget-controlled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

import httpx

from src.config import get_settings
from src.utils.logger import logger

LINE_API_BASE = "https://api.line.me/v2"
LINE_DATA_API_BASE = "https://api-data.line.me/v2"


class LineService:
    def __init__(self) -> None:
        settings = get_settings()
        self.channel_secret = settings.line_channel_secret
        self.channel_access_token = settings.line_channel_access_token
        self.push_fallback_enabled = settings.line_push_fallback_enabled
        self._push_monthly_limit = settings.line_push_monthly_limit
        self._current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        self._reply_fallback_push_count = 0
        self._direct_push_count = 0
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=30, write=30, pool=10),
            headers={
                "Authorization": f"Bearer {self.channel_access_token}",
                "Content-Type": "application/json",
            },
        )
        if self.push_fallback_enabled and self._push_monthly_limit > 0:
            logger.info(
                "LineService initialized "
                f"(reply fallback enabled, direct push limit={self._push_monthly_limit}/month)"
            )
        elif self.push_fallback_enabled:
            logger.warning(
                "LineService initialized "
                "(reply fallback enabled, direct push unlimited)"
            )
        else:
            logger.info("LineService initialized (all push usage disabled)")

    async def close(self) -> None:
        await self.client.aclose()

    # ── Reply (free) → Push (fallback) ───────────────────────

    async def send_text(
        self, reply_token: str, to: str, text: str
    ) -> bool:
        """Send text: try reply_token first, fall back to push."""
        if len(text) > 5000:
            text = text[:4997] + "..."

        # Step 1: try reply (free)
        if reply_token:
            success = await self._reply(reply_token, [{"type": "text", "text": text}])
            if success:
                return True

        # Step 2: fall back to push if reply is unavailable and push fallback is enabled
        if to:
            return await self._push(to, [{"type": "text", "text": text}])
        return False

    async def send_image(
        self, reply_token: str, to: str, image_url: str, preview_url: str | None = None
    ) -> bool:
        """Send image: try reply first, fall back to push."""
        msg = {
            "type": "image",
            "originalContentUrl": image_url,
            "previewImageUrl": preview_url or image_url,
        }
        if reply_token:
            success = await self._reply(reply_token, [msg])
            if success:
                return True
        if to:
            return await self._push(to, [msg])
        return False

    async def send_messages(
        self, reply_token: str, to: str, messages: list[dict]
    ) -> bool:
        """Send multiple messages (up to 5) in one reply/push. Used for image+text combos."""
        messages = messages[:5]  # LINE API limit
        if reply_token:
            success = await self._reply(reply_token, messages)
            if success:
                return True
        if to:
            return await self._push(to, messages)
        return False

    async def send_audio(
        self, reply_token: str, to: str, audio_url: str, duration_ms: int = 60000
    ) -> bool:
        """Send audio: try reply first, fall back to push."""
        msg = {
            "type": "audio",
            "originalContentUrl": audio_url,
            "duration": duration_ms,
        }
        if reply_token:
            success = await self._reply(reply_token, [msg])
            if success:
                return True
        if to:
            return await self._push(to, [msg])
        return False

    async def send_loading_animation(self, chat_id: str, seconds: int = 60) -> bool:
        """Send loading indicator (free, no quota cost)."""
        seconds = max(5, min(60, seconds))
        try:
            resp = await self.client.post(
                f"{LINE_API_BASE}/bot/chat/loading/start",
                json={"chatId": chat_id, "loadingSeconds": seconds},
            )
            return resp.status_code in (200, 202)
        except Exception:
            return False

    async def get_message_content(self, message_id: str) -> tuple[BytesIO | None, str | None]:
        """Download message content (image/audio/video)."""
        try:
            resp = await self.client.get(
                f"{LINE_DATA_API_BASE}/bot/message/{message_id}/content"
            )
            if resp.status_code == 200:
                return BytesIO(resp.content), resp.headers.get("content-type")
            return None, None
        except Exception as e:
            logger.error(f"Content download error: {e}")
            return None, None

    async def push_text(
        self, to: str, text: str, notification_disabled: bool = False
    ) -> bool:
        """Push a text message directly when the explicit direct-push budget allows it."""
        if len(text) > 5000:
            text = text[:4997] + "..."
        return await self._push(
            to, [{"type": "text", "text": text}],
            notification_disabled=notification_disabled,
            push_reason="direct_push",
        )

    # ── Internal ─────────────────────────────────────────────

    def _check_and_reset_push_budget(self) -> None:
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        if current_month != self._current_month:
            logger.info(
                f"LINE push budget reset: {self._current_month} -> {current_month}, "
                f"reply_fallback={self._reply_fallback_push_count}, "
                f"direct_push={self._direct_push_count}/{self._push_monthly_limit}"
            )
            self._current_month = current_month
            self._reply_fallback_push_count = 0
            self._direct_push_count = 0

    @property
    def direct_push_remaining(self) -> int | None:
        self._check_and_reset_push_budget()
        if not self.push_fallback_enabled:
            return 0
        if self._push_monthly_limit <= 0:
            return None
        return max(0, self._push_monthly_limit - self._direct_push_count)

    def _is_push_allowed(self, push_reason: str) -> bool:
        self._check_and_reset_push_budget()
        if not self.push_fallback_enabled:
            return False
        if push_reason == "reply_fallback":
            return True
        if self._push_monthly_limit <= 0:
            return True
        return self._direct_push_count < self._push_monthly_limit

    def get_push_stats(self) -> dict:
        self._check_and_reset_push_budget()
        return {
            "enabled": self.push_fallback_enabled,
            "month": self._current_month,
            "reply_fallback_used": self._reply_fallback_push_count,
            "direct_push_used": self._direct_push_count,
            "direct_push_limit": self._push_monthly_limit if self._push_monthly_limit > 0 else None,
            "direct_push_remaining": self.direct_push_remaining,
            "direct_push_unlimited": self.push_fallback_enabled and self._push_monthly_limit <= 0,
            "scope": "per-instance",
        }

    async def _reply(self, reply_token: str, messages: list[dict]) -> bool:
        try:
            resp = await self.client.post(
                f"{LINE_API_BASE}/bot/message/reply",
                json={"replyToken": reply_token, "messages": messages},
            )
            if resp.status_code == 200:
                self._cache_sent_messages(messages, resp)
                logger.info("Reply sent (free)")
                return True
            logger.warning(f"Reply failed {resp.status_code}: {resp.text[:200]}")
            return False
        except Exception as e:
            logger.warning(f"Reply error: {e}")
            return False

    async def _push(
        self,
        to: str,
        messages: list[dict],
        *,
        notification_disabled: bool = False,
        push_reason: str = "reply_fallback",
    ) -> bool:
        if not self._is_push_allowed(push_reason):
            limit = self._push_monthly_limit if self._push_monthly_limit > 0 else "unlimited"
            logger.warning(
                f"Push blocked ({push_reason}): "
                f"enabled={self.push_fallback_enabled}, "
                f"direct_push={self._direct_push_count}/{limit}"
            )
            return False

        try:
            resp = await self.client.post(
                f"{LINE_API_BASE}/bot/message/push",
                json={
                    "to": to,
                    "messages": messages,
                    "notificationDisabled": notification_disabled,
                },
            )
            if resp.status_code == 200:
                self._cache_sent_messages(messages, resp)
                if push_reason == "reply_fallback":
                    self._reply_fallback_push_count += 1
                else:
                    self._direct_push_count += 1
                limit = self._push_monthly_limit if self._push_monthly_limit > 0 else "unlimited"
                logger.warning(
                    f"Push message sent ({push_reason}) to {to[:8]}..., "
                    f"reply_fallback={self._reply_fallback_push_count}, "
                    f"direct_push={self._direct_push_count}/{limit}"
                )
                return True
            logger.error(f"Push failed {resp.status_code}: {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Push error: {e}")
            return False

    def _cache_sent_messages(self, messages: list[dict], response) -> None:
        try:
            payload = response.json()
        except Exception:
            return

        sent_messages = payload.get("sentMessages")
        if not isinstance(sent_messages, list) or not sent_messages:
            return

        from src.services.message_cache_service import get_message_cache_service

        cache = get_message_cache_service()
        for message_payload, sent_meta in zip(messages, sent_messages):
            if not isinstance(message_payload, dict) or not isinstance(sent_meta, dict):
                continue

            message_id = str(sent_meta.get("id", "")).strip()
            message_type = str(message_payload.get("type", "")).strip()
            if not message_id or message_type not in {"text", "image"}:
                continue

            cache.cache_bot_message(
                message_id,
                message_type,
                text=str(message_payload.get("text", "")).strip(),
                image_url=str(message_payload.get("originalContentUrl", "")).strip(),
            )


# ── Singleton ────────────────────────────────────────────────

_instance: Optional[LineService] = None


def get_line_service() -> LineService:
    global _instance
    if _instance is None:
        _instance = LineService()
    return _instance


async def close_line_service() -> None:
    global _instance
    if _instance:
        await _instance.close()
        _instance = None
