"""LINE Messaging Service.

Core strategy: ALWAYS try reply_token first (free), then fall back to
push_message when the reply certainly failed (an expired or invalid token,
or a request that never left).

A send whose outcome is unknown (a timeout or dropped connection after the
request went out, or a 5xx) counts as sent: LINE may have delivered it, and
a second send (the push after a reply, or the text after an image) would
deliver the answer twice. LINE's X-Line-Retry-Key cannot help here: it
dedupes a repeated push, and the reply API takes no retry key.

No push is counted, capped or refused here, the reply fallback and direct
pushes (scheduled messages) alike: LINE's own monthly quota is the only
limit. Each send that goes out is noted on the event's summary line
(``delivery=reply|push``, :func:`src.utils.request_trace.note_delivery`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

import httpx

from src.config import Settings, get_settings
from src.utils import request_trace
from src.utils.logger import logger

LINE_API_BASE = "https://api.line.me/v2"
LINE_DATA_API_BASE = "https://api-data.line.me/v2"

# Transport errors raised before any byte of the request left, so LINE
# cannot have delivered it. Any other failure is unclear.
_NOT_SENT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.UnsupportedProtocol,
)


class LineService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """``transport`` exists for tests."""
        settings = settings or get_settings()
        self.channel_secret = settings.line_channel_secret
        self.channel_access_token = settings.line_channel_access_token
        self._profile_cache_ttl = max(0, settings.line_profile_cache_ttl_seconds)
        # Maps "{source_type}::{chat_id}::{user_id}" → (cached_at, display_name)
        self._profile_cache: dict[str, tuple[float, str]] = {}
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=30, write=30, pool=10),
            headers={
                "Authorization": f"Bearer {self.channel_access_token}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

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

        # Step 2: fall back to push when the reply certainly failed
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

    # ── User profile lookup (display name) ───────────────────

    async def fetch_display_name(
        self,
        *,
        source_type: str,
        chat_id: str,
        user_id: str,
    ) -> str:
        """Return the LINE displayName for ``user_id``, with TTL cache.

        Uses ``/v2/bot/profile/{userId}`` for 1:1 chats and
        ``/v2/bot/{group|room}/{chatId}/member/{userId}`` for groups and
        rooms. The group/room endpoints work even when the user has not
        added the bot as a friend.

        Returns an empty string on any failure so callers can degrade
        gracefully (e.g. continue to use the masked ``User_xxxx`` tag).
        """
        if not user_id or not self.channel_access_token:
            return ""

        source = (source_type or "user").strip().lower() or "user"
        cache_key = f"{source}::{chat_id}::{user_id}"
        now = datetime.now(UTC).timestamp()

        cached = self._profile_cache.get(cache_key)
        if cached is not None and self._profile_cache_ttl > 0:
            cached_at, name = cached
            if now - cached_at <= self._profile_cache_ttl:
                return name
            self._profile_cache.pop(cache_key, None)

        if source == "group" and chat_id:
            url = f"{LINE_API_BASE}/bot/group/{chat_id}/member/{user_id}"
        elif source == "room" and chat_id:
            url = f"{LINE_API_BASE}/bot/room/{chat_id}/member/{user_id}"
        else:
            url = f"{LINE_API_BASE}/bot/profile/{user_id}"

        try:
            resp = await self.client.get(url)
        except Exception as exc:
            logger.warning(f"Display name fetch error for {user_id[:8]}…: {exc}")
            return ""

        if resp.status_code != 200:
            # 404 is common for users who have not added the bot in DM
            # context; downgrade to debug-level to avoid log spam.
            if resp.status_code in (404, 403):
                logger.debug(
                    f"Display name unavailable ({resp.status_code}) for "
                    f"{source}:{user_id[:8]}…"
                )
            else:
                logger.warning(
                    f"Display name fetch returned {resp.status_code} for "
                    f"{source}:{user_id[:8]}…"
                )
            return ""

        try:
            payload = resp.json()
        except Exception:
            return ""

        name = str(payload.get("displayName", "")).strip()
        if self._profile_cache_ttl > 0:
            self._profile_cache[cache_key] = (now, name)
        return name

    async def push_text(
        self, to: str, text: str, notification_disabled: bool = False
    ) -> bool:
        """Push a text message directly (scheduled messages)."""
        if len(text) > 5000:
            text = text[:4997] + "..."
        return await self._push(
            to, [{"type": "text", "text": text}], notification_disabled=notification_disabled
        )

    # ── Internal ─────────────────────────────────────────────

    async def _reply(self, reply_token: str, messages: list[dict]) -> bool:
        """False only when the reply certainly did not go out."""
        try:
            resp = await self.client.post(
                f"{LINE_API_BASE}/bot/message/reply",
                json={"replyToken": reply_token, "messages": messages},
            )
        except _NOT_SENT_ERRORS as e:
            logger.warning(f"Reply error: {e!r}")
            return False
        except Exception as e:
            logger.error(f"Reply outcome unknown ({e!r}); counted as sent, not pushed again")
            request_trace.note_delivery(request_trace.REPLY)
            return True
        if resp.status_code == 200:
            self._cache_sent_messages(messages, resp)
            logger.info("Reply sent (free)")
            request_trace.note_delivery(request_trace.REPLY)
            return True
        if resp.status_code >= 500:
            logger.error(
                f"Reply outcome unknown ({resp.status_code}: {resp.text[:200]}); "
                "counted as sent, not pushed again"
            )
            request_trace.note_delivery(request_trace.REPLY)
            return True
        logger.warning(f"Reply failed {resp.status_code}: {resp.text[:200]}")
        return False

    async def _push(
        self, to: str, messages: list[dict], *, notification_disabled: bool = False
    ) -> bool:
        """False only when the push certainly did not go out."""
        try:
            resp = await self.client.post(
                f"{LINE_API_BASE}/bot/message/push",
                json={
                    "to": to,
                    "messages": messages,
                    "notificationDisabled": notification_disabled,
                },
            )
        except _NOT_SENT_ERRORS as e:
            logger.error(f"Push error: {e!r}")
            return False
        except Exception as e:
            logger.error(f"Push outcome unknown ({e!r}); counted as sent")
            request_trace.note_delivery(request_trace.PUSH)
            return True
        if resp.status_code == 200:
            self._cache_sent_messages(messages, resp)
            logger.warning(f"Push message sent to {to[:8]}...")
            request_trace.note_delivery(request_trace.PUSH)
            return True
        if resp.status_code >= 500:
            logger.error(
                f"Push outcome unknown ({resp.status_code}: {resp.text[:200]}); counted as sent"
            )
            request_trace.note_delivery(request_trace.PUSH)
            return True
        logger.error(f"Push failed {resp.status_code}: {resp.text[:200]}")
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
        # strict=False on purpose: LINE may return fewer sentMessages
        # than we sent, and this is a best-effort cache update.
        for message_payload, sent_meta in zip(messages, sent_messages, strict=False):
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

_instance: LineService | None = None


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
