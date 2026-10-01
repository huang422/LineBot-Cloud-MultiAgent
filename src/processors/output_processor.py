"""Output processor: send agent response to LINE in the correct format.

Strategy: every send goes through LineService, which tries the free
reply token first and pushes only when the reply certainly failed (an
expired or invalid token); no push is counted, capped or refused.
Handles text, voice (TTS → GCS → LINE audio), and image (GCS → LINE image).
Uploaded media expires through the bucket lifecycle rule; an object is only
deleted here, inline, when delivering it failed.

Every text that goes out (a reply, an image caption, the text read aloud,
a fallback) is plain text first (:func:`~src.utils.line_text.to_line_text`
drops Markdown and LaTeX, which LINE would show raw), then Traditional
Chinese (OpenCC s2t). A reply that is blank once its markup is gone (an
empty code fence) is never sent as is, since LINE rejects an empty text
message: the default text goes out instead.
"""

from __future__ import annotations

from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.services.line_service import get_line_service
from src.services.storage_service import UploadedMedia, get_storage_service
from src.utils import request_trace
from src.utils.line_text import to_line_text
from src.utils.logger import logger

NO_REPLY_TEXT = "抱歉，無法產生回覆。"

# Simplified → Traditional Chinese converter (lazy init for safety)
_s2t = None


def _get_s2t():
    """Lazy-init OpenCC converter to avoid import-time crash."""
    global _s2t
    if _s2t is None:
        try:
            import opencc
            _s2t = opencc.OpenCC("s2t")
        except Exception as e:
            logger.warning(f"OpenCC initialization failed, s2t disabled: {e}")
    return _s2t


async def send_response(request: AgentRequest, response: AgentResponse) -> bool:
    """Send the agent's response to the user via LINE."""
    line = get_line_service()
    to = request.group_id or request.user_id
    output = response.output_format or "text"

    # ── Image output ──────────────────────────────────────────
    if output == "image" and response.image_base64:
        # Normalize: some models return {"url": "..."} instead of a string
        if isinstance(response.image_base64, dict):
            response.image_base64 = response.image_base64.get("url", "")

        if not isinstance(response.image_base64, str) or not response.image_base64:
            return await _send_text_fallback(
                line, request, to, response.text,
                default_text="圖片生成完成，但格式無法辨識。",
            )

        # Resolve image URL (direct URL or upload base64 to GCS)
        image_url: str | None = None
        uploaded_image = None

        if response.image_base64.startswith("http"):
            image_url = response.image_base64
        else:
            storage = get_storage_service()
            uploaded_image = await storage.upload_base64_image(response.image_base64)
            if uploaded_image:
                image_url = uploaded_image.public_url

        if not image_url:
            if uploaded_image:
                await _delete_undelivered(storage, uploaded_image)
            return await _send_text_fallback(
                line, request, to, response.text,
                default_text="圖片生成完成，但暫時無法傳送。",
            )

        # Build messages: image + optional text description (LINE allows up to 5 messages)
        sent = await _send_image_with_text(
            line, request, to, image_url, response.text
        )
        if sent:
            return True

        if uploaded_image:
            await _delete_undelivered(storage, uploaded_image)
        logger.warning("Image delivery failed, falling back to text output")

        return await _send_text_fallback(
            line, request, to, response.text,
            default_text="圖片生成完成，但暫時無法傳送。",
        )

    # ── Voice output ──────────────────────────────────────────
    if output == "voice" and response.audio_url:
        sent = await line.send_audio(request.reply_token, to, response.audio_url)
        if sent:
            return True

        logger.warning("Audio delivery failed, falling back to text output")
        return await _send_text_fallback(
            line,
            request,
            to,
            response.text,
            default_text="語音生成完成，但暫時無法傳送。",
        )

    text = _line_text(response.text)
    if output == "voice" and text:
        from src.processors.tts_processor import text_to_speech

        uploaded_audio = await text_to_speech(text)
        if uploaded_audio:
            storage = get_storage_service()
            duration = uploaded_audio.duration_ms or 60000
            sent = await line.send_audio(
                request.reply_token, to, uploaded_audio.public_url, duration_ms=duration
            )
            if sent:
                request_trace.note_tts("edge-tts")
                return True

            request_trace.note_tts("edge-tts(send failed->text)")
            await _delete_undelivered(storage, uploaded_audio)
            logger.warning("Audio delivery failed, falling back to text output")
        else:
            request_trace.note_tts("edge-tts(failed->text)")
            logger.warning("TTS failed, falling back to text output")
        return await line.send_text(request.reply_token, to, text)

    # ── Default: text ─────────────────────────────────────────
    return await line.send_text(request.reply_token, to, text or NO_REPLY_TEXT)


async def _delete_undelivered(storage, media: UploadedMedia) -> None:
    """Delete an upload LINE never received; the lifecycle rule covers the rest."""
    if media.blob_name:
        await storage.delete_blob(media.blob_name)


def _line_text(text: str | None) -> str:
    """*text* as a LINE message: plain text, then Traditional Chinese.

    Empty when nothing but markup and whitespace was left.
    """
    converted = _convert_s2t(to_line_text(text or ""))
    return converted if converted.strip() else ""


def _convert_s2t(text: str) -> str:
    """Convert simplified Chinese to traditional Chinese."""
    try:
        converter = _get_s2t()
        if converter:
            return converter.convert(text)
        return text
    except Exception:
        return text


async def _send_image_with_text(
    line,
    request: AgentRequest,
    to: str,
    image_url: str,
    text: str | None,
) -> bool:
    """Send image + text description together as multiple LINE messages."""
    image_msg = {
        "type": "image",
        "originalContentUrl": image_url,
        "previewImageUrl": image_url,
    }
    messages = [image_msg]

    # Append text description if available and meaningful
    if text and text.strip() and text.strip() != "已根據你的需求生成圖片。":
        converted = _line_text(text)
        if len(converted) > 5000:
            converted = converted[:4997] + "..."
        if converted:
            messages.append({"type": "text", "text": converted})

    return await line.send_messages(request.reply_token, to, messages)


async def _send_text_fallback(
    line,
    request: AgentRequest,
    to: str,
    text: str | None,
    *,
    default_text: str,
) -> bool:
    fallback_text = _line_text(text) or default_text
    return await line.send_text(request.reply_token, to, fallback_text)
