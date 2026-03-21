"""Output processor: send agent response to LINE in the correct format.

Strategy: ALWAYS try reply_token first (free), fall back to push.
Handles text, voice (TTS → GCS → LINE audio), and image (GCS → LINE image).
"""

from __future__ import annotations

from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.services.line_service import get_line_service
from src.services.storage_service import get_storage_service
from src.utils.logger import logger

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

        # If it's already a URL, send directly
        if isinstance(response.image_base64, str) and response.image_base64.startswith("http"):
            sent = await line.send_image(request.reply_token, to, response.image_base64)
            if sent:
                return True
            return await _send_text_fallback(
                line,
                request,
                to,
                response.text,
                default_text="圖片生成完成，但暫時無法傳送。",
            )

        # Upload base64 image to GCS to get a public URL
        storage = get_storage_service()
        uploaded_image = await storage.upload_base64_image(response.image_base64)
        if uploaded_image:
            sent = await line.send_image(request.reply_token, to, uploaded_image.public_url)
            if sent:
                storage.schedule_cleanup(uploaded_image)
                return True

            storage.schedule_cleanup(uploaded_image, delay_seconds=0)
            logger.warning("Image delivery failed, falling back to text output")

        # Fallback: send text description if available
        return await _send_text_fallback(
            line,
            request,
            to,
            response.text,
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

    if output == "voice" and response.text:
        from src.processors.tts_processor import text_to_speech

        text = _convert_s2t(response.text)
        uploaded_audio = await text_to_speech(text)
        if uploaded_audio:
            storage = get_storage_service()
            sent = await line.send_audio(request.reply_token, to, uploaded_audio.public_url)
            if sent:
                storage.schedule_cleanup(uploaded_audio)
                return True

            storage.schedule_cleanup(uploaded_audio, delay_seconds=0)
            logger.warning("Audio delivery failed, falling back to text output")
        else:
            logger.warning("TTS failed, falling back to text output")
        return await line.send_text(request.reply_token, to, text)

    # ── Default: text ─────────────────────────────────────────
    if response.text:
        text = _convert_s2t(response.text)
        return await line.send_text(request.reply_token, to, text)

    return await line.send_text(request.reply_token, to, "抱歉，無法產生回覆。")


def _convert_s2t(text: str) -> str:
    """Convert simplified Chinese to traditional Chinese."""
    try:
        converter = _get_s2t()
        if converter:
            return converter.convert(text)
        return text
    except Exception:
        return text


async def _send_text_fallback(
    line,
    request: AgentRequest,
    to: str,
    text: str | None,
    *,
    default_text: str,
) -> bool:
    fallback_text = _convert_s2t(text) if text else default_text
    return await line.send_text(request.reply_token, to, fallback_text)
