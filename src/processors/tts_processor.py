"""TTS Processor — converts text to speech using edge-tts (completely free).

Generates MP3 audio, uploads to GCS, and returns a public URL for LINE
audio messages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from src.config import get_settings
from src.services.storage_service import UploadedMedia, get_storage_service
from src.utils.logger import logger


async def text_to_speech(text: str, voice: str | None = None) -> UploadedMedia | None:
    """Convert text to speech and return uploaded temporary media info.

    Args:
        text: Text to convert to speech
        voice: TTS voice name (defaults to settings.tts_voice)

    Returns:
        Uploaded media info for the MP3 file, or None on failure
    """
    settings = get_settings()
    if not settings.tts_enabled:
        logger.info("TTS disabled")
        return None

    voice = voice or settings.tts_voice

    try:
        import edge_tts
    except ImportError:
        logger.error("edge-tts package not installed")
        return None

    # Truncate very long text to avoid huge audio files
    if len(text) > 2000:
        text = text[:2000]

    tmp_path = Path(tempfile.gettempdir()) / f"{uuid4()}.mp3"

    try:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(str(tmp_path))
        logger.info(f"TTS generated: {tmp_path.name} ({tmp_path.stat().st_size} bytes)")

        # Upload to GCS
        storage = get_storage_service()
        uploaded = await storage.upload_file(str(tmp_path), content_type="audio/mpeg")

        if uploaded:
            logger.info(f"TTS uploaded to GCS")
            return uploaded
        else:
            logger.warning("TTS upload failed, GCS may not be configured")
            return None

    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        return None
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
