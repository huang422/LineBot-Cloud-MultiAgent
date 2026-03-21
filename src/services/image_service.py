"""Image processing: download from LINE, resize, base64 encode."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO

from PIL import Image

from src.utils.logger import logger

MAX_DIMENSION = 800
MIN_DIMENSION = 64
JPEG_QUALITY = 85
_DATA_URL_PREFIX = "data:image/jpeg;base64,"


def _encode_jpeg_data_url(img: Image.Image, *, quality: int = JPEG_QUALITY) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"{_DATA_URL_PREFIX}{b64}"


def _process_image_sync(content: BytesIO) -> str:
    """CPU-bound image processing (runs in executor)."""
    img = Image.open(content)
    img = img.convert("RGB")

    # Resize
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    elif max(w, h) < MIN_DIMENSION:
        ratio = MIN_DIMENSION / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    logger.info(f"Image processed: {img.size[0]}x{img.size[1]}")
    return _encode_jpeg_data_url(img)


def fit_image_data_url(data_url: str, max_chars: int) -> str | None:
    """Shrink a JPEG data URL so it stays valid and under a cache size limit."""
    if not data_url or len(data_url) <= max_chars:
        return data_url
    if max_chars <= len(_DATA_URL_PREFIX):
        return None

    try:
        _, b64data = data_url.split(",", 1)
        image_bytes = base64.b64decode(b64data, validate=True)
        original = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.warning(f"Could not compact cached image: {e}")
        return None

    current_max_dimension = max(original.size)
    dimensions: list[int] = []
    for dimension in (current_max_dimension, 640, 512, 384, 256):
        capped = min(current_max_dimension, dimension)
        if capped >= MIN_DIMENSION and capped not in dimensions:
            dimensions.append(capped)

    for dimension in dimensions:
        candidate = original.copy()
        if dimension < current_max_dimension:
            ratio = dimension / current_max_dimension
            candidate = candidate.resize(
                (
                    max(1, int(candidate.width * ratio)),
                    max(1, int(candidate.height * ratio)),
                ),
                Image.LANCZOS,
            )

        for quality in (85, 75, 65, 55, 45):
            compacted = _encode_jpeg_data_url(candidate, quality=quality)
            if len(compacted) <= max_chars:
                logger.info(
                    "Compacted cached image from %s to %s chars",
                    len(data_url),
                    len(compacted),
                )
                return compacted

    logger.warning(
        "Could not compact cached image below %s chars (current=%s)",
        max_chars,
        len(data_url),
    )
    return None


async def process_image(content: BytesIO) -> str:
    """Resize image and return as base64 data URL.

    Runs PIL operations in a thread executor to avoid blocking the event loop.

    Returns:
        "data:image/jpeg;base64,..." string ready for OpenRouter vision API.
    """
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _process_image_sync, content)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise
