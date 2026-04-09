"""OpenRouter API provider using the OpenAI-compatible chat completions endpoint.

The project defaults still point at free-tier-friendly model IDs, but the
client no longer hard-enforces them against a local whitelist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from src.utils.logger import logger
from src.utils.rate_tracker import RateTracker


class RateLimitError(Exception):
    """Raised when OpenRouter returns 429."""

    def __init__(self, model: str, retry_after: int | None = None):
        self.model = model
        self.retry_after = retry_after
        super().__init__(f"Rate limited: {model}")


class ProviderError(Exception):
    """Raised on non-rate-limit API errors."""

    def __init__(self, model: str, status: int, detail: str, *, provider: str = ""):
        self.model = model
        self.status = status
        self.detail = detail
        self.provider = provider
        label = f"{provider} error" if provider else f"Provider error"
        super().__init__(f"{label} {status} for {model}: {detail}")


@dataclass
class ProviderResponse:
    """Unified response from any LLM provider."""

    text: str | None = None
    images: list[str] | None = None  # base64 data URLs
    model: str = ""
    usage: dict | None = None
    reasoning_content: str | None = None


_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_openai_response(data: dict) -> tuple[str | None, list[str] | None, str | None]:
    """Parse an OpenAI-compatible response, returning (text, images, reasoning_content).

    Reasoning is extracted from (in priority order):
      1. ``message.reasoning`` — OpenRouter primary field
      2. ``message.reasoning_content`` — alias / NVIDIA field
      3. Content parts with ``type: "thinking"``
      4. ``<think>…</think>`` tags embedded in text (native Qwen3 format)
    """
    text = None
    images = None
    reasoning_content = None
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    # Extract reasoning from dedicated fields
    # OpenRouter uses "reasoning" as primary, "reasoning_content" as alias
    for field in ("reasoning", "reasoning_content"):
        val = message.get(field)
        if isinstance(val, str) and val.strip():
            reasoning_content = val.strip()
            break

    if isinstance(message.get("content"), str):
        text = message["content"]
    elif isinstance(message.get("content"), list):
        text_parts: list[str] = []
        for part in message["content"]:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                part_text = part.get("text")
                if isinstance(part_text, str):
                    text_parts.append(part_text)
            elif part.get("type") == "image_url":
                if images is None:
                    images = []
                images.append(part["image_url"]["url"])
            elif part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                reasoning_content = reasoning_content or part["thinking"].strip()
        if text_parts:
            text = "".join(text_parts)

    # Some image models return images in a separate field
    if message.get("images"):
        images = images or []
        for img in message["images"]:
            if isinstance(img, dict) and "image_url" in img:
                val = img["image_url"]
                images.append(val["url"] if isinstance(val, dict) else val)
            elif isinstance(img, str):
                images.append(img)

    # Extract <think> content before stripping (native Qwen3 reasoning on NVIDIA)
    if text and not reasoning_content:
        think_match = _THINK_TAG_RE.search(text)
        if think_match:
            extracted = think_match.group(0)
            reasoning_content = re.sub(
                r"</?think>", "", extracted
            ).strip() or None

    # Strip leaked <think> tags from final user-facing text
    if text:
        text = _THINK_TAG_RE.sub("", text).strip() or None

    return text, images, reasoning_content


class OpenRouterProvider:
    """Async client for the OpenRouter chat completions API."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        rate_tracker: RateTracker,
        *,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        reasoning_exclude: bool = True,
        thinking_budget: int = 4096,
    ) -> None:
        self.api_key = api_key.strip()
        self.rate_tracker = rate_tracker
        self._reasoning_enabled = reasoning_enabled
        self._thinking_budget = thinking_budget
        self._reasoning_config = None
        if reasoning_enabled:
            self._reasoning_config = {
                "enabled": True,
                "effort": reasoning_effort,
                "exclude": reasoning_exclude,
            }
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/linebot-cloud-agent",
            "X-OpenRouter-Title": "LineBot-CloudAgent",
        }
        self._client = httpx.AsyncClient(timeout=120, headers=self._headers)

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        modalities: list[str] | None = None,
        require_reasoning_tokens: bool = False,
        disable_thinking: bool = False,
    ) -> ProviderResponse:
        """Call OpenRouter and return a unified response.

        Args:
            model: Model ID (e.g. "nvidia/nemotron-3-super-120b-a12b:free")
            messages: OpenAI-format message list
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens
            modalities: Set to ["image", "text"] for image generation
        """
        from src.providers.model_registry import supports_reasoning as _supports_reasoning

        model = model.strip()
        if not model:
            raise ProviderError("<empty>", 400, "Model ID must not be empty", provider="OpenRouter")

        # Only attach reasoning for models known to support it
        use_reasoning = (
            self._reasoning_enabled
            and self._reasoning_config
            and not modalities
            and _supports_reasoning(model)
            and not disable_thinking
        )
        if disable_thinking and self._reasoning_enabled:
            logger.info(f"OpenRouter: thinking OFF for {model} (disable_thinking=True)")
        elif use_reasoning:
            logger.info(f"OpenRouter: thinking ON for {model}")

        # Reasoning tokens count against max_tokens on OpenRouter,
        # so add the thinking budget to ensure actual output has room.
        effective_max_tokens = max_tokens
        if use_reasoning:
            effective_max_tokens = self._thinking_budget + max_tokens

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        if modalities:
            payload["modalities"] = modalities
        elif use_reasoning:
            payload["reasoning"] = dict(self._reasoning_config)
            # Top-level flag to include reasoning output in the response
            payload["include_reasoning"] = True

        logger.info(
            f"OpenRouter request → {model} "
            f"(temp={temperature}, max_tok={max_tokens}, reasoning={use_reasoning})"
        )

        try:
            resp = await self._client.post(self.BASE_URL, json=payload)
        except httpx.HTTPError as e:
            detail = str(e)
            logger.error(f"OpenRouter request failed for {model}: {detail}")
            raise ProviderError(model, 0, detail, provider="OpenRouter") from e

        # Update rate-limit tracking from headers
        self.rate_tracker.update_from_headers(model, dict(resp.headers))
        self.rate_tracker.record_request(model)

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            ra = int(retry_after) if retry_after else None
            self.rate_tracker.record_limit_hit(model, ra)
            raise RateLimitError(model, ra)

        if resp.status_code != 200:
            detail = resp.text[:500]
            if resp.status_code in (401, 403):
                logger.error(
                    f"OpenRouter auth error {resp.status_code} for {model}: {detail}. "
                    "Check OPENROUTER_API_KEY and model access."
                )
            else:
                logger.error(f"OpenRouter error {resp.status_code}: {detail}")
            raise ProviderError(model, resp.status_code, detail, provider="OpenRouter")
        try:
            data = resp.json()
        except ValueError as e:
            logger.error(f"OpenRouter returned invalid JSON for {model}: {e}")
            raise ProviderError(model, resp.status_code, "Invalid JSON response", provider="OpenRouter") from e
        text, images, reasoning_content = parse_openai_response(data)

        usage = data.get("usage")
        # reasoning_tokens may be top-level or nested under completion_tokens_details
        reasoning_tokens = (usage or {}).get("reasoning_tokens", 0)
        if not reasoning_tokens:
            reasoning_tokens = (
                (usage or {}).get("completion_tokens_details") or {}
            ).get("reasoning_tokens", 0)
        if require_reasoning_tokens and use_reasoning and reasoning_tokens <= 0 and not reasoning_content:
            logger.warning(
                f"OpenRouter response from {model} has no reasoning tokens and no reasoning content; "
                "the model may not support reasoning despite being configured for it"
            )
        if reasoning_content:
            logger.info(
                f"OpenRouter reasoning ← {model} "
                f"({len(reasoning_content)} chars)"
            )
        logger.info(
            f"OpenRouter response ← {model} "
            f"(text={bool(text)}, images={len(images or [])}, "
            f"reasoning_tokens={reasoning_tokens})"
        )

        return ProviderResponse(
            text=text,
            images=images,
            model=data.get("model", model),
            usage=usage,
            reasoning_content=reasoning_content,
        )
