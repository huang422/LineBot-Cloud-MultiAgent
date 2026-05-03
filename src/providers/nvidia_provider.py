"""NVIDIA API provider — chat completions + image generation endpoints.

Chat completions use the OpenAI-compatible endpoint on build.nvidia.com.
Image generation uses NVIDIA Visual GenAI NIM endpoints on ai.api.nvidia.com
with model-specific request/response formats.
"""

from __future__ import annotations

import httpx

from src.providers.openrouter_provider import (
    ProviderResponse,
    RateLimitError,
    ProviderError,
    parse_openai_response,
)
from src.utils.logger import logger
from src.utils.rate_tracker import RateTracker

# NVIDIA image generation endpoint base
_IMAGE_GEN_BASE = "https://ai.api.nvidia.com/v1/genai"

# Model-specific endpoint mapping. NVIDIA's public FLUX endpoint uses the
# lowercase slug even though the model is commonly written as FLUX.1-dev.
_FLUX_1_DEV_MODEL = "black-forest-labs/flux.1-dev"
_IMAGE_MODEL_ENDPOINTS: dict[str, str] = {
    _FLUX_1_DEV_MODEL: f"{_IMAGE_GEN_BASE}/{_FLUX_1_DEV_MODEL}",
    "black-forest-labs/FLUX.1-dev": f"{_IMAGE_GEN_BASE}/{_FLUX_1_DEV_MODEL}",
}


class NvidiaProvider:
    """Async client for NVIDIA's free inference API.

    Reasoning activation is model-family-specific:
    - Qwen3/3.5: ``/think`` soft switch + ``chat_template_kwargs`` (top-level).
    - Gemma 4 / Nemotron 3 Nano/Super: ``chat_template_kwargs`` (top-level).
    - Nemotron Super V1 / Ultra V1: system prompt ``"detailed thinking on"``.
    - GPT-OSS: ``reasoning_effort`` at top level.
    ``parse_openai_response`` extracts reasoning from all response formats.
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        rate_tracker: RateTracker,
        *,
        thinking_enabled: bool = False,
        thinking_budget: int = 4096,
        thinking_model: str = "",
        primary_model: str = "",
    ) -> None:
        self.api_key = api_key
        self.rate_tracker = rate_tracker
        self._thinking_enabled = thinking_enabled
        self._thinking_budget = thinking_budget
        self._thinking_model = thinking_model.strip()
        self._primary_model = primary_model.strip()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=300, headers=self._headers)

    def resolve_model(self, model: str, *, disable_thinking: bool = False) -> str:
        from src.providers.model_registry import supports_reasoning as _supports_reasoning

        model = model.strip()
        if (
            self._thinking_enabled
            and not disable_thinking
            and self._thinking_model
            and model != self._thinking_model
            and _supports_reasoning(model)
            and (not self._primary_model or model == self._primary_model)
        ):
            return self._thinking_model
        return model

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
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ProviderResponse:
        """Call NVIDIA chat completions API and return a unified response.

        Reasoning mode depends on the model family (per official docs):
        - **Qwen3/3.5**: ``/think`` soft switch + ``chat_template_kwargs``
          (recommended temp=0.6, top_p=0.95).
        - **Gemma 4 / Nemotron 3 Nano / Super**: ``chat_template_kwargs: {enable_thinking: true}``.
        - **Nemotron Super V1 / Ultra V1**: system prompt ``"detailed thinking on"``.
        - **GPT-OSS**: ``reasoning_effort`` at top level.
        """
        from src.providers.model_registry import supports_reasoning as _supports_reasoning

        requested_model = model.strip()
        if not requested_model:
            raise ProviderError("<empty>", 400, "Model ID must not be empty", provider="NVIDIA")

        model = self.resolve_model(requested_model, disable_thinking=disable_thinking)
        expect_reasoning = self._thinking_enabled and _supports_reasoning(model) and not disable_thinking

        if expect_reasoning and model != requested_model:
            logger.info(f"NVIDIA: swapping {requested_model} → {model} for thinking mode")

        if disable_thinking and self._thinking_enabled:
            logger.info(f"NVIDIA: thinking OFF for {model} (disable_thinking=True)")
        elif expect_reasoning:
            logger.info(f"NVIDIA: thinking ON for {model}")
        model_lower = model.lower()
        uses_chat_template_payload = (
            "gemma-4" in model_lower
            or "nemotron-3-" in model_lower
            or "qwen" in model_lower
        )

        # --- Explicit thinking deactivation ---
        if disable_thinking and self._thinking_enabled and "qwen" in model_lower:
            # Qwen3/3.5: inject /no_think soft switch to explicitly suppress thinking
            messages = list(messages)
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        # Remove any leftover /think and append /no_think
                        cleaned = content.replace(" /think", "").replace("/think", "")
                        messages[i] = {**msg, "content": cleaned + " /no_think"}
                    elif isinstance(content, list):
                        new_content = list(content)
                        for j, part in enumerate(new_content):
                            if part.get("type") == "text" and part.get("text", "").strip():
                                cleaned = part["text"].replace(" /think", "").replace("/think", "")
                                new_content[j] = {**part, "text": cleaned + " /no_think"}
                                break
                        messages[i] = {**msg, "content": new_content}
                    break

        # --- Model-family-specific reasoning activation (per official docs) ---
        if expect_reasoning:
            if "gemma-4" in model_lower or "nemotron-3-" in model_lower:
                pass  # message injection not needed; payload handled below
            elif "nemotron" in model_lower:
                # Nemotron Super V1 / Ultra V1: system prompt "detailed thinking on"
                messages = list(messages)
                if not messages or messages[0].get("role") != "system":
                    messages.insert(0, {"role": "system", "content": "detailed thinking on"})
                elif "detailed thinking" not in messages[0].get("content", ""):
                    # Merge into existing system message to avoid duplicates
                    messages[0] = {
                        **messages[0],
                        "content": "detailed thinking on\n" + messages[0]["content"],
                    }
            elif "gpt-oss" in model_lower:
                pass  # payload handled below
            elif "qwen" in model_lower:
                # Qwen3/3.5: /think soft switch to force thinking mode.
                # Append /think to last user message (per Qwen3 docs).
                messages = list(messages)
                injected = False
                for i in range(len(messages) - 1, -1, -1):
                    msg = messages[i]
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            if content.strip() and "/think" not in content:
                                messages[i] = {**msg, "content": content + " /think"}
                                injected = True
                        elif isinstance(content, list):
                            # Multimodal message: append /think to first text part
                            new_content = list(content)
                            for j, part in enumerate(new_content):
                                if part.get("type") == "text" and part.get("text", "").strip() and "/think" not in part.get("text", ""):
                                    new_content[j] = {**part, "text": part["text"] + " /think"}
                                    injected = True
                                    break
                            if injected:
                                messages[i] = {**msg, "content": new_content}
                        break
                if not injected:
                    logger.warning(f"Qwen /think: could not inject into messages for {model}")

        # Qwen3 and Gemma 4 reasoning both need extra room so the final answer
        # is not squeezed out by thinking tokens.
        effective_max_tokens = max_tokens
        if expect_reasoning and ("qwen" in model_lower or "gemma-4" in model_lower):
            effective_max_tokens = self._thinking_budget + max_tokens

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }

        if expect_reasoning:
            if "gpt-oss" in model_lower:
                payload["reasoning_effort"] = "high"
            elif uses_chat_template_payload:
                payload["chat_template_kwargs"] = {"enable_thinking": True}
            # Nemotron V1/Ultra: reasoning via system prompt (injected above).
        elif disable_thinking and self._thinking_enabled:
            # Explicitly disable thinking to prevent the model from
            # entering thinking mode by default.
            if uses_chat_template_payload:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        logger.info(
            f"NVIDIA request → {model} "
            f"(temp={temperature}, max_tok={max_tokens}, "
            f"expect_reasoning={expect_reasoning})"
        )

        try:
            resp = await self._client.post(self.BASE_URL, json=payload)
        except httpx.HTTPError as e:
            detail = str(e)
            logger.error(f"NVIDIA request failed for {model}: {detail}")
            raise ProviderError(model, 0, detail, provider="NVIDIA") from e

        self.rate_tracker.record_request(model)

        if resp.status_code == 429:
            self.rate_tracker.record_limit_hit(model, None)
            raise RateLimitError(model, None)

        if resp.status_code != 200:
            detail = resp.text[:500]
            logger.error(f"NVIDIA error {resp.status_code}: {detail}")
            raise ProviderError(model, resp.status_code, detail, provider="NVIDIA")

        try:
            data = resp.json()
        except ValueError as e:
            logger.error(f"NVIDIA returned invalid JSON for {model}: {e}")
            raise ProviderError(model, resp.status_code, "Invalid JSON response", provider="NVIDIA") from e
        text, images, reasoning_content, tool_calls, raw_message = parse_openai_response(data)

        usage = data.get("usage")
        reasoning_tokens = (usage or {}).get("reasoning_tokens", 0)
        if not reasoning_tokens:
            reasoning_tokens = (
                (usage or {}).get("completion_tokens_details") or {}
            ).get("reasoning_tokens", 0)
        if require_reasoning_tokens and expect_reasoning and reasoning_tokens <= 0 and not reasoning_content:
            logger.warning(
                f"NVIDIA response from {model} has no reasoning tokens and no reasoning content; "
                "Qwen3.5 may have skipped <think> tags for this request"
            )
        if reasoning_content:
            logger.info(
                f"NVIDIA reasoning ← {model} "
                f"({len(reasoning_content)} chars)"
            )
        logger.info(
            f"NVIDIA response ← {model} "
            f"(text={bool(text)}, images={len(images or [])}, "
            f"reasoning_tokens={reasoning_tokens})"
        )

        return ProviderResponse(
            text=text,
            images=images,
            model=data.get("model", model),
            usage=usage,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            raw_message=raw_message,
        )

    # ── Image Generation ──────────────────────────────────────

    async def generate_image(
        self,
        model: str,
        prompt: str,
        *,
        negative_prompt: str = "",
        steps: int = 50,
        cfg_scale: int = 5,
        width: int = 1024,
        height: int = 1024,
    ) -> ProviderResponse:
        """Call NVIDIA Visual GenAI NIM image generation API.

        Returns a ProviderResponse with base64 image data in images[].
        """
        requested_model = model.strip()
        if not requested_model:
            raise ProviderError("<empty>", 400, "Model ID must not be empty", provider="NVIDIA")
        model = _normalize_image_model(requested_model)
        endpoint = _IMAGE_MODEL_ENDPOINTS.get(model, f"{_IMAGE_GEN_BASE}/{model}")

        is_flux = _is_flux_model(model)
        is_sd35 = "3.5" in model
        is_qwen_image = _is_qwen_image_model(model)
        payload = self._build_image_payload(
            model,
            prompt,
            negative_prompt,
            steps,
            cfg_scale,
            width,
            height,
            is_flux,
            is_sd35,
            is_qwen_image,
        )

        logger.info(f"NVIDIA image request → {model} (steps={steps}, {width}x{height})")

        try:
            resp = await self._client.post(endpoint, json=payload)
        except httpx.HTTPError as e:
            detail = str(e)
            logger.error(f"NVIDIA image request failed for {model}: {detail}")
            raise ProviderError(model, 0, detail, provider="NVIDIA") from e

        self.rate_tracker.record_request(model)

        if resp.status_code == 429:
            self.rate_tracker.record_limit_hit(model, None)
            raise RateLimitError(model, None)

        if resp.status_code != 200:
            detail = resp.text[:500]
            logger.error(f"NVIDIA image error {resp.status_code}: {detail}")
            raise ProviderError(model, resp.status_code, detail, provider="NVIDIA")

        try:
            data = resp.json()
        except ValueError as e:
            logger.error(f"NVIDIA image returned invalid JSON for {model}: {e}")
            raise ProviderError(model, resp.status_code, "Invalid JSON response", provider="NVIDIA") from e

        base64_image, finish_reason = self._parse_image_response(data)

        if finish_reason == "CONTENT_FILTERED":
            logger.warning(f"NVIDIA image {model}: content filtered")
            return ProviderResponse(text="圖片內容被安全過濾，請換個描述再試。", model=model)

        if not base64_image:
            logger.warning(f"NVIDIA image {model}: no image in response")
            return ProviderResponse(model=model)

        # Return as data URL so downstream can handle uniformly
        data_url = f"data:image/png;base64,{base64_image}"
        logger.info(f"NVIDIA image response ← {model} (success)")

        return ProviderResponse(
            images=[data_url],
            model=model,
        )

    @staticmethod
    def _build_image_payload(
        model: str,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: int,
        width: int,
        height: int,
        is_flux: bool,
        is_sd35: bool,
        is_qwen_image: bool = False,
    ) -> dict:
        """Build the request payload based on model variant."""
        if is_qwen_image:
            # NVIDIA NIM Qwen-Image schema. The endpoint accepts only
            # ``prompt`` / ``mode`` / ``seed`` / ``steps`` and *both*
            # ``width`` & ``height`` together (or neither). Sending
            # ``cfg_scale`` / ``samples`` / ``num_inference_steps``
            # results in HTTP 422. Steps default to a small value
            # (4–8 typical) so we cap to avoid 422 from over-large
            # diffusion budgets.
            payload: dict = {
                "prompt": prompt,
                "mode": "base",
                "seed": 0,
                "steps": max(1, min(steps, 8)),
            }
            if width and height:
                payload["width"] = int(width)
                payload["height"] = int(height)
            return payload
        if is_flux:
            # NVIDIA NIM FLUX.1-dev schema (see
            # https://docs.api.nvidia.com/nim/reference/black-forest-labs-flux_1-dev-infer):
            # prompt / width(=1024) / height(=1024) / cfg_scale / steps /
            # seed / mode / samples. Field names differ from the
            # HuggingFace diffusers convention — using num_inference_steps
            # or guidance_scale here returns HTTP 422.
            payload = {
                "prompt": prompt,
                "mode": "base",
                "width": 1024,
                "height": 1024,
                "cfg_scale": float(cfg_scale),
                "steps": steps,
                "samples": 1,
                "seed": 0,
            }
        elif is_sd35:
            # Legacy SD 3.5 endpoint uses width/height.
            payload = {
                "prompt": prompt,
                "mode": "base",
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "samples": 1,
            }
        else:
            # Some legacy image endpoints use aspect_ratio instead of dimensions.
            aspect = _nearest_aspect_ratio(width, height)
            payload = {
                "prompt": prompt,
                "cfg_scale": cfg_scale,
                "steps": steps,
                "aspect_ratio": aspect,
            }
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
        return payload

    @staticmethod
    def _parse_image_response(data: dict) -> tuple[str | None, str | None]:
        """Extract base64 image and finish reason from the response."""
        finish_reason = data.get("finish_reason") or data.get("finishReason")

        artifacts = data.get("artifacts") or []
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                item_finish_reason = (
                    artifact.get("finishReason")
                    or artifact.get("finish_reason")
                    or finish_reason
                )
                image = (
                    artifact.get("base64")
                    or artifact.get("image")
                    or artifact.get("b64_json")
                )
                if image or item_finish_reason:
                    return image, item_finish_reason

        image = data.get("image") or data.get("base64") or data.get("b64_json")
        if image:
            return image, finish_reason

        images = data.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, str):
                return first, finish_reason
            if isinstance(first, dict):
                return (
                    first.get("base64")
                    or first.get("image")
                    or first.get("b64_json"),
                    first.get("finishReason") or first.get("finish_reason") or finish_reason,
                )

        data_items = data.get("data")
        if isinstance(data_items, list) and data_items:
            first = data_items[0]
            if isinstance(first, dict):
                return (
                    first.get("b64_json")
                    or first.get("base64")
                    or first.get("image"),
                    first.get("finishReason") or first.get("finish_reason") or finish_reason,
                )

        return None, finish_reason


def _normalize_image_model(model: str) -> str:
    value = model.strip()
    if value.lower() in {"flux.1-dev", _FLUX_1_DEV_MODEL}:
        return _FLUX_1_DEV_MODEL
    if value.lower() in {"flux.1-schnell", "black-forest-labs/flux.1-schnell"}:
        return "black-forest-labs/flux.1-schnell"
    return value


def _is_flux_model(model: str) -> bool:
    return "flux" in model.lower()


def _is_qwen_image_model(model: str) -> bool:
    return "qwen-image" in model.lower()


def _nearest_aspect_ratio(width: int, height: int) -> str:
    """Map width/height to the nearest legacy image API aspect ratio."""
    ratio = width / height if height else 1.0
    options = [
        (1 / 1, "1:1"),
        (16 / 9, "16:9"),
        (9 / 16, "9:16"),
        (5 / 4, "5:4"),
        (4 / 5, "4:5"),
        (3 / 2, "3:2"),
        (2 / 3, "2:3"),
    ]
    return min(options, key=lambda x: abs(x[0] - ratio))[1]
