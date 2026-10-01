"""NVIDIA API provider — chat completions + image generation endpoints.

Chat completions use the OpenAI-compatible endpoint on build.nvidia.com.
Image generation uses NVIDIA Visual GenAI NIM endpoints on ai.api.nvidia.com
with model-specific request/response formats. Both go through
:func:`~src.providers.openai_compat.post_chat_completion`, which maps HTTP
statuses onto the provider errors and polls NVCF 202 answers.

Request shapes follow each exact model id's profile
(:mod:`src.providers.model_profiles`) and the tier options the fallback
chain passes along (thinking, reasoning_budget, json_mode, max_tokens,
temperature; steps for images). There is no substring family detection,
because similar-looking ids (e.g. nemotron-3 vs nemotron-3.5) take
different parameters.

Thinking is switched per protocol: ``chat_template_kwargs.enable_thinking``
(Nemotron 3 adds the profile's extra kwargs, e.g. Ultra's
``medium_effort``, and a top-level ``reasoning_budget`` only where the model
takes one), or a top-level ``reasoning_effort`` (muse-glimmer).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import httpx

from src.providers.base import (
    ContentFilteredError,
    PayloadError,
    ProviderError,
    ProviderResponse,
    SkipTierError,
    parse_openai_response,
    require_output,
)
from src.providers.messages import has_image_parts, normalize_messages
from src.providers.model_profiles import (
    DEFAULT_THINKING_BUDGET,
    MAX_REASONING_BUDGET,
    NIM_CHAT_PROTOCOLS,
    NIM_FLUX1_DEV,
    NIM_FLUX2_KLEIN,
    NIM_REASONING_EFFORT,
    ModelProfile,
    cap_max_tokens,
    clamp_temperature,
    generic_profile,
    get_profile,
    response_format_for,
    tier_sampling,
)
from src.providers.openai_compat import post_chat_completion
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.services.model_health import ModelHealth

# Upper bound for one request (202 polling included); the fallback chain's
# per-tier limit is normally much shorter.
_REQUEST_TIMEOUT_SECONDS = 300.0


# reasoning_effort values NIM_REASONING_EFFORT models get for off and on.
_EFFORT_OFF = "none"
_EFFORT_ON = "medium"


def chat_profile(model: str) -> ModelProfile:
    """The model's profile, or the generic NIM stand-in for an unknown id."""
    profile = get_profile("nvidia", model)
    if profile is None or profile.protocol not in NIM_CHAT_PROTOCOLS:
        profile = generic_profile("nvidia", model)
    assert profile is not None  # NVIDIA always has a generic chat protocol
    return profile


def build_chat_payload(
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    options: Mapping[str, Any] | None = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    json_schema: dict | None = None,
) -> dict:
    """The chat completions body for *model* with these tier *options*.

    Profiled models always say whether to think (every NIM chat model
    thinks by default): ``chat_template_kwargs.enable_thinking`` (with the
    profile's extra thinking kwargs when on), or ``reasoning_effort``
    none/medium. Thinking adds the reasoning budget
    (``options["reasoning_budget"]`` or *thinking_budget*) to max_tokens,
    and sends it as a top-level ``reasoning_budget`` only to models whose
    profile takes one. Sampling: the tier's ``max_tokens``/``temperature``
    win, then the profile's default sampling (temperature and top_p from
    the model card), then the call's temperature, clamped to the profile.
    ``json_schema`` becomes a ``response_format`` only when the tier's
    ``json_mode`` asks for one.
    """
    options = options or {}
    profile = chat_profile(model)
    temperature, max_tokens = tier_sampling(
        options, temperature=temperature, max_tokens=max_tokens, profile=profile
    )

    payload: dict = {"model": model, "messages": messages, "stream": False}
    if profile.protocol in NIM_CHAT_PROTOCOLS:
        if profile.protocol == NIM_REASONING_EFFORT:
            payload["reasoning_effort"] = _EFFORT_ON if thinking else _EFFORT_OFF
        else:
            kwargs: dict[str, Any] = {"enable_thinking": thinking}
            if thinking:
                kwargs.update(profile.thinking_template_kwargs)
            payload["chat_template_kwargs"] = kwargs
        if thinking:
            budget = min(int(options.get("reasoning_budget") or thinking_budget), MAX_REASONING_BUDGET)
            # The answer still needs max_tokens after the reasoning budget.
            max_tokens += budget
            if profile.reasoning_budget:
                payload["reasoning_budget"] = budget
    payload["max_tokens"] = cap_max_tokens(profile, max_tokens)
    payload["temperature"] = clamp_temperature(profile, temperature)
    if profile.default_top_p is not None:
        payload["top_p"] = profile.default_top_p

    if tools and profile.tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    response_format = response_format_for(options.get("json_mode"), json_schema)
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


# ── Image generation ──────────────────────────────────────────

_IMAGE_GEN_BASE = "https://ai.api.nvidia.com/v1/genai"

# FLUX.1-dev guidance: its schema takes (1, 9]; the examples use 3.5-5.
_FLUX1_DEV_CFG_SCALE = 5.0


def _flux2_klein_payload(prompt: str, *, steps: int | None) -> dict:
    # klein's schema allows 1-4 steps (default 4), pins cfg_scale at 0 and
    # rejects unknown fields, so mode/samples/cfg_scale/negative_prompt are
    # omitted.
    return {
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "seed": 0,  # 0 = random
        "steps": max(1, min(int(steps or 4), 4)),
    }


def _flux1_dev_payload(prompt: str, *, steps: int | None) -> dict:
    # NVIDIA NIM FLUX.1-dev schema (see
    # https://docs.api.nvidia.com/nim/reference/black-forest-labs-flux_1-dev-infer).
    # Field names differ from the HuggingFace diffusers convention: using
    # num_inference_steps or guidance_scale here returns HTTP 422.
    return {
        "prompt": prompt,
        "mode": "base",
        "width": 1024,
        "height": 1024,
        "cfg_scale": _FLUX1_DEV_CFG_SCALE,
        "steps": max(5, min(int(steps or 50), 100)),
        "samples": 1,
        "seed": 0,
    }


_IMAGE_PAYLOAD_BUILDERS: dict[str, Callable[..., dict]] = {
    NIM_FLUX2_KLEIN: _flux2_klein_payload,
    NIM_FLUX1_DEV: _flux1_dev_payload,
}


def build_image_payload(model: str, prompt: str, options: Mapping[str, Any] | None = None) -> dict | None:
    """The Visual GenAI body for *model*; None for an id without a profile."""
    profile = get_profile("nvidia", model)
    builder = _IMAGE_PAYLOAD_BUILDERS.get(profile.protocol) if profile else None
    if builder is None:
        return None
    return builder(prompt, steps=(options or {}).get("steps"))


class NvidiaProvider:
    """Async client for NVIDIA's free inference API.

    Thinking is controlled only through ``chat_template_kwargs``
    (``enable_thinking``, plus the profile's extra kwargs and, where the
    model takes one, a top-level ``reasoning_budget``) or a top-level
    ``reasoning_effort``, see :func:`build_chat_payload`.
    ``parse_openai_response`` extracts reasoning from every response format.
    """

    name = "nvidia"
    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        health: ModelHealth,
        *,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.health = health
        self._thinking_budget = thinking_budget
        self._warned_unprofiled: set[str] = set()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # ``transport`` lets scripts/live_gate.py carry the exact requests
        # this class builds; production always uses httpx's default.
        self._client = httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT_SECONDS, headers=self._headers, transport=transport
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        disable_thinking: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        json_schema: dict | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> ProviderResponse:
        """Call NVIDIA chat completions API and return a unified response.

        ``options`` are the tier's (see :func:`build_chat_payload`); the
        chain has already folded ``options["thinking"]`` into
        ``disable_thinking``.

        Raises:
            SkipTierError: Images for a text-only model (no request is sent).
            ProviderError: A subclass per failure, see src/providers/base.py;
                EmptyCompletionError when a 200 carries nothing usable.
        """
        model = model.strip()
        if not model:
            raise PayloadError("<empty>", 400, "Model ID must not be empty", self.name)
        profile = chat_profile(model)
        if has_image_parts(messages) and "image" not in profile.modalities:
            # Refuse locally: the API would reject it after a round trip.
            raise SkipTierError(model, "text-only model cannot take image input", self.name)
        if not profile.known and model not in self._warned_unprofiled:
            self._warned_unprofiled.add(model)
            logger.warning(
                f"NVIDIA: no protocol profile for {model}; sending a generic "
                "OpenAI payload without thinking controls"
            )

        payload = build_chat_payload(
            model,
            normalize_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=not disable_thinking,
            options=options,
            thinking_budget=self._thinking_budget,
            tools=tools,
            tool_choice=tool_choice,
            json_schema=json_schema,
        )
        thinking = payload.get("chat_template_kwargs", {}).get("enable_thinking", False) or (
            payload.get("reasoning_effort", _EFFORT_OFF) != _EFFORT_OFF
        )

        logger.info(
            f"NVIDIA request → {model} "
            f"(temp={payload['temperature']}, top_p={payload.get('top_p')}, "
            f"max_tok={payload['max_tokens']}, thinking={thinking})"
        )
        self.health.record_request(self.name, model)
        data = await post_chat_completion(
            self._client,
            self.BASE_URL,
            payload,
            provider=self.name,
            model=model,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response = parse_openai_response(data, model=model, provider=self.name)
        if response.reasoning_content:
            logger.info(f"NVIDIA reasoning ← {model} ({len(response.reasoning_content)} chars)")
        usage = response.usage or {}
        logger.info(
            f"NVIDIA response ← {model} "
            f"(text={bool(response.text)}, finish={response.finish_reason}, "
            f"prompt_tokens={usage.get('prompt_tokens')}, completion_tokens={usage.get('completion_tokens')}, "
            f"reasoning_tokens={response.reasoning_tokens})"
        )
        return require_output(response)

    # ── Image Generation ──────────────────────────────────────

    async def generate_image(
        self,
        model: str,
        prompt: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ProviderResponse:
        """Call NVIDIA Visual GenAI NIM image generation API.

        Returns a ProviderResponse with a base64 data URL in images[], or
        no image when the model failed the request (so the caller can try
        its next model). The payload follows the model's
        exact id (:func:`build_image_payload`); ``options["steps"]`` is the
        image tier's step count.

        Image calls do not go through the fallback chain, so this method
        checks and records the model's health itself.

        Raises:
            SkipTierError: The model is dead, cooling down or over budget
                (no request is sent).
            ContentFilteredError: The model's safety filter withheld the
                image; the model itself stays healthy.
            ProviderError: A subclass per HTTP failure, see src/providers/base.py.
        """
        model = model.strip()
        if not model:
            raise PayloadError("<empty>", 400, "Model ID must not be empty", self.name)
        blocked = self.health.unavailable(self.name, model)
        if blocked is not None:
            raise SkipTierError(model, f"{blocked.kind}: {blocked.reason}", self.name)
        endpoint = f"{_IMAGE_GEN_BASE}/{model}"

        payload = build_image_payload(model, prompt, options)
        if payload is None:
            if model not in self._warned_unprofiled:
                self._warned_unprofiled.add(model)
                logger.warning(
                    f"NVIDIA: no payload builder for image model {model}; "
                    "sending only the prompt (the one field every schema requires)"
                )
            payload = {"prompt": prompt}

        logger.info(f"NVIDIA image request → {model} (steps={payload.get('steps')})")
        self.health.record_request(self.name, model)
        try:
            data = await post_chat_completion(
                self._client,
                endpoint,
                payload,
                provider=self.name,
                model=model,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except ProviderError as exc:
            logger.error(f"NVIDIA image error {exc.status} for {model}: {exc.detail}")
            self.health.record_failure(self.name, model, exc)
            raise
        self.health.record_success(self.name, model)

        base64_image, finish_reason = self._parse_image_response(data)

        if finish_reason == "CONTENT_FILTERED":
            logger.warning(f"NVIDIA image {model}: content filtered")
            raise ContentFilteredError(model, self.name)

        if finish_reason == "ERROR":
            logger.warning(f"NVIDIA image {model}: generation reported ERROR")
            return ProviderResponse(model=model)

        if not base64_image:
            logger.warning(f"NVIDIA image {model}: no image in response")
            return ProviderResponse(model=model)

        # Return as data URL so downstream can handle uniformly
        data_url = _to_data_url(base64_image)
        logger.info(f"NVIDIA image response ← {model} (success)")

        return ProviderResponse(
            images=[data_url],
            model=model,
        )

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


def _to_data_url(image_b64: str) -> str:
    if image_b64.startswith("data:"):
        return image_b64
    return f"data:{_sniff_image_mime(image_b64)};base64,{image_b64}"


def _sniff_image_mime(image_b64: str) -> str:
    """Mime type from the decoded magic bytes (NVIDIA FLUX returns JPEG)."""
    try:
        # 16 base64 chars decode to the 12 bytes a WEBP header needs.
        head = base64.b64decode(image_b64[:16], validate=False)
    except (binascii.Error, ValueError):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
