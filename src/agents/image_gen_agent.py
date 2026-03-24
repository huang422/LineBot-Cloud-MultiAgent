"""Image Generation Agent — two-stage pipeline.

Stage 1: Qwen3.5 (NVIDIA) refines the user's request into an optimised
         English image generation prompt.
Stage 2: NVIDIA Stable Diffusion generates the actual image
         (SD3 Medium primary → SD3.5 Large fallback).
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.providers.fallback_chain import (
    AllModelsRateLimitedError,
    AllProvidersFailedError,
    FallbackChain,
    Target,
)
from src.providers.nvidia_provider import NvidiaProvider
from src.providers.openrouter_provider import (
    ProviderError,
    RateLimitError,
)
from src.utils.logger import logger

_PROMPT_REFINE_SYSTEM = (
    "Convert the user's request into a detailed, optimized English prompt for Stable Diffusion image generation. "
    "If a reference image is provided, preserve its important subject, composition, or style cues unless the user asks to change them. "
    "Output ONLY the prompt text, nothing else. "
    "Include: subject, style, lighting, composition, mood, color palette, camera angle, and any relevant details. "
    "Use concrete, specific descriptors instead of vague words (e.g. 'golden hour side lighting' not just 'nice lighting'). "
    "Prioritize quality tokens: 'masterpiece, best quality, highly detailed, sharp focus, professional'. "
    "Keep it under 200 words."
)

_DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, out of focus, deformed, disfigured, "
    "extra limbs, bad anatomy, bad proportions, watermark, text, signature, "
    "cropped, poorly drawn, mutation, ugly, duplicate"
)


class ImageGenAgent(BaseAgent):
    name = "image_gen"

    def __init__(
        self,
        settings: Settings,
        fallback_chain: FallbackChain,
        targets: list[Target] | None = None,
        nvidia_provider: NvidiaProvider | None = None,
    ) -> None:
        super().__init__(settings, fallback_chain, targets)
        self._nvidia = nvidia_provider

    async def process(self, request: AgentRequest) -> AgentResponse:
        self.call_count += 1
        logger.info(f"[{request.request_id}] ImageGenAgent processing (two-stage)")

        # ── Stage 1: Refine prompt via text-agent reasoning chain ──
        refined_prompt = await self._refine_prompt(request)
        logger.info(f"[{request.request_id}] Stage 1 refined prompt: {refined_prompt[:100]}...")

        # ── Stage 2: Generate image via NVIDIA SD (primary → fallback) ──
        if not self._nvidia:
            return AgentResponse(
                text="圖片生成服務未設定（需要 NVIDIA API Key）。",
                agent_name=self.name,
                model_used="",
                output_format="image",
            )

        image_models = [self.settings.image_gen_primary_model]
        fallback = self.settings.image_gen_fallback_model.strip()
        if fallback and fallback != self.settings.image_gen_primary_model:
            image_models.append(fallback)

        image_data, text, model_used = await self._generate_with_fallback(
            request, refined_prompt, image_models
        )

        if image_data and not text:
            text = "已根據你的需求生成圖片。"

        return AgentResponse(
            text=text,
            image_base64=image_data,
            agent_name=self.name,
            model_used=model_used,
            output_format="image",
        )

    async def _generate_with_fallback(
        self,
        request: AgentRequest,
        refined_prompt: str,
        image_models: list[str],
    ) -> tuple[str | None, str | None, str]:
        """Try each NVIDIA image model in order, returning (image_data, text, model_used)."""
        last_error: Exception | None = None

        for image_model in image_models:
            try:
                resp = await self._nvidia.generate_image(
                    model=image_model,
                    prompt=refined_prompt,
                    negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
                    steps=self.settings.image_gen_steps,
                    cfg_scale=self.settings.image_gen_cfg_scale,
                )
                image_data = resp.images[0] if resp.images else None
                model_used = resp.model or image_model
                if image_data or resp.text:
                    logger.info(
                        f"[{request.request_id}] Stage 2 success with {image_model}"
                    )
                    return image_data, resp.text, model_used
                logger.warning(
                    f"[{request.request_id}] Stage 2 {image_model} returned empty, trying next"
                )
            except RateLimitError as e:
                logger.warning(
                    f"[{request.request_id}] Stage 2 {image_model} rate-limited, trying next"
                )
                last_error = e
            except ProviderError as e:
                logger.warning(
                    f"[{request.request_id}] Stage 2 {image_model} failed ({e.status}), trying next"
                )
                last_error = e

        # All models failed
        if isinstance(last_error, ProviderError) and last_error.status in (401, 403):
            return None, "圖片生成功能驗證失敗，請檢查 NVIDIA_API_KEY 是否有效。", ""
        if isinstance(last_error, RateLimitError):
            return None, "圖片生成目前較忙，請稍後再試。", ""
        return None, "圖片生成暫時失敗，請稍後再試或換個描述。", ""

    async def _refine_prompt(self, request: AgentRequest) -> str:
        """Stage 1: Use the text-agent reasoning chain to refine the user's request."""
        messages = self._build_messages(
            request,
            system_override=self._build_refine_system_prompt(),
        )

        try:
            resp = await self.fallback_chain.generate(
                targets=self.targets,
                messages=messages,
                temperature=0.5,
                max_tokens=300,
                require_reasoning_tokens=self.settings.require_reasoning_tokens,
            )
            return resp.text or request.text or "A beautiful creative image"
        except (AllModelsRateLimitedError, AllProvidersFailedError) as e:
            logger.warning(f"Prompt refinement failed, using raw text: {e}")
            return request.text or "A beautiful creative image"

    def _build_refine_system_prompt(self) -> str:
        return "\n\n".join(
            part for part in (self.system_prompt.strip(), _PROMPT_REFINE_SYSTEM) if part
        )
