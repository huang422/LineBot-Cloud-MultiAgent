"""Image Generation Agent — two-stage pipeline.

Stage 1: the utility tiers (text only, thinking off) refine the user's
         request into an optimised English image prompt.
Stage 2: the image role's tiers (NVIDIA Visual GenAI: FLUX.2 klein, then
         FLUX.1-dev) generate the actual image, first answer wins. Each tier
         runs within its own ``timeout_s`` and the turn deadline, in a 1:1
         chat and in a group alike: an image that misses the reply token
         goes out by push. A prompt one model's safety filter refuses goes
         to the next model; when none draws it, the user is told the filter
         blocked it (CONTENT_FILTERED_TEXT), not the generic failure.

The short caption sent with the image is optional: the utility tiers write
it alongside both stages, and it never holds up the image. A caption that
is not ready when the image is gets dropped for the default one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.agents.base_agent import BaseAgent
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.providers.base import (
    ContentFilteredError,
    ProviderError,
    RateLimitError,
    SkipTierError,
    TransientError,
)
from src.providers.fallback_chain import (
    AllModelsRateLimitedError,
    AllProvidersFailedError,
    FallbackChain,
    Tier,
)
from src.providers.messages import strip_image_parts
from src.utils import request_trace
from src.utils.logger import logger
from src.utils.request_trace import Attempt

if TYPE_CHECKING:
    from src.services.model_health import ModelHealth

_PROMPT_REFINE_SYSTEM = (
    "Convert the user's request into a detailed, optimized English prompt for a FLUX text-to-image model. "
    "Neither you nor the image model can see a reference image: if the user refers to one, carry the subject, composition and style cues they describe in words into the prompt, unless they ask to change them. "
    "Output ONLY the prompt text, nothing else. "
    "Include: subject, style, lighting, composition, mood, color palette, camera angle, and any relevant details. "
    "Use concrete, specific descriptors instead of vague words (e.g. 'golden hour side lighting' not just 'nice lighting'). "
    "Keep it under 200 words."
)

CONTENT_FILTERED_TEXT = "這個請求被圖片模型的安全過濾擋下了，沒辦法生成。可以換個描述再試試，例如避開品牌、真實人物或敏感內容。"


class ImageGenAgent(BaseAgent):
    name = "image_gen"
    # The prompt refiner draws what the user asked for; facts about the
    # user (hobbies, pets) would leak into unrelated images.
    sees_user_profile = False

    def __init__(
        self,
        settings: Settings,
        fallback_chain: FallbackChain,
        targets: Sequence[Tier] | None = None,
        image_tiers: Sequence[Tier] | None = None,
        *,
        health: ModelHealth | None = None,
    ) -> None:
        """``targets`` are the utility tiers (stage 1, text only).

        ``image_tiers`` are the image role: providers with
        ``generate_image()``, in order. ``health`` records a tier that runs
        out of time (the provider records every other outcome itself).
        """
        super().__init__(settings, fallback_chain, targets)
        self.image_tiers: list[Tier] = list(image_tiers or [])
        self._health = health

    async def process(self, request: AgentRequest) -> AgentResponse:
        self.call_count += 1
        logger.info(f"[{request.request_id}] ImageGenAgent processing")

        if not self.image_tiers:
            return AgentResponse(
                text="圖片生成服務未設定（需要 NVIDIA API Key）。",
                agent_name=self.name,
                model_used="",
                output_format="image",
            )

        # The caption depends only on the user's text: it runs alongside
        # both stages, and stage 2 never waits for it.
        desc_task = asyncio.create_task(self._generate_description(request))
        try:
            # ── Stage 1: Refine the prompt ──
            refined_prompt = await self._refine_prompt(request)
            logger.info(f"[{request.request_id}] Stage 1 refined prompt: {refined_prompt[:100]}...")

            # ── Stage 2: Generate the image, one image tier after another ──
            # An image is slow like a thinking answer: it may use the whole
            # turn in any chat, and goes out by push once the reply token
            # is gone.
            image_data, text, model_used = await self._generate_with_fallback(
                request, refined_prompt, deadline=request.turn_deadline()
            )
        finally:
            desc_task.cancel()  # late: the image goes out with the default caption

        if image_data:
            description = (
                desc_task.result() if desc_task.done() and not desc_task.cancelled() else None
            )
            text = description or "已根據你的需求生成圖片。"
        elif not text:
            text = "圖片生成暫時失敗，請稍後再試。"

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
        *,
        deadline: float,
    ) -> tuple[str | None, str | None, str]:
        """Try each image tier in order, returning (image_data, text, model_used).

        A tier runs for at most ``min(tier.timeout_s, time left)``; no tier
        starts with less than MIN_TIER_SECONDS before *deadline*
        (``time.monotonic()``). A tier that runs out of time counts as a
        transient failure of its model, and the next tier is tried; so is
        one whose safety filter refused the prompt. The tiers' outcomes go
        to the request's summary line as one walk.
        """
        attempts: list[Attempt] = []
        try:
            return await self._walk_image_tiers(request, refined_prompt, deadline, attempts)
        finally:
            request_trace.record_walk(attempts)

    async def _walk_image_tiers(
        self,
        request: AgentRequest,
        refined_prompt: str,
        deadline: float,
        attempts: list[Attempt],
    ) -> tuple[str | None, str | None, str]:
        last_error: Exception | None = None
        filtered = False

        for tier in self.image_tiers:
            image_model = tier.model
            started = time.monotonic()
            remaining = deadline - started
            if remaining < self.settings.min_tier_seconds:
                logger.warning(
                    f"[{request.request_id}] Stage 2 stopped: {remaining:.0f}s left, "
                    f"not enough for {image_model}"
                )
                attempts.append(request_trace.skipped(tier.label, "no time left"))
                break
            limit = min(tier.timeout_s, remaining)
            try:
                async with asyncio.timeout(limit):
                    resp = await tier.provider.generate_image(
                        model=image_model,
                        prompt=refined_prompt,
                        options=tier.options,
                    )
                image_data = resp.images[0] if resp.images else None
                if image_data:
                    logger.info(
                        f"[{request.request_id}] Stage 2 success with {image_model}"
                    )
                    attempts.append(Attempt(tier.label, request_trace.OK, time.monotonic() - started))
                    return image_data, None, resp.model or image_model
                logger.warning(
                    f"[{request.request_id}] Stage 2 {image_model} returned empty, trying next"
                )
                attempts.append(Attempt(tier.label, request_trace.EMPTY, time.monotonic() - started))
            except TimeoutError:
                attempts.append(Attempt(tier.label, request_trace.TIMEOUT, time.monotonic() - started))
                timeout = TransientError(
                    image_model, 0, f"no image within {limit:.3g}s", tier.provider_name
                )
                logger.warning(f"[{request.request_id}] Stage 2 {image_model} {timeout.detail}, trying next")
                if self._health is not None:
                    self._health.record_failure(tier.provider_name, image_model, timeout)
                last_error = timeout
            except RateLimitError as e:
                attempts.append(Attempt(tier.label, request_trace.RATE_LIMITED, time.monotonic() - started))
                logger.warning(
                    f"[{request.request_id}] Stage 2 {image_model} rate-limited, trying next"
                )
                last_error = e
            except ContentFilteredError as e:
                # The prompt, not the model: the next model may still draw it.
                attempts.append(
                    Attempt(tier.label, request_trace.error_outcome(e), time.monotonic() - started)
                )
                logger.warning(f"[{request.request_id}] Stage 2 {image_model} content filtered, trying next")
                filtered = True
            except SkipTierError as e:
                # Dead, cooling down or over budget: nothing was sent.
                attempts.append(request_trace.skipped(tier.label, e.detail))
                logger.info(f"[{request.request_id}] Stage 2 skipping {image_model} ({e.detail})")
                last_error = e
            except ProviderError as e:
                attempts.append(
                    Attempt(tier.label, request_trace.error_outcome(e), time.monotonic() - started)
                )
                logger.warning(
                    f"[{request.request_id}] Stage 2 {image_model} failed ({e.status}), trying next"
                )
                last_error = e

        # All models failed
        if filtered:
            return None, CONTENT_FILTERED_TEXT, ""
        if isinstance(last_error, ProviderError) and last_error.status in (401, 403):
            return None, "圖片生成功能驗證失敗，請檢查 NVIDIA_API_KEY 是否有效。", ""
        if isinstance(last_error, RateLimitError):
            return None, "圖片生成目前較忙，請稍後再試。", ""
        return None, "圖片生成暫時失敗，請稍後再試或換個描述。", ""

    async def _refine_prompt(self, request: AgentRequest) -> str:
        """Stage 1: the utility tiers turn the request into an English prompt.

        The utility tiers are text only, so image parts are removed first;
        the hosted FLUX models cannot use a reference image anyway.
        """
        messages = strip_image_parts(
            self._build_messages(
                request, system_override=self._build_refine_system_prompt(), sees_images=False
            )
        )

        try:
            with request_trace.phase(request_trace.IMAGE_PROMPT):
                resp = await self.fallback_chain.generate(
                    targets=self.targets,
                    messages=messages,
                    temperature=self.settings.image_gen_temperature,
                    max_tokens=self.settings.image_gen_max_tokens,
                    deadline=self._deadline_for(request, thinking=False),
                    disable_thinking=True,
                )
            return resp.text or request.text or "A beautiful creative image"
        except (AllModelsRateLimitedError, AllProvidersFailedError) as e:
            logger.warning(f"Prompt refinement failed, using raw text: {e}")
            return request.text or "A beautiful creative image"

    async def _generate_description(self, request: AgentRequest) -> str | None:
        """Generate a short user-friendly description to accompany the image."""
        system = (
            "你是圖片生成助手。根據使用者的原始請求，用繁體中文寫一段簡短說明（1-2句），"
            "描述你為他生成了什麼樣的圖片。語氣自然友善，不要提到技術細節或 prompt。"
        )
        user_text = request.text.strip() or "生成一張圖片"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        try:
            with request_trace.phase(request_trace.IMAGE_CAPTION):
                resp = await self.fallback_chain.generate(
                    targets=self.targets,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150,
                    disable_thinking=True,
                    deadline=self._deadline_for(request, thinking=False),
                )
            return resp.text.strip() if resp.text else None
        except Exception as e:
            logger.warning(f"Description generation failed: {e}")
            return None

    def _build_refine_system_prompt(self) -> str:
        return "\n\n".join(
            part for part in (self.system_prompt.strip(), _PROMPT_REFINE_SYSTEM) if part
        )
