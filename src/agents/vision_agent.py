"""Vision Agent — analyses images and answers questions about visual content.

Uses NVIDIA Qwen3.5 397B as the primary vision model. When the primary NVIDIA
leg needs deeper reasoning, it swaps to NVIDIA Gemma 4 31B; OpenRouter Gemma 4
31B serves as the fallback.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.utils.logger import logger


class VisionAgent(BaseAgent):
    name = "vision"

    async def process(self, request: AgentRequest) -> AgentResponse:
        self.call_count += 1
        messages = self._build_messages(request)

        logger.info(f"[{request.request_id}] VisionAgent processing")

        resp = await self.fallback_chain.generate(
            targets=self.targets,
            messages=messages,
            temperature=self.settings.vision_temperature,
            max_tokens=self.settings.vision_max_tokens,
            require_reasoning_tokens=self.settings.require_reasoning_tokens,
            thinking_timeout=self.settings.thinking_timeout_seconds,
            disable_thinking=request.disable_thinking,
        )

        return AgentResponse(
            text=resp.text,
            agent_name=self.name,
            model_used=resp.model,
            output_format=request.output_format,
        )
