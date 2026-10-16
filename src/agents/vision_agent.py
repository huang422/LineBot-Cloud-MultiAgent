"""Vision Agent — analyses images and answers questions about visual content.

Runs on the vision role's tiers (src/runtime/model_roles.py): Colab qwen3.8
once it is READY, then Nemotron 3 Nano Omni (15 s) and muse-glimmer 30B
(35 s) on NVIDIA and the OpenRouter free dots-3 (20 s), each taking image
input (Nemotron Ultra and Super, the text tiers' NIM models, cannot). The
API tiers never think; a request that needs thinking lets Colab think
(COLAB_DEEP_THINK), then runs every tier with thinking off
(``BaseAgent.targets_for``). A quoted image that reaches the agent as a
URL (the message cache or GCS) goes straight to the API tiers: Colab takes
inline images only and skips it by design (``skip(by design: url image)``
in the summary line).

The prompt is kept near 4k tokens, as for every image request
(``BaseAgent._build_messages``, ``IMAGE_CONTEXT_CHARS``).
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
        tiers, deadline = self._plan(request)

        logger.info(f"[{request.request_id}] VisionAgent processing")

        resp = await self.fallback_chain.generate(
            targets=tiers,
            messages=messages,
            temperature=self.settings.vision_temperature,
            max_tokens=self.settings.vision_max_tokens,
            deadline=deadline,
            disable_thinking=request.disable_thinking,
        )

        return AgentResponse(
            text=resp.text,
            agent_name=self.name,
            model_used=resp.model,
            output_format=request.output_format,
        )
