"""Chat Agent — handles general conversation, translation, coding, creative writing.

This is the default agent that handles all text-based tasks.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.utils.logger import logger


class ChatAgent(BaseAgent):
    name = "chat"

    async def process(self, request: AgentRequest) -> AgentResponse:
        self.call_count += 1
        messages = self._build_messages(request)

        logger.info(f"[{request.request_id}] ChatAgent processing")

        resp = await self.fallback_chain.generate(
            targets=self.targets,
            messages=messages,
            temperature=self.settings.chat_temperature,
            max_tokens=self.settings.chat_max_tokens,
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
