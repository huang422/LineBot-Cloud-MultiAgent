"""Chat Agent — handles general conversation, translation, coding, creative writing.

This is the default agent that handles all text-based tasks. When a
``ToolRegistry`` is supplied at construction time the agent runs the
agentic tool-calling loop so the model can actively pull longer-term
memory (``recall_memory``), persist user facts (``update_user_profile``)
or fall back to a web search when a query needs fresh data.

If the tool loop fails for any reason (provider lacks tool support,
JSON-decode error, hits ``max_iterations``) we fall back to the plain
``fallback_chain.generate`` path so the user always gets a reply.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.agents.tools import ToolContext, ToolRegistry
from src.agents.tool_runner import ToolLoopError, run_tool_loop
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.utils.logger import logger


class ChatAgent(BaseAgent):
    name = "chat"

    def __init__(
        self,
        settings,
        fallback_chain,
        targets=None,
        *,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(settings, fallback_chain, targets)
        self._tool_registry = tool_registry

    @property
    def tool_registry(self) -> ToolRegistry | None:
        return self._tool_registry

    def set_tool_registry(self, registry: ToolRegistry | None) -> None:
        self._tool_registry = registry

    async def process(self, request: AgentRequest) -> AgentResponse:
        self.call_count += 1
        messages = self._build_messages(request)

        logger.info(f"[{request.request_id}] ChatAgent processing")

        if self._tool_registry is not None and self._tool_registry.names():
            try:
                ctx = ToolContext(
                    user_id=request.user_id,
                    chat_id=request.group_id,
                    source_type=request.source_type,
                )
                loop_result = await run_tool_loop(
                    fallback_chain=self.fallback_chain,
                    targets=self.targets,
                    messages=messages,
                    registry=self._tool_registry,
                    context=ctx,
                    tool_choice="auto",
                    max_iterations=self.settings.tool_loop_max_iterations,
                    temperature=self.settings.chat_temperature,
                    max_tokens=self.settings.chat_max_tokens,
                    require_reasoning_tokens=self.settings.require_reasoning_tokens,
                    thinking_timeout=self.settings.thinking_timeout_seconds,
                    disable_thinking=request.disable_thinking,
                )
                resp = loop_result.response
                if loop_result.tool_calls_executed:
                    logger.info(
                        f"[{request.request_id}] ChatAgent tool loop "
                        f"used: {loop_result.tool_calls_executed}"
                    )
                return AgentResponse(
                    text=resp.text,
                    agent_name=self.name,
                    model_used=resp.model,
                    output_format=request.output_format,
                )
            except ToolLoopError as exc:
                logger.warning(
                    f"[{request.request_id}] Tool loop unusable, "
                    f"falling back to plain generate: {exc}"
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"[{request.request_id}] Tool loop raised "
                    f"{exc!r}, falling back to plain generate"
                )

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
