"""Chat Agent — handles general conversation, translation, coding, creative writing.

This is the default agent that handles all text-based tasks. When a
``ToolRegistry`` is supplied at construction time the agent runs the
agentic tool-calling loop so the model can actively pull longer-term
memory (``recall_memory``), persist user facts (``update_user_profile``)
or fall back to a web search when a query needs fresh data.

When the loop cannot answer, one plain ``fallback_chain.generate`` call
(no tools, the original messages, what is left of the same deadline, and
no model that already timed out in the loop) does: after a
:class:`~src.agents.tool_runner.ToolLoopError` (a misconfigured
registry, or no text even after the loop's final
``tool_choice="none"`` request), and after an ``AllProvidersFailedError``,
since a tier may refuse a request with tools or tool turns (HTTP 400)
and still answer one without. A ``DeadlineExceededError`` (no time left)
or an ``AllModelsRateLimitedError`` (every tier now cooling down)
propagates: a retry could not answer.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.agents.tool_runner import ToolLoopError, run_tool_loop
from src.agents.tools import ToolContext, ToolRegistry
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.providers.fallback_chain import (
    AllProvidersFailedError,
    DeadlineExceededError,
    TimedOutModels,
)
from src.utils.logger import logger


class ChatAgent(BaseAgent):
    name = "chat"

    def __init__(
        self,
        settings,
        fallback_chain,
        targets=None,
        *,
        thinking_targets=None,
        vision_targets=None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(
            settings,
            fallback_chain,
            targets,
            thinking_targets=thinking_targets,
            vision_targets=vision_targets,
        )
        self._tool_registry = tool_registry

    @property
    def tool_registry(self) -> ToolRegistry | None:
        return self._tool_registry

    def set_tool_registry(self, registry: ToolRegistry | None) -> None:
        self._tool_registry = registry

    async def process(self, request: AgentRequest) -> AgentResponse:
        self.call_count += 1
        messages = self._build_messages(request)
        tiers, deadline = self._plan(request)
        # Shared by the loop's requests and the plain retry after it.
        timed_out: TimedOutModels = {}

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
                    targets=tiers,
                    messages=messages,
                    registry=self._tool_registry,
                    context=ctx,
                    tool_choice="auto",
                    max_iterations=self.settings.tool_loop_max_iterations,
                    timed_out=timed_out,
                    temperature=self.settings.chat_temperature,
                    max_tokens=self.settings.chat_max_tokens,
                    deadline=deadline,
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
            except DeadlineExceededError:
                raise
            except (ToolLoopError, AllProvidersFailedError) as exc:
                logger.warning(
                    f"[{request.request_id}] Tool loop could not answer, "
                    f"falling back to plain generate: {exc}"
                )

        resp = await self.fallback_chain.generate(
            targets=tiers,
            messages=messages,
            temperature=self.settings.chat_temperature,
            max_tokens=self.settings.chat_max_tokens,
            deadline=deadline,
            disable_thinking=request.disable_thinking,
            timed_out=timed_out,
        )

        return AgentResponse(
            text=resp.text,
            agent_name=self.name,
            model_used=resp.model,
            output_format=request.output_format,
        )
