"""Orchestrator — the central dispatcher for the multi-agent system.

Analyses every incoming message and decides:
1. Which specialist agent should handle it
2. What output format (text / voice / image)
3. A task description to guide the specialist agent

Uses fast rules for obvious cases (zero LLM calls) and falls back to
LLM classification only for ambiguous text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.agents.base_agent import BaseAgent
from src.config import Settings
from src.models.agent_request import AgentRequest, InputType
from src.models.agent_response import AgentResponse
from src.providers.fallback_chain import FallbackChain, Target
from src.utils.logger import logger


@dataclass
class RouterDecision:
    agent: str  # chat | vision | web_search | image_gen
    output_format: str  # text | voice | image
    task_description: str = ""
    reasoning: str = ""
    disable_thinking: bool = False  # True = skip reasoning for simple queries


# The regex rulebook and the tolerant JSON reader live in their own
# modules; they are pure and change for different reasons than the LLM
# fallback path below. Re-exported under their original private names so
# existing callers and tests keep working.
from src.agents.routing_rules import (  # noqa: E402
    IMAGE_GEN_KEYWORDS as _IMAGE_GEN_KEYWORDS,
)
from src.agents.routing_rules import (
    IMAGE_GEN_NEGATIVES as _IMAGE_GEN_NEGATIVES,
)
from src.agents.routing_rules import (
    SIMPLE_QUERY_PATTERNS as _SIMPLE_QUERY_PATTERNS,
)
from src.agents.routing_rules import (
    URL_PATTERN as _URL_PATTERN,
)
from src.agents.routing_rules import (
    VALID_AGENTS as _VALID_AGENTS,
)
from src.agents.routing_rules import (
    VALID_OUTPUTS as _VALID_OUTPUTS,
)
from src.agents.routing_rules import (
    build_follow_up_task_description as _build_follow_up_task_description,
)
from src.agents.routing_rules import (
    clip_text as _clip_text,
)
from src.agents.routing_rules import (
    has_follow_up_context as _has_follow_up_context,
)
from src.agents.routing_rules import (
    infer_follow_up_output_format as _infer_follow_up_output_format,
)
from src.agents.routing_rules import (
    infer_previous_agent as _infer_previous_agent,
)
from src.agents.routing_rules import (
    last_history_text as _last_history_text,
)
from src.agents.routing_rules import (
    looks_like_follow_up as _looks_like_follow_up,
)
from src.agents.routing_rules import (
    needs_web_search as _needs_web_search,
)
from src.agents.routing_rules import (
    prefers_text_output as _prefers_text_output,
)
from src.agents.routing_rules import (
    prefers_voice_output as _prefers_voice_output,
)
from src.agents.routing_rules import (
    should_disable_thinking as _should_disable_thinking,
)
from src.utils.jsonish import (  # noqa: E402
    iter_json_candidates as _iter_json_candidates,
)
from src.utils.jsonish import (
    load_jsonish_dict as _load_jsonish_dict,
)


class Orchestrator(BaseAgent):
    """Central dispatcher: fast rules + LLM classification."""

    name = "orchestrator"

    def __init__(
        self,
        settings: Settings,
        fallback_chain: FallbackChain,
        targets: list[Target] | None = None,
    ) -> None:
        super().__init__(settings, fallback_chain, targets)

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Not used directly — use route() instead."""
        raise NotImplementedError("Use Orchestrator.route()")

    def _build_routing_messages(
        self,
        request: AgentRequest,
        *,
        system_override: str | None = None,
    ) -> list[dict]:
        """Build routing messages without long-term memory summaries.

        Long-term memory is useful for downstream answer generation, but it can
        distract the routing model away from the required single-line JSON
        output. Keep quoted context and recent conversation history, but drop
        the summary layer for classification stability.
        """
        routing_request = replace(request, memory_summary="")
        messages = self._build_messages(routing_request, system_override=system_override)

        context_lines: list[str] = []
        if _looks_like_follow_up(routing_request):
            context_lines.append("- 這次訊息疑似續問：是")

        previous_agent = _infer_previous_agent(routing_request)
        if routing_request.previous_agent:
            context_lines.append(f"- 上一輪已完成 agent: {routing_request.previous_agent}")
            context_lines.append(
                f"- 上一輪 output_format: {routing_request.previous_output_format or 'text'}"
            )
            if routing_request.previous_task_description.strip():
                context_lines.append(
                    f"- 上一輪 task_description: {_clip_text(routing_request.previous_task_description, 120)}"
                )
            if routing_request.previous_routing_reasoning.strip():
                context_lines.append(
                    f"- 上一輪 routing_reasoning: {_clip_text(routing_request.previous_routing_reasoning, 120)}"
                )
        elif _has_follow_up_context(routing_request):
            context_lines.append(f"- 從最近對話推定上一輪 agent: {previous_agent}")

        last_user_text = _last_history_text(routing_request.conversation_history, "user")
        last_assistant_text = _last_history_text(routing_request.conversation_history, "assistant")
        if last_user_text:
            context_lines.append(f"- 最近一則使用者訊息: {_clip_text(last_user_text, 120)}")
        if last_assistant_text:
            context_lines.append(f"- 最近一則助理回覆重點: {_clip_text(last_assistant_text, 160)}")

        if context_lines:
            context_message = {
                "role": "user",
                "content": (
                    "以下是供路由判斷使用的結構化上下文，不是要你回答的內容：\n"
                    + "\n".join(context_lines)
                    + "\n若本輪是續問且沒有明確轉題，優先延續上一輪 agent。"
                ),
            }
            insert_at = 1 if messages and messages[0].get("role") == "system" else 0
            messages.insert(insert_at, context_message)

        return messages

    def _build_repair_messages(self, request: AgentRequest, raw_text: str) -> list[dict]:
        repair_system = (
            f"{self.system_prompt}\n\n"
            "你上一則回覆無法被程式解析。\n"
            "這次只允許輸出單行 JSON 物件；不要輸出任何前後說明、Markdown、註解、換行、額外文字、分析過程或推理步驟。\n"
            "直接以 { 開頭輸出 JSON，不要有任何前導文字。"
        )
        messages = self._build_routing_messages(request, system_override=repair_system)
        messages.append({"role": "assistant", "content": raw_text or "(empty)"})
        messages.append(
            {
                "role": "user",
                "content": (
                    "請把你上一則回覆修正為可解析的單行 JSON。"
                    "欄位固定為 agent、output_format、needs_thinking、task_description、reasoning。"
                    "若不確定，請輸出保守結果：chat + text + needs_thinking=false。"
                    "直接輸出 JSON，第一個字元必須是 {"
                ),
            }
        )
        # Assistant prefill to force the model to start with JSON
        messages.append({"role": "assistant", "content": '{"agent":"'})
        return messages

    async def route(self, request: AgentRequest) -> RouterDecision:
        """Determine which agent and output format to use."""
        self.call_count += 1

        # ── Fast rules (no LLM call needed) ──────────────────
        decision = self._apply_fast_rules(request)
        if decision:
            logger.info(
                f"[{request.request_id}] Orchestrator fast-rule → "
                f"agent={decision.agent}, output={decision.output_format}, "
                f"thinking={'off' if decision.disable_thinking else 'on'}"
            )
            return decision

        # ── LLM classification (ambiguous text only) ─────────
        decision = await self._llm_classify(request)
        logger.info(
            f"[{request.request_id}] Orchestrator LLM → "
            f"agent={decision.agent}, output={decision.output_format}, "
            f"thinking={'off' if decision.disable_thinking else 'on'}"
        )
        return decision

    # ── Fast rules ───────────────────────────────────────────

    def _apply_fast_rules(self, request: AgentRequest) -> RouterDecision | None:
        has_image = request.input_type in (InputType.IMAGE, InputType.IMAGE_TEXT)
        text = request.text.strip()
        no_think = _should_disable_thinking(text, has_image=has_image)

        wants_image_gen = (
            _IMAGE_GEN_KEYWORDS.search(text)
            and not _IMAGE_GEN_NEGATIVES.search(text)
        )

        # Image with generation keywords — prompt refinement doesn't need deep thinking
        if has_image and wants_image_gen:
            return RouterDecision("image_gen", "image", text, "image + gen keywords", disable_thinking=True)

        # Image with no text or analysis text → vision
        if has_image and not text:
            return RouterDecision("vision", "text", "描述圖片內容", "image without text", disable_thinking=no_think)

        if has_image:
            return RouterDecision("vision", "text", text, "image with question", disable_thinking=no_think)

        # Text-only rules
        if wants_image_gen:
            return RouterDecision("image_gen", "image", text, "draw/generate keywords", disable_thinking=True)

        if _URL_PATTERN.search(text):
            output = "voice" if _prefers_voice_output(text) else "text"
            return RouterDecision("web_search", output, text, "URL detected", disable_thinking=no_think)

        if _needs_web_search(text):
            output = "voice" if _prefers_voice_output(text) else "text"
            return RouterDecision(
                "web_search",
                output,
                text,
                "time-sensitive or externally verifiable query",
                disable_thinking=no_think,
            )

        if _looks_like_follow_up(request):
            previous_agent = _infer_previous_agent(request)
            if previous_agent in {"chat", "vision", "web_search"}:
                return RouterDecision(
                    previous_agent,
                    _infer_follow_up_output_format(request, previous_agent),
                    _build_follow_up_task_description(request),
                    "follow-up continuation from previous route",
                    disable_thinking=no_think,
                )

        if _prefers_voice_output(text):
            return RouterDecision("chat", "voice", text, "voice output keywords", disable_thinking=no_think)

        if _prefers_text_output(text):
            return RouterDecision("chat", "text", text, "text output keywords", disable_thinking=no_think)

        # Only catch obvious greetings/chitchat; let everything else go to LLM
        # so it can correctly route to voice or other output formats.
        if _SIMPLE_QUERY_PATTERNS.match(text):
            return RouterDecision("chat", "text", text, "simple greeting/chitchat", disable_thinking=True)

        # No fast rule matched — need LLM to decide
        return None

    # ── LLM classification ───────────────────────────────────

    async def _llm_classify(self, request: AgentRequest) -> RouterDecision:
        """Use a lightweight LLM to classify ambiguous text.

        Routing stays in non-thinking mode for JSON stability. The
        ``needs_thinking`` field still controls whether downstream agents
        should switch to a reasoning-capable model.
        """
        messages = self._build_routing_messages(request)

        try:
            resp = await self.fallback_chain.generate(
                targets=self.targets,
                messages=messages,
                temperature=self.settings.orchestrator_temperature,
                max_tokens=self.settings.orchestrator_max_tokens,
                require_reasoning_tokens=False,
                thinking_timeout=self.settings.thinking_timeout_seconds,
                disable_thinking=True,
            )
            decision = self._try_parse_llm_response(resp.text or "", request.text)
            if decision is not None:
                return decision

            logger.warning(
                f"Orchestrator LLM returned unparseable response, retrying once: "
                f"{(resp.text or '')[:200]}"
            )
            repair_resp = await self.fallback_chain.generate(
                targets=self.targets,
                messages=self._build_repair_messages(request, resp.text or ""),
                temperature=self.settings.orchestrator_temperature,
                max_tokens=self.settings.orchestrator_max_tokens,
                require_reasoning_tokens=False,
                thinking_timeout=self.settings.thinking_timeout_seconds,
                disable_thinking=True,
            )
            # Prepend the assistant prefill so the response completes the JSON
            repair_text = '{"agent":"' + (repair_resp.text or "")
            decision = self._try_parse_llm_response(repair_text, request.text)
            if decision is not None:
                logger.info("Orchestrator LLM parse recovered on strict retry")
                return decision

            logger.warning(
                f"Orchestrator LLM returned unparseable response after retry: "
                f"{(repair_resp.text or '')[:200]}"
            )
            return RouterDecision(
                "chat",
                "text",
                request.text,
                "could not parse LLM response",
                disable_thinking=True,
            )
        except Exception as e:
            logger.error(f"Orchestrator LLM classification failed: {e}")
            return RouterDecision(
                "chat",
                "text",
                request.text,
                "fallback on error",
                disable_thinking=True,
            )

    def _try_parse_llm_response(self, text: str, user_text: str) -> RouterDecision | None:
        # Strip leading reasoning/narrative text before the first JSON brace
        stripped = text
        brace_idx = text.find("{")
        if brace_idx > 0:
            stripped = text[brace_idx:]

        for candidate in _iter_json_candidates(stripped):
            data = _load_jsonish_dict(candidate)
            if not isinstance(data, dict):
                continue

            agent = data.get("agent", "chat")
            output = data.get("output_format", "text")

            if agent not in _VALID_AGENTS:
                agent = "chat"

            if output not in _VALID_OUTPUTS:
                output = "text"

            # Enforce consistency: image_gen ↔ image output
            if agent == "image_gen":
                output = "image"
            elif output == "image" and agent != "image_gen":
                output = "text"

            # Default to non-thinking unless the classifier explicitly asks for it.
            needs_thinking = data.get("needs_thinking", False)
            if isinstance(needs_thinking, str):
                needs_thinking = needs_thinking.lower() not in ("false", "no", "0")

            return RouterDecision(
                agent=agent,
                output_format=output,
                task_description=data.get("task_description") or user_text,
                reasoning=data.get("reasoning", ""),
                disable_thinking=not needs_thinking,
            )

        return None

    def _parse_llm_response(self, text: str, user_text: str) -> RouterDecision:
        """Parse the JSON response from the orchestrator LLM."""
        decision = self._try_parse_llm_response(text, user_text)
        if decision is not None:
            return decision

        logger.warning(f"Orchestrator LLM returned unparseable response: {text[:200]}")
        return RouterDecision(
            "chat",
            "text",
            user_text,
            "could not parse LLM response",
            disable_thinking=True,
        )
