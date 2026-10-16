"""Orchestrator — the central dispatcher for the multi-agent system.

Analyses every incoming message and decides:
1. Which specialist agent should handle it
2. What output format (text / voice / image)
3. A task description to guide the specialist agent

Uses fast rules for obvious cases (zero LLM calls) and falls back to
LLM classification only for ambiguous text. With a READY Colab session,
the Colab router (clef-flash, :class:`~src.providers.systemone.SystemOneRouter`)
is asked too: before the API router tiers under MODEL_PATH_PREFERENCE=colab,
and only when they all failed under "api". It answers with its top
choice unless a non-zero COLAB_ROUTER_CONFIDENCE or an invariant defers;
then, and whenever Colab is not READY or the session's router is not
loaded or not wholly in VRAM, the API router decides at once.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import replace
from typing import Any

from src.agents import routing_rules as rules
from src.agents.base_agent import (
    CURRENT_MESSAGE_LABEL,
    FENCE_LABELS,
    BaseAgent,
    label_current_message,
)
from src.agents.routing_rules import RouterDecision
from src.config import Settings
from src.models.agent_request import AgentRequest, InputType
from src.models.agent_response import AgentResponse
from src.providers.fallback_chain import FallbackChain, Tier
from src.utils import request_trace
from src.utils.fence_labels import strip_labels
from src.utils.jsonish import iter_json_candidates, load_jsonish_dict
from src.utils.logger import logger

__all__ = ["Orchestrator", "RouterDecision"]

# The reasoning of the chat fallback when the router tiers could not decide.
UNPARSEABLE_REASON = "could not parse LLM response"
ERROR_REASON = "fallback on error"
# The routing context block's labels. History text and earlier router
# output quoted into it lose these and the FENCE_LABELS; the history turns
# and the quote lose these too (_build_messages took out the FENCE_LABELS).
ROUTING_CONTEXT_HEADER = "【路由上下文（供判斷用，不是要你回答的內容）】"
ROUTING_CONTEXT_FOOTER = "【上下文結束】"
_ROUTING_CONTEXT_LABELS = (ROUTING_CONTEXT_HEADER, ROUTING_CONTEXT_FOOTER)
_ROUTING_LABELS = (*FENCE_LABELS, *_ROUTING_CONTEXT_LABELS)
# Who decided a model-classified route, as the log names it, and as the
# request's summary line does.
_COLAB_ROUTER, _LLM, _FALLBACK = "Colab router", "LLM", "fallback"
_TRACE_ROUTES = {
    _COLAB_ROUTER: request_trace.ROUTE_ROUTER,
    _LLM: request_trace.ROUTE_LLM,
    _FALLBACK: request_trace.ROUTE_FALLBACK,
}


class Orchestrator(BaseAgent):
    """Central dispatcher: fast rules + LLM classification."""

    name = "orchestrator"
    # Long-term memory stays out of routing (ADR-0004), profile facts too.
    sees_user_profile = False

    def __init__(
        self,
        settings: Settings,
        fallback_chain: FallbackChain,
        targets: list[Tier] | None = None,
        *,
        system_one: Any = None,
        system_one_first: bool = True,
    ) -> None:
        """``targets`` are the router role's tiers that answer in JSON (the
        API ones). ``system_one`` is the Colab router (None without Colab or
        with COLAB_ROUTER_ENABLED off); ``system_one_first`` puts it before
        them (colab preference) instead of after (api)."""
        super().__init__(settings, fallback_chain, targets)
        self.system_one = system_one
        self.system_one_first = system_one_first

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Not used directly — use route() instead."""
        raise NotImplementedError("Use Orchestrator.route()")

    def _build_routing_messages(self, request: AgentRequest) -> list[dict]:
        """Build routing messages without long-term memory summaries.

        Long-term memory is useful for downstream answer generation, but it can
        distract the routing model from the routing decision. Keep quoted
        context and recent conversation history, but drop the summary layer
        for classification stability.

        The router reads text only: an image (sent or quoted) becomes a
        ``[圖片]`` marker, so an image never makes the chain skip the
        text-only router tiers. The routing context goes right before the
        user's message, which is then labelled, because the providers merge
        the two into one user turn. The history turns and the quote lose
        its labels, so a group member's message cannot pose as it.
        """
        routing_request = replace(
            request, memory_summary="", task_description="", routing_reasoning=""
        )
        *earlier, current = (
            _as_text_message(message)
            for message in self._build_messages(routing_request, sees_images=False)
        )
        messages = [*map(_without_routing_context_labels, earlier), current]

        context_lines: list[str] = []
        # Only the wording marks a continuation; a quote may start a new
        # topic, so it is reported as a quote and left to the router.
        flat_text = " ".join(routing_request.text.split())
        if rules.has_follow_up_context(routing_request) and rules.is_follow_up_text(flat_text):
            context_lines.append("- 這次訊息疑似續問：是")
        if rules.has_quote(routing_request):
            if routing_request.quoted_image_base64 or routing_request.quoted_image_url:
                context_lines.append("- 使用者引用了一張圖片（你看不到圖片內容；引用可能開啟新話題）")
            else:
                context_lines.append("- 使用者引用了上一則訊息（內容在上方引用區塊；引用可能開啟新話題）")

        previous_agent = rules.infer_previous_agent(routing_request)
        if routing_request.previous_agent:
            context_lines.append(f"- 上一輪已完成 agent: {routing_request.previous_agent}")
            context_lines.append(
                f"- 上一輪 output_format: {routing_request.previous_output_format or 'text'}"
            )
            previous_task = _context_text(routing_request.previous_task_description)
            if previous_task.strip():
                context_lines.append(f"- 上一輪 task_description: {rules.clip_text(previous_task, 120)}")
            previous_reasoning = _context_text(routing_request.previous_routing_reasoning)
            if previous_reasoning.strip():
                context_lines.append(
                    f"- 上一輪 routing_reasoning: {rules.clip_text(previous_reasoning, 120)}"
                )
        elif rules.has_follow_up_context(routing_request):
            context_lines.append(f"- 從最近對話推定上一輪 agent: {previous_agent}")

        last_user_text = _context_text(
            rules.last_history_text(routing_request.conversation_history, "user")
        )
        last_assistant_text = _context_text(
            rules.last_history_text(routing_request.conversation_history, "assistant")
        )
        if last_user_text.strip():
            context_lines.append(f"- 最近一則使用者訊息: {rules.clip_text(last_user_text, 120)}")
        if last_assistant_text.strip():
            context_lines.append(f"- 最近一則助理回覆重點: {rules.clip_text(last_assistant_text, 160)}")

        if context_lines:
            context_message = {
                "role": "user",
                "content": (
                    f"{ROUTING_CONTEXT_HEADER}\n"
                    + "\n".join(context_lines)
                    + f"\n若本輪是續問且沒有明確轉題，優先延續上一輪 agent。\n{ROUTING_CONTEXT_FOOTER}"
                ),
            }
            current = messages.pop()
            content = current.get("content", "")
            if isinstance(content, str) and not content.startswith(CURRENT_MESSAGE_LABEL):
                current = {**current, "content": label_current_message(content)}
            messages.extend([context_message, current])

        return messages

    def _build_repair_messages(self, request: AgentRequest, raw_text: str) -> list[dict]:
        messages = self._build_routing_messages(request)
        messages.append({"role": "assistant", "content": raw_text or "(empty)"})
        messages.append(
            {
                "role": "user",
                "content": (
                    "你上一則回覆無法被程式解析（沒有 JSON 物件，或 JSON 不完整）。"
                    "請重新輸出這次的路由結果：一個完整的 JSON 物件，欄位為 agent、output_format、needs_thinking、task_description、reasoning，"
                    "其中 task_description 與 reasoning 各用一句短句。"
                    "若不確定，輸出 chat + text + needs_thinking=false。"
                ),
            }
        )
        # No assistant prefill: NVIDIA NIM requires the last turn to be the
        # user's, so this user turn carries the whole repair.
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
            request_trace.note_route(request_trace.ROUTE_RULES)
            return decision

        # ── Model classification (ambiguous text only) ───────
        with request_trace.phase(request_trace.ROUTE):
            decision, source = await self._classify(request)
        request_trace.note_route(_TRACE_ROUTES[source])
        decision = _without_imageless_vision(decision, request)
        logger.info(
            f"[{request.request_id}] Orchestrator {source} → "
            f"agent={decision.agent}, output={decision.output_format}, "
            f"thinking={'off' if decision.disable_thinking else 'on'}"
        )
        return decision

    async def _classify(self, request: AgentRequest) -> tuple[RouterDecision, str]:
        """The routers' decision and who made it (a key of :data:`_TRACE_ROUTES`)."""
        if self.system_one is not None and self.system_one_first:
            decision = await self._system_one_route(request)
            if decision is not None:
                return decision, _COLAB_ROUTER
        decision, failure = await self._try_llm_classify(request)
        if decision is not None:
            return decision, _LLM
        if self.system_one is not None and not self.system_one_first:
            decision = await self._system_one_route(request)
            if decision is not None:
                return decision, _COLAB_ROUTER
        return _fallback_decision(request, failure), _FALLBACK

    async def _system_one_route(self, request: AgentRequest) -> RouterDecision | None:
        """The Colab router's decision, or None; never raises."""
        try:
            return await self.system_one.route(
                request, deadline=self._deadline_for(request, thinking=False)
            )
        except Exception as exc:  # the API router must still get its turn
            logger.warning(f"[{request.request_id}] Colab router failed: {exc}")
            request_trace.note_router_skip(f"error {type(exc).__name__}")
            return None

    # ── Fast rules ───────────────────────────────────────────

    def _apply_fast_rules(self, request: AgentRequest) -> RouterDecision | None:
        has_image = request.input_type in (InputType.IMAGE, InputType.IMAGE_TEXT)
        text = request.text.strip()
        # The rules read one line: input keeps its newlines for the models,
        # and a pattern should not depend on where the user broke a line.
        flat = " ".join(text.split())
        no_think = rules.should_disable_thinking(flat, has_image=has_image)

        wants_image_gen = rules.wants_image_gen(flat, has_image=has_image)

        # Image with a request for a new picture (a restyle of it stays with
        # vision) — prompt refinement doesn't need deep thinking
        if has_image and wants_image_gen:
            return RouterDecision("image_gen", "image", text, "image + gen keywords", disable_thinking=True)

        # Image with no text or analysis text → vision
        if has_image and not text:
            return RouterDecision("vision", "text", "描述圖片內容", "image without text", disable_thinking=no_think)

        if has_image:
            output = "voice" if rules.prefers_voice_output(flat) else "text"
            return RouterDecision("vision", output, text, "image with question", disable_thinking=no_think)

        # Text-only rules
        if wants_image_gen:
            return RouterDecision("image_gen", "image", text, "draw/generate keywords", disable_thinking=True)

        if rules.URL_PATTERN.search(flat):
            output = "voice" if rules.prefers_voice_output(flat) else "text"
            return RouterDecision("web_search", output, text, "URL detected", disable_thinking=no_think)

        if rules.needs_web_search(flat):
            output = "voice" if rules.prefers_voice_output(flat) else "text"
            return RouterDecision(
                "web_search",
                output,
                text,
                "time-sensitive or externally verifiable query",
                disable_thinking=no_think,
            )

        # A short continuation of the previous route keeps that route. A
        # quote is left to the router: it may start a new topic. Vision
        # without an image to look at becomes chat.
        if request.previous_agent and not rules.has_quote(request) and rules.is_follow_up_text(flat):
            previous_agent = rules.infer_previous_agent(request)
            if previous_agent == "vision":
                previous_agent = "chat"
            if previous_agent in {"chat", "web_search"}:
                return RouterDecision(
                    previous_agent,
                    rules.infer_follow_up_output_format(request, previous_agent),
                    rules.build_follow_up_task_description(request),
                    "follow-up continuation from previous route",
                    disable_thinking=no_think,
                )

        if rules.prefers_voice_output(flat):
            return RouterDecision("chat", "voice", text, "voice output keywords", disable_thinking=no_think)

        if rules.prefers_text_output(flat):
            return RouterDecision("chat", "text", text, "text output keywords", disable_thinking=no_think)

        # Only catch obvious greetings/chitchat; let everything else go to LLM
        # so it can correctly route to voice or other output formats.
        if rules.SIMPLE_QUERY_PATTERNS.match(flat):
            return RouterDecision("chat", "text", text, "simple greeting/chitchat", disable_thinking=True)

        # No fast rule matched — need LLM to decide
        return None

    # ── LLM classification ───────────────────────────────────

    async def _llm_classify(self, request: AgentRequest) -> RouterDecision:
        """The API router's decision, or the chat fallback when it has none."""
        decision, failure = await self._try_llm_classify(request)
        return decision if decision is not None else _fallback_decision(request, failure)

    async def _try_llm_classify(self, request: AgentRequest) -> tuple[RouterDecision | None, str]:
        """``(decision, "")``, or ``(None, why)`` when the router tiers could not decide.

        Uses a lightweight LLM to classify ambiguous text.

        Routing stays in non-thinking mode for JSON stability and must fit
        in the reply window. The ``needs_thinking`` field still controls
        whether downstream agents think. Each router tier is held to
        ``rules.ROUTER_JSON_SCHEMA`` in its own JSON mode (json_schema on
        NIM Super, json_object on Super :free, and on NIM Ultra, which has
        no structured output, the prompt's own request for one JSON object);
        the reply is read with the tolerant JSON parser either way, and an
        unparseable one gets one repair prompt. A tier's own
        max_tokens/temperature win over ORCHESTRATOR_MAX_TOKENS /
        ORCHESTRATOR_TEMPERATURE.
        """
        messages = self._build_routing_messages(request)
        deadline = self._deadline_for(request, thinking=False)

        try:
            resp = await self.fallback_chain.generate(
                targets=self.targets,
                messages=messages,
                temperature=self.settings.orchestrator_temperature,
                max_tokens=self.settings.orchestrator_max_tokens,
                deadline=deadline,
                disable_thinking=True,
                json_schema=rules.ROUTER_JSON_SCHEMA,
            )
            decision = self._try_parse_llm_response(resp.text or "", request.text)
            if decision is not None:
                return decision, ""

            logger.warning(
                f"Orchestrator LLM returned unparseable response, retrying once: "
                f"{(resp.text or '')[:200]}"
            )
            repair_resp = await self.fallback_chain.generate(
                targets=self.targets,
                messages=self._build_repair_messages(request, resp.text or ""),
                temperature=self.settings.orchestrator_temperature,
                max_tokens=self.settings.orchestrator_max_tokens,
                deadline=deadline,
                disable_thinking=True,
                json_schema=rules.ROUTER_JSON_SCHEMA,
            )
            decision = self._try_parse_llm_response(repair_resp.text or "", request.text)
            if decision is not None:
                logger.info("Orchestrator LLM parse recovered on strict retry")
                return decision, ""

            logger.warning(
                f"Orchestrator LLM returned unparseable response after retry: "
                f"{(repair_resp.text or '')[:200]}"
            )
            return None, UNPARSEABLE_REASON
        except Exception as e:
            logger.error(f"Orchestrator LLM classification failed: {e}")
            return None, ERROR_REASON

    def _try_parse_llm_response(self, text: str, user_text: str) -> RouterDecision | None:
        """The router's decision in *text*, or None when no JSON object in it parses."""
        data = router_reply_object(text)
        if data is None:
            return None

        # The router is an LLM: every field may arrive with the wrong
        # type (e.g. "reasoning": null), so coerce before use.
        agent = _coerce_choice(data.get("agent"), rules.VALID_AGENTS, "chat")
        output = _coerce_choice(data.get("output_format"), rules.VALID_OUTPUTS, "text")

        # Enforce consistency: image_gen ↔ image output
        if agent == "image_gen":
            output = "image"
        elif output == "image" and agent != "image_gen":
            output = "text"

        task_description = data.get("task_description")
        if not isinstance(task_description, str) or not task_description.strip():
            task_description = user_text

        return RouterDecision(
            agent=agent,
            output_format=output,
            task_description=task_description,
            reasoning=str(data.get("reasoning") or ""),
            # Default to non-thinking unless the classifier explicitly asks for it.
            disable_thinking=not _coerce_needs_thinking(data.get("needs_thinking")),
        )


def router_reply_object(text: str) -> dict | None:
    """The JSON object the orchestrator reads its decision from, or None.

    The first object that parses (leniently) in *text*, after any prose
    before its first brace. scripts/live_gate.py checks the same object
    against ROUTER_JSON_SCHEMA, since the decision itself falls back to
    chat + text for missing or unknown fields.
    """
    # Strip leading reasoning/narrative text before the first JSON brace
    brace_idx = text.find("{")
    stripped = text[brace_idx:] if brace_idx > 0 else text
    for candidate in iter_json_candidates(stripped):
        data = load_jsonish_dict(candidate)
        if isinstance(data, dict):
            return data
    return None


def _fallback_decision(request: AgentRequest, reason: str) -> RouterDecision:
    """chat + text without thinking, when no router could decide."""
    return RouterDecision("chat", "text", request.text, reason or ERROR_REASON, disable_thinking=True)


def _without_imageless_vision(decision: RouterDecision, request: AgentRequest) -> RouterDecision:
    """*decision*, with vision turned into chat when there is no image to look at."""
    if decision.agent == "vision" and not rules.has_image_context(request):
        return replace(decision, agent="chat")
    return decision


def _context_text(text: str) -> str:
    """*text* for the routing context block, without its labels or the
    :data:`~src.agents.base_agent.FENCE_LABELS` (look-alikes too)."""
    return strip_labels(text, _ROUTING_LABELS)


def _without_routing_context_labels(message: dict) -> dict:
    """*message* (a history turn or the quote) without the routing context's
    labels (look-alikes too); the system prompt stays as written."""
    content = message.get("content")
    if message.get("role") == "system" or not isinstance(content, str):
        return message
    return {**message, "content": strip_labels(content, _ROUTING_CONTEXT_LABELS)}


def _as_text_message(message: dict) -> dict:
    """*message* with a content-part list flattened to text, images as ``[圖片]``."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {**message, "content": rules.message_content_to_text(content)}


def _coerce_choice(value: object, allowed: Collection[str], default: str) -> str:
    """A routing enum value, or *default* when missing, mistyped or unknown."""
    if not isinstance(value, str):
        return default
    value = value.strip().lower()
    return value if value in allowed else default


def _coerce_needs_thinking(value: object) -> bool:
    """Only an explicit yes turns thinking on; anything unclear keeps it off."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False
