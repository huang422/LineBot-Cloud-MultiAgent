"""Orchestrator — the central dispatcher for the multi-agent system.

Analyses every incoming message and decides:
1. Which specialist agent should handle it
2. What output format (text / voice / image)
3. A task description to guide the specialist agent

Uses fast rules for obvious cases (zero LLM calls) and falls back to
LLM classification only for ambiguous text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

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


# Keywords that suggest the user wants an image generated
_IMAGE_GEN_KEYWORDS = re.compile(
    r"("
    # Chinese: explicit generation verbs
    r"(?:幫我|請|幫忙|可以)?(?:畫|繪製|繪|生成|產生|做|創建|製作|設計|重畫|改畫|重新畫|重新繪|再畫)"
    r"(?:一[張幅個]|[張幅個]|出)?(?:圖|圖片|畫|照片|海報|插畫|壁紙|頭像|logo|LOGO|貼圖|漫畫)?"
    # Chinese: "I want a picture of..." — allow optional 的/.../的 between quantity and noun
    r"|(?:我想要|我要|給我|來)(?:一[張幅個]|[張幅個]).{0,6}(?:圖|圖片|畫|照片|海報|插畫|壁紙|頭像|貼圖|漫畫)"
    # Chinese: "generate/create image" compound
    r"|生成圖|產生圖|生成一|產生一|做一張|做張"
    # English
    r"|(?:please\s+)?(?:draw|paint|sketch|illustrate|generate|create|make|design|produce)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:image|picture|photo|illustration|poster|wallpaper|avatar|logo|icon|art|drawing|painting)"
    r"|generate\s+(?:a\s+|an\s+)?image"
    r"|create\s+(?:a\s+|an\s+)?(?:image|picture)"
    r"|draw\s+(?:a\s+|an\s+|me\s+)?"
    r")",
    re.IGNORECASE,
)

# Negative patterns that look like image_gen but aren't
_IMAGE_GEN_NEGATIVES = re.compile(
    r"("
    # "畫面" (screen/scene), "畫素" (pixel), "畫質" (quality) are NOT requests to draw
    r"畫面|畫素|畫質|畫風(?:是|分析|像|怎麼)|畫法"
    # Asking about drawing concepts, not requesting generation
    r"|(?:怎麼|如何|什麼是|怎樣)(?:畫|繪|生成|畫圖|繪圖)"
    # Drawing tools/techniques, not generation requests
    r"|(?:畫|繪|生成)(?:軟體|工具|app|技巧|方法|步驟)"
    # "圖表" (chart), "地圖" (map) analysis
    r"|看(?:這[張個])?圖|分析(?:這[張個])?圖|圖(?:表|中|裡|上|內)"
    r")",
    re.IGNORECASE,
)

# Topics that usually require external lookup or time-sensitive verification
_WEB_SEARCH_TOPIC_KEYWORDS = re.compile(
    r"(時事|新聞|天氣|氣溫|降雨|降水|股價|股市|大盤|匯率|匯價|油價|金價|票價|地址|營業時間|開盤|收盤|現價|賽果|比分|賽程|路況|交通|航班|班次|地震|颱風|公告|政策|法規|疫情|選情|結果)",
    re.IGNORECASE,
)

# Time-sensitive phrasings that imply the user needs current external facts
_WEB_SEARCH_TIME_PATTERNS = re.compile(
    r"("
    r"(今天|今日|目前|現在|最新|最近|近期).*(新聞|時事|消息|天氣|股價|股市|匯率|票價|營業時間|地址|賽果|比分|路況|航班|公告|政策|結果)"
    r"|"
    r"(新聞|時事|天氣|股價|股市|匯率|票價|營業時間|地址|賽果|比分|路況|航班|公告|政策|結果).*(今天|今日|目前|現在|最新|最近|近期)"
    r"|"
    r"最近.*發生什麼"
    r"|"
    r"近期.*發生什麼"
    r"|"
    r"today.*(news|weather|stock|price|result)"
    r"|"
    r"latest.*(news|weather|stock|price|exchange|hours|address|result)"
    r"|"
    r"current.*(weather|stock|price|exchange|hours|address|status)"
    r")",
    re.IGNORECASE,
)

# Keywords suggesting the user explicitly wants text output
_TEXT_OUTPUT_KEYWORDS = re.compile(
    r"(文字(?:回覆|回答|說明)?|用文字(?:回覆|回答|說明)?|純文字|打字(?:回覆|回答)?|text(?: reply| response)?)",
    re.IGNORECASE,
)

# Negations that should override any positive voice keyword match
_NEGATIVE_VOICE_OUTPUT_KEYWORDS = re.compile(
    r"((?:不要|別|不用|不必|先不要|請勿).{0,4}(?:語音|聲音|音訊|錄音|朗讀|讀出來|唸|念|說給我聽|講給我聽|voice|audio|tts))",
    re.IGNORECASE,
)

# Keywords suggesting the user wants voice output
_VOICE_OUTPUT_KEYWORDS = re.compile(
    r"("
    r"語音(?:回覆|回答|朗讀|讀出來|播放|說明)?"
    r"|用語音(?:回覆|回答|朗讀|讀出來|說|講|唸|念)?"
    r"|聲音(?:回覆|回答|朗讀|播放)?"
    r"|音訊(?:回覆|回答|播放)?"
    r"|錄音(?:回覆|回答|播放)?"
    r"|唸|念|朗讀|讀出來|唸給我|念給我|唸給我聽|念給我聽|說給我聽|講給我聽"
    r"|read.*aloud|speak(?: it)?|say.*out|voice(?: reply| response)?|audio(?: reply| response)?|tts"
    r")",
    re.IGNORECASE,
)

# URL pattern
_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_VALID_AGENTS = {"chat", "vision", "web_search", "image_gen"}
_VALID_OUTPUTS = {"text", "voice", "image"}


def _iter_json_candidates(text: str):
    seen: set[str] = set()

    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}") and candidate not in seen:
            seen.add(candidate)
            yield candidate

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}") and stripped not in seen:
        seen.add(stripped)
        yield stripped

    for start_match in re.finditer(r"\{", text):
        depth = 0
        in_string = False
        escaped = False

        for index in range(start_match.start(), len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_match.start(): index + 1].strip()
                    if candidate not in seen:
                        seen.add(candidate)
                        yield candidate
                    break


def _needs_web_search(text: str) -> bool:
    return bool(
        _WEB_SEARCH_TOPIC_KEYWORDS.search(text)
        or _WEB_SEARCH_TIME_PATTERNS.search(text)
    )


def _prefers_text_output(text: str) -> bool:
    return bool(
        _TEXT_OUTPUT_KEYWORDS.search(text)
        or _NEGATIVE_VOICE_OUTPUT_KEYWORDS.search(text)
    )


def _prefers_voice_output(text: str) -> bool:
    return not _prefers_text_output(text) and bool(_VOICE_OUTPUT_KEYWORDS.search(text))


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

    async def route(self, request: AgentRequest) -> RouterDecision:
        """Determine which agent and output format to use."""
        self.call_count += 1

        # ── Fast rules (no LLM call needed) ──────────────────
        decision = self._apply_fast_rules(request)
        if decision:
            logger.info(
                f"[{request.request_id}] Orchestrator fast-rule → "
                f"agent={decision.agent}, output={decision.output_format}"
            )
            return decision

        # ── LLM classification (ambiguous text only) ─────────
        decision = await self._llm_classify(request)
        logger.info(
            f"[{request.request_id}] Orchestrator LLM → "
            f"agent={decision.agent}, output={decision.output_format}"
        )
        return decision

    # ── Fast rules ───────────────────────────────────────────

    def _apply_fast_rules(self, request: AgentRequest) -> RouterDecision | None:
        has_image = request.input_type in (InputType.IMAGE, InputType.IMAGE_TEXT)
        text = request.text.strip()

        wants_image_gen = (
            _IMAGE_GEN_KEYWORDS.search(text)
            and not _IMAGE_GEN_NEGATIVES.search(text)
        )

        # Image with generation keywords
        if has_image and wants_image_gen:
            return RouterDecision("image_gen", "image", text, "image + gen keywords")

        # Image with no text or analysis text → vision
        if has_image and not text:
            return RouterDecision("vision", "text", "描述圖片內容", "image without text")

        if has_image:
            return RouterDecision("vision", "text", text, "image with question")

        # Text-only rules
        if wants_image_gen:
            return RouterDecision("image_gen", "image", text, "draw/generate keywords")

        if _URL_PATTERN.search(text):
            output = "voice" if _prefers_voice_output(text) else "text"
            return RouterDecision("web_search", output, text, "URL detected")

        if _needs_web_search(text):
            output = "voice" if _prefers_voice_output(text) else "text"
            return RouterDecision(
                "web_search",
                output,
                text,
                "time-sensitive or externally verifiable query",
            )

        if _prefers_voice_output(text):
            return RouterDecision("chat", "voice", text, "voice output keywords")

        if _prefers_text_output(text):
            return RouterDecision("chat", "text", text, "text output keywords")

        # No fast rule matched — need LLM
        return None

    # ── LLM classification ───────────────────────────────────

    async def _llm_classify(self, request: AgentRequest) -> RouterDecision:
        """Use a lightweight LLM to classify ambiguous text."""
        messages = self._build_messages(request)

        try:
            resp = await self.fallback_chain.generate(
                targets=self.targets,
                messages=messages,
                temperature=self.settings.orchestrator_temperature,
                max_tokens=self.settings.orchestrator_max_tokens,
                require_reasoning_tokens=self.settings.require_reasoning_tokens,
            )

            return self._parse_llm_response(resp.text or "", request.text)
        except Exception as e:
            logger.error(f"Orchestrator LLM classification failed: {e}")
            return RouterDecision("chat", "text", request.text, "fallback on error")

    def _parse_llm_response(self, text: str, user_text: str) -> RouterDecision:
        """Parse the JSON response from the orchestrator LLM."""
        for candidate in _iter_json_candidates(text):
            try:
                data = json.loads(candidate)
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

                return RouterDecision(
                    agent=agent,
                    output_format=output,
                    task_description=data.get("task_description") or user_text,
                    reasoning=data.get("reasoning", ""),
                )
            except json.JSONDecodeError:
                continue

        logger.warning(f"Orchestrator LLM returned unparseable response: {text[:200]}")
        return RouterDecision("chat", "text", user_text, "could not parse LLM response")
