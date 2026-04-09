"""Orchestrator — the central dispatcher for the multi-agent system.

Analyses every incoming message and decides:
1. Which specialist agent should handle it
2. What output format (text / voice / image)
3. A task description to guide the specialist agent

Uses fast rules for obvious cases (zero LLM calls) and falls back to
LLM classification only for ambiguous text.
"""

from __future__ import annotations

import ast
import json
import re
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


# Keywords that suggest the user wants an image generated.
_IMAGE_GEN_KEYWORDS = re.compile(
    r"("
    # ── A: 生成動詞 + 圖片名詞（中間不限距離）──
    r"(?:幫我|請|幫忙|可以)?(?:繪製|生成|產生|創建|製作|設計|重新繪製).*?(?:圖|圖片|照片|海報|插畫|壁紙|頭像|logo|LOGO|貼圖|漫畫)"
    # ── B: 「畫」系列 ──
    r"|(?:幫我|請|幫忙|可以)?(?:重新畫|重新繪|重畫|改畫|再畫)"
    r"|(?:幫我|請|幫忙)畫"
    r"|畫(?:一[張幅個隻條匹朵]|[張幅個]|出)"
    # ── C: 「我想要/我要/給我/來 + 量詞 + 圖片名詞」──
    r"|(?:我想要|我要|給我|來)(?:一[張幅個]|[張幅個]).*?(?:圖|圖片|畫|照片|海報|插畫|壁紙|頭像|貼圖|漫畫)"
    # ── D: English ──
    r"|(?:please\s+)?(?:draw|paint|sketch|illustrate|generate|create|make|design|produce)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:image|picture|photo|illustration|poster|wallpaper|avatar|logo|icon|art|drawing|painting)"
    r"|generate\s+(?:a\s+|an\s+)?image"
    r"|create\s+(?:a\s+|an\s+)?(?:image|picture)"
    r"|draw\s+(?:me\s+)?(?:a\s+|an\s+)"
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
    r"|(?:畫|繪|生成)(?:軟體|工具|app|技巧|方法|步驟|式)"
    # "圖表" (chart), "地圖" (map) analysis
    r"|看(?:這[張個])?圖|分析(?:這[張個])?圖|圖(?:表|中|裡|上|內)"
    # Non-image usages of generation verbs
    r"|(?:製作|設計|創建|產生)(?:方法|理念|模式|帳號|錯誤|問題|人)"
    r")",
    re.IGNORECASE,
)

# Topics that usually require external lookup or time-sensitive verification
_WEB_SEARCH_TOPIC_KEYWORDS = re.compile(
    r"(時事|新聞|天氣|氣溫|降雨|降水|股價|股市|大盤|匯率|匯價|油價|金價|票價"
    r"|地址|營業時間|開盤|收盤|現價|賽果|比分|賽程|路況|交通|航班|班次"
    r"|地震|颱風|公告|政策|法規|疫情|選情|結果"
    r"|評價|推薦|排名|怎麼去|在哪|哪裡買|多少錢|價[格錢]"
    r"|review|recommend|price|where|how to get)",
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
_JSONISH_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "：": ":",
        "，": ",",
    }
)

# Simple queries that don't need thinking — greetings, short chitchat, trivial requests
_SIMPLE_QUERY_PATTERNS = re.compile(
    r"^("
    # Greetings / farewells
    r"(嗨|哈囉|你好|早安|午安|晚安|安安|掰掰|拜拜|再見|謝謝|感謝|好的|OK|ok|收到|了解|bye|hello|hi|hey|thanks|thank you|good morning|good night|gn|gm)"
    r")$",
    re.IGNORECASE,
)

# Follow-up indicators: messages that reference prior context and need LLM routing.
_FOLLOW_UP_INDICATORS = re.compile(
    r"("
    # Demonstratives / continuations referencing prior content
    r"那(?:個|呢)?|然後呢?|接下來呢?|繼續|還有(?:呢|嗎)?"
    r"|再說多一點|更多|多說一點|多講一點"
    # Questions about what was just said
    r"|什麼意思|為什麼|為何|怎麼說|怎麼辦|怎麼了"
    # Explicit back-references
    r"|上面|剛剛|前面|之前|上一(?:個|則|條|篇)|你剛"
    # Continuation / expansion requests
    r"|例如呢?|舉例|比如呢|詳細一點|再詳細一點|更詳細一點"
    r"|深入一點|再深入一點|更深入一點|補充說明|延伸說明"
    # English follow-ups
    r"|go on|continue|more|and then|what about|what do you mean|why is that|how so"
    r")",
    re.IGNORECASE,
)

# Explicitly complex text queries that benefit from reasoning/thinking.
_COMPLEX_QUERY_PATTERNS = re.compile(
    r"("
    # Explicit deep-analysis wording
    r"(?:詳細|深入|仔細|一步一步|逐步).*(?:分析|比較|評估|解釋|推理|拆解)"
    # Multi-step / planning / strategy
    r"|(?:多步驟|一步一步|逐步).*(?:規劃|計畫|策略|方案|實作|設計)"
    # Code / technical reasoning
    r"|(?:debug|troubleshoot|traceback|stack trace|exception|refactor)"
    r"|(?:程式|代碼|code).*(?:怎麼修|如何修|錯在哪|哪裡錯|排查|重構)"
    r"|(?:怎麼修|如何修|錯在哪|哪裡錯|排查).*(?:程式|代碼|code)"
    # Math / logic
    r"|(?:證明|推導|逐步計算|邏輯推理|數學證明)"
    r"|(?:prove|derive|logic puzzle|mathematical proof)"
    # Creative writing / long translation
    r"|(?:寫|撰寫).*(?:完整|長篇).*(?:文章|故事|報告|企劃|提案|信)"
    r"|翻譯.*(?:整段|整篇|全文|長文)"
    r"|translate.*(?:paragraph|article|full text|long passage)"
    r")",
    re.IGNORECASE,
)

# Image questions that still warrant thinking, even though image input usually stays non-thinking.
_COMPLEX_IMAGE_QUERY_PATTERNS = re.compile(
    r"("
    r"(?:錯在哪|哪裡錯|怎麼修|如何修|排查|故障|異常|崩潰)"
    r"|(?:traceback|stack trace|exception|troubleshoot|debug|diagnos)"
    r"|(?:合約|契約|財報|報表|文件|技術文件|錯誤截圖|日誌|log|console).*(?:分析|風險|問題|重點|排查|解釋)"
    r"|(?:contract|report|document|screenshot|log).*(?:analy[sz]e|risk|issue|problem|debug)"
    r")",
    re.IGNORECASE,
)

# Short image questions that are usually just asking for a direct description
_SIMPLE_IMAGE_QUERY_PATTERNS = re.compile(
    r"("
    r"這是什麼|這張(?:圖|圖片|照片|截圖)是什麼|請描述(?:一下)?|描述(?:一下)?"
    r"|圖裡有什麼|圖片裡有什麼|照片裡有什麼"
    r"|what(?:'s| is) this|what is in (?:the )?(?:image|picture|photo)"
    r"|describe (?:the )?(?:image|picture|photo)"
    r")",
    re.IGNORECASE,
)


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


def _load_jsonish_dict(candidate: str) -> dict | None:
    normalized = candidate.strip().translate(_JSONISH_TRANSLATION)

    try:
        parsed = json.loads(normalized)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    pythonish = re.sub(r"\btrue\b", "True", normalized, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)

    try:
        parsed = ast.literal_eval(pythonish)
        return parsed if isinstance(parsed, dict) else None
    except (SyntaxError, ValueError):
        return None


def _should_disable_thinking(text: str, *, has_image: bool = False) -> bool:
    """Heuristic: return True for queries that don't need reasoning.

    Default is non-thinking (return True). Thinking is only enabled for
    queries that explicitly match complex patterns requiring deep reasoning.

    Logic:
    - If text matches complex patterns → thinking ON (return False)
    - For image requests, only complex image analysis gets thinking
    - Everything else → thinking OFF (return True)
    """
    stripped = text.strip()
    if not stripped:
        return True  # Empty → no thinking needed

    # Complex signals get thinking
    if _COMPLEX_QUERY_PATTERNS.search(stripped):
        return False

    if has_image:
        if _SIMPLE_IMAGE_QUERY_PATTERNS.search(stripped):
            return True
        if _COMPLEX_IMAGE_QUERY_PATTERNS.search(stripped):
            return False
        return True

    # Default: no thinking — only complex patterns above trigger thinking
    return True


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


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif item.get("type") == "image_url":
                parts.append("[圖片]")
        return " ".join(parts).strip()

    return str(content).strip()


def _last_history_text(messages: list[dict] | None, role: str) -> str:
    for message in reversed(messages or []):
        if message.get("role") != role:
            continue
        text = _message_content_to_text(message.get("content", ""))
        if text:
            return text
    return ""


def _has_follow_up_context(request: AgentRequest) -> bool:
    return bool(
        request.previous_agent
        or request.conversation_history
        or request.quoted_message_id
        or request.quoted_text
        or request.quoted_image_base64
        or request.quoted_image_url
    )


def _looks_like_follow_up(request: AgentRequest) -> bool:
    if not request.text.strip() or not _has_follow_up_context(request):
        return False

    if (
        request.quoted_message_id
        or request.quoted_text
        or request.quoted_image_base64
        or request.quoted_image_url
    ):
        return True

    return bool(_FOLLOW_UP_INDICATORS.search(request.text))


def _infer_previous_agent(request: AgentRequest) -> str:
    if request.previous_agent in _VALID_AGENTS:
        return request.previous_agent

    last_user_text = _last_history_text(request.conversation_history, "user")
    last_assistant_text = _last_history_text(request.conversation_history, "assistant")
    history_blob = " ".join(
        part
        for part in (
            last_user_text,
            last_assistant_text,
            request.quoted_text.strip(),
        )
        if part
    )

    if any(
        marker in history_blob
        for marker in ("[使用者傳送圖片]", "[使用者引用圖片]", "[發送了圖片]", "[已傳送圖片]", "[圖片]")
    ):
        if _IMAGE_GEN_KEYWORDS.search(last_user_text) and not _IMAGE_GEN_NEGATIVES.search(last_user_text):
            return "image_gen"
        return "vision"

    if last_user_text and _IMAGE_GEN_KEYWORDS.search(last_user_text) and not _IMAGE_GEN_NEGATIVES.search(last_user_text):
        return "image_gen"

    if last_user_text and (_URL_PATTERN.search(last_user_text) or _needs_web_search(last_user_text)):
        return "web_search"

    return "chat"


def _infer_follow_up_output_format(request: AgentRequest, previous_agent: str) -> str:
    current_text = request.text.strip()

    if previous_agent == "image_gen":
        return "image"

    if _prefers_text_output(current_text):
        return "text"
    if _prefers_voice_output(current_text):
        return "voice"

    previous_output = request.previous_output_format.strip()
    if previous_output in {"text", "voice"}:
        return previous_output

    last_user_text = _last_history_text(request.conversation_history, "user")
    if _prefers_voice_output(last_user_text):
        return "voice"
    return "text"


def _clip_text(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}…"


def _build_follow_up_task_description(request: AgentRequest) -> str:
    current_text = request.text.strip() or "延續上一輪主題回答使用者"
    previous_task = request.previous_task_description.strip()
    if previous_task:
        return f"延續上一輪「{_clip_text(previous_task, 28)}」的脈絡，回應使用者續問：{current_text}"
    return f"延續上一輪主題，回應使用者續問：{current_text}"


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
