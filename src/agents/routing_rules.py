"""The orchestrator's regex rulebook and routing heuristics.

Pure, side-effect-free classification logic: no LLM calls, no I/O. Kept
separate from :mod:`src.agents.orchestrator` so the rulebook can be read,
tested, and tuned without touching the LLM fallback path — the two change
for entirely different reasons.

Every function here answers one question about a message and nothing else.
"""

from __future__ import annotations

import re

from src.models.agent_request import AgentRequest

# Keywords that suggest the user wants an image generated.
IMAGE_GEN_KEYWORDS = re.compile(
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
IMAGE_GEN_NEGATIVES = re.compile(
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
WEB_SEARCH_TOPIC_KEYWORDS = re.compile(
    r"(時事|新聞|天氣|氣溫|降雨|降水|股價|股市|大盤|匯率|匯價|油價|金價|票價"
    r"|地址|營業時間|開盤|收盤|現價|賽果|比分|賽程|路況|交通|航班|班次"
    r"|地震|颱風|公告|政策|法規|疫情|選情|結果"
    r"|評價|推薦|排名|怎麼去|在哪|哪裡買|多少錢|價[格錢]"
    r"|review|recommend|price|where|how to get)",
    re.IGNORECASE,
)

# Time-sensitive phrasings that imply the user needs current external facts
WEB_SEARCH_TIME_PATTERNS = re.compile(
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
TEXT_OUTPUT_KEYWORDS = re.compile(
    r"(文字(?:回覆|回答|說明)?|用文字(?:回覆|回答|說明)?|純文字|打字(?:回覆|回答)?|text(?: reply| response)?)",
    re.IGNORECASE,
)

# Negations that should override any positive voice keyword match
NEGATIVE_VOICE_OUTPUT_KEYWORDS = re.compile(
    r"((?:不要|別|不用|不必|先不要|請勿).{0,4}(?:語音|聲音|音訊|錄音|朗讀|讀出來|唸|念|說給我聽|講給我聽|voice|audio|tts))",
    re.IGNORECASE,
)

# Keywords suggesting the user wants voice output
VOICE_OUTPUT_KEYWORDS = re.compile(
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
URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
VALID_AGENTS = {"chat", "vision", "web_search", "image_gen"}
VALID_OUTPUTS = {"text", "voice", "image"}

# Simple queries that don't need thinking — greetings, short chitchat, trivial requests
SIMPLE_QUERY_PATTERNS = re.compile(
    r"^("
    # Greetings / farewells
    r"(嗨|哈囉|你好|早安|午安|晚安|安安|掰掰|拜拜|再見|謝謝|感謝|好的|OK|ok|收到|了解|bye|hello|hi|hey|thanks|thank you|good morning|good night|gn|gm)"
    r")$",
    re.IGNORECASE,
)

# Follow-up indicators: messages that reference prior context and need LLM routing.
FOLLOW_UP_INDICATORS = re.compile(
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
COMPLEX_QUERY_PATTERNS = re.compile(
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
COMPLEX_IMAGE_QUERY_PATTERNS = re.compile(
    r"("
    r"(?:錯在哪|哪裡錯|怎麼修|如何修|排查|故障|異常|崩潰)"
    r"|(?:traceback|stack trace|exception|troubleshoot|debug|diagnos)"
    r"|(?:合約|契約|財報|報表|文件|技術文件|錯誤截圖|日誌|log|console).*(?:分析|風險|問題|重點|排查|解釋)"
    r"|(?:contract|report|document|screenshot|log).*(?:analy[sz]e|risk|issue|problem|debug)"
    r")",
    re.IGNORECASE,
)

# Short image questions that are usually just asking for a direct description
SIMPLE_IMAGE_QUERY_PATTERNS = re.compile(
    r"("
    r"這是什麼|這張(?:圖|圖片|照片|截圖)是什麼|請描述(?:一下)?|描述(?:一下)?"
    r"|圖裡有什麼|圖片裡有什麼|照片裡有什麼"
    r"|what(?:'s| is) this|what is in (?:the )?(?:image|picture|photo)"
    r"|describe (?:the )?(?:image|picture|photo)"
    r")",
    re.IGNORECASE,
)


def should_disable_thinking(text: str, *, has_image: bool = False) -> bool:
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
    if COMPLEX_QUERY_PATTERNS.search(stripped):
        return False

    if has_image:
        if SIMPLE_IMAGE_QUERY_PATTERNS.search(stripped):
            return True
        if COMPLEX_IMAGE_QUERY_PATTERNS.search(stripped):
            return False
        return True

    # Default: no thinking — only complex patterns above trigger thinking
    return True


def needs_web_search(text: str) -> bool:
    return bool(
        WEB_SEARCH_TOPIC_KEYWORDS.search(text)
        or WEB_SEARCH_TIME_PATTERNS.search(text)
    )


def prefers_text_output(text: str) -> bool:
    return bool(
        TEXT_OUTPUT_KEYWORDS.search(text)
        or NEGATIVE_VOICE_OUTPUT_KEYWORDS.search(text)
    )


def prefers_voice_output(text: str) -> bool:
    return not prefers_text_output(text) and bool(VOICE_OUTPUT_KEYWORDS.search(text))


def message_content_to_text(content) -> str:
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


def last_history_text(messages: list[dict] | None, role: str) -> str:
    for message in reversed(messages or []):
        if message.get("role") != role:
            continue
        text = message_content_to_text(message.get("content", ""))
        if text:
            return text
    return ""


def has_follow_up_context(request: AgentRequest) -> bool:
    return bool(
        request.previous_agent
        or request.conversation_history
        or request.quoted_message_id
        or request.quoted_text
        or request.quoted_image_base64
        or request.quoted_image_url
    )


def looks_like_follow_up(request: AgentRequest) -> bool:
    if not request.text.strip() or not has_follow_up_context(request):
        return False

    if (
        request.quoted_message_id
        or request.quoted_text
        or request.quoted_image_base64
        or request.quoted_image_url
    ):
        return True

    return bool(FOLLOW_UP_INDICATORS.search(request.text))


def infer_previous_agent(request: AgentRequest) -> str:
    if request.previous_agent in VALID_AGENTS:
        return request.previous_agent

    last_user_text = last_history_text(request.conversation_history, "user")
    last_assistant_text = last_history_text(request.conversation_history, "assistant")
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
        if IMAGE_GEN_KEYWORDS.search(last_user_text) and not IMAGE_GEN_NEGATIVES.search(last_user_text):
            return "image_gen"
        return "vision"

    if last_user_text and IMAGE_GEN_KEYWORDS.search(last_user_text) and not IMAGE_GEN_NEGATIVES.search(last_user_text):
        return "image_gen"

    if last_user_text and (URL_PATTERN.search(last_user_text) or needs_web_search(last_user_text)):
        return "web_search"

    return "chat"


def infer_follow_up_output_format(request: AgentRequest, previous_agent: str) -> str:
    current_text = request.text.strip()

    if previous_agent == "image_gen":
        return "image"

    if prefers_text_output(current_text):
        return "text"
    if prefers_voice_output(current_text):
        return "voice"

    previous_output = request.previous_output_format.strip()
    if previous_output in {"text", "voice"}:
        return previous_output

    last_user_text = last_history_text(request.conversation_history, "user")
    if prefers_voice_output(last_user_text):
        return "voice"
    return "text"


def clip_text(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}…"


def build_follow_up_task_description(request: AgentRequest) -> str:
    current_text = request.text.strip() or "延續上一輪主題回答使用者"
    previous_task = request.previous_task_description.strip()
    if previous_task:
        return f"延續上一輪「{clip_text(previous_task, 28)}」的脈絡，回應使用者續問：{current_text}"
    return f"延續上一輪主題，回應使用者續問：{current_text}"
