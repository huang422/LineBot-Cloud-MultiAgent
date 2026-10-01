"""The orchestrator's regex rulebook and routing heuristics.

Pure, side-effect-free classification logic: no LLM calls, no I/O. Kept
separate from :mod:`src.agents.orchestrator` so the rulebook can be read,
tested, and tuned without touching the LLM fallback path — the two change
for entirely different reasons.

Every function here answers one question about a message and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.agent_request import AgentRequest


@dataclass
class RouterDecision:
    """Where a message goes: the agent, the reply's form, and whether to think."""

    agent: str  # chat | vision | web_search | image_gen
    output_format: str  # text | voice | image
    task_description: str = ""
    reasoning: str = ""
    disable_thinking: bool = False  # True = skip reasoning for simple queries


def _en(words: str) -> str:
    """The English alternatives *words* (inflections spelled out) as whole
    words: "prove" not in "improve", "log" not in "logo" or "blog". ASCII
    words are bounded by letters, not \\b: a CJK character is a word
    character too ("用python", "幫我debug")."""
    return rf"(?<![A-Za-z])(?:{words})(?![A-Za-z])"


# 畫 inside a planning word (計畫, 企畫, 規畫, 策畫, 籌畫), a literary or
# writing word (刻畫, 勾畫, 描畫, 筆畫) or a genre (動畫, 漫畫) is not a
# drawing verb; 漫畫 is listed as an image noun where it is one. Nor is it
# one at the head of a noun: 畫家, 畫師, 畫作, 畫展, 畫廊, 畫筆, 畫布, ...
_DRAW = r"(?<![計企規策籌刻勾描筆動漫])畫(?![家師作展廊筆布室框冊])"
# Programming: drawing, generating or reading something out in code
# ("用 python 畫一個愛心", "產生 logout API 的程式碼", "把 json 檔讀出來")
# asks for code, not a picture or a spoken reply.
_CODE_CONTEXT = (
    "(?:"
    + _en(
        r"python|javascript|typescript|java|js|html|css|svg|canvas|sql|api|json|csv"
        r"|null|code|bug|regex|script|markdown|turtle|matplotlib|c\+\+"
    )
    + r"|程式|代碼|函式|函數|變數|腳本|演算法)"
)
# A how or why question asks for an explanation, not the thing itself:
# "如何設計一張好的海報", "製作LINE貼圖要怎麼上架", "請問how do I design a logo?".
_HOW_WHY = (
    r"(?:如何|怎麼|怎樣|為什麼|為何|什麼是|"
    + _en(r"how\s+(?:do|does|did|can|could|should|would|to)|why")
    + ")"
)
# An English image noun that names the thing itself, not one modifying
# another noun ("a drawing app", "a photo album", "an icon font").
_IMAGE_NOUN_EN = (
    r"(?:image|picture|photo|illustration|poster|wallpaper|avatar|logo|icon"
    r"|drawing|painting|portrait|sketch|cartoon)s?\b"
    r"(?!\s+(?:apps?|applications?|albums?|galler(?:y|ies)|fonts?|sets?|systems?|library|libraries"
    r"|editors?|viewers?|tools?|websites?|sites?|pages?|components?|uploads?|uploaders?|files?"
    r"|formats?|filters?|sizes?|resolution)\b)"
)

# Requests to generate an image. English verbs need a word boundary, an
# article and an image noun, so "make the icon bigger" and "withdraw" stay
# out; see IMAGE_GEN_NEGATIVES for what still looks alike.
IMAGE_GEN_KEYWORDS = re.compile(
    r"("
    # ── A: 生成動詞 + 圖片名詞（相距 15 字內）──
    r"(?:繪製|生成|產生|創建|製作|設計|重新繪製).{0,15}?"
    rf"(?:圖|照片|海報|插畫|壁紙|頭像|{_en('logos?')}|貼圖|漫畫)"
    # ── B: 「畫」系列（請 不在 邀請、申請、聘請 裡）──
    r"|重新畫|重新繪|重畫|改畫|再畫"
    rf"|(?:幫我|(?<![邀申聘])請|幫忙){_DRAW}"
    # A bare 畫一張 opens the request or follows 你/再/那/就: after
    # anyone else it tells what they did ("我女兒畫一張卡片送我", "老闆只會
    # 畫個大餅", "教我畫一隻貓").
    r"|(?:^|(?<=[\s，,。！!？?、～~：:你妳再先那就也]))"
    rf"{_DRAW}(?:一[張幅個隻條匹朵]|[張幅個隻]|出)"
    # ── C: 「我想要/我要/給我/來 + 量詞 + 圖片名詞」──
    # 來 opens the request ("來張…", "再來一張…"), not 後來/原來/本來, and
    # 個 is not 個人 ("我要一個人去拍照片").
    r"|(?:我想要|我要|給我|(?:^|(?<=[\s，,。！!？?、～~再先那就]))來)(?:一[張幅個]|[張幅個])(?!人)"
    rf".{{0,15}}?(?:圖|{_DRAW}|照片|海報|插畫|壁紙|頭像|貼圖|漫畫)"
    # ── D: English ──
    r"|\b(?:draw|paint|sketch|illustrate|generate|create|make|design|produce|render)\s+"
    rf"(?:me\s+)?an?\s+(?:[\w-]+\s+){{0,2}}?{_IMAGE_NOUN_EN}"
    rf"|\bgenerate\s+(?:an?\s+)?(?=images?\b){_IMAGE_NOUN_EN}"
    r")",
    re.IGNORECASE,
)

# English negatives are whole words (_en), also right next to CJK text:
# "我要設計一個logo，有什麼tips?", "Don't draw a picture, just explain".
_ADVICE_EN = _en(r"tips?|advice|suggestions?|recommend(?:s|ed|ing|ations?)?|should\s+i")
_DONT_DRAW_EN = _en(r"don['’]?t|do\s+not|no\s+need\s+to|stop") + (
    r"\s+(?:\w+\s+)?(?:draw|paint|sketch|illustrat|generat|creat|mak|design|produc|render)"
)
_IDIOM_DRAW_EN = _en(r"draw\s+(?:a\s+)?(?:conclusion|attention|line)s?")

# Negative patterns that look like image_gen but aren't
IMAGE_GEN_NEGATIVES = re.compile(
    r"("
    # "畫面" (screen/scene), "畫素" (pixel), "畫質" (quality) are NOT requests to draw
    r"畫面|畫素|畫質|畫風(?:是|分析|像|怎麼)|畫法"
    # A question about drawing or design, and drawing in code
    rf"|{_HOW_WHY}|哪些|哪裡|在哪|{_CODE_CONTEXT}"
    # Asking to be taught, advised or quoted: "教我畫一隻貓", "我要設計一個
    # logo，有什麼建議", "what software should I use?", "製作貼圖要多少錢"
    rf"|教我|教學|建議|推薦|技巧|訣竅|{_ADVICE_EN}|多少錢|價[格錢位]|費用|收費|報價"
    # Turning a drawing down: "不要再畫了", "不用幫我畫了", "Don't draw a
    # picture". 別 opens the clause (not 特別, 分別), and 要不要 offers one.
    r"|(?:(?<![要用需])不(?:要|用|必|需要)|(?:^|(?<=[\s，,。！!？?、～~：:你妳就也請託]))別)"
    rf"[^\s，,。！!？?]{{0,3}}?(?:{_DRAW}|繪|生成|產生|製作|設計)|{_DONT_DRAW_EN}"
    # Tables, timelines and schedules are text
    r"|表格|時間軸|時間線|(?:時程|課程?|行程|進度|時間|日程|排程|排班|值班|價目|對照|作息|預算|比較|對比)表"
    # Drawing tools/techniques, not generation requests ("畫圖app", not "畫apple")
    rf"|(?:畫|繪|生成)(?:軟體|工具|{_en('apps?')}|技巧|方法|步驟|式)"
    # A picture that already exists: one to look at ("幫我看這張設計圖",
    # "分析圖表") or one someone made ("這是AI生成的圖片嗎", "他畫的圖",
    # "我上禮拜設計了一張海報")
    r"|看(?:一下)?[這那]|分析(?:這[張個])?圖|圖(?:表|中|裡|上|內)"
    rf"|(?:繪製|生成|產生|創建|製作|設計|{_DRAW})(?:出來)?[的了]"
    # 圖 inside another word: 圖書館, 圖檔, 圖例, 圖層, 圖庫, 圖鑑; maps,
    # blueprints and intentions (地圖, 藍圖, 意圖, 企圖, 試圖, 拼圖); and
    # diagrams, which are code or text (結構圖, ER 圖, 路線圖, ...), and
    # some even without 圖 ("幫我畫一下系統架構", "幫我畫一張 UML", "幫我畫個流程")
    r"|圖(?:書|檔|例|層|庫|鑑)|[地藍意企試拼]圖"
    r"|流程|架構|時序|甘特|拓撲"
    r"|(?:結構|示意|關係|關聯|心智|思維導|時程|循序|類別|狀態|樹狀|長條|圓餅|折線"
    r"|雷達|散[佈布點]|直方|漏斗|魚骨|韋恩|文氏|泳道|路線|統計|組織|電路|網路)\s*圖"
    rf"|(?<![A-Za-z])ER\s*圖|{_en('uml|mermaid|plantuml|graphviz')}"
    # Non-image usages of generation verbs, and the people and tools that
    # do it (設計師, 製作者, 產生器)
    r"|(?:製作|設計|創建|產生)(?:方法|理念|模式|帳號|錯誤|問題|人|師|者|器)"
    # Comics to read, taking photos, highlighting, and idioms: 畫大餅
    # (empty promises), "draw a conclusion"
    r"|漫畫書|拍照|(?:畫|劃)(?:出|一下|個)?重點|重點(?:畫|劃)|畫(?:一?個|了)?大?餅"
    rf"|withdraw|{_IDIOM_DRAW_EN}"
    r")",
    re.IGNORECASE,
)

# With an image, turning that picture into something ("幫我畫成卡通風格",
# "把這張照片製作成貼圖") restyles it: the image model cannot see it, so
# it stays with vision. A redraw (重新畫成, 重畫成, 再畫成) asks for a new one.
IMAGE_RESTYLE = re.compile(r"(?<![新重再改])(?:畫|繪製?|製作|設計|生成|產生)成")

# Topics that almost always need an external, current lookup. Generic
# words (結果, 在哪, 推薦, 評價, review, where, price, ...) are left to the
# router, which can tell "推薦一本書" from a lookup.
_TOPIC = (
    r"(?:時事|新聞|天氣|氣溫|降雨機率|股價|股市|大盤|匯率|油價|金價|票價|營業時間"
    r"|開盤|收盤|賽果|比分|賽程|路況|航班|班次(?!數)|地震|颱風)"
)
WEB_SEARCH_TOPIC_KEYWORDS = re.compile(
    rf"({_TOPIC}|哪裡買"
    # "X怎麼去" ends the question; "怎麼去解決", "怎麼去除" are "how to"
    r"|怎麼去(?=[\s，,。！!？?～~]|$|[呢啊比最才])"
    r"|\bweather\b|\bexchange rates?\b|\bstock prices?\b|\bflight status\b|\bopening hours\b)",
    re.IGNORECASE,
)

# A topic word in a how/why question, a writing task or a field of study
# is general knowledge, not a lookup: "颱風是怎麼形成的", "地震的原理",
# "匯率是怎麼決定的", "比分怎麼算", "幫我寫一篇新聞稿", "新聞系要讀什麼".
# Nor is one the user turns away from: "今天好累，不想看新聞了".
WEB_SEARCH_NEGATIVES = re.compile(
    r"((?:怎麼|如何|怎樣)(?:形成|產生|發生|決定|計算|算|運作|判斷|定義|影響)"
    r"|原理|成因|的定義"
    r"|新聞(?:稿|系|學|寫作)|(?:寫|撰寫|擬|草擬).{0,6}(?:新聞|報導)"
    r"|(?:不想|不要|不用|不必|懶得|不再)再?(?:看|聽|查|管|關心|知道))",
)

_FRESH = r"(?:今天|今日|目前|現在|最新|最近|近期|即時)"
_FRESH_TOPIC = (
    r"(?:新聞|時事|消息|天氣|股價|股市|匯率|票價|營業時間|賽果|比分|路況|航班|公告|政策"
    r"|比賽結果|開獎|選舉結果)"
)

# Time-sensitive phrasings that imply the user needs current external facts
WEB_SEARCH_TIME_PATTERNS = re.compile(
    rf"({_FRESH}.{{0,10}}?{_FRESH_TOPIC}"
    rf"|{_FRESH_TOPIC}.{{0,10}}?{_FRESH}"
    r"|(?:最近|近期).{0,6}?發生(?:了)?什麼"
    r"|\btoday\b.{0,30}?\b(?:news|weather|stocks?|prices?|results?|scores?)\b"
    r"|\blatest\b.{0,30}?\b(?:news|weather|stocks?|prices?|exchange|hours|address|results?)\b"
    r"|\bcurrent\b.{0,30}?\b(?:weather|stocks?|prices?|exchange|hours|address|status)\b)",
    re.IGNORECASE,
)

# A topic is looked up only when the message asks for it: a question or a
# request ("台北今天天氣怎樣", "幫我查明天高雄天氣"), or a bare query that
# ends on the topic ("美元匯率", "今天台北天氣"). A message that only
# mentions one ("今天天氣真好，心情很好", "最近股市很差，我好焦慮") is left
# to the router.
WEB_SEARCH_REQUEST = re.compile(
    r"([？?]|嗎|呢|多少|幾(?!乎)|什麼|甚麼|啥|如何|怎樣|怎麼|哪|誰|會不會|有沒有|是不是|是否"
    r"|查|搜尋|告訴我|給我|想知道|幫我|幫忙|請問"
    r"|\b(?:what|how|when|where|who|which|tell me|show me|check|find|search|look up)\b"
    r"|^\s*(?:is|are|does|do|did|will|can|any)\b"
    r"|^\s*(?:the\s+)?(?:latest|current|today'?s)\b)",
    re.IGNORECASE,
)
WEB_SEARCH_BARE_QUERY = re.compile(rf"^[^，,。！!？?我]{{0,8}}?(?:{_TOPIC}|{_FRESH_TOPIC})[\s？?。]*$")

# Keywords suggesting the user explicitly wants text output
TEXT_OUTPUT_KEYWORDS = re.compile(
    r"((?:用|以)(?:純)?文字|純文字|文字(?:回覆|回答|說明)|打字(?:回覆|回答)"
    rf"|{_en('text (?:reply|response|only)')})",
    re.IGNORECASE,
)

# Negations that should override any positive voice keyword match
NEGATIVE_VOICE_OUTPUT_KEYWORDS = re.compile(
    r"((?:不要|別|不用|不必|先不要|請勿).{0,4}"
    r"(?:語音|聲音|音訊|錄音|朗讀|(?:唸|念)(?:出來|[^，,。！!？?]{0,15}?給我聽)|讀(?:出來|給我聽)"
    rf"|(?:說|講)給我聽|{_en('voices?|audio|tts')}))",
    re.IGNORECASE,
)

# Explicit requests for a spoken reply. Bare 念, 語音, 聲音, speak or
# "say ... out" are not enough: "我想念我的貓", "語音辨識怎麼做" and
# "Do you speak Japanese?" all want text. 念 ... 給我聽 is one ("念一下今天的
# 頭條新聞給我聽"), but not 想念, 懷念 ("好懷念阿嬤唱歌給我聽") or a
# reading that is over ("他念了一首詩給我聽").
_READ_ALOUD = r"(?<![想紀懷思觀概理悼掛惦])(?:唸|念)"
VOICE_OUTPUT_KEYWORDS = re.compile(
    r"("
    r"(?:用|以)語音(?:再|跟我|對我|給我|幫我)?(?:回覆|回答|朗讀|告訴|說|講|唸|念|讀)"
    rf"|{_READ_ALOUD}(?:出來(?!的)|[^，,。！!？?了過]{{0,15}}?給我聽)"
    # 讀出來 is a program reading data too ("讀出來的值都是 null"): only
    # an instruction that ends on it asks for a spoken reply.
    r"|讀給我聽|讀出來(?=給我聽|[\s。！!～~]*$)"
    r"|朗讀"
    r"|(?:說|講)給我聽"
    r"|\bread (?:it |this )?aloud\b"
    r"|\bvoice (?:reply|message|response)\b"
    r"|\baudio (?:reply|response)\b"
    r"|\btts\b"
    r")",
    re.IGNORECASE,
)

_TTS_TOPIC_EN = _en(r"tts\s+(?:engines?|models?|apis?|librar(?:y|ies)|services?|voices?|works?)")

# The spoken-reply words used as a topic, reported speech, a how/why
# question or a program's input, not as a request: "朗讀和寫作技巧", "how
# does tts work", "他說給我聽聽看", "這個字怎麼唸出來", "把 json 檔讀出來",
# "如何用語音告訴 Google 助理開燈". A match leaves the output format to the router.
VOICE_WORD_NOT_A_REQUEST = re.compile(
    r"(朗讀(?:和|與|跟|及|比賽|技巧|能力|課|的)"
    r"|(?:他|她|它|他們|她們|別人|人家)(?:說|講)給我聽"
    rf"|(?:{_HOW_WHY}|{_CODE_CONTEXT})[^，,。！!？?]{{0,15}}?(?:唸|念|讀)出來"
    rf"|{_HOW_WHY}[^，,。！!？?]{{0,15}}?(?:用|以)語音"
    rf"|{_en('how|what|why|which|when')}[^.?!]{{0,30}}?{_en('tts')}"
    rf"|{_TTS_TOPIC_EN})",
    re.IGNORECASE,
)

# A URL ends at whitespace, quotes, angle brackets, or CJK or full-width
# punctuation ("https://example.com/a，謝謝", "「https://example.com/b」").
# Its host also ends at the first CJK letter, since zh-TW writers put their
# question right after the link ("https://example.com這篇在講什麼"); its
# path, query and fragment keep them ("https://zh.wikipedia.org/wiki/臺北";
# httpx percent-encodes them). A question glued to a path is read as part
# of it: that page then fails to load, where cutting the path would read
# another page without a word.
_URL_STOP = (
    r"\s<>\"'"
    r"\u2018-\u201F\u2026"  # curly quotes, ellipsis
    r"\u3000-\u303F"  # CJK symbols and punctuation (、。「」)
    r"\uFF00-\uFFEF"  # full-width forms (，！？（）)
)
_CJK_LETTERS = (
    r"\u3040-\u30FF"  # kana
    r"\u3400-\u4DBF\u4E00-\u9FFF"  # CJK ideographs
    r"\uAC00-\uD7AF"  # Hangul
)
URL_PATTERN = re.compile(
    rf"https?://[^{_URL_STOP}{_CJK_LETTERS}/?#]+(?:[/?#][^{_URL_STOP}]*)?",
    re.IGNORECASE,
)
# ASCII punctuation that ends a sentence rather than the URL before it; a
# ")" that closes a "(" in the URL is part of it ("wiki/Python_(language)").
_URL_TRAILING_PUNCTUATION = ".,!?;:)]}>'\""
VALID_AGENTS = {"chat", "vision", "web_search", "image_gen"}
VALID_OUTPUTS = {"text", "voice", "image"}

# The router's reply as a JSON schema. Endpoints with structured output
# (NVIDIA Super, via response_format json_schema) are held to it; the
# others still get it described in the prompt and parsed tolerantly.
ROUTER_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "enum": sorted(VALID_AGENTS)},
        "output_format": {"type": "string", "enum": sorted(VALID_OUTPUTS)},
        "needs_thinking": {"type": "boolean"},
        "task_description": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["agent", "output_format", "needs_thinking", "task_description", "reasoning"],
    "additionalProperties": False,
}

# Simple queries that don't need thinking — greetings, short chitchat, trivial requests
SIMPLE_QUERY_PATTERNS = re.compile(
    r"^("
    # Greetings / farewells
    r"(嗨|哈囉|你好|早安|午安|晚安|安安|掰掰|拜拜|再見|謝謝|感謝|好的|OK|ok|收到|了解|bye|hello|hi|hey|thanks|thank you|good morning|good night|gn|gm)"
    r")$",
    re.IGNORECASE,
)

# Short follow-ups that only make sense as a continuation of the previous
# turn. Each one is the whole message (anchored at both ends, with a short
# fixed tail) and the message is at most FOLLOW_UP_MAX_CHARS characters,
# so "為什麼天空是藍的", "那我們明天見", "再來聊別的" or "還有什麼好看的電影"
# (a new question, a goodbye, a change of topic) are not taken for one;
# the router, which sees the history, decides everything else.
FOLLOW_UP_MAX_CHARS = 12
_FOLLOW_UP_END = r"[？?～~。！!…]*$"
_FOLLOW_UP_PARTICLE = r"(?:吧|啊|呀|嗎|呢|咧|勒|喔|哦)?"
_FOLLOW_UP_AMOUNT = (
    r"(?:一下|一次|一遍|一些|一點點?|一個|幾個|多一點|詳細一點|清楚一點|明白一點"
    r"|(?:一|幾)?個例子|例子|點|些|下|遍|次|個)?"
)
FOLLOW_UP_INDICATORS = re.compile(
    r"^(?:"
    # 那高雄呢 / 那如果下雨呢 / 那呢
    rf"那(?:麼|如果|要是)?[^，。！？?!,\s]{{0,8}}?[呢咧勒]{_FOLLOW_UP_END}"
    rf"|(?:然後|接下來|接著)(?:呢|咧|勒)?{_FOLLOW_UP_END}"
    rf"|還有(?:什麼|啥|別的|其他|其它)?的?{_FOLLOW_UP_PARTICLE}{_FOLLOW_UP_END}"
    rf"|繼續(?:說|講|寫|介紹|解釋|回答|下去)?{_FOLLOW_UP_PARTICLE}{_FOLLOW_UP_END}"
    r"|再(?:多說|多講|說明|說|講|解釋|補充|整理|介紹|找|查|舉|給我?|來|多|詳細|深入|具體|仔細|清楚)"
    rf"{_FOLLOW_UP_AMOUNT}{_FOLLOW_UP_PARTICLE}{_FOLLOW_UP_END}"
    rf"|更多(?:資訊|細節|例子|內容|資料)?{_FOLLOW_UP_PARTICLE}{_FOLLOW_UP_END}"
    rf"|(?:詳細|多說|多講|說多|講多|深入|具體)一點點?{_FOLLOW_UP_PARTICLE}{_FOLLOW_UP_END}"
    r"|(?:舉(?:個|一個|幾個)?例(?:子)?|例如|比如說?|補充說明|延伸說明|是?什麼意思)"
    rf"(?:一下)?{_FOLLOW_UP_PARTICLE}{_FOLLOW_UP_END}"
    r"|(?:go on|continue|tell me more|more(?: details)?|and then|what else)\b[\s?.!]*$"
    r")",
    re.IGNORECASE,
)

# Two keywords count when they are close ("請一步一步分析", "幫我寫一篇完整的
# 企劃書"). An unbounded ".*" between them backtracks over the rest of the
# message from every start, and two of them cube it: these run on every
# message, on the event loop, where 4,000 characters of "寫完整" took 20 s.
_NEAR = r".{0,30}"

# English words for tracking down a fault, and their inflections.
_FAULT_EN = r"debug(?:s|ged|ging|gers?)?|troubleshoot(?:s|ed|ing|ers?)?|tracebacks?|stack traces?|exceptions?"
_CODE_EN = _en("codes?")

# Explicitly complex text queries that benefit from reasoning/thinking.
# English keywords are whole words (_en): "improve" is not "prove", nor
# "barcode" code or "exceptional" an exception.
COMPLEX_QUERY_PATTERNS = re.compile(
    r"("
    # Explicit deep-analysis wording
    rf"(?:詳細|深入|仔細|一步一步|逐步){_NEAR}(?:分析|比較|評估|解釋|推理|拆解)"
    # Multi-step / planning / strategy
    rf"|(?:多步驟|一步一步|逐步){_NEAR}(?:規劃|計畫|策略|方案|實作|設計)"
    # Code / technical reasoning
    rf"|{_en(_FAULT_EN + '|refactor(?:s|ed|ing)?')}"
    rf"|(?:程式|代碼|{_CODE_EN}){_NEAR}(?:怎麼修|如何修|錯在哪|哪裡錯|排查|重構)"
    rf"|(?:怎麼修|如何修|錯在哪|哪裡錯|排查){_NEAR}(?:程式|代碼|{_CODE_EN})"
    # Math / logic
    r"|(?:證明|推導|逐步計算|邏輯推理|數學證明)"
    rf"|{_en('(?:dis)?prov(?:e[sdn]?|ing)|deriv(?:e[sd]?|ing)|logic puzzles?|mathematical proofs?')}"
    # Creative writing / long translation
    r"|(?:寫|撰寫).{0,20}(?:完整|長篇).{0,10}(?:文章|故事|報告|企劃|提案|信)"
    rf"|翻譯{_NEAR}(?:整段|整篇|全文|長文)"
    rf"|{_en('translat(?:e[sd]?|ing)')}{_NEAR}{_en('paragraphs?|articles?|full text|long passages?')}"
    r")",
    re.IGNORECASE,
)

# Image questions that still warrant thinking, even though image input
# usually stays non-thinking. English keywords are whole words: "logo",
# "login" and "blog" are not a log, nor "tissue" an issue.
COMPLEX_IMAGE_QUERY_PATTERNS = re.compile(
    r"("
    r"(?:錯在哪|哪裡錯|怎麼修|如何修|排查|故障|異常|崩潰)"
    rf"|{_en(_FAULT_EN + '|diagnos(?:e[sd]?|ing|is|tics?)')}"
    rf"|(?:合約|契約|財報|報表|文件|技術文件|錯誤截圖|日誌|{_en('log(?:s|ging)?|consoles?')})"
    rf"{_NEAR}(?:分析|風險|問題|重點|排查|解釋)"
    rf"|{_en('contracts?|report(?:s|ed|ing)?|document(?:s|ation)?|screenshots?|log(?:s|ging)?')}"
    rf"{_NEAR}{_en('analy[sz](?:e[sdr]?|ing)|risk(?:s|y)?|issues?|problem(?:s|atic)?|debug(?:s|ged|ging)?')}"
    r")",
    re.IGNORECASE,
)

_PHOTO_EN = r"(?:images?|pictures?|photo(?:s|graphs?)?)"
_DESCRIBE_EN = _en(
    rf"what(?:'s| is) this|what is in (?:the )?{_PHOTO_EN}|describe (?:the )?{_PHOTO_EN}"
)

# Short image questions that are usually just asking for a direct description
SIMPLE_IMAGE_QUERY_PATTERNS = re.compile(
    r"("
    r"這是什麼|這張(?:圖|圖片|照片|截圖)是什麼|請描述(?:一下)?|描述(?:一下)?"
    r"|圖裡有什麼|圖片裡有什麼|照片裡有什麼"
    rf"|{_DESCRIBE_EN}"
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


def wants_image_gen(text: str, *, has_image: bool = False) -> bool:
    """True for a request for a new picture; with an image, not a restyle of it."""
    if not IMAGE_GEN_KEYWORDS.search(text) or IMAGE_GEN_NEGATIVES.search(text):
        return False
    return not (has_image and IMAGE_RESTYLE.search(text))


def needs_web_search(text: str) -> bool:
    if WEB_SEARCH_NEGATIVES.search(text):
        return False
    mentions_topic = WEB_SEARCH_TOPIC_KEYWORDS.search(text) or WEB_SEARCH_TIME_PATTERNS.search(text)
    return bool(mentions_topic and (WEB_SEARCH_REQUEST.search(text) or WEB_SEARCH_BARE_QUERY.match(text)))


def prefers_text_output(text: str) -> bool:
    return bool(
        TEXT_OUTPUT_KEYWORDS.search(text)
        or NEGATIVE_VOICE_OUTPUT_KEYWORDS.search(text)
    )


def prefers_voice_output(text: str) -> bool:
    return (
        not prefers_text_output(text)
        and bool(VOICE_OUTPUT_KEYWORDS.search(text))
        and not VOICE_WORD_NOT_A_REQUEST.search(text)
    )


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


def has_quote(request: AgentRequest) -> bool:
    """True when the message quotes an earlier LINE message (text or image)."""
    return bool(
        request.quoted_message_id
        or request.quoted_text
        or request.quoted_image_base64
        or request.quoted_image_url
    )


def has_image_context(request: AgentRequest) -> bool:
    """True when an image (sent or quoted) reaches the agent with this message."""
    return bool(request.image_base64 or request.quoted_image_base64 or request.quoted_image_url)


def has_follow_up_context(request: AgentRequest) -> bool:
    """A follow-up needs something to follow: the previous route or a quote."""
    return bool(request.previous_agent or has_quote(request))


def is_follow_up_text(text: str) -> bool:
    """True for a short message that only continues the previous turn."""
    stripped = text.strip()
    return 0 < len(stripped) <= FOLLOW_UP_MAX_CHARS and bool(FOLLOW_UP_INDICATORS.match(stripped))


def looks_like_follow_up(request: AgentRequest) -> bool:
    """A quote, or a short continuation of the previous route."""
    if not request.text.strip() or not has_follow_up_context(request):
        return False
    return has_quote(request) or is_follow_up_text(request.text)


def extract_urls(text: str) -> list[str]:
    """The URLs in *text*, in order and without duplicates or trailing punctuation."""
    urls: list[str] = []
    for raw in URL_PATTERN.findall(text):
        end = len(raw.rstrip(_URL_TRAILING_PUNCTUATION))
        unclosed = raw.count("(", 0, end) - raw.count(")", 0, end)
        while unclosed > 0 and raw.startswith(")", end):
            end, unclosed = end + 1, unclosed - 1
        url = raw[:end]
        if url and url not in urls:
            urls.append(url)
    return urls


def strip_urls(text: str) -> str:
    """*text* without its URLs, whitespace collapsed."""
    return " ".join(URL_PATTERN.sub(" ", text).split())


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
        if wants_image_gen(last_user_text):
            return "image_gen"
        return "vision"

    if last_user_text and wants_image_gen(last_user_text):
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
