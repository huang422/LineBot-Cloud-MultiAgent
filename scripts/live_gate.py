"""Manual live contract gate for the v3 model line-up.

A human runs this before a deploy, with the real keys and settings from .env.
It needs Python 3.12+ and the project deps, so run it from the .venv312 that
`UV_PROJECT_ENVIRONMENT=.venv312 uv sync --locked -p 3.12` builds from uv.lock
(see DEPLOY.md):

    .venv312/bin/python scripts/live_gate.py                    # every check
    .venv312/bin/python scripts/live_gate.py --path nvidia      # NVIDIA NIM only
    .venv312/bin/python scripts/live_gate.py --path roles       # each role's first tier
    .venv312/bin/python scripts/live_gate.py --path tiers       # every tier, one call each
    .venv312/bin/python scripts/live_gate.py --only nvidia.super.router

Every chat, embedding and image check goes through the bot's own code, so a
PASS means the requests the deployed bot sends work:

- providers come from ``src.runtime.composition.build_providers`` and the
  tiers from ``RoleTable.from_settings`` over them, configured from the
  same Settings the app reads (budgets, temperatures, the ``:free`` guard,
  MODEL_ROLES, MODEL_PATH_PREFERENCE, FORCE_TIER), so each call carries its
  tier's options (JSON mode, effort, reasoning budget, image steps);
- messages come from the real agents where the bot has a builder (chat,
  vision, orchestrator routing), and the router and tool checks run the
  real ``Orchestrator._llm_classify`` / ``run_tool_loop``;
- answers are read back by the providers' own parser.

Only the transport is swapped: a bridge carries each request over the
gate's HTTP client and keeps a copy of the exchange for evidence. It waits
in a worker thread, so the bot's own time limits (a tier's ``timeout_s`` in
the fallback chain) fire as they do in the bot: a model slower than its
tier fails its check.

Four kinds of checks:

- model checks (``nvidia.*``, ``openrouter.*``) pin one id of the v3
  defaults in one role. They SKIP when the configured table has no such
  tier (a MODEL_ROLES override) or FORCE_TIER switched it off, and FAIL
  when it is off for want of a key;
- role checks (``roles.<role>``, ``--path roles``) call each role's first
  enabled tier that is not Colab, whatever the table says it is;
- tier checks (``tier.<role>.<n>.<provider>:<model>``, ``--path tiers``)
  make one minimal call to every tier of the configured table, in call
  order: a routing decision the orchestrator's parser reads from JSON that
  holds to the router schema (nimble through SystemOneRouter), a zh-TW chat answer, a zh-TW reasoning answer
  with thinking on, the red circle described as red and round, a zh-TW
  memory summary, an embedding EMBEDDING_DIMENSIONS wide, image bytes.
  Each may take its tier's ``timeout_s`` plus TIER_SLACK_SECONDS. They are
  spaced to stay within the free per-minute limits (NVIDIA 40, OpenRouter
  20), and once OpenRouter answers with the account's daily free-model cap
  the remaining OpenRouter checks SKIP ("daily cap") without a request.
  Tiers switched off by FORCE_TIER or COLAB_ROUTER_ENABLED, and Colab tiers
  without a READY session, SKIP; a tier off for want of a key FAILs. Their
  own table lists role, tier number, model, result, latency and a note;
- Colab checks (``--path colab``) need a READY Colab session (ADR-0010).
  The gate reads the session document from Firestore with this machine's
  ADC (``gcloud auth application-default login`` as an account that can
  read the project's Firestore, never the Colab account), derives the
  bearer from COLAB_SHARED_SECRET in the env file, re-checks the tunnel
  rule, then runs the bot's readiness probe (with both models wholly in
  VRAM), a qwen3.8 chat reply at COLAB_CHAT_THINK, the red-circle vision
  check and a nimble routing decision through the bot's own
  OllamaChatProvider and SystemOneRouter.
  Without COLAB_ENABLED, a readable document or a READY session each
  check SKIPs with the reason; with COLAB_DRY_RUN (a stub Ollama) only the
  probe runs. The gate records no demand, so the session is still
  released on schedule.

The router checks send the bot's own JSON mode (``response_format``
json_schema on NIM Super, json_object on Super ``:free``, none on NIM Ultra,
which has no structured output and is asked for the JSON in the prompt).
The bot routes any JSON object (chat + text fill the gaps), so a router
check passes only when that object has every field of the schema, a known
agent and output, and a real task_description.
Two probes also try the shapes on top of the bot's Super router payload
without its own and report which ones work (``nvext.guided_json`` is
expected to fail). The Ultra thinking check expects ``medium_effort`` and no
``reasoning_budget`` (NIM answers HTTP 400 to one). The vision checks send
the red-circle PNG and give each model only its tier's time limit, as the
bot's fallback chain would (Nano Omni 15 s, muse-glimmer 35 s). Nano Omni's
workers are often busy: an HTTP 503 "ResourceExhausted" FAILs its check
(the bot moves on to muse-glimmer at once), so rerun it with ``--only``.

The script prints PASS/FAIL/SKIP with short evidence and latency, never API
keys or full model output, and exits 1 when any selected check fails. A full
run uses a little of the free quotas: about 33 NVIDIA and 15-18 OpenRouter
requests, of which the tier checks of the default table send 11 and 7.
Tests import the helpers with a mocked transport; tests and CI never send
live requests.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import io
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Run as ``python3 scripts/live_gate.py``: make ``src`` importable.
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import ValidationError  # noqa: E402

from src.agents.chat_agent import ChatAgent  # noqa: E402
from src.agents.orchestrator import Orchestrator, router_reply_object  # noqa: E402
from src.agents.routing_rules import ROUTER_JSON_SCHEMA, VALID_AGENTS, VALID_OUTPUTS  # noqa: E402
from src.agents.tool_runner import run_tool_loop  # noqa: E402
from src.agents.tools import (  # noqa: E402
    ToolContext,
    ToolDefinition,
    build_default_registry,
)
from src.agents.vision_agent import VisionAgent  # noqa: E402
from src.config import MIN_COLAB_SECRET_LENGTH, Settings  # noqa: E402
from src.models.agent_request import AgentRequest, InputType  # noqa: E402
from src.providers.base import (  # noqa: E402
    ProviderError,
    ProviderResponse,
    RateLimitError,
)
from src.providers.fallback_chain import (  # noqa: E402
    AllModelsRateLimitedError,
    AllProvidersFailedError,
    FallbackChain,
    Tier,
)
from src.providers.model_profiles import SYSTEMONE, get_profile  # noqa: E402
from src.providers.ollama_provider import OllamaChatProvider  # noqa: E402
from src.providers.systemone import RouterGates, SystemOneRouter  # noqa: E402
from src.runtime.composition import (  # noqa: E402
    build_model_health,
    build_providers,
    new_colab_endpoint,
)
from src.runtime.model_roles import (  # noqa: E402
    COLAB,
    COLAB_NIMBLE,
    COLAB_QWEN,
    ROLES,
    RoleConfigError,
    RoleTable,
    RoleTier,
    TierSpec,
    role_specs,
)
from src.runtime.model_roles import FLUX_DEV as FLUX_DEV  # noqa: E402
from src.runtime.model_roles import FLUX_KLEIN as FLUX_KLEIN  # noqa: E402
from src.runtime.model_roles import NIM_EMBED_1B as EMBED  # noqa: E402
from src.runtime.model_roles import NIM_MUSE as MUSE  # noqa: E402
from src.runtime.model_roles import NIM_OMNI as OMNI  # noqa: E402
from src.runtime.model_roles import NIM_SUPER as SUPER  # noqa: E402
from src.runtime.model_roles import NIM_ULTRA as ULTRA  # noqa: E402
from src.runtime.model_roles import OR_QWEN as OR_QWEN  # noqa: E402
from src.runtime.model_roles import OR_SUPER as OR_SUPER  # noqa: E402
from src.runtime.model_roles import OR_ULTRA as OR_ULTRA  # noqa: E402
from src.services.colab import protocol  # noqa: E402
from src.services.colab.endpoint import ColabEndpoint  # noqa: E402
from src.services.colab.state import (  # noqa: E402
    READY,
    ColabState,
    InMemoryColabStateStore,
    build_state_store,
)
from src.services.embedding_service import build_embedding_service  # noqa: E402
from src.services.memory_service import summary_messages  # noqa: E402
from src.utils.line_text import to_line_text  # noqa: E402

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# "roles" runs each role's first tier, "tiers" every tier of every role;
# "colab" needs a READY Colab session.
PATHS = ("nvidia", "openrouter", "colab", "roles", "tiers")
KEY_NAMES = {"nvidia": "NVIDIA_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

EMBED_DIMENSIONS = 2048
# NVIDIA_THINKING_BUDGET for the thinking checks: max_tokens headroom (and
# the reasoning_budget of the models that take one; Ultra does not).
THINKING_BUDGET = 2048
# What Ultra thinking must send (live gate 2026-09-30): thinking on at
# medium effort; no reasoning_budget anywhere.
ULTRA_THINKING_TEMPLATE_KWARGS = {"enable_thinking": True, "medium_effort": True}
SLOW_TIMEOUT_SECONDS = 180.0
TOOL_LOOP_ITERATIONS = 3

ROUTER_USER_MESSAGE = "今天台北天氣如何？下午會下雨嗎？"
# Decisions _llm_classify returns when the model could not route the message.
_ROUTER_FALLBACK_REASONS = ("could not parse LLM response", "fallback on error")

TIME_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "取得指定 IANA 時區目前的日期與時間",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "例如 Asia/Taipei"}},
            "required": ["timezone"],
        },
    },
}
TOOL_SYSTEM_PROMPT = "你是 LINE 聊天助理。需要現在時間時一定要呼叫工具，請用繁體中文回答。"
TOOL_RESULT = {"timezone": "Asia/Taipei", "now": "2026-09-29T15:04:00+08:00"}

# (prompt, text the answer should contain or None): zh-TW quality of a
# thinking answer, which runs on the thinking role's Ultra tier.
THINKING_PROMPTS: tuple[tuple[str, str | None], ...] = (
    (
        "水池有甲、乙兩個進水管，甲管單獨注滿要 6 小時，乙管單獨注滿要 3 小時。"
        "兩管同時打開，幾小時可以注滿？",
        "2",
    ),
    ("小明比小華大 3 歲，五年後兩人年齡加起來是 31 歲。小明現在幾歲？", "12"),
    ("請用三句話說明台灣夏天為什麼常有午後雷陣雨。", None),
)
VISION_QUESTION = "這張圖片裡有什麼形狀？是什麼顏色？請用繁體中文一句話回答。"
FLUX_PROMPT = "a red apple on a white wooden table, soft studio lighting, photorealistic"

# ── Tier checks (--path tiers) ──
TIER_CHECK_PREFIX = "tier."
# Seconds a tier check may run past its tier's timeout_s (building the
# clients, the bridge's hop to its worker thread).
TIER_SLACK_SECONDS = 2.0
# The free per-minute request limits (src/services/model_health.py): the
# tier checks are spaced so a run stays within them.
TIER_RPM: dict[str, int] = {"nvidia": 40, "openrouter": 20}
# OpenRouter names the account's daily free-model cap in its 429 body.
OPENROUTER_DAILY_CAP_MARKER = "free-models-per-day"
DAILY_CAP_NOTE = "daily cap: OpenRouter's free-models-per-day limit was reached in this run"
# The tier table's notes stay this short (no full model output).
TIER_NOTE_LIMIT = 80
# Least share of CJK among an answer's letters for a zh-TW answer.
ZH_MIN_CJK = 0.3
EMBED_TEXT = "台北今天會下雨嗎？"
# The recent dialogue of the memory summary check, as
# MemoryService._format_recent_for_summary writes it.
UTILITY_DIALOGUE = (
    "User: 我最近開始學鋼琴，每週六早上上課。\n"
    "Assistant: 很棒！固定練習會進步很快。\n"
    "User: 我也想在年底前去京都看楓葉，預算大約五萬元。"
)
_RED_RE = re.compile(r"紅|red", re.IGNORECASE)
_ROUND_RE = re.compile(r"圓|circle|round", re.IGNORECASE)

_THINK_PAIR_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CHANNEL_BLOCK_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL)
_KEY_LIKE_RE = re.compile(r"\b(?:nvapi-|sk-or-v1-|sk-or-)[A-Za-z0-9_\-]{8,}")
_WHITESPACE_RE = re.compile(r"\s+")


class GateError(Exception):
    """A check failed; the message is short evidence for the report."""


class GateSkip(Exception):
    """A check does not apply to this configuration; the message says why."""


# ── Helpers (pure; covered by tests/test_live_gate.py) ───────


def short(text: str, limit: int = 100) -> str:
    """Collapse whitespace and truncate, so evidence never carries full output."""
    flat = _WHITESPACE_RE.sub(" ", text or "").strip()
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return _KEY_LIKE_RE.sub("***", text)


@dataclass(frozen=True)
class ChatOutput:
    answer: str
    reasoning: str
    think_tags: bool
    channel_markers: bool
    finish_reason: str
    tool_calls: list


def read_chat_output(data: Mapping) -> ChatOutput:
    """Split a raw OpenAI-style completion into answer and reasoning.

    Reasoning may arrive as ``reasoning_content`` / ``reasoning``, inside
    ``<think>`` tags, before a lone ``</think>``, or as a
    ``<|channel>thought ... <channel|>`` block; none of it counts as answer.
    The checks use it on the raw response to see what the provider had to
    strip; the answer users get is the provider's own parse.
    """
    choices = data.get("choices") or [{}]
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    think_tags = "<think>" in content or "</think>" in content
    if "<think>" in content:
        reasoning += "".join(_THINK_PAIR_RE.findall(content))
        content = _THINK_PAIR_RE.sub("", content)
    if "</think>" in content:
        before, _, content = content.rpartition("</think>")
        reasoning += before
    channel_markers = bool(_CHANNEL_BLOCK_RE.search(content))
    content = _CHANNEL_BLOCK_RE.sub("", content)
    return ChatOutput(
        answer=content.strip(),
        reasoning=reasoning.strip(),
        think_tags=think_tags,
        channel_markers=channel_markers,
        finish_reason=str(choice.get("finish_reason") or ""),
        tool_calls=list(message.get("tool_calls") or []),
    )


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
    )


def cjk_ratio(text: str) -> float:
    """Share of CJK ideographs among letters and digits."""
    letters = [char for char in text if char.isalnum()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if _is_cjk(char)) / len(letters)


def has_traditional_chars(text: str) -> bool | None:
    """True when OpenCC t2s changes the text; None without OpenCC installed."""
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    return OpenCC("t2s").convert(text) != text


def sniff_image_mime(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "unknown"


def red_circle_png_base64(size: int = 256) -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), "white")
    margin = size // 8
    ImageDraw.Draw(image).ellipse(
        (margin, margin, size - margin, size - margin), fill=(220, 20, 20)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def reasoning_tokens(data: Mapping) -> int | None:
    details = (data.get("usage") or {}).get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return int(value) if isinstance(value, (int, float)) else None


def parse_route(text: str) -> dict:
    """Parse a JSON-mode router reply; GateError unless it is a complete route.

    Strict (the whole reply is the JSON): for the probes of constrained
    decoding. The bot's own reading is :func:`bot_route`.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        route = json.loads(cleaned)
    except ValueError as exc:
        raise GateError(f"not JSON ({len(cleaned)} chars)") from exc
    if not isinstance(route, dict):
        raise GateError("JSON is not an object")
    return check_route(route)


def bot_route(text: str) -> dict:
    """The route the orchestrator reads from *text*; GateError unless it is a complete one.

    The orchestrator's parser takes any JSON object and falls back to
    chat + text for what is missing or unknown, so a reply that ignores the
    schema (``{}``, an unknown agent, the prompt's template echoed back)
    would still route, all to chat. A router tier passes only when the
    object it answered holds to ROUTER_JSON_SCHEMA.
    """
    route = router_reply_object(text)
    if route is None:
        raise GateError(f"no JSON object in the reply ({len(text)} chars)")
    return check_route(route)


# The task_description of the prompt's own JSON example, and other text
# that describes no task.
_PLACEHOLDER_CHARS = " \t\r\n.…。"


def check_route(route: Mapping[str, Any]) -> dict:
    """*route* when it holds to ROUTER_JSON_SCHEMA; GateError naming what does not."""
    missing = [key for key in ROUTER_JSON_SCHEMA["required"] if key not in route]
    if missing:
        raise GateError("missing " + ",".join(missing))
    agent, output = route["agent"], route["output_format"]
    if not isinstance(agent, str) or agent not in VALID_AGENTS:
        raise GateError(f"agent={short(str(agent), 20)!r} is not a router agent")
    if not isinstance(output, str) or output not in VALID_OUTPUTS:
        raise GateError(f"output_format={short(str(output), 20)!r} is not a router output")
    task = route["task_description"]
    if not isinstance(task, str) or not task.strip(_PLACEHOLDER_CHARS):
        raise GateError(
            f"task_description={short(str(task), 20)!r} describes no task (the prompt's placeholder?)"
        )
    return dict(route)


def untested_overrides(settings: Settings) -> list[str]:
    """Env keys that change the models the gate checks (names only)."""
    keys = []
    if settings.model_roles.strip():
        keys.append("MODEL_ROLES")
    if settings.force_tier:
        keys.append("FORCE_TIER")
    return keys


def role_table(settings: Settings, providers: Mapping[str, Any]) -> RoleTable:
    """The app's role table over *providers*; GateError when the app would refuse it."""
    try:
        return RoleTable.from_settings(settings, providers)
    except RoleConfigError as exc:
        raise GateError(f"role table: {short(str(exc), 150)}") from None


def _usable(row: RoleTier) -> Tier:
    """*row*'s tier; GateSkip when switched off on purpose, GateError otherwise."""
    if row.tier is not None:
        return row.tier
    if row.spec.provider == COLAB or row.disabled_reason.startswith("FORCE_TIER"):
        raise GateSkip(f"{row.label} is off in the {row.role} role ({row.disabled_reason})")
    raise GateError(f"{row.label} is off in the {row.role} role ({row.disabled_reason})")


def role_tier(table: RoleTable, role: str, path: str, model: str) -> Tier:
    """The bound *role* tier for *path*:*model*, as the app would call it."""
    for row in table.rows(role):
        if row.spec.provider == path and row.spec.model == model:
            return _usable(row)
    raise GateSkip(f"{path}:{model} is not in the configured {role} role (MODEL_ROLES)")


def tier_timeout(settings: Settings, role: str, path: str, model: str) -> float | None:
    """The configured time limit of the *role* tier *path*:*model*, if it has one."""
    try:
        specs = role_specs(settings)
    except RoleConfigError:
        return None  # role_table() reports the configuration itself
    for spec in specs[role]:
        if spec.provider == path and spec.model == model:
            return spec.timeout_s
    return None


def first_api_tier(table: RoleTable, role: str) -> RoleTier:
    """*role*'s first enabled tier that does not wait on Colab."""
    rows = table.rows(role)
    for row in rows:
        if row.enabled and row.spec.provider != COLAB:
            return row
    reasons = "; ".join(f"{row.label}: {row.disabled_reason}" for row in rows if not row.enabled)
    raise GateError(f"no enabled non-Colab tier ({reasons or 'none configured'})")


def tier_kwargs(tier: Tier, *, disable_thinking: bool) -> dict[str, Any]:
    """What FallbackChain passes the provider for *tier*.

    The tier's own ``thinking`` option wins over the call's flag.
    """
    thinking = tier.options.get("thinking")
    return {
        "options": dict(tier.options),
        "disable_thinking": disable_thinking if thinking is None else not thinking,
    }


# ── HTTP context ─────────────────────────────────────────────


@dataclass
class GateContext:
    client: httpx.Client
    nvidia_key: str
    openrouter_key: str
    timeout: float = 120.0
    sleep: Callable[[float], None] = time.sleep
    # Measures a bot call against its tier's time limit.
    clock: Callable[[], float] = time.monotonic
    # The app's Settings (from the gate's env file); code defaults when None.
    settings: Settings | None = None
    # Reads the Colab session document (default: load_colab_state, Firestore
    # with this machine's ADC); tests pass a stand-in.
    colab_loader: Callable[[Settings], ColabState] | None = None
    poll_count: int = field(default=0, init=False)
    # The READY session, or the GateSkip/GateError saying why there is none;
    # resolved once per run by colab_session().
    colab_state: ColabState | GateSkip | GateError | None = field(default=None, init=False)
    # COLAB_SHARED_SECRET, the session's bearer and its tunnel host.
    colab_secrets: list[str] = field(default_factory=list, init=False)
    # clock() of each provider's last paced check (see pace()).
    paced_at: dict[str, float] = field(default_factory=dict, init=False)
    # The check that met OpenRouter's daily free-model cap; the OpenRouter
    # checks after it SKIP without a request.
    openrouter_capped_by: str = field(default="", init=False)

    def pace(self, provider: str) -> None:
        """Wait until a request to *provider* keeps within its free per-minute limit."""
        rpm = TIER_RPM.get(provider)
        if not rpm:
            return
        interval = 60.0 / rpm
        now = self.clock()
        last = self.paced_at.get(provider)
        if last is not None and now - last < interval:
            self.sleep(interval - (now - last))
            now = max(self.clock(), last + interval)
        self.paced_at[provider] = now

    def key_for(self, path: str) -> str:
        return self.nvidia_key if path == "nvidia" else self.openrouter_key

    def redact(self, text: str) -> str:
        return redact(text, (self.nvidia_key, self.openrouter_key, *self.colab_secrets))

    def bot_settings(self, **overrides: Any) -> Settings:
        """The app's settings with the gate's keys applied."""
        if self.settings is None:
            self.settings = Settings(_env_file=None)
        update: dict[str, Any] = {
            "nvidia_api_key": self.nvidia_key,
            "openrouter_api_key": self.openrouter_key,
        }
        update.update(overrides)
        return self.settings.model_copy(update=update)

    def post(self, url: str, key: str, payload: Mapping, *, timeout: float | None = None) -> dict:
        """Raw request for the probes; polls NVCF 202 answers until done."""
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if url.startswith("https://openrouter.ai/"):
            # Same attribution headers as the bot's OpenRouterProvider.
            headers["HTTP-Referer"] = "https://github.com/linebot-cloud-agent"
            headers["X-OpenRouter-Title"] = "LineBot-CloudAgent"
        limit = timeout or self.timeout
        deadline = time.monotonic() + limit
        response = self.client.post(url, json=payload, headers=headers, timeout=limit)
        if response.status_code == 202:
            response = self._poll(response, headers, deadline)
        if response.status_code != 200:
            raise GateError(f"HTTP {response.status_code}: {short(response.text)}")
        try:
            return response.json()
        except ValueError as exc:
            raise GateError(f"HTTP 200 with a non-JSON body ({len(response.content)} bytes)") from exc

    def _poll(self, response: httpx.Response, headers: dict, deadline: float) -> httpx.Response:
        """NVIDIA answers 202 + NVCF-REQID for slow jobs; poll /v1/status/{id}."""
        request_id = response.headers.get("NVCF-REQID")
        if not request_id:
            raise GateError("HTTP 202 without an NVCF-REQID header")
        url = response.request.url
        status_url = f"{url.scheme}://{url.host}/v1/status/{request_id}"
        while time.monotonic() < deadline:
            self.sleep(1.5)
            self.poll_count += 1
            response = self.client.get(
                status_url, headers=headers, timeout=max(1.0, deadline - time.monotonic())
            )
            if response.status_code != 202:
                return response
        raise GateError(f"HTTP 202: still pending after {self.poll_count} polls")

    def nvidia_chat(self, payload: Mapping, *, timeout: float | None = None) -> dict:
        return self.post(NVIDIA_CHAT_URL, self.nvidia_key, payload, timeout=timeout)

    def openrouter_chat(self, payload: Mapping) -> dict:
        return self.post(OPENROUTER_CHAT_URL, self.openrouter_key, payload)


# ── Bot path: the app's own clients over a recording bridge ──


@dataclass(frozen=True)
class Exchange:
    """One request a bot client sent and what came back."""

    url: str
    payload: dict
    status: int
    data: dict | None
    text: str


_DROPPED_REQUEST_HEADERS = frozenset({"host", "content-length"})
# The body handed back is already decoded, so these would lie about it.
_DROPPED_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


class BridgeTransport(httpx.AsyncBaseTransport):
    """Carries a bot client's requests over the gate's (sync) client.

    The bot's request is sent unchanged, including its own headers, and the
    real answer (any status) goes back to the bot's code. Each exchange is
    kept for evidence. An NVCF 202 goes back too: the providers poll the
    status URL themselves, through this bridge, as the deployed bot does.
    The client is called in a worker thread, so the event loop stays free
    and the bot's ``asyncio.timeout`` limits cancel a request that runs
    past them (recorded as status 0).
    """

    def __init__(self, client: httpx.Client, timeout: float) -> None:
        self._client = client
        self._timeout = timeout
        self.exchanges: list[Exchange] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _DROPPED_REQUEST_HEADERS
        }
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        started = time.monotonic()
        try:
            response = await asyncio.to_thread(
                self._client.request,
                request.method,
                str(request.url),
                content=body,
                headers=headers,
                timeout=self._timeout,
            )
        except asyncio.CancelledError:
            # The bot's time limit ran out; the thread finishes on its own.
            waited = time.monotonic() - started
            self.exchanges.append(
                Exchange(str(request.url), payload, 0, None, f"cancelled by the bot's time limit after {waited:.1f}s")
            )
            raise
        except httpx.HTTPError as exc:
            # Recorded as status 0; the bot's code sees the same exception.
            self.exchanges.append(
                Exchange(str(request.url), payload, 0, None, f"{type(exc).__name__}: {exc}")
            )
            raise
        try:
            data = response.json()
        except ValueError:
            data = None
        self.exchanges.append(
            Exchange(
                url=str(request.url),
                payload=payload,
                status=response.status_code,
                data=data if isinstance(data, dict) else None,
                text=response.text[:300],
            )
        )
        return httpx.Response(
            response.status_code,
            headers=[
                (name, value)
                for name, value in response.headers.items()
                if name.lower() not in _DROPPED_RESPONSE_HEADERS
            ],
            content=response.content,
            request=request,
        )


def _gate_error(exc: Exception) -> GateError:
    """Short evidence for a failure the bot's code raised."""
    if isinstance(exc, AllProvidersFailedError) and exc.last_error is not None:
        exc = exc.last_error
    if isinstance(exc, (RateLimitError, AllModelsRateLimitedError)):
        # The chain's error keeps only the text of the last 429 it met
        # ("All targets exhausted. Last error: ..."), which still names
        # the cap.
        detail = (exc.detail if isinstance(exc, RateLimitError) else str(exc)) or ""
        if "upstream" in detail.lower():
            # OpenRouter passing on one model's upstream limit (the bot cools
            # only that model); the account's own free-model cap reads differently.
            return GateError("HTTP 429 (rate limited upstream; retry later)")
        if OPENROUTER_DAILY_CAP_MARKER in detail.lower():
            # The account's daily cap: run_checks stops calling OpenRouter.
            return GateError(f"HTTP 429 (rate limited: OpenRouter's {OPENROUTER_DAILY_CAP_MARKER} cap)")
        return GateError("HTTP 429 (rate limited)")
    if isinstance(exc, ProviderError):
        if exc.status == 0:
            return GateError(f"request failed: {short(exc.detail) or 'network error or timeout'}")
        return GateError(f"HTTP {exc.status}: {short(exc.detail)}")
    return GateError(f"{type(exc).__name__}: {short(str(exc))}")


def run_bot[T](
    ctx: GateContext,
    work: Callable[[BridgeTransport], Awaitable[T]],
    *,
    timeout: float | None = None,
) -> tuple[T, list[Exchange]]:
    """Run *work* (bot code wired to a bridge) and return its result and exchanges."""
    bridge = BridgeTransport(ctx.client, timeout or ctx.timeout)
    try:
        result = asyncio.run(work(bridge))
    except (ProviderError, AllProvidersFailedError, AllModelsRateLimitedError) as exc:
        raise _gate_error(exc) from exc
    return result, bridge.exchanges


def _raise_for_status(exchanges: list[Exchange]) -> None:
    """GateError when the last exchange was not a 200.

    For bot code that swallows failures (EmbeddingService returns [], the
    orchestrator falls back to chat) the exchange is the only evidence.
    EmbeddingService does not poll an NVCF 202, so one can end up here.
    """
    if not exchanges or exchanges[-1].status == 200:
        return
    last = exchanges[-1]
    if last.status == 0:
        raise GateError(f"request failed: {short(last.text)}")
    if last.status == 202:
        raise GateError(f"HTTP 202 (NVCF job still pending): {short(last.text)}")
    raise GateError(f"HTTP {last.status}: {short(last.text)}")


def _last_data(exchanges: list[Exchange]) -> dict:
    return (exchanges[-1].data or {}) if exchanges else {}


def colab_endpoint(settings: Settings, state: ColabState, transport: httpx.AsyncBaseTransport) -> ColabEndpoint:
    """The bot's endpoint for the READY session *state*, over *transport*.

    The session counts as just checked: ``colab.probe`` is the check that
    tests its health; the others only call it, as the bot would.
    """
    fresh = replace(state, last_health_at=time.time())
    return new_colab_endpoint(settings, InMemoryColabStateStore(fresh), transport=transport)


@contextlib.asynccontextmanager
async def bot_roles(
    ctx: GateContext,
    transport: httpx.AsyncBaseTransport,
    *,
    health: Any = None,
    colab: ColabState | None = None,
    **settings_overrides: Any,
):
    """The app's providers over *transport*, bound into its role table.

    With *colab* (a READY session) the Colab provider is bound too, as
    composition binds it with COLAB_ENABLED.
    """
    settings = ctx.bot_settings(**settings_overrides)
    health = health or build_model_health(settings)
    providers = build_providers(settings, health, transport=transport)
    endpoint: ColabEndpoint | None = None
    try:
        if colab is not None:
            endpoint = colab_endpoint(settings, colab, transport)
            providers[COLAB] = OllamaChatProvider(
                endpoint,
                default_think=settings.colab_chat_think,
                thinking_budget=settings.nvidia_thinking_budget,
            )
        yield role_table(settings, providers)
    finally:
        for provider in providers.values():
            await provider.close()
        if endpoint is not None:
            await endpoint.close()


@dataclass(frozen=True)
class BotCall:
    response: ProviderResponse
    exchanges: list[Exchange]

    @property
    def payload(self) -> dict:
        """The last request body sent (NVCF status polls carry none)."""
        return next((exchange.payload for exchange in reversed(self.exchanges) if exchange.payload), {})

    @property
    def data(self) -> dict:
        return _last_data(self.exchanges)


def bot_chat(
    ctx: GateContext,
    role: str,
    path: str,
    model: str,
    messages: list[dict],
    *,
    disable_thinking: bool,
    timeout: float | None = None,
    tier_limit: bool = False,
    settings_overrides: Mapping[str, Any] | None = None,
    colab: ColabState | None = None,
    **generate_kwargs: Any,
) -> BotCall:
    """One ``provider.generate`` call for the *role* tier, as the bot makes it.

    With *tier_limit* the whole call (an NVCF 202 poll sequence included)
    may take only the tier's ``timeout_s``, as in the bot's fallback chain,
    instead of the gate's timeout: each request gets that limit, the call
    runs under ``asyncio.timeout`` (which cancels a request in flight, or a
    poll sequence between requests), and a call that still finishes past
    the limit fails. *colab* is the
    READY session a Colab tier calls.
    """
    limit: float | None = None
    if tier_limit:
        limit = tier_timeout(ctx.bot_settings(**dict(settings_overrides or {})), role, path, model)
        timeout = limit or timeout

    async def work(bridge: BridgeTransport) -> tuple[ProviderResponse, float]:
        async with bot_roles(ctx, bridge, colab=colab, **dict(settings_overrides or {})) as table:
            tier = role_tier(table, role, path, model)
            call = tier.provider.generate(
                tier.model,
                messages,
                **tier_kwargs(tier, disable_thinking=disable_thinking),
                **generate_kwargs,
            )
            if not limit:
                return await call, 0.0
            # Only the call counts against the limit, as in the bot, where
            # the clients and the table are already built.
            started = ctx.clock()
            async with asyncio.timeout(limit):
                response = await call
            return response, ctx.clock() - started

    try:
        (response, elapsed), exchanges = run_bot(ctx, work, timeout=timeout)
    except TimeoutError:
        raise GateError(f"no answer within the tier's {limit:g}s limit") from None
    if limit and elapsed > limit:
        raise GateError(f"answered in {elapsed:.1f}s, past the tier's {limit:g}s limit")
    return BotCall(response, exchanges)


def bot_payload(
    ctx: GateContext, role: str, path: str, model: str, messages: list[dict], **generate_kwargs: Any
) -> dict:
    """The body the bot's provider builds for this call, captured and not sent."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    async def work() -> None:
        async with bot_roles(ctx, httpx.MockTransport(handler)) as table:
            tier = role_tier(table, role, path, model)
            await tier.provider.generate(
                tier.model,
                messages,
                **tier_kwargs(tier, disable_thinking=True),
                **generate_kwargs,
            )

    asyncio.run(work())
    return captured[-1]


def _chat_messages(settings: Settings, text: str) -> list[dict]:
    """What ChatAgent sends for a plain text message (its system prompt included)."""
    return ChatAgent(settings, None)._build_messages(AgentRequest(text=text))


def _vision_messages(settings: Settings) -> list[dict]:
    """What VisionAgent sends for a question about the red-circle PNG."""
    request = AgentRequest(
        input_type=InputType.IMAGE_TEXT,
        text=VISION_QUESTION,
        image_base64=f"data:image/png;base64,{red_circle_png_base64()}",
    )
    return VisionAgent(settings, None)._build_messages(request)


def _router_messages(settings: Settings) -> list[dict]:
    """What the orchestrator sends to classify ROUTER_USER_MESSAGE."""
    return Orchestrator(settings, None)._build_routing_messages(AgentRequest(text=ROUTER_USER_MESSAGE))


def _traditional_note(text: str) -> str:
    flag = has_traditional_chars(text)
    return "traditional: n/a" if flag is None else f"traditional: {'yes' if flag else 'no'}"


def _answer_and_raw(call: BotCall) -> tuple[str, ChatOutput]:
    return call.response.text or "", read_chat_output(call.data)


# ── Checks: return evidence on PASS, raise GateError on FAIL ─


def _markup_note(answer: str) -> str:
    """Whether the answer carried Markdown/LaTeX that LINE would show raw."""
    return f"markup: {'yes (stripped before LINE)' if to_line_text(answer) != answer else 'no'}"


def check_ultra_plain(ctx: GateContext) -> str:
    settings = ctx.bot_settings()
    call = bot_chat(
        ctx,
        "chat",
        "nvidia",
        ULTRA,
        _chat_messages(settings, "用一句話介紹台北 101。"),
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
        disable_thinking=True,
    )
    answer, raw = _answer_and_raw(call)
    evidence = (
        f"{len(answer)} chars, CJK {cjk_ratio(answer):.0%}, "
        f"{_traditional_note(answer)}, {_markup_note(answer)}, finish={raw.finish_reason}"
    )
    problems = []
    if not answer:
        problems.append("empty answer")
    if raw.reasoning:
        problems.append(f"reasoning present ({len(raw.reasoning)} chars)")
    if raw.think_tags:
        problems.append("<think> in content")
    if problems:
        raise GateError("; ".join(problems) + " | " + evidence)
    return evidence + ", no reasoning"


def _gate_tool_registry():
    """The bot's own tool schemas plus a deterministic clock tool."""

    async def current_time(args: dict, context: ToolContext) -> dict:
        return dict(TOOL_RESULT)

    registry = build_default_registry()
    function = TIME_TOOL["function"]
    registry.register(
        ToolDefinition(
            name=function["name"],
            description=function["description"],
            parameters=function["parameters"],
            executor=current_time,
        )
    )
    return registry


def _tool_round_trip(ctx: GateContext, path: str, model: str) -> str:
    """A real run_tool_loop round trip: tool call, tool result, final answer.

    The bot's recall_memory / update_user_profile / web_search schemas ride
    along, so a schema the endpoint rejects fails the check too.
    """
    settings = ctx.bot_settings()
    registry = _gate_tool_registry()
    messages: list[dict] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": "現在台北幾點？"},
    ]

    async def work(bridge: BridgeTransport):
        health = build_model_health(settings)
        async with bot_roles(ctx, bridge, health=health) as table:
            return await run_tool_loop(
                fallback_chain=FallbackChain(health),
                targets=[role_tier(table, "chat", path, model)],
                messages=messages,
                registry=registry,
                context=ToolContext(),
                tool_choice="auto",
                max_iterations=TOOL_LOOP_ITERATIONS,
                temperature=settings.chat_temperature,
                max_tokens=settings.chat_max_tokens,
                disable_thinking=True,
            )

    # Each request may take the tier's limit, as the chain gives it.
    result, exchanges = run_bot(ctx, work, timeout=tier_timeout(settings, "chat", path, model))
    answer = result.response.text or ""
    executed = result.tool_calls_executed
    if "get_current_time" not in executed:
        raise GateError(
            f"no get_current_time call (executed: {','.join(executed) or 'none'}; "
            f"{len(answer)} chars of text)"
        )
    if not answer:
        raise GateError(f"empty answer after the tool message ({len(exchanges)} requests)")
    uses_result = "15" in answer
    return (
        f"tool_calls={len(executed)} ({','.join(executed)}); final {len(answer)} chars; "
        f"uses tool result: {'yes' if uses_result else 'no'}; "
        f"{len(registry.names())} tool schemas accepted"
    )


def check_ultra_tools(ctx: GateContext) -> str:
    return _tool_round_trip(ctx, "nvidia", ULTRA)


def _router_check(ctx: GateContext, path: str, model: str) -> str:
    """The bot's router call: Orchestrator._llm_classify with thinking off."""
    settings = ctx.bot_settings()
    request = AgentRequest(text=ROUTER_USER_MESSAGE)

    async def work(bridge: BridgeTransport):
        health = build_model_health(settings)
        async with bot_roles(ctx, bridge, health=health) as table:
            tier = role_tier(table, "router", path, model)
            orchestrator = Orchestrator(settings, FallbackChain(health), targets=[tier])
            return await orchestrator._llm_classify(request)

    # Each request may take the tier's limit, as the chain gives it.
    decision, exchanges = run_bot(ctx, work, timeout=tier_timeout(settings, "router", path, model))
    if not exchanges:
        raise GateError("no request was sent")
    if decision.reasoning in _ROUTER_FALLBACK_REASONS:
        _raise_for_status(exchanges)
        answer = read_chat_output(_last_data(exchanges)).answer
        raise GateError(
            f"{decision.reasoning} after {len(exchanges)} request(s) "
            f"(last answer {len(answer)} chars)"
        )
    # The answer the decision was read from (the repair prompt's, if sent).
    _hold_to_schema(read_chat_output(_last_data(exchanges)).answer, decision.agent)
    attempts = (
        "parsed first try"
        if len(exchanges) == 1
        else f"recovered by the repair prompt ({len(exchanges)} requests)"
    )
    return (
        f"agent={decision.agent}, output={decision.output_format}, "
        f"thinking={'off' if decision.disable_thinking else 'on'}; {attempts}"
    )


def _hold_to_schema(answer: str, routed_to: str) -> None:
    """GateError when the router JSON in *answer* breaks ROUTER_JSON_SCHEMA.

    The bot still routes such a reply (to *routed_to*), filling the gaps
    with chat + text, so only this check notices a tier that ignores the
    schema (Ultra has no structured output to hold it).
    """
    try:
        bot_route(answer)
    except GateError as exc:
        raise GateError(f"route JSON breaks the schema: {exc} (the bot would send it to {routed_to})") from None


def check_super_router(ctx: GateContext) -> str:
    return _router_check(ctx, "nvidia", SUPER)


def check_ultra_router(ctx: GateContext) -> str:
    """Ultra has no structured output: the prompt asks for the JSON, the
    orchestrator's tolerant parser (and its one repair prompt) reads it."""
    return _router_check(ctx, "nvidia", ULTRA)


def _router_probe_payload(ctx: GateContext, path: str, model: str) -> dict:
    """The bot's router body without its JSON mode, for the probes to add theirs."""
    settings = ctx.bot_settings()
    payload = bot_payload(
        ctx,
        "router",
        path,
        model,
        _router_messages(settings),
        temperature=settings.orchestrator_temperature,
        max_tokens=settings.orchestrator_max_tokens,
    )
    payload.pop("response_format", None)
    return payload


def check_super_router_json(ctx: GateContext) -> str:
    """Probe: which constrained-JSON shapes Super accepts (the bot sends json_schema)."""
    base = _router_probe_payload(ctx, "nvidia", SUPER)
    variants = {
        "guided_json": {"nvext": {"guided_json": ROUTER_JSON_SCHEMA}},
        "json_schema": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "route", "strict": True, "schema": ROUTER_JSON_SCHEMA},
            }
        },
    }
    outcomes: list[str] = []
    working = 0
    for name, extra in variants.items():
        started = time.monotonic()
        try:
            data = ctx.nvidia_chat({**base, **extra})
            route = parse_route(read_chat_output(data).answer)
        except (GateError, httpx.HTTPError) as exc:
            outcomes.append(f"{name}: FAIL {short(str(exc) or type(exc).__name__, 60)}")
            continue
        working += 1
        outcomes.append(f"{name}: ok agent={route['agent']} {time.monotonic() - started:.1f}s")
    evidence = "; ".join(outcomes)
    if not working:
        raise GateError(evidence)
    return evidence


def check_ultra_thinking(ctx: GateContext) -> str:
    """The thinking role's Ultra tier: thinking on at medium_effort, no reasoning_budget."""
    settings = ctx.bot_settings()
    outputs: list[tuple[BotCall, str | None]] = []
    for prompt, expected in THINKING_PROMPTS:
        call = bot_chat(
            ctx,
            "thinking",
            "nvidia",
            ULTRA,
            _chat_messages(settings, prompt),
            timeout=max(ctx.timeout, SLOW_TIMEOUT_SECONDS),
            # NVIDIA_THINKING_BUDGET only widens max_tokens on Ultra.
            settings_overrides={"nvidia_thinking_budget": THINKING_BUDGET},
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_tokens,
            disable_thinking=False,
        )
        outputs.append((call, expected))
    sent = outputs[0][0].payload
    if "reasoning_budget" in sent or "reasoning_budget" in (sent.get("chat_template_kwargs") or {}):
        raise GateError("the bot sent a reasoning_budget, which Ultra on NIM rejects (HTTP 400)")
    if sent.get("chat_template_kwargs") != ULTRA_THINKING_TEMPLATE_KWARGS:
        raise GateError(
            "the bot sent chat_template_kwargs="
            f"{json.dumps(sent.get('chat_template_kwargs'), sort_keys=True)}, "
            f"expected {json.dumps(ULTRA_THINKING_TEMPLATE_KWARGS, sort_keys=True)}"
        )

    served = sorted({call.payload.get("model", "?") for call, _ in outputs})
    answers = [call.response.text or "" for call, _ in outputs]
    answered = sum(1 for answer in answers if answer)
    reasoned = sum(
        1 for call, _ in outputs if call.response.reasoning_content or call.response.reasoning_tokens
    )
    flags = [has_traditional_chars(answer) for answer in answers]
    traditional = (
        "n/a" if any(flag is None for flag in flags) else f"{sum(bool(flag) for flag in flags)}/{len(flags)}"
    )
    expected_total = sum(1 for _, expected in outputs if expected)
    expected_hits = sum(
        1 for (_, expected), answer in zip(outputs, answers, strict=True) if expected and expected in answer
    )
    evidence = (
        f"served by {','.join(served)}; answers {answered}/{len(outputs)}; "
        f"with reasoning {reasoned}/{len(outputs)}; "
        f"CJK {'/'.join(f'{cjk_ratio(answer):.0%}' for answer in answers)}; "
        f"t2s changes (traditional) {traditional}; expected {expected_hits}/{expected_total}; "
        f"reasoning chars {'/'.join(str(len(call.response.reasoning_content or '')) for call, _ in outputs)}; "
        f"chat_template_kwargs={json.dumps(sent.get('chat_template_kwargs'), sort_keys=True)}"
    )
    if answered < len(outputs) or reasoned < len(outputs) or served != [ULTRA]:
        raise GateError(evidence)
    return evidence


def _vision_evidence(call: BotCall) -> str:
    answer, raw = _answer_and_raw(call)
    mentions_red = bool(re.search(r"紅|red", answer, re.IGNORECASE))
    evidence = (
        f"mentions red: {'yes' if mentions_red else 'no'}; {len(answer)} chars; "
        f"channel markers: {'stripped' if raw.channel_markers else 'none'}"
    )
    if not mentions_red:
        raise GateError(evidence)
    return evidence


def _nim_vision_check(ctx: GateContext, model: str) -> str:
    """The vision role's NIM tier on the red-circle PNG, within its time limit."""
    settings = ctx.bot_settings()
    call = bot_chat(
        ctx,
        "vision",
        "nvidia",
        model,
        _vision_messages(settings),
        temperature=settings.vision_temperature,
        max_tokens=settings.vision_max_tokens,
        disable_thinking=True,
        tier_limit=True,
    )
    return _vision_evidence(call)


def check_muse_vision(ctx: GateContext) -> str:
    return _nim_vision_check(ctx, MUSE)


def check_omni_vision(ctx: GateContext) -> str:
    return _nim_vision_check(ctx, OMNI)


def _embed_with(settings: Settings, tier: Tier, transport: httpx.AsyncBaseTransport):
    """EmbeddingService for the embed *tier*, as the app builds it."""
    return build_embedding_service(settings, tier=tier, transport=transport)


def check_embed(ctx: GateContext) -> str:
    """EmbeddingService as the app builds it: 2048 from the API, 1024 kept."""
    settings = ctx.bot_settings()

    async def work(bridge: BridgeTransport) -> list[float] | None:
        async with bot_roles(ctx, bridge) as table:
            service = _embed_with(settings, role_tier(table, "embed", "nvidia", EMBED), bridge)
            try:
                return await service.embed_text("台北今天會下雨嗎？")
            finally:
                await service.close()

    vector, exchanges = run_bot(ctx, work)
    _raise_for_status(exchanges)
    items = sorted(_last_data(exchanges).get("data") or [], key=lambda item: item.get("index", 0))
    api_dim = len(items[0].get("embedding") or []) if items else 0
    if not vector:
        raise GateError(f"no vector from the app (api dim={api_dim})")
    norm = math.sqrt(sum(value * value for value in vector))
    evidence = f"api dim={api_dim}; app vector dim={len(vector)}, norm={norm:.2f}"
    if api_dim != EMBED_DIMENSIONS:
        raise GateError(f"{evidence} (expected {EMBED_DIMENSIONS})")
    if len(vector) != settings.embedding_dimensions or abs(norm - 1.0) > 1e-3:
        raise GateError(f"{evidence} (expected {settings.embedding_dimensions}, norm 1)")
    return evidence


def _flux_check(ctx: GateContext, model: str) -> str:
    """NvidiaProvider.generate_image with the image tier ImageGenAgent uses."""

    async def work(bridge: BridgeTransport) -> ProviderResponse:
        async with bot_roles(ctx, bridge) as table:
            tier = role_tier(table, "image", "nvidia", model)
            return await tier.provider.generate_image(
                model=tier.model, prompt=FLUX_PROMPT, options=tier.options
            )

    response, exchanges = run_bot(ctx, work, timeout=max(ctx.timeout, SLOW_TIMEOUT_SECONDS))
    data = _last_data(exchanges)
    artifact = (data.get("artifacts") or [{}])[0] or {}
    finish = artifact.get("finishReason") or data.get("finishReason") or "?"
    if not response.images:
        note = f"; app text: {short(response.text, 40)}" if response.text else ""
        raise GateError(f"no image (finishReason={finish}){note}")
    data_url = response.images[0]
    mime = data_url[len("data:") : data_url.index(";")] if data_url.startswith("data:") else "?"
    encoded = data_url.split(",", 1)[-1]
    evidence = f"{mime}, {len(base64.b64decode(encoded)) // 1024} KB, finishReason={finish}"
    if finish != "SUCCESS":
        raise GateError(evidence)
    return evidence


def check_flux_klein(ctx: GateContext) -> str:
    return _flux_check(ctx, FLUX_KLEIN)


def check_flux_dev(ctx: GateContext) -> str:
    return _flux_check(ctx, FLUX_DEV)


def _openrouter_no_reasoning(ctx: GateContext, model: str) -> str:
    """A chat-role OpenRouter tier with effort none: an answer and no reasoning."""
    settings = ctx.bot_settings()
    call = bot_chat(
        ctx,
        "chat",
        "openrouter",
        model,
        _chat_messages(settings, "用一句話說明什麼是颱風。"),
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
        disable_thinking=True,
    )
    answer, raw = _answer_and_raw(call)
    tokens = reasoning_tokens(call.data)
    evidence = (
        f"reasoning_tokens={'absent' if tokens is None else tokens}; "
        f"reasoning text {len(raw.reasoning)} chars; answer {len(answer)} chars"
    )
    if answer and not tokens and not raw.reasoning:
        return evidence
    # Report whether the documented alternative switch would work instead.
    try:
        alternative = reasoning_tokens(
            ctx.openrouter_chat({**call.payload, "reasoning": {"enabled": False}})
        )
        evidence += f" | enabled:false -> reasoning_tokens={'absent' if alternative is None else alternative}"
    except (GateError, httpx.HTTPError) as exc:
        evidence += f" | enabled:false -> {short(str(exc) or type(exc).__name__, 60)}"
    raise GateError(evidence)


def check_openrouter_qwen_no_reasoning(ctx: GateContext) -> str:
    return _openrouter_no_reasoning(ctx, OR_QWEN)


def check_openrouter_ultra_no_reasoning(ctx: GateContext) -> str:
    return _openrouter_no_reasoning(ctx, OR_ULTRA)


def _openrouter_thinking(ctx: GateContext, model: str) -> str:
    """The bot's thinking request on a thinking-role OpenRouter tier, with its effort."""
    settings = ctx.bot_settings()
    prompt, expected = THINKING_PROMPTS[0]
    call = bot_chat(
        ctx,
        "thinking",
        "openrouter",
        model,
        _chat_messages(settings, prompt),
        timeout=max(ctx.timeout, SLOW_TIMEOUT_SECONDS),
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
        disable_thinking=False,
    )
    answer = call.response.text or ""
    sent = call.payload.get("reasoning") or {}
    tokens = reasoning_tokens(call.data)
    reasoning_chars = len(call.response.reasoning_content or "")
    evidence = (
        f"effort={sent.get('effort', '?')}; reasoning_tokens={'absent' if tokens is None else tokens}; "
        f"reasoning text {reasoning_chars} chars; answer {len(answer)} chars; "
        f"expected answer: {'yes' if expected and expected in answer else 'no'}"
    )
    if not answer:
        raise GateError(f"empty answer | {evidence}")
    if not tokens and not reasoning_chars:
        raise GateError(f"no reasoning despite thinking on | {evidence}")
    return evidence


def check_openrouter_qwen_thinking(ctx: GateContext) -> str:
    """The bot's thinking request when NVIDIA is down: qwen3.8 with its effort."""
    return _openrouter_thinking(ctx, OR_QWEN)


def check_openrouter_ultra_thinking(ctx: GateContext) -> str:
    return _openrouter_thinking(ctx, OR_ULTRA)


def check_openrouter_qwen_tools(ctx: GateContext) -> str:
    return _tool_round_trip(ctx, "openrouter", OR_QWEN)


def check_openrouter_qwen_vision(ctx: GateContext) -> str:
    """The vision role's OpenRouter tier (effort none) on the red-circle PNG."""
    settings = ctx.bot_settings()
    call = bot_chat(
        ctx,
        "vision",
        "openrouter",
        OR_QWEN,
        _vision_messages(settings),
        temperature=settings.vision_temperature,
        max_tokens=settings.vision_max_tokens,
        disable_thinking=True,
    )
    return _vision_evidence(call)


def check_openrouter_super_router(ctx: GateContext) -> str:
    return _router_check(ctx, "openrouter", OR_SUPER)


# ── Role checks: each role's first enabled non-Colab tier answers ──

_ROLE_TEXT = "用一句話介紹台北 101。"


async def _role_answer(settings: Settings, role: str, tier: Tier, bridge: BridgeTransport) -> str:
    """Evidence that *tier* answered a request of *role*; GateError when not."""
    if role == "embed":
        service = _embed_with(settings, tier, bridge)
        try:
            vector = await service.embed_text("台北今天會下雨嗎？")
        finally:
            await service.close()
        if not vector:
            # EmbeddingService swallows failures: the exchange is the evidence.
            _raise_for_status(bridge.exchanges)
            raise GateError("no vector")
        return f"vector dim={len(vector)}"
    if role == "image":
        response = await tier.provider.generate_image(
            model=tier.model, prompt=FLUX_PROMPT, options=tier.options
        )
        if not response.images:
            raise GateError(f"no image{f' ({short(response.text, 40)})' if response.text else ''}")
        return "image returned"

    thinking = role == "thinking"
    extra: dict[str, Any] = {}
    if role == "router":
        messages = _router_messages(settings)
        extra = {
            "json_schema": ROUTER_JSON_SCHEMA,
            "temperature": settings.orchestrator_temperature,
            "max_tokens": settings.orchestrator_max_tokens,
        }
    elif role == "vision":
        messages = _vision_messages(settings)
    else:
        messages = _chat_messages(settings, THINKING_PROMPTS[0][0] if thinking else _ROLE_TEXT)
    response = await tier.provider.generate(
        tier.model, messages, **tier_kwargs(tier, disable_thinking=not thinking), **extra
    )
    answer = response.text or ""
    if role == "router":
        route = bot_route(answer)
        return f"agent={route['agent']}"
    return f"{len(answer)} chars"


def _role_provider(role: str) -> Callable[[GateContext], str]:
    """The API the *role* check will call: its first enabled non-Colab tier's, "" if none."""

    def resolve(ctx: GateContext) -> str:
        # The table binds a tier only where a key builds the client
        # (composition.build_providers); the stand-ins are never called.
        keyed = {path: path for path in KEY_NAMES if ctx.key_for(path)}
        try:
            return first_api_tier(role_table(ctx.bot_settings(), keyed), role).spec.provider
        except GateError:
            return ""  # the check itself reports why

    return resolve


def _role_check(role: str) -> Callable[[GateContext], str]:
    def run(ctx: GateContext) -> str:
        settings = ctx.bot_settings()

        async def work(bridge: BridgeTransport) -> str:
            async with bot_roles(ctx, bridge) as table:
                row = first_api_tier(table, role)
                assert row.tier is not None
                return f"{row.label}: {await _role_answer(settings, role, row.tier, bridge)}"

        slow = role in ("thinking", "image")
        evidence, _ = run_bot(ctx, work, timeout=max(ctx.timeout, SLOW_TIMEOUT_SECONDS) if slow else None)
        return evidence

    return run


# ── Colab checks: a READY session, called as the bot calls it ──


def load_colab_state(settings: Settings) -> ColabState:
    """The session document (Firestore ``{prefix}_colab/state``), read with this machine's ADC."""
    return asyncio.run(build_state_store(settings).load())


def colab_session(ctx: GateContext) -> ColabState:
    """The READY session the Colab checks call; GateSkip when there is none.

    Resolved once per run; the shared secret, the session's bearer and its
    tunnel host join the values every evidence line is redacted of.
    """
    if ctx.colab_state is None:
        try:
            ctx.colab_state = _resolve_colab_session(ctx)
        except (GateSkip, GateError) as exc:
            ctx.colab_state = exc
    if isinstance(ctx.colab_state, (GateSkip, GateError)):
        raise type(ctx.colab_state)(str(ctx.colab_state))
    return ctx.colab_state


def _resolve_colab_session(ctx: GateContext) -> ColabState:
    settings = ctx.bot_settings()
    if not settings.colab_enabled:
        raise GateSkip("COLAB_ENABLED is off")
    if not settings.firestore_enabled:
        raise GateSkip("COLAB_ENABLED needs FIRESTORE_ENABLED=true (the session state lives in Firestore)")
    secret = settings.colab_shared_secret
    if len(secret) < MIN_COLAB_SECRET_LENGTH:
        raise GateSkip(f"COLAB_SHARED_SECRET is under {MIN_COLAB_SECRET_LENGTH} characters, so the tier is off")
    ctx.colab_secrets.append(secret)
    try:
        state = (ctx.colab_loader or load_colab_state)(settings)
    except Exception as exc:  # no ADC, no access, no google-cloud-firestore
        raise GateSkip(
            f"the session state is unreadable ({type(exc).__name__}: {short(str(exc), 80)}); run "
            "gcloud auth application-default login as an account that reads the project's Firestore"
        ) from None
    if state.status != READY or not state.tunnel_url or not state.nbh:
        error = f"; last error: {short(state.last_error, 60)}" if state.last_error else ""
        raise GateSkip(
            f"no READY Colab session (status {state.status}{error}); send !colab on and "
            "rerun once !colab status says READY"
        )
    try:
        host = protocol.check_tunnel_url(state.tunnel_url, settings.colab_tunnel_host)
    except protocol.TunnelURLError as exc:
        raise GateError(f"the session's tunnel_url breaks the tunnel rule ({exc}); nothing was sent") from None
    ctx.colab_secrets.extend([protocol.derive_proxy_token(secret, state.nbh), host])
    return state


def _inference_session(ctx: GateContext) -> ColabState:
    """:func:`colab_session`, for the checks that need real models."""
    state = colab_session(ctx)
    if ctx.bot_settings().colab_dry_run:
        raise GateSkip("COLAB_DRY_RUN=true: the VM runs a stub Ollama that answers no inference; colab.probe covers it")
    return state


def check_colab_probe(ctx: GateContext) -> str:
    """The bot's readiness probe through the tunnel (endpoint.probe)."""
    state = colab_session(ctx)
    settings = ctx.bot_settings()

    async def work(bridge: BridgeTransport) -> list[str]:
        endpoint = colab_endpoint(settings, state, bridge)
        try:
            return await endpoint.probe(state.tunnel_url, state.nbh)
        finally:
            await endpoint.close()

    problems, exchanges = run_bot(ctx, work)
    if problems:
        raise GateError("; ".join(problems))
    version = next(
        (str((exchange.data or {}).get("version")) for exchange in exchanges if exchange.url.endswith("/api/version")),
        "?",
    )
    running = next((exchange.data for exchange in exchanges if exchange.url.endswith("/api/ps")), None) or {}
    pins = protocol.pinned_models()
    in_vram = ", ".join(
        f"{entry['name']} {protocol.mib(protocol.model_size(entry.get('size_vram')))} MiB"
        for entry in running.get("models") or []
        if isinstance(entry, dict) and entry.get("name") in pins
    )
    think = "/".join(str(value).lower() for value in protocol.CHAT_THINK_VALUES)
    return (
        f"Ollama {version}; {', '.join(pins)} loaded wholly in VRAM ({in_vram}) with the pinned "
        f"digests; think values {think}"
    )


def _colab_answer_evidence(call: BotCall, think: str) -> str:
    answer = call.response.text or ""
    sent = call.payload.get("think")
    expected: bool | str = False if think == "false" else think
    thinking = call.response.reasoning_content or ""
    evidence = (
        f"{len(answer)} chars, CJK {cjk_ratio(answer):.0%}, {_traditional_note(answer)}, "
        f"think={sent}, thinking {len(thinking)} chars, done={call.data.get('done_reason')}, "
        f"{_markup_note(answer)}"
    )
    problems = []
    if not answer:
        problems.append("empty answer")
    if sent != expected:
        problems.append(f"sent think={sent!r}, not COLAB_CHAT_THINK={think}")
    if problems:
        raise GateError("; ".join(problems) + " | " + evidence)
    return evidence


def check_colab_chat(ctx: GateContext) -> str:
    """The chat role's Colab tier: a zh-TW reply at COLAB_CHAT_THINK, within its limit."""
    state = _inference_session(ctx)
    settings = ctx.bot_settings()
    call = bot_chat(
        ctx,
        "chat",
        COLAB,
        COLAB_QWEN,
        _chat_messages(settings, _ROLE_TEXT),
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
        disable_thinking=True,
        tier_limit=True,
        colab=state,
    )
    return _colab_answer_evidence(call, settings.colab_chat_think)


def check_colab_vision(ctx: GateContext) -> str:
    """The vision role's Colab tier on the red-circle PNG, within its limit."""
    state = _inference_session(ctx)
    settings = ctx.bot_settings()
    call = bot_chat(
        ctx,
        "vision",
        COLAB,
        COLAB_QWEN,
        _vision_messages(settings),
        temperature=settings.vision_temperature,
        max_tokens=settings.vision_max_tokens,
        disable_thinking=True,
        tier_limit=True,
        colab=state,
    )
    images = sum(len(message.get("images") or ()) for message in call.payload.get("messages") or [])
    return f"{_vision_evidence(call)}; {images} image(s) as raw base64, think={call.payload.get('think')}"


def _scored(answers: Mapping, name: str, allowed: set[str]) -> tuple[str, float]:
    answer = answers.get(name)
    choice = answer.get("choice") if isinstance(answer, dict) else None
    confidence = answer.get("confidence") if isinstance(answer, dict) else None
    if choice not in allowed or not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise GateError(f"malformed {name} answer: {short(json.dumps(answer, ensure_ascii=False), 80)}")
    return choice, float(confidence)


def check_colab_router(ctx: GateContext) -> str:
    """nimble on /v1/systemone through the bot's SystemOneRouter, within its limit."""
    return _colab_router_check(ctx, COLAB_NIMBLE)


def _colab_router_check(ctx: GateContext, model: str) -> str:
    """The router role's Colab tier *model* (a systemone one) decides ROUTER_USER_MESSAGE."""
    state = _inference_session(ctx)
    settings = ctx.bot_settings()
    limit = tier_timeout(settings, "router", COLAB, model) or ctx.timeout
    request = AgentRequest(text=ROUTER_USER_MESSAGE)
    gates = RouterGates.from_settings(settings)

    async def work(bridge: BridgeTransport):
        health = build_model_health(settings)
        async with bot_roles(ctx, bridge, health=health, colab=state) as table:
            tier = role_tier(table, "router", COLAB, model)
            router = SystemOneRouter(tier.provider.endpoint, tier, gates=gates, health=health)
            # Only the routing call counts against the limit, as in the
            # bot, where the clients and the table are already built.
            started = ctx.clock()
            decision = await router.route(request)
            return decision, ctx.clock() - started

    (decision, elapsed), exchanges = run_bot(ctx, work, timeout=limit)
    if not exchanges:
        raise GateError(f"no request was sent ({model} was skipped)")
    _raise_for_status(exchanges)
    if elapsed > limit:
        raise GateError(
            f"answered in {elapsed:.1f}s, past {model}'s {limit:g}s limit (the API router would decide)"
        )
    data = _last_data(exchanges)
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise GateError("no answers object")
    agent, agent_confidence = _scored(answers, "agent", VALID_AGENTS)
    output, output_confidence = _scored(answers, "output_format", VALID_OUTPUTS)
    noul = (answers.get("needs_thinking") or {}).get("noul")
    if not isinstance(noul, (int, float)) or isinstance(noul, bool):
        raise GateError("the needs_thinking answer has no noul")
    verdict = (
        f"decides {decision.agent}"
        if decision is not None
        else f"below the {gates.confidence:g} gate, the API router decides"
    )
    expected = "" if agent == "web_search" else " (expected web_search)"
    tokens = (data.get("usage") or {}).get("input_tokens", "?")
    return (
        f"agent={agent}{expected} ({agent_confidence:.2f}), format={output} ({output_confidence:.2f}), "
        f"thinking p={noul:.2f}; {verdict}; {tokens} input tokens, {elapsed:.1f}s"
    )


@dataclass(frozen=True)
class Check:
    check_id: str
    path: str
    title: str
    run: Callable[[GateContext], str]
    # The API a tier check sends its one request to: run_checks spaces such
    # checks within the provider's free per-minute limit, and skips them
    # once OpenRouter's daily cap is met.
    provider: str = ""
    # For a check whose API depends on the keys and settings (a role
    # check): resolves *provider* when the check is about to run.
    provider_for: Callable[[GateContext], str] | None = None

    def provider_in(self, ctx: GateContext) -> str:
        if self.provider or self.provider_for is None:
            return self.provider
        return self.provider_for(ctx)


CHECKS: tuple[Check, ...] = (
    Check("nvidia.ultra.plain", "nvidia", "Ultra chat reply (bot payload, thinking off)", check_ultra_plain),
    Check("nvidia.ultra.tools", "nvidia", "Ultra run_tool_loop round-trip", check_ultra_tools),
    Check(
        "nvidia.super.router",
        "nvidia",
        "Super router via Orchestrator (response_format json_schema)",
        check_super_router,
    ),
    Check(
        "nvidia.super.router_json",
        "nvidia",
        "Super router JSON probe: guided_json and json_schema",
        check_super_router_json,
    ),
    Check(
        "nvidia.ultra.thinking",
        "nvidia",
        "Ultra with thinking on, zh-TW x3 (medium_effort, no reasoning_budget)",
        check_ultra_thinking,
    ),
    Check(
        "nvidia.ultra.router",
        "nvidia",
        "Ultra router via Orchestrator (JSON asked for in the prompt; no structured output)",
        check_ultra_router,
    ),
    Check(
        "nvidia.omni.vision",
        "nvidia",
        "Nemotron 3 Nano Omni vision, thinking off (red circle PNG, 15 s)",
        check_omni_vision,
    ),
    Check("nvidia.muse.vision", "nvidia", "muse-glimmer 30B vision (red circle PNG, 35 s)", check_muse_vision),
    Check("nvidia.embed.dim", "nvidia", "EmbeddingService: 2048-d from the API, 1024 kept", check_embed),
    Check("nvidia.flux2_klein.image", "nvidia", "FLUX.2 klein 4B returns an image", check_flux_klein),
    Check("nvidia.flux1_dev.image", "nvidia", "FLUX.1-dev returns an image", check_flux_dev),
    Check(
        "openrouter.qwen.no_reasoning",
        "openrouter",
        "qwen3.8-27b:free chat with reasoning effort none",
        check_openrouter_qwen_no_reasoning,
    ),
    Check(
        "openrouter.qwen.thinking",
        "openrouter",
        "qwen3.8-27b:free thinking request (the thinking role's effort)",
        check_openrouter_qwen_thinking,
    ),
    Check(
        "openrouter.qwen.tools",
        "openrouter",
        "qwen3.8-27b:free run_tool_loop round-trip",
        check_openrouter_qwen_tools,
    ),
    Check(
        "openrouter.ultra.no_reasoning",
        "openrouter",
        "nemotron-3-ultra:free chat with reasoning effort none",
        check_openrouter_ultra_no_reasoning,
    ),
    Check(
        "openrouter.ultra.thinking",
        "openrouter",
        "nemotron-3-ultra:free thinking request (effort medium)",
        check_openrouter_ultra_thinking,
    ),
    Check(
        "openrouter.qwen.vision",
        "openrouter",
        "qwen3.8-27b:free vision with reasoning effort none (red circle PNG)",
        check_openrouter_qwen_vision,
    ),
    Check(
        "openrouter.super.router",
        "openrouter",
        "nemotron-3-super:free router via Orchestrator (json_object, effort none)",
        check_openrouter_super_router,
    ),
    Check(
        "colab.probe",
        "colab",
        "Colab readiness probe: Ollama 0.35.0, both models loaded wholly in VRAM, pinned digests, think values",
        check_colab_probe,
    ),
    Check("colab.qwen.chat", "colab", "qwen3.8 zh-TW chat reply at COLAB_CHAT_THINK (tier limit)", check_colab_chat),
    Check("colab.qwen.vision", "colab", "qwen3.8 vision (red circle PNG, tier limit)", check_colab_vision),
    Check(
        "colab.nimble.router",
        "colab",
        "nimble routing decision via SystemOneRouter (/v1/systemone)",
        check_colab_router,
    ),
    *(
        Check(
            f"roles.{role}",
            "roles",
            f"{role}: the first enabled non-Colab tier answers",
            _role_check(role),
            provider_for=_role_provider(role),
        )
        for role in ROLES
    ),
)


# ── Tier checks: one minimal call per tier of the configured role table ──


def _tier_spec(settings: Settings, role: str, index: int, label: str) -> TierSpec:
    """Tier *index* of *role* in the configured table; GateError unless it is *label*."""
    try:
        specs = role_specs(settings)[role]
    except RoleConfigError as exc:
        raise GateError(f"role table: {short(str(exc), 150)}") from None
    if index >= len(specs) or specs[index].label != label:
        raise GateError(f"the {role} role no longer has {label} as tier {index + 1}")
    return specs[index]


def _zh_answer(answer: str, what: str) -> str:
    """Evidence for a zh-TW *answer*; GateError when it is empty or not Chinese."""
    if not answer:
        raise GateError(f"empty {what}")
    ratio = cjk_ratio(answer)
    evidence = f"{len(answer)} chars, CJK {ratio:.0%}, {_traditional_note(answer)}"
    if ratio < ZH_MIN_CJK:
        raise GateError(f"{what} is not zh-TW ({evidence}): {short(answer, 40)}")
    return evidence


def _utility_messages() -> list[dict]:
    """What MemoryService._summarize_memory sends for a first summary of UTILITY_DIALOGUE."""
    return summary_messages("", UTILITY_DIALOGUE)


def _image_bytes(data_url: str) -> bytes:
    try:
        return base64.b64decode(data_url.split(",", 1)[-1], validate=False)
    except (binascii.Error, ValueError):
        return b""


async def _tier_answer(settings: Settings, role: str, tier: Tier, bridge: BridgeTransport) -> str:
    """One minimal request of *role* to *tier*, as the bot sends it; evidence or GateError."""
    if role == "embed":
        service = _embed_with(settings, tier, bridge)
        try:
            vector = await service.embed_text(EMBED_TEXT)
        finally:
            await service.close()
        if not vector:
            # EmbeddingService swallows failures: the exchange is the evidence.
            _raise_for_status(bridge.exchanges)
            items = _last_data(bridge.exchanges).get("data") or [{}]
            api_dim = len((items[0] or {}).get("embedding") or [])
            raise GateError(
                f"no vector from the app (api dim={api_dim}, EMBEDDING_DIMENSIONS={settings.embedding_dimensions})"
            )
        if len(vector) != settings.embedding_dimensions:
            raise GateError(f"dim={len(vector)}, not EMBEDDING_DIMENSIONS={settings.embedding_dimensions}")
        return f"dim={len(vector)} = EMBEDDING_DIMENSIONS"
    if role == "image":
        response = await tier.provider.generate_image(
            model=tier.model, prompt=FLUX_PROMPT, options=tier.options
        )
        image = _image_bytes(response.images[0]) if response.images else b""
        if not image:
            note = f" ({short(response.text, 40)})" if response.text else ""
            raise GateError(f"no image bytes{note}")
        return f"{sniff_image_mime(image)}, {len(image) // 1024} KB"

    thinking = role == "thinking"
    kwargs = tier_kwargs(tier, disable_thinking=not thinking)
    if role == "router":
        response = await tier.provider.generate(
            tier.model,
            _router_messages(settings),
            json_schema=ROUTER_JSON_SCHEMA,
            temperature=settings.orchestrator_temperature,
            max_tokens=settings.orchestrator_max_tokens,
            **kwargs,
        )
        text = response.text or ""
        # The orchestrator's own (tolerant) parser decides, as in the bot.
        decision = Orchestrator(settings, None)._try_parse_llm_response(text, ROUTER_USER_MESSAGE)
        if decision is None:
            raise GateError(f"no routing decision in the reply ({len(text)} chars): {short(text, 40)}")
        _hold_to_schema(text, decision.agent)
        return (
            f"agent={decision.agent}, output={decision.output_format}, "
            f"thinking={'off' if decision.disable_thinking else 'on'}"
        )
    if role == "vision":
        response = await tier.provider.generate(
            tier.model,
            _vision_messages(settings),
            temperature=settings.vision_temperature,
            max_tokens=settings.vision_max_tokens,
            **kwargs,
        )
        answer = response.text or ""
        red, round_ = bool(_RED_RE.search(answer)), bool(_ROUND_RE.search(answer))
        if not (red and round_):
            raise GateError(
                f"red: {'yes' if red else 'no'}, circle: {'yes' if round_ else 'no'}: {short(answer, 40)}"
            )
        return f"red circle: {short(answer, 50)}"
    if role == "utility":
        response = await tier.provider.generate(
            tier.model,
            _utility_messages(),
            temperature=settings.memory_summary_temperature,
            max_tokens=settings.memory_summary_max_tokens,
            **kwargs,
        )
        answer = (response.text or "").strip()
        if response.finish_reason == "length":
            # MemoryService rejects a summary cut off at max_tokens.
            raise GateError(f"summary cut off at max_tokens ({len(answer)} chars)")
        return _zh_answer(answer, "summary")

    prompt, expected = THINKING_PROMPTS[0] if thinking else (_ROLE_TEXT, None)
    response = await tier.provider.generate(
        tier.model,
        _chat_messages(settings, prompt),
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
        **kwargs,
    )
    evidence = _zh_answer(response.text or "", "answer")
    if not thinking:
        return evidence
    reasoning = len(response.reasoning_content or "")
    tokens = response.reasoning_tokens
    if not kwargs["disable_thinking"] and not (reasoning or tokens):
        raise GateError(f"no reasoning despite thinking on ({evidence})")
    found = "yes" if expected and expected in (response.text or "") else "no"
    return f"reasoning {reasoning} chars/{tokens} tokens, expected answer: {found}; {evidence}"


def _tier_check(role: str, index: int, label: str) -> Callable[[GateContext], str]:
    """Tier *index* (*label*) of *role*: one request through the bot's own provider."""

    def run(ctx: GateContext) -> str:
        settings = ctx.bot_settings()
        spec = _tier_spec(settings, role, index, label)
        profile = get_profile(spec.provider, spec.model)
        if profile is not None and profile.protocol == SYSTEMONE:
            # nimble answers on /v1/systemone, not in JSON: the bot asks
            # SystemOneRouter, so the gate does too.
            return _colab_router_check(ctx, spec.model)
        state = _inference_session(ctx) if spec.provider == COLAB else None
        limit = float(spec.timeout_s or ctx.timeout) + TIER_SLACK_SECONDS

        async def work(bridge: BridgeTransport) -> str:
            async with bot_roles(ctx, bridge, colab=state) as table:
                tier = _usable(table.rows(role)[index])
                async with asyncio.timeout(limit):
                    return await _tier_answer(settings, role, tier, bridge)

        try:
            evidence, _ = run_bot(ctx, work, timeout=limit)
        except TimeoutError:
            raise GateError(
                f"no answer within {limit:g}s (the tier's {spec.timeout_s:g}s + {TIER_SLACK_SECONDS:g}s)"
            ) from None
        return evidence

    return run


def tier_checks(settings: Settings) -> list[Check]:
    """One check per tier of *settings*' role table, role by role in call order."""
    try:
        specs = role_specs(settings)
    except RoleConfigError as exc:
        message = f"role table: {short(str(exc), 150)}"

        def broken(ctx: GateContext) -> str:
            raise GateError(message)

        return [Check(f"{TIER_CHECK_PREFIX}table", "tiers", "the configured model role table", broken)]
    return [
        Check(
            f"{TIER_CHECK_PREFIX}{role}.{index + 1}.{spec.label}",
            "tiers",
            f"{role} tier {index + 1}: {spec.label}",
            _tier_check(role, index, spec.label),
            provider=spec.provider,
        )
        for role in ROLES
        for index, spec in enumerate(specs[role])
    ]


# ── Runner ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool
    evidence: str
    seconds: float
    # Did not apply to this configuration; neither passed nor failed.
    skipped: bool = False


def select_checks(path: str, only: Iterable[str] = (), settings: Settings | None = None) -> list[Check]:
    """The checks of *path* ("all": every one), narrowed to the *only* ids or id prefixes.

    The tier checks come from *settings*' role table (code defaults when None).
    """
    chosen = [check for check in CHECKS if path == "all" or check.path == path]
    if path in ("all", "tiers"):
        chosen += tier_checks(settings if settings is not None else Settings(_env_file=None))
    wanted = [item.strip() for item in only if item.strip()]
    if wanted:
        chosen = [
            check
            for check in chosen
            if any(check.check_id == item or check.check_id.startswith(item + ".") for item in wanted)
        ]
    return chosen


def run_checks(
    ctx: GateContext,
    checks: Iterable[Check],
    *,
    progress: Callable[[str], None] | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        if check.path in KEY_NAMES and not ctx.key_for(check.path):
            results.append(CheckResult(check.check_id, False, f"{KEY_NAMES[check.path]} is not set", 0.0))
            continue
        provider = check.provider_in(ctx)
        if ctx.openrouter_capped_by and "openrouter" in (check.path, provider):
            results.append(CheckResult(check.check_id, False, DAILY_CAP_NOTE, 0.0, skipped=True))
            continue
        if progress is not None:
            progress(f"[run] {check.check_id}: {check.title}")
        if provider:
            # Before the clock starts: the wait is not the model's latency.
            ctx.pace(provider)
        polls_before = ctx.poll_count
        started = time.monotonic()
        skipped = False
        try:
            evidence, passed = check.run(ctx), True
        except GateSkip as exc:
            evidence, passed, skipped = str(exc), False, True
        except GateError as exc:
            evidence, passed = str(exc), False
        except httpx.TimeoutException:
            evidence, passed = "timed out", False
        except Exception as exc:  # network errors and unexpected response shapes
            evidence, passed = f"{type(exc).__name__}: {exc}", False
        if ctx.poll_count > polls_before:
            evidence += f" (HTTP 202, polled {ctx.poll_count - polls_before}x)"
        if not passed and OPENROUTER_DAILY_CAP_MARKER in evidence:
            # The account's daily cap: no more OpenRouter calls this run.
            ctx.openrouter_capped_by = ctx.openrouter_capped_by or check.check_id
        results.append(
            CheckResult(
                check.check_id,
                passed,
                short(ctx.redact(evidence), 200),
                time.monotonic() - started,
                skipped=skipped,
            )
        )
    return results


def format_table(results: list[CheckResult]) -> str:
    width = max([len("CHECK"), *(len(result.check_id) for result in results)])
    lines = [f"{'CHECK':<{width}}  RESULT  TIME     EVIDENCE"]
    for result in results:
        verdict = "SKIP" if result.skipped else "PASS" if result.passed else "FAIL"
        lines.append(
            f"{result.check_id:<{width}}  {verdict:<6}  {result.seconds:6.1f}s  {result.evidence}"
        )
    passed = sum(1 for result in results if result.passed)
    skipped = sum(1 for result in results if result.skipped)
    summary = f"{passed}/{len(results) - skipped} checks passed"
    lines.append(summary + (f", {skipped} skipped" if skipped else ""))
    return "\n".join(lines)


def is_tier_result(result: CheckResult) -> bool:
    return result.check_id.startswith(TIER_CHECK_PREFIX)


def format_tier_table(results: list[CheckResult]) -> str:
    """The tier checks as role, tier #, provider:model, result, latency and a short note."""
    rows: list[tuple[str, str, str, str, str, str]] = []
    for result in results:
        parts = result.check_id[len(TIER_CHECK_PREFIX) :].split(".", 2)
        role, number, label = parts if len(parts) == 3 else ("-", "-", parts[0])
        verdict = "SKIP" if result.skipped else "PASS" if result.passed else "FAIL"
        rows.append(
            (role, number, label, verdict, f"{result.seconds:.1f}s", short(result.evidence, TIER_NOTE_LIMIT))
        )
    header = ("ROLE", "#", "PROVIDER:MODEL", "RESULT", "LATENCY", "NOTE")
    widths = [max(len(row[column]) for row in (header, *rows)) for column in range(5)]
    lines = [
        "  ".join(
            [*(cell.ljust(width) for cell, width in zip(row[:5], widths, strict=True)), row[5]]
        ).rstrip()
        for row in (header, *rows)
    ]
    passed = sum(1 for result in results if result.passed)
    skipped = sum(1 for result in results if result.skipped)
    summary = f"{passed}/{len(results) - skipped} tiers passed"
    lines.append(summary + (f", {skipped} skipped" if skipped else ""))
    return "\n".join(lines)


def load_keys(env_file: Path, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """API keys per path; real environment variables win over the dotenv file."""
    environ = os.environ if environ is None else environ
    file_values: Mapping[str, str | None] = {}
    if env_file.is_file():
        from dotenv import dotenv_values

        file_values = dotenv_values(env_file)
    keys: dict[str, str] = {}
    for path, name in KEY_NAMES.items():
        value = (environ.get(name) or file_values.get(name) or "").strip()
        keys[path] = "" if value.lower().startswith("your_") else value
    return keys


def load_settings(env_file: Path) -> Settings:
    """The app's Settings from *env_file* (code defaults when it is missing).

    Raises:
        GateError: The file does not validate; names the fields only, since
            pydantic's own message echoes the offending values.
    """
    try:
        return Settings(_env_file=str(env_file) if env_file.is_file() else None)
    except ValidationError as exc:
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        raise GateError(f"invalid settings in {env_file.name}: {', '.join(fields)}") from None


@contextlib.contextmanager
def _quiet_app_logs() -> Iterator[None]:
    """Keep the app's own INFO/ERROR lines (which may quote model output) off the console."""
    app_logger = logging.getLogger("linebot")
    previous = app_logger.level
    app_logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        app_logger.setLevel(previous)


def build_parser() -> argparse.ArgumentParser:
    listing = "\n".join(f"  {check.check_id:<32} {check.title}" for check in CHECKS)
    listing += (
        f"\n  {'tier.<role>.<n>.<provider>:<model>':<32} --path tiers: one call per tier of the "
        "configured role table"
    )
    parser = argparse.ArgumentParser(
        description="Live contract checks for the v3 models (manual; uses real API keys).",
        epilog=f"checks:\n{listing}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        choices=("all", *PATHS),
        default="all",
        help="Provider path to check (default: all).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only these check ids or id prefixes (repeatable or comma-separated).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=_REPO_ROOT / ".env",
        help="Dotenv file with the keys and app settings (default: repo .env).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (thinking and image checks use at least 180).",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    client: httpx.Client | None = None,
    colab_loader: Callable[[Settings], ColabState] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    only = [item for value in args.only for item in value.split(",")]

    keys = load_keys(args.env_file)
    try:
        settings = load_settings(args.env_file)
    except GateError as exc:
        print(f"[live-gate] {exc}", file=sys.stderr)
        return 2
    # The tier checks follow the env file's role table, as the bot would.
    checks = select_checks(args.path, only, settings)
    if not checks:
        parser.error("no checks match --path/--only")
    overrides = untested_overrides(settings)
    if overrides:
        print(
            f"[live-gate] note: {', '.join(overrides)} change the role table; model checks "
            "whose tier is gone or off report SKIP; roles.* and tier.* cover the configured tiers",
            file=sys.stderr,
        )
    print(f"[live-gate] {len(checks)} check(s), path={args.path}", file=sys.stderr)
    with _quiet_app_logs(), client or httpx.Client(timeout=args.timeout) as http:
        ctx = GateContext(
            client=http,
            nvidia_key=keys["nvidia"],
            openrouter_key=keys["openrouter"],
            timeout=args.timeout,
            settings=settings,
            colab_loader=colab_loader,
        )
        results = run_checks(
            ctx, checks, progress=lambda line: print(line, file=sys.stderr, flush=True)
        )
    model_results = [result for result in results if not is_tier_result(result)]
    tier_results = [result for result in results if is_tier_result(result)]
    if model_results:
        print(format_table(model_results))
    if tier_results:
        if model_results:
            print()
        print(format_tier_table(tier_results))
    return 0 if all(result.passed or result.skipped for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
