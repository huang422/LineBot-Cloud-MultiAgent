"""Centralized configuration loaded from .env file.

All parameters, tokens, and model choices are configurable via environment
variables so users can easily change them without touching code.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, NoDecode

from src.providers.model_profiles import (
    DEFAULT_THINKING_BUDGET,
    MAX_REASONING_BUDGET,
    PROVIDERS,
    THINK_LEVELS,
)

# The dotenv files Settings reads, in order; a later file wins, a missing
# one is skipped, and a real environment variable beats both. Locally that
# is .env (relative to the working dir). On Cloud Run the deploy mounts the
# Secret Manager secret linebot-dotenv (a dotenv blob made from .env) at
# SECRET_ENV_FILE and sets only APP_VERSION as a plain variable. ADR-0008.
ENV_FILE = ".env"
SECRET_ENV_FILE = "/secrets/app/app.env"
ENV_FILES: tuple[str, ...] = (ENV_FILE, SECRET_ENV_FILE)

# Model ids NVIDIA has retired: the API answers HTTP 410 "reached its end
# of life". scripts/envfile.py imports this set so a deploy refuses an env
# file that still pins one of them.
RETIRED_MODEL_IDS: frozenset[str] = frozenset(
    {
        "qwen/qwen3.5-397b-a17b",  # end of life 2026-07-27
        "qwen/qwen3.5-122b-a10b",
        "nvidia/nv-embedqa-e5-v5",  # end of life 2026-08-25
    }
)

# Env keys earlier releases read and the current Settings ignores
# (extra="ignore"), so a stale value would fail silently rather than loudly.
LEGACY_ENV_KEYS: frozenset[str] = frozenset(
    {"ORCHESTRATOR_MODEL", "ORCHESTRATOR_FALLBACK_MODEL"}
)

# Env keys removed because the behaviour behind them is gone. Unlike
# LEGACY_ENV_KEYS a stale value is harmless, so startup only mentions it and
# the deploy guard (scripts/envfile.py check-v3) does not fail on it.
_ROLE_TABLE_NOTE = (
    "models, their order and their options now come from the model role table; "
    "override a role with MODEL_ROLES (see .env.example)"
)
REMOVED_ENV_KEYS: dict[str, str] = {
    "GCS_MEDIA_CLEANUP_DELAY_SECONDS": (
        "media now expires through the bucket lifecycle rule (3-day Delete)"
    ),
    "THINKING_TIMEOUT_SECONDS": (
        "every model tier now has its own time limit inside the reply or turn "
        "deadline (TURN_DEADLINE_SECONDS), and the thinking tiers leave half the "
        "time left to the non-thinking fallback"
    ),
    "NVIDIA_THINKING_MODEL": (
        "there is no separate thinking model; the thinking role of the model role "
        "table runs Nemotron Ultra with thinking on"
    ),
    **{
        key: _ROLE_TABLE_NOTE
        for key in (
            "NVIDIA_MODEL",
            "NVIDIA_VISION_MODEL",
            "NVIDIA_UTILITY_MODEL",
            "NVIDIA_EMBEDDING_MODEL",
            "ROUTER_MODEL",
            "ROUTER_FALLBACK_MODEL",
            "AGENT_FALLBACK_MODEL",
            "AGENT_FALLBACK_MODEL_2",
            "VISION_FALLBACK_MODEL",
            "VISION_FALLBACK_MODEL_2",
            "UTILITY_FALLBACK_MODEL",
            "IMAGE_GEN_PRIMARY_MODEL",
            "IMAGE_GEN_FALLBACK_MODEL",
            "IMAGE_GEN_STEPS",
            "IMAGE_GEN_CFG_SCALE",
        )
    },
    "NVIDIA_THINKING_ENABLED": (
        "thinking is switched per tier (the thinking role, or a tier's "
        "thinking/effort/think option in MODEL_ROLES)"
    ),
    "OPENROUTER_REASONING_ENABLED": (
        "OpenRouter reasoning is set per tier (effort in MODEL_ROLES); off is "
        "always sent explicitly"
    ),
    "OPENROUTER_REASONING_EFFORT": "OpenRouter effort is set per tier (effort in MODEL_ROLES)",
    "OPENROUTER_REASONING_EXCLUDE": "reasoning text is always returned and kept out of replies",
    "OPENROUTER_THINKING_BUDGET": (
        "NVIDIA_THINKING_BUDGET (or a tier's reasoning_budget) sizes every thinking tier"
    ),
    "REQUIRE_REASONING_MODELS": (
        "the registry filter is gone; the role table lists exactly the tiers that are called"
    ),
    "REQUIRE_REASONING_TOKENS": "a reply without reasoning tokens is no longer flagged",
    "WEB_SEARCH_MONTHLY_QUOTA": (
        "search has no app-side quota; Tavily reports the credits it used "
        "(/health/detail web_search)"
    ),
    **{
        key: (
            "self-hosted SearXNG search was removed on 2026-10-01 to stay in the GCP "
            "free tier; web search is Tavily only (ADR-0009)"
        )
        for key in (
            "SEARXNG_BASE_URL",
            "SEARXNG_TIMEOUT_SECONDS",
            "SEARXNG_TIMEOUT_LIMIT_SECONDS",
        )
    },
    "COLAB_REPLY_WINDOW_SECONDS": (
        "since 2026-10-01 every request has the one answer window REPLY_WINDOW_SECONDS "
        "(90 s from the LINE event); a Colab tier still stops 20 s before it so an API "
        "tier can answer"
    ),
    "FIRESTORE_LOCATION": (
        "nothing ever read it: a Firestore database's location is fixed when it is "
        "created (gcloud firestore databases create --location, DEPLOY.md)"
    ),
    **{
        key: (
            "the push budget was removed on 2026-10-01: every answer tries the free "
            "reply first and pushes only when the reply certainly failed, and no push "
            "is counted, capped or switched off (LINE's monthly quota is the only limit)"
        )
        for key in (
            "LINE_PUSH_FALLBACK_ENABLED",
            "LINE_REPLY_FALLBACK_PUSH_ENABLED",
            "LINE_PUSH_RESERVE",
            "LINE_PUSH_MONTHLY_LIMIT",
        )
    },
}

# MODEL_PATH_PREFERENCE: Colab tiers first ("colab") or last ("api").
MODEL_PATH_PREFERENCES: tuple[str, ...] = ("colab", "api")

# COLAB_SHARED_SECRET is an HMAC key: 32 random characters or more.
MIN_COLAB_SECRET_LENGTH = 32
# COLAB_TUNNEL_MODE: a Cloudflare quick tunnel, or a named tunnel on
# COLAB_TUNNEL_HOST (needs COLAB_TUNNEL_TOKEN).
COLAB_TUNNEL_MODES: tuple[str, ...] = ("quick", "named")
# Colab reclaims a runtime about 22-25 minutes after the last kernel
# activity; the keep-alive cell must run well inside that.
MAX_COLAB_HEARTBEAT_SECONDS = 900
# The defaults of CLOUD_TASKS_QUEUE and CLOUD_TASKS_COLAB_QUEUE (the deploy
# script's too).
DEFAULT_TASKS_QUEUE = "linebot-events"
DEFAULT_COLAB_TASKS_QUEUE = "linebot-colab"
_HOSTNAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+")

TASKS_MODES: frozenset[str] = frozenset({"auto", "cloud", "inline"})

# Settings whose empty (or blank) value means the default, as
# scripts/deploy_cloud_run.sh reads them: it checks the default in place of
# an empty value (and sets --timeout and the queues from it), so the app
# must not refuse what the deploy accepted. Keys whose default is empty
# need no entry; CLOUD_TASKS_LOCATION stays out on purpose (empty means the
# Cloud Run region). tests/test_config_empty_values.py checks the list
# against the deploy script. SCHEDULED_MESSAGES_TIMEZONE, which the deploy
# does not read, is here too: an empty zone name is no zone at all.
EMPTY_MEANS_DEFAULT: tuple[str, ...] = (
    "tasks_mode",
    "task_dispatch_deadline_seconds",
    "task_handler_budget_seconds",
    "cloud_tasks_queue",
    "cloud_tasks_colab_queue",
    "scheduled_messages_enabled",
    "scheduled_weekly_messages",
    "scheduled_yearly_messages",
    "scheduled_messages_timezone",
    "passive_group_recording_enabled",
    "colab_enabled",
    "colab_adc_path",
)


class ScheduledWeeklyMessage(BaseModel):
    """Declarative weekly scheduler entry loaded from env."""

    id: str
    day_of_week: str
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    message: str

    @field_validator("id", "message")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Value must not be empty")
        return v

    @field_validator("day_of_week")
    @classmethod
    def validate_day_of_week(cls, v: str) -> str:
        v = v.strip().lower()
        allowed = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        if v not in allowed:
            raise ValueError(f"Invalid day_of_week: {v}")
        return v


class ScheduledYearlyMessage(BaseModel):
    """Declarative yearly scheduler entry loaded from env."""

    id: str
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    message: str

    @field_validator("id", "message")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Value must not be empty")
        return v


class Settings(BaseSettings):
    """Application settings – every field maps to an env var."""

    # ── LINE API ──────────────────────────────────────────────
    line_channel_secret: str = ""
    line_channel_access_token: str = ""

    # ── OpenRouter API ────────────────────────────────────────
    openrouter_api_key: str = ""
    # Only ":free" ids (and the openrouter/free router) may be configured
    # unless this is true; the role table refuses a paid id at startup.
    openrouter_allow_paid: bool = False

    # ── NVIDIA API ────────────────────────────────────────────
    nvidia_api_key: str = ""
    # Thinking budget of a thinking tier that sets none: max_tokens headroom
    # for the thoughts, also sent as reasoning_budget to the models that take
    # one (Nemotron Super; Ultra rejects it and thinks at medium_effort).
    nvidia_thinking_budget: int = Field(default=DEFAULT_THINKING_BUDGET, le=MAX_REASONING_BUDGET)
    nvidia_embedding_endpoint: str = "https://integrate.api.nvidia.com/v1/embeddings"
    # The model returns 2048 dims; the client keeps this many leading dims
    # (re-normalised) so vectors fit the existing 1024-d Firestore index.
    embedding_dimensions: int = 1024

    # ── Model role table (src/runtime/model_roles.py) ─────────
    # JSON object {role: [tier, ...]}; each role given replaces that role's
    # code default. Roles: router, chat, thinking, vision, utility, embed,
    # image. Empty = the defaults.
    model_roles: str = ""
    # "colab": Colab tiers first in every role; "api": NVIDIA -> OpenRouter,
    # with Colab tiers last (used only when Colab is already up).
    model_path_preference: str = "colab"
    # Ollama think level of the Colab qwen3.8 tiers: chat and vision, and
    # the thinking role. false, low, medium or xhigh.
    colab_chat_think: str = "low"
    colab_deep_think: str = "medium"

    # ── Colab tier (ADR-0010) ─────────────────────────────────
    # Master switch for the Colab Ollama tiers. Needs FIRESTORE_ENABLED (the
    # session state lives in {prefix}_colab/state), COLAB_SHARED_SECRET,
    # INTERNAL_CRON_SECRET (the controller runs from /internal/cron) and a
    # readable COLAB_ADC_PATH; without them startup logs a warning and the
    # tier stays off (src/runtime/composition.py colab_config_problems).
    colab_enabled: bool = False
    # Both sides derive each session's registration key and Ollama bearer
    # from it (src/services/colab/protocol.py); at least 32 characters.
    colab_shared_secret: str = ""
    # The Colab account's authorized_user refresh token (Secret Manager
    # linebot-colab-adc, mounted by the deploy); read by the control plane
    # only, never exported as GOOGLE_APPLICATION_CREDENTIALS.
    colab_adc_path: str = "/secrets/colab/adc.json"
    # "quick": https://<random>.trycloudflare.com; "named": a named tunnel
    # on COLAB_TUNNEL_HOST whose token is COLAB_TUNNEL_TOKEN.
    colab_tunnel_mode: str = "quick"
    # The named tunnel's hostname; with quick tunnels leave it empty.
    colab_tunnel_host: str = ""
    colab_tunnel_token: str = ""
    # The Colab router (nimble on /v1/systemone) routes first (colab
    # preference) or last (api) once the session is READY. Its gates belong
    # to the router model: below COLAB_ROUTER_CONFIDENCE (normalised
    # entropy) on either choice the API router decides; needs_thinking at
    # or above THINK_ON turns thinking on, at or below THINK_OFF off, and
    # in between the rulebook decides. These are provisional, strict values
    # for nimble:9b, whose probabilities are uncalibrated: re-tune them
    # with scripts/evals/routing_eval.py (ADR-0010).
    colab_router_enabled: bool = True
    colab_router_confidence: float = 0.75
    colab_router_think_on: float = 0.85
    colab_router_think_off: float = 0.15
    # The session is released after this long without a triggered message,
    # or at this age; no start below the balance floor or past the daily
    # cap (compute units, from Colab's consumption rate).
    colab_idle_ttl_seconds: int = 900
    colab_max_session_seconds: int = 39600
    colab_min_cu_balance: float = 10.0
    colab_daily_cu_cap: float = 30.0
    # A session that has not registered READY this long after its assign is
    # released (and counts as a boot failure; three in a row disable it).
    colab_boot_timeout_seconds: int = 900
    # The keep-alive cell's interval while a session is booting or in use
    # (Colab only counts kernel activity, not tunnel traffic).
    colab_heartbeat_seconds: int = 300
    # True: the Colab account is the bot's alone, so runtimes the bot does
    # not know are released too; false: only logged (they may be yours).
    colab_account_dedicated: bool = False
    # Dry run: a CPU runtime (no A100) and DRYRUN=1 on the VM, whose stub
    # Ollama passes the readiness probe but answers no inference, so the
    # whole control path runs for a fraction of the compute units.
    colab_dry_run: bool = False
    # LINE user ids (comma-separated) allowed to run !colab on|off|status;
    # send !whoami to the bot in a 1:1 chat to see yours.
    line_admin_user_ids: str = ""

    # ── Router (orchestrator LLM classification) ──────────────
    # NVIDIA NIM rejects temperature 0, so routing uses a small positive one.
    orchestrator_temperature: float = 0.1
    orchestrator_max_tokens: int = 384

    # ── Tool calling (Phase C/B) ──────────────────────────────
    # Hard ceiling on agentic tool-loop iterations per request so a
    # mis-behaved model cannot pin Cloud Run for the full request
    # timeout. Internal knob only — not exposed via .env.
    tool_loop_max_iterations: int = 4

    # ── OpenRouter free quota ─────────────────────────────────
    # Account-wide cap on ":free" requests per day (50, or 1000 once the
    # account has bought $10 of credits); shared by every free id.
    openrouter_free_daily_limit: int = 50

    # ── Model tiers ───────────────────────────────────────────
    # Hard limit for one turn, counted from when work on the event starts:
    # a thinking request or an image, in any chat, or a request that started
    # too late for the reply token (each answered by push once the token is
    # gone). Below the task handler budget, so the handler is never
    # cancelled first.
    turn_deadline_seconds: int = 240
    # An answer that starts with less than this left of the reply window
    # works to the turn deadline instead (delivery still tries the reply
    # first; push only if LINE refuses it).
    # Also the least time a tier must have had before a deadline timeout
    # counts against its model.
    min_tier_seconds: int = 5
    # Test-only. "provider" or "provider:model": the role table disables
    # every other tier, e.g. FORCE_TIER=openrouter to exercise one path end
    # to end. Empty = off.
    force_tier: str = ""

    # ── Per-agent temperature & max_tokens ────────────────────
    # The temperatures reach only tiers whose model has no card sampling in
    # its profile and that set none of their own (model_profiles.tier_sampling).
    chat_temperature: float = 0.7
    chat_max_tokens: int = 2048

    vision_temperature: float = 0.5
    vision_max_tokens: int = 1024

    web_search_temperature: float = 0.2
    web_search_max_tokens: int = 3072

    image_gen_temperature: float = 0.7
    image_gen_max_tokens: int = 1024

    # ── GCP Cloud Storage ─────────────────────────────────────
    gcs_bucket_name: str = ""
    # Objects expire through the bucket's 3-day lifecycle Delete rule, which
    # scripts/deploy_cloud_run.sh enforces; the app never waits to delete.
    gcs_signed_url_expiry_hours: int = 48

    # ── Web search (ADR-0009): Tavily, then no search ──
    # How many of the top results are read (page text instead of the
    # snippet); 0 = snippets only. Pages are capped at max_bytes, and all
    # reads of one search (or of the URLs in one message) share the budget.
    search_fetch_top_n: int = 3
    search_fetch_max_bytes: int = 2_000_000
    search_fetch_budget_seconds: float = 6.0
    # Reorder results by embedding similarity before reading them.
    search_rerank_enabled: bool = False
    # Optional. Without a key, Tavily runs keyless (free, rate-limited)
    # only with TAVILY_KEYLESS_ENABLED: v2 documented an empty key as "no
    # web search", and such a .env must not start sending queries to Tavily.
    tavily_api_key: str = ""
    tavily_keyless_enabled: bool = False

    # ── TTS ───────────────────────────────────────────────────
    tts_voice: str = "zh-TW-HsiaoChenNeural"
    tts_enabled: bool = True

    # ── Bot ───────────────────────────────────────────────────
    bot_name: str = "Assistant"
    line_bot_user_id: str = ""

    # ── Scheduler ──────────────────────────────────────────────
    scheduled_messages_enabled: bool = False
    scheduled_group_id: str = ""
    # JSON arrays, decoded by decode_job_list (so an empty value is no jobs).
    scheduled_weekly_messages: Annotated[list[ScheduledWeeklyMessage], NoDecode] = Field(
        default_factory=list
    )
    scheduled_yearly_messages: Annotated[list[ScheduledYearlyMessage], NoDecode] = Field(
        default_factory=list
    )
    scheduled_messages_timezone: str = "Asia/Taipei"
    # Shared secret used by /internal/cron. Cloud Scheduler sends this in the
    # X-Cron-Token header; the endpoint refuses access when the secret is empty
    # or does not match. Leave blank in dev to disable the endpoint entirely.
    internal_cron_secret: str = ""

    # ── Post-webhook work (Cloud Tasks) ───────────────────────
    # auto: Cloud Tasks on Cloud Run (K_SERVICE set) when INTERNAL_TASKS_SECRET
    # is set, otherwise run the work inline inside the request. See ADR-0006.
    tasks_mode: str = "auto"
    # Empty is the default, as in the deploy script (EMPTY_MEANS_DEFAULT).
    cloud_tasks_queue: str = DEFAULT_TASKS_QUEUE
    # The Colab controller's ``colab`` tasks (seconds to minutes each) go to
    # their own queue, so they never take the event queue's dispatch slots
    # (ADR-0010). Must differ from CLOUD_TASKS_QUEUE; empty is the default.
    cloud_tasks_colab_queue: str = DEFAULT_COLAB_TASKS_QUEUE
    # Empty: the Cloud Run region from the metadata server.
    cloud_tasks_location: str = "us-west1"
    # Empty: derived from the first authenticated request's Host header.
    public_base_url: str = ""
    # Shared secret Cloud Tasks sends in X-Tasks-Token. Separate from
    # INTERNAL_CRON_SECRET on purpose; empty makes /internal/tasks refuse all.
    internal_tasks_secret: str = ""
    # Cloud Tasks allows 15 s..30 min. The deploy script sets Cloud Run's
    # --timeout to max(300, this), so a task request is never cut off by
    # Cloud Run first (a 504 leaves the handler running without CPU).
    task_dispatch_deadline_seconds: int = Field(default=300, ge=15, le=1800)
    # Each task handler is cancelled after this many seconds. Must stay
    # below the dispatch deadline (the deploy script refuses otherwise).
    task_handler_budget_seconds: int = 280
    # How long after the LINE event a plain answer may take, whoever answers
    # it, in 1:1 chats and groups alike. LINE only guarantees a reply token
    # for about a minute: the reply is still tried first, and an answer
    # whose token LINE refuses goes out by push. Below TURN_DEADLINE_SECONDS.
    reply_window_seconds: int = 90

    # ── Rate limiting ─────────────────────────────────────────
    rate_limit_max_requests: int = 30
    rate_limit_window_seconds: int = 60

    # ── Conversation Memory ───────────────────────────────────
    memory_recent_message_limit: int = 6
    memory_summary_timeout_seconds: int = 180
    memory_summary_temperature: float = 0.2
    memory_summary_max_tokens: int = 768
    # A chat that crossed the limit within this many seconds of its last
    # compaction waits for a later message (or for its window to reach half
    # the hard cap) before it is compacted again; 0 turns it off.
    compaction_debounce_seconds: int = 120
    # At most one episode per chat in this many seconds; compactions in
    # between add their conversation to the next one, which keeps the
    # newest 2000 characters of it. 0 turns it off.
    episode_min_interval_seconds: int = 3600

    # ── Passive group recording (Firestore writes for non-triggered group chat) ─
    # When enabled, every text message in a group/room is buffered and
    # eventually written to Firestore so the bot can recall the full
    # conversation thread, not just turns where it was @-mentioned. To
    # keep Firestore writes bounded, messages are coalesced per chat and
    # written by a ``passive_flush`` task once the buffer holds
    # ``passive_group_flush_threshold`` messages or a new message arrives
    # after the oldest one is ``passive_group_flush_delay_seconds`` old;
    # right before the bot answers a trigger in that chat; by the
    # per-minute /internal/cron sweep once the oldest message is older than
    # the delay (only with INTERNAL_CRON_SECRET set, so up to a minute
    # late); or on shutdown. There is no timer (ADR-0006).
    passive_group_recording_enabled: bool = True
    passive_group_flush_threshold: int = 8
    passive_group_flush_delay_seconds: int = 30

    # ── Persistent memory backend (Firestore Native) ──────────
    # When firestore_enabled is False (default), MemoryService keeps using
    # the in-memory backend, which is wiped on Cloud Run cold start /
    # redeploy. Setting firestore_enabled=true switches to Firestore.
    # All other firestore_* fields are only consulted when enabled.
    firestore_enabled: bool = False
    firestore_project_id: str = ""
    firestore_database: str = "(default)"
    firestore_collection_prefix: str = "linebot"
    memory_cache_ttl_seconds: int = 60
    line_profile_cache_ttl_seconds: int = 86400

    # ── Server ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    @field_validator(*EMPTY_MEANS_DEFAULT, mode="before")
    @classmethod
    def empty_means_default(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise PydanticUseDefault()
        return v

    @field_validator("scheduled_weekly_messages", "scheduled_yearly_messages", mode="before")
    @classmethod
    def decode_job_list(cls, v: object) -> object:
        # A blank value is left to empty_means_default.
        return json.loads(v) if isinstance(v, str) and v.strip() else v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {v}")
        return v

    @field_validator(
        "orchestrator_max_tokens",
        "chat_max_tokens",
        "vision_max_tokens",
        "web_search_max_tokens",
        "image_gen_max_tokens",
        "nvidia_thinking_budget",
        "embedding_dimensions",
        "rate_limit_max_requests",
        "tool_loop_max_iterations",
        "rate_limit_window_seconds",
        "memory_recent_message_limit",
        "memory_summary_timeout_seconds",
        "memory_summary_max_tokens",
        "memory_cache_ttl_seconds",
        "line_profile_cache_ttl_seconds",
        "passive_group_flush_threshold",
        "passive_group_flush_delay_seconds",
        "port",
        "gcs_signed_url_expiry_hours",
        "task_handler_budget_seconds",
        "reply_window_seconds",
        "turn_deadline_seconds",
        "min_tier_seconds",
        "openrouter_free_daily_limit",
        "search_fetch_max_bytes",
        "colab_idle_ttl_seconds",
        "colab_max_session_seconds",
        "colab_boot_timeout_seconds",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be greater than 0")
        return v

    @field_validator(
        "search_fetch_top_n",
        "compaction_debounce_seconds",
        "episode_min_interval_seconds",
    )
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be greater than or equal to 0")
        return v

    @field_validator(
        "orchestrator_temperature",
        "chat_temperature",
        "vision_temperature",
        "web_search_temperature",
        "image_gen_temperature",
        "memory_summary_temperature",
        "colab_min_cu_balance",
    )
    @classmethod
    def validate_non_negative_float(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Value must be greater than or equal to 0")
        return v

    @field_validator(
        "search_fetch_budget_seconds",
        "colab_daily_cu_cap",
    )
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Value must be greater than 0")
        return v

    @field_validator("tasks_mode")
    @classmethod
    def validate_tasks_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in TASKS_MODES:
            raise ValueError(f"Invalid tasks_mode: {v} (expected auto, cloud or inline)")
        return v

    @field_validator("public_base_url")
    @classmethod
    def validate_public_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        # Cloud Tasks sends X-Tasks-Token to this URL, and an http target
        # would only hit run.app's redirect, which a single-attempt queue
        # counts as a failed task.
        if v and not v.lower().startswith("https://"):
            raise ValueError("public_base_url must start with https://")
        return v

    # An empty queue name means the default queue (EMPTY_MEANS_DEFAULT); an
    # empty name would fail every CreateTask, running each task inline.
    @field_validator(
        "cloud_tasks_location",
        "internal_tasks_secret",
        "cloud_tasks_queue",
        "cloud_tasks_colab_queue",
    )
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

    @field_validator("force_tier")
    @classmethod
    def validate_force_tier(cls, v: str) -> str:
        v = v.strip()
        provider = v.partition(":")[0]
        if v and provider not in PROVIDERS:
            raise ValueError(
                f"Invalid force_tier provider: {provider} (expected {', '.join(PROVIDERS)})"
            )
        return v

    @field_validator("model_path_preference")
    @classmethod
    def validate_model_path_preference(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in MODEL_PATH_PREFERENCES:
            raise ValueError(f"Invalid model_path_preference: {v} (expected colab or api)")
        return v

    @field_validator("colab_chat_think", "colab_deep_think")
    @classmethod
    def validate_colab_think(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in THINK_LEVELS:
            raise ValueError(f"Invalid Colab think level: {v} (expected false, low, medium or xhigh)")
        return v

    @field_validator("colab_router_confidence", "colab_router_think_on")
    @classmethod
    def validate_colab_router_high_gate(cls, v: float) -> float:
        # Under 0.5 a gate would take a guess for an answer; 1 never passes.
        if not 0.5 <= v <= 0.99:
            raise ValueError(
                "colab_router_confidence and colab_router_think_on must be between 0.5 and 0.99"
            )
        return v

    @field_validator("colab_router_think_off")
    @classmethod
    def validate_colab_router_think_off(cls, v: float) -> float:
        if not 0.01 <= v <= 0.5:
            raise ValueError("colab_router_think_off must be between 0.01 and 0.5")
        return v

    @model_validator(mode="after")
    def validate_reply_window_below_turn_deadline(self) -> Settings:
        # The turn deadline stays below the task handler budget (a startup
        # warning otherwise), so a plain answer is never cut off first.
        if self.reply_window_seconds >= self.turn_deadline_seconds:
            raise ValueError("REPLY_WINDOW_SECONDS must be below TURN_DEADLINE_SECONDS")
        return self

    @model_validator(mode="after")
    def validate_colab_router_think_gates(self) -> Settings:
        if self.colab_router_think_on <= self.colab_router_think_off:
            raise ValueError("COLAB_ROUTER_THINK_ON must be above COLAB_ROUTER_THINK_OFF")
        return self

    @field_validator("colab_shared_secret", "line_admin_user_ids", "colab_tunnel_token", "colab_adc_path")
    @classmethod
    def strip_secret_text(cls, v: str) -> str:
        return v.strip()

    @field_validator("colab_tunnel_mode")
    @classmethod
    def validate_colab_tunnel_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in COLAB_TUNNEL_MODES:
            raise ValueError(f"Invalid colab_tunnel_mode: {v} (expected quick or named)")
        return v

    @field_validator("colab_heartbeat_seconds")
    @classmethod
    def validate_colab_heartbeat(cls, v: int) -> int:
        # The cron runs once a minute; Colab reclaims an idle kernel in ~22 min.
        if not 60 <= v <= MAX_COLAB_HEARTBEAT_SECONDS:
            raise ValueError(
                f"colab_heartbeat_seconds must be between 60 and {MAX_COLAB_HEARTBEAT_SECONDS}"
            )
        return v

    @field_validator("colab_tunnel_host")
    @classmethod
    def validate_colab_tunnel_host(cls, v: str) -> str:
        v = v.strip().rstrip(".").lower()
        if v and not _HOSTNAME_RE.fullmatch(v):
            raise ValueError("colab_tunnel_host must be a bare hostname (no scheme, port or path)")
        return v

    @property
    def admin_user_ids(self) -> frozenset[str]:
        """LINE_ADMIN_USER_IDS as a set (commas or whitespace separate ids)."""
        return frozenset(part for part in re.split(r"[\s,]+", self.line_admin_user_ids) if part)

    model_config = {
        "env_file": ENV_FILES,
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


def legacy_config_warnings(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = ENV_FILES,
) -> list[str]:
    """Describe stale configuration the current Settings would silently accept.

    Covers legacy env keys that are now ignored and keys whose behaviour was
    removed (the per-model settings the role table replaced among them).
    Messages name env keys only, never values. Retired model ids in
    MODEL_ROLES fail the role table instead.

    Args:
        environ: Process environment; defaults to ``os.environ``.
        env_file: Dotenv file, or files, to scan for legacy keys (by
            default the ones Settings reads); missing files are skipped,
            ``None`` skips them all.
    """
    sources: list[tuple[str, list[str]]] = [
        ("the environment", list(os.environ if environ is None else environ))
    ]
    if env_file is None:
        env_files: Sequence[str | os.PathLike[str]] = ()
    elif isinstance(env_file, (str, os.PathLike)):
        env_files = (env_file,)
    else:
        env_files = env_file
    for path in env_files:
        if not Path(path).is_file():
            continue
        try:
            # Only key names are used; interpolation is pointless here.
            keys = list(dotenv_values(path, interpolate=False))
        except (OSError, UnicodeDecodeError):
            keys = []
        sources.append((str(path), keys))

    messages: list[str] = []
    for source, keys in sources:
        legacy = sorted({key.upper() for key in keys if key.upper() in LEGACY_ENV_KEYS})
        if legacy:
            messages.append(
                f"{', '.join(legacy)} in {source} is no longer read; routing now "
                "uses the router role of the model role table. Remove the stale key."
            )

        removed = sorted({key.upper() for key in keys if key.upper() in REMOVED_ENV_KEYS})
        for key in removed:
            messages.append(
                f"{key} in {source} is no longer read: {REMOVED_ENV_KEYS[key]}. "
                "Remove the stale key."
            )
    return messages
