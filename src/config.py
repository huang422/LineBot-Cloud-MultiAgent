"""Centralized configuration loaded from .env file.

All parameters, tokens, and model choices are configurable via environment
variables so users can easily change them without touching code.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


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
    openrouter_reasoning_enabled: bool = True
    openrouter_reasoning_effort: str = "high"
    openrouter_reasoning_exclude: bool = False
    require_reasoning_models: bool = True
    require_reasoning_tokens: bool = True

    # ── NVIDIA API ────────────────────────────────────────────
    nvidia_api_key: str = ""
    nvidia_model: str = "qwen/qwen3.5-122b-a10b"
    nvidia_thinking_enabled: bool = True
    nvidia_thinking_budget: int = 4096

    # ── Orchestrator (routing / task dispatch) ────────────────
    orchestrator_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    orchestrator_fallback_model: str = "qwen/qwen3.5-122b-a10b"
    orchestrator_temperature: float = 0.0
    orchestrator_max_tokens: int = 200

    # ── Text agents / Vision fallback ─────────────────────────
    agent_fallback_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    vision_fallback_model: str = "google/gemma-3-27b-it:free"

    # ── Per-agent temperature & max_tokens ────────────────────
    chat_temperature: float = 0.7
    chat_max_tokens: int = 2048

    vision_temperature: float = 0.5
    vision_max_tokens: int = 1024

    web_search_temperature: float = 0.3
    web_search_max_tokens: int = 3072

    image_gen_temperature: float = 0.7
    image_gen_max_tokens: int = 1024

    # ── Image generation (NVIDIA Stable Diffusion) ────────────
    image_gen_primary_model: str = "stabilityai/stable-diffusion-3-medium"
    image_gen_fallback_model: str = ""
    image_gen_steps: int = 50
    image_gen_cfg_scale: int = 5

    # ── Cost guardrails ────────────────────────────────────────
    line_push_fallback_enabled: bool = True
    line_push_monthly_limit: int = 0

    # ── GCP Cloud Storage ─────────────────────────────────────
    gcs_bucket_name: str = ""
    gcs_signed_url_expiry_hours: int = 48
    gcs_media_cleanup_delay_seconds: int = 172800  # best-effort in-process cleanup (2 days)

    # ── Tavily Web Search ─────────────────────────────────────
    tavily_api_key: str = ""
    web_search_monthly_quota: int = 1000

    # ── TTS ───────────────────────────────────────────────────
    tts_voice: str = "zh-TW-HsiaoChenNeural"
    tts_enabled: bool = True

    # ── Bot ───────────────────────────────────────────────────
    bot_name: str = "Assistant"
    line_bot_user_id: str = ""

    # ── Scheduler ──────────────────────────────────────────────
    scheduled_messages_enabled: bool = False
    scheduled_group_id: str = ""
    scheduled_weekly_messages: list[ScheduledWeeklyMessage] = Field(default_factory=list)
    scheduled_yearly_messages: list[ScheduledYearlyMessage] = Field(default_factory=list)

    # ── Conversation & Limits ─────────────────────────────────
    max_conversation_history: int = 10
    conversation_ttl_seconds: int = 3600
    rate_limit_max_requests: int = 30
    rate_limit_window_seconds: int = 60

    # ── Server ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {v}")
        return v

    @field_validator(
        "nvidia_model",
        "orchestrator_model",
        "orchestrator_fallback_model",
        "agent_fallback_model",
        "vision_fallback_model",
        "image_gen_primary_model",
    )
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Model ID must not be empty")
        return v

    @field_validator(
        "orchestrator_max_tokens",
        "chat_max_tokens",
        "vision_max_tokens",
        "web_search_max_tokens",
        "image_gen_max_tokens",
        "image_gen_steps",
        "image_gen_cfg_scale",
        "nvidia_thinking_budget",
        "max_conversation_history",
        "conversation_ttl_seconds",
        "rate_limit_max_requests",
        "rate_limit_window_seconds",
        "port",
        "gcs_media_cleanup_delay_seconds",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be greater than 0")
        return v

    @field_validator(
        "line_push_monthly_limit",
        "web_search_monthly_quota",
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
    )
    @classmethod
    def validate_non_negative_float(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Value must be greater than or equal to 0")
        return v

    @field_validator("openrouter_reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str) -> str:
        v = v.strip().lower()
        allowed = {"xhigh", "high", "medium", "low", "minimal", "none"}
        if v not in allowed:
            raise ValueError(f"Invalid openrouter_reasoning_effort: {v}")
        return v

    @field_validator("gcs_signed_url_expiry_hours")
    @classmethod
    def validate_positive_expiry(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("GCS signed URL expiry must be greater than 0")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
