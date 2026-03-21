"""Unified response model from any agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Result from an agent, ready for output processing."""

    text: str | None = None
    image_base64: str | None = None  # base64 data URL for generated images
    audio_url: str | None = None  # public URL to TTS audio
    agent_name: str = ""
    model_used: str = ""
    output_format: str = "text"  # text | voice | image
