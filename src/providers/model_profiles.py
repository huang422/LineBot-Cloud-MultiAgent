"""What each model the bot may call accepts, keyed by exact (provider, id).

Profiles are facts, verified against the providers' docs, the model cards
and live calls (2026-09-29/30), never guessed from a substring:
similar-looking ids (e.g. nemotron-3 vs nemotron-3.5) take different
parameters. The role table (:mod:`src.runtime.model_roles`) validates its
tiers against them and the providers build their payloads from them.

A profile may carry the model card's default sampling (temperature, top_p).
It replaces the call's temperature (CHAT_TEMPERATURE and friends), so every
tier of a mixed-model role samples the way its own model is meant to; a
tier's own ``temperature`` still wins (the router and utility tiers keep a
deliberately low one, see :mod:`src.runtime.model_roles`).

An NVIDIA or OpenRouter id without a profile still works:
:func:`generic_profile` stands in with the provider's plain protocol, and
the role table reports it as a config warning instead of dropping the tier.
Colab serves only the models its VM pins, all profiled here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# ── Protocols: how a request to the model is shaped ──────────

# chat_template_kwargs.enable_thinking (plus the profile's extra thinking
# kwargs, and a top-level reasoning_budget where the model takes one).
NIM_NEMOTRON3 = "nim_nemotron3"
# Top-level reasoning_effort ("none" off, "medium" on); no chat_template_kwargs.
NIM_REASONING_EFFORT = "nim_reasoning_effort"
OPENROUTER = "openrouter"  # OpenAI chat + a "reasoning" object
OLLAMA_CHAT = "ollama_chat"  # Ollama native /api/chat ("think")
SYSTEMONE = "systemone"  # Ollama /v1/systemone multiple-choice scoring
NIM_EMBED = "nim_embed"
NIM_FLUX2_KLEIN = "nim_flux2_klein"
NIM_FLUX1_DEV = "nim_flux1_dev"
# An id without a profile: plain OpenAI chat on NVIDIA (no thinking
# controls), or the prompt alone for an unknown NVIDIA image model.
GENERIC = "generic"

# The profiled NVIDIA NIM chat protocols (anything else on NVIDIA is GENERIC).
NIM_CHAT_PROTOCOLS: frozenset[str] = frozenset({NIM_NEMOTRON3, NIM_REASONING_EFFORT})
CHAT_PROTOCOLS: frozenset[str] = NIM_CHAT_PROTOCOLS | {OPENROUTER, OLLAMA_CHAT, GENERIC}
EMBED_PROTOCOLS: frozenset[str] = frozenset({NIM_EMBED, GENERIC})
IMAGE_PROTOCOLS: frozenset[str] = frozenset({NIM_FLUX2_KLEIN, NIM_FLUX1_DEV, GENERIC})

# What a role asks of a model (RoleTable maps each role onto one kind).
CHAT = "chat"
EMBED = "embed"
IMAGE = "image"

# ── Option vocabularies ──────────────────────────────────────
# The one copy of each; src.config and src.runtime.model_roles import them.

# Providers a tier can name (FORCE_TIER, MODEL_ROLES).
PROVIDERS: tuple[str, ...] = ("colab", "nvidia", "openrouter")

JSON_SCHEMA = "json_schema"
JSON_OBJECT = "json_object"
PROMPT = "prompt"
JSON_MODES: frozenset[str] = frozenset({JSON_SCHEMA, JSON_OBJECT, PROMPT})

# Ollama's think values for qwen3.8 (0.35.0). "false" is sent as JSON false.
THINK_LEVELS: tuple[str, ...] = ("false", "low", "medium", "xhigh")

TEXT: frozenset[str] = frozenset({"text"})
TEXT_IMAGE: frozenset[str] = frozenset({"text", "image"})
TEXT_IMAGE_VIDEO: frozenset[str] = frozenset({"text", "image", "video"})

# NIM rejects temperature 0 (its schemas use exclusiveMinimum 0).
NIM_MIN_TEMPERATURE = 0.01
# reasoning_budget accepts at most this many tokens on every NIM schema.
MAX_REASONING_BUDGET = 32_768
# A thinking tier without its own reasoning_budget gets NVIDIA_THINKING_BUDGET
# (max_tokens headroom for the thoughts; also sent as reasoning_budget to the
# models that take one); this is that setting's default.
DEFAULT_THINKING_BUDGET = 4096


@dataclass(frozen=True)
class ModelProfile:
    """How one model is called and what it can do."""

    provider: str
    model: str
    protocol: str
    # Input the model accepts; image requests skip text-only tiers.
    modalities: frozenset[str] = TEXT
    # Temperature is clamped into [temperature_min, temperature_max].
    temperature_min: float = 0.0
    temperature_max: float = 2.0
    # Hard ceiling for max_tokens (thinking + answer); None = not published.
    max_tokens_cap: int | None = None
    # Can be asked to think at all.
    can_think: bool = True
    # Thinks when a request says nothing, so "off" must be sent explicitly
    # (OpenRouter qwen3.8 thinks at xhigh unless told effort "none").
    thinking_default: bool = False
    # Takes NIM's top-level reasoning_budget while thinking.
    reasoning_budget: bool = False
    # Extra chat_template_kwargs sent with enable_thinking=true (NIM).
    thinking_template_kwargs: tuple[tuple[str, Any], ...] = ()
    # The model card's sampling; None = the call's temperature / no top_p.
    default_temperature: float | None = None
    default_top_p: float | None = None
    # OpenRouter efforts the endpoint lists (besides "none"); () = any.
    efforts: tuple[str, ...] = ()
    # Ollama think values the model accepts (Colab only).
    think_levels: tuple[str, ...] = ()
    # Ways the model can be held to JSON; "prompt" means asking in words.
    json_modes: frozenset[str] = frozenset({PROMPT})
    # Accepts OpenAI tool definitions.
    tools: bool = True
    # Width of the vectors the embedding endpoint returns.
    native_dim: int | None = None
    # build.nvidia.com page slug for the daily end-of-life check (FLUX is
    # not in /v1/models, so its page is the only signal).
    page_slug: str = ""
    # Colab: the Ollama manifest digest the session must have pulled
    # (/api/tags); neither Ollama nor its registry fetches by digest, so it
    # is checked after pulling the tag. The chat model's must match for the
    # session to be READY; a router with another one is only skipped for
    # that session (ADR-0010).
    digest: str = ""
    # False for the stand-in :func:`generic_profile` returns.
    known: bool = True

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


_NIM_CHAT = dict(
    temperature_min=NIM_MIN_TEMPERATURE,
    temperature_max=1.0,
    max_tokens_cap=32_768,
    # Every NIM chat model thinks by default, so
    # chat_template_kwargs.enable_thinking is always sent explicitly.
    thinking_default=True,
)
# Nemotron 3 model cards: "temperature=1.0 and top_p=0.95 across all tasks".
_NEMOTRON3_SAMPLING = dict(default_temperature=1.0, default_top_p=0.95)

_PROFILES: tuple[ModelProfile, ...] = (
    # ── NVIDIA NIM (integrate.api.nvidia.com) ──
    # Live gate 2026-09-30: a reasoning_budget (top level or inside
    # chat_template_kwargs) is HTTP 400 "thinking_token_budget is not yet
    # supported by the V2 model runner"; thinking runs at medium_effort and
    # max_tokens caps its length. Text only, and no structured output (only
    # Super takes json_schema), so a router tier on it asks for JSON in the
    # prompt and the orchestrator's tolerant parser reads the reply.
    ModelProfile(
        "nvidia",
        "nvidia/nemotron-3-ultra-550b-a55b",
        NIM_NEMOTRON3,
        thinking_template_kwargs=(("medium_effort", True),),
        **_NEMOTRON3_SAMPLING,
        **_NIM_CHAT,
    ),
    # The only NIM chat model with structured output: response_format
    # json_schema works, nvext.guided_json is rejected (HTTP 400, live gate).
    ModelProfile(
        "nvidia",
        "nvidia/nemotron-3-super-120b-a12b",
        NIM_NEMOTRON3,
        reasoning_budget=True,
        json_modes=frozenset({JSON_SCHEMA, PROMPT}),
        **_NEMOTRON3_SAMPLING,
        **_NIM_CHAT,
    ),
    # Takes OpenAI image_url data URIs and answers in Traditional Chinese.
    # It reasons by default (a little even at reasoning_effort "none");
    # temperature (0, 1], default 0.6, max_tokens up to 16384. Tool calling
    # is unverified.
    ModelProfile(
        "nvidia",
        "meta/muse-glimmer-30b",
        NIM_REASONING_EFFORT,
        modalities=TEXT_IMAGE,
        temperature_min=NIM_MIN_TEMPERATURE,
        temperature_max=1.0,
        max_tokens_cap=16_384,
        thinking_default=True,
        default_temperature=0.6,
        tools=False,
    ),
    # Nemotron 3 Nano Omni: image_url data URIs, answers in Traditional
    # Chinese. Vision probe 2026-10-01 (the bot's vision prompt, thinking
    # off, temperature 0.2, max_tokens 1024, no top_p): 1.8-6.3 s; a busy
    # worker answers HTTP 503 "ResourceExhausted" in under a second (a
    # TransientError, so the chain moves on at once). Tool calling is
    # unverified.
    ModelProfile(
        "nvidia",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        NIM_NEMOTRON3,
        modalities=TEXT_IMAGE,
        reasoning_budget=True,
        tools=False,
        **_NIM_CHAT,
    ),
    # Always 2048 wide; the client keeps a prefix (EMBEDDING_DIMENSIONS).
    ModelProfile("nvidia", "nvidia/nemotron-3-embed-1b", NIM_EMBED, can_think=False, tools=False, native_dim=2048),
    # Visual GenAI (ai.api.nvidia.com). klein: steps 1-4, cfg_scale pinned
    # at 0, Apache-2.0. flux.1-dev: mode base, cfg (1, 9], non-commercial.
    ModelProfile(
        "nvidia",
        "black-forest-labs/flux.2-klein-4b",
        NIM_FLUX2_KLEIN,
        can_think=False,
        tools=False,
        page_slug="black-forest-labs/flux_2-klein-4b",
    ),
    ModelProfile(
        "nvidia",
        "black-forest-labs/flux.1-dev",
        NIM_FLUX1_DEV,
        can_think=False,
        tools=False,
        page_slug="black-forest-labs/flux_1-dev",
    ),
    # ── OpenRouter free ids (per-endpoint facts from /models/{id}/endpoints) ──
    # Not served by NVIDIA, so they survive an NIM outage.
    ModelProfile(
        "openrouter",
        "qwen/qwen3.8-27b:free",
        OPENROUTER,
        modalities=TEXT_IMAGE_VIDEO,
        thinking_default=True,  # at xhigh
        efforts=("low", "medium", "xhigh"),
        json_modes=frozenset({JSON_SCHEMA, PROMPT}),
    ),
    # Served by NVIDIA itself: they fail together with NIM. Ultra :free is
    # text only and takes no response_format.
    ModelProfile(
        "openrouter",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        OPENROUTER,
        thinking_default=True,
        efforts=("medium", "high"),
    ),
    ModelProfile(
        "openrouter",
        "nvidia/nemotron-3-super-120b-a12b:free",
        OPENROUTER,
        thinking_default=True,
        efforts=("low", "medium"),
        json_modes=frozenset({JSON_SCHEMA, JSON_OBJECT, PROMPT}),
    ),
    # Picks a free model at random, some of which reason by default.
    ModelProfile("openrouter", "openrouter/free", OPENROUTER, modalities=TEXT_IMAGE, can_think=False, thinking_default=True),
    # ── Colab Ollama 0.35 ──
    # The -mtp tag has qwen3.8:27b's weights with multi-token prediction on
    # and a single-platform digest, so the pin below is stable.
    ModelProfile(
        "colab",
        "qwen3.8:27b-mtp-q4_K_M",
        OLLAMA_CHAT,
        modalities=TEXT_IMAGE,
        thinking_default=True,  # "medium" when think is omitted
        think_levels=THINK_LEVELS,
        json_modes=frozenset({JSON_SCHEMA, PROMPT}),
        digest="22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643",
    ),
    # The router (ADR-0010): nimble:9b = 9b-q8_0 = latest, the 2026-09-30
    # manifest (Q8_0, 9,527,501,312-byte GGUF, Nimble's own SYSTEM layer,
    # {"num_ctx":8194}, Apache-2.0). Bespoke republishes it often: re-pin
    # here and in colab/versions.json together.
    ModelProfile(
        "colab",
        "nimble:9b",
        SYSTEMONE,
        can_think=False,
        tools=False,
        json_modes=frozenset(),
        digest="24e550a16a7081881be2f1f0d91e8cc13a597472735c04119f035a0a85c67e0c",
    ),
)

PROFILES: dict[tuple[str, str], ModelProfile] = {(p.provider, p.model): p for p in _PROFILES}


def get_profile(provider: str, model: str) -> ModelProfile | None:
    """The profile of exactly this id, or None."""
    return PROFILES.get((provider, model))


def generic_profile(provider: str, model: str, kind: str = CHAT) -> ModelProfile | None:
    """A permissive stand-in for an id without a profile.

    None when *provider* cannot serve *kind* at all (only NVIDIA embeds and
    draws) or serves only profiled ids (Colab, whose VM pulls only the
    pinned models). Nothing is assumed that would refuse a request: every
    input type, JSON mode and effort is allowed, and "off" is always sent.
    """
    if kind == CHAT:
        if provider == "nvidia":
            return ModelProfile(
                provider,
                model,
                GENERIC,
                modalities=TEXT_IMAGE,
                temperature_min=NIM_MIN_TEMPERATURE,
                can_think=False,
                json_modes=JSON_MODES,
                known=False,
            )
        if provider == "openrouter":
            return ModelProfile(
                provider,
                model,
                OPENROUTER,
                modalities=TEXT_IMAGE,
                thinking_default=True,
                json_modes=JSON_MODES,
                known=False,
            )
        return None
    if provider == "nvidia" and kind in (EMBED, IMAGE):
        return ModelProfile(provider, model, GENERIC, can_think=False, tools=False, known=False)
    return None


def clamp_temperature(profile: ModelProfile, temperature: float) -> float:
    return max(profile.temperature_min, min(temperature, profile.temperature_max))


def cap_max_tokens(profile: ModelProfile, max_tokens: int) -> int:
    cap = profile.max_tokens_cap
    return max_tokens if cap is None else min(cap, max_tokens)


# ── Tier options shared by the OpenAI-compatible payload builders ──


def tier_sampling(
    options: Mapping[str, Any],
    *,
    temperature: float,
    max_tokens: int,
    profile: ModelProfile | None = None,
) -> tuple[float, int]:
    """``(temperature, max_tokens)`` for a call.

    A tier's own values win. Otherwise the temperature is the model card's
    (``profile.default_temperature``) when the profile has one, and the
    call's when it does not.
    """
    if options.get("max_tokens"):
        max_tokens = int(options["max_tokens"])
    if options.get("temperature") is not None:
        temperature = float(options["temperature"])
    elif profile is not None and profile.default_temperature is not None:
        temperature = profile.default_temperature
    return temperature, max_tokens


def response_format_for(json_mode: str | None, json_schema: dict | None) -> dict | None:
    """The ``response_format`` a tier's *json_mode* asks for, or None.

    Only a call that has a schema is held to JSON; "prompt" (and no mode)
    leaves it to the prompt's own words.
    """
    if json_schema is None:
        return None
    if json_mode == JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {"name": "reply", "strict": True, "schema": json_schema},
        }
    if json_mode == JSON_OBJECT:
        return {"type": "json_object"}
    return None
