"""Registry of model metadata used for local documentation/reference.

Updated: 2026-03-20. The defaults in this project still point to free-tier-
friendly models, but configured model IDs are no longer hard-enforced against
this catalog at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    name: str
    params: str  # e.g. "405B", "27B", "196B MoE"
    context: int  # tokens
    vision: bool = False
    image_gen: bool = False
    tool_calling: bool = False
    reasoning: bool = False


# ── Complete free model catalog ──────────────────────────────

MODELS: dict[str, ModelInfo] = {}


def _register(*models: ModelInfo) -> None:
    for m in models:
        MODELS[m.model_id] = m


_register(
    # --- NVIDIA hosted models ---
    ModelInfo("qwen/qwen3.5-397b-a17b", "Qwen3.5 397B VLM", "397B MoE (17B active)", 262_144, vision=True, reasoning=True),

    # --- Largest text models (OpenRouter free) ---
    ModelInfo("qwen/qwen3-coder:free", "Qwen3 Coder 480B", "480B", 262_144, tool_calling=True),
    ModelInfo("nousresearch/hermes-3-llama-3.1-405b:free", "Hermes 3 405B", "405B", 131_072),
    ModelInfo("stepfun/step-3.5-flash:free", "Step 3.5 Flash", "196B MoE", 256_000, tool_calling=True, reasoning=True),
    ModelInfo("nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super 120B", "120B MoE", 262_144, tool_calling=True, reasoning=True),
    ModelInfo("openai/gpt-oss-120b:free", "GPT-OSS 120B", "120B MoE", 131_072, tool_calling=True),
    ModelInfo("qwen/qwen3-next-80b-a3b-instruct:free", "Qwen3 Next 80B", "80B", 262_144, tool_calling=True),
    ModelInfo("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B", "70B", 66_000, tool_calling=True),
    ModelInfo("nvidia/nemotron-3-nano-30b-a3b:free", "Nemotron 3 Nano 30B", "30B", 256_000, tool_calling=True),
    ModelInfo("minimax/minimax-m2.5:free", "MiniMax M2.5", "?", 197_000, tool_calling=True),
    ModelInfo("arcee-ai/trinity-large-preview:free", "Trinity Large", "400B MoE", 131_072, tool_calling=True, reasoning=True),
    ModelInfo("openai/gpt-oss-20b:free", "GPT-OSS 20B", "20B", 131_072, tool_calling=True),
    ModelInfo("z-ai/glm-4.5-air:free", "GLM 4.5 Air", "?", 131_072, tool_calling=True),
    ModelInfo("arcee-ai/trinity-mini:free", "Trinity Mini", "?", 131_072, tool_calling=True),
    ModelInfo("xiaomi/mimo-v2-flash:free", "MiMo v2 Flash", "?", 262_144),
    ModelInfo("qwen/qwen3-4b:free", "Qwen3 4B", "4B", 41_000, tool_calling=True),
    ModelInfo("meta-llama/llama-3.2-3b-instruct:free", "Llama 3.2 3B", "3B", 131_072),
    ModelInfo("liquid/lfm-2.5-1.2b-thinking:free", "LFM 2.5 Thinking", "1.2B", 33_000, reasoning=True),
    ModelInfo("liquid/lfm-2.5-1.2b-instruct:free", "LFM 2.5 Instruct", "1.2B", 33_000),
    ModelInfo("cognitivecomputations/dolphin-mistral-24b-venice-edition:free", "Uncensored", "24B", 33_000),
    ModelInfo("google/gemma-3n-e2b-it:free", "Gemma 3n 2B", "2B", 8_000),
    ModelInfo("google/gemma-3n-e4b-it:free", "Gemma 3n 4B", "4B", 8_000),

    # --- Vision models ---
    ModelInfo("google/gemma-3-27b-it:free", "Gemma 3 27B", "27B", 131_072, vision=True),
    ModelInfo("google/gemma-3-12b-it:free", "Gemma 3 12B", "12B", 33_000, vision=True),
    ModelInfo("google/gemma-3-4b-it:free", "Gemma 3 4B", "4B", 33_000, vision=True),
    ModelInfo("nvidia/nemotron-nano-12b-v2-vl:free", "Nemotron Nano VL", "12B", 128_000, vision=True, tool_calling=True),
    ModelInfo("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1", "24B", 128_000, vision=True, tool_calling=True),

    # --- Image generation ---
    ModelInfo("sourceful/riverflow-v2-pro", "Riverflow v2 Pro", "?", 32_000, image_gen=True),
    ModelInfo("google/gemini-2.5-flash-image-preview:free", "Gemini 2.5 Flash Image", "?", 32_000, vision=True, image_gen=True),

    # --- Auto router ---
    ModelInfo("openrouter/free", "Free Router", "auto", 200_000, vision=True, tool_calling=True),
)


def get_vision_models() -> list[str]:
    """Return model IDs that support image input."""
    return [m.model_id for m in MODELS.values() if m.vision]


def get_image_gen_models() -> list[str]:
    """Return model IDs that support image generation."""
    return [m.model_id for m in MODELS.values() if m.image_gen]


def get_model_info(model_id: str) -> ModelInfo | None:
    """Return model metadata when the model is known to the registry."""
    return MODELS.get(model_id)


def is_known_nvidia_model(model_id: str) -> bool:
    """Return True for direct NVIDIA-hosted models tracked in the registry."""
    info = get_model_info(model_id)
    return bool(info and not model_id.endswith(":free"))


def is_free_openrouter_model(model_id: str) -> bool:
    """Return True only for registered free OpenRouter model IDs."""
    if model_id == "openrouter/free":
        return model_id in MODELS
    info = get_model_info(model_id)
    return bool(info and model_id.endswith(":free"))


def is_image_generation_model(model_id: str) -> bool:
    """Return True for registered image-generation models."""
    info = get_model_info(model_id)
    return bool(info and info.image_gen)


def supports_vision(model_id: str) -> bool:
    info = MODELS.get(model_id)
    return info.vision if info else False


def supports_reasoning(model_id: str) -> bool:
    info = MODELS.get(model_id)
    return info.reasoning if info else False
