"""OpenRouter API provider using the OpenAI-compatible chat completions endpoint.

Only free model ids (``*:free`` and the ``openrouter/free`` router) are
called unless the deployment opts into paid ids: a stray paid id would
otherwise bill account credits without anyone noticing.

Reasoning and JSON mode follow the model's profile
(:mod:`src.providers.model_profiles`) and the tier options: thinking on
sends ``reasoning.effort`` (the tier's effort, else medium); thinking off
sends ``effort: "none"`` to models that reason by default (qwen3.8 would
think at xhigh otherwise) and nothing to the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from src.providers.base import (
    PayloadError,
    ProviderResponse,
    SkipTierError,
    parse_openai_response,
    require_output,
)
from src.providers.messages import normalize_messages
from src.providers.model_profiles import (
    DEFAULT_THINKING_BUDGET,
    clamp_temperature,
    generic_profile,
    get_profile,
    response_format_for,
    tier_sampling,
)
from src.providers.openai_compat import post_chat_completion
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.services.model_health import ModelHealth

# The free auto-router is billed like a ":free" id.
FREE_ROUTER_MODEL = "openrouter/free"
# Upper bound for one request; the fallback chain's per-tier limit is
# normally much shorter.
_REQUEST_TIMEOUT_SECONDS = 120.0


# Effort a thinking tier gets when it names none.
_DEFAULT_EFFORT = "medium"


def is_free_model_id(model: str) -> bool:
    """True for OpenRouter ids that never bill credits."""
    return model.endswith(":free") or model == FREE_ROUTER_MODEL


def build_chat_payload(
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    options: Mapping[str, Any] | None = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    json_schema: dict | None = None,
) -> dict:
    """The chat completions body for *model* with these tier *options*.

    Reasoning tokens count against max_tokens on OpenRouter, so thinking
    adds the budget (``options["reasoning_budget"]`` or *thinking_budget*).
    Sampling follows :func:`~src.providers.model_profiles.tier_sampling`
    (tier, then the profile's default, then the call).
    ``json_schema`` becomes a ``response_format`` only when the tier's
    ``json_mode`` asks for one (json_schema or json_object).
    """
    options = options or {}
    profile = get_profile("openrouter", model) or generic_profile("openrouter", model)
    assert profile is not None  # OpenRouter always has a generic chat protocol
    temperature, max_tokens = tier_sampling(
        options, temperature=temperature, max_tokens=max_tokens, profile=profile
    )

    payload: dict = {"model": model, "messages": messages}
    if thinking and profile.can_think:
        effort = options.get("effort")
        payload["reasoning"] = {"effort": effort if effort not in (None, "none") else _DEFAULT_EFFORT}
        max_tokens += int(options.get("reasoning_budget") or thinking_budget)
    elif profile.thinking_default:
        # Omitting the object is not "off" for these: say so.
        payload["reasoning"] = {"effort": "none"}
    payload["max_tokens"] = max_tokens
    payload["temperature"] = clamp_temperature(profile, temperature)
    if profile.default_top_p is not None:
        payload["top_p"] = profile.default_top_p

    if tools and profile.tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    response_format = response_format_for(options.get("json_mode"), json_schema)
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


class OpenRouterProvider:
    """Async client for the OpenRouter chat completions API."""

    name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        health: ModelHealth,
        *,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        allow_paid: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.health = health
        self._thinking_budget = thinking_budget
        self._allow_paid = allow_paid
        self._blocked_paid_models: set[str] = set()
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/linebot-cloud-agent",
            "X-OpenRouter-Title": "LineBot-CloudAgent",
        }
        # ``transport`` lets scripts/live_gate.py carry the exact requests
        # this class builds; production always uses httpx's default.
        self._client = httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT_SECONDS, headers=self._headers, transport=transport
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        disable_thinking: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        json_schema: dict | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> ProviderResponse:
        """Call OpenRouter and return a unified response.

        Args:
            model: Model ID (e.g. "qwen/qwen3.8-27b:free")
            messages: OpenAI-format message list
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens
            json_schema: Sent in the tier's ``json_mode`` (see
                :func:`build_chat_payload`); without one the prompt alone
                asks for JSON.
            options: The tier's options; the chain has already folded
                ``options["thinking"]`` into ``disable_thinking``.

        Raises:
            SkipTierError: Paid model id without ``allow_paid`` (no request
                is sent).
            ProviderError: A subclass per failure, see src/providers/base.py;
                EmptyCompletionError when a 200 carries nothing usable.
        """
        model = model.strip()
        if not model:
            raise PayloadError("<empty>", 400, "Model ID must not be empty", self.name)

        if not self._allow_paid and not is_free_model_id(model):
            if model not in self._blocked_paid_models:
                self._blocked_paid_models.add(model)
                logger.warning(
                    f"OpenRouter: refusing paid model {model}; use a ':free' id "
                    "or set OPENROUTER_ALLOW_PAID=true"
                )
            raise SkipTierError(
                model,
                "paid OpenRouter model blocked (OPENROUTER_ALLOW_PAID=false)",
                self.name,
            )

        payload = build_chat_payload(
            model,
            normalize_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=not disable_thinking,
            options=options,
            thinking_budget=self._thinking_budget,
            tools=tools,
            tool_choice=tool_choice,
            json_schema=json_schema,
        )
        reasoning = payload.get("reasoning", {}).get("effort", "default")

        logger.info(
            f"OpenRouter request → {model} "
            f"(temp={payload['temperature']}, max_tok={payload['max_tokens']}, reasoning={reasoning})"
        )
        self.health.record_request(self.name, model)
        data = await post_chat_completion(
            self._client,
            self.BASE_URL,
            payload,
            provider=self.name,
            model=model,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response = parse_openai_response(data, model=model, provider=self.name)
        if response.reasoning_content:
            logger.info(f"OpenRouter reasoning ← {model} ({len(response.reasoning_content)} chars)")
        usage = response.usage or {}
        logger.info(
            f"OpenRouter response ← {model} "
            f"(text={bool(response.text)}, finish={response.finish_reason}, "
            f"prompt_tokens={usage.get('prompt_tokens')}, completion_tokens={usage.get('completion_tokens')}, "
            f"reasoning_tokens={response.reasoning_tokens})"
        )
        return require_output(response)
