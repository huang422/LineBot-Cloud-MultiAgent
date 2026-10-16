"""OpenRouter API provider using the OpenAI-compatible chat completions endpoint.

Only free model ids (``*:free`` and the ``openrouter/free`` router) are
called unless the deployment opts into paid ids: a stray paid id would
otherwise bill account credits without anyone noticing.

Reasoning and JSON mode follow the model's profile
(:mod:`src.providers.model_profiles`) and the tier options: thinking on
sends ``reasoning.effort`` (the tier's effort, else medium); thinking off
sends ``effort: "none"`` to models that reason by default (dots-3 and the
Nemotron ``:free`` ids would reason otherwise) and nothing to the rest.

A 502 or 503 that says the model's upstream was unavailable or overloaded
(:func:`is_upstream_unavailable`), as an HTTP status or as the ``code`` of
an error in a 200 body, is sent once more after
UPSTREAM_RETRY_PAUSE_SECONDS, if the model may still take a request
(:class:`ModelHealth`: not cooling down, within the free budget, which the
retry counts against too). Nothing else is retried here: 4xx, 429,
timeouts and a blocked paid id go straight back to the fallback chain,
whose tier limit also bounds the retry (it is cancelled there like any
slow call).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from src.providers.base import (
    PayloadError,
    ProviderError,
    ProviderResponse,
    SkipTierError,
    TransientError,
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

# Seconds before the one retry of an unavailable upstream. dots-3 has one
# upstream (AtlasCloud); on 2026-10-16 it answered a 200 whose error object
# was {"message": "JSON error injected into SSE stream", "code": 502,
# "metadata": {"error_type": "provider_unavailable"}} to a memory summary
# after 15 s and once in a live gate run, where the retries a few seconds
# later passed.
UPSTREAM_RETRY_PAUSE_SECONDS = 1.0
# OpenRouter's metadata.error_type ("provider_unavailable",
# "provider_overloaded") and the upstream's words ("Upstream error from
# Nvidia: Service temporarily overloaded").
_UPSTREAM_UNAVAILABLE_RE = re.compile(
    r"provider[_ ]unavailable|overloaded|temporarily unavailable", re.IGNORECASE
)


def is_free_model_id(model: str) -> bool:
    """True for OpenRouter ids that never bill credits."""
    return model.endswith(":free") or model == FREE_ROUTER_MODEL


def is_upstream_unavailable(error: ProviderError) -> bool:
    """A 502 or 503 whose error says the model's upstream was unavailable or
    overloaded: worth one more try on the same model."""
    return (
        isinstance(error, TransientError)
        and error.status in (502, 503)
        and bool(_UPSTREAM_UNAVAILABLE_RE.search(error.detail or ""))
    )


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
            model: Model ID (e.g. "dots-studio/dots-3-note-preview:free")
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
                EmptyCompletionError when a 200 carries nothing usable. An
                unavailable upstream raises only when its one retry fails
                too (the retry's error).
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
        data = await self._post(model, payload)
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

    async def _post(self, model: str, payload: dict) -> dict:
        """POST *payload*, and once more when the upstream was unavailable
        (see the module docstring); each POST counts against the budget."""
        self.health.record_request(self.name, model)
        try:
            return await self._post_once(model, payload)
        except TransientError as exc:
            if not is_upstream_unavailable(exc):
                raise
            if self.health.unavailable(self.name, model) is not None:
                raise  # cooling down, or the budget is spent
            logger.warning(
                f"OpenRouter: {model} answered {exc.status} (upstream unavailable); "
                f"retrying once in {UPSTREAM_RETRY_PAUSE_SECONDS:g}s"
            )
        await asyncio.sleep(UPSTREAM_RETRY_PAUSE_SECONDS)
        self.health.record_request(self.name, model)
        return await self._post_once(model, payload)

    async def _post_once(self, model: str, payload: dict) -> dict:
        return await post_chat_completion(
            self._client,
            self.BASE_URL,
            payload,
            provider=self.name,
            model=model,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
