"""Rate-limit-aware fallback chain across multiple providers and models.

Tries (provider, model) targets in order; on 429 or unavailability,
falls through to the next target.  Supports mixed providers (NVIDIA +
OpenRouter) in a single chain.

When *thinking_timeout* is set, the first attempt uses thinking/reasoning
mode.  If it doesn't complete within the timeout, the chain retries every
target with ``disable_thinking=True`` so models answer without the
(potentially slow) reasoning step.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from src.providers.openrouter_provider import (
    ProviderResponse,
    RateLimitError,
    ProviderError,
)
from src.utils.logger import logger
from src.utils.rate_tracker import RateTracker


class LLMProvider(Protocol):
    """Structural typing for any provider that has a generate() method."""

    async def generate(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        modalities: list[str] | None = None,
        require_reasoning_tokens: bool = False,
        disable_thinking: bool = False,
    ) -> ProviderResponse: ...


# Target = (provider_instance, model_id)
Target = tuple[LLMProvider, str]


def _rate_limit_model(provider: LLMProvider, model: str, *, disable_thinking: bool) -> str:
    resolver = getattr(provider, "resolve_model", None)
    if callable(resolver):
        resolved = resolver(model, disable_thinking=disable_thinking)
        if isinstance(resolved, str) and resolved.strip():
            return resolved.strip()
    return model


class AllModelsRateLimitedError(Exception):
    """Raised when every target in the chain is exhausted."""


class AllProvidersFailedError(Exception):
    """Raised when every target fails for reasons other than pure rate limiting."""

    def __init__(self, message: str, last_error: Exception | None = None) -> None:
        self.last_error = last_error
        super().__init__(message)


class FallbackChain:
    """Tries a list of (provider, model) targets with automatic fallback."""

    def __init__(self, rate_tracker: RateTracker) -> None:
        self.rate_tracker = rate_tracker
        self.fallback_count: int = 0

    async def generate(
        self,
        targets: list[Target],
        messages: list[dict],
        *,
        thinking_timeout: float | None = None,
        **kwargs,
    ) -> ProviderResponse:
        """Try targets in order, falling back on rate limits or errors.

        Args:
            targets: Ordered list of (provider, model_id) to try
            messages: OpenAI-format messages
            thinking_timeout: If set, abort the thinking-mode attempt after
                this many seconds and retry the entire chain with thinking
                disabled.
            **kwargs: Forwarded to provider.generate()

        Returns:
            ProviderResponse from the first successful target

        Raises:
            AllModelsRateLimitedError: Every target is unavailable due to rate limiting
            AllProvidersFailedError: Targets failed for other provider/service reasons
        """
        should_timeout = (
            thinking_timeout is not None
            and thinking_timeout > 0
            and not kwargs.get("disable_thinking", False)
        )
        if should_timeout:
            try:
                return await asyncio.wait_for(
                    self._run_chain(targets, messages, **kwargs),
                    timeout=thinking_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Thinking mode timed out after {thinking_timeout}s — "
                    "retrying with thinking disabled"
                )
                retry_kwargs = {**kwargs, "disable_thinking": True}
                return await self._run_chain(targets, messages, **retry_kwargs)

        return await self._run_chain(targets, messages, **kwargs)

    async def _run_chain(
        self,
        targets: list[Target],
        messages: list[dict],
        **kwargs,
    ) -> ProviderResponse:
        """Core chain logic: try each target in order."""
        if not targets:
            raise AllProvidersFailedError("No fallback targets configured")

        last_error: Exception | None = None
        attempted = 0
        saw_provider_error = False
        disable_thinking = kwargs.get("disable_thinking", False)

        for provider, model in targets:
            rate_limit_model = _rate_limit_model(
                provider,
                model,
                disable_thinking=disable_thinking,
            )
            if not self.rate_tracker.is_available(rate_limit_model):
                logger.info(f"Skipping {rate_limit_model} (rate limited)")
                continue

            try:
                attempted += 1
                result = await provider.generate(
                    model=model, messages=messages, **kwargs
                )
                if attempted > 1:
                    self.fallback_count += 1
                return result
            except RateLimitError as e:
                logger.warning(
                    f"Fallback: {getattr(e, 'model', rate_limit_model)} rate limited, trying next"
                )
                last_error = e
                continue
            except ProviderError as e:
                logger.warning(f"Fallback: {model} error {e.status}, trying next")
                saw_provider_error = True
                last_error = e
                continue

        if saw_provider_error:
            raise AllProvidersFailedError(
                f"All targets failed. Last error: {last_error}",
                last_error=last_error,
            )

        raise AllModelsRateLimitedError(
            f"All targets exhausted. Last error: {last_error}"
        )
