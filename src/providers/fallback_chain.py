"""Rate-limit-aware fallback chain across multiple providers and models.

Tries (provider, model) targets in order; on 429 or unavailability,
falls through to the next target.  Supports mixed providers (NVIDIA +
OpenRouter) in a single chain.
"""

from __future__ import annotations

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
    ) -> ProviderResponse: ...


# Target = (provider_instance, model_id)
Target = tuple[LLMProvider, str]


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
        **kwargs,
    ) -> ProviderResponse:
        """Try targets in order, falling back on rate limits or errors.

        Args:
            targets: Ordered list of (provider, model_id) to try
            messages: OpenAI-format messages
            **kwargs: Forwarded to provider.generate()

        Returns:
            ProviderResponse from the first successful target

        Raises:
            AllModelsRateLimitedError: Every target is unavailable due to rate limiting
            AllProvidersFailedError: Targets failed for other provider/service reasons
        """
        if not targets:
            raise AllProvidersFailedError("No fallback targets configured")

        last_error: Exception | None = None
        attempted = 0
        saw_provider_error = False

        for provider, model in targets:
            if not self.rate_tracker.is_available(model):
                logger.info(f"Skipping {model} (rate limited)")
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
                logger.warning(f"Fallback: {model} rate limited, trying next")
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
