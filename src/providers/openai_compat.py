"""One JSON POST with the upstream's HTTP status mapped onto provider errors.

NVIDIA NIM and OpenRouter speak the same OpenAI-compatible contract, and
NVIDIA's Visual GenAI endpoints share its status codes, so every provider
request goes through :func:`post_chat_completion`. It returns the decoded
JSON body of a 200 and raises one :mod:`src.providers.base` error for
anything else. NVIDIA Cloud Functions may answer 202 with an ``NVCF-REQID``
header for a slow job; the result is then polled from the status URL until
*timeout*. A 404 from that status URL means it does not know the job, which
says nothing about the model, so it is transient, never a ``ModelGoneError``.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

from src.providers.base import (
    AuthError,
    CreditsError,
    ModelGoneError,
    PayloadError,
    ProviderError,
    RateLimitError,
    TransientError,
)

NVCF_STATUS_URL = "https://integrate.api.nvidia.com/v1/status/{request_id}"
NVCF_POLL_INTERVAL_SECONDS = 1.0

# 410 bodies read "... has reached its end of life on 2026-07-27T00:00:00Z ...".
_EOL_RE = re.compile(r"end of life on (\S+Z)")
# OpenRouter's 404 when no endpoint of the model can take this request ("No
# endpoints found that support tool use", "... that support image input"):
# the request's problem, not the model's ("No endpoints found for <id>").
_NO_ENDPOINT_FOR_REQUEST = "no endpoints found that"
_DETAIL_LIMIT = 500


async def post_chat_completion(
    client: Any,
    url: str,
    payload: dict,
    *,
    provider: str,
    model: str,
    timeout: float,
    poll_interval: float | None = None,
) -> dict:
    """POST *payload* to *url* and return the JSON body of the final 200.

    *timeout* bounds the whole exchange, 202 polling included; a 202 is
    polled every *poll_interval* seconds (default ``NVCF_POLL_INTERVAL_SECONDS``).

    Raises:
        RateLimitError, PayloadError, AuthError, CreditsError,
        ModelGoneError, TransientError: see :mod:`src.providers.base`.
    """
    deadline = time.monotonic() + timeout
    if poll_interval is None:
        poll_interval = NVCF_POLL_INTERVAL_SECONDS
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        if response.status_code == 202:
            response = await _poll_nvcf(
                client,
                response,
                provider=provider,
                model=model,
                deadline=deadline,
                poll_interval=poll_interval,
            )
    except httpx.TimeoutException as exc:
        raise TransientError(
            model, 0, f"timed out ({str(exc) or type(exc).__name__})", provider
        ) from exc
    except httpx.HTTPError as exc:
        raise TransientError(model, 0, str(exc) or type(exc).__name__, provider) from exc

    if response.status_code != 200:
        raise error_for_status(response, provider=provider, model=model)
    try:
        data = response.json()
    except ValueError as exc:
        raise TransientError(model, 200, "invalid JSON body", provider) from exc
    if not isinstance(data, dict):
        raise TransientError(model, 200, "JSON body is not an object", provider)
    return data


async def _poll_nvcf(
    client: Any,
    response: Any,
    *,
    provider: str,
    model: str,
    deadline: float,
    poll_interval: float,
) -> Any:
    """Poll a pending NVCF job until it stops answering 202 or time runs out."""
    request_id = response.headers.get("NVCF-REQID")
    if not request_id:
        raise TransientError(model, 202, "HTTP 202 without an NVCF-REQID header", provider)
    status_url = NVCF_STATUS_URL.format(request_id=request_id)
    while response.status_code == 202:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransientError(
                model, 202, f"NVCF job {request_id} still pending at the timeout", provider
            )
        await asyncio.sleep(min(poll_interval, remaining))
        response = await client.get(status_url, timeout=max(1.0, deadline - time.monotonic()))
    if response.status_code == 404:
        raise TransientError(model, 404, f"NVCF status for job {request_id} not found", provider)
    return response


def error_for_status(response: Any, *, provider: str, model: str) -> ProviderError:
    """The provider error a non-200 HTTP answer stands for."""
    status = response.status_code
    detail = (response.text or "")[:_DETAIL_LIMIT]
    if status == 429:
        return RateLimitError(
            model,
            parse_retry_after(response.headers.get("retry-after")),
            provider=provider,
            detail=detail or "rate limited",
        )
    if status in (401, 403):
        return AuthError(model, status, detail, provider)
    if status == 402:
        return CreditsError(model, status, detail, provider)
    if status == 404 and _NO_ENDPOINT_FOR_REQUEST in detail.lower():
        return PayloadError(model, status, detail, provider)
    if status in (404, 410):
        match = _EOL_RE.search(detail)
        return ModelGoneError(model, status, detail, provider, eol=match.group(1) if match else None)
    if 400 <= status < 500 and status != 408:
        return PayloadError(model, status, detail, provider)
    # 5xx (503 ResourceExhausted, Cloudflare 524, ...), 408 and anything odd.
    return TransientError(model, status, detail, provider)


def parse_retry_after(value: str | None) -> int | None:
    """Seconds from a Retry-After header; None when absent or not numeric."""
    if not value:
        return None
    try:
        return max(1, int(float(value)))
    except ValueError:
        # An HTTP-date is legal but rare here; the default cooldown applies.
        return None
