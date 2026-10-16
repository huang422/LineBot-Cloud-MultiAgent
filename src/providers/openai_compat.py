"""One JSON POST with the upstream's HTTP status mapped onto provider errors.

NVIDIA NIM and OpenRouter speak the same OpenAI-compatible contract, and
NVIDIA's Visual GenAI endpoints share its status codes, so every provider
request goes through :func:`post_chat_completion`. It returns the decoded
JSON body of a 200 and raises one :mod:`src.providers.base` error for
anything else. NVIDIA Cloud Functions may answer 202 with an ``NVCF-REQID``
header for a slow job; the result is then polled from the status URL until
*timeout*. A 404 from that status URL means it does not know the job, which
says nothing about the model, so it is transient, never a ``ModelGoneError``.

A 200 is not always an answer: once OpenRouter has routed a request it can
report an upstream failure only in the body, as a top-level ``error``
object whose ``code`` is the HTTP status it stands for, alone or beside
partial output with ``finish_reason`` "error" (OpenRouter's errors doc), or
as the same object on the choice itself.
On 2026-10-16 Ultra ``:free`` answered so (code 503, "Upstream error from
Nvidia: Service temporarily overloaded"); read as an empty completion it
earned no strike and no cooldown, so every reply tried it again. Such a
body raises the error its code maps to, as that status itself would
(:func:`error_for_body`); ``"error": null`` is no error.
"""

from __future__ import annotations

import asyncio
import json
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
    A 200 whose body holds an ``error`` object raises like a non-200.

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
    if isinstance(data.get("error"), dict):
        # Even beside choices: partial output cut by an error is no answer.
        raise error_for_body(data["error"], response, provider=provider, model=model)
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if isinstance(choice, dict):
        # OpenRouter's schema also lets the choice itself carry the error
        # (NonStreamingChoice.error) beside its partial text; the first
        # choice is the one read as the answer.
        if isinstance(choice.get("error"), dict):
            raise error_for_body(choice["error"], response, provider=provider, model=model)
        if choice.get("finish_reason") == "error":
            raise TransientError(model, 200, 'finish_reason "error" without an error object', provider)
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
    return _error_for(
        response.status_code,
        (response.text or "")[:_DETAIL_LIMIT],
        response.headers.get("retry-after"),
        provider=provider,
        model=model,
    )


def error_for_body(error: dict, response: Any, *, provider: str, model: str) -> ProviderError:
    """The provider error a 200 whose body is an ``error`` object stands for.

    ``error["code"]`` is mapped as that HTTP status would be; a code that
    is missing, not a number or not an error status (400-599) is transient
    with status 200. The detail is the object as JSON, its ``message``,
    ``code``, ``metadata.error_type`` and ``metadata.raw`` first: ModelHealth
    tells OpenRouter's account cap (``free-models-per-day``) from one
    upstream's 429 (``rate-limited upstream``) by those words, and
    OpenRouterProvider retries an unavailable upstream by them, so they
    must survive the cut to 500 characters whatever else the object carries.
    """
    detail = json.dumps({"error": _lead_with_reason(error)}, ensure_ascii=False)[:_DETAIL_LIMIT]
    status = _error_status(error.get("code"))
    if status is None:
        return TransientError(model, 200, detail, provider)
    return _error_for(
        status, detail, response.headers.get("retry-after"), provider=provider, model=model
    )


def _error_status(code: Any) -> int | None:
    """*code* as an HTTP error status, or None."""
    if isinstance(code, bool):
        return None
    if isinstance(code, str):
        # isdigit() would pass "²", which int() rejects; a ValueError here
        # would escape the ProviderError contract and earn no strike.
        if not code.strip().isdecimal():
            return None
        try:
            code = int(code)
        except ValueError:  # past int()'s digit limit
            return None
    if isinstance(code, int) and 400 <= code <= 599:
        return code
    return None


def _lead_with_reason(error: dict) -> dict:
    """*error* with ``message``, ``code``, ``metadata.error_type`` and
    ``metadata.raw`` in front."""
    lead = {key: error[key] for key in ("message", "code") if key in error}
    metadata = error.get("metadata")
    if isinstance(metadata, dict):
        first = {key: metadata[key] for key in ("error_type", "raw") if key in metadata}
        lead["metadata"] = {**first, **metadata}
    return {**lead, **{key: value for key, value in error.items() if key not in lead}}


def _error_for(
    status: int, detail: str, retry_after: str | None, *, provider: str, model: str
) -> ProviderError:
    """The provider error *status* stands for, with *detail* (already cut)."""
    if status == 429:
        return RateLimitError(
            model,
            parse_retry_after(retry_after),
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
