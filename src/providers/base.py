"""Contracts every LLM provider shares: the response, the errors, the parser.

Providers turn each upstream failure into one :class:`ProviderError`
subclass, so the fallback chain and :class:`~src.services.model_health.
ModelHealth` decide what happens next from the error class alone:

========================  ==========================  ==========================
Error                     Raised for                  Effect
========================  ==========================  ==========================
``RateLimitError``        HTTP 429                    cooldown, next tier
``PayloadError``          HTTP 400 / 422 / other      next tier, never resent
                          4xx, OpenRouter's 404 "No
                          endpoints found that ..."
``AuthError``             HTTP 401 / 403              provider cooldown (401, or
                                                      a 403 naming the key);
                                                      other 403: that model 2 min
``CreditsError``          HTTP 402                    provider cooldown
``ModelGoneError``        HTTP 404 / 410              model taken out
``TransientError``        5xx, 524, timeouts, bad     next tier
                          JSON, transport errors
``EmptyCompletionError``  200 with nothing usable     next tier
``ContentFilteredError``  200 whose image the         next image tier (the
                          model's safety filter       model stays healthy)
                          withheld
``SkipTierError``         tier cannot serve this      not attempted
                          request (busy, modality)
========================  ==========================  ==========================
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class ProviderError(Exception):
    """A provider call failed; the subclass says how."""

    def __init__(self, model: str, status: int, detail: str, provider: str = "") -> None:
        self.model = model
        self.status = status
        self.detail = detail
        self.provider = provider
        label = f"{provider} error" if provider else "Provider error"
        super().__init__(f"{label} {status} for {model}: {detail}")


class RateLimitError(ProviderError):
    """HTTP 429. ``retry_after`` is the server's Retry-After in seconds, if any."""

    def __init__(
        self,
        model: str,
        retry_after: int | None = None,
        *,
        provider: str = "",
        detail: str = "rate limited",
    ) -> None:
        self.retry_after = retry_after
        super().__init__(model, 429, detail, provider)


class PayloadError(ProviderError):
    """HTTP 400 / 422: the request itself is wrong, so it is never resent as is."""


class AuthError(ProviderError):
    """HTTP 401 / 403. A 401, or a 403 whose body names the key, refuses the
    key (provider cooldown); any other 403 refuses only that model."""


class CreditsError(ProviderError):
    """HTTP 402: the account is out of credits."""


class ModelGoneError(ProviderError):
    """HTTP 404 / 410 (not OpenRouter's "No endpoints found that ...", a
    :class:`PayloadError`). ``eol`` is the end-of-life date a 410 body names."""

    def __init__(
        self,
        model: str,
        status: int,
        detail: str,
        provider: str = "",
        *,
        eol: str | None = None,
    ) -> None:
        self.eol = eol
        super().__init__(model, status, detail, provider)


class TransientError(ProviderError):
    """Worth trying elsewhere now: 5xx, a timeout, a transport error, bad JSON.

    ``status`` is 0 when no HTTP answer arrived at all.
    """


class EmptyCompletionError(ProviderError):
    """HTTP 200 without text, tool calls or images (e.g. every token went to reasoning)."""

    def __init__(self, model: str, provider: str = "", *, finish_reason: str | None = None) -> None:
        self.finish_reason = finish_reason
        note = f" (finish_reason={finish_reason})" if finish_reason else ""
        super().__init__(model, 200, f"empty completion (no text, tool calls or images){note}", provider)


class ContentFilteredError(ProviderError):
    """HTTP 200 whose image the model's safety filter withheld
    (finishReason ``CONTENT_FILTERED``): the model works, the prompt was
    refused, so another model may still draw it."""

    def __init__(self, model: str, provider: str = "") -> None:
        super().__init__(model, 200, "content filtered", provider)


class SkipTierError(ProviderError):
    """This tier cannot serve this request; the chain counts it as not attempted."""

    def __init__(self, model: str, detail: str, provider: str = "") -> None:
        super().__init__(model, 0, detail, provider)


@dataclass
class ProviderResponse:
    """Unified response from any LLM provider."""

    text: str | None = None
    images: list[str] | None = None  # base64 data URLs
    model: str = ""
    usage: dict | None = None
    reasoning_content: str | None = None
    # OpenAI-compatible tool call directives requested by the model.
    # Each item has the shape:
    #   {"id": str, "type": "function",
    #    "function": {"name": str, "arguments": str (JSON-encoded)}}
    # ``None`` (or empty list) means the model produced a regular text reply.
    tool_calls: list[dict] | None = None
    # "stop", "length", "tool_calls", ... as the upstream reported it.
    finish_reason: str | None = None
    reasoning_tokens: int = 0
    # Which provider answered ("nvidia", "openrouter", ...).
    provider: str = ""


def is_empty_completion(response: ProviderResponse) -> bool:
    """No text, no tool calls and no images: nothing a caller could use."""
    return not (
        getattr(response, "text", None)
        or getattr(response, "tool_calls", None)
        or getattr(response, "images", None)
    )


def require_output(response: ProviderResponse) -> ProviderResponse:
    """*response* unchanged, or :class:`EmptyCompletionError` when it is empty."""
    if is_empty_completion(response):
        raise EmptyCompletionError(
            response.model, response.provider, finish_reason=response.finish_reason
        )
    return response


# Inline reasoning markup that must never reach the user:
# - a leading ``<think>…</think>`` block (Qwen / Nemotron without a
#   reasoning parser);
# - the text before the first ``</think>`` when no ``<think>`` comes before
#   it: the server swallowed the opening tag;
# - a ``<|channel>thought…<channel|>`` channel block, which some chat
#   templates emit (possibly empty) even with thinking turned off; stripped
#   for every model, whichever one answers.
# Think tags anywhere else are the answer's own text (one that explains
# them, say), and so is a ``</think>`` after reasoning that came back in its
# own field.
_LEADING_THINK_RE = re.compile(r"\s*<think>(.*?)</think>\s*", re.DOTALL)
_CHANNEL_THOUGHT_RE = re.compile(r"<\|channel>thought(.*?)<channel\|>\s*", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_CHANNEL_THOUGHT_OPEN = "<|channel>thought"


def _split_inline_reasoning(
    text: str, *, separate_reasoning: bool, finished: bool
) -> tuple[str | None, str | None]:
    """Separate inline reasoning markup from the answer.

    Returns ``(answer, reasoning)``; either may be ``None``.
    *separate_reasoning*: the reasoning came back in its own field, so a
    ``</think>`` is markup only with nothing but whitespace before it.
    An opening marker at the very start with no closer means the reply was
    cut off mid-reasoning, so all of it is reasoning and there is no answer;
    except that a reply the model *finished* itself (finish_reason "stop")
    may start with a literal ``<think>``.
    """
    reasoning_parts: list[str] = []

    def _keep(part: str) -> str:
        part = part.strip()
        if part:
            reasoning_parts.append(part)
        return ""

    text = _CHANNEL_THOUGHT_RE.sub(lambda match: _keep(match.group(1)), text)

    leading = _LEADING_THINK_RE.match(text)
    if leading:
        _keep(leading.group(1))
        text = text[leading.end():]
    elif _THINK_CLOSE in text:
        before, _, after = text.partition(_THINK_CLOSE)
        if _THINK_OPEN not in before and not (separate_reasoning and before.strip()):
            _keep(before)
            text = after

    stripped = text.lstrip()
    openers = (_CHANNEL_THOUGHT_OPEN,) if finished else (_CHANNEL_THOUGHT_OPEN, _THINK_OPEN)
    for opener in openers:
        if stripped.startswith(opener):
            _keep(stripped[len(opener):])
            text = ""
            break

    answer = text.strip() or None
    reasoning = "\n\n".join(reasoning_parts) or None
    return answer, reasoning


def _reasoning_tokens(usage: dict | None) -> int:
    """Reasoning tokens, top-level or under ``completion_tokens_details``."""
    usage = usage if isinstance(usage, dict) else {}
    value = usage.get("reasoning_tokens") or (
        usage.get("completion_tokens_details") or {}
    ).get("reasoning_tokens")
    return int(value) if isinstance(value, (int, float)) else 0


def parse_openai_response(data: dict, *, model: str = "", provider: str = "") -> ProviderResponse:
    """Parse an OpenAI-compatible chat completion into a :class:`ProviderResponse`.

    Reasoning is extracted from (in priority order):
      1. ``message.reasoning_content`` — NVIDIA's field
      2. ``message.reasoning`` — OpenRouter's field
      3. Content parts with ``type: "thinking"``
      4. Inline markup in the text: a leading ``<think>…</think>`` block,
         the text before a ``</think>`` whose opening tag the server
         swallowed (after 1-3, only a bare stray ``</think>``), and a
         ``<|channel>thought…<channel|>`` block

    That markup is stripped from the returned text, and think tags anywhere
    else are kept (see :func:`_split_inline_reasoning`); if nothing but
    reasoning remains, ``text`` is ``None``.

    Tool calls follow the OpenAI Chat Completions format:
    ``message.tool_calls = [{"id", "type": "function",
    "function": {"name", "arguments"}}, ...]``. Arguments always come back
    as a JSON string so the agentic loop can echo them verbatim.

    ``model`` is used when the body does not name one.
    """
    text = None
    images = None
    reasoning_content = None
    tool_calls: list[dict] | None = None
    choices = data.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) or {}

    for field in ("reasoning_content", "reasoning"):
        val = message.get(field)
        if isinstance(val, str) and val.strip():
            reasoning_content = val.strip()
            break

    if isinstance(message.get("content"), str):
        text = message["content"]
    elif isinstance(message.get("content"), list):
        text_parts: list[str] = []
        for part in message["content"]:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                part_text = part.get("text")
                if isinstance(part_text, str):
                    text_parts.append(part_text)
            elif part.get("type") == "image_url":
                if images is None:
                    images = []
                images.append(part["image_url"]["url"])
            elif part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                reasoning_content = reasoning_content or part["thinking"].strip()
        if text_parts:
            text = "".join(text_parts)

    # Some image models return images in a separate field
    if message.get("images"):
        images = images or []
        for img in message["images"]:
            if isinstance(img, dict) and "image_url" in img:
                val = img["image_url"]
                images.append(val["url"] if isinstance(val, dict) else val)
            elif isinstance(img, str):
                images.append(img)

    finish_reason = choice.get("finish_reason")
    if text:
        text, inline_reasoning = _split_inline_reasoning(
            text,
            separate_reasoning=reasoning_content is not None,
            finished=finish_reason == "stop",
        )
        reasoning_content = reasoning_content or inline_reasoning

    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list) and raw_tool_calls:
        normalized: list[dict] = []
        for call in raw_tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, dict):
                # Some providers already deliver parsed args; re-encode so
                # downstream consumers always see a JSON string.
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    args = ""
            normalized.append(
                {
                    "id": call.get("id") or "",
                    "type": call.get("type", "function"),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args if isinstance(args, str) else "",
                    },
                }
            )
        if normalized:
            tool_calls = normalized

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return ProviderResponse(
        text=text,
        images=images,
        model=str(data.get("model") or model),
        usage=usage,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        reasoning_tokens=_reasoning_tokens(usage),
        provider=provider,
    )
