"""The Colab tiers' chat client: Ollama's native ``POST /api/chat``.

Before any request, :meth:`OllamaChatProvider.generate` raises
:class:`~src.providers.base.SkipTierError` (the chain moves on, nothing is
counted) when the messages cannot go to Ollama (an image given as a URL,
see below), or when the
:class:`~src.services.colab.endpoint.ColabEndpoint` says the session is
not READY, its breaker is open, or the model is already serving another
request. None of that touches the network.

The body is always ``{model, messages, stream: false, think, keep_alive: -1,
options}`` plus ``tools`` when the call has them (and ``format`` for a
json_schema tier):

- ``think`` is ``false`` or the tier's level ("low", "medium", "xhigh");
  never ``true``, "high" or "max", which Ollama silently runs as medium;
- ``options`` holds sampling only (Qwen's recommendations: 0.7 / 0.8 / 20
  with presence_penalty 1.5 without thinking, 1.0 / 0.95 / 20 with it) and
  ``num_predict``, which thinking and the answer share: the answer's
  max_tokens (the tier's, else the call's, CHAT_MAX_TOKENS or
  VISION_MAX_TOKENS) plus, while thinking, the thinking budget
  (NVIDIA_THINKING_BUDGET), as the API tiers get it. ``num_ctx``,
  ``draft_num_predict`` and ``shift`` are never sent: they are load-time
  options, and a request that changes one makes Ollama reload the model.

Messages become Ollama's shape: one leading system message (Qwen folds
every system turn into it anyway), user images as raw base64 in
``images`` (only ``data:`` URIs), assistant tool calls with dict arguments, and tool results as ``{role: "tool",
tool_name, tool_call_id, content}``. Reasoning is never replayed. An
image given as an http(s) URL (a quoted image from the message cache or
GCS) skips the tier by design, ``skip(by design: url image)`` in the
request's summary line: the API vision tiers fetch it themselves, and the
VM is never asked to.

The answer's ``thinking`` becomes ``reasoning_content``; tool call
arguments come back as JSON strings with Ollama's ``call_…`` ids kept. A
thinking answer cut at ``num_predict`` (``done_reason: "length"``) is no
answer (EmptyCompletionError, the next tier gets the request): the thinking
may have left only a few words of it.

Errors: 401/403 are the auth proxy refusing this session's bearer (or,
with an empty 403, Ollama itself refusing the Host header, i.e. a
misconfigured proxy) → the endpoint marked unhealthy until the next
passing health check re-validates the bearer (three failed checks drain
the session) and a TransientError, never the AuthError whose 10-minute
provider cooldown is meant for a refused API key; 404 means the model is
not pulled or the session not ready → endpoint unhealthy and a
TransientError, never the NVIDIA end-of-life breaker; 499, 503, 524 and
the client's own timeouts are transient. A call that is cancelled (the
chain's time limit for the tier, the turn's deadline, a shutdown) is not
counted against the endpoint: the chain judges its timeouts (never for a
tier thinking for a request that needs it; Colab chat thinking at
COLAB_CHAT_THINK=low for a plain request counts) and ModelHealth cools the
model down after two it counts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from src.providers.base import (
    EmptyCompletionError,
    PayloadError,
    ProviderError,
    ProviderResponse,
    SkipTierError,
    TransientError,
    parse_openai_response,
    require_output,
)
from src.providers.messages import normalize_messages
from src.providers.model_profiles import DEFAULT_THINKING_BUDGET, JSON_SCHEMA, THINK_LEVELS
from src.providers.openai_compat import error_for_status
from src.services.colab.endpoint import ColabEndpoint
from src.services.search.page_fetcher import FetchError
from src.utils.logger import logger

# Qwen3's recommended sampling (non-thinking / thinking).
_PLAIN_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
_THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
# The answer's max_tokens when neither the tier nor the call gives one.
DEFAULT_MAX_TOKENS = 2048
_DATA_URI_PREFIX = "data:"
_BASE64_MARKER = ";base64,"
_URL_PREFIXES = ("http://", "https://")
# The SkipTierError detail for an image given as a URL (the summary line's
# skip reason): not a failure, the API vision tiers answer it.
URL_IMAGE_SKIP = "by design: url image"
_DETAIL_LIMIT = 300


def think_value(options: Mapping[str, Any], *, disable_thinking: bool, default: str) -> bool | str:
    """``false`` or an Ollama think level for this call (never ``true``)."""
    if disable_thinking:
        return False
    level = str(options.get("think") or default).strip().lower()
    if level not in THINK_LEVELS or level == "false":
        return False
    return level


def build_chat_body(
    model: str,
    messages: list[dict],
    *,
    think: bool | str,
    options: Mapping[str, Any] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    json_schema: dict | None = None,
) -> dict:
    """The ``/api/chat`` body. See the module docstring.

    *max_tokens* is the call's answer budget (a tier's ``max_tokens`` wins);
    *thinking_budget* is added while thinking.

    Raises:
        SkipTierError: An image that is not a ``data:`` URI, or no user turn.
    """
    options = options or {}
    thinking = think is not False
    sampling: dict[str, Any] = dict(_THINKING_SAMPLING if thinking else _PLAIN_SAMPLING)
    if options.get("temperature") is not None:
        sampling["temperature"] = float(options["temperature"])
    answer = int(options.get("max_tokens") or max_tokens)
    sampling["num_predict"] = answer + thinking_budget if thinking else answer
    body: dict[str, Any] = {
        "model": model,
        "messages": to_ollama_messages(model, messages),
        "stream": False,
        "think": think,
        "keep_alive": -1,
        "options": sampling,
    }
    # Ollama has no tool_choice: "none" (the tool loop's last call) means
    # no tools at all; anything else offers them.
    if tools and tool_choice != "none":
        body["tools"] = tools
    if json_schema is not None and options.get("json_mode") == JSON_SCHEMA:
        body["format"] = json_schema
    return body


def to_ollama_messages(model: str, messages: list[dict]) -> list[dict]:
    """OpenAI-format *messages* in Ollama's shape (the input is not mutated).

    Raises:
        SkipTierError: An image that is not inline data, or no user turn.
    """
    system_parts: list[str] = []
    converted: list[dict] = []
    tool_names: dict[str, str] = {}
    for message in normalize_messages(messages):
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue
        if role == "user":
            text, images = _user_content(model, content)
            item: dict[str, Any] = {"role": "user", "content": text}
            if images:
                item["images"] = images
            converted.append(item)
        elif role == "assistant":
            item = {"role": "assistant", "content": content if isinstance(content, str) else ""}
            calls = [_tool_call(call) for call in message.get("tool_calls") or [] if isinstance(call, dict)]
            if calls:
                item["tool_calls"] = calls
                tool_names.update({call["id"]: call["function"]["name"] for call in calls if call.get("id")})
            converted.append(item)
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            converted.append(
                {
                    "role": "tool",
                    "tool_name": str(message.get("name") or tool_names.get(call_id, "")),
                    "tool_call_id": call_id,
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                }
            )
    if not any(item["role"] == "user" for item in converted):
        raise SkipTierError(model, "no user message to answer", "colab")
    if system_parts:
        converted.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return converted


def _user_content(model: str, content: Any) -> tuple[str, list[str]]:
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[str] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif part.get("type") == "image_url":
            images.append(_raw_base64(model, part.get("image_url")))
    return "\n\n".join(text for text in texts if text), images


def _raw_base64(model: str, image_url: Any) -> str:
    """The base64 payload of a ``data:…;base64,`` URI (Ollama wants it bare)."""
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    url = url if isinstance(url, str) else ""
    if url.startswith(_URL_PREFIXES):
        raise SkipTierError(model, URL_IMAGE_SKIP, "colab")
    if not url.startswith(_DATA_URI_PREFIX) or _BASE64_MARKER not in url:
        raise SkipTierError(model, "image is not inline base64 data", "colab")
    return url.split(_BASE64_MARKER, 1)[1]


def _tool_call(call: dict) -> dict:
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            arguments = {}
    item: dict[str, Any] = {
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": arguments if isinstance(arguments, dict) else {},
        }
    }
    if call.get("id"):
        item["id"] = str(call["id"])
    return item


def parse_chat_response(data: dict, *, model: str) -> ProviderResponse:
    """An ``/api/chat`` answer as a :class:`ProviderResponse` (may be empty)."""
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    calls = []
    for index, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        calls.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {"name": function.get("name", ""), "arguments": function.get("arguments") or {}},
            }
        )
    done_reason = data.get("done_reason")
    usage = {
        key: value
        for key, value in (
            ("prompt_tokens", data.get("prompt_eval_count")),
            ("completion_tokens", data.get("eval_count")),
            ("prompt_cached_tokens", data.get("prompt_eval_cached_count")),
        )
        if isinstance(value, int)
    }
    # The OpenAI parser strips inline reasoning markup and re-encodes the
    # dict arguments as JSON strings, like every other provider's answer.
    return parse_openai_response(
        {
            "model": data.get("model") or model,
            "choices": [
                {
                    "message": {
                        "content": message.get("content") or "",
                        "reasoning_content": message.get("thinking") or "",
                        "tool_calls": calls,
                    },
                    "finish_reason": done_reason if isinstance(done_reason, str) else None,
                }
            ],
            "usage": usage or None,
        },
        model=model,
        provider="colab",
    )


class OllamaChatProvider:
    """The Colab chat provider; see the module docstring."""

    name = "colab"

    def __init__(
        self,
        endpoint: ColabEndpoint,
        *,
        default_think: str = "low",
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
    ) -> None:
        """``default_think`` (COLAB_CHAT_THINK) applies to a Colab tier that
        sets no think level of its own and is asked to think;
        ``thinking_budget`` (NVIDIA_THINKING_BUDGET) is added to num_predict
        while thinking. The endpoint is shared (router, registration) and
        closed by its owner, the container."""
        self.endpoint = endpoint
        self._default_think = default_think
        self._thinking_budget = thinking_budget

    async def close(self) -> None:
        """Nothing of its own to close (the endpoint holds the client)."""

    async def generate(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        json_schema: dict | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> ProviderResponse:
        """One ``/api/chat`` call, or SkipTierError when Colab cannot take it now.

        ``temperature`` (the agent's) is not used: the Qwen sampling above
        applies. ``max_tokens`` is the answer's budget unless the tier sets
        one (see :func:`build_chat_body`).

        Raises:
            SkipTierError: The input cannot go to Ollama (checked first, so
                an image URL is skipped as such even with Colab down), not
                READY, breaker open or model busy (no request is sent).
            ProviderError: A subclass per failure (see the module docstring).
        """
        options = options or {}
        think = think_value(options, disable_thinking=disable_thinking, default=self._default_think)
        body = build_chat_body(
            model,
            messages,
            think=think,
            options=options,
            max_tokens=max_tokens,
            thinking_budget=self._thinking_budget,
            tools=tools,
            tool_choice=tool_choice,
            json_schema=json_schema,
        )
        endpoint = self.endpoint
        if not await endpoint.ready():
            raise SkipTierError(model, "Colab session not READY", self.name)
        if endpoint.breaker_open():
            raise SkipTierError(model, "Colab breaker open after repeated failures", self.name)
        gate = endpoint.semaphore(model)
        if gate.locked():
            raise SkipTierError(model, "Colab model busy with another request", self.name)

        async with gate:
            logger.info(
                f"Colab request → {model} (think={think}, num_predict={body['options']['num_predict']}, "
                f"images={sum(len(m.get('images', ())) for m in body['messages'])}, tools={bool(body.get('tools'))})"
            )
            # A cancellation (the chain's time limit, the turn's deadline, a
            # shutdown) is not counted here: the chain decides whether a
            # timeout says anything about the model (ModelHealth).
            try:
                response = await endpoint.call("POST", "/api/chat", json=body)
            except (httpx.HTTPError, FetchError, RuntimeError) as exc:
                endpoint.record_failure(f"{type(exc).__name__}: {exc}")
                if isinstance(exc, httpx.TimeoutException):
                    raise TransientError(model, 0, f"timed out ({type(exc).__name__})", self.name) from exc
                raise TransientError(model, 0, f"{type(exc).__name__}: {exc}", self.name) from exc
        if response.status_code != 200:
            raise self._error(model, response)
        try:
            data = response.json()
        except ValueError as exc:
            endpoint.record_failure("invalid JSON body")
            raise TransientError(model, 200, "invalid JSON body", self.name) from exc
        if not isinstance(data, dict):
            endpoint.record_failure("JSON body is not an object")
            raise TransientError(model, 200, "JSON body is not an object", self.name)
        endpoint.record_success()

        result = parse_chat_response(data, model=model)
        logger.info(
            f"Colab response ← {model} (text={bool(result.text)}, finish={result.finish_reason}, "
            f"thinking_chars={len(result.reasoning_content or '')}, tool_calls={len(result.tool_calls or [])}, "
            f"load_ms={int((data.get('load_duration') or 0) / 1e6)}, usage={result.usage})"
        )
        if think is not False and result.finish_reason == "length" and result.text:
            # Thinking and the answer share num_predict: what is left of the
            # answer may be a few words. Like a reply that is all reasoning,
            # it is no answer (not the model's or the endpoint's fault).
            logger.warning(
                f"Colab: {model} used up num_predict={body['options']['num_predict']} while thinking"
            )
            raise EmptyCompletionError(model, self.name, finish_reason="length")
        return require_output(result)

    def _error(self, model: str, response: httpx.Response) -> ProviderError:
        """The provider error for a non-200 answer, with the endpoint told."""
        status = response.status_code
        detail = (response.text or "")[:_DETAIL_LIMIT]
        endpoint = self.endpoint
        if status == 403 and not detail.strip():
            # Ollama's own Host check: the proxy did not rewrite Host.
            logger.error(
                "Colab: Ollama refused the request's Host header (HTTP 403, empty body); "
                "the VM's auth proxy must send Host 127.0.0.1:11434"
            )
            endpoint.record_failure("Ollama refused the Host header (403)")
            endpoint.mark_unhealthy("Ollama refused the Host header (403)")
            return TransientError(model, status, "Ollama refused the Host header", self.name)
        if status in (401, 403):
            # The session's bearer, not an account key: the health check
            # decides whether the session is still usable.
            logger.error(f"Colab: the auth proxy refused the bearer (HTTP {status})")
            endpoint.record_failure(f"auth proxy refused the bearer ({status})")
            endpoint.mark_unhealthy(f"auth proxy refused the bearer ({status})")
            return TransientError(model, status, detail or "bearer refused", self.name)
        if status in (404, 410):
            # Not pulled, or not this session: never the end-of-life breaker.
            endpoint.record_failure(f"HTTP {status}: model or route not found")
            endpoint.mark_unhealthy(f"HTTP {status}: model or route not found")
            return TransientError(model, status, detail or "not found", self.name)
        if status == 499:
            # Ollama's "request canceled": the upstream side gave up.
            endpoint.record_failure("HTTP 499: request canceled upstream")
            return TransientError(model, status, detail or "request canceled", self.name)
        error = error_for_status(response, provider=self.name, model=model)
        if isinstance(error, TransientError):
            endpoint.record_failure(f"HTTP {status}")
        elif isinstance(error, PayloadError):
            logger.warning(f"Colab: {model} refused the request (HTTP {status}): {detail[:200]}")
        return error
