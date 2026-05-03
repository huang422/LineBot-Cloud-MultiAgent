"""Agentic tool-calling loop for OpenAI-compatible providers.

The loop drives a chat completion request that has ``tools=[...]``
attached. While the model responds with ``tool_calls``, we execute
each call locally, append the ``tool`` result message, and re-enter
the chain. Once the model returns a plain text reply (no
``tool_calls``) the loop exits and returns the final response.

This module is intentionally provider-agnostic. The provider used to
issue requests is supplied via a ``FallbackChain`` instance, so we
inherit rate-limit handling, thinking-mode timeouts, and provider
fallback semantics for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.agents.tools import (
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    parse_tool_arguments,
)
from src.providers.fallback_chain import FallbackChain, Target
from src.providers.openrouter_provider import ProviderResponse
from src.utils.logger import logger


class ToolLoopError(Exception):
    """Raised when the loop cannot make progress."""


@dataclass
class ToolLoopResult:
    response: ProviderResponse
    iterations: int
    tool_calls_executed: list[str]


async def run_tool_loop(
    *,
    fallback_chain: FallbackChain,
    targets: list[Target],
    messages: list[dict],
    registry: ToolRegistry,
    context: ToolContext,
    tool_choice: str | dict = "auto",
    allowed_tools: list[str] | None = None,
    max_iterations: int = 4,
    thinking_timeout: float | None = None,
    **generate_kwargs,
) -> ToolLoopResult:
    """Run the agentic tool-calling loop.

    Args:
        fallback_chain: Provider fallback chain to issue requests with.
        targets: Ordered ``(provider, model_id)`` list. All targets must
            support ``tools`` / ``tool_calls`` (OpenAI-compatible). The
            caller is responsible for pre-filtering.
        messages: Initial conversation messages (must already contain a
            user turn). The loop appends assistant + tool messages as
            it iterates and DOES mutate this list in place; pass a copy
            if the caller wants to preserve the original.
        registry: Source of tool schemas + executors.
        context: Per-request context handed to every executor.
        tool_choice: ``"auto"`` (let the LLM decide), ``"none"`` (force
            text reply), ``"required"`` (force a tool call), or a
            specific function spec.
        allowed_tools: Subset of registered tool names to expose. None
            exposes every registered tool.
        max_iterations: Hard upper bound on tool-call rounds. Anything
            beyond this returns the latest response even if it still
            contains tool_calls (caller should treat this as a hint
            that the loop diverged).
        thinking_timeout: Forwarded to FallbackChain on every iteration.
        **generate_kwargs: Forwarded verbatim to ``provider.generate``
            (temperature, max_tokens, disable_thinking, …).

    Returns:
        ToolLoopResult with the terminal ProviderResponse, the number
        of iterations spent, and the list of executed tool names.
    """
    if max_iterations < 1:
        raise ToolLoopError("max_iterations must be >= 1")
    if not registry.names():
        raise ToolLoopError("Tool registry is empty")

    schemas = registry.schemas(allowed_tools)
    if not schemas:
        raise ToolLoopError("No tools matched the requested allow-list")

    executed: list[str] = []
    last_response: ProviderResponse | None = None

    for iteration in range(1, max_iterations + 1):
        last_response = await fallback_chain.generate(
            targets,
            messages,
            tools=schemas,
            tool_choice=tool_choice,
            thinking_timeout=thinking_timeout,
            **generate_kwargs,
        )

        tool_calls = last_response.tool_calls or []
        if not tool_calls:
            # Plain assistant reply; we are done.
            logger.info(
                f"Tool loop: finished after {iteration} iteration(s) "
                f"({len(executed)} tool call(s) executed)"
            )
            return ToolLoopResult(
                response=last_response,
                iterations=iteration,
                tool_calls_executed=executed,
            )

        # Append the assistant turn verbatim so subsequent tool messages
        # reference the right tool_call_id values.
        assistant_turn = _assistant_turn_from_response(last_response, tool_calls)
        messages.append(assistant_turn)

        # Execute each tool call, appending a `tool` role message for each.
        for call in tool_calls:
            name = (call.get("function") or {}).get("name", "")
            call_id = call.get("id") or ""
            if not call_id:
                # OpenAI-compatible providers reject tool messages with
                # empty tool_call_id. If the model emits a malformed
                # tool_calls entry we skip it; the loop will then exit
                # on the next iteration (no further tool_calls expected
                # because the assistant turn we appended already
                # contains the malformed call list).
                logger.warning(
                    f"Tool loop: dropping tool call {name!r} with empty id"
                )
                continue
            args_raw = (call.get("function") or {}).get("arguments")
            args = parse_tool_arguments(args_raw)

            tool: ToolDefinition | None = registry.get(name)
            if tool is None or (
                allowed_tools is not None and name not in allowed_tools
            ):
                logger.warning(f"Tool loop: model called unknown tool {name!r}")
                content = json.dumps(
                    {"status": "error", "reason": f"unknown tool: {name}"},
                    ensure_ascii=False,
                )
            else:
                try:
                    result = await tool.executor(args, context)
                    content = _serialize_tool_result(result)
                    executed.append(name)
                    logger.info(
                        f"Tool loop: executed {name} "
                        f"(arg_keys={sorted(args.keys())})"
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        f"Tool loop: executor for {name} raised {exc!r}"
                    )
                    content = json.dumps(
                        {"status": "error", "reason": str(exc)},
                        ensure_ascii=False,
                    )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": content,
                }
            )

    logger.warning(
        f"Tool loop: hit max_iterations={max_iterations} without a "
        "plain reply; returning the latest response anyway"
    )
    assert last_response is not None  # loop ran at least once
    return ToolLoopResult(
        response=last_response,
        iterations=max_iterations,
        tool_calls_executed=executed,
    )


def _assistant_turn_from_response(
    response: ProviderResponse, tool_calls: list[dict]
) -> dict:
    """Reconstruct the assistant message we send back to the provider.

    OpenAI-style providers expect the assistant message to mirror the
    one they emitted, including ``tool_calls`` and any text. If the
    provider returned ``raw_message`` we prefer that verbatim; if not
    (older mock paths in tests) we synthesise one from the parsed
    fields.
    """
    raw = response.raw_message
    if isinstance(raw, dict):
        # Some providers omit role on multipart responses; backfill.
        if raw.get("role") != "assistant":
            raw = {**raw, "role": "assistant"}
        return raw
    return {
        "role": "assistant",
        "content": response.text or "",
        "tool_calls": tool_calls,
    }


def _serialize_tool_result(result) -> str:
    """Best-effort JSON encoding of a tool executor's return value."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)
