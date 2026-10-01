"""Agentic tool-calling loop for OpenAI-compatible providers.

The loop drives a chat completion request that has ``tools=[...]``
attached. While the model responds with ``tool_calls``, we execute
each call locally, append the ``tool`` result message, and re-enter
the chain. Once the model returns a plain text reply (no
``tool_calls``) the loop exits and returns the final response. After
``max_iterations`` rounds of tool calls, one last request with
``tool_choice="none"`` asks for the answer, so the user gets text and
the last round's tool results are not wasted.

The first round has the request's whole deadline, like a request without
tools (most turns end there, with a plain answer), and the chain's usual
strike rules. Once the model asks for tools, the last
:data:`ANSWER_RESERVE_SECONDS` before the deadline are kept for that last
request: every tool call and every later tool-offering round must end
before them. A tool that runs out of time answers with an error, and tool
calls that arrive inside the reserve are answered that way without being
run. A later round that does not fit (or is cut off, which is not held
against its model) goes straight to that last request, which asks for the
answer with what the model has, in the time left. A tool with a per-turn
cap (``ToolDefinition.max_calls_per_turn``; web_search: 2) runs at most that
many times in the whole loop, every round counted: a call past the cap is
answered ``limit`` without being run. Tool results lose the
prompt's fence labels (:data:`~src.agents.base_agent.FENCE_LABELS`): they
carry stored text other people wrote.

This module is intentionally provider-agnostic. The provider used to
issue requests is supplied via a ``FallbackChain`` instance, so we
inherit rate-limit handling, per-tier time limits, and provider
fallback semantics for free. The assistant turns it sends back are
rebuilt in OpenAI format from the parsed response, so they replay
safely on whichever tier answers the next round.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from src.agents.base_agent import unlabelled
from src.agents.tools import (
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    parse_tool_arguments,
)
from src.providers.base import ProviderResponse
from src.providers.fallback_chain import DeadlineExceededError, FallbackChain, Tier
from src.utils.logger import logger

# Seconds of the deadline kept for the answer once the model has asked for
# tools: a tool call, and a tool-offering round after the first, must end
# this long before it (the web search agent keeps the same reserve).
ANSWER_RESERVE_SECONDS = 20.0
# The result of a tool call that ran out of time, or had none to start.
_NO_TIME_LEFT = json.dumps(
    {"status": "error", "reason": "no time left for this tool; answer with what you have"}
)


class ToolLoopError(Exception):
    """Raised when the loop cannot make progress."""


@dataclass
class ToolLoopResult:
    response: ProviderResponse
    iterations: int
    tool_calls_executed: list[str]
    # The whole exchange: the caller's messages, then every assistant
    # tool-call turn and tool result the loop added.
    messages: list[dict]


async def run_tool_loop(
    *,
    fallback_chain: FallbackChain,
    targets: list[Tier],
    messages: list[dict],
    registry: ToolRegistry,
    context: ToolContext,
    tool_choice: str | dict = "auto",
    allowed_tools: list[str] | None = None,
    max_iterations: int = 4,
    **generate_kwargs,
) -> ToolLoopResult:
    """Run the agentic tool-calling loop.

    Args:
        fallback_chain: Provider fallback chain to issue requests with.
        targets: Ordered tiers. The providers send ``tools`` only to
            models whose profile accepts them (see
            :mod:`src.providers.model_profiles`); the others answer in text.
        messages: Initial conversation messages (must already contain a
            user turn). Never modified: the loop works on a copy, which
            the result returns as ``messages``.
        registry: Source of tool schemas + executors.
        context: Per-request context handed to every executor.
        tool_choice: ``"auto"`` (let the LLM decide), ``"none"`` (force
            text reply), ``"required"`` (force a tool call), or a
            specific function spec.
        allowed_tools: Subset of registered tool names to expose. None
            exposes every registered tool.
        max_iterations: Hard upper bound on tool-call rounds. After that
            many rounds one more request, with ``tool_choice="none"``,
            asks for the answer.
        **generate_kwargs: Forwarded to ``FallbackChain.generate`` on
            every request (deadline, temperature, max_tokens,
            disable_thinking, …). One ``deadline`` bounds the whole loop.
            The first round may use all of it, under the chain's usual
            strike rules. Once the model asks for tools, the tool calls
            and every later tool-offering round must end
            ANSWER_RESERVE_SECONDS before it (a later round that reserve
            cuts short is not held against its model; tool calls that
            arrive inside it are not run), and only the final
            ``tool_choice="none"`` request uses the reserve.

    Returns:
        ToolLoopResult with the terminal ProviderResponse, the number
        of requests made, the list of executed tool names and the
        messages exchanged.

    Raises:
        ToolLoopError: The loop is misconfigured, or the final
            ``tool_choice="none"`` request still produced no text.
        DeadlineExceededError: The first round or the final request ran
            out of the deadline (a later round that does is not an error).
        AllProvidersFailedError, AllModelsRateLimitedError: From the
            chain, unchanged.
    """
    if max_iterations < 1:
        raise ToolLoopError("max_iterations must be >= 1")
    if not registry.names():
        raise ToolLoopError("Tool registry is empty")

    schemas = registry.schemas(allowed_tools)
    if not schemas:
        raise ToolLoopError("No tools matched the requested allow-list")

    conversation = list(messages)
    executed: list[str] = []
    deadline = generate_kwargs.get("deadline")
    # Where tool calls, and the tool-offering rounds after the first, must end.
    tools_until = None if deadline is None else deadline - ANSWER_RESERVE_SECONDS
    # The first round is the whole answer on most turns: it gets the whole
    # deadline. The later ones follow tool calls and must leave the reserve;
    # a round that reserve cuts short is not held against its model.
    later_round_kwargs = dict(generate_kwargs)
    if tools_until is not None:
        later_round_kwargs.update(deadline=tools_until, hold_deadline_timeouts=False)

    rounds = 0
    while rounds < max_iterations:
        if rounds and tools_until is not None and time.monotonic() >= tools_until:
            logger.warning(
                "Tool loop: no time left for another tool round; asking for the answer "
                "with tool_choice='none'"
            )
            break
        rounds += 1
        try:
            response = await fallback_chain.generate(
                targets,
                conversation,
                tools=schemas,
                tool_choice=tool_choice,
                **(generate_kwargs if rounds == 1 else later_round_kwargs),
            )
        except DeadlineExceededError:
            if rounds == 1:
                # The whole deadline is gone: nothing left for an answer.
                raise
            logger.warning(
                "Tool loop: the round ran into the answer reserve; asking for the answer "
                "with tool_choice='none'"
            )
            break

        tool_calls = _normalized_tool_calls(response.tool_calls or [], rounds)
        if not tool_calls:
            # Plain assistant reply; we are done.
            logger.info(
                f"Tool loop: finished after {rounds} iteration(s) "
                f"({len(executed)} tool call(s) executed)"
            )
            return ToolLoopResult(
                response=response,
                iterations=rounds,
                tool_calls_executed=executed,
                messages=conversation,
            )

        conversation.append(
            {"role": "assistant", "content": response.text or "", "tool_calls": tool_calls}
        )
        # A call that arrives inside the reserve (the first round may run
        # into it) is answered with _NO_TIME_LEFT, never run, and the next
        # pass goes straight to the answer.
        for call in tool_calls:
            conversation.append(
                await _run_tool_call(call, registry, allowed_tools, context, executed, tools_until)
            )
    else:
        logger.warning(
            f"Tool loop: hit max_iterations={max_iterations}; asking for the answer "
            "with tool_choice='none'"
        )
    response = await fallback_chain.generate(
        targets,
        conversation,
        tools=schemas,
        tool_choice="none",
        **generate_kwargs,
    )
    if not response.text:
        raise ToolLoopError(
            f"No text after {rounds} tool round(s) and a tool_choice='none' request"
        )
    return ToolLoopResult(
        response=response,
        iterations=rounds + 1,
        tool_calls_executed=executed,
        messages=conversation,
    )


def _normalized_tool_calls(tool_calls: list[dict], iteration: int) -> list[dict]:
    """*tool_calls* in OpenAI format, built from the parsed fields only.

    Provider-specific extras (reasoning, channel blocks) never reach the
    next request, dict arguments become JSON strings, and a call the
    model sent without an id gets one, so its tool result can answer it.
    """
    normalized: list[dict] = []
    for index, call in enumerate(tool_calls):
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str) or not arguments.strip():
            arguments = "{}"
        call_id = str(call.get("id") or "") or f"call_{iteration}_{index}"
        normalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return normalized


async def _run_tool_call(
    call: dict,
    registry: ToolRegistry,
    allowed_tools: list[str] | None,
    context: ToolContext,
    executed: list[str],
    until: float | None,
) -> dict:
    """Execute one tool call, ending by *until* (``time.monotonic()``) if
    given (once *until* has passed it is not run); the ``tool`` message
    that answers it."""
    name = call["function"]["name"]
    args = parse_tool_arguments(call["function"]["arguments"])

    tool: ToolDefinition | None = registry.get(name)
    if tool is None or (allowed_tools is not None and name not in allowed_tools):
        logger.warning(f"Tool loop: model called unknown tool {name!r}")
        content = json.dumps(
            {"status": "error", "reason": f"unknown tool: {name}"},
            ensure_ascii=False,
        )
    elif tool.max_calls_per_turn is not None and executed.count(name) >= tool.max_calls_per_turn:
        logger.warning(f"Tool loop: {name} not run, this turn already ran it {tool.max_calls_per_turn} times")
        content = json.dumps(
            {
                "status": "limit",
                "reason": (
                    f"this reply already ran {name} {tool.max_calls_per_turn} times; "
                    "answer with the results you have"
                ),
            },
            ensure_ascii=False,
        )
    elif until is not None and time.monotonic() >= until:
        logger.warning(f"Tool loop: no time left to run {name}")
        content = _NO_TIME_LEFT
    else:
        try:
            async with asyncio.timeout_at(until):
                result = await tool.executor(args, context)
            content = _serialize_tool_result(result)
            executed.append(name)
            logger.info(f"Tool loop: executed {name} (arg_keys={sorted(args.keys())})")
        except TimeoutError:
            logger.warning(f"Tool loop: {name} ran out of time")
            content = _NO_TIME_LEFT
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Tool loop: executor for {name} raised {exc!r}")
            content = json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False)

    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": name,
        "content": unlabelled(content),
    }


def _serialize_tool_result(result) -> str:
    """Best-effort JSON encoding of a tool executor's return value."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)
