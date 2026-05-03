"""Phase C tool-calling infrastructure tests.

These tests cover:

* ``parse_openai_response`` returns the new 5-tuple including
  ``tool_calls`` and ``raw_message`` and re-encodes dict arguments to
  JSON strings.
* ``OpenRouterProvider.generate`` and ``NvidiaProvider.generate``
  forward ``tools`` / ``tool_choice`` into the request payload.
* ``ToolRegistry`` schema generation + executor swap-in.
* ``run_tool_loop`` exits cleanly when the model returns a plain text
  reply, dispatches the right executor when the model emits
  ``tool_calls``, respects the allow-list, and stops at
  ``max_iterations``.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.tool_runner import ToolLoopError, run_tool_loop
from src.agents.tools import (
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    build_default_registry,
    parse_tool_arguments,
)
from src.config import Settings
from src.providers.fallback_chain import FallbackChain
from src.providers.nvidia_provider import NvidiaProvider
from src.providers.openrouter_provider import (
    OpenRouterProvider,
    ProviderResponse,
    parse_openai_response,
)
from src.utils.rate_tracker import RateTracker


def _make_settings() -> Settings:
    return Settings(_env_file=None)


class ParseOpenAIResponseToolCallTests(unittest.TestCase):
    def test_tool_calls_are_extracted_and_arguments_json_stringified(self) -> None:
        text, images, reasoning, tool_calls, raw = parse_openai_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "recall_memory",
                                        "arguments": '{"query": "去年生日"}',
                                    },
                                },
                                {
                                    "id": "call_b",
                                    "function": {
                                        "name": "update_user_profile",
                                        # Some providers deliver a parsed dict;
                                        # parser must re-encode to JSON string.
                                        "arguments": {"facts": ["likes coffee"]},
                                    },
                                },
                            ],
                        }
                    }
                ]
            }
        )

        self.assertIsNone(text)
        self.assertIsNone(images)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["function"]["name"], "recall_memory")
        self.assertIn("去年生日", tool_calls[0]["function"]["arguments"])
        self.assertEqual(tool_calls[1]["type"], "function")
        # Dict arguments should have been stringified
        self.assertIsInstance(tool_calls[1]["function"]["arguments"], str)
        self.assertIn("likes coffee", tool_calls[1]["function"]["arguments"])
        self.assertIsInstance(raw, dict)
        self.assertEqual(raw.get("role"), "assistant")

    def test_no_tool_calls_means_none_not_empty_list(self) -> None:
        _, _, _, tool_calls, _ = parse_openai_response(
            {"choices": [{"message": {"content": "hi"}}]}
        )
        self.assertIsNone(tool_calls)


class ProviderToolForwardingTests(unittest.TestCase):
    """Both provider implementations must forward tools/tool_choice."""

    def _post_payload(self, post_mock) -> dict:
        return post_mock.call_args.kwargs.get("json") or post_mock.call_args[1]["json"]

    def test_openrouter_payload_contains_tools(self) -> None:
        provider = OpenRouterProvider("k", RateTracker(), reasoning_enabled=False)
        provider._client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        provider._client.post = AsyncMock(return_value=resp)

        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]

        async def go():
            return await provider.generate(
                "openrouter/auto",
                [{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice="auto",
            )

        asyncio.run(go())
        payload = self._post_payload(provider._client.post)
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_nvidia_payload_contains_tools(self) -> None:
        provider = NvidiaProvider("k", RateTracker(), thinking_enabled=False)
        provider._client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        provider._client.post = AsyncMock(return_value=resp)

        tools = [
            {
                "type": "function",
                "function": {"name": "recall_memory", "parameters": {}},
            }
        ]

        async def go():
            return await provider.generate(
                "qwen/qwen3.5-397b-a17b",
                [{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "recall_memory"}},
                disable_thinking=True,
            )

        asyncio.run(go())
        payload = self._post_payload(provider._client.post)
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"]["function"]["name"], "recall_memory")


class ToolRegistryTests(unittest.TestCase):
    def test_default_registry_exposes_recall_and_profile(self) -> None:
        reg = build_default_registry()
        names = reg.names()
        self.assertIn("recall_memory", names)
        self.assertIn("update_user_profile", names)
        schemas = reg.schemas()
        self.assertEqual(len(schemas), len(names))
        for s in schemas:
            self.assertEqual(s["type"], "function")
            self.assertIn("parameters", s["function"])

    def test_replace_executor_swaps_implementation(self) -> None:
        reg = build_default_registry()

        async def real_recall(args, ctx):
            return {"matches": [args["query"]]}

        reg.replace_executor("recall_memory", real_recall)

        async def go():
            tool = reg.get("recall_memory")
            return await tool.executor({"query": "hello"}, ToolContext())

        out = asyncio.run(go())
        self.assertEqual(out, {"matches": ["hello"]})

    def test_replace_unknown_raises(self) -> None:
        reg = ToolRegistry()
        with self.assertRaises(KeyError):
            reg.replace_executor("nope", lambda a, c: None)  # type: ignore[arg-type]

    def test_schemas_filter_by_allow_list(self) -> None:
        reg = build_default_registry()
        schemas = reg.schemas(["recall_memory"])
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "recall_memory")

    def test_parse_tool_arguments_handles_garbage(self) -> None:
        self.assertEqual(parse_tool_arguments(""), {})
        self.assertEqual(parse_tool_arguments(None), {})
        self.assertEqual(parse_tool_arguments("not json"), {})
        self.assertEqual(parse_tool_arguments('{"x":1}'), {"x": 1})
        self.assertEqual(parse_tool_arguments("[1,2,3]"), {})


class ToolLoopTests(unittest.TestCase):
    def _make_chain(self) -> FallbackChain:
        return FallbackChain(RateTracker())

    def _make_targets(self):
        # The chain forwards kwargs to provider.generate; we replace the
        # chain itself with a stub so targets are inert.
        return [(SimpleNamespace(), "test-model")]

    def test_loop_returns_immediately_when_no_tool_calls(self) -> None:
        reg = build_default_registry()
        chain = self._make_chain()
        chain.generate = AsyncMock(
            return_value=ProviderResponse(text="hi", tool_calls=None)
        )

        async def go():
            return await run_tool_loop(
                fallback_chain=chain,
                targets=self._make_targets(),
                messages=[{"role": "user", "content": "hello"}],
                registry=reg,
                context=ToolContext(user_id="U1"),
            )

        result = asyncio.run(go())
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_calls_executed, [])
        self.assertEqual(result.response.text, "hi")
        self.assertEqual(chain.generate.await_count, 1)

    def test_loop_executes_tool_then_returns_final_text(self) -> None:
        reg = build_default_registry()

        async def fake_recall(args, ctx):
            return {"matches": [f"recalled: {args['query']}"]}

        reg.replace_executor("recall_memory", fake_recall)

        chain = self._make_chain()
        # Round 1: model emits a tool_call. Round 2: model returns text.
        responses = [
            ProviderResponse(
                text=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "recall_memory",
                            "arguments": '{"query": "birthday"}',
                        },
                    }
                ],
                raw_message={"role": "assistant", "content": None},
            ),
            ProviderResponse(text="based on my recall: …", tool_calls=None),
        ]
        chain.generate = AsyncMock(side_effect=responses)

        messages = [{"role": "user", "content": "do you remember my birthday?"}]

        async def go():
            return await run_tool_loop(
                fallback_chain=chain,
                targets=self._make_targets(),
                messages=messages,
                registry=reg,
                context=ToolContext(user_id="U1"),
            )

        result = asyncio.run(go())
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.tool_calls_executed, ["recall_memory"])
        self.assertEqual(result.response.text, "based on my recall: …")
        # Loop should have appended assistant + tool messages
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        tool_msg = messages[-1]
        self.assertEqual(tool_msg["tool_call_id"], "call_1")
        self.assertEqual(tool_msg["name"], "recall_memory")
        self.assertIn("recalled: birthday", tool_msg["content"])

    def test_loop_returns_error_for_unknown_tool(self) -> None:
        reg = build_default_registry()
        chain = self._make_chain()
        chain.generate = AsyncMock(
            side_effect=[
                ProviderResponse(
                    text=None,
                    tool_calls=[
                        {
                            "id": "x",
                            "type": "function",
                            "function": {
                                "name": "no_such_tool",
                                "arguments": "{}",
                            },
                        }
                    ],
                ),
                ProviderResponse(text="ok", tool_calls=None),
            ]
        )

        messages = [{"role": "user", "content": "?"}]

        async def go():
            return await run_tool_loop(
                fallback_chain=chain,
                targets=self._make_targets(),
                messages=messages,
                registry=reg,
                context=ToolContext(),
            )

        asyncio.run(go())
        tool_msg = messages[-1]
        self.assertEqual(tool_msg["role"], "tool")
        payload = json.loads(tool_msg["content"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("no_such_tool", payload["reason"])

    def test_loop_respects_allowed_tools(self) -> None:
        reg = build_default_registry()
        chain = self._make_chain()
        chain.generate = AsyncMock(
            return_value=ProviderResponse(text="ok", tool_calls=None)
        )

        async def go():
            return await run_tool_loop(
                fallback_chain=chain,
                targets=self._make_targets(),
                messages=[{"role": "user", "content": "hi"}],
                registry=reg,
                context=ToolContext(),
                allowed_tools=["recall_memory"],
            )

        asyncio.run(go())
        forwarded_tools = chain.generate.await_args.kwargs["tools"]
        self.assertEqual(len(forwarded_tools), 1)
        self.assertEqual(
            forwarded_tools[0]["function"]["name"], "recall_memory"
        )

    def test_loop_caps_at_max_iterations(self) -> None:
        reg = build_default_registry()
        chain = self._make_chain()
        infinite_tool_call = ProviderResponse(
            text=None,
            tool_calls=[
                {
                    "id": "loop",
                    "type": "function",
                    "function": {
                        "name": "recall_memory",
                        "arguments": '{"query":"x"}',
                    },
                }
            ],
        )
        chain.generate = AsyncMock(return_value=infinite_tool_call)

        async def go():
            return await run_tool_loop(
                fallback_chain=chain,
                targets=self._make_targets(),
                messages=[{"role": "user", "content": "hi"}],
                registry=reg,
                context=ToolContext(),
                max_iterations=2,
            )

        result = asyncio.run(go())
        self.assertEqual(result.iterations, 2)
        self.assertEqual(chain.generate.await_count, 2)

    def test_empty_registry_raises(self) -> None:
        chain = self._make_chain()

        async def go():
            return await run_tool_loop(
                fallback_chain=chain,
                targets=self._make_targets(),
                messages=[{"role": "user", "content": "hi"}],
                registry=ToolRegistry(),
                context=ToolContext(),
            )

        with self.assertRaises(ToolLoopError):
            asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
