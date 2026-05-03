"""OpenAI-compatible tool definitions used by the agentic tool loop.

Each tool is described by:

* a JSON schema that we send to the LLM (under ``tools=[...]``), and
* an async ``executor`` that runs when the LLM emits a matching
  ``tool_calls`` directive.

The schemas follow the OpenAI / NVIDIA / OpenRouter ``function`` style::

    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "...",
            "parameters": { JSON-schema },
        },
    }

Phase C ships the schemas and a shared registry. Phase B replaces the
``recall_memory`` and ``update_user_profile`` executor stubs with real
Firestore-backed implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

ToolExecutor = Callable[[dict, "ToolContext"], Awaitable[Any]]


@dataclass(frozen=True)
class ToolContext:
    """Per-request context handed to every tool executor.

    Tools should treat every attribute as optional and degrade
    gracefully when something is missing — the orchestrator may invoke
    the loop in contexts where ``user_id`` is empty (DM with a
    non-friend, anonymous webhook replay, etc.).
    """

    user_id: str = ""
    chat_id: str = ""
    source_type: str = "user"
    locale: str = "zh-TW"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    executor: ToolExecutor

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Mutable registry of tool definitions used by the agentic loop."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name:
            raise ValueError("Tool name must not be empty")
        self._tools[tool.name] = tool

    def replace_executor(self, name: str, executor: ToolExecutor) -> None:
        existing = self._tools.get(name)
        if existing is None:
            raise KeyError(f"Tool {name!r} is not registered")
        self._tools[name] = ToolDefinition(
            name=existing.name,
            description=existing.description,
            parameters=existing.parameters,
            executor=executor,
        )

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        if names is None:
            return [t.schema() for t in self._tools.values()]
        out: list[dict] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                out.append(tool.schema())
        return out


# ── Tool executor stubs ──────────────────────────────────────
# These run if the loop is invoked before Phase B fills in the real
# implementations. They return JSON-serialisable structures so the LLM
# always receives a deterministic ``tool`` message back.


async def _stub_recall_memory(args: dict, ctx: ToolContext) -> dict:
    query = (args.get("query") or "").strip()
    return {
        "status": "unavailable",
        "query": query,
        "matches": [],
        "note": (
            "Episodic memory recall is not enabled yet. Phase B will wire "
            "this tool to Firestore vector search."
        ),
    }


async def _stub_update_user_profile(args: dict, ctx: ToolContext) -> dict:
    facts = args.get("facts") or []
    if not isinstance(facts, list):
        facts = [facts]
    return {
        "status": "noop",
        "received_facts": [str(f) for f in facts if f is not None],
        "user_id": ctx.user_id,
        "note": "User profile updates are disabled until Phase B.",
    }


async def _stub_web_search(args: dict, ctx: ToolContext) -> dict:
    return {
        "status": "unavailable",
        "query": str(args.get("query") or "").strip(),
        "note": (
            "Web search executor not wired in this process. The "
            "orchestrator fast-rule path still serves news / URL / "
            "finance queries."
        ),
    }


# ── Schemas for the tools we ship ────────────────────────────

_RECALL_MEMORY_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Topic or keywords to recall from the user's older "
                "conversation episodes. Keep it concise (<= 30 chars)."
            ),
        },
        "k": {
            "type": "integer",
            "description": "How many top matches to return (default 3).",
            "minimum": 1,
            "maximum": 5,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_UPDATE_USER_PROFILE_PARAMS = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "description": (
                "Long-term facts or preferences worth remembering "
                "across chats (e.g. nickname, hobbies, location). "
                "Skip ephemeral details."
            ),
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "confidence": {
            "type": "number",
            "description": "0.0 – 1.0; only call when >= 0.7.",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

_WEB_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Web search query in the user's language. Use only "
                "when the answer requires fresh information that the "
                "model cannot know (news, prices, sports scores, "
                "release dates, current weather, recent events) AND "
                "no URL is present in the user message. Skip for "
                "general knowledge or chit-chat."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def build_default_registry() -> ToolRegistry:
    """Return a registry pre-populated with stub-backed tools.

    The orchestrator wires this in at startup; Phase B then swaps the
    executors via :meth:`ToolRegistry.replace_executor` once Firestore
    Vector Search and the embedding service are configured.
    """
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="recall_memory",
            description=(
                "Look up older conversation snippets that are no longer in "
                "the recent window or summary. Use for callbacks like "
                "'remember when we…' or when the user references something "
                "older than the last few turns."
            ),
            parameters=_RECALL_MEMORY_PARAMS,
            executor=_stub_recall_memory,
        )
    )
    registry.register(
        ToolDefinition(
            name="update_user_profile",
            description=(
                "Persist long-term facts/preferences about the user that "
                "should survive across chats (nickname, hobbies, "
                "language, location, work). Only call when you are highly "
                "confident the fact is durable."
            ),
            parameters=_UPDATE_USER_PROFILE_PARAMS,
            executor=_stub_update_user_profile,
        )
    )
    registry.register(
        ToolDefinition(
            name="web_search",
            description=(
                "Search the web for fresh, time-sensitive information. "
                "Use sparingly — most general questions can be answered "
                "from your own knowledge. Prefer this when the answer "
                "obviously depends on current data."
            ),
            parameters=_WEB_SEARCH_PARAMS,
            executor=_stub_web_search,
        )
    )
    return registry


def parse_tool_arguments(raw: str | None) -> dict:
    """Parse the JSON-encoded ``arguments`` string from a tool_call.

    Returns an empty dict on any decode failure so the executor can
    short-circuit without raising. We never trust the model's output to
    be valid JSON.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
