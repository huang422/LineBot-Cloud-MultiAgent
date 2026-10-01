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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from src.services.memory_service import (
    EPISODE_SNIPPET_CHARS,
    PROFILE_FACT_MAX_CHARS,
    PROFILE_FACTS_MAX,
    PROFILE_FACTS_MAX_CHARS,
)

ToolExecutor = Callable[[dict, "ToolContext"], Awaitable[Any]]

# Limits the real executors (src/runtime/composition.py) apply; the tool
# descriptions quote them, so the model knows what it gets back.
RECALL_SNIPPET_CHARS = 600
RECALL_SUMMARY_CHARS = 300
WEB_SEARCH_RESULTS = 5
WEB_SEARCH_CONTEXT_CHARS = 4000
# Searches one reply may run through the web_search tool, every tool round
# of the turn counted (the tool loop enforces it); each costs a model round
# trip inside the reply window and a Tavily credit.
WEB_SEARCH_TOOL_MAX_CALLS_PER_TURN = 2


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
    # Runs one turn's tool loop may make, every round counted; a call past
    # it is answered "limit" without being run. None: no cap.
    max_calls_per_turn: int | None = None

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
        self._tools[name] = replace(existing, executor=executor)

    def remove(self, name: str) -> None:
        """Stop offering tool *name* (nothing happens if it is not there)."""
        self._tools.pop(name, None)

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
                "What to look for: a short topic phrase in the user's "
                "language naming the person, event or preference in "
                "question. It is matched by meaning, not by exact words."
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
            "description": (
                "How sure you are that the sender stated these facts "
                "about themself and that they are durable, 0.0-1.0; "
                "below 0.7 the call is rejected."
            ),
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
                "Search query in the user's language, including the "
                "names, places and dates that narrow it down."
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
                "Search this chat's archived conversation. When the "
                "recent-message window is compacted, the conversation it "
                "held is archived as an episode together with the long-term "
                "summary as it stood then; a chat gets at most one episode "
                "an hour by default, so one episode can hold several "
                f"compactions (the newest {EPISODE_SNIPPET_CHARS} characters of "
                "them). Episodes expire after about 90 days. The query is "
                "embedded and the nearest episodes of this chat only are "
                "returned (this 1:1 chat, group or room; never other chats). "
                "There is no relevance cut-off: up to k episodes come back "
                "whenever any exist, so check that they are actually about "
                "the query before using them. Each match has snippet (up to "
                f"{RECALL_SNIPPET_CHARS} characters of the archived messages, "
                "one 'Speaker: text' line each, taken around the first place "
                "the query's words appear, else the newest part; '…' marks a "
                "cut), summary (up to "
                f"{RECALL_SUMMARY_CHARS} characters of the summary then), ts "
                "(Unix seconds when it was saved) and score (similarity; null "
                "on the Firestore backend). Status is ok, "
                "empty (nothing found: no matching episodes, or memory "
                "search is unavailable right now; never tell the user they "
                "have no history because of it), noop (empty query or no "
                "chat id) or error. Use it when the user refers to "
                "something from earlier that is neither in the current "
                "long-term summary nor in the recent messages, or when a "
                "good answer depends on something the user said before "
                "(their work, interests or preferences) that you cannot see "
                "here. What was said since the last compaction is already in "
                "the conversation, and the newest compactions may not be "
                "archived yet (the long-term summary covers them), so this "
                "searches older conversation only. Do not use it "
                "for general knowledge, current events or a new topic, or "
                "when the conversation already has what you need."
            ),
            parameters=_RECALL_MEMORY_PARAMS,
            executor=_stub_recall_memory,
        )
    )
    registry.register(
        ToolDefinition(
            name="update_user_profile",
            description=(
                "Save durable facts about the user who sent the current "
                "message (name or nickname, job, location, language, "
                "long-running interests or preferences) to that user's "
                "profile, which is shared across every chat they have with "
                "this bot and shown to you in later conversations (the "
                f"newest {PROFILE_FACTS_MAX} facts, up to {PROFILE_FACTS_MAX_CHARS} characters, in a "
                "block marked [使用者 profile，僅供參考]; in a group, do not bring them up unless the "
                "sender does). Record only facts the sender states about "
                "themself, including corrections to earlier facts; never "
                "record facts about other people (group members, family, "
                "partners, friends or colleagues), health or mental-health "
                "details, guesses drawn from context, one-off moods, "
                "temporary plans or jokes. Save all "
                "new facts from one message in a single call, each as one short "
                f"line (longer ones are cut to {PROFILE_FACT_MAX_CHARS} characters). New facts are "
                "merged with the stored ones (case-insensitive duplicates "
                "are dropped and only the 20 most recent are kept). A call "
                "with confidence below 0.7 is rejected. Returns status ok "
                "(with the facts added and total_facts), rejected, noop (no "
                "user id or no facts) or error."
            ),
            parameters=_UPDATE_USER_PROFILE_PARAMS,
            executor=_stub_update_user_profile,
        )
    )
    registry.register(
        ToolDefinition(
            name="web_search",
            description=(
                "Search the web for questions whose answer changes over "
                "time or is newer than your training data: news, prices "
                "and exchange rates, scores, weather, schedules, recent "
                "releases or events. Also use it when the user asks you to "
                "look something up online, or when you are not sure an "
                "answer about a changing topic is still correct. Do not use "
                "it for general knowledge, programming syntax, historical "
                "facts or math; for small talk, emotions, writing or "
                "translation; or when the user has already given the "
                "information needed. Runs one web search (a hosted search "
                f"API) for the top {WEB_SEARCH_RESULTS} results and returns "
                f"context: at most {WEB_SEARCH_CONTEXT_CHARS} characters of "
                "result titles, URLs, publication dates when known and "
                "snippets, after a short generated summary when the search "
                "has one. Results are "
                "not filtered by date, so put "
                "the date, the year or a word like 'latest' in the query "
                "when recency matters; the current time is in the system "
                "prompt. It cannot open or read a specific URL. One reply "
                f"may search at most {WEB_SEARCH_TOOL_MAX_CALLS_PER_TURN} times; "
                "put everything you need into those queries. Status is ok, "
                "empty (nothing found, or no search could answer right now: "
                "answer from what you know and say it may be out of date), "
                "noop (empty query), limit (this reply "
                "already used its searches: answer with what you have), "
                "unavailable (search not configured: answer as for empty) or "
                "error (the search itself crashed)."
            ),
            parameters=_WEB_SEARCH_PARAMS,
            executor=_stub_web_search,
            max_calls_per_turn=WEB_SEARCH_TOOL_MAX_CALLS_PER_TURN,
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
