"""Phase B tests: embedding service, episodic memory, recall + profile tools."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings
from src.services.memory_backends import InMemoryBackend
from src.services.memory_service import MemoryService


def _make_settings() -> Settings:
    return Settings(_env_file=None)


class FakeEmbeddingService:
    """Deterministic fake — converts each char to a code-point vector."""

    is_configured = True
    model = "fake-embed"

    def __init__(self) -> None:
        self.call_log: list[tuple[str, str]] = []

    async def embed_text(self, text: str) -> list[float]:
        self.call_log.append(("query", text))
        return self._encode(text)

    async def embed_passage(self, text: str) -> list[float]:
        self.call_log.append(("passage", text))
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> list[float]:
        # Tiny 8-dim "embedding" — bag-of-chars hash. Enough for the
        # in-memory cosine search to discriminate between strings in
        # tests without dragging real NIM into CI.
        vec = [0.0] * 8
        for ch in text:
            vec[ord(ch) % 8] += 1.0
        return vec


class EpisodeStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_inmemory_save_and_search_returns_top_match(self):
        backend = InMemoryBackend()
        embed = FakeEmbeddingService()
        chocolate = await embed.embed_passage("最愛吃巧克力")
        sushi = await embed.embed_passage("我喜歡壽司")
        await backend.save_episode(
            "user::U1",
            {"summary": "聊天提到巧克力", "ts": 1.0, "embedding": chocolate},
        )
        await backend.save_episode(
            "user::U1",
            {"summary": "聊到壽司餐廳", "ts": 2.0, "embedding": sushi},
        )

        query = await embed.embed_text("巧克力推薦")
        results = await backend.search_episodes("user::U1", query, k=2)

        self.assertEqual(len(results), 2)
        # The chocolate episode must be ranked first.
        self.assertIn("巧克力", results[0]["summary"])
        # Embeddings must be stripped from returned snapshots.
        self.assertNotIn("embedding", results[0])
        self.assertIn("score", results[0])

    async def test_search_episodes_returns_empty_when_no_data(self):
        backend = InMemoryBackend()
        results = await backend.search_episodes(
            "user::U1", [0.1, 0.2, 0.3], k=3
        )
        self.assertEqual(results, [])

    async def test_delete_chat_also_purges_episodes(self):
        backend = InMemoryBackend()
        await backend.save_episode(
            "user::U1",
            {"summary": "x", "embedding": [1.0, 0.0]},
        )
        self.assertIn("user::U1", backend.episodes)
        await backend.delete_chat("user::U1")
        self.assertNotIn("user::U1", backend.episodes)

    async def test_persist_episode_sets_firestore_ttl_field(self):
        backend = InMemoryBackend()
        service = MemoryService(
            _make_settings(),
            backend=backend,
            embedding_service=FakeEmbeddingService(),
        )

        await service._persist_episode_safe(
            key="user::U1",
            source_type="user",
            chat_id="U1",
            summary_text="記得使用者喜歡巧克力",
            recent_snapshot=[],
            summary_version=1,
        )

        episode = backend.episodes["user::U1"][0]
        self.assertIn("expires_at", episode)
        self.assertGreater(episode["expires_at"].timestamp(), episode["ts"])


class RecallEpisodesTests(unittest.IsolatedAsyncioTestCase):
    async def test_recall_episodes_returns_matching_episode(self):
        embed = FakeEmbeddingService()
        backend = InMemoryBackend()
        # Pre-seed an episode for chat user::U1
        emb = await embed.embed_passage("我們聊過鋼琴課")
        await backend.save_episode(
            "user::U1",
            {"summary": "鋼琴課討論", "ts": 10.0, "embedding": emb},
        )
        service = MemoryService(
            _make_settings(), backend=backend, embedding_service=embed
        )

        results = await service.recall_episodes(
            source_type="user", chat_id="U1", query="鋼琴", k=3
        )

        self.assertEqual(len(results), 1)
        self.assertIn("鋼琴", results[0]["summary"])
        # Query path triggers a "query" embedding call
        self.assertTrue(any(t == "query" for t, _ in embed.call_log))

    async def test_recall_episodes_returns_empty_when_embedding_disabled(self):
        backend = InMemoryBackend()
        service = MemoryService(_make_settings(), backend=backend)
        results = await service.recall_episodes(
            source_type="user", chat_id="U1", query="anything"
        )
        self.assertEqual(results, [])

    async def test_recall_episodes_returns_empty_for_blank_inputs(self):
        embed = FakeEmbeddingService()
        service = MemoryService(
            _make_settings(),
            backend=InMemoryBackend(),
            embedding_service=embed,
        )
        self.assertEqual(
            await service.recall_episodes(
                source_type="user", chat_id="", query="abc"
            ),
            [],
        )
        self.assertEqual(
            await service.recall_episodes(
                source_type="user", chat_id="U1", query="   "
            ),
            [],
        )


class UpdateUserFactsTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_user_facts_merges_and_deduplicates(self):
        backend = InMemoryBackend()
        backend.user_profiles["U1"] = {
            "user_id": "U1",
            "facts": ["喜歡爵士樂"],
        }
        service = MemoryService(_make_settings(), backend=backend)

        result = await service.update_user_facts(
            user_id="U1",
            facts=["喜歡爵士樂", "住台北"],
            confidence=0.9,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_facts"], 2)
        self.assertEqual(set(backend.user_profiles["U1"]["facts"]),
                         {"喜歡爵士樂", "住台北"})

    async def test_update_user_facts_rejects_low_confidence(self):
        service = MemoryService(_make_settings(), backend=InMemoryBackend())
        result = await service.update_user_facts(
            user_id="U1", facts=["x"], confidence=0.3
        )
        self.assertEqual(result["status"], "rejected")

    async def test_update_user_facts_noop_for_blank_inputs(self):
        service = MemoryService(_make_settings(), backend=InMemoryBackend())
        self.assertEqual(
            (await service.update_user_facts(user_id="", facts=["x"]))["status"],
            "noop",
        )
        self.assertEqual(
            (await service.update_user_facts(user_id="U1", facts=[]))["status"],
            "noop",
        )

    async def test_update_user_facts_caps_at_max(self):
        backend = InMemoryBackend()
        service = MemoryService(_make_settings(), backend=backend)
        # 25 distinct facts; max_facts default = 20
        facts = [f"fact-{i}" for i in range(25)]
        result = await service.update_user_facts(
            user_id="U1", facts=facts, confidence=1.0, max_facts=20
        )
        self.assertEqual(result["total_facts"], 20)


class EmbeddingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_text_returns_none_when_unconfigured(self):
        from src.services.embedding_service import EmbeddingService

        svc = EmbeddingService(api_key="")
        try:
            self.assertFalse(svc.is_configured)
            self.assertIsNone(await svc.embed_text("hello"))
        finally:
            await svc.close()

    async def test_embed_text_parses_response(self):
        from src.services.embedding_service import EmbeddingService

        svc = EmbeddingService(api_key="fake")
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            with patch.object(svc, "_client") as mock_client:
                mock_client.post = AsyncMock(return_value=mock_response)
                result = await svc.embed_text("hi")
            self.assertEqual(result, [0.1, 0.2, 0.3])
        finally:
            await svc.close()

    async def test_embed_text_handles_http_error(self):
        from src.services.embedding_service import EmbeddingService

        svc = EmbeddingService(api_key="fake")
        try:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "boom"
            with patch.object(svc, "_client") as mock_client:
                mock_client.post = AsyncMock(return_value=mock_response)
                result = await svc.embed_text("hi")
            self.assertIsNone(result)
        finally:
            await svc.close()


class ChatAgentToolLoopTests(unittest.IsolatedAsyncioTestCase):
    """Verify ChatAgent uses the tool loop when registry is set."""

    async def test_chat_agent_falls_back_to_plain_generate_without_registry(self):
        from src.agents.chat_agent import ChatAgent
        from src.models.agent_request import AgentRequest

        settings = _make_settings()
        chain = MagicMock()
        chain.generate = AsyncMock(return_value=MagicMock(text="hi", model="m"))
        agent = ChatAgent(settings, chain, targets=[(MagicMock(), "m")])
        agent.system_prompt = ""

        result = await agent.process(AgentRequest(text="你好"))

        self.assertEqual(result.text, "hi")
        chain.generate.assert_awaited_once()
        # Ensure no tools= kwarg leaked into the call
        kwargs = chain.generate.await_args.kwargs
        self.assertNotIn("tools", kwargs)

    async def test_chat_agent_runs_tool_loop_when_registry_present(self):
        from src.agents.chat_agent import ChatAgent
        from src.agents.tools import (
            ToolDefinition,
            ToolRegistry,
        )
        from src.models.agent_request import AgentRequest
        from src.providers.openrouter_provider import ProviderResponse

        async def _exec(args, ctx):
            return {"status": "ok", "matches": [{"summary": "old"}]}

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="recall_memory",
                description="x",
                parameters={"type": "object", "properties": {}},
                executor=_exec,
            )
        )

        first = ProviderResponse(
            text="",
            model="m",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "recall_memory",
                        "arguments": '{"query":"old"}',
                    },
                }
            ],
            raw_message={"role": "assistant", "content": "", "tool_calls": []},
        )
        second = ProviderResponse(text="final answer", model="m")

        chain = MagicMock()
        chain.generate = AsyncMock(side_effect=[first, second])

        settings = _make_settings()
        agent = ChatAgent(
            settings,
            chain,
            targets=[(MagicMock(), "m")],
            tool_registry=registry,
        )
        agent.system_prompt = ""

        result = await agent.process(
            AgentRequest(text="還記得我說過甚麼嗎", user_id="U1", group_id="U1")
        )
        self.assertEqual(result.text, "final answer")
        self.assertEqual(chain.generate.await_count, 2)


if __name__ == "__main__":
    unittest.main()
