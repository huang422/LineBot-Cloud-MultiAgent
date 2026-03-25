from __future__ import annotations

import asyncio
import base64
from collections import deque
from datetime import datetime
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import main as main_module
from src.agents import orchestrator as orchestrator_module
from src.agents import web_search_agent as web_search_agent_module
from src.agents.chat_agent import ChatAgent
from src.agents.image_gen_agent import ImageGenAgent
from src.agents.orchestrator import Orchestrator
from src.agents.vision_agent import VisionAgent
from src.agents.web_search_agent import WebSearchAgent
from src.config import Settings
from src.handlers import webhook_handler
from src.models.agent_request import AgentRequest, InputType
from src.models.conversation import ConversationMessage
from src.processors import input_processor as input_processor_module
from src.processors import output_processor as output_processor_module
from src.processors import tts_processor as tts_processor_module
from src.providers.fallback_chain import (
    AllModelsRateLimitedError,
    AllProvidersFailedError,
    FallbackChain,
)
from src.providers.openrouter_provider import (
    OpenRouterProvider,
    ProviderError,
    ProviderResponse,
    RateLimitError,
    parse_openai_response,
)
from src.services import conversation_service as conversation_service_module
from src.services import image_service as image_service_module
from src.services import message_cache_service as message_cache_service_module
from src.services.storage_service import UploadedMedia
from src.services.message_cache_service import CachedMessage
from src.services import rate_limit_service as rate_limit_service_module
from src.services import web_search_service as web_search_service_module
from src.utils.rate_tracker import RateTracker
from PIL import Image


class OpenRouterProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_openai_response_concatenates_multipart_text(self) -> None:
        text, images, reasoning = parse_openai_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "第一段"},
                                {"type": "text", "text": "第二段"},
                                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                            ]
                        }
                    }
                ]
            }
        )

        self.assertEqual(text, "第一段第二段")
        self.assertEqual(images, ["https://example.com/image.png"])
        self.assertIsNone(reasoning)

    async def test_generate_includes_reasoning_for_text_requests(self) -> None:
        provider = OpenRouterProvider(
            "test-key",
            RateTracker(),
            reasoning_enabled=True,
            reasoning_effort="high",
            reasoning_exclude=True,
            thinking_budget=1024,
        )

        class FakeResponse:
            status_code = 200
            headers = {}
            text = "ok"

            def json(self):
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "model": "nvidia/nemotron-3-super-120b-a12b:free",
                    "usage": {},
                }

        fake_post = AsyncMock(return_value=FakeResponse())
        provider._client = SimpleNamespace(post=fake_post)

        await provider.generate(
            "nvidia/nemotron-3-super-120b-a12b:free",
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(
            fake_post.await_args.kwargs["json"]["reasoning"],
            {"enabled": True, "effort": "high", "exclude": True},
        )
        self.assertEqual(fake_post.await_args.kwargs["json"]["max_tokens"], 3072)

    async def test_generate_skips_reasoning_for_image_requests(self) -> None:
        provider = OpenRouterProvider(
            "test-key",
            RateTracker(),
            reasoning_enabled=True,
            reasoning_effort="high",
            reasoning_exclude=True,
        )

        class FakeResponse:
            status_code = 200
            headers = {}
            text = "ok"

            def json(self):
                return {
                    "choices": [{"message": {"images": ["data:image/png;base64,abc"]}}],
                    "model": "sourceful/riverflow-v2-pro",
                    "usage": {},
                }

        fake_post = AsyncMock(return_value=FakeResponse())
        provider._client = SimpleNamespace(post=fake_post)

        await provider.generate(
            "sourceful/riverflow-v2-pro",
            [{"role": "user", "content": "draw a dog"}],
            modalities=["image"],
        )

        self.assertNotIn("reasoning", fake_post.await_args.kwargs["json"])

    async def test_generate_continues_when_reasoning_tokens_are_missing(self) -> None:
        provider = OpenRouterProvider(
            "test-key",
            RateTracker(),
            reasoning_enabled=True,
            reasoning_effort="high",
            reasoning_exclude=True,
        )

        class FakeResponse:
            status_code = 200
            headers = {}
            text = "ok"

            def json(self):
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "model": "nvidia/nemotron-3-super-120b-a12b:free",
                    "usage": {"reasoning_tokens": 0},
                }

        provider._client = SimpleNamespace(post=AsyncMock(return_value=FakeResponse()))

        response = await provider.generate(
            "nvidia/nemotron-3-super-120b-a12b:free",
            [{"role": "user", "content": "hello"}],
            require_reasoning_tokens=True,
        )

        self.assertEqual(response.text, "ok")


class NvidiaProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_continues_when_reasoning_tokens_are_missing(self) -> None:
        from src.providers.nvidia_provider import NvidiaProvider

        provider = NvidiaProvider(
            "test-key",
            RateTracker(),
            thinking_enabled=True,
            thinking_budget=1024,
        )

        class FakeResponse:
            status_code = 200
            text = "ok"

            def json(self):
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "model": "qwen/qwen3.5-397b-a17b",
                    "usage": {"reasoning_tokens": 0},
                }

        provider._client = SimpleNamespace(post=AsyncMock(return_value=FakeResponse()))

        response = await provider.generate(
            "qwen/qwen3.5-397b-a17b",
            [{"role": "user", "content": "hello"}],
            require_reasoning_tokens=True,
        )

        self.assertEqual(response.text, "ok")

    async def test_fallback_chain_skips_rate_limited_dedicated_thinking_model(self) -> None:
        from src.providers.nvidia_provider import NvidiaProvider

        tracker = RateTracker()
        provider = NvidiaProvider(
            "test-key",
            tracker,
            thinking_enabled=True,
            thinking_budget=1024,
            thinking_model="qwen/qwen3.5-122b-a10b",
        )
        provider._client = SimpleNamespace(post=AsyncMock())
        tracker.record_limit_hit("qwen/qwen3.5-122b-a10b", 60)

        backup = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(text="backup", model="model-2")
            )
        )
        chain = FallbackChain(tracker)

        response = await chain.generate(
            targets=[
                (provider, "qwen/qwen3.5-397b-a17b"),
                (backup, "model-2"),
            ],
            messages=[{"role": "user", "content": "請幫我分析這個問題"}],
        )

        self.assertEqual(response.text, "backup")
        provider._client.post.assert_not_awaited()
        backup.generate.assert_awaited_once()


class WebhookTriggerTests(unittest.TestCase):
    def test_group_message_ignores_other_user_mentions(self) -> None:
        settings = Settings(
            _env_file=None,
            bot_name="小助手",
            line_bot_user_id="U_BOT",
        )
        event = {
            "type": "message",
            "source": {"type": "group"},
            "message": {
                "type": "text",
                "text": "@Tom 你好",
                "mention": {
                    "mentionees": [
                        {"text": "@Tom", "userId": "U_OTHER"},
                    ]
                },
            },
        }

        with patch.object(webhook_handler, "get_settings", return_value=settings):
            self.assertFalse(webhook_handler.should_handle(event))

    def test_group_message_handles_bot_mentions_only(self) -> None:
        settings = Settings(
            _env_file=None,
            bot_name="小助手",
            line_bot_user_id="U_BOT",
        )
        event = {
            "type": "message",
            "source": {"type": "group"},
            "message": {
                "type": "text",
                "text": "@小助手 幫我翻譯",
                "mention": {
                    "mentionees": [
                        {"text": "@小助手", "userId": "U_BOT"},
                    ]
                },
            },
        }

        with patch.object(webhook_handler, "get_settings", return_value=settings):
            self.assertTrue(webhook_handler.should_handle(event))

    def test_group_message_ignores_other_bot_mentions(self) -> None:
        settings = Settings(
            _env_file=None,
            bot_name="小助手",
            line_bot_user_id="U_BOT",
        )
        event = {
            "type": "message",
            "source": {"type": "group"},
            "message": {
                "type": "text",
                "text": "@別的機器人 你好",
                "mention": {
                    "mentionees": [
                        {"type": "bot", "text": "@別的機器人", "userId": "U_OTHER_BOT"},
                    ]
                },
            },
        }

        with patch.object(webhook_handler, "get_settings", return_value=settings):
            self.assertFalse(webhook_handler.should_handle(event))


class ConversationRecordingTests(unittest.TestCase):
    def test_record_conversation_skips_undelivered_assistant_reply(self) -> None:
        """record_conversation only records assistant replies (user messages
        are handled proactively by _record_group_message in main.py)."""
        recorded: list[tuple[str, str, str]] = []

        class FakeConversationService:
            def add_message(self, chat_id: str, msg) -> None:
                recorded.append((chat_id, msg.role, msg.content))

        request = AgentRequest(user_id="U1", group_id="G1", text="hello")

        with patch.object(
            webhook_handler,
            "get_conversation_service",
            return_value=FakeConversationService(),
        ):
            webhook_handler.record_conversation(
                request,
                "assistant reply",
                assistant_delivered=False,
            )

        # No assistant recorded because assistant_delivered=False;
        # user messages are not recorded here (handled elsewhere).
        self.assertEqual(recorded, [])


class WebSearchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_search_agent_returns_safe_message_when_search_unavailable(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(side_effect=AssertionError("LLM should not run"))
        )
        agent = WebSearchAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(text="今天新聞", output_format="text")

        fake_service = SimpleNamespace(is_configured=False, is_quota_available=False)

        with patch.object(
            web_search_agent_module,
            "get_web_search_service",
            return_value=fake_service,
        ):
            response = await agent.process(request)

        self.assertIn("搜尋服務暫時不可用", response.text or "")
        self.assertEqual(response.agent_name, "web_search")

    async def test_web_search_agent_does_not_block_on_app_quota_flag(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(
                    text="整理好的搜尋結果",
                    model="nvidia/test",
                )
            )
        )
        agent = WebSearchAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(text="今天新聞", output_format="text")

        fake_service = SimpleNamespace(
            is_configured=True,
            is_quota_available=False,
            search=AsyncMock(
                return_value=SimpleNamespace(
                    has_results=True,
                    answer="ok",
                    to_context_text=lambda: "search context",
                )
            ),
        )

        with patch.object(
            web_search_agent_module,
            "get_web_search_service",
            return_value=fake_service,
        ):
            response = await agent.process(request)

        self.assertEqual(response.text, "整理好的搜尋結果")
        fake_service.search.assert_awaited()
        fallback.generate.assert_awaited_once()

    async def test_web_search_agent_extracts_webpage_content_from_urls(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(
                    text="這篇文章的重點如下",
                    model="nvidia/test",
                )
            )
        )
        agent = WebSearchAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(text="幫我整理這篇 https://example.com/post", output_format="text")

        fake_service = SimpleNamespace(
            is_configured=True,
            is_quota_available=True,
            extract=AsyncMock(
                return_value=web_search_service_module.ExtractResponse(
                    results=[
                        web_search_service_module.ExtractResult(
                            url="https://example.com/post",
                            content="這是網頁正文內容",
                        )
                    ],
                    failed_urls=[],
                )
            ),
            search=AsyncMock(),
        )

        with patch.object(
            web_search_agent_module,
            "get_web_search_service",
            return_value=fake_service,
        ):
            response = await agent.process(request)

        self.assertEqual(response.text, "這篇文章的重點如下")
        fake_service.extract.assert_awaited_once_with(["https://example.com/post"])
        fake_service.search.assert_not_called()

        generate_call = fallback.generate.await_args
        messages = generate_call.kwargs["messages"]
        self.assertEqual(messages[-1]["content"], "幫我整理這篇 https://example.com/post")
        self.assertIn("從使用者指定網址擷取的網頁內容", messages[-2]["content"])
        self.assertIn("這是網頁正文內容", messages[-2]["content"])

    async def test_web_search_agent_prefixes_current_datetime_in_search_query(self) -> None:
        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 3, 25, 14, 30, tzinfo=tz)

        fallback = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(
                    text="整理好的搜尋結果",
                    model="nvidia/test",
                )
            )
        )
        agent = WebSearchAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(text="今天新聞", output_format="text")

        fake_service = SimpleNamespace(
            is_configured=True,
            is_quota_available=True,
            search=AsyncMock(
                return_value=SimpleNamespace(
                    has_results=True,
                    answer="ok",
                    to_context_text=lambda: "search context",
                )
            ),
        )

        with patch.object(
            web_search_agent_module,
            "get_web_search_service",
            return_value=fake_service,
        ), patch.object(
            web_search_agent_module,
            "datetime",
            FixedDateTime,
        ):
            response = await agent.process(request)

        self.assertEqual(response.text, "整理好的搜尋結果")
        self.assertEqual(
            fake_service.search.await_args.args[0],
            "2026-03-25 14:30 UTC+8 今天新聞",
        )
        self.assertEqual(fake_service.search.await_args.kwargs["time_range"], "week")
        fallback.generate.assert_awaited_once()


class WebSearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_quota_stats_are_informational_only(self) -> None:
        settings = Settings(
            _env_file=None,
            tavily_api_key="tavily-key",
            web_search_monthly_quota=1000,
        )

        with patch.object(
            web_search_service_module,
            "get_settings",
            return_value=settings,
        ):
            service = web_search_service_module.WebSearchService()

        stats = service.get_quota_stats()

        self.assertTrue(service.is_quota_available)
        self.assertFalse(stats["enforced"])
        self.assertEqual(stats["configured_app_quota"], 1000)
        self.assertIsNone(stats["used"])
        self.assertIsNone(stats["quota"])
        self.assertIsNone(stats["remaining"])

    async def test_extract_parses_raw_content_and_failed_urls(self) -> None:
        settings = Settings(_env_file=None, tavily_api_key="tavily-key")

        with patch.object(
            web_search_service_module,
            "get_settings",
            return_value=settings,
        ):
            service = web_search_service_module.WebSearchService()

        fake_client = SimpleNamespace(
            extract=Mock(
                return_value={
                    "results": [
                        {
                            "url": "https://example.com/article",
                            "raw_content": "full page content",
                        }
                    ],
                    "failed_results": [
                        {
                            "url": "https://example.com/broken",
                            "error": "403",
                        }
                    ],
                }
            )
        )

        with patch.object(service, "_get_client", return_value=fake_client):
            response = await service.extract(
                [
                    "https://example.com/article",
                    "https://example.com/broken",
                ]
            )

        self.assertTrue(response.has_results)
        self.assertEqual(response.results[0].url, "https://example.com/article")
        self.assertEqual(response.results[0].content, "full page content")
        self.assertEqual(response.failed_urls, ["https://example.com/broken"])


class AgentPromptContextTests(unittest.TestCase):
    def test_agent_messages_include_routing_context(self) -> None:
        fallback = SimpleNamespace(generate=AsyncMock())
        agent = ChatAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(
            text="原始問題",
            task_description="請先把需求翻成英文再回答",
            routing_reasoning="orchestrator detected a translation-oriented request",
        )

        messages = agent._build_messages(request)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("調度任務摘要", messages[0]["content"])
        self.assertIn("請先把需求翻成英文再回答", messages[0]["content"])
        self.assertIn("調度補充說明", messages[0]["content"])

    def test_agent_messages_include_quoted_text_context(self) -> None:
        fallback = SimpleNamespace(generate=AsyncMock())
        agent = ChatAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(
            text="幫我翻譯這句",
            quoted_message_id="M_QUOTED",
            quoted_text="Hello world",
        )

        messages = agent._build_messages(request)

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("引用的上一則訊息", messages[1]["content"])
        self.assertIn("Hello world", messages[1]["content"])

    def test_agent_messages_support_quoted_image_url_context(self) -> None:
        fallback = SimpleNamespace(generate=AsyncMock())
        agent = ChatAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(
            text="請解釋這張圖",
            quoted_message_id="M_QUOTED",
            quoted_image_url="https://example.com/quoted.png",
        )

        messages = agent._build_messages(request)

        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"][1]["image_url"]["url"], "https://example.com/quoted.png")

    def test_agent_messages_promote_quote_only_image_url_to_primary_image(self) -> None:
        fallback = SimpleNamespace(generate=AsyncMock())
        agent = ChatAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(
            quoted_message_id="M_QUOTED",
            quoted_image_url="https://example.com/quoted.png",
            input_type=InputType.IMAGE,
        )

        messages = agent._build_messages(request)

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("請描述這張使用者引用的圖片", messages[1]["content"][0]["text"])
        self.assertEqual(messages[1]["content"][1]["image_url"]["url"], "https://example.com/quoted.png")

    def test_agent_messages_include_non_text_output_context(self) -> None:
        fallback = SimpleNamespace(generate=AsyncMock())
        agent = ChatAgent(Settings(_env_file=None), fallback, targets=[])
        request = AgentRequest(
            text="念給我聽",
            output_format="voice",
            task_description="用自然口語回答",
        )

        messages = agent._build_messages(request)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("調度指定輸出形式：voice", messages[0]["content"])


class AgentTimeoutPropagationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_agent_passes_thinking_timeout_to_fallback_chain(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(return_value=ProviderResponse(text="ok", model="model-1"))
        )
        agent = ChatAgent(
            Settings(_env_file=None, thinking_timeout_seconds=12),
            fallback,
            targets=[],
        )

        await agent.process(AgentRequest(text="你好"))

        self.assertEqual(fallback.generate.await_args.kwargs["thinking_timeout"], 12)

    async def test_vision_agent_passes_thinking_timeout_to_fallback_chain(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(return_value=ProviderResponse(text="ok", model="model-1"))
        )
        agent = VisionAgent(
            Settings(_env_file=None, thinking_timeout_seconds=18),
            fallback,
            targets=[],
        )

        await agent.process(
            AgentRequest(
                text="這張圖是什麼？",
                image_base64="data:image/jpeg;base64,abc",
            )
        )

        self.assertEqual(fallback.generate.await_args.kwargs["thinking_timeout"], 18)

    async def test_web_search_agent_passes_thinking_timeout_to_fallback_chain(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(return_value=ProviderResponse(text="ok", model="model-1"))
        )
        agent = WebSearchAgent(
            Settings(_env_file=None, thinking_timeout_seconds=24),
            fallback,
            targets=[],
        )
        fake_service = SimpleNamespace(
            is_configured=True,
            is_quota_available=True,
            search=AsyncMock(
                return_value=SimpleNamespace(
                    has_results=True,
                    answer="ok",
                    to_context_text=lambda: "search context",
                )
            ),
        )

        with patch.object(
            web_search_agent_module,
            "get_web_search_service",
            return_value=fake_service,
        ):
            await agent.process(AgentRequest(text="今天新聞"))

        self.assertEqual(fallback.generate.await_args.kwargs["thinking_timeout"], 24)

    async def test_orchestrator_passes_thinking_timeout_to_fallback_chain(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(
                    text='{"agent":"chat","output_format":"text","task_description":"回答問題","reasoning":"test"}',
                    model="model-1",
                )
            )
        )
        orchestrator = Orchestrator(
            Settings(_env_file=None, thinking_timeout_seconds=9),
            fallback,
            targets=[],
        )

        await orchestrator._llm_classify(AgentRequest(text="幫我處理一下"))

        self.assertEqual(fallback.generate.await_args.kwargs["thinking_timeout"], 9)
        self.assertEqual(fallback.generate.await_args.kwargs["max_tokens"], 384)


class ImageGenPromptRefinementTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_gen_refinement_uses_configured_settings_and_timeout(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(
                    text="a detailed prompt",
                    model="qwen/test",
                )
            )
        )
        nvidia = SimpleNamespace(
            generate_image=AsyncMock(
                return_value=ProviderResponse(
                    images=["data:image/jpeg;base64,abc"],
                    model="stabilityai/stable-diffusion-3-medium",
                )
            )
        )
        agent = ImageGenAgent(
            Settings(
                _env_file=None,
                image_gen_temperature=0.33,
                image_gen_max_tokens=456,
                thinking_timeout_seconds=21,
            ),
            fallback,
            targets=[],
            nvidia_provider=nvidia,
        )

        await agent.process(AgentRequest(text="幫我畫一隻柴犬"))

        refine_kwargs = next(
            call.kwargs
            for call in fallback.generate.await_args_list
            if call.kwargs.get("thinking_timeout") == 21
        )
        self.assertEqual(refine_kwargs["temperature"], 0.33)
        self.assertEqual(refine_kwargs["max_tokens"], 456)
        self.assertEqual(refine_kwargs["thinking_timeout"], 21)


class ImageGenMessageTests(unittest.TestCase):
    def test_image_gen_uses_nvidia_provider(self) -> None:
        agent = ImageGenAgent(
            Settings(_env_file=None),
            SimpleNamespace(generate=AsyncMock()),
            targets=[],
            nvidia_provider=SimpleNamespace(generate_image=AsyncMock()),
        )

        self.assertIsNotNone(agent._nvidia)
        self.assertEqual(agent.name, "image_gen")


class QuotedMessageInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_input_uses_cached_quoted_image(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "user", "userId": "U1"},
            "message": {
                "type": "text",
                "id": "M2",
                "quotedMessageId": "M1",
                "text": "請解釋這張照片",
            },
        }
        fake_cache = SimpleNamespace(
            get=lambda message_id: CachedMessage(
                message_id=message_id,
                message_type="image",
                image_base64="data:image/jpeg;base64,quoted",
            )
        )

        with patch.object(
            input_processor_module,
            "get_message_cache_service",
            return_value=fake_cache,
        ):
            request = await input_processor_module.process_input(event)

        self.assertIsNotNone(request)
        self.assertEqual(request.quoted_message_id, "M1")
        self.assertEqual(request.quoted_message_type, "image")
        self.assertEqual(request.image_base64, "data:image/jpeg;base64,quoted")
        self.assertEqual(request.input_type.value, "image_text")

    async def test_process_input_records_cached_quoted_text(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "user", "userId": "U1"},
            "message": {
                "type": "text",
                "id": "M3",
                "quotedMessageId": "M_TEXT",
                "text": "幫我翻譯這句",
            },
        }
        fake_cache = SimpleNamespace(
            get=lambda message_id: CachedMessage(
                message_id=message_id,
                message_type="text",
                text="Hello world",
            )
        )

        with patch.object(
            input_processor_module,
            "get_message_cache_service",
            return_value=fake_cache,
        ):
            request = await input_processor_module.process_input(event)

        self.assertIsNotNone(request)
        self.assertEqual(request.quoted_text, "Hello world")
        self.assertEqual(request.text, "幫我翻譯這句")


class InputLoggingTests(unittest.IsolatedAsyncioTestCase):
    def test_build_input_log_summary_logs_only_current_user_input(self) -> None:
        request = AgentRequest(
            request_id="REQ12345",
            input_type=InputType.TEXT,
            text="第一行\n第二行",
            quoted_message_id="M_QUOTED",
            quoted_text="這段不該出現在 log",
            quoted_image_url="https://example.com/quoted.png",
        )

        summary = input_processor_module._build_input_log_summary(request)

        self.assertIn('user_text="第一行\\n第二行"', summary)
        self.assertNotIn("這段不該出現在 log", summary)
        self.assertNotIn("https://example.com/quoted.png", summary)

    async def test_process_input_records_cached_quoted_bot_image_url(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "user", "userId": "U1"},
            "message": {
                "type": "text",
                "id": "M4",
                "quotedMessageId": "M_BOT_IMG",
                "text": "請解釋這張圖",
            },
        }
        fake_cache = SimpleNamespace(
            get=lambda message_id: CachedMessage(
                message_id=message_id,
                message_type="image",
                image_url="https://example.com/generated.png",
            )
        )

        with patch.object(
            input_processor_module,
            "get_message_cache_service",
            return_value=fake_cache,
        ):
            request = await input_processor_module.process_input(event)

        self.assertIsNotNone(request)
        self.assertEqual(request.quoted_message_id, "M_BOT_IMG")
        self.assertEqual(request.quoted_message_type, "image")
        self.assertEqual(request.quoted_image_url, "https://example.com/generated.png")
        self.assertEqual(request.input_type.value, "image_text")


class FallbackChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_chain_uses_next_target_after_provider_error(self) -> None:
        class BrokenProvider:
            async def generate(self, model: str, messages: list[dict], **kwargs):
                raise ProviderError(model, 500, "boom")

        class WorkingProvider:
            async def generate(self, model: str, messages: list[dict], **kwargs):
                return ProviderResponse(text="ok", model=model)

        chain = FallbackChain(RateTracker())
        response = await chain.generate(
            targets=[(BrokenProvider(), "model-1"), (WorkingProvider(), "model-2")],
            messages=[{"role": "user", "content": "hello"}],
        )

        self.assertEqual(response.text, "ok")
        self.assertEqual(response.model, "model-2")
        self.assertEqual(chain.fallback_count, 1)

    async def test_fallback_chain_distinguishes_provider_failures(self) -> None:
        class BrokenProvider:
            async def generate(self, model: str, messages: list[dict], **kwargs):
                raise ProviderError(model, 500, "boom")

        chain = FallbackChain(RateTracker())

        with self.assertRaises(AllProvidersFailedError):
            await chain.generate(
                targets=[(BrokenProvider(), "model-1")],
                messages=[{"role": "user", "content": "hello"}],
            )

    async def test_fallback_chain_reports_true_rate_limit_exhaustion(self) -> None:
        class LimitedProvider:
            async def generate(self, model: str, messages: list[dict], **kwargs):
                raise RateLimitError(model, 60)

        chain = FallbackChain(RateTracker())

        with self.assertRaises(AllModelsRateLimitedError):
            await chain.generate(
                targets=[(LimitedProvider(), "model-1")],
                messages=[{"role": "user", "content": "hello"}],
            )

    async def test_fallback_chain_retries_without_thinking_after_timeout(self) -> None:
        class SlowThinkingProvider:
            def __init__(self) -> None:
                self.disable_thinking_calls: list[bool] = []

            async def generate(self, model: str, messages: list[dict], **kwargs):
                self.disable_thinking_calls.append(kwargs.get("disable_thinking", False))
                if kwargs.get("disable_thinking"):
                    return ProviderResponse(text="ok", model=model)
                await asyncio.sleep(0.05)
                return ProviderResponse(text="slow", model=model)

        provider = SlowThinkingProvider()
        chain = FallbackChain(RateTracker())

        response = await chain.generate(
            targets=[(provider, "model-1")],
            messages=[{"role": "user", "content": "hello"}],
            thinking_timeout=0.01,
        )

        self.assertEqual(response.text, "ok")
        self.assertEqual(provider.disable_thinking_calls, [False, True])

    async def test_fallback_chain_skips_timeout_when_thinking_is_already_disabled(self) -> None:
        class NonThinkingProvider:
            async def generate(self, model: str, messages: list[dict], **kwargs):
                await asyncio.sleep(0.02)
                return ProviderResponse(
                    text="ok",
                    model=model,
                    usage={"disable_thinking": kwargs.get("disable_thinking")},
                )

        chain = FallbackChain(RateTracker())
        response = await chain.generate(
            targets=[(NonThinkingProvider(), "model-1")],
            messages=[{"role": "user", "content": "hello"}],
            thinking_timeout=0.001,
            disable_thinking=True,
        )

        self.assertEqual(response.text, "ok")
        self.assertEqual(response.usage, {"disable_thinking": True})


class RateLimitServiceTests(unittest.TestCase):
    def test_rate_limit_sliding_window_is_explicit(self) -> None:
        settings = Settings(
            _env_file=None,
            rate_limit_max_requests=2,
            rate_limit_window_seconds=60,
        )

        with patch.object(rate_limit_service_module, "get_settings", return_value=settings):
            service = rate_limit_service_module.RateLimitService()

        with patch.object(rate_limit_service_module, "time", side_effect=[0, 10, 20, 71]):
            self.assertEqual(service.check("U1"), (True, 1))
            self.assertEqual(service.check("U1"), (True, 0))
            self.assertEqual(service.check("U1"), (False, 0))
            self.assertEqual(service.check("U1"), (True, 1))


class MainTargetBuilderTests(unittest.TestCase):
    def test_text_targets_filter_non_reasoning_fallbacks(self) -> None:
        settings = Settings(
            _env_file=None,
            agent_fallback_model="nvidia/nemotron-3-super-120b-a12b:free",
            nvidia_model="qwen/qwen3.5-397b-a17b",
        )
        openrouter = object()
        nvidia = object()

        targets = main_module._build_text_agent_targets(settings, openrouter, nvidia)

        self.assertEqual(
            targets,
            [
                (nvidia, "qwen/qwen3.5-397b-a17b"),
                (openrouter, "nvidia/nemotron-3-super-120b-a12b:free"),
            ],
        )

    def test_vision_targets_keep_only_reasoning_capable_target_when_available(self) -> None:
        settings = Settings(
            _env_file=None,
            nvidia_model="qwen/qwen3.5-397b-a17b",
            vision_fallback_model="google/gemma-3-27b-it:free",
        )
        openrouter = object()
        nvidia = object()

        targets = main_module._build_vision_agent_targets(settings, openrouter, nvidia)

        self.assertEqual(
            targets,
            [
                (nvidia, "qwen/qwen3.5-397b-a17b"),
            ],
        )

    def test_vision_targets_keep_original_fallback_if_no_reasoning_target_exists(self) -> None:
        settings = Settings(
            _env_file=None,
            nvidia_api_key="",
            vision_fallback_model="google/gemma-3-27b-it:free",
        )
        openrouter = object()

        targets = main_module._build_vision_agent_targets(settings, openrouter, None)

        self.assertEqual(targets, [(openrouter, "google/gemma-3-27b-it:free")])


class LifespanInitTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_passes_openrouter_thinking_budget_to_provider(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
            openrouter_thinking_budget=1536,
            require_reasoning_models=False,
        )
        fake_openrouter = SimpleNamespace(close=AsyncMock())

        with patch.object(main_module, "get_settings", return_value=settings), patch.object(
            main_module, "setup_logger"
        ), patch.object(
            main_module, "OpenRouterProvider", return_value=fake_openrouter
        ) as openrouter_cls, patch.object(
            main_module, "FallbackChain", return_value=object()
        ), patch.object(
            main_module, "Orchestrator", return_value=object()
        ), patch.object(
            main_module, "ChatAgent", return_value=object()
        ), patch(
            "src.agents.vision_agent.VisionAgent", return_value=object()
        ), patch(
            "src.agents.web_search_agent.WebSearchAgent", return_value=object()
        ), patch(
            "src.agents.image_gen_agent.ImageGenAgent", return_value=object()
        ), patch.object(
            main_module, "get_line_service", return_value=object()
        ), patch.object(
            main_module, "close_line_service", new=AsyncMock()
        ), patch(
            "src.services.scheduler_service.close_scheduler_service"
        ), patch(
            "src.services.web_search_service.close_web_search_service",
            new=AsyncMock(),
        ):
            async with main_module.lifespan(main_module.app):
                pass

        self.assertEqual(openrouter_cls.call_args.kwargs["thinking_budget"], 1536)


class SchedulerRegistrationTests(unittest.TestCase):
    def test_register_scheduled_jobs_uses_configured_entries(self) -> None:
        settings = Settings(
            _env_file=None,
            scheduled_weekly_messages=[
                {
                    "id": "weekly-reminder",
                    "day_of_week": "mon",
                    "hour": 21,
                    "minute": 0,
                    "message": "明天操一下嗎",
                }
            ],
            scheduled_yearly_messages=[
                {
                    "id": "bday-tom",
                    "month": 4,
                    "day": 22,
                    "hour": 9,
                    "minute": 0,
                    "message": "@湯姆 生日快樂！",
                }
            ],
        )
        scheduler = SimpleNamespace(
            add_weekly_message=Mock(return_value=True),
            add_yearly_message=Mock(return_value=True),
        )

        weekly_registered, yearly_registered = main_module._register_scheduled_jobs(
            settings,
            scheduler,
            "C1234567890abcdef",
        )

        self.assertEqual((weekly_registered, yearly_registered), (1, 1))
        scheduler.add_weekly_message.assert_called_once_with(
            "weekly-reminder",
            "mon",
            21,
            0,
            "C1234567890abcdef",
            "明天操一下嗎",
        )
        scheduler.add_yearly_message.assert_called_once_with(
            "bday-tom",
            4,
            22,
            9,
            0,
            "C1234567890abcdef",
            "@湯姆 生日快樂！",
        )


class HealthStatusTests(unittest.TestCase):
    def test_health_status_is_degraded_without_required_credentials(self) -> None:
        settings = Settings(_env_file=None)

        status, ready, warnings = main_module._get_health_status(settings)

        self.assertEqual(status, "degraded")
        self.assertFalse(ready)
        self.assertTrue(warnings)

    def test_health_status_is_healthy_when_required_credentials_exist(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
        )

        status, ready, warnings = main_module._get_health_status(settings)

        self.assertEqual(status, "healthy")
        self.assertTrue(ready)
        self.assertEqual(warnings, [])

    def test_health_status_is_degraded_without_openrouter_even_if_nvidia_exists(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            nvidia_api_key="nv-key",
        )

        status, ready, warnings = main_module._get_health_status(settings)

        self.assertEqual(status, "degraded")
        self.assertFalse(ready)
        self.assertIn("OPENROUTER_API_KEY not set", warnings)

    def test_health_status_warns_when_scheduler_is_disabled_by_push_setting(self) -> None:
        settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
            scheduled_messages_enabled=True,
            scheduled_group_id="C1234567890abcdef",
            line_push_fallback_enabled=False,
            line_push_monthly_limit=0,
        )

        status, ready, warnings = main_module._get_health_status(settings)

        self.assertEqual(status, "healthy")
        self.assertTrue(ready)
        self.assertIn(
            "Scheduled messages are configured but LINE push is disabled",
            warnings,
        )


class CleanedTextTests(unittest.TestCase):
    def test_apply_cleaned_text_allows_quote_only_image_requests(self) -> None:
        request = AgentRequest(
            text="!hej",
            image_base64="data:image/jpeg;base64,quoted",
        )
        event = {
            "message": {
                "type": "text",
                "text": "!hej",
            }
        }

        with patch.object(main_module, "extract_text", return_value=""):
            main_module._apply_cleaned_text(request, event)

        self.assertEqual(request.text, "")
        self.assertEqual(request.input_type.value, "image")

    def test_apply_cleaned_text_preserves_quote_only_image_url_requests(self) -> None:
        request = AgentRequest(
            text="!hej",
            quoted_image_url="https://example.com/quoted.png",
        )
        event = {
            "message": {
                "type": "text",
                "text": "!hej",
            }
        }

        with patch.object(main_module, "extract_text", return_value=""):
            main_module._apply_cleaned_text(request, event)

        self.assertEqual(request.text, "")
        self.assertEqual(request.input_type.value, "image")


class RequestBlockMessageTests(unittest.TestCase):
    def test_request_block_message_distinguishes_empty_and_rate_limit(self) -> None:
        empty_request = AgentRequest()
        self.assertEqual(
            main_module._get_request_block_message(empty_request),
            "請附上想問的內容，或直接引用訊息／圖片再提問。",
        )

        limited_request = AgentRequest(rate_limited=True)
        self.assertEqual(
            main_module._get_request_block_message(limited_request),
            "⚠️ 請求太頻繁，請稍後再試。",
        )

    def test_request_block_message_allows_quote_only_image_url(self) -> None:
        request = AgentRequest(
            quoted_image_url="https://example.com/quoted.png",
            input_type=InputType.IMAGE,
        )

        self.assertIsNone(main_module._get_request_block_message(request))


class OrchestratorFastRuleTests(unittest.TestCase):
    def test_url_with_voice_keywords_routes_to_web_search_voice(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(text="請唸給我聽 https://example.com/news")
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent, "web_search")
        self.assertEqual(decision.output_format, "voice")

    def test_voice_keywords_route_to_chat_voice(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        for text in (
            "請用語音回覆今天的重點",
            "用聲音說給我聽",
            "請幫我語音朗讀這段內容",
        ):
            with self.subTest(text=text):
                decision = orchestrator._apply_fast_rules(AgentRequest(text=text))
                self.assertIsNotNone(decision)
                self.assertEqual(decision.agent, "chat")
                self.assertEqual(decision.output_format, "voice")

    def test_negative_voice_keyword_keeps_url_request_as_text(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(text="不要語音，用文字回答 https://example.com/news")
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent, "web_search")
        self.assertEqual(decision.output_format, "text")

    def test_explicit_text_keywords_route_to_chat_text(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(text="請用文字回答這題")
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent, "chat")
        self.assertEqual(decision.output_format, "text")

    def test_current_events_keywords_route_to_web_search(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(text="最近有什麼時事？")
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent, "web_search")
        self.assertEqual(decision.output_format, "text")

    def test_weather_and_stock_keywords_route_to_web_search(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        for text in ("台北今天天氣怎樣", "台積電現在股價多少"):
            with self.subTest(text=text):
                decision = orchestrator._apply_fast_rules(AgentRequest(text=text))
                self.assertIsNotNone(decision)
                self.assertEqual(decision.agent, "web_search")

    def test_today_without_search_topic_does_not_force_web_search(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(text="今天心情不好")
        )

        # "今天心情不好" (6 chars) is NOT a simple greeting — should go to LLM
        # to allow correct output_format classification (e.g., voice)
        self.assertIsNone(decision)

    def test_general_why_question_stays_non_thinking(self) -> None:
        self.assertTrue(
            orchestrator_module._should_disable_thinking("為什麼今天會下雨")
        )

    def test_general_comparison_stays_non_thinking(self) -> None:
        self.assertTrue(
            orchestrator_module._should_disable_thinking("比較 iPhone 16 和 S24 的差異")
        )

    def test_explicit_step_by_step_debugging_enables_thinking(self) -> None:
        self.assertFalse(
            orchestrator_module._should_disable_thinking(
                "請一步一步分析這段 traceback 並提出修復方案"
            )
        )

    def test_simple_code_request_stays_non_thinking(self) -> None:
        self.assertTrue(
            orchestrator_module._should_disable_thinking("幫我寫一段 Python code")
        )

    def test_simple_image_description_disables_thinking(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(
                text="這張圖是什麼",
                image_base64="data:image/jpeg;base64,abc",
                input_type=InputType.IMAGE_TEXT,
            )
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent, "vision")
        self.assertTrue(decision.disable_thinking)

    def test_detailed_but_literal_image_question_stays_non_thinking(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(
                text="請幫我看看這張照片裡的人穿什麼、手上拿什麼、背景在哪裡",
                image_base64="data:image/jpeg;base64,abc",
                input_type=InputType.IMAGE_TEXT,
            )
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent, "vision")
        self.assertTrue(decision.disable_thinking)

    def test_complex_image_question_keeps_thinking_enabled(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._apply_fast_rules(
            AgentRequest(
                text="這裡錯在哪",
                image_base64="data:image/jpeg;base64,abc",
                input_type=InputType.IMAGE_TEXT,
            )
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.agent, "vision")
        self.assertFalse(decision.disable_thinking)


class ConversationHistoryExpiryTests(unittest.TestCase):
    def test_get_history_prunes_expired_messages_from_deque(self) -> None:
        settings = Settings(_env_file=None, max_conversation_history=3)

        with patch.object(
            conversation_service_module,
            "get_settings",
            return_value=settings,
        ):
            service = conversation_service_module.ConversationService()

        service._history["chat"] = deque(
            [
                ConversationMessage(role="user", content="old", timestamp=0),
                ConversationMessage(role="assistant", content="fresh", timestamp=3500),
            ],
            maxlen=service._max,
        )

        with patch.object(conversation_service_module, "time", return_value=3601):
            history = service.get_history("chat")

        self.assertEqual(history, [{"role": "assistant", "content": "fresh"}])
        self.assertEqual(len(service._history["chat"]), 1)


class MessageCacheBehaviorTests(unittest.TestCase):
    def test_cache_processed_request_prefers_cleaned_text_and_valid_image_fit(self) -> None:
        service = message_cache_service_module.MessageCacheService()
        event = {"message": {"type": "image", "id": "M1", "text": "!hej 幫我看"}}
        request = AgentRequest(
            input_type=InputType.IMAGE_TEXT,
            text="幫我看",
            image_base64="data:image/jpeg;base64,oversized",
        )

        with patch.object(
            message_cache_service_module,
            "fit_image_data_url",
            return_value="data:image/jpeg;base64,compact",
        ):
            service.cache_processed_request(event, request)

        cached = service.get("M1")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.text, "幫我看")
        self.assertEqual(cached.image_base64, "data:image/jpeg;base64,compact")

    def test_cache_bot_message_keeps_image_url_for_bot_generated_images(self) -> None:
        service = message_cache_service_module.MessageCacheService()

        service.cache_bot_message(
            "BOT_IMG_1",
            "image",
            image_url="https://example.com/generated.png",
        )

        cached = service.get("BOT_IMG_1")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.message_type, "image")
        self.assertEqual(cached.image_url, "https://example.com/generated.png")


class ImageFitTests(unittest.TestCase):
    def test_fit_image_data_url_returns_valid_smaller_data_url(self) -> None:
        image = Image.new("RGB", (1024, 1024), color="red")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        data_url = "data:image/jpeg;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        fitted = image_service_module.fit_image_data_url(data_url, max_chars=5_000)

        self.assertIsNotNone(fitted)
        self.assertTrue(fitted.startswith("data:image/jpeg;base64,"))
        self.assertLessEqual(len(fitted), 5_000)


class OutputProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_response_schedules_cleanup_for_uploaded_image(self) -> None:
        request = AgentRequest(user_id="U1", group_id="G1", reply_token="reply")
        response = SimpleNamespace(
            image_base64="data:image/png;base64,abc",
            text="圖片好了",
            output_format="image",
        )
        line = SimpleNamespace(
            send_messages=AsyncMock(return_value=True),
            send_text=AsyncMock(return_value=False),
        )
        uploaded = UploadedMedia(
            public_url="https://example.com/image.png",
            blob_name="img/test.png",
            size_bytes=123,
        )
        storage = SimpleNamespace(
            upload_base64_image=AsyncMock(return_value=uploaded),
            schedule_cleanup=Mock(),
        )

        with patch.object(
            output_processor_module, "get_line_service", return_value=line
        ), patch.object(
            output_processor_module, "get_storage_service", return_value=storage
        ):
            sent = await output_processor_module.send_response(request, response)

        self.assertTrue(sent)
        line.send_messages.assert_awaited_once_with(
            "reply",
            "G1",
            [
                {
                    "type": "image",
                    "originalContentUrl": "https://example.com/image.png",
                    "previewImageUrl": "https://example.com/image.png",
                },
                {"type": "text", "text": "圖片好了"},
            ],
        )
        line.send_text.assert_not_called()
        storage.schedule_cleanup.assert_called_once_with(uploaded)

    async def test_send_response_falls_back_to_text_and_cleans_up_failed_audio(self) -> None:
        request = AgentRequest(user_id="U1", group_id="G1", reply_token="reply")
        response = SimpleNamespace(
            text="請聽這段",
            audio_url=None,
            output_format="voice",
        )
        line = SimpleNamespace(
            send_audio=AsyncMock(return_value=False),
            send_text=AsyncMock(return_value=True),
        )
        uploaded = UploadedMedia(
            public_url="https://example.com/audio.mp3",
            blob_name="audio/test.mp3",
            size_bytes=456,
        )
        storage = SimpleNamespace(schedule_cleanup=Mock())

        with patch.object(
            output_processor_module, "get_line_service", return_value=line
        ), patch.object(
            output_processor_module, "get_storage_service", return_value=storage
        ), patch(
            "src.processors.tts_processor.text_to_speech",
            AsyncMock(return_value=uploaded),
        ):
            sent = await output_processor_module.send_response(request, response)

        self.assertTrue(sent)
        line.send_audio.assert_awaited_once_with("reply", "G1", "https://example.com/audio.mp3", duration_ms=60000)
        line.send_text.assert_awaited_once_with("reply", "G1", "請聽這段")
        storage.schedule_cleanup.assert_called_once_with(uploaded, delay_seconds=0)


class TtsProcessorTests(unittest.TestCase):
    def test_mp3_duration_fallback_uses_128kbps_estimate(self) -> None:
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mutagen.mp3":
                raise ImportError("mutagen unavailable")
            return original_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.mp3"
            path.write_bytes(b"x" * 32000)

            with patch("builtins.__import__", side_effect=fake_import):
                duration_ms = tts_processor_module._get_mp3_duration_ms(path)

        self.assertEqual(duration_ms, 2000)


class OrchestratorParsingTests(unittest.TestCase):
    def test_parse_llm_response_accepts_fenced_json(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._parse_llm_response(
            '```json\n{"agent":"web_search","output_format":"voice","task_description":"查詢今天新聞並口語回答","reasoning":"需要即時資訊"}\n```',
            "今天新聞念給我聽",
        )

        self.assertEqual(decision.agent, "web_search")
        self.assertEqual(decision.output_format, "voice")
        self.assertEqual(decision.reasoning, "需要即時資訊")

    def test_parse_llm_response_accepts_json_with_surrounding_text(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._parse_llm_response(
            '結果如下：{"agent":"chat","output_format":"text","task_description":"回答翻譯需求","reasoning":"一般文字任務"} 請處理',
            "幫我翻譯這句",
        )

        self.assertEqual(decision.agent, "chat")
        self.assertEqual(decision.output_format, "text")
        self.assertEqual(decision.task_description, "回答翻譯需求")
        self.assertTrue(decision.disable_thinking)

    def test_parse_llm_response_inverts_needs_thinking_into_disable_thinking(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._parse_llm_response(
            '{"agent":"chat","output_format":"text","needs_thinking":false,"task_description":"簡短回答","reasoning":"簡單問題"}',
            "你好",
        )

        self.assertEqual(decision.agent, "chat")
        self.assertTrue(decision.disable_thinking)

    def test_parse_llm_response_defaults_to_non_thinking_when_field_missing(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._parse_llm_response(
            '{"agent":"chat","output_format":"text","task_description":"回答翻譯需求","reasoning":"一般文字任務"}',
            "幫我翻譯這句",
        )

        self.assertEqual(decision.agent, "chat")
        self.assertTrue(decision.disable_thinking)

    def test_parse_llm_response_unparseable_falls_back_to_non_thinking_chat(self) -> None:
        orchestrator = Orchestrator(
            Settings(_env_file=None),
            SimpleNamespace(),
            targets=[],
        )

        decision = orchestrator._parse_llm_response("not json", "幫我翻譯這句")

        self.assertEqual(decision.agent, "chat")
        self.assertEqual(decision.output_format, "text")
        self.assertTrue(decision.disable_thinking)


class ImageGenErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_auth_failure_returns_actionable_message(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(
                    text="a detailed prompt",
                    model="qwen/test",
                )
            )
        )
        nvidia = SimpleNamespace(
            generate_image=AsyncMock(
                side_effect=ProviderError(
                    "stabilityai/stable-diffusion-3-medium",
                    401,
                    '{"error":"Unauthorized"}',
                )
            )
        )
        agent = ImageGenAgent(
            Settings(_env_file=None),
            fallback,
            targets=[],
            nvidia_provider=nvidia,
        )

        response = await agent.process(AgentRequest(text="幫我畫一隻柴犬"))

        self.assertEqual(response.output_format, "image")
        self.assertIsNone(response.image_base64)
        self.assertIn("NVIDIA_API_KEY", response.text)

    async def test_nvidia_sd_image_gen_success(self) -> None:
        fallback = SimpleNamespace(
            generate=AsyncMock(
                return_value=ProviderResponse(
                    text="a detailed prompt",
                    model="qwen/test",
                )
            )
        )
        nvidia = SimpleNamespace(
            generate_image=AsyncMock(
                return_value=ProviderResponse(
                    images=["data:image/jpeg;base64,abc"],
                    model="stabilityai/stable-diffusion-3-medium",
                )
            )
        )
        agent = ImageGenAgent(
            Settings(_env_file=None),
            fallback,
            targets=[],
            nvidia_provider=nvidia,
        )

        response = await agent.process(AgentRequest(text="幫我畫一隻柴犬"))

        self.assertEqual(response.output_format, "image")
        self.assertEqual(response.image_base64, "data:image/jpeg;base64,abc")
