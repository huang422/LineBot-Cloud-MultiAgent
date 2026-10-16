from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import unittest
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient

import main as main_module
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.pipeline import event_pipeline as event_pipeline_module
from src.pipeline.event_pipeline import PipelineDeps, process_event
from src.services.line_service import LineService


def _sign_body(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _reset_event_dedupe(test: unittest.TestCase) -> None:
    """The webhook / task dedupe sets are process-wide; isolate each test."""
    for seen in (main_module._webhook_seen, main_module._task_seen):
        seen.clear()
        test.addCleanup(seen.clear)


def _skip_startup_env_scan(test: unittest.TestCase) -> None:
    """Keep the app hermetic: no developer .env, no live lifecycle check.

    Lifespan's legacy-config check scans .env, and ``get_line_service()``
    builds LineService from the cached ``src.config.get_settings()``, which
    reads .env too (a broken local .env used to fail these tests). Tests
    that need a specific LINE fake patch ``get_line_service`` again inside.
    The first ``/internal/cron`` of a day queues the model end-of-life
    check, which the inline queue would run against the live catalogs;
    ``ModelLifecycle.run`` is stubbed (tests/test_eol_checker.py covers it
    with a MockTransport).
    """
    for name, replacement in (
        ("legacy_config_warnings", Mock(return_value=[])),
        ("get_line_service", Mock(return_value=SimpleNamespace())),
        ("close_line_service", AsyncMock()),
    ):
        patcher = patch.object(main_module, name, new=replacement)
        patcher.start()
        test.addCleanup(patcher.stop)
    patcher = patch("src.services.model_lifecycle.ModelLifecycle.run", new=AsyncMock(return_value={}))
    patcher.start()
    test.addCleanup(patcher.stop)


class ApiEndpointIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_event_dedupe(self)
        _skip_startup_env_scan(self)

        self.settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
            log_level="ERROR",
        )
        self.fake_line = SimpleNamespace()
        self.fake_memory = SimpleNamespace(
            get_stats=lambda: {
                "backend": "memory",
                "persistent": False,
                "recent_message_limit": 5,
                "summary_timeout_seconds": 180,
                "summary_tasks": 0,
                "tracked_chats": 0,
            }
        )
        self.fake_scheduler = SimpleNamespace(
            get_stats=lambda: {
                "running": False,
                "job_count": 0,
                "jobs": [],
            }
        )
        self.fake_storage = SimpleNamespace(
            get_usage_stats=lambda: {
                "configured": False,
                "monthly_uploads": 0,
                "monthly_limit": 0,
            }
        )

    @contextmanager
    def _client(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main_module, "get_settings", return_value=self.settings)
            )
            stack.enter_context(
                patch.object(main_module, "get_line_service", return_value=self.fake_line)
            )
            stack.enter_context(
                patch.object(main_module, "close_line_service", new=AsyncMock())
            )
            stack.enter_context(
                patch.object(main_module, "get_memory_service", return_value=self.fake_memory)
            )
            stack.enter_context(
                patch.object(main_module, "close_memory_service", new=AsyncMock())
            )
            stack.enter_context(
                patch(
                    "src.services.scheduler_service.peek_scheduler_service",
                    return_value=self.fake_scheduler,
                )
            )
            stack.enter_context(
                patch("src.services.scheduler_service.close_scheduler_service")
            )
            stack.enter_context(
                patch(
                    "src.services.storage_service.get_storage_service",
                    return_value=self.fake_storage,
                )
            )
            client = stack.enter_context(TestClient(main_module.app))
            yield client

    def _signed_headers(self, body: bytes) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Line-Signature": _sign_body(self.settings.line_channel_secret, body),
        }

    def test_health_endpoint_returns_the_minimal_public_status(self) -> None:
        with self._client() as client, patch.dict("os.environ", {"APP_VERSION": "v3-test"}):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        # Everything else moved to the authenticated /health/detail.
        self.assertEqual(
            response.json(), {"status": "healthy", "ready_for_webhook": True, "version": "v3-test"}
        )

    def test_health_detail_reports_the_running_app(self) -> None:
        self.settings = self.settings.model_copy(update={"internal_cron_secret": "cron-secret"})
        with self._client() as client:
            response = client.get("/health/detail", headers={"X-Cron-Token": "cron-secret"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual((payload["status"], payload["ready_for_webhook"]), ("healthy", True))
        self.assertEqual(payload["warnings"], [])
        self.assertIn("orchestrator", payload["agents_stats"])
        self.assertIn("chat", payload["agents_stats"])
        self.assertNotIn("conversation", payload)
        self.assertEqual(payload["memory"]["backend"], "memory")
        self.assertEqual(payload["memory"]["summary_timeout_seconds"], 180)
        self.assertEqual(payload["scheduler"], {"running": False, "jobs": []})
        self.assertNotIn("push", payload)
        self.assertEqual(
            [tier["model"] for tier in payload["roles"]["chat"][1:]],
            [
                "nvidia/nemotron-3-super-120b-a12b",
                "nvidia/nemotron-3-ultra-550b-a55b",
                "dots-studio/dots-3-note-preview:free",
                "nvidia/nemotron-3-ultra-550b-a55b:free",
            ],
        )
        # Off Cloud Run (no K_SERVICE) post-webhook work runs inline.
        self.assertEqual(payload["tasks_mode"], "inline")

    def test_webhook_rejects_invalid_signature(self) -> None:
        body = b'{"events":[]}'

        with self._client() as client:
            response = client.post(
                "/webhook",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Line-Signature": "invalid-signature",
                },
            )

        self.assertEqual(response.status_code, 403)

    def test_webhook_rejects_everything_when_channel_secret_is_empty(self) -> None:
        # An empty secret makes the HMAC key public: anyone could sign.
        self.settings = Settings(
            _env_file=None,
            line_channel_secret="",
            line_channel_access_token="token",
            openrouter_api_key="key",
            log_level="ERROR",
        )
        body = b'{"events":[]}'

        with self._client() as client:
            response = client.post(
                "/webhook",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Line-Signature": _sign_body("", body),
                },
            )

        self.assertEqual(response.status_code, 403)

    def test_webhook_rejects_non_ascii_signature_with_403(self) -> None:
        body = b'{"events":[]}'

        with self._client() as client:
            response = client.post(
                "/webhook",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    # Raw UTF-8 bytes; the server sees them as non-ASCII
                    # latin-1 text, which used to crash compare_digest (500).
                    "X-Line-Signature": "簽章".encode(),
                },
            )

        self.assertEqual(response.status_code, 403)

    def test_webhook_returns_400_for_invalid_json(self) -> None:
        body = b"{not-json"

        with self._client() as client:
            response = client.post(
                "/webhook",
                content=body,
                headers=self._signed_headers(body),
            )

        self.assertEqual(response.status_code, 400)

    def test_webhook_caches_non_triggered_message_events(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "G1", "userId": "U1"},
            "message": {"type": "text", "id": "M1", "text": "只是路過"},
        }
        body = json.dumps({"events": [event]}).encode("utf-8")
        cache_service = SimpleNamespace(
            cache_event_text=Mock(return_value=True), cache_event_message=AsyncMock()
        )

        with self._client() as client, patch.object(
            main_module, "should_handle", return_value=False
        ), patch.object(
            main_module, "get_message_cache_service", return_value=cache_service
        ), patch.object(
            main_module, "_submit_task", AsyncMock()
        ) as submit_task, patch.object(
            main_module, "_record_group_message", return_value=None
        ) as record_group_message:
            response = client.post(
                "/webhook",
                content=body,
                headers=self._signed_headers(body),
            )

        self.assertEqual(response.status_code, 200)
        # Text is cached inline and synchronously; nothing is queued for it.
        cache_service.cache_event_text.assert_called_once_with(event)
        cache_service.cache_event_message.assert_not_called()
        submit_task.assert_not_called()
        record_group_message.assert_called_once_with(event)

    def test_webhook_submits_triggered_events_as_event_tasks(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "G1", "userId": "U1"},
            "message": {"type": "text", "id": "M2", "text": "!hej 幫我整理"},
        }
        body = json.dumps({"events": [event]}).encode("utf-8")
        cache_service = SimpleNamespace(
            cache_event_text=Mock(), cache_event_message=AsyncMock()
        )

        with self._client() as client, patch.object(
            main_module, "should_handle", return_value=True
        ), patch.object(
            main_module, "get_message_cache_service", return_value=cache_service
        ), patch.object(
            main_module, "_submit_task", AsyncMock()
        ) as submit_task, patch.object(
            main_module, "_record_group_message"
        ) as record_group_message:
            response = client.post(
                "/webhook",
                content=body,
                headers=self._signed_headers(body),
            )

        self.assertEqual(response.status_code, 200)
        # The webhook awaits the submission instead of fire-and-forget.
        submit_task.assert_awaited_once_with("event", {"event": event})
        cache_service.cache_event_text.assert_not_called()
        cache_service.cache_event_message.assert_not_called()
        record_group_message.assert_not_called()


class EventPipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_event_routes_and_sends_response(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "G1", "userId": "U1"},
            "message": {"type": "text", "id": "M3", "text": "!hej 幫我整理重點"},
        }
        request = AgentRequest(
            user_id="U1",
            group_id="G1",
            reply_token="reply-token",
            text="!hej 幫我整理重點",
        )
        line_service = SimpleNamespace(
            send_loading_animation=AsyncMock(),
            send_text=AsyncMock(),
            fetch_display_name=AsyncMock(return_value="Tom"),
        )
        cache_service = SimpleNamespace(cache_processed_request=Mock())
        decision = SimpleNamespace(
            agent="chat",
            output_format="text",
            task_description="整理使用者需求後回覆",
            reasoning="一般文字任務",
            disable_thinking=False,
        )
        response = AgentResponse(
            text="這是整理後的回覆",
            agent_name="chat",
            output_format="text",
        )
        agent = SimpleNamespace(process=AsyncMock(return_value=response))
        orchestrator = SimpleNamespace(route=AsyncMock(return_value=decision))
        memory_service = SimpleNamespace(
            record_interaction=AsyncMock(),
            touch_user_profile=AsyncMock(),
            load_profile_facts=AsyncMock(return_value=["住台北"]),
        )
        order: list[str] = []
        memory_service.touch_user_profile.side_effect = lambda **kwargs: order.append("touch")

        send_response = AsyncMock(return_value=True)
        send_response.side_effect = lambda *args: order.append("send") or True
        deps = PipelineDeps(
            line=line_service,
            memory=memory_service,
            message_cache=cache_service,
            orchestrator=orchestrator,
            get_agent=lambda name: agent,
            process_input=AsyncMock(return_value=request),
            enrich_request=AsyncMock(return_value=request),
            extract_text=Mock(return_value="幫我整理重點"),
            send_response=send_response,
        )
        await process_event(event, deps)

        line_service.send_loading_animation.assert_awaited_once_with("G1")
        cache_service.cache_processed_request.assert_called_once_with(event, request)
        orchestrator.route.assert_awaited_once_with(request)
        agent.process.assert_awaited_once_with(request)
        send_response.assert_awaited_once_with(request, response)
        memory_service.record_interaction.assert_awaited_once_with(
            source_type="user",
            chat_scope="user",
            chat_id="G1",
            user_id="U1",
            user_text="幫我整理重點",
            assistant_text="這是整理後的回覆",
            agent_name="chat",
            output_format="text",
            task_description="整理使用者需求後回覆",
            routing_reasoning="一般文字任務",
            disable_thinking=False,
            user_created_at=request.received_at,
        )
        # The sender's facts reach the agent; the profile touch waits for the reply.
        memory_service.load_profile_facts.assert_awaited_once_with("U1")
        self.assertEqual(request.user_profile_facts, ["住台北"])
        memory_service.touch_user_profile.assert_awaited_once_with(
            user_id="U1", display_name="Tom", source_type="user", chat_id="G1"
        )
        self.assertEqual(order, ["send", "touch"])
        line_service.send_text.assert_not_called()
        self.assertEqual(request.text, "幫我整理重點")
        self.assertEqual(request.target_agent, "chat")
        self.assertEqual(request.output_format, "text")
        self.assertFalse(request.disable_thinking)

    async def test_slow_answers_go_ahead_without_asking_for_a_push_budget(self) -> None:
        # A thinking answer or an image may outlive the reply token in any
        # chat: it then goes out by push, which nothing counts or refuses.
        for source_type, disable_thinking, output_format in (
            ("user", False, "text"),
            ("group", False, "text"),
            ("room", False, "text"),
            ("user", True, "image"),
            ("group", True, "image"),
        ):
            with self.subTest(source_type=source_type, output_format=output_format):
                request = AgentRequest(
                    user_id="U1", group_id="U1", reply_token="r", text="分析", source_type=source_type
                )
                agent_process = AsyncMock(
                    return_value=AgentResponse(text="答", agent_name="chat", output_format="text")
                )
                decision = SimpleNamespace(
                    agent="image_gen" if output_format == "image" else "chat",
                    output_format=output_format,
                    task_description="",
                    reasoning="",
                    disable_thinking=disable_thinking,
                )
                # No can_push: the pipeline must not need one.
                line = SimpleNamespace(
                    send_loading_animation=AsyncMock(),
                    send_text=AsyncMock(),
                    fetch_display_name=AsyncMock(return_value=""),
                )
                send_response = AsyncMock(return_value=True)
                deps = PipelineDeps(
                    line=line,
                    memory=SimpleNamespace(
                        record_interaction=AsyncMock(),
                        touch_user_profile=AsyncMock(),
                        load_profile_facts=AsyncMock(return_value=[]),
                        flush_passive=AsyncMock(),
                    ),
                    message_cache=SimpleNamespace(cache_processed_request=Mock()),
                    orchestrator=SimpleNamespace(route=AsyncMock(return_value=decision)),
                    get_agent=lambda name, process=agent_process: SimpleNamespace(process=process),
                    process_input=AsyncMock(return_value=request),
                    enrich_request=AsyncMock(return_value=request),
                    extract_text=Mock(return_value="分析"),
                    send_response=send_response,
                )
                event = {"type": "message", "replyToken": "r", "source": {"type": "user", "userId": "U1"}}

                await process_event(event, deps)

                agent_process.assert_awaited_once_with(request)
                send_response.assert_awaited_once()
                line.send_text.assert_not_called()

    async def test_an_answer_80_s_after_the_event_still_tries_the_reply_first(self) -> None:
        # Delivery never skips the reply because time has passed: only LINE
        # knows whether the token still works. Here it refuses it (400), so
        # the answer goes out by push, once.
        calls: list[str] = []

        def line_api(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/v2/bot/message/reply":
                return httpx.Response(400, json={"message": "Invalid reply token"})
            return httpx.Response(200, json={"sentMessages": []})

        line = LineService(
            Settings(_env_file=None, line_channel_access_token="token"),
            transport=httpx.MockTransport(line_api),
        )
        self.addAsyncCleanup(line.close)
        received_at = time.time() - 80
        request = AgentRequest(
            user_id="U1", group_id="U1", reply_token="r", text="你好", received_at=received_at
        )
        decision = SimpleNamespace(
            agent="chat", output_format="text", task_description="", reasoning="", disable_thinking=True
        )
        answer = AgentResponse(text="答案", agent_name="chat", output_format="text")
        memory = SimpleNamespace(
            record_interaction=AsyncMock(),
            touch_user_profile=AsyncMock(),
            load_profile_facts=AsyncMock(return_value=[]),
        )
        deps = PipelineDeps(
            line=SimpleNamespace(
                send_loading_animation=AsyncMock(),
                send_text=AsyncMock(),
                fetch_display_name=AsyncMock(return_value=""),
            ),
            memory=memory,
            message_cache=SimpleNamespace(cache_processed_request=Mock()),
            orchestrator=SimpleNamespace(route=AsyncMock(return_value=decision)),
            get_agent=lambda name: SimpleNamespace(process=AsyncMock(return_value=answer)),
            process_input=AsyncMock(return_value=request),
            enrich_request=AsyncMock(return_value=request),
            extract_text=Mock(return_value="你好"),
        )
        event = {
            "type": "message",
            "replyToken": "r",
            "timestamp": int(received_at * 1000),
            "source": {"type": "user", "userId": "U1"},
        }

        with patch("src.processors.output_processor.get_line_service", return_value=line):
            await process_event(event, deps)

        self.assertEqual(calls, ["/v2/bot/message/reply", "/v2/bot/message/push"])
        memory.record_interaction.assert_awaited_once()

    def _profile_deps(self, *, sent: bool, load_profile_facts) -> tuple[PipelineDeps, SimpleNamespace, AsyncMock]:
        request = AgentRequest(user_id="U1", group_id="U1", reply_token="r", text="你好")
        memory = SimpleNamespace(
            record_interaction=AsyncMock(),
            touch_user_profile=AsyncMock(),
            load_profile_facts=load_profile_facts,
        )
        agent_process = AsyncMock(
            return_value=AgentResponse(text="哈囉", agent_name="chat", output_format="text")
        )
        decision = SimpleNamespace(
            agent="chat", output_format="text", task_description="", reasoning="", disable_thinking=True
        )
        deps = PipelineDeps(
            line=SimpleNamespace(
                send_loading_animation=AsyncMock(),
                send_text=AsyncMock(),
                fetch_display_name=AsyncMock(return_value="Tom"),
            ),
            memory=memory,
            message_cache=SimpleNamespace(cache_processed_request=Mock()),
            orchestrator=SimpleNamespace(route=AsyncMock(return_value=decision)),
            get_agent=lambda name: SimpleNamespace(process=agent_process),
            process_input=AsyncMock(return_value=request),
            enrich_request=AsyncMock(return_value=request),
            extract_text=Mock(return_value="你好"),
            send_response=AsyncMock(return_value=sent),
        )
        return deps, memory, agent_process

    async def test_an_undelivered_reply_leaves_the_profile_untouched(self) -> None:
        deps, memory, _ = self._profile_deps(sent=False, load_profile_facts=AsyncMock(return_value=[]))
        event = {"type": "message", "replyToken": "r", "source": {"type": "user", "userId": "U1"}}

        await process_event(event, deps)

        memory.record_interaction.assert_not_called()
        memory.touch_user_profile.assert_not_called()
        deps.line.fetch_display_name.assert_not_called()

    async def test_no_error_message_follows_a_delivered_answer(self) -> None:
        # The reply token is spent, so an error message after the answer
        # would go out by push: a second reply to one message.
        deps, memory, _ = self._profile_deps(sent=True, load_profile_facts=AsyncMock(return_value=[]))
        memory.record_interaction.side_effect = RuntimeError("bookkeeping failed")
        event = {"type": "message", "replyToken": "r", "source": {"type": "user", "userId": "U1"}}

        with patch.object(event_pipeline_module.logger, "error") as log_error:
            await process_event(event, deps)

        deps.send_response.assert_awaited_once()
        deps.line.send_text.assert_not_called()
        self.assertIn("Event processing error", log_error.call_args.args[0])

    async def test_the_agent_sees_the_facts_and_a_failed_load_does_not_stop_the_reply(self) -> None:
        event = {"type": "message", "replyToken": "r", "source": {"type": "user", "userId": "U1"}}
        seen: list[list[str]] = []

        deps, memory, agent_process = self._profile_deps(
            sent=True, load_profile_facts=AsyncMock(return_value=["住台北"])
        )
        agent_process.side_effect = lambda request: seen.append(list(request.user_profile_facts)) or (
            AgentResponse(text="哈囉", agent_name="chat", output_format="text")
        )
        await process_event(event, deps)
        self.assertEqual(seen, [["住台北"]])

        deps, memory, agent_process = self._profile_deps(
            sent=True, load_profile_facts=AsyncMock(side_effect=RuntimeError("firestore down"))
        )
        with patch.object(event_pipeline_module.logger, "warning") as log_warning:
            await process_event(event, deps)
        self.assertIn("Profile facts not loaded", log_warning.call_args.args[0])
        deps.send_response.assert_awaited_once()
        memory.touch_user_profile.assert_awaited_once()

    async def test_the_facts_load_while_the_router_classifies(self) -> None:
        # Each side waits until the other has started: this only finishes
        # (with the facts) when the two run at the same time.
        event = {"type": "message", "replyToken": "r", "source": {"type": "user", "userId": "U1"}}
        load_started, route_started = asyncio.Event(), asyncio.Event()
        seen: list[list[str]] = []

        async def load(user_id):
            load_started.set()
            await asyncio.wait_for(route_started.wait(), 1)
            return ["住台北"]

        deps, memory, agent_process = self._profile_deps(sent=True, load_profile_facts=AsyncMock(side_effect=load))
        decision = deps.orchestrator.route.return_value

        async def route(request):
            route_started.set()
            await asyncio.wait_for(load_started.wait(), 1)
            return decision

        deps.orchestrator.route.side_effect = route
        agent_process.side_effect = lambda request: seen.append(list(request.user_profile_facts)) or (
            AgentResponse(text="哈囉", agent_name="chat", output_format="text")
        )

        await process_event(event, deps)

        self.assertEqual(seen, [["住台北"]])

    async def test_a_failed_route_cancels_the_facts_load(self) -> None:
        event = {"type": "message", "replyToken": "r", "source": {"type": "user", "userId": "U1"}}
        blocked = asyncio.Event()
        cancelled: list[bool] = []

        async def load(user_id):
            try:
                await asyncio.wait_for(blocked.wait(), 1)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return []

        async def route(request):
            await asyncio.sleep(0)  # the load is under way
            raise RuntimeError("router down")

        deps, memory, agent_process = self._profile_deps(sent=True, load_profile_facts=AsyncMock(side_effect=load))
        deps.orchestrator.route.side_effect = route

        with patch.object(event_pipeline_module.logger, "error"):
            await process_event(event, deps)
        await asyncio.sleep(0)

        self.assertEqual(cancelled, [True])
        agent_process.assert_not_called()
        deps.line.send_text.assert_awaited_once_with("r", "U1", "❌ 處理請求時發生錯誤，請稍後再試。")

    async def test_process_event_handles_new_chat_without_routing(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "G1", "userId": "U1"},
            "message": {"type": "text", "id": "M4", "text": "!new"},
        }
        request = AgentRequest(
            user_id="U1",
            group_id="G1",
            reply_token="reply-token",
            text="!new",
            source_type="group",
            chat_scope="multi",
        )
        line_service = SimpleNamespace(
            send_loading_animation=AsyncMock(),
            send_text=AsyncMock(),
        )
        memory_service = SimpleNamespace(clear_chat=AsyncMock())
        cache_service = SimpleNamespace(cache_processed_request=Mock())

        enrich_request = AsyncMock()
        send_response = AsyncMock()
        deps = PipelineDeps(
            line=line_service,
            memory=memory_service,
            message_cache=cache_service,
            orchestrator=SimpleNamespace(route=AsyncMock()),
            get_agent=lambda name: None,
            process_input=AsyncMock(return_value=request),
            enrich_request=enrich_request,
            extract_text=Mock(return_value="!new"),
            send_response=send_response,
        )
        await process_event(event, deps)

        line_service.send_loading_animation.assert_not_called()
        # Messages sent after !new (by its event time) are kept.
        memory_service.clear_chat.assert_awaited_once_with(
            source_type="group",
            chat_scope="multi",
            chat_id="G1",
            up_to=request.received_at,
        )
        line_service.send_text.assert_awaited_once_with(
            "reply-token",
            "G1",
            "Let's start a new chat!",
        )
        cache_service.cache_processed_request.assert_not_called()
        enrich_request.assert_not_called()
        send_response.assert_not_called()


class InternalCronEndpointTests(unittest.TestCase):
    """Cover /internal/cron auth, time parsing, and dispatch wiring."""

    def setUp(self) -> None:
        _reset_event_dedupe(self)
        _skip_startup_env_scan(self)
        self.settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
            internal_cron_secret="super-secret",
            log_level="ERROR",
        )

    @contextmanager
    def _client_with_scheduler(self, scheduler):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main_module, "get_settings", return_value=self.settings)
            )
            stack.enter_context(
                patch("src.services.scheduler_service.peek_scheduler_service", return_value=scheduler)
            )
            client = stack.enter_context(TestClient(main_module.app))
            yield client

    def test_lifespan_never_loads_settings_from_the_local_env(self) -> None:
        # Every cached get_settings() outside main reads the real .env.
        reads_env = AssertionError("get_settings() would read the local .env")
        scheduler = SimpleNamespace(
            dispatch_due_jobs=AsyncMock(return_value={"fired": [], "failed": [], "skipped_duplicate": []})
        )
        with ExitStack() as stack:
            for module in (
                "src.services.line_service",
                "src.services.memory_service",
                "src.services.rate_limit_service",
                "src.services.storage_service",
            ):
                stack.enter_context(patch(f"{module}.get_settings", side_effect=reads_env))
            client = stack.enter_context(self._client_with_scheduler(scheduler))
            response = client.post("/internal/cron", headers={"X-Cron-Token": "super-secret"})

        self.assertEqual(response.status_code, 200)

    def test_internal_cron_rejects_missing_token(self) -> None:
        scheduler = SimpleNamespace(
            dispatch_due_jobs=AsyncMock(return_value={"fired": [], "failed": [], "skipped_duplicate": []})
        )
        with self._client_with_scheduler(scheduler) as client:
            response = client.post("/internal/cron")
        self.assertEqual(response.status_code, 403)
        scheduler.dispatch_due_jobs.assert_not_called()

    def test_internal_cron_rejects_wrong_token(self) -> None:
        scheduler = SimpleNamespace(
            dispatch_due_jobs=AsyncMock(return_value={"fired": [], "failed": [], "skipped_duplicate": []})
        )
        with self._client_with_scheduler(scheduler) as client:
            response = client.post("/internal/cron", headers={"X-Cron-Token": "wrong"})
        self.assertEqual(response.status_code, 403)
        scheduler.dispatch_due_jobs.assert_not_called()

    def test_internal_cron_rejects_when_secret_empty(self) -> None:
        empty_settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
            internal_cron_secret="",
            log_level="ERROR",
        )
        scheduler = SimpleNamespace(
            dispatch_due_jobs=AsyncMock(return_value={"fired": [], "failed": [], "skipped_duplicate": []})
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main_module, "get_settings", return_value=empty_settings)
            )
            stack.enter_context(
                patch("src.services.scheduler_service.peek_scheduler_service", return_value=scheduler)
            )
            client = stack.enter_context(TestClient(main_module.app))
            response = client.post("/internal/cron", headers={"X-Cron-Token": "anything"})
        self.assertEqual(response.status_code, 403)
        scheduler.dispatch_due_jobs.assert_not_called()

    def test_internal_cron_dispatches_with_schedule_time_header(self) -> None:
        dispatched = {}

        async def fake_dispatch(*, reference_time=None, sender=None):
            dispatched["ref"] = reference_time
            dispatched["sender"] = sender
            return {"fired": ["weekly1"], "failed": [], "skipped_duplicate": []}

        scheduler = SimpleNamespace(dispatch_due_jobs=fake_dispatch)
        with self._client_with_scheduler(scheduler) as client:
            response = client.post(
                "/internal/cron",
                headers={
                    "X-Cron-Token": "super-secret",
                    "X-CloudScheduler-ScheduleTime": "2026-05-25T13:00:00Z",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["fired"], ["weekly1"])
        # The reference time must be tz-aware and equal to 2026-05-25T13:00:00 UTC.
        ref = dispatched["ref"]
        self.assertIsNotNone(ref)
        self.assertEqual(ref.tzinfo.utcoffset(ref), timedelta(0))
        self.assertEqual(ref.year, 2026)
        self.assertEqual(ref.month, 5)
        self.assertEqual(ref.day, 25)
        self.assertEqual(ref.hour, 13)
        self.assertEqual(ref.minute, 0)

    def test_internal_cron_handles_missing_schedule_time(self) -> None:
        async def fake_dispatch(*, reference_time=None, sender=None):
            return {
                "reference_time": "ignored",
                "fired": [],
                "failed": [],
                "skipped_duplicate": [],
                "reference_passed_as_none": reference_time is None,
            }

        scheduler = SimpleNamespace(dispatch_due_jobs=fake_dispatch)
        with self._client_with_scheduler(scheduler) as client:
            response = client.post(
                "/internal/cron",
                headers={"X-Cron-Token": "super-secret"},
            )

        self.assertEqual(response.status_code, 200)
        # When no schedule-time header is provided, endpoint passes
        # reference_time=None so SchedulerService falls back to wall-clock.
        self.assertTrue(response.json()["reference_passed_as_none"])

    def test_internal_cron_returns_200_when_scheduler_not_initialized(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main_module, "get_settings", return_value=self.settings)
            )
            stack.enter_context(
                patch("src.services.scheduler_service.peek_scheduler_service", return_value=None)
            )
            client = stack.enter_context(TestClient(main_module.app))
            response = client.post(
                "/internal/cron",
                headers={"X-Cron-Token": "super-secret"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reason"], "scheduler_not_initialized")


class SchedulerServiceDispatchTests(unittest.IsolatedAsyncioTestCase):
    """Verify the new pull-style SchedulerService logic that replaces APScheduler."""

    def _build(self):
        from src.services.scheduler_service import SchedulerService

        return SchedulerService(timezone_name="Asia/Taipei")

    async def test_weekly_job_fires_when_minute_matches_in_local_timezone(self) -> None:
        scheduler = self._build()
        scheduler.add_weekly_message("weekly1", "mon", 21, 0, "G", "msg")

        sent: list[tuple[str, str]] = []

        async def sender(g, m):
            sent.append((g, m))
            return True

        # 2026-05-25 Mon 21:00 Asia/Taipei == 13:00 UTC
        ref = datetime(2026, 5, 25, 13, 0, tzinfo=UTC)
        result = await scheduler.dispatch_due_jobs(reference_time=ref, sender=sender)

        self.assertEqual(result["fired"], ["weekly1"])
        self.assertEqual(sent, [("G", "msg")])

    async def test_weekly_job_skipped_when_minute_does_not_match(self) -> None:
        scheduler = self._build()
        scheduler.add_weekly_message("weekly1", "mon", 21, 0, "G", "msg")

        async def sender(g, m):
            return True

        # Mon 21:01 Asia/Taipei == 13:01 UTC -- not matching :00
        ref = datetime(2026, 5, 25, 13, 1, tzinfo=UTC)
        result = await scheduler.dispatch_due_jobs(reference_time=ref, sender=sender)
        self.assertEqual(result["fired"], [])

    async def test_dedupe_within_same_minute(self) -> None:
        scheduler = self._build()
        scheduler.add_weekly_message("weekly1", "mon", 21, 0, "G", "msg")

        async def sender(g, m):
            return True

        ref = datetime(2026, 5, 25, 13, 0, tzinfo=UTC)
        first = await scheduler.dispatch_due_jobs(reference_time=ref, sender=sender)
        second = await scheduler.dispatch_due_jobs(reference_time=ref, sender=sender)
        self.assertEqual(first["fired"], ["weekly1"])
        self.assertEqual(second["fired"], [])
        self.assertEqual(second["skipped_duplicate"], ["weekly1"])

    async def test_yearly_job_matches_in_local_timezone(self) -> None:
        scheduler = self._build()
        scheduler.add_yearly_message("bday", 4, 22, 9, 0, "G", "happy bday")

        sent: list[tuple[str, str]] = []

        async def sender(g, m):
            sent.append((g, m))
            return True

        # 2027-04-22 09:00 Asia/Taipei == 01:00 UTC
        ref = datetime(2027, 4, 22, 1, 0, tzinfo=UTC)
        result = await scheduler.dispatch_due_jobs(reference_time=ref, sender=sender)
        self.assertEqual(result["fired"], ["bday"])
        self.assertEqual(sent, [("G", "happy bday")])

    # ── Leap-day (Feb 29) yearly jobs ────────────────────────
    # Policy: on non-leap years a 02-29 job is clamped to 02-28 so the
    # reminder still fires every year. See ADR-0002.

    def _leap_job(self):
        from src.services.scheduler_service import _YearlyJob

        return _YearlyJob(
            id="leap", month=2, day=29, hour=9, minute=0,
            group_id="G", message="leap bday",
        )

    def test_leap_day_next_run_does_not_raise_in_non_leap_years(self) -> None:
        """Regression: next_run used to raise ValueError when ref.year + 1
        was also a non-leap year, which crashed lifespan startup."""
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Taipei")
        job = self._leap_job()
        # Every year in a full leap cycle must be safe.
        for year in range(2024, 2034):
            for month, day in ((1, 1), (3, 1), (12, 31)):
                ref = datetime(year, month, day, 0, 0, tzinfo=tz)
                with self.subTest(ref=ref.isoformat()):
                    nxt = job.next_run(ref)  # must not raise
                    self.assertGreater(nxt, ref)
                    self.assertEqual(nxt.month, 2)
                    self.assertIn(nxt.day, (28, 29))

    def test_leap_day_next_run_clamps_to_feb_28_on_non_leap_years(self) -> None:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Taipei")
        job = self._leap_job()

        # 2026 is not a leap year -> clamp to 02-28
        nxt = job.next_run(datetime(2026, 1, 1, 0, 0, tzinfo=tz))
        self.assertEqual((nxt.year, nxt.month, nxt.day), (2026, 2, 28))

        # 2028 IS a leap year -> real 02-29
        nxt = job.next_run(datetime(2028, 1, 1, 0, 0, tzinfo=tz))
        self.assertEqual((nxt.year, nxt.month, nxt.day), (2028, 2, 29))

        # Already past this year's occurrence -> roll to next year
        nxt = job.next_run(datetime(2028, 3, 1, 0, 0, tzinfo=tz))
        self.assertEqual((nxt.year, nxt.month, nxt.day), (2029, 2, 28))

    def test_leap_day_job_fires_on_feb_28_in_non_leap_year(self) -> None:
        """A 02-29 reminder must still be delivered in non-leap years."""
        job = self._leap_job()
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Taipei")
        # 2026 is not a leap year: the job should match 02-28.
        self.assertTrue(job.matches(datetime(2026, 2, 28, 9, 0, tzinfo=tz)))
        # 2028 is a leap year: it must match the real 02-29, not 02-28.
        self.assertTrue(job.matches(datetime(2028, 2, 29, 9, 0, tzinfo=tz)))
        self.assertFalse(job.matches(datetime(2028, 2, 28, 9, 0, tzinfo=tz)))

    def test_get_stats_with_leap_day_job_does_not_raise(self) -> None:
        """/health and lifespan both call list_jobs(); it must never raise."""
        scheduler = self._build()
        scheduler.add_yearly_message("leap", 2, 29, 9, 0, "G", "leap bday")
        stats = scheduler.get_stats()  # must not raise
        self.assertEqual(stats["job_count"], 1)
        self.assertIn("next_run", stats["jobs"][0])

    async def test_failed_sender_reported_but_does_not_raise(self) -> None:
        scheduler = self._build()
        scheduler.add_weekly_message("weekly1", "mon", 21, 0, "G", "msg")

        async def sender(g, m):
            return False

        ref = datetime(2026, 5, 25, 13, 0, tzinfo=UTC)
        result = await scheduler.dispatch_due_jobs(reference_time=ref, sender=sender)
        self.assertEqual(result["fired"], [])
        self.assertEqual(result["failed"], ["weekly1"])

    async def test_remove_job_excludes_from_dispatch(self) -> None:
        scheduler = self._build()
        scheduler.add_weekly_message("weekly1", "mon", 21, 0, "G", "msg")
        self.assertTrue(scheduler.remove_job("weekly1"))

        async def sender(g, m):
            return True

        ref = datetime(2026, 5, 25, 13, 0, tzinfo=UTC)
        result = await scheduler.dispatch_due_jobs(reference_time=ref, sender=sender)
        self.assertEqual(result["fired"], [])

    async def test_get_stats_shape_preserves_health_dashboard_contract(self) -> None:
        scheduler = self._build()
        scheduler.add_weekly_message("weekly1", "mon", 21, 0, "G", "msg")
        scheduler.add_yearly_message("bday", 4, 22, 9, 0, "G", "happy")
        stats = scheduler.get_stats()
        self.assertIn("running", stats)
        self.assertEqual(stats["job_count"], 2)
        self.assertEqual(len(stats["jobs"]), 2)
        for job in stats["jobs"]:
            self.assertIn("id", job)
            self.assertIn("name", job)
            self.assertIn("next_run", job)
            self.assertIn("trigger", job)
