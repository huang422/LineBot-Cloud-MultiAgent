from __future__ import annotations

import base64
import hashlib
import hmac
import json
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

import main as main_module
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse


def _sign_body(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


class ApiEndpointIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        main_module._background_tasks.clear()
        self.addCleanup(main_module._background_tasks.clear)

        self.settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
            log_level="ERROR",
        )
        self.fake_line = SimpleNamespace(
            get_push_stats=lambda: {
                "enabled": True,
                "monthly_limit": 0,
                "used": 0,
                "remaining": 0,
            }
        )
        self.fake_conversation = SimpleNamespace(
            get_stats=lambda: {
                "groups_tracked": 0,
                "total_messages": 0,
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
        self.fake_web_search = SimpleNamespace(
            get_quota_stats=lambda: {
                "configured": False,
                "used": 0,
                "quota": 0,
                "remaining": 0,
                "scope": "per-instance",
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
                patch.object(
                    main_module,
                    "get_conversation_service",
                    return_value=self.fake_conversation,
                )
            )
            stack.enter_context(
                patch.object(main_module, "close_line_service", new=AsyncMock())
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
            stack.enter_context(
                patch(
                    "src.services.web_search_service.get_web_search_service",
                    return_value=self.fake_web_search,
                )
            )
            stack.enter_context(
                patch(
                    "src.services.web_search_service.close_web_search_service",
                    new=AsyncMock(),
                )
            )
            client = stack.enter_context(TestClient(main_module.app))
            yield client

    def _signed_headers(self, body: bytes) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Line-Signature": _sign_body(self.settings.line_channel_secret, body),
        }

    def test_health_endpoint_returns_structured_status(self) -> None:
        with self._client() as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["status"], "healthy")
        self.assertTrue(payload["ready_for_webhook"])
        self.assertEqual(payload["warnings"], [])
        self.assertTrue(payload["providers"]["line"])
        self.assertTrue(payload["providers"]["openrouter"])
        self.assertIn("orchestrator", payload["agents_stats"])
        self.assertIn("chat", payload["agents_stats"])
        self.assertEqual(payload["conversation"]["groups_tracked"], 0)
        self.assertEqual(payload["scheduler"]["job_count"], 0)
        self.assertTrue(payload["cost_controls"]["line_push"]["enabled"])

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
        cache_service = SimpleNamespace(cache_event_message=AsyncMock())

        with self._client() as client, patch.object(
            main_module, "should_handle", return_value=False
        ), patch.object(
            main_module, "get_message_cache_service", return_value=cache_service
        ), patch.object(
            main_module, "_process_event", AsyncMock()
        ) as process_event:
            response = client.post(
                "/webhook",
                content=body,
                headers=self._signed_headers(body),
            )

        self.assertEqual(response.status_code, 200)
        cache_service.cache_event_message.assert_called_once_with(event)
        process_event.assert_not_called()

    def test_webhook_schedules_background_processing_for_triggered_events(self) -> None:
        event = {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "G1", "userId": "U1"},
            "message": {"type": "text", "id": "M2", "text": "!hej 幫我整理"},
        }
        body = json.dumps({"events": [event]}).encode("utf-8")
        cache_service = SimpleNamespace(cache_event_message=AsyncMock())

        with self._client() as client, patch.object(
            main_module, "should_handle", return_value=True
        ), patch.object(
            main_module, "get_message_cache_service", return_value=cache_service
        ), patch.object(
            main_module, "_process_event", AsyncMock()
        ) as process_event:
            response = client.post(
                "/webhook",
                content=body,
                headers=self._signed_headers(body),
            )

        self.assertEqual(response.status_code, 200)
        process_event.assert_called_once_with(event)
        cache_service.cache_event_message.assert_not_called()


class BackgroundProcessingIntegrationTests(unittest.IsolatedAsyncioTestCase):
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
        )
        cache_service = SimpleNamespace(cache_processed_request=Mock())
        decision = SimpleNamespace(
            agent="chat",
            output_format="text",
            task_description="整理使用者需求後回覆",
            reasoning="一般文字任務",
        )
        response = AgentResponse(
            text="這是整理後的回覆",
            agent_name="chat",
            output_format="text",
        )
        agent = SimpleNamespace(process=AsyncMock(return_value=response))
        orchestrator = SimpleNamespace(route=AsyncMock(return_value=decision))

        with patch.object(
            main_module, "get_line_service", return_value=line_service
        ), patch.object(
            main_module, "process_input", AsyncMock(return_value=request)
        ), patch.object(
            main_module, "extract_text", return_value="幫我整理重點"
        ), patch.object(
            main_module, "get_message_cache_service", return_value=cache_service
        ), patch.object(
            main_module, "enrich_request", AsyncMock(return_value=request)
        ), patch.object(
            main_module, "orchestrator", orchestrator
        ), patch.object(
            main_module, "_get_agent", return_value=agent
        ), patch.object(
            main_module, "send_response", AsyncMock(return_value=True)
        ) as send_response, patch.object(
            main_module, "record_conversation"
        ) as record_conversation:
            await main_module._process_event(event)

        line_service.send_loading_animation.assert_awaited_once_with("G1")
        cache_service.cache_processed_request.assert_called_once_with(event, request)
        orchestrator.route.assert_awaited_once_with(request)
        agent.process.assert_awaited_once_with(request)
        send_response.assert_awaited_once_with(request, response)
        line_service.send_text.assert_not_called()
        record_conversation.assert_called_once_with(
            request,
            "這是整理後的回覆",
            assistant_delivered=True,
        )
        self.assertEqual(request.text, "幫我整理重點")
        self.assertEqual(request.target_agent, "chat")
        self.assertEqual(request.output_format, "text")
