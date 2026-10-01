"""Cloud Tasks for post-webhook work (ADR-0006).

Covers the queue client (request shape, limits, failures), the task
handlers, the ``/internal/tasks/{kind}`` endpoint, the webhook's
enqueue / inline-fallback behaviour, the cron maintenance sweep, compaction
scheduling, and the reply-window groundwork. No network: Cloud Tasks is an
httpx.MockTransport and everything else is a test double.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient

import main as main_module
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.models.agent_response import AgentResponse
from src.pipeline import event_pipeline as event_pipeline_module
from src.pipeline.event_pipeline import PipelineDeps, process_event, record_group_message
from src.pipeline.maintenance import CronDeps, run_cron_tick
from src.pipeline.task_handlers import TaskDeps, event_dedupe_key, handle_task
from src.processors.input_processor import event_received_at, process_input
from src.providers.base import ProviderResponse
from src.providers.fallback_chain import Tier
from src.runtime.composition import build_compaction_scheduler, build_task_queue
from src.services import task_queue as task_queue_module
from src.services.memory_backends import InMemoryBackend, MemoryBackendError
from src.services.memory_service import MemoryService, MemoryServiceError
from src.services.message_cache_service import MessageCacheService
from src.services.model_health import ModelHealth
from src.services.task_queue import (
    MAX_TASK_PAYLOAD_BYTES,
    BaseUrlResolver,
    CloudTasksQueue,
    InlineTaskQueue,
    TaskMaybeQueuedError,
    TaskNotQueuedError,
    enqueue_only,
    enqueue_or_run,
    select_tasks_mode,
)
from src.utils.dedupe import RecentKeys
from src.utils.gcp_metadata import MetadataError
from tests.test_api_integration import _reset_event_dedupe, _sign_body, _skip_startup_env_scan

_TASKS_SECRET = "tasks-secret-value"
_TOKEN = "ya29.metadata-token"


class _FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ── The queue client ─────────────────────────────────────────


class CloudTasksQueueTests(unittest.IsolatedAsyncioTestCase):
    def _queue(self, handler, **overrides) -> tuple[CloudTasksQueue, list[httpx.Request]]:
        requests: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(record))
        self.addAsyncCleanup(client.aclose)
        kwargs = {
            "project": "my-project",
            "location": "us-west1",
            "queue": "linebot-events",
            "secret": _TASKS_SECRET,
            "base_url_provider": lambda: "https://bot.example.run.app",
            "token_provider": AsyncMock(return_value=_TOKEN),
            "client": client,
            "dispatch_deadline_seconds": 300,
        }
        kwargs.update(overrides)
        return CloudTasksQueue(**kwargs), requests

    async def test_request_shape(self) -> None:
        queue, requests = self._queue(lambda request: httpx.Response(200, json={"name": "t1"}))
        payload = {"event": {"type": "message", "message": {"text": "幫我整理"}}}

        self.assertTrue(await queue.enqueue("event", payload))

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            str(request.url),
            "https://cloudtasks.googleapis.com/v2/projects/my-project/locations/us-west1"
            "/queues/linebot-events/tasks",
        )
        self.assertEqual(request.headers["Authorization"], f"Bearer {_TOKEN}")
        task = json.loads(request.content)["task"]
        # Named, so a retry cannot create a second task; random, not sequential.
        self.assertRegex(
            task["name"],
            r"^projects/my-project/locations/us-west1/queues/linebot-events/tasks/[0-9a-f]{32}$",
        )
        self.assertEqual(task["dispatchDeadline"], "300s")
        http_request = task["httpRequest"]
        self.assertEqual(http_request["url"], "https://bot.example.run.app/internal/tasks/event")
        self.assertEqual(http_request["httpMethod"], "POST")
        self.assertEqual(
            http_request["headers"],
            {"Content-Type": "application/json", "X-Tasks-Token": _TASKS_SECRET},
        )
        body = base64.b64decode(http_request["body"])
        self.assertEqual(json.loads(body), payload)
        # ensure_ascii=False keeps the payload (and the billed size) small.
        self.assertIn("幫我整理".encode(), body)

    async def test_a_kind_with_its_own_queue_goes_there(self) -> None:
        queue, requests = self._queue(
            lambda request: httpx.Response(200, json={}), kind_queues={"colab": "linebot-colab", "blank": " "}
        )

        for kind in ("colab", "event", "blank"):
            self.assertTrue(await queue.enqueue(kind, {"action": "reconcile"}))

        self.assertEqual(
            [str(request.url).split("/queues/")[1] for request in requests],
            ["linebot-colab/tasks", "linebot-events/tasks", "linebot-events/tasks"],
        )
        # The task still targets the kind's own handler.
        task = json.loads(requests[0].content)["task"]
        self.assertEqual(task["httpRequest"]["url"], "https://bot.example.run.app/internal/tasks/colab")

    async def test_unset_project_and_location_come_from_metadata_once(self) -> None:
        project_provider = AsyncMock(return_value="meta-project")
        location_provider = AsyncMock(return_value="asia-east1")
        queue, requests = self._queue(
            lambda request: httpx.Response(200, json={}),
            project="",
            location="",
            project_provider=project_provider,
            location_provider=location_provider,
        )

        self.assertTrue(await queue.enqueue("compact", {"chat_id": "U1"}))
        self.assertTrue(await queue.enqueue("compact", {"chat_id": "U2"}))

        self.assertIn(
            "/projects/meta-project/locations/asia-east1/queues/linebot-events/tasks",
            str(requests[1].url),
        )
        project_provider.assert_awaited_once()
        location_provider.assert_awaited_once()

    async def test_oversized_payload_is_not_sent(self) -> None:
        queue, requests = self._queue(lambda request: httpx.Response(200))
        payload = {"event": {"text": "x" * MAX_TASK_PAYLOAD_BYTES}}

        with patch.object(task_queue_module.logger, "warning") as log_warning:
            self.assertFalse(await queue.enqueue("event", payload))

        self.assertEqual(requests, [])
        self.assertIn("not queued", log_warning.call_args.args[0])

    async def test_unknown_base_url_is_not_sent(self) -> None:
        queue, requests = self._queue(
            lambda request: httpx.Response(200), base_url_provider=lambda: ""
        )

        self.assertFalse(await queue.enqueue("event", {"event": {}}))
        self.assertEqual(requests, [])

    async def test_a_4xx_returns_false_without_a_retry_and_never_logs_secrets(self) -> None:
        for status in (400, 403, 404, 409, 429):
            with self.subTest(status=status):
                queue, requests = self._queue(
                    lambda request, status=status: httpx.Response(status, text=_TASKS_SECRET)
                )
                with patch.object(task_queue_module.logger, "warning") as log_warning:
                    self.assertFalse(await queue.enqueue("event", {"event": {}}))

                self.assertEqual(len(requests), 1)
                message = log_warning.call_args.args[0]
                self.assertIn(f"HTTP {status}", message)
                self.assertNotIn(_TASKS_SECRET, message)
                self.assertNotIn(_TOKEN, message)

    async def test_google_error_status_and_message_are_logged_with_the_secret_masked(self) -> None:
        # 404 NOT_FOUND is both "no such queue" and "queue re-activating
        # after 30 idle days"; the status and message tell them apart.
        error = {
            "error": {
                "code": 404,
                "status": "NOT_FOUND",
                "message": f"Requested entity was not found. {_TASKS_SECRET} " + "x" * 400,
            }
        }
        queue, _ = self._queue(lambda request: httpx.Response(404, json=error))
        with patch.object(task_queue_module.logger, "warning") as log_warning:
            self.assertFalse(await queue.enqueue("event", {"event": {}}))

        message = log_warning.call_args.args[0]
        self.assertIn("HTTP 404", message)
        self.assertIn("NOT_FOUND: Requested entity was not found.", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn(_TASKS_SECRET, message)
        self.assertLess(len(message), 400)

    async def test_metadata_errors_are_logged_by_message(self) -> None:
        queue, requests = self._queue(
            lambda request: httpx.Response(200),
            token_provider=AsyncMock(
                side_effect=MetadataError("metadata instance/service-accounts/default/token returned HTTP 500")
            ),
        )
        with patch.object(task_queue_module.logger, "warning") as log_warning:
            self.assertFalse(await queue.enqueue("event", {"event": {}}))

        self.assertEqual(requests, [])
        self.assertIn("token returned HTTP 500", log_warning.call_args.args[0])

    async def test_transport_and_token_failures_return_false(self) -> None:
        # Nothing was sent, so nothing can have been created: no retry.
        for error in (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
            with self.subTest(error=error.__name__):

                def unreachable(request: httpx.Request, error=error) -> httpx.Response:
                    raise error("unreachable", request=request)

                queue, requests = self._queue(unreachable)
                self.assertFalse(await queue.enqueue("event", {"event": {}}))
                self.assertEqual(len(requests), 1)

        queue, requests = self._queue(
            lambda request: httpx.Response(200),
            token_provider=AsyncMock(side_effect=RuntimeError(f"bad {_TASKS_SECRET}")),
        )
        with patch.object(task_queue_module.logger, "warning") as log_warning:
            self.assertFalse(await queue.enqueue("event", {"event": {}}))
        self.assertEqual(requests, [])
        self.assertNotIn(_TASKS_SECRET, log_warning.call_args.args[0])

    def _cloud_tasks(self, *first_answers):
        """A Cloud Tasks stand-in that dedupes task names like the real one.

        Each entry of *first_answers* answers one call: an int is an error
        status and ConnectError never reaches the server (nothing created);
        any other exception class is raised after the create (the answer is
        lost). Later calls behave.
        """
        created: set[str] = set()
        answers = list(first_answers)

        def handler(request: httpx.Request) -> httpx.Response:
            name = json.loads(request.content)["task"]["name"]
            answer = answers.pop(0) if answers else None
            if isinstance(answer, int):
                return httpx.Response(answer, json={"error": {"code": answer, "status": "X"}})
            if answer is httpx.ConnectError:
                raise answer(_TASKS_SECRET, request=request)
            exists = name in created
            created.add(name)
            if answer is not None:
                raise answer(_TASKS_SECRET, request=request)
            if exists:
                return httpx.Response(409, json={"error": {"code": 409, "status": "ALREADY_EXISTS"}})
            return httpx.Response(200, json={"name": name})

        return handler, created

    async def test_an_unclear_answer_is_retried_under_the_same_name(self) -> None:
        # The first call created the task but its answer was lost: the retry
        # gets 409 ALREADY_EXISTS, which means queued (once).
        for error in (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ReadError):
            with self.subTest(error=error.__name__):
                handler, created = self._cloud_tasks(error)
                queue, requests = self._queue(handler)

                self.assertTrue(await queue.enqueue("cache_image", {"event": {}}))

                names = [json.loads(r.content)["task"]["name"] for r in requests]
                self.assertEqual(len(names), 2)
                self.assertEqual(names[0], names[1])
                self.assertEqual(len(created), 1)

    async def test_a_5xx_is_retried_under_the_same_name(self) -> None:
        handler, created = self._cloud_tasks(503)
        queue, requests = self._queue(handler)

        self.assertTrue(await queue.enqueue("passive_flush", {"key": "group::G1"}))

        self.assertEqual(len(requests), 2)
        self.assertEqual(len(created), 1)

    async def test_still_unclear_after_the_retry_raises_maybe_queued(self) -> None:
        # Only a 2xx or ALREADY_EXISTS on the retry settles it; a refusal or
        # an unsent retry says nothing about the first call.
        cases = {
            "timeouts": (httpx.ReadTimeout, httpx.ReadTimeout),
            "5xx twice": (500, 500),
            "then unreachable": (httpx.ReadTimeout, httpx.ConnectError),
            "then refused": (httpx.ReadTimeout, 403),
        }
        for label, answers in cases.items():
            with self.subTest(label):
                handler, _ = self._cloud_tasks(*answers)
                queue, requests = self._queue(handler)

                with self.assertRaises(TaskMaybeQueuedError) as caught:
                    await queue.enqueue("passive_flush", {"key": "group::G1"})

                self.assertEqual(len(requests), 2)
                self.assertIn("passive_flush", str(caught.exception))
                self.assertNotIn(_TASKS_SECRET, str(caught.exception))
                self.assertNotIn(_TOKEN, str(caught.exception))

    async def test_an_unclear_enqueue_does_not_also_run_the_work_inline(self) -> None:
        # Regression: a ReadTimeout after Cloud Tasks created the task used
        # to count as "not queued", so the work ran inline and again later.
        handler, created = self._cloud_tasks(httpx.ReadTimeout)
        queue, _ = self._queue(handler)
        run_inline = AsyncMock()

        for kind in ("cache_image", "passive_flush", "event"):
            self.assertTrue(await enqueue_or_run(queue, kind, {"event": {}}, run_inline))

        self.assertEqual(len(created), 3)
        run_inline.assert_not_called()


class TaskQueueHelpersTests(unittest.IsolatedAsyncioTestCase):
    async def test_inline_queue_awaits_the_dispatcher(self) -> None:
        dispatcher = AsyncMock(return_value=False)
        queue = InlineTaskQueue(dispatcher)

        self.assertTrue(await queue.enqueue("compact", {"chat_id": "U1"}))
        dispatcher.assert_awaited_once_with("compact", {"chat_id": "U1"})

    async def test_a_failing_inline_handler_runs_once(self) -> None:
        dispatcher = AsyncMock(side_effect=RuntimeError("deps unavailable"))
        queue = InlineTaskQueue(dispatcher)

        with patch.object(task_queue_module.logger, "error") as log_error:
            queued = await enqueue_or_run(queue, "event", {"event": {}}, dispatcher)

        # It was attempted, so there is no second "inline fallback" run.
        self.assertTrue(queued)
        dispatcher.assert_awaited_once_with("event", {"event": {}})
        log_error.assert_called_once()
        self.assertIn("failed while running inline", log_error.call_args.args[0])

    async def test_enqueue_or_run_only_runs_inline_when_queueing_fails(self) -> None:
        cases = {
            "queued": (AsyncMock(return_value=True), False),
            "returned False": (AsyncMock(return_value=False), True),
            "raised": (AsyncMock(side_effect=httpx.ConnectError("down")), True),
        }
        for label, (enqueue, runs_inline) in cases.items():
            with self.subTest(label):
                queue = SimpleNamespace(enqueue=enqueue)
                run_inline = AsyncMock()
                with patch.object(task_queue_module.logger, "error") as log_error:
                    queued = await enqueue_or_run(queue, "event", {"event": {}}, run_inline)

                self.assertEqual(queued, not runs_inline)
                if runs_inline:
                    run_inline.assert_awaited_once_with("event", {"event": {}})
                    self.assertIn("running it inline", log_error.call_args.args[0])
                else:
                    run_inline.assert_not_called()
                    log_error.assert_not_called()

    async def test_a_possibly_queued_task_runs_inline_only_when_asked(self) -> None:
        maybe = TaskMaybeQueuedError("task passive_flush (abc): ReadTimeout after 6000 ms")
        for inline_if_unsure in (False, True):
            with self.subTest(inline_if_unsure=inline_if_unsure):
                queue = SimpleNamespace(enqueue=AsyncMock(side_effect=maybe))
                run_inline = AsyncMock()
                with patch.object(task_queue_module.logger, "error") as log_error:
                    queued = await enqueue_or_run(
                        queue, "passive_flush", {"key": "k"}, run_inline, inline_if_unsure=inline_if_unsure
                    )

                self.assertEqual(queued, not inline_if_unsure)
                self.assertEqual(run_inline.await_count, int(inline_if_unsure))
                message = log_error.call_args.args[0]
                # DEPLOY.md's log filter matches "not queued".
                self.assertIn("not queued", message)
                self.assertIn("ReadTimeout", message)

    async def test_enqueue_only_never_runs_the_work_here(self) -> None:
        await enqueue_only(SimpleNamespace(enqueue=AsyncMock(return_value=True)), "colab", {"action": "assign"})
        for label, enqueue in {
            "returned False": AsyncMock(return_value=False),
            "raised": AsyncMock(side_effect=httpx.ConnectError("down")),
            "maybe queued": AsyncMock(side_effect=TaskMaybeQueuedError("task colab: ReadTimeout")),
        }.items():
            with self.subTest(label), self.assertRaises(TaskNotQueuedError) as caught:
                await enqueue_only(SimpleNamespace(enqueue=enqueue), "colab", {"action": "assign"})
            self.assertIn("colab", str(caught.exception))

    async def test_a_colab_task_that_cannot_be_queued_is_not_run_inline(self) -> None:
        # The cron request must never run a minutes-long Colab call itself.
        dispatch = AsyncMock()
        queue = SimpleNamespace(enqueue=AsyncMock(return_value=False))
        with (
            patch.object(main_module, "container", SimpleNamespace(task_queue=queue)),
            patch.object(main_module, "_dispatch_task", dispatch),
            self.assertRaises(TaskNotQueuedError),
        ):
            await main_module._queue_colab_task("colab", {"action": "assign", "nbh": "n"})
        queue.enqueue.assert_awaited_once_with("colab", {"action": "assign", "nbh": "n"})
        dispatch.assert_not_called()
        with patch.object(main_module, "container", None), self.assertRaises(TaskNotQueuedError):
            await main_module._queue_colab_task("colab", {"action": "reconcile"})

    def test_select_tasks_mode(self) -> None:
        def settings(mode: str, secret: str) -> Settings:
            return Settings(_env_file=None, tasks_mode=mode, internal_tasks_secret=secret)

        on_cloud_run = {"K_SERVICE": "linebot-cloud-agent"}
        cases = [
            ("auto", "s", on_cloud_run, "cloud"),
            ("auto", "", on_cloud_run, "inline"),
            ("auto", "s", {}, "inline"),
            ("cloud", "s", {}, "cloud"),
            # Without the secret every task would be refused with 403.
            ("cloud", "", on_cloud_run, "inline"),
            ("inline", "s", on_cloud_run, "inline"),
        ]
        for mode, secret, environ, expected in cases:
            with self.subTest(mode=mode, secret=bool(secret), environ=environ):
                self.assertEqual(select_tasks_mode(settings(mode, secret), environ), expected)

    def test_build_task_queue_picks_the_queue_class(self) -> None:
        dispatcher = AsyncMock()
        cloud = build_task_queue(
            Settings(_env_file=None, internal_tasks_secret="s", task_dispatch_deadline_seconds=600),
            dispatcher=dispatcher,
            base_url_provider=lambda: "https://x",
            environ={"K_SERVICE": "svc"},
        )
        self.assertIsInstance(cloud, CloudTasksQueue)
        self.assertEqual(cloud._deadline, 600)
        self.assertEqual(cloud._queue, "linebot-events")
        self.assertEqual(cloud._location, "us-west1")
        # No Colab, no ``colab`` tasks: nothing routed elsewhere.
        self.assertEqual(cloud.queue_for("colab"), "linebot-events")

        with patch("src.runtime.composition.logger.info") as log_info:
            colab = build_task_queue(
                Settings(_env_file=None, internal_tasks_secret="s", colab_enabled=True, cloud_tasks_colab_queue="cq"),
                dispatcher=dispatcher,
                base_url_provider=lambda: "https://x",
                environ={"K_SERVICE": "svc"},
            )
        self.assertEqual((colab.queue_for("colab"), colab.queue_for("event")), ("cq", "linebot-events"))
        self.assertIn("colab_queue=cq", log_info.call_args.args[0])

        inline = build_task_queue(
            Settings(_env_file=None, internal_tasks_secret="s"),
            dispatcher=dispatcher,
            base_url_provider=lambda: "",
            environ={},
        )
        self.assertIsInstance(inline, InlineTaskQueue)

    def test_an_empty_queue_name_means_the_default_queue(self) -> None:
        # As in the deploy script, which then creates linebot-events. An
        # empty name made every CreateTask fail (…/queues//tasks), so every
        # task ran inline inside /webhook.
        settings = Settings(_env_file=None, internal_tasks_secret="s", cloud_tasks_queue=" ")
        self.assertEqual(settings.cloud_tasks_queue, "linebot-events")
        queue = build_task_queue(
            settings,
            dispatcher=AsyncMock(),
            base_url_provider=lambda: "https://x",
            environ={"K_SERVICE": "svc"},
        )
        self.assertEqual(queue.queue_for("event"), "linebot-events")
        self.assertIn(
            '[[ -z "$CLOUD_TASKS_QUEUE_VALUE" ]] && CLOUD_TASKS_QUEUE_VALUE="linebot-events"',
            _DEPLOY_SCRIPT.read_text(encoding="utf-8"),
        )

    def test_build_task_queue_warns_when_the_turn_outlasts_the_handler(self) -> None:
        from src.runtime import composition

        # The last EVENT_NOTICE_SECONDS (10) of the 280 s budget are the notice's.
        for turn, warned in ((240, False), (269, False), (270, True), (280, True)):
            with self.subTest(turn=turn), patch.object(composition.logger, "warning") as warning:
                build_task_queue(
                    Settings(_env_file=None, turn_deadline_seconds=turn),
                    dispatcher=AsyncMock(),
                    base_url_provider=lambda: "",
                    environ={},
                )
                messages = [call.args[0] for call in warning.call_args_list]
                self.assertEqual(
                    any("TURN_DEADLINE_SECONDS should stay below" in m for m in messages), warned
                )

    def test_base_url_resolver(self) -> None:
        configured = BaseUrlResolver("https://bot.example.com/")
        configured.remember({"host": "evil.example.com"})
        self.assertEqual(configured(), "https://bot.example.com")

        derived = BaseUrlResolver("")
        self.assertEqual(derived(), "")
        derived.remember({"host": "svc-abc-uw.a.run.app", "x-forwarded-proto": "https, http"})
        self.assertEqual(derived(), "https://svc-abc-uw.a.run.app")
        derived.remember({"host": "other.example.com"})  # cached: first wins
        self.assertEqual(derived(), "https://svc-abc-uw.a.run.app")

        # Always https: the task carries the secret, and an http target
        # would hit run.app's redirect (one failed, dropped attempt).
        plain = BaseUrlResolver("")
        plain.remember({"host": "svc-abc-uw.a.run.app", "x-forwarded-proto": "http"})
        self.assertEqual(plain(), "https://svc-abc-uw.a.run.app")

        rejected = BaseUrlResolver("")
        for headers in ({"host": "a.example.com/evil"}, {"host": ""}, {}):
            rejected.remember(headers)
        self.assertEqual(rejected(), "")


class RecentKeysTests(unittest.TestCase):
    def test_add_reports_duplicates_until_the_ttl_passes(self) -> None:
        clock = _FakeClock()
        seen = RecentKeys(ttl_seconds=900, max_size=10, clock=clock)

        self.assertTrue(seen.add("e1"))
        self.assertFalse(seen.add("e1"))
        clock.now += 899
        self.assertIn("e1", seen)
        clock.now += 1
        self.assertNotIn("e1", seen)
        self.assertTrue(seen.add("e1"))

    def test_size_is_bounded_oldest_first(self) -> None:
        seen = RecentKeys(ttl_seconds=900, max_size=3, clock=_FakeClock())
        for key in ("a", "b", "c", "d"):
            seen.add(key)

        self.assertEqual(len(seen), 3)
        self.assertNotIn("a", seen)
        self.assertIn("d", seen)

    def test_discard_and_clear(self) -> None:
        seen = RecentKeys(clock=_FakeClock())
        seen.add("a")
        seen.add("b")
        seen.discard("a")
        self.assertTrue(seen.add("a"))
        seen.clear()
        self.assertEqual(len(seen), 0)


# ── Task handlers ────────────────────────────────────────────


def _pipeline(**overrides) -> PipelineDeps:
    values = {
        "line": SimpleNamespace(),
        "memory": SimpleNamespace(flush_passive=AsyncMock(), run_compaction=AsyncMock()),
        "message_cache": SimpleNamespace(cache_event_message=AsyncMock()),
        "orchestrator": SimpleNamespace(),
        "get_agent": lambda name: None,
    }
    values.update(overrides)
    return PipelineDeps(**values)


def _group_event(event_id: str = "01EVT", text: str = "!hej 你好", **extra) -> dict:
    event = {
        "type": "message",
        "webhookEventId": event_id,
        "replyToken": "reply-token",
        "source": {"type": "group", "groupId": "G1", "userId": "U1"},
        "message": {"type": "text", "id": f"M-{event_id}", "text": text},
    }
    event.update(extra)
    return event


class TaskHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_kind_reaches_its_collaborator(self) -> None:
        pipeline = _pipeline()
        process = AsyncMock()
        deps = TaskDeps(pipeline=pipeline, process_event=process)
        event = _group_event()

        self.assertTrue(await handle_task("event", {"event": event}, deps))
        process.assert_awaited_once()
        self.assertIs(process.await_args.args[0], event)
        self.assertIs(process.await_args.args[1].memory, pipeline.memory)

        self.assertTrue(await handle_task("passive_flush", {"key": "group::G1"}, deps))
        pipeline.memory.flush_passive.assert_awaited_once_with("group::G1")

        self.assertTrue(await handle_task("cache_image", {"event": event}, deps))
        pipeline.message_cache.cache_event_message.assert_awaited_once_with(event)

        payload = {"source_type": "group", "chat_scope": "multi", "chat_id": "G1"}
        self.assertTrue(await handle_task("compact", payload, deps))
        pipeline.memory.run_compaction.assert_awaited_once_with(
            source_type="group", chat_scope="multi", chat_id="G1"
        )

    async def test_the_eol_task_runs_the_lifecycle_check(self) -> None:
        lifecycle = SimpleNamespace(run=AsyncMock(return_value={}))

        self.assertTrue(
            await handle_task("eol", {}, TaskDeps(pipeline=_pipeline(), lifecycle=lifecycle))
        )
        lifecycle.run.assert_awaited_once_with()

        with patch("src.pipeline.task_handlers.logger.warning") as log_warning:
            self.assertFalse(await handle_task("eol", {}, TaskDeps(pipeline=_pipeline())))
        log_warning.assert_called_once()

    async def test_unknown_kind_and_malformed_payloads_are_ignored(self) -> None:
        pipeline = _pipeline()
        process = AsyncMock()
        deps = TaskDeps(pipeline=pipeline, process_event=process)

        with patch("src.pipeline.task_handlers.logger.warning") as log_warning:
            self.assertFalse(await handle_task("colab", {"x": 1}, deps))
            self.assertFalse(await handle_task("event", {"event": "not-a-dict"}, deps))
            self.assertFalse(await handle_task("passive_flush", {}, deps))
            self.assertFalse(await handle_task("cache_image", {}, deps))
            self.assertFalse(await handle_task("compact", {"source_type": "user"}, deps))

        self.assertEqual(log_warning.call_count, 5)
        process.assert_not_called()
        pipeline.memory.flush_passive.assert_not_called()
        pipeline.memory.run_compaction.assert_not_called()

    async def test_the_colab_task_runs_the_controller_action(self) -> None:
        colab = SimpleNamespace(run_action=AsyncMock(return_value=True))
        deps = TaskDeps(pipeline=_pipeline(), colab=colab)

        payload = {"action": "assign", "nbh": "n-1"}
        self.assertTrue(await handle_task("colab", payload, deps))
        colab.run_action.assert_awaited_once_with("assign", payload)

        with patch("src.pipeline.task_handlers.logger.warning") as log_warning:
            self.assertFalse(await handle_task("colab", {"nbh": "n-1"}, deps))
            self.assertFalse(await handle_task("colab", {"action": "assign"}, TaskDeps(pipeline=_pipeline())))
        self.assertEqual(log_warning.call_count, 2)
        colab.run_action.return_value = False
        self.assertFalse(await handle_task("colab", {"action": "reboot"}, deps))

    async def test_a_colab_action_is_cut_off_by_the_handler_budget(self) -> None:
        async def slow(action, payload):
            await asyncio.sleep(10)

        deps = TaskDeps(pipeline=_pipeline(), colab=SimpleNamespace(run_action=slow), budget_seconds=0.05)
        with patch("src.pipeline.task_handlers.logger.error"):
            self.assertFalse(await handle_task("colab", {"action": "assign"}, deps))

    async def test_duplicate_event_deliveries_run_once(self) -> None:
        process = AsyncMock()
        deps = TaskDeps(pipeline=_pipeline(), process_event=process)

        self.assertTrue(await handle_task("event", {"event": _group_event("01A")}, deps))
        self.assertTrue(await handle_task("event", {"event": _group_event("01A")}, deps))
        self.assertTrue(await handle_task("event", {"event": _group_event("01B")}, deps))

        self.assertEqual(process.await_count, 2)

    def test_event_dedupe_key_prefers_webhook_event_id(self) -> None:
        self.assertEqual(event_dedupe_key(_group_event("01EVT")), "01EVT")
        no_id = _group_event("")
        self.assertEqual(event_dedupe_key(no_id), "message:M-")
        no_id["message"]["id"] = "M9"
        self.assertEqual(event_dedupe_key(no_id), "message:M9")
        self.assertEqual(event_dedupe_key({"type": "follow"}), "")

    async def test_handler_budget_cancels_slow_work(self) -> None:
        cancelled = asyncio.Event()

        async def slow(event, pipeline):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        deps = TaskDeps(pipeline=_pipeline(), budget_seconds=0.01, process_event=slow)
        with patch("src.pipeline.task_handlers.logger.error") as log_error:
            self.assertFalse(await handle_task("event", {"event": _group_event()}, deps))

        self.assertTrue(cancelled.is_set())
        self.assertIn("budget", log_error.call_args.args[0])

    async def test_a_turn_cut_off_by_the_budget_tells_the_user_before_the_budget_ends(self) -> None:
        from src.pipeline import task_handlers

        loop = asyncio.get_running_loop()
        sent_at: list[float] = []

        async def send_text(reply_token, to, text):
            sent_at.append(loop.time())
            return True

        async def slow(event, pipeline):
            await asyncio.sleep(5)  # a TTS or Firestore stall before the answer goes out

        line = SimpleNamespace(send_text=AsyncMock(side_effect=send_text))
        deps = TaskDeps(pipeline=_pipeline(line=line), budget_seconds=0.2, process_event=slow)
        started = loop.time()
        with patch.object(task_handlers, "EVENT_NOTICE_SECONDS", 0.1), patch(
            "src.pipeline.task_handlers.logger.error"
        ):
            self.assertFalse(await handle_task("event", {"event": _group_event()}, deps))

        line.send_text.assert_awaited_once_with("reply-token", "G1", task_handlers.EVENT_TIMEOUT_TEXT)
        self.assertLess(sent_at[0] - started, 0.2)

    async def test_no_timeout_notice_after_the_answer_went_out(self) -> None:
        from src.pipeline import task_handlers

        line = SimpleNamespace(send_text=AsyncMock(return_value=True))

        async def answered_then_stalled(event, pipeline):
            self.assertTrue(await pipeline.send_response(SimpleNamespace(), SimpleNamespace()))
            await asyncio.sleep(5)  # the memory write after the reply hangs

        pipeline = _pipeline(line=line, send_response=AsyncMock(return_value=True))
        deps = TaskDeps(pipeline=pipeline, budget_seconds=0.2, process_event=answered_then_stalled)
        with patch.object(task_handlers, "EVENT_NOTICE_SECONDS", 0.1), patch(
            "src.pipeline.task_handlers.logger.error"
        ):
            self.assertFalse(await handle_task("event", {"event": _group_event()}, deps))

        pipeline.send_response.assert_awaited_once()
        line.send_text.assert_not_called()

    async def test_handler_errors_are_reported_not_raised(self) -> None:
        pipeline = _pipeline(
            memory=SimpleNamespace(flush_passive=AsyncMock(side_effect=RuntimeError("boom")))
        )
        with patch("src.pipeline.task_handlers.logger.error") as log_error:
            ok = await handle_task("passive_flush", {"key": "group::G1"}, TaskDeps(pipeline=pipeline))

        self.assertFalse(ok)
        log_error.assert_called_once()


# ── HTTP surface ─────────────────────────────────────────────


class _RecordingQueue:
    """Stands in for CloudTasksQueue; records what the app queued."""

    mode = "fake"

    def __init__(self, result: bool = True, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.result = result
        self.error = error

    async def enqueue(self, kind: str, payload: dict) -> bool:
        self.calls.append((kind, payload))
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        return None


class _AppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        _reset_event_dedupe(self)
        _skip_startup_env_scan(self)
        self.settings = Settings(
            _env_file=None,
            line_channel_secret="secret",
            line_channel_access_token="token",
            openrouter_api_key="key",
            internal_cron_secret="cron-secret",
            internal_tasks_secret=_TASKS_SECRET,
            log_level="ERROR",
        )
        self.memory = SimpleNamespace(
            enqueue_passive_message=Mock(return_value=False),
            flush_passive=AsyncMock(),
            flush_due_passive=AsyncMock(return_value=[]),
            run_compaction=AsyncMock(),
        )
        self.cache = SimpleNamespace(
            cache_event_text=Mock(return_value=True),
            cache_event_message=AsyncMock(),
        )

    @contextmanager
    def _client(self, *, queue=None, scheduler=None):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main_module, "get_settings", return_value=self.settings)
            )
            stack.enter_context(
                patch.object(main_module, "get_memory_service", return_value=self.memory)
            )
            stack.enter_context(patch.object(main_module, "close_memory_service", new=AsyncMock()))
            stack.enter_context(
                patch.object(main_module, "get_message_cache_service", return_value=self.cache)
            )
            stack.enter_context(
                patch(
                    "src.services.scheduler_service.peek_scheduler_service",
                    return_value=scheduler,
                )
            )
            stack.enter_context(patch("src.services.scheduler_service.close_scheduler_service"))
            client = stack.enter_context(TestClient(main_module.app))
            if queue is not None:
                main_module.container.task_queue = queue
            yield client

    def _post_webhook(self, client, events: list[dict]):
        body = json.dumps({"events": events}).encode("utf-8")
        return client.post(
            "/webhook",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Line-Signature": _sign_body(self.settings.line_channel_secret, body),
            },
        )


class InternalTasksEndpointTests(_AppTestCase):
    def _post(self, client, kind: str, body, token: str | bytes | None = _TASKS_SECRET):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["X-Tasks-Token"] = token
        content = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        return client.post(f"/internal/tasks/{kind}", content=content, headers=headers)

    def test_rejects_missing_wrong_and_cron_tokens(self) -> None:
        with self._client() as client:
            # Raw UTF-8 bytes arrive as non-ASCII text; must be 403, not 500.
            for token in (None, "wrong", "cron-secret", "", "簽章".encode()):
                with self.subTest(token=token):
                    response = self._post(client, "passive_flush", {"key": "group::G1"}, token)
                    self.assertEqual(response.status_code, 403)
        self.memory.flush_passive.assert_not_called()

    def test_rejects_everything_when_the_secret_is_empty(self) -> None:
        self.settings = self.settings.model_copy(update={"internal_tasks_secret": ""})
        with self._client() as client:
            for token in (None, ""):
                with self.subTest(token=token):
                    response = self._post(client, "passive_flush", {"key": "group::G1"}, token)
                    self.assertEqual(response.status_code, 403)
        self.memory.flush_passive.assert_not_called()

    def test_bad_json_is_a_400(self) -> None:
        with self._client() as client:
            self.assertEqual(self._post(client, "event", b"{not json").status_code, 400)
            self.assertEqual(self._post(client, "event", ["a", "list"]).status_code, 400)

    def test_runs_the_task_and_reports_ok_kind_and_ms(self) -> None:
        with self._client() as client:
            response = self._post(client, "passive_flush", {"key": "group::G1"})

            # An authenticated task request also teaches the app its URL.
            self.assertEqual(main_module.container.base_url(), "https://testserver")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["kind"], "passive_flush")
        self.assertIsInstance(payload["ms"], int)
        self.memory.flush_passive.assert_awaited_once_with("group::G1")

    def test_failures_still_answer_200(self) -> None:
        self.memory.flush_passive.side_effect = RuntimeError("firestore down")
        with self._client() as client:
            failed = self._post(client, "passive_flush", {"key": "group::G1"})
            unknown = self._post(client, "no_such_kind", {})

        self.assertEqual(failed.status_code, 200)
        self.assertEqual(failed.json()["ok"], False)
        self.assertEqual(unknown.status_code, 200)
        self.assertEqual(unknown.json(), {"ok": False, "kind": "no_such_kind", "ms": unknown.json()["ms"]})


    def test_deps_failure_is_a_failed_task_not_a_crash(self) -> None:
        with self._client() as client, patch.object(
            main_module, "_build_pipeline_deps", side_effect=RuntimeError("line down")
        ), patch.object(main_module.logger, "error") as log_error:
            response = self._post(client, "passive_flush", {"key": "group::G1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ok"], False)
        self.assertIn("could not start", log_error.call_args.args[0])


class WebhookTaskTests(_AppTestCase):
    def test_inline_deps_failure_is_not_run_a_second_time(self) -> None:
        # No K_SERVICE: the lifespan picks the InlineTaskQueue.
        with self._client() as client, patch.object(
            main_module, "_build_pipeline_deps", side_effect=RuntimeError("line down")
        ) as build_deps, patch.object(task_queue_module.logger, "error") as queue_error:
            self.assertIsInstance(main_module.container.task_queue, InlineTaskQueue)
            response = self._post_webhook(client, [_group_event("01D")])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(build_deps.call_count, 1)
        for call in queue_error.call_args_list:
            self.assertNotIn("could not be queued", call.args[0])

    def test_triggered_event_is_queued_not_processed_in_the_request(self) -> None:
        queue = _RecordingQueue()
        event = _group_event("01T")
        with self._client(queue=queue) as client, patch.object(
            main_module, "_dispatch_task", AsyncMock()
        ) as dispatch:
            response = self._post_webhook(client, [event])

            self.assertEqual(main_module.container.base_url(), "https://testserver")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(queue.calls, [("event", {"event": event})])
        dispatch.assert_not_called()
        self.memory.enqueue_passive_message.assert_not_called()
        self.cache.cache_event_text.assert_not_called()

    def test_untriggered_group_text_is_buffered_and_cached_inline(self) -> None:
        queue = _RecordingQueue()
        event = _group_event("01P", text="只是路過", timestamp=1_790_000_000_000)
        with self._client(queue=queue) as client:
            response = self._post_webhook(client, [event])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(queue.calls, [])
        # Stamped with when LINE received it, which orders the window.
        self.memory.enqueue_passive_message.assert_called_once_with(
            source_type="group", chat_scope="multi", chat_id="G1", user_id="U1", text="只是路過",
            created_at=1_790_000_000.0,
        )
        self.cache.cache_event_text.assert_called_once_with(event)
        self.cache.cache_event_message.assert_not_called()

    def test_passive_threshold_queues_a_flush_for_that_chat(self) -> None:
        queue = _RecordingQueue()
        self.memory.enqueue_passive_message.return_value = True
        with self._client(queue=queue) as client:
            self._post_webhook(client, [_group_event("01F", text="第八則")])

        self.assertEqual(queue.calls, [("passive_flush", {"key": "group::G1"})])

    def test_untriggered_image_is_queued_for_caching(self) -> None:
        queue = _RecordingQueue()
        event = _group_event("01I")
        event["message"] = {"type": "image", "id": "IMG1"}
        with self._client(queue=queue) as client:
            self._post_webhook(client, [event])

        self.assertEqual(queue.calls, [("cache_image", {"event": event})])
        self.cache.cache_event_text.assert_not_called()
        self.memory.enqueue_passive_message.assert_called_once()

    def test_line_redelivery_is_processed_once(self) -> None:
        queue = _RecordingQueue()
        event = _group_event("01R")
        redelivered = dict(event, deliveryContext={"isRedelivery": True})
        with self._client(queue=queue) as client:
            self._post_webhook(client, [event])
            self._post_webhook(client, [redelivered])

        self.assertEqual(queue.calls, [("event", {"event": event})])

    def test_enqueue_failure_falls_back_to_inline(self) -> None:
        for label, queue in {
            "returned False": _RecordingQueue(result=False),
            "raised": _RecordingQueue(error=httpx.ConnectError("down")),
        }.items():
            with self.subTest(label):
                _reset_event_dedupe(self)
                event = _group_event("01X")
                with self._client(queue=queue) as client, patch.object(
                    main_module, "_dispatch_task", AsyncMock(return_value=True)
                ) as dispatch, patch.object(task_queue_module.logger, "error") as log_error:
                    response = self._post_webhook(client, [event])

                self.assertEqual(response.status_code, 200)
                dispatch.assert_awaited_once_with("event", {"event": event})
                self.assertIn("running it inline", log_error.call_args.args[0])

    def test_only_an_event_runs_inline_when_it_may_already_be_queued(self) -> None:
        # The event handler drops a second delivery (_task_seen); the other
        # kinds would run twice, so they are left to Cloud Tasks.
        image = _group_event("01MI")
        image["message"] = {"type": "image", "id": "IMG-M"}
        self.memory.enqueue_passive_message.return_value = True
        triggered = _group_event("01ME")
        queue = _RecordingQueue(error=TaskMaybeQueuedError("ReadTimeout"))
        with self._client(queue=queue) as client, patch.object(
            main_module, "_dispatch_task", AsyncMock(return_value=True)
        ) as dispatch:
            response = self._post_webhook(client, [image, triggered])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            sorted(kind for kind, _ in queue.calls), ["cache_image", "event", "passive_flush"]
        )
        dispatch.assert_awaited_once_with("event", {"event": triggered})

    def test_events_are_enqueued_concurrently(self) -> None:
        class _OverlapQueue(_RecordingQueue):
            """Succeeds only if a second enqueue starts while the first waits."""

            async def enqueue(self, kind, payload):
                self.calls.append((kind, payload))
                for _ in range(50):
                    if len(self.calls) >= 2:
                        return True
                    await asyncio.sleep(0)
                return False

        queue = _OverlapQueue()
        with self._client(queue=queue) as client, patch.object(
            main_module, "_dispatch_task", AsyncMock()
        ) as dispatch:
            self._post_webhook(client, [_group_event("01C1"), _group_event("01C2")])

        self.assertEqual([kind for kind, _ in queue.calls], ["event", "event"])
        dispatch.assert_not_called()

    def test_inline_mode_runs_the_whole_pipeline_inside_the_request(self) -> None:
        # No K_SERVICE: lifespan picks the InlineTaskQueue.
        request = AgentRequest(
            user_id="U1", group_id="G1", reply_token="reply-token", text="你好",
            source_type="group", chat_scope="multi",
        )
        response = AgentResponse(text="哈囉", agent_name="chat", output_format="text")
        send_response = AsyncMock(return_value=True)
        memory = SimpleNamespace(
            record_interaction=AsyncMock(),
            touch_user_profile=AsyncMock(),
            load_profile_facts=AsyncMock(return_value=[]),
            flush_passive=AsyncMock(return_value=False),
        )
        deps = PipelineDeps(
            line=SimpleNamespace(
                send_loading_animation=AsyncMock(), fetch_display_name=AsyncMock(return_value="")
            ),
            memory=memory,
            message_cache=SimpleNamespace(cache_processed_request=Mock()),
            orchestrator=SimpleNamespace(
                route=AsyncMock(
                    return_value=SimpleNamespace(
                        agent="chat", output_format="text", task_description="",
                        reasoning="", disable_thinking=True,
                    )
                )
            ),
            get_agent=lambda name: SimpleNamespace(process=AsyncMock(return_value=response)),
            process_input=AsyncMock(return_value=request),
            enrich_request=AsyncMock(return_value=request),
            extract_text=Mock(return_value="你好"),
            send_response=send_response,
        )
        with self._client() as client, patch.object(
            main_module, "_build_pipeline_deps", return_value=deps
        ):
            self.assertIsInstance(main_module.container.task_queue, InlineTaskQueue)
            http_response = self._post_webhook(client, [_group_event("01E")])

        self.assertEqual(http_response.status_code, 200)
        send_response.assert_awaited_once_with(request, response)
        memory.flush_passive.assert_awaited_once_with("group::G1", compact=False)
        memory.record_interaction.assert_awaited_once()


class CronMaintenanceTests(_AppTestCase):
    def test_passive_sweep_runs_without_a_scheduler(self) -> None:
        self.memory.flush_due_passive.return_value = ["group::G1", "room::R1"]
        with self._client(scheduler=None) as client:
            response = client.post("/internal/cron", headers={"X-Cron-Token": "cron-secret"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reason"], "scheduler_not_initialized")
        self.assertEqual(payload["passive_flushed"], 2)
        self.assertEqual(payload["maintenance_errors"], [])
        self.memory.flush_due_passive.assert_awaited_once()
        self.assertEqual(set(self.memory.flush_due_passive.await_args.kwargs), {"deadline"})

    def test_sweep_result_is_merged_with_scheduled_dispatch(self) -> None:
        scheduler = SimpleNamespace(
            dispatch_due_jobs=AsyncMock(
                return_value={"fired": ["weekly1"], "failed": [], "skipped_duplicate": []}
            )
        )
        with self._client(scheduler=scheduler) as client:
            response = client.post("/internal/cron", headers={"X-Cron-Token": "cron-secret"})

        payload = response.json()
        self.assertEqual(payload["fired"], ["weekly1"])
        self.assertEqual(payload["passive_flushed"], 0)
        self.memory.flush_due_passive.assert_awaited_once()

    def test_unauthenticated_cron_runs_no_maintenance(self) -> None:
        with self._client() as client:
            response = client.post("/internal/cron", headers={"X-Cron-Token": "wrong"})

        self.assertEqual(response.status_code, 403)
        self.memory.flush_due_passive.assert_not_called()

    def test_the_first_cron_of_a_day_runs_the_eol_check_as_a_task(self) -> None:
        with self._client() as client:
            lifecycle = main_module.container.model_lifecycle
            run = AsyncMock(return_value={})
            with patch.object(lifecycle, "run", new=run):
                first = client.post("/internal/cron", headers={"X-Cron-Token": "cron-secret"})
                second = client.post("/internal/cron", headers={"X-Cron-Token": "cron-secret"})

        self.assertTrue(first.json()["eol_check_queued"])
        self.assertFalse(second.json()["eol_check_queued"])
        # Inline queue here: the `eol` task handler ran the check once.
        run.assert_awaited_once_with()


class ColabWiringTests(_AppTestCase):
    def test_the_cron_ticks_the_colab_controller(self) -> None:
        controller = SimpleNamespace(tick=AsyncMock(return_value={"status": "IDLE", "queued": []}))
        with self._client() as client:
            main_module.container.colab_controller = controller
            response = client.post("/internal/cron", headers={"X-Cron-Token": "cron-secret"})

        self.assertEqual(response.json()["colab"], {"status": "IDLE", "queued": []})
        controller.tick.assert_awaited_once_with()

    def test_colab_tasks_reach_the_controller(self) -> None:
        controller = SimpleNamespace(run_action=AsyncMock(return_value=True))
        with self._client() as client:
            main_module.container.colab_controller = controller
            response = client.post(
                "/internal/tasks/colab",
                json={"action": "heartbeat", "nbh": "n", "endpoint": "e"},
                headers={"X-Tasks-Token": _TASKS_SECRET},
            )

        self.assertEqual(response.json()["ok"], True)
        controller.run_action.assert_awaited_once_with("heartbeat", {"action": "heartbeat", "nbh": "n", "endpoint": "e"})

    def test_startup_builds_the_controller_on_the_endpoint_and_the_task_queue(self) -> None:
        built = []

        def build(settings, colab, *, submit, bot_url):
            built.append((colab, submit, bot_url))
            return "controller"

        with patch.object(main_module, "build_colab_controller", side_effect=build), self._client():
            container = main_module.container
            self.assertEqual(container.colab_controller, "controller")
        [(colab, submit, bot_url)] = built
        self.assertIs(colab, container.colab)
        # Queue only: never _submit_task's inline fallback.
        self.assertIs(submit, main_module._queue_colab_task)
        self.assertIs(bot_url, container.base_url)

    def test_colab_off_has_no_controller(self) -> None:
        with self._client():
            self.assertIsNone(main_module.container.colab_controller)


class RunCronTickTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_colab_controller_ticks_every_minute(self) -> None:
        memory = SimpleNamespace(flush_due_passive=AsyncMock(return_value=[]))
        colab = SimpleNamespace(tick=AsyncMock(return_value={"status": "READY", "queued": ["reconcile"]}))

        summary = await run_cron_tick(CronDeps(memory=memory, colab=colab))

        colab.tick.assert_awaited_once_with()
        self.assertEqual(summary["colab"], {"status": "READY", "queued": ["reconcile"]})
        self.assertEqual(summary["maintenance_errors"], [])

    async def test_a_failing_colab_tick_is_reported_and_the_sweep_still_runs(self) -> None:
        memory = SimpleNamespace(flush_due_passive=AsyncMock(return_value=["group::G1"]))
        colab = SimpleNamespace(tick=AsyncMock(side_effect=RuntimeError("boom")))
        with patch("src.pipeline.maintenance.logger.error"):
            summary = await run_cron_tick(CronDeps(memory=memory, colab=colab))
        self.assertEqual(summary["maintenance_errors"], ["colab_tick_failed:RuntimeError"])
        self.assertEqual((summary["colab"], summary["passive_flushed"]), (None, 1))

    async def test_errors_are_reported_not_raised(self) -> None:
        memory = SimpleNamespace(flush_due_passive=AsyncMock(side_effect=RuntimeError("boom")))
        with patch("src.pipeline.maintenance.logger.error"):
            summary = await run_cron_tick(CronDeps(memory=memory), None)

        self.assertEqual(summary["passive_flushed"], 0)
        self.assertEqual(summary["maintenance_errors"], ["passive_flush_failed:RuntimeError"])

    async def test_the_budget_becomes_a_deadline_not_a_cancellation(self) -> None:
        memory = SimpleNamespace(flush_due_passive=AsyncMock(return_value=["group::G1"]))

        summary = await run_cron_tick(
            CronDeps(memory=memory, budget_seconds=30, clock=lambda: 100.0)
        )

        memory.flush_due_passive.assert_awaited_once_with(deadline=130.0)
        self.assertEqual(
            summary,
            {"passive_flushed": 1, "eol_check_queued": False, "colab": None, "maintenance_errors": []},
        )

    async def test_the_eol_check_is_queued_once_per_taipei_day(self) -> None:
        from datetime import UTC, datetime

        from src.services.model_lifecycle import ModelLifecycle

        memory = SimpleNamespace(flush_due_passive=AsyncMock(return_value=[]))
        lifecycle = ModelLifecycle([], ModelHealth())
        submit = AsyncMock()
        deps = CronDeps(memory=memory, lifecycle=lifecycle, submit=submit)
        # 15:59 UTC is 23:59 in Taipei; 16:00 UTC is already the next day there.
        minutes = (
            datetime(2026, 9, 30, 15, 58, tzinfo=UTC),
            datetime(2026, 9, 30, 15, 59, tzinfo=UTC),
            datetime(2026, 9, 30, 16, 0, tzinfo=UTC),
            datetime(2026, 9, 30, 16, 1, tzinfo=UTC),
        )

        queued = [(await run_cron_tick(deps, minute))["eol_check_queued"] for minute in minutes]

        self.assertEqual(queued, [True, False, True, False])
        self.assertEqual(submit.await_args_list, [unittest.mock.call("eol", {})] * 2)

    async def test_a_failed_eol_enqueue_is_reported_not_raised(self) -> None:
        from src.services.model_lifecycle import ModelLifecycle

        memory = SimpleNamespace(flush_due_passive=AsyncMock(return_value=[]))
        deps = CronDeps(
            memory=memory,
            lifecycle=ModelLifecycle([], ModelHealth()),
            submit=AsyncMock(side_effect=RuntimeError("queue down")),
        )

        with patch("src.pipeline.maintenance.logger.error"):
            summary = await run_cron_tick(deps)

        self.assertFalse(summary["eol_check_queued"])
        self.assertEqual(summary["maintenance_errors"], ["eol_enqueue_failed:RuntimeError"])

    async def test_a_slow_sweep_stops_between_chats_and_loses_nothing(self) -> None:
        # Three due chats, each write takes 0.1 s, budget 0.15 s: two are
        # written, the third stays buffered for the next tick.
        class _SlowBackend(InMemoryBackend):
            async def load_chat(self, key):
                await asyncio.sleep(0.1)
                return await super().load_chat(key)

        service = MemoryService(
            Settings(_env_file=None, passive_group_recording_enabled=True),
            backend=_SlowBackend(),
        )
        clock = _FakeClock()
        service._passive._clock = clock
        chats = [f"G{i}" for i in range(3)]
        for chat_id in chats:
            service.enqueue_passive_message(
                source_type="group", chat_scope="multi", chat_id=chat_id, user_id="U1",
                text=f"hello {chat_id}",
            )
        clock.now += 60

        summary = await run_cron_tick(CronDeps(memory=service, budget_seconds=0.15))

        stored = set(service.backend_instance.store)
        buffered = set(service._passive.pending_keys())
        self.assertEqual(summary["maintenance_errors"], [])
        self.assertEqual(summary["passive_flushed"], len(stored))
        self.assertEqual(len(stored), 2)
        self.assertEqual(stored | buffered, {f"group::{chat_id}" for chat_id in chats})
        self.assertFalse(stored & buffered)
        for key in buffered:
            self.assertEqual(service._passive.pending_count(key), 1)


# ── Compaction scheduling ────────────────────────────────────


class CompactionSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def _service(self, scheduler, *, generate=None, **overrides) -> MemoryService:
        self.chain = SimpleNamespace(
            generate=generate
            or AsyncMock(return_value=ProviderResponse(text="摘要", model="m"))
        )
        values = {"memory_recent_message_limit": 2, **overrides}
        return MemoryService(
            Settings(_env_file=None, **values),
            summary_chain=self.chain,
            summary_targets=[Tier(object(), "m")],
            backend=InMemoryBackend(),
            compaction_scheduler=scheduler,
        )

    async def _turn(self, service: MemoryService, chat_id: str = "U1", text: str = "q") -> None:
        await service.record_interaction(
            source_type="user", chat_scope="user", chat_id=chat_id, user_id=chat_id,
            user_text=text, assistant_text="a",
        )

    async def test_crossing_the_limit_queues_one_compact_task_per_chat(self) -> None:
        submitted: list[tuple[str, dict]] = []

        async def submit(kind, payload):
            submitted.append((kind, payload))

        service = self._service(build_compaction_scheduler(submit))
        await self._turn(service, "U1")
        await self._turn(service, "U1")  # still pending: not queued again
        await self._turn(service, "U2")

        self.assertEqual(
            submitted,
            [
                ("compact", {"source_type": "user", "chat_scope": "user", "chat_id": "U1"}),
                ("compact", {"source_type": "user", "chat_scope": "user", "chat_id": "U2"}),
            ],
        )
        # Queued, not run: no summary call happened in this "request".
        self.chain.generate.assert_not_awaited()
        self.assertEqual(service.get_stats()["summary_tasks"], 2)

    async def test_compact_task_runs_the_compaction_and_frees_the_chat(self) -> None:
        submitted: list[tuple[str, dict]] = []

        async def submit(kind, payload):
            submitted.append((kind, payload))

        # No debounce: this is about the pending mark being released.
        service = self._service(build_compaction_scheduler(submit), compaction_debounce_seconds=0)
        await self._turn(service)
        kind, payload = submitted[0]

        ok = await handle_task(kind, payload, TaskDeps(pipeline=_pipeline(memory=service)))

        self.assertTrue(ok)
        memory = await service.load_context(source_type="user", chat_scope="user", chat_id="U1")
        self.assertEqual(memory.summary_text, "摘要")
        self.assertEqual(memory.recent_count, 0)
        self.assertEqual(service.get_stats()["summary_tasks"], 0)
        await self._turn(service)
        self.assertEqual(len(submitted), 2)

    async def test_window_still_over_the_limit_schedules_another_pass(self) -> None:
        submitted: list[tuple[str, dict]] = []

        async def submit(kind, payload):
            submitted.append((kind, payload))

        async def generate(**kwargs):
            if self.chain.generate.await_count == 1:
                # A new turn lands while the summary is being generated.
                await self._turn(service, text="mid-flight")
            return ProviderResponse(text="摘要", model="m")

        # No debounce (tests/test_memory_compaction.py covers it with one).
        service = self._service(
            build_compaction_scheduler(submit),
            generate=AsyncMock(side_effect=generate),
            compaction_debounce_seconds=0,
        )
        await self._turn(service)
        self.assertEqual(len(submitted), 1)

        again = await service.run_compaction(source_type="user", chat_scope="user", chat_id="U1")

        self.assertTrue(again)
        # The mid-flight turn was deduped while the first pass was pending,
        # then picked up by the pass's own "still due" check.
        self.assertEqual(len(submitted), 2)
        memory = await service.load_context(source_type="user", chat_scope="user", chat_id="U1")
        self.assertEqual([m.content for m in memory.recent_messages], ["mid-flight", "a"])

    async def test_scheduler_failure_releases_the_chat(self) -> None:
        scheduler = AsyncMock(side_effect=RuntimeError("queue down"))
        service = self._service(scheduler)

        with patch("src.services.memory_service.logger.error") as log_error:
            await self._turn(service)
            await self._turn(service)

        self.assertEqual(scheduler.await_count, 2)
        self.assertIn("could not be scheduled", log_error.call_args.args[0])
        self.assertEqual(service.get_stats()["summary_tasks"], 0)

    async def test_cancelled_scheduling_releases_the_chat(self) -> None:
        scheduler = AsyncMock(side_effect=asyncio.CancelledError())
        service = self._service(scheduler)

        with self.assertRaises(asyncio.CancelledError):
            await self._turn(service)

        self.assertEqual(service.get_stats()["summary_tasks"], 0)

    async def test_passive_flush_schedules_compaction_unless_told_not_to(self) -> None:
        scheduler = AsyncMock()
        service = self._service(
            scheduler,
            passive_group_recording_enabled=True,
            passive_group_flush_threshold=10,
        )

        def enqueue(text: str) -> None:
            service.enqueue_passive_message(
                source_type="group", chat_scope="multi", chat_id="G1", user_id="U1", text=text
            )

        for i in range(3):
            enqueue(f"m{i}")
        self.assertTrue(await service.flush_passive("group::G1", compact=False))
        scheduler.assert_not_called()

        enqueue("m3")
        self.assertTrue(await service.flush_passive("group::G1"))
        scheduler.assert_awaited_once_with(source_type="group", chat_scope="multi", chat_id="G1")
        self.assertFalse(await service.flush_passive("group::G1"))  # nothing pending

    async def test_close_flushes_passive_buffers_without_starting_compaction(self) -> None:
        scheduler = AsyncMock()
        service = self._service(
            scheduler,
            passive_group_recording_enabled=True,
            passive_group_flush_threshold=10,
        )
        for i in range(3):
            service.enqueue_passive_message(
                source_type="group", chat_scope="multi", chat_id="G1", user_id="U1", text=f"m{i}"
            )

        await service.close()

        stored = service.backend_instance.store["group::G1"]
        self.assertEqual(len(stored["recent_messages"]), 3)
        scheduler.assert_not_called()


# ── A failed read never overwrites stored memory ─────────────


class LoadFailureTests(unittest.IsolatedAsyncioTestCase):
    """Writers must not save a default document over one they could not read."""

    def _service(self, **overrides) -> MemoryService:
        values = {
            "memory_recent_message_limit": 2,
            "passive_group_recording_enabled": True,
            "passive_group_flush_threshold": 10,
            **overrides,
        }
        self.chain = SimpleNamespace(
            generate=AsyncMock(return_value=ProviderResponse(text="新摘要", model="m"))
        )
        service = MemoryService(
            Settings(_env_file=None, **values),
            summary_chain=self.chain,
            summary_targets=[Tier(object(), "m")],
            backend=InMemoryBackend(),
            compaction_scheduler=AsyncMock(),
        )
        self.backend = service.backend_instance
        self.backend.store["group::G1"] = {
            "chat_scope": "multi",
            "source_type": "group",
            "chat_id": "G1",
            "summary_text": "舊摘要",
            "recent_messages": [
                {"role": "user", "content": f"old{i}", "user_id": "U1", "created_at": 1.0}
                for i in range(3)
            ],
            "recent_count": 3,
            "summary_version": 4,
        }
        self.backend.user_profiles["U1"] = {"user_id": "U1", "facts": ["likes tea"]}
        self.stored = json.loads(json.dumps(self.backend.store))
        self.profiles = json.loads(json.dumps(self.backend.user_profiles))
        return service

    def _failing_reads(self):
        failure = MemoryBackendError("firestore unavailable")
        stack = ExitStack()
        stack.enter_context(
            patch.object(self.backend, "load_chat", AsyncMock(side_effect=failure))
        )
        stack.enter_context(
            patch.object(self.backend, "load_user_profile", AsyncMock(side_effect=failure))
        )
        return stack

    async def test_record_interaction_raises_instead_of_overwriting(self) -> None:
        service = self._service()
        with self._failing_reads(), self.assertRaises(MemoryServiceError):
            await service.record_interaction(
                source_type="group", chat_scope="multi", chat_id="G1", user_id="U1",
                user_text="q", assistant_text="a",
            )
        self.assertEqual(self.backend.store, self.stored)

    async def test_passive_flush_keeps_the_batch_until_a_read_works(self) -> None:
        service = self._service()
        for text in ("p0", "p1"):
            service.enqueue_passive_message(
                source_type="group", chat_scope="multi", chat_id="G1", user_id="U2", text=text
            )

        with self._failing_reads():
            self.assertFalse(await service.flush_passive("group::G1"))
        self.assertEqual(self.backend.store, self.stored)
        self.assertEqual(service._passive.pending_count("group::G1"), 2)

        self.assertTrue(await service.flush_passive("group::G1"))
        stored = self.backend.store["group::G1"]
        self.assertEqual(stored["summary_text"], "舊摘要")
        self.assertEqual(
            [m["content"] for m in stored["recent_messages"]], ["old0", "old1", "old2", "p0", "p1"]
        )

    async def test_compaction_skips_the_pass_instead_of_overwriting(self) -> None:
        service = self._service()
        with self._failing_reads(), patch("src.services.memory_service.logger.error"):
            again = await service.run_compaction(
                source_type="group", chat_scope="multi", chat_id="G1"
            )
        self.assertFalse(again)
        self.chain.generate.assert_not_awaited()
        self.assertEqual(self.backend.store, self.stored)

    async def test_read_only_context_still_degrades_to_the_default(self) -> None:
        service = self._service()
        with self._failing_reads():
            memory = await service.load_context(
                source_type="group", chat_scope="multi", chat_id="G1"
            )
        self.assertEqual((memory.summary_text, memory.recent_messages), ("", []))

    async def test_profile_writers_skip_the_save(self) -> None:
        service = self._service()
        with self._failing_reads():
            await service.touch_user_profile(user_id="U1", display_name="Alice")
            result = await service.update_user_facts(user_id="U1", facts=["likes coffee"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.backend.user_profiles, self.profiles)


# ── Group context before a reply ─────────────────────────────


class PassiveFlushBeforeReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_group_messages_precede_the_triggered_turn(self) -> None:
        service = MemoryService(
            Settings(
                _env_file=None,
                passive_group_recording_enabled=True,
                passive_group_flush_threshold=8,
            ),
            backend=InMemoryBackend(),
        )
        for text in ("p1", "p2"):
            self.assertIsNone(record_group_message(_group_event(text=text), service))

        request = AgentRequest(
            user_id="U1", group_id="G1", reply_token="r", text="問題",
            source_type="group", chat_scope="multi",
        )
        context_seen: list[str] = []

        async def enrich(req):
            memory = await service.load_context(
                source_type="group", chat_scope="multi", chat_id="G1"
            )
            context_seen.extend(m.content for m in memory.recent_messages)
            return req

        deps = _pipeline(
            line=SimpleNamespace(
                send_loading_animation=AsyncMock(), fetch_display_name=AsyncMock(return_value="")
            ),
            memory=service,
            message_cache=SimpleNamespace(cache_processed_request=Mock()),
            orchestrator=SimpleNamespace(
                route=AsyncMock(
                    return_value=SimpleNamespace(
                        agent="chat", output_format="text", task_description="",
                        reasoning="", disable_thinking=True,
                    )
                )
            ),
            get_agent=lambda name: SimpleNamespace(
                process=AsyncMock(
                    return_value=AgentResponse(text="答案", agent_name="chat", output_format="text")
                )
            ),
            process_input=AsyncMock(return_value=request),
            enrich_request=enrich,
            extract_text=Mock(return_value="問題"),
            send_response=AsyncMock(return_value=True),
        )

        await process_event(_group_event("01Q", text="!hej 問題"), deps)

        # The reply saw them, and they are stored before the turn.
        self.assertEqual(context_seen, ["p1", "p2"])
        stored = service.backend_instance.store["group::G1"]["recent_messages"]
        self.assertEqual([m["content"] for m in stored], ["p1", "p2", "問題", "答案"])
        self.assertEqual(service._passive.pending_keys(), [])

    async def test_only_group_and_room_chats_flush_and_failures_are_logged(self) -> None:
        flush = AsyncMock()
        deps = _pipeline(memory=SimpleNamespace(flush_passive=flush))
        dm = AgentRequest(user_id="U1", group_id="U1", source_type="user", chat_scope="user")
        room = AgentRequest(user_id="U1", group_id="R1", source_type="room", chat_scope="multi")

        await event_pipeline_module._flush_passive_before_reply(deps, dm)
        flush.assert_not_called()
        await event_pipeline_module._flush_passive_before_reply(deps, room)
        flush.assert_awaited_once_with("room::R1", compact=False)

        flush.side_effect = RuntimeError("boom")
        with patch.object(event_pipeline_module.logger, "warning") as log_warning:
            await event_pipeline_module._flush_passive_before_reply(deps, room)
        self.assertIn("Passive flush before the reply failed", log_warning.call_args.args[0])


# ── Passive recording and the reply window ───────────────────


class RecordGroupMessageTests(unittest.TestCase):
    def test_returns_the_chat_key_when_the_buffer_reaches_its_threshold(self) -> None:
        memory = SimpleNamespace(enqueue_passive_message=Mock(return_value=True))
        self.assertEqual(record_group_message(_group_event(text="hi"), memory), "group::G1")

        memory.enqueue_passive_message.return_value = False
        self.assertIsNone(record_group_message(_group_event(text="hi"), memory))

        dm = _group_event(text="hi", source={"type": "user", "userId": "U1"})
        self.assertIsNone(record_group_message(dm, memory))

        room = _group_event(text="hi", source={"type": "room", "roomId": "R1", "userId": "U1"})
        memory.enqueue_passive_message.return_value = True
        self.assertEqual(record_group_message(room, memory), "room::R1")


class MessageCacheTextTests(unittest.TestCase):
    def test_cache_event_text_is_synchronous_and_text_only(self) -> None:
        cache = MessageCacheService()

        self.assertTrue(cache.cache_event_text(_group_event(text="  記住這句  ")))
        cached = cache.get("M-01EVT")
        self.assertEqual((cached.message_type, cached.text), ("text", "記住這句"))

        image = _group_event("01IMG")
        image["message"] = {"type": "image", "id": "IMG1"}
        self.assertFalse(cache.cache_event_text(image))
        self.assertIsNone(cache.get("IMG1"))


class ReplyWindowTests(unittest.IsolatedAsyncioTestCase):
    def test_deadline_is_received_at_plus_the_window(self) -> None:
        request = AgentRequest(received_at=1_000.0, reply_window_seconds=50)
        self.assertEqual(request.deadline, 1_050.0)
        self.assertEqual(AgentRequest(received_at=10.0).deadline, 100.0)

    def test_event_received_at_reads_the_line_timestamp(self) -> None:
        self.assertEqual(event_received_at({"timestamp": 1_700_000_000_123}), 1_700_000_000.123)
        for event in ({}, {"timestamp": None}, {"timestamp": True}, {"timestamp": -5}, {"timestamp": "1"}):
            with self.subTest(event=event):
                self.assertEqual(event_received_at(event, now=42.0), 42.0)

    async def test_process_input_sets_received_at(self) -> None:
        event = _group_event(text="你好", timestamp=1_700_000_000_000)
        request = await process_input(event)
        self.assertEqual(request.received_at, 1_700_000_000.0)

    async def _run(self, *, age: float, window: float = 50, turn: float = 240):
        request = AgentRequest(group_id="G1", reply_token="r", text="!new", received_at=1_000.0)
        deps = _pipeline(
            line=SimpleNamespace(send_text=AsyncMock()),
            memory=SimpleNamespace(clear_chat=AsyncMock()),
            process_input=AsyncMock(return_value=request),
            extract_text=Mock(return_value="!new"),
            reply_window_seconds=window,
            turn_deadline_seconds=turn,
            clock=lambda: 1_000.0 + age,
        )
        event = _group_event(text="!new", timestamp=1_000_000)
        with patch.object(event_pipeline_module.logger, "warning") as log_warning:
            await process_event(event, deps)
        return request, log_warning

    async def test_late_start_logs_that_the_reply_token_likely_expired(self) -> None:
        _, log_warning = await self._run(age=75)
        log_warning.assert_called_once()
        self.assertIn("reply token has likely expired", log_warning.call_args.args[0])

    async def test_timely_start_logs_nothing_and_applies_the_configured_window(self) -> None:
        request, log_warning = await self._run(age=10, window=42, turn=200)
        log_warning.assert_not_called()
        self.assertEqual(request.reply_window_seconds, 42)
        self.assertEqual(request.deadline, 1_042.0)
        self.assertEqual(request.turn_deadline_seconds, 200)
        # The turn counts from when work started, whatever the event's age.
        self.assertAlmostEqual(request.turn_deadline() - request.started_at, 200)


# ── Deploy script ──────────────────────────────────────────


_DEPLOY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_cloud_run.sh"


class DeployScriptTasksTests(unittest.TestCase):
    def _script_commands(self) -> list[str]:
        text = _DEPLOY_SCRIPT.read_text(encoding="utf-8").replace("\\\n", " ")
        return [line.strip() for line in text.splitlines() if not line.strip().startswith("#")]

    def test_queue_is_single_attempt_without_a_retry_duration(self) -> None:
        commands = self._script_commands()
        joined = "\n".join(commands)
        self.assertIn("cloudtasks.googleapis.com", joined)
        self.assertIn("--max-attempts=1", joined)
        self.assertIn("--max-concurrent-dispatches=4", joined)
        self.assertIn("--max-dispatches-per-second=5", joined)
        self.assertIn("gcloud tasks queues create", joined)
        self.assertIn("gcloud tasks queues update", joined)
        # maxAttempts=1 only holds while maxRetryDuration is unset.
        self.assertFalse(
            [line for line in commands if re.search(r"--max-retry-duration", line)],
            "the deploy must never set --max-retry-duration",
        )

    def _run_deploy(self, env_lines: list[str], gcloud_cases: str = "") -> tuple[int, str, str]:
        """Run the deploy against a fake gcloud; ``gcloud_cases`` adds sh case arms."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calls = tmp_path / "gcloud-calls.log"
            fake_gcloud = tmp_path / "bin" / "gcloud"
            fake_gcloud.parent.mkdir()
            fake_gcloud.write_text(
                "#!/bin/sh\n"
                f'echo "$*" >> "{calls}"\n'
                'case "$*" in\n'
                '  "auth list"*) echo "deployer@example.com" ;;\n'
                f"{gcloud_cases}"
                "esac\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_gcloud.chmod(fake_gcloud.stat().st_mode | stat.S_IXUSR)
            # The deploy requires curl before its first check, and
            # python:3.12-slim (Cloud Build's test image) has none.
            fake_curl = tmp_path / "bin" / "curl"
            fake_curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_curl.chmod(fake_curl.stat().st_mode | stat.S_IXUSR)
            env_file = tmp_path / "deploy.env"
            env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
            env = dict(os.environ)
            env["PATH"] = f"{fake_gcloud.parent}{os.pathsep}{env.get('PATH', '')}"
            env["TMPDIR"] = tmp
            result = subprocess.run(
                ["bash", str(_DEPLOY_SCRIPT), "--env-file", str(env_file)],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            gcloud_calls = calls.read_text(encoding="utf-8") if calls.exists() else ""
        return result.returncode, result.stderr, gcloud_calls

    _BASE_ENV = (
        "GCP_PROJECT_ID=test-project",
        "LINE_CHANNEL_SECRET=line-secret",
        "LINE_CHANNEL_ACCESS_TOKEN=line-token",
        "OPENROUTER_API_KEY=sk-or-test",
    )

    def test_missing_tasks_secret_fails_before_anything_is_built(self) -> None:
        # TASKS_MODE defaults to auto, which needs the secret on Cloud Run.
        code, stderr, gcloud_calls = self._run_deploy(list(self._BASE_ENV))

        self.assertEqual(code, 1, stderr)
        self.assertIn("INTERNAL_TASKS_SECRET", stderr)
        self.assertIn("openssl rand -hex 32", stderr)
        self.assertNotIn("builds submit", gcloud_calls)
        self.assertNotIn("tasks queues", gcloud_calls)

    def test_inline_mode_needs_no_tasks_secret(self) -> None:
        code, stderr, _ = self._run_deploy([*self._BASE_ENV, "TASKS_MODE=inline"])

        # Past the guard: the fake gcloud lists no upload files, the next stop.
        self.assertEqual(code, 1)
        self.assertNotIn("INTERNAL_TASKS_SECRET", stderr)
        self.assertIn("no files to upload", stderr)

    def test_invalid_tasks_mode_is_rejected(self) -> None:
        code, stderr, _ = self._run_deploy([*self._BASE_ENV, "TASKS_MODE=celery"])

        self.assertEqual(code, 1)
        self.assertIn("TASKS_MODE must be one of", stderr)

    # Lets the fake gcloud pass the upload leak guard and find linebot-builder
    # and its staging bucket (without them nothing is built), so the deploy
    # runs on.
    _UPLOAD_OK = (
        '  "meta list-files-for-upload"*) echo "main.py" ;;\n'
        '  "iam service-accounts describe linebot-builder@"*) echo "linebot-builder" ;;\n'
        '  "storage buckets list"*) echo "test-project-cb-src-usw1" ;;\n'
    )

    def test_handler_budget_must_stay_below_the_dispatch_deadline(self) -> None:
        # The default deadline is 300 s; a budget equal to it is refused.
        code, stderr, gcloud_calls = self._run_deploy(
            [*self._BASE_ENV, "TASKS_MODE=inline", "TASK_HANDLER_BUDGET_SECONDS=300"]
        )
        self.assertEqual(code, 1)
        self.assertIn("must be below TASK_DISPATCH_DEADLINE_SECONDS (300)", stderr)
        self.assertNotIn("builds submit", gcloud_calls)

    def test_dispatch_deadline_must_be_within_the_cloud_tasks_limits(self) -> None:
        for value in ("10", "abc"):
            with self.subTest(value=value):
                code, stderr, _ = self._run_deploy(
                    [*self._BASE_ENV, "TASKS_MODE=inline", f"TASK_DISPATCH_DEADLINE_SECONDS={value}"]
                )
                self.assertEqual(code, 1)
                self.assertIn("TASK_DISPATCH_DEADLINE_SECONDS", stderr)

    def _deploy_timeout(self, env_lines: list[str], gcloud_cases: str = "") -> tuple[str, str]:
        code, stderr, gcloud_calls = self._run_deploy(
            [*self._BASE_ENV, *env_lines], self._UPLOAD_OK + gcloud_cases
        )
        # The fake describe returns no URL, the first stop after the deploy.
        self.assertEqual(code, 1, stderr)
        self.assertIn("Could not determine deployed Cloud Run URL", stderr)
        deploy = [line for line in gcloud_calls.splitlines() if line.startswith("run deploy")]
        self.assertEqual(len(deploy), 1, gcloud_calls)
        match = re.search(r"--timeout (\S+)", deploy[0])
        self.assertIsNotNone(match, deploy[0])
        return match.group(1), gcloud_calls

    def test_cloud_run_timeout_follows_the_dispatch_deadline(self) -> None:
        # Never below 300 s: the inline fallback runs inside the webhook.
        timeout, _ = self._deploy_timeout(
            [
                "TASKS_MODE=inline",
                "TASK_DISPATCH_DEADLINE_SECONDS=120",
                "TASK_HANDLER_BUDGET_SECONDS=100",
            ]
        )
        self.assertEqual(timeout, "300")

        # A RUNNING queue lets the deploy go on.
        timeout, gcloud_calls = self._deploy_timeout(
            ["INTERNAL_TASKS_SECRET=" + "a1" * 32, "TASK_DISPATCH_DEADLINE_SECONDS=600"],
            '  *"value(state)"*) echo "RUNNING" ;;\n',
        )
        self.assertEqual(timeout, "600")
        self.assertIn("tasks queues update", gcloud_calls)

    def test_a_paused_queue_stops_the_deploy_before_the_build(self) -> None:
        code, stderr, gcloud_calls = self._run_deploy(
            [*self._BASE_ENV, "INTERNAL_TASKS_SECRET=" + "a1" * 32],
            self._UPLOAD_OK + '  *"value(state)"*) echo "PAUSED" ;;\n',
        )

        self.assertEqual(code, 1, stderr)
        self.assertIn("is PAUSED", stderr)
        self.assertIn("gcloud tasks queues resume linebot-events", stderr)
        self.assertIn("tasks queues update", gcloud_calls)
        self.assertNotIn("builds submit", gcloud_calls)
        self.assertNotIn("run deploy", gcloud_calls)


if __name__ == "__main__":
    unittest.main()
