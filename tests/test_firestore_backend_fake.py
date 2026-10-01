"""FirestoreBackend against a fake Firestore client.

The fake keeps documents in a dict keyed by path and mirrors the calls the
backend makes on ``google.cloud.firestore`` (``collection().document()``,
``get``/``set``/``delete``, ``add`` on the episodes subcollection,
``list_documents`` and ``find_nearest``). The fake's signature of every one
of those methods is checked against the real client library, so a drift
there fails here instead of in production. No credentials or network are
involved.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import threading
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as core_retry
from google.api_core.retry import retry_base, retry_unary
from google.cloud.firestore_v1.base_collection import BaseCollectionReference
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.collection import CollectionReference
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1.vector import Vector

from src.config import Settings
from src.pipeline.task_handlers import EVENT_NOTICE_SECONDS
from src.providers.base import ProviderResponse
from src.providers.fallback_chain import Tier
from src.services import memory_backends as memory_backends_module
from src.services.memory_backends import WRITE_TIMEOUT_SECONDS, FirestoreBackend, MemoryBackendError
from src.services.memory_service import MemoryService


class _Snapshot:
    def __init__(self, ref: _DocRef, data: dict | None) -> None:
        self.reference = ref
        self.id = ref.id
        self.exists = data is not None
        self._data = data

    def to_dict(self) -> dict | None:
        return deepcopy(self._data)


class _DocRef:
    def __init__(self, client: FakeFirestore, path: str) -> None:
        self._client = client
        self.path = path
        self.id = path.rsplit("/", 1)[-1]

    def get(self) -> _Snapshot:
        self._client.check("get", self.path)
        return _Snapshot(self, deepcopy(self._client.docs.get(self.path)))

    def set(
        self,
        document_data: dict,
        merge: bool = False,
        retry: object = gapic_v1.method.DEFAULT,
        timeout: float | None = None,
    ) -> None:
        self._client.check("set", self.path)
        self._client.calls.append(("set", self.path, merge))
        self._client.write_options.append(("set", retry, timeout))
        self._client.docs[self.path] = deepcopy(document_data)

    def delete(
        self,
        option: object = None,
        retry: object = gapic_v1.method.DEFAULT,
        timeout: float | None = None,
    ) -> None:
        self._client.check("delete", self.path)
        self._client.calls.append(("delete", self.path))
        self._client.write_options.append(("delete", retry, timeout))
        self._client.docs.pop(self.path, None)

    def collection(self, collection_id: str) -> _Collection:
        return _Collection(self._client, f"{self.path}/{collection_id}")


class _VectorQuery:
    def __init__(self, collection: _Collection, kwargs: dict) -> None:
        self._collection = collection
        self._kwargs = kwargs

    def stream(self):
        self._collection._client.check("find_nearest", self._collection.path)
        query = list(self._kwargs["query_vector"])

        def distance(snapshot: _Snapshot) -> float:
            stored = list(snapshot.to_dict()["embedding"])
            dot = sum(a * b for a, b in zip(query, stored, strict=True))
            norm = (sum(a * a for a in query) * sum(b * b for b in stored)) ** 0.5
            return 1.0 - dot / norm

        with_vectors = [s for s in self._collection.stream() if "embedding" in (s.to_dict() or {})]
        return sorted(with_vectors, key=distance)[: self._kwargs["limit"]]


class _Collection:
    def __init__(self, client: FakeFirestore, path: str) -> None:
        self._client = client
        self.path = path

    def document(self, document_id: str | None = None) -> _DocRef:
        return _DocRef(self._client, f"{self.path}/{document_id or f'auto{next(self._client.ids)}'}")

    def add(self, document_data: dict, document_id: str | None = None):
        ref = self.document(document_id)
        ref.set(document_data)
        return None, ref

    def _children(self) -> list[str]:
        prefix = f"{self.path}/"
        return [p for p in self._client.docs if p.startswith(prefix) and "/" not in p[len(prefix):]]

    def list_documents(self, page_size: int | None = None):
        self._client.calls.append(("list_documents", self.path, page_size))
        return [_DocRef(self._client, path) for path in self._children()]

    def stream(self):
        return [_Snapshot(_DocRef(self._client, p), deepcopy(self._client.docs[p])) for p in self._children()]

    def find_nearest(
        self,
        vector_field,
        query_vector,
        limit,
        distance_measure,
        *,
        distance_result_field=None,
        distance_threshold=None,
    ) -> _VectorQuery:
        kwargs = {
            "vector_field": vector_field,
            "query_vector": query_vector,
            "limit": limit,
            "distance_measure": distance_measure,
            "distance_result_field": distance_result_field,
            "distance_threshold": distance_threshold,
        }
        self._client.calls.append(("find_nearest", self.path, kwargs))
        return _VectorQuery(self, kwargs)


class FakeFirestore:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.calls: list[tuple] = []
        # (operation, retry, timeout) of each set and delete.
        self.write_options: list[tuple] = []
        self.ids = itertools.count(1)
        self.fail: set[str] = set()
        self.closed = False

    def check(self, op: str, path: str) -> None:
        if op in self.fail:
            raise RuntimeError(f"{op} unavailable for {path}")

    def collection(self, name: str) -> _Collection:
        return _Collection(self, name)

    def close(self) -> None:
        self.closed = True


def _backend(prefix: str = "linebot") -> tuple[FirestoreBackend, FakeFirestore]:
    client = FakeFirestore()
    return FirestoreBackend(project_id="p", collection_prefix=prefix, client=client), client


def _parameters(method) -> list[tuple[str, object]]:
    return [(p.name, p.default) for p in inspect.signature(method).parameters.values()]


class FakeMatchesTheRealClientTests(unittest.TestCase):
    def test_find_nearest_takes_the_real_parameters(self) -> None:
        real = list(inspect.signature(BaseCollectionReference.find_nearest).parameters)
        fake = list(inspect.signature(_Collection.find_nearest).parameters)
        self.assertEqual(fake, real)

    def test_the_other_methods_the_backend_calls_start_like_the_real_ones(self) -> None:
        # The fake leaves out the trailing retry / timeout / read_time
        # parameters the backend never passes; the ones it has match the
        # real ones in order, name and default.
        pairs = {
            "DocumentReference.get": (DocumentReference.get, _DocRef.get),
            "DocumentReference.set": (DocumentReference.set, _DocRef.set),
            "DocumentReference.delete": (DocumentReference.delete, _DocRef.delete),
            "DocumentReference.collection": (DocumentReference.collection, _DocRef.collection),
            "CollectionReference.document": (CollectionReference.document, _Collection.document),
            "CollectionReference.add": (CollectionReference.add, _Collection.add),
            "CollectionReference.list_documents": (
                CollectionReference.list_documents,
                _Collection.list_documents,
            ),
            "CollectionReference.stream": (CollectionReference.stream, _Collection.stream),
        }
        for name, (real, fake) in pairs.items():
            with self.subTest(method=name):
                fake_parameters = _parameters(fake)
                real_parameters = _parameters(real)
                self.assertEqual(real_parameters[: len(fake_parameters)], fake_parameters)


class ChatAndProfileDocumentTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_documents_round_trip_at_their_path(self) -> None:
        backend, client = _backend()
        doc = {"summary_text": "摘要", "recent_messages": [{"id": "a1", "role": "user", "content": "嗨"}]}

        self.assertIsNone(await backend.load_chat("user::U1"))
        await backend.save_chat("user::U1", doc)
        doc["summary_text"] = "changed after saving"

        self.assertIn(("set", "linebot_chat_memory/user::U1", False), client.calls)
        loaded = await backend.load_chat("user::U1")
        self.assertEqual(loaded["summary_text"], "摘要")
        loaded["summary_text"] = "tampered"
        self.assertEqual(client.docs["linebot_chat_memory/user::U1"]["summary_text"], "摘要")

    async def test_profiles_live_in_their_own_collection(self) -> None:
        backend, client = _backend(prefix="bot_")
        await backend.save_user_profile("U1", {"facts": ["住台北"]})
        await backend.save_user_profile("", {"facts": ["ignored"]})

        self.assertEqual(list(client.docs), ["bot_user_profile/U1"])
        self.assertEqual((await backend.load_user_profile("U1"))["facts"], ["住台北"])
        self.assertIsNone(await backend.load_user_profile(""))

    async def test_read_and_write_failures_become_backend_errors(self) -> None:
        backend, client = _backend()
        client.fail = {"get", "set"}
        with self.assertRaises(MemoryBackendError):
            await backend.load_chat("user::U1")
        with self.assertRaises(MemoryBackendError):
            await backend.save_user_profile("U1", {})


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_episodes_store_the_vector_as_a_firestore_vector(self) -> None:
        backend, client = _backend()
        episode = {"summary": "摘要", "snippet": "User: 嗨", "ts": 1.0, "embedding": [1, 0]}

        await backend.save_episode("user::U1", episode)

        self.assertEqual(episode["embedding"], [1, 0])  # the caller's dict is untouched
        (path, stored), = client.docs.items()
        self.assertTrue(path.startswith("linebot_episodes/user::U1/items/"))
        self.assertIsInstance(stored["embedding"], Vector)
        self.assertEqual(list(stored["embedding"]), [1.0, 0.0])
        self.assertEqual((stored["summary"], stored["snippet"]), ("摘要", "User: 嗨"))

    async def test_vector_search_call_shape_and_results(self) -> None:
        backend, client = _backend()
        await backend.save_episode("user::U1", {"summary": "近", "embedding": [1.0, 0.0]})
        await backend.save_episode("user::U1", {"summary": "遠", "embedding": [0.0, 1.0]})
        await backend.save_episode("user::U2", {"summary": "別的聊天", "embedding": [1.0, 0.0]})

        results = await backend.search_episodes("user::U1", [0.9, 0.1], k=1)

        (_, path, kwargs), = [c for c in client.calls if c[0] == "find_nearest"]
        self.assertEqual(path, "linebot_episodes/user::U1/items")
        self.assertEqual(kwargs["vector_field"], "embedding")
        self.assertIsInstance(kwargs["query_vector"], Vector)
        self.assertEqual(list(kwargs["query_vector"]), [0.9, 0.1])
        self.assertEqual(kwargs["distance_measure"], DistanceMeasure.COSINE)
        self.assertEqual(kwargs["limit"], 1)
        self.assertEqual(results, [{"summary": "近"}])  # no vector in the result

    async def test_a_failed_episode_save_raises_and_a_failed_search_is_empty(self) -> None:
        # The compaction keeps the conversation pending when the save raises.
        backend, client = _backend()
        client.fail = {"set", "find_nearest"}
        with self.assertRaises(MemoryBackendError):
            await backend.save_episode("user::U1", {"summary": "x"})
        with patch.object(memory_backends_module.logger, "warning") as log_warning:
            self.assertEqual(await backend.search_episodes("user::U1", [1.0], k=3), [])
        self.assertEqual(log_warning.call_count, 1)

    async def test_delete_chat_purges_the_chats_episodes_only(self) -> None:
        backend, client = _backend()
        await backend.save_chat("user::U1", {"summary_text": "x"})
        for chat in ("user::U1", "user::U1", "user::U2"):
            await backend.save_episode(chat, {"summary": "e", "embedding": [1.0]})

        await backend.delete_chat("user::U1")

        self.assertIn(("list_documents", "linebot_episodes/user::U1/items", 200), client.calls)
        self.assertEqual(
            [p.rsplit("/", 1)[0] for p in client.docs], ["linebot_episodes/user::U2/items"]
        )

    async def test_a_failed_purge_does_not_undo_the_chat_delete(self) -> None:
        backend, client = _backend()
        await backend.save_chat("user::U1", {"summary_text": "x"})
        await backend.save_episode("user::U1", {"summary": "e"})

        def failing_list(self, page_size=None):
            raise RuntimeError("listing unavailable")

        with patch.object(_Collection, "list_documents", failing_list), patch.object(
            memory_backends_module.logger, "warning"
        ) as log_warning:
            await backend.delete_chat("user::U1")

        self.assertNotIn("linebot_chat_memory/user::U1", client.docs)
        self.assertIn("episode purge failed", log_warning.call_args.args[0])

    async def test_close_closes_the_client(self) -> None:
        backend, client = _backend()
        await backend.close()
        self.assertTrue(client.closed)


class MemoryServiceOverFirestoreTests(unittest.IsolatedAsyncioTestCase):
    """The service's documents survive the Firestore round trip."""

    async def test_turns_compaction_episode_and_recall(self) -> None:
        class _Embedding:
            model = "fake-embed"

            async def embed_passage(self, text):
                return [1.0, 0.0]

            async def embed_text(self, text):
                return [1.0, 0.0]

        backend, client = _backend()
        service = MemoryService(
            Settings(_env_file=None, memory_recent_message_limit=4, memory_cache_ttl_seconds=1),
            summary_chain=SimpleNamespace(
                generate=AsyncMock(return_value=ProviderResponse(text="使用者在學鋼琴", model="m"))
            ),
            summary_targets=[Tier(object(), "m")],
            backend=backend,
            embedding_service=_Embedding(),
        )

        await service.record_interaction(
            source_type="user", chat_scope="user", chat_id="U1", user_id="U1",
            user_text="我在學鋼琴", assistant_text="加油", user_created_at=100.0,
        )
        stored = client.docs["linebot_chat_memory/user::U1"]["recent_messages"]
        self.assertEqual([m["created_at"] for m in stored][0], 100.0)
        self.assertTrue(all(len(m["id"]) == 12 for m in stored))

        await service.record_interaction(
            source_type="user", chat_scope="user", chat_id="U1", user_id="U1",
            user_text="每週三上課", assistant_text="好",
        )  # the fourth message: compacts inline

        chat = client.docs["linebot_chat_memory/user::U1"]
        self.assertEqual((chat["summary_text"], chat["recent_messages"]), ("使用者在學鋼琴", []))
        recalled = await service.recall_episodes(source_type="user", chat_id="U1", query="鋼琴")
        self.assertEqual(recalled[0]["summary"], "使用者在學鋼琴")
        self.assertIn("User: 我在學鋼琴", recalled[0]["snippet"])

        await service.clear_chat(source_type="user", chat_scope="user", chat_id="U1")
        self.assertEqual(client.docs, {})

    async def test_a_save_cancelled_after_it_landed_does_not_leave_a_stale_cache(self) -> None:
        """Regression: the read-through cache was refreshed only after a
        save returned, so a save that reached Firestore but was cancelled
        (the task handler's timeout) left the document from before it
        cached, and the next write within the TTL erased the landed one."""
        backend, client = _backend()
        service = MemoryService(
            Settings(_env_file=None, memory_recent_message_limit=20, memory_cache_ttl_seconds=60),
            backend=backend,
        )

        async def turn(text: str) -> None:
            await service.record_interaction(
                source_type="user", chat_scope="user", chat_id="U1", user_id="U1",
                user_text=text, assistant_text="好",
            )

        await turn("一")  # cached
        landed, release = threading.Event(), threading.Event()
        real_set = _DocRef.set

        def set_then_hang(ref, document_data, merge=False, **options):
            real_set(ref, document_data, merge, **options)
            landed.set()
            release.wait(5)

        with patch.object(_DocRef, "set", set_then_hang):
            cancelled = asyncio.create_task(turn("二"))
            self.assertTrue(await asyncio.to_thread(landed.wait, 5))
            cancelled.cancel()
            release.set()  # the cancelled save waits for its write to return
            with self.assertRaises(asyncio.CancelledError):
                await cancelled

        await turn("三")

        stored = client.docs["linebot_chat_memory/user::U1"]["recent_messages"]
        self.assertEqual([m["content"] for m in stored], ["一", "好", "二", "好", "三", "好"])

    async def test_a_save_cancelled_before_it_landed_is_not_lost(self) -> None:
        """Regression: cancelling a turn (the task handler's timeout) did not
        stop its write, which runs in an executor thread. The turn released
        the chat's lock at once, the next turn loaded the document without
        the write, and the write then landed over the next turn's: one of
        the two was lost. Now the cancelled turn waits for its write, under
        the lock, and the timeout still propagates."""
        backend, client = _backend()
        service = MemoryService(
            Settings(_env_file=None, memory_recent_message_limit=20, memory_cache_ttl_seconds=60),
            backend=backend,
        )

        async def turn(text: str) -> None:
            await service.record_interaction(
                source_type="user", chat_scope="user", chat_id="U1", user_id="U1",
                user_text=text, assistant_text="好",
            )

        async def turn_with_deadline(text: str) -> None:
            async with asyncio.timeout(0.05):  # like the task handler's budget
                await turn(text)

        def contents() -> list[str]:
            return [m["content"] for m in client.docs["linebot_chat_memory/user::U1"]["recent_messages"]]

        await turn("一")
        started, release = threading.Event(), threading.Event()
        real_set = _DocRef.set

        def slow_first_set(ref, document_data, merge=False, **options):
            if not started.is_set():
                started.set()
                release.wait(5)
            real_set(ref, document_data, merge, **options)

        with patch.object(_DocRef, "set", slow_first_set):
            timed_out = asyncio.create_task(turn_with_deadline("二"))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            for _ in range(200):  # until the deadline has passed
                if timed_out.done() or timed_out.cancelling():
                    break
                await asyncio.sleep(0.01)
            next_turn = asyncio.create_task(turn("三"))
            await asyncio.sleep(0.05)
            # The timed-out turn still waits for its write, holding the lock.
            waiting = (timed_out.done(), next_turn.done(), contents())
            release.set()
            with self.assertRaises(TimeoutError):
                await timed_out
            await next_turn

        self.assertEqual(contents(), ["一", "好", "二", "好", "三", "好"])
        self.assertEqual(waiting, (False, False, ["一", "好"]))


class BoundedWriteTests(unittest.IsolatedAsyncioTestCase):
    """Regression (review finding 2): a write was waited for after its
    caller's cancellation for as long as the client's default (a 60 s retry
    deadline), so a turn the task handler timed out could wait past the
    handler's whole budget. Its two cancellations then kept the event's
    timeout from becoming a TimeoutError: no timeout notice, and a task
    running past the dispatch deadline. Each write now gives up within
    about twice WRITE_TIMEOUT_SECONDS."""

    async def test_chat_and_profile_writes_pass_the_bounded_retry_and_timeout(self) -> None:
        backend, client = _backend()
        await backend.save_chat("user::U1", {"summary_text": "x"})
        await backend.save_user_profile("U1", {"facts": []})
        await backend.delete_chat("user::U1")

        self.assertEqual([operation for operation, _, _ in client.write_options], ["set", "set", "delete"])
        for operation, retry, timeout in client.write_options:
            with self.subTest(operation=operation):
                self.assertEqual(timeout, WRITE_TIMEOUT_SECONDS)
                self.assertIsInstance(retry, core_retry.Retry)
                self.assertEqual(retry.timeout, WRITE_TIMEOUT_SECONDS)
                # Retried like the client's default commit retry, no more.
                self.assertTrue(retry._predicate(core_exceptions.ServiceUnavailable("x")))
                self.assertTrue(retry._predicate(core_exceptions.ResourceExhausted("x")))
                self.assertFalse(retry._predicate(core_exceptions.DeadlineExceeded("x")))

    def test_the_bound_leaves_the_timeout_notice_its_time(self) -> None:
        self.assertLessEqual(2 * WRITE_TIMEOUT_SECONDS + 2, EVENT_NOTICE_SECONDS)

    def test_a_failing_commit_stops_retrying_within_the_bound(self) -> None:
        """The options against the real retry machinery, wrapped like the
        client wraps commit (60 s retry deadline, 60 s timeout), on a fake
        clock: a commit that keeps answering ServiceUnavailable."""
        now = [0.0]
        attempts: list[float] = []

        def commit(*, timeout, **_):
            attempts.append(timeout)
            now[0] += 0.5
            raise core_exceptions.ServiceUnavailable("busy")

        def sleep(seconds: float) -> None:
            now[0] += seconds

        default_retry = retry_unary.Retry(
            initial=0.1,
            maximum=60.0,
            multiplier=1.3,
            predicate=core_retry.if_exception_type(
                core_exceptions.ResourceExhausted, core_exceptions.ServiceUnavailable
            ),
            timeout=60.0,
        )
        wrapped = gapic_v1.method.wrap_method(commit, default_retry=default_retry, default_timeout=60.0)
        options = FirestoreBackend(project_id="p", client=FakeFirestore())._write_options
        clock = SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
        with (
            patch.object(retry_unary, "time", clock),
            patch.object(retry_base, "time", clock),
            patch.object(retry_base.random, "uniform", lambda a, b: b),
        ):
            with self.assertRaises(core_exceptions.RetryError):
                wrapped(**options)
            bounded = now[0]
            now[0], attempts_bounded = 0.0, len(attempts)
            with self.assertRaises(core_exceptions.RetryError):
                wrapped()  # the client's default

        self.assertGreater(attempts_bounded, 1)  # still retried
        self.assertLessEqual(bounded, WRITE_TIMEOUT_SECONDS + 0.5)
        self.assertGreater(now[0], 50.0)  # the default kept retrying for about a minute
        self.assertTrue(all(timeout <= WRITE_TIMEOUT_SECONDS for timeout in attempts[:attempts_bounded]))


class CancelledWriteTests(unittest.IsolatedAsyncioTestCase):
    """A cancelled chat or profile write is waited for, then re-cancelled."""

    async def _cancel_during(self, write, operation, *, fails: bool = False):
        started, release = threading.Event(), threading.Event()
        real = getattr(_DocRef, operation)

        def slow(ref, *args, **kwargs):
            started.set()
            release.wait(5)
            if fails:
                raise RuntimeError("commit failed")
            return real(ref, *args, **kwargs)

        with patch.object(_DocRef, operation, slow):
            task = asyncio.create_task(write())
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            task.cancel()
            await asyncio.sleep(0.02)
            task.cancel()  # a second cancellation is absorbed the same way
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_a_cancelled_save_and_delete_finish_first(self) -> None:
        backend, client = _backend()
        await self._cancel_during(lambda: backend.save_chat("user::U1", {"summary_text": "x"}), "set")
        self.assertEqual(client.docs["linebot_chat_memory/user::U1"], {"summary_text": "x"})

        await self._cancel_during(lambda: backend.delete_chat("user::U1"), "delete")
        self.assertNotIn("linebot_chat_memory/user::U1", client.docs)

    async def test_a_write_failing_after_the_cancellation_is_logged(self) -> None:
        backend, client = _backend()
        with patch.object(memory_backends_module.logger, "warning") as log_warning:
            await self._cancel_during(
                lambda: backend.save_user_profile("U1", {"facts": []}), "set", fails=True
            )
        self.assertEqual(client.docs, {})
        self.assertIn("after its caller was cancelled", log_warning.call_args.args[0])
