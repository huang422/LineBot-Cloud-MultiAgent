"""The Colab state document and its stores (in memory; Firestore on a fake client)."""

from __future__ import annotations

import inspect
import math
import unittest
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from src.config import Settings
from src.services.colab.state import (
    AUTO,
    BOOTING,
    IDLE,
    OFF,
    READY,
    UPDATE_ATTEMPTS,
    ColabState,
    ColabStateError,
    FirestoreColabStateStore,
    InMemoryColabStateStore,
    build_state_store,
    clip,
)

try:  # google-cloud-firestore is in uv.lock; skip cleanly without it.
    from google.api_core import exceptions as gexc
    from google.cloud.firestore_v1.client import Client as RealClient
    from google.cloud.firestore_v1.document import DocumentReference as RealDocument
except ImportError:  # pragma: no cover - depends on the environment
    gexc = None


class StateDocumentTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        state = ColabState(
            status=READY,
            since=10.0,
            desired=OFF,
            nbh="n",
            tunnel_url="https://a.trycloudflare.com",
            digests={"qwen3.8:27b-mtp-q4_K_M": "2213"},
            vram={"nimble:9b": {"size": 10_601_505_280, "size_vram": 10_601_505_280}},
            gpu={"name": "A100", "vram_mib": 40960, "router_on_gpu": "10110/10110 MiB"},
            ccu_balance=12.5,
            failures=2,
            manual_demand_at=11.0,
            a100_eligible=False,
            starts_today=2,
            cooldown_step=1,
            health_failures=1,
            action="assign",
            action_at=9.0,
            attempts=2,
            known_endpoints=("ep-1", "ep-2"),
            unanswered_nbhs=("nbh-a",),
            unanswered_at=7.0,
            cu_accounted_at=8.0,
            reconciled_day="2026-09-30",
            hold_reason="CU 餘額不足",
            version=7,
        )
        doc = state.to_doc()
        # Firestore stores lists; nothing in the document is a tuple.
        self.assertEqual(doc["known_endpoints"], ["ep-1", "ep-2"])
        self.assertEqual(doc["unanswered_nbhs"], ["nbh-a"])
        self.assertEqual(ColabState.from_doc(doc), state)

    def test_a_missing_or_broken_document_falls_back_field_by_field(self) -> None:
        self.assertEqual(ColabState.from_doc(None), ColabState())
        self.assertEqual(ColabState.from_doc([]), ColabState())

        state = ColabState.from_doc(
            {
                "status": "EXPLODED",
                "desired": "maybe",
                "since": "yesterday",
                "failures": 2.0,
                "ccu_balance": math.nan,
                "ccu_rate": True,
                "tunnel_url": 42,
                "digests": ["not", "a", "map"],
                "vram": "all of it",
                "cu_used_today": 1,
                "ready_at": None,
                "cu_day": None,
                "unknown_field": "ignored",
                "known_endpoints": ["ok", 3],
                "a100_eligible": "yes",
                "attempts": "2",
                "version": 4.0,
            }
        )

        self.assertEqual((state.status, state.desired, state.since, state.failures), (IDLE, AUTO, 0.0, 2))
        self.assertEqual((state.ccu_balance, state.ccu_rate, state.tunnel_url), (None, None, ""))
        self.assertEqual((state.digests, state.cu_used_today, state.ready_at, state.cu_day), ({}, 1.0, None, ""))
        self.assertEqual(state.vram, {})
        self.assertEqual((state.known_endpoints, state.a100_eligible, state.attempts, state.version), ((), None, 0, 4))
        # A list of names is kept (the newest ones only).
        many = [f"ep-{i}" for i in range(30)]
        self.assertEqual(ColabState.from_doc({"known_endpoints": many}).known_endpoints, tuple(many[-20:]))
        self.assertEqual(ColabState.from_doc({"a100_eligible": True}).a100_eligible, True)

    def test_since_moves_only_with_the_status(self) -> None:
        state = ColabState(status=BOOTING, since=5.0)

        self.assertEqual(state.with_status(BOOTING, 9.0, boot_step="pull").since, 5.0)
        self.assertEqual(state.with_status(READY, 9.0).since, 9.0)
        with self.assertRaises(ValueError):
            state.with_status("RUNNING", 9.0)
        self.assertTrue(state.active)
        self.assertFalse(ColabState().active)

    def test_clip(self) -> None:
        self.assertEqual(clip("a\n  b"), "a b")
        self.assertEqual(len(clip("x" * 500)), 300)
        self.assertTrue(clip("x" * 500).endswith("…"))


class InMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_update_and_cache(self) -> None:
        store = InMemoryColabStateStore(ColabState(status=BOOTING))
        self.assertIsNone(store.cached)

        self.assertEqual((await store.load()).status, BOOTING)
        updated = await store.update(lambda s: s.with_status(READY, 1.0))
        unchanged = await store.update(lambda s: None)
        same = await store.update(lambda s: replace(s))

        self.assertEqual((updated.status, unchanged.status, same.status, store.cached.status), (READY,) * 4)
        self.assertEqual(store.writes, 1)
        # Every write bumps the version; a no-op does not.
        self.assertEqual((updated.version, store.cached.version), (1, 1))
        self.assertEqual((await store.update(lambda s: replace(s, failures=1))).version, 2)


class _Snapshot:
    def __init__(self, data: dict | None, update_time: int) -> None:
        self.exists = data is not None
        self.update_time = update_time
        self._data = deepcopy(data)

    def to_dict(self) -> dict | None:
        return deepcopy(self._data)


class _Document:
    def __init__(self, client: _FakeFirestore, path: str) -> None:
        self._client = client
        self.path = path

    def get(self, *, timeout=None) -> _Snapshot:
        self._client.calls.append("get")
        self._client.fail("get")
        return _Snapshot(self._client.docs.get(self.path), self._client.times.get(self.path, 0))

    def create(self, document_data: dict, *, timeout=None) -> None:
        self._client.calls.append("create")
        self._client.before_write()
        if self.path in self._client.docs:
            raise gexc.Conflict("exists")
        self._write(document_data)

    def update(self, field_updates: dict, option=None, *, timeout=None) -> None:
        self._client.calls.append("update")
        self._client.before_write()
        if option is None or option.last_update_time != self._client.times.get(self.path):
            raise gexc.FailedPrecondition("changed")
        self._write({**self._client.docs[self.path], **field_updates})

    def _write(self, data: dict) -> None:
        self._client.docs[self.path] = deepcopy(data)
        self._client.times[self.path] = self._client.times.get(self.path, 0) + 1


class _FakeFirestore:
    """The calls FirestoreColabStateStore makes, over a dict."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.times: dict[str, int] = {}
        self.calls: list[str] = []
        self.racers = 0  # writes by "someone else" just before ours
        self.errors: dict[str, Exception] = {}

    def collection(self, name: str):
        return SimpleNamespace(document=lambda doc_id: _Document(self, f"{name}/{doc_id}"))

    def write_option(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def fail(self, op: str) -> None:
        if op in self.errors:
            raise self.errors[op]

    def before_write(self) -> None:
        if self.racers:
            self.racers -= 1
            path = "linebot_colab/state"
            self.docs[path] = {**self.docs.get(path, {}), "last_error": "raced"}
            self.times[path] = self.times.get(path, 0) + 1


@unittest.skipIf(gexc is None, "google-cloud-firestore is not installed")
class FirestoreStoreTests(unittest.IsolatedAsyncioTestCase):
    def test_the_fake_matches_the_client_library(self) -> None:
        for fake, real in (
            (_Document.get, RealDocument.get),
            (_Document.create, RealDocument.create),
            (_Document.update, RealDocument.update),
        ):
            with self.subTest(method=real.__name__):
                real_params = inspect.signature(real).parameters
                for name in inspect.signature(fake).parameters:
                    self.assertIn(name, real_params)
        self.assertIn("kwargs", inspect.signature(RealClient.write_option).parameters)

    async def test_missing_document_is_idle_and_created_on_first_write(self) -> None:
        client = _FakeFirestore()
        store = FirestoreColabStateStore(client=client)

        self.assertEqual(await store.load(), ColabState())
        state = await store.update(lambda s: replace(s, last_demand_at=5.0))

        self.assertEqual(state.last_demand_at, 5.0)
        self.assertEqual(client.docs["linebot_colab/state"]["last_demand_at"], 5.0)
        self.assertEqual(client.calls, ["get", "get", "create"])
        self.assertEqual(store.cached, state)
        self.assertEqual((state.version, client.docs["linebot_colab/state"]["version"]), (1, 1))
        state = await store.update(lambda s: replace(s, known_endpoints=("ep-1",)))
        self.assertEqual(client.docs["linebot_colab/state"]["known_endpoints"], ["ep-1"])
        self.assertEqual(((await store.load()).known_endpoints, state.version), (("ep-1",), 2))

    async def test_updates_are_conditional_and_retried_when_someone_else_wrote(self) -> None:
        client = _FakeFirestore()
        store = FirestoreColabStateStore(client=client, collection_prefix="linebot")
        await store.update(lambda s: s.with_status(BOOTING, 1.0, nbh="n"))
        client.racers = 1
        client.calls.clear()

        state = await store.update(lambda s: s.with_status(READY, 2.0))

        # The second read saw the other writer's change, and kept it.
        self.assertEqual((state.status, state.last_error), (READY, "raced"))
        self.assertEqual(client.calls, ["get", "update", "get", "update"])

    async def test_a_no_op_change_writes_nothing(self) -> None:
        client = _FakeFirestore()
        store = FirestoreColabStateStore(client=client)

        await store.update(lambda s: None)
        await store.update(lambda s: s)

        self.assertEqual(client.calls, ["get", "get"])

    async def test_it_gives_up_after_a_few_lost_races(self) -> None:
        client = _FakeFirestore()
        store = FirestoreColabStateStore(client=client)
        await store.update(lambda s: replace(s, failures=1))
        client.racers = UPDATE_ATTEMPTS

        with self.assertRaisesRegex(ColabStateError, "kept changing"):
            await store.update(lambda s: replace(s, failures=s.failures + 1))

    async def test_client_errors_become_state_errors(self) -> None:
        client = _FakeFirestore()
        client.errors["get"] = gexc.ServiceUnavailable("down")
        store = FirestoreColabStateStore(client=client)

        with self.assertRaises(ColabStateError):
            await store.load()
        with self.assertRaises(ColabStateError):
            await store.update(lambda s: replace(s, failures=1))
        self.assertIsNone(store.cached)

    def test_the_client_is_built_on_first_use_and_the_prefix_is_used(self) -> None:
        store = FirestoreColabStateStore(project_id="p", collection_prefix="bot_")
        self.assertIsNone(store._client)
        self.assertEqual(store.collection, "bot_colab")


class BuildStoreTests(unittest.TestCase):
    def test_firestore_only_when_enabled(self) -> None:
        self.assertIsInstance(build_state_store(Settings(_env_file=None)), InMemoryColabStateStore)
        store = build_state_store(
            Settings(_env_file=None, firestore_enabled=True, firestore_collection_prefix="lb")
        )
        self.assertIsInstance(store, FirestoreColabStateStore)
        self.assertEqual(store.collection, "lb_colab")


if __name__ == "__main__":
    unittest.main()
