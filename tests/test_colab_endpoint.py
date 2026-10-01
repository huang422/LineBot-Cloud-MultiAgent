"""ColabEndpoint: readiness, the busy gate, the breaker, health, demand."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest.mock import patch

import httpx

from src.providers.base import ProviderResponse, SkipTierError, TransientError
from src.providers.ollama_provider import OllamaChatProvider
from src.services.colab import protocol
from src.services.colab.endpoint import (
    BREAKER_FAILURES,
    BREAKER_OPEN_SECONDS,
    DEMAND_INTERVAL_SECONDS,
    READY_HEALTH_TTL_SECONDS,
    STATE_LOAD_RETRY_SECONDS,
    ColabEndpoint,
    ColabLimits,
)
from src.services.colab.state import (
    BOOTING,
    IDLE,
    READY,
    ColabState,
    ColabStateError,
    InMemoryColabStateStore,
)
from src.services.search.page_fetcher import BlockedURLError
from tests.colab_fakes import (
    NBH,
    NIMBLE,
    QWEN,
    SECRET,
    TUNNEL,
    Clock,
    FakeTunnel,
    bearer,
    healthy_routes,
    make_endpoint,
    on_gpu,
    ready_state,
)

_USER = [{"role": "user", "content": "你好"}]


class _FailingStore(InMemoryColabStateStore):
    def __init__(self, error: BaseException | None = None, delay: float = 0) -> None:
        super().__init__(ready_state())
        self.loads = 0
        self._error = error
        self._delay = delay

    async def load(self) -> ColabState:
        self.loads += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return await super().load()


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_needs_status_tunnel_session_and_a_fresh_health_check(self) -> None:
        clock = Clock()
        now = clock()
        cases = [
            (ready_state(now), True),
            (ready_state(now, last_health_at=now - READY_HEALTH_TTL_SECONDS), True),
            (ready_state(now, last_health_at=now - READY_HEALTH_TTL_SECONDS - 1), False),
            (ready_state(now, last_health_at=None), False),
            (ready_state(now, status=BOOTING), False),
            (ready_state(now, tunnel_url=""), False),
            (ready_state(now, nbh=""), False),
        ]
        for state, ready in cases:
            endpoint, _ = make_endpoint(state, clock=clock)
            with self.subTest(state=state):
                self.assertFalse(endpoint.is_ready())  # nothing read yet
                self.assertEqual(await endpoint.ready(), ready)
            await endpoint.close()

    async def test_the_state_is_read_once_per_process(self) -> None:
        store = _FailingStore()
        endpoint = ColabEndpoint(SECRET, store, transport=FakeTunnel().transport())

        for _ in range(5):
            self.assertTrue(await endpoint.ready())

        self.assertEqual(store.loads, 1)
        await endpoint.close()

    async def test_an_unreadable_state_means_not_ready_and_a_paced_retry(self) -> None:
        clock = Clock()
        for store in (_FailingStore(ColabStateError("firestore down")), _FailingStore(delay=10)):
            endpoint = ColabEndpoint(SECRET, store, transport=FakeTunnel().transport(), clock=clock)
            with self.subTest(store=store._error or "slow"):
                if store._delay:
                    with patch("src.services.colab.endpoint.STATE_LOAD_TIMEOUT_SECONDS", 0.05):
                        self.assertFalse(await endpoint.ready())
                else:
                    self.assertFalse(await endpoint.ready())
                self.assertFalse(await endpoint.ready())
                self.assertEqual(store.loads, 1)
                clock.now += STATE_LOAD_RETRY_SECONDS
                store._error, store._delay = None, 0
                self.assertTrue(await endpoint.ready())
                self.assertEqual(store.loads, 2)
            await endpoint.close()

    async def test_unhealthy_until_the_next_passing_check(self) -> None:
        clock = Clock()
        endpoint, tunnel = make_endpoint(clock=clock)
        self.assertTrue(await endpoint.ready())

        clock.now += 1
        endpoint.mark_unhealthy("404")
        self.assertFalse(endpoint.is_ready())
        clock.now += 1
        self.assertTrue(await endpoint.check_health())
        self.assertTrue(endpoint.is_ready())
        self.assertEqual(tunnel.paths(), ["GET /api/version", "GET /api/ps"])
        self.assertTrue(all(r.headers["authorization"] == f"Bearer {bearer()}" for r in tunnel.requests))
        await endpoint.close()

    async def test_a_new_session_starts_with_a_clean_slate(self) -> None:
        clock = Clock()
        store = InMemoryColabStateStore(ready_state(clock()))
        endpoint, _ = make_endpoint(store=store, clock=clock)
        await endpoint.ready()
        for _ in range(BREAKER_FAILURES):
            endpoint.record_failure("boom")
        endpoint.mark_unhealthy("boom")
        self.assertTrue(endpoint.breaker_open())

        await store.update(lambda s: ready_state(clock(), nbh="new-session"))

        self.assertTrue(endpoint.is_ready())
        self.assertFalse(endpoint.breaker_open())
        self.assertEqual(endpoint.bearer(), bearer("new-session"))
        await endpoint.close()

    async def test_the_health_check_fails_on_a_wrong_or_silent_session(self) -> None:
        clock = Clock()
        cases = {
            "old ollama": {"GET /api/version": (200, {"version": "0.34.4"})},
            "model unloaded": {"GET /api/ps": (200, {"models": []})},
            "chat unloaded": {"GET /api/ps": (200, {"models": [{"name": NIMBLE, **on_gpu(NIMBLE)}]})},
            "proxy down": {"GET /api/version": httpx.ConnectError("refused")},
            "slow": {"GET /api/ps": ("hang", 30)},
        }
        for name, routes in cases.items():
            endpoint, _ = make_endpoint(tunnel=FakeTunnel(**routes), clock=clock)
            await endpoint.ready()
            clock.now += 1
            with self.subTest(name):
                with patch("src.services.colab.endpoint.HEALTH_TIMEOUT_SECONDS", 0.1):
                    self.assertFalse(await endpoint.check_health())
                self.assertFalse(endpoint.is_ready())
                self.assertIn("health check", endpoint.last_error)
            await endpoint.close()
        endpoint, tunnel = make_endpoint(ready_state(clock(), status=IDLE, tunnel_url=""), clock=clock)
        await endpoint.ready()
        self.assertFalse(await endpoint.check_health())
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()


    async def test_a_router_gone_from_the_gpu_sits_the_session_out_and_chat_goes_on(self) -> None:
        # qwen3.8 reloaded with nimble resident: Ollama evicts nimble.
        clock = Clock()
        only_chat = {"GET /api/ps": (200, {"models": [{"name": QWEN, **on_gpu(QWEN)}]})}
        endpoint, tunnel = make_endpoint(tunnel=FakeTunnel(**only_chat), clock=clock)
        await endpoint.ready()
        self.assertEqual(endpoint.off_gpu(), {})

        clock.now += 60
        with self.assertLogs("linebot", "WARNING") as logs:
            self.assertTrue(await endpoint.check_health())
            self.assertTrue(await endpoint.check_health())

        self.assertTrue(endpoint.is_ready())
        self.assertEqual((endpoint.last_error, endpoint.breaker_open()), ("", False))
        self.assertEqual(endpoint.off_gpu(), {NIMBLE: (0, 0)})
        (warning,) = logs.output
        self.assertIn(f"health check found {NIMBLE} not loaded; the router sits the rest of this session out", warning)
        # Loaded again (nothing here calls it, but say it were): still out,
        # since a call could evict qwen3.8 the next time.
        tunnel.routes.update(healthy_routes())
        self.assertTrue(await endpoint.check_health())
        self.assertEqual(endpoint.off_gpu(), {NIMBLE: (0, 0)})
        self.assertEqual(endpoint.view()["off_gpu"], {NIMBLE: {"size_vram": 0, "size": 0}})
        # A new session starts with its router.
        await endpoint.store.update(lambda s: ready_state(clock(), nbh="new-session"))
        self.assertEqual(endpoint.off_gpu(), {})
        await endpoint.close()

    async def test_a_router_partly_on_the_cpu_at_a_health_check_sits_the_session_out(self) -> None:
        clock = Clock()
        spilled = {"GET /api/ps": (200, {"models": [{"name": QWEN, **on_gpu(QWEN)}, {"name": NIMBLE, **on_gpu(NIMBLE, 1 << 30)}]})}
        endpoint, _ = make_endpoint(tunnel=FakeTunnel(**spilled), clock=clock)
        await endpoint.ready()

        with self.assertLogs("linebot", "WARNING") as logs:
            self.assertTrue(await endpoint.check_health())

        self.assertEqual(endpoint.off_gpu(), {NIMBLE: (1 << 30, on_gpu(NIMBLE)["size"])})
        self.assertIn("not wholly in VRAM", logs.output[0])
        await endpoint.close()

    async def test_a_failed_health_check_drops_no_router(self) -> None:
        clock = Clock()
        endpoint, _ = make_endpoint(tunnel=FakeTunnel(**{"GET /api/ps": (200, {"models": []})}), clock=clock)
        await endpoint.ready()

        self.assertFalse(await endpoint.check_health())

        self.assertIn(f"not loaded: {QWEN}", endpoint.last_error)
        self.assertNotIn(NIMBLE, endpoint.last_error)
        # A failed check says nothing about the router: it stays in.
        self.assertEqual(endpoint.off_gpu(), {})
        await endpoint.close()


class NotReadySkipTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_not_ready_tier_is_skipped_instantly_without_network(self) -> None:
        for status in (IDLE, BOOTING, "DRAINING", "COOLDOWN", "DISABLED"):
            tunnel = FakeTunnel()
            endpoint, _ = make_endpoint(ColabState(status=status), tunnel)
            started = time.perf_counter()
            with self.subTest(status=status), self.assertRaises(SkipTierError):
                await OllamaChatProvider(endpoint).generate(QWEN, _USER)
            self.assertLess(time.perf_counter() - started, 0.1)
            self.assertEqual(tunnel.requests, [])
            await endpoint.close()


class BusyGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_busy_model_is_skipped_not_queued(self) -> None:
        release = asyncio.Event()

        async def slow_chat(request):
            await release.wait()
            return httpx.Response(200, json={"message": {"content": "第一個"}, "done_reason": "stop"})

        endpoint, tunnel = make_endpoint(tunnel=FakeTunnel(**{"POST /api/chat": slow_chat}))
        provider = OllamaChatProvider(endpoint)
        first = asyncio.create_task(provider.generate(QWEN, _USER))
        while not endpoint.semaphore(QWEN).locked():
            await asyncio.sleep(0)

        started = time.perf_counter()
        with self.assertRaisesRegex(SkipTierError, "busy"):
            await provider.generate(QWEN, _USER)
        self.assertLess(time.perf_counter() - started, 0.1)

        release.set()
        self.assertEqual((await first).text, "第一個")
        self.assertFalse(endpoint.semaphore(QWEN).locked())
        self.assertEqual(len(tunnel.bodies("POST /api/chat")), 1)
        self.assertIsInstance(await provider.generate(QWEN, _USER), ProviderResponse)
        await endpoint.close()

    async def test_each_model_has_its_own_gate(self) -> None:
        endpoint, _ = make_endpoint()
        async with endpoint.semaphore(QWEN):
            self.assertTrue(endpoint.semaphore(QWEN).locked())
            self.assertFalse(endpoint.semaphore(NIMBLE).locked())
        self.assertIsNot(endpoint.semaphore(QWEN), endpoint.semaphore(NIMBLE))
        self.assertIs(endpoint.semaphore(QWEN), endpoint.semaphore(QWEN))
        await endpoint.close()


class BreakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_failures_in_a_row_open_it_for_two_minutes(self) -> None:
        clock = Clock()
        endpoint, _ = make_endpoint(clock=clock)
        self.assertTrue(await endpoint.ready())
        self.assertEqual((BREAKER_FAILURES, BREAKER_OPEN_SECONDS), (3, 120))

        endpoint.record_failure("a")
        endpoint.record_failure("b")
        endpoint.record_success()  # a success resets the count
        endpoint.record_failure("c")
        endpoint.record_failure("d")
        self.assertFalse(endpoint.breaker_open())
        endpoint.record_failure("e")
        self.assertTrue(endpoint.breaker_open())

        with self.assertRaisesRegex(SkipTierError, "breaker"):
            await OllamaChatProvider(endpoint).generate(QWEN, _USER)
        clock.now += BREAKER_OPEN_SECONDS - 1
        self.assertTrue(endpoint.breaker_open())
        clock.now += 1
        self.assertFalse(endpoint.breaker_open())
        await endpoint.close()

    async def test_failed_calls_open_it(self) -> None:
        endpoint, tunnel = make_endpoint(tunnel=FakeTunnel(**{"POST /api/chat": (502, "bad gateway")}))
        provider = OllamaChatProvider(endpoint)
        for _ in range(BREAKER_FAILURES):
            with self.assertRaises(TransientError):
                await provider.generate(QWEN, _USER)

        with self.assertRaises(SkipTierError):
            await provider.generate(QWEN, _USER)
        self.assertEqual(len(tunnel.requests), BREAKER_FAILURES)
        await endpoint.close()


class TunnelClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_default_transport_refuses_addresses_inside(self) -> None:
        endpoint = ColabEndpoint(SECRET, InMemoryColabStateStore(ready_state()))
        for base in ("https://127.0.0.1", "https://10.1.2.3", "https://169.254.169.254", "https://localhost"):
            with self.subTest(base=base), self.assertRaises(BlockedURLError):
                await endpoint.call("GET", "/api/version", base_url=base, nbh=NBH)
        await endpoint.close()

    async def test_no_session_no_call(self) -> None:
        endpoint, tunnel = make_endpoint(ColabState())
        await endpoint.ready()
        with self.assertRaises(RuntimeError):
            await endpoint.call("GET", "/api/version")
        self.assertEqual(tunnel.requests, [])
        await endpoint.close()

    async def test_redirects_are_not_followed(self) -> None:
        tunnel = FakeTunnel(**{"GET /api/version": httpx.Response(302, headers={"location": "https://10.0.0.1/"})})
        endpoint, _ = make_endpoint(tunnel=tunnel)
        await endpoint.ready()

        response = await endpoint.call("GET", "/api/version")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(tunnel.requests), 1)
        await endpoint.close()


class DemandTests(unittest.IsolatedAsyncioTestCase):
    async def test_at_most_one_write_a_minute(self) -> None:
        clock = Clock()
        store = InMemoryColabStateStore(ColabState())
        endpoint, _ = make_endpoint(store=store, clock=clock)

        self.assertTrue(await endpoint.note_demand())
        self.assertFalse(await endpoint.note_demand())
        clock.now += DEMAND_INTERVAL_SECONDS - 1
        self.assertFalse(await endpoint.note_demand())
        self.assertEqual(store.writes, 1)
        clock.now += 1
        self.assertTrue(await endpoint.note_demand())
        self.assertEqual((store.writes, store.cached.last_demand_at), (2, clock.now))
        await endpoint.close()

    async def test_a_newer_stored_demand_is_kept(self) -> None:
        clock = Clock()
        store = InMemoryColabStateStore(ColabState(last_demand_at=clock.now + 30))
        endpoint, _ = make_endpoint(store=store, clock=clock)

        await endpoint.note_demand()

        self.assertEqual((store.writes, store.cached.last_demand_at), (0, clock.now + 30))
        await endpoint.close()

    async def test_a_slow_write_gives_up_without_raising(self) -> None:
        store = InMemoryColabStateStore()

        async def slow(change):
            await asyncio.sleep(30)

        store.update = slow
        endpoint, _ = make_endpoint(store=store)
        started = time.monotonic()

        with patch("src.services.colab.endpoint.DEMAND_WRITE_TIMEOUT_SECONDS", 0.2), self.assertLogs("linebot", "WARNING"):
            self.assertFalse(await endpoint.note_demand())

        self.assertLess(time.monotonic() - started, 1.0)
        await endpoint.close()

    async def test_a_failed_write_is_swallowed(self) -> None:
        store = InMemoryColabStateStore()

        async def broken(change):
            raise ColabStateError("firestore down")

        store.update = broken
        endpoint, _ = make_endpoint(store=store)

        self.assertFalse(await endpoint.note_demand())
        await endpoint.close()


class ViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_view_has_no_url_bearer_or_ids(self) -> None:
        clock = Clock()
        state = ready_state(
            clock(),
            endpoint="colab-endpoint-id",
            kernel_id="kernel-1",
            ccu_balance=123.4,
            ccu_rate=5.37,
            cu_used_today=2.5,
            cu_day="2026-09-30",
        )
        endpoint, _ = make_endpoint(state, clock=clock)
        endpoint.limits = ColabLimits(daily_cu_cap=30)
        before = endpoint.view()
        await endpoint.ready()

        view = endpoint.view()

        self.assertEqual((before["loaded"], "status" in before), (False, False))
        self.assertEqual((view["status"], view["ready"], view["tunnel"]), (READY, True, "quick"))
        self.assertEqual((view["ccu_balance"], view["ccu_rate_per_hour"], view["cu_used_today"]), (123.4, 5.37, 2.5))
        self.assertEqual(view["limits"]["daily_cu_cap"], 30)
        self.assertIsNone(view["unpinned"])
        self.assertIsNone(view["off_gpu"])
        dumped = json.dumps(view)
        for secret in (TUNNEL, bearer(), NBH, "colab-endpoint-id", "kernel-1", SECRET):
            self.assertNotIn(secret, dumped)

    async def test_a_republished_router_shows_both_digests(self) -> None:
        pins = protocol.pinned_models()
        other = "adfcdbb4070d303fb43fc11fc06a14269ec2bc7136a55369e86d412b86ba2831"
        endpoint, _ = make_endpoint(ready_state(digests={QWEN: pins[QWEN], NIMBLE: other}))
        await endpoint.ready()

        view = endpoint.view()

        self.assertEqual(view["unpinned"], {NIMBLE: {"pinned": pins[NIMBLE], "actual": other}})
        self.assertIsNone(view["off_gpu"])
        self.assertTrue(view["ready"])
        await endpoint.close()

    async def test_a_router_with_no_digest_or_not_wholly_in_vram_shows_why_it_sits_out(self) -> None:
        pins = protocol.pinned_models()
        spilled = {QWEN: on_gpu(QWEN), NIMBLE: on_gpu(NIMBLE, size_vram=1 << 30)}
        endpoint, _ = make_endpoint(ready_state(digests={QWEN: pins[QWEN]}, vram=spilled))
        await endpoint.ready()

        view = endpoint.view()

        self.assertEqual(view["unpinned"], {NIMBLE: {"pinned": pins[NIMBLE], "actual": None}})
        self.assertEqual(view["off_gpu"], {NIMBLE: {"size_vram": 1 << 30, "size": on_gpu(NIMBLE)["size"]}})
        self.assertTrue(view["ready"])
        await endpoint.close()

    async def test_no_router_sits_out_while_no_session_is_ready(self) -> None:
        # No probe has looked yet (or the session ended): nothing to report.
        for status in (IDLE, BOOTING, "DRAINING", "COOLDOWN", "DISABLED"):
            endpoint, _ = make_endpoint(ColabState(status=status))
            await endpoint.ready()
            with self.subTest(status=status):
                self.assertEqual((endpoint.unpinned(), endpoint.off_gpu()), ({}, {}))
                view = endpoint.view()
                self.assertEqual((view["unpinned"], view["off_gpu"]), (None, None))
            await endpoint.close()

    async def test_a_named_tunnel_shows_as_named(self) -> None:
        endpoint = ColabEndpoint(
            SECRET,
            InMemoryColabStateStore(ready_state(tunnel_url="https://ollama.example.org")),
            tunnel_host="ollama.example.org",
            transport=FakeTunnel().transport(),
        )
        await endpoint.ready()
        self.assertEqual(endpoint.view()["tunnel"], "named")
        await endpoint.close()


if __name__ == "__main__":
    unittest.main()
