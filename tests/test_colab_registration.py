"""The VM's signed registration: protocol, checks, probe and the HTTP route."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main as main_module
from src.models.agent_request import AgentRequest
from src.providers.fallback_chain import Tier
from src.providers.systemone import RouterGates, SystemOneRouter
from src.services.colab import protocol
from src.services.colab.registration import ColabRegistrar
from src.services.colab.state import (
    ASSIGNING,
    BOOTING,
    COOLDOWN,
    DISABLED,
    DRAINING,
    IDLE,
    OFF,
    READY,
    ColabState,
    ColabStateError,
    InMemoryColabStateStore,
)
from tests.colab_fakes import (
    NBH,
    QWEN,
    ROUTER,
    SECRET,
    TUNNEL,
    TUNNEL_HOST,
    Clock,
    FakeTunnel,
    make_endpoint,
    on_gpu,
)

_NOW = 1_790_000_000.0
_PUBLIC = ["104.16.230.132"]


async def _public(host: str, port: int) -> list[str]:
    return list(_PUBLIC)


def _body(event: str = "ready", **extra) -> bytes:
    payload = {"event": event, "nbh": NBH}
    if event == "ready":
        payload.update(
            tunnel_url=TUNNEL,
            versions={"ollama": "0.40.5", "cloudflared": "2026.9.3"},
            digests=protocol.colab_models(),
            gpu={
                "name": "NVIDIA A100-SXM4-40GB",
                "vram_mib": 40960,
                "driver": "550.54.15",
                "chat_on_gpu": "24128/24128 MiB",
                "router_on_gpu": "10110/10110 MiB",
                "used_mib": 35012,
            },
        )
    payload.update(extra)
    return json.dumps(payload).encode()


def _signed(body: bytes, *, at: float = _NOW, nbh: str = NBH, secret: str = SECRET) -> str:
    return protocol.sign(protocol.derive_reg_key(secret, nbh), int(at), body)


class ProtocolTests(unittest.TestCase):
    def test_key_derivation_matches_the_documented_formula(self) -> None:
        reg = hmac.new(SECRET.encode(), f"reg:{NBH}".encode(), hashlib.sha256).hexdigest()
        proxy = base64.urlsafe_b64encode(
            hmac.new(SECRET.encode(), f"proxy:{NBH}".encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()

        self.assertEqual(protocol.derive_reg_key(SECRET, NBH), reg)
        self.assertEqual(protocol.derive_proxy_token(SECRET, NBH), proxy)
        self.assertEqual((len(reg), len(proxy)), (64, 43))
        self.assertNotIn("=", proxy)
        self.assertNotEqual(protocol.derive_reg_key(SECRET, NBH + "x"), reg)
        self.assertNotEqual(protocol.derive_proxy_token(SECRET + "x", NBH), proxy)

    def test_signature_is_the_hex_hmac_of_timestamp_dot_body_keyed_by_the_hex_reg_key(self) -> None:
        reg_key = protocol.derive_reg_key(SECRET, NBH)
        body = b'{"event":"ready"}'
        expected = hmac.new(reg_key.encode("ascii"), b"1790000000." + body, hashlib.sha256).hexdigest()

        self.assertEqual(protocol.sign(reg_key, 1790000000, body), f"t=1790000000,v1={expected}")
        parsed = protocol.parse_signature(protocol.sign(reg_key, 1790000000, body))
        self.assertEqual((parsed.timestamp, parsed.v1), (1790000000, expected))
        self.assertTrue(protocol.signature_matches(reg_key, parsed, body))
        self.assertFalse(protocol.signature_matches(reg_key, parsed, body + b" "))

    def test_malformed_signature_headers(self) -> None:
        good = "a" * 64
        for header in (
            None,
            "",
            f"v1={good}",
            "t=1790000000",
            f"t=abc,v1={good}",
            f"t=1790000000,v1={'g' * 64}",
            f"t=1790000000,v1={good[:-1]}",
            f"t=1,t=2,v1={good}",
            f"t=1790000000,v1={good},v0=x",
            f"t=1790000000;v1={good}",
            f"t=1790000000,v1={good}" + " " * 200,
            # Digits that are not ASCII: "²" passes str.isdigit() but not int().
            f"t=²,v1={good}",
            f"t=179000000¹,v1={good}",
            f"t=١٧٩٠,v1={good}",
            f"t=1234567890123,v1={good}",
        ):
            with self.subTest(header=header):
                self.assertIsNone(protocol.parse_signature(header))
        self.assertIsNotNone(protocol.parse_signature(f" t=1790000000 , v1={good.upper()} "))

    def test_tunnel_url_rule(self) -> None:
        self.assertEqual(protocol.check_tunnel_url(TUNNEL), TUNNEL_HOST)
        self.assertEqual(protocol.check_tunnel_url(TUNNEL + "/"), TUNNEL_HOST)
        self.assertEqual(protocol.check_tunnel_url("https://ollama.example.org", "ollama.example.org"), "ollama.example.org")
        refused = [
            None,
            42,
            "",
            f" {TUNNEL}",
            f"http://{TUNNEL_HOST}",
            f"https://user:pw@{TUNNEL_HOST}",
            f"https://user@{TUNNEL_HOST}",
            f"https://{TUNNEL_HOST}:443",
            f"https://{TUNNEL_HOST}:8443",
            f"{TUNNEL}/api",
            f"{TUNNEL}?x=1",
            f"{TUNNEL}#frag",
            "https://trycloudflare.com",
            "https://a.b.trycloudflare.com",
            "https://evil.trycloudflare.com.attacker.net",
            "https://attacker.net",
            "https://10.0.0.5",
            "https://[::1]",
            "https://ollama.example.org",  # no COLAB_TUNNEL_HOST set
            "https://x" + "a" * 400 + ".trycloudflare.com",
        ]
        for url in refused:
            with self.subTest(url=url), self.assertRaises(protocol.TunnelURLError):
                protocol.check_tunnel_url(url)
        with self.assertRaises(protocol.TunnelURLError):
            protocol.check_tunnel_url("https://other.example.org", "ollama.example.org")

    def test_colab_models_come_from_the_model_profiles(self) -> None:
        self.assertEqual(
            protocol.colab_models(),
            {
                QWEN: "22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643",
                ROUTER: "73e026ff389b28464bb26f8978f3164851a76832f0d339032747e2c56f98ea29",
            },
        )
        self.assertEqual(protocol.chat_models(), [QWEN])
        self.assertEqual(protocol.router_models(), [ROUTER])
        self.assertEqual(protocol.CHAT_THINK_VALUES, [False, "low", "medium", "xhigh"])
        # From colab/versions.json: clef-flash's config "requires" 0.40.4.
        self.assertEqual(protocol.OLLAMA_MIN_VERSION, "0.40.4")

    def test_moved_tags_names_a_model_at_another_digest_for_display(self) -> None:
        verified = protocol.colab_models()
        other = "adfcdbb4" + "0" * 56

        self.assertEqual(protocol.moved_tags(verified), {})
        self.assertEqual(protocol.moved_tags({QWEN: verified[QWEN], ROUTER: other}), {ROUTER: (verified[ROUTER], other)})
        self.assertEqual(protocol.moved_tags({QWEN: other}), {QWEN: (verified[QWEN], other)})
        self.assertEqual(protocol.moved_tags({ROUTER: "sha256:" + verified[ROUTER].upper()}), {})
        # Nothing recorded is nothing moved; an unknown model is not ours.
        self.assertEqual(protocol.moved_tags({}), {})
        self.assertEqual(protocol.moved_tags({QWEN: "", "llama3:8b": other}), {})

    def test_the_ollama_version_rule(self) -> None:
        problem = protocol.ollama_version_problem
        # A session from before 2026-10-16 recorded no release: the minimum alone.
        for version in ("0.40.4", "0.40.5", "0.41.0", "1.0.0"):
            with self.subTest(version=version):
                self.assertEqual(problem(version), "")
        for found in ("0.40.3", "0.35.0", "0.41.0-rc1", "", None, 41, ["0.40.5"], "x" * 100):
            with self.subTest(found=found):
                self.assertIn("expected 0.40.4 or later", problem(found))
        self.assertLessEqual(len(problem("9" * 100)), 120)
        # Launched with a release: exactly it, a newer one is wrong too.
        self.assertEqual(problem("0.41.2", "0.41.2"), "")
        self.assertEqual(
            problem("0.41.3", "0.41.2"), "Ollama 0.41.3, expected 0.41.2 (the release this session was launched with)"
        )
        self.assertIn("expected 0.40.4 or later", problem("0.40.3", "0.40.3"))

    def test_off_gpu_names_a_router_not_wholly_in_vram_or_without_sizes(self) -> None:
        whole = {QWEN: on_gpu(QWEN), ROUTER: on_gpu(ROUTER)}
        size = whole[ROUTER]["size"]

        self.assertEqual(protocol.off_gpu(whole), {})
        self.assertEqual(protocol.off_gpu({**whole, ROUTER: on_gpu(ROUTER, size - 1)}), {ROUTER: (size - 1, size)})
        self.assertEqual(protocol.off_gpu({QWEN: on_gpu(QWEN)}), {ROUTER: (0, 0)})
        for broken in ({"size": 0, "size_vram": 0}, {"size": "10", "size_vram": "10"}, {"size": True}, None, "x"):
            with self.subTest(broken=broken):
                self.assertIn(ROUTER, protocol.off_gpu({**whole, ROUTER: broken}))
        # Only a router is judged here; the chat model's share is the probe's.
        self.assertEqual(protocol.off_gpu({**whole, QWEN: on_gpu(QWEN, 0)}), {})


class RegistrarTests(unittest.IsolatedAsyncioTestCase):
    def _registrar(self, status: str = BOOTING, tunnel: FakeTunnel | None = None, *, resolver=_public, **state):
        self.clock = Clock(_NOW)
        self.store = InMemoryColabStateStore(ColabState(status=status, since=_NOW - 300, nbh=NBH, **state))
        self.endpoint, self.tunnel = make_endpoint(tunnel=tunnel, store=self.store, clock=self.clock)
        return ColabRegistrar(self.endpoint, secret=SECRET, clock=self.clock, resolver=resolver)

    async def asyncTearDown(self) -> None:
        if hasattr(self, "endpoint"):
            await self.endpoint.close()

    async def test_a_signed_ready_with_a_passing_probe_is_ready(self) -> None:
        registrar = self._registrar()
        body = _body()

        result = await registrar.handle(body, _signed(body))

        self.assertEqual((result.status_code, result.body), (200, {"ok": True, "status": READY}))
        state = self.store.cached
        self.assertEqual((state.status, state.since, state.tunnel_url), (READY, _NOW, TUNNEL))
        self.assertEqual((state.ready_at, state.last_health_at, state.boot_step), (_NOW, _NOW, "ready"))
        self.assertEqual(state.versions, {"ollama": "0.40.5", "cloudflared": "2026.9.3"})
        self.assertEqual(state.gpu["name"], "NVIDIA A100-SXM4-40GB")
        self.assertEqual((state.digests, self.endpoint.moved_tags()), (protocol.colab_models(), {}))
        self.assertTrue(self.endpoint.is_ready())
        # The probe went through the tunnel with the session's bearer.
        self.assertEqual(
            sorted(set(self.tunnel.paths())),
            ["GET /api/ps", "GET /api/tags", "GET /api/version", "POST /api/show"],
        )
        bearer = protocol.derive_proxy_token(SECRET, NBH)
        self.assertTrue(all(r.headers["authorization"] == f"Bearer {bearer}" for r in self.tunnel.requests))
        self.assertEqual(self.tunnel.bodies("POST /api/show"), [{"model": QWEN}])

    async def test_ready_is_logged_with_the_actual_version_and_each_digest(self) -> None:
        registrar = self._registrar()
        body = _body()

        with self.assertLogs("linebot", "INFO") as logs:
            await registrar.handle(body, _signed(body))

        lines = [record.getMessage() for record in logs.records]
        self.assertIn("Colab session READY (Ollama 0.40.5, gpu=NVIDIA A100-SXM4-40GB)", lines)
        for model, digest in protocol.colab_models().items():
            self.assertIn(f"Colab session READY: {model} at digest {digest}, the verified one", lines)
        self.assertEqual([r for r in logs.records if r.levelname != "INFO"], [])

    async def test_the_probe_wants_the_release_the_session_was_launched_with(self) -> None:
        for launched, served, status in (
            ("0.41.2", "0.41.2", 200),
            ("0.41.2", "0.40.5", 503),
            ("0.40.5", "0.41.2", 503),
            # A state from before 2026-10-16: any release at the minimum or later.
            ("", "0.41.2", 200),
            ("", "0.40.3", 503),
        ):
            tunnel = FakeTunnel(**{"GET /api/version": (200, {"version": served})})
            registrar = self._registrar(tunnel=tunnel, ollama_version=launched)
            body = _body()
            with self.subTest(launched=launched, served=served), self.assertLogs("linebot", "INFO") as logs:
                result = await registrar.handle(body, _signed(body))
                self.assertEqual(result.status_code, status)
                if status == 200:
                    self.assertIn(f"Colab session READY (Ollama {served},", "\n".join(logs.output))
                    # The health check keeps wanting it.
                    self.assertTrue(await self.endpoint.check_health())
                else:
                    self.assertIn(f"Ollama {served}, expected", " ".join(result.body["problems"]))
            await self.endpoint.close()

    async def test_ready_is_accepted_while_assigning_booting_or_ready(self) -> None:
        for status in (ASSIGNING, BOOTING, READY):
            registrar = self._registrar(status)
            body = _body()
            with self.subTest(status=status):
                self.assertEqual((await registrar.handle(body, _signed(body))).status_code, 200)
                self.assertEqual(self.store.cached.status, READY)
            await self.endpoint.close()

    async def test_events_the_session_cannot_take_are_conflicts(self) -> None:
        for status in (IDLE, DRAINING, COOLDOWN, DISABLED):
            registrar = self._registrar(status)
            body = _body()
            with self.subTest(status=status):
                result = await registrar.handle(body, _signed(body))
                self.assertEqual(result.status_code, 409)
                self.assertEqual(self.store.cached.status, status)
                self.assertEqual(self.tunnel.requests, [])
            await self.endpoint.close()

    async def test_signature_problems_are_refused_before_anything_changes(self) -> None:
        body = _body()
        cases = {
            "missing": None,
            "garbage": "nope",
            "wrong secret": _signed(body, secret="t" * 40),
            "other session's key": _signed(body, nbh="another-nbh"),
            "other body": _signed(_body(tunnel_url="https://evil-1.trycloudflare.com")),
            "too old": _signed(body, at=_NOW - protocol.MAX_CLOCK_SKEW_SECONDS - 1),
            "from the future": _signed(body, at=_NOW + protocol.MAX_CLOCK_SKEW_SECONDS + 1),
        }
        for name, header in cases.items():
            registrar = self._registrar()
            with self.subTest(name):
                result = await registrar.handle(body, header)
                self.assertEqual(result.status_code, 403)
                self.assertEqual(self.store.cached.status if self.store.cached else BOOTING, BOOTING)
                self.assertEqual(self.store.writes, 0)
                self.assertEqual(self.tunnel.requests, [])
            await self.endpoint.close()

    async def test_an_unsigned_request_costs_no_state_read_and_no_warning(self) -> None:
        # The route is public: nothing but the signature may be checked first.
        body = _body("failed", nbh="4d1f7c52-1b0e-4a8e-9a53-6f0f2d9e7b11")
        forged = f"t={int(_NOW)},v1={'0' * 64}"
        for name, (payload, header) in {
            "forged, unknown nbh": (body, forged),
            "forged, current nbh": (_body("stopping"), forged),
            "other session's key": (_body("stopping"), _signed(_body("stopping"), nbh="another-nbh")),
        }.items():
            registrar = self._registrar(READY, tunnel_url=TUNNEL)
            loads = 0
            original = self.store.load

            async def counting(original=original):
                nonlocal loads
                loads += 1
                return await original()

            self.store.load = counting
            with self.subTest(name), self.assertNoLogs("linebot", "INFO"):
                result = await registrar.handle(payload, header)
                self.assertEqual((result.status_code, result.body["error"]), (403, "bad signature"))
                self.assertEqual((loads, self.store.writes), (0, 0))
            await self.endpoint.close()

    async def test_unsigned_noise_is_refused_quietly_never_a_crash(self) -> None:
        # Anyone can send a well-formed header; what the body holds must not
        # crash the check (a 500 and a traceback) before the signature.
        forged = f"t={int(_NOW)},v1={'0' * 64}"
        cases = {
            "no header, empty body": (b"", None, 403),
            "no header, garbage": (b"\xff\xfe garbage", None, 403),
            "garbage header": (b"garbage", "nope", 403),
            "forged, empty body": (b"", forged, 400),
            "forged, garbage": (b"\xff\xfe garbage", forged, 400),
            "forged, event a list": (json.dumps({"event": [], "nbh": NBH}).encode(), forged, 400),
            "forged, event an object": (json.dumps({"event": {}, "nbh": NBH}).encode(), forged, 400),
            "forged, nbh a lone surrogate": (b'{"event": "stopping", "nbh": "\\ud800"}', forged, 400),
            "forged, well-formed body": (_body("stopping"), forged, 403),
        }
        for name, (body, header, status) in cases.items():
            registrar = self._registrar(READY, tunnel_url=TUNNEL)
            with self.subTest(name), self.assertNoLogs("linebot", "INFO"):
                result = await registrar.handle(body, header)
                self.assertEqual(result.status_code, status)
                # Nothing read, nothing written.
                self.assertEqual((self.store.writes, self.store.cached), (0, None))
            await self.endpoint.close()

    async def test_a_state_read_failure_does_not_burn_the_signature(self) -> None:
        registrar = self._registrar(READY, tunnel_url=TUNNEL)
        original = self.store.load

        async def broken() -> ColabState:
            raise ColabStateError("firestore down")

        body = _body("stopping")
        header = _signed(body)
        self.store.load = broken
        self.assertEqual((await registrar.handle(body, header)).status_code, 503)
        # The VM's retry with the same signature is applied, once.
        self.store.load = original
        self.assertEqual((await registrar.handle(body, header)).status_code, 200)
        self.assertEqual(self.store.cached.status, DRAINING)
        self.assertEqual((await registrar.handle(body, header)).status_code, 403)

    async def test_a_non_ascii_digit_timestamp_is_refused_not_a_crash(self) -> None:
        registrar = self._registrar()
        body = _body()
        for header in (f"t=²,v1={'a' * 64}", f"t=١٧٩٠,v1={'a' * 64}"):
            with self.subTest(header=header):
                result = await registrar.handle(body, header)
                self.assertEqual((result.status_code, result.body["error"]), (403, "missing or malformed signature"))
        self.assertEqual(self.store.writes, 0)

    async def test_the_skew_limit_itself_is_accepted(self) -> None:
        registrar = self._registrar()
        body = _body()
        header = _signed(body, at=_NOW - protocol.MAX_CLOCK_SKEW_SECONDS)
        self.assertEqual((await registrar.handle(body, header)).status_code, 200)

    async def test_an_nbh_that_is_not_the_current_session_is_refused(self) -> None:
        registrar = self._registrar()
        body = _body(nbh="old-session")

        result = await registrar.handle(body, _signed(body, nbh="old-session"))

        self.assertEqual((result.status_code, result.body["error"]), (403, "unknown session"))
        self.assertEqual(self.store.writes, 0)

    async def test_a_replayed_signature_is_refused(self) -> None:
        registrar = self._registrar(READY, tunnel_url=TUNNEL)
        body = _body("stopping")
        header = _signed(body)

        self.assertEqual((await registrar.handle(body, header)).status_code, 200)
        self.assertEqual((await registrar.handle(body, header)).status_code, 403)

    async def test_bad_bodies(self) -> None:
        cases = {
            "not json": (b"{not json", 400),
            "a list": (b"[]", 400),
            "unknown event": (json.dumps({"event": "hello", "nbh": NBH}).encode(), 400),
            "no nbh": (json.dumps({"event": "ready"}).encode(), 400),
            "nbh not text": (json.dumps({"event": "ready", "nbh": 5}).encode(), 400),
            "too large": (b"{" + b" " * protocol.MAX_REGISTER_BYTES + b"}", 413),
        }
        for name, (body, status) in cases.items():
            registrar = self._registrar()
            with self.subTest(name):
                self.assertEqual((await registrar.handle(body, _signed(body))).status_code, status)
                self.assertEqual(self.store.writes, 0)
            await self.endpoint.close()

    async def test_tunnel_urls_off_the_rule_are_refused(self) -> None:
        for url in (
            "http://brave-otter-42.trycloudflare.com",
            "https://attacker.example",
            "https://brave-otter-42.trycloudflare.com:8443",
            "https://brave-otter-42.trycloudflare.com/api",
            "https://u:p@brave-otter-42.trycloudflare.com",
            None,
        ):
            registrar = self._registrar()
            body = _body(tunnel_url=url)
            with self.subTest(url=url):
                result = await registrar.handle(body, _signed(body))
                self.assertEqual(result.status_code, 422)
                self.assertNotEqual(self.store.cached.status, READY)
                self.assertIn("tunnel_url refused", self.store.cached.last_error)
                self.assertEqual(self.tunnel.requests, [])
            await self.endpoint.close()

    async def test_a_tunnel_host_resolving_inside_is_refused(self) -> None:
        for addresses in (
            ["10.0.0.7"],
            ["169.254.169.254"],
            ["127.0.0.1"],
            ["100.64.1.1"],
            ["::ffff:10.0.0.7"],
            ["104.16.230.132", "192.168.1.9"],
            [],
        ):

            async def resolver(host: str, port: int, addresses=addresses) -> list[str]:
                return addresses

            registrar = self._registrar(resolver=resolver)
            body = _body()
            with self.subTest(addresses=addresses):
                result = await registrar.handle(body, _signed(body))
                self.assertEqual(result.status_code, 422)
                self.assertEqual(self.store.cached.status, BOOTING)
                self.assertEqual(self.tunnel.requests, [])
            await self.endpoint.close()

    async def test_a_named_tunnel_host_is_accepted_when_configured(self) -> None:
        registrar = self._registrar()
        registrar = ColabRegistrar(
            self.endpoint, secret=SECRET, tunnel_host="ollama.example.org", clock=self.clock, resolver=_public
        )
        body = _body(tunnel_url="https://ollama.example.org")

        self.assertEqual((await registrar.handle(body, _signed(body))).status_code, 200)
        self.assertEqual(self.store.cached.tunnel_url, "https://ollama.example.org")

    async def test_a_failing_probe_keeps_booting_and_asks_to_retry(self) -> None:
        pins = protocol.colab_models()
        cases = {
            "old ollama": {"GET /api/version": (200, {"version": "0.34.4"})},
            # The one 2026-10-15's VM ran, which the registry no longer serves.
            "ollama under the minimum": {"GET /api/version": (200, {"version": "0.35.0"})},
            "a pre-release": {"GET /api/version": (200, {"version": "0.41.0-rc1"})},
            "chat not loaded": {"GET /api/ps": (200, {"models": [{"name": ROUTER, **on_gpu(ROUTER)}]})},
            "chat not wholly in VRAM": {
                "GET /api/ps": (
                    200,
                    {"models": [{"name": QWEN, **on_gpu(QWEN, 20 << 30)}, {"name": ROUTER, **on_gpu(ROUTER)}]},
                )
            },
            "chat sizes missing": {"GET /api/ps": (200, {"models": [{"name": QWEN}, {"name": ROUTER, **on_gpu(ROUTER)}]})},
            "chat not pulled": {"GET /api/tags": (200, {"models": [{"name": ROUTER, "digest": pins[ROUTER]}]})},
            "think values": {"POST /api/show": (200, {"thinking": {"values": [False, "low"]}})},
            "proxy refuses": {"GET /api/version": (401, {"error": "unauthorized"})},
            "not json": {"GET /api/ps": (200, b"<html>")},
            "hangs": {"GET /api/tags": ("hang", 30)},
        }
        for name, routes in cases.items():
            registrar = self._registrar(tunnel=FakeTunnel(**routes))
            body = _body()
            with self.subTest(name), patch("src.services.colab.protocol.PROBE_TIMEOUT_SECONDS", 0.2):
                result = await registrar.handle(body, _signed(body))
                self.assertEqual(result.status_code, 503)
                self.assertTrue(result.body["problems"])
                self.assertEqual(self.store.cached.status, BOOTING)
                self.assertIn("readiness probe", self.store.cached.last_error)
                self.assertFalse(self.endpoint.is_ready())
            await self.endpoint.close()

    async def test_moved_tags_are_ready_and_logged(self) -> None:
        # The models follow their tags (2026-10-16): another digest than the
        # verified one, the chat model's too, is news at INFO, never a refusal.
        verified = protocol.colab_models()
        other_chat, other_router = "0" * 64, "adfcdbb4070d303fb43fc11fc06a14269ec2bc7136a55369e86d412b86ba2831"
        tags = {"models": [{"name": QWEN, "digest": other_chat}, {"name": ROUTER, "digest": "sha256:" + other_router}]}
        registrar = self._registrar(tunnel=FakeTunnel(**{"GET /api/tags": (200, tags)}))
        # The session keeps what the probe saw, not what the VM says.
        body = _body(digests=verified)

        with self.assertLogs("linebot", "INFO") as logs:
            result = await registrar.handle(body, _signed(body))

        self.assertEqual((result.status_code, result.body), (200, {"ok": True, "status": READY}))
        self.assertEqual(self.store.cached.digests, {QWEN: other_chat, ROUTER: other_router})
        self.assertTrue(self.endpoint.is_ready())
        self.assertEqual(self.endpoint.off_gpu(), {})
        self.assertEqual(
            self.endpoint.moved_tags(), {QWEN: (verified[QWEN], other_chat), ROUTER: (verified[ROUTER], other_router)}
        )
        self.assertEqual(
            self.endpoint.view()["tags_moved"][ROUTER], {"verified": verified[ROUTER], "actual": other_router}
        )
        lines = [record.getMessage() for record in logs.records]
        self.assertIn(
            f"Colab session READY: {ROUTER} at digest {other_router}, tag moved since it was verified (73e026ff389b)",
            lines,
        )
        self.assertEqual([r for r in logs.records if r.levelname != "INFO"], [])
        # The live gate's probe finds nothing missing either.
        self.assertEqual(await self.endpoint.probe(TUNNEL, NBH), [])

    async def test_a_router_not_pulled_is_ready_without_it_and_says_why_once(self) -> None:
        pins = protocol.colab_models()
        routes = {
            "GET /api/ps": (200, {"models": [{"name": QWEN, **on_gpu(QWEN)}]}),
            "GET /api/tags": (200, {"models": [{"name": QWEN, "digest": pins[QWEN]}]}),
        }
        registrar = self._registrar(tunnel=FakeTunnel(**routes))
        # 2026-10-15: what the VM saw, which the bot's log never showed.
        why = f"pulling {ROUTER} failed 3 times: HTTP 412 pull model manifest: 412"
        body = _body(error=why, digests={QWEN: pins[QWEN]})

        with self.assertLogs("linebot", "INFO") as logs:
            result = await registrar.handle(body, _signed(body))

        self.assertEqual((result.status_code, result.body), (200, {"ok": True, "status": READY}))
        (error,) = [record.getMessage() for record in logs.records if record.levelname == "ERROR"]
        self.assertEqual(
            error,
            f"Colab session READY without its router: {ROUTER} not pulled (/api/tags does not list it); "
            f"the VM says: {why}; the API router decides this session (colab/README.md, 疑難排解)",
        )
        state = self.store.cached
        self.assertEqual((state.digests, state.vram), ({QWEN: pins[QWEN]}, {QWEN: on_gpu(QWEN)}))
        self.assertEqual(self.endpoint.off_gpu(), {ROUTER: (0, 0)})
        self.assertTrue(self.endpoint.is_ready())
        # The live gate's probe names it; READY and the health check do not
        # need it, and the health check does not warn again about it.
        self.assertEqual(await self.endpoint.probe(TUNNEL, NBH), [f"not listed: {ROUTER}", f"not loaded: {ROUTER}"])
        with self.assertNoLogs("linebot", "WARNING"):
            self.assertTrue(await self.endpoint.check_health())
            self.assertTrue(await self.endpoint.check_health())
        # Nor does the router when messages come: that ERROR was the one.
        router = SystemOneRouter(
            self.endpoint,
            Tier(SimpleNamespace(name="colab"), ROUTER, timeout_s=6),
            gates=RouterGates(confidence=0.90, think_on=0.85, think_off=0.15),
        )
        with self.assertNoLogs("linebot", "WARNING"):
            for _ in range(2):
                self.assertIsNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))
        self.assertNotIn("POST /v1/systemone", self.tunnel.paths())

    async def test_a_router_pulled_but_not_loaded_is_named_so(self) -> None:
        routes = {"GET /api/ps": (200, {"models": [{"name": QWEN, **on_gpu(QWEN)}]})}
        registrar = self._registrar(tunnel=FakeTunnel(**routes))
        why = f"{ROUTER} did not load: loading {ROUTER} failed: HTTP 500 llama runner process has terminated"
        body = _body(error=why)

        with self.assertLogs("linebot", "ERROR") as logs:
            self.assertEqual((await registrar.handle(body, _signed(body))).status_code, 200)

        (error,) = logs.output
        self.assertIn(f"{ROUTER} not loaded (/api/ps does not list it); the VM says: {why}", error)

    async def test_a_router_not_wholly_in_vram_is_ready_without_it(self) -> None:
        running = {"models": [{"name": QWEN, **on_gpu(QWEN)}, {"name": ROUTER, **on_gpu(ROUTER, 7 << 30)}]}
        registrar = self._registrar(tunnel=FakeTunnel(**{"GET /api/ps": (200, running)}))
        body = _body()

        with self.assertLogs("linebot", "ERROR") as logs:
            result = await registrar.handle(body, _signed(body))

        self.assertEqual((result.status_code, result.body), (200, {"ok": True, "status": READY}))
        # No word from the VM (its ready had no error): what the probe saw.
        (error,) = logs.output
        self.assertIn(f"without its router: {ROUTER} not wholly in VRAM (7168 of 10110 MiB); the API router", error)
        self.assertTrue(error.endswith("(colab/README.md, VRAM)"))
        state = self.store.cached
        self.assertEqual(state.vram, {QWEN: on_gpu(QWEN), ROUTER: on_gpu(ROUTER, 7 << 30)})
        self.assertEqual(state.digests, protocol.colab_models())
        self.assertEqual(self.endpoint.off_gpu(), {ROUTER: (7 << 30, on_gpu(ROUTER)["size"])})
        self.assertTrue(self.endpoint.is_ready())
        # What the VM reported is kept for display.
        self.assertEqual((state.gpu["router_on_gpu"], state.gpu["used_mib"]), ("10110/10110 MiB", 35012))
        # The live gate's probe names it; READY does not need it.
        self.assertEqual(await self.endpoint.probe(TUNNEL, NBH), [f"{ROUTER} not fully on the GPU (7168 of 10110 MiB in VRAM)"])
        # The per-minute health check judges only what is loaded, not where,
        # and says nothing more about a router READY already reported.
        with self.assertNoLogs("linebot", "WARNING"):
            self.assertTrue(await self.endpoint.check_health())

    async def test_a_ready_with_its_router_logs_no_error(self) -> None:
        registrar = self._registrar()
        body = _body()
        with self.assertNoLogs("linebot", "WARNING"):
            self.assertEqual((await registrar.handle(body, _signed(body))).status_code, 200)

    async def test_failed_drains_and_counts(self) -> None:
        registrar = self._registrar(BOOTING, failures=1)
        body = _body("failed", step="pull", error="digest mismatch for qwen3.8:27b-mtp-q4_K_M " + "x" * 500)

        result = await registrar.handle(body, _signed(body))

        self.assertEqual((result.status_code, result.body["status"]), (200, DRAINING))
        state = self.store.cached
        self.assertEqual((state.status, state.failures, state.boot_step), (DRAINING, 2, "pull"))
        self.assertTrue(state.last_error.startswith("VM bootstrap failed at pull: digest mismatch"))
        self.assertLessEqual(len(state.last_error), 300)

    async def test_a_failed_counts_as_a_boot_failure_only_while_assigning_or_booting(self) -> None:
        for status, counted in ((ASSIGNING, 2), (BOOTING, 2), (READY, 1)):
            registrar = self._registrar(status, failures=1)
            body = _body("failed", step="tunnel", error="cloudflared exited")
            with self.subTest(status=status):
                self.assertEqual((await registrar.handle(body, _signed(body))).status_code, 200)
                self.assertEqual((self.store.cached.status, self.store.cached.failures), (DRAINING, counted))
            await self.endpoint.close()

    async def test_events_on_a_draining_session_only_note_the_error(self) -> None:
        # !colab off drained it; the VM's ready then got 409 and it reports failed.
        for event in ("failed", "stopping"):
            registrar = self._registrar(DRAINING, failures=1, last_error="!colab off")
            body = _body(event, step="register", error="HTTP 409")
            with self.subTest(event=event):
                result = await registrar.handle(body, _signed(body))
                state = self.store.cached
                self.assertEqual((result.status_code, result.body["status"]), (200, DRAINING))
                self.assertEqual((state.failures, state.since), (1, _NOW - 300))
                self.assertIn("HTTP 409", state.last_error)
            await self.endpoint.close()

    async def test_a_ready_after_colab_off_drains_and_is_refused(self) -> None:
        for status in (ASSIGNING, BOOTING, READY):
            registrar = self._registrar(status, desired=OFF, tunnel_url=TUNNEL if status == READY else "")
            body = _body()
            with self.subTest(status=status):
                result = await registrar.handle(body, _signed(body))
                state = self.store.cached
                self.assertEqual(result.status_code, 409)
                self.assertEqual((state.status, state.desired, state.last_error), (DRAINING, OFF, "!colab off"))
                self.assertFalse(self.endpoint.is_ready())
                # Refused before the tunnel was probed.
                self.assertEqual(self.tunnel.requests, [])
            await self.endpoint.close()

    async def test_colab_off_between_the_probe_and_the_write_still_wins(self) -> None:
        registrar = self._registrar(BOOTING)
        store = self.store
        original = store.update
        switched = False

        async def update(change):
            nonlocal switched
            if not switched:
                switched = True
                # !colab off lands while the probe runs.
                await original(lambda s: replace(s, desired=OFF))
            return await original(change)

        store.update = update
        body = _body()

        result = await registrar.handle(body, _signed(body))

        self.assertEqual(result.status_code, 409)
        self.assertEqual((store.cached.status, store.cached.last_error), (DRAINING, "!colab off"))
        self.assertFalse(self.endpoint.is_ready())

    async def test_stopping_drains_a_ready_session_at_once(self) -> None:
        registrar = self._registrar(READY, tunnel_url=TUNNEL, last_health_at=_NOW)
        await self.endpoint.ready()
        self.assertTrue(self.endpoint.is_ready())
        body = _body("stopping", error="idle for 1020 s")

        result = await registrar.handle(body, _signed(body))

        self.assertEqual(result.status_code, 200)
        self.assertEqual((self.store.cached.status, self.store.cached.failures), (DRAINING, 0))
        self.assertIn("idle for 1020 s", self.store.cached.last_error)
        self.assertFalse(self.endpoint.is_ready())

    async def test_stopping_an_idle_session_changes_nothing(self) -> None:
        registrar = self._registrar(IDLE)
        body = _body("stopping")

        result = await registrar.handle(body, _signed(body))

        self.assertEqual((result.status_code, result.body["status"]), (200, IDLE))
        self.assertEqual(self.store.writes, 0)

    async def test_an_unreadable_state_asks_to_retry(self) -> None:
        registrar = self._registrar()

        async def broken() -> ColabState:
            raise ColabStateError("firestore down")

        self.store.load = broken
        body = _body()

        self.assertEqual((await registrar.handle(body, _signed(body))).status_code, 503)

    async def test_what_the_vm_reports_is_kept_small(self) -> None:
        registrar = self._registrar()
        body = _body(gpu={"name": "A" * 500, "vram_mib": 40960, "nested": {"x": 1}, "flag": True, **{f"k{i}": i for i in range(20)}})

        await registrar.handle(body, _signed(body))

        gpu = self.store.cached.gpu
        self.assertEqual(len(gpu["name"]), 100)
        self.assertNotIn("nested", gpu)
        self.assertNotIn("flag", gpu)
        self.assertLessEqual(len(gpu), 8)


class RegisterRouteTests(unittest.TestCase):
    @contextmanager
    def _client(self, registrar):
        container = SimpleNamespace(colab_registrar=registrar)
        with patch.object(main_module, "container", container):
            # Not entered as a context manager: no lifespan, so the container is ours.
            yield TestClient(main_module.app)

    def test_disabled_colab_has_no_route(self) -> None:
        with self._client(None) as client:
            self.assertEqual(client.post("/internal/colab/register", content=b"{}").status_code, 404)

    def test_the_route_hands_body_and_signature_to_the_registrar(self) -> None:
        calls = []

        class Registrar:
            async def handle(self, body, header):
                calls.append((body, header))
                return SimpleNamespace(status_code=202, body={"ok": True})

        with self._client(Registrar()) as client:
            response = client.post(
                "/internal/colab/register", content=b'{"event":"ready"}', headers={"X-Colab-Signature": "t=1,v1=ab"}
            )

        self.assertEqual((response.status_code, response.json()), (202, {"ok": True}))
        self.assertEqual(calls, [(b'{"event":"ready"}', "t=1,v1=ab")])

    def test_oversized_bodies_are_refused_before_the_registrar(self) -> None:
        class Registrar:
            async def handle(self, body, header):
                raise AssertionError("must not be called")

        with self._client(Registrar()) as client:
            response = client.post(
                "/internal/colab/register", content=b"x" * (protocol.MAX_REGISTER_BYTES + 1)
            )

        self.assertEqual(response.status_code, 413)

    def test_a_latin1_superscript_digit_in_the_header_is_a_403(self) -> None:
        # h11 lets obs-text bytes through and Starlette decodes them as latin-1.
        store = InMemoryColabStateStore(ColabState(status=BOOTING, nbh=NBH))
        endpoint, _ = make_endpoint(store=store, clock=Clock())
        registrar = ColabRegistrar(endpoint, secret=SECRET, resolver=_public)
        with self._client(registrar) as client:
            response = client.post(
                "/internal/colab/register",
                content=_body(),
                headers={"X-Colab-Signature": ("t=²,v1=" + "a" * 64).encode("latin-1")},
            )
        self.assertEqual(response.status_code, 403)

    def test_an_unsigned_post_with_an_odd_body_is_no_server_error(self) -> None:
        store = InMemoryColabStateStore(ColabState(status=BOOTING, nbh=NBH))
        endpoint, _ = make_endpoint(store=store, clock=Clock())
        registrar = ColabRegistrar(endpoint, secret=SECRET, resolver=_public)
        forged = f"t={int(Clock()())},v1={'0' * 64}"
        with self._client(registrar) as client:
            for body in (b'{"event": [], "nbh": "x"}', b'{"event": "failed", "nbh": "\\udfff"}'):
                with self.subTest(body=body):
                    response = client.post("/internal/colab/register", content=body, headers={"X-Colab-Signature": forged})
                    self.assertEqual(response.status_code, 400)
            self.assertEqual(client.post("/internal/colab/register", content=b"").status_code, 403)

    def test_a_signed_ready_through_the_app(self) -> None:
        clock = Clock()
        store = InMemoryColabStateStore(ColabState(status=BOOTING, nbh=NBH))
        endpoint, _ = make_endpoint(store=store, clock=clock)
        registrar = ColabRegistrar(endpoint, secret=SECRET, clock=clock, resolver=_public)
        body = _body()

        with self._client(registrar) as client:
            response = client.post(
                "/internal/colab/register",
                content=body,
                headers={"X-Colab-Signature": _signed(body, at=clock()), "Content-Type": "application/json"},
            )

        self.assertEqual((response.status_code, response.json()["status"]), (200, READY))
        self.assertEqual(store.cached.status, READY)


if __name__ == "__main__":
    unittest.main()
