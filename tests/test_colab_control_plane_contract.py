"""The Colab control plane against the google-colab-cli 0.7.4 wire contract.

The session backend (``colab.research.google.com/tun/m``) is a private API,
so these fixtures are its recorded shapes, built from the 0.7.4 client
source (``colab_cli/client.py``: GetAssignmentResponse, Assignment,
PostAssignmentResponse, ListedAssignments, GetUnassignRequest;
``colab_cli/consumption.py``: CcuInfo) and its tests. Every body carries the
XSSI prefix ``)]}'\\n``. A fake session plays the backend and the real
``colab_cli.client.Client`` talks to it, so a library upgrade that changes
the requests or the parsing fails here. The Contents API runs against a
local HTTP server; the kernel runtime is a fake with ColabRuntime's shape.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from src.services.colab import control_plane
from src.services.colab.control_plane import (
    CELL_DISCONNECTED,
    CELL_ERROR,
    CELL_OK,
    CELL_TIMEOUT,
    LIBRARY_LOGGERS,
    CcuInfo,
    ColabControlPlane,
    ControlPlaneError,
    ControlPlaneTimeoutError,
    CredentialOrPolicyError,
    NoCapacityError,
    RuntimeAssignment,
    TransientControlPlaneError,
    adc_file_problem,
    redact,
)

HAS_COLAB_CLI = importlib.util.find_spec("colab_cli") is not None
needs_colab_cli = unittest.skipUnless(
    HAS_COLAB_CLI, "google-colab-cli is not installed (see DEPLOY.md, 本機測試)"
)
HAS_REQUESTS = importlib.util.find_spec("requests") is not None
HAS_GOOGLE_AUTH = importlib.util.find_spec("google.auth") is not None

XSSI = ")]}'\n"
BACKEND = "https://colab.research.google.com"
NBH = "6f1c1f4e-0b8e-4c55-9f5a-2b4d6c8e0a13"
# uuid_to_web_safe_base64: '-' -> '_', right-padded with '.' to 44 characters.
NBH_PARAM = NBH.replace("-", "_") + "." * 8
ENDPOINT = "gpu-a100-s-2x7kq1v3pm9ad"
PROXY_URL = "https://8080-gpu-a100-s-2x7kq1v3pm9ad-a.us-west1-0.prod.colab.dev"
# Secrets that must never reach a log line or an error message.
PROXY_TOKEN = "proxytokSECRET7c1e"
ACCESS_TOKEN = "ya29.a0SECRETaccess"
XSRF_ASSIGN = "xsrfSECRETassign"
XSRF_UNASSIGN = "xsrfSECRETunassign"
CLIENT_SECRET = "d-SECRETclient"
REFRESH_TOKEN = "1//0gSECRETrefreshtokenabcdefghij"
SECRETS = (PROXY_TOKEN, ACCESS_TOKEN, XSRF_ASSIGN, XSRF_UNASSIGN, CLIENT_SECRET, REFRESH_TOKEN)


def proxy_info(token: str = PROXY_TOKEN, url: str = PROXY_URL) -> dict[str, Any]:
    return {"token": token, "tokenExpiresInSeconds": 3600, "url": url}


# GET /tun/m/assign for an nbh with no runtime yet: the XSRF token for the POST.
ASSIGN_GET_NEW = {"acc": "A100", "nbh": NBH_PARAM, "token": XSRF_ASSIGN, "variant": "GPU"}
# GET /tun/m/assign for an nbh that already has a runtime: no POST follows.
ASSIGN_GET_EXISTING = {"endpoint": ENDPOINT, "runtimeProxyInfo": proxy_info()}
# POST /tun/m/assign; variant is AssignmentVariant's int (GPU = 1).
ASSIGN_POST = {
    "accelerator": "A100",
    "endpoint": ENDPOINT,
    "runtimeProxyInfo": proxy_info(),
    "variant": 1,
}
# GET /tun/m/assignments; machineShape is Shape's int (STANDARD = 0).
ASSIGNMENTS = {
    "assignments": [
        {
            "accelerator": "A100",
            "endpoint": ENDPOINT,
            "variant": 1,
            "machineShape": 0,
            "runtimeProxyInfo": proxy_info(),
        },
        {
            "accelerator": "NONE",
            "endpoint": "m-s-browser1",
            "variant": 0,
            "machineShape": 1,
            "runtimeProxyInfo": proxy_info("otherSECRET", "https://8080-m-s-browser1.prod.colab.dev"),
        },
    ]
}
# GET /tun/m/unassign/<endpoint>: the XSRF token for the POST (which answers 204).
UNASSIGN_GET = {"token": XSRF_UNASSIGN}
# GET /tun/m/ccu-info: CcuInfo's fields plus the ones only the raw answer keeps.
CCU_INFO = {
    "currentBalance": 298.53,
    "consumptionRateHourly": 5.37,
    "assignmentsCount": 1,
    "eligibleGpus": ["T4", "L4", "A100"],
    "ineligibleGpus": ["H100"],
    "freeCcuQuotaInfo": {"remainingTokens": "0", "nextRefillTimestampSec": 1790000000},
}
ADC = {
    "type": "authorized_user",
    "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
    "client_secret": CLIENT_SECRET,
    "refresh_token": REFRESH_TOKEN,
}


@dataclass
class Call:
    method: str
    url: str
    headers: dict[str, str]
    params: dict[str, str]
    kwargs: dict[str, Any]

    @property
    def path(self) -> str:
        return urlsplit(self.url).path

    @property
    def query(self) -> dict[str, str]:
        found = {key: values[-1] for key, values in parse_qs(urlsplit(self.url).query).items()}
        return {**found, **self.params}


class FakeResponse:
    """What colab_cli's Client reads from a ``requests.Response``."""

    def __init__(self, status: int, body: Any, request_headers: dict[str, str]) -> None:
        self.status_code = status
        self.ok = status < 400
        self.reason = {204: "No Content", 412: "Precondition Failed"}.get(status, "Status")
        if isinstance(body, (dict, list)):
            body = XSSI + json.dumps(body)
        self.text = body
        self.headers = {"Content-Type": "application/json"}
        self.request = type("Request", (), {"headers": request_headers})()


class FakeBackend:
    """Plays the session backend: records requests and answers from ``routes``.

    A route's answer is ``(status, body)``, an exception to raise, or a
    callable returning either.
    """

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = dict(routes)
        self.calls: list[Call] = []

    def request(self, method: str, url: str, headers=None, params=None, **kwargs: Any) -> Any:
        call = Call(method, url, dict(headers or {}), dict(params or {}), kwargs)
        self.calls.append(call)
        answer = self.routes[(method, call.path)]
        if callable(answer):
            answer = answer()
        if isinstance(answer, BaseException):
            raise answer
        status, body = answer
        # What an AuthorizedSession adds; colab_cli logs it at DEBUG.
        return FakeResponse(status, body, {**call.headers, "Authorization": f"Bearer {ACCESS_TOKEN}"})


class _PlaneMixin:
    def make_plane(self, backend: FakeBackend, **kwargs: Any) -> ColabControlPlane:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.adc_path = Path(tmp.name) / "adc.json"
        self.adc_path.write_text(json.dumps(ADC), encoding="utf-8")
        self.credentials: list[Any] = []

        def session_factory(credentials: Any) -> FakeBackend:
            self.credentials.append(credentials)
            return backend

        return ColabControlPlane(self.adc_path, session_factory=session_factory, **kwargs)


def assignment(url: str = PROXY_URL, token: str = PROXY_TOKEN) -> RuntimeAssignment:
    return RuntimeAssignment(endpoint=ENDPOINT, proxy_url=url, proxy_token=token)


@needs_colab_cli
class SessionBackendContractTests(_PlaneMixin, unittest.IsolatedAsyncioTestCase):
    def assert_backend_call(self, call: Call, method: str, path: str, budget: float) -> None:
        self.assertEqual((call.method, call.url.split("?")[0]), (method, BACKEND + path))
        self.assertEqual(call.params, {"authuser": "0"})
        self.assertEqual(call.headers["Accept"], "application/json")
        self.assertEqual(call.headers["X-Colab-Client-Agent"], "colab-cli")
        connect, read = call.kwargs["timeout"]
        self.assertLessEqual(connect, control_plane.CONNECT_TIMEOUT_SECONDS)
        self.assertLessEqual(read, budget)
        self.assertGreater(read, budget - 5)
        self.assertLessEqual(call.kwargs["max_allowed_time"], budget)

    async def test_assign_asks_for_the_40gb_a100_in_two_steps(self) -> None:
        backend = FakeBackend({
            ("GET", "/tun/m/assign"): (200, ASSIGN_GET_NEW),
            ("POST", "/tun/m/assign"): (200, ASSIGN_POST),
        })
        result = await self.make_plane(backend).assign(NBH)

        self.assertEqual(
            result,
            RuntimeAssignment(
                endpoint=ENDPOINT,
                proxy_url=PROXY_URL,
                proxy_token=PROXY_TOKEN,
                token_expires_in=3600,
                accelerator="A100",
                variant="GPU",
                reused=False,
            ),
        )
        self.assertNotIn(PROXY_TOKEN, repr(result))
        get, post = backend.calls
        for call in (get, post):
            self.assert_backend_call(call, call.method, "/tun/m/assign", control_plane.ASSIGN_TIMEOUT_SECONDS)
            self.assertEqual(
                call.query,
                {"nbh": NBH_PARAM, "variant": "GPU", "accelerator": "A100", "authuser": "0"},
            )  # no shape=hm: never the 80 GB high-mem box
        self.assertEqual((get.method, post.method), ("GET", "POST"))
        self.assertNotIn("X-Goog-Colab-Token", get.headers)
        self.assertEqual(post.headers["X-Goog-Colab-Token"], XSRF_ASSIGN)

    async def test_the_same_nbh_again_returns_its_runtime_without_a_post(self) -> None:
        backend = FakeBackend({("GET", "/tun/m/assign"): (200, ASSIGN_GET_EXISTING)})
        result = await self.make_plane(backend).assign(NBH)

        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(result.reused)
        self.assertEqual((result.endpoint, result.proxy_token), (ENDPOINT, PROXY_TOKEN))

    async def test_a_cpu_only_assign_asks_for_the_default_runtime(self) -> None:
        backend = FakeBackend({
            ("GET", "/tun/m/assign"): (200, {**ASSIGN_GET_NEW, "acc": "NONE", "variant": "DEFAULT"}),
            ("POST", "/tun/m/assign"): (200, {**ASSIGN_POST, "accelerator": "NONE", "variant": 0}),
        })
        result = await self.make_plane(backend).assign(NBH, cpu_only=True)

        self.assertEqual((result.accelerator, result.variant), ("NONE", "DEFAULT"))
        self.assertEqual(backend.calls[0].query["variant"], "DEFAULT")
        self.assertEqual(backend.calls[0].query["accelerator"], "NONE")

    async def test_assign_failures_are_classified(self) -> None:
        import requests

        denylisted = json.dumps({"error": {"code": 403, "details": [{"reason": "DENYLISTED"}]}})
        cases = [
            ("POST", (412, "Precondition Failed"), NoCapacityError, 412),
            ("POST", (503, "Service Unavailable"), NoCapacityError, 503),
            ("GET", (503, "Service Unavailable"), NoCapacityError, 503),
            ("POST", (400, "Bad Request"), CredentialOrPolicyError, 400),
            ("GET", (401, "Unauthorized"), CredentialOrPolicyError, 401),
            ("POST", (403, denylisted), CredentialOrPolicyError, 403),
            ("POST", (500, "Internal"), TransientControlPlaneError, 500),
            ("POST", (429, "Too Many Requests"), TransientControlPlaneError, 429),
            ("POST", requests.exceptions.ReadTimeout("read timed out"), ControlPlaneTimeoutError, None),
            ("GET", requests.exceptions.ConnectionError("reset"), TransientControlPlaneError, None),
            # The answer's shape changed: not a known failure mode.
            ("POST", (200, {"accelerator": "A100", "variant": 1}), ControlPlaneError, None),
        ]
        for method, answer, expected, status in cases:
            with self.subTest(method=method, answer=repr(answer)[:40]):
                routes = {
                    ("GET", "/tun/m/assign"): (200, ASSIGN_GET_NEW),
                    ("POST", "/tun/m/assign"): (200, ASSIGN_POST),
                    (method, "/tun/m/assign"): answer,
                }
                with self.assertRaises(ControlPlaneError) as caught:
                    await self.make_plane(FakeBackend(routes)).assign(NBH)
                self.assertIs(type(caught.exception), expected)
                self.assertEqual(caught.exception.status, status)
        # A timed-out assign is retried with the same nbh: it must be transient.
        self.assertTrue(issubclass(ControlPlaneTimeoutError, TransientControlPlaneError))

    async def test_a_400_on_the_cpu_assign_is_not_an_entitlement_refusal(self) -> None:
        # colab_cli reads 400 as "no entitlement" only when an accelerator was
        # asked for; a dry run must not disable the tier over it.
        for method in ("GET", "POST"):
            with self.subTest(method=method):
                routes = {
                    ("GET", "/tun/m/assign"): (200, {**ASSIGN_GET_NEW, "acc": "NONE", "variant": "DEFAULT"}),
                    ("POST", "/tun/m/assign"): (200, ASSIGN_POST),
                    (method, "/tun/m/assign"): (400, "Bad Request"),
                }
                with self.assertRaises(ControlPlaneError) as caught:
                    await self.make_plane(FakeBackend(routes)).assign(NBH, cpu_only=True)
                self.assertIs(type(caught.exception), ControlPlaneError)
                self.assertEqual(caught.exception.status, 400)
                self.assertNotIn("entitlement", str(caught.exception))

    async def test_find_assignment_only_looks(self) -> None:
        found = FakeBackend({("GET", "/tun/m/assign"): (200, ASSIGN_GET_EXISTING)})
        result = await self.make_plane(found).find_assignment(NBH)
        self.assertEqual((result.endpoint, result.proxy_token, result.reused), (ENDPOINT, PROXY_TOKEN, True))
        [call] = found.calls
        self.assert_backend_call(call, "GET", "/tun/m/assign", control_plane.LIST_TIMEOUT_SECONDS)
        self.assertEqual(call.query, {"nbh": NBH_PARAM, "variant": "GPU", "accelerator": "A100", "authuser": "0"})

        # No runtime for the nbh: the GET hands out the XSRF token, and no POST follows.
        none = FakeBackend({("GET", "/tun/m/assign"): (200, ASSIGN_GET_NEW)})
        self.assertIsNone(await self.make_plane(none).find_assignment(NBH))
        self.assertEqual([call.method for call in none.calls], ["GET"])
        cpu = FakeBackend({("GET", "/tun/m/assign"): (200, {**ASSIGN_GET_NEW, "acc": "NONE", "variant": "DEFAULT"})})
        self.assertIsNone(await self.make_plane(cpu).find_assignment(NBH, cpu_only=True))
        self.assertEqual((cpu.calls[0].query["variant"], cpu.calls[0].query["accelerator"]), ("DEFAULT", "NONE"))

        for answer, expected in (
            ((401, "Unauthorized"), CredentialOrPolicyError),
            ((502, "Bad Gateway"), TransientControlPlaneError),
        ):
            with self.subTest(answer=answer), self.assertRaises(expected):
                await self.make_plane(FakeBackend({("GET", "/tun/m/assign"): answer})).find_assignment(NBH)

    @unittest.skipUnless(HAS_GOOGLE_AUTH, "google-auth is not installed")
    async def test_a_dead_refresh_token_is_a_credential_error(self) -> None:
        from google.auth import exceptions as gexc

        cases = [
            (gexc.RefreshError("invalid_grant: Token has been expired or revoked."), CredentialOrPolicyError),
            (
                gexc.RefreshError(
                    "Reauthentication is needed. Please run `gcloud auth application-default "
                    "login` to reauthenticate."
                ),
                CredentialOrPolicyError,
            ),
            (gexc.RefreshError("temporarily unavailable", retryable=True), TransientControlPlaneError),
            (gexc.TransportError("connection reset"), TransientControlPlaneError),
        ]
        for error, expected in cases:
            with self.subTest(error=str(error)):
                backend = FakeBackend({("GET", "/tun/m/assignments"): error})
                with self.assertRaises(ControlPlaneError) as caught:
                    await self.make_plane(backend).list_assignments()
                self.assertIs(type(caught.exception), expected)

    @unittest.skipUnless(HAS_GOOGLE_AUTH, "google-auth is not installed")
    async def test_only_a_refresh_that_says_the_token_is_dead_disables(self) -> None:
        # google-auth calls all of these non-retryable, a token-endpoint blip
        # (a 502, an HTML page, a 200 without a token) included: only the
        # answer itself tells a dead credential from a blip.
        from google.auth import exceptions as gexc
        from google.oauth2.credentials import Credentials

        def refresh_error(status: int, body: str) -> Exception:
            answer = type("Answer", (), {"status": status, "data": body.encode(), "headers": {}})()
            try:
                Credentials.from_authorized_user_info(ADC).refresh(lambda **kwargs: answer)
            except gexc.RefreshError as exc:
                return exc
            raise AssertionError("the refresh did not fail")

        cases = [
            (502, "<html><title>502 Bad Gateway</title></html>", TransientControlPlaneError),
            (200, json.dumps({"expires_in": 3599}), TransientControlPlaneError),
            (400, json.dumps({"error": "invalid_grant", "error_description": "Token has been expired or revoked."}), CredentialOrPolicyError),
            (400, json.dumps({"error": "invalid_grant", "error_subtype": "invalid_rapt"}), CredentialOrPolicyError),
            (401, json.dumps({"error": "invalid_client", "error_description": "Unauthorized"}), CredentialOrPolicyError),
        ]
        for status, body, expected in cases:
            with self.subTest(status=status, body=body[:40]):
                error = refresh_error(status, body)
                self.assertFalse(error.retryable)
                backend = FakeBackend({("GET", "/tun/m/assignments"): error})
                with self.assertRaises(ControlPlaneError) as caught:
                    await self.make_plane(backend).list_assignments()
                self.assertIs(type(caught.exception), expected)

    async def test_list_assignments_returns_every_runtime_with_a_fresh_token(self) -> None:
        backend = FakeBackend({("GET", "/tun/m/assignments"): (200, ASSIGNMENTS)})
        runtimes = await self.make_plane(backend).list_assignments()

        (call,) = backend.calls
        self.assert_backend_call(call, "GET", "/tun/m/assignments", control_plane.LIST_TIMEOUT_SECONDS)
        self.assertEqual(
            runtimes,
            [
                RuntimeAssignment(ENDPOINT, PROXY_URL, PROXY_TOKEN, 3600, "A100", "GPU", "STANDARD"),
                RuntimeAssignment(
                    "m-s-browser1",
                    "https://8080-m-s-browser1.prod.colab.dev",
                    "otherSECRET",
                    3600,
                    "NONE",
                    "DEFAULT",
                    "HIGH_RAM",
                ),
            ],
        )

    async def test_a_runtime_of_a_kind_the_library_does_not_know_is_listed(self) -> None:
        # The owner's browser runtime on a shared account: an accelerator,
        # variant and shape 0.7.4 has no enum value for, or no shape at all.
        body = json.loads(json.dumps(ASSIGNMENTS))
        body["assignments"][1].update(accelerator="H200", variant=3)
        del body["assignments"][1]["machineShape"]
        backend = FakeBackend({("GET", "/tun/m/assignments"): (200, body)})

        ours, browser = await self.make_plane(backend).list_assignments()

        self.assertEqual(ours, RuntimeAssignment(ENDPOINT, PROXY_URL, PROXY_TOKEN, 3600, "A100", "GPU", "STANDARD"))
        self.assertEqual((browser.endpoint, browser.accelerator, browser.variant, browser.machine_shape),
                         ("m-s-browser1", "H200", "3", ""))

    async def test_list_failures_are_classified(self) -> None:
        for answer, expected in [
            ((401, "Unauthorized"), CredentialOrPolicyError),
            ((403, "Forbidden"), CredentialOrPolicyError),
            # Only assign turns 503 into "no capacity".
            ((503, "Service Unavailable"), TransientControlPlaneError),
            ((200, {"assignments": [{"endpoint": ENDPOINT}]}), ControlPlaneError),
            ((200, {"assignments": [{"runtimeProxyInfo": proxy_info()}]}), ControlPlaneError),
            ((200, {}), ControlPlaneError),
        ]:
            with self.subTest(answer=answer):
                backend = FakeBackend({("GET", "/tun/m/assignments"): answer})
                with self.assertRaises(ControlPlaneError) as caught:
                    await self.make_plane(backend).list_assignments()
                self.assertIs(type(caught.exception), expected)
                self.assertNotIn(PROXY_TOKEN, str(caught.exception))

    async def test_unassign_posts_with_the_xsrf_token(self) -> None:
        path = f"/tun/m/unassign/{ENDPOINT}"
        backend = FakeBackend({("GET", path): (200, UNASSIGN_GET), ("POST", path): (204, "")})

        self.assertTrue(await self.make_plane(backend).unassign(ENDPOINT))
        get, post = backend.calls
        self.assert_backend_call(get, "GET", path, control_plane.UNASSIGN_TIMEOUT_SECONDS)
        self.assert_backend_call(post, "POST", path, control_plane.UNASSIGN_TIMEOUT_SECONDS)
        self.assertEqual(post.headers["X-Goog-Colab-Token"], XSRF_UNASSIGN)

    async def test_unassign_of_a_released_runtime_is_not_an_error(self) -> None:
        path = f"/tun/m/unassign/{ENDPOINT}"
        backend = FakeBackend({("GET", path): (404, "Not Found")})

        self.assertFalse(await self.make_plane(backend).unassign(ENDPOINT))
        self.assertEqual(len(backend.calls), 1)

    async def test_unassign_failures_are_classified(self) -> None:
        path = f"/tun/m/unassign/{ENDPOINT}"
        for answer, expected in [
            ((500, "Internal"), TransientControlPlaneError),
            ((401, "Unauthorized"), CredentialOrPolicyError),
        ]:
            with self.subTest(answer=answer):
                backend = FakeBackend({("GET", path): (200, UNASSIGN_GET), ("POST", path): answer})
                with self.assertRaises(ControlPlaneError) as caught:
                    await self.make_plane(backend).unassign(ENDPOINT)
                self.assertIs(type(caught.exception), expected)

    async def test_ccu_info_keeps_the_fields_the_library_drops(self) -> None:
        backend = FakeBackend({("GET", "/tun/m/ccu-info"): (200, CCU_INFO)})
        info = await self.make_plane(backend).ccu_info()

        (call,) = backend.calls
        self.assert_backend_call(call, "GET", "/tun/m/ccu-info", control_plane.CCU_TIMEOUT_SECONDS)
        self.assertEqual(
            info,
            CcuInfo(
                current_balance=298.53,
                consumption_rate_hourly=5.37,
                assignments_count=1,
                eligible_gpus=("T4", "L4", "A100"),
                ineligible_gpus=("H100",),
            ),
        )
        self.assertTrue(info.gpu_eligible("A100"))
        self.assertFalse(info.gpu_eligible("H100"))

    async def test_ccu_info_without_gpu_lists_says_nothing_about_eligibility(self) -> None:
        body = {"currentBalance": 12.5, "consumptionRateHourly": 0.0, "assignmentsCount": 0}
        backend = FakeBackend({("GET", "/tun/m/ccu-info"): (200, body)})
        info = await self.make_plane(backend).ccu_info()

        self.assertEqual((info.current_balance, info.consumption_rate_hourly), (12.5, 0.0))
        self.assertIsNone(info.gpu_eligible("A100"))

    async def test_credentials_come_from_the_adc_file_read_once(self) -> None:
        backend = FakeBackend({("GET", "/tun/m/ccu-info"): (200, CCU_INFO)})
        plane = self.make_plane(backend)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            await plane.ccu_info()
            self.adc_path.unlink()
            await plane.ccu_info()
            self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", os.environ)

        (credentials,) = self.credentials
        self.assertEqual(credentials.client_id, ADC["client_id"])
        self.assertEqual(credentials.refresh_token, REFRESH_TOKEN)
        self.assertEqual(len(backend.calls), 2)

    async def test_an_unusable_adc_file_is_a_credential_error(self) -> None:
        backend = FakeBackend({("GET", "/tun/m/ccu-info"): (200, CCU_INFO)})
        for content in (None, "{not json", json.dumps({**ADC, "type": "service_account"}), json.dumps({**ADC, "refresh_token": ""})):
            with self.subTest(content=content):
                plane = self.make_plane(backend)
                if content is None:
                    self.adc_path.unlink()
                else:
                    self.adc_path.write_text(content, encoding="utf-8")
                with self.assertRaises(CredentialOrPolicyError) as caught:
                    await plane.ccu_info()
                self.assertEqual(self.credentials, [])
                for secret in SECRETS:
                    self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(backend.calls, [])

    async def test_a_hung_call_gives_up_after_its_budget(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)

        def hang() -> tuple[int, Any]:
            release.wait(5)
            return 200, ASSIGNMENTS

        plane = self.make_plane(FakeBackend({("GET", "/tun/m/assignments"): hang}))
        started = time.monotonic()
        with (
            patch.object(control_plane, "LIST_TIMEOUT_SECONDS", 0.2),
            patch.object(control_plane, "CALL_GRACE_SECONDS", 0.1),
            self.assertRaises(ControlPlaneTimeoutError),
        ):
            await plane.list_assignments()
        self.assertLess(time.monotonic() - started, 2.0)

    async def test_a_spent_budget_sends_no_further_request(self) -> None:
        def slow_get() -> tuple[int, Any]:
            time.sleep(0.3)
            return 200, ASSIGN_GET_NEW

        backend = FakeBackend({
            ("GET", "/tun/m/assign"): slow_get,
            ("POST", "/tun/m/assign"): (200, ASSIGN_POST),
        })
        with (
            patch.object(control_plane, "ASSIGN_TIMEOUT_SECONDS", 0.2),
            self.assertRaises(ControlPlaneTimeoutError),
        ):
            await self.make_plane(backend).assign(NBH)
        self.assertEqual([call.method for call in backend.calls], ["GET"])


class FakeKernel:
    """``ColabRuntime.kernel_client``: a jupyter-kernel-client KernelClient."""

    def __init__(self, kernel_id: str = "k-new", reply: Any = None, error: BaseException | None = None) -> None:
        self.id = kernel_id
        self.reply = reply if reply is not None else {"status": "ok", "outputs": []}
        self.error = error
        self.executed: list[dict[str, Any]] = []
        # Set: execute() spins like jupyter-kernel-client 0.9.0 after its
        # deadline (google-colab-cli#82) until the websocket is closed.
        self.spin: threading.Event | None = None

    def execute(self, code: str, **kwargs: Any) -> Any:
        self.executed.append({"code": code, **kwargs})
        if self.spin is not None:
            self.spin.wait(5)
            raise RuntimeError("Connection was lost.")
        if self.error is not None:
            raise self.error
        return self.reply


class FakeRuntime:
    """``colab_cli.runtime.ColabRuntime``, as the control plane uses it."""

    def __init__(self, url: str, token: str, kernel_id: str | None, session_id: str | None, kernel: FakeKernel, connect_error: BaseException | None = None) -> None:
        self.url, self.token = url, token
        self.given = (kernel_id, session_id)
        self.kernel_id = kernel_id
        self.session_id = session_id
        self.kernel = kernel
        self.connect_error = connect_error
        self.stops = 0

    @property
    def kernel_client(self) -> FakeKernel:
        if self.connect_error is not None:
            raise self.connect_error
        # Like ColabRuntime: ids are captured only when none was given.
        self.kernel_id = self.kernel_id or self.kernel.id
        self.session_id = self.session_id or "session-new"
        return self.kernel

    def stop(self) -> None:
        self.stops += 1
        if self.kernel.spin is not None:
            self.kernel.spin.set()


class RunCellTests(unittest.IsolatedAsyncioTestCase):
    def make_plane(self, kernel: FakeKernel | None = None, connect_error: BaseException | None = None) -> ColabControlPlane:
        self.kernel = kernel or FakeKernel()
        self.runtimes: list[FakeRuntime] = []

        def factory(url: str, token: str, kernel_id: str | None, session_id: str | None) -> FakeRuntime:
            runtime = FakeRuntime(url, token, kernel_id, session_id, self.kernel, connect_error)
            self.runtimes.append(runtime)
            return runtime

        return ColabControlPlane("/nonexistent/adc.json", runtime_factory=factory)

    async def test_a_cell_runs_in_the_saved_kernel(self) -> None:
        reply = {
            "status": "ok",
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": "BOOTSTRAP_LAUNCHED 4242\n"},
                {"output_type": "stream", "name": "stderr", "text": "warning\n"},
                {"output_type": "execute_result", "data": {"text/plain": "7"}},
            ],
        }
        plane = self.make_plane(FakeKernel("k-1", reply))
        result = await plane.run_cell(assignment(), "print(1)", budget_s=30, kernel_id="k-1", session_id="s-1")

        self.assertEqual(result.status, CELL_OK)
        self.assertTrue(result.ok)
        self.assertEqual((result.kernel_id, result.session_id), ("k-1", "s-1"))
        self.assertEqual(result.stdout, "BOOTSTRAP_LAUNCHED 4242\n7\n")
        self.assertEqual(result.stderr, "warning\n")
        (runtime,) = self.runtimes
        self.assertEqual((runtime.url, runtime.token), (PROXY_URL, PROXY_TOKEN))
        self.assertEqual(runtime.stops, 1)
        (execute,) = self.kernel.executed
        self.assertEqual((execute["silent"], execute["store_history"]), (False, True))
        self.assertLessEqual(execute["timeout"], 30)

    async def test_without_a_kernel_id_the_new_kernel_is_reported(self) -> None:
        plane = self.make_plane(FakeKernel("k-new"))
        result = await plane.run_cell(assignment(), "1", budget_s=30)

        self.assertEqual(self.runtimes[0].given, (None, None))
        self.assertEqual((result.kernel_id, result.session_id), ("k-new", "session-new"))

    async def test_a_vanished_kernel_reports_the_one_actually_used(self) -> None:
        # The saved kernel is gone: the client starts another, while
        # ColabRuntime keeps the id it was given.
        plane = self.make_plane(FakeKernel("k-replacement"))
        result = await plane.run_cell(assignment(), "1", budget_s=30, kernel_id="k-gone")

        self.assertEqual(self.runtimes[0].kernel_id, "k-gone")
        self.assertEqual(result.kernel_id, "k-replacement")

    async def test_a_silent_cell_keeps_no_history(self) -> None:
        plane = self.make_plane()
        await plane.run_cell(assignment(), "write the env file", budget_s=30, silent=True)

        (execute,) = self.kernel.executed
        self.assertEqual((execute["silent"], execute["store_history"]), (True, False))

    async def test_a_cell_that_raises_is_an_error_result(self) -> None:
        reply = {
            "status": "error",
            "outputs": [{"output_type": "error", "ename": "FileNotFoundError", "evalue": "/content/x", "traceback": []}],
        }
        plane = self.make_plane(FakeKernel("k-1", reply))
        result = await plane.run_cell(assignment(), "open('/content/x')", budget_s=30)

        self.assertEqual(result.status, CELL_ERROR)
        self.assertEqual(result.error, "FileNotFoundError: /content/x")
        self.assertEqual(self.runtimes[0].stops, 1)

    async def test_an_error_status_without_an_error_output_is_an_error(self) -> None:
        plane = self.make_plane(FakeKernel("k-1", {"status": "aborted", "outputs": []}))
        result = await plane.run_cell(assignment(), "1", budget_s=30)

        self.assertEqual(result.status, CELL_ERROR)
        self.assertIn("aborted", result.error)

    async def test_a_cell_that_outlives_its_budget_is_cut_by_the_timer(self) -> None:
        kernel = FakeKernel("k-1")
        kernel.spin = threading.Event()
        plane = self.make_plane(kernel)
        started = time.monotonic()
        result = await plane.run_cell(assignment(), "while True: print(1)", budget_s=0.2, kernel_id="k-1")

        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(result.status, CELL_TIMEOUT)
        self.assertEqual(result.kernel_id, "k-1")
        # The timer's stop, then the one in finally.
        self.assertEqual(self.runtimes[0].stops, 2)

    async def test_a_dropped_websocket_is_reported(self) -> None:
        plane = self.make_plane(FakeKernel("k-1", error=RuntimeError("Connection was lost.")))
        result = await plane.run_cell(assignment(), "1", budget_s=30)

        self.assertEqual(result.status, CELL_DISCONNECTED)
        self.assertEqual(self.runtimes[0].stops, 1)

    async def test_no_reply_in_time_is_a_timeout(self) -> None:
        plane = self.make_plane(FakeKernel("k-1", error=TimeoutError("Timeout waiting for reply")))
        result = await plane.run_cell(assignment(), "1", budget_s=30)

        self.assertEqual(result.status, CELL_TIMEOUT)
        self.assertEqual(result.kernel_id, "k-1")

    async def test_output_is_clipped_to_its_end(self) -> None:
        text = "x" * control_plane.OUTPUT_LIMIT + "END\n"
        reply = {"status": "ok", "outputs": [{"output_type": "stream", "name": "stdout", "text": text}]}
        result = await self.make_plane(FakeKernel("k", reply)).run_cell(assignment(), "1", budget_s=30)

        self.assertEqual(len(result.stdout), control_plane.OUTPUT_LIMIT)
        self.assertTrue(result.stdout.endswith("END\n"))

    @unittest.skipUnless(HAS_REQUESTS, "requests is not installed")
    async def test_failing_to_reach_the_kernel_raises_and_still_stops(self) -> None:
        import requests

        refused = requests.Response()
        refused.status_code = 403
        cases = [
            (requests.HTTPError("403 Client Error: Forbidden for url: x", response=refused), TransientControlPlaneError),
            (requests.ConnectionError("proxy unreachable"), TransientControlPlaneError),
            (requests.ReadTimeout("kernel start"), ControlPlaneTimeoutError),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                plane = self.make_plane(connect_error=error)
                with self.assertRaises(ControlPlaneError) as caught:
                    await plane.run_cell(assignment(), "1", budget_s=30)
                self.assertIs(type(caught.exception), expected)
                self.assertEqual(self.runtimes[0].stops, 1)

    async def test_the_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            await self.make_plane().run_cell(assignment(), "1", budget_s=0)


class _ContentsServer:
    """A local Jupyter Contents API: records each request, answers ``status``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 201
        self.delay = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_PUT(self) -> None:  # noqa: N802 - http.server's naming
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({
                    "path": urlsplit(self.path).path,
                    "query": {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()},
                    "headers": dict(self.headers),
                    "json": json.loads(body or b"null"),
                })
                if outer.delay:
                    time.sleep(outer.delay)
                payload = json.dumps({"name": "x", "type": "file"}).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        # A stalled request's client is gone by the time it answers.
        self.server.handle_error = lambda *args: None
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@needs_colab_cli
class UploadContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.server = _ContentsServer()
        self.addCleanup(self.server.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "bootstrap.py").write_bytes(b"print('boot')\n")
        (self.dir / "versions.json").write_bytes(b'{"ollama": "0.40.5"}')
        self.plane = ColabControlPlane(self.dir / "adc.json")
        self.runtime = assignment(url=self.server.url)

    async def test_each_file_is_put_through_the_contents_api(self) -> None:
        await self.plane.upload(
            self.runtime,
            {
                "content/bootstrap.py": self.dir / "bootstrap.py",
                "content/versions.json": str(self.dir / "versions.json"),
            },
        )

        first, second = self.server.requests
        self.assertEqual(first["path"], "/api/contents/content/bootstrap.py")
        self.assertEqual(first["query"], {"authuser": "0", "colab-runtime-proxy-token": PROXY_TOKEN})
        self.assertEqual(
            first["json"],
            {
                "name": "bootstrap.py",
                "path": "content/bootstrap.py",
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(b"print('boot')\n").decode("ascii"),
                "chunk": 1,
            },
        )
        self.assertEqual(second["path"], "/api/contents/content/versions.json")
        for request in (first, second):
            # The owner's Google token never goes to the runtime.
            self.assertNotIn("Authorization", request["headers"])

    async def test_upload_failures_are_classified(self) -> None:
        for status, expected in [
            (403, TransientControlPlaneError),
            (502, TransientControlPlaneError),
            (404, ControlPlaneError),
        ]:
            with self.subTest(status=status):
                self.server.status = status
                with self.assertRaises(ControlPlaneError) as caught:
                    await self.plane.upload(self.runtime, {"content/bootstrap.py": self.dir / "bootstrap.py"})
                self.assertIs(type(caught.exception), expected)
                self.assertNotIn(PROXY_TOKEN, str(caught.exception))

    async def test_a_stalled_upload_times_out(self) -> None:
        self.server.delay = 1.0
        with (
            patch.object(control_plane, "UPLOAD_TIMEOUT_SECONDS", 0.3),
            self.assertRaises(ControlPlaneTimeoutError),
        ):
            await self.plane.upload(self.runtime, {"content/bootstrap.py": self.dir / "bootstrap.py"})


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@needs_colab_cli
class LogScrubTests(_PlaneMixin, unittest.IsolatedAsyncioTestCase):
    """Nothing a control-plane call logs, at any level, carries a token."""

    def setUp(self) -> None:
        self.capture = _Capture()
        # Both outputs: the app's logger does not propagate to the root one
        # (src/utils/logger.py), where the libraries' records end up.
        # Everything at DEBUG except what the control plane itself holds back
        # (the app's logger may have been set higher by an earlier test).
        for name in ("", "linebot"):
            configured = logging.getLogger(name)
            configured.addHandler(self.capture)
            self.addCleanup(configured.removeHandler, self.capture)
            self.addCleanup(configured.setLevel, configured.level)
            configured.setLevel(logging.DEBUG)

    async def test_no_token_reaches_the_logs(self) -> None:
        import requests

        token_url = f"{PROXY_URL}/api/contents/x?authuser=0&colab-runtime-proxy-token={PROXY_TOKEN}"
        unassign = f"/tun/m/unassign/{ENDPOINT}"
        backend = FakeBackend({
            ("GET", "/tun/m/assign"): (200, ASSIGN_GET_NEW),
            ("POST", "/tun/m/assign"): (200, ASSIGN_POST),
            ("GET", "/tun/m/assignments"): (200, ASSIGNMENTS),
            ("GET", "/tun/m/ccu-info"): (200, CCU_INFO),
            ("GET", unassign): (200, UNASSIGN_GET),
            ("POST", unassign): (204, ""),
        })
        plane = self.make_plane(backend)
        await plane.assign(NBH)
        await plane.list_assignments()
        await plane.ccu_info()
        await plane.unassign(ENDPOINT)

        failures = [
            (403, json.dumps({"runtimeProxyInfo": {"token": PROXY_TOKEN}, "reason": "DENYLISTED"})),
            (500, f"upstream said Authorization: Bearer {ACCESS_TOKEN}"),
            requests.exceptions.ReadTimeout(f"HTTPSConnectionPool: Read timed out. (url: {token_url})"),
        ]
        for failure in failures:
            backend.routes[("POST", "/tun/m/assign")] = failure
            with self.assertRaises(ControlPlaneError):
                await plane.assign(NBH)

        server = _ContentsServer()
        self.addCleanup(server.close)
        source = self.adc_path.with_name("bootstrap.py")
        source.write_text("print(1)\n", encoding="utf-8")
        await plane.upload(assignment(url=server.url), {"content/bootstrap.py": source})
        server.status = 403
        with self.assertRaises(ControlPlaneError):
            await plane.upload(assignment(url=server.url), {"content/bootstrap.py": source})

        ws_url = f"wss://proxy/api/kernels/k/channels?session_id=s&token={PROXY_TOKEN}"
        kernel = FakeKernel("k-1")

        def noisy_factory(*args: Any) -> FakeRuntime:
            # What the libraries log when the kernel websocket fails.
            logging.getLogger("jupyter_kernel_client").error(
                "Unable to open websocket connection with %s",
                ws_url,
                exc_info=RuntimeError(f"Handshake failed: {ws_url}"),
            )
            logging.getLogger("websocket").error(f"error from callback: Bearer {ACCESS_TOKEN}")
            logging.getLogger("urllib3.connectionpool").warning(
                "Retrying (%r) after connection broken by '%r': %s", 0, "reset", token_url
            )
            kernel.error = RuntimeError(f"Connection was lost. ({ws_url})")
            return FakeRuntime(*args, kernel=kernel)

        plane_ws = ColabControlPlane(self.adc_path, runtime_factory=noisy_factory)
        result = await plane_ws.run_cell(assignment(), "1", budget_s=5)
        self.assertEqual(result.status, CELL_DISCONNECTED)
        self.assertNotIn(PROXY_TOKEN, result.error)

        logged = "\n".join(self.capture.lines)
        # The capture works: the failures above were logged.
        self.assertIn("Colab assign", logged)
        self.assertIn("Colab upload", logged)
        self.assertIn("Unable to open websocket connection", logged)
        for forbidden in ("Bearer ", "colab-runtime-proxy-token", "token=", *SECRETS):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, logged)

    def test_a_kernel_client_that_never_connected_leaves_no_cleanup_traceback(self) -> None:
        # jupyter-kernel-client 0.9: when the constructor raises (here the
        # saved kernel cannot be looked up), __del__ logs "Failed to stop the
        # kernel client" with an AttributeError on _manager at ERROR, once
        # per failed connect. The failed call reports itself.
        import gc
        import socket

        import jupyter_kernel_client

        ColabControlPlane("/nonexistent/adc.json")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
        try:
            jupyter_kernel_client.KernelClient(server_url=closed, token="t", kernel_id="k")
        except Exception:
            pass
        else:
            self.fail(f"a kernel client connected to {closed}")
        gc.collect()
        # The same message with any other error still gets through.
        logging.getLogger("jupyter_kernel_client").error(
            "Failed to stop the kernel client", exc_info=RuntimeError("socket gone")
        )

        logged = "\n".join(self.capture.lines)
        self.assertEqual(logged.count("Failed to stop the kernel client"), 1, logged)
        self.assertIn("RuntimeError: socket gone", logged)
        self.assertNotIn("_manager", logged)


class HelperTests(unittest.TestCase):
    def test_library_loggers_are_held_at_warning_and_scrubbed(self) -> None:
        ColabControlPlane("/nonexistent/adc.json")
        for name in LIBRARY_LOGGERS:
            with self.subTest(logger=name):
                library_logger = logging.getLogger(name)
                self.assertEqual(library_logger.level, logging.WARNING)
                self.assertEqual(
                    sum(isinstance(f, control_plane._ScrubFilter) for f in library_logger.filters), 1
                )
        ColabControlPlane("/nonexistent/adc.json")  # idempotent
        self.assertEqual(
            sum(isinstance(f, control_plane._ScrubFilter) for f in logging.getLogger("websocket").filters), 1
        )

    def test_redact_removes_every_token_shape(self) -> None:
        samples = [
            f"Authorization: Bearer {ACCESS_TOKEN}",
            f"{{'Authorization': 'Bearer {ACCESS_TOKEN}'}}",
            f"GET {PROXY_URL}/api/kernels?colab-runtime-proxy-token={PROXY_TOKEN}&authuser=0",
            f"wss://x/channels?session_id=s&token={PROXY_TOKEN}",
            json.dumps({"runtimeProxyInfo": {"token": PROXY_TOKEN, "tokenExpiresInSeconds": 3600}}),
            f"{{'X-Colab-Runtime-Proxy-Token': '{PROXY_TOKEN}'}}",
            json.dumps({"refresh_token": REFRESH_TOKEN, "client_secret": CLIENT_SECRET}),
            f"raw access token {ACCESS_TOKEN} and refresh token {REFRESH_TOKEN}",
        ]
        for sample in samples:
            with self.subTest(sample=sample[:40]):
                clean = redact(sample)
                self.assertIn("[redacted]", clean)
                for secret in SECRETS:
                    self.assertNotIn(secret, clean)
        self.assertIn("tokenExpiresInSeconds", redact(samples[4]))
        self.assertEqual(redact("assign failed: HTTP 412"), "assign failed: HTTP 412")

    def test_errors_are_redacted_when_built(self) -> None:
        error = TransientControlPlaneError(f"upload failed for url ?colab-runtime-proxy-token={PROXY_TOKEN}", status=403)
        self.assertNotIn(PROXY_TOKEN, str(error))
        self.assertEqual(error.status, 403)

    def test_ccu_info_parsing_tolerates_odd_values(self) -> None:
        info = CcuInfo.from_json({
            "currentBalance": "41.5",
            "consumptionRateHourly": True,
            "assignmentsCount": 2.0,
            "eligibleGpus": [{"name": "A100"}],
            "ineligibleGpus": [],
        })
        self.assertEqual(info.current_balance, 41.5)
        self.assertIsNone(info.consumption_rate_hourly)
        self.assertEqual(info.assignments_count, 2)
        # An unknown list shape says nothing; an empty list says "none eligible".
        self.assertIsNone(info.eligible_gpus)
        self.assertIsNone(info.gpu_eligible("A100"))
        self.assertEqual(info.ineligible_gpus, ())
        self.assertFalse(CcuInfo(eligible_gpus=()).gpu_eligible("A100"))
        with self.assertRaises(ControlPlaneError):
            CcuInfo.from_json(["not", "an", "object"])

    def test_adc_file_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adc.json"
            self.assertIn("cannot be read", adc_file_problem(path))
            path.write_text(json.dumps(ADC), encoding="utf-8")
            self.assertEqual(adc_file_problem(path), "")
            path.write_text(json.dumps({**ADC, "type": "service_account"}), encoding="utf-8")
            self.assertIn("authorized_user", adc_file_problem(path))
            path.write_text(json.dumps({"type": "authorized_user", "client_id": "x"}), encoding="utf-8")
            problem = adc_file_problem(path)
            self.assertIn("client_secret", problem)
            self.assertIn("refresh_token", problem)

    def test_importing_the_module_loads_no_colab_library(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys; import src.services.colab.control_plane; "
            "print(sorted(m for m in ('colab_cli', 'jupyter_kernel_client', 'google.oauth2.credentials', 'requests') if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "[]")


if __name__ == "__main__":
    unittest.main()
