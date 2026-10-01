from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import io
import json
import os
import unittest
from unittest.mock import patch

import httpx

from tests import smoke_test_cloud

_SECRET = "line-channel-secret-for-tests"
_BASE_URL = "https://linebot.example.run.app"


class _FakeService:
    """Answers like a deployed service and records every request."""

    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "healthy" if self.ready else "degraded",
                    "ready_for_webhook": self.ready,
                    "version": "dev",
                },
            )
        if request.url.path == "/webhook":
            expected = base64.b64encode(
                hmac.new(_SECRET.encode(), request.content, hashlib.sha256).digest()
            ).decode()
            if request.headers.get("X-Line-Signature") != expected:
                return httpx.Response(403)
            return httpx.Response(200)
        return httpx.Response(404)

    def signed_webhook_bodies(self) -> list[dict]:
        return [
            json.loads(request.content)
            for request in self.requests
            if request.url.path == "/webhook"
            and request.headers.get("X-Line-Signature") != "invalid-signature"
        ]


def _run(argv: list[str], service: _FakeService) -> tuple[int, str]:
    client = httpx.Client(transport=httpx.MockTransport(service))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = smoke_test_cloud.main(argv, client=client)
    return code, out.getvalue()


class SmokeTestCloudTests(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(smoke_test_cloud.WEBHOOK_SECRET_ENV, None)

    def test_signed_webhook_check_sends_no_events(self) -> None:
        service = _FakeService()

        code, out = _run(
            ["--base-url", _BASE_URL, "--expect-ready", "--webhook-secret", _SECRET], service
        )

        self.assertEqual(code, 0)
        # Nothing for production memory to record: the event list is empty.
        self.assertEqual(
            service.signed_webhook_bodies(),
            [{"destination": "cloud-smoke-test", "events": []}],
        )
        self.assertIn("[done] smoke test passed", out)

    def test_webhook_secret_can_come_from_the_environment(self) -> None:
        service = _FakeService()
        os.environ[smoke_test_cloud.WEBHOOK_SECRET_ENV] = _SECRET

        code, out = _run(["--base-url", _BASE_URL], service)

        self.assertEqual(code, 0)
        self.assertEqual(len(service.signed_webhook_bodies()), 1)
        self.assertNotIn(_SECRET, out)

    def test_signed_check_is_skipped_without_a_secret(self) -> None:
        service = _FakeService()

        code, out = _run(["--base-url", _BASE_URL], service)

        self.assertEqual(code, 0)
        self.assertEqual(service.signed_webhook_bodies(), [])
        self.assertIn("[skip]", out)

    def test_the_minimal_public_health_is_enough(self) -> None:
        self.assertEqual(smoke_test_cloud._REQUIRED_HEALTH_KEYS, {"status", "ready_for_webhook"})

        class _Minimal(_FakeService):
            def __call__(self, request):
                if request.url.path == "/health":
                    self.requests.append(request)
                    return httpx.Response(200, json={"status": "healthy", "ready_for_webhook": True})
                return super().__call__(request)

        code, out = _run(["--base-url", _BASE_URL, "--expect-ready"], _Minimal())

        self.assertEqual(code, 0)
        self.assertIn("ready_for_webhook=True", out)

    def test_a_health_payload_without_readiness_fails(self) -> None:
        class _Broken(_FakeService):
            def __call__(self, request):
                if request.url.path == "/health":
                    return httpx.Response(200, json={"status": "healthy"})
                return super().__call__(request)

        with self.assertRaisesRegex(RuntimeError, "missing keys: ready_for_webhook"):
            _run(["--base-url", _BASE_URL], _Broken())

    def test_expect_ready_fails_on_a_degraded_service(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ready_for_webhook=false"):
            _run(["--base-url", _BASE_URL, "--expect-ready"], _FakeService(ready=False))

    def test_cloud_run_env_file_option_is_gone(self) -> None:
        # Cloud Build no longer runs the smoke step, so no env YAML to read.
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke_test_cloud.parse_args(["--base-url", _BASE_URL, "--cloud-run-env-file", "x.yaml"])
