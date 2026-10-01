"""Post-deploy smoke checks for a deployed LineBot-CloudAgent service.

scripts/deploy_cloud_run.sh runs this after ``gcloud run deploy``. The signed
webhook check posts a body with an empty ``events`` list, so it proves the
signature path works without writing anything to production memory.

The LINE channel secret comes from ``--webhook-secret`` or, preferably, the
SMOKE_WEBHOOK_SECRET environment variable (it stays out of ``ps`` output).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys

import httpx

WEBHOOK_SECRET_ENV = "SMOKE_WEBHOOK_SECRET"
# The public /health is minimal; the rest is on the authenticated /health/detail.
_REQUIRED_HEALTH_KEYS = {"status", "ready_for_webhook"}


def _sign_body(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _build_noop_event() -> dict:
    # No events: the webhook verifies the signature and returns 200 without
    # recording, caching or routing anything.
    return {"destination": "cloud-smoke-test", "events": []}


def _require_keys(payload: dict, keys: set[str]) -> None:
    missing = sorted(keys - payload.keys())
    if missing:
        raise RuntimeError(f"Response missing keys: {', '.join(missing)}")


def run_health_check(client: httpx.Client, base_url: str, expect_ready: bool) -> None:
    response = client.get(f"{base_url}/health")
    if response.status_code != 200:
        raise RuntimeError(f"/health returned {response.status_code}: {response.text[:200]}")

    payload = response.json()
    _require_keys(payload, _REQUIRED_HEALTH_KEYS)

    print(f"[ok] /health status={payload['status']}")
    print(f"     ready_for_webhook={payload['ready_for_webhook']}")

    if expect_ready and not payload["ready_for_webhook"]:
        raise RuntimeError("Deployment is reachable, but ready_for_webhook=false")


def run_invalid_signature_check(client: httpx.Client, base_url: str) -> None:
    response = client.post(
        f"{base_url}/webhook",
        content=b'{"events":[]}',
        headers={
            "Content-Type": "application/json",
            "X-Line-Signature": "invalid-signature",
        },
    )
    if response.status_code != 403:
        raise RuntimeError(
            f"/webhook invalid signature check expected 403, got {response.status_code}"
        )
    print("[ok] /webhook rejects invalid signatures")


def run_valid_webhook_check(
    client: httpx.Client,
    base_url: str,
    webhook_secret: str,
) -> None:
    payload = _build_noop_event()
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    response = client.post(
        f"{base_url}/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Line-Signature": _sign_body(webhook_secret, body),
        },
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"/webhook valid signed request expected 200, got {response.status_code}: "
            f"{response.text[:200]}"
        )
    print("[ok] /webhook accepts a valid signed request (no events, nothing recorded)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test a deployed LineBot-CloudAgent service."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Cloud Run base URL, e.g. https://linebot-cloudagent-xxxxx.a.run.app",
    )
    parser.add_argument(
        "--webhook-secret",
        help=(
            "LINE channel secret for the signed webhook check. Defaults to the "
            f"{WEBHOOK_SECRET_ENV} environment variable; the check is skipped when neither is set."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--expect-ready",
        action="store_true",
        help="Fail if /health reports ready_for_webhook=false.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, client: httpx.Client | None = None) -> int:
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/")
    webhook_secret = args.webhook_secret or os.environ.get(WEBHOOK_SECRET_ENV, "")

    with client or httpx.Client(timeout=args.timeout) as http:
        run_health_check(http, base_url, args.expect_ready)
        run_invalid_signature_check(http, base_url)

        if webhook_secret:
            run_valid_webhook_check(http, base_url, webhook_secret)
        else:
            print("[skip] valid signed webhook check skipped (no webhook secret available)")

    print("[done] smoke test passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
