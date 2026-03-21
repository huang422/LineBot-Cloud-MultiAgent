from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import sys
from pathlib import Path

import httpx


def _sign_body(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _build_noop_event() -> dict:
    return {
        "destination": "cloud-smoke-test",
        "events": [
            {
                "type": "message",
                "replyToken": "smoke-test-reply-token",
                "source": {
                    "type": "group",
                    "groupId": "CLOUD_SMOKE_GROUP",
                    "userId": "CLOUD_SMOKE_USER",
                },
                "timestamp": 1735689600000,
                "message": {
                    "type": "text",
                    "id": "smoke-message-id",
                    "text": "smoke noop",
                },
            }
        ],
    }


def _require_keys(payload: dict, keys: set[str]) -> None:
    missing = sorted(keys - payload.keys())
    if missing:
        raise RuntimeError(f"Response missing keys: {', '.join(missing)}")


def _load_webhook_secret_from_cloud_run_env(path: str | None) -> str | None:
    if not path:
        return None

    env_file = Path(path)
    if not env_file.exists():
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("LINE_CHANNEL_SECRET:"):
            continue
        return json.loads(line.split(":", 1)[1].strip())

    return None


def run_health_check(client: httpx.Client, base_url: str, expect_ready: bool) -> None:
    response = client.get(f"{base_url}/health")
    if response.status_code != 200:
        raise RuntimeError(f"/health returned {response.status_code}: {response.text}")

    payload = response.json()
    _require_keys(payload, {"status", "ready_for_webhook", "providers", "cost_controls"})

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
            f"/webhook valid signed request expected 200, got {response.status_code}: {response.text}"
        )
    print("[ok] /webhook accepts a valid signed no-op event")


def parse_args() -> argparse.Namespace:
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
        help="Optional LINE channel secret. When provided, the script also sends a valid signed webhook event.",
    )
    parser.add_argument(
        "--cloud-run-env-file",
        help="Optional Cloud Run env-vars YAML file. Used to read LINE_CHANNEL_SECRET for the valid webhook check.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    webhook_secret = args.webhook_secret or _load_webhook_secret_from_cloud_run_env(
        args.cloud_run_env_file
    )

    with httpx.Client(timeout=args.timeout) as client:
        run_health_check(client, base_url, args.expect_ready)
        run_invalid_signature_check(client, base_url)

        if webhook_secret:
            run_valid_webhook_check(client, base_url, webhook_secret)
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
