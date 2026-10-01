"""Shared fakes for the Colab tests: a scripted tunnel and a READY session.

:class:`FakeTunnel` stands in for the Cloudflare tunnel + auth proxy +
Ollama 0.35.0 behind an ``httpx.MockTransport``: it records every request
and answers each path from a script (a status and JSON, a callable, or
``("hang", seconds)``). By default it answers like a healthy session with
both pinned models loaded wholly in VRAM, and :func:`ready_state` records
what a passing readiness probe saw.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from src.services.colab import protocol
from src.services.colab.endpoint import ColabEndpoint
from src.services.colab.state import READY, ColabState, InMemoryColabStateStore

SECRET = "s" * 40
NBH = "6f1c1f4e-0b8e-4c55-9f5a-2b4d6c8e0a13"
TUNNEL_HOST = "brave-otter-42.trycloudflare.com"
TUNNEL = f"https://{TUNNEL_HOST}"
QWEN = "qwen3.8:27b-mtp-q4_K_M"
NIMBLE = "nimble:9b"
# What /api/ps reports for each pinned model (bytes, all of it in VRAM).
LOADED_SIZES = {QWEN: 25_300_000_000, NIMBLE: 10_601_505_280}


def write_adc(directory: str | Path) -> Path:
    """A well-formed (fake) authorized_user credential file for COLAB_ADC_PATH."""
    path = Path(directory) / "adc.json"
    path.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "764086051850-fake.apps.googleusercontent.com",
                "client_secret": "fake-client-secret",
                "refresh_token": "1//fake-refresh-token",
            }
        ),
        encoding="utf-8",
    )
    return path


def bearer(nbh: str = NBH) -> str:
    return protocol.derive_proxy_token(SECRET, nbh)


def on_gpu(model: str, size_vram: int | None = None) -> dict[str, int]:
    """*model*'s ``{"size", "size_vram"}``: all of it in VRAM unless *size_vram* says less."""
    size = LOADED_SIZES[model]
    return {"size": size, "size_vram": size if size_vram is None else size_vram}


def ready_state(now: float | None = None, **changes: Any) -> ColabState:
    """A READY session whose probe saw the pinned digests, all in VRAM."""
    now = time.time() if now is None else now
    values: dict[str, Any] = {
        "status": READY,
        "since": now - 60,
        "nbh": NBH,
        "tunnel_url": TUNNEL,
        "ready_at": now - 60,
        "last_health_at": now,
        "digests": protocol.pinned_models(),
        "vram": {model: on_gpu(model) for model in protocol.pinned_models()},
    }
    values.update(changes)
    return ColabState(**values)


def healthy_routes() -> dict[str, Any]:
    pins = protocol.pinned_models()
    return {
        "GET /api/version": (200, {"version": protocol.OLLAMA_VERSION}),
        "GET /api/ps": (200, {"models": [{"name": model, "model": model, **on_gpu(model)} for model in pins]}),
        "GET /api/tags": (
            200,
            {"models": [{"name": model, "model": model, "digest": digest} for model, digest in pins.items()]},
        ),
        "POST /api/show": (
            200,
            {"thinking": {"values": [False, "low", "medium", "xhigh"], "default": "medium"}},
        ),
        "POST /api/chat": (
            200,
            {"model": QWEN, "message": {"role": "assistant", "content": "Colab 的回答"}, "done": True, "done_reason": "stop"},
        ),
        "POST /v1/systemone": (200, systemone_answer()),
    }


def systemone_answer(
    agent: str = "chat",
    agent_confidence: float = 0.9,
    output: str = "text",
    output_confidence: float = 0.95,
    noul: float = 0.05,
) -> dict:
    return {
        "model": NIMBLE,
        "answers": {
            "agent": {"type": "choice", "choice": agent, "probabilities": {agent: 0.97}, "confidence": agent_confidence},
            "output_format": {
                "type": "choice",
                "choice": output,
                "probabilities": {output: 0.98},
                "confidence": output_confidence,
            },
            "needs_thinking": {"type": "noul", "noul": noul},
        },
        "usage": {"input_tokens": 900, "output_tokens": 3},
    }


class FakeTunnel:
    """Records requests; answers ``"METHOD /path"`` from :attr:`routes`."""

    def __init__(self, **overrides: Any) -> None:
        self.routes: dict[str, Any] = healthy_routes()
        self.routes.update(overrides)
        self.requests: list[httpx.Request] = []

    def paths(self) -> list[str]:
        return [f"{request.method} {request.url.path}" for request in self.requests]

    def bodies(self, route: str) -> list[dict]:
        return [
            json.loads(request.content)
            for request in self.requests
            if f"{request.method} {request.url.path}" == route
        ]

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get(f"{request.method} {request.url.path}", (404, {"error": "not found"}))
        if callable(answer):
            answer = answer(request)
            if asyncio.iscoroutine(answer):
                answer = await answer
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, BaseException):
            raise answer
        if answer[0] == "hang":
            await asyncio.sleep(answer[1])
            return httpx.Response(200, json={})
        status, body = answer
        if isinstance(body, (bytes, str)):
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=body)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


class Clock:
    """A settable epoch clock."""

    def __init__(self, now: float | None = None) -> None:
        self.now = time.time() if now is None else now

    def __call__(self) -> float:
        return self.now


def make_endpoint(
    state: ColabState | None = None,
    tunnel: FakeTunnel | None = None,
    *,
    store: Any = None,
    clock: Callable[[], float] = time.time,
    auto_start: bool = True,
) -> tuple[ColabEndpoint, FakeTunnel]:
    """An endpoint over *tunnel* and an in-memory store holding *state*
    (default: READY and healthy). Nothing is read yet: ``ready()`` reads
    the store the first time, like after a cold start."""
    tunnel = tunnel or FakeTunnel()
    store = store or InMemoryColabStateStore(state if state is not None else ready_state(clock()))
    endpoint = ColabEndpoint(SECRET, store, auto_start=auto_start, transport=tunnel.transport(), clock=clock)
    return endpoint, tunnel


# ── The control plane (Colab's session backend and the runtime proxy) ──


def runtime(endpoint: str = "ep-1", accelerator: str = "A100") -> Any:
    """A RuntimeAssignment as list_assignments/assign hand it out."""
    from src.services.colab.control_plane import RuntimeAssignment

    return RuntimeAssignment(
        endpoint=endpoint,
        proxy_url=f"https://{endpoint}.proxy.example",
        proxy_token=f"proxy-token-{endpoint}",
        token_expires_in=3600,
        accelerator=accelerator,
        variant="GPU",
    )


class FakeControlPlane:
    """Stands in for ColabControlPlane: records calls, answers from scripts.

    ``runtimes`` is what the account has assigned (``assign`` adds one,
    ``unassign`` removes it), and ``by_nbh`` which nbh each belongs to.
    ``assign_results`` / ``cell_results`` are consumed in order (an
    exception is raised); when empty, assign hands out a new runtime and a
    cell succeeds, printing the launch sentinel or the VM state line when
    the code asks for it. With ``timeouts_leave_runtimes`` a time-out still
    assigns one (googlecolab/google-colab-cli discussion #141).
    """

    def __init__(self) -> None:
        from src.services.colab.control_plane import CcuInfo

        self.calls: list[tuple[Any, ...]] = []
        self.runtimes: dict[str, Any] = {}
        self.assign_results: list[Any] = []
        self.cell_results: list[Any] = []
        self.ccu: Any = CcuInfo(
            current_balance=100.0,
            consumption_rate_hourly=0.0,
            assignments_count=0,
            eligible_gpus=("A100", "L4"),
            ineligible_gpus=(),
        )
        self.list_error: Exception | None = None
        self.unassign_error: Exception | None = None
        self.find_error: Exception | None = None
        self.timeouts_leave_runtimes = False
        self.by_nbh: dict[str, str] = {}
        self.vm_state: dict[str, Any] = {"step": "pull", "ts": 1, "error": None}
        self.uploads: list[dict[str, Any]] = []
        self._next = 0

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]

    def _new_runtime(self, nbh: str, cpu_only: bool) -> Any:
        existing = self.by_nbh.get(nbh)
        if existing in self.runtimes:
            return self.runtimes[existing]
        self._next += 1
        result = runtime(f"ep-{self._next}", "NONE" if cpu_only else "A100")
        self.by_nbh[nbh] = result.endpoint
        return result

    async def assign(self, nbh: Any, *, cpu_only: bool = False) -> Any:
        from src.services.colab.control_plane import ControlPlaneTimeoutError

        self.calls.append(("assign", str(nbh), cpu_only))
        if self.assign_results:
            result = self.assign_results.pop(0)
            if isinstance(result, BaseException):
                if self.timeouts_leave_runtimes and isinstance(result, ControlPlaneTimeoutError):
                    left = self._new_runtime(str(nbh), cpu_only)
                    self.runtimes[left.endpoint] = left
                raise result
        else:
            result = self._new_runtime(str(nbh), cpu_only)
        self.runtimes[result.endpoint] = result
        return result

    async def find_assignment(self, nbh: Any, *, cpu_only: bool = False) -> Any:
        self.calls.append(("find_assignment", str(nbh), cpu_only))
        if self.find_error is not None:
            raise self.find_error
        return self.runtimes.get(self.by_nbh.get(str(nbh), ""))

    async def list_assignments(self) -> list[Any]:
        self.calls.append(("list_assignments",))
        if self.list_error is not None:
            raise self.list_error
        return list(self.runtimes.values())

    async def unassign(self, endpoint: str) -> bool:
        self.calls.append(("unassign", endpoint))
        if self.unassign_error is not None:
            raise self.unassign_error
        return self.runtimes.pop(endpoint, None) is not None

    async def ccu_info(self) -> Any:
        self.calls.append(("ccu_info",))
        if isinstance(self.ccu, BaseException):
            raise self.ccu
        return self.ccu

    async def upload(self, assignment: Any, files: Any) -> None:
        self.calls.append(("upload", assignment.endpoint))
        self.uploads.append(dict(files))

    async def run_cell(
        self,
        assignment: Any,
        code: str,
        *,
        budget_s: float,
        kernel_id: str = "",
        session_id: str = "",
        silent: bool = False,
    ) -> Any:
        from src.services.colab.control_plane import CELL_OK, CellResult
        from src.services.colab.controller import HEARTBEAT_MARKER
        from src.services.colab.launcher import LAUNCH_SENTINEL

        self.calls.append(("run_cell", assignment.endpoint, code, kernel_id, session_id, silent, budget_s))
        if self.cell_results:
            result = self.cell_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        stdout = ""
        if LAUNCH_SENTINEL in code:
            stdout = f"{LAUNCH_SENTINEL} 4242\n"
        elif HEARTBEAT_MARKER in code:
            stdout = HEARTBEAT_MARKER + json.dumps(self.vm_state) + "\n"
        return CellResult(CELL_OK, kernel_id or "kernel-1", session_id or "session-1", stdout=stdout)
