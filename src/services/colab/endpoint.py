"""What the request path needs to know about the Colab session, without waiting.

:class:`ColabEndpoint` answers, from memory, whether a Colab tier may be
called right now, and carries the one HTTP client that talks to the
session's tunnel:

- **READY?** The state document says READY with a tunnel URL, and a health
  check (or readiness probe, or a successful call) passed within
  :data:`READY_HEALTH_TTL_SECONDS`, and nothing since has marked the
  endpoint unhealthy. The document is read once after a cold start
  (bounded by :data:`STATE_LOAD_TIMEOUT_SECONDS`); after that every writer
  in this process goes through the same store, whose cache this reads. A
  tier that is not READY is skipped at once, with no network call, so a
  message never waits for Colab.
- **Busy?** Ollama serves the qwen3.5-family models one request at a time
  and never answers 503 for a loaded, busy model: a second request just
  queues. So each Colab model has one ``asyncio.Semaphore(1)`` here, and a
  tier whose model is busy is skipped rather than queued. This holds
  because the service runs with ``--max-instances=1``.
- **Breaker.** :data:`BREAKER_FAILURES` failed chat calls in a row
  (transport errors, the client's own timeouts, 5xx, a 404, a refused
  bearer) open it for :data:`BREAKER_OPEN_SECONDS`; a success closes it.
  A call cancelled by the chain's time limit is not a failure here: the
  chain judges those (ModelHealth), and a hung tunnel fails the health
  check. The router's outcomes
  are its own (``ModelHealth``), so a slow router never skips Colab chat.
- **The router is optional.** qwen3.8 must be loaded for READY and for
  every health check; the router need not be. A router a health check
  finds not loaded (evicted when qwen3.8 was reloaded) or not wholly in
  VRAM sits the rest of the session out (:meth:`off_gpu`), and nothing
  calls it: a call would load it again, and loading it can evict qwen3.8.

The tunnel client uses the SSRF-safe transport of the page fetcher (port
443 only, every address public, the connection pinned to the address that
passed), ``trust_env`` off and no redirects, and a read timeout under
Cloudflare's 125 s edge limit. The bearer is derived per session
(:func:`.protocol.derive_proxy_token`), never stored.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.services.colab import protocol
from src.services.colab.state import READY, ColabState, ColabStateError, ColabStateStore, clip
from src.services.search.page_fetcher import ssrf_safe_transport
from src.utils.logger import logger

READY_HEALTH_TTL_SECONDS = 120.0
BREAKER_FAILURES = 3
BREAKER_OPEN_SECONDS = 120.0
# A triggered message records demand at most this often, and the write
# gives up after DEMAND_WRITE_TIMEOUT_SECONDS (the pipeline joins it only
# after the reply went out, so a slow Firestore never delays an answer).
DEMAND_INTERVAL_SECONDS = 60.0
DEMAND_WRITE_TIMEOUT_SECONDS = 3.0
STATE_LOAD_TIMEOUT_SECONDS = 3.0
# After a failed cold-start read, the next try waits this long.
STATE_LOAD_RETRY_SECONDS = 30.0
HEALTH_TIMEOUT_SECONDS = 5.0
# Cloudflare cuts a response that takes 125 s (HTTP 524); stay under it.
REQUEST_TIMEOUT = httpx.Timeout(115.0, connect=5.0)
TUNNEL_PORTS: tuple[int, ...] = (443,)


@dataclass(frozen=True)
class ColabLimits:
    """The session limits from settings (the controller enforces them)."""

    idle_ttl_seconds: float = 900.0
    max_session_seconds: float = 39600.0
    min_cu_balance: float = 10.0
    daily_cu_cap: float = 30.0

    @classmethod
    def from_settings(cls, settings: Any) -> ColabLimits:
        return cls(
            idle_ttl_seconds=float(settings.colab_idle_ttl_seconds),
            max_session_seconds=float(settings.colab_max_session_seconds),
            min_cu_balance=float(settings.colab_min_cu_balance),
            daily_cu_cap=float(settings.colab_daily_cu_cap),
        )


class ColabEndpoint:
    """The session as the request path sees it. See the module docstring."""

    name = "colab"

    def __init__(
        self,
        secret: str,
        store: ColabStateStore,
        *,
        auto_start: bool = True,
        tunnel_host: str = "",
        limits: ColabLimits | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """``auto_start``: MODEL_PATH_PREFERENCE=colab, so triggered
        messages record demand and the controller starts a session for
        them; with "api" messages record nothing and only ``!colab on``
        (``manual_demand_at``, :mod:`.admin`) starts one. ``transport`` is
        for tests (an ``httpx.MockTransport`` standing in for the tunnel)."""
        self._secret = secret
        self.store = store
        self.auto_start = auto_start
        self.tunnel_host = tunnel_host
        self.limits = limits or ColabLimits()
        self._clock = clock
        self._client = httpx.AsyncClient(
            transport=transport or ssrf_safe_transport(allowed_ports=TUNNEL_PORTS),
            trust_env=False,
            follow_redirects=False,
            timeout=REQUEST_TIMEOUT,
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._load_failed_at: float | None = None
        # Per session (nbh): in-process health and the breaker.
        self._session = ""
        self._ok_at: float | None = None
        self._bad_at: float | None = None
        self._failures = 0
        self._open_until = 0.0
        # Routers a health check found not loaded or not wholly in VRAM:
        # model → (size_vram, size), 0s when not loaded.
        self._routers_dropped: dict[str, tuple[int, int]] = {}
        self.last_error = ""
        self._demand_noted_at: float | None = None

    # ── State ────────────────────────────────────────────────

    @property
    def state(self) -> ColabState | None:
        """The last known state; None until it was read once."""
        return self.store.cached

    async def ensure_loaded(self) -> ColabState | None:
        """The state, read from the store once per process; None if unreadable.

        A failed read is retried at most every STATE_LOAD_RETRY_SECONDS,
        so a Firestore outage costs a message at most one bounded wait.
        """
        state = self.store.cached
        if state is not None:
            return state
        now = self._clock()
        if self._load_failed_at is not None and now - self._load_failed_at < STATE_LOAD_RETRY_SECONDS:
            return None
        try:
            async with asyncio.timeout(STATE_LOAD_TIMEOUT_SECONDS):
                return await self.store.load()
        except (TimeoutError, ColabStateError) as exc:
            self._load_failed_at = now
            logger.warning(f"Colab: state not loaded, tiers skipped for now: {exc or type(exc).__name__}")
            return None

    def _sync_session(self) -> ColabState | None:
        """The known state; health and breaker belong to one session (nbh),
        so a new one starts afresh."""
        state = self.store.cached
        if state is not None and state.nbh != self._session:
            self._session = state.nbh
            self._ok_at = self._bad_at = None
            self._failures = 0
            self._open_until = 0.0
            self._routers_dropped = {}
        return state

    def is_ready(self) -> bool:
        """READY, with a passing health check in the last READY_HEALTH_TTL_SECONDS."""
        state = self._sync_session()
        if state is None or state.status != READY or not state.tunnel_url or not state.nbh:
            return False
        healthy = max(state.last_health_at or 0.0, self._ok_at or 0.0)
        if not healthy or self._clock() - healthy > READY_HEALTH_TTL_SECONDS:
            return False
        return self._bad_at is None or healthy > self._bad_at

    async def ready(self) -> bool:
        """:meth:`is_ready`, after the one cold-start read of the state."""
        await self.ensure_loaded()
        return self.is_ready()

    def unpinned(self) -> dict[str, tuple[str, str]]:
        """The READY session's routers at another digest than their pin (or
        none recorded): model → (pinned, actual), from the digests its
        readiness probe saw. The router then stays out of the session.
        Empty while no session is READY."""
        state = self._sync_session()
        if state is None or state.status != READY:
            return {}
        return protocol.unpinned(state.digests)

    def off_gpu(self) -> dict[str, tuple[int, int]]:
        """The READY session's routers not wholly in VRAM, or not loaded:
        when its readiness probe looked (or with no sizes recorded), or at a
        later health check. model → (size_vram, size), (0, 0) when not
        loaded. The router then stays out of the session. Empty while no
        session is READY."""
        state = self._sync_session()
        if state is None or state.status != READY:
            return {}
        return {**protocol.off_gpu(state.vram), **self._routers_dropped}

    @property
    def base_url(self) -> str | None:
        state = self.store.cached
        return state.tunnel_url if state is not None and state.tunnel_url else None

    def bearer(self, nbh: str | None = None) -> str | None:
        """The auth proxy's bearer for *nbh* (default: the current session)."""
        if nbh is None:
            state = self.store.cached
            nbh = state.nbh if state is not None else ""
        return protocol.derive_proxy_token(self._secret, nbh) if nbh else None

    # ── Busy gate and breaker ────────────────────────────────

    def semaphore(self, model: str) -> asyncio.Semaphore:
        """The one-at-a-time gate of *model* (Ollama queues, it never refuses)."""
        return self._semaphores.setdefault(model, asyncio.Semaphore(1))

    def breaker_open(self) -> bool:
        self._sync_session()
        return self._clock() < self._open_until

    def record_success(self) -> None:
        self._sync_session()
        self._failures = 0
        self._open_until = 0.0
        self._ok_at = self._clock()

    def record_failure(self, reason: str) -> None:
        """One failed call; BREAKER_FAILURES in a row open the breaker."""
        self._sync_session()
        self.last_error = clip(reason)
        self._failures += 1
        if self._failures >= BREAKER_FAILURES:
            self._failures = 0
            self._open_until = self._clock() + BREAKER_OPEN_SECONDS
            logger.warning(
                f"Colab: {BREAKER_FAILURES} failed calls in a row; tiers skipped for "
                f"{BREAKER_OPEN_SECONDS:.0f}s (last: {self.last_error})"
            )

    def mark_unhealthy(self, reason: str) -> None:
        """Not READY until the next passing health check (a 404, a refused bearer)."""
        self._sync_session()
        self._bad_at = self._clock()
        self.last_error = clip(reason)

    # ── Tunnel HTTP ──────────────────────────────────────────

    async def call(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        timeout: float | httpx.Timeout | None = None,
        base_url: str | None = None,
        nbh: str | None = None,
    ) -> httpx.Response:
        """One request through the tunnel with the session's bearer.

        Raises:
            httpx.HTTPError, BlockedURLError, FetchError: Transport failures.
            RuntimeError: No tunnel URL or session to call.
        """
        base = base_url or self.base_url
        token = self.bearer(nbh)
        if not base or not token:
            raise RuntimeError("no Colab tunnel to call")
        kwargs: dict[str, Any] = {"headers": {"Authorization": f"Bearer {token}"}}
        if json is not None:
            kwargs["json"] = json
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await self._client.request(method, f"{base}{path}", **kwargs)

    async def check_health(self) -> bool:
        """``/api/version`` and ``/api/ps`` answer through the tunnel (≤ 5 s),
        with qwen3.8 loaded.

        For the controller's per-minute tick while READY. A pass keeps the
        tiers usable for READY_HEALTH_TTL_SECONDS; a failure marks the
        endpoint unhealthy. A router not loaded or not wholly in VRAM is no
        failure: it sits the rest of the session out (:meth:`off_gpu`).
        Health calls do not count as Colab activity: the VM's idle timer
        only moves on inference calls.
        """
        if self.base_url is None or self.bearer() is None:
            return False
        try:
            async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
                problems, sizes = await self._check_serving(self.base_url, self.bearer())
        except TimeoutError:
            problems = [f"no answer within {HEALTH_TIMEOUT_SECONDS:.0f}s"]
        except Exception as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
        if problems:
            self.mark_unhealthy("health check: " + "; ".join(problems))
            return False
        self._sync_session()
        self._ok_at = self._clock()
        self._drop_routers_off_gpu(sizes)
        return True

    def _drop_routers_off_gpu(self, sizes: dict[str, dict[str, int]]) -> None:
        """Keeps each router *sizes* (a passing health check's ``/api/ps``)
        does not show wholly in VRAM out of the rest of the session."""
        for model, share in protocol.off_gpu(sizes).items():
            if model not in self._routers_dropped:
                self._routers_dropped[model] = share
                logger.warning(
                    f"Colab: health check found {model} "
                    + ("not loaded" if not share[1] else "not wholly in VRAM")
                    + "; the router sits the rest of this session out, chat goes on"
                )

    async def readiness(self, tunnel_url: str, nbh: str) -> ProbeReport:
        """The readiness probe after a ``ready`` registration.

        Within PROBE_TIMEOUT_SECONDS: the Ollama version, the chat model
        loaded, listed, wholly in VRAM and at its pinned digest, and its
        think values. A router that is not listed or loaded, at another
        digest or not wholly in VRAM is no problem, only missing from or
        shown in the report's digests and sizes (:func:`.protocol.unpinned`,
        :func:`.protocol.off_gpu`).
        """
        token = self.bearer(nbh)
        try:
            async with asyncio.timeout(protocol.PROBE_TIMEOUT_SECONDS):
                (serving, vram), (listed, digests), *think = await asyncio.gather(
                    self._check_serving(tunnel_url, token),
                    self._check_digests(tunnel_url, token),
                    *(self._check_think(tunnel_url, token, model) for model in protocol.chat_models()),
                )
        except TimeoutError:
            return ProbeReport([f"no answer within {protocol.PROBE_TIMEOUT_SECONDS:.0f}s"])
        except Exception as exc:
            return ProbeReport([f"{type(exc).__name__}: {exc}"])
        routers = protocol.router_models()
        spilled = [
            _gpu_problem(model, sizes)
            for model, sizes in vram.items()
            if model not in routers and not protocol.on_gpu(sizes)
        ]
        problems = [*serving, *spilled, *listed, *(problem for problems in think for problem in problems)]
        return ProbeReport(problems, digests, vram)

    async def probe(self, tunnel_url: str, nbh: str) -> list[str]:
        """Everything that differs from the pins, the router's too (listed,
        loaded, its digest and VRAM); [] means nothing does. The live gate's
        check (scripts/live_gate.py): a session can be READY without its
        router (:meth:`readiness`)."""
        report = await self.readiness(tunnel_url, nbh)
        problems = list(report.problems)
        for model, (pinned, actual) in protocol.unpinned(report.digests).items():
            problems.append(_digest_problem(model, actual, pinned) if actual else f"not listed: {model}")
        for model in protocol.off_gpu(report.vram):
            problems.append(_gpu_problem(model, report.vram[model]) if model in report.vram else f"not loaded: {model}")
        return problems

    async def _get_json(self, base: str, token: str | None, method: str, path: str, body: Any = None) -> Any:
        response = await self._client.request(
            method,
            f"{base}{path}",
            headers={"Authorization": f"Bearer {token}"},
            **({"json": body} if body is not None else {}),
        )
        if response.status_code != 200:
            raise _ProbeFailed(f"{path} answered HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise _ProbeFailed(f"{path} answered something that is not JSON") from None

    async def _check_serving(self, base: str, token: str | None) -> tuple[list[str], dict[str, dict[str, int]]]:
        """The problems (a pinned model other than a router not loaded), and
        each loaded pinned model's ``/api/ps`` sizes (``{"size",
        "size_vram"}``, which the callers judge)."""
        try:
            version = await self._get_json(base, token, "GET", "/api/version")
            running = await self._get_json(base, token, "GET", "/api/ps")
        except _ProbeFailed as exc:
            return [str(exc)], {}
        problems: list[str] = []
        found = version.get("version") if isinstance(version, dict) else None
        if found != protocol.OLLAMA_VERSION:
            problems.append(f"Ollama {clip(str(found), 40)}, expected {protocol.OLLAMA_VERSION}")
        pins = protocol.pinned_models()
        loaded = _loaded_sizes(running)
        missing = sorted(set(pins) - set(loaded) - set(protocol.router_models()))
        if missing:
            problems.append(f"not loaded: {', '.join(missing)}")
        return problems, {model: sizes for model, sizes in loaded.items() if model in pins}

    async def _check_digests(self, base: str, token: str | None) -> tuple[list[str], dict[str, str]]:
        """The problems, and each listed pinned model's digest in ``/api/tags``:
        every pinned model but a router must be listed at its pin."""
        try:
            tags = await self._get_json(base, token, "GET", "/api/tags")
        except _ProbeFailed as exc:
            return [str(exc)], {}
        listed: dict[str, str] = {}
        for entry in tags.get("models", []) if isinstance(tags, dict) else []:
            if isinstance(entry, dict):
                digest = protocol.normalize_digest(entry.get("digest"))
                for key in ("name", "model"):
                    if isinstance(entry.get(key), str):
                        listed[entry[key]] = digest
        pins = protocol.pinned_models()
        routers = protocol.router_models()
        problems = [
            _digest_problem(model, listed.get(model, "missing"), expected)
            for model, expected in sorted(pins.items())
            if listed.get(model) != expected and model not in routers
        ]
        return problems, {model: listed[model] for model in pins if model in listed}

    async def _check_think(self, base: str, token: str | None, model: str) -> list[str]:
        try:
            shown = await self._get_json(base, token, "POST", "/api/show", {"model": model})
        except _ProbeFailed as exc:
            return [str(exc)]
        thinking = shown.get("thinking") if isinstance(shown, dict) else None
        values = thinking.get("values") if isinstance(thinking, dict) else None
        if values != protocol.CHAT_THINK_VALUES:
            return [f"{model} think values {clip(str(values), 60)}, expected {protocol.CHAT_THINK_VALUES}"]
        return []

    # ── Demand ───────────────────────────────────────────────

    async def note_demand(self) -> bool:
        """Record that a triggered message wanted Colab (at most once a minute).

        True when it was written. Never raises and never takes longer than
        DEMAND_WRITE_TIMEOUT_SECONDS: a failed or slow write only delays the
        next session start by a minute.
        """
        now = self._clock()
        if self._demand_noted_at is not None and now - self._demand_noted_at < DEMAND_INTERVAL_SECONDS:
            return False
        self._demand_noted_at = now

        def change(state: ColabState) -> ColabState | None:
            if state.last_demand_at is not None and state.last_demand_at >= now:
                return None
            return replace(state, last_demand_at=now)

        try:
            async with asyncio.timeout(DEMAND_WRITE_TIMEOUT_SECONDS):
                await self.store.update(change)
        except TimeoutError:
            logger.warning(f"Colab: demand not recorded within {DEMAND_WRITE_TIMEOUT_SECONDS:g}s")
            return False
        except ColabStateError as exc:
            logger.warning(f"Colab: demand not recorded: {exc}")
            return False
        return True

    # ── /health/detail ───────────────────────────────────────

    def view(self) -> dict[str, Any]:
        """The session for ``/health/detail`` (no URL, bearer or ids)."""
        state = self.store.cached
        now = self._clock()
        view: dict[str, Any] = {
            "enabled": True,
            "loaded": state is not None,
            "auto_start": self.auto_start,
            "ready": self.is_ready(),
            "breaker_open_until": self._open_until if self.breaker_open() else None,
            "busy": sorted(model for model, gate in self._semaphores.items() if gate.locked()),
            "endpoint_error": self.last_error or None,
            "limits": {
                "idle_ttl_seconds": self.limits.idle_ttl_seconds,
                "max_session_seconds": self.limits.max_session_seconds,
                "min_cu_balance": self.limits.min_cu_balance,
                "daily_cu_cap": self.limits.daily_cu_cap,
            },
        }
        if state is None:
            return view
        view.update(
            {
                "status": state.status,
                "since": state.since or None,
                "desired": state.desired,
                "tunnel": _tunnel_kind(state.tunnel_url, self.tunnel_host),
                "ready_at": state.ready_at,
                "last_health_at": max(state.last_health_at or 0.0, self._ok_at or 0.0) or None,
                "last_demand_at": state.last_demand_at,
                "manual_demand_at": state.manual_demand_at,
                "last_heartbeat_at": state.last_heartbeat_at,
                "boot_step": state.boot_step or None,
                "versions": dict(state.versions),
                "gpu": dict(state.gpu),
                # A republished router, or one not loaded or partly on the
                # CPU: the READY session routes without it.
                "unpinned": {
                    model: {"pinned": pinned, "actual": actual or None}
                    for model, (pinned, actual) in self.unpinned().items()
                }
                or None,
                "off_gpu": {
                    model: {"size_vram": in_vram, "size": size}
                    for model, (in_vram, size) in self.off_gpu().items()
                }
                or None,
                "last_error": state.last_error or None,
                "failures": state.failures,
                "health_failures": state.health_failures,
                "pending": state.action or None,
                "hold_reason": state.hold_reason or None,
                "cooldown_until": state.cooldown_until if (state.cooldown_until or 0) > now else None,
                "disabled_reason": state.disabled_reason or None,
                "ccu_balance": state.ccu_balance,
                "ccu_rate_per_hour": state.ccu_rate,
                "ccu_checked_at": state.ccu_checked_at,
                "cu_day": state.cu_day or None,
                "cu_used_today": state.cu_used_today,
                "starts_today": state.starts_today,
            }
        )
        return view

    async def close(self) -> None:
        await self._client.aclose()


@dataclass(frozen=True)
class ProbeReport:
    """What the readiness probe found."""

    # Each one keeps the session from READY.
    problems: list[str]
    # Each pinned model's digest in /api/tags (those it lists).
    digests: dict[str, str] = field(default_factory=dict)
    # Each pinned model's {"size", "size_vram"} in /api/ps (those it lists).
    vram: dict[str, dict[str, int]] = field(default_factory=dict)


class _ProbeFailed(Exception):
    """One probe call did not answer 200 with JSON."""


def _digest_problem(model: str, actual: str, pinned: str) -> str:
    return f"{model} digest {actual[:12]}, expected {pinned[:12]}"


def _gpu_problem(model: str, sizes: dict[str, int]) -> str:
    in_vram, size = protocol.mib(sizes.get("size_vram", 0)), protocol.mib(sizes.get("size", 0))
    return f"{model} not fully on the GPU ({in_vram} of {size} MiB in VRAM)"


def _tunnel_kind(tunnel_url: str, named_host: str) -> str | None:
    """"named", "quick" or None, never the URL itself."""
    if not tunnel_url:
        return None
    host = urlsplit(tunnel_url).hostname or ""
    return "named" if named_host and host == named_host else "quick"


def _loaded_sizes(payload: Any) -> dict[str, dict[str, int]]:
    """``/api/ps``: each loaded model (by name and by model) → its sizes."""
    loaded: dict[str, dict[str, int]] = {}
    for entry in payload.get("models", []) if isinstance(payload, dict) else []:
        if isinstance(entry, dict):
            sizes = {key: protocol.model_size(entry.get(key)) for key in ("size", "size_vram")}
            loaded.update((value, sizes) for key in ("name", "model") if isinstance(value := entry.get(key), str))
    return loaded
