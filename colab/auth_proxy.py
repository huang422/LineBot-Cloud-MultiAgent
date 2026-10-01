"""The bearer-token proxy between the tunnel and Ollama, and the VM's watchdog (ADR-0010).

cloudflared forwards the public tunnel to 127.0.0.1:8081, where this proxy:

- checks ``Authorization: Bearer <COLAB_PROXY_TOKEN>`` in constant time
  (401 otherwise);
- serves only ``ROUTES`` (plus ``OPENCODE_ROUTES`` when OPENCODE_ENABLED);
  everything else, pull, push, delete, create, copy and blobs included, is 404;
- forwards without Authorization, Origin, Cookie or hop-by-hop headers, with
  ``Host: 127.0.0.1:11434`` (Ollama on loopback refuses any other host);
- streams the answer back as it comes, caps request bodies at 16 MB and
  upstream calls at 150 s;
- aborts the Ollama call when the client goes away (cloudflared closes its
  connection when the bot gives up): ``serve`` runs the app with
  ``handler_cancellation``, so the handler is cancelled, closes the
  upstream connection, and Ollama stops generating; the log says
  ``cancelled after N s``;
- counts only inference (``ACTIVITY_PATHS``) as activity, never the health
  and readiness polls, so polling cannot keep the VM alive.

The watchdog, every 30 s, is the second guard behind the bot's controller:
idle for longer than IDLE_TTL + 120 s, older than MAX_SESSION + 120 s, or
Ollama down after one restart, and the VM sends a signed ``stopping`` event,
releases itself (``POST http://$TBE_RUNTIME_ADDR/unassign``) and exits.
Nothing that fails ends that guard: a check that raises is logged and the
next one runs; WATCHDOG_MAX_ERRORS in a row, a stop that raises, or the
watchdog task itself dying release the VM with the bare self-unassign (no
state.json, no event) and exit.

Bootstrap starts ``python auth_proxy.py`` with the session settings in its
environment (``register.BOOTSTRAP_ENV_KEYS`` and SESSION_STARTED_AT). The
helpers above ``build_app`` are pure; aiohttp is imported only to serve.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

if __package__:
    from . import bootstrap, register
else:  # run as /content/auth_proxy.py, with its siblings beside it
    import bootstrap
    import register

logger = logging.getLogger("colab.auth_proxy")

LISTEN_HOST = bootstrap.PROXY_HOST
LISTEN_PORT = bootstrap.PROXY_PORT
UPSTREAM_URL = bootstrap.OLLAMA_URL
UPSTREAM_HOST_HEADER = bootstrap.OLLAMA_ADDR
MAX_BODY_BYTES = 16 * 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 150.0
UPSTREAM_CONNECT_SECONDS = 5.0
WATCHDOG_INTERVAL_SECONDS = 30.0
WATCHDOG_GRACE_SECONDS = 120
# Checks that raise in a row before the watchdog releases the VM directly.
WATCHDOG_MAX_ERRORS = 3

ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/version"),
        ("HEAD", "/api/version"),
        ("GET", "/api/ps"),
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
        ("POST", "/api/chat"),
        ("POST", "/v1/systemone"),
    }
)
# Ollama's OpenAI-compatible API, for opencode (colab/README.md).
OPENCODE_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {("POST", "/v1/chat/completions"), ("GET", "/v1/models")}
)
ACTIVITY_PATHS: frozenset[str] = frozenset({"/api/chat", "/v1/systemone", "/v1/chat/completions"})

HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
# Never forwarded to Ollama: credentials, browser context, and what the
# client connection alone decides (aiohttp sets its own length).
_DROPPED_REQUEST_HEADERS = HOP_BY_HOP | {"authorization", "origin", "cookie", "host", "content-length", "expect"}


# --- pure helpers ------------------------------------------------------------


def route_allowed(method: str, path: str, *, opencode_enabled: bool) -> bool:
    route = (method.upper(), path)
    return route in ROUTES or (opencode_enabled and route in OPENCODE_ROUTES)


def counts_as_activity(method: str, path: str) -> bool:
    """Inference only: health and readiness polls must not keep the VM alive."""
    return method.upper() == "POST" and path in ACTIVITY_PATHS


def bearer_ok(header: str | None, token: str) -> bool:
    """``Authorization: Bearer <token>``, compared in constant time."""
    if not header or not token:
        return False
    scheme, _, credentials = header.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(credentials.strip().encode(), token.encode())


def _connection_listed(pairs: Iterable[tuple[str, str]]) -> set[str]:
    """Header names the Connection header marks hop-by-hop."""
    listed: set[str] = set()
    for name, value in pairs:
        if name.lower() == "connection":
            listed.update(part.strip().lower() for part in value.split(",") if part.strip())
    return listed


def forward_request_headers(
    pairs: Iterable[tuple[str, str]], host_header: str = UPSTREAM_HOST_HEADER
) -> list[tuple[str, str]]:
    pairs = list(pairs)
    dropped = _DROPPED_REQUEST_HEADERS | _connection_listed(pairs)
    kept = [(name, value) for name, value in pairs if name.lower() not in dropped]
    return kept + [("Host", host_header)]


def forward_response_headers(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    pairs = list(pairs)
    dropped = HOP_BY_HOP | _connection_listed(pairs)
    return [(name, value) for name, value in pairs if name.lower() not in dropped]


class Activity:
    """When the last inference request began or ended, and how many are running."""

    def __init__(self, now: float) -> None:
        self.last = now
        self.in_flight = 0

    def begin(self, now: float) -> None:
        self.in_flight += 1
        self.last = max(self.last, now)

    def end(self, now: float) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.last = max(self.last, now)

    def idle_for(self, now: float) -> float:
        """0 while a request runs, else the seconds since the last one."""
        return 0.0 if self.in_flight else max(0.0, now - self.last)


OK, RESTART, STOP = "ok", "restart", "stop"


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str = ""


def watchdog_decision(
    now: float,
    *,
    started_at: float,
    idle_for: float,
    idle_ttl: int,
    max_session: int,
    upstream_ok: bool,
    restarts: int,
    grace: int = WATCHDOG_GRACE_SECONDS,
) -> Decision:
    """What the watchdog does on this tick. The limits get *grace* on top: the
    bot's controller releases the VM at the limits themselves, and this only
    steps in when it has not."""
    age = now - started_at
    if age > max_session + grace:
        return Decision(STOP, f"session is {int(age)} s old, past MAX_SESSION {max_session} s + {grace} s")
    if idle_for > idle_ttl + grace:
        return Decision(STOP, f"idle for {int(idle_for)} s, past IDLE_TTL {idle_ttl} s + {grace} s")
    if not upstream_ok:
        if restarts == 0:
            return Decision(RESTART, "Ollama is not answering")
        return Decision(STOP, "Ollama is down after one restart")
    return Decision(OK)


@dataclass(frozen=True)
class ProxyConfig:
    proxy_token: str = field(repr=False)
    listen_host: str = LISTEN_HOST
    listen_port: int = LISTEN_PORT
    upstream: str = UPSTREAM_URL
    upstream_host_header: str = UPSTREAM_HOST_HEADER
    opencode_enabled: bool = False
    max_body_bytes: int = MAX_BODY_BYTES
    upstream_timeout: float = UPSTREAM_TIMEOUT_SECONDS
    idle_ttl: int = 900
    max_session: int = 39600
    started_at: float = 0.0
    watchdog_interval: float = WATCHDOG_INTERVAL_SECONDS

    @classmethod
    def from_env(cls, env: bootstrap.BootstrapEnv, *, started_at: float) -> ProxyConfig:
        return cls(
            proxy_token=env.proxy_token,
            opencode_enabled=env.opencode_enabled,
            idle_ttl=env.idle_ttl,
            max_session=env.max_session,
            started_at=started_at,
        )


class Watchdog:
    """Runs ``watchdog_decision`` every ``config.watchdog_interval`` seconds.

    *stop* is the orderly release (:func:`release_session`); *unassign* is
    the bare self-unassign, the last resort when *stop* raises or the
    checks keep failing, so it shares none of their parts.
    """

    def __init__(
        self,
        config: ProxyConfig,
        activity: Activity,
        *,
        check_upstream: Callable[[], Awaitable[bool]],
        restart_upstream: Callable[[], Awaitable[bool]],
        stop: Callable[[str], Awaitable[bool]],
        unassign: Callable[[], Awaitable[bool]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._activity = activity
        self._check_upstream = check_upstream
        self._restart_upstream = restart_upstream
        self._stop = stop
        self._unassign = unassign
        self._clock = clock
        self.restarts = 0
        # Checks that raised in a row.
        self.errors = 0
        # Whether Colab accepted the self-unassign; None until the watchdog stops.
        self.released: bool | None = None

    async def tick(self) -> Decision:
        now = self._clock()
        upstream_ok = await self._check_upstream()
        decision = watchdog_decision(
            now,
            started_at=self._config.started_at,
            idle_for=self._activity.idle_for(now),
            idle_ttl=self._config.idle_ttl,
            max_session=self._config.max_session,
            upstream_ok=upstream_ok,
            restarts=self.restarts,
        )
        if decision.action == RESTART:
            self.restarts += 1
            logger.warning("watchdog: %s; restarting it once", decision.reason)
            if not await self._restart_upstream():
                decision = Decision(STOP, "Ollama did not come back after a restart")
        if decision.action == STOP:
            logger.warning("watchdog: %s; releasing the VM", decision.reason)
            try:
                self.released = await self._stop(decision.reason)
            except Exception as exc:
                await self.last_resort(f"the release failed ({type(exc).__name__})")
        return decision

    async def last_resort(self, reason: str) -> None:
        """The bare self-unassign; never raises."""
        logger.error("watchdog: %s; releasing the VM directly", reason)
        try:
            self.released = await self._unassign()
        except Exception as exc:
            logger.error("watchdog: the self-unassign failed too: %s", type(exc).__name__)
            self.released = False

    async def run(self, done: asyncio.Event) -> None:
        """Ticks until it stops the session (then sets *done*) or *done* is set.

        A tick that raises is logged; WATCHDOG_MAX_ERRORS in a row release
        the VM through :meth:`last_resort` and set *done*.
        """
        while not done.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(done.wait(), self._config.watchdog_interval)
            if done.is_set():
                return
            try:
                decision = await self.tick()
            except Exception as exc:
                self.errors += 1
                logger.error(
                    "watchdog: check %d of %d in a row failed: %s", self.errors, WATCHDOG_MAX_ERRORS, type(exc).__name__
                )
                if self.errors >= WATCHDOG_MAX_ERRORS:
                    await self.last_resort(f"{self.errors} checks in a row failed")
                    done.set()
                continue
            self.errors = 0
            if decision.action == STOP:
                done.set()


def release_session(
    env: bootstrap.BootstrapEnv,
    reason: str,
    *,
    state: bootstrap.StateFile,
    ops: bootstrap.Ops,
) -> bool:
    """The watchdog's stop: state.json, the signed ``stopping`` event, then the
    self-unassign (True when Colab answered 200)."""
    state.write("stopping", reason)
    bootstrap.send_event_quietly(env, "stopping", step="watchdog", error=reason, send=ops.http, now=ops.now)
    return bootstrap.self_unassign(ops.environ(), send=ops.http)


# --- the server --------------------------------------------------------------


def build_app(config: ProxyConfig, activity: Activity, *, clock: Callable[[], float] = time.time) -> Any:
    """The aiohttp application (aiohttp is imported here, not at module import)."""
    from aiohttp import ClientError, ClientSession, ClientTimeout, web
    from multidict import CIMultiDict

    session_key = web.AppKey("upstream_session", ClientSession)

    async def upstream_session(app: web.Application):
        timeout = ClientTimeout(total=config.upstream_timeout, connect=UPSTREAM_CONNECT_SECONDS)
        async with ClientSession(timeout=timeout, auto_decompress=False, trust_env=False) as session:
            app[session_key] = session
            yield

    def error(status: int, message: str, headers: Mapping[str, str] | None = None) -> web.Response:
        return web.json_response({"error": message}, status=status, headers=headers)

    async def handle(request: web.Request) -> web.StreamResponse:
        method, path = request.method, request.path
        if not bearer_ok(request.headers.get("Authorization"), config.proxy_token):
            return error(401, "unauthorized", {"WWW-Authenticate": "Bearer"})
        if not route_allowed(method, path, opencode_enabled=config.opencode_enabled):
            return error(404, "not found")
        if request.content_length is not None and request.content_length > config.max_body_bytes:
            return error(413, "request body too large")
        body = await request.read()  # a chunked body over client_max_size raises 413
        tracked = counts_as_activity(method, path)
        started = time.monotonic()
        if tracked:
            activity.begin(clock())
        response: web.StreamResponse | None = None
        try:
            async with request.app[session_key].request(
                method,
                config.upstream + path,
                headers=CIMultiDict(forward_request_headers(request.headers.items(), config.upstream_host_header)),
                data=body or None,
                allow_redirects=False,
            ) as upstream:
                response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
                for name, value in forward_response_headers(upstream.headers.items()):
                    response.headers.add(name, value)
                await response.prepare(request)
                if method != "HEAD":
                    async for chunk in upstream.content.iter_any():
                        await response.write(chunk)
                await response.write_eof()
                if tracked:
                    logger.info("%s %s -> %s in %.1f s", method, path, upstream.status, time.monotonic() - started)
                return response
        except (TimeoutError, ClientError) as exc:
            if response is not None and response.prepared:
                # Too late for a status: dropping the connection tells the client.
                logger.warning("%s %s cut off: %s", method, path, type(exc).__name__)
                raise
            logger.warning("%s %s failed upstream: %s", method, path, type(exc).__name__)
            if isinstance(exc, TimeoutError):
                return error(504, "upstream timed out")
            return error(502, "upstream unavailable")
        except asyncio.CancelledError:
            # The client hung up (serve()'s handler_cancellation): leaving the
            # ``async with`` above closed the Ollama connection, which stops it.
            elapsed = time.monotonic() - started
            logger.info("%s %s cancelled after %.1f s: the client went away", method, path, elapsed)
            raise
        finally:
            if tracked:
                activity.end(clock())

    app = web.Application(client_max_size=config.max_body_bytes)
    app.cleanup_ctx.append(upstream_session)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


async def serve(
    config: ProxyConfig,
    *,
    check_upstream: Callable[[], Awaitable[bool]],
    restart_upstream: Callable[[], Awaitable[bool]],
    stop: Callable[[str], Awaitable[bool]],
    unassign: Callable[[], Awaitable[bool]],
    clock: Callable[[], float] = time.time,
) -> int:
    """Serves until the watchdog releases the VM or SIGTERM; 0 unless the self-unassign failed.

    Should the watchdog task itself die, the VM is released through its
    last resort and serving ends: a proxy without its watchdog would keep
    the runtime billing.
    """
    from aiohttp import web

    activity = Activity(clock())
    runner = web.AppRunner(build_app(config, activity, clock=clock), handler_cancellation=True, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, config.listen_host, config.listen_port).start()
    logger.info("auth proxy on %s:%s -> %s", config.listen_host, config.listen_port, config.upstream)
    done = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, done.set)
    watchdog = Watchdog(
        config,
        activity,
        check_upstream=check_upstream,
        restart_upstream=restart_upstream,
        stop=stop,
        unassign=unassign,
        clock=clock,
    )
    task = asyncio.create_task(watchdog.run(done))
    stopped = asyncio.create_task(done.wait())
    try:
        await asyncio.wait({task, stopped}, return_when=asyncio.FIRST_COMPLETED)
        if task.done() and not done.is_set():
            error = None if task.cancelled() else task.exception()
            await watchdog.last_resort(f"the watchdog died ({type(error).__name__ if error else 'cancelled'})")
    finally:
        for pending in (task, stopped):
            pending.cancel()
        await asyncio.gather(task, stopped, return_exceptions=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.remove_signal_handler(sig)
        await runner.cleanup()
    return 1 if watchdog.released is False else 0


def main(environ: Mapping[str, str] | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    environ = os.environ if environ is None else environ
    try:
        env = bootstrap.load_env(environ)
    except bootstrap.EnvError as exc:
        logger.error("auth proxy: %s", exc)
        return 2
    bootstrap.install_redaction(env.secrets())
    try:
        started_at = float(environ.get("SESSION_STARTED_AT", ""))
    except ValueError:
        started_at = time.time()
    paths = bootstrap.Paths()
    ops = bootstrap.Ops()
    versions = register.load_versions(paths.versions)
    state = bootstrap.StateFile(paths.state_file, secrets=env.secrets())
    config = ProxyConfig.from_env(env, started_at=started_at)

    async def check_upstream() -> bool:
        return await asyncio.to_thread(lambda: bootstrap.ollama_version(ops) is not None)

    async def restart_upstream() -> bool:
        return await asyncio.to_thread(bootstrap.restart_upstream, env, versions, paths=paths, ops=ops)

    async def stop(reason: str) -> bool:
        return await asyncio.to_thread(release_session, env, reason, state=state, ops=ops)

    async def unassign() -> bool:
        return await asyncio.to_thread(bootstrap.self_unassign, ops.environ(), send=ops.http)

    return asyncio.run(
        serve(config, check_upstream=check_upstream, restart_upstream=restart_upstream, stop=stop, unassign=unassign)
    )


if __name__ == "__main__":
    sys.exit(main())
