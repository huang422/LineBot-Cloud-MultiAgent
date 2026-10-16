"""Sets up one Colab VM for the bot, then leaves it to the auth proxy (ADR-0010).

The bot's launcher uploads ``colab.VM_FILES`` to /content, writes
/content/.bootstrap.env (``register.BOOTSTRAP_ENV_KEYS``) through a silent
kernel execute and starts ``python /content/bootstrap.py`` detached
(``start_new_session``, output in /content/bootstrap.log). The steps, each
recorded in /content/state.json as ``{step, ts, error}`` (never a secret; the
controller's heartbeat cell prints its first 300 characters):

- ``env``: read and delete the env file, then take /content/bootstrap.lock
  (the auth proxy inherits it, so a second bootstrap on this VM exits quietly);
- ``python``: the auth proxy's aiohttp and its dependencies, exactly as
  ``proxy-requirements.lock`` pins them: every wheel sha256-checked
  (``--require-hashes``), nothing resolved at boot (``--no-deps``), nothing
  built from source (``--only-binary=:all:``);
- ``zstd``, ``gpu`` (nvidia-smi: at least 38000 MiB and driver 550),
  ``ollama_install`` (the tarball versions.json names, its size and
  sha256 checked, ``zstd -dc | tar`` into /usr: the launcher wrote that
  entry, the latest release or the pinned fallback), ``ollama_start`` (``ollama serve`` with
  OLLAMA_SERVE_ENV), ``pull`` (both tags, 3 tries; the models follow their
  tags, so each digest is only logged, beside the last verified one: the
  chat model must pull, a router that did not is logged at ERROR and boots
  on, see register.py) and ``warmup`` (load qwen3.8 first, then the router
  (clef-flash) with one tiny /v1/systemone call, then record what each got
  in VRAM: /api/ps and nvidia-smi's memory.used go to state.json and the
  ready event's ``gpu``; a router that did not load or is not wholly in
  VRAM is logged at ERROR and boots on, and one whose loading evicted
  qwen3.8 is unloaded and qwen3.8 loaded again alone). Why the router is
  missing, if it is, goes to the bot in the ready event's ``error``.
  With DRYRUN=1 all of these become
  ``stub_start``: a stub upstream on the Ollama port that answers the
  readiness probe with canned values, so the whole control path runs on a
  CPU runtime;
- ``proxy``: auth_proxy.py on 127.0.0.1:8081, checked with and without the
  bearer token;
- ``probe``: ``register.probe_problems`` through the proxy, the same check
  the bot makes through the tunnel;
- ``tunnel``: the pinned cloudflared (size and sha256 checked), a quick
  tunnel (host from the metrics server's /quicktunnel) or a named one
  (TUNNEL_TOKEN in its environment, never in argv; its public host is
  TUNNEL_HOST), retried with ``--protocol http2`` when not ready in 90 s;
  ``tunnel_check`` then calls /api/version through it (a failure there is
  only logged);
- ``register``: the signed ``ready`` event, retried on 5xx and 422 (the bot
  could not resolve a brand-new tunnel host yet); the VM is then ``ready``.

Any failure writes the error to state.json, sends a signed ``failed`` event,
stops what this run started and releases the VM (a raw POST to
``http://$TBE_RUNTIME_ADDR/unassign``; ``google.colab.runtime.unassign()``
needs a kernel and fails in a detached process). Secrets never reach argv,
logs or state.json: the proxy gets them in its environment, cloudflared only
its tunnel token.

``python bootstrap.py --stub-upstream`` runs the DRYRUN stub itself.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import http.client
import importlib.metadata
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__:
    from . import register
else:  # run as /content/bootstrap.py, with register.py beside it
    import register

logger = logging.getLogger("colab.bootstrap")

HERE = Path(__file__).resolve().parent
OLLAMA_ADDR = "127.0.0.1:11434"
OLLAMA_URL = f"http://{OLLAMA_ADDR}"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8081
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
METRICS_ADDR = "127.0.0.1:20241"
METRICS_URL = f"http://{METRICS_ADDR}"

# The auth proxy's packages, aiohttp and its whole dependency closure, with
# every file's sha256 (the same pins and hashes as the bot image's lock):
#   uv pip compile colab/proxy-requirements.in --universal --python-version 3.11 \
#     --generate-hashes -o colab/proxy-requirements.lock
# The proxy holds the session's keys, so nothing it runs on is resolved or
# built at boot.
PROXY_REQUIREMENTS = "proxy-requirements.lock"
PIP_LOCKED_FLAGS: tuple[str, ...] = ("--require-hashes", "--no-deps", "--only-binary=:all:")
_LOCK_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)\s*(;)?")

# `ollama serve` on the VM. The context length is set once here: a request
# that sends num_ctx (or draft_num_predict) reloads the model. OLLAMA_ORIGINS
# stays unset so no browser origin passes Ollama's CORS check.
OLLAMA_SERVE_ENV: dict[str, str] = {
    "OLLAMA_HOST": OLLAMA_ADDR,
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_CONTEXT_LENGTH": "32768",
    "OLLAMA_MAX_LOADED_MODELS": "2",
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_NO_CLOUD": "1",
    "OLLAMA_MAX_QUEUE": "8",
    "OLLAMA_LOAD_TIMEOUT": "10m",
}
_UNSET_FOR_OLLAMA = ("OLLAMA_ORIGINS",)
# What a loaded but busy Ollama answers once its queue is full (two spaces).
OLLAMA_BUSY_ERROR = "server busy, please try again.  maximum pending requests exceeded"
INFERENCE_PATHS: tuple[str, ...] = ("/api/chat", "/v1/systemone", "/v1/chat/completions")

MAX_STATE_ERROR_CHARS = 200
# The controller's heartbeat cell prints this much of state.json.
MAX_STATE_CHARS = 300
OLLAMA_START_SECONDS = 60
PULL_ATTEMPTS = 3
PULL_BACKOFF_SECONDS = (15, 45)
PULL_TIMEOUT_SECONDS = 1800
LOAD_TIMEOUT_SECONDS = 660
PROXY_START_SECONDS = 30
TUNNEL_READY_SECONDS = 90
TUNNEL_CHECK_SECONDS = 60
REGISTER_ATTEMPTS = 4
REGISTER_BACKOFF_SECONDS = (5, 10, 20)
# 422: the bot could not resolve the tunnel host (yet); 5xx: its probe or
# state store failed. Any other 4xx is final.
REGISTER_RETRY_STATUSES = frozenset({422})

_HOSTNAME = re.compile(r"^(?=.{1,253}$)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+$")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})
_DOWNLOAD_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class BootstrapError(Exception):
    """A step failed; the message (secrets redacted) goes to state.json and the bot."""


class DownloadError(BootstrapError):
    """A pinned download failed, or did not match its size or sha256."""


class EnvError(ValueError):
    """The bootstrap env is unusable. Messages name keys, never values."""


# --- paths -------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """Where things live on the VM; tests point them at a temporary directory."""

    content: Path = Path("/content")
    code: Path = HERE
    usr: Path = Path("/usr")
    cloudflared: Path = Path("/usr/local/bin/cloudflared")
    cloudflared_config_dir: Path = field(default_factory=lambda: Path.home() / ".cloudflared")

    @property
    def env_file(self) -> Path:
        return self.content / ".bootstrap.env"

    @property
    def state_file(self) -> Path:
        return self.content / "state.json"

    @property
    def lock_file(self) -> Path:
        return self.content / "bootstrap.lock"

    @property
    def upstream_pid(self) -> Path:
        return self.content / "upstream.pid"

    @property
    def ollama_archive(self) -> Path:
        return self.content / "ollama-linux-amd64.tar.zst"

    @property
    def ollama_bin(self) -> Path:
        return self.usr / "bin" / "ollama"

    @property
    def versions(self) -> Path:
        return self.code / "versions.json"

    @property
    def proxy_requirements(self) -> Path:
        return self.code / PROXY_REQUIREMENTS

    def log(self, name: str) -> Path:
        return self.content / f"{name}.log"


# --- the env file ------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapEnv:
    """The launcher's settings for this session (register.BOOTSTRAP_ENV_KEYS)."""

    reg_key: str = field(repr=False)
    proxy_token: str = field(repr=False)
    bot_url: str
    nbh: str
    idle_ttl: int = 900
    max_session: int = 39600
    tunnel_mode: str = "quick"
    tunnel_token: str = field(default="", repr=False)
    tunnel_host: str = ""
    opencode_enabled: bool = False
    dryrun: bool = False

    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.reg_key, self.proxy_token, self.tunnel_token) if s)

    def proxy_env(self) -> dict[str, str]:
        """What the auth proxy needs, as the same keys: no tunnel settings, so
        the named-tunnel token stays with cloudflared alone."""
        return {
            "COLAB_REG_KEY": self.reg_key,
            "COLAB_PROXY_TOKEN": self.proxy_token,
            "BOT_URL": self.bot_url,
            "NBH": self.nbh,
            "IDLE_TTL": str(self.idle_ttl),
            "MAX_SESSION": str(self.max_session),
            "OPENCODE_ENABLED": "1" if self.opencode_enabled else "0",
            "DRYRUN": "1" if self.dryrun else "0",
        }


def _flag(values: Mapping[str, str], key: str) -> bool:
    text = values.get(key, "").strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise EnvError(f"{key} is not a boolean (1/0, true/false)")


def _seconds(values: Mapping[str, str], key: str, default: int) -> int:
    text = values.get(key, "").strip()
    if not text:
        return default
    try:
        number = int(text)
    except ValueError:
        raise EnvError(f"{key} is not a whole number of seconds") from None
    if number <= 0:
        raise EnvError(f"{key} must be positive")
    return number


def load_env(values: Mapping[str, str]) -> BootstrapEnv:
    """Checks the launcher's values; raises EnvError naming the first bad key."""

    def required(key: str) -> str:
        text = values.get(key, "").strip()
        if not text:
            raise EnvError(f"{key} is missing")
        return text

    reg_key = required("COLAB_REG_KEY")
    if not re.fullmatch(r"[0-9a-f]{64}", reg_key):
        raise EnvError("COLAB_REG_KEY is not 64 lowercase hex characters")
    proxy_token = required("COLAB_PROXY_TOKEN")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", proxy_token):
        raise EnvError("COLAB_PROXY_TOKEN is not 43 base64url characters")
    bot_url = required("BOT_URL").rstrip("/")
    parts = urllib.parse.urlsplit(bot_url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise EnvError("BOT_URL is not an https URL without credentials, query or fragment")
    nbh = required("NBH")
    try:
        uuid.UUID(nbh)
    except ValueError:
        raise EnvError("NBH is not a UUID") from None
    tunnel_mode = values.get("TUNNEL_MODE", "").strip().lower() or "quick"
    if tunnel_mode not in ("quick", "named"):
        raise EnvError("TUNNEL_MODE is not quick or named")
    tunnel_token = values.get("TUNNEL_TOKEN", "").strip()
    if tunnel_mode == "named" and not tunnel_token:
        raise EnvError("TUNNEL_TOKEN is missing for a named tunnel")
    tunnel_host = values.get("TUNNEL_HOST", "").strip().lower()
    if tunnel_mode == "named" and not tunnel_host:
        # A named tunnel cannot report its public host, and the bot only
        # accepts a ready event that names it (COLAB_TUNNEL_HOST).
        raise EnvError("TUNNEL_HOST is missing for a named tunnel")
    if tunnel_host and not _HOSTNAME.match(tunnel_host):
        raise EnvError("TUNNEL_HOST is not a host name")
    return BootstrapEnv(
        reg_key=reg_key,
        proxy_token=proxy_token,
        bot_url=bot_url,
        nbh=nbh,
        idle_ttl=_seconds(values, "IDLE_TTL", 900),
        max_session=_seconds(values, "MAX_SESSION", 39600),
        tunnel_mode=tunnel_mode,
        tunnel_token=tunnel_token if tunnel_mode == "named" else "",
        tunnel_host=tunnel_host if tunnel_mode == "named" else "",
        opencode_enabled=_flag(values, "OPENCODE_ENABLED"),
        dryrun=_flag(values, "DRYRUN"),
    )


def read_env_file(path: Path) -> str | None:
    """The env file's text (None when there is none); the file is deleted either way."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    finally:
        path.unlink(missing_ok=True)


def acquire_lock(path: Path) -> int | None:
    """An fd holding an exclusive flock on *path*, or None when another process holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


# --- redaction and state -----------------------------------------------------


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


class RedactingFilter(logging.Filter):
    """Belt and braces: nothing logs a secret, and this removes one if it does."""

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message, self._secrets)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


def install_redaction(secrets: Iterable[str]) -> None:
    redacting = RedactingFilter(secrets)
    for handler in logging.getLogger().handlers:
        handler.addFilter(redacting)


class StateFile:
    """/content/state.json: ``{step, ts, error}``, and ``vram`` once the
    models are warm (:meth:`Bootstrap.record_vram`); replaced atomically and
    never longer than MAX_STATE_CHARS (a long error is cut shorter)."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time, secrets: Iterable[str] = ()):
        self.path = path
        self._clock = clock
        self._secrets = tuple(secrets)
        self.vram: dict[str, Any] = {}

    def write(self, step: str, error: str | None = None) -> None:
        if error is not None:
            error = redact(error, self._secrets)[:MAX_STATE_ERROR_CHARS]
        payload: dict[str, Any] = {"step": step, "ts": int(self._clock()), "error": error}
        if self.vram:
            payload["vram"] = self.vram
        text = json.dumps(payload)
        if len(text) > MAX_STATE_CHARS and error:
            # Cut the error's tail until the JSON fits (a character takes
            # 1 to 12 characters of it: "\u..." escapes).
            over, keep = len(text) - MAX_STATE_CHARS, len(error)
            while keep and over > 0:
                keep -= 1
                over -= len(json.dumps(error[keep])) - 2
            payload["error"] = error[:keep]
            text = json.dumps(payload)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.path)

    def read(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None


# --- side effects ------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    url: str,
    dest: Path,
    sha256: str,
    size: int,
    *,
    open_url: Callable[..., Any] = _DOWNLOAD_OPENER.open,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 60.0,
) -> None:
    """Puts *url* at *dest* only if it is *size* bytes with sha256 *sha256*.
    A file already there that matches is kept. Network errors and a body cut
    short are retried; a body longer than *size* is cut off at once (it
    never fills the disk) and, like a wrong hash, is not retried."""
    if dest.exists() and dest.stat().st_size == size and file_sha256(dest) == sha256:
        return
    name = dest.name
    part = dest.with_name(name + ".part")
    for attempt in range(1, attempts + 1):
        digest = hashlib.sha256()
        received = 0
        try:
            with open_url(url, timeout=timeout) as response, open(part, "wb") as out:
                while chunk := response.read(1 << 20):
                    received += len(chunk)
                    if received > size:
                        raise DownloadError(f"{name} is longer than the pinned {size} bytes")
                    digest.update(chunk)
                    out.write(chunk)
            if received < size:
                # The connection ended early: a network failure, retried.
                raise http.client.IncompleteRead(b"", size - received)
        except DownloadError:
            part.unlink(missing_ok=True)
            raise
        except (OSError, http.client.HTTPException) as exc:
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise DownloadError(f"downloading {name} failed {attempts} times: {type(exc).__name__}") from exc
            logger.warning("downloading %s failed (%s); retrying", name, type(exc).__name__)
            sleep(10 * attempt)
            continue
        got = digest.hexdigest()
        if got != sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(f"{name} has sha256 {got[:12]}, not the pinned {sha256[:12]}")
        os.replace(part, dest)
        return


class Ops:
    """Every side effect bootstrap and the proxy's watchdog have; tests replace it."""

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def environ(self) -> dict[str, str]:
        return dict(os.environ)

    def package_version(self, name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def run(self, cmd: list[str], *, timeout: float, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def spawn(
        self,
        cmd: list[str],
        *,
        env: Mapping[str, str],
        log_path: Path,
        pass_fds: Iterable[int] = (),
    ) -> subprocess.Popen:
        """Starts *cmd* detached (its own session), output appended to *log_path*."""
        with open(log_path, "ab") as log:
            return subprocess.Popen(
                cmd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=tuple(pass_fds),
            )

    def http(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> tuple[int, bytes]:
        return register.http_request(method, url, body, headers, timeout)

    def download(self, url: str, dest: Path, sha256: str, size: int) -> None:
        download_verified(url, dest, sha256, size, sleep=self.sleep)

    def terminate(self, proc: subprocess.Popen) -> None:
        """Stops a process this run started, with its process group."""
        if proc.poll() is not None:
            return
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _signal_group(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)

    def kill_pidfile(self, path: Path, grace: float = 10.0) -> None:
        """Stops the process group recorded in *path* (the upstream a restart replaces)."""
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return
        if not _signal_group(pid, signal.SIGTERM):
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not _signal_group(pid, 0):
                return
            time.sleep(0.5)
        _signal_group(pid, signal.SIGKILL)


def _signal_group(pid: int, sig: int) -> bool:
    """Signals the process group led by *pid*; False once it is gone."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)
    return True


def _tail(result: subprocess.CompletedProcess, limit: int = 300) -> str:
    text = (result.stderr or "").strip() or (result.stdout or "").strip()
    return text[-limit:]


def _json(body: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(body or b"null")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def poll(ops: Ops, seconds: float, check: Callable[[], Any], interval: float = 1.0) -> Any:
    """``check()`` until it returns something other than None, for at most *seconds*."""
    deadline = ops.now() + seconds
    while True:
        result = check()
        if result is not None:
            return result
        if ops.now() >= deadline:
            return None
        ops.sleep(interval)


def child_env(base: Mapping[str, str]) -> dict[str, str]:
    """*base* without the bootstrap keys, so no child inherits one by accident."""
    return {k: v for k, v in base.items() if k not in register.BOOTSTRAP_ENV_KEYS}


def upstream_env(base: Mapping[str, str]) -> dict[str, str]:
    env = {k: v for k, v in child_env(base).items() if k not in _UNSET_FOR_OLLAMA}
    env.update(OLLAMA_SERVE_ENV)
    return env


def locked_pins(text: str) -> dict[str, str]:
    """Package → version for a hash lock's pins that apply everywhere.

    A pin behind an environment marker (typing-extensions below Python 3.13)
    is left to pip.
    """
    pins: dict[str, str] = {}
    for line in text.splitlines():
        match = _LOCK_PIN.match(line)
        if match and not match.group(3):
            pins[match.group(1).lower()] = match.group(2)
    return pins


# --- the GPU -----------------------------------------------------------------


def parse_gpu(output: str) -> dict[str, Any]:
    """The first GPU of ``nvidia-smi --query-gpu=name,memory.total,driver_version
    --format=csv,noheader,nounits``: ``{name, vram_mib, driver}``."""
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            try:
                vram = int(float(parts[1]))
            except ValueError:
                break
            return {"name": parts[0], "vram_mib": vram, "driver": parts[2]}
    raise BootstrapError("nvidia-smi printed no GPU")


def parse_memory_used(output: str) -> int | None:
    """MiB from ``nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits``
    (its first GPU); None when it printed none."""
    for line in output.splitlines():
        if line.strip():
            try:
                return int(float(line.strip()))
            except ValueError:
                return None
    return None


def gpu_problems(gpu: Mapping[str, Any], limits: Mapping[str, int]) -> list[str]:
    """Why this GPU cannot run the pinned models; empty when it can.

    VRAM is checked as a number: never match on "80", which an A100 40GB
    also shows (its compute capability is sm_80)."""
    problems = []
    if gpu["vram_mib"] < limits["min_vram_mib"]:
        problems.append(f"{gpu['name']} has {gpu['vram_mib']} MiB, under {limits['min_vram_mib']} MiB")
    try:
        major = int(str(gpu["driver"]).split(".")[0])
    except ValueError:
        major = -1
    if major < limits["min_driver_major"]:
        problems.append(f"NVIDIA driver {gpu['driver']} is older than {limits['min_driver_major']}")
    return problems


# --- the upstream (Ollama, or the DRYRUN stub) -------------------------------


def start_upstream(env: BootstrapEnv, *, paths: Paths, ops: Ops) -> subprocess.Popen:
    """Starts ``ollama serve`` (or the DRYRUN stub) detached and records its pid."""
    if env.dryrun:
        cmd = [sys.executable, str(paths.code / "bootstrap.py"), "--stub-upstream"]
        log = paths.log("stub_upstream")
    else:
        cmd = [str(paths.ollama_bin), "serve"]
        log = paths.log("ollama")
    proc = ops.spawn(cmd, env=upstream_env(ops.environ()), log_path=log)
    paths.upstream_pid.write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc


def ollama_version(ops: Ops, base: str = OLLAMA_URL, headers: Mapping[str, str] | None = None) -> str | None:
    try:
        status, body = ops.http("GET", base + "/api/version", None, headers, 5.0)
    except OSError:
        return None
    if status != 200:
        return None
    version = _json(body).get("version")
    return str(version) if version else None


def models_in_load_order(versions: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """The models, the chat model first. Ollama loads a model beside the
    loaded ones only while it predicts it fits (else it evicts one), and
    qwen3.8 at 32K context is the one that must stay: the router then takes
    what is left (colab/README.md, "VRAM")."""
    return sorted(versions["models"].items(), key=lambda item: item[1]["role"] != "chat")


def warm_up(
    versions: Mapping[str, Any], *, ops: Ops, base: str = OLLAMA_URL, skip: Iterable[str] = ()
) -> list[str]:
    """Loads every model, the chat model first, and keeps it loaded (keep_alive -1).

    The chat model must load (else BootstrapError). A router need not: one
    that does not load is logged at ERROR and left out, and so is one whose
    loading evicted the chat model (Ollama evicts an idle model rather than
    load the next one when it predicts no fit, colab/README.md "VRAM"): it
    is unloaded and the chat model loaded again, alone. The bot then routes
    the session without it. A router in *skip* (it did not pull) is not
    tried. Returns why each router is left out, for the ready event.
    """
    chat = [(model, pin) for model, pin in models_in_load_order(versions) if register.required(pin)]
    skipped = set(skip)
    problems: list[str] = []
    for model, pin in models_in_load_order(versions):
        if register.required(pin):
            _load(model, pin, ops=ops, base=base)
            continue
        if model in skipped:
            continue
        try:
            _load(model, pin, ops=ops, base=base)
        except (BootstrapError, OSError) as exc:
            # Ollama may have evicted the chat model before this load failed.
            logger.error("%s did not load (%s); booting on, the bot will route without it", model, exc)
            problems.append(f"{model} did not load: {exc}")
        running = _running_models(ops, base)
        evicted = [name for name, _ in chat if running is not None and name not in running]
        if not evicted:
            continue
        logger.error(
            "loading %s evicted %s (not enough VRAM for both); unloading %s and booting on without it",
            model, ", ".join(evicted), model,
        )
        problems.append(f"loading {model} evicted {', '.join(evicted)} (not enough VRAM for both); unloaded it")
        _unload(model, ops=ops, base=base)
        for name, chat_pin in chat:
            _load(name, chat_pin, ops=ops, base=base)
    return problems


def _load(model: str, pin: Mapping[str, Any], *, ops: Ops, base: str) -> None:
    """Loads *model* and keeps it loaded; raises BootstrapError when it does not load."""
    if pin["role"] == "chat":
        # An empty message list only loads the model: done_reason "load".
        payload: dict[str, Any] = {"model": model, "messages": [], "keep_alive": -1}
        path = "/api/chat"
    else:
        # The bot's request shape (src/providers/systemone.py build_request):
        # the state an object with its fields, so a warm-up Ollama accepts
        # is one the router's requests pass too.
        payload = {
            "model": model,
            "keep_alive": -1,
            "state": {"message": "warm-up", "has_image": False, "quoted": False, "previous_agent": "none"},
            "questions": {
                "ready": {
                    "type": "noul",
                    "instructions": "Is this a warm-up request?",
                    "criteria": {"false": "No.", "true": "Yes."},
                }
            },
        }
        path = "/v1/systemone"
    status, body = ops.http(
        "POST",
        base + path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
        float(LOAD_TIMEOUT_SECONDS),
    )
    answer = _json(body)
    if status != 200:
        raise BootstrapError(f"loading {model} failed: HTTP {status} {str(answer.get('error', ''))[:200]}")
    if pin["role"] == "chat" and answer.get("done_reason") != "load":
        raise BootstrapError(f"loading {model} answered done_reason {answer.get('done_reason')!r}, not 'load'")
    if pin["role"] == "router" and not isinstance(answer.get("answers"), dict):
        raise BootstrapError(f"{model} gave no /v1/systemone answers")


def _unload(model: str, *, ops: Ops, base: str) -> None:
    """Unloads *model* (``/api/generate`` with no prompt and keep_alive 0,
    which Ollama answers before any capability check). A failure is only
    logged: loading the chat model again evicts an idle router anyway."""
    try:
        status, body = ops.http(
            "POST",
            base + "/api/generate",
            json.dumps({"model": model, "keep_alive": 0}).encode(),
            {"Content-Type": "application/json"},
            60.0,
        )
    except OSError as exc:
        logger.warning("unloading %s failed: %s", model, type(exc).__name__)
        return
    if status != 200 or _json(body).get("done_reason") != "unload":
        logger.warning("unloading %s answered HTTP %s", model, status)


def _running_models(ops: Ops, base: str) -> set[str] | None:
    """The model names ``/api/ps`` lists; None when it does not answer."""
    try:
        status, body = ops.http("GET", base + "/api/ps", None, None, 10.0)
    except OSError:
        return None
    if status != 200:
        return None
    return {
        str(entry[key])
        for entry in _json(body).get("models") or []
        if isinstance(entry, dict)
        for key in ("name", "model")
        if entry.get(key)
    }


def restart_upstream(
    env: BootstrapEnv,
    versions: Mapping[str, Any],
    *,
    paths: Paths | None = None,
    ops: Ops | None = None,
) -> bool:
    """The watchdog's one restart: stop the upstream, start it, reload the models."""
    paths = paths or Paths()
    ops = ops or Ops()
    ops.kill_pidfile(paths.upstream_pid)
    start_upstream(env, paths=paths, ops=ops)
    want = versions["ollama"]["version"]
    if poll(ops, OLLAMA_START_SECONDS, lambda: ollama_version(ops)) != want:
        return False
    if env.dryrun:
        return True
    try:
        warm_up(versions, ops=ops)
    except (BootstrapError, OSError) as exc:
        logger.warning("reloading the models after a restart failed: %s", exc)
        return False
    return True


# --- the tunnel --------------------------------------------------------------


def tunnel_command(cloudflared: Path, mode: str, protocol: str | None = None) -> list[str]:
    """cloudflared's argv; a named tunnel's token comes from TUNNEL_TOKEN, never argv."""
    cmd = [str(cloudflared), "tunnel", "--no-autoupdate", "--metrics", METRICS_ADDR]
    if protocol:
        cmd += ["--protocol", protocol]
    if mode == "quick":
        return cmd + ["--url", PROXY_URL]
    return cmd + ["run"]


def move_cloudflared_config_aside(config_dir: Path) -> list[Path]:
    """A quick tunnel refuses to start while ~/.cloudflared/config.yaml exists."""
    moved = []
    for name in ("config.yaml", "config.yml"):
        path = config_dir / name
        if path.exists():
            target = path.with_name(name + ".disabled")
            os.replace(path, target)
            moved.append(target)
    return moved


# --- registration and release ------------------------------------------------


def send_event_quietly(
    env: BootstrapEnv,
    event: str,
    *,
    step: str | None = None,
    error: str | None = None,
    send: register.Sender = register.http_request,
    now: Callable[[], float] = time.time,
) -> int | None:
    """One signed ``failed`` or ``stopping`` event; returns the status, None when it did not arrive."""
    try:
        body = register.build_event(event, env.nbh, step=step, error=error)
        status = register.post_event(env.bot_url, env.reg_key, body, send=send, now=now, timeout=15.0)
    except (OSError, ValueError) as exc:
        logger.warning("could not send the %s event: %s", event, type(exc).__name__)
        return None
    logger.info("sent the %s event: HTTP %s", event, status)
    return status


def self_unassign(environ: Mapping[str, str], *, send: register.Sender = register.http_request) -> bool:
    """The VM releases itself; True when Colab answered 200."""
    addr = environ.get("TBE_RUNTIME_ADDR", "").strip()
    if not addr:
        logger.error("TBE_RUNTIME_ADDR is not set: this is not a Colab runtime, nothing to release")
        return False
    try:
        status, _ = send("POST", f"http://{addr}/unassign", b"", {}, 30.0)
    except OSError as exc:
        logger.error("self-unassign failed: %s", type(exc).__name__)
        return False
    if status != 200:
        logger.error("self-unassign answered HTTP %s", status)
    return status == 200


# --- the DRYRUN stub upstream ------------------------------------------------


def stub_payloads(versions: Mapping[str, Any]) -> dict[str, Any]:
    """What the stub answers: canned /api/version, /api/ps, /api/tags and
    /api/show bodies built from versions.json (each model at its verified
    digest), shaped like Ollama 0.40.5's, which pass the readiness probe
    (every model loaded fully on the GPU)."""
    context = int(OLLAMA_SERVE_ENV["OLLAMA_CONTEXT_LENGTH"])
    ps, tags, shows = [], [], {}
    for name, pin in versions["models"].items():
        chat = pin["role"] == "chat"
        family = "qwen35" if chat else "clef"
        details = {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": "27.3B" if chat else "9.1B",
            "quantization_level": "Q4_K_M" if chat else "Q8_0",
        }
        # Loaded: the plan's figure for qwen3.8 at 32K; for clef-flash
        # Ollama 0.40.5's own preflight estimate (the 9,657,260,192-byte
        # GGUF + 2 GiB of KV at num_ctx 16384 + the 2 GiB batch allowance of
        # a decision model). Both unmeasured.
        loaded = 25_300_000_000 if chat else 13_952_227_488
        ps.append(
            {
                "name": name,
                "model": name,
                "size": loaded,
                "digest": pin["verified_digest"],
                "details": details,
                "expires_at": "2318-01-01T00:00:00Z",
                "size_vram": loaded,
                # A model's own num_ctx wins over OLLAMA_CONTEXT_LENGTH.
                "context_length": int(pin.get("num_ctx", context)),
            }
        )
        tags.append(
            {
                "name": name,
                "model": name,
                "modified_at": "2026-09-30T00:00:00Z",
                # The manifests' config and layers (clef-flash's: the GGUF,
                # its vision projector, params and config).
                "size": 17_741_860_480 if chat else 10_578_965_269,
                "digest": pin["verified_digest"],
                "details": details,
            }
        )
        show: dict[str, Any] = {
            "details": details,
            "model_info": {},
            # A decision model reports only "decision" (since Ollama 0.35.1).
            "capabilities": ["completion", "vision", "tools", "thinking"] if chat else ["decision"],
            "modified_at": "2026-09-30T00:00:00Z",
        }
        if pin.get("thinking") is not None:
            show["thinking"] = {"values": list(pin["thinking"]), "default": "medium"}
        shows[name] = show
    return {
        "version": {"version": versions["ollama"]["version"]},
        "ps": {"models": ps},
        "tags": {"models": tags},
        "show": shows,
    }


def make_stub_server(address: tuple[str, int], versions: Mapping[str, Any]) -> ThreadingHTTPServer:
    """The DRYRUN upstream. Inference paths answer Ollama's busy 503, so a bot
    that routes a real message here during a dry run falls back to its API
    tiers at once instead of getting a canned reply."""
    payloads = stub_payloads(versions)

    class StubHandler(BaseHTTPRequestHandler):
        server_version = "ollama-dryrun-stub"

        def _send(self, status: int, payload: Any, *, head: bool = False) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _path(self) -> str:
            return urllib.parse.urlsplit(self.path).path

        def do_HEAD(self) -> None:
            if self._path() == "/api/version":
                self._send(200, payloads["version"], head=True)
            else:
                self._send(404, {"error": "404 page not found"}, head=True)

        def do_GET(self) -> None:
            routes = {"/api/version": "version", "/api/ps": "ps", "/api/tags": "tags"}
            key = routes.get(self._path())
            if key is None:
                self._send(404, {"error": "404 page not found"})
            else:
                self._send(200, payloads[key])

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(min(length, 1 << 20)) if length > 0 else b""
            path = self._path()
            if path == "/api/show":
                request = _json(raw)
                name = str(request.get("model") or request.get("name") or "")
                if name in payloads["show"]:
                    self._send(200, payloads["show"][name])
                else:
                    self._send(404, {"error": f"model '{name}' not found"})
            elif path in INFERENCE_PATHS:
                self._send(503, {"error": OLLAMA_BUSY_ERROR})
            else:
                self._send(404, {"error": "404 page not found"})

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("stub: " + format, *args)

    server = ThreadingHTTPServer(address, StubHandler)
    server.daemon_threads = True
    return server


def serve_stub(versions: Mapping[str, Any]) -> int:
    host, _, port = OLLAMA_ADDR.partition(":")
    server = make_stub_server((host, int(port)), versions)
    logger.info("DRYRUN stub upstream on %s", OLLAMA_ADDR)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


# --- the run -----------------------------------------------------------------


def abort(
    env: BootstrapEnv | None,
    step: str,
    error: str,
    *,
    state: StateFile,
    ops: Ops,
    cleanup: Callable[[], None] | None = None,
) -> int:
    """The failure path: state.json, the signed ``failed`` event (when the env
    has a key to sign it), *cleanup*, then the VM releases itself."""
    if env is not None:
        error = redact(error, env.secrets())
    logger.error("step %s failed: %s", step, error)
    state.write(step, error)
    if env is not None:
        send_event_quietly(env, "failed", step=step, error=error, send=ops.http, now=ops.now)
    if cleanup is not None:
        cleanup()
    self_unassign(ops.environ(), send=ops.http)
    return 1


class Bootstrap:
    """One run of the steps in the module docstring."""

    def __init__(
        self,
        env: BootstrapEnv,
        versions: Mapping[str, Any],
        *,
        paths: Paths,
        ops: Ops,
        state: StateFile,
        lock_fd: int | None = None,
    ) -> None:
        self.env = env
        self.versions = versions
        self.paths = paths
        self.ops = ops
        self.state = state
        self.lock_fd = lock_fd
        self.children: list[subprocess.Popen] = []
        self.started_at = ops.now()
        self.step = "start"
        self.gpu: dict[str, Any] | None = None
        self.digests: dict[str, str] = {}
        self.ollama_version = ""
        # Why the router is not serving (it did not pull or load, or is not
        # wholly in VRAM): the ready event's error, for the bot's log.
        self.router_problems: list[str] = []

    # The proxy's bearer header, for the checks this run makes through it.
    @property
    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.env.proxy_token}"}

    def run(self) -> int:
        try:
            self._enter("python")
            self.ensure_python_packages()
            if self.env.dryrun:
                self._enter("stub_start")
                self.start_upstream_and_wait()
            else:
                self._enter("zstd")
                self.ensure_zstd()
                self._enter("gpu")
                self.check_gpu()
                self._enter("ollama_install")
                self.install_ollama()
                self._enter("ollama_start")
                self.start_upstream_and_wait()
                self._enter("pull")
                unpulled = self.pull_models()
                self._enter("warmup")
                self.router_problems += warm_up(self.versions, ops=self.ops, skip=unpulled)
                self.record_vram()
            self._enter("proxy")
            self.start_proxy()
            self._enter("probe")
            self.probe()
            self._enter("tunnel")
            tunnel_url = self.start_tunnel()
            self._enter("tunnel_check")
            self.check_tunnel(tunnel_url)
            self._enter("register")
            self.register_ready(tunnel_url)
        except BootstrapError as exc:
            return self.fail(str(exc))
        except Exception as exc:  # every failure must still release the VM
            return self.fail(f"{type(exc).__name__}: {exc}")
        self.state.write("ready")
        logger.info("ready%s", " (DRYRUN)" if self.env.dryrun else "")
        return 0

    def _enter(self, step: str) -> None:
        self.step = step
        self.state.write(step)
        logger.info("step %s", step)

    def fail(self, error: str) -> int:
        return abort(self.env, self.step, error, state=self.state, ops=self.ops, cleanup=self.stop_children)

    def stop_children(self) -> None:
        for proc in reversed(self.children):
            with contextlib.suppress(OSError):
                self.ops.terminate(proc)
        self.children.clear()

    def _run(self, cmd: list[str], *, timeout: float, env: Mapping[str, str] | None = None, what: str) -> None:
        result = self.ops.run(cmd, timeout=timeout, env=env)
        if result.returncode:
            raise BootstrapError(f"{what} failed (exit {result.returncode}): {_tail(result)}")

    def ensure_python_packages(self) -> None:
        """The proxy's packages as locked, unless the runtime already has them."""
        lock = self.paths.proxy_requirements
        pins = locked_pins(lock.read_text(encoding="utf-8"))
        if "aiohttp" not in pins:
            raise BootstrapError(f"{lock.name} does not pin aiohttp")
        if all(self.ops.package_version(name) == version for name, version in pins.items()):
            return
        self._run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-input",
             *PIP_LOCKED_FLAGS, "-r", str(lock)],
            timeout=300,
            what=f"pip install -r {lock.name}",
        )

    def ensure_zstd(self) -> None:
        """Ollama's linux-amd64 build ships only as .tar.zst (0.35.0 through
        0.40.5 at least), whichever release the launcher picked."""
        if self.ops.which("zstd"):
            return
        env = {**child_env(self.ops.environ()), "DEBIAN_FRONTEND": "noninteractive"}
        install = ["apt-get", "install", "-y", "-qq", "zstd"]
        if self.ops.run(install, timeout=300, env=env).returncode == 0:
            return
        # A fresh image's package lists can be stale: update once, then retry.
        self._run(["apt-get", "update", "-qq"], timeout=300, env=env, what="apt-get update")
        self._run(install, timeout=300, env=env, what="apt-get install zstd")

    def check_gpu(self) -> None:
        if not self.ops.which("nvidia-smi"):
            raise BootstrapError("nvidia-smi is missing: this runtime has no GPU")
        result = self.ops.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            timeout=60,
        )
        if result.returncode:
            raise BootstrapError(f"nvidia-smi failed (exit {result.returncode}): {_tail(result)}")
        gpu = parse_gpu(result.stdout)
        problems = gpu_problems(gpu, self.versions["gpu"])
        if problems:
            raise BootstrapError("; ".join(problems))
        self.gpu = gpu
        logger.info("GPU %s, %s MiB, driver %s", gpu["name"], gpu["vram_mib"], gpu["driver"])

    def install_ollama(self) -> None:
        pin = self.versions["ollama"]
        archive = self.paths.ollama_archive
        self.ops.download(pin["url"], archive, pin["sha256"], pin["size"])
        self._run(
            ["bash", "-o", "pipefail", "-c", 'zstd -dc -- "$1" | tar -x -C "$2"', "extract",
             str(archive), str(self.paths.usr)],
            timeout=900,
            what=f"extracting {archive.name}",
        )
        archive.unlink(missing_ok=True)

    def start_upstream_and_wait(self) -> None:
        self.children.append(start_upstream(self.env, paths=self.paths, ops=self.ops))
        want = self.versions["ollama"]["version"]
        got = poll(self.ops, OLLAMA_START_SECONDS, lambda: ollama_version(self.ops))
        if got is None:
            raise BootstrapError(f"the upstream did not answer /api/version within {OLLAMA_START_SECONDS} s")
        if got != want:
            raise BootstrapError(f"the upstream is Ollama {got}, not {want}")

    def pull_models(self) -> set[str]:
        """Pulls every tag; returns the routers that did not pull.

        The models follow their tags (2026-10-16): a digest other than the
        last verified one is logged at INFO, never refused. Only the chat
        model must pull; a router that does not is logged at ERROR and the
        bot routes the session without it.
        """
        unpulled: set[str] = set()
        for model, pin in models_in_load_order(self.versions):
            try:
                self._pull(model)
            except BootstrapError as exc:
                if register.required(pin):
                    raise
                # The ready event reports no digest for it and the bot
                # routes the session without it.
                logger.error("%s; booting on, the bot will route without it", exc)
                self.router_problems.append(str(exc))
                unpulled.add(model)
        status, body = self.ops.http("GET", OLLAMA_URL + "/api/tags", None, None, 10.0)
        if status != 200:
            raise BootstrapError(f"/api/tags answered HTTP {status}")
        listed = {
            str(m.get("name")): register.normalize_digest(m.get("digest"))
            for m in _json(body).get("models", [])
            if isinstance(m, dict)
        }
        for model, pin in self.versions["models"].items():
            got = listed.get(model)
            if got is None:
                if model not in unpulled:
                    # The probe names a missing chat model; the bot reads a
                    # router missing from /api/tags as not pulled.
                    logger.warning("%s pulled but /api/tags does not list it", model)
                    if not register.required(pin):
                        self.router_problems.append(f"{model} pulled but /api/tags does not list it")
                continue
            verified = pin["verified_digest"]
            note = "the verified one" if got == verified else f"tag moved since it was verified ({verified[:12]})"
            logger.info("%s is at digest %s, %s", model, got or "(none)", note)
        return unpulled

    def _pull(self, model: str) -> None:
        payload = json.dumps({"model": model, "stream": False}).encode()
        error = ""
        for attempt in range(1, PULL_ATTEMPTS + 1):
            try:
                status, body = self.ops.http(
                    "POST", OLLAMA_URL + "/api/pull", payload,
                    {"Content-Type": "application/json"}, float(PULL_TIMEOUT_SECONDS),
                )
            except OSError as exc:
                error = type(exc).__name__
            else:
                answer = _json(body)
                if status == 200 and answer.get("status") == "success":
                    logger.info("pulled %s", model)
                    return
                error = f"HTTP {status} {str(answer.get('error', ''))[:200]}".strip()
            if attempt < PULL_ATTEMPTS:
                logger.warning("pulling %s failed (%s); retrying", model, error)
                self.ops.sleep(PULL_BACKOFF_SECONDS[attempt - 1])
        raise BootstrapError(f"pulling {model} failed {PULL_ATTEMPTS} times: {error}")

    def record_vram(self) -> None:
        """What each warm model got in VRAM (``/api/ps`` size_vram/size) and
        nvidia-smi's memory.used, into state.json and the ready event's
        ``gpu``: colab/README.md's first-run check reads them. A router not
        wholly in VRAM is logged at ERROR and boots on (the bot routes
        without it; one not loaded at all was logged by the warm-up); the
        chat model's check is the probe's."""
        status, body = self.ops.http("GET", OLLAMA_URL + "/api/ps", None, None, 10.0)
        running: dict[str, dict[str, Any]] = {}
        if status == 200:
            running = {str(m.get("name")): m for m in _json(body).get("models") or [] if isinstance(m, dict)}
        report: dict[str, Any] = {}
        for model, pin in models_in_load_order(self.versions):
            entry = running.get(model)
            if entry is None:
                continue  # the probe names the chat model; warm_up logged a router
            in_vram, size = register.gpu_share(entry)
            report[f"{pin['role']}_on_gpu"] = f"{register.mib(in_vram)}/{register.mib(size)} MiB"
            if not register.on_gpu(entry) and not register.required(pin):
                logger.error(
                    "%s is not fully on the GPU (%s of %s MiB in VRAM); booting on, the bot will route without it",
                    model, register.mib(in_vram), register.mib(size),
                )
                self.router_problems.append(
                    f"{model} is not fully on the GPU ({register.mib(in_vram)} of {register.mib(size)} MiB in VRAM)"
                )
        used = self.gpu_memory_used()
        if used is not None:
            report["used_mib"] = used
        self.state.vram = report
        self.gpu = {**(self.gpu or {}), **report}
        logger.info("VRAM after the warm-up: %s", json.dumps(report, sort_keys=True))

    def gpu_memory_used(self) -> int | None:
        """nvidia-smi's memory.used in MiB; None (logged) when it cannot tell."""
        try:
            result = self.ops.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=60
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("nvidia-smi memory.used failed: %s", type(exc).__name__)
            return None
        used = parse_memory_used(result.stdout) if result.returncode == 0 else None
        if used is None:
            logger.warning("nvidia-smi printed no memory.used (exit %s)", result.returncode)
        return used

    def start_proxy(self) -> None:
        env = {
            **child_env(self.ops.environ()),
            **self.env.proxy_env(),
            "SESSION_STARTED_AT": repr(self.started_at),
        }
        proc = self.ops.spawn(
            [sys.executable, str(self.paths.code / "auth_proxy.py")],
            env=env,
            log_path=self.paths.log("auth_proxy"),
            # The proxy keeps the lock after this process exits.
            pass_fds=(self.lock_fd,) if self.lock_fd is not None else (),
        )
        self.children.append(proc)
        if poll(self.ops, PROXY_START_SECONDS, lambda: ollama_version(self.ops, PROXY_URL, self._auth)) is None:
            raise BootstrapError(f"the auth proxy did not answer within {PROXY_START_SECONDS} s")
        try:
            status, _ = self.ops.http("GET", PROXY_URL + "/api/version", None, None, 5.0)
        except OSError as exc:
            raise BootstrapError(f"the auth proxy failed without a token: {type(exc).__name__}") from exc
        if status != 401:
            raise BootstrapError(f"the auth proxy answered HTTP {status} without a token, not 401")

    def probe(self) -> None:
        """The bot's readiness probe, run here first through the proxy."""

        def get(path: str) -> Any:
            status, body = self.ops.http("GET", PROXY_URL + path, None, self._auth, 10.0)
            return _json(body) if status == 200 else None

        shows = {}
        for model, pin in self.versions["models"].items():
            if pin.get("thinking") is None:
                continue
            status, body = self.ops.http(
                "POST", PROXY_URL + "/api/show", json.dumps({"model": model}).encode(),
                {**self._auth, "Content-Type": "application/json"}, 10.0,
            )
            shows[model] = _json(body) if status == 200 else None
        version, ps, tags = get("/api/version"), get("/api/ps"), get("/api/tags")
        problems = register.probe_problems(version=version, ps=ps, tags=tags, shows=shows, versions=self.versions)
        if problems:
            raise BootstrapError("readiness probe: " + "; ".join(problems))
        self.ollama_version = str(version["version"])
        self.digests = {
            str(m["name"]): register.normalize_digest(m.get("digest"))
            for m in tags["models"]
            if isinstance(m, dict) and m.get("name") in self.versions["models"]
        }

    def start_tunnel(self) -> str:
        pin = self.versions["cloudflared"]
        self.ops.download(pin["url"], self.paths.cloudflared, pin["sha256"], pin["size"])
        os.chmod(self.paths.cloudflared, 0o755)
        for moved in move_cloudflared_config_aside(self.paths.cloudflared_config_dir):
            logger.warning("moved %s aside: quick tunnels refuse to run with it", moved)
        env = child_env(self.ops.environ())
        if self.env.tunnel_mode == "named":
            env["TUNNEL_TOKEN"] = self.env.tunnel_token
        for protocol in (None, "http2"):
            proc = self.ops.spawn(
                tunnel_command(self.paths.cloudflared, self.env.tunnel_mode, protocol),
                env=env,
                log_path=self.paths.log("cloudflared"),
            )
            self.children.append(proc)
            host = self._wait_for_tunnel(proc)
            if host is not None:
                logger.info("tunnel ready: https://%s", host)
                return f"https://{host}"
            self.children.remove(proc)
            self.ops.terminate(proc)
            if protocol is None:
                logger.warning("cloudflared was not ready in %s s; retrying with --protocol http2", TUNNEL_READY_SECONDS)
        raise BootstrapError(f"cloudflared was not ready in {TUNNEL_READY_SECONDS} s over QUIC or HTTP/2")

    def _wait_for_tunnel(self, proc: subprocess.Popen) -> str | None:
        """The public host once cloudflared has a connection; None when it
        exited or got none in time."""
        exited = object()

        def ready() -> Any:
            if proc.poll() is not None:
                return exited
            try:
                status, _ = self.ops.http("GET", METRICS_URL + "/ready", None, None, 5.0)
                if status != 200:
                    return None
                if self.env.tunnel_mode == "named":
                    return self.env.tunnel_host
                status, body = self.ops.http("GET", METRICS_URL + "/quicktunnel", None, None, 5.0)
            except OSError:
                return None
            host = str(_json(body).get("hostname") or "").strip().lower() if status == 200 else ""
            if not host:
                return None
            if not register.tunnel_host_ok(host):
                raise BootstrapError(f"cloudflared reported an unexpected quick-tunnel host {host[:80]!r}")
            return host

        host = poll(self.ops, TUNNEL_READY_SECONDS, ready, interval=2.0)
        return None if host is None or host is exited else host

    def check_tunnel(self, tunnel_url: str) -> None:
        """/api/version through the public URL, as the bot will call it. A
        failure is only logged: the VM's own DNS may lag, the bot probes anyway."""
        if poll(self.ops, TUNNEL_CHECK_SECONDS, lambda: ollama_version(self.ops, tunnel_url, self._auth), 3.0) is None:
            logger.warning("the tunnel did not answer from this VM within %s s; registering anyway", TUNNEL_CHECK_SECONDS)

    def register_ready(self, tunnel_url: str) -> None:
        body = register.build_event(
            "ready",
            self.env.nbh,
            tunnel_url=tunnel_url,
            versions={"ollama": self.ollama_version, "cloudflared": self.versions["cloudflared"]["version"]},
            digests=self.digests,
            gpu=self.gpu,
            # Why the router is not serving, for the bot's log (the VM's own
            # log never reaches Cloud Logging); None when it is.
            error="; ".join(self.router_problems) or None,
        )
        error = ""
        for attempt in range(1, REGISTER_ATTEMPTS + 1):
            try:
                status = register.post_event(
                    self.env.bot_url, self.env.reg_key, body, send=self.ops.http, now=self.ops.now, timeout=15.0
                )
            except OSError as exc:
                error = type(exc).__name__
            else:
                if 200 <= status < 300:
                    return
                if 400 <= status < 500 and status not in REGISTER_RETRY_STATUSES:
                    raise BootstrapError(f"the bot refused the ready registration (HTTP {status})")
                error = f"HTTP {status}"
            if attempt < REGISTER_ATTEMPTS:
                logger.warning("the ready registration failed (%s); retrying", error)
                # Each try is signed afresh at a new second (the bot refuses
                # a signature it has seen).
                self.ops.sleep(REGISTER_BACKOFF_SECONDS[attempt - 1])
        raise BootstrapError(f"the ready registration failed {REGISTER_ATTEMPTS} times: {error}")


def main(argv: list[str] | None = None, *, paths: Paths | None = None, ops: Ops | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up this Colab VM for the LINE bot (colab/README.md).")
    parser.add_argument("--stub-upstream", action="store_true", help="run the DRYRUN stub upstream on the Ollama port")
    args = parser.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    paths = paths or Paths()
    ops = ops or Ops()
    if args.stub_upstream:
        return serve_stub(register.load_versions(paths.versions))

    text = read_env_file(paths.env_file)
    lock_fd = acquire_lock(paths.lock_file)
    if lock_fd is None:
        logger.info("another bootstrap or auth proxy holds %s; exiting", paths.lock_file)
        return 0
    try:
        return _run_locked(text, lock_fd, paths=paths, ops=ops)
    finally:
        # The auth proxy inherited the lock (same open file): it stays held
        # while the proxy runs, and is released when a failed run stopped it.
        os.close(lock_fd)


def _run_locked(text: str | None, lock_fd: int, *, paths: Paths, ops: Ops) -> int:
    try:
        if text is None:
            raise EnvError(f"{paths.env_file} is missing")
        env = load_env(register.parse_bootstrap_env(text))
    except ValueError as exc:
        # No trusted key to sign a failed event with: record it and release.
        return abort(None, "env", str(exc), state=StateFile(paths.state_file, clock=ops.now), ops=ops)
    install_redaction(env.secrets())
    state = StateFile(paths.state_file, clock=ops.now, secrets=env.secrets())
    try:
        versions = register.load_versions(paths.versions)
    except (OSError, ValueError) as exc:
        return abort(env, "versions", f"{type(exc).__name__}: {exc}", state=state, ops=ops)
    return Bootstrap(env, versions, paths=paths, ops=ops, state=state, lock_fd=lock_fd).run()


if __name__ == "__main__":
    sys.exit(main())
