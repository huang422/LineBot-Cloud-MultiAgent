"""The Colab VM package (colab/, ADR-0010): bootstrap, auth proxy, registration.

Nothing here reaches Colab, Ollama, Cloudflare or the bot. The bootstrap
runs against ``FakeVM``, a miniature VM behind ``bootstrap.Ops`` (Ollama or
the DRYRUN stub, the auth proxy, cloudflared's metrics server, the bot's
register endpoint and Colab's runtime endpoint), on a fake clock. The auth
proxy runs for real (aiohttp) against local upstreams. The protocol is
checked against the bot's own copy (src/services/colab/protocol.py), and
the bot's registrar and readiness probe run against the VM's events and the
real proxy in front of the DRYRUN stub.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from colab import VM_FILES, auth_proxy, bootstrap, register

_REPO = Path(__file__).resolve().parents[1]
_COLAB = _REPO / "colab"
_VERSIONS = register.load_versions(_COLAB / "versions.json")
_PROXY_LOCK = _COLAB / bootstrap.PROXY_REQUIREMENTS
_CHAT = "qwen3.8:27b-mtp-q4_K_M"
_ROUTER = "nimble:9b"
_SECRET = "test-shared-secret-do-not-print"
_NBH = "3f2a8c1e-5b7d-4e2f-9a1c-0d4b6e8f1a2b"
_REG_KEY = register.derive_reg_key(_SECRET, _NBH)
_PROXY_TOKEN = register.derive_proxy_token(_SECRET, _NBH)
_TUNNEL_TOKEN = "eyJhIjoiYWNjb3VudC10YWciLCJ0IjoidHVubmVsLWlkIiwicyI6InNlY3JldCJ9"
_BOT_URL = "https://linebot-cloud-agent-test.us-west1.run.app"
_TBE = "172.28.0.12:8008"
_QUICK_HOST = "cool-fox-123.trycloudflare.com"
_NAMED_HOST = "colab.example.com"
_AUTH = {"Authorization": f"Bearer {_PROXY_TOKEN}"}
_HAS_AIOHTTP = importlib.util.find_spec("aiohttp") is not None


def _env_values(**overrides: str | None) -> dict[str, str]:
    values: dict[str, str | None] = {
        "COLAB_REG_KEY": _REG_KEY,
        "COLAB_PROXY_TOKEN": _PROXY_TOKEN,
        "BOT_URL": _BOT_URL,
        "NBH": _NBH,
        "IDLE_TTL": "900",
        "MAX_SESSION": "39600",
        "TUNNEL_MODE": "quick",
        "OPENCODE_ENABLED": "0",
        "DRYRUN": "0",
    }
    values.update(overrides)
    return {key: value for key, value in values.items() if value is not None}


def _env(**overrides: str | None) -> bootstrap.BootstrapEnv:
    return bootstrap.load_env(_env_values(**overrides))


def _named_env(**overrides: str | None) -> bootstrap.BootstrapEnv:
    return _env(TUNNEL_MODE="named", TUNNEL_TOKEN=_TUNNEL_TOKEN, TUNNEL_HOST=_NAMED_HOST, **overrides)


def _json_reply(status: int, payload: object) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _role(cmd: list[str]) -> str:
    if cmd[-1] == "serve" or "--stub-upstream" in cmd:
        return "upstream"
    if cmd[-1].endswith("auth_proxy.py"):
        return "proxy"
    if Path(cmd[0]).name == "cloudflared":
        return "tunnel"
    return "other"


def _protocol(cmd: list[str]) -> str | None:
    return cmd[cmd.index("--protocol") + 1] if "--protocol" in cmd else None


class _FakeProc:
    def __init__(self, cmd: list[str], env: dict[str, str], pid: int, *, exited: bool = False) -> None:
        self.cmd = list(cmd)
        self.env = dict(env)
        self.pid = pid
        self.returncode: int | None = 1 if exited else None

    def poll(self) -> int | None:
        return self.returncode


class FakeVM(bootstrap.Ops):
    """A Colab VM in miniature behind bootstrap.Ops. HTTP goes to whichever
    fake process serves the address; flags make each part misbehave."""

    def __init__(self) -> None:
        self.clock = 1_790_000_000.0
        self.calls: list[tuple] = []
        self.procs: list[_FakeProc] = []
        self.present = {"zstd", "nvidia-smi"}
        self.packages = bootstrap.locked_pins(_PROXY_LOCK.read_text(encoding="utf-8"))
        # cmd key -> CompletedProcess | Exception | list of them (popped in order)
        self.run_results: dict[str, object] = {}
        self.gpu_line = "NVIDIA A100-SXM4-40GB, 40960, 550.54.15"
        self.memory_used_line = "35012"
        self.upstream_dead = False
        self.upstream_version = _VERSIONS["ollama"]["version"]
        self.pull_failures: dict[str, int] = {}
        self.pulled: list[str] = []
        self.tag_digests = {model: pin["digest"] for model, pin in _VERSIONS["models"].items()}
        self.chat_load: dict = {"done_reason": "load"}
        self.systemone_status = 200
        # Models /api/ps leaves out (evicted or unloaded). With
        # no_room_for_both, loading one model evicts the other, as Ollama
        # does when it predicts the next one will not fit beside it.
        self.unloaded: set[str] = set()
        self.no_room_for_both = False
        self.stub = bootstrap.stub_payloads(_VERSIONS)
        self.proxy_dies = False
        self.proxy_needs_token = True
        self.tunnel_protocols: set[str | None] = {None, "http2"}
        self.cloudflared_exits = False
        self.quick_host = _QUICK_HOST
        self.tunnel_unreachable = False
        self.bot_unreachable = False
        self.register_statuses: list[int] = []
        self.events: list[tuple[dict, str, bool]] = []
        self.unassign_status = 200
        self.unassigns = 0
        self.download_error: Exception | None = None
        # The kernel's environment, with strays a child must not inherit.
        self.base_environ = {
            "PATH": "/usr/local/bin:/usr/bin",
            "HOME": "/root",
            "TBE_RUNTIME_ADDR": _TBE,
            "OLLAMA_ORIGINS": "*",
            "DRYRUN": "1",
        }

    # --- Ops --------------------------------------------------------------

    def now(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.calls.append(("sleep", seconds))
        self.clock += seconds

    def which(self, name: str) -> str | None:
        return f"/usr/bin/{name}" if name in self.present else None

    def environ(self) -> dict[str, str]:
        return dict(self.base_environ)

    def package_version(self, name: str) -> str | None:
        return self.packages.get(name)

    def run(self, cmd, *, timeout, env=None):
        cmd = list(cmd)
        self.calls.append(("run", cmd, dict(env) if env is not None else None))
        if cmd[0] == "apt-get":
            key = f"apt-get {cmd[1]}"
        elif cmd[0] == sys.executable and cmd[1:3] == ["-m", "pip"]:
            key = "pip"
        else:
            key = Path(cmd[0]).name
        result = self.run_results.get(key)
        if isinstance(result, list):
            result = result.pop(0) if result else None
        if isinstance(result, Exception):
            raise result
        if result is not None:
            return result
        stdout = ""
        if key == "nvidia-smi":
            stdout = (self.memory_used_line if "--query-gpu=memory.used" in cmd else self.gpu_line) + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    def spawn(self, cmd, *, env, log_path, pass_fds=()):
        cmd = list(cmd)
        self.calls.append(("spawn", cmd, dict(env), tuple(pass_fds)))
        role = _role(cmd)
        exited = (role == "tunnel" and self.cloudflared_exits) or (role == "proxy" and self.proxy_dies)
        proc = _FakeProc(cmd, env, 4000 + len(self.procs), exited=exited)
        self.procs.append(proc)
        return proc

    def terminate(self, proc) -> None:
        self.calls.append(("terminate", proc.cmd))
        proc.returncode = -15

    def kill_pidfile(self, path, grace: float = 10.0) -> None:
        self.calls.append(("kill_pidfile", Path(path)))
        for proc in self._running("upstream"):
            proc.returncode = -15

    def download(self, url, dest, sha256) -> None:
        self.calls.append(("download", url, Path(dest), sha256))
        if self.download_error is not None:
            raise self.download_error
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"pinned binary")

    def http(self, method, url, body=None, headers=None, timeout=10.0):
        self.calls.append(("http", method, url))
        parts = urllib.parse.urlsplit(url)
        headers = dict(headers or {})
        if parts.netloc == bootstrap.OLLAMA_ADDR:
            return self._upstream(method, parts.path, body)
        if parts.netloc == f"{bootstrap.PROXY_HOST}:{bootstrap.PROXY_PORT}":
            return self._proxy(method, parts.path, body, headers)
        if parts.netloc == bootstrap.METRICS_ADDR:
            return self._metrics(parts.path)
        if parts.netloc == _TBE:
            return self._unassign(method, parts.path)
        if url.startswith(_BOT_URL):
            return self._bot(parts.path, body, headers)
        if parts.scheme == "https" and self._running("tunnel") and not self.tunnel_unreachable:
            return self._proxy(method, parts.path, body, headers)
        raise ConnectionRefusedError(url)

    # --- the parts of the VM ----------------------------------------------

    def _running(self, role: str) -> list[_FakeProc]:
        return [p for p in self.procs if p.poll() is None and _role(p.cmd) == role]

    def spawned(self, role: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == "spawn" and _role(call[1]) == role]

    def ran(self, name: str) -> list[list[str]]:
        return [call[1] for call in self.calls if call[0] == "run" and Path(call[1][0]).name == name]

    def _upstream(self, method: str, path: str, body: bytes | None) -> tuple[int, bytes]:
        if self.upstream_dead or not self._running("upstream"):
            raise ConnectionRefusedError(bootstrap.OLLAMA_ADDR)
        request = json.loads(body) if body else {}
        route = (method, path)
        if route in (("GET", "/api/version"), ("HEAD", "/api/version")):
            return _json_reply(200, {"version": self.upstream_version})
        if route == ("POST", "/api/pull"):
            model = request["model"]
            self.pulled.append(model)
            if self.pull_failures.get(model, 0) > 0:
                self.pull_failures[model] -= 1
                return _json_reply(500, {"error": "pull model manifest: EOF"})
            return _json_reply(200, {"status": "success"})
        if route == ("GET", "/api/tags"):
            models = [
                {**m, "digest": self.tag_digests[m["name"]]}
                for m in self.stub["tags"]["models"]
                if m["name"] in self.tag_digests
            ]
            return _json_reply(200, {"models": models})
        if route == ("GET", "/api/ps"):
            return _json_reply(200, {"models": [m for m in self.stub["ps"]["models"] if m["name"] not in self.unloaded]})
        if route == ("POST", "/api/show"):
            show = self.stub["show"].get(request.get("model"))
            return _json_reply(200, show) if show else _json_reply(404, {"error": "model not found"})
        if route == ("POST", "/api/chat"):
            self._load(request["model"])
            return _json_reply(200, {"model": request["model"], "done": True, **self.chat_load})
        if route == ("POST", "/v1/systemone"):
            if self.systemone_status != 200:
                return _json_reply(self.systemone_status, {"error": "scoring failed"})
            self._load(request["model"])
            return _json_reply(200, {"model": request["model"], "answers": {"ready": {"type": "noul", "noul": 0.9}}})
        if route == ("POST", "/api/generate") and request.get("keep_alive") == 0 and not request.get("prompt"):
            self.unloaded.add(request["model"])
            return _json_reply(200, {"model": request["model"], "done": True, "done_reason": "unload"})
        return _json_reply(404, {"error": "404 page not found"})

    def _load(self, model: str) -> None:
        self.unloaded.discard(model)
        if self.no_room_for_both:
            self.unloaded.update(name for name in _VERSIONS["models"] if name != model)

    def _proxy(self, method, path, body, headers) -> tuple[int, bytes]:
        running = self._running("proxy")
        if not running:
            raise ConnectionRefusedError(bootstrap.PROXY_URL)
        token = running[-1].env.get("COLAB_PROXY_TOKEN", "")
        if self.proxy_needs_token and headers.get("Authorization") != f"Bearer {token}":
            return _json_reply(401, {"error": "unauthorized"})
        return self._upstream(method, path, body)

    def _metrics(self, path: str) -> tuple[int, bytes]:
        running = self._running("tunnel")
        if not running:
            raise ConnectionRefusedError(bootstrap.METRICS_ADDR)
        cmd = running[-1].cmd
        if path == "/ready":
            if _protocol(cmd) in self.tunnel_protocols:
                return _json_reply(200, {"status": 200, "readyConnections": 1})
            return _json_reply(503, {"status": 503, "readyConnections": 0})
        if path == "/quicktunnel":
            return _json_reply(200, {"hostname": self.quick_host if "--url" in cmd else ""})
        return _json_reply(404, {})

    def _bot(self, path: str, body: bytes, headers: dict) -> tuple[int, bytes]:
        if self.bot_unreachable:
            raise ConnectionResetError(_BOT_URL)
        assert path == register.REGISTER_PATH, path
        header = headers.get(register.SIGNATURE_HEADER, "")
        valid = register.verify(_REG_KEY, body, header, now=self.clock)
        self.events.append((json.loads(body), header, valid))
        status = self.register_statuses.pop(0) if self.register_statuses else 200
        return _json_reply(status, {"ok": status == 200})

    def _unassign(self, method: str, path: str) -> tuple[int, bytes]:
        assert (method, path) == ("POST", "/unassign"), (method, path)
        self.unassigns += 1
        return self.unassign_status, b""


class _RecordingState(bootstrap.StateFile):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.writes: list[tuple[str, str | None]] = []

    def write(self, step: str, error: str | None = None) -> None:
        self.writes.append((step, error))
        super().write(step, error)


def _paths(root: Path, code: Path = _COLAB) -> bootstrap.Paths:
    return bootstrap.Paths(
        content=root,
        code=code,
        usr=root / "usr",
        cloudflared=root / "bin" / "cloudflared",
        cloudflared_config_dir=root / ".cloudflared",
    )


class _TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


# --- pins ----------------------------------------------------------------------


class VersionsPinTests(unittest.TestCase):
    def test_versions_json_pins_what_the_protocol_names(self) -> None:
        register.validate_versions(_VERSIONS)
        ollama, cloudflared = _VERSIONS["ollama"], _VERSIONS["cloudflared"]
        self.assertEqual(ollama["version"], "0.35.0")
        self.assertEqual(
            ollama["url"], "https://github.com/ollama/ollama/releases/download/v0.35.0/ollama-linux-amd64.tar.zst"
        )
        self.assertEqual(ollama["sha256"], "1c114a6b220c5efca2ef2b1e5f01d1e535e26f6cd6d1678c8489325d2835e525")
        self.assertEqual(cloudflared["version"], "2026.9.3")
        self.assertEqual(
            cloudflared["url"],
            "https://github.com/cloudflare/cloudflared/releases/download/2026.9.3/cloudflared-linux-amd64",
        )
        self.assertEqual(cloudflared["sha256"], "77e26d8d900e0b8469f416239d14b5f296525fdf79fee6f511ef55609e3fbac2")
        self.assertEqual(
            {model: pin["digest"] for model, pin in _VERSIONS["models"].items()},
            {
                _CHAT: "22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643",
                _ROUTER: "24e550a16a7081881be2f1f0d91e8cc13a597472735c04119f035a0a85c67e0c",
            },
        )
        self.assertEqual(_VERSIONS["models"][_CHAT]["thinking"], [False, "low", "medium", "xhigh"])
        # The router's own num_ctx param (the stub reports it; the bot never sends one).
        self.assertEqual(_VERSIONS["models"][_ROUTER]["num_ctx"], 8194)
        self.assertEqual(_VERSIONS["gpu"], {"min_vram_mib": 38000, "min_driver_major": 550})

    def test_the_proxy_lock_is_the_bot_image_lock_for_its_packages(self) -> None:
        # The proxy runs on the aiohttp stack these tests run on: same
        # versions, same files (re-lock both when one changes).
        proxy: dict[str, tuple[str, frozenset[str]]] = {}
        for block in re.split(r"\n(?=[a-z0-9])", _PROXY_LOCK.read_text(encoding="utf-8")):
            match = re.match(r"([a-z0-9._-]+)==(\S+)", block)
            if match:
                proxy[match.group(1)] = (match.group(2), frozenset(re.findall(r"sha256:([0-9a-f]{64})", block)))
        image: dict[str, tuple[str, frozenset[str]]] = {}
        for package in tomllib.loads((_REPO / "uv.lock").read_text(encoding="utf-8"))["package"]:
            files = [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]
            image[package["name"]] = (
                package["version"],
                frozenset(file["hash"].removeprefix("sha256:") for file in files),
            )

        self.assertIn("aiohttp", proxy)
        self.assertLessEqual({"multidict", "yarl", "propcache", "frozenlist", "aiosignal", "attrs"}, set(proxy))
        for name, (version, hashes) in proxy.items():
            with self.subTest(name):
                self.assertTrue(hashes)
                self.assertEqual(version, image[name][0])
                # uv.lock (Python >= 3.12) leaves out the cp311 wheels the
                # proxy lock (Python >= 3.11, for Colab) keeps; every file it
                # does list must be one the proxy lock has.
                self.assertTrue(image[name][1])
                self.assertLessEqual(image[name][1], hashes)

    def test_locked_pins_leave_marker_pins_to_pip(self) -> None:
        text = (
            "# header\n"
            "aiohttp==3.14.3 \\\n    --hash=sha256:" + "a" * 64 + "\n"
            "Typing-Extensions==4.16.0 ; python_full_version < '3.13' \\\n"
            "yarl==1.25.1\n"
            "    # via aiohttp\n"
        )
        self.assertEqual(bootstrap.locked_pins(text), {"aiohttp": "3.14.3", "yarl": "1.25.1"})
        pins = bootstrap.locked_pins(_PROXY_LOCK.read_text(encoding="utf-8"))
        self.assertIn("aiohttp", pins)
        self.assertNotIn("typing-extensions", pins)

    def test_ollama_serve_environment_is_the_protocol_one(self) -> None:
        self.assertEqual(
            bootstrap.OLLAMA_SERVE_ENV,
            {
                "OLLAMA_HOST": "127.0.0.1:11434",
                "OLLAMA_KEEP_ALIVE": "-1",
                "OLLAMA_CONTEXT_LENGTH": "32768",
                "OLLAMA_MAX_LOADED_MODELS": "2",
                "OLLAMA_NUM_PARALLEL": "1",
                "OLLAMA_FLASH_ATTENTION": "1",
                "OLLAMA_NO_CLOUD": "1",
                "OLLAMA_MAX_QUEUE": "8",
                "OLLAMA_LOAD_TIMEOUT": "10m",
            },
        )
        env = bootstrap.upstream_env({"PATH": "/usr/bin", "OLLAMA_ORIGINS": "*", "COLAB_PROXY_TOKEN": "x"})
        self.assertNotIn("OLLAMA_ORIGINS", env)
        self.assertNotIn("COLAB_PROXY_TOKEN", env)
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_malformed_versions_are_refused(self) -> None:
        cases = [
            ([], "not an object"),
            ({**_VERSIONS, "ollama": {**_VERSIONS["ollama"], "url": "http://x"}}, "ollama.url"),
            ({**_VERSIONS, "cloudflared": {**_VERSIONS["cloudflared"], "sha256": "abc"}}, "cloudflared.sha256"),
            ({**_VERSIONS, "cloudflared": {**_VERSIONS["cloudflared"], "version": ""}}, "cloudflared.version"),
            ({k: v for k, v in _VERSIONS.items() if k != "ollama"}, "ollama is missing"),
            ({**_VERSIONS, "models": {}}, "models is missing"),
            ({**_VERSIONS, "models": {_CHAT: {**_VERSIONS["models"][_CHAT], "digest": "sha256:x"}}}, "digest"),
            ({**_VERSIONS, "models": {_CHAT: {**_VERSIONS["models"][_CHAT], "role": "vision"}}}, "role"),
            ({**_VERSIONS, "models": {_CHAT: {**_VERSIONS["models"][_CHAT], "thinking": "low"}}}, "thinking"),
            ({**_VERSIONS, "models": {_CHAT: _VERSIONS["models"][_CHAT]}}, "one chat and one router"),
            (
                {**_VERSIONS, "models": {**_VERSIONS["models"], "nimble:9b-q4_K_M": _VERSIONS["models"][_ROUTER]}},
                "exactly one chat and one router",
            ),
        ]
        for versions, fragment in cases:
            with self.subTest(fragment=fragment), self.assertRaisesRegex(ValueError, re.escape(fragment)):
                register.validate_versions(versions)


# --- imports -------------------------------------------------------------------


class ImportTests(_TempDirTestCase):
    """The VM scripts import without aiohttp, as a package and flat in /content."""

    def _python(self, args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        base = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and k not in register.BOOTSTRAP_ENV_KEYS}
        base["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, *args], cwd=cwd, env={**base, **(env or {})}, capture_output=True, text=True, timeout=60
        )

    def test_package_imports_without_aiohttp(self) -> None:
        code = textwrap.dedent(
            """
            import sys
            sys.modules["aiohttp"] = None
            sys.modules["multidict"] = None
            from colab import auth_proxy, bootstrap, register
            assert auth_proxy.route_allowed("GET", "/api/version", opencode_enabled=False)
            try:
                auth_proxy.build_app(auth_proxy.ProxyConfig(proxy_token="x"), auth_proxy.Activity(0.0))
            except ImportError:
                print("aiohttp only when serving")
            """
        )
        result = self._python(["-c", code], cwd=_REPO, env={"PYTHONPATH": str(_REPO)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("aiohttp only when serving", result.stdout)

    def test_flat_vm_layout_imports_siblings(self) -> None:
        for name in VM_FILES:
            shutil.copy(_COLAB / name, self.tmp / name)
        code = textwrap.dedent(
            """
            import sys
            sys.modules["aiohttp"] = None
            import auth_proxy, bootstrap, register
            assert auth_proxy.bootstrap is bootstrap and bootstrap.register is register
            assert bootstrap.HERE == auth_proxy.bootstrap.Paths().code
            print(sorted(register.load_versions(bootstrap.Paths().versions)["models"]))
            """
        )
        result = self._python(["-c", code], cwd=self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(_CHAT, result.stdout)

        help_result = self._python([str(self.tmp / "bootstrap.py"), "--help"], cwd=self.tmp)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--stub-upstream", help_result.stdout)

        # Without its settings the proxy refuses to start, naming the key.
        proxy_result = self._python([str(self.tmp / "auth_proxy.py")], cwd=self.tmp)
        self.assertEqual(proxy_result.returncode, 2)
        self.assertIn("COLAB_REG_KEY is missing", proxy_result.stderr)

    def test_vm_files_are_the_package(self) -> None:
        self.assertEqual(
            sorted(VM_FILES),
            sorted(p.name for p in _COLAB.iterdir() if p.suffix in (".py", ".json", ".lock") and p.name != "__init__.py"),
        )


# --- the env file, lock, state -------------------------------------------------


class EnvFileTests(_TempDirTestCase):
    def test_render_then_parse_round_trips(self) -> None:
        values = _env_values(TUNNEL_MODE="named", TUNNEL_TOKEN=_TUNNEL_TOKEN + "==", TUNNEL_HOST=_NAMED_HOST)
        self.assertEqual(register.parse_bootstrap_env(register.render_bootstrap_env(values)), values)

    def test_render_refuses_unknown_keys_and_line_breaks(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown bootstrap env key PATH"):
            register.render_bootstrap_env({"PATH": "/bin"})
        with self.assertRaisesRegex(ValueError, "BOT_URL contains a line break"):
            register.render_bootstrap_env({"BOT_URL": "https://x\nDRYRUN=1"})

    def test_parse_skips_comments_blank_lines_unknown_keys_and_quotes(self) -> None:
        text = "# launcher\n\n  BOT_URL = 'https://bot.example'\nNBH=\"abc\"\nPATH=/bin\nTUNNEL_TOKEN=a=b==\nDRYRUN='1\n"
        self.assertEqual(
            register.parse_bootstrap_env(text),
            {"BOT_URL": "https://bot.example", "NBH": "abc", "TUNNEL_TOKEN": "a=b==", "DRYRUN": "'1"},
        )

    def test_parse_error_names_the_line_not_its_content(self) -> None:
        with self.assertRaises(ValueError) as caught:
            register.parse_bootstrap_env("NBH=x\nsuper-secret-value\n")
        self.assertIn("line 2", str(caught.exception))
        self.assertNotIn("super-secret-value", str(caught.exception))

    def test_defaults(self) -> None:
        env = bootstrap.load_env({"COLAB_REG_KEY": _REG_KEY, "COLAB_PROXY_TOKEN": _PROXY_TOKEN, "BOT_URL": _BOT_URL + "/", "NBH": _NBH})
        self.assertEqual(env.bot_url, _BOT_URL)
        self.assertEqual((env.idle_ttl, env.max_session, env.tunnel_mode), (900, 39600, "quick"))
        self.assertFalse(env.opencode_enabled)
        self.assertFalse(env.dryrun)
        self.assertEqual(env.secrets(), (_REG_KEY, _PROXY_TOKEN))

    def test_flags_and_named_tunnel(self) -> None:
        env = _named_env(OPENCODE_ENABLED="true", DRYRUN="YES", IDLE_TTL="60", MAX_SESSION="120")
        self.assertTrue(env.opencode_enabled)
        self.assertTrue(env.dryrun)
        self.assertEqual((env.idle_ttl, env.max_session), (60, 120))
        self.assertEqual((env.tunnel_mode, env.tunnel_token, env.tunnel_host), ("named", _TUNNEL_TOKEN, _NAMED_HOST))
        self.assertEqual(env.secrets(), (_REG_KEY, _PROXY_TOKEN, _TUNNEL_TOKEN))

    def test_quick_tunnel_drops_named_settings(self) -> None:
        env = _env(TUNNEL_TOKEN=_TUNNEL_TOKEN, TUNNEL_HOST=_NAMED_HOST)
        self.assertEqual((env.tunnel_token, env.tunnel_host), ("", ""))

    def test_bad_values_are_refused_by_key_without_echoing_values(self) -> None:
        cases = [
            ({"COLAB_REG_KEY": None}, "COLAB_REG_KEY is missing"),
            ({"COLAB_REG_KEY": _REG_KEY.upper()}, "COLAB_REG_KEY is not 64"),
            ({"COLAB_REG_KEY": _REG_KEY[:-1]}, "COLAB_REG_KEY is not 64"),
            ({"COLAB_PROXY_TOKEN": None}, "COLAB_PROXY_TOKEN is missing"),
            ({"COLAB_PROXY_TOKEN": _PROXY_TOKEN[:-1]}, "COLAB_PROXY_TOKEN is not 43"),
            ({"COLAB_PROXY_TOKEN": "+" + _PROXY_TOKEN[1:]}, "COLAB_PROXY_TOKEN is not 43"),
            ({"BOT_URL": None}, "BOT_URL is missing"),
            ({"BOT_URL": "http://bot.example"}, "BOT_URL is not an https URL"),
            ({"BOT_URL": "https://user:pw@bot.example"}, "BOT_URL is not an https URL"),
            ({"BOT_URL": "https://bot.example/?x=1"}, "BOT_URL is not an https URL"),
            ({"BOT_URL": "https:///path"}, "BOT_URL is not an https URL"),
            ({"NBH": None}, "NBH is missing"),
            ({"NBH": "not-a-uuid"}, "NBH is not a UUID"),
            ({"IDLE_TTL": "15m"}, "IDLE_TTL is not a whole number"),
            ({"IDLE_TTL": "0"}, "IDLE_TTL must be positive"),
            ({"MAX_SESSION": "-5"}, "MAX_SESSION must be positive"),
            ({"TUNNEL_MODE": "ssh"}, "TUNNEL_MODE is not quick or named"),
            ({"TUNNEL_MODE": "named", "TUNNEL_HOST": _NAMED_HOST}, "TUNNEL_TOKEN is missing"),
            ({"TUNNEL_MODE": "named", "TUNNEL_TOKEN": _TUNNEL_TOKEN}, "TUNNEL_HOST is missing"),
            ({"TUNNEL_MODE": "named", "TUNNEL_TOKEN": _TUNNEL_TOKEN, "TUNNEL_HOST": "bad host"}, "TUNNEL_HOST is not"),
            ({"DRYRUN": "maybe"}, "DRYRUN is not a boolean"),
            ({"OPENCODE_ENABLED": "2"}, "OPENCODE_ENABLED is not a boolean"),
        ]
        for overrides, fragment in cases:
            with self.subTest(fragment=fragment):
                with self.assertRaises(bootstrap.EnvError) as caught:
                    _env(**overrides)
                message = str(caught.exception)
                self.assertIn(fragment, message)
                for secret in (_REG_KEY, _PROXY_TOKEN, _TUNNEL_TOKEN, "pw"):
                    self.assertNotIn(secret, message)

    def test_repr_hides_secrets(self) -> None:
        text = repr(_named_env())
        for secret in (_REG_KEY, _PROXY_TOKEN, _TUNNEL_TOKEN):
            self.assertNotIn(secret, text)

    def test_proxy_env_round_trips_without_tunnel_settings(self) -> None:
        env = _named_env(OPENCODE_ENABLED="1", DRYRUN="1", IDLE_TTL="300")
        proxy_env = env.proxy_env()
        self.assertFalse({"TUNNEL_MODE", "TUNNEL_TOKEN", "TUNNEL_HOST"} & set(proxy_env))
        self.assertNotIn(_TUNNEL_TOKEN, proxy_env.values())
        again = bootstrap.load_env(proxy_env)
        for name in ("reg_key", "proxy_token", "bot_url", "nbh", "idle_ttl", "max_session", "opencode_enabled", "dryrun"):
            self.assertEqual(getattr(again, name), getattr(env, name), name)

    def test_read_env_file_deletes_it(self) -> None:
        path = self.tmp / ".bootstrap.env"
        path.write_text("NBH=x\n", encoding="utf-8")
        self.assertEqual(bootstrap.read_env_file(path), "NBH=x\n")
        self.assertFalse(path.exists())
        self.assertIsNone(bootstrap.read_env_file(path))


class LockTests(_TempDirTestCase):
    def test_second_holder_is_refused_until_the_first_lets_go(self) -> None:
        path = self.tmp / "bootstrap.lock"
        fd = bootstrap.acquire_lock(path)
        self.assertIsNotNone(fd)
        self.assertIsNone(bootstrap.acquire_lock(path))
        os.close(fd)
        again = bootstrap.acquire_lock(path)
        self.assertIsNotNone(again)
        os.close(again)

    def test_a_child_given_the_lock_keeps_it_after_the_parent_closes_its_fd(self) -> None:
        path = self.tmp / "bootstrap.lock"
        fd = bootstrap.acquire_lock(path)
        ops = bootstrap.Ops()
        child = ops.spawn(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=os.environ,
            log_path=self.tmp / "child.log",
            pass_fds=(fd,),
        )
        self.addCleanup(lambda: child.poll() is None and child.kill())
        os.close(fd)
        self.assertIsNone(bootstrap.acquire_lock(path))
        ops.terminate(child)
        released = bootstrap.acquire_lock(path)
        self.assertIsNotNone(released)
        os.close(released)


class StateFileTests(_TempDirTestCase):
    def test_writes_step_ts_and_error_only(self) -> None:
        state = bootstrap.StateFile(self.tmp / "state.json", clock=lambda: 1_790_000_000.7)
        state.write("pull")
        self.assertEqual(state.read(), {"step": "pull", "ts": 1_790_000_000, "error": None})
        self.assertEqual(os.stat(state.path).st_mode & 0o777, 0o644)
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["state.json"])

    def test_error_is_redacted_cut_and_fits_the_heartbeat(self) -> None:
        state = bootstrap.StateFile(self.tmp / "state.json", secrets=(_PROXY_TOKEN,))
        state.write("tunnel", f"token {_PROXY_TOKEN} leaked " + "x" * 1000)
        text = state.path.read_text(encoding="utf-8")
        self.assertNotIn(_PROXY_TOKEN, text)
        self.assertIn("[redacted]", text)
        self.assertLessEqual(len(text), 300)
        self.assertEqual(len(json.loads(text)["error"]), bootstrap.MAX_STATE_ERROR_CHARS)

    def test_vram_is_kept_and_a_long_error_is_cut_to_fit_the_heartbeat(self) -> None:
        state = bootstrap.StateFile(self.tmp / "state.json", clock=lambda: 1_790_000_000)
        state.vram = {"chat_on_gpu": "24128/24128 MiB", "router_on_gpu": "10110/10110 MiB", "used_mib": 35012}
        state.write("proxy")
        self.assertEqual(state.read(), {"step": "proxy", "ts": 1_790_000_000, "error": None, "vram": state.vram})
        for error in ("x" * 1000, "逾時" * 500):
            with self.subTest(error=error[:2]):
                state.write("tunnel", error)
                text = state.path.read_text(encoding="utf-8")
                self.assertLessEqual(len(text.strip()), bootstrap.MAX_STATE_CHARS)
                written = json.loads(text)
                self.assertEqual(written["vram"], state.vram)
                self.assertTrue(written["error"] and error.startswith(written["error"]))
        # The heartbeat cell's cut keeps it whole.
        from src.services.colab.controller import HEARTBEAT_MARKER, vm_state

        cell = HEARTBEAT_MARKER + state.path.read_text(encoding="utf-8")[: bootstrap.MAX_STATE_CHARS].strip()
        self.assertEqual(vm_state(cell)["step"], "tunnel")

    def test_read_of_a_missing_or_torn_file_is_none(self) -> None:
        state = bootstrap.StateFile(self.tmp / "state.json")
        self.assertIsNone(state.read())
        state.path.write_text("{", encoding="utf-8")
        self.assertIsNone(state.read())


class RedactionTests(unittest.TestCase):
    def test_redact_and_the_logging_filter(self) -> None:
        self.assertEqual(bootstrap.redact("a SECRET b", ("SECRET", "")), "a [redacted] b")
        record = logging.LogRecord("colab", logging.INFO, __file__, 1, "key %s", (_REG_KEY,), None)
        self.assertTrue(bootstrap.RedactingFilter((_REG_KEY,)).filter(record))
        self.assertEqual(record.getMessage(), "key [redacted]")
        clean = logging.LogRecord("colab", logging.INFO, __file__, 1, "step %s", ("pull",), None)
        bootstrap.RedactingFilter((_REG_KEY,)).filter(clean)
        self.assertEqual((clean.msg, clean.args), ("step %s", ("pull",)))


# --- the protocol ----------------------------------------------------------------


class SignatureTests(unittest.TestCase):
    def test_keys_are_hmacs_of_the_shared_secret(self) -> None:
        import base64
        import hashlib
        import hmac

        self.assertEqual(_REG_KEY, hmac.new(_SECRET.encode(), f"reg:{_NBH}".encode(), hashlib.sha256).hexdigest())
        raw = hmac.new(_SECRET.encode(), f"proxy:{_NBH}".encode(), hashlib.sha256).digest()
        self.assertEqual(_PROXY_TOKEN, base64.urlsafe_b64encode(raw).decode().rstrip("="))
        self.assertEqual(len(_PROXY_TOKEN), 43)
        self.assertNotEqual(_REG_KEY, register.derive_reg_key(_SECRET, "another-nbh"))

    def test_sign_and_header(self) -> None:
        import hashlib
        import hmac

        body = b'{"event":"ready"}'
        expected = hmac.new(_REG_KEY.encode(), b"1790000000." + body, hashlib.sha256).hexdigest()
        self.assertEqual(register.sign(_REG_KEY, body, 1_790_000_000), expected)
        self.assertEqual(register.signature_header(_REG_KEY, body, 1_790_000_000), f"t=1790000000,v1={expected}")
        self.assertEqual(register.parse_signature_header(f" t=1790000000,v1={expected} "), (1_790_000_000, expected))

    def test_verify(self) -> None:
        body = register.build_event("stopping", _NBH, step="watchdog", error="idle")
        header = register.signature_header(_REG_KEY, body, 1_790_000_000)
        self.assertTrue(register.verify(_REG_KEY, body, header, now=1_790_000_300))
        self.assertTrue(register.verify(_REG_KEY, body, header, now=1_789_999_700))
        self.assertFalse(register.verify(_REG_KEY, body, header, now=1_790_000_301))
        self.assertFalse(register.verify(_REG_KEY, body + b" ", header, now=1_790_000_000))
        self.assertFalse(register.verify(register.derive_reg_key("other", _NBH), body, header, now=1_790_000_000))
        for bad in (None, "", "t=1,v1=zz", f"v1={'0' * 64},t=1", header.replace("v1=", "v2="), header + ",x=1"):
            with self.subTest(header=bad):
                self.assertFalse(register.verify(_REG_KEY, body, bad, now=1_790_000_000))

    def test_build_event(self) -> None:
        body = register.build_event("failed", _NBH, step="pull", error="e" * 2000, gpu=None)
        payload = json.loads(body)
        self.assertEqual(set(payload), {"event", "nbh", "step", "error"})
        self.assertEqual(len(payload["error"]), register.MAX_ERROR_CHARS)
        self.assertEqual(body, json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        ready = json.loads(
            register.build_event(
                "ready",
                _NBH,
                tunnel_url=f"https://{_QUICK_HOST}",
                versions={"ollama": "0.35.0", "cloudflared": "2026.9.3"},
                digests={_CHAT: "a" * 64},
                gpu={"name": "A100", "vram_mib": 40960, "driver": "550.54.15"},
            )
        )
        self.assertEqual(ready["gpu"]["vram_mib"], 40960)
        with self.assertRaisesRegex(ValueError, "unknown event"):
            register.build_event("started", _NBH)
        with self.assertRaisesRegex(ValueError, "nbh is empty"):
            register.build_event("ready", "")
        with self.assertRaisesRegex(ValueError, "over 8192"):
            register.build_event("ready", _NBH, digests={f"m{i}": "a" * 64 for i in range(200)})

    def test_post_event_signs_at_send_time(self) -> None:
        sent = []

        def send(method, url, body, headers, timeout):
            sent.append((method, url, body, headers, timeout))
            return 200, b"{}"

        body = register.build_event("stopping", _NBH)
        status = register.post_event(_BOT_URL + "/", _REG_KEY, body, send=send, now=lambda: 1_790_000_123.9, timeout=7.0)
        self.assertEqual(status, 200)
        method, url, sent_body, headers, timeout = sent[0]
        self.assertEqual((method, url, sent_body, timeout), ("POST", _BOT_URL + "/internal/colab/register", body, 7.0))
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertTrue(headers[register.SIGNATURE_HEADER].startswith("t=1790000123,v1="))
        self.assertTrue(register.verify(_REG_KEY, body, headers[register.SIGNATURE_HEADER], now=1_790_000_124))

    def test_tunnel_host_ok(self) -> None:
        for host in (_QUICK_HOST, "A-B.trycloudflare.com", "x.trycloudflare.com"):
            self.assertTrue(register.tunnel_host_ok(host), host)
        for host in ("trycloudflare.com", "a.b.trycloudflare.com", "a.trycloudflare.com.evil.com", "a_b.trycloudflare.com", ""):
            self.assertFalse(register.tunnel_host_ok(host), host)
        self.assertTrue(register.tunnel_host_ok("Colab.Example.com", _NAMED_HOST))
        self.assertFalse(register.tunnel_host_ok(_QUICK_HOST, _NAMED_HOST))


class HttpRequestTests(unittest.TestCase):
    """register.http_request: statuses come back, redirects are not followed."""

    @classmethod
    def setUpClass(cls) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "http://169.254.169.254/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 404 if self.path == "/missing" else 200
                body = json.dumps({"path": self.path}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_statuses(self) -> None:
        self.assertEqual(register.http_request("GET", self.base + "/ok"), (200, b'{"path": "/ok"}'))
        self.assertEqual(register.http_request("GET", self.base + "/missing")[0], 404)
        self.assertEqual(register.http_request("GET", self.base + "/redirect")[0], 302)
        with self.assertRaises(OSError):
            register.http_request("GET", f"http://127.0.0.1:{_free_port()}/", timeout=2.0)

    def test_a_garbled_answer_is_an_oserror_too(self) -> None:
        # The failure path relies on it: a bot or runtime endpoint that answers
        # garbage must not keep the VM from releasing itself.
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)

            def answer() -> None:
                conn, _ = server.accept()
                with conn:
                    conn.recv(65536)
                    conn.sendall(b"NOT-HTTP garbage\r\n\r\n")

            thread = threading.Thread(target=answer, daemon=True)
            thread.start()
            with self.assertRaisesRegex(ConnectionError, "HTTP protocol error: BadStatusLine"):
                register.http_request("POST", f"http://127.0.0.1:{server.getsockname()[1]}/unassign", b"", timeout=5.0)
            thread.join(5)


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        stub = bootstrap.stub_payloads(_VERSIONS)
        self.responses = {
            "version": stub["version"],
            "ps": stub["ps"],
            "tags": stub["tags"],
            "shows": {_CHAT: stub["show"][_CHAT]},
        }

    def problems(self, **changes) -> list[str]:
        return register.probe_problems(**{**self.responses, **changes}, versions=_VERSIONS)

    def test_the_stub_passes(self) -> None:
        self.assertEqual(self.problems(), [])

    def test_each_failure_is_named(self) -> None:
        stub = self.responses
        prefixed = {"models": [{**m, "digest": "sha256:" + m["digest"]} for m in stub["tags"]["models"]]}
        wrong_digest = {"models": [{**m, "digest": "0" * 64} if m["name"] == _CHAT else m for m in stub["tags"]["models"]]}
        cases = [
            ({"version": {"version": "0.34.4"}}, "Ollama version is '0.34.4', not 0.35.0"),
            ({"version": None}, "Ollama version is None"),
            ({"ps": {"models": "x"}}, f"{_CHAT} is not loaded"),
            (
                {"ps": {"models": [{**m, "size_vram": 20 << 30} if m["name"] == _CHAT else m for m in stub["ps"]["models"]]}},
                f"{_CHAT} is not fully on the GPU (/api/ps: 20480 of 24128 MiB in VRAM)",
            ),
            (
                {"ps": {"models": [{"name": m["name"]} for m in stub["ps"]["models"]]}},
                f"{_CHAT} is not fully on the GPU (/api/ps: 0 of 0 MiB in VRAM)",
            ),
            ({"tags": {"models": []}}, f"{_CHAT} is not pulled"),
            ({"tags": wrong_digest}, f"{_CHAT} digest is 000000000000, not the pinned 22130167c4c2"),
            ({"shows": {}}, f"{_CHAT} think values are None"),
            ({"shows": {_CHAT: {"thinking": {"values": [0, "low", "medium", "xhigh"]}}}}, "think values are [0,"),
            ({"shows": {_CHAT: {"thinking": {"values": [False, "low", "medium"]}}}}, "think values"),
        ]
        for changes, fragment in cases:
            with self.subTest(fragment=fragment):
                problems = self.problems(**changes)
                self.assertTrue(any(fragment in p for p in problems), problems)
        self.assertEqual(self.problems(tags=prefixed), [])

    def test_a_republished_router_passes(self) -> None:
        # The bot routes that session without it; only the chat model must be its pin.
        tags = {
            "models": [{**m, "digest": "d18e9174" + "0" * 56} if m["name"] == _ROUTER else m
                       for m in self.responses["tags"]["models"]]
        }
        self.assertEqual(self.problems(tags=tags), [])
        self.assertFalse(register.must_be_exact(_VERSIONS["models"][_ROUTER]))
        self.assertTrue(register.must_be_exact(_VERSIONS["models"][_CHAT]))

    def test_a_router_not_loaded_or_not_pulled_passes(self) -> None:
        # The bot routes that session without it; only the chat model must be there.
        only_chat_loaded = {"models": [m for m in self.responses["ps"]["models"] if m["name"] == _CHAT]}
        only_chat_pulled = {"models": [m for m in self.responses["tags"]["models"] if m["name"] == _CHAT]}
        self.assertEqual(self.problems(ps=only_chat_loaded), [])
        self.assertEqual(self.problems(tags=only_chat_pulled), [])
        self.assertEqual(self.problems(ps=only_chat_loaded, tags=only_chat_pulled), [])
        only_router_loaded = {"models": [m for m in self.responses["ps"]["models"] if m["name"] == _ROUTER]}
        self.assertEqual(self.problems(ps=only_router_loaded), [f"{_CHAT} is not loaded (/api/ps)"])

    def test_a_router_not_wholly_in_vram_passes(self) -> None:
        # The bot routes that session without it; only the chat model must fit.
        ps = {
            "models": [{**m, "size_vram": 6 << 30} if m["name"] == _ROUTER else m
                       for m in self.responses["ps"]["models"]]
        }
        self.assertEqual(self.problems(ps=ps), [])

    def test_gpu_shares_agree_with_the_bots_copy(self) -> None:
        from src.services.colab import protocol

        for entry in ({"size": 10, "size_vram": 10}, {"size": 10, "size_vram": 11}, {"size": 10, "size_vram": 9},
                      {"size": 0, "size_vram": 0}, {"size": 10}, {"size": "10", "size_vram": "10"},
                      {"size": True, "size_vram": True}, {"size": float("nan"), "size_vram": float("inf")},
                      {"size": float("-inf"), "size_vram": 1}, {}):
            with self.subTest(entry=entry):
                self.assertEqual(register.on_gpu(entry), protocol.on_gpu(entry))
                self.assertEqual(
                    register.gpu_share(entry),
                    (protocol.model_size(entry.get("size_vram")), protocol.model_size(entry.get("size"))),
                )
        self.assertEqual((register.mib(10_601_505_280), protocol.mib(10_601_505_280)), (10110, 10110))


class StubServerTests(unittest.TestCase):
    """The DRYRUN stub over real HTTP."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = bootstrap.make_stub_server(("127.0.0.1", 0), _VERSIONS)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method: str, path: str, payload: object = None) -> tuple[int, object]:
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = register.http_request(method, self.base + path, body, {"Content-Type": "application/json"})
        return status, json.loads(raw) if raw else None

    def test_answers_the_readiness_probe(self) -> None:
        _, version = self.call("GET", "/api/version")
        _, ps = self.call("GET", "/api/ps")
        _, tags = self.call("GET", "/api/tags")
        status, show = self.call("POST", "/api/show", {"model": _CHAT})
        self.assertEqual(status, 200)
        self.assertEqual(
            register.probe_problems(version=version, ps=ps, tags=tags, shows={_CHAT: show}, versions=_VERSIONS), []
        )
        # The stricter checks a probe may add: fully on the GPU, the served context.
        contexts = {m["name"]: m["context_length"] for m in ps["models"]}
        self.assertEqual(contexts, {_CHAT: 32768, _ROUTER: 8194})
        for model in ps["models"]:
            self.assertEqual(model["size_vram"], model["size"])
        self.assertEqual(show["thinking"], {"values": [False, "low", "medium", "xhigh"], "default": "medium"})
        self.assertEqual(self.call("POST", "/api/show", {"name": _ROUTER})[0], 200)

    def test_head_and_unknown_paths(self) -> None:
        status, raw = register.http_request("HEAD", self.base + "/api/version")
        self.assertEqual((status, raw), (200, b""))
        self.assertEqual(register.http_request("HEAD", self.base + "/api/tags")[0], 404)
        self.assertEqual(self.call("GET", "/api/version?x=1")[0], 200)
        self.assertEqual(self.call("GET", "/api/status")[0], 404)
        self.assertEqual(self.call("POST", "/api/pull", {"model": _CHAT})[0], 404)
        self.assertEqual(self.call("POST", "/api/show", {"model": "llama3:8b"}), (404, {"error": "model 'llama3:8b' not found"}))

    def test_inference_answers_busy_so_the_bot_falls_back(self) -> None:
        for path in bootstrap.INFERENCE_PATHS:
            with self.subTest(path=path):
                self.assertEqual(self.call("POST", path, {"model": _CHAT}), (503, {"error": bootstrap.OLLAMA_BUSY_ERROR}))


# --- the auth proxy: pure parts ------------------------------------------------


class ProxyHelperTests(unittest.TestCase):
    def test_route_table(self) -> None:
        allowed = [
            ("GET", "/api/version"),
            ("HEAD", "/api/version"),
            ("GET", "/api/ps"),
            ("GET", "/api/tags"),
            ("POST", "/api/show"),
            ("POST", "/api/chat"),
            ("POST", "/v1/systemone"),
            ("post", "/api/chat"),
        ]
        refused = [
            ("POST", "/api/pull"),
            ("POST", "/api/push"),
            ("DELETE", "/api/delete"),
            ("POST", "/api/create"),
            ("POST", "/api/copy"),
            ("HEAD", "/api/blobs/sha256:abc"),
            ("POST", "/api/blobs/sha256:abc"),
            ("POST", "/api/experimental/web_search"),
            ("POST", "/api/me"),
            ("POST", "/api/signout"),
            ("POST", "/api/generate"),
            ("POST", "/api/embed"),
            ("GET", "/api/status"),
            ("GET", "/"),
            ("HEAD", "/api/tags"),
            ("GET", "/api/chat"),
            ("POST", "/api/version"),
            ("GET", "/api/version/"),
            ("GET", "/API/VERSION"),
            ("GET", "//api/version"),
            ("GET", "/api/../api/version"),
            ("POST", "/v1/chat/completions"),
            ("GET", "/v1/models"),
            ("POST", "/v1/completions"),
            ("POST", "/v1/embeddings"),
        ]
        for method, path in allowed:
            with self.subTest(method=method, path=path):
                self.assertTrue(auth_proxy.route_allowed(method, path, opencode_enabled=False))
        for method, path in refused:
            with self.subTest(method=method, path=path):
                self.assertFalse(auth_proxy.route_allowed(method, path, opencode_enabled=False))

    def test_opencode_opens_only_its_two_routes(self) -> None:
        self.assertTrue(auth_proxy.route_allowed("POST", "/v1/chat/completions", opencode_enabled=True))
        self.assertTrue(auth_proxy.route_allowed("GET", "/v1/models", opencode_enabled=True))
        for method, path in (("POST", "/v1/completions"), ("POST", "/api/pull"), ("GET", "/v1/chat/completions")):
            self.assertFalse(auth_proxy.route_allowed(method, path, opencode_enabled=True))

    def test_only_inference_counts_as_activity(self) -> None:
        for path in ("/api/chat", "/v1/systemone", "/v1/chat/completions"):
            self.assertTrue(auth_proxy.counts_as_activity("POST", path), path)
        for method, path in (("GET", "/api/version"), ("HEAD", "/api/version"), ("GET", "/api/ps"),
                             ("GET", "/api/tags"), ("POST", "/api/show"), ("GET", "/v1/models"), ("GET", "/api/chat")):
            self.assertFalse(auth_proxy.counts_as_activity(method, path), path)

    def test_bearer(self) -> None:
        self.assertTrue(auth_proxy.bearer_ok(f"Bearer {_PROXY_TOKEN}", _PROXY_TOKEN))
        self.assertTrue(auth_proxy.bearer_ok(f"  bearer   {_PROXY_TOKEN} ", _PROXY_TOKEN))
        for header in (None, "", _PROXY_TOKEN, f"Basic {_PROXY_TOKEN}", f"Bearer {_PROXY_TOKEN}x",
                       f"Bearer {_PROXY_TOKEN[:-1]}", "Bearer", "Bearer 令牌"):
            with self.subTest(header=header):
                self.assertFalse(auth_proxy.bearer_ok(header, _PROXY_TOKEN))
        self.assertFalse(auth_proxy.bearer_ok("Bearer ", ""))

    def test_request_headers_lose_credentials_browser_context_and_hop_by_hop(self) -> None:
        forwarded = auth_proxy.forward_request_headers(
            [
                ("Authorization", f"Bearer {_PROXY_TOKEN}"),
                ("Origin", "https://evil.example"),
                ("Cookie", "a=b"),
                ("Host", f"{_QUICK_HOST}"),
                ("Connection", "keep-alive, X-Hop"),
                ("X-Hop", "1"),
                ("Keep-Alive", "timeout=5"),
                ("Transfer-Encoding", "chunked"),
                ("Content-Length", "12"),
                ("Proxy-Authorization", "Basic x"),
                ("Upgrade", "websocket"),
                ("TE", "trailers"),
                ("Content-Type", "application/json"),
                ("X-Request-Id", "abc"),
                ("CF-Connecting-IP", "203.0.113.9"),
            ]
        )
        self.assertEqual(
            forwarded,
            [
                ("Content-Type", "application/json"),
                ("X-Request-Id", "abc"),
                ("CF-Connecting-IP", "203.0.113.9"),
                ("Host", "127.0.0.1:11434"),
            ],
        )

    def test_response_headers_lose_hop_by_hop(self) -> None:
        forwarded = auth_proxy.forward_response_headers(
            [
                ("Content-Type", "application/x-ndjson"),
                ("Transfer-Encoding", "chunked"),
                ("Connection", "close, X-Private"),
                ("X-Private", "1"),
                ("Content-Length", "10"),
                ("Date", "today"),
            ]
        )
        self.assertEqual(forwarded, [("Content-Type", "application/x-ndjson"), ("Content-Length", "10"), ("Date", "today")])


class ActivityAndDecisionTests(unittest.TestCase):
    def test_activity(self) -> None:
        activity = auth_proxy.Activity(100.0)
        self.assertEqual(activity.idle_for(160.0), 60.0)
        activity.begin(200.0)
        self.assertEqual(activity.idle_for(5000.0), 0.0)  # a long request is not idle
        activity.begin(210.0)
        activity.end(300.0)
        self.assertEqual(activity.in_flight, 1)
        self.assertEqual(activity.idle_for(5000.0), 0.0)
        activity.end(310.0)
        self.assertEqual(activity.idle_for(400.0), 90.0)
        activity.end(50.0)  # an extra end neither goes negative nor moves time back
        self.assertEqual((activity.in_flight, activity.last), (0, 310.0))
        self.assertEqual(activity.idle_for(0.0), 0.0)

    def decide(self, **overrides) -> auth_proxy.Decision:
        values = {
            "started_at": 0.0,
            "idle_for": 0.0,
            "idle_ttl": 900,
            "max_session": 39600,
            "upstream_ok": True,
            "restarts": 0,
        }
        now = overrides.pop("now", 1000.0)
        values.update(overrides)
        return auth_proxy.watchdog_decision(now, **values)

    def test_decisions(self) -> None:
        self.assertEqual(self.decide(), auth_proxy.Decision("ok"))
        self.assertEqual(self.decide(idle_for=1020).action, "ok")
        idle = self.decide(idle_for=1020.5)
        self.assertEqual(idle.action, "stop")
        self.assertIn("idle for 1020 s, past IDLE_TTL 900 s + 120 s", idle.reason)
        self.assertEqual(self.decide(now=39720).action, "ok")
        old = self.decide(now=39721)
        self.assertEqual(old.action, "stop")
        self.assertIn("MAX_SESSION 39600 s", old.reason)
        self.assertEqual(self.decide(upstream_ok=False), auth_proxy.Decision("restart", "Ollama is not answering"))
        self.assertEqual(
            self.decide(upstream_ok=False, restarts=1), auth_proxy.Decision("stop", "Ollama is down after one restart")
        )
        # Age and idleness win over a restart: no point reviving a VM that goes.
        self.assertEqual(self.decide(upstream_ok=False, idle_for=5000).action, "stop")


def _outcome(value):
    """*value*, or raise it when it is an exception (a callback that fails)."""
    if isinstance(value, BaseException):
        raise value
    return value


class WatchdogTests(unittest.IsolatedAsyncioTestCase):
    def make(
        self, *, upstream=(True,), restart_ok=True, now=1000.0, idle_for_last=None, released=True, unassigned=True
    ):
        """Each of *upstream*'s items, *restart_ok*, *released* and *unassigned*
        may be an exception, which that callback then raises."""
        self.now = now
        self.upstream = list(upstream)
        self.stops: list[str] = []
        self.restart_calls = 0
        self.unassigns = 0
        config = auth_proxy.ProxyConfig(proxy_token=_PROXY_TOKEN, started_at=0.0, watchdog_interval=0.01)
        self.activity = auth_proxy.Activity(idle_for_last if idle_for_last is not None else now)

        async def check() -> bool:
            return _outcome(self.upstream.pop(0) if len(self.upstream) > 1 else self.upstream[0])

        async def restart() -> bool:
            self.restart_calls += 1
            return _outcome(restart_ok)

        async def stop(reason: str) -> bool:
            self.stops.append(reason)
            return _outcome(released)

        async def unassign() -> bool:
            self.unassigns += 1
            return _outcome(unassigned)

        return auth_proxy.Watchdog(
            config,
            self.activity,
            check_upstream=check,
            restart_upstream=restart,
            stop=stop,
            unassign=unassign,
            clock=lambda: self.now,
        )

    async def _until(self, condition) -> None:
        for _ in range(500):
            if condition():
                return
            await asyncio.sleep(0.01)
        self.fail("the watchdog never got there")

    async def test_healthy_tick_does_nothing(self) -> None:
        watchdog = self.make()
        self.assertEqual((await watchdog.tick()).action, "ok")
        self.assertEqual((self.stops, self.restart_calls), ([], 0))

    async def test_idle_session_is_released(self) -> None:
        watchdog = self.make(now=5000.0, idle_for_last=5000.0 - 1021)
        decision = await watchdog.tick()
        self.assertEqual(decision.action, "stop")
        self.assertEqual(self.stops, [decision.reason])
        self.assertTrue(watchdog.released)

    async def test_one_restart_then_release(self) -> None:
        watchdog = self.make(upstream=(False, True, False))
        self.assertEqual((await watchdog.tick()).action, "restart")
        self.assertEqual((self.restart_calls, watchdog.restarts), (1, 1))
        self.assertEqual((await watchdog.tick()).action, "ok")
        decision = await watchdog.tick()
        self.assertEqual(decision, auth_proxy.Decision("stop", "Ollama is down after one restart"))
        self.assertEqual(self.restart_calls, 1)
        self.assertEqual(self.stops, ["Ollama is down after one restart"])

    async def test_failed_restart_releases_at_once(self) -> None:
        watchdog = self.make(upstream=(False,), restart_ok=False, released=False)
        decision = await watchdog.tick()
        self.assertEqual(decision, auth_proxy.Decision("stop", "Ollama did not come back after a restart"))
        self.assertFalse(watchdog.released)

    async def test_run_sets_done_after_releasing(self) -> None:
        watchdog = self.make(now=50_000.0)
        done = asyncio.Event()
        await asyncio.wait_for(watchdog.run(done), 5)
        self.assertTrue(done.is_set())
        self.assertEqual(len(self.stops), 1)

    async def test_run_returns_when_done_is_set_elsewhere(self) -> None:
        watchdog = self.make()
        done = asyncio.Event()
        task = asyncio.create_task(watchdog.run(done))
        await asyncio.sleep(0.05)
        done.set()
        await asyncio.wait_for(task, 5)
        self.assertEqual(self.stops, [])

    # A callback that raises must never end the cost guard: the A100 would
    # bill on with nothing left to release it.

    async def test_a_restart_that_raises_leaves_the_watchdog_running(self) -> None:
        # Popen or the pid file failing in start_upstream, os.kill refused.
        watchdog = self.make(upstream=(False, True), restart_ok=OSError("no such file"))
        done = asyncio.Event()
        task = asyncio.create_task(watchdog.run(done))
        with self.assertLogs("colab.auth_proxy", level="ERROR") as logs:
            await self._until(lambda: self.restart_calls == 1)
            await asyncio.sleep(0.05)
        self.assertFalse(task.done())
        self.assertIn("OSError", "\n".join(logs.output))
        self.now += 50_000  # past MAX_SESSION: the next tick still sees it
        await asyncio.wait_for(task, 5)
        self.assertTrue(done.is_set())
        self.assertEqual(len(self.stops), 1)
        self.assertIn("MAX_SESSION", self.stops[0])
        self.assertTrue(watchdog.released)
        self.assertEqual(self.unassigns, 0)

    async def test_a_stop_that_raises_falls_back_to_the_bare_self_unassign(self) -> None:
        # state.json on a full disk (ENOSPC) before the event and the release.
        watchdog = self.make(now=50_000.0, released=OSError(28, "No space left on device"))
        done = asyncio.Event()
        with self.assertLogs("colab.auth_proxy", level="ERROR") as logs:
            await asyncio.wait_for(watchdog.run(done), 5)
        self.assertTrue(done.is_set())
        self.assertEqual((len(self.stops), self.unassigns), (1, 1))
        self.assertTrue(watchdog.released)
        self.assertIn("OSError", "\n".join(logs.output))

    async def test_checks_that_keep_failing_release_the_vm_directly(self) -> None:
        watchdog = self.make(upstream=(TypeError("bug"),))
        done = asyncio.Event()
        with self.assertLogs("colab.auth_proxy", level="ERROR"):
            await asyncio.wait_for(watchdog.run(done), 5)
        self.assertTrue(done.is_set())
        # Not the stop callback, whose parts may be what fails.
        self.assertEqual((self.stops, self.unassigns), ([], 1))
        self.assertTrue(watchdog.released)

    async def test_one_failed_check_between_good_ones_is_only_logged(self) -> None:
        watchdog = self.make(upstream=(True, TypeError("once"), True, True))
        done = asyncio.Event()
        task = asyncio.create_task(watchdog.run(done))
        with self.assertLogs("colab.auth_proxy", level="ERROR"):
            await self._until(lambda: len(self.upstream) == 1)
            await asyncio.sleep(0.05)
        self.assertFalse(task.done())
        done.set()
        await asyncio.wait_for(task, 5)
        self.assertEqual((self.stops, self.unassigns), ([], 0))

    async def test_a_last_resort_that_fails_too_still_ends_the_watchdog(self) -> None:
        watchdog = self.make(now=50_000.0, released=OSError("disk"), unassigned=RuntimeError("no loop"))
        done = asyncio.Event()
        with self.assertLogs("colab.auth_proxy", level="ERROR"):
            await asyncio.wait_for(watchdog.run(done), 5)
        self.assertTrue(done.is_set())
        self.assertIs(watchdog.released, False)
        self.assertEqual(self.unassigns, 1)


class ReleaseSessionTests(_TempDirTestCase):
    def test_stopping_event_then_self_unassign(self) -> None:
        vm = FakeVM()
        state = bootstrap.StateFile(self.tmp / "state.json", clock=vm.now)
        self.assertTrue(auth_proxy.release_session(_env(), "idle for 1021 s", state=state, ops=vm))
        self.assertEqual(state.read()["step"], "stopping")
        self.assertEqual(state.read()["error"], "idle for 1021 s")
        payload, _, valid = vm.events[0]
        self.assertTrue(valid)
        self.assertEqual(payload, {"event": "stopping", "nbh": _NBH, "step": "watchdog", "error": "idle for 1021 s"})
        self.assertEqual(vm.unassigns, 1)
        order = [c[2] for c in vm.calls if c[0] == "http"]
        self.assertEqual(order, [_BOT_URL + register.REGISTER_PATH, f"http://{_TBE}/unassign"])

    def test_release_still_happens_when_the_bot_is_unreachable(self) -> None:
        vm = FakeVM()
        vm.bot_unreachable = True
        state = bootstrap.StateFile(self.tmp / "state.json")
        self.assertTrue(auth_proxy.release_session(_env(), "max session", state=state, ops=vm))
        self.assertEqual(vm.unassigns, 1)

    def test_failed_unassign_is_reported(self) -> None:
        vm = FakeVM()
        vm.unassign_status = 500
        state = bootstrap.StateFile(self.tmp / "state.json")
        with self.assertLogs("colab", level="ERROR") as logs:
            self.assertFalse(auth_proxy.release_session(_env(), "idle", state=state, ops=vm))
        self.assertIn("self-unassign answered HTTP 500", "\n".join(logs.output))

    def test_without_tbe_runtime_addr_nothing_is_released(self) -> None:
        vm = FakeVM()
        del vm.base_environ["TBE_RUNTIME_ADDR"]
        with self.assertLogs("colab", level="ERROR") as logs:
            self.assertFalse(bootstrap.self_unassign(vm.environ(), send=vm.http))
        self.assertIn("not a Colab runtime", "\n".join(logs.output))
        self.assertEqual(vm.unassigns, 0)

    def test_unreachable_runtime_endpoint(self) -> None:
        def send(*args):
            raise ConnectionRefusedError("tbe")

        with self.assertLogs("colab", level="ERROR"):
            self.assertFalse(bootstrap.self_unassign({"TBE_RUNTIME_ADDR": _TBE}, send=send))


# --- the auth proxy, served ----------------------------------------------------


class _RecordingUpstream:
    """An Ollama stand-in that records what reaches it."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.activity: auth_proxy.Activity | None = None
        self.in_flight_seen: list[int] = []

    def app(self):
        from aiohttp import web

        async def handler(request: web.Request) -> web.StreamResponse:
            body = await request.read()
            headers = {k.lower(): v for k, v in request.headers.items()}
            self.requests.append((request.method, request.path, headers, body))
            if body == b'{"stream": true}':
                response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
                await response.prepare(request)
                await response.write(b'{"n":1}\n')
                await self.release.wait()
                await response.write(b'{"n":2}\n')
                await response.write_eof()
                return response
            if body == b'{"slow": true}':
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
            if body == b'{"peek": true}' and self.activity is not None:
                self.in_flight_seen.append(self.activity.in_flight)
            return web.json_response(
                {"method": request.method, "path": request.path, "version": "0.35.0"},
                headers={"X-Upstream": "ollama", "Connection": "keep-alive"},
            )

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        return app


@unittest.skipUnless(_HAS_AIOHTTP, "aiohttp is not installed")
class ProxyAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from aiohttp import test_utils

        self.upstream = _RecordingUpstream()
        self.upstream_server = test_utils.TestServer(self.upstream.app())
        await self.upstream_server.start_server()
        self.now = 1000.0
        self.proxy = None
        await self.start_proxy()

    async def start_proxy(self, **overrides) -> None:
        from aiohttp import test_utils

        if self.proxy is not None:
            await self.proxy.close()
        config = auth_proxy.ProxyConfig(
            proxy_token=_PROXY_TOKEN, upstream=str(self.upstream_server.make_url("")).rstrip("/"), **overrides
        )
        self.activity = auth_proxy.Activity(self.now)
        self.upstream.activity = self.activity
        app = auth_proxy.build_app(config, self.activity, clock=lambda: self.now)
        self.proxy = test_utils.TestClient(test_utils.TestServer(app))
        await self.proxy.start_server()

    async def asyncTearDown(self) -> None:
        await self.proxy.close()
        await self.upstream_server.close()

    async def test_bearer_is_required(self) -> None:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {_PROXY_TOKEN}"}):
            with self.subTest(headers=headers):
                response = await self.proxy.get("/api/version", headers=headers)
                self.assertEqual(response.status, 401)
                self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")
                self.assertEqual(await response.json(), {"error": "unauthorized"})
        # Unauthenticated callers learn nothing about the route table.
        self.assertEqual((await self.proxy.post("/api/pull")).status, 401)
        self.assertEqual(self.upstream.requests, [])

    async def test_unlisted_routes_never_reach_ollama(self) -> None:
        for method, path in (("POST", "/api/pull"), ("DELETE", "/api/delete"), ("POST", "/api/create"),
                             ("POST", "/api/copy"), ("POST", "/api/push"), ("POST", "/api/generate"),
                             ("GET", "/api/status"), ("GET", "/"), ("POST", "/v1/chat/completions"),
                             ("GET", "/v1/models"), ("GET", "/api/chat"), ("GET", "/api/version/")):
            with self.subTest(method=method, path=path):
                response = await self.proxy.request(method, path, headers=_AUTH)
                self.assertEqual(response.status, 404)
                self.assertEqual(await response.json(), {"error": "not found"})
        self.assertEqual(self.upstream.requests, [])

    async def test_forwarding_rewrites_host_and_drops_credentials(self) -> None:
        response = await self.proxy.post(
            "/api/chat?stream=true",
            data=b'{"x": 1}',
            headers={
                **_AUTH,
                "Origin": "https://evil.example",
                "Cookie": "a=b",
                "Connection": "keep-alive, X-Hop",
                "X-Hop": "1",
                "X-Request-Id": "abc",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["X-Upstream"], "ollama")
        method, path, headers, body = self.upstream.requests[-1]
        self.assertEqual((method, path, body), ("POST", "/api/chat", b'{"x": 1}'))
        self.assertEqual(headers["host"], "127.0.0.1:11434")
        for name in ("authorization", "origin", "cookie", "x-hop"):
            self.assertNotIn(name, headers)
        self.assertEqual(headers["x-request-id"], "abc")
        self.assertEqual(headers["content-type"], "application/json")

    async def test_head_version(self) -> None:
        response = await self.proxy.head("/api/version", headers=_AUTH)
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"")
        self.assertEqual(self.upstream.requests[-1][:2], ("HEAD", "/api/version"))

    async def test_streams_before_the_upstream_finishes(self) -> None:
        response = await self.proxy.post("/api/chat", data=b'{"stream": true}', headers=_AUTH)
        self.assertEqual(response.headers["Content-Type"], "application/x-ndjson")
        first = await asyncio.wait_for(response.content.readline(), 5)
        self.assertEqual(first, b'{"n":1}\n')  # while the upstream still holds the answer open
        self.upstream.release.set()
        self.assertEqual(await asyncio.wait_for(response.content.read(), 5), b'{"n":2}\n')

    async def test_only_inference_moves_the_idle_clock(self) -> None:
        self.now = 2000.0
        for method, path in (("GET", "/api/version"), ("GET", "/api/ps"), ("GET", "/api/tags"), ("POST", "/api/show")):
            await self.proxy.request(method, path, headers=_AUTH, data=b"{}" if method == "POST" else None)
        self.assertEqual(self.activity.last, 1000.0)
        await self.proxy.get("/api/nope", headers=_AUTH)
        await self.proxy.post("/api/chat", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(self.activity.last, 1000.0)
        response = await self.proxy.post("/api/chat", data=b'{"peek": true}', headers=_AUTH)
        self.assertEqual(response.status, 200)
        self.assertEqual(self.upstream.in_flight_seen, [1])
        self.assertEqual((self.activity.last, self.activity.in_flight), (2000.0, 0))
        self.now = 3000.0
        await self.proxy.post("/v1/systemone", data=b"{}", headers=_AUTH)
        self.assertEqual(self.activity.last, 3000.0)

    async def test_opencode_routes_when_enabled(self) -> None:
        await self.start_proxy(opencode_enabled=True)
        self.assertEqual((await self.proxy.get("/v1/models", headers=_AUTH)).status, 200)
        self.now = 5000.0
        self.assertEqual((await self.proxy.post("/v1/chat/completions", data=b"{}", headers=_AUTH)).status, 200)
        self.assertEqual(self.activity.last, 5000.0)

    async def test_body_limit(self) -> None:
        await self.start_proxy(max_body_bytes=1024)
        response = await self.proxy.post("/api/chat", data=b"x" * 2048, headers=_AUTH)
        self.assertEqual(response.status, 413)

        async def chunks():
            for _ in range(4):
                yield b"y" * 512

        response = await self.proxy.post("/api/chat", data=chunks(), headers=_AUTH)
        self.assertEqual(response.status, 413)
        self.assertEqual(self.upstream.requests, [])
        self.assertEqual(self.activity.in_flight, 0)

    async def test_upstream_down_is_502(self) -> None:
        await self.proxy.close()
        from aiohttp import test_utils

        config = auth_proxy.ProxyConfig(proxy_token=_PROXY_TOKEN, upstream=f"http://127.0.0.1:{_free_port()}")
        activity = auth_proxy.Activity(0.0)
        self.proxy = test_utils.TestClient(test_utils.TestServer(auth_proxy.build_app(config, activity)))
        await self.proxy.start_server()
        response = await self.proxy.post("/api/chat", data=b"{}", headers=_AUTH)
        self.assertEqual(response.status, 502)
        self.assertEqual(await response.json(), {"error": "upstream unavailable"})
        self.assertEqual(activity.in_flight, 0)

    async def test_upstream_timeout_is_504_and_cancels_ollama(self) -> None:
        await self.start_proxy(upstream_timeout=0.3)
        response = await self.proxy.post("/api/chat", data=b'{"slow": true}', headers=_AUTH)
        self.assertEqual(response.status, 504)
        self.assertEqual(await response.json(), {"error": "upstream timed out"})
        await asyncio.wait_for(self.upstream.cancelled.wait(), 5)
        self.assertEqual(self.activity.in_flight, 0)

    async def test_client_going_away_cancels_the_upstream_call(self) -> None:
        # TestServer always cancels handlers; ServeTests checks serve() does too.
        task = asyncio.create_task(self.proxy.post("/api/chat", data=b'{"slow": true}', headers=_AUTH))
        for _ in range(200):
            if self.upstream.requests:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.activity.in_flight, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(self.upstream.cancelled.wait(), 5)
        for _ in range(200):
            if self.activity.in_flight == 0:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.activity.in_flight, 0)


@unittest.skipUnless(_HAS_AIOHTTP, "aiohttp is not installed")
class ServeTests(unittest.IsolatedAsyncioTestCase):
    async def test_serves_until_the_watchdog_releases_the_vm(self) -> None:
        import aiohttp

        port = _free_port()
        self.now = 100.0
        reasons: list[str] = []
        config = auth_proxy.ProxyConfig(
            proxy_token=_PROXY_TOKEN, listen_port=port, started_at=0.0, max_session=600, watchdog_interval=0.02
        )

        async def healthy() -> bool:
            return True

        async def stop(reason: str) -> bool:
            reasons.append(reason)
            return False  # the self-unassign failed: exit code 1

        task = asyncio.create_task(
            auth_proxy.serve(
                config,
                check_upstream=healthy,
                restart_upstream=healthy,
                stop=stop,
                unassign=healthy,
                clock=lambda: self.now,
            )
        )
        async with aiohttp.ClientSession() as session:
            for _ in range(200):
                try:
                    async with session.get(f"http://127.0.0.1:{port}/api/version") as response:
                        self.assertEqual(response.status, 401)
                        break
                except aiohttp.ClientConnectionError:
                    await asyncio.sleep(0.01)
            else:
                self.fail("the proxy never listened")
        self.assertFalse(task.done())
        self.now = 10_000.0
        self.assertEqual(await asyncio.wait_for(task, 5), 1)
        self.assertEqual(len(reasons), 1)
        self.assertIn("MAX_SESSION 600 s", reasons[0])

    async def test_a_client_that_hangs_up_aborts_the_ollama_call(self) -> None:
        # serve()'s own runner, not TestServer (which always cancels handlers):
        # cloudflared closes its origin connection when the bot gives up.
        from aiohttp import test_utils

        upstream = _RecordingUpstream()
        upstream_server = test_utils.TestServer(upstream.app())
        await upstream_server.start_server()
        self.addAsyncCleanup(upstream_server.close)
        port = _free_port()
        self.now = 100.0
        config = auth_proxy.ProxyConfig(
            proxy_token=_PROXY_TOKEN,
            listen_port=port,
            upstream=str(upstream_server.make_url("")).rstrip("/"),
            max_session=600,
            watchdog_interval=0.02,
        )

        async def healthy(*_args) -> bool:
            return True

        task = asyncio.create_task(
            auth_proxy.serve(
                config,
                check_upstream=healthy,
                restart_upstream=healthy,
                stop=healthy,
                unassign=healthy,
                clock=lambda: self.now,
            )
        )
        body = b'{"slow": true}'
        request = (
            f"POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {_PROXY_TOKEN}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
        ).encode() + body
        for _ in range(200):
            try:
                _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except OSError:
                await asyncio.sleep(0.01)
        else:
            self.fail("the proxy never listened")
        with self.assertLogs("colab.auth_proxy", level="INFO") as logs:
            writer.write(request)
            await writer.drain()
            for _ in range(200):
                if upstream.requests:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(len(upstream.requests), 1)
            writer.close()
            await asyncio.wait_for(upstream.cancelled.wait(), 5)
            for _ in range(200):
                if any("cancelled" in line for line in logs.output):
                    break
                await asyncio.sleep(0.01)
        self.assertIn("POST /api/chat cancelled after", "\n".join(logs.output))
        self.now = 10_000.0
        self.assertEqual(await asyncio.wait_for(task, 5), 0)

    async def test_a_watchdog_that_dies_releases_the_vm_and_ends_serving(self) -> None:
        config = auth_proxy.ProxyConfig(proxy_token=_PROXY_TOKEN, listen_port=_free_port(), watchdog_interval=0.02)
        unassigns: list[str] = []

        async def healthy(*_args) -> bool:
            return True

        async def unassign() -> bool:
            unassigns.append("unassign")
            return True

        async def dies(self, done: asyncio.Event) -> None:
            raise RuntimeError("a bug in the watchdog loop itself")

        with mock.patch.object(auth_proxy.Watchdog, "run", dies), self.assertLogs("colab.auth_proxy", "ERROR") as logs:
            code = await asyncio.wait_for(
                auth_proxy.serve(
                    config, check_upstream=healthy, restart_upstream=healthy, stop=healthy, unassign=unassign
                ),
                5,
            )
        self.assertEqual(code, 0)
        self.assertEqual(unassigns, ["unassign"])
        self.assertIn("RuntimeError", "\n".join(logs.output))

    async def test_main_refuses_an_incomplete_environment(self) -> None:
        with self.assertLogs("colab", level="ERROR") as logs:
            self.assertEqual(auth_proxy.main({"COLAB_REG_KEY": _REG_KEY}), 2)
        self.assertIn("COLAB_PROXY_TOKEN is missing", "\n".join(logs.output))
        self.assertNotIn(_REG_KEY, "\n".join(logs.output))


# --- side effects, for real ----------------------------------------------------


class DownloadTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.source = self.tmp / "source.bin"
        self.source.write_bytes(b"ollama tarball")
        self.sha = bootstrap.file_sha256(self.source)
        self.dest = self.tmp / "out" / "ollama.tar.zst"
        self.dest.parent.mkdir()

    def test_verified_download(self) -> None:
        bootstrap.download_verified(self.source.as_uri(), self.dest, self.sha)
        self.assertEqual(self.dest.read_bytes(), b"ollama tarball")
        self.assertEqual(list(self.dest.parent.iterdir()), [self.dest])

    def test_wrong_hash_is_refused_and_left_nowhere(self) -> None:
        sleeps = []
        with self.assertRaisesRegex(bootstrap.DownloadError, "not the pinned 000000000000"):
            bootstrap.download_verified(self.source.as_uri(), self.dest, "0" * 64, sleep=sleeps.append)
        self.assertEqual(list(self.dest.parent.iterdir()), [])
        self.assertEqual(sleeps, [])

    def test_a_good_file_already_there_is_kept(self) -> None:
        self.dest.write_bytes(b"ollama tarball")

        def must_not_open(*args, **kwargs):
            raise AssertionError("downloaded again")

        bootstrap.download_verified("https://example.invalid/x", self.dest, self.sha, open_url=must_not_open)

    def test_network_errors_are_retried_then_given_up(self) -> None:
        attempts = []

        def flaky(url, timeout):
            attempts.append(url)
            if len(attempts) < 3:
                raise ConnectionResetError("reset")
            return io.BytesIO(b"ollama tarball")

        sleeps: list[float] = []
        bootstrap.download_verified("https://example.invalid/x", self.dest, self.sha, open_url=flaky, sleep=sleeps.append)
        self.assertEqual((len(attempts), sleeps), (3, [10, 20]))
        self.dest.unlink()

        def broken(url, timeout):
            raise TimeoutError("slow")

        with self.assertRaisesRegex(bootstrap.DownloadError, "failed 3 times: TimeoutError"):
            bootstrap.download_verified("https://example.invalid/x", self.dest, self.sha, open_url=broken, sleep=lambda s: None)
        self.assertEqual(list(self.dest.parent.iterdir()), [])


class OpsTests(_TempDirTestCase):
    def test_run_spawn_terminate_and_kill_pidfile(self) -> None:
        ops = bootstrap.Ops()
        result = ops.run([sys.executable, "-c", "print('hi')"], timeout=30)
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "hi"))

        proc = ops.spawn([sys.executable, "-c", "import time; print('up', flush=True); time.sleep(30)"],
                         env=os.environ, log_path=self.tmp / "child.log")
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        ops.terminate(proc)
        self.assertEqual(proc.returncode, -15)
        ops.terminate(proc)  # already gone: nothing to do

        other = ops.spawn([sys.executable, "-c", "import time; time.sleep(30)"], env=os.environ,
                          log_path=self.tmp / "other.log")
        self.addCleanup(lambda: other.poll() is None and other.kill())
        pidfile = self.tmp / "upstream.pid"
        pidfile.write_text(f"{other.pid}\n", encoding="utf-8")
        ops.kill_pidfile(pidfile, grace=1.0)
        self.assertEqual(other.wait(timeout=10), -15)
        ops.kill_pidfile(self.tmp / "missing.pid")  # nothing recorded: nothing to do
        pidfile.write_text("not a pid", encoding="utf-8")
        ops.kill_pidfile(pidfile)

    def test_real_ops_basics(self) -> None:
        ops = bootstrap.Ops()
        self.assertIsNotNone(ops.which("sh"))
        self.assertIsNone(ops.package_version("surely-not-installed-package"))
        self.assertEqual(ops.environ(), dict(os.environ))


class GpuTests(unittest.TestCase):
    limits = _VERSIONS["gpu"]

    def test_parse(self) -> None:
        self.assertEqual(
            bootstrap.parse_gpu("NVIDIA A100-SXM4-40GB, 40960, 550.54.15\nNVIDIA A100-SXM4-40GB, 40960, 550.54.15\n"),
            {"name": "NVIDIA A100-SXM4-40GB", "vram_mib": 40960, "driver": "550.54.15"},
        )
        for output in ("", "\n", "NVIDIA A100, lots, 550"):
            with self.subTest(output=output), self.assertRaisesRegex(bootstrap.BootstrapError, "no GPU"):
                bootstrap.parse_gpu(output)

    def test_limits(self) -> None:
        a100 = bootstrap.parse_gpu("NVIDIA A100-SXM4-40GB, 40960, 550.54.15")
        self.assertEqual(bootstrap.gpu_problems(a100, self.limits), [])
        self.assertEqual(bootstrap.gpu_problems({**a100, "vram_mib": 81920}, self.limits), [])
        t4 = bootstrap.parse_gpu("Tesla T4, 15360, 535.104.05")
        self.assertEqual(
            bootstrap.gpu_problems(t4, self.limits),
            ["Tesla T4 has 15360 MiB, under 38000 MiB", "NVIDIA driver 535.104.05 is older than 550"],
        )
        self.assertEqual(len(bootstrap.gpu_problems({**a100, "driver": "N/A"}, self.limits)), 1)


class TunnelCommandTests(_TempDirTestCase):
    def test_commands(self) -> None:
        binary = Path("/usr/local/bin/cloudflared")
        base = [str(binary), "tunnel", "--no-autoupdate", "--metrics", "127.0.0.1:20241"]
        self.assertEqual(bootstrap.tunnel_command(binary, "quick"), base + ["--url", "http://127.0.0.1:8081"])
        self.assertEqual(
            bootstrap.tunnel_command(binary, "quick", "http2"),
            base + ["--protocol", "http2", "--url", "http://127.0.0.1:8081"],
        )
        self.assertEqual(bootstrap.tunnel_command(binary, "named"), base + ["run"])

    def test_config_is_moved_aside(self) -> None:
        config_dir = self.tmp / ".cloudflared"
        self.assertEqual(bootstrap.move_cloudflared_config_aside(config_dir), [])
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("tunnel: x\n", encoding="utf-8")
        (config_dir / "config.yml").write_text("tunnel: y\n", encoding="utf-8")
        (config_dir / "cert.pem").write_text("keep\n", encoding="utf-8")
        moved = bootstrap.move_cloudflared_config_aside(config_dir)
        self.assertEqual([p.name for p in moved], ["config.yaml.disabled", "config.yml.disabled"])
        self.assertEqual(sorted(p.name for p in config_dir.iterdir()),
                         ["cert.pem", "config.yaml.disabled", "config.yml.disabled"])


# --- the bootstrap run ---------------------------------------------------------

_REAL_STEPS = ["python", "zstd", "gpu", "ollama_install", "ollama_start", "pull", "warmup",
               "proxy", "probe", "tunnel", "tunnel_check", "register", "ready"]
_DRYRUN_STEPS = ["python", "stub_start", "proxy", "probe", "tunnel", "tunnel_check", "register", "ready"]


class BootstrapFlowTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.paths = _paths(self.tmp)
        self.vm = FakeVM()

    def boot(self, env: bootstrap.BootstrapEnv | None = None) -> int:
        self.env = env or _env()
        self.state = _RecordingState(self.paths.state_file, clock=self.vm.now, secrets=self.env.secrets())
        with self.assertLogs("colab", level="DEBUG") as logs:
            code = bootstrap.Bootstrap(
                self.env, _VERSIONS, paths=self.paths, ops=self.vm, state=self.state, lock_fd=7
            ).run()
        self.log_text = "\n".join(logs.output)
        self.assert_no_secret_leaked()
        return code

    def steps(self) -> list[str]:
        return [step for step, _ in self.state.writes]

    def ready_events(self) -> list[dict]:
        return [payload for payload, _, _ in self.vm.events if payload["event"] == "ready"]

    def assert_no_secret_leaked(self) -> None:
        secrets = self.env.secrets()
        for call in self.vm.calls:
            if call[0] in ("run", "spawn"):
                for secret in secrets:
                    self.assertNotIn(secret, " ".join(call[1]))
            if call[0] == "spawn":
                role = _role(call[1])
                env_values = set(call[2].values())
                if role == "proxy":
                    self.assertEqual(call[2]["COLAB_PROXY_TOKEN"], self.env.proxy_token)
                    self.assertEqual(call[2]["COLAB_REG_KEY"], self.env.reg_key)
                    self.assertNotIn("TUNNEL_TOKEN", call[2])
                elif role == "tunnel" and self.env.tunnel_mode == "named":
                    self.assertEqual(env_values & set(secrets), {self.env.tunnel_token})
                else:
                    self.assertEqual(env_values & set(secrets), set(), role)
                self.assertNotIn("DRYRUN", {k for k in call[2] if role != "proxy"})
        for secret in secrets:
            self.assertNotIn(secret, self.log_text)
            if self.paths.state_file.exists():
                self.assertNotIn(secret, self.paths.state_file.read_text(encoding="utf-8"))
            for payload, _, _ in self.vm.events:
                self.assertNotIn(secret, json.dumps(payload))

    def assert_failed(self, code: int, step: str, fragment: str) -> None:
        self.assertEqual(code, 1)
        last_step, error = self.state.writes[-1]
        self.assertEqual(last_step, step)
        self.assertIn(fragment, error or "")
        failed = [(payload, valid) for payload, _, valid in self.vm.events if payload["event"] == "failed"]
        self.assertEqual(len(failed), 1, self.vm.events)
        payload, valid = failed[0]
        self.assertTrue(valid)
        self.assertEqual((payload["nbh"], payload["step"]), (_NBH, step))
        self.assertIn(fragment, payload["error"])
        self.assertEqual(self.vm.unassigns, 1)
        self.assertTrue(all(proc.poll() is not None for proc in self.vm.procs), "a child was left running")
        self.assertEqual(self.ready_events(), [] if step != "register" else self.ready_events())

    # --- success ---

    def test_a100_run(self) -> None:
        self.assertEqual(self.boot(), 0)
        self.assertEqual(self.steps(), _REAL_STEPS)
        vram = {"chat_on_gpu": "24128/24128 MiB", "router_on_gpu": "10110/10110 MiB", "used_mib": 35012}
        self.assertEqual(self.state.read(), {"step": "ready", "ts": int(self.vm.clock), "error": None, "vram": vram})

        self.assertEqual(self.vm.ran("nvidia-smi"), [
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        ])
        self.assertIn('VRAM after the warm-up: {"chat_on_gpu": "24128/24128 MiB"', self.log_text)
        extract = self.vm.ran("bash")[0]
        self.assertEqual(extract[-2:], [str(self.paths.ollama_archive), str(self.paths.usr)])
        self.assertIn('zstd -dc -- "$1" | tar -x -C "$2"', extract)
        downloads = [(c[1], c[2], c[3]) for c in self.vm.calls if c[0] == "download"]
        self.assertEqual(downloads, [
            (_VERSIONS["ollama"]["url"], self.paths.ollama_archive, _VERSIONS["ollama"]["sha256"]),
            (_VERSIONS["cloudflared"]["url"], self.paths.cloudflared, _VERSIONS["cloudflared"]["sha256"]),
        ])
        self.assertFalse(self.paths.ollama_archive.exists())
        self.assertEqual(os.stat(self.paths.cloudflared).st_mode & 0o777, 0o755)

        (upstream,) = self.vm.spawned("upstream")
        self.assertEqual(upstream[1], [str(self.paths.ollama_bin), "serve"])
        for key, value in bootstrap.OLLAMA_SERVE_ENV.items():
            self.assertEqual(upstream[2][key], value)
        self.assertNotIn("OLLAMA_ORIGINS", upstream[2])
        self.assertEqual(self.paths.upstream_pid.read_text(encoding="utf-8").strip(), str(self.vm.procs[0].pid))

        pulls = [c for c in self.vm.calls if c[0] == "http" and c[2].endswith("/api/pull")]
        self.assertEqual(len(pulls), 2)
        warm = [c[2].rsplit("/", 2)[-2:] for c in self.vm.calls if c[0] == "http" and c[1] == "POST"
                and c[2].startswith(bootstrap.OLLAMA_URL) and not c[2].endswith("/api/pull")]
        self.assertEqual(warm, [["api", "chat"], ["v1", "systemone"]])

        (proxy,) = self.vm.spawned("proxy")
        self.assertEqual(proxy[1], [sys.executable, str(_COLAB / "auth_proxy.py")])
        self.assertEqual(proxy[3], (7,))
        self.assertEqual(float(proxy[2]["SESSION_STARTED_AT"]), 1_790_000_000.0)
        self.assertEqual(bootstrap.load_env(proxy[2]).proxy_token, _PROXY_TOKEN)

        (tunnel,) = self.vm.spawned("tunnel")
        self.assertEqual(tunnel[1], bootstrap.tunnel_command(self.paths.cloudflared, "quick"))

        (ready,) = self.ready_events()
        self.assertEqual(ready, {
            "event": "ready",
            "nbh": _NBH,
            "tunnel_url": f"https://{_QUICK_HOST}",
            "versions": {"ollama": "0.35.0", "cloudflared": "2026.9.3"},
            "digests": {model: pin["digest"] for model, pin in _VERSIONS["models"].items()},
            "gpu": {"name": "NVIDIA A100-SXM4-40GB", "vram_mib": 40960, "driver": "550.54.15", **vram},
        })
        self.assertTrue(all(valid for _, _, valid in self.vm.events))
        self.assertEqual(self.vm.unassigns, 0)
        self.assertTrue(all(proc.poll() is None for proc in self.vm.procs), "children must outlive bootstrap")
        # The public URL was tried, with the token, before registering.
        self.assertIn(("http", "GET", f"https://{_QUICK_HOST}/api/version"), self.vm.calls)

    def test_dryrun_skips_the_gpu_and_ollama(self) -> None:
        self.vm.present = set()  # no zstd, no nvidia-smi: a CPU runtime
        self.assertEqual(self.boot(_env(DRYRUN="1")), 0)
        self.assertEqual(self.steps(), _DRYRUN_STEPS)
        self.assertEqual([c for c in self.vm.calls if c[0] == "run"], [])
        self.assertEqual([c[1] for c in self.vm.calls if c[0] == "download"], [_VERSIONS["cloudflared"]["url"]])
        (upstream,) = self.vm.spawned("upstream")
        self.assertEqual(upstream[1], [sys.executable, str(_COLAB / "bootstrap.py"), "--stub-upstream"])
        self.assertEqual(self.vm.spawned("proxy")[0][2]["DRYRUN"], "1")
        (ready,) = self.ready_events()
        self.assertNotIn("gpu", ready)
        self.assertEqual(ready["digests"], {m: p["digest"] for m, p in _VERSIONS["models"].items()})
        self.assertFalse([c for c in self.vm.calls if c[0] == "http" and "/api/pull" in c[2]])

    def test_named_tunnel(self) -> None:
        self.assertEqual(self.boot(_named_env(DRYRUN="1")), 0)
        (tunnel,) = self.vm.spawned("tunnel")
        self.assertEqual(tunnel[1], bootstrap.tunnel_command(self.paths.cloudflared, "named"))
        self.assertEqual(tunnel[2]["TUNNEL_TOKEN"], _TUNNEL_TOKEN)
        self.assertEqual(self.ready_events()[0]["tunnel_url"], f"https://{_NAMED_HOST}")
        self.assertFalse([c for c in self.vm.calls if c[0] == "http" and c[2].endswith("/quicktunnel")])

    def test_http2_fallback_when_quic_never_connects(self) -> None:
        self.vm.tunnel_protocols = {"http2"}
        self.assertEqual(self.boot(_env(DRYRUN="1")), 0)
        first, second = self.vm.spawned("tunnel")
        self.assertNotIn("--protocol", first[1])
        self.assertEqual(_protocol(second[1]), "http2")
        self.assertIn(("terminate", first[1]), self.vm.calls)
        self.assertIn("retrying with --protocol http2", self.log_text)

    def test_zstd_is_installed_when_missing(self) -> None:
        self.vm.present = {"nvidia-smi"}
        self.assertEqual(self.boot(), 0)
        apt = [c for c in self.vm.calls if c[0] == "run" and c[1][0] == "apt-get"]
        self.assertEqual([c[1] for c in apt], [["apt-get", "install", "-y", "-qq", "zstd"]])
        self.assertEqual(apt[0][2]["DEBIAN_FRONTEND"], "noninteractive")

    def test_stale_package_lists_are_updated_once(self) -> None:
        self.vm.present = {"nvidia-smi"}
        failed = subprocess.CompletedProcess([], 100, "", "E: Unable to locate package zstd")
        self.vm.run_results["apt-get install"] = [failed]
        self.assertEqual(self.boot(), 0)
        apt = [c[1][:2] for c in self.vm.calls if c[0] == "run" and c[1][0] == "apt-get"]
        self.assertEqual(apt, [["apt-get", "install"], ["apt-get", "update"], ["apt-get", "install"]])

    def test_the_locked_proxy_packages_are_installed_by_hash_when_any_differs(self) -> None:
        for installed in ({**self.vm.packages, "aiohttp": "3.9.0"}, {**self.vm.packages, "yarl": "1.0.0"}):
            with self.subTest(installed=installed):
                self.vm.calls.clear()
                self.vm.packages = installed
                self.assertEqual(self.boot(_env(DRYRUN="1")), 0)
                pip = [c[1] for c in self.vm.calls if c[0] == "run" and c[1][0] == sys.executable]
                # Wheels only, every file's sha256 checked, nothing resolved at boot.
                self.assertEqual(
                    pip,
                    [
                        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                         "--no-input", "--require-hashes", "--no-deps", "--only-binary=:all:",
                         "-r", str(_PROXY_LOCK)],
                    ],
                )

    def test_nothing_is_installed_when_the_runtime_has_the_locked_packages(self) -> None:
        self.assertEqual(self.boot(_env(DRYRUN="1")), 0)
        self.assertEqual([c for c in self.vm.calls if c[0] == "run" and c[1][0] == sys.executable], [])

    def test_pull_retries_then_succeeds(self) -> None:
        self.vm.pull_failures = {_CHAT: 2}
        self.assertEqual(self.boot(), 0)
        self.assertEqual([c[1] for c in self.vm.calls if c[0] == "sleep" and c[1] in (15, 45)], [15, 45])

    def test_register_retries_422_and_signs_each_try_afresh(self) -> None:
        self.vm.register_statuses = [422, 503]
        self.assertEqual(self.boot(_env(DRYRUN="1")), 0)
        headers = [header for payload, header, _ in self.vm.events if payload["event"] == "ready"]
        self.assertEqual(len(headers), 3)
        self.assertEqual(len(set(headers)), 3)
        self.assertTrue(all(valid for _, _, valid in self.vm.events))

    def test_tunnel_check_failure_is_only_logged(self) -> None:
        self.vm.tunnel_unreachable = True
        self.assertEqual(self.boot(_env(DRYRUN="1")), 0)
        self.assertIn("registering anyway", self.log_text)

    # --- failures ---

    def test_python_package_install_fails(self) -> None:
        self.vm.packages = {}
        self.vm.run_results["pip"] = subprocess.CompletedProcess([], 1, "", "No matching distribution")
        self.assert_failed(self.boot(), "python", "No matching distribution")

    def test_a_lock_without_aiohttp_fails_before_pip(self) -> None:
        code = self.tmp / "code"
        shutil.copytree(_COLAB, code, ignore=shutil.ignore_patterns("__pycache__"))
        (code / bootstrap.PROXY_REQUIREMENTS).write_text("yarl==1.25.1\n", encoding="utf-8")
        self.paths = _paths(self.tmp, code)
        self.assert_failed(self.boot(), "python", "does not pin aiohttp")
        self.assertEqual([c for c in self.vm.calls if c[0] == "run" and c[1][0] == sys.executable], [])

    def test_apt_update_fails(self) -> None:
        self.vm.present = {"nvidia-smi"}
        self.vm.run_results["apt-get install"] = [subprocess.CompletedProcess([], 100, "", "E: nope")]
        self.vm.run_results["apt-get update"] = subprocess.CompletedProcess([], 100, "", "E: Temporary failure")
        self.assert_failed(self.boot(), "zstd", "apt-get update failed (exit 100): E: Temporary failure")

    def test_no_gpu(self) -> None:
        self.vm.present = {"zstd"}
        self.assert_failed(self.boot(), "gpu", "nvidia-smi is missing")

    def test_nvidia_smi_fails(self) -> None:
        self.vm.run_results["nvidia-smi"] = subprocess.CompletedProcess([], 9, "", "NVIDIA-SMI has failed")
        self.assert_failed(self.boot(), "gpu", "NVIDIA-SMI has failed")

    def test_gpu_too_small_or_driver_too_old(self) -> None:
        self.vm.gpu_line = "Tesla T4, 15360, 535.104.05"
        code = self.boot()
        self.assert_failed(code, "gpu", "Tesla T4 has 15360 MiB, under 38000 MiB; NVIDIA driver 535.104.05")
        self.assertFalse([c for c in self.vm.calls if c[0] == "download"])

    def test_tarball_hash_mismatch(self) -> None:
        self.vm.download_error = bootstrap.DownloadError("ollama-linux-amd64.tar.zst has sha256 000000000000")
        self.assert_failed(self.boot(), "ollama_install", "has sha256 000000000000")

    def test_extraction_fails(self) -> None:
        self.vm.run_results["bash"] = subprocess.CompletedProcess([], 2, "", "zstd: corrupted block")
        self.assert_failed(self.boot(), "ollama_install", "extracting ollama-linux-amd64.tar.zst failed (exit 2)")

    def test_unexpected_exception_still_releases(self) -> None:
        self.vm.run_results["bash"] = FileNotFoundError(2, "No such file or directory", "bash")
        self.assert_failed(self.boot(), "ollama_install", "FileNotFoundError")

    def test_ollama_never_answers(self) -> None:
        self.vm.upstream_dead = True
        self.assert_failed(self.boot(), "ollama_start", "did not answer /api/version within 60 s")

    def test_wrong_ollama_version(self) -> None:
        self.vm.upstream_version = "0.34.4"
        self.assert_failed(self.boot(), "ollama_start", "the upstream is Ollama 0.34.4, not 0.35.0")

    def test_pull_fails_three_times(self) -> None:
        self.vm.pull_failures = {_CHAT: 3}
        self.assert_failed(self.boot(), "pull", f"pulling {_CHAT} failed 3 times: HTTP 500 pull model manifest: EOF")
        self.assertEqual(self.vm.pulled, [_CHAT] * 3)

    def test_a_router_that_does_not_pull_boots_without_it(self) -> None:
        self.vm.pull_failures = {_ROUTER: 3}
        del self.vm.tag_digests[_ROUTER]
        self.vm.systemone_status = 404
        self.vm.unloaded.add(_ROUTER)

        self.assertEqual(self.boot(), 0)

        self.assertEqual(self.steps(), _REAL_STEPS)
        (ready,) = self.ready_events()
        self.assertEqual(ready["digests"], {_CHAT: _VERSIONS["models"][_CHAT]["digest"]})
        self.assertNotIn("router_on_gpu", ready["gpu"])
        errors = [line for line in self.log_text.splitlines() if line.startswith("ERROR")]
        self.assertEqual(len(errors), 2, errors)
        self.assertIn(f"pulling {_ROUTER} failed 3 times: HTTP 500 pull model manifest: EOF; booting on", errors[0])
        self.assertIn(f"{_ROUTER} did not load (loading {_ROUTER} failed: HTTP 404", errors[1])
        self.assertEqual(self.vm.unassigns, 0)

    def test_a_chat_model_off_its_pin_stops_the_boot(self) -> None:
        self.vm.tag_digests[_CHAT] = "0" * 64
        self.assert_failed(self.boot(), "pull", f"{_CHAT} has digest 000000000000, not the pinned 22130167c4c2")
        self.assertFalse([c for c in self.vm.calls if c[0] == "http" and c[2].endswith("/api/chat")])

    def test_a_router_missing_after_the_pull_boots_without_it(self) -> None:
        del self.vm.tag_digests[_ROUTER]

        self.assertEqual(self.boot(), 0)

        (ready,) = self.ready_events()
        self.assertEqual(ready["digests"], {_CHAT: _VERSIONS["models"][_CHAT]["digest"]})
        (error,) = [line for line in self.log_text.splitlines() if line.startswith("ERROR")]
        self.assertIn(f"{_ROUTER} has digest (none), not the pinned {_VERSIONS['models'][_ROUTER]['digest']}", error)
        self.assertEqual(self.vm.unassigns, 0)

    def test_a_chat_model_missing_after_the_pull_stops_the_boot(self) -> None:
        del self.vm.tag_digests[_CHAT]
        self.assert_failed(self.boot(), "pull", f"{_CHAT} has digest (none), not the pinned 22130167c4c2")

    def test_a_republished_router_boots_and_reports_its_digest(self) -> None:
        republished = "adfcdbb4070d303fb43fc11fc06a14269ec2bc7136a55369e86d412b86ba2831"
        self.vm.tag_digests[_ROUTER] = republished

        self.assertEqual(self.boot(), 0)

        self.assertEqual(self.steps(), _REAL_STEPS)
        self.assertEqual(self.state.read()["error"], None)
        (ready,) = self.ready_events()
        self.assertEqual(ready["digests"], {_CHAT: _VERSIONS["models"][_CHAT]["digest"], _ROUTER: republished})
        (error,) = [line for line in self.log_text.splitlines() if line.startswith("ERROR")]
        self.assertIn(f"{_ROUTER} has digest {republished}, not the pinned {_VERSIONS['models'][_ROUTER]['digest']}", error)
        # It is still loaded, so the probe's /api/ps check holds.
        warm = [c[2] for c in self.vm.calls if c[0] == "http" and c[2] == bootstrap.OLLAMA_URL + "/v1/systemone"]
        self.assertEqual(len(warm), 1)
        self.assertEqual(self.vm.unassigns, 0)

    def test_the_chat_model_is_pulled_and_loaded_first_whatever_the_pin_order(self) -> None:
        versions = {**_VERSIONS, "models": dict(reversed(list(_VERSIONS["models"].items())))}
        self.assertEqual(list(versions["models"]), [_ROUTER, _CHAT])
        self.env = _env()
        state = bootstrap.StateFile(self.paths.state_file, clock=self.vm.now)
        with self.assertLogs("colab", level="INFO"):
            code = bootstrap.Bootstrap(self.env, versions, paths=self.paths, ops=self.vm, state=state, lock_fd=7).run()
        self.assertEqual(code, 0)
        self.assertEqual(self.vm.pulled, [_CHAT, _ROUTER])
        loads = [c[2].rsplit("/", 1)[-1] for c in self.vm.calls if c[0] == "http" and c[1] == "POST"
                 and c[2].startswith(bootstrap.OLLAMA_URL) and not c[2].endswith("/api/pull")]
        self.assertEqual(loads, ["chat", "systemone"])

    def test_a_router_not_wholly_in_vram_boots_and_reports_it(self) -> None:
        for model in self.vm.stub["ps"]["models"]:
            if model["name"] == _ROUTER:
                model["size_vram"] = 6 << 30

        self.assertEqual(self.boot(), 0)

        self.assertEqual(self.steps(), _REAL_STEPS)
        (ready,) = self.ready_events()
        self.assertEqual(ready["gpu"]["router_on_gpu"], "6144/10110 MiB")
        self.assertEqual(self.state.read()["vram"]["router_on_gpu"], "6144/10110 MiB")
        (error,) = [line for line in self.log_text.splitlines() if line.startswith("ERROR")]
        self.assertIn(f"{_ROUTER} is not fully on the GPU (6144 of 10110 MiB in VRAM); booting on", error)
        self.assertEqual(self.vm.unassigns, 0)

    def test_a_chat_model_not_wholly_in_vram_fails_the_probe(self) -> None:
        for model in self.vm.stub["ps"]["models"]:
            if model["name"] == _CHAT:
                model["size_vram"] = 20 << 30
        self.assert_failed(self.boot(), "probe", f"{_CHAT} is not fully on the GPU (/api/ps: 20480 of 24128 MiB")
        # What it got is in state.json for the heartbeat, next to the error.
        self.assertEqual(self.state.read()["vram"]["chat_on_gpu"], "20480/24128 MiB")

    def test_a_chat_model_that_does_not_stay_loaded_fails_the_probe(self) -> None:
        self.vm.stub["ps"]["models"] = [m for m in self.vm.stub["ps"]["models"] if m["name"] != _CHAT]
        self.assert_failed(self.boot(), "probe", f"{_CHAT} is not loaded (/api/ps)")
        self.assertNotIn("chat_on_gpu", self.state.read()["vram"])

    def test_a_router_that_evicts_the_chat_model_is_unloaded_and_the_boot_goes_on(self) -> None:
        # Ollama evicts idle qwen3.8 when it predicts nimble will not fit
        # beside it (sched.go's preflight); the chat tier must not go with it.
        self.vm.no_room_for_both = True

        self.assertEqual(self.boot(), 0)

        self.assertEqual(self.steps(), _REAL_STEPS)
        loads = [c[2].rsplit("/", 1)[-1] for c in self.vm.calls if c[0] == "http" and c[1] == "POST"
                 and c[2].startswith(bootstrap.OLLAMA_URL) and not c[2].endswith("/api/pull")]
        self.assertEqual(loads, ["chat", "systemone", "generate", "chat"])
        self.assertEqual(self.vm.unloaded, {_ROUTER})
        (error,) = [line for line in self.log_text.splitlines() if line.startswith("ERROR")]
        self.assertIn(f"loading {_ROUTER} evicted {_CHAT} (not enough VRAM for both); unloading {_ROUTER}", error)
        (ready,) = self.ready_events()
        self.assertEqual(ready["gpu"]["chat_on_gpu"], "24128/24128 MiB")
        self.assertNotIn("router_on_gpu", ready["gpu"])
        self.assertEqual(self.vm.unassigns, 0)

    def test_memory_used_is_left_out_when_nvidia_smi_cannot_tell(self) -> None:
        gpu = subprocess.CompletedProcess([], 0, self.vm.gpu_line + "\n", "")
        for failure in (subprocess.CompletedProcess([], 9, "", "failed"), OSError("gone")):
            with self.subTest(failure=failure):
                self.vm = FakeVM()
                self.vm.run_results["nvidia-smi"] = [gpu, failure]
                self.assertEqual(self.boot(), 0)
                (ready,) = self.ready_events()
                self.assertNotIn("used_mib", ready["gpu"])
                self.assertEqual(ready["gpu"]["router_on_gpu"], "10110/10110 MiB")
                self.assertIn("memory.used", self.log_text)

    def test_chat_model_does_not_load(self) -> None:
        self.vm.chat_load = {"done_reason": "stop"}
        self.assert_failed(self.boot(), "warmup", "done_reason 'stop', not 'load'")

    def test_a_router_that_evicts_the_chat_model_and_then_fails_is_left_out_too(self) -> None:
        # Ollama evicts idle qwen3.8 first, then nimble's runner fails to start.
        self.vm.no_room_for_both = True
        self.vm.systemone_status = 500
        self.vm.unloaded.add(_ROUTER)
        original = self.vm._upstream

        def evict_then_fail(method, path, body):
            if (method, path) == ("POST", "/v1/systemone"):
                self.vm.unloaded.add(_CHAT)
            return original(method, path, body)

        self.vm._upstream = evict_then_fail

        self.assertEqual(self.boot(), 0)

        loads = [c[2].rsplit("/", 1)[-1] for c in self.vm.calls if c[0] == "http" and c[1] == "POST"
                 and c[2].startswith(bootstrap.OLLAMA_URL) and not c[2].endswith("/api/pull")]
        self.assertEqual(loads, ["chat", "systemone", "generate", "chat"])
        errors = [line for line in self.log_text.splitlines() if line.startswith("ERROR")]
        self.assertEqual(len(errors), 2, errors)
        self.assertIn(f"loading {_ROUTER} evicted {_CHAT}", errors[1])
        self.assertEqual(self.vm.unassigns, 0)

    def test_a_router_that_does_not_load_boots_without_it(self) -> None:
        self.vm.systemone_status = 500
        self.vm.unloaded.add(_ROUTER)

        self.assertEqual(self.boot(), 0)

        self.assertEqual(self.steps(), _REAL_STEPS)
        (error,) = [line for line in self.log_text.splitlines() if line.startswith("ERROR")]
        self.assertIn(f"{_ROUTER} did not load (loading {_ROUTER} failed: HTTP 500 scoring failed); booting on", error)
        (ready,) = self.ready_events()
        self.assertNotIn("router_on_gpu", ready["gpu"])
        self.assertEqual(self.vm.unassigns, 0)

    def test_proxy_never_answers(self) -> None:
        self.vm.proxy_dies = True
        self.assert_failed(self.boot(_env(DRYRUN="1")), "proxy", "did not answer within 30 s")

    def test_proxy_open_without_a_token(self) -> None:
        self.vm.proxy_needs_token = False
        self.assert_failed(self.boot(_env(DRYRUN="1")), "proxy", "answered HTTP 200 without a token, not 401")

    def test_probe_failure(self) -> None:
        self.vm.stub["show"][_CHAT]["thinking"] = {"values": [False, "low"], "default": "low"}
        self.assert_failed(self.boot(_env(DRYRUN="1")), "probe", "readiness probe: ")

    def test_tunnel_never_connects(self) -> None:
        self.vm.tunnel_protocols = set()
        self.assert_failed(self.boot(_env(DRYRUN="1")), "tunnel", "not ready in 90 s over QUIC or HTTP/2")
        self.assertEqual(len(self.vm.spawned("tunnel")), 2)

    def test_cloudflared_exits_at_once(self) -> None:
        self.vm.cloudflared_exits = True
        start = self.vm.clock
        self.assert_failed(self.boot(_env(DRYRUN="1")), "tunnel", "not ready")
        self.assertLess(self.vm.clock - start, bootstrap.TUNNEL_READY_SECONDS)

    def test_unexpected_quick_tunnel_host(self) -> None:
        self.vm.quick_host = "evil.example.com"
        self.assert_failed(self.boot(_env(DRYRUN="1")), "tunnel", "unexpected quick-tunnel host 'evil.example.com'")

    def test_bot_refuses_the_registration(self) -> None:
        self.vm.register_statuses = [403]
        self.assert_failed(self.boot(_env(DRYRUN="1")), "register", "refused the ready registration (HTTP 403)")
        self.assertEqual(len(self.ready_events()), 1)

    def test_bot_keeps_failing(self) -> None:
        self.vm.register_statuses = [503, 503, 503, 503]
        self.assert_failed(self.boot(_env(DRYRUN="1")), "register", "failed 4 times: HTTP 503")
        self.assertEqual(len(self.ready_events()), 4)
        self.assertEqual([c[1] for c in self.vm.calls if c[0] == "sleep" and c[1] in (5, 10, 20)][-3:], [5, 10, 20])

    def test_bot_unreachable(self) -> None:
        self.vm.bot_unreachable = True
        code = self.boot(_env(DRYRUN="1"))
        self.assertEqual(code, 1)
        self.assertEqual(self.state.writes[-1], ("register", "the ready registration failed 4 times: ConnectionResetError"))
        self.assertEqual(self.vm.events, [])
        self.assertEqual(self.vm.unassigns, 1)
        self.assertIn("could not send the failed event", self.log_text)

    def test_failed_self_unassign_is_logged(self) -> None:
        self.vm.present = {"zstd"}
        self.vm.unassign_status = 500
        self.assertEqual(self.boot(), 1)
        self.assertIn("self-unassign answered HTTP 500", self.log_text)


class RestartUpstreamTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.paths = _paths(self.tmp)
        self.vm = FakeVM()
        bootstrap.start_upstream(_env(), paths=self.paths, ops=self.vm)

    def test_restart_replaces_the_upstream_and_reloads_the_models(self) -> None:
        self.assertTrue(bootstrap.restart_upstream(_env(), _VERSIONS, paths=self.paths, ops=self.vm))
        self.assertEqual([c[0] for c in self.vm.calls if c[0] in ("spawn", "kill_pidfile")],
                         ["spawn", "kill_pidfile", "spawn"])
        self.assertEqual([p.poll() for p in self.vm.procs], [-15, None])
        self.assertEqual(self.paths.upstream_pid.read_text(encoding="utf-8").strip(), str(self.vm.procs[1].pid))
        posts = [c[2] for c in self.vm.calls if c[0] == "http" and c[1] == "POST"]
        self.assertEqual(posts, [bootstrap.OLLAMA_URL + "/api/chat", bootstrap.OLLAMA_URL + "/v1/systemone"])

    def test_dryrun_restart_needs_no_models(self) -> None:
        self.assertTrue(bootstrap.restart_upstream(_env(DRYRUN="1"), _VERSIONS, paths=self.paths, ops=self.vm))
        self.assertEqual(self.vm.spawned("upstream")[-1][1][-1], "--stub-upstream")
        self.assertFalse([c for c in self.vm.calls if c[0] == "http" and c[1] == "POST"])

    def test_restart_fails(self) -> None:
        self.vm.upstream_version = "0.34.4"
        self.assertFalse(bootstrap.restart_upstream(_env(), _VERSIONS, paths=self.paths, ops=self.vm))
        self.vm.upstream_version = _VERSIONS["ollama"]["version"]
        self.vm.chat_load = {"done_reason": "stop"}
        with self.assertLogs("colab", level="WARNING"):
            self.assertFalse(bootstrap.restart_upstream(_env(), _VERSIONS, paths=self.paths, ops=self.vm))

    def test_a_router_that_does_not_come_back_does_not_fail_the_restart(self) -> None:
        # The watchdog releases the VM when a restart fails; chat must not
        # go down with a router that no longer loads.
        self.vm.systemone_status = 500
        with self.assertLogs("colab", level="ERROR") as logs:
            self.assertTrue(bootstrap.restart_upstream(_env(), _VERSIONS, paths=self.paths, ops=self.vm))
        self.assertIn(f"{_ROUTER} did not load", "\n".join(logs.output))


class MainTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.paths = _paths(self.tmp)
        self.vm = FakeVM()

    def write_env(self, values: dict[str, str]) -> None:
        self.paths.env_file.write_text(register.render_bootstrap_env(values), encoding="utf-8")
        os.chmod(self.paths.env_file, 0o600)

    def main(self) -> tuple[int, str]:
        with self.assertLogs("colab", level="DEBUG") as logs:
            code = bootstrap.main([], paths=self.paths, ops=self.vm)
        return code, "\n".join(logs.output)

    def state(self) -> dict:
        return json.loads(self.paths.state_file.read_text(encoding="utf-8"))

    def test_dryrun_through_main(self) -> None:
        self.write_env(_env_values(DRYRUN="1"))
        code, logs = self.main()
        self.assertEqual(code, 0)
        self.assertFalse(self.paths.env_file.exists())
        self.assertEqual(self.state()["step"], "ready")
        self.assertNotIn(_REG_KEY, logs)
        # main released its own lock fd (a real proxy would hold its copy).
        fd = bootstrap.acquire_lock(self.paths.lock_file)
        self.assertIsNotNone(fd)
        os.close(fd)

    def test_missing_env_file(self) -> None:
        code, _ = self.main()
        self.assertEqual(code, 1)
        self.assertEqual(self.state()["step"], "env")
        self.assertIn(".bootstrap.env is missing", self.state()["error"])
        self.assertEqual(self.vm.events, [])  # no key to sign with
        self.assertEqual(self.vm.unassigns, 1)

    def test_invalid_env_file(self) -> None:
        self.write_env(_env_values(NBH="not-a-uuid"))
        code, logs = self.main()
        self.assertEqual(code, 1)
        self.assertFalse(self.paths.env_file.exists())
        self.assertEqual(self.state()["error"], "NBH is not a UUID")
        self.assertNotIn(_REG_KEY, self.paths.state_file.read_text(encoding="utf-8") + logs)
        self.assertEqual((self.vm.events, self.vm.unassigns), ([], 1))

    def test_unparsable_env_file(self) -> None:
        self.paths.env_file.write_text(f"{_REG_KEY}\n", encoding="utf-8")
        code, logs = self.main()
        self.assertEqual(code, 1)
        self.assertIn("line 1", self.state()["error"])
        self.assertNotIn(_REG_KEY, self.paths.state_file.read_text(encoding="utf-8") + logs)

    def test_a_second_bootstrap_exits_quietly(self) -> None:
        held = bootstrap.acquire_lock(self.paths.lock_file)
        self.addCleanup(os.close, held)
        self.write_env(_env_values())
        code, logs = self.main()
        self.assertEqual(code, 0)
        self.assertFalse(self.paths.env_file.exists())  # its secrets are gone anyway
        self.assertFalse(self.paths.state_file.exists())
        self.assertEqual((self.vm.calls, self.vm.unassigns), ([], 0))
        self.assertIn("holds", logs)

    def test_broken_versions_file(self) -> None:
        code_dir = self.tmp / "code"
        code_dir.mkdir()
        (code_dir / "versions.json").write_text('{"ollama": {}}', encoding="utf-8")
        self.paths = _paths(self.tmp, code=code_dir)
        self.write_env(_env_values())
        code, _ = self.main()
        self.assertEqual(code, 1)
        self.assertEqual(self.state()["step"], "versions")
        payload, _, valid = self.vm.events[0]
        self.assertTrue(valid)
        self.assertEqual((payload["event"], payload["step"]), ("failed", "versions"))
        self.assertEqual(self.vm.unassigns, 1)


# --- against the bot's side ----------------------------------------------------


class ServerAgreementTests(unittest.TestCase):
    """colab/register.py against the bot's copy (src/services/colab/protocol.py)."""

    def setUp(self) -> None:
        from src.services.colab import protocol

        self.protocol = protocol

    def test_keys_agree(self) -> None:
        for secret, nbh in ((_SECRET, _NBH), ("密鑰-ünïcode", "11111111-2222-4333-8444-555555555555")):
            self.assertEqual(register.derive_reg_key(secret, nbh), self.protocol.derive_reg_key(secret, nbh))
            self.assertEqual(register.derive_proxy_token(secret, nbh), self.protocol.derive_proxy_token(secret, nbh))

    def test_signatures_agree_both_ways(self) -> None:
        body = register.build_event("failed", _NBH, step="pull", error="EOF")
        ts = 1_790_000_000
        self.assertEqual(register.sign(_REG_KEY, body, ts), self.protocol.signature_for(_REG_KEY, ts, body))
        vm_header = register.signature_header(_REG_KEY, body, ts)
        self.assertEqual(vm_header, self.protocol.sign(_REG_KEY, ts, body))
        parsed = self.protocol.parse_signature(vm_header)
        self.assertEqual((parsed.timestamp, parsed.v1), register.parse_signature_header(vm_header))
        self.assertTrue(self.protocol.signature_matches(_REG_KEY, parsed, body))
        self.assertFalse(self.protocol.signature_matches(_REG_KEY, parsed, body + b" "))
        self.assertTrue(register.verify(_REG_KEY, body, self.protocol.sign(_REG_KEY, ts, body), now=ts))

    def test_pins_and_limits_agree(self) -> None:
        models = _VERSIONS["models"]
        self.assertEqual({m: p["digest"] for m, p in models.items()}, self.protocol.pinned_models())
        self.assertEqual(_VERSIONS["ollama"]["version"], self.protocol.OLLAMA_VERSION)
        self.assertEqual([m for m, p in models.items() if p["role"] == "chat"], self.protocol.chat_models())
        # The one model both sides let run at another digest than its pin,
        # or not wholly in VRAM.
        self.assertEqual([m for m, p in models.items() if not register.must_be_exact(p)], self.protocol.router_models())
        self.assertEqual(json.dumps(models[_CHAT]["thinking"]), json.dumps(self.protocol.CHAT_THINK_VALUES))
        self.assertEqual(register.MAX_BODY_BYTES, self.protocol.MAX_REGISTER_BYTES)
        self.assertEqual(register.MAX_CLOCK_SKEW_SECONDS, self.protocol.MAX_CLOCK_SKEW_SECONDS)
        self.assertEqual(register.SIGNATURE_HEADER, self.protocol.SIGNATURE_HEADER)
        self.assertEqual(register.EVENTS, self.protocol.EVENTS)

    def test_the_router_warm_up_asks_in_the_bots_request_shape(self) -> None:
        from src.models.agent_request import AgentRequest
        from src.providers.systemone import build_request

        sent: list[dict] = []

        class Upstream:
            def http(self, method, url, body=None, headers=None, timeout=10.0):
                sent.append(json.loads(body))
                return 200, json.dumps({"answers": {"ready": {"type": "noul", "noul": 0.5}}}).encode()

        bootstrap._load(_ROUTER, _VERSIONS["models"][_ROUTER], ops=Upstream(), base=bootstrap.OLLAMA_URL)

        (warm,) = sent
        bot = build_request(_ROUTER, AgentRequest(text="你好"))
        self.assertEqual(set(warm), set(bot))
        # Whatever type Ollama gives the state, the warm-up passes where the router's requests do.
        self.assertEqual({k: type(v) for k, v in warm["state"].items()}, {k: type(v) for k, v in bot["state"].items()})

    def test_tunnel_hosts_agree(self) -> None:
        for host in (_QUICK_HOST, "x.trycloudflare.com", "trycloudflare.com", "a.b.trycloudflare.com",
                     "a_b.trycloudflare.com", "x.trycloudflare.com.evil.com"):
            with self.subTest(host=host):
                self.assertEqual(register.tunnel_host_ok(host), bool(self.protocol.QUICK_TUNNEL_HOST.fullmatch(host)))
        # What the VM registers passes the bot's URL rule, quick and named.
        self.assertEqual(self.protocol.check_tunnel_url(f"https://{_QUICK_HOST}"), _QUICK_HOST)
        self.assertEqual(self.protocol.check_tunnel_url(f"https://{_NAMED_HOST}", _NAMED_HOST), _NAMED_HOST)


@unittest.skipUnless(_HAS_AIOHTTP, "aiohttp is not installed")
class _RoundTripTestCase(unittest.IsolatedAsyncioTestCase):
    """The VM's signed events through the bot's registrar, whose readiness
    probe goes through the real auth proxy to the DRYRUN stub serving
    :attr:`versions`."""

    versions = _VERSIONS

    async def asyncSetUp(self) -> None:
        import httpx
        from aiohttp import test_utils

        from src.services.colab.endpoint import ColabEndpoint
        from src.services.colab.registration import ColabRegistrar
        from src.services.colab.state import BOOTING, ColabState, InMemoryColabStateStore

        self.stub = bootstrap.make_stub_server(("127.0.0.1", 0), self.versions)
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        config = auth_proxy.ProxyConfig(proxy_token=_PROXY_TOKEN, upstream=f"http://127.0.0.1:{self.stub.server_address[1]}")
        self.proxy_server = test_utils.TestServer(auth_proxy.build_app(config, auth_proxy.Activity(0.0)))
        await self.proxy_server.start_server()
        proxy_base = str(self.proxy_server.make_url("")).rstrip("/")
        self.forwarded: list[str] = []

        async def through_the_tunnel(request: httpx.Request) -> httpx.Response:
            # The tunnel: https://<quick host>/... lands on the auth proxy.
            self.forwarded.append(f"{request.url.host}{request.url.path}")
            async with httpx.AsyncClient(trust_env=False) as client:
                answer = await client.request(
                    request.method, proxy_base + request.url.path, headers=request.headers.raw, content=request.content
                )
            return httpx.Response(answer.status_code, content=answer.content,
                                  headers={"Content-Type": answer.headers.get("content-type", "")})

        self.store = InMemoryColabStateStore(ColabState(status=BOOTING, nbh=_NBH))
        self.endpoint = ColabEndpoint(_SECRET, self.store, transport=httpx.MockTransport(through_the_tunnel))

        async def public(host: str, port: int) -> list[str]:
            return ["104.16.230.132"]

        self.registrar = ColabRegistrar(self.endpoint, secret=_SECRET, resolver=public)

    async def asyncTearDown(self) -> None:
        await self.endpoint.close()
        await self.proxy_server.close()
        self.stub.shutdown()
        self.stub.server_close()

    def vm_sends(self, event: str, **fields) -> tuple[bytes, str]:
        sent = []

        def capture(method, url, body, headers, timeout):
            sent.append((body, headers[register.SIGNATURE_HEADER]))
            return 200, b"{}"

        body = register.build_event(event, _NBH, **fields)
        register.post_event(_BOT_URL, _REG_KEY, body, send=capture)
        return sent[0]


class RegistrationRoundTripTests(_RoundTripTestCase):
    async def test_ready_event_and_probe_reach_ready(self) -> None:
        body, header = self.vm_sends(
            "ready",
            tunnel_url=f"https://{_QUICK_HOST}",
            versions={"ollama": "0.35.0", "cloudflared": "2026.9.3"},
            digests={m: p["digest"] for m, p in _VERSIONS["models"].items()},
        )
        result = await self.registrar.handle(body, header)
        self.assertEqual((result.status_code, result.body), (200, {"ok": True, "status": "READY"}))
        state = await self.store.load()
        self.assertEqual((state.status, state.tunnel_url), ("READY", f"https://{_QUICK_HOST}"))
        self.assertIn(f"{_QUICK_HOST}/api/show", self.forwarded)
        # The bot's probe used the bearer the VM's proxy holds.
        self.assertEqual(self.endpoint.bearer(_NBH), _PROXY_TOKEN)

    async def test_failed_and_stopping_events_drain_the_session(self) -> None:
        sent = []

        def capture(method, url, body, headers, timeout):
            sent.append((body, headers[register.SIGNATURE_HEADER]))
            return 200, b"{}"

        env = _env()
        self.assertEqual(bootstrap.send_event_quietly(env, "failed", step="pull", error="EOF", send=capture), 200)
        result = await self.registrar.handle(*sent[0])
        self.assertEqual(result.status_code, 200)
        state = await self.store.load()
        self.assertEqual((state.status, state.boot_step, state.failures), ("DRAINING", "pull", 1))
        self.assertIn("EOF", state.last_error)

        time.sleep(1.01)  # a new second: a new signature (the bot refuses replays)
        bootstrap.send_event_quietly(env, "stopping", step="watchdog", error="idle for 1021 s", send=capture)
        result = await self.registrar.handle(*sent[1])
        self.assertEqual(result.status_code, 200)

    async def test_the_bot_refuses_what_a_stranger_signs(self) -> None:
        body = register.build_event("stopping", _NBH)
        header = register.signature_header(register.derive_reg_key("not-the-secret", _NBH), body, int(time.time()))
        self.assertEqual((await self.registrar.handle(body, header)).status_code, 403)


_REPUBLISHED = "adfcdbb4070d303fb43fc11fc06a14269ec2bc7136a55369e86d412b86ba2831"


class RepublishedRouterRoundTripTests(_RoundTripTestCase):
    """nimble's tag republished under another digest than the pin: the VM
    boots on, the bot's session is READY, and nimble routes none of it."""

    versions = {
        **_VERSIONS,
        "models": {**_VERSIONS["models"], _ROUTER: {**_VERSIONS["models"][_ROUTER], "digest": _REPUBLISHED}},
    }

    async def test_ready_without_the_router(self) -> None:
        from types import SimpleNamespace

        from src.models.agent_request import AgentRequest
        from src.providers.fallback_chain import Tier
        from src.providers.systemone import RouterGates, SystemOneRouter

        pinned = _VERSIONS["models"][_ROUTER]["digest"]
        body, header = self.vm_sends(
            "ready",
            tunnel_url=f"https://{_QUICK_HOST}",
            versions={"ollama": "0.35.0", "cloudflared": "2026.9.3"},
            digests={_CHAT: _VERSIONS["models"][_CHAT]["digest"], _ROUTER: _REPUBLISHED},
        )
        result = await self.registrar.handle(body, header)

        self.assertEqual((result.status_code, result.body), (200, {"ok": True, "status": "READY"}))
        state = await self.store.load()
        self.assertEqual(state.digests[_ROUTER], _REPUBLISHED)
        self.assertEqual(self.endpoint.unpinned(), {_ROUTER: (pinned, _REPUBLISHED)})
        # The stub has it wholly in VRAM: only the digest keeps it out.
        self.assertEqual(self.endpoint.off_gpu(), {})
        # The live gate's probe still names it.
        self.assertEqual(
            await self.endpoint.probe(f"https://{_QUICK_HOST}", _NBH),
            [f"{_ROUTER} digest {_REPUBLISHED[:12]}, expected {pinned[:12]}"],
        )
        gates = RouterGates(confidence=0.75, think_on=0.85, think_off=0.15)
        router = SystemOneRouter(self.endpoint, Tier(SimpleNamespace(name="colab"), _ROUTER, timeout_s=6), gates=gates)
        forwarded = len(self.forwarded)
        with self.assertLogs("linebot", "ERROR"):
            self.assertIsNone(await router.route(AgentRequest(text="你覺得這個計畫如何")))
        self.assertEqual(len(self.forwarded), forwarded)
