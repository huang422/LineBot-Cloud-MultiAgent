"""The launcher: the Ollama release it picks, what it uploads, the env file
it writes (silently) and the detached start it checks, including running
both cells for real in a temp dir."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

from colab import VERSIONS_FILE, VM_FILES, bootstrap, register
from src.config import Settings
from src.services.colab import protocol
from src.services.colab.control_plane import (
    CELL_ERROR,
    CELL_OK,
    CellResult,
    TransientControlPlaneError,
)
from src.services.colab.launcher import (
    LAUNCH_SENTINEL,
    MISSING_ENV_SENTINEL,
    OLLAMA_ASSET,
    OLLAMA_LATEST_RELEASE_URL,
    RELEASE_PICKS_KEPT,
    VM_DIR,
    ColabLauncher,
    LaunchError,
    LaunchResult,
    LaunchSettings,
    OllamaReleases,
    env_cell,
    launched_pid,
    start_cell,
)
from tests.colab_fakes import NBH, SECRET, FakeControlPlane, runtime

_REPO = Path(__file__).resolve().parents[1]
_BOT = "https://linebot-abc-uw.a.run.app"
_SHA = "ab" * 32
_FALLBACK = register.load_versions(VERSIONS_FILE)["ollama"]


def _release(tag: str = "v0.41.2", **asset_changes) -> dict:
    """A GitHub release object as /releases/latest gives it (trimmed)."""
    asset = {
        "name": OLLAMA_ASSET,
        "browser_download_url": f"https://github.com/ollama/ollama/releases/download/{tag}/{OLLAMA_ASSET}",
        "digest": f"sha256:{_SHA}",
        "size": 1_500_000_000,
        **asset_changes,
    }
    other = {
        "name": "ollama-linux-arm64.tar.zst",
        "browser_download_url": f"https://github.com/ollama/ollama/releases/download/{tag}/ollama-linux-arm64.tar.zst",
        "digest": "sha256:" + "cd" * 32,
        "size": 1_400_000_000,
    }
    return {"tag_name": tag, "draft": False, "prerelease": False, "assets": [other, asset]}


class _GitHub:
    """Stands in for api.github.com: answers *answer* (a release, an
    httpx.Response, an exception or a coroutine function) and counts."""

    def __init__(self, answer=None) -> None:
        self.answer = _release() if answer is None else answer
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answer
        if callable(answer):
            answer = await answer(request)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def releases(self) -> OllamaReleases:
        return OllamaReleases(transport=httpx.MockTransport(self))


def _launcher(
    cp: FakeControlPlane | None = None, *, bot_url: str = _BOT, github: _GitHub | None = None, **settings
) -> tuple[ColabLauncher, FakeControlPlane]:
    cp = cp or FakeControlPlane()
    launcher = ColabLauncher(
        cp,
        secret=SECRET,
        bot_url=lambda: bot_url,
        settings=LaunchSettings(**settings),
        releases=(github or _GitHub()).releases(),
    )
    return launcher, cp


class BootstrapEnvTests(unittest.TestCase):
    def test_the_vm_accepts_the_env_file(self) -> None:
        launcher, _ = _launcher(idle_ttl_seconds=600, max_session_seconds=7200)

        values = register.parse_bootstrap_env(launcher.bootstrap_env(NBH))
        env = bootstrap.load_env(values)

        self.assertEqual(env.reg_key, protocol.derive_reg_key(SECRET, NBH))
        self.assertEqual(env.proxy_token, protocol.derive_proxy_token(SECRET, NBH))
        self.assertEqual((env.bot_url, env.nbh, env.idle_ttl, env.max_session), (_BOT, NBH, 600, 7200))
        self.assertEqual((env.tunnel_mode, env.tunnel_token, env.opencode_enabled, env.dryrun), ("quick", "", False, False))
        self.assertNotIn("TUNNEL_TOKEN", values)
        self.assertLessEqual(set(values), set(register.BOOTSTRAP_ENV_KEYS))

    def test_named_tunnel_and_dry_run(self) -> None:
        launcher, _ = _launcher(tunnel_mode="named", tunnel_host="ollama.example.org", tunnel_token="tunnel-secret", dry_run=True)

        env = bootstrap.load_env(register.parse_bootstrap_env(launcher.bootstrap_env(NBH)))

        self.assertEqual((env.tunnel_mode, env.tunnel_host, env.tunnel_token, env.dryrun), ("named", "ollama.example.org", "tunnel-secret", True))

    def test_settings_come_from_the_app_settings(self) -> None:
        settings = Settings(
            _env_file=None,
            colab_idle_ttl_seconds=600,
            colab_tunnel_mode="named",
            colab_tunnel_host="ollama.example.org",
            colab_tunnel_token="t",
            colab_dry_run=True,
        )
        self.assertEqual(
            LaunchSettings.from_settings(settings),
            LaunchSettings(600, 39600, "named", "ollama.example.org", "t", False, True),
        )
        self.assertNotIn("tunnel_token", repr(LaunchSettings(tunnel_token="t")))

    def test_no_public_https_url_no_launch(self) -> None:
        for url in ("", "http://linebot.example"):
            launcher, cp = _launcher(bot_url=url)
            with self.subTest(url=url), self.assertRaisesRegex(LaunchError, "PUBLIC_BASE_URL"):
                launcher.bootstrap_env(NBH)

    def test_a_line_break_is_refused_without_quoting_the_value(self) -> None:
        launcher, _ = _launcher(tunnel_mode="named", tunnel_host="h.example.org", tunnel_token="abc\nEVIL=1")
        with self.assertRaises(LaunchError) as caught:
            launcher.bootstrap_env(NBH)
        self.assertNotIn("abc", str(caught.exception))


class LaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_silent_env_then_detached_start(self) -> None:
        launcher, cp = _launcher()
        vm = runtime("ep-1")

        result = await launcher.launch(vm, NBH)

        self.assertEqual(result, LaunchResult("kernel-1", "session-1", 4242, "0.41.2"))
        self.assertEqual(cp.names(), ["upload", "run_cell", "run_cell"])
        [files] = cp.uploads
        [contents] = cp.uploaded
        self.assertEqual(sorted(files), sorted(f"content/{name}" for name in VM_FILES))
        for name in VM_FILES:
            if name != "versions.json":
                self.assertEqual(contents[f"content/{name}"], (VM_DIR / name).read_bytes(), name)
        # The temporary versions.json is gone once it was uploaded.
        self.assertFalse(Path(files["content/versions.json"]).exists())
        _, _, env_code, kernel, _, silent, budget = cp.calls[1]
        self.assertEqual((kernel, silent, budget), ("", True, 30.0))
        self.assertIn(protocol.derive_reg_key(SECRET, NBH), env_code)
        _, _, start_code, kernel, session, silent, _ = cp.calls[2]
        # The start runs in the kernel the env cell made, and not silently.
        self.assertEqual((kernel, session, silent), ("kernel-1", "session-1", False))
        self.assertIn("start_new_session=True", start_code)
        self.assertNotIn(protocol.derive_reg_key(SECRET, NBH), start_code)

    async def test_no_url_means_no_colab_calls(self) -> None:
        github = _GitHub()
        launcher, cp = _launcher(bot_url="", github=github)
        with self.assertRaises(LaunchError):
            await launcher.launch(runtime(), NBH)
        self.assertEqual((cp.calls, github.requests), ([], []))

    async def test_the_vm_gets_the_latest_release_in_versions_json(self) -> None:
        launcher, cp = _launcher()

        await launcher.launch(runtime(), NBH)

        [contents] = cp.uploaded
        uploaded = json.loads(contents["content/versions.json"])
        # What the VM reads (register.load_versions) and installs.
        register.validate_versions(uploaded)
        self.assertEqual(
            uploaded["ollama"],
            {
                "version": "0.41.2",
                "url": f"https://github.com/ollama/ollama/releases/download/v0.41.2/{OLLAMA_ASSET}",
                "sha256": _SHA,
                "size": 1_500_000_000,
                "min_version": _FALLBACK["min_version"],
            },
        )
        repository = register.load_versions(VERSIONS_FILE)
        self.assertEqual({k: v for k, v in uploaded.items() if k != "ollama"}, {k: v for k, v in repository.items() if k != "ollama"})
        # The stub upstream (DRYRUN) and the VM's probe report what it names.
        self.assertEqual(bootstrap.stub_payloads(uploaded)["version"], {"version": "0.41.2"})

    async def test_a_broken_versions_json_is_a_launch_error_before_any_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vm_dir = Path(tmp)
            for name in VM_FILES:
                (vm_dir / name).write_bytes((VM_DIR / name).read_bytes())
            broken = json.loads((VM_DIR / "versions.json").read_text(encoding="utf-8"))
            del broken["ollama"]["min_version"]
            (vm_dir / "versions.json").write_text(json.dumps(broken), encoding="utf-8")
            cp = FakeControlPlane()
            launcher = ColabLauncher(
                cp, secret=SECRET, bot_url=lambda: _BOT, settings=LaunchSettings(), vm_dir=vm_dir,
                releases=_GitHub().releases(),
            )

            with self.assertRaisesRegex(LaunchError, "versions.json: .*min_version"):
                await launcher.launch(runtime(), NBH)

        self.assertEqual(cp.calls, [])

    async def test_without_a_usable_release_the_vm_gets_the_fallback(self) -> None:
        launcher, cp = _launcher(github=_GitHub(httpx.Response(500)))

        with self.assertLogs("linebot", "WARNING"):
            result = await launcher.launch(runtime(), NBH)

        [contents] = cp.uploaded
        self.assertEqual(json.loads(contents["content/versions.json"]), register.load_versions(VERSIONS_FILE))
        self.assertEqual(result.ollama_version, _FALLBACK["version"])

    async def test_a_failed_env_cell_is_a_launch_error_without_secrets(self) -> None:
        launcher, cp = _launcher()
        secret_error = f"NameError: {protocol.derive_proxy_token(SECRET, NBH)}"
        cp.cell_results = [CellResult(CELL_ERROR, "kernel-1", "session-1", error=secret_error)]

        with self.assertRaises(LaunchError) as caught:
            await launcher.launch(runtime(), NBH)

        self.assertEqual(str(caught.exception), "writing the bootstrap env file ended error")
        self.assertEqual(cp.names(), ["upload", "run_cell"])

    async def test_no_sentinel_no_launch(self) -> None:
        for name, cell in (
            ("env file missing", CellResult(CELL_OK, "k", "s", stdout=f"{MISSING_ENV_SENTINEL}\n")),
            ("cell raised", CellResult(CELL_ERROR, "k", "s", error="OSError: no space")),
            ("no output", CellResult(CELL_OK, "k", "s")),
            ("timed out", CellResult("timeout", "k", "s", error="no reply within the budget")),
        ):
            launcher, cp = _launcher()
            cp.cell_results = [CellResult(CELL_OK, "k", "s"), cell]
            with self.subTest(name), self.assertRaisesRegex(LaunchError, "did not start"):
                await launcher.launch(runtime(), NBH)

    async def test_colab_errors_pass_through_for_a_retry(self) -> None:
        launcher, cp = _launcher()
        cp.cell_results = [TransientControlPlaneError("run_cell: proxy 502")]
        with self.assertRaises(TransientControlPlaneError):
            await launcher.launch(runtime(), NBH)


class OllamaReleaseTests(unittest.IsolatedAsyncioTestCase):
    """The pick: the latest release when it is usable, else the fallback
    (with a WARNING saying why); never the network."""

    async def test_the_latest_release(self) -> None:
        github = _GitHub()

        with self.assertLogs("linebot", "INFO") as logs:
            entry = await github.releases().pick(_FALLBACK, NBH)

        self.assertEqual(entry["version"], "0.41.2")
        self.assertEqual((entry["sha256"], entry["size"], entry["min_version"]), (_SHA, 1_500_000_000, _FALLBACK["min_version"]))
        [request] = github.requests
        self.assertEqual(str(request.url), OLLAMA_LATEST_RELEASE_URL)
        self.assertEqual(request.headers["Accept"], "application/vnd.github+json")
        self.assertIn("Colab launch: Ollama 0.41.2, the latest release", logs.output[0])
        self.assertTrue(all(record.levelname == "INFO" for record in logs.records))

    async def test_the_minimum_itself_is_used(self) -> None:
        entry = await _GitHub(_release(f"v{_FALLBACK['min_version']}")).releases().pick(_FALLBACK, NBH)
        self.assertEqual(entry["version"], _FALLBACK["min_version"])

    async def test_anything_unusable_picks_the_fallback_and_says_why(self) -> None:
        def without(key: str) -> dict:
            release = _release()
            del release[key]
            return release

        cases = {
            "an HTTP error": (httpx.Response(502), "HTTP 502"),
            "a rate limit": (httpx.Response(403, headers={"x-ratelimit-remaining": "0"}), "HTTP 403 (rate limited)"),
            "a secondary rate limit": (httpx.Response(429), "HTTP 429 (rate limited)"),
            "no connection": (httpx.ConnectError("refused"), "GitHub unreachable (ConnectError)"),
            "a read timeout": (httpx.ReadTimeout("slow"), "GitHub unreachable (ReadTimeout)"),
            "anything unexpected": (RuntimeError("boom"), "RuntimeError: boom"),
            "not JSON": (httpx.Response(200, content=b"<html>"), "not JSON"),
            "not an object": ([], "not an object"),
            "a draft": ({**_release(), "draft": True}, "a draft or a pre-release"),
            "a pre-release": ({**_release(), "prerelease": True}, "a draft or a pre-release"),
            "no draft flag": (without("draft"), "a draft or a pre-release"),
            "a release candidate tag": (_release("v0.41.0-rc1"), "is not a release version"),
            "no tag": (without("tag_name"), "is not a release version"),
            "no assets": (without("assets"), f"has no {OLLAMA_ASSET}"),
            "no amd64 asset": ({**_release(), "assets": _release()["assets"][:1]}, f"has no {OLLAMA_ASSET}"),
            "a foreign URL": (_release(browser_download_url=f"https://evil.example/{OLLAMA_ASSET}"), "is not on"),
            "a lookalike URL": (
                _release(browser_download_url=f"https://github.com/ollama/ollama-fork/releases/download/v0.41.2/{OLLAMA_ASSET}"),
                "is not on",
            ),
            "no digest": (_release(digest=None), "has no sha256 digest"),
            "a digest of another kind": (_release(digest="md5:" + "a" * 32), "has no sha256 digest"),
            "a short digest": (_release(digest="sha256:" + "a" * 63), "has no sha256 digest"),
            "an upper-case digest": (_release(digest="sha256:" + "A" * 64), "has no sha256 digest"),
            "no size": (_release(size=None), "has no size"),
            "a zero size": (_release(size=0), "has no size"),
            "a boolean size": (_release(size=True), "has no size"),
            "an older release": (_release("v0.40.3"), "is older than 0.40.4"),
        }
        for name, (answer, why) in cases.items():
            with self.subTest(name), self.assertLogs("linebot", "WARNING") as logs:
                entry = await _GitHub(answer).releases().pick(_FALLBACK, NBH)
                self.assertEqual(entry, _FALLBACK)
                self.assertIn(f"Colab launch: Ollama {_FALLBACK['version']}, the pinned fallback", logs.output[0])
                self.assertIn(why, logs.output[0])

    async def test_no_answer_within_the_time_limit_picks_the_fallback(self) -> None:
        async def hang(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(5)
            return httpx.Response(200, json=_release())

        with mock.patch("src.services.colab.launcher.RELEASE_TIMEOUT_SECONDS", 0.05), self.assertLogs("linebot", "WARNING") as logs:
            entry = await _GitHub(hang).releases().pick(_FALLBACK, NBH)

        self.assertEqual(entry, _FALLBACK)
        self.assertIn("GitHub did not answer within 0.05s", logs.output[0])

    async def test_a_session_keeps_its_pick_and_the_next_session_asks_again(self) -> None:
        github = _GitHub()
        releases = github.releases()

        first = await releases.pick(_FALLBACK, "session-1")
        # A launch retry of the same session gets the same entry, even after
        # GitHub replaced the asset (a new digest) in between.
        github.answer = _release(digest="sha256:" + "ef" * 32)
        self.assertEqual(await releases.pick(_FALLBACK, "session-1"), first)
        self.assertEqual(len(github.requests), 1)
        # The next session, say after that session's download failed its
        # sha256 check, asks GitHub again and gets the new digest.
        self.assertEqual((await releases.pick(_FALLBACK, "session-2"))["sha256"], "ef" * 32)
        self.assertEqual(len(github.requests), 2)

    async def test_a_session_on_the_fallback_keeps_it(self) -> None:
        github = _GitHub(httpx.Response(503))
        releases = github.releases()

        with self.assertLogs("linebot", "WARNING"):
            await releases.pick(_FALLBACK, "session-1")
        github.answer = _release()
        # Its bootstrap may already be running from the fallback's upload.
        self.assertEqual(await releases.pick(_FALLBACK, "session-1"), _FALLBACK)
        self.assertEqual((await releases.pick(_FALLBACK, "session-2"))["version"], "0.41.2")
        self.assertEqual(len(github.requests), 2)

    async def test_only_the_last_sessions_are_kept(self) -> None:
        github = _GitHub()
        releases = github.releases()
        for n in range(RELEASE_PICKS_KEPT + 1):
            await releases.pick(_FALLBACK, f"session-{n}")
        await releases.pick(_FALLBACK, f"session-{RELEASE_PICKS_KEPT}")
        self.assertEqual(len(github.requests), RELEASE_PICKS_KEPT + 1)
        await releases.pick(_FALLBACK, "session-0")
        self.assertEqual(len(github.requests), RELEASE_PICKS_KEPT + 2)

    async def test_a_returned_entry_is_a_copy(self) -> None:
        releases = _GitHub().releases()
        (await releases.pick(_FALLBACK, NBH))["version"] = "9.9.9"
        self.assertEqual((await releases.pick(_FALLBACK, NBH))["version"], "0.41.2")


class CellCodeTests(unittest.TestCase):
    """Both cells run here for real, pointed at a temporary /content."""

    def test_the_env_cell_writes_the_file_0600_before_any_byte(self) -> None:
        with tempfile.TemporaryDirectory() as content:
            target = Path(content) / ".bootstrap.env"
            target.write_text("OLD=1\n" * 10, encoding="utf-8")
            os.chmod(target, 0o644)
            text = register.render_bootstrap_env({"NBH": NBH, "BOT_URL": _BOT, "COLAB_PROXY_TOKEN": "x'y\"z\\"})

            exec(compile(env_cell(text, content), "<env cell>", "exec"), {})

            self.assertEqual(target.read_text(encoding="utf-8"), text)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

    def test_the_start_cell_starts_bootstrap_detached_only_with_an_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as content:
            marker = Path(content) / "started"
            (Path(content) / "bootstrap.py").write_text(
                f"import os, pathlib\npathlib.Path({str(marker)!r}).write_text(str(os.getsid(0)))\nprint('bootstrap ran')\n",
                encoding="utf-8",
            )
            code = start_cell(content)

            missing = self._run(code)
            self.assertEqual(missing.strip(), MISSING_ENV_SENTINEL)
            self.assertFalse(marker.exists())

            (Path(content) / ".bootstrap.env").write_text("NBH=x\n", encoding="utf-8")
            started = self._run(code)
            pid = launched_pid(started)
            self.assertIsNotNone(pid)
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            # Its own session (start_new_session): the kernel's end is not its end.
            self.assertEqual(marker.read_text(), str(pid))
            self.assertIn("bootstrap ran", (Path(content) / "bootstrap.log").read_text())

    @staticmethod
    def _run(code: str) -> str:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=True).stdout

    def test_the_sentinel(self) -> None:
        self.assertEqual(launched_pid(f"x\n{LAUNCH_SENTINEL} 123\n"), 123)
        self.assertIsNone(launched_pid(f"{LAUNCH_SENTINEL}\n"))
        self.assertIsNone(launched_pid(f"echo {LAUNCH_SENTINEL} 1"))


class ShippingTests(unittest.TestCase):
    def test_the_vm_files_are_in_the_image(self) -> None:
        self.assertEqual(VM_DIR, _REPO / "colab")
        for name in VM_FILES:
            self.assertTrue((VM_DIR / name).is_file(), name)
        ignored = (_REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertFalse([line for line in ignored if line.strip().strip("/").startswith(("colab", "**/colab"))])


if __name__ == "__main__":
    unittest.main()
