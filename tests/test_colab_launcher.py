"""The launcher: what it uploads, the env file it writes (silently) and the
detached start it checks, including running both cells for real in a temp dir."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from colab import VM_FILES, bootstrap, register
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
    VM_DIR,
    ColabLauncher,
    LaunchError,
    LaunchResult,
    LaunchSettings,
    env_cell,
    launched_pid,
    start_cell,
)
from tests.colab_fakes import NBH, SECRET, FakeControlPlane, runtime

_REPO = Path(__file__).resolve().parents[1]
_BOT = "https://linebot-abc-uw.a.run.app"


def _launcher(cp: FakeControlPlane | None = None, *, bot_url: str = _BOT, **settings) -> tuple[ColabLauncher, FakeControlPlane]:
    cp = cp or FakeControlPlane()
    launcher = ColabLauncher(cp, secret=SECRET, bot_url=lambda: bot_url, settings=LaunchSettings(**settings))
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

        self.assertEqual(result, LaunchResult("kernel-1", "session-1", 4242))
        self.assertEqual(cp.names(), ["upload", "run_cell", "run_cell"])
        [files] = cp.uploads
        self.assertEqual(sorted(files), sorted(f"content/{name}" for name in VM_FILES))
        self.assertTrue(all(Path(path).is_file() for path in files.values()))
        _, _, env_code, kernel, _, silent, budget = cp.calls[1]
        self.assertEqual((kernel, silent, budget), ("", True, 30.0))
        self.assertIn(protocol.derive_reg_key(SECRET, NBH), env_code)
        _, _, start_code, kernel, session, silent, _ = cp.calls[2]
        # The start runs in the kernel the env cell made, and not silently.
        self.assertEqual((kernel, session, silent), ("kernel-1", "session-1", False))
        self.assertIn("start_new_session=True", start_code)
        self.assertNotIn(protocol.derive_reg_key(SECRET, NBH), start_code)

    async def test_no_url_means_no_colab_calls(self) -> None:
        launcher, cp = _launcher(bot_url="")
        with self.assertRaises(LaunchError):
            await launcher.launch(runtime(), NBH)
        self.assertEqual(cp.calls, [])

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
