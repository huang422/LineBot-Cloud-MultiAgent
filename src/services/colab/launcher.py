"""Starts the VM side on a freshly assigned Colab runtime (ADR-0010, colab/README.md).

Three steps through the control plane (:mod:`.control_plane`), all in one
kernel whose ids the controller keeps for its keep-alive cells:

1. Upload ``colab.VM_FILES`` (bootstrap.py, auth_proxy.py, register.py,
   versions.json, proxy-requirements.lock) flat into ``/content`` through
   the Jupyter Contents API.
2. Write ``/content/.bootstrap.env`` (mode 0600; ``register.BOOTSTRAP_ENV_KEYS``)
   with a *silent* execute: no output, no IPython history. The values never
   pass through ``colab exec --env``, argv, a log or the state document; the
   registration key and the proxy bearer are derived from
   ``COLAB_SHARED_SECRET`` and the session's ``nbh`` (:mod:`.protocol`), and
   bootstrap.py deletes the file as soon as it has read it.
3. Start ``python /content/bootstrap.py`` detached (``start_new_session``,
   output in /content/bootstrap.log) and print ``BOOTSTRAP_LAUNCHED <pid>``,
   only when the env file is there. A cell that fails still "succeeds" as
   an execute, so the launch counts only when that line comes back.

A failure of any step raises :class:`LaunchError` (a boot failure) or a
:class:`~.control_plane.ControlPlaneError` (Colab trouble; retried).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colab import VM_FILES, register
from src.services.colab import protocol

# The VM package in this repository (and in the image).
VM_DIR = Path(__file__).resolve().parents[3] / "colab"
CONTENT_DIR = "/content"
ENV_FILE_NAME = ".bootstrap.env"
LAUNCH_SENTINEL = "BOOTSTRAP_LAUNCHED"
MISSING_ENV_SENTINEL = "BOOTSTRAP_ENV_MISSING"
CELL_BUDGET_SECONDS = 30.0
_LAUNCHED = re.compile(rf"^{LAUNCH_SENTINEL} (\d+)$", re.MULTILINE)


class LaunchError(Exception):
    """The VM side did not start; the message never carries a secret."""


@dataclass(frozen=True)
class LaunchResult:
    kernel_id: str
    session_id: str
    pid: int


@dataclass(frozen=True)
class LaunchSettings:
    """The VM's settings (Settings: COLAB_IDLE_TTL_SECONDS, … COLAB_DRY_RUN)."""

    idle_ttl_seconds: int = 900
    max_session_seconds: int = 39600
    tunnel_mode: str = "quick"
    tunnel_host: str = ""
    tunnel_token: str = field(default="", repr=False)
    # The auth proxy's /v1 routes for a future opencode client; no setting
    # yet (colab/README.md "opencode").
    opencode_enabled: bool = False
    dry_run: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> LaunchSettings:
        return cls(
            idle_ttl_seconds=int(settings.colab_idle_ttl_seconds),
            max_session_seconds=int(settings.colab_max_session_seconds),
            tunnel_mode=settings.colab_tunnel_mode,
            tunnel_host=settings.colab_tunnel_host,
            tunnel_token=settings.colab_tunnel_token,
            dry_run=bool(settings.colab_dry_run),
        )


def env_cell(text: str, content: str = CONTENT_DIR) -> str:
    """The silent cell that writes the env file *text* (0600 before any byte)."""
    path = f"{content}/{ENV_FILE_NAME}"
    return (
        "import os\n"
        f"_fd = os.open({path!r}, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)\n"
        "os.fchmod(_fd, 0o600)\n"
        "with os.fdopen(_fd, 'w', encoding='utf-8') as _file:\n"
        f"    _file.write({text!r})\n"
        "del _fd, _file\n"
    )


def start_cell(content: str = CONTENT_DIR) -> str:
    """The cell that starts bootstrap.py detached and prints the sentinel."""
    return (
        "import os, subprocess, sys\n"
        f"if os.path.exists({content + '/' + ENV_FILE_NAME!r}):\n"
        f"    with open({content + '/bootstrap.log'!r}, 'ab') as _log:\n"
        "        _proc = subprocess.Popen(\n"
        f"            [sys.executable, {content + '/bootstrap.py'!r}],\n"
        f"            cwd={content!r},\n"
        "            stdin=subprocess.DEVNULL,\n"
        "            stdout=_log,\n"
        "            stderr=subprocess.STDOUT,\n"
        "            start_new_session=True,\n"
        "        )\n"
        f"    print({LAUNCH_SENTINEL!r}, _proc.pid)\n"
        "    del _log, _proc\n"
        "else:\n"
        f"    print({MISSING_ENV_SENTINEL!r})\n"
    )


def launched_pid(stdout: str) -> int | None:
    match = _LAUNCHED.search(stdout or "")
    return int(match.group(1)) if match else None


class ColabLauncher:
    """See the module docstring."""

    def __init__(
        self,
        control_plane: Any,
        *,
        secret: str,
        bot_url: Callable[[], str],
        settings: LaunchSettings,
        vm_dir: Path = VM_DIR,
    ) -> None:
        """``bot_url()`` is the service's public https URL (``BaseUrlResolver``):
        where the VM registers."""
        self._control_plane = control_plane
        self._secret = secret
        self._bot_url = bot_url
        self.settings = settings
        self._vm_dir = vm_dir

    def bootstrap_env(self, nbh: str) -> str:
        """The env file for session *nbh*. Secret: never log it.

        Raises:
            LaunchError: The bot's public URL is not known yet.
        """
        bot_url = (self._bot_url() or "").strip().rstrip("/")
        if not bot_url.startswith("https://"):
            raise LaunchError(
                "the bot's public https URL is unknown (set PUBLIC_BASE_URL, or wait for the first cron)"
            )
        cfg = self.settings
        values = {
            "COLAB_REG_KEY": protocol.derive_reg_key(self._secret, nbh),
            "COLAB_PROXY_TOKEN": protocol.derive_proxy_token(self._secret, nbh),
            "BOT_URL": bot_url,
            "NBH": nbh,
            "IDLE_TTL": str(cfg.idle_ttl_seconds),
            "MAX_SESSION": str(cfg.max_session_seconds),
            "TUNNEL_MODE": cfg.tunnel_mode,
            "OPENCODE_ENABLED": "1" if cfg.opencode_enabled else "0",
            "DRYRUN": "1" if cfg.dry_run else "0",
        }
        if cfg.tunnel_mode == "named":
            values["TUNNEL_TOKEN"] = cfg.tunnel_token
            values["TUNNEL_HOST"] = cfg.tunnel_host
        try:
            return register.render_bootstrap_env(values)
        except ValueError as exc:
            # Names the key only (render_bootstrap_env never quotes a value).
            raise LaunchError(f"bootstrap env: {exc}") from None

    async def launch(
        self, runtime: Any, nbh: str, *, kernel_id: str = "", session_id: str = ""
    ) -> LaunchResult:
        """Upload, write the env file, start bootstrap.py; see the module docstring.

        ``runtime`` is the session's :class:`~.control_plane.RuntimeAssignment`
        with a fresh proxy token.
        """
        env_text = self.bootstrap_env(nbh)
        await self._control_plane.upload(
            runtime, {f"{CONTENT_DIR.lstrip('/')}/{name}": self._vm_dir / name for name in VM_FILES}
        )
        written = await self._control_plane.run_cell(
            runtime,
            env_cell(env_text),
            budget_s=CELL_BUDGET_SECONDS,
            kernel_id=kernel_id,
            session_id=session_id,
            silent=True,
        )
        if not written.ok:
            # Only the status: a silent cell's traceback could quote its code.
            raise LaunchError(f"writing the bootstrap env file ended {written.status}")
        started = await self._control_plane.run_cell(
            runtime,
            start_cell(),
            budget_s=CELL_BUDGET_SECONDS,
            kernel_id=written.kernel_id,
            session_id=written.session_id,
        )
        pid = launched_pid(started.stdout) if started.ok else None
        if pid is None:
            detail = started.error or started.stdout.strip()[-200:] or "no output"
            raise LaunchError(f"bootstrap.py did not start ({started.status}): {detail}")
        return LaunchResult(started.kernel_id, started.session_id, pid)
