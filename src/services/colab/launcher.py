"""Starts the VM side on a freshly assigned Colab runtime (ADR-0010, colab/README.md).

First it picks the Ollama release the session installs
(:class:`OllamaReleases`): the latest GitHub release when it is at least
versions.json's ``ollama.min_version``, else versions.json's pinned
fallback. The bot does this rather than the VM so the choice is in Cloud
Logging and outlives the VM; the VM still checks the tarball's size and
sha256, so a tampered or cut-off download fails the boot as before.

Then three steps through the control plane (:mod:`.control_plane`), all in
one kernel whose ids the controller keeps for its keep-alive cells:

1. Upload ``colab.VM_FILES`` (bootstrap.py, auth_proxy.py, register.py,
   versions.json, proxy-requirements.lock) flat into ``/content`` through
   the Jupyter Contents API; the versions.json uploaded is the repository's
   with its ``ollama`` entry replaced by the release picked.
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

import asyncio
import json
import re
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from colab import VM_FILES, register
from src.services.colab import protocol
from src.utils.logger import logger

# The VM package in this repository (and in the image).
VM_DIR = Path(__file__).resolve().parents[3] / "colab"
CONTENT_DIR = "/content"
ENV_FILE_NAME = ".bootstrap.env"
LAUNCH_SENTINEL = "BOOTSTRAP_LAUNCHED"
MISSING_ENV_SENTINEL = "BOOTSTRAP_ENV_MISSING"
CELL_BUDGET_SECONDS = 30.0
_LAUNCHED = re.compile(rf"^{LAUNCH_SENTINEL} (\d+)$", re.MULTILINE)

# The Ollama release lookup (GitHub's REST API, unauthenticated: 60
# requests an hour per address; one per session is far below that).
OLLAMA_LATEST_RELEASE_URL = "https://api.github.com/repos/ollama/ollama/releases/latest"
OLLAMA_ASSET = "ollama-linux-amd64.tar.zst"
OLLAMA_DOWNLOAD_PREFIX = "https://github.com/ollama/ollama/releases/download/"
RELEASE_TIMEOUT_SECONDS = 10.0
# Picks kept, one per session (nbh): a session's launch retries reuse its
# pick, and sessions are a few a day.
RELEASE_PICKS_KEPT = 8
_ASSET_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


class LaunchError(Exception):
    """The VM side did not start; the message never carries a secret."""


@dataclass(frozen=True)
class LaunchResult:
    kernel_id: str
    session_id: str
    pid: int
    # The Ollama release the uploaded versions.json names (the session's).
    ollama_version: str = ""


class _ReleaseUnusable(Exception):
    """The latest release cannot be used; the message says why."""


class OllamaReleases:
    """Which Ollama release the next session installs.

    The latest GitHub release (``/releases/latest``: never a draft or a
    pre-release) when it is at least the fallback's ``min_version`` and its
    ``ollama-linux-amd64.tar.zst`` asset has a GitHub download URL, a
    ``sha256:`` digest and a size; else the fallback (versions.json's pinned
    ``ollama`` entry). Any failure (an HTTP error, no answer within
    RELEASE_TIMEOUT_SECONDS, a rate limit, a missing digest, an odd shape)
    picks the fallback with a WARNING saying why; a usable latest release is
    logged at INFO. ``transport`` is for tests.

    The pick is kept per session (nbh), never across sessions: every
    launch retry of one session uploads the same entry, so a retry cannot
    name another release than the bootstrap already running from the
    first upload (the probe would then refuse a healthy session), and each
    new session asks GitHub afresh, so an asset GitHub replaced after
    publishing (it rebuilt v0.40.5's a day later) costs at most the one
    session whose download no longer matches. Only this process keeps it:
    with ``--max-instances=1`` a retry comes back here unless the instance
    was replaced in between.
    """

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport
        self._picks: OrderedDict[str, dict[str, Any]] = OrderedDict()

    async def pick(self, fallback: Mapping[str, Any], nbh: str) -> dict[str, Any]:
        """versions.json's ``ollama`` entry for session *nbh*: ``version``,
        ``url``, ``sha256``, ``size`` and the fallback's ``min_version``."""
        if nbh in self._picks:
            return dict(self._picks[nbh])
        minimum = str(fallback["min_version"])
        try:
            latest = await self._latest()
            if not register.version_at_least(latest["version"], minimum):
                raise _ReleaseUnusable(f"the latest release {latest['version']} is older than {minimum}")
        except Exception as exc:  # whatever went wrong, the fallback boots
            why = str(exc) if isinstance(exc, _ReleaseUnusable) else f"{type(exc).__name__}: {exc}"
            entry = dict(fallback)
            logger.warning(
                f"Colab launch: Ollama {entry['version']}, the pinned fallback (the latest release is not used: {why})"
            )
        else:
            entry = {**latest, "min_version": minimum}
            logger.info(f"Colab launch: Ollama {entry['version']}, the latest release")
        self._picks[nbh] = entry
        while len(self._picks) > RELEASE_PICKS_KEPT:
            self._picks.popitem(last=False)
        return dict(entry)

    async def _latest(self) -> dict[str, Any]:
        """The latest release's entry; raises _ReleaseUnusable."""
        try:
            async with asyncio.timeout(RELEASE_TIMEOUT_SECONDS):
                async with httpx.AsyncClient(
                    transport=self._transport,
                    timeout=RELEASE_TIMEOUT_SECONDS,
                    follow_redirects=False,
                    headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
                ) as client:
                    response = await client.get(OLLAMA_LATEST_RELEASE_URL)
        except TimeoutError:
            raise _ReleaseUnusable(f"GitHub did not answer within {RELEASE_TIMEOUT_SECONDS:g}s") from None
        except httpx.HTTPError as exc:
            raise _ReleaseUnusable(f"GitHub unreachable ({type(exc).__name__})") from None
        if response.status_code != 200:
            limited = response.status_code == 429 or response.headers.get("x-ratelimit-remaining") == "0"
            raise _ReleaseUnusable(f"GitHub answered HTTP {response.status_code}{' (rate limited)' if limited else ''}")
        try:
            release = response.json()
        except ValueError:
            raise _ReleaseUnusable("GitHub answered something that is not JSON") from None
        return _release_entry(release)


def _release_entry(release: Any) -> dict[str, Any]:
    """``{"version", "url", "sha256", "size"}`` from a GitHub release object;
    raises _ReleaseUnusable naming what is wrong."""
    if not isinstance(release, dict):
        raise _ReleaseUnusable("the release is not an object")
    if release.get("draft") is not False or release.get("prerelease") is not False:
        raise _ReleaseUnusable("the release is a draft or a pre-release")
    tag = release.get("tag_name")
    version = tag.removeprefix("v") if isinstance(tag, str) else ""
    if register.parse_version(version) is None:
        raise _ReleaseUnusable(f"the tag {str(tag)[:40]!r} is not a release version")
    assets = release.get("assets")
    found = [
        item for item in (assets if isinstance(assets, list) else [])
        if isinstance(item, dict) and item.get("name") == OLLAMA_ASSET
    ]
    if not found:
        raise _ReleaseUnusable(f"{version} has no {OLLAMA_ASSET}")
    asset = found[0]
    url = asset.get("browser_download_url")
    if not isinstance(url, str) or not url.startswith(OLLAMA_DOWNLOAD_PREFIX) or any(c.isspace() for c in url):
        raise _ReleaseUnusable(f"{version}'s {OLLAMA_ASSET} is not on {OLLAMA_DOWNLOAD_PREFIX}")
    # GitHub's own sha256 of the asset ("sha256:<hex>"): the VM checks the
    # download against it, so a tampered or partial file fails the boot.
    digest = asset.get("digest")
    sha256 = _ASSET_DIGEST.fullmatch(digest) if isinstance(digest, str) else None
    if sha256 is None:
        raise _ReleaseUnusable(f"{version}'s {OLLAMA_ASSET} has no sha256 digest")
    size = asset.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise _ReleaseUnusable(f"{version}'s {OLLAMA_ASSET} has no size")
    return {"version": version, "url": url, "sha256": sha256.group(1), "size": size}


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
        releases: OllamaReleases | None = None,
    ) -> None:
        """``bot_url()`` is the service's public https URL (``BaseUrlResolver``):
        where the VM registers. ``releases`` picks the Ollama release (tests
        pass one over a fake transport)."""
        self._control_plane = control_plane
        self._secret = secret
        self._bot_url = bot_url
        self.settings = settings
        self._vm_dir = vm_dir
        self._releases = releases or OllamaReleases()

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
        try:
            versions = register.load_versions(self._vm_dir / "versions.json")
            versions["ollama"] = await self._releases.pick(versions["ollama"], nbh)
            # The entry the VM will read passes the VM's own check first.
            register.validate_versions(versions)
        except (OSError, ValueError) as exc:
            raise LaunchError(f"versions.json: {exc}") from None
        with tempfile.TemporaryDirectory(prefix="colab-launch-") as scratch:
            picked = Path(scratch) / "versions.json"
            picked.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
            files = {name: picked if name == "versions.json" else self._vm_dir / name for name in VM_FILES}
            await self._control_plane.upload(
                runtime, {f"{CONTENT_DIR.lstrip('/')}/{name}": path for name, path in files.items()}
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
        return LaunchResult(started.kernel_id, started.session_id, pid, versions["ollama"]["version"])
