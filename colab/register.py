"""What the Colab VM and the bot agree on (ADR-0010).

Everything both sides must compute the same way lives here, in the
standard library only: the VM runs it before anything is installed, and the
bot can import it (``from colab import register``) or check its own copy
against it (tests/test_colab_vm_scripts.py).

Keys
    Both come from ``COLAB_SHARED_SECRET`` and the session's ``nbh`` (the
    UUID the bot assigned the runtime with), so nothing secret is stored:
    the bot recomputes them after a restart. The secret and nbh are UTF-8.

    - ``reg_key = hex(HMAC_SHA256(secret, "reg:" + nbh))`` signs what the VM
      sends to the bot. It is used as an HMAC key in that hex (ASCII) form.
    - ``proxy_token = base64url(HMAC_SHA256(secret, "proxy:" + nbh))``
      without padding is the bearer token of the VM's auth proxy.

Registration
    ``POST {BOT_URL}/internal/colab/register`` with a JSON body of at most
    8 KB, ``{event: ready|failed|stopping, nbh, tunnel_url?, versions?,
    digests?, gpu?, step?, error?}``, and the header ``X-Colab-Signature:
    t=<unix ts>,v1=<hex HMAC_SHA256(reg_key, f"{ts}.{raw body}")>``. The bot
    rejects a timestamp more than 300 s off, an nbh other than the current
    session's, a bad signature and an oversized body.

Bootstrap env file
    ``/content/.bootstrap.env`` (mode 0600), written by the launcher through a
    silent kernel execute and deleted by bootstrap.py as soon as it is read:
    one ``KEY=VALUE`` per line, no quoting, keys ``BOOTSTRAP_ENV_KEYS``.
    ``render_bootstrap_env`` writes it and ``parse_bootstrap_env`` reads it.

Readiness probe
    Once the VM says ``ready``, the bot checks through the tunnel (with the
    bearer token): ``/api/version`` is the Ollama this session installed,
    ``/api/ps`` lists the chat model wholly in VRAM (``size_vram == size``),
    ``/api/tags`` lists it, and ``/api/show`` of qwen3.8 reports the
    expected think values.
    ``probe_problems`` is that check; the VM runs it on itself before it
    registers, and the DRYRUN stub upstream is built to pass it.

The Ollama release
    The launcher picks it at each launch (the latest GitHub release when it
    is at least ``ollama.min_version``, else the pinned fallback in
    versions.json) and uploads a versions.json whose ``ollama`` entry is
    that choice, so the VM installs, sha256-checks and expects exactly it.

Model tags and VRAM
    Ollama pulls tags, and the models follow them (since 2026-10-16): each
    boot pulls whatever the tag points at, and ``verified_digest`` only
    records the last digest someone checked by hand (colab/README.md); a
    different one is logged, never refused. The chat model must be pulled,
    loaded and wholly in VRAM (``required``). The router need not be: one
    that did not pull or load, or that Ollama could not fit on the GPU
    beside qwen3.8, still boots; the ``ready`` event reports what it has
    (and in ``error`` why the router is missing), and the bot routes that
    session through its API tiers instead.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REGISTER_PATH = "/internal/colab/register"
SIGNATURE_HEADER = "X-Colab-Signature"
MAX_BODY_BYTES = 8 * 1024
MAX_CLOCK_SKEW_SECONDS = 300
MAX_ERROR_CHARS = 500
EVENTS: frozenset[str] = frozenset({"ready", "failed", "stopping"})
QUICK_TUNNEL_HOST = re.compile(r"^[a-z0-9-]+\.trycloudflare\.com$")

# The keys the launcher writes to /content/.bootstrap.env. TUNNEL_TOKEN is
# set for named tunnels only; TUNNEL_HOST (optional, named tunnels) is the
# public host name the VM registers, since a named tunnel cannot tell it.
BOOTSTRAP_ENV_KEYS: tuple[str, ...] = (
    "COLAB_REG_KEY",
    "COLAB_PROXY_TOKEN",
    "BOT_URL",
    "NBH",
    "IDLE_TTL",
    "MAX_SESSION",
    "TUNNEL_MODE",
    "TUNNEL_TOKEN",
    "TUNNEL_HOST",
    "OPENCODE_ENABLED",
    "DRYRUN",
)
SECRET_ENV_KEYS: frozenset[str] = frozenset({"COLAB_REG_KEY", "COLAB_PROXY_TOKEN", "TUNNEL_TOKEN"})

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"v?([0-9]{1,6})\.([0-9]{1,6})\.([0-9]{1,6})")
_SIGNATURE = re.compile(r"^t=([0-9]{1,12}),v1=([0-9a-f]{64})$")

# (method, url, body, headers, timeout) -> (status, body). HTTP error
# statuses are returned, not raised; network failures raise OSError.
Sender = Callable[..., tuple[int, bytes]]


# --- keys and signatures ---------------------------------------------------


def derive_reg_key(shared_secret: str, nbh: str) -> str:
    """The session's registration signing key, as 64 hex characters."""
    return hmac.new(shared_secret.encode(), f"reg:{nbh}".encode(), hashlib.sha256).hexdigest()


def derive_proxy_token(shared_secret: str, nbh: str) -> str:
    """The session's auth-proxy bearer token (unpadded base64url, 43 characters)."""
    digest = hmac.new(shared_secret.encode(), f"proxy:{nbh}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def sign(reg_key: str, body: bytes, ts: int) -> str:
    """hex(HMAC_SHA256(reg_key, f"{ts}." + body)): the ``v1`` value."""
    return hmac.new(reg_key.encode(), f"{int(ts)}.".encode() + body, hashlib.sha256).hexdigest()


def signature_header(reg_key: str, body: bytes, ts: int) -> str:
    """The ``X-Colab-Signature`` value for *body* sent at *ts*."""
    return f"t={int(ts)},v1={sign(reg_key, body, ts)}"


def parse_signature_header(value: str | None) -> tuple[int, str] | None:
    """``(ts, v1)`` from a well-formed header, else None."""
    match = _SIGNATURE.match((value or "").strip())
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def verify(
    reg_key: str,
    body: bytes,
    header: str | None,
    *,
    now: float,
    max_skew: int = MAX_CLOCK_SKEW_SECONDS,
) -> bool:
    """True when *header* signs *body* with *reg_key* within *max_skew* of *now*."""
    parsed = parse_signature_header(header)
    if parsed is None:
        return False
    ts, received = parsed
    if abs(now - ts) > max_skew:
        return False
    return hmac.compare_digest(sign(reg_key, body, ts), received)


# --- the event body ----------------------------------------------------------


def build_event(
    event: str,
    nbh: str,
    *,
    tunnel_url: str | None = None,
    versions: Mapping[str, str] | None = None,
    digests: Mapping[str, str] | None = None,
    gpu: Mapping[str, Any] | None = None,
    step: str | None = None,
    error: str | None = None,
) -> bytes:
    """The raw JSON body of one registration event; fields left None are omitted.

    ``error`` is cut to MAX_ERROR_CHARS. Raises ValueError for an unknown
    event, an empty nbh or a body over MAX_BODY_BYTES.
    """
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}")
    if not nbh:
        raise ValueError("nbh is empty")
    payload: dict[str, Any] = {"event": event, "nbh": nbh}
    optional: dict[str, Any] = {
        "tunnel_url": tunnel_url,
        "versions": dict(versions) if versions is not None else None,
        "digests": dict(digests) if digests is not None else None,
        "gpu": dict(gpu) if gpu is not None else None,
        "step": step,
        "error": error[:MAX_ERROR_CHARS] if error is not None else None,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    if len(body) > MAX_BODY_BYTES:
        raise ValueError(f"event body is {len(body)} bytes, over {MAX_BODY_BYTES}")
    return body


def tunnel_host_ok(host: str, named_host: str = "") -> bool:
    """A quick-tunnel host, or exactly the configured named-tunnel host.

    Syntax only: the bot also resolves the host and refuses non-global
    addresses before it sends anything there.
    """
    name = (host or "").strip().lower()
    if named_host:
        return name == named_host.strip().lower()
    return bool(QUICK_TUNNEL_HOST.match(name))


# --- sending -----------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect comes back as its own status instead of being followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# No proxies from the environment, no redirects.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def http_request(
    method: str,
    url: str,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, bytes]:
    """``(status, body)``. An HTTP error status is returned; a network or
    protocol failure (a garbled status line, a cut-off body) raises OSError,
    so callers handle one exception type."""
    request = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
    try:
        try:
            with _OPENER.open(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, exc.read()
    except http.client.HTTPException as exc:
        raise ConnectionError(f"HTTP protocol error: {type(exc).__name__}") from exc


def post_event(
    bot_url: str,
    reg_key: str,
    body: bytes,
    *,
    send: Sender = http_request,
    now: Callable[[], float] = time.time,
    timeout: float = 15.0,
) -> int:
    """Signs *body* and POSTs it to the bot; returns the HTTP status (network errors raise)."""
    headers = {
        "Content-Type": "application/json",
        SIGNATURE_HEADER: signature_header(reg_key, body, int(now())),
    }
    status, _ = send("POST", bot_url.rstrip("/") + REGISTER_PATH, body, headers, timeout)
    return status


# --- bootstrap env file ------------------------------------------------------


def render_bootstrap_env(values: Mapping[str, str]) -> str:
    """The env file text for *values*; unknown keys and line breaks raise ValueError."""
    lines = []
    for key, value in values.items():
        if key not in BOOTSTRAP_ENV_KEYS:
            raise ValueError(f"unknown bootstrap env key {key}")
        text = str(value)
        if "\n" in text or "\r" in text:
            raise ValueError(f"{key} contains a line break")
        lines.append(f"{key}={text}\n")
    return "".join(lines)


def parse_bootstrap_env(text: str) -> dict[str, str]:
    """``KEY=VALUE`` lines to a dict. Blank lines and ``#`` comments are skipped,
    a value wrapped in matching quotes loses them, and unknown keys are ignored.
    Raises ValueError (naming the line number, never its content) for a line
    without ``=``."""
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"line {number} of the bootstrap env file has no '='")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key in BOOTSTRAP_ENV_KEYS:
            values[key] = value
    return values


# --- pins and the readiness probe -------------------------------------------


def load_versions(path: Path) -> dict[str, Any]:
    """versions.json, checked for the fields the VM and the probe use."""
    versions = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_versions(versions)
    return versions


def validate_versions(versions: Any) -> None:
    """Raises ValueError naming the first malformed entry."""
    if not isinstance(versions, dict):
        raise ValueError("versions.json is not an object")
    for tool in ("ollama", "cloudflared"):
        entry = versions.get(tool)
        if not isinstance(entry, dict):
            raise ValueError(f"versions.json: {tool} is missing")
        if not str(entry.get("url", "")).startswith("https://"):
            raise ValueError(f"versions.json: {tool}.url is not https")
        if not _HEX64.match(str(entry.get("sha256", ""))):
            raise ValueError(f"versions.json: {tool}.sha256 is not 64 hex characters")
        if not entry.get("version"):
            raise ValueError(f"versions.json: {tool}.version is missing")
        # The VM stops a download at this many bytes and wants all of them.
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"versions.json: {tool}.size is not a positive byte count")
    ollama = versions["ollama"]
    if parse_version(ollama.get("min_version")) is None:
        raise ValueError("versions.json: ollama.min_version is not a version like 0.40.4")
    if not version_at_least(ollama["version"], ollama["min_version"]):
        raise ValueError("versions.json: ollama.version is older than ollama.min_version")
    models = versions.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("versions.json: models is missing")
    roles = []
    for name, pin in models.items():
        if not isinstance(pin, dict) or not _HEX64.match(str(pin.get("verified_digest", ""))):
            raise ValueError(f"versions.json: models[{name}].verified_digest is not 64 hex characters")
        if pin.get("role") not in ("chat", "router"):
            raise ValueError(f"versions.json: models[{name}].role is not chat or router")
        thinking = pin.get("thinking")
        if thinking is not None and not isinstance(thinking, list):
            raise ValueError(f"versions.json: models[{name}].thinking is not a list")
        roles.append(pin["role"])
    if sorted(roles) != ["chat", "router"]:
        raise ValueError("versions.json: models needs exactly one chat and one router model")


def parse_version(value: Any) -> tuple[int, int, int] | None:
    """``"0.40.5"`` (or ``"v0.40.5"``) as ``(0, 40, 5)``; None for anything
    else, a pre-release suffix included (``"0.41.0-rc1"``): only plain
    releases are compared."""
    match = _VERSION.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def version_at_least(version: Any, minimum: Any) -> bool:
    """Whether *version* parses and is *minimum* or newer (False when either
    does not parse)."""
    have, want = parse_version(version), parse_version(minimum)
    return have is not None and want is not None and have >= want


def normalize_digest(value: Any) -> str:
    """A digest as bare lowercase hex (Ollama reports it without ``sha256:``)."""
    text = str(value or "").strip().lower()
    return text.removeprefix("sha256:")


def required(pin: Mapping[str, Any]) -> bool:
    """Whether a model not pulled, not loaded or not wholly in VRAM stops
    the boot: every model but the router (see "Model tags and VRAM" above).
    Its digest never does."""
    return pin["role"] != "router"


def model_size(value: Any) -> int:
    """A byte count from ``/api/ps`` (``size``, ``size_vram``); 0 when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return max(0, int(value))


def gpu_share(entry: Mapping[str, Any]) -> tuple[int, int]:
    """``(size_vram, size)`` of an ``/api/ps`` entry, in bytes."""
    return model_size(entry.get("size_vram")), model_size(entry.get("size"))


def on_gpu(entry: Mapping[str, Any]) -> bool:
    """Whether an ``/api/ps`` entry has the whole model in VRAM."""
    in_vram, size = gpu_share(entry)
    return size > 0 and in_vram >= size


def mib(size: int) -> int:
    """Bytes as whole MiB."""
    return round(size / (1 << 20))


def _models(listing: Any) -> list[dict[str, Any]]:
    if not isinstance(listing, dict) or not isinstance(listing.get("models"), list):
        return []
    return [item for item in listing["models"] if isinstance(item, dict)]


def probe_problems(
    *,
    version: Any,
    ps: Any,
    tags: Any,
    shows: Mapping[str, Any],
    versions: Mapping[str, Any],
) -> list[str]:
    """What keeps an Ollama from passing the readiness probe; empty means it passes.

    *version*, *ps* and *tags* are the parsed ``/api/version``, ``/api/ps``
    and ``/api/tags`` bodies, *shows* maps a model to its ``/api/show`` body.
    The Ollama must be exactly the one *versions* names (this session's
    install). Every model but the router must be loaded, pulled and wholly
    in VRAM (``required``); no digest is compared (the models follow their
    tags).
    """
    problems: list[str] = []
    want_version = versions["ollama"]["version"]
    got_version = version.get("version") if isinstance(version, dict) else None
    if got_version != want_version:
        problems.append(f"Ollama version is {got_version!r}, not {want_version}")
    loaded = {m[key]: m for m in _models(ps) for key in ("name", "model") if isinstance(m.get(key), str)}
    listed = {m[key] for m in _models(tags) for key in ("name", "model") if m.get(key)}
    for model, pin in versions["models"].items():
        running = loaded.get(model)
        if required(pin):
            if running is None:
                problems.append(f"{model} is not loaded (/api/ps)")
            elif not on_gpu(running):
                in_vram, size = gpu_share(running)
                problems.append(
                    f"{model} is not fully on the GPU (/api/ps: {mib(in_vram)} of {mib(size)} MiB in VRAM)"
                )
            if model not in listed:
                problems.append(f"{model} is not pulled (/api/tags)")
        want_thinking = pin.get("thinking")
        if want_thinking is not None:
            show = shows.get(model)
            thinking = show.get("thinking") if isinstance(show, dict) else None
            values = thinking.get("values") if isinstance(thinking, dict) else None
            # json.dumps tells false from 0, which == does not.
            if json.dumps(values) != json.dumps(want_thinking):
                problems.append(f"{model} think values are {values!r}, not {want_thinking!r}")
    return problems
