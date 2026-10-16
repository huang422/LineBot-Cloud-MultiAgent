"""The wire contract between the bot and its Colab VM (ADR-0010).

Nothing secret is stored anywhere but ``COLAB_SHARED_SECRET``. Both sides
derive two values per session from it and the session's ``nbh`` (the UUID
the controller picks before it asks Colab for a runtime)::

    reg_key     = hex(HMAC_SHA256(secret, "reg:" + nbh))
    proxy_token = base64url(HMAC_SHA256(secret, "proxy:" + nbh)), no "=" padding

The HMAC key is ``secret`` as UTF-8 bytes; the messages are UTF-8 too. The
launcher hands the VM both values in ``/content/.bootstrap.env``; the bot
recomputes them whenever it needs them, so a restart loses nothing.

**Registration.** The VM POSTs ``{BOT_URL}/internal/colab/register`` with a
JSON body of at most :data:`MAX_REGISTER_BYTES`::

    {"event": "ready" | "failed" | "stopping", "nbh": "<uuid>",
     "tunnel_url"?: "https://<host>", "versions"?: {"ollama", "cloudflared"},
     "digests"?: {"<model>": "<digest>"}, "gpu"?: {"name", "vram_mib", "driver",
     "used_mib"?, "chat_on_gpu"?, "router_on_gpu"?}, "step"?: "<boot step>",
     "error"?: "<text>"}

(``gpu``: ``used_mib`` is nvidia-smi's memory.used once the models are
warm, ``<role>_on_gpu`` that model's ``/api/ps`` size_vram/size in MiB,
e.g. ``"10110/10110 MiB"``; shown in ``/health/detail``, never trusted.)

signed in the :data:`SIGNATURE_HEADER` header as
``t=<unix seconds>,v1=<hex HMAC_SHA256(reg_key, f"{t}." + raw body)>``,
where the HMAC key is ``reg_key``'s hex text as ASCII bytes and the raw
body is the exact bytes sent. The bot refuses a timestamp more than
:data:`MAX_CLOCK_SKEW_SECONDS` away from its clock, an ``nbh`` other than
the current session's, a signature that does not match (constant-time
compare), a signature it has already accepted, and a body over the limit.

**Tunnel.** ``tunnel_url`` is ``https://`` plus a host and nothing else (no
user info, port, path, query or fragment). The host is a quick tunnel
(``<name>.trycloudflare.com``) or exactly ``COLAB_TUNNEL_HOST`` (a named
tunnel), and every address it resolves to must be public
(:func:`src.services.search.page_fetcher.is_public_address`).

(``error`` on a ``ready`` says why the router is not serving, when it is
not; the bot logs it, clipped.)

**Readiness.** After a signed ``ready`` the bot probes the tunnel with
``Authorization: Bearer <proxy_token>`` within :data:`PROBE_TIMEOUT_SECONDS`:
``GET /api/version`` passes :func:`ollama_version_problem` (at least
:data:`OLLAMA_MIN_VERSION`, and the release the launcher picked for the
session when the state recorded one), ``GET /api/ps`` lists the chat model
wholly in VRAM (``size_vram == size``), ``GET /api/tags`` lists it, and
``POST /api/show`` for the chat model lists the think values
:data:`CHAT_THINK_VALUES`. Only then is the session READY. No digest is
compared: the models follow their tags (2026-10-16), and the probe's
digests are kept for display beside the last verified ones
(:func:`colab_models`). A router (:func:`router_models`) that is not listed
or loaded, or not wholly in VRAM, does not keep the session from READY:
the session keeps the sizes the probe saw, and :func:`off_gpu` keeps the
router out of it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from colab import VERSIONS_FILE, register
from src.providers.model_profiles import OLLAMA_CHAT, PROFILES, SYSTEMONE, THINK_LEVELS

SIGNATURE_HEADER = "X-Colab-Signature"
MAX_REGISTER_BYTES = 8 * 1024
MAX_CLOCK_SKEW_SECONDS = 300
EVENT_READY = "ready"
EVENT_FAILED = "failed"
EVENT_STOPPING = "stopping"
EVENTS: frozenset[str] = frozenset({EVENT_READY, EVENT_FAILED, EVENT_STOPPING})

# The oldest Ollama a session may run, from colab/versions.json (the one
# copy): clef-flash's config says "requires": "0.40.4", and the registry
# answers HTTP 412 to an older client's pull. The launcher installs the
# latest release at or above it (or versions.json's pinned fallback).
OLLAMA_MIN_VERSION: str = register.load_versions(VERSIONS_FILE)["ollama"]["min_version"]
# What POST /api/show reports for qwen3.8 ("false" is JSON false), at
# 0.35.0 and at 0.40.5 alike: the guard against a release that changes
# how it thinks.
CHAT_THINK_VALUES: list[bool | str] = [False if level == "false" else level for level in THINK_LEVELS]
PROBE_TIMEOUT_SECONDS = 10.0

QUICK_TUNNEL_HOST = re.compile(r"[a-z0-9-]+\.trycloudflare\.com")
_SIGNATURE_PART = re.compile(r"\s*(t|v1)=([^,\s]*)\s*")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
# ASCII digits only: str.isdigit() also takes "²" (Starlette decodes header
# bytes as latin-1), which int() then refuses.
_TIMESTAMP = re.compile(r"[0-9]{1,12}")


def colab_models() -> dict[str, str]:
    """Colab model id → its last verified Ollama digest (information only:
    the models follow their tags, nothing compares it).

    Read from the model profiles (``ModelProfile.verified_digest``), the one
    copy of the Colab models on the bot side.
    """
    return {p.model: p.verified_digest for p in PROFILES.values() if p.provider == "colab"}


def chat_models() -> list[str]:
    """The Colab models that chat (``/api/chat``), for the think check."""
    return [p.model for p in PROFILES.values() if p.provider == "colab" and p.protocol == OLLAMA_CHAT]


def router_models() -> list[str]:
    """The Colab models that route (``/v1/systemone``): the only ones a
    session may be READY without (not pulled, not loaded, off the GPU)."""
    return [p.model for p in PROFILES.values() if p.provider == "colab" and p.protocol == SYSTEMONE]


def normalize_digest(value: object) -> str:
    """A digest as bare lowercase hex (``/api/tags`` has no ``sha256:``)."""
    return str(value or "").strip().lower().removeprefix("sha256:")


def moved_tags(digests: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    """model → (verified, actual) for each Colab model *digests* (what a
    probe saw in /api/tags) shows at another digest than its last verified
    one: the tag moved since someone checked it. For display only."""
    found: dict[str, tuple[str, str]] = {}
    for model, verified in colab_models().items():
        actual = normalize_digest(digests.get(model))
        if actual and actual != verified:
            found[model] = (verified, actual)
    return found


def ollama_version_problem(found: object, launched: str = "") -> str:
    """Why *found* (``/api/version``'s ``version``) is not an Ollama this
    session may run, or "".

    It must be at least :data:`OLLAMA_MIN_VERSION`, and exactly *launched*
    (the release the launcher picked, :attr:`.state.ColabState.ollama_version`)
    when the session recorded one: a state written before 2026-10-16 has
    none, so only the minimum applies.
    """
    text = found if isinstance(found, str) else ""
    if not register.version_at_least(text, OLLAMA_MIN_VERSION):
        return f"Ollama {_clip_version(found)}, expected {OLLAMA_MIN_VERSION} or later"
    if launched and text != launched:
        return f"Ollama {_clip_version(found)}, expected {launched} (the release this session was launched with)"
    return ""


def _clip_version(value: object) -> str:
    """*value* as at most 40 characters, for a message."""
    text = str(value)
    return text if len(text) <= 40 else text[:39] + "…"


def model_size(value: object) -> int:
    """A byte count from ``/api/ps`` (``size``, ``size_vram``); 0 when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return max(0, int(value))


def on_gpu(sizes: object) -> bool:
    """Whether ``{"size", "size_vram"}`` (an ``/api/ps`` entry's) puts the
    whole model in VRAM: a model partly on the CPU runs many times slower."""
    if not isinstance(sizes, Mapping):
        return False
    size = model_size(sizes.get("size"))
    return size > 0 and model_size(sizes.get("size_vram")) >= size


def off_gpu(vram: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    """router → (size_vram, size) in bytes for each router *vram* (the
    probe's ``/api/ps`` sizes) does not show wholly in VRAM, or has no
    sizes for ((0, 0)): the session routes without it."""
    found: dict[str, tuple[int, int]] = {}
    for model in router_models():
        sizes = vram.get(model)
        if not on_gpu(sizes):
            sizes = sizes if isinstance(sizes, Mapping) else {}
            found[model] = (model_size(sizes.get("size_vram")), model_size(sizes.get("size")))
    return found


def mib(size: int) -> int:
    """Bytes as whole MiB, for messages."""
    return round(size / (1 << 20))


def _hmac(key: bytes, message: bytes) -> bytes:
    return hmac.new(key, message, hashlib.sha256).digest()


def derive_reg_key(secret: str, nbh: str) -> str:
    """The session's registration signing key, as 64 hex characters."""
    return _hmac(secret.encode("utf-8"), f"reg:{nbh}".encode()).hex()


def derive_proxy_token(secret: str, nbh: str) -> str:
    """The bearer the session's auth proxy accepts (unpadded base64url)."""
    digest = _hmac(secret.encode("utf-8"), f"proxy:{nbh}".encode())
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def signature_for(reg_key: str, timestamp: int, body: bytes) -> str:
    """The ``v1`` hex digest for *body* sent at *timestamp*."""
    return _hmac(reg_key.encode("ascii"), f"{timestamp}.".encode() + body).hex()


def sign(reg_key: str, timestamp: int, body: bytes) -> str:
    """The full :data:`SIGNATURE_HEADER` value (what the VM sends)."""
    return f"t={timestamp},v1={signature_for(reg_key, timestamp, body)}"


@dataclass(frozen=True)
class Signature:
    timestamp: int
    v1: str


def parse_signature(header: str | None) -> Signature | None:
    """``t=<int>,v1=<64 hex>`` → :class:`Signature`; None when malformed."""
    if not header or len(header) > 200:
        return None
    parts: dict[str, str] = {}
    for chunk in header.split(","):
        match = _SIGNATURE_PART.fullmatch(chunk)
        if match is None or match.group(1) in parts:
            return None
        parts[match.group(1)] = match.group(2)
    timestamp, v1 = parts.get("t", ""), parts.get("v1", "").lower()
    if not _TIMESTAMP.fullmatch(timestamp) or not _HEX_SHA256.fullmatch(v1):
        return None
    return Signature(int(timestamp), v1)


def signature_matches(reg_key: str, signature: Signature, body: bytes) -> bool:
    expected = signature_for(reg_key, signature.timestamp, body)
    return hmac.compare_digest(expected.encode("ascii"), signature.v1.encode("ascii"))


class TunnelURLError(ValueError):
    """``tunnel_url`` is not a URL the bot will send traffic to."""


def check_tunnel_url(url: object, named_host: str = "") -> str:
    """The host of *url* when it follows the tunnel rule; else raises.

    The rule: ``https://`` and a host, nothing else; the host is a quick
    tunnel or exactly *named_host*. Resolution is checked separately
    (:meth:`~src.services.colab.registration.ColabRegistrar`), since it
    needs the network.

    Raises:
        TunnelURLError: The reason, without echoing the URL.
    """
    if not isinstance(url, str) or not url or len(url) > 300 or url != url.strip():
        raise TunnelURLError("tunnel_url must be a non-empty string")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise TunnelURLError("tunnel_url is not a valid URL") from exc
    if parts.scheme != "https":
        raise TunnelURLError("tunnel_url must use https")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise TunnelURLError("tunnel_url must not carry user info")
    if port is not None or parts.netloc.endswith(":"):
        raise TunnelURLError("tunnel_url must not name a port")
    if parts.path not in ("", "/") or parts.query or parts.fragment or "?" in url or "#" in url:
        raise TunnelURLError("tunnel_url must not have a path, query or fragment")
    host = (parts.hostname or "").lower()
    if parts.netloc.lower() != host:
        raise TunnelURLError("tunnel_url host is not in canonical form")
    named = named_host.strip().rstrip(".").lower()
    if QUICK_TUNNEL_HOST.fullmatch(host) or (named and host == named):
        return host
    raise TunnelURLError("tunnel_url host is neither a quick tunnel nor COLAB_TUNNEL_HOST")


def tunnel_base_url(host: str) -> str:
    """The base URL requests go to for a checked *host*."""
    return f"https://{host}"
