"""Suite-wide guards: no real network, and Settings from code defaults only.

Tests talk to fakes (``httpx.MockTransport``, servers on 127.0.0.1) and
never to the internet; this makes that a rule instead of a habit.

* Sockets of this process may connect only to loopback (127.0.0.0/8, ::1,
  ``localhost``) or over AF_UNIX, and ``getaddrinfo`` looks up only
  ``localhost`` and numeric addresses (no DNS). ``create_connection`` goes
  through both. Anything else raises :class:`NetworkBlockedError`, and a
  test that tried fails at teardown even if the code under test swallowed
  the error (the search chain and the task queue catch everything).
  Subprocesses are not patched; tests hand them fakes of their own.
* Environment variables named after Settings fields are dropped at start,
  so a developer's shell (``TAVILY_API_KEY``, ``NVIDIA_API_KEY``, ...)
  cannot switch a live backend on in ``Settings(_env_file=None)``.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import MutableMapping

import pytest

from src.config import Settings

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo
_INET_FAMILIES = (socket.AF_INET, socket.AF_INET6)

# What the guard refused since the current test started.
BLOCKED_ATTEMPTS: list[str] = []


class NetworkBlockedError(RuntimeError):
    """A test tried to reach the network; use a fake instead."""


def is_loopback(host: str | bytes) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = host.strip("[]").split("%", 1)[0].rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped and mapped.is_loopback)


def _is_numeric(host: str | bytes) -> bool:
    try:
        _REAL_GETADDRINFO(host, None, flags=socket.AI_NUMERICHOST)
    except (OSError, UnicodeError):
        return False
    return True


def _refuse(what: str) -> NetworkBlockedError:
    message = (
        f"tests must not use the network: {what}; use a fake "
        "(httpx.MockTransport, a server on 127.0.0.1)"
    )
    BLOCKED_ATTEMPTS.append(message)
    return NetworkBlockedError(message)


def _check_address(sock: socket.socket, address) -> None:
    if sock.family in _INET_FAMILIES and not is_loopback(address[0]):
        raise _refuse(f"connect to {address!r}")


def _connect(self: socket.socket, address) -> None:
    _check_address(self, address)
    return _REAL_CONNECT(self, address)


def _connect_ex(self: socket.socket, address) -> int:
    _check_address(self, address)
    return _REAL_CONNECT_EX(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if host in (None, "", b"") or is_loopback(host) or _is_numeric(host):
        return _REAL_GETADDRINFO(host, *args, **kwargs)
    raise _refuse(f"DNS lookup of {host!r}")


def scrub_settings_env(environ: MutableMapping[str, str]) -> list[str]:
    """Drop (and return) the keys Settings would read from *environ*."""
    dropped = sorted(key for key in environ if key.lower() in Settings.model_fields)
    for key in dropped:
        del environ[key]
    return dropped


def network_guard_installed() -> bool:
    return socket.getaddrinfo is _getaddrinfo and socket.socket.connect is _connect


def pytest_configure(config: pytest.Config) -> None:
    scrub_settings_env(os.environ)
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.getaddrinfo = _getaddrinfo


def pytest_unconfigure(config: pytest.Config) -> None:
    socket.socket.connect = _REAL_CONNECT
    socket.socket.connect_ex = _REAL_CONNECT_EX
    socket.getaddrinfo = _REAL_GETADDRINFO


@pytest.fixture(autouse=True)
def _fail_on_network_attempts():
    BLOCKED_ATTEMPTS.clear()
    yield
    attempts = list(dict.fromkeys(BLOCKED_ATTEMPTS))
    BLOCKED_ATTEMPTS.clear()
    if attempts:
        pytest.fail("; ".join(attempts), pytrace=False)
