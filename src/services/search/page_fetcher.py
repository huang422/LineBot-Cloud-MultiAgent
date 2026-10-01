"""Fetching a web page without letting a URL reach inside the network.

The bot fetches pages whose URLs come from users and from search results,
from a Cloud Run service that can reach the metadata server (and any
private network it is ever connected to). So every fetch is checked before
and while it connects:

- the URL: http or https only, port 80 or 443, and no ``localhost``,
  ``metadata``, ``metadata.google.internal``, ``*.internal``, ``*.local``
  or ``*.localhost`` host;
- the host is resolved here (getaddrinfo, RESOLVE_TIMEOUT_SECONDS; a
  PageFetcher resolves on RESOLVE_WORKERS threads of its own, never on the
  loop's resolver, see :func:`_pooled_getaddrinfo`) and every address it
  resolves to must be public (:func:`is_public_address`); a single private
  one refuses the fetch. Numeric forms such as ``2130706433`` resolve like
  any name, so they are caught too;
- the connection goes to an address that passed, never to a second lookup,
  so there is no rebinding window. TLS still verifies the certificate
  against, and sends SNI and Host for, the hostname;
- redirects are followed by hand, at most MAX_REDIRECTS, each hop checked
  as above;
- only ``text/html``, ``application/xhtml+xml`` and ``text/plain`` bodies are
  read, and at most ``max_bytes`` of them (after decompression, which is
  bounded as it goes: a small gzip body cannot inflate past the cap), in
  memory only.

``trust_env`` is off everywhere: an ``HTTP(S)_PROXY`` in the environment
would route around every check here. The one client serves every user and
chat, so it keeps no cookies: a ``Set-Cookie`` from one fetch never goes
out with another.
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import socket
import zlib
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from http.cookiejar import CookieJar, DefaultCookiePolicy

import httpcore
import httpx

MAX_REDIRECTS = 5
RESOLVE_TIMEOUT_SECONDS = 2.0
# At most this many of a PageFetcher's lookups run at once (_pooled_getaddrinfo).
RESOLVE_WORKERS = 4
DEFAULT_MAX_BYTES = 2_000_000
ALLOWED_PORTS: tuple[int, ...] = (80, 443)
ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
USER_AGENT = "Mozilla/5.0 (compatible; LineBot-CloudAgent/3; page reader for a chat bot)"

_BLOCKED_HOSTS = frozenset({"localhost", "metadata", "metadata.google.internal"})
_BLOCKED_SUFFIXES = (".localhost", ".internal", ".local")
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_METADATA_IP = ipaddress.ip_address("169.254.169.254")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], Awaitable[list[str]]]


class FetchError(Exception):
    """The page could not be fetched."""


class BlockedURLError(FetchError):
    """The URL, or an address it resolves to, is not one the fetcher visits."""


@dataclass
class FetchedPage:
    url: str
    body: bytes
    content_type: str
    # From the Content-Type header, when it names one.
    charset: str | None = None
    # The body stopped at the byte cap.
    truncated: bool = False


def is_public_address(ip: IPAddress) -> bool:
    """True for a globally routable unicast address.

    An IPv4-mapped (``::ffff:a.b.c.d``) or 6to4 (``2002::/16``) address is
    judged by the IPv4 address inside it. ``is_global`` alone would pass
    multicast, NAT64 (``64:ff9b::/96``, reserved) and the deprecated IPv6
    site-local range (``fec0::/10``), and ``is_private`` alone would miss
    carrier-grade NAT (100.64.0.0/10), so all of them are checked.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        inner = ip.ipv4_mapped or ip.sixtofour
        if inner is not None:
            ip = inner
    if ip == _METADATA_IP or (ip.version == 4 and ip in _CGNAT):
        return False
    return ip.is_global and not (
        ip.is_multicast
        or ip.is_reserved
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or (ip.version == 6 and ip.is_site_local)
    )


def is_blocked_host(host: str) -> bool:
    """Names that point inside the machine or the cloud, before any lookup."""
    name = host.strip().rstrip(".").lower()
    return not name or name in _BLOCKED_HOSTS or name.endswith(_BLOCKED_SUFFIXES)


def check_url(url: str | httpx.URL, allowed_ports: Iterable[int] = ALLOWED_PORTS) -> httpx.URL:
    """*url* if it may be fetched (scheme, port, host name); else raises."""
    try:
        parsed = url if isinstance(url, httpx.URL) else httpx.URL(str(url).strip())
    except (httpx.InvalidURL, TypeError, ValueError) as exc:
        raise BlockedURLError("not a valid URL") from exc
    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError(f"scheme {parsed.scheme or '(none)'} is not http or https")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in tuple(allowed_ports):
        raise BlockedURLError(f"port {port} is not allowed")
    if is_blocked_host(parsed.host):
        raise BlockedURLError(f"host {parsed.host or '(none)'} is internal")
    return parsed


async def _getaddrinfo(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _pooled_getaddrinfo(pool: Executor) -> Resolver:
    """getaddrinfo on *pool*'s threads instead of the loop's resolver.

    ``asyncio.wait_for`` cannot stop a lookup: one that hangs keeps its
    thread until the system resolver gives up. The loop's resolver (libuv's
    four threads under uvloop, the default executor otherwise) is the one
    every other outbound connection resolves on, so a few slow-DNS URLs in
    one message stalled LINE and Cloud Tasks calls for every chat. On a pool
    of their own, hung lookups hold only its threads, and one queued behind
    them times out without ever running.
    """

    async def resolve(host: str, port: int) -> list[str]:
        lookup = functools.partial(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
        infos = await asyncio.get_running_loop().run_in_executor(pool, lookup)
        return [str(info[4][0]) for info in infos]

    return resolve


async def resolve_public(
    host: str,
    port: int,
    *,
    resolver: Resolver = _getaddrinfo,
    address_allowed: Callable[[IPAddress], bool] = is_public_address,
    timeout: float = RESOLVE_TIMEOUT_SECONDS,
) -> list[str]:
    """Every address *host* resolves to, IPv4 first; raises BlockedURLError
    unless all of them pass *address_allowed*."""
    if is_blocked_host(host):
        raise BlockedURLError(f"host {host} is internal")
    try:
        raw = await asyncio.wait_for(resolver(host, port), timeout)
    except (TimeoutError, OSError) as exc:
        raise FetchError(f"cannot resolve {host}") from exc
    addresses: list[IPAddress] = []
    for text in raw:
        try:
            ip = ipaddress.ip_address(str(text).split("%", 1)[0])
        except ValueError as exc:
            raise FetchError(f"{host} resolved to something that is not an address") from exc
        if not address_allowed(ip):
            raise BlockedURLError(f"{host} resolves to a non-public address")
        if ip not in addresses:
            addresses.append(ip)
    if not addresses:
        raise FetchError(f"{host} has no address")
    return [str(ip) for ip in sorted(addresses, key=lambda ip: ip.version)]


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolves and checks the host itself, then connects to that address.

    httpcore hands ``connect_tcp`` the URL's host; TLS is started afterwards
    with the same host as SNI and for certificate checks, so only the
    address changes.
    """

    def __init__(self, resolve: Callable[[str, int], Awaitable[list[str]]], allowed_ports: Iterable[int]):
        self._resolve = resolve
        self._allowed_ports = tuple(allowed_ports)
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if port not in self._allowed_ports:
            raise BlockedURLError(f"port {port} is not allowed")
        last_error: Exception | None = None
        for address in await self._resolve(host, port):
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last_error = exc
        raise httpcore.ConnectError(f"could not connect to {host}") from last_error

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise BlockedURLError("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _PinnedTransport(httpx.AsyncHTTPTransport):
    """httpx's transport over a connection pool that uses the pinned backend.

    httpx has no public hook for the network backend, so the pool built by
    the parent is swapped for one that has it (httpx and httpcore are pinned
    in requirements.lock; tests/test_page_fetcher.py checks the swap still
    takes effect).
    """

    def __init__(self, network_backend: httpcore.AsyncNetworkBackend) -> None:
        super().__init__(trust_env=False, http1=True, http2=False)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False),
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=5.0,
            http1=True,
            http2=False,
            network_backend=network_backend,
        )


def ssrf_safe_transport(
    *,
    allowed_ports: Iterable[int] = ALLOWED_PORTS,
    resolver: Resolver = _getaddrinfo,
    address_allowed: Callable[[IPAddress], bool] = is_public_address,
) -> httpx.AsyncBaseTransport:
    """An httpx transport that connects only to checked public addresses.

    Every connection resolves its host here, refuses it unless every
    address passes *address_allowed* (and the port is allowed), and
    connects to the address that passed. The client using it must still
    set ``trust_env=False`` and ``follow_redirects=False``; the Colab
    tunnel client (src/services/colab/endpoint.py) does.
    """
    ports = tuple(allowed_ports)

    async def resolve(host: str, port: int) -> list[str]:
        return await resolve_public(host, port, resolver=resolver, address_allowed=address_allowed)

    return _PinnedTransport(_PinnedNetworkBackend(resolve, ports))


class PageFetcher:
    """Fetches one page at a time under the checks in the module docstring."""

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: httpx.Timeout | None = None,
        resolver: Resolver | None = None,
        address_allowed: Callable[[IPAddress], bool] = is_public_address,
        allowed_ports: Iterable[int] = ALLOWED_PORTS,
    ) -> None:
        """``resolver``, ``address_allowed`` and ``allowed_ports`` exist for
        tests (a local server on a spare port); production keeps the defaults."""
        self.max_bytes = max_bytes
        self._allowed_ports = tuple(allowed_ports)
        # The hosts come from users and search results: see _pooled_getaddrinfo.
        self._dns_pool = ThreadPoolExecutor(RESOLVE_WORKERS, thread_name_prefix="page-dns")
        self._client = httpx.AsyncClient(
            transport=ssrf_safe_transport(
                allowed_ports=self._allowed_ports,
                resolver=resolver or _pooled_getaddrinfo(self._dns_pool),
                address_allowed=address_allowed,
            ),
            trust_env=False,
            # A jar that refuses every cookie (no domain is allowed).
            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
            follow_redirects=False,
            timeout=timeout or httpx.Timeout(5.0, connect=3.0, pool=2.0),
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8,*/*;q=0.1",
                "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.6",
                "Accept-Encoding": "gzip, deflate",
            },
        )

    async def fetch(self, url: str) -> FetchedPage:
        """The page at *url*; raises FetchError (BlockedURLError when a check refused it)."""
        target = check_url(url, self._allowed_ports)
        for hop in range(MAX_REDIRECTS + 1):
            try:
                response = await self._client.send(
                    self._client.build_request("GET", target), stream=True
                )
            except FetchError:
                raise
            except httpx.HTTPError as exc:
                raise FetchError(f"{type(exc).__name__} fetching {target.host}") from exc
            try:
                if response.is_redirect:
                    if hop == MAX_REDIRECTS:
                        raise FetchError(f"more than {MAX_REDIRECTS} redirects")
                    target = check_url(
                        target.join(response.headers["location"]), self._allowed_ports
                    )
                    continue
                if response.status_code != 200:
                    raise FetchError(f"HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type not in ALLOWED_CONTENT_TYPES:
                    raise FetchError(f"content type {content_type or '(none)'} is not a page")
                body, truncated = await self._read_capped(response)
                return FetchedPage(
                    url=str(target),
                    body=body,
                    content_type=content_type,
                    charset=response.charset_encoding,
                    truncated=truncated,
                )
            finally:
                await response.aclose()
        raise FetchError("redirect loop")  # pragma: no cover - the loop returns or raises

    async def _read_capped(self, response: httpx.Response) -> tuple[bytes, bool]:
        """At most max_bytes of the decoded body, decompressing as it goes."""
        encoding = response.headers.get("content-encoding", "").strip().lower()
        if encoding in ("", "identity"):
            decoder = None
        elif encoding in ("gzip", "x-gzip", "deflate"):
            # 32 + MAX_WBITS: accept a gzip or a zlib header.
            decoder = zlib.decompressobj(32 + zlib.MAX_WBITS)
        else:
            raise FetchError(f"content encoding {encoding} is not supported")
        body = bytearray()
        raw_read = 0
        try:
            async for chunk in response.aiter_raw():
                raw_read += len(chunk)
                if decoder is None:
                    body += chunk[: self.max_bytes - len(body)]
                else:
                    body += decoder.decompress(chunk, self.max_bytes - len(body))
                if len(body) >= self.max_bytes or raw_read >= self.max_bytes:
                    return bytes(body[: self.max_bytes]), True
        except zlib.error as exc:
            raise FetchError("the compressed body is corrupt") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__} while reading") from exc
        if decoder is not None:
            try:
                body += decoder.flush(self.max_bytes - len(body))
            except zlib.error as exc:
                raise FetchError("the compressed body is corrupt") from exc
        return bytes(body), False

    async def close(self) -> None:
        await self._client.aclose()
        self._dns_pool.shutdown(wait=False, cancel_futures=True)


def describe_failure(exc: BaseException) -> str:
    """For logs: the reason, without the page's full URL."""
    return f"{type(exc).__name__}: {exc}" if isinstance(exc, FetchError) else type(exc).__name__

