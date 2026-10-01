"""SSRF-safe page fetching (src/services/search/page_fetcher.py, ADR-0009).

Numeric addresses go through the real getaddrinfo (it resolves them
locally, no DNS); names are resolved by a fake resolver. The transport
tests run a local HTTP server and let the fetcher treat 127.0.0.1 (and the
server's port) as allowed, so the pinned connection path is the real one.
"""

from __future__ import annotations

import asyncio
import gzip
import ipaddress
import socket
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpcore

from src.services.search.page_fetcher import (
    ALLOWED_CONTENT_TYPES,
    MAX_REDIRECTS,
    RESOLVE_WORKERS,
    BlockedURLError,
    FetchError,
    PageFetcher,
    _PinnedNetworkBackend,
    check_url,
    is_blocked_host,
    is_public_address,
    resolve_public,
)

_LOCALHOST = ipaddress.ip_address("127.0.0.1")


def _ip(text: str):
    return ipaddress.ip_address(text)


class PublicAddressTests(unittest.TestCase):
    def test_non_public_addresses_are_refused(self) -> None:
        for address in (
            "169.254.169.254",  # the metadata server
            "10.138.0.10",  # an address on the default VPC network
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "100.64.1.1",  # carrier-grade NAT: not "private", not global
            "127.0.0.1",
            "0.0.0.0",
            "224.0.0.1",  # multicast is "global"
            "239.1.1.1",
            "255.255.255.255",
            "::1",
            "::",
            "fe80::1",
            "fec0::1",  # deprecated site-local: "global" to Python
            "fec0::a9fe:a9fe",
            "feff::1",
            "fd20:ce::254",  # the IPv6 metadata server
            "ff02::1",
            "::ffff:169.254.169.254",  # IPv4-mapped
            "::ffff:10.0.0.1",
            "2002:a9fe:a9fe::1",  # 6to4 of 169.254.169.254
            "2002:0a00:0001::1",  # 6to4 of 10.0.0.1
            "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254: global but reserved
        ):
            with self.subTest(address=address):
                self.assertFalse(is_public_address(_ip(address)))

    def test_public_addresses_pass(self) -> None:
        for address in ("8.8.8.8", "1.1.1.1", "140.112.8.116", "2001:4860:4860::8888", "::ffff:8.8.8.8"):
            with self.subTest(address=address):
                self.assertTrue(is_public_address(_ip(address)))


class UrlCheckTests(unittest.TestCase):
    def test_internal_host_names_are_refused_before_any_lookup(self) -> None:
        for host in (
            "localhost",
            "LOCALHOST.",
            "metadata",
            "metadata.google.internal",
            "anything.internal",
            "printer.local",
            "app.localhost",
        ):
            with self.subTest(host=host):
                self.assertTrue(is_blocked_host(host))
                with self.assertRaises(BlockedURLError):
                    check_url(f"http://{host}/")
        self.assertFalse(is_blocked_host("example.com"))

    def test_only_http_and_https_on_80_and_443(self) -> None:
        for url in ("ftp://example.com/", "file:///etc/passwd", "gopher://example.com/", "http://example.com:22/", "https://example.com:8443/", "http://example.com:8080/"):
            with self.subTest(url=url):
                with self.assertRaises(BlockedURLError):
                    check_url(url)
        self.assertEqual(check_url("https://example.com/a?b=1").host, "example.com")
        self.assertEqual(check_url("http://example.com:80/").port, None)


class ResolveTests(unittest.IsolatedAsyncioTestCase):
    async def test_numeric_forms_resolve_and_are_refused(self) -> None:
        # getaddrinfo reads these the way a browser or curl would.
        for host in (
            "169.254.169.254",
            "10.138.0.10",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "2130706433",  # decimal 127.0.0.1
            "0x7f000001",
            "127.1",
            "::ffff:169.254.169.254",
            "64:ff9b::a9fe:a9fe",
        ):
            with self.subTest(host=host):
                with self.assertRaises(BlockedURLError):
                    await resolve_public(host, 80)

    async def test_every_address_must_be_public(self) -> None:
        async def rebinding(host, port):
            return ["93.184.215.14", "10.0.0.7"]

        with self.assertRaises(BlockedURLError):
            await resolve_public("mixed.example", 443, resolver=rebinding)

    async def test_ipv4_first_and_zone_ids_dropped(self) -> None:
        async def resolver(host, port):
            return ["2001:4860:4860::8888%eth0", "8.8.8.8", "8.8.8.8"]

        self.assertEqual(
            await resolve_public("dns.example", 443, resolver=resolver),
            ["8.8.8.8", "2001:4860:4860::8888"],
        )

    async def test_lookup_failures_and_timeouts_are_fetch_errors(self) -> None:
        async def failing(host, port):
            raise OSError("nodename nor servname provided")

        async def slow(host, port):
            await asyncio.sleep(5)
            return ["8.8.8.8"]

        async def empty(host, port):
            return []

        for resolver in (failing, empty):
            with self.subTest(resolver=resolver.__name__):
                with self.assertRaises(FetchError) as caught:
                    await resolve_public("nowhere.example", 80, resolver=resolver)
                self.assertNotIsInstance(caught.exception, BlockedURLError)
        with self.assertRaises(FetchError):
            await resolve_public("slow.example", 80, resolver=slow, timeout=0.05)

    async def test_metadata_name_never_reaches_the_resolver(self) -> None:
        asked: list[str] = []

        async def resolver(host, port):
            asked.append(host)
            return ["8.8.8.8"]

        with self.assertRaises(BlockedURLError):
            await resolve_public("metadata.google.internal", 80, resolver=resolver)
        self.assertEqual(asked, [])


class OwnResolverThreadsTests(unittest.IsolatedAsyncioTestCase):
    async def test_hung_lookups_never_hold_the_loops_resolver(self) -> None:
        # Review: wait_for gives up on a lookup but cannot stop it. Under
        # uvloop the abandoned ones held libuv's four resolver threads, so
        # one message of slow-DNS URLs stalled every LINE and Cloud Tasks
        # connection. A one-thread default executor stands in for them here.
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        guarded = socket.getaddrinfo  # tests/conftest.py: numeric and localhost only
        release = threading.Event()
        lock = threading.Lock()
        hung = {"now": 0, "most": 0}

        def getaddrinfo(host, *args, **kwargs):
            if not str(host).endswith(".hung.test"):
                return guarded(host, *args, **kwargs)
            with lock:
                hung["now"] += 1
                hung["most"] = max(hung["most"], hung["now"])
            release.wait(5)
            with lock:
                hung["now"] -= 1
            raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

        fetcher = PageFetcher()
        with patch.object(socket, "getaddrinfo", getaddrinfo):
            fetches = [
                asyncio.create_task(fetcher.fetch(f"http://u{i}.hung.test/"))
                for i in range(RESOLVE_WORKERS + 2)
            ]
            try:
                for _ in range(100):
                    if hung["now"]:
                        break
                    await asyncio.sleep(0.01)
                infos = await asyncio.wait_for(
                    loop.getaddrinfo("127.0.0.1", 443, type=socket.SOCK_STREAM), 1.0
                )
                await asyncio.sleep(0.1)
                most = hung["most"]
            finally:
                release.set()
                outcomes = await asyncio.gather(*fetches, return_exceptions=True)
                await fetcher.close()

        self.assertEqual(infos[0][4][0], "127.0.0.1")
        # The fetcher's own threads cap how many hung lookups pile up.
        self.assertEqual(most, RESOLVE_WORKERS)
        for outcome in outcomes:
            self.assertIsInstance(outcome, FetchError)


class _FakeStream:
    def __init__(self) -> None:
        self.server_hostname = None

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self.server_hostname = server_hostname
        raise httpcore.ConnectError("stop here")

    async def aclose(self) -> None:
        pass

    def get_extra_info(self, info):
        return None


class _FakeInner:
    def __init__(self) -> None:
        self.connected: list[tuple[str, int]] = []
        self.stream = _FakeStream()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connected.append((host, port))
        return self.stream

    async def sleep(self, seconds):
        pass


class PinnedConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_connects_to_the_checked_address_with_sni_for_the_host(self) -> None:
        async def resolve(host, port):
            return ["93.184.215.14"]

        backend = _PinnedNetworkBackend(resolve, (80, 443))
        inner = _FakeInner()
        backend._inner = inner
        async with httpcore.AsyncConnectionPool(network_backend=backend) as pool:
            with self.assertRaises(httpcore.ConnectError):
                await pool.request("GET", "https://www.example.com/")

        self.assertEqual(inner.connected, [("93.184.215.14", 443)])
        self.assertEqual(inner.stream.server_hostname, "www.example.com")

    async def test_ports_and_unix_sockets_are_refused_at_connect(self) -> None:
        async def resolve(host, port):  # pragma: no cover - never reached
            return ["93.184.215.14"]

        backend = _PinnedNetworkBackend(resolve, (80, 443))
        with self.assertRaises(BlockedURLError):
            await backend.connect_tcp("example.com", 22)
        with self.assertRaises(BlockedURLError):
            await backend.connect_unix_socket("/var/run/docker.sock")


class _Site(BaseHTTPRequestHandler):
    """Pages for the transport tests."""

    big5 = (
        "<html><head><meta charset='big5'><title>台北天氣</title></head>"
        "<body><p>今天台北多雲時晴，氣溫攝氏二十八度。</p></body></html>"
    ).encode("cp950")
    pages = {
        "/html": (200, {"Content-Type": "text/html; charset=utf-8"}, "<p>你好</p>".encode()),
        "/big5-header": (200, {"Content-Type": "text/html; charset=big5"}, big5),
        "/big5-meta": (200, {"Content-Type": "text/html"}, big5),
        "/plain": (200, {"Content-Type": "text/plain"}, b"plain text"),
        "/pdf": (200, {"Content-Type": "application/pdf"}, b"%PDF-1.7"),
        "/json": (200, {"Content-Type": "application/json"}, b"{}"),
        "/none": (200, {}, b"<p>no type</p>"),
        "/big": (200, {"Content-Type": "text/html"}, b"<p>" + b"x" * 5000 + b"</p>"),
        "/gzip-bomb": (
            200,
            {"Content-Type": "text/html", "Content-Encoding": "gzip"},
            gzip.compress(b"<p>" + b"0" * 3_000_000 + b"</p>"),
        ),
        "/br": (200, {"Content-Type": "text/html", "Content-Encoding": "br"}, b"\x00"),
        "/missing": (404, {"Content-Type": "text/html"}, b"gone"),
        "/to-private": (302, {"Location": "http://10.0.0.5/admin"}, b""),
        "/to-metadata": (301, {"Location": "http://metadata.google.internal/computeMetadata/v1/"}, b""),
        "/to-port-22": (302, {"Location": "http://public.test:22/"}, b""),
        "/to-ftp": (302, {"Location": "ftp://public.test/"}, b""),
    }

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.startswith("/loop"):
            hop = int(self.path.removeprefix("/loop") or "0")
            self.send_response(302)
            self.send_header("Location", f"/loop{hop + 1}")
            self.end_headers()
            return
        if self.path == "/hop":
            self.send_response(307)
            self.send_header("Location", "/html")
            self.end_headers()
            return
        if self.path in ("/set-cookie", "/cookie-hop"):
            # A session cookie for the whole site (and, on /cookie-hop, a
            # redirect to the page that shows what came back).
            self.send_response(302 if self.path == "/cookie-hop" else 200)
            self.send_header("Set-Cookie", "session=user-a-secret; Path=/")
            if self.path == "/cookie-hop":
                self.send_header("Location", "/echo-cookie")
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/echo-cookie":
            body = f"cookie=[{self.headers.get('Cookie', '')}]".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, headers, body = self.pages.get(self.path, (404, {}, b""))
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class FetchTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    async def asyncSetUp(self) -> None:
        self.resolved: list[str] = []

        async def resolver(host, port):
            self.resolved.append(host)
            if host == "public.test":
                return ["127.0.0.1"]
            try:
                return [str(ipaddress.ip_address(host))]
            except ValueError:
                raise OSError(f"unknown host {host}") from None

        # The local server stands in for a public site: 127.0.0.1 and its
        # port are allowed, everything else follows the production rules.
        self.fetcher = PageFetcher(
            max_bytes=1000,
            resolver=resolver,
            address_allowed=lambda ip: ip == _LOCALHOST or is_public_address(ip),
            allowed_ports=(80, 443, self.port),
        )

    async def asyncTearDown(self) -> None:
        await self.fetcher.close()

    def url(self, path: str) -> str:
        return f"http://public.test:{self.port}{path}"

    async def test_a_page_is_fetched_through_the_pinned_connection(self) -> None:
        page = await self.fetcher.fetch(self.url("/html"))

        self.assertEqual(page.body, "<p>你好</p>".encode())
        self.assertEqual((page.content_type, page.charset, page.truncated), ("text/html", "utf-8", False))
        self.assertEqual(page.url, self.url("/html"))
        # The host was resolved by the fetcher's own check, not by the OS.
        self.assertEqual(self.resolved, ["public.test"])

    async def test_redirects_are_followed_and_every_hop_is_checked(self) -> None:
        page = await self.fetcher.fetch(self.url("/hop"))
        self.assertEqual(page.url, self.url("/html"))

        for path in ("/to-private", "/to-metadata", "/to-port-22", "/to-ftp"):
            with self.subTest(path=path):
                with self.assertRaises(BlockedURLError):
                    await self.fetcher.fetch(self.url(path))

    async def test_at_most_max_redirects(self) -> None:
        with self.assertRaises(FetchError) as caught:
            await self.fetcher.fetch(self.url("/loop"))
        self.assertIn(f"more than {MAX_REDIRECTS} redirects", str(caught.exception))

    async def test_only_page_content_types_are_read(self) -> None:
        self.assertEqual(
            ALLOWED_CONTENT_TYPES, {"text/html", "application/xhtml+xml", "text/plain"}
        )
        self.assertEqual((await self.fetcher.fetch(self.url("/plain"))).content_type, "text/plain")
        for path in ("/pdf", "/json", "/none"):
            with self.subTest(path=path):
                with self.assertRaises(FetchError) as caught:
                    await self.fetcher.fetch(self.url(path))
                self.assertIn("is not a page", str(caught.exception))

    async def test_a_body_stops_at_the_byte_cap(self) -> None:
        page = await self.fetcher.fetch(self.url("/big"))

        self.assertEqual(len(page.body), 1000)
        self.assertTrue(page.truncated)

    async def test_compressed_bodies_are_capped_while_they_inflate(self) -> None:
        page = await self.fetcher.fetch(self.url("/gzip-bomb"))

        self.assertEqual(len(page.body), 1000)
        self.assertTrue(page.body.startswith(b"<p>000"))
        self.assertTrue(page.truncated)
        with self.assertRaises(FetchError):
            await self.fetcher.fetch(self.url("/br"))

    async def test_errors_are_fetch_errors(self) -> None:
        with self.assertRaises(FetchError) as caught:
            await self.fetcher.fetch(self.url("/missing"))
        self.assertIn("HTTP 404", str(caught.exception))
        with self.assertRaises(FetchError):
            await self.fetcher.fetch(f"http://unresolvable.test:{self.port}/")

    async def test_no_cookie_outlives_its_response(self) -> None:
        # One client serves every user: a cookie one fetch received must
        # never go out with a later fetch, nor on a redirect hop.
        await self.fetcher.fetch(self.url("/set-cookie"))
        later = await self.fetcher.fetch(self.url("/echo-cookie"))
        hopped = await self.fetcher.fetch(self.url("/cookie-hop"))

        self.assertEqual(later.body, b"cookie=[]")
        self.assertEqual(hopped.body, b"cookie=[]")
        self.assertEqual(list(self.fetcher._client.cookies.jar), [])

    async def test_big5_pages_keep_their_bytes_and_charset(self) -> None:
        from src.services.search.extract import decode_html

        header = await self.fetcher.fetch(self.url("/big5-header"))
        meta = await self.fetcher.fetch(self.url("/big5-meta"))

        self.assertEqual(header.charset, "big5")
        self.assertIsNone(meta.charset)
        for page in (header, meta):
            with self.subTest(charset=page.charset):
                self.assertIn("今天台北多雲時晴", decode_html(page.body, page.charset))

    async def test_production_defaults_refuse_the_classic_targets(self) -> None:
        fetcher = PageFetcher()
        try:
            for url in (
                "http://169.254.169.254/computeMetadata/v1/",
                "http://metadata.google.internal/computeMetadata/v1/",
                "http://10.138.0.10:80/search",
                "http://100.64.0.1/",
                "http://0.0.0.0/",
                "http://[::ffff:169.254.169.254]/",
                "http://[64:ff9b::a9fe:a9fe]/",
                "http://224.0.0.1/",
                "http://2130706433/",
                "http://127.0.0.1/",
                "http://localhost/",
                "http://example.com:22/",
                f"http://127.0.0.1:{self.port}/html",
            ):
                with self.subTest(url=url):
                    with self.assertRaises(BlockedURLError):
                        await fetcher.fetch(url)
        finally:
            await fetcher.close()

    async def test_the_environment_proxy_is_ignored(self) -> None:
        from unittest.mock import patch

        with patch.dict("os.environ", {"HTTP_PROXY": "http://10.0.0.9:3128", "ALL_PROXY": "http://10.0.0.9:3128"}):
            fetcher = PageFetcher(
                resolver=lambda host, port: _const(["127.0.0.1"]),
                address_allowed=lambda ip: ip == _LOCALHOST,
                allowed_ports=(self.port,),
            )
            try:
                page = await fetcher.fetch(self.url("/html"))
            finally:
                await fetcher.close()
        self.assertEqual(page.body, "<p>你好</p>".encode())


async def _const(value):
    return value


if __name__ == "__main__":
    unittest.main()
