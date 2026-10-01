"""The metadata-server helpers, against an httpx.MockTransport."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import httpx

from src.utils import gcp_metadata
from src.utils.gcp_metadata import GcpMetadata, MetadataError, parse_region


class _FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _MetadataServer:
    """Answers metadata paths from a dict and records every request."""

    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/computeMetadata/v1/")
        return self.routes.get(path, httpx.Response(404))


def _metadata(server: _MetadataServer, clock: _FakeClock | None = None) -> GcpMetadata:
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return GcpMetadata(client, clock=clock or _FakeClock())


class GcpMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_access_token_is_cached_until_shortly_before_expiry(self) -> None:
        tokens = iter(["ya29.first", "ya29.second"])
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200, json={"access_token": next(tokens), "expires_in": 3599, "token_type": "Bearer"}
            )

        clock = _FakeClock()
        metadata = GcpMetadata(
            httpx.AsyncClient(transport=httpx.MockTransport(handler)), clock=clock
        )
        self.addAsyncCleanup(metadata._client.aclose)

        self.assertEqual(await metadata.get_access_token(), "ya29.first")
        clock.now += 3599 - 61  # still before expires_in - 60 s
        self.assertEqual(await metadata.get_access_token(), "ya29.first")
        self.assertEqual(len(requests), 1)

        clock.now += 2  # now inside the 60 s refresh margin
        self.assertEqual(await metadata.get_access_token(), "ya29.second")
        self.assertEqual(len(requests), 2)

        request = requests[0]
        self.assertEqual(
            str(request.url),
            "http://metadata.google.internal/computeMetadata/v1/"
            "instance/service-accounts/default/token",
        )
        self.assertEqual(request.headers["Metadata-Flavor"], "Google")

    async def test_project_region_and_email(self) -> None:
        server = _MetadataServer(
            {
                "project/project-id": httpx.Response(200, text="my-project\n"),
                "instance/region": httpx.Response(200, text="projects/123456/regions/us-west1"),
                "instance/service-accounts/default/email": httpx.Response(
                    200, text="runtime@my-project.iam.gserviceaccount.com"
                ),
            }
        )
        metadata = _metadata(server)
        self.addAsyncCleanup(metadata._client.aclose)

        self.assertEqual(await metadata.get_project_id(), "my-project")
        self.assertEqual(await metadata.get_region(), "us-west1")
        self.assertEqual(
            await metadata.get_service_account_email(),
            "runtime@my-project.iam.gserviceaccount.com",
        )
        for request in server.requests:
            self.assertEqual(request.headers["Metadata-Flavor"], "Google")

    async def test_errors_raise_metadata_error(self) -> None:
        cases = {
            "http 500": httpx.Response(500),
            "malformed token": httpx.Response(200, json={"no_token": True}),
            "not json": httpx.Response(200, text="<html>"),
        }
        for label, response in cases.items():
            with self.subTest(label):
                server = _MetadataServer(
                    {"instance/service-accounts/default/token": response}
                )
                metadata = _metadata(server)
                with self.assertRaises(MetadataError):
                    await metadata.get_access_token()
                await metadata._client.aclose()

    async def test_unreachable_server_raises_metadata_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to metadata.google.internal", request=request)

        metadata = GcpMetadata(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        self.addAsyncCleanup(metadata._client.aclose)

        with self.assertRaises(MetadataError):
            await metadata.get_project_id()

    async def test_empty_value_raises(self) -> None:
        server = _MetadataServer({"project/project-id": httpx.Response(200, text="  ")})
        metadata = _metadata(server)
        self.addAsyncCleanup(metadata._client.aclose)

        with self.assertRaises(MetadataError):
            await metadata.get_project_id()

    def test_parse_region(self) -> None:
        self.assertEqual(parse_region("projects/123/regions/us-west1"), "us-west1")
        self.assertEqual(parse_region("us-central1\n"), "us-central1")


class SyncServiceAccountEmailTests(unittest.TestCase):
    def _urlopen_returning(self, code: int, body: bytes) -> MagicMock:
        response = MagicMock()
        response.getcode.return_value = code
        response.read.return_value = body
        context = MagicMock()
        context.__enter__.return_value = response
        context.__exit__.return_value = False
        return MagicMock(return_value=context)

    def test_returns_the_email_from_the_metadata_server(self) -> None:
        urlopen = self._urlopen_returning(200, b"runtime@example.com\n")
        with patch.object(gcp_metadata, "urlopen", urlopen):
            self.assertEqual(gcp_metadata.get_service_account_email_sync(), "runtime@example.com")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Metadata-flavor"), "Google")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3.0)

    def test_returns_empty_string_on_failure(self) -> None:
        cases = {
            "non-200": self._urlopen_returning(404, b"runtime@example.com"),
            "not an email": self._urlopen_returning(200, b"default"),
            "unreachable": MagicMock(side_effect=OSError("offline")),
        }
        for label, urlopen in cases.items():
            with self.subTest(label), patch.object(gcp_metadata, "urlopen", urlopen):
                self.assertEqual(gcp_metadata.get_service_account_email_sync(), "")


if __name__ == "__main__":
    unittest.main()
