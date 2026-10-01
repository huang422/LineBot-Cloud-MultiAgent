"""Helpers over the GCE / Cloud Run metadata server.

Only reachable on GCP. Every call carries ``Metadata-Flavor: Google`` and a
short timeout, so off GCP they fail fast instead of hanging a request.

The async :class:`GcpMetadata` backs the Cloud Tasks client (access token,
project, region). The sync :func:`get_service_account_email_sync` is for
``StorageService``, which resolves the signing identity in its constructor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import monotonic
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import httpx

METADATA_BASE_URL = "http://metadata.google.internal/computeMetadata/v1/"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}
DEFAULT_TIMEOUT_SECONDS = 2.0
# Refresh the cached access token this long before the server says it expires.
TOKEN_REFRESH_MARGIN_SECONDS = 60.0

_TOKEN_PATH = "instance/service-accounts/default/token"
_EMAIL_PATH = "instance/service-accounts/default/email"
_PROJECT_PATH = "project/project-id"
_REGION_PATH = "instance/region"


class MetadataError(Exception):
    """The metadata server was unreachable or answered unexpectedly."""


def parse_region(raw: str) -> str:
    """``projects/123/regions/us-west1`` -> ``us-west1``."""
    return raw.strip().rstrip("/").rsplit("/", 1)[-1]


class GcpMetadata:
    """Async metadata client with a cached access token."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._clock = clock
        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _get(self, path: str) -> httpx.Response:
        try:
            response = await self._http().get(
                METADATA_BASE_URL + path,
                headers=METADATA_HEADERS,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise MetadataError(f"metadata {path} unreachable: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise MetadataError(f"metadata {path} returned HTTP {response.status_code}")
        return response

    async def _get_text(self, path: str) -> str:
        value = (await self._get(path)).text.strip()
        if not value:
            raise MetadataError(f"metadata {path} returned an empty value")
        return value

    async def get_access_token(self) -> str:
        """OAuth2 token of the runtime service account, cached until near expiry."""
        async with self._token_lock:
            if self._token and self._clock() < self._token_expires_at:
                return self._token
            try:
                data = (await self._get(_TOKEN_PATH)).json()
                token = str(data["access_token"])
                expires_in = float(data.get("expires_in", 0))
            except (ValueError, KeyError, TypeError) as exc:
                raise MetadataError("metadata token response was malformed") from exc
            if not token:
                raise MetadataError("metadata token response was empty")
            self._token = token
            self._token_expires_at = self._clock() + max(
                0.0, expires_in - TOKEN_REFRESH_MARGIN_SECONDS
            )
            return token

    async def get_project_id(self) -> str:
        return await self._get_text(_PROJECT_PATH)

    async def get_region(self) -> str:
        return parse_region(await self._get_text(_REGION_PATH))

    async def get_service_account_email(self) -> str:
        return await self._get_text(_EMAIL_PATH)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


_default: GcpMetadata | None = None


def get_metadata() -> GcpMetadata:
    """Process-wide client, so the token cache is shared."""
    global _default
    if _default is None:
        _default = GcpMetadata()
    return _default


async def get_access_token() -> str:
    return await get_metadata().get_access_token()


async def get_project_id() -> str:
    return await get_metadata().get_project_id()


async def get_region() -> str:
    return await get_metadata().get_region()


async def get_service_account_email() -> str:
    return await get_metadata().get_service_account_email()


async def close_metadata() -> None:
    global _default
    if _default is not None:
        await _default.aclose()
        _default = None


def get_service_account_email_sync(timeout: float = 3.0) -> str:
    """The runtime service account email, or "" when it cannot be resolved.

    On Cloud Run / GCE the default credentials report ``default`` as the
    email, but IAM signBlob needs the real address.
    """
    try:
        request = UrlRequest(METADATA_BASE_URL + _EMAIL_PATH, headers=METADATA_HEADERS)
        with urlopen(request, timeout=timeout) as response:
            if response.getcode() != 200:
                return ""
            email = response.read().decode("utf-8").strip()
            if "@" in email:
                return email
    except (OSError, UnicodeDecodeError):
        pass
    return ""
