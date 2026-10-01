"""Where post-webhook work runs: Cloud Tasks in production, inline otherwise.

The service runs with request-based billing (``--cpu-throttling``), so work
started with ``asyncio.create_task`` after a response is sent gets almost no
CPU. Every piece of such work is instead a *task*: a ``kind`` plus a small
JSON payload. :class:`CloudTasksQueue` turns it into an HTTP request back to
``/internal/tasks/{kind}``, which is a real request and therefore gets CPU.
:class:`InlineTaskQueue` just awaits the handler (local dev and tests).

Enqueueing returns False when the task was certainly not created, and
:func:`enqueue_or_run` then runs the handler inline so nothing is dropped.
When Cloud Tasks gives no clear answer (a timeout or 5xx after the request
went out) the task may exist, so running it inline could run it twice: the
queue retries once under the same task name, and if that is unclear too it
raises :class:`TaskMaybeQueuedError`. See ADR-0006.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

import httpx

from src.utils import gcp_metadata
from src.utils.logger import logger

TASKS_API_BASE_URL = "https://cloudtasks.googleapis.com/v2"
TASKS_PATH_PREFIX = "/internal/tasks/"
# One Cloud Tasks billable operation covers 32 KB; bigger payloads cost more
# and a LINE event never needs it, so they run inline instead.
MAX_TASK_PAYLOAD_BYTES = 32 * 1024
ENQUEUE_TIMEOUT_SECONDS = 3.0
# How much of a Cloud Tasks error message goes into the log.
ERROR_MESSAGE_LOG_CHARS = 200

Dispatcher = Callable[[str, dict[str, Any]], Awaitable[Any]]

_HOST_RE = re.compile(r"^[A-Za-z0-9.-]+(:\d{1,5})?$")

# Transport errors raised before any byte of the request left, so Cloud
# Tasks cannot have created the task. Any other failure is unclear.
_NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol)

# Outcomes of one CreateTask call.
_CREATED = "created"
_REFUSED = "refused"  # certainly not created
_UNCLEAR = "unclear"  # may or may not have been created


class TaskMaybeQueuedError(RuntimeError):
    """The queue cannot tell whether it took the task (it may run later)."""


class TaskQueue(Protocol):
    mode: str  # "cloud" or "inline", reported by /health

    async def enqueue(self, kind: str, payload: dict[str, Any]) -> bool:
        """Hand ``payload`` to the ``kind`` handler; False if it was not queued.

        Raises:
            TaskMaybeQueuedError: It may or may not have been queued.
        """

    async def aclose(self) -> None:
        """Release any HTTP client the queue owns."""


class InlineTaskQueue:
    """Runs the handler right away, inside the caller's request."""

    mode = "inline"

    def __init__(self, dispatcher: Dispatcher) -> None:
        self._dispatcher = dispatcher

    async def enqueue(self, kind: str, payload: dict[str, Any]) -> bool:
        """Run the handler; True even if it failed, since it was attempted.

        Returning False (or raising) would make :func:`enqueue_or_run` run
        the same handler a second time as its "inline fallback".
        """
        try:
            await self._dispatcher(kind, payload)
        except Exception as exc:
            logger.error(f"Task {kind} failed while running inline: {exc}", exc_info=True)
        return True

    async def aclose(self) -> None:
        return None


class BaseUrlResolver:
    """The service's own public URL, used as the Cloud Tasks target.

    ``PUBLIC_BASE_URL`` wins when set. Otherwise the URL is derived from the
    ``Host`` of the first authenticated request and cached. Callers only feed
    it requests that already passed signature or token checks, so a forged
    Host header cannot redirect tasks. The scheme is always https, whatever
    ``X-Forwarded-Proto`` says: tasks carry the shared secret, and an http
    target would also hit run.app's redirect, which a single-attempt queue
    counts as a failed (and dropped) task.
    """

    def __init__(self, configured: str = "") -> None:
        self._configured = (configured or "").strip().rstrip("/")
        self._derived = ""

    def remember(self, headers: Mapping[str, str]) -> None:
        if self._configured or self._derived:
            return
        host = (headers.get("host") or "").strip()
        if not _HOST_RE.match(host):
            return
        self._derived = f"https://{host}"
        logger.info(f"Cloud Tasks target base URL derived from request: {self._derived}")

    def __call__(self) -> str:
        return self._configured or self._derived


class CloudTasksQueue:
    """Creates HTTP tasks through the Cloud Tasks REST API."""

    mode = "cloud"

    def __init__(
        self,
        *,
        project: str = "",
        location: str = "",
        queue: str,
        secret: str,
        base_url_provider: Callable[[], str],
        kind_queues: Mapping[str, str] | None = None,
        token_provider: Callable[[], Awaitable[str]] = gcp_metadata.get_access_token,
        project_provider: Callable[[], Awaitable[str]] = gcp_metadata.get_project_id,
        location_provider: Callable[[], Awaitable[str]] = gcp_metadata.get_region,
        client: httpx.AsyncClient | None = None,
        dispatch_deadline_seconds: int = 300,
        timeout: float = ENQUEUE_TIMEOUT_SECONDS,
    ) -> None:
        self._project = project.strip()
        self._location = location.strip()
        self._queue = queue.strip()
        # Kinds with a queue of their own (the Colab controller's slow
        # ``colab`` tasks), so they never hold the event queue's slots.
        self._kind_queues = {kind: name.strip() for kind, name in (kind_queues or {}).items() if name.strip()}
        self._secret = secret
        self._base_url_provider = base_url_provider
        self._token_provider = token_provider
        self._project_provider = project_provider
        self._location_provider = location_provider
        self._client = client
        self._owns_client = client is None
        self._deadline = int(dispatch_deadline_seconds)
        self._timeout = timeout

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def queue_for(self, kind: str) -> str:
        """The queue *kind*'s tasks go to."""
        return self._kind_queues.get(kind, self._queue)

    async def _queue_path(self, kind: str) -> str:
        # Unset project / location come from the metadata server once.
        if not self._project:
            self._project = (await self._project_provider()).strip()
        if not self._location:
            self._location = (await self._location_provider()).strip()
        return f"projects/{self._project}/locations/{self._location}/queues/{self.queue_for(kind)}"

    def build_task(self, base_url: str, kind: str, body: bytes, name: str = "") -> dict[str, Any]:
        return {
            "task": {
                **({"name": name} if name else {}),
                "dispatchDeadline": f"{self._deadline}s",
                "httpRequest": {
                    "url": base_url.rstrip("/") + TASKS_PATH_PREFIX + kind,
                    "httpMethod": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "X-Tasks-Token": self._secret,
                    },
                    "body": base64.b64encode(body).decode("ascii"),
                },
            }
        }

    async def enqueue(self, kind: str, payload: dict[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > MAX_TASK_PAYLOAD_BYTES:
            logger.warning(
                f"Task {kind} payload is {len(body)} bytes (> {MAX_TASK_PAYLOAD_BYTES}); "
                "not queued"
            )
            return False
        base_url = self._base_url_provider()
        if not base_url:
            logger.warning(f"Task {kind} not queued: the service base URL is not known yet")
            return False

        started = time.perf_counter()
        try:
            queue_path = await self._queue_path(kind)
            token = await self._token_provider()
        except gcp_metadata.MetadataError as exc:
            # Built by gcp_metadata from the path and status only.
            logger.warning(f"Task {kind} not queued: {self._redact(str(exc))}")
            return False
        except Exception as exc:
            # Only the exception type: messages from auth libraries can
            # echo request details, and the task carries the shared secret.
            logger.warning(f"Task {kind} not queued: {type(exc).__name__}")
            return False

        # A named task makes CreateTask idempotent: creating the same name
        # again answers 409 ALREADY_EXISTS while the task exists and for
        # about an hour after it ran or was deleted (queues made with gcloud
        # or the API; queue.yaml queues keep names ~9 days). So a retry after
        # an unclear answer cannot make a second task. The name is random
        # per call, not derived from the payload: equal payloads are queued
        # again on purpose within that hour (a chat's passive_flush or
        # compact, Colab actions) and must not come back as silent 409s.
        # Google asks for random or hashed ids, never sequential ones.
        task_id = uuid.uuid4().hex
        url = f"{TASKS_API_BASE_URL}/{queue_path}/tasks"
        task = self.build_task(base_url, kind, body, name=f"{queue_path}/tasks/{task_id}")
        headers = {"Authorization": f"Bearer {token}"}

        outcome, detail = await self._create(url, task, headers, retry=False)
        if outcome == _UNCLEAR:
            logger.warning(f"Task {kind}: no clear answer from Cloud Tasks ({detail}); retrying {task_id}")
            outcome, detail = await self._create(url, task, headers, retry=True)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if outcome == _CREATED:
            logger.info(f"Task {kind} queued as {task_id} in {elapsed_ms} ms")
            return True
        if outcome == _REFUSED:
            logger.warning(f"Task {kind} not queued: {detail} after {elapsed_ms} ms")
            return False
        raise TaskMaybeQueuedError(f"task {kind} ({task_id}): {detail} after {elapsed_ms} ms")

    async def _create(
        self, url: str, task: dict[str, Any], headers: dict[str, str], *, retry: bool
    ) -> tuple[str, str]:
        """One CreateTask call: its outcome and a log-safe detail.

        On the retry only a 2xx or ALREADY_EXISTS settles it; any other
        answer says nothing about whether the first call created the task.
        """
        try:
            response = await self._http().post(url, json=task, headers=headers, timeout=self._timeout)
        except _NOT_SENT_ERRORS as exc:
            return (_UNCLEAR if retry else _REFUSED), type(exc).__name__
        except Exception as exc:
            # Only the exception type (see enqueue); sent, perhaps processed.
            return _UNCLEAR, type(exc).__name__
        if 200 <= response.status_code < 300:
            return _CREATED, ""
        detail = f"Cloud Tasks returned HTTP {response.status_code}{self._error_detail(response)}"
        if retry and response.status_code == 409 and self._google_status(response) == "ALREADY_EXISTS":
            return _CREATED, detail  # the first call created it
        if retry or response.status_code >= 500:
            return _UNCLEAR, detail
        return _REFUSED, detail

    def _redact(self, text: str) -> str:
        return text.replace(self._secret, "[redacted]") if self._secret else text

    @staticmethod
    def _google_error(response: httpx.Response) -> dict[str, Any]:
        try:
            error = response.json().get("error")
        except (ValueError, AttributeError):
            return {}
        return error if isinstance(error, dict) else {}

    @classmethod
    def _google_status(cls, response: httpx.Response) -> str:
        return str(cls._google_error(response).get("status") or "").strip()

    def _error_detail(self, response: httpx.Response) -> str:
        """``error.status`` and a short ``error.message`` from a Google error.

        They tell a missing queue from one re-activating after 30 idle days
        (both 404), or missing IAM from a disabled API (both 403). Google
        does not echo the task body; the secret is masked anyway.
        """
        error = self._google_error(response)
        status = self._google_status(response)
        message = " ".join(str(error.get("message") or "").split())[:ERROR_MESSAGE_LOG_CHARS]
        detail = ": ".join(part for part in (status, message) if part)
        return f" ({self._redact(detail)})" if detail else ""

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


async def enqueue_or_run(
    queue: TaskQueue,
    kind: str,
    payload: dict[str, Any],
    run_inline: Dispatcher,
    *,
    inline_if_unsure: bool = False,
) -> bool:
    """Queue the work; if that fails, run it inline so it is not lost.

    When the queue cannot tell whether it took the work
    (:class:`TaskMaybeQueuedError`), running it here could run it twice, so
    it runs inline only with ``inline_if_unsure``: for work whose handler
    drops a second delivery. Otherwise it is left to the queue.

    Returns True when the work was (or may have been) queued and did not
    run here, False when it ran inline.
    """
    try:
        if await queue.enqueue(kind, payload):
            return True
        reason = "enqueue returned False"
    except TaskMaybeQueuedError as exc:
        if not inline_if_unsure:
            logger.error(
                f"Task {kind} possibly not queued ({exc}); not running it inline, "
                "which could run it twice"
            )
            return True
        reason = f"possibly not queued: {exc}; a second delivery is dropped"
    except Exception as exc:
        reason = f"enqueue raised {type(exc).__name__}"
    logger.error(f"Task {kind} could not be queued ({reason}); running it inline")
    await run_inline(kind, payload)
    return False


class TaskNotQueuedError(RuntimeError):
    """:func:`enqueue_only` could not queue the task (and did not run it)."""


async def enqueue_only(queue: TaskQueue, kind: str, payload: dict[str, Any]) -> None:
    """Queue the work, never run it here: no inline fallback.

    For work that must not run inside the request that queues it, such as
    the Colab controller's calls (seconds to minutes; the cron request has
    a 60 s deadline, a task a handler budget). The caller queues it again
    later (the controller's lease or next tick). An :class:`InlineTaskQueue`
    still runs it at once: that is local development (ADR-0010 requires
    Cloud Tasks for Colab on Cloud Run).

    Raises:
        TaskNotQueuedError: The queue did not take it.
    """
    try:
        queued = await queue.enqueue(kind, payload)
    except Exception as exc:
        raise TaskNotQueuedError(f"task {kind}: enqueue raised {type(exc).__name__}") from None
    if not queued:
        raise TaskNotQueuedError(f"task {kind} was not queued")


def select_tasks_mode(settings, environ: Mapping[str, str] | None = None) -> str:
    """Resolve TASKS_MODE to ``cloud`` or ``inline``.

    ``auto`` means cloud only on Cloud Run (``K_SERVICE`` is set) with an
    ``INTERNAL_TASKS_SECRET``. An explicit ``cloud`` without the secret also
    falls back to inline: every task would be refused with 403 and lost.
    """
    environ = os.environ if environ is None else environ
    mode = settings.tasks_mode
    has_secret = bool(settings.internal_tasks_secret)
    if mode == "inline":
        return "inline"
    if mode == "cloud":
        if has_secret:
            return "cloud"
        logger.error(
            "TASKS_MODE=cloud but INTERNAL_TASKS_SECRET is empty; running tasks inline"
        )
        return "inline"
    if environ.get("K_SERVICE") and has_secret:
        return "cloud"
    return "inline"
