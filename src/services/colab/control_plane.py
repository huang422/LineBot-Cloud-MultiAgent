"""The Colab control plane: assign, list, release and meter runtimes, run cells.

An async facade over the pinned ``google-colab-cli`` 0.7.4 *library*, never
the ``colab`` binary (the binary needs a writable HOME, checks PyPI on every
call, logs bearer tokens at DEBUG and writes ``--env`` values into its
history). Only :mod:`colab_cli.client`, :mod:`colab_cli.runtime` and
:mod:`colab_cli.contents` are used, imported on first use; ``colab_cli.common``,
``.auth`` and ``.commands`` never are (they keep global state, set up file
logging and call ``exit()``). See ADR-0010.

- **Credentials.** The owner's ``authorized_user`` refresh token, minted once
  by a human and mounted from Secret Manager at ``COLAB_ADC_PATH``. The file
  is read once and becomes a google-auth ``AuthorizedSession``;
  ``GOOGLE_APPLICATION_CREDENTIALS`` is never set, so Firestore, Cloud
  Storage and Cloud Tasks keep using the service account.
- **Session backend** (``colab.research.google.com/tun/m``, JSON behind an
  XSSI prefix): :meth:`~ColabControlPlane.assign` asks for the 40 GB A100
  (``Variant.GPU`` + ``Accelerator.A100``, never the high-mem shape). The
  same ``nbh`` returns the runtime already assigned to it, so an assign
  that timed out is retried with the same ``nbh``, and an ``nbh`` given up
  without an answer is looked up with :meth:`~ColabControlPlane.find_assignment`
  (the GET half only), so a runtime its timed-out POST left is found and
  released. :meth:`~ColabControlPlane.list_assignments` also hands out a fresh proxy
  token (valid for an hour) with each runtime, and needs only each one's
  endpoint and proxy: an accelerator, variant or shape the library does not
  know keeps its raw value, so one new kind of runtime on a shared account
  cannot fail every listing;
  :meth:`~ColabControlPlane.unassign` treats 404 as already released;
  :meth:`~ColabControlPlane.ccu_info` keeps the raw ``ccu-info`` fields the
  library drops (``eligibleGpus``).
- **Runtime proxy** (an assignment's ``runtimeProxyInfo``):
  :meth:`~ColabControlPlane.run_cell` runs one cell in a kernel and
  returns the kernel and session ids to keep (without them every call
  starts a new kernel that is never shut down);
  :meth:`~ColabControlPlane.upload` PUTs files through the Jupyter Contents
  API. The proxy token is never stored or logged.

Every call runs in a worker thread under an explicit budget: each HTTP
request gets the time left as its timeout (and as google-auth's
``max_allowed_time``, which also bounds a token refresh), and the await
gives up :data:`CALL_GRACE_SECONDS` after the budget. A cell is bounded by a
``threading.Timer`` scoped to the call that closes the kernel websocket: the
kernel client's own timeout never fires while a cell keeps producing output
(googlecolab/google-colab-cli#82). These calls take seconds to minutes, so
they run only from Cloud Tasks handlers and the cron tick, never from
``/webhook``.

Failures come out as :class:`ControlPlaneError` or one of its subclasses,
with tokens scrubbed from the message:

- :class:`NoCapacityError`: assign got 412 or 503. Cool down, then retry.
- :class:`CredentialOrPolicyError`: the refresh token is dead
  (``invalid_grant``, ``invalid_client``, reauthentication needed), the
  account is refused (400 on the A100 assign: no entitlement; 401/403 from
  the backend; DENYLISTED). A human must act, so disable the tier (not on an
  unassign's alone: Colab may refuse to release a runtime already gone,
  so the controller leaves it to the next listing). A 400 on the CPU
  assign of a dry run is a plain :class:`ControlPlaneError` (colab_cli
  reads 400 as "no entitlement" only when an accelerator was asked for).
- :class:`TransientControlPlaneError` (:class:`ControlPlaneTimeoutError` for
  a spent budget): network trouble, 429/5xx, a token refresh that failed
  without saying the token is dead (google-auth calls a 502, an HTML page or
  a 200 without a token non-retryable too), or the runtime proxy refused
  its token. Retry on a later tick.
- :class:`ControlPlaneError` itself: an answer this client does not
  understand. The tunnel API is private and may change without notice.

At DEBUG the libraries log the Google bearer header, proxy tokens in
response bodies and in URLs, and the kernel client logs its websocket URL,
token included, even at ERROR. :func:`quiet_library_loggers` holds their
loggers at WARNING and scrubs whatever they still log.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote, urljoin

from src.utils.logger import logger

DEFAULT_ADC_PATH = "/secrets/colab/adc.json"

# Budget of one call (seconds), every request and token refresh included.
LIST_TIMEOUT_SECONDS = 15.0
CCU_TIMEOUT_SECONDS = 15.0
ASSIGN_TIMEOUT_SECONDS = 120.0
UNASSIGN_TIMEOUT_SECONDS = 120.0
UPLOAD_TIMEOUT_SECONDS = 60.0
CONNECT_TIMEOUT_SECONDS = 10.0
# The await gives up this long after the worker's own deadline, so the
# worker's classified error normally comes out first.
CALL_GRACE_SECONDS = 5.0
# Printed output a CellResult keeps per stream (the end of it).
OUTPUT_LIMIT = 8000

# Held at WARNING (see the module docstring).
LIBRARY_LOGGERS: tuple[str, ...] = ("colab_cli", "jupyter_kernel_client", "urllib3", "websocket")
# Also scrubbed: a logger filter sees only records logged to that exact
# logger, and these children log URLs at WARNING or above.
_SCRUBBED_LOGGERS: tuple[str, ...] = (
    *LIBRARY_LOGGERS,
    "colab_cli.client",
    "urllib3.connectionpool",
)

# CellResult.status
CELL_OK = "ok"
CELL_ERROR = "error"  # the cell raised
CELL_TIMEOUT = "timeout"  # the budget ran out (the cell may still be running)
CELL_DISCONNECTED = "disconnected"  # the kernel websocket dropped

_BACKEND_OPS = frozenset({"assign", "find_assignment", "list_assignments", "unassign", "ccu_info"})
_ADC_FIELDS = ("client_id", "client_secret", "refresh_token")

_SECRETS = re.compile(
    r"bearer\s+[^\s,;'\"]+"
    # token=, colab-runtime-proxy-token= (URLs, the kernel websocket URL)
    r"|[\w-]*token=[^&\s'\"]*"
    # "token": "…", 'X-Colab-Runtime-Proxy-Token': '…', "refresh_token": "…"
    r"|[\"'][\w-]*(?:token|secret)[\"']\s*:\s*[\"'][^\"']*[\"']"
    # Google access and refresh tokens wherever they appear
    r"|ya29\.[\w.-]+|1//[\w-]{20,}",
    re.IGNORECASE,
)
# A dead credential (OAuth's invalid_grant, invalid_client and
# unauthorized_client; google-auth's reauth) or a refused account.
_CREDENTIAL_MARKERS = re.compile(
    r"invalid_grant|invalid_client|unauthorized_client|Reauthentication is needed|DENYLISTED|abusive",
    re.IGNORECASE,
)


def redact(text: object) -> str:
    """``text`` with bearer headers, tokens and secrets replaced by ``[redacted]``."""
    return _SECRETS.sub("[redacted]", str(text))


class ControlPlaneError(Exception):
    """A control-plane call failed; see the module docstring for the subclasses.

    ``status`` is the HTTP status when there was a response.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(redact(message))
        self.status = status


class NoCapacityError(ControlPlaneError):
    """Assign refused with 412 (too many runtimes, a usage or capacity limit) or 503."""


class CredentialOrPolicyError(ControlPlaneError):
    """The account cannot use Colab from here until a human acts: disable the tier."""


class TransientControlPlaneError(ControlPlaneError):
    """Network trouble, 429/5xx, or a refused proxy token: retry later."""


class ControlPlaneTimeoutError(TransientControlPlaneError):
    """The call spent its budget. An assign may have succeeded anyway: retry it
    with the same ``nbh``, which returns that runtime instead of a second one."""


@dataclass(frozen=True)
class RuntimeAssignment:
    """A Colab runtime assigned to the account.

    ``proxy_token`` opens the runtime's Jupyter server, which means running
    code on the VM. It lasts about an hour, comes fresh with every
    :meth:`ColabControlPlane.list_assignments`, and is never logged or stored
    (``repr`` leaves it out).
    """

    endpoint: str
    proxy_url: str
    proxy_token: str = field(repr=False)
    token_expires_in: int = 0
    # "A100", "NONE", …; "" when the answer did not say.
    accelerator: str = ""
    # "DEFAULT", "GPU" or "TPU".
    variant: str = ""
    # "STANDARD" or "HIGH_RAM"; only listed assignments report it.
    machine_shape: str = ""
    # assign() only: the nbh already had this runtime (a retried assign).
    reused: bool = False


@dataclass(frozen=True)
class CcuInfo:
    """``GET /tun/m/ccu-info``: the account's compute units. Missing fields are None."""

    current_balance: float | None = None
    # CU per hour across every runtime assigned to the account.
    consumption_rate_hourly: float | None = None
    assignments_count: int | None = None
    eligible_gpus: tuple[str, ...] | None = None
    ineligible_gpus: tuple[str, ...] | None = None

    @classmethod
    def from_json(cls, raw: Any) -> CcuInfo:
        if not isinstance(raw, dict):
            raise ControlPlaneError(f"ccu-info is not a JSON object: {type(raw).__name__}")
        count = _number(raw.get("assignmentsCount"))
        return cls(
            current_balance=_number(raw.get("currentBalance")),
            consumption_rate_hourly=_number(raw.get("consumptionRateHourly")),
            assignments_count=None if count is None else int(count),
            eligible_gpus=_names(raw.get("eligibleGpus")),
            ineligible_gpus=_names(raw.get("ineligibleGpus")),
        )

    def gpu_eligible(self, gpu: str = "A100") -> bool | None:
        """Whether the account may get ``gpu`` now; None when ccu-info did not say."""
        if self.eligible_gpus is None:
            return None
        wanted = gpu.upper()
        return any(wanted in name.upper() for name in self.eligible_gpus)


@dataclass(frozen=True)
class CellResult:
    """One cell's outcome. Keep ``kernel_id`` and ``session_id`` for the next call."""

    status: str
    kernel_id: str = ""
    session_id: str = ""
    stdout: str = ""
    stderr: str = ""
    # "ename: evalue" when the cell raised; the reason when it did not finish.
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == CELL_OK


class ColabControlPlane:
    """See the module docstring."""

    def __init__(
        self,
        adc_path: str | os.PathLike[str] = DEFAULT_ADC_PATH,
        *,
        session_factory: Callable[[Any], Any] | None = None,
        runtime_factory: Callable[[str, str, str | None, str | None], Any] | None = None,
    ) -> None:
        """``session_factory(credentials)`` builds the session-backend HTTP
        session (default: google-auth's ``AuthorizedSession``);
        ``runtime_factory(proxy_url, proxy_token, kernel_id, session_id)``
        builds the kernel runtime (default: colab_cli's ``ColabRuntime``).
        Both are seams for tests."""
        self._adc_path = Path(adc_path)
        self._session_factory = session_factory or _authorized_session
        self._runtime_factory = runtime_factory or _colab_runtime
        self._session: Any = None
        self._session_lock = threading.Lock()
        quiet_library_loggers()

    # ── session backend ───────────────────────────────────────

    async def assign(self, nbh: str | uuid.UUID, *, cpu_only: bool = False) -> RuntimeAssignment:
        """Assign a 40 GB A100 runtime to ``nbh`` (idempotent per ``nbh``).

        ``cpu_only`` asks for the default CPU runtime instead, for a dry run.
        """
        notebook_hash = uuid.UUID(str(nbh))
        return await self._call(
            "assign", ASSIGN_TIMEOUT_SECONDS, self._assign_sync, notebook_hash, cpu_only
        )

    async def find_assignment(self, nbh: str | uuid.UUID, *, cpu_only: bool = False) -> RuntimeAssignment | None:
        """The runtime already assigned to ``nbh``, or None; never assigns one.

        Only the GET of assign's two steps, for an ``nbh`` whose assign ended
        without an answer: a timed-out POST may still have left a runtime.
        """
        notebook_hash = uuid.UUID(str(nbh))
        return await self._call(
            "find_assignment", LIST_TIMEOUT_SECONDS, self._find_sync, notebook_hash, cpu_only
        )

    async def list_assignments(self) -> list[RuntimeAssignment]:
        """Every runtime assigned to the account, including ones this bot did not start."""
        return await self._call("list_assignments", LIST_TIMEOUT_SECONDS, self._list_sync)

    async def unassign(self, endpoint: str) -> bool:
        """Release a runtime. False: it was already gone (404)."""
        return await self._call("unassign", UNASSIGN_TIMEOUT_SECONDS, self._unassign_sync, endpoint)

    async def ccu_info(self) -> CcuInfo:
        return await self._call("ccu_info", CCU_TIMEOUT_SECONDS, self._ccu_sync)

    # ── runtime proxy ─────────────────────────────────────────

    async def run_cell(
        self,
        assignment: RuntimeAssignment,
        code: str,
        *,
        budget_s: float,
        kernel_id: str = "",
        session_id: str = "",
        silent: bool = False,
    ) -> CellResult:
        """Run ``code`` in the kernel ``kernel_id`` (a new one when empty).

        A cell that outlives ``budget_s`` comes back as ``CELL_TIMEOUT``: the
        websocket is closed, but the cell keeps running on the VM and holds
        the kernel, so a cell must return quickly (start long work detached).
        ``silent`` runs it without IPython history (``store_history`` off),
        for code that carries secrets. Errors come out as a
        :class:`CellResult` once the kernel is reached; failing to reach it
        raises.
        """
        if budget_s <= 0:
            raise ValueError("budget_s must be positive")
        return await self._call(
            "run_cell",
            float(budget_s),
            self._run_cell_sync,
            assignment,
            code,
            kernel_id,
            session_id,
            silent,
        )

    async def upload(
        self, assignment: RuntimeAssignment, files: Mapping[str, str | os.PathLike[str]]
    ) -> None:
        """PUT local files onto the runtime: ``{remote path: local path}``.

        Remote paths are relative to the Jupyter root, which is ``/`` on
        Colab (``content/bootstrap.py`` is ``/content/bootstrap.py``).
        """
        await self._call(
            "upload", UPLOAD_TIMEOUT_SECONDS, self._upload_sync, assignment, dict(files)
        )

    # ── the workers (in threads) ──────────────────────────────

    async def _call(self, op: str, budget: float, work: Callable[..., Any], *args: Any) -> Any:
        deadline = _Deadline(time.monotonic() + budget)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run, op, work, deadline, *args),
                budget + CALL_GRACE_SECONDS,
            )
        except TimeoutError:
            error = ControlPlaneTimeoutError(f"{op}: no answer within {budget:g} s")
            raise _logged(op, error) from None

    @staticmethod
    def _run(op: str, work: Callable[..., Any], deadline: _Deadline, *args: Any) -> Any:
        try:
            return work(deadline, *args)
        except Exception as exc:
            # Never chained: library exceptions can carry tokens (URLs, bodies).
            raise _logged(op, _translate(op, exc)) from None

    def _client(self, deadline: _Deadline) -> Any:
        from colab_cli.client import Client, Prod

        return Client(Prod(), _DeadlineSession(self._backend_session(), deadline))

    def _backend_session(self) -> Any:
        with self._session_lock:
            if self._session is None:
                self._session = self._session_factory(self._load_credentials())
            return self._session

    def _load_credentials(self) -> Any:
        info = _read_adc(self._adc_path)
        from google.oauth2.credentials import Credentials

        try:
            return Credentials.from_authorized_user_info(info)
        except ValueError as exc:
            raise CredentialOrPolicyError(f"{self._adc_path}: {exc}") from None

    def _assign_sync(
        self, deadline: _Deadline, nbh: uuid.UUID, cpu_only: bool
    ) -> RuntimeAssignment:
        from colab_cli.client import Assignment, ColabRequestError, TooManyAssignmentsError

        client = self._client(deadline)
        try:
            result = client.assign(nbh, **_runtime_kind(cpu_only))
        except TooManyAssignmentsError as exc:
            raise NoCapacityError(f"assign refused (HTTP 412): {exc}", status=412) from None
        except ColabRequestError as exc:
            if cpu_only and _status_of(exc) == 400:
                # 400 means "no entitlement" only for an accelerator (colab_cli
                # 0.7.4 commands/session.py); for the CPU runtime the cause is
                # unknown, so it is retried like any answer not understood.
                raise ControlPlaneError(
                    f"assign failed: HTTP 400 for the CPU runtime | {_body_of(exc)}", status=400
                ) from None
            raise
        return _assignment(result, reused=isinstance(result, Assignment))

    def _find_sync(self, deadline: _Deadline, nbh: uuid.UUID, cpu_only: bool) -> RuntimeAssignment | None:
        from colab_cli.client import Assignment

        # The GET half of assign: it names the runtime ``nbh`` already has,
        # and otherwise hands out the XSRF token the POST would need.
        result = self._client(deadline)._get_assignment(nbh, **_runtime_kind(cpu_only))
        return _assignment(result, reused=True) if isinstance(result, Assignment) else None

    def _list_sync(self, deadline: _Deadline) -> list[RuntimeAssignment]:
        from colab_cli.client import TUN_ENDPOINT

        client = self._client(deadline)
        # The library's request path, but schema=dict: its models reject the
        # whole listing over one runtime of a kind 0.7.4 does not know (a new
        # accelerator in the owner's browser on a shared account).
        url = urljoin(client.colab_domain, f"{TUN_ENDPOINT}/assignments")
        answer = client._issue_request(url, schema=dict)
        items = answer.get("assignments") if isinstance(answer, dict) else None
        if not isinstance(items, list):
            raise ControlPlaneError("list_assignments: the answer has no assignments list")
        return [_listed(item) for item in items]

    def _unassign_sync(self, deadline: _Deadline, endpoint: str) -> bool:
        from colab_cli.client import ColabRequestError

        try:
            self._client(deadline).unassign(endpoint)
        except ColabRequestError as exc:
            if _status_of(exc) == 404:
                logger.info(f"Colab unassign: {endpoint} was already released")
                return False
            raise
        return True

    def _ccu_sync(self, deadline: _Deadline) -> CcuInfo:
        from colab_cli.client import TUN_ENDPOINT

        client = self._client(deadline)
        # The library's own request path (headers, authuser, XSSI, errors),
        # but schema=dict keeps every field; get_consumption_user_info()
        # would drop eligibleGpus.
        url = urljoin(client.colab_domain, f"{TUN_ENDPOINT}/ccu-info")
        return CcuInfo.from_json(client._issue_request(url, schema=dict))

    def _run_cell_sync(
        self,
        deadline: _Deadline,
        assignment: RuntimeAssignment,
        code: str,
        kernel_id: str,
        session_id: str,
        silent: bool,
    ) -> CellResult:
        runtime = self._runtime_factory(
            assignment.proxy_url, assignment.proxy_token, kernel_id or None, session_id or None
        )
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            runtime.stop()

        budget = max(0.0, deadline.at - time.monotonic())
        timer = threading.Timer(budget, expire)
        timer.daemon = True
        timer.start()
        kernel = None
        try:
            kernel = runtime.kernel_client
            if expired.is_set():
                return _cell(CELL_TIMEOUT, runtime, kernel, error="connecting used up the budget")
            reply = kernel.execute(code, silent=silent, store_history=not silent, timeout=budget)
        except TimeoutError:
            return _cell(CELL_TIMEOUT, runtime, kernel, error="no reply within the budget")
        except RuntimeError as exc:
            if kernel is None:
                raise
            # "Connection was lost.": the timer closed the websocket, or it dropped.
            status = CELL_TIMEOUT if expired.is_set() else CELL_DISCONNECTED
            return _cell(status, runtime, kernel, error=redact(exc))
        finally:
            timer.cancel()
            runtime.stop()
        return _cell_from_reply(reply, runtime, kernel)

    def _upload_sync(
        self,
        deadline: _Deadline,
        assignment: RuntimeAssignment,
        files: dict[str, str | os.PathLike[str]],
    ) -> None:
        import requests

        with requests.Session() as http:
            # Never the AuthorizedSession: the Google token must not reach the
            # runtime; no .netrc credentials or proxies from the environment.
            http.trust_env = False
            contents = _contents_client_class()(assignment, http, deadline)
            for remote, local in files.items():
                contents.upload(os.fspath(local), remote)


def quiet_library_loggers() -> None:
    """Hold the libraries' loggers at WARNING and scrub tokens from what they log."""
    for name in LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _SCRUBBED_LOGGERS:
        library_logger = logging.getLogger(name)
        if not any(isinstance(f, _ScrubFilter) for f in library_logger.filters):
            library_logger.addFilter(_ScrubFilter())


def adc_file_problem(path: str | os.PathLike[str]) -> str:
    """Why the Colab credential file at ``path`` cannot be used ("" when it looks usable)."""
    try:
        _read_adc(Path(path))
    except CredentialOrPolicyError as exc:
        return str(exc)
    return ""


class _ScrubFilter(logging.Filter):
    """Removes tokens from a record's message and traceback before any handler sees it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


class _Deadline:
    """When the current call must be done (``time.monotonic``)."""

    def __init__(self, at: float) -> None:
        self.at = at

    def timeouts(self) -> tuple[float, float]:
        """(connect, read) timeouts for the next request: the time left."""
        left = self.at - time.monotonic()
        if left <= 0:
            raise ControlPlaneTimeoutError("the call's time budget ran out")
        return (min(CONNECT_TIMEOUT_SECONDS, left), left)


class _DeadlineSession:
    """What colab_cli's ``Client`` calls ``request`` on.

    0.7.4 passes no timeout, so google-auth's default (120 s per request)
    would apply; here each request gets the time left in the call, also as
    ``max_allowed_time``, which bounds a token refresh as well.
    """

    def __init__(self, session: Any, deadline: _Deadline) -> None:
        self._session = session
        self._deadline = deadline

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        timeouts = self._deadline.timeouts()
        kwargs["timeout"] = timeouts
        kwargs["max_allowed_time"] = timeouts[1]
        return self._session.request(method, url, **kwargs)


@functools.cache
def _contents_client_class() -> type:
    """colab_cli's ``ContentsClient`` with a session and a timeout (0.7.4 sets neither)."""
    from colab_cli.contents import ContentsClient

    class BoundedContentsClient(ContentsClient):
        def __init__(self, assignment: RuntimeAssignment, http: Any, deadline: _Deadline) -> None:
            super().__init__(SimpleNamespace(url=assignment.proxy_url, token=assignment.proxy_token))
            self._http = http
            self._deadline = deadline

        def _request(
            self,
            method: str,
            path: str,
            params: dict[str, str] | None = None,
            json_data: dict[str, Any] | None = None,
        ) -> Any:
            # ContentsClient._request (0.7.4), through self._http with a timeout.
            url = f"{self.base_url}/api/contents/{quote(path.strip('/'), safe='/')}"
            query = {"authuser": "0", "colab-runtime-proxy-token": self.token, **(params or {})}
            response = self._http.request(
                method, url, params=query, json=json_data, timeout=self._deadline.timeouts()
            )
            if response.status_code == 404:
                raise FileNotFoundError(f"File or directory not found: {path}")
            response.raise_for_status()
            return None if method == "DELETE" else response.json()

    return BoundedContentsClient


def _runtime_kind(cpu_only: bool) -> dict[str, Any]:
    """assign's variant and accelerator: the 40 GB A100, or the CPU runtime of a dry run."""
    from colab_cli.client import Accelerator, Variant

    if cpu_only:
        return {"variant": Variant.DEFAULT, "accelerator": Accelerator.NONE}
    # shape stays None: "hm" would be the 80 GB box at a higher rate.
    return {"variant": Variant.GPU, "accelerator": Accelerator.A100}


def _authorized_session(credentials: Any) -> Any:
    from google.auth.transport.requests import AuthorizedSession

    return AuthorizedSession(credentials)


def _colab_runtime(url: str, token: str, kernel_id: str | None, session_id: str | None) -> Any:
    from colab_cli.runtime import ColabRuntime

    return ColabRuntime(url, token, kernel_id=kernel_id, session_id=session_id)


def _read_adc(path: Path) -> dict[str, Any]:
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # The type only: a decode error may quote the file.
        raise CredentialOrPolicyError(f"{path} cannot be read: {type(exc).__name__}") from None
    if not isinstance(info, dict) or info.get("type") != "authorized_user":
        raise CredentialOrPolicyError(f"{path} is not an authorized_user credential")
    missing = [name for name in _ADC_FIELDS if not info.get(name)]
    if missing:
        raise CredentialOrPolicyError(f"{path} lacks {', '.join(missing)}")
    return info


def _assignment(item: Any, *, reused: bool = False) -> RuntimeAssignment:
    proxy = item.runtime_proxy_info
    return RuntimeAssignment(
        endpoint=item.endpoint,
        proxy_url=proxy.url,
        proxy_token=proxy.token,
        token_expires_in=int(proxy.token_expires_in_seconds),
        accelerator=_enum_name(getattr(item, "accelerator", None)),
        variant=_enum_name(getattr(item, "variant", None)),
        machine_shape=_enum_name(getattr(item, "machine_shape", None)),
        reused=reused,
    )


def _listed(item: Any) -> RuntimeAssignment:
    """One runtime of the listing. Only what the bot uses must be there (the
    endpoint and its proxy); a kind the library does not know keeps its raw value."""
    from colab_cli.client import AssignmentVariant, Shape

    entry = item if isinstance(item, dict) else {}
    endpoint, proxy = entry.get("endpoint"), entry.get("runtimeProxyInfo")
    proxy = proxy if isinstance(proxy, dict) else {}
    url, token = proxy.get("url"), proxy.get("token")
    if not (endpoint and isinstance(endpoint, str) and isinstance(url, str) and isinstance(token, str)):
        # Never the entry itself in the message: it may hold a proxy token.
        raise ControlPlaneError("list_assignments: a runtime without an endpoint or a proxy")
    expires = _number(proxy.get("tokenExpiresInSeconds"))
    return RuntimeAssignment(
        endpoint=endpoint,
        proxy_url=url,
        proxy_token=token,
        token_expires_in=int(expires) if expires is not None and math.isfinite(expires) else 0,
        accelerator=_listed_name(entry.get("accelerator")),
        variant=_listed_name(entry.get("variant"), AssignmentVariant),
        machine_shape=_listed_name(entry.get("machineShape"), Shape),
    )


def _listed_name(value: Any, names: type[Enum] | None = None) -> str:
    """The library's name for a listed value, else the raw value ("" when absent)."""
    if value is None:
        return ""
    if names is not None:
        try:
            return names(value).name
        except ValueError:
            pass
    return str(value)


def _enum_name(value: Any) -> str:
    return "" if value is None else str(getattr(value, "name", value))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _names(value: Any) -> tuple[str, ...] | None:
    """A list of names; None when absent or not a list of strings (an unknown shape)."""
    if not isinstance(value, list):
        return None
    names = tuple(item for item in value if isinstance(item, str))
    return names if len(names) == len(value) else None


def _cell(status: str, runtime: Any, kernel: Any, *, error: str = "") -> CellResult:
    return CellResult(status, *_ids(runtime, kernel), error=error)


def _ids(runtime: Any, kernel: Any) -> tuple[str, str]:
    # The client's id wins: when the saved kernel is gone the client starts a
    # new one, while ColabRuntime keeps reporting the old id.
    kernel_id = getattr(kernel, "id", None) or getattr(runtime, "kernel_id", None) or ""
    return str(kernel_id), str(getattr(runtime, "session_id", None) or "")


def _cell_from_reply(reply: Any, runtime: Any, kernel: Any) -> CellResult:
    reply = reply if isinstance(reply, dict) else {}
    stdout: list[str] = []
    stderr: list[str] = []
    errors: list[str] = []
    for output in reply.get("outputs") or []:
        kind = output.get("output_type")
        if kind == "stream":
            (stdout if output.get("name") == "stdout" else stderr).append(str(output.get("text", "")))
        elif kind == "execute_result":
            text = (output.get("data") or {}).get("text/plain")
            if text:
                stdout.append(f"{text}\n")
        elif kind == "error":
            errors.append(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}")
    status = reply.get("status")
    if status != "ok" and not errors:
        errors.append(f"the kernel answered {status or 'nothing'}")
    return CellResult(
        CELL_OK if not errors else CELL_ERROR,
        *_ids(runtime, kernel),
        stdout="".join(stdout)[-OUTPUT_LIMIT:],
        stderr="".join(stderr)[-OUTPUT_LIMIT:],
        error="; ".join(errors)[:OUTPUT_LIMIT],
    )


def _status_of(exc: BaseException) -> int | None:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _body_of(exc: BaseException) -> str:
    body = getattr(exc, "response_body", None)
    if body is None:
        body = getattr(getattr(exc, "response", None), "text", None)
    return body[:300] if isinstance(body, str) else ""


def _translate(op: str, exc: BaseException) -> ControlPlaneError:
    """Classify a library failure (see the module docstring)."""
    if isinstance(exc, ControlPlaneError):
        return exc
    status = _status_of(exc)
    detail = f"{type(exc).__name__}: {exc}"
    body = _body_of(exc)
    if body:
        detail = f"{detail} | {body}"
    text = f"{op} failed: {detail}"
    # Only a loaded module can have raised its exceptions.
    google_auth = sys.modules.get("google.auth.exceptions")
    if google_auth is not None and isinstance(exc, google_auth.GoogleAuthError):
        # The answer decides, not ``retryable``: google-auth also calls a
        # token-endpoint blip non-retryable (a 502, an HTML page, a 200
        # without a token).
        if isinstance(exc, google_auth.RefreshError) and _CREDENTIAL_MARKERS.search(detail):
            return CredentialOrPolicyError(f"{op}: the Colab credential cannot be refreshed: {detail}")
        return TransientControlPlaneError(text)
    if _CREDENTIAL_MARKERS.search(detail):
        return CredentialOrPolicyError(text, status=status)
    if status is not None:
        return _by_status(op, status, text)
    requests_errors = sys.modules.get("requests.exceptions")
    if requests_errors is not None:
        if isinstance(exc, requests_errors.Timeout):
            return ControlPlaneTimeoutError(text)
        if isinstance(exc, requests_errors.RequestException):
            return TransientControlPlaneError(text)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return TransientControlPlaneError(text)
    return ControlPlaneError(text)


def _by_status(op: str, status: int, text: str) -> ControlPlaneError:
    if op == "assign":
        if status in (412, 503):
            return NoCapacityError(text, status=status)
        if status == 400:
            return CredentialOrPolicyError(f"{text} (no entitlement for the accelerator?)", status=status)
    if status in (401, 403):
        if op in _BACKEND_OPS:
            return CredentialOrPolicyError(text, status=status)
        # The runtime proxy refused its token; a fresh one comes with the next list.
        return TransientControlPlaneError(text, status=status)
    if status in (408, 429) or status >= 500:
        return TransientControlPlaneError(text, status=status)
    return ControlPlaneError(text, status=status)


def _logged(op: str, error: ControlPlaneError) -> ControlPlaneError:
    if isinstance(error, CredentialOrPolicyError):
        logger.error(f"Colab {op}: {error}")
    else:
        logger.warning(f"Colab {op}: {type(error).__name__}: {error}")
    return error
