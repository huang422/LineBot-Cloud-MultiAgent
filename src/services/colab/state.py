"""The Colab session's state document and where it is kept.

One document, Firestore ``{prefix}_colab/state``, describes the one Colab
session the bot may have. It survives Cloud Run cold starts: a VM can
outlive the process that started it, so the next process must know about
it. Everything in it can be shown to an operator; nothing in it is a
secret (the keys are derived from ``COLAB_SHARED_SECRET`` and ``nbh``, see
:mod:`.protocol`).

Statuses (the controller's state machine, ADR-0010)::

    IDLE ──demand──▶ ASSIGNING ──runtime──▶ BOOTING ──register(ready)+probe──▶ READY
      ▲                                        │ register(failed)/timeout      │ idle / age / off
      └──────────── released ◀── DRAINING ◀────┴───────────────────────────────┘
    COOLDOWN (after a refused assign, until cooldown_until)
    DISABLED (credentials or account refused, 3 boot failures in a row; until !colab on)

Writers go through :meth:`ColabStateStore.update` with a *change*
function: it gets the current state and returns the new one (or None, or
the same object, to write nothing). The Firestore store may call it more
than once when another writer got in first, so it must be pure. Every
store keeps the last state it read or wrote in :attr:`~ColabStateStore.cached`,
which is what the request path reads (no I/O per message).

Who writes what: the controller (:mod:`.controller`, the per-minute cron and
its ``colab`` tasks) owns the lifecycle (ASSIGNING → BOOTING, timeouts,
DRAINING → IDLE, COOLDOWN, DISABLED, compute units, kernel ids);
registration moves BOOTING → READY and any active status → DRAINING;
``!colab on|off`` sets ``desired`` (and ``manual_demand_at``); the event
pipeline records ``last_demand_at``. Every write bumps ``version``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Protocol

from src.utils.logger import logger

IDLE = "IDLE"
ASSIGNING = "ASSIGNING"
BOOTING = "BOOTING"
READY = "READY"
DRAINING = "DRAINING"
COOLDOWN = "COOLDOWN"
DISABLED = "DISABLED"
STATUSES: tuple[str, ...] = (IDLE, ASSIGNING, BOOTING, READY, DRAINING, COOLDOWN, DISABLED)
# A Colab runtime may exist in these.
ACTIVE_STATUSES: frozenset[str] = frozenset({ASSIGNING, BOOTING, READY, DRAINING})

# ``desired``: "auto" starts on demand (MODEL_PATH_PREFERENCE=colab) or on
# ``!colab on``; "off" (``!colab off``) keeps it down until ``!colab on``.
AUTO = "auto"
OFF = "off"
DESIRED: tuple[str, ...] = (AUTO, OFF)

STATE_DOCUMENT = "state"
# Firestore round trips (each get/update/create) give up after this long.
FIRESTORE_TIMEOUT_SECONDS = 10.0
# Tries of a read-modify-write that another writer keeps beating.
UPDATE_ATTEMPTS = 3
_TEXT_LIMIT = 300
# How many of our own past runtimes ``known_endpoints`` remembers.
KNOWN_ENDPOINTS_LIMIT = 20


class ColabStateError(Exception):
    """The state document could not be read or written."""


@dataclass(frozen=True)
class ColabState:
    """The session, as the state document stores it. Times are epoch seconds."""

    status: str = IDLE
    # When ``status`` last changed.
    since: float = 0.0
    desired: str = AUTO
    # The session's UUID (the controller picks it before assigning).
    nbh: str = ""
    # Colab's id for the runtime, its kernel and Jupyter session (controller).
    endpoint: str = ""
    kernel_id: str = ""
    session_id: str = ""
    assigned_at: float | None = None
    # https://<host>, set by a registration whose probe passed.
    tunnel_url: str = ""
    ready_at: float | None = None
    # The last health check (or readiness probe) that passed.
    last_health_at: float | None = None
    # The last triggered message that would have used Colab (at most once
    # a minute), the last ``!colab on`` (starts a session under either
    # MODEL_PATH_PREFERENCE), and the controller's last keep-alive cell.
    last_demand_at: float | None = None
    manual_demand_at: float | None = None
    last_heartbeat_at: float | None = None
    # How far the VM's bootstrap got (from its registrations).
    boot_step: str = ""
    # The Ollama release the launcher picked for this session (the latest,
    # or versions.json's fallback): probes and health checks want exactly
    # it. "" for a session launched before 2026-10-16 (then only the
    # minimum, protocol.OLLAMA_MIN_VERSION, applies).
    ollama_version: str = ""
    # What the VM reported: {"ollama", "cloudflared"} and {"name",
    # "vram_mib", "driver", ...} (protocol.py). What the readiness probe
    # saw: {model: digest} in /api/tags (shown beside the last verified
    # digests, never judged) and {model: {"size", "size_vram"}} in /api/ps;
    # a router missing from /api/ps or not wholly in VRAM is kept out of
    # the session (protocol.off_gpu).
    versions: dict[str, str] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    vram: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    # Compute units: balance and burn rate from Colab's ccu-info (when
    # read), and what today's (Asia/Taipei) sessions used.
    ccu_balance: float | None = None
    ccu_rate: float | None = None
    ccu_checked_at: float | None = None
    # Whether ccu-info lists the A100 as available (None: it did not say).
    a100_eligible: bool | None = None
    cu_day: str = ""
    cu_used_today: float = 0.0
    # Compute units are counted up to this time (while a runtime is ours).
    cu_accounted_at: float | None = None
    # Sessions started on ``cu_day``.
    starts_today: int = 0
    cooldown_until: float | None = None
    # Refused assigns in a row: the next cooldown's step (5, 10, 20, 60 min).
    cooldown_step: int = 0
    # Boot failures in a row (a READY session resets it); 3 → DISABLED.
    failures: int = 0
    # Failed health checks in a row while READY; 3 → DRAINING.
    health_failures: int = 0
    # The status's slow step queued as a ``colab`` task ("assign", "launch"
    # or "unassign"), when, and how many times in this status: a task that
    # never ran is queued again once its lease is over.
    action: str = ""
    action_at: float | None = None
    attempts: int = 0
    # Runtimes this bot was assigned (newest last), so a stale one of ours is
    # released even when COLAB_ACCOUNT_DEDICATED is off.
    known_endpoints: tuple[str, ...] = ()
    # Sessions given up while their assign had no answer (it timed out, or
    # its task was lost): a runtime may exist for such an nbh although its
    # endpoint was never learned. A reconcile looks each one up (GET only)
    # and releases what it finds; ``unanswered_at`` is when the newest came.
    unanswered_nbhs: tuple[str, ...] = ()
    unanswered_at: float | None = None
    # The Asia/Taipei day the last full reconcile (list_assignments) was queued.
    reconciled_day: str = ""
    # Why a wanted session is not being started (balance, cap, cooldown…).
    hold_reason: str = ""
    last_error: str = ""
    disabled_reason: str = ""
    # Bumped by every write (the stores do it).
    version: int = 0

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def with_status(self, status: str, now: float, **changes: Any) -> ColabState:
        """A copy in *status* (``since`` moves only when it changes) with *changes*."""
        if status not in STATUSES:
            raise ValueError(f"unknown Colab status {status!r}")
        since = now if status != self.status else self.since
        return replace(self, status=status, since=since, **changes)

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["known_endpoints"] = list(self.known_endpoints)
        doc["unanswered_nbhs"] = list(self.unanswered_nbhs)
        return doc

    @classmethod
    def from_doc(cls, doc: dict[str, Any] | None) -> ColabState:
        """The state a stored document describes; unknown or broken fields fall back."""
        if not isinstance(doc, dict):
            return cls()
        default = cls()
        values: dict[str, Any] = {}
        for spec in fields(cls):
            if spec.name not in doc:
                continue
            value = _coerce(doc[spec.name], getattr(default, spec.name), spec.type)
            if value is not _INVALID:
                values[spec.name] = value
        state = cls(**values)
        if state.status not in STATUSES:
            logger.warning(f"Colab state: unknown status {state.status!r} read as IDLE")
            state = replace(state, status=IDLE)
        if state.desired not in DESIRED:
            state = replace(state, desired=AUTO)
        return state


_INVALID = object()


def _coerce(value: Any, default: Any, annotation: Any) -> Any:
    """*value* in the shape of the field, or ``_INVALID``."""
    optional = "None" in str(annotation)
    if value is None:
        return None if optional else _INVALID
    if isinstance(default, dict) or "dict" in str(annotation):
        return dict(value) if isinstance(value, dict) else _INVALID
    if isinstance(default, tuple):
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return tuple(value)[-KNOWN_ENDPOINTS_LIMIT:]
        return _INVALID
    if isinstance(default, bool) or "bool" in str(annotation):
        return value if isinstance(value, bool) else _INVALID
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else _INVALID
    if isinstance(default, float) or "float" in str(annotation):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
        return _INVALID
    if isinstance(default, str):
        return str(value)[:_TEXT_LIMIT * 4] if isinstance(value, str) else _INVALID
    return _INVALID


def clip(text: str, limit: int = _TEXT_LIMIT) -> str:
    """*text* on one line, at most *limit* characters (for ``last_error``)."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


Change = Callable[[ColabState], "ColabState | None"]


class ColabStateStore(Protocol):
    """Where the state document lives. See the module docstring."""

    @property
    def cached(self) -> ColabState | None:
        """The last state read or written by this process; None before the first."""

    async def load(self) -> ColabState:
        """Read the stored state (a missing document is the default IDLE state).

        Raises:
            ColabStateError: The store could not be read.
        """

    async def update(self, change: Change) -> ColabState:
        """Apply *change* to the stored state atomically; returns the result.

        Raises:
            ColabStateError: The store could not be read or written.
        """


class InMemoryColabStateStore:
    """Process-local state: tests, and a bot without Firestore (Colab stays off)."""

    persistent = False

    def __init__(self, initial: ColabState | None = None) -> None:
        self._state = initial or ColabState()
        self._cached: ColabState | None = None
        self._lock = asyncio.Lock()
        self.writes = 0

    @property
    def cached(self) -> ColabState | None:
        return self._cached

    async def load(self) -> ColabState:
        self._cached = self._state
        return self._state

    async def update(self, change: Change) -> ColabState:
        async with self._lock:
            new = change(self._state)
            if new is not None and new != self._state:
                self._state = replace(new, version=self._state.version + 1)
                self.writes += 1
            self._cached = self._state
            return self._state


class FirestoreColabStateStore:
    """``{prefix}_colab/state`` in Firestore, written with optimistic concurrency.

    An update reads the document, applies the change and writes it back
    only if the document is unchanged since the read (``last_update_time``
    precondition; ``create`` when it did not exist), retrying the whole
    read-modify-write up to :data:`UPDATE_ATTEMPTS` times; the written
    document carries ``version`` + 1 of the one it replaced. The client is
    synchronous, so every call runs in a worker thread, and it is built on
    first use so startup needs no credentials.
    """

    persistent = True

    def __init__(
        self,
        *,
        project_id: str = "",
        database: str = "(default)",
        collection_prefix: str = "linebot",
        client: Any | None = None,
    ) -> None:
        prefix = (collection_prefix or "linebot").strip().rstrip("_") or "linebot"
        self.collection = f"{prefix}_colab"
        self._project_id = project_id
        self._database = database
        self._client = client
        self._cached: ColabState | None = None

    @property
    def cached(self) -> ColabState | None:
        return self._cached

    def _document(self):
        if self._client is None:
            from google.cloud import firestore  # type: ignore

            kwargs: dict[str, Any] = {}
            if self._project_id:
                kwargs["project"] = self._project_id
            if self._database and self._database != "(default)":
                kwargs["database"] = self._database
            self._client = firestore.Client(**kwargs)
        return self._client.collection(self.collection).document(STATE_DOCUMENT)

    async def load(self) -> ColabState:
        try:
            snapshot = await asyncio.to_thread(
                lambda: self._document().get(timeout=FIRESTORE_TIMEOUT_SECONDS)
            )
        except Exception as exc:
            raise ColabStateError(f"reading {self.collection}/{STATE_DOCUMENT} failed: {type(exc).__name__}: {exc}") from exc
        state = ColabState.from_doc(snapshot.to_dict() if snapshot.exists else None)
        self._cached = state
        return state

    async def update(self, change: Change) -> ColabState:
        try:
            state = await asyncio.to_thread(self._update_sync, change)
        except ColabStateError:
            raise
        except Exception as exc:
            raise ColabStateError(f"writing {self.collection}/{STATE_DOCUMENT} failed: {type(exc).__name__}: {exc}") from exc
        self._cached = state
        return state

    def _update_sync(self, change: Change) -> ColabState:
        from google.api_core import exceptions as gexc  # type: ignore

        document = self._document()
        for _ in range(UPDATE_ATTEMPTS):
            snapshot = document.get(timeout=FIRESTORE_TIMEOUT_SECONDS)
            current = ColabState.from_doc(snapshot.to_dict() if snapshot.exists else None)
            new = change(current)
            if new is None or new == current:
                return current
            new = replace(new, version=current.version + 1)
            try:
                if snapshot.exists:
                    document.update(
                        new.to_doc(),
                        option=self._client.write_option(last_update_time=snapshot.update_time),
                        timeout=FIRESTORE_TIMEOUT_SECONDS,
                    )
                else:
                    document.create(new.to_doc(), timeout=FIRESTORE_TIMEOUT_SECONDS)
            except (gexc.FailedPrecondition, gexc.Conflict):
                # Another writer got in between the read and the write.
                continue
            return new
        raise ColabStateError(
            f"{self.collection}/{STATE_DOCUMENT} kept changing; gave up after {UPDATE_ATTEMPTS} tries"
        )


def build_state_store(settings: Any) -> ColabStateStore:
    """Firestore when FIRESTORE_ENABLED (which COLAB_ENABLED requires), else in memory."""
    if not getattr(settings, "firestore_enabled", False):
        return InMemoryColabStateStore()
    return FirestoreColabStateStore(
        project_id=(settings.firestore_project_id or "").strip(),
        database=(settings.firestore_database or "(default)").strip() or "(default)",
        collection_prefix=settings.firestore_collection_prefix,
    )
