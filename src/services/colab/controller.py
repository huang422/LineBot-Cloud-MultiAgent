"""The Colab session's lifecycle: a pure state machine and the service that runs it.

Decisions are pure functions of the state document, what was just observed
and the settings; they return the next state and the ``colab`` tasks to
queue (:class:`Decision`). The per-minute cron (:meth:`ColabController.tick`)
and the ``colab`` Cloud Tasks (:meth:`ColabController.run_action`) apply
them through the state store, which may re-run a decision when another
writer got in first, so they never do I/O themselves. ADR-0010.

Transitions (``decide`` unless noted; the registrar owns BOOTING → READY
and the VM's ``failed``/``stopping``, :mod:`.admin` owns ``!colab on|off``):

- IDLE → ASSIGNING when a session is wanted: ``desired`` is auto and
  either a triggered message (MODEL_PATH_PREFERENCE=colab only) or
  ``!colab on`` (either preference) came within START_DEMAND_SECONDS; and
  nothing holds it (``hold_reason``): a ccu-info read within
  CCU_FRESH_SECONDS (else a ``reconcile`` is queued to read it), and room
  for one start nobody uses (``start_cost``: a boot as long as
  COLAB_BOOT_TIMEOUT_SECONDS plus one idle TTL) above the balance floor and
  under today's cap, and the A100 eligible when ccu-info says. The new
  ``nbh`` is written first, then ``assign`` is queued with it.
- ASSIGNING: the ``assign`` task's runtime → BOOTING and ``launch``
  (:func:`after_assign`); a time-out or a transient error is retried with
  the same ``nbh`` (Colab hands back the runtime already assigned to it)
  up to MAX_ASSIGN_ATTEMPTS times, then COOLDOWN; HTTP 412 or 503 →
  COOLDOWN for 5, 10, 20, then 60 minutes; a dead credential or a refused
  account → DISABLED. A session given up while its assign had no answer
  (that COOLDOWN, a drain that waited in vain, a lost task), or refused
  only on a retry, keeps its ``nbh`` in ``unanswered_nbhs``: a timed-out
  request can still leave a runtime. A reconcile
  UNANSWERED_LOOKUP_DELAY_SECONDS later looks each one up
  (``find_assignment``, assign's GET only) and releases what it finds; a
  lookup that fails is tried again every minute for
  UNANSWERED_RETRY_SECONDS, then only by the reconciles that run anyway.
- BOOTING: ``launch`` starts the VM side (:mod:`.launcher`); a launch that
  fails, a ``failed`` from the VM, a runtime that vanishes or no READY
  within COLAB_BOOT_TIMEOUT_SECONDS → DRAINING (or IDLE) and one more boot
  failure; the third in a row ends in DISABLED instead of IDLE. A balance
  under the floor or the daily cap drains it too (not a boot failure). A
  runtime not listed yet within RECLAIM_GRACE_SECONDS of its assign is
  launched again, not taken for gone.
- READY → DRAINING on ``!colab off``, MAX_HEALTH_FAILURES failed health
  checks in a row, no use for COLAB_IDLE_TTL_SECONDS (counted from the
  later of the last triggered message, the last ``!colab on`` and READY),
  the session age limit, a balance under the floor or the daily cap.
- DRAINING → ``unassign`` → IDLE (``unassign`` answering 404 is fine). An
  unassign that fails, whatever Colab answers (401/403 included: it may
  refuse to release a runtime the VM already released), never disables
  the tier; the next tick queues it again with a ``reconcile``, whose
  listing ends the drain once the runtime is gone and disables the tier
  when it is refused as well.
- Any active status whose runtime is missing from ``list_assignments``
  (Colab reclaimed it, or the VM released itself) → IDLE; before READY
  that is a boot failure as well, since the VM's ``failed`` may never
  have arrived.
- COOLDOWN → IDLE once ``cooldown_until`` passes. DISABLED stays until
  ``!colab on`` and makes no Colab calls but unassigns: a move to DISABLED
  first queues the release of the runtime the session holds.

While BOOTING or READY and in use, a ``heartbeat`` every
COLAB_HEARTBEAT_SECONDS runs a tiny cell in the saved kernel: Colab keeps a
runtime only while its kernel is active (tunnel traffic does not count).
Every tick while BOOTING or READY (or DRAINING after an unassign that did
not end it), and once a day otherwise, queues a ``reconcile``
(list_assignments, ccu-info, the health check). Runtimes on
the account that are not the current session are released when they are
ours (``known_endpoints``) or COLAB_ACCOUNT_DEDICATED is on, and otherwise
only logged at ERROR, once per process.

Compute units: while a runtime is ours, ``cu_used_today`` grows by
ccu-info's consumption rate (read after the assign; before that an
estimate) times the time held, per Asia/Taipei day.

The tick reads and writes the state within TICK_STATE_TIMEOUT_SECONDS and
queues tasks; it never calls Colab. Slow work (Colab calls take seconds to
minutes) runs in the ``colab`` tasks, off the event loop in the facade's
threads, and never on the request path. ``submit`` only queues
(:func:`src.services.task_queue.enqueue_only`): a task that cannot be
queued raises, is logged, and the lease or the next tick queues it again;
it never runs inside the cron request or the task that queued it.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.services.colab.control_plane import (
    CcuInfo,
    CellResult,
    ControlPlaneError,
    CredentialOrPolicyError,
    NoCapacityError,
    RuntimeAssignment,
)
from src.services.colab.launcher import LaunchError, LaunchResult
from src.services.colab.state import (
    ASSIGNING,
    AUTO,
    BOOTING,
    COOLDOWN,
    DISABLED,
    DRAINING,
    IDLE,
    KNOWN_ENDPOINTS_LIMIT,
    OFF,
    READY,
    ColabState,
    ColabStateError,
    clip,
)
from src.utils.logger import logger

TASK_KIND = "colab"
ACTIONS: tuple[str, ...] = ("assign", "launch", "heartbeat", "unassign", "reconcile")

MAX_ASSIGN_ATTEMPTS = 3
MAX_LAUNCH_ATTEMPTS = 2
MAX_BOOT_FAILURES = 3
MAX_HEALTH_FAILURES = 3
COOLDOWN_STEPS_SECONDS: tuple[float, ...] = (300.0, 600.0, 1200.0, 3600.0)
# A queued step that has not reported back after its lease is queued again
# (a Cloud Task is attempted once; the facade's budgets fit inside these).
LEASE_SECONDS: dict[str, float] = {"assign": 180.0, "launch": 240.0, "unassign": 180.0}
# A session starts only for demand this recent (booting takes 4-8 minutes).
START_DEMAND_SECONDS = 300.0
# A start reads ccu-info again when the last read is older than this.
CCU_FRESH_SECONDS = 300.0
# A runtime this young missing from list_assignments is not "reclaimed" yet.
RECLAIM_GRACE_SECONDS = 60.0
# An nbh given up without an assign answer is looked up this long after
# (Colab may still be finishing the request that timed out), and at most
# this many wait for it.
UNANSWERED_LOOKUP_DELAY_SECONDS = 60.0
UNANSWERED_LIMIT = 5
# A lookup that keeps failing queues a reconcile of its own every minute only
# this long after the newest nbh came; then it rides on the reconciles that
# run anyway (once a day, every tick of a session).
UNANSWERED_RETRY_SECONDS = 1800.0
# The cron comes once a minute: a heartbeat due within this is due now.
HEARTBEAT_SLACK_SECONDS = 5.0
# Compute units per hour until ccu-info reports the rate with our runtime
# assigned (the 40 GB A100; the CPU runtime of a dry run).
ESTIMATED_A100_CU_PER_HOUR = 5.37
ESTIMATED_CPU_CU_PER_HOUR = 0.1
TICK_STATE_TIMEOUT_SECONDS = 5.0
HEARTBEAT_BUDGET_SECONDS = 30.0
# ``boot_step`` from the assign until the launch reports (then "launched",
# then the VM's own steps from the heartbeat).
ASSIGNED_STEP = "assigned"
HEARTBEAT_MARKER = "COLAB_VM_STATE "
# The keep-alive cell: prints the VM's /content/state.json (at most 300
# characters, never a secret) so the controller can follow the boot.
HEARTBEAT_CODE = (
    "import pathlib\n"
    "_state = pathlib.Path('/content/state.json')\n"
    f"print({HEARTBEAT_MARKER!r} + (_state.read_text()[:300].strip() if _state.exists() else '{{}}'))\n"
    "del _state\n"
)
DAY_TIMEZONE = ZoneInfo("Asia/Taipei")

Task = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class ControllerConfig:
    """The settings the decisions read (see ``Settings`` for each one)."""

    # MODEL_PATH_PREFERENCE=colab: triggered messages start sessions.
    auto_start: bool = True
    idle_ttl_seconds: float = 900.0
    max_session_seconds: float = 39600.0
    boot_timeout_seconds: float = 900.0
    heartbeat_seconds: float = 300.0
    min_cu_balance: float = 10.0
    daily_cu_cap: float = 30.0
    account_dedicated: bool = False
    dry_run: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> ControllerConfig:
        return cls(
            auto_start=settings.model_path_preference == "colab",
            idle_ttl_seconds=float(settings.colab_idle_ttl_seconds),
            max_session_seconds=float(settings.colab_max_session_seconds),
            boot_timeout_seconds=float(settings.colab_boot_timeout_seconds),
            heartbeat_seconds=float(settings.colab_heartbeat_seconds),
            min_cu_balance=float(settings.colab_min_cu_balance),
            daily_cu_cap=float(settings.colab_daily_cu_cap),
            account_dedicated=bool(settings.colab_account_dedicated),
            dry_run=bool(settings.colab_dry_run),
        )


@dataclass(frozen=True)
class Observations:
    """What one pass saw. None: not observed in this pass."""

    now: float
    # Used only when this pass starts a session.
    new_nbh: str = ""
    ccu: CcuInfo | None = None
    assignments: tuple[RuntimeAssignment, ...] | None = None
    # The health check through the tunnel (READY only).
    healthy: bool | None = None
    # find_assignment for ``unanswered_nbhs``: nbh → the endpoint it has
    # ("" = none). A lookup that failed is left out (tried again later).
    lookups: tuple[tuple[str, str], ...] = ()
    # The cron's pass: it queues the periodic ``reconcile``.
    tick: bool = False


@dataclass(frozen=True)
class Decision:
    state: ColabState
    tasks: tuple[Task, ...] = ()
    # Runtimes on the account that are not ours, left alone.
    foreign: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def taipei_day(now: float) -> str:
    return datetime.fromtimestamp(now, DAY_TIMEZONE).date().isoformat()


def _day_start(now: float) -> float:
    moment = datetime.fromtimestamp(now, DAY_TIMEZONE)
    return moment.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _last_use(state: ColabState) -> float:
    """When the session was last wanted, or became usable (the idle clock)."""
    marks = (
        state.last_demand_at,
        state.manual_demand_at,
        state.ready_at if state.ready_at is not None else state.assigned_at,
    )
    return max((mark for mark in marks if mark is not None), default=state.since)


def _remember(known: tuple[str, ...], endpoint: str) -> tuple[str, ...]:
    if not endpoint or endpoint in known:
        return known
    return (*known, endpoint)[-KNOWN_ENDPOINTS_LIMIT:]


def lookups_due(state: ColabState, now: float) -> tuple[str, ...]:
    """The unanswered nbhs a reconcile at *now* should look up (none too early)."""
    waiting = tuple(nbh for nbh in state.unanswered_nbhs if nbh != state.nbh)
    if not waiting:
        return ()
    since = state.unanswered_at
    if since is not None and now - since < UNANSWERED_LOOKUP_DELAY_SECONDS:
        return ()
    return waiting


def _no_session() -> dict[str, Any]:
    """Everything a session holds, cleared (fresh maps each time)."""
    return {
        "nbh": "",
        "endpoint": "",
        "kernel_id": "",
        "session_id": "",
        "assigned_at": None,
        "tunnel_url": "",
        "ready_at": None,
        "last_health_at": None,
        "last_heartbeat_at": None,
        "boot_step": "",
        "versions": {},
        "digests": {},
        "vram": {},
        "gpu": {},
        "health_failures": 0,
        "cu_accounted_at": None,
    }


class _Pass:
    """One decision in the making: the state so far, the tasks, the notes."""

    def __init__(self, state: ColabState, obs: Observations, cfg: ControllerConfig) -> None:
        self.s = state
        self.obs = obs
        self.cfg = cfg
        self.now = obs.now
        self.tasks: list[Task] = []
        self.foreign: list[str] = []
        self.notes: list[str] = []
        self._roll_day()
        self._account_cu()

    def decision(self) -> Decision:
        return Decision(self.s, tuple(self.tasks), tuple(self.foreign), tuple(self.notes))

    def set(self, **changes: Any) -> None:
        self.s = replace(self.s, **changes)

    def move(self, status: str, *, keep_action: bool = False, **changes: Any) -> None:
        """To *status*; a new status starts its own step count."""
        if status != self.s.status:
            self.notes.append(f"{self.s.status} → {status}")
            if not keep_action:
                changes = {"action": "", "action_at": None, "attempts": 0, **changes}
        self.s = self.s.with_status(status, self.now, **changes)

    def queue(self, action: str, **payload: Any) -> None:
        """Queue the status's own step and start its lease."""
        attempts = self.s.attempts + 1 if self.s.action == action else 1
        self.set(action=action, action_at=self.now, attempts=attempts)
        self.tasks.append((action, payload))

    def in_flight(self, action: str) -> bool:
        s = self.s
        return (
            s.action == action
            and s.action_at is not None
            and self.now - s.action_at < LEASE_SECONDS[action]
        )

    # ── compute units ──────────────────────────────────────────

    def _roll_day(self) -> None:
        today = taipei_day(self.now)
        if self.s.cu_day == today:
            return
        accounted = self.s.cu_accounted_at
        if accounted is not None:
            accounted = max(accounted, _day_start(self.now))
        self.set(cu_day=today, cu_used_today=0.0, starts_today=0, cu_accounted_at=accounted)

    def _account_cu(self) -> None:
        s = self.s
        if s.cu_accounted_at is None or not s.endpoint or not s.active:
            return
        held = max(0.0, self.now - s.cu_accounted_at)
        self.set(cu_used_today=s.cu_used_today + self._rate() * held / 3600.0, cu_accounted_at=self.now)

    def _rate(self) -> float:
        s = self.s
        measured = (
            s.ccu_rate is not None
            and s.ccu_rate > 0
            and s.ccu_checked_at is not None
            and s.assigned_at is not None
            and s.ccu_checked_at > s.assigned_at
        )
        if measured:
            return float(s.ccu_rate)
        return ESTIMATED_CPU_CU_PER_HOUR if self.cfg.dry_run else ESTIMATED_A100_CU_PER_HOUR

    # ── observations ───────────────────────────────────────────

    def take_ccu(self, ccu: CcuInfo) -> None:
        self.set(
            ccu_balance=ccu.current_balance,
            ccu_rate=ccu.consumption_rate_hourly,
            ccu_checked_at=self.now,
            a100_eligible=ccu.gpu_eligible("A100"),
        )

    def take_lookups(self, lookups: tuple[tuple[str, str], ...]) -> None:
        """Runtimes left by assigns that never answered: ours, so released."""
        s = self.s
        waiting = list(s.unanswered_nbhs)
        known = s.known_endpoints
        for nbh, endpoint in lookups:
            if nbh not in waiting or nbh == s.nbh:
                continue
            waiting.remove(nbh)
            if endpoint:
                known = _remember(known, endpoint)
                self.release_orphan(endpoint)
                self.notes.append("found the runtime of an assign that never answered; released")
        self.set(
            unanswered_nbhs=tuple(waiting),
            unanswered_at=s.unanswered_at if waiting else None,
            known_endpoints=known,
        )

    def take_assignments(self, assignments: tuple[RuntimeAssignment, ...]) -> None:
        s = self.s
        listed = {item.endpoint for item in assignments}
        current = s.endpoint if s.active else ""
        young = s.assigned_at is not None and self.now - s.assigned_at < RECLAIM_GRACE_SECONDS
        if current and current not in listed and not young:
            self.lost("the runtime is gone (reclaimed by Colab or released by the VM)")
        if s.status == ASSIGNING:
            # An assign in flight may have created a runtime we cannot name yet.
            return
        for endpoint in sorted(listed - {current}):
            if endpoint in self.s.known_endpoints or self.cfg.account_dedicated:
                self.release_orphan(endpoint)
            else:
                self.foreign.append(endpoint)

    def release_orphan(self, endpoint: str) -> None:
        if not any(payload.get("endpoint") == endpoint for _, payload in self.tasks):
            self.tasks.append(("unassign", {"endpoint": endpoint, "orphan": True}))

    # ── status changes ─────────────────────────────────────────

    def ended(self, *, answered: str = "") -> dict[str, Any]:
        """The fields that end the session (:func:`_no_session`).

        A session with an nbh but no endpoint may still own a runtime: its
        assign timed out or never reported, and a timed-out request can
        leave one (googlecolab/google-colab-cli discussion #141). Its nbh
        joins ``unanswered_nbhs`` for a lookup, unless *answered* is that
        nbh (Colab refused its first and only try, so there is none).
        """
        s = self.s
        changes = _no_session()
        if s.nbh and not s.endpoint and s.nbh != answered:
            others = tuple(nbh for nbh in s.unanswered_nbhs if nbh != s.nbh)
            changes["unanswered_nbhs"] = (*others, s.nbh)[-UNANSWERED_LIMIT:]
            changes["unanswered_at"] = self.now
        return changes

    def released(self, *, answered: str = "") -> None:
        """The runtime is gone: IDLE, or DISABLED after MAX_BOOT_FAILURES."""
        s = self.s
        if s.failures >= MAX_BOOT_FAILURES:
            reason = f"連續 {s.failures} 次開機失敗（最後：{s.last_error or '?'}）"
            self.move(DISABLED, disabled_reason=clip(reason), **self.ended(answered=answered))
        else:
            self.move(IDLE, **self.ended(answered=answered))

    def lost(self, reason: str) -> None:
        """The runtime vanished: Colab reclaimed it, or the VM released itself.

        Before READY this is one more boot failure. The VM's own ``failed``
        may never have arrived (it is sent once, and not at all when the
        env file cannot be read), and a tier that keeps losing runtimes
        must end in DISABLED instead of assigning A100 after A100.
        """
        s = self.s
        changes: dict[str, Any] = {}
        if s.status != DRAINING:
            changes["last_error"] = clip(reason)
        if s.status in (ASSIGNING, BOOTING):
            changes["failures"] = s.failures + 1
        self.set(**changes)
        self.released()

    def drain(self, reason: str, *, boot_failure: bool = False) -> None:
        failures = self.s.failures + (1 if boot_failure else 0)
        self.move(DRAINING, last_error=clip(reason), failures=failures)
        self.draining()

    def cool_down(self, reason: str, *, answered: str = "") -> None:
        step = self.s.cooldown_step
        wait = COOLDOWN_STEPS_SECONDS[min(step, len(COOLDOWN_STEPS_SECONDS) - 1)]
        self.move(
            COOLDOWN,
            cooldown_until=self.now + wait,
            cooldown_step=step + 1,
            last_error=clip(reason),
            **self.ended(answered=answered),
        )

    def disable(self, reason: str, *, answered: str = "") -> None:
        """A human must act (credential, account): no more Colab calls until !colab on,
        but the runtime the session holds is released first (the endpoint goes
        with the session, and the A100 would bill until the VM's watchdog)."""
        if self.s.endpoint:
            self.tasks.append(("unassign", {"endpoint": self.s.endpoint}))
        self.move(
            DISABLED,
            disabled_reason=clip(f"Colab 憑證或帳號被拒：{reason}"),
            last_error=clip(reason),
            **self.ended(answered=answered),
        )

    def queue_reconcile(self) -> None:
        self.tasks.append(("reconcile", {}))
        self.set(reconciled_day=taipei_day(self.now))

    def periodic_reconcile(self) -> None:
        """Outside a session: once a day, and when an unanswered nbh is due its
        lookup (for UNANSWERED_RETRY_SECONDS: a lookup may keep failing)."""
        if not self.obs.tick:
            return
        since = self.s.unanswered_at
        retrying = since is not None and self.now - since <= UNANSWERED_RETRY_SECONDS
        if self.s.reconciled_day != taipei_day(self.now) or (retrying and lookups_due(self.s, self.now)):
            self.queue_reconcile()

    def heartbeat_if_due(self) -> None:
        s = self.s
        if not s.kernel_id or not s.endpoint:
            return
        if self.now - _last_use(s) > self.cfg.idle_ttl_seconds:
            return
        last = s.last_heartbeat_at
        if last is not None and self.now - last < self.cfg.heartbeat_seconds - HEARTBEAT_SLACK_SECONDS:
            return
        self.set(last_heartbeat_at=self.now)
        self.tasks.append(("heartbeat", {"nbh": s.nbh, "endpoint": s.endpoint}))

    # ── per status ─────────────────────────────────────────────

    def step(self) -> None:
        handler = {
            IDLE: self.idle,
            ASSIGNING: self.assigning,
            BOOTING: self.booting,
            READY: self.ready,
            DRAINING: self.draining,
            COOLDOWN: self.cooldown,
            DISABLED: lambda: None,
        }[self.s.status]
        handler()

    def wanted(self) -> bool:
        s = self.s
        if s.desired != AUTO:
            return False
        fresh = [s.manual_demand_at]
        if self.cfg.auto_start:
            fresh.append(s.last_demand_at)
        return any(mark is not None and self.now - mark <= START_DEMAND_SECONDS for mark in fresh)

    def start_cost(self) -> float:
        """Compute units a start may cost with nobody using it: a boot up to
        COLAB_BOOT_TIMEOUT_SECONDS, then one idle TTL, at the expected rate."""
        return self._rate() * (self.cfg.boot_timeout_seconds + self.cfg.idle_ttl_seconds) / 3600.0

    def hold(self) -> str | None:
        """Why a wanted start must wait ("" = nothing holds it; None = read ccu-info first).

        A start must leave room for :meth:`start_cost` under the daily cap
        and above the balance floor; otherwise the session would be drained
        the moment it is READY, having answered nothing.
        """
        s = self.s
        cfg = self.cfg
        fresh = s.ccu_checked_at is not None and self.now - s.ccu_checked_at <= CCU_FRESH_SECONDS
        if not fresh:
            return None if self.obs.tick else "讀不到 CU 餘額（ccu-info）"
        if s.ccu_balance is None:
            return "ccu-info 沒有回報 CU 餘額"
        cost = self.start_cost()
        if s.ccu_balance < cfg.min_cu_balance:
            return f"CU 餘額 {s.ccu_balance:.1f} 低於下限 {cfg.min_cu_balance:g}"
        if s.ccu_balance - cost < cfg.min_cu_balance:
            return (
                f"CU 餘額 {s.ccu_balance:.1f} 扣掉一次開機加閒置（約 {cost:.1f} CU）"
                f"會低於下限 {cfg.min_cu_balance:g}"
            )
        if s.cu_used_today >= cfg.daily_cu_cap:
            return f"今日已用 {s.cu_used_today:.1f} CU，達每日上限 {cfg.daily_cu_cap:g}"
        if s.cu_used_today + cost > cfg.daily_cu_cap:
            return (
                f"今日已用 {s.cu_used_today:.1f} CU，再開一次（約 {cost:.1f} CU）"
                f"會超過每日上限 {cfg.daily_cu_cap:g}"
            )
        if not cfg.dry_run and s.a100_eligible is False:
            return "Colab 目前不提供 A100 給這個帳號（ccu-info）"
        return ""

    def budget_spent(self) -> str:
        """Why a session that holds a runtime must be released for cost ("" = it need not)."""
        s = self.s
        if s.ccu_balance is not None and s.ccu_balance < self.cfg.min_cu_balance:
            return f"CU balance {s.ccu_balance:.1f} below {self.cfg.min_cu_balance:g}"
        if s.cu_used_today >= self.cfg.daily_cu_cap:
            return f"daily CU cap reached ({s.cu_used_today:.1f} of {self.cfg.daily_cu_cap:g})"
        return ""

    def idle(self) -> None:
        if not self.wanted():
            if self.s.hold_reason:
                self.set(hold_reason="")
            self.periodic_reconcile()
            return
        reason = self.hold()
        if reason is None:
            self.queue_reconcile()
            return
        if reason:
            if reason != self.s.hold_reason:
                self.notes.append(f"start held: {reason}")
            self.set(hold_reason=reason)
            self.periodic_reconcile()
            return
        if not self.obs.new_nbh:
            return
        self.move(
            ASSIGNING,
            hold_reason="",
            starts_today=self.s.starts_today + 1,
            **{**_no_session(), "nbh": self.obs.new_nbh},
        )
        self.queue("assign", nbh=self.obs.new_nbh)

    def assigning(self) -> None:
        s = self.s
        if s.desired == OFF:
            self.move(DRAINING, keep_action=True, last_error="!colab off")
            self.draining()
            return
        if self.in_flight("assign"):
            return
        if s.action == "assign" and s.attempts >= MAX_ASSIGN_ATTEMPTS:
            self.cool_down(f"assign gave no runtime after {s.attempts} tries ({s.last_error or 'no answer'})")
            return
        self.queue("assign", nbh=s.nbh)

    def booting(self) -> None:
        s = self.s
        if s.desired == OFF:
            self.drain("!colab off")
            return
        spent = self.budget_spent()
        if spent:
            # Not a boot failure; READY would release it at once anyway.
            self.drain(spent)
            return
        started = s.assigned_at if s.assigned_at is not None else s.since
        if self.now - started > self.cfg.boot_timeout_seconds:
            self.drain(
                f"not READY within {self.cfg.boot_timeout_seconds:.0f}s (last step: {s.boot_step or '?'})",
                boot_failure=True,
            )
            return
        if s.boot_step == ASSIGNED_STEP and not self.in_flight("launch"):
            if s.action == "launch" and s.attempts >= MAX_LAUNCH_ATTEMPTS:
                self.drain(f"launch failed {s.attempts} times ({s.last_error or 'no answer'})", boot_failure=True)
                return
            self.queue("launch", nbh=s.nbh, endpoint=s.endpoint)
        self.heartbeat_if_due()
        if self.obs.tick:
            self.queue_reconcile()

    def ready(self) -> None:
        s = self.s
        if s.desired == OFF:
            self.drain("!colab off")
            return
        if self.obs.healthy is True:
            self.set(health_failures=0, last_health_at=self.now)
        elif self.obs.healthy is False:
            self.set(health_failures=s.health_failures + 1)
            if self.s.health_failures >= MAX_HEALTH_FAILURES:
                self.drain(f"{self.s.health_failures} health checks in a row failed")
                return
        s = self.s
        idle = self.now - _last_use(s)
        age = self.now - (s.assigned_at if s.assigned_at is not None else s.since)
        spent = self.budget_spent()
        if idle > self.cfg.idle_ttl_seconds:
            self.drain(f"idle for {idle:.0f}s")
        elif age > self.cfg.max_session_seconds:
            self.drain(f"session age {age:.0f}s reached the limit")
        elif spent:
            self.drain(spent)
        else:
            self.heartbeat_if_due()
            if self.obs.tick:
                self.queue_reconcile()

    def draining(self) -> None:
        s = self.s
        if not s.endpoint:
            if self.in_flight("assign"):
                # Its answer names the runtime (after_assign), or there is none.
                return
            self.released()
            return
        if self.in_flight("unassign"):
            return
        retried = s.action == "unassign"
        self.queue("unassign", endpoint=s.endpoint)
        if retried and self.obs.tick:
            # Colab may answer anything but 404 for a runtime the VM already
            # released: the listing ends the drain once the runtime is gone.
            self.queue_reconcile()

    def cooldown(self) -> None:
        until = self.s.cooldown_until
        if until is None or self.now >= until:
            self.move(IDLE, cooldown_until=None)
            self.idle()
            return
        self.periodic_reconcile()


# ── the decisions ────────────────────────────────────────────


def decide(state: ColabState, obs: Observations, cfg: ControllerConfig) -> Decision:
    """The tick's or a reconcile's decision. Pure."""
    p = _Pass(state, obs, cfg)
    if obs.ccu is not None:
        p.take_ccu(obs.ccu)
    if obs.lookups:
        p.take_lookups(obs.lookups)
    if obs.assignments is not None:
        p.take_assignments(obs.assignments)
    p.step()
    return p.decision()


def after_assign(
    state: ColabState,
    nbh: str,
    outcome: RuntimeAssignment | ControlPlaneError,
    now: float,
    cfg: ControllerConfig,
) -> Decision:
    """The ``assign`` task's answer for *nbh*. Pure."""
    p = _Pass(state, Observations(now), cfg)
    s = p.s
    current = s.nbh == nbh and s.active
    if isinstance(outcome, RuntimeAssignment):
        endpoint = outcome.endpoint
        known = _remember(s.known_endpoints, endpoint)
        if current and s.status == ASSIGNING:
            p.move(
                BOOTING,
                endpoint=endpoint,
                assigned_at=now,
                cu_accounted_at=now,
                boot_step=ASSIGNED_STEP,
                known_endpoints=known,
            )
            p.queue("launch", nbh=nbh, endpoint=endpoint)
        elif current and not s.endpoint:
            # Registered early, or switched off meanwhile: keep it named, so
            # it is released like any other runtime of this session.
            p.set(endpoint=endpoint, assigned_at=s.assigned_at or now, cu_accounted_at=now, known_endpoints=known)
            if s.status == DRAINING:
                p.draining()
        elif not (current and s.endpoint == endpoint):
            # The session moved on: nobody wants this runtime.
            p.set(known_endpoints=known)
            p.tasks.append(("unassign", {"endpoint": endpoint, "orphan": True}))
            p.notes.append("assigned runtime no longer wanted; released")
        return p.decision()
    # Colab refused it (412/503, a dead credential): this request made no
    # runtime. Only a first try's refusal vouches for the nbh: an earlier
    # try that never answered (a time-out, a lost task) may still leave one,
    # so the nbh is kept and looked up (ended()).
    refused = isinstance(outcome, (NoCapacityError, CredentialOrPolicyError))
    answered = nbh if refused and s.attempts <= 1 else ""
    if isinstance(outcome, CredentialOrPolicyError):
        p.disable(f"assign: {outcome}", answered=answered)
    elif current and s.status == DRAINING and not s.endpoint:
        # The drain waited for this answer.
        p.released(answered=answered)
    elif not (current and s.status == ASSIGNING):
        pass
    elif isinstance(outcome, NoCapacityError):
        p.cool_down(f"assign refused: {outcome}", answered=answered)
    elif s.action == "assign":
        # Time-out or transient: the tick retries with the same nbh.
        p.set(action_at=None, last_error=clip(f"assign: {outcome}"))
    return p.decision()


def after_launch(
    state: ColabState,
    nbh: str,
    endpoint: str,
    outcome: LaunchResult | LaunchError | ControlPlaneError | None,
    now: float,
    cfg: ControllerConfig,
) -> Decision:
    """The ``launch`` task's answer (None: the runtime was no longer listed). Pure."""
    p = _Pass(state, Observations(now), cfg)
    s = p.s
    if isinstance(outcome, CredentialOrPolicyError):
        p.disable(f"launch: {outcome}")
        return p.decision()
    if not (s.status == BOOTING and s.nbh == nbh and s.endpoint == endpoint):
        return p.decision()
    if isinstance(outcome, LaunchResult):
        p.set(
            kernel_id=outcome.kernel_id,
            session_id=outcome.session_id,
            boot_step="launched",
            last_heartbeat_at=now,
        )
    elif outcome is None and s.assigned_at is not None and now - s.assigned_at < RECLAIM_GRACE_SECONDS:
        # Colab may not list a runtime this young yet: the tick launches it
        # again (and drains it after MAX_LAUNCH_ATTEMPTS) instead of forgetting it.
        p.set(action_at=None, last_error="launch: the runtime is not listed yet")
    elif outcome is None:
        p.lost("the runtime is gone (before the launch)")
    elif isinstance(outcome, LaunchError):
        p.drain(f"launch: {outcome}", boot_failure=True)
    elif s.action == "launch":
        p.set(action_at=None, last_error=clip(f"launch: {outcome}"))
    return p.decision()


def after_heartbeat(
    state: ColabState,
    nbh: str,
    endpoint: str,
    outcome: CellResult | ControlPlaneError | None,
    now: float,
    cfg: ControllerConfig,
) -> Decision:
    """The ``heartbeat`` task's cell (None: the runtime was no longer listed). Pure."""
    p = _Pass(state, Observations(now), cfg)
    s = p.s
    if isinstance(outcome, CredentialOrPolicyError):
        p.disable(f"heartbeat: {outcome}")
        return p.decision()
    if not (s.status in (BOOTING, READY) and s.nbh == nbh and s.endpoint == endpoint):
        return p.decision()
    if outcome is None:
        p.lost("the runtime is gone (reclaimed by Colab or released by the VM)")
        return p.decision()
    if not isinstance(outcome, CellResult):
        return p.decision()
    if outcome.kernel_id and outcome.kernel_id != s.kernel_id:
        p.notes.append("heartbeat ran in a new kernel")
        p.set(kernel_id=outcome.kernel_id, session_id=outcome.session_id or s.session_id)
    vm = vm_state(outcome.stdout) if outcome.ok else {}
    step = str(vm.get("step") or "")[:40]
    error = str(vm.get("error") or "")
    if s.status == BOOTING and step and step != s.boot_step:
        p.set(boot_step=step)
    if error:
        p.set(last_error=clip(f"VM {step or '?'}: {error}"))
    return p.decision()


def after_unassign(
    state: ColabState,
    endpoint: str,
    outcome: bool | ControlPlaneError,
    now: float,
    cfg: ControllerConfig,
) -> Decision:
    """The ``unassign`` task's answer (True: released; False: already gone). Pure.

    A refusal, 401/403 included, never disables the tier: Colab may refuse
    to release a runtime the VM already released. The drain's next tick
    queues the unassign again with a reconcile, whose listing ends the
    session once the runtime is gone, and disables the tier when it is
    refused as well.
    """
    p = _Pass(state, Observations(now), cfg)
    s = p.s
    if not (s.status == DRAINING and s.endpoint == endpoint):
        pass
    elif isinstance(outcome, bool):
        p.released()
    elif s.action == "unassign":
        p.set(action_at=None, last_error=clip(f"unassign: {outcome}"))
    return p.decision()


def vm_state(stdout: str) -> dict[str, Any]:
    """The VM's state.json from the heartbeat cell's output ({} when absent)."""
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(HEARTBEAT_MARKER):
            try:
                parsed = json.loads(line[len(HEARTBEAT_MARKER) :])
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


# ── the service ──────────────────────────────────────────────

Submit = Callable[[str, dict[str, Any]], Awaitable[Any]]


class ColabController:
    """Runs the decisions: the cron's tick and the ``colab`` tasks."""

    def __init__(
        self,
        endpoint: Any,
        control_plane: Any,
        launcher: Any,
        *,
        submit: Submit,
        config: ControllerConfig,
        clock: Callable[[], float] = time.time,
        new_nbh: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        """``endpoint`` is the :class:`~.endpoint.ColabEndpoint` (its store is
        the one the request path reads, so writes here reach it at once)."""
        self._endpoint = endpoint
        self.store = endpoint.store
        self._control_plane = control_plane
        self._launcher = launcher
        self._submit = submit
        self.config = config
        self._clock = clock
        self._new_nbh = new_nbh
        # Runtimes not ours already reported at ERROR (once per process).
        self._reported: set[str] = set()
        self.last_tick: dict[str, Any] | None = None

    # ── cron ───────────────────────────────────────────────────

    async def tick(self) -> dict[str, Any]:
        """The per-minute pass: state in and out, tasks queued; never Colab itself."""
        now = self._clock()
        nbh = self._new_nbh()
        try:
            async with asyncio.timeout(TICK_STATE_TIMEOUT_SECONDS):
                decision = await self._apply(
                    lambda state: decide(state, Observations(now, new_nbh=nbh, tick=True), self.config)
                )
        except TimeoutError:
            logger.warning(f"Colab tick: the state took over {TICK_STATE_TIMEOUT_SECONDS:g}s; next minute")
            summary: dict[str, Any] = {"error": "state_timeout"}
        except ColabStateError as exc:
            logger.warning(f"Colab tick: {exc}")
            summary = {"error": "state_unavailable"}
        else:
            queued = await self._submit_all(decision)
            summary = {"status": decision.state.status, "queued": queued, "changes": list(decision.notes)}
        self.last_tick = {"at": now, **summary}
        return summary

    # ── tasks ──────────────────────────────────────────────────

    async def run_action(self, action: str, payload: dict[str, Any]) -> bool:
        """One ``colab`` task. False when it could not run (bad payload, no state)."""
        handler = {
            "assign": self._assign,
            "launch": self._launch,
            "heartbeat": self._heartbeat,
            "unassign": self._unassign,
            "reconcile": self._reconcile,
        }.get(action)
        if handler is None:
            logger.warning(f"Colab task: unknown action {action!r}")
            return False
        try:
            return await handler(payload)
        except ColabStateError as exc:
            logger.warning(f"Colab {action}: {exc}")
            return False

    async def _assign(self, payload: dict[str, Any]) -> bool:
        nbh = str(payload.get("nbh") or "")
        state = await self.store.load()
        if not nbh or state.status != ASSIGNING or state.nbh != nbh:
            logger.info("Colab assign: the session moved on; nothing to do")
            return True
        outcome: RuntimeAssignment | ControlPlaneError
        try:
            outcome = await self._control_plane.assign(nbh, cpu_only=self.config.dry_run)
        except ControlPlaneError as exc:
            outcome = exc
        now = self._clock()
        decision = await self._apply(lambda s: after_assign(s, nbh, outcome, now, self.config))
        await self._submit_all(decision)
        return True

    async def _launch(self, payload: dict[str, Any]) -> bool:
        nbh, endpoint = str(payload.get("nbh") or ""), str(payload.get("endpoint") or "")
        state = await self.store.load()
        if state.status != BOOTING or state.nbh != nbh or state.endpoint != endpoint or state.boot_step != ASSIGNED_STEP:
            logger.info("Colab launch: the session moved on or is launched; nothing to do")
            return True
        outcome: LaunchResult | LaunchError | ControlPlaneError | None
        try:
            runtime = await self._runtime(endpoint)
            outcome = None if runtime is None else await self._launcher.launch(runtime, nbh)
        except (LaunchError, ControlPlaneError) as exc:
            outcome = exc
        if isinstance(outcome, LaunchResult):
            logger.info(f"Colab launch: bootstrap started (pid {outcome.pid})")
        elif outcome is not None:
            logger.warning(f"Colab launch failed: {type(outcome).__name__}: {outcome}")
        now = self._clock()
        decision = await self._apply(lambda s: after_launch(s, nbh, endpoint, outcome, now, self.config))
        await self._submit_all(decision)
        return True

    async def _heartbeat(self, payload: dict[str, Any]) -> bool:
        nbh, endpoint = str(payload.get("nbh") or ""), str(payload.get("endpoint") or "")
        state = await self.store.load()
        if state.status not in (BOOTING, READY) or state.nbh != nbh or state.endpoint != endpoint:
            return True
        outcome: CellResult | ControlPlaneError | None
        try:
            runtime = await self._runtime(endpoint)
            outcome = (
                None
                if runtime is None
                else await self._control_plane.run_cell(
                    runtime,
                    HEARTBEAT_CODE,
                    budget_s=HEARTBEAT_BUDGET_SECONDS,
                    kernel_id=state.kernel_id,
                    session_id=state.session_id,
                )
            )
        except ControlPlaneError as exc:
            outcome = exc
        if isinstance(outcome, CellResult) and not outcome.ok:
            logger.warning(f"Colab heartbeat: the cell ended {outcome.status}: {clip(outcome.error, 200)}")
        now = self._clock()
        decision = await self._apply(lambda s: after_heartbeat(s, nbh, endpoint, outcome, now, self.config))
        await self._submit_all(decision)
        return True

    async def _unassign(self, payload: dict[str, Any]) -> bool:
        endpoint = str(payload.get("endpoint") or "")
        if not endpoint:
            logger.warning("Colab unassign: payload has no endpoint")
            return False
        # Run even when DISABLED: releasing a runtime of ours is the one call
        # worth making then (the disable queues the one its session held).
        orphan = bool(payload.get("orphan"))
        outcome: bool | ControlPlaneError
        try:
            outcome = await self._control_plane.unassign(endpoint)
        except ControlPlaneError as exc:
            outcome = exc
        if orphan and outcome is True:
            logger.warning("Colab: released a runtime that was not the current session")
        now = self._clock()
        decision = await self._apply(lambda s: after_unassign(s, endpoint, outcome, now, self.config))
        await self._submit_all(decision)
        return True

    async def _reconcile(self, payload: dict[str, Any]) -> bool:
        state = await self.store.load()
        if state.status == DISABLED:
            # No Colab calls until !colab on.
            return True
        wanted = lookups_due(state, self._clock())
        listed, ccu, healthy, *found = await asyncio.gather(
            self._control_plane.list_assignments(),
            self._control_plane.ccu_info(),
            self._health(state),
            *(self._control_plane.find_assignment(nbh, cpu_only=self.config.dry_run) for nbh in wanted),
            return_exceptions=True,
        )
        reads = (listed, ccu, *found)
        dead = next((item for item in reads if isinstance(item, CredentialOrPolicyError)), None)
        now = self._clock()
        if dead is not None:
            decision = await self._apply(lambda s: _disabled(s, f"reconcile: {dead}", now, self.config))
            await self._submit_all(decision)
            return True
        for problem in (item for item in (*reads, healthy) if isinstance(item, BaseException)):
            if not isinstance(problem, ControlPlaneError):
                logger.warning(f"Colab reconcile: {type(problem).__name__}: {problem}")
        obs = Observations(
            now,
            new_nbh=self._new_nbh(),
            ccu=ccu if isinstance(ccu, CcuInfo) else None,
            assignments=tuple(listed) if isinstance(listed, list) else None,
            healthy=healthy if isinstance(healthy, bool) else None,
            lookups=tuple(
                (nbh, item.endpoint if item is not None else "")
                for nbh, item in zip(wanted, found, strict=True)
                if item is None or isinstance(item, RuntimeAssignment)
            ),
        )
        decision = await self._apply(lambda s: decide(s, obs, self.config))
        for endpoint in decision.foreign:
            if endpoint not in self._reported:
                self._reported.add(endpoint)
                logger.error(
                    "Colab: a runtime this bot did not start is assigned to the account; left "
                    "alone (set COLAB_ACCOUNT_DEDICATED=true to release such runtimes)"
                )
        await self._submit_all(decision)
        return True

    # ── helpers ────────────────────────────────────────────────

    async def _health(self, state: ColabState) -> bool | None:
        if state.status != READY or not state.tunnel_url:
            return None
        return await self._endpoint.check_health()

    async def _runtime(self, endpoint: str) -> RuntimeAssignment | None:
        """Our runtime with a fresh proxy token (list_assignments hands one out)."""
        for runtime in await self._control_plane.list_assignments():
            if runtime.endpoint == endpoint:
                return runtime
        return None

    async def _apply(self, decide_for: Callable[[ColabState], Decision]) -> Decision:
        """Write the decision taken on the stored state; its tasks are the last try's."""
        taken: list[Decision] = []

        def change(state: ColabState) -> ColabState:
            decision = decide_for(state)
            taken.append(decision)
            return decision.state

        stored = await self.store.update(change)
        decision = replace(taken[-1], state=stored)
        for note in decision.notes:
            if note.endswith(f"→ {DISABLED}"):
                # A human must act (!colab on after fixing the cause).
                logger.error(f"Colab: {note}: {stored.disabled_reason}")
            else:
                logger.info(f"Colab: {note}")
        return decision

    async def _submit_all(self, decision: Decision) -> list[str]:
        queued: list[str] = []
        for action, payload in decision.tasks:
            try:
                await self._submit(TASK_KIND, {"action": action, **payload})
            except Exception as exc:
                # The lease queues it again; a periodic one comes next minute.
                logger.warning(f"Colab: could not queue {action}: {exc}")
                continue
            queued.append(action)
        return queued

    def view(self) -> dict[str, Any]:
        """The controller's settings and last tick, for ``/health/detail``."""
        cfg = self.config
        return {
            "dry_run": cfg.dry_run,
            "account_dedicated": cfg.account_dedicated,
            "heartbeat_seconds": cfg.heartbeat_seconds,
            "boot_timeout_seconds": cfg.boot_timeout_seconds,
            "last_tick": self.last_tick,
        }


def _disabled(state: ColabState, reason: str, now: float, cfg: ControllerConfig) -> Decision:
    p = _Pass(state, Observations(now), cfg)
    p.disable(reason)
    return p.decision()
