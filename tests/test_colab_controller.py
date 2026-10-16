"""The Colab controller: every transition of the pure decisions, then the service
(the cron tick and the ``colab`` tasks) over a fake control plane and a fake clock."""

from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from src.services.colab.admin import turn_off, turn_on
from src.services.colab.control_plane import (
    CELL_ERROR,
    CcuInfo,
    CellResult,
    ControlPlaneError,
    ControlPlaneTimeoutError,
    CredentialOrPolicyError,
    NoCapacityError,
    TransientControlPlaneError,
)
from src.services.colab.controller import (
    ACTIONS,
    COOLDOWN_STEPS_SECONDS,
    ESTIMATED_A100_CU_PER_HOUR,
    ESTIMATED_CPU_CU_PER_HOUR,
    HEARTBEAT_CODE,
    HEARTBEAT_MARKER,
    LEASE_SECONDS,
    MAX_ASSIGN_ATTEMPTS,
    START_DEMAND_SECONDS,
    TASK_KIND,
    UNANSWERED_RETRY_SECONDS,
    ColabController,
    ControllerConfig,
    Decision,
    Observations,
    after_assign,
    after_heartbeat,
    after_launch,
    after_unassign,
    decide,
    lookups_due,
    taipei_day,
    vm_state,
)
from src.services.colab.launcher import LaunchError, LaunchResult
from src.services.colab.state import (
    ASSIGNING,
    BOOTING,
    COOLDOWN,
    DISABLED,
    DRAINING,
    IDLE,
    OFF,
    READY,
    ColabState,
    ColabStateError,
    InMemoryColabStateStore,
)
from tests.colab_fakes import (
    NBH,
    TUNNEL,
    Clock,
    FakeControlPlane,
    FakeTunnel,
    make_endpoint,
    runtime,
)

NOW = 1_790_000_000.0
NEW_NBH = "0b6f8f5e-6a51-4c0b-8f7e-1d2c3b4a5f60"
CFG = ControllerConfig()


def _ccu(**overrides: Any) -> dict[str, Any]:
    """State fields of a fresh ccu-info read with plenty of balance."""
    values: dict[str, Any] = {
        "ccu_balance": 100.0,
        "ccu_rate": 0.0,
        "ccu_checked_at": NOW - 60,
        "a100_eligible": True,
        "cu_day": taipei_day(NOW),
    }
    values.update(overrides)
    return values


def _idle(**overrides: Any) -> ColabState:
    values = {**_ccu(), "last_demand_at": NOW - 10}
    values.update(overrides)
    return ColabState(**values)


def _booting(**overrides: Any) -> ColabState:
    values: dict[str, Any] = {
        **_ccu(),
        "status": BOOTING,
        "since": NOW - 120,
        "nbh": NBH,
        "endpoint": "ep-1",
        "assigned_at": NOW - 120,
        "cu_accounted_at": NOW - 60,
        "kernel_id": "kernel-1",
        "session_id": "session-1",
        "boot_step": "launched",
        "last_heartbeat_at": NOW - 100,
        "last_demand_at": NOW - 60,
        "known_endpoints": ("ep-1",),
    }
    values.update(overrides)
    return ColabState(**values)


def _ready(**overrides: Any) -> ColabState:
    values: dict[str, Any] = {
        "status": READY,
        "since": NOW - 300,
        "ready_at": NOW - 300,
        "tunnel_url": TUNNEL,
        "last_health_at": NOW - 30,
        "last_heartbeat_at": NOW - 60,
        "boot_step": "ready",
        "assigned_at": NOW - 900,
    }
    values.update(overrides)
    return _booting(**values)


def _draining(**overrides: Any) -> ColabState:
    return _booting(**{"status": DRAINING, "since": NOW - 5, **overrides})


def tick(state: ColabState, cfg: ControllerConfig = CFG, now: float = NOW) -> Decision:
    return decide(state, Observations(now, new_nbh=NEW_NBH, tick=True), cfg)


def seen(
    state: ColabState,
    *,
    ccu: CcuInfo | None = None,
    runtimes: list | None = None,
    healthy: bool | None = None,
    cfg: ControllerConfig = CFG,
    now: float = NOW,
) -> Decision:
    """A reconcile's pass (not the tick)."""
    listed = tuple(runtimes) if runtimes is not None else None
    return decide(state, Observations(now, new_nbh=NEW_NBH, ccu=ccu, assignments=listed, healthy=healthy), cfg)


def actions(decision: Decision) -> list[str]:
    return [action for action, _ in decision.tasks]


# ── IDLE: starting a session ─────────────────────────────────


class StartTests(unittest.TestCase):
    def test_fresh_demand_starts_with_a_new_nbh_written_first(self) -> None:
        decision = tick(_idle(starts_today=1))

        state = decision.state
        self.assertEqual((state.status, state.since, state.nbh), (ASSIGNING, NOW, NEW_NBH))
        self.assertEqual((state.action, state.action_at, state.attempts), ("assign", NOW, 1))
        self.assertEqual(state.starts_today, 2)
        self.assertEqual(decision.tasks, (("assign", {"nbh": NEW_NBH}),))

    def test_no_start_without_recent_demand_or_when_switched_off(self) -> None:
        cases = {
            "no demand": _idle(last_demand_at=None),
            "old demand": _idle(last_demand_at=NOW - START_DEMAND_SECONDS - 1),
            "!colab off": _idle(desired=OFF),
        }
        for name, state in cases.items():
            with self.subTest(name):
                decision = tick(replace(state, reconciled_day=taipei_day(NOW)))
                self.assertEqual((decision.state.status, decision.tasks), (IDLE, ()))

    def test_api_preference_never_starts_for_messages_but_does_for_colab_on(self) -> None:
        api = ControllerConfig(auto_start=False)
        state = _idle(reconciled_day=taipei_day(NOW))

        self.assertEqual((tick(state, api).state.status, tick(state, api).tasks), (IDLE, ()))
        manual = turn_on(replace(state, last_demand_at=None), NOW - 5)
        for cfg in (api, CFG):
            with self.subTest(auto_start=cfg.auto_start):
                self.assertEqual(tick(manual, cfg).state.status, ASSIGNING)

    def test_a_stale_ccu_read_is_refreshed_before_a_start(self) -> None:
        state = _idle(ccu_checked_at=NOW - 3600)

        decision = tick(state)

        self.assertEqual((decision.state.status, actions(decision)), (IDLE, ["reconcile"]))
        self.assertEqual(decision.state.reconciled_day, taipei_day(NOW))
        # The reconcile reads ccu-info and starts in the same pass.
        started = seen(decision.state, ccu=CcuInfo(current_balance=50.0, consumption_rate_hourly=0.0), runtimes=[])
        self.assertEqual((started.state.status, started.state.ccu_balance), (ASSIGNING, 50.0))
        self.assertEqual(actions(started), ["assign"])

    def test_holds_keep_it_idle_with_the_reason(self) -> None:
        cases = {
            "餘額 5.0 低於下限": _idle(ccu_balance=5.0),
            "沒有回報 CU 餘額": _idle(ccu_balance=None),
            "達每日上限": _idle(cu_used_today=30.0),
            "不提供 A100": _idle(a100_eligible=False),
        }
        for reason, state in cases.items():
            with self.subTest(reason):
                decision = tick(replace(state, reconciled_day=taipei_day(NOW)))
                self.assertEqual((decision.state.status, decision.tasks), (IDLE, ()))
                self.assertIn(reason, decision.state.hold_reason)
        # A reconcile whose ccu-info read failed holds it too.
        failed = seen(_idle(ccu_checked_at=None), runtimes=[])
        self.assertEqual((failed.state.status, failed.tasks), (IDLE, ()))
        self.assertIn("讀不到 CU 餘額", failed.state.hold_reason)
        # The hold is cleared once nobody wants a session.
        self.assertEqual(tick(replace(failed.state, last_demand_at=None)).state.hold_reason, "")

    def test_a_start_needs_room_for_a_boot_and_an_idle_ttl(self) -> None:
        # 5.37 CU/h × (900 s boot + 900 s idle) ≈ 2.7 CU: less would be drained at READY.
        cost = ESTIMATED_A100_CU_PER_HOUR * (CFG.boot_timeout_seconds + CFG.idle_ttl_seconds) / 3600
        self.assertAlmostEqual(cost, 2.685)
        held = {
            "會超過每日上限": _idle(cu_used_today=CFG.daily_cu_cap - cost + 0.1),
            "會低於下限": _idle(ccu_balance=CFG.min_cu_balance + cost - 0.1),
        }
        for reason, state in held.items():
            with self.subTest(reason):
                decision = tick(replace(state, reconciled_day=taipei_day(NOW)))
                self.assertEqual((decision.state.status, decision.tasks), (IDLE, ()))
                self.assertIn(reason, decision.state.hold_reason)
                self.assertIn("約 2.7 CU", decision.state.hold_reason)
        for name, state in {
            "cap": _idle(cu_used_today=CFG.daily_cu_cap - cost - 0.1),
            "floor": _idle(ccu_balance=CFG.min_cu_balance + cost + 0.1),
        }.items():
            with self.subTest(room=name):
                self.assertEqual(tick(state).state.status, ASSIGNING)
        # A dry run's CPU runtime costs next to nothing.
        self.assertEqual(tick(_idle(cu_used_today=29.9), ControllerConfig(dry_run=True)).state.status, ASSIGNING)

    def test_a_dry_run_ignores_a100_eligibility(self) -> None:
        decision = tick(_idle(a100_eligible=False), ControllerConfig(dry_run=True))
        self.assertEqual(decision.state.status, ASSIGNING)

    def test_an_idle_account_is_reconciled_once_a_day(self) -> None:
        state = ColabState(**_ccu())

        first = tick(state)
        again = tick(first.state, now=NOW + 60)

        self.assertEqual((actions(first), actions(again)), (["reconcile"], []))
        self.assertEqual(actions(tick(first.state, now=NOW + 86400)), ["reconcile"])


# ── ASSIGNING ────────────────────────────────────────────────


def _assigning(**overrides: Any) -> ColabState:
    return tick(_idle()).state if not overrides else replace(tick(_idle()).state, **overrides)


class AssignTests(unittest.TestCase):
    def test_an_assign_in_flight_is_left_alone_until_its_lease_ends(self) -> None:
        state = _assigning()

        self.assertEqual(tick(state, now=NOW + 60).tasks, ())
        retried = tick(state, now=NOW + LEASE_SECONDS["assign"] + 1)
        # A lost task: queued again, same nbh.
        self.assertEqual(retried.tasks, (("assign", {"nbh": NEW_NBH}),))
        self.assertEqual(retried.state.attempts, 2)

    def test_the_runtime_boots(self) -> None:
        decision = after_assign(_assigning(), NEW_NBH, runtime("ep-9"), NOW + 30, CFG)

        state = decision.state
        self.assertEqual((state.status, state.endpoint, state.assigned_at), (BOOTING, "ep-9", NOW + 30))
        self.assertEqual((state.boot_step, state.cu_accounted_at, state.known_endpoints), ("assigned", NOW + 30, ("ep-9",)))
        self.assertEqual(decision.tasks, (("launch", {"nbh": NEW_NBH, "endpoint": "ep-9"}),))
        self.assertEqual((state.action, state.attempts), ("launch", 1))

    def test_a_time_out_is_retried_with_the_same_nbh_then_cools_down(self) -> None:
        state = _assigning()
        for attempt in range(1, MAX_ASSIGN_ATTEMPTS + 1):
            with self.subTest(attempt=attempt):
                self.assertEqual((state.attempts, state.nbh), (attempt, NEW_NBH))
                failed = after_assign(state, NEW_NBH, ControlPlaneTimeoutError("assign: no answer"), NOW, CFG).state
                self.assertEqual((failed.status, failed.action_at), (ASSIGNING, None))
                self.assertIn("no answer", failed.last_error)
                decision = tick(failed, now=NOW + 60)
                state = decision.state
        # Three tries without a runtime: cool down, the session over, but its
        # nbh kept: a timed-out request may have left a runtime after all.
        self.assertEqual((state.status, state.nbh, state.cooldown_step), (COOLDOWN, "", 1))
        self.assertEqual(state.cooldown_until, NOW + 60 + COOLDOWN_STEPS_SECONDS[0])
        self.assertEqual((state.unanswered_nbhs, state.unanswered_at), ((NEW_NBH,), NOW + 60))

    def test_refused_assigns_back_off_5_10_20_60_minutes(self) -> None:
        state = _idle()
        waits = []
        for step in range(5):
            state = tick(replace(state, cooldown_until=None, status=IDLE, last_demand_at=NOW - 10)).state
            refusal = NoCapacityError("assign refused (HTTP 412)", status=412 if step % 2 else 503)
            state = after_assign(state, state.nbh, refusal, NOW, CFG).state
            self.assertEqual(state.status, COOLDOWN)
            waits.append(state.cooldown_until - NOW)
        self.assertEqual(waits, [300.0, 600.0, 1200.0, 3600.0, 3600.0])

    def test_credential_or_policy_errors_disable_the_tier(self) -> None:
        decision = after_assign(_assigning(), NEW_NBH, CredentialOrPolicyError("DENYLISTED"), NOW, CFG)

        state = decision.state
        self.assertEqual((state.status, state.nbh), (DISABLED, ""))
        self.assertIn("DENYLISTED", state.disabled_reason)
        # Nothing more happens until !colab on.
        self.assertEqual(tick(replace(state, last_demand_at=NOW)).tasks, ())
        self.assertEqual(tick(turn_on(state, NOW)).state.status, ASSIGNING)

    def test_a_runtime_that_arrives_after_colab_off_is_released(self) -> None:
        state = turn_off(_assigning(), NOW + 5)
        self.assertEqual(state.status, DRAINING)
        # While the assign is in flight the drain waits for its answer.
        self.assertEqual((tick(state, now=NOW + 10).state.status, tick(state, now=NOW + 10).tasks), (DRAINING, ()))

        decision = after_assign(state, NEW_NBH, runtime("ep-5"), NOW + 20, CFG)

        self.assertEqual((decision.state.status, decision.state.endpoint), (DRAINING, "ep-5"))
        self.assertEqual(decision.tasks, (("unassign", {"endpoint": "ep-5"}),))
        # No answer within the lease: released, the nbh kept for a lookup.
        lapsed = tick(state, now=NOW + LEASE_SECONDS["assign"] + 1).state
        self.assertEqual((lapsed.status, lapsed.nbh, lapsed.unanswered_nbhs), (IDLE, "", (NEW_NBH,)))

    def test_a_drain_waiting_for_an_assign_ends_with_its_answer(self) -> None:
        state = turn_off(_assigning(), NOW + 5)
        # No runtime named (a time-out, an error): released now, the nbh looked up later.
        for outcome in (ControlPlaneTimeoutError("assign: no answer"), TransientControlPlaneError("502")):
            with self.subTest(outcome=type(outcome).__name__):
                released = after_assign(state, NEW_NBH, outcome, NOW + 130, CFG)
                self.assertEqual((released.state.status, released.tasks), (IDLE, ()))
                self.assertEqual(released.state.unanswered_nbhs, (NEW_NBH,))
        # Refused: Colab made no runtime, so there is nothing to look up.
        refused = after_assign(state, NEW_NBH, NoCapacityError("412"), NOW + 130, CFG).state
        self.assertEqual((refused.status, refused.unanswered_nbhs), (IDLE, ()))

    def test_refused_assigns_leave_nothing_to_look_up(self) -> None:
        cooled = after_assign(_assigning(), NEW_NBH, NoCapacityError("412"), NOW, CFG).state
        disabled = after_assign(_assigning(), NEW_NBH, CredentialOrPolicyError("401"), NOW, CFG).state
        self.assertEqual((cooled.status, cooled.unanswered_nbhs), (COOLDOWN, ()))
        self.assertEqual((disabled.status, disabled.unanswered_nbhs), (DISABLED, ()))
        # A stale refusal for another nbh does not vouch for the current one.
        other = after_assign(_assigning(), "old-nbh", CredentialOrPolicyError("401"), NOW, CFG).state
        self.assertEqual((other.status, other.unanswered_nbhs), (DISABLED, (NEW_NBH,)))

    def test_a_refused_retry_does_not_vouch_for_the_try_that_never_answered(self) -> None:
        # The first try timed out (or its task was lost) and Colab may still
        # be making its runtime; the retry, same nbh, is refused: only that
        # request made nothing, so the nbh is still looked up.
        timed_out = after_assign(_assigning(), NEW_NBH, ControlPlaneTimeoutError("assign: no answer"), NOW, CFG).state
        retries = {
            "after a time-out": tick(timed_out, now=NOW + 60).state,
            "after a lost task": tick(_assigning(), now=NOW + LEASE_SECONDS["assign"] + 1).state,
        }
        for name, retry in retries.items():
            self.assertEqual((retry.status, retry.attempts), (ASSIGNING, 2))
            later = NOW + 300
            for refusal, status in ((NoCapacityError("412"), COOLDOWN), (CredentialOrPolicyError("401"), DISABLED)):
                with self.subTest(name, refusal=type(refusal).__name__):
                    ended = after_assign(retry, NEW_NBH, refusal, later, CFG).state
                    self.assertEqual((ended.status, ended.nbh), (status, ""))
                    self.assertEqual((ended.unanswered_nbhs, ended.unanswered_at), ((NEW_NBH,), later))
            with self.subTest(name, drain=True):
                drained = after_assign(turn_off(retry, later), NEW_NBH, NoCapacityError("412"), later, CFG).state
                self.assertEqual((drained.status, drained.unanswered_nbhs), (IDLE, (NEW_NBH,)))

    def test_a_runtime_for_a_session_that_moved_on_is_released(self) -> None:
        decision = after_assign(ColabState(status=IDLE), NEW_NBH, runtime("ep-7"), NOW, CFG)

        self.assertEqual(decision.tasks, (("unassign", {"endpoint": "ep-7", "orphan": True}),))
        self.assertEqual((decision.state.status, decision.state.known_endpoints), (IDLE, ("ep-7",)))

    def test_stale_failures_change_nothing(self) -> None:
        state = ColabState(status=IDLE, cu_day=taipei_day(NOW))
        for outcome in (NoCapacityError("412"), TransientControlPlaneError("x"), ControlPlaneError("?")):
            with self.subTest(outcome=type(outcome).__name__):
                self.assertEqual(after_assign(state, NEW_NBH, outcome, NOW, CFG).state, state)


# ── BOOTING ──────────────────────────────────────────────────


class BootTests(unittest.TestCase):
    def test_the_launch_is_queued_until_it_reports(self) -> None:
        state = _booting(kernel_id="", session_id="", boot_step="assigned", action="launch", action_at=NOW - 10, attempts=1)

        self.assertNotIn("launch", actions(tick(state)))
        late = tick(state, now=NOW + LEASE_SECONDS["launch"])
        self.assertIn(("launch", {"nbh": NBH, "endpoint": "ep-1"}), late.tasks)

        launched = after_launch(state, NBH, "ep-1", LaunchResult("kernel-9", "session-9", 77, "0.41.2"), NOW, CFG).state
        self.assertEqual((launched.kernel_id, launched.session_id, launched.boot_step), ("kernel-9", "session-9", "launched"))
        # The release the launcher picked: the probe and health checks want it.
        self.assertEqual(launched.ollama_version, "0.41.2")
        self.assertEqual(launched.last_heartbeat_at, NOW)
        self.assertNotIn("launch", actions(tick(launched, now=NOW + 400)))

    def test_a_failed_launch_is_a_boot_failure(self) -> None:
        state = _booting(kernel_id="", boot_step="assigned", action="launch", action_at=NOW, attempts=1, failures=1)

        decision = after_launch(state, NBH, "ep-1", LaunchError("bootstrap.py did not start"), NOW, CFG)

        self.assertEqual((decision.state.status, decision.state.failures), (DRAINING, 2))
        self.assertEqual(decision.tasks, (("unassign", {"endpoint": "ep-1"}),))

    def test_colab_trouble_during_the_launch_is_retried_once(self) -> None:
        state = _booting(kernel_id="", boot_step="assigned", action="launch", action_at=NOW, attempts=1)

        failed = after_launch(state, NBH, "ep-1", TransientControlPlaneError("upload: 502"), NOW, CFG).state
        again = tick(failed, now=NOW + 60)
        self.assertEqual((again.state.attempts, actions(again)[0]), (2, "launch"))

        failed = after_launch(again.state, NBH, "ep-1", TransientControlPlaneError("upload: 502"), NOW + 70, CFG).state
        gave_up = tick(failed, now=NOW + 120)
        self.assertEqual((gave_up.state.status, gave_up.state.failures), (DRAINING, 1))
        self.assertIn("launch failed 2 times", gave_up.state.last_error)

    def test_a_runtime_gone_before_the_launch_is_released(self) -> None:
        state = _booting(kernel_id="", boot_step="assigned", action="launch", action_at=NOW)
        decision = after_launch(state, NBH, "ep-1", None, NOW, CFG)
        self.assertEqual((decision.state.status, decision.state.endpoint), (IDLE, ""))

    def test_a_runtime_not_listed_yet_is_launched_again_not_forgotten(self) -> None:
        # Seconds after the assign the listing may not show it (the reconcile's grace).
        state = _booting(
            kernel_id="", boot_step="assigned", action="launch", action_at=NOW - 3, attempts=1, assigned_at=NOW - 5
        )
        waited = after_launch(state, NBH, "ep-1", None, NOW, CFG).state
        self.assertEqual((waited.status, waited.endpoint, waited.failures), (BOOTING, "ep-1", 0))
        self.assertIn(("launch", {"nbh": NBH, "endpoint": "ep-1"}), tick(waited, now=NOW + 60).tasks)
        # Still unlisted within the grace on the last try: drained, so released.
        again = replace(waited, attempts=2)
        self.assertEqual(tick(after_launch(again, NBH, "ep-1", None, NOW + 1, CFG).state, now=NOW + 60).tasks,
                         (("unassign", {"endpoint": "ep-1"}),))

    def test_the_boot_timeout_drains_and_counts(self) -> None:
        state = _booting(boot_step="pull", assigned_at=NOW - 901)

        decision = tick(state)

        self.assertEqual((decision.state.status, decision.state.failures), (DRAINING, 1))
        self.assertIn("last step: pull", decision.state.last_error)
        self.assertEqual(actions(decision), ["unassign"])
        custom = ControllerConfig(boot_timeout_seconds=1200)
        self.assertEqual(tick(state, custom).state.status, BOOTING)

    def test_three_boot_failures_in_a_row_disable_it(self) -> None:
        state = _draining(failures=3, last_error="VM bootstrap failed at pull: digest")

        decision = after_unassign(state, "ep-1", True, NOW, CFG)

        self.assertEqual(decision.state.status, DISABLED)
        self.assertIn("連續 3 次開機失敗", decision.state.disabled_reason)
        self.assertEqual(after_unassign(replace(state, failures=2), "ep-1", True, NOW, CFG).state.status, IDLE)

    def test_booting_keeps_the_kernel_alive_and_is_reconciled_every_tick(self) -> None:
        decision = tick(_booting(last_heartbeat_at=NOW - 300))
        self.assertEqual(actions(decision), ["heartbeat", "reconcile"])
        self.assertEqual(decision.tasks[0], ("heartbeat", {"nbh": NBH, "endpoint": "ep-1"}))
        self.assertEqual(decision.state.last_heartbeat_at, NOW)
        # Not again before the interval (the next minute).
        self.assertEqual(actions(tick(decision.state, now=NOW + 60)), ["reconcile"])

    def test_colab_off_while_booting_drains(self) -> None:
        state = turn_off(_booting(), NOW)
        self.assertEqual(actions(tick(state)), ["unassign"])

    def test_the_cap_or_the_floor_crossed_while_booting_drains_it(self) -> None:
        # READY would release it at once: stop paying for the rest of the boot.
        for reason, state in {
            "daily CU cap": _booting(cu_used_today=30.0, failures=1),
            "CU balance": _booting(ccu_balance=9.0, failures=1),
        }.items():
            with self.subTest(reason):
                decision = tick(state)
                self.assertEqual((decision.state.status, actions(decision)), (DRAINING, ["unassign"]))
                self.assertIn(reason, decision.state.last_error)
                # Not a boot failure.
                self.assertEqual(decision.state.failures, 1)


# ── READY ────────────────────────────────────────────────────


class ReadyTests(unittest.TestCase):
    def test_in_use_it_stays_with_heartbeats_and_reconciles(self) -> None:
        decision = tick(_ready(last_heartbeat_at=NOW - 296))
        self.assertEqual((decision.state.status, actions(decision)), (READY, ["heartbeat", "reconcile"]))

    def test_health_checks_three_failures_in_a_row_drain(self) -> None:
        state = _ready(health_failures=0)
        for failures in (1, 2):
            state = seen(state, healthy=False).state
            self.assertEqual((state.status, state.health_failures), (READY, failures))
        ok = seen(state, healthy=True).state
        self.assertEqual((ok.health_failures, ok.last_health_at), (0, NOW))
        drained = seen(state, healthy=False)
        self.assertEqual((drained.state.status, actions(drained)), (DRAINING, ["unassign"]))
        self.assertIn("3 health checks", drained.state.last_error)
        # Not a boot failure.
        self.assertEqual(drained.state.failures, 0)

    def test_released_when_idle_old_poor_capped_or_switched_off(self) -> None:
        cases = {
            "idle for": _ready(last_demand_at=NOW - 901, ready_at=NOW - 1000),
            "session age": _ready(assigned_at=NOW - 39601),
            "CU balance": _ready(ccu_balance=9.5),
            "daily CU cap": _ready(cu_used_today=30.5),
            "!colab off": turn_off(_ready(), NOW),
        }
        for reason, state in cases.items():
            with self.subTest(reason):
                decision = tick(state)
                self.assertEqual((decision.state.status, actions(decision)), (DRAINING, ["unassign"]))
                self.assertIn(reason, decision.state.last_error)

    def test_the_idle_clock_starts_no_earlier_than_ready(self) -> None:
        # Demand came 20 minutes ago, but READY only 5 minutes ago.
        state = _ready(last_demand_at=NOW - 1200, ready_at=NOW - 300)
        self.assertEqual(tick(state).state.status, READY)
        # !colab on keeps a session under MODEL_PATH_PREFERENCE=api going.
        manual = _ready(last_demand_at=None, ready_at=NOW - 2000, manual_demand_at=NOW - 100)
        self.assertEqual(tick(manual, ControllerConfig(auto_start=False)).state.status, READY)

    def test_no_heartbeat_for_a_session_nobody_uses(self) -> None:
        # Idle past the TTL: released instead of kept alive.
        decision = tick(_ready(last_demand_at=NOW - 1000, ready_at=NOW - 1000, last_heartbeat_at=NOW - 600))
        self.assertNotIn("heartbeat", actions(decision))


# ── DRAINING, reclaim, orphans, cooldown ─────────────────────


class ReleaseTests(unittest.TestCase):
    def test_draining_unassigns_then_idles(self) -> None:
        decision = tick(_draining())
        self.assertEqual(decision.tasks, (("unassign", {"endpoint": "ep-1"}),))
        self.assertEqual(tick(decision.state, now=NOW + 60).tasks, ())

        for outcome in (True, False):  # released, or already gone (404)
            with self.subTest(outcome=outcome):
                state = after_unassign(decision.state, "ep-1", outcome, NOW + 30, CFG).state
                self.assertEqual((state.status, state.endpoint, state.nbh, state.kernel_id), (IDLE, "", "", ""))
                self.assertEqual(state.known_endpoints, ("ep-1",))

    def test_a_failed_unassign_is_retried_with_a_reconcile(self) -> None:
        queued = tick(_draining()).state
        failed = after_unassign(queued, "ep-1", TransientControlPlaneError("503"), NOW, CFG).state
        self.assertEqual((failed.status, failed.action_at), (DRAINING, None))
        self.assertEqual(actions(tick(failed, now=NOW + 60)), ["unassign", "reconcile"])

    def test_a_runtime_the_vm_released_ends_the_drain_whatever_unassign_answers(self) -> None:
        # The VM reported failed (or stopping) and released itself; Colab
        # answers the controller's unassign with anything but 404.
        queued = tick(_draining(last_error="VM bootstrap failed at pull: digest", last_demand_at=NOW - 600)).state
        for refusal in (
            ControlPlaneError("unassign failed: 400", status=400),
            TransientControlPlaneError("unassign failed: 500", status=500),
            CredentialOrPolicyError("unassign failed: 403 Forbidden", status=403),
        ):
            with self.subTest(refusal=type(refusal).__name__):
                failed = after_unassign(queued, "ep-1", refusal, NOW + 30, CFG).state
                self.assertEqual((failed.status, failed.disabled_reason), (DRAINING, ""))
                retried = tick(failed, now=NOW + 60)
                self.assertIn("reconcile", actions(retried))
                released = seen(retried.state, runtimes=[], now=NOW + 70).state
                self.assertEqual((released.status, released.endpoint), (IDLE, ""))
                # Nothing more is counted once it is gone.
                later = tick(released, now=NOW + 3600).state
                self.assertEqual(later.cu_used_today, released.cu_used_today)

    def test_a_reclaimed_runtime_idles_the_session(self) -> None:
        for state in (_booting(), _ready(), _draining()):
            state = replace(state, last_demand_at=NOW - 600)
            with self.subTest(status=state.status):
                decision = seen(state, runtimes=[])
                self.assertEqual((decision.state.status, decision.state.endpoint), (IDLE, ""))
        self.assertIn("gone", seen(_ready(), runtimes=[]).state.last_error)
        # A runtime assigned seconds ago may not be listed yet.
        young = _booting(assigned_at=NOW - 10)
        self.assertEqual(seen(young, runtimes=[]).state.status, BOOTING)

    def test_a_reclaimed_ready_session_may_start_again_for_fresh_demand(self) -> None:
        decision = seen(_ready(last_demand_at=NOW - 20), runtimes=[], ccu=CcuInfo(current_balance=80.0))
        self.assertEqual((decision.state.status, decision.state.nbh), (ASSIGNING, NEW_NBH))

    def test_what_a_session_saw_ends_with_it(self) -> None:
        # The probe's digests and /api/ps sizes belong to one session: none
        # of them may reach the next session's /health/detail.
        probed = {
            "ollama_version": "0.41.2",
            "versions": {"ollama": "0.41.2"},
            "digests": {"clef-flash:9b": "a" * 64},
            "vram": {"clef-flash:9b": {"size": 10, "size_vram": 10}},
            "gpu": {"name": "A100"},
        }
        released = after_unassign(tick(_draining(**probed)).state, "ep-1", True, NOW + 30, CFG).state
        restarted = seen(_ready(last_demand_at=NOW - 20, **probed), runtimes=[], ccu=CcuInfo(current_balance=80.0)).state
        for state in (released, restarted):
            with self.subTest(status=state.status):
                self.assertEqual((state.versions, state.digests, state.vram, state.gpu), ({}, {}, {}, {}))
                self.assertEqual(state.ollama_version, "")

    def test_orphans_ours_are_released_foreign_only_when_dedicated(self) -> None:
        state = _ready(known_endpoints=("ep-old", "ep-1"))
        listed = [runtime("ep-1"), runtime("ep-old"), runtime("ep-browser")]

        shared = seen(state, runtimes=listed)
        self.assertEqual(shared.tasks, (("unassign", {"endpoint": "ep-old", "orphan": True}),))
        self.assertEqual(shared.foreign, ("ep-browser",))

        dedicated = seen(state, runtimes=listed, cfg=ControllerConfig(account_dedicated=True))
        self.assertEqual(
            sorted(payload["endpoint"] for _, payload in dedicated.tasks), ["ep-browser", "ep-old"]
        )
        self.assertEqual(dedicated.foreign, ())

    def test_no_orphan_is_touched_while_an_assign_is_in_flight(self) -> None:
        decision = seen(_assigning(), runtimes=[runtime("ep-new")], cfg=ControllerConfig(account_dedicated=True))
        self.assertEqual((decision.tasks, decision.foreign), ((), ()))

    def test_a_stale_runtime_of_ours_is_released_after_colab_on(self) -> None:
        # Disabled while READY (credentials), fixed, then !colab on.
        state = turn_on(after_heartbeat(_ready(), NBH, "ep-1", CredentialOrPolicyError("401"), NOW, CFG).state, NOW)
        decision = seen(state, runtimes=[runtime("ep-1")], ccu=CcuInfo(current_balance=80.0))
        self.assertIn(("unassign", {"endpoint": "ep-1", "orphan": True}), decision.tasks)

    def test_cooldown_ends_then_starts_if_wanted(self) -> None:
        state = replace(_idle(), status=COOLDOWN, cooldown_until=NOW + 100, reconciled_day=taipei_day(NOW))
        self.assertEqual((tick(state).state.status, tick(state).tasks), (COOLDOWN, ()))
        later = replace(state, last_demand_at=NOW + 150)
        decision = tick(later, now=NOW + 200)
        self.assertEqual((decision.state.status, decision.state.cooldown_until), (ASSIGNING, None))

    def test_disabled_makes_no_colab_calls(self) -> None:
        state = replace(_idle(), status=DISABLED, disabled_reason="x", reconciled_day="")
        self.assertEqual(tick(state).tasks, ())

    def test_disabling_a_session_that_holds_a_runtime_releases_it_first(self) -> None:
        # The endpoint leaves the state with the session; unreleased, the A100
        # would bill until the VM's watchdog gives up on it.
        refused = CredentialOrPolicyError("invalid_grant")
        for path, decision in {
            "heartbeat": after_heartbeat(_ready(), NBH, "ep-1", refused, NOW, CFG),
            "launch": after_launch(_booting(boot_step="assigned"), NBH, "ep-1", refused, NOW, CFG),
            "a stale assign": after_assign(_ready(), "old-nbh", refused, NOW, CFG),
            "draining": after_heartbeat(_draining(), NBH, "ep-1", refused, NOW, CFG),
        }.items():
            with self.subTest(path):
                self.assertEqual((decision.state.status, decision.state.endpoint), (DISABLED, ""))
                self.assertEqual(decision.tasks, (("unassign", {"endpoint": "ep-1"}),))
                self.assertIn("ep-1", decision.state.known_endpoints)


class LostRuntimeTests(unittest.TestCase):
    """A runtime that vanishes before READY is a boot failure (its ``failed`` may be lost)."""

    def test_every_way_of_noticing_counts_before_ready(self) -> None:
        state = _booting(failures=1, last_demand_at=NOW - 600)
        noticed = {
            "reconcile": seen(state, runtimes=[]).state,
            "launch": after_launch(
                replace(state, boot_step="assigned", kernel_id=""), NBH, "ep-1", None, NOW, CFG
            ).state,
            "heartbeat": after_heartbeat(state, NBH, "ep-1", None, NOW, CFG).state,
        }
        for path, after in noticed.items():
            with self.subTest(path):
                self.assertEqual((after.status, after.failures), (IDLE, 2))
                self.assertIn("gone", after.last_error)

    def test_the_third_loss_in_a_row_disables_instead_of_starting_again(self) -> None:
        # Fresh demand would otherwise assign another A100 in the same pass.
        state = _booting(failures=2, last_demand_at=NOW - 10)
        for path, decision in {
            "reconcile": seen(state, runtimes=[], ccu=CcuInfo(current_balance=80.0)),
            "launch": after_launch(replace(state, boot_step="assigned"), NBH, "ep-1", None, NOW, CFG),
            "heartbeat": after_heartbeat(state, NBH, "ep-1", None, NOW, CFG),
        }.items():
            with self.subTest(path):
                self.assertEqual((decision.state.status, decision.tasks), (DISABLED, ()))
                self.assertIn("連續 3 次開機失敗", decision.state.disabled_reason)

    def test_losing_a_ready_or_draining_runtime_is_not_a_boot_failure(self) -> None:
        for state in (_ready(failures=0), _draining(failures=1, last_error="!colab off")):
            state = replace(state, last_demand_at=NOW - 600)
            with self.subTest(status=state.status):
                after = seen(state, runtimes=[]).state
                self.assertEqual((after.status, after.failures), (IDLE, state.failures))
        self.assertEqual(after.last_error, "!colab off")
        self.assertEqual(after_heartbeat(_ready(), NBH, "ep-1", None, NOW, CFG).state.failures, 0)


class UnansweredAssignTests(unittest.TestCase):
    """An nbh whose assign never answered is looked up, and its runtime released."""

    def _cooled(self) -> ColabState:
        return replace(
            _idle(last_demand_at=None),
            status=COOLDOWN,
            cooldown_until=NOW + 600,
            unanswered_nbhs=("nbh-a",),
            unanswered_at=NOW,
            reconciled_day=taipei_day(NOW),
        )

    def test_the_lookup_waits_then_rides_on_a_reconcile(self) -> None:
        state = self._cooled()
        self.assertEqual(tick(state, now=NOW + 30).tasks, ())
        self.assertEqual(actions(tick(state, now=NOW + 60)), ["reconcile"])
        # IDLE too; and once looked up, back to one reconcile a day.
        idle = replace(state, status=IDLE, cooldown_until=None)
        self.assertEqual(actions(tick(idle, now=NOW + 60)), ["reconcile"])
        self.assertEqual(tick(replace(idle, unanswered_nbhs=(), unanswered_at=None), now=NOW + 60).tasks, ())

    def test_a_runtime_found_for_it_is_ours_and_released(self) -> None:
        # The listing alone calls it foreign; the lookup names it as ours.
        self.assertEqual(seen(self._cooled(), runtimes=[runtime("ep-orphan")], now=NOW + 70).foreign, ("ep-orphan",))
        obs = Observations(NOW + 70, lookups=(("nbh-a", "ep-orphan"),), assignments=(runtime("ep-orphan"),))
        decision = decide(self._cooled(), obs, CFG)

        self.assertEqual(decision.tasks, (("unassign", {"endpoint": "ep-orphan", "orphan": True}),))
        self.assertEqual(decision.foreign, ())
        self.assertEqual((decision.state.unanswered_nbhs, decision.state.unanswered_at), ((), None))
        self.assertIn("ep-orphan", decision.state.known_endpoints)
        self.assertEqual(decision.state.status, COOLDOWN)

    def test_a_lookup_that_keeps_failing_stops_queuing_reconciles_of_its_own(self) -> None:
        # A failed lookup leaves the nbh waiting (below): retried every minute
        # for a while, then only with the reconciles that run anyway.
        state = self._cooled()
        self.assertEqual(actions(tick(state, now=NOW + UNANSWERED_RETRY_SECONDS)), ["reconcile"])
        later = NOW + UNANSWERED_RETRY_SECONDS + 60
        self.assertEqual(tick(state, now=later).tasks, ())
        self.assertEqual(tick(replace(state, status=IDLE, cooldown_until=None), now=later).tasks, ())
        # Still looked up: by the next day's reconcile, or a session's.
        self.assertEqual(lookups_due(state, later), ("nbh-a",))
        self.assertEqual(actions(tick(state, now=NOW + 86400)), ["reconcile"])

    def test_no_runtime_found_clears_it_and_a_failed_lookup_keeps_it(self) -> None:
        state = replace(self._cooled(), unanswered_nbhs=("nbh-a", "nbh-b"))
        decision = decide(state, Observations(NOW + 70, lookups=(("nbh-a", ""),)), CFG)
        self.assertEqual((decision.tasks, decision.state.unanswered_nbhs), ((), ("nbh-b",)))
        self.assertEqual(decision.state.unanswered_at, NOW)

    def test_the_current_session_is_never_looked_up(self) -> None:
        # The same nbh being assigned again is the retry's business.
        state = replace(_assigning(), unanswered_nbhs=(NEW_NBH,), unanswered_at=NOW - 600)
        decision = decide(state, Observations(NOW, lookups=((NEW_NBH, "ep-9"),)), CFG)
        self.assertEqual((decision.tasks, decision.state.unanswered_nbhs), ((), (NEW_NBH,)))

    def test_it_is_looked_up_even_during_the_next_session(self) -> None:
        state = _ready(unanswered_nbhs=("nbh-a",), unanswered_at=NOW - 600)
        obs = Observations(NOW, lookups=(("nbh-a", "ep-orphan"),), assignments=(runtime("ep-1"), runtime("ep-orphan")))
        decision = decide(state, obs, CFG)
        self.assertEqual(
            [task for task in decision.tasks if task[0] == "unassign"],
            [("unassign", {"endpoint": "ep-orphan", "orphan": True})],
        )
        self.assertEqual(decision.state.status, READY)

    def test_the_waiting_list_is_bounded(self) -> None:
        state = replace(_assigning(), unanswered_nbhs=tuple(f"nbh-{i}" for i in range(9)))
        cooled = tick(replace(state, attempts=MAX_ASSIGN_ATTEMPTS, action_at=None), now=NOW + 60).state
        self.assertEqual(len(cooled.unanswered_nbhs), 5)
        self.assertEqual(cooled.unanswered_nbhs[-1], NEW_NBH)


# ── compute units ────────────────────────────────────────────


class ComputeUnitTests(unittest.TestCase):
    def test_estimated_until_ccu_info_reports_our_runtime(self) -> None:
        state = _ready(cu_used_today=1.0, cu_accounted_at=NOW - 360, ccu_rate=0.0, ccu_checked_at=NOW - 2000)

        estimated = tick(state).state
        self.assertAlmostEqual(estimated.cu_used_today, 1.0 + ESTIMATED_A100_CU_PER_HOUR * 0.1)
        self.assertEqual(estimated.cu_accounted_at, NOW)

        measured = seen(estimated, ccu=CcuInfo(current_balance=90.0, consumption_rate_hourly=6.0), now=NOW + 1).state
        after = tick(measured, now=NOW + 1 + 360).state
        self.assertAlmostEqual(after.cu_used_today - measured.cu_used_today, 0.6)

    def test_a_dry_run_estimates_the_cpu_runtime(self) -> None:
        state = _ready(cu_used_today=0.0, cu_accounted_at=NOW - 3600, ccu_checked_at=None)
        self.assertAlmostEqual(tick(state, ControllerConfig(dry_run=True)).state.cu_used_today, ESTIMATED_CPU_CU_PER_HOUR)

    def test_nothing_is_counted_without_a_runtime(self) -> None:
        state = ColabState(**_ccu(cu_used_today=2.0, cu_accounted_at=NOW - 3600))
        self.assertEqual(tick(state).state.cu_used_today, 2.0)

    def test_a_new_taipei_day_starts_from_zero(self) -> None:
        yesterday = taipei_day(NOW - 86400)
        state = _ready(cu_day=yesterday, cu_used_today=29.0, starts_today=4, cu_accounted_at=NOW - 60)
        state = replace(state, ccu_rate=None)

        today = tick(state).state

        self.assertEqual((today.cu_day, today.starts_today), (taipei_day(NOW), 0))
        self.assertLess(today.cu_used_today, 1.0)

    def test_the_cap_reached_mid_session_releases_it(self) -> None:
        state = _ready(cu_used_today=29.99, cu_accounted_at=NOW - 3600, ccu_checked_at=NOW - 10, ccu_rate=5.0)
        self.assertEqual(tick(state).state.status, DRAINING)


# ── heartbeat outcomes ───────────────────────────────────────


class HeartbeatTests(unittest.TestCase):
    def test_the_vm_state_line_is_parsed(self) -> None:
        stdout = f"noise\n{HEARTBEAT_MARKER}{{\"step\": \"pull\", \"ts\": 5, \"error\": null}}\n"
        self.assertEqual(vm_state(stdout), {"step": "pull", "ts": 5, "error": None})
        self.assertEqual(vm_state("nothing"), {})
        self.assertEqual(vm_state(f"{HEARTBEAT_MARKER}not json"), {})
        # The cell prints at most 300 characters of the file, never more.
        self.assertIn("[:300]", HEARTBEAT_CODE)

    def test_a_heartbeat_follows_the_boot_and_keeps_new_kernel_ids(self) -> None:
        state = _booting(boot_step="launched")
        cell = CellResult("ok", "kernel-2", "session-2", stdout=f'{HEARTBEAT_MARKER}{{"step": "pull", "error": null}}\n')

        after = after_heartbeat(state, NBH, "ep-1", cell, NOW, CFG).state

        self.assertEqual((after.boot_step, after.kernel_id, after.session_id), ("pull", "kernel-2", "session-2"))
        error_cell = CellResult("ok", "kernel-1", "session-1", stdout=f'{HEARTBEAT_MARKER}{{"step": "tunnel", "error": "no host"}}')
        self.assertIn("VM tunnel: no host", after_heartbeat(state, NBH, "ep-1", error_cell, NOW, CFG).state.last_error)

    def test_heartbeat_failures(self) -> None:
        state = _ready(cu_accounted_at=NOW)
        self.assertEqual(after_heartbeat(state, NBH, "ep-1", None, NOW, CFG).state.status, IDLE)
        # Retried at the next interval; nothing changes meanwhile.
        self.assertEqual(after_heartbeat(state, NBH, "ep-1", TransientControlPlaneError("x"), NOW, CFG).state, state)
        self.assertEqual(after_heartbeat(state, NBH, "ep-1", CellResult(CELL_ERROR, "kernel-1"), NOW, CFG).state, state)
        self.assertEqual(
            after_heartbeat(state, NBH, "ep-1", CredentialOrPolicyError("invalid_grant"), NOW, CFG).state.status,
            DISABLED,
        )
        # A heartbeat for another session changes nothing.
        self.assertEqual(after_heartbeat(state, "other", "ep-1", None, NOW, CFG).state, state)


# ── the service ──────────────────────────────────────────────


class _Harness:
    """A controller over an in-memory store, a fake control plane, a fake clock,
    and a queue that records tasks for the test to run like Cloud Tasks."""

    def __init__(self, state: ColabState | None = None, *, cfg: ControllerConfig = CFG, launcher: Any = None) -> None:
        self.clock = Clock(NOW)
        self.store = InMemoryColabStateStore(state or ColabState())
        self.tunnel = FakeTunnel()
        self.endpoint, _ = make_endpoint(store=self.store, tunnel=self.tunnel, clock=self.clock)
        self.cp = FakeControlPlane()
        self.queued: list[tuple[str, dict]] = []
        self.launcher = launcher or _Launcher()
        nbhs = iter(f"00000000-0000-4000-8000-{index:012d}" for index in range(1, 100))
        self.controller = ColabController(
            self.endpoint,
            self.cp,
            self.launcher,
            submit=self.submit,
            config=cfg,
            clock=self.clock,
            new_nbh=lambda: next(nbhs),
        )

    async def submit(self, kind: str, payload: dict) -> None:
        self.queued.append((kind, payload))

    def actions(self) -> list[str]:
        return [payload["action"] for _, payload in self.queued]

    async def run_queued(self) -> list[str]:
        """Run every queued task (and the ones they queue), like Cloud Tasks."""
        ran = []
        while self.queued:
            kind, payload = self.queued.pop(0)
            assert kind == TASK_KIND
            ran.append(payload["action"])
            self.assert_ok(await self.controller.run_action(payload["action"], payload))
        return ran

    @staticmethod
    def assert_ok(ok: bool) -> None:
        assert ok, "the task did not complete"

    async def close(self) -> None:
        await self.endpoint.close()


class _Launcher:
    """Records launches; succeeds unless told otherwise."""

    def __init__(self) -> None:
        self.launches: list[tuple[str, str]] = []
        self.error: Exception | None = None

    async def launch(self, runtime_, nbh: str) -> LaunchResult:
        self.launches.append((runtime_.endpoint, nbh))
        if self.error is not None:
            raise self.error
        return LaunchResult("kernel-1", "session-1", 4242)


class ControllerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.h = _Harness(ColabState(last_demand_at=NOW - 5))
        self.addAsyncCleanup(self.h.close)

    async def test_a_whole_session_from_demand_to_release(self) -> None:
        h = self.h
        # Demand, no ccu read yet: the tick asks a reconcile to read it.
        summary = await h.controller.tick()
        self.assertEqual((summary["status"], summary["queued"]), (IDLE, ["reconcile"]))
        self.assertEqual(h.cp.calls, [])

        ran = await h.run_queued()
        self.assertEqual(ran, ["reconcile", "assign", "launch"])
        state = h.store.cached
        self.assertEqual((state.status, state.endpoint, state.kernel_id, state.starts_today), (BOOTING, "ep-1", "kernel-1", 1))
        self.assertEqual(h.cp.calls[2], ("assign", "00000000-0000-4000-8000-000000000002", False))
        self.assertEqual(h.launcher.launches, [("ep-1", state.nbh)])

        # The VM registers (the registrar's write), then the minutes go by.
        await h.store.update(lambda s: s.with_status(READY, h.clock.now, tunnel_url=TUNNEL, ready_at=h.clock.now, last_health_at=h.clock.now))
        h.clock.now += 300
        await h.store.update(lambda s: replace(s, last_demand_at=h.clock.now))
        summary = await h.controller.tick()
        self.assertEqual(summary["queued"], ["heartbeat", "reconcile"])
        await h.run_queued()
        cells = [call for call in h.cp.calls if call[0] == "run_cell"]
        self.assertEqual(cells[-1][2:5], (HEARTBEAT_CODE, "kernel-1", "session-1"))
        # The reconcile's health check went through the tunnel and was kept.
        self.assertIn("GET /api/version", h.tunnel.paths())
        self.assertEqual(h.store.cached.last_health_at, h.clock.now)
        self.assertTrue(h.endpoint.is_ready())

        # Nobody uses it for 15 minutes: released.
        h.clock.now += 901
        await h.controller.tick()
        self.assertEqual(h.store.cached.status, DRAINING)
        self.assertFalse(h.endpoint.is_ready())
        await h.run_queued()
        self.assertEqual((h.store.cached.status, h.cp.runtimes), (IDLE, {}))
        self.assertGreater(h.store.cached.cu_used_today, 0)

    async def test_the_tick_never_calls_colab(self) -> None:
        h = self.h
        for state in (_idle(), _assigning(), _booting(), _ready(), _draining(), replace(_idle(), status=COOLDOWN, cooldown_until=NOW + 60)):
            await h.store.update(lambda s, state=state: state)
            await h.controller.tick()
            with self.subTest(status=state.status):
                self.assertEqual(h.cp.calls, [])

    async def test_the_tick_is_bounded_when_the_state_store_hangs(self) -> None:
        async def hang(change):
            await asyncio.sleep(30)

        self.h.store.update = hang
        started = time.monotonic()
        with patch("src.services.colab.controller.TICK_STATE_TIMEOUT_SECONDS", 0.2):
            summary = await self.h.controller.tick()
        self.assertEqual(summary, {"error": "state_timeout"})
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(self.h.queued, [])

    async def test_an_unreadable_state_is_reported_not_raised(self) -> None:
        async def broken(change):
            raise ColabStateError("firestore down")

        self.h.store.update = broken
        self.assertEqual(await self.h.controller.tick(), {"error": "state_unavailable"})
        self.h.store.load = lambda: broken(None)
        self.assertFalse(await self.h.controller.run_action("reconcile", {}))

    async def test_a_lost_queue_is_logged_and_the_lease_queues_again(self) -> None:
        h = self.h
        await h.store.update(lambda s: _idle())

        async def down(kind, payload):
            raise RuntimeError("cloud tasks down")

        h.controller._submit = down
        with self.assertLogs("linebot", "WARNING"):
            summary = await h.controller.tick()
        self.assertEqual((summary["status"], summary["queued"]), (ASSIGNING, []))
        h.controller._submit = h.submit
        h.clock.now += LEASE_SECONDS["assign"] + 1
        await h.controller.tick()
        self.assertEqual(h.actions(), ["assign"])

    async def test_same_nbh_retry_after_an_assign_time_out(self) -> None:
        h = self.h
        await h.store.update(lambda s: _idle())
        h.cp.assign_results = [ControlPlaneTimeoutError("assign: no answer within 120 s")]
        await h.controller.tick()
        await h.run_queued()
        self.assertEqual(h.store.cached.status, ASSIGNING)
        h.clock.now += 60
        await h.controller.tick()
        await h.run_queued()
        assigns = [call[1] for call in h.cp.calls if call[0] == "assign"]
        self.assertEqual(len(assigns), 2)
        self.assertEqual(assigns[0], assigns[1])
        self.assertEqual(h.store.cached.status, BOOTING)

    async def test_a_runtime_left_by_timed_out_assigns_is_found_and_released(self) -> None:
        h = self.h
        await h.store.update(lambda s: _idle())
        # Every try times out, yet Colab assigned the nbh a runtime anyway.
        h.cp.timeouts_leave_runtimes = True
        h.cp.assign_results = [ControlPlaneTimeoutError("assign: no answer within 120 s")] * MAX_ASSIGN_ATTEMPTS
        for _ in range(MAX_ASSIGN_ATTEMPTS + 1):
            await h.controller.tick()
            await h.run_queued()
            h.clock.now += 60
        state = h.store.cached
        nbh = state.unanswered_nbhs[0]
        self.assertEqual((state.status, state.nbh, list(h.cp.runtimes)), (COOLDOWN, "", ["ep-1"]))
        # The listing alone cannot tell it from a browser's runtime.
        self.assertNotIn("ep-1", state.known_endpoints)

        await h.controller.tick()
        self.assertEqual(h.actions(), ["reconcile"])
        await h.run_queued()

        self.assertIn(("find_assignment", nbh, False), h.cp.calls)
        self.assertEqual(h.cp.runtimes, {})
        state = h.store.cached
        self.assertEqual((state.status, state.unanswered_nbhs), (COOLDOWN, ()))
        self.assertIn("ep-1", state.known_endpoints)

    async def test_a_failed_lookup_is_tried_again_and_a_dead_credential_disables(self) -> None:
        h = self.h
        await h.store.update(
            lambda s: replace(_idle(last_demand_at=None), unanswered_nbhs=(NEW_NBH,), unanswered_at=NOW - 120)
        )
        h.cp.find_error = TransientControlPlaneError("find_assignment failed: 502")
        self.assertTrue(await h.controller.run_action("reconcile", {}))
        self.assertEqual(h.store.cached.unanswered_nbhs, (NEW_NBH,))
        h.cp.find_error = None
        self.assertTrue(await h.controller.run_action("reconcile", {}))
        self.assertEqual(h.store.cached.unanswered_nbhs, ())

        await h.store.update(lambda s: replace(s, unanswered_nbhs=(NEW_NBH,), unanswered_at=NOW - 120))
        h.cp.find_error = CredentialOrPolicyError("find_assignment: invalid_grant")
        with self.assertLogs("linebot", "ERROR"):
            await h.controller.run_action("reconcile", {})
        self.assertEqual(h.store.cached.status, DISABLED)

    async def test_no_lookup_before_its_delay(self) -> None:
        h = self.h
        await h.store.update(lambda s: replace(_idle(last_demand_at=None), unanswered_nbhs=(NEW_NBH,), unanswered_at=NOW - 10))
        await h.controller.run_action("reconcile", {})
        self.assertNotIn("find_assignment", h.cp.names())
        self.assertEqual(h.store.cached.unanswered_nbhs, (NEW_NBH,))

    async def test_credential_errors_in_a_reconcile_disable_it(self) -> None:
        h = self.h
        await h.store.update(lambda s: _ready())
        h.cp.list_error = CredentialOrPolicyError("the Colab credential cannot be refreshed: invalid_grant")
        with self.assertLogs("linebot", "ERROR") as logs:
            self.assertTrue(await h.controller.run_action("reconcile", {}))
        self.assertIn("READY → DISABLED", "\n".join(logs.output))
        state = h.store.cached
        self.assertEqual(state.status, DISABLED)
        self.assertIn("invalid_grant", state.disabled_reason)
        self.assertFalse(h.endpoint.is_ready())

    async def test_a_reconcile_survives_one_failing_read(self) -> None:
        h = self.h
        await h.store.update(lambda s: _ready())
        h.cp.runtimes = {"ep-1": runtime("ep-1")}
        h.cp.ccu = TransientControlPlaneError("ccu_info failed: 502")
        self.assertTrue(await h.controller.run_action("reconcile", {}))
        self.assertEqual(h.store.cached.status, READY)
        h.cp.ccu = CcuInfo(current_balance=70.0)
        h.cp.list_error = TransientControlPlaneError("list failed: 503")
        self.assertTrue(await h.controller.run_action("reconcile", {}))
        self.assertEqual((h.store.cached.status, h.store.cached.ccu_balance), (READY, 70.0))

    async def test_a_foreign_runtime_is_logged_at_error_once(self) -> None:
        h = self.h
        await h.store.update(lambda s: _ready())
        h.cp.runtimes = {"ep-1": runtime("ep-1"), "ep-browser": runtime("ep-browser")}
        with self.assertLogs("linebot", "ERROR") as logs:
            await h.controller.run_action("reconcile", {})
            await h.controller.run_action("reconcile", {})
        self.assertEqual(sum("did not start" in line for line in logs.output), 1)
        self.assertNotIn("ep-browser", "\n".join(logs.output))
        self.assertEqual(h.queued, [])
        self.assertIn("ep-browser", h.cp.runtimes)

    async def test_orphans_of_ours_are_released_by_their_task(self) -> None:
        h = self.h
        await h.store.update(lambda s: _ready(known_endpoints=("ep-0", "ep-1")))
        h.cp.runtimes = {"ep-1": runtime("ep-1"), "ep-0": runtime("ep-0")}
        await h.controller.run_action("reconcile", {})
        self.assertEqual(h.actions(), ["unassign"])
        with self.assertLogs("linebot", "WARNING"):
            await h.run_queued()
        self.assertEqual(sorted(h.cp.runtimes), ["ep-1"])
        self.assertEqual(h.store.cached.status, READY)

    async def test_a_drain_after_the_vm_released_itself_ends_whatever_unassign_answers(self) -> None:
        h = self.h
        for refusal in (
            ControlPlaneError("unassign failed: 400", status=400),
            CredentialOrPolicyError("unassign failed: 403 Forbidden", status=403),
        ):
            with self.subTest(refusal=type(refusal).__name__):
                # The VM reported failed and released itself: Colab no longer lists it.
                await h.store.update(lambda s: _draining(last_demand_at=NOW - 600, failures=1))
                h.cp.runtimes, h.cp.unassign_error = {}, refusal
                for _ in range(3):
                    h.clock.now += 60
                    await h.controller.tick()
                    await h.run_queued()
                state = h.store.cached
                self.assertEqual((state.status, state.endpoint, state.disabled_reason), (IDLE, "", ""))

    async def test_a_runtime_held_when_a_reconcile_disables_the_tier_is_released(self) -> None:
        h = self.h
        await h.store.update(lambda s: _ready())
        h.cp.runtimes = {"ep-1": runtime("ep-1")}
        h.cp.list_error = CredentialOrPolicyError("list_assignments failed: 403 DENYLISTED", status=403)
        with self.assertLogs("linebot", "ERROR"):
            await h.controller.run_action("reconcile", {})
        self.assertEqual(h.actions(), ["unassign"])
        await h.run_queued()
        self.assertEqual((h.store.cached.status, h.cp.runtimes), (DISABLED, {}))

    async def test_a_disabled_tier_only_releases_runtimes(self) -> None:
        h = self.h
        await h.store.update(lambda s: replace(_idle(), status=DISABLED, disabled_reason="invalid_grant"))
        self.assertTrue(await h.controller.run_action("reconcile", {}))
        self.assertEqual(h.cp.calls, [])
        # An unassign queued by the disable (or just before it) still runs.
        h.cp.runtimes = {"ep-1": runtime("ep-1")}
        self.assertTrue(await h.controller.run_action("unassign", {"endpoint": "ep-1", "orphan": True}))
        self.assertEqual((h.cp.names(), h.cp.runtimes, h.store.cached.status), (["unassign"], {}, DISABLED))

    async def test_stale_tasks_do_nothing(self) -> None:
        h = self.h
        await h.store.update(lambda s: _ready())
        for action, payload in (
            ("assign", {"nbh": "old"}),
            ("launch", {"nbh": NBH, "endpoint": "ep-1"}),  # already launched
            ("heartbeat", {"nbh": "old", "endpoint": "ep-1"}),
        ):
            with self.subTest(action=action):
                self.assertTrue(await h.controller.run_action(action, payload))
        self.assertEqual(h.cp.calls, [])
        self.assertFalse(await h.controller.run_action("reboot", {}))
        self.assertFalse(await h.controller.run_action("unassign", {}))

    async def test_the_launch_task_uses_a_fresh_runtime_token_and_reports(self) -> None:
        h = self.h
        await h.store.update(lambda s: _booting(kernel_id="", boot_step="assigned", action="launch", action_at=NOW))
        h.cp.runtimes = {"ep-1": runtime("ep-1")}
        await h.controller.run_action("launch", {"nbh": NBH, "endpoint": "ep-1"})
        self.assertEqual(h.cp.names(), ["list_assignments"])
        self.assertEqual((h.store.cached.kernel_id, h.store.cached.boot_step), ("kernel-1", "launched"))

        # Gone before the launch: released at once.
        await h.store.update(lambda s: _booting(kernel_id="", boot_step="assigned", action="launch", action_at=NOW))
        h.cp.runtimes = {}
        await h.controller.run_action("launch", {"nbh": NBH, "endpoint": "ep-1"})
        self.assertEqual(h.store.cached.status, IDLE)

        # A launch error drains (a boot failure).
        await h.store.update(lambda s: _booting(kernel_id="", boot_step="assigned", action="launch", action_at=NOW))
        h.cp.runtimes = {"ep-1": runtime("ep-1")}
        h.launcher.error = LaunchError("bootstrap.py did not start")
        with self.assertLogs("linebot", "WARNING"):
            await h.controller.run_action("launch", {"nbh": NBH, "endpoint": "ep-1"})
        self.assertEqual((h.store.cached.status, h.store.cached.failures), (DRAINING, 1))
        self.assertEqual(h.actions(), ["unassign"])

    async def test_a_heartbeat_runs_in_the_saved_kernel(self) -> None:
        h = self.h
        await h.store.update(lambda s: _booting(boot_step="launched"))
        h.cp.runtimes = {"ep-1": runtime("ep-1")}
        h.cp.vm_state = {"step": "tunnel", "ts": 1, "error": None}
        await h.controller.run_action("heartbeat", {"nbh": NBH, "endpoint": "ep-1"})
        [cell] = [call for call in h.cp.calls if call[0] == "run_cell"]
        self.assertEqual((cell[3], cell[4], cell[5]), ("kernel-1", "session-1", False))
        self.assertEqual(h.store.cached.boot_step, "tunnel")

        h.cp.runtimes = {}
        await h.controller.run_action("heartbeat", {"nbh": NBH, "endpoint": "ep-1"})
        self.assertEqual(h.store.cached.status, IDLE)

    async def test_dry_run_assigns_a_cpu_runtime(self) -> None:
        h = _Harness(_idle(), cfg=ControllerConfig(dry_run=True))
        self.addAsyncCleanup(h.close)
        await h.controller.tick()
        await h.run_queued()
        self.assertEqual([call[2] for call in h.cp.calls if call[0] == "assign"], [True])

    async def test_colab_on_and_off_through_the_controller(self) -> None:
        for auto_start in (True, False):
            h = _Harness(ColabState(**_ccu()), cfg=ControllerConfig(auto_start=auto_start))
            self.addAsyncCleanup(h.close)
            await h.store.update(lambda s: turn_on(s, NOW))
            await h.controller.tick()
            with self.subTest(auto_start=auto_start):
                self.assertEqual(h.store.cached.status, ASSIGNING)
                await h.run_queued()
                self.assertEqual(h.store.cached.status, BOOTING)
                await h.store.update(lambda s: turn_off(s, NOW))
                await h.controller.tick()
                await h.run_queued()
                self.assertEqual((h.store.cached.status, h.cp.runtimes), (IDLE, {}))
                # Off stays off, even with demand.
                await h.store.update(lambda s: replace(s, last_demand_at=NOW, manual_demand_at=NOW))
                await h.controller.tick()
                self.assertEqual(h.store.cached.status, IDLE)
                self.assertNotIn("assign", h.actions())

    async def test_the_view_has_the_settings_and_the_last_tick(self) -> None:
        await self.h.controller.tick()
        view = self.h.controller.view()
        self.assertEqual((view["dry_run"], view["account_dedicated"]), (False, False))
        self.assertEqual(view["last_tick"]["at"], NOW)
        self.assertEqual(set(ACTIONS), {"assign", "launch", "heartbeat", "unassign", "reconcile"})


class ConfigTests(unittest.TestCase):
    def test_from_settings(self) -> None:
        from src.config import Settings

        settings = Settings(
            _env_file=None,
            model_path_preference="api",
            colab_idle_ttl_seconds=600,
            colab_boot_timeout_seconds=1200,
            colab_heartbeat_seconds=240,
            colab_account_dedicated=True,
            colab_dry_run=True,
        )
        cfg = ControllerConfig.from_settings(settings)
        self.assertEqual(
            (cfg.auto_start, cfg.idle_ttl_seconds, cfg.boot_timeout_seconds, cfg.heartbeat_seconds),
            (False, 600.0, 1200.0, 240.0),
        )
        self.assertEqual((cfg.account_dedicated, cfg.dry_run, cfg.min_cu_balance, cfg.daily_cu_cap), (True, True, 10.0, 30.0))


if __name__ == "__main__":
    unittest.main()
