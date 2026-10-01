"""Housekeeping that rides on the per-minute ``/internal/cron`` request.

Cloud Scheduler already wakes the service every minute (ADR-0001), and that
request has CPU, so it is where time-based upkeep lives now that the app has
no timers of its own. Three duties: flushing passive group buffers that never
reached their threshold (ADR-0006), queueing the daily model end-of-life
check, an ``eol`` task, once per Asia/Taipei day (ADR-0007), and the Colab
controller's tick (a state read, a decision, ``colab`` tasks queued; its
Colab calls run in those tasks, ADR-0010).

Everything here is bounded (a key limit and a time budget) and swallows its
own errors, because the cron endpoint must answer well inside Cloud
Scheduler's 60 s attempt deadline and must never be retried. The budget is
checked *between* chats, never enforced by cancelling a flush half-way: a
cancelled write could leave a chat's batch neither stored nor buffered.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

from src.utils.logger import logger

# Well inside the Cloud Scheduler job's --attempt-deadline=60s.
CRON_MAINTENANCE_BUDGET_SECONDS = 30.0
# The day the end-of-life check belongs to.
LIFECYCLE_TIMEZONE = ZoneInfo("Asia/Taipei")


@dataclass
class CronDeps:
    memory: Any
    budget_seconds: float = CRON_MAINTENANCE_BUDGET_SECONDS
    # Must match the clock ``flush_due_passive`` compares its deadline with.
    clock: Callable[[], float] = monotonic
    # The model lifecycle checker (``claim_day``) and the task submitter;
    # without both, no ``eol`` task is queued.
    lifecycle: Any = None
    submit: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None
    # The Colab controller (src/services/colab/controller.py); None without Colab.
    colab: Any = None


def lifecycle_day(reference_time: datetime | None = None) -> date:
    """The Asia/Taipei date of *reference_time* (now when None)."""
    moment = reference_time or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(LIFECYCLE_TIMEZONE).date()


async def run_cron_tick(deps: CronDeps, reference_time: datetime | None = None) -> dict[str, Any]:
    """Run the maintenance duties once; returns a summary for the response.

    ``reference_time`` is the Cloud Scheduler minute. Buffer ages are
    measured on wall-clock time instead, since the scheduled minute can lag
    a cold start. No new chat is flushed after ``budget_seconds``; the rest
    are due again next minute.
    """
    flushed: list[str] = []
    errors: list[str] = []
    colab: dict[str, Any] | None = None
    if deps.colab is not None:
        # Bounded inside (its state I/O); its tasks are only queued (a
        # Cloud Tasks call each), never run here (main._queue_colab_task).
        try:
            colab = await deps.colab.tick()
        except Exception as exc:
            errors.append(f"colab_tick_failed:{type(exc).__name__}")
            logger.error(f"Cron maintenance: Colab tick failed: {exc}", exc_info=True)
    try:
        flushed = await deps.memory.flush_due_passive(
            deadline=deps.clock() + deps.budget_seconds
        )
    except Exception as exc:
        errors.append(f"passive_flush_failed:{type(exc).__name__}")
        logger.error(f"Cron maintenance: passive flush failed: {exc}", exc_info=True)

    if flushed:
        logger.info(f"Cron maintenance: flushed {len(flushed)} passive buffer(s)")

    eol_queued = False
    if deps.lifecycle is not None and deps.submit is not None:
        if deps.lifecycle.claim_day(lifecycle_day(reference_time)):
            try:
                await deps.submit("eol", {})
                eol_queued = True
            except Exception as exc:
                errors.append(f"eol_enqueue_failed:{type(exc).__name__}")
                logger.error(f"Cron maintenance: could not queue the eol check: {exc}", exc_info=True)
    return {
        "passive_flushed": len(flushed),
        "eol_check_queued": eol_queued,
        "colab": colab,
        "maintenance_errors": errors,
    }
