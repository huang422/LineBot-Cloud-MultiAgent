"""Scheduler Service for scheduled group messages.

Drives the weekly / yearly LINE push reminders. Cloud Run scales to zero
between requests, so an in-process timer (APScheduler / asyncio.sleep)
cannot fire reliably while the instance is hibernated. The service
therefore exposes a pull-style ``dispatch_due_jobs`` that an external
trigger – Cloud Scheduler hitting ``/internal/cron`` once per minute –
calls to deliver any messages whose cron expression matches the current
minute in ``Asia/Taipei``.

``add_weekly_message`` / ``add_yearly_message`` keep the same signatures
the previous APScheduler-backed implementation used, and ``start`` /
``shutdown`` remain as no-ops so existing callers and tests continue to
work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

try:
    from zoneinfo import ZoneInfo
    from zoneinfo import ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - py<3.9 only
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment]

from src.utils.logger import logger


_DAY_NAME_TO_NUM = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


def _resolve_timezone(name: str):
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        logger.warning(f"Timezone {name!r} not available; falling back to UTC")
        return timezone.utc


@dataclass
class _WeeklyJob:
    id: str
    day_of_week: str
    hour: int
    minute: int
    group_id: str
    message: str

    @property
    def name(self) -> str:
        return f"Weekly: {self.message[:20]}"

    def trigger_label(self) -> str:
        return (
            f"cron[day_of_week='{self.day_of_week}', "
            f"hour='{self.hour}', minute='{self.minute}']"
        )

    def matches(self, dt: datetime) -> bool:
        return (
            dt.weekday() == _DAY_NAME_TO_NUM[self.day_of_week]
            and dt.hour == self.hour
            and dt.minute == self.minute
        )

    def next_run(self, ref: datetime) -> datetime:
        target_wd = _DAY_NAME_TO_NUM[self.day_of_week]
        candidate = ref.replace(
            hour=self.hour, minute=self.minute, second=0, microsecond=0,
        )
        days_ahead = (target_wd - ref.weekday()) % 7
        candidate = candidate + timedelta(days=days_ahead)
        if candidate <= ref:
            candidate = candidate + timedelta(days=7)
        return candidate


@dataclass
class _YearlyJob:
    id: str
    month: int
    day: int
    hour: int
    minute: int
    group_id: str
    message: str

    @property
    def name(self) -> str:
        return f"Yearly: {self.message[:20]}"

    def trigger_label(self) -> str:
        return (
            f"cron[month='{self.month}', day='{self.day}', "
            f"hour='{self.hour}', minute='{self.minute}']"
        )

    def matches(self, dt: datetime) -> bool:
        return (
            dt.month == self.month
            and dt.day == self.day
            and dt.hour == self.hour
            and dt.minute == self.minute
        )

    def next_run(self, ref: datetime) -> datetime:
        try:
            candidate = ref.replace(
                month=self.month, day=self.day,
                hour=self.hour, minute=self.minute,
                second=0, microsecond=0,
            )
        except ValueError:
            # Handles 02-29 leap-day rollovers by snapping to the next valid year.
            candidate = ref.replace(year=ref.year + 1)
            candidate = candidate.replace(
                month=self.month, day=self.day,
                hour=self.hour, minute=self.minute,
                second=0, microsecond=0,
            )
        if candidate <= ref:
            candidate = candidate.replace(year=ref.year + 1)
        return candidate


class SchedulerService:
    """Manages scheduled tasks dispatched by an external trigger."""

    def __init__(self, timezone_name: str = "Asia/Taipei") -> None:
        self._weekly: dict[str, _WeeklyJob] = {}
        self._yearly: dict[str, _YearlyJob] = {}
        self._timezone_name = timezone_name
        self._tz = _resolve_timezone(timezone_name)
        self._fire_lock = asyncio.Lock()
        self._recent_fires: dict[str, datetime] = {}
        logger.info(f"SchedulerService initialized (tz={timezone_name})")

    @property
    def timezone(self):
        return self._tz

    @property
    def timezone_name(self) -> str:
        return self._timezone_name

    def start(self) -> None:
        """No-op kept for backwards compatibility."""

    def shutdown(self) -> None:
        """No-op kept for backwards compatibility."""

    def add_weekly_message(
        self,
        job_id: str,
        day_of_week: str,
        hour: int,
        minute: int,
        group_id: str,
        message: str,
    ) -> bool:
        try:
            day = day_of_week.strip().lower()
            if day not in _DAY_NAME_TO_NUM:
                logger.error(f"Invalid day_of_week: {day_of_week!r}")
                return False
            self._weekly[job_id] = _WeeklyJob(
                id=job_id,
                day_of_week=day,
                hour=hour,
                minute=minute,
                group_id=group_id,
                message=message,
            )
            logger.info(
                f"Added weekly job: {job_id} ({day} {hour:02d}:{minute:02d})"
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to add weekly job: {exc}", exc_info=True)
            return False

    def add_yearly_message(
        self,
        job_id: str,
        month: int,
        day: int,
        hour: int,
        minute: int,
        group_id: str,
        message: str,
    ) -> bool:
        try:
            self._yearly[job_id] = _YearlyJob(
                id=job_id,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                group_id=group_id,
                message=message,
            )
            logger.info(
                f"Added yearly job: {job_id} "
                f"({month:02d}-{day:02d} {hour:02d}:{minute:02d})"
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to add yearly job: {exc}", exc_info=True)
            return False

    def remove_job(self, job_id: str) -> bool:
        if job_id in self._weekly:
            del self._weekly[job_id]
            logger.info(f"Removed job: {job_id}")
            return True
        if job_id in self._yearly:
            del self._yearly[job_id]
            logger.info(f"Removed job: {job_id}")
            return True
        logger.warning(f"Failed to remove job {job_id}: not found")
        return False

    def list_jobs(self) -> list[dict]:
        now = datetime.now(self._tz)
        jobs: list[dict] = []
        for job in self._weekly.values():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run(now).isoformat(),
                "trigger": job.trigger_label(),
            })
        for job in self._yearly.values():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run(now).isoformat(),
                "trigger": job.trigger_label(),
            })
        return jobs

    def get_stats(self) -> dict:
        return {
            "running": True,
            "job_count": len(self._weekly) + len(self._yearly),
            "jobs": self.list_jobs(),
        }

    async def dispatch_due_jobs(
        self,
        *,
        reference_time: Optional[datetime] = None,
        sender: Optional[Callable[[str, str], Awaitable[bool]]] = None,
    ) -> dict:
        if reference_time is None:
            reference_time = datetime.now(self._tz)
        elif reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=self._tz)
        else:
            reference_time = reference_time.astimezone(self._tz)

        if sender is None:
            from src.services.line_service import get_line_service

            line = get_line_service()

            async def _default_sender(group_id: str, message: str) -> bool:
                return await line.push_text(to=group_id, text=message)

            sender = _default_sender

        minute_key = reference_time.strftime("%Y%m%d%H%M")
        fired: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        due: list[tuple[str, str, str]] = []  # (job_id, group_id, message)

        async with self._fire_lock:
            cutoff = reference_time - timedelta(minutes=5)
            for stale_key in [k for k, ts in self._recent_fires.items() if ts < cutoff]:
                del self._recent_fires[stale_key]

            for weekly in self._weekly.values():
                if not weekly.matches(reference_time):
                    continue
                dedupe_key = f"{weekly.id}:{minute_key}"
                if dedupe_key in self._recent_fires:
                    skipped.append(weekly.id)
                    continue
                self._recent_fires[dedupe_key] = reference_time
                due.append((weekly.id, weekly.group_id, weekly.message))

            for yearly in self._yearly.values():
                if not yearly.matches(reference_time):
                    continue
                dedupe_key = f"{yearly.id}:{minute_key}"
                if dedupe_key in self._recent_fires:
                    skipped.append(yearly.id)
                    continue
                self._recent_fires[dedupe_key] = reference_time
                due.append((yearly.id, yearly.group_id, yearly.message))

        for job_id, group_id, message in due:
            try:
                ok = await sender(group_id, message)
            except Exception as exc:
                failed.append(job_id)
                logger.error(
                    f"Scheduled send [{job_id}] error: {exc}", exc_info=True
                )
                continue
            if ok:
                fired.append(job_id)
                logger.info(f"Scheduled send [{job_id}] ok")
            else:
                failed.append(job_id)
                logger.error(f"Scheduled send [{job_id}] failed")

        return {
            "reference_time": reference_time.isoformat(),
            "fired": fired,
            "failed": failed,
            "skipped_duplicate": skipped,
        }


# Global singleton
_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service(timezone_name: str = "Asia/Taipei") -> SchedulerService:
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService(timezone_name=timezone_name)
    return _scheduler_service


def peek_scheduler_service() -> SchedulerService | None:
    return _scheduler_service


def close_scheduler_service() -> None:
    global _scheduler_service
    if _scheduler_service:
        _scheduler_service.shutdown()
        _scheduler_service = None
