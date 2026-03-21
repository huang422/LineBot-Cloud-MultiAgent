"""Scheduler Service for scheduled group messages.

Handles recurring tasks like:
- Weekly reminder messages
- Yearly birthday messages
- Custom scheduled notifications

Uses APScheduler with AsyncIO support, timezone Asia/Taipei.
"""

from __future__ import annotations

from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.utils.logger import logger


class SchedulerService:
    """Manages scheduled tasks using APScheduler."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        logger.info("SchedulerService initialized")

    def start(self) -> None:
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("Scheduler started")

    def shutdown(self) -> None:
        if self.is_running:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("Scheduler stopped")

    def add_weekly_message(
        self,
        job_id: str,
        day_of_week: str,
        hour: int,
        minute: int,
        group_id: str,
        message: str,
    ) -> bool:
        """Add a weekly scheduled message.

        Args:
            job_id: Unique identifier for this job
            day_of_week: Day name (mon, tue, wed, thu, fri, sat, sun)
            hour: Hour (0-23)
            minute: Minute (0-59)
            group_id: LINE group ID to send message to
            message: Message text to send
        """
        try:
            from src.services.line_service import get_line_service

            async def send():
                line = get_line_service()
                logger.info(f"Scheduled send [{job_id}]: {message[:30]}")
                success = await line.push_text(to=group_id, text=message)
                if not success:
                    logger.error(f"Scheduled message failed: {job_id}")

            trigger = CronTrigger(
                day_of_week=day_of_week,
                hour=hour,
                minute=minute,
                timezone="Asia/Taipei",
            )

            self.scheduler.add_job(
                send,
                trigger=trigger,
                id=job_id,
                name=f"Weekly: {message[:20]}",
                replace_existing=True,
            )

            logger.info(f"Added weekly job: {job_id} ({day_of_week} {hour:02d}:{minute:02d})")
            return True
        except Exception as e:
            logger.error(f"Failed to add weekly job: {e}", exc_info=True)
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
        """Add a yearly scheduled message (e.g. birthday).

        Args:
            job_id: Unique identifier for this job
            month: Month (1-12)
            day: Day of month (1-31)
            hour: Hour (0-23)
            minute: Minute (0-59)
            group_id: LINE group ID
            message: Message text
        """
        try:
            from src.services.line_service import get_line_service

            async def send():
                line = get_line_service()
                logger.info(f"Scheduled send [{job_id}]: {message[:30]}")
                success = await line.push_text(to=group_id, text=message)
                if not success:
                    logger.error(f"Scheduled message failed: {job_id}")

            trigger = CronTrigger(
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                timezone="Asia/Taipei",
            )

            self.scheduler.add_job(
                send,
                trigger=trigger,
                id=job_id,
                name=f"Yearly: {message[:20]}",
                replace_existing=True,
            )

            logger.info(f"Added yearly job: {job_id} ({month:02d}-{day:02d} {hour:02d}:{minute:02d})")
            return True
        except Exception as e:
            logger.error(f"Failed to add yearly job: {e}", exc_info=True)
            return False

    def remove_job(self, job_id: str) -> bool:
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove job {job_id}: {e}")
            return False

    def list_jobs(self) -> list[dict]:
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
        return jobs

    def get_stats(self) -> dict:
        return {
            "running": self.is_running,
            "job_count": len(self.scheduler.get_jobs()),
            "jobs": self.list_jobs(),
        }


# Global singleton
_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service


def peek_scheduler_service() -> SchedulerService | None:
    return _scheduler_service


def close_scheduler_service() -> None:
    global _scheduler_service
    if _scheduler_service:
        _scheduler_service.shutdown()
        _scheduler_service = None
