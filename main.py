"""LineBot-CloudAgent: Multi-Agent LINE Bot on free cloud resources.

FastAPI transport layer. Assembly lives in :mod:`src.runtime.composition`,
request handling in :mod:`src.pipeline.event_pipeline`; this module owns
the HTTP surface and the process lifecycle and little else.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from src.config import get_settings
from src.handlers.webhook_handler import extract_text, should_handle
from src.pipeline.event_pipeline import PipelineDeps, process_event, record_group_message
from src.pipeline.event_pipeline import (  # noqa: F401
    apply_cleaned_text as _apply_cleaned_text_impl,
)
from src.pipeline.event_pipeline import (
    get_request_block_message as _get_request_block_message,
)
from src.runtime.composition import build_container, wire_tool_registry

# Re-exported so existing callers and tests keep their import paths.
from src.runtime.composition import (  # noqa: F401
    build_text_agent_targets as _build_text_agent_targets,
)
from src.runtime.composition import (
    build_vision_agent_targets as _build_vision_agent_targets,
)
from src.runtime.composition import (
    register_scheduled_jobs as _register_scheduled_jobs,
)
from src.services.line_service import close_line_service, get_line_service
from src.services.memory_service import (
    close_memory_service,
    configure_memory_service,
    get_memory_service,
)
from src.services.message_cache_service import get_message_cache_service
from src.utils.logger import logger, setup_logger
from src.utils.rate_tracker import RateTracker
from src.utils.validators import validate_signature

# ── Process-wide state ───────────────────────────────────────

rate_tracker = RateTracker()
container = None

# Background tasks (prevent garbage collection of fire-and-forget tasks)
_background_tasks: set[asyncio.Task] = set()


def _apply_cleaned_text(request, event: dict) -> None:
    """Thin wrapper so ``extract_text`` stays patchable on this module."""
    _apply_cleaned_text_impl(request, event, extract_text)


def _build_pipeline_deps() -> PipelineDeps:
    return PipelineDeps(
        line=get_line_service(),
        memory=get_memory_service(),
        message_cache=get_message_cache_service(),
        orchestrator=container.orchestrator if container else None,
        get_agent=_get_agent,
    )


def _get_agent(name: str):
    return container.get_agent(name) if container else None


async def _process_event(event: dict) -> None:
    await process_event(event, _build_pipeline_deps())


def _record_group_message(event: dict) -> None:
    """Buffer an untriggered group message; resolving the service is our job."""
    record_group_message(event, get_memory_service())


# ── Lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global container

    settings = get_settings()
    setup_logger(level=settings.log_level)

    if not settings.line_channel_secret or not settings.line_channel_access_token:
        logger.warning("LINE channel credentials are incomplete")
    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not set — routing and fallback may fail")

    container = build_container(settings, rate_tracker)

    get_line_service()
    if settings.line_push_fallback_enabled and settings.line_push_monthly_limit > 0:
        logger.warning(
            "LINE reply-first then push-fallback is enabled. "
            f"Direct push budget={settings.line_push_monthly_limit}/month"
        )
    elif settings.line_push_fallback_enabled:
        logger.warning(
            "LINE reply-first then push-fallback is enabled. "
            "Direct push is enabled without a monthly cap"
        )
    else:
        logger.warning("LINE push fallback is disabled; reply failures cannot fall back to push")

    _start_scheduler(settings)

    configure_memory_service(settings, nvidia_provider=container.nvidia_provider)

    from src.services.embedding_service import (
        close_embedding_service,
        configure_embedding_service,
    )

    try:
        embedding_service = configure_embedding_service(settings)
        memory_service = get_memory_service()
        memory_service.set_embedding_service(embedding_service)
        wire_tool_registry(container, memory_service)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Memory/tool wiring skipped: {exc}")

    logger.info("LineBot-CloudAgent started (Orchestrator + NVIDIA Qwen3.5 architecture)")
    yield

    # Shutdown
    from src.services.scheduler_service import close_scheduler_service
    from src.services.web_search_service import close_web_search_service

    close_scheduler_service()
    await close_web_search_service()
    await close_line_service()
    await close_memory_service()
    try:
        await close_embedding_service()
    except Exception:  # pragma: no cover - shutdown best effort
        pass
    if container is not None:
        await container.aclose()
    logger.info("LineBot-CloudAgent stopped")


def _start_scheduler(settings) -> None:
    """Register scheduled jobs for /health visibility.

    Dispatch itself is pulled by Cloud Scheduler via /internal/cron: Cloud
    Run scales to zero and cannot fire in-process timers. See ADR-0001.
    """
    from src.services.scheduler_service import get_scheduler_service

    if not (settings.scheduled_messages_enabled and settings.scheduled_group_id):
        logger.info("Scheduler disabled (SCHEDULED_MESSAGES_ENABLED=false or no group ID)")
        return

    if not settings.line_push_fallback_enabled:
        logger.warning(
            "Scheduled messages requested but not registered because LINE push is disabled"
        )
        return

    scheduler = get_scheduler_service(settings.scheduled_messages_timezone)
    configured = len(settings.scheduled_weekly_messages) + len(settings.scheduled_yearly_messages)
    weekly, yearly = _register_scheduled_jobs(settings, scheduler, settings.scheduled_group_id)
    registered = weekly + yearly

    if configured == 0:
        logger.warning("Scheduled messages enabled but no scheduled jobs are configured")
        return
    if registered == 0:
        logger.warning("Scheduled messages enabled but no scheduled jobs could be registered")
        return

    if registered < configured:
        logger.warning(
            f"Some scheduled jobs could not be registered ({registered}/{configured})"
        )
    scheduler.start()
    if not settings.internal_cron_secret:
        logger.warning(
            "Scheduled jobs registered but INTERNAL_CRON_SECRET is empty; "
            "/internal/cron will refuse traffic. Set the secret and "
            "provision a Cloud Scheduler job pointing at /internal/cron."
        )
    logger.info(
        f"Scheduler ready with {len(scheduler.list_jobs())} job(s); "
        "dispatch is triggered by Cloud Scheduler hitting /internal/cron"
    )


app = FastAPI(title="LineBot-CloudAgent", lifespan=lifespan)


# ── Health ───────────────────────────────────────────────────


def _get_health_status(settings) -> tuple[str, bool, list[str]]:
    warnings: list[str] = []
    line_ready = bool(settings.line_channel_secret and settings.line_channel_access_token)
    openrouter_ready = bool(settings.openrouter_api_key)

    if not line_ready:
        warnings.append("LINE channel credentials are incomplete")
    if not openrouter_ready:
        warnings.append("OPENROUTER_API_KEY not set")
    if (
        settings.scheduled_messages_enabled
        and settings.scheduled_group_id
        and not settings.line_push_fallback_enabled
    ):
        warnings.append("Scheduled messages are configured but LINE push is disabled")

    ready_for_webhook = line_ready and openrouter_ready
    status = "healthy" if ready_for_webhook else "degraded"
    return status, ready_for_webhook, warnings


def _track_background_task(task: asyncio.Task) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_log_background_task_exception)
    task.add_done_callback(_background_tasks.discard)


def _log_background_task_exception(task: asyncio.Task) -> None:
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return

    if exc is not None:
        logger.error(
            f"Background task failed: {exc}",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


@app.get("/health")
async def health():
    settings = get_settings()
    status, ready_for_webhook, warnings = _get_health_status(settings)

    from src.services.scheduler_service import peek_scheduler_service
    from src.services.storage_service import get_storage_service
    from src.services.web_search_service import get_web_search_service

    scheduler = peek_scheduler_service()
    line = get_line_service()

    return {
        "status": status,
        "ready_for_webhook": ready_for_webhook,
        "warnings": warnings,
        "providers": {
            "line": bool(settings.line_channel_secret and settings.line_channel_access_token),
            "openrouter": bool(settings.openrouter_api_key),
            "nvidia": bool(settings.nvidia_api_key),
        },
        "models_status": rate_tracker.get_status(),
        "agents_stats": container.agents_stats() if container else {},
        "fallback_count": container.fallback_chain.fallback_count if container else 0,
        "memory": get_memory_service().get_stats(),
        "scheduler": (
            scheduler.get_stats()
            if scheduler is not None
            else {"running": False, "job_count": 0, "jobs": []}
        ),
        "cost_controls": {
            "line_push": line.get_push_stats(),
            "web_search": get_web_search_service().get_quota_stats(),
            "storage": get_storage_service().get_usage_stats(),
        },
    }


# ── Cloud Scheduler trigger ──────────────────────────────────


def _parse_cloud_scheduler_time(raw: str) -> datetime | None:
    """Parse the X-CloudScheduler-ScheduleTime header into a tz-aware datetime."""
    if not raw:
        return None
    raw = raw.strip()
    # Accept the RFC3339 form Cloud Scheduler sends: ``...Z`` or with offset.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@app.post("/internal/cron")
async def internal_cron(request: Request):
    """Dispatch any scheduled message jobs that match the current minute.

    Called by Cloud Scheduler every minute. Auth is a shared secret in
    ``X-Cron-Token`` (INTERNAL_CRON_SECRET). Always returns 200 with a
    structured summary so Cloud Scheduler does not retry on transient
    send failures. See ADR-0001.
    """
    settings = get_settings()
    expected = settings.internal_cron_secret
    if not expected:
        return Response(status_code=403)
    provided = request.headers.get("X-Cron-Token", "")
    if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        return Response(status_code=403)

    reference_time = _parse_cloud_scheduler_time(
        request.headers.get("X-CloudScheduler-ScheduleTime", "")
    )

    from src.services.scheduler_service import peek_scheduler_service

    scheduler = peek_scheduler_service()
    if scheduler is None:
        return JSONResponse(
            {"fired": [], "failed": [], "skipped_duplicate": [], "reason": "scheduler_not_initialized"}
        )

    return JSONResponse(await scheduler.dispatch_due_jobs(reference_time=reference_time))


# ── Webhook ──────────────────────────────────────────────────


@app.post("/webhook")
async def webhook(request: Request):
    settings = get_settings()

    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()

    if not validate_signature(body, signature, settings.line_channel_secret):
        return Response(status_code=403)

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Webhook received invalid JSON body")
        return Response(status_code=400)

    # Return 200 immediately and process in the background: LINE retries
    # aggressively on non-200, and a retry storm is worse than a late reply.
    for event in data.get("events", []):
        should_process = should_handle(event)

        if event.get("type") == "message" and not should_process:
            _record_group_message(event)
            _track_background_task(
                asyncio.create_task(
                    get_message_cache_service().cache_event_message(event)
                )
            )

        if should_process:
            _track_background_task(asyncio.create_task(_process_event(event)))

    return Response(status_code=200)


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
