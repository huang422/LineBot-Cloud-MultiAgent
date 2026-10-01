"""LineBot-CloudAgent: Multi-Agent LINE Bot on free cloud resources.

FastAPI transport layer. Assembly lives in :mod:`src.runtime.composition`,
request handling in :mod:`src.pipeline.event_pipeline`, post-webhook work in
:mod:`src.pipeline.task_handlers`; this module owns the HTTP surface and the
process lifecycle and little else.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from src.config import get_settings, legacy_config_warnings
from src.handlers.webhook_handler import should_handle
from src.pipeline.event_pipeline import PipelineDeps, record_group_message
from src.pipeline.maintenance import CronDeps, run_cron_tick
from src.pipeline.task_handlers import (
    TaskDeps,
    event_dedupe_key,
    handle_task,
    new_event_dedupe,
)
from src.runtime.composition import (
    build_colab_controller,
    build_compaction_scheduler,
    build_container,
    build_task_queue,
    register_scheduled_jobs,
    wire_tool_registry,
)
from src.services.colab.protocol import MAX_REGISTER_BYTES, SIGNATURE_HEADER
from src.services.line_service import close_line_service, get_line_service
from src.services.memory_service import (
    close_memory_service,
    configure_memory_service,
    get_memory_service,
)
from src.services.message_cache_service import get_message_cache_service
from src.services.task_queue import (
    BaseUrlResolver,
    TaskNotQueuedError,
    enqueue_only,
    enqueue_or_run,
)
from src.utils.logger import logger, setup_logger
from src.utils.validators import validate_signature

# ── Process-wide state ───────────────────────────────────────

container = None

# Two dedupe sets on purpose: the webhook drops LINE redeliveries, the task
# handler drops Cloud Tasks duplicates. One set would make every queued
# event look like a duplicate to its own task. See ADR-0006.
_webhook_seen = new_event_dedupe()
_task_seen = new_event_dedupe()


def _build_pipeline_deps() -> PipelineDeps:
    settings = get_settings()
    return PipelineDeps(
        line=get_line_service(),
        memory=get_memory_service(),
        message_cache=get_message_cache_service(),
        orchestrator=container.orchestrator if container else None,
        get_agent=_get_agent,
        reply_window_seconds=settings.reply_window_seconds,
        turn_deadline_seconds=settings.turn_deadline_seconds,
        colab=container.colab if container else None,
        admin_user_ids=settings.admin_user_ids,
        model_path_preference=settings.model_path_preference,
    )


def _get_agent(name: str):
    return container.get_agent(name) if container else None


def _record_group_message(event: dict) -> str | None:
    """Buffer an untriggered group message; returns a key due for flushing."""
    return record_group_message(event, get_memory_service())


async def _dispatch_task(kind: str, payload: dict) -> bool:
    """Run one task here and now (the inline queue and the inline fallback).

    Never raises, like ``handle_task``: a failure while building the deps
    is logged and reported as False.
    """
    try:
        deps = TaskDeps(
            pipeline=_build_pipeline_deps(),
            budget_seconds=get_settings().task_handler_budget_seconds,
            seen_events=_task_seen,
            lifecycle=container.model_lifecycle if container else None,
            colab=container.colab_controller if container else None,
        )
    except Exception as exc:
        logger.error(f"Task {kind} could not start: {exc}", exc_info=True)
        return False
    return await handle_task(kind, payload, deps)


async def _submit_task(kind: str, payload: dict) -> None:
    """Queue post-webhook work; runs it inline when it cannot be queued."""
    queue = container.task_queue if container is not None else None
    if queue is None:
        await _dispatch_task(kind, payload)
        return
    # When Cloud Tasks may have the task already, only an `event` runs here
    # too: its handler drops the second delivery (_task_seen); any other
    # kind would run twice.
    await enqueue_or_run(queue, kind, payload, _dispatch_task, inline_if_unsure=kind == "event")


async def _queue_colab_task(kind: str, payload: dict) -> None:
    """Queue a Colab controller task; never run it here (ADR-0010).

    Its Colab calls take seconds to minutes, too long for the cron request
    or the task that queues the next step. A task that cannot be queued
    raises; the controller logs it and its lease or next tick queues it again.
    """
    queue = container.task_queue if container is not None else None
    if queue is None:
        raise TaskNotQueuedError(f"task {kind}: no task queue yet")
    await enqueue_only(queue, kind, payload)


def _remember_base_url(request: Request) -> None:
    """Learn the service URL from an authenticated request (task target)."""
    if container is not None and container.base_url is not None:
        container.base_url.remember(request.headers)


# ── Lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global container

    settings = get_settings()
    setup_logger(level=settings.log_level)

    legacy_warnings = legacy_config_warnings()
    for message in legacy_warnings:
        logger.warning(f"Config: {message}")

    if not settings.line_channel_secret or not settings.line_channel_access_token:
        logger.warning("LINE channel credentials are incomplete")
    container = build_container(settings)
    container.config_warnings.extend(legacy_warnings)
    container.base_url = BaseUrlResolver(settings.public_base_url)
    container.task_queue = build_task_queue(
        settings, dispatcher=_dispatch_task, base_url_provider=container.base_url
    )
    # The Colab lifecycle runs on the cron tick and `colab` tasks (ADR-0010),
    # queued only: a Colab call never runs inline in the cron request.
    container.colab_controller = build_colab_controller(
        settings, container.colab, submit=_queue_colab_task, bot_url=container.base_url
    )

    get_line_service()
    _start_scheduler(settings)

    # Summaries walk the utility tiers through the shared chain, so they
    # survive an NVIDIA outage or a retired model by falling back.
    configure_memory_service(
        settings,
        summary_chain=container.fallback_chain,
        summary_targets=container.role_table.tiers("utility"),
        # Compaction runs as a `compact` task, never as an in-process
        # background task (CPU is throttled after the response).
        compaction_scheduler=build_compaction_scheduler(_submit_task),
    )

    from src.services.embedding_service import (
        close_embedding_service,
        configure_embedding_service,
    )

    try:
        memory_service = get_memory_service()
        embed_tiers = container.role_table.tiers("embed")
        if embed_tiers:
            embedding_service = configure_embedding_service(
                settings, tier=embed_tiers[0], health=container.model_health
            )
            memory_service.set_embedding_service(embedding_service)
        else:
            reasons = {row.disabled_reason for row in container.role_table.rows("embed")}
            logger.warning(f"Embeddings off ({', '.join(sorted(reasons))}): no vector recall")
        wire_tool_registry(container, memory_service)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Memory/tool wiring skipped: {exc}")

    logger.info("LineBot-CloudAgent started (orchestrator + model role table)")
    yield

    # Shutdown
    from src.services.scheduler_service import close_scheduler_service
    from src.utils.gcp_metadata import close_metadata

    close_scheduler_service()
    await close_line_service()
    await close_memory_service()
    try:
        await close_embedding_service()
    except Exception:  # pragma: no cover - shutdown best effort
        pass
    if container is not None:
        await container.aclose()
    await close_metadata()
    logger.info("LineBot-CloudAgent stopped")


def _start_scheduler(settings) -> None:
    """Register scheduled jobs for /health/detail visibility.

    Dispatch itself is pulled by Cloud Scheduler via /internal/cron: Cloud
    Run scales to zero and cannot fire in-process timers. See ADR-0001.
    """
    from src.services.scheduler_service import get_scheduler_service

    if not (settings.scheduled_messages_enabled and settings.scheduled_group_id):
        logger.info("Scheduler disabled (SCHEDULED_MESSAGES_ENABLED=false or no group ID)")
        return

    scheduler = get_scheduler_service(settings.scheduled_messages_timezone)
    configured = len(settings.scheduled_weekly_messages) + len(settings.scheduled_yearly_messages)
    weekly, yearly = register_scheduled_jobs(settings, scheduler, settings.scheduled_group_id)
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


def _token_ok(request: Request, header: str, expected: str) -> bool:
    """True when *header* carries *expected*; an empty secret refuses all."""
    if not expected:
        return False
    provided = request.headers.get(header, "")
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _app_version() -> str:
    return os.environ.get("APP_VERSION", "").strip() or "dev"


def _get_health_status(settings, role_table) -> tuple[str, bool, list[str]]:
    """``(status, ready_for_webhook, warnings)``.

    Ready means LINE credentials plus at least one enabled chat tier that
    does not depend on Colab (a cold Colab must never block a reply).
    """
    warnings: list[str] = []
    line_ready = bool(settings.line_channel_secret and settings.line_channel_access_token)
    chat_ready = role_table is not None and role_table.has_api_tier("chat")

    if not line_ready:
        warnings.append("LINE channel credentials are incomplete")
    if role_table is None:
        warnings.append("model role table not built yet")
    elif not chat_ready:
        reasons = sorted(
            {row.disabled_reason for row in role_table.rows("chat") if not row.enabled}
        )
        warnings.append(f"no usable chat model tier ({'; '.join(reasons) or 'none configured'})")

    ready_for_webhook = line_ready and chat_ready
    status = "healthy" if ready_for_webhook else "degraded"
    return status, ready_for_webhook, warnings


@app.get("/health")
async def health():
    """Public and minimal: whether the service can take LINE traffic.

    Everything else (models, budgets, memory, jobs) is on the authenticated
    ``/health/detail``.
    """
    status, ready_for_webhook, _ = _get_health_status(
        get_settings(), container.role_table if container else None
    )
    return {"status": status, "ready_for_webhook": ready_for_webhook, "version": _app_version()}


@app.get("/health/detail")
async def health_detail(request: Request):
    """The operator view, behind ``X-Cron-Token`` (INTERNAL_CRON_SECRET).

    Model tiers with their health and last error, provider budgets, config
    warnings, the daily end-of-life results, tasks mode, memory, storage,
    search usage, the Colab session, and scheduled job ids with
    their next run. No message text: scheduled jobs are listed without
    their content.
    """
    settings = get_settings()
    if not _token_ok(request, "X-Cron-Token", settings.internal_cron_secret):
        return Response(status_code=403)

    from src.services.scheduler_service import peek_scheduler_service
    from src.services.storage_service import get_storage_service

    search = container.search if container else None
    colab = container.colab if container else None
    colab_controller = container.colab_controller if container else None
    role_table = container.role_table if container else None
    status, ready_for_webhook, warnings = _get_health_status(settings, role_table)
    model_health = container.model_health if container else None
    lifecycle = container.model_lifecycle if container else None
    scheduler = peek_scheduler_service()
    scheduler_stats = (
        scheduler.get_stats() if scheduler is not None else {"running": False, "jobs": []}
    )

    return {
        "status": status,
        "ready_for_webhook": ready_for_webhook,
        "version": _app_version(),
        "warnings": warnings,
        "config_warnings": list(container.config_warnings) if container else [],
        "model_path_preference": settings.model_path_preference,
        "force_tier": settings.force_tier or None,
        "roles": role_table.view(model_health, lifecycle) if role_table else {},
        "budgets": model_health.get_status()["providers"] if model_health else {},
        "lifecycle": lifecycle.view() if lifecycle else None,
        "agents_stats": container.agents_stats() if container else {},
        "fallback_count": container.fallback_chain.fallback_count if container else 0,
        # "cloud" (Cloud Tasks) or "inline"; see ADR-0006.
        "tasks_mode": container.task_queue.mode if container and container.task_queue else None,
        "memory": get_memory_service().get_stats(),
        "storage": get_storage_service().get_usage_stats(),
        "scheduler": {
            "running": scheduler_stats.get("running", False),
            "jobs": [
                {"id": job.get("id"), "next_run": job.get("next_run")}
                for job in scheduler_stats.get("jobs", [])
            ],
        },
        # Per backend: configured, Tavily's mode and credits, cooldowns and
        # the last error; never a key.
        "web_search": search.get_status() if search is not None else {"configured": False},
        # The session's status, readiness, last error and compute units;
        # never its tunnel URL, bearer or Colab ids. ADR-0010.
        "colab": _colab_view(colab, colab_controller, settings),
    }


def _colab_view(colab, controller, settings) -> dict:
    if colab is None:
        reason = container.colab_disabled_reason if container else ""
        if not reason:
            reason = "COLAB_ENABLED=false" if not settings.colab_enabled else "not built"
        return {"enabled": False, "reason": reason}
    view = colab.view()
    if controller is not None:
        view["controller"] = controller.view()
    return view


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
    """Dispatch due scheduled messages, then run the maintenance sweep.

    Called by Cloud Scheduler every minute. Auth is a shared secret in
    ``X-Cron-Token`` (INTERNAL_CRON_SECRET). Always returns 200 with a
    structured summary so Cloud Scheduler does not retry on transient
    send failures. Maintenance (stale passive buffers, the daily model
    end-of-life check queued as an ``eol`` task, and the Colab controller's
    tick) runs even when scheduled messages are disabled. See ADR-0001,
    ADR-0006, ADR-0007 and ADR-0010.
    """
    settings = get_settings()
    if not _token_ok(request, "X-Cron-Token", settings.internal_cron_secret):
        return Response(status_code=403)

    reference_time = _parse_cloud_scheduler_time(
        request.headers.get("X-CloudScheduler-ScheduleTime", "")
    )

    from src.services.scheduler_service import peek_scheduler_service

    _remember_base_url(request)
    scheduler = peek_scheduler_service()
    if scheduler is None:
        result = {
            "fired": [],
            "failed": [],
            "skipped_duplicate": [],
            "reason": "scheduler_not_initialized",
        }
    else:
        result = await scheduler.dispatch_due_jobs(reference_time=reference_time)

    maintenance = await run_cron_tick(
        CronDeps(
            memory=get_memory_service(),
            lifecycle=container.model_lifecycle if container else None,
            submit=_submit_task,
            colab=container.colab_controller if container else None,
        ),
        reference_time,
    )
    return JSONResponse({**result, **maintenance})


# ── Cloud Tasks target ───────────────────────────────────────


@app.post("/internal/tasks/{kind}")
async def internal_tasks(kind: str, request: Request):
    """Run one queued task (see ``src.pipeline.task_handlers``).

    Auth is a shared secret in ``X-Tasks-Token`` (INTERNAL_TASKS_SECRET,
    separate from the cron secret); an empty secret refuses everything.
    Once authenticated it always answers 200 with ``{"ok", "kind", "ms"}``:
    the queue makes one attempt per task, so a 5xx would buy nothing but
    noise. See ADR-0006.
    """
    settings = get_settings()
    if not _token_ok(request, "X-Tasks-Token", settings.internal_tasks_secret):
        return Response(status_code=403)

    try:
        payload = json.loads(await request.body())
    except (json.JSONDecodeError, ValueError):
        payload = None
    if not isinstance(payload, dict):
        logger.warning(f"Task {kind}: body is not a JSON object")
        return Response(status_code=400)

    _remember_base_url(request)
    started = time.perf_counter()
    try:
        ok = await _dispatch_task(kind, payload)
    except Exception as exc:  # pragma: no cover - _dispatch_task already catches
        logger.error(f"Task {kind} crashed: {exc}", exc_info=True)
        ok = False
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "Task done "
        + json.dumps(
            {
                "kind": kind,
                "ok": ok,
                "ms": elapsed_ms,
                "task": request.headers.get("X-CloudTasks-TaskName", ""),
                "retry": request.headers.get("X-CloudTasks-TaskRetryCount", ""),
            },
            ensure_ascii=False,
        )
    )
    return JSONResponse({"ok": ok, "kind": kind, "ms": elapsed_ms})


# ── Colab VM registration ────────────────────────────────────


async def _read_capped(request: Request, limit: int) -> bytes | None:
    """The request body, or None once it grows past *limit* bytes."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


@app.post("/internal/colab/register")
async def colab_register(request: Request):
    """The Colab VM reports ready, failed or stopping (ADR-0010).

    Public on purpose (the VM has no Google identity): the HMAC signature
    in ``X-Colab-Signature`` is the authentication, checked by
    :class:`~src.services.colab.registration.ColabRegistrar`, which also
    probes a new tunnel before the session counts as READY. 404 unless
    COLAB_ENABLED; 413 over 8 KB.
    """
    registrar = container.colab_registrar if container else None
    if registrar is None:
        return Response(status_code=404)
    body = await _read_capped(request, MAX_REGISTER_BYTES)
    if body is None:
        return JSONResponse({"ok": False, "error": "body too large"}, status_code=413)
    result = await registrar.handle(body, request.headers.get(SIGNATURE_HEADER))
    return JSONResponse(result.body, status_code=result.status_code)


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

    _remember_base_url(request)

    # Anything slow becomes a task, so the webhook answers LINE well inside
    # its 2 s limit. Nothing is started with create_task: CPU is throttled
    # once the response is sent. See ADR-0006.
    submissions: list[tuple[str, dict]] = []
    for event in data.get("events", []):
        dedupe_key = event_dedupe_key(event)
        if dedupe_key and not _webhook_seen.add(dedupe_key):
            logger.info(f"Webhook: duplicate event {dedupe_key} skipped")
            continue

        if should_handle(event):
            submissions.append(("event", {"event": event}))
            continue
        if event.get("type") != "message":
            continue

        flush_key = _record_group_message(event)
        if flush_key:
            submissions.append(("passive_flush", {"key": flush_key}))
        if event.get("message", {}).get("type") == "image":
            submissions.append(("cache_image", {"event": event}))
        else:
            get_message_cache_service().cache_event_text(event)

    if submissions:
        results = await asyncio.gather(
            *(_submit_task(kind, payload) for kind, payload in submissions),
            return_exceptions=True,
        )
        for (kind, _), result in zip(submissions, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Webhook: task {kind} failed: {result!r}")

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
