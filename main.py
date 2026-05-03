"""LineBot-CloudAgent: Multi-Agent LINE Bot on free cloud resources.

FastAPI entry point with Cloud Run background processing.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response

from src.config import get_settings
from src.models.agent_request import InputType
from src.utils.logger import logger, setup_logger
from src.utils.validators import validate_signature
from src.utils.rate_tracker import RateTracker

# Services
from src.services.line_service import get_line_service, close_line_service
from src.services.message_cache_service import get_message_cache_service
from src.services.memory_service import (
    configure_memory_service,
    get_memory_service,
    close_memory_service,
    MemoryServiceError,
)

# Providers
from src.providers.openrouter_provider import OpenRouterProvider
from src.providers.nvidia_provider import NvidiaProvider
from src.providers.fallback_chain import (
    AllModelsRateLimitedError,
    AllProvidersFailedError,
    FallbackChain,
)
from src.providers.model_registry import supports_reasoning

# Agents
from src.agents.orchestrator import Orchestrator
from src.agents.chat_agent import ChatAgent

# Processors
from src.processors.input_processor import process_input
from src.processors.output_processor import send_response

# Handlers
from src.handlers.webhook_handler import (
    should_handle,
    extract_text,
    enrich_request,
)

# ── Global instances ─────────────────────────────────────────

rate_tracker = RateTracker()
openrouter_provider: OpenRouterProvider | None = None
nvidia_provider: NvidiaProvider | None = None
fallback_chain: FallbackChain | None = None

# Agents
orchestrator: Orchestrator | None = None
chat_agent: ChatAgent | None = None

vision_agent = None
web_search_agent = None
image_gen_agent = None

# Background tasks (prevent garbage collection of fire-and-forget tasks)
_background_tasks: set[asyncio.Task] = set()


def _build_recall_memory_executor(memory_service):
    """Returns a tool executor that pulls episodic memory snippets."""

    async def _executor(args, ctx):
        query = str(args.get("query") or "").strip()
        if not query or not ctx.chat_id:
            return {"status": "noop", "matches": [], "query": query}
        k = args.get("k")
        try:
            k_int = int(k) if k is not None else 3
        except (TypeError, ValueError):
            k_int = 3
        try:
            episodes = await memory_service.recall_episodes(
                source_type=ctx.source_type,
                chat_id=ctx.chat_id,
                query=query,
                k=k_int,
            )
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "matches": []}

        matches = []
        for ep in episodes:
            matches.append({
                "summary": (ep.get("summary") or "")[:600],
                "ts": ep.get("ts"),
                "score": ep.get("score"),
            })
        return {
            "status": "ok" if matches else "empty",
            "query": query,
            "matches": matches,
        }

    return _executor


def _build_update_user_profile_executor(memory_service):
    async def _executor(args, ctx):
        if not ctx.user_id:
            return {"status": "noop", "reason": "no user_id in context"}
        facts = args.get("facts") or []
        if not isinstance(facts, list):
            facts = [facts]
        confidence = args.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else 0.7
        except (TypeError, ValueError):
            confidence = 0.7
        return await memory_service.update_user_facts(
            user_id=ctx.user_id,
            facts=[str(f) for f in facts if f is not None],
            confidence=confidence,
        )

    return _executor


def _build_web_search_executor(get_web_search_service):
    async def _executor(args, ctx):
        query = str(args.get("query") or "").strip()
        if not query:
            return {"status": "noop", "reason": "empty query"}
        svc = get_web_search_service()
        if svc is None or not svc.is_configured:
            return {"status": "unavailable", "reason": "web search not configured"}
        try:
            result = await svc.search(
                query,
                include_answer="advanced",
                search_depth="advanced",
                max_results=5,
            )
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}
        if not result.has_results and not result.answer:
            return {"status": "empty", "query": query}
        return {
            "status": "ok",
            "query": query,
            "context": result.to_context_text()[:4000],
        }

    return _executor


def _build_text_agent_targets(settings, openrouter_provider, nvidia_provider):
    targets = []
    if nvidia_provider:
        targets.append((nvidia_provider, settings.nvidia_model))
    targets.append((openrouter_provider, settings.agent_fallback_model))
    targets.append((openrouter_provider, "openrouter/free"))
    return _filter_reasoning_targets(settings, targets, "text")


def _build_vision_agent_targets(settings, openrouter_provider, nvidia_provider):
    targets = []
    if nvidia_provider:
        targets.append((nvidia_provider, settings.nvidia_model))
    targets.append((openrouter_provider, settings.vision_fallback_model))
    return _filter_reasoning_targets(settings, targets, "vision")


def _supports_reasoning_target(settings, provider, model_id: str) -> bool:
    if isinstance(provider, NvidiaProvider):
        return settings.nvidia_thinking_enabled and supports_reasoning(model_id)
    if isinstance(provider, OpenRouterProvider):
        return settings.openrouter_reasoning_enabled and supports_reasoning(model_id)
    return supports_reasoning(model_id)


def _filter_reasoning_targets(settings, targets, target_group: str):
    if not settings.require_reasoning_models:
        return targets

    reasoning_targets = [
        (provider, model_id)
        for provider, model_id in targets
        if _supports_reasoning_target(settings, provider, model_id)
    ]
    if reasoning_targets:
        skipped = len(targets) - len(reasoning_targets)
        if skipped > 0:
            logger.info(
                f"Skipping {skipped} non-reasoning target(s) for {target_group} requests"
            )
        return reasoning_targets

    logger.warning(
        f"No reasoning-capable targets configured for {target_group}; "
        "keeping original targets as a fallback"
    )
    return targets


def _register_scheduled_jobs(settings, scheduler, group_id: str) -> tuple[int, int]:
    weekly_registered = 0
    yearly_registered = 0

    for job in settings.scheduled_weekly_messages:
        if scheduler.add_weekly_message(
            job.id,
            job.day_of_week,
            job.hour,
            job.minute,
            group_id,
            job.message,
        ):
            weekly_registered += 1

    for job in settings.scheduled_yearly_messages:
        if scheduler.add_yearly_message(
            job.id,
            job.month,
            job.day,
            job.hour,
            job.minute,
            group_id,
            job.message,
        ):
            yearly_registered += 1

    return weekly_registered, yearly_registered


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global openrouter_provider, nvidia_provider, fallback_chain
    global orchestrator, chat_agent, vision_agent, web_search_agent, image_gen_agent

    settings = get_settings()
    setup_logger(level=settings.log_level)

    if not settings.line_channel_secret or not settings.line_channel_access_token:
        logger.warning("LINE channel credentials are incomplete")
    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not set — routing and fallback may fail")

    # ── Init providers ────────────────────────────────────
    openrouter_provider = OpenRouterProvider(
        settings.openrouter_api_key,
        rate_tracker,
        reasoning_enabled=settings.openrouter_reasoning_enabled,
        reasoning_effort=settings.openrouter_reasoning_effort,
        reasoning_exclude=settings.openrouter_reasoning_exclude,
        thinking_budget=settings.openrouter_thinking_budget,
    )
    fallback_chain = FallbackChain(rate_tracker)

    if settings.nvidia_api_key:
        nvidia_provider = NvidiaProvider(
            settings.nvidia_api_key,
            rate_tracker,
            thinking_enabled=settings.nvidia_thinking_enabled,
            thinking_budget=settings.nvidia_thinking_budget,
            thinking_model=settings.nvidia_thinking_model,
            primary_model=settings.nvidia_model,
        )
        logger.info(
            f"NvidiaProvider initialized "
            f"(thinking={settings.nvidia_thinking_enabled}, "
            f"budget={settings.nvidia_thinking_budget})"
        )
    else:
        logger.warning("NVIDIA_API_KEY not set — agents will use OpenRouter fallback only")

    # ── Build target lists ────────────────────────────────
    # Orchestrator: OpenRouter primary → NVIDIA fallback (cross-provider)
    orchestrator_targets = [
        (openrouter_provider, settings.orchestrator_model),
    ]
    if nvidia_provider:
        orchestrator_targets.append((nvidia_provider, settings.orchestrator_fallback_model))
    else:
        orchestrator_targets.append((openrouter_provider, settings.orchestrator_fallback_model))
    orchestrator_targets = _filter_reasoning_targets(
        settings,
        orchestrator_targets,
        "orchestrator",
    )

    text_agent_targets = _build_text_agent_targets(
        settings,
        openrouter_provider,
        nvidia_provider,
    )
    vision_agent_targets = _build_vision_agent_targets(
        settings,
        openrouter_provider,
        nvidia_provider,
    )

    # ── Init agents ───────────────────────────────────────
    orchestrator = Orchestrator(settings, fallback_chain, targets=orchestrator_targets)
    chat_agent = ChatAgent(settings, fallback_chain, targets=text_agent_targets)

    from src.agents.vision_agent import VisionAgent
    vision_agent = VisionAgent(settings, fallback_chain, targets=vision_agent_targets)
    logger.info("VisionAgent loaded")

    from src.agents.web_search_agent import WebSearchAgent
    web_search_agent = WebSearchAgent(settings, fallback_chain, targets=text_agent_targets)
    logger.info("WebSearchAgent loaded")

    from src.agents.image_gen_agent import ImageGenAgent
    image_gen_agent = ImageGenAgent(
        settings, fallback_chain,
        targets=text_agent_targets,
        nvidia_provider=nvidia_provider,
    )
    logger.info(
        f"ImageGenAgent loaded (two-stage: text refinement → "
        f"NVIDIA {settings.image_gen_primary_model} / {settings.image_gen_fallback_model})"
    )

    # Init LINE service
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

    # Init scheduler
    from src.services.scheduler_service import get_scheduler_service, close_scheduler_service
    if settings.scheduled_messages_enabled and settings.scheduled_group_id:
        if not settings.line_push_fallback_enabled:
            logger.warning(
                "Scheduled messages requested but not started because LINE push is disabled"
            )
        else:
            scheduler = get_scheduler_service()
            gid = settings.scheduled_group_id
            total_configured_jobs = (
                len(settings.scheduled_weekly_messages)
                + len(settings.scheduled_yearly_messages)
            )
            weekly_registered, yearly_registered = _register_scheduled_jobs(
                settings,
                scheduler,
                gid,
            )
            total_registered_jobs = weekly_registered + yearly_registered

            if total_configured_jobs == 0:
                logger.warning(
                    "Scheduled messages enabled but no scheduled jobs are configured"
                )
            elif total_registered_jobs == 0:
                logger.warning(
                    "Scheduled messages enabled but no scheduled jobs could be registered"
                )
            else:
                if total_registered_jobs < total_configured_jobs:
                    logger.warning(
                        "Some scheduled jobs could not be registered "
                        f"({total_registered_jobs}/{total_configured_jobs})"
                    )
                scheduler.start()
                logger.info(f"Scheduler started with {len(scheduler.list_jobs())} jobs")
    else:
        logger.info("Scheduler disabled (SCHEDULED_MESSAGES_ENABLED=false or no group ID)")

    configure_memory_service(settings, nvidia_provider=nvidia_provider)

    # ── Phase B: tool-calling registry with real executors ────
    # Embedding service powers vector recall; failures degrade to
    # stub executors so the bot still replies even if NIM is down.
    from src.services.embedding_service import (
        configure_embedding_service,
        close_embedding_service,
    )
    from src.agents.tools import build_default_registry
    from src.services.web_search_service import get_web_search_service

    try:
        embedding_service = configure_embedding_service(settings)
        memory_service = get_memory_service()
        if hasattr(memory_service, "set_embedding_service"):
            memory_service.set_embedding_service(embedding_service)

        tool_registry = build_default_registry()
        tool_registry.replace_executor(
            "recall_memory",
            _build_recall_memory_executor(memory_service),
        )
        tool_registry.replace_executor(
            "update_user_profile",
            _build_update_user_profile_executor(memory_service),
        )
        tool_registry.replace_executor(
            "web_search",
            _build_web_search_executor(get_web_search_service),
        )
        if hasattr(chat_agent, "set_tool_registry"):
            chat_agent.set_tool_registry(tool_registry)
            logger.info(
                "ChatAgent tool registry wired: "
                f"{', '.join(sorted(tool_registry.names()))}"
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"Tool registry wiring skipped: {exc}; chat agent will run "
            "without tool calling"
        )

    logger.info("LineBot-CloudAgent started (Orchestrator + NVIDIA Qwen3.5 architecture)")
    yield

    # Shutdown
    close_scheduler_service()
    from src.services.web_search_service import close_web_search_service
    await close_web_search_service()
    await close_line_service()
    await close_memory_service()
    try:
        await close_embedding_service()
    except Exception:  # pragma: no cover - shutdown best effort
        pass
    if openrouter_provider:
        await openrouter_provider.close()
    if nvidia_provider:
        await nvidia_provider.close()
    logger.info("LineBot-CloudAgent stopped")


app = FastAPI(title="LineBot-CloudAgent", lifespan=lifespan)


def _get_health_status(settings) -> tuple[str, bool, list[str]]:
    warnings: list[str] = []
    line_ready = bool(settings.line_channel_secret and settings.line_channel_access_token)
    openrouter_ready = bool(settings.openrouter_api_key)
    nvidia_ready = bool(settings.nvidia_api_key)

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


def _is_new_chat_command(text: str) -> bool:
    return text.strip().lower() == "!new"


def _apply_cleaned_text(request, event: dict) -> None:
    if request.text or event.get("message", {}).get("type") == "text":
        request.text = extract_text(event)
        if _request_has_image_context(request):
            request.input_type = InputType.IMAGE_TEXT if request.text else InputType.IMAGE
        else:
            request.input_type = InputType.TEXT


def _request_has_image_context(request) -> bool:
    return bool(
        request.image_base64
        or request.quoted_image_base64
        or request.quoted_image_url
    )


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


def _get_request_block_message(request) -> str | None:
    if request.rate_limited:
        return "⚠️ 請求太頻繁，請稍後再試。"
    if not request.text and not _request_has_image_context(request):
        return "請附上想問的內容，或直接引用訊息／圖片再提問。"
    return None


def _build_user_memory_text(request) -> str:
    text = request.text.strip()
    if text:
        return text
    if request.quoted_image_base64 or request.quoted_image_url:
        return "[使用者引用圖片]"
    if request.input_type in (InputType.IMAGE, InputType.IMAGE_TEXT) or request.image_base64:
        return "[使用者傳送圖片]"
    return ""


def _build_assistant_memory_text(response) -> str:
    text = (response.text or "").strip()
    if text:
        return text
    if response.output_format == "image":
        return "[已傳送圖片]"
    if response.output_format == "voice":
        return "[已傳送語音]"
    return ""


# ── Health ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    settings = get_settings()
    status, ready_for_webhook, warnings = _get_health_status(settings)
    agents_stats = {}
    for name, agent in [
        ("orchestrator", orchestrator),
        ("chat", chat_agent),
        ("vision", vision_agent),
        ("web_search", web_search_agent),
        ("image_gen", image_gen_agent),
    ]:
        if agent:
            agents_stats[name] = {"calls": agent.call_count}

    from src.services.scheduler_service import peek_scheduler_service
    from src.services.storage_service import get_storage_service
    from src.services.web_search_service import get_web_search_service

    scheduler = peek_scheduler_service()
    storage = get_storage_service()
    web_search = get_web_search_service()
    line = get_line_service()
    memory = get_memory_service()

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
        "agents_stats": agents_stats,
        "fallback_count": fallback_chain.fallback_count if fallback_chain else 0,
        "memory": memory.get_stats(),
        "scheduler": (
            scheduler.get_stats()
            if scheduler is not None
            else {"running": False, "job_count": 0, "jobs": []}
        ),
        "cost_controls": {
            "line_push": line.get_push_stats(),
            "web_search": web_search.get_quota_stats(),
            "storage": storage.get_usage_stats(),
        },
    }


# ── Webhook ──────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request):
    settings = get_settings()

    # Validate signature
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()

    if not validate_signature(body, signature, settings.line_channel_secret):
        return Response(status_code=403)

    # Parse events
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Webhook received invalid JSON body")
        return Response(status_code=400)
    events = data.get("events", [])

    # Return 200 immediately, process in background
    for event in events:
        should_process = should_handle(event)

        if event.get("type") == "message" and not should_process:
            _record_group_message(event)
            task = asyncio.create_task(
                get_message_cache_service().cache_event_message(event)
            )
            _track_background_task(task)

        if should_process:
            task = asyncio.create_task(_process_event(event))
            _track_background_task(task)

    return Response(status_code=200)


# ── Background processing ────────────────────────────────────


def _record_group_message(event: dict) -> None:
    """Buffer a non-triggered group/room message into the passive memory queue.

    Triggered messages bypass this path because ``record_interaction`` already
    persists the user's text to ``memory.recent_messages``.
    """
    if event.get("type") != "message":
        return

    message = event.get("message", {})
    msg_type = message.get("type", "")
    source = event.get("source", {})
    user_id = source.get("userId", "")
    src_type = source.get("type", "")
    group_id = source.get("groupId") or source.get("roomId") or user_id

    if not user_id or not group_id:
        return
    if src_type not in {"group", "room"}:
        return

    try:
        memory_svc = get_memory_service()
    except Exception:
        return

    text = ""
    if msg_type == "text":
        text = message.get("text", "").strip()
    elif msg_type == "image":
        text = "[圖片]"
    elif msg_type == "sticker":
        text = "[貼圖]"
    elif msg_type == "audio":
        text = "[語音]"

    if not text:
        return

    try:
        memory_svc.enqueue_passive_message(
            source_type=src_type,
            chat_scope="multi",
            chat_id=group_id,
            user_id=user_id,
            text=text,
        )
    except Exception as exc:
        logger.debug(f"enqueue_passive_message failed: {exc}")

async def _process_event(event: dict) -> None:
    """Process a single LINE event in the background."""
    source = event.get("source", {})
    chat_id = source.get("groupId") or source.get("roomId") or source.get("userId", "")
    reply_token = event.get("replyToken", "")

    try:
        line = get_line_service()

        # Build request from event
        request = await process_input(event)
        if request is None:
            return

        # Override text with cleaned version (remove !hej prefix etc.)
        _apply_cleaned_text(request, event)

        if _is_new_chat_command(request.text):
            await get_memory_service().clear_chat(
                source_type=request.source_type,
                chat_scope=request.chat_scope,
                chat_id=request.group_id,
            )
            await line.send_text(
                request.reply_token,
                chat_id,
                "Let's start a new chat!",
            )
            return

        # Send loading animation (free)
        await line.send_loading_animation(chat_id)

        get_message_cache_service().cache_processed_request(event, request)

        # Enrich: rate limit + memory context
        request = await enrich_request(request)
        block_message = _get_request_block_message(request)
        if block_message:
            await line.send_text(
                request.reply_token,
                chat_id,
                block_message,
            )
            return

        # Best-effort: resolve LINE display name + upsert user profile.
        # Failures here must never block the main reply path. The
        # SimpleNamespace mocks used in test_api_integration may not
        # provide these methods, so we swallow AttributeError as well.
        await _touch_user_profile_safe(line, request)

        # Route via Orchestrator
        decision = await orchestrator.route(request)
        request.target_agent = decision.agent
        request.output_format = decision.output_format
        request.task_description = decision.task_description
        request.routing_reasoning = decision.reasoning
        request.disable_thinking = decision.disable_thinking

        logger.info(
            f"[{request.request_id}] → {decision.agent} "
            f"(output={decision.output_format}, "
            f"thinking={'off' if decision.disable_thinking else 'on'})"
        )

        # Dispatch to agent
        agent = _get_agent(decision.agent)
        if agent is None:
            await line.send_text(
                request.reply_token, chat_id,
                f"Agent '{decision.agent}' 尚未實作。"
            )
            return

        response = await agent.process(request)

        # Output
        sent = await send_response(request, response)
        if not sent:
            logger.warning(
                f"[{request.request_id}] Response could not be delivered "
                "(reply failed or push budget unavailable)"
            )
        else:
            try:
                await get_memory_service().record_interaction(
                    source_type=request.source_type,
                    chat_scope=request.chat_scope,
                    chat_id=request.group_id,
                    user_id=request.user_id,
                    user_text=_build_user_memory_text(request),
                    assistant_text=_build_assistant_memory_text(response),
                    agent_name=request.target_agent,
                    output_format=request.output_format,
                    task_description=request.task_description,
                    routing_reasoning=request.routing_reasoning,
                    disable_thinking=request.disable_thinking,
                )
            except MemoryServiceError as e:
                logger.error(f"Memory interaction record failed: {e}")

    except AllModelsRateLimitedError:
        logger.error("All models rate limited!")
        await _send_error_message(reply_token, chat_id, "⚠️ AI 模型暫時忙碌中，請稍後再試。")

    except AllProvidersFailedError as e:
        logger.error(f"All provider targets failed: {e}", exc_info=True)
        await _send_error_message(reply_token, chat_id, "⚠️ AI 服務暫時不可用，請稍後再試。")

    except Exception as e:
        logger.error(f"Event processing error: {e}", exc_info=True)
        await _send_error_message(reply_token, chat_id, "❌ 處理請求時發生錯誤，請稍後再試。")


async def _send_error_message(reply_token: str, chat_id: str, text: str) -> None:
    """Best-effort error message to user."""
    try:
        line = get_line_service()
        await line.send_text(reply_token, chat_id, text)
    except Exception as e:
        logger.error(f"Failed to deliver error message: {e}", exc_info=True)


async def _touch_user_profile_safe(line, request) -> None:
    """Resolve LINE displayName and upsert per-user profile.

    All failures are swallowed: the per-user profile is metadata, not
    on the critical path for replying to the user. Logs at debug level
    to keep noise low when the user has not added the bot as a friend.
    """
    user_id = (request.user_id or "").strip()
    if not user_id:
        return

    display_name = ""
    try:
        display_name = await line.fetch_display_name(
            source_type=request.source_type,
            chat_id=request.group_id,
            user_id=user_id,
        )
    except AttributeError:
        # Test doubles may not implement fetch_display_name; that is fine.
        pass
    except Exception as exc:
        logger.debug(f"Display name lookup skipped for {user_id[:8]}…: {exc}")

    try:
        await get_memory_service().touch_user_profile(
            user_id=user_id,
            display_name=display_name,
            source_type=request.source_type,
            chat_id=request.group_id,
        )
    except AttributeError:
        # Older mocks without touch_user_profile — skip silently
        pass
    except Exception as exc:
        logger.debug(f"Profile touch skipped for {user_id[:8]}…: {exc}")


def _get_agent(name: str):
    """Get agent instance by name."""
    agents = {
        "chat": chat_agent,
        "vision": vision_agent,
        "web_search": web_search_agent,
        "image_gen": image_gen_agent,
    }
    return agents.get(name)


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
