"""Composition root: builds every provider, agent, and tool wiring.

Which model answers what, in what order, is the role table's business
(:mod:`src.runtime.model_roles`); this module builds the providers, binds
the table to them and hands each agent its role's tiers, so that
``main.py`` stays a transport layer and the agents stay unaware of how they
were assembled.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.agents.chat_agent import ChatAgent
from src.agents.orchestrator import Orchestrator
from src.agents.tools import (
    RECALL_SNIPPET_CHARS,
    RECALL_SUMMARY_CHARS,
    WEB_SEARCH_CONTEXT_CHARS,
    WEB_SEARCH_RESULTS,
)
from src.config import MIN_COLAB_SECRET_LENGTH
from src.pipeline.task_handlers import EVENT_NOTICE_SECONDS
from src.providers.fallback_chain import FallbackChain, Tier
from src.providers.model_profiles import SYSTEMONE
from src.providers.nvidia_provider import NvidiaProvider
from src.providers.ollama_provider import OllamaChatProvider
from src.providers.openrouter_provider import OpenRouterProvider
from src.providers.systemone import RouterGates, SystemOneRouter
from src.runtime.container import AppContainer
from src.runtime.model_roles import ROLES, RoleTable
from src.services.colab.control_plane import ColabControlPlane, adc_file_problem
from src.services.colab.controller import TASK_KIND as COLAB_TASK_KIND
from src.services.colab.controller import ColabController, ControllerConfig
from src.services.colab.endpoint import ColabEndpoint, ColabLimits
from src.services.colab.launcher import ColabLauncher, LaunchSettings
from src.services.colab.registration import ColabRegistrar
from src.services.colab.state import ColabStateStore, build_state_store
from src.services.embedding_service import get_embedding_service
from src.services.memory_service import episode_excerpt
from src.services.model_health import ModelHealth
from src.services.model_lifecycle import ModelLifecycle
from src.services.search.backend import ChainedSearchBackend, SearchBackend
from src.services.search.extract import PageReader, TextExtractor
from src.services.search.page_fetcher import PageFetcher
from src.services.search.query import SearchQueryPlanner
from src.services.search.rerank import SearchReranker
from src.services.search.tavily import TavilyBackend
from src.services.task_queue import (
    CloudTasksQueue,
    Dispatcher,
    InlineTaskQueue,
    TaskQueue,
    select_tasks_mode,
)
from src.utils.logger import logger

# ── Scheduled jobs ───────────────────────────────────────────


def register_scheduled_jobs(settings, scheduler, group_id: str) -> tuple[int, int]:
    weekly_registered = 0
    yearly_registered = 0

    for job in settings.scheduled_weekly_messages:
        if scheduler.add_weekly_message(
            job.id, job.day_of_week, job.hour, job.minute, group_id, job.message
        ):
            weekly_registered += 1

    for job in settings.scheduled_yearly_messages:
        if scheduler.add_yearly_message(
            job.id, job.month, job.day, job.hour, job.minute, group_id, job.message
        ):
            yearly_registered += 1

    return weekly_registered, yearly_registered


# ── Tool executors ───────────────────────────────────────────


def build_recall_memory_executor(memory_service):
    """Returns a tool executor that pulls episodic memory snippets.

    Each match carries the part of the episode's conversation that holds
    the query's words (else its newest part), RECALL_SNIPPET_CHARS at most,
    and the start of the summary saved with it.
    """

    async def _executor(args, ctx):
        query = str(args.get("query") or "").strip()
        if not query or not ctx.chat_id:
            return {"status": "noop", "matches": [], "query": query}
        try:
            k_int = int(args.get("k")) if args.get("k") is not None else 3
        except (TypeError, ValueError):
            k_int = 3
        try:
            episodes = await memory_service.recall_episodes(
                source_type=ctx.source_type, chat_id=ctx.chat_id, query=query, k=k_int
            )
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "matches": []}

        matches = [
            {
                "snippet": episode_excerpt(
                    str(ep.get("snippet") or ""), query, RECALL_SNIPPET_CHARS
                ),
                "summary": str(ep.get("summary") or "")[:RECALL_SUMMARY_CHARS],
                "ts": ep.get("ts"),
                "score": ep.get("score"),
            }
            for ep in episodes
        ]
        return {
            "status": "ok" if matches else "empty",
            "query": query,
            "matches": matches,
        }

    return _executor


def build_update_user_profile_executor(memory_service):
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


def build_web_search_executor(search: SearchBackend | None):
    """The web_search tool: one search through the same backend chain as the
    web search agent (snippets only, no page reads). The tool loop runs it at
    most WEB_SEARCH_TOOL_MAX_CALLS_PER_TURN times per reply.
    """

    async def _executor(args, ctx):
        query = str(args.get("query") or "").strip()
        if not query:
            return {"status": "noop", "reason": "empty query"}
        if search is None or not search.is_configured:
            return {
                "status": "unavailable",
                "reason": (
                    "web search is not configured; answer from what you know "
                    "and say it may be out of date"
                ),
            }
        try:
            result = await search.search(query, max_results=WEB_SEARCH_RESULTS)
        except Exception as exc:
            return {"status": "error", "reason": type(exc).__name__}
        if result.is_empty:
            return {"status": "empty", "query": query}
        return {
            "status": "ok",
            "query": query,
            "context": result.to_context_text()[:WEB_SEARCH_CONTEXT_CHARS],
        }

    return _executor


# ── Web search ───────────────────────────────────────────────


def build_search_backend(settings, *, tavily_transport=None) -> ChainedSearchBackend:
    """Tavily (with a key, or keyless with TAVILY_KEYLESS_ENABLED=true), then
    no search. ADR-0009."""
    backends: list[SearchBackend] = []
    tavily = TavilyBackend(
        settings.tavily_api_key,
        keyless_enabled=settings.tavily_keyless_enabled,
        transport=tavily_transport,
    )
    if tavily.is_configured:
        backends.append(tavily)
        mode = " (keyless)" if tavily.mode == "keyless" else ""
        logger.info(f"Web search: tavily{mode} → answer without search")
    else:
        logger.warning(
            "Web search is off (no TAVILY_API_KEY, TAVILY_KEYLESS_ENABLED not true): "
            "answers go without search"
        )
    return ChainedSearchBackend(backends)


def build_page_reader(settings) -> PageReader:
    """SSRF-safe page reads for the web search agent (URLs and top results)."""
    return PageReader(
        PageFetcher(max_bytes=settings.search_fetch_max_bytes),
        TextExtractor(),
        budget_seconds=settings.search_fetch_budget_seconds,
    )


def build_search_reranker(settings) -> SearchReranker | None:
    """The embedding rerank, only with SEARCH_RERANK_ENABLED. The embedding
    service is configured after the container, so it is looked up per call."""
    if not settings.search_rerank_enabled:
        return None
    return SearchReranker(get_embedding_service)


# ── The composition root ─────────────────────────────────────


def build_model_health(settings) -> ModelHealth:
    """Model and provider availability, with the account's free-model cap."""
    return ModelHealth(openrouter_free_daily_limit=settings.openrouter_free_daily_limit)


def build_openrouter_provider(settings, health, *, transport=None) -> OpenRouterProvider | None:
    """The OpenRouter client as the app configures it; None without a key.

    scripts/live_gate.py builds its clients here too (through
    :func:`build_providers`, with a recording ``transport``), so the gate
    sends exactly what the bot sends.
    """
    if not settings.openrouter_api_key.strip():
        return None
    return OpenRouterProvider(
        settings.openrouter_api_key,
        health,
        thinking_budget=settings.nvidia_thinking_budget,
        allow_paid=settings.openrouter_allow_paid,
        transport=transport,
    )


def build_nvidia_provider(settings, health, *, transport=None) -> NvidiaProvider | None:
    """The NVIDIA client as the app configures it; None without a key."""
    if not settings.nvidia_api_key:
        return None
    return NvidiaProvider(
        settings.nvidia_api_key,
        health,
        thinking_budget=settings.nvidia_thinking_budget,
        transport=transport,
    )


def build_providers(settings, health, *, transport=None) -> dict[str, Any]:
    """The API providers by role-table name ("nvidia", "openrouter").

    Colab registers under "colab" in :func:`build_container`, around the
    session endpoint. ``transport`` is for scripts/live_gate.py.
    """
    candidates = {
        "nvidia": build_nvidia_provider(settings, health, transport=transport),
        "openrouter": build_openrouter_provider(settings, health, transport=transport),
    }
    return {name: provider for name, provider in candidates.items() if provider is not None}


# ── Colab (ADR-0010) ─────────────────────────────────────────


def colab_config_problems(settings, environ: Mapping[str, str] | None = None) -> list[str]:
    """Why COLAB_ENABLED cannot be honoured; empty when it can. Key names only.

    The session state is a Firestore document the per-minute controller and
    the request path share (an in-memory copy would be lost on a cold start
    with a VM still running); the shared secret keys the VM's registration
    and bearer; the ADC file is the Colab account's credential. The
    controller runs only from the cron tick, which /internal/cron refuses
    without INTERNAL_CRON_SECRET: no session would ever start. On Cloud
    Run the controller's calls (up to minutes) need Cloud Tasks: with the
    inline queue they would run inside the cron request, past Cloud
    Scheduler's 60 s deadline.
    """
    environ = os.environ if environ is None else environ
    problems: list[str] = []
    if not settings.firestore_enabled:
        problems.append("COLAB_ENABLED needs FIRESTORE_ENABLED=true")
    if not settings.internal_cron_secret:
        problems.append(
            "COLAB_ENABLED needs INTERNAL_CRON_SECRET (the per-minute cron runs the Colab controller)"
        )
    if environ.get("K_SERVICE") and (settings.tasks_mode == "inline" or not settings.internal_tasks_secret):
        problems.append("COLAB_ENABLED on Cloud Run needs Cloud Tasks (INTERNAL_TASKS_SECRET, TASKS_MODE auto or cloud)")
    if settings.cloud_tasks_colab_queue == settings.cloud_tasks_queue:
        problems.append("CLOUD_TASKS_COLAB_QUEUE must differ from CLOUD_TASKS_QUEUE (Colab tasks would hold the event slots)")
    if len(settings.colab_shared_secret) < MIN_COLAB_SECRET_LENGTH:
        problems.append(
            f"COLAB_ENABLED needs COLAB_SHARED_SECRET of at least {MIN_COLAB_SECRET_LENGTH} characters"
        )
    adc = adc_file_problem(settings.colab_adc_path)
    if adc:
        problems.append(f"COLAB_ADC_PATH: {adc}")
    if settings.colab_tunnel_mode == "named":
        if not settings.colab_tunnel_host:
            problems.append("COLAB_TUNNEL_MODE=named needs COLAB_TUNNEL_HOST")
        if not settings.colab_tunnel_token:
            problems.append("COLAB_TUNNEL_MODE=named needs COLAB_TUNNEL_TOKEN")
    return problems


def build_colab_endpoint(
    settings,
    *,
    store: ColabStateStore | None = None,
    transport=None,
) -> ColabEndpoint | None:
    """The Colab session endpoint with COLAB_ENABLED, else None.

    A requirement that is not met (:func:`colab_config_problems`) keeps the
    tier off with a warning instead of stopping the app: every Colab tier is
    then disabled and ``/internal/colab/register`` answers 404. Demand
    starts sessions only under MODEL_PATH_PREFERENCE=colab. ``store`` and
    ``transport`` are for tests.
    """
    if not settings.colab_enabled:
        return None
    problems = colab_config_problems(settings)
    if problems:
        logger.warning(f"Colab tier off: {'; '.join(problems)}")
        return None
    return new_colab_endpoint(settings, store or build_state_store(settings), transport=transport)


def new_colab_endpoint(settings, store: ColabStateStore, *, transport=None) -> ColabEndpoint:
    """The session endpoint over *store*, configured as the app configures it.

    No requirement checks: :func:`build_colab_endpoint` makes them for the
    app; scripts/live_gate.py calls this directly for a session read from
    Firestore (the ADC file exists only on Cloud Run), with a recording
    ``transport``, so the gate talks to the tunnel exactly as the bot does.
    """
    return ColabEndpoint(
        settings.colab_shared_secret,
        store,
        auto_start=settings.model_path_preference == "colab",
        tunnel_host=settings.colab_tunnel_host,
        limits=ColabLimits.from_settings(settings),
        transport=transport,
    )


def split_router_tiers(role_table: RoleTable) -> tuple[list[Tier], Tier | None, bool]:
    """``(json_tiers, system_one_tier, system_one_first)`` from the router role.

    The Colab router (nimble) answers on /v1/systemone, not in JSON, so it
    is not a chain tier: the orchestrator asks it directly, first when it
    comes before every JSON tier in the table (MODEL_PATH_PREFERENCE=colab)
    and after them otherwise.
    """
    rows = [row for row in role_table.rows("router") if row.enabled]
    json_tiers = [row.tier for row in rows if row.profile.protocol != SYSTEMONE]
    system_one_rows = [index for index, row in enumerate(rows) if row.profile.protocol == SYSTEMONE]
    if not system_one_rows:
        return json_tiers, None, False
    first_json = next(
        (index for index, row in enumerate(rows) if row.profile.protocol != SYSTEMONE), len(rows)
    )
    return json_tiers, rows[system_one_rows[0]].tier, system_one_rows[0] < first_json


def build_container(
    settings,
    model_health: ModelHealth | None = None,
    *,
    extra_providers: Mapping[str, Any] | None = None,
    colab: ColabEndpoint | None = None,
) -> AppContainer:
    """Build every provider and agent for this process.

    ``extra_providers`` adds (or replaces) providers by role-table name;
    tests pass fakes here. ``colab`` replaces the endpoint COLAB_ENABLED
    builds (tests pass one over a fake tunnel and an in-memory state).

    Raises:
        RoleConfigError: The model role table cannot be used.
    """
    model_health = model_health or build_model_health(settings)
    colab_problems: list[str] = []
    if colab is None and settings.colab_enabled:
        colab_problems = colab_config_problems(settings)
        colab = build_colab_endpoint(settings)
    providers = build_providers(settings, model_health)
    if colab is not None:
        providers["colab"] = OllamaChatProvider(
            colab,
            default_think=settings.colab_chat_think,
            thinking_budget=settings.nvidia_thinking_budget,
        )
    providers.update(extra_providers or {})
    role_table = RoleTable.from_settings(settings, providers)
    for warning in role_table.warnings:
        logger.warning(f"Config: {warning}")
    if settings.force_tier:
        logger.warning(
            f"FORCE_TIER={settings.force_tier} (test-only): every other model tier is disabled"
        )

    fallback_chain = FallbackChain(model_health, min_tier_seconds=settings.min_tier_seconds)
    chat_tiers = role_table.tiers("chat")
    vision_tiers = role_table.tiers("vision")
    utility_tiers = role_table.tiers("utility")
    # What BaseAgent.targets_for switches to for thinking and image requests.
    answer_tiers = {
        "thinking_targets": role_table.tiers("thinking"),
        "vision_targets": vision_tiers,
    }

    from src.agents.image_gen_agent import ImageGenAgent
    from src.agents.vision_agent import VisionAgent
    from src.agents.web_search_agent import WebSearchAgent

    search = build_search_backend(settings)
    page_reader = build_page_reader(settings)
    router_tiers, system_one_tier, system_one_first = split_router_tiers(role_table)
    system_one = (
        SystemOneRouter(
            colab,
            system_one_tier,
            gates=RouterGates.from_settings(settings),
            health=model_health,
        )
        if colab is not None and system_one_tier is not None
        else None
    )
    container = AppContainer(
        model_health=model_health,
        fallback_chain=fallback_chain,
        role_table=role_table,
        providers=providers,
        model_lifecycle=ModelLifecycle.for_role_table(role_table, model_health),
        config_warnings=[*role_table.warnings, *(f"Colab tier off: {problem}" for problem in colab_problems)],
        orchestrator=Orchestrator(
            settings,
            fallback_chain,
            targets=router_tiers,
            system_one=system_one,
            system_one_first=system_one_first,
        ),
        chat_agent=ChatAgent(settings, fallback_chain, targets=chat_tiers, **answer_tiers),
        vision_agent=VisionAgent(settings, fallback_chain, targets=vision_tiers, **answer_tiers),
        search=search,
        page_reader=page_reader,
        # Query planning is background work: the utility tiers.
        web_search_agent=WebSearchAgent(
            settings,
            fallback_chain,
            targets=chat_tiers,
            query_planner=SearchQueryPlanner(fallback_chain, utility_tiers),
            search=search,
            page_reader=page_reader,
            reranker=build_search_reranker(settings),
            **answer_tiers,
        ),
        # Prompt refinement and the caption are background work: utility.
        image_gen_agent=ImageGenAgent(
            settings,
            fallback_chain,
            targets=utility_tiers,
            image_tiers=role_table.tiers("image"),
            health=model_health,
        ),
        colab=colab,
        colab_registrar=(
            ColabRegistrar(
                colab, secret=settings.colab_shared_secret, tunnel_host=settings.colab_tunnel_host
            )
            if colab is not None
            else None
        ),
        colab_disabled_reason="; ".join(colab_problems),
    )
    if colab is not None:
        router = (
            f"{system_one.model} router {'first' if system_one_first else 'last'}"
            if system_one is not None
            else "Colab router off"
        )
        logger.info(
            f"Colab tier on ({'starts on demand' if colab.auto_start else 'starts on !colab on'}; "
            f"{router})"
        )
    logger.info(
        "Model roles (MODEL_PATH_PREFERENCE="
        f"{settings.model_path_preference}): "
        + "; ".join(f"{role} → {role_table.describe(role)}" for role in ROLES)
    )
    return container


def build_colab_controller(
    settings,
    colab: ColabEndpoint | None,
    *,
    submit: Callable[[str, dict[str, Any]], Awaitable[Any]],
    bot_url: Callable[[], str],
    control_plane: Any = None,
) -> ColabController | None:
    """The Colab lifecycle controller (cron tick + ``colab`` tasks); None without Colab.

    It writes through ``colab.store``, the store the endpoint reads, so a
    transition reaches the request path at once. ``submit`` only queues a
    task and raises when it cannot (main's ``_queue_colab_task``, never the
    inline fallback); ``bot_url`` is where the VM registers (the
    ``BaseUrlResolver``). The control plane reads COLAB_ADC_PATH on its
    first call and imports the Colab libraries then, not here.
    ``control_plane`` is for tests.
    """
    if colab is None:
        return None
    control_plane = control_plane or ColabControlPlane(settings.colab_adc_path)
    launcher = ColabLauncher(
        control_plane,
        secret=settings.colab_shared_secret,
        bot_url=bot_url,
        settings=LaunchSettings.from_settings(settings),
    )
    return ColabController(
        colab,
        control_plane,
        launcher,
        submit=submit,
        config=ControllerConfig.from_settings(settings),
    )


def wire_tool_registry(container: AppContainer, memory_service) -> None:
    """Attach the tool registry to the chat agent. The web_search tool is
    offered only when a search backend is configured.

    Best-effort: if the embedding service or registry cannot be built the
    bot still replies, just without tool calling.
    """
    from src.agents.tools import build_default_registry

    try:
        registry = build_default_registry()
        registry.replace_executor(
            "recall_memory", build_recall_memory_executor(memory_service)
        )
        registry.replace_executor(
            "update_user_profile", build_update_user_profile_executor(memory_service)
        )
        if container.search is not None and container.search.is_configured:
            registry.replace_executor(
                "web_search", build_web_search_executor(container.search)
            )
        else:
            # Without Tavily the tool could only ever answer unavailable, a
            # wasted round trip: the model answers from what it knows.
            registry.remove("web_search")
        container.chat_agent.set_tool_registry(registry)
        logger.info(
            f"ChatAgent tool registry wired: {', '.join(sorted(registry.names()))}"
        )
    except Exception as exc:
        logger.warning(
            f"Tool registry wiring skipped: {exc}; chat agent will run "
            "without tool calling"
        )


# ── Post-webhook work ────────────────────────────────────────


def build_task_queue(
    settings,
    *,
    dispatcher: Dispatcher,
    base_url_provider: Callable[[], str],
    environ: Mapping[str, str] | None = None,
) -> TaskQueue:
    """Cloud Tasks on Cloud Run with a tasks secret, else run work inline.

    ``dispatcher(kind, payload)`` is what the inline queue awaits; it is the
    same function ``/internal/tasks/{kind}`` calls. See ADR-0006.
    """
    environ = os.environ if environ is None else environ
    mode = select_tasks_mode(settings, environ)
    if settings.task_handler_budget_seconds >= settings.task_dispatch_deadline_seconds:
        logger.warning(
            "TASK_HANDLER_BUDGET_SECONDS should stay below TASK_DISPATCH_DEADLINE_SECONDS, "
            "or Cloud Tasks gives up on a task before its handler does"
        )
    if settings.turn_deadline_seconds >= settings.task_handler_budget_seconds - EVENT_NOTICE_SECONDS:
        logger.warning(
            f"TURN_DEADLINE_SECONDS should stay below TASK_HANDLER_BUDGET_SECONDS - {EVENT_NOTICE_SECONDS:g} s, "
            "or an event turn is cut off before its answer goes out"
        )

    if mode == "cloud":
        location = settings.cloud_tasks_location or "(Cloud Run region)"
        # ``colab`` tasks only exist with COLAB_ENABLED; the deploy creates
        # their queue then (ADR-0010).
        kind_queues = {COLAB_TASK_KIND: settings.cloud_tasks_colab_queue} if settings.colab_enabled else {}
        colab_note = f" colab_queue={settings.cloud_tasks_colab_queue}" if kind_queues else ""
        logger.info(
            f"Post-webhook work runs on Cloud Tasks: queue={settings.cloud_tasks_queue}{colab_note} "
            f"location={location} deadline={settings.task_dispatch_deadline_seconds}s"
        )
        return CloudTasksQueue(
            location=settings.cloud_tasks_location,
            queue=settings.cloud_tasks_queue,
            kind_queues=kind_queues,
            secret=settings.internal_tasks_secret,
            base_url_provider=base_url_provider,
            dispatch_deadline_seconds=settings.task_dispatch_deadline_seconds,
        )

    if environ.get("K_SERVICE"):
        # CPU is throttled outside requests, so inline is the only safe
        # option here, but the webhook now waits for the work.
        logger.warning(
            f"Post-webhook work runs inline on Cloud Run (TASKS_MODE={settings.tasks_mode}); "
            "set INTERNAL_TASKS_SECRET to use Cloud Tasks"
        )
    else:
        logger.info(f"Post-webhook work runs inline (TASKS_MODE={settings.tasks_mode})")
    return InlineTaskQueue(dispatcher)


def build_compaction_scheduler(
    submit: Callable[[str, dict[str, Any]], Awaitable[Any]],
) -> Callable[..., Awaitable[None]]:
    """MemoryService's compaction scheduler: one ``compact`` task per trigger."""

    async def schedule(*, source_type: str, chat_scope: str, chat_id: str) -> None:
        await submit(
            "compact",
            {"source_type": source_type, "chat_scope": chat_scope, "chat_id": chat_id},
        )

    return schedule
