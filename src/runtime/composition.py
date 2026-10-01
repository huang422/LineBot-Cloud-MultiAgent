"""Composition root: builds every provider, agent, and tool wiring.

All the "which model, in what order, with what fallback" decisions live
here so that ``main.py`` stays a transport layer and the agents stay
unaware of how they were assembled.
"""

from __future__ import annotations

from src.agents.chat_agent import ChatAgent
from src.agents.orchestrator import Orchestrator
from src.providers.fallback_chain import FallbackChain
from src.providers.model_registry import supports_reasoning
from src.providers.nvidia_provider import NvidiaProvider
from src.providers.openrouter_provider import OpenRouterProvider
from src.runtime.container import AppContainer
from src.utils.logger import logger

# ── Reasoning-capability filtering ───────────────────────────


def supports_reasoning_target(settings, provider, model_id: str) -> bool:
    if isinstance(provider, NvidiaProvider):
        return settings.nvidia_thinking_enabled and supports_reasoning(model_id)
    if isinstance(provider, OpenRouterProvider):
        return settings.openrouter_reasoning_enabled and supports_reasoning(model_id)
    return supports_reasoning(model_id)


def filter_reasoning_targets(settings, targets, target_group: str):
    """Drop non-reasoning targets when the deployment demands reasoning.

    Falls back to the unfiltered list rather than returning nothing: a
    degraded answer beats no answer.
    """
    if not settings.require_reasoning_models:
        return targets

    reasoning_targets = [
        (provider, model_id)
        for provider, model_id in targets
        if supports_reasoning_target(settings, provider, model_id)
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


def build_text_agent_targets(settings, openrouter_provider, nvidia_provider):
    targets = []
    if nvidia_provider:
        targets.append((nvidia_provider, settings.nvidia_model))
    targets.append((openrouter_provider, settings.agent_fallback_model))
    targets.append((openrouter_provider, "openrouter/free"))
    return filter_reasoning_targets(settings, targets, "text")


def build_vision_agent_targets(settings, openrouter_provider, nvidia_provider):
    targets = []
    if nvidia_provider:
        targets.append((nvidia_provider, settings.nvidia_model))
    targets.append((openrouter_provider, settings.vision_fallback_model))
    return filter_reasoning_targets(settings, targets, "vision")


def build_orchestrator_targets(settings, openrouter_provider, nvidia_provider):
    """OpenRouter primary, NVIDIA fallback — deliberately cross-provider."""
    targets = [(openrouter_provider, settings.orchestrator_model)]
    if nvidia_provider:
        targets.append((nvidia_provider, settings.orchestrator_fallback_model))
    else:
        targets.append((openrouter_provider, settings.orchestrator_fallback_model))
    return filter_reasoning_targets(settings, targets, "orchestrator")


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
    """Returns a tool executor that pulls episodic memory snippets."""

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
                "summary": (ep.get("summary") or "")[:600],
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


def build_web_search_executor(get_web_search_service):
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


# ── The composition root ─────────────────────────────────────


def build_container(settings, rate_tracker) -> AppContainer:
    """Build every provider and agent for this process."""
    openrouter_provider = OpenRouterProvider(
        settings.openrouter_api_key,
        rate_tracker,
        reasoning_enabled=settings.openrouter_reasoning_enabled,
        reasoning_effort=settings.openrouter_reasoning_effort,
        reasoning_exclude=settings.openrouter_reasoning_exclude,
        thinking_budget=settings.openrouter_thinking_budget,
    )

    nvidia_provider = None
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
            f"NvidiaProvider initialized (thinking={settings.nvidia_thinking_enabled}, "
            f"budget={settings.nvidia_thinking_budget})"
        )
    else:
        logger.warning("NVIDIA_API_KEY not set — agents will use OpenRouter fallback only")

    fallback_chain = FallbackChain(rate_tracker)

    text_targets = build_text_agent_targets(settings, openrouter_provider, nvidia_provider)
    vision_targets = build_vision_agent_targets(settings, openrouter_provider, nvidia_provider)
    orch_targets = build_orchestrator_targets(settings, openrouter_provider, nvidia_provider)

    from src.agents.image_gen_agent import ImageGenAgent
    from src.agents.vision_agent import VisionAgent
    from src.agents.web_search_agent import WebSearchAgent

    container = AppContainer(
        rate_tracker=rate_tracker,
        fallback_chain=fallback_chain,
        openrouter_provider=openrouter_provider,
        nvidia_provider=nvidia_provider,
        orchestrator=Orchestrator(settings, fallback_chain, targets=orch_targets),
        chat_agent=ChatAgent(settings, fallback_chain, targets=text_targets),
        vision_agent=VisionAgent(settings, fallback_chain, targets=vision_targets),
        web_search_agent=WebSearchAgent(settings, fallback_chain, targets=text_targets),
        image_gen_agent=ImageGenAgent(
            settings, fallback_chain, targets=text_targets, nvidia_provider=nvidia_provider
        ),
    )
    logger.info(
        f"Agents loaded (image_gen two-stage: text refinement → NVIDIA "
        f"{settings.image_gen_primary_model} / {settings.image_gen_fallback_model})"
    )
    return container


def wire_tool_registry(container: AppContainer, memory_service) -> None:
    """Attach the tool registry to the chat agent.

    Best-effort: if the embedding service or registry cannot be built the
    bot still replies, just without tool calling.
    """
    from src.agents.tools import build_default_registry
    from src.services.web_search_service import get_web_search_service

    try:
        registry = build_default_registry()
        registry.replace_executor(
            "recall_memory", build_recall_memory_executor(memory_service)
        )
        registry.replace_executor(
            "update_user_profile", build_update_user_profile_executor(memory_service)
        )
        registry.replace_executor(
            "web_search", build_web_search_executor(get_web_search_service)
        )
        container.chat_agent.set_tool_registry(registry)
        logger.info(
            f"ChatAgent tool registry wired: {', '.join(sorted(registry.names()))}"
        )
    except Exception as exc:
        logger.warning(
            f"Tool registry wiring skipped: {exc}; chat agent will run "
            "without tool calling"
        )
