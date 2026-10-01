"""The built application: providers, agents, and the chain between them.

A plain holder created once at startup by :mod:`src.runtime.composition`.
Having a single object to pass around is what lets the request pipeline
take its collaborators as arguments instead of reaching for module-level
globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppContainer:
    model_health: Any
    fallback_chain: Any
    # Which models serve each role (src/runtime/model_roles.py); the
    # utility tiers run background work such as memory summaries.
    role_table: Any = None
    # Provider clients by role-table name ("nvidia", "openrouter", "colab").
    providers: dict[str, Any] = field(default_factory=dict)
    # The daily end-of-life check (src/services/model_lifecycle.py).
    model_lifecycle: Any = None
    # Startup configuration warnings (key names only), for /health/detail.
    config_warnings: list[str] = field(default_factory=list)
    orchestrator: Any = None
    chat_agent: Any = None
    vision_agent: Any = None
    web_search_agent: Any = None
    image_gen_agent: Any = None
    # The web search backend chain (Tavily today), shared by the web
    # search agent and the chat agent's web_search tool, and the page
    # reader the agent fetches pages with (ADR-0009).
    search: Any = None
    page_reader: Any = None
    # Where post-webhook work goes (Cloud Tasks or inline) and the service
    # URL tasks call back to; set by lifespan, see ADR-0006.
    task_queue: Any = None
    base_url: Any = None
    # The Colab session endpoint, the VM's registration handler and the
    # lifecycle controller (src/services/colab/; the controller is built by
    # lifespan with the task queue); None unless COLAB_ENABLED and its
    # requirements are met, else why not. ADR-0010.
    colab: Any = None
    colab_registrar: Any = None
    colab_controller: Any = None
    colab_disabled_reason: str = ""

    def get_agent(self, name: str):
        """Resolve a specialist agent by the orchestrator's routing name."""
        return {
            "chat": self.chat_agent,
            "vision": self.vision_agent,
            "web_search": self.web_search_agent,
            "image_gen": self.image_gen_agent,
        }.get(name)

    def agents_stats(self) -> dict[str, dict[str, int]]:
        """Per-agent call counts for ``/health/detail``."""
        pairs = [
            ("orchestrator", self.orchestrator),
            ("chat", self.chat_agent),
            ("vision", self.vision_agent),
            ("web_search", self.web_search_agent),
            ("image_gen", self.image_gen_agent),
        ]
        return {name: {"calls": agent.call_count} for name, agent in pairs if agent}

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.close()
        for closable in (self.search, self.page_reader, self.colab):
            if closable is not None:
                await closable.close()
        if self.task_queue is not None:
            await self.task_queue.aclose()
