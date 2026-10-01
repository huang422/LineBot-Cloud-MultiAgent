"""The built application: providers, agents, and the chain between them.

A plain holder created once at startup by :mod:`src.runtime.composition`.
Having a single object to pass around is what lets the request pipeline
take its collaborators as arguments instead of reaching for module-level
globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AppContainer:
    rate_tracker: Any
    fallback_chain: Any
    openrouter_provider: Any = None
    nvidia_provider: Any = None
    orchestrator: Any = None
    chat_agent: Any = None
    vision_agent: Any = None
    web_search_agent: Any = None
    image_gen_agent: Any = None

    def get_agent(self, name: str):
        """Resolve a specialist agent by the orchestrator's routing name."""
        return {
            "chat": self.chat_agent,
            "vision": self.vision_agent,
            "web_search": self.web_search_agent,
            "image_gen": self.image_gen_agent,
        }.get(name)

    def agents_stats(self) -> dict[str, dict[str, int]]:
        """Per-agent call counts for ``/health``."""
        pairs = [
            ("orchestrator", self.orchestrator),
            ("chat", self.chat_agent),
            ("vision", self.vision_agent),
            ("web_search", self.web_search_agent),
            ("image_gen", self.image_gen_agent),
        ]
        return {name: {"calls": agent.call_count} for name, agent in pairs if agent}

    async def aclose(self) -> None:
        for provider in (self.openrouter_provider, self.nvidia_provider):
            if provider is not None:
                await provider.close()
