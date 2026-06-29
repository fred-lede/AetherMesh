from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("providers.registry")


class Capability(str, Enum):
    CHAT = "chat"
    TOOLS = "tools"
    THINKING = "thinking"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    RESPONSES = "responses"
    MCP = "mcp"
    WEB_SEARCH = "web_search"
    STREAMING = "streaming"
    DOCUMENTS = "documents"


CAPABILITY_ALIASES: dict[str, Capability] = {
    "chat": Capability.CHAT,
    "tool": Capability.TOOLS,
    "tools": Capability.TOOLS,
    "function": Capability.TOOLS,
    "function_calling": Capability.TOOLS,
    "thinking": Capability.THINKING,
    "reasoning": Capability.THINKING,
    "vision": Capability.VISION,
    "image": Capability.VISION,
    "audio": Capability.AUDIO,
    "embedding": Capability.EMBEDDINGS,
    "embeddings": Capability.EMBEDDINGS,
    "rerank": Capability.RERANK,
    "reranking": Capability.RERANK,
    "responses": Capability.RESPONSES,
    "mcp": Capability.MCP,
    "web_search": Capability.WEB_SEARCH,
    "stream": Capability.STREAMING,
    "streaming": Capability.STREAMING,
    "documents": Capability.DOCUMENTS,
    "document": Capability.DOCUMENTS,
    "file": Capability.DOCUMENTS,
    "files": Capability.DOCUMENTS,
}


def parse_capabilities(raw: list[str] | set[str] | None) -> set[Capability]:
    if not raw:
        return set()
    result: set[Capability] = set()
    for item in raw:
        key = item.strip().lower()
        cap = CAPABILITY_ALIASES.get(key)
        if cap:
            result.add(cap)
    return result


@dataclass
class ProviderCapabilityEntry:
    name: str
    capabilities: set[Capability] = field(default_factory=set)
    health_url: str = ""
    requires_key: bool = False
    base_url_env: str = ""
    api_key_env: str = ""
    default_base_url: str = ""
    latency_ms: float = 0.0
    healthy: bool = True
    gpu_pressure: float = 0.0
    cost_score: float = 0.5

    def has_capability(self, cap: Capability | str) -> bool:
        if isinstance(cap, str):
            cap = CAPABILITY_ALIASES.get(cap, Capability.CHAT)
        return cap in self.capabilities

    def supports_all(self, required: set[Capability]) -> bool:
        return required.issubset(self.capabilities)


def _default_weight(required: set[Capability]) -> float:
    if Capability.THINKING in required:
        return 0.08
    if Capability.VISION in required:
        return 0.06
    return 0.04


class ProviderCapabilityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ProviderCapabilityEntry] = {}

    def register(self, entry: ProviderCapabilityEntry) -> None:
        self._entries[entry.name] = entry
        logger.info("Registered provider: %s (%d capabilities)", entry.name, len(entry.capabilities))

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def get(self, name: str) -> ProviderCapabilityEntry | None:
        return self._entries.get(name)

    def list_providers(self) -> list[ProviderCapabilityEntry]:
        return list(self._entries.values())

    def get_providers_for(self, required: set[Capability | str]) -> list[ProviderCapabilityEntry]:
        parsed: set[Capability] = set()
        for r in required:
            if isinstance(r, Capability):
                parsed.add(r)
            else:
                cap = CAPABILITY_ALIASES.get(r.lower())
                if cap:
                    parsed.add(cap)
        if not parsed:
            parsed = {Capability.CHAT}
        return [e for e in self._entries.values() if e.supports_all(parsed)]

    def score_provider(
        self,
        name: str,
        required: set[Capability | str],
        latency_ms: float = 0.0,
        health_ok: bool = True,
        gpu_pressure: float = 0.0,
        tool_requirement: bool = False,
        is_cloud: bool = False,
    ) -> float:
        entry = self._entries.get(name)
        if not entry:
            return 0.0

        parsed: set[Capability] = set()
        for r in required:
            if isinstance(r, Capability):
                parsed.add(r)
            else:
                cap = CAPABILITY_ALIASES.get(r.lower())
                if cap:
                    parsed.add(cap)

        if not entry.supports_all(parsed):
            return 0.0

        base = 100.0
        weight = _default_weight(parsed)

        if not health_ok:
            base *= 0.3

        latency_penalty = min(latency_ms / 100, weight) if latency_ms > 0 else 0.0
        base *= 1 - latency_penalty

        if gpu_pressure > 0:
            gpu_penalty = min(gpu_pressure, weight)
            base *= 1 - gpu_penalty

        if is_cloud:
            base *= 1 - entry.cost_score * 0.2

        if tool_requirement and Capability.TOOLS in entry.capabilities:
            base *= 1.1

        return round(base, 1)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": e.name,
                    "description": f"Provider: {e.name}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "capabilities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": sorted(c.value for c in e.capabilities),
                            },
                        },
                    },
                },
            }
            for e in self._entries.values()
        ]


provider_capability_registry = ProviderCapabilityRegistry()


def _auto_register_tts() -> None:
    """Register the XTTS provider if TTS is enabled."""
    from config.settings import settings
    if settings.tts_enabled:
        provider_capability_registry.register(
            ProviderCapabilityEntry(
                name="xtts",
                capabilities={Capability.AUDIO},
                healthy=True,
                latency_ms=0,
                requires_key=False,
            )
        )


_auto_register_tts()
