from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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

    def has_capability(self, cap: Capability | str) -> bool:
        if isinstance(cap, str):
            cap = CAPABILITY_ALIASES.get(cap, Capability.CHAT)
        return cap in self.capabilities

    def supports_all(self, required: set[Capability]) -> bool:
        return required.issubset(self.capabilities)
