from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mcp.registry")


MCP_SERVER_SCHEMA = "2025-03-26"


@dataclass
class MCPServerEntry:
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    capabilities: set[str] = field(default_factory=set)
    auth_token: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_stdio(self) -> bool:
        return self.transport == "stdio"

    @property
    def is_sse(self) -> bool:
        return self.transport == "sse"


@dataclass
class MCPToolEntry:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


class MCPRegistry:
    def __init__(self) -> None:
        self._servers: dict[str, MCPServerEntry] = {}
        self._tools: dict[str, MCPToolEntry] = {}

    def register_server(self, entry: MCPServerEntry) -> None:
        self._servers[entry.name] = entry
        logger.info("MCP server registered: %s (%s)", entry.name, entry.transport)

    def unregister_server(self, name: str) -> None:
        self._servers.pop(name, None)
        self._tools = {k: v for k, v in self._tools.items() if v.server_name != name}

    def get_server(self, name: str) -> MCPServerEntry | None:
        return self._servers.get(name)

    def list_servers(self) -> list[MCPServerEntry]:
        return [s for s in self._servers.values() if s.enabled]

    def register_tool(self, entry: MCPToolEntry) -> None:
        self._tools[entry.name] = entry

    def get_tool(self, name: str) -> MCPToolEntry | None:
        return self._tools.get(name)

    def list_tools(self) -> list[MCPToolEntry]:
        return list(self._tools.values())

    def clear(self) -> None:
        self._servers.clear()
        self._tools.clear()


mcp_registry = MCPRegistry()
