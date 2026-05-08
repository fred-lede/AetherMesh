from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mcp.capability")

MCP_PROTOCOL_VERSION = "2025-03-26"

SUPPORTED_ROOTS = {"filesystem"}
SUPPORTED_SAMPLING = {"text", "image"}
SUPPORTED_TRANSPORTS = {"stdio", "sse"}


@dataclass
class MCPCapability:
    name: str
    version: str = MCP_PROTOCOL_VERSION
    supported: bool = False
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPCapabilitySet:
    protocol_version: str = MCP_PROTOCOL_VERSION
    roots: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    experimental: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        caps: dict[str, Any] = {}
        if self.roots:
            caps["roots"] = self.roots
        if self.sampling:
            caps["sampling"] = self.sampling
        if self.experimental:
            caps["experimental"] = self.experimental
        if self.logging:
            caps["logging"] = {"supported": True}
        return caps

    @classmethod
    def client_defaults(cls) -> MCPCapabilitySet:
        return cls(
            sampling={},
            experimental={},
            logging={"supported": True},
        )

    @classmethod
    def server_defaults(cls) -> MCPCapabilitySet:
        return cls(
            roots={"supported": True},
            experimental={},
        )


def negotiate_capabilities(
    client_caps: dict[str, Any],
    server_caps: dict[str, Any],
) -> dict[str, Any]:
    negotiated: dict[str, Any] = {"protocolVersion": MCP_PROTOCOL_VERSION}

    if "roots" in client_caps and "roots" in server_caps:
        negotiated["roots"] = {"supported": True}

    if "sampling" in client_caps:
        negotiated["sampling"] = {
            k: v for k, v in client_caps.get("sampling", {}).items()
            if k in SUPPORTED_SAMPLING
        }

    if "logging" in client_caps:
        negotiated["logging"] = {"supported": True}

    if "experimental" in client_caps and "experimental" in server_caps:
        negotiated["experimental"] = {
            k: v for k, v in client_caps["experimental"].items()
            if k in server_caps.get("experimental", {})
        }

    logger.debug(
        "Negotiated capabilities: client=%s server=%s result=%s",
        set(client_caps.keys()),
        set(server_caps.keys()),
        set(negotiated.keys()),
    )
    return negotiated


def supports_transport(server_caps: dict[str, Any], transport: str) -> bool:
    return transport in SUPPORTED_TRANSPORTS


def supports_root(server_caps: dict[str, Any], root_name: str) -> bool:
    return root_name in SUPPORTED_ROOTS
