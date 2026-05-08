from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mcp.auth")


@dataclass
class MCPAuthConfig:
    server_token: str = ""
    token_header: str = "Authorization"
    token_scheme: str = "Bearer"
    enabled: bool = False


@dataclass
class MCPAuthResult:
    ok: bool = False
    server_name: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_default_config = MCPAuthConfig()


def configure_auth(
    server_token: str = "",
    token_header: str = "Authorization",
    token_scheme: str = "Bearer",
    enabled: bool = False,
) -> MCPAuthConfig:
    global _default_config
    _default_config = MCPAuthConfig(
        server_token=server_token,
        token_header=token_header,
        token_scheme=token_scheme,
        enabled=enabled,
    )
    return _default_config


def validate_request(
    headers: dict[str, str],
    expected_token: str = "",
    config: MCPAuthConfig | None = None,
) -> MCPAuthResult:
    cfg = config or _default_config
    if not cfg.enabled:
        return MCPAuthResult(ok=True)

    token = expected_token or cfg.server_token
    if not token:
        logger.warning("MCP auth enabled but no token configured")
        return MCPAuthResult(ok=False, error="MCP authentication not configured")

    header_name = cfg.token_header.lower()
    auth_value = ""
    for key, value in headers.items():
        if key.lower() == header_name:
            auth_value = value
            break

    if not auth_value:
        return MCPAuthResult(ok=False, error="Missing authorization header")

    _, _, received_token = auth_value.partition(f"{cfg.token_scheme} ")
    received_token = received_token.strip()

    if not _constant_time_compare(received_token, token):
        return MCPAuthResult(ok=False, error="Invalid token")

    return MCPAuthResult(ok=True, metadata={"scheme": cfg.token_scheme})


def generate_token(length: int = 32) -> str:
    return secrets.token_urlsafe(length)


def _constant_time_compare(a: str, b: str) -> bool:
    try:
        return hmac.compare_digest(a, b)
    except TypeError:
        return secrets.compare_digest(a.encode(), b.encode())
