from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SERVER_TOOL_NAMES = frozenset({"web_search", "web_fetch"})


@dataclass(frozen=True)
class ServerToolPolicyResult:
    listed_tools: set[str]
    forced_tool: str | None
    mode: str = "reject"
    error: str | None = None

    @property
    def has_server_tools(self) -> bool:
        return bool(self.listed_tools)

    @property
    def should_handle_locally(self) -> bool:
        return self.mode == "local" and self.forced_tool in self.listed_tools


def server_tool_name(tool: dict[str, Any]) -> str | None:
    name = str(tool.get("name") or "").strip()
    if name in SERVER_TOOL_NAMES:
        return name

    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        for candidate in SERVER_TOOL_NAMES:
            if tool_type == candidate or tool_type.startswith(f"{candidate}_"):
                return candidate
    return None


def listed_server_tools(payload: dict[str, Any]) -> set[str]:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = server_tool_name(tool)
        if name:
            names.add(name)
    return names


def forced_server_tool(payload: dict[str, Any]) -> str | None:
    tool_choice = payload.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "tool":
        return None
    name = str(tool_choice.get("name") or "").strip()
    return name if name in SERVER_TOOL_NAMES else None


def evaluate_server_tool_policy(
    payload: dict[str, Any],
    *,
    provider: str,
    mode: str = "reject",
    local_web_tools_enabled: bool | None = None,
) -> ServerToolPolicyResult:
    policy_mode = _normalize_mode(mode, local_web_tools_enabled=local_web_tools_enabled)
    listed = listed_server_tools(payload)
    forced = forced_server_tool(payload)
    if not listed:
        return ServerToolPolicyResult(listed_tools=set(), forced_tool=forced, mode=policy_mode)

    if policy_mode == "passthrough":
        return ServerToolPolicyResult(listed_tools=listed, forced_tool=forced, mode=policy_mode)

    if policy_mode == "local" and forced in listed:
        return ServerToolPolicyResult(listed_tools=listed, forced_tool=forced, mode=policy_mode)

    if forced:
        return ServerToolPolicyResult(
            listed_tools=listed,
            forced_tool=forced,
            mode=policy_mode,
            error=(
                f"tool_choice forces Anthropic server tool {forced!r}, but AIIH local "
                "web server tools are not enabled for this OpenAI-compatible provider. "
                "Set AIIH_SERVER_TOOL_MODE=local to let AIIH handle it, or "
                "AIIH_SERVER_TOOL_MODE=passthrough to leave it to the client/runtime."
            ),
        )

    return ServerToolPolicyResult(
        listed_tools=listed,
        forced_tool=forced,
        mode=policy_mode,
        error=(
            f"Provider {provider!r} receives Anthropic server tools "
            f"{sorted(listed)!r}, but OpenAI-compatible upstreams cannot execute "
            "web_search/web_fetch server-tool semantics. Enable AIIH local web tools, "
            "use passthrough/client search, a native Anthropic transport, or remove these tools from the request."
        ),
    )


def _normalize_mode(mode: str, *, local_web_tools_enabled: bool | None) -> str:
    if local_web_tools_enabled:
        return "local"
    normalized = str(mode or "reject").strip().lower()
    if normalized in {"local", "passthrough", "reject"}:
        return normalized
    return "reject"
