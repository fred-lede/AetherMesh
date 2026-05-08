from __future__ import annotations

import logging
from typing import Any

from runtime.mcp.mcp_session_manager import mcp_session_manager
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("mcp.bridge")


class MCPToolBridge:
    def call_mcp_tool(self, call: ToolCall, server_name: str | None = None) -> ToolResult:
        try:
            if server_name:
                session = mcp_session_manager.get_or_create(server_name)
            else:
                from runtime.mcp.mcp_registry import mcp_registry
                tool_entry = mcp_registry.get_tool(call.name)
                if not tool_entry or not tool_entry.server_name:
                    return ToolResult(call=call, output=f"MCP tool '{call.name}' not found in registry", is_error=True)
                session = mcp_session_manager.get_or_create(tool_entry.server_name)

            result = session.call_tool(call.name, call.arguments)
            content = result.get("content", [])
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            return ToolResult(call=call, output="\n".join(text_parts))
        except Exception as e:
            logger.exception("MCP tool call failed: %s", call.name)
            return ToolResult(call=call, output=str(e), is_error=True)


mcp_tool_bridge = MCPToolBridge()
