from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from typing import Any

from runtime.mcp.mcp_registry import MCPServerEntry, mcp_registry

logger = logging.getLogger("mcp.session")

MCP_JSON_RPC_VERSION = "2.0"


class MCPSession:
    def __init__(self, server_entry: MCPServerEntry) -> None:
        self._entry = server_entry
        self._process: subprocess.Popen | None = None
        self._session_id: str = f"mcp_ses_{uuid.uuid4().hex[:12]}"
        self._capabilities: dict[str, Any] = {}
        self._initialized = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def server_name(self) -> str:
        return self._entry.name

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> dict[str, Any]:
        if self._entry.is_stdio:
            self._process = subprocess.Popen(
                [self._entry.command] + self._entry.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        result = self._send_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "AetherMesh", "version": "4.0.0"},
        })
        self._capabilities = result.get("capabilities", {})
        self._send_notification("notifications/initialized")
        self._initialized = True
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._send_request("tools/call", {"name": name, "arguments": arguments})

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request = {
            "jsonrpc": MCP_JSON_RPC_VERSION,
            "id": str(uuid.uuid4().hex[:8]),
            "method": method,
            "params": params,
        }
        if self._entry.is_stdio and self._process and self._process.stdin:
            line = json.dumps(request, ensure_ascii=True) + "\n"
            self._process.stdin.write(line)
            self._process.stdin.flush()
            if self._process.stdout:
                response_line = self._process.stdout.readline()
                response = json.loads(response_line)
                if "error" in response:
                    raise RuntimeError(f"MCP error: {response['error']}")
                return response.get("result", {})
        raise RuntimeError(f"MCP transport not supported: {self._entry.transport}")

    def _send_notification(self, method: str) -> None:
        notification = {
            "jsonrpc": MCP_JSON_RPC_VERSION,
            "method": method,
        }
        if self._entry.is_stdio and self._process and self._process.stdin:
            self._process.stdin.write(json.dumps(notification, ensure_ascii=True) + "\n")
            self._process.stdin.flush()

    def close(self) -> None:
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
        self._initialized = False


class MCPSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, MCPSession] = {}

    def get_or_create(self, server_name: str) -> MCPSession:
        entry = mcp_registry.get_server(server_name)
        if not entry:
            raise ValueError(f"MCP server not found: {server_name}")
        for session in self._sessions.values():
            if session.server_name == server_name and session.initialized:
                return session
        session = MCPSession(entry)
        session.initialize()
        for tool in session.list_tools():
            mcp_registry.register_tool(
                __import__("runtime.mcp.mcp_registry", fromlist=["MCPToolEntry"]).MCPToolEntry(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=server_name,
                )
            )
        self._sessions[session.session_id] = session
        return session

    def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            session.close()

    def close_all(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()

    def active_sessions(self) -> list[MCPSession]:
        return list(self._sessions.values())


mcp_session_manager = MCPSessionManager()
