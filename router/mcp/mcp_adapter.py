from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException, Request

from runtime.mcp.mcp_auth import validate_request
from runtime.mcp.mcp_registry import MCPToolEntry, mcp_registry
from runtime.mcp.mcp_session_manager import mcp_session_manager

logger = logging.getLogger("mcp.adapter")

MCP_PROTOCOL_VERSION = "2025-03-26"


def create_mcp_router() -> APIRouter:
    router = APIRouter(prefix="/mcp")

    @router.post("/v1/servers")
    def list_servers() -> dict[str, Any]:
        servers = mcp_registry.list_servers()
        return {
            "servers": [
                {
                    "name": s.name,
                    "transport": s.transport,
                    "capabilities": sorted(s.capabilities),
                    "enabled": s.enabled,
                }
                for s in servers
            ]
        }

    @router.get("/v1/tools")
    def list_tools() -> dict[str, Any]:
        tools = mcp_registry.list_tools()
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "server_name": t.server_name,
                }
                for t in tools
            ]
        }

    @router.post("/v1/servers/{server_name}/session")
    def create_session(
        server_name: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict[str, Any]:
        auth_result = validate_request({"Authorization": authorization or ""})
        if not auth_result.ok:
            raise HTTPException(status_code=401, detail=auth_result.error)
        try:
            session = mcp_session_manager.get_or_create(server_name)
            return {"session_id": session.session_id, "server": server_name}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/v1/tools/{tool_name}/call")
    def call_tool(
        tool_name: str,
        payload: dict[str, Any] = Body(...),
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict[str, Any]:
        auth_result = validate_request({"Authorization": authorization or ""})
        if not auth_result.ok:
            raise HTTPException(status_code=401, detail=auth_result.error)
        arguments = payload.get("arguments", {})
        entry = mcp_registry.get_tool(tool_name)
        if not entry:
            raise HTTPException(status_code=404, detail=f"MCP tool not found: {tool_name}")
        try:
            session = mcp_session_manager.get_or_create(entry.server_name)
            result = session.call_tool(tool_name, arguments)
            return {"result": result}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

    @router.post("/v1/servers/{server_name}/close")
    def close_session(server_name: str) -> dict[str, Any]:
        sessions = mcp_session_manager.active_sessions()
        for session in sessions:
            if session.server_name == server_name:
                mcp_session_manager.close_session(session.session_id)
        return {"ok": True, "server": server_name}

    return router
