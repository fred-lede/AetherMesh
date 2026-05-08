from __future__ import annotations

from typing import Any

import requests


class MCPClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8001", auth_token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if auth_token:
            self.session.headers["Authorization"] = f"Bearer {auth_token}"

    def list_servers(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/mcp/v1/servers", timeout=10)
        resp.raise_for_status()
        return resp.json().get("servers", [])

    def list_tools(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/mcp/v1/tools", timeout=10)
        resp.raise_for_status()
        return resp.json().get("tools", [])

    def create_session(self, server_name: str) -> dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/mcp/v1/servers/{server_name}/session", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/mcp/v1/tools/{tool_name}/call", json={"arguments": arguments}, timeout=60)
        resp.raise_for_status()
        return resp.json()
