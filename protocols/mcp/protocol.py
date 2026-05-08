from __future__ import annotations

import json
import uuid
from typing import Any

MCP_JSON_RPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-03-26"


def create_request(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": MCP_JSON_RPC_VERSION,
        "id": uuid.uuid4().hex[:8],
        "method": method,
        "params": params or {},
    }


def create_notification(method: str) -> dict[str, Any]:
    return {
        "jsonrpc": MCP_JSON_RPC_VERSION,
        "method": method,
    }


def create_success_response(request_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": MCP_JSON_RPC_VERSION,
        "id": request_id,
        "result": result,
    }


def create_error_response(request_id: str, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": MCP_JSON_RPC_VERSION,
        "id": request_id,
        "error": error,
    }


def serialize_message(msg: dict[str, Any]) -> str:
    return json.dumps(msg, ensure_ascii=True) + "\n"
