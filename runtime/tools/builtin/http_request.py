from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from runtime.tools.tool_registry import ToolDescriptor, ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("builtin.http_request")

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
            "description": "HTTP method",
            "default": "GET",
        },
        "url": {
            "type": "string",
            "description": "Request URL",
        },
        "headers": {
            "type": "object",
            "description": "Optional HTTP headers as key-value pairs",
            "default": {},
        },
        "body": {
            "type": "string",
            "description": "Request body (for POST/PUT/PATCH). JSON strings will be sent as application/json.",
            "default": "",
        },
        "timeout_s": {
            "type": "integer",
            "description": "Timeout in seconds",
            "default": 30,
        },
    },
    "required": ["url"],
}

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}


def _http_handler(call: ToolCall) -> ToolResult:
    method = str(call.arguments.get("method", "GET")).upper()
    url = str(call.arguments.get("url", ""))
    headers = call.arguments.get("headers", {}) or {}
    body = str(call.arguments.get("body", "") or "")
    timeout_s = int(call.arguments.get("timeout_s", 30))

    if not url:
        return ToolResult(call=call, output="No URL provided", is_error=True, duration_ms=0)
    if method not in _METHODS:
        return ToolResult(call=call, output=f"Unsupported HTTP method: {method}", is_error=True, duration_ms=0)

    request_headers = dict(headers) if isinstance(headers, dict) else {}
    data: str | None = body if body else None

    if data and "content-type" not in {k.lower() for k in request_headers}:
        request_headers.setdefault("Content-Type", "application/json")

    start = time.monotonic()
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=request_headers,
            data=data,
            timeout=timeout_s,
        )
        duration_ms = (time.monotonic() - start) * 1000

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_output = json.dumps(response.json(), indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                body_output = response.text[:50_000]
        else:
            body_output = response.text[:50_000]

        output = (
            f"HTTP {response.status_code} ({duration_ms:.0f}ms)\n"
            f"Content-Type: {content_type}\n\n"
            f"{body_output}"
        )
        return ToolResult(call=call, output=output, duration_ms=duration_ms, is_error=not response.ok)
    except requests.Timeout:
        duration_ms = (time.monotonic() - start) * 1000
        return ToolResult(call=call, output=f"Request timed out after {timeout_s}s", is_error=True, duration_ms=duration_ms)
    except requests.ConnectionError as e:
        duration_ms = (time.monotonic() - start) * 1000
        return ToolResult(call=call, output=f"Connection error: {e}", is_error=True, duration_ms=duration_ms)
    except Exception as e:
        duration_ms = (time.monotonic() - start) * 1000
        logger.exception("HTTP request failed")
        return ToolResult(call=call, output=str(e), is_error=True, duration_ms=duration_ms)


HTTP_DESCRIPTOR = ToolDescriptor(
    name="http_request",
    description="Make an HTTP request to an external API or service. Supports GET, POST, PUT, PATCH, DELETE, HEAD.",
    input_schema=INPUT_SCHEMA,
    handler=_http_handler,
    source="builtin",
    requires_confirmation=True,
    timeout_s=60,
)


def register(registry: ToolRegistry | None = None) -> ToolDescriptor:
    reg = registry or default_registry
    reg.register(HTTP_DESCRIPTOR)
    logger.info("Registered builtin tool: http_request")
    return HTTP_DESCRIPTOR


def available() -> bool:
    return True
