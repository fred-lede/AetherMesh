from __future__ import annotations

import logging
from typing import Any

from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("responses.runtime")


class ResponseRuntime:
    def __init__(self) -> None:
        self._responses: dict[str, dict[str, Any]] = {}

    def create_response(self, response_id: str, model: str, content: list[dict[str, Any]]) -> dict[str, Any]:
        response = {
            "id": response_id,
            "object": "response",
            "model": model,
            "status": "completed",
            "content": content,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        self._responses[response_id] = response
        return response

    def get_response(self, response_id: str) -> dict[str, Any] | None:
        return self._responses.get(response_id)

    def update_response(self, response_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        response = self._responses.get(response_id)
        if response:
            response.update(updates)
        return response

    def add_tool_call_output(self, response: dict[str, Any], call: ToolCall, result: ToolResult) -> dict[str, Any]:
        output_block = {
            "type": "tool_call_output",
            "tool_call_id": call.id,
            "output": result.to_text(),
            "is_error": result.is_error,
        }
        response.setdefault("output", []).append(output_block)
        return response

    def create_stream_event(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"type": event_type, "data": data}


response_runtime = ResponseRuntime()
