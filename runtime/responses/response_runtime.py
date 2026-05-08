from __future__ import annotations

import logging
import time
from typing import Any

from runtime.responses.response_models import (
    ResponseObject,
    ResponseStatus,
    ResponseUsage,
    OutputItem,
    OutputItemType,
    ContentPart,
    ContentPartType,
    make_text_output,
    make_tool_call_output,
)

logger = logging.getLogger("responses.runtime")


class ResponseRuntime:
    def __init__(self) -> None:
        self._responses: dict[str, ResponseObject] = {}
        self._max_responses: int = 1000

    def register(self, response: ResponseObject) -> None:
        self._responses[response.id] = response
        if len(self._responses) > self._max_responses:
            oldest = min(self._responses.keys(), key=lambda k: self._responses[k].created)
            del self._responses[oldest]

    def get(self, response_id: str) -> ResponseObject | None:
        return self._responses.get(response_id)

    def get_response_dict(self, response_id: str) -> dict[str, Any] | None:
        resp = self._responses.get(response_id)
        return resp.to_dict() if resp else None

    def update_status(self, response_id: str, status: ResponseStatus) -> bool:
        resp = self._responses.get(response_id)
        if resp:
            resp.status = status
            return True
        return False

    def add_output(self, response_id: str, item: OutputItem) -> bool:
        resp = self._responses.get(response_id)
        if resp:
            resp.output.append(item)
            return True
        return False

    def update_usage(self, response_id: str, usage: ResponseUsage) -> bool:
        resp = self._responses.get(response_id)
        if resp:
            resp.usage = usage
            return True
        return False

    def set_error(self, response_id: str, message: str, code: str = "server_error") -> bool:
        resp = self._responses.get(response_id)
        if resp:
            resp.status = ResponseStatus.FAILED
            resp.error = {"message": message, "code": code, "type": "server_error"}
            return True
        return False

    def from_completion(
        self,
        completion: dict[str, Any],
        model: str,
        response_id: str = "",
        instructions: str = "",
        previous_response_id: str = "",
    ) -> ResponseObject:
        from runtime.responses.output_converter import chat_completion_to_response
        response = chat_completion_to_response(
            completion=completion,
            model=model,
            response_id=response_id,
            instructions=instructions,
            previous_response_id=previous_response_id,
        )
        self.register(response)
        return response

    def list_responses(self, limit: int = 20) -> list[dict[str, Any]]:
        sorted_responses = sorted(
            self._responses.values(),
            key=lambda r: r.created,
            reverse=True,
        )
        return [r.to_dict() for r in sorted_responses[:limit]]

    def delete(self, response_id: str) -> bool:
        return self._responses.pop(response_id, None) is not None

    def clear(self) -> None:
        self._responses.clear()

    @property
    def count(self) -> int:
        return len(self._responses)


response_runtime = ResponseRuntime()
