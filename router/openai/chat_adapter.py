from __future__ import annotations

from typing import Any

from fastapi import Body
from fastapi.responses import StreamingResponse

from router.streaming_router import stream_response


def create_chat_completions_route(service: Any):
    def chat_completions(payload: dict[str, Any] = Body(...)):
        if payload.get("stream"):
            return stream_response(service.handle_streaming_chat(payload))
        return service.handle_chat(payload)
    return chat_completions
