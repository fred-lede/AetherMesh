from __future__ import annotations

from typing import Any

from fastapi import Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from router.streaming_router import async_stream_response
from runtime.orchestration.structured_output import StructuredOutputError


def create_chat_completions_route(service: Any):
    async def chat_completions(request: Request, payload: dict[str, Any] = Body(...)):
        user_id = getattr(request.state, "user_id", None)
        api_key_id = getattr(request.state, "api_key_id", None)
        if payload.get("stream"):
            result = service.handle_streaming_chat(payload, user_id=user_id, api_key_id=api_key_id)
            return await async_stream_response(result.iterator, request, adapter=result.adapter)
        try:
            return service.handle_chat(payload, user_id=user_id, api_key_id=api_key_id)
        except StructuredOutputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return chat_completions
