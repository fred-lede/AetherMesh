from __future__ import annotations

import json
import logging
import queue as q_module
import threading
from typing import Any, AsyncIterable, Iterable

from fastapi import APIRouter, Body, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("responses.adapter")


def _stream_with_keepalive(
    sync_gen: Iterable[str],
    keepalive_interval: float = 8.0,
) -> AsyncIterable[str]:
    _q: q_module.Queue = q_module.Queue(maxsize=100)
    _sentinel = object()
    _response_id: str = ""
    _event_count: int = 0

    def _reader() -> None:
        try:
            for chunk in sync_gen:
                _q.put(chunk)
        except GeneratorExit:
            pass
        except BaseException as e:
            logger.error("Keepalive reader exception: %s", e)
            _q.put(e)
        finally:
            _q.put(_sentinel)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    logger.info("Keepalive reader thread started")

    while True:
        try:
            item = _q.get(timeout=keepalive_interval)
        except q_module.Empty:
            keepalive_data = {
                "type": "response.in_progress",
                "response": {"id": _response_id},
            }
            yield f"data: {json.dumps(keepalive_data, ensure_ascii=False)}\n\n"
            continue
        if item is _sentinel:
            logger.info("Keepalive reader sentinel received, total events: %d", _event_count)
            break
        if isinstance(item, BaseException):
            logger.error("Keepalive reader exception item: %s", item)
            raise item
        if not _response_id and "response.created" in item:
            try:
                data_part = item.split("data: ", 1)[1].rstrip()
                parsed = json.loads(data_part)
                _response_id = parsed.get("response", {}).get("id", "")
                logger.info("Captured response_id: %s", _response_id)
            except (IndexError, json.JSONDecodeError):
                pass
        try:
            data_json = item.split("data: ", 1)[1].rstrip()
            event_type = json.loads(data_json).get("type", "?")
        except (IndexError, json.JSONDecodeError):
            event_type = "?"
        logger.debug("SSE event: %s (id=%s)", event_type, _response_id or "?")
        _event_count += 1
        yield item

    reader_thread.join(timeout=3)
    logger.info("Keepalive wrapper done, total events: %d", _event_count)


def create_responses_router(service: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/responses")
    async def responses(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        user_id = getattr(request.state, "user_id", None)
        api_key_id = getattr(request.state, "api_key_id", None)
        stream = payload.get("stream", False)
        if stream:
            sync_gen = service.handle_streaming_responses(payload, user_id=user_id, api_key_id=api_key_id)
            async_gen = _stream_with_keepalive(sync_gen)
            return StreamingResponse(
                async_gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return service.handle_responses(payload, user_id=user_id, api_key_id=api_key_id)

    @router.get("/v1/responses/{response_id}")
    def get_response(response_id: str) -> dict[str, Any]:
        return service.get_response(response_id)

    @router.delete("/v1/responses/{response_id}")
    def delete_response(response_id: str) -> dict[str, Any]:
        return service.delete_response(response_id)

    @router.patch("/v1/responses/{response_id}")
    def update_response(response_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.update_response(response_id, payload)

    return router
