from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable

from fastapi import Request
from fastapi.responses import StreamingResponse


def format_sse_event(item: dict[str, Any] | str) -> str:
    if isinstance(item, str):
        payload = item
    else:
        payload = json.dumps(item, ensure_ascii=False)
    return f"data: {payload}\n\n"


def stream_response(iterator: Iterable[dict[str, Any] | str]) -> StreamingResponse:
    def event_generator() -> Iterable[str]:
        for item in iterator:
            yield format_sse_event(item)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def async_stream_response(
    iterator: Iterable[dict[str, Any] | str],
    request: Request,
    adapter: Any = None,
) -> StreamingResponse:
    """Async-aware SSE wrapper that monitors client disconnect.
    
    When the client disconnects mid-stream, aborts the underlying HTTP 
    request (if an adapter with abort_stream() is provided) and stops iterating.
    This releases the GPU worker immediately instead of waiting for the 
    per-chunk read timeout (default 30s).
    """

    async def event_generator() -> Iterable[str]:
        loop = asyncio.get_running_loop()
        iter_obj = iter(iterator)

        while True:
            if await request.is_disconnected():
                if adapter is not None:
                    adapter.abort_stream()
                return

            try:
                item = await loop.run_in_executor(None, next, iter_obj)
            except StopIteration:
                return
            else:
                yield format_sse_event(item)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
