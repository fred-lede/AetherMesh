from __future__ import annotations

import json
from typing import Any, Iterable

from fastapi.responses import StreamingResponse


def format_sse_event(item: dict[str, Any] | str) -> str:
    if isinstance(item, str):
        payload = item
    else:
        payload = json.dumps(item, ensure_ascii=True)
    return f"data: {payload}\n\n"


def stream_response(iterator: Iterable[dict[str, Any] | str]) -> StreamingResponse:
    def event_generator() -> Iterable[str]:
        for item in iterator:
            yield format_sse_event(item)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
