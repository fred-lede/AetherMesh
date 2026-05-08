from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterable

from runtime.responses.output_converter import (
    make_response_start_event,
    make_response_stream_error,
    streaming_chunk_to_response_event,
)

logger = logging.getLogger("responses.stream")


class ResponseStreamEncoder:
    def encode(self, event: dict[str, Any]) -> str:
        event_type = event.get("type", "")
        data = event.get("data", event)
        lines = [
            f"event: {event_type}",
            f"data: {json.dumps(data, ensure_ascii=False, default=str)}",
            "",
        ]
        return "\n".join(lines)

    def encode_done(self) -> str:
        return "event: response.done\ndata: [DONE]\n\n"


response_stream_encoder = ResponseStreamEncoder()


def wrap_streaming_chunks(
    chunks: Iterable[dict[str, Any] | str],
    response_id: str,
    model: str,
) -> Iterable[str]:
    yield response_stream_encoder.encode(make_response_start_event(response_id, model))

    for chunk in chunks:
        if isinstance(chunk, str):
            if chunk == "[DONE]":
                break
            if isinstance(chunk, str):
                try:
                    chunk_data = json.loads(chunk)
                except (json.JSONDecodeError, TypeError):
                    continue
                events = streaming_chunk_to_response_event(chunk_data, response_id, model)
                for event in events:
                    yield response_stream_encoder.encode(event)
            continue

        if isinstance(chunk, dict):
            if "error" in chunk:
                yield response_stream_encoder.encode(
                    make_response_stream_error(response_id, str(chunk["error"]))
                )
                yield response_stream_encoder.encode_done()
                return

            events = streaming_chunk_to_response_event(chunk, response_id, model)
            for event in events:
                yield response_stream_encoder.encode(event)

    yield response_stream_encoder.encode({
        "type": "response.completed",
        "data": {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "model": model,
                "status": "completed",
                "output": [],
                "usage": {},
            },
        },
    })
    yield response_stream_encoder.encode_done()
