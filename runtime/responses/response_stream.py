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


def make_function_call_queue_event(
    response_id: str,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str = "",
) -> dict[str, Any]:
    return {
        "type": "response.function_call.queue",
        "data": {
            "type": "response.function_call.queue",
            "response": {"id": response_id},
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "parsed_arguments": _try_parse_args(arguments),
                "arguments": arguments,
                "status": "in_progress",
            },
        },
    }


def make_function_call_call_event(
    response_id: str,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str = "",
) -> dict[str, Any]:
    return {
        "type": "response.function_call.call",
        "data": {
            "type": "response.function_call.call",
            "response": {"id": response_id},
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "parsed_arguments": _try_parse_args(arguments),
                "arguments": arguments,
                "status": "completed",
            },
        },
    }


def make_function_call_output_event(
    response_id: str,
    call_id: str,
    output: str,
    is_error: bool = False,
) -> dict[str, Any]:
    return {
        "type": "response.function_call.output",
        "data": {
            "type": "response.function_call.output",
            "response": {"id": response_id},
            "call_id": call_id,
            "output": output,
        },
    }


def make_function_call_arguments_delta_event(
    response_id: str,
    delta: str,
) -> dict[str, Any]:
    return {
        "type": "response.function_call.arguments.delta",
        "data": {
            "type": "response.function_call.arguments.delta",
            "response": {"id": response_id},
            "delta": delta,
        },
    }


def make_text_delta_event(
    response_id: str,
    delta: str,
    index: int = 0,
) -> dict[str, Any]:
    return {
        "type": "response.text.delta",
        "data": {
            "type": "response.text.delta",
            "response": {"id": response_id},
            "delta": delta,
            "index": index,
        },
    }


def make_output_item_added_event(
    response_id: str,
    item_id: str,
    item_type: str,
    role: str,
    call_id: str = "",
    name: str = "",
    arguments: str = "",
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "type": item_type,
    }
    if role:
        item["role"] = role
    if item_type == "function_call":
        item.update({
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        })
    return {
        "type": "response.output_item.added",
        "data": {
            "type": "response.output_item.added",
            "response": {"id": response_id},
            "item": item,
        },
    }


def make_content_part_added_event(
    response_id: str,
    part: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "response.content_part.added",
        "data": {
            "type": "response.content_part.added",
            "response": {"id": response_id},
            "part": part,
        },
    }


def _try_parse_args(raw: str) -> Any:
    """Try to parse JSON arguments string, return raw on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
