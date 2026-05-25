from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterable

from runtime.responses.output_converter import (
    make_response_start_event,
    make_response_stream_error,
    make_response_completed_event,
)

logger = logging.getLogger("responses.stream")


class ResponseStreamEncoder:
    """Encode events as OpenAI Responses API SSE format: `event:\ndata:\n\n`."""

    def encode(self, event: dict[str, Any]) -> str:
        event_type = event.get("type", "event")
        data = event.get("data", event)
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    def encode_done(self) -> str:
        return "event: response.done\ndata: [DONE]\n\n"


response_stream_encoder = ResponseStreamEncoder()


def wrap_streaming_chunks(
    chunks: Iterable[dict[str, Any] | str],
    response_id: str,
    model: str,
) -> Iterable[str]:
    yield response_stream_encoder.encode(make_response_start_event(response_id, model))

    emitted_done = False
    text_parts: list[str] = []
    item_id = f"msg_{uuid.uuid4().hex[:16]}"
    item_started = False
    content_started = False

    for chunk in chunks:
        if isinstance(chunk, str):
            if chunk == "[DONE]":
                break
            try:
                chunk = json.loads(chunk)
            except (json.JSONDecodeError, TypeError):
                continue

        if isinstance(chunk, dict):
            if "error" in chunk:
                yield response_stream_encoder.encode(
                    make_response_stream_error(response_id, str(chunk["error"]))
                )
                yield response_stream_encoder.encode_done()
                return

            for content in _chunk_text_deltas(chunk):
                if content and not item_started:
                    yield response_stream_encoder.encode(
                        _make_output_item_added_event(response_id, item_id)
                    )
                    item_started = True
                if content and not content_started:
                    yield response_stream_encoder.encode(
                        _make_content_part_added_event(response_id, item_id)
                    )
                    content_started = True
                if content:
                    text_parts.append(content)
                    yield response_stream_encoder.encode(
                        _make_output_text_delta_event(response_id, item_id, content)
                    )

            if _chunk_finished(chunk):
                full_text = "".join(text_parts)
                if content_started:
                    yield response_stream_encoder.encode(
                        _make_output_text_done_event(response_id, item_id, full_text)
                    )
                    yield response_stream_encoder.encode(
                        _make_content_part_done_event(response_id, item_id, full_text)
                    )
                if item_started:
                    yield response_stream_encoder.encode(
                        _make_output_item_done_event(response_id, item_id, full_text)
                    )
                yield response_stream_encoder.encode(
                    make_response_completed_event(
                        response_id,
                        model,
                        usage=_responses_usage(chunk),
                        output_text=full_text,
                    )
                )
                emitted_done = True
                break

    if not emitted_done:
        full_text = "".join(text_parts)
        if content_started:
            yield response_stream_encoder.encode(
                _make_output_text_done_event(response_id, item_id, full_text)
            )
            yield response_stream_encoder.encode(
                _make_content_part_done_event(response_id, item_id, full_text)
            )
        if item_started:
            yield response_stream_encoder.encode(
                _make_output_item_done_event(response_id, item_id, full_text)
            )
        yield response_stream_encoder.encode(
            make_response_completed_event(response_id, model, output_text=full_text)
        )

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
        "type": "response.output_text.delta",
        "data": {
            "type": "response.output_text.delta",
            "response": {"id": response_id},
            "delta": delta,
            "index": index,
        },
    }


def _chunk_text_deltas(chunk: dict[str, Any]) -> list[str]:
    deltas: list[str] = []
    choices = chunk.get("choices", [])
    if not isinstance(choices, list):
        return deltas
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content", "")
            if content:
                deltas.append(str(content))
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content", "")
            if content:
                deltas.append(str(content))
    return deltas


def _chunk_finished(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices", [])
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict) and bool(choice.get("finish_reason"))
        for choice in choices
    )


def _responses_usage(chunk: dict[str, Any]) -> dict[str, Any]:
    usage = chunk.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _make_output_item_added_event(response_id: str, item_id: str) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "data": {
            "type": "response.output_item.added",
            "response": {"id": response_id},
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    }


def _make_content_part_added_event(response_id: str, item_id: str) -> dict[str, Any]:
    return {
        "type": "response.content_part.added",
        "data": {
            "type": "response.content_part.added",
            "response": {"id": response_id},
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    }


def _make_output_text_delta_event(response_id: str, item_id: str, delta: str) -> dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "data": {
            "type": "response.output_text.delta",
            "response": {"id": response_id},
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
        },
    }


def _make_output_text_done_event(response_id: str, item_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "response.output_text.done",
        "data": {
            "type": "response.output_text.done",
            "response": {"id": response_id},
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
        },
    }


def _make_content_part_done_event(response_id: str, item_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "response.content_part.done",
        "data": {
            "type": "response.content_part.done",
            "response": {"id": response_id},
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        },
    }


def _make_output_item_done_event(response_id: str, item_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "data": {
            "type": "response.output_item.done",
            "response": {"id": response_id},
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }],
            },
        },
    }


def _capture_text_delta(event: dict[str, Any], text_parts: list[str]) -> None:
    if event.get("type") not in {"response.output_text.delta", "response.text.delta"}:
        return
    data = event.get("data", {})
    if isinstance(data, dict):
        delta = data.get("delta", "")
        if delta:
            text_parts.append(str(delta))


def _attach_completed_output(event: dict[str, Any], text: str) -> None:
    if event.get("type") != "response.completed" or not text:
        return
    data = event.get("data", {})
    if not isinstance(data, dict):
        return
    response = data.get("response")
    if not isinstance(response, dict):
        return
    if response.get("output"):
        return
    response["output"] = make_response_completed_event(
        str(response.get("id") or ""),
        str(response.get("model") or ""),
        output_text=text,
    )["data"]["response"]["output"]
    response["output_text"] = text


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
