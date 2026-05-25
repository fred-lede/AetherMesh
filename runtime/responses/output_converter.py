from __future__ import annotations

import time
import uuid
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


def chat_completion_to_response(
    completion: dict[str, Any],
    model: str,
    response_id: str = "",
    instructions: str = "",
    previous_response_id: str = "",
    metadata: dict[str, Any] | None = None,
    include: list[str] | None = None,
) -> ResponseObject:
    resp = ResponseObject(
        id=response_id or f"resp_{uuid.uuid4().hex[:24]}",
        model=model,
        status=ResponseStatus.COMPLETED,
        instructions=instructions,
        previous_response_id=previous_response_id,
        metadata=metadata or {},
    )

    usage_data = completion.get("usage", {})
    resp.usage = ResponseUsage(
        input_tokens=usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0)),
        output_tokens=usage_data.get("completion_tokens", usage_data.get("output_tokens", 0)),
        total_tokens=usage_data.get("total_tokens", 0),
    )

    include_set = set(include or [])

    choices = completion.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        content = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        reasoning = message.get("reasoning")

        output_items = _message_to_output_items(content, tool_calls, reasoning=reasoning)
        resp.output.extend(output_items)

        if "reasoning" in include_set and reasoning:
            resp.output.append(_make_reasoning_output(str(reasoning)))

    return resp


def _message_to_output_items(
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: Any = None,
) -> list[OutputItem]:
    items: list[OutputItem] = []

    if content:
        items.append(make_text_output(content))

    if reasoning:
        items.append(_make_reasoning_output(str(reasoning)))

    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            items.append(make_tool_call_output(
                tool_call_id=str(tc.get("id", "")),
                tool_name=str(fn.get("name", "")),
                arguments=_stringify_arguments(fn.get("arguments", "{}")),
            ))

    return items


def _make_reasoning_output(text: str) -> OutputItem:
    return OutputItem(
        type=OutputItemType.REASONING,
        content=[ContentPart(type=ContentPartType.REASONING, text=text)],
    )


def error_response(
    model: str,
    error_message: str,
    error_code: str = "server_error",
    response_id: str = "",
) -> ResponseObject:
    return ResponseObject(
        id=response_id or f"resp_{uuid.uuid4().hex[:24]}",
        model=model,
        status=ResponseStatus.FAILED,
        error={
            "message": error_message,
            "code": error_code,
            "type": "server_error",
        },
    )


def streaming_chunk_to_response_event(
    chunk: dict[str, Any],
    response_id: str,
    model: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    choices = chunk.get("choices", [])
    for choice in choices:
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        content = delta.get("content", "")
        if content:
            events.append({
                "type": "response.output_item.added",
                "data": {
                    "type": "response.output_item.added",
                    "item": {"id": f"item_{uuid.uuid4().hex[:16]}", "type": "message", "role": "assistant", "content": []},
                },
            })
            events.append({
                "type": "response.content_part.added",
                "data": {
                    "type": "response.content_part.added",
                    "part": {"type": "output_text", "text": ""},
                },
            })
            events.append({
                "type": "response.text.delta",
                "data": {"type": "response.text.delta", "delta": content, "index": 0},
            })
            events.append({
                "type": "response.text.done",
                "data": {"type": "response.text.done", "text": content, "index": 0},
            })

        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            arguments = fn.get("arguments", "")
            tool_call_id = str(tc.get("id", ""))
            if tool_name:
                events.append({
                    "type": "response.output_item.added",
                    "data": {
                        "type": "response.output_item.added",
                        "item": {
                            "id": f"item_{uuid.uuid4().hex[:16]}",
                            "type": "tool_call",
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "arguments": arguments,
                        },
                    },
                })

        if finish_reason:
            usage = chunk.get("usage", {})
            events.append({
                "type": "response.completed",
                "data": {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "model": model,
                        "status": "completed",
                        "usage": {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    },
                },
            })

    return events


def _stringify_arguments(args: Any) -> str:
    if isinstance(args, str):
        return args
    import json
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))


def make_response_start_event(
    response_id: str,
    model: str,
) -> dict[str, Any]:
    return {
        "type": "response.created",
        "data": {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "model": model,
                "status": "in_progress",
                "output": [],
                "usage": {},
            },
        },
    }


def make_response_stream_error(
    response_id: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "type": "response.failed",
        "data": {
            "type": "response.failed",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "failed",
                "error": {"message": error_message, "type": "server_error", "code": "server_error"},
            },
        },
    }


def make_response_completed_event(
    response_id: str,
    model: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "data": {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "model": model,
                "status": "completed",
                "output": [],
                "usage": usage or {},
            },
        },
    }
