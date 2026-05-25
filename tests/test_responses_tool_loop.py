from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any

from runtime.responses.tool_loop import ResponsesToolLoop, DEFAULT_MAX_TURNS
from runtime.responses.response_models import (
    ResponseStatus,
    OutputItemType,
    FunctionCallStatus,
    make_function_call_output,
    make_function_call_output_item,
)
from runtime.responses.input_converter import responses_input_to_messages
from runtime.responses.response_stream import (
    make_function_call_queue_event,
    make_function_call_call_event,
    make_function_call_output_event,
    make_output_item_added_event,
    make_text_delta_event,
    wrap_streaming_chunks,
)
from runtime.tools.tool_registry import ToolRegistry, ToolDescriptor
from runtime.tools.tool_result import ToolCall, ToolResult


# ── Helpers ─────────────────────────────────────────────────────────

def _make_mock_adapter(
    responses: list[dict[str, Any]],
    *,
    stream_responses: list[dict[str, Any]] | None = None,
) -> MagicMock:
    adapter = MagicMock()
    idx = [0]

    def chat(payload: dict[str, Any]) -> dict[str, Any]:
        response = responses[idx[0]] if idx[0] < len(responses) else responses[-1]
        if idx[0] < len(responses) - 1:
            idx[0] += 1
        return response

    def stream(payload: dict[str, Any]) -> Any:
        items = stream_responses or responses
        for item in items:
            yield item

    adapter.chat.side_effect = chat
    adapter.stream.side_effect = stream
    return adapter


def _make_completion(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": model,
        "choices": [
            {
                "message": message,
                "finish_reason": "stop" if not tool_calls else "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _register_tool(
    registry: ToolRegistry,
    name: str,
    *,
    output: str = "OK",
    is_error: bool = False,
) -> None:
    def handler(call: ToolCall) -> ToolResult:
        return ToolResult(call=call, output=output, is_error=is_error)

    registry.register(
        ToolDescriptor(
            name=name,
            description=f"Test tool {name}",
            input_schema={"type": "object", "properties": {}},
            handler=handler,
        )
    )


# ── Unit Tests: Input Converter ──────────────────────────────────────

def test_responses_input_to_messages_simple_string():
    messages = responses_input_to_messages("Hello world")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello world"


def test_responses_input_to_messages_with_instructions():
    messages = responses_input_to_messages("Hello", instructions="You are helpful")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"
    assert messages[1]["role"] == "user"


def test_responses_input_to_messages_with_function_call_output():
    input_value = [
        {"type": "message", "role": "user", "content": "Question?"},
        {"type": "function_call_output", "call_id": "call_123", "output": "Result"},
    ]
    messages = responses_input_to_messages(input_value)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_123"
    assert messages[1]["content"] == "Result"


def test_responses_input_truncation_keeps_last_turns():
    """With truncation='auto', oldest messages should be dropped."""
    items = [
        {"type": "message", "role": "user", "content": f"msg {i}"}
        for i in range(20)
    ]
    messages = responses_input_to_messages(
        items,
        truncation="auto",
        max_tokens=100,
    )
    assert len(messages) <= 8  # min_turns_to_keep * 2


def test_responses_input_truncation_disabled_preserves_all():
    items = [
        {"type": "message", "role": "user", "content": f"msg {i}"}
        for i in range(20)
    ]
    messages = responses_input_to_messages(
        items,
        truncation="disabled",
    )
    assert len(messages) == 20


def test_responses_input_accepts_bare_input_text_part():
    messages = responses_input_to_messages([
        {"type": "input_text", "text": "Hello from Responses"},
    ])

    assert messages == [{"role": "user", "content": "Hello from Responses"}]


def test_responses_input_accepts_bare_text_dict():
    messages = responses_input_to_messages({"text": "Plain text block"})

    assert messages == [{"role": "user", "content": "Plain text block"}]


# ── Unit Tests: Tool Loop Core ───────────────────────────────────────

def test_tool_loop_returns_completed_without_tools():
    registry = ToolRegistry()
    loop = ResponsesToolLoop(registry=registry)
    adapter = _make_mock_adapter([
        _make_completion(content="Final answer")
    ])

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=None,
        instructions="Be nice",
        response_id="resp_test_1",
        model="test-model",
        input_value="Hello",
    )

    assert response.status == ResponseStatus.COMPLETED
    assert len(response.output) == 1
    assert response.output[0].type == OutputItemType.MESSAGE
    assert response.to_dict()["output_text"] == "Final answer"


def test_tool_loop_executes_one_tool_call():
    registry = ToolRegistry()
    _register_tool(registry, "get_weather", output="Sunny, 25C")
    loop = ResponsesToolLoop(registry=registry)

    adapter = _make_mock_adapter([
        _make_completion(
            content="",
            tool_calls=[
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Taipei"}',
                    },
                }
            ],
        ),
        _make_completion(content="The weather in Taipei is sunny, 25C."),
    ])

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }
        ],
        instructions="",
        response_id="resp_test_2",
        model="test-model",
        input_value="What's the weather in Taipei?",
    )

    assert response.status == ResponseStatus.COMPLETED
    assert len(response.output) == 1
    assert "sunny" in response.output[0].to_dict()["content"][0]["text"].lower()


def test_tool_loop_two_turns():
    registry = ToolRegistry()
    _register_tool(registry, "search", output="Results...")
    _register_tool(registry, "calculate", output="42")
    loop = ResponsesToolLoop(registry=registry)

    adapter = _make_mock_adapter([
        _make_completion(
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{}'}}],
        ),
        _make_completion(
            tool_calls=[{"id": "call_2", "type": "function", "function": {"name": "calculate", "arguments": '{}'}}],
        ),
        _make_completion(content="The answer is 42."),
    ])

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[],
        instructions="",
        response_id="resp_test_3",
        model="test-model",
        input_value="Search and calculate",
    )

    assert response.status == ResponseStatus.COMPLETED


def test_tool_loop_max_turns_reached():
    registry = ToolRegistry()
    _register_tool(registry, "loop_tool", output="still looping")
    loop = ResponsesToolLoop(registry=registry, max_turns=3)

    adapter = _make_mock_adapter([
        _make_completion(
            tool_calls=[
                {"id": f"call_{i}", "type": "function", "function": {"name": "loop_tool", "arguments": "{}"}}
            ]
        )
        for i in range(10)
    ])

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[],
        instructions="",
        response_id="resp_test_4",
        model="test-model",
        input_value="Loop forever",
    )

    assert response.status == ResponseStatus.COMPLETED
    adapter.chat.assert_called()
    assert adapter.chat.call_count == 3


def test_tool_loop_unknown_tool_returns_error_in_result():
    loop = ResponsesToolLoop()
    adapter = _make_mock_adapter([
        _make_completion(
            tool_calls=[
                {"id": "call_x", "type": "function", "function": {"name": "nonexistent_tool", "arguments": "{}"}}
            ],
        ),
        _make_completion(content="Got tool result."),
    ])

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[],
        instructions="",
        response_id="resp_test_5",
        model="test-model",
        input_value="Test",
    )

    assert response.status == ResponseStatus.COMPLETED


def test_tool_loop_adapter_error_creates_failed_response():
    def raise_error(payload):
        raise Exception("Connection refused")

    adapter = MagicMock()
    adapter.chat.side_effect = raise_error

    loop = ResponsesToolLoop()
    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=None,
        instructions="",
        response_id="resp_test_err",
        model="test-model",
        input_value="Test",
    )

    assert response.status == ResponseStatus.FAILED
    assert response.error is not None
    assert "Connection refused" in response.error["message"]


def test_tool_loop_accumulates_usage():
    registry = ToolRegistry()
    _register_tool(registry, "fast_tool", output="done")
    loop = ResponsesToolLoop(registry=registry)

    adapter = _make_mock_adapter([
        _make_completion(
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "fast_tool", "arguments": "{}"}}],
            prompt_tokens=10,
            completion_tokens=5,
        ),
        _make_completion(content="Done.", prompt_tokens=15, completion_tokens=20),
    ])

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[],
        instructions="",
        response_id="resp_usage",
        model="test-model",
        input_value="Test",
    )

    assert response.usage.input_tokens == 25
    assert response.usage.output_tokens == 25
    assert response.usage.total_tokens == 50


# ── Unit Tests: SSE Event Helpers ────────────────────────────────────

def test_make_function_call_queue_event():
    event = make_function_call_queue_event(
        response_id="resp_1",
        item_id="item_1",
        call_id="call_1",
        name="test_tool",
        arguments='{"key": "value"}',
    )
    assert event["type"] == "response.function_call.queue"
    assert event["data"]["item"]["call_id"] == "call_1"
    assert event["data"]["item"]["name"] == "test_tool"
    assert event["data"]["item"]["parsed_arguments"] == {"key": "value"}


def test_make_function_call_call_event():
    event = make_function_call_call_event(
        response_id="resp_1",
        item_id="item_2",
        call_id="call_2",
        name="another_tool",
        arguments='{"x": 1}',
    )
    assert event["type"] == "response.function_call.call"
    assert event["data"]["item"]["status"] == "completed"


def test_make_function_call_output_event():
    event = make_function_call_output_event(
        response_id="resp_1",
        call_id="call_3",
        output="Success output",
        is_error=False,
    )
    assert event["type"] == "response.function_call.output"
    assert event["data"]["call_id"] == "call_3"
    assert event["data"]["output"] == "Success output"


def test_make_output_item_added_event():
    event = make_output_item_added_event(
        response_id="resp_1",
        item_id="item_5",
        item_type="function_call",
        role="assistant",
        call_id="call_5",
        name="my_tool",
        arguments='{}',
    )
    assert event["type"] == "response.output_item.added"
    assert event["data"]["item"]["type"] == "function_call"


def test_make_text_delta_event():
    event = make_text_delta_event(response_id="resp_1", delta="Hello")
    assert event["type"] == "response.output_text.delta"
    assert event["data"]["delta"] == "Hello"


def test_wrap_streaming_chunks_completed_event_contains_output_text():
    chunks = [
        {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
    ]

    events = list(wrap_streaming_chunks(chunks, response_id="resp_1", model="test-model"))
    completed = [
        json.loads(event.split("data: ", 1)[1])
        for event in events
        if event.startswith("event: response.completed")
    ][0]

    output = completed["response"]["output"]
    assert output[0]["content"][0]["text"] == "Hello world"
    assert completed["response"]["output_text"] == "Hello world"


# ── Unit Tests: Response Model Helpers ──────────────────────────────

def test_make_function_call_output_model():
    item = make_function_call_output(
        call_id="call_99",
        name="calculator",
        arguments='{"expr": "2+2"}',
        status="completed",
    )
    d = item.to_dict()
    assert d["type"] == "function_call"
    assert d["call_id"] == "call_99"
    assert d["name"] == "calculator"
    assert d["status"] == "completed"


def test_make_function_call_output_item_model():
    item = make_function_call_output_item(
        call_id="call_100",
        output="Hello from tool",
        is_error=False,
    )
    d = item.to_dict()
    assert d["type"] == "function_call_output"
    assert d["call_id"] == "call_100"
    assert d["output"] == "Hello from tool"


def test_function_call_status_enum():
    assert FunctionCallStatus.IN_PROGRESS.value == "in_progress"
    assert FunctionCallStatus.COMPLETED.value == "completed"
    assert FunctionCallStatus.CANCELLED_AND_RETRIED.value == "cancelled_and_retried"


def test_response_status_requires_action():
    assert ResponseStatus.REQUIRES_ACTION.value == "requires_action"


def test_output_item_type_function_call():
    assert OutputItemType.FUNCTION_CALL.value == "function_call"
