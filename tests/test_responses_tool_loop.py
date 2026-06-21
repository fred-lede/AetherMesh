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
from runtime.responses.response_stream import ResponseStreamEncoder
from runtime.orchestration.openai_handler import RouterService


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


def test_ollama_model_context_cap_is_applied(monkeypatch):
    service = RouterService()
    monkeypatch.setattr(
        service,
        "_find_registry_model",
        lambda model: {"ollama_options": {"num_ctx": 32768}},
    )

    payload = service._normalize_payload_for_provider(
        {"model": "gemma4:12b", "messages": [], "options": {"num_ctx": 100000}},
        "ollama",
    )

    assert payload["options"]["num_ctx"] == 32768


def test_ollama_worker_context_cap_overrides_model_cap(monkeypatch):
    service = RouterService()
    monkeypatch.setattr(
        service,
        "_find_registry_model",
        lambda model: {
            "ollama_options": {"num_ctx": 32768},
            "worker_bindings": [{
                "base_url": "http://192.168.1.123:11434",
                "ollama_options": {"num_ctx": 16384},
            }],
        },
    )

    payload = service._normalize_payload_for_provider(
        {"model": "gemma4:12b", "messages": [], "options": {"num_ctx": 100000}},
        "ollama",
        {"base_url": "http://192.168.1.123:11434"},
    )

    assert payload["options"]["num_ctx"] == 16384


def test_alternate_ollama_fallback_does_not_change_requested_model():
    service = RouterService()
    service.registry = {
        "models": [
            {
                "name": "qwen3-coder:30b",
                "provider": "ollama",
                "capabilities": ["chat", "tools"],
                "worker_bindings": [{"base_url": "http://127.0.0.1:11434"}],
            },
            {
                "name": "gemma4:12b",
                "provider": "ollama",
                "capabilities": ["chat", "tools"],
                "worker_bindings": [{"base_url": "http://127.0.0.1:11435"}],
            },
        ]
    }

    fallback = service._alternate_ollama_fallback(
        {"model": "qwen3-coder:30b", "messages": []},
        excluded_base_url="http://127.0.0.1:11434",
    )

    assert fallback is None


def test_responses_input_to_messages_with_instructions():
    messages = responses_input_to_messages("Hello", instructions="You are helpful")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"
    assert messages[1]["role"] == "user"


def test_responses_input_maps_developer_role_to_system():
    messages = responses_input_to_messages([
        {"type": "message", "role": "developer", "content": "Follow policy."},
        {"type": "message", "role": "user", "content": "Hello"},
    ])

    assert messages[0] == {"role": "system", "content": "Follow policy."}
    assert messages[1] == {"role": "user", "content": "Hello"}


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


def test_responses_input_round_trips_function_call_and_output():
    input_value = [
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "list_files",
            "arguments": '{"path":"."}',
        },
        {"type": "function_call_output", "call_id": "call_123", "output": "README.md"},
    ]

    messages = responses_input_to_messages(input_value)

    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"] == [{
        "id": "call_123",
        "type": "function",
        "function": {"name": "list_files", "arguments": '{"path":"."}'},
    }]
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "README.md",
    }


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
    payloads = _payloads(events)
    completed = [p for p in payloads if p["type"] == "response.completed"][0]

    output = completed["response"]["output"]
    assert output[0]["content"][0]["text"] == "Hello world"
    assert completed["response"]["output_text"] == "Hello world"

    deltas = [p for p in payloads if p["type"] == "response.output_text.delta"]
    assert deltas[0]["item_id"].startswith("msg_")
    assert deltas[0]["output_index"] == 0
    assert deltas[0]["content_index"] == 0


def test_wrap_streaming_chunks_emits_done_marker():
    chunks = [
        {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
    ]

    events = list(wrap_streaming_chunks(chunks, response_id="resp_1", model="test-model"))

    assert events[-1] == "data: [DONE]\n\n"


def test_wrap_streaming_chunks_returns_client_function_call():
    chunks = [{
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "list_files", "arguments": "{}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }]

    payloads = _payloads(list(wrap_streaming_chunks(chunks, response_id="resp_1", model="test-model")))

    assert any(p["type"] == "response.function_call_arguments.delta" for p in payloads)
    completed = next(p for p in payloads if p["type"] == "response.completed")
    assert completed["response"]["output"][0]["name"] == "list_files"


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


# ── E2E Tests: Streaming Tool Loop ──────────────────────────────────

def test_streaming_tool_loop_no_tools():
    registry = ToolRegistry()
    loop = ResponsesToolLoop(registry=registry)
    adapter = _make_mock_adapter(
        [_make_completion(content="Hello from stream")],
    )
    encoder = ResponseStreamEncoder()

    events = list(loop.run_streaming(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=None,
        instructions="",
        response_id="resp_stream_1",
        model="test-model",
        input_value="Hi",
        encoder=encoder,
    ))

    payloads = _payloads(events)
    types = [p["type"] for p in payloads]
    assert "response.created" in types
    assert "response.completed" in types
    completed = next(p for p in payloads if p["type"] == "response.completed")
    assert completed["response"]["status"] == "completed"


def test_streaming_tool_loop_one_tool():
    registry = ToolRegistry()
    _register_tool(registry, "search", output="search results")
    loop = ResponsesToolLoop(registry=registry)

    def stream_turn_1(payload):
        yield {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                    "function": {"name": "search", "arguments": '{"q": "test"}'}}],
                },
                "finish_reason": "tool_calls",
            }],
        }

    def stream_turn_2(payload):
        yield {"choices": [{"delta": {"role": "assistant", "content": "Found: "}, "finish_reason": None}]}
        yield {"choices": [{"delta": {"content": "search results"}, "finish_reason": "stop"}]}
        yield "[DONE]"

    adapter = MagicMock()
    adapter.stream.side_effect = [stream_turn_1({}), stream_turn_2({})]
    encoder = ResponseStreamEncoder()

    events = list(loop.run_streaming(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}}}],
        instructions="",
        response_id="resp_stream_2",
        model="test-model",
        input_value="Search something",
        encoder=encoder,
    ))

    payloads = _payloads(events)
    types = [p["type"] for p in payloads]
    assert "response.function_call.queue" in types
    assert "response.function_call.arguments.delta" in types
    assert "response.function_call.call" in types
    assert "response.function_call.output" in types
    assert "response.completed" in types
    completed = next(p for p in payloads if p["type"] == "response.completed")
    assert completed["response"]["status"] == "completed"


def test_streaming_tool_loop_empty_choices():
    """Streaming should not crash on empty choices list."""
    registry = ToolRegistry()
    loop = ResponsesToolLoop(registry=registry)
    adapter = MagicMock()
    adapter.stream.return_value = iter([
        {"choices": []},
        {"choices": [{"delta": {"content": "survived"}, "finish_reason": "stop"}]},
        "[DONE]",
    ])
    encoder = ResponseStreamEncoder()

    events = list(loop.run_streaming(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=None,
        instructions="",
        response_id="resp_stream_ec",
        model="test-model",
        input_value="Hi",
        encoder=encoder,
    ))

    payloads = _payloads(events)
    completed = next(p for p in payloads if p["type"] == "response.completed")
    assert completed["response"]["status"] == "completed"
    assert "survived" in completed["response"].get("output_text", "")


def test_tool_loop_empty_choices_run():
    """Non-streaming should not crash on empty choices list."""
    loop = ResponsesToolLoop()
    adapter = MagicMock()
    adapter.chat.return_value = {"choices": []}

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=None,
        instructions="",
        response_id="resp_ec",
        model="test-model",
        input_value="Hi",
    )

    assert response.status == ResponseStatus.FAILED


def test_tool_loop_run_with_client_tools_cleanup():
    """run_with_client_tools must unregister temp tools after run."""
    registry = ToolRegistry()
    _register_tool(registry, "builtin_tool", output="builtin")
    loop = ResponsesToolLoop(registry=registry)

    adapter = _make_mock_adapter([
        _make_completion(
            content="",
            tool_calls=[{
                "id": "call_c1", "type": "function",
                "function": {"name": "client_tool", "arguments": "{}"},
            }],
        ),
        _make_completion(content="done"),
    ])

    client_tools = [{
        "type": "function",
        "function": {"name": "client_tool", "description": "Client tool", "parameters": {"type": "object", "properties": {}}},
    }]

    assert registry.resolve("client_tool") is None
    response, _ = loop.run_with_client_tools(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=client_tools,
        instructions="",
        response_id="resp_ct",
        model="test-model",
        input_value="Use client tool",
    )
    assert response.status == ResponseStatus.COMPLETED
    assert registry.resolve("client_tool") is None, "client_tool was not unregistered"


def test_tool_loop_parallel_tool_calls():
    """Multiple tool calls in one turn should all execute."""
    registry = ToolRegistry()
    _register_tool(registry, "tool_a", output="result_a")
    _register_tool(registry, "tool_b", output="result_b")
    loop = ResponsesToolLoop(registry=registry)

    adapter = _make_mock_adapter([
        _make_completion(
            content="",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
            ],
        ),
        _make_completion(content="Both done."),
    ])

    response = loop.run(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[],
        instructions="",
        response_id="resp_par",
        model="test-model",
        input_value="Run both",
    )

    assert response.status == ResponseStatus.COMPLETED
    assert "Both done." in response.to_dict().get("output_text", "")


def test_wrap_streaming_chunks_in_progress_event():
    chunks = [
        {"choices": [{"delta": {"content": "a"}, "finish_reason": "stop"}]},
    ]
    events = list(wrap_streaming_chunks(chunks, response_id="resp_ip", model="test"))
    payloads = _payloads(events)
    types = [p["type"] for p in payloads]
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert "response.completed" in types


def test_streaming_tool_loop_arguments_delta_has_item_id():
    registry = ToolRegistry()
    _register_tool(registry, "search", output="results")
    loop = ResponsesToolLoop(registry=registry)

    def stream_fn(payload):
        yield {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                    "function": {"name": "search", "arguments": '{"q": "'}}],
                },
                "finish_reason": None,
            }],
        }
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                    "function": {"arguments": "test}"}}],
                },
                "finish_reason": "tool_calls",
            }],
        }
        yield "[DONE]"

    def stream_turn_2(payload):
        yield {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}

    adapter = MagicMock()
    adapter.stream.side_effect = [stream_fn({}), stream_turn_2({})]
    encoder = ResponseStreamEncoder()

    events = list(loop.run_streaming(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}}}],
        instructions="",
        response_id="resp_arg_done",
        model="test-model",
        input_value="Search",
        encoder=encoder,
    ))

    payloads = _payloads(events)
    delta_events = [p for p in payloads if p["type"] == "response.function_call.arguments.delta"]
    done_events = [p for p in payloads if p["type"] == "response.function_call.arguments.done"]

    assert len(delta_events) >= 1
    assert len(done_events) >= 1
    for d in delta_events:
        assert "item_id" in d, "arguments.delta missing item_id"
        assert "output_index" in d, "arguments.delta missing output_index"
    for d in done_events:
        assert "item_id" in d, "arguments.done missing item_id"
        assert "output_index" in d, "arguments.done missing output_index"
        assert "arguments" in d


def test_streaming_tool_loop_provider_error_emits_failed():
    loop = ResponsesToolLoop()
    adapter = MagicMock()
    from providers.base import ProviderError
    adapter.stream.side_effect = ProviderError("Ollama unavailable", status_code=503)
    encoder = ResponseStreamEncoder()

    events = list(loop.run_streaming(
        adapter=adapter,
        chat_payload={"model": "test"},
        tools=[],
        instructions="",
        response_id="resp_pe",
        model="test-model",
        input_value="Hi",
        encoder=encoder,
    ))

    payloads = _payloads(events)
    types = [p["type"] for p in payloads]
    assert "response.failed" in types
    assert "[DONE]" not in [p["type"] for p in payloads]


def _payloads(events: list[str]) -> list[dict[str, Any]]:
    payloads = []
    for event in events:
        payload = None
        event_type = ""
        for line in event.splitlines():
            if line.startswith("event: "):
                event_type = line.split(":", 1)[1].strip()
            if line.startswith("data: "):
                data = line.split(":", 1)[1].strip()
                if data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
        if payload is not None:
            if "type" not in payload and event_type:
                payload["type"] = event_type
            payloads.append(payload)
    return payloads
