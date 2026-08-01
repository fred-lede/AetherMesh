from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from runtime.tools.tool_registry import ToolRegistry, ToolDescriptor
from runtime.tools.tool_result import ToolCall, ToolResult


def _make_completion(
    content: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    message: dict = {"content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": "test-model",
        "choices": [{"message": message, "finish_reason": "stop" if not tool_calls else "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()

    def _get_weather(call: ToolCall) -> ToolResult:
        return ToolResult(call=call, output=json.dumps({"location": "Taipei", "temp": "25C", "condition": "sunny"}))

    def _calculate(call: ToolCall) -> ToolResult:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        a = args.get("a", 0)
        b = args.get("b", 0)
        return ToolResult(call=call, output=str(a + b))

    r.register(ToolDescriptor(
        name="get_weather",
        description="Get weather for a location",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        handler=_get_weather,
    ))
    r.register(ToolDescriptor(
        name="calculate",
        description="Add two numbers",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        handler=_calculate,
    ))
    return r


@pytest.fixture
def mock_adapter() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(mock_adapter: MagicMock, registry: ToolRegistry) -> TestClient:
    with patch("runtime.orchestration.openai_handler.routing_engine"), \
         patch("runtime.orchestration.openai_handler.provider_for_model", return_value="ollama"), \
         patch("runtime.orchestration.openai_handler.capabilities_for_model", return_value=["chat", "tools"]), \
         patch("runtime.orchestration.openai_handler.find_registry_model", return_value=None), \
         patch("runtime.orchestration.provider_router.find_registry_model", return_value=None), \
         patch("runtime.orchestration.openai_handler.settings") as mock_settings, \
         patch("runtime.orchestration.provider_router.adapter", return_value=mock_adapter), \
         patch("runtime.responses.tool_loop.default_registry", registry), \
         patch("runtime.responses.tool_loop.default_executor") as mock_exec_cls:

        mock_settings.resolve_model_alias.side_effect = lambda m: m
        mock_settings.model_registry.return_value = {}
        mock_settings.queue_timeout_s = 5
        mock_settings.debug_responses = False
        mock_settings.rate_limit_enabled = False

        from runtime.tools.tool_executor import ToolExecutor
        mock_exec_cls.return_value = ToolExecutor(registry=registry)

        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("ollama", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_normalize_payload_for_provider", side_effect=lambda p, pr, w=None: p), \
             patch.object(svc, "_apply_generation_defaults", side_effect=lambda p: p), \
             patch.object(svc, "_is_async_requested", return_value=False), \
             patch.object(svc, "_record_metrics"), \
             patch.object(svc, "_resolve_max_turns", return_value=16):

            from router.openai.responses_adapter import create_responses_router
            app = FastAPI()
            app.include_router(create_responses_router(svc))
            yield TestClient(app)


class TestResponsesE2EToolLoop:
    def test_with_tools_returns_function_call_to_client(self, client: TestClient, mock_adapter: MagicMock) -> None:
        mock_adapter.chat.side_effect = None
        mock_adapter.chat.return_value = _make_completion(
            content="",
            tool_calls=[{
                "id": "call_weather_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Taipei"}'},
            }],
        )

        payload = {
            "model": "test-model",
            "input": "What's the weather?",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }],
        }
        resp = client.post("/v1/responses", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "completed"

    def test_no_tools_direct_completion(self, client: TestClient, mock_adapter: MagicMock) -> None:
        mock_adapter.chat.side_effect = None
        mock_adapter.chat.return_value = _make_completion(content="Hello, I am a helpful assistant.")

        payload = {"model": "test-model", "input": "Hello"}
        resp = client.post("/v1/responses", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "completed"

    def test_follow_up_with_function_call_output(self, client: TestClient, mock_adapter: MagicMock) -> None:
        mock_adapter.chat.side_effect = None
        mock_adapter.chat.return_value = _make_completion(
            content="The weather in Taipei is sunny and 25 degrees Celsius."
        )

        payload = {
            "model": "test-model",
            "input": [
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {"type": "function_call", "call_id": "call_w1", "name": "get_weather", "arguments": '{"location":"Taipei"}'},
                {"type": "function_call_output", "call_id": "call_w1", "output": '{"temp":"25C","condition":"sunny"}'},
            ],
        }
        resp = client.post("/v1/responses", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "completed"
        output_text = body.get("output_text", "")
        if not output_text:
            for item in body.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        output_text += part.get("text", "")
        assert "sunny" in output_text.lower()


class TestResponsesToolLoopDirectExecution:
    def test_run_single_tool_execution(self, mock_adapter: MagicMock, registry: ToolRegistry) -> None:
        from runtime.responses.tool_loop import ResponsesToolLoop
        from runtime.responses.response_models import ResponseStatus

        call_count = [0]
        responses = [
            _make_completion(
                tool_calls=[{
                    "id": "call_w1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Taipei"}'},
                }],
            ),
            _make_completion(content="The weather in Taipei is sunny, 25C."),
        ]

        def chat(payload):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        mock_adapter.chat.side_effect = chat

        loop = ResponsesToolLoop(registry=registry)
        response = loop.run(
            adapter=mock_adapter,
            chat_payload={"model": "test-model"},
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }],
            instructions="",
            response_id="resp_e2e_1",
            model="test-model",
            input_value="What's the weather in Taipei?",
        )

        assert response.status == ResponseStatus.COMPLETED
        assert mock_adapter.chat.call_count == 2
        assert "sunny" in response.to_dict().get("output_text", "").lower()

    def test_run_multi_turn_tool_execution(self, mock_adapter: MagicMock, registry: ToolRegistry) -> None:
        from runtime.responses.tool_loop import ResponsesToolLoop
        from runtime.responses.response_models import ResponseStatus

        call_count = [0]
        responses = [
            _make_completion(
                tool_calls=[{
                    "id": "call_w1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
                }],
            ),
            _make_completion(
                tool_calls=[{
                    "id": "call_c1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"a": 3, "b": 7}'},
                }],
            ),
            _make_completion(content="In Tokyo it's sunny, and 3+7=10."),
        ]

        def chat(payload):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        mock_adapter.chat.side_effect = chat

        loop = ResponsesToolLoop(registry=registry)
        response = loop.run(
            adapter=mock_adapter,
            chat_payload={"model": "test-model"},
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Add numbers",
                        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
                    },
                },
            ],
            instructions="",
            response_id="resp_e2e_2",
            model="test-model",
            input_value="Check Tokyo weather and calculate 3+7",
        )

        assert response.status == ResponseStatus.COMPLETED
        assert mock_adapter.chat.call_count == 3

    def test_run_unknown_tool_returns_error(self, mock_adapter: MagicMock, registry: ToolRegistry) -> None:
        from runtime.responses.tool_loop import ResponsesToolLoop
        from runtime.responses.response_models import ResponseStatus

        call_count = [0]
        responses = [
            _make_completion(
                tool_calls=[{
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "nonexistent_tool", "arguments": "{}"},
                }],
            ),
            _make_completion(content="The tool was not found, but I can still help."),
        ]

        def chat(payload):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        mock_adapter.chat.side_effect = chat

        loop = ResponsesToolLoop(registry=registry)
        response = loop.run(
            adapter=mock_adapter,
            chat_payload={"model": "test-model"},
            tools=[],
            instructions="",
            response_id="resp_e2e_3",
            model="test-model",
            input_value="Use a missing tool",
        )

        assert response.status == ResponseStatus.COMPLETED


class TestResponsesOpenAIProviderToolNormalization:
    """OpenAI-compatible passthrough must run _ensure_openai_tools so every
    function tool carries a valid `parameters` field (strict upstreams reject
    `tools[i].function: missing field parameters`)."""

    def test_streaming_openai_normalizes_tools(self) -> None:
        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        mock_adapter = MagicMock()
        mock_adapter.stream.return_value = []

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("openai", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_record_metrics"), \
             patch.object(svc, "_finalize_request"):
            list(svc.handle_streaming_responses({
                "model": "gpt-4o",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "shell", "description": "run shell", "inputSchema": {}},
                    {"type": "namespace", "name": "codegraph", "description": "cg", "tools": [
                        {"type": "function", "name": "explore", "description": "x", "inputSchema": {"type": "object"}},
                    ]},
                ],
            }))

        sent = mock_adapter.stream.call_args[0][0]
        assert sent["stream"] is True
        assert sent["messages"] == [{"role": "user", "content": "hi"}]
        assert "input" not in sent
        tools = sent["tools"]
        assert len(tools) == 2
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "shell"
        assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}, "additionalProperties": False}
        assert tools[1]["type"] == "function"
        assert tools[1]["function"]["name"] == "codegraph.explore"
        assert tools[1]["function"]["parameters"] == {"type": "object"}

    def test_non_streaming_openai_normalizes_tools(self) -> None:
        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        mock_adapter = MagicMock()
        mock_adapter.responses.return_value = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [],
            "usage": {},
        }

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("openai", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_record_metrics"):
            svc.handle_responses({
                "model": "gpt-4o",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "shell", "description": "run shell", "inputSchema": {}},
                    {"type": "namespace", "name": "codegraph", "description": "cg", "tools": [
                        {"type": "function", "name": "explore", "description": "x", "inputSchema": {"type": "object"}},
                    ]},
                ],
            })

        sent = mock_adapter.responses.call_args[0][0]
        assert "input" in sent
        tools = sent["tools"]
        assert len(tools) == 2
        assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}, "additionalProperties": False}
        assert tools[1]["function"]["name"] == "codegraph.explore"
        assert tools[1]["function"]["parameters"] == {"type": "object"}

    def test_streaming_openai_drops_tool_choice_without_tools(self) -> None:
        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        mock_adapter = MagicMock()
        mock_adapter.stream.return_value = []

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("openai", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_record_metrics"), \
             patch.object(svc, "_finalize_request"):
            list(svc.handle_streaming_responses({
                "model": "gpt-4o",
                "input": "hi",
                "tool_choice": {"type": "auto"},
            }))

        sent = mock_adapter.stream.call_args[0][0]
        assert "tool_choice" not in sent

    def test_streaming_openai_guarantees_parameters_for_plugin_subtools(self) -> None:
        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        mock_adapter = MagicMock()
        mock_adapter.stream.return_value = []

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("openai", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_record_metrics"), \
             patch.object(svc, "_finalize_request"):
            list(svc.handle_streaming_responses({
                "model": "gpt-4o",
                "input": "hi",
                "tools": [
                    {"type": "plugin", "name": "codex", "tools": [
                        {"type": "function", "function": {"name": "read", "description": "read file"}},
                    ]},
                ],
            }))

        sent = mock_adapter.stream.call_args[0][0]
        tools = sent["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "read"
        assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}, "additionalProperties": False}

    def test_streaming_openai_drops_unsupported_tool_types(self) -> None:
        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        mock_adapter = MagicMock()
        mock_adapter.stream.return_value = []

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("openai", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_record_metrics"), \
             patch.object(svc, "_finalize_request"):
            list(svc.handle_streaming_responses({
                "model": "gpt-4o",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "shell", "description": "run shell", "inputSchema": {}},
                    {"type": "web_search", "name": ""},
                ],
            }))

        sent = mock_adapter.stream.call_args[0][0]
        tools = sent["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "shell"
        assert "web_search" not in {t.get("type") for t in sent["tools"]}

    def test_non_streaming_openai_drops_unsupported_tool_types(self) -> None:
        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        mock_adapter = MagicMock()
        mock_adapter.responses.return_value = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [],
            "usage": {},
        }

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("openai", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_record_metrics"):
            svc.handle_responses({
                "model": "gpt-4o",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "shell", "description": "run shell", "inputSchema": {}},
                    {"type": "web_search", "name": ""},
                ],
            })

        sent = mock_adapter.responses.call_args[0][0]
        tools = sent["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert "web_search" not in {t.get("type") for t in sent["tools"]}

    def test_non_streaming_openai_drops_tool_choice_without_tools(self) -> None:
        from runtime.orchestration.openai_handler import RouterService
        svc = RouterService()
        svc.registry = {"models": []}

        mock_adapter = MagicMock()
        mock_adapter.responses.return_value = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [],
            "usage": {},
        }

        with patch.object(svc, "_resolve_provider_and_worker", return_value=("openai", None)), \
             patch.object(svc, "_adapter", return_value=mock_adapter), \
             patch.object(svc, "_record_metrics"):
            svc.handle_responses({
                "model": "gpt-4o",
                "input": "hi",
                "tool_choice": {"type": "auto"},
            })

        sent = mock_adapter.responses.call_args[0][0]
        assert "tool_choice" not in sent
