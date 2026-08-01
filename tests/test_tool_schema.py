from __future__ import annotations

from typing import Any

from runtime.orchestration.anthropic_converter import AnthropicRouter
from runtime.orchestration.openai_handler import RouterService
from runtime.tools.tool_registry import (
    ToolDescriptor,
    ToolRegistry,
    ensure_parameters_schema,
)

EMPTY_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


# ── ensure_parameters_schema ────────────────────────────────────────


def test_missing_schema_gets_empty_object_default() -> None:
    assert ensure_parameters_schema(None) == EMPTY_SCHEMA
    assert ensure_parameters_schema({}) == EMPTY_SCHEMA


def test_invalid_schema_gets_empty_object_default() -> None:
    assert ensure_parameters_schema("not-a-dict") == EMPTY_SCHEMA
    assert ensure_parameters_schema({"properties": {}}) == EMPTY_SCHEMA


def test_valid_schema_preserved() -> None:
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    assert ensure_parameters_schema(schema) == schema


# ── ToolRegistry.get_openai_tools ───────────────────────────────────


def test_registry_tool_without_schema_gets_valid_parameters() -> None:
    registry = ToolRegistry()
    registry.register(ToolDescriptor(name="ping", description="Ping", input_schema={}))

    tools = registry.get_openai_tools()
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "ping"
    assert fn["description"] == "Ping"
    assert fn["parameters"] == EMPTY_SCHEMA


def test_registry_tool_with_schema_preserved() -> None:
    registry = ToolRegistry()
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    registry.register(ToolDescriptor(name="calc", description="Calc", input_schema=schema))

    tools = registry.get_openai_tools()
    assert tools[0]["function"]["parameters"] == schema


# ── RouterService._ensure_openai_tools ──────────────────────────────


def test_flat_function_tool_without_parameters_gets_default() -> None:
    tools = RouterService._ensure_openai_tools(
        [{"type": "function", "name": "example_tool", "description": "Example tool"}]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "example_tool",
                "description": "Example tool",
                "parameters": EMPTY_SCHEMA,
            },
        }
    ]


def test_nested_function_tool_without_parameters_gets_default() -> None:
    tools = RouterService._ensure_openai_tools(
        [{"type": "function", "function": {"name": "example_tool", "description": "Example tool"}}]
    )
    assert tools[0]["function"]["parameters"] == EMPTY_SCHEMA


def test_nested_function_tool_with_parameters_preserved() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    tools = RouterService._ensure_openai_tools(
        [{"type": "function", "function": {"name": "example_tool", "description": "Example tool", "parameters": schema}}]
    )
    assert tools[0]["function"]["parameters"] == schema


def test_plugin_tool_gets_default_schema() -> None:
    tools = RouterService._ensure_openai_tools(
        [{"type": "plugin", "name": "my_plugin", "description": "Plugin"}]
    )
    fn = tools[0]["function"]
    assert fn["name"] == "plugin_my_plugin"
    assert fn["parameters"] == EMPTY_SCHEMA


def test_non_function_tools_passed_through() -> None:
    tool = {"type": "web_search"}
    tools = RouterService._ensure_openai_tools([tool])
    assert tools == [tool]


def test_plugin_with_tools_unwraps_function_tools() -> None:
    inner = {
        "type": "function",
        "function": {"name": "sub", "description": "Sub", "parameters": EMPTY_SCHEMA},
    }
    tools = RouterService._ensure_openai_tools([{"type": "plugin", "tools": [inner]}])
    assert tools == [inner]


def test_namespace_tool_flattens_codex_flat_format() -> None:
    schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
    tools = RouterService._ensure_openai_tools(
        [
            {
                "type": "namespace",
                "name": "tickets",
                "description": "Ticket tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_ticket",
                        "description": "Look up a ticket",
                        "inputSchema": schema,
                        "deferLoading": True,
                    }
                ],
            }
        ]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "tickets.lookup_ticket",
                "description": "Look up a ticket",
                "parameters": schema,
            },
        }
    ]


def test_namespace_tool_flattens_nested_format() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    inner = {
        "type": "function",
        "function": {"name": "search", "description": "Search", "parameters": schema},
    }
    tools = RouterService._ensure_openai_tools(
        [{"type": "namespace", "name": "web", "description": "Web tools", "tools": [inner]}]
    )
    assert tools == [
        {
            "type": "function",
            "function": {"name": "web.search", "description": "Search", "parameters": schema},
        }
    ]


def test_namespace_sub_tool_without_schema_gets_default() -> None:
    tools = RouterService._ensure_openai_tools(
        [
            {
                "type": "namespace",
                "name": "misc",
                "tools": [{"type": "function", "name": "noop", "description": "No-op"}],
            }
        ]
    )
    assert tools[0]["function"]["name"] == "misc.noop"
    assert tools[0]["function"]["parameters"] == EMPTY_SCHEMA


def test_namespace_without_name_keeps_sub_tool_name() -> None:
    tools = RouterService._ensure_openai_tools(
        [
            {
                "type": "namespace",
                "tools": [{"type": "function", "name": "noop", "description": "No-op"}],
            }
        ]
    )
    assert tools[0]["function"]["name"] == "noop"


# ── AnthropicRouter._anthropic_tools_to_openai ──────────────────────


def test_anthropic_tool_without_input_schema_gets_default() -> None:
    router = AnthropicRouter()
    tools = router._anthropic_tools_to_openai(
        [{"type": "function", "name": "get_weather", "description": "Get weather"}]
    )
    fn = tools[0]["function"]
    assert fn["name"] == "get_weather"
    assert fn["parameters"] == EMPTY_SCHEMA


def test_anthropic_tool_with_input_schema_preserved() -> None:
    router = AnthropicRouter()
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    tools = router._anthropic_tools_to_openai(
        [{"type": "function", "name": "get_weather", "description": "Get weather", "input_schema": schema}]
    )
    assert tools[0]["function"]["parameters"] == schema


def test_anthropic_server_tool_skipped() -> None:
    router = AnthropicRouter()
    tools = router._anthropic_tools_to_openai(
        [{"type": "function", "name": "web_search", "description": "Search"}]
    )
    assert tools == []
