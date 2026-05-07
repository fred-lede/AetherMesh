from __future__ import annotations

from router.anthropic_router import AnthropicRouter
from router.server_tool_policy import (
    evaluate_server_tool_policy,
    forced_server_tool,
    listed_server_tools,
    server_tool_name,
)


def test_detects_anthropic_server_tools_by_name_and_type() -> None:
    assert server_tool_name({"name": "web_search", "type": "web_search_20250305"}) == "web_search"
    assert server_tool_name({"name": "web_fetch", "type": "web_fetch_20250305"}) == "web_fetch"
    assert server_tool_name({"name": "Bash", "type": "function"}) is None


def test_lists_and_forces_server_tools() -> None:
    payload = {
        "tools": [
            {"name": "web_search", "type": "web_search_20250305"},
            {"name": "Bash", "type": "function"},
        ],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }

    assert listed_server_tools(payload) == {"web_search"}
    assert forced_server_tool(payload) == "web_search"


def test_openai_compatible_provider_rejects_listed_server_tools() -> None:
    result = evaluate_server_tool_policy(
        {"tools": [{"name": "web_search", "type": "web_search_20250305"}]},
        provider="nvidia_nim",
    )

    assert result.listed_tools == {"web_search"}
    assert result.error is not None
    assert "cannot execute" in result.error


def test_passthrough_mode_allows_client_side_server_tools() -> None:
    result = evaluate_server_tool_policy(
        {"tools": [{"name": "web_search", "type": "web_search_20250305"}]},
        provider="nvidia_nim",
        mode="passthrough",
    )

    assert result.mode == "passthrough"
    assert result.error is None
    assert not result.should_handle_locally


def test_local_mode_handles_forced_server_tool() -> None:
    result = evaluate_server_tool_policy(
        {
            "tools": [{"name": "web_search", "type": "web_search_20250305"}],
            "tool_choice": {"type": "tool", "name": "web_search"},
        },
        provider="nvidia_nim",
        mode="local",
    )

    assert result.mode == "local"
    assert result.error is None
    assert result.should_handle_locally


def test_openai_tool_conversion_drops_server_tools() -> None:
    router = AnthropicRouter()
    converted = router._anthropic_tools_to_openai(
        [
            {"name": "web_search", "type": "web_search_20250305"},
            {
                "name": "Bash",
                "type": "function",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        ]
    )

    assert len(converted) == 1
    assert converted[0]["function"]["name"] == "Bash"
