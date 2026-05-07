from __future__ import annotations

import json

from router.anthropic_router import AnthropicRouter, _stream_anthropic
from router.tool_call_normalizer import ToolCallNormalizer


def _sse_payloads(events: list[str]) -> list[dict]:
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
    return payloads


def test_raw_python_tool_use_text_normalizes() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_text(
        "Claude responded: {'type': 'tooluse', 'id': 'call_1', "
        "'name': 'WebSearch', 'input': {'query': 'Tesla latest news'}}"
    )

    assert len(calls) == 1
    assert calls[0].name == "WebSearch"
    assert calls[0].input == {"query": "Tesla latest news"}


def test_qwen_function_tag_tool_call_normalizes() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_text(
        "<function=WebSearch><parameter=query>Tesla 2026 forecast</parameter></function>"
    )

    assert len(calls) == 1
    assert calls[0].name == "WebSearch"
    assert calls[0].input == {"query": "Tesla 2026 forecast"}


def test_glm_arg_tag_tool_call_normalizes() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_text(
        "WebSearch<arg_key>query</arg_key><arg_value>Tesla Q1 earnings</arg_value>"
    )

    assert len(calls) == 1
    assert calls[0].name == "WebSearch"
    assert calls[0].input == {"query": "Tesla Q1 earnings"}


def test_namespaced_tool_call_normalizes() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_text(
        '<minimax:tool_call><invoke name="WebSearch">'
        '<parameter name="query">Tesla news</parameter>'
        "</invoke></minimax:tool_call>"
    )

    assert len(calls) == 1
    assert calls[0].name == "WebSearch"
    assert calls[0].input == {"query": "Tesla news"}


def test_bracket_tool_call_normalizes() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_text('[Tool call: WebSearch({"query":"Tesla"})]')

    assert len(calls) == 1
    assert calls[0].name == "WebSearch"
    assert calls[0].input == {"query": "Tesla"}


def test_gemma_call_tool_call_normalizes() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_text('call:WebSearch{"query":"Tesla"}')

    assert len(calls) == 1
    assert calls[0].name == "WebSearch"
    assert calls[0].input == {"query": "Tesla"}


def test_tool_call_inside_thinking_is_not_executed_with_visible_text() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_text(
        "<think>[Tool call: WebSearch({\"query\":\"Tesla\"})]</think>\n"
        "I should search, but I can answer from context."
    )

    assert calls == []


def test_tool_call_inside_thinking_requires_declared_tool() -> None:
    normalizer = ToolCallNormalizer()

    calls = normalizer.from_content_with_thinking(
        "",
        '[Tool call: WebSearch({"query":"Tesla"})]',
        allowed_tool_names={"WebSearch"},
    )

    assert len(calls) == 1
    assert calls[0].name == "WebSearch"
    assert calls[0].input == {"query": "Tesla"}


def test_non_stream_blocks_undeclared_tool_payload() -> None:
    router = AnthropicRouter()
    response = router._to_anthropic_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "{'type':'tool_use','id':'call_1','name':'WebSearch','input':{'query':'Tesla'}}",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        "claude-sonnet-4-5",
        allowed_tool_names=set(),
    )

    assert response["stop_reason"] == "end_turn"
    assert response["content"] == []


def test_non_stream_forwards_declared_tool() -> None:
    router = AnthropicRouter()
    response = router._to_anthropic_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "WebSearch", "arguments": "{\"query\":\"Tesla\"}"},
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        "claude-sonnet-4-5",
        allowed_tool_names={"WebSearch"},
    )

    assert response["stop_reason"] == "tool_use"
    assert response["content"][0]["type"] == "tool_use"
    assert response["content"][0]["name"] == "WebSearch"
    assert response["content"][0]["input"] == {"query": "Tesla"}


def test_streaming_buffers_and_blocks_raw_tool_text() -> None:
    events = list(
        _stream_anthropic(
            iter(
                [
                    {"choices": [{"delta": {"content": "Claude responded: {'type': 'tool"}}]},
                    {"choices": [{"delta": {"content": "_use', 'name': 'WebSearch', 'input': {'query': 'Tesla'}}"}}]},
                    {"choices": [{"delta": {"content": "Searched the web"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}},
                ]
            ),
            "claude-sonnet-4-5",
            allowed_tool_names=set(),
        )
    )

    rendered = "\n".join(events)
    assert "Claude responded" not in rendered
    assert "Searched the web" not in rendered
    assert "{'type'" not in rendered
    assert "Tool `WebSearch`" not in rendered
    assert any(payload.get("type") == "message_stop" for payload in _sse_payloads(events))


def test_streaming_forwards_declared_native_tool_fragments() -> None:
    events = list(
        _stream_anthropic(
            iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "WebSearch", "arguments": ""},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": "{\"query\":\"Tesla\"}"},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 1}},
                ]
            ),
            "claude-sonnet-4-5",
            allowed_tool_names={"WebSearch"},
        )
    )

    rendered = "\n".join(events)
    assert '"name": "WebSearch"' in rendered
    assert '\\"query\\":\\"Tesla\\"' in rendered
    assert "tool_unavailable" not in rendered


def test_streaming_suppresses_undeclared_native_tool_fragments_without_none_message() -> None:
    events = list(
        _stream_anthropic(
            iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "WebSearch", "arguments": ""},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": "{\"query\":\"Tesla\"}"},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {"content": "Searched the web"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 1}},
                ]
            ),
            "claude-sonnet-4-5",
            allowed_tool_names=set(),
        )
    )

    rendered = "\n".join(events)
    assert "Tool None" not in rendered
    assert "WebSearch" not in rendered
    assert "Searched the web" not in rendered
    assert "input_json_delta" not in rendered
    assert any(payload.get("type") == "message_stop" for payload in _sse_payloads(events))
