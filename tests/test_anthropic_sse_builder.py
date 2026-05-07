from __future__ import annotations

import json

from router.anthropic_sse_builder import AnthropicSSEBuilder


def _payloads(events: list[str]) -> list[dict]:
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
    return payloads


def test_tool_arguments_before_name_are_buffered() -> None:
    sse = AnthropicSSEBuilder("claude-sonnet-4-5")
    events = [sse.message_start()]
    events.extend(
        sse.process_tool_call_delta(
            {
                "index": 0,
                "function": {"arguments": "{\"query\":"},
            }
        )
    )
    events.extend(
        sse.process_tool_call_delta(
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "WebSearch", "arguments": "\"Tesla\"}"},
            }
        )
    )
    events.extend(sse.close_all_blocks())
    events.append(sse.message_delta("tool_use", 1))
    events.append(sse.message_stop())

    rendered = "".join(events)
    assert '"name": "WebSearch"' in rendered
    assert '\\"query\\":' in rendered
    assert '\\"Tesla\\"' in rendered
    assert sum(1 for payload in _payloads(events) if payload["type"] == "content_block_stop") == 1


def test_tool_name_fragments_are_merged() -> None:
    sse = AnthropicSSEBuilder("claude-sonnet-4-5")
    events = []
    events.extend(sse.process_tool_call_delta({"index": 0, "function": {"name": "Web", "arguments": ""}}))
    events.extend(sse.process_tool_call_delta({"index": 0, "function": {"name": "WebSearch", "arguments": "{}"}}))

    rendered = "".join(events)
    assert '"name": "Web"' in rendered
    assert "WebSearch" not in rendered
    assert '"partial_json": "{}"' in rendered


def test_text_and_thinking_blocks_close_before_switching() -> None:
    sse = AnthropicSSEBuilder("claude-sonnet-4-5")
    events = []
    events.extend(sse.emit_text_delta("hello"))
    events.extend(sse.emit_thinking_delta("reason"))
    events.extend(sse.close_all_blocks())

    payloads = _payloads(events)
    names = [payload["type"] for payload in payloads]
    assert names == [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
