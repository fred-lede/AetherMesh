from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable


STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def format_sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def map_stop_reason(openai_reason: str | None) -> str:
    return STOP_REASON_MAP.get(str(openai_reason or ""), "end_turn")


@dataclass
class ToolStreamState:
    block_index: int = -1
    tool_id: str = ""
    name: str = ""
    started: bool = False
    pre_start_args: str = ""


@dataclass
class ContentBlockState:
    next_index: int = 0
    current_block_type: str | None = None
    current_block_index: int = -1
    tool_states: dict[int, ToolStreamState] = field(default_factory=dict)

    def allocate_index(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index

    def ensure_tool_state(self, tool_index: int) -> ToolStreamState:
        if tool_index not in self.tool_states:
            self.tool_states[tool_index] = ToolStreamState()
        return self.tool_states[tool_index]

    def register_tool_id(self, tool_index: int, tool_id: Any) -> None:
        if tool_id:
            self.ensure_tool_state(tool_index).tool_id = str(tool_id)

    def register_tool_name(self, tool_index: int, name: Any) -> None:
        if name is None:
            return
        incoming = str(name)
        state = self.ensure_tool_state(tool_index)
        prev = state.name
        if not prev or incoming.startswith(prev):
            state.name = incoming
        elif not prev.startswith(incoming):
            state.name = prev + incoming


class AnthropicSSEBuilder:
    def __init__(self, model: str, *, message_id: str | None = None) -> None:
        self.model = model
        self.message_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
        self.blocks = ContentBlockState()
        self.emitted_tool_use = False

    def message_start(self) -> str:
        return format_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    def message_delta(self, stop_reason: str, output_tokens: int = 0) -> str:
        return format_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": int(output_tokens) if output_tokens else 0},
            },
        )

    def message_stop(self) -> str:
        return format_sse("message_stop", {"type": "message_stop"})

    def error(self, message: str) -> str:
        return format_sse("error", {"type": "error", "error": {"type": "api_error", "message": message}})

    def ensure_text_block(self) -> Iterable[str]:
        yield from self._ensure_block("text", {"type": "text", "text": ""})

    def emit_text_delta(self, text: str) -> Iterable[str]:
        yield from self.ensure_text_block()
        yield format_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.blocks.current_block_index,
                "delta": {"type": "text_delta", "text": text},
            },
        )

    def ensure_thinking_block(self) -> Iterable[str]:
        yield from self._ensure_block("thinking", {"type": "thinking", "thinking": "", "signature": ""})

    def emit_thinking_delta(self, thinking: str) -> Iterable[str]:
        yield from self.ensure_thinking_block()
        yield format_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.blocks.current_block_index,
                "delta": {"type": "thinking_delta", "thinking": thinking},
            },
        )

    def process_tool_call_delta(self, tool_call: dict[str, Any]) -> Iterable[str]:
        tool_index = int(tool_call.get("index", 0) or 0)
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        incoming_name = function.get("name")
        arguments = str(function.get("arguments") or "")

        self.blocks.register_tool_id(tool_index, tool_call.get("id"))
        self.blocks.register_tool_name(tool_index, incoming_name)
        state = self.blocks.ensure_tool_state(tool_index)

        if not state.started and state.name.strip():
            yield from self.close_current_block()
            state.block_index = self.blocks.allocate_index()
            state.tool_id = state.tool_id or str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
            state.started = True
            self.blocks.current_block_type = "tool_use"
            self.blocks.current_block_index = state.block_index
            self.emitted_tool_use = True
            yield format_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.block_index,
                    "content_block": {"type": "tool_use", "id": state.tool_id, "name": state.name, "input": {}},
                },
            )
            if state.pre_start_args:
                pre = state.pre_start_args
                state.pre_start_args = ""
                yield self.emit_tool_delta(tool_index, pre)

        if not arguments:
            return
        if not state.started:
            state.pre_start_args += arguments
            return
        yield self.emit_tool_delta(tool_index, arguments)

    def start_text_tool_call(self, *, tool_id: str, name: str, arguments: str) -> Iterable[str]:
        yield from self.close_current_block()
        block_index = self.blocks.allocate_index()
        self.blocks.current_block_type = "tool_use"
        self.blocks.current_block_index = block_index
        self.emitted_tool_use = True
        yield format_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            },
        )
        if arguments:
            yield format_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": arguments},
                },
            )
        yield format_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
        self.blocks.current_block_type = None
        self.blocks.current_block_index = -1

    def emit_tool_delta(self, tool_index: int, partial_json: str) -> str:
        state = self.blocks.tool_states[tool_index]
        return format_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": state.block_index,
                "delta": {"type": "input_json_delta", "partial_json": partial_json},
            },
        )

    def close_current_block(self) -> Iterable[str]:
        block_type = self.blocks.current_block_type
        block_index = self.blocks.current_block_index
        if block_type is None or block_index < 0:
            return
        yield format_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
        if block_type == "tool_use":
            for state in self.blocks.tool_states.values():
                if state.block_index == block_index:
                    state.started = False
        self.blocks.current_block_type = None
        self.blocks.current_block_index = -1

    def close_all_blocks(self) -> Iterable[str]:
        yield from self.close_current_block()
        for state in list(self.blocks.tool_states.values()):
            if state.started and state.block_index >= 0:
                yield format_sse("content_block_stop", {"type": "content_block_stop", "index": state.block_index})
                state.started = False

    def _ensure_block(self, block_type: str, content_block: dict[str, Any]) -> Iterable[str]:
        if self.blocks.current_block_type == block_type:
            return
        yield from self.close_current_block()
        index = self.blocks.allocate_index()
        self.blocks.current_block_type = block_type
        self.blocks.current_block_index = index
        yield format_sse(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": content_block},
        )
