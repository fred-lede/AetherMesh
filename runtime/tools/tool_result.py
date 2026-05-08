from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    source_provider: str = ""
    source_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class ToolResult:
    call: ToolCall
    output: Any = ""
    is_error: bool = False
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        if self.is_error:
            return f"Error executing {self.call.name}: {self.output}"
        return str(self.output)

    def to_tool_result_block(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.call.id,
            "content": self.to_text(),
            "is_error": self.is_error,
        }
