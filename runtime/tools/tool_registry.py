from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("tool_registry")

ToolHandler = Callable[[ToolCall], ToolResult | Awaitable[ToolResult]]


@dataclass
class ToolDescriptor:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler | None = None
    source: str = "builtin"
    requires_confirmation: bool = False
    timeout_s: int = 30


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}

    def register(self, descriptor: ToolDescriptor) -> None:
        self._tools[descriptor.name] = descriptor
        logger.debug("Registered tool: %s (source: %s)", descriptor.name, descriptor.source)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def resolve(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDescriptor]:
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    def clear(self) -> None:
        self._tools.clear()


tool_registry = ToolRegistry()
