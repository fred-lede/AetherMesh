from __future__ import annotations

import logging
from typing import Any

from runtime.tools.tool_executor import ToolExecutor, tool_executor as default_executor
from runtime.tools.tool_normalizer import ToolCallNormalizer
from runtime.tools.tool_registry import ToolDescriptor, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("tool_runtime")


class ToolRuntime:
    def __init__(
        self,
        registry=None,
        executor: ToolExecutor | None = None,
        normalizer: ToolCallNormalizer | None = None,
    ) -> None:
        from runtime.tools import tool_registry as tr
        from runtime.tools.tool_normalizer import ToolCallNormalizer as TCN

        self._registry = registry or tr.tool_registry
        self._executor = executor or default_executor
        self._normalizer = normalizer or TCN()

    @property
    def registry(self):
        return self._registry

    def normalize(self, content: str | dict | list, source_provider: str = "", source_model: str = "") -> list[ToolCall]:
        raw_calls = self._normalizer.normalize(content)
        return [
            ToolCall(
                id=rc.id,
                name=rc.name,
                arguments=rc.input,
                source_provider=source_provider,
                source_model=source_model,
            )
            for rc in raw_calls
        ]

    def execute(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        return self._executor.execute(call, timeout_s=timeout_s)

    async def execute_async(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        return await self._executor.execute_async(call, timeout_s=timeout_s)

    def execute_all(self, calls: list[ToolCall], timeout_s: int = 30) -> list[ToolResult]:
        return [self.execute(c, timeout_s=timeout_s) for c in calls]

    def register_tool(self, descriptor: ToolDescriptor) -> None:
        self._registry.register(descriptor)


tool_runtime = ToolRuntime()
