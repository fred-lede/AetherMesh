from __future__ import annotations

from abc import abstractmethod
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.context.execution_context import ExecutionContext


class ToolRuntimeInterface(RuntimeComponent):
    @abstractmethod
    async def normalize(
        self,
        ctx: ExecutionContext,
        raw_calls: Any,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def execute(
        self,
        ctx: ExecutionContext,
        tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def execute_all(
        self,
        ctx: ExecutionContext,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...
