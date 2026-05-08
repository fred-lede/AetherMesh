from __future__ import annotations

from abc import abstractmethod
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.context.execution_context import ExecutionContext


class AgentRuntimeInterface(RuntimeComponent):
    @abstractmethod
    async def execute_task(
        self,
        ctx: ExecutionContext,
        task: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def delegate(
        self,
        ctx: ExecutionContext,
        agent_id: str,
        subtask: str,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def fan_out(
        self,
        ctx: ExecutionContext,
        agent_ids: list[str],
        task: str,
    ) -> list[dict[str, Any]]:
        ...
