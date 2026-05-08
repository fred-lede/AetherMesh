from __future__ import annotations

from abc import abstractmethod
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.context.execution_context import ExecutionContext


class GPURuntimeInterface(RuntimeComponent):
    @abstractmethod
    async def allocate(
        self,
        ctx: ExecutionContext,
        model: str,
        vram_mb: int,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def release(
        self,
        ctx: ExecutionContext,
        allocation_id: str,
    ) -> None:
        ...

    @abstractmethod
    async def get_status(
        self,
        ctx: ExecutionContext,
    ) -> list[dict[str, Any]]:
        ...
