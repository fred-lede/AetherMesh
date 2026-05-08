from __future__ import annotations

from abc import abstractmethod
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.context.execution_context import ExecutionContext


class MemoryRuntimeInterface(RuntimeComponent):
    @abstractmethod
    async def store(
        self,
        ctx: ExecutionContext,
        key: str,
        value: Any,
    ) -> None:
        ...

    @abstractmethod
    async def retrieve(
        self,
        ctx: ExecutionContext,
        key: str,
    ) -> Any | None:
        ...

    @abstractmethod
    async def search(
        self,
        ctx: ExecutionContext,
        query: str,
        limit: int = 10,
    ) -> list[Any]:
        ...
