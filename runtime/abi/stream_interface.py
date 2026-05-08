from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.context.execution_context import ExecutionContext


class StreamingRuntimeInterface(RuntimeComponent):
    @abstractmethod
    async def stream(
        self,
        ctx: ExecutionContext,
        request: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        ...

    @abstractmethod
    async def drain(self, ctx: ExecutionContext) -> None:
        ...

    @abstractmethod
    async def interrupt(self, ctx: ExecutionContext) -> None:
        ...
