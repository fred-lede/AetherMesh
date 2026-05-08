from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from runtime.context.execution_context import ExecutionContext


class RuntimeComponent(ABC):
    @abstractmethod
    async def initialize(self, ctx: ExecutionContext) -> None:
        ...

    @abstractmethod
    async def start(self, ctx: ExecutionContext) -> None:
        ...

    @abstractmethod
    async def pause(self, ctx: ExecutionContext) -> None:
        ...

    @abstractmethod
    async def resume(self, ctx: ExecutionContext) -> None:
        ...

    @abstractmethod
    async def cancel(self, ctx: ExecutionContext) -> None:
        ...

    @abstractmethod
    async def shutdown(self, ctx: ExecutionContext) -> None:
        ...
