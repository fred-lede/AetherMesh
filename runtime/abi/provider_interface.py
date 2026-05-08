from __future__ import annotations

from abc import abstractmethod
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.context.execution_context import ExecutionContext


class ProviderRuntimeInterface(RuntimeComponent):
    @abstractmethod
    async def resolve(
        self,
        ctx: ExecutionContext,
        capabilities: list[str],
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def execute(
        self,
        ctx: ExecutionContext,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def score_candidates(
        self,
        ctx: ExecutionContext,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...
