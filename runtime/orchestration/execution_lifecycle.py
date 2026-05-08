from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionPhase(Enum):
    RECEIVE = "receive"
    RESOLVE_PROVIDER = "resolve_provider"
    ROUTE = "route"
    EXECUTE = "execute"
    STREAM = "stream"
    TOOL_CALL = "tool_call"
    COMPLETE = "complete"
    ERROR = "error"


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    TOOL_CALL = "tool_call"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str = ""
    phase: ExecutionPhase = ExecutionPhase.RECEIVE
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    provider: str = ""
    model: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.correlation_id:
            self.correlation_id = uuid.uuid4().hex

    def start(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.status = ExecutionStatus.RUNNING

    def complete(self) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.status = ExecutionStatus.COMPLETED
        self.phase = ExecutionPhase.COMPLETE

    def fail(self, error: str) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.status = ExecutionStatus.FAILED
        self.phase = ExecutionPhase.ERROR
        self.error = error

    def elapsed_ms(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds() * 1000


class ExecutionLifecycle:
    def __init__(self) -> None:
        self._hooks: dict[ExecutionPhase, list[callable]] = {}

    def register_hook(self, phase: ExecutionPhase, hook: callable) -> None:
        self._hooks.setdefault(phase, []).append(hook)

    async def run_hooks(self, phase: ExecutionPhase, ctx: ExecutionContext) -> None:
        for hook in self._hooks.get(phase, []):
            await hook(ctx) if hasattr(hook, "__call__") else hook(ctx)


lifecycle = ExecutionLifecycle()
