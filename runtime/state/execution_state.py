from __future__ import annotations

from enum import Enum
from typing import Any

from runtime.events.event import RuntimeEvent, event_from_type
from runtime.events.event_types import EventType


class RuntimeStatus(Enum):
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_TOOL = "waiting_tool"
    STREAMING = "streaming"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    TOOL_CALL = "tool_call"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


class SessionStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class ProviderStatus(Enum):
    IDLE = "idle"
    RESOLVING = "resolving"
    SELECTED = "selected"
    CALLING = "calling"
    FAILED = "failed"
    COOLDOWN = "cooldown"


_EXECUTION_TRANSITIONS: dict[RuntimeStatus, set[RuntimeStatus]] = {
    RuntimeStatus.CREATED: {RuntimeStatus.PLANNING, RuntimeStatus.CANCELLED, RuntimeStatus.FAILED},
    RuntimeStatus.PLANNING: {RuntimeStatus.EXECUTING, RuntimeStatus.CANCELLED, RuntimeStatus.FAILED},
    RuntimeStatus.EXECUTING: {RuntimeStatus.WAITING_TOOL, RuntimeStatus.STREAMING, RuntimeStatus.COMPLETED, RuntimeStatus.FAILED, RuntimeStatus.PAUSED, RuntimeStatus.CANCELLED},
    RuntimeStatus.WAITING_TOOL: {RuntimeStatus.EXECUTING, RuntimeStatus.CANCELLED, RuntimeStatus.FAILED},
    RuntimeStatus.STREAMING: {RuntimeStatus.COMPLETED, RuntimeStatus.FAILED, RuntimeStatus.PAUSED, RuntimeStatus.CANCELLED},
    RuntimeStatus.PAUSED: {RuntimeStatus.EXECUTING, RuntimeStatus.CANCELLED, RuntimeStatus.FAILED},
    RuntimeStatus.CANCELLED: set(),
    RuntimeStatus.FAILED: set(),
    RuntimeStatus.COMPLETED: set(),
}

_STREAM_TRANSITIONS: dict[StreamStatus, set[StreamStatus]] = {
    StreamStatus.ACTIVE: {StreamStatus.PAUSED, StreamStatus.DRAINING, StreamStatus.COMPLETED, StreamStatus.INTERRUPTED},
    StreamStatus.PAUSED: {StreamStatus.ACTIVE, StreamStatus.DRAINING, StreamStatus.COMPLETED, StreamStatus.INTERRUPTED},
    StreamStatus.DRAINING: {StreamStatus.COMPLETED, StreamStatus.INTERRUPTED},
    StreamStatus.COMPLETED: set(),
    StreamStatus.INTERRUPTED: set(),
}

_SESSION_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.ACTIVE: {SessionStatus.IDLE, SessionStatus.EXPIRED, SessionStatus.TERMINATED},
    SessionStatus.IDLE: {SessionStatus.ACTIVE, SessionStatus.EXPIRED, SessionStatus.TERMINATED},
    SessionStatus.EXPIRED: set(),
    SessionStatus.TERMINATED: set(),
}

_AGENT_TRANSITIONS: dict[AgentStatus, set[AgentStatus]] = {
    AgentStatus.IDLE: {AgentStatus.WORKING, AgentStatus.COMPLETED, AgentStatus.FAILED},
    AgentStatus.WORKING: {AgentStatus.WAITING, AgentStatus.COMPLETED, AgentStatus.FAILED},
    AgentStatus.WAITING: {AgentStatus.WORKING, AgentStatus.FAILED},
    AgentStatus.COMPLETED: set(),
    AgentStatus.FAILED: set(),
}

_PROVIDER_TRANSITIONS: dict[ProviderStatus, set[ProviderStatus]] = {
    ProviderStatus.IDLE: {ProviderStatus.RESOLVING, ProviderStatus.FAILED},
    ProviderStatus.RESOLVING: {ProviderStatus.SELECTED, ProviderStatus.FAILED, ProviderStatus.COOLDOWN},
    ProviderStatus.SELECTED: {ProviderStatus.CALLING, ProviderStatus.FAILED},
    ProviderStatus.CALLING: {ProviderStatus.IDLE, ProviderStatus.FAILED, ProviderStatus.COOLDOWN},
    ProviderStatus.FAILED: {ProviderStatus.IDLE, ProviderStatus.COOLDOWN},
    ProviderStatus.COOLDOWN: {ProviderStatus.IDLE, ProviderStatus.FAILED},
}


def validate_transition(
    current: RuntimeStatus | StreamStatus | SessionStatus | AgentStatus | ProviderStatus,
    target: RuntimeStatus | StreamStatus | SessionStatus | AgentStatus | ProviderStatus,
) -> bool:
    if isinstance(current, RuntimeStatus) and isinstance(target, RuntimeStatus):
        return target in _EXECUTION_TRANSITIONS.get(current, set())
    if isinstance(current, StreamStatus) and isinstance(target, StreamStatus):
        return target in _STREAM_TRANSITIONS.get(current, set())
    if isinstance(current, SessionStatus) and isinstance(target, SessionStatus):
        return target in _SESSION_TRANSITIONS.get(current, set())
    if isinstance(current, AgentStatus) and isinstance(target, AgentStatus):
        return target in _AGENT_TRANSITIONS.get(current, set())
    if isinstance(current, ProviderStatus) and isinstance(target, ProviderStatus):
        return target in _PROVIDER_TRANSITIONS.get(current, set())
    return False


def transition_event(
    execution_id: str,
    source_type: str,
    from_status: str,
    to_status: str,
) -> RuntimeEvent:
    return event_from_type(
        EventType.STATE_TRANSITION,
        execution_id=execution_id,
        source="state_machine",
        source_type=source_type,
        from_state=from_status,
        to_state=to_status,
    )
