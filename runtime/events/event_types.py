from __future__ import annotations

from enum import Enum


class EventType(Enum):
    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"
    PROVIDER_SELECTED = "provider_selected"
    PROVIDER_FAILED = "provider_failed"
    GPU_ALLOCATED = "gpu_allocated"
    GPU_RELEASED = "gpu_released"
    GRAPH_NODE_STARTED = "graph_node_started"
    GRAPH_NODE_COMPLETED = "graph_node_completed"
    SESSION_CREATED = "session_created"
    MEMORY_RETRIEVED = "memory_retrieved"
    MEMORY_STORED = "memory_stored"
    STREAM_STARTED = "stream_started"
    STREAM_COMPLETED = "stream_completed"
    EXECUTION_CREATED = "execution_created"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_PAUSED = "execution_paused"
    EXECUTION_CANCELLED = "execution_cancelled"
    AGENT_DELEGATED = "agent_delegated"
    AGENT_COMPLETED = "agent_completed"
    GRAPH_STARTED = "graph_started"
    GRAPH_COMPLETED = "graph_completed"
    STATE_TRANSITION = "state_transition"
    RATE_LIMITED = "rate_limited"
    VALIDATION_FAILED = "validation_failed"
