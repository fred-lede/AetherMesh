from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("metrics.extended")


@dataclass
class ToolMetric:
    name: str
    call_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    source: str = "builtin"

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / max(self.call_count, 1)

    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.call_count, 1)


@dataclass
class AgentMetric:
    agent_type: str = "single"
    run_count: int = 0
    total_steps: int = 0
    total_duration_ms: float = 0.0
    tool_calls_per_run: float = 0.0

    @property
    def avg_steps(self) -> float:
        return self.total_steps / max(self.run_count, 1)

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / max(self.run_count, 1)


@dataclass
class MCPSessionMetric:
    server_name: str = ""
    tool_call_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    session_count: int = 0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / max(self.tool_call_count, 1)


@dataclass
class CapabilityRoutingMetric:
    capability: str = ""
    match_count: int = 0
    fallback_count: int = 0
    cooldown_count: int = 0


class ExtendedMetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tool_metrics: dict[str, ToolMetric] = {}
        self._agent_metrics: dict[str, AgentMetric] = defaultdict(AgentMetric)
        self._mcp_metrics: dict[str, MCPSessionMetric] = {}
        self._capability_metrics: dict[str, CapabilityRoutingMetric] = {}
        self._session_count: int = 0

    def record_tool_call(self, name: str, duration_ms: float, is_error: bool = False, source: str = "builtin") -> None:
        with self._lock:
            metric = self._tool_metrics.setdefault(name, ToolMetric(name=name, source=source))
            metric.call_count += 1
            metric.total_duration_ms += duration_ms
            if is_error:
                metric.error_count += 1

    def record_agent_run(self, agent_type: str, step_count: int, duration_ms: float, tool_calls: int) -> None:
        with self._lock:
            metric = self._agent_metrics[agent_type]
            metric.agent_type = agent_type
            metric.run_count += 1
            metric.total_steps += step_count
            metric.total_duration_ms += duration_ms
            metric.tool_calls_per_run = (metric.tool_calls_per_run * (metric.run_count - 1) + tool_calls) / metric.run_count

    def record_mcp_call(self, server_name: str, duration_ms: float, is_error: bool = False) -> None:
        with self._lock:
            metric = self._mcp_metrics.setdefault(server_name, MCPSessionMetric(server_name=server_name))
            metric.tool_call_count += 1
            metric.total_duration_ms += duration_ms
            if is_error:
                metric.error_count += 1

    def record_mcp_session(self, server_name: str) -> None:
        with self._lock:
            metric = self._mcp_metrics.setdefault(server_name, MCPSessionMetric(server_name=server_name))
            metric.session_count += 1
            self._session_count += 1

    def record_capability_routing(self, capability: str, matched: bool = True, fell_back: bool = False, cooled_down: bool = False) -> None:
        with self._lock:
            metric = self._capability_metrics.setdefault(capability, CapabilityRoutingMetric(capability=capability))
            metric.match_count += 1 if matched else 0
            metric.fallback_count += 1 if fell_back else 0
            metric.cooldown_count += 1 if cooled_down else 0

    def get_tool_metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": m.name,
                    "call_count": m.call_count,
                    "error_count": m.error_count,
                    "avg_duration_ms": round(m.avg_duration_ms, 1),
                    "error_rate": round(m.error_rate, 3),
                    "source": m.source,
                }
                for m in sorted(self._tool_metrics.values(), key=lambda x: x.call_count, reverse=True)
            ]

    def get_agent_metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "agent_type": m.agent_type,
                    "run_count": m.run_count,
                    "avg_steps": round(m.avg_steps, 1),
                    "avg_duration_ms": round(m.avg_duration_ms, 1),
                    "tool_calls_per_run": round(m.tool_calls_per_run, 1),
                }
                for m in self._agent_metrics.values()
            ]

    def get_mcp_metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "server_name": m.server_name,
                    "tool_call_count": m.tool_call_count,
                    "error_count": m.error_count,
                    "avg_duration_ms": round(m.avg_duration_ms, 1),
                    "session_count": m.session_count,
                }
                for m in sorted(self._mcp_metrics.values(), key=lambda x: x.tool_call_count, reverse=True)
            ]

    def get_capability_metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "capability": m.capability,
                    "match_count": m.match_count,
                    "fallback_count": m.fallback_count,
                    "cooldown_count": m.cooldown_count,
                    "fallback_rate": round(m.fallback_count / max(m.match_count, 1), 3),
                }
                for m in self._capability_metrics.values()
            ]

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tool_metrics": self.get_tool_metrics(),
                "agent_metrics": self.get_agent_metrics(),
                "mcp_metrics": self.get_mcp_metrics(),
                "capability_metrics": self.get_capability_metrics(),
                "active_mcp_sessions": self._session_count,
            }


extended_metrics = ExtendedMetricsCollector()
