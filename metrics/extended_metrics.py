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


@dataclass
class ReasoningMetric:
    model: str = ""
    total_requests: int = 0
    total_thinking_tokens: int = 0
    total_reasoning_steps: int = 0
    total_budget_used: int = 0

    @property
    def avg_thinking_tokens(self) -> float:
        return self.total_thinking_tokens / max(self.total_requests, 1)


@dataclass
class GPUMetric:
    gpu_name: str = ""
    vram_total_mb: float = 0.0
    model_load_count: int = 0
    model_unload_count: int = 0
    kv_cache_hit_count: int = 0
    kv_cache_miss_count: int = 0

    @property
    def kv_cache_hit_rate(self) -> float:
        total = self.kv_cache_hit_count + self.kv_cache_miss_count
        return self.kv_cache_hit_count / max(total, 1)


@dataclass
class SessionMetric:
    session_id: str = ""
    message_count: int = 0
    duration_ms: float = 0.0
    tool_call_count: int = 0
    active: bool = True


class ExtendedMetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tool_metrics: dict[str, ToolMetric] = {}
        self._agent_metrics: dict[str, AgentMetric] = defaultdict(AgentMetric)
        self._mcp_metrics: dict[str, MCPSessionMetric] = {}
        self._capability_metrics: dict[str, CapabilityRoutingMetric] = {}
        self._reasoning_metrics: dict[str, ReasoningMetric] = {}
        self._gpu_metrics: dict[str, GPUMetric] = {}
        self._session_metrics: dict[str, SessionMetric] = {}
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

    def record_reasoning(
        self,
        model: str,
        thinking_tokens: int = 0,
        reasoning_steps: int = 0,
        budget_used: int = 0,
    ) -> None:
        with self._lock:
            metric = self._reasoning_metrics.setdefault(model, ReasoningMetric(model=model))
            metric.total_requests += 1
            metric.total_thinking_tokens += thinking_tokens
            metric.total_reasoning_steps += reasoning_steps
            metric.total_budget_used += budget_used

    def record_gpu_load(self, gpu_name: str, vram_total_mb: float) -> None:
        with self._lock:
            metric = self._gpu_metrics.setdefault(gpu_name, GPUMetric(gpu_name=gpu_name, vram_total_mb=vram_total_mb))
            metric.model_load_count += 1

    def record_gpu_unload(self, gpu_name: str) -> None:
        with self._lock:
            metric = self._gpu_metrics.setdefault(gpu_name, GPUMetric(gpu_name=gpu_name))
            metric.model_unload_count += 1

    def record_kv_cache(self, gpu_name: str, hit: bool) -> None:
        with self._lock:
            metric = self._gpu_metrics.setdefault(gpu_name, GPUMetric(gpu_name=gpu_name))
            if hit:
                metric.kv_cache_hit_count += 1
            else:
                metric.kv_cache_miss_count += 1

    def record_session_message(self, session_id: str, duration_ms: float = 0.0) -> None:
        with self._lock:
            metric = self._session_metrics.setdefault(
                session_id, SessionMetric(session_id=session_id)
            )
            metric.message_count += 1
            metric.duration_ms = duration_ms

    def record_session_tool_call(self, session_id: str) -> None:
        with self._lock:
            metric = self._session_metrics.setdefault(
                session_id, SessionMetric(session_id=session_id)
            )
            metric.tool_call_count += 1

    def close_session(self, session_id: str) -> None:
        with self._lock:
            metric = self._session_metrics.get(session_id)
            if metric:
                metric.active = False

    def get_reasoning_metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "model": m.model,
                    "total_requests": m.total_requests,
                    "total_thinking_tokens": m.total_thinking_tokens,
                    "avg_thinking_tokens": round(m.avg_thinking_tokens, 1),
                    "total_reasoning_steps": m.total_reasoning_steps,
                    "total_budget_used": m.total_budget_used,
                }
                for m in sorted(
                    self._reasoning_metrics.values(),
                    key=lambda x: x.total_requests,
                    reverse=True,
                )
            ]

    def get_gpu_metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "gpu_name": m.gpu_name,
                    "vram_total_mb": m.vram_total_mb,
                    "model_load_count": m.model_load_count,
                    "model_unload_count": m.model_unload_count,
                    "kv_cache_hit_rate": round(m.kv_cache_hit_rate, 3),
                    "kv_cache_hit_count": m.kv_cache_hit_count,
                    "kv_cache_miss_count": m.kv_cache_miss_count,
                }
                for m in self._gpu_metrics.values()
            ]

    def get_session_metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": m.session_id[:12] + "...",
                    "message_count": m.message_count,
                    "tool_call_count": m.tool_call_count,
                    "duration_ms": round(m.duration_ms, 1),
                    "active": m.active,
                }
                for m in sorted(
                    self._session_metrics.values(),
                    key=lambda x: x.message_count,
                    reverse=True,
                )
            ]

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tool_metrics": self.get_tool_metrics(),
                "agent_metrics": self.get_agent_metrics(),
                "mcp_metrics": self.get_mcp_metrics(),
                "capability_metrics": self.get_capability_metrics(),
                "reasoning_metrics": self.get_reasoning_metrics(),
                "gpu_metrics": self.get_gpu_metrics(),
                "session_metrics": self.get_session_metrics(),
                "active_mcp_sessions": self._session_count,
                "active_sessions": sum(
                    1 for s in self._session_metrics.values() if s.active
                ),
            }


extended_metrics = ExtendedMetricsCollector()
