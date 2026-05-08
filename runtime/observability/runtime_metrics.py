from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from runtime.context.execution_context import ExecutionContext
from runtime.observability.metrics import metrics_collector


@dataclass
class RuntimeMetricsSnapshot:
    execution_count: int = 0
    active_executions: int = 0
    completed_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    avg_execution_ms: float = 0.0
    total_tool_calls: int = 0
    total_provider_calls: int = 0
    total_gpu_allocations: int = 0
    total_events_emitted: int = 0


class RuntimeMetrics:
    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._execution_times: list[float] = []

    def record_execution_completed(self, ctx: ExecutionContext) -> None:
        elapsed = ctx.elapsed_ms()
        self._execution_times.append(elapsed)
        metrics_collector.record("runtime.execution.duration_ms", elapsed)
        metrics_collector.increment("runtime.execution.completed")

    def record_execution_failed(self, ctx: ExecutionContext) -> None:
        metrics_collector.increment("runtime.execution.failed")

    def record_execution_cancelled(self, ctx: ExecutionContext) -> None:
        metrics_collector.increment("runtime.execution.cancelled")

    def record_tool_call(self, tool_name: str, duration_ms: float) -> None:
        metrics_collector.increment("runtime.tool.calls")
        metrics_collector.record(f"runtime.tool.{tool_name}.duration_ms", duration_ms)

    def record_provider_call(self, provider: str, duration_ms: float) -> None:
        metrics_collector.increment("runtime.provider.calls")
        metrics_collector.record(f"runtime.provider.{provider}.duration_ms", duration_ms)

    def record_gpu_allocation(self, device: str, vram_mb: int) -> None:
        metrics_collector.increment("runtime.gpu.allocations")
        metrics_collector.set_gauge(f"runtime.gpu.{device}.vram_mb", vram_mb)

    def snapshot(self) -> RuntimeMetricsSnapshot:
        completed = metrics_collector.get_counter("runtime.execution.completed")
        failed = metrics_collector.get_counter("runtime.execution.failed")
        cancelled = metrics_collector.get_counter("runtime.execution.cancelled")
        durations = metrics_collector.get_histogram("runtime.execution.duration_ms") or [0]
        return RuntimeMetricsSnapshot(
            execution_count=completed + failed + cancelled,
            active_executions=completed + failed + cancelled,
            completed_executions=completed,
            failed_executions=failed,
            cancelled_executions=cancelled,
            avg_execution_ms=sum(durations) / len(durations) if durations else 0.0,
            total_tool_calls=metrics_collector.get_counter("runtime.tool.calls"),
            total_provider_calls=metrics_collector.get_counter("runtime.provider.calls"),
            total_gpu_allocations=metrics_collector.get_counter("runtime.gpu.allocations"),
            total_events_emitted=metrics_collector.get_counter("runtime.events.emitted"),
        )

    def uptime_s(self) -> float:
        return time.time() - self._start_time


runtime_metrics = RuntimeMetrics()
