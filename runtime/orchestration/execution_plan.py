from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.orchestration.graph import ExecutionGraph
from runtime.orchestration.graph_executor import GraphExecutionResult
from runtime.orchestration.planner import Planner


@dataclass
class ExecutionPlan:
    task: str
    graph: ExecutionGraph
    result: GraphExecutionResult | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: str, planner: Planner | None = None) -> ExecutionPlan:
        p = planner or Planner()
        graph = p.plan(task)
        return cls(task=task, graph=graph)

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.result.success

    @property
    def duration_ms(self) -> float:
        return self.result.elapsed_ms if self.result else 0.0

    @property
    def node_count(self) -> int:
        return len(self.graph.nodes)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task[:100],
            "nodes": self.node_count,
            "succeeded": self.succeeded,
            "duration_ms": self.duration_ms,
            "node_results": list(self.result.node_results.keys()) if self.result else [],
            "node_errors": self.result.node_errors if self.result else {},
        }
