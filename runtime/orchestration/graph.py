from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("orchestration.graph")


class NodeType(Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    RETRY = "retry"


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionNode:
    id: str
    node_type: NodeType
    config: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str | None = None


@dataclass
class ExecutionGraph:
    nodes: dict[str, ExecutionNode] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)

    def add_node(self, node: ExecutionNode) -> None:
        self.nodes[node.id] = node
        if not node.dependencies:
            self.entry_points.append(node.id)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    errors.append(f"Node {node_id} depends on missing node {dep}")
        visited: set[str] = set()
        path: list[str] = []

        def _dfs(nid: str) -> bool:
            if nid in path:
                cycle_start = path[path.index(nid):]
                errors.append(f"Cycle detected: {' -> '.join(cycle_start + [nid])}")
                return True
            if nid in visited:
                return False
            path.append(nid)
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for dep in node.dependencies:
                    if _dfs(dep):
                        return True
            path.pop()
            return False

        for nid in list(self.nodes.keys()):
            _dfs(nid)
        return errors

    def topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = {}
        dependents: dict[str, list[str]] = {}
        for nid in self.nodes:
            in_degree.setdefault(nid, 0)
            dependents.setdefault(nid, [])
            node = self.nodes[nid]
            for dep in node.dependencies:
                in_degree[nid] += 1
                dependents.setdefault(dep, []).append(nid)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[str] = []

        while queue:
            queue.sort()
            nid = queue.pop(0)
            result.append(nid)
            for dep_id in dependents.get(nid, []):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        return result

    def parallel_groups(self) -> list[list[str]]:
        in_degree: dict[str, int] = {}
        dependents: dict[str, list[str]] = {}
        for nid in self.nodes:
            in_degree.setdefault(nid, 0)
            dependents.setdefault(nid, [])
            node = self.nodes[nid]
            for dep in node.dependencies:
                in_degree[nid] += 1
                dependents.setdefault(dep, []).append(nid)

        remaining = set(self.nodes.keys())
        groups: list[list[str]] = []
        while remaining:
            batch = sorted([nid for nid in remaining if in_degree.get(nid, 0) == 0])
            if not batch:
                break
            groups.append(batch)
            for nid in batch:
                remaining.discard(nid)
                for dep_id in dependents.get(nid, []):
                    in_degree[dep_id] -= 1
        return groups
