from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from runtime.orchestration.graph import ExecutionGraph, ExecutionNode, NodeStatus
from runtime.orchestration.retry_policy import RetryPolicy

logger = logging.getLogger("orchestration.executor")


class GraphExecutionResult:
    def __init__(self) -> None:
        self.node_results: dict[str, Any] = {}
        self.node_errors: dict[str, str] = {}
        self.node_durations: dict[str, float] = {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.cancelled: bool = False

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def success(self) -> bool:
        return not self.node_errors and not self.cancelled

    @property
    def output(self) -> Any:
        if self.node_results:
            last_id = list(self.node_results.keys())[-1]
            return self.node_results[last_id]
        return None


class GraphExecutor:
    def __init__(self, retry_policy: RetryPolicy | None = None) -> None:
        self._retry_policy = retry_policy or RetryPolicy()
        self._handlers: dict[str, Callable[[ExecutionNode], Any]] = {}

    def register_handler(
        self, node_type: str, handler: Callable[[ExecutionNode], Any],
    ) -> None:
        self._handlers[node_type] = handler

    async def execute(
        self,
        graph: ExecutionGraph,
        context: dict[str, Any] | None = None,
    ) -> GraphExecutionResult:
        result = GraphExecutionResult()
        result.start_time = time.time()
        errors = graph.validate()
        if errors:
            for err in errors:
                logger.error("Graph validation error: %s", err)
            result.end_time = time.time()
            result.node_errors["_graph"] = "; ".join(errors)
            return result

        groups = graph.parallel_groups()
        node_states: dict[str, ExecutionNode] = {
            nid: ExecutionNode(
                id=n.id, node_type=n.node_type,
                config=dict(n.config), dependencies=list(n.dependencies),
                status=NodeStatus.PENDING,
            )
            for nid, n in graph.nodes.items()
        }

        for group in groups:
            tasks: list[asyncio.Task] = []
            for nid in group:
                node = node_states.get(nid)
                if not node or node.status != NodeStatus.PENDING:
                    continue
                tasks.append(asyncio.create_task(
                    self._execute_node(node, context, result)
                ))

            if tasks:
                await asyncio.gather(*tasks)

        result.end_time = time.time()
        return result

    async def _execute_node(
        self,
        node: ExecutionNode,
        context: dict[str, Any] | None,
        result: GraphExecutionResult,
    ) -> None:
        node.status = NodeStatus.RUNNING
        start = time.time()
        try:
            handler = self._handlers.get(node.node_type.value)
            if handler:
                node.result = await self._retry_policy.execute(
                    lambda: handler(node),
                    node_id=node.id,
                )
            node.status = NodeStatus.COMPLETED
        except Exception as exc:
            node.status = NodeStatus.FAILED
            node.error = str(exc)
            result.node_errors[node.id] = str(exc)
            logger.error("Node %s failed: %s", node.id, exc)
        finally:
            duration = (time.time() - start) * 1000
            result.node_results[node.id] = node.result
            result.node_durations[node.id] = duration
            logger.debug(
                "Node %s (%s) %s in %.0fms",
                node.id, node.node_type.value, node.status.value, duration,
            )
