from __future__ import annotations

import logging
from typing import Any

from runtime.orchestration.graph import ExecutionGraph, ExecutionNode, NodeType

logger = logging.getLogger("orchestration.planner")


class Planner:
    def plan(self, task: str, context: dict[str, Any] | None = None) -> ExecutionGraph:
        task_lower = task.lower()
        if any(kw in task_lower for kw in ("search", "research", "find", "look up", "what is", "tell me about")):
            return self._research_graph(task)
        if any(kw in task_lower for kw in ("write", "create", "generate", "draft", "compose")):
            return self._generate_graph(task)
        if any(kw in task_lower for kw in ("analyze", "summarize", "explain", "compare")):
            return self._analyze_graph(task)
        return self._simple_graph(task)

    def _research_graph(self, task: str) -> ExecutionGraph:
        graph = ExecutionGraph()
        search = ExecutionNode(
            id="search",
            node_type=NodeType.TOOL_CALL,
            config={"tool": "web_search", "query": task},
        )
        fetch = ExecutionNode(
            id="fetch",
            node_type=NodeType.TOOL_CALL,
            config={"tool": "web_fetch"},
            dependencies=["search"],
        )
        summarize = ExecutionNode(
            id="summarize",
            node_type=NodeType.LLM_CALL,
            config={"prompt": "Summarize the following research results", "input_from": "fetch"},
            dependencies=["fetch"],
        )
        generate = ExecutionNode(
            id="generate",
            node_type=NodeType.LLM_CALL,
            config={"prompt": "Generate final response based on research", "input_from": "summarize"},
            dependencies=["summarize"],
        )
        graph.add_node(search)
        graph.add_node(fetch)
        graph.add_node(summarize)
        graph.add_node(generate)
        return graph

    def _generate_graph(self, task: str) -> ExecutionGraph:
        graph = ExecutionGraph()
        plan = ExecutionNode(
            id="plan",
            node_type=NodeType.LLM_CALL,
            config={"prompt": "Plan the structure for: " + task},
        )
        generate = ExecutionNode(
            id="generate",
            node_type=NodeType.LLM_CALL,
            config={"prompt": "Generate content based on plan", "input_from": "plan"},
            dependencies=["plan"],
        )
        graph.add_node(plan)
        graph.add_node(generate)
        return graph

    def _analyze_graph(self, task: str) -> ExecutionGraph:
        graph = ExecutionGraph()
        search = ExecutionNode(
            id="search",
            node_type=NodeType.TOOL_CALL,
            config={"tool": "web_search", "query": task},
        )
        analyze = ExecutionNode(
            id="analyze",
            node_type=NodeType.LLM_CALL,
            config={"prompt": "Analyze the following information", "input_from": "search"},
            dependencies=["search"],
        )
        graph.add_node(search)
        graph.add_node(analyze)
        return graph

    def _simple_graph(self, task: str) -> ExecutionGraph:
        graph = ExecutionGraph()
        respond = ExecutionNode(
            id="respond",
            node_type=NodeType.LLM_CALL,
            config={"prompt": task},
        )
        graph.add_node(respond)
        return graph
