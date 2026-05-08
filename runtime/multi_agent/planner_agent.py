from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("multi_agent.planner")


@dataclass
class SubTask:
    id: str = ""
    agent_id: str = ""
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    tools: list[str] | None = None
    expected_output: str = ""


class PlannerAgent:
    def __init__(self, agent_id: str = "planner") -> None:
        self.agent_id = agent_id

    def plan(
        self,
        task: str,
        available_agents: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[SubTask]:
        task_lower = task.lower()
        has_research = any(kw in task_lower for kw in ("search", "research", "find", "look up"))
        has_generate = any(kw in task_lower for kw in ("write", "create", "generate", "draft", "compose"))
        has_analyze = any(kw in task_lower for kw in ("analyze", "summarize", "explain", "compare"))

        subtasks: list[SubTask] = []
        used: set[str] = set()
        pool = [a for a in available_agents if a != self.agent_id]

        if has_research:
            agent = pool[len(used) % len(pool)] if pool else "worker"
            used.add(agent)
            subtasks.append(SubTask(
                id="research",
                agent_id=agent,
                description=f"Search and gather information about: {task}",
                tools=["web_search", "web_fetch"],
                expected_output="raw research findings with sources",
            ))

        if has_analyze:
            agent = pool[len(used) % len(pool)] if pool else "worker"
            used.add(agent)
            deps = ["research"] if has_research else []
            subtasks.append(SubTask(
                id="analyze",
                agent_id=agent,
                description=f"Analyze and interpret findings for: {task}",
                dependencies=deps,
                expected_output="structured analysis with key insights",
            ))

        if has_generate:
            agent = pool[len(used) % len(pool)] if pool else "worker"
            used.add(agent)
            deps: list[str] = []
            if has_research:
                deps.append("research")
            if has_analyze:
                deps.append("analyze")
            subtasks.append(SubTask(
                id="generate",
                agent_id=agent,
                description=f"Produce final output for: {task}",
                dependencies=deps,
                expected_output="complete final response",
            ))

        if not subtasks:
            agent = pool[0] if pool else "worker"
            subtasks.append(SubTask(
                id="execute",
                agent_id=agent,
                description=task,
                expected_output="complete response",
            ))

        return subtasks
