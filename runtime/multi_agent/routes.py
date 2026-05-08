from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter

from runtime.multi_agent import coordinator, shared_memory
from runtime.multi_agent.coordinator import OrchestrationPlan
from runtime.multi_agent.planner_agent import PlannerAgent
from runtime.multi_agent.worker_agent import WorkerAgent

agent_router = APIRouter(prefix="/v1/agent", tags=["multi-agent"])
planner = PlannerAgent()


@agent_router.get("/status")
def agent_status():
    return {
        "agents": coordinator.list_agents(),
        "shared_memory_keys": shared_memory.global_keys(),
    }


@agent_router.post("/plan")
def agent_plan(task: str):
    available = coordinator.list_agents() or ["worker"]
    subtasks = planner.plan(task, available)
    return {
        "task": task,
        "subtasks": [
            {
                "id": s.id,
                "agent_id": s.agent_id,
                "description": s.description,
                "dependencies": s.dependencies,
                "tools": s.tools,
            }
            for s in subtasks
        ],
    }


@agent_router.post("/execute")
async def agent_execute(task: str, agent_id: str = ""):
    if agent_id:
        result = await coordinator.delegate(task, agent_id)
        return {"ok": result.success, "output": str(result.output), "duration_ms": result.duration_ms, "error": result.error}
    available = coordinator.list_agents() or ["worker"]
    subtasks = planner.plan(task, available)
    plan = OrchestrationPlan(steps=[
        {"id": s.id, "agent_id": s.agent_id, "task": s.description, "dependencies": s.dependencies}
        for s in subtasks
    ])
    orchestration_result = await coordinator.orchestrate(plan)
    final_output = None
    for step_id, sub_result in orchestration_result.results.items():
        if sub_result.success and sub_result.output is not None:
            final_output = sub_result.output
    return {
        "ok": orchestration_result.success,
        "output": str(final_output) if final_output else "",
        "elapsed_ms": orchestration_result.elapsed_ms,
        "steps": {k: v.to_dict() for k, v in orchestration_result.results.items()},
    }


@agent_router.post("/register")
def agent_register(agent_id: str):
    if not coordinator.get_agent(agent_id):
        agent = WorkerAgent(agent_id=agent_id)
        coordinator.register_agent(agent_id, agent)
    return {"ok": True, "agent_id": agent_id}
