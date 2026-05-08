from __future__ import annotations

import asyncio

from runtime.multi_agent.shared_memory import SharedMemory
from runtime.multi_agent.worker_agent import WorkerAgent
from runtime.multi_agent.planner_agent import PlannerAgent
from runtime.multi_agent.coordinator import Coordinator, OrchestrationPlan


def test_shared_memory_scoped() -> None:
    sm = SharedMemory()
    sm.write("a1", "key1", "val1")
    sm.write("a2", "key1", "val2")
    assert sm.read("a1", "key1") == "val1"
    assert sm.read("a2", "key1") == "val2"


def test_shared_memory_broadcast() -> None:
    sm = SharedMemory()
    sm.broadcast("global_key", "global_val", source_agent="a1")
    assert sm.read_global("global_key") == "global_val"


def test_shared_memory_clear() -> None:
    sm = SharedMemory()
    sm.write("a1", "k", "v")
    sm.broadcast("gk", "gv")
    sm.clear_agent("a1")
    assert sm.read("a1", "k") is None
    assert sm.read_global("gk") == "gv"
    sm.clear_all()
    assert sm.read_global("gk") is None


async def test_worker_execute_with_handler() -> None:
    async def handler(task: str, ctx: dict) -> str:
        return f"done: {task}"
    w = WorkerAgent(agent_id="w1", handler=handler)
    r = await w.execute("test task")
    assert r.success is True
    assert r.output == "done: test task"
    assert r.agent_id == "w1"


async def test_worker_execute_without_handler() -> None:
    w = WorkerAgent(agent_id="w1")
    r = await w.execute("hello")
    assert r.success is True
    assert "[w1] processed:" in r.output


async def test_worker_execute_error() -> None:
    async def fail(task: str, ctx: dict) -> str:
        raise RuntimeError("oops")
    w = WorkerAgent(agent_id="w1", handler=fail)
    r = await w.execute("fail")
    assert r.success is False
    assert "oops" in r.error


def test_planner_agent_decomposes_research() -> None:
    p = PlannerAgent()
    subtasks = p.plan("search for AI trends and write a summary", ["worker-1"])
    ids = [s.id for s in subtasks]
    assert "research" in ids
    assert "generate" in ids


def test_planner_agent_simple_task() -> None:
    p = PlannerAgent()
    subtasks = p.plan("hello", ["worker-1"])
    assert len(subtasks) == 1
    assert subtasks[0].id == "execute"


async def test_coordinator_delegate() -> None:
    c = Coordinator()
    async def handler(task: str, ctx: dict) -> str:
        return f"done: {task}"
    c.register_agent("w1", WorkerAgent(agent_id="w1", handler=handler))
    r = await c.delegate("hello", "w1")
    assert r.success is True
    assert r.output == "done: hello"


async def test_coordinator_delegate_unknown() -> None:
    c = Coordinator()
    r = await c.delegate("hello", "nobody")
    assert r.success is False
    assert "Unknown agent" in r.error


async def test_coordinator_fan_out() -> None:
    c = Coordinator()
    c.register_agent("w1", WorkerAgent(agent_id="w1"))
    c.register_agent("w2", WorkerAgent(agent_id="w2"))
    results = await c.fan_out("task", ["w1", "w2"])
    assert len(results) == 2
    assert all(r.success for r in results)


async def test_coordinator_orchestrate() -> None:
    c = Coordinator()
    call_order: list[str] = []
    async def make_handler(aid: str):
        async def h(task: str, ctx: dict) -> str:
            call_order.append(aid)
            return f"{aid}: {task}"
        return h
    c.register_agent("w1", WorkerAgent(agent_id="w1", handler=await make_handler("w1")))
    c.register_agent("w2", WorkerAgent(agent_id="w2", handler=await make_handler("w2")))
    plan = OrchestrationPlan(steps=[
        {"id": "step1", "agent_id": "w1", "task": "first"},
        {"id": "step2", "agent_id": "w2", "task": "second", "dependencies": ["step1"]},
    ])
    r = await c.orchestrate(plan)
    assert r.success is True
    assert len(r.results) == 2
    assert r.results["step1"].success is True
    assert call_order == ["w1", "w2"]
