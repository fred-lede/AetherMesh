from __future__ import annotations

import pytest

from runtime.memory.short_term import ShortTermMemory
from runtime.memory.semantic_memory import SemanticMemory
from runtime.memory.episodic_memory import EpisodicMemory
from runtime.memory.memory_manager import MemoryManager
from runtime.sessions.session_store import session_store


@pytest.fixture(autouse=True)
def _isolate_session_store(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_path", tmp_path / "sessions.json")
    session_store._sessions.clear()
    yield
    session_store._sessions.clear()


def test_short_term_roundtrip() -> None:
    mem = ShortTermMemory()
    mem.add_message("s1", "user", "hello")
    mem.add_message("s1", "assistant", "world")
    ctx = mem.get_context("s1")
    assert len(ctx) == 2
    assert ctx[0]["content"] == "hello"
    mem.clear("s1")
    assert mem.get_context("s1") == []


def test_short_term_scoped() -> None:
    mem = ShortTermMemory()
    mem.add_message("s1", "user", "v1")
    mem.add_message("s2", "user", "v2")
    assert len(mem.get_context("s1")) == 1
    assert len(mem.get_context("s2")) == 1


def test_semantic_store_and_search() -> None:
    mem = SemanticMemory()
    id1 = mem.store("the cat sat on the mat")
    id2 = mem.store("the dog played in the yard")
    results = mem.search("cat mat", top_k=2)
    assert len(results) == 1  # only doc1 matches
    assert results[0].id == id1
    assert results[0].score > 0


def test_semantic_empty_search() -> None:
    mem = SemanticMemory()
    assert mem.search("nothing") == []


def test_semantic_get_delete() -> None:
    mem = SemanticMemory()
    eid = mem.store("hello world")
    assert mem.get(eid) is not None
    mem.delete(eid)
    assert mem.get(eid) is None
    assert mem.count() == 0


def test_episodic_record_and_query() -> None:
    mem = EpisodicMemory()
    mem.record(session_id="s1", provider="p1", model="m1", success=True, duration_ms=100)
    mem.record(session_id="s1", provider="p2", model="m2", success=False, duration_ms=50, error="fail")
    results = mem.by_session("s1")
    assert len(results) == 2
    results = mem.by_provider("p1")
    assert len(results) == 1
    assert results[0].success is True
    assert mem.success_rate("p1") == 1.0
    assert mem.success_rate("p2") == 0.0


async def test_memory_manager_provides_all() -> None:
    st = ShortTermMemory()
    sm = SemanticMemory()
    em = EpisodicMemory()
    mm = MemoryManager(short_term=st, semantic=sm, episodic=em)
    assert mm.short_term is st
    assert mm.semantic is sm
    assert mm.episodic is em
    await mm.record_execution(session_id="s1", provider="p1", model="m1", success=True, duration_ms=100)
    assert mm.session_context("s1") is not None
