from __future__ import annotations

import pytest

from config.settings import settings
from runtime.rag.rag_store import RagStore, _cosine, chunk_text
from runtime.rag.injector import inject_rag_context


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    store = RagStore(path=tmp_path / "rag_store.json")
    monkeypatch.setattr("runtime.rag.rag_store.rag_store", store)
    monkeypatch.setattr("runtime.rag.injector.rag_store", store)
    yield store


def test_chunk_text_single_short():
    assert chunk_text("hello") == ["hello"]


def test_chunk_text_long_splits_and_overlaps():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) >= 20 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_ingest_and_count(isolated_store):
    ids = isolated_store.ingest("first chunk of data")
    assert len(ids) == 1
    assert isolated_store.count() == 1


def test_ingest_multiple_chunks(isolated_store):
    ids = isolated_store.ingest("paragraph one.\n\n" + "long content " * 300)
    assert len(ids) >= 2


def test_search_keyword_ranking(isolated_store):
    isolated_store.ingest("The cat sat on the mat.")
    isolated_store.ingest("Dogs like to run in the park.")
    results = isolated_store.search("cat mat", top_k=2)
    assert results[0]["text"] == "The cat sat on the mat."
    assert results[0]["score"] > results[1]["score"]


def test_search_empty_store(isolated_store):
    assert isolated_store.search("anything") == []


def test_search_filter_metadata(isolated_store):
    isolated_store.ingest("alpha data", metadata={"kind": "a"})
    isolated_store.ingest("alpha data", metadata={"kind": "b"})
    results = isolated_store.search("alpha", filter_metadata={"kind": "a"})
    assert len(results) == 1
    assert results[0]["metadata"]["kind"] == "a"


def test_search_with_embedding_cosine(isolated_store):
    isolated_store.ingest("apple fruit", embedding=[1.0, 0.0])
    isolated_store.ingest("car engine", embedding=[0.0, 1.0])
    results = isolated_store.search("fruit", embedding=[0.9, 0.1], top_k=2)
    assert results[0]["text"] == "apple fruit"
    assert results[0]["score"] > 0.5


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([], [1]) == 0.0


def test_delete_and_clear(isolated_store):
    ids = isolated_store.ingest("data")
    assert isolated_store.delete(ids[0]) is True
    assert isolated_store.delete(ids[0]) is False
    isolated_store.ingest("more data")
    assert isolated_store.clear() == 1
    assert isolated_store.count() == 0


def test_persistence_reload(tmp_path):
    store = RagStore(path=tmp_path / "rag_store.json")
    store.ingest("persisted content", metadata={"source": "test"})
    reloaded = RagStore(path=tmp_path / "rag_store.json")
    assert reloaded.count() == 1
    results = reloaded.search("persisted")
    assert results[0]["metadata"]["source"] == "test"


def test_inject_rag_context_appends_system(isolated_store, monkeypatch):
    isolated_store.ingest("AetherMesh supports GPU scheduling.")
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "rag_auto_inject", True)
    payload = {"model": "m", "messages": [{"role": "user", "content": "what is GPU scheduling?"}]}
    result = inject_rag_context(payload)
    system = result["messages"][0]
    assert system["role"] == "system"
    assert "GPU scheduling" in system["content"]
    assert "AetherMesh RAG context" in system["content"]


def test_inject_rag_context_disabled(isolated_store, monkeypatch):
    isolated_store.ingest("content")
    monkeypatch.setattr(settings, "rag_enabled", False)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert inject_rag_context(payload) == payload


def test_inject_rag_context_idempotent(isolated_store, monkeypatch):
    isolated_store.ingest("topic details")
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "rag_auto_inject", True)
    payload = {"model": "m", "messages": [{"role": "user", "content": "topic"}]}
    first = inject_rag_context(payload)
    second = inject_rag_context(first)
    systems = [m for m in second["messages"] if m["role"] == "system"]
    assert len(systems) == 1


def test_inject_rag_context_empty_store(isolated_store, monkeypatch):
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "rag_auto_inject", True)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert inject_rag_context(payload) == payload
