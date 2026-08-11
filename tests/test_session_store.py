from __future__ import annotations

from pathlib import Path

from config.settings import settings
from runtime.sessions.session_store import SessionStore


def make_store(tmp_path: Path) -> SessionStore:
    return SessionStore(path=tmp_path / "sessions.json")


def test_create_and_get(tmp_path):
    store = make_store(tmp_path)
    session = store.create("s1", client_type="cli", ttl_s=60)
    assert store.get("s1").id == "s1"
    assert session.client_type == "cli"


def test_append_and_context(tmp_path):
    store = make_store(tmp_path)
    store.create("s1")
    store.append_message("s1", {"role": "user", "content": "hello"})
    store.append_message("s1", {"role": "assistant", "content": "hi"})
    context = store.get_context("s1", max_messages=1)
    assert context == [{"role": "assistant", "content": "hi"}]


def test_delete(tmp_path):
    store = make_store(tmp_path)
    store.create("s1")
    assert store.delete("s1") is True
    assert store.delete("s1") is False
    assert store.get("s1") is None


def test_update_metadata(tmp_path):
    store = make_store(tmp_path)
    store.create("s1")
    store.update_metadata("s1", {"topic": "gpu"})
    assert store.get("s1").metadata["topic"] == "gpu"
    assert store.update_metadata("missing", {"a": 1}) is None


def test_persistence_reload(tmp_path):
    store = make_store(tmp_path)
    store.create("s1", client_type="web", ttl_s=3600)
    store.append_message("s1", {"role": "user", "content": "persisted"})
    store.update_metadata("s1", {"k": "v"})
    reloaded = make_store(tmp_path)
    session = reloaded.get("s1")
    assert session.message_count == 1
    assert session.metadata["k"] == "v"


def test_evict_expired(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.create("s1", ttl_s=1)
    session = store.get("s1")
    session.expires_at = 1.0
    assert store.evict_expired() == 1
    assert store.get("s1") is None


def test_summarize_and_trim(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.create("s1")
    for i in range(20):
        store.append_message("s1", {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})
    session = store.summarize_and_trim("s1", lambda transcript: "SUMMARY", threshold=15, keep_last=4)
    assert session.metadata["summary"] == "SUMMARY"
    assert len(session.messages) == 1 + 4
    assert session.messages[0]["content"].startswith("[Prior conversation summary]")


def test_summarize_and_trim_under_threshold(tmp_path):
    store = make_store(tmp_path)
    store.create("s1")
    store.append_message("s1", {"role": "user", "content": "hi"})
    session = store.summarize_and_trim("s1", lambda transcript: "X", threshold=40)
    assert "summary" not in session.metadata
    assert session.message_count == 1


def test_summarize_failure_keeps_session(tmp_path):
    store = make_store(tmp_path)
    store.create("s1")
    for i in range(10):
        store.append_message("s1", {"role": "user", "content": f"m{i}"})

    def boom(transcript):
        raise RuntimeError("provider down")

    session = store.summarize_and_trim("s1", boom, threshold=5, keep_last=2)
    assert session.message_count == 10
    assert "summary" not in session.metadata


def test_summarize_combines_prior_summaries(tmp_path):
    store = make_store(tmp_path)
    store.create("s1")
    for i in range(8):
        store.append_message("s1", {"role": "user", "content": f"m{i}"})
    store.summarize_and_trim("s1", lambda t: "FIRST", threshold=5, keep_last=2)
    session = store.get("s1")
    for i in range(5):
        store.append_message("s1", {"role": "user", "content": f"extra{i}"})
    store.summarize_and_trim("s1", lambda t: "SECOND", threshold=5, keep_last=2)
    assert "FIRST\nSECOND" in store.get("s1").metadata["summary"]


def test_settings_threshold_default():
    assert hasattr(settings, "session_summarize_threshold")
    assert settings.session_summarize_threshold > 0
