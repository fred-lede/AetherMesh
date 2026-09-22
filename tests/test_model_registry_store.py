from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.orchestration import model_registry_store as store


@pytest.fixture()
def models_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.yaml"
    monkeypatch.setattr(store, "models_path", lambda: path)
    store.reload_models()
    return path


def test_load_missing_file_returns_empty(models_file: Path):
    assert store.load_models() == []


def test_load_null_models_returns_empty(models_file: Path):
    models_file.write_text("models:\n", encoding="utf-8")
    assert store.load_models() == []


def test_get_models_caches_until_mtime_changes(models_file: Path):
    models_file.write_text("models:\n- name: a\n  provider: ollama\n", encoding="utf-8")
    first = store.get_models()
    assert first[0]["name"] == "a"
    assert store.get_models() is first
    models_file.write_text("models:\n- name: b\n  provider: ollama\n", encoding="utf-8")
    os.utime(models_file, (models_file.stat().st_mtime + 10, models_file.stat().st_mtime + 10))
    assert store.get_models()[0]["name"] == "b"


def test_save_models_writes_and_backs_up(models_file: Path):
    store.save_models([{"name": "x", "provider": "ollama"}])
    assert "name: x" in models_file.read_text(encoding="utf-8")
    store.save_models([{"name": "y", "provider": "ollama"}])
    bak = models_file.with_suffix(".yaml.bak")
    assert bak.exists()
    assert "name: x" in bak.read_text(encoding="utf-8")


def test_save_preserves_other_top_level_keys(models_file: Path):
    models_file.write_text("version: 2\nmodels: []\n", encoding="utf-8")
    store.save_models([{"name": "x", "provider": "ollama"}])
    text = models_file.read_text(encoding="utf-8")
    assert "version: 2" in text
    assert "name: x" in text


def test_save_retries_on_permission_error(models_file: Path, monkeypatch: pytest.MonkeyPatch):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky)
    store.save_models([{"name": "x", "provider": "ollama"}])
    assert calls["n"] == 3
    assert "name: x" in models_file.read_text(encoding="utf-8")


def test_normalize_canonicalizes_capabilities():
    out = store.normalize_model({"name": " m ", "provider": "ollama", "capabilities": ["Vision", "image", "image"]})
    assert out["name"] == "m"
    assert out["capabilities"] == ["image_gen", "vision"]


def test_normalize_local_keeps_worker_bindings():
    out = store.normalize_model({
        "name": "m", "provider": "ollama",
        "worker_bindings": [{"node_id": "node-01", "port": "11434"}],
    })
    assert out["worker_bindings"] == [{"node_id": "node-01", "port": 11434}]
    assert "worker_ports" not in out


def test_normalize_cloud_uses_worker_ports():
    out = store.normalize_model({"name": "m", "provider": "openai", "worker_ports": []})
    assert out["worker_ports"] == []
    assert "worker_bindings" not in out


def test_is_local_model():
    assert store.is_local_model({"provider": "ollama"}) is True
    assert store.is_local_model({"provider": "openai"}) is False


def test_validate_rejects_missing_name():
    _, errors = store.validate_model({"provider": "ollama"})
    assert any("name" in e for e in errors)


def test_validate_rejects_duplicate_name():
    _, errors = store.validate_model({"name": "a", "provider": "ollama"}, existing_names={"a"})
    assert any("duplicate" in e for e in errors)


def test_validate_rejects_bad_context_length():
    _, errors = store.validate_model({"name": "a", "provider": "ollama", "context_length": 0})
    assert any("context_length" in e for e in errors)


def test_validate_rejects_local_without_bindings():
    _, errors = store.validate_model({"name": "a", "provider": "ollama"})
    assert any("worker_bindings" in e for e in errors)


def test_validate_rejects_bad_port():
    entry = {"name": "a", "provider": "ollama", "worker_bindings": [{"node_id": "n", "port": 70000}]}
    _, errors = store.validate_model(entry)
    assert any("port" in e for e in errors)


def test_validate_accepts_good_local_model():
    entry = {"name": "a", "provider": "ollama", "worker_bindings": [{"node_id": "node-01", "port": 11434}],
             "capabilities": ["chat"], "context_length": 32768}
    clean, errors = store.validate_model(entry)
    assert errors == []
    assert clean["context_length"] == 32768


def test_load_filters_entries_without_name(models_file: Path):
    models_file.write_text("models:\n- {provider: ollama}\n- not-a-dict\n- name: ok\n  provider: ollama\n", encoding="utf-8")
    assert [m["name"] for m in store.load_models()] == ["ok"]


def test_validate_non_numeric_context_does_not_raise():
    _, errors = store.validate_model(
        {"name": "a", "provider": "openai", "worker_ports": [], "capabilities": ["chat"], "context_length": "abc"}
    )
    assert any("context_length" in e for e in errors)


def test_validate_string_binding_does_not_raise():
    _, errors = store.validate_model({"name": "a", "provider": "ollama", "worker_bindings": ["oops"]})
    assert errors


def test_validate_rejects_case_insensitive_duplicate():
    _, errors = store.validate_model(
        {"name": "Foo", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]}, existing_names={"foo"}
    )
    assert any("duplicate" in e for e in errors)


def test_validate_rejects_unknown_provider():
    _, errors = store.validate_model({"name": "a", "provider": "bogus", "worker_ports": [], "capabilities": ["chat"]})
    assert any("provider" in e for e in errors)


def test_validate_rejects_unknown_node():
    entry = {"name": "a", "provider": "ollama", "worker_bindings": [{"node_id": "ghost", "port": 11434}], "capabilities": ["chat"]}
    _, errors = store.validate_model(entry, allowed_nodes={"node-01"})
    assert any("node" in e for e in errors)


def test_is_local_model_treats_custom_provider_as_cloud(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(store, "cloud_providers", lambda: {"agnes"})
    assert store.is_local_model({"provider": "agnes"}) is False


