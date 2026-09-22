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
