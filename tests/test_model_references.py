from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from runtime.orchestration import model_references as refs


@pytest.fixture()
def rules_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "routing_rules.yaml"
    monkeypatch.setattr(refs, "_rules_path", lambda: path)
    return path


def test_scan_finds_aliases_and_fallbacks(rules_file: Path):
    rules_file.write_text(
        yaml.safe_dump({
            "aliases": {"fast": "old-model"},
            "fallbacks": {"old-model": ["a", "b"]},
            "rules": [{"model": "old-model", "provider": "ollama"}],
        }),
        encoding="utf-8",
    )
    found = refs.scan_model_references("old-model")
    paths = {f["path"] for f in found}
    assert "aliases.fast" in paths
    assert "fallbacks.old-model" in paths
    assert "rules[0].model" in paths


def test_scan_returns_empty_when_absent(rules_file: Path):
    rules_file.write_text(yaml.safe_dump({"aliases": {"fast": "other"}}), encoding="utf-8")
    assert refs.scan_model_references("old-model") == []


def test_update_rewrites_values_and_preserves_others(rules_file: Path):
    rules_file.write_text(
        yaml.safe_dump({
            "aliases": {"fast": "old-model", "keep": "other"},
            "fallbacks": {"old-model": ["a"]},
        }),
        encoding="utf-8",
    )
    refs.update_model_references("old-model", "new-model")
    data = yaml.safe_load(rules_file.read_text(encoding="utf-8"))
    assert data["aliases"]["fast"] == "new-model"
    assert data["aliases"]["keep"] == "other"
    assert "new-model" in data["fallbacks"]
    assert "old-model" not in data["fallbacks"]
