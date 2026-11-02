from __future__ import annotations

from config.settings import Settings, settings


def test_asr_provider_env_default(monkeypatch):
    monkeypatch.delenv("AIIH_ASR_PROVIDER", raising=False)
    assert Settings().asr_provider == "faster_whisper"


def test_asr_provider_env_override(monkeypatch):
    monkeypatch.setenv("AIIH_ASR_PROVIDER", "qwen_asr")
    assert Settings().asr_provider == "qwen_asr"


def test_load_asr_config_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    assert settings.load_asr_config() == {}


def test_asr_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    data = {"target_id": "local:cuda:1", "provider": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B", "updated_at": "2026-11-03T00:00:00+00:00"}
    settings.save_asr_config(data)
    assert settings.load_asr_config() == data
    assert (tmp_path / "asr_config.json").exists()
    assert not (tmp_path / "asr_config.tmp").exists()


def test_load_asr_config_corrupt_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    (tmp_path / "asr_config.json").write_text("{not json", encoding="utf-8")
    assert settings.load_asr_config() == {}


def test_load_asr_config_non_dict_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    (tmp_path / "asr_config.json").write_text("[1, 2]", encoding="utf-8")
    assert settings.load_asr_config() == {}
