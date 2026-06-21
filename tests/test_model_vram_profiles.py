from __future__ import annotations

from config.settings import Settings


def test_learned_vram_profile_is_persisted_and_used(tmp_path) -> None:
    settings = Settings(config_dir=tmp_path)

    settings.record_model_vram_profiles({"learned-model": 12345})

    assert settings.model_vram_estimate_mb("learned-model") == 12345
    assert (tmp_path / "model_vram_profiles.json").exists()


def test_configured_vram_estimate_overrides_learned_profile(tmp_path) -> None:
    (tmp_path / "models.yaml").write_text(
        "models:\n  - name: configured-model\n    estimated_vram_mb: 20000\n",
        encoding="utf-8",
    )
    settings = Settings(config_dir=tmp_path)
    settings.record_model_vram_profiles({"configured-model": 10000})

    assert settings.model_vram_estimate_mb("configured-model") == 20000
