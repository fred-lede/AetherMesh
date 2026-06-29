from __future__ import annotations

from providers.asr_base import ASRProviderAdapter, ASRProviderError


def test_asr_provider_error_is_runtime_error() -> None:
    assert issubclass(ASRProviderError, RuntimeError)


def test_asr_provider_error_default_status() -> None:
    err = ASRProviderError("boom")
    assert err.status_code == 500


def test_asr_provider_error_custom_status() -> None:
    err = ASRProviderError("not found", status_code=404)
    assert err.status_code == 404


def test_asr_provider_adapter_is_abstract() -> None:
    import pytest
    with pytest.raises(TypeError):
        ASRProviderAdapter()  # type: ignore[abstract]


def test_abstract_methods_exist() -> None:
    import inspect
    methods = [
        m for m in ASRProviderAdapter.__dict__
        if not m.startswith("_") or m == "__init__"
    ]
    assert "transcribe" in inspect.getsource(ASRProviderAdapter.transcribe)


def test_provider_name_default() -> None:
    assert ASRProviderAdapter.provider_name == "asr_base"
