from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature

import pytest

from providers.tts_base import TTSProviderAdapter, TTSProviderError


class TestTTSProviderAdapterInterface:
    """Verify the ABC contract: all abstract methods exist."""

    def test_is_abstract(self) -> None:
        assert issubclass(TTSProviderAdapter, ABC)

    def test_abstract_methods(self) -> None:
        expected = {"tts", "list_voices", "register_voice", "delete_voice", "health_check"}
        abstract = {m for m in TTSProviderAdapter.__abstractmethods__}
        assert abstract == expected, f"Missing abstracts: {expected - abstract}"

    def test_tts_signature(self) -> None:
        sig = signature(TTSProviderAdapter.tts)
        assert "payload" in sig.parameters

    def test_list_voices_signature(self) -> None:
        sig = signature(TTSProviderAdapter.list_voices)
        assert len(sig.parameters) == 1

    def test_register_voice_signature(self) -> None:
        sig = signature(TTSProviderAdapter.register_voice)
        for param in ("name", "audio_data"):
            assert param in sig.parameters

    def test_delete_voice_signature(self) -> None:
        sig = signature(TTSProviderAdapter.delete_voice)
        assert "voice_id" in sig.parameters

    def test_provider_name_default(self) -> None:
        assert TTSProviderAdapter.provider_name == "tts_base"


class TestTTSProviderError:
    def test_default_status_code(self) -> None:
        err = TTSProviderError("oops")
        assert err.status_code == 500

    def test_custom_status_code(self) -> None:
        err = TTSProviderError("not found", status_code=404)
        assert err.status_code == 404

    def test_is_runtime_error(self) -> None:
        assert issubclass(TTSProviderError, RuntimeError)
