from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "dashboard" / "templates" / "admin_audio.html"
SCRIPT = REPO_ROOT / "dashboard" / "static" / "admin_audio.js"


def _template_window_refs() -> set[str]:
    html = TEMPLATE.read_text(encoding="utf-8")
    return set(re.findall(r"window\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", html))


def _script_window_exports() -> set[str]:
    js = SCRIPT.read_text(encoding="utf-8")
    return set(re.findall(r"window\.([A-Za-z_][A-Za-z0-9_]*)\s*=", js))


def test_template_window_refs_are_all_exported_by_script() -> None:
    refs = _template_window_refs()
    exports = _script_window_exports()
    missing = sorted(refs - exports)
    assert not missing, (
        "admin_audio.html calls window.* functions that admin_audio.js never "
        f"exports: {missing}. Inline handlers raise TypeError at runtime."
    )


def test_speech_model_change_handler_is_exported() -> None:
    exports = _script_window_exports()
    assert "onSpeechModelChanged" in exports, (
        "window.onSpeechModelChanged is not exported; switching the Speech tab "
        "model never refreshes the voice menu."
    )


def test_download_speech_handler_is_exported() -> None:
    exports = _script_window_exports()
    assert "downloadSpeech" in exports, (
        "window.downloadSpeech is not exported; the Speech tab download "
        "button's inline handler raises TypeError."
    )


@pytest.mark.parametrize("name", ["onSpeechModelChanged", "downloadSpeech"])
def test_exported_handlers_are_callable_functions(name: str) -> None:
    js = SCRIPT.read_text(encoding="utf-8")
    patterns = [
        rf"window\.{name}\s*=\s*function",
        rf"window\.{name}\s*=\s*async\s+function",
        rf"window\.{name}\s*=\s*\w+",
    ]
    assert any(re.search(p, js) for p in patterns), (
        f"window.{name} must be assigned a callable function"
    )
