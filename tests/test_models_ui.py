from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_models_tab_present_in_template():
    html = (ROOT / "dashboard" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'data-tab="models"' in html
    assert 'id="mm-table"' in html
    assert 'id="mm-category-filter"' in html


def test_models_js_functions_present():
    js = (ROOT / "dashboard" / "static" / "dashboard.js").read_text(encoding="utf-8")
    for name in ["loadModels", "renderModelsTable", "openModelDrawer", "deleteModel", "duplicateModel", "reloadModels", "saveModel", "fetchModelContext", "toggleModelCategory"]:
        assert f"function {name}" in js, name
