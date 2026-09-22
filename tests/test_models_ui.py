from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_models_tab_present_in_template():
    html = (ROOT / "dashboard" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'data-tab="models"' in html
    assert 'id="mm-table"' in html
    assert 'id="mm-category-filter"' in html
    assert 'id="mm-status"' in html
    assert 'id="mm-backdrop"' in html


def test_models_js_functions_present():
    js = (ROOT / "dashboard" / "static" / "dashboard.js").read_text(encoding="utf-8")
    for name in ["loadModels", "renderModelsTable", "openModelDrawer", "deleteModel", "duplicateModel",
                 "reloadModels", "saveModel", "fetchModelContext", "toggleModelCategory",
                 "setModelStatus", "encodeModelName"]:
        assert f"function {name}" in js, name


def test_drawer_is_outside_the_model_card():
    html = (ROOT / "dashboard" / "templates" / "index.html").read_text(encoding="utf-8")
    assert html.index('id="mm-drawer"') > html.index('id="traces-panel"')
    assert html.index('id="mm-backdrop"') > html.index('id="traces-panel"')
