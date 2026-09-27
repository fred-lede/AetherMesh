from __future__ import annotations

import pytest

from runtime.tools import web_search as ws


def test_default_order_when_no_config(monkeypatch):
    monkeypatch.setattr(ws, "_load_order", lambda: [])
    monkeypatch.setattr(ws, "_order_mtime", lambda: 1.0)
    assert [p.name for p in ws.WebSearchManager().providers] == ["exa", "tavily", "serper", "duckduckgo"]


def test_order_from_config(monkeypatch):
    monkeypatch.setattr(ws, "_load_order", lambda: ["serper", "exa", "tavily"])
    monkeypatch.setattr(ws, "_order_mtime", lambda: 1.0)
    names = [p.name for p in ws.WebSearchManager().providers]
    assert names[:3] == ["serper", "exa", "tavily"]
    assert "duckduckgo" in names


def test_order_reloads_on_mtime_change(monkeypatch):
    state = {"mtime": 1.0}
    monkeypatch.setattr(ws, "_order_mtime", lambda: state["mtime"])
    monkeypatch.setattr(ws, "_load_order", lambda: ["tavily", "exa", "serper", "duckduckgo"])
    mgr = ws.WebSearchManager()
    assert mgr.providers[0].name == "tavily"
    state["mtime"] = 2.0
    monkeypatch.setattr(ws, "_load_order", lambda: ["exa", "tavily", "serper", "duckduckgo"])
    assert mgr.providers[0].name == "exa"


def test_unknown_names_ignored_and_missing_appended(monkeypatch):
    monkeypatch.setattr(ws, "_load_order", lambda: ["bogus", "tavily"])
    monkeypatch.setattr(ws, "_order_mtime", lambda: 1.0)
    names = [p.name for p in ws.WebSearchManager().providers]
    assert "bogus" not in names
    assert names[0] == "tavily"
    assert set(names) == {"exa", "tavily", "serper", "duckduckgo"}
