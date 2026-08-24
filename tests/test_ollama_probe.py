from __future__ import annotations

import json as _json

import pytest
import requests
from unittest.mock import MagicMock

import runtime.health.watchdog as watchdog_mod
from runtime.health.ollama_probe import probe_ollama
from runtime.health.watchdog import Watchdog, merged_watchdog_config


class FakeHTTP:
    def __init__(self, tags=None, ps=None, gen=None, tags_exc=None, gen_exc=None):
        self._tags = tags if tags is not None else {"models": [{"name": "m:30b"}]}
        self._ps = ps if ps is not None else {"models": [{"name": "m:30b", "size_vram": 16000000000}]}
        self._gen = gen if gen is not None else {"done": True}
        self._tags_exc = tags_exc
        self._gen_exc = gen_exc
        self.gen_calls: list[dict] = []

    def get(self, url, timeout=None):
        if self._tags_exc:
            raise self._tags_exc
        if url.endswith("/api/tags"):
            r = requests.Response()
            r.status_code = 200
            r._content = b"{}"
            return r
        r = requests.Response()
        r.status_code = 200
        r._content = _json.dumps(self._ps).encode()
        return r

    def post(self, url, json=None, timeout=None):
        if self._gen_exc:
            raise self._gen_exc
        self.gen_calls.append(json)
        r = requests.Response()
        r.status_code = 200
        r._content = _json.dumps(self._gen).encode()
        return r


class TestProbe:
    def test_ok(self):
        res = probe_ollama("http://x:11434", session=FakeHTTP())
        assert res.status == "ok"
        assert res.model == "m:30b"
        assert res.stale_vram is False

    def test_unreachable_on_tags_error(self):
        res = probe_ollama("http://x", session=FakeHTTP(tags_exc=requests.ConnectionError("refused")))
        assert res.status == "unreachable"

    def test_idle_when_nothing_loaded(self):
        res = probe_ollama("http://x", session=FakeHTTP(ps={"models": []}))
        assert res.status == "idle"

    def test_infer_failure_on_timeout(self):
        res = probe_ollama("http://x", session=FakeHTTP(gen_exc=requests.Timeout("hang")))
        assert res.status == "infer_failed"
        assert "m:30b" in res.detail

    def test_infer_failure_when_done_missing(self):
        res = probe_ollama("http://x", session=FakeHTTP(gen={"done": False}))
        assert res.status == "infer_failed"

    def test_stale_vram_flagged_but_generate_heals(self):
        ps = {"models": [{"name": "m:30b", "size_vram": 0}]}
        res = probe_ollama("http://x", session=FakeHTTP(ps=ps))
        assert res.status == "ok"
        assert res.stale_vram is True

    def test_explicit_model_used(self):
        http = FakeHTTP()
        probe_ollama("http://x", model="my-model", session=http)
        assert http.gen_calls[0]["model"] == "my-model"


def _watchdog(oc_overrides):
    alerts = MagicMock()
    alerts.dispatch.return_value = 1
    wd = Watchdog(
        get_services=lambda: {},
        alert_manager=alerts,
        config_section={},
    )
    original_load = wd.load_config

    def load():
        cfg = original_load()
        cfg["ollama_deep_check"] = {**cfg["ollama_deep_check"], **oc_overrides}
        return cfg

    wd.load_config = load
    return wd, alerts


def _ollama_alerts(alerts):
    return [
        c
        for c in alerts.dispatch.call_args_list
        if "ollama" in str(c.kwargs.get("rule_key", ""))
    ]


class TestWatchdogIntegration:
    def test_disabled_returns_none(self):
        wd, alerts = _watchdog({"enabled": False})
        assert wd.check_once()["ollama"] is None
        assert _ollama_alerts(alerts) == []

    @pytest.fixture(autouse=True)
    def _fast_interval(self):
        return True

    def test_failures_trigger_alert_then_recovery(self, monkeypatch):
        fake = FakeHTTP(gen_exc=requests.Timeout("hang"))
        monkeypatch.setattr(
            "runtime.health.ollama_probe.probe_ollama",
            lambda **kw: _probe_from(fake),
        )
        wd, alerts = _watchdog({"enabled": True, "interval_s": 30, "failures_to_alert": 2})
        s1 = wd.check_once()
        assert s1["ollama"]["status"] == "infer_failed"
        assert _ollama_alerts(alerts) == []
        wd._ollama_last_run = 0.0
        s2 = wd.check_once()
        assert s2["ollama"]["status"] == "infer_failed"
        titles = [str(c.kwargs.get("title", "")) for c in alerts.dispatch.call_args_list]
        assert any("推論失敗" in t for t in titles)

        ok_fake = FakeHTTP()

        def ok_probe(**kw):
            return _probe_from(ok_fake)

        monkeypatch.setattr(
            "runtime.health.ollama_probe.probe_ollama", lambda **kw: ok_probe()
        )
        wd._ollama_last_run = 0.0
        wd.check_once()
        titles = [str(c.kwargs.get("title", "")) for c in alerts.dispatch.call_args_list]
        assert any("恢復" in t for t in titles)
        assert wd._ollama_failures == 0

    def test_interval_gating_skips_rerun(self, monkeypatch):
        calls = {"n": 0}

        def counting_probe(**kw):
            calls["n"] += 1
            return _probe_from(FakeHTTP())

        monkeypatch.setattr(
            "runtime.health.ollama_probe.probe_ollama", lambda **kw: counting_probe(**kw)
        )
        wd, _ = _watchdog({"enabled": True, "interval_s": 3600})
        wd.check_once()
        wd.check_once()
        assert calls["n"] == 1


def _probe_from(http):
    return probe_ollama("http://fake", session=http)


class TestMergedConfig:
    def test_ollama_defaults_merged_not_replaced(self):
        cfg = merged_watchdog_config({"ollama_deep_check": {"enabled": True}})
        oc = cfg["ollama_deep_check"]
        assert oc["enabled"] is True
        assert oc["timeout_s"] == 30
        assert oc["base_url"] == "http://127.0.0.1:11434"
