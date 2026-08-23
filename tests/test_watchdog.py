from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import runtime.health.watchdog as watchdog_mod
from runtime.health.watchdog import Watchdog, merged_watchdog_config


class FakeSP:
    def __init__(self, status: str = "running", pid: int | None = None) -> None:
        self.status = status
        self.process = SimpleNamespace(pid=pid, poll=lambda: None) if pid else None


def _watchdog(services, restart_fn=None):
    alerts = MagicMock()
    alerts.dispatch.return_value = 1
    wd = Watchdog(
        get_services=lambda: services,
        alert_manager=alerts,
        restart_fn=restart_fn,
        config_section={},
    )
    return wd, alerts


@pytest.fixture(autouse=True)
def _env_ports(monkeypatch):
    monkeypatch.setenv("AIIH_ROUTER_PORT", "58001")
    yield


class TestMergedConfig:
    def test_defaults(self) -> None:
        cfg = merged_watchdog_config(None)
        assert cfg["enabled"] is True
        assert cfg["interval_s"] == 30
        assert "service_rss_mb" in cfg["rules"]
        assert cfg["auto_restart"]["enabled"] is False

    def test_override_merges_rules_not_replace(self) -> None:
        cfg = merged_watchdog_config(
            {
                "interval_s": 10,
                "rules": {"service_rss_mb": {"warn": 100}},
                "auto_restart": {"enabled": True},
            }
        )
        assert cfg["interval_s"] == 10
        assert cfg["rules"]["service_rss_mb"]["warn"] == 100
        assert cfg["rules"]["disk_free_pct"] is not None
        assert cfg["auto_restart"]["restart_after_s"] == 60

    def test_non_dict_ignored(self) -> None:
        assert merged_watchdog_config("junk")["enabled"] is True


class TestServiceChecks:
    def test_dead_service_dispatches_critical(self) -> None:
        wd, alerts = _watchdog({"openai_router": FakeSP(status="crashed(1)")})
        summary = wd.check_once()
        assert summary["services"]["openai_router"]["alive"] is False
        calls = alerts.dispatch.call_args_list
        assert any("停止運作" in str(c.kwargs.get("title", "")) for c in calls)

    def test_alive_but_no_response_counts_hang(self, monkeypatch) -> None:
        import requests as requests_lib

        def timeout(url, timeout=None):
            raise requests_lib.ConnectionError("no response")

        monkeypatch.setattr(watchdog_mod.requests, "get", timeout)
        wd, alerts = _watchdog({"openai_router": FakeSP(pid=1234)})
        state = wd._states.setdefault("openai_router", watchdog_mod._ServiceState())
        state.hang_since = time.time() - 400
        original_load = wd.load_config

        def load():
            cfg = original_load()
            cfg["hang_failures_to_alert"] = 1
            return cfg

        wd.load_config = load
        wd.check_once()
        titles = [str(c.kwargs.get("title", "")) for c in alerts.dispatch.call_args_list]
        assert any("無回應" in t for t in titles)

    def test_recovery_alerts_info(self, monkeypatch) -> None:
        monkeypatch.setattr(
            watchdog_mod.requests,
            "get",
            lambda url, timeout=None: MagicMock(status_code=200),
        )
        wd, alerts = _watchdog({"openai_router": FakeSP(pid=1234)})
        state = wd._states.setdefault("openai_router", watchdog_mod._ServiceState())
        state.consecutive_hang = 5
        wd.check_once()
        calls = alerts.dispatch.call_args_list
        assert any(c.kwargs.get("severity") == watchdog_mod.Severity.INFO for c in calls)

    def test_rss_threshold_triggers_warning(self, monkeypatch) -> None:
        monkeypatch.setattr(
            watchdog_mod.requests,
            "get",
            lambda url, timeout=None: MagicMock(status_code=200),
        )
        monkeypatch.setattr(Watchdog, "_rss_mb", staticmethod(lambda pid: 99999.0))
        wd, alerts = _watchdog({"openai_router": FakeSP(pid=1234)})
        wd.check_once()
        titles = [str(c.kwargs.get("title", "")) for c in alerts.dispatch.call_args_list]
        assert any("記憶體" in t for t in titles)


class TestAutoRestart:
    def _wd_with_restart(self, services, **ar_overrides):
        ar = {"enabled": True, "restart_after_s": 0, "cooldown_s": 0}
        ar.update(ar_overrides)
        wd, alerts = _watchdog(
            services,
            restart_fn=MagicMock(return_value=True),
        )
        original_load = wd.load_config

        def load():
            cfg = original_load()
            cfg["auto_restart"] = {**cfg["auto_restart"], **ar}
            return cfg

        wd.load_config = load
        return wd, alerts

    def test_restarts_dead_service(self) -> None:
        sp = FakeSP(status="crashed(1)")
        wd, alerts = self._wd_with_restart({"openai_router": sp})
        wd._check_services(wd.load_config())
        assert wd._restart_fn.call_count == 1
        wd._restart_fn.assert_called_with("openai_router")

    def test_respects_exclude_list(self) -> None:
        wd, _ = self._wd_with_restart(
            {"openai_router": FakeSP(status="crashed(1)")}, exclude=["openai_router"]
        )
        wd._check_services(wd.load_config())
        assert wd._restart_fn.call_count == 0

    def test_waits_for_restart_after_s(self) -> None:
        wd, _ = self._wd_with_restart(
            {"openai_router": FakeSP(status="crashed(1)")}, restart_after_s=3600
        )
        state = watchdog_mod._ServiceState()
        state.dead_since = time.time()
        wd._states["openai_router"] = state
        wd._check_services(wd.load_config())
        assert wd._restart_fn.call_count == 0

    def test_max_per_day_cap(self) -> None:
        wd, _ = self._wd_with_restart(
            {"openai_router": FakeSP(status="crashed(1)")}, max_per_day=1
        )
        state = watchdog_mod._ServiceState(restart_day=time.strftime("%Y-%m-%d"))
        state.restart_count_today = 1
        wd._states["openai_router"] = state
        result = wd._maybe_autorestart(
            wd.load_config(), state, "openai_router", time.time(), reason="test"
        )
        assert result is False
        assert wd._restart_fn.call_count == 0

    def test_disabled_no_restart(self) -> None:
        wd, _ = self._wd_with_restart({"openai_router": FakeSP(status="crashed(1)")})
        original_load = wd.load_config

        def load():
            cfg = original_load()
            cfg["auto_restart"]["enabled"] = False
            return cfg

        wd.load_config = load
        wd._check_services(wd.load_config())
        assert wd._restart_fn.call_count == 0

    def test_restart_failure_dispatches_critical(self) -> None:
        wd, alerts = _watchdog(
            {"openai_router": FakeSP(status="crashed(1)")},
            restart_fn=MagicMock(side_effect=RuntimeError("boom")),
        )
        cfg = wd.load_config()
        cfg["auto_restart"] = {**cfg["auto_restart"], "enabled": True}
        state = watchdog_mod._ServiceState(dead_since=time.time() - 500)
        wd._states["openai_router"] = state
        ok = wd._maybe_autorestart(cfg, state, "openai_router", time.time(), reason="r")
        assert ok is False
        severities = [
            c.kwargs.get("severity") for c in alerts.dispatch.call_args_list
        ]
        assert watchdog_mod.Severity.CRITICAL in severities


class TestStartStop:
    def test_start_disabled_by_config(self) -> None:
        wd, _ = _watchdog({})
        wd.load_config = lambda: {"enabled": False, "interval_s": 5}
        wd.start()
        assert wd._thread is None

    def test_stop_is_idempotent(self) -> None:
        wd, _ = _watchdog({})
        wd.stop()
        wd.stop()
