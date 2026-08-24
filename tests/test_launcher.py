from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import runtime.launcher.launcher as launcher_mod
from runtime.launcher.launcher import Launcher, ServiceProcess


@pytest.fixture()
def cfg_file(tmp_path, monkeypatch):
    path = tmp_path / "services.json"
    monkeypatch.setattr(
        Launcher, "_services_config_path", staticmethod(lambda: path)
    )
    return path


def _sp(name: str = "openai_router") -> ServiceProcess:
    return ServiceProcess(name=name, cmd=["python", "-c", "pass"], log_path=None)


class TestDesiredEnabled:
    def test_missing_file_defaults_enabled(self, cfg_file) -> None:
        lc = Launcher(log_dir="logs")
        assert lc.desired_enabled("openai_router") is True

    def test_disabled_entry(self, cfg_file) -> None:
        cfg_file.write_text(json.dumps({"openai_router": {"enabled": False}}), encoding="utf-8")
        lc = Launcher(log_dir="logs")
        assert lc.desired_enabled("openai_router") is False
        assert lc.desired_enabled("dashboard") is True

    def test_mtime_cache_skips_reread(self, cfg_file) -> None:
        import os

        cfg_file.write_text(json.dumps({"openai_router": {"enabled": False}}), encoding="utf-8")
        lc = Launcher(log_dir="logs")
        assert lc.desired_enabled("openai_router") is False
        cfg_file.write_text(json.dumps({}), encoding="utf-8")
        st = cfg_file.stat()
        os.utime(cfg_file, (st.st_atime + 10, st.st_mtime + 10))
        assert lc.desired_enabled("openai_router") is True

    def test_malformed_json_defaults_enabled(self, cfg_file) -> None:
        cfg_file.write_text("{not json", encoding="utf-8")
        lc = Launcher(log_dir="logs")
        assert lc.desired_enabled("openai_router") is True


class TestReconcile:
    def test_stops_running_service_when_disabled(self, cfg_file) -> None:
        cfg_file.write_text(json.dumps({"openai_router": {"enabled": False}}), encoding="utf-8")
        lc = Launcher(log_dir="logs")
        sp = _sp()
        sp.process = MagicMock(poll=lambda: None)
        sp.intentionally_stopped = False
        lc.services["openai_router"] = sp
        lc._reconcile_services()
        assert sp.intentionally_stopped is True

    def test_starts_flagged_service_when_enabled(self, cfg_file) -> None:
        lc = Launcher(log_dir="logs")
        sp = _sp()
        sp.process = MagicMock(poll=lambda: 1)
        sp.intentionally_stopped = True
        lc.services["openai_router"] = sp
        sp.start = MagicMock()
        lc._reconcile_services()
        sp.start.assert_called_once()

    def test_ignores_crashed_service_when_enabled(self, cfg_file) -> None:
        lc = Launcher(log_dir="logs")
        sp = _sp()
        sp.process = MagicMock(poll=lambda: 1)
        sp.intentionally_stopped = False
        lc.services["openai_router"] = sp
        sp.start = MagicMock()
        lc._reconcile_services()
        sp.start.assert_not_called()


class TestStartAllSkipsDisabled:
    def test_skipped_services_marked_intentionally_stopped(self, cfg_file, monkeypatch) -> None:
        cfg_file.write_text(json.dumps({"dashboard": {"enabled": False}}), encoding="utf-8")
        monkeypatch.setattr(Launcher, "_start_watchdog", lambda self: None)
        monkeypatch.setattr(Launcher, "_register_signals", lambda self: None)
        monkeypatch.setattr(ServiceProcess, "start", lambda self: None)
        lc = Launcher(log_dir="logs")
        lc.start_all(names=["dashboard"], daemon=True)
        sp = lc.services["dashboard"]
        assert sp.intentionally_stopped is True
        assert sp.status == "stopped"


class TestIntentionalFlag:
    def test_stop_sets_flag_and_start_clears(self, tmp_path) -> None:
        sp = ServiceProcess(
            name="openai_router", cmd=["python", "-c", "pass"], log_path=tmp_path / "x.log"
        )
        sp.process = MagicMock()
        sp.process.poll.return_value = None
        sp.process.wait.return_value = 0
        sp.stop(timeout=0.1)
        assert sp.intentionally_stopped is True
        sp.start()
        assert sp.intentionally_stopped is False
        sp.stop(0.1)
