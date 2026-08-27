from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import runtime.launcher.supervisor as sup_mod
from runtime.launcher.supervisor import LauncherSupervisor


def _sup(tmp_path, **kw):
    pid_file = tmp_path / "launcher.pid"
    s = LauncherSupervisor(pid_file=str(pid_file), **kw)
    return s, pid_file


class TestLiveness:
    def test_returns_false_when_nothing_running(self, tmp_path) -> None:
        s, _ = _sup(tmp_path)
        assert s._launcher_alive() is False

    def test_pid_file_dead_process(self, tmp_path) -> None:
        s, pid_file = _sup(tmp_path)
        pid_file.write_text("999999", encoding="utf-8")
        with patch.object(s, "_pid_alive", return_value=False):
            assert s._launcher_alive() is False

    def test_pid_file_alive_process(self, tmp_path) -> None:
        s, pid_file = _sup(tmp_path)
        pid_file.write_text("12345", encoding="utf-8")
        with patch.object(s, "_pid_alive", return_value=True):
            assert s._launcher_alive() is True

    def test_sentry_port_alive(self, tmp_path) -> None:
        s, pid_file = _sup(tmp_path)
        sentry = pid_file.with_name("launcher_sentry.json")
        sentry.write_text(json.dumps({"dashboard": 9001}), encoding="utf-8")
        with patch.object(s, "_port_alive", return_value=True):
            assert s._launcher_alive() is True

    def test_sentry_ports_dead(self, tmp_path) -> None:
        s, pid_file = _sup(tmp_path)
        sentry = pid_file.with_name("launcher_sentry.json")
        sentry.write_text(json.dumps({"dashboard": 9001}), encoding="utf-8")
        with patch.object(s, "_port_alive", return_value=False):
            assert s._launcher_alive() is False

    def test_pid_file_too_old_ignored(self, tmp_path, monkeypatch) -> None:
        import os
        import time as _t

        s, pid_file = _sup(tmp_path)
        pid_file.write_text("12345", encoding="utf-8")
        old = _t.time() - 500
        os.utime(pid_file, (old, old))
        with patch.object(s, "_pid_alive", return_value=True):
            assert s._launcher_alive() is False


class TestRestart:
    def test_restart_launches_script(self, tmp_path, monkeypatch) -> None:
        s, _ = _sup(tmp_path, launch_script=r"C:\scripts\dummy.bat")
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 111
        monkeypatch.setattr(sup_mod.subprocess, "Popen", lambda *a, **k: proc)
        s._last_alert = 0
        s._restart_launcher()
        assert s._restart_count == 1
        assert s._launcher_proc is proc

    def test_restart_respects_cooldown(self, tmp_path, monkeypatch) -> None:
        s, _ = _sup(tmp_path)
        s._last_alert = __import__("time").time()
        monkeypatch.setattr(sup_mod.subprocess, "Popen", lambda *a, **k: MagicMock())
        s._restart_launcher()
        assert s._restart_count == 0

    def test_restart_popen_failure_logged(self, tmp_path, monkeypatch) -> None:
        s, _ = _sup(tmp_path, launch_script="boom.bat")
        monkeypatch.setattr(
            sup_mod.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no start")),
        )
        s._last_alert = 0
        s._restart_launcher()
        assert s._restart_count == 1


class TestPorts:
    def test_probe_port(self) -> None:
        import socket

        from runtime.supervisor_util import probe_port

        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.listen(1)
        assert probe_port(port) is True
        srv.close()
        assert probe_port(port) is False


class TestLauncherSentinel:
    def test_launcher_writes_sentinel_pid_and_ports(self, tmp_path) -> None:
        import runtime.launcher.launcher as launcher_mod

        launcher = launcher_mod.Launcher(log_dir="logs")
        launcher._sentinel_dir = tmp_path
        launcher.services["dashboard"] = MagicMock()
        launcher.services["dashboard"].status = "running"
        launcher.services["openai_router"] = MagicMock()
        launcher.services["openai_router"].status = "stopped"

        launcher._write_supervisor_sentinel()

        pid = (tmp_path / "launcher.pid").read_text(encoding="utf-8").strip()
        assert pid == str(__import__("os").getpid())
        sentry = json.loads((tmp_path / "launcher_sentry.json").read_text(encoding="utf-8"))
        assert sentry == {"dashboard": 9001}
