from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIIH_SUPERVISOR_FORE", raising=False)


def test_spawn_detached_returns_child_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    from runtime.launcher import __main__ as launcher_main

    fake_proc = MagicMock()
    fake_proc.pid = 9999
    with patch.object(launcher_main.subprocess, "Popen", return_value=fake_proc) as popen:
        pid = launcher_main._spawn_detached_supervisor(15.0, sys.executable)
    assert pid == 9999
    popen.assert_called_once()
    cmd = popen.call_args[0][0]
    assert cmd[0] == sys.executable
    assert "supervise" in cmd


def test_supervisor_already_running_true_when_pid_alive(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from runtime.launcher import __main__ as launcher_main

    pid_file = tmp_path / "supervisor.pid"
    pid_file.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(launcher_main, "SUPERVISOR_PID_FILE", pid_file)

    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = True
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert launcher_main._supervisor_already_running() is True


def test_supervisor_already_running_false_when_pid_dead(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from runtime.launcher import __main__ as launcher_main

    pid_file = tmp_path / "supervisor.pid"
    pid_file.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(launcher_main, "SUPERVISOR_PID_FILE", pid_file)

    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = False
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert launcher_main._supervisor_already_running() is False


def test_supervisor_already_running_false_when_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from runtime.launcher import __main__ as launcher_main

    monkeypatch.setattr(launcher_main, "SUPERVISOR_PID_FILE", tmp_path / "nope.pid")

    assert launcher_main._supervisor_already_running() is False
