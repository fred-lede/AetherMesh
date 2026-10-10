from __future__ import annotations

from pathlib import Path

import pytest

from runtime.health import watchdog as wd
from runtime.launcher import launcher


class _FakeCompleted:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


def _watchdog(**cfg) -> wd.Watchdog:
    section = {"enabled": True, "hang_dump_cooldown_s": cfg.pop("hang_dump_cooldown_s", 0), **cfg}
    return wd.Watchdog(get_services=lambda: {}, config_section=section)


def test_hang_dump_writes_stack_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    obj = _watchdog(hang_dump_cooldown_s=0)
    monkeypatch.setattr(wd.shutil, "which", lambda name: "py-spy")
    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="Thread 1\n  foo()"))
    monkeypatch.chdir(tmp_path)
    state = wd._ServiceState()
    path = obj._maybe_dump_hang(obj.load_config(), "openai_router", 1234, state, 1000.0)
    assert path is not None
    assert Path(path).exists()
    assert "foo()" in Path(path).read_text(encoding="utf-8")
    assert state.last_dump_ts == 1000.0


def test_hang_dump_skipped_without_pyspy(monkeypatch: pytest.MonkeyPatch):
    obj = _watchdog(hang_dump_cooldown_s=0)
    monkeypatch.setattr(wd, "_find_py_spy", lambda: None)
    assert obj._maybe_dump_hang(obj.load_config(), "x", 1, wd._ServiceState(), 1.0) is None


def test_hang_dump_respects_cooldown(monkeypatch: pytest.MonkeyPatch):
    obj = _watchdog(hang_dump_cooldown_s=600)
    monkeypatch.setattr(wd.shutil, "which", lambda name: "py-spy")
    state = wd._ServiceState()
    state.last_dump_ts = 1000.0
    assert obj._maybe_dump_hang(obj.load_config(), "x", 1, state, 1100.0) is None


def test_hang_dump_skipped_without_pid(monkeypatch: pytest.MonkeyPatch):
    obj = _watchdog(hang_dump_cooldown_s=0)
    monkeypatch.setattr(wd.shutil, "which", lambda name: "py-spy")
    assert obj._maybe_dump_hang(obj.load_config(), "x", None, wd._ServiceState(), 1.0) is None


def test_launcher_uses_uvicorn_log_config():
    svc = next(s for s in launcher.SERVICE_DEFS if s.get("name") == "openai_router")
    cmd = launcher._cmd_for(svc)
    assert "--log-config" in cmd
    assert "uvicorn_log_config.json" in " ".join(cmd)
