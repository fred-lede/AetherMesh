from __future__ import annotations

from pathlib import Path

import pytest

from runtime.health import watchdog as wd


class _FakeSp:
    """Fake service process as the launcher exposes to the watchdog."""

    def __init__(self, log_path: Path | None = None, rc: int | None = 1) -> None:
        self.intentionally_stopped = False
        self.status = "running" if rc is None else "crashed(1)"
        self.log_path = log_path
        self._rc = rc

    @property
    def process(self):
        proc = self

        def poll() -> int | None:
            return self._rc

        proc.poll = poll
        proc.pid = 4242 if self._rc is not None else 4242
        return proc


def test_dead_service_alert_includes_log_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log = tmp_path / "openai_router.log"
    log.write_text("line 1\nline 2\nTraceback (most recent call last):\nRuntimeError: boom\n", encoding="utf-8")
    sp = _FakeSp(log_path=log, rc=1)

    obj = wd.Watchdog(
        get_services=lambda: {"openai_router": sp},
        config_section={"enabled": True},
    )
    captured: list[dict] = []

    class _Recorder:
        def dispatch(self, **kw):
            captured.append(kw)

    obj._alerts = _Recorder()  # type: ignore[assignment]
    results = obj._check_services(obj.load_config())
    assert len(results) == 1
    assert "RuntimeError: boom" in results[0].detail
    # _evaluate_service should have fired a dead alert that includes the tail
    obj._evaluate_service(obj.load_config(), obj._states["openai_router"], results[0], 1_000_000.0)
    dead_alerts = [c for c in captured if c["rule_key"].startswith("dead:")]
    assert dead_alerts, f"no dead alert fired; captured={[c['rule_key'] for c in captured]}"
    assert "RuntimeError: boom" in dead_alerts[0]["message"]
