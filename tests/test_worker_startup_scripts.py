from __future__ import annotations

import re
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _launcher_services(script: str) -> list[str]:
    match = re.search(r"-m\s+runtime\.launcher\s+start\s+([^\r\n\"']+)", script)
    assert match is not None, "launcher start command not found"
    return match.group(1).split()


def test_start_worker_bat_starts_only_worker_services() -> None:
    script = (_SCRIPTS / "start-worker.bat").read_text(encoding="utf-8")
    assert _launcher_services(script) == ["%WORKER_SERVICES%"]
    assert 'set "WORKER_SERVICES=node_agent worker_agent asr_worker"' in script


def test_start_worker_bat_adds_rerank_only_when_llama_server_is_configured() -> None:
    script = (_SCRIPTS / "start-worker.bat").read_text(encoding="utf-8")
    assert 'set "WORKER_SERVICES=node_agent worker_agent asr_worker"' in script
    assert 'set "WORKER_SERVICES=%WORKER_SERVICES% rerank"' in script
    assert 'findstr /i /c:"AIIH_LLAMA_SERVER" ".env"' in script


def test_register_worker_task_bat_disables_host_and_installs_worker() -> None:
    script = (_SCRIPTS / "register-worker-task.bat").read_text(encoding="utf-8")
    assert 'schtasks /Change /TN "%TASK_HOST%" /DISABLE' in script
    assert '/TN "%TASK_WORKER%"' in script
    assert "start-worker.bat" in script
