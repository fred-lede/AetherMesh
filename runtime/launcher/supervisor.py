from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("launcher.supervisor")


class LauncherSupervisor:
    """Standalone persistent process that keeps the launcher (and its services)
    alive. Runs independently of the launcher, so if the whole service stack is
    killed it is restarted instead of dying silently."""

    def __init__(
        self,
        check_interval_s: float = 30.0,
        launch_script: str | None = None,
        pid_file: str = "runtime/launcher/launcher.pid",
        log_fallback: str = "logs/launcher_supervisor.log",
    ) -> None:
        self._check_interval = max(5.0, float(check_interval_s))
        self._launch_script = launch_script or str(
            Path(__file__).resolve().parent.parent.parent / "scripts" / "start_all.bat"
        )
        self._pid_file = Path(pid_file)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._launcher_proc: subprocess.Popen | None = None
        self._last_alert = 0.0
        self._restart_count = 0
        self._log = Path(log_fallback)
        self._sentry: dict[str, str] = {}

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="aether-supervisor")
        self._thread.start()
        logger.info("supervisor started (interval=%ss)", self._check_interval)
        self._log_write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] supervisor started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._check_interval):
            try:
                if not self._launcher_alive():
                    self._restart_launcher()
            except Exception:
                logger.exception("supervisor check failed")

    # ── launcher liveness ──────────────────────────────────────────────

    def _launcher_alive(self) -> bool:
        # 1) via pid file
        pid = self._read_pid()
        if pid and self._pid_alive(pid):
            return True
        # 2) via registered sentry ports
        sentry = self._read_sentry()
        if sentry:
            for name, port in sentry.items():
                alive = self._port_alive(port)
                if alive:
                    return True
                self._log_write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] sentry {name} port {port} down"
                )
        # not running
        return False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            import psutil

            return pid > 0 and psutil.pid_exists(pid)
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False

    @staticmethod
    def _port_alive(port: int) -> bool:
        try:
            from runtime.supervisor_util import probe_port

            return probe_port(port)
        except Exception:
            return False

    def _read_pid(self) -> int | None:
        try:
            if self._pid_file.exists():
                raw = self._pid_file.read_text(encoding="utf-8").strip()
                now = time.time()
                return int(raw.strip()) if raw and now - int(os.path.getmtime(self._pid_file)) < 60 else None
        except Exception:
            pass
        return None

    def _read_sentry(self) -> dict[str, str]:
        try:
            p = self._pid_file
            sentry_file = p.with_name("launcher_sentry.json")
            if sentry_file.exists():
                data = json.loads(sentry_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {k: str(v) for k, v in data.items() if isinstance(v, (int, str))}
        except Exception:
            pass
        return self._sentry

    def _restart_launcher(self) -> None:
        now = time.time()
        cooldown = 60
        if now - self._last_alert < cooldown:
            return
        self._last_alert = now
        self._restart_count += 1
        self._log_write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] launcher not alive — restarting "
            f"(attempt #{self._restart_count})\n  cmd: {self._launch_script}"
        )
        logger.warning("launcher not alive — restarting (attempt #%s)", self._restart_count)
        try:
            if self._launcher_proc is not None and self._launcher_proc.poll() is None:
                self._launcher_proc.terminate()
                try:
                    self._launcher_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._launcher_proc.kill()
            self._launcher_proc = None
        except Exception:
            logger.warning("failed to clean prior launcher proc", exc_info=True)

        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self._launcher_proc = subprocess.Popen(
                [self._launch_script],
                shell=True,
                cwd=str(Path(self._launch_script).parent.parent),
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
            self._log_write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] launcher restarted (pid {self._launcher_proc.pid})"
            )
        except Exception as exc:
            logger.exception("launcher restart failed: %s", exc)
            self._log_write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] launcher restart FAILED: {exc}"
            )

    def _log_write(self, line: str) -> None:
        try:
            self._log.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
