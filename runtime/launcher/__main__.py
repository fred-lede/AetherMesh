from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from runtime.launcher import Launcher


SUPERVISOR_PID_FILE = Path("runtime/launcher/supervisor.pid")


def _supervisor_already_running() -> bool:
    try:
        if not SUPERVISOR_PID_FILE.exists():
            return False
        raw = SUPERVISOR_PID_FILE.read_text(encoding="utf-8").strip()
        pid = int(raw)
        if pid <= 0:
            return False
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return False


def _spawn_detached_supervisor(check_interval: float, python_exe: str) -> int:
    """Spawn a detached background process running the supervisor loop.

    Returns the detached child's PID. The parent returns immediately so that
    Task Scheduler sees the task as completed successfully.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    env = os.environ.copy()
    env["AIIH_SUPERVISOR_FORE"] = "1"
    cmd = [python_exe, "-m", "runtime.launcher", "supervise", "--check-interval", str(check_interval)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        creationflags=creationflags,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    return proc.pid


def run_supervisor(check_interval: float = 30.0) -> None:
    """Blocking entry point for the standalone launcher supervisor.

    When AIIH_SUPERVISOR_FORE is set, run the supervisor loop in the
    foreground (this is the detached worker). Otherwise, spawn a detached
    worker and return so the calling bat/scheduler exits cleanly.
    """
    import logging
    import signal as _signal

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from runtime.launcher.supervisor import LauncherSupervisor

    if not os.environ.get("AIIH_SUPERVISOR_FORE"):
        if _supervisor_already_running():
            print("supervisor already running", flush=True)
            return
        pid = _spawn_detached_supervisor(check_interval, sys.executable)
        SUPERVISOR_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        SUPERVISOR_PID_FILE.write_text(str(pid), encoding="utf-8")
        print(f"supervisor spawned (pid {pid})", flush=True)
        # Detach from any inherited console/handles and exit immediately so the
        # caller (cmd.exe, Task Scheduler) sees the task as completed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    sup = LauncherSupervisor(check_interval_s=check_interval)
    sentry_file = sup._pid_file.with_name("launcher_sentry.json")
    if sup._read_pid() or sup._read_sentry():
        # already running stack — just supervise
        sup.start()
    else:
        # nothing running: ensure the stack comes up first
        sup._restart_launcher()
        sup.start()

    def _halt(_sig, _frame):
        sup.stop()
        try:
            SUPERVISOR_PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _halt)
    _signal.signal(_signal.SIGTERM, _halt)
    if hasattr(_signal, "pause"):
        _signal.pause()
    else:
        # Windows: signal.pause() is unavailable. Block until the supervisor
        # thread's stop event is set by _halt().
        sup._stop_event.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m runtime.launcher",
        description="AetherMesh Service Launcher — start/stop/status for all services",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=("start", "stop", "status", "restart", "supervise"),
        help="Command (default: start)",
    )
    parser.add_argument(
        "services",
        nargs="*",
        metavar="service",
        help="Service name(s) to target (default: all)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Log directory (default: logs/)",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=30.0,
        help="Supervisor liveness check interval in seconds (default: 30)",
    )

    args = parser.parse_args()
    launcher = Launcher(log_dir=args.log_dir)

    if args.command == "supervise":
        run_supervisor(check_interval=args.check_interval)
        return

    if args.command == "start":
        names = args.services if args.services else None
        launcher.start_all(names=names)

    elif args.command == "stop":
        launcher.stop_all()

    elif args.command == "status":
        launcher.status()

    elif args.command == "restart":
        launcher.stop_all()
        names = args.services if args.services else None
        launcher.start_all(names=names)


if __name__ == "__main__":
    main()
