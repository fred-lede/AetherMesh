from __future__ import annotations

import argparse
import sys

from runtime.launcher import Launcher


def run_supervisor(check_interval: float = 30.0) -> None:
    """Blocking entry point for the standalone launcher supervisor."""
    import logging
    import signal as _signal

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from runtime.launcher.supervisor import LauncherSupervisor

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
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _halt)
    _signal.signal(_signal.SIGTERM, _halt)
    _signal.pause()


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
