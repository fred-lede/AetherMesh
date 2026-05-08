from __future__ import annotations

import argparse
import sys

from runtime.launcher import Launcher


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m runtime.launcher",
        description="AetherMesh Service Launcher — start/stop/status for all services",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=("start", "stop", "status", "restart"),
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

    args = parser.parse_args()
    launcher = Launcher(log_dir=args.log_dir)

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
