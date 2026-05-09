from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from dotenv import load_dotenv


_ENV_LOADED = False


def _ensure_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _ENV_LOADED = True


SERVICE_DEFS: list[dict[str, Any]] = [
    {
        "name": "control_plane",
        "module": "control_plane.cluster_manager:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_CONTROL_PORT",
        "port_default": 9200,
        "desc": "Cluster control plane",
    },
    {
        "name": "openai_router",
        "module": "router.openai_router:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_ROUTER_PORT",
        "port_default": 8001,
        "desc": "OpenAI-compatible API (chat, responses, embeddings)",
    },
    {
        "name": "anthropic_router",
        "module": "router.anthropic_router:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_ANTHROPIC_PORT",
        "port_default": 8002,
        "desc": "Anthropic-compatible API (messages, tools, vision)",
    },
    {
        "name": "dashboard",
        "module": "dashboard.dashboard_server:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_DASHBOARD_PORT",
        "port_default": 9001,
        "desc": "Web dashboard",
    },
    {
        "name": "metrics",
        "module": "metrics.prometheus_exporter:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_METRICS_PORT",
        "port_default": 9100,
        "desc": "Prometheus metrics exporter",
    },
    {
        "name": "node_agent",
        "module": "node.node_agent:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_NODE_PORT",
        "port_default": 9400,
        "desc": "Node agent (registration + heartbeat)",
    },
    {
        "name": "worker_agent",
        "module": "node.worker_agent:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_WORKER_PORT",
        "port_default": 9300,
        "desc": "Worker agent (health reporting)",
    },
    {
        "name": "task_worker",
        "module": None,
        "host": None,
        "port_env": None,
        "port_default": None,
        "desc": "Async task queue worker",
        "python_module": "ai_queue.task_worker",
    },
]


def _port(svc: dict[str, Any]) -> int:
    key = svc.get("port_env")
    if key:
        val = os.getenv(key)
        if val:
            return int(val)
    return svc["port_default"]


def _cmd_for(svc: dict[str, Any]) -> list[str]:
    pm = svc.get("python_module")
    if pm:
        return [sys.executable, "-m", pm]
    return [
        sys.executable,
        "-m",
        "uvicorn",
        svc["module"],
        "--host",
        svc.get("host", "0.0.0.0"),
        "--port",
        str(_port(svc)),
        "--log-level",
        "info",
    ]


class ServiceProcess:
    def __init__(self, name: str, cmd: list[str], log_path: Path) -> None:
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self.start_time: float | None = None
        self.return_code: int | None = None
        self.log_file: TextIO | None = None
        self.thread: threading.Thread | None = None
        self._stopped = False

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        self.log_file.write(
            f"\n{'=' * 60}\n"
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {self.name}\n"
            f"{'=' * 60}\n"
        )
        self.log_file.flush()

        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        self.start_time = time.time()

        self.thread = threading.Thread(target=self._pipe_logger, daemon=True)
        self.thread.start()

    def _pipe_logger(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in iter(self.process.stdout.readline, b""):
            if self._stopped:
                break
            if self.log_file is not None:
                self.log_file.write(line.decode("utf-8", errors="replace"))
                self.log_file.flush()
        self.return_code = self.process.wait()
        if self.log_file is not None:
            self.log_file.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Exited with code {self.return_code}\n"
            )
            self.log_file.flush()

    def stop(self, timeout: float = 5.0) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self._stopped = True
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    @property
    def status(self) -> str:
        if self.process is None:
            return "stopped"
        rc = self.process.poll()
        if rc is None:
            return "running"
        if rc == 0:
            return "exited"
        return f"crashed({rc})"

    @property
    def uptime(self) -> str:
        if self.start_time is None or self.process is None or self.process.poll() is not None:
            return "-"
        elapsed = int(time.time() - self.start_time)
        if elapsed < 60:
            return f"{elapsed}s"
        return f"{elapsed // 60}m{elapsed % 60}s"


class Launcher:
    def __init__(self, log_dir: str = "logs") -> None:
        _ensure_env()
        self.log_dir = Path(log_dir)
        self.services: dict[str, ServiceProcess] = {}
        self._running = False

    def start_all(self, names: list[str] | None = None, daemon: bool = False) -> None:
        defs = [s for s in SERVICE_DEFS if names is None or s["name"] in names]
        self._running = True

        for svc in defs:
            name = svc["name"]
            cmd = _cmd_for(svc)
            log_path = self.log_dir / f"{name}.log"
            sp = ServiceProcess(name=name, cmd=cmd, log_path=log_path)
            self.services[name] = sp
            sp.start()
            port = svc.get("port_default", "-")
            print(f"  [{name:<16}] started  (port {str(port):<5}  log: {log_path})")

        self._register_signals()
        print()
        self.status()

        if not daemon:
            self._wait_loop()

    def stop_all(self) -> None:
        print()
        for name in list(self.services.keys()):
            sp = self.services[name]
            print(f"  Stopping {name}...", end=" ")
            sp.stop()
            print("done")
        print("  All services stopped.")

    def status(self) -> None:
        running = 0
        for svc in SERVICE_DEFS:
            name = svc["name"]
            sp = self.services.get(name)
            if sp and sp.status == "running":
                running += 1
            port = str(svc.get("port_default", "-"))
            s = sp.status if sp else "not started"
            print(f"  {name:<20} {port:<6} {s:<14} {sp.uptime if sp else '-'}")
        total = len(SERVICE_DEFS)
        print(f"  {running}/{total} running")

    def _register_signals(self) -> None:
        def _handler(sig: int, frame: object) -> None:
            self._running = False
            self.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _wait_loop(self) -> None:
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_all()
