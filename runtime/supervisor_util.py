from __future__ import annotations

import socket
import time
from urllib.request import urlopen


def probe_port(port: int, host: str = "127.0.0.1", timeout_s: float = 3.0) -> bool:
    """Return True if something is accepting TCP connections on the given port."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def probe_http(port: int, path: str = "/health", timeout_s: float = 5.0) -> bool:
    """Return True if an HTTP GET on the given port returns a non-5xx response."""
    try:
        with urlopen(f"http://127.0.0.1:{int(port)}{path}", timeout=timeout_s) as resp:
            return resp.status < 500
    except Exception:
        return False
