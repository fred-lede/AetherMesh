from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".env.example"

TARGETS: dict[Path, dict[str, str]] = {
    ROOT / "profiles" / "control-plane" / ".env.example": {
        "AIIH_NODE_ID": "node-control-01",
        "AIIH_NODE_IP": "",
    },
    ROOT / "profiles" / "worker-node" / ".env.example": {
        "AIIH_CONTROL_URL": "http://<CONTROL_PLANE_HOST_IP>:9200",
        "AIIH_ROUTER_URL": "http://<CONTROL_PLANE_HOST_IP>:8001",
        "AIIH_METRICS_URL": "http://<CONTROL_PLANE_HOST_IP>:9100",
        "AIIH_REDIS_URL": "redis://<REDIS_HOST_IP>:6379/0",
        "AIIH_NODE_ID": "node-worker-01",
        "AIIH_NODE_IP": "<WORKER_LAN_IP>",
    },
}


def apply_overrides(lines: list[str], overrides: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            rendered.append(line)
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in overrides:
            rendered.append(f"{key}={overrides[key]}")
        else:
            rendered.append(f"{key}={value}")
    return rendered


def main() -> None:
    source_lines = SOURCE.read_text(encoding="utf-8").splitlines()
    for target, overrides in TARGETS.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "# GENERATED FILE: do not edit directly.",
            "# Source: /.env.example",
            "# Update source then run: python scripts/sync_env_examples.py",
            "",
        ]
        target_lines = header + apply_overrides(source_lines, overrides)
        target.write_text("\n".join(target_lines) + "\n", encoding="utf-8", newline="\n")
        print(f"synced: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
