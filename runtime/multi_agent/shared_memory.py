from __future__ import annotations

import time
from typing import Any


class SharedMemory:
    def __init__(self) -> None:
        self._agent_scoped: dict[str, dict[str, Any]] = {}
        self._global: dict[str, Any] = {}
        self._broadcast_log: list[dict[str, Any]] = []

    def write(self, agent_id: str, key: str, value: Any) -> None:
        self._agent_scoped.setdefault(agent_id, {})[key] = value

    def read(self, agent_id: str, key: str, default: Any = None) -> Any:
        return self._agent_scoped.get(agent_id, {}).get(key, default)

    def keys(self, agent_id: str) -> list[str]:
        return list(self._agent_scoped.get(agent_id, {}).keys())

    def broadcast(self, key: str, value: Any, source_agent: str = "") -> None:
        self._global[key] = value
        self._broadcast_log.append({
            "key": key,
            "source_agent": source_agent,
            "timestamp": time.time(),
        })

    def read_global(self, key: str, default: Any = None) -> Any:
        return self._global.get(key, default)

    def global_keys(self) -> list[str]:
        return list(self._global.keys())

    def clear_agent(self, agent_id: str) -> None:
        self._agent_scoped.pop(agent_id, None)

    def clear_all(self) -> None:
        self._agent_scoped.clear()
        self._global.clear()
        self._broadcast_log.clear()
