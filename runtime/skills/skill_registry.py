from __future__ import annotations

import logging
import time
from typing import Any

from runtime.skills.skill_descriptor import SkillDescriptor

logger = logging.getLogger("skills.registry")


class SkillError(RuntimeError):
    def __init__(self, message: str, skill: str = "") -> None:
        super().__init__(message)
        self.skill = skill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDescriptor] = {}

    def register(self, descriptor: SkillDescriptor) -> None:
        self._skills[descriptor.name] = descriptor
        logger.info("Registered skill: %s (type: %s)", descriptor.name, descriptor.type)

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)
        logger.info("Unregistered skill: %s", name)

    def get(self, name: str) -> SkillDescriptor | None:
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDescriptor]:
        return list(self._skills.values())

    def find_by_capability(self, capability: str) -> list[SkillDescriptor]:
        return [s for s in self._skills.values() if s.capabilities and capability in s.capabilities]

    def find_by_type(self, skill_type: str) -> list[SkillDescriptor]:
        return [s for s in self._skills.values() if s.type == skill_type]

    async def execute(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        skill = self._skills.get(name)
        if not skill:
            raise SkillError(f"Unknown skill: {name}", skill=name)
        if not skill.handler:
            raise SkillError(f"Skill {name} has no handler", skill=name)

        started = time.time()
        try:
            result = skill.handler(name, {"params": params or {}, "context": context or {}})
            if hasattr(result, "__await__"):
                result = await result
            duration_ms = (time.time() - started) * 1000
            logger.debug("Skill %s executed in %.0fms", name, duration_ms)
            return {"success": True, "output": result, "duration_ms": duration_ms}
        except Exception as exc:
            duration_ms = (time.time() - started) * 1000
            logger.error("Skill %s failed after %.0fms: %s", name, duration_ms, exc)
            return {"success": False, "error": str(exc), "duration_ms": duration_ms}

    def clear(self) -> None:
        self._skills.clear()


skill_registry = SkillRegistry()
