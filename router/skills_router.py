from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from runtime.skills.skill_descriptor import SkillDescriptor
from runtime.skills.skill_registry import skill_registry
import runtime.skills.skill_lifecycle  # noqa: F401 - register lifecycle component

logger = logging.getLogger("skills_router")

skills_router = APIRouter(prefix="/v1/skills", tags=["skills"])


def _descriptor_to_dict(s: SkillDescriptor) -> dict[str, Any]:
    return {
        "name": s.name,
        "description": s.description,
        "type": s.type,
        "parameters": s.parameters or {},
        "capabilities": s.capabilities or [],
        "requires_confirmation": s.requires_confirmation,
        "timeout_s": s.timeout_s,
    }


@skills_router.get("")
def list_skills() -> dict[str, Any]:
    skills = skill_registry.list_skills()
    return {"skills": [_descriptor_to_dict(s) for s in skills], "total": len(skills)}


@skills_router.get("/{name}")
def get_skill(name: str) -> dict[str, Any]:
    skill = skill_registry.get(name)
    if not skill:
        raise HTTPException(status_code=404, detail={"message": f"Skill '{name}' not found", "code": "skill_not_found"})
    return {"skill": _descriptor_to_dict(skill)}


@skills_router.post("/{name}/execute")
async def execute_skill(name: str, payload: dict[str, Any] = {}) -> dict[str, Any]:
    skill = skill_registry.get(name)
    if not skill:
        raise HTTPException(status_code=404, detail={"message": f"Skill '{name}' not found", "code": "skill_not_found"})
    if not skill.handler:
        raise HTTPException(status_code=400, detail={"message": f"Skill '{name}' has no executable handler", "code": "skill_no_handler"})
    params = payload.get("params", payload.get("parameters", {}))
    context = payload.get("context", {})
    result = await skill_registry.execute(name, params=params, context=context)
    return result
