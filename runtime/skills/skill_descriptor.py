from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union


SkillHandler = Callable[[str, dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass
class SkillDescriptor:
    name: str
    description: str
    type: str = "builtin"
    handler: SkillHandler | None = None
    parameters: dict[str, Any] | None = None
    capabilities: list[str] | None = None
    requires_confirmation: bool = False
    timeout_s: int = 30
