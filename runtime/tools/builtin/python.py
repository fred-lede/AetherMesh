from __future__ import annotations

import logging
import shutil
from typing import Any

from runtime.security.tool_sandbox import tool_sandbox
from runtime.tools.tool_registry import ToolDescriptor, ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("builtin.python")

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python code to execute",
        },
        "timeout_s": {
            "type": "integer",
            "description": "Timeout in seconds",
            "default": 30,
        },
    },
    "required": ["code"],
}


def _python_handler(call: ToolCall) -> ToolResult:
    return tool_sandbox.run_python(call)


PYTHON_DESCRIPTOR = ToolDescriptor(
    name="python",
    description="Execute Python code in a subprocess. Useful for data analysis, text processing, scripting.",
    input_schema=INPUT_SCHEMA,
    handler=_python_handler,
    source="builtin",
    requires_confirmation=True,
    timeout_s=60,
)


def register(registry: ToolRegistry | None = None) -> ToolDescriptor:
    reg = registry or default_registry
    reg.register(PYTHON_DESCRIPTOR)
    logger.info("Registered builtin tool: python")
    return PYTHON_DESCRIPTOR


def available() -> bool:
    return bool(shutil.which("python3") or shutil.which("python"))
