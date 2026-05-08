from __future__ import annotations

import logging
import shutil
from typing import Any

from runtime.security.tool_sandbox import tool_sandbox
from runtime.tools.tool_registry import ToolDescriptor, ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("builtin.shell")

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute",
        },
        "timeout_s": {
            "type": "integer",
            "description": "Timeout in seconds",
            "default": 30,
        },
    },
    "required": ["command"],
}


def _shell_handler(call: ToolCall) -> ToolResult:
    return tool_sandbox.run_shell(call)


SHELL_DESCRIPTOR = ToolDescriptor(
    name="shell",
    description="Execute a shell command on the local system. Use for file operations, git, system admin, etc.",
    input_schema=INPUT_SCHEMA,
    handler=_shell_handler,
    source="builtin",
    requires_confirmation=True,
    timeout_s=60,
)


def register(registry: ToolRegistry | None = None) -> ToolDescriptor:
    reg = registry or default_registry
    reg.register(SHELL_DESCRIPTOR)
    logger.info("Registered builtin tool: shell")
    return SHELL_DESCRIPTOR


def available() -> bool:
    return bool(shutil.which("sh") or shutil.which("bash") or shutil.which("cmd"))
