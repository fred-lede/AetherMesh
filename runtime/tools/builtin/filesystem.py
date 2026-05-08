from __future__ import annotations

import logging
from typing import Any

from runtime.security.tool_sandbox import tool_sandbox
from runtime.tools.tool_registry import ToolDescriptor, ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("builtin.filesystem")


def _read_handler(call: ToolCall) -> ToolResult:
    return tool_sandbox.read_file(call)


def _write_handler(call: ToolCall) -> ToolResult:
    return tool_sandbox.write_file(call)


READ_DESCRIPTOR = ToolDescriptor(
    name="read_file",
    description="Read the contents of a file from the local filesystem. Returns text content (UTF-8).",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
        },
        "required": ["path"],
    },
    handler=_read_handler,
    source="builtin",
    requires_confirmation=False,
    timeout_s=30,
)

WRITE_DESCRIPTOR = ToolDescriptor(
    name="write_file",
    description="Write content to a file on the local filesystem. Creates parent directories if needed.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file (UTF-8)",
            },
        },
        "required": ["path", "content"],
    },
    handler=_write_handler,
    source="builtin",
    requires_confirmation=True,
    timeout_s=30,
)


def register(registry: ToolRegistry | None = None) -> list[ToolDescriptor]:
    reg = registry or default_registry
    reg.register(READ_DESCRIPTOR)
    reg.register(WRITE_DESCRIPTOR)
    logger.info("Registered builtin tools: read_file, write_file")
    return [READ_DESCRIPTOR, WRITE_DESCRIPTOR]


def available() -> bool:
    return True
