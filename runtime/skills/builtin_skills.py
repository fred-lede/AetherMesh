from __future__ import annotations

import logging
from typing import Any

from runtime.skills.skill_descriptor import SkillDescriptor
from runtime.skills.skill_registry import skill_registry

logger = logging.getLogger("skills.builtin")

_registered = False


async def _web_search_handler(name: str, ctx: dict[str, Any]) -> list[dict[str, str]]:
    params = ctx.get("params", {})
    query = params.get("query", "")
    max_results = int(params.get("max_results", 5))
    timeout_s = int(params.get("timeout_s", 15))
    if not query:
        return []
    from runtime.tools.builtin.web_search import run_web_search
    return run_web_search(query, max_results, timeout_s)


async def _web_fetch_handler(name: str, ctx: dict[str, Any]) -> dict[str, str]:
    params = ctx.get("params", {})
    url = params.get("url", "")
    timeout_s = int(params.get("timeout_s", 15))
    if not url:
        raise ValueError("url is required")
    from runtime.tools.builtin.web_search import run_web_fetch
    return run_web_fetch(url, timeout_s)


async def _code_interpreter_handler(name: str, ctx: dict[str, Any]) -> str:
    params = ctx.get("params", {})
    code = params.get("code", params.get("command", ""))
    if not code:
        return "No code provided"
    from runtime.tools.tool_executor import ToolExecutor
    from runtime.tools.tool_registry import tool_registry
    from runtime.tools.tool_result import ToolCall

    descriptor = tool_registry.resolve("python")
    if not descriptor or not descriptor.handler:
        return "Python tool not available"
    result = await descriptor.handler(ToolCall(name="python", arguments={"code": code}))
    return str(result.output)


async def _shell_handler(name: str, ctx: dict[str, Any]) -> str:
    params = ctx.get("params", {})
    command = params.get("command", "")
    if not command:
        return "No command provided"
    from runtime.tools.tool_executor import ToolExecutor
    from runtime.tools.tool_registry import tool_registry
    from runtime.tools.tool_result import ToolCall

    descriptor = tool_registry.resolve("shell")
    if not descriptor or not descriptor.handler:
        return "Shell tool not available"
    result = await descriptor.handler(ToolCall(name="shell", arguments={"command": command}))
    return str(result.output)


async def _file_operations_handler(name: str, ctx: dict[str, Any]) -> str:
    params = ctx.get("params", {})
    operation = params.get("operation", "")
    path = params.get("path", "")
    content = params.get("content", "")
    if not operation or not path:
        return "operation and path are required"
    from runtime.tools.tool_executor import ToolExecutor
    from runtime.tools.tool_registry import tool_registry
    from runtime.tools.tool_result import ToolCall

    tool_name = {"read": "read", "write": "write", "edit": "edit", "list": "list_directory"}.get(operation)
    if not tool_name:
        return f"Unknown operation: {operation}"
    descriptor = tool_registry.resolve(tool_name)
    if not descriptor or not descriptor.handler:
        return f"File tool {tool_name} not available"
    args: dict[str, Any] = {"path": path}
    if content:
        args["content"] = content
    result = await descriptor.handler(ToolCall(name=tool_name, arguments=args))
    return str(result.output)


def register_builtin_skills() -> None:
    global _registered
    if _registered:
        return

    skills: list[SkillDescriptor] = [
        SkillDescriptor(
            name="web_search",
            description="Search the web for current information",
            capabilities=["web_search", "search"],
            handler=_web_search_handler,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        SkillDescriptor(
            name="web_fetch",
            description="Fetch and read the contents of a URL",
            capabilities=["web_search", "fetch"],
            handler=_web_fetch_handler,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                },
                "required": ["url"],
            },
        ),
        SkillDescriptor(
            name="code_interpreter",
            description="Execute Python code in a sandboxed environment",
            capabilities=["code", "python", "execute"],
            handler=_code_interpreter_handler,
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        ),
        SkillDescriptor(
            name="shell_commands",
            description="Execute shell commands in a sandboxed environment",
            capabilities=["shell", "execute"],
            handler=_shell_handler,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        ),
        SkillDescriptor(
            name="file_operations",
            description="Read, write, edit, and list files on the filesystem",
            capabilities=["filesystem", "file"],
            handler=_file_operations_handler,
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "edit", "list"],
                        "description": "File operation to perform",
                    },
                    "path": {"type": "string", "description": "Path to the file or directory"},
                    "content": {"type": "string", "description": "Content to write (for write/edit)"},
                },
                "required": ["operation", "path"],
            },
        ),
    ]

    for skill in skills:
        skill_registry.register(skill)

    _registered = True
    logger.info("Registered %d built-in skills", len(skills))
