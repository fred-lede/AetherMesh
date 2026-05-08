from __future__ import annotations

"""Built-in tool registration. Importing this module registers all builtin tools."""

import logging

from runtime.tools.tool_registry import tool_registry

logger = logging.getLogger("builtin")

_registered = False


def register_all() -> None:
    global _registered
    if _registered:
        return

    from runtime.tools.builtin import filesystem, http_request, python, shell

    shell.register(tool_registry)
    filesystem.register(tool_registry)
    python.register(tool_registry)
    http_request.register(tool_registry)

    _registered = True
    logger.info(
        "Registered %d builtin tools",
        len(tool_registry.list_tools()),
    )


register_all()
