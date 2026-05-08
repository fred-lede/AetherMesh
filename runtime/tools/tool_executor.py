from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from runtime.tools.tool_registry import ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("tool_executor")


class ToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or default_registry

    def execute(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        descriptor = self._registry.resolve(call.name)
        if not descriptor:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' not found in registry",
                is_error=True,
            )
        if not descriptor.handler:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' has no handler registered",
                is_error=True,
            )

        start = time.monotonic()
        try:
            result = descriptor.handler(call)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import asyncio.tasks

                    result = asyncio.run_coroutine_threadsafe(result, loop).result(timeout=timeout_s)
                else:
                    result = asyncio.run(result)
            duration = (time.monotonic() - start) * 1000
            if isinstance(result, ToolResult):
                result.duration_ms = duration
                return result
            return ToolResult(call=call, output=result, duration_ms=duration)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.exception("Tool %s execution failed", call.name)
            return ToolResult(call=call, output=str(e), is_error=True, duration_ms=duration)

    async def execute_async(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        descriptor = self._registry.resolve(call.name)
        if not descriptor:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' not found in registry",
                is_error=True,
            )
        if not descriptor.handler:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' has no handler registered",
                is_error=True,
            )

        start = time.monotonic()
        try:
            result = descriptor.handler(call)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=timeout_s)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(descriptor.handler, call), timeout=timeout_s
                )
            duration = (time.monotonic() - start) * 1000
            if isinstance(result, ToolResult):
                result.duration_ms = duration
                return result
            return ToolResult(call=call, output=result, duration_ms=duration)
        except asyncio.TimeoutError:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' timed out after {timeout_s}s",
                is_error=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.exception("Tool %s async execution failed", call.name)
            return ToolResult(call=call, output=str(e), is_error=True, duration_ms=duration)


tool_executor = ToolExecutor()
