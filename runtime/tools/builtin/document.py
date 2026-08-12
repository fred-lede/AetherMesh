from __future__ import annotations

import logging
import time
from typing import Any

from config.settings import settings
from runtime.tools.tool_registry import ToolDescriptor, ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("builtin.document")

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the document to convert (pdf, docx, pptx, xlsx, or image)",
        },
        "output_dir": {
            "type": "string",
            "description": "Optional directory for the extracted markdown (defaults to <file>/mineru_out)",
            "default": "",
        },
    },
    "required": ["path"],
}


def _document_handler(call: ToolCall) -> ToolResult:
    args = call.arguments
    path = str(args.get("path") or "").strip()
    out_dir = str(args.get("output_dir") or "").strip()
    if not path:
        return ToolResult(call=call, output="Missing required argument 'path'", is_error=True)
    if not settings.mineru_enabled:
        return ToolResult(
            call=call,
            output="MinerU is disabled. Set AIIH_MINERU_ENABLED=true to enable document extraction.",
            is_error=True,
        )
    from runtime.documents.mineru_converter import MinerUError, convert_document

    started = time.time()
    try:
        result = convert_document(path, out_dir=out_dir or None)
        duration_ms = int((time.time() - started) * 1000)
        markdown = result["markdown"]
        summary = (
            f"Converted {result['source']} -> {result['output_path']} "
            f"({result['chars']} chars, {duration_ms}ms)\n\n"
        )
        return ToolResult(
            call=call,
            output=summary + markdown[:20000],
            duration_ms=duration_ms,
            metadata={"output_path": result["output_path"]},
        )
    except MinerUError as exc:
        return ToolResult(call=call, output=str(exc), is_error=True)
    except Exception as exc:
        return ToolResult(
            call=call,
            output=f"MinerU conversion failed: {type(exc).__name__}: {exc}",
            is_error=True,
        )


DOCUMENT_DESCRIPTOR = ToolDescriptor(
    name="document_to_markdown",
    description=(
        "Convert a PDF, DOCX, PPTX, XLSX, or image file into Markdown text using MinerU. "
        "Use this to extract the content of documents for the agent to read."
    ),
    input_schema=INPUT_SCHEMA,
    handler=_document_handler,
    source="builtin",
    requires_confirmation=True,
    timeout_s=900,
)


def register(registry: ToolRegistry | None = None) -> ToolDescriptor:
    reg = registry or default_registry
    reg.register(DOCUMENT_DESCRIPTOR)
    logger.info("Registered builtin tool: document_to_markdown")
    return DOCUMENT_DESCRIPTOR


def available() -> bool:
    if not settings.mineru_enabled:
        return False
    from runtime.documents.mineru_converter import mineru_available

    return mineru_available()
