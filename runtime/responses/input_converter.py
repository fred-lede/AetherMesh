from __future__ import annotations

from typing import Any

from runtime.responses.response_models import InputItem, InputItemType
from runtime.tools.content_blocks import resolve_file_blocks


def responses_input_to_messages(
    input_value: Any,
    instructions: str = "",
    max_tokens: int | None = None,
    truncation: str = "auto",
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if isinstance(input_value, dict):
        item = _parse_input_item(input_value)
        messages.extend(_input_item_to_messages(item))
        return messages

    if isinstance(input_value, list):
        if truncation == "auto":
            messages = _truncate_input_list(input_value, max_tokens=max_tokens)
        else:
            for item_data in input_value:
                if isinstance(item_data, str):
                    messages.append({"role": "user", "content": item_data})
                elif isinstance(item_data, dict):
                    item = _parse_input_item(item_data)
                    messages.extend(_input_item_to_messages(item))

    return messages


def _truncate_input_list(
    items: list[Any],
    max_tokens: int | None = None,
    min_turns_to_keep: int = 4,
) -> list[dict[str, Any]]:
    """Truncate old messages to stay within a reasonable token budget.

    When `truncation: "auto"`, we drop the oldest user/assistant exchanges while
    preserving the last `min_turns_to_keep` turns + system prompt.
    """
    parsed: list[dict[str, Any]] = []
    for item_data in items:
        if isinstance(item_data, str):
            parsed.append({"role": "user", "content": item_data})
        elif isinstance(item_data, dict):
            item = _parse_input_item(item_data)
            parsed.extend(_input_item_to_messages(item))

    system_msgs = [m for m in parsed if m.get("role") == "system"]
    non_system = [m for m in parsed if m.get("role") != "system"]

    if len(non_system) <= 2 * min_turns_to_keep:
        return parsed

    keep_from = len(non_system) - 2 * min_turns_to_keep
    if keep_from < 0:
        keep_from = 0
    return system_msgs + non_system[keep_from:]


def _parse_input_item(item: dict[str, Any]) -> InputItem:
    if _is_bare_content_part(item):
        content_part = dict(item)
        if "text" in content_part and "type" not in content_part:
            content_part["type"] = "text"
        return InputItem(
            id=str(item.get("id", f"input_{id(item)}")),
            type=InputItemType.MESSAGE,
            role=_normalize_message_role(item.get("role", "user")),
            content=[content_part],
        )

    item_type_str = str(item.get("type", "message"))
    try:
        item_type = InputItemType(item_type_str)
    except ValueError:
        item_type = InputItemType.MESSAGE

    result = InputItem(
        id=str(item.get("id", f"input_{id(item)}")),
        type=item_type,
        role=_normalize_message_role(item.get("role", "user")),
    )

    if item_type == InputItemType.MESSAGE:
        content = item.get("content", "")
        if isinstance(content, str):
            result.content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            result.content = content
        elif "text" in item:
            result.content = [{"type": "text", "text": str(item.get("text", ""))}]
        else:
            result.content = [{"type": "text", "text": str(content)}]

    elif item_type == InputItemType.TOOL_CALL:
        result.tool_call_id = str(item.get("tool_call_id", ""))
        result.tool_name = str(item.get("tool_name", ""))
        result.arguments = _stringify_arguments(item.get("arguments", "{}"))

    elif item_type in (InputItemType.TOOL_RESULT, InputItemType.FUNCTION_CALL_OUTPUT):
        result.tool_call_id = str(item.get("tool_call_id", item.get("call_id", "")))
        result.output = str(item.get("output", ""))
        result.is_error = bool(item.get("is_error", False))

    elif item_type == InputItemType.FILE:
        result.file_id = str(item.get("file_id", ""))
        result.filename = str(item.get("filename", ""))

    return result


def _input_item_to_messages(item: InputItem) -> list[dict[str, Any]]:
    if item.type == InputItemType.MESSAGE:
        content = _resolve_file_content_parts(item.content)
        content = _content_list_to_various(content)
        return [{"role": item.role, "content": content}]

    if item.type == InputItemType.TOOL_CALL:
        return [{
            "role": "assistant",
            "tool_calls": [{
                "id": item.tool_call_id,
                "type": "function",
                "function": {
                    "name": item.tool_name,
                    "arguments": item.arguments,
                },
            }],
        }]

    if item.type in (InputItemType.TOOL_RESULT, InputItemType.FUNCTION_CALL_OUTPUT):
        return [{
            "role": "tool",
            "tool_call_id": item.tool_call_id,
            "content": item.output,
        }]

    if item.type == InputItemType.FILE:
        resolved = resolve_file_blocks([item.file_id], "generic")
        if resolved:
            return [{"role": "user", "content": resolved}]
        return [{"role": "user", "content": f"[file: {item.filename or item.file_id}]"}]

    return [{"role": "user", "content": str(item)}]


def _normalize_message_role(role: Any) -> str:
    raw = str(role or "user").strip().lower()
    if raw == "developer":
        return "system"
    if raw in {"system", "user", "assistant", "tool"}:
        return raw
    return "user"


def _resolve_file_content_parts(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    resolved: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            resolved.append(part)
            continue
        if part.get("type") in ("file", "input_file") and "file_id" in part:
            blocks = resolve_file_blocks([part["file_id"]], "generic")
            if blocks:
                resolved.extend(blocks)
            else:
                name = part.get("filename") or part.get("name") or part["file_id"]
                resolved.append({"type": "text", "text": f"[file: {name}]"})
        else:
            resolved.append(part)
    return resolved


def _content_list_to_various(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    if len(content) == 1 and content[0].get("type") in {"text", "input_text", "output_text"}:
        return str(content[0].get("text", ""))
    return content


def _is_bare_content_part(item: dict[str, Any]) -> bool:
    item_type = str(item.get("type", "")).lower()
    if item_type in {
        "text",
        "input_text",
        "output_text",
        "image_url",
        "input_image",
        "image",
        "input_file",
        "file",
        "input_audio",
        "audio",
    }:
        return "content" not in item
    return "text" in item and "content" not in item and "role" not in item


def _stringify_arguments(args: Any) -> str:
    if isinstance(args, str):
        return args
    import json
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))


def chat_completion_to_response_messages(
    completion: dict[str, Any],
    model: str,
    previous_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result = list(previous_messages or [])
    choices = completion.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        result.append(message)
    return result
