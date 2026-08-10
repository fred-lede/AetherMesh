from __future__ import annotations

import json
from typing import Any


class StructuredOutputError(ValueError):
    pass


def response_format_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return None
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            schema = json_schema.get("schema", json_schema)
            if isinstance(schema, dict):
                return {"kind": "schema", "schema": schema, "name": str(json_schema.get("name", "output"))}
        return {"kind": "schema", "schema": {}, "name": "output"}
    if fmt_type == "json_object":
        return {"kind": "json_object", "schema": {}, "name": "json_object"}
    return None


def extract_json_content(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start, end = _find_json_bounds(stripped)
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError("response is not valid JSON")


def _find_json_bounds(text: str) -> tuple[int, int]:
    open_char = None
    open_idx = -1
    for i, ch in enumerate(text):
        if ch in "{[":  # noqa: SIM114
            open_char = ch
            open_idx = i
            break
        if ch in "}]":
            break
    if open_char is None:
        return -1, -1
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return open_idx, i + 1
    return -1, -1


def validate_json(value: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(schema, dict):
        return True, ""
    return _validate_value(value, schema, "value")


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> tuple[bool, str]:
    if not isinstance(schema, dict):
        return True, ""
    schema_type = schema.get("type")

    if "enum" in schema and value not in schema["enum"]:
        return False, f"{path}: value not in enum"

    if schema_type == "object":
        if not isinstance(value, dict):
            return False, f"{path}: expected object, got {type(value).__name__}"
        for required in schema.get("required", []):
            if required not in value:
                return False, f"{path}: missing required property '{required}'"
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child in value.items():
                if key in properties:
                    ok, err = _validate_value(child, properties[key], f"{path}.{key}")
                    if not ok:
                        return False, err
        return True, ""

    if schema_type == "array":
        if not isinstance(value, list):
            return False, f"{path}: expected array, got {type(value).__name__}"
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False, f"{path}: fewer than minItems {schema['minItems']}"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False, f"{path}: more than maxItems {schema['maxItems']}"
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                ok, err = _validate_value(item, items, f"{path}[{index}]")
                if not ok:
                    return False, err
        return True, ""

    if schema_type == "string":
        if not isinstance(value, str):
            return False, f"{path}: expected string, got {type(value).__name__}"
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False, f"{path}: shorter than minLength {schema['minLength']}"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False, f"{path}: longer than maxLength {schema['maxLength']}"
        return True, ""

    if schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"{path}: expected number, got {type(value).__name__}"
        if "minimum" in schema and value < schema["minimum"]:
            return False, f"{path}: less than minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return False, f"{path}: greater than maximum {schema['maximum']}"
        return True, ""

    if schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f"{path}: expected integer, got {type(value).__name__}"
        if "minimum" in schema and value < schema["minimum"]:
            return False, f"{path}: less than minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return False, f"{path}: greater than maximum {schema['maximum']}"
        return True, ""

    if schema_type == "boolean":
        if not isinstance(value, bool):
            return False, f"{path}: expected boolean, got {type(value).__name__}"
        return True, ""

    if schema_type == "null":
        if value is not None:
            return False, f"{path}: expected null, got {type(value).__name__}"
        return True, ""

    return True, ""


def build_repair_messages(
    messages: list[dict[str, Any]],
    schema_info: dict[str, Any],
    invalid_content: str,
    error: str,
) -> list[dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            "Your previous response was not valid JSON matching the required schema.\n"
            "Reply with ONLY valid JSON conforming to this JSON Schema:\n"
            f"{json.dumps(schema_info.get('schema', {}), ensure_ascii=False)}\n"
        ),
    }
    correction = {
        "role": "user",
        "content": (
            f"Invalid output:\n{invalid_content[:2000]}\n\n"
            f"Validation error: {error}\n\n"
            "Please output only the corrected JSON object."
        ),
    }
    cleaned = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    return [system, *cleaned, correction]


def apply_structured_output(
    payload: dict[str, Any],
    response: dict[str, Any],
    adapter: Any,
    max_retries: int = 2,
) -> dict[str, Any]:
    schema_info = response_format_schema(payload)
    if schema_info is None:
        return response
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return response
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        return response

    provider_payload = {k: v for k, v in payload.items() if k != "response_format"}
    last_error = "invalid JSON"
    for _ in range(max_retries + 1):
        try:
            parsed = extract_json_content(content)
        except ValueError:
            last_error = "response is not valid JSON"
            parsed = None
        if parsed is not None:
            if schema_info["kind"] == "json_object":
                valid = isinstance(parsed, dict)
                last_error = "response is not a JSON object" if not valid else ""
            else:
                valid, last_error = validate_json(parsed, schema_info.get("schema", {}))
            if valid:
                message["content"] = json.dumps(parsed, ensure_ascii=False)
                return response
        repair_messages = build_repair_messages(
            payload.get("messages") or [],
            schema_info,
            content,
            last_error,
        )
        repair_payload = {**provider_payload, "messages": repair_messages}
        try:
            repaired = adapter.chat(repair_payload)
        except Exception as exc:
            raise StructuredOutputError(f"structured output repair failed: {exc}") from exc
        repaired_choices = repaired.get("choices") or []
        if not repaired_choices or not isinstance(repaired_choices[0], dict):
            break
        repaired_message = repaired_choices[0].get("message") or {}
        content = repaired_message.get("content")
        if not isinstance(content, str):
            break

    raise StructuredOutputError(
        f"assistant response does not match required JSON schema after {max_retries} repairs: {last_error}"
    )
