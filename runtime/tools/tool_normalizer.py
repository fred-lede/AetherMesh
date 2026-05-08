from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedToolCall:
    id: str
    name: str
    input: dict[str, Any]
    source: str = "unknown"

    @property
    def arguments_text(self) -> str:
        return json.dumps(self.input, ensure_ascii=False, separators=(",", ":"))

    def to_openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_text,
            },
        }

    def to_anthropic_content_block(self) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.id if self.id.startswith("toolu_") else f"toolu_{uuid.uuid4().hex[:24]}",
            "name": self.name,
            "input": self.input,
        }


class ToolCallNormalizer:
    def from_openai_tool_calls(self, tool_calls: Any) -> list[NormalizedToolCall]:
        if not isinstance(tool_calls, list):
            return []

        normalized: list[NormalizedToolCall] = []
        for index, item in enumerate(tool_calls):
            if not isinstance(item, dict):
                continue
            normalized.extend(self._normalize_object(item, source="openai", index=index))
        return self.dedupe(normalized)

    def from_text(self, content: str) -> list[NormalizedToolCall]:
        stripped = str(content or "").strip()
        if not stripped:
            return []

        calls: list[NormalizedToolCall] = []
        parseable_text = self._strip_thinking_blocks(stripped)
        calls.extend(self._parse_markup_calls(parseable_text, source="text"))
        for candidate in self._text_candidates(parseable_text):
            parsed = self._parse_text(candidate)
            for index, item in enumerate(self._coerce_to_objects(parsed)):
                calls.extend(self._normalize_object(item, source="text", index=index))
        return self.dedupe(calls)

    def from_content_with_thinking(
        self,
        content: str,
        thinking: str | None = None,
        *,
        allowed_tool_names: set[str] | None = None,
    ) -> list[NormalizedToolCall]:
        """Parse tool calls from visible content, then cautiously from thinking.

        Some local models put tool-call markup in their reasoning stream. Treat
        that as an executable call only when there is no visible assistant text
        and the tool name was declared by the request.
        """
        visible_calls = self.from_text(content)
        if visible_calls:
            return visible_calls

        if str(content or "").strip() or not thinking:
            return []

        thinking_calls = self.from_text(thinking)
        if allowed_tool_names is None:
            return thinking_calls
        return [call for call in thinking_calls if call.name in allowed_tool_names]

    def to_openai_tool_calls(self, content: str) -> list[dict[str, Any]]:
        return [call.to_openai_tool_call() for call in self.from_text(content)]

    def looks_like_fragment(self, content: str) -> bool:
        text = str(content or "")
        lowered = text.lower()
        if not text.strip():
            return False
        if "claude responded:" in lowered or "tool responded:" in lowered:
            return True
        if "<tool_use" in lowered or "</tool_use>" in lowered:
            return True
        if any(
            token in lowered
            for token in (
                "<tool_call",
                "</tool_call>",
                "<function=",
                "[tool call:",
                "[calling tool:",
                "call:",
                "websearch",
                "askuserquestion",
                "mcp__",
                "tool_calls",
            )
        ):
            return True
        return "{" in text and (
            "tool_use" in lowered
            or "tooluse" in lowered
            or "tool_name" in lowered
            or ("type" in lowered and "tool" in lowered)
        )

    def dedupe(self, calls: list[NormalizedToolCall]) -> list[NormalizedToolCall]:
        seen: set[tuple[str, str]] = set()
        deduped: list[NormalizedToolCall] = []
        for call in calls:
            if not call.name:
                continue
            key = (call.name, call.arguments_text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(call)
        return deduped

    def _normalize_object(self, item: dict[str, Any], *, source: str, index: int) -> list[NormalizedToolCall]:
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            calls: list[NormalizedToolCall] = []
            for child_index, child in enumerate(tool_calls):
                if isinstance(child, dict):
                    calls.extend(self._normalize_object(child, source=source, index=child_index))
            return calls

        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", {})
            if name:
                return [
                    NormalizedToolCall(
                        id=self._normalize_id(item.get("id"), index),
                        name=str(name),
                        input=self._parse_input(arguments),
                        source=source,
                    )
                ]

        name = item.get("name") or item.get("tool_name") or item.get("toolName")
        input_value = item.get("input", item.get("tool_args", item.get("arguments", {})))
        item_type = str(item.get("type", "")).replace("_", "").lower()
        if name and (item_type in ("", "function", "tooluse") or "input" in item or "tool_args" in item):
            return [
                NormalizedToolCall(
                    id=self._normalize_id(item.get("id"), index),
                    name=str(name),
                    input=self._parse_input(input_value),
                    source=source,
                )
            ]

        return []

    def _coerce_to_objects(self, parsed: Any) -> list[dict[str, Any]]:
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    def _text_candidates(self, content: str) -> list[str]:
        candidates = [content]
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(r"<tool_use>\s*(.*?)\s*</tool_use>", content, re.DOTALL | re.IGNORECASE)
        )
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(
                r"<tool_call>\s*(.*?)\s*</tool_call>",
                content,
                re.DOTALL | re.IGNORECASE,
            )
        )
        candidates.extend(line.strip() for line in content.splitlines() if line.strip().startswith(("{", "[")))
        if "{" in content and "}" in content:
            candidates.append(content[content.find("{"): content.rfind("}") + 1])
        candidates.extend(self._balanced_brace_candidates(content))
        return [candidate for candidate in candidates if candidate]

    def _strip_thinking_blocks(self, content: str) -> str:
        return re.sub(r"<think\b[^>]*>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)

    def _parse_markup_calls(self, text: str, *, source: str) -> list[NormalizedToolCall]:
        calls: list[NormalizedToolCall] = []
        calls.extend(self._parse_function_tag_calls(text, source=source))
        calls.extend(self._parse_glm_arg_tag_calls(text, source=source))
        calls.extend(self._parse_namespaced_tool_calls(text, source=source))
        calls.extend(self._parse_bracket_tool_calls(text, source=source))
        calls.extend(self._parse_gemma_call_calls(text, source=source))
        return calls

    def _parse_function_tag_calls(self, text: str, *, source: str) -> list[NormalizedToolCall]:
        calls: list[NormalizedToolCall] = []
        pattern = re.compile(r"<function=([A-Za-z_][\w.\-:]*)>(.*?)</function>", re.DOTALL)
        for index, match in enumerate(pattern.finditer(text)):
            args: dict[str, Any] = {}
            body = match.group(2)
            for param in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL):
                args[param.group(1).strip()] = self._coerce_scalar(param.group(2).strip())
            calls.append(
                NormalizedToolCall(
                    id=f"call_{index}",
                    name=match.group(1),
                    input=args,
                    source=source,
                )
            )
        return calls

    def _parse_glm_arg_tag_calls(self, text: str, *, source: str) -> list[NormalizedToolCall]:
        calls: list[NormalizedToolCall] = []
        pattern = re.compile(
            r"(?P<name>[A-Za-z_][\w.\-:]*)\s*"
            r"<arg_key>(?P<key>.*?)</arg_key>\s*"
            r"<arg_value>(?P<value>.*?)</arg_value>",
            re.DOTALL,
        )
        for index, match in enumerate(pattern.finditer(text)):
            calls.append(
                NormalizedToolCall(
                    id=f"call_{index}",
                    name=match.group("name"),
                    input={
                        match.group("key").strip(): self._coerce_scalar(
                            match.group("value").strip()
                        )
                    },
                    source=source,
                )
            )
        return calls

    def _parse_namespaced_tool_calls(self, text: str, *, source: str) -> list[NormalizedToolCall]:
        calls: list[NormalizedToolCall] = []
        envelope = re.compile(r"<[\w.\-]+:tool_call\b[^>]*>(.*?)</[\w.\-]+:tool_call>", re.DOTALL)
        invoke = re.compile(r"<invoke\s+name=['\"]([^'\"]+)['\"]\s*>(.*?)</invoke>", re.DOTALL)
        param = re.compile(r"<parameter\s+name=['\"]([^'\"]+)['\"]\s*>(.*?)</parameter>", re.DOTALL)
        for index, env_match in enumerate(envelope.finditer(text)):
            inv_match = invoke.search(env_match.group(1))
            if not inv_match:
                continue
            args = {
                p.group(1).strip(): self._coerce_scalar(p.group(2).strip())
                for p in param.finditer(inv_match.group(2))
            }
            calls.append(
                NormalizedToolCall(
                    id=f"call_{index}",
                    name=inv_match.group(1),
                    input=args,
                    source=source,
                )
            )
        return calls

    def _parse_bracket_tool_calls(self, text: str, *, source: str) -> list[NormalizedToolCall]:
        calls: list[NormalizedToolCall] = []
        pattern = re.compile(
            r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.\-:]*)\s*(?:\((.*?)\))?\]",
            re.DOTALL | re.IGNORECASE,
        )
        for index, match in enumerate(pattern.finditer(text)):
            calls.append(
                NormalizedToolCall(
                    id=f"call_{index}",
                    name=match.group(1),
                    input=self._parse_input(match.group(2) or {}),
                    source=source,
                )
            )
        return calls

    def _parse_gemma_call_calls(self, text: str, *, source: str) -> list[NormalizedToolCall]:
        calls: list[NormalizedToolCall] = []
        pattern = re.compile(r"\bcall:([A-Za-z_][\w.\-:]*)\s*(\{.*?\})", re.DOTALL)
        for index, match in enumerate(pattern.finditer(text)):
            calls.append(
                NormalizedToolCall(
                    id=f"call_{index}",
                    name=match.group(1),
                    input=self._parse_input(match.group(2)),
                    source=source,
                )
            )
        return calls

    def _balanced_brace_candidates(self, content: str) -> list[str]:
        candidates: list[str] = []
        starts = [i for i, char in enumerate(content) if char in "{["]
        for start in starts:
            stack: list[str] = []
            quote: str | None = None
            escaped = False
            for pos in range(start, len(content)):
                char = content[pos]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = None
                    continue
                if char in ("'", '"'):
                    quote = char
                elif char in "{[":
                    stack.append("}" if char == "{" else "]")
                elif stack and char == stack[-1]:
                    stack.pop()
                    if not stack:
                        candidates.append(content[start: pos + 1])
                        break
        return candidates

    def _parse_text(self, text: str) -> Any:
        for parser_input in (text, self._pythonize_json_literals(text)):
            try:
                return json.loads(parser_input)
            except (TypeError, ValueError):
                pass
            try:
                return ast.literal_eval(parser_input)
            except (SyntaxError, ValueError, TypeError):
                pass
        return self._parse_malformed_write_tool_use_text(text)

    def _parse_malformed_write_tool_use_text(self, text: str) -> dict[str, Any] | None:
        if "Write" not in text or "file_path" not in text or "content" not in text:
            return None

        id_match = re.search(r"['\"]id['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
        name_match = re.search(r"['\"]name['\"]\s*:\s*['\"]Write['\"]", text)
        content_match = re.search(
            r"['\"]content['\"]\s*:\s*(['\"])(.*?)\1\s*,\s*['\"]file_path['\"]",
            text,
            re.DOTALL,
        )
        file_path_match = re.search(r"['\"]file_path['\"]\s*:\s*(['\"])(.*?)\1", text, re.DOTALL)
        if not name_match or not content_match or not file_path_match:
            return None

        return {
            "type": "tool_use",
            "id": id_match.group(1) if id_match else None,
            "name": "Write",
            "input": {
                "content": content_match.group(2),
                "file_path": file_path_match.group(2),
            },
        }

    def _parse_input(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed = ast.literal_eval(self._pythonize_json_literals(value))
                except (SyntaxError, ValueError, TypeError):
                    return {"value": value}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {"value": value}

    def _coerce_scalar(self, value: str) -> Any:
        stripped = value.strip()
        if not stripped:
            return ""
        for parser_input in (stripped, self._pythonize_json_literals(stripped)):
            try:
                return json.loads(parser_input)
            except (TypeError, ValueError):
                pass
            try:
                return ast.literal_eval(parser_input)
            except (SyntaxError, ValueError, TypeError):
                pass
        return stripped

    def _pythonize_json_literals(self, text: str) -> str:
        converted = re.sub(r"\bfalse\b", "False", text)
        converted = re.sub(r"\btrue\b", "True", converted)
        return re.sub(r"\bnull\b", "None", converted)

    def _normalize_id(self, raw_id: Any, index: int) -> str:
        call_id = str(raw_id or "").strip()
        if not call_id:
            return f"call_{index}"
        if call_id.startswith("call_"):
            return call_id
        if call_id.startswith("call"):
            suffix = call_id[len("call"):]
            if suffix:
                return f"call_{suffix}"
        return call_id
