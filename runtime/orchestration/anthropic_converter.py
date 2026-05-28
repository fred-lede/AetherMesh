from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Any

from config.settings import settings
from protocols.anthropic.sse_builder import map_stop_reason
from runtime.orchestration.capabilities import required_anthropic_capabilities
from runtime.orchestration.provider_router import (
    adapter,
    capabilities_for_model,
    local_ollama_fallback,
    resolve_provider,
)
from runtime.orchestration.routing_engine import routing_engine
from runtime.security.tool_policy import server_tool_name
from runtime.tools.content_blocks import anthropic_block_to_openai_parts, anthropic_content_to_openai_parts
from runtime.tools.tool_normalizer import NormalizedToolCall, ToolCallNormalizer

logger = logging.getLogger("anthropic_converter")


class AnthropicRouter:
    def __init__(self) -> None:
        self.registry = settings.model_registry()
        self.tool_call_normalizer = ToolCallNormalizer()

    def list_models(self) -> dict[str, Any]:
        data = []
        for model in self.registry.get("models", []):
            data.append(
                {
                    "id": model["name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": model.get("provider", "ollama"),
                    "capabilities": model.get("capabilities", []),
                }
            )
        alias_prefix = settings.model_alias_prefix()
        for alias, target in settings.model_alias_entries().items():
            model_id = f"{alias_prefix}/{alias}" if alias_prefix else alias
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "alias",
                    "capabilities": self._capabilities_for_model(target),
                    "target": target,
                }
            )
        return {"object": "list", "data": data}

    def _to_openai_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = []
        system = payload.get("system")
        if system:
            if isinstance(system, str):
                messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                parts: list[str] = []
                for block in system:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        parts.append(block.get("text", ""))
                if parts:
                    messages.append({"role": "system", "content": "\n".join(parts)})

        for msg in payload.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant" and isinstance(content, str) and content:
                messages.append({"role": "assistant", "content": content})
                continue
            if isinstance(content, list):
                messages.append({"role": role, "content": anthropic_content_to_openai_parts(content)})
            else:
                messages.append({"role": role, "content": str(content)})

        openai_payload: dict[str, Any] = {
            "model": self._strip_model_prefix(payload["model"]),
            "messages": messages,
            "stream": payload.get("stream", False),
        }

        max_tokens = payload.get("max_tokens")
        if max_tokens is not None:
            openai_payload["max_completion_tokens"] = max_tokens

        temperature = payload.get("temperature")
        if temperature is not None:
            openai_payload["temperature"] = temperature

        top_p = payload.get("top_p")
        if top_p is not None:
            openai_payload["top_p"] = top_p

        top_k = payload.get("top_k")
        if top_k is not None:
            openai_payload["top_k"] = top_k

        stop = payload.get("stop_sequences")
        if stop:
            openai_payload["stop"] = stop

        thinking = payload.get("thinking")
        if thinking and isinstance(thinking, dict):
            budget = thinking.get("budget_tokens")
            if budget:
                openai_payload["thinking"] = {
                    "budget_tokens": budget,
                    "type": thinking.get("type", "enabled"),
                }

        tools = payload.get("tools")
        if tools and isinstance(tools, list):
            openai_payload["tools"] = self._anthropic_tools_to_openai(tools)

        tool_choice = payload.get("tool_choice")
        if tool_choice:
            openai_payload["tool_choice"] = self._anthropic_tool_choice_to_openai(tool_choice)

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            user_id = metadata.get("user_id")
            if user_id:
                openai_payload["user"] = user_id

        return openai_payload

    def _anthropic_block_to_openai(self, block: Any) -> dict[str, Any] | None:
        parts = anthropic_block_to_openai_parts(block)
        return parts[0] if parts else None

    def _strip_model_prefix(self, model: str) -> str:
        return settings.resolve_model_alias(model)

    def _capabilities_for_model(self, model: str) -> list[str]:
        return capabilities_for_model(model, self.registry)

    def _anthropic_tools_to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        openai_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if server_tool_name(tool):
                continue
            ttype = tool.get("type", "function")
            name = tool.get("name", "")

            if ttype != "function":
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{ttype}_{name}",
                            "description": f"External {ttype} tool: {tool.get('description', '')}",
                            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                        },
                    }
                )
                continue

            description = tool.get("description", "")
            input_schema = tool.get("input_schema", {})

            fn_def: dict[str, Any] = {
                "name": name,
                "description": description,
                "parameters": input_schema,
            }

            strict = tool.get("strict")
            if strict is not None:
                fn_def["strict"] = strict

            openai_tools.append({"type": "function", "function": fn_def})
        return openai_tools

    def _anthropic_tool_choice_to_openai(self, tool_choice: dict[str, Any]) -> dict[str, Any]:
        ttype = tool_choice.get("type", "auto")
        if ttype == "any":
            return {"type": "required"}
        if ttype == "tool":
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
        if ttype == "none":
            return {"type": "none"}
        return {"type": "auto"}

    def _to_anthropic_response(
        self,
        openai_response: dict[str, Any],
        model: str,
        allowed_tool_names: set[str] | None = None,
    ) -> dict[str, Any]:
        choice = openai_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        normalized_tool_calls = self.tool_call_normalizer.from_openai_tool_calls(message.get("tool_calls"))
        if settings.debug_responses:
            tcs = message.get("tool_calls") or []
            if isinstance(tcs, list):
                for tc in tcs:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    logger.info("_to_anthropic_response RAW tool_call: id=%s name=%s args=%s",
                        tc.get("id", "?"),
                        fn.get("name", "?") if isinstance(fn, dict) else "?",
                        fn.get("arguments", "?") if isinstance(fn, dict) else "?",
                    )
            if normalized_tool_calls:
                for nc in normalized_tool_calls:
                    logger.info("_to_anthropic_response NORMALIZED: id=%s name=%s input=%s",
                        nc.id, nc.name, nc.input,
                    )
        if isinstance(content, str):
            thinking_text = message.get("reasoning") or message.get("reasoning_content")
            combined_calls = self.tool_call_normalizer.from_content_with_thinking(
                content, thinking_text, allowed_tool_names=allowed_tool_names
            )
            if combined_calls:
                normalized_tool_calls = self.tool_call_normalizer.dedupe(normalized_tool_calls + combined_calls)
                content = ""

        content_blocks: list[dict[str, Any]] = []

        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            content_blocks.append({
                "type": "thinking",
                "thinking": str(reasoning),
                "signature": "",
            })

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        content_blocks.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        img_url = block.get("image_url", {})
                        content_blocks.append({"type": "image", "source": img_url})
                    elif block.get("type") == "thinking":
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": block.get("thinking", ""),
                            "signature": block.get("signature", ""),
                        })
                    else:
                        content_blocks.append({"type": "text", "text": str(block)})
                else:
                    content_blocks.append({"type": "text", "text": str(block)})
        elif content:
            content_blocks.append({"type": "text", "text": str(content)})

        if content_blocks and content_blocks[0].get("type") == "text":
            text = content_blocks[0]["text"]
            if text.startswith("[thinking]"):
                m = re.search(r"\[thinking\](.*?)\[/thinking\]", text, re.DOTALL)
                if m:
                    thinking_text = m.group(1).strip()
                    content_blocks.insert(0, {
                        "type": "thinking",
                        "thinking": thinking_text,
                        "signature": "",
                    })
                    content_blocks[1]["text"] = text[m.end():].strip()
                    if not content_blocks[1]["text"]:
                        content_blocks.pop(1)

        forwarded_tool_calls = []
        blocked_tool_calls = []
        for call in normalized_tool_calls:
            if self._tool_call_allowed(call, allowed_tool_names):
                forwarded_tool_calls.append(call)
            else:
                blocked_tool_calls.append(call)

        for call in forwarded_tool_calls:
            content_blocks.append(call.to_anthropic_content_block())

        if blocked_tool_calls:
            logger.info(
                "Suppressed undeclared upstream tool calls in non-streaming Anthropic response: %s",
                ", ".join(call.name for call in blocked_tool_calls),
            )

        finish_reason = choice.get("finish_reason", "stop")
        if forwarded_tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        stop_reason = self._openai_finish_to_stop_reason(finish_reason)

        usage = openai_response.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cache_creation = usage.get("prompt_tokens_details", {}).get("cache_creation_input_tokens", 0)
        cache_read = usage.get("prompt_tokens_details", {}).get("cache_read_input_tokens", 0)

        usage_dict: dict[str, Any] = {
            "input_tokens": int(input_tokens) if input_tokens else 0,
            "output_tokens": int(output_tokens) if output_tokens else 0,
        }
        if cache_creation or cache_read:
            usage_dict["cache_creation_input_tokens"] = int(cache_creation)
            usage_dict["cache_read_input_tokens"] = int(cache_read)

        response_dict: dict[str, Any] = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage_dict,
        }

        thinking_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        if thinking_tokens:
            response_dict["usage"]["output_tokens"] = int(output_tokens) if output_tokens else 0
            response_dict["usage"].setdefault("extra", {})["reasoning_tokens"] = int(thinking_tokens)

        return response_dict

    def _parse_tool_input(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return {"value": arguments}
        if isinstance(arguments, dict):
            return arguments
        return {"value": arguments}

    def _text_to_openai_tool_calls(self, content: str) -> list[dict[str, Any]]:
        return self.tool_call_normalizer.to_openai_tool_calls(content)

    def _extract_text_tool_use_objects(self, content: str) -> list[dict[str, Any]]:
        return [
            {"id": call.id, "name": call.name, "input": call.input}
            for call in self.tool_call_normalizer.from_text(content)
        ]

    def _normalize_text_tool_use_items(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(item, dict):
            return []

        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            objects: list[dict[str, Any]] = []
            for call in tool_calls:
                if isinstance(call, dict):
                    objects.extend(self._normalize_text_tool_use_items(call))
            return objects

        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", {})
            if name:
                return [{"id": item.get("id"), "name": name, "input": self._parse_tool_input(arguments)}]

        name = item.get("name") or item.get("tool_name")
        input_value = item.get("input", item.get("tool_args", {}))
        item_type = str(item.get("type", "")).replace("_", "").lower()
        if name and (item_type in ("", "function", "tooluse") or "input" in item or "tool_args" in item):
            return [{"id": item.get("id"), "name": name, "input": input_value}]

        return []

    def _tool_use_text_candidates(self, content: str) -> list[str]:
        candidates = [content]
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(r"<tool_use>\s*(.*?)\s*</tool_use>", content, re.DOTALL | re.IGNORECASE)
        )
        candidates.extend(line.strip() for line in content.splitlines() if line.strip().startswith(("{", "[")))
        if "{" in content and "}" in content:
            candidates.append(content[content.find("{"): content.rfind("}") + 1])
        return candidates

    def _parse_tool_use_text(self, text: str) -> Any:
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            pass
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError, TypeError):
            pass
        try:
            return ast.literal_eval(self._pythonize_json_literals(text))
        except (SyntaxError, ValueError, TypeError):
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

    def _pythonize_json_literals(self, text: str) -> str:
        converted = re.sub(r"\bfalse\b", "False", text)
        converted = re.sub(r"\btrue\b", "True", converted)
        return re.sub(r"\bnull\b", "None", converted)

    def _normalize_tool_call_id(self, raw_id: Any, index: int) -> str:
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

    def _looks_like_text_tool_use_fragment(self, content: str) -> bool:
        return self.tool_call_normalizer.looks_like_fragment(content)

    def _request_tool_names(self, payload: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return names
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name") or tool.get("type")
            if isinstance(name, str) and name:
                names.add(name)
        return names

    def _tool_call_allowed(self, call: NormalizedToolCall, allowed_tool_names: set[str] | None) -> bool:
        return allowed_tool_names is None or call.name in allowed_tool_names

    def _log_suppressed_tool_call(self, name: Any, *, streaming: bool) -> None:
        tool_name = str(name or "unknown").strip() or "unknown"
        logger.info(
            "Suppressed undeclared upstream tool call in %s Anthropic response: %s",
            "streaming" if streaming else "non-streaming",
            tool_name,
        )

    def _looks_like_tool_status_text(self, content: str) -> bool:
        normalized = " ".join(str(content or "").strip().lower().split())
        if not normalized:
            return False
        return normalized in {
            "searched the web",
            "searched the web.",
            "used a skill",
            "used a skill.",
            "running skill",
            "running skill.",
            "ran a command",
            "ran a command.",
            "running command",
            "running command.",
        }

    def _is_valid_json(self, text: str) -> bool:
        if not isinstance(text, str) or not text:
            return False
        try:
            json.loads(text)
        except (TypeError, ValueError):
            return False
        return True

    def _openai_finish_to_stop_reason(self, finish_reason: str) -> str:
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
            "function_call": "tool_use",
        }
        return mapping.get(finish_reason, "end_turn")

    def _resolve_provider(self, model: str) -> tuple[str, dict[str, Any] | None]:
        return resolve_provider(model, self.registry)

    def _adapter(self, provider: str, worker: dict[str, Any] | None):
        return adapter(provider, worker)

    def _local_ollama_fallback(
        self,
        required_capabilities: list[str],
    ) -> tuple[str, dict[str, Any]] | None:
        return local_ollama_fallback(required_capabilities, self.registry)


anthropic_service = AnthropicRouter()
