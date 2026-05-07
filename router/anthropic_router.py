from __future__ import annotations

import ast
import json
import logging
import re
import time
import uuid
from typing import Any, Iterable

from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import settings
from providers.base import ProviderError
from providers.gemini_adapter import GeminiAdapter
from providers.nvidia_nim_adapter import NvidiaNIMAdapter
from providers.ollama_adapter import OllamaAdapter
from providers.ollama_cloud_adapter import OllamaCloudAdapter
from providers.openai_adapter import OpenAIAdapter
from providers.http_client import get_session
from metrics.request_metrics import RequestRecord, request_metrics
from router.routing_engine import routing_engine
from router.anthropic_sse_builder import AnthropicSSEBuilder, format_sse as _format_sse, map_stop_reason
from router.capabilities import required_anthropic_capabilities
from router.content_blocks import anthropic_block_to_openai_parts, anthropic_content_to_openai_parts
from router.server_tool_policy import evaluate_server_tool_policy, server_tool_name
from router.tool_call_normalizer import NormalizedToolCall, ToolCallNormalizer
from router.web_server_tools import stream_web_server_tool_response


class ASCIISafeJSONResponse(JSONResponse):
    """JSON response that cannot be mojibaked by charset-unaware clients."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")


class AnthropicRouter:
    def __init__(self) -> None:
        self.registry = settings.model_registry()
        self.tool_call_normalizer = ToolCallNormalizer()

    def list_models(self) -> dict[str, Any]:
        data = []
        for model in self.registry.get("models", []):
            data.append(
                {
                    "name": model["name"],
                    "type": "model",
                    "created": 0,
                    "owner": model.get("provider", "ollama"),
                    "capabilities": model.get("capabilities", []),
                }
            )
        alias_prefix = settings.model_alias_prefix()
        for alias, target in settings.model_alias_entries().items():
            name = f"{alias_prefix}/{alias}" if alias_prefix else alias
            data.append(
                {
                    "name": name,
                    "type": "model",
                    "created": 0,
                    "owner": "alias",
                    "capabilities": self._capabilities_for_model(target),
                    "target": target,
                }
            )
        return {"type": "list", "data": data}

    # ── Request conversion: Anthropic → OpenAI ──────────────────────────

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

            # assistant prefill: if role=assistant and content is plain text,
            # treat it as a prefill hint appended as a system directive.
            # OpenAI doesn't have native prefill, so we pass it through content
            # and downstream providers may support it.
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

        # Extended Thinking
        thinking = payload.get("thinking")
        if thinking and isinstance(thinking, dict):
            budget = thinking.get("budget_tokens")
            if budget:
                # Pass thinking directly as a top-level parameter (supported by OpenAI/NVIDIA)
                openai_payload["thinking"] = {
                    "budget_tokens": budget,
                    "type": thinking.get("type", "enabled"),
                }

        # Tools
        tools = payload.get("tools")
        if tools and isinstance(tools, list):
            openai_payload["tools"] = self._anthropic_tools_to_openai(tools)

        tool_choice = payload.get("tool_choice")
        if tool_choice:
            openai_payload["tool_choice"] = self._anthropic_tool_choice_to_openai(tool_choice)

        # Metadata and stream_options are OpenAI-specific and may be rejected by
        # other providers (like NVIDIA NIM). We omit them to prevent validation errors.

        return openai_payload

    def _anthropic_block_to_openai(self, block: Any) -> dict[str, Any] | None:
        parts = anthropic_block_to_openai_parts(block)
        return parts[0] if parts else None

    def _strip_model_prefix(self, model: str) -> str:
        return settings.resolve_model_alias(model)

    def _capabilities_for_model(self, model: str) -> list[str]:
        for item in self.registry.get("models", []):
            if item.get("name") == model:
                caps = item.get("capabilities", [])
                if isinstance(caps, list):
                    return caps
        return ["chat"]

    def _anthropic_tools_to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        openai_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if server_tool_name(tool):
                continue
            ttype = tool.get("type", "function")
            name = tool.get("name", "")

            # Computer Use / Bash / other non-function tools
            if ttype != "function":
                # Map to a function description that downstream can interpret
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

            # Strict mode (OpenAI specific)
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
        return {"type": "auto"}

    # ── Response conversion: OpenAI → Anthropic ─────────────────────────

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
        if isinstance(content, str):
            text_tool_calls = self.tool_call_normalizer.from_text(content)
            if text_tool_calls:
                normalized_tool_calls = self.tool_call_normalizer.dedupe(normalized_tool_calls + text_tool_calls)
                content = ""

        content_blocks: list[dict[str, Any]] = []

        # Handle thinking tokens in reasoning models
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

        # Parse thinking tags from text content if no explicit reasoning field
        if content_blocks and content_blocks[0].get("type") == "text":
            text = content_blocks[0]["text"]
            if text.startswith("[thinking]"):
                import re
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

        # Collect rate limit headers if present
        resp_headers = openai_response.get("_headers", {})

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

        # Add thinking to usage if present
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

    # ── Streaming conversion: OpenAI chunks → Anthropic SSE events ─────

    def _openai_chunk_to_anthropic_events(
        self,
        chunk: dict[str, Any],
        model: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        if delta.get("role") == "assistant":
            events.append(
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": f"msg_{uuid.uuid4().hex[:24]}",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )
            )

        # Thinking blocks
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if reasoning:
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    },
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "thinking_delta", "thinking": str(reasoning)},
                    },
                )
            )

        # Text content
        content = delta.get("content")
        if content:
            # Check if we need a content_block_start first
            if not any(e[0] == "content_block_start" for e in events):
                events.append(
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": content},
                    },
                )
            )

        # Tool calls in stream
        tool_calls = delta.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for idx, tc in enumerate(tool_calls):
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    call_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    tool_index = idx + 1

                    if fn.get("name"):
                        events.append(
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": tool_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": call_id,
                                        "name": fn["name"],
                                        "input": {},
                                    },
                                },
                            )
                        )
                    if fn.get("arguments"):
                        events.append(
                            (
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": tool_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": fn["arguments"],
                                    },
                                },
                            )
                        )

        # Finish
        if finish_reason:
            stop_reason = self._openai_finish_to_stop_reason(finish_reason)
            usage = chunk.get("usage") or {}
            output_tokens = usage.get("completion_tokens", 0)

            events.append(
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {
                            "output_tokens": int(output_tokens) if output_tokens else 0,
                        },
                    },
                )
            )

            # Close all content blocks
            if not any(e[0] == "content_block_start" for e in events):
                events.append(
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                )
            # Find the last content block index
            max_index = 0
            for e in events:
                if e[0] == "content_block_start":
                    max_index = max(max_index, e[1].get("index", 0))
            events.append(
                (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": max_index},
                )
            )
            events.append(
                (
                    "message_stop",
                    {"type": "message_stop"},
                )
            )

        return events

    # ── Provider routing ────────────────────────────────────────────────

    def _adapter(self, provider: str, worker: dict[str, Any] | None):
        if provider == "ollama":
            if worker is None:
                raise HTTPException(status_code=503, detail="No worker was assigned.")
            return OllamaAdapter(worker["base_url"])
        if provider == "ollama_cloud":
            return OllamaCloudAdapter()
        if provider == "openai":
            return OpenAIAdapter()
        if provider == "gemini":
            return GeminiAdapter()
        if provider == "nvidia_nim":
            return NvidiaNIMAdapter()
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    def _local_ollama_fallback(
        self,
        required_capabilities: list[str],
    ) -> tuple[str, dict[str, Any]] | None:
        required = set(required_capabilities or ["chat"])
        configured_model = settings.ollama_fallback_model()
        if configured_model:
            configured = self._ollama_model_for_fallback(configured_model, required)
            if configured is not None:
                return configured

        fallback = None
        for model in self.registry.get("models", []):
            if str(model.get("provider", "ollama")).lower() != "ollama":
                continue
            if not model.get("worker_bindings"):
                continue
            capabilities = set(model.get("capabilities", []))
            if required.issubset(capabilities):
                fallback = model
                break
        if fallback is None:
            return None

        binding = fallback.get("worker_bindings", [])[0]
        base_url = settings.worker_base_url(binding)
        if not base_url:
            return None
        return str(fallback.get("name")), {"base_url": base_url}

    def _ollama_model_for_fallback(
        self,
        model_name: str,
        required: set[str],
    ) -> tuple[str, dict[str, Any]] | None:
        explicit_base_url = settings.ollama_fallback_base_url()
        for model in self.registry.get("models", []):
            if str(model.get("provider", "ollama")).lower() != "ollama":
                continue
            if model.get("name") != model_name:
                continue
            capabilities = set(model.get("capabilities", []))
            if required and not required.issubset(capabilities):
                return None
            if explicit_base_url:
                return model_name, {"base_url": explicit_base_url}

            bindings = model.get("worker_bindings", [])
            if not bindings:
                return None
            binding = bindings[0]
            base_url = settings.worker_base_url(binding)
            if not base_url:
                return None
            return model_name, {"base_url": base_url}
        return None

    def _resolve_provider(self, model: str) -> tuple[str, dict[str, Any] | None]:
        clean_model = self._strip_model_prefix(model)
        for item in self.registry.get("models", []):
            if item.get("name") in (model, clean_model):
                provider = str(item.get("provider", "ollama")).lower()
                if provider in ("openai", "gemini", "nvidia_nim", "ollama_cloud"):
                    return provider, None
                if provider == "ollama":
                    bindings = item.get("worker_bindings", [])
                    if bindings:
                        b = bindings[0]
                        base_url = settings.worker_base_url(b)
                        if base_url:
                            return provider, {"base_url": base_url}
                return provider, None

        if clean_model.endswith("-cloud") or clean_model.endswith("-cloud-latest"):
            return "ollama_cloud", None
        if clean_model.startswith("gemini"):
            return "gemini", None
        if clean_model.startswith("gpt") or clean_model.startswith(("o1", "o3", "o4")):
            return "openai", None
        nvidia_prefixes = ("meta/", "mistralai/", "nvidia/", "google/", "microsoft/", "baichuan-inc/",
                           "deepseek/", "upstage/", "snowflake/", "ibm/", "yola/", "writer/", "z-ai/")
        if any(clean_model.startswith(p) for p in nvidia_prefixes) or clean_model.startswith("nemotron"):
            return "nvidia_nim", None
        return "ollama", None


anthropic_service = AnthropicRouter()
logger = logging.getLogger("anthropic_router")
app = FastAPI(title="AI Inference Hub - Anthropic Compatible", version="4.0.0")

# Rate limit headers to forward from upstream
ANTHROPIC_RATE_LIMIT_HEADERS = [
    "request-id",
    "retry-after",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-ratelimit-tokens-reset",
    "anthropic-ratelimit-input-tokens-limit",
    "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-input-tokens-reset",
    "anthropic-ratelimit-output-tokens-limit",
    "anthropic-ratelimit-output-tokens-remaining",
    "anthropic-ratelimit-output-tokens-reset",
    "anthropic-ratelimit-retry-after",
    "anthropic-ratelimit-tier",
    "cf-ray",
]


def _stream_anthropic_with_metrics(
    iterator: Iterable[dict[str, Any] | str],
    model: str,
    provider: str,
    request_id: str,
    start_time: float,
    allowed_tool_names: set[str] | None = None,
) -> Iterable[str]:
    total_output_tokens = 0
    last_error = None
    try:
        for item in _stream_anthropic(iterator, model, allowed_tool_names=allowed_tool_names):
            yield item
            if isinstance(item, str) and "content_block_delta" in item and "text_delta" in item:
                data = json.loads(item.split("data: ", 1)[1]) if "data: " in item else {}
                total_output_tokens += len(data.get("delta", {}).get("text", ""))
            if isinstance(item, str) and "error" in item and "event:" in item:
                last_error = item
    except Exception as e:
        logger.warning(f"Upstream stream interrupted for {model}: {type(e).__name__}: {e}")
        last_error = f"Stream interrupted: {e}"
    finally:
        latency_ms = (time.time() - start_time) * 1000
        request_metrics.record_request(RequestRecord(
            request_id=request_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            streaming=True,
            latency_ms=latency_ms,
            output_tokens=max(1, total_output_tokens // 4),
            error=last_error is not None,
            error_message=str(last_error or ""),
        ))
        routing_engine.set_provider_latency(provider, latency_ms)
        routing_engine.set_provider_health(provider, last_error is None)


def _stream_anthropic(
    iterator: Iterable[dict[str, Any] | str],
    model: str,
    allowed_tool_names: set[str] | None = None,
) -> Iterable[str]:
    sse = AnthropicSSEBuilder(model)
    pending_text_tool_content = ""
    emitted_tool_use = False
    suppress_tool_status_text = False
    suppressed_stream_tool_indexes: set[int] = set()

    yield sse.message_start()

    for item in iterator:
        if isinstance(item, str):
            if item == "[DONE]":
                break
            continue

        if isinstance(item, dict):
            if item.get("error"):
                yield sse.error(str(item["error"]))
                return

            choice = item.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")
            text_tool_call_objects: list[NormalizedToolCall] = []
            if content:
                if suppress_tool_status_text and anthropic_service._looks_like_tool_status_text(str(content)):
                    content = None
                else:
                    suppress_tool_status_text = False
            if content:
                combined_content = f"{pending_text_tool_content}{content}"
                text_tool_call_objects = anthropic_service.tool_call_normalizer.from_text(combined_content)
                if text_tool_call_objects:
                    pending_text_tool_content = ""
                    tool_calls = list(tool_calls or []) + [call.to_openai_tool_call() for call in text_tool_call_objects]
                    content = None
                    suppress_tool_status_text = True
                elif pending_text_tool_content or anthropic_service._looks_like_text_tool_use_fragment(str(content)):
                    pending_text_tool_content = combined_content
                    content = None

            # 1. Handle Reasoning / Thinking
            if reasoning:
                yield from sse.emit_thinking_delta(str(reasoning))

            # 2. Handle Text Content
            elif content:
                yield from sse.emit_text_delta(str(content))

            # 3. Handle Tool Calls
            if tool_calls and isinstance(tool_calls, list):
                normalized_text_calls = text_tool_call_objects
                for normalized_call in normalized_text_calls:
                    if not anthropic_service._tool_call_allowed(normalized_call, allowed_tool_names):
                        anthropic_service._log_suppressed_tool_call(normalized_call.name, streaming=True)
                        suppress_tool_status_text = True
                        continue

                    emitted_tool_use = True
                    suppress_tool_status_text = True
                    yield from sse.start_text_tool_call(
                        tool_id=normalized_call.id or f"toolu_{uuid.uuid4().hex[:24]}",
                        name=normalized_call.name,
                        arguments=normalized_call.arguments_text,
                    )
                native_tool_calls = tool_calls[: len(tool_calls) - len(normalized_text_calls)] if normalized_text_calls else tool_calls
                for tc in native_tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tool_index = int(tc.get("index", 0) or 0)
                    fn = tc.get("function", {})
                    tool_name = str(fn.get("name") or "")
                    if tool_name:
                        synthetic_call = NormalizedToolCall(
                            id=str(tc.get("id", "")),
                            name=tool_name,
                            input={},
                            source="stream",
                        )
                        if not anthropic_service._tool_call_allowed(synthetic_call, allowed_tool_names):
                            anthropic_service._log_suppressed_tool_call(tool_name, streaming=True)
                            suppress_tool_status_text = True
                            suppressed_stream_tool_indexes.add(tool_index)
                            continue
                        suppressed_stream_tool_indexes.discard(tool_index)
                    elif tool_index in suppressed_stream_tool_indexes:
                        continue
                    emitted_tool_use = True
                    suppress_tool_status_text = True
                    yield from sse.process_tool_call_delta(tc)

            # 4. Handle Finish
            if finish_reason:
                if pending_text_tool_content:
                    pending_calls = anthropic_service.tool_call_normalizer.from_text(pending_text_tool_content)
                    if pending_calls:
                        for call in pending_calls:
                            if anthropic_service._tool_call_allowed(call, allowed_tool_names):
                                emitted_tool_use = True
                                yield from sse.start_text_tool_call(
                                    tool_id=call.id,
                                    name=call.name,
                                    arguments=call.arguments_text,
                                )
                            else:
                                anthropic_service._log_suppressed_tool_call(call.name, streaming=True)
                                suppress_tool_status_text = True
                    else:
                        yield from sse.emit_text_delta(pending_text_tool_content)
                    pending_text_tool_content = ""
                if emitted_tool_use and finish_reason == "stop":
                    finish_reason = "tool_calls"
                stop_reason = map_stop_reason(finish_reason)
                if emitted_tool_use:
                    pending_text_tool_content = ""
                usage = item.get("usage") or {}
                output_tokens = usage.get("completion_tokens", 0)

                yield from sse.close_all_blocks()
                yield sse.message_delta(stop_reason, int(output_tokens) if output_tokens else 0)
                yield sse.message_stop()
                return


@app.get("/api/metrics/requests")
def metrics_requests() -> dict[str, Any]:
    return request_metrics.get_summary()


@app.get("/api/metrics/providers")
def metrics_providers() -> dict[str, Any]:
    return {"providers": request_metrics.get_provider_metrics()}


@app.get("/api/metrics/provider-diagnostics")
def metrics_provider_diagnostics() -> dict[str, Any]:
    return {"providers": request_metrics.get_provider_diagnostics()}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "anthropic_router"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return anthropic_service.list_models()


@app.post("/v1/messages")
def messages(
    payload: dict[str, Any] = Body(...),
    api_key: str | None = Header(default=None, alias="x-api-key"),
    anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(default=None, alias="anthropic-beta"),
):
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail={"type": "invalid_request_error", "message": "model is required"})

    request_id = f"req_{uuid.uuid4().hex[:24]}"
    is_streaming = payload.get("stream", False)
    start_time = time.time()
    allowed_tool_names = anthropic_service._request_tool_names(payload)

    openai_payload = anthropic_service._to_openai_payload(payload)
    openai_payload["model"] = anthropic_service._strip_model_prefix(model)

    required_caps = sorted(required_anthropic_capabilities(payload))

    routing_decision = routing_engine.route(
        model=model,
        required_capabilities=required_caps,
        registry_models=anthropic_service.registry.get("models", []),
        request_payload=payload,
    )
    if any(rule.startswith("capability_missing_no_fallback") for rule in routing_decision.rules_applied):
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_request_error",
                "message": (
                    f"Model {model} does not support required capabilities: "
                    f"{', '.join(required_caps)}"
                ),
            },
        )
    
    if routing_decision.score < 10:
        logger.info(f"Routing engine low score ({routing_decision.score}) for {model}, falling back to default resolver")
        provider, worker = anthropic_service._resolve_provider(model)
    else:
        provider = routing_decision.provider
        worker = routing_decision.worker
        openai_payload["model"] = routing_decision.model

    logger.info(
        "Routing %s -> model=%s provider=%s worker=%s score=%s",
        model,
        openai_payload["model"],
        provider,
        "yes" if worker else "no",
        routing_decision.score,
    )

    server_tool_policy = evaluate_server_tool_policy(
        payload,
        provider=provider,
        mode=settings.server_tool_mode,
        local_web_tools_enabled=settings.web_server_tools_enabled,
    )
    if server_tool_policy.error:
        raise HTTPException(
            status_code=400,
            detail={"type": "invalid_request_error", "message": server_tool_policy.error},
        )
    if server_tool_policy.should_handle_locally:
        request_metrics.record_request(RequestRecord(
            request_id=request_id,
            model=model,
            provider="aiih_web_tools",
            endpoint="/v1/messages",
            streaming=True,
            latency_ms=0,
        ))
        return StreamingResponse(
            stream_web_server_tool_response(
                payload,
                model=model,
                timeout_s=settings.web_tool_timeout_s,
                max_results=settings.web_search_max_results,
            ),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Request-Id": request_id, "X-AIIH-Server-Tool": server_tool_policy.forced_tool},
        )

    # Note: anthropic_beta, metadata, stream_options are Anthropic/OpenAI-specific
    # and are stripped out before sending to other upstream providers to prevent validation errors.
    # They are safely ignored here as the adapter only uses standard OpenAI keys.

    try:
        adapter = anthropic_service._adapter(provider, worker)
        if is_streaming:
            iterator = iter(adapter.stream(openai_payload))
            try:
                first_item = next(iterator)
            except StopIteration:
                first_item = "[DONE]"
            except ProviderError as exc:
                latency_ms = (time.time() - start_time) * 1000
                logger.warning("ProviderError before stream start for model=%s provider=%s: %s", model, provider, exc)
                request_metrics.record_request(RequestRecord(
                    request_id=request_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/messages",
                    streaming=True,
                    latency_ms=latency_ms,
                    error=True,
                    error_message=str(exc),
                ))
                routing_engine.set_provider_latency(provider, latency_ms)
                routing_engine.set_provider_failure(
                    provider,
                    code=str(getattr(exc, "code", "") or ""),
                    message=str(exc),
                )
                fallback = anthropic_service._local_ollama_fallback(required_caps) if provider != "ollama" else None
                if fallback is None:
                    raise
                fallback_model, fallback_worker = fallback
                fallback_payload = dict(openai_payload)
                fallback_payload["model"] = fallback_model
                provider = "ollama"
                worker = fallback_worker
                adapter = anthropic_service._adapter(provider, worker)
                iterator = iter(adapter.stream(fallback_payload))
                try:
                    first_item = next(iterator)
                except StopIteration:
                    first_item = "[DONE]"
                except ProviderError as fallback_exc:
                    logger.error("Local streaming fallback failed for model=%s: %s", fallback_model, fallback_exc)
                    raise exc

            def stream_with_first() -> Iterable[dict[str, Any] | str]:
                yield first_item
                yield from iterator

            response_headers = {"Cache-Control": "no-cache", "X-Request-Id": request_id}
            return StreamingResponse(
                _stream_anthropic_with_metrics(
                    stream_with_first(),
                    model,
                    provider,
                    request_id,
                    start_time,
                    allowed_tool_names=allowed_tool_names,
                ),
                media_type="text/event-stream; charset=utf-8",
                headers=response_headers,
            )
        else:
            response = adapter.chat(openai_payload)
            latency_ms = (time.time() - start_time) * 1000
            usage = response.get("usage") or {}
            request_metrics.record_request(RequestRecord(
                request_id=request_id,
                model=model,
                provider=provider,
                endpoint="/v1/messages",
                streaming=False,
                latency_ms=latency_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ))
            routing_engine.set_provider_latency(provider, latency_ms)
            routing_engine.set_provider_health(provider, True)
            result = anthropic_service._to_anthropic_response(response, model, allowed_tool_names=allowed_tool_names)
            return ASCIISafeJSONResponse(
                content=result,
                media_type="application/json; charset=utf-8",
                headers={"X-Request-Id": request_id},
            )
    except HTTPException:
        raise
    except ProviderError as exc:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"ProviderError for model={model}, provider={provider}: {exc}")
        status_code = int(getattr(exc, "status_code", None) or 502)
        retry_after = getattr(exc, "retry_after", None)
        error_code = str(getattr(exc, "code", "") or "")
        error_type = "api_error"
        if status_code == 429:
            error_type = "rate_limit_error"
        elif status_code in {502, 503}:
            error_type = "overloaded_error"
        elif status_code == 404 or error_code == "model_not_found":
            error_type = "invalid_request_error"
        request_metrics.record_request(RequestRecord(
            request_id=request_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            streaming=is_streaming,
            latency_ms=latency_ms,
            error=True,
            error_message=str(exc),
        ))
        routing_engine.set_provider_latency(provider, latency_ms)
        routing_engine.set_provider_failure(provider, code=error_code, message=str(exc))
        fallback = anthropic_service._local_ollama_fallback(required_caps) if provider != "ollama" else None
        if fallback is not None and (status_code in {404, 429, 502, 503, 504} or error_code == "model_not_found"):
            fallback_model, fallback_worker = fallback
            fallback_payload = dict(openai_payload)
            fallback_payload["model"] = fallback_model
            logger.warning(
                "Falling back %s/%s to local Ollama model=%s after provider error: %s",
                provider,
                model,
                fallback_model,
                exc,
            )
            try:
                fallback_adapter = anthropic_service._adapter("ollama", fallback_worker)
                response = fallback_adapter.chat(fallback_payload)
                fallback_latency_ms = (time.time() - start_time) * 1000
                usage = response.get("usage") or {}
                request_metrics.record_request(RequestRecord(
                    request_id=request_id,
                    model=model,
                    provider="ollama",
                    endpoint="/v1/messages",
                    streaming=False,
                    latency_ms=fallback_latency_ms,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                ))
                routing_engine.set_provider_latency("ollama", fallback_latency_ms)
                routing_engine.set_provider_health("ollama", True)
                result = anthropic_service._to_anthropic_response(response, model, allowed_tool_names=allowed_tool_names)
                return ASCIISafeJSONResponse(
                    content=result,
                    media_type="application/json; charset=utf-8",
                    headers={"X-Request-Id": request_id, "X-AIIH-Fallback": f"{provider}->ollama"},
                )
            except ProviderError as fallback_exc:
                logger.error("Local fallback failed for model=%s: %s", fallback_model, fallback_exc)
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        raise HTTPException(
            status_code=status_code,
            detail={"type": error_type, "message": str(exc)},
            headers=headers,
        )
    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        logger.exception(f"Unexpected error for model={model}, provider={provider}")
        request_metrics.record_request(RequestRecord(
            request_id=request_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            streaming=is_streaming,
            latency_ms=latency_ms,
            error=True,
            error_message=str(exc),
        ))
        routing_engine.set_provider_latency(provider, latency_ms)
        routing_engine.set_provider_failure(provider, code="api_error", message=str(exc))
        raise HTTPException(status_code=500, detail={"type": "api_error", "message": str(exc)})


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = int(exc.status_code)
    detail = exc.detail

    # Map to Anthropic error types
    error_type = "api_error"
    if isinstance(detail, dict):
        error_type = detail.get("type", error_type)

    if status_code == 429:
        error_type = "rate_limit_error"
    elif status_code == 400:
        error_type = detail.get("type", "invalid_request_error") if isinstance(detail, dict) else "invalid_request_error"
    elif status_code in (502, 503, 504):
        error_type = "overloaded_error" if status_code == 503 else "api_error"

    if isinstance(detail, dict):
        error_payload = {
            "type": error_type,
            "message": detail.get("message", "Request failed."),
        }
    elif isinstance(detail, str):
        error_payload = {"type": error_type, "message": detail}
    else:
        error_payload = {"type": error_type, "message": "Request failed."}

    headers: dict[str, str] = dict(exc.headers or {})
    headers["X-Request-Id"] = f"req_{uuid.uuid4().hex[:24]}"
    if status_code == 429:
        headers.setdefault("Retry-After", "60")

    return ASCIISafeJSONResponse(
        status_code=status_code,
        content={"type": "error", "error": error_payload},
        media_type="application/json; charset=utf-8",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
