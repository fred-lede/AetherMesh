from __future__ import annotations

import ast
import json
import logging
import re
import time
import uuid
from typing import Any, Iterable

import requests

from config.settings import settings

from .base import ProviderAdapter, ProviderError
from .http_client import get_session
from cluster.circuit_breaker import get_circuit_registry

LOGGER = logging.getLogger("aiih.ollama")


class OllamaAdapter(ProviderAdapter):
    provider_name = "ollama"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.debug_tool_calls = settings.debug_tool_calls
        # Create circuit breaker for this worker
        self._circuit = get_circuit_registry().get_or_create(
            base_url,
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
        )

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._chat_payload(payload, stream=False)
        response = self._post_chat_with_retry(body, stream=False)
        response.encoding = "utf-8"
        if not response.ok:
            raise ProviderError(response.text)
        data = response.json()
        return self._to_chat_completion(payload.get("model", "unknown"), data)

    def responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        completion = self.chat(payload)
        message = completion["choices"][0]["message"]
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created": int(time.time()),
            "model": completion["model"],
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": message.get("content", ""),
                        }
                    ],
                }
            ],
            "usage": completion.get("usage", {}),
        }

    def stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any] | str]:
        body = self._chat_payload(payload, stream=True)
        response = self._post_chat_with_retry(body, stream=True)
        response.encoding = "utf-8"
        if not response.ok:
            raise ProviderError(response.text)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = payload.get("model", "unknown")
        argument_buffers: dict[str, str] = {}
        emitted_tool_calls = False
        pending_text_tool_content = ""
        buffer_text_when_tools = bool(body.get("tools"))

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            item = json.loads(raw_line)
            if item.get("done"):
                if pending_text_tool_content:
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": pending_text_tool_content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    pending_text_tool_content = ""
                finish_reason = item.get("done_reason", "stop")
                if emitted_tool_calls and finish_reason == "stop":
                    finish_reason = "tool_calls"
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield "[DONE]"
                return

            message = item.get("message", {})
            delta: dict[str, Any] = {"role": "assistant"}

            content = str(message.get("content") or "")
            if content:
                combined_content = f"{pending_text_tool_content}{content}"
                text_tool_calls = self._text_to_openai_tool_calls(combined_content)
                if text_tool_calls:
                    emitted_tool_calls = True
                    pending_text_tool_content = ""
                    delta["tool_calls"] = text_tool_calls
                elif (
                    buffer_text_when_tools
                    or pending_text_tool_content
                    or self._looks_like_text_tool_use_fragment(content)
                ):
                    pending_text_tool_content = combined_content
                else:
                    delta["content"] = content

            raw_tool_calls = message.get("tool_calls")
            self._debug_tool_calls(
                "stream.raw_tool_calls",
                {
                    "base_url": self.base_url,
                    "summary": self._summarize_tool_calls(raw_tool_calls),
                },
            )

            tool_calls = self._to_openai_tool_calls_for_stream(raw_tool_calls, argument_buffers)
            if tool_calls:
                emitted_tool_calls = True
                pending_text_tool_content = ""
                delta["tool_calls"] = tool_calls
                self._debug_tool_calls(
                    "stream.normalized_tool_calls",
                    {
                        "base_url": self.base_url,
                        "summary": self._summarize_tool_calls(tool_calls),
                    },
                )

            if len(delta) > 1:
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }

    def _post_chat_with_retry(self, body: dict[str, Any], *, stream: bool) -> requests.Response:
        # Check circuit breaker first
        if not self._circuit.is_available():
            raise ProviderError(f"Circuit breaker OPEN for {self.base_url}, skipping request")

        attempts = 5  # Increase max attempts for robustness
        base_delay = 1.0  # Initial delay in seconds
        last_response: requests.Response | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = get_session().post(
                    f"{self.base_url}/api/chat",
                    json=body,
                    timeout=settings.request_timeout_s,
                    stream=stream,
                )
                last_response = response

                # Check for successful status code range (200-299)
                if 200 <= response.status_code < 300:
                    self._circuit.record_success()
                    return response

                # Check for transient/recoverable errors
                if response.status_code in [429, 500, 502, 503, 504]:
                    self._circuit.record_failure()
                    if attempt < attempts:
                        wait_time = base_delay * (2 ** (attempt - 1)) + (attempt * 0.1) # Exponential backoff + small jitter
                        wait_time = min(wait_time, 30) # Cap wait time at 30 seconds
                        LOGGER.warning(
                            "Ollama chat transient failure (%s, status code %s) on attempt %s/%s. Retrying in %.2f seconds...",
                            self.base_url,
                            response.status_code,
                            attempt,
                            attempts,
                            wait_time
                        )
                        time.sleep(wait_time)
                        continue  # Retry the loop
                    else:
                        # Last attempt failed, proceed to final raise
                        break
                else:
                    # Other errors are recorded as failures
                    self._circuit.record_failure() 
            
            except requests.exceptions.Timeout:
                if attempt < attempts:
                    wait_time = base_delay * (2 ** (attempt - 1)) + (attempt * 0.1)
                    wait_time = min(wait_time, 30)
                    LOGGER.warning(
                        "Ollama chat REQUEST TIMEOUT on attempt %s/%s. Retrying in %.2f seconds...",
                        attempt,
                        attempts,
                        wait_time
                    )
                    time.sleep(wait_time)
                    continue  # Retry the loop
                else:
                    raise requests.exceptions.Timeout(f"Failed to connect (Timeout) after {attempts} attempts.") from None

            except requests.exceptions.ConnectionError as e:
                if attempt < attempts:
                    wait_time = base_delay * (2 ** (attempt - 1)) + (attempt * 0.1)
                    wait_time = min(wait_time, 30)
                    LOGGER.warning(
                        "Ollama chat CONNECTION ERROR on attempt %s/%s. Retrying in %.2f seconds... Error: %s",
                        attempt,
                        attempts,
                        wait_time,
                        e
                    )
                    time.sleep(wait_time)
                    continue # Retry the loop
                else:
                    raise requests.exceptions.ConnectionError(f"Failed to connect to Ollama after {attempts} attempts.") from e

        # If we exited the loop because of failed status/non-retryable error
        if last_response and not last_response.ok and response.status_code not in [429, 500, 502, 503, 504]:
            raise ProviderError(f"Final API failure: {last_response.text}")
        
        # If the loop finished due to exhaustion of attempts
        if last_response and last_response.status_code not in [200, 201, 202]:
            raise ProviderError(f"Final API failure after {attempts} attempts: {last_response.text}")
        
        # Should not reach here if logic is correct, but for safety:
        raise ProviderError("Unknown connection error occurred during API call.")

        assert last_response is not None
        return last_response

    def _is_transient_runner_stop(self, text: str) -> bool:
        lowered = (text or "").lower()
        return "model runner has unexpectedly stopped" in lowered
    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": payload["model"],
            "input": payload.get("input", []),
        }
        response = get_session().post(
            f"{self.base_url}/api/embed",
            json=body,
            timeout=settings.request_timeout_s,
        )
        response.encoding = "utf-8"
        if not response.ok:
            raise ProviderError(response.text)
        data = response.json()
        embeddings = data.get("embeddings", [])
        rows = []
        for index, embedding in enumerate(embeddings):
            rows.append({"object": "embedding", "index": index, "embedding": embedding})
        return {"object": "list", "data": rows, "model": payload["model"], "usage": {}}
    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": payload["model"],
            "query": str(payload.get("query", "")),
            "documents": payload.get("documents", []),
        }
        if "top_n" in payload:
            body["top_n"] = payload.get("top_n")

        response = get_session().post(
            f"{self.base_url}/api/rerank",
            json=body,
            timeout=settings.request_timeout_s,
        )
        response.encoding = "utf-8"
        if not response.ok:
            raise ProviderError(response.text)

        data = response.json()
        rows: list[dict[str, Any]] = []
        documents = body.get("documents", [])

        for index, item in enumerate(data.get("results", [])):
            if not isinstance(item, dict):
                continue
            doc_index = int(item.get("index", index))
            doc_value = item.get("document")
            if doc_value is None and isinstance(documents, list) and 0 <= doc_index < len(documents):
                doc_value = documents[doc_index]
            rows.append(
                {
                    "index": doc_index,
                    "relevance_score": _safe_float(item.get("relevance_score", item.get("score", 0.0))),
                    "document": doc_value,
                }
            )

        return {
            "object": "list",
            "data": rows,
            "model": payload["model"],
            "usage": data.get("usage", {}),
        }

    def health_check(self) -> dict[str, Any]:
        response = get_session().get(f"{self.base_url}/api/tags", timeout=5)
        response.encoding = "utf-8"
        return {"ok": response.ok, "status_code": response.status_code, "base_url": self.base_url}

    def _chat_payload(self, payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": payload["model"],
            "messages": self._messages_for_ollama(payload.get("messages", [])),
            "stream": stream,
        }

        # Pass through options/controls only when requested by caller.
        options: dict[str, Any] = dict(payload.get("options", {}))
        max_completion_tokens = payload.get("max_completion_tokens")
        max_tokens = payload.get("max_tokens")
        num_predict = max_completion_tokens if max_completion_tokens is not None else max_tokens
        if num_predict is not None:
            try:
                options["num_predict"] = max(1, int(num_predict))
            except (TypeError, ValueError):
                pass

        for source_key, option_key in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop", "stop")):
            if source_key in payload:
                options[option_key] = payload[source_key]

        if options:
            body["options"] = options

        if "tools" in payload:
            body["tools"] = self._tools_for_ollama(payload["tools"])

        # Ollama accepts tools, but many versions reject OpenAI/Anthropic
        # tool_choice shapes such as {"type": "auto"} or {"type": "required"}.
        for key in ("format", "keep_alive", "think"):
            if key in payload:
                body[key] = payload[key]
        return body

    def _messages_for_ollama(self, messages: Any) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append(self._message_for_ollama(message))
        return normalized

    def _message_for_ollama(self, message: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {"role": str(message.get("role", "user"))}
        content = message.get("content", "")
        images: list[str] = []

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                text = self._content_part_text_for_ollama(part)
                if text:
                    text_parts.append(text)
                image = self._content_part_image_for_ollama(part)
                if image:
                    images.append(image)
            normalized["content"] = "\n".join(text_parts)
        else:
            normalized["content"] = "" if content is None else str(content)

        if images:
            normalized["images"] = images

        tool_calls = message.get("tool_calls")
        if tool_calls:
            normalized["tool_calls"] = tool_calls

        return normalized

    def _content_part_text_for_ollama(self, part: Any) -> str:
        if isinstance(part, str):
            return part
        if not isinstance(part, dict):
            return "" if part is None else str(part)

        part_type = part.get("type")
        if part_type == "text":
            return str(part.get("text", ""))
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url", ""))
                if url and not url.startswith("data:"):
                    return f"[image: {url}]"
            return ""
        return str(part)

    def _content_part_image_for_ollama(self, part: Any) -> str | None:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            return None
        image_url = part.get("image_url")
        if not isinstance(image_url, dict):
            return None
        url = str(image_url.get("url", ""))
        if not url.startswith("data:"):
            return None
        return url.split(",", 1)[1] if "," in url else url

    def _tools_for_ollama(self, tools: Any) -> Any:
        if not isinstance(tools, list):
            return tools

        normalized_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                normalized_tools.append(tool)
                continue

            function = dict(tool["function"])
            function.pop("strict", None)
            normalized_tools.append({"type": "function", "function": function})
        return normalized_tools

    def _debug_tool_calls(self, stage: str, payload: Any) -> None:
        if not self.debug_tool_calls:
            return
        if isinstance(payload, dict) and payload.get("summary") == []:
            return
        try:
            LOGGER.debug("tool_calls.%s %s", stage, json.dumps(payload, ensure_ascii=False))
        except Exception:
            LOGGER.debug("tool_calls.%s %s", stage, str(payload))

    def _summarize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []

        summary: list[dict[str, Any]] = []
        for idx, item in enumerate(tool_calls):
            if not isinstance(item, dict):
                continue

            function = item.get("function") if isinstance(item.get("function"), dict) else {}
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str):
                arg_type = "str"
                arg_len = len(arguments)
                arg_preview = arguments[:120]
            elif isinstance(arguments, dict):
                arg_type = "dict"
                arg_len = len(arguments)
                arg_preview = ""
            elif isinstance(arguments, list):
                arg_type = "list"
                arg_len = len(arguments)
                arg_preview = ""
            elif arguments is None:
                arg_type = "none"
                arg_len = 0
                arg_preview = ""
            else:
                arg_type = type(arguments).__name__
                arg_len = 0
                arg_preview = str(arguments)[:120]

            summary.append(
                {
                    "index": idx,
                    "id": str(item.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments_type": arg_type,
                    "arguments_len": arg_len,
                    "arguments_preview": arg_preview,
                }
            )

        return summary

    def _to_openai_tool_calls(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_tool_calls, list):
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw_tool_calls):
            if not isinstance(item, dict):
                continue

            function = item.get("function")
            if not isinstance(function, dict):
                continue

            name = str(function.get("name") or "")
            arguments = function.get("arguments", "")
            if isinstance(arguments, str):
                arguments_text = arguments
            else:
                arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

            call_id = str(item.get("id") or f"call_{index}")
            normalized.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_text,
                    },
                }
            )

        return normalized

    def _text_to_openai_tool_calls(self, content: str) -> list[dict[str, Any]]:
        calls = self._extract_text_tool_use_objects(content)
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for index, call in enumerate(calls):
            name = str(call.get("name") or "")
            if not name:
                continue
            input_value = call.get("input", {})
            if isinstance(input_value, str):
                arguments_text = input_value if self._is_valid_json(input_value) else json.dumps(
                    {"value": input_value},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            else:
                arguments_text = json.dumps(input_value, ensure_ascii=False, separators=(",", ":"))

            dedupe_key = (name, arguments_text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            normalized.append(
                {
                    "id": self._normalize_tool_call_id(call.get("id"), index),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_text,
                    },
                }
            )

        return normalized

    def _parse_tool_input(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return {"value": arguments}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        if isinstance(arguments, dict):
            return arguments
        return {"value": arguments}

    def _extract_text_tool_use_objects(self, content: str) -> list[dict[str, Any]]:
        stripped = content.strip()
        if not stripped:
            return []

        objects: list[dict[str, Any]] = []
        for candidate in self._tool_use_text_candidates(stripped):
            parsed = self._parse_tool_use_text(candidate)
            if isinstance(parsed, dict):
                parsed_objects = [parsed]
            elif isinstance(parsed, list):
                parsed_objects = [item for item in parsed if isinstance(item, dict)]
            else:
                parsed_objects = []

            for item in parsed_objects:
                objects.extend(self._normalize_text_tool_use_items(item))

        return objects

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

    def _looks_like_text_tool_use_fragment(self, content: str) -> bool:
        lowered = content.lower()
        return "{" in content and (
            "tool_use" in lowered
            or "tooluse" in lowered
            or "tool_calls" in lowered
            or "tool_name" in lowered
            or ("type" in lowered and "tool" in lowered)
        )

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

    def _to_openai_tool_calls_for_stream(
        self,
        raw_tool_calls: Any,
        argument_buffers: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_tool_calls, list):
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw_tool_calls):
            if not isinstance(item, dict):
                continue

            function = item.get("function")
            if not isinstance(function, dict):
                continue

            call_id = str(item.get("id") or f"call_{index}")
            name = str(function.get("name") or "")
            arguments_text = self._normalize_stream_tool_arguments(
                call_id,
                function.get("arguments", ""),
                argument_buffers,
            )
            if arguments_text is None:
                continue

            normalized.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_text,
                    },
                }
            )

        return normalized

    def _normalize_stream_tool_arguments(
        self,
        call_id: str,
        arguments: Any,
        argument_buffers: dict[str, str],
    ) -> str | None:
        if isinstance(arguments, (dict, list)):
            argument_buffers.pop(call_id, None)
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

        argument_piece = "" if arguments is None else str(arguments)

        if self._is_valid_json(argument_piece):
            argument_buffers.pop(call_id, None)
            return argument_piece

        buffered = argument_buffers.get(call_id, "") + argument_piece
        self._debug_tool_calls(
            "stream.arguments_buffering",
            {"call_id": call_id, "buffer_length": len(buffered)},
        )

        if self._is_valid_json(buffered):
            argument_buffers.pop(call_id, None)
            self._debug_tool_calls(
                "stream.arguments_buffer_complete",
                {"call_id": call_id, "length": len(buffered)},
            )
            return buffered

        argument_buffers[call_id] = buffered
        return None

    def _is_valid_json(self, text: str) -> bool:
        if not isinstance(text, str) or not text:
            return False
        try:
            json.loads(text)
        except (TypeError, ValueError):
            return False
        return True

    def _to_chat_completion(self, model: str, data: dict[str, Any]) -> dict[str, Any]:
        message = data.get("message", {})
        tool_calls = self._to_openai_tool_calls(message.get("tool_calls"))
        content = str(message.get("content") or "")
        if not tool_calls:
            tool_calls = self._text_to_openai_tool_calls(content)
            if tool_calls:
                content = ""
        finish_reason = data.get("done_reason", "stop")
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
