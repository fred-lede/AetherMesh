from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterable

from providers.base import ProviderError

from runtime.responses.response_models import (
    ResponseObject,
    ResponseStatus,
    ResponseUsage,
    OutputItem,
    OutputItemType,
    ContentPart,
    ContentPartType,
    InputItem,
    InputItemType,
    FunctionCallStatus,
    make_text_output,
    make_function_call_output,
)
from runtime.responses.input_converter import (
    responses_input_to_messages,
    _input_item_to_messages,
    _parse_input_item,
)
from runtime.responses.output_converter import chat_completion_to_response

from runtime.tools.tool_registry import ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_executor import ToolExecutor, tool_executor as default_executor
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("responses.tool_loop")

DEFAULT_MAX_TURNS = 16
DEFAULT_TOOL_TIMEOUT_S = 30


class ResponsesToolLoop:
    """OpenAI Responses API 多輪工具調用循環。

    負責在非 OpenAI 供應商上執行工具調用循環：
    Model → tool_calls → 執行 tools → 注入 results → Model → ... → completed
    """

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        tool_timeout_s: int = DEFAULT_TOOL_TIMEOUT_S,
        parallel_tool_calls: bool = True,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        self._max_turns = max_turns
        self._tool_timeout_s = tool_timeout_s
        self._parallel_tool_calls = parallel_tool_calls
        self._registry = registry or default_registry
        self._executor = executor or default_executor

    def run(
        self,
        *,
        adapter: Any,
        chat_payload: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        instructions: str,
        response_id: str,
        model: str,
        previous_response_id: str = "",
        metadata: dict[str, Any] | None = None,
        input_value: Any = "",
    ) -> ResponseObject:
        """執行多輪工具循環，回傳最終 ResponseObject。

        流程：
        1. 將 Responses input 轉換為 messages
        2. 加入 tools 到 payload
        3. 呼叫 model
        4. 若有 tool_calls → 執行 → 注入 messages → 再 call model
        5. 直到沒有 tool_calls 或超過 max_turns
        """
        messages = self._build_messages(input_value, instructions)
        payload = dict(chat_payload)
        payload["messages"] = messages
        temp_tool_ids: list[str] = []
        if tools:
            payload["tools"] = tools
            temp_tool_ids = self._register_temp_tools(tools)

        usage_accum = self._accumulate_usage()

        try:
            for turn in range(1, self._max_turns + 1):
                logger.debug("Tool loop turn %d/%d", turn, self._max_turns)
                try:
                    completion = adapter.chat(payload)
                except Exception as exc:
                    return self._make_error_response(
                        response_id, model, str(exc),
                        f"provider_error_turn_{turn}",
                        instructions, previous_response_id, metadata,
                    )

                usage_data = completion.get("usage", {})
                usage_accum.add(usage_data)

                choices = completion.get("choices")
                if not choices:
                    return self._make_error_response(
                        response_id, model, "Empty choices in completion",
                        "empty_completion", instructions, previous_response_id, metadata,
                    )
                message = choices[0].get("message", {})
                content = str(message.get("content") or "")
                tool_calls = message.get("tool_calls") or []

                if not tool_calls:
                    resp = chat_completion_to_response(
                        completion, model, response_id,
                        instructions, previous_response_id, metadata,
                    )
                    resp.status = ResponseStatus.COMPLETED
                    resp.id = response_id
                    resp.usage = ResponseUsage(**usage_accum.to_dict())
                    return resp

                logger.debug(
                    "Turn %d: got %d tool_calls",
                    turn, len(tool_calls),
                )

                messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": tool_calls,
                })

                tool_results = self._execute_tool_calls(tool_calls)

                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.call.id,
                        "content": tr.output,
                    })
                    logger.debug(
                        "Tool %s (%s) → %d chars, error=%s",
                        tr.call.name, tr.call.id, len(str(tr.output)), tr.is_error,
                    )

            logger.warning("Tool loop exceeded %d turns, returning partial", self._max_turns)
            return self._build_partial_response(
                response_id, model, messages, usage_accum,
                instructions, previous_response_id, metadata,
            )
        finally:
            for tid in temp_tool_ids:
                self._registry.unregister(tid)

    def run_with_client_tools(
        self,
        *,
        adapter: Any,
        chat_payload: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        instructions: str,
        response_id: str,
        model: str,
        previous_response_id: str = "",
        metadata: dict[str, Any] | None = None,
        input_value: Any = "",
        store: bool = True,
    ) -> tuple[ResponseObject, dict[str, Any] | None]:
        """完整版本：先動態註冊 client tools，執行 loop，後清理。

        回傳 (response_object, requires_action_dict)
        如果 model 回傳 require_action (tool calls but client 未提供 tools),
        第二個值會是 require_action dict。
        """
        messages = self._build_messages(input_value, instructions)
        payload = dict(chat_payload)
        payload["messages"] = messages
        if tools:
            payload["tools"] = tools

        completion = adapter.chat(payload)
        response = chat_completion_to_response(
            completion,
            model,
            response_id,
            instructions,
            previous_response_id,
            metadata,
        )
        response.status = ResponseStatus.COMPLETED
        return response, None

    def run_streaming(
        self,
        *,
        adapter: Any,
        chat_payload: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        instructions: str,
        response_id: str,
        model: str,
        previous_response_id: str = "",
        metadata: dict[str, Any] | None = None,
        input_value: Any = "",
        encoder: Any,
    ) -> Iterable[str]:
        """串流版本：產生 SSE events。

        OpenAI Responses API streaming events:
        - response.created
        - response.output_item.added
        - response.content_part.added
        - response.output_text.delta
        - response.output_text.done
        - response.content_part.done
        - response.output_item.done
        - response.function_call.queue
        - response.function_call.arguments.delta
        - response.function_call.call
        - response.function_call.output
        - response.in_progress
        - response.completed
        """
        messages = self._build_messages(input_value, instructions)
        payload = dict(chat_payload)
        payload["messages"] = messages
        temp_tool_ids: list[str] = []
        if tools:
            payload["tools"] = tools
            temp_tool_ids = self._register_temp_tools(tools)

        yield encoder.encode({
            "type": "response.created",
            "data": {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "model": model,
                    "status": "in_progress",
                },
            },
        })

        text_content: list[str] = []

        try:
            for turn in range(1, self._max_turns + 1):
                logger.debug("Streaming tool loop turn %d/%d", turn, self._max_turns)
                yield encoder.encode({
                    "type": "response.in_progress",
                    "data": {
                        "type": "response.in_progress",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "model": model,
                            "status": "in_progress",
                        },
                    },
                })

                yield encoder.encode({
                    "type": "response.output_item.added",
                    "data": {
                        "type": "response.output_item.added",
                        "response": {"id": response_id},
                        "output_index": turn - 1,
                        "item": {
                            "id": f"item_{uuid.uuid4().hex[:16]}",
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                })

                accumulated_tool_calls: dict[int, dict[str, Any]] = {}
                _tc_item_ids: dict[int, str] = {}
                turn_text_parts: list[str] = []

                yield encoder.encode({
                    "type": "response.content_part.added",
                    "data": {
                        "type": "response.content_part.added",
                        "response": {"id": response_id},
                        "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    },
                })

                try:
                    _chunk_source = adapter.stream(payload)
                except (OSError, ConnectionError, TimeoutError, ProviderError) as stream_exc:
                    logger.error("Streaming tool loop adapter.stream() failed: %s", stream_exc)
                    yield encoder.encode({
                        "type": "response.failed",
                        "data": {
                            "type": "response.failed",
                            "response": {
                                "id": response_id,
                                "object": "response",
                                "model": model,
                                "status": "failed",
                                "error": {"message": str(stream_exc), "type": "server_error", "code": "provider_error"},
                            },
                        },
                    })
                    yield encoder.encode_done()
                    return

                for chunk in _chunk_source:
                    if isinstance(chunk, str):
                        if chunk == "[DONE]":
                            break
                        continue

                    if isinstance(chunk, dict):
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        tc = delta.get("tool_calls")
                        content = delta.get("content", "")
                        finish_reason = choice.get("finish_reason")

                        if content:
                            turn_text_parts.append(content)
                            yield encoder.encode({
                                "type": "response.output_text.delta",
                                "data": {
                                    "type": "response.output_text.delta",
                                    "response": {"id": response_id},
                                    "delta": content,
                                },
                            })

                        if tc and isinstance(tc, list):
                            for tool_call in tc:
                                idx = tool_call.get("index", 0)
                                if idx not in accumulated_tool_calls:
                                    item_id = f"fc_{uuid.uuid4().hex[:16]}"
                                    _tc_item_ids[idx] = item_id
                                    accumulated_tool_calls[idx] = {}
                                    yield from self._queue_function_call(
                                        encoder, response_id, tool_call, item_id=item_id,
                                    )
                                existing = accumulated_tool_calls[idx]
                                for key, value in tool_call.items():
                                    if key == "index":
                                        continue
                                    if key == "function":
                                        func = existing.get("function", {})
                                        for fk, fv in (value or {}).items():
                                            func[fk] = func.get(fk, "") + (fv or "")
                                        existing["function"] = func
                                    elif key == "id":
                                        existing["id"] = existing.get("id", "") + (value or "")
                                    else:
                                        existing[key] = existing.get(key, "") + (value or "")
                                fn = tool_call.get("function", {})
                                partial_args = fn.get("arguments", "")
                                if partial_args:
                                    yield encoder.encode({
                                        "type": "response.function_call.arguments.delta",
                                        "data": {
                                            "type": "response.function_call.arguments.delta",
                                            "response": {"id": response_id},
                                            "item_id": _tc_item_ids.get(idx, ""),
                                            "output_index": turn - 1,
                                            "delta": partial_args,
                                        },
                                    })

                        if finish_reason:
                            break

                for idx in sorted(accumulated_tool_calls.keys()):
                    item_id = _tc_item_ids.get(idx, "")
                    tc_data = accumulated_tool_calls[idx]
                    fn = tc_data.get("function", {})
                    yield encoder.encode({
                        "type": "response.function_call.arguments.done",
                        "data": {
                            "type": "response.function_call.arguments.done",
                            "response": {"id": response_id},
                            "item_id": item_id,
                            "output_index": turn - 1,
                            "call_id": tc_data.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        },
                    })

                turn_text = "".join(turn_text_parts)
                text_content.append(turn_text)

                yield encoder.encode({
                    "type": "response.output_text.done",
                    "data": {
                        "type": "response.output_text.done",
                        "response": {"id": response_id},
                        "text": turn_text,
                    },
                })

                yield encoder.encode({
                    "type": "response.content_part.done",
                    "data": {
                        "type": "response.content_part.done",
                        "response": {"id": response_id},
                    },
                })

                tool_calls_list: list[dict[str, Any]] = []
                for idx in sorted(accumulated_tool_calls.keys()):
                    tc_data = accumulated_tool_calls[idx]
                    tc_call: dict[str, Any] = {
                        "id": tc_data.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                        "type": "function",
                        "function": {
                            "name": tc_data.get("function", {}).get("name", ""),
                            "arguments": tc_data.get("function", {}).get("arguments", "{}"),
                        },
                    }
                    tool_calls_list.append(tc_call)

                if not tool_calls_list:
                    completed_text = "\n".join(text_content)
                    yield encoder.encode({
                        "type": "response.output_item.done",
                        "data": {
                            "type": "response.output_item.done",
                            "response": {"id": response_id},
                            "output_index": turn - 1,
                            "item": {
                                "id": f"item_{uuid.uuid4().hex[:16]}",
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": completed_text}],
                            },
                        },
                    })
                    resp = ResponseObject(
                        id=response_id,
                        model=model,
                        status=ResponseStatus.COMPLETED,
                        instructions=instructions,
                        previous_response_id=previous_response_id,
                        metadata=metadata or {},
                    )
                    if completed_text:
                        resp.output.append(make_text_output(completed_text))
                    yield encoder.encode({
                        "type": "response.completed",
                        "data": {
                            "type": "response.completed",
                            "response": resp.to_dict(),
                        },
                    })
                    yield encoder.encode_done()
                    return

                yield encoder.encode({
                    "type": "response.output_item.done",
                    "data": {
                        "type": "response.output_item.done",
                        "response": {"id": response_id},
                        "output_index": turn - 1,
                    },
                })

                for idx, tc in enumerate(tool_calls_list):
                    fn = tc.get("function", {})
                    fc_item_id = _tc_item_ids.get(idx, tc.get("id", f"fc_{uuid.uuid4().hex[:16]}"))
                    yield encoder.encode({
                        "type": "response.function_call.call",
                        "data": {
                            "type": "response.function_call.call",
                            "response": {"id": response_id},
                            "item_id": fc_item_id,
                            "output_index": turn - 1,
                            "item": {
                                "id": fc_item_id,
                                "type": "function_call",
                                "call_id": tc.get("id", ""),
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", ""),
                                "status": "in_progress",
                            },
                        },
                    })

                tool_results = self._execute_tool_calls(tool_calls_list)

                for tr in tool_results:
                    yield encoder.encode({
                        "type": "response.function_call.output",
                        "data": {
                            "type": "response.function_call.output",
                            "response": {"id": response_id},
                            "call_id": tr.call.id,
                            "output": tr.output,
                        },
                    })

                messages.append({
                    "role": "assistant",
                    "content": turn_text or None,
                    "tool_calls": tool_calls_list,
                })
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.call.id,
                        "content": tr.output,
                    })

            # max_turns exceeded: build partial response with accumulated text
            partial_text = "\n".join(text_content)
            yield encoder.encode({
                "type": "response.completed",
                "data": {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "model": model,
                        "status": "completed",
                        "output": [{"type": "text", "text": partial_text}] if partial_text else [],
                        "usage": {},
                    },
                },
            })
            yield encoder.encode_done()
        finally:
            for tid in temp_tool_ids:
                self._registry.unregister(tid)

    # --- Private helpers ---

    def _build_messages(self, input_value: Any, instructions: str) -> list[dict[str, Any]]:
        messages = responses_input_to_messages(input_value, instructions=instructions)
        return messages

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            call_id = str(tc.get("id", f"call_{uuid.uuid4().hex[:16]}"))
            name = str(fn.get("name", ""))
            raw_args = fn.get("arguments", "{}")

            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {"_raw": str(raw_args)}

            tool_call = ToolCall(
                id=call_id,
                name=name,
                arguments=args,
                source_provider="responses",
            )
            result = self._executor.execute(tool_call, timeout_s=self._tool_timeout_s)
            results.append(result)
        return results

    def _register_temp_tools(self, tools: list[dict[str, Any]]) -> list[str]:
        """Temporarily register client-provided function tools for execution."""
        from runtime.tools.tool_registry import ToolDescriptor, ensure_parameters_schema

        registered: list[str] = []
        for tool in tools:
            fn = tool.get("function", {})
            name = str(fn.get("name", ""))
            if not name or self._registry.resolve(name):
                continue

            def _noop_handler(call: ToolCall) -> ToolResult:
                return ToolResult(
                    call=call,
                    output=f"Client tool '{call.name}' has no implementation in AetherMesh tool registry. "
                           "This should not happen if tool was properly registered.",
                    is_error=True,
                )

            descriptor = ToolDescriptor(
                name=name,
                description=str(fn.get("description", "")),
                input_schema=ensure_parameters_schema(fn.get("parameters")),
                handler=_noop_handler,
                source="client",
            )
            self._registry.register(descriptor)
            registered.append(name)
        return registered

    def _accumulate_usage(self) -> _UsageAccumulator:
        return _UsageAccumulator()

    def _make_completed_response(
        self,
        response_id: str,
        response_object: ResponseObject,
    ) -> ResponseObject:
        response_object.status = ResponseStatus.COMPLETED
        response_object.id = response_id
        return response_object

    def _make_error_response(
        self,
        response_id: str,
        model: str,
        message: str,
        code: str,
        instructions: str,
        previous_response_id: str,
        metadata: dict[str, Any] | None,
    ) -> ResponseObject:
        resp = ResponseObject(
            id=response_id,
            model=model,
            status=ResponseStatus.FAILED,
            instructions=instructions,
            previous_response_id=previous_response_id,
            error={"message": message, "type": "server_error", "code": code},
            metadata=metadata or {},
        )
        return resp

    def _build_partial_response(
        self,
        response_id: str,
        model: str,
        messages: list[dict[str, Any]],
        usage: _UsageAccumulator,
        instructions: str,
        previous_response_id: str,
        metadata: dict[str, Any] | None,
    ) -> ResponseObject:
        last_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_text = msg["content"]
                break

        resp = ResponseObject(
            id=response_id,
            model=model,
            status=ResponseStatus.COMPLETED,
            instructions=instructions,
            previous_response_id=previous_response_id,
            metadata=metadata or {},
            usage=ResponseUsage(**usage.to_dict()),
        )
        if last_text:
            resp.output.append(make_text_output(last_text))
        return resp

    def _assemble_completion_from_stream(
        self,
        adapter: Any,
        payload: dict[str, Any],
        response_id: str,
        model: str,
    ) -> dict[str, Any] | None:
        """For non-streaming-capable adapters, fall back to synchronous call."""
        try:
            return adapter.chat(payload)
        except Exception as exc:
            logger.warning("Could not assemble completion: %s", exc)
            return None

    def _queue_function_call(
        self,
        encoder: Any,
        response_id: str,
        tool_call: dict[str, Any],
        item_id: str = "",
    ) -> Iterable[str]:
        fn = tool_call.get("function", {})
        fc_item_id = item_id or tool_call.get("id", "") or f"call_{uuid.uuid4().hex[:16]}"
        yield encoder.encode({
            "type": "response.function_call.queue",
            "data": {
                "type": "response.function_call.queue",
                "response": {"id": response_id},
                "item": {
                    "id": fc_item_id,
                    "type": "function_call",
                    "call_id": tool_call.get("id", "") or f"call_{uuid.uuid4().hex[:16]}",
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", ""),
                    "status": "in_progress",
                },
            },
        })


class _UsageAccumulator:
    def __init__(self) -> None:
        self._input = 0
        self._output = 0

    def add(self, usage: dict[str, Any]) -> None:
        self._input += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        self._output += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self._input,
            "output_tokens": self._output,
            "total_tokens": self._input + self._output,
        }


responses_tool_loop = ResponsesToolLoop()
