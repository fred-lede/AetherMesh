from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Iterable

from metrics.request_metrics import RequestRecord, request_metrics
from protocols.anthropic.sse_builder import AnthropicSSEBuilder, map_stop_reason
from runtime.orchestration.anthropic_converter import AnthropicRouter
from runtime.orchestration.routing_engine import routing_engine
from runtime.security.auth.token_tracker import record_token_usage
from runtime.security.database import SessionLocal
from runtime.tools.tool_normalizer import NormalizedToolCall

_MAX_PENDING_TOOL_CONTENT = 100000  # chars before flushing as text


def stream_anthropic_with_metrics(
    anthropic_service: AnthropicRouter,
    iterator: Iterable[dict[str, Any] | str],
    model: str,
    provider: str,
    request_id: str,
    start_time: float,
    allowed_tool_names: set[str] | None = None,
    user_id: int | None = None,
    api_key_id: int | None = None,
) -> Iterable[str]:
    total_output_tokens = 0
    last_error = None
    try:
        for item in stream_anthropic(anthropic_service, iterator, model, allowed_tool_names=allowed_tool_names):
            yield item
            if isinstance(item, str) and "content_block_delta" in item and "text_delta" in item:
                data = json.loads(item.split("data: ", 1)[1]) if "data: " in item else {}
                total_output_tokens += len(data.get("delta", {}).get("text", ""))
            if isinstance(item, str) and "error" in item and "event:" in item:
                last_error = item
    except Exception as e:
        logger.warning(f"Upstream stream interrupted for {model}: {type(e).__name__}: {e}")
        last_error = f"Stream interrupted: {e}"
        try:
            from protocols.anthropic.sse_builder import AnthropicSSEBuilder
            yield AnthropicSSEBuilder(model).error(str(last_error))
        except Exception:
            pass
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
        if user_id is not None and last_error is None:
            try:
                db = SessionLocal()
                try:
                    record_token_usage(
                        db, user_id=user_id, api_key_id=api_key_id,
                        input_tokens=0,
                        output_tokens=max(1, total_output_tokens // 4),
                        provider=provider, model=model,
                    )
                finally:
                    db.close()
            except Exception:
                logger.exception("Failed to record streaming token usage")


def stream_anthropic(
    anthropic_service: AnthropicRouter,
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

            choices = item.get("choices")
            if not choices:
                continue
            choice = choices[0]
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
                    if len(combined_content) > _MAX_PENDING_TOOL_CONTENT:
                        content = f"{pending_text_tool_content}{content}"
                        pending_text_tool_content = ""
                    else:
                        pending_text_tool_content = combined_content
                        content = None

            if reasoning:
                yield from sse.emit_thinking_delta(str(reasoning))

            elif content:
                yield from sse.emit_text_delta(str(content))

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
