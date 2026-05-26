from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Iterable

import requests
from fastapi import HTTPException

from datetime import UTC, datetime

from config.settings import settings

logger = logging.getLogger("openai_handler")
from metrics.request_metrics import request_metrics
from providers.base import ProviderError
from providers.http_client import get_session
from runtime.orchestration.capabilities import required_openai_capabilities
from runtime.orchestration.provider_router import (
    adapter,
    capabilities_for_model,
    find_registry_model,
    local_ollama_fallback,
    provider_for_model,
)
from runtime.orchestration.routing_engine import routing_engine
from runtime.memory import memory_manager
from runtime.security.auth.token_tracker import record_token_usage
from runtime.security.database import SessionLocal
from runtime.tools.builtin.web_search import (
    extract_search_query,
    latest_user_text,
    run_web_search,
)
from runtime.tools.content_blocks import content_part_to_text_and_images, normalize_image_ref


class RouterService:
    def __init__(self) -> None:
        self.registry = settings.model_registry()

    def list_models(self) -> dict[str, Any]:
        models = []
        for model in self.registry.get("models", []):
            models.append(
                {
                    "id": model["name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": model.get("provider", "ollama"),
                    "metadata": {
                        "worker_ports": model.get("worker_ports", []),
                        "worker_bindings": model.get("worker_bindings", []),
                        "capabilities": model.get("capabilities", []),
                    },
                }
            )
        alias_prefix = settings.model_alias_prefix()
        for alias, target in settings.model_alias_entries().items():
            model_id = f"{alias_prefix}/{alias}" if alias_prefix else alias
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "alias",
                    "metadata": {
                        "target": target,
                        "capabilities": self._capabilities_for_model(target),
                    },
                }
            )
        return {"object": "list", "data": models}

    def _inject_web_search(self, payload: dict[str, Any]) -> None:
        if not settings.web_tools_auto_search:
            return
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return
        tools = payload.get("tools")
        if isinstance(tools, list):
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                if fn.get("name") in ("web_search", "web_fetch"):
                    return
        query = extract_search_query(latest_user_text(payload))
        if not query:
            return
        try:
            results = run_web_search(query, settings.web_search_max_results, settings.web_tool_timeout_s)
            if not results:
                return
            now = datetime.now(UTC).strftime("%Y-%m-%d")
            lines = [
                f"Today's date: {now}",
                f"Web search results for: {query}",
                "",
            ]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}\n   URL: {r['url']}")
            lines.extend([
                "",
                "Answer the user's question based on the web search results above.",
                "Include the source URL when you reference information from a result.",
            ])
            context = "\n".join(lines)
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    msg["content"] = context + "\n\n" + str(msg.get("content", ""))
                    return
            messages.insert(0, {"role": "system", "content": context})
        except Exception as exc:
            logger.warning("Auto web search failed: %s", exc)

    def handle_chat(self, payload: dict[str, Any], user_id: int | None = None, api_key_id: int | None = None) -> dict[str, Any]:
        self._inject_web_search(payload)
        prepared_payload = self._apply_generation_defaults(payload)
        if self._is_async_requested(prepared_payload):
            return self._enqueue_async_task("/v1/chat/completions", prepared_payload)
        provider, worker = self._resolve_provider_and_worker(prepared_payload, allow_queue=False)
        effective_payload = self._normalize_payload_for_provider(prepared_payload, provider)
        adapter = self._adapter(provider, worker)
        started = time.perf_counter()
        error = False
        error_code = ""
        try:
            response = adapter.chat(effective_payload)
            if user_id is not None:
                usage = response.get("usage") or {}
                self._record_token_usage(user_id, api_key_id,
                                         usage.get("prompt_tokens", 0),
                                         usage.get("completion_tokens", 0),
                                         provider, effective_payload.get("model", ""))
            usage = response.get("usage") or {}
            memory_manager.episodic.record(
                model=effective_payload.get("model", ""),
                provider=provider,
                duration_ms=(time.perf_counter() - started) * 1000,
                success=True,
                token_count={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            )
            return response
        except ProviderError as exc:
            error = True
            error_code = getattr(exc, "code", "") or self._classify_error_text(str(exc))
            fallback = self._local_ollama_fallback(prepared_payload) if provider != "ollama" else None
            if fallback is not None and self._should_fallback_provider_error(exc):
                self._finalize_request(
                    endpoint="/v1/chat/completions",
                    payload=effective_payload,
                    provider=provider,
                    worker=worker,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=True,
                    error_code=error_code,
                )
                fallback_payload, fallback_worker = fallback
                provider = "ollama"
                worker = fallback_worker
                effective_payload = fallback_payload
                adapter_instance = self._adapter(provider, worker)
                error = False
                error_code = ""
                try:
                    fallback_response = adapter_instance.chat(effective_payload)
                    memory_manager.episodic.record(
                        model=effective_payload.get("model", ""),
                        provider=provider,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=True,
                    )
                    return fallback_response
                except ProviderError as fallback_exc:
                    error = True
                    error_code = getattr(fallback_exc, "code", "") or self._classify_error_text(str(fallback_exc))
                    memory_manager.episodic.record(
                        model=effective_payload.get("model", ""),
                        provider=provider,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=False,
                        error=str(fallback_exc)[:200],
                    )
                    raise self._provider_http_error(fallback_exc, code=error_code) from fallback_exc
            memory_manager.episodic.record(
                model=effective_payload.get("model", ""),
                provider=provider,
                duration_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=str(exc)[:200],
            )
            raise self._provider_http_error(exc, code=error_code) from exc
        except requests.Timeout as exc:
            error = True
            error_code = "provider_timeout"
            fallback = self._local_ollama_fallback(prepared_payload) if provider != "ollama" else None
            if fallback is not None:
                self._finalize_request(
                    endpoint="/v1/chat/completions",
                    payload=effective_payload,
                    provider=provider,
                    worker=worker,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=True,
                    error_code=error_code,
                )
                fallback_payload, fallback_worker = fallback
                provider = "ollama"
                worker = fallback_worker
                effective_payload = fallback_payload
                adapter_instance = self._adapter(provider, worker)
                error = False
                error_code = ""
                try:
                    fallback_response = adapter_instance.chat(effective_payload)
                    memory_manager.episodic.record(
                        model=effective_payload.get("model", ""),
                        provider=provider,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=True,
                    )
                    return fallback_response
                except ProviderError as fallback_exc:
                    error = True
                    error_code = getattr(fallback_exc, "code", "") or self._classify_error_text(str(fallback_exc))
                    memory_manager.episodic.record(
                        model=effective_payload.get("model", ""),
                        provider=provider,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=False,
                        error=str(fallback_exc)[:200],
                    )
                    raise self._provider_http_error(fallback_exc, code=error_code) from fallback_exc
            memory_manager.episodic.record(
                model=effective_payload.get("model", ""),
                provider=provider,
                duration_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=str(exc)[:200],
            )
            raise self._as_http_error(status_code=504, code=error_code, message=str(exc)) from exc
        except requests.RequestException as exc:
            error = True
            error_code = "provider_unreachable"
            memory_manager.episodic.record(
                model=effective_payload.get("model", ""),
                provider=provider,
                duration_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=str(exc)[:200],
            )
            raise self._as_http_error(status_code=502, code=error_code, message=str(exc)) from exc
        finally:
            self._finalize_request(
                endpoint="/v1/chat/completions",
                payload=effective_payload,
                provider=provider,
                worker=worker,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error,
                error_code=error_code,
            )

    def handle_streaming_chat(self, payload: dict[str, Any], user_id: int | None = None, api_key_id: int | None = None) -> Iterable[dict[str, Any] | str]:
        self._inject_web_search(payload)
        prepared_payload = self._apply_generation_defaults(payload)
        provider, worker = self._resolve_provider_and_worker(prepared_payload, allow_queue=False)
        effective_payload = self._normalize_payload_for_provider(prepared_payload, provider)
        adapter = self._adapter(provider, worker)
        started = time.perf_counter()
        state = {"provider": provider, "worker": worker, "payload": effective_payload}

        def wrapped() -> Iterable[dict[str, Any] | str]:
            error = False
            error_code = ""
            last_chunk = None
            try:
                for item in adapter.stream(effective_payload):
                    if isinstance(item, dict):
                        last_chunk = item
                    yield item
            except ProviderError as exc:
                error = True
                error_code = getattr(exc, "code", "") or self._classify_error_text(str(exc))
                fallback = self._local_ollama_fallback(prepared_payload) if provider != "ollama" else None
                if fallback is not None and self._should_fallback_provider_error(exc):
                    self._finalize_request(
                        endpoint="/v1/chat/completions",
                        payload=effective_payload,
                        provider=provider,
                        worker=worker,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error=True,
                        error_code=error_code,
                    )
                    fallback_payload, fallback_worker = fallback
                    state["provider"] = "ollama"
                    state["worker"] = fallback_worker
                    state["payload"] = fallback_payload
                    error = False
                    error_code = ""
                    try:
                        yield from self._adapter("ollama", fallback_worker).stream(fallback_payload)
                    except ProviderError as fallback_exc:
                        error = True
                        memory_manager.episodic.record(
                            model=state["payload"].get("model", ""),
                            provider=str(state["provider"]),
                            duration_ms=(time.perf_counter() - started) * 1000,
                            success=False,
                            error=str(fallback_exc)[:200],
                        )
                        if last_chunk is None:
                            yield {
                                "error": {
                                    "message": str(fallback_exc),
                                    "type": "provider_error",
                                    "code": getattr(fallback_exc, "code", "") or self._classify_error_text(str(fallback_exc)),
                                }
                            }
                    return
                error_type = "rate_limit_error" if getattr(exc, "status_code", None) == 429 else "provider_error"
                payload = {"message": str(exc), "type": error_type, "code": error_code}
                retry_after = getattr(exc, "retry_after", None)
                if retry_after:
                    payload["retry_after"] = retry_after
                memory_manager.episodic.record(
                    model=state["payload"].get("model", ""),
                    provider=str(state["provider"]),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    success=False,
                    error=str(exc)[:200],
                )
                yield {"error": payload}
            except requests.Timeout as exc:
                error = True
                error_code = "provider_timeout"
                fallback = self._local_ollama_fallback(prepared_payload) if provider != "ollama" else None
                if fallback is not None:
                    self._finalize_request(
                        endpoint="/v1/chat/completions",
                        payload=effective_payload,
                        provider=provider,
                        worker=worker,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error=True,
                        error_code=error_code,
                    )
                    fallback_payload, fallback_worker = fallback
                    state["provider"] = "ollama"
                    state["worker"] = fallback_worker
                    state["payload"] = fallback_payload
                    error = False
                    error_code = ""
                    try:
                        yield from self._adapter("ollama", fallback_worker).stream(fallback_payload)
                    except ProviderError as fallback_exc:
                        error = True
                        memory_manager.episodic.record(
                            model=state["payload"].get("model", ""),
                            provider=str(state["provider"]),
                            duration_ms=(time.perf_counter() - started) * 1000,
                            success=False,
                            error=str(fallback_exc)[:200],
                        )
                        if last_chunk is None:
                            yield {
                                "error": {
                                    "message": str(fallback_exc),
                                    "type": "provider_error",
                                    "code": getattr(fallback_exc, "code", "") or self._classify_error_text(str(fallback_exc)),
                                }
                            }
                    return
                memory_manager.episodic.record(
                    model=state["payload"].get("model", ""),
                    provider=str(state["provider"]),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    success=False,
                    error=str(exc)[:200],
                )
                yield {"error": {"message": str(exc), "type": "provider_timeout", "code": error_code}}
            except requests.RequestException as exc:
                error = True
                error_code = "provider_unreachable"
                memory_manager.episodic.record(
                    model=state["payload"].get("model", ""),
                    provider=str(state["provider"]),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    success=False,
                    error=str(exc)[:200],
                )
                yield {"error": {"message": str(exc), "type": "provider_unreachable", "code": error_code}}
            finally:
                self._finalize_request(
                    endpoint="/v1/chat/completions",
                    payload=state["payload"],
                    provider=str(state["provider"]),
                    worker=state["worker"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=error,
                    error_code=error_code,
                )
                if not error:
                    memory_manager.episodic.record(
                        model=state["payload"].get("model", ""),
                        provider=str(state["provider"]),
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=True,
                    )
                if not error and user_id is not None and isinstance(last_chunk, dict):
                    usage = last_chunk.get("usage") or {}
                    pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                    ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                    if pt or ct:
                        self._record_token_usage(
                            user_id, api_key_id,
                            input_tokens=pt, output_tokens=ct,
                            provider=str(state["provider"]),
                            model=state["payload"].get("model", ""),
                        )

        return wrapped()

    def handle_responses(self, payload: dict[str, Any], user_id: int | None = None, api_key_id: int | None = None) -> dict[str, Any]:
        from runtime.responses.response_models import ResponseObject, ResponseStatus
        from runtime.responses.input_converter import responses_input_to_messages
        from runtime.responses.output_converter import chat_completion_to_response, error_response
        from runtime.responses.response_runtime import response_runtime

        prepared_payload = self._apply_generation_defaults(payload)
        if self._is_async_requested(prepared_payload):
            return self._enqueue_async_task("/v1/responses", prepared_payload)

        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        model = str(payload.get("model", ""))
        instructions = str(payload.get("instructions", ""))
        previous_response_id = str(payload.get("previous_response_id", ""))
        metadata = payload.get("metadata", {})
        store = bool(payload.get("store", True))
        input_value = payload.get("input", "")
        stream = payload.get("stream", False)

        messages = responses_input_to_messages(input_value, instructions=instructions)
        self._trace_responses(
            "input_converted",
            response_id=response_id,
            payload=payload,
            messages=messages,
        )

        chat_payload = dict(payload)
        chat_payload["messages"] = messages
        chat_payload.pop("input", None)
        chat_payload.pop("instructions", None)
        chat_payload.pop("previous_response_id", None)
        chat_payload.pop("store", None)
        chat_payload.pop("stream", None)

        provider, worker = self._resolve_provider_and_worker(chat_payload, allow_queue=False)
        self._trace_responses(
            "route_selected",
            response_id=response_id,
            provider=provider,
            worker=worker,
            effective_payload=chat_payload,
        )

        original_payload = dict(payload)
        original_payload.pop("stream", None)
        effective_payload = self._normalize_payload_for_provider(chat_payload, provider)
        adapter_instance = self._adapter(provider, worker)
        started = time.perf_counter()
        error = False
        error_code = ""
        try:
            if provider == "openai":
                result = adapter_instance.responses(original_payload)
                if store:
                    from runtime.responses.response_models import ResponseObject
                    from runtime.responses.response_models import ResponseUsage
                    resp = ResponseObject(
                        id=result.get("id", response_id),
                        model=result.get("model", model),
                        status=ResponseStatus(result.get("status", "completed")),
                        instructions=instructions,
                        previous_response_id=previous_response_id,
                        metadata=metadata,
                    )
                    usage_data = result.get("usage", {})
                    resp.usage = ResponseUsage(
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )
                    output_items = result.get("output", [])
                    from runtime.responses.response_models import OutputItem
                    for oi in output_items:
                        item = OutputItem(
                            id=oi.get("id", f"item_{uuid.uuid4().hex[:16]}"),
                            role=oi.get("role", "assistant"),
                        )
                        item.type = oi.get("type", "message")
                        content_parts = oi.get("content", [])
                        from runtime.responses.response_models import ContentPart, ContentPartType
                        for cp in content_parts:
                            part = ContentPart(
                                type=ContentPartType.OUTPUT_TEXT,
                                text=cp.get("text", cp.get("refusal", "")),
                            )
                            item.content.append(part)
                        if oi.get("type") == "tool_call":
                            item.tool_call_id = oi.get("tool_call_id", "")
                            item.tool_name = oi.get("tool_name", "")
                            item.arguments = oi.get("arguments", "")
                        resp.output.append(item)
                    response_runtime.register(resp)
                usage_data = result.get("usage", {})
                self._record_token_usage(user_id, api_key_id,
                                         usage_data.get("input_tokens", 0),
                                         usage_data.get("output_tokens", 0),
                                         provider, model)
                return result
            else:
                tools = self._ensure_openai_tools(payload.get("tools") or [])
                max_turns = int(payload.get("max_turns", self._resolve_max_turns()))
                parallel_tool_calls = payload.get("parallel_tool_calls", True)

                if tools and provider != "openai":
                    from runtime.responses.tool_loop import responses_tool_loop
                    loop = responses_tool_loop
                    if max_turns != loop._max_turns or parallel_tool_calls != loop._parallel_tool_calls:
                        loop = responses_tool_loop.__class__(
                            max_turns=max_turns,
                            parallel_tool_calls=parallel_tool_calls,
                        )
                    response_object = loop.run(
                        adapter=adapter_instance,
                        chat_payload=effective_payload,
                        tools=tools,
                        instructions=instructions,
                        response_id=response_id,
                        model=model,
                        previous_response_id=previous_response_id,
                        metadata=metadata,
                        input_value=input_value,
                    )
                    self._record_response_usage(user_id, api_key_id,
                                                response_object, provider, model)
                    if store:
                        response_runtime.register(response_object)
                    return response_object.to_dict()
                else:
                    completion = adapter_instance.chat(effective_payload)
                    self._trace_responses(
                        "provider_completion",
                        response_id=response_id,
                        provider=provider,
                        completion=completion,
                    )
                    usage = completion.get("usage") or {}
                    self._record_token_usage(user_id, api_key_id,
                                             usage.get("prompt_tokens", 0),
                                             usage.get("completion_tokens", 0),
                                             provider, effective_payload.get("model", ""))
                    response = chat_completion_to_response(
                        completion=completion,
                        model=model,
                        response_id=response_id,
                        instructions=instructions,
                        previous_response_id=previous_response_id,
                        metadata=metadata,
                    )
                    if store:
                        response_runtime.register(response)
                    self._trace_responses(
                        "response_converted",
                        response_id=response_id,
                        response=response,
                    )
                    return response.to_dict()
        except ProviderError as exc:
            error = True
            error_code = getattr(exc, "code", "") or self._classify_error_text(str(exc))
            fallback = self._local_ollama_fallback(prepared_payload) if provider != "ollama" else None
            if fallback is not None and self._should_fallback_provider_error(exc):
                self._finalize_request(
                    endpoint="/v1/responses",
                    payload=effective_payload,
                    provider=provider,
                    worker=worker,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=True,
                    error_code=error_code,
                )
                fallback_payload, fallback_worker = fallback
                provider = "ollama"
                worker = fallback_worker
                effective_payload = fallback_payload
                adapter_instance = self._adapter(provider, worker)
                error = False
                error_code = ""
                try:
                    completion = adapter_instance.chat(effective_payload)
                    response = chat_completion_to_response(
                        completion=completion,
                        model=model,
                        response_id=response_id,
                    )
                    if store:
                        response_runtime.register(response)
                    return response.to_dict()
                except ProviderError as fallback_exc:
                    error = True
                    error_code = getattr(fallback_exc, "code", "") or self._classify_error_text(str(fallback_exc))
                    err_resp = error_response(model, str(fallback_exc), error_code, response_id)
                    if store:
                        response_runtime.register(err_resp)
                    return err_resp.to_dict()
            err_resp = error_response(model, str(exc), error_code, response_id)
            if store:
                response_runtime.register(err_resp)
            raise self._provider_http_error(exc, code=error_code) from exc
        except requests.Timeout as exc:
            error = True
            error_code = "provider_timeout"
            fallback = self._local_ollama_fallback(prepared_payload) if provider != "ollama" else None
            if fallback is not None:
                self._finalize_request(
                    endpoint="/v1/responses",
                    payload=effective_payload,
                    provider=provider,
                    worker=worker,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=True,
                    error_code=error_code,
                )
                fallback_payload, fallback_worker = fallback
                provider = "ollama"
                worker = fallback_worker
                effective_payload = fallback_payload
                adapter_instance = self._adapter(provider, worker)
                error = False
                error_code = ""
                try:
                    completion = adapter_instance.chat(effective_payload)
                    response = chat_completion_to_response(completion=completion, model=model, response_id=response_id)
                    if store:
                        response_runtime.register(response)
                    return response.to_dict()
                except ProviderError as fallback_exc:
                    error = True
                    error_code = getattr(fallback_exc, "code", "") or self._classify_error_text(str(fallback_exc))
                    err_resp = error_response(model, str(fallback_exc), error_code, response_id)
                    if store:
                        response_runtime.register(err_resp)
                    return err_resp.to_dict()
            raise self._as_http_error(status_code=504, code=error_code, message=str(exc)) from exc
        except requests.RequestException as exc:
            error = True
            error_code = "provider_unreachable"
            raise self._as_http_error(status_code=502, code=error_code, message=str(exc)) from exc
        finally:
            self._finalize_request(
                endpoint="/v1/responses",
                payload=effective_payload,
                provider=provider,
                worker=worker,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error,
                error_code=error_code,
            )

    def handle_streaming_responses(self, payload: dict[str, Any], user_id: int | None = None, api_key_id: int | None = None) -> Iterable[str]:
        from runtime.responses.input_converter import responses_input_to_messages
        from runtime.responses.response_stream import wrap_streaming_chunks, response_stream_encoder
        from runtime.responses.response_runtime import response_runtime

        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        model = str(payload.get("model", ""))
        instructions = str(payload.get("instructions", ""))
        previous_response_id = str(payload.get("previous_response_id", ""))
        metadata = payload.get("metadata", {})
        input_value = payload.get("input", "")
        messages = responses_input_to_messages(input_value, instructions=instructions)
        self._trace_responses(
            "stream.input_converted",
            response_id=response_id,
            payload=payload,
            messages=messages,
        )

        chat_payload = dict(payload)
        chat_payload["messages"] = messages
        chat_payload.pop("input", None)
        chat_payload.pop("instructions", None)
        chat_payload.pop("previous_response_id", None)
        chat_payload.pop("store", None)
        chat_payload.pop("stream", None)

        provider, worker = self._resolve_provider_and_worker(chat_payload, allow_queue=False)
        effective_payload = self._normalize_payload_for_provider(chat_payload, provider)
        started = time.perf_counter()
        self._trace_responses(
            "stream.route_selected",
            response_id=response_id,
            provider=provider,
            worker=worker,
            effective_payload=effective_payload,
        )
        outer_state: dict[str, Any] = {"provider": provider, "worker": worker, "payload": effective_payload}

        def _with_tracking(raw_chunks) -> Iterable[dict[str, Any] | str]:
            last_chunk = None
            for chunk in raw_chunks:
                if isinstance(chunk, dict):
                    last_chunk = chunk
                yield chunk
            if user_id is not None and isinstance(last_chunk, dict):
                usage = last_chunk.get("usage") or {}
                pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                if pt or ct:
                    model_name = last_chunk.get("model", model)
                    self._record_token_usage(
                        user_id, api_key_id,
                        input_tokens=pt, output_tokens=ct,
                        provider=outer_state["provider"], model=model_name,
                    )

        def _finalize(error: bool, error_code: str = "") -> None:
            err_msg = error_code or ("provider_error" if error else "")
            self._finalize_request(
                endpoint="/v1/responses",
                payload=outer_state["payload"],
                provider=str(outer_state["provider"]),
                worker=outer_state["worker"],
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error,
                error_code=err_msg,
            )

        def error_yield(exc: Exception, code: str = "server_error") -> Iterable[str]:
            self._trace_responses(
                "stream.failed",
                response_id=response_id,
                provider=outer_state["provider"],
                worker=outer_state["worker"],
                effective_payload=outer_state["payload"],
                error=str(exc),
            )
            yield response_stream_encoder.encode({
                "type": "response.failed",
                "data": {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "model": model,
                        "status": "failed",
                        "error": {"message": str(exc), "type": "server_error", "code": code},
                    },
                },
            })
            yield response_stream_encoder.encode_done()

        def stream_yield() -> Iterable[str]:
            stream_error = False
            try:
                if outer_state["provider"] == "openai":
                    openai_payload = dict(payload)
                    openai_payload["stream"] = True
                    try:
                        adapter_instance = self._adapter(outer_state["provider"], outer_state["worker"])
                        raw_chunks = adapter_instance.stream(openai_payload)
                        yield from wrap_streaming_chunks(_with_tracking(raw_chunks), response_id, model)
                        return
                    except ProviderError as exc:
                        fallback = self._local_ollama_fallback(chat_payload)
                        if fallback is None or not self._should_fallback_provider_error(exc):
                            stream_error = True
                            yield from error_yield(exc, self._classify_error_text(str(exc)))
                            return
                        effective_payload, worker = fallback
                        outer_state["provider"] = "ollama"
                        outer_state["worker"] = worker
                        outer_state["payload"] = effective_payload
                        self._trace_responses(
                            "stream.openai_fallback",
                            response_id=response_id,
                            provider=outer_state["provider"],
                            worker=outer_state["worker"],
                            effective_payload=outer_state["payload"],
                            error=str(exc),
                        )

                tools = self._ensure_openai_tools(payload.get("tools") or [])
                adapter_instance = self._adapter(outer_state["provider"], outer_state["worker"])
                if tools:
                    from runtime.responses.tool_loop import responses_tool_loop
                    loop = responses_tool_loop
                    max_turns = int(payload.get("max_turns", self._resolve_max_turns()))
                    parallel_tool_calls = payload.get("parallel_tool_calls", True)
                    if max_turns != loop._max_turns or parallel_tool_calls != loop._parallel_tool_calls:
                        from runtime.responses.tool_loop import ResponsesToolLoop, DEFAULT_MAX_TURNS, DEFAULT_TOOL_TIMEOUT_S
                        loop = ResponsesToolLoop(
                            max_turns=max_turns,
                            parallel_tool_calls=parallel_tool_calls,
                        )
                    yield from loop.run_streaming(
                        adapter=adapter_instance,
                        chat_payload=outer_state["payload"],
                        tools=tools,
                        instructions=instructions,
                        response_id=response_id,
                        model=model,
                        previous_response_id=previous_response_id,
                        metadata=metadata,
                        input_value=input_value,
                        encoder=response_stream_encoder,
                    )
                    store_flag = bool(payload.get("store", True))
                    if store_flag:
                        final_resp = ResponseObject(
                            id=response_id,
                            model=model,
                            status=ResponseStatus.COMPLETED,
                            instructions=instructions,
                            previous_response_id=previous_response_id,
                            metadata=metadata,
                        )
                        response_runtime.register(final_resp)
                else:
                    raw_chunks = adapter_instance.stream(outer_state["payload"])
                    yield from wrap_streaming_chunks(_with_tracking(raw_chunks), response_id, model)
            except Exception as exc:
                stream_error = True
                yield from error_yield(exc)
            finally:
                _finalize(stream_error)

        return stream_yield()

    def get_response(self, response_id: str) -> dict[str, Any]:
        from runtime.responses.response_runtime import response_runtime
        resp = response_runtime.get(response_id)
        if not resp:
            raise self._as_http_error(status_code=404, code="not_found", message=f"Response {response_id} not found")
        return resp.to_dict()

    def delete_response(self, response_id: str) -> dict[str, Any]:
        from runtime.responses.response_runtime import response_runtime
        if response_runtime.delete(response_id):
            return {"id": response_id, "object": "response", "deleted": True}
        raise self._as_http_error(status_code=404, code="not_found", message=f"Response {response_id} not found")

    def update_response(self, response_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        from runtime.responses.response_runtime import response_runtime
        resp = response_runtime.get(response_id)
        if not resp:
            raise self._as_http_error(status_code=404, code="not_found", message=f"Response {response_id} not found")
        if "metadata" in payload:
            resp.metadata.update(payload["metadata"])
        if "instructions" in payload:
            resp.instructions = str(payload["instructions"])
        return resp.to_dict()

    def handle_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._is_async_requested(payload):
            return self._enqueue_async_task("/v1/embeddings", payload)
        provider, worker = self._resolve_provider_and_worker(payload, allow_queue=False)
        adapter = self._adapter(provider, worker)
        started = time.perf_counter()
        error = False
        error_code = ""
        try:
            return adapter.embeddings(payload)
        except ProviderError as exc:
            error = True
            error_code = getattr(exc, "code", "") or self._classify_error_text(str(exc))
            raise self._provider_http_error(exc, code=error_code) from exc
        except requests.Timeout as exc:
            error = True
            error_code = "provider_timeout"
            raise self._as_http_error(status_code=504, code=error_code, message=str(exc)) from exc
        except requests.RequestException as exc:
            error = True
            error_code = "provider_unreachable"
            raise self._as_http_error(status_code=502, code=error_code, message=str(exc)) from exc
        finally:
            self._finalize_request(
                endpoint="/v1/embeddings",
                payload=payload,
                provider=provider,
                worker=worker,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error,
                error_code=error_code,
            )

    def handle_rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._is_async_requested(payload):
            return self._enqueue_async_task("/v1/rerank", payload)
        provider, worker = self._resolve_provider_and_worker(payload, allow_queue=False)
        adapter = self._adapter(provider, worker)
        started = time.perf_counter()
        error = False
        error_code = ""
        try:
            return adapter.rerank(payload)
        except ProviderError as exc:
            error = True
            error_code = getattr(exc, "code", "") or self._classify_error_text(str(exc))
            raise self._provider_http_error(exc, code=error_code) from exc
        except requests.Timeout as exc:
            error = True
            error_code = "provider_timeout"
            raise self._as_http_error(status_code=504, code=error_code, message=str(exc)) from exc
        except requests.RequestException as exc:
            error = True
            error_code = "provider_unreachable"
            raise self._as_http_error(status_code=502, code=error_code, message=str(exc)) from exc
        finally:
            self._finalize_request(
                endpoint="/v1/rerank",
                payload=payload,
                provider=provider,
                worker=worker,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error,
                error_code=error_code,
            )

    def _as_http_error(self, *, status_code: int, code: str, message: str) -> HTTPException:
        return HTTPException(status_code=status_code, detail={"code": code, "message": message})

    def _provider_http_error(self, exc: ProviderError, *, code: str) -> HTTPException:
        status_code = int(getattr(exc, "status_code", None) or 502)
        retry_after = getattr(exc, "retry_after", None)
        detail: dict[str, Any] = {"code": code, "message": str(exc)}
        headers = None
        if retry_after:
            detail["retry_after"] = retry_after
            headers = {"Retry-After": str(retry_after)}
        return HTTPException(status_code=status_code, detail=detail, headers=headers)

    def _should_fallback_provider_error(self, exc: ProviderError) -> bool:
        status_code = int(getattr(exc, "status_code", None) or 0)
        code = str(getattr(exc, "code", "") or "") or self._classify_error_text(str(exc))
        return status_code in {404, 429, 502, 503, 504} or code in {
            "model_not_found",
            "provider_rate_limited",
            "provider_overloaded",
            "provider_timeout",
            "provider_unconfigured",
        }

    def _classify_error_text(self, text: str) -> str:
        lowered = str(text or "").lower()
        if "404" in lowered or "not found" in lowered:
            return "model_not_found"
        if "502" in lowered or "bad gateway" in lowered:
            return "provider_overloaded"
        if "timed out" in lowered or "timeout" in lowered or "context deadline exceeded" in lowered:
            return "provider_timeout"
        if "model runner has unexpectedly stopped" in lowered:
            return "runner_stopped"
        if "connection refused" in lowered or "max retries exceeded" in lowered or "name or service not known" in lowered:
            return "provider_unreachable"
        if "api_key is not configured" in lowered or "api key is not configured" in lowered:
            return "provider_unconfigured"
        if "no worker available" in lowered or "no worker was assigned" in lowered:
            return "worker_unavailable"
        return "provider_error"

    def _local_ollama_fallback(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        required = self._required_capabilities(payload)
        fallback = self._configured_ollama_fallback(required) or self._first_ollama_fallback(required)
        if fallback is None:
            return None

        model_name, worker = fallback
        fallback_payload = dict(payload)
        fallback_payload.pop("provider", None)
        fallback_payload["model"] = model_name
        return self._normalize_payload_for_provider(fallback_payload, "ollama"), worker

    def _configured_ollama_fallback(self, required: set[str]) -> tuple[str, dict[str, Any]] | None:
        model_name = settings.ollama_fallback_model()
        if not model_name:
            return None
        return self._ollama_model_for_fallback(model_name, required)

    def _first_ollama_fallback(self, required: set[str]) -> tuple[str, dict[str, Any]] | None:
        return local_ollama_fallback(required, self.registry)

    def _ollama_model_for_fallback(self, model_name: str, required: set[str]) -> tuple[str, dict[str, Any]] | None:
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
            if not model.get("worker_bindings"):
                return None
            return self._ollama_model_tuple(model)
        return None

    def _prioritize_warm_models(self, models: list[str]) -> list[str]:
        warm = []
        cold = []
        for m in models:
            if self._is_warm(m):
                warm.append(m)
            else:
                cold.append(m)
        return warm + cold

    @staticmethod
    def _is_warm(model: str) -> bool:
        try:
            from runtime.gpu.warm_pool import warm_pool
            for entry in warm_pool.warm_models():
                if entry.model_name == model:
                    return True
        except ImportError:
            pass
        return False

    def _ollama_model_tuple(self, model: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        binding = model.get("worker_bindings", [])[0]
        base_url = settings.worker_base_url(binding)
        if not base_url:
            return None
        return str(model.get("name")), {"base_url": base_url}

    def _required_capabilities(self, payload: dict[str, Any]) -> set[str]:
        return required_openai_capabilities(payload)

    C_FALLBACK_MAP = {
        "gemma4:31b": ["gemma3:12b", "qwen3.5:9b", "gemma4:e4b"],
        "qwen3.5:27b": ["gemma3:12b", "qwen3.5:9b", "gemma4:e4b"],
        "qwen3.6:27b": ["gemma4:e4b", "qwen3.5:9b", "qwen3.5:0.8b"],
        "qwen3.6:35b": ["gemma4:e4b", "qwen3.5:9b", "qwen3.5:0.8b"],
        "gemma4:26b": ["gemma3:12b", "qwen3.5:9b", "gemma4:e4b"],
        "gemma3:27b": ["gemma3:12b", "qwen3.5:9b", "gemma4:e4b"],
        "nemotron-cascade-2:30b": ["gemma3:12b", "qwen3.5:9b", "gemma4:e4b"],
        "glm-4.7-flash:q4_K_M": ["gemma3:12b", "qwen3.5:9b", "gemma4:e4b"],
        "qwen3-vl:8b": ["gemma3:12b", "translategemma:12b", "qwen3.5:9b"],
    }

    def _resolve_provider_and_worker(
        self,
        payload: dict[str, Any],
        *,
        allow_queue: bool,
    ) -> tuple[str, dict[str, Any] | None]:
        requested_model = payload.get("model")
        if not requested_model:
            raise self._as_http_error(status_code=400, code="bad_request", message="The request must include a model.")
        original_model = settings.resolve_model_alias(str(requested_model))
        if original_model != requested_model:
            payload["model"] = original_model
        required = self._required_capabilities(payload)
        registry_model = self._find_registry_model(original_model)
        if registry_model is not None and not self._model_supports_required(registry_model, required):
            fallback = self._configured_ollama_fallback(required) or self._first_ollama_fallback(required)
            if fallback is None:
                missing = ", ".join(sorted(required))
                raise self._as_http_error(
                    status_code=400,
                    code="unsupported_capabilities",
                    message=f"Model {original_model} does not support required capabilities: {missing}",
                )
            original_model, fallback_worker = fallback
            payload["model"] = original_model
            return "ollama", fallback_worker

        try_models = [original_model]
        fallbacks = self.C_FALLBACK_MAP.get(original_model, [])
        try_models.extend(fallbacks)
        try_models = self._prioritize_warm_models(try_models)

        last_error_resp = None

        for current_model in try_models:
            provider = (payload.get("provider") or provider_for_model(current_model, self.registry)).lower()
            if provider in ("openai", "gemini", "nvidia_nim", "ollama_cloud"):
                if current_model == original_model:
                    return provider, None
                continue

            try:
                response = get_session().post(
                    f"{settings.control_plane_url}/cluster/dispatch",
                    json={
                        "model": current_model,
                        "provider": provider,
                        "allow_queue": allow_queue,
                        "task_payload": payload,
                    },
                    timeout=10,
                )

                if response.status_code == 200:
                    dispatch = response.json()
                    if dispatch.get("status") == "assigned":
                        payload["model"] = current_model
                        worker = dict(dispatch["worker"])
                        if dispatch.get("assignment_id"):
                            worker["assignment_id"] = dispatch["assignment_id"]
                        return provider, worker
                    else:
                        last_error_resp = response
                elif response.status_code in {429, 503}:
                    last_error_resp = response
                else:
                    raise self._as_http_error(status_code=502, code="control_plane_error", message=response.text)

            except requests.RequestException as exc:
                last_error_resp = exc
                if current_model == original_model:
                    pass

        if isinstance(last_error_resp, requests.Response):
            status = last_error_resp.status_code
            if status == 429:
                detail = last_error_resp.json().get("detail", {})
                retry_after = 3
                if isinstance(detail, dict):
                    message = str(detail.get("message", "All matching workers are at queue capacity."))
                    retry_after = int(detail.get("retry_after", retry_after) or retry_after)
                else:
                    message = str(detail or "All matching workers are at queue capacity.")
                raise HTTPException(
                    status_code=429,
                    detail={"code": "worker_queue_full", "message": message, "retry_after": retry_after},
                    headers={"Retry-After": str(retry_after)},
                )
            if status == 503:
                message = last_error_resp.json().get("detail", "No worker available.")
                if isinstance(message, dict):
                    message = str(message.get("message", "No worker available."))
                raise self._as_http_error(status_code=503, code="worker_unavailable", message=str(message))
            raise self._as_http_error(status_code=502, code="control_plane_error", message=last_error_resp.text)

        elif isinstance(last_error_resp, Exception):
            raise self._as_http_error(status_code=503, code="provider_unreachable", message=str(last_error_resp))

        raise self._as_http_error(status_code=503, code="worker_unavailable", message="No suitable worker found in fallback chain.")

    def _adapter(self, provider: str, worker: dict[str, Any] | None):
        return adapter(provider, worker)

    def _capabilities_for_model(self, model: str) -> list[str]:
        return capabilities_for_model(model, self.registry)

    def _find_registry_model(self, model: str) -> dict[str, Any] | None:
        return find_registry_model(model, self.registry)

    def _model_supports_required(self, model: dict[str, Any], required: set[str]) -> bool:
        capabilities = set(model.get("capabilities", []))
        return required.issubset(capabilities)

    def _apply_generation_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @staticmethod
    def _ensure_openai_tools(tools: Any) -> Any:
        if not isinstance(tools, list):
            return tools
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                normalized.append(tool)
                continue
            if tool.get("type") == "function" and "function" not in tool and "name" in tool:
                fn_def = {k: v for k, v in tool.items() if k != "type"}
                normalized.append({"type": "function", "function": fn_def})
            else:
                normalized.append(tool)
        return normalized

    def _normalize_payload_for_provider(self, payload: dict[str, Any], provider: str) -> dict[str, Any]:
        normalized = dict(payload)
        if "tools" in normalized:
            normalized["tools"] = self._ensure_openai_tools(normalized["tools"])
        if provider == "ollama":
            messages = self._extract_messages_from_payload(normalized)
            if messages is not None:
                normalized["messages"] = [self._normalize_ollama_message(message) for message in messages]
        return normalized

    def _extract_messages_from_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]] | None:
        messages = payload.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        if "input" not in payload:
            return None

        return self._responses_input_to_messages(payload.get("input"))

    def _responses_input_to_messages(self, input_value: Any) -> list[dict[str, Any]] | None:
        if isinstance(input_value, str):
            return [{"role": "user", "content": input_value}]

        if isinstance(input_value, dict):
            role = str(input_value.get("role", "user"))
            if "content" in input_value:
                return [{"role": role, "content": input_value.get("content")}]
            if "text" in input_value:
                return [{"role": role, "content": str(input_value.get("text", ""))}]
            return None

        if not isinstance(input_value, list):
            return None

        messages: list[dict[str, Any]] = []
        for item in input_value:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue

            role = str(item.get("role", "user"))
            if "content" in item:
                messages.append({"role": role, "content": item.get("content")})
                continue
            if "text" in item:
                messages.append({"role": role, "content": str(item.get("text", ""))})
                continue
        return messages or None

    def _normalize_ollama_message(self, message: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(message)
        self._normalize_inbound_tool_calls_for_ollama(normalized)
        content = normalized.get("content")
        if not isinstance(content, list):
            return normalized

        text_parts: list[str] = []
        images: list[str] = list(normalized.get("images", [])) if isinstance(normalized.get("images"), list) else []

        for part in content:
            text, extracted_images = self._extract_content_part(part)
            if text:
                text_parts.append(text)
            if extracted_images:
                images.extend(extracted_images)

        normalized["content"] = "\n".join(text_parts)
        if images:
            normalized["images"] = images
        return normalized

    def _normalize_inbound_tool_calls_for_ollama(self, message: dict[str, Any]) -> None:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return

        normalized_calls: list[dict[str, Any]] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue

            call = dict(item)
            fn = dict(function)
            arguments = fn.get("arguments")
            if isinstance(arguments, str):
                parsed = self._try_parse_json(arguments)
                if isinstance(parsed, (dict, list)):
                    fn["arguments"] = parsed
                elif parsed is None:
                    fn["arguments"] = {}
                else:
                    fn["arguments"] = {"value": parsed}
            elif arguments is None:
                fn["arguments"] = {}

            call["function"] = fn
            normalized_calls.append(call)

        message["tool_calls"] = normalized_calls

    def _try_parse_json(self, value: str) -> Any | None:
        raw = value.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def _extract_content_part(self, part: Any) -> tuple[str, list[str]]:
        return content_part_to_text_and_images(part)

    def _normalize_image_ref(self, value: str) -> str:
        return normalize_image_ref(value)

    def _is_async_requested(self, payload: dict[str, Any]) -> bool:
        for key in ("async", "background", "queue"):
            value = payload.get(key)
            if isinstance(value, bool) and value:
                return True
            if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
                return True
        return False

    def _enqueue_async_task(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        requested_model = payload.get("model")
        if not requested_model:
            raise self._as_http_error(status_code=400, code="bad_request", message="The request must include a model.")
        model = settings.resolve_model_alias(str(requested_model))

        provider = (payload.get("provider") or provider_for_model(model, self.registry)).lower()
        task_payload = dict(payload)
        task_payload["model"] = model
        task_payload["endpoint"] = endpoint
        task_payload["provider"] = provider
        task_payload.pop("async", None)
        task_payload.pop("background", None)
        task_payload.pop("queue", None)
        task_payload.pop("stream", None)

        try:
            response = get_session().post(
                f"{settings.control_plane_url}/cluster/tasks",
                json={"payload": task_payload},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise self._as_http_error(status_code=503, code="control_plane_unavailable", message=f"Control plane unavailable: {exc}") from exc

        if not response.ok:
            raise self._as_http_error(status_code=502, code="control_plane_error", message=response.text)

        task = response.json().get("task", {})
        task_id = str(task.get("task_id") or "")
        return {
            "id": f"task_{task_id or uuid.uuid4().hex}",
            "object": "task",
            "status": "queued",
            "task_id": task_id,
            "model": model,
            "created": int(task.get("created_at", time.time())),
            "poll_url": f"{settings.control_plane_url}/cluster/tasks/{task_id}" if task_id else "",
        }

    def _resolve_max_turns(self) -> int:
        return settings.responses_max_turns

    def _record_token_usage(self, user_id: int | None, api_key_id: int | None,
                             input_tokens: int, output_tokens: int,
                             provider: str, model: str) -> None:
        if user_id is None:
            return
        try:
            db = SessionLocal()
            try:
                record_token_usage(db, user_id=user_id, api_key_id=api_key_id,
                                   input_tokens=input_tokens, output_tokens=output_tokens,
                                   provider=provider, model=model)
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to record token usage")

    def _record_response_usage(
        self,
        user_id: int | None,
        api_key_id: int | None,
        response: Any,
        provider: str,
        model: str,
    ) -> None:
        if user_id is None:
            return
        usage = getattr(response, "usage", None)
        if not usage:
            return
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        self._record_token_usage(user_id, api_key_id,
                                 input_tokens, output_tokens, provider, model)

    def _trace_responses(self, stage: str, **data: Any) -> None:
        if not settings.debug_responses:
            return
        safe: dict[str, Any] = {"stage": stage}
        for key, value in data.items():
            if key in {"payload", "effective_payload"}:
                safe[key] = self._summarize_payload(value)
            elif key == "messages":
                safe[key] = self._summarize_messages(value)
            elif key == "worker":
                safe[key] = self._summarize_worker(value)
            elif key == "completion":
                safe[key] = self._summarize_completion(value)
            elif key == "response":
                safe[key] = self._summarize_response(value)
            else:
                safe[key] = value
        try:
            logger.warning("responses.trace %s", json.dumps(safe, ensure_ascii=False, default=str))
        except Exception:
            logger.warning("responses.trace %s", safe)

    def _summarize_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"type": type(payload).__name__}
        input_value = payload.get("input")
        messages = payload.get("messages")
        return {
            "model": payload.get("model", ""),
            "provider": payload.get("provider", ""),
            "stream": bool(payload.get("stream", False)),
            "input_type": type(input_value).__name__ if "input" in payload else "",
            "input_len": self._value_len(input_value) if "input" in payload else 0,
            "messages": self._summarize_messages(messages),
            "tools_count": len(payload.get("tools") or []) if isinstance(payload.get("tools"), list) else 0,
        }

    def _summarize_messages(self, messages: Any) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        summary: list[dict[str, Any]] = []
        for item in messages[:6]:
            if not isinstance(item, dict):
                summary.append({"type": type(item).__name__})
                continue
            content = item.get("content", "")
            summary.append({
                "role": item.get("role", ""),
                "content_type": type(content).__name__,
                "content_len": self._value_len(content),
                "tool_calls": len(item.get("tool_calls") or []) if isinstance(item.get("tool_calls"), list) else 0,
            })
        if len(messages) > 6:
            summary.append({"remaining": len(messages) - 6})
        return summary

    def _summarize_worker(self, worker: Any) -> dict[str, Any]:
        if not isinstance(worker, dict):
            return {}
        return {
            "worker_id": worker.get("worker_id", ""),
            "base_url": worker.get("base_url", ""),
            "assignment_id": worker.get("assignment_id", ""),
        }

    def _summarize_completion(self, completion: Any) -> dict[str, Any]:
        if not isinstance(completion, dict):
            return {"type": type(completion).__name__}
        choices = completion.get("choices")
        message: dict[str, Any] = {}
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            raw_message = choices[0].get("message")
            message = raw_message if isinstance(raw_message, dict) else {}
        content = message.get("content", "")
        return {
            "model": completion.get("model", ""),
            "choices": len(choices) if isinstance(choices, list) else 0,
            "content_len": self._value_len(content),
            "tool_calls": len(message.get("tool_calls") or []) if isinstance(message.get("tool_calls"), list) else 0,
            "finish_reason": choices[0].get("finish_reason", "") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else "",
        }

    def _summarize_response(self, response: Any) -> dict[str, Any]:
        if hasattr(response, "to_dict"):
            response = response.to_dict()
        if not isinstance(response, dict):
            return {"type": type(response).__name__}
        return {
            "id": response.get("id", ""),
            "model": response.get("model", ""),
            "status": response.get("status", ""),
            "output_count": len(response.get("output") or []) if isinstance(response.get("output"), list) else 0,
            "output_text_len": self._value_len(response.get("output_text", "")),
        }

    def _value_len(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        if isinstance(value, list):
            total = 0
            for item in value:
                if isinstance(item, dict):
                    total += self._value_len(item.get("text", item.get("content", "")))
                else:
                    total += self._value_len(item)
            return total
        if isinstance(value, dict):
            return self._value_len(value.get("text", value.get("content", "")))
        return len(str(value))

    def _finalize_request(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        provider: str,
        worker: dict[str, Any] | None,
        latency_ms: float,
        error: bool,
        error_code: str = "",
    ) -> None:
        routing_engine.set_provider_latency(provider, latency_ms)
        if error:
            routing_engine.set_provider_failure(provider, code=error_code, message=error_code)
        else:
            routing_engine.set_provider_health(provider, True)

        if worker is not None:
            try:
                worker_id = worker.get("worker_id")
                if worker_id:
                    release_payload = {"worker_id": worker_id, "success": not error}
                    assignment_id = worker.get("assignment_id")
                    if assignment_id:
                        release_payload["assignment_id"] = assignment_id
                    gpu = worker.get("gpu_utilization")
                    if gpu is not None:
                        release_payload["gpu_utilization"] = float(gpu)
                    temp = worker.get("temperature")
                    if temp is not None:
                        release_payload["temperature"] = float(temp)
                    get_session().post(
                        f"{settings.control_plane_url}/cluster/release",
                        json=release_payload,
                        timeout=5,
                    )
            except requests.RequestException:
                pass
        try:
            get_session().post(
                f"{settings.control_plane_url}/cluster/telemetry",
                json={
                    "endpoint": endpoint,
                    "latency_ms": latency_ms,
                    "model": payload.get("model", ""),
                    "provider": provider,
                    "worker_id": worker.get("worker_id", "") if worker is not None else "",
                    "error": error,
                    "error_code": error_code,
                },
                timeout=5,
            )
        except requests.RequestException:
            pass
