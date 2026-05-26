from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from fastapi import Body, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import settings
from metrics.request_metrics import RequestRecord, request_metrics
from providers.base import ProviderError
from runtime.orchestration.anthropic_converter import AnthropicRouter
from runtime.orchestration.capabilities import required_anthropic_capabilities
from runtime.intelligence import execution_selector
from runtime.memory import memory_manager
from runtime.orchestration.routing_engine import routing_engine
from runtime.orchestration.streaming import stream_anthropic_with_metrics
from runtime.security.auth.token_tracker import record_token_usage
from runtime.security.database import SessionLocal
from runtime.security.tool_policy import evaluate_server_tool_policy, listed_server_tools
from runtime.tools.content_blocks import resolve_file_blocks
from runtime.tools.file_cleanup import get_file_cleanup_manager
from runtime.tools.builtin.web_search import (
    append_references_to_stream,
    extract_search_query,
    inject_search_context,
    latest_user_text,
    run_web_search,
    stream_web_server_tool_response,
)

logger = logging.getLogger("anthropic.messages_adapter")


class ASCIISafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")


def _record_token_usage(user_id: int | None, api_key_id: int | None, input_tokens: int, output_tokens: int, provider: str, model: str):
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


def _resolve_file_content_blocks(
    payload: dict[str, Any],
    upload_dir: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    file_ids: list[str] = []
    messages = payload.get("messages", [])
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "file_id":
                fid = block["file_id"]
                file_ids.append(fid)
                resolved = resolve_file_blocks([fid], "generic", upload_dir)
                new_content.extend(resolved)
            else:
                new_content.append(block)
        msg["content"] = new_content
    return payload, file_ids


def create_messages_routes(app, anthropic_service: AnthropicRouter):

    @app.post("/v1/messages")
    def messages(
        request: Request,
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

        payload, file_ids = _resolve_file_content_blocks(payload)

        openai_payload = anthropic_service._to_openai_payload(payload)
        openai_payload["model"] = anthropic_service._strip_model_prefix(model)

        required_caps = sorted(required_anthropic_capabilities(payload))

        routing_decision = routing_engine.route(
            model=model,
            required_capabilities=required_caps,
            registry_models=anthropic_service.registry.get("models", []),
            request_payload=payload,
        )
        routing_decision = execution_selector.rerank(
            routing_decision,
            model=model,
            required_capabilities=required_caps,
            has_tools=bool(allowed_tool_names),
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
            model, openai_payload["model"], provider,
            "yes" if worker else "no", routing_decision.score,
        )

        if file_ids:
            cleanup_mgr = get_file_cleanup_manager()
            cleanup_mgr.set_current_request(request_id)
            for fid in file_ids:
                cleanup_mgr.track_current(fid)

        web_search_results: list[dict[str, str]] | None = None
        if settings.web_tools_auto_search and not listed_server_tools(payload):
            query = extract_search_query(latest_user_text(payload))
            if query:
                try:
                    results = run_web_search(query, settings.web_search_max_results, settings.web_tool_timeout_s)
                    if results:
                        web_search_results = results
                        payload = dict(payload)
                        inject_search_context(payload, query, results)
                        openai_payload = anthropic_service._to_openai_payload(payload)
                        openai_payload["model"] = routing_decision.model if routing_decision.score >= 10 else anthropic_service._strip_model_prefix(model)
                except Exception as exc:
                    logger.warning(f"Auto web search failed: {exc}")

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

        old_short_circuit = server_tool_policy.should_handle_locally and not web_search_results
        if old_short_circuit:
            request_metrics.record_request(RequestRecord(
                request_id=request_id, model=model, provider="aiih_web_tools",
                endpoint="/v1/messages", streaming=True, latency_ms=0,
            ))
            if file_ids:
                get_file_cleanup_manager().cleanup_request(request_id)
            return StreamingResponse(
                stream_web_server_tool_response(
                    payload, model=model, timeout_s=settings.web_tool_timeout_s,
                    max_results=settings.web_search_max_results,
                ),
                media_type="text/event-stream; charset=utf-8",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id, "X-AIIH-Server-Tool": server_tool_policy.forced_tool},
            )

        try:
            try:
                adapter = anthropic_service._adapter(provider, worker)
            except ValueError as exc:
                raise HTTPException(
                    status_code=503,
                    detail={"type": "overloaded_error", "message": str(exc)},
                )
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
                        request_id=request_id, model=model, provider=provider,
                        endpoint="/v1/messages", streaming=True, latency_ms=latency_ms,
                        error=True, error_message=str(exc),
                    ))
                    routing_engine.set_provider_latency(provider, latency_ms)
                    routing_engine.set_provider_failure(provider, code=str(getattr(exc, "code", "") or ""), message=str(exc))
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
                        raise fallback_exc

                def stream_with_first() -> Iterable[dict[str, Any] | str]:
                    yield first_item
                    yield from iterator

                response_headers = {"Cache-Control": "no-cache", "X-Request-Id": request_id}
                inner = stream_anthropic_with_metrics(
                    anthropic_service, stream_with_first(), model, provider,
                    request_id, start_time, allowed_tool_names=allowed_tool_names,
                    user_id=getattr(request.state, "user_id", None),
                    api_key_id=getattr(request.state, "api_key_id", None),
                )
                if web_search_results:
                    inner = append_references_to_stream(inner, web_search_results)
                return StreamingResponse(
                    inner,
                    media_type="text/event-stream; charset=utf-8",
                    headers=response_headers,
                )
            else:
                response = adapter.chat(openai_payload)
                latency_ms = (time.time() - start_time) * 1000
                usage = response.get("usage") or {}
                request_metrics.record_request(RequestRecord(
                    request_id=request_id, model=model, provider=provider,
                    endpoint="/v1/messages", streaming=False, latency_ms=latency_ms,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                ))
                routing_engine.set_provider_latency(provider, latency_ms)
                routing_engine.set_provider_health(provider, True)
                result = anthropic_service._to_anthropic_response(response, model, allowed_tool_names=allowed_tool_names)
                memory_manager.episodic.record(
                    session_id=request_id,
                    model=model,
                    provider=provider,
                    duration_ms=latency_ms,
                    success=True,
                    token_count=dict(usage) if usage else None,
                )
                _record_token_usage(
                    user_id=getattr(request.state, "user_id", None),
                    api_key_id=getattr(request.state, "api_key_id", None),
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    provider=provider, model=model,
                )
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
            memory_manager.episodic.record(
                session_id=request_id,
                model=model,
                provider=provider,
                duration_ms=latency_ms,
                success=False,
                error=str(exc)[:200],
            )
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
                request_id=request_id, model=model, provider=provider,
                endpoint="/v1/messages", streaming=is_streaming, latency_ms=latency_ms,
                error=True, error_message=str(exc),
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
                    provider, model, fallback_model, exc,
                )
                try:
                    fallback_adapter = anthropic_service._adapter("ollama", fallback_worker)
                    response = fallback_adapter.chat(fallback_payload)
                    fallback_latency_ms = (time.time() - start_time) * 1000
                    usage = response.get("usage") or {}
                    request_metrics.record_request(RequestRecord(
                        request_id=request_id, model=model, provider="ollama",
                        endpoint="/v1/messages", streaming=False, latency_ms=fallback_latency_ms,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    ))
                    routing_engine.set_provider_latency("ollama", fallback_latency_ms)
                    routing_engine.set_provider_health("ollama", True)
                    _record_token_usage(
                        user_id=getattr(request.state, "user_id", None),
                        api_key_id=getattr(request.state, "api_key_id", None),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        provider="ollama", model=model,
                    )
                    memory_manager.episodic.record(
                        session_id=request_id,
                        model=model,
                        provider="ollama",
                        duration_ms=fallback_latency_ms,
                        success=True,
                        token_count=dict(usage) if usage else None,
                    )
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
                request_id=request_id, model=model, provider=provider,
                endpoint="/v1/messages", streaming=is_streaming, latency_ms=latency_ms,
                error=True, error_message=str(exc),
            ))
            routing_engine.set_provider_latency(provider, latency_ms)
            routing_engine.set_provider_failure(provider, code="api_error", message=str(exc))
            raise HTTPException(status_code=500, detail={"type": "api_error", "message": str(exc)})
        finally:
            if file_ids:
                get_file_cleanup_manager().cleanup_request(request_id)
