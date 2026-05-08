from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Iterable

from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import settings
from metrics.request_metrics import RequestRecord, request_metrics
from providers.base import ProviderError
from runtime.orchestration.anthropic_converter import AnthropicRouter
from runtime.orchestration.capabilities import required_anthropic_capabilities
from runtime.orchestration.routing_engine import routing_engine
from runtime.orchestration.streaming import stream_anthropic, stream_anthropic_with_metrics
from runtime.security.tool_policy import evaluate_server_tool_policy
from runtime.tools.builtin.web_search import stream_web_server_tool_response

anthropic_service = AnthropicRouter()
logger = logging.getLogger("anthropic_router")


class ASCIISafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")


ANTHROPIC_RATE_LIMIT_HEADERS = [
    "request-id", "retry-after",
    "anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset", "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining", "anthropic-ratelimit-tokens-reset",
    "anthropic-ratelimit-input-tokens-limit", "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-input-tokens-reset", "anthropic-ratelimit-output-tokens-limit",
    "anthropic-ratelimit-output-tokens-remaining", "anthropic-ratelimit-output-tokens-reset",
    "anthropic-ratelimit-retry-after", "anthropic-ratelimit-tier", "cf-ray",
]


app = FastAPI(title="AetherMesh - Anthropic Compatible", version="4.0.0")


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
        model, openai_payload["model"], provider,
        "yes" if worker else "no", routing_decision.score,
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
            request_id=request_id, model=model, provider="aiih_web_tools",
            endpoint="/v1/messages", streaming=True, latency_ms=0,
        ))
        return StreamingResponse(
            stream_web_server_tool_response(
                payload, model=model, timeout_s=settings.web_tool_timeout_s,
                max_results=settings.web_search_max_results,
            ),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Request-Id": request_id, "X-AIIH-Server-Tool": server_tool_policy.forced_tool},
        )

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
                    raise exc

            def stream_with_first() -> Iterable[dict[str, Any] | str]:
                yield first_item
                yield from iterator

            response_headers = {"Cache-Control": "no-cache", "X-Request-Id": request_id}
            return StreamingResponse(
                stream_anthropic_with_metrics(
                    anthropic_service, stream_with_first(), model, provider,
                    request_id, start_time, allowed_tool_names=allowed_tool_names,
                ),
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


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = int(exc.status_code)
    detail = exc.detail

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
        error_payload = {"type": error_type, "message": detail.get("message", "Request failed.")}
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
