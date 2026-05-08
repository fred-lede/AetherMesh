from __future__ import annotations
import os
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import requests
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, Response
from jinja2 import Environment, FileSystemLoader, Template

from config.settings import settings
from providers.http_client import get_session
from metrics.request_metrics import request_metrics
from runtime.orchestration.routing_engine import routing_engine

BASE_DIR = Path(__file__).resolve().parent

_TEMPLATES_DIR = str(BASE_DIR / "templates")
_ENV = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))
_ENV.cache = None

_compiled_templates: dict[str, Template] = {}
for _tpl_name in ("index.html", "login.html", "task_detail.html"):
    _source, _filename, _uptodate = _ENV.loader.get_source(_ENV, _tpl_name)
    _compiled_templates[_tpl_name] = _ENV.from_string(_source)


class _Templates:
    """Pre-compiled template renderer that bypasses Jinja2 3.1.5 _load_template
    bug where get_template() passes globals dict as the first positional arg."""

    def TemplateResponse(
        self,
        name: str,
        context: dict[str, Any],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
    ) -> HTMLResponse:
        template = _compiled_templates.get(name)
        if template is None:
            template = _compiled_templates.get("index.html")
        content = template.render(**context)
        return HTMLResponse(content, status_code=status_code, headers=headers, media_type=media_type)


templates = _Templates()
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AetherMesh Dashboard", version="4.0.0")

AUTH_EXEMPT_PATHS = {"/health", "/api/health", "/favicon.ico", "/login"}
AUTH_EXEMPT_PREFIXES = ("/static/",)
DASHBOARD_SESSION_COOKIE = "aiih_dashboard_session"
DASHBOARD_SESSION_TOKEN = secrets.token_urlsafe(32)


def _unauthorized_response() -> Response:
    return Response(
        content="Dashboard authentication required.",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="AetherMesh Dashboard"'},
    )


def _auth_credentials_configured() -> bool:
    return bool(settings.dashboard_auth_username and settings.dashboard_auth_password)


def _basic_auth_valid(request: Request) -> bool:
    return _basic_auth_username(request) is not None


def _basic_auth_username(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, encoded = auth.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return None

    import base64
    import binascii

    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None

    username, separator, password = decoded.partition(":")
    if not separator:
        return None

    username_ok = secrets.compare_digest(username, settings.dashboard_auth_username)
    password_ok = secrets.compare_digest(password, settings.dashboard_auth_password)
    return username if username_ok and password_ok else None


def _session_auth_valid(request: Request) -> bool:
    session = request.cookies.get(DASHBOARD_SESSION_COOKIE, "")
    return bool(session) and secrets.compare_digest(session, DASHBOARD_SESSION_TOKEN)


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept or "*/*" in accept


def _dashboard_actor(request: Request) -> str:
    if settings.dashboard_auth_enabled:
        basic_username = _basic_auth_username(request)
        if basic_username:
            return basic_username
        if _session_auth_valid(request):
            return settings.dashboard_auth_username
    return "local-dashboard"


@app.middleware("http")
async def dashboard_basic_auth(request: Request, call_next):
    path = request.url.path
    if (
        not settings.dashboard_auth_enabled
        or path in AUTH_EXEMPT_PATHS
        or any(path.startswith(prefix) for prefix in AUTH_EXEMPT_PREFIXES)
    ):
        return await call_next(request)

    if not _auth_credentials_configured():
        return Response(
            content="Dashboard authentication is enabled but credentials are not configured.",
            status_code=503,
        )

    if _session_auth_valid(request) or _basic_auth_valid(request):
        return await call_next(request)

    if request.url.path.startswith("/api/"):
        return _unauthorized_response()

    if request.method == "GET" and _wants_html(request):
        return RedirectResponse(url="/login", status_code=303)

    return RedirectResponse(url="/login", status_code=303)


def _web_search_providers() -> list[dict[str, Any]]:
    try:
        from runtime.tools.web_search import web_search_manager
        return [
            {
                "name": p.name,
                "configured": bool(getattr(p, "configured", False)),
            }
            for p in web_search_manager.providers
        ]
    except Exception:
        return []


def _check_cloud_provider(name: str, base_url_env: str, api_key_env: str, default_base: str) -> dict[str, Any]:
    """Check health of a cloud provider by calling its /models or /api/tags endpoint."""
    base_url = os.getenv(base_url_env, default_base).rstrip("/")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        return {"name": name, "ok": False, "status": "not_configured", "message": f"{api_key_env} is not set"}

    headers = {"Authorization": f"Bearer {api_key}"}
    endpoints_to_try = ["/models", "/api/tags"]
    
    for endpoint in endpoints_to_try:
        try:
            response = get_session().get(f"{base_url}{endpoint}", headers=headers, timeout=2)
            if response.ok:
                data = response.json()
                model_count = 0
                if isinstance(data, dict):
                    if "data" in data:
                        model_count = len(data["data"])
                    elif "models" in data:
                        model_count = len(data["models"])
                elif isinstance(data, list):
                    model_count = len(data)
                return {
                    "name": name,
                    "ok": True,
                    "status": "healthy",
                    "base_url": base_url,
                    "model_count": model_count,
                    "latency_ms": int(response.elapsed.total_seconds() * 1000),
                }
        except requests.RequestException:
            continue

    return {
        "name": name,
        "ok": False,
        "status": "unreachable",
        "base_url": base_url,
        "message": f"Failed to connect to {base_url}",
    }


def _check_cloud_providers() -> list[dict[str, Any]]:
    """Check health of all configured cloud providers."""
    providers = []

    # NVIDIA NIM
    if os.getenv("NVIDIA_NIM_API_KEY"):
        providers.append(
            _check_cloud_provider(
                "nvidia_nim",
                "NVIDIA_NIM_API_BASE",
                "NVIDIA_NIM_API_KEY",
                "https://integrate.api.nvidia.com/v1",
            )
        )

    # Ollama Cloud
    if os.getenv("OLLAMA_CLOUD_API_KEY"):
        providers.append(
            _check_cloud_provider(
                "ollama_cloud",
                "OLLAMA_CLOUD_API_BASE",
                "OLLAMA_CLOUD_API_KEY",
                "https://ollama.com",
            )
        )

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers.append(
            _check_cloud_provider(
                "openai",
                "OPENAI_API_BASE",
                "OPENAI_API_KEY",
                "https://api.openai.com/v1",
            )
        )

    # Gemini
    if os.getenv("GEMINI_API_KEY"):
        providers.append(
            _check_cloud_provider(
                "gemini",
                "GEMINI_API_BASE",
                "GEMINI_API_KEY",
                "https://generativelanguage.googleapis.com/v1beta",
            )
        )

    return providers


def _probe_local_ollama() -> dict[str, Any]:
    base_url = settings.ollama_fallback_base_url() or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    base_url = base_url.rstrip("/")
    try:
        response = get_session().get(f"{base_url}/api/tags", timeout=2)
    except requests.RequestException as exc:
        return {
            "name": "ollama",
            "ok": False,
            "status": "unreachable",
            "base_url": base_url,
            "message": str(exc),
        }

    if not response.ok:
        return {
            "name": "ollama",
            "ok": False,
            "status": "unhealthy",
            "base_url": base_url,
            "message": f"Ollama returned HTTP {response.status_code}",
        }

    data = response.json()
    models = data.get("models", []) if isinstance(data, dict) else []
    return {
        "name": "ollama",
        "ok": True,
        "status": "healthy",
        "base_url": base_url,
        "model_count": len(models),
        "latency_ms": int(response.elapsed.total_seconds() * 1000),
    }


def _probe_provider(provider: str) -> dict[str, Any]:
    provider = provider.lower()
    cloud_configs = {
        "nvidia_nim": (
            "NVIDIA_NIM_API_BASE",
            "NVIDIA_NIM_API_KEY",
            "https://integrate.api.nvidia.com/v1",
        ),
        "ollama_cloud": ("OLLAMA_CLOUD_API_BASE", "OLLAMA_CLOUD_API_KEY", "https://ollama.com"),
        "openai": ("OPENAI_API_BASE", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        "gemini": ("GEMINI_API_BASE", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta"),
    }
    if provider == "ollama":
        return _probe_local_ollama()
    if provider not in cloud_configs:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    base_url_env, api_key_env, default_base = cloud_configs[provider]
    return _check_cloud_provider(provider, base_url_env, api_key_env, default_base)


def _fetch_json(path: str) -> Any:
    try:
        response = get_session().get(f"{settings.control_plane_url}{path}", timeout=5)
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Control plane unavailable: {exc}") from exc
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


def _fetch_overview_json(path: str, fallback: dict[str, Any], errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        data = _fetch_json(path)
        return data if isinstance(data, dict) else fallback
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        errors.append({"path": path, "status": str(exc.status_code), "message": detail})
    except Exception as exc:
        errors.append({"path": path, "status": "error", "message": str(exc)})
    return fallback


def _fetch_router_metrics(endpoint: str) -> dict[str, Any]:
    """Fetch metrics directly from the Anthropic router process."""
    try:
        response = get_session().get(f"{settings.anthropic_router_url}{endpoint}", timeout=5)
        if response.ok:
            return response.json()
    except requests.RequestException:
        pass
    return {}


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    try:
        response = requests.post(f"{settings.control_plane_url}/cluster/tasks/{task_id}/cancel", timeout=5)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workers/{worker_id}/restart")
async def restart_worker(worker_id: str):
    try:
        response = requests.post(f"{settings.control_plane_url}/cluster/workers/{worker_id}/restart", timeout=5)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _enrich_models(models: list[dict[str, Any]], workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    worker_ids = {str(worker.get("worker_id", "")) for worker in workers}
    workers_by_port: dict[int, list[str]] = {}
    # Map who is resident where
    resident_map: dict[str, list[str]] = {}
    
    for worker in workers:
        worker_id = str(worker.get("worker_id", ""))
        port = worker.get("port")
        if port is None or not worker_id:
            continue
        workers_by_port.setdefault(int(port), []).append(worker_id)
        
        # Track resident models
        metadata = worker.get("metadata", {})
        ps_models = metadata.get("ps_models", [])
        if isinstance(ps_models, list):
            for m_name in ps_models:
                resident_map.setdefault(m_name, []).append(worker_id)

    enriched: list[dict[str, Any]] = []
    for model in models:
        model_name = model.get("name", "") # Registry uses 'name' or 'model'
        if not model_name:
            # Fallback to checking both common keys
            model_name = model.get("model", "")

        configured: list[str] = []
        for binding in model.get("worker_bindings", []):
            node_id = str(binding.get("node_id", "")).strip()
            port = binding.get("port")
            if not node_id or port is None:
                continue
            configured.append(f"{node_id}:{int(port)}")

        if not configured:
            for port in model.get("worker_ports", []):
                configured.extend(sorted(workers_by_port.get(int(port), [])))

        configured = sorted(set(configured))
        online = [worker_id for worker_id in configured if worker_id in worker_ids]
        resident = resident_map.get(model_name, [])

        row = dict(model)
        row["workers_configured"] = configured
        row["workers_online"] = online
        row["workers_resident"] = resident
        row["workers_configured_count"] = len(configured)
        row["workers_online_count"] = len(online)
        row["workers_resident_count"] = len(resident)
        enriched.append(row)
    return enriched


def _get_gpu_tier(gpu_name: str) -> str:
    name = gpu_name.lower()
    if "5090" in name: return "S"
    if "4070 ti super" in name: return "A"
    if "p40" in name: return "B"
    return "C"

def _build_alerts(
    *,
    nodes: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    queue_length = int(metrics.get("queue_length", 0))
    vision_stats = metrics.get("vision_error_rate_5m", {}).get("model", {})
    vision_error_rate = float(vision_stats.get("error_rate", 0.0))
    vision_requests = int(vision_stats.get("requests", 0))

    if not workers:
        alerts.append({"level": "critical", "message": "No workers registered in cluster."})

    dead_workers = [w for w in workers if str(w.get("status", "")).lower() == "dead"]
    if dead_workers:
        alerts.append(
            {
                "level": "critical",
                "message": f"{len(dead_workers)} worker(s) marked dead.",
            }
        )

    degraded_workers = [
        w for w in workers if str(w.get("status", "")).lower() in {"degraded", "stale"}
    ]
    if degraded_workers:
        alerts.append(
            {
                "level": "warning",
                "message": f"{len(degraded_workers)} worker(s) degraded/stale.",
            }
        )

    unhealthy_nodes = [n for n in nodes if str(n.get("status", "")).lower() != "healthy"]
    if unhealthy_nodes:
        alerts.append(
            {
                "level": "warning",
                "message": f"{len(unhealthy_nodes)} node(s) not healthy.",
            }
        )

    if queue_length >= settings.max_worker_queue_size:
        alerts.append(
            {
                "level": "warning",
                "message": (
                    f"Queue length {queue_length} reached worker queue limit "
                    f"{settings.max_worker_queue_size}."
                ),
            }
        )

    if vision_requests >= 3 and vision_error_rate >= 0.20:
        alerts.append(
            {
                "level": "warning",
                "message": (
                    f"Vision error rate 5m is {vision_error_rate * 100:.1f}% "
                    f"({vision_stats.get('errors', 0)}/{vision_requests})."
                ),
            }
        )

    if not alerts:
        alerts.append({"level": "ok", "message": "No active alerts."})
    return alerts


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "dashboard"}


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """
    Detailed health check for monitoring systems.
    Returns status suitable for Prometheus /healthz or Kubernetes liveness probes.
    """
    status = "healthy"
    checks: dict[str, Any] = {}

    # Check control plane connectivity
    try:
        cp = _fetch_json("/cluster/metrics")
        checks["control_plane"] = {"ok": True, "workers": len(cp.get("workers", []))}
    except Exception as e:
        status = "degraded"
        checks["control_plane"] = {"ok": False, "error": str(e)}

    # Check nodes and workers
    try:
        nodes = _fetch_json("/cluster/nodes")
        workers = _fetch_json("/cluster/workers")
        healthy_workers = [w for w in workers.get("workers", []) if w.get("status") == "healthy"]
        checks["cluster"] = {
            "ok": True,
            "nodes": len(nodes.get("nodes", [])),
            "workers_total": len(workers.get("workers", [])),
            "workers_healthy": len(healthy_workers),
        }
    except Exception as e:
        status = "degraded"
        checks["cluster"] = {"ok": False, "error": str(e)}

    # Check queue
    try:
        tasks = _fetch_json("/cluster/tasks")
        queue_length = len(tasks.get("tasks", []))
        queued = [t for t in tasks.get("tasks", []) if t.get("status") in {"pending", "queued"}]
        checks["queue"] = {
            "ok": True,
            "total": queue_length,
            "pending": len(queued),
        }
    except Exception as e:
        checks["queue"] = {"ok": False, "error": str(e)}

    return {
        "status": status,
        "service": "dashboard",
        "checks": checks,
    }


@app.get("/favicon.ico")
def favicon():
    return ""


@app.get("/static/{file_path:path}")
def static_files(file_path: str):
    full_path = STATIC_DIR / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(full_path)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str | None = None):
    if not settings.dashboard_auth_enabled:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": error,
            "credentials_configured": _auth_credentials_configured(),
        },
    )


@app.post("/login")
async def login(request: Request):
    if not settings.dashboard_auth_enabled:
        return RedirectResponse(url="/", status_code=303)
    if not _auth_credentials_configured():
        return RedirectResponse(url="/login?error=missing_credentials", status_code=303)

    raw_body = (await request.body()).decode("utf-8")
    form = parse_qs(raw_body, keep_blank_values=True)
    username = form.get("username", [""])[0]
    password = form.get("password", [""])[0]

    username_ok = secrets.compare_digest(username, settings.dashboard_auth_username)
    password_ok = secrets.compare_digest(password, settings.dashboard_auth_password)
    if not (username_ok and password_ok):
        return RedirectResponse(url="/login?error=invalid_credentials", status_code=303)

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        DASHBOARD_SESSION_COOKIE,
        DASHBOARD_SESSION_TOKEN,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(DASHBOARD_SESSION_COOKIE)
    return response


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "refresh_seconds": settings.dashboard_refresh_s,
            "control_plane_url": settings.control_plane_url,
            "auth_enabled": settings.dashboard_auth_enabled,
        },
    )


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    overview_errors: list[dict[str, str]] = []
    nodes = _fetch_overview_json("/cluster/nodes", {"nodes": []}, overview_errors)
    workers = _fetch_overview_json("/cluster/workers", {"workers": []}, overview_errors)
    gpus = _fetch_overview_json("/cluster/gpu", {"gpus": []}, overview_errors)
    tasks = _fetch_overview_json("/cluster/tasks", {"tasks": []}, overview_errors)
    metrics = _fetch_overview_json("/cluster/metrics", {}, overview_errors)
    models = _fetch_overview_json("/cluster/models", {"models": []}, overview_errors)

    node_items = nodes.get("nodes", [])
    worker_items = workers.get("workers", [])
    model_items = models.get("models", [])
    provider_status = metrics.get("provider_status", {})
    overview_metrics = dict(metrics)
    overview_metrics["max_worker_queue_size"] = settings.max_worker_queue_size
    models_enriched = _enrich_models(model_items, worker_items)
    alerts = _build_alerts(nodes=node_items, workers=worker_items, metrics=metrics)
    if overview_errors:
        alerts = [alert for alert in alerts if alert.get("level") != "ok"]
    for error in overview_errors:
        alerts.append(
            {
                "level": "critical" if error["status"] in {"503", "504"} else "warning",
                "message": f"Overview source {error['path']} returned {error['status']}: {error['message']}",
            }
        )
    cloud_providers = _check_cloud_providers()

    # Enrich GPUs and Workers with Tiers
    gpus_enriched = [
        {**g, "tier": _get_gpu_tier(g.get("name", ""))} 
        for g in gpus.get("gpus", [])
    ]
    workers_enriched = [
        {**w, "tier": _get_gpu_tier(w.get("gpu_name", ""))} 
        for w in worker_items
    ]

    return {
        "nodes": node_items,
        "workers": workers_enriched,
        "gpus": gpus_enriched,
        "tasks": tasks.get("tasks", []),
        "metrics": overview_metrics,
        "models": model_items,
        "models_enriched": models_enriched,
        "provider_status": provider_status,
        "alerts": alerts,
        "cloud_providers": cloud_providers,
        "request_metrics": _fetch_router_metrics("/api/metrics/requests") or request_metrics.get_summary(),
        "provider_metrics": (_fetch_router_metrics("/api/metrics/providers") or {}).get("providers", request_metrics.get_provider_metrics()),
        "provider_diagnostics": (_fetch_router_metrics("/api/metrics/provider-diagnostics") or {}).get("providers", request_metrics.get_provider_diagnostics()),
        "routing_status": routing_engine.get_routing_status(),
        "overview_status": "degraded" if overview_errors else "ok",
        "overview_errors": overview_errors,
        "web_search_providers": _web_search_providers(),
    }


@app.get("/api/providers/health")
def providers_health() -> dict[str, Any]:
    """Check health of all cloud providers."""
    return {"providers": _check_cloud_providers()}


@app.get("/api/web-search/status")
def web_search_status() -> dict[str, Any]:
    from runtime.tools.web_search import web_search_manager
    providers_info = []
    for p in web_search_manager.providers:
        providers_info.append({
            "name": p.name,
            "configured": bool(getattr(p, "configured", False)),
        })
    return {"providers": providers_info}


@app.post("/api/providers/{provider}/probe")
def provider_probe(provider: str) -> dict[str, Any]:
    """Probe one provider and feed the result back into routing health."""
    result = _probe_provider(provider)
    ok = bool(result.get("ok"))
    routing_engine.set_provider_health(provider, ok)
    latency_ms = result.get("latency_ms")
    if ok and latency_ms is not None:
        routing_engine.set_provider_latency(provider, float(latency_ms))
    return {"ok": ok, "provider": provider, "result": result}

@app.get("/task/{task_id}", response_class=HTMLResponse)
def task_detail_page(request: Request, task_id: str):
    task = _fetch_json(f"/cluster/tasks/{task_id}")
    return templates.TemplateResponse(
        "task_detail.html",
        {
            "request": request,
            "task": task,
        },
    )


@app.get("/api/task/{task_id}")
def task_detail(task_id: str) -> dict[str, Any]:
    return _fetch_json(f"/cluster/tasks/{task_id}")


@app.get("/api/requests/recent")
def requests_recent() -> dict[str, Any]:
    metrics = _fetch_json("/cluster/metrics")
    return {"events": metrics.get("recent_events", [])}


@app.get("/api/metrics/requests")
def metrics_requests() -> dict[str, Any]:
    return request_metrics.get_summary()


@app.get("/api/metrics/requests/recent")
def metrics_requests_recent(limit: int = 50) -> dict[str, Any]:
    return {"requests": request_metrics.get_recent_requests(limit)}


@app.get("/api/metrics/providers")
def metrics_providers() -> dict[str, Any]:
    return {"providers": request_metrics.get_provider_metrics()}


@app.get("/api/routing/status")
def routing_status() -> dict[str, Any]:
    return routing_engine.get_routing_status()


@app.post("/api/routing/overrides")
def routing_override_set(request: Request, payload: dict[str, str] = Body(...)) -> dict[str, Any]:
    model = payload.get("model", "")
    provider = payload.get("provider", "")
    if not model or not provider:
        raise HTTPException(status_code=400, detail="model and provider are required")
    routing_engine.set_model_override(model, provider, actor=_dashboard_actor(request))
    return {"ok": True, "model": model, "provider": provider}


@app.delete("/api/routing/overrides/{model}")
def routing_override_remove(request: Request, model: str) -> dict[str, Any]:
    routing_engine.clear_model_override(model, actor=_dashboard_actor(request))
    return {"ok": True, "model": model}


@app.post("/api/providers/{provider}/enable")
def provider_enable(request: Request, provider: str) -> dict[str, Any]:
    routing_engine.set_provider_enabled(provider, True, actor=_dashboard_actor(request))
    return {"ok": True, "provider": provider, "enabled": True}


@app.post("/api/providers/{provider}/disable")
def provider_disable(request: Request, provider: str) -> dict[str, Any]:
    routing_engine.set_provider_enabled(provider, False, actor=_dashboard_actor(request))
    return {"ok": True, "provider": provider, "enabled": False}


@app.post("/api/routing/local-only/enable")
def routing_local_only_enable(request: Request) -> dict[str, Any]:
    routing_engine.set_local_only_mode(True, actor=_dashboard_actor(request))
    return {"ok": True, "local_only": True}


@app.post("/api/routing/local-only/disable")
def routing_local_only_disable(request: Request) -> dict[str, Any]:
    routing_engine.set_local_only_mode(False, actor=_dashboard_actor(request))
    return {"ok": True, "local_only": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
