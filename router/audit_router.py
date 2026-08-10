from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi import Query

from config.settings import settings
from runtime.security.audit_log import audit_log

router = APIRouter(prefix="/v1/audit")

SOURCE_PATHS = {
    "security": lambda: audit_log._path,
    "routing": lambda: settings.config_path("routing_audit.jsonl"),
}


def _read_source(source: str) -> list[dict[str, Any]]:
    try:
        path = SOURCE_PATHS[source]()
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown audit source '{source}'")
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            item.setdefault("source", source)
            events.append(item)
    return events


@router.get("/logs")
def audit_logs(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    action: str | None = None,
    actor: str | None = None,
    source: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict[str, Any]:
    sources = [source] if source else list(SOURCE_PATHS)
    events: list[dict[str, Any]] = []
    for src in sources:
        events.extend(_read_source(src))
    filtered: list[dict[str, Any]] = []
    for event in events:
        if action and event.get("action") != action:
            continue
        if actor and event.get("actor") != actor:
            continue
        ts = event.get("timestamp")
        if isinstance(ts, (int, float)):
            if start_time is not None and ts < start_time:
                continue
            if end_time is not None and ts > end_time:
                continue
        filtered.append(event)
    filtered.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    page = filtered[offset : offset + limit]
    return {"object": "list", "data": page, "has_more": offset + limit < len(filtered)}


@router.get("/sources")
def audit_sources() -> dict[str, Any]:
    return {"object": "list", "data": [{"name": name, "path": str(path())} for name, path in SOURCE_PATHS.items()]}
