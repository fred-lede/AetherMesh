# Dashboard Model Manager Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Dashboard "Models" tab that creates, edits, deletes, and renames models in `config/models.yaml` through a UI, with hot reload and local/cloud awareness.

**Architecture:** A new `runtime/orchestration/model_registry_store.py` owns atomic read/write/validation of `models.yaml`; consumers refresh their snapshots via file-mtime checks so edits take effect without restarting services. The dashboard exposes CRUD endpoints; the front-end renders a table plus a slide-out drawer.

**Tech Stack:** Python 3.14, FastAPI, PyYAML, pytest (`asyncio_mode = auto`), vanilla JS in `dashboard/static/dashboard.js`.

**Spec:** `docs/superpowers/specs/2026-09-23-model-manager-tab-design.md`

## Global Constraints

- Python: 4-space indent, `from __future__ import annotations` at top of new modules, type hints on public APIs, no docstrings/comments unless required.
- Atomic write contract: write `<file>.tmp`, `os.replace` to target, `PermissionError` retry ×5 with `time.sleep(0.1 * (attempt + 1))`, keep `<file>.bak`. Exact pattern from `runtime/orchestration/routing_engine.py:163-175`.
- Hot reload must work with **no service restart**.
- Mutating endpoints are admin-only via `_require_admin()`; read endpoints require only dashboard auth.
- Canonical capability values (verbatim): `chat`, `tools`, `thinking`, `vision`, `audio`, `embeddings`, `rerank`, `responses`, `mcp`, `web_search`, `streaming`, `documents`, `image_gen`, `video`.
- `image_gen`/`video` are annotation + UI only this phase — do NOT add routing scores.
- Model `name` regex: `^[A-Za-z0-9._:/+-]+$`.
- Test command: `.venv\Scripts\python.exe -m pytest <path> -v`.

## Review Focus

- A malformed or partial `models.yaml` (`models:` null, an entry missing `name`, a non-dict entry) must not crash reads or `/v1/models`; invalid entries are skipped, not fatal.
- A save while another process holds the file must retry and must never leave a truncated `models.yaml`.
- A rename to a name differing only by case/whitespace must be treated as a duplicate (409), not silently written.
- `context_length` of `0`, negative, or a non-numeric string must return 400 and never be written.
- A local model with no `worker_bindings` (or a binding with a bad port) must be rejected with a clear message.

---

### Task 1: Capability vocabulary — add image_gen/video, split image alias

**Files:**
- Modify: `providers/registry.py:11-50`
- Test: `tests/test_providers_registry.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `Capability.IMAGE_GEN`, `Capability.VIDEO`, `CANONICAL_CAPABILITIES: tuple[str, ...]`, updated `CAPABILITY_ALIASES` (`"image"` → `IMAGE_GEN`).

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from providers.registry import (
    CANONICAL_CAPABILITIES,
    CAPABILITY_ALIASES,
    Capability,
    parse_capabilities,
)


def test_image_alias_maps_to_image_gen():
    assert CAPABILITY_ALIASES["image"] is Capability.IMAGE_GEN


def test_image_gen_and_video_are_canonical():
    assert Capability.IMAGE_GEN.value == "image_gen"
    assert Capability.VIDEO.value == "video"
    assert "image_gen" in CANONICAL_CAPABILITIES
    assert "video" in CANONICAL_CAPABILITIES


def test_parse_capabilities_handles_new_values():
    assert parse_capabilities(["image", "video"]) == {Capability.IMAGE_GEN, Capability.VIDEO}


def test_parse_capabilities_keeps_vision_separate():
    assert parse_capabilities(["vision", "image"]) == {Capability.VISION, Capability.IMAGE_GEN}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_providers_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'CANONICAL_CAPABILITIES'`.

- [ ] **Step 3: Write minimal implementation**

In `providers/registry.py`, replace the `Capability` enum (lines 11-23) with:

```python
class Capability(str, Enum):
    CHAT = "chat"
    TOOLS = "tools"
    THINKING = "thinking"
    VISION = "vision"
    IMAGE_GEN = "image_gen"
    VIDEO = "video"
    AUDIO = "audio"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    RESPONSES = "responses"
    MCP = "mcp"
    WEB_SEARCH = "web_search"
    STREAMING = "streaming"
    DOCUMENTS = "documents"


CANONICAL_CAPABILITIES: tuple[str, ...] = tuple(c.value for c in Capability)
```

In `CAPABILITY_ALIASES`, replace the `"image"` line with:

```python
    "image": Capability.IMAGE_GEN,
    "image_gen": Capability.IMAGE_GEN,
    "imagegen": Capability.IMAGE_GEN,
    "text_to_image": Capability.IMAGE_GEN,
    "video": Capability.VIDEO,
```

(Keep `"vision": Capability.VISION`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_providers_registry.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add providers/registry.py tests/test_providers_registry.py
git commit -m "feat: add image_gen/video capabilities and split image alias from vision"
```

---

### Task 2: model_registry_store — load, mtime cache, atomic save

**Files:**
- Create: `runtime/orchestration/model_registry_store.py`
- Test: `tests/test_model_registry_store.py` (create)

**Interfaces:**
- Consumes: `providers.registry.CAPABILITY_ALIASES` (Task 1), `config.settings.settings`.
- Produces:
  - `models_path() -> Path`
  - `models_mtime() -> float`
  - `load_models() -> list[dict[str, Any]]`
  - `get_models() -> list[dict[str, Any]]`
  - `save_models(models: list[dict[str, Any]]) -> None`
  - `reload_models() -> None`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.orchestration import model_registry_store as store


@pytest.fixture()
def models_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.yaml"
    monkeypatch.setattr(store, "models_path", lambda: path)
    store.reload_models()
    return path


def test_load_missing_file_returns_empty(models_file: Path):
    assert store.load_models() == []


def test_load_null_models_returns_empty(models_file: Path):
    models_file.write_text("models:\n", encoding="utf-8")
    assert store.load_models() == []


def test_get_models_caches_until_mtime_changes(models_file: Path):
    models_file.write_text("models:\n- name: a\n  provider: ollama\n", encoding="utf-8")
    first = store.get_models()
    assert first[0]["name"] == "a"
    assert store.get_models() is first
    models_file.write_text("models:\n- name: b\n  provider: ollama\n", encoding="utf-8")
    os.utime(models_file, (models_file.stat().st_mtime + 10, models_file.stat().st_mtime + 10))
    assert store.get_models()[0]["name"] == "b"


def test_save_models_writes_and_backs_up(models_file: Path):
    store.save_models([{"name": "x", "provider": "ollama"}])
    assert "name: x" in models_file.read_text(encoding="utf-8")
    store.save_models([{"name": "y", "provider": "ollama"}])
    bak = models_file.with_suffix(".yaml.bak")
    assert bak.exists()
    assert "name: x" in bak.read_text(encoding="utf-8")


def test_save_preserves_other_top_level_keys(models_file: Path):
    models_file.write_text("version: 2\nmodels: []\n", encoding="utf-8")
    store.save_models([{"name": "x", "provider": "ollama"}])
    text = models_file.read_text(encoding="utf-8")
    assert "version: 2" in text
    assert "name: x" in text


def test_save_retries_on_permission_error(models_file: Path, monkeypatch: pytest.MonkeyPatch):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky)
    store.save_models([{"name": "x", "provider": "ollama"}])
    assert calls["n"] == 3
    assert "name: x" in models_file.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_registry_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.orchestration.model_registry_store'`.

- [ ] **Step 3: Write minimal implementation**

Create `runtime/orchestration/model_registry_store.py`:

```python
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from config.settings import settings

logger = logging.getLogger("runtime.orchestration.model_registry_store")

_TOP_LEVEL_KEY = "models"
_cache: list[dict[str, Any]] | None = None
_cache_mtime: float = -1.0


def models_path() -> Path:
    return settings.config_path("models.yaml")


def models_mtime() -> float:
    try:
        return models_path().stat().st_mtime
    except OSError:
        return -1.0


def load_models() -> list[dict[str, Any]]:
    path = models_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        logger.warning("models.yaml is not a mapping; ignoring")
        return []
    models = doc.get(_TOP_LEVEL_KEY) or []
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict)]


def get_models() -> list[dict[str, Any]]:
    global _cache, _cache_mtime
    current = models_mtime()
    if _cache is not None and current == _cache_mtime:
        return _cache
    _cache = load_models()
    _cache_mtime = current
    return _cache


def save_models(models: list[dict[str, Any]]) -> None:
    path = models_path()
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            existing = loaded
    existing[_TOP_LEVEL_KEY] = list(models)
    text = yaml.safe_dump(existing, allow_unicode=True, sort_keys=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    tmp.write_text(text, encoding="utf-8")
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            if path.exists():
                os.replace(path, bak)
            os.replace(tmp, path)
            reload_models()
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def reload_models() -> None:
    global _cache, _cache_mtime
    _cache = None
    _cache_mtime = -1.0
    get_models()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_registry_store.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/model_registry_store.py tests/test_model_registry_store.py
git commit -m "feat: add model_registry_store with mtime cache and atomic save"
```

---

### Task 3: model_registry_store — normalize, local/cloud, validate

**Files:**
- Modify: `runtime/orchestration/model_registry_store.py`
- Test: `tests/test_model_registry_store.py` (append)

**Interfaces:**
- Consumes: Task 2 store; `providers.registry.CAPABILITY_ALIASES`, `CANONICAL_CAPABILITIES`.
- Produces:
  - `LOCAL_PROVIDERS: set[str]`, `CLOUD_PROVIDERS: set[str]`
  - `is_local_model(entry: dict[str, Any]) -> bool`
  - `normalize_model(entry: dict[str, Any]) -> dict[str, Any]`
  - `validate_model(entry: dict[str, Any], existing_names: set[str] | None = None) -> tuple[dict[str, Any], list[str]]`

- [ ] **Step 1: Write the failing test**

```python
def test_normalize_canonicalizes_capabilities():
    out = store.normalize_model({"name": " m ", "provider": "ollama", "capabilities": ["Vision", "image", "image"]})
    assert out["name"] == "m"
    assert out["capabilities"] == ["image_gen", "vision"]


def test_normalize_local_keeps_worker_bindings():
    out = store.normalize_model({
        "name": "m", "provider": "ollama",
        "worker_bindings": [{"node_id": "node-01", "port": "11434"}],
    })
    assert out["worker_bindings"] == [{"node_id": "node-01", "port": 11434}]
    assert "worker_ports" not in out


def test_normalize_cloud_uses_worker_ports():
    out = store.normalize_model({"name": "m", "provider": "openai", "worker_ports": []})
    assert out["worker_ports"] == []
    assert "worker_bindings" not in out


def test_is_local_model():
    assert store.is_local_model({"provider": "ollama"}) is True
    assert store.is_local_model({"provider": "openai"}) is False


def test_validate_rejects_missing_name():
    _, errors = store.validate_model({"provider": "ollama"})
    assert any("name" in e for e in errors)


def test_validate_rejects_duplicate_name():
    _, errors = store.validate_model({"name": "a", "provider": "ollama"}, existing_names={"a"})
    assert any("duplicate" in e for e in errors)


def test_validate_rejects_bad_context_length():
    _, errors = store.validate_model({"name": "a", "provider": "ollama", "context_length": 0})
    assert any("context_length" in e for e in errors)


def test_validate_rejects_local_without_bindings():
    _, errors = store.validate_model({"name": "a", "provider": "ollama"})
    assert any("worker_bindings" in e for e in errors)


def test_validate_rejects_bad_port():
    entry = {"name": "a", "provider": "ollama", "worker_bindings": [{"node_id": "n", "port": 70000}]}
    _, errors = store.validate_model(entry)
    assert any("port" in e for e in errors)


def test_validate_accepts_good_local_model():
    entry = {"name": "a", "provider": "ollama", "worker_bindings": [{"node_id": "node-01", "port": 11434}],
             "capabilities": ["chat"], "context_length": 32768}
    clean, errors = store.validate_model(entry)
    assert errors == []
    assert clean["context_length"] == 32768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_registry_store.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'normalize_model'`.

- [ ] **Step 3: Write minimal implementation**

Append to `runtime/orchestration/model_registry_store.py` (add imports `re` and the registry symbols at the top):

```python
import re

from providers.registry import CAPABILITY_ALIASES, CANONICAL_CAPABILITIES
```

```python
LOCAL_PROVIDERS: set[str] = {"ollama", "xtts"}
CLOUD_PROVIDERS: set[str] = {"openai", "gemini", "nvidia_nim", "ollama_cloud"}
_NAME_RE = re.compile(r"^[A-Za-z0-9._:/+-]+$")


def is_local_model(entry: dict[str, Any]) -> bool:
    return str(entry.get("provider", "")).strip() not in CLOUD_PROVIDERS


def _canonical_capability(value: str) -> str:
    key = str(value).strip().lower()
    parsed = CAPABILITY_ALIASES.get(key)
    if parsed is not None:
        return parsed.value
    return key


def normalize_model(entry: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": str(entry.get("name", "")).strip(),
        "provider": str(entry.get("provider", "")).strip(),
    }
    if entry.get("worker_bindings") is not None:
        out["worker_bindings"] = [
            {"node_id": str(b.get("node_id", "")).strip(), "port": int(b.get("port", 0))}
            for b in (entry.get("worker_bindings") or [])
        ]
    else:
        out["worker_ports"] = list(entry.get("worker_ports") or [])
    caps = sorted({_canonical_capability(c) for c in (entry.get("capabilities") or [])})
    out["capabilities"] = caps
    vram = entry.get("estimated_vram_mb")
    out["estimated_vram_mb"] = int(vram) if vram is not None else None
    ctx = entry.get("context_length")
    out["context_length"] = int(ctx) if ctx is not None else None
    return out


def validate_model(
    entry: dict[str, Any],
    existing_names: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    name = str(entry.get("name", "")).strip()
    if not name:
        errors.append("name is required")
    elif not _NAME_RE.match(name):
        errors.append(f"invalid name: {name!r}")
    elif existing_names and name in existing_names:
        errors.append(f"duplicate name: {name!r}")

    provider = str(entry.get("provider", "")).strip()
    if not provider:
        errors.append("provider is required")

    bindings = entry.get("worker_bindings")
    if is_local_model(entry):
        if not bindings:
            errors.append("worker_bindings is required for local models")
        else:
            for binding in bindings:
                port = binding.get("port")
                try:
                    port_int = int(port)
                except (TypeError, ValueError):
                    errors.append(f"invalid port: {port!r}")
                    continue
                if not 1 <= port_int <= 65535:
                    errors.append(f"port out of range: {port_int}")

    caps = entry.get("capabilities") or []
    for cap in caps:
        if _canonical_capability(cap) not in CANONICAL_CAPABILITIES:
            errors.append(f"unknown capability: {cap!r}")

    ctx = entry.get("context_length")
    if ctx is not None:
        try:
            if int(ctx) <= 0:
                errors.append("context_length must be a positive integer")
        except (TypeError, ValueError):
            errors.append(f"invalid context_length: {ctx!r}")

    if errors:
        return normalize_model(entry), errors
    return normalize_model(entry), []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_registry_store.py -v`
Expected: PASS (16 passed).

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/model_registry_store.py tests/test_model_registry_store.py
git commit -m "feat: add model normalize/local-cloud/validate to model_registry_store"
```

---

### Task 4: model_references — scan and update routing_rules.yaml references

**Files:**
- Create: `runtime/orchestration/model_references.py`
- Test: `tests/test_model_references.py` (create)

**Interfaces:**
- Consumes: nothing (reads `routing_rules.yaml` via `settings.config_path`).
- Produces:
  - `scan_model_references(old_name: str) -> list[dict[str, str]]`
  - `update_model_references(old_name: str, new_name: str) -> None`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from runtime.orchestration import model_references as refs


@pytest.fixture()
def rules_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "routing_rules.yaml"
    monkeypatch.setattr(refs, "_rules_path", lambda: path)
    return path


def test_scan_finds_aliases_and_fallbacks(rules_file: Path):
    rules_file.write_text(
        yaml.safe_dump({
            "aliases": {"fast": "old-model"},
            "fallbacks": {"old-model": ["a", "b"]},
            "rules": [{"model": "old-model", "provider": "ollama"}],
        }),
        encoding="utf-8",
    )
    found = refs.scan_model_references("old-model")
    paths = {f["path"] for f in found}
    assert "aliases.fast" in paths
    assert "fallbacks.old-model" in paths
    assert "rules[0].model" in paths


def test_scan_returns_empty_when_absent(rules_file: Path):
    rules_file.write_text(yaml.safe_dump({"aliases": {"fast": "other"}}), encoding="utf-8")
    assert refs.scan_model_references("old-model") == []


def test_update_rewrites_values_and_preserves_others(rules_file: Path):
    rules_file.write_text(
        yaml.safe_dump({
            "aliases": {"fast": "old-model", "keep": "other"},
            "fallbacks": {"old-model": ["a"]},
        }),
        encoding="utf-8",
    )
    refs.update_model_references("old-model", "new-model")
    data = yaml.safe_load(rules_file.read_text(encoding="utf-8"))
    assert data["aliases"]["fast"] == "new-model"
    assert data["aliases"]["keep"] == "other"
    assert "new-model" in data["fallbacks"]
    assert "old-model" not in data["fallbacks"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_references.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.orchestration.model_references'`.

- [ ] **Step 3: Write minimal implementation**

Create `runtime/orchestration/model_references.py`:

```python
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from config.settings import settings

logger = logging.getLogger("runtime.orchestration.model_references")


def _rules_path() -> Path:
    return settings.config_path("routing_rules.yaml")


def _load_rules() -> dict[str, Any]:
    path = _rules_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def scan_model_references(old_name: str) -> list[dict[str, str]]:
    data = _load_rules()
    found: list[dict[str, str]] = []
    aliases = data.get("aliases") or {}
    if isinstance(aliases, dict):
        for alias, target in aliases.items():
            if str(target) == old_name:
                found.append({"source": "routing_rules", "path": f"aliases.{alias}", "value": old_name})
    fallbacks = data.get("fallbacks") or {}
    if isinstance(fallbacks, dict) and old_name in fallbacks:
        found.append({"source": "routing_rules", "path": f"fallbacks.{old_name}", "value": old_name})
    rules = data.get("rules") or []
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if isinstance(rule, dict) and str(rule.get("model")) == old_name:
                found.append({"source": "routing_rules", "path": f"rules[{index}].model", "value": old_name})
    return found


def _rewrite(node: Any, old_name: str, new_name: str) -> Any:
    if isinstance(node, str):
        return new_name if node == old_name else node
    if isinstance(node, list):
        return [_rewrite(item, old_name, new_name) for item in node]
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in node.items():
            new_key = new_name if key == old_name else key
            result[new_key] = _rewrite(value, old_name, new_name)
        return result
    return node


def update_model_references(old_name: str, new_name: str) -> None:
    path = _rules_path()
    if not path.exists():
        return
    data = _load_rules()
    rewritten = _rewrite(data, old_name, new_name)
    text = yaml.safe_dump(rewritten, allow_unicode=True, sort_keys=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_references.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/model_references.py tests/test_model_references.py
git commit -m "feat: add model reference scan/update for routing_rules.yaml"
```

---

### Task 5: Hot reload — refresh consumer snapshots on mtime change

**Files:**
- Modify: `runtime/orchestration/openai_handler.py` (RouterService.__init__ + entry points)
- Modify: `runtime/orchestration/anthropic_converter.py` (AnthropicRouter entry points)
- Modify: `runtime/orchestration/routing_engine.py` (`route()`)
- Test: `tests/test_model_hot_reload.py` (create)

**Interfaces:**
- Consumes: `model_registry_store.models_mtime()`, `settings.model_registry()`.
- Produces: `RouterService._ensure_registry()`, `AnthropicRouter._ensure_registry()`, routing engine mtime guard.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from unittest.mock import MagicMock

from runtime.orchestration import openai_handler


def test_ensure_registry_refreshes_on_mtime_change(monkeypatch):
    service = openai_handler.RouterService.__new__(openai_handler.RouterService)
    service.registry = [{"name": "old"}]
    service._registry_mtime = 1.0
    monkeypatch.setattr(openai_handler.settings, "model_registry", lambda: [{"name": "new"}])
    import runtime.orchestration.model_registry_store as store
    monkeypatch.setattr(store, "models_mtime", lambda: 2.0)
    service._ensure_registry()
    assert service.registry == [{"name": "new"}]


def test_ensure_registry_keeps_cache_when_unchanged(monkeypatch):
    service = openai_handler.RouterService.__new__(openai_handler.RouterService)
    service.registry = [{"name": "old"}]
    service._registry_mtime = 5.0
    called = MagicMock()
    monkeypatch.setattr(openai_handler.settings, "model_registry", called)
    import runtime.orchestration.model_registry_store as store
    monkeypatch.setattr(store, "models_mtime", lambda: 5.0)
    service._ensure_registry()
    assert service.registry == [{"name": "old"}]
    called.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_hot_reload.py -v`
Expected: FAIL — `AttributeError: 'RouterService' object has no attribute '_ensure_registry'`.

- [ ] **Step 3: Write minimal implementation**

In `runtime/orchestration/openai_handler.py`, change `RouterService.__init__` to initialize `self.registry` and `self._registry_mtime`:

```python
        self.registry = settings.model_registry()
        self._registry_mtime = model_registry_store.models_mtime()
```

Import at module level:

```python
from runtime.orchestration import model_registry_store
```

Add the method to `RouterService`:

```python
    def _ensure_registry(self) -> None:
        current = model_registry_store.models_mtime()
        if current != self._registry_mtime:
            self.registry = settings.model_registry()
            self._registry_mtime = current
```

Call `self._ensure_registry()` as the first line of `handle_chat`, `handle_streaming_chat`, `handle_responses`, `handle_streaming_responses`, and `list_models`.

Repeat the same three edits in `runtime/orchestration/anthropic_converter.py` for `AnthropicRouter` (its `__init__`, entry methods, and `_ensure_registry`).

In `runtime/orchestration/routing_engine.py`, inside `ModelRoutingEngine.route()`, before the registry is used:

```python
        current_mtime = model_registry_store.models_mtime()
        if current_mtime != getattr(self, "_registry_mtime", -1.0):
            self._load_config()
            self._registry_mtime = current_mtime
```

Import `model_registry_store` at module level in `routing_engine.py` too.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_hot_reload.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the affected suites**

Run: `.venv\Scripts\python.exe -m pytest tests/test_orchestration.py tests/test_routing_engine.py tests/test_model_hot_reload.py -q`
Expected: PASS, no new failures versus baseline.

- [ ] **Step 6: Commit**

```bash
git add runtime/orchestration/openai_handler.py runtime/orchestration/anthropic_converter.py runtime/orchestration/routing_engine.py tests/test_model_hot_reload.py
git commit -m "feat: hot-reload model registry in router and routing engine via mtime"
```

---

### Task 6: Dashboard API endpoints for models

**Files:**
- Modify: `dashboard/dashboard_server.py` (add endpoints near `/api/custom-providers`, ~line 1058)
- Test: `tests/test_models_api.py` (create)

**Interfaces:**
- Consumes: store (Tasks 2-3), references (Task 4), `_require_admin`.
- Produces (all under `/api`): `GET /models`, `GET /models/capabilities`, `GET /models/providers`, `GET /models/nodes`, `POST /models`, `PUT /models/{name}`, `DELETE /models/{name}`, `POST /models/{name}/fetch-context`, `POST /models/reload`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard import dashboard_server as dash


def _client(admin: bool = True) -> TestClient:
    app = dash.app
    if not admin:
        app.dependency_overrides[dash._require_admin] = lambda: None
    return TestClient(app)


def test_capabilities_endpoint_lists_vocabulary(monkeypatch):
    client = TestClient(dash.app)
    resp = client.get("/api/models/capabilities")
    assert resp.status_code == 200
    values = {item["value"] for item in resp.json()["capabilities"]}
    assert "image_gen" in values and "video" in values
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_models_api.py -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Write minimal implementation**

In `dashboard/dashboard_server.py`, import at top:

```python
from runtime.orchestration import model_references, model_registry_store
```

Add, after the custom-providers endpoints:

```python
_CAPABILITY_GROUPS: list[dict[str, str]] = [
    {"value": "chat", "label": "Chat", "group": "core"},
    {"value": "responses", "label": "Responses", "group": "core"},
    {"value": "streaming", "label": "Streaming", "group": "core"},
    {"value": "thinking", "label": "Reasoning", "group": "reasoning"},
    {"value": "tools", "label": "Tool", "group": "reasoning"},
    {"value": "mcp", "label": "MCP", "group": "reasoning"},
    {"value": "web_search", "label": "Web Search", "group": "reasoning"},
    {"value": "vision", "label": "Vision", "group": "modality"},
    {"value": "image_gen", "label": "Image", "group": "modality"},
    {"value": "audio", "label": "Audio", "group": "modality"},
    {"value": "video", "label": "Video", "group": "modality"},
    {"value": "documents", "label": "Documents", "group": "modality"},
    {"value": "embeddings", "label": "Embedding", "group": "other"},
    {"value": "rerank", "label": "Reranker", "group": "other"},
]

_LOCAL_PROVIDERS = ["ollama", "xtts"]
_CLOUD_PROVIDERS = ["openai", "gemini", "nvidia_nim", "ollama_cloud"]


def _model_public(entry: dict[str, Any]) -> dict[str, Any]:
    local = model_registry_store.is_local_model(entry)
    workers = entry.get("worker_bindings") or []
    return {
        **entry,
        "category": "local" if local else "cloud",
        "workers": [f"{b.get('node_id')}:{b.get('port')}" for b in workers],
    }


@api.get("/models")
def list_models():
    models = model_registry_store.get_models()
    return {"models": [_model_public(m) for m in models]}


@api.get("/models/capabilities")
def list_model_capabilities():
    return {"capabilities": _CAPABILITY_GROUPS}


@api.get("/models/providers")
def list_model_providers():
    custom = sorted(settings.load_custom_providers().keys()) if hasattr(settings, "load_custom_providers") else []
    return {
        "local": _LOCAL_PROVIDERS,
        "cloud": _CLOUD_PROVIDERS + custom,
    }


@api.get("/models/nodes")
def list_model_nodes():
    cluster = settings.load_yaml("cluster.yaml") or {}
    hosts = cluster.get("node_hosts") or {}
    return {"nodes": sorted(hosts.keys())}


@api.post("/models")
def create_model(payload: dict[str, Any], _: None = Depends(_require_admin)):
    models = model_registry_store.get_models()
    names = {m.get("name") for m in models}
    clean, errors = model_registry_store.validate_model(payload, existing_names=names)
    if errors:
        raise HTTPException(status_code=400, detail=errors)
    models.append(clean)
    model_registry_store.save_models(models)
    return {"model": _model_public(clean)}


@api.put("/models/{name}")
def update_model(name: str, payload: dict[str, Any], update_references: bool = False, _: None = Depends(_require_admin)):
    models = model_registry_store.get_models()
    index = next((i for i, m in enumerate(models) if m.get("name") == name), None)
    if index is None:
        raise HTTPException(status_code=404, detail="model not found")
    new_name = str(payload.get("name", name)).strip()
    others = {m.get("name") for i, m in enumerate(models) if i != index}
    clean, errors = model_registry_store.validate_model(payload, existing_names=others)
    if new_name != name:
        references = model_references.scan_model_references(name)
        if references and not update_references:
            raise HTTPException(status_code=409, detail={"message": "references exist", "references": references})
        if update_references:
            model_references.update_model_references(name, new_name)
    if errors:
        raise HTTPException(status_code=400, detail=errors)
    models[index] = clean
    model_registry_store.save_models(models)
    return {"model": _model_public(clean)}


@api.delete("/models/{name}")
def delete_model(name: str, _: None = Depends(_require_admin)):
    models = model_registry_store.get_models()
    remaining = [m for m in models if m.get("name") != name]
    if len(remaining) == len(models):
        raise HTTPException(status_code=404, detail="model not found")
    model_registry_store.save_models(remaining)
    return {"deleted": name}


@api.post("/models/{name}/fetch-context")
def fetch_model_context(name: str, _: None = Depends(_require_admin)):
    from runtime.orchestration import model_context
    models = model_registry_store.get_models()
    entry = next((m for m in models if m.get("name") == name), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="model not found")
    value = model_context.fetch_context_length(entry.get("provider", ""), name, entry)
    if value is None:
        raise HTTPException(status_code=422, detail="could not determine context length")
    return {"context_length": value}


@api.post("/models/reload")
def reload_model_registry(_: None = Depends(_require_admin)):
    model_registry_store.reload_models()
    return {"reloaded": True, "count": len(model_registry_store.get_models())}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_models_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/dashboard_server.py tests/test_models_api.py
git commit -m "feat: add /api/models CRUD endpoints to dashboard"
```

---

### Task 7: Front-end — Models tab and list rendering

**Files:**
- Modify: `dashboard/templates/index.html` (tab strip ~line 621, panel area ~line 947)
- Modify: `dashboard/static/dashboard.js` (`renderOverview` ~line 490, add render function)

**Interfaces:**
- Consumes: `GET /api/models`, `switchTab()`.
- Produces: `renderModels(data)` and tab button `data-tab="models"`.

- [ ] **Step 1: Add the tab button**

In `dashboard/templates/index.html`, add alongside the existing `dash-tab` spans:

```html
      <span class="dash-tab" data-tab="models" onclick="switchTab('models')">Models</span>
```

- [ ] **Step 2: Add the panel**

Add a `tab-content` panel with a table:

```html
    <div class="tab-content" data-tab="models">
      <div class="panel-header">
        <button class="btn primary" onclick="openModelDrawer()">＋ New Model</button>
        <input id="models-search" placeholder="Search name" oninput="renderModelsTable()">
        <select id="models-provider-filter" onchange="renderModelsTable()"></select>
        <select id="models-category-filter" onchange="renderModelsTable()">
          <option value="">All</option>
          <option value="local">Local</option>
          <option value="cloud">Cloud</option>
        </select>
        <button class="btn" onclick="reloadModels()">Reload</button>
      </div>
      <div id="models-summary"></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Name</th><th>Category</th><th>Provider</th><th>Capabilities</th><th>Context</th><th>Workers</th><th>Actions</th></tr></thead>
        <tbody id="models-table"></tbody>
      </table></div>
    </div>
```

- [ ] **Step 3: Add the render functions**

In `dashboard/static/dashboard.js`, add:

```javascript
let modelsData = [];

async function loadModels() {
  const resp = await fetch('/api/models');
  const data = await resp.json();
  modelsData = data.models || [];
  renderModelsTable();
}

function openModelDrawer() { /* Task 8 */ }

function renderModelsTable() {
  const search = (document.getElementById('models-search')?.value || '').toLowerCase();
  const provider = document.getElementById('models-provider-filter')?.value || '';
  const category = document.getElementById('models-category-filter')?.value || '';
  const rows = modelsData.filter(m =>
    (!search || (m.name || '').toLowerCase().includes(search)) &&
    (!provider || m.provider === provider) &&
    (!category || m.category === category));
  document.getElementById('models-table').innerHTML = rows.map(m => `
    <tr>
      <td>${escapeHtml(m.name)}</td>
      <td>${m.category}</td>
      <td>${escapeHtml(m.provider)}</td>
      <td>${(m.capabilities || []).map(c => `<span class="badge">${escapeHtml(c)}</span>`).join(' ')}</td>
      <td>${m.context_length ?? '—'}</td>
      <td>${(m.workers || []).join(', ') || '—'}</td>
      <td>
        <button class="btn" onclick="openModelDrawer('${escapeHtml(m.name)}')">Edit</button>
        <button class="btn" onclick="duplicateModel('${escapeHtml(m.name)}')">Copy</button>
        <button class="btn danger" onclick="deleteModel('${escapeHtml(m.name)}')">Delete</button>
      </td>
    </tr>`).join('') || '<tr><td colspan="7">No models.</td></tr>';
  document.getElementById('models-summary').textContent =
    `Total ${modelsData.length} · Local ${modelsData.filter(m => m.category === 'local').length} · Cloud ${modelsData.filter(m => m.category === 'cloud').length}`;
}

async function reloadModels() {
  await mutateDashboard('/api/models/reload', 'POST');
  await loadModels();
}

async function deleteModel(name) {
  if (!confirm(`Delete ${name}?`)) return;
  await mutateDashboard(`/api/models/${encodeURIComponent(name)}`, 'DELETE');
  await loadModels();
}

function duplicateModel(name) {
  const src = modelsData.find(m => m.name === name);
  if (src) openModelDrawer(null, { ...src, name: `${name}-copy` });
}
```

Call `loadModels()` inside `switchTab` when the models tab is selected (add a branch in `switchTab`).

- [ ] **Step 4: Manual verification**

Run the dashboard, open the Models tab: the table lists models, category/provider filters work, Reload works. (Backend restart required if Task 6 code is new.)

- [ ] **Step 5: Commit**

```bash
git add dashboard/templates/index.html dashboard/static/dashboard.js
git commit -m "feat: add Models tab with list, filters, and delete/copy actions"
```

---

### Task 8: Front-end — edit drawer, capability grid, context fetch, rename flow

**Files:**
- Modify: `dashboard/templates/index.html` (drawer container)
- Modify: `dashboard/static/dashboard.js` (`openModelDrawer`, save, fetch-context, rename dialog)

**Interfaces:**
- Consumes: Task 6 endpoints, `loadModels()`.
- Produces: `openModelDrawer(name, preset)`, `saveModel()`, `fetchModelContext()`, `toggleModelCategory()`.

- [ ] **Step 1: Add the drawer container**

In `dashboard/templates/index.html`:

```html
    <div id="model-drawer" class="drawer" style="display:none">
      <h3 id="model-drawer-title"></h3>
      <label>Name <input id="model-name"></label>
      <label>Category
        <select id="model-category" onchange="toggleModelCategory()">
          <option value="local">Local</option>
          <option value="cloud">Cloud</option>
        </select>
      </label>
      <label>Provider <select id="model-provider"></select></label>
      <div id="model-bindings"></div>
      <div id="model-capabilities"></div>
      <label>Context window <input id="model-context" type="number"></label>
      <button class="btn" onclick="fetchModelContext()">Auto-fetch</button>
      <div id="model-errors"></div>
      <button class="btn primary" onclick="saveModel()">Save</button>
      <button class="btn" onclick="closeModelDrawer()">Cancel</button>
    </div>
```

- [ ] **Step 2: Implement drawer logic**

In `dashboard/static/dashboard.js`:

```javascript
let modelProviders = { local: [], cloud: [] };
let modelCapabilities = [];
let editingModelName = null;

async function loadModelMeta() {
  modelProviders = await (await fetch('/api/models/providers')).json();
  modelCapabilities = (await (await fetch('/api/models/capabilities')).json()).capabilities || [];
}

async function openModelDrawer(name = null, preset = null) {
  await loadModelMeta();
  editingModelName = name;
  const model = preset || modelsData.find(m => m.name === name) || { capabilities: [], context_length: null };
  document.getElementById('model-drawer-title').textContent = name ? `Edit: ${name}` : 'New Model';
  document.getElementById('model-name').value = model.name || '';
  document.getElementById('model-category').value = model.category || 'local';
  document.getElementById('model-context').value = model.context_length ?? '';
  renderModelProviderOptions(model.provider);
  renderModelCapabilities(model.capabilities || []);
  renderModelBindings(model.worker_bindings || []);
  toggleModelCategory();
  document.getElementById('model-drawer').style.display = 'block';
}

function closeModelDrawer() { document.getElementById('model-drawer').style.display = 'none'; }

function toggleModelCategory() {
  const local = document.getElementById('model-category').value === 'local';
  document.getElementById('model-bindings').style.display = local ? 'block' : 'none';
}

function renderModelProviderOptions(selected) {
  const category = document.getElementById('model-category').value;
  const list = modelProviders[category] || [];
  document.getElementById('model-provider').innerHTML = list
    .map(p => `<option value="${escapeHtml(p)}" ${p === selected ? 'selected' : ''}>${escapeHtml(p)}</option>`).join('');
}

function renderModelCapabilities(selected) {
  const groups = ['core', 'reasoning', 'modality', 'other'];
  document.getElementById('model-capabilities').innerHTML = groups.map(g => `
    <div><strong>${g}</strong><br>${modelCapabilities.filter(c => c.group === g).map(c =>
      `<label><input type="checkbox" value="${c.value}" ${selected.includes(c.value) ? 'checked' : ''}> ${c.label}</label>`).join(' ')}</div>`).join('');
}

function renderModelBindings(bindings) {
  document.getElementById('model-bindings').innerHTML =
    `<div><strong>Worker bindings</strong></div>` +
    bindings.map((b, i) => `<div><input value="${escapeHtml(b.node_id || '')}" data-binding-node="${i}"><input type="number" value="${b.port || ''}" data-binding-port="${i}"></div>`).join('') +
    `<button class="btn" onclick="addModelBinding()">＋ Binding</button>`;
}

function addModelBinding() {
  const container = document.getElementById('model-bindings');
  const index = container.querySelectorAll('[data-binding-node]').length;
  container.insertAdjacentHTML('beforeend', `<div><input data-binding-node="${index}"><input type="number" data-binding-port="${index}"></div>`);
}

function collectModelPayload() {
  const category = document.getElementById('model-category').value;
  const payload = {
    name: document.getElementById('model-name').value.trim(),
    provider: document.getElementById('model-provider').value,
    capabilities: Array.from(document.querySelectorAll('#model-capabilities input:checked')).map(i => i.value),
    context_length: document.getElementById('model-context').value ? Number(document.getElementById('model-context').value) : null,
  };
  if (category === 'local') {
    const nodes = document.querySelectorAll('[data-binding-node]');
    const ports = document.querySelectorAll('[data-binding-port]');
    payload.worker_bindings = Array.from(nodes).map((n, i) => ({ node_id: n.value.trim(), port: Number(ports[i].value) }));
  } else {
    payload.worker_ports = [];
  }
  return payload;
}

async function saveModel() {
  const payload = collectModelPayload();
  document.getElementById('model-errors').textContent = '';
  let resp;
  if (editingModelName) {
    resp = await fetch(`/api/models/${encodeURIComponent(editingModelName)}?update_references=true`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  } else {
    resp = await fetch('/api/models', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  }
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    document.getElementById('model-errors').textContent = JSON.stringify(err.detail || err);
    return;
  }
  closeModelDrawer();
  await loadModels();
}

async function fetchModelContext() {
  const name = document.getElementById('model-name').value.trim();
  if (!name) return;
  const resp = await fetch(`/api/models/${encodeURIComponent(name)}/fetch-context`, { method: 'POST' });
  if (resp.ok) document.getElementById('model-context').value = (await resp.json()).context_length;
}
```

- [ ] **Step 3: Manual verification**

Run the dashboard: create a local model with a binding, save, confirm it appears and `/v1/models` shows it without restarting the router; edit context via Auto-fetch; rename a model that has an alias and confirm references update.

- [ ] **Step 4: Commit**

```bash
git add dashboard/templates/index.html dashboard/static/dashboard.js
git commit -m "feat: add model edit drawer with capabilities, bindings, and context fetch"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md` (endpoint list ~line 388)
- Modify: `PROGRESS.md` (append)
- Modify: `TASK.md` (Model Manager section)

- [ ] **Step 1: README**

Under the OpenAI-compatible endpoint list, add a Dashboard subsection:

```markdown
### Dashboard — Models Tab
Manage `config/models.yaml` from the dashboard (add/edit/delete/rename), with local/cloud
classification, multi-select capabilities, editable worker bindings, and context-window fetch.
Backing endpoints: `GET/POST /api/models`, `PUT/DELETE /api/models/{name}`,
`POST /api/models/{name}/fetch-context`, `POST /api/models/reload`. Edits hot-reload without
restarting the router.
```

- [ ] **Step 2: PROGRESS.md**

Append a dated section describing the Models tab, the `model_registry_store`, hot-reload via mtime, the capability additions, and the test counts.

- [ ] **Step 3: TASK.md**

Mark the Model Manager checklist items complete and add an execution-log row.

- [ ] **Step 4: Commit**

```bash
git add README.md PROGRESS.md TASK.md
git commit -m "docs: document Dashboard Models tab and model_registry_store"
```

---

## Self-Review

- **Spec coverage:** D1 (Approach A) → Tasks 2, 5; D2 (image_gen/video annotation) → Task 1; D3 (worker_bindings) → Tasks 3, 8; D4 (rename scan) → Tasks 4, 6, 8. UI → Tasks 7-8; API → Task 6; tests → each task; docs → Task 9.
- **Placeholder scan:** front-end `openModelDrawer` in Task 7 is intentionally a stub and is fully implemented in Task 8; no other TBDs.
- **Type consistency:** store function names (`get_models`, `save_models`, `normalize_model`, `validate_model`, `is_local_model`, `models_mtime`) are used consistently in Tasks 5-8; `scan_model_references`/`update_model_references` consistent between Tasks 4 and 6.
- **Review Focus:** malformed yaml → Task 2 tests; locked file → Task 2 retry test; case/whitespace rename → Task 3 `normalize_model` trims + `validate_model` duplicate check; bad context → Task 3 test; local without bindings / bad port → Task 3 tests.
