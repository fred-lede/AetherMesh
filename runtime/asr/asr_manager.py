from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from cluster.gpu_discovery import discover_gpus
from config.settings import settings

logger = logging.getLogger("runtime.asr.asr_manager")

LOCAL_ENGINES: dict[str, list[str]] = {
    "faster_whisper": ["large-v3", "medium", "small"],
    "qwen_asr": ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"],
}

_DEFAULT_MODELS: dict[str, str] = {
    "faster_whisper": "large-v3",
    "qwen_asr": "Qwen/Qwen3-ASR-1.7B",
}

_GPU_CACHE_TTL_S = 5.0


class ASRConfigError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


@dataclass
class ASRTarget:
    target_id: str
    kind: str
    node_id: str | None = None
    base_url: str | None = None
    device_index: int | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_local(target: ASRTarget, provider: str, model: str) -> Any:
    cuda = target.kind == "local_cuda"
    device_index = target.device_index if target.device_index is not None else 0
    if provider == "qwen_asr":
        from providers.qwen_asr_adapter import QwenASRAdapter

        return QwenASRAdapter(
            model_name=model,
            device="cuda" if cuda else "cpu",
            device_index=device_index,
            download_dir=settings.asr_models_dir,
        )
    from providers.faster_whisper_adapter import FasterWhisperAdapter

    return FasterWhisperAdapter(
        model_name=model,
        device="cuda" if cuda else "cpu",
        device_index=device_index,
        compute_type=settings.asr_compute_type if cuda else "int8",
        download_dir=settings.asr_models_dir,
    )


def _build_node(target: ASRTarget, provider: str, model: str) -> Any:
    from providers.remote_asr_adapter import RemoteASRAdapter

    return RemoteASRAdapter(
        node_id=target.node_id or "",
        base_url=target.base_url or "",
        token=os.getenv("AIIH_ASR_WORKER_TOKEN") or None,
    )


DEFAULT_BUILDERS: dict[str, Callable[[ASRTarget, str, str], Any]] = {
    "local_cuda": _build_local,
    "local_cpu": _build_local,
    "node": _build_node,
}


def default_dispose_adapter(adapter: Any) -> None:
    del adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _fetch_cluster_nodes() -> list[dict[str, Any]]:
    response = requests.get(f"{settings.control_plane_url}/cluster/nodes", timeout=3)
    response.raise_for_status()
    payload = response.json()
    nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    return [node for node in nodes if isinstance(node, dict)]


class ASRManager:
    def __init__(
        self,
        builders: dict[str, Callable[[ASRTarget, str, str], Any]] | None = None,
        dispose: Callable[[Any], None] | None = None,
        persist: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._builders = builders if builders is not None else DEFAULT_BUILDERS
        self._dispose = dispose if dispose is not None else default_dispose_adapter
        self._persist = persist if persist is not None else settings.save_asr_config
        self._lock = threading.RLock()
        self._state = "idle"
        self._active_adapter: Any = None
        self._active_info: dict[str, Any] | None = None
        self._requested_info: dict[str, Any] | None = None
        self._error: str | None = None
        self._version = 0
        self._gpu_cache: list[dict[str, Any]] = []
        self._gpu_cache_at = 0.0

    def version(self) -> int:
        with self._lock:
            return self._version

    def get_adapter(self) -> Any:
        with self._lock:
            if self._state == "disabled":
                return None
            return self._active_adapter

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "active": dict(self._active_info) if self._active_info else None,
                "requested": dict(self._requested_info) if self._requested_info else None,
                "error": self._error,
                "version": self._version,
            }

    def _gpus_cached(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if self._gpu_cache and now - self._gpu_cache_at < _GPU_CACHE_TTL_S:
                return self._gpu_cache
        try:
            gpus = discover_gpus()
        except Exception:
            gpus = []
        with self._lock:
            self._gpu_cache = gpus
            self._gpu_cache_at = now
        return gpus

    def resolve_target(self, target_id: str) -> ASRTarget:
        if not target_id or not isinstance(target_id, str):
            raise ASRConfigError("target_id is required")
        if target_id == "local:cpu":
            return ASRTarget(target_id=target_id, kind="local_cpu")
        if target_id.startswith("local:cuda:"):
            try:
                index = int(target_id.rsplit(":", 1)[1])
            except ValueError as exc:
                raise ASRConfigError(f"invalid target_id '{target_id}'") from exc
            return ASRTarget(target_id=target_id, kind="local_cuda", device_index=index)
        if target_id.startswith("node:"):
            node_id = target_id[len("node:"):]
            base_url = self._resolve_node_base_url(node_id)
            return ASRTarget(target_id=target_id, kind="node", node_id=node_id, base_url=base_url)
        raise ASRConfigError(
            f"unknown target_id '{target_id}'",
            details={"targets": [t["id"] for t in self.list_targets()["local"]]},
        )

    def _resolve_node_base_url(self, node_id: str) -> str:
        try:
            nodes = _fetch_cluster_nodes()
        except requests.RequestException as exc:
            raise ASRConfigError(f"control-plane unreachable: {exc}", status_code=503) from exc
        for node in nodes:
            if node.get("node_id") != node_id:
                continue
            metadata = node.get("metadata") or {}
            asr = metadata.get("asr_worker") or {}
            if not asr.get("healthy"):
                raise ASRConfigError(
                    f"node '{node_id}' has no reachable asr_worker",
                    details={"node_id": node_id},
                )
            ip = node.get("ip") or "127.0.0.1"
            return f"http://{ip}:{int(asr.get('port') or 9450)}"
        raise ASRConfigError(
            f"unknown node '{node_id}'",
            status_code=404,
            details={"known_nodes": [n.get("node_id") for n in nodes]},
        )

    def _validate(self, target: ASRTarget, provider: str, model: str) -> tuple[str, str]:
        if target.kind == "node":
            return "remote_asr", ""
        if provider not in LOCAL_ENGINES:
            raise ASRConfigError(
                f"unknown provider '{provider}'",
                details={"valid_providers": sorted(LOCAL_ENGINES)},
            )
        if not model:
            model = _DEFAULT_MODELS[provider]
        elif model not in LOCAL_ENGINES[provider]:
            raise ASRConfigError(
                f"unknown model '{model}' for provider '{provider}'",
                details={"valid_models": list(LOCAL_ENGINES[provider])},
            )
        if target.kind == "local_cuda":
            gpu_ids = [gpu.get("id") for gpu in self._gpus_cached()]
            if target.device_index not in gpu_ids:
                raise ASRConfigError(
                    f"device_index {target.device_index} is not available on this host",
                    details={"valid_gpu_ids": gpu_ids},
                )
        return provider, model

    def preload(self) -> None:
        with self._lock:
            if self._state != "idle":
                return
            if not settings.asr_enabled:
                self._state = "disabled"
                logger.info("ASR disabled; manager state=disabled")
                return
            config = settings.load_asr_config()
            if config:
                logger.info("ASR config loaded from asr_config.json: %s", config)
            target_id = str(config.get("target_id") or self._env_default_target_id())
            provider = str(config.get("provider") or settings.asr_provider).strip().lower()
            model = str(config.get("model") or "").strip()
            self._requested_info = {"target_id": target_id, "provider": provider, "model": model, "loaded_at": None}
            self._state = "loading"
        thread = threading.Thread(
            target=self._preload_worker,
            args=(target_id, provider, model),
            daemon=True,
            name="asr-preload",
        )
        thread.start()

    def _env_default_target_id(self) -> str:
        if settings.asr_device.strip().lower() == "cpu":
            return "local:cpu"
        return f"local:cuda:{settings.asr_device_index}"

    def _preload_worker(self, target_id: str, provider: str, model: str) -> None:
        try:
            target = self.resolve_target(target_id)
            provider, model = self._validate(target, provider, model)
        except ASRConfigError as exc:
            with self._lock:
                self._state = "failed"
                self._error = exc.message
            logger.error("ASR preload failed: %s", exc.message)
            return
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._error = str(exc)
            logger.exception("ASR preload failed")
            return
        self._load_worker(target, provider, model)

    def reload(self, target_id: str, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        resolved_provider = (provider or settings.asr_provider).strip().lower()
        resolved_model = (model or "").strip()
        target = self.resolve_target(target_id)
        resolved_provider, resolved_model = self._validate(target, resolved_provider, resolved_model)
        with self._lock:
            if self._state == "loading":
                raise ASRConfigError("an ASR reload is already in progress", status_code=409)
            self._requested_info = {
                "target_id": target.target_id,
                "provider": resolved_provider,
                "model": resolved_model,
                "loaded_at": None,
            }
            self._state = "loading"
        try:
            self._persist({
                "target_id": target.target_id,
                "provider": resolved_provider,
                "model": resolved_model,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.exception("ASR config persist failed; continuing with reload")
        thread = threading.Thread(
            target=self._load_worker,
            args=(target, resolved_provider, resolved_model),
            daemon=True,
            name="asr-reload",
        )
        with self._lock:
            thread.start()
            return self.snapshot()

    def _load_worker(self, target: ASRTarget, provider: str, model: str) -> None:
        started = time.monotonic()
        try:
            builder = self._builders.get(target.kind)
            if builder is None:
                raise ASRConfigError(f"unsupported target kind '{target.kind}'")
            adapter = builder(target, provider, model)
            if target.kind == "node":
                health = adapter.health_check()
                if not health.get("reachable", True):
                    raise ASRConfigError(
                        f"asr worker on node '{target.node_id}' unreachable at {target.base_url}",
                        status_code=503,
                    )
                provider = str(health.get("engine") or "remote_asr")
                model = str(health.get("model") or "")
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._error = str(exc)
            logger.exception(
                "ASR load failed (target=%s provider=%s model=%s); keeping previous adapter",
                target.target_id, provider, model,
            )
            return
        with self._lock:
            previous = self._active_adapter
            self._active_adapter = adapter
            self._active_info = {
                "target_id": target.target_id,
                "provider": provider,
                "model": model,
                "loaded_at": _now_iso(),
            }
            self._requested_info = dict(self._active_info)
            self._version += 1
            self._state = "ready"
            self._error = None
        logger.info(
            "ASR adapter active target=%s provider=%s model=%s load=%.1fs version=%d",
            target.target_id, provider, model, time.monotonic() - started, self.version(),
        )
        if previous is not None:
            try:
                self._dispose(previous)
            except Exception:
                logger.exception("ASR adapter disposal failed")

    def list_targets(self) -> dict[str, Any]:
        gpus = self._gpus_cached()
        local = [
            {
                "id": f"local:cuda:{gpu.get('id')}",
                "name": gpu.get("name"),
                "memory_free_mb": gpu.get("memory_free_mb"),
            }
            for gpu in gpus
        ]
        local.append({"id": "local:cpu", "name": "CPU", "memory_free_mb": None})
        nodes: list[dict[str, Any]] = []
        nodes_unreachable = False
        try:
            cluster_nodes = _fetch_cluster_nodes()
        except requests.RequestException:
            nodes_unreachable = True
            cluster_nodes = []
        for node in cluster_nodes:
            metadata = node.get("metadata") or {}
            asr = metadata.get("asr_worker") or {}
            nodes.append({
                "id": f"node:{node.get('node_id')}",
                "reachable": bool(asr.get("healthy")),
                "engine": asr.get("engine"),
                "model": asr.get("model"),
            })
        return {"local": local, "nodes": nodes, "nodes_unreachable": nodes_unreachable}


asr_manager = ASRManager()
