from __future__ import annotations

import logging
import time
from typing import Any

from config.settings import settings
from providers.credential_pool import Credential, CredentialPool
from providers.gemini_adapter import GeminiAdapter
from providers.nvidia_nim_adapter import NvidiaNIMAdapter
from providers.ollama_adapter import OllamaAdapter
from providers.ollama_cloud_adapter import OllamaCloudAdapter
from providers.openai_adapter import OpenAIAdapter

logger = logging.getLogger("runtime.orchestration.provider_router")

try:
    from providers.xtts_adapter import XTTSAdapter
except ImportError:
    XTTSAdapter = None  # type: ignore[assignment, misc]

try:
    from providers.faster_whisper_adapter import FasterWhisperAdapter
except ImportError:
    FasterWhisperAdapter = None  # type: ignore[assignment, misc]

NVIDIA_PREFIXES = (
    "meta/", "mistralai/", "nvidia/", "google/", "microsoft/", "baichuan-inc/",
    "deepseek/", "upstage/", "snowflake/", "ibm/", "yola/", "writer/", "z-ai/",
)

_CLOUD_ADAPTERS: dict[str, type] = {
    "nvidia_nim": NvidiaNIMAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "ollama_cloud": OllamaCloudAdapter,
}

_credential_pools: dict[str, CredentialPool] = {}


def _cloud_pool(provider: str) -> CredentialPool | None:
    """Return a cached CredentialPool for *provider*, or None if none configured."""
    if provider not in _CLOUD_ADAPTERS:
        return None
    existing = _credential_pools.get(provider)
    if existing is not None:
        return existing

    cred_configs = settings.cloud_credentials_for(provider)
    if not cred_configs:
        return None

    credentials = [
        Credential(
            api_key=c["api_key"],
            base_url=c.get("base_url"),
            label=c.get("label", ""),
            cooldown_s=c.get("cooldown_s"),
        )
        for c in cred_configs
    ]
    pool = CredentialPool(
        adapter_cls=_CLOUD_ADAPTERS[provider],
        credentials=credentials,
    )
    _credential_pools[provider] = pool
    return pool


def credential_pool_status() -> dict[str, list[dict[str, Any]]]:
    """Return cooldown status for all active credential pools (for dashboard)."""
    status: dict[str, list[dict[str, Any]]] = {}
    for provider_name, pool in _credential_pools.items():
        status[provider_name] = pool.get_cooldown_status()
    return status


def reload_credential_pools() -> None:
    """Clear cached pools so they are re-created from config on next use."""
    _credential_pools.clear()


def adapter(provider: str, worker: dict[str, Any] | None = None) -> Any:
    if provider == "ollama":
        if worker is None:
            raise ValueError("No worker was assigned for Ollama provider.")
        return OllamaAdapter(worker["base_url"], worker=worker)

    pool = _cloud_pool(provider)
    if pool is not None:
        return pool

    if provider == "ollama_cloud":
        return OllamaCloudAdapter()
    if provider == "openai":
        return OpenAIAdapter()
    if provider == "gemini":
        return GeminiAdapter()
    if provider == "nvidia_nim":
        return NvidiaNIMAdapter()
    if provider == "xtts":
        if XTTSAdapter is None:
            raise ValueError("XTTS adapter not available (TTS not installed)")
        return _get_tts_adapter()
    if provider == "asr":
        if FasterWhisperAdapter is None:
            raise ValueError("ASR adapter not available (faster-whisper not installed)")
        return _get_asr_adapter()
    raise ValueError(f"Unsupported provider: {provider}")


_tts_adapter: Any | None = None
_tts_adapter_failed_at: float = 0.0
_ADAPTER_RETRY_COOLDOWN = 30.0


def _get_tts_adapter() -> Any:
    global _tts_adapter, _tts_adapter_failed_at
    if _tts_adapter is not None:
        return _tts_adapter
    if not settings.tts_enabled:
        return None
    if _tts_adapter_failed_at and (time.monotonic() - _tts_adapter_failed_at < _ADAPTER_RETRY_COOLDOWN):
        logger.warning("TTS adapter recently failed, retrying after cooldown (%.0fs remaining)",
                       _ADAPTER_RETRY_COOLDOWN - (time.monotonic() - _tts_adapter_failed_at))
        return None
    try:
        _tts_adapter = XTTSAdapter(
            model_name=settings.tts_model_name,
            device=settings.tts_device,
            voices_dir=settings.tts_voices_dir,
            models_dir=settings.tts_models_dir,
            dtype=settings.tts_dtype,
            max_ref_seconds=settings.tts_max_ref_seconds,
        )
        _tts_adapter_failed_at = 0.0
        return _tts_adapter
    except Exception:
        _tts_adapter_failed_at = time.monotonic()
        logger.exception("TTS adapter creation failed")
        return None


_asr_adapter: Any | None = None
_asr_adapter_failed_at: float = 0.0


def _get_asr_adapter() -> Any:
    global _asr_adapter, _asr_adapter_failed_at
    if _asr_adapter is not None:
        return _asr_adapter
    if not settings.asr_enabled:
        return None
    if _asr_adapter_failed_at and (time.monotonic() - _asr_adapter_failed_at < _ADAPTER_RETRY_COOLDOWN):
        logger.warning("ASR adapter recently failed, retrying after cooldown (%.0fs remaining)",
                       _ADAPTER_RETRY_COOLDOWN - (time.monotonic() - _asr_adapter_failed_at))
        return None
    try:
        _asr_adapter = FasterWhisperAdapter(
            model_name=settings.asr_model,
            device=settings.asr_device,
            device_index=settings.asr_device_index,
            compute_type=settings.asr_compute_type,
            download_dir=settings.asr_models_dir,
        )
        _asr_adapter_failed_at = 0.0
        return _asr_adapter
    except Exception:
        _asr_adapter_failed_at = time.monotonic()
        logger.exception("ASR adapter creation failed")
        return None


def resolve_provider(model: str, registry: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    for prefix, hinted_provider in ROUTE_PREFIXES.items():
        if model.startswith(prefix):
            stripped = model[len(prefix):]
            if hinted_provider in ("openai", "gemini", "nvidia_nim", "ollama_cloud", "xtts", "asr"):
                return hinted_provider, None
            for item in registry.get("models", []):
                if item.get("name") == stripped:
                    bindings = item.get("worker_bindings", [])
                    for b in bindings:
                        base_url = settings.worker_base_url(b)
                        if base_url:
                            return "ollama", {"base_url": base_url}
            return "ollama", None
    clean_model = settings.resolve_model_alias(model)
    for item in registry.get("models", []):
        if item.get("name") in (model, clean_model):
            p = str(item.get("provider", "ollama")).lower()
            if p in ("openai", "gemini", "nvidia_nim", "ollama_cloud"):
                return p, None
            if p == "ollama":
                bindings = item.get("worker_bindings", [])
                for b in bindings:
                    base_url = settings.worker_base_url(b)
                    if base_url:
                        return p, {"base_url": base_url}
            return p, None
    if clean_model.endswith("-cloud") or clean_model.endswith("-cloud-latest"):
        return "ollama_cloud", None
    if clean_model.startswith("gemini"):
        return "gemini", None
    if clean_model.startswith("gpt") or clean_model.startswith(("o1", "o3", "o4")):
        return "openai", None
    if any(clean_model.startswith(p) for p in NVIDIA_PREFIXES) or clean_model.startswith("nemotron"):
        return "nvidia_nim", None
    return "ollama", None


ROUTE_PREFIXES = {
    "ollama_cloud/": "ollama_cloud",
    "nvidia_nim/": "nvidia_nim",
    "anthropic/": "ollama",
    "ollama/": "ollama",
    "openai/": "openai",
    "gemini/": "gemini",
    "xtts/": "xtts",
    "asr/": "asr",
}


def provider_for_model(model: str, registry: dict[str, Any]) -> str:
    for prefix, provider in ROUTE_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    model = settings.resolve_model_alias(model)
    for item in registry.get("models", []):
        if item.get("name") == model:
            return str(item.get("provider", "ollama"))
    if model.endswith("-cloud") or model.endswith("-cloud-latest"):
        return "ollama_cloud"
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("gpt") or model.startswith(("o1", "o3", "o4")):
        return "openai"
    if any(model.startswith(p) for p in NVIDIA_PREFIXES) or model.startswith("nemotron"):
        return "nvidia_nim"
    return "ollama"


def capabilities_for_model(model: str, registry: dict[str, Any]) -> list[str]:
    for item in registry.get("models", []):
        if item.get("name") == model:
            caps = item.get("capabilities", [])
            if isinstance(caps, list):
                return caps
    return ["chat"]


def find_registry_model(model: str, registry: dict[str, Any]) -> dict[str, Any] | None:
    for item in registry.get("models", []):
        if item.get("name") == model:
            return item
    return None


def local_ollama_fallback(
    required_capabilities: list[str] | set[str],
    registry: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    required = set(required_capabilities or ["chat"])
    configured_model = settings.ollama_fallback_model()
    if configured_model:
        configured = ollama_model_for_fallback(configured_model, required, registry)
        if configured is not None:
            return configured
    for model in registry.get("models", []):
        if str(model.get("provider", "ollama")).lower() != "ollama":
            continue
        if not model.get("worker_bindings"):
            continue
        capabilities = set(model.get("capabilities", []))
        if required.issubset(capabilities):
            for binding in model.get("worker_bindings", []):
                base_url = settings.worker_base_url(binding)
                if base_url:
                    return str(model.get("name")), {"base_url": base_url}
    return None


def ollama_model_for_fallback(
    model_name: str,
    required: set[str],
    registry: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    explicit_base_url = settings.ollama_fallback_base_url()
    for model in registry.get("models", []):
        if str(model.get("provider", "ollama")).lower() != "ollama":
            continue
        if model.get("name") != model_name:
            continue
        capabilities = set(model.get("capabilities", []))
        if required and not required.issubset(capabilities):
            return None
        if explicit_base_url:
            return model_name, {"base_url": explicit_base_url}
        for binding in model.get("worker_bindings", []):
            base_url = settings.worker_base_url(binding)
            if base_url:
                return model_name, {"base_url": base_url}
        return None
    return None
