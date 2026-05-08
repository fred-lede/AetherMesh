from __future__ import annotations

from typing import Any

from config.settings import settings
from providers.gemini_adapter import GeminiAdapter
from providers.nvidia_nim_adapter import NvidiaNIMAdapter
from providers.ollama_adapter import OllamaAdapter
from providers.ollama_cloud_adapter import OllamaCloudAdapter
from providers.openai_adapter import OpenAIAdapter

NVIDIA_PREFIXES = (
    "meta/", "mistralai/", "nvidia/", "google/", "microsoft/", "baichuan-inc/",
    "deepseek/", "upstage/", "snowflake/", "ibm/", "yola/", "writer/", "z-ai/",
)


def adapter(provider: str, worker: dict[str, Any] | None = None) -> Any:
    if provider == "ollama":
        if worker is None:
            raise ValueError("No worker was assigned for Ollama provider.")
        return OllamaAdapter(worker["base_url"])
    if provider == "ollama_cloud":
        return OllamaCloudAdapter()
    if provider == "openai":
        return OpenAIAdapter()
    if provider == "gemini":
        return GeminiAdapter()
    if provider == "nvidia_nim":
        return NvidiaNIMAdapter()
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_provider(model: str, registry: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    clean_model = settings.resolve_model_alias(model)
    for item in registry.get("models", []):
        if item.get("name") in (model, clean_model):
            p = str(item.get("provider", "ollama")).lower()
            if p in ("openai", "gemini", "nvidia_nim", "ollama_cloud"):
                return p, None
            if p == "ollama":
                bindings = item.get("worker_bindings", [])
                if bindings:
                    b = bindings[0]
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


def provider_for_model(model: str, registry: dict[str, Any]) -> str:
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
            binding = model.get("worker_bindings", [])[0]
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
        bindings = model.get("worker_bindings", [])
        if not bindings:
            return None
        binding = bindings[0]
        base_url = settings.worker_base_url(binding)
        if not base_url:
            return None
        return model_name, {"base_url": base_url}
    return None
