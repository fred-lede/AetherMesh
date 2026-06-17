from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _env_bool(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _detect_config_dir() -> Path:
    env_dir = os.getenv("AIIH_CONFIG_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    try:
        file_dir = Path(__file__).resolve().parent
        if (file_dir / "models.yaml").exists():
            return file_dir
    except NameError:
        pass
    cwd_config = Path.cwd() / "config"
    if (cwd_config / "models.yaml").exists():
        return cwd_config
    return Path.cwd()


@dataclass(slots=True)
class Settings:
    host: str = field(default_factory=lambda: os.getenv("AIIH_HOST", "0.0.0.0"))
    router_port: int = field(default_factory=lambda: _env_int("AIIH_ROUTER_PORT", 8001))
    dashboard_port: int = field(default_factory=lambda: _env_int("AIIH_DASHBOARD_PORT", 9001))
    metrics_port: int = field(default_factory=lambda: _env_int("AIIH_METRICS_PORT", 9100))
    control_plane_port: int = field(default_factory=lambda: _env_int("AIIH_CONTROL_PORT", 9200))
    worker_rpc_port: int = field(default_factory=lambda: _env_int("AIIH_WORKER_RPC_PORT", 9300))
    node_agent_port: int = field(default_factory=lambda: _env_int("AIIH_NODE_PORT", 9400))
    control_plane_url: str = field(default_factory=lambda: os.getenv("AIIH_CONTROL_URL", "http://127.0.0.1:9200"))
    router_url: str = field(default_factory=lambda: os.getenv("AIIH_ROUTER_URL", "http://127.0.0.1:8001"))
    anthropic_router_url: str = field(default_factory=lambda: os.getenv("AIIH_ANTHROPIC_URL", "http://127.0.0.1:8002"))
    metrics_url: str = field(default_factory=lambda: os.getenv("AIIH_METRICS_URL", "http://127.0.0.1:9100"))
    redis_url: str = field(default_factory=lambda: os.getenv("AIIH_REDIS_URL", "redis://127.0.0.1:6379/0"))
    request_timeout_s: int = field(default_factory=lambda: _env_int("AIIH_REQUEST_TIMEOUT", 120))
    heartbeat_interval_s: int = field(default_factory=lambda: _env_int("AIIH_HEARTBEAT_INTERVAL", 15))
    stale_after_s: int = field(default_factory=lambda: _env_int("AIIH_STALE_AFTER", 45))
    max_worker_queue_size: int = field(default_factory=lambda: _env_int("AIIH_MAX_WORKER_QUEUE", 8))
    worker_assignment_ttl_s: int = field(default_factory=lambda: _env_int("AIIH_WORKER_ASSIGNMENT_TTL", 900))
    max_task_retries: int = field(default_factory=lambda: _env_int("AIIH_MAX_TASK_RETRIES", 3))
    worker_degrade_after_errors: int = field(default_factory=lambda: _env_int("AIIH_WORKER_DEGRADE_AFTER_ERRORS", 2))
    worker_degrade_cooldown_s: int = field(default_factory=lambda: _env_int("AIIH_WORKER_DEGRADE_COOLDOWN", 30))
    provider_cooldown_s: int = field(default_factory=lambda: _env_int("AIIH_PROVIDER_COOLDOWN", 180))
    dashboard_refresh_s: int = field(default_factory=lambda: _env_int("AIIH_DASHBOARD_REFRESH", 5))
    dashboard_auth_enabled: bool = field(default_factory=lambda: _env_bool("AIIH_DASHBOARD_AUTH_ENABLED", "false"))
    dashboard_auth_username: str = field(default_factory=lambda: os.getenv("AIIH_DASHBOARD_AUTH_USERNAME", "admin"))
    dashboard_auth_password: str = field(default_factory=lambda: os.getenv("AIIH_DASHBOARD_AUTH_PASSWORD", ""))
    debug_tool_calls: bool = field(default_factory=lambda: _env_bool("AIIH_DEBUG_TOOL_CALLS", "false"))
    debug_responses: bool = field(default_factory=lambda: _env_bool("AIIH_DEBUG_RESPONSES", "false"))
    server_tool_mode: str = field(default_factory=lambda: os.getenv("AIIH_SERVER_TOOL_MODE", "reject").strip().lower())
    web_server_tools_enabled: bool = field(default_factory=lambda: _env_bool("AIIH_WEB_SERVER_TOOLS_ENABLED", "false"))
    web_tools_auto_search: bool = field(default_factory=lambda: _env_bool("AIIH_WEB_TOOLS_AUTO_SEARCH", "false"))
    web_tool_timeout_s: int = field(default_factory=lambda: _env_int("AIIH_WEB_TOOL_TIMEOUT", 15))
    web_search_max_results: int = field(default_factory=lambda: _env_int("AIIH_WEB_SEARCH_MAX_RESULTS", 5))
    task_prune_enabled: bool = field(default_factory=lambda: _env_bool("AIIH_TASK_PRUNE_ENABLED", "true"))
    task_prune_hour: int = field(default_factory=lambda: _env_int("AIIH_TASK_PRUNE_HOUR", 3))
    task_prune_minute: int = field(default_factory=lambda: _env_int("AIIH_TASK_PRUNE_MINUTE", 30))
    task_retention_hours: int = field(default_factory=lambda: _env_int("AIIH_TASK_RETENTION_HOURS", 72))
    task_prune_statuses: list[str] = field(
        default_factory=lambda: _env_csv("AIIH_TASK_PRUNE_STATUSES", "completed,failed")
    )
    # Rate limiting (per IP)
    rate_limit_enabled: bool = field(default_factory=lambda: _env_bool("AIIH_RATE_LIMIT_ENABLED", "true"))
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int("AIIH_RATE_LIMIT_PER_MINUTE", 60))
    rate_limit_burst: int = field(default_factory=lambda: _env_int("AIIH_RATE_LIMIT_BURST", 10))
    ssl_certfile: str = field(default_factory=lambda: os.getenv("AIIH_SSL_CERTFILE", "").strip())
    ssl_keyfile: str = field(default_factory=lambda: os.getenv("AIIH_SSL_KEYFILE", "").strip())
    local_worker_ports: list[int] = field(default_factory=lambda: [11434, 11435, 11436, 11437])
    config_dir: Path = field(default_factory=_detect_config_dir)
    responses_max_turns: int = field(default_factory=lambda: _env_int("AIIH_RESPONSES_MAX_TURNS", 16))
    sandbox_profiles: dict[str, Any] = field(default_factory=dict)
    upload_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AIIH_UPLOAD_DIR", "/tmp/aethermesh/uploads"))
    )
    max_upload_size_mb: int = field(default_factory=lambda: _env_int("AIIH_MAX_UPLOAD_SIZE_MB", 50))
    allowed_upload_mime_types: list[str] = field(
        default_factory=lambda: _env_csv("AIIH_ALLOWED_UPLOAD_MIME_TYPES",
            ",".join([
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "text/plain",
                "text/markdown",
            ])
        )
    )
    tesseract_langs: str = field(default_factory=lambda: os.getenv("AIIH_TESSERACT_LANGS", "eng"))
    file_cleanup_ttl_seconds: int = field(default_factory=lambda: _env_int("AIIH_FILE_CLEANUP_TTL", 600))
    ocr_concurrency: int = field(default_factory=lambda: _env_int("AIIH_OCR_CONCURRENCY", 2))

    @property
    def tls_enabled(self) -> bool:
        return bool(self.ssl_certfile and self.ssl_keyfile)

    @property
    def api_scheme(self) -> str:
        return "https" if self.tls_enabled else "http"

    def config_path(self, filename: str) -> Path:
        return self.config_dir / filename

    def load_yaml(self, filename: str) -> dict[str, Any]:
        path = self.config_path(filename)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def model_registry(self) -> dict[str, Any]:
        return self.load_yaml("models.yaml")

    def routing_rules_config(self) -> dict[str, Any]:
        return self.load_yaml("routing_rules.yaml")

    def model_aliases(self) -> dict[str, str]:
        config = self.routing_rules_config().get("model_aliases", {})
        if not isinstance(config, dict):
            return {}

        entries = config.get("entries")
        if isinstance(entries, dict):
            aliases = {str(alias): str(target) for alias, target in entries.items()}
            prefix = str(config.get("alias_prefix", "") or "").strip().strip("/")
            if prefix:
                aliases.update({f"{prefix}/{alias}": target for alias, target in list(aliases.items())})
            return aliases

        reserved_keys = {"alias_prefix", "entries"}
        return {str(alias): str(target) for alias, target in config.items() if alias not in reserved_keys}

    def model_alias_entries(self) -> dict[str, str]:
        config = self.routing_rules_config().get("model_aliases", {})
        if not isinstance(config, dict):
            return {}
        entries = config.get("entries")
        if isinstance(entries, dict):
            return {str(alias): str(target) for alias, target in entries.items()}
        reserved_keys = {"alias_prefix", "entries"}
        return {str(alias): str(target) for alias, target in config.items() if alias not in reserved_keys}

    def model_alias_prefix(self) -> str:
        config = self.routing_rules_config().get("model_aliases", {})
        if not isinstance(config, dict):
            return ""
        return str(config.get("alias_prefix", "") or "").strip().strip("/")

    def strip_model_route_prefix(self, model: str) -> str:
        clean_model = str(model or "")
        for prefix in ("anthropic/", "nvidia_nim/", "ollama_cloud/", "ollama/", "openai/", "gemini/"):
            if clean_model.startswith(prefix):
                return clean_model[len(prefix):]
        return clean_model

    def resolve_model_alias(self, model: str) -> str:
        aliases = self.model_aliases()
        if not aliases:
            return self.strip_model_route_prefix(model)

        raw_model = str(model or "")
        clean_model = self.strip_model_route_prefix(raw_model)
        for candidate in (raw_model, clean_model):
            target = aliases.get(candidate)
            if target:
                return target
        return clean_model

    def ollama_fallback_model(self) -> str:
        from_env = os.getenv("AIIH_OLLAMA_FALLBACK_MODEL", "").strip()
        if from_env:
            return from_env
        fallback = self.routing_rules_config().get("fallback", {})
        if isinstance(fallback, dict):
            return str(fallback.get("ollama_default_model", "") or "").strip()
        return ""

    def ollama_fallback_base_url(self) -> str:
        from_env = os.getenv("AIIH_OLLAMA_FALLBACK_BASE_URL", "").strip()
        if from_env:
            return from_env.rstrip("/")
        fallback = self.routing_rules_config().get("fallback", {})
        if isinstance(fallback, dict):
            return str(fallback.get("ollama_base_url", "") or "").strip().rstrip("/")
        return ""

    def cluster_config(self) -> dict[str, Any]:
        return self.load_yaml("cluster.yaml")

    def node_hosts(self) -> dict[str, str]:
        configured = self.cluster_config().get("node_hosts", {})
        if not isinstance(configured, dict):
            return {}
        return {str(node_id): str(host) for node_id, host in configured.items() if str(host).strip()}

    def worker_base_url(self, binding: dict[str, Any]) -> str | None:
        explicit_base_url = str(binding.get("base_url", "") or "").strip().rstrip("/")
        if explicit_base_url:
            return explicit_base_url

        port = binding.get("port")
        if port is None:
            return None

        host = str(binding.get("host") or binding.get("ip") or "").strip()
        node_id = str(binding.get("node_id", "") or "").strip()
        if not host and node_id:
            host = self.node_hosts().get(node_id, "")
        if not host and node_id:
            host = node_id if self._looks_like_host(node_id) else "127.0.0.1"
        if not host:
            host = "127.0.0.1"
        return f"http://{host}:{int(port)}"

    def _looks_like_host(self, value: str) -> bool:
        lowered = value.lower()
        if lowered in {"localhost", "127.0.0.1", "::1"}:
            return True
        return "." in value or ":" in value

    def local_node_id(self) -> str:
        from_env = os.getenv("AIIH_NODE_ID", "").strip()
        if from_env:
            return from_env
        configured = self.cluster_config().get("local_node_id")
        return str(configured or "node-local")

    def local_node_ip(self) -> str | None:
        configured = os.getenv("AIIH_NODE_IP", "").strip()
        return configured or None

    def sandbox_manager(self) -> Any:
        from runtime.security.sandbox.manager import SandboxManager
        from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles

        profiles = builtin_profiles()
        for tool_name, overrides in self.sandbox_profiles.items():
            if not isinstance(overrides, dict):
                continue
            if tool_name in profiles:
                existing = profiles[tool_name]
                profiles[tool_name] = SandboxProfile(
                    enabled=overrides.get("enabled", existing.enabled),
                    sandbox_type=overrides.get("sandbox_type", existing.sandbox_type),
                    allow_network=overrides.get("allow_network", existing.allow_network),
                    max_cpu_time_sec=overrides.get("max_cpu_time_sec", existing.max_cpu_time_sec),
                    max_memory_bytes=overrides.get("max_memory_bytes", existing.max_memory_bytes),
                    allowed_paths=overrides.get("allowed_paths", existing.allowed_paths),
                    allowed_domains=overrides.get("allowed_domains", existing.allowed_domains),
                    timeout_sec=overrides.get("timeout_sec", existing.timeout_sec),
                )
            else:
                profiles[tool_name] = SandboxProfile(
                    enabled=overrides.get("enabled", True),
                    sandbox_type=overrides.get("sandbox_type", "process"),
                    allow_network=overrides.get("allow_network", False),
                    max_cpu_time_sec=overrides.get("max_cpu_time_sec", 30.0),
                    max_memory_bytes=overrides.get("max_memory_bytes", 524288000),
                    allowed_paths=overrides.get("allowed_paths", None),
                    allowed_domains=overrides.get("allowed_domains", None),
                    timeout_sec=overrides.get("timeout_sec", 30),
                )

        return SandboxManager(profiles=profiles)


settings = Settings()
