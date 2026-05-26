# Execution Environment Abstraction

**Date**: 2026-05-25
**Status**: Design Approved
**Area**: Security / Tool Execution

## Problem

AetherMesh has 5 builtin tools (`python`, `shell`, `filesystem`, `web_fetch`, `http_request`) that execute on the host with no process-level isolation. The existing `ToolSandbox` class (`runtime/security/tool_sandbox.py`) provides basic policy checks (path scoping via `ALLOWED_SANDBOX_PATHS`) and uses `subprocess.run()` for `python`/`shell` execution, but:

1. No resource limits (CPU, memory) — a rogue `python` tool can OOM the host
2. No per-execution scratch directory — temp files leak between calls
3. No network control — `shell` tool can initiate outbound connections
4. No integration with `ToolExecutor` — `ToolSandbox` is a standalone class, never called by the execution pipeline
5. Platform differences are not abstracted — `unshare()` / `seccomp()` only available on Linux
6. `ALLOWED_SANDBOX_PATHS` is a module-level global instead of config-driven

## Scope

Build a proper execution environment abstraction layer with two isolation modes:

| Mode | Tools | Mechanism |
|---|---|---|
| **Process isolation** | `python`, `shell` | Subprocess with resource limits + filesystem isolation + network restriction |
| **Policy isolation** | `filesystem`, `web_fetch`, `http_request` | In-process path/network policy checks |

| Subsystem | Files | Action |
|---|---|---|
| Config | `config/settings.py` | Add `sandbox_profiles` field, loaded from YAML |
| Security | `runtime/security/sandbox/` | New module: `profile.py`, `platform.py`, `manager.py`, `mac_sandbox.py`, `linux_sandbox.py` |
| Tool executor | `runtime/tools/tool_executor.py` | Add optional `SandboxManager` injection |
| Tests | `tests/test_execution_environment.py` | Unit tests for all sandbox components |

NOT changing:
- `ToolRegistry`, `ToolCall`, `ToolResult` models
- Existing `ToolSandbox` class (will be superseded, keep for migration period)
- `GraphExecutor`, `AgentLoop`, routing
- Provider adapters

## Design

### 1. SandboxProfile (`runtime/security/sandbox/profile.py`)

```python
@dataclass
class SandboxProfile:
    enabled: bool = True
    sandbox_type: str = "process"          # "process" | "policy"
    allow_network: bool = False
    max_cpu_time_sec: float = 30.0
    max_memory_bytes: int = 524288000      # 500 MB
    allowed_paths: list[str] | None = None
    allowed_domains: list[str] | None = None
    timeout_sec: int = 30
```

Builtin defaults (hardcoded in module):

| Tool | sandbox_type | Network | Memory | Paths |
|---|---|---|---|---|
| `python` | process | blocked | 500 MB | — |
| `shell` | process | blocked | 500 MB | — |
| `filesystem` | policy | — | — | `[CWD]` + scratch |
| `web_fetch` | policy | whitelist | — | — |
| `http_request` | policy | whitelist | — | — |

### 2. PlatformSandbox ABC (`runtime/security/sandbox/platform.py`)

```python
@dataclass
class SandboxResult:
    output: str
    return_code: int
    duration_ms: float
    timed_out: bool = False

class PlatformSandbox(ABC):
    @abstractmethod
    def execute(
        command: list[str],
        profile: SandboxProfile,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult: ...
```

### 3. MacSandbox (`runtime/security/sandbox/mac_sandbox.py`)

| Constraint | Implementation |
|---|---|
| CPU time | `resource.setrlimit(resource.RLIMIT_CPU, (profile.max_cpu_time_sec, profile.max_cpu_time_sec))` |
| Memory | `resource.setrlimit(resource.RLIMIT_DATA, (profile.max_memory_bytes, profile.max_memory_bytes))` |
| Filesystem | `tempfile.mkdtemp()` before exec, `shutil.rmtree()` in `finally` |
| Network | `env={"NO_PROXY": "*"}` + no env proxy vars (best-effort; macOS cannot fully block without firewall) |

### 4. LinuxSandbox (`runtime/security/sandbox/linux_sandbox.py`)

| Constraint | Implementation |
|---|---|
| CPU time | `resource.setrlimit(resource.RLIMIT_CPU, ...)` |
| Memory | `resource.setrlimit(resource.RLIMIT_AS, ...)` |
| Filesystem | `tempfile.mkdtemp()` + `unshare(CLONE_NEWNS)` for mount isolation |
| Network | `unshare(CLONE_NEWNET)` — process has no network interfaces |
| User namespace | `unshare(CLONE_NEWUSER)` maps to nobody (future enhancement) |

### 5. SandboxManager (`runtime/security/sandbox/manager.py`)

```python
class SandboxManager:
    def __init__(
        self,
        profiles: dict[str, SandboxProfile] | None = None,
    ) -> None:
        self._platform: PlatformSandbox
        if sys.platform == "linux":
            self._platform = LinuxSandbox()
        else:
            self._platform = MacSandbox()
        self._profiles = profiles or _builtin_profiles()

    def execute(
        self,
        tool_name: str,
        handler_fn: Callable[[ToolCall], ToolResult],
        call: ToolCall,
    ) -> ToolResult:
        profile = self._profiles.get(tool_name)
        if profile is None:
            profile = _default_profile(tool_name)
        if not profile.enabled:
            return handler_fn(call)

        if profile.sandbox_type == "policy":
            return self._execute_policy(profile, handler_fn, call)
        return self._execute_process(profile, call)
```

### 5a. Policy Isolation (`_execute_policy`)

For `filesystem`, `web_fetch`, `http_request` tools, isolation is in-process policy enforcement:

```python
def _execute_policy(self, profile: SandboxProfile, handler_fn: Callable, call: ToolCall) -> ToolResult:
    if call.name in ("read_file", "write_file", "filesystem"):
        path = str(call.arguments.get("path", ""))
        if profile.allowed_paths:
            resolved = Path(path).resolve()
            allowed = any(str(resolved).startswith(p) for p in profile.allowed_paths)
            if not allowed:
                return ToolResult(call=call, output=f"Path not allowed: {path}", is_error=True)
    elif call.name in ("web_fetch", "http_request"):
        url = str(call.arguments.get("url", ""))
        if profile.allowed_domains and profile.allowed_domains != ["*"]:
            parsed = urllib.parse.urlparse(url)
            if not any(parsed.hostname.endswith(d) for d in profile.allowed_domains):
                return ToolResult(call=call, output=f"Domain not allowed: {parsed.hostname}", is_error=True)
        if not profile.allow_network:
            return ToolResult(call=call, output="Network access disabled for this tool", is_error=True)

    # policy check passed — run the actual handler
    return handler_fn(call)
```

### 5b. Process Isolation (`_execute_process`)

For `python` / `shell` tools, execution is delegated to `PlatformSandbox`:

```python
def _execute_process(self, profile: SandboxProfile, call: ToolCall) -> ToolResult:
    scratch_dir = tempfile.mkdtemp(prefix="aether_sandbox_")
    try:
        if call.name == "python":
            code = str(call.arguments.get("code", ""))
            result = self._platform.execute(
                ["python3", "-c", code],
                profile=profile,
                cwd=scratch_dir,
            )
        elif call.name == "shell":
            command = str(call.arguments.get("command", ""))
            result = self._platform.execute(
                ["sh", "-c", command],
                profile=profile,
                cwd=scratch_dir,
            )
        else:
            return ToolResult(call=call, output=f"No process handler for {call.name}", is_error=True)

        return ToolResult(
            call=call,
            output=result.output,
            is_error=result.return_code != 0,
            duration_ms=result.duration_ms,
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
```

### 6. ToolExecutor Integration

```python
class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        sandbox_manager: SandboxManager | None = None,
    ) -> None:
        self._registry = registry or default_registry
        self._sandbox_manager = sandbox_manager

    def execute(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        descriptor = self._registry.resolve(call.name)
        if not descriptor:
            return ToolResult(call=call, output=f"Tool '{call.name}' not found", is_error=True)
        if not descriptor.handler:
            return ToolResult(call=call, output=f"Tool '{call.name}' has no handler", is_error=True)

        if self._sandbox_manager:
            return self._sandbox_manager.execute(call.name, descriptor.handler, call)

        return descriptor.handler(call)
```

### 7. Config Integration

Settings gains a `sandbox_profiles` field:

```python
@dataclass
class Settings:
    ...
    sandbox_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
```

YAML (`config/sandbox.yaml` or `settings.sandbox_profiles` in main config):

```yaml
sandbox:
  profiles:
    python:
      enabled: true
      max_memory_bytes: 268435456   # 256 MB for python
    shell:
      enabled: true
      allow_network: false
    filesystem:
      enabled: true
      sandbox_type: policy
      allowed_paths: ["/tmp/workspace", "/home/user/data"]
```

Values not specified per tool fall through to builtin defaults.

### 8. Error Handling

| Scenario | Behavior |
|---|---|
| CPU limit exceeded | Subprocess killed by SIGXCPU, `SandboxResult(timed_out=True)` |
| Memory limit exceeded | Subprocess killed by OOM, `SandboxResult(return_code=-9)` |
| Timeout | `subprocess.TimeoutExpired` caught, `SandboxResult(timed_out=True)` |
| Policy violation | `ToolResult(is_error=True, output="Path not allowed: /etc/passwd")` |
| Scratch cleanup fails | Logged as warning, never crashes caller |
| Unknown tool (no profile) | Falls through to `_default_profile()` with conservative settings |

### 9. Testing

| Test | What it covers |
|---|---|
| `test_sandbox_profile_defaults` | Default values by tool name |
| `test_sandbox_profile_override` | Config merge with builtin defaults |
| `test_mac_sandbox_execute` | subprocess execution with resource limits |
| `test_mac_sandbox_cpu_timeout` | CPU limit enforcement |
| `test_mac_sandbox_memory_oom` | Memory limit enforcement (large allocation) |
| `test_sandbox_manager_passthrough` | profile.enabled=False → handler_fn called directly |
| `test_sandbox_manager_policy` | Policy isolation for filesystem tool |
| `test_sandbox_manager_process` | Process isolation for python tool |
| `test_sandbox_scratch_cleanup` | Temp directory removed after execution |
| `test_tool_executor_integration` | ToolExecutor with injected SandboxManager |
| `test_linux_sandbox_network_isolation` | unshare(CLONE_NEWNET) blocks network |

## Data Flow

```
Client → router → ToolExecutor.execute(call)
                           │
                    sandbox_manager? ──no──→ handler_fn(call) → ToolResult
                           │
                          yes
                           │
                    SandboxManager.execute(tool_name, handler_fn, call)
                           │
                    resolve profile for tool
                           │
                    enabled? ──no──→ handler_fn(call) → ToolResult
                           │
                          yes
                           │
                    ┌──── type? ────┐
                    │               │
                "policy"        "process"
                    │               │
              policy check     PlatformSandbox.execute()
              on paths/network   ├── resource.setrlimit()
                    │            ├── tempfile.mkdtemp()
              pass/fail →        ├── subprocess.run()
              ToolResult         └── cleanup → SandboxResult
                                       │
                                 ToolResult
```

## Migration

Existing `ToolSandbox` class remains in place for one release cycle. All new code paths use `SandboxManager`. After migration is confirmed stable:
1. Remove `runtime/security/tool_sandbox.py`
2. Remove `ALLOWED_SANDBOX_PATHS` global
3. Remove `configure_sandbox_paths()`
