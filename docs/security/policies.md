# Security Layer

## Components

| Module | Purpose |
|---|---|
| `tool_sandbox.py` | Sandbox for shell/python/filesystem tool execution |
| `prompt_firewall.py` | Detect prompt injection attempts |
| `secret_detection.py` | Detect and redact secrets in model output |
| `tool_policy.py` | Server tool policy (reject/local/passthrough) |
| `audit_log.py` | Security audit event logging |

## Tool Policy

Three modes for Anthropic server tools:
- `reject` (default): Block server tools for OpenAI-compatible providers
- `local`: AetherMesh handles web_search/web_fetch locally
- `passthrough`: Let client/runtime handle server tools

## Tool Sandbox

The tool sandbox restricts dangerous operations:

- **Shell**: Execution via subprocess with timeout
- **Filesystem**: Path whitelist (`ALLOWED_SANDBOX_PATHS`)
- **Python**: Sandboxed subprocess with timeout

Configure allowed paths:

```python
from runtime.security.tool_sandbox import configure_sandbox_paths
configure_sandbox_paths(["/home/user/projects", "/tmp/ai"])
```

## Configuration

```yaml
# config/security_policy.yaml
tool_permissions:
  shell:
    allowed: false
  filesystem:
    allowed_paths: ["/home/user/projects", "/tmp/ai"]
    read_only: false
  web_search:
    allowed: true
  python:
    allowed: false
```
