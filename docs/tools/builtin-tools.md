# Built-in Tools

AetherMesh ships with the following built-in tools, auto-registered via `runtime/tools/builtin/__init__.py`:

| Tool | Description | Requires Confirmation |
|------|-------------|----------------------|
| `shell` | Execute shell commands | Yes |
| `read_file` | Read file contents (UTF-8) | No |
| `write_file` | Write content to file | Yes |
| `python` | Execute Python code | Yes |
| `http_request` | Make HTTP requests | Yes |
| `web_search` | Search the web (Tavily/Serper/DuckDuckGo) | Yes |
| `web_fetch` | Fetch and extract web page content | Yes |

## Registration
Tools are auto-registered at import time by `runtime/tools/builtin/__init__.py` calling `register_all()`. Each tool module provides:
- `register(registry)`: Registers the tool with a ToolRegistry
- `available()`: Returns whether the tool is available on the current platform

## Input Schemas
Each tool defines an OpenAI-compatible JSON Schema for its parameters. The schema is used for:
- Tool call validation
- Documentation generation
- Model tool-use training

## Sandbox Integration
Tools that execute code or access files (`shell`, `python`, `read_file`, `write_file`) route through `runtime/security/tool_sandbox.py` for path restriction and timeout enforcement.
