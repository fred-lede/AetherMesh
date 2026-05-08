# Security Sandbox

## Overview
The security sandbox provides isolation for tool execution, preventing untrusted code or commands from affecting the host system beyond configured boundaries.

## Components

### ToolSandbox (`runtime/security/tool_sandbox.py`)
- `run_shell()`: Executes shell commands via subprocess with timeout
- `run_python()`: Executes Python code via isolated subprocess
- `read_file()`: Reads files with path restriction
- `write_file()`: Writes files with path restriction

### MCPSandbox (`runtime/mcp/mcp_sandbox.py`)
- `validate_tool_call()`: Checks tool arguments against policy before execution
- `run_stdio()`: Runs MCP server processes with timeout and output limits
- `create_temp_dir()`: Creates isolated temporary directories

## Path Restrictions
Allowed file paths are configured via `configure_sandbox_paths()`. By default, all paths are allowed. When configured, only paths starting with an allowed prefix are accessible.

## Timeout Controls
- Shell/Python execution: 30-60s default timeout
- HTTP requests: 30-60s timeout
- All timeouts are configurable per tool descriptor

## Audit
All sandbox operations are logged through the audit system, recording:
- Tool name and arguments
- Execution duration
- Success/failure status
- Error messages (if any)
