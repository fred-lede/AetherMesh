# MCP Gateway

AetherMesh acts as an MCP (Model Context Protocol) Gateway, proxying connections
between clients (Claude Code, Cursor, etc.) and MCP servers.

## Architecture

```
Claude Code
  ↓ (MCP stdio/SSE)
AetherMesh MCP Gateway (runtime/mcp/)
  ├── filesystem MCP server
  ├── git MCP server
  ├── browser MCP server
  └── custom MCP servers
```

## Components

| Module | Purpose |
|---|---|
| `mcp_registry.py` | Register/discover MCP servers and their tools |
| `mcp_session_manager.py` | Manage client→MCP session lifecycle |
| `mcp_tool_bridge.py` | Bridge MCP tools into Tool Runtime |

## MCP Tool Bridge

MCP tools are registered into the Tool Runtime via `mcp_tool_bridge.py`,
allowing seamless execution alongside built-in tools:

```python
ToolCall(name="read_file")
  → ToolRegistry resolves to MCP tool entry
  → MCPToolBridge calls MCP server via MCPSession
  → Result returned as ToolResult
```

## Configuration

Add MCP servers via config file or API:

```yaml
# config/mcp_servers.yaml
servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
  git:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git"]
```
