# MCP Bridge Pattern

## Overview
The MCP Tool Bridge connects the Tool Runtime with external MCP servers, allowing tools registered on remote MCP servers to be called through the same unified ToolCall/ToolResult interface as built-in tools.

## Architecture

```
Client Request
  → Tool Runtime (runtime/tools/)
    → Tool Registry resolves tool name
      → MCPToolBridge (runtime/mcp/mcp_tool_bridge.py)
        → MCPSessionManager (creates/reuses sessions)
          → MCPSession (stdio or SSE transport)
            → External MCP Server
```

## How It Works

1. Tools are discovered when `MCPSessionManager.get_or_create()` calls `session.list_tools()`
2. Each discovered tool is registered in `MCPRegistry` with a `server_name` reference
3. When `MCPToolBridge.call_mcp_tool()` is invoked, it:
   - Looks up the tool in `MCPRegistry` to find the server
   - Gets or creates a session to that server
   - Sends a `tools/call` JSON-RPC request
   - Wraps the response in a standard `ToolResult`

## Transport Support
- **stdio**: Subprocess with stdin/stdout JSON-RPC
- **SSE**: HTTP Server-Sent Events transport (planned)

## Sandbox Integration
All MCP tool calls pass through `MCPSandbox.validate_tool_call()` before execution, enforcing path restrictions and URL allowlists.
