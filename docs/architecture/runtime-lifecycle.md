# Runtime Lifecycle

## Request Execution Flow

```
1. Client sends request → Protocol Adapter (router/)
2. Adapter parses format → converts to internal representation
3. Capability Detection (runtime/orchestration/capabilities.py)
   - Infer required capabilities from request
4. Routing Engine (runtime/orchestration/routing_engine.py)
   - Score providers by capabilities, latency, health, GPU pressure, cost
   - Select best provider/model/worker
5. Provider Call (providers/)
   - Execute via selected provider adapter
6. Streaming/Response
   - Stream response back via protocol adapter
7. Metrics Recording (metrics/)
   - Record request, latency, tokens, errors
```

## Tool Execution Flow

```
1. Model generates tool call text
2. ToolCallNormalizer (runtime/tools/tool_normalizer.py)
   - Parse tool calls from multiple formats
3. ToolRuntime (runtime/tools/tool_runtime.py)
   - Coordinate execution
4. ToolRegistry (runtime/tools/tool_registry.py)
   - Look up registered tool handler
5. ToolPolicy (runtime/security/tool_policy.py)
   - Check permissions
6. ToolExecutor (runtime/tools/tool_executor.py)
   - Execute tool handler (builtin or MCP bridge)
7. ToolResult (runtime/tools/tool_result.py)
   - Package result
8. Inject result back into model context
9. Continue generation
```
