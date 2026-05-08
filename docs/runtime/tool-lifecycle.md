# Tool Runtime

The Tool Runtime provides a unified system for tool registration, execution, and result management.

## Components

- **tool_registry.py**: Register/lookup tools
- **tool_runtime.py**: Central coordinator
- **tool_executor.py**: Execute tool handlers (sync/async)
- **tool_result.py**: ToolCall + ToolResult data models
- **tool_normalizer.py**: Parse tool calls from model output

## Built-in Tools

Located in `runtime/tools/builtin/`:

| Tool | Description | Sandbox Support |
|---|---|---|
| `web_search` | Web search via Tavily/Serper/DuckDuckGo | No |
| `web_fetch` | URL content fetching | No |
| `shell` | Shell command execution | Yes |
| `filesystem` | File read/write operations | Yes |
| `python` | Python code execution | Yes |
| `http_request` | HTTP request tool | No |

## Tool Registration

```python
from runtime.tools.tool_registry import ToolRegistry, ToolDescriptor

registry = ToolRegistry()
registry.register(
    ToolDescriptor(
        name="my_tool",
        description="Description",
        input_schema={"type": "object", "properties": {}},
        handler=my_handler,
    )
)
```

## Tool Execution

```python
from runtime.tools.tool_runtime import tool_runtime
from runtime.tools.tool_result import ToolCall

call = ToolCall(id="call_1", name="web_search", arguments={"query": "hello"})
result = tool_runtime.execute(call)
```
