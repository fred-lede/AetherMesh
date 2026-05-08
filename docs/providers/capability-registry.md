# Provider Capability Registry

The provider capability registry (`providers/registry.py`) enables capability-based
routing beyond simple model name matching.

## Capabilities

| Capability | Description |
|---|---|
| `chat` | Text generation/chat |
| `tools` | Function calling/tool use |
| `thinking` | Extended thinking/reasoning |
| `vision` | Image understanding |
| `audio` | Audio input processing |
| `embeddings` | Text embedding generation |
| `rerank` | Result reranking |
| `responses` | OpenAI Responses API |
| `mcp` | MCP protocol support |
| `web_search` | Built-in web search |
| `streaming` | Streaming response support |

## Provider Capability Map

```yaml
providers:
  ollama:
    capabilities: [chat, tools, vision, audio, embeddings]
  openai:
    capabilities: [chat, tools, responses, embeddings, web_search]
  anthropic:
    capabilities: [chat, tools, thinking, vision, audio, mcp]
  gemini:
    capabilities: [chat, tools, vision, audio]
  nvidia_nim:
    capabilities: [chat, tools, embeddings, rerank]
  ollama_cloud:
    capabilities: [chat, tools, embeddings, rerank]
```

## Multi-Dimension Routing

The routing engine scores providers based on:
1. **Capabilities match** (required ∩ supported)
2. **Latency** (rolling P50/P95)
3. **Health** (error rate, circuit breaker state)
4. **GPU pressure** (VRAM utilization, queue depth)
5. **Cost** (local = 0, cloud = weighted)
6. **Tool requirements** (builtin vs MCP bridge)
