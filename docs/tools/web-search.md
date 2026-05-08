# Web Search

## Overview
AetherMesh provides web search as a built-in capability through `runtime/tools/web_search/`. Multiple search providers are supported with automatic fallback.

## Providers

| Provider | Type | API Key Required |
|----------|------|-----------------|
| Tavily | Cloud API | `TAVILY_API_KEY` |
| Serper.dev | Cloud API | `SERPER_API_KEY` |
| DuckDuckGo | HTML scraper | None (fallback) |

## Architecture
```
runtime/tools/web_search/
  search_provider.py   # Abstract base class (SearchProvider)
  tavily.py            # Tavily API implementation
  serper.py            # Serper.dev API implementation
  duckduckgo.py        # DuckDuckGo HTML scraper (fallback)
```

## Execution Flow
1. `web_search` tool call received by Tool Runtime
2. Tool Runtime calls `web_search_manager.search()`
3. SearchManager tries providers in order: Tavily → Serper → DuckDuckGo
4. Returns unified `SearchResult` list with title, url, snippet

## Tools
- `web_search`: Search the web and return ranked results with URLs
- `web_fetch`: Fetch a URL and extract its readable content
