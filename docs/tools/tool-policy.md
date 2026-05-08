# Tool Policy

## Overview
The tool policy system controls which server-side tools (web search, web fetch) are available to models and under what conditions they execute.

## Policy Evaluation
`runtime/security/tool_policy.py` evaluates incoming requests to determine:
1. Which server tools are listed in the `tools` parameter
2. Whether a specific tool is forced via `tool_choice`
3. Whether to reject the request or handle it locally

## Modes
| Mode | Behavior |
|------|----------|
| `reject` | Refuse requests that list server tools |
| `local` | Handle server tools locally without passing to provider |

## Server Tool Names
- `web_search`: Web search via configured search providers
- `web_fetch`: URL content fetching

## Forced Tools
When `tool_choice: {type: "tool", name: "web_search"}` is set, the tool is executed directly on the server without any provider round-trip.

## Audit
All policy decisions are logged via the security audit system for traceability.
