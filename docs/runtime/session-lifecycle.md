# Session Runtime Lifecycle

## Overview
Sessions provide persistent state across multiple requests, enabling multi-turn conversations, resumable tasks, and shared agent memory.

## Components

### SessionStore
- Backend-agnostic storage (Redis, memory, or file)
- Keyed by session_id with TTL-based expiration
- Supports atomic read/write/delete operations

### SessionManager
- Creates, retrieves, and closes sessions
- Integrates with agent runtime for memory persistence
- Tracks message count, tool calls, duration via ExtendedMetrics

## Features
- **Persistent**: Sessions survive server restarts (Redis backend)
- **Resumable**: Interrupted executions can continue from last state
- **Multi-client**: Multiple clients share the same session context
- **Agent memory**: Read/write persistent memory accessible to agents

## Storage Backends
| Backend | Persistence | Use Case |
|---------|-------------|----------|
| Memory  | Volatile    | Dev/test |
| Redis   | Durable     | Production |
