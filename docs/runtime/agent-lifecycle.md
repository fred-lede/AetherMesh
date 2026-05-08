# Agent Runtime Lifecycle

## Overview
The agent runtime manages multi-step AI agent execution with thinking, tool use, and observation loops.

## Components

### AgentContext
- Holds session state, memory, tool registry, and execution metadata
- Created per agent run, persists across steps

### AgentLoop
- Orchestrates the think → act → observe cycle
- Supports max_steps limit to prevent runaway execution
- Returns AgentResult with full step history

### AgentStep
- Single iteration: think (model call) → act (tool calls) → observe (tool results)
- Tracks duration, reasoning, errors per step

### AgentResult
- Final output with step-by-step trace
- Includes tool call/error counts, total duration

## Execution Flow
1. Create AgentContext with tools, system prompt
2. AgentLoop.run(context, task) starts the loop
3. Each step: model generates response → tools execute → results feed back
4. Loop terminates on max_steps, tool-free response, or error
