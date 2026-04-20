# Ajantis

> Named after [Ajantis Ilvastarr](https://baldursgate.fandom.com/wiki/Ajantis_Ilvastarr), the paladin companion from Baldur's Gate 1.
>
> ![Ajantis Ilvastarr](https://static.wikia.nocookie.net/baldursgategame/images/7/7c/Ajantis_Ilvastarr_AJANTIS_Portrait_BG1.png)

A desktop app for orchestrating multi-agent workflows with LM Studio, built with Tauri + Rust.

## What it does

Ajantis lets you build pipelines of LLM agents that talk to each other. Each agent runs on a model loaded in LM Studio. You wire them together with routing rules and a message sent to the first agent cascades through the graph automatically.

Key features:
- **Multi-agent routing** — define agents and connections; messages flow recursively through the graph based on priority and optional conditions
- **Manager agents** — special agents that get a tool-call loop and can spawn, kill, pause/resume, broadcast to, or pipe messages between other agents at runtime
- **Manager enforcement** — managers are constrained to orchestration and delegation; spawned workers can use tools and self-message for short reflection loops
- **Streaming** — responses stream token by token to the frontend; tool calls and results are surfaced as visible events
- **Persistent memory** — each agent's conversation history is kept in a shared memory pool and fed back as context on subsequent turns
- **Model management** — list, load, unload, and download models directly from the UI; supports `on_the_fly` mode (load before turn, unload after)
- **MCP server** — an embedded JSON-RPC server (port 4785) exposes file system tools (`read_file`, `write_file`, `edit_file`, `bash`, `glob_search`, `grep_search`) and all agent management tools to the LLM

## Architecture

```
Frontend (HTML/JS)
    │  Tauri IPC (invoke / Channel<StreamEvent>)
    ▼
Rust backend (src-tauri/src/lib.rs)
    ├── route_message     → recursive agent dispatch
    ├── send_message      → single LM Studio /responses call (with MCP integration)
    └── MCP server (axum, :4785)
            └── tools/call → file system + agent management handlers
```

The backend connects to LM Studio at `http://localhost:1234` by default. Override with the `LM_STUDIO_URL` env var.

## Configuration

Agents, routing rules, and the selected UI theme are stored in `~/.ajantis/ajantis-config.json` and edited from the app's **Agents** and **Settings** tabs.

Runtime config files under `~/.ajantis/` are also reachable through the MCP file tools.

Each agent has:
| Field | Description |
|---|---|
| `id` | Unique identifier used in routing rules |
| `model_key` | LM Studio model key |
| `role` | System prompt |
| `mode` | `stay_awake` (default) or `on_the_fly` (load/unload per turn) |
| `is_manager` | Enables the MCP tool-call loop |
| `armed` | Disabling stops the agent from receiving messages |

## Requirements

- [Rust](https://rustup.rs/) (stable)
- [Node.js](https://nodejs.org/) ≥ 18
- [LM Studio](https://lmstudio.ai/) running locally with at least one model loaded

## Getting started

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

The packaged app will be in `src-tauri/target/release/bundle/`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_URL` | `http://localhost:1234` | Base URL of your LM Studio instance |
| `LM_STUDIO_MODEL` | `lmstudio/lmstudio-1B` | Default model used for the single-agent chat tab |
| `LM_API_TOKEN` | — | Bearer token if your LM Studio instance requires auth |
