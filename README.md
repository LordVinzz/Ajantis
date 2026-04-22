# Ajantis

> Named after [Ajantis Ilvastarr](https://baldursgate.fandom.com/wiki/Ajantis_Ilvastarr), the paladin companion from Baldur's Gate 1.
>
> ![Ajantis Ilvastarr](https://static.wikia.nocookie.net/baldursgategame/images/7/7c/Ajantis_Ilvastarr_AJANTIS_Portrait_BG1.png)

A Rust app for orchestrating multi-agent workflows with LM Studio.

## What it does

Ajantis lets you build pipelines of LLM agents that talk to each other. Each agent runs on a model loaded in LM Studio. You wire them together with routing rules and a message sent to the first agent cascades through the graph automatically.

Key features:
- **Multi-agent routing** — define agents and connections; messages flow recursively through the graph based on priority and optional conditions
- **Manager agents** — special agents that get a tool-call loop and can spawn, kill, pause/resume, broadcast to, or pipe messages between other agents at runtime
- **Manager enforcement** — managers are constrained to orchestration and delegation; spawned workers can use tools and self-message for short reflection loops
- **Streaming** — responses stream token by token to the active interface; tool calls and results are surfaced as visible events
- **Persistent memory** — each agent's conversation history is kept in a shared memory pool and fed back as context on subsequent turns
- **Model management** — list, load, unload, and download models from the runtime; supports `on_the_fly` mode (load before turn, unload after)
- **MCP server** — an embedded JSON-RPC server (port 4785) exposes file system tools (`read_file`, `write_file`, `edit_file`, `bash`, `glob_search`, `grep_search`) and all agent management tools to the LLM

## Architecture

```text
Rust runtime (src-tauri/src/lib.rs)
    ├── route_message     → recursive agent dispatch
    ├── send_message      → single LM Studio /responses call (with MCP integration)
    └── MCP server (axum, :4785)
            └── tools/call → file system + agent management handlers

TUI client (src-tauri/src/bin/tui.rs)
    └── app_lib::runtime  → direct runtime access without React/Vite
```

The backend connects to LM Studio at `http://localhost:1234` by default. Override with the `LM_STUDIO_URL` env var.

## Configuration

Agents, routing rules, and runtime settings are stored in `~/.ajantis/ajantis-config.json`.

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
- [LM Studio](https://lmstudio.ai/) running locally with at least one model loaded

## Getting started

```bash
cargo run --manifest-path src-tauri/Cargo.toml --bin ajantis-tui --features tui
```

## Build

```bash
cargo build --manifest-path src-tauri/Cargo.toml --release --bin ajantis-tui --features tui
```

If you still want the desktop shell, it now serves a static page from `src-static/` instead of a React app.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_URL` | `http://localhost:1234` | Base URL of your LM Studio instance |
| `LM_STUDIO_MODEL` | `lmstudio/lmstudio-1B` | Default model used when the runtime starts |
| `LM_API_TOKEN` | — | Bearer token if your LM Studio instance requires auth |
