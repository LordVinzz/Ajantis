use axum::{
    extract::State as AxumState,
    http::{HeaderValue, Method},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

use crate::agent_config::{resolve_audit_behavior_config, Agent, AgentConfig, AgentLoadConfig};
use crate::chat::{call_chat_blocking, validate_audit_worker_response, StreamEvent};
use crate::config_persistence::ajantis_dir;
use crate::event_sink::SharedEventSink;
use crate::helpers::{
    apply_runtime_agent_context, compute_context_budget, extract_explicit_audit_refs,
    is_manager_blocked_tool, is_manager_only_tool, resolve_active_behaviors,
};
use crate::memory::{CommandExecution, CommandHistory, MemoryEntry, MemoryPool};
use crate::models::fetch_models;
use crate::runs::{
    primary_run_id, record_dossier_blocked_command, record_dossier_command, ActiveRuns,
};
use crate::state::{BehaviorTriggerCache, McpTool};

// ── Background task tracking ───────────────────────────────────────

pub(crate) struct AjantisTask {
    pub id: String,
    pub description: String,
    pub status: Arc<Mutex<String>>, // "running" | "completed" | "failed: …" | "stopped"
    pub output: Arc<Mutex<String>>,
    pub abort_handle: Option<tokio::task::AbortHandle>,
    pub started_at: String,
}

// ── MCP server state ───────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct McpState {
    pub(crate) tools: Vec<McpTool>,
    /// Static root used as fallback when no workspace is selected.
    #[allow(dead_code)]
    pub(crate) workspace_root: PathBuf,
    /// Dynamically updated to the selected workspace path.
    /// All file/command operations are sandboxed to this path.
    pub(crate) active_workspace: Arc<Mutex<PathBuf>>,
    pub(crate) todo_list: Arc<Mutex<Vec<Value>>>,
    pub(crate) memory_pool: Arc<Mutex<MemoryPool>>,
    /// Shared agent registry — manager tools read/write this at runtime.
    pub(crate) agent_config: Arc<Mutex<AgentConfig>>,
    /// Self-reference port so manager tools can call other MCP tools.
    pub(crate) mcp_port: u16,
    /// Shared with AppState — allows MCP tool handlers to emit stream events.
    pub(crate) event_channel: Arc<Mutex<Option<SharedEventSink>>>,
    /// Background tasks spawned via TaskCreate.
    pub(crate) tasks: Arc<Mutex<HashMap<String, AjantisTask>>>,
    /// Active thread-scoped command execution history.
    pub(crate) command_history: Arc<Mutex<CommandHistory>>,
    /// Per-agent cache of file slices already returned to the model.
    pub(crate) read_cache: Arc<Mutex<HashMap<String, HashSet<String>>>>,
    /// Per-agent cache of files discovered by glob_search.
    pub(crate) glob_cache: Arc<Mutex<HashMap<String, HashSet<String>>>>,
    /// Runtime embeddings cache for behavior-trigger evaluation.
    pub(crate) behavior_trigger_cache: Arc<Mutex<BehaviorTriggerCache>>,
    /// Active built-in behaviors for turns currently executing through the tool loop.
    pub(crate) active_behavior_contexts: Arc<Mutex<HashMap<String, HashSet<String>>>>,
    /// Active routed runs available to tool handlers for event emission and resumption.
    pub(crate) active_runs: ActiveRuns,
    /// Run-scoped key-value scratchpad — cleared at run start, never written to disk.
    pub(crate) scratchpad: Arc<Mutex<HashMap<String, String>>>,
}

#[derive(Deserialize)]
pub(crate) struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Serialize)]
pub(crate) struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

pub(crate) async fn handle_jsonrpc(
    AxumState(state): AxumState<McpState>,
    Json(req): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    if req.jsonrpc != "2.0" {
        return Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: None,
            error: Some(json!({"code": -32600, "message": "Invalid JSON-RPC version"})),
        });
    }
    if req.id.is_none() {
        return Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: None,
            result: Some(Value::Null),
            error: None,
        });
    }
    let id = req.id.clone();
    match req.method.as_str() {
        "initialize" => Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "ajantis-mcp", "version": "2.0.0" },
            })),
            error: None,
        }),
        "tools/list" => {
            let tools: Vec<Value> = state
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    })
                })
                .collect();
            Json(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(json!({ "tools": tools })),
                error: None,
            })
        }
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let name = params["name"].as_str().unwrap_or("").to_string();
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let caller_agent_id = params
                .get("caller_agent_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let result = handle_tool_call(&name, &args, &state, caller_agent_id.as_deref()).await;
            Json(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(result),
                error: None,
            })
        }
        "resources/list" => Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({ "resources": [] })),
            error: None,
        }),
        _ => Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(
                json!({"code": -32601, "message": format!("Unknown method: {}", req.method)}),
            ),
        }),
    }
}

pub(crate) async fn handle_tool_call(
    name: &str,
    args: &Value,
    state: &McpState,
    caller_agent_id: Option<&str>,
) -> Value {
    // Resolve the active sandbox root: selected workspace if set, fallback to workspace_root.
    let sandbox: PathBuf = state.active_workspace.lock().unwrap().clone();
    if let Err(reason) = enforce_tool_policy(name, caller_agent_id, state) {
        if let Some(run_id) = primary_run_id(&state.active_runs) {
            record_dossier_blocked_command(&state.active_runs, &run_id, &reason);
        }
        return mcp_error(&reason);
    }

    match name {
        // ── File system ───────────────────────────────────────────
        "bash" => {
            let command = args["command"].as_str().unwrap_or("").trim().to_string();
            if command.is_empty() {
                return mcp_error("Error: command is required.");
            }
            if let Err(reason) = check_command_sandbox(&command, &sandbox, state) {
                if let Some(run_id) = primary_run_id(&state.active_runs) {
                    record_dossier_blocked_command(&state.active_runs, &run_id, &reason);
                }
                return mcp_error(&reason);
            }
            let cwd = sandbox.to_string_lossy().to_string();
            let normalized = normalize_command(&command);
            if let Some(cached) = reuse_cached_command(state, "bash", &normalized, &cwd) {
                let reused = format!(
                    "[cached result reused | tool: bash | cwd: {} | command: {}]\n{}",
                    cwd, command, cached.result
                );
                return mcp_ok(&reused);
            }
            match Command::new("bash")
                .arg("-lc")
                .arg(&command)
                .current_dir(&sandbox)
                .output()
            {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    let text = if stderr.is_empty() {
                        stdout
                    } else {
                        format!("{}\n{}", stdout, stderr)
                    };
                    remember_command_result(
                        state,
                        "bash",
                        &command,
                        &normalized,
                        &cwd,
                        out.status.success(),
                        if text.is_empty() {
                            "[no output]"
                        } else {
                            &text
                        },
                    );
                    mcp_ok(if text.is_empty() {
                        "[no output]"
                    } else {
                        &text
                    })
                }
                Err(e) => mcp_error(&format!("Error: {}", e)),
            }
        }
        "read_file" => {
            let file_path = args["path"].as_str().unwrap_or("");
            if file_path.is_empty() {
                return mcp_error("Error: path is required.");
            }
            let abs = match resolve_allowed_path(file_path, &sandbox) {
                Ok(path) => path,
                Err(reason) => return mcp_error(&reason),
            };
            if let Err(reason) = enforce_glob_match(&abs, caller_agent_id, state) {
                return mcp_error(&reason);
            }
            match fs::read_to_string(&abs) {
                Ok(data) => {
                    let scope = args["scope"].as_str();
                    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
                    let requested_limit =
                        args["limit"].as_u64().map(|l| l as usize).unwrap_or(4_000);
                    let hard_limit = requested_limit.min(4_000);
                    let (slice, scope_label) =
                        match extract_file_scope(&data, scope, offset, hard_limit) {
                            Ok(result) => result,
                            Err(reason) => return mcp_error(&reason),
                        };
                    let cache_key = format!("{}::{}", abs.to_string_lossy(), scope_label);
                    if let Some(agent_id) = caller_agent_id {
                        let mut cache = state.read_cache.lock().unwrap();
                        let seen = cache.entry(agent_id.to_string()).or_default();
                        if !seen.insert(cache_key.clone()) {
                            return mcp_ok("Already in context: this exact file slice was returned earlier. Reuse it unless you need a different scope.");
                        }
                    }
                    let char_count = slice.chars().count();
                    let payload = format!(
                        "[path: {} | scope: {} | chars: {}]\n{}",
                        abs.to_string_lossy(),
                        scope_label,
                        char_count,
                        slice,
                    );
                    remember_command_result(
                        state,
                        "read_file",
                        &format!("read_file {}", abs.to_string_lossy()),
                        &format!("read_file {} {}", abs.to_string_lossy(), scope_label),
                        &sandbox.to_string_lossy(),
                        true,
                        &payload,
                    );
                    mcp_ok(&payload)
                }
                Err(e) => mcp_error(&format!("Error reading file: {}", e)),
            }
        }
        "write_file" => {
            let file_path = args["path"].as_str().unwrap_or("");
            let content = args["content"].as_str().unwrap_or("");
            if file_path.is_empty() {
                return mcp_error("Error: path is required.");
            }
            let abs = match resolve_allowed_path(file_path, &sandbox) {
                Ok(path) => path,
                Err(reason) => return mcp_error(&reason),
            };
            if let Some(parent) = abs.parent() {
                let _ = fs::create_dir_all(parent);
            }
            match fs::write(&abs, content) {
                Ok(_) => mcp_ok("File written successfully."),
                Err(e) => mcp_error(&format!("Error writing file: {}", e)),
            }
        }
        "edit_file" => {
            let file_path = args["path"].as_str().unwrap_or("");
            let old_str = args["old_string"].as_str().unwrap_or("");
            let new_str = args["new_string"].as_str().unwrap_or("");
            let replace_all = args["replace_all"].as_bool().unwrap_or(false);
            if file_path.is_empty() {
                return mcp_error("Error: path is required.");
            }
            let abs = match resolve_allowed_path(file_path, &sandbox) {
                Ok(path) => path,
                Err(reason) => return mcp_error(&reason),
            };
            match fs::read_to_string(&abs) {
                Ok(data) => {
                    let result = if replace_all {
                        data.replace(old_str, new_str)
                    } else {
                        data.replacen(old_str, new_str, 1)
                    };
                    match fs::write(&abs, result) {
                        Ok(_) => mcp_ok("File edited successfully."),
                        Err(e) => mcp_error(&format!("Error editing file: {}", e)),
                    }
                }
                Err(e) => mcp_error(&format!("Error reading file: {}", e)),
            }
        }
        "glob_search" => {
            let pattern = args["pattern"].as_str().unwrap_or("");
            if pattern.is_empty() {
                return mcp_error("Error: pattern is required.");
            }
            let base = match args["path"].as_str() {
                Some(path) => match resolve_allowed_path(path, &sandbox) {
                    Ok(base) => base,
                    Err(reason) => return mcp_error(&reason),
                },
                None => sandbox.clone(),
            };
            let full_pattern = base.join(pattern).to_string_lossy().to_string();
            match glob::glob(&full_pattern) {
                Ok(paths) => {
                    let results: Vec<String> = paths
                        .filter_map(|r| r.ok())
                        .map(|p| normalize_path(&p))
                        .filter(|p| is_allowed_path(p, &sandbox))
                        .map(|p| p.to_string_lossy().to_string())
                        .collect();
                    if let Some(agent_id) = caller_agent_id {
                        let mut cache = state.glob_cache.lock().unwrap();
                        let seen = cache.entry(agent_id.to_string()).or_default();
                        seen.clear();
                        seen.extend(results.iter().cloned());
                    }
                    if results.is_empty() {
                        remember_command_result(
                            state,
                            "glob_search",
                            &format!("glob_search {} {}", base.to_string_lossy(), pattern),
                            &format!("glob_search {} {}", base.to_string_lossy(), pattern),
                            &sandbox.to_string_lossy(),
                            true,
                            "No files matched.",
                        );
                        mcp_ok("No files matched.")
                    } else {
                        let truncated: Vec<String> = results.iter().take(200).cloned().collect();
                        let body = if results.len() > 200 {
                            format!(
                                "{}\n…[truncated, {} matches total]",
                                truncated.join("\n"),
                                results.len()
                            )
                        } else {
                            truncated.join("\n")
                        };
                        remember_command_result(
                            state,
                            "glob_search",
                            &format!("glob_search {} {}", base.to_string_lossy(), pattern),
                            &format!("glob_search {} {}", base.to_string_lossy(), pattern),
                            &sandbox.to_string_lossy(),
                            true,
                            &body,
                        );
                        mcp_ok(&body)
                    }
                }
                Err(e) => mcp_error(&format!("Invalid glob pattern: {}", e)),
            }
        }
        "grep_search" => {
            let pattern = args["pattern"].as_str().unwrap_or("");
            if pattern.is_empty() {
                return mcp_error("Error: pattern is required.");
            }
            let search_path = match args["path"].as_str() {
                Some(path) => match resolve_allowed_path(path, &sandbox) {
                    Ok(path) => path,
                    Err(reason) => return mcp_error(&reason),
                },
                None => sandbox.clone(),
            };
            let glob_pat = args["glob"].as_str();
            if search_path.is_dir() && glob_pat.is_none() {
                return mcp_error("Error: grep_search on a directory requires a glob filter. Use glob_search first, then grep within that scope.");
            }
            if search_path.is_file() {
                if let Err(reason) = enforce_glob_match(&search_path, caller_agent_id, state) {
                    return mcp_error(&reason);
                }
            }
            let case_flag = args["-i"].as_bool().unwrap_or(false);
            let mut cmd = Command::new("rg");
            cmd.arg("--no-heading").arg("-n");
            if case_flag {
                cmd.arg("-i");
            }
            if let Some(glob_pat) = glob_pat {
                cmd.arg("--glob").arg(glob_pat);
            }
            let context = args["context"].as_u64();
            if let Some(c) = args["-C"].as_u64().or(context) {
                cmd.arg("-C").arg(c.to_string());
            } else {
                if let Some(before) = args["-B"].as_u64() {
                    cmd.arg("-B").arg(before.to_string());
                }
                if let Some(after) = args["-A"].as_u64() {
                    cmd.arg("-A").arg(after.to_string());
                }
            }
            if args["multiline"].as_bool().unwrap_or(false) {
                cmd.arg("--multiline");
            }
            if let Some(file_type) = args["type"].as_str() {
                if !file_type.is_empty() {
                    cmd.arg("--type").arg(file_type);
                }
            }
            cmd.arg(pattern).arg(&search_path).current_dir(&sandbox);
            match cmd.output().or_else(|_| {
                let mut fallback = Command::new("grep");
                fallback.arg("-r").arg("-n");
                if case_flag {
                    fallback.arg("-i");
                }
                fallback
                    .arg(pattern)
                    .arg(&search_path)
                    .current_dir(&sandbox);
                fallback.output()
            }) {
                Ok(out) => {
                    let text = String::from_utf8_lossy(&out.stdout).to_string();
                    if text.trim().is_empty() {
                        remember_command_result(
                            state,
                            "grep_search",
                            &format!("grep_search {} {}", search_path.to_string_lossy(), pattern),
                            &format!("grep_search {} {}", search_path.to_string_lossy(), pattern),
                            &sandbox.to_string_lossy(),
                            out.status.success(),
                            "No matches found.",
                        );
                        mcp_ok("No matches found.")
                    } else {
                        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
                        let head_limit = args["head_limit"].as_u64().unwrap_or(50) as usize;
                        let lines: Vec<&str> = text.lines().skip(offset).take(head_limit).collect();
                        let truncated = text.lines().count() > offset + head_limit;
                        let body = if truncated {
                            format!("{}\n…[truncated]", lines.join("\n"))
                        } else {
                            lines.join("\n")
                        };
                        remember_command_result(
                            state,
                            "grep_search",
                            &format!("grep_search {} {}", search_path.to_string_lossy(), pattern),
                            &format!("grep_search {} {}", search_path.to_string_lossy(), pattern),
                            &sandbox.to_string_lossy(),
                            out.status.success(),
                            &body,
                        );
                        mcp_ok(&body)
                    }
                }
                Err(e) => mcp_error(&format!("Search failed: {}", e)),
            }
        }
        // ── Todo + memory ─────────────────────────────────────────
        "TodoWrite" => {
            if let Value::Array(items) = args["todos"].clone() {
                let mut list = state.todo_list.lock().unwrap();
                list.extend(items);
                let json = serde_json::to_string_pretty(&*list).unwrap_or_default();
                mcp_ok(&format!("Todo list updated. Current items:\n{}", json))
            } else {
                mcp_ok("Todo list unchanged.")
            }
        }
        "memory_pool" => {
            let action = args["action"].as_str().unwrap_or("list");
            let pool = state.memory_pool.lock().unwrap();
            match action {
                "search" => {
                    let query = args["query"].as_str().unwrap_or("");
                    if query.is_empty() {
                        return mcp_error("Error: query is required for search.");
                    }
                    let results: Vec<&MemoryEntry> = pool.search(query);
                    let j = serde_json::to_string_pretty(&results).unwrap_or_default();
                    mcp_ok(&format!("Found {} entries:\n{}", results.len(), j))
                }
                "list" => {
                    let limit = args["limit"].as_u64().unwrap_or(50) as usize;
                    let entries: Vec<&MemoryEntry> =
                        pool.entries.iter().rev().take(limit).collect();
                    let j = serde_json::to_string_pretty(&entries).unwrap_or_default();
                    mcp_ok(&format!(
                        "{} entries (showing last {}):\n{}",
                        pool.entries.len(),
                        entries.len(),
                        j
                    ))
                }
                "count" => mcp_ok(&format!(
                    "Memory pool contains {} entries.",
                    pool.entries.len()
                )),
                _ => mcp_error(&format!(
                    "Unknown action '{}'. Use: list, search, count.",
                    action
                )),
            }
        }
        // ── Run-scoped scratchpad ─────────────────────────────────
        "scratchpad_write" => {
            let key = args["key"].as_str().unwrap_or("").trim().to_string();
            let content = args["content"].as_str().unwrap_or("").to_string();
            if key.is_empty() {
                return mcp_error("Error: key is required.");
            }
            let mut pad = state.scratchpad.lock().unwrap();
            let overwrite = pad.contains_key(&key);
            pad.insert(key.clone(), content.clone());
            let char_count = content.chars().count();
            if overwrite {
                mcp_ok(&format!("Scratchpad key '{}' updated ({} chars).", key, char_count))
            } else {
                mcp_ok(&format!("Scratchpad key '{}' written ({} chars).", key, char_count))
            }
        }
        "scratchpad_read" => {
            let key = args["key"].as_str().unwrap_or("").trim();
            if key.is_empty() {
                return mcp_error("Error: key is required.");
            }
            match state.scratchpad.lock().unwrap().get(key) {
                Some(content) => mcp_ok(content),
                None => mcp_error(&format!("Scratchpad key '{}' not found.", key)),
            }
        }
        "scratchpad_list" => {
            let pad = state.scratchpad.lock().unwrap();
            if pad.is_empty() {
                return mcp_ok("Scratchpad is empty.");
            }
            let lines: Vec<String> = pad
                .iter()
                .map(|(k, v)| format!("  {} ({} chars)", k, v.chars().count()))
                .collect();
            mcp_ok(&format!("Scratchpad keys:\n{}", lines.join("\n")))
        }
        "scratchpad_delete" => {
            let key = args["key"].as_str().unwrap_or("").trim();
            if key.is_empty() {
                return mcp_error("Error: key is required.");
            }
            let removed = state.scratchpad.lock().unwrap().remove(key).is_some();
            if removed {
                mcp_ok(&format!("Scratchpad key '{}' deleted.", key))
            } else {
                mcp_error(&format!("Scratchpad key '{}' not found.", key))
            }
        }
        // ── Manager tools ─────────────────────────────────────────
        "spawn_agent" => {
            let role = args["role"].as_str().unwrap_or("worker");
            let sys_prompt = args["system_prompt"].as_str().unwrap_or("");
            let requested_model = args["model"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let ctx_limit = args["context_limit"].as_u64();
            let initial_message = args["initial_message"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let await_reply = args["await_reply"].as_bool().unwrap_or(true);
            let allowed_tools = args["tools"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            });

            let agent_id = match spawn_agent_record(
                state,
                caller_agent_id,
                role,
                sys_prompt,
                requested_model,
                ctx_limit,
                allowed_tools,
            )
            .await
            {
                Ok(agent_id) => agent_id,
                Err(err) => return mcp_error(&err),
            };

            if let Some(message) = initial_message {
                match run_agent_turn(
                    state,
                    &agent_id,
                    &message,
                    await_reply,
                    caller_active_behaviors(caller_agent_id, state),
                )
                .await
                {
                    Ok(payload) => mcp_ok(&payload),
                    Err(err) => mcp_error(&err),
                }
            } else {
                let payload = json!({
                    "agent_id": agent_id,
                    "status": "spawned",
                    "next_step": "Call send_message with this exact agent_id to start the worker."
                });
                mcp_ok(&serde_json::to_string(&payload).unwrap_or_default())
            }
        }
        "send_message" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            let content = args["content"].as_str().unwrap_or("");
            if agent_id.is_empty() {
                return mcp_error("Error: agent_id is required.");
            }
            if content.is_empty() {
                return mcp_error("Error: content is required.");
            }
            let await_reply = args["await_reply"].as_bool().unwrap_or(true);
            if let Err(reason) = enforce_send_message_target(caller_agent_id, agent_id, state) {
                return mcp_error(&reason);
            }

            match run_agent_turn(
                state,
                agent_id,
                content,
                await_reply,
                caller_active_behaviors(caller_agent_id, state),
            )
            .await
            {
                Ok(payload) => mcp_ok(&payload),
                Err(err) => mcp_error(&err),
            }
        }
        "send_parallel" => {
            let (enabled, max_agents) = {
                let config = state.agent_config.lock().unwrap();
                (
                    config.parallel_inference.enabled,
                    config.parallel_inference.max_parallel_agents,
                )
            };
            if !enabled {
                return mcp_error(
                    "Error: parallel inference is disabled. Enable it in Global Policy > Parallel inference.",
                );
            }
            let Some(agents_arr) = args["agents"].as_array().cloned() else {
                return mcp_error("Error: 'agents' array is required.");
            };
            if agents_arr.is_empty() {
                return mcp_error("Error: 'agents' array must not be empty.");
            }
            if agents_arr.len() as u32 > max_agents {
                return mcp_error(&format!(
                    "Error: {} agents requested but max_parallel_agents limit is {}. Adjust in Global Policy > Parallel inference.",
                    agents_arr.len(),
                    max_agents
                ));
            }

            // Spawn all agent records sequentially (lightweight, no inference).
            let mut agent_tasks: Vec<(String, String)> = Vec::new();
            for entry in &agents_arr {
                let role = entry["role"].as_str().unwrap_or("worker");
                let sys_prompt = entry["system_prompt"].as_str().unwrap_or("");
                let model = entry["model"]
                    .as_str()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let message = entry["initial_message"].as_str().unwrap_or("").to_string();
                match spawn_agent_record(
                    state,
                    caller_agent_id,
                    role,
                    sys_prompt,
                    model,
                    None,
                    None,
                )
                .await
                {
                    Ok(agent_id) => agent_tasks.push((agent_id, message)),
                    Err(err) => return mcp_error(&err),
                }
            }

            // Credit spawned agents to the run budget.
            if let Some(run_id) = primary_run_id(&state.active_runs) {
                let mut runs = state.active_runs.lock().unwrap();
                if let Some(run) = runs.get_mut(&run_id) {
                    run.usage.spawned_agents =
                        run.usage.spawned_agents.saturating_add(agent_tasks.len() as u32);
                }
            }

            // Run all agents concurrently, waiting for every one to finish.
            let inherited_behaviors = caller_active_behaviors(caller_agent_id, state);
            let mut join_handles = Vec::new();
            for (agent_id, message) in agent_tasks {
                let state_clone = state.clone();
                let behaviors_clone = inherited_behaviors.clone();
                let handle: tokio::task::JoinHandle<(String, Result<String, String>)> =
                    tokio::spawn(async move {
                        let result = run_agent_turn(
                            &state_clone,
                            &agent_id,
                            &message,
                            true,
                            behaviors_clone,
                        )
                        .await;
                        (agent_id, result)
                    });
                join_handles.push(handle);
            }

            let mut results = Vec::new();
            for handle in join_handles {
                match handle.await {
                    Ok((agent_id, Ok(reply))) => {
                        results.push(json!({"agent_id": agent_id, "status": "ok", "reply": reply}))
                    }
                    Ok((agent_id, Err(err))) => {
                        results.push(json!({"agent_id": agent_id, "status": "error", "error": err}))
                    }
                    Err(err) => {
                        results.push(json!({"status": "error", "error": format!("Task panicked: {}", err)}))
                    }
                }
            }

            let payload =
                serde_json::to_string(&json!({"results": results})).unwrap_or_default();
            mcp_ok(&payload)
        }
        "read_agent_messages" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() {
                return mcp_error("Error: agent_id is required.");
            }
            if let Err(reason) = enforce_agent_introspection(caller_agent_id, agent_id, state) {
                return mcp_error(&reason);
            }

            let roles_filter: Option<Vec<String>> = args["roles"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });
            let limit = args["limit"].as_u64().map(|l| l as usize);
            let offset = args["offset"].as_u64().unwrap_or(0) as usize;

            let pool = state.memory_pool.lock().unwrap();
            let filtered: Vec<&MemoryEntry> = pool
                .entries
                .iter()
                .filter(|e| e.agent_id == agent_id)
                .filter(|e| {
                    roles_filter
                        .as_ref()
                        .map(|rf| rf.iter().any(|r| r == &e.role))
                        .unwrap_or(true)
                })
                .collect();
            let history_json: Vec<Value> = filtered
                .iter()
                .map(|e| json!({"role": e.role, "content": e.content}))
                .collect();

            let total = filtered.len();
            let page: Vec<&MemoryEntry> = filtered
                .into_iter()
                .skip(offset)
                .take(limit.unwrap_or(usize::MAX))
                .collect();

            let context_limit = agent_context_limit(agent_id, state);
            let budget = compute_context_budget("", &history_json, "", context_limit, 0);
            let payload = json!({
                "agent_id": agent_id,
                "total_messages": total,
                "returned_messages": page.len(),
                "context_limit": budget.limit,
                "estimated_used_tokens": budget.estimated_used,
                "estimated_remaining_tokens": budget.remaining,
                "messages": page,
            });
            mcp_ok(&serde_json::to_string_pretty(&payload).unwrap_or_default())
        }
        "list_agents" => {
            let status_filter: Option<Vec<String>> = args["status_filter"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });
            let role_filter = args["role_filter"].as_str();

            let config = state.agent_config.lock().unwrap();
            let pool = state.memory_pool.lock().unwrap();

            let agents: Vec<Value> = config
                .agents
                .iter()
                .filter(|a| a.agent_type != "user")
                .filter(|a| !a.is_manager) // exclude self — manager must not call itself via send_message
                .filter(|a| {
                    role_filter
                        .map(|r| a.name.contains(r) || a.role.as_deref().unwrap_or("").contains(r))
                        .unwrap_or(true)
                })
                .filter_map(|a| {
                    let status = if !a.armed {
                        "done"
                    } else if a.paused {
                        "paused"
                    } else {
                        "idle"
                    };
                    if status_filter
                        .as_ref()
                        .map(|sf| sf.iter().any(|s| s == status))
                        .unwrap_or(false)
                        && status_filter.is_some()
                        && !status_filter.as_ref().unwrap().iter().any(|s| s == status)
                    {
                        return None;
                    }
                    let msg_count = pool.entries.iter().filter(|e| e.agent_id == a.id).count();
                    let last_preview = pool
                        .entries
                        .iter()
                        .rev()
                        .find(|e| e.agent_id == a.id && e.role == "assistant")
                        .map(|e| e.content.chars().take(120).collect::<String>());
                    let history_json: Vec<Value> = pool
                        .entries
                        .iter()
                        .filter(|e| {
                            e.agent_id == a.id && (e.role == "user" || e.role == "assistant")
                        })
                        .map(|e| json!({"role": e.role, "content": e.content}))
                        .collect();
                    let budget = compute_context_budget(
                        "",
                        &history_json,
                        "",
                        a.load_config.as_ref().and_then(|cfg| cfg.context_length),
                        0,
                    );
                    let tool_enabled = a
                        .allowed_tools
                        .as_ref()
                        .map(|tools| !tools.is_empty())
                        .unwrap_or(true);
                    Some(json!({
                        "agent_id": a.id,
                        "name": a.name,
                        "model": a.model_key,
                        "status": status,
                        "is_manager": a.is_manager,
                        "tool_enabled": tool_enabled,
                        "message_count": msg_count,
                        "last_output_preview": last_preview,
                        "context_limit": budget.limit,
                        "estimated_used_tokens": budget.estimated_used,
                        "estimated_remaining_tokens": budget.remaining,
                    }))
                })
                .collect();

            mcp_ok(&format!(
                "{} agents:\n{}",
                agents.len(),
                serde_json::to_string_pretty(&agents).unwrap_or_default()
            ))
        }
        "get_agent_state" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() {
                return mcp_error("Error: agent_id is required.");
            }
            if let Err(reason) = enforce_agent_introspection(caller_agent_id, agent_id, state) {
                return mcp_error(&reason);
            }

            let config = state.agent_config.lock().unwrap();
            match config.agents.iter().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => {
                    let pool = state.memory_pool.lock().unwrap();
                    let messages: Vec<&MemoryEntry> = pool
                        .entries
                        .iter()
                        .filter(|e| e.agent_id == agent_id)
                        .collect();
                    let last_output = messages
                        .iter()
                        .rev()
                        .find(|e| e.role == "assistant")
                        .map(|e| e.content.clone());
                    let status = if !a.armed {
                        "done"
                    } else if a.paused {
                        "paused"
                    } else {
                        "idle"
                    };
                    let history_json: Vec<Value> = messages
                        .iter()
                        .filter(|e| e.role == "user" || e.role == "assistant")
                        .map(|e| json!({"role": e.role, "content": e.content}))
                        .collect();
                    let budget = compute_context_budget(
                        "",
                        &history_json,
                        "",
                        a.load_config.as_ref().and_then(|cfg| cfg.context_length),
                        0,
                    );
                    let result = json!({
                        "agent_id": a.id, "name": a.name, "model": a.model_key,
                        "status": status, "is_manager": a.is_manager,
                        "paused": a.paused, "armed": a.armed,
                        "message_count": messages.len(), "last_output": last_output,
                        "context_limit": budget.limit,
                        "estimated_used_tokens": budget.estimated_used,
                        "estimated_remaining_tokens": budget.remaining,
                    });
                    mcp_ok(&serde_json::to_string_pretty(&result).unwrap_or_default())
                }
            }
        }
        "kill_agent" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() {
                return mcp_error("Error: agent_id is required.");
            }
            let reason = args["reason"].as_str().unwrap_or("terminated by manager");
            let mut config = state.agent_config.lock().unwrap();
            match config.agents.iter_mut().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => {
                    a.armed = false;
                    mcp_ok(&format!(
                        "Agent '{}' terminated. Reason: {}",
                        agent_id, reason
                    ))
                }
            }
        }
        "pause_agent" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() {
                return mcp_error("Error: agent_id is required.");
            }
            let mut config = state.agent_config.lock().unwrap();
            match config.agents.iter_mut().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => {
                    a.paused = true;
                    mcp_ok(&format!("Agent '{}' paused.", agent_id))
                }
            }
        }
        "resume_agent" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() {
                return mcp_error("Error: agent_id is required.");
            }
            let mut config = state.agent_config.lock().unwrap();
            match config.agents.iter_mut().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => {
                    a.paused = false;
                    mcp_ok(&format!("Agent '{}' resumed.", agent_id))
                }
            }
        }
        "broadcast_message" => {
            let content = args["content"].as_str().unwrap_or("");
            if content.is_empty() {
                return mcp_error("Error: content is required.");
            }
            let await_reply = args["await_reply"].as_bool().unwrap_or(true);

            let raw_ids: Vec<String> = match args["agent_ids"].as_array() {
                None => return mcp_error("Error: agent_ids is required."),
                Some(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
            };

            // Resolve targets (extract ids while holding lock, then release)
            let targets: Vec<String> = {
                let config = state.agent_config.lock().unwrap();
                let ids: Vec<String> = if raw_ids.len() == 1 && raw_ids[0] == "*" {
                    config
                        .agents
                        .iter()
                        .filter(|a| a.agent_type != "user" && !a.is_manager && a.armed && !a.paused)
                        .map(|a| a.id.clone())
                        .collect()
                } else {
                    raw_ids
                };
                ids.into_iter()
                    .filter_map(|id| {
                        config
                            .agents
                            .iter()
                            .find(|a| a.id == id && a.armed && !a.paused && a.model_key.is_some())
                            .map(|_| id)
                    })
                    .collect()
            };

            if !await_reply {
                let mut pool = state.memory_pool.lock().unwrap();
                for id in &targets {
                    pool.push(id, id, "user", content);
                }
                let ids: Vec<&str> = targets.iter().map(|id| id.as_str()).collect();
                return mcp_ok(&format!(
                    r#"{{"queued":{},"agent_ids":{}}}"#,
                    targets.len(),
                    serde_json::to_string(&ids).unwrap_or_default()
                ));
            }

            let mut results = serde_json::Map::new();
            for agent_id in &targets {
                match run_agent_turn(
                    state,
                    agent_id,
                    content,
                    true,
                    caller_active_behaviors(caller_agent_id, state),
                )
                .await
                {
                    Ok(payload) => {
                        let parsed = serde_json::from_str::<Value>(&payload)
                            .unwrap_or_else(|_| json!({"status":"ok","response": payload}));
                        results.insert(agent_id.clone(), parsed);
                    }
                    Err(e) => {
                        results.insert(agent_id.clone(), json!({"status":"error","error":e}));
                    }
                }
            }
            mcp_ok(&serde_json::to_string_pretty(&results).unwrap_or_default())
        }
        "fork_agent" => {
            let source_id = args["source_agent_id"].as_str().unwrap_or("");
            if source_id.is_empty() {
                return mcp_error("Error: source_agent_id is required.");
            }
            let new_role = args["role"].as_str();
            let sp_override = args["system_prompt_override"].as_str();
            let truncate_at = args["truncate_at"].as_u64().map(|n| n as usize);
            let ts = chrono::Utc::now().timestamp_millis();
            let new_id = format!("{}-fork-{}", source_id, ts);

            let (new_agent, entries_to_copy) = {
                let config = state.agent_config.lock().unwrap();
                let source = match config.agents.iter().find(|a| a.id == source_id) {
                    None => return mcp_error(&format!("Error: agent '{}' not found.", source_id)),
                    Some(a) => a,
                };
                let mut cloned = source.clone();
                cloned.id = new_id.clone();
                cloned.name = new_role
                    .unwrap_or(&format!("{}-fork", source.name))
                    .to_string();
                cloned.paused = false;
                cloned.armed = true;
                if let Some(sp) = sp_override {
                    cloned.role = Some(sp.to_string());
                }

                let pool = state.memory_pool.lock().unwrap();
                let src_entries: Vec<&MemoryEntry> = pool
                    .entries
                    .iter()
                    .filter(|e| e.agent_id == source_id)
                    .collect();
                let limit = truncate_at.unwrap_or(src_entries.len());
                let copies: Vec<MemoryEntry> = src_entries
                    .into_iter()
                    .take(limit)
                    .map(|e| {
                        let mut c = e.clone();
                        c.agent_id = new_id.clone();
                        c
                    })
                    .collect();
                (cloned, copies)
            };

            state.agent_config.lock().unwrap().agents.push(new_agent);
            state
                .memory_pool
                .lock()
                .unwrap()
                .entries
                .extend(entries_to_copy);
            mcp_ok(&format!(
                r#"{{"agent_id":"{}","forked_from":"{}"}}"#,
                new_id, source_id
            ))
        }
        "aggregate_results" => {
            let raw_ids: Vec<String> = match args["agent_ids"].as_array() {
                None => return mcp_error("Error: agent_ids is required."),
                Some(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
            };
            let fmt = args["format"].as_str().unwrap_or("structured");
            let synthesis_prompt = args["synthesis_prompt"].as_str();
            let synthesis_model = args["synthesis_model"].as_str();

            let target_ids: Vec<String> = if raw_ids.len() == 1 && raw_ids[0] == "*" {
                state
                    .agent_config
                    .lock()
                    .unwrap()
                    .agents
                    .iter()
                    .filter(|a| a.agent_type != "user")
                    .map(|a| a.id.clone())
                    .collect()
            } else {
                raw_ids
            };

            let mut collected: serde_json::Map<String, Value> = serde_json::Map::new();
            {
                let pool = state.memory_pool.lock().unwrap();
                for id in &target_ids {
                    let last = pool
                        .entries
                        .iter()
                        .rev()
                        .find(|e| e.agent_id == *id && e.role == "assistant")
                        .map(|e| e.content.clone())
                        .unwrap_or_else(|| "[no output]".to_string());
                    collected.insert(id.clone(), Value::String(last));
                }
            }

            if let Some(sp) = synthesis_prompt {
                let combined = collected
                    .iter()
                    .map(|(id, v)| format!("Agent {}:\n{}", id, v.as_str().unwrap_or("")))
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n");
                let full_prompt = format!("{}\n\nAgent outputs:\n{}", sp, combined);
                let model_to_use = synthesis_model.map(|s| s.to_string()).unwrap_or_else(|| {
                    state
                        .agent_config
                        .lock()
                        .unwrap()
                        .agents
                        .iter()
                        .find(|a| a.agent_type != "user" && a.model_key.is_some())
                        .and_then(|a| a.model_key.clone())
                        .unwrap_or_default()
                });
                if !model_to_use.is_empty() {
                    return match call_chat_blocking(&model_to_use, "", &full_prompt, &[], None)
                        .await
                    {
                        Ok(synthesis) => mcp_ok(&synthesis),
                        Err(e) => mcp_error(&format!("Synthesis failed: {}", e)),
                    };
                }
            }

            match fmt {
                "raw" => mcp_ok(
                    &collected
                        .values()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n---\n\n"),
                ),
                "summary" => mcp_ok(
                    &collected
                        .iter()
                        .map(|(id, v)| format!("**{}**: {}", id, v.as_str().unwrap_or("")))
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                ),
                _ => mcp_ok(&serde_json::to_string_pretty(&collected).unwrap_or_default()),
            }
        }
        "pipe_message" => {
            let to_id = args["to_agent_id"].as_str().unwrap_or("");
            let raw = args["content"].as_str().unwrap_or("");
            if to_id.is_empty() {
                return mcp_error("Error: to_agent_id is required.");
            }
            if raw.is_empty() {
                return mcp_error("Error: content is required.");
            }

            // Build the final message with optional prefix / suffix
            let mut content = String::new();
            if let Some(pre) = args["prefix"].as_str() {
                if !pre.is_empty() {
                    content.push_str(pre);
                    content.push('\n');
                }
            }
            content.push_str(raw);
            if let Some(suf) = args["suffix"].as_str() {
                if !suf.is_empty() {
                    content.push('\n');
                    content.push_str(suf);
                }
            }

            // Resolve target agent
            let (model_key, sys_prompt, agent_name, context_limit) = {
                let config = state.agent_config.lock().unwrap();
                match config.agents.iter().find(|a| a.id == to_id) {
                    None => return mcp_error(&format!("Error: agent '{}' not found.", to_id)),
                    Some(a) => {
                        if !a.armed {
                            return mcp_error(&format!("Error: agent '{}' is disarmed.", to_id));
                        }
                        if a.paused {
                            return mcp_error(&format!("Error: agent '{}' is paused.", to_id));
                        }
                        (
                            a.model_key.clone().unwrap_or_default(),
                            a.role.clone().unwrap_or_default(),
                            a.name.clone(),
                            a.load_config.as_ref().and_then(|cfg| cfg.context_length),
                        )
                    }
                }
            };
            if model_key.is_empty() {
                return mcp_error(&format!(
                    "Error: agent '{}' has no model configured.",
                    to_id
                ));
            }

            // Per-agent conversation history
            let history: Vec<Value> = {
                let pool = state.memory_pool.lock().unwrap();
                pool.entries
                    .iter()
                    .filter(|e| e.agent_id == to_id && (e.role == "user" || e.role == "assistant"))
                    .map(|e| json!({"role": e.role, "content": e.content}))
                    .collect()
            };

            let maybe_ch = state.event_channel.lock().unwrap().clone();
            let run_id =
                primary_run_id(&state.active_runs).unwrap_or_else(|| "detached-run".to_string());
            if let Some(ref ch) = maybe_ch {
                let budget =
                    compute_context_budget(&sys_prompt, &history, &content, context_limit, 0);
                let _ = ch.send(StreamEvent::AgentStart {
                    run_id: run_id.clone(),
                    agent_id: to_id.to_string(),
                    agent_name: agent_name.clone(),
                    model_key: model_key.clone(),
                    mode: "stay_awake".to_string(),
                    is_manager: false,
                    context_limit: budget.limit,
                    estimated_input_tokens: budget.estimated_used,
                    estimated_remaining_tokens: budget.remaining,
                });
                let _ = ch.send(StreamEvent::AgentStatus {
                    run_id: run_id.clone(),
                    agent_id: to_id.to_string(),
                    stage: "thinking".to_string(),
                    detail: "Generating response".to_string(),
                });
            }

            match call_chat_blocking(&model_key, &sys_prompt, &content, &history, context_limit)
                .await
            {
                Ok(response) => {
                    {
                        let mut pool = state.memory_pool.lock().unwrap();
                        pool.push(to_id, &agent_name, "user", &content);
                        pool.push(to_id, &agent_name, "assistant", &response);
                    }
                    if let Some(ref ch) = maybe_ch {
                        let _ = ch.send(StreamEvent::Token {
                            run_id: run_id.clone(),
                            agent_id: to_id.to_string(),
                            content: response.clone(),
                        });
                        let _ = ch.send(StreamEvent::AgentEnd {
                            run_id: run_id.clone(),
                            agent_id: to_id.to_string(),
                        });
                    }
                    let payload = json!({ "to_agent_id": to_id, "response": response });
                    mcp_ok(&serde_json::to_string(&payload).unwrap_or_default())
                }
                Err(e) => {
                    if let Some(ref ch) = maybe_ch {
                        let _ = ch.send(StreamEvent::Error {
                            run_id: run_id.clone(),
                            agent_id: to_id.to_string(),
                            agent_name: agent_name.clone(),
                            message: e.clone(),
                        });
                    }
                    mcp_error(&format!("Error calling agent: {}", e))
                }
            }
        }
        // ── Web ───────────────────────────────────────────────────
        "WebFetch" => {
            let url = args["url"].as_str().unwrap_or("");
            if url.is_empty() {
                return mcp_error("Error: url is required.");
            }
            let client = reqwest::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; Ajantis/1.0)")
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default();
            match client.get(url).send().await {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        return mcp_error(&format!("HTTP {}", resp.status()));
                    }
                    let body = resp.text().await.unwrap_or_default();
                    let text = strip_html(&body);
                    let char_count = text.chars().count();
                    let out = if char_count > 8000 {
                        let head: String = text.chars().take(8000).collect();
                        format!("{}…[truncated, {} chars total]", head, char_count)
                    } else {
                        text
                    };
                    mcp_ok(&format!("URL: {}\n\n{}", url, out))
                }
                Err(e) => mcp_error(&format!("Fetch failed: {}", e)),
            }
        }
        "WebSearch" => {
            let query = args["query"].as_str().unwrap_or("");
            if query.is_empty() {
                return mcp_error("Error: query is required.");
            }
            let client = reqwest::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; Ajantis/1.0)")
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default();
            match client
                .get("https://api.duckduckgo.com/")
                .query(&[
                    ("q", query),
                    ("format", "json"),
                    ("no_html", "1"),
                    ("skip_disambig", "1"),
                ])
                .send()
                .await
            {
                Err(e) => mcp_error(&format!("Search failed: {}", e)),
                Ok(resp) => {
                    let data: Value = resp.json().await.unwrap_or(json!({}));
                    let mut lines: Vec<String> = Vec::new();
                    if let Some(abs) = data["AbstractText"].as_str() {
                        if !abs.is_empty() {
                            lines.push(format!("**Summary**: {}", abs));
                            if let Some(src) = data["AbstractURL"].as_str() {
                                if !src.is_empty() {
                                    lines.push(format!("Source: {}", src));
                                }
                            }
                        }
                    }
                    if let Some(topics) = data["RelatedTopics"].as_array() {
                        for t in topics.iter().take(6) {
                            if let Some(text) = t["Text"].as_str() {
                                if !text.is_empty() {
                                    let url = t["FirstURL"].as_str().unwrap_or("");
                                    lines.push(format!("- {} ({})", text, url));
                                }
                            }
                        }
                    }
                    if lines.is_empty() {
                        mcp_ok(&format!("No instant-answer results for '{}'. Try a more specific query or use WebFetch with a direct URL.", query))
                    } else {
                        mcp_ok(&format!("Results for '{}':\n{}", query, lines.join("\n")))
                    }
                }
            }
        }

        // ── Execution ─────────────────────────────────────────────
        "REPL" => {
            let code = args["code"].as_str().unwrap_or("");
            let lang = args["language"].as_str().unwrap_or("python").to_lowercase();
            if code.is_empty() {
                return mcp_error("Error: code is required.");
            }
            if let Err(reason) = check_command_sandbox(code, &sandbox, state) {
                return mcp_error(&reason);
            }
            let interpreter = match lang.as_str() {
                "python" | "python3" | "py" => "python3",
                "javascript" | "js" | "node" => "node",
                "ruby" | "rb" => "ruby",
                "perl" => "perl",
                "lua" => "lua",
                "bash" | "sh" | "shell" => "bash",
                _ => return mcp_error(&format!("Unsupported language: {}", lang)),
            };
            let flag = if interpreter == "bash" { "-c" } else { "-e" };
            let cwd = sandbox.to_string_lossy().to_string();
            let command_repr = format!("{} {} {}", interpreter, flag, code);
            let normalized = normalize_command(&command_repr);
            if let Some(cached) = reuse_cached_command(state, "REPL", &normalized, &cwd) {
                let reused = format!(
                    "[cached result reused | tool: REPL | cwd: {} | command: {}]\n{}",
                    cwd, command_repr, cached.result
                );
                return mcp_ok(&reused);
            }
            match Command::new(interpreter)
                .arg(flag)
                .arg(code)
                .current_dir(&sandbox)
                .output()
            {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    let result_text = if !out.status.success() && !stderr.is_empty() {
                        format!("Runtime error:\n{}", stderr)
                    } else if stderr.is_empty() {
                        if stdout.is_empty() {
                            "[no output]".to_string()
                        } else {
                            stdout.clone()
                        }
                    } else {
                        let text = format!("{}\n{}", stdout, stderr);
                        if text.is_empty() {
                            "[no output]".to_string()
                        } else {
                            text
                        }
                    };
                    remember_command_result(
                        state,
                        "REPL",
                        &command_repr,
                        &normalized,
                        &cwd,
                        out.status.success(),
                        &result_text,
                    );
                    if !out.status.success() && !stderr.is_empty() {
                        mcp_error(&format!("Runtime error:\n{}", stderr))
                    } else {
                        let text = if stderr.is_empty() {
                            stdout
                        } else {
                            format!("{}\n{}", stdout, stderr)
                        };
                        mcp_ok(if text.is_empty() {
                            "[no output]"
                        } else {
                            &text
                        })
                    }
                }
                Err(e) => mcp_error(&format!("Failed to run '{}': {}", interpreter, e)),
            }
        }
        "PowerShell" => {
            let command = args["command"].as_str().unwrap_or("");
            if command.is_empty() {
                return mcp_error("Error: command is required.");
            }
            if let Err(reason) = check_command_sandbox(command, &sandbox, state) {
                return mcp_error(&reason);
            }
            let cwd = sandbox.to_string_lossy().to_string();
            let normalized = normalize_command(command);
            if let Some(cached) = reuse_cached_command(state, "PowerShell", &normalized, &cwd) {
                let reused = format!(
                    "[cached result reused | tool: PowerShell | cwd: {} | command: {}]\n{}",
                    cwd, command, cached.result
                );
                return mcp_ok(&reused);
            }
            match Command::new("pwsh")
                .arg("-Command")
                .arg(command)
                .current_dir(&sandbox)
                .output()
            {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    let text = if stderr.is_empty() {
                        stdout
                    } else {
                        format!("{}\n{}", stdout, stderr)
                    };
                    remember_command_result(
                        state,
                        "PowerShell",
                        command,
                        &normalized,
                        &cwd,
                        out.status.success(),
                        if text.is_empty() {
                            "[no output]"
                        } else {
                            &text
                        },
                    );
                    mcp_ok(if text.is_empty() {
                        "[no output]"
                    } else {
                        &text
                    })
                }
                Err(_) => mcp_error("PowerShell (pwsh) is not available on this system."),
            }
        }
        "Sleep" => {
            let ms = args["duration_ms"]
                .as_u64()
                .or_else(|| args["seconds"].as_f64().map(|s| (s * 1000.0) as u64))
                .unwrap_or(1000)
                .min(30_000); // cap at 30 s
            tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
            mcp_ok(&format!("Slept {}ms.", ms))
        }

        // ── UI / user interaction ─────────────────────────────────
        "SendUserMessage" => {
            let message = args["message"].as_str().unwrap_or("");
            if message.is_empty() {
                return mcp_error("Error: message is required.");
            }
            let maybe_ch = state.event_channel.lock().unwrap().clone();
            let run_id =
                primary_run_id(&state.active_runs).unwrap_or_else(|| "detached-run".to_string());
            if let Some(ref ch) = maybe_ch {
                let _ = ch.send(StreamEvent::AgentStart {
                    run_id: run_id.clone(),
                    agent_id: "agent-notification".to_string(),
                    agent_name: "Notification".to_string(),
                    model_key: "system/notification".to_string(),
                    mode: "ephemeral".to_string(),
                    is_manager: false,
                    context_limit: 0,
                    estimated_input_tokens: 0,
                    estimated_remaining_tokens: 0,
                });
                let _ = ch.send(StreamEvent::Token {
                    run_id: run_id.clone(),
                    agent_id: "agent-notification".to_string(),
                    content: message.to_string(),
                });
                let _ = ch.send(StreamEvent::AgentEnd {
                    run_id: run_id.clone(),
                    agent_id: "agent-notification".to_string(),
                });
                mcp_ok("Message delivered to user.")
            } else {
                mcp_error("No active session; message could not be delivered.")
            }
        }
        "AskUserQuestion" => {
            let question = args["question"].as_str().unwrap_or("").trim();
            let msg = if question.is_empty() {
                "AskUserQuestion is unavailable in this environment. Continue by inspecting the workspace or making a reasonable assumption."
            } else {
                "AskUserQuestion is unavailable in this environment. Do not ask the user this question now; inspect the workspace or make a reasonable assumption and continue."
            };
            mcp_error(msg)
        }

        // ── Output / tools ────────────────────────────────────────
        "StructuredOutput" => {
            let output = args.get("output").unwrap_or(args);
            mcp_ok(&serde_json::to_string_pretty(output).unwrap_or_default())
        }
        "ToolSearch" => {
            let query = args["query"].as_str().unwrap_or("").to_lowercase();
            let max = args["max_results"].as_u64().unwrap_or(5) as usize;
            if query.is_empty() {
                return mcp_error("Error: query is required.");
            }
            let hits: Vec<Value> = state.tools.iter()
                .filter(|t| t.name.to_lowercase().contains(&query)
                    || t.description.to_lowercase().contains(&query))
                .take(max)
                .map(|t| json!({ "name": t.name, "description": &t.description[..t.description.len().min(100)] }))
                .collect();
            if hits.is_empty() {
                mcp_ok(&format!("No tools matched '{}'.", query))
            } else {
                mcp_ok(&format!(
                    "{} tool(s) found:\n{}",
                    hits.len(),
                    serde_json::to_string_pretty(&hits).unwrap_or_default()
                ))
            }
        }
        "Config" => {
            let cfg = json!({
                "mcp_port": state.mcp_port,
                "workspace_root": sandbox.to_string_lossy(),
                "agent_count": state.agent_config.lock().unwrap().agents.len(),
                "memory_entries": state.memory_pool.lock().unwrap().entries.len(),
                "background_tasks": state.tasks.lock().unwrap().len(),
            });
            let key = args["key"].as_str().unwrap_or("");
            if key.is_empty() {
                mcp_ok(&serde_json::to_string_pretty(&cfg).unwrap_or_default())
            } else {
                match cfg.get(key) {
                    Some(v) => mcp_ok(&v.to_string()),
                    None => mcp_error(&format!("Config key '{}' not found.", key)),
                }
            }
        }

        // ── Background tasks ──────────────────────────────────────
        "TaskCreate" => {
            let command = args["command"].as_str().unwrap_or("");
            let description = args["description"].as_str().unwrap_or("background task");
            if command.is_empty() {
                return mcp_error("Error: command is required.");
            }
            create_task_internal(command, description, &sandbox, state).await
        }
        "RunTaskPacket" => {
            let command = args["command"]
                .as_str()
                .or_else(|| args["script"].as_str())
                .unwrap_or("");
            let description = args["description"].as_str().unwrap_or("task packet");
            if command.is_empty() {
                return mcp_error("Error: command or script is required.");
            }
            create_task_internal(command, description, &sandbox, state).await
        }
        "TaskGet" => {
            let id = args["task_id"].as_str().unwrap_or("");
            if id.is_empty() {
                return mcp_error("Error: task_id is required.");
            }
            let tasks = state.tasks.lock().unwrap();
            match tasks.get(id) {
                None => mcp_error(&format!("Task '{}' not found.", id)),
                Some(t) => {
                    let status = t.status.lock().unwrap().clone();
                    let preview = t
                        .output
                        .lock()
                        .unwrap()
                        .chars()
                        .take(300)
                        .collect::<String>();
                    let r = json!({
                        "task_id": t.id, "description": t.description,
                        "status": status, "started_at": t.started_at,
                        "output_preview": preview,
                    });
                    mcp_ok(&serde_json::to_string_pretty(&r).unwrap_or_default())
                }
            }
        }
        "TaskList" => {
            let tasks = state.tasks.lock().unwrap();
            let list: Vec<Value> = tasks
                .values()
                .map(|t| {
                    json!({
                        "task_id": t.id,
                        "description": t.description,
                        "status": t.status.lock().unwrap().clone(),
                        "started_at": t.started_at,
                    })
                })
                .collect();
            mcp_ok(&format!(
                "{} task(s):\n{}",
                list.len(),
                serde_json::to_string_pretty(&list).unwrap_or_default()
            ))
        }
        "TaskStop" => {
            let id = args["task_id"].as_str().unwrap_or("");
            if id.is_empty() {
                return mcp_error("Error: task_id is required.");
            }
            let mut tasks = state.tasks.lock().unwrap();
            match tasks.get_mut(id) {
                None => mcp_error(&format!("Task '{}' not found.", id)),
                Some(t) => {
                    if let Some(ref h) = t.abort_handle {
                        h.abort();
                    }
                    *t.status.lock().unwrap() = "stopped".to_string();
                    mcp_ok(&format!("Task '{}' stopped.", id))
                }
            }
        }
        "TaskOutput" => {
            let id = args["task_id"].as_str().unwrap_or("");
            if id.is_empty() {
                return mcp_error("Error: task_id is required.");
            }
            let tasks = state.tasks.lock().unwrap();
            match tasks.get(id) {
                None => mcp_error(&format!("Task '{}' not found.", id)),
                Some(t) => {
                    let status = t.status.lock().unwrap().clone();
                    let out = t.output.lock().unwrap().clone();
                    mcp_ok(&format!(
                        "[Task {} — {}]\n{}",
                        id,
                        status,
                        if out.is_empty() {
                            "[no output yet]".to_string()
                        } else {
                            out
                        }
                    ))
                }
            }
        }
        "TaskUpdate" => {
            // Process-based tasks don't support runtime stdin injection.
            mcp_ok("Acknowledged — process-based tasks do not support runtime message injection.")
        }

        // ── Compatibility shims for Claude Code-specific tools ────
        "Agent" => {
            let description = args["description"].as_str().unwrap_or("").trim();
            let prompt = args["prompt"].as_str().unwrap_or("").trim();
            if description.is_empty() {
                return mcp_error("Error: Agent.description is required.");
            }
            if prompt.is_empty() {
                return mcp_error("Error: Agent.prompt is required.");
            }
            let role = args["name"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    args["subagent_type"]
                        .as_str()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                })
                .unwrap_or("worker");
            let compat_prompt = build_compat_agent_prompt(role, prompt);
            let initial_task = if prompt.is_empty() || prompt == description {
                description.to_string()
            } else {
                format!(
                    "Task:\n{}\n\nAdditional instructions from the caller:\n{}",
                    description, prompt
                )
            };
            let requested_model = args["model"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());

            let agent_id = match spawn_agent_record(
                state,
                caller_agent_id,
                role,
                &compat_prompt,
                requested_model,
                None,
                None,
            )
            .await
            {
                Ok(agent_id) => agent_id,
                Err(err) => return mcp_error(&err),
            };

            match run_agent_turn(
                state,
                &agent_id,
                &initial_task,
                true,
                caller_active_behaviors(caller_agent_id, state),
            )
            .await
            {
                Ok(payload) => mcp_ok(&payload),
                Err(err) => mcp_error(&err),
            }
        }
        _ => mcp_error(&format!(
            "Tool '{}' is not recognised by this MCP server.",
            name
        )),
    }
}

fn default_worker_tools(tools: &[McpTool]) -> Vec<String> {
    tools
        .iter()
        .filter(|tool| !is_manager_only_tool(&tool.name))
        .map(|tool| tool.name.clone())
        .collect()
}

async fn spawn_agent_record(
    state: &McpState,
    caller_agent_id: Option<&str>,
    role: &str,
    sys_prompt: &str,
    requested_model: Option<String>,
    ctx_limit: Option<u64>,
    allowed_tools: Option<Vec<String>>,
) -> Result<String, String> {
    let inherited_model = caller_agent_id.and_then(|agent_id| {
        let config = state.agent_config.lock().unwrap();
        config
            .agents
            .iter()
            .find(|agent| agent.id == agent_id)
            .and_then(|agent| agent.model_key.clone())
    });
    let available_models = fetch_models()
        .await
        .map_err(|e| format!("Error fetching local models: {}", e))?;
    let selected_model = requested_model.or(inherited_model);
    let Some(model_key) = selected_model else {
        return Err(
            "Error: spawn_agent requires a model, or the caller must already have one to inherit."
                .to_string(),
        );
    };
    let Some(model_info) = available_models.iter().find(|model| model.key == model_key) else {
        let available = available_models
            .iter()
            .take(20)
            .map(|model| model.key.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Error: model '{}' is not available on this computer. Choose one of the locally available models{}{}",
            model_key,
            if available.is_empty() { "" } else { ": " },
            available,
        ));
    };
    if model_info.model_type.as_deref() == Some("embedding") {
        return Err(format!(
            "Error: model '{}' is an embedding model and cannot be used for spawn_agent chat workers.",
            model_key
        ));
    }

    let ts = chrono::Utc::now().timestamp_millis();
    let agent_id = format!("agent-{}-{}", role, ts);
    let new_agent = Agent {
        id: agent_id.clone(),
        name: role.to_string(),
        agent_type: "model".to_string(),
        model_key: Some(model_key),
        model_type: model_info
            .model_type
            .clone()
            .or_else(|| Some("llm".to_string())),
        role: Some(sys_prompt.to_string()),
        load_config: ctx_limit.map(|cl| AgentLoadConfig {
            context_length: Some(cl),
            eval_batch_size: None,
            flash_attention: None,
            num_experts: None,
            offload_kv_cache_to_gpu: None,
        }),
        mode: Some("stay_awake".to_string()),
        allowed_tools: Some(allowed_tools.unwrap_or_else(|| default_worker_tools(&state.tools))),
        armed: true,
        is_manager: false,
        paused: false,
    };
    state.agent_config.lock().unwrap().agents.push(new_agent);
    Ok(agent_id)
}

fn build_compat_agent_prompt(role_hint: &str, caller_prompt: &str) -> String {
    let normalized_role = normalize_role_hint(role_hint);
    let specialization = match normalized_role.as_str() {
        "explorer" => {
            "Focus on identifying the minimum relevant files, directories, and entry points needed for the task, not whole-tree inventories. Return a compact result with `Entry points`, `Relevant paths`, and `Coverage gaps`. Do not stop at a blocked summary if the next obvious file to inspect is already implied."
        }
        "executor" => {
            "Prefer direct execution with concise evidence. If asked to show command output, run the command and report the actual result."
        }
        "analyst" | "analyzer" => {
            "Read the relevant sources and produce a factual, deduplicated summary grounded in the files you inspected. Return `Observed evidence`, `Inferences`, and `Coverage gaps`. Every substantive conclusion must cite the file or observed code/config behavior behind it. If the caller names a broad scope, treat it as a coverage checklist and explicitly call out anything still uninspected."
        }
        "security_auditor" => {
            "Look for concrete vulnerabilities, risky configurations, and unsafe patterns. Prefer verified findings over speculation. Do not assign severity without direct code/config evidence. If a concern is plausible but unproven, label it as a lower-confidence hypothesis only if you can cite a concrete file/config or observed behavior and state what proof is still missing. If the caller names a broad scope, treat it as a coverage checklist and explicitly call out anything still uninspected."
        }
        "code_reviewer" => {
            "Review behavior critically. Call out bugs, regressions, unsafe assumptions, and missing tests before giving any summary. Base every finding on code you actually inspected."
        }
        "verifier" => {
            "Verify whether the implementation actually satisfies the request. Prefer concrete checks over intent."
        }
        _ => "Use the available tools carefully and complete the assigned sub-task directly.",
    };

    format!(
        "You are a specialized {} subagent working inside a local repository.\n\n{}\n\nGeneral rules:\n- Execute the assigned sub-task directly.\n- Prefer targeted tool calls, narrow reads, and relevant entrypoints over exhaustive scans.\n- Avoid dependency, build, and generated directories by default (`node_modules`, `dist`, `target`, `.git`, caches) unless the task is specifically about them.\n- If the task is to build, implement, create, or modify workspace files, use the file tools to materialize the result on disk; a chat-only code block does not count as completion unless the caller explicitly asked for chat-only output.\n- If no target path is specified for a file task, choose a sensible path and state it in your result.\n- For audit/review work, prefer direct file or config inspection over speculation.\n- Treat explicitly requested files, directories, file classes, and subsystems as a coverage checklist; if you did not inspect one, call it out explicitly as a coverage gap.\n- Quote the file/path and the specific observed behavior behind each conclusion.\n- Do not emit generic or dependency-only risk templates.\n- Do not ask the user follow-up questions in this environment.\n- Do not stop at a plan, status update, or statement of intent.\n- If a tool is blocked or too broad, immediately adapt to a narrower command or read.\n- If evidence is insufficient, say `insufficient evidence` instead of inventing a finding.\n- Return the concrete result requested by the caller as one deduplicated answer.\n\nSupplementary guidance from the caller:\n{}",
        role_hint,
        specialization,
        if caller_prompt.trim().is_empty() {
            "[none]"
        } else {
            caller_prompt
        }
    )
}

#[cfg(test)]
mod tests {
    use super::build_compat_agent_prompt;

    #[test]
    fn compat_prompt_requires_writing_file_tasks_to_disk() {
        let prompt = build_compat_agent_prompt("executor", "Create a Python game");
        assert!(prompt.contains("materialize the result on disk"));
        assert!(prompt.contains("chat-only code block does not count as completion"));
        assert!(prompt.contains("choose a sensible path"));
    }

    #[test]
    fn compat_prompt_steers_targeted_reads_and_deduplicated_summaries() {
        let explorer = build_compat_agent_prompt("explorer", "Inspect the repo");
        assert!(explorer.contains("minimum relevant files"));
        assert!(explorer.contains("not whole-tree inventories"));

        let analyzer = build_compat_agent_prompt("analyzer", "Summarize findings");
        assert!(analyzer.contains("deduplicated summary"));
        assert!(analyzer.contains("coverage checklist"));

        let general = build_compat_agent_prompt("executor", "Run checks");
        assert!(general.contains("Avoid dependency, build, and generated directories"));
        assert!(general.contains("blocked or too broad"));
        assert!(general.contains("one deduplicated answer"));
    }
}

fn normalize_role_hint(role_hint: &str) -> String {
    let mut normalized = String::with_capacity(role_hint.len());
    let mut last_was_sep = false;
    for ch in role_hint.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            normalized.push('_');
            last_was_sep = true;
        }
    }
    normalized.trim_matches('_').to_string()
}

fn normalize_command(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn reuse_cached_command(
    state: &McpState,
    tool_name: &str,
    normalized_command: &str,
    cwd: &str,
) -> Option<CommandExecution> {
    state
        .command_history
        .lock()
        .unwrap()
        .find_exact(tool_name, normalized_command, cwd)
}

fn remember_command_result(
    state: &McpState,
    tool_name: &str,
    command: &str,
    normalized_command: &str,
    cwd: &str,
    success: bool,
    result: &str,
) {
    let mut history = state.command_history.lock().unwrap();
    history.push(tool_name, command, normalized_command, cwd, success, result);
    let last_entry = history.entries.last().cloned();
    drop(history);
    if let (Some(run_id), Some(entry)) = (primary_run_id(&state.active_runs), last_entry.as_ref()) {
        record_dossier_command(&state.active_runs, &run_id, entry);
    }
}

fn caller_active_behaviors(
    caller_agent_id: Option<&str>,
    state: &McpState,
) -> Option<HashSet<String>> {
    let caller_id = caller_agent_id?;
    if caller_id.is_empty() {
        return None;
    }
    state
        .active_behavior_contexts
        .lock()
        .unwrap()
        .get(caller_id)
        .cloned()
}

async fn run_agent_turn(
    state: &McpState,
    agent_id: &str,
    content: &str,
    await_reply: bool,
    inherited_behaviors: Option<HashSet<String>>,
) -> Result<String, String> {
    let (
        model_key,
        sys_prompt,
        agent_name,
        is_manager,
        allowed_tools,
        context_limit,
        behavior_triggers,
    ) = {
        let config = state.agent_config.lock().unwrap();
        match config.agents.iter().find(|a| a.id == agent_id) {
            None => return Err(format!("Error: agent '{}' not found.", agent_id)),
            Some(a) => {
                if !a.armed {
                    return Err(format!("Error: agent '{}' is disarmed.", agent_id));
                }
                if a.paused {
                    return Err(format!("Error: agent '{}' is paused.", agent_id));
                }
                (
                    a.model_key.clone().unwrap_or_default(),
                    apply_runtime_agent_context(
                        &a.role.clone().unwrap_or_default(),
                        a.is_manager,
                        &state.command_history.lock().unwrap().clone(),
                    ),
                    a.name.clone(),
                    a.is_manager,
                    if a.is_manager {
                        Vec::new()
                    } else {
                        a.allowed_tools
                            .clone()
                            .unwrap_or_else(|| default_worker_tools(&state.tools))
                    },
                    a.load_config.as_ref().and_then(|cfg| cfg.context_length),
                    config.behavior_triggers.clone(),
                )
            }
        }
    };

    if model_key.is_empty() {
        return Err(format!("Error: agent '{}' has no model.", agent_id));
    }

    if !await_reply {
        let payload = json!({
            "status": "queued",
            "agent_id": agent_id,
            "note": "No reply was awaited. To execute the agent immediately, call send_message with await_reply=true."
        });
        return Ok(serde_json::to_string(&payload).unwrap_or_default());
    }

    let (active_behaviors, _embed_calls) = resolve_active_behaviors(
        content,
        inherited_behaviors.as_ref(),
        &behavior_triggers,
        &state.behavior_trigger_cache,
    )
    .await;
    let audit_config = resolve_audit_behavior_config(&active_behaviors, &behavior_triggers);

    let history: Vec<Value> = {
        let pool = state.memory_pool.lock().unwrap();
        pool.entries
            .iter()
            .filter(|e| e.agent_id == agent_id && (e.role == "user" || e.role == "assistant"))
            .map(|e| json!({"role": e.role, "content": e.content}))
            .collect()
    };

    let maybe_ch = state.event_channel.lock().unwrap().clone();
    let run_id = primary_run_id(&state.active_runs).unwrap_or_else(|| "detached-run".to_string());
    if let Some(ref ch) = maybe_ch {
        let budget = compute_context_budget(&sys_prompt, &history, &content, context_limit, 0);
        let _ = ch.send(StreamEvent::AgentStart {
            run_id: run_id.clone(),
            agent_id: agent_id.to_string(),
            agent_name: agent_name.clone(),
            model_key: model_key.clone(),
            mode: "stay_awake".to_string(),
            is_manager,
            context_limit: budget.limit,
            estimated_input_tokens: budget.estimated_used,
            estimated_remaining_tokens: budget.remaining,
        });
        let _ = ch.send(StreamEvent::AgentStatus {
            run_id: run_id.clone(),
            agent_id: agent_id.to_string(),
            stage: "thinking".to_string(),
            detail: "Generating response".to_string(),
        });
    }
    if (is_manager || !allowed_tools.is_empty()) && maybe_ch.is_none() {
        return Err("Error: tool-enabled agents require an active routed session.".to_string());
    }

    let uses_tool_loop = is_manager || !allowed_tools.is_empty();

    let response_result = if uses_tool_loop {
        let allowed = if is_manager {
            None
        } else {
            Some(allowed_tools.as_slice())
        };
        let glob_ready = state
            .glob_cache
            .lock()
            .unwrap()
            .get(agent_id)
            .map(|matches| !matches.is_empty())
            .unwrap_or(false);
        crate::chat::call_chat_with_tools(
            &model_key,
            &sys_prompt,
            content,
            &run_id,
            agent_id,
            &state.tools,
            allowed,
            is_manager,
            is_manager,
            context_limit,
            glob_ready,
            state,
            Some(&active_behaviors),
            maybe_ch.as_ref().unwrap_or_else(|| unreachable!()),
            &history,
        )
        .await
    } else {
        call_chat_blocking(&model_key, &sys_prompt, content, &history, context_limit).await
    };

    match response_result {
        Ok(response) => {
            if let Some(config) = audit_config.as_ref() {
                let requested_refs = extract_explicit_audit_refs(content);
                if let Err(reason) =
                    validate_audit_worker_response(&response, &requested_refs, config)
                {
                    return Err(format!("Agent '{}' {}", agent_name, reason));
                }
            }
            if response.trim().is_empty() {
                return Err(format!(
                    "Agent '{}' completed its turn without returning a usable answer.",
                    agent_name
                ));
            }
            {
                let mut pool = state.memory_pool.lock().unwrap();
                pool.push(agent_id, &agent_name, "user", content);
                pool.push(agent_id, &agent_name, "assistant", &response);
            }
            if let Some(ref ch) = maybe_ch {
                if !uses_tool_loop {
                    let _ = ch.send(StreamEvent::Token {
                        run_id: run_id.clone(),
                        agent_id: agent_id.to_string(),
                        content: response.clone(),
                    });
                }
                let _ = ch.send(StreamEvent::AgentEnd {
                    run_id: run_id.clone(),
                    agent_id: agent_id.to_string(),
                });
            }
            let payload = json!({ "agent_id": agent_id, "response": response });
            Ok(serde_json::to_string(&payload).unwrap_or_default())
        }
        Err(e) => {
            if let Some(ref ch) = maybe_ch {
                let _ = ch.send(StreamEvent::Error {
                    run_id: run_id.clone(),
                    agent_id: agent_id.to_string(),
                    agent_name: agent_name.clone(),
                    message: e.clone(),
                });
            }
            Err(format!("Error calling agent: {}", e))
        }
    }
}

fn enforce_tool_policy(
    name: &str,
    caller_agent_id: Option<&str>,
    state: &McpState,
) -> Result<(), String> {
    let Some(caller_id) = caller_agent_id.filter(|id| !id.is_empty()) else {
        return Ok(());
    };
    let config = state.agent_config.lock().unwrap();
    let Some(caller) = config.agents.iter().find(|agent| agent.id == caller_id) else {
        return Ok(());
    };
    if caller.is_manager && is_manager_blocked_tool(name) {
        return Err(format!(
            "Managers cannot call '{}' directly. Delegate this work to a sub-agent instead.",
            name
        ));
    }
    if !caller.is_manager && is_manager_only_tool(name) {
        return Err(format!(
            "Tool '{}' is manager-only. Non-manager agents can only use worker tools plus self-introspection.",
            name
        ));
    }
    Ok(())
}

fn enforce_send_message_target(
    caller_agent_id: Option<&str>,
    target_agent_id: &str,
    state: &McpState,
) -> Result<(), String> {
    let Some(caller_id) = caller_agent_id.filter(|id| !id.is_empty()) else {
        return Ok(());
    };
    let config = state.agent_config.lock().unwrap();
    let Some(caller) = config.agents.iter().find(|agent| agent.id == caller_id) else {
        return Ok(());
    };
    if caller.is_manager || caller.id == target_agent_id {
        Ok(())
    } else {
        Err("Non-manager agents may only send messages to themselves. Cross-agent delegation is manager-only.".to_string())
    }
}

fn enforce_agent_introspection(
    caller_agent_id: Option<&str>,
    target_agent_id: &str,
    state: &McpState,
) -> Result<(), String> {
    let Some(caller_id) = caller_agent_id.filter(|id| !id.is_empty()) else {
        return Ok(());
    };
    let config = state.agent_config.lock().unwrap();
    let Some(caller) = config.agents.iter().find(|agent| agent.id == caller_id) else {
        return Ok(());
    };
    if caller.is_manager || caller.id == target_agent_id {
        Ok(())
    } else {
        Err(
            "Non-manager agents may only inspect their own runtime state and message history."
                .to_string(),
        )
    }
}

fn agent_context_limit(agent_id: &str, state: &McpState) -> Option<u64> {
    let config = state.agent_config.lock().unwrap();
    config
        .agents
        .iter()
        .find(|agent| agent.id == agent_id)
        .and_then(|agent| {
            agent
                .load_config
                .as_ref()
                .and_then(|cfg| cfg.context_length)
        })
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        dirs_next::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(rest)
    } else {
        PathBuf::from(path)
    }
}

fn allowed_roots(sandbox: &PathBuf) -> Vec<PathBuf> {
    vec![normalize_path(sandbox), normalize_path(&ajantis_dir())]
}

fn is_allowed_path(path: &Path, sandbox: &PathBuf) -> bool {
    let normalized = normalize_path(path);
    allowed_roots(sandbox)
        .iter()
        .any(|root| normalized.starts_with(root))
}

fn resolve_allowed_path(raw_path: &str, sandbox: &PathBuf) -> Result<PathBuf, String> {
    let raw = raw_path.trim();
    let candidate = if raw.starts_with("~/") {
        expand_tilde(raw)
    } else {
        let path = PathBuf::from(raw);
        if path.is_absolute() {
            path
        } else {
            sandbox.join(path)
        }
    };
    let normalized = normalize_path(&candidate);
    if is_allowed_path(&normalized, sandbox) {
        Ok(normalized)
    } else {
        Err("Error: access outside the workspace or ~/.ajantis is not allowed.".to_string())
    }
}

fn enforce_glob_match(
    path: &Path,
    caller_agent_id: Option<&str>,
    state: &McpState,
) -> Result<(), String> {
    let Some(agent_id) = caller_agent_id.filter(|id| !id.is_empty()) else {
        return Ok(());
    };
    let cache = state.glob_cache.lock().unwrap();
    let Some(matches) = cache.get(agent_id) else {
        return Err(
            "Error: use glob_search first, then read or grep only files returned by that glob."
                .to_string(),
        );
    };
    let target = path.to_string_lossy().to_string();
    if matches.contains(&target) {
        Ok(())
    } else {
        Err(format!(
            "Error: '{}' was not returned by your latest glob_search. Narrow the scope with glob_search first.",
            target
        ))
    }
}

fn extract_file_scope(
    data: &str,
    scope: Option<&str>,
    offset: usize,
    limit: usize,
) -> Result<(String, String), String> {
    match scope.map(str::trim).filter(|s| !s.is_empty()) {
        Some(scope_spec) if scope_spec.starts_with("line:") || scope_spec.starts_with("lines:") => {
            let range = scope_spec
                .split_once(':')
                .map(|(_, rest)| rest)
                .ok_or_else(|| "Invalid line scope. Use 'line:start-end'.".to_string())?;
            let (start, end) = parse_inclusive_range(range)?;
            let start_index = start.saturating_sub(1);
            let line_count = end.saturating_sub(start_index);
            let scoped = data
                .lines()
                .skip(start_index)
                .take(line_count)
                .collect::<Vec<_>>()
                .join("\n");
            let slice: String = scoped.chars().skip(offset).take(limit).collect();
            Ok((slice, format!("lines:{}-{}", start, end)))
        }
        Some(scope_spec) if scope_spec.starts_with("char:") || scope_spec.starts_with("chars:") => {
            let range = scope_spec
                .split_once(':')
                .map(|(_, rest)| rest)
                .ok_or_else(|| "Invalid char scope. Use 'char:start-end'.".to_string())?;
            let (start, end) = parse_inclusive_range(range)?;
            let slice: String = data
                .chars()
                .skip(start.saturating_sub(1) + offset)
                .take(end.saturating_sub(start.saturating_sub(1)).min(limit))
                .collect();
            Ok((slice, format!("chars:{}-{}", start, end)))
        }
        Some(_) => Err("Invalid scope. Use 'line:start-end' or 'char:start-end'.".to_string()),
        None => {
            let slice: String = data.chars().skip(offset).take(limit).collect();
            let end = offset + slice.chars().count();
            Ok((slice, format!("chars:{}-{}", offset + 1, end)))
        }
    }
}

fn parse_inclusive_range(raw: &str) -> Result<(usize, usize), String> {
    let (start, end) = raw
        .split_once('-')
        .ok_or_else(|| "Invalid range. Use 'start-end'.".to_string())?;
    let start = start
        .trim()
        .parse::<usize>()
        .map_err(|_| "Invalid range start.".to_string())?;
    let end = end
        .trim()
        .parse::<usize>()
        .map_err(|_| "Invalid range end.".to_string())?;
    if start == 0 || end < start {
        return Err("Invalid range bounds.".to_string());
    }
    Ok((start, end))
}

// ── Sandbox helpers ────────────────────────────────────────────────

/// Returns `Err(reason)` if the command is blocked by the sandbox policy or
/// contains a path that would escape `workspace_root`.
fn check_command_sandbox(command: &str, sandbox: &PathBuf, state: &McpState) -> Result<(), String> {
    // ── 1. Reject obvious path-traversal attempts ─────────────────
    //  Covers "../", "/..", cd /absolute/outside, etc.
    if command.contains("../") || command.contains("/..") || command.contains("\\..") {
        return Err("Blocked: path traversal ('..') is not allowed.".into());
    }

    let roots = allowed_roots(sandbox);
    for token in command.split_whitespace() {
        let candidate = token.trim_matches(|c| matches!(c, '"' | '\'' | ',' | ';' | ')'));
        if candidate.starts_with('/') || candidate.starts_with("~/") {
            let expanded = if candidate.starts_with("~/") {
                expand_tilde(candidate)
            } else {
                PathBuf::from(candidate)
            };
            let normalized = normalize_path(&expanded);
            let allowed = roots.iter().any(|root| normalized.starts_with(root));
            if !allowed {
                let roots_text = roots
                    .iter()
                    .map(|root| root.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "Blocked: absolute path '{}' is outside the allowed roots: {}.",
                    candidate, roots_text
                ));
            }
        } else if candidate.contains("/..") || candidate.starts_with("../") {
            return Err(format!(
                "Blocked: path '{}' escapes the allowed roots.",
                candidate
            ));
        }
    }

    // ── 2. Check CommandPolicy from AgentConfig ───────────────────
    let policy = state.agent_config.lock().unwrap().command_policy.clone();

    let cmd_lower = command.trim().to_lowercase();

    // Denylist has highest priority.
    for denied in &policy.denylist {
        if cmd_lower.starts_with(&denied.to_lowercase()) {
            return Err(format!(
                "Blocked by denylist: command matches '{}'.",
                denied
            ));
        }
    }

    // Allowlist: if non-empty the command must match at least one entry.
    if !policy.allowlist.is_empty() {
        let allowed = policy
            .allowlist
            .iter()
            .any(|a| cmd_lower.starts_with(&a.to_lowercase()));
        if !allowed {
            return Err(format!(
                "Blocked: command is not in the allowlist. Allowed prefixes: {}",
                policy.allowlist.join(", ")
            ));
        }
    }

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────

/// Strip HTML tags and collapse whitespace to plain text.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Shared helper — spawns a shell command as a background task and returns the task id.
async fn create_task_internal(
    command: &str,
    description: &str,
    sandbox: &PathBuf,
    state: &McpState,
) -> Value {
    if let Err(reason) = check_command_sandbox(command, sandbox, state) {
        return mcp_error(&reason);
    }
    let id = format!("task-{}", chrono::Utc::now().timestamp_millis());
    let output = Arc::new(Mutex::new(String::new()));
    let status = Arc::new(Mutex::new("running".to_string()));
    let out_c = output.clone();
    let stat_c = status.clone();
    let cmd_str = command.to_string();
    let ws = sandbox.clone();

    let join = tokio::task::spawn(async move {
        match tokio::process::Command::new("bash")
            .arg("-c")
            .arg(&cmd_str)
            .current_dir(&ws)
            .output()
            .await
        {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                *out_c.lock().unwrap() = format!(
                    "{}{}",
                    stdout,
                    if stderr.is_empty() {
                        String::new()
                    } else {
                        format!("\nstderr:\n{}", stderr)
                    }
                );
                *stat_c.lock().unwrap() = if o.status.success() {
                    "completed".to_string()
                } else {
                    format!("failed (exit {:?})", o.status.code())
                };
            }
            Err(e) => {
                *stat_c.lock().unwrap() = format!("failed: {}", e);
            }
        }
    });

    let task = AjantisTask {
        id: id.clone(),
        description: description.to_string(),
        status,
        output,
        abort_handle: Some(join.abort_handle()),
        started_at: chrono::Utc::now().to_rfc3339(),
    };
    state.tasks.lock().unwrap().insert(id.clone(), task);
    mcp_ok(&format!(r#"{{"task_id":"{}","status":"running"}}"#, id))
}

pub(crate) fn mcp_ok(text: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

pub(crate) fn mcp_error(text: &str) -> Value {
    json!({ "isError": true, "content": [{ "type": "text", "text": text }] })
}

pub(crate) async fn start_mcp_server(port: u16, mcp_state: McpState) {
    let cors = CorsLayer::new()
        .allow_origin("*".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/", post(handle_jsonrpc))
        .route(
            "/list_tools",
            get({
                let tools = mcp_state.tools.clone();
                move || async move { Json(json!({ "tools": tools })) }
            }),
        )
        .layer(cors)
        .with_state(mcp_state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .expect("Failed to bind MCP server port");
    log::info!("MCP server listening on http://localhost:{}", port);
    axum::serve(listener, app).await.ok();
}

/// Load tools from the `tools.json` embedded at compile time.
/// The path is relative to `src-tauri/` (where Cargo.toml lives).
pub(crate) fn load_tools_embedded() -> Vec<McpTool> {
    const TOOLS_JSON: &str = include_str!("../../tools.json");
    let parsed: serde_json::Value =
        serde_json::from_str(TOOLS_JSON).expect("tools.json is not valid JSON");
    parsed
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|entry| {
                    let func = entry.get("function")?;
                    Some(McpTool {
                        name: func["name"].as_str().unwrap_or("").to_string(),
                        description: func["description"].as_str().unwrap_or("").to_string(),
                        input_schema: func
                            .get("parameters")
                            .cloned()
                            .unwrap_or(serde_json::json!({"type": "object"})),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

#[allow(dead_code)]
pub(crate) fn load_tools(tools_path: &PathBuf) -> Vec<McpTool> {
    match fs::read_to_string(tools_path) {
        Ok(content) => {
            let parsed: Value = serde_json::from_str(&content).unwrap_or(Value::Null);
            parsed
                .get("tools")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|entry| {
                            let func = entry.get("function")?;
                            Some(McpTool {
                                name: func["name"].as_str().unwrap_or("").to_string(),
                                description: func["description"].as_str().unwrap_or("").to_string(),
                                input_schema: func
                                    .get("parameters")
                                    .cloned()
                                    .unwrap_or(json!({"type": "object"})),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        Err(e) => {
            log::error!("Failed to load tools.json: {}", e);
            vec![]
        }
    }
}
