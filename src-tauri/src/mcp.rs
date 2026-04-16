use axum::{
    extract::State as AxumState,
    http::{HeaderValue, Method},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tauri::ipc::Channel;
use tower_http::cors::CorsLayer;

use crate::agent_config::{Agent, AgentConfig, AgentLoadConfig};
use crate::chat::{call_chat_blocking, StreamEvent};
use crate::memory::{MemoryEntry, MemoryPool};
use crate::state::McpTool;

// ── Background task tracking ───────────────────────────────────────

pub(crate) struct AjantisTask {
    pub id: String,
    pub description: String,
    pub status: Arc<Mutex<String>>,  // "running" | "completed" | "failed: …" | "stopped"
    pub output: Arc<Mutex<String>>,
    pub abort_handle: Option<tokio::task::AbortHandle>,
    pub started_at: String,
}

// ── MCP server state ───────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct McpState {
    pub(crate) tools: Vec<McpTool>,
    /// Static root used as fallback when no workspace is selected.
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
    pub(crate) event_channel: Arc<Mutex<Option<Channel<StreamEvent>>>>,
    /// Background tasks spawned via TaskCreate.
    pub(crate) tasks: Arc<Mutex<HashMap<String, AjantisTask>>>,
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
            jsonrpc: "2.0".to_string(), id: req.id, result: None,
            error: Some(json!({"code": -32600, "message": "Invalid JSON-RPC version"})),
        });
    }
    if req.id.is_none() {
        return Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(), id: None, result: Some(Value::Null), error: None,
        });
    }
    let id = req.id.clone();
    match req.method.as_str() {
        "initialize" => Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(), id,
            result: Some(json!({
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "ajantis-mcp", "version": "2.0.0" },
            })),
            error: None,
        }),
        "tools/list" => {
            let tools: Vec<Value> = state.tools.iter().map(|t| json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            })).collect();
            Json(JsonRpcResponse {
                jsonrpc: "2.0".to_string(), id,
                result: Some(json!({ "tools": tools })),
                error: None,
            })
        }
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let name = params["name"].as_str().unwrap_or("").to_string();
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let result = handle_tool_call(&name, &args, &state).await;
            Json(JsonRpcResponse {
                jsonrpc: "2.0".to_string(), id,
                result: Some(result), error: None,
            })
        }
        "resources/list" => Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(), id,
            result: Some(json!({ "resources": [] })), error: None,
        }),
        _ => Json(JsonRpcResponse {
            jsonrpc: "2.0".to_string(), id, result: None,
            error: Some(json!({"code": -32601, "message": format!("Unknown method: {}", req.method)})),
        }),
    }
}

pub(crate) async fn handle_tool_call(name: &str, args: &Value, state: &McpState) -> Value {
    // Resolve the active sandbox root: selected workspace if set, fallback to workspace_root.
    let sandbox: PathBuf = state.active_workspace.lock().unwrap().clone();

    match name {
        // ── File system ───────────────────────────────────────────
        "bash" => {
            let command = args["command"].as_str().unwrap_or("").trim().to_string();
            if command.is_empty() { return mcp_error("Error: command is required."); }
            if let Err(reason) = check_command_sandbox(&command, &sandbox, state) {
                return mcp_error(&reason);
            }
            match Command::new("bash").arg("-lc").arg(&command)
                .current_dir(&sandbox).output()
            {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    let text = if stderr.is_empty() { stdout } else { format!("{}\n{}", stdout, stderr) };
                    mcp_ok(if text.is_empty() { "[no output]" } else { &text })
                }
                Err(e) => mcp_error(&format!("Error: {}", e)),
            }
        }
        "read_file" => {
            let file_path = args["path"].as_str().unwrap_or("");
            if file_path.is_empty() { return mcp_error("Error: path is required."); }
            let abs = sandbox.join(file_path);
            if !abs.starts_with(&sandbox) {
                return mcp_error("Error: access outside workspace is not allowed.");
            }
            match fs::read_to_string(&abs) {
                Ok(data) => {
                    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
                    let limit  = args["limit"].as_u64().map(|l| l as usize);
                    // Slice by chars to avoid panicking on multi-byte UTF-8 boundaries.
                    let slice: String = data.chars()
                        .skip(offset)
                        .take(limit.unwrap_or(usize::MAX))
                        .collect();
                    mcp_ok(&slice)
                }
                Err(e) => mcp_error(&format!("Error reading file: {}", e)),
            }
        }
        "write_file" => {
            let file_path = args["path"].as_str().unwrap_or("");
            let content = args["content"].as_str().unwrap_or("");
            if file_path.is_empty() { return mcp_error("Error: path is required."); }
            let abs = sandbox.join(file_path);
            if !abs.starts_with(&sandbox) {
                return mcp_error("Error: access outside workspace is not allowed.");
            }
            if let Some(parent) = abs.parent() { let _ = fs::create_dir_all(parent); }
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
            if file_path.is_empty() { return mcp_error("Error: path is required."); }
            let abs = sandbox.join(file_path);
            if !abs.starts_with(&sandbox) {
                return mcp_error("Error: access outside workspace is not allowed.");
            }
            match fs::read_to_string(&abs) {
                Ok(data) => {
                    let result = if replace_all { data.replace(old_str, new_str) }
                                 else { data.replacen(old_str, new_str, 1) };
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
            if pattern.is_empty() { return mcp_error("Error: pattern is required."); }
            let base = args["path"].as_str()
                .map(|p| sandbox.join(p))
                .unwrap_or_else(|| sandbox.clone());
            let full_pattern = base.join(pattern).to_string_lossy().to_string();
            match glob::glob(&full_pattern) {
                Ok(paths) => {
                    let results: Vec<String> = paths
                        .filter_map(|r| r.ok())
                        .filter(|p| p.starts_with(&sandbox))
                        .map(|p| p.to_string_lossy().to_string())
                        .collect();
                    if results.is_empty() {
                        mcp_ok("No files matched.")
                    } else {
                        mcp_ok(&results.join("\n"))
                    }
                }
                Err(e) => mcp_error(&format!("Invalid glob pattern: {}", e)),
            }
        }
        "grep_search" => {
            let pattern = args["pattern"].as_str().unwrap_or("");
            if pattern.is_empty() { return mcp_error("Error: pattern is required."); }
            let search_path = args["path"].as_str()
                .map(|p| sandbox.join(p))
                .unwrap_or_else(|| sandbox.clone());
            let case_flag = args["-i"].as_bool().unwrap_or(false);
            let mut cmd = Command::new("rg");
            cmd.arg("--no-heading").arg("-n");
            if case_flag { cmd.arg("-i"); }
            if let Some(glob_pat) = args["glob"].as_str() {
                cmd.arg("--glob").arg(glob_pat);
            }
            cmd.arg(pattern).arg(&search_path).current_dir(&sandbox);
            match cmd.output().or_else(|_| {
                let mut fallback = Command::new("grep");
                fallback.arg("-r").arg("-n");
                if case_flag { fallback.arg("-i"); }
                fallback.arg(pattern).arg(&search_path).current_dir(&sandbox);
                fallback.output()
            }) {
                Ok(out) => {
                    let text = String::from_utf8_lossy(&out.stdout).to_string();
                    mcp_ok(if text.trim().is_empty() { "No matches found." } else { text.trim() })
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
                    if query.is_empty() { return mcp_error("Error: query is required for search."); }
                    let results: Vec<&MemoryEntry> = pool.search(query);
                    let j = serde_json::to_string_pretty(&results).unwrap_or_default();
                    mcp_ok(&format!("Found {} entries:\n{}", results.len(), j))
                }
                "list" => {
                    let limit = args["limit"].as_u64().unwrap_or(50) as usize;
                    let entries: Vec<&MemoryEntry> = pool.entries.iter().rev().take(limit).collect();
                    let j = serde_json::to_string_pretty(&entries).unwrap_or_default();
                    mcp_ok(&format!("{} entries (showing last {}):\n{}", pool.entries.len(), entries.len(), j))
                }
                "count" => mcp_ok(&format!("Memory pool contains {} entries.", pool.entries.len())),
                _ => mcp_error(&format!("Unknown action '{}'. Use: list, search, count.", action)),
            }
        }
        // ── Manager tools ─────────────────────────────────────────
        "spawn_agent" => {
            let role     = args["role"].as_str().unwrap_or("worker");
            let sys_prompt = args["system_prompt"].as_str().unwrap_or("");
            let model    = args["model"].as_str().map(|s| s.to_string());
            let ctx_limit = args["context_limit"].as_u64();

            let ts = chrono::Utc::now().timestamp_millis();
            let agent_id = format!("agent-{}-{}", role, ts);

            let new_agent = Agent {
                id: agent_id.clone(),
                name: role.to_string(),
                agent_type: "model".to_string(),
                model_key: model,
                model_type: Some("llm".to_string()),
                role: Some(sys_prompt.to_string()),
                load_config: ctx_limit.map(|cl| AgentLoadConfig {
                    context_length: Some(cl),
                    eval_batch_size: None, flash_attention: None,
                    num_experts: None, offload_kv_cache_to_gpu: None,
                }),
                mode: Some("stay_awake".to_string()),
                armed: true,
                is_manager: false,
                paused: false,
            };

            state.agent_config.lock().unwrap().agents.push(new_agent);
            mcp_ok(&format!(r#"{{"agent_id": "{}"}}"#, agent_id))
        }
        "send_message" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            let content  = args["content"].as_str().unwrap_or("");
            if agent_id.is_empty() { return mcp_error("Error: agent_id is required."); }
            if content.is_empty()  { return mcp_error("Error: content is required."); }
            let await_reply = args["await_reply"].as_bool().unwrap_or(true);

            let (model_key, sys_prompt, agent_name) = {
                let config = state.agent_config.lock().unwrap();
                match config.agents.iter().find(|a| a.id == agent_id) {
                    None => return mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                    Some(a) => {
                        if !a.armed  { return mcp_error(&format!("Error: agent '{}' is disarmed.", agent_id)); }
                        if a.paused  { return mcp_error(&format!("Error: agent '{}' is paused.", agent_id)); }
                        (
                            a.model_key.clone().unwrap_or_default(),
                            a.role.clone().unwrap_or_default(),
                            a.name.clone(),
                        )
                    }
                }
            };

            if model_key.is_empty() { return mcp_error(&format!("Error: agent '{}' has no model.", agent_id)); }

            if !await_reply {
                state.memory_pool.lock().unwrap().push(agent_id, &agent_name, "user", content);
                return mcp_ok(&format!(r#"{{"status":"queued","agent_id":"{}"}}"#, agent_id));
            }

            // Reconstruct history for this sub-agent BEFORE adding the current message
            let history: Vec<Value> = {
                let pool = state.memory_pool.lock().unwrap();
                pool.entries.iter()
                    .filter(|e| e.agent_id == agent_id && (e.role == "user" || e.role == "assistant"))
                    .map(|e| json!({"role": e.role, "content": e.content}))
                    .collect()
            };

            // Emit AgentStart so frontend creates a bubble for this sub-agent
            let maybe_ch = state.event_channel.lock().unwrap().clone();
            if let Some(ref ch) = maybe_ch {
                let _ = ch.send(StreamEvent::AgentStart {
                    agent_id: agent_id.to_string(),
                    agent_name: agent_name.clone(),
                });
            }

            match call_chat_blocking(&model_key, &sys_prompt, content, &history).await {
                Ok(response) => {
                    // Store both sides of this turn in memory
                    {
                        let mut pool = state.memory_pool.lock().unwrap();
                        pool.push(agent_id, &agent_name, "user", content);
                        pool.push(agent_id, &agent_name, "assistant", &response);
                    }
                    // Emit the sub-agent's response as a visible bubble token
                    if let Some(ref ch) = maybe_ch {
                        let _ = ch.send(StreamEvent::Token {
                            agent_id: agent_id.to_string(),
                            content: response.clone(),
                        });
                        let _ = ch.send(StreamEvent::AgentEnd {
                            agent_id: agent_id.to_string(),
                        });
                    }
                    let payload = json!({ "agent_id": agent_id, "response": response });
                    mcp_ok(&serde_json::to_string(&payload).unwrap_or_default())
                }
                Err(e) => {
                    if let Some(ref ch) = maybe_ch {
                        let _ = ch.send(StreamEvent::Error {
                            agent_id: agent_id.to_string(),
                            agent_name: agent_name.clone(),
                            message: e.clone(),
                        });
                    }
                    mcp_error(&format!("Error calling agent: {}", e))
                }
            }
        }
        "read_agent_messages" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() { return mcp_error("Error: agent_id is required."); }

            let roles_filter: Option<Vec<String>> = args["roles"].as_array().map(|arr| {
                arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
            });
            let limit  = args["limit"].as_u64().map(|l| l as usize);
            let offset = args["offset"].as_u64().unwrap_or(0) as usize;

            let pool = state.memory_pool.lock().unwrap();
            let filtered: Vec<&MemoryEntry> = pool.entries.iter()
                .filter(|e| e.agent_id == agent_id)
                .filter(|e| roles_filter.as_ref()
                    .map(|rf| rf.iter().any(|r| r == &e.role))
                    .unwrap_or(true))
                .collect();

            let total = filtered.len();
            let page: Vec<&MemoryEntry> = filtered.into_iter()
                .skip(offset)
                .take(limit.unwrap_or(usize::MAX))
                .collect();

            let j = serde_json::to_string_pretty(&page).unwrap_or_default();
            mcp_ok(&format!("Agent '{}' — {} total messages, showing {}:\n{}", agent_id, total, page.len(), j))
        }
        "list_agents" => {
            let status_filter: Option<Vec<String>> = args["status_filter"].as_array().map(|arr| {
                arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
            });
            let role_filter = args["role_filter"].as_str();

            let config = state.agent_config.lock().unwrap();
            let pool   = state.memory_pool.lock().unwrap();

            let agents: Vec<Value> = config.agents.iter()
                .filter(|a| a.agent_type != "user")
                .filter(|a| !a.is_manager) // exclude self — manager must not call itself via send_message
                .filter(|a| role_filter.map(|r| a.name.contains(r)
                    || a.role.as_deref().unwrap_or("").contains(r)).unwrap_or(true))
                .filter_map(|a| {
                    let status = if !a.armed { "done" } else if a.paused { "paused" } else { "idle" };
                    if status_filter.as_ref().map(|sf| sf.iter().any(|s| s == status)).unwrap_or(false)
                        && status_filter.is_some()
                        && !status_filter.as_ref().unwrap().iter().any(|s| s == status) {
                        return None;
                    }
                    let msg_count = pool.entries.iter().filter(|e| e.agent_id == a.id).count();
                    let last_preview = pool.entries.iter().rev()
                        .find(|e| e.agent_id == a.id && e.role == "assistant")
                        .map(|e| e.content.chars().take(120).collect::<String>());
                    Some(json!({
                        "agent_id": a.id,
                        "name": a.name,
                        "model": a.model_key,
                        "status": status,
                        "is_manager": a.is_manager,
                        "message_count": msg_count,
                        "last_output_preview": last_preview,
                    }))
                })
                .collect();

            mcp_ok(&format!("{} agents:\n{}", agents.len(),
                serde_json::to_string_pretty(&agents).unwrap_or_default()))
        }
        "get_agent_state" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() { return mcp_error("Error: agent_id is required."); }

            let config = state.agent_config.lock().unwrap();
            match config.agents.iter().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => {
                    let pool = state.memory_pool.lock().unwrap();
                    let messages: Vec<&MemoryEntry> = pool.entries.iter()
                        .filter(|e| e.agent_id == agent_id).collect();
                    let last_output = messages.iter().rev()
                        .find(|e| e.role == "assistant").map(|e| e.content.clone());
                    let status = if !a.armed { "done" } else if a.paused { "paused" } else { "idle" };
                    let result = json!({
                        "agent_id": a.id, "name": a.name, "model": a.model_key,
                        "status": status, "is_manager": a.is_manager,
                        "paused": a.paused, "armed": a.armed,
                        "message_count": messages.len(), "last_output": last_output,
                    });
                    mcp_ok(&serde_json::to_string_pretty(&result).unwrap_or_default())
                }
            }
        }
        "kill_agent" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() { return mcp_error("Error: agent_id is required."); }
            let reason = args["reason"].as_str().unwrap_or("terminated by manager");
            let mut config = state.agent_config.lock().unwrap();
            match config.agents.iter_mut().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => { a.armed = false; mcp_ok(&format!("Agent '{}' terminated. Reason: {}", agent_id, reason)) }
            }
        }
        "pause_agent" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() { return mcp_error("Error: agent_id is required."); }
            let mut config = state.agent_config.lock().unwrap();
            match config.agents.iter_mut().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => { a.paused = true; mcp_ok(&format!("Agent '{}' paused.", agent_id)) }
            }
        }
        "resume_agent" => {
            let agent_id = args["agent_id"].as_str().unwrap_or("");
            if agent_id.is_empty() { return mcp_error("Error: agent_id is required."); }
            let mut config = state.agent_config.lock().unwrap();
            match config.agents.iter_mut().find(|a| a.id == agent_id) {
                None => mcp_error(&format!("Error: agent '{}' not found.", agent_id)),
                Some(a) => { a.paused = false; mcp_ok(&format!("Agent '{}' resumed.", agent_id)) }
            }
        }
        "broadcast_message" => {
            let content = args["content"].as_str().unwrap_or("");
            if content.is_empty() { return mcp_error("Error: content is required."); }
            let await_reply = args["await_reply"].as_bool().unwrap_or(false);

            let raw_ids: Vec<String> = match args["agent_ids"].as_array() {
                None => return mcp_error("Error: agent_ids is required."),
                Some(arr) => arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
            };

            // Resolve targets (extract data while holding lock, then release)
            let targets: Vec<(String, String, String)> = {
                let config = state.agent_config.lock().unwrap();
                let ids: Vec<String> = if raw_ids.len() == 1 && raw_ids[0] == "*" {
                    config.agents.iter()
                        .filter(|a| a.agent_type != "user" && !a.is_manager && a.armed && !a.paused)
                        .map(|a| a.id.clone()).collect()
                } else { raw_ids };
                ids.into_iter().filter_map(|id| {
                    config.agents.iter().find(|a| a.id == id && a.armed && !a.paused)
                        .and_then(|a| a.model_key.as_ref().map(|m| (
                            id, m.clone(), a.role.clone().unwrap_or_default()
                        )))
                }).collect()
            };

            if !await_reply {
                let mut pool = state.memory_pool.lock().unwrap();
                for (id, _, _) in &targets { pool.push(id, id, "user", content); }
                let ids: Vec<&str> = targets.iter().map(|(id, _, _)| id.as_str()).collect();
                return mcp_ok(&format!(r#"{{"queued":{},"agent_ids":{}}}"#,
                    targets.len(), serde_json::to_string(&ids).unwrap_or_default()));
            }

            let mut results = serde_json::Map::new();
            for (agent_id, model_key, sys_prompt) in &targets {
                let history: Vec<Value> = {
                    let pool = state.memory_pool.lock().unwrap();
                    pool.entries.iter()
                        .filter(|e| e.agent_id == *agent_id && (e.role == "user" || e.role == "assistant"))
                        .map(|e| json!({"role": e.role, "content": e.content}))
                        .collect()
                };
                state.memory_pool.lock().unwrap().push(agent_id, agent_id, "user", content);
                match call_chat_blocking(model_key, sys_prompt, content, &history).await {
                    Ok(resp) => {
                        state.memory_pool.lock().unwrap().push(agent_id, agent_id, "assistant", &resp);
                        results.insert(agent_id.clone(), json!({"status":"ok","response":resp}));
                    }
                    Err(e) => { results.insert(agent_id.clone(), json!({"status":"error","error":e})); }
                }
            }
            mcp_ok(&serde_json::to_string_pretty(&results).unwrap_or_default())
        }
        "fork_agent" => {
            let source_id = args["source_agent_id"].as_str().unwrap_or("");
            if source_id.is_empty() { return mcp_error("Error: source_agent_id is required."); }
            let new_role   = args["role"].as_str();
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
                cloned.id     = new_id.clone();
                cloned.name   = new_role.unwrap_or(&format!("{}-fork", source.name)).to_string();
                cloned.paused = false;
                cloned.armed  = true;
                if let Some(sp) = sp_override { cloned.role = Some(sp.to_string()); }

                let pool = state.memory_pool.lock().unwrap();
                let src_entries: Vec<&MemoryEntry> = pool.entries.iter()
                    .filter(|e| e.agent_id == source_id).collect();
                let limit = truncate_at.unwrap_or(src_entries.len());
                let copies: Vec<MemoryEntry> = src_entries.into_iter().take(limit).map(|e| {
                    let mut c = e.clone(); c.agent_id = new_id.clone(); c
                }).collect();
                (cloned, copies)
            };

            state.agent_config.lock().unwrap().agents.push(new_agent);
            state.memory_pool.lock().unwrap().entries.extend(entries_to_copy);
            mcp_ok(&format!(r#"{{"agent_id":"{}","forked_from":"{}"}}"#, new_id, source_id))
        }
        "aggregate_results" => {
            let raw_ids: Vec<String> = match args["agent_ids"].as_array() {
                None => return mcp_error("Error: agent_ids is required."),
                Some(arr) => arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
            };
            let fmt             = args["format"].as_str().unwrap_or("structured");
            let synthesis_prompt = args["synthesis_prompt"].as_str();
            let synthesis_model  = args["synthesis_model"].as_str();

            let target_ids: Vec<String> = if raw_ids.len() == 1 && raw_ids[0] == "*" {
                state.agent_config.lock().unwrap().agents.iter()
                    .filter(|a| a.agent_type != "user").map(|a| a.id.clone()).collect()
            } else { raw_ids };

            let mut collected: serde_json::Map<String, Value> = serde_json::Map::new();
            {
                let pool = state.memory_pool.lock().unwrap();
                for id in &target_ids {
                    let last = pool.entries.iter().rev()
                        .find(|e| e.agent_id == *id && e.role == "assistant")
                        .map(|e| e.content.clone())
                        .unwrap_or_else(|| "[no output]".to_string());
                    collected.insert(id.clone(), Value::String(last));
                }
            }

            if let Some(sp) = synthesis_prompt {
                let combined = collected.iter()
                    .map(|(id, v)| format!("Agent {}:\n{}", id, v.as_str().unwrap_or("")))
                    .collect::<Vec<_>>().join("\n\n---\n\n");
                let full_prompt = format!("{}\n\nAgent outputs:\n{}", sp, combined);
                let model_to_use = synthesis_model.map(|s| s.to_string()).unwrap_or_else(|| {
                    state.agent_config.lock().unwrap().agents.iter()
                        .find(|a| a.agent_type != "user" && a.model_key.is_some())
                        .and_then(|a| a.model_key.clone()).unwrap_or_default()
                });
                if !model_to_use.is_empty() {
                    return match call_chat_blocking(&model_to_use, "", &full_prompt, &[]).await {
                        Ok(synthesis) => mcp_ok(&synthesis),
                        Err(e) => mcp_error(&format!("Synthesis failed: {}", e)),
                    };
                }
            }

            match fmt {
                "raw" => mcp_ok(&collected.values()
                    .filter_map(|v| v.as_str()).collect::<Vec<_>>().join("\n\n---\n\n")),
                "summary" => mcp_ok(&collected.iter()
                    .map(|(id, v)| format!("**{}**: {}", id, v.as_str().unwrap_or("")))
                    .collect::<Vec<_>>().join("\n\n")),
                _ => mcp_ok(&serde_json::to_string_pretty(&collected).unwrap_or_default()),
            }
        }
        "pipe_message" => {
            let to_id = args["to_agent_id"].as_str().unwrap_or("");
            let raw   = args["content"].as_str().unwrap_or("");
            if to_id.is_empty() { return mcp_error("Error: to_agent_id is required."); }
            if raw.is_empty()   { return mcp_error("Error: content is required."); }

            // Build the final message with optional prefix / suffix
            let mut content = String::new();
            if let Some(pre) = args["prefix"].as_str() {
                if !pre.is_empty() { content.push_str(pre); content.push('\n'); }
            }
            content.push_str(raw);
            if let Some(suf) = args["suffix"].as_str() {
                if !suf.is_empty() { content.push('\n'); content.push_str(suf); }
            }

            // Resolve target agent
            let (model_key, sys_prompt, agent_name) = {
                let config = state.agent_config.lock().unwrap();
                match config.agents.iter().find(|a| a.id == to_id) {
                    None => return mcp_error(&format!("Error: agent '{}' not found.", to_id)),
                    Some(a) => {
                        if !a.armed { return mcp_error(&format!("Error: agent '{}' is disarmed.", to_id)); }
                        if a.paused { return mcp_error(&format!("Error: agent '{}' is paused.", to_id)); }
                        (
                            a.model_key.clone().unwrap_or_default(),
                            a.role.clone().unwrap_or_default(),
                            a.name.clone(),
                        )
                    }
                }
            };
            if model_key.is_empty() {
                return mcp_error(&format!("Error: agent '{}' has no model configured.", to_id));
            }

            // Per-agent conversation history
            let history: Vec<Value> = {
                let pool = state.memory_pool.lock().unwrap();
                pool.entries.iter()
                    .filter(|e| e.agent_id == to_id && (e.role == "user" || e.role == "assistant"))
                    .map(|e| json!({"role": e.role, "content": e.content}))
                    .collect()
            };

            let maybe_ch = state.event_channel.lock().unwrap().clone();
            if let Some(ref ch) = maybe_ch {
                let _ = ch.send(StreamEvent::AgentStart {
                    agent_id: to_id.to_string(),
                    agent_name: agent_name.clone(),
                });
            }

            match call_chat_blocking(&model_key, &sys_prompt, &content, &history).await {
                Ok(response) => {
                    {
                        let mut pool = state.memory_pool.lock().unwrap();
                        pool.push(to_id, &agent_name, "user", &content);
                        pool.push(to_id, &agent_name, "assistant", &response);
                    }
                    if let Some(ref ch) = maybe_ch {
                        let _ = ch.send(StreamEvent::Token {
                            agent_id: to_id.to_string(),
                            content: response.clone(),
                        });
                        let _ = ch.send(StreamEvent::AgentEnd {
                            agent_id: to_id.to_string(),
                        });
                    }
                    let payload = json!({ "to_agent_id": to_id, "response": response });
                    mcp_ok(&serde_json::to_string(&payload).unwrap_or_default())
                }
                Err(e) => {
                    if let Some(ref ch) = maybe_ch {
                        let _ = ch.send(StreamEvent::Error {
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
            if url.is_empty() { return mcp_error("Error: url is required."); }
            let client = reqwest::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; Ajantis/1.0)")
                .timeout(std::time::Duration::from_secs(30))
                .build().unwrap_or_default();
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
                    } else { text };
                    mcp_ok(&format!("URL: {}\n\n{}", url, out))
                }
                Err(e) => mcp_error(&format!("Fetch failed: {}", e)),
            }
        }
        "WebSearch" => {
            let query = args["query"].as_str().unwrap_or("");
            if query.is_empty() { return mcp_error("Error: query is required."); }
            let client = reqwest::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; Ajantis/1.0)")
                .timeout(std::time::Duration::from_secs(15))
                .build().unwrap_or_default();
            match client.get("https://api.duckduckgo.com/")
                .query(&[("q", query), ("format", "json"), ("no_html", "1"), ("skip_disambig", "1")])
                .send().await
            {
                Err(e) => mcp_error(&format!("Search failed: {}", e)),
                Ok(resp) => {
                    let data: Value = resp.json().await.unwrap_or(json!({}));
                    let mut lines: Vec<String> = Vec::new();
                    if let Some(abs) = data["AbstractText"].as_str() {
                        if !abs.is_empty() {
                            lines.push(format!("**Summary**: {}", abs));
                            if let Some(src) = data["AbstractURL"].as_str() {
                                if !src.is_empty() { lines.push(format!("Source: {}", src)); }
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
            if code.is_empty() { return mcp_error("Error: code is required."); }
            if let Err(reason) = check_command_sandbox(code, &sandbox, state) {
                return mcp_error(&reason);
            }
            let interpreter = match lang.as_str() {
                "python" | "python3" | "py" => "python3",
                "javascript" | "js" | "node" => "node",
                "ruby" | "rb"               => "ruby",
                "perl"                      => "perl",
                "lua"                       => "lua",
                "bash" | "sh" | "shell"     => "bash",
                _ => return mcp_error(&format!("Unsupported language: {}", lang)),
            };
            let flag = if interpreter == "bash" { "-c" } else { "-e" };
            match Command::new(interpreter).arg(flag).arg(code)
                .current_dir(&sandbox).output()
            {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    if !out.status.success() && !stderr.is_empty() {
                        mcp_error(&format!("Runtime error:\n{}", stderr))
                    } else {
                        let text = if stderr.is_empty() { stdout }
                                   else { format!("{}\n{}", stdout, stderr) };
                        mcp_ok(if text.is_empty() { "[no output]" } else { &text })
                    }
                }
                Err(e) => mcp_error(&format!("Failed to run '{}': {}", interpreter, e)),
            }
        }
        "PowerShell" => {
            let command = args["command"].as_str().unwrap_or("");
            if command.is_empty() { return mcp_error("Error: command is required."); }
            if let Err(reason) = check_command_sandbox(command, &sandbox, state) {
                return mcp_error(&reason);
            }
            match Command::new("pwsh").arg("-Command").arg(command)
                .current_dir(&sandbox).output()
            {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    let text = if stderr.is_empty() { stdout }
                               else { format!("{}\n{}", stdout, stderr) };
                    mcp_ok(if text.is_empty() { "[no output]" } else { &text })
                }
                Err(_) => mcp_error("PowerShell (pwsh) is not available on this system."),
            }
        }
        "Sleep" => {
            let ms = args["duration_ms"].as_u64()
                .or_else(|| args["seconds"].as_f64().map(|s| (s * 1000.0) as u64))
                .unwrap_or(1000)
                .min(30_000); // cap at 30 s
            tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
            mcp_ok(&format!("Slept {}ms.", ms))
        }

        // ── UI / user interaction ─────────────────────────────────
        "SendUserMessage" => {
            let message = args["message"].as_str().unwrap_or("");
            if message.is_empty() { return mcp_error("Error: message is required."); }
            let maybe_ch = state.event_channel.lock().unwrap().clone();
            if let Some(ref ch) = maybe_ch {
                let _ = ch.send(StreamEvent::AgentStart {
                    agent_id: "agent-notification".to_string(),
                    agent_name: "Notification".to_string(),
                });
                let _ = ch.send(StreamEvent::Token {
                    agent_id: "agent-notification".to_string(),
                    content: message.to_string(),
                });
                let _ = ch.send(StreamEvent::AgentEnd {
                    agent_id: "agent-notification".to_string(),
                });
                mcp_ok("Message delivered to user.")
            } else {
                mcp_error("No active session; message could not be delivered.")
            }
        }
        "AskUserQuestion" => {
            // Synchronous Q&A is not feasible in the current async pipeline.
            // Emit the question as a notification so the user at least sees it.
            let question = args["question"].as_str().unwrap_or("");
            let maybe_ch = state.event_channel.lock().unwrap().clone();
            if let Some(ref ch) = maybe_ch {
                let _ = ch.send(StreamEvent::AgentStart {
                    agent_id: "agent-notification".to_string(),
                    agent_name: "Question".to_string(),
                });
                let _ = ch.send(StreamEvent::Token {
                    agent_id: "agent-notification".to_string(),
                    content: format!("❓ {}", question),
                });
                let _ = ch.send(StreamEvent::AgentEnd {
                    agent_id: "agent-notification".to_string(),
                });
            }
            mcp_ok("[Question displayed to user. Synchronous responses are not supported — the user's next chat message will serve as the answer.]")
        }

        // ── Output / tools ────────────────────────────────────────
        "StructuredOutput" => {
            let output = args.get("output").unwrap_or(args);
            mcp_ok(&serde_json::to_string_pretty(output).unwrap_or_default())
        }
        "ToolSearch" => {
            let query = args["query"].as_str().unwrap_or("").to_lowercase();
            let max   = args["max_results"].as_u64().unwrap_or(5) as usize;
            if query.is_empty() { return mcp_error("Error: query is required."); }
            let hits: Vec<Value> = state.tools.iter()
                .filter(|t| t.name.to_lowercase().contains(&query)
                    || t.description.to_lowercase().contains(&query))
                .take(max)
                .map(|t| json!({ "name": t.name, "description": &t.description[..t.description.len().min(100)] }))
                .collect();
            if hits.is_empty() {
                mcp_ok(&format!("No tools matched '{}'.", query))
            } else {
                mcp_ok(&format!("{} tool(s) found:\n{}", hits.len(),
                    serde_json::to_string_pretty(&hits).unwrap_or_default()))
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
                    None    => mcp_error(&format!("Config key '{}' not found.", key)),
                }
            }
        }

        // ── Background tasks ──────────────────────────────────────
        "TaskCreate" => {
            let command     = args["command"].as_str().unwrap_or("");
            let description = args["description"].as_str().unwrap_or("background task");
            if command.is_empty() { return mcp_error("Error: command is required."); }
            create_task_internal(command, description, &sandbox, state).await
        }
        "RunTaskPacket" => {
            let command = args["command"].as_str()
                .or_else(|| args["script"].as_str())
                .unwrap_or("");
            let description = args["description"].as_str().unwrap_or("task packet");
            if command.is_empty() { return mcp_error("Error: command or script is required."); }
            create_task_internal(command, description, &sandbox, state).await
        }
        "TaskGet" => {
            let id = args["task_id"].as_str().unwrap_or("");
            if id.is_empty() { return mcp_error("Error: task_id is required."); }
            let tasks = state.tasks.lock().unwrap();
            match tasks.get(id) {
                None => mcp_error(&format!("Task '{}' not found.", id)),
                Some(t) => {
                    let status = t.status.lock().unwrap().clone();
                    let preview = t.output.lock().unwrap().chars().take(300).collect::<String>();
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
            let list: Vec<Value> = tasks.values().map(|t| json!({
                "task_id": t.id,
                "description": t.description,
                "status": t.status.lock().unwrap().clone(),
                "started_at": t.started_at,
            })).collect();
            mcp_ok(&format!("{} task(s):\n{}", list.len(),
                serde_json::to_string_pretty(&list).unwrap_or_default()))
        }
        "TaskStop" => {
            let id = args["task_id"].as_str().unwrap_or("");
            if id.is_empty() { return mcp_error("Error: task_id is required."); }
            let mut tasks = state.tasks.lock().unwrap();
            match tasks.get_mut(id) {
                None => mcp_error(&format!("Task '{}' not found.", id)),
                Some(t) => {
                    if let Some(ref h) = t.abort_handle { h.abort(); }
                    *t.status.lock().unwrap() = "stopped".to_string();
                    mcp_ok(&format!("Task '{}' stopped.", id))
                }
            }
        }
        "TaskOutput" => {
            let id = args["task_id"].as_str().unwrap_or("");
            if id.is_empty() { return mcp_error("Error: task_id is required."); }
            let tasks = state.tasks.lock().unwrap();
            match tasks.get(id) {
                None => mcp_error(&format!("Task '{}' not found.", id)),
                Some(t) => {
                    let status = t.status.lock().unwrap().clone();
                    let out    = t.output.lock().unwrap().clone();
                    mcp_ok(&format!("[Task {} — {}]\n{}",
                        id, status,
                        if out.is_empty() { "[no output yet]".to_string() } else { out }))
                }
            }
        }
        "TaskUpdate" => {
            // Process-based tasks don't support runtime stdin injection.
            mcp_ok("Acknowledged — process-based tasks do not support runtime message injection.")
        }

        // ── Stubs for Claude Code-specific tools ──────────────────
        "Agent" => mcp_ok("Use 'spawn_agent' + 'send_message' to manage sub-agents in Ajantis."),
        "Skill" => mcp_error("Skill loading is a Claude Code feature, not available here."),
        "NotebookEdit" => mcp_error("Notebook editing requires a Jupyter kernel."),
        "EnterPlanMode" | "ExitPlanMode" => mcp_ok("Plan mode is a Claude Code feature; no-op here."),
        "ListMcpResources" => mcp_ok(r#"{"resources":[]}"#),
        "ReadMcpResource" => mcp_error("No resources are exposed on this MCP server."),
        "McpAuth" => mcp_error("This MCP server does not require authentication."),
        "RemoteTrigger" => mcp_error("Remote triggers are not implemented."),
        "MCP" => mcp_error("Nested MCP execution is not supported."),
        "TestingPermission" => mcp_ok(r#"{"granted":true}"#),
        "WorkerCreate" | "WorkerGet" | "WorkerObserve" | "WorkerResolveTrust"
        | "WorkerAwaitReady" | "WorkerSendPrompt" | "WorkerRestart"
        | "WorkerTerminate" | "WorkerObserveCompletion" => {
            mcp_error("Worker management is a Claude Code feature. Use agent tools instead.")
        }
        "TeamCreate" | "TeamDelete" => {
            mcp_ok("Use 'spawn_agent' + 'broadcast_message' to manage agent groups.")
        }
        "CronCreate" | "CronList" | "CronDelete" => {
            mcp_error("Scheduled tasks are not yet implemented. Use TaskCreate for one-shot background tasks.")
        }
        "LSP" => mcp_error("LSP queries are not available in this context."),

        _ => mcp_error(&format!("Tool '{}' is not recognised by this MCP server.", name)),
    }
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

    // Reject absolute paths that are NOT under the active sandbox.
    let ws = sandbox.to_string_lossy();
    for token in command.split_whitespace() {
        if token.starts_with('/') && !token.starts_with(ws.as_ref()) {
            return Err(format!(
                "Blocked: absolute path '{}' is outside the workspace '{}'.",
                token, ws
            ));
        }
    }

    // ── 2. Check CommandPolicy from AgentConfig ───────────────────
    let policy = state.agent_config.lock().unwrap().command_policy.clone();

    let cmd_lower = command.trim().to_lowercase();

    // Denylist has highest priority.
    for denied in &policy.denylist {
        if cmd_lower.starts_with(&denied.to_lowercase()) {
            return Err(format!("Blocked by denylist: command matches '{}'.", denied));
        }
    }

    // Allowlist: if non-empty the command must match at least one entry.
    if !policy.allowlist.is_empty() {
        let allowed = policy.allowlist.iter()
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
    let output  = Arc::new(Mutex::new(String::new()));
    let status  = Arc::new(Mutex::new("running".to_string()));
    let out_c   = output.clone();
    let stat_c  = status.clone();
    let cmd_str = command.to_string();
    let ws      = sandbox.clone();

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
                *out_c.lock().unwrap() =
                    format!("{}{}", stdout, if stderr.is_empty() { String::new() } else { format!("\nstderr:\n{}", stderr) });
                *stat_c.lock().unwrap() = if o.status.success() {
                    "completed".to_string()
                } else {
                    format!("failed (exit {:?})", o.status.code())
                };
            }
            Err(e) => { *stat_c.lock().unwrap() = format!("failed: {}", e); }
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
        .route("/list_tools", get({
            let tools = mcp_state.tools.clone();
            move || async move { Json(json!({ "tools": tools })) }
        }))
        .layer(cors)
        .with_state(mcp_state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await.expect("Failed to bind MCP server port");
    log::info!("MCP server listening on http://localhost:{}", port);
    axum::serve(listener, app).await.ok();
}

/// Load tools from the `tools.json` embedded at compile time.
/// The path is relative to `src-tauri/` (where Cargo.toml lives).
pub(crate) fn load_tools_embedded() -> Vec<McpTool> {
    const TOOLS_JSON: &str = include_str!("../../tools.json");
    let parsed: serde_json::Value = serde_json::from_str(TOOLS_JSON)
        .expect("tools.json is not valid JSON");
    parsed.get("tools").and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|entry| {
            let func = entry.get("function")?;
            Some(McpTool {
                name: func["name"].as_str().unwrap_or("").to_string(),
                description: func["description"].as_str().unwrap_or("").to_string(),
                input_schema: func.get("parameters").cloned()
                    .unwrap_or(serde_json::json!({"type": "object"})),
            })
        }).collect())
        .unwrap_or_default()
}

pub(crate) fn load_tools(tools_path: &PathBuf) -> Vec<McpTool> {
    match fs::read_to_string(tools_path) {
        Ok(content) => {
            let parsed: Value = serde_json::from_str(&content).unwrap_or(Value::Null);
            parsed.get("tools").and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|entry| {
                    let func = entry.get("function")?;
                    Some(McpTool {
                        name: func["name"].as_str().unwrap_or("").to_string(),
                        description: func["description"].as_str().unwrap_or("").to_string(),
                        input_schema: func.get("parameters").cloned()
                            .unwrap_or(json!({"type": "object"})),
                    })
                }).collect())
                .unwrap_or_default()
        }
        Err(e) => { log::error!("Failed to load tools.json: {}", e); vec![] }
    }
}
