use axum::{
    extract::State as AxumState,
    http::{HeaderValue, Method},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use tauri::ipc::Channel;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

// ── Helpers ────────────────────────────────────────────────────────

fn lm_base_url() -> String {
    let url = std::env::var("LM_STUDIO_URL")
        .unwrap_or_else(|_| "http://localhost:1234/api/v1/chat".to_string());
    url.trim_end_matches('/')
        .trim_end_matches("/chat")
        .trim_end_matches("/v1")
        .trim_end_matches("/api")
        .to_string()
}

fn default_true() -> bool { true }
fn default_priority() -> u8 { 128 }

// ── Agent config types ─────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
struct AgentLoadConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eval_batch_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flash_attention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    num_experts: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offload_kv_cache_to_gpu: Option<bool>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Agent {
    id: String,
    name: String,
    #[serde(rename = "type")]
    agent_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    load_config: Option<AgentLoadConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mode: Option<String>,
    #[serde(default = "default_true")]
    armed: bool,
    /// Enables MCP tool-call loop path (spawn_agent, send_message, etc.)
    #[serde(default)]
    is_manager: bool,
    /// Runtime pause flag — not persisted to config.
    #[serde(default, skip_serializing)]
    paused: bool,
}

/// Replaces Connection. Backward-compatible via #[serde(default)].
#[derive(Clone, Serialize, Deserialize)]
struct RoutingRule {
    from: String,
    to: String,
    /// Execution order among siblings: lower = called first. Default 128.
    #[serde(default = "default_priority")]
    priority: u8,
    /// Optional substring that must appear in the outgoing message for this route to fire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    condition: Option<String>,
    #[serde(default = "default_true")]
    enabled: bool,
}

#[derive(Clone, Serialize, Deserialize)]
struct AgentConfig {
    agents: Vec<Agent>,
    connections: Vec<RoutingRule>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            agents: vec![Agent {
                id: "user".to_string(),
                name: "User".to_string(),
                agent_type: "user".to_string(),
                model_key: None,
                model_type: None,
                role: None,
                load_config: None,
                mode: None,
                armed: true,
                is_manager: false,
                paused: false,
            }],
            connections: vec![],
        }
    }
}

// ── Workspace config ───────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
struct WorkspaceThread {
    id: String,
    name: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct Workspace {
    id: String,
    name: String,
    path: String,
    #[serde(default)]
    threads: Vec<WorkspaceThread>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct WorkspaceConfig {
    workspaces: Vec<Workspace>,
}

fn workspace_config_path(workspace: &PathBuf) -> PathBuf {
    workspace.join("workspace_config.json")
}

#[tauri::command]
async fn pick_folder() -> Result<Option<String>, String> {
    let handle = rfd::AsyncFileDialog::new()
        .set_title("Choose a workspace folder")
        .pick_folder()
        .await;
    Ok(handle.map(|f| f.path().to_string_lossy().to_string()))
}

#[tauri::command]
async fn load_workspace_config(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<WorkspaceConfig, String> {
    let path = workspace_config_path(&state.workspace_root);
    if !path.exists() { return Ok(WorkspaceConfig::default()); }
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read workspace config: {}", e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse workspace config: {}", e))
}

#[tauri::command]
async fn save_workspace_config(
    state: tauri::State<'_, Arc<AppState>>,
    config: WorkspaceConfig,
) -> Result<(), String> {
    let path = workspace_config_path(&state.workspace_root);
    let json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write workspace config: {}", e))
}

// ── Memory pool ────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
struct MemoryEntry {
    timestamp: String,
    agent_id: String,
    agent_name: String,
    role: String,
    content: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct MemoryPool {
    entries: Vec<MemoryEntry>,
}

impl MemoryPool {
    fn push(&mut self, agent_id: &str, agent_name: &str, role: &str, content: &str) {
        self.entries.push(MemoryEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            agent_id: agent_id.to_string(),
            agent_name: agent_name.to_string(),
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        let q = query.to_lowercase();
        self.entries.iter().filter(|e| {
            e.content.to_lowercase().contains(&q)
                || e.agent_name.to_lowercase().contains(&q)
                || e.role.to_lowercase().contains(&q)
        }).collect()
    }
}

// ── App state ──────────────────────────────────────────────────────

struct AppState {
    current_model: Mutex<String>,
    last_response_id: Mutex<Option<String>>,
    mcp_port: u16,
    workspace_root: PathBuf,
    mcp_tools: Vec<McpTool>,
    todo_list: Arc<Mutex<Vec<Value>>>,
    agent_config: Arc<Mutex<AgentConfig>>,
    memory_pool: Arc<Mutex<MemoryPool>>,
    /// Shared with McpState so MCP tool handlers can emit stream events to the frontend.
    event_channel: Arc<Mutex<Option<Channel<StreamEvent>>>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct McpTool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

// ── Tauri commands ─────────────────────────────────────────────────

#[derive(Serialize)]
struct SendMessageResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[tauri::command]
async fn send_message(
    state: tauri::State<'_, Arc<AppState>>,
    user_message: String,
) -> Result<SendMessageResponse, String> {
    let base = lm_base_url();
    let api_url = format!("{}/api/v1/chat", base);
    let model = state.current_model.lock().unwrap().clone();
    let mcp_port = state.mcp_port;
    let tool_names: Vec<String> = state.mcp_tools.iter().map(|t| t.name.clone()).collect();

    let mut body = json!({
        "model": model,
        "input": user_message,
        "integrations": [{
            "type": "ephemeral_mcp",
            "server_label": "local",
            "server_url": format!("http://localhost:{}", mcp_port),
            "allowed_tools": tool_names,
        }],
        "context_length": 8000,
        "store": true,
    });

    {
        let last_id = state.last_response_id.lock().unwrap();
        if let Some(ref id) = *last_id {
            body["previous_response_id"] = json!(id);
        }
    }

    let client = reqwest::Client::new();
    let mut req = client.post(&api_url).json(&body);
    if let Ok(token) = std::env::var("LM_API_TOKEN") {
        req = req.bearer_auth(token);
    }

    match req.send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Ok(SendMessageResponse {
                    ok: false, data: None,
                    error: Some(format!("LM Studio API error: {} {}", status, text)),
                });
            }
            match resp.json::<Value>().await {
                Ok(data) => {
                    if let Some(rid) = data.get("response_id").and_then(|v| v.as_str()) {
                        *state.last_response_id.lock().unwrap() = Some(rid.to_string());
                    }
                    Ok(SendMessageResponse { ok: true, data: Some(data), error: None })
                }
                Err(e) => Ok(SendMessageResponse {
                    ok: false, data: None,
                    error: Some(format!("Failed to parse response: {}", e)),
                }),
            }
        }
        Err(e) => Ok(SendMessageResponse {
            ok: false, data: None,
            error: Some(format!("Request failed: {}", e)),
        }),
    }
}

// ── Model listing ──────────────────────────────────────────────────

#[derive(Serialize)]
struct ModelInfo {
    key: String,
    display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_context_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    params_string: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trained_for_tool_use: Option<bool>,
}

#[derive(Serialize)]
struct LoadedInstance {
    instance_id: String,
    context_length: Option<u64>,
    flash_attention: Option<bool>,
}

#[derive(Serialize)]
struct LoadedModelInfo {
    key: String,
    display_name: String,
    instances: Vec<LoadedInstance>,
}

#[tauri::command]
async fn fetch_models() -> Result<Vec<ModelInfo>, String> {
    let base = lm_base_url();
    let client = reqwest::Client::new();
    let resp = client.get(format!("{}/api/v1/models", base)).send().await
        .map_err(|e| format!("Failed to fetch models: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Failed to fetch models: {}", resp.status()));
    }
    let data: Value = resp.json().await.map_err(|e| format!("Invalid response: {}", e))?;
    let models = data.get("models").and_then(|v| v.as_array()).ok_or("Invalid models response")?;
    Ok(models.iter().map(|m| {
        let capabilities = m.get("capabilities");
        ModelInfo {
            key: m["key"].as_str().unwrap_or("").to_string(),
            display_name: m.get("display_name").or_else(|| m.get("key"))
                .and_then(|v| v.as_str()).unwrap_or("").to_string(),
            model_type: m["type"].as_str().map(|s| s.to_string()),
            max_context_length: m["max_context_length"].as_u64(),
            format: m["format"].as_str().map(|s| s.to_string()),
            quantization: m.get("quantization").and_then(|q| q.get("name"))
                .and_then(|v| v.as_str()).map(|s| s.to_string()),
            params_string: m["params_string"].as_str().map(|s| s.to_string()),
            vision: capabilities.and_then(|c| c["vision"].as_bool()),
            trained_for_tool_use: capabilities.and_then(|c| c["trained_for_tool_use"].as_bool()),
        }
    }).collect())
}

#[tauri::command]
async fn fetch_loaded_models() -> Result<Vec<LoadedModelInfo>, String> {
    let base = lm_base_url();
    let client = reqwest::Client::new();
    let resp = client.get(format!("{}/api/v1/models", base)).send().await
        .map_err(|e| format!("Failed to fetch models: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Failed to fetch models: {}", resp.status()));
    }
    let data: Value = resp.json().await.map_err(|e| format!("Invalid response: {}", e))?;
    let models = data.get("models").and_then(|v| v.as_array()).ok_or("Invalid models response")?;
    Ok(models.iter().filter_map(|m| {
        let instances = m.get("loaded_instances").and_then(|v| v.as_array())?;
        if instances.is_empty() { return None; }
        Some(LoadedModelInfo {
            key: m["key"].as_str().unwrap_or("").to_string(),
            display_name: m.get("display_name").or_else(|| m.get("key"))
                .and_then(|v| v.as_str()).unwrap_or("").to_string(),
            instances: instances.iter().map(|inst| LoadedInstance {
                instance_id: inst["id"].as_str().unwrap_or("").to_string(),
                context_length: inst["context_length"].as_u64(),
                flash_attention: inst["flash_attention"].as_bool(),
            }).collect(),
        })
    }).collect())
}

// ── Model management ───────────────────────────────────────────────

#[tauri::command]
async fn set_model(state: tauri::State<'_, Arc<AppState>>, model: String) -> Result<bool, String> {
    if !model.is_empty() {
        *state.current_model.lock().unwrap() = model;
    }
    Ok(true)
}

#[derive(Deserialize)]
struct LoadConfig {
    model: String,
    #[serde(default)] context_length: Option<u64>,
    #[serde(default)] eval_batch_size: Option<u64>,
    #[serde(default)] flash_attention: Option<bool>,
    #[serde(default)] num_experts: Option<u64>,
    #[serde(default)] offload_kv_cache_to_gpu: Option<bool>,
}

async fn load_model_internal(config: &AgentLoadConfig, model_key: &str) -> Result<(), String> {
    let url = format!("{}/api/v1/models/load", lm_base_url());
    let mut body = json!({ "model": model_key });
    if let Some(v) = config.context_length    { body["context_length"] = json!(v); }
    if let Some(v) = config.eval_batch_size   { body["eval_batch_size"] = json!(v); }
    if let Some(v) = config.flash_attention   { body["flash_attention"] = json!(v); }
    if let Some(v) = config.num_experts       { body["num_experts"] = json!(v); }
    if let Some(v) = config.offload_kv_cache_to_gpu { body["offload_kv_cache_to_gpu"] = json!(v); }
    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await
        .map_err(|e| format!("Load failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Load failed: {}", resp.text().await.unwrap_or_default()));
    }
    Ok(())
}

async fn unload_model_internal(instance_id: &str) -> Result<(), String> {
    let url = format!("{}/api/v1/models/unload", lm_base_url());
    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&json!({ "instance_id": instance_id })).send().await
        .map_err(|e| format!("Unload failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Unload failed: {}", resp.text().await.unwrap_or_default()));
    }
    Ok(())
}

#[tauri::command]
async fn load_model(config: LoadConfig) -> Result<(), String> {
    load_model_internal(&AgentLoadConfig {
        context_length: config.context_length,
        eval_batch_size: config.eval_batch_size,
        flash_attention: config.flash_attention,
        num_experts: config.num_experts,
        offload_kv_cache_to_gpu: config.offload_kv_cache_to_gpu,
    }, &config.model).await
}

#[tauri::command]
async fn unload_model(instance_id: String) -> Result<(), String> {
    unload_model_internal(&instance_id).await
}

#[tauri::command]
async fn download_model(model: String) -> Result<(), String> {
    let url = format!("{}/api/v1/models/download", lm_base_url());
    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&json!({ "model": model })).send().await
        .map_err(|e| format!("Download failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Download failed: {}", resp.text().await.unwrap_or_default()));
    }
    Ok(())
}

// ── LLM helpers ───────────────────────────────────────────────────

/// Non-streaming single-turn call. Used by manager MCP tools.
/// `history` contains previous [user / assistant] turns for this agent.
async fn call_chat_blocking(
    model_key: &str, system_prompt: &str, message: &str, history: &[Value],
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", lm_base_url());
    let mut messages = vec![];
    if !system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    for h in history { messages.push(h.clone()); }
    messages.push(json!({"role": "user", "content": message}));
    let client = reqwest::Client::new();
    let resp = client.post(&url)
        .json(&json!({ "model": model_key, "messages": messages, "stream": false }))
        .send().await
        .map_err(|e| format!("Request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("LLM error: {}", resp.text().await.unwrap_or_default()));
    }
    let data: Value = resp.json().await.map_err(|e| format!("Parse error: {}", e))?;
    Ok(data["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
}

/// Manager agent path: iterates the tool-call loop until the LLM returns a text response.
/// Emits Token events so the frontend streams the final answer.
/// `history` contains previous [user / assistant] turns for this agent.
async fn call_chat_with_tools(
    model_key: &str,
    system_prompt: &str,
    message: &str,
    agent_id: &str,
    tools: &[McpTool],
    mcp_port: u16,
    on_event: &Channel<StreamEvent>,
    history: &[Value],
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", lm_base_url());
    let mut messages = vec![];
    if !system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    for h in history { messages.push(h.clone()); }
    messages.push(json!({"role": "user", "content": message}));

    let tool_defs: Vec<Value> = tools.iter().map(|t| json!({
        "type": "function",
        "function": { "name": t.name, "description": t.description, "parameters": t.input_schema }
    })).collect();

    let client = reqwest::Client::new();

    for _ in 0u8..16 {
        let body = json!({
            "model": model_key,
            "messages": messages,
            "tools": tool_defs,
            "stream": false,
        });

        let resp = client.post(&url).json(&body).send().await
            .map_err(|e| format!("Request failed: {}", e))?;
        if !resp.status().is_success() {
            return Err(format!("LLM error: {}", resp.text().await.unwrap_or_default()));
        }
        let data: Value = resp.json().await.map_err(|e| format!("Parse error: {}", e))?;

        let choice = &data["choices"][0];
        let _finish_reason = choice["finish_reason"].as_str().unwrap_or("");
        let assistant_msg = choice["message"].clone();

        // Only treat as tool call if the array is non-empty ([] means no calls)
        let tool_calls_arr = assistant_msg["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let has_tool_calls = !tool_calls_arr.is_empty();

        if has_tool_calls {
            messages.push(assistant_msg.clone());
            for tc in tool_calls_arr {
                let call_id  = tc["id"].as_str().unwrap_or("").to_string();
                let fn_name  = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let fn_args: Value = serde_json::from_str(
                    tc["function"]["arguments"].as_str().unwrap_or("{}")
                ).unwrap_or(json!({}));

                // Notify frontend: tool about to be called
                let _ = on_event.send(StreamEvent::ToolCall {
                    agent_id: agent_id.to_string(),
                    tool_name: fn_name.clone(),
                    args: serde_json::to_string_pretty(&fn_args).unwrap_or_default(),
                });

                let mcp_req = json!({
                    "jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": { "name": fn_name, "arguments": fn_args }
                });

                let tool_result = match client
                    .post(format!("http://localhost:{}", mcp_port))
                    .json(&mcp_req).send().await
                {
                    Ok(r) => r.json::<Value>().await.ok()
                        .and_then(|v| {
                            v["result"]["content"][0]["text"].as_str()
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| "[tool returned no text]".to_string()),
                    Err(e) => format!("[tool call failed: {}]", e),
                };

                // Notify frontend: tool result received (truncate very long results)
                let preview = if tool_result.len() > 2000 {
                    format!("{}…[truncated, {} chars total]", &tool_result[..2000], tool_result.len())
                } else {
                    tool_result.clone()
                };
                let _ = on_event.send(StreamEvent::ToolResult {
                    agent_id: agent_id.to_string(),
                    tool_name: fn_name.clone(),
                    result: preview,
                });

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result,
                }));
            }
        } else {
            let content = assistant_msg["content"].as_str().unwrap_or("").to_string();
            if !content.is_empty() {
                let _ = on_event.send(StreamEvent::Token {
                    agent_id: agent_id.to_string(),
                    content: content.clone(),
                });
            }
            return Ok(content);
        }
    }
    Err("Max tool-call iterations (16) reached.".to_string())
}

// ── Streaming types ────────────────────────────────────────────────

#[derive(Clone, Serialize)]
#[serde(tag = "event")]
enum StreamEvent {
    #[serde(rename = "agent_start")]
    AgentStart { agent_id: String, agent_name: String },
    #[serde(rename = "token")]
    Token { agent_id: String, content: String },
    #[serde(rename = "agent_end")]
    AgentEnd { agent_id: String },
    #[serde(rename = "error")]
    Error { agent_id: String, agent_name: String, message: String },
    /// Emitted just before a manager tool call is dispatched.
    #[serde(rename = "tool_call")]
    ToolCall { agent_id: String, tool_name: String, args: String },
    /// Emitted after the tool returns its result.
    #[serde(rename = "tool_result")]
    ToolResult { agent_id: String, tool_name: String, result: String },
    #[serde(rename = "done")]
    Done,
}

async fn send_chat_completion_streaming(
    model_key: &str,
    system_prompt: &str,
    message: &str,
    agent_id: &str,
    on_event: &Channel<StreamEvent>,
    history: &[Value],
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", lm_base_url());
    let mut messages = vec![];
    if !system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    for h in history { messages.push(h.clone()); }
    messages.push(json!({"role": "user", "content": message}));

    let body = json!({ "model": model_key, "messages": messages, "stream": true });
    let client = reqwest::Client::new();
    let mut resp = client.post(&url).json(&body).send().await
        .map_err(|e| format!("Chat request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Chat failed: {}", resp.text().await.unwrap_or_default()));
    }

    let mut full_content = String::new();
    let mut buffer = String::new();

    while let Some(chunk) = resp.chunk().await.map_err(|e| format!("Stream read error: {}", e))? {
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim_end().to_string();
            buffer = buffer[pos + 1..].to_string();
            if line.is_empty() || line.starts_with(':') { continue; }
            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" { return Ok(full_content); }
                if let Ok(parsed) = serde_json::from_str::<Value>(data) {
                    if let Some(delta_content) = parsed["choices"][0]["delta"]["content"].as_str() {
                        if !delta_content.is_empty() {
                            full_content.push_str(delta_content);
                            let _ = on_event.send(StreamEvent::Token {
                                agent_id: agent_id.to_string(),
                                content: delta_content.to_string(),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(full_content)
}

// ── Agent config persistence ───────────────────────────────────────

fn config_path(workspace: &PathBuf) -> PathBuf {
    workspace.join("ajantis-config.json")
}

#[tauri::command]
async fn save_agent_config(
    state: tauri::State<'_, Arc<AppState>>,
    config: AgentConfig,
) -> Result<(), String> {
    *state.agent_config.lock().unwrap() = config.clone();
    let path = config_path(&state.workspace_root);
    let json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write config: {}", e))
}

#[tauri::command]
async fn load_agent_config(state: tauri::State<'_, Arc<AppState>>) -> Result<AgentConfig, String> {
    let path = config_path(&state.workspace_root);
    let config: AgentConfig = if path.exists() {
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config: {}", e))?
    } else {
        AgentConfig::default()
    };
    *state.agent_config.lock().unwrap() = config.clone();
    Ok(config)
}

// ── Memory pool commands ───────────────────────────────────────────

#[tauri::command]
async fn get_memory_pool(state: tauri::State<'_, Arc<AppState>>) -> Result<MemoryPool, String> {
    Ok(state.memory_pool.lock().unwrap().clone())
}

#[tauri::command]
async fn search_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
    query: String,
) -> Result<Vec<MemoryEntry>, String> {
    let pool = state.memory_pool.lock().unwrap();
    Ok(pool.search(&query).into_iter().cloned().collect())
}

#[tauri::command]
async fn clear_memory_pool(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    state.memory_pool.lock().unwrap().entries.clear();
    Ok(())
}

// ── Message routing ────────────────────────────────────────────────

#[tauri::command]
async fn route_message(
    state: tauri::State<'_, Arc<AppState>>,
    from_agent_id: String,
    message: String,
    on_event: Channel<StreamEvent>,
) -> Result<(), String> {
    let config = state.agent_config.lock().unwrap().clone();
    let memory = state.memory_pool.clone();
    let tools_arc = Arc::new(state.mcp_tools.clone());
    let mcp_port = state.mcp_port;
    let mut visited: HashSet<String> = HashSet::new();

    let model_type_map: HashMap<String, String> = match fetch_models().await {
        Ok(models) => models.into_iter()
            .filter_map(|m| m.model_type.map(|t| (m.key, t)))
            .collect(),
        Err(_) => HashMap::new(),
    };

    // Make the channel available to MCP tool handlers (e.g. send_message can emit sub-agent bubbles)
    *state.event_channel.lock().unwrap() = Some(on_event.clone());

    route_recursive(
        &config, &from_agent_id, &message, &on_event,
        &mut visited, &memory, &model_type_map, tools_arc, mcp_port,
    ).await;

    *state.event_channel.lock().unwrap() = None;
    let _ = on_event.send(StreamEvent::Done);
    Ok(())
}

#[async_recursion::async_recursion]
async fn route_recursive(
    config: &AgentConfig,
    from_id: &str,
    message: &str,
    on_event: &Channel<StreamEvent>,
    visited: &mut HashSet<String>,
    memory: &Arc<Mutex<MemoryPool>>,
    model_type_map: &HashMap<String, String>,
    tools: Arc<Vec<McpTool>>,
    mcp_port: u16,
) {
    // Collect enabled targets sorted by priority (lower = first)
    let mut targets: Vec<(String, u8)> = config.connections.iter()
        .filter(|c| c.from == from_id && c.enabled)
        .filter(|c| match &c.condition {
            Some(cond) if !cond.is_empty() =>
                message.to_lowercase().contains(&cond.to_lowercase()),
            _ => true,
        })
        .map(|c| (c.to.clone(), c.priority))
        .collect();
    targets.sort_by_key(|(_, p)| *p);

    for (target_id, _) in targets {
        let visit_key = format!("{}→{}", from_id, target_id);
        if visited.contains(&visit_key) { continue; }
        visited.insert(visit_key);

        // Extract all agent fields before any await to release the borrow cleanly
        let (agent_name, model_key, mode, role, load_cfg, is_manager, model_type_resolved) = {
            match config.agents.iter().find(|a| a.id == target_id) {
                None => continue,
                Some(agent) => {
                    if agent.agent_type == "user" || !agent.armed { continue; }
                    if agent.paused {
                        let _ = on_event.send(StreamEvent::Error {
                            agent_id: target_id.clone(),
                            agent_name: agent.name.clone(),
                            message: format!("Agent '{}' is paused.", agent.name),
                        });
                        continue;
                    }
                    let resolved = agent.model_type.clone()
                        .or_else(|| agent.model_key.as_deref()
                            .and_then(|k| model_type_map.get(k).cloned()));
                    (
                        agent.name.clone(),
                        agent.model_key.clone().unwrap_or_default(),
                        agent.mode.as_deref().unwrap_or("stay_awake").to_string(),
                        agent.role.clone().unwrap_or_default(),
                        agent.load_config.clone().unwrap_or(AgentLoadConfig {
                            context_length: None, eval_batch_size: None,
                            flash_attention: None, num_experts: None,
                            offload_kv_cache_to_gpu: None,
                        }),
                        agent.is_manager,
                        resolved,
                    )
                }
            }
        };

        if model_type_resolved.as_deref() == Some("embedding") {
            let _ = on_event.send(StreamEvent::Error {
                agent_id: target_id.clone(), agent_name: agent_name.clone(),
                message: format!("Agent '{}' uses an embedding model — not usable for chat.", agent_name),
            });
            continue;
        }

        if model_key.is_empty() {
            let _ = on_event.send(StreamEvent::Error {
                agent_id: target_id.clone(), agent_name: agent_name.clone(),
                message: format!("Agent '{}' has no model configured.", agent_name),
            });
            continue;
        }

        if mode == "on_the_fly" {
            if let Err(e) = load_model_internal(&load_cfg, &model_key).await {
                let _ = on_event.send(StreamEvent::Error {
                    agent_id: target_id.clone(), agent_name: agent_name.clone(),
                    message: format!("Failed to load model: {}", e),
                });
                continue;
            }
        }

        let _ = on_event.send(StreamEvent::AgentStart {
            agent_id: target_id.clone(),
            agent_name: agent_name.clone(),
        });

        // Reconstruct conversation history for this agent (all previous user/assistant turns)
        let history: Vec<Value> = {
            let pool = memory.lock().unwrap();
            pool.entries.iter()
                .filter(|e| e.agent_id == target_id && (e.role == "user" || e.role == "assistant"))
                .map(|e| json!({"role": e.role, "content": e.content}))
                .collect()
        };

        let result = if is_manager {
            // Only pass the manager-relevant tools to avoid bloating the prompt
            // (50+ tools → 12k token prompts → Channel Error on final response generation)
            const MANAGER_TOOLS: &[&str] = &[
                "list_agents", "send_message", "broadcast_message",
                "read_agent_messages", "memory_pool", "aggregate_results",
                "pause_agent", "resume_agent", "spawn_agent", "kill_agent",
                "get_agent_state", "fork_agent", "pipe_message",
            ];
            let manager_tools: Vec<McpTool> = tools.iter()
                .filter(|t| MANAGER_TOOLS.contains(&t.name.as_str()))
                .cloned()
                .collect();
            call_chat_with_tools(&model_key, &role, message, &target_id, &manager_tools, mcp_port, on_event, &history).await
        } else {
            send_chat_completion_streaming(&model_key, &role, message, &target_id, on_event, &history).await
        };

        match result {
            Ok(full_response) => {
                let _ = on_event.send(StreamEvent::AgentEnd { agent_id: target_id.clone() });
                // Store both sides of this turn so history is complete for the next message
                {
                    let mut pool = memory.lock().unwrap();
                    pool.push(&target_id, &agent_name, "user", message);
                    pool.push(&target_id, &agent_name, "assistant", &full_response);
                }
                if mode == "on_the_fly" { let _ = unload_model_internal(&model_key).await; }
                route_recursive(
                    config, &target_id, &full_response, on_event,
                    visited, memory, model_type_map, tools.clone(), mcp_port,
                ).await;
            }
            Err(e) => {
                if mode == "on_the_fly" { let _ = unload_model_internal(&model_key).await; }
                let _ = on_event.send(StreamEvent::Error {
                    agent_id: target_id.clone(),
                    agent_name: agent_name.clone(),
                    message: e,
                });
            }
        }
    }
}

// ── MCP Server ─────────────────────────────────────────────────────

#[derive(Clone)]
struct McpState {
    tools: Vec<McpTool>,
    workspace_root: PathBuf,
    todo_list: Arc<Mutex<Vec<Value>>>,
    memory_pool: Arc<Mutex<MemoryPool>>,
    /// Shared agent registry — manager tools read/write this at runtime.
    agent_config: Arc<Mutex<AgentConfig>>,
    /// Self-reference port so manager tools can call other MCP tools.
    mcp_port: u16,
    /// Shared with AppState — allows MCP tool handlers to emit stream events.
    event_channel: Arc<Mutex<Option<Channel<StreamEvent>>>>,
}

#[derive(Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

async fn handle_jsonrpc(
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

async fn handle_tool_call(name: &str, args: &Value, state: &McpState) -> Value {
    match name {
        // ── File system ───────────────────────────────────────────
        "bash" => {
            let command = args["command"].as_str().unwrap_or("").trim().to_string();
            if command.is_empty() { return mcp_error("Error: command is required."); }
            if command.contains('|') || command.contains(";&")
                || command.contains('>') || command.contains('<')
            {
                return mcp_error("Error: unsafe characters in command.");
            }
            match Command::new("bash").arg("-lc").arg(&command)
                .current_dir(&state.workspace_root).output()
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
            let abs = state.workspace_root.join(file_path);
            if !abs.starts_with(&state.workspace_root) {
                return mcp_error("Error: access outside workspace is not allowed.");
            }
            match fs::read_to_string(&abs) {
                Ok(data) => {
                    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
                    let limit = args["limit"].as_u64().map(|l| l as usize);
                    let slice = match limit {
                        Some(l) => &data[offset.min(data.len())..(offset + l).min(data.len())],
                        None => &data[offset.min(data.len())..],
                    };
                    mcp_ok(slice)
                }
                Err(e) => mcp_error(&format!("Error reading file: {}", e)),
            }
        }
        "write_file" => {
            let file_path = args["path"].as_str().unwrap_or("");
            let content = args["content"].as_str().unwrap_or("");
            if file_path.is_empty() { return mcp_error("Error: path is required."); }
            let abs = state.workspace_root.join(file_path);
            if !abs.starts_with(&state.workspace_root) {
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
            let abs = state.workspace_root.join(file_path);
            if !abs.starts_with(&state.workspace_root) {
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
                .map(|p| state.workspace_root.join(p))
                .unwrap_or_else(|| state.workspace_root.clone());
            let full_pattern = base.join(pattern).to_string_lossy().to_string();
            match glob::glob(&full_pattern) {
                Ok(paths) => {
                    let results: Vec<String> = paths
                        .filter_map(|r| r.ok())
                        .filter(|p| p.starts_with(&state.workspace_root))
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
                .map(|p| state.workspace_root.join(p))
                .unwrap_or_else(|| state.workspace_root.clone());
            let case_flag = args["-i"].as_bool().unwrap_or(false);
            let mut cmd = Command::new("rg");
            cmd.arg("--no-heading").arg("-n");
            if case_flag { cmd.arg("-i"); }
            if let Some(glob_pat) = args["glob"].as_str() {
                cmd.arg("--glob").arg(glob_pat);
            }
            cmd.arg(pattern).arg(&search_path).current_dir(&state.workspace_root);
            match cmd.output().or_else(|_| {
                let mut fallback = Command::new("grep");
                fallback.arg("-r").arg("-n");
                if case_flag { fallback.arg("-i"); }
                fallback.arg(pattern).arg(&search_path).current_dir(&state.workspace_root);
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
        _ => mcp_ok(&format!("Tool '{}' is not implemented in this server.", name)),
    }
}

fn mcp_ok(text: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

fn mcp_error(text: &str) -> Value {
    json!({ "isError": true, "content": [{ "type": "text", "text": text }] })
}

async fn start_mcp_server(port: u16, mcp_state: McpState) {
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

// ── Tauri entry point ──────────────────────────────────────────────

fn load_tools(tools_path: &PathBuf) -> Vec<McpTool> {
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // tools.json lives at the project root; when running from src-tauri/ look one level up
    let tools_path = {
        let in_cwd = workspace_root.join("tools.json");
        let in_parent = workspace_root.parent().map(|p| p.join("tools.json"));
        match in_parent.filter(|p| p.exists()) {
            Some(p) if !in_cwd.exists() => p,
            _ => in_cwd,
        }
    };
    let mcp_tools = load_tools(&tools_path);
    let mcp_port: u16 = 4785;

    let agent_config: AgentConfig = {
        let path = config_path(&workspace_root);
        if path.exists() {
            fs::read_to_string(&path).ok()
                .and_then(|c| serde_json::from_str(&c).ok())
                .unwrap_or_default()
        } else {
            AgentConfig::default()
        }
    };

    let agent_config_arc  = Arc::new(Mutex::new(agent_config));
    let memory_pool_arc   = Arc::new(Mutex::new(MemoryPool::default()));
    let todo_list_arc     = Arc::new(Mutex::new(vec![]));
    let event_channel_arc: Arc<Mutex<Option<Channel<StreamEvent>>>> =
        Arc::new(Mutex::new(None));

    let app_state = Arc::new(AppState {
        current_model: Mutex::new(
            std::env::var("LM_STUDIO_MODEL").unwrap_or_else(|_| "lmstudio/lmstudio-1B".to_string()),
        ),
        last_response_id: Mutex::new(None),
        mcp_port,
        workspace_root: workspace_root.clone(),
        mcp_tools: mcp_tools.clone(),
        todo_list: todo_list_arc.clone(),
        agent_config: agent_config_arc.clone(),
        memory_pool: memory_pool_arc.clone(),
        event_channel: event_channel_arc.clone(),
    });

    let mcp_state = McpState {
        tools: mcp_tools,
        workspace_root,
        todo_list: todo_list_arc,
        memory_pool: memory_pool_arc,
        agent_config: agent_config_arc,
        mcp_port,
        event_channel: event_channel_arc,
    };

    tauri::Builder::default()
        .manage(app_state)
        .setup(move |app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            let mcp_state_clone = mcp_state.clone();
            tauri::async_runtime::spawn(async move {
                start_mcp_server(mcp_port, mcp_state_clone).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            send_message,
            fetch_models,
            fetch_loaded_models,
            set_model,
            load_model,
            unload_model,
            download_model,
            save_agent_config,
            load_agent_config,
            get_memory_pool,
            search_memory_pool,
            clear_memory_pool,
            route_message,
            pick_folder,
            load_workspace_config,
            save_workspace_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
