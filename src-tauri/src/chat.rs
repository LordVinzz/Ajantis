use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tauri::ipc::Channel;

use crate::helpers::lm_base_url;
use crate::state::{AppState, McpTool};

#[derive(Serialize)]
pub(crate) struct SendMessageResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[tauri::command]
pub(crate) async fn send_message(
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

/// Non-streaming single-turn call. Used by manager MCP tools.
/// `history` contains previous [user / assistant] turns for this agent.
pub(crate) async fn call_chat_blocking(
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
pub(crate) async fn call_chat_with_tools(
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

    let tool_defs: Vec<Value> = tools.iter().map(|t| {
        // Compress parameter schemas: keep name/type/required but strip verbose descriptions
        // and nested details. This cuts token usage ~10x while keeping full tool coverage.
        let compressed_params = compress_schema(&t.input_schema);
        // Truncate description to first sentence (≤120 chars) to save tokens.
        let short_desc = t.description
            .split(['.', '\n']).next().unwrap_or(&t.description)
            .trim().chars().take(120).collect::<String>();
        json!({
            "type": "function",
            "function": { "name": t.name, "description": short_desc, "parameters": compressed_params }
        })
    }).collect();

    // Separate clients: LLM calls can take minutes, MCP tool calls should be fast.
    let llm_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build().unwrap_or_default();
    let mcp_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build().unwrap_or_default();

    for _ in 0u8..16 {
        let body = json!({
            "model": model_key,
            "messages": messages,
            "tools": tool_defs,
            "stream": false,
        });

        let resp = llm_client.post(&url).json(&body).send().await
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

                let tool_result = match mcp_client
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
                let char_count = tool_result.chars().count();
                let preview = if char_count > 2000 {
                    let head: String = tool_result.chars().take(2000).collect();
                    format!("{}…[truncated, {} chars total]", head, char_count)
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
            // No more tool calls — stream the final answer so it appears progressively.
            let stream_body = json!({
                "model": model_key,
                "messages": messages,
                "tools": tool_defs,
                "stream": true,
            });
            let stream_resp = llm_client.post(&url).json(&stream_body).send().await
                .map_err(|e| format!("Stream request failed: {}", e))?;
            if !stream_resp.status().is_success() {
                return Err(format!("LLM stream error: {}", stream_resp.text().await.unwrap_or_default()));
            }
            let mut full_content = String::new();
            let mut buffer = String::new();
            let mut stream_resp = stream_resp;
            while let Some(chunk) = stream_resp.chunk().await.map_err(|e| format!("Stream read: {}", e))? {
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim_end().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    if line.is_empty() || line.starts_with(':') { continue; }
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data.trim() == "[DONE]" { return Ok(full_content); }
                        if let Ok(parsed) = serde_json::from_str::<Value>(data) {
                            // If the model decided to call a tool after all, fall back to non-streaming.
                            if parsed["choices"][0]["delta"]["tool_calls"].is_array() {
                                return Ok(full_content);
                            }
                            if let Some(delta) = parsed["choices"][0]["delta"]["content"].as_str() {
                                if !delta.is_empty() {
                                    full_content.push_str(delta);
                                    let _ = on_event.send(StreamEvent::Token {
                                        agent_id: agent_id.to_string(),
                                        content: delta.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            return Ok(full_content);
        }
    }
    Err("Max tool-call iterations (16) reached.".to_string())
}

#[derive(Clone, Serialize)]
#[serde(tag = "event")]
pub(crate) enum StreamEvent {
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

/// Strip verbose descriptions from a JSON Schema while keeping the structure
/// the LLM needs to call the function: property names, types, and required list.
fn compress_schema(schema: &Value) -> Value {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return json!({"type": "object", "properties": {}}),
    };

    let schema_type = obj.get("type").cloned().unwrap_or(json!("object"));

    let compressed_props: serde_json::Map<String, Value> = obj
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|props| {
            props.iter().map(|(k, v)| {
                let prop_type = v.get("type").cloned().unwrap_or(json!("string"));
                // Keep enum values — they're load-bearing for the LLM.
                let mut compressed = json!({ "type": prop_type });
                if let Some(en) = v.get("enum") {
                    compressed["enum"] = en.clone();
                }
                // Keep nested object structure one level deep (e.g. task params).
                if prop_type == "object" {
                    if let Some(nested) = v.get("properties") {
                        compressed["properties"] = nested.clone();
                    }
                }
                (k.clone(), compressed)
            }).collect()
        })
        .unwrap_or_default();

    let mut out = json!({
        "type": schema_type,
        "properties": compressed_props,
    });
    if let Some(req) = obj.get("required") {
        out["required"] = req.clone();
    }
    out
}

pub(crate) async fn send_chat_completion_streaming(
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
