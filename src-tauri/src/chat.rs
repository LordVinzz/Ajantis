use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tauri::ipc::Channel;

use crate::agent_config::{
    RedundancyDetectionConfig, GROUNDED_AUDIT_BEHAVIOR_ID,
    MAX_SEMANTIC_SIMILARITY_THRESHOLD, MIN_SEMANTIC_SIMILARITY_THRESHOLD,
};
use crate::helpers::{
    audit_response_acknowledges_refs, compute_context_budget, extract_explicit_audit_refs,
    grounded_audit_behavior_prompt, has_file_reference, is_low_value_audit_response,
    is_manager_blocked_tool, is_manager_only_tool, is_path_like_audit_ref, lm_base_url,
    normalize_audit_ref, resolve_active_behaviors, with_context_budget,
};
use crate::mcp::{handle_tool_call, McpState};
use crate::models::create_embeddings;
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
                    ok: false,
                    data: None,
                    error: Some(format!("LM Studio API error: {} {}", status, text)),
                });
            }
            match resp.json::<Value>().await {
                Ok(data) => {
                    if let Some(rid) = data.get("response_id").and_then(|v| v.as_str()) {
                        *state.last_response_id.lock().unwrap() = Some(rid.to_string());
                    }
                    Ok(SendMessageResponse {
                        ok: true,
                        data: Some(data),
                        error: None,
                    })
                }
                Err(e) => Ok(SendMessageResponse {
                    ok: false,
                    data: None,
                    error: Some(format!("Failed to parse response: {}", e)),
                }),
            }
        }
        Err(e) => Ok(SendMessageResponse {
            ok: false,
            data: None,
            error: Some(format!("Request failed: {}", e)),
        }),
    }
}

/// Non-streaming single-turn call. Used by manager MCP tools.
/// `history` contains previous [user / assistant] turns for this agent.
pub(crate) async fn call_chat_blocking(
    model_key: &str,
    system_prompt: &str,
    message: &str,
    history: &[Value],
    context_limit: Option<u64>,
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", lm_base_url());
    let budget = compute_context_budget(system_prompt, history, message, context_limit, 0);
    let effective_system_prompt = with_context_budget(system_prompt, budget);
    let mut messages = vec![];
    if !effective_system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": effective_system_prompt}));
    }
    for h in history {
        messages.push(h.clone());
    }
    messages.push(json!({"role": "user", "content": message}));
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&json!({ "model": model_key, "messages": messages, "stream": false }))
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "LLM error: {}",
            resp.text().await.unwrap_or_default()
        ));
    }
    let data: Value = resp
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;
    Ok(data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string())
}

struct ActiveBehaviorContextGuard<'a> {
    state: &'a McpState,
    agent_id: String,
}

impl Drop for ActiveBehaviorContextGuard<'_> {
    fn drop(&mut self) {
        self.state
            .active_behavior_contexts
            .lock()
            .unwrap()
            .remove(&self.agent_id);
    }
}

/// Manager agent path: iterates the tool-call loop until the LLM returns a text response.
/// Emits Token events so the frontend streams the final answer.
/// `history` contains previous [user / assistant] turns for this agent.
#[async_recursion::async_recursion]
pub(crate) async fn call_chat_with_tools(
    model_key: &str,
    system_prompt: &str,
    message: &str,
    agent_id: &str,
    tools: &[McpTool],
    allowed_tools: Option<&[String]>,
    allow_manager_tools: bool,
    require_delegation: bool,
    context_limit: Option<u64>,
    glob_ready: bool,
    tool_state: &McpState,
    inherited_behaviors: Option<&HashSet<String>>,
    on_event: &Channel<StreamEvent>,
    history: &[Value],
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", lm_base_url());
    let mut messages = vec![];
    let (redundancy_config, behavior_triggers) = {
        let config = tool_state.agent_config.lock().unwrap();
        (
            config.redundancy_detection.clone(),
            config.behavior_triggers.clone(),
        )
    };
    let active_behaviors = resolve_active_behaviors(
        message,
        inherited_behaviors,
        &behavior_triggers,
        &tool_state.behavior_trigger_cache,
    )
    .await;
    let grounded_audit_mode = active_behaviors.contains(GROUNDED_AUDIT_BEHAVIOR_ID);
    tool_state
        .active_behavior_contexts
        .lock()
        .unwrap()
        .insert(agent_id.to_string(), active_behaviors);
    let _behavior_context_guard = ActiveBehaviorContextGuard {
        state: tool_state,
        agent_id: agent_id.to_string(),
    };
    let audit_request_summary = summarize_audit_request(message);
    let budget_tools = visible_tools_for_agent(tools, allowed_tools, allow_manager_tools, glob_ready);
    let tool_defs: Vec<Value> = budget_tools
        .iter()
        .map(|t| {
            // Compress parameter schemas: keep name/type/required but strip verbose descriptions
            // and nested details. This cuts token usage ~10x while keeping full tool coverage.
            let compressed_params = compress_schema(&t.input_schema);
            // Truncate description to first sentence (≤120 chars) to save tokens.
            let short_desc = t
                .description
                .split(['.', '\n'])
                .next()
                .unwrap_or(&t.description)
                .trim()
                .chars()
                .take(120)
                .collect::<String>();
            json!({
                "type": "function",
                "function": { "name": t.name, "description": short_desc, "parameters": compressed_params }
            })
        })
        .collect();
    let tool_overhead = serde_json::to_string(&tool_defs)
        .ok()
        .map(|text| text.chars().count() as u64)
        .unwrap_or(0)
        .div_ceil(4);
    let budget = compute_context_budget(
        system_prompt,
        history,
        message,
        context_limit,
        tool_overhead,
    );
    let effective_system_prompt = with_context_budget(system_prompt, budget);
    if !effective_system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": effective_system_prompt}));
    }
    for h in history {
        messages.push(h.clone());
    }
    messages.push(json!({"role": "user", "content": message}));

    // Separate clients: LLM calls can take minutes, MCP tool calls should be fast.
    let llm_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap_or_default();
    let mut delegated = false;
    let mut tool_burst = 0usize;
    let mut non_progress_count = 0usize;
    let mut failure_signatures: HashSet<String> = HashSet::new();
    let mut glob_scope_ready = glob_ready;
    let mut must_finish_from_evidence = false;
    let mut covered_audit_topics: Vec<CoveredAuditTopic> = Vec::new();
    let mut redundant_audit_retries = 0usize;
    let mut weak_delegation_retries: HashMap<String, usize> = HashMap::new();
    let mut coverage_manifest = if grounded_audit_mode {
        initialize_audit_coverage_manifest(message)
    } else {
        Vec::new()
    };
    const NON_PROGRESS_LIMIT: usize = 4;

    for _ in 0u8..64 {
        let redundancy_note =
            audit_runtime_note(grounded_audit_mode, &covered_audit_topics, &coverage_manifest);
        let mut request_messages = vec![];
        if grounded_audit_mode {
            request_messages.push(json!({
                "role": "system",
                "content": grounded_audit_behavior_prompt(),
            }));
        }
        request_messages.extend(messages.clone());
        if let Some(note) = redundancy_note {
            request_messages.push(json!({"role": "system", "content": note}));
        }
        let visible_tools =
            visible_tools_for_agent(tools, allowed_tools, allow_manager_tools, glob_scope_ready);
        let tool_defs: Vec<Value> = visible_tools
            .iter()
            .map(|t| {
                let compressed_params = compress_schema(&t.input_schema);
                let short_desc = t
                    .description
                    .split(['.', '\n'])
                    .next()
                    .unwrap_or(&t.description)
                    .trim()
                    .chars()
                    .take(120)
                    .collect::<String>();
                json!({
                    "type": "function",
                    "function": { "name": t.name, "description": short_desc, "parameters": compressed_params }
                })
            })
            .collect();
        let body = json!({
            "model": model_key,
            "messages": request_messages,
            "tools": tool_defs,
            "stream": false,
        });

        let resp = llm_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;
        if !resp.status().is_success() {
            return Err(format!(
                "LLM error: {}",
                resp.text().await.unwrap_or_default()
            ));
        }
        let data: Value = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

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
            let mut iteration_progress = false;
            for tc in tool_calls_arr {
                let call_id = tc["id"].as_str().unwrap_or("").to_string();
                let fn_name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let fn_args: Value =
                    serde_json::from_str(tc["function"]["arguments"].as_str().unwrap_or("{}"))
                        .unwrap_or(json!({}));
                let args_text = serde_json::to_string(&fn_args).unwrap_or_default();
                if is_delegation_tool(&fn_name) {
                    delegated = true;
                }
                if grounded_audit_mode {
                    extend_audit_coverage_manifest(&mut coverage_manifest, &args_text);
                }
                tool_burst += 1;

                // Notify frontend: tool about to be called
                let _ = on_event.send(StreamEvent::ToolCall {
                    agent_id: agent_id.to_string(),
                    tool_name: fn_name.clone(),
                    args: serde_json::to_string_pretty(&fn_args).unwrap_or_default(),
                });

                let pre_dispatch_topic = if grounded_audit_mode && allow_manager_tools {
                    Some(
                        materialize_audit_topic(
                            &fn_name,
                            &args_text,
                            None,
                            &audit_request_summary,
                            &redundancy_config,
                        )
                        .await,
                    )
                } else {
                    None
                };

                let redundant_before_dispatch = if let Some(Some(candidate)) = pre_dispatch_topic.as_ref() {
                    detect_redundant_audit_topic(
                        candidate,
                        &covered_audit_topics,
                        &redundancy_config,
                    )
                } else {
                    None
                };

                let (tool_is_error, tool_text) = if let Some(reason) = redundant_before_dispatch.clone()
                {
                    (
                        false,
                        format!(
                            "[redundant audit topic skipped: {}. Stop re-checking this area and synthesize from current evidence.]",
                            reason
                        ),
                    )
                } else {
                    dispatch_tool_call(tool_state, &fn_name, &fn_args, agent_id).await
                };

                // Notify frontend: tool result received (truncate very long results)
                let char_count = tool_text.chars().count();
                let preview = if char_count > 2000 {
                    let head: String = tool_text.chars().take(2000).collect();
                    format!("{}…[truncated, {} chars total]", head, char_count)
                } else {
                    tool_text.clone()
                };
                let _ = on_event.send(StreamEvent::ToolResult {
                    agent_id: agent_id.to_string(),
                    tool_name: fn_name.clone(),
                    result: preview,
                });

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_text,
                }));

                if redundant_before_dispatch.is_some() {
                    redundant_audit_retries += 1;
                    non_progress_count += 1;
                    if allow_manager_tools
                        && has_usable_evidence(&messages, grounded_audit_mode)
                        && redundant_audit_retries
                            > usize::from(redundancy_config.max_redundant_audit_retries)
                    {
                        must_finish_from_evidence = true;
                    }
                    continue;
                }

                let failure_signature = classify_non_progress_tool_result(&fn_name, &tool_text);
                if fn_name == "glob_search" {
                    glob_scope_ready = !tool_is_error && !tool_text.trim().is_empty() && tool_text != "No files matched.";
                }
                if grounded_audit_mode {
                    update_audit_coverage_manifest_from_result(
                        &mut coverage_manifest,
                        &fn_name,
                        &args_text,
                        &tool_text,
                    );
                }
                if tool_is_error || failure_signature.is_some() {
                    let signature = failure_signature
                        .clone()
                        .unwrap_or_else(|| format!("{}::generic_error", fn_name));
                    if !failure_signatures.insert(signature) {
                        non_progress_count += 1;
                    }
                    if allow_manager_tools
                        && (is_delegation_tool(&fn_name)
                            || matches!(
                                failure_signature.as_deref(),
                                Some(sig) if sig.ends_with("::agent_error")
                                    || sig.ends_with("::internal_dispatch_failure")
                            ))
                        && has_usable_evidence(&messages, grounded_audit_mode)
                    {
                        must_finish_from_evidence = true;
                    }
                } else if is_delegation_tool(&fn_name)
                    && is_unusable_delegation_result(&tool_text)
                {
                    non_progress_count += 1;
                    if allow_manager_tools && has_usable_evidence(&messages, grounded_audit_mode) {
                        must_finish_from_evidence = true;
                    }
                } else if !tool_text.trim().is_empty() {
                    iteration_progress = true;
                    non_progress_count = 0;
                }

                if grounded_audit_mode && allow_manager_tools {
                    if let Some(topic) = materialize_audit_topic(
                        &fn_name,
                        &args_text,
                        Some(&tool_text),
                        &audit_request_summary,
                        &redundancy_config,
                    )
                    .await
                    {
                        if let Some(_reason) = detect_redundant_audit_topic(
                            &topic,
                            &covered_audit_topics,
                            &redundancy_config,
                        ) {
                            redundant_audit_retries += 1;
                            non_progress_count += 1;
                            if allow_manager_tools
                                && has_usable_evidence(&messages, grounded_audit_mode)
                                && redundant_audit_retries
                                    > usize::from(redundancy_config.max_redundant_audit_retries)
                            {
                                must_finish_from_evidence = true;
                            }
                        } else if !tool_is_error && !tool_text.trim().is_empty() {
                            mark_audit_coverage_manifest_inspected(&mut coverage_manifest, &topic);
                            covered_audit_topics.push(topic);
                        }
                    }
                }

                if grounded_audit_mode && is_delegation_tool(&fn_name) && !tool_is_error {
                    let delegation_topic_key =
                        stable_delegation_retry_key(&args_text, &audit_request_summary);
                    if is_weak_audit_delegation_result(&tool_text, &covered_audit_topics) {
                        let retries = weak_delegation_retries
                            .entry(delegation_topic_key.clone())
                            .or_insert(0);
                        *retries += 1;
                        if *retries == 1 {
                            non_progress_count += 1;
                            messages.push(json!({
                                "role": "user",
                                "content": format!(
                                    "The delegated audit result on `{}` did not add new useful evidence. Retry this topic at most once with one concrete file or function to inspect and one concrete question to answer. Do not return a plan, status update, or generic hypotheses.",
                                    delegation_topic_key
                                )
                            }));
                        } else {
                            non_progress_count += 2;
                            redundant_audit_retries += 1;
                            if allow_manager_tools && has_usable_evidence(&messages, grounded_audit_mode) {
                                must_finish_from_evidence = true;
                            }
                        }
                    }
                }

                if grounded_audit_mode
                    && allow_manager_tools
                    && has_usable_evidence(&messages, true)
                    && should_force_audit_synthesis(
                        &messages,
                        &covered_audit_topics,
                        &coverage_manifest,
                        redundant_audit_retries,
                        &redundancy_config,
                    )
                {
                    must_finish_from_evidence = true;
                }

                if tool_burst >= 3 {
                    let reflection =
                        force_tool_free_reflection(&llm_client, &url, model_key, &messages).await?;
                    if !reflection.is_empty() {
                        messages.push(json!({"role": "assistant", "content": reflection}));
                    }
                    tool_burst = 0;
                }
            }
            if !iteration_progress {
                non_progress_count += 1;
            }
            if must_finish_from_evidence || non_progress_count >= NON_PROGRESS_LIMIT {
                let missing_scope_refs =
                    unresolved_audit_coverage_manifest_entries(&coverage_manifest, None);
                return Ok(force_non_progress_summary(
                    &llm_client,
                    &url,
                    model_key,
                    &messages,
                    grounded_audit_mode,
                    &missing_scope_refs,
                ).await);
            }
        } else {
            if require_delegation && !delegated {
                messages.push(json!({
                    "role": "user",
                    "content": "You are the manager. You must delegate at least one concrete sub-task before giving a final answer. Use agent-management tools first, then synthesize."
                }));
                continue;
            }

            // We already have the final assistant message in this non-streaming response.
            // Re-issuing a second streaming request can lose the answer and return an empty string.
            let final_content = assistant_msg["content"]
                .as_str()
                .unwrap_or("")
                .to_string();

            let final_content = if grounded_audit_mode
                && audit_response_needs_rewrite(&final_content, audit_evidence_grade(&messages))
            {
                rewrite_grounded_audit_response(
                    &llm_client,
                    &url,
                    model_key,
                    &messages,
                    &final_content,
                )
                .await
            } else {
                final_content
            };

            let missing_scope_refs = if grounded_audit_mode {
                unresolved_audit_coverage_manifest_entries(&coverage_manifest, Some(&final_content))
            } else {
                Vec::new()
            };

            if final_content.trim().is_empty() || is_progress_only_response(&final_content) {
                non_progress_count += 1;
                if non_progress_count >= NON_PROGRESS_LIMIT {
                    let missing_scope_refs =
                        unresolved_audit_coverage_manifest_entries(&coverage_manifest, None);
                    return Ok(force_non_progress_summary(
                        &llm_client,
                        &url,
                        model_key,
                        &messages,
                        grounded_audit_mode,
                        &missing_scope_refs,
                    ).await);
                }
                messages.push(json!({
                    "role": "user",
                    "content": "Your previous response did not contain a usable final answer. Stop exploring and provide either concrete findings or a short bounded summary of what is known, what blocked progress, and the smallest remaining useful next step."
                }));
                continue;
            }
            if grounded_audit_mode && !missing_scope_refs.is_empty() {
                messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "Before concluding, you must either inspect these required audit scopes or name them explicitly under `Coverage gaps`: {}. Do not omit them.",
                        missing_scope_refs.join(", ")
                    )
                }));
                continue;
            }
            if grounded_audit_mode && hypotheses_need_rewrite(&final_content) {
                messages.push(json!({
                    "role": "user",
                    "content": "Your `Hypotheses / lower-confidence risks` section is still too generic or template-like. Keep only hypotheses tied to a concrete file/config or observed behavior already in context, and state the missing proof needed to confirm them. Remove generic dependency-presence or stack-template risks."
                }));
                continue;
            }

            if !final_content.is_empty() {
                let _ = on_event.send(StreamEvent::Token {
                    agent_id: agent_id.to_string(),
                    content: final_content.clone(),
                });
            }

            return Ok(final_content);
        }
    }
    Ok(force_non_progress_summary(
        &llm_client,
        &url,
        model_key,
        &messages,
        grounded_audit_mode,
        &unresolved_audit_coverage_manifest_entries(&coverage_manifest, None),
    ).await)
}

#[derive(Clone, Serialize)]
#[serde(tag = "event")]
pub(crate) enum StreamEvent {
    #[serde(rename = "agent_start")]
    AgentStart {
        agent_id: String,
        agent_name: String,
    },
    #[serde(rename = "token")]
    Token { agent_id: String, content: String },
    #[serde(rename = "agent_end")]
    AgentEnd { agent_id: String },
    #[serde(rename = "error")]
    Error {
        agent_id: String,
        agent_name: String,
        message: String,
    },
    /// Emitted just before a manager tool call is dispatched.
    #[serde(rename = "tool_call")]
    ToolCall {
        agent_id: String,
        tool_name: String,
        args: String,
    },
    /// Emitted after the tool returns its result.
    #[serde(rename = "tool_result")]
    ToolResult {
        agent_id: String,
        tool_name: String,
        result: String,
    },
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
            props
                .iter()
                .map(|(k, v)| {
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
                })
                .collect()
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

fn visible_tools_for_agent(
    tools: &[McpTool],
    allowed_tools: Option<&[String]>,
    allow_manager_tools: bool,
    glob_ready: bool,
) -> Vec<McpTool> {
    tools
        .iter()
        .filter(|tool| {
            allowed_tools
                .map(|names| names.iter().any(|name| name == &tool.name))
                .unwrap_or(true)
        })
        .filter(|tool| {
            if allow_manager_tools {
                !is_manager_blocked_tool(&tool.name)
            } else {
                !is_manager_only_tool(&tool.name)
            }
        })
        .filter(|tool| {
            if glob_ready {
                true
            } else {
                !matches!(tool.name.as_str(), "read_file" | "grep_search")
            }
        })
        .cloned()
        .collect()
}

fn is_delegation_tool(name: &str) -> bool {
    matches!(
        name,
        "spawn_agent" | "send_message" | "broadcast_message" | "fork_agent" | "pipe_message"
    )
}

#[derive(Clone)]
struct CoveredAuditTopic {
    lexical_signature: String,
    summary: String,
    refs: HashSet<String>,
    target_refs: HashSet<String>,
    inspection_depth: AuditInspectionDepth,
    embedding: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AuditInspectionDepth {
    Discovery,
    Targeted,
    EvidenceBacked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuditCoverageStatus {
    NotYetCovered,
    Inspected,
    ReportedGap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuditCoverageScope {
    BackendRust,
    Frontend,
    Configs,
    Capabilities,
}

#[derive(Clone, Debug)]
struct AuditCoverageEntry {
    key: String,
    label: String,
    status: AuditCoverageStatus,
    scope: Option<AuditCoverageScope>,
}

fn initialize_audit_coverage_manifest(message: &str) -> Vec<AuditCoverageEntry> {
    let mut manifest = Vec::new();
    for scope in requested_broad_audit_scopes(message) {
        push_coverage_scope_entry(&mut manifest, scope);
    }
    for reference in extract_explicit_audit_refs(message) {
        push_coverage_file_entry(&mut manifest, &reference);
    }
    manifest
}

fn extend_audit_coverage_manifest(manifest: &mut Vec<AuditCoverageEntry>, text: &str) {
    for scope in requested_broad_audit_scopes(text) {
        push_coverage_scope_entry(manifest, scope);
    }
    for reference in extract_explicit_audit_refs(text) {
        if !manifest.iter().any(|entry| entry.scope.is_some())
            || (is_high_signal_audit_ref(&reference)
                && manifest_scope_matches_ref(manifest, &reference))
        {
            push_coverage_file_entry(manifest, &reference);
        }
    }
}

fn push_coverage_scope_entry(manifest: &mut Vec<AuditCoverageEntry>, scope: AuditCoverageScope) {
    let key = format!("scope:{}", audit_scope_key(scope));
    if manifest.iter().any(|entry| entry.key == key) {
        return;
    }
    manifest.push(AuditCoverageEntry {
        key,
        label: audit_scope_label(scope).to_string(),
        status: AuditCoverageStatus::NotYetCovered,
        scope: Some(scope),
    });
}

fn push_coverage_file_entry(manifest: &mut Vec<AuditCoverageEntry>, reference: &str) {
    let normalized = normalize_audit_ref(reference).unwrap_or_else(|| reference.to_lowercase());
    let key = format!("file:{}", normalized);
    if manifest.iter().any(|entry| entry.key == key) {
        return;
    }
    manifest.push(AuditCoverageEntry {
        key,
        label: normalized,
        status: AuditCoverageStatus::NotYetCovered,
        scope: None,
    });
}

fn requested_broad_audit_scopes(text: &str) -> Vec<AuditCoverageScope> {
    let lower = text.to_lowercase();
    let mut scopes = Vec::new();
    if lower.contains(".rs")
        || lower.contains(" rust")
        || lower.contains("backend")
        || lower.contains("src-tauri")
    {
        scopes.push(AuditCoverageScope::BackendRust);
    }
    if lower.contains("frontend")
        || lower.contains("src/")
        || lower.contains(".js")
        || lower.contains(".ts")
        || lower.contains(".html")
        || lower.contains("renderer")
    {
        scopes.push(AuditCoverageScope::Frontend);
    }
    if lower.contains("config")
        || lower.contains("configs")
        || lower.contains(".json")
        || lower.contains(".toml")
        || lower.contains("package.json")
        || lower.contains("tauri.conf")
    {
        scopes.push(AuditCoverageScope::Configs);
    }
    if lower.contains("capabilities")
        || lower.contains("permission")
        || lower.contains("default.json")
    {
        scopes.push(AuditCoverageScope::Capabilities);
    }
    scopes.sort_by_key(|scope| audit_scope_key(*scope));
    scopes.dedup();
    scopes
}

fn audit_scope_key(scope: AuditCoverageScope) -> &'static str {
    match scope {
        AuditCoverageScope::BackendRust => "backend_rust",
        AuditCoverageScope::Frontend => "frontend",
        AuditCoverageScope::Configs => "configs",
        AuditCoverageScope::Capabilities => "capabilities",
    }
}

fn audit_scope_label(scope: AuditCoverageScope) -> &'static str {
    match scope {
        AuditCoverageScope::BackendRust => "backend Rust (.rs / src-tauri)",
        AuditCoverageScope::Frontend => "frontend (src / JS / TS / HTML)",
        AuditCoverageScope::Configs => "configs (.json / .toml / package / tauri)",
        AuditCoverageScope::Capabilities => "capabilities / permissions",
    }
}

fn is_high_signal_audit_ref(reference: &str) -> bool {
    let lower = reference.to_lowercase();
    lower.ends_with(".rs")
        || lower.ends_with(".js")
        || lower.ends_with(".ts")
        || lower.ends_with(".html")
        || lower.ends_with(".json")
        || lower.ends_with(".toml")
}

fn manifest_scope_matches_ref(manifest: &[AuditCoverageEntry], reference: &str) -> bool {
    manifest
        .iter()
        .filter_map(|entry| entry.scope)
        .any(|scope| audit_scope_matches_ref(scope, reference))
}

fn audit_scope_matches_ref(scope: AuditCoverageScope, reference: &str) -> bool {
    let lower = reference.to_lowercase();
    match scope {
        AuditCoverageScope::BackendRust => lower.ends_with(".rs"),
        AuditCoverageScope::Frontend => {
            lower.starts_with("src/")
                || lower.ends_with(".js")
                || lower.ends_with(".ts")
                || lower.ends_with(".html")
        }
        AuditCoverageScope::Configs => lower.ends_with(".json") || lower.ends_with(".toml"),
        AuditCoverageScope::Capabilities => lower.contains("capabilities/") || lower.contains("permissions"),
    }
}

fn update_audit_coverage_manifest_from_result(
    manifest: &mut Vec<AuditCoverageEntry>,
    _tool_name: &str,
    args_text: &str,
    tool_text: &str,
) {
    extend_audit_coverage_manifest(manifest, args_text);
    if manifest.iter().any(|entry| entry.scope.is_some()) {
        for reference in extract_explicit_audit_refs(tool_text) {
            if is_high_signal_audit_ref(&reference) && manifest_scope_matches_ref(manifest, &reference) {
                push_coverage_file_entry(manifest, &reference);
            }
        }
    }
    if tool_text.to_lowercase().contains("coverage gaps") {
        mark_audit_coverage_manifest_reported_gap(manifest, tool_text);
    }
}

fn mark_audit_coverage_manifest_inspected(
    manifest: &mut [AuditCoverageEntry],
    topic: &CoveredAuditTopic,
) {
    let refs = topic
        .target_refs
        .iter()
        .chain(topic.refs.iter())
        .cloned()
        .collect::<HashSet<_>>();
    for entry in manifest.iter_mut() {
        if entry.status == AuditCoverageStatus::ReportedGap {
            continue;
        }
        if let Some(scope) = entry.scope {
            if refs.iter().any(|reference| audit_scope_matches_ref(scope, reference))
                && topic.inspection_depth >= AuditInspectionDepth::Targeted
            {
                entry.status = AuditCoverageStatus::Inspected;
            }
        } else if refs.iter().any(|reference| audit_ref_entry_matches(reference, &entry.label))
            && topic.inspection_depth >= AuditInspectionDepth::EvidenceBacked
        {
            entry.status = AuditCoverageStatus::Inspected;
        }
    }
}

fn mark_audit_coverage_manifest_reported_gap(
    manifest: &mut [AuditCoverageEntry],
    response: &str,
) {
    let coverage_gaps = extract_audit_section(response, "coverage gaps", &[]);
    let lower = coverage_gaps.to_lowercase();
    let seen_refs = extract_explicit_audit_refs(&coverage_gaps);
    for entry in manifest.iter_mut() {
        if let Some(scope) = entry.scope {
            if lower.contains(audit_scope_key(scope))
                || lower.contains(&audit_scope_label(scope).to_lowercase())
                || gap_mentions_scope(&lower, scope)
            {
                entry.status = AuditCoverageStatus::ReportedGap;
            }
        } else if audit_response_acknowledges_refs(&coverage_gaps, &[entry.label.clone()])
            || seen_refs
                .iter()
                .any(|seen| audit_ref_entry_matches(seen, &entry.label))
        {
            entry.status = AuditCoverageStatus::ReportedGap;
        }
    }
}

fn gap_mentions_scope(lower: &str, scope: AuditCoverageScope) -> bool {
    match scope {
        AuditCoverageScope::BackendRust => {
            lower.contains(".rs") || lower.contains("backend") || lower.contains("src-tauri")
        }
        AuditCoverageScope::Frontend => {
            lower.contains("frontend") || lower.contains("src/") || lower.contains(".js")
        }
        AuditCoverageScope::Configs => {
            lower.contains("config") || lower.contains(".json") || lower.contains(".toml")
        }
        AuditCoverageScope::Capabilities => {
            lower.contains("capabilities") || lower.contains("permission")
        }
    }
}

fn unresolved_audit_coverage_manifest_entries(
    manifest: &[AuditCoverageEntry],
    response: Option<&str>,
) -> Vec<String> {
    let mut snapshot = manifest.to_vec();
    if let Some(text) = response {
        mark_audit_coverage_manifest_reported_gap(&mut snapshot, text);
    }
    let mut missing = snapshot
        .iter()
        .filter(|entry| entry.status == AuditCoverageStatus::NotYetCovered)
        .map(|entry| entry.label.clone())
        .collect::<Vec<_>>();
    missing.sort();
    missing.dedup();
    missing
}

fn audit_ref_entry_matches(left: &str, right: &str) -> bool {
    let left = left.to_lowercase();
    let right = right.to_lowercase();
    left == right
        || left.ends_with(&format!("/{}", right))
        || right.ends_with(&format!("/{}", left))
}

fn summarize_audit_request(message: &str) -> String {
    normalize_topic_text(message, 18)
}

fn audit_runtime_note(
    grounded_audit_mode: bool,
    covered_topics: &[CoveredAuditTopic],
    coverage_manifest: &[AuditCoverageEntry],
) -> Option<String> {
    if !grounded_audit_mode || (covered_topics.is_empty() && coverage_manifest.is_empty()) {
        return None;
    }

    let items = covered_topics
        .iter()
        .rev()
        .take(4)
        .map(|topic| format!("- {}", topic.summary))
        .collect::<Vec<_>>()
        .join("\n");
    let items = if items.is_empty() {
        "- none yet".to_string()
    } else {
        items
    };

    let unresolved = unresolved_audit_coverage_manifest_entries(coverage_manifest, None);
    let unresolved_note = if unresolved.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nRequired audit coverage still unresolved:\n{}",
            unresolved
                .iter()
                .take(6)
                .map(|entry| format!("- {}", entry))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    Some(format!(
        "Audit topics already covered in this turn:\n{}\n\nAvoid re-delegating or re-checking the same file/function/theme unless you are opening a genuinely new code/config area. If current evidence is already sufficient, synthesize now.{}",
        items,
        unresolved_note
    ))
}

async fn materialize_audit_topic(
    tool_name: &str,
    args_text: &str,
    tool_result: Option<&str>,
    audit_request_summary: &str,
    config: &RedundancyDetectionConfig,
) -> Option<CoveredAuditTopic> {
    let combined = format!(
        "{}\n{}\n{}\n{}",
        tool_name,
        audit_request_summary,
        args_text,
        tool_result.unwrap_or("")
    );
    if !looks_like_audit_topic(&combined) {
        return None;
    }

    let refs = extract_topic_refs(&combined);
    let target_refs = extract_target_refs(args_text);
    let terms = extract_topic_terms(&combined);
    if refs.is_empty() && terms.is_empty() {
        return None;
    }

    let inspection_depth =
        infer_inspection_depth(tool_name, args_text, tool_result, &target_refs);
    let lexical_signature = lexical_topic_signature(tool_name, audit_request_summary, &refs, &terms);
    let summary = format_topic_summary(tool_name, audit_request_summary, &refs, &terms);
    let embedding = if config.enabled {
        match config.embedding_model_key.as_deref() {
            Some(model_key) if !model_key.trim().is_empty() => create_embeddings(
                model_key,
                &[summary.clone()],
            )
            .await
            .ok()
            .and_then(|mut vectors| vectors.drain(..).next()),
            _ => None,
        }
    } else {
        None
    };

    Some(CoveredAuditTopic {
        lexical_signature,
        summary,
        refs: refs.into_iter().collect(),
        target_refs,
        inspection_depth,
        embedding,
    })
}

fn detect_redundant_audit_topic(
    candidate: &CoveredAuditTopic,
    covered_topics: &[CoveredAuditTopic],
    config: &RedundancyDetectionConfig,
) -> Option<String> {
    if !config.enabled || covered_topics.is_empty() {
        return None;
    }

    if candidate
        .target_refs
        .iter()
        .any(|target| !has_target_ref_coverage(covered_topics, target, candidate.inspection_depth))
    {
        return None;
    }

    let effective_threshold = effective_similarity_threshold(config);
    for existing in covered_topics {
        let same_signature = candidate.lexical_signature == existing.lexical_signature;
        let no_new_refs = candidate.refs.is_subset(&existing.refs)
            && candidate.target_refs.is_subset(&existing.target_refs);
        let same_or_lower_depth = candidate.inspection_depth <= existing.inspection_depth;
        if same_signature && no_new_refs && same_or_lower_depth {
            return Some(format!("same lexical topic as {}", existing.summary));
        }
    }

    let Some(candidate_embedding) = candidate.embedding.as_ref() else {
        return None;
    };

    covered_topics.iter().find_map(|existing| {
        let existing_embedding = existing.embedding.as_ref()?;
        let similarity = cosine_similarity(candidate_embedding, existing_embedding)?;
        let overlaps_target_refs = !candidate.target_refs.is_empty()
            && !candidate.target_refs.is_disjoint(&existing.target_refs);
        let overlaps_refs = !candidate.refs.is_disjoint(&existing.refs);
        let same_or_lower_depth = candidate.inspection_depth <= existing.inspection_depth;
        if similarity >= effective_threshold
            && same_or_lower_depth
            && (overlaps_target_refs
                || (candidate.target_refs.is_empty() && candidate.refs.is_subset(&existing.refs))
                || (candidate.target_refs.is_empty() && overlaps_refs))
        {
            Some(format!(
                "semantic overlap {:.2} with {}",
                similarity, existing.summary
            ))
        } else {
            None
        }
    })
}

fn effective_similarity_threshold(config: &RedundancyDetectionConfig) -> f32 {
    config
        .semantic_similarity_threshold
        .clamp(
            MIN_SEMANTIC_SIMILARITY_THRESHOLD,
            MAX_SEMANTIC_SIMILARITY_THRESHOLD,
        )
}

fn has_target_ref_coverage(
    covered_topics: &[CoveredAuditTopic],
    target_ref: &str,
    candidate_depth: AuditInspectionDepth,
) -> bool {
    covered_topics.iter().any(|existing| {
        existing.target_refs.contains(target_ref) && existing.inspection_depth >= candidate_depth
    })
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (l, r) in left.iter().zip(right.iter()) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    let denom = left_norm.sqrt() * right_norm.sqrt();
    if denom <= f32::EPSILON {
        None
    } else {
        Some(dot / denom)
    }
}

fn looks_like_audit_topic(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        "audit",
        "security",
        "review",
        "finding",
        "findings",
        "vulnerab",
        "risk",
        "issue",
        ".rs",
        ".json",
        "fn ",
        "grep",
        "read_file",
        "mcp.rs",
        "workspace.rs",
        "models.rs",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn lexical_topic_signature(
    tool_name: &str,
    audit_request_summary: &str,
    refs: &[String],
    terms: &[String],
) -> String {
    let mut parts = vec![tool_name.to_lowercase()];
    if !audit_request_summary.is_empty() {
        parts.push(audit_request_summary.to_string());
    }
    parts.extend(refs.iter().cloned());
    parts.extend(terms.iter().take(8).cloned());
    parts.join(" | ")
}

fn format_topic_summary(
    tool_name: &str,
    audit_request_summary: &str,
    refs: &[String],
    terms: &[String],
) -> String {
    let refs_text = if refs.is_empty() {
        "no specific file refs".to_string()
    } else {
        refs.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
    };
    let term_text = if terms.is_empty() {
        String::new()
    } else {
        format!(
            " ({})",
            terms.iter().take(4).cloned().collect::<Vec<_>>().join(", ")
        )
    };
    if audit_request_summary.is_empty() {
        format!("{} on {}{}", tool_name, refs_text, term_text)
    } else {
        format!(
            "{} on {} for {}{}",
            tool_name, refs_text, audit_request_summary, term_text
        )
    }
}

fn extract_topic_refs(text: &str) -> Vec<String> {
    let mut refs = text
        .split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}'))
        .filter_map(clean_topic_token)
        .filter(|token| is_path_like_audit_ref(token))
        .collect::<Vec<_>>();
    refs.sort();
    refs.dedup();
    refs.truncate(8);
    refs
}

fn extract_target_refs(text: &str) -> HashSet<String> {
    extract_topic_refs(text).into_iter().collect()
}

fn extract_topic_terms(text: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "the", "and", "with", "from", "that", "this", "into", "then", "than", "tool", "call",
        "please", "report", "here", "check", "verify", "about", "because", "using", "still",
        "would", "could", "should", "there", "their", "issue", "issues", "audit", "review",
    ];

    let mut scores: HashMap<String, usize> = HashMap::new();
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != ':' && c != '/') {
        let Some(token) = clean_topic_token(raw) else {
            continue;
        };
        if token.len() < 4 || STOP_WORDS.iter().any(|stop| stop == &token.as_str()) {
            continue;
        }
        *scores.entry(token).or_insert(0) += 1;
    }

    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked.truncate(10);
    ranked.into_iter().map(|(token, _)| token).collect()
}

fn clean_topic_token(raw: &str) -> Option<String> {
    normalize_audit_ref(raw)
}

fn normalize_topic_text(text: &str, max_terms: usize) -> String {
    extract_topic_terms(text)
        .into_iter()
        .take(max_terms)
        .collect::<Vec<_>>()
        .join(" ")
}

fn stable_delegation_retry_key(args_text: &str, audit_request_summary: &str) -> String {
    let refs = extract_explicit_audit_refs(args_text);
    if !refs.is_empty() {
        return refs.join(" | ");
    }

    let terms = extract_topic_terms(args_text);
    if !terms.is_empty() {
        return terms.into_iter().take(5).collect::<Vec<_>>().join(" ");
    }

    if !audit_request_summary.is_empty() {
        audit_request_summary.to_string()
    } else {
        normalize_topic_text(args_text, 8)
    }
}

fn infer_inspection_depth(
    tool_name: &str,
    args_text: &str,
    tool_result: Option<&str>,
    target_refs: &HashSet<String>,
) -> AuditInspectionDepth {
    let args_lower = args_text.to_lowercase();
    let result_lower = tool_result.unwrap_or("").to_lowercase();

    if let Some(result) = tool_result {
        if classify_audit_evidence(result) >= AuditEvidenceGrade::ConfigContent
            || (result_lower.contains("confirmed findings") && has_file_reference(result))
        {
            return AuditInspectionDepth::EvidenceBacked;
        }
    }

    if !target_refs.is_empty()
        || [
            "read ",
            "analyze ",
            "analyse ",
            "audit ",
            "examine ",
            "inspect ",
            "review ",
            "focus on",
        ]
        .iter()
        .any(|needle| args_lower.contains(needle))
        || matches!(tool_name, "read_file" | "grep_search")
    {
        AuditInspectionDepth::Targeted
    } else {
        AuditInspectionDepth::Discovery
    }
}

async fn force_tool_free_reflection(
    client: &reqwest::Client,
    url: &str,
    model_key: &str,
    messages: &[Value],
) -> Result<String, String> {
    let reflection_messages = {
        let mut copy = messages.to_vec();
        copy.push(json!({
            "role": "user",
            "content": "Pause tool use. Briefly summarize what you learned, what is still uncertain, and the smallest next step. Do not call any tools in this response."
        }));
        copy
    };
    let resp = client
        .post(url)
        .json(&json!({
            "model": model_key,
            "messages": reflection_messages,
            "stream": false,
        }))
        .send()
        .await
        .map_err(|e| format!("Reflection request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Reflection failed: {}",
            resp.text().await.unwrap_or_default()
        ));
    }
    let data: Value = resp
        .json()
        .await
        .map_err(|e| format!("Reflection parse error: {}", e))?;
    Ok(data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string())
}

async fn force_non_progress_summary(
    client: &reqwest::Client,
    url: &str,
    model_key: &str,
    messages: &[Value],
    grounded_audit_mode: bool,
    missing_scope_refs: &[String],
) -> String {
    let reflection_messages = {
        let mut copy = messages.to_vec();
        let missing_scope_note = if grounded_audit_mode && !missing_scope_refs.is_empty() {
            format!(
                "\n- Under `Coverage gaps`, explicitly list these requested scopes that remain uninspected: {}.",
                missing_scope_refs.join(", ")
            )
        } else {
            String::new()
        };
        copy.push(json!({
            "role": "user",
            "content": if grounded_audit_mode {
                format!("Tooling or delegation has stalled. Using only the concrete evidence already present in this conversation, produce a grounded audit report with these sections exactly: `Confirmed findings`, `Hypotheses / lower-confidence risks`, and `Coverage gaps`. Only keep findings as confirmed if they are directly supported by code or config already shown. Downgrade unsupported concerns into hypotheses. Every hypothesis must cite a concrete file/config or observed behavior already in context, or explicitly state what evidence is still missing to confirm it. Remove generic dependency-presence or stack-template risks. Do not call any tools.{}", missing_scope_note)
            } else {
                "Stop now because progress has stalled. In 3 short bullets, state: 1) what is known, 2) what blocked progress, 3) the smallest remaining useful next step. Do not call any tools.".to_string()
            }
        }));
        copy
    };

    match client
        .post(url)
        .json(&json!({
            "model": model_key,
            "messages": reflection_messages,
            "stream": false,
        }))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
            Ok(data) => {
                let text = data["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if text.is_empty() {
                    if grounded_audit_mode {
                        fallback_non_progress_summary(true, missing_scope_refs)
                    } else {
                        fallback_non_progress_summary(false, missing_scope_refs)
                    }
                } else {
                    text
                }
            }
            Err(_) => fallback_non_progress_summary(grounded_audit_mode, missing_scope_refs),
        },
        _ => fallback_non_progress_summary(grounded_audit_mode, missing_scope_refs),
    }
}

fn classify_non_progress_tool_result(tool_name: &str, result: &str) -> Option<String> {
    let lower = result.to_lowercase();
    if lower.contains("not recognised by this mcp server") {
        Some(format!("{}::unknown_tool", tool_name))
    } else if lower.contains("use glob_search first") {
        Some(format!("{}::missing_glob_scope", tool_name))
    } else if lower.contains("was not returned by your latest glob_search") {
        Some(format!("{}::glob_miss", tool_name))
    } else if lower.contains("requires a glob filter") {
        Some(format!("{}::missing_glob_filter", tool_name))
    } else if lower.contains("error calling agent:") {
        Some(format!("{}::agent_error", tool_name))
    } else if lower.contains("[internal tool dispatch failed:") {
        Some(format!("{}::internal_dispatch_failure", tool_name))
    } else if lower.contains("[tool call failed:") || lower.contains("[tool returned no text]") {
        Some(format!("{}::transport_or_empty", tool_name))
    } else {
        None
    }
}

async fn dispatch_tool_call(
    tool_state: &McpState,
    tool_name: &str,
    args: &Value,
    caller_agent_id: &str,
) -> (bool, String) {
    let result = handle_tool_call(tool_name, args, tool_state, Some(caller_agent_id)).await;
    (
        result["isError"].as_bool().unwrap_or(false),
        result["content"][0]["text"]
            .as_str()
            .unwrap_or("[tool returned no text]")
            .to_string(),
    )
}

fn has_concrete_repo_evidence(messages: &[Value]) -> bool {
    audit_evidence_grade(messages) >= AuditEvidenceGrade::CommandOnly
}

fn fallback_non_progress_summary(grounded_audit_mode: bool, missing_scope_refs: &[String]) -> String {
    if grounded_audit_mode {
        let extra_gap = if missing_scope_refs.is_empty() {
            String::new()
        } else {
            format!(
                "\n- Requested but not inspected: {}.",
                missing_scope_refs.join(", ")
            )
        };
        format!("Confirmed findings\n- None supported strongly enough to promote after tooling stalled.\n\nHypotheses / lower-confidence risks\n- Some concerns may remain, but the current conversation does not contain enough direct code/config evidence to confirm them.\n\nCoverage gaps\n- Tooling stalled before full inspection. Additional direct reads would be needed to upgrade any hypothesis into a confirmed finding.{}", extra_gap)
    } else {
        "Stopping early due to repeated non-progress. Known facts were collected, but the agent kept retrying without advancing. The smallest next step is to retry with a narrower task or inspect the blocked area directly.".to_string()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AuditEvidenceGrade {
    Inferred,
    CommandOnly,
    ConfigContent,
    CodeContent,
}

fn has_usable_evidence(messages: &[Value], grounded_audit_mode: bool) -> bool {
    if grounded_audit_mode {
        audit_evidence_grade(messages) >= AuditEvidenceGrade::ConfigContent
    } else {
        has_concrete_repo_evidence(messages)
    }
}

fn should_force_audit_synthesis(
    messages: &[Value],
    covered_topics: &[CoveredAuditTopic],
    coverage_manifest: &[AuditCoverageEntry],
    redundant_audit_retries: usize,
    config: &RedundancyDetectionConfig,
) -> bool {
    if redundant_audit_retries > usize::from(config.max_redundant_audit_retries) {
        return true;
    }

    let completed_reports = count_completed_audit_reports(messages);
    let issue_like_reports = count_issue_like_audit_reports(messages);
    let has_gaps = messages.iter().any(|message| {
        message["role"].as_str() == Some("tool")
            && message["content"]
                .as_str()
                .unwrap_or("")
                .to_lowercase()
                .contains("coverage gaps")
    });
    let targeted_topics = covered_topics
        .iter()
        .filter(|topic| topic.inspection_depth >= AuditInspectionDepth::Targeted)
        .count();
    let evidence_backed_topics = covered_topics
        .iter()
        .filter(|topic| topic.inspection_depth >= AuditInspectionDepth::EvidenceBacked)
        .count();
    let unresolved_manifest_entries =
        unresolved_audit_coverage_manifest_entries(coverage_manifest, None).len();

    ((completed_reports >= 3 && targeted_topics >= 3)
        || (completed_reports >= 2 && issue_like_reports >= 1 && has_gaps)
        || (evidence_backed_topics >= 3 && has_gaps))
        && (coverage_manifest.is_empty()
            || unresolved_manifest_entries == 0
            || has_gaps)
}

fn count_completed_audit_reports(messages: &[Value]) -> usize {
    messages
        .iter()
        .filter(|message| message["role"].as_str() == Some("tool"))
        .filter(|message| {
            let lower = message["content"].as_str().unwrap_or("").to_lowercase();
            lower.contains("confirmed findings")
                && lower.contains("coverage gaps")
        })
        .count()
}

fn count_issue_like_audit_reports(messages: &[Value]) -> usize {
    messages
        .iter()
        .filter(|message| message["role"].as_str() == Some("tool"))
        .filter(|message| {
            let content = message["content"].as_str().unwrap_or("");
            audit_report_has_issue_like_finding(content)
        })
        .count()
}

fn audit_evidence_grade(messages: &[Value]) -> AuditEvidenceGrade {
    messages
        .iter()
        .filter(|message| message["role"].as_str() == Some("tool"))
        .map(|message| classify_audit_evidence(message["content"].as_str().unwrap_or("")))
        .max()
        .unwrap_or(AuditEvidenceGrade::Inferred)
}

fn classify_audit_evidence(content: &str) -> AuditEvidenceGrade {
    let lower = content.to_lowercase();
    if content.contains("#[tauri::command]")
        || content.contains("pub(crate)")
        || content.contains("async fn ")
        || content.contains("fn ")
        || content.contains("impl ")
        || lower.contains("use std::")
        || lower.contains("match ")
    {
        AuditEvidenceGrade::CodeContent
    } else if content.contains("\"$schema\"")
        || content.contains("\"permissions\"")
        || content.contains("\"scripts\"")
        || content.contains("\"devDependencies\"")
        || content.contains("[package]")
        || content.contains("[dependencies]")
        || lower.contains("core:default")
        || lower.contains("\"csp\"")
    {
        AuditEvidenceGrade::ConfigContent
    } else if lower.contains("src-tauri/")
        || lower.contains("package.json")
        || lower.contains("workspace.rs")
        || lower.contains("models.rs")
        || lower.contains("src-tauri/src")
    {
        AuditEvidenceGrade::CommandOnly
    } else {
        AuditEvidenceGrade::Inferred
    }
}

fn audit_response_needs_rewrite(
    response: &str,
    evidence_grade: AuditEvidenceGrade,
) -> bool {
    let lower = response.to_lowercase();
    if response.trim().is_empty() {
        return true;
    }
    if !lower.contains("confirmed findings") || !lower.contains("hypotheses") {
        return true;
    }
    if contains_severity_label(&lower) && evidence_grade < AuditEvidenceGrade::ConfigContent {
        return true;
    }
    if contains_speculation_marker(&lower) && !lower.contains("hypotheses") {
        return true;
    }
    if confirmed_findings_need_rewrite(response) {
        return true;
    }
    if hypotheses_need_rewrite(response) {
        return true;
    }
    false
}

fn contains_severity_label(lower: &str) -> bool {
    ["critical", "high severity", "medium severity", "low severity", "severity:"]
        .iter()
        .any(|marker| lower.contains(marker))
}

fn contains_speculation_marker(lower: &str) -> bool {
    [
        "if an attacker",
        "could lead to",
        "might allow",
        "potentially",
        "possible",
        "may be vulnerable",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

async fn rewrite_grounded_audit_response(
    client: &reqwest::Client,
    url: &str,
    model_key: &str,
    messages: &[Value],
    candidate: &str,
) -> String {
    let rewrite_messages = {
        let mut copy = messages.to_vec();
        copy.push(json!({
            "role": "user",
            "content": format!(
                "Rewrite the audit answer below using only the evidence already present in this conversation.\n\nRules:\n- Use exactly these sections: `Confirmed findings`, `Hypotheses / lower-confidence risks`, `Coverage gaps`.\n- A confirmed finding must be directly supported by code or config already shown.\n- Keep only actual issues in `Confirmed findings`; move neutral observations, mitigations, architecture facts, and absence-of-risk statements out of that section.\n- Downgrade any unsupported concern into hypotheses.\n- Every hypothesis must cite a concrete file/config or observed behavior already in context, or explicitly state what evidence is still missing to confirm it.\n- Remove dependency-presence, stack-template, or generic speculative risks that are not tied to observed behavior.\n- Do not invent new evidence.\n- Do not label anything `High` or `Critical` without direct code/config support.\n\nCandidate answer:\n{}",
                candidate
            )
        }));
        copy
    };

    match client
        .post(url)
        .json(&json!({
            "model": model_key,
            "messages": rewrite_messages,
            "stream": false,
        }))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
            Ok(data) => data["choices"][0]["message"]["content"]
                .as_str()
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
                .unwrap_or_else(|| candidate.to_string()),
            Err(_) => candidate.to_string(),
        },
        _ => candidate.to_string(),
    }
}

fn confirmed_findings_need_rewrite(response: &str) -> bool {
    let confirmed = extract_audit_section(
        response,
        "confirmed findings",
        &["hypotheses / lower-confidence risks", "coverage gaps"],
    );
    if confirmed.trim().is_empty() {
        return false;
    }

    confirmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| {
            line.starts_with('-')
                || line.starts_with('*')
                || line.starts_with("1.")
                || line.starts_with("2.")
                || line.starts_with("3.")
                || line.starts_with("####")
        })
        .any(|line| {
            let lower = line.to_lowercase();
            is_non_issue_observation(&lower) || !is_issue_like_statement(&lower)
        })
}

fn hypotheses_need_rewrite(response: &str) -> bool {
    let hypotheses = extract_audit_section(
        response,
        "hypotheses / lower-confidence risks",
        &["coverage gaps"],
    );
    if hypotheses.trim().is_empty() {
        return false;
    }

    hypotheses
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| {
            line.starts_with('-')
                || line.starts_with('*')
                || line.starts_with("1.")
                || line.starts_with("2.")
                || line.starts_with("3.")
        })
        .any(hypothesis_line_needs_rewrite)
}

fn extract_audit_section(response: &str, start: &str, end_markers: &[&str]) -> String {
    let lower = response.to_lowercase();
    let Some(start_idx) = lower.find(start) else {
        return String::new();
    };
    let content_start = response[start_idx..]
        .find('\n')
        .map(|offset| start_idx + offset + 1)
        .unwrap_or(response.len());
    let end_idx = end_markers
        .iter()
        .filter_map(|marker| lower[content_start..].find(marker).map(|idx| content_start + idx))
        .min()
        .unwrap_or(response.len());
    response[content_start..end_idx].trim().to_string()
}

fn audit_report_has_issue_like_finding(content: &str) -> bool {
    let confirmed = extract_audit_section(
        content,
        "confirmed findings",
        &["hypotheses / lower-confidence risks", "coverage gaps"],
    );
    confirmed
        .lines()
        .map(str::trim)
        .any(|line| {
            let lower = line.to_lowercase();
            !lower.is_empty()
                && !is_non_issue_observation(&lower)
                && is_issue_like_statement(&lower)
        })
}

fn hypothesis_line_needs_rewrite(line: &str) -> bool {
    let lower = line.to_lowercase();
    let has_anchor = has_file_reference(line)
        || hypothesis_mentions_observed_behavior(&lower)
        || hypothesis_mentions_missing_proof(&lower);
    let template = is_template_hypothesis(&lower);
    let dependency_only = is_dependency_presence_hypothesis(&lower);
    (template || dependency_only) && !has_anchor
}

fn is_issue_like_statement(lower: &str) -> bool {
    [
        "injection",
        "xss",
        "csp",
        "race condition",
        "unbounded",
        "exhaustion",
        "overflow",
        "leak",
        "bypass",
        "unsafe",
        "unauthorized",
        "path traversal",
        "resource exhaustion",
        "disk exhaustion",
        "out-of-memory",
        "oom",
        "missing content security policy",
        "withglobaltauri",
        "shell metacharacter",
        "sensitive",
        "exposed",
        "vulnerab",
        "risk",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn is_non_issue_observation(lower: &str) -> bool {
    [
        "xss mitigation",
        "absence of dangerous",
        "no explicitly defined broad scopes",
        "no usage of",
        "no broad scopes",
        "project is",
        "backend is built",
        "frontend architecture includes",
        "contains modules",
        "hardcoded network port",
        "loaded models",
        "uses a hardcoded port",
        "no explicitly defined",
        "mitigation",
    ]
        .iter()
        .any(|marker| lower.contains(marker))
}

fn is_template_hypothesis(lower: &str) -> bool {
    [
        "presence of ",
        "possible risk",
        "potential risk",
        "could lead to",
        "could allow",
        "could be vulnerable if",
        "may allow",
        "may be vulnerable",
        "if later",
        "if exposed",
        "if combined",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn is_dependency_presence_hypothesis(lower: &str) -> bool {
    [
        "dependency",
        "crate",
        "package",
        "reqwest",
        "axum",
        "rfd",
        "serde",
        "tokio",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        && !hypothesis_mentions_observed_behavior(lower)
}

fn hypothesis_mentions_missing_proof(lower: &str) -> bool {
    [
        "insufficient evidence",
        "not enough evidence",
        "unable to confirm",
        "would need to inspect",
        "would need to verify",
        "requires further inspection",
        "missing proof",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn hypothesis_mentions_observed_behavior(lower: &str) -> bool {
    [
        "uses ",
        "calls ",
        "passes ",
        "spawns ",
        "reads ",
        "writes ",
        "normalizes ",
        "sets ",
        "disables ",
        "allows ",
        "constructs ",
        "joins ",
        "matches ",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn is_unusable_delegation_result(result: &str) -> bool {
    if result.contains("\"response\":\"\"") || result.contains("\"status\":\"queued\"") {
        return true;
    }

    if let Ok(payload) = serde_json::from_str::<Value>(result) {
        if payload["status"].as_str() == Some("queued") {
            return true;
        }

        if let Some(response) = payload["response"].as_str() {
            return response.trim().is_empty() || is_progress_only_response(response);
        }
    }

    false
}

fn extract_delegation_response_text(result: &str) -> Option<String> {
    if let Ok(payload) = serde_json::from_str::<Value>(result) {
        return payload["response"]
            .as_str()
            .map(|response| response.trim().to_string())
            .filter(|response| !response.is_empty());
    }
    None
}

fn is_weak_audit_delegation_result(
    result: &str,
    covered_topics: &[CoveredAuditTopic],
) -> bool {
    let Some(response) = extract_delegation_response_text(result) else {
        return false;
    };

    if is_progress_only_response(&response) || is_low_value_audit_response(&response) {
        return true;
    }

    let response_refs = extract_explicit_audit_refs(&response)
        .into_iter()
        .collect::<HashSet<_>>();
    let seen_refs = covered_topics
        .iter()
        .flat_map(|topic| topic.refs.iter().chain(topic.target_refs.iter()))
        .cloned()
        .collect::<HashSet<_>>();
    let has_new_refs = response_refs
        .iter()
        .any(|reference| !seen_refs.iter().any(|seen| audit_ref_entry_matches(reference, seen)));
    let has_new_issue = audit_report_has_issue_like_finding(&response);
    let has_evidence = classify_audit_evidence(&response) >= AuditEvidenceGrade::ConfigContent;
    let has_specific_gap = extract_audit_section(&response, "coverage gaps", &[])
        .lines()
        .map(str::trim)
        .any(|line| has_file_reference(line));

    !(has_new_refs || has_new_issue || has_evidence || has_specific_gap)
}

fn is_progress_only_response(response: &str) -> bool {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return true;
    }

    let lower = trimmed.to_lowercase();
    [
        "i asked another agent",
        "i have asked another agent",
        "i delegated",
        "i have delegated",
        "i'm waiting",
        "i am waiting",
        "currently waiting",
        "wait, i am currently waiting",
        "status update request",
        "information plan",
        "next step:",
        "let me check",
        "i will now",
        "i have already issued",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(crate) async fn send_chat_completion_streaming(
    model_key: &str,
    system_prompt: &str,
    message: &str,
    agent_id: &str,
    on_event: &Channel<StreamEvent>,
    history: &[Value],
    context_limit: Option<u64>,
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", lm_base_url());
    let budget = compute_context_budget(system_prompt, history, message, context_limit, 0);
    let effective_system_prompt = with_context_budget(system_prompt, budget);
    let mut messages = vec![];
    if !effective_system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": effective_system_prompt}));
    }
    for h in history {
        messages.push(h.clone());
    }
    messages.push(json!({"role": "user", "content": message}));

    let body = json!({ "model": model_key, "messages": messages, "stream": true });
    let client = reqwest::Client::new();
    let mut resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Chat request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Chat failed: {}",
            resp.text().await.unwrap_or_default()
        ));
    }

    let mut full_content = String::new();
    let mut buffer = String::new();

    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("Stream read error: {}", e))?
    {
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim_end().to_string();
            buffer = buffer[pos + 1..].to_string();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    return Ok(full_content);
                }
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
