use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use crate::agent_config::{
    resolve_audit_behavior_config, AuditEvidenceGrade, FinalizerConfig, RedundancyDetectionConfig,
    ResolvedAuditBehaviorConfig, MAX_SEMANTIC_SIMILARITY_THRESHOLD,
    MIN_SEMANTIC_SIMILARITY_THRESHOLD,
};
use crate::event_sink::SharedEventSink;
use crate::helpers::{
    audit_response_acknowledges_refs, compute_context_budget, extract_explicit_audit_refs,
    has_file_reference, is_manager_blocked_tool, is_manager_only_tool, is_path_like_audit_ref,
    lm_base_url, normalize_audit_ref, resolve_active_behaviors, trim_history_to_budget,
    with_context_budget,
};
use crate::mcp::{handle_tool_call, McpState};
use crate::models::create_embeddings;
use crate::runs::{
    emit_run_event, run_budget_applies, PausedRunState, RunDossier, RunDossierWorkerOutcome,
    RunLimitHit, RunWindowUsage,
};
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

#[derive(Clone, Default)]
struct StreamUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    reasoning_output_tokens: Option<u64>,
    tokens_per_second: Option<f64>,
    time_to_first_token_seconds: Option<f64>,
}

#[derive(Default)]
struct ToolCallDelta {
    id: String,
    tool_type: String,
    function_name: String,
    function_arguments: String,
}

struct StreamedAssistantTurn {
    assistant_message: Value,
    content: String,
    usage: StreamUsage,
}

fn parse_stream_usage(chunk: &Value) -> StreamUsage {
    let usage = chunk.get("usage");
    let stats = chunk.get("result").and_then(|result| result.get("stats"));

    let input_tokens = usage
        .and_then(|value| value.get("prompt_tokens").and_then(Value::as_u64))
        .or_else(|| usage.and_then(|value| value.get("input_tokens").and_then(Value::as_u64)))
        .or_else(|| stats.and_then(|value| value.get("input_tokens").and_then(Value::as_u64)));

    let output_tokens = usage
        .and_then(|value| value.get("completion_tokens").and_then(Value::as_u64))
        .or_else(|| usage.and_then(|value| value.get("output_tokens").and_then(Value::as_u64)))
        .or_else(|| {
            stats.and_then(|value| value.get("total_output_tokens").and_then(Value::as_u64))
        });

    let reasoning_output_tokens = usage
        .and_then(|value| value.get("reasoning_output_tokens").and_then(Value::as_u64))
        .or_else(|| {
            stats.and_then(|value| value.get("reasoning_output_tokens").and_then(Value::as_u64))
        });

    let tokens_per_second =
        stats.and_then(|value| value.get("tokens_per_second").and_then(Value::as_f64));
    let time_to_first_token_seconds = stats.and_then(|value| {
        value
            .get("time_to_first_token_seconds")
            .and_then(Value::as_f64)
    });

    StreamUsage {
        input_tokens,
        output_tokens,
        reasoning_output_tokens,
        tokens_per_second,
        time_to_first_token_seconds,
    }
}

fn emit_generation_metrics(
    on_event: &SharedEventSink,
    run_id: &str,
    agent_id: &str,
    stage: &str,
    usage: &StreamUsage,
    estimated_output_tokens: u64,
) {
    let _ = on_event.send(StreamEvent::AgentMetrics {
        run_id: run_id.to_string(),
        agent_id: agent_id.to_string(),
        stage: stage.to_string(),
        estimated_output_tokens,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        reasoning_output_tokens: usage.reasoning_output_tokens,
        tokens_per_second: usage.tokens_per_second,
        time_to_first_token_seconds: usage.time_to_first_token_seconds,
    });
}

fn estimated_stream_output_tokens(text: &str) -> u64 {
    if text.trim().is_empty() {
        0
    } else {
        crate::helpers::estimate_text_tokens(text)
    }
}

async fn stream_chat_completion_turn(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
    run_id: &str,
    agent_id: &str,
    on_event: &SharedEventSink,
) -> Result<StreamedAssistantTurn, String> {
    let mut resp = client
        .post(url)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "LLM error: {}",
            resp.text().await.unwrap_or_default()
        ));
    }

    let started_at = Instant::now();
    let mut first_delta_at: Option<std::time::Duration> = None;
    let mut full_content = String::new();
    let mut buffer = String::new();
    let mut assistant_role = "assistant".to_string();
    let mut tool_calls: Vec<ToolCallDelta> = vec![];
    let mut usage = StreamUsage::default();

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
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data.trim() == "[DONE]" {
                let estimated_output_tokens = estimated_stream_output_tokens(&full_content);
                if usage.output_tokens.is_none() && !full_content.is_empty() {
                    usage.output_tokens = Some(estimated_output_tokens);
                }
                if usage.time_to_first_token_seconds.is_none() {
                    usage.time_to_first_token_seconds =
                        first_delta_at.map(|elapsed| elapsed.as_secs_f64());
                }
                if usage.tokens_per_second.is_none() && !full_content.is_empty() {
                    let elapsed = started_at.elapsed().as_secs_f64();
                    if elapsed > 0.0 {
                        usage.tokens_per_second =
                            usage.output_tokens.map(|tokens| tokens as f64 / elapsed);
                    }
                }
                let tool_calls_json: Vec<Value> = tool_calls
                    .into_iter()
                    .map(|tool_call| {
                        json!({
                            "id": tool_call.id,
                            "type": if tool_call.tool_type.is_empty() { "function".to_string() } else { tool_call.tool_type },
                            "function": {
                                "name": tool_call.function_name,
                                "arguments": tool_call.function_arguments,
                            }
                        })
                    })
                    .collect();
                let assistant_message = if tool_calls_json.is_empty() {
                    json!({
                        "role": assistant_role,
                        "content": full_content,
                    })
                } else {
                    json!({
                        "role": assistant_role,
                        "content": if full_content.is_empty() { Value::Null } else { json!(full_content) },
                        "tool_calls": tool_calls_json,
                    })
                };
                return Ok(StreamedAssistantTurn {
                    assistant_message,
                    content: full_content,
                    usage,
                });
            }

            let parsed =
                serde_json::from_str::<Value>(data).map_err(|e| format!("Parse error: {}", e))?;

            let parsed_usage = parse_stream_usage(&parsed);
            if parsed_usage.input_tokens.is_some() {
                usage.input_tokens = parsed_usage.input_tokens;
            }
            if parsed_usage.output_tokens.is_some() {
                usage.output_tokens = parsed_usage.output_tokens;
            }
            if parsed_usage.reasoning_output_tokens.is_some() {
                usage.reasoning_output_tokens = parsed_usage.reasoning_output_tokens;
            }
            if parsed_usage.tokens_per_second.is_some() {
                usage.tokens_per_second = parsed_usage.tokens_per_second;
            }
            if parsed_usage.time_to_first_token_seconds.is_some() {
                usage.time_to_first_token_seconds = parsed_usage.time_to_first_token_seconds;
            }

            let choice = parsed
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first());
            let delta = choice.and_then(|value| value.get("delta"));

            if let Some(role) = delta
                .and_then(|value| value.get("role"))
                .and_then(Value::as_str)
            {
                assistant_role = role.to_string();
            }

            if let Some(content) = delta
                .and_then(|value| value.get("content"))
                .and_then(Value::as_str)
            {
                if !content.is_empty() {
                    if first_delta_at.is_none() {
                        first_delta_at = Some(started_at.elapsed());
                        if usage.time_to_first_token_seconds.is_none() {
                            usage.time_to_first_token_seconds =
                                first_delta_at.map(|elapsed| elapsed.as_secs_f64());
                        }
                    }
                    full_content.push_str(content);
                    let _ = on_event.send(StreamEvent::Token {
                        run_id: run_id.to_string(),
                        agent_id: agent_id.to_string(),
                        content: content.to_string(),
                    });
                    emit_generation_metrics(
                        on_event,
                        run_id,
                        agent_id,
                        "streaming",
                        &usage,
                        estimated_stream_output_tokens(&full_content),
                    );
                }
            }

            if let Some(tool_call_deltas) = delta
                .and_then(|value| value.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for tool_call_delta in tool_call_deltas {
                    let idx = tool_call_delta
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or(tool_calls.len() as u64) as usize;
                    if idx >= tool_calls.len() {
                        tool_calls.resize_with(idx + 1, ToolCallDelta::default);
                    }
                    if let Some(id) = tool_call_delta.get("id").and_then(Value::as_str) {
                        tool_calls[idx].id = id.to_string();
                    }
                    if let Some(tool_type) = tool_call_delta.get("type").and_then(Value::as_str) {
                        tool_calls[idx].tool_type = tool_type.to_string();
                    }
                    if let Some(function) = tool_call_delta.get("function") {
                        if let Some(name) = function.get("name").and_then(Value::as_str) {
                            tool_calls[idx].function_name.push_str(name);
                        }
                        if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                            tool_calls[idx].function_arguments.push_str(arguments);
                        }
                    }
                }
            }
        }
    }

    Err("Stream ended before completion.".to_string())
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
    let trimmed_history = trim_history_to_budget(system_prompt, history, message, context_limit, 0);
    let budget = compute_context_budget(system_prompt, &trimmed_history, message, context_limit, 0);
    let effective_system_prompt = with_context_budget(system_prompt, budget);
    let mut messages = vec![];
    if !effective_system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": effective_system_prompt}));
    }
    for h in &trimmed_history {
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

pub(crate) const RUN_PAUSED_ERROR: &str = "__run_paused__";
/// Returned when a budget-hit summary was generated and streamed: routing must stop without re-routing the summary text.
pub(crate) const RUN_BUDGET_ENDED_ERROR: &str = "__run_budget_ended__";
/// Returned when a user-requested cancellation should stop routing after the current turn/tool step.
pub(crate) const RUN_CANCELLED_ERROR: &str = "__run_cancelled__";

pub(crate) fn run_cancel_requested(tool_state: &McpState, run_id: &str) -> bool {
    tool_state
        .active_runs
        .lock()
        .unwrap()
        .get(run_id)
        .map(|run| run.cancelled)
        .unwrap_or(true)
}

pub(crate) fn finalize_cancelled_run(tool_state: &McpState, run_id: &str) -> Result<(), String> {
    let maybe_run = tool_state.active_runs.lock().unwrap().remove(run_id);
    let Some(run) = maybe_run else {
        return Ok(());
    };
    let _ = run.channel.send(StreamEvent::RunCancelled {
        run_id: run_id.to_string(),
        message: "Run cancelled by user.".to_string(),
    });
    let _ = run.channel.send(StreamEvent::Done {
        run_id: run_id.to_string(),
    });
    Ok(())
}

fn sync_active_run_behaviors(
    tool_state: &McpState,
    run_id: &str,
    active_behaviors: &HashSet<String>,
) {
    if let Some(run) = tool_state.active_runs.lock().unwrap().get_mut(run_id) {
        run.active_behaviors = active_behaviors.clone();
    }
}

fn check_run_budget_for_llm_call(
    tool_state: &McpState,
    run_id: &str,
    active_behaviors: &HashSet<String>,
) -> Result<(), RunLimitHit> {
    let mut runs = tool_state.active_runs.lock().unwrap();
    let Some(run) = runs.get_mut(run_id) else {
        return Ok(());
    };
    run.active_behaviors = active_behaviors.clone();
    // Always count so the UI can display real-time usage regardless of behavior scope.
    run.usage.llm_calls += 1;
    // Only enforce limits when the budget is scoped to an active behavior.
    if !run_budget_applies(&run.budgets, active_behaviors) {
        return Ok(());
    }
    let elapsed_seconds = run.window_started_at.elapsed().as_secs();
    if elapsed_seconds >= run.budgets.wall_clock_seconds_per_window {
        return Err(RunLimitHit {
            kind: "wall_clock_seconds_per_window".to_string(),
            limit: run.budgets.wall_clock_seconds_per_window,
            observed: elapsed_seconds,
        });
    }
    if u64::from(run.usage.llm_calls) > u64::from(run.budgets.llm_calls_per_window) {
        return Err(RunLimitHit {
            kind: "llm_calls_per_window".to_string(),
            limit: u64::from(run.budgets.llm_calls_per_window),
            observed: u64::from(run.usage.llm_calls),
        });
    }
    Ok(())
}

fn check_run_budget_for_tool_call(
    tool_state: &McpState,
    run_id: &str,
    active_behaviors: &HashSet<String>,
    tool_name: &str,
) -> Result<(), RunLimitHit> {
    let mut runs = tool_state.active_runs.lock().unwrap();
    let Some(run) = runs.get_mut(run_id) else {
        return Ok(());
    };
    run.active_behaviors = active_behaviors.clone();
    // Always count.
    run.usage.tool_calls += 1;
    let is_spawn = matches!(tool_name, "spawn_agent" | "Agent" | "send_parallel");
    if is_spawn {
        run.usage.spawned_agents += 1;
    }
    // Only enforce when budget applies.
    if !run_budget_applies(&run.budgets, active_behaviors) {
        return Ok(());
    }
    let elapsed_seconds = run.window_started_at.elapsed().as_secs();
    if elapsed_seconds >= run.budgets.wall_clock_seconds_per_window {
        return Err(RunLimitHit {
            kind: "wall_clock_seconds_per_window".to_string(),
            limit: run.budgets.wall_clock_seconds_per_window,
            observed: elapsed_seconds,
        });
    }
    if u64::from(run.usage.tool_calls) > u64::from(run.budgets.tool_calls_per_window) {
        return Err(RunLimitHit {
            kind: "tool_calls_per_window".to_string(),
            limit: u64::from(run.budgets.tool_calls_per_window),
            observed: u64::from(run.usage.tool_calls),
        });
    }
    if is_spawn
        && u64::from(run.usage.spawned_agents) > u64::from(run.budgets.spawned_agents_per_window)
    {
        return Err(RunLimitHit {
            kind: "spawned_agents_per_window".to_string(),
            limit: u64::from(run.budgets.spawned_agents_per_window),
            observed: u64::from(run.usage.spawned_agents),
        });
    }
    Ok(())
}

fn record_run_streamed_tokens(
    tool_state: &McpState,
    run_id: &str,
    active_behaviors: &HashSet<String>,
    streamed_tokens: u64,
) -> Result<(), RunLimitHit> {
    let mut runs = tool_state.active_runs.lock().unwrap();
    let Some(run) = runs.get_mut(run_id) else {
        return Ok(());
    };
    run.active_behaviors = active_behaviors.clone();
    // Always count.
    run.usage.streamed_tokens += streamed_tokens;
    // Only enforce when budget applies.
    if !run_budget_applies(&run.budgets, active_behaviors) {
        return Ok(());
    }
    if run.usage.streamed_tokens > run.budgets.streamed_tokens_per_window {
        return Err(RunLimitHit {
            kind: "streamed_tokens_per_window".to_string(),
            limit: run.budgets.streamed_tokens_per_window,
            observed: run.usage.streamed_tokens,
        });
    }
    Ok(())
}

fn capture_run_usage(tool_state: &McpState, run_id: &str) -> RunWindowUsage {
    tool_state
        .active_runs
        .lock()
        .unwrap()
        .get(run_id)
        .map(|run| run.usage.clone())
        .unwrap_or_default()
}

fn pause_run_for_confirmation(
    tool_state: &McpState,
    run_id: &str,
    paused_state: PausedRunState,
) -> Result<(), String> {
    {
        let mut runs = tool_state.active_runs.lock().unwrap();
        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| format!("Run '{}' not found.", run_id))?;
        run.waiting_confirmation = true;
        run.paused = Some(paused_state.clone());
    }
    emit_run_event(
        &tool_state.active_runs,
        run_id,
        StreamEvent::RunLimitReached {
            run_id: run_id.to_string(),
            kind: paused_state.limit_hit.kind.clone(),
            limit: paused_state.limit_hit.limit,
            observed: paused_state.limit_hit.observed,
        },
    )?;
    emit_run_event(
        &tool_state.active_runs,
        run_id,
        StreamEvent::RunWaitingConfirmation {
            run_id: run_id.to_string(),
            message: "This review hit the configured soft budget. Continue to open a fresh budget window, or stop and keep the partial result.".to_string(),
        },
    )
}

fn emit_usage_update(tool_state: &McpState, run_id: &str, on_event: &SharedEventSink) {
    let snapshot = {
        let runs = tool_state.active_runs.lock().unwrap();
        runs.get(run_id)
            .map(|run| (run.usage.clone(), run.window_started_at.elapsed().as_secs()))
    };
    if let Some((usage, wall_clock_seconds)) = snapshot {
        let _ = on_event.send(StreamEvent::RunUsageUpdate {
            run_id: run_id.to_string(),
            llm_calls: usage.llm_calls,
            tool_calls: usage.tool_calls,
            spawned_agents: usage.spawned_agents,
            streamed_tokens: usage.streamed_tokens,
            embedding_calls: usage.embedding_calls,
            wall_clock_seconds,
        });
    }
}

/// Stores the manager agent's current conversation state so that budget-hit summarization
/// can always use the full manager context regardless of which agent triggers the limit.
fn update_manager_context(
    tool_state: &McpState,
    run_id: &str,
    agent_id: &str,
    model_key: &str,
    messages: &[Value],
) {
    let mut runs = tool_state.active_runs.lock().unwrap();
    if let Some(run) = runs.get_mut(run_id) {
        run.manager_agent_id = Some(agent_id.to_string());
        run.manager_model_key = Some(model_key.to_string());
        run.manager_messages = messages.to_vec();
    }
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let head: String = trimmed.chars().take(max_chars).collect();
        format!("{}...", head)
    }
}

fn dedupe_preserve_order(items: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for item in items {
        let normalized = item.trim().to_lowercase();
        if normalized.is_empty() || !seen.insert(normalized) {
            continue;
        }
        deduped.push(item.trim().to_string());
    }
    deduped
}

fn extract_path_like_snippets(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| {
            token.trim_matches(|ch: char| {
                matches!(
                    ch,
                    ',' | '.' | ':' | ';' | '(' | ')' | '[' | ']' | '`' | '"' | '\''
                )
            })
        })
        .filter(|token| {
            token.contains('/')
                && !token.starts_with("http")
                && token.len() <= 160
                && token.chars().any(|ch| ch.is_ascii_alphabetic())
        })
        .map(ToString::to_string)
        .collect()
}

fn extract_coverage_gap_candidates(text: &str) -> Vec<String> {
    dedupe_preserve_order(
        text.lines()
            .map(str::trim)
            .filter(|line| {
                let lower = line.to_lowercase();
                lower.contains("coverage gap")
                    || lower.contains("not inspect")
                    || lower.contains("uninspected")
                    || lower.contains("insufficient evidence")
                    || lower.contains("did not inspect")
            })
            .map(ToString::to_string)
            .collect(),
    )
}

fn build_worker_outcome(
    agent_id: &str,
    agent_name: &str,
    content: &str,
) -> RunDossierWorkerOutcome {
    let paragraphs = content
        .split("\n\n")
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    let summary = paragraphs
        .iter()
        .find(|paragraph| !paragraph.starts_with('#') && !paragraph.starts_with('*'))
        .copied()
        .unwrap_or_else(|| paragraphs.first().copied().unwrap_or(""))
        .to_string();

    let observed_evidence = dedupe_preserve_order(
        content
            .lines()
            .map(str::trim)
            .filter(|line| {
                let lower = line.to_lowercase();
                lower.starts_with("observed")
                    || lower.starts_with("- ") && extract_path_like_snippets(line).len() > 0
            })
            .map(ToString::to_string)
            .chain(extract_path_like_snippets(content))
            .take(8)
            .collect(),
    );

    let inferences = dedupe_preserve_order(
        content
            .lines()
            .map(str::trim)
            .filter(|line| {
                let lower = line.to_lowercase();
                lower.starts_with("inference")
                    || lower.contains("recommend")
                    || lower.contains("should ")
            })
            .map(ToString::to_string)
            .take(6)
            .collect(),
    );

    let coverage_gaps = extract_coverage_gap_candidates(content);

    RunDossierWorkerOutcome {
        agent_id: agent_id.to_string(),
        agent_name: agent_name.to_string(),
        summary: truncate_for_summary(&summary, 320),
        observed_evidence,
        inferences,
        coverage_gaps,
    }
}

fn compact_manager_draft(content: &str) -> String {
    let paragraphs = content
        .split("\n\n")
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .filter(|segment| !segment.starts_with("###") && !segment.starts_with("####"))
        .take(6)
        .map(|segment| truncate_for_summary(segment, 240))
        .collect::<Vec<_>>();
    paragraphs.join("\n\n")
}

fn refresh_dossier_caution_flags(dossier: &mut RunDossier) {
    let mut flags = dossier
        .caution_flags
        .iter()
        .filter(|flag| flag.as_str() == "Blocked commands occurred during this run.")
        .cloned()
        .collect::<Vec<_>>();

    let inspected_count = dossier.inspected_paths.len();
    if inspected_count <= 3 {
        flags.push("Inspected scope is narrow.".to_string());
    }
    if dossier.counts.broad_full_file_reads + dossier.counts.broad_directory_scans >= 3 {
        flags.push("Exploration leaned broad rather than targeted.".to_string());
    }
    if dossier.counts.dependency_or_generated_scans > 0 {
        flags.push("Dependency/build/generated paths were touched.".to_string());
    }
    if !dossier.coverage_gaps.is_empty() {
        flags.push("Coverage gaps remain.".to_string());
    }
    dossier.caution_flags = dedupe_preserve_order(flags);
}

pub(crate) fn record_agent_output_in_dossier(
    tool_state: &McpState,
    run_id: &str,
    agent_id: &str,
    agent_name: &str,
    is_manager: bool,
    content: &str,
) {
    let mut runs = tool_state.active_runs.lock().unwrap();
    let Some(run) = runs.get_mut(run_id) else {
        return;
    };
    if is_manager {
        run.dossier.manager_draft_summary = compact_manager_draft(content);
        run.dossier
            .coverage_gaps
            .extend(extract_coverage_gap_candidates(content));
    } else {
        let outcome = build_worker_outcome(agent_id, agent_name, content);
        run.dossier
            .coverage_gaps
            .extend(outcome.coverage_gaps.clone());
        run.dossier.worker_outcomes.push(outcome);
    }
    run.dossier.coverage_gaps = dedupe_preserve_order(run.dossier.coverage_gaps.clone());
    refresh_dossier_caution_flags(&mut run.dossier);
}

fn current_run_dossier(tool_state: &McpState, run_id: &str) -> Option<RunDossier> {
    tool_state
        .active_runs
        .lock()
        .unwrap()
        .get(run_id)
        .map(|run| run.dossier.clone())
}

pub(crate) fn emit_run_dossier_update(
    tool_state: &McpState,
    run_id: &str,
    on_event: &SharedEventSink,
) {
    if let Some(dossier) = current_run_dossier(tool_state, run_id) {
        let _ = on_event.send(StreamEvent::RunDossierUpdated {
            run_id: run_id.to_string(),
            dossier,
        });
    }
}

fn record_embedding_calls(tool_state: &McpState, run_id: &str, count: u32) {
    if count == 0 {
        return;
    }
    let mut runs = tool_state.active_runs.lock().unwrap();
    if let Some(run) = runs.get_mut(run_id) {
        run.usage.embedding_calls += count;
    }
}

enum BudgetLimitOutcome {
    /// A finalizer response was emitted — routing must stop without re-routing text.
    Summarized,
    /// Stop immediately — routing must stop.
    Stopped,
    /// Pause and wait for user confirmation (original behaviour).
    Paused,
}

fn finalizer_enabled_for_mode(config: &FinalizerConfig, mode: &str) -> bool {
    if !config.enabled {
        return false;
    }
    match mode {
        "budget_stop" => config.run_on_budget_stop,
        _ => config.run_on_completion,
    }
}

/// Handles a budget limit hit: finalize, stop, or pause based on config.
async fn resolve_budget_limit(
    tool_state: &McpState,
    run_id: &str,
    model_key: &str,
    _messages: &[Value],
    agent_id: &str,
    paused_state: PausedRunState,
    on_event: &SharedEventSink,
) -> Result<BudgetLimitOutcome, String> {
    let cfg = tool_state.agent_config.lock().unwrap().clone();
    let on_limit = cfg.run_budgets.on_limit.clone();
    match on_limit.as_str() {
        "summarize" if finalizer_enabled_for_mode(&cfg.finalizer, "budget_stop") => {
            let (final_text, final_model_key) = run_finalizer(
                tool_state,
                run_id,
                agent_id,
                model_key,
                "",
                &cfg.finalizer,
                "budget_stop",
            )
            .await;
            emit_run_dossier_update(tool_state, run_id, on_event);
            let _ = on_event.send(StreamEvent::FinalizerOutput {
                run_id: run_id.to_string(),
                agent_id: format!("finalizer::{}", agent_id),
                agent_name: cfg.finalizer.agent_name.clone(),
                source_agent_id: agent_id.to_string(),
                source_agent_name: agent_id.to_string(),
                content: final_text,
                mode: Some("budget_stop".to_string()),
                model_key: final_model_key,
            });
            Ok(BudgetLimitOutcome::Summarized)
        }
        "stop" => Ok(BudgetLimitOutcome::Stopped),
        _ => {
            // "pause" or "summarize" with disabled finalization
            pause_run_for_confirmation(tool_state, run_id, paused_state)?;
            emit_run_dossier_update(tool_state, run_id, on_event);
            Ok(BudgetLimitOutcome::Paused)
        }
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
    run_id: &str,
    agent_id: &str,
    tools: &[McpTool],
    allowed_tools: Option<&[String]>,
    allow_manager_tools: bool,
    require_delegation: bool,
    context_limit: Option<u64>,
    glob_ready: bool,
    tool_state: &McpState,
    inherited_behaviors: Option<&HashSet<String>>,
    on_event: &SharedEventSink,
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
    let (active_behaviors, behavior_embed_calls) = resolve_active_behaviors(
        message,
        inherited_behaviors,
        &behavior_triggers,
        &tool_state.behavior_trigger_cache,
    )
    .await;
    record_embedding_calls(tool_state, run_id, behavior_embed_calls);
    let audit_config = resolve_audit_behavior_config(&active_behaviors, &behavior_triggers);
    let audit_mode = audit_config.is_some();
    sync_active_run_behaviors(tool_state, run_id, &active_behaviors);
    tool_state
        .active_behavior_contexts
        .lock()
        .unwrap()
        .insert(agent_id.to_string(), active_behaviors.clone());
    let _behavior_context_guard = ActiveBehaviorContextGuard {
        state: tool_state,
        agent_id: agent_id.to_string(),
    };
    let audit_request_summary = summarize_audit_request(message);
    let budget_tools =
        visible_tools_for_agent(tools, allowed_tools, allow_manager_tools, glob_ready);
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
        &trim_history_to_budget(
            system_prompt,
            history,
            message,
            context_limit,
            tool_overhead,
        ),
        message,
        context_limit,
        tool_overhead,
    );
    let effective_system_prompt = with_context_budget(system_prompt, budget);
    let trimmed_history = trim_history_to_budget(
        system_prompt,
        history,
        message,
        context_limit,
        tool_overhead,
    );
    if !effective_system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": effective_system_prompt}));
    }
    for h in &trimmed_history {
        messages.push(h.clone());
    }
    messages.push(json!({"role": "user", "content": message}));

    // Separate clients: LLM calls can take minutes, MCP tool calls should be fast.
    let llm_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
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
    let mut coverage_manifest = if audit_config
        .as_ref()
        .map(|config| config.coverage_manifest.enabled)
        .unwrap_or(false)
    {
        initialize_audit_coverage_manifest(message)
    } else {
        Vec::new()
    };
    let non_progress_limit = audit_config
        .as_ref()
        .and_then(|config| config.non_progress.limit)
        .unwrap_or(4);

    for _ in 0u8..64 {
        // Keep manager context snapshot fresh so budget-hit summarization always has
        // the latest tool results, regardless of which agent triggers the limit.
        if allow_manager_tools {
            update_manager_context(tool_state, run_id, agent_id, model_key, &messages);
        }
        let redundancy_note = audit_runtime_note(
            audit_config.as_ref(),
            &covered_audit_topics,
            &coverage_manifest,
        );
        let mut request_messages = vec![];
        if let Some(config) = audit_config.as_ref() {
            if let Some(injection) = config.system_prompt_injection.as_deref() {
                request_messages.push(json!({
                    "role": "system",
                        "content": injection,
                }));
            }
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
        let _ = on_event.send(StreamEvent::AgentStatus {
            run_id: run_id.to_string(),
            agent_id: agent_id.to_string(),
            stage: "thinking".to_string(),
            detail: if tool_burst == 0 {
                "Generating response".to_string()
            } else {
                format!("Generating follow-up after {} tool call(s)", tool_burst)
            },
        });
        if let Err(limit_hit) = check_run_budget_for_llm_call(tool_state, run_id, &active_behaviors)
        {
            let paused_state = PausedRunState {
                agent_id: agent_id.to_string(),
                model_key: model_key.to_string(),
                system_prompt: system_prompt.to_string(),
                messages: messages.clone(),
                allowed_tools: allowed_tools.map(|tools| tools.to_vec()),
                allow_manager_tools,
                require_delegation,
                context_limit,
                glob_ready: glob_scope_ready,
                active_behaviors: active_behaviors.clone(),
                usage: capture_run_usage(tool_state, run_id),
                limit_hit,
            };
            match resolve_budget_limit(
                tool_state,
                run_id,
                model_key,
                &messages,
                agent_id,
                paused_state,
                on_event,
            )
            .await?
            {
                BudgetLimitOutcome::Summarized => return Err(RUN_BUDGET_ENDED_ERROR.to_string()),
                BudgetLimitOutcome::Stopped => return Err(RUN_BUDGET_ENDED_ERROR.to_string()),
                BudgetLimitOutcome::Paused => return Err(RUN_PAUSED_ERROR.to_string()),
            }
        }

        let body = json!({
            "model": model_key,
            "messages": request_messages,
            "tools": tool_defs,
            "stream": true,
            "stream_options": { "include_usage": true },
        });

        let streamed_turn =
            stream_chat_completion_turn(&llm_client, &url, &body, run_id, agent_id, on_event)
                .await?;
        if run_cancel_requested(tool_state, run_id) {
            return Err(RUN_CANCELLED_ERROR.to_string());
        }
        let assistant_msg = streamed_turn.assistant_message.clone();

        emit_generation_metrics(
            on_event,
            run_id,
            agent_id,
            "generated",
            &streamed_turn.usage,
            estimated_stream_output_tokens(&streamed_turn.content),
        );
        if let Err(limit_hit) = record_run_streamed_tokens(
            tool_state,
            run_id,
            &active_behaviors,
            estimated_stream_output_tokens(&streamed_turn.content),
        ) {
            let paused_state = PausedRunState {
                agent_id: agent_id.to_string(),
                model_key: model_key.to_string(),
                system_prompt: system_prompt.to_string(),
                messages: messages.clone(),
                allowed_tools: allowed_tools.map(|tools| tools.to_vec()),
                allow_manager_tools,
                require_delegation,
                context_limit,
                glob_ready: glob_scope_ready,
                active_behaviors: active_behaviors.clone(),
                usage: capture_run_usage(tool_state, run_id),
                limit_hit,
            };
            match resolve_budget_limit(
                tool_state,
                run_id,
                model_key,
                &messages,
                agent_id,
                paused_state,
                on_event,
            )
            .await?
            {
                BudgetLimitOutcome::Summarized => return Err(RUN_BUDGET_ENDED_ERROR.to_string()),
                BudgetLimitOutcome::Stopped => return Err(RUN_BUDGET_ENDED_ERROR.to_string()),
                BudgetLimitOutcome::Paused => return Err(RUN_PAUSED_ERROR.to_string()),
            }
        }

        emit_usage_update(tool_state, run_id, on_event);

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
                if run_cancel_requested(tool_state, run_id) {
                    return Err(RUN_CANCELLED_ERROR.to_string());
                }
                let call_id = tc["id"].as_str().unwrap_or("").to_string();
                let fn_name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let fn_args: Value =
                    serde_json::from_str(tc["function"]["arguments"].as_str().unwrap_or("{}"))
                        .unwrap_or(json!({}));
                let args_text = serde_json::to_string(&fn_args).unwrap_or_default();
                if is_delegation_tool(&fn_name) {
                    delegated = true;
                }
                if audit_config
                    .as_ref()
                    .map(|config| config.coverage_manifest.enabled)
                    .unwrap_or(false)
                {
                    extend_audit_coverage_manifest(&mut coverage_manifest, &args_text);
                }
                tool_burst += 1;

                // Notify frontend: tool about to be called
                if let Err(limit_hit) =
                    check_run_budget_for_tool_call(tool_state, run_id, &active_behaviors, &fn_name)
                {
                    let paused_state = PausedRunState {
                        agent_id: agent_id.to_string(),
                        model_key: model_key.to_string(),
                        system_prompt: system_prompt.to_string(),
                        messages: messages.clone(),
                        allowed_tools: allowed_tools.map(|tools| tools.to_vec()),
                        allow_manager_tools,
                        require_delegation,
                        context_limit,
                        glob_ready: glob_scope_ready,
                        active_behaviors: active_behaviors.clone(),
                        usage: capture_run_usage(tool_state, run_id),
                        limit_hit,
                    };
                    match resolve_budget_limit(
                        tool_state,
                        run_id,
                        model_key,
                        &messages,
                        agent_id,
                        paused_state,
                        on_event,
                    )
                    .await?
                    {
                        BudgetLimitOutcome::Summarized => {
                            return Err(RUN_BUDGET_ENDED_ERROR.to_string())
                        }
                        BudgetLimitOutcome::Stopped => {
                            return Err(RUN_BUDGET_ENDED_ERROR.to_string())
                        }
                        BudgetLimitOutcome::Paused => return Err(RUN_PAUSED_ERROR.to_string()),
                    }
                }
                let _ = on_event.send(StreamEvent::AgentStatus {
                    run_id: run_id.to_string(),
                    agent_id: agent_id.to_string(),
                    stage: "tool_call".to_string(),
                    detail: format!("Calling tool {}", fn_name),
                });
                let _ = on_event.send(StreamEvent::ToolCall {
                    run_id: run_id.to_string(),
                    agent_id: agent_id.to_string(),
                    tool_name: fn_name.clone(),
                    args: serde_json::to_string_pretty(&fn_args).unwrap_or_default(),
                });

                let pre_dispatch_topic = if audit_mode && allow_manager_tools {
                    let (topic, embed_called) = materialize_audit_topic(
                        &fn_name,
                        &args_text,
                        None,
                        &audit_request_summary,
                        &redundancy_config,
                    )
                    .await;
                    if embed_called {
                        record_embedding_calls(tool_state, run_id, 1);
                    }
                    Some(topic)
                } else {
                    None
                };

                let redundant_before_dispatch =
                    if let Some(Some(candidate)) = pre_dispatch_topic.as_ref() {
                        detect_redundant_audit_topic(
                            candidate,
                            &covered_audit_topics,
                            &redundancy_config,
                        )
                    } else {
                        None
                    };

                let (tool_is_error, tool_text) = if let Some(reason) =
                    redundant_before_dispatch.clone()
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
                if run_cancel_requested(tool_state, run_id) {
                    return Err(RUN_CANCELLED_ERROR.to_string());
                }

                // Notify frontend: tool result received (truncate very long results)
                let char_count = tool_text.chars().count();
                let preview = if char_count > 2000 {
                    let head: String = tool_text.chars().take(2000).collect();
                    format!("{}…[truncated, {} chars total]", head, char_count)
                } else {
                    tool_text.clone()
                };
                let _ = on_event.send(StreamEvent::ToolResult {
                    run_id: run_id.to_string(),
                    agent_id: agent_id.to_string(),
                    tool_name: fn_name.clone(),
                    result: preview,
                });
                let _ = on_event.send(StreamEvent::AgentStatus {
                    run_id: run_id.to_string(),
                    agent_id: agent_id.to_string(),
                    stage: if tool_is_error {
                        "tool_error"
                    } else {
                        "tool_result"
                    }
                    .to_string(),
                    detail: format!("{} returned {} chars", fn_name, tool_text.chars().count()),
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
                        && has_usable_evidence(&messages, audit_config.as_ref())
                        && redundant_audit_retries
                            > usize::from(redundancy_config.max_redundant_audit_retries)
                    {
                        must_finish_from_evidence = true;
                    }
                    continue;
                }

                let failure_signature = classify_non_progress_tool_result(&fn_name, &tool_text);
                if fn_name == "glob_search" {
                    glob_scope_ready = !tool_is_error
                        && !tool_text.trim().is_empty()
                        && tool_text != "No files matched.";
                }
                if audit_config
                    .as_ref()
                    .map(|config| config.coverage_manifest.enabled)
                    .unwrap_or(false)
                {
                    update_audit_coverage_manifest_from_result(
                        &mut coverage_manifest,
                        &fn_name,
                        &args_text,
                        &tool_text,
                        audit_config.as_ref(),
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
                        && has_usable_evidence(&messages, audit_config.as_ref())
                    {
                        must_finish_from_evidence = true;
                    }
                } else if is_delegation_tool(&fn_name) && is_unusable_delegation_result(&tool_text)
                {
                    non_progress_count += 1;
                    if allow_manager_tools && has_usable_evidence(&messages, audit_config.as_ref())
                    {
                        must_finish_from_evidence = true;
                    }
                } else if !tool_text.trim().is_empty() {
                    iteration_progress = true;
                    non_progress_count = 0;
                }

                if audit_mode && allow_manager_tools {
                    let (post_topic_opt, post_embed_called) = materialize_audit_topic(
                        &fn_name,
                        &args_text,
                        Some(&tool_text),
                        &audit_request_summary,
                        &redundancy_config,
                    )
                    .await;
                    if post_embed_called {
                        record_embedding_calls(tool_state, run_id, 1);
                    }
                    if let Some(topic) = post_topic_opt {
                        if let Some(_reason) = detect_redundant_audit_topic(
                            &topic,
                            &covered_audit_topics,
                            &redundancy_config,
                        ) {
                            redundant_audit_retries += 1;
                            non_progress_count += 1;
                            if allow_manager_tools
                                && has_usable_evidence(&messages, audit_config.as_ref())
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

                if let Some(config) = audit_config
                    .as_ref()
                    .filter(|config| config.delegation_validation.enabled)
                {
                    if is_delegation_tool(&fn_name) && !tool_is_error {
                        let delegation_topic_key =
                            stable_delegation_retry_key(&args_text, &audit_request_summary);
                        if is_weak_audit_delegation_result(
                            &tool_text,
                            &covered_audit_topics,
                            config,
                        ) {
                            let retries = weak_delegation_retries
                                .entry(delegation_topic_key.clone())
                                .or_insert(0);
                            *retries += 1;
                            let max_weak_retries =
                                config.delegation_validation.max_weak_retries.unwrap_or(1);
                            if *retries <= max_weak_retries {
                                non_progress_count += 1;
                                messages.push(json!({
                                "role": "user",
                                    "content": render_audit_template(
                                        config
                                            .delegation_validation
                                            .retry_prompt_template
                                            .as_deref()
                                            .unwrap_or("The delegated audit result on `{topic}` did not add new useful evidence. Retry this topic at most once with one concrete file or function to inspect and one concrete question to answer. Do not return a plan, status update, or generic hypotheses."),
                                        &[("topic", delegation_topic_key.clone())],
                                    )
                            }));
                            } else {
                                non_progress_count += 2;
                                redundant_audit_retries += 1;
                                if allow_manager_tools
                                    && has_usable_evidence(&messages, audit_config.as_ref())
                                {
                                    must_finish_from_evidence = true;
                                }
                            }
                        }
                    }
                }

                if audit_mode
                    && allow_manager_tools
                    && has_usable_evidence(&messages, audit_config.as_ref())
                    && should_force_audit_synthesis(
                        &messages,
                        &covered_audit_topics,
                        &coverage_manifest,
                        redundant_audit_retries,
                        &redundancy_config,
                        audit_config.as_ref(),
                    )
                {
                    must_finish_from_evidence = true;
                }

                if let Some(config) = audit_config.as_ref() {
                    if let Some(limit) = config.tool_burst_reflection.limit {
                        if tool_burst >= limit {
                            let reflection = force_tool_free_reflection(
                                &llm_client,
                                &url,
                                model_key,
                                &messages,
                                config,
                            )
                            .await?;
                            if !reflection.is_empty() {
                                messages.push(json!({"role": "assistant", "content": reflection}));
                            }
                            tool_burst = 0;
                        }
                    }
                }
            }
            if !iteration_progress {
                non_progress_count += 1;
            }
            emit_usage_update(tool_state, run_id, on_event);
            if must_finish_from_evidence || non_progress_count >= non_progress_limit {
                let missing_scope_refs = unresolved_audit_coverage_manifest_entries(
                    &coverage_manifest,
                    None,
                    audit_config.as_ref(),
                );
                return Ok(force_non_progress_summary(
                    &llm_client,
                    &url,
                    model_key,
                    &messages,
                    audit_config.as_ref(),
                    &missing_scope_refs,
                )
                .await);
            }
        } else {
            if run_cancel_requested(tool_state, run_id) {
                return Err(RUN_CANCELLED_ERROR.to_string());
            }
            if require_delegation && !delegated {
                messages.push(json!({
                    "role": "user",
                    "content": "You are the manager. You must delegate at least one concrete sub-task before giving a final answer. Use agent-management tools first, then synthesize."
                }));
                continue;
            }

            let final_content = assistant_msg["content"].as_str().unwrap_or("").to_string();

            let mut final_content = final_content;
            if let Some(config) = audit_config.as_ref() {
                if audit_response_needs_rewrite(
                    &final_content,
                    audit_evidence_grade(&messages, config),
                    config,
                ) {
                    final_content = rewrite_grounded_audit_response(
                        &llm_client,
                        &url,
                        model_key,
                        &messages,
                        &final_content,
                        config,
                    )
                    .await;
                }
            }
            if allow_manager_tools
                && messages
                    .iter()
                    .any(|message| message["role"].as_str() == Some("tool"))
                && manager_response_needs_compaction(&final_content)
            {
                final_content = rewrite_manager_response_compactly(
                    &llm_client,
                    &url,
                    model_key,
                    &messages,
                    &final_content,
                )
                .await;
            }

            let missing_scope_refs = if audit_config
                .as_ref()
                .map(|config| config.coverage_manifest.enabled)
                .unwrap_or(false)
            {
                unresolved_audit_coverage_manifest_entries(
                    &coverage_manifest,
                    Some(&final_content),
                    audit_config.as_ref(),
                )
            } else {
                Vec::new()
            };

            if final_content.trim().is_empty() || is_progress_only_response(&final_content) {
                non_progress_count += 1;
                if non_progress_count >= non_progress_limit {
                    let missing_scope_refs = unresolved_audit_coverage_manifest_entries(
                        &coverage_manifest,
                        None,
                        audit_config.as_ref(),
                    );
                    return Ok(force_non_progress_summary(
                        &llm_client,
                        &url,
                        model_key,
                        &messages,
                        audit_config.as_ref(),
                        &missing_scope_refs,
                    )
                    .await);
                }
                messages.push(json!({
                    "role": "user",
                    "content": audit_config
                        .as_ref()
                        .and_then(|config| config.non_progress.stall_prompt.clone())
                        .unwrap_or_else(|| "Your previous response did not contain a usable final answer. Stop exploring and provide either concrete findings or a short bounded summary of what is known, what blocked progress, and the smallest remaining useful next step.".to_string())
                }));
                continue;
            }
            if let Some(config) = audit_config.as_ref() {
                if config.coverage_manifest.require_resolution && !missing_scope_refs.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": unresolved_coverage_prompt(config, &missing_scope_refs),
                    }));
                    continue;
                }
                if let Some(prompt) = audit_section_followup_prompt(&final_content, config) {
                    messages.push(json!({
                        "role": "user",
                        "content": prompt,
                    }));
                    continue;
                }
            }

            return Ok(final_content);
        }
    }
    Ok(force_non_progress_summary(
        &llm_client,
        &url,
        model_key,
        &messages,
        audit_config.as_ref(),
        &unresolved_audit_coverage_manifest_entries(
            &coverage_manifest,
            None,
            audit_config.as_ref(),
        ),
    )
    .await)
}

#[derive(Clone, Serialize)]
#[serde(tag = "event")]
pub enum StreamEvent {
    #[serde(rename = "agent_start")]
    AgentStart {
        run_id: String,
        agent_id: String,
        agent_name: String,
        model_key: String,
        mode: String,
        is_manager: bool,
        context_limit: u64,
        estimated_input_tokens: u64,
        estimated_remaining_tokens: u64,
    },
    #[serde(rename = "token")]
    Token {
        run_id: String,
        agent_id: String,
        content: String,
    },
    #[serde(rename = "agent_status")]
    AgentStatus {
        run_id: String,
        agent_id: String,
        stage: String,
        detail: String,
    },
    #[serde(rename = "agent_metrics")]
    AgentMetrics {
        run_id: String,
        agent_id: String,
        stage: String,
        estimated_output_tokens: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens_per_second: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        time_to_first_token_seconds: Option<f64>,
    },
    #[serde(rename = "agent_end")]
    AgentEnd { run_id: String, agent_id: String },
    #[serde(rename = "run_dossier_updated")]
    RunDossierUpdated { run_id: String, dossier: RunDossier },
    #[serde(rename = "finalizer_output")]
    FinalizerOutput {
        run_id: String,
        agent_id: String,
        agent_name: String,
        source_agent_id: String,
        source_agent_name: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        mode: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        model_key: Option<String>,
    },
    #[serde(rename = "error")]
    Error {
        run_id: String,
        agent_id: String,
        agent_name: String,
        message: String,
    },
    /// Emitted just before a manager tool call is dispatched.
    #[serde(rename = "tool_call")]
    ToolCall {
        run_id: String,
        agent_id: String,
        tool_name: String,
        args: String,
    },
    /// Emitted after the tool returns its result.
    #[serde(rename = "tool_result")]
    ToolResult {
        run_id: String,
        agent_id: String,
        tool_name: String,
        result: String,
    },
    #[serde(rename = "run_limit_reached")]
    RunLimitReached {
        run_id: String,
        kind: String,
        limit: u64,
        observed: u64,
    },
    #[serde(rename = "run_waiting_confirmation")]
    RunWaitingConfirmation { run_id: String, message: String },
    #[serde(rename = "run_resumed")]
    RunResumed { run_id: String, message: String },
    #[serde(rename = "run_cancelled")]
    RunCancelled { run_id: String, message: String },
    #[serde(rename = "run_checkpoint_saved")]
    RunCheckpointSaved { run_id: String, checkpoint: String },
    #[serde(rename = "run_usage_update")]
    RunUsageUpdate {
        run_id: String,
        llm_calls: u32,
        tool_calls: u32,
        spawned_agents: u32,
        streamed_tokens: u64,
        embedding_calls: u32,
        wall_clock_seconds: u64,
    },
    #[serde(rename = "done")]
    Done { run_id: String },
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
        "spawn_agent"
            | "send_message"
            | "broadcast_message"
            | "fork_agent"
            | "pipe_message"
            | "send_parallel"
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
        || lower.contains(".py")
        || lower.contains(".go")
        || lower.contains(".java")
        || lower.contains(".kt")
        || lower.contains(".rb")
        || lower.contains(".php")
        || lower.contains(".cs")
        || lower.contains(".c")
        || lower.contains(".cpp")
        || lower.contains(".swift")
        || lower.contains(".scala")
        || lower.contains(" rust")
        || lower.contains("backend")
        || lower.contains("server")
        || lower.contains("api")
        || lower.contains("service")
    {
        scopes.push(AuditCoverageScope::BackendRust);
    }
    if lower.contains("frontend")
        || lower.contains("client")
        || lower.contains("ui")
        || lower.contains("src/")
        || lower.contains(".js")
        || lower.contains(".ts")
        || lower.contains(".jsx")
        || lower.contains(".tsx")
        || lower.contains(".html")
        || lower.contains(".css")
        || lower.contains(".scss")
        || lower.contains("renderer")
    {
        scopes.push(AuditCoverageScope::Frontend);
    }
    if lower.contains("config")
        || lower.contains("configs")
        || lower.contains(".json")
        || lower.contains(".toml")
        || lower.contains(".yaml")
        || lower.contains(".yml")
        || lower.contains(".xml")
        || lower.contains(".ini")
        || lower.contains(".env")
        || lower.contains("package.json")
        || lower.contains("dockerfile")
        || lower.contains("compose")
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
        AuditCoverageScope::BackendRust => "backend / application code",
        AuditCoverageScope::Frontend => "frontend / client code",
        AuditCoverageScope::Configs => "configs / manifests / deployment files",
        AuditCoverageScope::Capabilities => "permissions / capabilities / access control",
    }
}

fn is_high_signal_audit_ref(reference: &str) -> bool {
    let lower = reference.to_lowercase();
    lower.ends_with(".rs")
        || lower.ends_with(".py")
        || lower.ends_with(".go")
        || lower.ends_with(".java")
        || lower.ends_with(".kt")
        || lower.ends_with(".rb")
        || lower.ends_with(".php")
        || lower.ends_with(".cs")
        || lower.ends_with(".c")
        || lower.ends_with(".cpp")
        || lower.ends_with(".swift")
        || lower.ends_with(".scala")
        || lower.ends_with(".js")
        || lower.ends_with(".ts")
        || lower.ends_with(".jsx")
        || lower.ends_with(".tsx")
        || lower.ends_with(".html")
        || lower.ends_with(".css")
        || lower.ends_with(".json")
        || lower.ends_with(".toml")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".xml")
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
        AuditCoverageScope::BackendRust => {
            lower.ends_with(".rs")
                || lower.ends_with(".py")
                || lower.ends_with(".go")
                || lower.ends_with(".java")
                || lower.ends_with(".kt")
                || lower.ends_with(".rb")
                || lower.ends_with(".php")
                || lower.ends_with(".cs")
                || lower.ends_with(".c")
                || lower.ends_with(".cpp")
                || lower.ends_with(".swift")
                || lower.ends_with(".scala")
        }
        AuditCoverageScope::Frontend => {
            lower.starts_with("src/")
                || lower.starts_with("app/")
                || lower.ends_with(".js")
                || lower.ends_with(".ts")
                || lower.ends_with(".jsx")
                || lower.ends_with(".tsx")
                || lower.ends_with(".html")
                || lower.ends_with(".css")
        }
        AuditCoverageScope::Configs => {
            lower.ends_with(".json")
                || lower.ends_with(".toml")
                || lower.ends_with(".yaml")
                || lower.ends_with(".yml")
                || lower.ends_with(".xml")
                || lower.ends_with(".ini")
                || lower.ends_with(".env")
                || lower.ends_with("dockerfile")
        }
        AuditCoverageScope::Capabilities => {
            lower.contains("capabilities/") || lower.contains("permissions")
        }
    }
}

fn update_audit_coverage_manifest_from_result(
    manifest: &mut Vec<AuditCoverageEntry>,
    _tool_name: &str,
    args_text: &str,
    tool_text: &str,
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
) {
    extend_audit_coverage_manifest(manifest, args_text);
    if manifest.iter().any(|entry| entry.scope.is_some()) {
        for reference in extract_explicit_audit_refs(tool_text) {
            if is_high_signal_audit_ref(&reference)
                && manifest_scope_matches_ref(manifest, &reference)
            {
                push_coverage_file_entry(manifest, &reference);
            }
        }
    }
    let gap_label = audit_config
        .map(ResolvedAuditBehaviorConfig::gap_section_label)
        .unwrap_or("Coverage gaps")
        .to_lowercase();
    if tool_text.to_lowercase().contains(&gap_label) {
        mark_audit_coverage_manifest_reported_gap(manifest, tool_text, audit_config);
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
            if refs
                .iter()
                .any(|reference| audit_scope_matches_ref(scope, reference))
                && topic.inspection_depth >= AuditInspectionDepth::Targeted
            {
                entry.status = AuditCoverageStatus::Inspected;
            }
        } else if refs
            .iter()
            .any(|reference| audit_ref_entry_matches(reference, &entry.label))
            && topic.inspection_depth >= AuditInspectionDepth::EvidenceBacked
        {
            entry.status = AuditCoverageStatus::Inspected;
        }
    }
}

fn mark_audit_coverage_manifest_reported_gap(
    manifest: &mut [AuditCoverageEntry],
    response: &str,
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
) {
    let gap_label = audit_config
        .map(ResolvedAuditBehaviorConfig::gap_section_label)
        .unwrap_or("Coverage gaps");
    let coverage_gaps = extract_audit_section(response, gap_label, &[]);
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
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
) -> Vec<String> {
    let mut snapshot = manifest.to_vec();
    if let Some(text) = response {
        mark_audit_coverage_manifest_reported_gap(&mut snapshot, text, audit_config);
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

fn render_bullet_list(items: &[String], empty_label: &str) -> String {
    if items.is_empty() {
        format!("- {}", empty_label)
    } else {
        items
            .iter()
            .map(|item| format!("- {}", item))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn render_csv_list(items: &[String], empty_label: &str) -> String {
    if items.is_empty() {
        empty_label.to_string()
    } else {
        items.join(", ")
    }
}

fn render_required_sections_csv(config: &ResolvedAuditBehaviorConfig) -> String {
    if config.required_sections.is_empty() {
        "`Confirmed findings`, `Hypotheses / lower-confidence risks`, `Coverage gaps`".to_string()
    } else {
        config
            .required_sections
            .iter()
            .map(|section| format!("`{}`", section))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn render_audit_template(template: &str, replacements: &[(&str, String)]) -> String {
    render_prompt_template(template, replacements)
}

fn audit_runtime_note(
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
    covered_topics: &[CoveredAuditTopic],
    coverage_manifest: &[AuditCoverageEntry],
) -> Option<String> {
    let Some(config) = audit_config else {
        return None;
    };
    if !config.runtime_note_enabled || (covered_topics.is_empty() && coverage_manifest.is_empty()) {
        return None;
    }

    let covered_items = covered_topics
        .iter()
        .rev()
        .take(4)
        .map(|topic| topic.summary.clone())
        .collect::<Vec<_>>();
    let unresolved =
        unresolved_audit_coverage_manifest_entries(coverage_manifest, None, Some(config));
    let template = config
        .runtime_note_template
        .as_deref()
        .unwrap_or("Audit topics already covered in this turn:\n{covered_topics_bullets}\n\nAvoid re-delegating or re-checking the same file/function/theme unless you are opening a genuinely new code/config area. If current evidence is already sufficient, synthesize now.\n\nRequired audit coverage still unresolved:\n{unresolved_scopes_bullets}");
    Some(render_audit_template(
        template,
        &[
            (
                "covered_topics_bullets",
                render_bullet_list(&covered_items, "none yet"),
            ),
            (
                "covered_topics_csv",
                render_csv_list(&covered_items, "none yet"),
            ),
            (
                "unresolved_scopes_bullets",
                render_bullet_list(
                    &unresolved.iter().take(6).cloned().collect::<Vec<_>>(),
                    "none",
                ),
            ),
            (
                "unresolved_scopes_csv",
                render_csv_list(&unresolved, "none"),
            ),
            (
                "required_sections_bullets",
                render_bullet_list(&config.required_sections, "none"),
            ),
            (
                "required_sections_csv",
                render_required_sections_csv(config),
            ),
            ("gap_section_label", config.gap_section_label().to_string()),
        ],
    ))
}

/// Returns `(topic, embedding_api_called)` where the bool is true when `create_embeddings`
/// was actually invoked (not inferred from cache).
async fn materialize_audit_topic(
    tool_name: &str,
    args_text: &str,
    tool_result: Option<&str>,
    audit_request_summary: &str,
    config: &RedundancyDetectionConfig,
) -> (Option<CoveredAuditTopic>, bool) {
    let combined = format!(
        "{}\n{}\n{}\n{}",
        tool_name,
        audit_request_summary,
        args_text,
        tool_result.unwrap_or("")
    );
    if !looks_like_audit_topic(&combined) {
        return (None, false);
    }

    let refs = extract_topic_refs(&combined);
    let target_refs = extract_target_refs(args_text);
    let terms = extract_topic_terms(&combined);
    if refs.is_empty() && terms.is_empty() {
        return (None, false);
    }

    let inspection_depth = infer_inspection_depth(tool_name, args_text, tool_result, &target_refs);
    let lexical_signature =
        lexical_topic_signature(tool_name, audit_request_summary, &refs, &terms);
    let summary = format_topic_summary(tool_name, audit_request_summary, &refs, &terms);
    let mut embed_api_called = false;
    let embedding = if config.enabled {
        match config.embedding_model_key.as_deref() {
            Some(model_key) if !model_key.trim().is_empty() => {
                let result = create_embeddings(model_key, &[summary.clone()])
                    .await
                    .ok()
                    .and_then(|mut vectors| vectors.drain(..).next());
                if result.is_some() {
                    embed_api_called = true;
                }
                result
            }
            _ => None,
        }
    } else {
        None
    };

    (
        Some(CoveredAuditTopic {
            lexical_signature,
            summary,
            refs: refs.into_iter().collect(),
            target_refs,
            inspection_depth,
            embedding,
        }),
        embed_api_called,
    )
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
    config.semantic_similarity_threshold.clamp(
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
        .split(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '"' | '\'' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}'
                )
        })
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
        if classify_default_audit_evidence(result) >= AuditEvidenceGrade::ConfigContent
            || (result_lower.contains("confirmed findings") && has_file_reference(result))
        {
            return AuditInspectionDepth::EvidenceBacked;
        }
    }

    if !target_refs.is_empty()
        || [
            "read ", "analyze ", "analyse ", "audit ", "examine ", "inspect ", "review ",
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
    audit_config: &ResolvedAuditBehaviorConfig,
) -> Result<String, String> {
    let reflection_messages = {
        let mut copy = messages.to_vec();
        copy.push(json!({
            "role": "user",
            "content": render_audit_template(
                audit_config
                    .tool_burst_reflection
                    .prompt
                    .as_deref()
                    .unwrap_or("Pause tool use. Briefly summarize what you learned, what is still uncertain, and the smallest next step. Do not call any tools in this response."),
                &[
                    (
                        "tool_burst_limit",
                        audit_config
                            .tool_burst_reflection
                            .limit
                            .unwrap_or(0)
                            .to_string(),
                    ),
                    (
                        "required_sections_csv",
                        render_required_sections_csv(audit_config),
                    ),
                    (
                        "gap_section_label",
                        audit_config.gap_section_label().to_string(),
                    ),
                ],
            )
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
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
    missing_scope_refs: &[String],
) -> String {
    let reflection_messages = {
        let mut copy = messages.to_vec();
        copy.push(json!({
            "role": "user",
            "content": non_progress_summary_prompt(audit_config, missing_scope_refs),
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
                    fallback_non_progress_summary(audit_config, missing_scope_refs)
                } else {
                    text
                }
            }
            Err(_) => fallback_non_progress_summary(audit_config, missing_scope_refs),
        },
        _ => fallback_non_progress_summary(audit_config, missing_scope_refs),
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

fn non_progress_summary_prompt(
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
    missing_scope_refs: &[String],
) -> String {
    let Some(config) = audit_config else {
        return "Stop now because progress has stalled. In 3 short bullets, state: 1) what is known, 2) what blocked progress, 3) the smallest remaining useful next step. Do not call any tools.".to_string();
    };
    render_audit_template(
        config
            .force_synthesis
            .prompt
            .as_deref()
            .unwrap_or("Tooling or delegation has stalled. Using only the concrete evidence already present in this conversation, produce a grounded audit report with these sections exactly: {required_sections_csv}. Only keep findings as confirmed if they are directly supported by code or config already shown. Downgrade unsupported concerns into hypotheses. Every hypothesis must cite a concrete file/config or observed behavior already in context, or explicitly state what evidence is still missing to confirm it. Remove generic dependency-presence or stack-template risks. Do not call any tools.\n\nIf some requested scopes remain uninspected, explicitly list them under `{gap_section_label}`: {missing_scope_refs_csv}."),
        &[
            (
                "required_sections_csv",
                render_required_sections_csv(config),
            ),
            (
                "gap_section_label",
                config.gap_section_label().to_string(),
            ),
            (
                "missing_scope_refs_csv",
                render_csv_list(missing_scope_refs, "none"),
            ),
        ],
    )
}

fn has_concrete_repo_evidence(messages: &[Value]) -> bool {
    messages
        .iter()
        .filter(|message| message["role"].as_str() == Some("tool"))
        .map(|message| classify_default_audit_evidence(message["content"].as_str().unwrap_or("")))
        .max()
        .unwrap_or(AuditEvidenceGrade::Inferred)
        >= AuditEvidenceGrade::CommandOnly
}

fn fallback_non_progress_summary(
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
    missing_scope_refs: &[String],
) -> String {
    let Some(config) = audit_config else {
        return "Stopping early due to repeated non-progress. Known facts were collected, but the agent kept retrying without advancing. The smallest next step is to retry with a narrower task or inspect the blocked area directly.".to_string();
    };
    render_audit_template(
        config
            .force_synthesis
            .fallback_text
            .as_deref()
            .unwrap_or("Confirmed findings\n- None supported strongly enough to promote after tooling stalled.\n\nHypotheses / lower-confidence risks\n- Some concerns may remain, but the current conversation does not contain enough direct code/config evidence to confirm them.\n\nCoverage gaps\n- Tooling stalled before full inspection. Additional direct reads would be needed to upgrade any hypothesis into a confirmed finding.\n- Requested but not inspected: {missing_scope_refs_csv}."),
        &[
            (
                "missing_scope_refs_csv",
                render_csv_list(missing_scope_refs, "none"),
            ),
            (
                "gap_section_label",
                config.gap_section_label().to_string(),
            ),
        ],
    )
}

fn has_usable_evidence(
    messages: &[Value],
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
) -> bool {
    if let Some(config) = audit_config {
        audit_evidence_grade(messages, config)
            >= config
                .evidence_grading
                .min_grade_to_synthesize
                .unwrap_or(AuditEvidenceGrade::ConfigContent)
    } else {
        has_concrete_repo_evidence(messages)
    }
}

fn should_force_audit_synthesis(
    messages: &[Value],
    covered_topics: &[CoveredAuditTopic],
    coverage_manifest: &[AuditCoverageEntry],
    redundant_audit_retries: usize,
    redundancy_config: &RedundancyDetectionConfig,
    audit_config: Option<&ResolvedAuditBehaviorConfig>,
) -> bool {
    let Some(config) = audit_config else {
        return false;
    };
    if redundant_audit_retries > usize::from(redundancy_config.max_redundant_audit_retries) {
        return true;
    }

    let completed_reports = count_completed_audit_reports(messages, config);
    let issue_like_reports = count_issue_like_audit_reports(messages, config);
    let has_gaps = messages.iter().any(|message| {
        message["role"].as_str() == Some("tool")
            && has_named_section(
                message["content"].as_str().unwrap_or(""),
                config.gap_section_label(),
            )
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
        unresolved_audit_coverage_manifest_entries(coverage_manifest, None, Some(config)).len();
    let coverage_gate = !config.force_synthesis.require_coverage_gap_signal
        || has_gaps
        || coverage_manifest.is_empty()
        || unresolved_manifest_entries == 0;
    let completed_gate = config
        .force_synthesis
        .after_n_completed_reports
        .map(|min| completed_reports >= min)
        .unwrap_or(false)
        && config
            .force_synthesis
            .min_targeted_topics
            .map(|min| targeted_topics >= min)
            .unwrap_or(true);
    let issue_gate = config
        .force_synthesis
        .after_n_issue_reports
        .map(|min| issue_like_reports >= min)
        .unwrap_or(false)
        && completed_reports
            >= config
                .force_synthesis
                .after_n_completed_reports
                .unwrap_or(1)
                .saturating_sub(1);
    let evidence_gate = config
        .force_synthesis
        .min_evidence_backed_topics
        .map(|min| evidence_backed_topics >= min)
        .unwrap_or(false);

    coverage_gate && (completed_gate || issue_gate || evidence_gate)
}

fn find_matching_section_name<'a>(
    config: &'a ResolvedAuditBehaviorConfig,
    needle: &str,
    fallback: &'a str,
) -> &'a str {
    config
        .required_sections
        .iter()
        .find(|section| section.eq_ignore_ascii_case(needle))
        .map(String::as_str)
        .or_else(|| {
            config
                .section_rules
                .iter()
                .find(|rule| rule.section_name.eq_ignore_ascii_case(needle))
                .map(|rule| rule.section_name.as_str())
        })
        .unwrap_or(fallback)
}

fn count_completed_audit_reports(
    messages: &[Value],
    config: &ResolvedAuditBehaviorConfig,
) -> usize {
    let confirmed_label =
        find_matching_section_name(config, "Confirmed findings", "Confirmed findings");
    messages
        .iter()
        .filter(|message| message["role"].as_str() == Some("tool"))
        .filter(|message| {
            let content = message["content"].as_str().unwrap_or("");
            has_named_section(content, confirmed_label)
                && has_named_section(content, config.gap_section_label())
        })
        .count()
}

fn count_issue_like_audit_reports(
    messages: &[Value],
    config: &ResolvedAuditBehaviorConfig,
) -> usize {
    messages
        .iter()
        .filter(|message| message["role"].as_str() == Some("tool"))
        .filter(|message| {
            let content = message["content"].as_str().unwrap_or("");
            audit_report_has_issue_like_finding(content, config)
        })
        .count()
}

fn classify_evidence_against_signals(
    content: &str,
    code_signals: &[String],
    config_signals: &[String],
    command_signals: &[String],
) -> AuditEvidenceGrade {
    let lower = content.to_lowercase();
    if code_signals
        .iter()
        .any(|signal| content.contains(signal) || lower.contains(&signal.to_lowercase()))
    {
        AuditEvidenceGrade::CodeContent
    } else if config_signals
        .iter()
        .any(|signal| content.contains(signal) || lower.contains(&signal.to_lowercase()))
    {
        AuditEvidenceGrade::ConfigContent
    } else if command_signals
        .iter()
        .any(|signal| content.contains(signal) || lower.contains(&signal.to_lowercase()))
    {
        AuditEvidenceGrade::CommandOnly
    } else {
        AuditEvidenceGrade::Inferred
    }
}

fn classify_default_audit_evidence(content: &str) -> AuditEvidenceGrade {
    classify_evidence_against_signals(
        content,
        &[
            "#[tauri::command]".to_string(),
            "pub(crate)".to_string(),
            "async fn ".to_string(),
            "fn ".to_string(),
            "impl ".to_string(),
            "use std::".to_string(),
            "match ".to_string(),
        ],
        &[
            "\"$schema\"".to_string(),
            "\"permissions\"".to_string(),
            "\"scripts\"".to_string(),
            "\"devDependencies\"".to_string(),
            "[package]".to_string(),
            "[dependencies]".to_string(),
            "core:default".to_string(),
            "\"csp\"".to_string(),
        ],
        &[
            "src-tauri/".to_string(),
            "package.json".to_string(),
            "workspace.rs".to_string(),
            "models.rs".to_string(),
            "src-tauri/src".to_string(),
        ],
    )
}

fn audit_evidence_grade(
    messages: &[Value],
    config: &ResolvedAuditBehaviorConfig,
) -> AuditEvidenceGrade {
    messages
        .iter()
        .filter(|message| message["role"].as_str() == Some("tool"))
        .map(|message| classify_audit_evidence(message["content"].as_str().unwrap_or(""), config))
        .max()
        .unwrap_or(AuditEvidenceGrade::Inferred)
}

fn classify_audit_evidence(
    content: &str,
    config: &ResolvedAuditBehaviorConfig,
) -> AuditEvidenceGrade {
    classify_evidence_against_signals(
        content,
        &config.evidence_grading.code_signals,
        &config.evidence_grading.config_signals,
        &config.evidence_grading.command_signals,
    )
}

fn audit_response_needs_rewrite(
    response: &str,
    evidence_grade: AuditEvidenceGrade,
    config: &ResolvedAuditBehaviorConfig,
) -> bool {
    if !config.response_rewrite.enabled {
        return false;
    }
    let lower = response.to_lowercase();
    if response.trim().is_empty() {
        return true;
    }
    if !missing_required_sections(response, config).is_empty() {
        return true;
    }
    if contains_severity_label(&lower)
        && evidence_grade
            < config
                .response_rewrite
                .min_evidence_grade_for_severity
                .unwrap_or(AuditEvidenceGrade::ConfigContent)
    {
        return true;
    }
    let hypotheses_label = find_matching_section_name(
        config,
        "Hypotheses / lower-confidence risks",
        "Hypotheses / lower-confidence risks",
    );
    if contains_speculation_marker(&lower) && !has_named_section(response, hypotheses_label) {
        return true;
    }
    config
        .section_rules
        .iter()
        .any(|rule| section_rule_failed(response, rule, config))
}

fn contains_severity_label(lower: &str) -> bool {
    [
        "critical",
        "high severity",
        "medium severity",
        "low severity",
        "severity:",
    ]
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
    config: &ResolvedAuditBehaviorConfig,
) -> String {
    let rewrite_messages = {
        let mut copy = messages.to_vec();
        copy.push(json!({
            "role": "user",
            "content": render_audit_template(
                config
                    .response_rewrite
                    .rewrite_prompt
                    .as_deref()
                    .unwrap_or("Rewrite the audit answer below using only the evidence already present in this conversation.\n\nRules:\n- Use exactly these sections: {required_sections_csv}.\n- A confirmed finding must be directly supported by code or config already shown.\n- Keep only actual issues in `Confirmed findings`; move neutral observations, mitigations, architecture facts, and absence-of-risk statements out of that section.\n- Downgrade any unsupported concern into hypotheses.\n- Every hypothesis must cite a concrete file/config or observed behavior already in context, or explicitly state what evidence is still missing to confirm it.\n- Remove dependency-presence, stack-template, or generic speculative risks that are not tied to observed behavior.\n- Do not invent new evidence.\n- Do not label anything `High` or `Critical` without direct code/config support.\n\nCandidate answer:\n{candidate_response}"),
                &[
                    (
                        "required_sections_csv",
                        render_required_sections_csv(config),
                    ),
                    ("candidate_response", candidate.to_string()),
                    (
                        "gap_section_label",
                        config.gap_section_label().to_string(),
                    ),
                ],
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

fn normalize_compaction_fingerprint(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn markdown_heading_fingerprints(response: &str) -> Vec<String> {
    response
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('#'))
        .map(normalize_compaction_fingerprint)
        .filter(|line| !line.is_empty())
        .collect()
}

fn markdown_table_header_fingerprints(response: &str) -> Vec<String> {
    let lines = response.lines().map(str::trim).collect::<Vec<_>>();
    lines
        .windows(2)
        .filter_map(|window| {
            let header = window[0];
            let divider = window[1];
            if header.starts_with('|')
                && header.ends_with('|')
                && divider.contains("---")
                && divider.contains('|')
            {
                Some(normalize_compaction_fingerprint(header))
            } else {
                None
            }
        })
        .filter(|line| !line.is_empty())
        .collect()
}

fn long_paragraph_fingerprints(response: &str) -> Vec<String> {
    response
        .split("\n\n")
        .map(str::trim)
        .filter(|paragraph| paragraph.len() >= 120)
        .filter(|paragraph| {
            !paragraph
                .lines()
                .all(|line| line.trim_start().starts_with('#'))
        })
        .map(normalize_compaction_fingerprint)
        .filter(|paragraph| paragraph.len() >= 80)
        .collect()
}

fn count_duplicate_fingerprints(items: &[String]) -> usize {
    let mut seen = HashSet::new();
    let mut duplicates = 0usize;
    for item in items {
        if !seen.insert(item.clone()) {
            duplicates += 1;
        }
    }
    duplicates
}

fn manager_response_needs_compaction(response: &str) -> bool {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return false;
    }

    let heading_fingerprints = markdown_heading_fingerprints(trimmed);
    let table_fingerprints = markdown_table_header_fingerprints(trimmed);
    let paragraph_fingerprints = long_paragraph_fingerprints(trimmed);
    let heading_count = heading_fingerprints.len();

    count_duplicate_fingerprints(&heading_fingerprints) > 0
        || count_duplicate_fingerprints(&table_fingerprints) > 0
        || count_duplicate_fingerprints(&paragraph_fingerprints) > 0
        || (heading_count >= 8 && trimmed.len() >= 3_000)
}

async fn rewrite_manager_response_compactly(
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
                "Rewrite the draft answer below into one final answer for the user using only the evidence already present in this conversation.\n\nRules:\n- Answer the user's request directly.\n- Collapse overlapping worker reports into one synthesis.\n- Remove repeated headings, repeated tables, duplicated recommendations, and stitched draft structure.\n- Preserve only claims supported by evidence already in the conversation.\n- Keep uncertainty explicit where evidence is incomplete.\n- Do not invent new evidence, tool calls, or follow-up work.\n- Keep the answer concise but complete.\n\nDraft answer:\n{}",
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

fn render_prompt_template(template: &str, replacements: &[(&str, String)]) -> String {
    let mut rendered = template.to_string();
    for (key, value) in replacements {
        rendered = rendered.replace(&format!("{{{}}}", key), value);
    }
    rendered
}

fn format_internal_transcript(messages: &[Value], max_chars: usize) -> String {
    if messages.is_empty() || max_chars == 0 {
        return "[omitted]".to_string();
    }
    let mut transcript = messages
        .iter()
        .filter_map(|message| {
            let role = message["role"].as_str()?;
            let content = message["content"].as_str().unwrap_or("").trim();
            if content.is_empty() {
                None
            } else {
                Some(format!("{}: {}", role, content.replace('\n', " ")))
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if transcript.chars().count() > max_chars {
        transcript = transcript
            .chars()
            .rev()
            .take(max_chars)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        transcript = format!("...[truncated]\n{}", transcript);
    }
    transcript
}

fn format_worker_outcomes_excerpt(dossier: &RunDossier) -> String {
    if dossier.worker_outcomes.is_empty() {
        return "[none]".to_string();
    }
    dossier
        .worker_outcomes
        .iter()
        .map(|outcome| {
            format!(
                "- {} ({})\n  summary: {}\n  observed evidence: {}\n  inferences: {}\n  coverage gaps: {}",
                outcome.agent_name,
                outcome.agent_id,
                outcome.summary,
                if outcome.observed_evidence.is_empty() {
                    "[none]".to_string()
                } else {
                    outcome.observed_evidence.join(" | ")
                },
                if outcome.inferences.is_empty() {
                    "[none]".to_string()
                } else {
                    outcome.inferences.join(" | ")
                },
                if outcome.coverage_gaps.is_empty() {
                    "[none]".to_string()
                } else {
                    outcome.coverage_gaps.join(" | ")
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_claim_calibration(dossier: &RunDossier) -> String {
    let breadth = dossier.inspected_paths.len();
    let caution = if breadth <= 3 {
        "Inspected scope is shallow. Narrow claims to the inspected entrypoints and state the missing areas explicitly."
    } else if breadth <= 8 {
        "Inspected scope is moderate. Prefer calibrated conclusions and note remaining gaps where they affect confidence."
    } else {
        "Inspected scope is broad enough for stronger conclusions, but still respect any recorded coverage gaps."
    };

    let methodology = if dossier.counts.broad_full_file_reads + dossier.counts.broad_directory_scans
        >= 3
    {
        "Recent methodology included broad reads/scans. Do not overstate confidence from those steps alone."
    } else {
        "Methodology was mostly targeted. You may rely more directly on the observed evidence."
    };

    format!(
        "{}\n{}\nCurrent caution flags: {}",
        caution,
        methodology,
        if dossier.caution_flags.is_empty() {
            "[none]".to_string()
        } else {
            dossier.caution_flags.join(" | ")
        }
    )
}

fn build_run_dossier_json(dossier: &RunDossier) -> String {
    serde_json::to_string_pretty(dossier).unwrap_or_else(|_| "{}".to_string())
}

pub(crate) async fn run_finalizer(
    tool_state: &crate::mcp::McpState,
    run_id: &str,
    manager_agent_id: &str,
    fallback_model_key: &str,
    manager_response: &str,
    config: &FinalizerConfig,
    mode: &str,
) -> (String, Option<String>) {
    let (manager_messages, manager_model, mut dossier) = {
        let runs = tool_state.active_runs.lock().unwrap();
        runs.get(run_id)
            .map(|run| {
                (
                    run.manager_messages.clone(),
                    run.manager_model_key.clone(),
                    run.dossier.clone(),
                )
            })
            .unwrap_or_default()
    };

    let model_key = config.model_key.clone().or(manager_model).or_else(|| {
        (!fallback_model_key.trim().is_empty()).then(|| fallback_model_key.to_string())
    });

    let Some(model_key) = model_key else {
        return (manager_response.to_string(), None);
    };

    if dossier.manager_draft_summary.trim().is_empty() {
        dossier.manager_draft_summary = compact_manager_draft(manager_response);
    }
    dossier.finalizer_mode = Some(mode.to_string());
    dossier.finalizer_input_summary = dedupe_preserve_order(vec![
        if config.include_run_dossier {
            "Run dossier included.".to_string()
        } else {
            "Run dossier omitted.".to_string()
        },
        if config.include_worker_outputs {
            "Worker outcomes included.".to_string()
        } else {
            "Worker outcomes omitted.".to_string()
        },
        if config.include_command_history {
            "Command history excerpt included.".to_string()
        } else {
            "Command history excerpt omitted.".to_string()
        },
        if config.include_internal_transcript {
            format!(
                "Internal transcript excerpt included (max {} chars).",
                config.max_transcript_chars
            )
        } else {
            "Internal transcript excerpt omitted.".to_string()
        },
    ]);
    refresh_dossier_caution_flags(&mut dossier);
    {
        let mut runs = tool_state.active_runs.lock().unwrap();
        if let Some(run) = runs.get_mut(run_id) {
            run.dossier = dossier.clone();
        }
    }

    let prompt_template = if mode == "budget_stop" {
        &config.prompt_budget_stop
    } else {
        &config.prompt_completion
    };
    let prompt = render_prompt_template(
        prompt_template,
        &[
            ("manager_response", manager_response.to_string()),
            ("manager_agent_id", manager_agent_id.to_string()),
            ("finalizer_agent_name", config.agent_name.clone()),
            ("finalizer_mode", mode.to_string()),
            (
                "run_dossier_json",
                if config.include_run_dossier {
                    build_run_dossier_json(&dossier)
                } else {
                    "[omitted]".to_string()
                },
            ),
            (
                "manager_draft_summary",
                if dossier.manager_draft_summary.trim().is_empty() {
                    "[none]".to_string()
                } else {
                    dossier.manager_draft_summary.clone()
                },
            ),
            (
                "worker_outcomes_excerpt",
                if config.include_worker_outputs {
                    format_worker_outcomes_excerpt(&dossier)
                } else {
                    "[omitted]".to_string()
                },
            ),
            (
                "command_history_excerpt",
                if config.include_command_history {
                    tool_state
                        .command_history
                        .lock()
                        .unwrap()
                        .summarize_recent(12)
                } else {
                    "[omitted]".to_string()
                },
            ),
            (
                "internal_transcript",
                if config.include_internal_transcript {
                    format_internal_transcript(&manager_messages, config.max_transcript_chars)
                } else {
                    "[omitted]".to_string()
                },
            ),
            ("claim_calibration", build_claim_calibration(&dossier)),
        ],
    );

    let request_messages = vec![json!({
        "role": "user",
        "content": prompt,
    })];

    let url = format!("{}/v1/chat/completions", lm_base_url());
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    match client
        .post(&url)
        .json(&json!({
            "model": model_key,
            "messages": request_messages,
            "stream": false,
        }))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
            Ok(data) => {
                let text = data["choices"][0]["message"]["content"]
                    .as_str()
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| manager_response.to_string());
                (text, Some(model_key))
            }
            Err(_) => (manager_response.to_string(), Some(model_key)),
        },
        _ => (manager_response.to_string(), Some(model_key)),
    }
}

fn extract_audit_section(response: &str, start: &str, end_markers: &[&str]) -> String {
    let lower = response.to_lowercase();
    let start_lower = start.to_lowercase();
    let Some(start_idx) = lower.find(&start_lower) else {
        return String::new();
    };
    let content_start = response[start_idx..]
        .find('\n')
        .map(|offset| start_idx + offset + 1)
        .unwrap_or(response.len());
    let end_idx = end_markers
        .iter()
        .filter_map(|marker| {
            lower[content_start..]
                .find(&marker.to_lowercase())
                .map(|idx| content_start + idx)
        })
        .min()
        .unwrap_or(response.len());
    response[content_start..end_idx].trim().to_string()
}

fn has_named_section(response: &str, section_name: &str) -> bool {
    response
        .to_lowercase()
        .contains(&section_name.to_lowercase())
}

fn configured_section_names(config: &ResolvedAuditBehaviorConfig) -> Vec<&str> {
    let mut names = config
        .required_sections
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    for rule in &config.section_rules {
        if !names
            .iter()
            .any(|name| name.eq_ignore_ascii_case(&rule.section_name))
        {
            names.push(rule.section_name.as_str());
        }
    }
    names
}

fn extract_configured_section(
    response: &str,
    section_name: &str,
    config: &ResolvedAuditBehaviorConfig,
) -> String {
    let end_markers = configured_section_names(config)
        .into_iter()
        .filter(|name| !name.eq_ignore_ascii_case(section_name))
        .collect::<Vec<_>>();
    extract_audit_section(response, section_name, &end_markers)
}

fn audit_section_bullets(section: &str) -> Vec<String> {
    section
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
        .map(str::to_string)
        .collect()
}

fn missing_required_sections(response: &str, config: &ResolvedAuditBehaviorConfig) -> Vec<String> {
    config
        .required_sections
        .iter()
        .filter(|section| !has_named_section(response, section))
        .cloned()
        .collect()
}

fn section_rule_failed(
    response: &str,
    rule: &crate::agent_config::SectionRule,
    config: &ResolvedAuditBehaviorConfig,
) -> bool {
    let section = extract_configured_section(response, &rule.section_name, config);
    if section.trim().is_empty() {
        return false;
    }
    let lower_name = rule.section_name.to_lowercase();
    let is_hypotheses = lower_name.contains("hypoth");
    let is_confirmed = lower_name.contains("confirmed");
    audit_section_bullets(&section).into_iter().any(|line| {
        let lower = line.to_lowercase();
        let has_disallowed_phrase = rule
            .disallow_template_phrases
            .iter()
            .any(|phrase| lower.contains(&phrase.to_lowercase()));
        if rule.require_file_reference && !has_file_reference(&line) {
            return true;
        }
        if is_hypotheses {
            return hypothesis_line_needs_rewrite(&line, rule);
        }
        if is_confirmed {
            return has_disallowed_phrase || !is_issue_like_statement(&lower);
        }
        has_disallowed_phrase
    })
}

fn audit_section_followup_prompt(
    response: &str,
    config: &ResolvedAuditBehaviorConfig,
) -> Option<String> {
    let missing_sections = missing_required_sections(response, config);
    if !missing_sections.is_empty() {
        return Some(format!(
            "Your response must include these sections exactly: {}. Missing: {}.",
            render_required_sections_csv(config),
            missing_sections.join(", ")
        ));
    }

    for rule in &config.section_rules {
        if section_rule_failed(response, rule, config) {
            let section_content = extract_configured_section(response, &rule.section_name, config);
            if let Some(prompt) = rule.rewrite_loop_prompt.as_deref() {
                return Some(render_audit_template(
                    prompt,
                    &[
                        ("section_name", rule.section_name.clone()),
                        ("section_content", section_content),
                        ("gap_section_label", config.gap_section_label().to_string()),
                    ],
                ));
            }
            return Some(format!(
                "Your `{}` section failed validation. Rework it using only grounded evidence already in context.",
                rule.section_name
            ));
        }
    }

    None
}

fn unresolved_coverage_prompt(
    config: &ResolvedAuditBehaviorConfig,
    missing_scope_refs: &[String],
) -> String {
    render_audit_template(
        config
            .coverage_manifest
            .unresolved_prompt
            .as_deref()
            .unwrap_or("Before concluding, you must either inspect these required audit scopes or name them explicitly under `{gap_section_label}`: {unresolved_scopes_csv}. Do not omit them."),
        &[
            (
                "gap_section_label",
                config.gap_section_label().to_string(),
            ),
            (
                "unresolved_scopes_csv",
                render_csv_list(missing_scope_refs, "none"),
            ),
            (
                "unresolved_scopes_bullets",
                render_bullet_list(missing_scope_refs, "none"),
            ),
        ],
    )
}

pub(crate) fn validate_audit_worker_response(
    response: &str,
    requested_refs: &[String],
    config: &ResolvedAuditBehaviorConfig,
) -> Result<(), String> {
    if response.trim().is_empty() {
        return Err("completed its turn without returning a usable answer".to_string());
    }
    if !requested_refs.is_empty() && !audit_response_acknowledges_refs(response, requested_refs) {
        return Err(format!(
            "did not address all explicitly requested audit scopes: {}.",
            requested_refs.join(", ")
        ));
    }
    if let Some(prompt) = audit_section_followup_prompt(response, config) {
        return Err(format!(
            "returned an audit response that needs revision: {}",
            prompt
        ));
    }
    if audit_response_needs_rewrite(response, classify_audit_evidence(response, config), config) {
        return Err(
            "returned an audit response without grounded, file-backed findings.".to_string(),
        );
    }
    Ok(())
}

fn audit_report_has_issue_like_finding(
    content: &str,
    config: &ResolvedAuditBehaviorConfig,
) -> bool {
    let confirmed_label =
        find_matching_section_name(config, "Confirmed findings", "Confirmed findings");
    let confirmed = extract_configured_section(content, confirmed_label, config);
    audit_section_bullets(&confirmed).into_iter().any(|line| {
        let lower = line.to_lowercase();
        !is_non_issue_observation(&lower) && is_issue_like_statement(&lower)
    })
}

fn hypothesis_line_needs_rewrite(line: &str, rule: &crate::agent_config::SectionRule) -> bool {
    let lower = line.to_lowercase();
    let has_anchor = has_file_reference(line)
        || hypothesis_mentions_observed_behavior(&lower)
        || hypothesis_mentions_missing_proof(&lower);
    let has_disallowed_phrase = rule
        .disallow_template_phrases
        .iter()
        .any(|phrase| lower.contains(&phrase.to_lowercase()))
        || is_template_hypothesis(&lower)
        || is_dependency_presence_hypothesis(&lower);
    has_disallowed_phrase && !has_anchor
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
    config: &ResolvedAuditBehaviorConfig,
) -> bool {
    let Some(response) = extract_delegation_response_text(result) else {
        return false;
    };

    if is_progress_only_response(&response)
        || audit_section_followup_prompt(&response, config).is_some()
        || audit_response_needs_rewrite(
            &response,
            classify_audit_evidence(&response, config),
            config,
        )
    {
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
    let has_new_refs = response_refs.iter().any(|reference| {
        !seen_refs
            .iter()
            .any(|seen| audit_ref_entry_matches(reference, seen))
    });
    let has_new_issue = audit_report_has_issue_like_finding(&response, config);
    let has_evidence =
        classify_audit_evidence(&response, config) >= AuditEvidenceGrade::ConfigContent;
    let has_specific_gap = extract_audit_section(&response, config.gap_section_label(), &[])
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
    run_id: &str,
    agent_id: &str,
    on_event: &SharedEventSink,
    history: &[Value],
    context_limit: Option<u64>,
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", lm_base_url());
    let trimmed_history = trim_history_to_budget(system_prompt, history, message, context_limit, 0);
    let budget = compute_context_budget(system_prompt, &trimmed_history, message, context_limit, 0);
    let effective_system_prompt = with_context_budget(system_prompt, budget);
    let mut messages = vec![];
    if !effective_system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": effective_system_prompt}));
    }
    for h in &trimmed_history {
        messages.push(h.clone());
    }
    messages.push(json!({"role": "user", "content": message}));

    let body = json!({
        "model": model_key,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    let client = reqwest::Client::new();
    let _ = on_event.send(StreamEvent::AgentStatus {
        run_id: run_id.to_string(),
        agent_id: agent_id.to_string(),
        stage: "thinking".to_string(),
        detail: "Generating response".to_string(),
    });
    let streamed_turn =
        stream_chat_completion_turn(&client, &url, &body, run_id, agent_id, on_event)
            .await
            .map_err(|e| format!("Chat failed: {}", e))?;
    emit_generation_metrics(
        on_event,
        run_id,
        agent_id,
        "generated",
        &streamed_turn.usage,
        estimated_stream_output_tokens(&streamed_turn.content),
    );
    Ok(streamed_turn.content)
}

pub(crate) async fn resume_paused_tool_loop(
    tool_state: Arc<McpState>,
    run_id: &str,
    paused: PausedRunState,
) -> Result<(), String> {
    let channel = {
        tool_state
            .active_runs
            .lock()
            .unwrap()
            .get(run_id)
            .ok_or_else(|| format!("Run '{}' not found.", run_id))?
            .channel
            .clone()
    };
    let response = call_chat_with_tools(
        &paused.model_key,
        "",
        "Continue the paused run from the current evidence. Do not restart completed steps.",
        run_id,
        &paused.agent_id,
        &tool_state.tools,
        paused.allowed_tools.as_deref(),
        paused.allow_manager_tools,
        paused.require_delegation,
        paused.context_limit,
        paused.glob_ready,
        tool_state.as_ref(),
        Some(&paused.active_behaviors),
        &channel,
        &paused.messages,
    )
    .await?;

    {
        let mut runs = tool_state.active_runs.lock().unwrap();
        if let Some(run) = runs.remove(run_id) {
            let _ = run.channel.send(StreamEvent::Done {
                run_id: run_id.to_string(),
            });
        }
    }
    let _ = response;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{BehaviorTriggerConfig, BehaviorTriggersConfig};

    fn grounded_config() -> ResolvedAuditBehaviorConfig {
        let config = BehaviorTriggersConfig {
            enabled: true,
            embedding_model_key: None,
            default_similarity_threshold: 0.90,
            behaviors: vec![BehaviorTriggerConfig::default_grounded_audit()],
        };
        let active = HashSet::from([crate::agent_config::GROUNDED_AUDIT_BEHAVIOR_ID.to_string()]);
        resolve_audit_behavior_config(&active, &config).unwrap()
    }

    fn sample_completed_report() -> String {
        "Confirmed findings\n- src-tauri/src/chat.rs: command handling risks resource exhaustion.\n\nHypotheses / lower-confidence risks\n- src-tauri/src/mcp.rs may allow escalation if exposed; insufficient evidence to confirm.\n\nCoverage gaps\n- src/renderer.js".to_string()
    }

    #[test]
    fn template_renderer_replaces_known_placeholders_and_keeps_unknown() {
        let rendered = render_audit_template(
            "{covered_topics_csv} | {gap_section_label} | {unknown}",
            &[
                ("covered_topics_csv", "chat.rs, mcp.rs".to_string()),
                ("gap_section_label", "Coverage gaps".to_string()),
            ],
        );

        assert_eq!(rendered, "chat.rs, mcp.rs | Coverage gaps | {unknown}");
    }

    #[test]
    fn evidence_grading_prefers_code_over_config_and_command() {
        let config = grounded_config();

        assert_eq!(
            classify_audit_evidence(
                "fn example() {}\n\"$schema\": \"value\"\nsrc-tauri/src",
                &config
            ),
            AuditEvidenceGrade::CodeContent
        );
        assert_eq!(
            classify_audit_evidence("package.json references src-tauri/src", &config),
            AuditEvidenceGrade::CommandOnly
        );
    }

    #[test]
    fn section_validation_catches_missing_and_generic_sections() {
        let config = grounded_config();
        let missing_response = "Confirmed findings\n- src/main.rs issue";
        let missing_prompt = audit_section_followup_prompt(missing_response, &config).unwrap();
        assert!(missing_prompt.contains("Missing"));

        let weak_hypothesis = "Confirmed findings\n- src-tauri/src/chat.rs risk\n\nHypotheses / lower-confidence risks\n- may allow compromise if exposed.\n\nCoverage gaps\n- none";
        let hypothesis_prompt = audit_section_followup_prompt(weak_hypothesis, &config).unwrap();
        assert!(hypothesis_prompt.contains("too generic"));
    }

    #[test]
    fn forced_synthesis_and_fallback_use_configured_thresholds() {
        let config = grounded_config();
        let messages = vec![
            json!({"role": "tool", "content": sample_completed_report()}),
            json!({"role": "tool", "content": sample_completed_report()}),
            json!({"role": "tool", "content": sample_completed_report()}),
        ];
        let covered_topics = vec![
            CoveredAuditTopic {
                lexical_signature: "a".to_string(),
                summary: "src-tauri/src/chat.rs".to_string(),
                refs: HashSet::from(["src-tauri/src/chat.rs".to_string()]),
                target_refs: HashSet::from(["src-tauri/src/chat.rs".to_string()]),
                inspection_depth: AuditInspectionDepth::Targeted,
                embedding: None,
            },
            CoveredAuditTopic {
                lexical_signature: "b".to_string(),
                summary: "src-tauri/src/mcp.rs".to_string(),
                refs: HashSet::from(["src-tauri/src/mcp.rs".to_string()]),
                target_refs: HashSet::from(["src-tauri/src/mcp.rs".to_string()]),
                inspection_depth: AuditInspectionDepth::Targeted,
                embedding: None,
            },
            CoveredAuditTopic {
                lexical_signature: "c".to_string(),
                summary: "src/renderer.js".to_string(),
                refs: HashSet::from(["src/renderer.js".to_string()]),
                target_refs: HashSet::from(["src/renderer.js".to_string()]),
                inspection_depth: AuditInspectionDepth::Targeted,
                embedding: None,
            },
        ];

        assert!(should_force_audit_synthesis(
            &messages,
            &covered_topics,
            &[],
            0,
            &RedundancyDetectionConfig::default(),
            Some(&config),
        ));

        let fallback =
            fallback_non_progress_summary(Some(&config), &["src/renderer.js".to_string()]);
        assert!(fallback.contains("src/renderer.js"));
    }

    #[test]
    fn weak_delegation_result_and_worker_validation_follow_configured_rules() {
        let config = grounded_config();
        let weak = serde_json::json!({
            "response": "Confirmed findings\n- None.\n\nHypotheses / lower-confidence risks\n- may allow compromise if exposed.\n\nCoverage gaps\n- none"
        })
        .to_string();

        assert!(is_weak_audit_delegation_result(&weak, &[], &config));
        assert!(validate_audit_worker_response(
            "Confirmed findings\n- src-tauri/src/chat.rs risk\n\nHypotheses / lower-confidence risks\n- may allow compromise if exposed.\n\nCoverage gaps\n- src/renderer.js",
            &["src-tauri/src/chat.rs".to_string()],
            &config
        )
        .is_err());
    }

    #[test]
    fn manager_compaction_detects_repeated_headings() {
        let repeated = "### Summary\nalpha\n\n### Summary\nbeta\n\n### Summary\ngamma";
        assert!(manager_response_needs_compaction(repeated));
    }

    #[test]
    fn manager_compaction_detects_duplicate_long_paragraphs() {
        let paragraph = "The backend state manager currently aggregates multiple responsibilities into one coordination layer, which makes changes ripple across unrelated subsystems and encourages repeated restatements in the final answer.";
        let repeated = format!("{0}\n\n{0}", paragraph);
        assert!(manager_response_needs_compaction(&repeated));
    }

    #[test]
    fn manager_compaction_skips_short_coherent_answers() {
        let coherent = "The repo has partial Rust tests, no frontend tests, and no CI. The next step is to add targeted coverage and automate the checks.";
        assert!(!manager_response_needs_compaction(coherent));
    }

    #[test]
    fn lexical_redundancy_detection_works_without_embeddings() {
        let config = RedundancyDetectionConfig {
            enabled: true,
            embedding_model_key: None,
            semantic_similarity_threshold: 0.9,
            max_redundant_audit_retries: 1,
        };
        let existing = CoveredAuditTopic {
            lexical_signature: "read_file | src-tauri/src/chat.rs".to_string(),
            summary: "chat.rs".to_string(),
            refs: HashSet::from(["src-tauri/src/chat.rs".to_string()]),
            target_refs: HashSet::from(["src-tauri/src/chat.rs".to_string()]),
            inspection_depth: AuditInspectionDepth::Targeted,
            embedding: None,
        };
        let candidate = CoveredAuditTopic {
            lexical_signature: "read_file | src-tauri/src/chat.rs".to_string(),
            summary: "chat.rs again".to_string(),
            refs: HashSet::from(["src-tauri/src/chat.rs".to_string()]),
            target_refs: HashSet::from(["src-tauri/src/chat.rs".to_string()]),
            inspection_depth: AuditInspectionDepth::Targeted,
            embedding: None,
        };

        let reason = detect_redundant_audit_topic(&candidate, &[existing], &config);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("same lexical topic"));
    }

    #[test]
    fn lexical_redundancy_still_applies_when_embeddings_are_missing() {
        let config = RedundancyDetectionConfig {
            enabled: true,
            embedding_model_key: Some("embedding-model".to_string()),
            semantic_similarity_threshold: 0.9,
            max_redundant_audit_retries: 1,
        };
        let existing = CoveredAuditTopic {
            lexical_signature: "grep_search | src/App.tsx".to_string(),
            summary: "App.tsx".to_string(),
            refs: HashSet::from(["src/App.tsx".to_string()]),
            target_refs: HashSet::from(["src/App.tsx".to_string()]),
            inspection_depth: AuditInspectionDepth::EvidenceBacked,
            embedding: None,
        };
        let candidate = CoveredAuditTopic {
            lexical_signature: "grep_search | src/App.tsx".to_string(),
            summary: "App.tsx follow-up".to_string(),
            refs: HashSet::from(["src/App.tsx".to_string()]),
            target_refs: HashSet::from(["src/App.tsx".to_string()]),
            inspection_depth: AuditInspectionDepth::Targeted,
            embedding: None,
        };

        let reason = detect_redundant_audit_topic(&candidate, &[existing], &config);
        assert!(reason.is_some());
    }
}
