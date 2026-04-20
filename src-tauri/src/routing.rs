use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tauri::ipc::Channel;

use crate::agent_config::{AgentConfig, AgentLoadConfig};
use crate::chat::{call_chat_with_tools, send_chat_completion_streaming, StreamEvent};
use crate::helpers::{apply_runtime_agent_context, compute_context_budget, is_manager_only_tool};
use crate::memory::{CommandHistory, MemoryPool};
use crate::models::{create_embeddings, fetch_models, load_model_internal, unload_model_internal};
use crate::runs::{
    emit_run_event, generate_run_id, journal_path, reset_run_window, ActiveRunState,
};
use crate::state::{AppState, McpTool};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn jaccard_word_similarity(a: &str, b: &str) -> f32 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Returns true if `response` is semantically close to any of the agent's recent responses.
/// Uses embedding cosine similarity when a model is available, Jaccard word overlap otherwise.
/// Embeds the first 1500 characters to keep token usage bounded.
async fn is_repetitive_response(
    response: &str,
    run_id: &str,
    mcp_state: &crate::mcp::McpState,
) -> bool {
    if response.len() < 100 {
        return false; // Too short to be a meaningful repetition
    }

    let embed_model = {
        let cfg = mcp_state.agent_config.lock().unwrap();
        cfg.behavior_triggers.embedding_model_key.clone()
    };

    if let Some(model_key) = embed_model {
        let sample: String = response.chars().take(1500).collect();
        if let Ok(embeddings) = create_embeddings(&model_key, &[sample]).await {
            if let Some(new_embed) = embeddings.into_iter().next() {
                let max_sim = {
                    let runs = mcp_state.active_runs.lock().unwrap();
                    runs.get(run_id)
                        .map(|run| {
                            run.recent_response_embeddings
                                .iter()
                                .map(|e| cosine_similarity(e, &new_embed))
                                .fold(0.0f32, f32::max)
                        })
                        .unwrap_or(0.0)
                };
                {
                    let mut runs = mcp_state.active_runs.lock().unwrap();
                    if let Some(run) = runs.get_mut(run_id) {
                        run.usage.embedding_calls += 1;
                        run.recent_response_embeddings.push(new_embed);
                        if run.recent_response_embeddings.len() > 6 {
                            run.recent_response_embeddings.remove(0);
                        }
                    }
                }
                return max_sim > 0.90;
            }
        }
        // Embedding call failed; fall through to Jaccard
    }

    // Jaccard fallback
    let max_sim = {
        let runs = mcp_state.active_runs.lock().unwrap();
        runs.get(run_id)
            .map(|run| {
                run.recent_response_texts
                    .iter()
                    .map(|prev| jaccard_word_similarity(prev, response))
                    .fold(0.0f32, f32::max)
            })
            .unwrap_or(0.0)
    };
    {
        let mut runs = mcp_state.active_runs.lock().unwrap();
        if let Some(run) = runs.get_mut(run_id) {
            let trimmed: String = response.chars().take(3000).collect();
            run.recent_response_texts.push(trimmed);
            if run.recent_response_texts.len() > 6 {
                run.recent_response_texts.remove(0);
            }
        }
    }
    max_sim > 0.72
}

#[tauri::command]
pub(crate) async fn route_message(
    state: tauri::State<'_, Arc<AppState>>,
    from_agent_id: String,
    message: String,
    workspace_id: Option<String>,
    thread_id: Option<String>,
    on_event: Channel<StreamEvent>,
) -> Result<String, String> {
    let config = state.agent_config.lock().unwrap().clone();
    let memory = state.memory_pool.clone();
    let command_history = state.command_history.clone();
    let glob_cache = state.glob_cache.clone();
    let mcp_state = state.mcp_state.clone();
    let tools_arc = Arc::new(state.mcp_tools.clone());
    let mut visited: HashSet<String> = HashSet::new();
    let run_id = generate_run_id();
    let journal_path = journal_path(
        workspace_id.as_deref().unwrap_or("unscoped"),
        thread_id.as_deref().unwrap_or("default"),
        &run_id,
    );

    let model_info_map: HashMap<String, (String, Option<u64>)> = match fetch_models().await {
        Ok(models) => models
            .into_iter()
            .filter_map(|m| m.model_type.map(|t| (m.key, (t, m.max_context_length))))
            .collect(),
        Err(_) => HashMap::new(),
    };

    // Make the channel available to MCP tool handlers (e.g. send_message can emit sub-agent bubbles)
    *state.event_channel.lock().unwrap() = Some(on_event.clone());
    state.active_runs.lock().unwrap().insert(
        run_id.clone(),
        ActiveRunState {
            run_id: run_id.clone(),
            workspace_id,
            thread_id,
            workspace_path: Some(
                state
                    .active_workspace
                    .lock()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
            ),
            journal_path,
            channel: on_event.clone(),
            budgets: config.run_budgets.clone(),
            active_behaviors: HashSet::new(),
            usage: crate::runs::RunWindowUsage::default(),
            window_started_at: std::time::Instant::now(),
            paused: None,
            waiting_confirmation: false,
            cancelled: false,
            recent_response_embeddings: Vec::new(),
            recent_response_texts: Vec::new(),
            manager_agent_id: None,
            manager_model_key: None,
            manager_messages: Vec::new(),
        },
    );

    route_recursive(
        &config,
        &run_id,
        &from_agent_id,
        &message,
        &on_event,
        &mut visited,
        &memory,
        &command_history,
        &glob_cache,
        &mcp_state,
        &model_info_map,
        tools_arc,
    )
    .await;
    let waiting_confirmation = state
        .active_runs
        .lock()
        .unwrap()
        .get(&run_id)
        .map(|run| run.waiting_confirmation)
        .unwrap_or(false);
    if waiting_confirmation {
        return Ok(run_id);
    }

    state.active_runs.lock().unwrap().remove(&run_id);
    *state.event_channel.lock().unwrap() = None;
    let _ = emit_run_event(
        &state.active_runs,
        &run_id,
        StreamEvent::Done {
            run_id: run_id.clone(),
        },
    );
    let _ = on_event.send(StreamEvent::Done {
        run_id: run_id.clone(),
    });
    Ok(run_id)
}

#[async_recursion::async_recursion]
pub(crate) async fn route_recursive(
    config: &AgentConfig,
    run_id: &str,
    from_id: &str,
    message: &str,
    on_event: &Channel<StreamEvent>,
    visited: &mut HashSet<String>,
    memory: &Arc<Mutex<MemoryPool>>,
    command_history: &Arc<Mutex<CommandHistory>>,
    glob_cache: &Arc<Mutex<HashMap<String, HashSet<String>>>>,
    mcp_state: &Arc<crate::mcp::McpState>,
    model_info_map: &HashMap<String, (String, Option<u64>)>,
    tools: Arc<Vec<McpTool>>,
) {
    // Collect enabled targets sorted by priority (lower = first)
    let mut targets: Vec<(String, u8)> = config
        .connections
        .iter()
        .filter(|c| c.from == from_id && c.enabled)
        .filter(|c| match &c.condition {
            Some(cond) if !cond.is_empty() => message.to_lowercase().contains(&cond.to_lowercase()),
            _ => true,
        })
        .map(|c| (c.to.clone(), c.priority))
        .collect();
    targets.sort_by_key(|(_, p)| *p);

    for (target_id, _) in targets {
        let visit_key = format!("{}→{}", from_id, target_id);
        if visited.contains(&visit_key) {
            continue;
        }
        visited.insert(visit_key);

        // Extract all agent fields before any await to release the borrow cleanly
        let (
            agent_name,
            model_key,
            mode,
            system_prompt,
            load_cfg,
            is_manager,
            allowed_tools,
            model_type_resolved,
            context_limit,
        ) = {
            match config.agents.iter().find(|a| a.id == target_id) {
                None => continue,
                Some(agent) => {
                    if agent.agent_type == "user" || !agent.armed {
                        continue;
                    }
                    if agent.paused {
                        let _ = on_event.send(StreamEvent::Error {
                            run_id: run_id.to_string(),
                            agent_id: target_id.clone(),
                            agent_name: agent.name.clone(),
                            message: format!("Agent '{}' is paused.", agent.name),
                        });
                        continue;
                    }
                    let resolved = agent.model_type.clone().or_else(|| {
                        agent.model_key.as_deref().and_then(|k| {
                            model_info_map
                                .get(k)
                                .map(|(model_type, _)| model_type.clone())
                        })
                    });
                    let max_context = agent
                        .model_key
                        .as_deref()
                        .and_then(|k| model_info_map.get(k).and_then(|(_, max_ctx)| *max_ctx));
                    (
                        agent.name.clone(),
                        agent.model_key.clone().unwrap_or_default(),
                        agent.mode.as_deref().unwrap_or("stay_awake").to_string(),
                        apply_runtime_agent_context(
                            &agent.role.clone().unwrap_or_default(),
                            agent.is_manager,
                            &command_history.lock().unwrap().clone(),
                        ),
                        agent.load_config.clone().unwrap_or(AgentLoadConfig {
                            context_length: None,
                            eval_batch_size: None,
                            flash_attention: None,
                            num_experts: None,
                            offload_kv_cache_to_gpu: None,
                        }),
                        agent.is_manager,
                        if agent.is_manager {
                            Vec::new()
                        } else {
                            agent.allowed_tools.clone().unwrap_or_else(|| {
                                tools
                                    .iter()
                                    .filter(|tool| !is_manager_only_tool(&tool.name))
                                    .map(|tool| tool.name.clone())
                                    .collect()
                            })
                        },
                        resolved,
                        agent
                            .load_config
                            .as_ref()
                            .and_then(|cfg| cfg.context_length)
                            .or(max_context),
                    )
                }
            }
        };

        if model_type_resolved.as_deref() == Some("embedding") {
            let _ = on_event.send(StreamEvent::Error {
                run_id: run_id.to_string(),
                agent_id: target_id.clone(),
                agent_name: agent_name.clone(),
                message: format!(
                    "Agent '{}' uses an embedding model — not usable for chat.",
                    agent_name
                ),
            });
            continue;
        }

        if model_key.is_empty() {
            let _ = on_event.send(StreamEvent::Error {
                run_id: run_id.to_string(),
                agent_id: target_id.clone(),
                agent_name: agent_name.clone(),
                message: format!("Agent '{}' has no model configured.", agent_name),
            });
            continue;
        }

        if mode == "on_the_fly" {
            let _ = on_event.send(StreamEvent::AgentStatus {
                run_id: run_id.to_string(),
                agent_id: target_id.clone(),
                stage: "loading_model".to_string(),
                detail: format!("Loading {} on demand", model_key),
            });
            if let Err(e) = load_model_internal(&load_cfg, &model_key).await {
                let _ = on_event.send(StreamEvent::Error {
                    run_id: run_id.to_string(),
                    agent_id: target_id.clone(),
                    agent_name: agent_name.clone(),
                    message: format!("Failed to load model: {}", e),
                });
                continue;
            }
        }

        // Reconstruct conversation history for this agent (all previous user/assistant turns)
        let history: Vec<Value> = {
            let pool = memory.lock().unwrap();
            pool.entries
                .iter()
                .filter(|e| e.agent_id == target_id && (e.role == "user" || e.role == "assistant"))
                .map(|e| json!({"role": e.role, "content": e.content}))
                .collect()
        };

        let startup_budget =
            compute_context_budget(&system_prompt, &history, message, context_limit, 0);

        let _ = on_event.send(StreamEvent::AgentStart {
            run_id: run_id.to_string(),
            agent_id: target_id.clone(),
            agent_name: agent_name.clone(),
            model_key: model_key.clone(),
            mode: mode.clone(),
            is_manager,
            context_limit: startup_budget.limit,
            estimated_input_tokens: startup_budget.estimated_used,
            estimated_remaining_tokens: startup_budget.remaining,
        });

        let result = if is_manager || !allowed_tools.is_empty() {
            let glob_ready = glob_cache
                .lock()
                .unwrap()
                .get(&target_id)
                .map(|matches| !matches.is_empty())
                .unwrap_or(false);
            call_chat_with_tools(
                &model_key,
                &system_prompt,
                message,
                run_id,
                &target_id,
                &tools,
                if is_manager {
                    None
                } else {
                    Some(allowed_tools.as_slice())
                },
                is_manager,
                is_manager,
                context_limit,
                glob_ready,
                mcp_state.as_ref(),
                None,
                on_event,
                &history,
            )
            .await
        } else {
            send_chat_completion_streaming(
                &model_key,
                &system_prompt,
                message,
                run_id,
                &target_id,
                on_event,
                &history,
                context_limit,
            )
            .await
        };

        match result {
            Ok(full_response) => {
                let _ = on_event.send(StreamEvent::AgentEnd {
                    run_id: run_id.to_string(),
                    agent_id: target_id.clone(),
                });
                // Store both sides of this turn so history is complete for the next message
                {
                    let mut pool = memory.lock().unwrap();
                    pool.push(&target_id, &agent_name, "user", message);
                    pool.push(&target_id, &agent_name, "assistant", &full_response);
                }
                if mode == "on_the_fly" {
                    let _ = on_event.send(StreamEvent::AgentStatus {
                        run_id: run_id.to_string(),
                        agent_id: target_id.clone(),
                        stage: "unloading_model".to_string(),
                        detail: format!("Unloading {}", model_key),
                    });
                    let _ = unload_model_internal(&model_key).await;
                }
                if is_repetitive_response(&full_response, run_id, mcp_state).await {
                    // Agent is looping — stop routing without re-routing the repeated response.
                    return;
                }
                route_recursive(
                    config,
                    run_id,
                    &target_id,
                    &full_response,
                    on_event,
                    visited,
                    memory,
                    command_history,
                    glob_cache,
                    mcp_state,
                    model_info_map,
                    tools.clone(),
                )
                .await;
            }
            Err(e) => {
                if e == crate::chat::RUN_PAUSED_ERROR {
                    return;
                }
                if e == crate::chat::RUN_BUDGET_ENDED_ERROR {
                    // Summary was already streamed as tokens; emit AgentEnd and stop routing.
                    let _ = on_event.send(StreamEvent::AgentEnd {
                        run_id: run_id.to_string(),
                        agent_id: target_id.clone(),
                    });
                    return;
                }
                if mode == "on_the_fly" {
                    let _ = on_event.send(StreamEvent::AgentStatus {
                        run_id: run_id.to_string(),
                        agent_id: target_id.clone(),
                        stage: "unloading_model".to_string(),
                        detail: format!("Unloading {}", model_key),
                    });
                    let _ = unload_model_internal(&model_key).await;
                }
                let _ = on_event.send(StreamEvent::Error {
                    run_id: run_id.to_string(),
                    agent_id: target_id.clone(),
                    agent_name: agent_name.clone(),
                    message: e,
                });
            }
        }
    }
}

#[tauri::command]
pub(crate) async fn cancel_route_run(
    state: tauri::State<'_, Arc<AppState>>,
    run_id: String,
) -> Result<(), String> {
    let maybe_run = state.active_runs.lock().unwrap().remove(&run_id);
    let Some(run) = maybe_run else {
        return Err(format!("Run '{}' not found.", run_id));
    };
    let _ = run.channel.send(StreamEvent::RunCancelled {
        run_id: run_id.clone(),
        message: "Run cancelled by user.".to_string(),
    });
    let _ = run.channel.send(StreamEvent::Done {
        run_id: run_id.clone(),
    });
    *state.event_channel.lock().unwrap() = None;
    Ok(())
}

#[tauri::command]
pub(crate) async fn continue_route_run(
    state: tauri::State<'_, Arc<AppState>>,
    run_id: String,
) -> Result<(), String> {
    let paused = {
        let mut runs = state.active_runs.lock().unwrap();
        let run = runs
            .get_mut(&run_id)
            .ok_or_else(|| format!("Run '{}' not found.", run_id))?;
        let paused = run
            .paused
            .clone()
            .ok_or_else(|| format!("Run '{}' is not waiting for confirmation.", run_id))?;
        run.waiting_confirmation = false;
        run.paused = None;
        reset_run_window(run);
        let _ = run.channel.send(StreamEvent::RunResumed {
            run_id: run_id.clone(),
            message: "Run resumed with a fresh soft-budget window.".to_string(),
        });
        paused
    };

    let result =
        crate::chat::resume_paused_tool_loop(state.mcp_state.clone(), &run_id, paused).await;
    if state.active_runs.lock().unwrap().get(&run_id).is_none() {
        *state.event_channel.lock().unwrap() = None;
    }
    result
}
