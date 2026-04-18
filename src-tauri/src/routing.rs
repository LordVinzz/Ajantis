use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tauri::ipc::Channel;

use crate::agent_config::{AgentConfig, AgentLoadConfig};
use crate::chat::{call_chat_with_tools, send_chat_completion_streaming, StreamEvent};
use crate::helpers::{apply_runtime_agent_context, is_manager_only_tool};
use crate::memory::{CommandHistory, MemoryPool};
use crate::models::{fetch_models, load_model_internal, unload_model_internal};
use crate::state::{AppState, McpTool};

#[tauri::command]
pub(crate) async fn route_message(
    state: tauri::State<'_, Arc<AppState>>,
    from_agent_id: String,
    message: String,
    on_event: Channel<StreamEvent>,
) -> Result<(), String> {
    let config = state.agent_config.lock().unwrap().clone();
    let memory = state.memory_pool.clone();
    let command_history = state.command_history.clone();
    let glob_cache = state.glob_cache.clone();
    let mcp_state = state.mcp_state.clone();
    let tools_arc = Arc::new(state.mcp_tools.clone());
    let mut visited: HashSet<String> = HashSet::new();

    let model_info_map: HashMap<String, (String, Option<u64>)> = match fetch_models().await {
        Ok(models) => models
            .into_iter()
            .filter_map(|m| m.model_type.map(|t| (m.key, (t, m.max_context_length))))
            .collect(),
        Err(_) => HashMap::new(),
    };

    // Make the channel available to MCP tool handlers (e.g. send_message can emit sub-agent bubbles)
    *state.event_channel.lock().unwrap() = Some(on_event.clone());

    route_recursive(
        &config,
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

    *state.event_channel.lock().unwrap() = None;
    let _ = on_event.send(StreamEvent::Done);
    Ok(())
}

#[async_recursion::async_recursion]
pub(crate) async fn route_recursive(
    config: &AgentConfig,
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
                            agent.allowed_tools
                                .clone()
                                .unwrap_or_else(|| tools.iter()
                                    .filter(|tool| !is_manager_only_tool(&tool.name))
                                    .map(|tool| tool.name.clone())
                                    .collect())
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
                agent_id: target_id.clone(),
                agent_name: agent_name.clone(),
                message: format!("Agent '{}' has no model configured.", agent_name),
            });
            continue;
        }

        if mode == "on_the_fly" {
            if let Err(e) = load_model_internal(&load_cfg, &model_key).await {
                let _ = on_event.send(StreamEvent::Error {
                    agent_id: target_id.clone(),
                    agent_name: agent_name.clone(),
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
            pool.entries
                .iter()
                .filter(|e| e.agent_id == target_id && (e.role == "user" || e.role == "assistant"))
                .map(|e| json!({"role": e.role, "content": e.content}))
                .collect()
        };

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
                &target_id,
                &tools,
                if is_manager { None } else { Some(allowed_tools.as_slice()) },
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
                    agent_id: target_id.clone(),
                });
                // Store both sides of this turn so history is complete for the next message
                {
                    let mut pool = memory.lock().unwrap();
                    pool.push(&target_id, &agent_name, "user", message);
                    pool.push(&target_id, &agent_name, "assistant", &full_response);
                }
                if mode == "on_the_fly" {
                    let _ = unload_model_internal(&model_key).await;
                }
                route_recursive(
                    config,
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
                if mode == "on_the_fly" {
                    let _ = unload_model_internal(&model_key).await;
                }
                let _ = on_event.send(StreamEvent::Error {
                    agent_id: target_id.clone(),
                    agent_name: agent_name.clone(),
                    message: e,
                });
            }
        }
    }
}
