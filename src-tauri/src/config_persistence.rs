use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::agent_config::{
    AgentConfig, BehaviorTriggerConfig, GROUNDED_AUDIT_BEHAVIOR_ID,
    MAX_SEMANTIC_SIMILARITY_THRESHOLD, MIN_SEMANTIC_SIMILARITY_THRESHOLD,
};
use crate::helpers::{canonical_manager_role_prompt, manager_prompt_needs_grounding};
use crate::state::AppState;

/// Returns `~/.ajantis/`, creating it if needed.
pub(crate) fn ajantis_dir() -> PathBuf {
    let dir = dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ajantis");
    let _ = fs::create_dir_all(&dir);
    dir
}

pub(crate) fn config_path(_workspace: &PathBuf) -> PathBuf {
    ajantis_dir().join("ajantis-config.json")
}

pub(crate) fn load_agent_config_from_disk(workspace_root: &PathBuf) -> AgentConfig {
    let path = config_path(workspace_root);
    let mut missing_redundancy_config = false;
    let mut missing_behavior_triggers = false;
    let config = if path.exists() {
        fs::read_to_string(&path)
            .ok()
            .map(|content| {
                let parsed = serde_json::from_str::<Value>(&content)
                    .ok()
                    .and_then(|value| value.as_object().cloned())
                    .unwrap_or_default();
                missing_redundancy_config = !parsed.contains_key("redundancy_detection");
                missing_behavior_triggers = !parsed.contains_key("behavior_triggers");
                serde_json::from_str(&content).unwrap_or_default()
            })
            .unwrap_or_default()
    } else {
        AgentConfig::default()
    };

    let (normalized, changed) = normalize_agent_config(config);
    let changed = changed || missing_redundancy_config || missing_behavior_triggers;
    if changed {
        let _ = write_agent_config_to_disk(workspace_root, &normalized);
    }
    normalized
}

fn normalize_agent_config(mut config: AgentConfig) -> (AgentConfig, bool) {
    let mut changed = false;
    let threshold = config.redundancy_detection.semantic_similarity_threshold;
    let normalized_threshold = threshold
        .clamp(MIN_SEMANTIC_SIMILARITY_THRESHOLD, MAX_SEMANTIC_SIMILARITY_THRESHOLD);
    if (normalized_threshold - threshold).abs() > f32::EPSILON {
        config.redundancy_detection.semantic_similarity_threshold = normalized_threshold;
        changed = true;
    }
    let behavior_threshold = config.behavior_triggers.default_similarity_threshold;
    let normalized_behavior_threshold = behavior_threshold.clamp(0.0, 1.0);
    if (normalized_behavior_threshold - behavior_threshold).abs() > f32::EPSILON {
        config.behavior_triggers.default_similarity_threshold = normalized_behavior_threshold;
        changed = true;
    }
    if !config
        .behavior_triggers
        .behaviors
        .iter()
        .any(|behavior| behavior.behavior_id == GROUNDED_AUDIT_BEHAVIOR_ID)
    {
        config
            .behavior_triggers
            .behaviors
            .push(BehaviorTriggerConfig::default_grounded_audit());
        changed = true;
    }
    for behavior in &mut config.behavior_triggers.behaviors {
        if let Some(threshold) = behavior.similarity_threshold {
            let normalized = threshold.clamp(0.0, 1.0);
            if (normalized - threshold).abs() > f32::EPSILON {
                behavior.similarity_threshold = Some(normalized);
                changed = true;
            }
        }
    }
    for agent in &mut config.agents {
        if agent.is_manager {
            let current_role = agent.role.as_deref().unwrap_or("");
            if manager_prompt_needs_grounding(current_role) {
                let grounded = canonical_manager_role_prompt().to_string();
                if agent.role.as_deref() != Some(grounded.as_str()) {
                    agent.role = Some(grounded);
                    changed = true;
                }
            }
        }
    }
    (config, changed)
}

fn write_agent_config_to_disk(workspace_root: &PathBuf, config: &AgentConfig) -> Result<(), String> {
    let path = config_path(workspace_root);
    let json = serde_json::to_string_pretty(config)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write config: {}", e))
}

#[tauri::command]
pub(crate) async fn save_agent_config(
    state: tauri::State<'_, Arc<AppState>>,
    config: AgentConfig,
) -> Result<(), String> {
    let (normalized, _) = normalize_agent_config(config);
    *state.agent_config.lock().unwrap() = normalized.clone();
    state
        .mcp_state
        .behavior_trigger_cache
        .lock()
        .unwrap()
        .embeddings
        .clear();
    state
        .mcp_state
        .active_behavior_contexts
        .lock()
        .unwrap()
        .clear();
    write_agent_config_to_disk(&state.workspace_root, &normalized)
}

#[tauri::command]
pub(crate) async fn load_agent_config(state: tauri::State<'_, Arc<AppState>>) -> Result<AgentConfig, String> {
    let config = load_agent_config_from_disk(&state.workspace_root);
    *state.agent_config.lock().unwrap() = config.clone();
    Ok(config)
}
