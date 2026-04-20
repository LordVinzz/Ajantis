use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::agent_config::{
    AgentConfig, BehaviorTriggerConfig, DEFAULT_THEME, GROUNDED_AUDIT_BEHAVIOR_ID,
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
    let mut missing_run_budgets = false;
    let mut missing_theme = false;
    let mut parsed_config = None;
    let config = if path.exists() {
        fs::read_to_string(&path)
            .ok()
            .map(|content| {
                let parsed = serde_json::from_str::<Value>(&content)
                    .ok()
                    .and_then(|value| value.as_object().cloned())
                    .unwrap_or_default();
                parsed_config = Some(parsed.clone());
                missing_redundancy_config = !parsed.contains_key("redundancy_detection");
                missing_behavior_triggers = !parsed.contains_key("behavior_triggers");
                missing_run_budgets = !parsed.contains_key("run_budgets");
                missing_theme = !parsed.contains_key("theme");
                serde_json::from_str(&content).unwrap_or_default()
            })
            .unwrap_or_default()
    } else {
        AgentConfig::default()
    };

    let mut config = config;
    let mut changed = parsed_config
        .as_ref()
        .map(|parsed| hydrate_grounded_audit_behavior(&mut config, parsed))
        .unwrap_or(false);
    let (normalized, normalized_changed) = normalize_agent_config(config);
    changed = changed
        || normalized_changed
        || missing_redundancy_config
        || missing_behavior_triggers
        || missing_run_budgets
        || missing_theme;
    if changed {
        let _ = write_agent_config_to_disk(workspace_root, &normalized);
    }
    normalized
}

fn normalize_agent_config(mut config: AgentConfig) -> (AgentConfig, bool) {
    let mut changed = false;
    let normalized_theme = match config.theme.as_str() {
        "win98" | "ubuntu" | "macos" => config.theme.clone(),
        _ => DEFAULT_THEME.to_string(),
    };
    if config.theme != normalized_theme {
        config.theme = normalized_theme;
        changed = true;
    }
    let threshold = config.redundancy_detection.semantic_similarity_threshold;
    let normalized_threshold = threshold.clamp(
        MIN_SEMANTIC_SIMILARITY_THRESHOLD,
        MAX_SEMANTIC_SIMILARITY_THRESHOLD,
    );
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

fn hydrate_grounded_audit_behavior(
    config: &mut AgentConfig,
    parsed: &serde_json::Map<String, Value>,
) -> bool {
    let Some(behaviors) = parsed
        .get("behavior_triggers")
        .and_then(|value| value.get("behaviors"))
        .and_then(Value::as_array)
    else {
        return false;
    };

    let mut changed = false;
    for behavior in &mut config.behavior_triggers.behaviors {
        if behavior.behavior_id != GROUNDED_AUDIT_BEHAVIOR_ID {
            continue;
        }
        let raw_behavior = behaviors.iter().find(|value| {
            value.get("behavior_id").and_then(Value::as_str) == Some(GROUNDED_AUDIT_BEHAVIOR_ID)
        });
        let Some(raw_behavior) = raw_behavior.and_then(Value::as_object) else {
            continue;
        };
        let needs_hydration = [
            "system_prompt_injection",
            "runtime_note_enabled",
            "runtime_note_template",
            "coverage_manifest",
            "required_sections",
            "section_rules",
            "response_rewrite",
            "evidence_grading",
            "force_synthesis",
            "delegation_validation",
            "tool_burst_reflection",
            "non_progress",
        ]
        .iter()
        .any(|key| !raw_behavior.contains_key(*key));
        if !needs_hydration {
            continue;
        }

        let defaults = BehaviorTriggerConfig::default_grounded_audit();
        behavior.system_prompt_injection = defaults.system_prompt_injection;
        behavior.runtime_note_enabled = defaults.runtime_note_enabled;
        behavior.runtime_note_template = defaults.runtime_note_template;
        behavior.coverage_manifest = defaults.coverage_manifest;
        behavior.required_sections = defaults.required_sections;
        behavior.section_rules = defaults.section_rules;
        behavior.response_rewrite = defaults.response_rewrite;
        behavior.evidence_grading = defaults.evidence_grading;
        behavior.force_synthesis = defaults.force_synthesis;
        behavior.delegation_validation = defaults.delegation_validation;
        behavior.tool_burst_reflection = defaults.tool_burst_reflection;
        behavior.non_progress = defaults.non_progress;
        changed = true;
    }

    changed
}

fn write_agent_config_to_disk(
    workspace_root: &PathBuf,
    config: &AgentConfig,
) -> Result<(), String> {
    let path = config_path(workspace_root);
    let json =
        serde_json::to_string_pretty(config).map_err(|e| format!("Serialization failed: {}", e))?;
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
pub(crate) async fn load_agent_config(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<AgentConfig, String> {
    let config = load_agent_config_from_disk(&state.workspace_root);
    *state.agent_config.lock().unwrap() = config.clone();
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrates_old_grounded_audit_entries_with_new_defaults() {
        let raw = serde_json::json!({
            "agents": [{
                "id": "user",
                "name": "User",
                "type": "user",
                "armed": true,
                "is_manager": false
            }],
            "connections": [],
            "behavior_triggers": {
                "behaviors": [{
                    "behavior_id": "grounded_audit",
                    "enabled": true,
                    "keyword_triggers": ["security audit"],
                    "embedding_trigger_phrases": [],
                    "similarity_threshold": null
                }]
            }
        });
        let parsed = raw.as_object().cloned().unwrap();
        let mut config: AgentConfig = serde_json::from_value(raw).unwrap();

        assert!(config.behavior_triggers.behaviors[0]
            .system_prompt_injection
            .is_none());

        let changed = hydrate_grounded_audit_behavior(&mut config, &parsed);

        assert!(changed);
        let behavior = &config.behavior_triggers.behaviors[0];
        assert!(behavior.runtime_note_enabled);
        assert!(behavior.coverage_manifest.enabled);
        assert_eq!(
            behavior.required_sections,
            vec![
                "Confirmed findings".to_string(),
                "Hypotheses / lower-confidence risks".to_string(),
                "Coverage gaps".to_string()
            ]
        );
        assert_eq!(behavior.non_progress.limit, Some(4));
    }
}
