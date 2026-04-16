use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crate::agent_config::AgentConfig;
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

#[tauri::command]
pub(crate) async fn save_agent_config(
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
pub(crate) async fn load_agent_config(state: tauri::State<'_, Arc<AppState>>) -> Result<AgentConfig, String> {
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
