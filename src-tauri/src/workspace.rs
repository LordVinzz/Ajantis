use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::state::AppState;

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct WorkspaceThread {
    pub(crate) id: String,
    pub(crate) name: String,
    /// Persisted HTML snapshot of the thread conversation.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) messages: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Workspace {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) path: String,
    #[serde(default)]
    pub(crate) threads: Vec<WorkspaceThread>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct WorkspaceConfig {
    workspaces: Vec<Workspace>,
}

pub(crate) fn workspace_config_path(workspace: &PathBuf) -> PathBuf {
    workspace.join("workspace_config.json")
}

/// Called by the frontend when the user selects a workspace.
/// Updates the shared active_workspace path used by MCP tool sandboxing.
#[tauri::command]
pub(crate) async fn set_active_workspace(
    state: tauri::State<'_, Arc<AppState>>,
    path: Option<String>,
) -> Result<(), String> {
    let new_path = match path {
        Some(ref p) if !p.is_empty() => PathBuf::from(p),
        _ => state.workspace_root.clone(),
    };
    *state.active_workspace.lock().unwrap() = new_path;
    Ok(())
}

#[tauri::command]
pub(crate) async fn pick_folder() -> Result<Option<String>, String> {
    let handle = rfd::AsyncFileDialog::new()
        .set_title("Choose a workspace folder")
        .pick_folder()
        .await;
    Ok(handle.map(|f| f.path().to_string_lossy().to_string()))
}

#[tauri::command]
pub(crate) async fn load_workspace_config(
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
pub(crate) async fn save_workspace_config(
    state: tauri::State<'_, Arc<AppState>>,
    config: WorkspaceConfig,
) -> Result<(), String> {
    let path = workspace_config_path(&state.workspace_root);
    let json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write workspace config: {}", e))
}
