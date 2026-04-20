use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::memory::{CommandExecution, MemoryEntry};
use crate::runs::{snapshot_path, thread_data_root};
use crate::state::AppState;

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct WorkspaceToolCall {
    pub(crate) tool_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) args: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) result: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) semantic: bool,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct WorkspaceMessageSignal {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) kind: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) text: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct WorkspaceThreadMessage {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) agent_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) is_sent: bool,
    #[serde(default)]
    pub(crate) is_error: bool,
    #[serde(default)]
    pub(crate) for_user: bool,
    #[serde(default)]
    pub(crate) internal: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<WorkspaceToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) signal: Option<WorkspaceMessageSignal>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct WorkspaceThread {
    pub(crate) id: String,
    pub(crate) name: String,
    /// Legacy persisted HTML snapshot of the thread conversation.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) messages: String,
    /// Structured thread messages used to reconstruct the UI.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) message_items: Vec<WorkspaceThreadMessage>,
    /// Persisted backend conversation state for the thread.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) memory_entries: Vec<MemoryEntry>,
    /// Persisted thread-scoped command execution history.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) command_history: Vec<CommandExecution>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct ThreadSnapshot {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) message_items: Vec<WorkspaceThreadMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) memory_entries: Vec<MemoryEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) command_history: Vec<CommandExecution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) active_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) updated_at: String,
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

pub(crate) fn workspace_config_path(_workspace: &PathBuf) -> PathBuf {
    crate::config_persistence::ajantis_dir().join("workspace_config.json")
}

fn load_workspace_config_from_disk(workspace_root: &PathBuf) -> Result<WorkspaceConfig, String> {
    let path = workspace_config_path(workspace_root);
    if !path.exists() {
        return Ok(WorkspaceConfig::default());
    }
    let content =
        fs::read_to_string(&path).map_err(|e| format!("Failed to read workspace config: {}", e))?;
    let mut config: WorkspaceConfig = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse workspace config: {}", e))?;
    if maybe_migrate_inline_thread_payloads(&mut config)? {
        save_workspace_config_to_disk(workspace_root, &config)?;
    }
    Ok(config)
}

fn save_workspace_config_to_disk(
    workspace_root: &PathBuf,
    config: &WorkspaceConfig,
) -> Result<(), String> {
    let path = workspace_config_path(workspace_root);
    let mut stripped = config.clone();
    for workspace in &mut stripped.workspaces {
        for thread in &mut workspace.threads {
            clear_inline_thread_payload(thread);
        }
    }
    let json = serde_json::to_string_pretty(&stripped)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write workspace config: {}", e))
}

fn clear_inline_thread_payload(thread: &mut WorkspaceThread) {
    thread.messages.clear();
    thread.message_items.clear();
    thread.memory_entries.clear();
    thread.command_history.clear();
}

fn maybe_migrate_inline_thread_payloads(config: &mut WorkspaceConfig) -> Result<bool, String> {
    let mut changed = false;
    for workspace in &mut config.workspaces {
        for thread in &mut workspace.threads {
            let has_inline_payload = !thread.messages.is_empty()
                || !thread.message_items.is_empty()
                || !thread.memory_entries.is_empty()
                || !thread.command_history.is_empty();
            if !has_inline_payload {
                continue;
            }
            let snapshot = ThreadSnapshot {
                message_items: thread.message_items.clone(),
                memory_entries: thread.memory_entries.clone(),
                command_history: thread.command_history.clone(),
                active_run_id: None,
                updated_at: chrono::Utc::now().to_rfc3339(),
            };
            let path = snapshot_path(&workspace.id, &thread.id);
            let json = serde_json::to_string_pretty(&snapshot)
                .map_err(|e| format!("Failed to serialize migrated thread snapshot: {}", e))?;
            fs::write(&path, json)
                .map_err(|e| format!("Failed to persist migrated thread snapshot: {}", e))?;
            clear_inline_thread_payload(thread);
            changed = true;
        }
    }
    Ok(changed)
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
    load_workspace_config_from_disk(&state.workspace_root)
}

#[tauri::command]
pub(crate) async fn save_workspace_config(
    state: tauri::State<'_, Arc<AppState>>,
    config: WorkspaceConfig,
) -> Result<(), String> {
    save_workspace_config_to_disk(&state.workspace_root, &config)
}

#[tauri::command]
pub(crate) async fn delete_thread(
    state: tauri::State<'_, Arc<AppState>>,
    workspace_id: String,
    thread_id: String,
) -> Result<(), String> {
    {
        let runs = state.active_runs.lock().unwrap();
        let has_active_run = runs.values().any(|run| {
            run.workspace_id.as_deref() == Some(workspace_id.as_str())
                && run.thread_id.as_deref() == Some(thread_id.as_str())
        });
        if has_active_run {
            return Err("Stop the active run for this thread before deleting it.".to_string());
        }
    }

    let mut config = load_workspace_config_from_disk(&state.workspace_root)?;
    let workspace = config
        .workspaces
        .iter_mut()
        .find(|workspace| workspace.id == workspace_id)
        .ok_or_else(|| format!("Workspace '{}' not found.", workspace_id))?;

    let original_len = workspace.threads.len();
    workspace.threads.retain(|thread| thread.id != thread_id);
    if workspace.threads.len() == original_len {
        return Err(format!("Thread '{}' not found.", thread_id));
    }

    save_workspace_config_to_disk(&state.workspace_root, &config)?;

    let thread_dir = thread_data_root().join(&workspace_id).join(&thread_id);
    if thread_dir.exists() {
        fs::remove_dir_all(&thread_dir)
            .map_err(|e| format!("Failed to delete thread data: {}", e))?;
    }

    let workspace_threads_dir = thread_data_root().join(&workspace_id);
    if workspace_threads_dir.exists()
        && fs::read_dir(&workspace_threads_dir)
            .map_err(|e| format!("Failed to inspect workspace thread data: {}", e))?
            .next()
            .is_none()
    {
        let _ = fs::remove_dir(&workspace_threads_dir);
    }

    Ok(())
}

#[tauri::command]
pub(crate) async fn load_thread_snapshot(
    workspace_id: String,
    thread_id: String,
) -> Result<ThreadSnapshot, String> {
    let path = snapshot_path(&workspace_id, &thread_id);
    if !path.exists() {
        return Ok(ThreadSnapshot::default());
    }
    let content =
        fs::read_to_string(&path).map_err(|e| format!("Failed to read thread snapshot: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse thread snapshot: {}", e))
}

#[tauri::command]
pub(crate) async fn save_thread_snapshot(
    workspace_id: String,
    thread_id: String,
    snapshot: ThreadSnapshot,
) -> Result<(), String> {
    let path = snapshot_path(&workspace_id, &thread_id);
    let json = serde_json::to_string_pretty(&snapshot)
        .map_err(|e| format!("Failed to serialize thread snapshot: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write thread snapshot: {}", e))
}
