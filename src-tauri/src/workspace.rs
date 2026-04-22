use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::memory::{CommandExecution, MemoryEntry};
use crate::runs::{snapshot_path, thread_data_root, RunDossier};
use crate::state::AppState;

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceToolCall {
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub args: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub result: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    #[serde(default)]
    pub semantic: bool,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceMessageSignal {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceThreadMessage {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub agent_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,
    #[serde(default)]
    pub is_sent: bool,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub for_user: bool,
    #[serde(default)]
    pub internal: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WorkspaceToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<WorkspaceMessageSignal>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WorkspaceThread {
    pub id: String,
    pub name: String,
    /// Legacy persisted HTML snapshot of the thread conversation.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub messages: String,
    /// Structured thread messages used to reconstruct the UI.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub message_items: Vec<WorkspaceThreadMessage>,
    /// Persisted backend conversation state for the thread.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_entries: Vec<MemoryEntry>,
    /// Persisted thread-scoped command execution history.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command_history: Vec<CommandExecution>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct ThreadSnapshot {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub message_items: Vec<WorkspaceThreadMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_entries: Vec<MemoryEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command_history: Vec<CommandExecution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_dossier: Option<RunDossier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub updated_at: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub threads: Vec<WorkspaceThread>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceConfig {
    pub workspaces: Vec<Workspace>,
}

pub fn workspace_config_path(_workspace: &PathBuf) -> PathBuf {
    crate::config_persistence::ajantis_dir().join("workspace_config.json")
}

pub fn load_workspace_config_from_disk(
    workspace_root: &PathBuf,
) -> Result<WorkspaceConfig, String> {
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

pub fn save_workspace_config_to_disk(
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

fn write_thread_snapshot_atomic(
    workspace_id: &str,
    thread_id: &str,
    snapshot: &ThreadSnapshot,
) -> Result<(), String> {
    let path = snapshot_path(workspace_id, thread_id);
    let json = serde_json::to_string_pretty(snapshot)
        .map_err(|e| format!("Failed to serialize thread snapshot: {}", e))?;
    let tmp_path = path.with_extension(format!(
        "json.tmp-{}",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs::write(&tmp_path, json)
        .map_err(|e| format!("Failed to write temp thread snapshot: {}", e))?;
    fs::rename(&tmp_path, &path)
        .map_err(|e| format!("Failed to replace thread snapshot atomically: {}", e))
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
                run_dossier: None,
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
pub fn set_active_workspace_path(
    state: &Arc<AppState>,
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
pub async fn set_active_workspace(
    state: tauri::State<'_, Arc<AppState>>,
    path: Option<String>,
) -> Result<(), String> {
    set_active_workspace_path(&state.inner().clone(), path)
}

pub fn pick_folder_blocking() -> Result<Option<String>, String> {
    let handle = rfd::FileDialog::new()
        .set_title("Choose a workspace folder")
        .pick_folder();
    Ok(handle.map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
pub async fn pick_folder() -> Result<Option<String>, String> {
    pick_folder_blocking()
}

#[tauri::command]
pub async fn load_workspace_config(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<WorkspaceConfig, String> {
    load_workspace_config_from_disk(&state.workspace_root)
}

#[tauri::command]
pub async fn save_workspace_config(
    state: tauri::State<'_, Arc<AppState>>,
    config: WorkspaceConfig,
) -> Result<(), String> {
    save_workspace_config_to_disk(&state.workspace_root, &config)
}

#[tauri::command]
pub async fn delete_thread(
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
pub async fn load_thread_snapshot(
    workspace_id: String,
    thread_id: String,
) -> Result<ThreadSnapshot, String> {
    load_thread_snapshot_from_disk(&workspace_id, &thread_id)
}

#[tauri::command]
pub async fn save_thread_snapshot(
    workspace_id: String,
    thread_id: String,
    snapshot: ThreadSnapshot,
) -> Result<(), String> {
    save_thread_snapshot_to_disk(&workspace_id, &thread_id, snapshot)
}

pub fn queue_thread_snapshot_save_for_state(
    state: &Arc<AppState>,
    workspace_id: String,
    thread_id: String,
    snapshot: ThreadSnapshot,
) -> Result<(), String> {
    let key = format!("{}/{}", workspace_id, thread_id);
    let version = {
        state
            .pending_thread_snapshots
            .lock()
            .unwrap()
            .insert(key.clone(), snapshot);
        let mut versions = state.pending_thread_snapshot_versions.lock().unwrap();
        let entry = versions.entry(key.clone()).or_insert(0);
        *entry += 1;
        *entry
    };

    let pending = state.pending_thread_snapshots.clone();
    let versions = state.pending_thread_snapshot_versions.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let should_flush = versions
            .lock()
            .unwrap()
            .get(&key)
            .copied()
            .map(|current| current == version)
            .unwrap_or(false);
        if !should_flush {
            return;
        }
        let latest = pending.lock().unwrap().remove(&key);
        if let Some(snapshot) = latest {
            let _ = write_thread_snapshot_atomic(&workspace_id, &thread_id, &snapshot);
        }
    });
    Ok(())
}

#[tauri::command]
pub async fn queue_thread_snapshot_save(
    state: tauri::State<'_, Arc<AppState>>,
    workspace_id: String,
    thread_id: String,
    snapshot: ThreadSnapshot,
) -> Result<(), String> {
    queue_thread_snapshot_save_for_state(&state.inner().clone(), workspace_id, thread_id, snapshot)
}
pub fn load_thread_snapshot_from_disk(
    workspace_id: &str,
    thread_id: &str,
) -> Result<ThreadSnapshot, String> {
    let path = snapshot_path(workspace_id, thread_id);
    if !path.exists() {
        return Ok(ThreadSnapshot::default());
    }
    let content =
        fs::read_to_string(&path).map_err(|e| format!("Failed to read thread snapshot: {}", e))?;
    match serde_json::from_str(&content) {
        Ok(snapshot) => Ok(snapshot),
        Err(error) => Ok(ThreadSnapshot {
            message_items: vec![WorkspaceThreadMessage {
                kind: "text".to_string(),
                agent_id: None,
                agent_name: "System".to_string(),
                content: format!(
                    "Thread snapshot could not be fully parsed and was recovered as empty state. Parse error: {}",
                    error
                ),
                is_sent: false,
                is_error: true,
                for_user: true,
                internal: false,
                tools: vec![],
                signal: None,
            }],
            memory_entries: vec![],
            command_history: vec![],
            run_dossier: None,
            active_run_id: None,
            updated_at: chrono::Utc::now().to_rfc3339(),
        }),
    }
}

pub fn save_thread_snapshot_to_disk(
    workspace_id: &str,
    thread_id: &str,
    snapshot: ThreadSnapshot,
) -> Result<(), String> {
    write_thread_snapshot_atomic(workspace_id, thread_id, &snapshot)
}
