use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::state::AppState;

pub(crate) const COMMAND_HISTORY_LIMIT: usize = 50;
pub(crate) const COMMAND_HISTORY_CONTEXT_LIMIT: usize = 10;
const COMMAND_RESULT_CHAR_LIMIT: usize = 4_000;
const MEMORY_ENTRY_LIMIT: usize = 400;
const MEMORY_ENTRY_CHAR_LIMIT: usize = 16_000;

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct MemoryEntry {
    pub(crate) timestamp: String,
    pub(crate) agent_id: String,
    pub(crate) agent_name: String,
    pub(crate) role: String,
    pub(crate) content: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct MemoryPool {
    pub(crate) entries: Vec<MemoryEntry>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct CommandExecution {
    pub(crate) timestamp: String,
    pub(crate) tool_name: String,
    pub(crate) command: String,
    pub(crate) normalized_command: String,
    pub(crate) cwd: String,
    pub(crate) success: bool,
    pub(crate) result: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct CommandHistory {
    pub(crate) entries: Vec<CommandExecution>,
}

impl MemoryPool {
    pub(crate) fn push(&mut self, agent_id: &str, agent_name: &str, role: &str, content: &str) {
        self.entries.push(MemoryEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            agent_id: agent_id.to_string(),
            agent_name: agent_name.to_string(),
            role: role.to_string(),
            content: truncate_chars(content, MEMORY_ENTRY_CHAR_LIMIT),
        });
        if self.entries.len() > MEMORY_ENTRY_LIMIT {
            let overflow = self.entries.len() - MEMORY_ENTRY_LIMIT;
            self.entries.drain(0..overflow);
        }
    }

    pub(crate) fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                e.content.to_lowercase().contains(&q)
                    || e.agent_name.to_lowercase().contains(&q)
                    || e.role.to_lowercase().contains(&q)
            })
            .collect()
    }
}

impl CommandHistory {
    pub(crate) fn push(
        &mut self,
        tool_name: &str,
        command: &str,
        normalized_command: &str,
        cwd: &str,
        success: bool,
        result: &str,
    ) {
        self.entries.push(CommandExecution {
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_name: tool_name.to_string(),
            command: command.to_string(),
            normalized_command: normalized_command.to_string(),
            cwd: cwd.to_string(),
            success,
            result: truncate_chars(result, COMMAND_RESULT_CHAR_LIMIT),
        });

        if self.entries.len() > COMMAND_HISTORY_LIMIT {
            let overflow = self.entries.len() - COMMAND_HISTORY_LIMIT;
            self.entries.drain(0..overflow);
        }
    }

    pub(crate) fn find_exact(
        &self,
        tool_name: &str,
        normalized_command: &str,
        cwd: &str,
    ) -> Option<CommandExecution> {
        self.entries
            .iter()
            .rev()
            .find(|entry| {
                entry.tool_name == tool_name
                    && entry.normalized_command == normalized_command
                    && entry.cwd == cwd
            })
            .cloned()
    }

    pub(crate) fn summarize_recent(&self, limit: usize) -> String {
        let start = self.entries.len().saturating_sub(limit);
        self.entries[start..]
            .iter()
            .map(|entry| {
                let preview = truncate_chars(&entry.result.replace('\n', " "), 120);
                format!(
                    "- [{} | {} | cwd: {}] {} => {}",
                    entry.tool_name,
                    if entry.success { "ok" } else { "error" },
                    entry.cwd,
                    entry.command,
                    if preview.is_empty() {
                        "[no output]".to_string()
                    } else {
                        preview
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        text.to_string()
    } else {
        let head: String = text.chars().take(max_chars).collect();
        format!("{}…[truncated, {} chars total]", head, char_count)
    }
}

#[tauri::command]
pub(crate) async fn get_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<MemoryPool, String> {
    Ok(state.memory_pool.lock().unwrap().clone())
}

#[tauri::command]
pub(crate) async fn set_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
    entries: Vec<MemoryEntry>,
) -> Result<(), String> {
    state.memory_pool.lock().unwrap().entries = entries;
    Ok(())
}

#[tauri::command]
pub(crate) async fn get_command_history(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<CommandHistory, String> {
    Ok(state.command_history.lock().unwrap().clone())
}

#[tauri::command]
pub(crate) async fn set_command_history(
    state: tauri::State<'_, Arc<AppState>>,
    entries: Vec<CommandExecution>,
) -> Result<(), String> {
    state.command_history.lock().unwrap().entries = entries;
    Ok(())
}

#[tauri::command]
pub(crate) async fn clear_command_history(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.command_history.lock().unwrap().entries.clear();
    Ok(())
}

#[tauri::command]
pub(crate) async fn search_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
    query: String,
) -> Result<Vec<MemoryEntry>, String> {
    let pool = state.memory_pool.lock().unwrap();
    Ok(pool.search(&query).into_iter().cloned().collect())
}

#[tauri::command]
pub(crate) async fn clear_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.memory_pool.lock().unwrap().entries.clear();
    Ok(())
}
