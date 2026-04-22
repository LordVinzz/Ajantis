use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::state::AppState;

pub const COMMAND_HISTORY_LIMIT: usize = 50;
pub const COMMAND_HISTORY_CONTEXT_LIMIT: usize = 10;
const COMMAND_RESULT_CHAR_LIMIT: usize = 4_000;
const MEMORY_ENTRY_LIMIT: usize = 400;
const MEMORY_ENTRY_CHAR_LIMIT: usize = 16_000;

#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub timestamp: String,
    pub agent_id: String,
    pub agent_name: String,
    pub role: String,
    pub content: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct MemoryPool {
    pub entries: Vec<MemoryEntry>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CommandExecution {
    pub timestamp: String,
    pub tool_name: String,
    pub command: String,
    pub normalized_command: String,
    pub cwd: String,
    pub success: bool,
    pub result: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub classification: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub touched_paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct CommandHistory {
    pub entries: Vec<CommandExecution>,
}

impl MemoryPool {
    pub fn push(&mut self, agent_id: &str, agent_name: &str, role: &str, content: &str) {
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

    pub fn search(&self, query: &str) -> Vec<&MemoryEntry> {
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
    pub fn push(
        &mut self,
        tool_name: &str,
        command: &str,
        normalized_command: &str,
        cwd: &str,
        success: bool,
        result: &str,
    ) {
        let (classification, touched_paths, notes) =
            classify_command_execution(tool_name, command, normalized_command, result);
        self.entries.push(CommandExecution {
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_name: tool_name.to_string(),
            command: command.to_string(),
            normalized_command: normalized_command.to_string(),
            cwd: cwd.to_string(),
            success,
            result: truncate_chars(result, COMMAND_RESULT_CHAR_LIMIT),
            classification,
            touched_paths,
            notes,
        });

        if self.entries.len() > COMMAND_HISTORY_LIMIT {
            let overflow = self.entries.len() - COMMAND_HISTORY_LIMIT;
            self.entries.drain(0..overflow);
        }
    }

    pub fn find_exact(
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

    pub fn summarize_recent(&self, limit: usize) -> String {
        let start = self.entries.len().saturating_sub(limit);
        self.entries[start..]
            .iter()
            .map(|entry| {
                let preview = truncate_chars(&entry.result.replace('\n', " "), 120);
                format!(
                    "- [{} | {} | {} | cwd: {}] {} => {}",
                    entry.tool_name,
                    if entry.success { "ok" } else { "error" },
                    if entry.classification.is_empty() {
                        "unclassified"
                    } else {
                        &entry.classification
                    },
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

fn classify_command_execution(
    tool_name: &str,
    command: &str,
    normalized_command: &str,
    result: &str,
) -> (String, Vec<String>, Vec<String>) {
    let mut notes = Vec::new();
    let mut touched_paths = extract_path_tokens(command);
    touched_paths.extend(extract_path_tokens(result));
    touched_paths.sort();
    touched_paths.dedup();

    let dependency_hit = touched_paths
        .iter()
        .any(|path| matches_dependency_or_generated_path(path))
        || matches_dependency_or_generated_path(command)
        || matches_dependency_or_generated_path(result);

    if dependency_hit {
        notes.push("Touches dependency/build/generated paths.".to_string());
    }

    let broad_scan = is_broad_directory_scan(tool_name, normalized_command);
    if broad_scan {
        notes.push("Broad directory scan heuristic triggered.".to_string());
    }

    let broad_full_read = is_broad_full_file_read(tool_name, normalized_command);
    if broad_full_read {
        notes.push("Broad full-file read heuristic triggered.".to_string());
    }

    let targeted_search = is_targeted_search(tool_name, normalized_command);
    if targeted_search {
        notes.push("Targeted search heuristic triggered.".to_string());
    }

    let targeted_read = is_targeted_read(tool_name, command, normalized_command);
    if targeted_read {
        notes.push("Targeted read heuristic triggered.".to_string());
    }

    let classification = if dependency_hit {
        "dependency_or_generated_scan"
    } else if broad_scan {
        "broad_directory_scan"
    } else if broad_full_read {
        "broad_full_file_read"
    } else if targeted_search {
        "targeted_search"
    } else if targeted_read {
        "targeted_read"
    } else {
        "other"
    };

    (classification.to_string(), touched_paths, notes)
}

fn is_targeted_search(tool_name: &str, normalized_command: &str) -> bool {
    matches!(tool_name, "grep_search" | "glob_search")
        || normalized_command.contains(" rg ")
        || normalized_command.starts_with("rg ")
        || normalized_command.contains(" grep ")
        || normalized_command.starts_with("grep ")
        || normalized_command.contains(" sed -n ")
        || normalized_command.starts_with("sed -n ")
}

fn is_targeted_read(tool_name: &str, command: &str, normalized_command: &str) -> bool {
    tool_name == "read_file"
        || command.contains("scope")
        || normalized_command.contains("sed -n")
        || normalized_command.contains("head ")
        || normalized_command.contains("tail ")
        || normalized_command.contains("nl -ba")
}

fn is_broad_full_file_read(tool_name: &str, normalized_command: &str) -> bool {
    tool_name == "read_file" && !normalized_command.contains("scope")
        || normalized_command.starts_with("cat ")
        || normalized_command.contains(" cat ")
}

fn is_broad_directory_scan(tool_name: &str, normalized_command: &str) -> bool {
    if tool_name == "glob_search" {
        return false;
    }
    normalized_command.starts_with("find .")
        || normalized_command.contains(" find .")
        || normalized_command.starts_with("ls -R")
        || normalized_command.contains(" ls -R")
        || normalized_command.starts_with("tree")
        || normalized_command.contains(" tree ")
}

fn matches_dependency_or_generated_path(text: &str) -> bool {
    [
        "node_modules",
        "dist",
        "target",
        ".git",
        ".cache",
        "build",
        "coverage",
    ]
    .iter()
    .any(|segment| text.contains(segment))
}

fn extract_path_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| {
                    matches!(
                        ch,
                        '"' | '\'' | '`' | ',' | '.' | ':' | ';' | '(' | ')' | '[' | ']'
                    )
                })
                .to_string()
        })
        .filter(|token| token.contains('/') && token.chars().any(|ch| ch.is_ascii_alphanumeric()))
        .collect()
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
pub async fn get_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<MemoryPool, String> {
    Ok(state.memory_pool.lock().unwrap().clone())
}

#[tauri::command]
pub async fn set_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
    entries: Vec<MemoryEntry>,
) -> Result<(), String> {
    state.memory_pool.lock().unwrap().entries = entries;
    Ok(())
}

#[tauri::command]
pub async fn get_command_history(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<CommandHistory, String> {
    Ok(state.command_history.lock().unwrap().clone())
}

#[tauri::command]
pub async fn set_command_history(
    state: tauri::State<'_, Arc<AppState>>,
    entries: Vec<CommandExecution>,
) -> Result<(), String> {
    state.command_history.lock().unwrap().entries = entries;
    Ok(())
}

#[tauri::command]
pub async fn clear_command_history(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.command_history.lock().unwrap().entries.clear();
    Ok(())
}

#[tauri::command]
pub async fn search_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
    query: String,
) -> Result<Vec<MemoryEntry>, String> {
    let pool = state.memory_pool.lock().unwrap();
    Ok(pool.search(&query).into_iter().cloned().collect())
}

#[tauri::command]
pub async fn clear_memory_pool(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.memory_pool.lock().unwrap().entries.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::classify_command_execution;

    #[test]
    fn classifies_broad_and_targeted_commands() {
        let (class_a, _, _) =
            classify_command_execution("bash", "cat src/App.tsx", "cat src/App.tsx", "content");
        assert_eq!(class_a, "broad_full_file_read");

        let (class_b, _, _) = classify_command_execution(
            "grep_search",
            "grep_search src pattern",
            "grep_search src pattern",
            "src/App.tsx:12:match",
        );
        assert_eq!(class_b, "targeted_search");

        let (class_c, _, _) = classify_command_execution(
            "bash",
            "find . -type f",
            "find . -type f",
            "src/App.tsx\nnode_modules/pkg",
        );
        assert_eq!(class_c, "dependency_or_generated_scan");
    }
}
