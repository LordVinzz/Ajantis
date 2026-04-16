use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::state::AppState;

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

impl MemoryPool {
    pub(crate) fn push(&mut self, agent_id: &str, agent_name: &str, role: &str, content: &str) {
        self.entries.push(MemoryEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            agent_id: agent_id.to_string(),
            agent_name: agent_name.to_string(),
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub(crate) fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        let q = query.to_lowercase();
        self.entries.iter().filter(|e| {
            e.content.to_lowercase().contains(&q)
                || e.agent_name.to_lowercase().contains(&q)
                || e.role.to_lowercase().contains(&q)
        }).collect()
    }
}

#[tauri::command]
pub(crate) async fn get_memory_pool(state: tauri::State<'_, Arc<AppState>>) -> Result<MemoryPool, String> {
    Ok(state.memory_pool.lock().unwrap().clone())
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
pub(crate) async fn clear_memory_pool(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    state.memory_pool.lock().unwrap().entries.clear();
    Ok(())
}
