use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::ipc::Channel;

use crate::agent_config::AgentConfig;
use crate::chat::StreamEvent;
use crate::mcp::McpState;
use crate::memory::{CommandHistory, MemoryPool};
use crate::runs::ActiveRuns;

#[derive(Default)]
pub(crate) struct BehaviorTriggerCache {
    pub(crate) embeddings: std::collections::HashMap<String, Vec<f32>>,
}

pub(crate) struct AppState {
    pub(crate) current_model: Mutex<String>,
    pub(crate) last_response_id: Mutex<Option<String>>,
    pub(crate) mcp_port: u16,
    pub(crate) workspace_root: PathBuf,
    /// Currently selected workspace path — updated when the user selects a workspace.
    /// Shared with McpState so all MCP tool file/command ops are sandboxed to it.
    pub(crate) active_workspace: Arc<Mutex<PathBuf>>,
    pub(crate) mcp_tools: Vec<McpTool>,
    pub(crate) todo_list: Arc<Mutex<Vec<Value>>>,
    pub(crate) agent_config: Arc<Mutex<AgentConfig>>,
    pub(crate) memory_pool: Arc<Mutex<MemoryPool>>,
    pub(crate) command_history: Arc<Mutex<CommandHistory>>,
    pub(crate) glob_cache:
        Arc<Mutex<std::collections::HashMap<String, std::collections::HashSet<String>>>>,
    pub(crate) mcp_state: Arc<McpState>,
    /// Shared with McpState so MCP tool handlers can emit stream events to the frontend.
    pub(crate) event_channel: Arc<Mutex<Option<Channel<StreamEvent>>>>,
    pub(crate) active_runs: ActiveRuns,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct McpTool {
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(rename = "inputSchema")]
    pub(crate) input_schema: Value,
}
