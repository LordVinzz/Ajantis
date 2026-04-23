use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent_config::AgentConfig;
use crate::event_sink::SharedEventSink;
use crate::mcp::McpState;
use crate::memory::{CommandHistory, MemoryPool};
use crate::runs::ActiveRuns;
use crate::workspace::ThreadSnapshot;

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
    #[allow(dead_code)]
    pub(crate) todo_list: Arc<Mutex<Vec<Value>>>,
    pub(crate) agent_config: Arc<Mutex<AgentConfig>>,
    pub(crate) memory_pool: Arc<Mutex<MemoryPool>>,
    pub(crate) command_history: Arc<Mutex<CommandHistory>>,
    pub(crate) glob_cache:
        Arc<Mutex<std::collections::HashMap<String, std::collections::HashSet<String>>>>,
    pub(crate) mcp_state: Arc<McpState>,
    /// Shared with McpState so MCP tool handlers can emit stream events to the frontend or TUI.
    pub(crate) event_channel: Arc<Mutex<Option<SharedEventSink>>>,
    pub(crate) active_runs: ActiveRuns,
    /// Run-scoped key-value scratchpad — cleared at run start, never written to disk.
    pub(crate) scratchpad: Arc<Mutex<std::collections::HashMap<String, String>>>,
    pub(crate) pending_thread_snapshots:
        Arc<Mutex<std::collections::HashMap<String, ThreadSnapshot>>>,
    pub(crate) pending_thread_snapshot_versions: Arc<Mutex<std::collections::HashMap<String, u64>>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct McpTool {
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(rename = "inputSchema")]
    pub(crate) input_schema: Value,
}
