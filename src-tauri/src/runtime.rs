use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub use crate::agent_config::{
    Agent, AgentConfig, AgentLoadConfig, BehaviorTriggerConfig, FinalizerConfig, RoutingRule,
};
pub use crate::chat::StreamEvent;
use crate::config_persistence::{
    load_agent_config_for_state, load_agent_config_from_disk, save_agent_config_for_state,
};
pub use crate::event_sink::{callback_event_sink, SharedEventSink};
use crate::mcp::{load_tools_embedded, start_mcp_server, McpState};
pub use crate::memory::{CommandExecution, CommandHistory, MemoryEntry, MemoryPool};
pub use crate::models::{LoadConfig, LoadedModelInfo, ModelInfo};
use crate::models::{fetch_loaded_models, fetch_models, load_model, unload_model};
use crate::routing::{
    cancel_route_run_for_state, continue_route_run_for_state, route_message_for_state,
};
pub use crate::runs::RunDossier;
use crate::state::{AppState, BehaviorTriggerCache};
pub use crate::workspace::{
    ThreadSnapshot, Workspace, WorkspaceConfig, WorkspaceThread, WorkspaceThreadMessage,
    WorkspaceToolCall,
};
use crate::workspace::{
    load_thread_snapshot_from_disk, load_workspace_config_from_disk, pick_folder_blocking,
    queue_thread_snapshot_save_for_state, save_thread_snapshot_to_disk,
    save_workspace_config_to_disk, set_active_workspace_path,
};

#[derive(Clone)]
pub struct RuntimeHandle {
    state: Arc<AppState>,
}

impl RuntimeHandle {
    pub fn new(workspace_root: PathBuf) -> Self {
        let state = build_runtime_state(workspace_root);
        let mcp_port = state.mcp_port;
        let mcp_state = state.mcp_state.clone();
        tokio::spawn(async move {
            start_mcp_server(mcp_port, mcp_state.as_ref().clone()).await;
        });
        Self { state }
    }

    pub fn workspace_root(&self) -> PathBuf {
        self.state.workspace_root.clone()
    }

    pub fn load_agent_config(&self) -> Result<AgentConfig, String> {
        Ok(load_agent_config_for_state(&self.state))
    }

    pub fn save_agent_config(&self, config: AgentConfig) -> Result<(), String> {
        save_agent_config_for_state(&self.state, config)
    }

    pub fn load_workspace_config(&self) -> Result<WorkspaceConfig, String> {
        load_workspace_config_from_disk(&self.state.workspace_root)
    }

    pub fn save_workspace_config(&self, config: WorkspaceConfig) -> Result<(), String> {
        save_workspace_config_to_disk(&self.state.workspace_root, &config)
    }

    pub fn load_thread_snapshot(
        &self,
        workspace_id: &str,
        thread_id: &str,
    ) -> Result<ThreadSnapshot, String> {
        load_thread_snapshot_from_disk(workspace_id, thread_id)
    }

    pub fn save_thread_snapshot(
        &self,
        workspace_id: &str,
        thread_id: &str,
        snapshot: ThreadSnapshot,
    ) -> Result<(), String> {
        save_thread_snapshot_to_disk(workspace_id, thread_id, snapshot)
    }

    pub fn queue_thread_snapshot_save(
        &self,
        workspace_id: String,
        thread_id: String,
        snapshot: ThreadSnapshot,
    ) -> Result<(), String> {
        queue_thread_snapshot_save_for_state(&self.state, workspace_id, thread_id, snapshot)
    }

    pub fn set_active_workspace(&self, path: Option<String>) -> Result<(), String> {
        set_active_workspace_path(&self.state, path)
    }

    pub fn get_memory_pool(&self) -> MemoryPool {
        self.state.memory_pool.lock().unwrap().clone()
    }

    pub fn set_memory_pool(&self, entries: Vec<MemoryEntry>) {
        self.state.memory_pool.lock().unwrap().entries = entries;
    }

    pub fn get_command_history(&self) -> CommandHistory {
        self.state.command_history.lock().unwrap().clone()
    }

    pub fn set_command_history(&self, entries: Vec<CommandExecution>) {
        self.state.command_history.lock().unwrap().entries = entries;
    }

    pub async fn fetch_models(&self) -> Result<Vec<ModelInfo>, String> {
        fetch_models().await
    }

    pub async fn fetch_loaded_models(&self) -> Result<Vec<LoadedModelInfo>, String> {
        fetch_loaded_models().await
    }

    pub async fn load_model(&self, config: LoadConfig) -> Result<(), String> {
        load_model(config).await
    }

    pub async fn unload_model(&self, instance_id: String) -> Result<(), String> {
        unload_model(instance_id).await
    }

    pub async fn route_message(
        &self,
        from_agent_id: String,
        message: String,
        workspace_id: Option<String>,
        thread_id: Option<String>,
        on_event: SharedEventSink,
    ) -> Result<String, String> {
        route_message_for_state(
            self.state.clone(),
            from_agent_id,
            message,
            workspace_id,
            thread_id,
            on_event,
        )
        .await
    }

    pub async fn continue_route_run(&self, run_id: String) -> Result<(), String> {
        continue_route_run_for_state(self.state.clone(), run_id).await
    }

    pub async fn cancel_route_run(&self, run_id: String) -> Result<(), String> {
        cancel_route_run_for_state(self.state.clone(), run_id).await
    }

    pub fn pick_folder_blocking(&self) -> Result<Option<String>, String> {
        pick_folder_blocking()
    }
}

fn build_runtime_state(workspace_root: PathBuf) -> Arc<AppState> {
    let mcp_tools = load_tools_embedded();
    let mcp_port: u16 = 4785;

    let agent_config: AgentConfig = load_agent_config_from_disk(&workspace_root);

    let agent_config_arc = Arc::new(Mutex::new(agent_config));
    let memory_pool_arc = Arc::new(Mutex::new(MemoryPool::default()));
    let command_history_arc = Arc::new(Mutex::new(CommandHistory::default()));
    let todo_list_arc = Arc::new(Mutex::new(vec![]));
    let event_channel_arc = Arc::new(Mutex::new(None));
    let active_workspace_arc = Arc::new(Mutex::new(workspace_root.clone()));
    let glob_cache_arc = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let behavior_trigger_cache_arc = Arc::new(Mutex::new(BehaviorTriggerCache::default()));
    let active_behavior_contexts_arc = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let active_runs_arc = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let pending_thread_snapshots_arc = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let pending_thread_snapshot_versions_arc =
        Arc::new(Mutex::new(std::collections::HashMap::new()));

    let mcp_state = Arc::new(McpState {
        tools: mcp_tools,
        workspace_root: workspace_root.clone(),
        active_workspace: active_workspace_arc,
        todo_list: todo_list_arc,
        memory_pool: memory_pool_arc,
        command_history: command_history_arc,
        agent_config: agent_config_arc,
        mcp_port,
        event_channel: event_channel_arc,
        tasks: Arc::new(Mutex::new(std::collections::HashMap::new())),
        read_cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        glob_cache: glob_cache_arc,
        behavior_trigger_cache: behavior_trigger_cache_arc,
        active_behavior_contexts: active_behavior_contexts_arc,
        active_runs: active_runs_arc.clone(),
    });

    Arc::new(AppState {
        current_model: Mutex::new(
            std::env::var("LM_STUDIO_MODEL").unwrap_or_else(|_| "lmstudio/lmstudio-1B".to_string()),
        ),
        last_response_id: Mutex::new(None),
        mcp_port,
        workspace_root,
        active_workspace: mcp_state.active_workspace.clone(),
        mcp_tools: mcp_state.tools.clone(),
        todo_list: mcp_state.todo_list.clone(),
        agent_config: mcp_state.agent_config.clone(),
        memory_pool: mcp_state.memory_pool.clone(),
        command_history: mcp_state.command_history.clone(),
        glob_cache: mcp_state.glob_cache.clone(),
        mcp_state: mcp_state.clone(),
        event_channel: mcp_state.event_channel.clone(),
        active_runs: active_runs_arc,
        pending_thread_snapshots: pending_thread_snapshots_arc,
        pending_thread_snapshot_versions: pending_thread_snapshot_versions_arc,
    })
}
