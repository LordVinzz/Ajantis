mod agent_config;
mod chat;
mod config_persistence;
mod helpers;
mod mcp;
mod memory;
mod models;
mod routing;
mod runs;
mod state;
mod workspace;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::ipc::Channel;

use crate::agent_config::AgentConfig;
use crate::chat::{send_message, StreamEvent};
use crate::config_persistence::{
    load_agent_config, load_agent_config_from_disk, save_agent_config,
};
use crate::mcp::{load_tools_embedded, start_mcp_server, McpState};
use crate::memory::{
    clear_command_history, clear_memory_pool, get_command_history, get_memory_pool,
    search_memory_pool, set_command_history, set_memory_pool, CommandHistory, MemoryPool,
};
use crate::models::{
    download_model, fetch_loaded_models, fetch_models, load_model, set_model, unload_model,
};
use crate::routing::{cancel_route_run, continue_route_run, route_message};
use crate::state::{AppState, BehaviorTriggerCache};
use crate::workspace::{
    delete_thread, load_thread_snapshot, load_workspace_config, pick_folder,
    save_thread_snapshot, save_workspace_config, set_active_workspace,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // tools.json is embedded at compile time — no need to ship it alongside the binary.
    let mcp_tools = load_tools_embedded();
    let mcp_port: u16 = 4785;

    let agent_config: AgentConfig = load_agent_config_from_disk(&workspace_root);

    let agent_config_arc = Arc::new(Mutex::new(agent_config));
    let memory_pool_arc = Arc::new(Mutex::new(MemoryPool::default()));
    let command_history_arc = Arc::new(Mutex::new(CommandHistory::default()));
    let todo_list_arc = Arc::new(Mutex::new(vec![]));
    let event_channel_arc: Arc<Mutex<Option<Channel<StreamEvent>>>> = Arc::new(Mutex::new(None));
    // Starts as workspace_root; updated by set_active_workspace when the user picks a workspace.
    let active_workspace_arc = Arc::new(Mutex::new(workspace_root.clone()));
    let glob_cache_arc = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let behavior_trigger_cache_arc = Arc::new(Mutex::new(BehaviorTriggerCache::default()));
    let active_behavior_contexts_arc = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let active_runs_arc = Arc::new(Mutex::new(std::collections::HashMap::new()));

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

    let app_state = Arc::new(AppState {
        current_model: Mutex::new(
            std::env::var("LM_STUDIO_MODEL").unwrap_or_else(|_| "lmstudio/lmstudio-1B".to_string()),
        ),
        last_response_id: Mutex::new(None),
        mcp_port,
        workspace_root: workspace_root.clone(),
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
    });

    tauri::Builder::default()
        .manage(app_state)
        .setup(move |app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            let mcp_state_clone = mcp_state.as_ref().clone();
            tauri::async_runtime::spawn(async move {
                start_mcp_server(mcp_port, mcp_state_clone).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            send_message,
            fetch_models,
            fetch_loaded_models,
            set_model,
            load_model,
            unload_model,
            download_model,
            save_agent_config,
            load_agent_config,
            get_memory_pool,
            set_memory_pool,
            get_command_history,
            set_command_history,
            search_memory_pool,
            clear_memory_pool,
            clear_command_history,
            route_message,
            cancel_route_run,
            continue_route_run,
            pick_folder,
            load_workspace_config,
            save_workspace_config,
            delete_thread,
            load_thread_snapshot,
            save_thread_snapshot,
            set_active_workspace,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
