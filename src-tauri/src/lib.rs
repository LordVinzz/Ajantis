mod helpers;
mod agent_config;
mod workspace;
mod memory;
mod state;
mod models;
mod chat;
mod config_persistence;
mod routing;
mod mcp;

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::ipc::Channel;

use crate::agent_config::AgentConfig;
use crate::chat::{send_message, StreamEvent};
use crate::config_persistence::{config_path, load_agent_config, save_agent_config};
use crate::mcp::{load_tools, start_mcp_server, McpState};
use crate::memory::{clear_memory_pool, get_memory_pool, search_memory_pool, MemoryPool};
use crate::models::{download_model, fetch_loaded_models, fetch_models, load_model, set_model, unload_model};
use crate::routing::route_message;
use crate::state::AppState;
use crate::workspace::{load_workspace_config, pick_folder, save_workspace_config, set_active_workspace};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // tools.json lives at the project root; when running from src-tauri/ look one level up
    let tools_path = {
        let in_cwd = workspace_root.join("tools.json");
        let in_parent = workspace_root.parent().map(|p| p.join("tools.json"));
        match in_parent.filter(|p| p.exists()) {
            Some(p) if !in_cwd.exists() => p,
            _ => in_cwd,
        }
    };
    let mcp_tools = load_tools(&tools_path);
    let mcp_port: u16 = 4785;

    let agent_config: AgentConfig = {
        let path = config_path(&workspace_root);
        if path.exists() {
            fs::read_to_string(&path).ok()
                .and_then(|c| serde_json::from_str(&c).ok())
                .unwrap_or_default()
        } else {
            AgentConfig::default()
        }
    };

    let agent_config_arc    = Arc::new(Mutex::new(agent_config));
    let memory_pool_arc     = Arc::new(Mutex::new(MemoryPool::default()));
    let todo_list_arc       = Arc::new(Mutex::new(vec![]));
    let event_channel_arc: Arc<Mutex<Option<Channel<StreamEvent>>>> =
        Arc::new(Mutex::new(None));
    // Starts as workspace_root; updated by set_active_workspace when the user picks a workspace.
    let active_workspace_arc = Arc::new(Mutex::new(workspace_root.clone()));

    let app_state = Arc::new(AppState {
        current_model: Mutex::new(
            std::env::var("LM_STUDIO_MODEL").unwrap_or_else(|_| "lmstudio/lmstudio-1B".to_string()),
        ),
        last_response_id: Mutex::new(None),
        mcp_port,
        workspace_root: workspace_root.clone(),
        active_workspace: active_workspace_arc.clone(),
        mcp_tools: mcp_tools.clone(),
        todo_list: todo_list_arc.clone(),
        agent_config: agent_config_arc.clone(),
        memory_pool: memory_pool_arc.clone(),
        event_channel: event_channel_arc.clone(),
    });

    let mcp_state = McpState {
        tools: mcp_tools,
        workspace_root,
        active_workspace: active_workspace_arc,
        todo_list: todo_list_arc,
        memory_pool: memory_pool_arc,
        agent_config: agent_config_arc,
        mcp_port,
        event_channel: event_channel_arc,
        tasks: Arc::new(Mutex::new(std::collections::HashMap::new())),
    };

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
            let mcp_state_clone = mcp_state.clone();
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
            search_memory_pool,
            clear_memory_pool,
            route_message,
            pick_folder,
            load_workspace_config,
            save_workspace_config,
            set_active_workspace,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
