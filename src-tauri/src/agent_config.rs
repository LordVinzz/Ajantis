use serde::{Deserialize, Serialize};

use crate::helpers::{default_priority, default_true};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct AgentLoadConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) eval_batch_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) flash_attention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) num_experts: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) offload_kv_cache_to_gpu: Option<bool>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Agent {
    pub(crate) id: String,
    pub(crate) name: String,
    #[serde(rename = "type")]
    pub(crate) agent_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) load_config: Option<AgentLoadConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) allowed_tools: Option<Vec<String>>,
    #[serde(default = "default_true")]
    pub(crate) armed: bool,
    /// Enables MCP tool-call loop path (spawn_agent, send_message, etc.)
    #[serde(default)]
    pub(crate) is_manager: bool,
    /// Runtime pause flag — not persisted to config.
    #[serde(default, skip_serializing)]
    pub(crate) paused: bool,
}

/// Replaces Connection. Backward-compatible via #[serde(default)].
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct RoutingRule {
    pub(crate) from: String,
    pub(crate) to: String,
    /// Execution order among siblings: lower = called first. Default 128.
    #[serde(default = "default_priority")]
    pub(crate) priority: u8,
    /// Optional substring that must appear in the outgoing message for this route to fire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) condition: Option<String>,
    #[serde(default = "default_true")]
    pub(crate) enabled: bool,
}

/// Sandbox policy for command-executing tools (bash, REPL, PowerShell, TaskCreate).
///
/// Rules (in priority order):
///   1. If `denylist` is non-empty and the command matches any entry → denied.
///   2. If `allowlist` is non-empty and the command matches NO entry → denied.
///   3. Otherwise → allowed.
///
/// Matching is prefix-based (the command starts with the entry string).
/// An empty `allowlist` means "allow all" (unless the denylist says otherwise).
#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct CommandPolicy {
    /// Commands (or prefixes) that are always blocked.
    #[serde(default)]
    pub(crate) denylist: Vec<String>,
    /// If non-empty, only commands matching one of these prefixes are allowed.
    #[serde(default)]
    pub(crate) allowlist: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct AgentConfig {
    pub(crate) agents: Vec<Agent>,
    pub(crate) connections: Vec<RoutingRule>,
    #[serde(default)]
    pub(crate) command_policy: CommandPolicy,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            agents: vec![Agent {
                id: "user".to_string(),
                name: "User".to_string(),
                agent_type: "user".to_string(),
                model_key: None,
                model_type: None,
                role: None,
                load_config: None,
                mode: None,
                allowed_tools: None,
                armed: true,
                is_manager: false,
                paused: false,
            }],
            connections: vec![],
            command_policy: CommandPolicy::default(),
        }
    }
}
