use serde::{Deserialize, Serialize};

use crate::helpers::{default_priority, default_true};

pub(crate) const MIN_SEMANTIC_SIMILARITY_THRESHOLD: f32 = 0.85;
pub(crate) const MAX_SEMANTIC_SIMILARITY_THRESHOLD: f32 = 0.99;
pub(crate) const GROUNDED_AUDIT_BEHAVIOR_ID: &str = "grounded_audit";

fn default_redundancy_enabled() -> bool {
    true
}

fn default_semantic_similarity_threshold() -> f32 {
    0.90
}

fn default_max_redundant_audit_retries() -> u8 {
    1
}

fn default_behavior_triggers_enabled() -> bool {
    true
}

fn default_behavior_similarity_threshold() -> f32 {
    0.90
}

fn default_grounded_audit_keyword_triggers() -> Vec<String> {
    vec![
        "security audit".to_string(),
        "security review".to_string(),
        "repo review".to_string(),
        "code review".to_string(),
        "find potential issues".to_string(),
        "find bugs".to_string(),
        "find vulnerabilities".to_string(),
        "report findings".to_string(),
        "report them here".to_string(),
    ]
}

fn default_grounded_audit_embedding_triggers() -> Vec<String> {
    vec![
        "audit this repository and report concrete issues".to_string(),
        "perform a security review of this codebase".to_string(),
        "inspect the repo for vulnerabilities and risky configurations".to_string(),
    ]
}

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

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct RedundancyDetectionConfig {
    #[serde(default = "default_redundancy_enabled")]
    pub(crate) enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) embedding_model_key: Option<String>,
    #[serde(default = "default_semantic_similarity_threshold")]
    pub(crate) semantic_similarity_threshold: f32,
    #[serde(default = "default_max_redundant_audit_retries")]
    pub(crate) max_redundant_audit_retries: u8,
}

impl Default for RedundancyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            embedding_model_key: None,
            semantic_similarity_threshold: 0.90,
            max_redundant_audit_retries: 1,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BehaviorTriggerConfig {
    pub(crate) behavior_id: String,
    #[serde(default = "default_true")]
    pub(crate) enabled: bool,
    #[serde(default)]
    pub(crate) keyword_triggers: Vec<String>,
    #[serde(default)]
    pub(crate) embedding_trigger_phrases: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) similarity_threshold: Option<f32>,
}

impl BehaviorTriggerConfig {
    pub(crate) fn default_grounded_audit() -> Self {
        Self {
            behavior_id: GROUNDED_AUDIT_BEHAVIOR_ID.to_string(),
            enabled: true,
            keyword_triggers: default_grounded_audit_keyword_triggers(),
            embedding_trigger_phrases: default_grounded_audit_embedding_triggers(),
            similarity_threshold: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BehaviorTriggersConfig {
    #[serde(default = "default_behavior_triggers_enabled")]
    pub(crate) enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) embedding_model_key: Option<String>,
    #[serde(default = "default_behavior_similarity_threshold")]
    pub(crate) default_similarity_threshold: f32,
    #[serde(default = "default_behavior_trigger_entries")]
    pub(crate) behaviors: Vec<BehaviorTriggerConfig>,
}

fn default_behavior_trigger_entries() -> Vec<BehaviorTriggerConfig> {
    vec![BehaviorTriggerConfig::default_grounded_audit()]
}

impl Default for BehaviorTriggersConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            embedding_model_key: None,
            default_similarity_threshold: 0.90,
            behaviors: default_behavior_trigger_entries(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct AgentConfig {
    pub(crate) agents: Vec<Agent>,
    pub(crate) connections: Vec<RoutingRule>,
    #[serde(default)]
    pub(crate) command_policy: CommandPolicy,
    #[serde(default)]
    pub(crate) redundancy_detection: RedundancyDetectionConfig,
    #[serde(default)]
    pub(crate) behavior_triggers: BehaviorTriggersConfig,
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
            redundancy_detection: RedundancyDetectionConfig::default(),
            behavior_triggers: BehaviorTriggersConfig::default(),
        }
    }
}
