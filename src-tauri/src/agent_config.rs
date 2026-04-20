use std::collections::{HashMap, HashSet};

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

fn default_run_budgets_enabled() -> bool {
    true
}

fn default_run_budget_llm_calls() -> u32 {
    20
}

fn default_run_budget_tool_calls() -> u32 {
    60
}

fn default_run_budget_spawned_agents() -> u32 {
    8
}

fn default_run_budget_streamed_tokens() -> u64 {
    75_000
}

fn default_run_budget_wall_clock_seconds() -> u64 {
    600
}

fn default_run_budget_behaviors() -> Vec<String> {
    vec![] // empty = apply to all runs unconditionally
}

fn default_on_limit() -> String {
    "summarize".to_string()
}

fn default_budget_summarization_enabled() -> bool {
    true
}

fn default_budget_summarization_prompt() -> String {
    "The run has hit its configured soft budget limit. Using only the evidence already present in this conversation, produce a final concise summary of:\n1. What was accomplished and key findings so far.\n2. What remains incomplete or uncertain.\n3. The next concrete step if the task were to continue.\n\nDo not call any tools. Do not apologize. Be direct and factual.".to_string()
}

fn default_runtime_note_enabled() -> bool {
    false
}

fn default_coverage_manifest_enabled() -> bool {
    false
}

fn default_coverage_manifest_require_resolution() -> bool {
    false
}

fn default_response_rewrite_enabled() -> bool {
    false
}

fn default_delegation_validation_enabled() -> bool {
    false
}

fn default_force_synthesis_require_coverage_gap_signal() -> bool {
    false
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

fn default_grounded_audit_system_prompt_injection() -> Option<String> {
    Some(
        "Grounded audit behavior is active for this task.\n- Treat explicitly requested files, directories, file classes, services, and subsystems as a coverage checklist.\n- Use the sections `Confirmed findings`, `Hypotheses / lower-confidence risks`, and `Coverage gaps` when reporting audit-style results.\n- A confirmed finding must be grounded in code, configuration, policy, or command evidence actually inspected.\n- Do not label anything `High` or `Critical` without direct supporting evidence already in context.\n- A hypothesis must cite a concrete file, config, endpoint, policy, or observed behavior, or explicitly state what evidence is still missing to confirm it.\n- If evidence is insufficient, say `insufficient evidence` instead of inventing a finding."
            .to_string(),
    )
}

fn default_grounded_audit_runtime_note_template() -> Option<String> {
    Some(
        "Audit topics already covered in this turn:\n{covered_topics_bullets}\n\nAvoid re-delegating or re-checking the same file/function/theme unless you are opening a genuinely new code/config area. If current evidence is already sufficient, synthesize now.\n\nRequired audit coverage still unresolved:\n{unresolved_scopes_bullets}".to_string(),
    )
}

fn default_grounded_audit_gap_section_label() -> Option<String> {
    Some("Coverage gaps".to_string())
}

fn default_grounded_audit_unresolved_prompt() -> Option<String> {
    Some(
        "Before concluding, you must either inspect these required audit scopes or name them explicitly under `{gap_section_label}`: {unresolved_scopes_csv}. Do not omit them.".to_string(),
    )
}

fn default_grounded_audit_required_sections() -> Vec<String> {
    vec![
        "Confirmed findings".to_string(),
        "Hypotheses / lower-confidence risks".to_string(),
        "Coverage gaps".to_string(),
    ]
}

fn default_grounded_audit_section_rules() -> Vec<SectionRule> {
    vec![
        SectionRule {
            section_name: "Confirmed findings".to_string(),
            require_file_reference: false,
            disallow_template_phrases: vec![
                "mitigation already present".to_string(),
                "no obvious issue".to_string(),
                "architecture overview".to_string(),
                "project uses".to_string(),
                "uses framework".to_string(),
                "contains modules".to_string(),
                "best practice".to_string(),
                "style issue".to_string(),
                "readability".to_string(),
                "mitigation".to_string(),
            ],
            rewrite_loop_prompt: None,
        },
        SectionRule {
            section_name: "Hypotheses / lower-confidence risks".to_string(),
            require_file_reference: false,
            disallow_template_phrases: vec![
                "presence of ".to_string(),
                "possible risk".to_string(),
                "potential risk".to_string(),
                "could lead to".to_string(),
                "could allow".to_string(),
                "could be vulnerable if".to_string(),
                "may allow".to_string(),
                "may be vulnerable".to_string(),
                "if later".to_string(),
                "if exposed".to_string(),
                "if combined".to_string(),
                "dependency".to_string(),
                "package".to_string(),
                "library".to_string(),
                "framework".to_string(),
                "sdk".to_string(),
                "module".to_string(),
                "orm".to_string(),
            ],
            rewrite_loop_prompt: Some(
                "Your `{section_name}` section is still too generic or template-like. Keep only hypotheses tied to a concrete file/config or observed behavior already in context, and state the missing proof needed to confirm them. Remove generic dependency-presence or stack-template risks.".to_string(),
            ),
        },
    ]
}

fn default_grounded_audit_response_rewrite_prompt() -> Option<String> {
    Some(
        "Rewrite the audit answer below using only the evidence already present in this conversation.\n\nRules:\n- Use exactly these sections: {required_sections_csv}.\n- A confirmed finding must be directly supported by code or config already shown.\n- Keep only actual issues in `Confirmed findings`; move neutral observations, mitigations, architecture facts, and absence-of-risk statements out of that section.\n- Downgrade any unsupported concern into hypotheses.\n- Every hypothesis must cite a concrete file/config or observed behavior already in context, or explicitly state what evidence is still missing to confirm it.\n- Remove dependency-presence, stack-template, or generic speculative risks that are not tied to observed behavior.\n- Do not invent new evidence.\n- Do not label anything `High` or `Critical` without direct code/config support.\n\nCandidate answer:\n{candidate_response}".to_string(),
    )
}

fn default_grounded_audit_code_signals() -> Vec<String> {
    vec![
        "fn ".to_string(),
        "function ".to_string(),
        "def ".to_string(),
        "func ".to_string(),
        "class ".to_string(),
        "interface ".to_string(),
        "struct ".to_string(),
        "enum ".to_string(),
        "impl ".to_string(),
        "async ".to_string(),
        "public ".to_string(),
        "private ".to_string(),
        "protected ".to_string(),
        "export ".to_string(),
        "import ".to_string(),
        "#include ".to_string(),
        "match ".to_string(),
    ]
}

fn default_grounded_audit_config_signals() -> Vec<String> {
    vec![
        "\"$schema\"".to_string(),
        "\"permissions\"".to_string(),
        "\"scripts\"".to_string(),
        "\"devDependencies\"".to_string(),
        "[package]".to_string(),
        "[dependencies]".to_string(),
        "apiVersion:".to_string(),
        "services:".to_string(),
        "provider ".to_string(),
        "[tool.".to_string(),
        "dockerfile".to_string(),
        "compose:".to_string(),
        "\"csp\"".to_string(),
    ]
}

fn default_grounded_audit_command_signals() -> Vec<String> {
    vec![
        "src/".to_string(),
        "app/".to_string(),
        "lib/".to_string(),
        "cmd/".to_string(),
        "internal/".to_string(),
        "pkg/".to_string(),
        "package.json".to_string(),
        "requirements.txt".to_string(),
        "pyproject.toml".to_string(),
        "cargo.toml".to_string(),
        "go.mod".to_string(),
        "pom.xml".to_string(),
        "build.gradle".to_string(),
        "gemfile".to_string(),
        "composer.json".to_string(),
        "dockerfile".to_string(),
    ]
}

fn default_grounded_audit_force_synthesis_prompt() -> Option<String> {
    Some(
        "Tooling or delegation has stalled. Using only the concrete evidence already present in this conversation, produce a grounded audit report with these sections exactly: {required_sections_csv}. Only keep findings as confirmed if they are directly supported by code or config already shown. Downgrade unsupported concerns into hypotheses. Every hypothesis must cite a concrete file/config or observed behavior already in context, or explicitly state what evidence is still missing to confirm it. Remove generic dependency-presence or stack-template risks. Do not call any tools.\n\nIf some requested scopes remain uninspected, explicitly list them under `{gap_section_label}`: {missing_scope_refs_csv}.".to_string(),
    )
}

fn default_grounded_audit_force_synthesis_fallback() -> Option<String> {
    Some(
        "Confirmed findings\n- None supported strongly enough to promote after tooling stalled.\n\nHypotheses / lower-confidence risks\n- Some concerns may remain, but the current conversation does not contain enough direct code/config evidence to confirm them.\n\nCoverage gaps\n- Tooling stalled before full inspection. Additional direct reads would be needed to upgrade any hypothesis into a confirmed finding.\n- Requested but not inspected: {missing_scope_refs_csv}.".to_string(),
    )
}

fn default_grounded_audit_retry_prompt_template() -> Option<String> {
    Some(
        "The delegated audit result on `{topic}` did not add new useful evidence. Retry this topic at most once with one concrete file or function to inspect and one concrete question to answer. Do not return a plan, status update, or generic hypotheses.".to_string(),
    )
}

fn default_grounded_audit_tool_burst_prompt() -> Option<String> {
    Some(
        "Pause tool use. Briefly summarize what you learned, what is still uncertain, and the smallest next step. Do not call any tools in this response.".to_string(),
    )
}

fn default_grounded_audit_stall_prompt() -> Option<String> {
    Some(
        "Your previous response did not contain a usable final answer. Stop exploring and provide either concrete findings or a short bounded summary of what is known, what blocked progress, and the smallest remaining useful next step.".to_string(),
    )
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
pub(crate) struct BudgetHitSummarizationConfig {
    #[serde(default = "default_budget_summarization_enabled")]
    pub(crate) enabled: bool,
    #[serde(default = "default_budget_summarization_prompt")]
    pub(crate) prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_key: Option<String>,
}

impl Default for BudgetHitSummarizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prompt: default_budget_summarization_prompt(),
            model_key: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct RunBudgetsConfig {
    #[serde(default = "default_run_budgets_enabled")]
    pub(crate) enabled: bool,
    #[serde(default = "default_run_budget_llm_calls")]
    pub(crate) llm_calls_per_window: u32,
    #[serde(default = "default_run_budget_tool_calls")]
    pub(crate) tool_calls_per_window: u32,
    #[serde(default = "default_run_budget_spawned_agents")]
    pub(crate) spawned_agents_per_window: u32,
    #[serde(default = "default_run_budget_streamed_tokens")]
    pub(crate) streamed_tokens_per_window: u64,
    #[serde(default = "default_run_budget_wall_clock_seconds")]
    pub(crate) wall_clock_seconds_per_window: u64,
    #[serde(default = "default_run_budget_behaviors")]
    pub(crate) applies_to_behaviors: Vec<String>,
    /// What to do when a soft budget is hit: "pause" | "summarize" | "stop"
    #[serde(default = "default_on_limit")]
    pub(crate) on_limit: String,
    #[serde(default)]
    pub(crate) summarization: BudgetHitSummarizationConfig,
}

impl Default for RunBudgetsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            llm_calls_per_window: 20,
            tool_calls_per_window: 60,
            spawned_agents_per_window: 8,
            streamed_tokens_per_window: 75_000,
            wall_clock_seconds_per_window: 600,
            applies_to_behaviors: vec![],
            on_limit: default_on_limit(),
            summarization: BudgetHitSummarizationConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AuditEvidenceGrade {
    Inferred,
    CommandOnly,
    ConfigContent,
    CodeContent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct CoverageManifestConfig {
    #[serde(default = "default_coverage_manifest_enabled")]
    pub(crate) enabled: bool,
    #[serde(default = "default_coverage_manifest_require_resolution")]
    pub(crate) require_resolution: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) unresolved_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) gap_section_label: Option<String>,
}

impl Default for CoverageManifestConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require_resolution: false,
            unresolved_prompt: None,
            gap_section_label: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct SectionRule {
    pub(crate) section_name: String,
    #[serde(default)]
    pub(crate) require_file_reference: bool,
    #[serde(default)]
    pub(crate) disallow_template_phrases: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rewrite_loop_prompt: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResponseRewriteConfig {
    #[serde(default = "default_response_rewrite_enabled")]
    pub(crate) enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) min_evidence_grade_for_severity: Option<AuditEvidenceGrade>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rewrite_prompt: Option<String>,
}

impl Default for ResponseRewriteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_evidence_grade_for_severity: None,
            rewrite_prompt: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct EvidenceGradingConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) min_grade_to_synthesize: Option<AuditEvidenceGrade>,
    #[serde(default)]
    pub(crate) code_signals: Vec<String>,
    #[serde(default)]
    pub(crate) config_signals: Vec<String>,
    #[serde(default)]
    pub(crate) command_signals: Vec<String>,
}

impl Default for EvidenceGradingConfig {
    fn default() -> Self {
        Self {
            min_grade_to_synthesize: None,
            code_signals: Vec::new(),
            config_signals: Vec::new(),
            command_signals: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ForceSynthesisConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) after_n_completed_reports: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) after_n_issue_reports: Option<usize>,
    #[serde(default = "default_force_synthesis_require_coverage_gap_signal")]
    pub(crate) require_coverage_gap_signal: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) min_targeted_topics: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) min_evidence_backed_topics: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) fallback_text: Option<String>,
}

impl Default for ForceSynthesisConfig {
    fn default() -> Self {
        Self {
            after_n_completed_reports: None,
            after_n_issue_reports: None,
            require_coverage_gap_signal: false,
            min_targeted_topics: None,
            min_evidence_backed_topics: None,
            prompt: None,
            fallback_text: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct DelegationValidationConfig {
    #[serde(default = "default_delegation_validation_enabled")]
    pub(crate) enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_weak_retries: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) retry_prompt_template: Option<String>,
}

impl Default for DelegationValidationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_weak_retries: None,
            retry_prompt_template: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub(crate) struct ToolBurstReflectionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prompt: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub(crate) struct NonProgressConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stall_prompt: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) system_prompt_injection: Option<String>,
    #[serde(default = "default_runtime_note_enabled")]
    pub(crate) runtime_note_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) runtime_note_template: Option<String>,
    #[serde(default)]
    pub(crate) coverage_manifest: CoverageManifestConfig,
    #[serde(default)]
    pub(crate) required_sections: Vec<String>,
    #[serde(default)]
    pub(crate) section_rules: Vec<SectionRule>,
    #[serde(default)]
    pub(crate) response_rewrite: ResponseRewriteConfig,
    #[serde(default)]
    pub(crate) evidence_grading: EvidenceGradingConfig,
    #[serde(default)]
    pub(crate) force_synthesis: ForceSynthesisConfig,
    #[serde(default)]
    pub(crate) delegation_validation: DelegationValidationConfig,
    #[serde(default)]
    pub(crate) tool_burst_reflection: ToolBurstReflectionConfig,
    #[serde(default)]
    pub(crate) non_progress: NonProgressConfig,
}

impl BehaviorTriggerConfig {
    pub(crate) fn default_grounded_audit() -> Self {
        Self {
            behavior_id: GROUNDED_AUDIT_BEHAVIOR_ID.to_string(),
            enabled: true,
            keyword_triggers: default_grounded_audit_keyword_triggers(),
            embedding_trigger_phrases: default_grounded_audit_embedding_triggers(),
            similarity_threshold: None,
            system_prompt_injection: default_grounded_audit_system_prompt_injection(),
            runtime_note_enabled: true,
            runtime_note_template: default_grounded_audit_runtime_note_template(),
            coverage_manifest: CoverageManifestConfig {
                enabled: true,
                require_resolution: true,
                unresolved_prompt: default_grounded_audit_unresolved_prompt(),
                gap_section_label: default_grounded_audit_gap_section_label(),
            },
            required_sections: default_grounded_audit_required_sections(),
            section_rules: default_grounded_audit_section_rules(),
            response_rewrite: ResponseRewriteConfig {
                enabled: true,
                min_evidence_grade_for_severity: Some(AuditEvidenceGrade::ConfigContent),
                rewrite_prompt: default_grounded_audit_response_rewrite_prompt(),
            },
            evidence_grading: EvidenceGradingConfig {
                min_grade_to_synthesize: Some(AuditEvidenceGrade::ConfigContent),
                code_signals: default_grounded_audit_code_signals(),
                config_signals: default_grounded_audit_config_signals(),
                command_signals: default_grounded_audit_command_signals(),
            },
            force_synthesis: ForceSynthesisConfig {
                after_n_completed_reports: Some(3),
                after_n_issue_reports: Some(1),
                require_coverage_gap_signal: true,
                min_targeted_topics: Some(3),
                min_evidence_backed_topics: Some(3),
                prompt: default_grounded_audit_force_synthesis_prompt(),
                fallback_text: default_grounded_audit_force_synthesis_fallback(),
            },
            delegation_validation: DelegationValidationConfig {
                enabled: true,
                max_weak_retries: Some(1),
                retry_prompt_template: default_grounded_audit_retry_prompt_template(),
            },
            tool_burst_reflection: ToolBurstReflectionConfig {
                limit: Some(3),
                prompt: default_grounded_audit_tool_burst_prompt(),
            },
            non_progress: NonProgressConfig {
                limit: Some(4),
                stall_prompt: default_grounded_audit_stall_prompt(),
            },
        }
    }

    pub(crate) fn has_audit_payload(&self) -> bool {
        self.system_prompt_injection.is_some()
            || self.runtime_note_enabled
            || self.runtime_note_template.is_some()
            || self.coverage_manifest != CoverageManifestConfig::default()
            || !self.required_sections.is_empty()
            || !self.section_rules.is_empty()
            || self.response_rewrite != ResponseRewriteConfig::default()
            || self.evidence_grading != EvidenceGradingConfig::default()
            || self.force_synthesis != ForceSynthesisConfig::default()
            || self.delegation_validation != DelegationValidationConfig::default()
            || self.tool_burst_reflection != ToolBurstReflectionConfig::default()
            || self.non_progress != NonProgressConfig::default()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedAuditBehaviorConfig {
    pub(crate) system_prompt_injection: Option<String>,
    pub(crate) runtime_note_enabled: bool,
    pub(crate) runtime_note_template: Option<String>,
    pub(crate) coverage_manifest: CoverageManifestConfig,
    pub(crate) required_sections: Vec<String>,
    pub(crate) section_rules: Vec<SectionRule>,
    pub(crate) response_rewrite: ResponseRewriteConfig,
    pub(crate) evidence_grading: EvidenceGradingConfig,
    pub(crate) force_synthesis: ForceSynthesisConfig,
    pub(crate) delegation_validation: DelegationValidationConfig,
    pub(crate) tool_burst_reflection: ToolBurstReflectionConfig,
    pub(crate) non_progress: NonProgressConfig,
}

impl Default for ResolvedAuditBehaviorConfig {
    fn default() -> Self {
        Self {
            system_prompt_injection: None,
            runtime_note_enabled: false,
            runtime_note_template: None,
            coverage_manifest: CoverageManifestConfig::default(),
            required_sections: Vec::new(),
            section_rules: Vec::new(),
            response_rewrite: ResponseRewriteConfig::default(),
            evidence_grading: EvidenceGradingConfig::default(),
            force_synthesis: ForceSynthesisConfig::default(),
            delegation_validation: DelegationValidationConfig::default(),
            tool_burst_reflection: ToolBurstReflectionConfig::default(),
            non_progress: NonProgressConfig::default(),
        }
    }
}

impl ResolvedAuditBehaviorConfig {
    pub(crate) fn gap_section_label(&self) -> &str {
        self.coverage_manifest
            .gap_section_label
            .as_deref()
            .unwrap_or("Coverage gaps")
    }
}

fn merge_optional_string(target: &mut Option<String>, source: &Option<String>) {
    if let Some(value) = source.as_ref() {
        *target = Some(value.clone());
    }
}

fn merge_optional_usize(target: &mut Option<usize>, source: &Option<usize>) {
    if let Some(value) = source {
        *target = Some(*value);
    }
}

fn merge_optional_grade(
    target: &mut Option<AuditEvidenceGrade>,
    source: &Option<AuditEvidenceGrade>,
) {
    if let Some(value) = source {
        *target = Some(*value);
    }
}

fn merge_string_list(target: &mut Vec<String>, source: &[String]) {
    for item in source {
        if !target.iter().any(|existing| existing == item) {
            target.push(item.clone());
        }
    }
}

fn merge_section_rules(target: &mut Vec<SectionRule>, source: &[SectionRule]) {
    let mut by_name = target
        .iter()
        .enumerate()
        .map(|(idx, rule)| (rule.section_name.to_lowercase(), idx))
        .collect::<HashMap<_, _>>();
    for rule in source {
        let key = rule.section_name.to_lowercase();
        if let Some(idx) = by_name.get(&key).copied() {
            target[idx] = rule.clone();
        } else {
            by_name.insert(key, target.len());
            target.push(rule.clone());
        }
    }
}

fn merge_behavior_into_audit_config(
    target: &mut ResolvedAuditBehaviorConfig,
    behavior: &BehaviorTriggerConfig,
) {
    merge_optional_string(
        &mut target.system_prompt_injection,
        &behavior.system_prompt_injection,
    );
    target.runtime_note_enabled |= behavior.runtime_note_enabled;
    merge_optional_string(
        &mut target.runtime_note_template,
        &behavior.runtime_note_template,
    );

    target.coverage_manifest.enabled |= behavior.coverage_manifest.enabled;
    target.coverage_manifest.require_resolution |= behavior.coverage_manifest.require_resolution;
    merge_optional_string(
        &mut target.coverage_manifest.unresolved_prompt,
        &behavior.coverage_manifest.unresolved_prompt,
    );
    merge_optional_string(
        &mut target.coverage_manifest.gap_section_label,
        &behavior.coverage_manifest.gap_section_label,
    );

    merge_string_list(&mut target.required_sections, &behavior.required_sections);
    merge_section_rules(&mut target.section_rules, &behavior.section_rules);

    target.response_rewrite.enabled |= behavior.response_rewrite.enabled;
    merge_optional_grade(
        &mut target.response_rewrite.min_evidence_grade_for_severity,
        &behavior.response_rewrite.min_evidence_grade_for_severity,
    );
    merge_optional_string(
        &mut target.response_rewrite.rewrite_prompt,
        &behavior.response_rewrite.rewrite_prompt,
    );

    merge_optional_grade(
        &mut target.evidence_grading.min_grade_to_synthesize,
        &behavior.evidence_grading.min_grade_to_synthesize,
    );
    merge_string_list(
        &mut target.evidence_grading.code_signals,
        &behavior.evidence_grading.code_signals,
    );
    merge_string_list(
        &mut target.evidence_grading.config_signals,
        &behavior.evidence_grading.config_signals,
    );
    merge_string_list(
        &mut target.evidence_grading.command_signals,
        &behavior.evidence_grading.command_signals,
    );

    merge_optional_usize(
        &mut target.force_synthesis.after_n_completed_reports,
        &behavior.force_synthesis.after_n_completed_reports,
    );
    merge_optional_usize(
        &mut target.force_synthesis.after_n_issue_reports,
        &behavior.force_synthesis.after_n_issue_reports,
    );
    target.force_synthesis.require_coverage_gap_signal |=
        behavior.force_synthesis.require_coverage_gap_signal;
    merge_optional_usize(
        &mut target.force_synthesis.min_targeted_topics,
        &behavior.force_synthesis.min_targeted_topics,
    );
    merge_optional_usize(
        &mut target.force_synthesis.min_evidence_backed_topics,
        &behavior.force_synthesis.min_evidence_backed_topics,
    );
    merge_optional_string(
        &mut target.force_synthesis.prompt,
        &behavior.force_synthesis.prompt,
    );
    merge_optional_string(
        &mut target.force_synthesis.fallback_text,
        &behavior.force_synthesis.fallback_text,
    );

    target.delegation_validation.enabled |= behavior.delegation_validation.enabled;
    merge_optional_usize(
        &mut target.delegation_validation.max_weak_retries,
        &behavior.delegation_validation.max_weak_retries,
    );
    merge_optional_string(
        &mut target.delegation_validation.retry_prompt_template,
        &behavior.delegation_validation.retry_prompt_template,
    );

    merge_optional_usize(
        &mut target.tool_burst_reflection.limit,
        &behavior.tool_burst_reflection.limit,
    );
    merge_optional_string(
        &mut target.tool_burst_reflection.prompt,
        &behavior.tool_burst_reflection.prompt,
    );

    merge_optional_usize(&mut target.non_progress.limit, &behavior.non_progress.limit);
    merge_optional_string(
        &mut target.non_progress.stall_prompt,
        &behavior.non_progress.stall_prompt,
    );
}

pub(crate) fn resolve_audit_behavior_config(
    active_behaviors: &HashSet<String>,
    config: &BehaviorTriggersConfig,
) -> Option<ResolvedAuditBehaviorConfig> {
    let mut resolved = ResolvedAuditBehaviorConfig::default();
    let mut found = false;

    for behavior in &config.behaviors {
        if !behavior.enabled || !active_behaviors.contains(&behavior.behavior_id) {
            continue;
        }
        if !behavior.has_audit_payload() {
            continue;
        }
        merge_behavior_into_audit_config(&mut resolved, behavior);
        found = true;
    }

    found.then_some(resolved)
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
    #[serde(default)]
    pub(crate) run_budgets: RunBudgetsConfig,
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
            run_budgets: RunBudgetsConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_grounded_audit_resolves_to_current_profile() {
        let config = BehaviorTriggersConfig::default();
        let active = HashSet::from([GROUNDED_AUDIT_BEHAVIOR_ID.to_string()]);

        let resolved = resolve_audit_behavior_config(&active, &config).unwrap();

        assert!(resolved.runtime_note_enabled);
        let prompt = resolved.system_prompt_injection.as_deref().unwrap_or("");
        assert!(prompt.contains("Grounded audit behavior is active"));
        assert!(prompt.contains("services"));
        assert!(prompt.contains("code, configuration, policy, or command evidence"));
        assert!(resolved.coverage_manifest.enabled);
        assert_eq!(
            resolved.coverage_manifest.gap_section_label.as_deref(),
            Some("Coverage gaps")
        );
        assert_eq!(
            resolved.response_rewrite.min_evidence_grade_for_severity,
            Some(AuditEvidenceGrade::ConfigContent)
        );
        assert!(resolved
            .evidence_grading
            .code_signals
            .contains(&"def ".to_string()));
        assert!(resolved
            .evidence_grading
            .command_signals
            .contains(&"go.mod".to_string()));
        assert_eq!(resolved.delegation_validation.max_weak_retries, Some(1));
        assert_eq!(resolved.tool_burst_reflection.limit, Some(3));
        assert_eq!(resolved.non_progress.limit, Some(4));
    }

    #[test]
    fn merges_active_behaviors_by_block_in_config_order() {
        let mut config = BehaviorTriggersConfig::default();
        config.behaviors.push(BehaviorTriggerConfig {
            behavior_id: "custom_override".to_string(),
            enabled: true,
            keyword_triggers: vec![],
            embedding_trigger_phrases: vec![],
            similarity_threshold: None,
            system_prompt_injection: Some("override prompt".to_string()),
            runtime_note_enabled: false,
            runtime_note_template: Some("Covered:\n{covered_topics_bullets}".to_string()),
            coverage_manifest: CoverageManifestConfig {
                enabled: true,
                require_resolution: false,
                unresolved_prompt: Some("Missing: {unresolved_scopes_csv}".to_string()),
                gap_section_label: Some("Custom gaps".to_string()),
            },
            required_sections: vec!["Confirmed findings".to_string(), "Custom gaps".to_string()],
            section_rules: vec![SectionRule {
                section_name: "Confirmed findings".to_string(),
                require_file_reference: true,
                disallow_template_phrases: vec!["neutral".to_string()],
                rewrite_loop_prompt: Some("rewrite confirmed".to_string()),
            }],
            response_rewrite: ResponseRewriteConfig {
                enabled: true,
                min_evidence_grade_for_severity: Some(AuditEvidenceGrade::CodeContent),
                rewrite_prompt: Some("rewrite {candidate_response}".to_string()),
            },
            evidence_grading: EvidenceGradingConfig {
                min_grade_to_synthesize: Some(AuditEvidenceGrade::CodeContent),
                code_signals: vec!["custom_code".to_string()],
                config_signals: vec![],
                command_signals: vec!["custom_command".to_string()],
            },
            force_synthesis: ForceSynthesisConfig {
                after_n_completed_reports: Some(5),
                after_n_issue_reports: Some(2),
                require_coverage_gap_signal: false,
                min_targeted_topics: Some(4),
                min_evidence_backed_topics: Some(2),
                prompt: Some("force {missing_scope_refs_csv}".to_string()),
                fallback_text: Some("fallback {missing_scope_refs_csv}".to_string()),
            },
            delegation_validation: DelegationValidationConfig {
                enabled: true,
                max_weak_retries: Some(2),
                retry_prompt_template: Some("retry {topic}".to_string()),
            },
            tool_burst_reflection: ToolBurstReflectionConfig {
                limit: Some(5),
                prompt: Some("reflect".to_string()),
            },
            non_progress: NonProgressConfig {
                limit: Some(6),
                stall_prompt: Some("stall".to_string()),
            },
        });

        let active = HashSet::from([
            GROUNDED_AUDIT_BEHAVIOR_ID.to_string(),
            "custom_override".to_string(),
        ]);
        let resolved = resolve_audit_behavior_config(&active, &config).unwrap();

        assert_eq!(
            resolved.system_prompt_injection.as_deref(),
            Some("override prompt")
        );
        assert!(resolved.runtime_note_enabled);
        assert_eq!(
            resolved.coverage_manifest.gap_section_label.as_deref(),
            Some("Custom gaps")
        );
        assert_eq!(
            resolved.required_sections,
            vec![
                "Confirmed findings".to_string(),
                "Hypotheses / lower-confidence risks".to_string(),
                "Coverage gaps".to_string(),
                "Custom gaps".to_string(),
            ]
        );
        let confirmed_rule = resolved
            .section_rules
            .iter()
            .find(|rule| rule.section_name == "Confirmed findings")
            .unwrap();
        assert!(confirmed_rule.require_file_reference);
        assert_eq!(
            resolved.response_rewrite.min_evidence_grade_for_severity,
            Some(AuditEvidenceGrade::CodeContent)
        );
        assert!(resolved
            .evidence_grading
            .code_signals
            .contains(&"custom_code".to_string()));
        assert!(resolved
            .evidence_grading
            .code_signals
            .contains(&"fn ".to_string()));
        assert_eq!(resolved.force_synthesis.after_n_completed_reports, Some(5));
        assert_eq!(resolved.delegation_validation.max_weak_retries, Some(2));
        assert_eq!(resolved.tool_burst_reflection.limit, Some(5));
        assert_eq!(resolved.non_progress.limit, Some(6));
    }
}
