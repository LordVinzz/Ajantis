use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::agent_config::RunBudgetsConfig;
use crate::chat::StreamEvent;
use crate::config_persistence::ajantis_dir;
use crate::event_sink::SharedEventSink;
use crate::memory::CommandExecution;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RunDossierCounts {
    pub(crate) targeted_reads: u32,
    pub(crate) targeted_searches: u32,
    pub(crate) broad_full_file_reads: u32,
    pub(crate) broad_directory_scans: u32,
    pub(crate) dependency_or_generated_scans: u32,
    pub(crate) blocked_commands: u32,
    pub(crate) tool_failures: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RunDossierWorkerOutcome {
    pub(crate) agent_id: String,
    pub(crate) agent_name: String,
    pub(crate) summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) observed_evidence: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) inferences: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) coverage_gaps: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RunDossier {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) inspected_paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) worker_outcomes: Vec<RunDossierWorkerOutcome>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) coverage_gaps: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) caution_flags: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) manager_draft_summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) finalizer_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) finalizer_input_summary: Vec<String>,
    #[serde(default)]
    pub(crate) counts: RunDossierCounts,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub(crate) struct RunWindowUsage {
    pub(crate) llm_calls: u32,
    pub(crate) tool_calls: u32,
    pub(crate) spawned_agents: u32,
    pub(crate) streamed_tokens: u64,
    pub(crate) embedding_calls: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct RunLimitHit {
    pub(crate) kind: String,
    pub(crate) limit: u64,
    pub(crate) observed: u64,
}

#[derive(Clone)]
pub(crate) struct PausedRunState {
    pub(crate) agent_id: String,
    pub(crate) model_key: String,
    pub(crate) system_prompt: String,
    pub(crate) messages: Vec<Value>,
    pub(crate) allowed_tools: Option<Vec<String>>,
    pub(crate) allow_manager_tools: bool,
    pub(crate) require_delegation: bool,
    pub(crate) context_limit: Option<u64>,
    pub(crate) glob_ready: bool,
    pub(crate) active_behaviors: HashSet<String>,
    pub(crate) usage: RunWindowUsage,
    pub(crate) limit_hit: RunLimitHit,
}

pub(crate) struct ActiveRunState {
    pub(crate) run_id: String,
    pub(crate) workspace_id: Option<String>,
    pub(crate) thread_id: Option<String>,
    pub(crate) workspace_path: Option<String>,
    pub(crate) journal_path: PathBuf,
    pub(crate) channel: SharedEventSink,
    pub(crate) budgets: RunBudgetsConfig,
    pub(crate) active_behaviors: HashSet<String>,
    pub(crate) usage: RunWindowUsage,
    pub(crate) window_started_at: Instant,
    pub(crate) paused: Option<PausedRunState>,
    pub(crate) waiting_confirmation: bool,
    pub(crate) cancelled: bool,
    /// Embeddings of recent text responses used for repetition detection (most recent last).
    pub(crate) recent_response_embeddings: Vec<Vec<f32>>,
    /// Fallback text store when no embedding model is configured.
    pub(crate) recent_response_texts: Vec<String>,
    /// Snapshot of the manager agent's conversation state, updated each loop iteration.
    /// Used so that budget-hit summarization always runs from the manager's full context
    /// (which includes all sub-agent tool results) rather than from a sub-agent's local view.
    pub(crate) manager_agent_id: Option<String>,
    pub(crate) manager_model_key: Option<String>,
    pub(crate) manager_messages: Vec<Value>,
    pub(crate) dossier: RunDossier,
}

pub(crate) type ActiveRuns = Arc<Mutex<HashMap<String, ActiveRunState>>>;

pub(crate) fn generate_run_id() -> String {
    format!("run-{}", chrono::Utc::now().timestamp_millis())
}

pub(crate) fn thread_data_root() -> PathBuf {
    let dir = ajantis_dir().join("thread-data");
    let _ = fs::create_dir_all(&dir);
    dir
}

pub(crate) fn thread_dir(workspace_id: &str, thread_id: &str) -> PathBuf {
    let dir = thread_data_root().join(workspace_id).join(thread_id);
    let _ = fs::create_dir_all(&dir);
    dir
}

pub(crate) fn snapshot_path(workspace_id: &str, thread_id: &str) -> PathBuf {
    thread_dir(workspace_id, thread_id).join("snapshot.json")
}

pub(crate) fn runs_dir(workspace_id: &str, thread_id: &str) -> PathBuf {
    let dir = thread_dir(workspace_id, thread_id).join("runs");
    let _ = fs::create_dir_all(&dir);
    dir
}

pub(crate) fn journal_path(workspace_id: &str, thread_id: &str, run_id: &str) -> PathBuf {
    runs_dir(workspace_id, thread_id).join(format!("{}.jsonl", run_id))
}

pub(crate) fn append_journal_entry(path: &PathBuf, record: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create run journal directory: {}", e))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Failed to open run journal: {}", e))?;
    let line = serde_json::to_string(record)
        .map_err(|e| format!("Failed to encode run journal: {}", e))?;
    writeln!(file, "{}", line).map_err(|e| format!("Failed to write run journal: {}", e))
}

pub(crate) fn emit_run_event(
    active_runs: &ActiveRuns,
    run_id: &str,
    event: StreamEvent,
) -> Result<(), String> {
    let maybe_record = active_runs
        .lock()
        .unwrap()
        .get(run_id)
        .map(|run| (run.channel.clone(), run.journal_path.clone()));

    if let Some((channel, journal_path)) = maybe_record {
        let _ = channel.send(event.clone());
        append_journal_entry(
            &journal_path,
            &json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "event": event,
            }),
        )?;
    }
    Ok(())
}

pub(crate) fn run_budget_applies(
    budgets: &RunBudgetsConfig,
    active_behaviors: &HashSet<String>,
) -> bool {
    if !budgets.enabled {
        return false;
    }
    // Empty list means "apply unconditionally to all runs".
    if budgets.applies_to_behaviors.is_empty() {
        return true;
    }
    budgets
        .applies_to_behaviors
        .iter()
        .any(|behavior| active_behaviors.contains(behavior))
}

pub(crate) fn reset_run_window(run: &mut ActiveRunState) {
    run.usage = RunWindowUsage::default();
    run.window_started_at = Instant::now();
}

pub(crate) fn primary_run_id(active_runs: &ActiveRuns) -> Option<String> {
    active_runs.lock().unwrap().keys().next().cloned()
}

pub(crate) fn record_dossier_command(
    active_runs: &ActiveRuns,
    run_id: &str,
    entry: &CommandExecution,
) {
    let mut runs = active_runs.lock().unwrap();
    let Some(run) = runs.get_mut(run_id) else {
        return;
    };

    for path in &entry.touched_paths {
        if !run
            .dossier
            .inspected_paths
            .iter()
            .any(|existing| existing == path)
        {
            run.dossier.inspected_paths.push(path.clone());
        }
    }

    match entry.classification.as_str() {
        "targeted_read" => run.dossier.counts.targeted_reads += 1,
        "targeted_search" => run.dossier.counts.targeted_searches += 1,
        "broad_full_file_read" => run.dossier.counts.broad_full_file_reads += 1,
        "broad_directory_scan" => run.dossier.counts.broad_directory_scans += 1,
        "dependency_or_generated_scan" => {
            run.dossier.counts.dependency_or_generated_scans += 1;
        }
        _ => {}
    }

    if !entry.success {
        run.dossier.counts.tool_failures += 1;
    }
}

pub(crate) fn record_dossier_blocked_command(active_runs: &ActiveRuns, run_id: &str, reason: &str) {
    let mut runs = active_runs.lock().unwrap();
    let Some(run) = runs.get_mut(run_id) else {
        return;
    };
    run.dossier.counts.blocked_commands += 1;
    if !run
        .dossier
        .caution_flags
        .iter()
        .any(|flag| flag == "Blocked commands occurred during this run.")
    {
        run.dossier
            .caution_flags
            .push("Blocked commands occurred during this run.".to_string());
    }
    if !reason.trim().is_empty() && !run.dossier.coverage_gaps.iter().any(|gap| gap == reason) {
        run.dossier.coverage_gaps.push(reason.to_string());
    }
}
