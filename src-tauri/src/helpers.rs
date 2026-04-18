use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::agent_config::{
    BehaviorTriggerConfig, BehaviorTriggersConfig, GROUNDED_AUDIT_BEHAVIOR_ID,
};
use crate::memory::{CommandHistory, COMMAND_HISTORY_CONTEXT_LIMIT};
use crate::models::create_embeddings;
use crate::state::BehaviorTriggerCache;

pub(crate) fn lm_base_url() -> String {
    let url = std::env::var("LM_STUDIO_URL")
        .unwrap_or_else(|_| "http://localhost:1234/api/v1/chat".to_string());
    url.trim_end_matches('/')
        .trim_end_matches("/chat")
        .trim_end_matches("/v1")
        .trim_end_matches("/api")
        .to_string()
}

pub(crate) fn default_true() -> bool {
    true
}
pub(crate) fn default_priority() -> u8 {
    128
}
pub(crate) const DEFAULT_CONTEXT_LIMIT: u64 = 8_000;

#[derive(Clone, Copy)]
pub(crate) struct ContextBudget {
    pub(crate) limit: u64,
    pub(crate) estimated_used: u64,
    pub(crate) remaining: u64,
}

pub(crate) fn estimate_text_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    chars.div_ceil(4).max(1)
}

pub(crate) fn estimate_message_tokens(messages: &[Value]) -> u64 {
    messages
        .iter()
        .map(|message| {
            let content = message["content"].as_str().unwrap_or("");
            estimate_text_tokens(content) + 8
        })
        .sum()
}

pub(crate) fn compute_context_budget(
    system_prompt: &str,
    history: &[Value],
    message: &str,
    context_limit: Option<u64>,
    extra_token_overhead: u64,
) -> ContextBudget {
    let limit = context_limit.unwrap_or(DEFAULT_CONTEXT_LIMIT);
    let estimated_used = estimate_text_tokens(system_prompt)
        + estimate_message_tokens(history)
        + estimate_text_tokens(message)
        + extra_token_overhead;
    let remaining = limit.saturating_sub(estimated_used);
    ContextBudget {
        limit,
        estimated_used,
        remaining,
    }
}

pub(crate) fn with_context_budget(system_prompt: &str, budget: ContextBudget) -> String {
    let note = format!(
        "Runtime context budget: estimated {} / {} tokens used before generation; about {} tokens remain. Plan your reads and tool usage accordingly.",
        budget.estimated_used, budget.limit, budget.remaining
    );
    if system_prompt.trim().is_empty() {
        note
    } else {
        format!("{}\n\n{}", note, system_prompt)
    }
}

pub(crate) fn apply_runtime_agent_rules(system_prompt: &str, is_manager: bool) -> String {
    let base_rules = if is_manager {
        "Runtime execution contract:\n- You own the task end-to-end and must either finish it or continue working internally.\n- Do not stop at a plan, status update, or \"I asked another agent\" unless the user explicitly requested that.\n- Prefer inspecting the workspace over asking the user when the answer can be discovered locally.\n- `AskUserQuestion` is not usable in this environment; make a reasonable assumption and continue.\n- `broadcast_message` with `await_reply=false` only queues a message; if you need results now, use `await_reply=true` or inspect the target agent directly.\n- If a delegated worker returns an empty, vague, or partial answer, retry once with a narrower evidence-seeking task naming one concrete file/function and one concrete question; if the retry is still weak, stop delegating and synthesize or inspect locally.\n- Before replying to the user, ensure you are returning the requested result, not merely progress on obtaining it."
    } else {
        "Runtime execution contract:\n- Execute the assigned task directly.\n- Do not stop at a plan, status update, or statement of intent.\n- If a tool is blocked, adapt and try another minimal approach.\n- Return the concrete result requested by the caller."
    };

    if system_prompt.trim().is_empty() {
        base_rules.to_string()
    } else {
        format!("{}\n\n{}", base_rules, system_prompt)
    }
}

pub(crate) fn apply_runtime_agent_context(
    system_prompt: &str,
    is_manager: bool,
    command_history: &CommandHistory,
) -> String {
    let base = apply_runtime_agent_rules(system_prompt, is_manager);
    let recent_commands = command_history.summarize_recent(COMMAND_HISTORY_CONTEXT_LIMIT);

    if recent_commands.is_empty() {
        base
    } else {
        format!(
            "{}\n\nRecent command executions in this thread:\n{}\n\nReuse these results instead of re-running the same command unless the situation has materially changed.",
            base,
            recent_commands
        )
    }
}

pub(crate) fn is_manager_only_tool(name: &str) -> bool {
    matches!(
        name,
        "spawn_agent"
            | "list_agents"
            | "kill_agent"
            | "pause_agent"
            | "resume_agent"
            | "broadcast_message"
            | "fork_agent"
            | "aggregate_results"
            | "pipe_message"
    )
}

pub(crate) fn is_manager_blocked_tool(name: &str) -> bool {
    matches!(
        name,
        "bash"
            | "read_file"
            | "write_file"
            | "edit_file"
            | "glob_search"
            | "grep_search"
            | "WebFetch"
            | "WebSearch"
            | "REPL"
            | "PowerShell"
            | "Sleep"
            | "TaskCreate"
            | "RunTaskPacket"
            | "TaskGet"
            | "TaskList"
            | "TaskStop"
            | "TaskOutput"
            | "TaskUpdate"
    )
}

pub(crate) fn canonical_manager_role_prompt() -> &'static str {
    "You are the manager agent responsible for completing the user’s task end-to-end.\n\nCore rules:\n- Finish the task; do not stop at an information plan, status update, or delegation summary.\n- Prefer targeted reads and concrete inspection over broad exploration.\n- Validate subagent outputs before trusting them.\n- If a worker is vague, blocked, speculative, or fails to add new useful evidence, redirect it once with a narrower evidence-seeking task, then stop delegating and synthesize locally from verified evidence.\n- If repeated work starts circling around the same file, function, or question, stop delegating and synthesize from the current verified evidence.\n\nOutput discipline:\n- Return the requested result, not progress toward it.\n- Prefer concise, evidence-first conclusions with file references when available."
}

pub(crate) fn grounded_audit_behavior_prompt() -> &'static str {
    "Grounded audit behavior is active for this task.\n- Treat explicitly requested files, directories, file classes, and subsystems as a coverage checklist.\n- Use the sections `Confirmed findings`, `Hypotheses / lower-confidence risks`, and `Coverage gaps` when reporting audit-style results.\n- A confirmed finding must be grounded in code or config actually inspected.\n- Do not label anything `High` or `Critical` without direct code/config evidence.\n- A hypothesis must cite a concrete file/config or observed behavior, or explicitly state what evidence is still missing to confirm it.\n- If evidence is insufficient, say `insufficient evidence` instead of inventing a finding."
}

pub(crate) async fn resolve_active_behaviors(
    message: &str,
    inherited: Option<&HashSet<String>>,
    config: &BehaviorTriggersConfig,
    cache: &Arc<Mutex<BehaviorTriggerCache>>,
) -> HashSet<String> {
    let mut active = HashSet::new();
    if !config.enabled {
        return active;
    }

    let normalized_message = normalize_behavior_trigger_text(message);
    if normalized_message.is_empty() {
        return active;
    }

    let global_model_key = config
        .embedding_model_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let mut embedding_candidates: Vec<&BehaviorTriggerConfig> = Vec::new();
    for behavior in &config.behaviors {
        if !behavior.enabled || !is_supported_behavior_id(&behavior.behavior_id) {
            continue;
        }
        if inherited
            .map(|set| set.contains(&behavior.behavior_id))
            .unwrap_or(false)
        {
            active.insert(behavior.behavior_id.clone());
            continue;
        }
        if behavior
            .keyword_triggers
            .iter()
            .map(|keyword| normalize_behavior_trigger_text(keyword))
            .filter(|keyword| !keyword.is_empty())
            .any(|keyword| normalized_message.contains(&keyword))
        {
            active.insert(behavior.behavior_id.clone());
            continue;
        }
        if global_model_key.is_some() && !behavior.embedding_trigger_phrases.is_empty() {
            embedding_candidates.push(behavior);
        }
    }

    let Some(model_key) = global_model_key else {
        return active;
    };
    if embedding_candidates.is_empty() {
        return active;
    }

    let Some(message_embedding) = get_or_create_cached_embedding(
        cache,
        &embedding_cache_key("message", None, model_key, &normalized_message),
        model_key,
        &normalized_message,
    )
    .await
    else {
        return active;
    };

    for behavior in embedding_candidates {
        let threshold = behavior
            .similarity_threshold
            .unwrap_or(config.default_similarity_threshold)
            .clamp(0.0, 1.0);
        let mut matched = false;
        for phrase in &behavior.embedding_trigger_phrases {
            let normalized_phrase = normalize_behavior_trigger_text(phrase);
            if normalized_phrase.is_empty() {
                continue;
            }
            let Some(phrase_embedding) = get_or_create_cached_embedding(
                cache,
                &embedding_cache_key(
                    "phrase",
                    Some(&behavior.behavior_id),
                    model_key,
                    &normalized_phrase,
                ),
                model_key,
                &normalized_phrase,
            )
            .await
            else {
                continue;
            };
            if cosine_similarity(&message_embedding, &phrase_embedding)
                .map(|similarity| similarity >= threshold)
                .unwrap_or(false)
            {
                matched = true;
                break;
            }
        }
        if matched {
            active.insert(behavior.behavior_id.clone());
        }
    }

    active
}

pub(crate) fn is_low_value_audit_response(response: &str) -> bool {
    let lower = response.trim().to_lowercase();
    if lower.is_empty() {
        return true;
    }

    if [
        "what is known:",
        "what blocked progress:",
        "smallest remaining useful next step:",
        "information plan",
        "status update request",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return true;
    }

    let mentions_risk = [
        "vulnerability",
        "security concern",
        "risk",
        "severity",
        "critical",
        "high severity",
        "medium severity",
        "low severity",
        "finding",
        "issue",
    ]
    .iter()
    .any(|marker| lower.contains(marker));

    if mentions_risk && !has_file_reference(response) && !lower.contains("insufficient evidence") {
        return true;
    }

    let has_confirmed = lower.contains("confirmed findings");
    let has_real_issue_signal = [
        "injection",
        "xss",
        "csp",
        "race condition",
        "unbounded",
        "exhaustion",
        "bypass",
        "unsafe",
        "unauthorized",
        "path traversal",
        "oom",
        "resource exhaustion",
        "withglobaltauri",
        "vulnerab",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    let only_observations = [
        "xss mitigation",
        "absence of dangerous",
        "no explicitly defined broad scopes",
        "no usage of",
        "project is",
        "backend is built",
        "frontend architecture includes",
        "contains modules",
        "hardcoded network port",
        "mitigation",
    ]
    .iter()
    .any(|marker| lower.contains(marker));

    if has_confirmed && only_observations && !has_real_issue_signal {
        return true;
    }

    has_template_only_hypotheses(response)
}

fn has_template_only_hypotheses(response: &str) -> bool {
    let hypotheses = extract_named_section(
        response,
        "hypotheses / lower-confidence risks",
        &["coverage gaps"],
    );
    if hypotheses.trim().is_empty() {
        return false;
    }

    let mut bullet_count = 0usize;
    let mut weak_count = 0usize;
    for line in hypotheses.lines().map(str::trim) {
        if line.is_empty() {
            continue;
        }
        if !(line.starts_with('-')
            || line.starts_with('*')
            || line.starts_with("1.")
            || line.starts_with("2.")
            || line.starts_with("3."))
        {
            continue;
        }
        bullet_count += 1;
        let lower = line.to_lowercase();
        let has_anchor = has_file_reference(line)
            || has_missing_proof_marker(&lower)
            || has_observed_behavior_marker(&lower);
        if is_template_hypothesis(&lower) && !has_anchor {
            weak_count += 1;
        }
    }

    bullet_count > 0 && weak_count == bullet_count
}

fn extract_named_section(response: &str, start: &str, end_markers: &[&str]) -> String {
    let lower = response.to_lowercase();
    let Some(start_idx) = lower.find(start) else {
        return String::new();
    };
    let content_start = response[start_idx..]
        .find('\n')
        .map(|offset| start_idx + offset + 1)
        .unwrap_or(response.len());
    let end_idx = end_markers
        .iter()
        .filter_map(|marker| lower[content_start..].find(marker).map(|idx| content_start + idx))
        .min()
        .unwrap_or(response.len());
    response[content_start..end_idx].trim().to_string()
}

fn has_missing_proof_marker(lower: &str) -> bool {
    [
        "insufficient evidence",
        "not enough evidence",
        "unable to confirm",
        "would need to inspect",
        "would need direct inspection",
        "would need to verify",
        "requires further inspection",
        "missing proof",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn has_observed_behavior_marker(lower: &str) -> bool {
    [
        "uses ",
        "calls ",
        "passes ",
        "spawns ",
        "reads ",
        "writes ",
        "normalizes ",
        "sets ",
        "disables ",
        "allows ",
        "constructs ",
        "joins ",
        "matches ",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn is_template_hypothesis(lower: &str) -> bool {
    [
        "presence of ",
        "possible risk",
        "potential risk",
        "could lead to",
        "could allow",
        "could be vulnerable if",
        "may allow",
        "may be vulnerable",
        "if later",
        "if exposed",
        "dependency could",
        "crate could",
        "package could",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(crate) fn has_file_reference(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        ".rs",
        ".js",
        ".ts",
        ".json",
        ".toml",
        ".md",
        "src-tauri/",
        "src/",
        "location:",
        "file:",
        "line ",
        "lines ",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(crate) fn extract_explicit_audit_refs(text: &str) -> Vec<String> {
    let mut refs = text
        .split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '`'))
        .filter_map(normalize_audit_ref)
        .filter(|token| is_path_like_audit_ref(token))
        .collect::<Vec<_>>();
    refs.sort();
    refs.dedup();
    refs
}

pub(crate) fn audit_response_acknowledges_refs(response: &str, refs: &[String]) -> bool {
    missing_audit_refs(refs, &extract_explicit_audit_refs(response)).is_empty()
}

pub(crate) fn missing_audit_refs(required_refs: &[String], seen_refs: &[String]) -> Vec<String> {
    required_refs
        .iter()
        .filter(|required| {
            !seen_refs.iter().any(|seen| audit_ref_matches(required, seen))
        })
        .cloned()
        .collect()
}

pub(crate) fn normalize_audit_ref(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| matches!(c, '.' | ':' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' | '`'))
        .trim_start_matches("./")
        .trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_lowercase())
}

pub(crate) fn is_path_like_audit_ref(token: &str) -> bool {
    let lower = token.to_lowercase();
    if lower.ends_with(".rs")
        || lower.ends_with(".js")
        || lower.ends_with(".ts")
        || lower.ends_with(".json")
        || lower.ends_with(".toml")
        || lower.ends_with(".html")
        || lower.ends_with(".md")
        || lower.ends_with(".py")
        || lower.ends_with(".css")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
    {
        return true;
    }

    if !lower.contains('/') {
        return matches!(
            lower.as_str(),
            "package.json" | "cargo.toml" | "cargo.lock" | "readme.md" | "tauri.conf.json"
        );
    }

    let segments = lower
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.len() < 2 {
        return false;
    }

    let has_known_dir = segments.iter().any(|segment| {
        matches!(
            *segment,
            "src"
                | "src-tauri"
                | "capabilities"
                | "permissions"
                | "components"
                | "pages"
                | "lib"
                | "app"
                | "tests"
                | "scripts"
                | "assets"
                | "public"
                | "backend"
                | "frontend"
        )
    });
    let last = segments.last().copied().unwrap_or_default();
    let last_looks_like_file = last.contains('.');

    has_known_dir || last_looks_like_file
}

fn audit_ref_matches(required: &str, seen: &str) -> bool {
    required == seen
        || seen.starts_with(&(required.to_string() + "/"))
        || required.starts_with(&(seen.to_string() + "/"))
        || required
            .rsplit('/')
            .next()
            .map(|basename| seen == basename || seen.ends_with(&format!("/{}", basename)))
            .unwrap_or(false)
}

pub(crate) fn manager_prompt_needs_grounding(role: &str) -> bool {
    role.trim().is_empty()
        || role.contains("Mandatory Step: Information Plan")
        || role.contains("Information Access Policy")
        || role.contains("Audit and review contract:")
}

fn is_supported_behavior_id(behavior_id: &str) -> bool {
    behavior_id == GROUNDED_AUDIT_BEHAVIOR_ID
}

fn normalize_behavior_trigger_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_lowercase()
}

fn embedding_cache_key(
    kind: &str,
    behavior_id: Option<&str>,
    model_key: &str,
    text: &str,
) -> String {
    match behavior_id {
        Some(behavior_id) => format!(
            "{}::{}::{}::{}",
            kind, behavior_id, model_key, text
        ),
        None => format!("{}::{}::{}", kind, model_key, text),
    }
}

async fn get_or_create_cached_embedding(
    cache: &Arc<Mutex<BehaviorTriggerCache>>,
    cache_key: &str,
    model_key: &str,
    text: &str,
) -> Option<Vec<f32>> {
    if let Some(existing) = cache.lock().unwrap().embeddings.get(cache_key).cloned() {
        return Some(existing);
    }

    let embedding = create_embeddings(model_key, &[text.to_string()])
        .await
        .ok()
        .and_then(|mut vectors| vectors.drain(..).next())?;
    cache
        .lock()
        .unwrap()
        .embeddings
        .insert(cache_key.to_string(), embedding.clone());
    Some(embedding)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (l, r) in left.iter().zip(right.iter()) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    let denom = left_norm.sqrt() * right_norm.sqrt();
    if denom <= f32::EPSILON {
        None
    } else {
        Some(dot / denom)
    }
}
