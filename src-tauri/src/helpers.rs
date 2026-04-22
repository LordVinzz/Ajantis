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
pub(crate) const HARD_PROMPT_HISTORY_LIMIT: u64 = 32_000;

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

fn summarize_trimmed_history(history: &[Value]) -> Option<Value> {
    if history.is_empty() {
        return None;
    }
    let mut lines = Vec::new();
    for message in history.iter().rev().take(12).rev() {
        let role = message["role"].as_str().unwrap_or("unknown");
        let content = message["content"].as_str().unwrap_or("").replace('\n', " ");
        let preview = truncate_for_summary(&content, 220);
        if !preview.is_empty() {
            lines.push(format!("- {}: {}", role, preview));
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(serde_json::json!({
            "role": "system",
            "content": format!(
                "Summary of older conversation context trimmed for budget:\n{}",
                lines.join("\n")
            ),
        }))
    }
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        text.to_string()
    } else {
        let head: String = text.chars().take(max_chars).collect();
        format!("{}...", head)
    }
}

pub(crate) fn trim_history_to_budget(
    system_prompt: &str,
    history: &[Value],
    message: &str,
    context_limit: Option<u64>,
    extra_token_overhead: u64,
) -> Vec<Value> {
    let hard_limit = context_limit
        .unwrap_or(DEFAULT_CONTEXT_LIMIT)
        .min(HARD_PROMPT_HISTORY_LIMIT);
    let reserved = estimate_text_tokens(system_prompt)
        + estimate_text_tokens(message)
        + extra_token_overhead
        + 256;
    let available_for_history = hard_limit.saturating_sub(reserved);
    if available_for_history == 0 {
        return Vec::new();
    }

    let mut kept = Vec::new();
    let mut kept_tokens = 0u64;
    let mut trimmed_prefix = Vec::new();

    for item in history.iter().rev() {
        let item_tokens = estimate_message_tokens(std::slice::from_ref(item));
        if kept_tokens + item_tokens > available_for_history {
            trimmed_prefix.push(item.clone());
            continue;
        }
        kept.push(item.clone());
        kept_tokens += item_tokens;
    }

    kept.reverse();
    trimmed_prefix.reverse();

    if let Some(summary) = summarize_trimmed_history(&trimmed_prefix) {
        let summary_tokens = estimate_message_tokens(std::slice::from_ref(&summary));
        if summary_tokens <= available_for_history.saturating_sub(kept_tokens) {
            let mut with_summary = vec![summary];
            with_summary.extend(kept);
            return with_summary;
        }
    }

    kept
}

pub(crate) fn apply_runtime_agent_rules(system_prompt: &str, is_manager: bool) -> String {
    let base_rules = if is_manager {
        "Runtime execution contract:\n- You own the task end-to-end and must either finish it or continue working internally.\n- Do not stop at a plan, status update, or \"I asked another agent\" unless the user explicitly requested that.\n- Prefer inspecting the workspace over asking the user when the answer can be discovered locally.\n- `AskUserQuestion` is not usable in this environment; make a reasonable assumption and continue.\n- `broadcast_message` with `await_reply=false` only queues a message; if you need results now, use `await_reply=true` or inspect the target agent directly.\n- If the task requires creating or changing workspace files, delegate to a worker with file tools and instruct it to materialize the result on disk with `write_file` or `edit_file`.\n- A pasted code block or prose-only implementation does not count as completing a file creation or edit task unless the caller explicitly asked for chat-only output.\n- If a delegated worker returns an empty, vague, partial, or chat-only answer for a file task, retry once with a narrower evidence-seeking task naming one concrete file/function and one concrete question; if the retry is still weak, stop delegating and synthesize or inspect locally.\n- Before replying to the user, ensure you are returning the requested result, not merely progress on obtaining it."
    } else {
        "Runtime execution contract:\n- Execute the assigned task directly.\n- Do not stop at a plan, status update, or statement of intent.\n- If the task is to implement, create, or modify workspace files, use `write_file` or `edit_file` to materialize the result on disk instead of only pasting code in chat.\n- If no file path is specified for a file task, choose a sensible path and mention it in the result.\n- If a tool is blocked, adapt and try another minimal approach.\n- Return the concrete result requested by the caller."
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
    let recovery_note = weak_exploration_recovery_note(command_history);

    let mut parts = vec![base];
    if !recent_commands.is_empty() {
        parts.push(format!(
            "Recent command executions in this thread:\n{}\n\nReuse these results instead of re-running the same command unless the situation has materially changed.",
            recent_commands
        ));
    }
    if let Some(note) = recovery_note {
        parts.push(note);
    }
    parts.join("\n\n")
}

fn weak_exploration_recovery_note(command_history: &CommandHistory) -> Option<String> {
    let recent = command_history
        .entries
        .iter()
        .rev()
        .take(COMMAND_HISTORY_CONTEXT_LIMIT)
        .collect::<Vec<_>>();
    if recent.is_empty() {
        return None;
    }

    let broad_reads = recent
        .iter()
        .filter(|entry| entry.classification == "broad_full_file_read")
        .count();
    let broad_scans = recent
        .iter()
        .filter(|entry| {
            matches!(
                entry.classification.as_str(),
                "broad_directory_scan" | "dependency_or_generated_scan"
            )
        })
        .count();
    let targeted = recent
        .iter()
        .filter(|entry| {
            matches!(
                entry.classification.as_str(),
                "targeted_read" | "targeted_search"
            )
        })
        .count();

    if broad_reads + broad_scans < 3 || targeted >= 2 {
        return None;
    }

    Some(
        "Methodology recovery note:\n- Recent exploration skewed broad rather than targeted.\n- Summarize what is actually known from the inspected evidence.\n- Choose the smallest next targeted read or search instead of another broad scan.\n- Avoid dependency/build/generated directories unless they are directly in scope.".to_string(),
    )
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
    "You are the manager agent responsible for completing the user’s task end-to-end.\n\nCore rules:\n- Finish the task; do not stop at an information plan, status update, or delegation summary.\n- Prefer targeted reads, relevant entrypoints, and concrete inspection over broad exploration.\n- Do not ask workers for full repo listings or exhaustive tree dumps unless the user explicitly asked for them.\n- Exclude dependency, build, and generated directories by default (`node_modules`, `dist`, `target`, `.git`, caches) unless the task is specifically about them.\n- Validate subagent outputs before trusting them.\n- If the task requires creating or modifying workspace files, delegate to a worker with file tools and require it to materialize the changes on disk; a chat-only code block is not a completed file task unless the user explicitly asked for chat-only output.\n- If a worker is vague, blocked, speculative, fails to add new useful evidence, or answers a file task without actually writing the file, redirect it once with a narrower evidence-seeking task, then stop delegating and synthesize locally from verified evidence.\n- If repeated work starts circling around the same file, function, or question, stop delegating and synthesize from the current verified evidence.\n- If multiple worker reports overlap, synthesize them into one final answer instead of merging drafts.\n- Before the user-facing answer is finalized, produce a compact internal synthesis that captures inspected evidence, recommendations, and coverage gaps.\n\nOutput discipline:\n- Return the requested result, not progress toward it.\n- Prefer concise, evidence-first conclusions with file references when available.\n- Deliver a single synthesized answer rather than repeated sections or stitched subreports."
}

#[cfg(test)]
mod tests {
    use super::{apply_runtime_agent_rules, canonical_manager_role_prompt};

    #[test]
    fn worker_rules_require_materializing_file_tasks() {
        let rules = apply_runtime_agent_rules("", false);
        assert!(rules.contains("write_file"));
        assert!(rules.contains("materialize the result on disk"));
        assert!(rules.contains("choose a sensible path"));
    }

    #[test]
    fn manager_prompt_rejects_chat_only_file_completion() {
        let prompt = canonical_manager_role_prompt();
        assert!(prompt.contains("materialize the changes on disk"));
        assert!(prompt.contains("chat-only code block"));
    }

    #[test]
    fn manager_prompt_requires_targeted_and_deduplicated_synthesis() {
        let prompt = canonical_manager_role_prompt();
        assert!(prompt.contains("full repo listings"));
        assert!(prompt.contains("Exclude dependency, build, and generated directories"));
        assert!(prompt.contains("relevant entrypoints"));
        assert!(prompt.contains("single synthesized answer"));
    }
}

/// Returns `(active_behaviors, embedding_api_calls)` where `embedding_api_calls` counts
/// actual calls to the embedding model API (cache hits are not counted).
pub(crate) async fn resolve_active_behaviors(
    message: &str,
    inherited: Option<&HashSet<String>>,
    config: &BehaviorTriggersConfig,
    cache: &Arc<Mutex<BehaviorTriggerCache>>,
) -> (HashSet<String>, u32) {
    let mut active = HashSet::new();
    let mut api_calls: u32 = 0;
    if !config.enabled {
        return (active, api_calls);
    }

    let normalized_message = normalize_behavior_trigger_text(message);
    if normalized_message.is_empty() {
        return (active, api_calls);
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
        return (active, api_calls);
    };
    if embedding_candidates.is_empty() {
        return (active, api_calls);
    }

    let (message_embedding_opt, msg_was_api) = get_or_create_cached_embedding(
        cache,
        &embedding_cache_key("message", None, model_key, &normalized_message),
        model_key,
        &normalized_message,
    )
    .await;
    if msg_was_api {
        api_calls += 1;
    }
    let Some(message_embedding) = message_embedding_opt else {
        return (active, api_calls);
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
            let (phrase_embedding_opt, phrase_was_api) = get_or_create_cached_embedding(
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
            .await;
            if phrase_was_api {
                api_calls += 1;
            }
            let Some(phrase_embedding) = phrase_embedding_opt else {
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

    (active, api_calls)
}

pub(crate) fn has_file_reference(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        ".rs",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".kt",
        ".go",
        ".rb",
        ".php",
        ".cs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".swift",
        ".scala",
        ".sh",
        ".sql",
        ".xml",
        ".yaml",
        ".yml",
        ".ini",
        ".env",
        ".json",
        ".toml",
        ".md",
        "src/",
        "app/",
        "lib/",
        "cmd/",
        "internal/",
        "pkg/",
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
        .split(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '"' | '\'' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '`'
                )
        })
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
            !seen_refs
                .iter()
                .any(|seen| audit_ref_matches(required, seen))
        })
        .cloned()
        .collect()
}

pub(crate) fn normalize_audit_ref(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| {
            matches!(
                c,
                '.' | ':' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' | '`'
            )
        })
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
        || lower.ends_with(".jsx")
        || lower.ends_with(".tsx")
        || lower.ends_with(".java")
        || lower.ends_with(".kt")
        || lower.ends_with(".go")
        || lower.ends_with(".rb")
        || lower.ends_with(".php")
        || lower.ends_with(".cs")
        || lower.ends_with(".c")
        || lower.ends_with(".cpp")
        || lower.ends_with(".h")
        || lower.ends_with(".hpp")
        || lower.ends_with(".swift")
        || lower.ends_with(".scala")
        || lower.ends_with(".sh")
        || lower.ends_with(".sql")
        || lower.ends_with(".xml")
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
            "package.json"
                | "cargo.toml"
                | "cargo.lock"
                | "readme.md"
                | "tauri.conf.json"
                | "pyproject.toml"
                | "requirements.txt"
                | "go.mod"
                | "pom.xml"
                | "build.gradle"
                | "build.gradle.kts"
                | "gemfile"
                | "composer.json"
                | "dockerfile"
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
                | "components"
                | "pages"
                | "lib"
                | "app"
                | "cmd"
                | "internal"
                | "pkg"
                | "tests"
                | "scripts"
                | "assets"
                | "public"
                | "backend"
                | "frontend"
                | "api"
                | "controllers"
                | "services"
                | "handlers"
                | "routes"
                | "config"
                | "configs"
                | "deploy"
                | "ops"
                | "infra"
                | "capabilities"
                | "permissions"
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
        Some(behavior_id) => format!("{}::{}::{}::{}", kind, behavior_id, model_key, text),
        None => format!("{}::{}::{}", kind, model_key, text),
    }
}

/// Returns the embedding vector and whether an actual API call was made (false = cache hit).
async fn get_or_create_cached_embedding(
    cache: &Arc<Mutex<BehaviorTriggerCache>>,
    cache_key: &str,
    model_key: &str,
    text: &str,
) -> (Option<Vec<f32>>, bool) {
    if let Some(existing) = cache.lock().unwrap().embeddings.get(cache_key).cloned() {
        return (Some(existing), false);
    }

    let embedding = create_embeddings(model_key, &[text.to_string()])
        .await
        .ok()
        .and_then(|mut vectors| vectors.drain(..).next());
    if let Some(ref vec) = embedding {
        cache
            .lock()
            .unwrap()
            .embeddings
            .insert(cache_key.to_string(), vec.clone());
    }
    (embedding, true)
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
