use serde_json::Value;

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
