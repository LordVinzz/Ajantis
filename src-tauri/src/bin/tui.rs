use std::fs;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::Duration;
use std::collections::{HashMap, HashSet};

use app_lib::runtime::{
    BackendDetected, BackendInstance, callback_event_sink, AgentConfig, BehaviorTriggerConfig,
    LoadConfig, ModelInfo, RoutingRule, RunDossier, RuntimeHandle, StreamEvent, ThreadSnapshot,
    WorkspaceConfig,
    WorkspaceThreadMessage, WorkspaceToolCall,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap},
    Frame, Terminal,
};
use tokio::sync::mpsc;

const CONFIG_PATH_HINT: &str = "~/.ajantis/ajantis-config.json";

#[derive(Clone, Copy, PartialEq)]
enum Panel {
    Chat = 0,
    Agents = 1,
    Routing = 2,
    Workspaces = 3,
    Settings = 4,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Normal,
    Insert,
}

#[derive(Clone, Copy, PartialEq)]
enum AgentFocus {
    List,
    Fields,
}

#[derive(Clone, Copy, PartialEq)]
enum AgentTarget {
    Agent(usize),
    GlobalPolicy,
}

#[derive(Clone)]
enum EditorKind {
    Section,
    Text { multiline: bool },
    Number,
    Picker { options: Vec<String> },
    Action,
}

#[derive(Clone)]
enum FieldKey {
    BackendType,
    BackendUrl,
    BackendApiKey,
    BackendDetect,
    BackendDetectedInfo,
    BackendInstanceUrl(usize),
    BackendInstanceModelHint(usize),
    BackendInstanceDelete(usize),
    BackendInstanceAdd,
    BackendDiscover,
    AgentName(usize),
    AgentManager(usize),
    AgentModel(usize),
    AgentMode(usize),
    AgentRole(usize),
    AgentContextLength(usize),
    AgentEvalBatch(usize),
    AgentExperts(usize),
    AgentFlashAttention(usize),
    AgentGpuKvOffload(usize),
    AgentArmed(usize),
    AgentLoadNow(usize),
    AgentDelete(usize),
    CommandDenylist,
    CommandAllowlist,
    RedundancyEnabled,
    RedundancyEmbeddingModel,
    RedundancyThreshold,
    RedundancyRetries,
    RunBudgetsEnabled,
    RunBudgetLlmCalls,
    RunBudgetToolCalls,
    RunBudgetSpawnedAgents,
    RunBudgetTokens,
    RunBudgetWallClock,
    RunBudgetAppliesToBehaviors,
    RunBudgetOnLimit,
    FinalizerEnabled,
    FinalizerRunOnCompletion,
    FinalizerRunOnBudgetStop,
    FinalizerAgentName,
    FinalizerModel,
    FinalizerTranscriptChars,
    FinalizerIncludeDossier,
    FinalizerIncludeCommands,
    FinalizerIncludeWorkers,
    FinalizerIncludeTranscript,
    FinalizerPromptCompletion,
    FinalizerPromptBudgetStop,
    BehaviorTriggersEnabled,
    BehaviorTriggersEmbeddingModel,
    BehaviorTriggersDefaultThreshold,
    BehaviorKeywords(usize),
    BehaviorJson(usize),
    ParallelInferenceEnabled,
    ParallelInferenceMaxAgents,
    RouteEnabled(String, String),
    RoutePriority(String, String),
    RouteCondition(String, String),
}

#[derive(Clone)]
struct EditorRow {
    label: String,
    display: String,
    raw: String,
    key: Option<FieldKey>,
    kind: EditorKind,
}

enum EditState {
    Text {
        key: FieldKey,
        buffer: String,
        cursor: usize,
        multiline: bool,
    },
    Picker {
        key: FieldKey,
        options: Vec<String>,
        index: usize,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum ChatFocus {
    Messages,
    Sections,
    Detail,
}

struct ActiveRunUi {
    run_id: String,
    waiting_confirmation: bool,
    limit_message: String,
}

#[derive(Default, Clone)]
struct RunUsageUi {
    llm_calls: u32,
    tool_calls: u32,
    spawned_agents: u32,
    streamed_tokens: u64,
    embedding_calls: u32,
    wall_clock_seconds: u64,
}

struct App {
    runtime: RuntimeHandle,
    panel: Panel,
    mode: Mode,
    status: String,
    backend_detail: String,

    config: AgentConfig,
    workspace_config: WorkspaceConfig,
    available_models: Vec<ModelInfo>,

    messages: Vec<WorkspaceThreadMessage>,
    active_message_by_agent: HashMap<String, usize>,
    expanded_tool_sections: HashSet<String>,
    current_run_dossier: Option<RunDossier>,
    active_run: Option<ActiveRunUi>,
    run_usage: Option<RunUsageUi>,
    active_model: Option<String>,

    active_workspace_id: Option<String>,
    active_thread_id: Option<String>,
    ws_expanded: Vec<bool>,
    ws_flat: Vec<WorkspaceRow>,
    ws_sel: usize,
    thread_updated_at: HashMap<String, String>,

    agent_focus: AgentFocus,
    agent_sel: usize,
    agent_row_sel: usize,
    routing_row_sel: usize,
    settings_row_sel: usize,
    edit_state: Option<EditState>,

    chat_input: String,
    chat_cursor: usize,
    chat_focus: ChatFocus,
    chat_message_sel: usize,
    chat_section_sel: usize,
    chat_detail_scroll: usize,
}

enum TuiEvent {
    Key(event::KeyEvent),
    Stream(StreamEvent),
    Notice(String),
    ModelsLoaded(Result<Vec<ModelInfo>, String>),
}

#[derive(Clone, Copy)]
enum ChatSection {
    Content,
    Tool(usize),
}

#[derive(Clone)]
enum WorkspaceRow {
    AddWorkspace,
    Workspace(usize),
    NewThread(usize),
    Thread {
        workspace_index: usize,
        thread_index: usize,
    },
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        original_hook(info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new().await;
    let result = run_app(&mut terminal, &mut app).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    result
}

impl App {
    async fn new() -> Self {
        let runtime = RuntimeHandle::new(
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        );
        let config = runtime.load_agent_config().unwrap_or_default();
        let workspace_config = runtime.load_workspace_config().unwrap_or_default();
        let available_models = runtime.fetch_models().await.unwrap_or_default();
        let ws_expanded = vec![false; workspace_config.workspaces.len()];
        let mut app = Self {
            runtime,
            panel: Panel::Chat,
            mode: Mode::Normal,
            status: "Ready".to_string(),
            backend_detail: format!("Config: {}", CONFIG_PATH_HINT),
            config,
            workspace_config,
            available_models,
            messages: Vec::new(),
            active_message_by_agent: HashMap::new(),
            expanded_tool_sections: HashSet::new(),
            current_run_dossier: None,
            active_run: None,
            run_usage: None,
            active_model: None,
            active_workspace_id: None,
            active_thread_id: None,
            ws_expanded,
            ws_flat: Vec::new(),
            ws_sel: 0,
            thread_updated_at: HashMap::new(),
            agent_focus: AgentFocus::List,
            agent_sel: 0,
            agent_row_sel: 0,
            routing_row_sel: 0,
            settings_row_sel: 0,
            edit_state: None,
            chat_input: String::new(),
            chat_cursor: 0,
            chat_focus: ChatFocus::Messages,
            chat_message_sel: 0,
            chat_section_sel: 0,
            chat_detail_scroll: 0,
        };
        app.refresh_thread_recency_cache();
        app.rebuild_workspace_rows();
        app.bootstrap_thread_state();
        app
    }

    fn bootstrap_thread_state(&mut self) {
        if let Some((wi, ti)) = self.ws_flat.iter().find_map(|row| match row {
            WorkspaceRow::Thread {
                workspace_index,
                thread_index,
            } => Some((*workspace_index, *thread_index)),
            _ => None,
        }) {
            let workspace_id = self.workspace_config.workspaces[wi].id.clone();
            let thread_id = self.workspace_config.workspaces[wi].threads[ti].id.clone();
            self.active_workspace_id = Some(workspace_id.clone());
            self.active_thread_id = Some(thread_id.clone());
            self.ws_expanded[wi] = true;
            self.rebuild_workspace_rows();
            self.load_thread(workspace_id, thread_id);
        }
    }

    fn load_thread(&mut self, workspace_id: String, thread_id: String) {
        if let Err(error) = self.ensure_runtime_workspace(&workspace_id) {
            self.messages = vec![system_error_message(error.clone())];
            self.runtime.set_memory_pool(vec![]);
            self.runtime.set_command_history(vec![]);
            self.current_run_dossier = None;
            self.active_run = None;
            self.chat_focus = ChatFocus::Messages;
            self.chat_message_sel = 0;
            self.chat_section_sel = 0;
            self.chat_detail_scroll = 0;
            self.status = error.clone();
            self.backend_detail = error;
            return;
        }
        match self.runtime.load_thread_snapshot(&workspace_id, &thread_id) {
            Ok(snapshot) => {
                if !snapshot.updated_at.is_empty() {
                    self.thread_updated_at.insert(
                        thread_recency_key(&workspace_id, &thread_id),
                        snapshot.updated_at.clone(),
                    );
                }
                self.messages = snapshot.message_items;
                self.active_message_by_agent.clear();
                self.expanded_tool_sections.clear();
                self.runtime.set_memory_pool(snapshot.memory_entries);
                self.runtime.set_command_history(snapshot.command_history);
                self.current_run_dossier = snapshot.run_dossier;
                self.active_run = snapshot.active_run_id.map(|run_id| ActiveRunUi {
                    run_id,
                    waiting_confirmation: false,
                    limit_message: String::new(),
                });
                self.chat_focus = ChatFocus::Messages;
                self.chat_message_sel = self.messages.len().saturating_sub(1);
                self.chat_section_sel = 0;
                self.chat_detail_scroll = 0;
                self.status = format!("Loaded thread {}/{}", workspace_id, thread_id);
                self.backend_detail = "Thread state restored.".to_string();
            }
            Err(error) => {
                self.messages = vec![system_error_message(error)];
                self.runtime.set_memory_pool(vec![]);
                self.runtime.set_command_history(vec![]);
                self.current_run_dossier = None;
                self.active_run = None;
                self.chat_focus = ChatFocus::Messages;
                self.chat_message_sel = 0;
                self.chat_section_sel = 0;
                self.chat_detail_scroll = 0;
            }
        }
        self.active_workspace_id = Some(workspace_id);
        self.active_thread_id = Some(thread_id);
    }

    fn rebuild_workspace_rows(&mut self) {
        self.ws_expanded
            .resize(self.workspace_config.workspaces.len(), false);
        self.ws_flat = build_ws_flat(self);
        self.ws_sel = self.ws_sel.min(self.ws_flat.len().saturating_sub(1));
    }

    fn refresh_thread_recency_cache(&mut self) {
        for workspace in &self.workspace_config.workspaces {
            for thread in &workspace.threads {
                if let Ok(snapshot) = self.runtime.load_thread_snapshot(&workspace.id, &thread.id) {
                    if !snapshot.updated_at.is_empty() {
                        self.thread_updated_at.insert(
                            thread_recency_key(&workspace.id, &thread.id),
                            snapshot.updated_at,
                        );
                    }
                }
            }
        }
    }

    fn sorted_thread_indices(&self, workspace_index: usize) -> Vec<usize> {
        let workspace = &self.workspace_config.workspaces[workspace_index];
        let mut indices = (0..workspace.threads.len()).collect::<Vec<_>>();
        indices.sort_by(|a, b| {
            let a_key = self
                .thread_updated_at
                .get(&thread_recency_key(&workspace.id, &workspace.threads[*a].id))
                .cloned()
                .unwrap_or_default();
            let b_key = self
                .thread_updated_at
                .get(&thread_recency_key(&workspace.id, &workspace.threads[*b].id))
                .cloned()
                .unwrap_or_default();
            b_key.cmp(&a_key).then_with(|| a.cmp(b))
        });
        indices
    }

    fn ensure_runtime_workspace(&mut self, workspace_id: &str) -> Result<(), String> {
        let workspace = self
            .workspace_config
            .workspaces
            .iter()
            .find(|workspace| workspace.id == workspace_id)
            .ok_or_else(|| format!("Workspace '{}' is not configured.", workspace_id))?;
        let canonical = fs::canonicalize(&workspace.path).map_err(|error| {
            format!(
                "Workspace '{}' is invalid or inaccessible: {}",
                workspace.path, error
            )
        })?;
        self.runtime
            .set_active_workspace(Some(canonical.to_string_lossy().to_string()))
    }

    fn add_workspace(&mut self, path: String) -> Result<(), String> {
        let canonical = fs::canonicalize(&path)
            .map_err(|error| format!("Workspace path is invalid or inaccessible: {}", error))?;
        let canonical_str = canonical.to_string_lossy().to_string();
        if let Some(existing_index) = self
            .workspace_config
            .workspaces
            .iter()
            .position(|workspace| {
                fs::canonicalize(&workspace.path)
                    .map(|candidate| candidate == canonical)
                    .unwrap_or(false)
            })
        {
            self.ws_expanded[existing_index] = true;
            self.rebuild_workspace_rows();
            self.ws_sel = self
                .ws_flat
                .iter()
                .position(|row| matches!(row, WorkspaceRow::Workspace(index) if *index == existing_index))
                .unwrap_or(0);
            self.status = "Workspace already exists.".to_string();
            return Ok(());
        }

        let name = canonical
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| canonical_str.clone());
        self.workspace_config.workspaces.push(app_lib::runtime::Workspace {
            id: generate_id("workspace"),
            name,
            path: canonical_str,
            threads: vec![],
        });
        self.ws_expanded.push(true);
        self.persist_workspace_config("Workspace added.");
        self.rebuild_workspace_rows();
        self.ws_sel = self.ws_flat.len().saturating_sub(1);
        Ok(())
    }

    fn create_thread_for_workspace(&mut self, workspace_index: usize) -> Result<(), String> {
        let Some(workspace) = self.workspace_config.workspaces.get_mut(workspace_index) else {
            return Err("Workspace not found.".to_string());
        };
        let workspace_id = workspace.id.clone();
        let thread_id = generate_id("thread");
        let updated_at = chrono::Utc::now().to_rfc3339();
        let thread_name = next_thread_name(&workspace.threads);
        workspace.threads.push(app_lib::runtime::WorkspaceThread {
            id: thread_id.clone(),
            name: thread_name.clone(),
            messages: String::new(),
            message_items: vec![],
            memory_entries: vec![],
            command_history: vec![],
        });
        self.thread_updated_at.insert(
            thread_recency_key(&workspace_id, &thread_id),
            updated_at.clone(),
        );
        let row_thread_id = thread_id.clone();
        self.runtime.save_thread_snapshot(
            &workspace_id,
            &thread_id,
            ThreadSnapshot {
                updated_at,
                ..ThreadSnapshot::default()
            },
        )?;
        self.ws_expanded[workspace_index] = true;
        self.persist_workspace_config("Thread created.");
        self.rebuild_workspace_rows();
        self.ws_sel = self
            .ws_flat
            .iter()
            .position(|row| {
                matches!(
                    row,
                    WorkspaceRow::Thread {
                        workspace_index: row_workspace_index,
                        thread_index
                    } if *row_workspace_index == workspace_index
                        && self.workspace_config.workspaces[workspace_index].threads[*thread_index].id == row_thread_id
                )
            })
            .unwrap_or(0);
        self.load_thread(workspace_id, thread_id);
        self.status = format!("Thread '{}' created.", thread_name);
        Ok(())
    }

    fn persist_thread_snapshot(&mut self) {
        let Some(workspace_id) = self.active_workspace_id.clone() else {
            return;
        };
        let Some(thread_id) = self.active_thread_id.clone() else {
            return;
        };
        let updated_at = chrono::Utc::now().to_rfc3339();
        self.thread_updated_at.insert(
            thread_recency_key(&workspace_id, &thread_id),
            updated_at.clone(),
        );
        let snapshot = ThreadSnapshot {
            message_items: self.messages.clone(),
            memory_entries: self.runtime.get_memory_pool().entries,
            command_history: self.runtime.get_command_history().entries,
            run_dossier: self.current_run_dossier.clone(),
            active_run_id: self.active_run.as_ref().map(|run| run.run_id.clone()),
            updated_at,
        };
        let _ = self
            .runtime
            .queue_thread_snapshot_save(workspace_id, thread_id, snapshot);
    }

    fn agent_routes_to_user(&self, from_agent_id: &str, message: &str) -> bool {
        let lower = message.to_lowercase();
        self.config.connections.iter().any(|rule| {
            rule.from == from_agent_id
                && rule.enabled
                && rule
                    .condition
                    .as_deref()
                    .map(|condition| lower.contains(&condition.to_lowercase()))
                    .unwrap_or(true)
                && self
                    .config
                    .agents
                    .iter()
                    .any(|agent| agent.id == rule.to && agent.agent_type == "user")
        })
    }

    fn push_bubble(&mut self, agent_id: &str, agent_name: &str, for_user: bool, internal: bool) {
        self.messages
            .push(create_bubble_message(agent_id, agent_name, for_user, internal));
        let idx = self.messages.len().saturating_sub(1);
        self.active_message_by_agent
            .insert(agent_id.to_string(), idx);
        self.chat_message_sel = idx;
        self.chat_section_sel = 0;
        self.chat_detail_scroll = 0;
    }

    fn append_token(&mut self, agent_id: &str, token: &str) {
        if let Some(idx) = self.active_message_by_agent.get(agent_id).copied() {
            if let Some(message) = self.messages.get_mut(idx) {
                message.content.push_str(token);
            }
        } else if let Some(message) = self.messages.iter_mut().rev().find(|message| {
            message.kind == "bubble" && message.agent_id.as_deref() == Some(agent_id)
        }) {
            message.content.push_str(token);
        }
    }

    fn active_bubble_mut(&mut self, agent_id: &str) -> Option<&mut WorkspaceThreadMessage> {
        if let Some(idx) = self.active_message_by_agent.get(agent_id).copied() {
            return self.messages.get_mut(idx);
        }
        self.messages.iter_mut().rev().find(|message| {
            message.kind == "bubble" && message.agent_id.as_deref() == Some(agent_id)
        })
    }

    fn clamp_chat_selection(&mut self) {
        self.chat_message_sel = self.chat_message_sel.min(self.messages.len().saturating_sub(1));
        let section_len = self.selected_message_sections().len();
        self.chat_section_sel = self.chat_section_sel.min(section_len.saturating_sub(1));
    }

    fn selected_message_sections(&self) -> Vec<ChatSection> {
        let Some(message) = self.messages.get(self.chat_message_sel) else {
            return Vec::new();
        };
        let mut sections = Vec::new();
        if !message.content.trim().is_empty() {
            sections.push(ChatSection::Content);
        }
        for index in 0..message.tools.len() {
            sections.push(ChatSection::Tool(index));
        }
        if sections.is_empty() {
            sections.push(ChatSection::Content);
        }
        sections
    }

    fn expand_selected_tool_section(&mut self) {
        let sections = self.selected_message_sections();
        let Some(section) = sections.get(self.chat_section_sel) else {
            return;
        };
        let ChatSection::Tool(tool_index) = section else {
            return;
        };
        let key = format!("{}:{}", self.chat_message_sel, tool_index);
        self.expanded_tool_sections.insert(key);
    }

    fn apply_stream_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::AgentStart {
                run_id,
                agent_id,
                agent_name,
                model_key,
                is_manager,
                ..
            } => {
                self.active_run = Some(ActiveRunUi {
                    run_id,
                    waiting_confirmation: false,
                    limit_message: String::new(),
                });
                if is_manager || self.active_model.is_none() {
                    self.active_model = Some(model_key);
                }
                self.backend_detail = format!("{} started", agent_name);
                self.push_bubble(
                    &agent_id,
                    &agent_name,
                    !is_manager && self.agent_routes_to_user(&agent_id, ""),
                    is_manager,
                );
                self.chat_focus = ChatFocus::Messages;
                self.chat_detail_scroll = 0;
            }
            StreamEvent::AgentStatus { detail, .. } => {
                self.backend_detail = detail;
            }
            StreamEvent::Token { agent_id, content, .. } => {
                self.append_token(&agent_id, &content);
            }
            StreamEvent::ToolCall {
                agent_id,
                tool_name,
                args,
                ..
            } => {
                if let Some(message) = self.active_bubble_mut(&agent_id) {
                    message.tools.push(WorkspaceToolCall {
                        tool_name,
                        args,
                        result: String::new(),
                        status: "pending".to_string(),
                        semantic: false,
                    });
                }
            }
            StreamEvent::ToolResult {
                agent_id,
                result,
                ..
            } => {
                if let Some(message) = self.active_bubble_mut(&agent_id) {
                    if let Some(tool) = message.tools.iter_mut().rev().find(|tool| tool.status == "pending")
                    {
                        tool.result = result.clone();
                        tool.status = if result.starts_with("[tool call failed") {
                            "error".to_string()
                        } else {
                            "done".to_string()
                        };
                    }
                }
            }
            StreamEvent::RunDossierUpdated { dossier, .. } => {
                self.current_run_dossier = Some(dossier);
            }
            StreamEvent::FinalizerOutput {
                run_id,
                agent_id,
                agent_name,
                content,
                ..
            } => {
                self.active_run = Some(ActiveRunUi {
                    run_id,
                    waiting_confirmation: false,
                    limit_message: String::new(),
                });
                let mut message = create_bubble_message(&agent_id, &agent_name, true, false);
                message.content = content;
                self.messages.push(message);
                self.chat_message_sel = self.messages.len().saturating_sub(1);
                self.chat_section_sel = 0;
                self.chat_detail_scroll = 0;
            }
            StreamEvent::Error {
                run_id,
                agent_id,
                agent_name,
                message,
            } => {
                self.active_run = Some(ActiveRunUi {
                    run_id,
                    waiting_confirmation: false,
                    limit_message: String::new(),
                });
                if let Some(existing) = self.active_bubble_mut(&agent_id) {
                    existing.content = message.clone();
                    existing.is_error = true;
                } else {
                    let mut bubble = create_bubble_message(
                        &agent_id,
                        &agent_name,
                        self.agent_routes_to_user(&agent_id, ""),
                        false,
                    );
                    bubble.content = message.clone();
                    bubble.is_error = true;
                    self.messages.push(bubble);
                    self.chat_message_sel = self.messages.len().saturating_sub(1);
                    self.chat_section_sel = 0;
                    self.chat_detail_scroll = 0;
                }
                self.backend_detail = message;
            }
            StreamEvent::RunLimitReached {
                run_id,
                kind,
                limit,
                observed,
            } => {
                self.active_run = Some(ActiveRunUi {
                    run_id,
                    waiting_confirmation: true,
                    limit_message: format!("{}: {} / {}", kind, observed, limit),
                });
            }
            StreamEvent::RunWaitingConfirmation { run_id, message } => {
                self.active_run = Some(ActiveRunUi {
                    run_id,
                    waiting_confirmation: true,
                    limit_message: message,
                });
            }
            StreamEvent::RunResumed { run_id, message } => {
                self.active_run = Some(ActiveRunUi {
                    run_id,
                    waiting_confirmation: false,
                    limit_message: message,
                });
            }
            StreamEvent::RunCancelled { message, .. } => {
                self.active_run = None;
                self.run_usage = None;
                self.active_model = None;
                self.backend_detail = message.clone();
                self.status = message;
            }
            StreamEvent::RunUsageUpdate {
                llm_calls,
                tool_calls,
                spawned_agents,
                streamed_tokens,
                embedding_calls,
                wall_clock_seconds,
                ..
            } => {
                self.run_usage = Some(RunUsageUi {
                    llm_calls,
                    tool_calls,
                    spawned_agents,
                    streamed_tokens,
                    embedding_calls,
                    wall_clock_seconds,
                });
            }
            StreamEvent::Done { .. } => {
                self.active_run = None;
                self.run_usage = None;
                self.active_model = None;
                self.backend_detail = "Run complete.".to_string();
            }
            StreamEvent::AgentEnd { agent_id, .. } => {
                self.active_message_by_agent.remove(&agent_id);
            }
            StreamEvent::AgentMetrics { .. } | StreamEvent::RunCheckpointSaved { .. } => {}
        }
        self.clamp_chat_selection();
        self.persist_thread_snapshot();
    }

    fn persist_config(&mut self, notice: &str) {
        match self.runtime.save_agent_config(self.config.clone()) {
            Ok(()) => match self.runtime.load_agent_config() {
                Ok(config) => {
                    self.config = config;
                    self.status = notice.to_string();
                }
                Err(error) => {
                    self.status = format!("Saved, but reload failed: {}", error);
                }
            },
            Err(error) => {
                self.status = format!("Save failed: {}", error);
            }
        }
        self.agent_sel = self
            .agent_sel
            .min(self.config.agents.len());
        self.agent_row_sel = 0;
        self.routing_row_sel = 0;
        self.settings_row_sel = 0;
    }

    fn persist_workspace_config(&mut self, notice: &str) {
        match self.runtime.save_workspace_config(self.workspace_config.clone()) {
            Ok(()) => {
                self.status = notice.to_string();
                self.rebuild_workspace_rows();
            }
            Err(error) => {
                self.status = format!("Workspace save failed: {}", error);
            }
        }
    }

    fn selected_agent_target(&self) -> AgentTarget {
        if self.agent_sel >= self.config.agents.len() {
            AgentTarget::GlobalPolicy
        } else {
            AgentTarget::Agent(self.agent_sel)
        }
    }

    fn agent_sidebar_len(&self) -> usize {
        self.config.agents.len() + 1
    }

    fn current_rows(&self) -> Vec<EditorRow> {
        match self.panel {
            Panel::Agents => build_agent_rows(self),
            Panel::Routing => build_routing_rows(self),
            Panel::Settings => build_settings_rows(self),
            _ => Vec::new(),
        }
    }

    fn current_row_selection(&self) -> usize {
        match self.panel {
            Panel::Agents => self.agent_row_sel,
            Panel::Routing => self.routing_row_sel,
            Panel::Settings => self.settings_row_sel,
            _ => 0,
        }
    }

    fn set_current_row_selection(&mut self, selection: usize) {
        match self.panel {
            Panel::Agents => self.agent_row_sel = selection,
            Panel::Routing => self.routing_row_sel = selection,
            Panel::Settings => self.settings_row_sel = selection,
            _ => {}
        }
    }

    fn move_rows(&mut self, down: bool) {
        let rows = self.current_rows();
        if rows.is_empty() {
            return;
        }
        let mut idx = self.current_row_selection();
        loop {
            idx = if down {
                (idx + 1) % rows.len()
            } else {
                (idx + rows.len() - 1) % rows.len()
            };
            if !matches!(rows[idx].kind, EditorKind::Section) || idx == self.current_row_selection() {
                break;
            }
        }
        self.set_current_row_selection(idx);
    }

    async fn apply_action(&mut self, key: &FieldKey, tx: &mpsc::UnboundedSender<TuiEvent>) {
        match key {
            FieldKey::AgentLoadNow(index) => {
                let Some(agent) = self.config.agents.get(*index).cloned() else {
                    return;
                };
                let Some(model_key) = agent.model_key.clone() else {
                    self.status = "Select a model first.".to_string();
                    return;
                };
                let load = LoadConfig {
                    model: model_key,
                    context_length: agent.load_config.as_ref().and_then(|cfg| cfg.context_length),
                    eval_batch_size: agent.load_config.as_ref().and_then(|cfg| cfg.eval_batch_size),
                    flash_attention: agent.load_config.as_ref().and_then(|cfg| cfg.flash_attention),
                    num_experts: agent.load_config.as_ref().and_then(|cfg| cfg.num_experts),
                    offload_kv_cache_to_gpu: agent
                        .load_config
                        .as_ref()
                        .and_then(|cfg| cfg.offload_kv_cache_to_gpu),
                };
                match self.runtime.load_model(load).await {
                    Ok(()) => {
                        self.status = format!("Loaded model for {}", agent.name);
                        refresh_models(self.runtime.clone(), tx.clone());
                    }
                    Err(error) => {
                        self.status = format!("Load failed: {}", error);
                    }
                }
            }
            FieldKey::AgentDelete(index) => {
                if *index >= self.config.agents.len() {
                    return;
                }
                if self.config.agents[*index].armed {
                    self.status = "Disarm the agent before deleting it.".to_string();
                    return;
                }
                let agent_id = self.config.agents[*index].id.clone();
                self.config.agents.remove(*index);
                self.config
                    .connections
                    .retain(|rule| rule.from != agent_id && rule.to != agent_id);
                self.persist_config("Agent deleted.");
                self.agent_sel = self.agent_sel.min(self.agent_sidebar_len().saturating_sub(1));
            }
            FieldKey::BackendDiscover => {
                self.status = "Scanning for running backend instances…".to_string();
                let found = self.runtime.discover_instances().await;
                let mut added = 0usize;
                for inst in found {
                    let already = self
                        .config
                        .backend
                        .extra_instances
                        .iter()
                        .any(|e| e.url == inst.url);
                    if !already && inst.url != self.config.backend.base_url {
                        self.config.backend.extra_instances.push(inst);
                        added += 1;
                    }
                }
                if added > 0 {
                    self.persist_config(&format!("Added {} new instance(s).", added));
                } else {
                    self.status = "No new instances found.".to_string();
                }
            }
            FieldKey::BackendInstanceAdd => {
                self.config
                    .backend
                    .extra_instances
                    .push(BackendInstance::default());
                self.persist_config("Instance added.");
            }
            FieldKey::BackendInstanceDelete(i) => {
                if *i < self.config.backend.extra_instances.len() {
                    self.config.backend.extra_instances.remove(*i);
                    self.persist_config("Instance removed.");
                }
            }
            FieldKey::BackendDetect => {
                let base_url = self.config.backend.base_url.clone();
                let backend_type = self.config.backend.backend_type.clone();
                self.status = "Detecting backend capabilities…".to_string();
                let detected: BackendDetected =
                    self.runtime.detect_backend(base_url, backend_type).await;
                if detected.ok {
                    self.config.backend.detected_version = detected.version.clone();
                    self.config.backend.detected_model = detected.model.clone();
                    self.config.backend.detected_parallel_slots = detected.parallel_slots;
                    self.config.backend.detected_tool_use_mode = detected.tool_use_mode.clone();
                    self.config.backend.detected_tool_use_notes = detected.tool_use_notes.clone();
                    self.config.backend.detected_features = detected.features.clone();
                    let manager_warning = self.config.agents.iter().any(|agent| agent.is_manager)
                        && detected.tool_use_mode != "native";
                    let summary = if detected.features.is_empty() && detected.tool_use_notes.is_empty() {
                        "Connected.".to_string()
                    } else {
                        let mut parts = Vec::new();
                        if !detected.features.is_empty() {
                            parts.push(detected.features.join(", "));
                        }
                        parts.push(format!("tool use: {}", detected.tool_use_mode));
                        if !detected.tool_use_notes.is_empty() {
                            parts.push(detected.tool_use_notes.join(", "));
                        }
                        if manager_warning {
                            parts.push(
                                "warning: manager agents may stall or require recovery on this backend"
                                    .to_string(),
                            );
                        }
                        format!("Connected — {}", parts.join(" | "))
                    };
                    self.persist_config(&summary);
                } else {
                    let err = detected.error.unwrap_or_else(|| "Unknown error".to_string());
                    self.status = format!("Detection failed: {}", err);
                }
            }
            _ => {}
        }
    }

    fn apply_field_value(&mut self, key: &FieldKey, value: String) -> Result<String, String> {
        match key {
            FieldKey::BackendType => {
                let url = match value.as_str() {
                    "lm_studio" => "http://localhost:1234",
                    "ollama" => "http://localhost:11434",
                    "llamacpp" => "http://localhost:8080",
                    _ => &self.config.backend.base_url,
                };
                self.config.backend.backend_type = value;
                self.config.backend.base_url = url.to_string();
                self.config.backend.detected_version = None;
                self.config.backend.detected_parallel_slots = None;
                self.config.backend.detected_features.clear();
                self.persist_config("Backend saved.");
            }
            FieldKey::BackendUrl => {
                self.config.backend.base_url = value;
                self.persist_config("Backend saved.");
            }
            FieldKey::BackendApiKey => {
                self.config.backend.api_key = if value.is_empty() { None } else { Some(value) };
                self.persist_config("Backend saved.");
            }
            FieldKey::BackendDetectedInfo => {}
            FieldKey::BackendInstanceUrl(i) => {
                if let Some(inst) = self.config.backend.extra_instances.get_mut(*i) {
                    inst.url = value;
                    self.persist_config("Instance saved.");
                }
            }
            FieldKey::BackendInstanceModelHint(i) => {
                if let Some(inst) = self.config.backend.extra_instances.get_mut(*i) {
                    inst.model_hint = value;
                    self.persist_config("Instance saved.");
                }
            }
            FieldKey::AgentName(index) => {
                self.config.agents[*index].name = value;
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentManager(index) => {
                self.config.agents[*index].is_manager = parse_bool(&value)?;
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentModel(index) => {
                self.config.agents[*index].model_key = if value.is_empty() {
                    None
                } else {
                    Some(value.clone())
                };
                self.config.agents[*index].model_type = if value.is_empty() {
                    None
                } else {
                    Some(
                        self.available_models
                            .iter()
                            .find(|model| model.key == value)
                            .and_then(|model| model.model_type.clone())
                            .unwrap_or_else(|| "llm".to_string()),
                    )
                };
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentMode(index) => {
                self.config.agents[*index].mode = if value.is_empty() {
                    None
                } else {
                    Some(value)
                };
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentRole(index) => {
                self.config.agents[*index].role = Some(value);
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentContextLength(index) => {
                let parsed = parse_optional_u64(&value)?;
                ensure_load_config(&mut self.config, *index).context_length = parsed;
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentEvalBatch(index) => {
                let parsed = parse_optional_u64(&value)?;
                ensure_load_config(&mut self.config, *index).eval_batch_size = parsed;
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentExperts(index) => {
                let parsed = parse_optional_u64(&value)?;
                ensure_load_config(&mut self.config, *index).num_experts = parsed;
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentFlashAttention(index) => {
                let parsed = parse_optional_bool(&value)?;
                ensure_load_config(&mut self.config, *index).flash_attention = parsed;
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentGpuKvOffload(index) => {
                let parsed = parse_optional_bool(&value)?;
                ensure_load_config(&mut self.config, *index).offload_kv_cache_to_gpu = parsed;
                self.persist_config("Agent saved.");
            }
            FieldKey::AgentArmed(index) => {
                self.config.agents[*index].armed = parse_bool(&value)?;
                self.persist_config("Agent saved.");
            }
            FieldKey::CommandDenylist => {
                self.config.command_policy.denylist = split_lines(&value);
                self.persist_config("Policy saved.");
            }
            FieldKey::CommandAllowlist => {
                self.config.command_policy.allowlist = split_lines(&value);
                self.persist_config("Policy saved.");
            }
            FieldKey::RedundancyEnabled => {
                self.config.redundancy_detection.enabled = parse_bool(&value)?;
                self.persist_config("Redundancy saved.");
            }
            FieldKey::RedundancyEmbeddingModel => {
                self.config.redundancy_detection.embedding_model_key =
                    if value.is_empty() { None } else { Some(value) };
                self.persist_config("Redundancy saved.");
            }
            FieldKey::RedundancyThreshold => {
                self.config.redundancy_detection.semantic_similarity_threshold =
                    parse_f32(&value)?;
                self.persist_config("Redundancy saved.");
            }
            FieldKey::RedundancyRetries => {
                self.config.redundancy_detection.max_redundant_audit_retries =
                    parse_u8(&value)?;
                self.persist_config("Redundancy saved.");
            }
            FieldKey::RunBudgetsEnabled => {
                self.config.run_budgets.enabled = parse_bool(&value)?;
                self.persist_config("Budgets saved.");
            }
            FieldKey::RunBudgetLlmCalls => {
                self.config.run_budgets.llm_calls_per_window = parse_u32(&value)?;
                self.persist_config("Budgets saved.");
            }
            FieldKey::RunBudgetToolCalls => {
                self.config.run_budgets.tool_calls_per_window = parse_u32(&value)?;
                self.persist_config("Budgets saved.");
            }
            FieldKey::RunBudgetSpawnedAgents => {
                self.config.run_budgets.spawned_agents_per_window = parse_u32(&value)?;
                self.persist_config("Budgets saved.");
            }
            FieldKey::RunBudgetTokens => {
                self.config.run_budgets.streamed_tokens_per_window = parse_u64(&value)?;
                self.persist_config("Budgets saved.");
            }
            FieldKey::RunBudgetWallClock => {
                self.config.run_budgets.wall_clock_seconds_per_window = parse_u64(&value)?;
                self.persist_config("Budgets saved.");
            }
            FieldKey::RunBudgetAppliesToBehaviors => {
                self.config.run_budgets.applies_to_behaviors = split_lines(&value);
                self.persist_config("Budgets saved.");
            }
            FieldKey::RunBudgetOnLimit => {
                self.config.run_budgets.on_limit = value;
                self.persist_config("Budgets saved.");
            }
            FieldKey::FinalizerEnabled => {
                self.config.finalizer.enabled = parse_bool(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerRunOnCompletion => {
                self.config.finalizer.run_on_completion = parse_bool(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerRunOnBudgetStop => {
                self.config.finalizer.run_on_budget_stop = parse_bool(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerAgentName => {
                self.config.finalizer.agent_name = value;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerModel => {
                self.config.finalizer.model_key = if value.is_empty() {
                    None
                } else {
                    Some(value)
                };
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerTranscriptChars => {
                self.config.finalizer.max_transcript_chars = parse_usize(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerIncludeDossier => {
                self.config.finalizer.include_run_dossier = parse_bool(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerIncludeCommands => {
                self.config.finalizer.include_command_history = parse_bool(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerIncludeWorkers => {
                self.config.finalizer.include_worker_outputs = parse_bool(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerIncludeTranscript => {
                self.config.finalizer.include_internal_transcript = parse_bool(&value)?;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerPromptCompletion => {
                self.config.finalizer.prompt_completion = value;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::FinalizerPromptBudgetStop => {
                self.config.finalizer.prompt_budget_stop = value;
                self.persist_config("Finalizer saved.");
            }
            FieldKey::BehaviorTriggersEnabled => {
                self.config.behavior_triggers.enabled = parse_bool(&value)?;
                self.persist_config("Behavior triggers saved.");
            }
            FieldKey::BehaviorTriggersEmbeddingModel => {
                self.config.behavior_triggers.embedding_model_key =
                    if value.is_empty() { None } else { Some(value) };
                self.persist_config("Behavior triggers saved.");
            }
            FieldKey::BehaviorTriggersDefaultThreshold => {
                self.config.behavior_triggers.default_similarity_threshold = parse_f32(&value)?;
                self.persist_config("Behavior triggers saved.");
            }
            FieldKey::BehaviorKeywords(index) => {
                self.config.behavior_triggers.behaviors[*index].keyword_triggers = split_lines(&value);
                self.persist_config("Behavior saved.");
            }
            FieldKey::BehaviorJson(index) => {
                let parsed: BehaviorTriggerConfig =
                    serde_json::from_str(&value).map_err(|error| error.to_string())?;
                self.config.behavior_triggers.behaviors[*index] = parsed;
                self.persist_config("Behavior saved.");
            }
            FieldKey::ParallelInferenceEnabled => {
                self.config.parallel_inference.enabled = parse_bool(&value)?;
                self.persist_config("Parallel inference saved.");
            }
            FieldKey::ParallelInferenceMaxAgents => {
                let v = parse_u32(&value)?;
                self.config.parallel_inference.max_parallel_agents = v.max(1);
                self.persist_config("Parallel inference saved.");
            }
            FieldKey::RouteEnabled(from, to) => {
                if parse_bool(&value)? {
                    let rule = ensure_route_rule(&mut self.config.connections, from, to);
                    rule.enabled = true;
                } else {
                    self.config
                        .connections
                        .retain(|rule| !(rule.from == *from && rule.to == *to));
                }
                self.persist_config("Routing saved.");
            }
            FieldKey::RoutePriority(from, to) => {
                let rule = ensure_route_rule(&mut self.config.connections, from, to);
                rule.priority = parse_u8(&value)?;
                self.persist_config("Routing saved.");
            }
            FieldKey::RouteCondition(from, to) => {
                let trimmed = value.trim().to_string();
                let rule = ensure_route_rule(&mut self.config.connections, from, to);
                rule.condition = if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                };
                self.persist_config("Routing saved.");
            }
            FieldKey::AgentLoadNow(_)
            | FieldKey::AgentDelete(_)
            | FieldKey::BackendDetect
            | FieldKey::BackendDiscover
            | FieldKey::BackendInstanceAdd
            | FieldKey::BackendInstanceDelete(_) => {}
        }
        Ok("Saved.".to_string())
    }
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
) -> io::Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<TuiEvent>();
    let tx_key = tx.clone();
    std::thread::spawn(move || loop {
        if event::poll(Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if tx_key.send(TuiEvent::Key(key)).is_err() {
                    break;
                }
            }
        }
    });

    loop {
        terminal.draw(|frame| ui(frame, app))?;
        match tokio::time::timeout(Duration::from_millis(50), rx.recv()).await {
            Ok(Some(TuiEvent::Key(key))) => {
                if handle_key(app, key, &tx).await {
                    return Ok(());
                }
            }
            Ok(Some(TuiEvent::Stream(event))) => app.apply_stream_event(event),
            Ok(Some(TuiEvent::Notice(message))) => app.status = message,
            Ok(Some(TuiEvent::ModelsLoaded(result))) => match result {
                Ok(models) => app.available_models = models,
                Err(error) => app.status = format!("Model refresh failed: {}", error),
            },
            _ => {}
        }
    }
}

async fn handle_key(
    app: &mut App,
    key: event::KeyEvent,
    tx: &mpsc::UnboundedSender<TuiEvent>,
) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return true;
    }
    if app.edit_state.is_some() {
        return handle_edit_key(app, key).await;
    }

    if app.mode == Mode::Insert {
        return handle_chat_insert_key(app, key, tx).await;
    }

    match key.code {
        KeyCode::Char('q') => return true,
        KeyCode::Char('1') => app.panel = Panel::Chat,
        KeyCode::Char('2') => app.panel = Panel::Agents,
        KeyCode::Char('3') => app.panel = Panel::Routing,
        KeyCode::Char('4') => {
            app.rebuild_workspace_rows();
            app.panel = Panel::Workspaces;
        }
        KeyCode::Char('5') => app.panel = Panel::Settings,
        KeyCode::Tab => {
            app.panel = match app.panel {
                Panel::Chat => Panel::Agents,
                Panel::Agents => Panel::Routing,
                Panel::Routing => Panel::Workspaces,
                Panel::Workspaces => Panel::Settings,
                Panel::Settings => Panel::Chat,
            };
            if app.panel == Panel::Workspaces {
                app.rebuild_workspace_rows();
            }
        }
        _ => match app.panel {
            Panel::Chat => handle_chat_key(app, key, tx).await,
            Panel::Agents => handle_agents_key(app, key, tx).await,
            Panel::Routing => handle_editor_panel_key(app, key, tx).await,
            Panel::Workspaces => handle_workspace_key(app, key, tx).await,
            Panel::Settings => handle_editor_panel_key(app, key, tx).await,
        },
    }
    false
}

async fn handle_chat_key(
    app: &mut App,
    key: event::KeyEvent,
    tx: &mpsc::UnboundedSender<TuiEvent>,
) {
    match key.code {
        KeyCode::Char('i') if app.chat_focus == ChatFocus::Messages => app.mode = Mode::Insert,
        KeyCode::Char('j') | KeyCode::Down => match app.chat_focus {
            ChatFocus::Messages => {
                if !app.messages.is_empty() {
                    app.chat_message_sel =
                        (app.chat_message_sel + 1).min(app.messages.len().saturating_sub(1));
                    app.chat_section_sel = 0;
                    app.chat_detail_scroll = 0;
                }
            }
            ChatFocus::Sections => {
                let len = app.selected_message_sections().len();
                if len > 0 {
                    app.chat_section_sel = (app.chat_section_sel + 1).min(len.saturating_sub(1));
                    app.chat_detail_scroll = 0;
                }
            }
            ChatFocus::Detail => {
                app.chat_detail_scroll = app.chat_detail_scroll.saturating_add(1);
            }
        },
        KeyCode::Char('k') | KeyCode::Up => match app.chat_focus {
            ChatFocus::Messages => {
                app.chat_message_sel = app.chat_message_sel.saturating_sub(1);
                app.chat_section_sel = 0;
                app.chat_detail_scroll = 0;
            }
            ChatFocus::Sections => {
                app.chat_section_sel = app.chat_section_sel.saturating_sub(1);
                app.chat_detail_scroll = 0;
            }
            ChatFocus::Detail => {
                app.chat_detail_scroll = app.chat_detail_scroll.saturating_sub(1);
            }
        },
        KeyCode::Enter => match app.chat_focus {
            ChatFocus::Messages => {
                if !app.messages.is_empty() {
                    app.chat_focus = ChatFocus::Sections;
                    app.chat_section_sel = 0;
                    app.chat_detail_scroll = 0;
                }
            }
            ChatFocus::Sections => {
                app.expand_selected_tool_section();
                app.chat_focus = ChatFocus::Detail;
                app.chat_detail_scroll = 0;
            }
            ChatFocus::Detail => {}
        },
        KeyCode::Esc => {
            match app.chat_focus {
                ChatFocus::Detail => app.chat_focus = ChatFocus::Sections,
                ChatFocus::Sections => app.chat_focus = ChatFocus::Messages,
                ChatFocus::Messages => {
                    if let Some(run) = app.active_run.as_ref() {
                        let runtime = app.runtime.clone();
                        let run_id = run.run_id.clone();
                        let tx_clone = tx.clone();
                        app.status = "Stopping after the current turn...".to_string();
                        tokio::spawn(async move {
                            if let Err(error) = runtime.cancel_route_run(run_id).await {
                                let _ = tx_clone.send(TuiEvent::Notice(format!("Cancel failed: {}", error)));
                            }
                        });
                    }
                }
            }
        }
        KeyCode::Left | KeyCode::Char('h') => match app.chat_focus {
            ChatFocus::Detail => app.chat_focus = ChatFocus::Sections,
            ChatFocus::Sections => app.chat_focus = ChatFocus::Messages,
            ChatFocus::Messages => {}
        },
        KeyCode::Char('c') if app.active_run.as_ref().map(|run| run.waiting_confirmation).unwrap_or(false) => {
            if let Some(run) = app.active_run.as_ref() {
                let runtime = app.runtime.clone();
                let run_id = run.run_id.clone();
                let tx_clone = tx.clone();
                tokio::spawn(async move {
                    if let Err(error) = runtime.continue_route_run(run_id).await {
                        let _ = tx_clone.send(TuiEvent::Notice(format!("Resume failed: {}", error)));
                    }
                });
            }
        }
        KeyCode::Char('x') if app.active_run.is_some() => {
            if let Some(run) = app.active_run.as_ref() {
                let runtime = app.runtime.clone();
                let run_id = run.run_id.clone();
                let tx_clone = tx.clone();
                tokio::spawn(async move {
                    if let Err(error) = runtime.cancel_route_run(run_id).await {
                        let _ = tx_clone.send(TuiEvent::Notice(format!("Cancel failed: {}", error)));
                    }
                });
            }
        }
        _ => {}
    }
}

async fn handle_chat_insert_key(
    app: &mut App,
    key: event::KeyEvent,
    tx: &mpsc::UnboundedSender<TuiEvent>,
) -> bool {
    match key.code {
        KeyCode::Esc => app.mode = Mode::Normal,
        KeyCode::Enter => submit_message(app, tx).await,
        _ => edit_buffer(&mut app.chat_input, &mut app.chat_cursor, &key, false),
    }
    false
}

async fn submit_message(app: &mut App, tx: &mpsc::UnboundedSender<TuiEvent>) {
    let message = app.chat_input.trim().to_string();
    if message.is_empty() {
        return;
    }
    let workspace_id = match ensure_active_workspace(app) {
        Ok(workspace_id) => workspace_id,
        Err(error) => {
            app.status = error;
            return;
        }
    };
    let thread_id = match ensure_active_thread(app, &workspace_id, &message) {
        Ok(thread_id) => thread_id,
        Err(error) => {
            app.status = error;
            return;
        }
    };
    if let Err(error) = app.ensure_runtime_workspace(&workspace_id) {
        app.status = error;
        return;
    }

    app.messages.push(create_text_message("You", &message, true, false));
    app.chat_message_sel = app.messages.len().saturating_sub(1);
    app.chat_focus = ChatFocus::Messages;
    app.chat_section_sel = 0;
    app.chat_detail_scroll = 0;
    app.chat_input.clear();
    app.chat_cursor = 0;
    app.mode = Mode::Normal;
    app.persist_thread_snapshot();

    let runtime = app.runtime.clone();
    let tx_clone = tx.clone();
    let tx_notice = tx.clone();
    let workspace_id_clone = workspace_id.clone();
    let thread_id_clone = thread_id.clone();
    tokio::spawn(async move {
        let sink = callback_event_sink(move |event| {
            tx_clone
                .send(TuiEvent::Stream(event))
                .map_err(|error| error.to_string())
        });
        if let Err(error) = runtime
            .route_message(
                "user".to_string(),
                message,
                Some(workspace_id_clone),
                Some(thread_id_clone),
                sink,
            )
            .await
        {
            let _ = tx_notice.send(TuiEvent::Notice(format!("Route failed: {}", error)));
        }
    });
}

fn ensure_active_workspace(app: &mut App) -> Result<String, String> {
    if let Some(workspace_id) = app.active_workspace_id.clone() {
        app.ensure_runtime_workspace(&workspace_id)?;
        return Ok(workspace_id);
    }
    let workspace = app
        .workspace_config
        .workspaces
        .first()
        .ok_or_else(|| "Add a workspace first.".to_string())?;
    let workspace_id = workspace.id.clone();
    app.active_workspace_id = Some(workspace_id.clone());
    app.ensure_runtime_workspace(&workspace_id)?;
    Ok(workspace_id)
}

fn ensure_active_thread(app: &mut App, workspace_id: &str, message: &str) -> Result<String, String> {
    if let Some(thread_id) = app.active_thread_id.clone() {
        return Ok(thread_id);
    }
    let workspace_index = app
        .workspace_config
        .workspaces
        .iter()
        .position(|workspace| workspace.id == workspace_id)
        .ok_or_else(|| format!("Workspace '{}' not found.", workspace_id))?;
    app.create_thread_for_workspace(workspace_index)?;
    let thread_id = app
        .active_thread_id
        .clone()
        .ok_or_else(|| format!("Failed to create thread for '{}'.", summarize_thread_name(message)))?;
    Ok(thread_id)
}

async fn handle_agents_key(
    app: &mut App,
    key: event::KeyEvent,
    tx: &mpsc::UnboundedSender<TuiEvent>,
) {
    match app.agent_focus {
        AgentFocus::List => match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                app.agent_sel = (app.agent_sel + 1).min(app.agent_sidebar_len().saturating_sub(1));
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.agent_sel = app.agent_sel.saturating_sub(1);
            }
            KeyCode::Right | KeyCode::Char('l') => app.agent_focus = AgentFocus::Fields,
            _ => {}
        },
        AgentFocus::Fields => match key.code {
            KeyCode::Left | KeyCode::Char('h') => app.agent_focus = AgentFocus::List,
            _ => handle_editor_panel_key(app, key, tx).await,
        },
    }
}

async fn handle_editor_panel_key(
    app: &mut App,
    key: event::KeyEvent,
    tx: &mpsc::UnboundedSender<TuiEvent>,
) {
    match key.code {
        KeyCode::Down | KeyCode::Char('j') => app.move_rows(true),
        KeyCode::Up | KeyCode::Char('k') => app.move_rows(false),
        KeyCode::Enter | KeyCode::Char('e') => {
            let rows = app.current_rows();
            let Some(row) = rows.get(app.current_row_selection()).cloned() else {
                return;
            };
            let Some(field_key) = row.key.clone() else {
                return;
            };
            match row.kind {
                EditorKind::Section => {}
                EditorKind::Action => app.apply_action(&field_key, tx).await,
                EditorKind::Text { multiline } => {
                    let raw = row.raw.clone();
                    app.edit_state = Some(EditState::Text {
                        key: field_key,
                        buffer: raw.clone(),
                        cursor: raw.chars().count(),
                        multiline,
                    });
                }
                EditorKind::Number => {
                    let raw = row.raw.clone();
                    app.edit_state = Some(EditState::Text {
                        key: field_key,
                        buffer: raw.clone(),
                        cursor: raw.chars().count(),
                        multiline: false,
                    });
                }
                EditorKind::Picker { options } => {
                    let index = options
                        .iter()
                        .position(|option| option == &row.raw)
                        .unwrap_or(0);
                    app.edit_state = Some(EditState::Picker {
                        key: field_key,
                        options,
                        index,
                    });
                }
            }
        }
        _ => {}
    }
}

async fn handle_edit_key(app: &mut App, key: event::KeyEvent) -> bool {
    let Some(mut edit_state) = app.edit_state.take() else {
        return false;
    };
    match edit_state {
        EditState::Text {
            key: ref field,
            ref mut buffer,
            ref mut cursor,
            multiline,
        } => match key.code {
            KeyCode::Esc => {}
            KeyCode::Enter if !multiline => {
                let result = app.apply_field_value(field, buffer.clone());
                if let Err(error) = result {
                    app.status = format!("Save failed: {}", error);
                }
            }
            KeyCode::Char('s') if multiline && key.modifiers.contains(KeyModifiers::CONTROL) => {
                let result = app.apply_field_value(field, buffer.clone());
                if let Err(error) = result {
                    app.status = format!("Save failed: {}", error);
                }
            }
            KeyCode::Enter => {
                insert_char(buffer, cursor, '\n');
                app.edit_state = Some(edit_state);
            }
            _ => {
                edit_buffer(buffer, cursor, &key, multiline);
                app.edit_state = Some(edit_state);
            }
        },
        EditState::Picker {
            key: ref field,
            ref options,
            ref mut index,
        } => match key.code {
            KeyCode::Esc => {}
            KeyCode::Up | KeyCode::Char('k') => {
                if options.is_empty() {
                    return false;
                }
                *index = if *index == 0 {
                    options.len() - 1
                } else {
                    *index - 1
                };
                app.edit_state = Some(edit_state);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if options.is_empty() {
                    return false;
                }
                *index = (*index + 1) % options.len();
                app.edit_state = Some(edit_state);
            }
            KeyCode::Enter => {
                let value = options.get(*index).cloned().unwrap_or_default();
                let result = app.apply_field_value(field, value);
                if let Err(error) = result {
                    app.status = format!("Save failed: {}", error);
                }
            }
            _ => app.edit_state = Some(edit_state),
        },
    }
    false
}

async fn handle_workspace_key(
    app: &mut App,
    key: event::KeyEvent,
    _tx: &mpsc::UnboundedSender<TuiEvent>,
) {
    if app.ws_flat.is_empty() {
        return;
    }
    match key.code {
        KeyCode::Down | KeyCode::Char('j') => {
            app.ws_sel = (app.ws_sel + 1).min(app.ws_flat.len().saturating_sub(1));
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.ws_sel = app.ws_sel.saturating_sub(1);
        }
        KeyCode::Enter | KeyCode::Char(' ') => {
            if let Some(row) = app.ws_flat.get(app.ws_sel).cloned() {
                match row {
                    WorkspaceRow::AddWorkspace => {
                        match app.runtime.pick_folder_blocking() {
                            Ok(Some(path)) => {
                                if let Err(error) = app.add_workspace(path) {
                                    app.status = error;
                                }
                            }
                            Ok(None) => {
                                app.status = "Workspace picker cancelled.".to_string();
                            }
                            Err(error) => {
                                app.status = format!("Workspace picker failed: {}", error);
                            }
                        }
                    }
                    WorkspaceRow::Workspace(workspace_index) => {
                        if let Some(expanded) = app.ws_expanded.get_mut(workspace_index) {
                            *expanded = !*expanded;
                        }
                        app.rebuild_workspace_rows();
                    }
                    WorkspaceRow::NewThread(workspace_index) => {
                        if let Err(error) = app.create_thread_for_workspace(workspace_index) {
                            app.status = error;
                        } else {
                            app.panel = Panel::Chat;
                        }
                    }
                    WorkspaceRow::Thread {
                        workspace_index,
                        thread_index,
                    } => {
                        let workspace = &app.workspace_config.workspaces[workspace_index];
                        let thread = &workspace.threads[thread_index];
                        app.load_thread(workspace.id.clone(), thread.id.clone());
                        app.panel = Panel::Chat;
                    }
                }
            }
        }
        _ => {}
    }
}

fn ui(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    frame.render_widget(
        Tabs::new(vec![" Chat ", " Agents ", " Routing ", " Workspaces ", " Settings "])
            .select(app.panel as usize)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Ajantis TUI "),
            )
            .style(Style::default().fg(Color::DarkGray))
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        chunks[0],
    );

    match app.panel {
        Panel::Chat => render_chat(frame, chunks[1], app),
        Panel::Agents => render_agents(frame, chunks[1], app),
        Panel::Routing => render_editor_panel(
            frame,
            chunks[1],
            " Routing ",
            routing_matrix_text(app),
            &build_routing_rows(app),
            app.routing_row_sel,
            app.edit_state.as_ref(),
            true,
        ),
        Panel::Workspaces => render_workspaces(frame, chunks[1], app),
        Panel::Settings => render_editor_panel(
            frame,
            chunks[1],
            " Settings ",
            format!("Theme changes are stored in {}", CONFIG_PATH_HINT),
            &build_settings_rows(app),
            app.settings_row_sel,
            app.edit_state.as_ref(),
            true,
        ),
    }
}

fn render_chat(frame: &mut Frame, area: Rect, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(3), Constraint::Length(2)])
        .split(area);
    let panes = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(outer[0]);

    let thread_label = active_thread_label(app);
    let items = render_thread_summaries(app);
    let visible_height = panes[0].height.saturating_sub(2) as usize;
    let selected = app.chat_message_sel.min(items.len().saturating_sub(1));
    let start = selected.saturating_sub(visible_height.saturating_sub(1));
    let visible = items.into_iter().skip(start).take(visible_height).collect::<Vec<_>>();
    let mut list_state = ListState::default().with_selected(Some(selected.saturating_sub(start)));
    frame.render_stateful_widget(
        List::new(visible)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(thread_label),
            )
            .highlight_style(selection_style(app.chat_focus, ChatFocus::Messages)),
        panes[0],
        &mut list_state,
    );

    let detail = render_chat_detail(app);
    frame.render_widget(
        Paragraph::new(detail)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(if app.chat_focus == ChatFocus::Detail {
                        " Message Detail  (j/k=scroll text, Esc=back) "
                    } else if app.chat_focus == ChatFocus::Sections {
                        " Message Detail  (j/k=sections, Enter=open section, Esc=back) "
                    } else {
                        " Message Detail  (Enter=sections) "
                    }),
            )
            .scroll((app.chat_detail_scroll.min(u16::MAX as usize) as u16, 0))
            .wrap(Wrap { trim: false }),
        panes[1],
    );

    let input_title = if app.mode == Mode::Insert {
        " Input  (Enter=send, Esc=cancel) "
    } else {
        " Input  (j/k=messages, Enter=sections, i=type, c=continue, x=cancel, Tab=next panel) "
    };
    frame.render_widget(
        Paragraph::new(app.chat_input.as_str())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(input_title),
            )
            .style(if app.mode == Mode::Insert {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            }),
        outer[1],
    );
    if app.mode == Mode::Insert {
        frame.set_cursor_position((outer[1].x + 1 + app.chat_cursor as u16, outer[1].y + 1));
    }

    let model_label = app.active_model.as_deref().map(|m| {
        // Trim to last path component in case it's a file path.
        let name = m.rsplit('/').next().unwrap_or(m);
        // Strip common file extensions.
        let name = name
            .strip_suffix(".gguf")
            .or_else(|| name.strip_suffix(".bin"))
            .unwrap_or(name);
        format!("model: {}", name)
    });
    let usage = app.run_usage.as_ref().map(|usage| {
        let tok_per_sec = if usage.wall_clock_seconds > 0 {
            usage.streamed_tokens / usage.wall_clock_seconds
        } else {
            usage.streamed_tokens
        };
        format!(
            "llm:{} tool:{} spawn:{}  {}/s tok  emb_hits:{}  {}s",
            usage.llm_calls,
            usage.tool_calls,
            usage.spawned_agents,
            tok_per_sec,
            usage.embedding_calls,
            usage.wall_clock_seconds
        )
    });
    let waiting = app.active_run.as_ref().filter(|run| run.waiting_confirmation).map(|run| {
        format!("Waiting: {}", run.limit_message)
    });
    let footer = [Some(app.status.clone()), Some(app.backend_detail.clone()), model_label, usage, waiting]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" | ");
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().fg(Color::DarkGray)),
        outer[2],
    );
}

fn render_agents(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(24), Constraint::Percentage(76)])
        .split(area);

    let sidebar_items = (0..app.agent_sidebar_len())
        .map(|index| {
            if index < app.config.agents.len() {
                let agent = &app.config.agents[index];
                let mut label = agent.name.clone();
                if agent.agent_type == "user" {
                    label.push_str(" [user]");
                } else if agent.is_manager {
                    label.push_str(" [manager]");
                }
                if !agent.armed {
                    label.push_str(" [disarmed]");
                }
                ListItem::new(label)
            } else {
                ListItem::new("Global policy")
            }
        })
        .collect::<Vec<_>>();
    let mut sidebar_state = ListState::default().with_selected(Some(app.agent_sel));
    frame.render_stateful_widget(
        List::new(sidebar_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Targets "),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        chunks[0],
        &mut sidebar_state,
    );

    let heading = match app.selected_agent_target() {
        AgentTarget::Agent(index) => app
            .config
            .agents
            .get(index)
            .map(|agent| format!(" {} ", agent.name))
            .unwrap_or_else(|| " Agent ".to_string()),
        AgentTarget::GlobalPolicy => " Global policy ".to_string(),
    };
    render_editor_panel(
        frame,
        chunks[1],
        &heading,
        "Enter edits. Boolean and enum fields open a picker; multiline fields save with Ctrl+S."
            .to_string(),
        &build_agent_rows(app),
        app.agent_row_sel,
        app.edit_state.as_ref(),
        app.agent_focus == AgentFocus::Fields,
    );
}

fn render_workspaces(frame: &mut Frame, area: Rect, app: &App) {
    let items = app
        .ws_flat
        .iter()
        .map(|row| match row {
            WorkspaceRow::AddWorkspace => ListItem::new(Line::from(vec![
                Span::styled("+ ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw("Add workspace"),
            ])),
            WorkspaceRow::Workspace(workspace_index) => {
                let workspace = &app.workspace_config.workspaces[*workspace_index];
                let expanded = app.ws_expanded.get(*workspace_index).copied().unwrap_or(false);
                ListItem::new(format!(
                    "{} {} ({})",
                    if expanded { "▼" } else { "▶" },
                    workspace.name,
                    workspace.path
                ))
            }
            WorkspaceRow::NewThread(workspace_index) => {
                let workspace = &app.workspace_config.workspaces[*workspace_index];
                ListItem::new(Line::from(vec![
                    Span::raw("    "),
                    Span::styled("+ ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::styled(
                        format!("New thread in {}", workspace.name),
                        Style::default().fg(Color::White),
                    ),
                ]))
            }
            WorkspaceRow::Thread {
                workspace_index,
                thread_index,
            } => {
                let workspace = &app.workspace_config.workspaces[*workspace_index];
                let thread = &workspace.threads[*thread_index];
                let active = app.active_workspace_id.as_deref() == Some(workspace.id.as_str())
                    && app.active_thread_id.as_deref() == Some(thread.id.as_str());
                ListItem::new(format!(
                    "    {} {}",
                    if active { "★" } else { " " },
                    thread.name
                ))
            }
        })
        .collect::<Vec<_>>();
    let mut list_state = ListState::default().with_selected(Some(app.ws_sel));
    frame.render_stateful_widget(
        List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Workspaces  (Enter=open, +=new rows) "),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        area,
        &mut list_state,
    );
}

fn render_editor_panel(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    intro: String,
    rows: &[EditorRow],
    selected: usize,
    edit_state: Option<&EditState>,
    focused: bool,
) {
    let editor_height = if edit_state.is_some() { 7 } else { 3 };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(4), Constraint::Length(editor_height)])
        .split(area);
    let items = rows
        .iter()
        .map(|row| match row.kind {
            EditorKind::Section => ListItem::new(Line::from(Span::styled(
                format!("  ── {} ", row.label),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ))),
            _ => ListItem::new(Line::from(vec![
                Span::styled(
                    format!("  {:<28}", row.label),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw(row.display.clone()),
            ])),
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(title)
                    .border_style(if focused {
                        Style::default().fg(Color::White)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    }),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        chunks[0],
        &mut state,
    );

    let editor_text = match edit_state {
        None => intro,
        Some(EditState::Text { buffer, multiline, .. }) => {
            if *multiline {
                format!("Multiline editor (Ctrl+S save, Esc cancel)\n{}", buffer)
            } else {
                format!("Edit (Enter save, Esc cancel)\n{}", buffer)
            }
        }
        Some(EditState::Picker { options, index, .. }) => format!(
            "Picker (Up/Down cycle, Enter save, Esc cancel)\n{}",
            options
                .iter()
                .enumerate()
                .map(|(idx, option)| {
                    if idx == *index {
                        format!("[{}]", option)
                    } else {
                        option.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join("  ")
        ),
    };
    frame.render_widget(
        Paragraph::new(editor_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .wrap(Wrap { trim: false }),
        chunks[1],
    );

    if let Some(EditState::Text { buffer, cursor, .. }) = edit_state {
        let (line, col) = cursor_line_col(buffer, *cursor);
        frame.set_cursor_position((chunks[1].x + 1 + col as u16, chunks[1].y + 2 + line as u16));
    }
}

fn build_agent_rows(app: &App) -> Vec<EditorRow> {
    match app.selected_agent_target() {
        AgentTarget::Agent(index) => {
            let Some(agent) = app.config.agents.get(index) else {
                return Vec::new();
            };
            if agent.agent_type == "user" {
                vec![
                    section_row("Identity"),
                    text_row("Visible name", agent.name.clone(), FieldKey::AgentName(index), false),
                ]
            } else {
                vec![
                    section_row("Identity"),
                    text_row("Name", agent.name.clone(), FieldKey::AgentName(index), false),
                    picker_row(
                        "Manager",
                        bool_raw(agent.is_manager),
                        FieldKey::AgentManager(index),
                        bool_options(),
                    ),
                    picker_row(
                        "Model",
                        agent.model_key.clone().unwrap_or_default(),
                        FieldKey::AgentModel(index),
                        model_options(&app.available_models, true),
                    ),
                    picker_row(
                        "Mode",
                        agent.mode.clone().unwrap_or_else(|| "stay_awake".to_string()),
                        FieldKey::AgentMode(index),
                        vec!["stay_awake".to_string(), "on_the_fly".to_string()],
                    ),
                    picker_row(
                        "Armed",
                        bool_raw(agent.armed),
                        FieldKey::AgentArmed(index),
                        bool_options(),
                    ),
                    section_row("Prompt"),
                    text_row(
                        "System prompt",
                        agent.role.clone().unwrap_or_default(),
                        FieldKey::AgentRole(index),
                        true,
                    ),
                    section_row("Load profile"),
                    number_row(
                        "Context length",
                        opt_u64(agent.load_config.as_ref().and_then(|cfg| cfg.context_length)),
                        FieldKey::AgentContextLength(index),
                    ),
                    number_row(
                        "Eval batch",
                        opt_u64(agent.load_config.as_ref().and_then(|cfg| cfg.eval_batch_size)),
                        FieldKey::AgentEvalBatch(index),
                    ),
                    number_row(
                        "Experts",
                        opt_u64(agent.load_config.as_ref().and_then(|cfg| cfg.num_experts)),
                        FieldKey::AgentExperts(index),
                    ),
                    picker_row(
                        "Flash attention",
                        opt_bool(agent.load_config.as_ref().and_then(|cfg| cfg.flash_attention)),
                        FieldKey::AgentFlashAttention(index),
                        tri_bool_options(),
                    ),
                    picker_row(
                        "GPU KV offload",
                        opt_bool(
                            agent.load_config
                                .as_ref()
                                .and_then(|cfg| cfg.offload_kv_cache_to_gpu),
                        ),
                        FieldKey::AgentGpuKvOffload(index),
                        tri_bool_options(),
                    ),
                    section_row("Actions"),
                    action_row("Load now", FieldKey::AgentLoadNow(index)),
                    action_row("Delete agent", FieldKey::AgentDelete(index)),
                ]
            }
        }
        AgentTarget::GlobalPolicy => {
            let behavior_rows = app
                .config
                .behavior_triggers
                .behaviors
                .iter()
                .enumerate()
                .flat_map(|(index, behavior)| {
                    vec![
                        section_row(&format!("Behavior {}", behavior.behavior_id)),
                        text_row(
                            "Trigger keywords",
                            behavior.keyword_triggers.join("\n"),
                            FieldKey::BehaviorKeywords(index),
                            true,
                        ),
                        text_row(
                            "Full JSON",
                            serde_json::to_string_pretty(behavior).unwrap_or_default(),
                            FieldKey::BehaviorJson(index),
                            true,
                        ),
                    ]
                })
                .collect::<Vec<_>>();

            let mut rows = vec![
                section_row("Command policy"),
                text_row(
                    "Denylist (1/line)",
                    app.config.command_policy.denylist.join("\n"),
                    FieldKey::CommandDenylist,
                    true,
                ),
                text_row(
                    "Allowlist (1/line)",
                    app.config.command_policy.allowlist.join("\n"),
                    FieldKey::CommandAllowlist,
                    true,
                ),
                section_row("Redundancy detection"),
                picker_row(
                    "Enabled",
                    bool_raw(app.config.redundancy_detection.enabled),
                    FieldKey::RedundancyEnabled,
                    bool_options(),
                ),
                picker_row(
                    "Embedding model",
                    app.config
                        .redundancy_detection
                        .embedding_model_key
                        .clone()
                        .unwrap_or_default(),
                    FieldKey::RedundancyEmbeddingModel,
                    model_options(&app.available_models, true),
                ),
                number_row(
                    "Similarity threshold",
                    format!("{:.2}", app.config.redundancy_detection.semantic_similarity_threshold),
                    FieldKey::RedundancyThreshold,
                ),
                number_row(
                    "Max redundant retries",
                    app.config
                        .redundancy_detection
                        .max_redundant_audit_retries
                        .to_string(),
                    FieldKey::RedundancyRetries,
                ),
                section_row("Run budgets"),
                picker_row(
                    "Enabled",
                    bool_raw(app.config.run_budgets.enabled),
                    FieldKey::RunBudgetsEnabled,
                    bool_options(),
                ),
                number_row(
                    "LLM calls / window",
                    app.config.run_budgets.llm_calls_per_window.to_string(),
                    FieldKey::RunBudgetLlmCalls,
                ),
                number_row(
                    "Tool calls / window",
                    app.config.run_budgets.tool_calls_per_window.to_string(),
                    FieldKey::RunBudgetToolCalls,
                ),
                number_row(
                    "Spawned agents / window",
                    app.config.run_budgets.spawned_agents_per_window.to_string(),
                    FieldKey::RunBudgetSpawnedAgents,
                ),
                number_row(
                    "Tokens / window",
                    app.config.run_budgets.streamed_tokens_per_window.to_string(),
                    FieldKey::RunBudgetTokens,
                ),
                number_row(
                    "Wall clock seconds",
                    app.config.run_budgets.wall_clock_seconds_per_window.to_string(),
                    FieldKey::RunBudgetWallClock,
                ),
                text_row(
                    "Affected behaviors",
                    app.config.run_budgets.applies_to_behaviors.join("\n"),
                    FieldKey::RunBudgetAppliesToBehaviors,
                    true,
                ),
                picker_row(
                    "On limit",
                    app.config.run_budgets.on_limit.clone(),
                    FieldKey::RunBudgetOnLimit,
                    vec![
                        "summarize".to_string(),
                        "pause".to_string(),
                        "stop".to_string(),
                    ],
                ),
                section_row("Finalizer"),
                picker_row(
                    "Enabled",
                    bool_raw(app.config.finalizer.enabled),
                    FieldKey::FinalizerEnabled,
                    bool_options(),
                ),
                picker_row(
                    "Run on completion",
                    bool_raw(app.config.finalizer.run_on_completion),
                    FieldKey::FinalizerRunOnCompletion,
                    bool_options(),
                ),
                picker_row(
                    "Run on budget stop",
                    bool_raw(app.config.finalizer.run_on_budget_stop),
                    FieldKey::FinalizerRunOnBudgetStop,
                    bool_options(),
                ),
                text_row(
                    "Visible agent name",
                    app.config.finalizer.agent_name.clone(),
                    FieldKey::FinalizerAgentName,
                    false,
                ),
                picker_row(
                    "Finalizer model",
                    app.config.finalizer.model_key.clone().unwrap_or_default(),
                    FieldKey::FinalizerModel,
                    model_options(&app.available_models, true),
                ),
                number_row(
                    "Transcript chars",
                    app.config.finalizer.max_transcript_chars.to_string(),
                    FieldKey::FinalizerTranscriptChars,
                ),
                picker_row(
                    "Include dossier",
                    bool_raw(app.config.finalizer.include_run_dossier),
                    FieldKey::FinalizerIncludeDossier,
                    bool_options(),
                ),
                picker_row(
                    "Include commands",
                    bool_raw(app.config.finalizer.include_command_history),
                    FieldKey::FinalizerIncludeCommands,
                    bool_options(),
                ),
                picker_row(
                    "Include workers",
                    bool_raw(app.config.finalizer.include_worker_outputs),
                    FieldKey::FinalizerIncludeWorkers,
                    bool_options(),
                ),
                picker_row(
                    "Include transcript",
                    bool_raw(app.config.finalizer.include_internal_transcript),
                    FieldKey::FinalizerIncludeTranscript,
                    bool_options(),
                ),
                text_row(
                    "Completion prompt",
                    app.config.finalizer.prompt_completion.clone(),
                    FieldKey::FinalizerPromptCompletion,
                    true,
                ),
                text_row(
                    "Budget-stop prompt",
                    app.config.finalizer.prompt_budget_stop.clone(),
                    FieldKey::FinalizerPromptBudgetStop,
                    true,
                ),
                section_row("Parallel inference"),
                picker_row(
                    "Enabled",
                    bool_raw(app.config.parallel_inference.enabled),
                    FieldKey::ParallelInferenceEnabled,
                    bool_options(),
                ),
                number_row(
                    "Max parallel agents",
                    app.config.parallel_inference.max_parallel_agents.to_string(),
                    FieldKey::ParallelInferenceMaxAgents,
                ),
                section_row("Behavior triggers"),
                picker_row(
                    "Enabled",
                    bool_raw(app.config.behavior_triggers.enabled),
                    FieldKey::BehaviorTriggersEnabled,
                    bool_options(),
                ),
                picker_row(
                    "Embedding model",
                    app.config
                        .behavior_triggers
                        .embedding_model_key
                        .clone()
                        .unwrap_or_default(),
                    FieldKey::BehaviorTriggersEmbeddingModel,
                    model_options(&app.available_models, true),
                ),
                number_row(
                    "Default threshold",
                    format!("{:.2}", app.config.behavior_triggers.default_similarity_threshold),
                    FieldKey::BehaviorTriggersDefaultThreshold,
                ),
            ];
            rows.extend(behavior_rows);
            rows
        }
    }
}

fn build_routing_rows(app: &App) -> Vec<EditorRow> {
    let mut rows = vec![section_row("Connection rules")];
    for from in &app.config.agents {
        for to in &app.config.agents {
            if from.id == to.id {
                continue;
            }
            let existing = app
                .config
                .connections
                .iter()
                .find(|rule| rule.from == from.id && rule.to == to.id);
            let enabled = existing.map(|rule| rule.enabled).unwrap_or(false);
            let priority = existing.map(|rule| rule.priority).unwrap_or(128);
            let condition = existing
                .and_then(|rule| rule.condition.clone())
                .unwrap_or_default();
            rows.push(section_row(&format!("{} -> {}", from.name, to.name)));
            rows.push(picker_row(
                "Enabled",
                bool_raw(enabled),
                FieldKey::RouteEnabled(from.id.clone(), to.id.clone()),
                bool_options(),
            ));
            rows.push(number_row(
                "Priority",
                priority.to_string(),
                FieldKey::RoutePriority(from.id.clone(), to.id.clone()),
            ));
            rows.push(text_row(
                "Condition",
                condition,
                FieldKey::RouteCondition(from.id.clone(), to.id.clone()),
                false,
            ));
        }
    }
    rows
}

fn build_settings_rows(app: &App) -> Vec<EditorRow> {
    let b = &app.config.backend;

    // Build detected info display string.
    let detected_text = {
        let mut parts = Vec::new();
        if let Some(ref v) = b.detected_version {
            parts.push(format!("version: {}", v));
        }
        if let Some(ref m) = b.detected_model {
            parts.push(format!("model: {}", m));
        }
        if let Some(slots) = b.detected_parallel_slots {
            parts.push(format!("parallel slots: {}", slots));
        }
        if !b.detected_tool_use_mode.is_empty() {
            parts.push(format!("tool use: {}", b.detected_tool_use_mode));
        }
        parts.extend_from_slice(&b.detected_tool_use_notes);
        parts.extend_from_slice(&b.detected_features);
        if b.detected_tool_use_mode != "native"
            && app.config.agents.iter().any(|agent| agent.is_manager)
        {
            parts.push(
                "warning: manager agents may need compatibility recovery on this backend"
                    .to_string(),
            );
        }
        if parts.is_empty() {
            "(not yet detected)".to_string()
        } else {
            parts.join("  |  ")
        }
    };

    let mut rows = vec![
        section_row("Backend connection"),
        picker_row(
            "Backend",
            b.backend_type.clone(),
            FieldKey::BackendType,
            vec![
                "lm_studio".to_string(),
                "ollama".to_string(),
                "llamacpp".to_string(),
                "custom".to_string(),
            ],
        ),
        text_row("Base URL", b.base_url.clone(), FieldKey::BackendUrl, false),
        text_row(
            "API key",
            b.api_key.clone().unwrap_or_default(),
            FieldKey::BackendApiKey,
            false,
        ),
        action_row("Detect capabilities", FieldKey::BackendDetect),
        text_row(
            "Detected",
            detected_text,
            FieldKey::BackendDetectedInfo,
            false,
        ),
    ];

    // Extra instances (for backends like llama.cpp that serve one model per process).
    rows.push(section_row("Extra instances"));
    if b.backend_type == "llamacpp" {
        rows.push(action_row(
            "Discover running servers",
            FieldKey::BackendDiscover,
        ));
    }
    rows.push(action_row("Add instance", FieldKey::BackendInstanceAdd));
    for (i, inst) in b.extra_instances.iter().enumerate() {
        rows.push(section_row(&format!("Instance {}", i + 1)));
        rows.push(text_row(
            "URL",
            inst.url.clone(),
            FieldKey::BackendInstanceUrl(i),
            false,
        ));
        rows.push(text_row(
            "Model",
            inst.model_hint.clone(),
            FieldKey::BackendInstanceModelHint(i),
            false,
        ));
        rows.push(action_row("Remove", FieldKey::BackendInstanceDelete(i)));
    }

    rows
}

fn routing_matrix_text(app: &App) -> String {
    let names = app
        .config
        .agents
        .iter()
        .map(|agent| truncate_display(&agent.name, 8))
        .collect::<Vec<_>>();
    if names.is_empty() {
        return "No agents configured.".to_string();
    }
    let mut lines = vec![format!("From\\To | {}", names.join(" | "))];
    for from in &app.config.agents {
        let mut cells = Vec::new();
        for to in &app.config.agents {
            if from.id == to.id {
                cells.push("—".to_string());
            } else {
                let enabled = app
                    .config
                    .connections
                    .iter()
                    .find(|rule| rule.from == from.id && rule.to == to.id)
                    .map(|rule| rule.enabled)
                    .unwrap_or(false);
                cells.push(if enabled { "Y" } else { "." }.to_string());
            }
        }
        lines.push(format!(
            "{:<7}| {}",
            truncate_display(&from.name, 7),
            cells.join(" | ")
        ));
    }
    lines.join("\n")
}

fn render_thread_summaries(app: &App) -> Vec<ListItem<'static>> {
    app.messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let prefix = if message.is_sent {
                "You"
            } else if message.is_error {
                "ERR"
            } else {
                message.agent_name.as_str()
            };
            let tool_summary = if message.tools.is_empty() {
                String::new()
            } else {
                let done = message.tools.iter().filter(|tool| tool.status == "done").count();
                let pending = message.tools.iter().filter(|tool| tool.status == "pending").count();
                let errored = message.tools.iter().filter(|tool| tool.status == "error").count();
                format!("  tools:{}/{} err:{}", done, pending, errored)
            };
            let preview_source = if message.content.trim().is_empty() {
                if let Some(tool) = message.tools.first() {
                    format!("tool {}", tool.tool_name)
                } else {
                    String::from("(empty)")
                }
            } else {
                message.content.replace('\n', " ")
            };
            let section_marker = if index == app.chat_message_sel && app.chat_focus == ChatFocus::Sections {
                "▸"
            } else {
                " "
            };
            let prefix_style = if message.is_error {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else if message.is_sent {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            };
            let marker_style = if index == app.chat_message_sel {
                selection_style(app.chat_focus, ChatFocus::Messages)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", section_marker), marker_style),
                Span::styled(format!("{:<14}", truncate_display(prefix, 14)), prefix_style),
                Span::raw(" "),
                Span::raw(truncate_display(&preview_source, 72)),
                Span::styled(tool_summary, Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect()
}

fn render_chat_detail(app: &App) -> Text<'static> {
    let Some(message) = app.messages.get(app.chat_message_sel) else {
        return Text::from("No message selected.");
    };
    let sections = app.selected_message_sections();
    if sections.is_empty() {
        return Text::from("No sections.");
    }

    let mut lines: Vec<Line<'static>> = Vec::new();
    let title_style = if app.chat_focus == ChatFocus::Messages {
        selection_style(app.chat_focus, ChatFocus::Messages)
    } else {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    };
    lines.push(Line::from(vec![
        Span::styled("Message: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(
                "{}{}",
                if message.is_sent { "You" } else { &message.agent_name },
                if message.is_error { " [error]" } else { "" }
            ),
            title_style,
        ),
    ]));

    let mut section_spans = vec![Span::styled(
        "Sections: ",
        Style::default().fg(Color::DarkGray),
    )];
    for (idx, section) in sections.iter().enumerate() {
        if idx > 0 {
            section_spans.push(Span::raw("  "));
        }
        let label = match section {
            ChatSection::Content => "content".to_string(),
            ChatSection::Tool(tool_idx) => {
                let tool = &message.tools[*tool_idx];
                format!("tool:{} [{}]", tool.tool_name, tool.status)
            }
        };
        let selected = idx == app.chat_section_sel;
        let style = if selected {
            selection_style(app.chat_focus, ChatFocus::Sections)
        } else {
            Style::default().fg(Color::Gray)
        };
        section_spans.push(Span::styled(format!("[{}]", label), style));
    }
    lines.push(Line::from(section_spans));
    lines.push(Line::default());

    match sections[app.chat_section_sel.min(sections.len().saturating_sub(1))] {
        ChatSection::Content => {
            let content = if message.content.trim().is_empty() {
                "(no content yet)".to_string()
            } else {
                message.content.clone()
            };
            lines.push(Line::from(Span::styled(
                "Content",
                section_heading_style(app.chat_focus),
            )));
            lines.push(Line::default());
            lines.extend(render_markdown_lines(&content));
        }
        ChatSection::Tool(tool_idx) => {
            let tool = &message.tools[tool_idx];
            let expanded = app
                .expanded_tool_sections
                .contains(&format!("{}:{}", app.chat_message_sel, tool_idx));
            lines.push(Line::from(Span::styled(
                format!("Tool: {}", tool.tool_name),
                section_heading_style(app.chat_focus),
            )));
            lines.push(Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::DarkGray)),
                Span::styled(tool.status.clone(), tool_status_style(&tool.status)),
            ]));
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                "Arguments",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
            )));
            if tool.args.trim().is_empty() {
                lines.push(Line::from("  (none)"));
            } else {
                lines.extend(render_markdown_lines(&tool.args));
            }
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                if expanded {
                    "Result"
                } else {
                    "Result [collapsed, press Enter to expand]"
                },
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
            )));
            if expanded {
                if tool.result.trim().is_empty() {
                    lines.push(Line::from("  (no result)"));
                } else {
                    lines.extend(render_markdown_lines(&tool.result));
                }
            }
        }
    }

    Text::from(lines)
}

fn selection_style(current_focus: ChatFocus, highlighted_focus: ChatFocus) -> Style {
    if current_focus == highlighted_focus {
        let bg = match highlighted_focus {
            ChatFocus::Messages => Color::Yellow,
            ChatFocus::Sections => Color::LightBlue,
            ChatFocus::Detail => Color::Green,
        };
        Style::default()
            .fg(Color::Black)
            .bg(bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(Color::White)
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD)
    }
}

fn section_heading_style(focus: ChatFocus) -> Style {
    selection_style(focus, focus)
}

fn tool_status_style(status: &str) -> Style {
    match status {
        "done" => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        "pending" => Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        "error" => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        _ => Style::default().fg(Color::White),
    }
}

fn render_markdown_lines(source: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let mut ordered_index = 1usize;

    for raw in source.lines() {
        let trimmed = raw.trim_end();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            lines.push(Line::from(Span::styled(
                trimmed.to_string(),
                Style::default().fg(Color::Green),
            )));
            continue;
        }
        if trimmed.is_empty() {
            lines.push(Line::default());
            ordered_index = 1;
            continue;
        }
        if let Some((level, content)) = parse_heading(trimmed) {
            lines.push(Line::from(vec![Span::styled(
                content.to_string(),
                heading_style(level),
            )]));
            continue;
        }
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            lines.push(Line::from(Span::styled(
                "────────────────────────",
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }
        if let Some(content) = trimmed.strip_prefix("> ") {
            lines.push(Line::from(prefix_and_parse_inline(
                "> ",
                Style::default()
                    .fg(Color::LightBlue)
                    .add_modifier(Modifier::BOLD),
                content,
                Style::default()
                    .fg(Color::LightBlue)
                    .add_modifier(Modifier::ITALIC),
            )));
            continue;
        }
        if let Some(content) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
            lines.push(Line::from(prefix_and_parse_inline(
                "- ",
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                content,
                Style::default(),
            )));
            continue;
        }
        if let Some((number, content)) = parse_ordered_item(trimmed) {
            let marker = format!("{}. ", number.unwrap_or(ordered_index));
            ordered_index = number.unwrap_or(ordered_index) + 1;
            lines.push(Line::from(prefix_and_parse_inline(
                &marker,
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                content,
                Style::default(),
            )));
            continue;
        }
        ordered_index = 1;
        lines.push(Line::from(parse_inline_markdown(
            trimmed,
            Style::default(),
        )));
    }

    if lines.is_empty() {
        lines.push(Line::default());
    }
    lines
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let hashes = line.chars().take_while(|ch| *ch == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let content = line[hashes..].trim_start();
    if content.is_empty() {
        None
    } else {
        Some((hashes, content))
    }
}

fn parse_ordered_item(line: &str) -> Option<(Option<usize>, &str)> {
    let mut split = line.splitn(2, ". ");
    let first = split.next()?;
    let rest = split.next()?;
    if first.chars().all(|ch| ch.is_ascii_digit()) {
        Some((first.parse::<usize>().ok(), rest))
    } else {
        None
    }
}

fn prefix_and_parse_inline(
    prefix: &str,
    prefix_style: Style,
    content: &str,
    base_style: Style,
) -> Vec<Span<'static>> {
    let mut spans = vec![Span::styled(prefix.to_string(), prefix_style)];
    spans.extend(parse_inline_markdown(content, base_style));
    spans
}

fn parse_inline_markdown(source: &str, base_style: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_marker(&chars, i + 2, '*') {
                let content: String = chars[i + 2..end].iter().collect();
                spans.push(Span::styled(
                    content,
                    base_style.add_modifier(Modifier::BOLD),
                ));
                i = end + 2;
                continue;
            }
        }
        if chars[i] == '*' || chars[i] == '_' {
            if let Some(end) = find_single_marker(&chars, i + 1, chars[i]) {
                let content: String = chars[i + 1..end].iter().collect();
                spans.push(Span::styled(
                    content,
                    base_style.add_modifier(Modifier::ITALIC),
                ));
                i = end + 1;
                continue;
            }
        }
        if chars[i] == '`' {
            if let Some(end) = find_single_marker(&chars, i + 1, '`') {
                let content: String = chars[i + 1..end].iter().collect();
                spans.push(Span::styled(
                    content,
                    Style::default()
                        .fg(Color::Green)
                        .bg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                ));
                i = end + 1;
                continue;
            }
        }
        if chars[i] == '[' {
            if let Some(close_bracket) = find_single_marker(&chars, i + 1, ']') {
                if close_bracket + 1 < chars.len() && chars[close_bracket + 1] == '(' {
                    if let Some(close_paren) = find_single_marker(&chars, close_bracket + 2, ')') {
                        let label: String = chars[i + 1..close_bracket].iter().collect();
                        let url: String = chars[close_bracket + 2..close_paren].iter().collect();
                        spans.push(Span::styled(
                            format!("{} ({})", label, url),
                            Style::default()
                                .fg(Color::LightBlue)
                                .add_modifier(Modifier::UNDERLINED),
                        ));
                        i = close_paren + 1;
                        continue;
                    }
                }
            }
        }

        let start = i;
        while i < chars.len()
            && !(chars[i] == '*'
                || chars[i] == '_'
                || chars[i] == '`'
                || chars[i] == '[')
        {
            i += 1;
        }
        if start == i {
            spans.push(Span::styled(chars[i].to_string(), base_style));
            i += 1;
        } else {
            let content: String = chars[start..i].iter().collect();
            if !content.is_empty() {
                spans.push(Span::styled(content, base_style));
            }
        }
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base_style));
    }
    spans
}

fn find_single_marker(chars: &[char], start: usize, marker: char) -> Option<usize> {
    (start..chars.len()).find(|&idx| chars[idx] == marker)
}

fn find_double_marker(chars: &[char], start: usize, marker: char) -> Option<usize> {
    let mut idx = start;
    while idx + 1 < chars.len() {
        if chars[idx] == marker && chars[idx + 1] == marker {
            return Some(idx);
        }
        idx += 1;
    }
    None
}

fn heading_style(level: usize) -> Style {
    match level {
        1 => Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
        2 => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        3 => Style::default()
            .fg(Color::LightBlue)
            .add_modifier(Modifier::BOLD),
        _ => Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    }
}

fn active_thread_label(app: &App) -> String {
    match (&app.active_workspace_id, &app.active_thread_id) {
        (Some(workspace_id), Some(thread_id)) => {
            let workspace = app
                .workspace_config
                .workspaces
                .iter()
                .find(|workspace| &workspace.id == workspace_id);
            let thread = workspace.and_then(|workspace| {
                workspace
                    .threads
                    .iter()
                    .find(|thread| &thread.id == thread_id)
            });
            match (workspace, thread) {
                (Some(workspace), Some(thread)) => {
                    format!(" {} / {} ", workspace.name, thread.name)
                }
                _ => " Chat ".to_string(),
            }
        }
        _ => " Chat ".to_string(),
    }
}

fn refresh_models(runtime: RuntimeHandle, tx: mpsc::UnboundedSender<TuiEvent>) {
    tokio::spawn(async move {
        let result = runtime.fetch_models().await;
        let _ = tx.send(TuiEvent::ModelsLoaded(result));
    });
}

fn build_ws_flat(app: &App) -> Vec<WorkspaceRow> {
    let mut flat = vec![WorkspaceRow::AddWorkspace];
    for (workspace_index, _) in app.workspace_config.workspaces.iter().enumerate() {
        flat.push(WorkspaceRow::Workspace(workspace_index));
        if app.ws_expanded.get(workspace_index).copied().unwrap_or(false) {
            flat.push(WorkspaceRow::NewThread(workspace_index));
            for thread_index in app.sorted_thread_indices(workspace_index) {
                flat.push(WorkspaceRow::Thread {
                    workspace_index,
                    thread_index,
                });
            }
        }
    }
    flat
}

fn section_row(label: &str) -> EditorRow {
    EditorRow {
        label: label.to_string(),
        display: String::new(),
        raw: String::new(),
        key: None,
        kind: EditorKind::Section,
    }
}

fn text_row(label: &str, value: String, key: FieldKey, multiline: bool) -> EditorRow {
    EditorRow {
        label: label.to_string(),
        display: if multiline {
            truncate_display(&value.replace('\n', " | "), 72)
        } else {
            truncate_display(&value, 72)
        },
        raw: value,
        key: Some(key),
        kind: EditorKind::Text { multiline },
    }
}

fn number_row(label: &str, value: String, key: FieldKey) -> EditorRow {
    EditorRow {
        label: label.to_string(),
        display: if value.is_empty() { "none".to_string() } else { value.clone() },
        raw: value,
        key: Some(key),
        kind: EditorKind::Number,
    }
}

fn picker_row(label: &str, value: String, key: FieldKey, options: Vec<String>) -> EditorRow {
    EditorRow {
        label: label.to_string(),
        display: if value.is_empty() { "none".to_string() } else { value.clone() },
        raw: value,
        key: Some(key),
        kind: EditorKind::Picker { options },
    }
}

fn action_row(label: &str, key: FieldKey) -> EditorRow {
    EditorRow {
        label: label.to_string(),
        display: "run".to_string(),
        raw: String::new(),
        key: Some(key),
        kind: EditorKind::Action,
    }
}

fn edit_buffer(buffer: &mut String, cursor: &mut usize, key: &event::KeyEvent, multiline: bool) {
    match key.code {
        KeyCode::Backspace => {
            if *cursor > 0 {
                *cursor -= 1;
                buffer.remove(char_to_byte(buffer, *cursor));
            }
        }
        KeyCode::Delete => {
            if *cursor < buffer.chars().count() {
                buffer.remove(char_to_byte(buffer, *cursor));
            }
        }
        KeyCode::Left => *cursor = cursor.saturating_sub(1),
        KeyCode::Right => {
            if *cursor < buffer.chars().count() {
                *cursor += 1;
            }
        }
        KeyCode::Home => *cursor = 0,
        KeyCode::End => *cursor = buffer.chars().count(),
        KeyCode::Tab if multiline => insert_char(buffer, cursor, '\t'),
        KeyCode::Char(ch) => insert_char(buffer, cursor, ch),
        _ => {}
    }
}

fn insert_char(buffer: &mut String, cursor: &mut usize, ch: char) {
    let byte = char_to_byte(buffer, *cursor);
    buffer.insert(byte, ch);
    *cursor += 1;
}

fn char_to_byte(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn cursor_line_col(text: &str, cursor: usize) -> (usize, usize) {
    let mut line = 0;
    let mut col = 0;
    for (idx, ch) in text.chars().enumerate() {
        if idx == cursor {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value.trim().to_lowercase().as_str() {
        "true" | "yes" | "enabled" => Ok(true),
        "false" | "no" | "disabled" => Ok(false),
        other => Err(format!("Expected boolean, got '{}'", other)),
    }
}

fn parse_optional_bool(value: &str) -> Result<Option<bool>, String> {
    if value.trim().is_empty() || value.trim().eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        parse_bool(value).map(Some)
    }
}

fn parse_u32(value: &str) -> Result<u32, String> {
    value.trim().parse::<u32>().map_err(|error| error.to_string())
}

fn parse_u64(value: &str) -> Result<u64, String> {
    value.trim().parse::<u64>().map_err(|error| error.to_string())
}

fn parse_u8(value: &str) -> Result<u8, String> {
    value.trim().parse::<u8>().map_err(|error| error.to_string())
}

fn parse_usize(value: &str) -> Result<usize, String> {
    value.trim().parse::<usize>().map_err(|error| error.to_string())
}

fn parse_optional_u64(value: &str) -> Result<Option<u64>, String> {
    if value.trim().is_empty() || value.trim().eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        parse_u64(value).map(Some)
    }
}

fn parse_f32(value: &str) -> Result<f32, String> {
    value.trim().parse::<f32>().map_err(|error| error.to_string())
}

fn split_lines(value: &str) -> Vec<String> {
    value
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

fn bool_raw(value: bool) -> String {
    if value {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

fn opt_bool(value: Option<bool>) -> String {
    value.map(bool_raw).unwrap_or_else(|| "".to_string())
}

fn opt_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn bool_options() -> Vec<String> {
    vec!["true".to_string(), "false".to_string()]
}

fn tri_bool_options() -> Vec<String> {
    vec!["".to_string(), "true".to_string(), "false".to_string()]
}

fn model_options(models: &[ModelInfo], include_none: bool) -> Vec<String> {
    let mut options = Vec::new();
    if include_none {
        options.push(String::new());
    }
    options.extend(models.iter().map(|model| model.key.clone()));
    options
}

fn truncate_display(value: &str, limit: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= limit {
        value.to_string()
    } else {
        let head = value.chars().take(limit.saturating_sub(1)).collect::<String>();
        format!("{}…", head)
    }
}

fn create_text_message(
    agent_name: &str,
    content: &str,
    is_sent: bool,
    is_error: bool,
) -> WorkspaceThreadMessage {
    WorkspaceThreadMessage {
        kind: "text".to_string(),
        agent_id: None,
        agent_name: agent_name.to_string(),
        content: content.to_string(),
        is_sent,
        is_error,
        for_user: false,
        internal: false,
        tools: vec![],
        signal: None,
    }
}

fn create_bubble_message(
    agent_id: &str,
    agent_name: &str,
    for_user: bool,
    internal: bool,
) -> WorkspaceThreadMessage {
    WorkspaceThreadMessage {
        kind: "bubble".to_string(),
        agent_id: Some(agent_id.to_string()),
        agent_name: agent_name.to_string(),
        content: String::new(),
        is_sent: false,
        is_error: false,
        for_user,
        internal,
        tools: vec![],
        signal: None,
    }
}

fn system_error_message(error: String) -> WorkspaceThreadMessage {
    create_text_message("System", &error, false, true)
}

fn ensure_load_config(config: &mut AgentConfig, index: usize) -> &mut app_lib::runtime::AgentLoadConfig {
    if config.agents[index].load_config.is_none() {
        config.agents[index].load_config = Some(app_lib::runtime::AgentLoadConfig {
            context_length: None,
            eval_batch_size: None,
            flash_attention: None,
            num_experts: None,
            offload_kv_cache_to_gpu: None,
        });
    }
    config.agents[index].load_config.as_mut().unwrap()
}

fn ensure_route_rule<'a>(
    rules: &'a mut Vec<RoutingRule>,
    from: &str,
    to: &str,
) -> &'a mut RoutingRule {
    if let Some(position) = rules
        .iter()
        .position(|rule| rule.from == from && rule.to == to)
    {
        return &mut rules[position];
    }
    rules.push(RoutingRule {
        from: from.to_string(),
        to: to.to_string(),
        priority: 128,
        condition: None,
        enabled: true,
    });
    rules.last_mut().unwrap()
}

fn summarize_thread_name(message: &str) -> String {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        "Thread".to_string()
    } else {
        truncate_display(trimmed, 20)
    }
}

fn next_thread_name(threads: &[app_lib::runtime::WorkspaceThread]) -> String {
    let mut candidate = "New Thread".to_string();
    let mut suffix = 2usize;
    let existing = threads
        .iter()
        .map(|thread| thread.name.as_str())
        .collect::<HashSet<_>>();
    while existing.contains(candidate.as_str()) {
        candidate = format!("New Thread {}", suffix);
        suffix += 1;
    }
    candidate
}

fn thread_recency_key(workspace_id: &str, thread_id: &str) -> String {
    format!("{}/{}", workspace_id, thread_id)
}

fn generate_id(prefix: &str) -> String {
    format!("{}-{}", prefix, chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default())
}
