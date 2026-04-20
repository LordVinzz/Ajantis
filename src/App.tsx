import {
  Channel,
  invoke,
} from "@tauri-apps/api/core";
import {
  memo,
  startTransition,
  type CSSProperties,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { streamStore, useStreamSnapshot } from "./eventStore";
import { renderMarkdown } from "./markdown";
import type {
  ActiveRunUiState,
  Agent,
  AgentConfig,
  BehaviorTriggerConfig,
  CommandExecution,
  LoadedModel,
  MemoryEntry,
  ModelInfo,
  RoutingRule,
  RuntimeState,
  StreamEvent,
  TabKey,
  ThreadSnapshot,
  WorkspaceEntry,
  WorkspaceThreadIndex,
  WorkspaceThreadMessage,
} from "./types";

const MIN_SEMANTIC_REDUNDANCY_THRESHOLD = 0.85;
const MAX_SEMANTIC_REDUNDANCY_THRESHOLD = 0.99;
const MIN_BEHAVIOR_TRIGGER_THRESHOLD = 0.0;
const MAX_BEHAVIOR_TRIGGER_THRESHOLD = 1.0;
const COLLAPSED_THREAD_RENDER_LIMIT = 150;
const MESSAGE_CARD_STYLE: CSSProperties = {
  contentVisibility: "auto",
  containIntrinsicSize: "240px",
};

function defaultGroundedAuditBehavior(): BehaviorTriggerConfig {
  return {
    behavior_id: "grounded_audit",
    enabled: true,
    keyword_triggers: [
      "security audit",
      "security review",
      "repo review",
      "code review",
      "find potential issues",
      "find bugs",
      "find vulnerabilities",
      "report findings",
      "report them here",
    ],
    embedding_trigger_phrases: [
      "audit this repository and report concrete issues",
      "perform a security review of this codebase",
      "inspect the repo for vulnerabilities and risky configurations",
    ],
    similarity_threshold: null,
    system_prompt_injection:
      "Grounded audit behavior is active for this task.\n- Treat explicitly requested files, directories, file classes, services, and subsystems as a coverage checklist.\n- Use the sections `Confirmed findings`, `Hypotheses / lower-confidence risks`, and `Coverage gaps` when reporting audit-style results.\n- A confirmed finding must be grounded in code, configuration, policy, or command evidence actually inspected.\n- Do not label anything `High` or `Critical` without direct supporting evidence already in context.\n- A hypothesis must cite a concrete file, config, endpoint, policy, or observed behavior, or explicitly state what evidence is still missing to confirm it.\n- If evidence is insufficient, say `insufficient evidence` instead of inventing a finding.",
    runtime_note_enabled: true,
    runtime_note_template:
      "Audit topics already covered in this turn:\n{covered_topics_bullets}\n\nAvoid re-delegating or re-checking the same file/function/theme unless you are opening a genuinely new code/config area. If current evidence is already sufficient, synthesize now.\n\nRequired audit coverage still unresolved:\n{unresolved_scopes_bullets}",
    coverage_manifest: {
      enabled: true,
      require_resolution: true,
      unresolved_prompt:
        "Before concluding, you must either inspect these required audit scopes or name them explicitly under `{gap_section_label}`: {unresolved_scopes_csv}. Do not omit them.",
      gap_section_label: "Coverage gaps",
    },
    required_sections: [
      "Confirmed findings",
      "Hypotheses / lower-confidence risks",
      "Coverage gaps",
    ],
    section_rules: [],
    response_rewrite: {
      enabled: true,
      min_evidence_grade_for_severity: "config_content",
      rewrite_prompt:
        "Rewrite the audit answer below using only the evidence already present in this conversation.",
    },
    evidence_grading: {
      min_grade_to_synthesize: "config_content",
      code_signals: ["fn ", "function ", "class ", "struct ", "async "],
      config_signals: ["\"$schema\"", "\"permissions\"", "\"scripts\"", "[dependencies]"],
      command_signals: ["src/", "package.json", "cargo.toml", "dockerfile"],
    },
    force_synthesis: {
      after_n_completed_reports: 3,
      after_n_issue_reports: 1,
      require_coverage_gap_signal: true,
      min_targeted_topics: 3,
      min_evidence_backed_topics: 3,
      prompt:
        "Tooling or delegation has stalled. Using only the evidence already present in this conversation, synthesize now.",
      fallback_text:
        "Confirmed findings\n- None supported strongly enough to promote after tooling stalled.",
    },
    delegation_validation: {
      enabled: true,
      max_weak_retries: 1,
      retry_prompt_template:
        "The delegated audit result on `{topic}` did not add new useful evidence. Retry this topic once with one concrete file or function to inspect.",
    },
    tool_burst_reflection: {
      limit: 3,
      prompt:
        "Pause tool use. Briefly summarize what you learned, what is still uncertain, and the smallest next step.",
    },
    non_progress: {
      limit: 4,
      stall_prompt:
        "Your previous response did not contain a usable final answer. Stop exploring and provide either concrete findings or a short bounded summary.",
    },
  };
}

function defaultAgentConfig(): AgentConfig {
  return {
    agents: [
      {
        id: "user",
        name: "User",
        type: "user",
        armed: true,
        is_manager: false,
      },
    ],
    connections: [],
    command_policy: { allowlist: [], denylist: [] },
    redundancy_detection: {
      enabled: true,
      embedding_model_key: null,
      semantic_similarity_threshold: 0.9,
      max_redundant_audit_retries: 1,
    },
    behavior_triggers: {
      enabled: true,
      embedding_model_key: null,
      default_similarity_threshold: 0.9,
      behaviors: [defaultGroundedAuditBehavior()],
    },
    run_budgets: {
      enabled: true,
      llm_calls_per_window: 20,
      tool_calls_per_window: 60,
      spawned_agents_per_window: 8,
      streamed_tokens_per_window: 75_000,
      wall_clock_seconds_per_window: 600,
      applies_to_behaviors: [],
      on_limit: "summarize" as const,
      summarization: {
        enabled: true,
        prompt: "The run has hit its configured soft budget limit. Using only the evidence already present in this conversation, produce a final concise summary of:\n1. What was accomplished and key findings so far.\n2. What remains incomplete or uncertain.\n3. The next concrete step if the task were to continue.\n\nDo not call any tools. Do not apologize. Be direct and factual.",
        model_key: null,
      },
    },
  };
}

function createTextMessage(agentName: string, content: string, isSent = false, isError = false): WorkspaceThreadMessage {
  return {
    kind: "text",
    agent_name: agentName,
    content,
    is_sent: isSent,
    is_error: isError,
    for_user: false,
    internal: false,
    tools: [],
    signal: null,
    runtime: null,
  };
}

function createBubbleMessage(agentId: string, agentName: string, forUser: boolean): WorkspaceThreadMessage {
  return {
    kind: "bubble",
    agent_id: agentId,
    agent_name: agentName,
    content: "",
    is_sent: false,
    is_error: false,
    for_user: forUser,
    internal: !forUser,
    tools: [],
    signal: null,
    runtime: {
      stage: "queued",
      detail: "Waiting for backend dispatch…",
      estimated_output_tokens: 0,
    },
  };
}

function generateId(prefix: string) {
  return `${prefix}-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

function formatTokenCount(value?: number | null) {
  if (value == null || Number.isNaN(Number(value))) {
    return "?";
  }
  return Number(value).toLocaleString("en-US");
}

function formatSpeed(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) {
    return null;
  }
  return `${Number(value).toFixed(1)} tok/s`;
}

function isTauriRuntime() {
  return typeof window !== "undefined" && Boolean((window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
}

/* ── ThreadMessageCard ──────────────────────────────────── */

const ThreadMessageCard = memo(function ThreadMessageCard({
  message,
}: {
  message: WorkspaceThreadMessage;
}) {
  const renderedContent = useMemo(
    () => renderMarkdown(message.content || ""),
    [message.content],
  );

  const msgClass = [
    "w98-msg",
    message.is_sent ? "w98-msg-out" : message.for_user ? "w98-msg-for-user" : "",
    message.is_error ? "w98-msg-error" : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={msgClass} style={MESSAGE_CARD_STYLE}>
      <div className={`w98-msg-header${message.is_sent ? " w98-msg-header-right" : ""}`}>
        <span>{message.agent_name}</span>
        {message.runtime && (
          <span className="w98-msg-meta">
            {[message.runtime.model_key, message.runtime.detail].filter(Boolean).join(" · ")}
          </span>
        )}
      </div>

      {message.tools.length > 0 && (
        <div style={{ marginBottom: "4px" }}>
          {message.tools.map((tool, toolIndex) => (
            <details
              className="w98-tool-detail"
              key={`${tool.tool_name}-${toolIndex}`}
              open={toolIndex === message.tools.length - 1}
            >
              <summary>{tool.tool_name} · {tool.status || "pending"}</summary>
              <div className="w98-tool-body">
                <div className="w98-label">Args</div>
                <pre className="w98-pre">{tool.args || ""}</pre>
                {tool.result && (
                  <>
                    <div className="w98-label" style={{ marginTop: "4px" }}>Result</div>
                    <pre className="w98-pre">{tool.result}</pre>
                  </>
                )}
              </div>
            </details>
          ))}
        </div>
      )}

      <div
        className="w98-prose"
        dangerouslySetInnerHTML={{ __html: renderedContent }}
      />
    </div>
  );
});

ThreadMessageCard.displayName = "ThreadMessageCard";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function stripNullish<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/* ── App ─────────────────────────────────────────────────── */

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("chat");
  const [agentConfig, setAgentConfig] = useState<AgentConfig>(defaultAgentConfig());
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [loadedModels, setLoadedModels] = useState<LoadedModel[]>([]);
  const [workspaceConfig, setWorkspaceConfig] = useState<{ workspaces: WorkspaceEntry[] }>({
    workspaces: [],
  });
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [currentThreadMessages, setCurrentThreadMessages] = useState<WorkspaceThreadMessage[]>([]);
  const [memoryEntries, setMemoryEntries] = useState<MemoryEntry[]>([]);
  const [commandHistory, setCommandHistory] = useState<CommandExecution[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [threadSearch, setThreadSearch] = useState("");
  const [liveAgentState, setLiveAgentState] = useState<Record<string, RuntimeState>>({});
  const [backendStatus, setBackendStatus] = useState({
    activeAgents: new Set<string>(),
    phase: "idle",
    detail: "No active backend work.",
  });
  const [activeRun, setActiveRun] = useState<ActiveRunUiState | null>(null);
  const [saveNotice, setSaveNotice] = useState<string>("");
  const [runUsage, setRunUsage] = useState<{
    llm_calls: number; tool_calls: number; spawned_agents: number;
    streamed_tokens: number; embedding_calls: number; wall_clock_seconds: number;
  } | null>(null);
  const tauriAvailable = useMemo(() => isTauriRuntime(), []);
  const deferredThreadSearch = useDeferredValue(threadSearch);
  const streamSnapshot = useStreamSnapshot();
  const messagesRef = useRef<WorkspaceThreadMessage[]>([]);
  const currentRunChannel = useRef<Channel<StreamEvent> | null>(null);
  const persistTimerRef = useRef<number | null>(null);
  const tokenBufferRef = useRef<Record<string, string>>({});
  const tokenFlushFrameRef = useRef<number | null>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const [showFullThreadHistory, setShowFullThreadHistory] = useState(false);

  const modelDetails = useMemo<Record<string, ModelInfo>>(
    () => Object.fromEntries(availableModels.map((model) => [model.key, model])),
    [availableModels],
  );

  const nonUserAgents = useMemo(
    () => agentConfig.agents.filter((agent) => (agent.type || agent.agent_type) !== "user"),
    [agentConfig.agents],
  );

  const loadedInstanceCount = useMemo(
    () => loadedModels.reduce((sum, model) => sum + (model.instances?.length ?? 0), 0),
    [loadedModels],
  );

  const selectedWorkspace = useMemo(
    () => workspaceConfig.workspaces.find((workspace) => workspace.id === activeWorkspaceId) ?? null,
    [workspaceConfig.workspaces, activeWorkspaceId],
  );

  const selectedThread = useMemo(
    () => selectedWorkspace?.threads.find((thread) => thread.id === activeThreadId) ?? null,
    [selectedWorkspace, activeThreadId],
  );

  const filteredWorkspaces = useMemo(() => {
    const query = deferredThreadSearch.trim().toLowerCase();
    if (!query) {
      return workspaceConfig.workspaces;
    }
    return workspaceConfig.workspaces
      .map((workspace) => ({
        ...workspace,
        threads: workspace.threads.filter((thread) =>
          thread.name.toLowerCase().includes(query),
        ),
      }))
      .filter(
        (workspace) =>
          workspace.name.toLowerCase().includes(query) || workspace.threads.length > 0,
      );
  }, [deferredThreadSearch, workspaceConfig.workspaces]);

  const hiddenThreadMessageCount = useMemo(
    () =>
      showFullThreadHistory || currentThreadMessages.length <= COLLAPSED_THREAD_RENDER_LIMIT
        ? 0
        : currentThreadMessages.length - COLLAPSED_THREAD_RENDER_LIMIT,
    [currentThreadMessages.length, showFullThreadHistory],
  );

  const visibleThreadMessages = useMemo(
    () =>
      hiddenThreadMessageCount > 0
        ? currentThreadMessages.slice(hiddenThreadMessageCount)
        : currentThreadMessages,
    [currentThreadMessages, hiddenThreadMessageCount],
  );

  useEffect(() => {
    messagesRef.current = currentThreadMessages;
  }, [currentThreadMessages]);

  useEffect(() => {
    setShowFullThreadHistory(false);
    tokenBufferRef.current = {};
    if (tokenFlushFrameRef.current != null) {
      window.cancelAnimationFrame(tokenFlushFrameRef.current);
      tokenFlushFrameRef.current = null;
    }
  }, [activeWorkspaceId, activeThreadId]);

  useEffect(() => {
    return () => {
      if (tokenFlushFrameRef.current != null) {
        window.cancelAnimationFrame(tokenFlushFrameRef.current);
      }
    };
  }, []);

  async function fetchAvailableModels() {
    const models = await invoke<ModelInfo[]>("fetch_models");
    startTransition(() => setAvailableModels(models));
  }

  async function refreshLoadedModels() {
    const models = await invoke<LoadedModel[]>("fetch_loaded_models");
    startTransition(() => setLoadedModels(models));
  }

  async function loadConfig() {
    const config = await invoke<AgentConfig>("load_agent_config");
    startTransition(() => setAgentConfig({ ...defaultAgentConfig(), ...config }));
  }

  async function saveConfig() {
    const cleaned: AgentConfig = {
      ...agentConfig,
      command_policy: {
        denylist: agentConfig.command_policy.denylist.map((s) => s.trim()).filter(Boolean),
        allowlist: agentConfig.command_policy.allowlist.map((s) => s.trim()).filter(Boolean),
      },
      run_budgets: {
        ...agentConfig.run_budgets,
        applies_to_behaviors: agentConfig.run_budgets.applies_to_behaviors.map((s) => s.trim()).filter(Boolean),
      },
      behavior_triggers: {
        ...agentConfig.behavior_triggers,
        behaviors: agentConfig.behavior_triggers.behaviors.map((b) => ({
          ...b,
          keyword_triggers: b.keyword_triggers.map((s) => s.trim()).filter(Boolean),
        })),
      },
    };
    await invoke("save_agent_config", { config: stripNullish(cleaned) });
    setSaveNotice("Configuration saved.");
    window.setTimeout(() => setSaveNotice(""), 1500);
  }

  async function loadWorkspaceIndex() {
    const config = await invoke<{ workspaces: WorkspaceEntry[] }>("load_workspace_config");
    startTransition(() => {
      setWorkspaceConfig(config);
      setActiveWorkspaceId((prev) => prev ?? config.workspaces[0]?.id ?? null);
    });
  }

  async function saveWorkspaceIndex(nextConfig: { workspaces: WorkspaceEntry[] }) {
    await invoke("save_workspace_config", { config: stripNullish(nextConfig) });
  }

  async function loadThreadSnapshot(workspaceId: string, threadId: string) {
    const snapshot = await invoke<ThreadSnapshot>("load_thread_snapshot", {
      workspaceId,
      threadId,
    });
    setCurrentThreadMessages(snapshot.message_items ?? []);
    setMemoryEntries(snapshot.memory_entries ?? []);
    setCommandHistory(snapshot.command_history ?? []);
    activeRunIdRef.current = snapshot.active_run_id ?? null;
    if (snapshot.active_run_id) {
      setActiveRun({
        runId: snapshot.active_run_id,
        waitingConfirmation: false,
      });
    } else {
      setActiveRun(null);
    }
    await invoke("set_memory_pool", { entries: snapshot.memory_entries ?? [] });
    await invoke("set_command_history", { entries: snapshot.command_history ?? [] });
  }

  async function persistThreadSnapshot(runIdOverride?: string | null) {
    if (!activeWorkspaceId || !activeThreadId) {
      return;
    }
    const memoryPool = await invoke<{ entries: MemoryEntry[] }>("get_memory_pool");
    const history = await invoke<{ entries: CommandExecution[] }>("get_command_history");
    setMemoryEntries(memoryPool.entries ?? []);
    setCommandHistory(history.entries ?? []);
    await invoke("save_thread_snapshot", {
      workspaceId: activeWorkspaceId,
      threadId: activeThreadId,
      snapshot: stripNullish({
        message_items: messagesRef.current,
        memory_entries: memoryPool.entries ?? [],
        command_history: history.entries ?? [],
        active_run_id: runIdOverride ?? activeRunIdRef.current,
        updated_at: new Date().toISOString(),
      }),
    });
  }

  useEffect(() => {
    if (!tauriAvailable) {
      return;
    }
    void Promise.all([fetchAvailableModels(), refreshLoadedModels(), loadConfig(), loadWorkspaceIndex()]);
  }, [tauriAvailable]);

  useEffect(() => {
    if (!activeWorkspaceId || !activeThreadId) {
      setCurrentThreadMessages([]);
      setMemoryEntries([]);
      setCommandHistory([]);
      return;
    }
    const workspace = workspaceConfig.workspaces.find((entry) => entry.id === activeWorkspaceId);
    const thread = workspace?.threads.find((entry) => entry.id === activeThreadId);
    if (!workspace || !thread) {
      return;
    }
    void invoke("set_active_workspace", { path: workspace.path }).then(() =>
      loadThreadSnapshot(workspace.id, thread.id),
    );
  }, [activeWorkspaceId, activeThreadId, workspaceConfig.workspaces]);

  useEffect(() => {
    if (!activeWorkspaceId || !activeThreadId) {
      return;
    }
    if (persistTimerRef.current) {
      window.clearTimeout(persistTimerRef.current);
    }
    persistTimerRef.current = window.setTimeout(() => {
      void persistThreadSnapshot();
    }, 350);
    return () => {
      if (persistTimerRef.current) {
        window.clearTimeout(persistTimerRef.current);
      }
    };
  }, [activeWorkspaceId, activeThreadId, currentThreadMessages]);

  useEffect(() => {
    if (streamSnapshot.events.length === 0) {
      return;
    }
    for (const event of streamSnapshot.events) {
      applyStreamEvent(event);
    }
    streamStore.clear();
  }, [streamSnapshot]);

  function routeTargets(fromAgentId: string, message: string) {
    return agentConfig.connections
      .filter((connection) => connection.from === fromAgentId && connection.enabled !== false)
      .filter((connection) => {
        const condition = String(connection.condition || "").trim().toLowerCase();
        return !condition || message.toLowerCase().includes(condition);
      })
      .sort((left, right) => (left.priority ?? 128) - (right.priority ?? 128))
      .map((connection) => agentConfig.agents.find((agent) => agent.id === connection.to))
      .filter((agent): agent is Agent => !!agent);
  }

  function agentRoutesToUser(agentId?: string | null) {
    if (!agentId) {
      return false;
    }
    return agentConfig.connections.some(
      (connection) =>
        connection.from === agentId &&
        agentConfig.agents.some(
          (agent) =>
            (agent.type || agent.agent_type) === "user" && agent.id === connection.to,
        ),
    );
  }

  function ensureRuntime(agentId: string) {
    return liveAgentState[agentId] ?? { stage: "idle", detail: "" };
  }

  function scheduleTokenFlush() {
    if (tokenFlushFrameRef.current != null) {
      return;
    }
    tokenFlushFrameRef.current = window.requestAnimationFrame(() => {
      tokenFlushFrameRef.current = null;
      const buffer = tokenBufferRef.current;
      tokenBufferRef.current = {};
      if (Object.keys(buffer).length === 0) {
        return;
      }
      setCurrentThreadMessages((messages) =>
        messages.map((message) => {
          if (message.kind !== "bubble" || !message.agent_id) {
            return message;
          }
          const addition = buffer[message.agent_id];
          if (!addition) {
            return message;
          }
          return {
            ...message,
            content: `${message.content}${addition}`,
          };
        }),
      );
    });
  }

  function upsertBubble(agentId: string, agentName: string) {
    setCurrentThreadMessages((messages) => {
      const existingIndex = messages.findIndex(
        (message) =>
          message.kind === "bubble" &&
          message.agent_id === agentId &&
          !message.content &&
          !message.is_error,
      );
      if (existingIndex !== -1) {
        return messages;
      }
      return [...messages, createBubbleMessage(agentId, agentName, agentRoutesToUser(agentId))];
    });
  }

  function applyStreamEvent(event: StreamEvent) {
    activeRunIdRef.current = event.run_id;
    switch (event.event) {
      case "agent_start":
        startTransition(() => {
          setBackendStatus((status) => {
            const nextActive = new Set(status.activeAgents);
            nextActive.add(event.agent_id);
            return {
              activeAgents: nextActive,
              phase: "running",
              detail: `${event.agent_name} started`,
            };
          });
          setLiveAgentState((state) => ({
            ...state,
            [event.agent_id]: {
              ...state[event.agent_id],
              model_key: event.model_key,
              mode: event.mode,
              is_manager: event.is_manager,
              context_limit: event.context_limit,
              estimated_input_tokens: event.estimated_input_tokens,
              estimated_remaining_tokens: event.estimated_remaining_tokens,
              stage: "starting",
              detail: "Starting generation",
            },
          }));
          upsertBubble(event.agent_id, event.agent_name);
        });
        break;
      case "agent_status":
        startTransition(() => {
          setBackendStatus((status) => ({ ...status, detail: event.detail }));
          setLiveAgentState((state) => ({
            ...state,
            [event.agent_id]: {
              ...state[event.agent_id],
              stage: event.stage,
              detail: event.detail,
            },
          }));
          setCurrentThreadMessages((messages) =>
            messages.map((message) =>
              message.kind === "bubble" && message.agent_id === event.agent_id
                ? {
                    ...message,
                    runtime: {
                      ...(message.runtime ?? {}),
                      stage: event.stage,
                      detail: event.detail,
                    },
                  }
                : message,
            ),
          );
        });
        break;
      case "agent_metrics":
        startTransition(() => {
          setLiveAgentState((state) => ({
            ...state,
            [event.agent_id]: {
              ...state[event.agent_id],
              stage: event.stage,
              estimated_output_tokens: event.estimated_output_tokens,
              input_tokens: event.input_tokens,
              output_tokens: event.output_tokens,
              reasoning_output_tokens: event.reasoning_output_tokens,
              tokens_per_second: event.tokens_per_second,
              time_to_first_token_seconds: event.time_to_first_token_seconds,
            },
          }));
          setCurrentThreadMessages((messages) =>
            messages.map((message) =>
              message.kind === "bubble" && message.agent_id === event.agent_id
                ? {
                    ...message,
                    runtime: {
                      ...(message.runtime ?? {}),
                      stage: event.stage,
                      estimated_output_tokens: event.estimated_output_tokens,
                      output_tokens: event.output_tokens,
                      tokens_per_second: event.tokens_per_second,
                      time_to_first_token_seconds: event.time_to_first_token_seconds,
                    },
                  }
                : message,
            ),
          );
        });
        break;
      case "tool_call":
        setCurrentThreadMessages((messages) =>
          messages.map((message) =>
            message.kind === "bubble" && message.agent_id === event.agent_id
              ? {
                  ...message,
                  tools: [
                    ...message.tools,
                    {
                      tool_name: event.tool_name,
                      args: event.args,
                      result: "",
                      status: "pending",
                      semantic: false,
                    },
                  ],
                }
              : message,
          ),
        );
        break;
      case "tool_result":
        setCurrentThreadMessages((messages) =>
          messages.map((message) => {
            if (message.kind !== "bubble" || message.agent_id !== event.agent_id) {
              return message;
            }
            const tools = [...message.tools];
            const lastPending = [...tools]
              .reverse()
              .findIndex((tool) => tool.status === "pending");
            if (lastPending === -1) {
              return message;
            }
            const idx = tools.length - 1 - lastPending;
            tools[idx] = {
              ...tools[idx],
              result: event.result,
              status: event.result.startsWith("[tool call failed") ? "error" : "done",
            };
            return { ...message, tools };
          }),
        );
        break;
      case "token":
        tokenBufferRef.current[event.agent_id] =
          (tokenBufferRef.current[event.agent_id] ?? "") + event.content;
        scheduleTokenFlush();
        break;
      case "agent_end":
        startTransition(() => {
          setBackendStatus((status) => {
            const nextActive = new Set(status.activeAgents);
            nextActive.delete(event.agent_id);
            return {
              activeAgents: nextActive,
              phase: nextActive.size ? "running" : "idle",
              detail: nextActive.size
                ? `${nextActive.size} agent(s) still running`
                : "Waiting for downstream routes…",
            };
          });
          setLiveAgentState((state) => ({
            ...state,
            [event.agent_id]: {
              ...state[event.agent_id],
              stage: "done",
              detail: "Generation complete",
            },
          }));
        });
        break;
      case "error":
        startTransition(() => {
          setBackendStatus((status) => {
            const nextActive = new Set(status.activeAgents);
            nextActive.delete(event.agent_id);
            return {
              activeAgents: nextActive,
              phase: "error",
              detail: event.message,
            };
          });
          setCurrentThreadMessages((messages) => {
            const existing = messages.find(
              (message) => message.kind === "bubble" && message.agent_id === event.agent_id,
            );
            if (!existing) {
              return [
                ...messages,
                {
                  ...createBubbleMessage(
                    event.agent_id,
                    event.agent_name,
                    agentRoutesToUser(event.agent_id),
                  ),
                  content: event.message,
                  is_error: true,
                },
              ];
            }
            return messages.map((message) =>
              message.kind === "bubble" && message.agent_id === event.agent_id
                ? { ...message, content: event.message, is_error: true }
                : message,
            );
          });
        });
        break;
      case "run_limit_reached":
        setActiveRun({
          runId: event.run_id,
          waitingConfirmation: true,
          limitMessage: `${event.kind}: ${event.observed} / ${event.limit}`,
        });
        break;
      case "run_waiting_confirmation":
        setActiveRun({
          runId: event.run_id,
          waitingConfirmation: true,
          limitMessage: event.message,
        });
        void persistThreadSnapshot(event.run_id);
        break;
      case "run_resumed":
        setActiveRun({ runId: event.run_id, waitingConfirmation: false });
        break;
      case "run_cancelled":
        setActiveRun(null);
        setRunUsage(null);
        void persistThreadSnapshot(null);
        break;
      case "run_checkpoint_saved":
        setSaveNotice(`Checkpoint saved: ${event.checkpoint}`);
        window.setTimeout(() => setSaveNotice(""), 1500);
        break;
      case "run_usage_update":
        setRunUsage({
          llm_calls: event.llm_calls,
          tool_calls: event.tool_calls,
          spawned_agents: event.spawned_agents,
          streamed_tokens: event.streamed_tokens,
          embedding_calls: event.embedding_calls,
          wall_clock_seconds: event.wall_clock_seconds,
        });
        break;
      case "done":
        setActiveRun(null);
        setRunUsage(null);
        activeRunIdRef.current = null;
        void refreshLoadedModels();
        void persistThreadSnapshot(null);
        startTransition(() =>
          setBackendStatus({ activeAgents: new Set(), phase: "idle", detail: "Backend idle." }),
        );
        break;
      default:
        break;
    }
  }

  async function submitMessage() {
    const message = chatInput.trim();
    if (!message || !activeWorkspaceId) {
      return;
    }

    let workspaceId = activeWorkspaceId;
    let threadId = activeThreadId;
    if (!threadId) {
      const newThread: WorkspaceThreadIndex = {
        id: generateId("thread"),
        name: message.slice(0, 20) + (message.length > 20 ? "…" : ""),
      };
      const nextConfig = {
        workspaces: workspaceConfig.workspaces.map((workspace) =>
          workspace.id === workspaceId
            ? { ...workspace, _open: true, threads: [...workspace.threads, newThread] }
            : workspace,
        ),
      };
      setWorkspaceConfig(nextConfig);
      await saveWorkspaceIndex(nextConfig);
      setActiveThreadId(newThread.id);
      threadId = newThread.id;
    }

    setCurrentThreadMessages((messages) => [
      ...messages,
      createTextMessage("You", message, true, false),
    ]);
    routeTargets("user", message).forEach((agent) => upsertBubble(agent.id, agent.name));
    setChatInput("");

    const channel = new Channel<StreamEvent>();
    channel.onmessage = (event) => {
      streamStore.publish(event);
    };
    currentRunChannel.current = channel;
    setBackendStatus({
      activeAgents: new Set(),
      phase: "routing",
      detail: "Dispatching message through the graph…",
    });
    const runId = await invoke<string>("route_message", {
      fromAgentId: "user",
      message,
      workspaceId,
      threadId,
      onEvent: channel,
    });
    setActiveRun({ runId, waitingConfirmation: false });
  }

  async function continueRun() {
    if (!activeRun) {
      return;
    }
    await invoke("continue_route_run", { runId: activeRun.runId });
  }

  async function cancelRun() {
    if (!activeRun) {
      return;
    }
    await invoke("cancel_route_run", { runId: activeRun.runId });
  }

  function updateAgent(index: number, patch: Partial<Agent>) {
    setAgentConfig((config) => {
      const nextAgents = [...config.agents];
      nextAgents[index] = { ...nextAgents[index], ...patch };
      return { ...config, agents: nextAgents };
    });
  }

  async function addWorkspace() {
    if (!tauriAvailable) {
      return;
    }
    const picked = await invoke<string | null>("pick_folder");
    if (!picked) {
      return;
    }
    const name = picked.split("/").filter(Boolean).pop() ?? picked;
    const newWorkspace: WorkspaceEntry = {
      id: generateId("ws"),
      name,
      path: picked,
      threads: [{ id: generateId("thread"), name: "Thread 1" }],
      _open: true,
    };
    const nextConfig = { workspaces: [...workspaceConfig.workspaces, newWorkspace] };
    setWorkspaceConfig(nextConfig);
    await saveWorkspaceIndex(nextConfig);
  }

  async function loadModelForAgent(agent: Agent) {
    if (!tauriAvailable || !agent.model_key) {
      return;
    }
    await invoke("load_model", {
      config: {
        model: agent.model_key,
        ...(agent.load_config ?? {}),
      },
    });
    await refreshLoadedModels();
  }

  /* ── JSX ───────────────────────────────────────────────── */

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: "var(--w98-gray)" }}>

      {/* ── Sidebar ── */}
      <aside
        className="w98-window"
        style={{
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
          width: sidebarCollapsed ? "28px" : "210px",
          minWidth: sidebarCollapsed ? "28px" : "210px",
        }}
      >
        <div className="w98-titlebar">
          {!sidebarCollapsed && <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>Workspaces</span>}
          <button
            className="w98-btn w98-btn-sm"
            style={{ marginLeft: sidebarCollapsed ? "auto" : "auto", minWidth: 0 }}
            onClick={() => setSidebarCollapsed((v) => !v)}
            type="button"
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? "»" : "«"}
          </button>
        </div>

        {!sidebarCollapsed && (
          <>
            <div style={{ display: "flex", gap: "2px", padding: "3px", borderBottom: "1px solid var(--w98-dark)", flexShrink: 0 }}>
              <input
                className="w98-input"
                style={{ flex: 1 }}
                value={threadSearch}
                onChange={(e) => setThreadSearch(e.target.value)}
                placeholder="Filtrer…"
              />
              <button
                className="w98-btn w98-btn-sm"
                disabled={!tauriAvailable}
                onClick={() => void addWorkspace()}
                type="button"
                title="Add workspace"
              >
                +
              </button>
            </div>

            <div style={{ flex: 1, overflowY: "auto", padding: "2px" }}>
              {filteredWorkspaces.map((workspace) => (
                <div key={workspace.id}>
                  <div
                    className={`w98-tree-item${workspace.id === activeWorkspaceId ? " selected" : ""}`}
                    onClick={() => {
                      setActiveWorkspaceId(workspace.id);
                      setActiveThreadId(null);
                      void invoke("set_active_workspace", { path: workspace.path });
                    }}
                  >
                    <span style={{ fontSize: "9px", flexShrink: 0 }}>{workspace._open ? "▽" : "▷"}</span>
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{workspace.name}</span>
                    <span style={{ color: "var(--w98-dark)", fontSize: "10px", flexShrink: 0 }}>{workspace.threads.length}</span>
                  </div>
                  <div style={{ fontSize: "10px", color: "var(--w98-dark)", paddingLeft: "14px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {workspace.path}
                  </div>
                  {workspace.threads.map((thread) => (
                    <div
                      key={thread.id}
                      className={`w98-tree-item${thread.id === activeThreadId && workspace.id === activeWorkspaceId ? " selected" : ""}`}
                      style={{ paddingLeft: "12px" }}
                      onClick={() => {
                        setActiveWorkspaceId(workspace.id);
                        setActiveThreadId(thread.id);
                      }}
                    >
                      <span style={{ fontSize: "9px", flexShrink: 0 }}>⎇</span>
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{thread.name}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </>
        )}
      </aside>

      {/* ── Main ── */}
      <main style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>

        {/* Tab bar */}
        <div className="w98-tabbar">
          {(["chat", "agents", "routing"] as TabKey[]).map((tab) => (
            <button
              key={tab}
              className={activeTab === tab ? "w98-tab-active" : "w98-tab"}
              onClick={() => setActiveTab(tab)}
              type="button"
            >
              {tab[0].toUpperCase() + tab.slice(1)}
            </button>
          ))}
          {saveNotice && (
            <span className="w98-notice" style={{ marginLeft: "auto", alignSelf: "center" }}>
              {saveNotice}
            </span>
          )}
        </div>

        {/* Tab panel */}
        <div className="w98-tabpanel" style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>

          {/* ══ CHAT TAB ══ */}
          {activeTab === "chat" && (
            <section style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>

              {/* Status bar */}
              <div className="w98-statusbar">
                <div className="w98-statusbar-cell">
                  {backendStatus.activeAgents.size
                    ? `${backendStatus.activeAgents.size} active agent(s)`
                    : "Idle"}
                </div>
                <div className="w98-statusbar-cell" style={{ flex: 1, overflow: "hidden" }}>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>
                    {backendStatus.detail}
                  </span>
                </div>
                <div className="w98-statusbar-cell">
                  LM Studio: {loadedModels.reduce((s, m) => s + (m.instances?.length ?? 0), 0)} inst.
                </div>
              </div>

              {/* Confirmation bar */}
              {activeRun?.waitingConfirmation && (
                <div style={{ padding: "4px 8px", background: "#FFFFC0", borderBottom: "1px solid var(--w98-dark)", display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
                  <span style={{ flex: 1, fontSize: "11px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {activeRun.limitMessage}
                  </span>
                  <button className="w98-btn w98-btn-sm" onClick={() => void continueRun()} type="button">Continue</button>
                  <button className="w98-btn w98-btn-sm" onClick={() => void cancelRun()} type="button">Stop</button>
                </div>
              )}

              {/* Messages + Runtime grid */}
              <div style={{ display: "flex", flex: 1, minHeight: 0 }}>

                {/* Message list */}
                <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0, minHeight: 0 }}>
                  <div style={{ flex: 1, overflowY: "auto", padding: "6px 8px", background: "var(--w98-white)" }}>
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>

                      {currentThreadMessages.length === 0 && (
                        <div style={{ border: "1px solid var(--w98-dark)", padding: "10px", color: "var(--w98-dark)", background: "var(--w98-gray)", fontSize: "11px" }}>
                          Select a workspace and send a message to start a thread.
                        </div>
                      )}

                      {hiddenThreadMessageCount > 0 && (
                        <div style={{ border: "1px solid var(--w98-dark)", padding: "6px 8px", background: "var(--w98-gray)", fontSize: "11px" }}>
                          <div>{hiddenThreadMessageCount.toLocaleString("en-US")} message(s) hidden to keep rendering smooth.</div>
                          <button
                            className="w98-btn w98-btn-sm"
                            style={{ marginTop: "4px" }}
                            onClick={() => setShowFullThreadHistory(true)}
                            type="button"
                          >
                            Show full history
                          </button>
                        </div>
                      )}

                      {showFullThreadHistory && currentThreadMessages.length > COLLAPSED_THREAD_RENDER_LIMIT && (
                        <div style={{ border: "1px solid var(--w98-dark)", padding: "6px 8px", background: "var(--w98-gray)", fontSize: "11px" }}>
                          <div>Full history is visible. Rendering may slow down on large threads.</div>
                          <button
                            className="w98-btn w98-btn-sm"
                            style={{ marginTop: "4px" }}
                            onClick={() => setShowFullThreadHistory(false)}
                            type="button"
                          >
                            Return to the last {COLLAPSED_THREAD_RENDER_LIMIT} messages
                          </button>
                        </div>
                      )}

                      {visibleThreadMessages.map((message, index) => (
                        <ThreadMessageCard
                          key={`${hiddenThreadMessageCount + index}-${message.agent_id ?? message.agent_name}-${message.kind}`}
                          message={message}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Input form */}
                  <form
                    style={{ display: "flex", gap: "4px", padding: "4px", borderTop: "2px solid", borderColor: "var(--w98-darker) var(--w98-white) var(--w98-white) var(--w98-darker)", background: "var(--w98-gray)", flexShrink: 0 }}
                    onSubmit={(e) => { e.preventDefault(); void submitMessage(); }}
                  >
                    <input
                      className="w98-input"
                      style={{ flex: 1 }}
                      disabled={!activeWorkspaceId}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder={
                        activeWorkspaceId
                          ? selectedThread
                            ? `Message - ${selectedThread.name}`
                            : "Send a message to create a thread..."
                          : "Select a workspace before sending..."
                      }
                      value={chatInput}
                    />
                    <button
                      className="w98-btn"
                      disabled={!activeWorkspaceId || !chatInput.trim()}
                      type="submit"
                    >
                      Send
                    </button>
                  </form>
                </div>

                {/* Runtime panel */}
                <aside
                  style={{
                    width: "190px",
                    minWidth: "190px",
                    overflowY: "auto",
                    padding: "4px",
                    borderLeft: "2px solid",
                    borderColor: "var(--w98-darker) var(--w98-white) var(--w98-white) var(--w98-darker)",
                    background: "var(--w98-gray)",
                  }}
                >
                  <div style={{ fontWeight: "bold", borderBottom: "1px solid var(--w98-dark)", marginBottom: "6px", paddingBottom: "2px" }}>
                    Runtime
                  </div>

                  {runUsage && (
                    <div className="w98-group" style={{ marginBottom: "8px" }}>
                      <span className="w98-group-title">Budgets</span>
                      <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
                        {([
                          { label: "LLM calls", value: runUsage.llm_calls, limit: agentConfig.run_budgets.llm_calls_per_window, unit: "" },
                          { label: "Tool calls", value: runUsage.tool_calls, limit: agentConfig.run_budgets.tool_calls_per_window, unit: "" },
                          { label: "Tokens", value: runUsage.streamed_tokens, limit: agentConfig.run_budgets.streamed_tokens_per_window, unit: "" },
                          { label: "Clock", value: runUsage.wall_clock_seconds, limit: agentConfig.run_budgets.wall_clock_seconds_per_window, unit: "s" },
                        ] as const).map(({ label, value, limit, unit }) => {
                          const pct = Math.min(100, Math.round((value / limit) * 100));
                          const hot = pct >= 80;
                          return (
                            <div key={label}>
                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", marginBottom: "1px" }}>
                                <span>{label}</span>
                                <span style={{ color: hot ? "#C08000" : "var(--w98-black)", fontWeight: hot ? "bold" : "normal" }}>
                                  {value.toLocaleString("en-US")}{unit} / {limit.toLocaleString("en-US")}{unit}
                                </span>
                              </div>
                              <div className="w98-progress">
                                <div className={hot ? "w98-progress-bar-warn" : "w98-progress-bar"} style={{ width: `${pct}%` }} />
                              </div>
                            </div>
                          );
                        })}
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", marginTop: "2px", alignItems: "center" }}>
                          <span>Embeddings</span>
                          <span className="w98-badge">{runUsage.embedding_calls}</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {agentConfig.agents
                    .filter((agent) => (agent.type || agent.agent_type) !== "user")
                    .map((agent) => {
                      const runtime = ensureRuntime(agent.id);
                      const info = agent.model_key ? modelDetails[agent.model_key] : undefined;
                      const loaded = loadedModels.find((m) => m.key === agent.model_key);
                      return (
                        <div key={agent.id} className="w98-raised" style={{ padding: "4px", marginBottom: "6px", background: "var(--w98-gray)" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2px", gap: "4px" }}>
                            <span style={{ fontWeight: "bold", fontSize: "11px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                              {agent.name}
                            </span>
                            <span className="w98-badge">{runtime.stage || "idle"}</span>
                          </div>
                          <div style={{ fontSize: "10px", color: "var(--w98-dark)", marginBottom: "3px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {info?.display_name || agent.model_key || "No model"}
                          </div>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: "2px" }}>
                            <span className="w98-badge">ctx {formatTokenCount(runtime.estimated_input_tokens)}/{formatTokenCount(runtime.context_limit)}</span>
                            <span className="w98-badge">gen {formatTokenCount(runtime.estimated_output_tokens)}</span>
                            {formatSpeed(runtime.tokens_per_second) && (
                              <span className="w98-badge">{formatSpeed(runtime.tokens_per_second)}</span>
                            )}
                          </div>
                          {loaded?.instances?.length ? (
                            <div style={{ marginTop: "4px", display: "flex", flexWrap: "wrap", gap: "2px" }}>
                              {loaded.instances.map((inst) => (
                                <button
                                  className="w98-btn w98-btn-sm"
                                  key={inst.instance_id}
                                  onClick={() => invoke("unload_model", { instanceId: inst.instance_id }).then(() => refreshLoadedModels())}
                                  type="button"
                                >
                                  Unload {formatTokenCount(inst.context_length)}
                                </button>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                </aside>
              </div>
            </section>
          )}

          {/* ══ AGENTS TAB ══ */}
          {activeTab === "agents" && (
            <section className="agents-shell">
              <div className="agents-page">
                <div className="w98-window agents-hero">
                  <div className="agents-hero-row">
                    <div className="agents-hero-copy">
                      <div className="agents-kicker">Agent Studio</div>
                      <h2 className="agents-title">Compose the crew before the thread gets loud.</h2>
                      <p className="agents-copy">
                        Keep prompts, model load settings, routing budgets, and behavioral guardrails visible in one place.
                        The left side owns individual agents; the right rail owns global policy.
                      </p>
                    </div>
                    <div className="agents-toolbar">
                      <button
                        className="w98-btn"
                        onClick={() =>
                          setAgentConfig((config) => ({
                            ...config,
                            agents: [
                              ...config.agents,
                              {
                                id: generateId("agent"),
                                name: "New Agent",
                                type: "model",
                                model_key: availableModels[0]?.key ?? "",
                                model_type: availableModels[0]?.model_type ?? "llm",
                                role: "",
                                load_config: {},
                                mode: "stay_awake",
                                armed: true,
                                is_manager: false,
                              },
                            ],
                          }))
                        }
                        type="button"
                      >
                        + Add agent
                      </button>
                      <button className="w98-btn" disabled={!tauriAvailable} onClick={() => void saveConfig()} type="button">
                        Save
                      </button>
                      <button className="w98-btn" disabled={!tauriAvailable} onClick={() => void loadConfig()} type="button">
                        Reload
                      </button>
                    </div>
                  </div>

                  {!tauriAvailable && (
                    <div className="w98-notice agents-browser-notice">
                      Browser mode: layout preview only. Tauri backend actions are disabled in `npm run dev`.
                    </div>
                  )}

                  <div className="agents-summary-grid">
                    <div className="w98-raised agents-stat">
                      <span className="agents-stat-label">Active agents</span>
                      <strong className="agents-stat-value">
                        {nonUserAgents.filter((agent) => agent.armed !== false).length}
                      </strong>
                      <span className="agents-stat-meta">{nonUserAgents.length} configured</span>
                    </div>
                    <div className="w98-raised agents-stat">
                      <span className="agents-stat-label">Managers</span>
                      <strong className="agents-stat-value">
                        {nonUserAgents.filter((agent) => !!agent.is_manager).length}
                      </strong>
                      <span className="agents-stat-meta">orchestration enabled</span>
                    </div>
                    <div className="w98-raised agents-stat">
                      <span className="agents-stat-label">Loaded models</span>
                      <strong className="agents-stat-value">{loadedInstanceCount}</strong>
                      <span className="agents-stat-meta">{loadedModels.length} model key(s)</span>
                    </div>
                    <div className="w98-raised agents-stat">
                      <span className="agents-stat-label">Behavior packs</span>
                      <strong className="agents-stat-value">
                        {agentConfig.behavior_triggers.behaviors.filter((behavior) => behavior.enabled).length}
                      </strong>
                      <span className="agents-stat-meta">
                        {agentConfig.behavior_triggers.enabled ? "global triggers on" : "global triggers off"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="agents-layout">
                  {agentConfig.agents.map((agent, index) => {
                    const info = agent.model_key ? modelDetails[agent.model_key] : undefined;
                    const isUser = (agent.type || agent.agent_type) === "user";
                    const selectedMode = agent.mode || "stay_awake";

                    return (
                      <div key={agent.id} className="w98-window agent-card">
                        <div className="w98-titlebar">
                          <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {agent.name}
                          </span>
                          <span className="w98-badge">{isUser ? "User" : agent.is_manager ? "Manager" : "Worker"}</span>
                          {!isUser && (
                            <span
                              className="w98-badge"
                              style={{
                                background: agent.armed === false ? "var(--w98-gray)" : "#004400",
                                color: agent.armed === false ? "var(--w98-dark)" : "var(--w98-white)",
                                borderColor: agent.armed === false ? undefined : "#006600 #002200 #002200 #006600",
                              }}
                            >
                              {agent.armed === false ? "Disarmed" : "Armed"}
                            </span>
                          )}
                        </div>

                        <div className="agent-card-body">
                          <div className="agent-meta-row">
                            <span className="w98-badge">ID {agent.id}</span>
                            {!isUser && (
                              <>
                                <span className="w98-badge">{info?.display_name || agent.model_key || "No model selected"}</span>
                                <span className="w98-badge">{selectedMode === "stay_awake" ? "Pinned in memory" : "Load on demand"}</span>
                              </>
                            )}
                          </div>

                          {isUser ? (
                            <div className="agent-user-panel">
                              <p className="agents-copy" style={{ margin: 0 }}>
                                The user node is the manual entrypoint for a thread and the sink for any response routed back to a human.
                              </p>
                              <label className="agent-field">
                                <span className="w98-label">Visible name</span>
                                <input
                                  className="w98-input"
                                  onChange={(e) => updateAgent(index, { name: e.target.value })}
                                  value={agent.name}
                                />
                              </label>
                            </div>
                          ) : (
                            <>
                              <div className="agent-section-grid">
                                <section className="agent-section">
                                  <div className="agent-section-header">
                                    <div>
                                      <div className="agent-section-kicker">Identity</div>
                                      <div className="agent-section-title">Who this agent is</div>
                                    </div>
                                    <label className="agent-inline-toggle">
                                      <input
                                        type="checkbox"
                                        checked={!!agent.is_manager}
                                        onChange={(e) => updateAgent(index, { is_manager: e.target.checked })}
                                      />
                                      Manager
                                    </label>
                                  </div>

                                  <div className="agent-field-grid">
                                    <label className="agent-field agent-field-wide">
                                      <span className="w98-label">Name</span>
                                      <input
                                        className="w98-input"
                                        onChange={(e) => updateAgent(index, { name: e.target.value })}
                                        value={agent.name}
                                      />
                                    </label>

                                    <label className="agent-field">
                                      <span className="w98-label">Model</span>
                                      <select
                                        className="w98-select"
                                        onChange={(e) => {
                                          const selected = availableModels.find((model) => model.key === e.target.value);
                                          updateAgent(index, {
                                            model_key: e.target.value,
                                            model_type: selected?.model_type ?? "llm",
                                          });
                                        }}
                                        value={agent.model_key ?? ""}
                                      >
                                        {!availableModels.length && <option value="">No model available</option>}
                                        {availableModels.map((model) => (
                                          <option key={model.key} value={model.key}>
                                            {model.display_name || model.key}
                                          </option>
                                        ))}
                                      </select>
                                      <span className="agent-field-hint">
                                        {info?.model_type?.toUpperCase() || "LLM"} · {info?.params_string || "unknown size"}
                                      </span>
                                    </label>

                                    <div className="agent-field">
                                      <span className="w98-label">Mode</span>
                                      <div className="agent-mode-row">
                                        {["stay_awake", "on_the_fly"].map((mode) => (
                                          <button
                                            key={mode}
                                            className="w98-btn w98-btn-sm"
                                            style={{
                                              background: selectedMode === mode ? "var(--w98-navy)" : undefined,
                                              color: selectedMode === mode ? "var(--w98-white)" : undefined,
                                            }}
                                            onClick={() => updateAgent(index, { mode })}
                                            type="button"
                                          >
                                            {mode === "stay_awake" ? "Active" : "On demand"}
                                          </button>
                                        ))}
                                      </div>
                                    </div>
                                  </div>
                                </section>

                                <section className="agent-section">
                                  <div className="agent-section-header">
                                    <div>
                                      <div className="agent-section-kicker">Prompt</div>
                                      <div className="agent-section-title">Role and instructions</div>
                                    </div>
                                  </div>
                                  <label className="agent-field">
                                    <span className="w98-label">System prompt</span>
                                    <textarea
                                      className="w98-textarea w98-textarea-sans"
                                      style={{ minHeight: "120px" }}
                                      onChange={(e) => updateAgent(index, { role: e.target.value })}
                                      value={agent.role ?? ""}
                                    />
                                  </label>
                                </section>
                              </div>

                              <section className="agent-section">
                                <div className="agent-section-header">
                                  <div>
                                    <div className="agent-section-kicker">Load profile</div>
                                    <div className="agent-section-title">Memory and acceleration</div>
                                  </div>
                                </div>

                                <div className="agent-load-grid">
                                  <label className="agent-field">
                                    <span className="w98-label">Context</span>
                                    <input
                                      className="w98-input"
                                      type="number"
                                      value={agent.load_config?.context_length ?? ""}
                                      onChange={(e) =>
                                        updateAgent(index, {
                                          load_config: {
                                            ...(agent.load_config ?? {}),
                                            context_length: e.target.value ? Number(e.target.value) : null,
                                          },
                                        })
                                      }
                                    />
                                  </label>
                                  <label className="agent-field">
                                    <span className="w98-label">Eval batch</span>
                                    <input
                                      className="w98-input"
                                      type="number"
                                      value={agent.load_config?.eval_batch_size ?? ""}
                                      onChange={(e) =>
                                        updateAgent(index, {
                                          load_config: {
                                            ...(agent.load_config ?? {}),
                                            eval_batch_size: e.target.value ? Number(e.target.value) : null,
                                          },
                                        })
                                      }
                                    />
                                  </label>
                                  <label className="agent-field">
                                    <span className="w98-label">Experts</span>
                                    <input
                                      className="w98-input"
                                      type="number"
                                      value={agent.load_config?.num_experts ?? ""}
                                      onChange={(e) =>
                                        updateAgent(index, {
                                          load_config: {
                                            ...(agent.load_config ?? {}),
                                            num_experts: e.target.value ? Number(e.target.value) : null,
                                          },
                                        })
                                      }
                                    />
                                  </label>
                                </div>

                                <div className="agent-toggle-grid">
                                  <label className="agent-inline-toggle">
                                    <input
                                      type="checkbox"
                                      checked={!!agent.load_config?.flash_attention}
                                      onChange={(e) =>
                                        updateAgent(index, {
                                          load_config: { ...(agent.load_config ?? {}), flash_attention: e.target.checked },
                                        })
                                      }
                                    />
                                    Flash attention
                                  </label>
                                  <label className="agent-inline-toggle">
                                    <input
                                      type="checkbox"
                                      checked={!!agent.load_config?.offload_kv_cache_to_gpu}
                                      onChange={(e) =>
                                        updateAgent(index, {
                                          load_config: { ...(agent.load_config ?? {}), offload_kv_cache_to_gpu: e.target.checked },
                                        })
                                      }
                                    />
                                    GPU KV offload
                                  </label>
                                </div>
                              </section>

                              <div className="agent-action-row">
                                <button
                                  className="w98-btn w98-btn-sm"
                                  disabled={!tauriAvailable || !agent.model_key}
                                  onClick={() => void loadModelForAgent(agent)}
                                  type="button"
                                >
                                  Load now
                                </button>
                                <button
                                  className="w98-btn w98-btn-sm"
                                  onClick={() => updateAgent(index, { armed: !(agent.armed !== false) })}
                                  type="button"
                                >
                                  {agent.armed === false ? "Arm" : "Disarm"}
                                </button>
                                <button
                                  className="w98-btn w98-btn-sm"
                                  disabled={agent.armed !== false}
                                  onClick={() =>
                                    setAgentConfig((config) => ({
                                      ...config,
                                      agents: config.agents.filter((entry) => entry.id !== agent.id),
                                      connections: config.connections.filter(
                                        (connection) => connection.from !== agent.id && connection.to !== agent.id,
                                      ),
                                    }))
                                  }
                                  style={{ color: agent.armed !== false ? undefined : "#CC0000" }}
                                  type="button"
                                >
                                  Delete
                                </button>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    );
                  })}

                  <div className="w98-window agents-panel-window">
                      <div className="w98-titlebar">Command sandbox</div>
                      <div className="agents-panel-body">
                        <p className="agents-copy" style={{ marginTop: 0 }}>
                          Define hard command boundaries that every agent inherits before routing kicks in.
                        </p>
                        <label className="agent-field">
                          <span className="w98-label">Denylist (one per line)</span>
                          <textarea
                            className="w98-textarea"
                            style={{ minHeight: "72px" }}
                            onChange={(e) =>
                              setAgentConfig((config) => ({
                                ...config,
                                command_policy: { ...config.command_policy, denylist: e.target.value.split("\n") },
                              }))
                            }
                            placeholder="Blocked prefixes, one per line"
                            value={agentConfig.command_policy.denylist.join("\n")}
                          />
                        </label>
                        <label className="agent-field">
                          <span className="w98-label">Allowlist (one per line)</span>
                          <textarea
                            className="w98-textarea"
                            style={{ minHeight: "72px" }}
                            onChange={(e) =>
                              setAgentConfig((config) => ({
                                ...config,
                                command_policy: { ...config.command_policy, allowlist: e.target.value.split("\n") },
                              }))
                            }
                            placeholder="Allowed prefixes, one per line"
                            value={agentConfig.command_policy.allowlist.join("\n")}
                          />
                        </label>
                      </div>
                    </div>

                  <div className="w98-window agents-panel-window">
                      <div className="w98-titlebar">Redundancy detection</div>
                      <div className="agents-panel-body">
                        <label className="agent-inline-toggle">
                          <input
                            type="checkbox"
                            checked={agentConfig.redundancy_detection.enabled}
                            onChange={(e) =>
                              setAgentConfig((config) => ({
                                ...config,
                                redundancy_detection: { ...config.redundancy_detection, enabled: e.target.checked },
                              }))
                            }
                          />
                          Enable detection
                        </label>
                        <div className="agent-load-grid">
                          <label className="agent-field">
                            <span className="w98-label">Similarity threshold</span>
                            <input
                              className="w98-input"
                              type="number"
                              min={MIN_SEMANTIC_REDUNDANCY_THRESHOLD}
                              max={MAX_SEMANTIC_REDUNDANCY_THRESHOLD}
                              step="0.01"
                              value={agentConfig.redundancy_detection.semantic_similarity_threshold}
                              onChange={(e) =>
                                setAgentConfig((config) => ({
                                  ...config,
                                  redundancy_detection: {
                                    ...config.redundancy_detection,
                                    semantic_similarity_threshold: Number(e.target.value),
                                  },
                                }))
                              }
                            />
                          </label>
                          <label className="agent-field">
                            <span className="w98-label">Max redundant retries</span>
                            <input
                              className="w98-input"
                              type="number"
                              value={agentConfig.redundancy_detection.max_redundant_audit_retries}
                              onChange={(e) =>
                                setAgentConfig((config) => ({
                                  ...config,
                                  redundancy_detection: {
                                    ...config.redundancy_detection,
                                    max_redundant_audit_retries: Number(e.target.value),
                                  },
                                }))
                              }
                            />
                          </label>
                        </div>
                      </div>
                    </div>

                  <div className="w98-window agents-panel-window">
                      <div className="w98-titlebar">Run budgets</div>
                      <div className="agents-panel-body">
                        <label className="agent-inline-toggle">
                          <input
                            type="checkbox"
                            checked={agentConfig.run_budgets.enabled}
                            onChange={(e) =>
                              setAgentConfig((config) => ({
                                ...config,
                                run_budgets: { ...config.run_budgets, enabled: e.target.checked },
                              }))
                            }
                          />
                          Enable budgets
                        </label>
                        <div className="agent-load-grid">
                          {(["llm_calls_per_window", "tool_calls_per_window", "spawned_agents_per_window", "streamed_tokens_per_window", "wall_clock_seconds_per_window"] as const).map((key) => (
                            <label className="agent-field" key={key}>
                              <span className="w98-label">{key.replace(/_/g, " ")}</span>
                              <input
                                className="w98-input"
                                type="number"
                                value={agentConfig.run_budgets[key]}
                                onChange={(e) =>
                                  setAgentConfig((config) => ({
                                    ...config,
                                    run_budgets: { ...config.run_budgets, [key]: Number(e.target.value) },
                                  }))
                                }
                              />
                            </label>
                          ))}
                        </div>
                        <label className="agent-field">
                          <span className="w98-label">Affected behaviors</span>
                          <textarea
                            className="w98-textarea"
                            style={{ minHeight: "44px" }}
                            placeholder="(empty = all runs)"
                            onChange={(e) =>
                              setAgentConfig((config) => ({
                                ...config,
                                run_budgets: { ...config.run_budgets, applies_to_behaviors: e.target.value.split("\n") },
                              }))
                            }
                            value={agentConfig.run_budgets.applies_to_behaviors.join("\n")}
                          />
                        </label>
                        <label className="agent-field">
                          <span className="w98-label">When a limit is hit</span>
                          <select
                            className="w98-select"
                            value={agentConfig.run_budgets.on_limit}
                            onChange={(e) =>
                              setAgentConfig((config) => ({
                                ...config,
                                run_budgets: {
                                  ...config.run_budgets,
                                  on_limit: e.target.value as "pause" | "summarize" | "stop",
                                },
                              }))
                            }
                          >
                            <option value="summarize">Summarize - auto summary then stop</option>
                            <option value="pause">Pause - wait for user confirmation</option>
                            <option value="stop">Immediate stop - no summary</option>
                          </select>
                        </label>
                        {agentConfig.run_budgets.on_limit === "summarize" && (
                          <div className="agents-inline-stack">
                            <label className="agent-inline-toggle">
                              <input
                                type="checkbox"
                                checked={agentConfig.run_budgets.summarization.enabled}
                                onChange={(e) =>
                                  setAgentConfig((config) => ({
                                    ...config,
                                    run_budgets: {
                                      ...config.run_budgets,
                                      summarization: {
                                        ...config.run_budgets.summarization,
                                        enabled: e.target.checked,
                                      },
                                    },
                                  }))
                                }
                              />
                              Enable auto-summary
                            </label>
                            <label className="agent-field">
                              <span className="w98-label">Summary model</span>
                              <select
                                className="w98-select"
                                value={agentConfig.run_budgets.summarization.model_key ?? ""}
                                onChange={(e) =>
                                  setAgentConfig((config) => ({
                                    ...config,
                                    run_budgets: {
                                      ...config.run_budgets,
                                      summarization: {
                                        ...config.run_budgets.summarization,
                                        model_key: e.target.value || null,
                                      },
                                    },
                                  }))
                                }
                              >
                                <option value="">- Agent model -</option>
                                {availableModels.map((model) => (
                                  <option key={model.key} value={model.key}>
                                    {model.display_name || model.key}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <label className="agent-field">
                              <span className="w98-label">Summary prompt</span>
                              <textarea
                                className="w98-textarea"
                                style={{ minHeight: "100px" }}
                                value={agentConfig.run_budgets.summarization.prompt}
                                onChange={(e) =>
                                  setAgentConfig((config) => ({
                                    ...config,
                                    run_budgets: {
                                      ...config.run_budgets,
                                      summarization: {
                                        ...config.run_budgets.summarization,
                                        prompt: e.target.value,
                                      },
                                    },
                                  }))
                                }
                              />
                            </label>
                          </div>
                        )}
                      </div>
                    </div>

                  <div className="w98-window agents-panel-window">
                      <div className="w98-titlebar">Behavior triggers</div>
                      <div className="agents-panel-body">
                        <div className="agents-panel-header">
                          <label className="agent-inline-toggle">
                            <input
                              type="checkbox"
                              checked={agentConfig.behavior_triggers.enabled}
                              onChange={(e) =>
                                setAgentConfig((config) => ({
                                  ...config,
                                  behavior_triggers: { ...config.behavior_triggers, enabled: e.target.checked },
                                  }))
                              }
                            />
                            Enable
                          </label>
                          <label className="agent-field" style={{ minWidth: "120px" }}>
                            <span className="w98-label">Global threshold</span>
                            <input
                              className="w98-input"
                              type="number"
                              min={MIN_BEHAVIOR_TRIGGER_THRESHOLD}
                              max={MAX_BEHAVIOR_TRIGGER_THRESHOLD}
                              step="0.01"
                              value={agentConfig.behavior_triggers.default_similarity_threshold}
                              onChange={(e) =>
                                setAgentConfig((config) => ({
                                  ...config,
                                  behavior_triggers: {
                                    ...config.behavior_triggers,
                                    default_similarity_threshold: Number(e.target.value),
                                  },
                                }))
                              }
                            />
                          </label>
                        </div>
                        <div className="agents-inline-stack">
                          {agentConfig.behavior_triggers.behaviors.map((behavior, index) => (
                            <details className="w98-tool-detail" key={behavior.behavior_id} open>
                              <summary>{behavior.behavior_id}</summary>
                              <div className="w98-tool-body">
                                <label className="agent-field">
                                  <span className="w98-label">Trigger keywords</span>
                                  <textarea
                                    className="w98-textarea"
                                    style={{ minHeight: "56px" }}
                                    onChange={(e) =>
                                      setAgentConfig((config) => {
                                        const next = [...config.behavior_triggers.behaviors];
                                        next[index] = { ...next[index], keyword_triggers: e.target.value.split("\n") };
                                        return { ...config, behavior_triggers: { ...config.behavior_triggers, behaviors: next } };
                                      })
                                    }
                                    placeholder="One trigger per line"
                                    value={behavior.keyword_triggers.join("\n")}
                                  />
                                </label>
                                <label className="agent-field">
                                  <span className="w98-label">Full JSON config</span>
                                  <textarea
                                    className="w98-textarea"
                                    style={{ minHeight: "120px", fontSize: "10px" }}
                                    onChange={(e) => {
                                      try {
                                        const parsed = JSON.parse(e.target.value) as BehaviorTriggerConfig;
                                        setAgentConfig((config) => {
                                          const next = [...config.behavior_triggers.behaviors];
                                          next[index] = parsed;
                                          return { ...config, behavior_triggers: { ...config.behavior_triggers, behaviors: next } };
                                        });
                                      } catch {
                                        // Keep freeform until valid JSON
                                      }
                                    }}
                                    value={JSON.stringify(behavior, null, 2)}
                                  />
                                </label>
                              </div>
                            </details>
                          ))}
                        </div>
                      </div>
                    </div>
                </div>
              </div>
            </section>
          )}

          {/* ══ ROUTING TAB ══ */}
          {activeTab === "routing" && (
            <section style={{ flex: 1, overflowY: "auto", padding: "8px" }}>
              <div style={{ maxWidth: "1000px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "8px" }}>

                {/* Routing matrix */}
                <div className="w98-window" style={{ padding: "0" }}>
                  <div className="w98-titlebar">Routing matrix</div>
                  <div style={{ padding: "6px", overflowX: "auto" }}>
                    <table style={{ borderCollapse: "collapse", minWidth: "100%", fontSize: "11px" }}>
                      <thead>
                        <tr>
                          <th style={{ border: "1px solid var(--w98-dark)", padding: "3px 8px", background: "var(--w98-gray)", textAlign: "left", fontWeight: "bold", whiteSpace: "nowrap" }}>
                            From \ To
                          </th>
                          {agentConfig.agents.map((agent) => (
                            <th
                              key={agent.id}
                              style={{ border: "1px solid var(--w98-dark)", padding: "3px 8px", background: "var(--w98-gray)", textAlign: "left", fontWeight: "bold", whiteSpace: "nowrap" }}
                            >
                              {agent.name}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {agentConfig.agents.map((source) => (
                          <tr key={source.id}>
                            <td style={{ border: "1px solid var(--w98-dark)", padding: "3px 8px", fontWeight: "bold", whiteSpace: "nowrap", background: "var(--w98-gray)" }}>
                              {source.name}
                            </td>
                            {agentConfig.agents.map((target) => {
                              if (source.id === target.id) {
                                return (
                                  <td key={target.id} style={{ border: "1px solid var(--w98-dark)", padding: "3px 8px", background: "#D0D0D0", textAlign: "center" }}>
                                    —
                                  </td>
                                );
                              }
                              const existing = agentConfig.connections.find(
                                (c) => c.from === source.id && c.to === target.id,
                              );
                              return (
                                <td key={target.id} style={{ border: "1px solid var(--w98-dark)", padding: "3px 8px", background: "var(--w98-white)", textAlign: "center" }}>
                                  <input
                                    type="checkbox"
                                    checked={existing?.enabled !== false && !!existing}
                                    onChange={(e) =>
                                      setAgentConfig((config) => {
                                        const found = config.connections.find(
                                          (c) => c.from === source.id && c.to === target.id,
                                        );
                                        if (e.target.checked) {
                                          if (found) {
                                            return { ...config, connections: config.connections.map((c) => c === found ? { ...c, enabled: true } : c) };
                                          }
                                          return { ...config, connections: [...config.connections, { from: source.id, to: target.id, priority: 128, enabled: true }] };
                                        }
                                        return { ...config, connections: config.connections.filter((c) => !(c.from === source.id && c.to === target.id)) };
                                      })
                                    }
                                  />
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Connection rules */}
                <div className="w98-window" style={{ padding: "0" }}>
                  <div className="w98-titlebar">Connection rules</div>
                  <div style={{ padding: "6px", display: "flex", flexDirection: "column", gap: "4px" }}>
                    {agentConfig.connections.map((connection, index) => (
                      <div
                        key={`${connection.from}-${connection.to}-${index}`}
                        className="w98-raised"
                        style={{ padding: "4px 6px", display: "grid", gridTemplateColumns: "1.4fr 0.4fr 1fr 0.5fr", gap: "4px", alignItems: "center", background: "var(--w98-gray)" }}
                      >
                        <div style={{ fontWeight: "bold", fontSize: "11px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {(agentConfig.agents.find((a) => a.id === connection.from)?.name ?? connection.from) +
                            " → " +
                            (agentConfig.agents.find((a) => a.id === connection.to)?.name ?? connection.to)}
                        </div>
                        <input
                          className="w98-input"
                          type="number"
                          value={connection.priority ?? 128}
                          onChange={(e) =>
                            setAgentConfig((config) => {
                              const next = [...config.connections];
                              next[index] = { ...next[index], priority: Number(e.target.value) };
                              return { ...config, connections: next };
                            })
                          }
                        />
                        <input
                          className="w98-input"
                          placeholder="Condition (substring)"
                          value={connection.condition ?? ""}
                          onChange={(e) =>
                            setAgentConfig((config) => {
                              const next = [...config.connections];
                              next[index] = { ...next[index], condition: e.target.value || null };
                              return { ...config, connections: next };
                            })
                          }
                        />
                        <label style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "11px", cursor: "pointer" }}>
                          <input
                            type="checkbox"
                            checked={connection.enabled !== false}
                            onChange={(e) =>
                              setAgentConfig((config) => {
                                const next = [...config.connections];
                                next[index] = { ...next[index], enabled: e.target.checked };
                                return { ...config, connections: next };
                              })
                            }
                          />
                          Enabled
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}

        </div>
      </main>
    </div>
  );
}
