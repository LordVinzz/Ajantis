export type TabKey = "chat" | "agents" | "routing";

export interface ModelInstance {
  instance_id: string;
  context_length?: number | null;
}

export interface LoadedModel {
  key: string;
  instances?: ModelInstance[];
}

export interface ModelInfo {
  key: string;
  display_name?: string;
  model_type?: string | null;
  params_string?: string | null;
  format?: string | null;
  quantization?: string | null;
  max_context_length?: number | null;
  vision?: boolean;
  trained_for_tool_use?: boolean;
}

export interface AgentLoadConfig {
  context_length?: number | null;
  eval_batch_size?: number | null;
  flash_attention?: boolean | null;
  num_experts?: number | null;
  offload_kv_cache_to_gpu?: boolean | null;
}

export interface Agent {
  id: string;
  name: string;
  type?: string;
  agent_type?: string;
  model_key?: string | null;
  model_type?: string | null;
  role?: string | null;
  load_config?: AgentLoadConfig | null;
  mode?: string | null;
  allowed_tools?: string[] | null;
  armed?: boolean;
  is_manager?: boolean;
  paused?: boolean;
}

export interface RoutingRule {
  from: string;
  to: string;
  priority?: number;
  condition?: string | null;
  enabled?: boolean;
}

export interface CommandPolicy {
  allowlist: string[];
  denylist: string[];
}

export interface RedundancyDetectionConfig {
  enabled: boolean;
  embedding_model_key?: string | null;
  semantic_similarity_threshold: number;
  max_redundant_audit_retries: number;
}

export interface BudgetHitSummarizationConfig {
  enabled: boolean;
  prompt: string;
  model_key?: string | null;
}

export interface RunBudgetsConfig {
  enabled: boolean;
  llm_calls_per_window: number;
  tool_calls_per_window: number;
  spawned_agents_per_window: number;
  streamed_tokens_per_window: number;
  wall_clock_seconds_per_window: number;
  applies_to_behaviors: string[];
  on_limit: "pause" | "summarize" | "stop";
  summarization: BudgetHitSummarizationConfig;
}

export interface CoverageManifestConfig {
  enabled: boolean;
  require_resolution: boolean;
  unresolved_prompt?: string | null;
  gap_section_label?: string | null;
}

export interface ResponseRewriteConfig {
  enabled: boolean;
  min_evidence_grade_for_severity?: string | null;
  rewrite_prompt?: string | null;
}

export interface EvidenceGradingConfig {
  min_grade_to_synthesize?: string | null;
  code_signals: string[];
  config_signals: string[];
  command_signals: string[];
}

export interface ForceSynthesisConfig {
  after_n_completed_reports?: number | null;
  after_n_issue_reports?: number | null;
  require_coverage_gap_signal: boolean;
  min_targeted_topics?: number | null;
  min_evidence_backed_topics?: number | null;
  prompt?: string | null;
  fallback_text?: string | null;
}

export interface DelegationValidationConfig {
  enabled: boolean;
  max_weak_retries?: number | null;
  retry_prompt_template?: string | null;
}

export interface ToolBurstReflectionConfig {
  limit?: number | null;
  prompt?: string | null;
}

export interface NonProgressConfig {
  limit?: number | null;
  stall_prompt?: string | null;
}

export interface SectionRule {
  section_name: string;
  require_file_reference: boolean;
  disallow_template_phrases: string[];
  rewrite_loop_prompt?: string | null;
}

export interface BehaviorTriggerConfig {
  behavior_id: string;
  enabled: boolean;
  keyword_triggers: string[];
  embedding_trigger_phrases: string[];
  similarity_threshold?: number | null;
  system_prompt_injection?: string | null;
  runtime_note_enabled: boolean;
  runtime_note_template?: string | null;
  coverage_manifest: CoverageManifestConfig;
  required_sections: string[];
  section_rules: SectionRule[];
  response_rewrite: ResponseRewriteConfig;
  evidence_grading: EvidenceGradingConfig;
  force_synthesis: ForceSynthesisConfig;
  delegation_validation: DelegationValidationConfig;
  tool_burst_reflection: ToolBurstReflectionConfig;
  non_progress: NonProgressConfig;
}

export interface BehaviorTriggersConfig {
  enabled: boolean;
  embedding_model_key?: string | null;
  default_similarity_threshold: number;
  behaviors: BehaviorTriggerConfig[];
}

export interface AgentConfig {
  agents: Agent[];
  connections: RoutingRule[];
  command_policy: CommandPolicy;
  redundancy_detection: RedundancyDetectionConfig;
  behavior_triggers: BehaviorTriggersConfig;
  run_budgets: RunBudgetsConfig;
}

export interface WorkspaceThreadIndex {
  id: string;
  name: string;
}

export interface WorkspaceEntry {
  id: string;
  name: string;
  path: string;
  threads: WorkspaceThreadIndex[];
  _open?: boolean;
}

export interface WorkspaceConfig {
  workspaces: WorkspaceEntry[];
}

export interface WorkspaceToolCall {
  tool_name: string;
  args?: string;
  result?: string;
  status?: string;
  semantic?: boolean;
}

export interface WorkspaceMessageSignal {
  kind?: string;
  text?: string;
}

export interface BubbleRuntime {
  model_key?: string;
  is_manager?: boolean;
  context_limit?: number | null;
  estimated_input_tokens?: number;
  estimated_remaining_tokens?: number;
  estimated_output_tokens?: number;
  output_tokens?: number;
  tokens_per_second?: number;
  detail?: string;
  stage?: string;
}

export interface WorkspaceThreadMessage {
  kind: "text" | "bubble";
  agent_id?: string | null;
  agent_name: string;
  content: string;
  is_sent: boolean;
  is_error: boolean;
  for_user: boolean;
  internal: boolean;
  tools: WorkspaceToolCall[];
  signal?: WorkspaceMessageSignal | null;
  runtime?: BubbleRuntime | null;
}

export interface MemoryEntry {
  timestamp: string;
  agent_id: string;
  agent_name: string;
  role: string;
  content: string;
}

export interface CommandExecution {
  timestamp: string;
  tool_name: string;
  command: string;
  normalized_command: string;
  cwd: string;
  success: boolean;
  result: string;
}

export interface ThreadSnapshot {
  message_items: WorkspaceThreadMessage[];
  memory_entries: MemoryEntry[];
  command_history: CommandExecution[];
  active_run_id?: string | null;
  updated_at?: string;
}

export interface RuntimeState {
  stage?: string;
  detail?: string;
  model_key?: string;
  mode?: string;
  is_manager?: boolean;
  context_limit?: number | null;
  estimated_input_tokens?: number;
  estimated_remaining_tokens?: number;
  estimated_output_tokens?: number;
  input_tokens?: number;
  output_tokens?: number;
  reasoning_output_tokens?: number;
  tokens_per_second?: number;
  time_to_first_token_seconds?: number;
}

export interface ActiveRunUiState {
  runId: string;
  waitingConfirmation: boolean;
  limitMessage?: string;
}

export type StreamEvent =
  | {
      event: "agent_start";
      run_id: string;
      agent_id: string;
      agent_name: string;
      model_key: string;
      mode: string;
      is_manager: boolean;
      context_limit: number;
      estimated_input_tokens: number;
      estimated_remaining_tokens: number;
    }
  | {
      event: "agent_status";
      run_id: string;
      agent_id: string;
      stage: string;
      detail: string;
    }
  | {
      event: "agent_metrics";
      run_id: string;
      agent_id: string;
      stage: string;
      estimated_output_tokens: number;
      input_tokens?: number;
      output_tokens?: number;
      reasoning_output_tokens?: number;
      tokens_per_second?: number;
      time_to_first_token_seconds?: number;
    }
  | {
      event: "tool_call";
      run_id: string;
      agent_id: string;
      tool_name: string;
      args: string;
    }
  | {
      event: "tool_result";
      run_id: string;
      agent_id: string;
      tool_name: string;
      result: string;
    }
  | {
      event: "token";
      run_id: string;
      agent_id: string;
      content: string;
    }
  | {
      event: "agent_end";
      run_id: string;
      agent_id: string;
    }
  | {
      event: "error";
      run_id: string;
      agent_id: string;
      agent_name: string;
      message: string;
    }
  | {
      event: "run_limit_reached";
      run_id: string;
      kind: string;
      limit: number;
      observed: number;
    }
  | {
      event: "run_waiting_confirmation";
      run_id: string;
      message: string;
    }
  | {
      event: "run_resumed";
      run_id: string;
      message: string;
    }
  | {
      event: "run_cancelled";
      run_id: string;
      message: string;
    }
  | {
      event: "run_checkpoint_saved";
      run_id: string;
      checkpoint: string;
    }
  | {
      event: "run_usage_update";
      run_id: string;
      llm_calls: number;
      tool_calls: number;
      spawned_agents: number;
      streamed_tokens: number;
      embedding_calls: number;
      wall_clock_seconds: number;
    }
  | {
      event: "done";
      run_id: string;
    };
