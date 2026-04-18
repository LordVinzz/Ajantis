/*
 * Renderer script for Ajantis — multi-agent routing app (Tauri).
 */

const { invoke, Channel } = window.__TAURI__.core;

// ── DOM refs ───────────────────────────────────────────────────────

const messagesDiv = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const loadedBar = document.getElementById('loaded-bar');
const agentsList = document.getElementById('agents-list');
const routingTable = document.getElementById('routing-matrix');

// ── State ──────────────────────────────────────────────────────────

let agentConfig = {
  agents: [],
  connections: [],
  command_policy: { allowlist: [], denylist: [] },
  redundancy_detection: {
    enabled: true,
    embedding_model_key: null,
    semantic_similarity_threshold: 0.90,
    max_redundant_audit_retries: 1,
  },
  behavior_triggers: {
    enabled: true,
    embedding_model_key: null,
    default_similarity_threshold: 0.90,
    behaviors: [],
  },
};
let availableModels = []; // ModelInfo[] from backend
let modelDetails = {};    // key -> ModelInfo
const MIN_SEMANTIC_REDUNDANCY_THRESHOLD = 0.85;
const MAX_SEMANTIC_REDUNDANCY_THRESHOLD = 0.99;
const MIN_BEHAVIOR_TRIGGER_THRESHOLD = 0.0;
const MAX_BEHAVIOR_TRIGGER_THRESHOLD = 1.0;

function defaultBehaviorTriggers() {
  return {
    enabled: true,
    embedding_model_key: null,
    default_similarity_threshold: 0.90,
    behaviors: [
      {
        behavior_id: 'grounded_audit',
        enabled: true,
        keyword_triggers: [
          'security audit',
          'security review',
          'repo review',
          'code review',
          'find potential issues',
          'find bugs',
          'find vulnerabilities',
          'report findings',
          'report them here',
        ],
        embedding_trigger_phrases: [
          'audit this repository and report concrete issues',
          'perform a security review of this codebase',
          'inspect the repo for vulnerabilities and risky configurations',
        ],
        similarity_threshold: null,
      },
    ],
  };
}

function defaultRedundancyDetection() {
  return {
    enabled: true,
    embedding_model_key: null,
    semantic_similarity_threshold: 0.90,
    max_redundant_audit_retries: 1,
  };
}

function ensureAgentConfigDefaults() {
  if (!agentConfig.command_policy) {
    agentConfig.command_policy = { allowlist: [], denylist: [] };
  }
  if (!agentConfig.redundancy_detection) {
    agentConfig.redundancy_detection = defaultRedundancyDetection();
  } else {
    agentConfig.redundancy_detection = {
      ...defaultRedundancyDetection(),
      ...agentConfig.redundancy_detection,
    };
  }
  const threshold = parseFloat(agentConfig.redundancy_detection.semantic_similarity_threshold);
  agentConfig.redundancy_detection.semantic_similarity_threshold = Number.isFinite(threshold)
    ? Math.min(MAX_SEMANTIC_REDUNDANCY_THRESHOLD, Math.max(MIN_SEMANTIC_REDUNDANCY_THRESHOLD, threshold))
    : 0.90;
  if (!agentConfig.behavior_triggers) {
    agentConfig.behavior_triggers = defaultBehaviorTriggers();
  } else {
    agentConfig.behavior_triggers = {
      ...defaultBehaviorTriggers(),
      ...agentConfig.behavior_triggers,
    };
  }
  const behaviorThreshold = parseFloat(agentConfig.behavior_triggers.default_similarity_threshold);
  agentConfig.behavior_triggers.default_similarity_threshold = Number.isFinite(behaviorThreshold)
    ? Math.min(MAX_BEHAVIOR_TRIGGER_THRESHOLD, Math.max(MIN_BEHAVIOR_TRIGGER_THRESHOLD, behaviorThreshold))
    : 0.90;
  if (!Array.isArray(agentConfig.behavior_triggers.behaviors)) {
    agentConfig.behavior_triggers.behaviors = defaultBehaviorTriggers().behaviors;
  }
  if (!agentConfig.behavior_triggers.behaviors.some((behavior) => behavior.behavior_id === 'grounded_audit')) {
    agentConfig.behavior_triggers.behaviors.push(defaultBehaviorTriggers().behaviors[0]);
  }
  agentConfig.behavior_triggers.behaviors = agentConfig.behavior_triggers.behaviors.map((behavior) => ({
    keyword_triggers: [],
    embedding_trigger_phrases: [],
    similarity_threshold: null,
    enabled: true,
    ...behavior,
    similarity_threshold: behavior.similarity_threshold == null
      ? null
      : Math.min(
          MAX_BEHAVIOR_TRIGGER_THRESHOLD,
          Math.max(MIN_BEHAVIOR_TRIGGER_THRESHOLD, parseFloat(behavior.similarity_threshold) || 0)
        ),
  }));
}

// ── Tabs ───────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'routing') { renderRoutingMatrix(); renderConnectionRules(); }
    if (btn.dataset.tab === 'agents') { renderAgentsList(); renderCommandPolicy(); renderRedundancyDetection(); renderBehaviorTriggers(); }
  });
});

// ── Markdown ───────────────────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return '';
  const rawHtml = marked.parse(text);
  return DOMPurify.sanitize(rawHtml);
}

// ── Chat ───────────────────────────────────────────────────────────

let currentThreadMessages = [];

function cloneThreadMessages(messages) {
  return JSON.parse(JSON.stringify(messages || []));
}

function createTextMessageRecord(agentName, content, isSent, isError) {
  return {
    kind: 'text',
    agent_name: agentName,
    content,
    is_sent: !!isSent,
    is_error: !!isError,
    for_user: false,
    internal: false,
    tools: [],
    signal: null,
  };
}

function createBubbleMessageRecord(agentId, agentName) {
  const forUser = agentRoutesToUser(agentId);
  return {
    kind: 'bubble',
    agent_id: agentId,
    agent_name: agentName,
    content: '',
    is_sent: false,
    is_error: false,
    for_user: !!forUser,
    internal: !forUser,
    tools: [],
    signal: null,
  };
}

function setUpInternalBubbleToggle(wrapper, hint) {
  wrapper.classList.add('expanded');
  wrapper.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('.tool-item')) return;
    wrapper.classList.toggle('expanded');
    hint.textContent = wrapper.classList.contains('expanded') ? '(click to collapse)' : '(click to expand)';
  });
}

function buildToolItem(tool) {
  const item = document.createElement('div');
  item.className = 'tool-item' + (tool.semantic ? ' semantic-hit' : '');

  const args = tool.args || '';
  const argsPreview = args.replace(/\s+/g, ' ').slice(0, 80) + (args.length > 80 ? '…' : '');
  item.innerHTML =
    '<div class="tool-header">' +
      '<span class="tool-icon">⚙</span>' +
      '<span class="tool-name">' + escHtml(tool.tool_name || '') + '</span>' +
      '<span class="tool-args-preview">' + escHtml(argsPreview) + '</span>' +
      '<span class="tool-status ' + escHtml(tool.status || 'pending') + '">' +
        escHtml(
          tool.status === 'error'
            ? '✗'
            : tool.status === 'done'
              ? '✓'
              : 'running…'
        ) +
      '</span>' +
    '</div>' +
    '<div class="tool-detail">' +
      '<div class="tool-detail-label">Args</div>' +
      '<div class="tool-args-text">' + escHtml(args) + '</div>' +
    '</div>';

  if (tool.semantic) {
    const statusEl = item.querySelector('.tool-status');
    const headerEl = item.querySelector('.tool-header');
    const badge = document.createElement('span');
    badge.className = 'tool-badge semantic';
    badge.textContent = 'Embeddings';
    headerEl.insertBefore(badge, statusEl);
  }

  if (tool.result) {
    const detailEl = item.querySelector('.tool-detail');
    const resultDiv = document.createElement('div');
    resultDiv.innerHTML =
      '<div class="tool-detail-label" style="margin-top:6px;">Result</div>' +
      '<div>' + escHtml(tool.result) + '</div>';
    detailEl.appendChild(resultDiv);
  }

  item.addEventListener('click', () => item.classList.toggle('expanded'));
  return item;
}

function buildTextMessageElement(message) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + (message.is_sent ? 'sent' : 'received');

  const nameEl = document.createElement('div');
  nameEl.className = 'msg-agent-name';
  nameEl.textContent = message.agent_name || '';
  wrapper.appendChild(nameEl);

  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';
  if (message.is_error) {
    bodyEl.style.color = '#c00';
    bodyEl.textContent = message.content || '';
  } else {
    bodyEl.innerHTML = renderMarkdown(message.content || '');
  }
  wrapper.appendChild(bodyEl);

  return { wrapper, bodyEl };
}

function applyBubbleSignal(signalEl, signal) {
  if (!signal || !signal.text) {
    signalEl.className = 'bubble-signal';
    signalEl.textContent = '';
    return;
  }
  signalEl.className = 'bubble-signal active' + (signal.kind ? ' ' + signal.kind : '');
  signalEl.textContent = signal.text;
}

function buildBubbleElement(message) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message received' + (message.for_user ? ' for-user' : ' internal');

  const nameEl = document.createElement('div');
  nameEl.className = 'msg-agent-name';
  nameEl.textContent = message.agent_name || '';
  wrapper.appendChild(nameEl);

  if (message.internal) {
    const hint = document.createElement('div');
    hint.className = 'msg-collapsed-hint';
    hint.textContent = '(click to collapse)';
    wrapper.appendChild(hint);
    setUpInternalBubbleToggle(wrapper, hint);
  }

  const toolsEl = document.createElement('div');
  toolsEl.className = 'tool-calls msg-body';
  (message.tools || []).forEach((tool) => toolsEl.appendChild(buildToolItem(tool)));
  wrapper.appendChild(toolsEl);

  const signalEl = document.createElement('div');
  signalEl.className = 'bubble-signal';
  applyBubbleSignal(signalEl, message.signal);
  wrapper.appendChild(signalEl);

  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';
  if (message.is_error) {
    bodyEl.style.color = '#c00';
    bodyEl.textContent = message.content || '';
  } else {
    bodyEl.innerHTML = renderMarkdown(message.content || '');
  }
  wrapper.appendChild(bodyEl);

  return { wrapper, bodyEl, toolsEl, signalEl };
}

function appendThreadMessageElement(message) {
  const refs = message.kind === 'bubble'
    ? buildBubbleElement(message)
    : buildTextMessageElement(message);
  messagesDiv.appendChild(refs.wrapper);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return refs;
}

function renderThreadMessages(messages) {
  messagesDiv.innerHTML = '';
  messages.forEach((message) => appendThreadMessageElement(message));
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function parseLegacyToolItem(itemEl) {
  const tool = {
    tool_name: itemEl.querySelector('.tool-name')?.textContent?.trim() || '',
    args: itemEl.querySelector('.tool-args-text')?.textContent || '',
    result: '',
    status: 'pending',
    semantic: itemEl.classList.contains('semantic-hit') || !!itemEl.querySelector('.tool-badge.semantic'),
  };

  const statusEl = itemEl.querySelector('.tool-status');
  if (statusEl) {
    if (statusEl.classList.contains('error')) {
      tool.status = 'error';
    } else if (statusEl.classList.contains('done')) {
      tool.status = 'done';
    }
  }

  const labels = itemEl.querySelectorAll('.tool-detail-label');
  labels.forEach((labelEl) => {
    if (labelEl.textContent.trim().toLowerCase() !== 'result') return;
    const valueEl = labelEl.nextElementSibling;
    if (valueEl) {
      tool.result = valueEl.textContent || '';
    }
  });

  return tool;
}

function parseLegacyThreadHtml(html) {
  if (!html) return [];
  const container = document.createElement('div');
  container.innerHTML = html;
  const messages = [];

  Array.from(container.children).forEach((messageEl) => {
    if (!messageEl.classList || !messageEl.classList.contains('message')) return;

    const agentName = messageEl.querySelector('.msg-agent-name')?.textContent?.trim() || '';
    const bodyEls = messageEl.querySelectorAll(':scope > .msg-body');
    const textBodyEl = bodyEls.length > 0 ? bodyEls[bodyEls.length - 1] : null;
    const content = textBodyEl ? (textBodyEl.textContent || '') : '';
    const isError = !!(textBodyEl && /#c00|rgb\(204,\s*0,\s*0\)/i.test(textBodyEl.getAttribute('style') || ''));

    if (messageEl.classList.contains('received') && messageEl.querySelector('.tool-calls')) {
      const signalEl = messageEl.querySelector(':scope > .bubble-signal');
      messages.push({
        kind: 'bubble',
        agent_id: null,
        agent_name: agentName,
        content,
        is_sent: false,
        is_error: isError,
        for_user: messageEl.classList.contains('for-user'),
        internal: messageEl.classList.contains('internal'),
        tools: Array.from(messageEl.querySelectorAll('.tool-item')).map(parseLegacyToolItem),
        signal: signalEl && signalEl.textContent
          ? {
              kind: signalEl.classList.contains('semantic') ? 'semantic' : '',
              text: signalEl.textContent,
            }
          : null,
      });
      return;
    }

    messages.push({
      kind: 'text',
      agent_name: agentName,
      content,
      is_sent: messageEl.classList.contains('sent'),
      is_error: isError,
      for_user: false,
      internal: false,
      tools: [],
      signal: null,
    });
  });

  return messages;
}

function appendMessage(agentName, content, isSent, isError) {
  const message = createTextMessageRecord(agentName, content, isSent, isError);
  currentThreadMessages.push(message);
  appendThreadMessageElement(message);
}

// Check if an agent has a direct connection to the user node
function agentRoutesToUser(agentId) {
  return agentConfig.connections.some(
    (c) => c.from === agentId && agentConfig.agents.some((a) => a.id === c.to && a.type === 'user')
  );
}

// Track in-progress streaming bubbles: agent_id -> { wrapper, bodyEl, toolsEl, content }
let streamingBubbles = {};
// Last pending tool-item DOM node per agent, waiting for its result
let pendingToolEl = {};

function toolResultUsesEmbeddings(result) {
  return /semantic overlap/i.test(result || '');
}

function markBubbleSemanticUsage(bubble, result) {
  if (!bubble || !bubble.signalEl) return;
  bubble.signalEl.className = 'bubble-signal semantic active';
  const match = String(result || '').match(/semantic overlap\s+([0-9.]+)/i);
  const similarity = match ? ' (similarity ' + match[1] + ')' : '';
  bubble.signalEl.textContent = 'Embeddings-based semantic redundancy detection was used for this audit decision' + similarity + '.';
}

function getOrCreateBubble(agentId, agentName) {
  if (streamingBubbles[agentId]) return streamingBubbles[agentId];
  const messageRef = createBubbleMessageRecord(agentId, agentName);
  currentThreadMessages.push(messageRef);
  const refs = appendThreadMessageElement(messageRef);
  const bubble = { ...refs, content: '', messageRef };
  streamingBubbles[agentId] = bubble;
  return bubble;
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  // ── Workspace guard ──────────────────────────────────────────────
  if (!activeWorkspaceId) {
    appendMessage('System', 'Sélectionne un workspace dans la barre latérale avant d\'envoyer un message.', false, true);
    return;
  }

  // ── Auto-create thread if none selected ─────────────────────────
  if (!activeThreadId) {
    const ws = workspaceConfig.workspaces.find((w) => w.id === activeWorkspaceId);
    if (ws) {
      const threadName = message.slice(0, 20) + (message.length > 20 ? '…' : '');
      const newThread = { id: generateThreadId(), name: threadName, messages: '', message_items: [] };
      ws.threads.push(newThread);
      activeThreadId = newThread.id;
      ws._open = true;
      await saveWorkspaceConfig();
      renderWorkspaceList();
    }
  }

  appendMessage('You', message, true, false);
  chatInput.value = '';
  streamingBubbles = {};
  pendingToolEl = {};

  const onEvent = new Channel();
  onEvent.onmessage = (evt) => {
    if (evt.event === 'agent_start') {
      getOrCreateBubble(evt.agent_id, evt.agent_name);

    } else if (evt.event === 'tool_call') {
      const bubble = streamingBubbles[evt.agent_id];
      if (!bubble) return;
      const toolRef = {
        tool_name: evt.tool_name,
        args: evt.args,
        result: '',
        status: 'pending',
        semantic: false,
      };
      bubble.messageRef.tools.push(toolRef);
      const item = buildToolItem(toolRef);
      bubble.toolsEl.appendChild(item);
      pendingToolEl[evt.agent_id] = { item, toolRef, bubble };
      messagesDiv.scrollTop = messagesDiv.scrollHeight;

    } else if (evt.event === 'tool_result') {
      const pending = pendingToolEl[evt.agent_id];
      if (pending) {
        const { item, toolRef, bubble } = pending;
        const statusEl = item.querySelector('.tool-status');
        const isError = evt.result.startsWith('[tool call failed') || evt.result.startsWith('[tool returned no text]');
        toolRef.status = isError ? 'error' : 'done';
        toolRef.result = evt.result;
        statusEl.textContent = isError ? '✗' : '✓';
        statusEl.className = 'tool-status ' + (isError ? 'error' : 'done');

        if (toolResultUsesEmbeddings(evt.result)) {
          toolRef.semantic = true;
          item.classList.add('semantic-hit');
          const headerEl = item.querySelector('.tool-header');
          if (headerEl && !item.querySelector('.tool-badge.semantic')) {
            const badge = document.createElement('span');
            badge.className = 'tool-badge semantic';
            badge.textContent = 'Embeddings';
            headerEl.insertBefore(badge, statusEl);
          }
          bubble.messageRef.signal = {
            kind: 'semantic',
            text: '',
          };
          markBubbleSemanticUsage(bubble, evt.result);
          bubble.messageRef.signal.text = bubble.signalEl.textContent;
        }

        const detailEl = item.querySelector('.tool-detail');
        const resultDiv = document.createElement('div');
        resultDiv.innerHTML =
          '<div class="tool-detail-label" style="margin-top:6px;">Result</div>' +
          '<div>' + escHtml(evt.result) + '</div>';
        detailEl.appendChild(resultDiv);
        delete pendingToolEl[evt.agent_id];
      }

    } else if (evt.event === 'token') {
      const bubble = streamingBubbles[evt.agent_id];
      if (bubble) {
        bubble.content += evt.content;
        bubble.messageRef.content = bubble.content;
        bubble.bodyEl.innerHTML = renderMarkdown(bubble.content);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
      }

    } else if (evt.event === 'agent_end') {
      delete streamingBubbles[evt.agent_id];
      delete pendingToolEl[evt.agent_id];

    } else if (evt.event === 'error') {
      const bubble = getOrCreateBubble(evt.agent_id, evt.agent_name);
      bubble.messageRef.is_error = true;
      bubble.messageRef.content = evt.message;
      bubble.bodyEl.style.color = '#c00';
      bubble.bodyEl.textContent = evt.message;
      delete streamingBubbles[evt.agent_id];
      delete pendingToolEl[evt.agent_id];

    } else if (evt.event === 'done') {
      streamingBubbles = {};
      pendingToolEl = {};
      refreshLoadedModels();
      if (activeThreadId) {
        captureThreadState(activeWorkspaceId, activeThreadId).then(() => {
          saveWorkspaceConfig();
        });
      }
    }
  };

  try {
    await invoke('route_message', {
      fromAgentId: 'user',
      message,
      onEvent,
    });
  } catch (err) {
    appendMessage('System', String(err), false, true);
  }
});

// ── Loaded models bar ──────────────────────────────────────────────

async function refreshLoadedModels() {
  try {
    const loaded = await invoke('fetch_loaded_models');
    if (loaded.length === 0) {
      loadedBar.textContent = 'No models currently loaded.';
    } else {
      loadedBar.innerHTML = '';
      const label = document.createElement('strong');
      label.textContent = 'Loaded: ';
      loadedBar.appendChild(label);
      loaded.forEach((m, mi) => {
        m.instances.forEach((inst, ii) => {
          if (mi > 0 || ii > 0) loadedBar.appendChild(document.createTextNode(' \u00b7 '));
          const span = document.createElement('span');
          // TODO: render remaining-context instead of only max context once the backend
          // starts tracking per-agent budget and surfaces it on instance updates.
          span.textContent = m.display_name + ' (ctx: ' + (inst.context_length || '?') + ')';
          loadedBar.appendChild(span);

          const btn = document.createElement('button');
          btn.textContent = '\u2715';
          btn.title = 'Unload';
          btn.style.cssText = 'margin-left:4px;padding:1px 6px;font-size:0.7rem;border-radius:10px;border:1px solid #ccc;background:#f5f5f5;cursor:pointer;color:#c00;';
          btn.addEventListener('click', async () => {
            try {
              await invoke('unload_model', { instanceId: inst.instance_id });
              refreshLoadedModels();
            } catch (e) {
              appendMessage('System', String(e), false, true);
            }
          });
          loadedBar.appendChild(btn);
        });
      });
    }
  } catch (err) {
    loadedBar.textContent = 'Could not fetch loaded models.';
  }
}

// ── Fetch available models from LM Studio ──────────────────────────

async function fetchAvailableModels() {
  try {
    availableModels = await invoke('fetch_models');
    modelDetails = {};
    availableModels.forEach((m) => { modelDetails[m.key] = m; });
  } catch (err) {
    appendMessage('System', 'Failed to fetch models: ' + err, false, true);
  }
}

// ── Agent config persistence ───────────────────────────────────────

async function loadConfig() {
  try {
    agentConfig = await invoke('load_agent_config');
    ensureAgentConfigDefaults();
    renderAgentsList();
    renderCommandPolicy();
    renderRedundancyDetection();
    renderBehaviorTriggers();
    renderRoutingMatrix();
  } catch (err) {
    appendMessage('System', 'Failed to load config: ' + err, false, true);
  }
}

async function saveConfig() {
  try {
    ensureAgentConfigDefaults();
    await invoke('save_agent_config', { config: agentConfig });
    appendMessage('System', 'Configuration saved.', false, false);
  } catch (err) {
    appendMessage('System', 'Failed to save config: ' + err, false, true);
  }
}

document.getElementById('saveConfigBtn').addEventListener('click', saveConfig);
document.getElementById('reloadConfigBtn').addEventListener('click', loadConfig);

// ── Agents list ────────────────────────────────────────────────────

function generateId() {
  return 'agent-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

document.getElementById('addAgentBtn').addEventListener('click', () => {
  const firstModel = availableModels.length > 0 ? availableModels[0] : null;
  agentConfig.agents.push({
    id: generateId(),
    name: 'New Agent',
    type: 'model',
    model_key: firstModel ? firstModel.key : '',
    model_type: firstModel ? (firstModel.model_type || 'llm') : 'llm',
    role: '',
    load_config: {},
    mode: 'stay_awake',
    armed: true,
    is_manager: false,
  });
  renderAgentsList();
});

function renderAgentsList() {
  agentsList.innerHTML = '';
  agentConfig.agents.forEach((agent, idx) => {
    // Default armed + is_manager for agents loaded from old configs
    if (agent.armed === undefined) agent.armed = true;
    if (agent.is_manager === undefined) agent.is_manager = false;

    const isArmed = agent.armed !== false;
    const isManager = !!agent.is_manager;
    const card = document.createElement('div');
    let cardClass = 'agent-card ';
    if (agent.type === 'user') cardClass += 'user-agent';
    else if (isManager) cardClass += 'model-agent manager-agent';
    else cardClass += 'model-agent';
    if (!isArmed) cardClass += ' disarmed';
    card.className = cardClass;

    let html = '';

    // Name + arm toggle on the same row
    html += '<div style="display:flex;align-items:center;gap:8px;">';
    html += '<div style="flex:1;"><label>Name</label>';
    html += '<input type="text" data-field="name" value="' + escHtml(agent.name) + '" />';
    if (isManager) html += '<span class="manager-badge">Manager</span>';
    html += '</div>';
    if (agent.type !== 'user') {
      html += '<label style="display:inline-flex;align-items:center;gap:4px;margin:0;font-size:0.8rem;white-space:nowrap;">';
      html += '<input type="checkbox" data-field="is_manager"' + (isManager ? ' checked' : '') + ' /> Manager</label>';
      html += '<button class="btn-arm ' + (isArmed ? 'armed' : 'disarmed') + '">' + (isArmed ? 'Armed' : 'Disarmed') + '</button>';
    }
    html += '</div>';

    if (agent.type === 'user') {
      html += '<div style="margin-top:6px;font-size:0.8rem;color:#007aff;">User agent (you)</div>';
    } else {
      // Model selector
      html += '<label>Model</label>';
      html += '<select data-field="model_key">';
      availableModels.forEach((m) => {
        const sel = m.key === agent.model_key ? ' selected' : '';
        html += '<option value="' + escHtml(m.key) + '"' + sel + '>' + escHtml(m.display_name || m.key) + '</option>';
      });
      html += '</select>';

      // Model info + type badge
      const info = modelDetails[agent.model_key];
      if (info) {
        const parts = [];
        if (info.model_type) parts.push(info.model_type.toUpperCase());
        if (info.params_string) parts.push(info.params_string);
        if (info.format) parts.push(info.format.toUpperCase());
        if (info.quantization) parts.push(info.quantization);
        if (info.max_context_length) parts.push('max ctx: ' + info.max_context_length);
        if (info.vision) parts.push('vision');
        if (info.trained_for_tool_use) parts.push('tool use');
        if (parts.length) {
          html += '<div style="font-size:0.75rem;color:#888;margin-top:2px;">' + escHtml(parts.join(' \u00b7 ')) + '</div>';
        }
      }

      // Role
      html += '<label>Role (system prompt)</label>';
      html += '<textarea data-field="role">' + escHtml(agent.role || '') + '</textarea>';

      // Mode
      html += '<label>Mode</label>';
      html += '<div class="mode-row">';
      html += '<label><input type="radio" name="mode-' + idx + '" value="stay_awake"' + (agent.mode !== 'on_the_fly' ? ' checked' : '') + ' data-field="mode" /> Stay awake</label>';
      html += '<label><input type="radio" name="mode-' + idx + '" value="on_the_fly"' + (agent.mode === 'on_the_fly' ? ' checked' : '') + ' data-field="mode" /> On the fly</label>';
      html += '</div>';

      // Load config
      const lc = agent.load_config || {};
      html += '<label>Load settings</label>';
      html += '<div class="config-row">';
      html += '<label>Context: <input type="number" data-cfg="context_length" value="' + (lc.context_length || '') + '" placeholder="' + (info && info.max_context_length ? 'max: ' + info.max_context_length : '') + '" min="1" /></label>';
      html += '<label><input type="checkbox" data-cfg="flash_attention"' + (lc.flash_attention ? ' checked' : '') + ' /> Flash attn</label>';
      html += '<label>Eval batch: <input type="number" data-cfg="eval_batch_size" value="' + (lc.eval_batch_size || '') + '" min="1" /></label>';
      html += '<label>Experts: <input type="number" data-cfg="num_experts" value="' + (lc.num_experts || '') + '" min="1" /></label>';
      html += '<label><input type="checkbox" data-cfg="offload_kv_cache_to_gpu"' + (lc.offload_kv_cache_to_gpu ? ' checked' : '') + ' /> GPU KV offload</label>';
      html += '</div>';
    }

    // Actions
    html += '<div class="agent-actions">';
    if (agent.type === 'model') {
      html += '<button class="btn-load" style="background:#34c759;color:#fff;border:none;">Load Now</button>';
    }
    if (agent.type !== 'user') {
      html += '<button class="btn-danger btn-delete"' + (isArmed ? ' disabled title="Disarm agent before deleting"' : '') + '>Delete</button>';
    }
    html += '</div>';

    card.innerHTML = html;

    // Bind change events
    card.querySelectorAll('[data-field]').forEach((el) => {
      const field = el.dataset.field;
      if (el.type === 'checkbox') {
        el.addEventListener('change', () => {
          agent[field] = el.checked;
          if (field === 'is_manager') renderAgentsList(); // refresh badge
        });
      } else if (el.type === 'radio') {
        el.addEventListener('change', () => { agent[field] = el.value; });
      } else {
        el.addEventListener('input', () => { agent[field] = el.value; });
      }
    });

    card.querySelectorAll('[data-cfg]').forEach((el) => {
      const key = el.dataset.cfg;
      const evt = el.type === 'checkbox' ? 'change' : 'input';
      el.addEventListener(evt, () => {
        if (!agent.load_config) agent.load_config = {};
        if (el.type === 'checkbox') {
          agent.load_config[key] = el.checked || null;
        } else {
          agent.load_config[key] = el.value ? parseInt(el.value, 10) : null;
        }
      });
    });

    // Model selector updates model info and model_type on change
    const modelSel = card.querySelector('[data-field="model_key"]');
    if (modelSel) {
      modelSel.addEventListener('change', () => {
        agent.model_key = modelSel.value;
        const info = modelDetails[modelSel.value];
        agent.model_type = info ? (info.model_type || 'llm') : 'llm';
        renderAgentsList(); // re-render to update info display
      });
    }

    // Arm/disarm toggle
    const armBtn = card.querySelector('.btn-arm');
    if (armBtn) {
      armBtn.addEventListener('click', () => {
        agent.armed = !agent.armed;
        renderAgentsList();
        renderRoutingMatrix();
      });
    }

    // Load Now button
    const loadBtn = card.querySelector('.btn-load');
    if (loadBtn) {
      loadBtn.addEventListener('click', async () => {
        const lc = agent.load_config || {};
        const config = { model: agent.model_key };
        if (lc.context_length) config.context_length = lc.context_length;
        if (lc.flash_attention) config.flash_attention = lc.flash_attention;
        if (lc.eval_batch_size) config.eval_batch_size = lc.eval_batch_size;
        if (lc.num_experts) config.num_experts = lc.num_experts;
        if (lc.offload_kv_cache_to_gpu) config.offload_kv_cache_to_gpu = lc.offload_kv_cache_to_gpu;
        try {
          await invoke('load_model', { config });
          appendMessage('System', 'Loaded ' + agent.name + ' (' + agent.model_key + ')', false, false);
          refreshLoadedModels();
        } catch (e) {
          appendMessage('System', String(e), false, true);
        }
      });
    }

    // Delete button (only works when disarmed)
    const delBtn = card.querySelector('.btn-delete');
    if (delBtn && !isArmed) {
      delBtn.addEventListener('click', () => {
        agentConfig.agents.splice(idx, 1);
        agentConfig.connections = agentConfig.connections.filter(
          (c) => c.from !== agent.id && c.to !== agent.id
        );
        renderAgentsList();
        renderRoutingMatrix();
      });
    }

    agentsList.appendChild(card);
  });
}

// ── Command sandbox policy ─────────────────────────────────────────

function renderCommandPolicy() {
  const container = document.getElementById('command-policy-section');
  if (!container) return;

  ensureAgentConfigDefaults();
  const policy = agentConfig.command_policy;

  container.innerHTML = `
    <h3 style="margin:0 0 8px;">Command Sandbox Policy</h3>
    <p style="font-size:0.8rem;color:#888;margin:0 0 10px;">
      Controls which shell commands agents may execute (bash, REPL, PowerShell, TaskCreate).<br>
      Path traversal (<code>..</code>) and absolute paths outside the workspace are always blocked.<br>
      <strong>Denylist</strong> takes priority over allowlist. Leave allowlist empty to allow all (except denylisted).
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div>
        <label style="font-weight:600;color:#f38ba8;">Denylist (always blocked)</label>
        <textarea id="policy-denylist" style="width:100%;height:90px;margin-top:4px;font-family:monospace;font-size:0.8rem;" placeholder="rm -rf&#10;sudo&#10;curl">${escHtml(policy.denylist.join('\n'))}</textarea>
        <div style="font-size:0.75rem;color:#888;">One prefix per line. Matching commands are always rejected.</div>
      </div>
      <div>
        <label style="font-weight:600;color:#a6e3a1;">Allowlist (whitelist)</label>
        <textarea id="policy-allowlist" style="width:100%;height:90px;margin-top:4px;font-family:monospace;font-size:0.8rem;" placeholder="ls&#10;cat&#10;python3">${escHtml(policy.allowlist.join('\n'))}</textarea>
        <div style="font-size:0.75rem;color:#888;">One prefix per line. If non-empty, only matching commands are allowed.</div>
      </div>
    </div>`;

  container.querySelector('#policy-denylist').addEventListener('input', (e) => {
    policy.denylist = e.target.value.split('\n').map((s) => s.trim()).filter(Boolean);
  });
  container.querySelector('#policy-allowlist').addEventListener('input', (e) => {
    policy.allowlist = e.target.value.split('\n').map((s) => s.trim()).filter(Boolean);
  });
}

function renderRedundancyDetection() {
  const container = document.getElementById('redundancy-detection-section');
  if (!container) return;

  ensureAgentConfigDefaults();
  const cfg = agentConfig.redundancy_detection;
  const embeddingModels = availableModels.filter((model) => (model.model_type || '').toLowerCase() === 'embedding');
  const options = ['<option value="">Lexical fallback only</option>']
    .concat(
      embeddingModels.map((model) => {
        const selected = model.key === cfg.embedding_model_key ? ' selected' : '';
        return '<option value="' + escHtml(model.key) + '"' + selected + '>' + escHtml(model.display_name || model.key) + '</option>';
      })
    )
    .join('');

  container.innerHTML = `
    <h3 style="margin:0 0 8px;">Loop / Redundancy Detection</h3>
    <p style="font-size:0.8rem;color:#666;margin:0 0 10px;">
      Detects repeated audit loops and pushes the manager to synthesize once the same topic keeps coming back.<br>
      Exact command dedupe still applies independently. Embeddings are optional; lexical fallback remains active.
    </p>
    <div class="config-row" style="margin-bottom:10px;">
      <label><input id="redundancy-enabled" type="checkbox"${cfg.enabled ? ' checked' : ''} /> Enable redundancy detection</label>
    </div>
    <div style="display:grid;grid-template-columns:1.4fr 0.8fr 0.8fr;gap:12px;">
      <div>
        <label style="font-weight:600;">Embeddings model</label>
        <select id="redundancy-embedding-model" style="width:100%;margin-top:4px;" ${cfg.enabled ? '' : 'disabled'}>
          ${options}
        </select>
        <div style="font-size:0.75rem;color:#888;margin-top:4px;">Only models reported as <code>embedding</code> are listed here.</div>
      </div>
      <div>
        <label style="font-weight:600;">Similarity threshold</label>
        <input id="redundancy-threshold" type="number" min="${MIN_SEMANTIC_REDUNDANCY_THRESHOLD}" max="${MAX_SEMANTIC_REDUNDANCY_THRESHOLD}" step="0.01" value="${Number(cfg.semantic_similarity_threshold || 0.90).toFixed(2)}" style="width:100%;margin-top:4px;" ${cfg.enabled ? '' : 'disabled'} />
        <div style="font-size:0.75rem;color:#888;margin-top:4px;">Higher means stricter semantic overlap before a topic is treated as redundant. Values below ${MIN_SEMANTIC_REDUNDANCY_THRESHOLD.toFixed(2)} are clamped.</div>
      </div>
      <div>
        <label style="font-weight:600;">Redundant retries</label>
        <input id="redundancy-retries" type="number" min="0" max="5" step="1" value="${parseInt(cfg.max_redundant_audit_retries ?? 1, 10)}" style="width:100%;margin-top:4px;" ${cfg.enabled ? '' : 'disabled'} />
        <div style="font-size:0.75rem;color:#888;margin-top:4px;">How many same-topic audit retries are tolerated before synthesis is forced.</div>
      </div>
    </div>`;

  const enabledEl = container.querySelector('#redundancy-enabled');
  const modelEl = container.querySelector('#redundancy-embedding-model');
  const thresholdEl = container.querySelector('#redundancy-threshold');
  const retriesEl = container.querySelector('#redundancy-retries');

  enabledEl.addEventListener('change', () => {
    cfg.enabled = enabledEl.checked;
    modelEl.disabled = !cfg.enabled;
    thresholdEl.disabled = !cfg.enabled;
    retriesEl.disabled = !cfg.enabled;
  });

  modelEl.addEventListener('change', () => {
    cfg.embedding_model_key = modelEl.value || null;
  });

  thresholdEl.addEventListener('input', () => {
    const value = parseFloat(thresholdEl.value);
    cfg.semantic_similarity_threshold = Number.isFinite(value)
      ? Math.min(MAX_SEMANTIC_REDUNDANCY_THRESHOLD, Math.max(MIN_SEMANTIC_REDUNDANCY_THRESHOLD, value))
      : 0.90;
  });

  retriesEl.addEventListener('input', () => {
    const value = parseInt(retriesEl.value, 10);
    cfg.max_redundant_audit_retries = Number.isFinite(value)
      ? Math.min(5, Math.max(0, value))
      : 1;
  });
}

function behaviorTriggerLabel(behaviorId) {
  if (behaviorId === 'grounded_audit') return 'Grounded Audit';
  return behaviorId;
}

function behaviorTriggerDescription(behaviorId) {
  if (behaviorId === 'grounded_audit') {
    return 'Activates the stricter audit pipeline: coverage checklist enforcement, confirmed findings vs hypotheses, and audit-specific subagent validation.';
  }
  return 'Built-in runtime behavior.';
}

function renderBehaviorTriggers() {
  const container = document.getElementById('behavior-triggers-section');
  if (!container) return;

  ensureAgentConfigDefaults();
  const cfg = agentConfig.behavior_triggers;
  const embeddingModels = availableModels.filter((model) => (model.model_type || '').toLowerCase() === 'embedding');
  const options = ['<option value="">Keywords only</option>']
    .concat(
      embeddingModels.map((model) => {
        const selected = model.key === cfg.embedding_model_key ? ' selected' : '';
        return '<option value="' + escHtml(model.key) + '"' + selected + '>' + escHtml(model.display_name || model.key) + '</option>';
      })
    )
    .join('');

  container.innerHTML = `
    <h3 style="margin:0 0 8px;">Behavior Triggers</h3>
    <p style="font-size:0.8rem;color:#666;margin:0 0 10px;">
      Built-in runtime behaviors can be activated by explicit keywords and/or semantic similarity with reference phrases.<br>
      Detection runs on the actual task message, not the manager system prompt.
    </p>
    <div class="config-row" style="margin-bottom:10px;">
      <label><input id="behavior-triggers-enabled" type="checkbox"${cfg.enabled ? ' checked' : ''} /> Enable configurable behavior triggers</label>
    </div>
    <div style="display:grid;grid-template-columns:1.4fr 0.8fr;gap:12px;margin-bottom:14px;">
      <div>
        <label style="font-weight:600;">Embeddings model</label>
        <select id="behavior-trigger-embedding-model" style="width:100%;margin-top:4px;" ${cfg.enabled ? '' : 'disabled'}>
          ${options}
        </select>
        <div style="font-size:0.75rem;color:#888;margin-top:4px;">Optional. If unset, only keyword triggers are used.</div>
      </div>
      <div>
        <label style="font-weight:600;">Default similarity threshold</label>
        <input id="behavior-trigger-threshold" type="number" min="${MIN_BEHAVIOR_TRIGGER_THRESHOLD}" max="${MAX_BEHAVIOR_TRIGGER_THRESHOLD}" step="0.01" value="${Number(cfg.default_similarity_threshold || 0.90).toFixed(2)}" style="width:100%;margin-top:4px;" ${cfg.enabled ? '' : 'disabled'} />
        <div style="font-size:0.75rem;color:#888;margin-top:4px;">Used when a behavior has no per-behavior override.</div>
      </div>
    </div>
    <div id="behavior-trigger-cards" style="display:flex;flex-direction:column;gap:12px;"></div>`;

  const enabledEl = container.querySelector('#behavior-triggers-enabled');
  const modelEl = container.querySelector('#behavior-trigger-embedding-model');
  const thresholdEl = container.querySelector('#behavior-trigger-threshold');
  const cardsEl = container.querySelector('#behavior-trigger-cards');

  cfg.behaviors.forEach((behavior, idx) => {
    const card = document.createElement('div');
    card.style.cssText = 'padding:12px;border:1px solid #d8d8d8;border-radius:8px;background:#fff;';
    card.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
        <div>
          <div style="font-weight:700;color:#333;">${escHtml(behaviorTriggerLabel(behavior.behavior_id))}</div>
          <div style="font-size:0.78rem;color:#666;margin-top:2px;">${escHtml(behaviorTriggerDescription(behavior.behavior_id))}</div>
        </div>
        <label style="display:inline-flex;align-items:center;gap:4px;margin:0;font-size:0.8rem;">
          <input type="checkbox" data-behavior-field="enabled"${behavior.enabled ? ' checked' : ''} ${cfg.enabled ? '' : 'disabled'} />
          Enabled
        </label>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 0.7fr;gap:12px;margin-top:10px;">
        <div>
          <label style="font-weight:600;">Keyword triggers</label>
          <textarea data-behavior-field="keyword_triggers" style="width:100%;height:96px;margin-top:4px;font-family:monospace;font-size:0.8rem;" ${cfg.enabled && behavior.enabled ? '' : 'disabled'}>${escHtml((behavior.keyword_triggers || []).join('\n'))}</textarea>
          <div style="font-size:0.75rem;color:#888;margin-top:4px;">One phrase per line. Simple substring matching on the user task.</div>
        </div>
        <div>
          <label style="font-weight:600;">Embedding trigger phrases</label>
          <textarea data-behavior-field="embedding_trigger_phrases" style="width:100%;height:96px;margin-top:4px;font-family:monospace;font-size:0.8rem;" ${cfg.enabled && behavior.enabled ? '' : 'disabled'}>${escHtml((behavior.embedding_trigger_phrases || []).join('\n'))}</textarea>
          <div style="font-size:0.75rem;color:#888;margin-top:4px;">Reference phrases compared semantically against the task using the configured embeddings model.</div>
        </div>
        <div>
          <label style="font-weight:600;">Threshold override</label>
          <input data-behavior-field="similarity_threshold" type="number" min="${MIN_BEHAVIOR_TRIGGER_THRESHOLD}" max="${MAX_BEHAVIOR_TRIGGER_THRESHOLD}" step="0.01" value="${behavior.similarity_threshold == null ? '' : Number(behavior.similarity_threshold).toFixed(2)}" placeholder="${Number(cfg.default_similarity_threshold || 0.90).toFixed(2)}" style="width:100%;margin-top:4px;" ${cfg.enabled && behavior.enabled ? '' : 'disabled'} />
          <div style="font-size:0.75rem;color:#888;margin-top:4px;">Leave blank to use the global default threshold.</div>
        </div>
      </div>`;

    const enableField = card.querySelector('[data-behavior-field="enabled"]');
    const keywordField = card.querySelector('[data-behavior-field="keyword_triggers"]');
    const phraseField = card.querySelector('[data-behavior-field="embedding_trigger_phrases"]');
    const thresholdField = card.querySelector('[data-behavior-field="similarity_threshold"]');

    enableField.addEventListener('change', () => {
      behavior.enabled = enableField.checked;
      keywordField.disabled = !(cfg.enabled && behavior.enabled);
      phraseField.disabled = !(cfg.enabled && behavior.enabled);
      thresholdField.disabled = !(cfg.enabled && behavior.enabled);
    });
    keywordField.addEventListener('input', () => {
      behavior.keyword_triggers = keywordField.value.split('\n').map((value) => value.trim()).filter(Boolean);
    });
    phraseField.addEventListener('input', () => {
      behavior.embedding_trigger_phrases = phraseField.value.split('\n').map((value) => value.trim()).filter(Boolean);
    });
    thresholdField.addEventListener('input', () => {
      const value = parseFloat(thresholdField.value);
      behavior.similarity_threshold = Number.isFinite(value)
        ? Math.min(MAX_BEHAVIOR_TRIGGER_THRESHOLD, Math.max(MIN_BEHAVIOR_TRIGGER_THRESHOLD, value))
        : null;
    });

    cardsEl.appendChild(card);
  });

  enabledEl.addEventListener('change', () => {
    cfg.enabled = enabledEl.checked;
    modelEl.disabled = !cfg.enabled;
    thresholdEl.disabled = !cfg.enabled;
    renderBehaviorTriggers();
  });
  modelEl.addEventListener('change', () => {
    cfg.embedding_model_key = modelEl.value || null;
  });
  thresholdEl.addEventListener('input', () => {
    const value = parseFloat(thresholdEl.value);
    cfg.default_similarity_threshold = Number.isFinite(value)
      ? Math.min(MAX_BEHAVIOR_TRIGGER_THRESHOLD, Math.max(MIN_BEHAVIOR_TRIGGER_THRESHOLD, value))
      : 0.90;
  });
}

// ── Routing matrix ─────────────────────────────────────────────────

function renderRoutingMatrix() {
  // Only show armed agents (user agent is always included)
  const agents = agentConfig.agents.filter((a) => a.type === 'user' || a.armed !== false);
  if (agents.length === 0) {
    routingTable.innerHTML = '<tr><td style="color:#888;">No agents configured.</td></tr>';
    renderConnectionRules();
    return;
  }

  // Only enabled rules count as active in the matrix
  const connSet = new Set(
    agentConfig.connections
      .filter((c) => c.enabled !== false)
      .map((c) => c.from + '→' + c.to)
  );

  // Manager badge in column headers
  function agentLabel(a) {
    return escHtml(a.name) + (a.is_manager ? ' <span class="manager-badge">M</span>' : '');
  }

  let html = '<tr><th>From \\ To</th>';
  agents.forEach((a) => { html += '<th>' + agentLabel(a) + '</th>'; });
  html += '</tr>';

  agents.forEach((rowAgent) => {
    html += '<tr><th>' + agentLabel(rowAgent) + '</th>';
    agents.forEach((colAgent) => {
      if (rowAgent.id === colAgent.id) {
        html += '<td class="diagonal">-</td>';
      } else {
        const key = rowAgent.id + '→' + colAgent.id;
        const checked = connSet.has(key) ? ' checked' : '';
        html += '<td><input type="checkbox" data-from="' + rowAgent.id + '" data-to="' + colAgent.id + '"' + checked + ' /></td>';
      }
    });
    html += '</tr>';
  });

  routingTable.innerHTML = html;

  // Bind checkbox events
  routingTable.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    cb.addEventListener('change', () => {
      const from = cb.dataset.from;
      const to   = cb.dataset.to;
      if (cb.checked) {
        const existing = agentConfig.connections.find((c) => c.from === from && c.to === to);
        if (existing) {
          existing.enabled = true;
        } else {
          agentConfig.connections.push({ from, to, priority: 128, enabled: true });
        }
      } else {
        // Remove rule entirely (priority/condition are lost — acceptable trade-off)
        agentConfig.connections = agentConfig.connections.filter(
          (c) => !(c.from === from && c.to === to)
        );
      }
      renderConnectionRules();
    });
  });

  renderConnectionRules();
}

function renderConnectionRules() {
  const rulesDiv = document.getElementById('connection-rules');
  if (!rulesDiv) return;
  const active = agentConfig.connections.filter((c) => c.enabled !== false);
  if (active.length === 0) {
    rulesDiv.innerHTML = '';
    return;
  }

  rulesDiv.innerHTML = '<h4>Connection Rules</h4>';
  active.forEach((rule) => {
    const fromAgent = agentConfig.agents.find((a) => a.id === rule.from);
    const toAgent   = agentConfig.agents.find((a) => a.id === rule.to);
    const fromName  = fromAgent ? fromAgent.name : rule.from;
    const toName    = toAgent   ? toAgent.name   : rule.to;

    const div = document.createElement('div');
    div.className = 'conn-rule';
    div.innerHTML =
      '<span class="conn-label">' + escHtml(fromName) + ' → ' + escHtml(toName) + '</span>' +
      '<label>Priority: <input type="number" class="conn-priority" min="0" max="255" value="' + (rule.priority != null ? rule.priority : 128) + '" /></label>' +
      '<label>Condition: <input type="text" class="conn-condition" value="' + escHtml(rule.condition || '') + '" placeholder="keyword filter…" /></label>';

    div.querySelector('.conn-priority').addEventListener('input', (e) => {
      rule.priority = parseInt(e.target.value, 10);
      if (isNaN(rule.priority)) rule.priority = 128;
    });
    div.querySelector('.conn-condition').addEventListener('input', (e) => {
      rule.condition = e.target.value || null;
    });

    rulesDiv.appendChild(div);
  });
}

// ── Chat input state ───────────────────────────────────────────────

function updateChatInputState() {
  if (!activeWorkspaceId) {
    chatInput.placeholder = 'Sélectionne un workspace pour commencer…';
    chatInput.disabled = true;
  } else if (!activeThreadId) {
    chatInput.placeholder = 'Envoyer un message créera un nouveau thread…';
    chatInput.disabled = false;
  } else {
    const ws = workspaceConfig.workspaces.find((w) => w.id === activeWorkspaceId);
    const thread = ws && ws.threads.find((t) => t.id === activeThreadId);
    chatInput.placeholder = thread ? 'Message — ' + thread.name : 'Message…';
    chatInput.disabled = false;
  }
}

// ── Util ───────────────────────────────────────────────────────────

function escHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Sidebar / Workspaces ───────────────────────────────────────────

let workspaceConfig = { workspaces: [] };
let activeWorkspaceId = null;
let activeThreadId = null;

function normalizeThreadState(thread) {
  if (!thread) return;
  if (!Array.isArray(thread.message_items)) {
    thread.message_items = [];
  }
  if (!Array.isArray(thread.memory_entries)) {
    thread.memory_entries = [];
  }
  if (!Array.isArray(thread.command_history)) {
    thread.command_history = [];
  }
  if (typeof thread.messages !== 'string') {
    thread.messages = '';
  }
}

async function captureThreadState(workspaceId, threadId) {
  if (!workspaceId || !threadId) return;
  const ws = workspaceConfig.workspaces.find((w) => w.id === workspaceId);
  const thread = ws && ws.threads.find((t) => t.id === threadId);
  if (!thread) return;

  normalizeThreadState(thread);
  thread.message_items = cloneThreadMessages(currentThreadMessages);
  thread.messages = '';

  try {
    const memory = await invoke('get_memory_pool');
    thread.memory_entries = Array.isArray(memory.entries) ? memory.entries : [];
  } catch (e) {
    console.error('Failed to capture thread memory:', e);
  }

  try {
    const commandHistory = await invoke('get_command_history');
    thread.command_history = Array.isArray(commandHistory.entries) ? commandHistory.entries : [];
  } catch (e) {
    console.error('Failed to capture thread command history:', e);
  }
}

async function restoreThreadState(thread) {
  normalizeThreadState(thread);
  if (thread.message_items.length === 0 && thread.messages) {
    thread.message_items = parseLegacyThreadHtml(thread.messages);
  }
  currentThreadMessages = cloneThreadMessages(thread.message_items);
  if (currentThreadMessages.length > 0) {
    renderThreadMessages(currentThreadMessages);
  } else {
    messagesDiv.innerHTML = thread.messages || '';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }
  streamingBubbles = {};
  pendingToolEl = {};

  try {
    await invoke('set_memory_pool', {
      entries: Array.isArray(thread.memory_entries) ? thread.memory_entries : [],
    });
  } catch (e) {
    console.error('Failed to restore thread memory:', e);
  }

  try {
    await invoke('set_command_history', {
      entries: Array.isArray(thread.command_history) ? thread.command_history : [],
    });
  } catch (e) {
    console.error('Failed to restore thread command history:', e);
  }
}

async function clearActiveThreadState() {
  currentThreadMessages = [];
  messagesDiv.innerHTML = '';
  streamingBubbles = {};
  pendingToolEl = {};

  try {
    await invoke('set_memory_pool', { entries: [] });
  } catch (e) {
    console.error('Failed to clear thread memory:', e);
  }

  try {
    await invoke('clear_command_history');
  } catch (e) {
    console.error('Failed to clear thread command history:', e);
  }
}

function generateWsId() {
  return 'ws-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
}
function generateThreadId() {
  return 'thread-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
}

async function loadWorkspaceConfig() {
  try {
    workspaceConfig = await invoke('load_workspace_config');
    workspaceConfig.workspaces.forEach((ws) => {
      ws.threads.forEach((thread) => {
        normalizeThreadState(thread);
      });
    });
    currentThreadMessages = [];
    await invoke('set_memory_pool', { entries: [] });
    await invoke('clear_command_history');
    renderWorkspaceList();
  } catch (e) {
    console.error('Failed to load workspace config:', e);
  }
}

async function saveWorkspaceConfig() {
  try {
    await invoke('save_workspace_config', { config: workspaceConfig });
  } catch (e) {
    console.error('Failed to save workspace config:', e);
  }
}

function renderWorkspaceList() {
  const list = document.getElementById('workspace-list');
  list.innerHTML = '';

  workspaceConfig.workspaces.forEach((ws) => {
    const item = document.createElement('div');
    item.className = 'ws-item' + (ws._open ? ' open' : '');
    item.dataset.id = ws.id;

    // Header row
    const header = document.createElement('div');
    header.className = 'ws-header' + (ws.id === activeWorkspaceId ? ' active' : '');
    header.innerHTML =
      '<span class="ws-chevron">&#9658;</span>' +
      '<span class="ws-icon">&#128193;</span>' +
      '<span class="ws-name" title="' + escHtml(ws.path) + '">' + escHtml(ws.name) + '</span>' +
      '<button class="ws-del" title="Remove workspace">&#10005;</button>';

    header.addEventListener('click', async (e) => {
      if (e.target.classList.contains('ws-del')) return;
      await captureThreadState(activeWorkspaceId, activeThreadId);
      await saveWorkspaceConfig();
      // Select this workspace, deselect thread
      activeWorkspaceId = ws.id;
      activeThreadId = null;
      // Notify backend of active sandbox path
      invoke('set_active_workspace', { path: ws.path }).catch(() => {});
      await clearActiveThreadState();
      // Toggle open/close
      ws._open = !ws._open;
      item.classList.toggle('open', ws._open);
      updateChatInputState();
      renderWorkspaceList();
    });
    header.querySelector('.ws-del').addEventListener('click', async () => {
      workspaceConfig.workspaces = workspaceConfig.workspaces.filter((w) => w.id !== ws.id);
      if (activeWorkspaceId === ws.id) {
        activeWorkspaceId = null;
        activeThreadId = null;
        await clearActiveThreadState();
        updateChatInputState();
      }
      saveWorkspaceConfig();
      renderWorkspaceList();
    });

    // Path subtitle
    const pathEl = document.createElement('div');
    pathEl.className = 'ws-path';
    pathEl.textContent = ws.path;

    // Threads
    const threadsEl = document.createElement('div');
    threadsEl.className = 'ws-threads';

    ws.threads.forEach((thread) => {
      const isActiveThread = thread.id === activeThreadId && ws.id === activeWorkspaceId;
      const tEl = document.createElement('div');
      tEl.className = 'thread-item' + (isActiveThread ? ' active' : '');
      tEl.innerHTML =
        '<span style="opacity:0.4;font-size:0.7rem;">&#9135;</span>' +
        '<span class="thread-name">' + escHtml(thread.name) + '</span>' +
        '<button class="thread-del" title="Delete thread">&#10005;</button>';
      tEl.addEventListener('click', async (e) => {
        if (e.target.classList.contains('thread-del')) return;
        await captureThreadState(activeWorkspaceId, activeThreadId);
        await saveWorkspaceConfig();
        activeWorkspaceId = ws.id;
        activeThreadId = thread.id;
        // Notify backend of active sandbox path
        invoke('set_active_workspace', { path: ws.path }).catch(() => {});
        await restoreThreadState(thread);
        updateChatInputState();
        renderWorkspaceList();
      });
      tEl.querySelector('.thread-del').addEventListener('click', async (e) => {
        e.stopPropagation();
        ws.threads = ws.threads.filter((t) => t.id !== thread.id);
        if (activeThreadId === thread.id) {
          activeThreadId = null;
          await clearActiveThreadState();
          updateChatInputState();
        }
        await saveWorkspaceConfig();
        renderWorkspaceList();
      });
      threadsEl.appendChild(tEl);
    });

    // Add thread button
    const addThread = document.createElement('div');
    addThread.className = 'thread-add';
    addThread.innerHTML = '<span>&#43;</span> New thread';
    addThread.addEventListener('click', () => {
      const n = ws.threads.length + 1;
      ws.threads.push({ id: generateThreadId(), name: 'Thread ' + n, messages: '', message_items: [] });
      saveWorkspaceConfig();
      ws._open = true;
      renderWorkspaceList();
    });
    threadsEl.appendChild(addThread);

    item.appendChild(header);
    item.appendChild(pathEl);
    item.appendChild(threadsEl);
    list.appendChild(item);
  });
}

document.getElementById('add-workspace-btn').addEventListener('click', async () => {
  try {
    const path = await invoke('pick_folder');
    if (!path) return;
    const name = path.split('/').filter(Boolean).pop() || path;
    workspaceConfig.workspaces.push({
      id: generateWsId(),
      name,
      path,
      threads: [{ id: generateThreadId(), name: 'Thread 1', messages: '', message_items: [] }],
      _open: true,
    });
    await saveWorkspaceConfig();
    renderWorkspaceList();
  } catch (e) {
    appendMessage('System', 'Failed to pick folder: ' + e, false, true);
  }
});

// Sidebar collapse toggle
const sidebar = document.getElementById('sidebar');
document.getElementById('sidebar-toggle').addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
});

// ── Init ───────────────────────────────────────────────────────────

(async function init() {
  await fetchAvailableModels();
  await loadConfig();
  await refreshLoadedModels();
  await loadWorkspaceConfig();
  updateChatInputState();
})();
