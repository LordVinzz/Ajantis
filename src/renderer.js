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

let agentConfig = { agents: [], connections: [], command_policy: { allowlist: [], denylist: [] } };
let availableModels = []; // ModelInfo[] from backend
let modelDetails = {};    // key -> ModelInfo

// ── Tabs ───────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'routing') { renderRoutingMatrix(); renderConnectionRules(); }
    if (btn.dataset.tab === 'agents') { renderAgentsList(); renderCommandPolicy(); }
  });
});

// ── Markdown ───────────────────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return '';
  const rawHtml = marked.parse(text);
  return DOMPurify.sanitize(rawHtml);
}

// ── Chat ───────────────────────────────────────────────────────────

function appendMessage(agentName, content, isSent, isError) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + (isSent ? 'sent' : 'received');

  const nameEl = document.createElement('div');
  nameEl.className = 'msg-agent-name';
  nameEl.textContent = agentName;
  wrapper.appendChild(nameEl);

  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';
  if (isError) {
    bodyEl.style.color = '#c00';
    bodyEl.textContent = content;
  } else {
    bodyEl.innerHTML = renderMarkdown(content);
  }
  wrapper.appendChild(bodyEl);

  messagesDiv.appendChild(wrapper);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
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

function getOrCreateBubble(agentId, agentName) {
  if (streamingBubbles[agentId]) return streamingBubbles[agentId];

  const forUser = agentRoutesToUser(agentId);
  const agent = agentConfig.agents.find((a) => a.id === agentId);
  const isManager = agent && agent.is_manager;

  const wrapper = document.createElement('div');
  wrapper.className = 'message received' + (forUser ? ' for-user' : ' internal');

  const nameEl = document.createElement('div');
  nameEl.className = 'msg-agent-name';
  nameEl.textContent = agentName;
  wrapper.appendChild(nameEl);

  if (!forUser) {
    const hint = document.createElement('div');
    hint.className = 'msg-collapsed-hint';
    hint.textContent = '(click to collapse)';
    wrapper.appendChild(hint);
    wrapper.classList.add('expanded'); // start expanded so tool calls & sub-agents are visible
    wrapper.addEventListener('click', (e) => {
      // Don't collapse when clicking inside a tool item
      if (e.target.closest && e.target.closest('.tool-item')) return;
      wrapper.classList.toggle('expanded');
      hint.textContent = wrapper.classList.contains('expanded') ? '(click to collapse)' : '(click to expand)';
    });
  }

  // Tool calls section (only rendered for manager agents)
  const toolsEl = document.createElement('div');
  toolsEl.className = 'tool-calls msg-body';
  wrapper.appendChild(toolsEl);

  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';
  wrapper.appendChild(bodyEl);
  messagesDiv.appendChild(wrapper);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  const bubble = { wrapper, bodyEl, toolsEl, content: '' };
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
      const newThread = { id: generateThreadId(), name: threadName };
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

      const item = document.createElement('div');
      item.className = 'tool-item';

      const argsPreview = evt.args.replace(/\s+/g, ' ').slice(0, 80) + (evt.args.length > 80 ? '…' : '');

      item.innerHTML =
        '<div class="tool-header">' +
          '<span class="tool-icon">⚙</span>' +
          '<span class="tool-name">' + escHtml(evt.tool_name) + '</span>' +
          '<span class="tool-args-preview">' + escHtml(argsPreview) + '</span>' +
          '<span class="tool-status pending">running…</span>' +
        '</div>' +
        '<div class="tool-detail">' +
          '<div class="tool-detail-label">Args</div>' +
          '<div class="tool-args-text">' + escHtml(evt.args) + '</div>' +
        '</div>';

      item.addEventListener('click', () => item.classList.toggle('expanded'));
      bubble.toolsEl.appendChild(item);
      pendingToolEl[evt.agent_id] = item;
      messagesDiv.scrollTop = messagesDiv.scrollHeight;

    } else if (evt.event === 'tool_result') {
      const item = pendingToolEl[evt.agent_id];
      if (item) {
        const statusEl = item.querySelector('.tool-status');
        const isError = evt.result.startsWith('[tool call failed') || evt.result.startsWith('[tool returned no text]');
        statusEl.textContent = isError ? '✗' : '✓';
        statusEl.className = 'tool-status ' + (isError ? 'error' : 'done');

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
        bubble.bodyEl.innerHTML = renderMarkdown(bubble.content);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
      }

    } else if (evt.event === 'agent_end') {
      delete streamingBubbles[evt.agent_id];
      delete pendingToolEl[evt.agent_id];

    } else if (evt.event === 'error') {
      const bubble = getOrCreateBubble(evt.agent_id, evt.agent_name);
      bubble.bodyEl.style.color = '#c00';
      bubble.bodyEl.textContent = evt.message;
      delete streamingBubbles[evt.agent_id];
      delete pendingToolEl[evt.agent_id];

    } else if (evt.event === 'done') {
      streamingBubbles = {};
      pendingToolEl = {};
      refreshLoadedModels();
      // Snapshot + persist the conversation for the current thread
      if (activeThreadId) {
        const snapshot = messagesDiv.innerHTML;
        threadMessages[activeThreadId] = snapshot;
        // Persist into workspaceConfig so it survives restarts
        const ws = workspaceConfig.workspaces.find((w) => w.id === activeWorkspaceId);
        const thread = ws && ws.threads.find((t) => t.id === activeThreadId);
        if (thread) {
          thread.messages = snapshot;
          // We'll handle memory entries in a more robust way later, 
          // but for now, we must ensure the backend gets them on load.
          saveWorkspaceConfig();
        }
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
    if (!agentConfig.command_policy) agentConfig.command_policy = { allowlist: [], denylist: [] };
    renderAgentsList();
    renderCommandPolicy();
    renderRoutingMatrix();
  } catch (err) {
    appendMessage('System', 'Failed to load config: ' + err, false, true);
  }
}

async function saveConfig() {
  try {
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

  if (!agentConfig.command_policy) {
    agentConfig.command_policy = { allowlist: [], denylist: [] };
  }
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
// Per-thread conversation snapshots: threadId → messagesDiv.innerHTML
let threadMessages = {};

function generateWsId() {
  return 'ws-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
}
function generateThreadId() {
  return 'thread-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
}

async function loadWorkspaceConfig() {
  try {
    workspaceConfig = await invoke('load_workspace_config');
    // Restore in-memory snapshots from persisted thread data
    threadMessages = {};
    workspaceConfig.workspaces.forEach((ws) => {
      ws.threads.forEach((thread) => {
        if (thread.messages) {
          threadMessages[thread.id] = thread.messages;
        }
      });
    });
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

    header.addEventListener('click', (e) => {
      if (e.target.classList.contains('ws-del')) return;
      // Select this workspace, deselect thread
      activeWorkspaceId = ws.id;
      activeThreadId = null;
      // Notify backend of active sandbox path
      invoke('set_active_workspace', { path: ws.path }).catch(() => {});
      // Clear chat — no thread selected yet
      messagesDiv.innerHTML = '';
      streamingBubbles = {};
      pendingToolEl = {};
      // Toggle open/close
      ws._open = !ws._open;
      item.classList.toggle('open', ws._open);
      updateChatInputState();
      renderWorkspaceList();
    });
    header.querySelector('.ws-del').addEventListener('click', () => {
      workspaceConfig.workspaces = workspaceConfig.workspaces.filter((w) => w.id !== ws.id);
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
      tEl.addEventListener('click', (e) => {
        if (e.target.classList.contains('thread-del')) return;
        activeWorkspaceId = ws.id;
        activeThreadId = thread.id;
        // Notify backend of active sandbox path
        invoke('set_active_workspace', { path: ws.path }).catch(() => {});
        // Restore this thread's conversation
        messagesDiv.innerHTML = threadMessages[thread.id] || '';
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        streamingBubbles = {};
        pendingToolEl = {};
        updateChatInputState();
        renderWorkspaceList();
      });
      tEl.querySelector('.thread-del').addEventListener('click', (e) => {
        e.stopPropagation();
        ws.threads = ws.threads.filter((t) => t.id !== thread.id);
        if (activeThreadId === thread.id) {
          activeThreadId = null;
          delete threadMessages[thread.id];
          messagesDiv.innerHTML = '';
          updateChatInputState();
        }
        saveWorkspaceConfig();
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
      ws.threads.push({ id: generateThreadId(), name: 'Thread ' + n });
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
      threads: [{ id: generateThreadId(), name: 'Thread 1' }],
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
