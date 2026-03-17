let currentTier = 'strict';

// ---- Conversation History (localStorage) ----
const HIST_KEY = 'ep-conversations';
const HIST_MAX = 100;
let currentConvId = null;

function getConversations() {
  try { return JSON.parse(localStorage.getItem(HIST_KEY) || '[]'); } catch(e) { return []; }
}
function saveConversations(convs) {
  try {
    // LRU eviction: keep only the newest HIST_MAX
    if (convs.length > HIST_MAX) convs = convs.slice(0, HIST_MAX);
    localStorage.setItem(HIST_KEY, JSON.stringify(convs));
  } catch(e) {}
}
function addConversation(prompt, resultHtml) {
  const convs = getConversations();
  const conv = {
    id: 'c-' + Date.now(),
    title: prompt.substring(0, 60) + (prompt.length > 60 ? '...' : ''),
    created_at: new Date().toISOString(),
    prompt: prompt,
    html: resultHtml
  };
  convs.unshift(conv);
  saveConversations(convs);
  currentConvId = conv.id;
  renderHistory();
}
function renderHistory() {
  const list = document.getElementById('histList');
  if (!list) return;
  const convs = getConversations();
  if (convs.length === 0) {
    list.innerHTML = '<div class="history-empty">No conversations yet</div>';
    return;
  }
  let h = '';
  convs.forEach(function(c) {
    const dt = new Date(c.created_at);
    const dateStr = dt.toLocaleDateString() + ' ' + dt.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    const cls = c.id === currentConvId ? ' active' : '';
    h += '<div class="history-item' + cls + '" onclick="loadConversation(\'' + c.id + '\')" title="' + esc(c.title).replace(/"/g,'&quot;') + '">';
    h += esc(c.title);
    h += '<span class="history-date">' + dateStr + '</span>';
    h += '</div>';
  });
  list.innerHTML = h;
}
function loadConversation(id) {
  const convs = getConversations();
  const conv = convs.find(function(c) { return c.id === id; });
  if (!conv) return;
  currentConvId = id;
  const ch = document.getElementById('ch');
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
  // Clear and restore
  ch.innerHTML = conv.html;
  toggleHistory();
  renderHistory();
}
function toggleHistory() {
  document.getElementById('histSidebar').classList.toggle('open');
  document.getElementById('histOverlay').classList.toggle('open');
}

// ---- Theme ----
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  try { localStorage.setItem('ep-theme', next); } catch(e) {}
}
(function initTheme() {
  try {
    const saved = localStorage.getItem('ep-theme');
    if (saved) { document.documentElement.setAttribute('data-theme', saved); return; }
  } catch(e) {}
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();

// ---- Feedback ----
async function sendFeedback(btn, rating, prompt) {
  const row = btn.parentElement;
  row.querySelectorAll('.fb-btn').forEach(function(b) { b.disabled = true; });
  btn.classList.add('fb-active');
  try {
    await fetch('/api/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({rating: rating, prompt: prompt})
    });
    row.innerHTML = '<span class="fb-thanks">Thanks for your feedback!</span>';
  } catch(e) {
    row.innerHTML = '<span class="fb-thanks">Could not send feedback.</span>';
  }
}

// ---- Rate Limit Counter ----
async function updateRateLimit() {
  try {
    const r = await fetch('/api/rate-limit');
    if (r.ok) {
      const d = await r.json();
      const el = document.getElementById('rl-counter');
      if (el) {
        el.textContent = d.remaining + '/' + d.limit;
        el.title = d.remaining + ' requests remaining this minute';
        el.className = 'rl-counter' + (d.remaining <= 3 ? ' rl-warn' : '');
      }
    }
  } catch(e) { /* best effort */ }
}
setInterval(updateRateLimit, 10000);

const tierDescs = {
  strict: 'Full Audit v8 rules \u2014 all claims verified, typicality stripped, bare stats require citations',
  standard: 'Balanced verification \u2014 evidence rules applied, softer thresholds for natural prose',
  light: 'Fact-check only \u2014 catches hallucinations but allows natural language and inference',
};
function setTier(el) {
  document.querySelectorAll('#tier-pills .tier-pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  currentTier = el.dataset.val;
  document.getElementById('tier-desc').textContent = tierDescs[currentTier] || '';
}
function tryExample(el) {
  const inp = document.getElementById('ui');
  inp.value = el.textContent;
  inp.focus();
  inp.dispatchEvent(new Event('input'));
}
function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}
function togglePipeline(btn, id) {
  btn.classList.toggle('open');
  document.getElementById(id).classList.toggle('open');
}
function tog() {
  document.getElementById('cd').classList.toggle('open');
  document.getElementById('co').classList.toggle('open');
}
function openStress() { document.getElementById('sp').classList.add('open'); }
function closeStress() { document.getElementById('sp').classList.remove('open'); }
function clearChat() {
  const ch = document.getElementById('ch');
  ch.innerHTML = '';
  const w = document.createElement('div');
  w.className = 'welcome';
  w.id = 'welcome';
  w.style.display = '';
  w.innerHTML = document.getElementById('welcomeTpl').innerHTML;
  ch.appendChild(w);
  document.getElementById('newChatBtn').classList.remove('show');
  document.getElementById('ui').focus();
}
function setStage(n) {
  ['stg1','stg2','stg3'].forEach(function(id, i) {
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('stage-active', 'stage-done');
    if (i + 1 === n) el.classList.add('stage-active');
    else if (i + 1 < n) el.classList.add('stage-done');
  });
}
function clearStages() {
  ['stg1','stg2','stg3'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.classList.remove('stage-active', 'stage-done');
  });
}

async function runStress() {
  const btn = document.getElementById('sr');
  const log = document.getElementById('sl');
  const scoreDiv = document.getElementById('ss');
  btn.disabled = true;
  log.innerHTML = '';
  scoreDiv.innerHTML = '';

  const cat = document.getElementById('sc').value || null;
  const cnt = parseInt(document.getElementById('sn').value) || null;
  const tier = document.getElementById('st').value;

  log.innerHTML = 'Starting stress test...\\n';
  let allResults = [];
  let gotSummary = false;
  const maxResumes = 5;   // prevent infinite reconnect loops
  let resumeCount = 0;
  let startIndex = 0;

  while (resumeCount <= maxResumes) {
    const body = {tier: tier, start_index: startIndex};
    if (cat) body.category = cat;
    if (cnt) body.count = cnt;

    let batchDone = false;
    try {
      const resp = await fetch('/api/stress', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const err = await resp.json();
        log.innerHTML += '<span class="fail">ERROR: ' + esc(err.detail || 'Request failed') + '</span>';
        break;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const {done, value} = await reader.read();
        if (done) { batchDone = true; break; }
        buf += decoder.decode(value, {stream: true});
        const lines = buf.split('\\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          const d = JSON.parse(line);
          if (d.type === 'heartbeat') continue;
          if (d.type === 'progress') {
            allResults.push(d);
            let cls = d.verdict === 'PASS' ? 'pass' : 'fail';
            let extra = '';
            if (d.error) extra += ' <span class="fail">' + esc(d.error) + '</span>';
            if (d.arbiter) extra += ' <span class="arb">arbiter:' + esc(d.arbiter) + '</span>';
            if (d.rewrite) extra += ' <span class="rew">[rewrite]</span>';
            log.innerHTML += '[' + d.index + '/' + d.total + '] ' + esc(d.id) + ' <span class="' + cls + '">' + d.verdict + '</span>' + extra + ' (' + d.duration_s + 's)\\n';
            log.scrollTop = log.scrollHeight;
          } else if (d.type === 'summary') {
            gotSummary = true;
            renderStressSummary(d, scoreDiv);
          }
        }
      }
    } catch(e) {
      // Stream dropped — try to resume
      const completed = allResults.length;
      const total = allResults.length > 0 ? allResults[0].total : '?';
      if (completed > 0 && completed < total && resumeCount < maxResumes) {
        resumeCount++;
        startIndex = completed;
        log.innerHTML += '<span class="arb">\\u21bb Connection lost — auto-resuming from test ' + (completed + 1) + '/' + total + ' (retry ' + resumeCount + '/' + maxResumes + ')...</span>\\n';
        log.scrollTop = log.scrollHeight;
        await new Promise(r => setTimeout(r, 1000)); // brief pause before retry
        continue;
      }
      log.innerHTML += '\\n<span class="fail">CONNECTION LOST after ' + completed + '/' + total + ' tests (' + esc(e.message) + ')</span>\\n';
      if (completed > 0 && !gotSummary) {
        log.innerHTML += '<span class="arb">Computing partial results...</span>\\n';
        renderPartialSummary(allResults, scoreDiv);
      }
    }
    break; // normal completion or unrecoverable error
  }
  btn.disabled = false;
}

function renderPartialSummary(results, el) {
  const total = results.length;
  const pass = results.filter(r => r.verdict === 'PASS').length;
  const fail = results.filter(r => r.verdict === 'FAIL').length;
  const errors = results.filter(r => r.verdict === 'ERROR').length;
  const avgDur = (results.reduce((s, r) => s + r.duration_s, 0) / total).toFixed(2);
  const passRate = ((pass / total) * 100).toFixed(1);

  let cls = 's0', band = 'PARTIAL — stream interrupted';
  if (passRate >= 90) cls = 's90';
  else if (passRate >= 75) cls = 's75';
  else if (passRate >= 60) cls = 's60';

  let h = '<div style="text-align:center;margin-bottom:8px;font-size:11px;color:var(--accent-amber);font-weight:600;letter-spacing:0.05em;">PARTIAL RESULTS (' + total + ' of ' + (results[0] ? results[0].total : '?') + ' tests)</div>';
  h += '<div class="pss-big ' + cls + '">' + passRate + '%</div>';
  h += '<div class="pss-band">' + band + '</div>';
  h += '<div style="text-align:center;font-size:12px;color:var(--text-muted);margin-bottom:16px;">Tests: ' + total + ' | PASS: ' + pass + ' | FAIL: ' + fail + ' | Errors: ' + errors + ' | Avg: ' + avgDur + 's</div>';

  // Category breakdown from partial data
  const cats = {};
  results.forEach(r => {
    const c = r.id.replace(/_\\d+$/, '');
    if (!cats[c]) cats[c] = {total: 0, pass: 0, fail: 0};
    cats[c].total++;
    if (r.verdict === 'PASS') cats[c].pass++;
    else cats[c].fail++;
  });
  h += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin-top:12px;">';
  Object.keys(cats).sort().forEach(c => {
    const d = cats[c];
    const rate = ((d.pass / d.total) * 100).toFixed(0);
    h += '<div class="metric-card"><div class="mv">' + rate + '%</div><div class="ml">' + esc(c) + '</div><div class="mp">' + d.pass + '/' + d.total + '</div></div>';
  });
  h += '</div>';
  el.innerHTML = h;
}

function renderStressSummary(d, el) {
  const pss = d.pss;
  const s = pss.score;
  let cls = 's0';
  let band = 'NOT STABLE';
  if (s >= 90) { cls = 's90'; band = 'PRODUCTION-STABLE'; }
  else if (s >= 75) { cls = 's75'; band = 'USABLE (needs calibration)'; }
  else if (s >= 60) { cls = 's60'; band = 'BRITTLE / PERMISSIVE'; }

  let h = '<div class="pss-big ' + cls + '">' + s.toFixed(1) + '</div>';
  h += '<div class="pss-band">' + band + '</div>';
  h += '<div style="text-align:center;font-size:12px;color:var(--text-muted);margin-bottom:16px;">Tests: ' + d.total_tests + ' | PASS: ' + d.total_pass + ' | FAIL: ' + d.total_fail + ' | Errors: ' + d.total_error + ' | Avg: ' + d.avg_duration_s + 's</div>';

  // Metrics cards
  const m = pss.metrics;
  const p = pss.penalties;
  const cards = [
    {label: 'HLR', desc: 'Hallucination Leakage', val: ((m.HLR||0)*100).toFixed(1) + '%', pen: p.P1},
    {label: 'FPF', desc: 'False-Positive FAIL', val: (m.FPF*100).toFixed(1) + '%', pen: p.P2},
    {label: 'MCP', desc: 'Min Compliance Pass', val: (m.MCP*100).toFixed(1) + '%', pen: p.P3},
    {label: 'RLS', desc: 'Rewrite Loop Avg', val: m.RLS.toFixed(2), pen: p.P4},
    {label: 'EOI', desc: 'Overreach Index', val: (m.EOI*100).toFixed(1) + '%', pen: p.P5},
  ];
  h += '<div class="metrics-grid">';
  cards.forEach(c => {
    h += '<div class="metric-card"><div class="mv">' + c.val + '</div><div class="ml">' + c.label + '</div><div class="ml">' + c.desc + '</div><div class="mp">-' + c.pen.toFixed(1) + '</div></div>';
  });
  h += '</div>';

  // Category table
  const cats = d.categories;
  h += '<table class="cat-table"><tr><th>Category</th><th>Pass</th><th>Fail</th><th>Err</th><th>Rate</th><th>Rewrites</th><th>Arbiter</th></tr>';
  for (const cat in cats) {
    const c = cats[cat];
    const rate = c.total > 0 ? ((c.pass / c.total) * 100).toFixed(0) : '0';
    const rc = parseInt(rate) >= 80 ? 'pass' : parseInt(rate) >= 50 ? 'arb' : 'fail';
    h += '<tr><td>' + esc(cat) + '</td><td>' + c.pass + '</td><td>' + c.fail + '</td><td>' + c.error + '</td><td class="pr"><span class="' + rc + '">' + rate + '%</span></td><td>' + c.rewrites + '</td><td>' + c.arbiter + '</td></tr>';
  }
  h += '</table>';

  // Top violations
  const viols = d.top_violations;
  if (Object.keys(viols).length > 0) {
    h += '<div style="margin-top:16px;font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">Top Violation Reasons</div>';
    for (const v in viols) {
      h += '<div style="font-size:12px;color:var(--accent-rose);margin:4px 0;padding:4px 8px;background:rgba(var(--accent-rose-rgb),0.04);border-radius:4px;">' + esc(v) + ': ' + viols[v] + '</div>';
    }
  }

  el.innerHTML = h;
}

async function lc() {
  try {
    const r = await fetch('/api/openai/config');
    const d = await r.json();
    const st = document.getElementById('st-oai');
    if (!st) return;
    if (d.key_set) {
      st.textContent = d.key_preview + ' | ' + d.model;
      st.className = 'cfg-st ok';
    } else {
      st.textContent = 'No key set';
      st.className = 'cfg-st';
    }
  } catch(e) {}
}

async function saveOai() {
  const k = document.getElementById('cfg-oai-key').value.trim();
  if (!k) return;
  await fetch('/api/openai/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({api_key: k, model: 'gpt-4o-mini'})
  });
  document.getElementById('cfg-oai-key').value = '';
  lc();
}

async function loadTav() {
  const st = document.getElementById('st-tav');
  try {
    const r = await fetch('/api/tavily/config');
    const d = await r.json();
    if (!st) return;
    if (d.key_set) {
      document.getElementById('cfg-tav-en').checked = d.enabled;
      st.textContent = d.enabled ? 'Search enabled' : 'Search disabled';
      st.className = d.enabled ? 'cfg-st ok' : 'cfg-st';
    } else {
      document.getElementById('cfg-tav-en').checked = false;
      st.textContent = 'No Tavily key set';
      st.className = 'cfg-st';
    }
  } catch(e) { if(st) st.textContent = 'Error loading config'; }
}

async function saveTav() {
  const k = document.getElementById('cfg-tav-key').value.trim();
  if (!k) return;
  const en = document.getElementById('cfg-tav-en').checked;
  await fetch('/api/tavily/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({api_key: k, enabled: en})
  });
  document.getElementById('cfg-tav-key').value = '';
  loadTav();
}

async function toggleTav(en) {
  try {
    await fetch('/api/tavily/toggle?enabled=' + en, {method: 'POST'});
    document.getElementById('st-tav').textContent = en ? 'Search enabled' : 'Search disabled';
    document.getElementById('st-tav').className = en ? 'cfg-st ok' : 'cfg-st';
  } catch(e) {
    document.getElementById('cfg-tav-en').checked = !en;
  }
}

function openCfg() {
  document.getElementById('cfg-drawer')?.classList.add('open');
  document.getElementById('cfg-overlay')?.classList.add('open');
}

function closeCfg() {
  document.getElementById('cfg-drawer')?.classList.remove('open');
  document.getElementById('cfg-overlay')?.classList.remove('open');
}

function stress() {
  ab('sys', 'SYSTEM', 'Diagnostic stress tests are not implemented in the new terminal interface.');
}

function renderSearchSources(sources) {
  if (!sources || !sources.length) return '';
  let h = '<table class="src-tbl"><thead><tr><th>#</th><th>Trust</th><th>Title</th><th>Snippet</th></tr></thead><tbody>';
  sources.forEach(function(s, i) {
    let trustClass = (s.trust_tier || 'Unknown').toLowerCase();
    h += '<tr><td>[' + (i+1) + ']</td>';
    h += '<td><span class="trust-badge ' + trustClass + '">' + esc(s.trust_tier || 'Unknown') + '</span></td>';
    h += '<td><a href="' + esc(s.url) + '" target="_blank">' + esc(s.title) + '</a></td>';
    h += '<td>' + esc(s.snippet ? s.snippet.slice(0, 180) : '') + '</td></tr>';
  });
  h += '</tbody></table>';
  return h;
}

function ab(cls, who, body) {
  const c = document.getElementById('ch');
  const d = document.createElement('div');
  d.className = 'b ' + cls;
  d.innerHTML = (who ? '<div class="w">' + who + '</div>' : '') + body;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}

function addDivider() {
  const c = document.getElementById('ch');
  const hr = document.createElement('hr');
  hr.className = 'divider';
  c.appendChild(hr);
}

function esc(t) {
  if (!t) return '';
  if (typeof t === 'object') t = JSON.stringify(t);
  const d = document.createElement('div');
  d.textContent = t;
  return d.innerHTML;
}

function catCls(cat) {
  const c = (cat || '').toLowerCase();
  if (c === 'supported') return 'cat-sup';
  if (c === 'inference') return 'cat-inf';
  if (c === 'hypothesis') return 'cat-hyp';
  if (c === 'unsupported') return 'cat-uns';
  if (c === 'user-provided') return 'cat-usr';
  return '';
}

function renderClaimTable(claims) {
  if (!claims || claims.length === 0) return '';
  let h = '<table class="ct"><tr><th>Claim</th><th>Category</th><th>Justification</th></tr>';
  claims.forEach(c => {
    h += '<tr><td>' + esc(c.claim) + '</td><td class="cat ' + catCls(c.category) + '">' + esc(c.category) + '</td><td>' + esc(c.justification) + '</td></tr>';
  });
  return h + '</table>';
}

function renderConfidence(conf) {
  if (!conf || conf.total_claims === 0) return '';
  let h = '<div class="conf-bar-wrap">';
  h += '<div class="conf-bar-label">Confidence Breakdown (' + conf.total_claims + ' claims)</div>';
  h += '<div class="conf-bar">';
  if (conf.observed_pct > 0) h += '<div class="seg seg-obs" style="width:' + conf.observed_pct + '%" title="Observed: ' + conf.observed_pct + '%"></div>';
  if (conf.inference_pct > 0) h += '<div class="seg seg-inf" style="width:' + conf.inference_pct + '%" title="Inference: ' + conf.inference_pct + '%"></div>';
  if (conf.hypothesis_pct > 0) h += '<div class="seg seg-hyp" style="width:' + conf.hypothesis_pct + '%" title="Hypothesis: ' + conf.hypothesis_pct + '%"></div>';
  if (conf.unsupported_pct > 0) h += '<div class="seg seg-uns" style="width:' + conf.unsupported_pct + '%" title="Unsupported: ' + conf.unsupported_pct + '%"></div>';
  if (conf.user_provided_pct > 0) h += '<div class="seg seg-usr" style="width:' + conf.user_provided_pct + '%" title="User-provided: ' + conf.user_provided_pct + '%"></div>';
  h += '</div>';
  h += '<div class="conf-legend">';
  if (conf.observed_pct > 0) h += '<span class="lg"><span class="dot" style="background:var(--accent-emerald)"></span>Observed ' + conf.observed_pct + '%</span>';
  if (conf.inference_pct > 0) h += '<span class="lg"><span class="dot" style="background:var(--accent-amber)"></span>Inference ' + conf.inference_pct + '%</span>';
  if (conf.hypothesis_pct > 0) h += '<span class="lg"><span class="dot" style="background:var(--accent-violet)"></span>Hypothesis ' + conf.hypothesis_pct + '%</span>';
  if (conf.unsupported_pct > 0) h += '<span class="lg"><span class="dot" style="background:var(--accent-rose)"></span>Unsupported ' + conf.unsupported_pct + '%</span>';
  if (conf.user_provided_pct > 0) h += '<span class="lg"><span class="dot" style="background:var(--accent-blue)"></span>User-provided ' + conf.user_provided_pct + '%</span>';
  h += '</div>';
  const lblCls = (conf.confidence_label || 'unknown').toLowerCase();
  h += '<div class="conf-badge ' + lblCls + '">Confidence: ' + esc(conf.confidence_label) + '</div>';
  if (conf.confidence_reasoning && conf.confidence_reasoning.length > 0) {
    h += '<details class="conf-reasoning"><summary>Why this confidence?</summary><ul>';
    conf.confidence_reasoning.forEach(function(r) { h += '<li>' + esc(r) + '</li>'; });
    h += '</ul></details>';
  }
  h += '</div>';
  return h;
}

const tripwireLabels = {
  'T1': 'Fabricated Evidence',
  'T2': 'Unsupported Evidence Reference',
  'T3': 'Causal Claim Stated as Fact',
  'T4': 'Missing Structural Qualifier',
  'T5': 'Prescriptive Creep',
  'T6': 'Reassurance Framing',
  'T7': 'Unverified Current Fact',
};

function expandViolation(v) {
  const trimmed = (v || '').trim();
  // Match standalone codes like "T7" or "T1"
  const m = trimmed.match(/^(T[1-7])$/);
  if (m && tripwireLabels[m[1]]) {
    return m[1] + ' \u2014 ' + tripwireLabels[m[1]];
  }
  // Match codes at the start like "T7: some detail"
  const m2 = trimmed.match(/^(T[1-7])(\s*[:\-]\s*)(.*)/);
  if (m2 && tripwireLabels[m2[1]]) {
    return m2[1] + ' (' + tripwireLabels[m2[1]] + ') \u2014 ' + m2[3];
  }
  return trimmed;
}

function inlineFormat(text) {
  var r = text;
  r = r.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  r = r.replace(/\\[(\\d+)\\]/g, '<span class="fo-cite">[$1]</span>');
  r = r.replace(/Unknown\\((Actionable|Structural)\\)/g, '<span class="fo-unknown">Unknown($1)</span>');
  r = r.replace(/\\[(verified|inference|unverified)\\]/gi, function(m, type) {
    var cls = type.toLowerCase() === 'verified' ? 'fo-tag-ver' : type.toLowerCase() === 'inference' ? 'fo-tag-inf' : 'fo-tag-unv';
    return '<span class="fo-tag ' + cls + '">' + type + '</span>';
  });
  r = r.replace(/^(High|Medium|Low)\\s*[\\u2014\\-]/, function(m, level) {
    return '<span class="fo-conf-lvl ' + level.toLowerCase() + '">' + level + '</span> \\u2014';
  });
  return r;
}

function formatOutput(text) {
  if (!text) return '';
  var escaped = esc(text);
  var lines = escaped.replace(/\\r\\n/g, '\\n').replace(/\\r/g, '\\n').split('\\n');
  var html = '<div class="fo-content">';
  var inSection = false;

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var trimmed = line.trim();
    if (!trimmed) continue;

    var cleanLine = trimmed.replace(/\\*\\*/g, '');
    
    // Strict Audit v8 Structural blocks
    if (/^(OBSERVED FACTS|FACTS)\\s*[:\\[]?/i.test(cleanLine)) {
      html += '<div class="fo-subhdr obs">[' + cleanLine.toUpperCase() + ']</div>';
      continue;
    }
    if (/^(UNKNOWNS|ACTIONABLE UNKNOWNS)\\s*[:\\[]?/i.test(cleanLine)) {
      html += '<div class="fo-subhdr unk">[' + cleanLine.toUpperCase() + ']</div>';
      continue;
    }
    if (/^(DISCRIMINATORS|VARIABLES)\\s*[:\\[]?/i.test(cleanLine)) {
      html += '<div class="fo-subhdr disc">[' + cleanLine.toUpperCase() + ']</div>';
      continue;
    }
    if (/^(INFERENCES)\\s*[:\\[]?/i.test(cleanLine)) {
      html += '<div class="fo-subhdr inf">[' + cleanLine.toUpperCase() + ']</div>';
      continue;
    }

    if (trimmed.match(/^[-\\u2022]\\s+/)) {
      var bulletText = trimmed.replace(/^[-\\u2022]\\s+/, '');
      html += '<div class="fo-bullet">' + inlineFormat(bulletText) + '</div>';
      continue;
    }

    html += '<p class="fo-para">' + inlineFormat(trimmed) + '</p>';
  }

  if (inSection) html += '</div>';
  html += '</div>';
  return html;
}

function renderViolations(viols) {
  if (!viols || viols.length === 0) return '<div class="no-viol">No violations detected</div>';
  let h = '<div class="viol">';
  viols.forEach(v => {
    h += '<div class="viol-item"><span class="viol-dot"></span>' + esc(expandViolation(v)) + '</div>';
  });
  return h + '</div>';
}

function editActionCls(a) {
  const al = (a||'').toUpperCase();
  if (al === 'DELETE') return 'del';
  if (al === 'REWRITE') return 'rew';
  if (al === 'MOVE_TO_UNKNOWN') return 'mtu';
  return '';
}

function makeLoader(spinnerCls, text) {
  return '<div class="sp ' + spinnerCls + '"><span class="dot-pulse" style="border-radius:0;"></span><span class="dot-pulse" style="border-radius:0;"></span><span class="dot-pulse" style="border-radius:0;"></span></div><span class="ld-text" style="font-family:var(--font-mono); font-size:11px; text-transform:uppercase;">' + text + '</span>';
}

async function go() {
  try {
    const inp = document.getElementById('ui');
    const btn = document.getElementById('sbtn');
    const prompt = inp.value.trim();
    if (!prompt) return;

    hideWelcome();
    ab('usr', 'You', esc(prompt));
    inp.value = '';
    if (btn) btn.disabled = true;

    const ch = document.getElementById('ch');
    const ld = document.createElement('div');
    ld.className = 'ld';
    // Note: makeLoader returns trusted static HTML with no user content
    ld.innerHTML = makeLoader('ss', 'Searching web...');
    ch.appendChild(ld);
    ch.scrollTop = ch.scrollHeight;

    setStage(0);
    const steps = [
      {t: 500, msg: '[SYS] Extracting logical parameters...', cls: '', stage: 1},
      {t: 1500, msg: '[L1] Generating base resolution...', cls: '', stage: 1},
      {t: 4000, msg: '[L2] Checking claims against T1-T7 guardrails...', cls: 's2', stage: 2},
      {t: 8000, msg: '[L3] Adjudicating flagged items...', cls: 's3', stage: 3},
      {t: 12000, msg: '[SYS] Enforcing rewrite constraints...', cls: '', stage: 1},
    ];
    // Note: makeLoader returns trusted static HTML with no user content
    const timers = steps.map(s => setTimeout(() => {
      if (ld.parentNode) ld.innerHTML = makeLoader(s.cls, s.msg);
      setStage(s.stage);
    }, s.t));

    try {
      const r = await fetch('/api/pipeline', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          prompt: prompt,
          gpt1_system: "",
          gpt2_system: "",
          gpt3_system: "",
          tier: currentTier,
          output_format: "annotated",
        })
      });

      timers.forEach(t => clearTimeout(t));
      clearStages();
      ld.remove();

      if (!r.ok) {
        const ct = r.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          const err = await r.json();
          ab('err', '', esc(err.detail || 'Request failed'));
        } else {
          const txt = await r.text();
          const hint = r.status === 504 || r.status === 502
            ? ' The pipeline may have timed out. Try again or use a simpler query.'
            : '';
          ab('err', '', 'Server error (HTTP ' + r.status + ').' + hint);
          console.error('Non-JSON error response:', r.status, txt.substring(0, 500));
        }
        return;
      }

      const d = await r.json();

      // ---- Build metadata strip ----
      let metaParts = [];
      const tierColors = {strict: 'var(--accent-rose)', standard: 'var(--accent-amber)', light: 'var(--accent-emerald)'};
      const tc = tierColors[d.tier] || 'var(--text-secondary)';
      metaParts.push('<span style="color:' + tc + ';font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">' + esc(d.tier) + '</span>');
      if (d.output_format) metaParts.push('<span style="color:var(--text-tertiary);">' + esc(d.output_format) + '</span>');
      metaParts.push('<span style="color:' + (d.final_verdict === 'PASS' ? 'var(--accent-emerald)' : 'var(--accent-rose)') + ';font-weight:600;">' + (d.final_verdict === 'PASS' ? '✓ PASS' : '✗ FAIL') + '</span>');
      if (d.search_performed) metaParts.push('<span style="color:var(--accent-violet);">web-grounded</span>');
      if (d.arbiter_invoked) metaParts.push('<span style="color:var(--accent-violet);">arbiter: ' + esc(d.arbiter_decision) + '</span>');
      if (d.confidence && d.confidence.confidence_label) metaParts.push('<span>confidence: ' + esc(d.confidence.confidence_label) + '</span>');
      const metaHtml = '<div class="fo-meta">' + metaParts.join('<span class="fo-meta-sep">&middot;</span>') + '</div>';

      // ---- Hero: Final Output with integrated metadata + feedback ----
      const fbHtml = '<div class="fb-row" aria-label="Was this verification correct?">' +
        '<span class="fb-label">Was this helpful?</span>' +
        '<button class="fb-btn fb-up" onclick="sendFeedback(this,\'accurate\',\''+esc(d.gpt1_input).replace(/\'/g,"\\\'")+'\')" aria-label="Mark as accurate" title="Accurate">&#x1F44D;</button>' +
        '<button class="fb-btn fb-down" onclick="sendFeedback(this,\'inaccurate\',\''+esc(d.gpt1_input).replace(/\'/g,"\\\'")+'\')" aria-label="Mark as inaccurate" title="Inaccurate">&#x1F44E;</button>' +
        '</div>';
      if (d.final_verdict === 'PASS') {
        ab('fo', 'Final Output', formatOutput(d.final_result) + metaHtml + fbHtml);
      } else {
        let blockMsg = 'Output blocked by verification pipeline';
        if (d.arbiter_invoked && d.arbiter_decision === 'BLOCK' && d.arbiter_rationale && d.arbiter_rationale.length > 0) {
          blockMsg += '\n\nArbiter rationale:\n' + d.arbiter_rationale.map(r => '• ' + r).join('\n');
        }
        ab('fo blk', 'Blocked', esc(blockMsg) + metaHtml + fbHtml);
      }

      // ---- L2: Verification Summary (auto-expand on FAIL) ----
      const l2Btn = document.createElement('button');
      l2Btn.className = 'pipeline-toggle l2-toggle';
      l2Btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg> Verification Summary';
      ch.appendChild(l2Btn);

      const l2Div = document.createElement('div');
      l2Div.className = 'pipeline-steps l2-summary';
      ch.appendChild(l2Div);

      l2Btn.onclick = function() {
        l2Btn.classList.toggle('open');
        l2Div.classList.toggle('open');
      };

      let l2Html = '<div class="pipeline-step done"><div class="pipeline-step-title" style="margin-bottom:8px;">Claim Table</div>' + renderClaimTable(d.claim_table) + '</div>';
      l2Html += '<div class="pipeline-step done"><div class="pipeline-step-title" style="margin-bottom:8px;">Violations</div>' + renderViolations(d.violations) + '</div>';
      l2Html += '<div class="pipeline-step done"><div class="pipeline-step-title" style="margin-bottom:8px;">Confidence Decomposition</div>' + renderConfidence(d.confidence) + '</div>';
      
      if (d.confidence && d.confidence.confidence_reasoning && d.confidence.confidence_reasoning.length > 0) {
        l2Html += '<div class="pipeline-step done"><div class="pipeline-step-title">Reasoning</div><ul class="vp-list" style="margin-top:6px; font-size:12px; color:var(--text-secondary); padding-left:14px;">';
        d.confidence.confidence_reasoning.forEach(function(r) { l2Html += '<li>' + esc(r) + '</li>'; });
        l2Html += '</ul></div>';
      }
      l2Div.innerHTML = l2Html;

      if (d.final_verdict !== 'PASS') {
        l2Btn.classList.add('open');
        l2Div.classList.add('open');
      }

      // ---- L3: Full Pipeline Trace ----
      const pdBtn = document.createElement('button');
      pdBtn.className = 'pipeline-toggle l3-toggle';
      pdBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg> Advanced: Full Pipeline Trace';
      ch.appendChild(pdBtn);

      const pdDiv = document.createElement('div');
      pdDiv.className = 'pipeline-steps';
      ch.appendChild(pdDiv);

      pdBtn.onclick = function() {
        pdBtn.classList.toggle('open');
        pdDiv.classList.toggle('open');
      };

      function pab(cls, who, body) {
        const dd = document.createElement('div');
        dd.className = 'b ' + cls;
        dd.innerHTML = (who ? '<div class="w">' + who + '</div>' : '') + body;
        pdDiv.appendChild(dd);
      }
      function phr() {
        const hr = document.createElement('hr');
        hr.className = 'divider';
        pdDiv.appendChild(hr);
      }

      if (d.search_performed && d.search_sources && d.search_sources.length) {
        pab('sr', 'Web Search (' + d.search_sources.length + ' sources)', renderSearchSources(d.search_sources));
      } else if (d.search_attempted && d.search_note) {
        pab('sr', 'Web Search', '<div class="search-note">' + esc(d.search_note) + '</div>');
      }

      pab('g1', 'GPT-1 (Generator)', esc(d.gpt1_output));

      if (d.bypassed) {
        pab('byp', '', 'Activation phrase detected - verification bypassed');
      } else {
        let g2body = renderClaimTable(d.claim_table) + renderConfidence(d.confidence) + renderViolations(d.violations);
        pab('g2', 'GPT-2 (Verifier) &mdash; ' + d.gpt2_verdict, g2body);

        if (d.gpt2_verdict !== 'PASS') {
          if (d.arbiter_invoked) {
            let g3body = '';
            const decLower = (d.arbiter_decision || '').toLowerCase().replace(/_/g, '');
            let decCls = 'blk';
            if (decLower === 'allowwithedits') decCls = 'awe';
            if (decLower === 'allowasunknownonly') decCls = 'auo';
            g3body += '<div class="arb-decision ' + decCls + '">' + esc(d.arbiter_decision) + '</div>';
            if (d.arbiter_rationale && d.arbiter_rationale.length > 0) {
              g3body += '<div class="arb-rationale">';
              d.arbiter_rationale.forEach(r => { g3body += '<div class="arb-item"><span class="arb-dot"></span>' + esc(r) + '</div>'; });
              g3body += '</div>';
            }
            if (d.arbiter_edits && d.arbiter_edits.length > 0) {
              g3body += '<div class="edit-list">';
              d.arbiter_edits.forEach(e => {
                g3body += '<div class="edit-item"><div class="edit-action ' + editActionCls(e.action) + '">' + esc(e.action) + '</div><div class="edit-target">' + esc(e.target) + '</div>';
                if (e.replacement) g3body += '<div class="edit-repl">&rarr; ' + esc(e.replacement) + '</div>';
                g3body += '</div>';
              });
              g3body += '</div>';
            }
            if (d.arbiter_policy_notes && d.arbiter_policy_notes.length > 0) {
              g3body += '<div class="policy-notes">';
              d.arbiter_policy_notes.forEach(n => { g3body += '<div>' + esc(n) + '</div>'; });
              g3body += '</div>';
            }
            pab('g3', 'GPT-3 (Arbiter)', g3body);
          }

          if (d.rewrite_occurred) {
            phr();
            pab('rw', 'GPT-1 (Rewrite)', esc(d.rewrite_output));
            let rvBody = renderClaimTable(d.rewrite_claim_table) + renderConfidence(d.confidence) + renderViolations(d.rewrite_violations);
            pab('rv', 'GPT-2 (Re-verify) &mdash; ' + d.rewrite_verdict, rvBody);
          }
        }
      }

      ch.scrollTop = ch.scrollHeight;
      addConversation(prompt, ch.innerHTML);

    } catch(err) {
      timers.forEach(t => clearTimeout(t));
      clearStages();
      ld.remove();
      const msg = err.message || String(err);
      if (msg.includes('Unexpected token') || msg.includes('not valid JSON')) {
        ab('err', '', 'Server returned a non-JSON response (timeout).');
      } else if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
        ab('err', '', 'Network error: could not reach the server.');
      } else {
        ab('err', '', 'Error: ' + esc(msg));
      }
      console.error('Pipeline error:', err);
    } finally {
      if (btn) btn.disabled = false;
      if (inp) inp.focus();
      updateRateLimit();
    }
  } catch (outerErr) {
    console.error('Fatal go() error:', outerErr);
    ab('err', 'CRITICAL ERROR', 'UI encountered a structural error: ' + esc(outerErr.message));
  }
}

lc();
loadTav();
updateRateLimit();
renderHistory();
document.getElementById('ui')?.focus();

function kd(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    go();
  }
}

function send() {
  go();
}

// ---- Tab Switching & Ledger Graph ----
function switchTab(tabId) {
  // Update buttons
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  
  // Show/hide views
  const chatView = document.getElementById('ch');
  const ibarView = document.getElementById('ibar-container');
  const ledgerView = document.getElementById('ledger-view');
  
  if (tabId === 'chat') {
    chatView.style.display = '';
    ibarView.style.display = '';
    ledgerView.style.display = 'none';
  } else if (tabId === 'ledger') {
    chatView.style.display = 'none';
    ibarView.style.display = 'none';
    ledgerView.style.display = 'flex';
    if (!window.ledgerGraphLoaded) {
      loadLedgerGraph();
      window.ledgerGraphLoaded = true;
    }
  }
}

let network = null;
async function loadLedgerGraph() {
  const container = document.getElementById('kg-network');
  container.innerHTML = '<div style="padding: 24px; color: var(--text-tertiary);">Loading graph data...</div>';
  
  try {
    const r = await fetch('/api/ledger');
    if (!r.ok) throw new Error('Failed to load ledger data');
    const data = await r.json();
    
    if (!data.nodes || data.nodes.length === 0) {
      container.innerHTML = '<div style="padding: 24px; color: var(--text-tertiary);">The Knowledge Graph is currently empty. Successful pipeline strict verifications will populate it.</div>';
      return;
    }
    
    container.innerHTML = ''; // clear loading text
    
    // Map to vis-network format
    const visNodes = data.nodes.map(n => {
      let colorSettings = {
        background: 'rgba(255,255,255,0.05)',
        border: 'rgba(255,255,255,0.2)'
      };
      
      if (n.type === 'entity') {
        colorSettings = {
          background: 'rgba(139, 92, 246, 0.15)', // violet
          border: 'var(--accent-violet)',
          highlight: { border: '#fff', background: 'rgba(139, 92, 246, 0.4)' }
        };
      } else if (n.type === 'source') {
        colorSettings = {
          background: 'rgba(59, 130, 246, 0.15)', // blue
          border: 'var(--accent-blue)',
          highlight: { border: '#fff', background: 'rgba(59, 130, 246, 0.4)' }
        };
      }
      
      return {
        id: n.id,
        label: n.name,
        title: n.type + ': ' + n.name,
        color: colorSettings,
        font: { color: 'var(--text-primary)', face: 'var(--font-mono)', size: 12 },
        shape: n.type === 'source' ? 'box' : 'ellipse',
        borderWidth: 1
      };
    });
    
    const visEdges = data.edges.map(e => {
      let edgeColor = 'rgba(255,255,255,0.2)';
      let dashes = false;
      
      if (e.confidence === 'Low') {
        edgeColor = 'rgba(245, 158, 11, 0.3)'; // amber
        dashes = true;
      } else if (e.confidence === 'High') {
        edgeColor = 'rgba(16, 185, 129, 0.4)'; // emerald
      }
      
      return {
        id: e.id,
        from: e.source,
        to: e.target,
        label: e.label,
        font: { color: 'var(--text-secondary)', face: 'var(--font-sans)', size: 10, align: 'top' },
        color: { color: edgeColor, highlight: '#fff' },
        arrows: 'to',
        dashes: dashes,
        smooth: { type: 'continuous' }
      };
    });
    
    const graphData = {
      nodes: new vis.DataSet(visNodes),
      edges: new vis.DataSet(visEdges)
    };
    
    const options = {
      physics: {
        forceAtlas2Based: {
          gravitationalConstant: -50,
          centralGravity: 0.01,
          springLength: 100,
          springConstant: 0.08
        },
        maxVelocity: 50,
        solver: 'forceAtlas2Based',
        timestep: 0.35,
        stabilization: { iterations: 150 }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true
      }
    };
    
    if (network !== null) {
      network.destroy();
      network = null;
    }
    
    network = new vis.Network(container, graphData, options);

  } catch(err) {
    container.innerHTML = '<div style="padding: 24px; color: var(--accent-rose);">Error loading graph: ' + window.esc(err.message) + '</div>';
  }
}