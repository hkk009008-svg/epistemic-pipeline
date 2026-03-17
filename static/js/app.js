// V7 Audit Workspace Logic
const logEl = document.getElementById('hookLog');
const btnRun = document.getElementById('runBtn');
const promptArea = document.getElementById('promptArea');
const repContainer = document.getElementById('reportContainer');
const repHeader = document.getElementById('reportHeader');
const pulse = document.getElementById('pulse');

function openConfig() {
  document.getElementById('configSidebar').classList.add('open');
  document.getElementById('overlay').classList.add('open');
  loadConfig();
}
function closeConfig() {
  document.getElementById('configSidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('open');
}

function esc(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function appendLog(cls, msg) {
  const div = document.createElement('div');
  div.className = 'log-entry ' + cls;
  div.innerHTML = esc(msg);
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

function formatSources(sources) {
  if (!sources || sources.length === 0) return '';
  let h = '<table class="sources-table"><thead><tr><th>Trust</th><th>Source</th></tr></thead><tbody>';
  sources.forEach(s => {
    let trust = s.score >= 0.8 ? 'High' : (s.score >= 0.5 ? 'Med' : 'Low');
    h += '<tr><td><span style="opacity:0.6">' + trust + '</span></td>';
    h += '<td><a href="' + esc(s.url) + '" target="_blank">' + esc(s.title || 'Link') + '</a>';
    if(s.snippet) h += '<div style="font-size:10px;color:var(--text-muted);margin-top:4px;">' + esc(s.snippet.substring(0, 150)) + '...</div>';
    h += '</td></tr>';
  });
  h += '</tbody></table>';
  return h;
}

function renderReport(data) {
  repHeader.style.display = 'block';
  
  const isPass = data.final_verdict === 'PASS';
  const vClass = isPass ? 'verdict-PASS' : 'verdict-FAIL';
  
  let h = '<div class="eval-card glass-panel">';
  h += '<div class="eval-header">';
  h += '<div class="verdict-badge ' + vClass + '">' + esc(data.final_verdict) + '</div>';
  h += '<div class="verdict-label">' + esc(data.verdict_label || '') + '</div>';
  h += '</div>';
  
  h += '<div class="eval-reasoning">' + esc(data.reasoning || data.final_result) + '</div>';
  
  // Render Hooks Data
  if (data.hook_results && data.hook_results.length > 0) {
    h += '<h3 style="margin-top:24px; font-size:12px; color:var(--accent-cyan); letter-spacing:0.05em; border-bottom:1px solid var(--border-light); padding-bottom:8px; margin-bottom:12px;">EVIDENCE LOG</h3>';
    
    data.hook_results.forEach(hr => {
      if (hr.tool === 'search') {
        h += '<div style="font-family:var(--font-mono); font-size:11px; margin-bottom:8px; color:var(--text-muted);">[' + esc(hr.tool.toUpperCase()) + '] ' + esc(hr.note) + '</div>';
        h += formatSources(hr.data);
      }
    });
  }
  
  h += '</div>';
  repContainer.innerHTML = h;
}

async function runAudit() {
  const prompt = promptArea.value.trim();
  if (!prompt) return;
  
  btnRun.disabled = true;
  btnRun.textContent = 'AUDITING...';
  pulse.style.display = 'inline-block';
  repHeader.style.display = 'none';
  repContainer.innerHTML = '';
  
  appendLog('run', '> INIT AUDIT_SESSION: ' + prompt.substring(0, 50) + '...');
  
  try {
    const res = await fetch('/api/pipeline', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt: prompt })
    });
    
    if (!res.ok) {
        const err = await res.json().catch(()=>({detail:'HTTP ' + res.status}));
        appendLog('err', 'System Error: ' + esc(err.detail));
    } else {
        const data = await res.json();
        appendLog('done', '> PIPELINE COMPLETE. Verdict: ' + data.final_verdict);
        data.hook_results.forEach(hr => {
            appendLog('sys', 'HOOK [' + hr.tool + '] -> ' + hr.note);
        });
        renderReport(data);
    }
  } catch (e) {
    appendLog('err', 'Network collapse. Backend unreachable.');
  } finally {
    btnRun.disabled = false;
    btnRun.textContent = 'INITIATE AUDIT';
    pulse.style.display = 'none';
  }
}

// Config sync
async function loadConfig() {
  try {
    const r = await fetch('/api/openai/config');
    const d = await r.json();
    const st = document.getElementById('stOai');
    if(d.key_set) { st.textContent = 'Active: ' + d.key_preview; st.className='status-text ok'; document.getElementById('cfgOai').value=''; }
    else { st.textContent = 'Missing key'; st.className='status-text'; }
  } catch(e){}
  
  try {
    const r2 = await fetch('/api/tavily/config');
    const d2 = await r2.json();
    const st2 = document.getElementById('stTav');
    if(d2.key_set) { 
        st2.textContent = 'Active: ' + (d2.enabled ? 'Enabled' : 'Disabled');
        st2.className = d2.enabled ? 'status-text ok' : 'status-text';
        document.getElementById('cfgTavEn').checked = d2.enabled;
    } else { st2.textContent = 'Missing key'; st2.className='status-text'; }
  } catch(e){}
}

async function saveOai() {
  const k = document.getElementById('cfgOai').value.trim();
  if(!k) return;
  await fetch('/api/openai/config', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({api_key:k, model:'gpt-4o-mini'})
  });
  loadConfig();
}

async function saveTav() {
  const k = document.getElementById('cfgTav').value.trim();
  if(!k) return;
  const en = document.getElementById('cfgTavEn').checked;
  await fetch('/api/tavily/config', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({api_key:k, enabled:en})
  });
  loadConfig();
}

async function toggleTav(en) {
  await fetch('/api/tavily/toggle?enabled='+en, {method:'POST'});
  loadConfig();
}