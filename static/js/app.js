// V7 Audit Workspace Logic
const repContainer = document.getElementById('reportContainer');

let currentMode = 'verify';

const promptsVerify = [
  "Did the FDA actually ban artificial food dyes in 2024?",
  "Fact check: Are there more trees on Earth than stars in the Milky Way?",
  "Is it true that the Eiffel Tower grows taller in the summer?"
];
const promptsDecision = [
  "Should I buy a MacBook Air or Pro for college programming?",
  "What is the optimal database scaling strategy for 1 million write operations?",
  "Compare renting an apartment versus buying a house in the 2024 market."
];

function renderChips(mode) {
  const container = document.getElementById('suggestionChips');
  if (!container) return;
  container.innerHTML = '';
  const list = mode === 'verify' ? promptsVerify : promptsDecision;
  list.forEach((txt, idx) => {
    const btn = document.createElement('button');
    btn.className = 'chip-btn';
    btn.textContent = txt;
    btn.style.animation = `fadeUp 0.3s ease forwards ${idx * 0.1}s`;
    btn.style.opacity = '0';
    btn.onclick = () => {
      document.getElementById('promptArea').value = txt;
      runAudit();
    };
    container.appendChild(btn);
  });
}

function setMode(mode) {
  currentMode = mode;
  document.getElementById('modeVerify').classList.remove('active');
  document.getElementById('modeDecision').classList.remove('active');
  const btnRun = document.getElementById('runBtn');
  const promptArea = document.getElementById('promptArea');
  
  if(mode === 'verify') {
    document.getElementById('modeVerify').classList.add('active');
    promptArea.placeholder = "Paste a claim or problem to epistemic audit...";
    btnRun.querySelector('.btn-text').textContent = "Run Audit";
  } else {
    document.getElementById('modeDecision').classList.add('active');
    promptArea.placeholder = "Describe your goal for deterministic curation...";
    btnRun.querySelector('.btn-text').textContent = "Run Curation";
  }
  renderChips(mode);
}

// Ensure chips load on boot
document.addEventListener('DOMContentLoaded', () => { renderChips('verify'); });

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

function formatSources(sources) {
  if (!sources || sources.length === 0) return '';
  let h = '<div class="sources-grid">';
  sources.forEach((s, idx) => {
    let domain = 'Link';
    try { domain = new URL(s.url).hostname.replace('www.', ''); } catch(e){}
    h += `<a href="${esc(s.url)}" target="_blank" class="source-card" style="animation: fadeUp 0.4s ease forwards ${idx * 0.05}s; opacity:0;">
            <div class="source-meta">
              <span class="source-index">[${idx+1}]</span> <span>${esc(domain)}</span>
            </div>
            <div class="source-title">${esc(s.title || 'Source Reference')}</div>
          </a>`;
  });
  h += '</div>';
  return h;
}

function renderReport(data) {
  const isPass = data.final_verdict === 'PASS';
  const vClass = isPass ? 'verdict-PASS' : 'verdict-FAIL';
  
  let h = '<div class="eval-card glass-panel">';
  
  // Hero Header Context
  let confidenceStr = '';
  if(data.confidence_score) confidenceStr = `<div class="scope-pill" style="border-color: var(--accent-purple); color: var(--accent-purple); box-shadow: 0 0 10px var(--accent-purple-glow);">EPISTEMIC VALIDITY SCORE: <strong>${data.confidence_score}</strong></div>`;
  
  h += '<div class="eval-header" style="flex-wrap:wrap;">';
  h += '<div class="verdict-wrapper"><div class="verdict-badge ' + vClass + '">' + esc(data.final_verdict) + '</div>';
  h += '<div class="verdict-label">' + esc(data.verdict_label || '') + '</div></div>';
  h += confidenceStr;
  h += '</div>';
  
  let rawText = data.final_result ? data.final_result : (Array.isArray(data.reasoning) ? data.reasoning.join('\n') : data.reasoning);
  
  // Custom Dashboard Parser
  let pDirective = '', pMatrix = '', pEvidence = '', pAlt = '', pTelemetry = '';
  
  const sections = rawText.split(/(?=1\. What This Means|2\. How This Was Calculated|3\. Why This Is Correct|🔄 Alternative Considerations|⚙️ System Telemetry)/i);
  
  sections.forEach(sec => {
    if(sec.match(/1\. What This Means/i)) pDirective = sec;
    else if(sec.match(/2\. How This Was Calculated/i)) pMatrix = sec;
    else if(sec.match(/3\. Why This Is Correct/i)) pEvidence = sec;
    else if(sec.match(/Alternative Considerations/i)) pAlt = sec;
    else if(sec.match(/⚙️ System Telemetry/i)) pTelemetry = sec;
    else if(!pDirective) pDirective = sec; // fallback
  });

  // Render Directive (Hero)
  if(pDirective) {
    let clean = pDirective.replace(/1\. What This Means.*?\(The Directive\)/i, '').trim();
    // remove any ### or ** 
    clean = clean.replace(/###/g, '').replace(/\*/g, '').replace(/>/g, '').trim();
    
    // highlight "Primary Path: XXX"
    clean = clean.replace(/Primary Path:\s*(.*)/i, '<div class="hero-primary-path">Primary Path: <span>$1</span></div>');
    h += `<div class="dashboard-section directive-section" style="animation: fadeUp 0.5s ease forwards; opacity:0;">
            <div class="dash-title">THE DIRECTIVE</div>
            <div class="dash-content hero-text">${clean.replace(/\n\n/g, '<br><br>')}</div>
          </div>`;
  }
  
  // Render Evidence FIRST (after Directive)
  if(pEvidence) {
    let clean = pEvidence.replace(/3\. Why This Is Correct.*?\(The Evidence\)/i, '').trim();
    clean = clean.replace(/###/g, ''); // strip potential headers
    // Split into Verified Facts, Logical Deductions, Gaps
    clean = clean.replace(/(?:\*\*)?(Verified Facts:|Logical Deductions:|Identified Gaps \(Unknowns\):)(?:\*\*)?/gi, '</div><div class="evidence-col"><h4>$1</h4>');
    
    clean = clean.replace(/(?:^|\n)(?:▶|-)\s*(.*)/g, '\n<div class="evidence-item"><span class="evidence-bullet"></span><span>$1</span></div>');
    // Remove all remaining asterisks/strong to keep it perfectly clean
    clean = clean.replace(/\*/g, '');
    
    h += `<div class="dashboard-section evidence-section" style="animation: fadeUp 0.5s ease forwards 0.1s; opacity:0;">
            <div class="dash-title">EVIDENCE & DEDUCTIONS</div>
            <div class="evidence-grid">
              ${clean.startsWith('</div>') ? clean.substring(6) : clean}
            </div>
          </div>`;
  }

  // Render Alternatives SECOND
  if(pAlt) {
    let clean = pAlt.replace(/🔄 Alternative Considerations/i, '').trim();
    clean = clean.replace(/###/g, '').replace(/\*\*/g, ' '); // remove headers and bold
    clean = clean.replace(/(?:^|\n)(?:▶|-)\s*(.*)/g, '\n<div class="alt-item"><span>$1</span></div>');
    clean = clean.replace(/\*(.*?)\*/g, '<span class="alt-score">$1</span>');
    clean = clean.replace(/\*/g, ''); // catch stray asterisks
    
    h += `<div class="dashboard-section align-section" style="animation: fadeUp 0.5s ease forwards 0.15s; opacity:0;">
            <div class="dash-title">ALTERNATIVE PATHS</div>
            <div class="alt-grid">${clean}</div>
          </div>`;
  }

  // ADVANCED DIAGNOSTICS TOGGLE (Wraps Matrix & Telemetry Logs)
    // Re-route Matrix weights to the pre-existing accordion
    if(pMatrix) {
      let clean = pMatrix.replace(/2\. How This Was Calculated.*?\(Deterministic Matrix\)/i, '').trim();
      clean = clean.replace(/###/g, ''); 
      let introSplit = clean.split('\n-');
      let intro = introSplit[0].replace(/\*/g, '').trim();
      clean = clean.replace(introSplit[0], '');
      
      const metricRegex = /(?:^|\n)(?:▶|-)\s*[\*\_]*(.*?)[\*\_]*\s*\(\*?(.*?weight)\*?\):\s*([\d.]+)/gi;
      let metricsHtml = '<div class="metric-grid">';
      let m, hasMetrics = false;
      let cleanMetrics = clean.replace(/\*\*/g, '');
      
      while((m = metricRegex.exec(cleanMetrics)) !== null) {
        hasMetrics = true;
        let label = esc(m[1].trim());
        let weight = esc(m[2].trim());
        let score = parseFloat(m[3]);
        let pct = Math.round(score * 100);
        let barColor = pct >= 80 ? 'var(--accent-green)' : (pct >= 50 ? 'var(--accent-cyan)' : 'var(--accent-red)');
        
        metricsHtml += `
          <div class="metric-card">
            <div class="metric-header">
              <span class="metric-label">${label}</span>
              <span class="metric-score" style="color:${barColor}">${pct}%</span>
            </div>
            <div class="metric-weight">${weight}</div>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill" style="width: ${pct}%; background: ${barColor};"></div>
            </div>
          </div>
        `;
      }
      metricsHtml += '</div>';

      const target = document.getElementById('matrixWeightsTarget');
      if(target) {
        if(hasMetrics) {
          target.innerHTML = `<div class="dashboard-section matrix-section" style="margin-top:20px;">
                  <div class="dash-title">DETERMINISTIC WEIGHTS (EPISTEMIC SCORE)</div>
                  <div class="dash-intro">${intro}</div>
                  ${metricsHtml}
                </div>`;
        } else {
          target.innerHTML = `<div class="dashboard-section matrix-section" style="margin-top:20px;">
                  <div class="dash-title">DETERMINISTIC WEIGHTS (EPISTEMIC SCORE)</div>
                  <div class="dash-content">${esc(clean.replace(/\*/g, '')).replace(/\n/g, '<br>')}</div>
                </div>`;
        }
      }
    }
    
    // Auto-close accordion when done to optimize reading flow (user request: "default-closed")
    const diagToggle = document.querySelector('.diagnostics-toggle');
    if(diagToggle) diagToggle.removeAttribute('open');
  
  // Render Hooks Data (Sources)
  if (data.hook_results && data.hook_results.length > 0) {
    h += `<div class="dashboard-section sources-section" style="animation: fadeUp 0.5s ease forwards 0.25s; opacity:0;">
            <div class="dash-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
              SOURCE REFERENCES
            </div>`;
    
    data.hook_results.forEach(hr => {
      if (hr.tool === 'search') {
        h += formatSources(hr.data);
      }
    });
    h += `</div>`;
  }
  
  h += '</div>';
  const target = document.getElementById('finalOutputTarget');
  if (target) {
    target.innerHTML = h;
  } else {
    repContainer.innerHTML = h;
  }
}

async function runAudit() {
  const promptArea = document.getElementById('promptArea');
  const prompt = promptArea.value.trim();
  const btnRun = document.getElementById('runBtn');
  if (!prompt) return;
  
  btnRun.disabled = true;
  btnRun.querySelector('.btn-text').textContent = 'INITIALIZING MATRIX...';
  
  // SSE Streaming Matrix UI
  repContainer.innerHTML = `
    <div id="finalOutputTarget">
      <div class="glass-panel loading-skeleton" id="loadingSkeleton">
        <div class="cyber-spinner"></div>
        <div class="loading-text" id="telStatus">Querying ${currentMode === 'verify' ? 'TruthLens' : 'OmniResolve Matrix'}...</div>
      </div>
    </div>
    
    <details class="diagnostics-toggle" style="margin-top: 20px;" open>
      <summary class="dash-title" style="cursor: pointer; display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
        VIEW THE DELICATE PROCESS (MATRIX LOGS)
      </summary>
      
      <div class="diagnostics-content telemetry-console" id="telStream" style="margin-top: 20px; min-height: 100px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; color: var(--accent-cyan);">Initializing Cognitive Workspace...</div>
      
      <!-- We will inject Deterministic Weights here at the end -->
      <div id="matrixWeightsTarget"></div>
      
    </details>
  `;
  repContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  
  try {
    const res = await fetch('/api/pipeline', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt: prompt, mode: currentMode, stream: true })
    });
    
    if (!res.ok) {
        const err = await res.json().catch(()=>({detail:'HTTP ' + res.status}));
        document.getElementById('finalOutputTarget').innerHTML = '<div class="eval-card glass-panel" style="color:var(--accent-red); font-weight:600; font-size:16px;">Matrix Error: ' + esc(err.detail) + '</div>';
        return;
    }
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let sseBuffer = '';
    let tokenBuffer = '';
    const telStream = document.getElementById('telStream');
    const telStatus = document.getElementById('telStatus');

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        sseBuffer += decoder.decode(value, { stream: true });
        
        let d_idx;
        while ((d_idx = sseBuffer.indexOf('\n\n')) >= 0) {
            let msg = sseBuffer.substring(0, d_idx);
            sseBuffer = sseBuffer.substring(d_idx + 2);
            
            let event = 'message';
            let data = '';
            msg.split('\n').forEach(line => {
                if (line.startsWith('event: ')) event = line.substring(7);
                else if (line.startsWith('data: ')) data += line.substring(6) + '\n';
            });
            
            if(data.endsWith('\n')) data = data.substring(0, data.length - 1);
            
            if (event === 'token') {
                tokenBuffer += data;
                // Live CSS highlighting logic
                let formatted = esc(tokenBuffer);
                formatted = formatted.replace(/\\[DOC\\]/g, '<span style="color:var(--accent-green);font-weight:bold;">[DOC]</span>');
                formatted = formatted.replace(/\\[INFERENCE\\]/g, '<span style="color:var(--accent-purple);font-weight:bold;">[INFERENCE]</span>');
                formatted = formatted.replace(/\\[UNKNOWN\\]/g, '<span style="color:var(--accent-red);font-weight:bold;">[UNKNOWN]</span>');
                formatted = formatted.replace(/Halt-and-cite/g, '<span style="color:var(--accent-cyan);background:rgba(0,255,255,0.1);padding:0 4px;border-radius:4px;">Halt-and-cite</span>');
                telStream.innerHTML = formatted;
                telStream.scrollTop = telStream.scrollHeight;
            } else if (event === 'status') {
                if(telStatus) telStatus.textContent = data;
            } else if (event === 'done') {
                const finalData = JSON.parse(data);
                renderReport(finalData); // This will populate finalOutputTarget
            } else if (event === 'error') {
                document.getElementById('finalOutputTarget').innerHTML = '<div class="eval-card glass-panel" style="color:var(--accent-red); font-weight:600;">Matrix Error: ' + esc(data) + '</div>';
            }
        }
    }
  } catch (e) {
    document.getElementById('finalOutputTarget').innerHTML = '<div class="eval-card glass-panel" style="color:var(--accent-red); font-weight:600; font-size:16px;">Network collapse. Backend unreachable.</div>';
  } finally {
    btnRun.disabled = false;
    btnRun.querySelector('.btn-text').textContent = currentMode === 'verify' ? "Run Audit" : "Run Curation";
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

window.addEventListener('DOMContentLoaded', loadConfig);