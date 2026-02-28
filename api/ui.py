"""Embedded chat UI served at /."""

UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT-1 &rarr; GPT-2 &rarr; GPT-3 Pipeline</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; }

  .top-bar { background: #111; border-bottom: 1px solid #222; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  .top-bar h1 { font-size: 15px; font-weight: 700; }
  .top-bar h1 .g1 { color: #4fc3f7; }
  .top-bar h1 .arr { color: #444; margin: 0 4px; }
  .top-bar h1 .g2 { color: #ff8a65; }
  .top-bar h1 .g3 { color: #ce93d8; }
  .top-bar .cfg-btn { background: #222; color: #aaa; border: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; font-weight: 600; }
  .top-bar .cfg-btn:hover { background: #333; }
  .kd { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .kd.on { background: #4caf50; }
  .kd.off { background: #ef5350; }

  .cfg-drawer { background: #0d0d0d; border-bottom: 1px solid #1a1a1a; overflow: hidden; max-height: 0; transition: max-height 0.3s ease; flex-shrink: 0; }
  .cfg-drawer.open { max-height: 700px; }
  .cfg-in { padding: 16px 24px; max-width: 780px; }
  .cfg-in label { display: block; font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; margin: 10px 0 4px; }
  .cfg-in label:first-child { margin-top: 0; }
  .cfg-in input, .cfg-in select, .cfg-in textarea { width: 100%; padding: 8px 10px; background: #111; border: 1px solid #222; border-radius: 6px; color: #ddd; font-size: 13px; font-family: inherit; outline: none; }
  .cfg-in textarea { resize: vertical; min-height: 44px; }
  .cfg-in input:focus, .cfg-in textarea:focus { border-color: #4fc3f7; }
  .cfg-row { display: flex; gap: 10px; align-items: flex-end; }
  .cfg-row input { flex: 1; }
  .btn-s { padding: 8px 14px; background: #222; color: #ccc; border: none; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; }
  .btn-s:hover { background: #333; }
  .cfg-st { font-size: 11px; color: #555; margin-top: 4px; }
  .cfg-st.ok { color: #4caf50; }

  .chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 14px; max-width: 860px; width: 100%; margin: 0 auto; }

  /* Bubbles */
  .b { max-width: 94%; padding: 14px 18px; border-radius: 14px; line-height: 1.6; font-size: 14px; white-space: pre-wrap; animation: fu 0.3s ease; }
  .b .w { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .b.usr { align-self: flex-end; background: #1a1a2e; border: 1px solid #2a2a4e; color: #ccc; }
  .b.usr .w { color: #8888cc; }
  .b.g1 { align-self: flex-start; background: #0a1a2a; border: 1px solid #1a3a5c; color: #ccc; }
  .b.g1 .w { color: #4fc3f7; }
  .b.g2 { align-self: flex-start; background: #1a120a; border: 1px solid #3a2515; color: #bbb; font-size: 13px; white-space: normal; }
  .b.g2 .w { color: #ff8a65; }
  .b.g3 { align-self: flex-start; background: #1a0a1a; border: 1px solid #3a1540; color: #dcc; font-size: 13px; white-space: normal; }
  .b.g3 .w { color: #ce93d8; }
  .b.rw { align-self: flex-start; background: #0a1a1a; border: 1px solid #154040; color: #8dd; font-size: 13px; white-space: pre-wrap; }
  .b.rw .w { color: #4db6ac; }
  .b.rv { align-self: flex-start; background: #1a1a0a; border: 1px solid #3a3515; color: #bba; font-size: 13px; white-space: normal; }
  .b.rv .w { color: #dce775; }

  .b.vp { align-self: center; background: #0d2a0d; border: 2px solid #1b5e1b; color: #66bb6a; text-align: center; font-weight: 700; font-size: 15px; padding: 16px 32px; white-space: normal; }
  .b.vf { align-self: center; background: #2a0d0d; border: 2px solid #5e1b1b; color: #ef5350; text-align: center; font-weight: 700; font-size: 15px; padding: 16px 32px; white-space: normal; }
  .b.fo { align-self: center; background: #111; border: 1px solid #2a2a2a; color: #e0e0e0; width: 100%; max-width: 100%; }
  .b.fo .w { color: #66bb6a; }
  .b.fo.blk .w { color: #ef5350; }
  .b.fo.blk { border-color: #3a1a1a; color: #ef5350; text-align: center; font-weight: 600; }
  .b.byp { align-self: center; background: #1a1a10; border: 1px solid #3a3a15; color: #c0c070; text-align: center; font-size: 12px; padding: 10px 20px; white-space: normal; }

  /* Divider */
  .divider { align-self: center; width: 80%; border: none; border-top: 1px dashed #222; margin: 4px 0; }

  /* Claim table */
  .ct { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }
  .ct th { text-align: left; padding: 4px 8px; color: #888; border-bottom: 1px solid #2a2a2a; font-size: 10px; text-transform: uppercase; }
  .ct td { padding: 6px 8px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }
  .ct .cat { font-weight: 600; }
  .cat-sup { color: #66bb6a; }
  .cat-inf { color: #ffb74d; }
  .cat-hyp { color: #ce93d8; }
  .cat-uns { color: #ef5350; }
  .cat-usr { color: #4fc3f7; }

  /* Violations */
  .viol { margin-top: 8px; }
  .viol-item { display: flex; gap: 6px; align-items: center; font-size: 12px; color: #ef5350; margin-bottom: 4px; }
  .viol-dot { width: 6px; height: 6px; border-radius: 50%; background: #ef5350; flex-shrink: 0; }
  .no-viol { color: #66bb6a; font-size: 12px; margin-top: 8px; }

  /* Arbiter details */
  .arb-rationale { margin-top: 8px; }
  .arb-item { display: flex; gap: 6px; align-items: flex-start; font-size: 12px; color: #ce93d8; margin-bottom: 4px; }
  .arb-dot { width: 6px; height: 6px; border-radius: 50%; background: #ce93d8; flex-shrink: 0; margin-top: 5px; }
  .arb-decision { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
  .arb-decision.blk { color: #ef5350; }
  .arb-decision.awe { color: #ffb74d; }
  .arb-decision.auo { color: #4db6ac; }
  .edit-list { margin-top: 8px; font-size: 12px; }
  .edit-item { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px; }
  .edit-action { font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 2px; }
  .edit-action.del { color: #ef5350; }
  .edit-action.rew { color: #ffb74d; }
  .edit-action.mtu { color: #4db6ac; }
  .edit-target { color: #888; font-style: italic; }
  .edit-repl { color: #aaa; margin-top: 2px; }
  .policy-notes { margin-top: 8px; padding: 8px 10px; background: #111; border: 1px solid #222; border-radius: 6px; font-size: 11px; color: #777; }

  .ld { align-self: center; padding: 20px; color: #555; font-size: 13px; text-align: center; }
  .sp { display: inline-block; width: 22px; height: 22px; border: 2px solid #222; border-top-color: #4fc3f7; border-radius: 50%; animation: spin 0.7s linear infinite; margin-bottom: 8px; }
  .sp.s2 { border-top-color: #ff8a65; }
  .sp.s3 { border-top-color: #ce93d8; }
  .err { background: #1a0a0a; border: 1px solid #3a1a1a; color: #ef5350; padding: 12px 16px; border-radius: 10px; font-size: 13px; align-self: center; white-space: normal; }

  .ibar { background: #111; border-top: 1px solid #222; padding: 14px 24px; flex-shrink: 0; }
  .ibar form { display: flex; gap: 10px; max-width: 860px; margin: 0 auto; }
  .ibar input { flex: 1; padding: 12px 14px; background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 10px; color: #e0e0e0; font-size: 14px; outline: none; }
  .ibar input:focus { border-color: #4fc3f7; }
  .ibar button { padding: 12px 24px; background: linear-gradient(135deg, #4fc3f7, #0277bd); color: #fff; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; }
  .ibar button:hover { opacity: 0.9; }
  .ibar button:disabled { opacity: 0.3; cursor: not-allowed; }

  /* Stress Test Panel */
  .stress-panel { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: #0a0a0aee; z-index: 100; overflow-y: auto; }
  .stress-panel.open { display: flex; flex-direction: column; align-items: center; padding: 40px 24px; }
  .stress-hdr { display: flex; align-items: center; justify-content: space-between; width: 100%; max-width: 900px; margin-bottom: 16px; }
  .stress-hdr h2 { font-size: 18px; font-weight: 700; color: #e0e0e0; }
  .stress-close { background: #222; color: #aaa; border: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; }
  .stress-controls { display: flex; gap: 10px; align-items: flex-end; width: 100%; max-width: 900px; margin-bottom: 16px; }
  .stress-controls select, .stress-controls input { padding: 8px 10px; background: #111; border: 1px solid #222; border-radius: 6px; color: #ddd; font-size: 13px; }
  .stress-run { padding: 8px 20px; background: linear-gradient(135deg, #ce93d8, #7b1fa2); color: #fff; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; }
  .stress-run:disabled { opacity: 0.3; cursor: not-allowed; }
  .stress-log { width: 100%; max-width: 900px; background: #0d0d0d; border: 1px solid #1a1a1a; border-radius: 10px; padding: 16px; font-family: 'SF Mono', monospace; font-size: 12px; color: #888; min-height: 200px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; line-height: 1.5; }
  .stress-log .pass { color: #66bb6a; }
  .stress-log .fail { color: #ef5350; }
  .stress-log .arb { color: #ce93d8; }
  .stress-log .rew { color: #4db6ac; }
  .stress-score { width: 100%; max-width: 900px; margin-top: 16px; }
  .pss-big { font-size: 48px; font-weight: 800; text-align: center; margin: 16px 0; }
  .pss-big.s90 { color: #66bb6a; }
  .pss-big.s75 { color: #ffb74d; }
  .pss-big.s60 { color: #ff8a65; }
  .pss-big.s0 { color: #ef5350; }
  .pss-band { text-align: center; font-size: 14px; font-weight: 600; margin-bottom: 16px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 16px; }
  .metric-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 12px; text-align: center; }
  .metric-card .mv { font-size: 22px; font-weight: 700; color: #e0e0e0; }
  .metric-card .ml { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
  .metric-card .mp { font-size: 11px; color: #ef5350; margin-top: 2px; }
  .cat-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .cat-table th { text-align: left; padding: 6px 8px; color: #666; border-bottom: 1px solid #222; font-size: 10px; text-transform: uppercase; }
  .cat-table td { padding: 6px 8px; border-bottom: 1px solid #1a1a1a; }
  .cat-table .pr { font-weight: 600; }

  @keyframes fu { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="top-bar">
  <h1><span class="g1">GPT-1</span><span class="arr">&rarr;</span><span class="g2">GPT-2</span><span class="arr">&rarr;</span><span class="g3">GPT-3</span> Pipeline</h1>
  <div style="display:flex;gap:8px;">
    <button class="cfg-btn" style="background:#2a1a2e;color:#ce93d8;" onclick="openStress()">Stress Test</button>
    <button class="cfg-btn" onclick="tog()"><span class="kd off" id="kd"></span>Settings</button>
  </div>
</div>

<div class="cfg-drawer" id="cd">
  <div class="cfg-in">
    <label>OpenAI API Key</label>
    <div class="cfg-row">
      <input type="password" id="ak" placeholder="sk-...">
      <select id="md" style="width:150px">
        <option value="gpt-4o-mini">gpt-4o-mini</option>
        <option value="gpt-4o">gpt-4o</option>
        <option value="gpt-4-turbo">gpt-4-turbo</option>
        <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
      </select>
      <button class="btn-s" onclick="sav()">Save</button>
    </div>
    <div class="cfg-st" id="ks">No key set</div>

    <label>GPT-1 System Prompt (Generator)</label>
    <textarea id="g1s" rows="4">You are GPT-1, a structured reasoning and synthesis engine.

Hard constraints:
- No fabricated sources, statutes, studies, metrics, or percentages.
- Do not use "studies/research/data suggest" unless you provide a specific citation AND a concrete number/quote.
- Do not provide advice/options unless the user explicitly asks what to do.
- If asked for percentages and none are available in provided/cited evidence, output Unknown(Actionable).
- When mentioning professionals (attorneys, brokers, consultants), use ONLY role-definition + uncertainty language.
  NEVER use benefit-language ("could help", "could assist", "may improve", "could potentially", "may provide guidance").
  CORRECT: "An attorney's function is to advise on requirements and prepare/submit filings; whether that changes outcomes is unknown."
  WRONG: "An attorney could potentially assist in navigating the process."

Default format:
1) Problem Framing
2) Assumptions (explicit)
3) Analysis (Facts; then Inferences labeled)
4) Unknowns (Actionable / Structural)
5) Confidence (High/Medium/Low + 1 sentence)

Only include "Options" if user asked for actions/choices.</textarea>

    <label>GPT-2 System Prompt Override (leave blank for strict verifier)</label>
    <textarea id="g2s" rows="2" placeholder="Leave blank for default claim validator..."></textarea>

    <label>GPT-3 System Prompt Override (leave blank for default arbiter)</label>
    <textarea id="g3s" rows="2" placeholder="Leave blank for default arbiter/adjudicator..."></textarea>
  </div>
</div>

<div class="chat" id="ch"></div>

<!-- Stress Test Panel -->
<div class="stress-panel" id="sp">
  <div class="stress-hdr">
    <h2>Pipeline Stability Score (PSS)</h2>
    <button class="stress-close" onclick="closeStress()">Close</button>
  </div>
  <div class="stress-controls">
    <select id="sc">
      <option value="">All categories (100 tests)</option>
      <option value="legal_future_year">legal_future_year</option>
      <option value="statistical_percentage_trap">statistical_percentage_trap</option>
      <option value="medical_structural_indeterminacy">medical_structural_indeterminacy</option>
      <option value="cross_border_tax">cross_border_tax</option>
      <option value="citizenship_inheritance">citizenship_inheritance</option>
      <option value="sanctions_export_controls">sanctions_export_controls</option>
      <option value="crypto_compliance">crypto_compliance</option>
      <option value="neutral_definitional">neutral_definitional</option>
      <option value="advice_requested_explicit">advice_requested_explicit</option>
      <option value="regulatory_facts_basic">regulatory_facts_basic</option>
    </select>
    <input type="number" id="sn" min="1" max="10" value="" placeholder="Per-cat limit" style="width:100px;">
    <button class="stress-run" id="sr" onclick="runStress()">Run Stress Test</button>
  </div>
  <div class="stress-log" id="sl"></div>
  <div class="stress-score" id="ss"></div>
</div>

<div class="ibar">
  <form onsubmit="go(event)">
    <input type="text" id="ui" placeholder="Ask anything..." autocomplete="off">
    <button type="submit" id="sb">Send</button>
  </form>
</div>

<script>
function tog() { document.getElementById('cd').classList.toggle('open'); }
function openStress() { document.getElementById('sp').classList.add('open'); }
function closeStress() { document.getElementById('sp').classList.remove('open'); }

async function runStress() {
  const btn = document.getElementById('sr');
  const log = document.getElementById('sl');
  const scoreDiv = document.getElementById('ss');
  btn.disabled = true;
  log.innerHTML = '';
  scoreDiv.innerHTML = '';

  const cat = document.getElementById('sc').value || null;
  const cnt = parseInt(document.getElementById('sn').value) || null;
  const body = {};
  if (cat) body.category = cat;
  if (cnt) body.count = cnt;

  log.innerHTML = 'Starting stress test...\\n';

  try {
    const resp = await fetch('/api/stress', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json();
      log.innerHTML += '<span class="fail">ERROR: ' + esc(err.detail || 'Request failed') + '</span>';
      btn.disabled = false;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const d = JSON.parse(line);
        if (d.type === 'progress') {
          let cls = d.verdict === 'PASS' ? 'pass' : 'fail';
          let extra = '';
          if (d.arbiter) extra += ' <span class="arb">arbiter:' + esc(d.arbiter) + '</span>';
          if (d.rewrite) extra += ' <span class="rew">[rewrite]</span>';
          log.innerHTML += '[' + d.index + '/' + d.total + '] ' + esc(d.id) + ' <span class="' + cls + '">' + d.verdict + '</span>' + extra + ' (' + d.duration_s + 's)\\n';
          log.scrollTop = log.scrollHeight;
        } else if (d.type === 'summary') {
          renderStressSummary(d, scoreDiv);
        }
      }
    }
  } catch(e) {
    log.innerHTML += '<span class="fail">ERROR: ' + esc(e.message) + '</span>';
  } finally {
    btn.disabled = false;
  }
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
  h += '<div style="text-align:center;font-size:12px;color:#666;margin-bottom:12px;">Tests: ' + d.total_tests + ' | PASS: ' + d.total_pass + ' | FAIL: ' + d.total_fail + ' | Errors: ' + d.total_error + ' | Avg: ' + d.avg_duration_s + 's</div>';

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
    h += '<div style="margin-top:12px;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;">Top Violation Reasons</div>';
    for (const v in viols) {
      h += '<div style="font-size:12px;color:#ef5350;margin:2px 0;">' + esc(v) + ': ' + viols[v] + '</div>';
    }
  }

  el.innerHTML = h;
}

async function lc() {
  try {
    const r = await fetch('/api/openai/config');
    const d = await r.json();
    const dot = document.getElementById('kd');
    const st = document.getElementById('ks');
    if (d.key_set) {
      dot.className = 'kd on';
      st.textContent = d.key_preview + ' | ' + d.model;
      st.className = 'cfg-st ok';
    } else {
      dot.className = 'kd off';
      st.textContent = 'No key set';
      st.className = 'cfg-st';
    }
  } catch(e) {}
}

async function sav() {
  const k = document.getElementById('ak').value.trim();
  if (!k) return;
  await fetch('/api/openai/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({api_key: k, model: document.getElementById('md').value})
  });
  document.getElementById('ak').value = '';
  lc();
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

function renderViolations(viols) {
  if (!viols || viols.length === 0) return '<div class="no-viol">No violations detected</div>';
  let h = '<div class="viol">';
  viols.forEach(v => {
    h += '<div class="viol-item"><span class="viol-dot"></span>' + esc(v) + '</div>';
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

async function go(e) {
  e.preventDefault();
  const inp = document.getElementById('ui');
  const btn = document.getElementById('sb');
  const prompt = inp.value.trim();
  if (!prompt) return;

  ab('usr', 'You', esc(prompt));
  inp.value = '';
  btn.disabled = true;

  const ch = document.getElementById('ch');
  const ld = document.createElement('div');
  ld.className = 'ld';
  ld.innerHTML = '<div class="sp"></div><br>GPT-1 generating...';
  ch.appendChild(ld);
  ch.scrollTop = ch.scrollHeight;

  const steps = [
    {t: 3000, msg: 'GPT-2 verifying...', cls: 'sp s2'},
    {t: 8000, msg: 'GPT-3 arbitrating...', cls: 'sp s3'},
    {t: 14000, msg: 'Rewriting & re-verifying...', cls: 'sp'},
  ];
  const timers = steps.map(s => setTimeout(() => {
    if (ld.parentNode) ld.innerHTML = '<div class="' + s.cls + '"></div><br>' + s.msg;
  }, s.t));

  try {
    const r = await fetch('/api/pipeline', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: prompt,
        gpt1_system: document.getElementById('g1s').value,
        gpt2_system: document.getElementById('g2s').value.trim(),
        gpt3_system: document.getElementById('g3s').value.trim(),
      })
    });

    timers.forEach(t => clearTimeout(t));
    ld.remove();

    if (!r.ok) {
      const err = await r.json();
      ab('err', '', esc(err.detail || 'Request failed'));
      return;
    }

    const d = await r.json();

    // ---- GPT-1 output ----
    ab('g1', 'GPT-1 (Generator)', esc(d.gpt1_output));

    // ---- Bypass ----
    if (d.bypassed) {
      ab('byp', '', 'Activation phrase detected - verification bypassed');
      ab('vp', '', '&#10003; PASS (bypassed)');
      ab('fo', 'Final Output', esc(d.final_result));
      return;
    }

    // ---- GPT-2 results ----
    let g2body = renderClaimTable(d.claim_table) + renderViolations(d.violations);
    ab('g2', 'GPT-2 (Verifier) &mdash; ' + d.gpt2_verdict, g2body);

    if (d.gpt2_verdict === 'PASS') {
      ab('vp', '', '&#10003; PASS');
      ab('fo', 'Final Output', esc(d.final_result));
      return;
    }

    // ---- GPT-2 FAIL: show verdict ----
    ab('vf', '', '&#10007; GPT-2 FAIL &mdash; escalating to Arbiter');
    addDivider();

    // ---- GPT-3 Arbiter ----
    if (d.arbiter_invoked) {
      let g3body = '';

      // Decision badge
      const decLower = (d.arbiter_decision || '').toLowerCase().replace(/_/g, '');
      let decCls = 'blk';
      if (decLower === 'allowwithedits') decCls = 'awe';
      if (decLower === 'allowasunknownonly') decCls = 'auo';
      g3body += '<div class="arb-decision ' + decCls + '">' + esc(d.arbiter_decision) + '</div>';

      // Rationale
      if (d.arbiter_rationale && d.arbiter_rationale.length > 0) {
        g3body += '<div class="arb-rationale">';
        d.arbiter_rationale.forEach(r => {
          g3body += '<div class="arb-item"><span class="arb-dot"></span>' + esc(r) + '</div>';
        });
        g3body += '</div>';
      }

      // Edits
      if (d.arbiter_edits && d.arbiter_edits.length > 0) {
        g3body += '<div class="edit-list">';
        d.arbiter_edits.forEach(e => {
          g3body += '<div class="edit-item">';
          g3body += '<div class="edit-action ' + editActionCls(e.action) + '">' + esc(e.action) + '</div>';
          g3body += '<div class="edit-target">' + esc(e.target) + '</div>';
          if (e.replacement) g3body += '<div class="edit-repl">&rarr; ' + esc(e.replacement) + '</div>';
          g3body += '</div>';
        });
        g3body += '</div>';
      }

      // Policy notes
      if (d.arbiter_policy_notes && d.arbiter_policy_notes.length > 0) {
        g3body += '<div class="policy-notes">';
        d.arbiter_policy_notes.forEach(n => {
          g3body += '<div>' + esc(n) + '</div>';
        });
        g3body += '</div>';
      }

      ab('g3', 'GPT-3 (Arbiter)', g3body);
    }

    // ---- Rewrite loop ----
    if (d.rewrite_occurred) {
      addDivider();
      ab('rw', 'GPT-1 (Rewrite)', esc(d.rewrite_output));

      // Re-verification
      let rvBody = renderClaimTable(d.rewrite_claim_table) + renderViolations(d.rewrite_violations);
      ab('rv', 'GPT-2 (Re-verify) &mdash; ' + d.rewrite_verdict, rvBody);
    }

    // ---- Final verdict ----
    addDivider();
    if (d.final_verdict === 'PASS') {
      ab('vp', '', '&#10003; FINAL PASS');
      ab('fo', 'Final Output (Shown to You)', esc(d.final_result));
    } else {
      ab('vf', '', '&#10007; FINAL FAIL');
      let blockMsg = 'NO PASS - Output blocked by verification';
      if (d.arbiter_invoked && d.arbiter_decision === 'BLOCK' && d.arbiter_rationale && d.arbiter_rationale.length > 0) {
        blockMsg += '\\n\\nArbiter rationale:\\n' + d.arbiter_rationale.map(r => '- ' + r).join('\\n');
      }
      ab('fo blk', 'Final Output', blockMsg);
    }

  } catch(err) {
    timers.forEach(t => clearTimeout(t));
    ld.remove();
    ab('err', '', 'Error: ' + esc(err.message));
  } finally {
    btn.disabled = false;
    inp.focus();
  }
}

lc();
document.getElementById('ui').focus();
</script>
</body>
</html>
"""
