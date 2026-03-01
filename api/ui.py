"""Embedded chat UI served at /."""

UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Epistemic Verification Pipeline</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg-root: #09090b;
    --bg-surface-0: #111113;
    --bg-surface-1: #161618;
    --bg-surface-2: #1c1c1f;
    --bg-surface-3: #232326;
    --border-subtle: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.1);
    --border-focus: rgba(96,165,250,0.4);
    --text-primary: #e4e4e7;
    --text-secondary: #a1a1aa;
    --text-tertiary: #71717a;
    --text-muted: #52525b;
    --accent-blue: #60a5fa;
    --accent-amber: #f59e0b;
    --accent-violet: #a78bfa;
    --accent-emerald: #34d399;
    --accent-rose: #f87171;
    --accent-teal: #2dd4bf;
    --accent-lime: #a3e635;
    --radius-sm: 8px;
    --radius-md: 10px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --radius-pill: 20px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.2);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.3);
    --shadow-glow-emerald: 0 0 20px rgba(52,211,153,0.15);
    --shadow-glow-rose: 0 0 20px rgba(248,113,113,0.15);
    --transition: 0.2s ease;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-root);
    color: var(--text-primary);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* ---- Scrollbar ---- */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

  /* ---- Top Bar ---- */
  .top-bar {
    background: rgba(17,17,19,0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-subtle);
    padding: 0 24px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    position: sticky;
    top: 0;
    z-index: 50;
  }
  .top-bar h1 {
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0;
  }
  .top-bar h1 .g1 { color: var(--accent-blue); }
  .top-bar h1 .arr { color: var(--text-muted); margin: 0 8px; font-size: 11px; }
  .top-bar h1 .g2 { color: var(--accent-amber); }
  .top-bar h1 .g3 { color: var(--accent-violet); }
  .top-bar .right-controls { display: flex; align-items: center; gap: 6px; }

  .cfg-btn {
    background: transparent;
    color: var(--text-tertiary);
    border: 1px solid var(--border-subtle);
    padding: 6px 12px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .cfg-btn:hover { background: var(--bg-surface-1); color: var(--text-secondary); border-color: var(--border-hover); }
  .cfg-btn.stress-btn { color: var(--accent-violet); border-color: rgba(167,139,250,0.2); }
  .cfg-btn.stress-btn:hover { background: rgba(167,139,250,0.08); border-color: rgba(167,139,250,0.3); }

  .kd { display: inline-block; width: 7px; height: 7px; border-radius: 50%; }
  .kd.on { background: var(--accent-emerald); box-shadow: 0 0 6px rgba(52,211,153,0.4); }
  .kd.off { background: var(--accent-rose); box-shadow: 0 0 6px rgba(248,113,113,0.3); }

  /* Gear icon */
  .gear-icon {
    width: 15px; height: 15px;
    stroke: currentColor; fill: none;
    stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
  }

  /* ---- Settings Side Panel ---- */
  .cfg-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: 90;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
  }
  .cfg-overlay.open { opacity: 1; pointer-events: auto; }

  .cfg-drawer {
    position: fixed;
    top: 0; right: 0; bottom: 0;
    width: 420px;
    max-width: 90vw;
    background: var(--bg-surface-0);
    border-left: 1px solid var(--border-subtle);
    z-index: 100;
    transform: translateX(100%);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
    box-shadow: -8px 0 30px rgba(0,0,0,0.3);
  }
  .cfg-drawer.open { transform: translateX(0); }

  .cfg-drawer-header {
    padding: 20px 24px 16px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  .cfg-drawer-header h2 {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .cfg-close-btn {
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 4px;
    border-radius: 6px;
    transition: all var(--transition);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .cfg-close-btn:hover { background: var(--bg-surface-2); color: var(--text-secondary); }

  .cfg-in {
    padding: 20px 24px;
    overflow-y: auto;
    flex: 1;
  }
  .cfg-section {
    margin-bottom: 24px;
  }
  .cfg-section-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 12px;
  }
  .cfg-section-title.tavily { color: var(--accent-violet); }

  .cfg-in label {
    display: block;
    font-size: 11px;
    color: var(--text-tertiary);
    font-weight: 500;
    margin: 12px 0 6px;
  }
  .cfg-in label:first-child { margin-top: 0; }
  .cfg-in input, .cfg-in select, .cfg-in textarea {
    width: 100%;
    padding: 9px 12px;
    background: var(--bg-root);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: all var(--transition);
  }
  .cfg-in input::placeholder, .cfg-in textarea::placeholder {
    color: var(--text-muted);
  }
  .cfg-in textarea { resize: vertical; min-height: 44px; line-height: 1.5; }
  .cfg-in input:focus, .cfg-in textarea:focus, .cfg-in select:focus {
    border-color: var(--border-focus);
    box-shadow: 0 0 0 3px rgba(96,165,250,0.1);
  }
  .cfg-in select { cursor: pointer; -webkit-appearance: none; }
  .cfg-row { display: flex; gap: 8px; align-items: flex-end; }
  .cfg-row input { flex: 1; }

  .btn-s {
    padding: 9px 16px;
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    white-space: nowrap;
  }
  .btn-s:hover { background: var(--bg-surface-3); color: var(--text-primary); border-color: var(--border-hover); }

  .cfg-st { font-size: 11px; color: var(--text-muted); margin-top: 6px; }
  .cfg-st.ok { color: var(--accent-emerald); }

  .cfg-divider {
    height: 1px;
    background: var(--border-subtle);
    margin: 20px 0;
  }

  .tavily-toggle {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    margin-top: 8px;
    padding: 8px 12px;
    background: var(--bg-root);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    transition: all var(--transition);
  }
  .tavily-toggle:hover { border-color: var(--border-hover); }
  .tavily-toggle input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: var(--accent-violet);
    cursor: pointer;
  }
  .tavily-toggle span {
    font-size: 12.5px;
    color: var(--text-secondary);
    font-weight: 400;
  }

  /* ---- Chat Area ---- */
  .chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px 24px 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 860px;
    width: 100%;
    margin: 0 auto;
    scroll-behavior: smooth;
  }

  /* ---- Message Bubbles ---- */
  .b {
    max-width: 92%;
    padding: 16px 20px;
    border-radius: var(--radius-lg);
    line-height: 1.65;
    font-size: 13.5px;
    white-space: pre-wrap;
    animation: msgIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    transition: all var(--transition);
  }
  .b .w {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .b .w::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* User message */
  .b.usr {
    align-self: flex-end;
    background: linear-gradient(135deg, rgba(96,165,250,0.12), rgba(96,165,250,0.06));
    border: 1px solid rgba(96,165,250,0.12);
    color: var(--text-primary);
  }
  .b.usr .w { color: var(--accent-blue); }
  .b.usr .w::before { background: var(--accent-blue); }

  /* GPT-1 */
  .b.g1 {
    align-self: flex-start;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-blue);
    color: var(--text-primary);
  }
  .b.g1 .w { color: var(--accent-blue); }
  .b.g1 .w::before { background: var(--accent-blue); }

  /* GPT-2 */
  .b.g2 {
    align-self: flex-start;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-amber);
    color: var(--text-secondary);
    font-size: 13px;
    white-space: normal;
  }
  .b.g2 .w { color: var(--accent-amber); }
  .b.g2 .w::before { background: var(--accent-amber); }

  /* GPT-3 */
  .b.g3 {
    align-self: flex-start;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-violet);
    color: var(--text-secondary);
    font-size: 13px;
    white-space: normal;
  }
  .b.g3 .w { color: var(--accent-violet); }
  .b.g3 .w::before { background: var(--accent-violet); }

  /* Search Results */
  .b.sr {
    align-self: flex-start;
    background: var(--bg-surface-1);
    border: 1px solid rgba(167,139,250,0.12);
    border-left: 3px solid var(--accent-violet);
    color: var(--text-secondary);
    font-size: 13px;
    white-space: normal;
  }
  .b.sr .w { color: var(--accent-violet); }
  .b.sr .w::before { background: var(--accent-violet); }
  .b.sr a { color: var(--accent-violet); text-decoration: none; transition: color var(--transition); }
  .b.sr a:hover { color: #c4b5fd; text-decoration: underline; }
  .b.sr .src-tbl { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
  .b.sr .src-tbl th {
    text-align: left;
    padding: 6px 10px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
  }
  .b.sr .src-tbl td {
    padding: 8px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    vertical-align: top;
  }
  .b.sr .src-tbl tr:hover td { background: rgba(255,255,255,0.02); }

  /* Rewrite */
  .b.rw {
    align-self: flex-start;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-teal);
    color: var(--text-secondary);
    font-size: 13px;
    white-space: pre-wrap;
  }
  .b.rw .w { color: var(--accent-teal); }
  .b.rw .w::before { background: var(--accent-teal); }

  /* Re-verify */
  .b.rv {
    align-self: flex-start;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-lime);
    color: var(--text-secondary);
    font-size: 13px;
    white-space: normal;
  }
  .b.rv .w { color: var(--accent-lime); }
  .b.rv .w::before { background: var(--accent-lime); }

  /* Verdict PASS */
  .b.vp {
    align-self: center;
    background: rgba(52,211,153,0.06);
    border: 1px solid rgba(52,211,153,0.2);
    color: var(--accent-emerald);
    text-align: center;
    font-weight: 600;
    font-size: 14px;
    padding: 14px 32px;
    white-space: normal;
    border-radius: var(--radius-pill);
    box-shadow: var(--shadow-glow-emerald);
  }

  /* Verdict FAIL */
  .b.vf {
    align-self: center;
    background: rgba(248,113,113,0.06);
    border: 1px solid rgba(248,113,113,0.2);
    color: var(--accent-rose);
    text-align: center;
    font-weight: 600;
    font-size: 14px;
    padding: 14px 32px;
    white-space: normal;
    border-radius: var(--radius-pill);
    box-shadow: var(--shadow-glow-rose);
  }

  /* Final Output */
  .b.fo {
    align-self: center;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    width: 100%;
    max-width: 100%;
    border-radius: var(--radius-lg);
  }
  .b.fo .w { color: var(--accent-emerald); }
  .b.fo .w::before { background: var(--accent-emerald); }
  .b.fo.blk .w { color: var(--accent-rose); }
  .b.fo.blk .w::before { background: var(--accent-rose); }
  .b.fo.blk { border-color: rgba(248,113,113,0.15); color: var(--accent-rose); text-align: center; font-weight: 500; }

  /* Bypass */
  .b.byp {
    align-self: center;
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.15);
    color: var(--accent-amber);
    text-align: center;
    font-size: 12px;
    padding: 10px 20px;
    white-space: normal;
    border-radius: var(--radius-pill);
  }

  /* Error */
  .err {
    background: rgba(248,113,113,0.06);
    border: 1px solid rgba(248,113,113,0.15);
    color: var(--accent-rose);
    padding: 14px 18px;
    border-radius: var(--radius-lg);
    font-size: 13px;
    align-self: center;
    white-space: normal;
  }

  /* ---- Divider ---- */
  .divider {
    align-self: center;
    width: 60%;
    border: none;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin: 6px 0;
  }

  /* ---- Claim Table ---- */
  .ct { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
  .ct th {
    text-align: left;
    padding: 8px 10px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
  }
  .ct td {
    padding: 10px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    vertical-align: top;
  }
  .ct tr:nth-child(even) td { background: rgba(255,255,255,0.015); }
  .ct tr:hover td { background: rgba(255,255,255,0.03); }
  .ct .cat { font-weight: 500; }

  /* Category pills */
  .cat-sup, .cat-inf, .cat-hyp, .cat-uns, .cat-usr {
    display: inline-block;
    padding: 2px 8px;
    border-radius: var(--radius-pill);
    font-size: 11px;
    font-weight: 500;
  }
  .cat-sup { color: var(--accent-emerald); background: rgba(52,211,153,0.1); }
  .cat-inf { color: var(--accent-amber); background: rgba(245,158,11,0.1); }
  .cat-hyp { color: var(--accent-violet); background: rgba(167,139,250,0.1); }
  .cat-uns { color: var(--accent-rose); background: rgba(248,113,113,0.1); }
  .cat-usr { color: var(--accent-blue); background: rgba(96,165,250,0.1); }

  /* ---- Confidence Bar ---- */
  .conf-bar-wrap { margin-top: 14px; }
  .conf-bar-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 6px;
  }
  .conf-bar {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background: var(--bg-root);
  }
  .conf-bar .seg { height: 100%; transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1); }
  .conf-bar .seg-obs { background: var(--accent-emerald); }
  .conf-bar .seg-inf { background: var(--accent-amber); }
  .conf-bar .seg-hyp { background: var(--accent-violet); }
  .conf-bar .seg-uns { background: var(--accent-rose); }
  .conf-bar .seg-usr { background: var(--accent-blue); }
  .conf-legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: 11px; color: var(--text-tertiary); }
  .conf-legend .lg { display: flex; align-items: center; gap: 5px; }
  .conf-legend .lg .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .conf-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 12px;
    border-radius: var(--radius-pill);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .conf-badge.high { background: rgba(52,211,153,0.1); color: var(--accent-emerald); border: 1px solid rgba(52,211,153,0.2); }
  .conf-badge.medium { background: rgba(245,158,11,0.1); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.2); }
  .conf-badge.low { background: rgba(248,113,113,0.1); color: var(--accent-rose); border: 1px solid rgba(248,113,113,0.2); }
  .conf-badge.unknown { background: rgba(255,255,255,0.04); color: var(--text-tertiary); border: 1px solid var(--border-subtle); }

  /* ---- Violations ---- */
  .viol { margin-top: 10px; }
  .viol-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    font-size: 12px;
    color: var(--accent-rose);
    margin-bottom: 6px;
    padding: 6px 10px;
    background: rgba(248,113,113,0.04);
    border-left: 2px solid rgba(248,113,113,0.3);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  }
  .viol-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent-rose); flex-shrink: 0; margin-top: 5px; }
  .no-viol {
    color: var(--accent-emerald);
    font-size: 12px;
    margin-top: 10px;
    padding: 6px 10px;
    background: rgba(52,211,153,0.04);
    border-left: 2px solid rgba(52,211,153,0.3);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  }

  /* ---- Arbiter Details ---- */
  .arb-rationale { margin-top: 10px; }
  .arb-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    font-size: 12px;
    color: var(--accent-violet);
    margin-bottom: 6px;
    padding: 6px 10px;
    background: rgba(167,139,250,0.04);
    border-left: 2px solid rgba(167,139,250,0.3);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  }
  .arb-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent-violet); flex-shrink: 0; margin-top: 5px; }
  .arb-decision {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 8px;
    padding: 4px 14px;
    border-radius: var(--radius-pill);
    display: inline-block;
  }
  .arb-decision.blk { color: var(--accent-rose); background: rgba(248,113,113,0.08); }
  .arb-decision.awe { color: var(--accent-amber); background: rgba(245,158,11,0.08); }
  .arb-decision.auo { color: var(--accent-teal); background: rgba(45,212,191,0.08); }

  .edit-list { margin-top: 10px; font-size: 12px; }
  .edit-item {
    background: var(--bg-root);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    margin-bottom: 6px;
    transition: all var(--transition);
  }
  .edit-item:hover { border-color: var(--border-hover); }
  .edit-action {
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
  }
  .edit-action.del { color: var(--accent-rose); background: rgba(248,113,113,0.1); }
  .edit-action.rew { color: var(--accent-amber); background: rgba(245,158,11,0.1); }
  .edit-action.mtu { color: var(--accent-teal); background: rgba(45,212,191,0.1); }
  .edit-target { color: var(--text-tertiary); font-style: italic; margin-top: 2px; }
  .edit-repl { color: var(--accent-emerald); margin-top: 4px; padding: 4px 8px; background: rgba(52,211,153,0.04); border-radius: 4px; }
  .policy-notes {
    margin-top: 10px;
    padding: 10px 14px;
    background: var(--bg-root);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    font-size: 11px;
    color: var(--text-tertiary);
    line-height: 1.6;
  }

  /* ---- Loading States ---- */
  .ld {
    align-self: center;
    padding: 24px;
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .sp {
    display: flex;
    gap: 4px;
    align-items: center;
    justify-content: center;
  }
  .sp .dot-pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-blue);
    animation: dotPulse 1.4s ease-in-out infinite;
  }
  .sp .dot-pulse:nth-child(2) { animation-delay: 0.2s; }
  .sp .dot-pulse:nth-child(3) { animation-delay: 0.4s; }
  .sp.s2 .dot-pulse { background: var(--accent-amber); }
  .sp.s3 .dot-pulse { background: var(--accent-violet); }

  .ld-text { font-size: 12px; color: var(--text-muted); font-weight: 500; }

  /* ---- Input Bar ---- */
  .ibar {
    padding: 16px 24px 20px;
    flex-shrink: 0;
    background: linear-gradient(to top, var(--bg-root) 60%, transparent);
    position: relative;
  }
  .ibar form {
    display: flex;
    gap: 10px;
    max-width: 860px;
    margin: 0 auto;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 4px 4px 4px 16px;
    align-items: center;
    transition: all var(--transition);
    box-shadow: var(--shadow-md);
  }
  .ibar form:focus-within {
    border-color: var(--border-focus);
    box-shadow: var(--shadow-md), 0 0 0 3px rgba(96,165,250,0.08);
  }
  .ibar input {
    flex: 1;
    padding: 10px 0;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 13.5px;
    outline: none;
    font-family: inherit;
  }
  .ibar input::placeholder { color: var(--text-muted); }
  .ibar button {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-blue);
    color: #fff;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all var(--transition);
    flex-shrink: 0;
  }
  .ibar button:hover { background: #4d94f7; transform: scale(1.04); }
  .ibar button:disabled { opacity: 0.25; cursor: not-allowed; transform: none; }
  .ibar button svg { width: 16px; height: 16px; }

  /* ---- Stress Test Panel ---- */
  .stress-panel {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(9,9,11,0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    z-index: 100;
    overflow-y: auto;
  }
  .stress-panel.open {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 24px;
    animation: fadeInPanel 0.3s ease;
  }
  .stress-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    max-width: 900px;
    margin-bottom: 20px;
  }
  .stress-hdr h2 { font-size: 18px; font-weight: 600; color: var(--text-primary); letter-spacing: -0.01em; }
  .stress-close {
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    padding: 7px 16px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
  }
  .stress-close:hover { background: var(--bg-surface-3); border-color: var(--border-hover); }

  .stress-controls {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    width: 100%;
    max-width: 900px;
    margin-bottom: 20px;
  }
  .stress-controls select, .stress-controls input {
    padding: 9px 12px;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: all var(--transition);
  }
  .stress-controls select:focus, .stress-controls input:focus {
    border-color: var(--border-focus);
    box-shadow: 0 0 0 3px rgba(96,165,250,0.1);
  }
  .stress-run {
    padding: 9px 24px;
    background: linear-gradient(135deg, var(--accent-violet), #7c3aed);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    white-space: nowrap;
  }
  .stress-run:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(167,139,250,0.3); }
  .stress-run:disabled { opacity: 0.3; cursor: not-allowed; transform: none; box-shadow: none; }

  .stress-log {
    width: 100%;
    max-width: 900px;
    background: var(--bg-surface-0);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 12px;
    color: var(--text-tertiary);
    min-height: 200px;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.7;
  }
  .stress-log .pass { color: var(--accent-emerald); }
  .stress-log .fail { color: var(--accent-rose); }
  .stress-log .arb { color: var(--accent-violet); }
  .stress-log .rew { color: var(--accent-teal); }

  .stress-score { width: 100%; max-width: 900px; margin-top: 24px; }

  .pss-big {
    font-size: 56px;
    font-weight: 800;
    text-align: center;
    margin: 20px 0 8px;
    letter-spacing: -0.02em;
    position: relative;
  }
  .pss-big::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 120px;
    height: 120px;
    border-radius: 50%;
    filter: blur(40px);
    opacity: 0.3;
    z-index: -1;
  }
  .pss-big.s90 { color: var(--accent-emerald); }
  .pss-big.s90::after { background: var(--accent-emerald); }
  .pss-big.s75 { color: var(--accent-amber); }
  .pss-big.s75::after { background: var(--accent-amber); }
  .pss-big.s60 { color: #fb923c; }
  .pss-big.s60::after { background: #fb923c; }
  .pss-big.s0 { color: var(--accent-rose); }
  .pss-big.s0::after { background: var(--accent-rose); }

  .pss-band {
    text-align: center;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 20px;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 20px;
  }
  .metric-card {
    background: linear-gradient(180deg, var(--bg-surface-1), var(--bg-surface-0));
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px;
    text-align: center;
    transition: all var(--transition);
  }
  .metric-card:hover { border-color: var(--border-hover); transform: translateY(-1px); }
  .metric-card .mv { font-size: 22px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.01em; }
  .metric-card .ml { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; font-weight: 500; }
  .metric-card .mp { font-size: 11px; color: var(--accent-rose); margin-top: 4px; font-weight: 500; }

  .cat-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .cat-table th {
    text-align: left;
    padding: 10px 12px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
  }
  .cat-table td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); }
  .cat-table tr:nth-child(even) td { background: rgba(255,255,255,0.015); }
  .cat-table tr:hover td { background: rgba(255,255,255,0.03); }
  .cat-table .pr { font-weight: 600; }

  /* ---- Animations ---- */
  @keyframes msgIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.2); }
  }
  @keyframes fadeInPanel {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  @keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  }

  /* ---- Responsive ---- */
  @media (max-width: 640px) {
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    .stress-controls { flex-wrap: wrap; }
    .cfg-drawer { width: 100%; max-width: 100%; }
    .b { max-width: 98%; }
    .top-bar { padding: 0 16px; }
    .chat { padding: 16px; }
    .ibar { padding: 12px 16px 16px; }
  }
</style>
</head>
<body>

<div class="top-bar">
  <h1>
    <span class="g1">GPT-1</span><span class="arr">&rarr;</span><span class="g2">GPT-2</span><span class="arr">&rarr;</span><span class="g3">GPT-3</span>
  </h1>
  <div class="right-controls">
    <button class="cfg-btn stress-btn" onclick="openStress()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      Stress Test
    </button>
    <button class="cfg-btn" onclick="tog()">
      <span class="kd off" id="kd"></span>
      <svg class="gear-icon" viewBox="0 0 24 24"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
    </button>
  </div>
</div>

<!-- Settings Overlay -->
<div class="cfg-overlay" id="co" onclick="tog()"></div>

<!-- Settings Side Panel -->
<div class="cfg-drawer" id="cd">
  <div class="cfg-drawer-header">
    <h2>Settings</h2>
    <button class="cfg-close-btn" onclick="tog()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>
  <div class="cfg-in">
    <div class="cfg-section">
      <div class="cfg-section-title">OpenAI Configuration</div>
      <label>API Key</label>
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
    </div>

    <div class="cfg-divider"></div>

    <div class="cfg-section">
      <div class="cfg-section-title tavily">Web Search (Tavily)</div>
      <label>API Key</label>
      <div class="cfg-row">
        <input id="tk" type="password" placeholder="tvly-...">
        <button class="btn-s" onclick="savTav()">Save</button>
      </div>
      <label class="tavily-toggle">
        <input id="te" type="checkbox" style="width:auto;">
        <span>Enable web search enrichment</span>
      </label>
      <div class="cfg-st" id="ts">Loading...</div>
    </div>

    <div class="cfg-divider"></div>

    <div class="cfg-section">
      <div class="cfg-section-title">Pipeline Prompts</div>
      <label>GPT-1 System Prompt (Generator)</label>
      <textarea id="g1s" rows="6">You are GPT-1, a structured reasoning and synthesis engine.

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
</div>

<div class="chat" id="ch"></div>

<!-- Stress Test Panel -->
<div class="stress-panel" id="sp">
  <div class="stress-hdr">
    <h2>Pipeline Stability Score (PSS)</h2>
    <button class="stress-close" onclick="closeStress()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      Close
    </button>
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
    <input type="number" id="sn" min="1" max="10" value="" placeholder="Per-cat limit" style="width:110px;">
    <button class="stress-run" id="sr" onclick="runStress()">Run Stress Test</button>
  </div>
  <div class="stress-log" id="sl"></div>
  <div class="stress-score" id="ss"></div>
</div>

<div class="ibar">
  <form onsubmit="go(event)">
    <input type="text" id="ui" placeholder="Ask anything..." autocomplete="off">
    <button type="submit" id="sb">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </form>
</div>

<script>
function tog() {
  document.getElementById('cd').classList.toggle('open');
  document.getElementById('co').classList.toggle('open');
}
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
          if (d.error) extra += ' <span class="fail">' + esc(d.error) + '</span>';
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
      h += '<div style="font-size:12px;color:var(--accent-rose);margin:4px 0;padding:4px 8px;background:rgba(248,113,113,0.04);border-radius:4px;">' + esc(v) + ': ' + viols[v] + '</div>';
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

async function loadTav() {
  const st = document.getElementById('ts');
  try {
    const r = await fetch('/api/tavily/config');
    const d = await r.json();
    if (d.key_set) {
      document.getElementById('te').checked = d.enabled;
      st.textContent = d.enabled ? 'Search enabled' : 'Search disabled';
      st.className = d.enabled ? 'cfg-st ok' : 'cfg-st';
    } else {
      document.getElementById('te').checked = false;
      st.textContent = 'No Tavily key set';
      st.className = 'cfg-st';
    }
  } catch(e) { st.textContent = 'Error loading config'; }
}

async function savTav() {
  const k = document.getElementById('tk').value.trim();
  if (!k) return;
  const en = document.getElementById('te').checked;
  await fetch('/api/tavily/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({api_key: k, enabled: en})
  });
  document.getElementById('tk').value = '';
  loadTav();
}

document.getElementById('te').addEventListener('change', async function() {
  const en = this.checked;
  try {
    await fetch('/api/tavily/toggle?enabled=' + en, {method: 'POST'});
    document.getElementById('ts').textContent = en ? 'Search enabled' : 'Search disabled';
    document.getElementById('ts').className = en ? 'cfg-st ok' : 'cfg-st';
  } catch(e) {
    this.checked = !en;
  }
});

function renderSearchSources(sources) {
  if (!sources || !sources.length) return '';
  let h = '<table class="src-tbl"><thead><tr><th>#</th><th>Title</th><th>Snippet</th></tr></thead><tbody>';
  sources.forEach(function(s, i) {
    h += '<tr><td>[' + (i+1) + ']</td>';
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
  h += '</div>';
  return h;
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

function makeLoader(spinnerCls, text) {
  return '<div class="sp ' + spinnerCls + '"><span class="dot-pulse"></span><span class="dot-pulse"></span><span class="dot-pulse"></span></div><span class="ld-text">' + text + '</span>';
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
  ld.innerHTML = makeLoader('', 'GPT-1 generating...');
  ch.appendChild(ld);
  ch.scrollTop = ch.scrollHeight;

  const steps = [
    {t: 3000, msg: 'GPT-2 verifying...', cls: 's2'},
    {t: 8000, msg: 'GPT-3 arbitrating...', cls: 's3'},
    {t: 14000, msg: 'Rewriting & re-verifying...', cls: ''},
  ];
  const timers = steps.map(s => setTimeout(() => {
    if (ld.parentNode) ld.innerHTML = makeLoader(s.cls, s.msg);
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

    // ---- Web Search ----
    if (d.search_performed && d.search_sources && d.search_sources.length) {
      ab('sr', 'Web Search (' + d.search_sources.length + ' sources)', renderSearchSources(d.search_sources));
    }

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
    let g2body = renderClaimTable(d.claim_table) + renderConfidence(d.confidence) + renderViolations(d.violations);
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
      let rvBody = renderClaimTable(d.rewrite_claim_table) + renderConfidence(d.confidence) + renderViolations(d.rewrite_violations);
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
loadTav();
document.getElementById('ui').focus();
</script>
</body>
</html>
"""
