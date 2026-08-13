"""Embedded chat UI served at /."""

UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Epistemic Verification Pipeline</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  :root, [data-theme="dark"] {
    /* Dark navy (readable blue—not neutral black) */
    --bg-root: #0f2847;
    --bg-surface-0: #15365d;
    --bg-surface-1: #1a3f6c;
    --bg-surface-2: #1f4878;
    --bg-surface-3: #255184;
    --border-subtle: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.1);
    --border-focus: rgba(59,130,246,0.55);
    --text-primary: #e4e4e7;
    --text-secondary: #a1a1aa;
    --text-tertiary: #71717a;
    --text-muted: #52525b;
    --accent-blue: #3b82f6;
    --accent-blue-hover: #2563eb;
    --accent-blue-rgb: 59,130,246;
    --accent-amber: #d97706;
    --accent-amber-rgb: 217,119,6;
    --accent-violet: #7c3aed;
    --accent-violet-rgb: 124,58,237;
    --accent-emerald: #059669;
    --accent-emerald-rgb: 5,150,105;
    --accent-rose: #e11d48;
    --accent-rose-rgb: 225,29,72;
    --accent-teal: #0d9488;
    --accent-teal-rgb: 13,148,136;
    --accent-lime: #65a30d;
    --accent-lime-rgb: 101,163,13;
    --radius-sm: 8px;
    --radius-md: 10px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --radius-pill: 20px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.2);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.3);
    --shadow-glow-emerald: 0 0 0 rgba(0,0,0,0);
    --shadow-glow-rose: 0 0 0 rgba(0,0,0,0);
    --transition: 0.2s ease;
  }
  [data-theme="light"] {
    --bg-root: #f8f9fa;
    --bg-surface-0: #ffffff;
    --bg-surface-1: #f1f3f5;
    --bg-surface-2: #e9ecef;
    --bg-surface-3: #dee2e6;
    --border-subtle: rgba(0,0,0,0.08);
    --border-hover: rgba(0,0,0,0.15);
    --border-focus: rgba(59,130,246,0.5);
    --text-primary: #1a1a2e;
    --text-secondary: #495057;
    --text-tertiary: #6c757d;
    --text-muted: #adb5bd;
    --accent-blue: #3b82f6;
    --accent-blue-hover: #2563eb;
    --accent-blue-rgb: 59,130,246;
    --accent-amber: #d97706;
    --accent-amber-rgb: 217,119,6;
    --accent-violet: #7c3aed;
    --accent-violet-rgb: 124,58,237;
    --accent-emerald: #059669;
    --accent-emerald-rgb: 5,150,105;
    --accent-rose: #e11d48;
    --accent-rose-rgb: 225,29,72;
    --accent-teal: #0d9488;
    --accent-teal-rgb: 13,148,136;
    --accent-lime: #65a30d;
    --accent-lime-rgb: 101,163,13;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
    --shadow-glow-emerald: 0 0 0 rgba(0,0,0,0);
    --shadow-glow-rose: 0 0 0 rgba(0,0,0,0);
  }

  html {
    background: var(--bg-root);
  }

  body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: var(--bg-root);
    background-image:
      linear-gradient(180deg, #173a63 0%, #0c213c 42%, #08182c 100%),
      radial-gradient(ellipse 90% 70% at 50% -15%, rgba(56, 189, 248, 0.12), transparent 55%),
      radial-gradient(ellipse 50% 50% at 100% 30%, rgba(99, 102, 241, 0.14), transparent 50%);
    background-attachment: fixed;
    color: var(--text-primary);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  [data-theme="light"] body {
    background-image: none;
    background-color: var(--bg-root);
  }

  /* ---- Workspace: main chat column + insight rail ---- */
  .workspace {
    flex: 1;
    display: flex;
    flex-direction: row;
    min-height: 0;
    min-width: 0;
    position: relative;
  }
  .main-stack {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
  }

  .insight-rail {
    width: min(300px, 34vw);
    flex-shrink: 0;
    border-left: 1px solid var(--border-subtle);
    background: var(--bg-surface-0);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    display: flex;
    flex-direction: column;
    max-height: calc(100vh - 52px);
    position: sticky;
    top: 52px;
    align-self: flex-start;
    transition: width 0.22s ease, border-color 0.2s ease, opacity 0.2s ease;
    z-index: 5;
  }
  .insight-rail.collapsed {
    width: 0;
    min-width: 0;
    border-left-color: transparent;
    opacity: 0;
    pointer-events: none;
    overflow: hidden;
  }
  .insight-rail-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--border-subtle);
    flex-shrink: 0;
  }
  .insight-rail-head span {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-tertiary);
  }
  .insight-rail-toggle {
    background: var(--bg-surface-2);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    border-radius: var(--radius-sm);
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    transition: var(--transition);
  }
  .insight-rail-toggle:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
  }
  .insight-rail-inner {
    flex: 1;
    overflow-y: auto;
    padding: 12px 14px 20px;
    min-height: 120px;
  }
  .insight-empty {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.55;
  }
  .insight-body.hidden { display: none; }
  .insight-verdict-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }
  .insight-badge {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 4px 10px;
    border-radius: var(--radius-pill);
    text-transform: uppercase;
  }
  .insight-badge.pass {
    background: rgba(var(--accent-emerald-rgb), 0.15);
    color: var(--accent-emerald);
    border: 1px solid rgba(var(--accent-emerald-rgb), 0.25);
  }
  .insight-badge.fail {
    background: rgba(var(--accent-rose-rgb), 0.12);
    color: var(--accent-rose);
    border: 1px solid rgba(var(--accent-rose-rgb), 0.22);
  }
  .insight-label {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
    flex: 1;
    min-width: 140px;
  }
  .insight-meter-wrap {
    margin-bottom: 14px;
  }
  .insight-meter-wrap .im-label {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .insight-meter {
    height: 6px;
    border-radius: 3px;
    background: var(--bg-surface-2);
    overflow: hidden;
  }
  .insight-meter > i {
    display: block;
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue));
    transition: width 0.4s ease;
  }
  .insight-kv {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-bottom: 6px;
    line-height: 1.45;
  }
  .insight-kv strong { color: var(--text-secondary); font-weight: 600; }
  .insight-section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin: 16px 0 8px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }
  .insight-section-title:first-of-type {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
  }
  .insight-claim {
    font-size: 11px;
    color: var(--text-secondary);
    padding: 8px 10px;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    line-height: 1.45;
  }
  .insight-claim .ic-cat {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent-violet);
    margin-bottom: 4px;
  }
  .insight-src-wrap { margin-top: 4px; }
  .insight-src-wrap .sr-item { font-size: 11px; margin-bottom: 8px; }
  .insight-src-wrap a { color: var(--accent-blue); text-decoration: none; word-break: break-all; }
  .insight-src-wrap a:hover { text-decoration: underline; }

  .insight-fab {
    display: none;
    position: fixed;
    bottom: 88px;
    right: 16px;
    z-index: 40;
    padding: 10px 14px;
    border-radius: var(--radius-pill);
    border: 1px solid var(--border-subtle);
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    box-shadow: var(--shadow-lg);
    transition: var(--transition);
  }
  .insight-fab:hover { color: var(--text-primary); border-color: var(--border-hover); }

  .brand-cluster {
    display: flex;
    flex-direction: column;
    gap: 2px;
    align-items: flex-start;
  }
  .brand-cluster h1 { margin: 0; }
  .brand-tagline {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0;
    padding-left: 1px;
  }

  /* ---- Scrollbar ---- */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

  /* ---- Top Bar ---- */
  .top-bar {
    background: rgba(15, 40, 71, 0.88);
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
  .cfg-btn.stress-btn { color: var(--accent-blue); border-color: rgba(var(--accent-blue-rgb),0.28); }
  .cfg-btn.stress-btn:hover { background: rgba(var(--accent-blue-rgb),0.08); border-color: rgba(var(--accent-blue-rgb),0.35); }

  .kd { display: inline-block; width: 7px; height: 7px; border-radius: 50%; }
  .kd.on { background: var(--accent-emerald); box-shadow: 0 0 6px rgba(var(--accent-emerald-rgb),0.4); }
  .kd.off { background: var(--accent-rose); box-shadow: 0 0 6px rgba(var(--accent-rose-rgb),0.3); }

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
    box-shadow: 0 0 0 3px rgba(var(--accent-blue-rgb),0.1);
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
    min-height: 0;
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
    background: rgba(var(--accent-blue-rgb),0.06);
    border: 1px solid rgba(var(--accent-blue-rgb),0.18);
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
    border: 1px solid rgba(var(--accent-violet-rgb),0.12);
    border-left: 3px solid var(--accent-violet);
    color: var(--text-secondary);
    font-size: 13px;
    white-space: normal;
  }
  .b.sr .w { color: var(--accent-violet); }
  .b.sr .w::before { background: var(--accent-violet); }
  .b.sr a { color: var(--accent-violet); text-decoration: none; transition: color var(--transition); }
  .b.sr a:hover { color: var(--accent-violet); text-decoration: underline; }
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
    background: rgba(var(--accent-emerald-rgb),0.06);
    border: 1px solid rgba(var(--accent-emerald-rgb),0.2);
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
    background: rgba(var(--accent-rose-rgb),0.06);
    border: 1px solid rgba(var(--accent-rose-rgb),0.2);
    color: var(--accent-rose);
    text-align: center;
    font-weight: 600;
    font-size: 14px;
    padding: 14px 32px;
    white-space: normal;
    border-radius: var(--radius-pill);
    box-shadow: var(--shadow-glow-rose);
  }

  /* Final Output — base overrides in hero section below */
  .b.fo .w { color: var(--accent-emerald); font-size: 11px; }
  .b.fo .w::before { background: var(--accent-emerald); }
  .b.fo.blk .w { color: var(--accent-rose); }
  .b.fo.blk .w::before { background: var(--accent-rose); }
  .b.fo.blk { border-color: rgba(var(--accent-rose-rgb),0.15); color: var(--accent-rose); text-align: center; font-weight: 500; background: rgba(var(--accent-rose-rgb),0.03); box-shadow: 0 0 30px rgba(var(--accent-rose-rgb),0.04); }

  /* Bypass */
  .b.byp {
    align-self: center;
    background: rgba(var(--accent-amber-rgb),0.06);
    border: 1px solid rgba(var(--accent-amber-rgb),0.15);
    color: var(--accent-amber);
    text-align: center;
    font-size: 12px;
    padding: 10px 20px;
    white-space: normal;
    border-radius: var(--radius-pill);
  }

  /* Error */
  .err {
    background: rgba(var(--accent-rose-rgb),0.06);
    border: 1px solid rgba(var(--accent-rose-rgb),0.15);
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
  .cat-sup { color: var(--accent-emerald); background: rgba(var(--accent-emerald-rgb),0.1); }
  .cat-inf { color: var(--accent-amber); background: rgba(var(--accent-amber-rgb),0.1); }
  .cat-hyp { color: var(--accent-violet); background: rgba(var(--accent-violet-rgb),0.1); }
  .cat-uns { color: var(--accent-rose); background: rgba(var(--accent-rose-rgb),0.1); }
  .cat-usr { color: var(--accent-blue); background: rgba(var(--accent-blue-rgb),0.1); }

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
  .conf-badge.high { background: rgba(var(--accent-emerald-rgb),0.1); color: var(--accent-emerald); border: 1px solid rgba(var(--accent-emerald-rgb),0.2); }
  .conf-badge.medium { background: rgba(var(--accent-amber-rgb),0.1); color: var(--accent-amber); border: 1px solid rgba(var(--accent-amber-rgb),0.2); }
  .conf-badge.low { background: rgba(var(--accent-rose-rgb),0.1); color: var(--accent-rose); border: 1px solid rgba(var(--accent-rose-rgb),0.2); }
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
    background: rgba(var(--accent-rose-rgb),0.04);
    border-left: 2px solid rgba(var(--accent-rose-rgb),0.3);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  }
  .viol-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent-rose); flex-shrink: 0; margin-top: 5px; }
  .no-viol {
    color: var(--accent-emerald);
    font-size: 12px;
    margin-top: 10px;
    padding: 6px 10px;
    background: rgba(var(--accent-emerald-rgb),0.04);
    border-left: 2px solid rgba(var(--accent-emerald-rgb),0.3);
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
    background: rgba(var(--accent-violet-rgb),0.04);
    border-left: 2px solid rgba(var(--accent-violet-rgb),0.3);
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
  .arb-decision.blk { color: var(--accent-rose); background: rgba(var(--accent-rose-rgb),0.08); }
  .arb-decision.awe { color: var(--accent-amber); background: rgba(var(--accent-amber-rgb),0.08); }
  .arb-decision.auo { color: var(--accent-teal); background: rgba(var(--accent-teal-rgb),0.08); }

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
  .edit-action.del { color: var(--accent-rose); background: rgba(var(--accent-rose-rgb),0.1); }
  .edit-action.rew { color: var(--accent-amber); background: rgba(var(--accent-amber-rgb),0.1); }
  .edit-action.mtu { color: var(--accent-teal); background: rgba(var(--accent-teal-rgb),0.1); }
  .edit-target { color: var(--text-tertiary); font-style: italic; margin-top: 2px; }
  .edit-repl { color: var(--accent-emerald); margin-top: 4px; padding: 4px 8px; background: rgba(var(--accent-emerald-rgb),0.04); border-radius: 4px; }
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
    padding: 8px 24px 20px;
    flex-shrink: 0;
    background: linear-gradient(to top, var(--bg-root) 60%, transparent);
    position: relative;
  }
  .tier-bar {
    display: flex;
    gap: 16px;
    max-width: 860px;
    margin: 0 auto 8px;
    padding: 0 4px;
    align-items: center;
  }
  .tier-group {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tier-label {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    user-select: none;
  }
  .tier-pills {
    display: inline-flex;
    gap: 2px;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 2px;
    flex-shrink: 0;
  }
  .tier-pill {
    font-size: 11px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 6px;
    border: none;
    background: transparent;
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .tier-pill:hover { color: var(--text-secondary); background: var(--bg-surface-2); }
  .tier-pill.active { background: var(--bg-surface-3); color: var(--text-primary); box-shadow: var(--shadow-sm); }
  .tier-pill.active[data-val="strict"] { color: var(--accent-rose); }
  .tier-pill.active[data-val="standard"] { color: var(--accent-amber); }
  .tier-pill.active[data-val="light"] { color: var(--accent-emerald); }
  .fmt-select {
    font-size: 11px;
    font-weight: 500;
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    background: var(--bg-surface-1);
    color: var(--text-secondary);
    cursor: pointer;
    font-family: inherit;
    transition: all var(--transition);
    outline: none;
  }
  .fmt-select:focus { border-color: var(--border-focus); }
  .ibar form {
    display: flex;
    gap: 10px;
    max-width: 860px;
    margin: 0 auto;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 8px 8px 8px 16px;
    align-items: flex-end;
    transition: all var(--transition);
    box-shadow: var(--shadow-md);
  }
  .ibar form:focus-within {
    border-color: var(--border-focus);
    box-shadow: var(--shadow-md), 0 0 0 3px rgba(var(--accent-blue-rgb),0.08);
  }
  .ibar input,
  .ibar textarea#ui {
    flex: 1;
    padding: 8px 0;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 13.5px;
    line-height: 1.45;
    outline: none;
    font-family: inherit;
    resize: none;
    min-height: 44px;
    max-height: 200px;
    field-sizing: content;
  }
  .ibar textarea#ui { padding-top: 10px; }
  @supports not (field-sizing: content) {
    .ibar textarea#ui { min-height: 48px; }
  }
  .ibar input::placeholder,
  .ibar textarea#ui::placeholder { color: var(--text-muted); }
  .ibar form button {
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
  .ibar form button:hover { background: var(--accent-blue-hover); }
  .ibar form button:disabled { opacity: 0.25; cursor: not-allowed; transform: none; }
  .ibar form button svg { width: 16px; height: 16px; }

  /* ---- Stress Test Panel ---- */
  .stress-panel {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(15, 40, 71, 0.97);
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
    box-shadow: 0 0 0 3px rgba(var(--accent-blue-rgb),0.1);
  }
  .stress-run {
    padding: 9px 24px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-hover));
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
  .stress-run:hover { opacity: 0.95; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(var(--accent-blue-rgb),0.25); }
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
  .pss-big.s60 { color: rgba(var(--accent-amber-rgb),0.9); }
  .pss-big.s60::after { background: rgba(var(--accent-amber-rgb),0.9); }
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

  /* ---- Welcome Empty State ---- */
  .welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    text-align: center;
    padding: 40px 24px;
    max-width: 640px;
    margin: 0 auto;
    animation: fadeInWelcome 0.6s ease;
  }
  .welcome-title {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin-bottom: 8px;
  }
  .welcome-sub {
    font-size: 14px;
    color: var(--text-tertiary);
    line-height: 1.6;
    margin-bottom: 28px;
    max-width: 480px;
  }
  .welcome-tiers {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    width: 100%;
    margin-bottom: 32px;
  }
  .welcome-tier-card {
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px 14px;
    text-align: left;
    transition: all var(--transition);
    cursor: pointer;
  }
  .welcome-tier-card:hover { border-color: var(--border-hover); transform: translateY(-2px); }
  .welcome-tier-card h3 {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .welcome-tier-card h3 .tier-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .welcome-tier-card p {
    font-size: 11.5px;
    color: var(--text-tertiary);
    line-height: 1.5;
  }
  .welcome-examples-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 10px;
  }
  .welcome-examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
  }
  .welcome-chip {
    padding: 8px 16px;
    background: var(--bg-surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill);
    color: var(--text-secondary);
    font-size: 12.5px;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
  }
  .welcome-chip:hover {
    background: var(--bg-surface-2);
    border-color: var(--border-hover);
    color: var(--text-primary);
    transform: translateY(-1px);
  }

  /* ---- Collapsible Pipeline Details ---- */
  .pipeline-details {
    width: 100%;
    max-width: 100%;
  }
  .pipeline-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 6px 12px;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    margin: 8px 0;
    align-self: flex-start;
  }
  .pipeline-toggle:hover { color: var(--text-secondary); border-color: var(--border-hover); background: var(--bg-surface-1); }
  .pipeline-toggle svg {
    width: 12px;
    height: 12px;
    transition: transform 0.2s ease;
  }
  .pipeline-toggle.open svg { transform: rotate(90deg); }
  .pipeline-steps {
    display: none;
    flex-direction: column;
    gap: 12px;
    animation: msgIn 0.3s ease;
  }
  .pipeline-steps.open { display: flex; }

  /* ---- Hero Final Output ---- */
  .b.fo {
    align-self: center;
    background: linear-gradient(180deg, var(--bg-surface-1), var(--bg-surface-0));
    border: 1px solid rgba(var(--accent-emerald-rgb),0.15);
    color: var(--text-primary);
    width: 100%;
    max-width: 100%;
    border-radius: var(--radius-lg);
    box-shadow: 0 0 30px rgba(var(--accent-emerald-rgb),0.04);
    font-size: 14px;
    line-height: 1.75;
  }

  /* ---- Formatted Output Typography ---- */
  .fo-content { font-size: 14px; line-height: 1.75; white-space: normal; }
  .fo-section { margin-bottom: 20px; }
  .fo-section:last-child { margin-bottom: 0; }
  .fo-section-hdr {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent-blue);
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(var(--accent-blue-rgb),0.08);
  }
  .fo-subhdr {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent-amber);
    margin: 14px 0 6px;
    letter-spacing: 0.02em;
  }
  .fo-bullet {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin: 6px 0;
    padding-left: 2px;
  }
  .fo-bullet::before {
    content: '';
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
    margin-top: 9px;
  }
  .fo-para { margin: 6px 0; }
  .fo-cite {
    font-size: 10px;
    color: var(--accent-violet);
    font-weight: 600;
    vertical-align: super;
    line-height: 0;
    opacity: 0.8;
  }
  .fo-unknown {
    display: inline-block;
    padding: 1px 8px;
    background: rgba(var(--accent-amber-rgb),0.08);
    border: 1px solid rgba(var(--accent-amber-rgb),0.15);
    border-radius: var(--radius-pill);
    color: var(--accent-amber);
    font-size: 12px;
    font-weight: 500;
  }
  .fo-conf-lvl {
    display: inline-block;
    padding: 2px 10px;
    border-radius: var(--radius-pill);
    font-size: 12px;
    font-weight: 600;
    margin-right: 4px;
  }
  .fo-conf-lvl.high { background: rgba(var(--accent-emerald-rgb),0.1); color: var(--accent-emerald); border: 1px solid rgba(var(--accent-emerald-rgb),0.2); }
  .fo-conf-lvl.medium { background: rgba(var(--accent-amber-rgb),0.1); color: var(--accent-amber); border: 1px solid rgba(var(--accent-amber-rgb),0.2); }
  .fo-conf-lvl.low { background: rgba(var(--accent-rose-rgb),0.1); color: var(--accent-rose); border: 1px solid rgba(var(--accent-rose-rgb),0.2); }
  .fo-tag {
    display: inline-block;
    padding: 0 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    vertical-align: middle;
    margin: 0 2px;
  }
  .fo-tag-ver { background: rgba(var(--accent-emerald-rgb),0.1); color: var(--accent-emerald); }
  .fo-tag-inf { background: rgba(var(--accent-amber-rgb),0.1); color: var(--accent-amber); }
  .fo-tag-unv { background: rgba(var(--accent-rose-rgb),0.1); color: var(--accent-rose); }
  .fo-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
    font-size: 11px;
  }
  .fo-meta-sep { color: var(--text-muted); }

  /* ---- Active Pipeline Stage ---- */
  .top-bar h1 span.stage-active {
    animation: stagePulse 1.5s ease-in-out infinite;
  }
  .top-bar h1 span.stage-done { opacity: 0.4; }

  /* ---- New Chat Button ---- */
  .new-chat-btn {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid transparent;
    padding: 6px 8px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    display: none;
    align-items: center;
  }
  .new-chat-btn.show { display: flex; }
  .new-chat-btn:hover { background: var(--bg-surface-1); color: var(--text-secondary); border-color: var(--border-subtle); }
  .new-chat-btn svg { width: 14px; height: 14px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

  /* ---- Tier description line ---- */
  .tier-desc {
    font-size: 10.5px;
    color: var(--text-muted);
    max-width: 860px;
    margin: -4px auto 6px;
    padding: 0 4px;
    transition: all 0.2s ease;
  }

  /* ---- Search loading dot color ---- */
  .sp.ss .dot-pulse { background: var(--accent-violet); }

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
  @keyframes fadeInWelcome {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes stagePulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
  }
  @keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  }

  /* ---- L2 Verification Summary ---- */
  .l2-toggle { color: var(--accent-blue); }
  .l3-toggle { font-size: 11px; color: var(--text-muted); }
  .l2-content { padding: 12px 0; }
  .l2-row { font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; line-height: 1.5; }
  .l2-label { color: var(--text-tertiary); font-weight: 600; margin-right: 4px; }
  .l2-signals { margin: 4px 0 0 16px; list-style: disc; }
  .l2-signals li { font-size: 11px; color: var(--text-secondary); line-height: 1.6; }

  /* ---- Theme Toggle ---- */
  .theme-toggle { background: none; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 4px 6px; cursor: pointer; color: var(--text-secondary); transition: var(--transition); display: flex; align-items: center; }
  .theme-toggle:hover { border-color: var(--border-hover); color: var(--text-primary); }
  .theme-toggle svg { width: 14px; height: 14px; }
  .theme-icon-sun { display: none; }
  .theme-icon-moon { display: block; }
  [data-theme="light"] .theme-icon-sun { display: block; }
  [data-theme="light"] .theme-icon-moon { display: none; }

  /* ---- Conversation History Sidebar ---- */
  .history-btn { background: none; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 4px 6px; cursor: pointer; color: var(--text-secondary); transition: var(--transition); display: flex; align-items: center; }
  .history-btn:hover { border-color: var(--border-hover); color: var(--text-primary); }
  .history-btn svg { width: 14px; height: 14px; }
  .history-sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: 260px; background: var(--bg-surface-0); border-right: 1px solid var(--border-subtle); z-index: 200; transform: translateX(-100%); transition: transform 0.2s ease; overflow-y: auto; padding: 12px; }
  .history-sidebar.open { transform: translateX(0); }
  .history-sidebar h3 { font-size: 13px; color: var(--text-primary); margin-bottom: 12px; font-weight: 600; }
  .history-item { padding: 8px 10px; border-radius: var(--radius-sm); cursor: pointer; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; transition: var(--transition); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .history-item:hover { background: var(--bg-surface-2); color: var(--text-primary); }
  .history-item.active { background: var(--bg-surface-2); color: var(--accent-blue); }
  .history-item .history-date { display: block; font-size: 10px; color: var(--text-muted); margin-top: 2px; }
  .history-empty { font-size: 11px; color: var(--text-muted); text-align: center; padding: 20px 0; }
  .history-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.3); z-index: 199; display: none; }
  .history-overlay.open { display: block; }

  /* ---- Feedback Buttons ---- */
  .fb-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-subtle); }
  .fb-label { font-size: 11px; color: var(--text-muted); letter-spacing: 0.03em; }
  .fb-btn { background: none; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 3px 8px; cursor: pointer; font-size: 13px; transition: var(--transition); }
  .fb-btn:hover { border-color: var(--border-hover); background: var(--bg-surface-1); }
  .fb-btn:disabled { opacity: 0.5; cursor: default; }
  .fb-btn.fb-active { border-color: var(--accent-emerald); background: rgba(var(--accent-emerald-rgb),0.1); }
  .fb-thanks { font-size: 11px; color: var(--accent-emerald); }

  /* ---- Confidence Reasoning ---- */
  .conf-reasoning { margin-top: 8px; }
  .conf-reasoning summary { font-size: 11px; color: var(--text-tertiary); cursor: pointer; letter-spacing: 0.03em; }
  .conf-reasoning summary:hover { color: var(--text-secondary); }
  .conf-reasoning ul { margin: 6px 0 0 16px; list-style: disc; }
  .conf-reasoning li { font-size: 11px; color: var(--text-secondary); line-height: 1.6; }

  /* ---- Rate Limit Counter ---- */
  .rl-counter { font-size: 10px; color: var(--text-muted); letter-spacing: 0.04em; padding: 2px 6px; border-radius: var(--radius-sm); background: var(--bg-surface-1); }
  .rl-counter:empty { display: none; }
  .rl-warn { color: var(--accent-rose); background: rgba(var(--accent-rose-rgb),0.1); }

  /* ---- Search Note ---- */
  .search-note { font-size: 12px; color: var(--text-tertiary); font-style: italic; padding: 8px 0; }

  /* ---- Responsive ---- */
  .insight-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    top: 52px;
    background: rgba(0,0,0,0.4);
    z-index: 48;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
  }
  .insight-backdrop.open {
    display: block;
    opacity: 1;
    pointer-events: auto;
  }

  @media (max-width: 960px) {
    .insight-rail {
      position: fixed;
      top: 52px;
      right: 0;
      bottom: 0;
      width: min(300px, 92vw);
      max-height: none;
      align-self: stretch;
      transform: translateX(104%);
      opacity: 1;
      pointer-events: none;
      box-shadow: -12px 0 40px rgba(0,0,0,0.35);
      transition: transform 0.25s ease, border-color 0.2s ease;
    }
    .insight-rail.collapsed {
      width: min(300px, 92vw);
      transform: translateX(104%);
      opacity: 1;
      pointer-events: none;
    }
    .insight-rail.mobile-open {
      transform: translateX(0);
      pointer-events: auto;
    }
    .insight-fab {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
  }

  @media (max-width: 640px) {
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    .stress-controls { flex-wrap: wrap; }
    .cfg-drawer { width: 100%; max-width: 100%; }
    .b { max-width: 98%; }
    .top-bar { padding: 0 16px; }
    .chat { padding: 16px; }
    .ibar { padding: 8px 16px 16px; }
    .tier-bar { flex-wrap: wrap; gap: 8px; }
    .welcome-tiers { grid-template-columns: 1fr; }
    .welcome-title { font-size: 22px; }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
  }
</style>
</head>
<body>

<!-- Conversation History Sidebar -->
<div class="history-overlay" id="histOverlay" onclick="toggleHistory()"></div>
<div class="history-sidebar" id="histSidebar" aria-label="Conversation history">
  <h3>History</h3>
  <div id="histList"></div>
</div>

<div class="top-bar">
  <div class="brand-cluster">
    <h1>
      <span class="g1" id="stg1">GPT-1</span><span class="arr">&rarr;</span><span class="g2" id="stg2">GPT-2</span><span class="arr">&rarr;</span><span class="g3" id="stg3">GPT-3</span>
    </h1>
    <p class="brand-tagline">Generate · verify · arbitrate</p>
  </div>
  <div class="right-controls">
    <button class="history-btn" onclick="toggleHistory()" title="Conversation history" aria-label="Conversation history">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>
    </button>
    <span class="rl-counter" id="rl-counter" title="Rate limit remaining" aria-label="Rate limit status"></span>
    <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" title="Toggle light/dark mode" aria-label="Toggle light/dark mode">
      <svg class="theme-icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      <svg class="theme-icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    </button>
    <button class="new-chat-btn" id="newChatBtn" onclick="clearChat()" title="New conversation" aria-label="Start new conversation">
      <svg viewBox="0 0 24 24"><path d="M12 5v14"/><path d="M5 12h14"/></svg>
    </button>
    <button class="cfg-btn stress-btn" onclick="openStress()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      Stress Test
    </button>
    <button class="cfg-btn" onclick="tog()" aria-label="Open settings">
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

<div class="workspace">
<div class="main-stack">

<div class="chat" id="ch" role="log" aria-live="polite" aria-relevant="additions" aria-label="Verification results">
  <div class="welcome" id="welcome">
    <div class="welcome-title">Epistemic Verification Pipeline</div>
    <div class="welcome-sub">
      Every response passes through a 3-stage verification chain:<br>
      <strong style="color:var(--accent-blue)">Generate</strong> &rarr;
      <strong style="color:var(--accent-amber)">Verify</strong> &rarr;
      <strong style="color:var(--accent-violet)">Arbitrate</strong>
    </div>
    <div class="welcome-tiers">
      <div class="welcome-tier-card" onclick="setTier(document.querySelector('[data-val=strict]'));document.getElementById('ui').focus();">
        <h3><span class="tier-dot" style="background:var(--accent-rose)"></span> Strict</h3>
        <p>Full Audit v7 rules. All claims verified, typicality stripped, bare stats blocked. Best for legal, medical, or compliance queries.</p>
      </div>
      <div class="welcome-tier-card" onclick="setTier(document.querySelector('[data-val=standard]'));document.getElementById('ui').focus();">
        <h3><span class="tier-dot" style="background:var(--accent-amber)"></span> Standard</h3>
        <p>Balanced verification. Evidence rules applied, softer thresholds. Good for research questions and general factual queries.</p>
      </div>
      <div class="welcome-tier-card" onclick="setTier(document.querySelector('[data-val=light]'));document.getElementById('ui').focus();">
        <h3><span class="tier-dot" style="background:var(--accent-emerald)"></span> Light</h3>
        <p>Fact-check only. Catches hallucinations but allows natural prose. Best for definitional questions and casual exploration.</p>
      </div>
    </div>
    <div class="welcome-examples-label">Try asking</div>
    <div class="welcome-examples">
      <button class="welcome-chip" onclick="tryExample(this)">What is an LLC?</button>
      <button class="welcome-chip" onclick="tryExample(this)">Is staking income taxable in the US?</button>
      <button class="welcome-chip" onclick="tryExample(this)">Who is the current US president?</button>
      <button class="welcome-chip" onclick="tryExample(this)">What should I do if I get audited?</button>
    </div>
  </div>
</div>

<template id="welcomeTpl">
  <div class="welcome-title">Epistemic Verification Pipeline</div>
  <div class="welcome-sub">
    Every response passes through a 3-stage verification chain:<br>
    <strong style="color:var(--accent-blue)">Generate</strong> &rarr;
    <strong style="color:var(--accent-amber)">Verify</strong> &rarr;
    <strong style="color:var(--accent-violet)">Arbitrate</strong>
  </div>
  <div class="welcome-tiers">
    <div class="welcome-tier-card" onclick="setTier(document.querySelector('[data-val=strict]'));document.getElementById('ui').focus();">
      <h3><span class="tier-dot" style="background:var(--accent-rose)"></span> Strict</h3>
      <p>Full Audit v7 rules. All claims verified, typicality stripped, bare stats blocked. Best for legal, medical, or compliance queries.</p>
    </div>
    <div class="welcome-tier-card" onclick="setTier(document.querySelector('[data-val=standard]'));document.getElementById('ui').focus();">
      <h3><span class="tier-dot" style="background:var(--accent-amber)"></span> Standard</h3>
      <p>Balanced verification. Evidence rules applied, softer thresholds. Good for research questions and general factual queries.</p>
    </div>
    <div class="welcome-tier-card" onclick="setTier(document.querySelector('[data-val=light]'));document.getElementById('ui').focus();">
      <h3><span class="tier-dot" style="background:var(--accent-emerald)"></span> Light</h3>
      <p>Fact-check only. Catches hallucinations but allows natural prose. Best for definitional questions and casual exploration.</p>
    </div>
  </div>
  <div class="welcome-examples-label">Try asking</div>
  <div class="welcome-examples">
    <button class="welcome-chip" onclick="tryExample(this)">What is an LLC?</button>
    <button class="welcome-chip" onclick="tryExample(this)">Is staking income taxable in the US?</button>
    <button class="welcome-chip" onclick="tryExample(this)">Who is the current US president?</button>
    <button class="welcome-chip" onclick="tryExample(this)">What should I do if I get audited?</button>
  </div>
</template>

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
    <select id="st" style="width:110px;">
      <option value="strict">Strict</option>
      <option value="standard">Standard</option>
      <option value="light">Light</option>
    </select>
    <button class="stress-run" id="sr" onclick="runStress()">Run Stress Test</button>
  </div>
  <div class="stress-log" id="sl"></div>
  <div class="stress-score" id="ss"></div>
</div>

<div class="ibar">
  <div class="tier-bar">
    <div class="tier-group">
      <span class="tier-label">Tier</span>
      <div class="tier-pills" id="tier-pills">
        <button class="tier-pill active" data-val="strict" onclick="setTier(this)">Strict</button>
        <button class="tier-pill" data-val="standard" onclick="setTier(this)">Standard</button>
        <button class="tier-pill" data-val="light" onclick="setTier(this)">Light</button>
      </div>
    </div>
    <div class="tier-group">
      <span class="tier-label">Format</span>
      <select class="fmt-select" id="fmt-select">
        <option value="auto">Auto</option>
        <option value="structured">Structured</option>
        <option value="annotated">Annotated</option>
        <option value="concise">Concise</option>
      </select>
    </div>
  </div>
  <div class="tier-desc" id="tier-desc">Full Audit v7 rules &mdash; all claims verified, typicality stripped, bare stats require citations</div>
  <form onsubmit="go(event)" role="search" aria-label="Submit a claim for verification">
    <textarea id="ui" rows="1" placeholder="Ask anything… (⌘/Ctrl+Enter to send)" autocomplete="off" aria-label="Enter text to verify"></textarea>
    <button type="submit" id="sb" aria-label="Submit for verification">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </form>
</div>

</div>

<div class="insight-backdrop" id="insightBackdrop" aria-hidden="true" onclick="closeInsightMobile()"></div>
<aside class="insight-rail" id="insightRail" aria-label="Run insights">
  <div class="insight-rail-head">
    <span>Insights</span>
    <button type="button" class="insight-rail-toggle" id="insightRailToggle" onclick="toggleInsightRail()" aria-expanded="true">Hide panel</button>
  </div>
  <div class="insight-rail-inner">
    <div class="insight-empty" id="insightEmpty">Run a query to see verdict, confidence, and sources here.</div>
    <div class="insight-body hidden" id="insightBody"></div>
  </div>
</aside>
</div>

<button type="button" class="insight-fab" id="insightFab" onclick="openInsightMobile()" aria-label="Open run insights panel">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
  Insights
</button>

<script>
let currentTier = 'strict';

// ---- Insights rail (verdict / confidence / claims / sources) ----
function isNarrowInsight() {
  return typeof window.matchMedia === 'function' && window.matchMedia('(max-width: 960px)').matches;
}
function closeInsightMobile() {
  const rail = document.getElementById('insightRail');
  const bd = document.getElementById('insightBackdrop');
  if (rail) rail.classList.remove('mobile-open');
  if (bd) { bd.classList.remove('open'); bd.setAttribute('aria-hidden', 'true'); }
}
function openInsightMobile() {
  if (!isNarrowInsight()) return;
  const rail = document.getElementById('insightRail');
  const bd = document.getElementById('insightBackdrop');
  if (rail) rail.classList.add('mobile-open');
  if (bd) { bd.classList.add('open'); bd.setAttribute('aria-hidden', 'false'); }
}
function syncInsightToggleLabel() {
  const rail = document.getElementById('insightRail');
  const btn = document.getElementById('insightRailToggle');
  if (!rail || !btn) return;
  if (isNarrowInsight()) {
    var mo = rail.classList.contains('mobile-open');
    btn.textContent = mo ? 'Close' : 'Open';
    btn.setAttribute('aria-expanded', mo ? 'true' : 'false');
  } else {
    var col = rail.classList.contains('collapsed');
    btn.textContent = col ? 'Show panel' : 'Hide panel';
    btn.setAttribute('aria-expanded', col ? 'false' : 'true');
  }
}
/** After each pipeline result: show rail (desktop) or open drawer (mobile). */
function revealInsightsAfterRun() {
  const rail = document.getElementById('insightRail');
  if (!rail) return;
  if (isNarrowInsight()) openInsightMobile();
  else rail.classList.remove('collapsed');
  syncInsightToggleLabel();
}
function toggleInsightRail() {
  const rail = document.getElementById('insightRail');
  const btn = document.getElementById('insightRailToggle');
  if (!rail || !btn) return;
  if (isNarrowInsight()) {
    if (rail.classList.contains('mobile-open')) closeInsightMobile();
    else openInsightMobile();
    syncInsightToggleLabel();
    return;
  }
  rail.classList.toggle('collapsed');
  syncInsightToggleLabel();
  closeInsightMobile();
}
function resetInsightPanel() {
  const empty = document.getElementById('insightEmpty');
  const body = document.getElementById('insightBody');
  if (empty) empty.classList.remove('hidden');
  if (body) { body.classList.add('hidden'); body.textContent = ''; }
}
function updateInsightPanel(d) {
  const empty = document.getElementById('insightEmpty');
  const body = document.getElementById('insightBody');
  if (!empty || !body || !d) return;
  empty.classList.add('hidden');
  body.classList.remove('hidden');
  const pass = d.final_verdict === 'PASS';
  const badgeCls = pass ? 'pass' : 'fail';
  const badgeTxt = pass ? 'Pass' : 'Fail';
  const label = (d.verdict_label || '').trim() || (pass ? 'Verified' : 'Blocked');
  const conf = d.confidence || {};
  const pct = typeof conf.observed_pct === 'number' ? conf.observed_pct : 0;
  const confLbl = (conf.confidence_label || 'Unknown').trim();
  let h = '';
  h += '<div class="insight-verdict-row">';
  h += '<span class="insight-badge ' + badgeCls + '">' + esc(badgeTxt) + '</span>';
  h += '<div class="insight-label">' + esc(label) + '</div>';
  h += '</div>';
  h += '<div class="insight-meter-wrap"><div class="im-label"><span>Observed claims</span><span>' + esc(String(Math.round(pct))) + '%</span></div>';
  h += '<div class="insight-meter"><i style="width:' + Math.min(100, Math.max(0, pct)) + '%"></i></div></div>';
  h += '<div class="insight-kv"><strong>Confidence</strong> · ' + esc(confLbl) + '</div>';
  h += '<div class="insight-kv"><strong>Tier</strong> · ' + esc(String(d.tier || '')) + ' · <strong>Format</strong> · ' + esc(String(d.output_format || '')) + '</div>';
  if (d.search_performed) {
    const n = d.search_sources ? d.search_sources.length : 0;
    h += '<div class="insight-kv"><strong>Web</strong> · ' + n + ' source(s)</div>';
  }
  if (d.arbiter_invoked) {
    h += '<div class="insight-kv"><strong>Arbiter</strong> · ' + esc(String(d.arbiter_decision || '')) + '</div>';
  }
  const claims = (d.rewrite_occurred && d.rewrite_claim_table && d.rewrite_claim_table.length)
    ? d.rewrite_claim_table
    : (d.claim_table || []);
  if (claims.length > 0) {
    h += '<div class="insight-section-title">Claims</div>';
    claims.slice(0, 8).forEach(function(c) {
      const cat = (c.category || '').trim() || '—';
      const tx = (c.claim || '').trim() || '—';
      h += '<div class="insight-claim"><div class="ic-cat">' + esc(cat) + '</div>' + esc(tx) + '</div>';
    });
    if (claims.length > 8) {
      h += '<div class="insight-kv">+' + (claims.length - 8) + ' more in chat trace</div>';
    }
  }
  if (d.search_performed && d.search_sources && d.search_sources.length > 0) {
    h += '<div class="insight-section-title">Sources</div><div class="insight-src-wrap">';
    d.search_sources.slice(0, 6).forEach(function(s) {
      const t = (s.title || 'Source').trim();
      const u = (s.url || '').trim();
      if (u) {
        var hu = encodeURI(u);
        h += '<div class="sr-item"><a href="' + hu + '" target="_blank" rel="noopener noreferrer">' + esc(t) + '</a></div>';
      } else {
        h += '<div class="sr-item">' + esc(t) + '</div>';
      }
    });
    h += '</div>';
  }
  body.innerHTML = h;
  revealInsightsAfterRun();
}

(function initInsightRail() {
  const rail = document.getElementById('insightRail');
  try {
    if (rail && !isNarrowInsight()) rail.classList.remove('collapsed');
  } catch (e) {}
  syncInsightToggleLabel();
})();

function autoResizePrompt() {
  const el = document.getElementById('ui');
  if (!el || el.tagName !== 'TEXTAREA') return;
  el.style.height = 'auto';
  var mh = 200;
  el.style.height = Math.min(el.scrollHeight, mh) + 'px';
}

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
    h += '<div class="history-item' + cls + '" onclick="loadConversation(\\'' + c.id + '\\')" title="' + esc(c.title).replace(/"/g,'&quot;') + '">';
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
  strict: 'Full Audit v7 rules \u2014 all claims verified, typicality stripped, bare stats require citations',
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
  resetInsightPanel();
  closeInsightMobile();
  syncInsightToggleLabel();
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
    el.classList.remove('stage-active', 'stage-done');
    if (i + 1 === n) el.classList.add('stage-active');
    else if (i + 1 < n) el.classList.add('stage-done');
  });
}
function clearStages() {
  ['stg1','stg2','stg3'].forEach(function(id) {
    document.getElementById(id).classList.remove('stage-active', 'stage-done');
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

    var secMatch = trimmed.match(/^(\\d+)[).]\\s+(.+)/);
    if (secMatch) {
      if (inSection) html += '</div>';
      var hdrText = secMatch[2].replace(/\\*\\*/g, '');
      html += '<div class="fo-section"><div class="fo-section-hdr">' + hdrText + '</div>';
      inSection = true;
      continue;
    }

    var cleanLine = trimmed.replace(/\\*\\*/g, '');
    if (/^(Facts|Inferences|Unknowns|Options)\\s*[:\\[]?/i.test(cleanLine) && cleanLine.length < 40) {
      html += '<div class="fo-subhdr">' + cleanLine + '</div>';
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
  return '<div class="sp ' + spinnerCls + '"><span class="dot-pulse"></span><span class="dot-pulse"></span><span class="dot-pulse"></span></div><span class="ld-text">' + text + '</span>';
}

async function go(e) {
  e.preventDefault();
  const inp = document.getElementById('ui');
  const btn = document.getElementById('sb');
  const prompt = inp.value.trim();
  if (!prompt) return;

  hideWelcome();
  document.getElementById('newChatBtn').classList.add('show');
  ab('usr', 'You', esc(prompt));
  inp.value = '';
  btn.disabled = true;

  const ch = document.getElementById('ch');
  const ld = document.createElement('div');
  ld.className = 'ld';
  // Note: makeLoader returns trusted static HTML with no user content
  ld.innerHTML = makeLoader('ss', 'Searching web...');
  ch.appendChild(ld);
  ch.scrollTop = ch.scrollHeight;

  setStage(0);
  const steps = [
    {t: 1500, msg: 'GPT-1 generating...', cls: '', stage: 1},
    {t: 5000, msg: 'GPT-2 verifying...', cls: 's2', stage: 2},
    {t: 10000, msg: 'GPT-3 arbitrating...', cls: 's3', stage: 3},
    {t: 16000, msg: 'Rewriting & re-verifying...', cls: '', stage: 1},
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
        gpt1_system: document.getElementById('g1s').value,
        gpt2_system: document.getElementById('g2s').value.trim(),
        gpt3_system: document.getElementById('g3s').value.trim(),
        tier: currentTier,
        output_format: document.getElementById('fmt-select').value,
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
    updateInsightPanel(d);

    // ---- Build metadata strip ----
    let metaParts = [];
    const tierColors = {strict: 'var(--accent-rose)', standard: 'var(--accent-amber)', light: 'var(--accent-emerald)'};
    const tc = tierColors[d.tier] || 'var(--text-secondary)';
    metaParts.push('<span style="color:' + tc + ';font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">' + esc(d.tier) + '</span>');
    if (d.output_format) metaParts.push('<span style="color:var(--text-tertiary);">' + esc(d.output_format) + '</span>');
    metaParts.push('<span style="color:' + (d.final_verdict === 'PASS' ? 'var(--accent-emerald)' : 'var(--accent-rose)') + ';font-weight:600;">' + (d.final_verdict === 'PASS' ? '\\u2713 PASS' : '\\u2717 FAIL') + '</span>');
    if (d.search_performed) metaParts.push('<span style="color:var(--accent-violet);">web-grounded</span>');
    if (d.arbiter_invoked) metaParts.push('<span style="color:var(--accent-violet);">arbiter: ' + esc(d.arbiter_decision) + '</span>');
    if (d.confidence && d.confidence.confidence_label) metaParts.push('<span>confidence: ' + esc(d.confidence.confidence_label) + '</span>');
    const metaHtml = '<div class="fo-meta">' + metaParts.join('<span class="fo-meta-sep">&middot;</span>') + '</div>';

    // ---- Hero: Final Output with integrated metadata + feedback ----
    const fbHtml = '<div class="fb-row" aria-label="Was this verification correct?">' +
      '<span class="fb-label">Was this helpful?</span>' +
      '<button class="fb-btn fb-up" data-rating="accurate" aria-label="Mark as accurate" title="Accurate">&#x1F44D;</button>' +
      '<button class="fb-btn fb-down" data-rating="inaccurate" aria-label="Mark as inaccurate" title="Inaccurate">&#x1F44E;</button>' +
      '</div>';
    if (d.final_verdict === 'PASS') {
      ab('fo', 'Final Output', formatOutput(d.final_result) + metaHtml + fbHtml);
    } else {
      let blockMsg = 'Output blocked by verification pipeline';
      if (d.arbiter_invoked && d.arbiter_decision === 'BLOCK' && d.arbiter_rationale && d.arbiter_rationale.length > 0) {
        blockMsg += '\\n\\nArbiter rationale:\\n' + d.arbiter_rationale.map(r => '\\u2022 ' + r).join('\\n');
      }
      ab('fo blk', 'Blocked', esc(blockMsg) + metaHtml + fbHtml);
    }
    const fbRows = ch.querySelectorAll('.fb-row');
    const lastFb = fbRows[fbRows.length - 1];
    if (lastFb) {
      lastFb.querySelectorAll('.fb-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
          sendFeedback(btn, btn.getAttribute('data-rating'), d.gpt1_input);
        });
      });
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

    // Build L2 summary
    let l2Html = '<div class="l2-content">';
    // Stages that ran
    let stages = ['GPT-1 (Generator)'];
    if (!d.bypassed) stages.push('GPT-2 (Verifier)');
    if (d.arbiter_invoked) stages.push('GPT-3 (Arbiter)');
    if (d.rewrite_occurred) stages.push('Rewrite + Re-verify');
    l2Html += '<div class="l2-row"><span class="l2-label">Stages:</span> ' + stages.join(' \\u2192 ') + '</div>';
    // Claim count
    const summaryClaims = (d.rewrite_occurred && d.rewrite_claim_table && d.rewrite_claim_table.length)
      ? d.rewrite_claim_table
      : (d.claim_table || []);
    const summaryViolations = d.rewrite_occurred ? (d.rewrite_violations || []) : (d.violations || []);
    if (summaryClaims.length > 0) {
      l2Html += '<div class="l2-row"><span class="l2-label">Claims verified:</span> ' + summaryClaims.length + '</div>';
    }
    // Violations summary in plain English
    if (summaryViolations.length > 0) {
      l2Html += '<div class="l2-row"><span class="l2-label">Issues found:</span> ' + summaryViolations.length + ' violation(s) \\u2014 ' + summaryViolations.map(function(v) { return esc(expandViolation(v)); }).join(', ') + '</div>';
    } else if (!d.bypassed) {
      l2Html += '<div class="l2-row"><span class="l2-label">Issues found:</span> None</div>';
    }
    // Search
    if (d.search_performed) {
      l2Html += '<div class="l2-row"><span class="l2-label">Web sources:</span> ' + (d.search_sources ? d.search_sources.length : 0) + ' sources used for grounding</div>';
    } else if (d.search_attempted && d.search_note) {
      l2Html += '<div class="l2-row"><span class="l2-label">Web search:</span> ' + esc(d.search_note) + '</div>';
    }
    // Confidence reasoning
    if (d.confidence && d.confidence.confidence_reasoning && d.confidence.confidence_reasoning.length > 0) {
      l2Html += '<div class="l2-row"><span class="l2-label">Confidence signals:</span></div>';
      l2Html += '<ul class="l2-signals">';
      d.confidence.confidence_reasoning.forEach(function(r) { l2Html += '<li>' + esc(r) + '</li>'; });
      l2Html += '</ul>';
    }
    l2Html += '</div>';
    l2Div.innerHTML = l2Html;

    // Auto-expand L2 on FAIL
    if (d.final_verdict !== 'PASS') {
      l2Btn.classList.add('open');
      l2Div.classList.add('open');
    }

    // ---- L3: Full Pipeline Trace (collapsed by default) ----
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

    // Build pipeline details inside the collapsible container
    function pab(cls, who, body) {
      const dd = document.createElement('div');
      dd.className = 'b ' + cls;
      // Note: who and body are pre-sanitized via esc() at call sites
      dd.innerHTML = (who ? '<div class="w">' + who + '</div>' : '') + body;
      pdDiv.appendChild(dd);
    }
    function phr() {
      const hr = document.createElement('hr');
      hr.className = 'divider';
      pdDiv.appendChild(hr);
    }

    // ---- Web Search ----
    if (d.search_performed && d.search_sources && d.search_sources.length) {
      pab('sr', 'Web Search (' + d.search_sources.length + ' sources)', renderSearchSources(d.search_sources));
    } else if (d.search_attempted && d.search_note) {
      pab('sr', 'Web Search', '<div class="search-note">' + esc(d.search_note) + '</div>');
    }

    // ---- GPT-1 output ----
    pab('g1', 'GPT-1 (Generator)', esc(d.gpt1_output));

    // ---- Bypass ----
    if (d.bypassed) {
      pab('byp', '', 'Activation phrase detected - verification bypassed');
    } else {
      // ---- GPT-2 results ----
      let g2body = renderClaimTable(d.claim_table) + renderConfidence(d.confidence) + renderViolations(d.violations);
      pab('g2', 'GPT-2 (Verifier) &mdash; ' + d.gpt2_verdict, g2body);

      if (d.gpt2_verdict !== 'PASS') {
        // ---- GPT-3 Arbiter ----
        if (d.arbiter_invoked) {
          let g3body = '';
          const decLower = (d.arbiter_decision || '').toLowerCase().replace(/_/g, '');
          let decCls = 'blk';
          if (decLower === 'allowwithedits') decCls = 'awe';
          if (decLower === 'allowasunknownonly') decCls = 'auo';
          g3body += '<div class="arb-decision ' + decCls + '">' + esc(d.arbiter_decision) + '</div>';
          if (d.arbiter_rationale && d.arbiter_rationale.length > 0) {
            g3body += '<div class="arb-rationale">';
            d.arbiter_rationale.forEach(r => {
              g3body += '<div class="arb-item"><span class="arb-dot"></span>' + esc(r) + '</div>';
            });
            g3body += '</div>';
          }
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
          if (d.arbiter_policy_notes && d.arbiter_policy_notes.length > 0) {
            g3body += '<div class="policy-notes">';
            d.arbiter_policy_notes.forEach(n => {
              g3body += '<div>' + esc(n) + '</div>';
            });
            g3body += '</div>';
          }
          pab('g3', 'GPT-3 (Arbiter)', g3body);
        }

        // ---- Rewrite loop ----
        if (d.rewrite_occurred) {
          phr();
          pab('rw', 'GPT-1 (Rewrite)', esc(d.rewrite_output));
          let rvBody = renderClaimTable(d.rewrite_claim_table) + renderConfidence(d.confidence) + renderViolations(d.rewrite_violations);
          pab('rv', 'GPT-2 (Re-verify) &mdash; ' + d.rewrite_verdict, rvBody);
        }
      }
    }

    ch.scrollTop = ch.scrollHeight;

    // Save to conversation history
    addConversation(prompt, ch.innerHTML);

  } catch(err) {
    timers.forEach(t => clearTimeout(t));
    clearStages();
    ld.remove();
    const msg = err.message || String(err);
    if (msg.includes('Unexpected token') || msg.includes('not valid JSON')) {
      ab('err', '', 'Server returned a non-JSON response (possible timeout or deployment issue). Check deployment logs for details.');
    } else if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
      ab('err', '', 'Network error: could not reach the server. Check your connection.');
    } else {
      ab('err', '', 'Error: ' + esc(msg));
    }
    console.error('Pipeline error:', err);
  } finally {
    btn.disabled = false;
    inp.focus();
    updateRateLimit();
  }
}

lc();
loadTav();
updateRateLimit();
renderHistory();
(function wirePromptField() {
  var el = document.getElementById('ui');
  if (!el) return;
  el.addEventListener('input', autoResizePrompt);
  el.addEventListener('keydown', function(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      go(e);
    }
  });
})();
window.addEventListener('resize', function() {
  if (!isNarrowInsight()) closeInsightMobile();
  syncInsightToggleLabel();
});
autoResizePrompt();
document.getElementById('ui').focus();
</script>
</body>
</html>
"""
