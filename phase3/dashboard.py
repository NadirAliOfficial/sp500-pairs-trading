"""Phase 3 Dashboard. Two things live here:
  1. The client's actual spec: "Include excluded symbols box. Here, we can
     manually submit symbols we don't want the bot to trade for 1 day."
  2. Live operational visibility (added 2026-07-28, porting the proven
     pattern from the sibling SPY bot's dashboard.py/main.py): a status file
     (data/bot_status.json) that main.py's trading loop writes every check
     cycle, read here with a 30s staleness TTL so a crashed/killed bot shows
     as offline quickly instead of freezing on its last-known state. Falls
     back to parsing the log file for a timeline when the status file is
     stale or missing (e.g. no session has run yet today).

Binds to 0.0.0.0 so the client can reach it directly at the VPS's public IP
without an SSH tunnel -- an explicit choice (client wanted it reachable by
URL, no login screen), not an oversight. There is no authentication: anyone
with the URL and port can add/remove blacklist symbols or stop the bot.
"""
import hmac
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from phase2.news_blacklist import ExcludedSymbols
from phase2.market_entry_list import add_pair, load_market_entry_list, remove_pair
from phase2.trading_lists import (add_special_pair, add_to_blacklist,
                                  load_permanent_blacklist, remove_from_blacklist,
                                  remove_special_pair, special_pairs_as_listed)
from phase3 import status

load_dotenv()

app = Flask(__name__)
excluded_symbols = ExcludedSymbols()
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN")

# Every change to a list the trading engine reads gets logged with the client
# IP. Without this there is no way to answer "who removed this pair" -- a pair
# vanished from the Market Entry List once and the only evidence available was
# the file's mtime, which cannot distinguish a dashboard click from a deploy.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(Path(__file__).resolve().parent.parent / "data" / "dashboard.log"),
              logging.StreamHandler(sys.stdout)],
)
audit = logging.getLogger("pairs_bot.dashboard.audit")


def _audit(action: str, detail: str) -> None:
    audit.info(f"{action}: {detail} (from {request.remote_addr})")
STOP_FLAG_PATH = Path(__file__).resolve().parent.parent / "data" / "bot_stop.txt"


@app.before_request
def _require_dashboard_token():
    """Independent audit Finding #10, confirmed real 2026-08-24: every
    list-editing route and /api/stop was reachable by anyone who found the
    VPS's public IP, since 0.0.0.0 binding with no auth at all was the
    original state. The 0.0.0.0 binding itself stays -- the client
    explicitly asked for "reachable by URL, no login screen" -- so this
    gates on a token in the URL instead of a browser login prompt. Once
    DASHBOARD_TOKEN is unset (local dev, tests) this is a no-op, so nothing
    here changes local/test behavior."""
    if not DASHBOARD_TOKEN:
        return
    supplied = request.args.get("key") or request.cookies.get("dash_key")
    # hmac.compare_digest, not != -- independent audit finding: a plain
    # string comparison short-circuits on the first mismatched byte, a
    # theoretical timing side channel for a secret comparison like this one.
    if not supplied or not hmac.compare_digest(supplied, DASHBOARD_TOKEN):
        return jsonify({"error": "unauthorized"}), 401


@app.after_request
def _persist_dashboard_token(response):
    """Bookmark-friendly: visit once with ?key=..., the browser carries the
    cookie on every request after that (including the JS poll and the plain
    HTML forms), so the token only has to be typed/pasted once."""
    key = request.args.get("key")
    if DASHBOARD_TOKEN and key == DASHBOARD_TOKEN:
        response.set_cookie("dash_key", key, httponly=True, samesite="Lax", max_age=60 * 60 * 24 * 365)
    return response

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pairs Bot Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #080b12; --s1: #0d1119; --s2: #111620;
  --border: #1a2438; --border2: #243050;
  --text: #e0eaf8; --muted: #4a6280; --faint: #151d2e;
  --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
  --ease: cubic-bezier(0.2, 0.8, 0.2, 1);
}
html { font-size: 16px; }
body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; }

header {
  background: var(--s1); border-bottom: 1px solid var(--border);
  padding: 0 20px; height: 56px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-icon {
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(135deg, var(--accent), #a78bfa);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 15px; color: #fff;
}
.brand-name { font-size: 15px; font-weight: 700; }
.brand-sub { font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; margin-top: 1px; }

.status-chip {
  display: flex; align-items: center; gap: 7px;
  padding: 6px 14px; border-radius: 20px;
  font-size: 12px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase;
  border: 1px solid transparent;
  transition: background 0.3s var(--ease), border-color 0.3s var(--ease), color 0.3s var(--ease);
}
.status-chip .dot { width: 7px; height: 7px; border-radius: 50%; transition: background 0.3s var(--ease); }
.status-chip.running { background:#22c55e12; color:var(--green); border-color:#22c55e35; }
.status-chip.running .dot { background:var(--green); animation:pulse 2s infinite; }
.status-chip.waiting { background:#f59e0b12; color:var(--yellow); border-color:#f59e0b35; }
.status-chip.waiting .dot { background:var(--yellow); }
.status-chip.stopped { background:#ef444412; color:var(--red); border-color:#ef444435; }
.status-chip.stopped .dot { background:var(--red); }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 #22c55e55} 50%{box-shadow:0 0 0 6px #22c55e00} }

.stop-btn {
  padding: 6px 16px; border-radius: 20px; border: 1px solid #ef444460;
  background: #ef444415; color: #ef4444; font-size: 12px; font-weight: 700;
  letter-spacing: 0.6px; text-transform: uppercase; cursor: pointer; display: none;
  transition: background 0.2s var(--ease), transform 0.15s var(--ease), box-shadow 0.2s var(--ease);
}
.stop-btn:hover { background: #ef444428; transform: translateY(-1px); box-shadow: 0 6px 16px -6px #ef444455; }
.stop-btn:active { transform: translateY(0) scale(0.97); }

.clock { text-align: right; line-height: 1.25; font-family: 'JetBrains Mono', monospace; }
.clock-time { font-size: 16px; font-weight: 700; color: var(--text); letter-spacing: 0.5px; }
.clock-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; }
.updated-ago {
  font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace;
  white-space: nowrap; border-left: 1px solid var(--border); padding-left: 18px;
}

main { max-width: 1100px; margin: 0 auto; padding: 20px; animation: page-in 0.5s var(--ease); }
@keyframes page-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
@media(max-width:900px) { .stats { grid-template-columns: repeat(2, 1fr); } }

.card {
  background: linear-gradient(180deg, var(--s2), var(--s1));
  border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px;
  box-shadow: 0 8px 20px -12px rgba(0,0,0,0.6);
  transition: border-color 0.25s var(--ease), transform 0.25s var(--ease), box-shadow 0.25s var(--ease);
}
.card:hover {
  border-color: var(--border2);
  transform: translateY(-2px);
  box-shadow: 0 16px 32px -14px rgba(0,0,0,0.7);
}
.card-top { display: flex; align-items: flex-start; gap: 12px; }
.card-icon {
  width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.25s var(--ease);
}
.card:hover .card-icon { transform: scale(1.08) rotate(-2deg); }
.card-icon svg { width: 17px; height: 17px; }
.card-icon.blue { background: #3b82f61a; color: var(--accent); }
.card-icon.violet { background: #a78bfa1a; color: #a78bfa; }
.card-icon.amber { background: #f59e0b1a; color: var(--yellow); }
.card-icon.green { background: #22c55e1a; color: var(--green); }
.card-label { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.card-value { font-size: 27px; font-weight: 800; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.5px; transition: color 0.3s var(--ease); }
/* Longer strings (the P&L card with a percentage suffix, "NEWS SCAN DOWN")
   overflowed the card at the default 27px and wrapped onto a second line --
   confirmed live 2026-08-17. Applied via JS only on those longer values, the
   short ones ($ figures, RUNNING/HALTED/Active) keep the full size. */
.card-value.small { font-size: 17px; }
.card-value.pos { color: var(--green); } .card-value.neg { color: var(--red); } .card-value.flat { color: var(--muted); } .card-value.blue { color: var(--accent); }
.card-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }

.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
@media(max-width:900px) { .panels { grid-template-columns: 1fr; } }
.panel {
  background: var(--s1); border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
  display: flex; flex-direction: column; box-shadow: 0 8px 20px -12px rgba(0,0,0,0.6);
  transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease);
}
.panel:hover { border-color: var(--border2); box-shadow: 0 14px 28px -14px rgba(0,0,0,0.7); }
.panel-head { padding: 12px 16px; background: var(--s2); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.panel-title { font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; display: inline-flex; align-items: center; gap: 7px; }
.section h2 { display: inline-flex; align-items: center; gap: 8px; }
.icon { width: 14px; height: 14px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; flex-shrink: 0; }
.icon.lg { width: 17px; height: 17px; }
.panel-badge { font-size: 11px; color: var(--muted); background: var(--faint); border: 1px solid var(--border); border-radius: 20px; padding: 2px 10px; }
.sort-btn {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 600; color: var(--muted);
  background: var(--faint); border: 1px solid var(--border); border-radius: 20px;
  padding: 3px 10px 3px 8px; cursor: pointer; font-family: inherit;
  transition: color 0.2s var(--ease), border-color 0.2s var(--ease), transform 0.15s var(--ease), background 0.2s var(--ease);
}
.sort-btn:hover { color: var(--text); border-color: var(--border2); background: var(--s2); transform: translateY(-1px); }
.sort-btn:active { transform: translateY(0) scale(0.97); }
.sort-btn svg { width: 12px; height: 12px; transition: transform 0.25s var(--ease); }
.sort-btn.desc svg { transform: scaleY(-1); }
.panel-body { overflow-y: auto; max-height: 260px; }
.pos-row { padding: 10px 16px; border-bottom: 1px solid var(--faint); font-size: 13px; transition: background 0.15s var(--ease); }
.pos-row:last-child { border-bottom: none; }
.pos-row:hover { background: var(--faint); }
.empty { padding: 30px 16px; text-align: center; color: var(--muted); font-size: 13px; }

.event-row { display: flex; gap: 10px; align-items: flex-start; padding: 10px 16px; border-bottom: 1px solid var(--faint); font-size: 13px; transition: background 0.15s var(--ease); }
.event-row:hover { background: var(--faint); }
.event-row:last-child { border-bottom: none; }
.event-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.event-dot.session { background: var(--muted); }
.event-dot.news { background: var(--yellow); }
.event-dot.enter { background: var(--accent); }
.event-dot.exit-win { background: var(--green); }
.event-dot.exit-loss { background: var(--red); }
.event-dot.reject { background: var(--red); }
.event-dot.other { background: var(--muted); }
.event-text { flex: 1; }
.event-ts { display: block; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); margin-top: 2px; }

.chips { display: flex; flex-wrap: wrap; gap: 8px; padding-top: 6px; }
.chip { display: inline-flex; align-items: center; gap: 4px; transition: transform 0.15s var(--ease); }
.chip:hover { transform: translateY(-1px); }
.inline-del { display: inline; margin: 0; }
.del {
  background: none; border: none; cursor: pointer; padding: 0 4px;
  color: var(--muted); font-size: 15px; line-height: 1; border-radius: 4px;
  transition: color 0.15s var(--ease), background 0.15s var(--ease), transform 0.15s var(--ease);
}
.del:hover { color: var(--red); background: #ef444418; transform: scale(1.15); }

.section {
  background: var(--s1); border: 1px solid var(--border); border-radius: 14px; margin-bottom: 20px;
  box-shadow: 0 8px 20px -12px rgba(0,0,0,0.6);
  transition: border-color 0.25s var(--ease);
}
.section:hover { border-color: var(--border2); }
.section h2 { font-size: 1.2rem; margin-bottom: 4px; }
.subtitle { color: var(--muted); font-size: 0.9rem; }
.section-summary {
  list-style: none; cursor: pointer; padding: 18px 24px;
  display: flex; justify-content: space-between; align-items: center;
  border-radius: 14px; transition: background 0.2s var(--ease);
}
.section-summary:hover { background: var(--faint); }
.section-summary::-webkit-details-marker { display: none; }
.section-summary .chevron { color: var(--muted); font-size: 12px; transition: transform 0.25s var(--ease); }
details[open] .chevron { transform: rotate(90deg); }
.section-body { padding: 0 24px 24px; }
form { display: flex; gap: 10px; margin-bottom: 8px; }
input[type=text] {
  flex: 1; padding: 10px 14px; font-size: 0.95rem; border: 1px solid var(--border); border-radius: 8px;
  background: var(--s2); color: var(--text); outline: none; transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease);
}
input[type=text]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px #3b82f622; }
button.submit {
  padding: 10px 18px; font-size: 0.95rem; font-weight: 600; color: #fff; background: var(--accent);
  border: none; border-radius: 8px; cursor: pointer;
  transition: transform 0.15s var(--ease), box-shadow 0.2s var(--ease), background 0.2s var(--ease);
}
button.submit:hover { background: #4f8ff7; transform: translateY(-1px); box-shadow: 0 8px 18px -8px #3b82f680; }
button.submit:active { transform: translateY(0) scale(0.97); }
.error { background: #3a1a1a; color: #f87171; padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; margin-bottom: 16px; }
.table-wrap { max-height: 320px; overflow-y: auto; overflow-x: auto; margin-top: 12px; border: 1px solid var(--border); border-radius: 10px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
th { color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; position: sticky; top: 0; background: var(--s1); }
tr { transition: background 0.15s var(--ease); }
tr:hover td { background: var(--faint); }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; background: var(--accent); color: #fff; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; transition: transform 0.15s var(--ease); }
.chip:hover .badge, .badge:hover { transform: scale(1.06); }

/* ---- Mobile ---- */
@media (max-width: 700px) {
  header { height: auto; flex-wrap: wrap; padding: 10px 16px; row-gap: 8px; }
  header > div:last-child { flex-wrap: wrap; justify-content: flex-end; row-gap: 8px; gap: 10px; }
  .clock { text-align: left; }
  .updated-ago { border-left: none; padding-left: 0; }
}
@media (max-width: 520px) {
  main { padding: 12px; }
  .stats { grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
  .panels { gap: 8px; margin-bottom: 12px; }
  .card { padding: 14px; }
  .card-value { font-size: 21px; }
  .card-label { font-size: 10px; margin-bottom: 6px; }
  .section { margin-bottom: 12px; }
  .section-summary { padding: 14px 16px; }
  .section-body { padding: 0 16px 16px; }
  .panel-head { padding: 10px 12px; }
  .event-row, .pos-row { padding: 10px 12px; }
  form { flex-direction: column; }
  button.submit { width: 100%; }
  .brand-name { font-size: 14px; }
  .brand-sub { font-size: 10px; }
}
@media (max-width: 380px) {
  .stats { grid-template-columns: 1fr; }
  .updated-ago { display: none; }
}
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-icon">
      <svg viewBox="0 0 24 24" width="18" height="18" stroke="#fff" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="2,14 8,14 10,8 14,18 16,10 22,10"/></svg>
    </div>
    <div>
      <div class="brand-name">Pairs Bot Dashboard</div>
      <div class="brand-sub" id="acc">Connecting...</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:18px;">
    <div class="clock">
      <div class="clock-time" id="et-clock">--:--:--</div>
      <div class="clock-label">Eastern Time</div>
    </div>
    <div class="updated-ago" id="updated-ago">Connecting&hellip;</div>
    <div class="status-chip stopped" id="chip"><span class="dot"></span><span id="chip-txt">Offline</span></div>
    <button class="stop-btn" id="stop-btn" onclick="stopBot()">STOP BOT</button>
  </div>
</header>

<main>
  <div class="stats">
    <div class="card">
      <div class="card-top">
        <div class="card-icon blue"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="20" x2="6" y2="12"/><line x1="12" y1="20" x2="12" y2="7"/><line x1="18" y1="20" x2="18" y2="4"/></svg></div>
        <div>
          <div class="card-label">Equity (ELV)</div>
          <div class="card-value blue" id="elv-val">--</div>
        </div>
      </div>
      <div class="card-sub">Frozen at 9:29am ET</div>
    </div>
    <div class="card">
      <div class="card-top">
        <div class="card-icon violet"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8 8V6a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="3" y1="13" x2="21" y2="13"/></svg></div>
        <div>
          <div class="card-label">Open Positions</div>
          <div class="card-value flat" id="pos-count">--</div>
        </div>
      </div>
      <div class="card-sub" id="pos-sub">of &mdash; tracked pairs</div>
    </div>
    <div class="card">
      <div class="card-top">
        <div class="card-icon amber"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3,17 9,11 13,15 21,6"/><polyline points="15,6 21,6 21,12"/></svg></div>
        <div>
          <div class="card-label">Today's P&amp;L</div>
          <div class="card-value flat" id="pnl-val">--</div>
        </div>
      </div>
      <div class="card-sub">Realized, closed positions only</div>
    </div>
    <div class="card">
      <div class="card-top">
        <div class="card-icon green"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v6c0 5-3.5 8-7 9-3.5-1-7-4-7-9V6l7-3z"/></svg></div>
        <div>
          <div class="card-label">Trading Status</div>
          <div class="card-value flat" id="halt-val">--</div>
        </div>
      </div>
      <div class="card-sub" id="halt-sub">Daily loss kill-switch</div>
    </div>
  </div>

  <details class="section"{% if error %} open{% endif %}>
    <summary class="section-summary">
      <div>
        <h2><svg class="icon lg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><line x1="5.5" y1="5.5" x2="18.5" y2="18.5"/></svg>Excluded Symbols</h2>
        <p class="subtitle">Manually block a symbol from being traded for 1 day.</p>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="panel-badge" id="excluded-count">{{ entries|length }} excluded</span>
        <span class="chevron">&#9656;</span>
      </div>
    </summary>
    <div class="section-body">
      {% if error %}<div class="error">{{ error }}</div>{% endif %}
      <form method="post" action="{{ url_for('exclude_symbol') }}">
        <input type="text" name="symbol" placeholder="e.g. AAPL" required autofocus>
        <button class="submit" type="submit">Exclude</button>
      </form>
      <div id="excluded-body">
      {% if entries %}
      <div class="table-wrap">
        <table>
          <tr><th>Symbol</th><th>Excluded Until (UTC)</th><th></th></tr>
          {% for symbol, expires_at in entries %}
          <tr><td><span class="badge">{{ symbol }}</span></td><td>{{ expires_at }}</td>
            <td style="text-align:right"><form method="post" action="{{ url_for('delete_excluded_symbol') }}" class="inline-del" onsubmit="return confirm('Stop excluding {{ symbol }}?')">
              <input type="hidden" name="symbol" value="{{ symbol }}">
              <button class="del" type="submit" title="Remove">&times;</button>
            </form></td></tr>
          {% endfor %}
        </table>
      </div>
      {% else %}
      <p class="empty" style="text-align:left;padding:12px 0;">No symbols currently excluded.</p>
      {% endif %}
      </div>
    </div>
  </details>

  <details class="section">
    <summary class="section-summary">
      <div>
        <h2><svg class="icon lg" viewBox="0 0 24 24"><circle cx="4" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="4" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="4" cy="18" r="1.3" fill="currentColor" stroke="none"/><line x1="9" y1="6" x2="21" y2="6"/><line x1="9" y1="12" x2="21" y2="12"/><line x1="9" y1="18" x2="21" y2="18"/></svg>Market Entry List</h2>
        <p class="subtitle">Universe pairs + Market Entry List will be traded. They must still pass screening.</p>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="panel-badge">{{ market_entry|length }} pairs</span>
        <span class="chevron">&#9656;</span>
      </div>
    </summary>
    <div class="section-body">
      <form method="post" action="{{ url_for('add_market_entry_pair') }}">
        <input type="text" name="symbol_a" placeholder="e.g. GOOG" required>
        <input type="text" name="symbol_b" placeholder="e.g. META" required>
        <button class="submit" type="submit">Add</button>
      </form>
      {% if market_entry %}
      <div class="chips">
        {% for pair in market_entry %}
        <span class="chip"><span class="badge">{{ pair }}</span>
          <form method="post" action="{{ url_for('delete_market_entry_pair') }}" class="inline-del" onsubmit="return confirm('Remove {{ pair }} from the Market Entry List? Universe screening may still trade it if it qualifies on its own.')">
            <input type="hidden" name="symbol_a" value="{{ pair.split('/')[0] }}">
            <input type="hidden" name="symbol_b" value="{{ pair.split('/')[1] }}">
            <button class="del" type="submit" title="Remove">&times;</button>
          </form></span>
        {% endfor %}
      </div>
      {% else %}
      <p class="empty" style="text-align:left;padding:12px 0;">
        List is empty. Universe screening still runs on its own and can trade any pair that qualifies.</p>
      {% endif %}
    </div>
  </details>

  <details class="section">
    <summary class="section-summary">
      <div>
        <h2><svg class="icon lg" viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>Permanent Blacklist</h2>
        <p class="subtitle">Symbols the bot will never trade. No expiry.</p>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="panel-badge">{{ blacklist|length }} symbols</span>
        <span class="chevron">&#9656;</span>
      </div>
    </summary>
    <div class="section-body">
      <form method="post" action="{{ url_for('add_blacklist_symbol') }}">
        <input type="text" name="symbol" placeholder="e.g. NVDA" required>
        <button class="submit" type="submit">Add</button>
      </form>
      {% if blacklist %}
      <div class="chips">
        {% for symbol in blacklist %}
        <span class="chip"><span class="badge">{{ symbol }}</span>
          <form method="post" action="{{ url_for('delete_blacklist_symbol') }}" class="inline-del" onsubmit="return confirm('Remove {{ symbol }} from the permanent blacklist? It will be allowed to trade again.')">
            <input type="hidden" name="symbol" value="{{ symbol }}">
            <button class="del" type="submit" title="Remove">&times;</button>
          </form></span>
        {% endfor %}
      </div>
      {% else %}
      <p class="empty" style="text-align:left;padding:12px 0;">Blacklist is empty.</p>
      {% endif %}
    </div>
  </details>

  <details class="section">
    <summary class="section-summary">
      <div>
        <h2><svg class="icon lg" viewBox="0 0 24 24"><path d="M12 3l2.6 5.9 6.4.6-4.8 4.3 1.4 6.2L12 16.8 6.4 20l1.4-6.2-4.8-4.3 6.4-.6L12 3z"/></svg>Special Pairs List</h2>
        <p class="subtitle">Exempt from market cap and volume filters only.</p>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="panel-badge">{{ special_pairs|length }} pairs</span>
        <span class="chevron">&#9656;</span>
      </div>
    </summary>
    <div class="section-body">
      <form method="post" action="{{ url_for('add_special_pair_route') }}">
        <input type="text" name="symbol_a" placeholder="e.g. PEP" required>
        <input type="text" name="symbol_b" placeholder="e.g. KO" required>
        <button class="submit" type="submit">Add</button>
      </form>
      {% if special_pairs %}
      <div class="chips">
        {% for pair in special_pairs %}
        <span class="chip"><span class="badge">{{ pair }}</span>
          <form method="post" action="{{ url_for('delete_special_pair') }}" class="inline-del" onsubmit="return confirm('Remove {{ pair }} from the special pairs list?')">
            <input type="hidden" name="symbol_a" value="{{ pair.split('/')[0] }}">
            <input type="hidden" name="symbol_b" value="{{ pair.split('/')[1] }}">
            <button class="del" type="submit" title="Remove">&times;</button>
          </form></span>
        {% endfor %}
      </div>
      {% else %}
      <p class="empty" style="text-align:left;padding:12px 0;">No special pairs.</p>
      {% endif %}
      <p class="empty" style="text-align:left;padding:10px 0 0;font-size:0.8rem;">
        These pairs skip the market cap, share volume and dollar volume checks. They must still
        pass correlation, cointegration, half-life and Hurst.</p>
    </div>
  </details>

  <div class="panels">
    <div class="panel">
      <div class="panel-head"><span class="panel-title"><svg class="icon" viewBox="0 0 24 24"><rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8 8V6a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="3" y1="13" x2="21" y2="13"/></svg>Open Positions</span><span class="panel-badge" id="pc">0</span></div>
      <div class="panel-body" id="positions-panel"><div class="empty">No open positions</div></div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title"><svg class="icon" viewBox="0 0 24 24"><path d="M6 10a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6"/><path d="M10 20a2 2 0 0 0 4 0"/></svg>Recent Events</span><span class="panel-badge" id="ec">0</span></div>
      <div class="panel-body" id="events-panel"><div class="empty">No session has run yet</div></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title"><svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.8" fill="currentColor" stroke="none"/></svg>Divergence Watch</span>
        <div style="display:flex;align-items:center;gap:8px;">
          <button class="sort-btn" id="divergence-sort-btn" onclick="toggleDivergenceSort()" title="Toggle sort order">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="6" x2="14" y2="6"/><line x1="4" y1="12" x2="11" y2="12"/><line x1="4" y1="18" x2="8" y2="18"/><polyline points="17,15 20,18 20,4"/></svg>
            <span id="divergence-sort-label">Closest first</span>
          </button>
          <span class="panel-badge" id="dc">0</span>
        </div>
      </div>
      <p class="subtitle" id="qp-loaded-sub" style="padding:0 16px 8px;"></p>
      <div class="panel-body" id="divergence-panel"><div class="empty">No data yet</div></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title"><svg class="icon" viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>Live Johansen/Hurst</span>
        <span class="panel-badge" id="lv-count">0</span>
      </div>
      <div class="panel-body" id="live-validation-panel">
        <div class="empty">No data yet</div>
      </div>
      <p class="subtitle" id="lv-subtitle" style="padding:0 16px 12px;">
        Raw live readings for every tracked pair, not only ones that qualified for Divergence Watch above.
        Needs Johansen Unconfirmed and Hurst above 0.50, each sticky once cleared this session.
      </p>
    </div>
  </div>
</main>

<script>
const REASON_LABELS = {
  take_profit: 'take profit', stop_loss: 'stop loss',
  hard_stop: 'hard stop', timed_exit: 'timed exit',
  max_holding_time: 'max holding time', eod: 'end of day',
  news_stop: 'news stop', bracket_abandoned: 'bracket abandoned',
};

function parsePyList(str) {
  try { return JSON.parse(str.replace(/'/g, '"')); } catch (e) { return []; }
}

function money(n) {
  const v = parseFloat(n);
  return (v >= 0 ? '+' : '-') + '$' + Math.abs(v).toFixed(2);
}

// Turns a raw log message into a plain-English line + a color tag for the
// dot next to it. Falls through to showing the raw message untouched for
// anything unrecognized, so nothing silently disappears from the feed.
function formatEvent(msg) {
  let m;
  if (msg.startsWith('=== Session start')) {
    return { text: 'Trading session started', tag: 'session' };
  }
  if (msg.startsWith('=== 3:59pm flatten')) {
    return { text: 'End of day: flattening all open positions', tag: 'session' };
  }
  if (msg.includes('market holiday')) {
    return { text: 'Market holiday, no trading today', tag: 'session' };
  }
  if (msg.includes('early close day')) {
    return { text: 'Early close day, no trading today', tag: 'session' };
  }
  if (m = msg.match(/^Blacklisted today: (\[.*\])$/)) {
    const symbols = parsePyList(m[1]);
    return { text: symbols.length ? `Pre-market news blacklist: ${symbols.join(', ')}` : 'Pre-market news scan: nothing blacklisted', tag: 'news' };
  }
  if (m = msg.match(/^Intraday news scan blacklisted: (\[.*\])$/)) {
    const symbols = parsePyList(m[1]);
    return { text: `Intraday news blacklist update: ${symbols.join(', ')}`, tag: 'news' };
  }
  // Trailing "(bracket TP orderId=.. SL orderId=..)" is optional in the
  // match since older log lines (before native bracket orders) don't have
  // it -- without the optional group this stopped matching entirely once
  // ENTER started appending it, and every entry fell through to the raw
  // fallback below instead of the plain-English line.
  if (m = msg.match(/^ENTER (\S+)\/(\S+): (\S+) ([\d.]+) \/ (\S+) ([\d.]+)(?: \(bracket.*\))?$/)) {
    return { text: `Opened ${m[1]}/${m[2]} — ${m[3]} ${Number(m[4]).toLocaleString()} / ${m[5]} ${Number(m[6]).toLocaleString()}`, tag: 'enter' };
  }
  if (m = msg.match(/^(\S+)\/(\S+) entry rejected by IBKR, not tracking a position -- (.+)$/)) {
    return { text: `Entry rejected: ${m[1]}/${m[2]} — ${m[3]}`, tag: 'reject' };
  }
  if (m = msg.match(/^(\S+)\/(\S+): entry order \d+ was cancelled without ever filling -- clearing the phantom position, no trade actually happened$/)) {
    return { text: `${m[1]}/${m[2]} — order was cancelled before it filled, no real position was opened`, tag: 'reject' };
  }
  if (msg.includes('Intraday news scan failed')) {
    return { text: 'Intraday news scan failed this cycle, will retry next cycle', tag: 'news' };
  }
  if (m = msg.match(/^IBKR error (\d+) \(reqId=[-\d]+\): (.+)$/)) {
    return { text: `IBKR error ${m[1]}: ${m[2]}`, tag: 'reject' };
  }
  // Confirmed real 2026-08-25 audit: several EXIT log lines carry a
  // parenthetical detail before the comma ("hard_stop (native bracket)",
  // "stop_loss (double touch)") that this regex never accounted for even
  // before tonight -- those fell through to raw, unformatted text in the
  // event feed. The (?:...)? group makes that suffix optional so both
  // forms match, without changing what gets logged to the actual log file.
  if (m = msg.match(/^EXIT (\S+)\/(\S+): (\w+)(?: \([^)]+\))?, pnl=(-?[\d.]+)$/)) {
    const pnl = parseFloat(m[4]);
    const reason = REASON_LABELS[m[3]] || m[3];
    return { text: `Closed ${m[1]}/${m[2]} — ${reason} (${money(pnl)})`, tag: pnl >= 0 ? 'exit-win' : 'exit-loss' };
  }
  if (m = msg.match(/^BETA BREAKDOWN (\S+)\/(\S+): hedge relationship invalidated, pnl=(-?[\d.]+)$/)) {
    const pnl = parseFloat(m[3]);
    return { text: `Beta breakdown ${m[1]}/${m[2]} — hedge relationship invalidated (${money(pnl)})`, tag: pnl >= 0 ? 'exit-win' : 'exit-loss' };
  }
  if (msg.includes('no warm-up data available')) {
    return { text: 'Morning warm-up data unavailable — pairs inactive until live bars build up enough history', tag: 'exit-loss' };
  }
  if (m = msg.match(/^Recovered mean divergence level for (\S+)\/(\S+) from live bars \(warm-up had failed\): ([\d.]+)$/)) {
    return { text: `${m[1]}/${m[2]} back online — entry trigger recovered from live data (${Number(m[3]).toFixed(2)})`, tag: 'exit-win' };
  }
  if (m = msg.match(/^INTERNAL CIRCUIT BREAKER raised.*internal circuit breaker: (\w+), SPY ([-+.\d]+)%/)) {
    return { text: `Circuit breaker: ${m[1]} halt, SPY ${m[2]}%`, tag: 'exit-loss' };
  }
  return { text: msg, tag: 'other' };
}

let lastUpdated = null;
let divergenceSortAsc = true;

function toggleDivergenceSort() {
  divergenceSortAsc = !divergenceSortAsc;
  const btn = document.getElementById('divergence-sort-btn');
  btn.classList.toggle('desc', !divergenceSortAsc);
  document.getElementById('divergence-sort-label').textContent = divergenceSortAsc ? 'Closest first' : 'Farthest first';
  refresh();  // re-render immediately with the new order, don't wait for the next poll
}

function tickClock() {
  const now = new Date();
  document.getElementById('et-clock').textContent = now.toLocaleTimeString('en-US', {
    timeZone: 'America/New_York', hour12: false,
  });
}
tickClock();
setInterval(tickClock, 1000);

function tickUpdatedAgo() {
  const el = document.getElementById('updated-ago');
  if (lastUpdated === null) { el.textContent = 'Connecting…'; return; }
  const secs = Math.floor((Date.now() - lastUpdated) / 1000);
  el.textContent = secs <= 1 ? 'Updated just now' : `Updated ${secs}s ago`;
}
setInterval(tickUpdatedAgo, 1000);

async function refresh() {
  try {
    const d = await fetch('/api/state').then(r => r.json());
    lastUpdated = Date.now();
    tickUpdatedAgo();

    // IBKR paper accounts always start with "DU"; live accounts don't.
    // This used to hardcode "(Paper)" unconditionally, a leftover from when
    // the bot only ever ran on paper -- fixed 2026-08-26 for the live cutover.
    document.getElementById('acc').textContent = d.account
      ? d.account + (d.account.startsWith('DU') ? ' (Paper)' : ' (Live)')
      : 'Not connected';

    const chip = document.getElementById('chip');
    const labels = {running:'Running', waiting:'Waiting', holiday:'Holiday', ended:'Session Ended', offline:'Offline'};
    const classes = {running:'running', waiting:'waiting', holiday:'waiting', ended:'waiting', offline:'stopped'};
    chip.className = 'status-chip ' + (classes[d.status] || 'stopped');
    document.getElementById('chip-txt').textContent = labels[d.status] || d.status;
    document.getElementById('stop-btn').style.display = d.status === 'running' ? 'block' : 'none';

    document.getElementById('elv-val').textContent = d.elv ? '$' + Number(d.elv).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : '--';

    // With no live session there is no position/kill-switch state to report.
    // Showing "0 open" and a green "Active" while the bot is Offline reads as
    // "running and healthy", which is the opposite of what's true.
    const live = d.status !== 'offline';

    const positions = d.open_positions || [];
    document.getElementById('pos-count').textContent = live ? positions.length : '--';
    // Same source array as Divergence Watch below (d.pair_divergence), not
    // d.qualified_pairs_count -- per the client 2026-08-07, the two cards
    // showing different numbers read as contradictory even when both were
    // technically correct ("Information should be uniform," "both numbers
    // should be equal"). d.qualified_pairs_count includes pairs still
    // warming up with no live trigger yet, which isn't a number this card
    // needs; reading off the exact same array Divergence Watch renders from
    // guarantees the two always match, by construction, not by coincidence.
    document.getElementById('pos-sub').textContent = 'of ' + (live ? (d.pair_divergence || []).length : '--') + ' tracked pairs';

    const pnlEl = document.getElementById('pnl-val');
    const pnl = d.daily_pnl;
    if (pnl !== null && pnl !== undefined) {
      const pnlPctTxt = (d.daily_pnl_pct !== null && d.daily_pnl_pct !== undefined)
        ? ` (${d.daily_pnl_pct >= 0 ? '+' : ''}${d.daily_pnl_pct.toFixed(2)}%)` : '';
      pnlEl.textContent = (pnl >= 0 ? '+' : '-') + '$' + Math.abs(pnl).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) + pnlPctTxt;
      pnlEl.className = 'card-value ' + (pnlPctTxt ? 'small ' : '') + (pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : 'flat');
    } else { pnlEl.textContent = '--'; pnlEl.className = 'card-value flat'; }

    const haltEl = document.getElementById('halt-val');
    const haltSub = document.getElementById('halt-sub');
    if (!live) {
      haltEl.textContent = '--';
      haltEl.className = 'card-value flat';
      haltSub.textContent = 'No session running';
    } else if (d.news_scan_healthy === false) {
      // Per the client 2026-08-17: "If Gemini news API is down, no
      // trading." -- new entries paused, but this is not the same as the
      // daily loss kill-switch (trading_halted), so it gets its own label
      // rather than being folded into "HALTED".
      haltEl.textContent = 'NEWS SCAN DOWN';
      haltEl.className = 'card-value small neg';
      haltSub.textContent = 'New entries paused until a scan succeeds';
    } else {
      haltEl.textContent = d.trading_halted ? 'HALTED' : 'Active';
      haltEl.className = 'card-value ' + (d.trading_halted ? 'neg' : 'pos');
      haltSub.textContent = 'Daily loss kill-switch';
    }

    const pp = document.getElementById('positions-panel');
    document.getElementById('pc').textContent = live ? positions.length : '--';
    pp.innerHTML = positions.length
      // Unrealized P&L removed on the client's request 2026-08-04: shown
      // beside a realized-only P&L card, the two numbers read as though they
      // should reconcile and they never do. Still in bot_status.json if it
      // is ever wanted back.
      ? positions.map(p => {
          // Single-leg "Expansion Leg" position (2026-09-03 pivot): only
          // ONE symbol is real money, and action_a is repurposed to
          // describe whichever one that is (p.traded_leg), not always
          // symbol_a. The old buy/sell-pair split below assumed action_a
          // always meant symbol_a specifically -- for a traded_leg="B"
          // position that showed the real BUY as a phantom SELL of
          // symbol_b at qty 0, backwards from what actually happened.
          const isSingleLeg = p.qty_a === 0 || p.qty_b === 0;
          let buyLeg, sellLeg;
          if (isSingleLeg) {
            const tradedSymbol = p.traded_leg === 'A' ? p.symbol_a : p.symbol_b;
            const tradedQty = p.traded_leg === 'A' ? p.qty_a : p.qty_b;
            const label = `${tradedSymbol} x${tradedQty}`;
            buyLeg = p.action_a === 'BUY' ? label : '(none)';
            sellLeg = p.action_a === 'SELL' ? label : '(none)';
          } else {
            buyLeg = p.action_a === 'BUY' ? `${p.symbol_a} x${p.qty_a}` : `${p.symbol_b} x${p.qty_b}`;
            sellLeg = p.action_a === 'SELL' ? `${p.symbol_a} x${p.qty_a}` : `${p.symbol_b} x${p.qty_b}`;
          }
          // Leg Size, Pair Entry Price, Position Size, and Pair Entry Initial
          // Margin added 2026-08-15/17 per the client. Position Size fixed
          // 2026-08-17 per his own worked example: "Leg Size: 15 MPWR, 91
          // MRVL. Position Size: (15+91)/2 = 53 Pairs. It is 53 Pairs. Not 1
          // Pair" -- Leg Size is qty_a/qty_b directly (not a reduced ratio),
          // Position Size is their average, not a gcd-based unit count.
          const pnlPct = p.unrealized_pnl_pct;
          const pnlClass = pnlPct > 0 ? 'pos' : pnlPct < 0 ? 'neg' : 'flat';
          const pnlSign = pnlPct >= 0 ? '+' : '';
          return `<div class="pos-row"><span class="badge">${p.symbol_a}/${p.symbol_b}</span> ${p.action_a} @ ${p.entry_spread.toFixed(2)} &middot; buy ${buyLeg} / sell ${sellLeg}<br>`
            + `Leg Size: ${p.qty_a} ${p.symbol_a} : ${p.qty_b} ${p.symbol_b} &middot; Position Size: ${p.pair_units} pairs &middot; `
            + `Entry Price: ${p.symbol_a} $${p.entry_price_a.toFixed(2)} / ${p.symbol_b} $${p.entry_price_b.toFixed(2)}<br>`
            + `Entry Initial Margin: $${p.entry_margin.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`
            + (p.entry_margin_ibkr_probed != null
                ? ` &middot; IBKR confirmed: $${p.entry_margin_ibkr_probed.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})} for ${p.entry_margin_ibkr_probed_qty} shares`
                : '') + `<br>`
            + `Points to trigger @ Entry = ${p.points_to_trigger_at_entry.toFixed(2)} &middot; Points past trigger @ Take Profit = ${p.take_profit_trigger_level.toFixed(2)}<br>`
            + `PnL: <span class="${pnlClass}">${pnlSign}${pnlPct.toFixed(2)}%</span> ($${p.unrealized_pnl.toFixed(2)}) &middot; opened ${new Date(p.entry_time).toLocaleTimeString('en-US', {hour12: false, timeZone: 'America/New_York'})} ET</div>`;
        }).join('')
      : `<div class="empty">${live ? 'No open positions' : 'Bot is offline'}</div>`;

    // Sorted by distance = trigger - current; ascending (default) puts the
    // pair nearest an actual entry -- or already past its trigger and just
    // waiting on another condition, which goes negative and sorts first --
    // at the top. divergenceSortAsc is flipped by the panel's sort button.
    // Per the client 2026-08-26's reference mockup: only pairs with a real,
    // established trigger appear here at all (see main.py's write_status).
    // distance can be null now (2026-09-13: a pair loaded from the 9am
    // cache shows here before live data arrives) -- treated as -Infinity
    // so it always sorts to the very top regardless of sort direction,
    // same "surface it prominently" treatment already-past-trigger rows get.
    const divergence = (d.pair_divergence || []).slice()
      .sort((a, b) => {
        const da = a.distance ?? -Infinity, db = b.distance ?? -Infinity;
        return divergenceSortAsc ? da - db : db - da;
      });
    const dp = document.getElementById('divergence-panel');
    document.getElementById('dc').textContent = live ? divergence.length : '--';
    // Per the client 2026-09-02: "But we are not seeing on the
    // Dashboard" -- d.qualified_pairs_count (the pairs loaded, either
    // from the 9am cache or the 9:30am live selection) was written to
    // the status file but never actually shown anywhere on the page.
    // Divergence Watch's own count only counts pairs that ALREADY
    // cleared Johansen/Hurst, so it stays 0 well after the real list has
    // loaded -- this line is what actually confirms the load happened.
    const qpSub = document.getElementById('qp-loaded-sub');
    if (live && d.qualified_pairs_count) {
      qpSub.textContent = `${d.qualified_pairs_count} pairs loaded`;
    } else if (d.cached_pairs_for_tomorrow_count != null) {
      // per the client repeatedly asking "how many pairs will load by 9am" --
      // while the bot itself is offline (the whole rest of this panel is
      // empty), this is the one place still worth showing something: the
      // count from the overnight scan (see main.py's
      // write_cached_pairs_status), which is ready long before the live
      // session actually starts and loads it.
      const builtAt = d.cached_pairs_for_tomorrow_at
        ? new Date(d.cached_pairs_for_tomorrow_at).toLocaleTimeString('en-US', {hour12: false, timeZone: 'America/New_York'}) + ' ET'
        : '';
      qpSub.textContent = `${d.cached_pairs_for_tomorrow_count} pairs cached for next load` + (builtAt ? ` (built ${builtAt})` : '');
    } else {
      qpSub.textContent = '';
    }
    dp.innerHTML = divergence.length
      // distance = trigger - current: positive means still short of triggering,
      // negative means the pair is already past its own trigger level (and
      // just not entering yet for some other reason -- gap condition, book
      // not flat, blacklist, time window). current/distance null (2026-09-13):
      // loaded from the 9am cache, live data hasn't arrived yet -- shows
      // trigger and cleared status only, per the client: "we want to see
      // the pairs. The pairs should be visible by 9am. Even if live data
      // is yet to come on."
      ? divergence.map(p => {
          const clearedText = p.cleared ? 'cleared' : 'not cleared';
          const status = p.current === null
            ? `$${p.trigger.toFixed(2)} trigger &middot; waiting for live data &middot; ${clearedText}`
            : `$${p.current.toFixed(2)} of $${p.trigger.toFixed(2)} trigger &middot; ${p.distance >= 0 ? '$' + p.distance.toFixed(2) + ' to go' : '$' + Math.abs(p.distance).toFixed(2) + ' past trigger'} &middot; ${clearedText}`;
          return `<div class="pos-row"><span class="badge">${p.symbol_a}/${p.symbol_b}</span> ${status}</div>`;
        }).join('')
      : `<div class="empty">${live ? 'No data yet' : 'Bot is offline'}</div>`;

    // Per the client 2026-09-01: "We need to see the data on the
    // Dashboard and be able to make decisions on the 0.60." Raw live
    // readings for every tracked pair, not just ones that already
    // qualified for Divergence Watch above -- sorted closest-to-passing
    // first (highest live_hurst) so the pairs worth watching sit at top.
    const liveValidation = (d.live_validation || []).slice()
      .sort((a, b) => (b.live_hurst ?? -1) - (a.live_hurst ?? -1));
    const lvp = document.getElementById('live-validation-panel');
    document.getElementById('lv-count').textContent = live ? liveValidation.length : '--';
    // Per the client 2026-09-02: this label went stale (still said 0.60)
    // the day after MIN_HURST was lowered to 0.50 -- reads the real,
    // current threshold from the bot every poll instead of a number
    // hardcoded here, so it can never go stale like that again. Falls
    // back to 0.50 (today's real value) only if an old status file
    // somehow lacks the field entirely.
    const minHurstTxt = (d.min_hurst ?? 0.50).toFixed(2);
    document.getElementById('lv-subtitle').textContent =
      `Raw live readings for every tracked pair, not only ones that qualified for Divergence Watch above. `
      + `Needs Pearson 60 Day >= 0.90, Pearson 20 Day < 0.70, Johansen 20 Day Unconfirmed, and Hurst 20 Day `
      + `above ${minHurstTxt}. Each live gate is sticky once cleared this session.`;
    lvp.innerHTML = liveValidation.length
      ? liveValidation.map(p => {
          const hurstText = p.live_hurst === null || p.live_hurst === undefined ? 'n/a' : p.live_hurst.toFixed(4);
          const hurstStyle = p.hurst_ever_above_threshold ? 'color:var(--green)' : '';
          const johansenText = p.johansen_ever_unconfirmed ? 'cleared' : (p.live_johansen_confirmed === null ? 'n/a' : 'not yet');
          const johansenStyle = p.johansen_ever_unconfirmed ? 'color:var(--green)' : '';
          // 1 Day Hurst, 1 Day Pearson, 1 Day Johansen and the condition A
          // mean-reversion row were all REMOVED from this panel as the
          // client dropped each as a gate (2026-09-08 and 2026-09-09):
          // leaving them rendered here read as if they were still active.
          // This panel now shows exactly the four gates that decide entry.
          const triggerText = p.trigger_reached_today ? 'past trigger today' : 'not yet at trigger';
          // The two daily-close Pearson gates, with their real directions:
          // 60 Day must stay high, 20 Day must have broken down.
          // Shown with an explicit cleared/not yet word, not just a colour --
          // per the client 2026-09-09: "On the display, we don't know if
          // Pearson has cleared." Both Pearson gates are decided at pair
          // SELECTION, so a pair only appears on this panel at all if it
          // already passed both; the word makes that visible instead of
          // leaving it to be inferred from the number.
          const c60 = p.correlation_60d === null || p.correlation_60d === undefined ? 'n/a' : p.correlation_60d.toFixed(4);
          const c20 = p.correlation_20d === null || p.correlation_20d === undefined ? 'n/a' : p.correlation_20d.toFixed(4);
          const c60Ok = p.correlation_60d !== null && p.correlation_60d !== undefined && p.correlation_60d >= 0.90;
          const c20Ok = p.correlation_20d !== null && p.correlation_20d !== undefined && p.correlation_20d < 0.70;
          const c60Style = c60Ok ? 'color:var(--green)' : '';
          const c20Style = c20Ok ? 'color:var(--green)' : '';
          const c60Word = c60 === 'n/a' ? '' : (c60Ok ? ' cleared' : ' not yet');
          const c20Word = c20 === 'n/a' ? '' : (c20Ok ? ' cleared' : ' not yet');
          return `<div class="pos-row"><span class="badge">${p.symbol_a}/${p.symbol_b}</span> `
            + `60d Pearson: <span style="${c60Style}">${c60}${c60Word}</span> (need &gt;=0.90) &middot; `
            + `20d Pearson: <span style="${c20Style}">${c20}${c20Word}</span> (need &lt;0.70)<br>`
            + `20d Hurst: <span style="${hurstStyle}">${hurstText}</span> (need &gt;${minHurstTxt}) &middot; `
            + `20d Johansen: <span style="${johansenStyle}">${johansenText}</span> &middot; ${triggerText}</div>`;
        }).join('')
      : `<div class="empty">${live ? 'No data yet' : 'Bot is offline'}</div>`;

    // Confirmed live 2026-08-11: this badge only ever rendered once, at
    // page load -- stayed frozen at whatever it showed then while the real
    // blacklist kept growing all day underneath it. Now refreshed every poll.
    if (d.excluded_count !== undefined) {
      document.getElementById('excluded-count').textContent = d.excluded_count + ' excluded';
    }
    // Confirmed live 2026-09-01: the badge above got fixed 2026-08-11, but
    // the table underneath it (and the "No symbols currently excluded"
    // empty state) was never wired into the poll -- same bug, one level
    // down. Delete still posts a normal form (a real page nav, not fetch),
    // same as every other delete button on this page.
    if (d.excluded_entries !== undefined) {
      const excludedBody = document.getElementById('excluded-body');
      excludedBody.innerHTML = d.excluded_entries.length
        ? `<div class="table-wrap"><table><tr><th>Symbol</th><th>Excluded Until (UTC)</th><th></th></tr>${
            d.excluded_entries.map(e => `<tr><td><span class="badge">${e.symbol}</span></td><td>${e.expires_at}</td>
              <td style="text-align:right"><form method="post" action="/exclude/delete" class="inline-del" onsubmit="return confirm('Stop excluding ${e.symbol}?')">
                <input type="hidden" name="symbol" value="${e.symbol}">
                <button class="del" type="submit" title="Remove">&times;</button>
              </form></td></tr>`).join('')
          }</table></div>`
        : '<p class="empty" style="text-align:left;padding:12px 0;">No symbols currently excluded.</p>';
    }

    const events = d.events || [];
    const ep = document.getElementById('events-panel');
    document.getElementById('ec').textContent = events.length;
    ep.innerHTML = events.length
      ? [...events].reverse().map(e => {
          const f = formatEvent(e.message);
          return `<div class="event-row"><span class="event-dot ${f.tag}"></span><div class="event-text">${f.text}<span class="event-ts">${e.ts}</span></div></div>`;
        }).join('')
      : '<div class="empty">No session has run yet</div>';

  } catch (e) { /* keep last known state on transient fetch errors */ }
}
refresh();
setInterval(refresh, 5000);

async function stopBot() {
  if (!confirm('Stop the bot and flatten all positions?')) return;
  const btn = document.getElementById('stop-btn');
  btn.textContent = 'STOPPING...'; btn.disabled = true;
  try {
    await fetch('/api/stop', {method: 'POST'});
  } finally {
    setTimeout(() => { btn.textContent = 'STOP BOT'; btn.disabled = false; }, 3000);
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    now = datetime.now(timezone.utc)
    entries = sorted(excluded_symbols.all_entries(now=now).items())
    entries = [(symbol, expires_at.strftime("%Y-%m-%d %H:%M")) for symbol, expires_at in entries]
    return render_template_string(
        PAGE_TEMPLATE, entries=entries, error=request.args.get("error"),
        market_entry=sorted(f"{a}/{b}" for a, b in (load_market_entry_list() or [])),
        blacklist=sorted(load_permanent_blacklist()),
        special_pairs=[f"{a}/{b}" for a, b in special_pairs_as_listed()],
    )


def _clean_symbol(field: str) -> str | None:
    """Tickers only. Rejects anything non-alphabetic so a stray form value
    can't land a junk entry in a list the trading engine reads."""
    value = request.form.get(field, "").strip().upper()
    return value if value and value.isalpha() and len(value) <= 6 else None


@app.route("/exclude", methods=["POST"])
def exclude_symbol():
    symbol = _clean_symbol("symbol")
    if symbol is None:
        return redirect(url_for("index", error="Enter a valid ticker symbol"))
    excluded_symbols.add(symbol)
    _audit("exclude add", symbol)
    return redirect(url_for("index"))


@app.route("/blacklist/add", methods=["POST"])
def add_blacklist_symbol():
    symbol = _clean_symbol("symbol")
    if symbol is None:
        return redirect(url_for("index", error="Enter a valid ticker symbol"))
    add_to_blacklist(symbol)
    _audit("blacklist add", symbol)
    return redirect(url_for("index"))


@app.route("/special-pairs/add", methods=["POST"])
def add_special_pair_route():
    symbol_a, symbol_b = _clean_symbol("symbol_a"), _clean_symbol("symbol_b")
    if symbol_a is None or symbol_b is None:
        return redirect(url_for("index", error="Enter two valid ticker symbols"))
    if symbol_a == symbol_b:
        return redirect(url_for("index", error="A pair needs two different symbols"))
    add_special_pair(symbol_a, symbol_b)
    _audit("special pair add", f"{symbol_a}/{symbol_b}")
    return redirect(url_for("index"))


@app.route("/market-entry/add", methods=["POST"])
def add_market_entry_pair():
    symbol_a, symbol_b = _clean_symbol("symbol_a"), _clean_symbol("symbol_b")
    if symbol_a is None or symbol_b is None:
        return redirect(url_for("index", error="Enter two valid ticker symbols"))
    if symbol_a == symbol_b:
        return redirect(url_for("index", error="A pair needs two different symbols"))
    add_pair(symbol_a, symbol_b)
    _audit("market entry add", f"{symbol_a}/{symbol_b}")
    return redirect(url_for("index"))


@app.route("/exclude/delete", methods=["POST"])
def delete_excluded_symbol():
    symbol = _clean_symbol("symbol")
    if symbol is not None:
        excluded_symbols.remove(symbol)
        _audit("exclude DELETE", symbol)
    return redirect(url_for("index"))


@app.route("/blacklist/delete", methods=["POST"])
def delete_blacklist_symbol():
    symbol = _clean_symbol("symbol")
    if symbol is not None:
        remove_from_blacklist(symbol)
        _audit("blacklist DELETE", symbol)
    return redirect(url_for("index"))


@app.route("/special-pairs/delete", methods=["POST"])
def delete_special_pair():
    symbol_a, symbol_b = _clean_symbol("symbol_a"), _clean_symbol("symbol_b")
    if symbol_a is not None and symbol_b is not None:
        remove_special_pair(symbol_a, symbol_b)
        _audit("special pair DELETE", f"{symbol_a}/{symbol_b}")
    return redirect(url_for("index"))


@app.route("/market-entry/delete", methods=["POST"])
def delete_market_entry_pair():
    symbol_a, symbol_b = _clean_symbol("symbol_a"), _clean_symbol("symbol_b")
    if symbol_a is not None and symbol_b is not None:
        remove_pair(symbol_a, symbol_b)
        _audit("market entry DELETE", f"{symbol_a}/{symbol_b}")
    return redirect(url_for("index"))


def _excluded_entries_for_json() -> list[dict]:
    """Same (symbol, expires_at) pairs index() renders server-side,
    reshaped for JSON so the poll can rebuild #excluded-body without a
    page reload -- see that div's own comment."""
    entries = sorted(excluded_symbols.all_entries(now=datetime.now(timezone.utc)).items())
    return [{"symbol": symbol, "expires_at": expires_at.strftime("%Y-%m-%d %H:%M")}
            for symbol, expires_at in entries]


@app.route("/api/state")
def api_state():
    bot_status = status.read_status_file(path=status.STATUS_PATH)
    events = status.parse_log_events(path=status.LOG_PATH)
    # No TTL on this one deliberately -- see read_cached_pairs_status's own
    # comment. Merged into BOTH branches below, since "how many pairs will
    # load tomorrow" is exactly the question that matters most while the
    # bot is between sessions (bot_status is None / offline).
    cached_pairs_status = status.read_cached_pairs_status()
    if bot_status is None:
        return jsonify({
            "status": "offline", "account": None, "elv": None,
            "open_positions": [], "daily_pnl": None, "trading_halted": False,
            # No fallback to the static qualified_pairs.csv row count here --
            # that file is Phase 1's one-time offline screen (per client
            # 2026-08-07, confirmed showing a stale "8" while the bot was
            # still in pre-market and hadn't computed anything live yet).
            # None means "no live count available", not "here's an old one".
            "qualified_pairs_count": None,
            "events": events,
            "excluded_count": len(excluded_symbols.all_entries()),
            "excluded_entries": _excluded_entries_for_json(),
            **cached_pairs_status,
        })
    return jsonify({
        **bot_status,
        "qualified_pairs_count": bot_status.get("qualified_pairs_count"),
        "events": events,
        # Confirmed live 2026-08-11: this badge only ever rendered once, at
        # page load -- unlike every other panel it was never part of the 5s
        # poll, so it stayed frozen at whatever it showed when the page was
        # first opened while the real blacklist kept growing underneath it.
        "excluded_count": len(excluded_symbols.all_entries()),
        # Confirmed live 2026-09-01: the badge above got fixed 2026-08-11,
        # but the actual table underneath it (and the "No symbols
        # currently excluded" empty state) was never added to the poll --
        # it stayed server-rendered once, at page load, same bug the badge
        # itself used to have. This is the table's own live data.
        "excluded_entries": _excluded_entries_for_json(),
        **cached_pairs_status,
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():
    try:
        STOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        STOP_FLAG_PATH.write_text("stop")
        return jsonify({"ok": True})
    except OSError as e:
        return jsonify({"ok": False, "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
