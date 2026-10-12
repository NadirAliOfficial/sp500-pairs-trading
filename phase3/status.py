"""Operational status checks for the dashboard -- separated from dashboard.py
so the logic (port check, pair count, log tail, status-file read, log-event
parsing) is unit testable without spinning up Flask. None of this holds a
live IBKR session itself; main.py's trading engine owns that connection and
periodically writes bot_status.json, this module just reads what's already
on disk -- same split as the sibling SPY bot's dashboard.py/main.py.
"""
import os
import re
import socket
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUALIFIED_PAIRS_PATH = DATA_DIR / "qualified_pairs.csv"
LOG_PATH = DATA_DIR / "pairs_bot.log"
STATUS_PATH = DATA_DIR / "bot_status.json"
STATUS_TTL_SECONDS = 150  # if bot_status.json is older than this, treat the bot as offline.
# The overnight pair cache's own count, written once by main.py's
# write_cached_pairs_status right after each evening's full S&P 500 scan
# finishes -- deliberately a separate file from STATUS_PATH, with no TTL
# of its own, so it stays visible through the hours the bot process isn't
# even running, instead of vanishing once bot_status.json goes stale.
CACHED_PAIRS_STATUS_PATH = DATA_DIR / "cached_pairs_status.json"
# Must comfortably exceed main.py's write interval (CHECK_INTERVAL_SECONDS = 60s) --
# a TTL shorter than or close to the write cadence would show "offline" for
# a real chunk of every single cycle even while the bot runs normally. The
# SPY bot's dashboard used 30s, but its status_writer refreshes every 5s;
# this bot's loop is slower, so the TTL has to scale with it.

LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ (INFO|WARNING|ERROR) ([\w.]+): (.+)$"
)
EVENT_KEYWORDS = ("ENTER ", "EXIT ", "BETA BREAKDOWN", "=== Session start", "=== 3:59pm flatten",
                   "Blacklisted today", "Intraday news scan blacklisted", "early close day", "market holiday",
                   "no warm-up data available", "Recovered mean divergence level",
                   "MARKET WIDE HALT", "INTERNAL CIRCUIT BREAKER",
                   # Rejections and other execution problems -- per the client
                   # 2026-08-07, order rejections need to be visible on the
                   # dashboard the same way the sibling SPY bot shows them,
                   # not just in the raw log file. "IBKR error" catches every
                   # broker-level rejection/warning (wrong price, insufficient
                   # margin, etc.); the other two catch this bot's own
                   # higher-level handling of a rejection.
                   "IBKR error", "entry rejected by IBKR", "clearing the phantom position",
                   "Intraday news scan failed")
# IBKR error codes that fire on every single order placed and mean nothing
# is wrong -- 10349 just confirms the TIF was set to DAY (always true and
# expected here), not a rejection or a problem. Excluded from the event feed
# so it doesn't drown out real rejections: confirmed live 2026-08-07, one
# entry attempt alone produced dozens of these in under two minutes.
BENIGN_IBKR_ERROR_CODES = re.compile(r"^IBKR error (10349)\b")


def is_gateway_reachable(host: str | None = None, port: int | None = None, timeout: float = 2.0) -> bool:
    """Cheap TCP connect check against the IBKR gateway's API port -- True
    only means something is listening and accepting connections, not that
    login/auth succeeded."""
    host = host or os.getenv("IBKR_HOST", "127.0.0.1")
    port = port or int(os.getenv("IBKR_PORT", "7497"))
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def count_qualified_pairs(path: Path = QUALIFIED_PAIRS_PATH) -> int | None:
    """Number of pairs Phase 1 qualified, or None if the file doesn't exist
    yet (Phase 1 screening hasn't been run)."""
    if not path.exists():
        return None
    with open(path) as f:
        line_count = sum(1 for _ in f)
    return max(line_count - 1, 0)  # minus the header row


def recent_log_lines(n: int = 15, path: Path = LOG_PATH) -> list[str]:
    """Last n lines from the bot's log file, most recent last. Empty list if
    no session has run yet (log file doesn't exist)."""
    if not path.exists():
        return []
    with open(path) as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-n:]]


def read_status_file(path: Path = STATUS_PATH, ttl_seconds: float = STATUS_TTL_SECONDS,
                      now: float | None = None) -> dict | None:
    """Reads bot_status.json -- returns None if missing, unparseable, or
    older than ttl_seconds (meaning main.py's writer loop has stopped
    updating it, so the bot is effectively offline even if the file exists
    from an earlier session)."""
    import json
    now = now if now is not None else time.time()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if now - data.get("ts", 0) > ttl_seconds:
        return None
    return data


def read_cached_pairs_status(path: Path = CACHED_PAIRS_STATUS_PATH) -> dict:
    """Reads the overnight cache's own count -- no TTL/staleness check,
    unlike read_status_file, since this is meant to stay visible even
    while the bot process isn't running at all (see the file's own write
    site, main.py's write_cached_pairs_status). Returns {} (not None) if
    missing or unparseable, so callers can always safely ** it."""
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def parse_log_events(n: int = 30, path: Path = LOG_PATH) -> list[dict]:
    """Extracts structured trade/session events (ENTER, EXIT, session
    milestones) from today's log lines, most recent last. Used as the
    always-available fallback/history view -- bot_status.json only holds
    the CURRENT state, not a timeline, so this is where the dashboard's
    event feed comes from regardless of whether the bot is live right now."""
    events = []
    for line in recent_log_lines(n=10_000, path=path):  # scan everything, filter, then trim
        match = LOG_LINE_RE.match(line)
        if not match:
            continue
        timestamp, level, logger_name, message = match.groups()
        if BENIGN_IBKR_ERROR_CODES.match(message):
            continue
        if any(keyword in message for keyword in EVENT_KEYWORDS):
            events.append({"ts": timestamp, "level": level, "message": message})
    return events[-n:]
