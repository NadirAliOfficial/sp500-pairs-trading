"""Unit tests for phase3/status.py's read_status_file and parse_log_events --
the read-side counterparts to phase2/main.py's write_status()."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase3.status import parse_log_events, read_cached_pairs_status, read_status_file


def _write_status(path: Path, ts: float, **fields):
    path.write_text(json.dumps({"ts": ts, **fields}))


def test_read_status_file_returns_data_when_fresh():
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    now = time.time()
    _write_status(tmp, ts=now, status="running", account="DU0000001")
    data = read_status_file(path=tmp, ttl_seconds=30, now=now + 5)
    assert data is not None
    assert data["status"] == "running"
    print("PASS: read_status_file returns data when the file is within the TTL")


def test_read_status_file_none_when_stale():
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    now = time.time()
    _write_status(tmp, ts=now, status="running")
    data = read_status_file(path=tmp, ttl_seconds=30, now=now + 31)
    assert data is None
    print("PASS: read_status_file returns None once the file is older than the TTL (bot treated as offline)")


def test_read_status_file_none_when_missing():
    assert read_status_file(path=Path("/tmp/does-not-exist-bot-status.json")) is None
    print("PASS: read_status_file returns None when no status file exists yet")


def test_read_status_file_none_when_corrupt():
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    tmp.write_text("not valid json{{{")
    assert read_status_file(path=tmp) is None
    print("PASS: read_status_file returns None instead of crashing on corrupt JSON")


def test_read_cached_pairs_status_returns_data_regardless_of_age():
    """Deliberately no TTL, unlike read_status_file -- per the client
    repeatedly asking "how many pairs will load by 9am", this has to
    stay visible through the hours the bot process isn't even running."""
    tmp = Path(tempfile.mkdtemp()) / "cached_pairs_status.json"
    tmp.write_text(json.dumps({
        "cached_pairs_for_tomorrow_count": 8,
        "cached_pairs_for_tomorrow_symbols": ["AAA/BBB"],
        "cached_pairs_for_tomorrow_at": "2026-09-14T16:12:00-04:00",
    }))
    data = read_cached_pairs_status(path=tmp)
    assert data["cached_pairs_for_tomorrow_count"] == 8
    print("PASS: read_cached_pairs_status returns data with no staleness check at all")


def test_read_cached_pairs_status_empty_dict_when_missing_or_corrupt():
    assert read_cached_pairs_status(path=Path("/tmp/does-not-exist-cached-pairs.json")) == {}
    tmp = Path(tempfile.mkdtemp()) / "cached_pairs_status.json"
    tmp.write_text("not valid json{{{")
    assert read_cached_pairs_status(path=tmp) == {}
    print("PASS: read_cached_pairs_status fails open to {} (not None) when missing or corrupt, so ** always works")


def test_parse_log_events_extracts_enter_and_exit_lines():
    tmp = Path(tempfile.mkdtemp()) / "pairs_bot.log"
    tmp.write_text(
        "2026-07-28 10:05:00,123 INFO pairs_bot.main: === Session start 2026-07-28 10:05:00-04:00 ===\n"
        "2026-07-28 10:05:01,456 INFO pairs_bot.ibkr_client: IBKR notice 2104 (reqId=-1): Market data farm connection is OK:usfarm\n"
        "2026-07-28 10:06:00,789 INFO pairs_bot.main: ENTER TYL/WDAY: BUY 100 / SELL 200\n"
        "2026-07-28 10:30:00,111 INFO pairs_bot.main: EXIT TYL/WDAY: take_profit, pnl=42.50\n"
    )
    events = parse_log_events(path=tmp)
    messages = [e["message"] for e in events]
    assert any("Session start" in m for m in messages)
    assert any("ENTER TYL/WDAY" in m for m in messages)
    assert any("EXIT TYL/WDAY" in m for m in messages)
    assert not any("Market data farm" in m for m in messages)  # noise filtered out
    print("PASS: parse_log_events extracts ENTER/EXIT/session events, filters out IBKR connection noise")


def test_parse_log_events_extracts_beta_breakdown_and_intraday_blacklist():
    # both were missing from EVENT_KEYWORDS -- confirmed real gaps, these
    # events never showed up in the dashboard's feed at all before the fix
    tmp = Path(tempfile.mkdtemp()) / "pairs_bot.log"
    tmp.write_text(
        "2026-07-28 10:06:00,789 WARNING pairs_bot.main: BETA BREAKDOWN TYL/WDAY: hedge relationship invalidated, pnl=-120.00\n"
        "2026-07-28 10:10:00,111 INFO pairs_bot.main: Intraday news scan blacklisted: ['AAPL']\n"
    )
    events = parse_log_events(path=tmp)
    messages = [e["message"] for e in events]
    assert any("BETA BREAKDOWN TYL/WDAY" in m for m in messages)
    assert any("Intraday news scan blacklisted" in m for m in messages)
    print("PASS: parse_log_events also captures beta breakdown exits and intraday blacklist updates")


def test_parse_log_events_empty_when_no_log():
    assert parse_log_events(path=Path("/tmp/does-not-exist-pairs-bot.log")) == []
    print("PASS: parse_log_events returns an empty list when no log file exists yet")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
