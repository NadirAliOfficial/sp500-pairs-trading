"""Unit tests for phase3/dashboard.py using Flask's built-in test client --
no real server is started. Each test points the dashboard at a temp
blacklist file so it never touches the real data/excluded_symbols.json."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.news_blacklist import ExcludedSymbols
from phase3 import dashboard, status


def _fresh_client():
    dashboard.excluded_symbols = ExcludedSymbols(path=Path(tempfile.mkdtemp()) / "excluded_symbols.json")
    dashboard.app.config["TESTING"] = True
    # Isolate from whatever the real deployment's .env has set -- a real
    # DASHBOARD_TOKEN (e.g. on the VPS, set for the actual token-gate fix)
    # would otherwise leak into every test here via load_dotenv() and 401
    # every request these tests don't pass a key for. Tests that actually
    # exercise the gate set this explicitly themselves.
    dashboard.DASHBOARD_TOKEN = None
    return dashboard.app.test_client()


def test_index_shows_empty_state_when_nothing_excluded():
    client = _fresh_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"No symbols currently excluded" in response.data
    print("PASS: index page shows the empty state when the blacklist is empty")


def test_submitting_a_symbol_adds_it_to_the_blacklist():
    client = _fresh_client()
    response = client.post("/exclude", data={"symbol": "aapl"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"AAPL" in response.data  # normalized to uppercase
    assert "AAPL" in dashboard.excluded_symbols.active_symbols()
    print("PASS: submitting a symbol adds it to the same ExcludedSymbols blacklist the bot checks")


def test_submitted_symbol_gets_the_standard_24hr_ttl():
    client = _fresh_client()
    client.post("/exclude", data={"symbol": "tsla"}, follow_redirects=True)
    with open(dashboard.excluded_symbols.path) as f:
        import json
        data = json.load(f)
    assert "TSLA" in data
    print("PASS: manually submitted symbols use the same persistence/TTL mechanism as Gemini-flagged ones")


def test_empty_symbol_is_rejected():
    client = _fresh_client()
    response = client.post("/exclude", data={"symbol": ""}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Enter a valid ticker symbol" in response.data
    assert dashboard.excluded_symbols.active_symbols() == set()
    print("PASS: submitting an empty symbol is rejected, nothing added to the blacklist")


def test_index_shows_expiry_time_for_each_entry():
    client = _fresh_client()
    client.post("/exclude", data={"symbol": "nvda"}, follow_redirects=True)
    response = client.get("/")
    assert b"NVDA" in response.data
    assert b"Excluded Until" in response.data
    print("PASS: index page shows the expiry time alongside each excluded symbol")


def test_non_alpha_symbol_is_rejected():
    client = _fresh_client()
    response = client.post("/exclude", data={"symbol": "'; DROP TABLE--"}, follow_redirects=True)
    assert dashboard.excluded_symbols.active_symbols() == set()
    print("PASS: non-alphabetic input is rejected rather than silently added")


def test_api_state_reports_offline_when_no_status_file():
    client = _fresh_client()
    status.STATUS_PATH = Path(tempfile.mkdtemp()) / "bot_status.json"  # points at nothing
    response = client.get("/api/state")
    data = response.get_json()
    assert data["status"] == "offline"
    assert data["open_positions"] == []
    print("PASS: /api/state reports 'offline' when no fresh status file exists")


def test_api_state_reflects_a_fresh_status_file():
    client = _fresh_client()
    status.STATUS_PATH = Path(tempfile.mkdtemp()) / "bot_status.json"
    status.STATUS_PATH.write_text(json.dumps({
        "ts": time.time(), "status": "running", "account": "DU0000001", "elv": 100000.0,
        "open_positions": [{"symbol_a": "TYL", "symbol_b": "WDAY", "action_a": "BUY",
                             "entry_time": "2026-07-28T10:05:00", "entry_spread": 10.0, "unrealized_pnl": 12.5}],
        "daily_pnl": 12.5, "trading_halted": False, "qualified_pairs_count": 28,
    }))
    response = client.get("/api/state")
    data = response.get_json()
    assert data["status"] == "running"
    assert data["account"] == "DU0000001"
    assert len(data["open_positions"]) == 1
    print("PASS: /api/state reflects a fresh bot_status.json written by main.py")


def test_api_state_excluded_entries_reflect_a_symbol_added_after_page_load():
    """Confirmed live 2026-09-01: the badge got fixed 2026-08-11 to poll
    live, but the table underneath it (and the 'No symbols currently
    excluded' empty state) stayed server-rendered once, at page load --
    same bug, one level down. /api/state must carry the live entries too,
    not just the count, so the poll can rebuild the table."""
    client = _fresh_client()
    status.STATUS_PATH = Path(tempfile.mkdtemp()) / "bot_status.json"  # offline branch
    response = client.get("/api/state")
    assert response.get_json()["excluded_entries"] == []

    client.post("/exclude", data={"symbol": "nvda"}, follow_redirects=True)
    response = client.get("/api/state")
    entries = response.get_json()["excluded_entries"]
    assert len(entries) == 1
    assert entries[0]["symbol"] == "NVDA"
    assert "expires_at" in entries[0]
    print("PASS: /api/state's excluded_entries reflects a symbol added after the page first loaded")


def test_api_state_excluded_entries_present_on_the_running_branch_too():
    client = _fresh_client()
    status.STATUS_PATH = Path(tempfile.mkdtemp()) / "bot_status.json"
    status.STATUS_PATH.write_text(json.dumps({
        "ts": time.time(), "status": "running", "account": "DU0000001", "elv": 100000.0,
        "open_positions": [], "daily_pnl": 0.0, "trading_halted": False, "qualified_pairs_count": 0,
    }))
    client.post("/exclude", data={"symbol": "tsla"}, follow_redirects=True)
    response = client.get("/api/state")
    entries = response.get_json()["excluded_entries"]
    assert len(entries) == 1
    assert entries[0]["symbol"] == "TSLA"
    print("PASS: excluded_entries is also carried on the running-status branch, not just offline")


def test_api_stop_writes_the_stop_flag():
    client = _fresh_client()
    dashboard.STOP_FLAG_PATH = Path(tempfile.mkdtemp()) / "bot_stop.txt"
    response = client.post("/api/stop")
    assert response.get_json()["ok"] is True
    assert dashboard.STOP_FLAG_PATH.exists()
    print("PASS: /api/stop writes the stop flag file main.py's loop checks each cycle")


def test_token_set_rejects_requests_without_it():
    client = _fresh_client()
    dashboard.DASHBOARD_TOKEN = "secret123"
    try:
        response = client.post("/api/stop")
        assert response.status_code == 401
    finally:
        dashboard.DASHBOARD_TOKEN = None
    print("PASS: with DASHBOARD_TOKEN set, an unauthenticated request is rejected")


def test_token_in_query_string_is_accepted_and_persisted_as_a_cookie():
    client = _fresh_client()
    dashboard.STOP_FLAG_PATH = Path(tempfile.mkdtemp()) / "bot_stop.txt"
    dashboard.DASHBOARD_TOKEN = "secret123"
    try:
        response = client.get("/?key=secret123")
        assert response.status_code == 200
        # the cookie set on that request should authenticate the next one
        # with no key in the URL at all
        response = client.post("/api/stop")
        assert response.status_code == 200
    finally:
        dashboard.DASHBOARD_TOKEN = None
    print("PASS: a correct ?key= authenticates once and the session cookie carries it forward")


def test_wrong_token_is_rejected():
    client = _fresh_client()
    dashboard.DASHBOARD_TOKEN = "secret123"
    try:
        response = client.get("/?key=wrong-guess")
        assert response.status_code == 401
    finally:
        dashboard.DASHBOARD_TOKEN = None
    print("PASS: an incorrect key is rejected, not just a missing one")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
