"""Round-trip tests for the dashboard's list editing: every Add and Delete
button must actually change what the trading engine reads.

These go through the real Flask routes rather than calling the list functions
directly -- a delete button that posts to the wrong place, or silently does
nothing, is exactly the failure that would have the client believe a pair was
removed from trading while it kept trading.

Each test points the list functions at temp files, so the live lists on disk
are never touched.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import phase3.dashboard as dash
from phase2.market_entry_list import load_market_entry_list, save_market_entry_list
from phase2.news_blacklist import ExcludedSymbols
from phase2.trading_lists import (load_permanent_blacklist, save_permanent_blacklist,
                                  save_special_pairs, special_pairs_as_listed)


def _bind(tmp: Path):
    """Points every list the dashboard touches at temp files. The list
    functions capture their path as a default argument at definition time, so
    patching the module constant is not enough -- the bound functions
    themselves have to be replaced."""
    mel, bl, sp = str(tmp / "mel.json"), str(tmp / "bl.json"), str(tmp / "sp.json")
    dash.add_pair = lambda a, b: __import__("phase2.market_entry_list", fromlist=["x"]).add_pair(a, b, mel)
    dash.remove_pair = lambda a, b: __import__("phase2.market_entry_list", fromlist=["x"]).remove_pair(a, b, mel)
    dash.load_market_entry_list = lambda: load_market_entry_list(mel)
    dash.add_to_blacklist = lambda s: __import__("phase2.trading_lists", fromlist=["x"]).add_to_blacklist(s, bl)
    dash.remove_from_blacklist = lambda s: __import__("phase2.trading_lists", fromlist=["x"]).remove_from_blacklist(s, bl)
    dash.load_permanent_blacklist = lambda: load_permanent_blacklist(bl)
    dash.add_special_pair = lambda a, b: __import__("phase2.trading_lists", fromlist=["x"]).add_special_pair(a, b, sp)
    dash.remove_special_pair = lambda a, b: __import__("phase2.trading_lists", fromlist=["x"]).remove_special_pair(a, b, sp)
    dash.special_pairs_as_listed = lambda: special_pairs_as_listed(sp)
    dash.excluded_symbols = ExcludedSymbols(path=tmp / "ex.json")
    return mel, bl, sp


def _setup():
    tmp = Path(tempfile.mkdtemp())
    mel, bl, sp = _bind(tmp)
    save_market_entry_list([("GE", "RTX"), ("GOOG", "META")], mel)
    save_permanent_blacklist(["NVDA", "TSLA"], bl)
    save_special_pairs([("PEP", "KO"), ("V", "MA")], sp)
    # Isolate from a real deployment's .env DASHBOARD_TOKEN -- see the same
    # reset in test_dashboard.py's _fresh_client() for why.
    dash.DASHBOARD_TOKEN = None
    return dash.app.test_client(), mel, bl, sp


def test_market_entry_add_and_delete_change_what_trades():
    client, mel, _, _ = _setup()
    client.post("/market-entry/add", data={"symbol_a": "BAC", "symbol_b": "JPM"})
    assert ("BAC", "JPM") in load_market_entry_list(mel)

    client.post("/market-entry/delete", data={"symbol_a": "GE", "symbol_b": "RTX"})
    remaining = load_market_entry_list(mel)
    assert ("GE", "RTX") not in remaining
    assert ("GOOG", "META") in remaining, "deleting one pair must not disturb the others"
    print("PASS: market entry add/delete actually changes the traded pair list")


def test_market_entry_delete_works_with_symbols_reversed():
    """The button posts whatever order the pair is displayed in."""
    client, mel, _, _ = _setup()
    client.post("/market-entry/delete", data={"symbol_a": "META", "symbol_b": "GOOG"})
    assert ("GOOG", "META") not in load_market_entry_list(mel)
    print("PASS: market entry delete works whichever way round the symbols are posted")


def test_blacklist_add_and_delete():
    client, _, bl, _ = _setup()
    client.post("/blacklist/add", data={"symbol": "amd"})
    assert "AMD" in load_permanent_blacklist(bl)
    client.post("/blacklist/delete", data={"symbol": "NVDA"})
    assert load_permanent_blacklist(bl) == {"TSLA", "AMD"}
    print("PASS: blacklist add/delete round-trips through the dashboard")


def test_special_pair_add_and_delete():
    client, _, _, sp = _setup()
    client.post("/special-pairs/add", data={"symbol_a": "XOM", "symbol_b": "CVX"})
    assert ("XOM", "CVX") in special_pairs_as_listed(sp)
    client.post("/special-pairs/delete", data={"symbol_a": "KO", "symbol_b": "PEP"})
    assert [f"{a}/{b}" for a, b in special_pairs_as_listed(sp)] == ["V/MA", "XOM/CVX"]
    print("PASS: special pair add/delete round-trips, delete is order independent")


def test_excluded_symbol_add_and_delete():
    client, _, _, _ = _setup()
    client.post("/exclude", data={"symbol": "AAPL"})
    assert "AAPL" in dash.excluded_symbols.active_symbols()
    client.post("/exclude/delete", data={"symbol": "AAPL"})
    assert "AAPL" not in dash.excluded_symbols.active_symbols()
    print("PASS: excluded symbol can be added and lifted before its 24 hours expire")


def test_junk_input_is_rejected_rather_than_stored():
    """These lists are read by the trading engine, so a stray form value must
    not land in one."""
    client, mel, bl, _ = _setup()
    before_bl, before_mel = load_permanent_blacklist(bl), load_market_entry_list(mel)
    client.post("/blacklist/add", data={"symbol": "../../etc/passwd"})
    client.post("/blacklist/add", data={"symbol": ""})
    client.post("/market-entry/add", data={"symbol_a": "GOOG", "symbol_b": "GOOG"})
    client.post("/market-entry/add", data={"symbol_a": "GOOG", "symbol_b": "123"})
    assert load_permanent_blacklist(bl) == before_bl
    assert load_market_entry_list(mel) == before_mel
    print("PASS: junk, empty and same-symbol input is rejected, lists unchanged")


def test_every_list_renders_a_delete_control():
    client, _, _, _ = _setup()
    client.post("/exclude", data={"symbol": "AAPL"})  # its delete only renders with a row present
    html = client.get("/").get_data(as_text=True)
    for action in ["/market-entry/delete", "/blacklist/delete",
                   "/special-pairs/delete", "/exclude/delete"]:
        assert action in html, f"{action} has no button on the page"
    assert "Market Entry List" in html
    print("PASS: all four lists render both an add form and delete controls")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
