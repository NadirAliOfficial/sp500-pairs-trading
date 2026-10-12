"""Unit tests for the Market Entry List -- the hand-maintained set of pairs
the bot is allowed to trade. Uses a temp file so the real list is untouched."""
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.market_entry_list import (add_pair, filter_to_market_entry_list,
                                      load_market_entry_list, pair_key,
                                      remove_pair, save_market_entry_list)


@dataclass
class FakeContext:
    symbol_a: str
    symbol_b: str


def _tmp() -> str:
    return str(Path(tempfile.mkdtemp()) / "market_entry_list.json")


CONTEXTS = [FakeContext("GE", "RTX"), FakeContext("ANET", "LITE"),
            FakeContext("BAC", "JPM"), FakeContext("APH", "GLW"),
            FakeContext("APH", "COHR"), FakeContext("GOOG", "META")]


def test_pair_identity_ignores_symbol_order():
    assert pair_key("GOOG", "META") == pair_key("META", "GOOG")
    assert pair_key(" goog ", "meta") == ("GOOG", "META")
    print("PASS: GOOG/META and META/GOOG are the same pair")


def test_only_listed_pairs_are_traded():
    path = _tmp()
    save_market_entry_list([("GE", "RTX"), ("BAC", "JPM"), ("APH", "GLW"), ("GOOG", "META")], path)
    kept = {(c.symbol_a, c.symbol_b) for c in filter_to_market_entry_list(CONTEXTS, path)}
    assert kept == {("GE", "RTX"), ("BAC", "JPM"), ("APH", "GLW"), ("GOOG", "META")}
    assert ("ANET", "LITE") not in kept
    assert ("APH", "COHR") not in kept, "the overlapping APH pair must stay excluded"
    print("PASS: only pairs on the market entry list are traded")


def test_list_is_matched_regardless_of_stored_order():
    path = _tmp()
    save_market_entry_list([("META", "GOOG")], path)
    kept = filter_to_market_entry_list([FakeContext("GOOG", "META")], path)
    assert len(kept) == 1, "a pair stored in the opposite order must still match"
    print("PASS: a pair listed in reverse order still matches the screened pair")


def test_listed_pair_that_failed_screening_is_still_not_traded():
    """Per the client, "stocks on the Market Entry List must pass screening
    test, as applicable". The list narrows what screening produced; it never
    promotes a pair past the statistical tests. PEP/KO is the live example --
    on his special list, hand-picked, and it fails correlation, so it must
    not trade however it is listed."""
    path = _tmp()
    save_market_entry_list([("GE", "RTX"), ("PEP", "KO")], path)
    kept = {(c.symbol_a, c.symbol_b) for c in filter_to_market_entry_list(CONTEXTS, path)}
    assert kept == {("GE", "RTX")}
    assert ("PEP", "KO") not in kept
    print("PASS: a listed pair that did not pass screening is still not traded")


def test_absent_list_falls_back_to_every_qualified_pair():
    """No list configured means nobody has set one up, so screening alone
    decides -- the behaviour before the list existed."""
    assert load_market_entry_list(_tmp()) is None
    assert len(filter_to_market_entry_list(CONTEXTS, _tmp())) == len(CONTEXTS)
    print("PASS: with no list configured, all qualified pairs are traded")


def test_empty_list_means_trade_nothing():
    """Distinct from absent: someone deleted every pair from the dashboard,
    which is an instruction, not a missing config."""
    path = _tmp()
    save_market_entry_list([], path)
    assert load_market_entry_list(path) == set()
    assert filter_to_market_entry_list(CONTEXTS, path) == []
    print("PASS: an empty list trades nothing, rather than falling back to everything")


def test_corrupt_list_trades_nothing_rather_than_everything():
    """Failing open here would trade the full qualified list while the client
    believes only their chosen pairs are live."""
    path = _tmp()
    Path(path).write_text("{not valid json")
    assert load_market_entry_list(path) == set()
    assert filter_to_market_entry_list(CONTEXTS, path) == []
    print("PASS: a corrupt list trades nothing instead of falling back to everything")


def test_add_and_remove_round_trip():
    path = _tmp()
    save_market_entry_list([("GE", "RTX")], path)
    add_pair("GOOG", "META", path)
    assert load_market_entry_list(path) == {("GE", "RTX"), ("GOOG", "META")}

    add_pair("META", "GOOG", path)  # same pair, reversed -- must not duplicate
    assert len(load_market_entry_list(path)) == 2

    remove_pair("META", "GOOG", path)  # removable in either order too
    assert load_market_entry_list(path) == {("GE", "RTX")}
    print("PASS: dashboard add/delete round-trips and is order independent")


def test_saved_file_is_valid_json_with_a_pairs_key():
    path = _tmp()
    save_market_entry_list([("GOOG", "META"), ("GE", "RTX")], path)
    data = json.loads(Path(path).read_text())
    assert data["pairs"] == [["GE", "RTX"], ["GOOG", "META"]], "stored sorted for stable diffs"
    print("PASS: the list persists as sorted, readable JSON")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
