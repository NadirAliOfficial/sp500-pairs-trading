"""Unit tests for the Permanent Blacklist and Special Pairs List."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.trading_lists import (DEFAULT_PERMANENT_BLACKLIST, DEFAULT_SPECIAL_PAIRS,
                                  add_special_pair, add_to_blacklist, is_blacklisted,
                                  is_special_pair, load_permanent_blacklist,
                                  load_special_pairs, pair_is_blacklisted,
                                  remove_from_blacklist, remove_special_pair,
                                  save_permanent_blacklist, save_special_pairs,
                                  special_pairs_as_listed)


def _tmp(name: str) -> str:
    return str(Path(tempfile.mkdtemp()) / name)


def test_defaults_are_the_clients_lists():
    """A fresh deployment must start with the spec's lists, not empty ones."""
    assert load_permanent_blacklist(_tmp("bl.json")) == set(DEFAULT_PERMANENT_BLACKLIST)
    assert len(load_special_pairs(_tmp("sp.json"))) == 6
    assert {"NVDA", "TSLA", "SPCX", "GOOG", "META"} <= set(DEFAULT_PERMANENT_BLACKLIST)
    assert len(DEFAULT_SPECIAL_PAIRS) == 6
    print("PASS: missing files fall back to the client's NVDA/TSLA/SPCX blacklist and 6 special pairs")


def test_blacklisted_leg_disqualifies_the_whole_pair():
    """The bot cannot trade one side of a pair, so one blacklisted leg has to
    take the pair with it. NVDA on the list is what removed AVGO/NVDA."""
    path = _tmp("bl.json")
    save_permanent_blacklist(["NVDA", "TSLA"], path)
    assert pair_is_blacklisted("AVGO", "NVDA", path)
    assert pair_is_blacklisted("NVDA", "AMD", path)
    assert not pair_is_blacklisted("GOOG", "META", path)
    print("PASS: one blacklisted leg disqualifies the pair")


def test_blacklist_is_case_and_whitespace_insensitive():
    path = _tmp("bl.json")
    save_permanent_blacklist([" nvda "], path)
    assert is_blacklisted("NVDA", path) and is_blacklisted("nvda", path)
    print("PASS: blacklist tolerates casing and stray whitespace from the dashboard form")


def test_special_pairs_display_in_the_order_the_client_wrote_them():
    """He maintains this list by hand as PEP/KO, XOM/CVX, MDT/ABT. Showing it
    back as KO/PEP, CVX/XOM, ABT/MDT reads as a bug on his own list."""
    path = _tmp("sp.json")
    save_special_pairs(DEFAULT_SPECIAL_PAIRS, path)
    shown = [f"{a}/{b}" for a, b in special_pairs_as_listed(path)]
    assert shown == ["PEP/KO", "V/MA", "XOM/CVX", "JPM/BAC", "CAT/DE", "MDT/ABT"]
    print("PASS: special pairs display in the client's own order")


def test_special_pair_matching_ignores_order():
    path = _tmp("sp.json")
    save_special_pairs([("PEP", "KO")], path)
    assert is_special_pair("PEP", "KO", path) and is_special_pair("KO", "PEP", path)
    print("PASS: special pair matching works whichever way round the symbols come")


def test_adding_the_same_pair_reversed_does_not_duplicate_it():
    path = _tmp("sp.json")
    save_special_pairs([("PEP", "KO")], path)
    add_special_pair("KO", "PEP", path)
    assert len(special_pairs_as_listed(path)) == 1
    print("PASS: re-adding a pair reversed does not duplicate it")


def test_add_and_remove_round_trip():
    bl = _tmp("bl.json")
    save_permanent_blacklist(["NVDA"], bl)
    add_to_blacklist("tsla", bl)
    assert load_permanent_blacklist(bl) == {"NVDA", "TSLA"}
    remove_from_blacklist("NVDA", bl)
    assert load_permanent_blacklist(bl) == {"TSLA"}

    sp = _tmp("sp.json")
    save_special_pairs([("PEP", "KO"), ("V", "MA")], sp)
    remove_special_pair("MA", "V", sp)  # reversed
    assert [f"{a}/{b}" for a, b in special_pairs_as_listed(sp)] == ["PEP/KO"]
    print("PASS: add/remove round-trips on both lists, order independent")


def test_corrupt_file_falls_back_to_defaults_rather_than_an_empty_blacklist():
    """An empty blacklist would let the blacklisted names trade. Falling back
    to the spec defaults is the safe direction for this list specifically."""
    path = _tmp("bl.json")
    Path(path).write_text("{ broken")
    assert load_permanent_blacklist(path) == set(DEFAULT_PERMANENT_BLACKLIST)
    print("PASS: a corrupt blacklist falls back to the defaults, never to empty")


def test_special_pairs_exempt_only_the_three_size_filters():
    """Screening honours the exemption for market cap and volume, and for
    nothing else -- PEP/KO fails correlation and must still fail."""
    from phase1.screening import PairResult

    def result(**kw):
        # johansen_confirmed=False, hurst=0.70 -- the passing values under
        # the client's 2026-08-25 qualification flip (trending, not
        # mean-reverting; see phase1/screening.py's own module docstring).
        base = dict(symbol_a="PEP", symbol_b="KO", industry="Beverages",
                    # correlation_recent broken down (< 0.70) -- per the client
                    # 2026-09-08 that's the passing side of the 20 day gate.
                    correlation=0.41, correlation_recent=0.50, adf_pvalue=0.001, cointegrated=True,
                    johansen_confirmed=False, half_life=20.0, hurst=0.70,
                    volatility_stable=True, liquidity_ok=True, market_cap_ok=True,
                    hedge_ratio=1.0)
        return PairResult(**{**base, **kw})

    assert not result().qualified, "0.41 correlation must fail even when size filters pass"
    assert result(correlation=0.95).qualified
    print("PASS: the size-filter exemption cannot rescue a pair failing correlation")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
