"""Unit tests for the S&P 500 universe grouping. Uses a synthetic frame --
fetch_sp500_universe() hits Wikipedia and isn't exercised here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from phase1.sp500_universe import drop_duplicate_share_classes, industry_groups


def _universe() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": "GOOGL", "security": "Alphabet Inc. (Class A)",
         "gics_sector": "Communication Services", "gics_sub_industry": "Interactive Media & Services"},
        {"symbol": "GOOG", "security": "Alphabet Inc. (Class C)",
         "gics_sector": "Communication Services", "gics_sub_industry": "Interactive Media & Services"},
        {"symbol": "META", "security": "Meta Platforms",
         "gics_sector": "Communication Services", "gics_sub_industry": "Interactive Media & Services"},
        {"symbol": "INTC", "security": "Intel",
         "gics_sector": "Information Technology", "gics_sub_industry": "Semiconductors"},
        {"symbol": "MU", "security": "Micron Technology",
         "gics_sector": "Information Technology", "gics_sub_industry": "Semiconductors"},
        {"symbol": "LONE", "security": "Lonely Co",
         "gics_sector": "Utilities", "gics_sub_industry": "Water Utilities"},
    ])


def test_duplicate_share_classes_collapse_to_one_ticker():
    kept = set(drop_duplicate_share_classes(_universe())["symbol"])
    assert ("GOOGL" in kept) != ("GOOG" in kept), "exactly one Alphabet class should survive"
    assert {"META", "INTC", "MU"} <= kept, "single-class companies must be untouched"
    print("PASS: a company listed under two share classes collapses to one ticker")


def test_dedupe_is_deterministic():
    first = drop_duplicate_share_classes(_universe())["symbol"].tolist()
    second = drop_duplicate_share_classes(_universe())["symbol"].tolist()
    assert first == second
    print("PASS: dedupe picks the same ticker every run")


def test_industry_groups_cannot_pair_two_classes_of_one_company():
    """Regression guard. A 1-minute screen genuinely returned GOOGL/META and
    GOOG/META as its only two qualifying "pairs" -- the same Alphabet vs Meta
    bet twice, which would have the bot carry double exposure while believing
    it held two independent positions."""
    groups = industry_groups(_universe())
    media = groups["Interactive Media & Services"]
    assert not {"GOOGL", "GOOG"} <= set(media), "both Alphabet classes reached the same pairing group"
    assert "META" in media and len(media) == 2
    print("PASS: both share classes of one company can never be paired against the same counterpart")


def test_industry_groups_drops_sub_industries_with_one_member():
    groups = industry_groups(_universe())
    assert "Water Utilities" not in groups, "a lone symbol can't form a pair"
    assert set(groups["Semiconductors"]) == {"INTC", "MU"}
    print("PASS: sub-industries with fewer than 2 members are excluded")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
