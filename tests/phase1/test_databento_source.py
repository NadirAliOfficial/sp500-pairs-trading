"""Validates the dividend-adjustment logic against known synthetic cases
before trusting it on real Databento data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import date
from unittest.mock import patch

import pandas as pd

from phase1.databento_source import _dividend_adjust, _fetch_intraday_rth, fetch_volume_stats_databento


def test_single_dividend_adjusts_prices_before_ex_date():
    closes = pd.Series([100.0] * 5, index=pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]).date)
    fake_divs = pd.Series([2.0], index=pd.to_datetime(["2024-01-03"]))

    with patch("phase1.databento_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.dividends = fake_divs
        adjusted = _dividend_adjust(closes, "TEST")

    expected = pd.Series([98.0, 98.0, 100.0, 100.0, 100.0], index=closes.index)
    assert (adjusted.round(4) == expected.round(4)).all(), f"{adjusted}\nvs\n{expected}"
    print("PASS: single dividend adjusts prices strictly before ex-date only")


def test_two_dividends_compound_correctly_newest_first():
    closes = pd.Series([100.0] * 7, index=pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
         "2024-01-05", "2024-01-06", "2024-01-07"]).date)
    fake_divs = pd.Series([2.0, 3.0], index=pd.to_datetime(["2024-01-03", "2024-01-06"]))

    with patch("phase1.databento_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.dividends = fake_divs
        adjusted = _dividend_adjust(closes, "TEST")

    # ratio formula is exact: prior_close * (1 - div/prior_close) = prior_close - div
    # day-6 div ($3) applies to days 1-5: 100-3=97
    # day-3 div ($2) applies to days 1-2, using the already-adjusted 97 level: 97-2=95
    expected = [95.0, 95.0, 97.0, 97.0, 97.0, 100.0, 100.0]
    for i, (got, exp) in enumerate(zip(adjusted.values, expected)):
        assert abs(got - exp) < 1e-6, f"day {i}: got {got}, expected {exp}"
    print("PASS: two compounding dividends adjust correctly in newest-to-oldest order")


def test_no_dividends_returns_unchanged():
    closes = pd.Series([50.0, 51.0, 52.0], index=pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03"]).date)
    with patch("phase1.databento_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.dividends = pd.Series([], dtype=float)
        adjusted = _dividend_adjust(closes, "TEST")
    assert (adjusted == closes).all()
    print("PASS: no dividend history leaves prices unchanged")


def test_rate_limited_none_falls_back_to_raw():
    # 2026-07-26 real bug: under bulk parallel load, yfinance returned None
    # (not an exception) for legitimate, definitely-not-delisted stocks like
    # NVDA/PG/PFE/ORCL -- crashed with AttributeError on divs.empty since
    # only exceptions were being caught, not a None return value.
    closes = pd.Series([50.0, 51.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]).date)
    with patch("phase1.databento_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.dividends = None
        adjusted = _dividend_adjust(closes, "TEST")
    assert (adjusted == closes).all()
    print("PASS: yfinance returning None (rate-limit symptom) falls back to unadjusted prices, no crash")


def test_fetch_failure_falls_back_to_raw():
    closes = pd.Series([50.0, 51.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]).date)
    with patch("phase1.databento_source.yf.Ticker", side_effect=Exception("network error")):
        adjusted = _dividend_adjust(closes, "TEST")
    assert (adjusted == closes).all()
    print("PASS: a dividend-fetch failure falls back to unadjusted prices instead of crashing")


def test_fetch_volume_stats_computes_both_metrics_from_one_fetch():
    fake_raw = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "BBB"],
        "close": [10.0, 12.0, 100.0, 110.0],
        "volume": [1_000_000, 2_000_000, 50_000, 60_000],
    })
    with patch("phase1.databento_source._fetch_raw", return_value=fake_raw):
        stats = fetch_volume_stats_databento(["AAA", "BBB"], "2024-01-01", "2024-01-02")

    assert stats["AAA"]["avg_share_volume"] == 1_500_000
    assert stats["AAA"]["avg_dollar_volume"] == (10.0 * 1_000_000 + 12.0 * 2_000_000) / 2
    assert stats["BBB"]["avg_share_volume"] == 55_000
    print("PASS: fetch_volume_stats_databento computes both share and dollar volume averages correctly")


def test_fetch_volume_stats_empty_when_no_data():
    with patch("phase1.databento_source._fetch_raw", return_value=pd.DataFrame()):
        stats = fetch_volume_stats_databento(["AAA"], "2024-01-01", "2024-01-02")
    assert stats == {}
    print("PASS: fetch_volume_stats_databento returns empty dict when no data is available")


def test_fetch_intraday_rth_trims_sessions_globally_not_per_chunk():
    """Per the independent audit's Finding #9, verified real: trimming to
    the last N sessions used to happen PER CHUNK, so if one chunk's fetch
    lagged behind another (a partial Databento outage, a chunk-specific
    delay), the final concatenated result could mix rows from MORE than N
    distinct calendar dates -- one chunk's newest session paired against a
    different chunk's own, different, newest session under the same
    nominal window. AAA (chunk 1) has bars through day 3; ZZZ (chunk 2)
    only ever got through day 2, simulating exactly that lag."""
    symbols = [f"SYM{i}" for i in range(61)]  # 61 symbols -- forces 2 chunks at SYMBOL_CHUNK=60

    def fake_fetch_raw(chunk_symbols, start, end, schema="ohlcv-1m", datasets=None):
        if chunk_symbols == symbols[:60]:
            return pd.DataFrame({
                "ts_event": pd.to_datetime([
                    "2026-01-05T15:00:00Z", "2026-01-06T15:00:00Z", "2026-01-07T15:00:00Z",
                ]),
                "symbol": ["AAA", "AAA", "AAA"],
                "close": [10.0, 11.0, 12.0],
            })
        return pd.DataFrame({
            "ts_event": pd.to_datetime(["2026-01-05T15:00:00Z", "2026-01-06T15:00:00Z"]),
            "symbol": ["ZZZ", "ZZZ"],
            "close": [100.0, 101.0],
        })

    with patch("phase1.databento_source._fetch_raw", side_effect=fake_fetch_raw):
        result = _fetch_intraday_rth(symbols, "2026-01-05T00:00:00", "2026-01-08T00:00:00", sessions=2)

    # The GLOBAL last 2 sessions are day 2 and day 3 (2026-01-06/07) -- ZZZ
    # never has day 3 data at all, and must not show up carrying day 1
    # data mislabeled alongside AAA's own day 2/3 rows.
    assert set(result["session"].unique()) == {date(2026, 1, 6), date(2026, 1, 7)}
    assert set(result[result["symbol"] == "AAA"]["session"]) == {date(2026, 1, 6), date(2026, 1, 7)}
    assert set(result[result["symbol"] == "ZZZ"]["session"]) == {date(2026, 1, 6)}
    print("PASS: session trim is computed globally across all chunks, not independently per chunk")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
