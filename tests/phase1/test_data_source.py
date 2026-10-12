"""Unit tests for phase1/data_source.py's market cap fetch -- mocked
yfinance calls, same resilience patterns already proven for dividends
(retry+backoff, None-return handling, modest thread count)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import MagicMock, patch

from phase1.data_source import _fetch_market_cap, fetch_market_caps_yfinance


def test_fetch_market_cap_returns_value_on_success():
    mock_ticker = MagicMock()
    mock_ticker.fast_info = {"market_cap": 25_000_000_000.0}
    with patch("phase1.data_source.yf.Ticker", return_value=mock_ticker):
        assert _fetch_market_cap("TEST") == 25_000_000_000.0
    print("PASS: _fetch_market_cap returns the value on a successful lookup")


def test_fetch_market_cap_retries_then_gives_up_on_persistent_none():
    # retries=2 keeps the real backoff delay short (0.5s) rather than mocking
    # time.sleep -- same convention as the existing dividend-fetch tests,
    # which don't mock the local `import time` either.
    mock_ticker = MagicMock()
    mock_ticker.fast_info = {}  # KeyError on ["market_cap"] every time
    with patch("phase1.data_source.yf.Ticker", return_value=mock_ticker):
        assert _fetch_market_cap("TEST", retries=2) is None
    print("PASS: _fetch_market_cap gives up and returns None after exhausting retries")


def test_fetch_market_cap_handles_exception_not_just_missing_key():
    with patch("phase1.data_source.yf.Ticker", side_effect=Exception("network error")):
        assert _fetch_market_cap("TEST", retries=2) is None
    print("PASS: _fetch_market_cap falls back to None on a raised exception, not just a missing key")


def test_fetch_market_caps_yfinance_filters_out_failed_symbols():
    def fake_fetch(symbol, retries=3):
        return {"AAA": 30_000_000_000.0, "BBB": None}[symbol]

    with patch("phase1.data_source._fetch_market_cap", side_effect=fake_fetch):
        result = fetch_market_caps_yfinance(["AAA", "BBB"])
    assert result == {"AAA": 30_000_000_000.0}
    print("PASS: fetch_market_caps_yfinance omits symbols that failed to resolve instead of including None")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
