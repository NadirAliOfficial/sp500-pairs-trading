"""Historical price data fetch layer.

Phase 1 uses yfinance daily bars for the correlation/cointegration screen --
these are structural, slow-moving relationships (shared industry, business
model) that show up clearly at daily resolution, and yfinance needs no paid
API key, so the pipeline can be built and validated end-to-end right now.

Once the client's Databento API key is available, replace fetch_daily_closes()
internals with a Databento pull (their `timeseries.get_range` with schema
"ohlcv-1d") -- the function signature and return shape stay the same, so
nothing in screening.py or screen_pairs.py needs to change.
"""
import pandas as pd
import yfinance as yf


def fetch_daily_closes(symbols: list[str], period: str = "2y") -> pd.DataFrame:
    """Returns a DataFrame of daily close prices, columns=symbols, index=date.
    Drops any symbol that fails to fetch or has too much missing data."""
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=True)
    if data.empty:
        return pd.DataFrame()
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.95))  # drop symbols missing >5% of bars
    return closes.dropna(how="any")  # align to common trading days only


def _fetch_market_cap(symbol: str, retries: int = 3) -> float | None:
    """Neither Databento nor a bulk yfinance download carries market cap --
    it's a per-ticker fundamentals lookup. Same retry/backoff shape as
    databento_source._fetch_dividends, since this hits the exact same
    Yahoo rate-limiting risk under bulk parallel load across ~500 symbols."""
    import time
    for attempt in range(retries):
        try:
            market_cap = yf.Ticker(symbol).fast_info["market_cap"]
        except Exception:
            market_cap = None
        if market_cap is not None:
            return float(market_cap)
        time.sleep(0.5 * (attempt + 1))
    return None


def fetch_market_caps_yfinance(symbols: list[str], max_workers: int = 5) -> dict[str, float]:
    """Market cap per symbol, for the "Market cap > $20 billion" screen. Kept
    to 5 parallel workers -- the same modest concurrency that fixed the
    dividend-fetch rate-limiting bug (2026-07-26: 16 workers tripped Yahoo's
    limiter on a ~470-symbol run and returned None for legitimate blue chips)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(lambda sym: (sym, _fetch_market_cap(sym)), symbols)
    return {symbol: cap for symbol, cap in results if cap is not None}


def fetch_volume_stats_yfinance(symbols: list[str], closes: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Average daily share/dollar volume per symbol, for the ADV/dollar-volume
    liquidity screen. Moved here from phase1/screen_pairs.py 2026-08-29 --
    that file was an offline, hand-run batch script; the client's own words:
    "there is nothing offline, that's phase by phase was only for payments,
    not code wise, all things should be wired, nothing should be offline."
    screen_pairs.py is gone, but this helper is still needed by both
    phase2/universe_screening.py's live intraday discovery and
    phase2/main.py's live 9:30am previous-session pair selection
    (select_pairs_from_previous_session), so it lives alongside
    fetch_market_caps_yfinance rather than with a script that no longer
    exists."""
    import yfinance as yf
    data = yf.download(symbols, period="3mo", auto_adjust=True, progress=False, threads=True)
    if data.empty:
        return {}
    vol = data["Volume"] if isinstance(data.columns, pd.MultiIndex) else data[["Volume"]]
    result = {}
    for sym in symbols:
        if sym not in vol.columns or sym not in closes.columns:
            continue
        avg_vol = vol[sym].mean()
        avg_price = closes[sym].mean()
        if pd.notna(avg_vol) and pd.notna(avg_price):
            result[sym] = {"avg_share_volume": float(avg_vol), "avg_dollar_volume": float(avg_vol * avg_price)}
    return result
