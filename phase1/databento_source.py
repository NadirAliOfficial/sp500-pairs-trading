"""Databento historical data fetch -- the production data source per spec,
replacing the yfinance placeholder once validated to give consistent results.

First attempt used DBEQ.BASIC (this key's lowest-cost tier), but that only
reports OHLCV from 4 secondary venues (NYSE Texas, NYSE National, IEX, MIAX
Pearl) -- not the primary listing exchange where most volume and official
closing prices are actually set. That gave meaningfully different, less
accurate results than yfinance (one qualified pair flipped to unqualified).

This key also has access to XNAS.ITCH (Nasdaq) and XNYS.PILLAR (NYSE), the
actual primary listing exchanges -- querying both and merging (each symbol
only has data on whichever one is its real listing venue) gives one clean,
authoritative daily bar per symbol per day, matching yfinance closely.

Remaining gap: these are raw (unadjusted) prices. Dividend-paying stocks show
an artificial price drop on each ex-dividend date that isn't a real economic
move -- for dividend-heavy sectors (utilities, REITs, banks, exactly where
this strategy is most likely to find valid pairs) that noise can be enough to
flip a cointegration test result (confirmed: LNT/AEP qualified on adjusted
yfinance prices, failed on Databento's raw prices). Databento's own datasets
have no dividends/corporate-actions schema on this account, so dividend
history is pulled from yfinance (free, the same source everyone uses for
this) and the standard backward price-adjustment is applied here, giving
Databento's real exchange prices the same dividend-adjustment yfinance
already does automatically.
"""
import os
import re
import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import databento as db
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")

load_dotenv()

PRIMARY_DATASETS = ["XNAS.ITCH", "XNYS.PILLAR"]

# Consolidated tape (CTA/UTP), used for the liquidity thresholds only.
#
# PRIMARY_DATASETS are direct exchange feeds: XNAS.ITCH sees only what trades
# on Nasdaq, XNYS.PILLAR only what trades on NYSE. Prices are near enough
# identical across venues, so those feeds are fine for the statistical tests,
# but VOLUME from a single venue is a small fraction of what a symbol actually
# trades -- measured against the tape, JPM showed 499k shares/day against a
# real 9.6M (19x low) and BAC 1.9M against 33.7M. That silently failed the
# "ADV > 10M shares OR $1B/day" screen for almost the whole universe and cut
# the qualified list from 7 pairs to 1.
#
# EQUS.SUMMARY is the consolidated daily summary and matches the tape. The
# other candidates on this plan are partial and were measured, not assumed:
# EQUS.MINI covers 3-7% of consolidated volume and XNAS.BASIC 24-68%, so
# neither can be used for a volume threshold.
CONSOLIDATED_DAILY_DATASET = "EQUS.SUMMARY"


def _client() -> db.Historical:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set (check .env)")
    return db.Historical(api_key)


def latest_available_date() -> str:
    """Real available end date for these datasets, per Databento's own
    metadata -- NOT date.today(), which silently produced a request for
    2026-07-26 when the dataset's actual data only goes up to 2026-07-25.
    That one-day-out-of-range request raised a BentoClientError that was
    (wrongly) being swallowed by a blanket except-clause below, which is
    why the very first production run reported 0 symbols with no visible
    error at all. Fixed by querying the real boundary instead of guessing."""
    client = _client()
    ends = [client.metadata.get_dataset_range(dataset=ds)["end"] for ds in PRIMARY_DATASETS]
    return min(pd.Timestamp(e) for e in ends).strftime("%Y-%m-%d")


_SCHEMA_NOT_AVAILABLE_END = re.compile(
    r"only available between .+ and (\d{4}-\d{2}-\d{2})T")


def _fetch_one_dataset(client: db.Historical, dataset: str, symbols: list[str],
                        start: str, end: str, schema: str) -> pd.DataFrame:
    """One dataset's worth of _fetch_raw, split out so retrying (network
    errors, and the schema-boundary case below) doesn't need nested loops
    with confusing control flow -- a plain function lets each retry path
    just `return` instead.

    Retries a transient network error (BentoError, read timeouts on the
    streaming connection, seen in production on a ~470-symbol request) up
    to 3 times. Separately, retries ONCE more with a clamped `end` if
    Databento reports the schema's real boundary is earlier than what was
    requested -- confirmed live 2026-08-31: requesting `end` as today's
    date against EQUS.SUMMARY (whose ohlcv-1d schema lags "now" by more
    than a day, not just excluding today) got a 422
    data_schema_not_fully_available, which the old code's plain
    except-clause turned into a silent empty result for that dataset
    rather than a visible error -- on the very first live run of
    select_pairs_from_previous_session, that disqualified all 124 Market
    Entry List pairs with "no daily close data" instead of the real cause.
    client.metadata.get_dataset_range(dataset=dataset) (the same
    pre-emptive clamp live_data.py's own _available_end does for the
    intraday warm-up fetch) does NOT work here -- it returned a boundary
    later than what the ohlcv-1d schema specifically supports
    (get_dataset_range has no per-schema parameter), so a request clamped
    to it still 422'd. Parsing Databento's own error message for the
    schema's real boundary and retrying once is the only reliable fix
    found."""
    import time
    capped_end = end
    for schema_attempt in range(2):
        for attempt in range(3):
            try:
                data = client.timeseries.get_range(
                    dataset=dataset, symbols=symbols, schema=schema,
                    start=start, end=capped_end, stype_in="raw_symbol",
                )
                return data.to_df().reset_index()  # ts_event comes back as the index; make it a column for concat/pivot
            except db.common.error.BentoClientError as e:
                match = _SCHEMA_NOT_AVAILABLE_END.search(str(e))
                if match and schema_attempt == 0 and match.group(1) < capped_end:
                    capped_end = match.group(1)
                    print(f"  {dataset}/{schema} end clamped to {capped_end} "
                          f"(schema's real available range)", file=sys.stderr)
                    break  # retry this dataset once with the real boundary
                # Symbols simply not listed on an exchange return an EMPTY
                # result (handled by the caller's `if not df.empty` check),
                # not an exception -- so if we get here, it's a REAL problem
                # (bad date range, auth issue, etc.). Surface it instead of
                # silently continuing, which previously masked a real bug
                # for an entire production run (0 results, no visible error).
                print(f"  WARNING: {dataset} request failed: {e}", file=sys.stderr)
                return pd.DataFrame()
            except db.common.error.BentoError as e:
                print(f"  {dataset} attempt {attempt+1}/3 failed ({e}), retrying...", file=sys.stderr)
                if attempt == 2:
                    print(f"  WARNING: {dataset} failed after 3 attempts, continuing without it", file=sys.stderr)
                else:
                    time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def _fetch_raw(symbols: list[str], start: str, end: str, schema: str = "ohlcv-1d",
                datasets: list[str] | None = None) -> pd.DataFrame:
    client = _client()
    frames = []
    for dataset in (datasets or PRIMARY_DATASETS):
        df = _fetch_one_dataset(client, dataset, symbols, start, end, schema)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined.drop_duplicates(subset=["ts_event", "symbol"], keep="first")


def _fetch_dividends(symbol: str, retries: int = 3):
    """Yahoo rate-limits aggressively under bulk parallel load (found running
    ~470 symbols at once: legitimate blue chips like NVDA/PG/PFE/ORCL came
    back as None, not an exception). Retry with backoff before giving up and
    falling back to unadjusted prices for that symbol."""
    import time
    for attempt in range(retries):
        try:
            divs = yf.Ticker(symbol).dividends
        except Exception:
            divs = None
        if divs is not None:
            return divs
        time.sleep(0.5 * (attempt + 1))
    return None


def _dividend_adjust(closes: pd.Series, symbol: str) -> pd.Series:
    """Standard backward price adjustment: for each ex-dividend date (most
    recent first), multiply every price strictly before that date by
    (1 - dividend / prior_close). This is the same method yfinance's
    auto_adjust=True uses, applied here to Databento's raw prices.

    Processing newest-to-oldest and mutating `adjusted` in place means each
    older dividend's ratio is computed against a price level that already
    reflects every later adjustment -- keeping the whole series expressed
    consistently in "today's dollars", same as standard practice."""
    divs = _fetch_dividends(symbol)
    if divs is None or divs.empty:
        return closes  # no dividend data available (or still rate-limited after retries) -- fall back to raw
    if divs.index.tz is not None:
        divs = divs.tz_localize(None)

    idx = pd.to_datetime(closes.index)
    adjusted = closes.copy()

    for ex_date, amount in divs.sort_index(ascending=False).items():
        ex_date = ex_date.normalize()
        n_prior = int((idx < ex_date).sum())
        if n_prior == 0:
            continue
        prior_close = adjusted.iloc[n_prior - 1]  # last close before the ex-dividend date
        if prior_close <= 0:
            continue
        ratio = 1 - (float(amount) / prior_close)
        if ratio <= 0:
            continue
        adjusted.iloc[:n_prior] *= ratio
    return adjusted


def fetch_daily_closes_databento(symbols: list[str], start: str, end: str, adjust_dividends: bool = True,
                                  datasets: list[str] | None = None) -> pd.DataFrame:
    """Returns a DataFrame of daily close prices, columns=symbols, index=date --
    same shape as data_source.fetch_daily_closes() so it's a drop-in swap.

    `datasets` defaults to the primary exchange feeds; pass
    [CONSOLIDATED_DAILY_DATASET] for the client's "RTH Consolidated tape,
    only" requirement on the Pearson / Correlation Breakdown tests."""
    raw = _fetch_raw(symbols, start, end, datasets=datasets)
    if raw.empty:
        return pd.DataFrame()
    closes = raw.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
    closes.index = closes.index.date
    # drop SYMBOLS missing too much history first (recent IPOs/spinoffs like
    # SNDK, XYZ, EXE, SW, GEV don't have the full 2y window) -- doing a
    # straight dropna(how="any") across all ~470 columns at once meant even
    # ONE such symbol wiped out that entire day for every other symbol too,
    # collapsing 501 days down to 105 (2026-07-26 finding: this variant
    # zeroed out the whole run). Same two-step pattern as data_source.py's
    # yfinance fetch: drop sparse columns, then align dates on what's left.
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.95))
    closes = closes.dropna(how="any")
    if adjust_dividends:
        from concurrent.futures import ThreadPoolExecutor
        # dividend fetches are one network call per symbol -- sequential would
        # be very slow across ~500 symbols, and the computation is I/O bound
        # (waiting on yfinance), so threads are a safe speedup. Kept modest
        # (was 16) since that was aggressive enough to trip Yahoo's rate
        # limiting on a ~470-symbol run; _fetch_dividends still retries with
        # backoff as a second line of defense.
        with ThreadPoolExecutor(max_workers=5) as pool:
            adjusted_cols = pool.map(lambda sym: (sym, _dividend_adjust(closes[sym], sym)), closes.columns)
        for sym, series in adjusted_cols:
            closes[sym] = series
    return closes


def fetch_consolidated_daily_closes(symbols: list[str], start: str, end: str,
                                     adjust_dividends: bool = True) -> pd.DataFrame:
    """Daily closes from the true consolidated tape (EQUS.SUMMARY), per the
    client's 2026-08-25 spec: "Data: RTH Consolidated tape, only." Feeds the
    Pearson correlation (60 day) and Correlation Breakdown (20 day) tests.
    A daily bar's close is already the RTH closing print/auction, so this
    needs no separate RTH filtering the way the 1-minute fetch does."""
    return fetch_daily_closes_databento(
        symbols, start, end, adjust_dividends=adjust_dividends, datasets=[CONSOLIDATED_DAILY_DATASET])


RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)


def _rth_only(raw: pd.DataFrame) -> pd.DataFrame:
    """Keeps only Regular Trading Hours bars, per spec: "Extended-hours
    (pre-market and after-hours) data shall be excluded from all
    calculations." ts_event is UTC, so convert to ET before filtering --
    the RTH boundary is a wall-clock time, and the UTC offset shifts with
    daylight saving."""
    ts = pd.to_datetime(raw["ts_event"], utc=True).dt.tz_convert(ET)
    within = (ts.dt.time >= RTH_OPEN) & (ts.dt.time < RTH_CLOSE)
    out = raw.loc[within].copy()
    out["session"] = ts.loc[within].dt.date
    return out


SYMBOL_CHUNK = 60


def _fetch_intraday_rth(symbols: list[str], start: str, end: str, sessions: int,
                         datasets: list[str] | None = None) -> pd.DataFrame:
    """Fetches 1-minute RTH bars in symbol chunks, then trims to the most
    recent `sessions` sessions ONCE, globally, across every chunk.

    Chunked deliberately. Asking for ~470 symbols of 1-minute data in one
    request materialises millions of rows at once, which on a 2.8GB VPS
    starved sshd badly enough that the box stopped accepting logins.

    The session trim itself has to run AFTER every chunk is collected, not
    per chunk -- fixed 2026-08-24 per the independent audit's Finding #9,
    verified real: computing "the last N sessions" from only one chunk's
    own symbols risked a different date list per chunk if a chunk's fetch
    happened to lag or gap on its most recent session (a partial Databento
    outage, a chunk-specific delay) while other chunks didn't. The
    resulting closes matrix would then carry DIFFERENT calendar dates
    under the same row index across symbol columns -- silently corrupting
    every cross-symbol calculation (correlation, hedge ratio, spread) that
    assumes one row is one real moment in time for every column."""
    frames = []
    for i in range(0, len(symbols), SYMBOL_CHUNK):
        chunk = symbols[i:i + SYMBOL_CHUNK]
        raw = _fetch_raw(chunk, start, end, schema="ohlcv-1m", datasets=datasets)
        if raw.empty:
            continue
        raw = _rth_only(raw)
        if raw.empty:
            continue
        frames.append(raw)
        print(f"    fetched {min(i + SYMBOL_CHUNK, len(symbols))}/{len(symbols)} symbols", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    keep = sorted(combined["session"].unique())[-sessions:]
    return combined[combined["session"].isin(keep)]


def fetch_intraday_closes_databento(
    symbols: list[str], start: str, end: str, sessions: int = 2,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """1-minute RTH closes, columns=symbols, index=timestamp -- the
    resolution the spec actually calls for ("Pulling 1-minute historical
    OHLCV bars across all S&P 500 components"), and the basis on which
    half-life is measured in MINUTES rather than days.

    Trimmed to the most recent `sessions` RTH sessions, which for the
    spec's default of 2 is "the previous RTH session together with the
    current RTH session". No dividend adjustment: an ex-dividend gap sits
    between sessions, not inside one, so it can't distort an intraday
    spread the way it distorted the daily series.

    `datasets` defaults to PRIMARY_DATASETS (XNAS.ITCH/XNYS.PILLAR) via
    _fetch_raw's own default when None -- pass an override (e.g.
    ["EQUS.MINI"], live_data.py's own consolidated-SIP-style feed) for a
    caller that can't reach the primary exchange feeds. See
    main.py._fetch_johansen_hurst_history's own comment, 2026-09-16: the
    client's account has no live-data license for XNAS.ITCH/XNYS.PILLAR
    as of 2026-09-15, confirmed via a real 403 license_not_found_unauthorized."""
    raw = _fetch_intraday_rth(symbols, start, end, sessions, datasets=datasets)
    if raw.empty:
        return pd.DataFrame()

    closes = raw.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
    # Same two-step as the daily fetch: drop sparse SYMBOLS first, otherwise a
    # single thinly-traded name with gaps wipes out those minutes for every
    # other symbol when the rows are aligned.
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.95))
    return closes.dropna(how="any")


def fetch_intraday_screening_data(
    symbols: list[str], start: str, end: str, sessions: int = 2,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """One fetch, both outputs the screen needs: the 1-minute close matrix for
    the statistical tests, and per-session volume aggregates for the liquidity
    filter. Calling the two separately downloads the same heavy 1-minute data
    twice."""
    raw = _fetch_intraday_rth(symbols, start, end, sessions)
    if raw.empty:
        return pd.DataFrame(), {}

    closes = raw.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
    # Drop symbols that barely print, but deliberately DO NOT dropna across
    # rows here. At 1-minute resolution a thinly traded name simply has no
    # bar in some minutes, so requiring every one of ~375 symbols to have a
    # bar at every timestamp collapsed 780 bars to 136 and left the whole
    # screen with too little data to test anything (0 pairs qualified).
    # Callers align each PAIR on its own two columns instead, which gives
    # every pair the maximum overlap it actually has.
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.80))

    return closes, fetch_consolidated_volume_stats(symbols, start, end)


def fetch_consolidated_volume_stats(
    symbols: list[str], start: str, end: str,
) -> dict[str, dict[str, float]]:
    """Average daily share and dollar volume from the consolidated tape, for
    the "ADV > 10M shares OR dollar volume > $1B/day" liquidity check.

    Deliberately a separate fetch from the 1-minute price data rather than
    aggregating those bars: the 1-minute bars come from single-venue feeds and
    carry only that venue's share of the volume. See
    CONSOLIDATED_DAILY_DATASET."""
    client = _client()
    frames = []
    for i in range(0, len(symbols), SYMBOL_CHUNK):
        chunk = symbols[i:i + SYMBOL_CHUNK]
        try:
            df = client.timeseries.get_range(
                dataset=CONSOLIDATED_DAILY_DATASET, schema="ohlcv-1d",
                symbols=chunk, start=start, end=end,
            ).to_df()
        except Exception as e:
            print(f"    consolidated volume fetch failed for chunk {i}: {e}", file=sys.stderr)
            continue
        if not df.empty:
            frames.append(df[["symbol", "close", "volume"]])
    if not frames:
        return {}

    raw = pd.concat(frames)
    raw["dollar_vol"] = raw["close"] * raw["volume"]
    averaged = raw.groupby("symbol").agg(
        avg_share_volume=("volume", "mean"), avg_dollar_volume=("dollar_vol", "mean"))
    return averaged.to_dict(orient="index")


def fetch_volume_stats_intraday_databento(
    symbols: list[str], start: str, end: str, sessions: int = 2,
) -> dict[str, dict[str, float]]:
    """Average DAILY share and dollar volume, derived from 1-minute RTH bars:
    sum each session, then average across sessions. The thresholds
    ("ADV > 10M shares OR dollar volume > $1B/day") are per-day figures, so
    the per-minute bars have to be aggregated up to a session before they
    mean anything."""
    raw = _fetch_intraday_rth(symbols, start, end, sessions)
    if raw.empty:
        return {}
    raw["dollar_vol"] = raw["close"] * raw["volume"]

    per_session = raw.groupby(["symbol", "session"]).agg(
        share_volume=("volume", "sum"), dollar_volume=("dollar_vol", "sum"))
    averaged = per_session.groupby("symbol").mean()
    averaged.columns = ["avg_share_volume", "avg_dollar_volume"]
    return averaged.to_dict(orient="index")


def fetch_volume_stats_databento(symbols: list[str], start: str, end: str) -> dict[str, dict[str, float]]:
    """Average daily share volume AND dollar volume per symbol, for the
    "ADV > 10M shares OR dollar volume > $1B/day" liquidity check -- both
    computed from one _fetch_raw() call rather than fetching the same raw
    OHLCV data twice."""
    raw = _fetch_raw(symbols, start, end)
    if raw.empty:
        return {}
    raw["dollar_vol"] = raw["close"] * raw["volume"]
    grouped = raw.groupby("symbol").agg(avg_share_volume=("volume", "mean"), avg_dollar_volume=("dollar_vol", "mean"))
    return grouped.to_dict(orient="index")
