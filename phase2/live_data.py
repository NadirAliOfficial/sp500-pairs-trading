"""Databento intraday feed for Phase 2: a morning warm-up historical pull
(previous RTH session + today so far, 1-minute bars) to seed the divergence
engine before trading starts, plus a live db.Live subscription for ongoing
bars during the session.

Reuses databento_source.py's API-key handling, but NOT its PRIMARY_DATASETS --
today's/live data needs a separate license Databento only bundles for
EQUS.MINI on the current plan (XNAS.ITCH/XNYS.PILLAR's live data needs their
$1500/mo Plus plan), so this module has its own LIVE_DATASETS. Live streaming
itself can't be exercised in a test without a real Databento connection
during market hours -- LiveBarBook (the pure record-aggregation logic) is
unit tested by feeding it fake records that duck-type the same shape as
databento_dbn's SymbolMappingMsg/OHLCVMsg, so the parsing logic is verified
without needing the real C-extension types or a live socket.
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import databento as db
import pandas as pd

from phase1.databento_source import _client

logger = logging.getLogger("pairs_bot.live_data")

# Phase 1 screening deliberately uses XNAS.ITCH/XNYS.PILLAR (the real primary
# listing exchanges -- see databento_source.py's own comment on why a
# consolidated feed gave less accurate results there). Live/today's data is a
# separate problem: those two datasets require Databento's $1500/mo Plus plan
# for live access, which is why Phase 2's live feed and warm-up (both need
# TODAY's data, not just history) use EQUS.MINI instead -- a consolidated feed
# that's included for live data on the existing Standard plan. Accuracy is a
# non-issue here since this only drives intraday divergence detection, not
# the pair-selection statistics Phase 1's accuracy comment was about.
LIVE_DATASETS = ["EQUS.MINI"]

# How old the newest 1-minute bar may be before latest_close() refuses to
# report it as a tradeable price. Bars arrive every minute per symbol and
# the bot's own check cycle is 60s, so 5 missed bars on a large-cap S&P 500
# name during RTH is well outside normal jitter and means the feed is
# genuinely not delivering. Deliberately not tighter: a false positive here
# blocks trading entirely, which is its own failure -- this is sized to
# catch a dead feed, not a thin minute.
MAX_BAR_AGE_SECONDS = 300


def _available_end(client, dataset: str) -> str | None:
    """The last timestamp the dataset actually holds, as a naive ISO string
    matching the format the warm-up window is built in. None if the metadata
    call fails, in which case the caller sends the request unclamped rather
    than skipping the fetch."""
    try:
        end = client.metadata.get_dataset_range(dataset=dataset).get("end")
    except Exception as e:
        logger.warning(f"could not read {dataset} available range: {e}")
        return None
    if not end:
        return None
    return str(end).replace("Z", "").split(".")[0]


def fetch_intraday_warmup(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """1-minute historical bars for the warm-up window (previous RTH session
    + today so far), used to seed the divergence engine before live trading
    begins for the day. Also reused (with a different start/end) by
    main.py's finalize_trigger_at_931 to fetch today's premarket bars."""
    client = _client()
    frames = []
    for dataset in LIVE_DATASETS:
        try:
            # Clamp to what the dataset actually holds. Asking for even a
            # minute past its available end makes Databento reject the WHOLE
            # request with 422 data_end_after_available_end -- it does not
            # return the bars it does have. Since this window always ends at
            # "now" and the feed sits a few minutes behind real time, the
            # unclamped request failed every morning, and the warm-up silently
            # degraded to "no data" while the data was in fact there.
            available_end = _available_end(client, dataset)
            capped_end = min(end, available_end) if available_end else end
            if capped_end != end:
                logger.info(f"{dataset} warm-up end clamped {end} -> {capped_end} (dataset boundary)")
            data = client.timeseries.get_range(
                dataset=dataset, symbols=symbols, schema="ohlcv-1m",
                start=start, end=capped_end, stype_in="raw_symbol",
            )
            df = data.to_df().reset_index()
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.warning(f"{dataset} intraday warmup fetch failed: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined.drop_duplicates(subset=["ts_event", "symbol"], keep="first")


class LiveBarBook:
    """Pure aggregation: turns raw live records into a per-symbol 1-minute
    close-price history. Duck-types on attribute presence rather than
    isinstance against databento_dbn's C-extension classes, so tests can feed
    it plain fake objects instead of needing a live connection."""

    def __init__(self):
        self._instrument_to_symbol: dict[int, str] = {}
        self.bars: dict[str, list[tuple[pd.Timestamp, float]]] = {}

    def handle_record(self, record) -> None:
        if hasattr(record, "stype_in_symbol"):  # SymbolMappingMsg
            self._instrument_to_symbol[record.instrument_id] = record.stype_in_symbol
        elif hasattr(record, "pretty_close"):  # OHLCVMsg
            symbol = self._instrument_to_symbol.get(record.instrument_id)
            if symbol is None:
                return  # mapping not received yet -- drop rather than misattribute
            ts = pd.Timestamp(record.ts_event, unit="ns", tz="UTC")
            self.bars.setdefault(symbol, []).append((ts, record.pretty_close))

    def latest_close(self, symbol: str, max_age_seconds: float = MAX_BAR_AGE_SECONDS) -> float | None:
        """The most recent close, or None if it's too old to trade on.

        The staleness check is the point: this used to return the last bar
        received no matter how old it was, while the bar's own timestamp
        sat unused in history[-1][0]. Combined with the feed's (previously)
        non-existent reconnect, a dropped socket meant every subsequent
        cycle read the SAME frozen price back, appended it to
        price_history stamped with a fresh `now`, and flatlined the spread
        -- with a real market order possible off data hours out of date,
        and nothing anywhere reporting a problem. Returning None instead
        makes main.py's existing `if price_a is None` check skip the pair,
        which is the correct fail-closed behavior.

        Pass max_age_seconds=None to opt out (the post-session report and
        other non-trading readers, where a last-known price is fine)."""
        history = self.bars.get(symbol)
        if not history:
            return None
        ts, close = history[-1]
        if max_age_seconds is None:
            return close
        age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds()
        if age > max_age_seconds:
            logger.warning(f"{symbol}: latest bar is {age:.0f}s old (>{max_age_seconds:.0f}s) -- "
                            f"treating as no price rather than trading on stale data")
            return None
        return close

    def closes_series(self, symbol: str) -> pd.Series:
        history = self.bars.get(symbol, [])
        if not history:
            return pd.Series(dtype=float)
        timestamps, closes = zip(*history)
        return pd.Series(closes, index=pd.DatetimeIndex(timestamps))


class LiveBarFeed:
    """One db.Live connection per dataset (SDK constraint: a single
    connection can only subscribe to one dataset), routing all records into
    a shared LiveBarBook."""

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.book = LiveBarBook()
        self._clients: list[db.Live] = []

    def start(self) -> None:
        api_key = os.getenv("DATABENTO_API_KEY")
        for dataset in LIVE_DATASETS:
            # reconnect_policy is explicit -- the SDK's own default is
            # ReconnectPolicy.NONE (verified against databento 0.82.0), so
            # without this a single dropped socket kills the price feed for
            # the rest of the session with no error and no retry. The bot
            # would keep polling latest_close() and get the last bar
            # received before the drop, forever (see LiveBarBook's own
            # staleness guard, which is the second half of this fix).
            client = db.Live(key=api_key, reconnect_policy="reconnect")
            client.subscribe(dataset=dataset, schema="ohlcv-1m",
                              symbols=self.symbols, stype_in="raw_symbol")
            client.add_callback(self.book.handle_record)
            client.start()
            self._clients.append(client)

    def add_symbols(self, symbols: list[str]) -> None:
        """Extends every already-running dataset connection with new symbols
        -- Live.subscribe() is documented as "add a new subscription to the
        session", so this doesn't need to stop/restart the feed. Used when
        universe screening (phase2/universe_screening.py) finds a new pair
        mid-session that isn't already being tracked."""
        new = [s for s in symbols if s not in self.symbols]
        if not new:
            return
        self.symbols.extend(new)
        for dataset, client in zip(LIVE_DATASETS, self._clients):
            client.subscribe(dataset=dataset, schema="ohlcv-1m", symbols=new, stype_in="raw_symbol")

    def stop(self) -> None:
        for client in self._clients:
            client.stop()
