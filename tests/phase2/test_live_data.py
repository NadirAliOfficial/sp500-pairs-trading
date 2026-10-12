"""Unit tests for live_data.LiveBarBook -- the pure record-aggregation
logic, fed fake objects that duck-type the same shape as databento_dbn's
SymbolMappingMsg/OHLCVMsg. fetch_intraday_warmup and LiveBarFeed.start()
make real network/socket calls and are not exercised here."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.live_data import LiveBarBook


def _symbol_mapping(instrument_id, symbol):
    return SimpleNamespace(instrument_id=instrument_id, stype_in_symbol=symbol)


def _ohlcv(instrument_id, ts_event_ns, close):
    return SimpleNamespace(instrument_id=instrument_id, ts_event=ts_event_ns, pretty_close=close)


def test_bar_dropped_before_symbol_mapping_arrives():
    book = LiveBarBook()
    book.handle_record(_ohlcv(instrument_id=42, ts_event_ns=1_700_000_000_000_000_000, close=100.5))
    assert book.latest_close("TYL") is None
    print("PASS: OHLCV record before its symbol mapping is dropped, not misattributed")


def test_bar_routes_to_correct_symbol_after_mapping():
    book = LiveBarBook()
    book.handle_record(_symbol_mapping(instrument_id=42, symbol="TYL"))
    book.handle_record(_symbol_mapping(instrument_id=99, symbol="WDAY"))
    book.handle_record(_ohlcv(instrument_id=42, ts_event_ns=1_700_000_000_000_000_000, close=100.5))
    book.handle_record(_ohlcv(instrument_id=99, ts_event_ns=1_700_000_000_000_000_000, close=200.0))

    # max_age_seconds=None -- this test is about instrument_id routing, not
    # bar freshness, and these fixture timestamps are deliberately fixed.
    assert book.latest_close("TYL", max_age_seconds=None) == 100.5
    assert book.latest_close("WDAY", max_age_seconds=None) == 200.0
    print("PASS: bars route to the correct symbol via the instrument_id mapping")


def test_latest_close_tracks_most_recent_bar():
    book = LiveBarBook()
    book.handle_record(_symbol_mapping(instrument_id=1, symbol="AAA"))
    book.handle_record(_ohlcv(1, 1_700_000_000_000_000_000, 10.0))
    book.handle_record(_ohlcv(1, 1_700_000_060_000_000_000, 11.0))
    book.handle_record(_ohlcv(1, 1_700_000_120_000_000_000, 12.0))
    assert book.latest_close("AAA", max_age_seconds=None) == 12.0
    print("PASS: latest_close reflects the most recently received bar")


def test_latest_close_refuses_a_stale_bar():
    """Audit finding 2026-08-27: latest_close used to return the newest bar
    no matter how old, while its timestamp sat unused. With the feed's
    reconnect policy also defaulting to NONE, a dropped socket meant every
    later cycle read the SAME frozen price back and could place a real
    order off hours-old data, with nothing reporting a problem."""
    import time as _time
    book = LiveBarBook()
    book.handle_record(_symbol_mapping(instrument_id=1, symbol="AAA"))
    stale_ns = int((_time.time() - 3600) * 1_000_000_000)  # one hour old
    book.handle_record(_ohlcv(1, stale_ns, 10.0))
    assert book.latest_close("AAA") is None  # default guard rejects it
    assert book.latest_close("AAA", max_age_seconds=None) == 10.0  # opt-out still reads it
    print("PASS: latest_close returns None for a stale bar rather than a tradeable price")


def test_latest_close_accepts_a_fresh_bar():
    import time as _time
    book = LiveBarBook()
    book.handle_record(_symbol_mapping(instrument_id=1, symbol="AAA"))
    fresh_ns = int((_time.time() - 5) * 1_000_000_000)  # five seconds old
    book.handle_record(_ohlcv(1, fresh_ns, 10.0))
    assert book.latest_close("AAA") == 10.0
    print("PASS: latest_close returns a fresh bar normally")


def test_closes_series_builds_a_time_indexed_series():
    book = LiveBarBook()
    book.handle_record(_symbol_mapping(instrument_id=1, symbol="AAA"))
    book.handle_record(_ohlcv(1, 1_700_000_000_000_000_000, 10.0))
    book.handle_record(_ohlcv(1, 1_700_000_060_000_000_000, 11.0))
    series = book.closes_series("AAA")
    assert len(series) == 2
    assert list(series.values) == [10.0, 11.0]
    print("PASS: closes_series builds a correctly ordered time-indexed series")


def test_closes_series_empty_for_unknown_symbol():
    book = LiveBarBook()
    series = book.closes_series("NOPE")
    assert len(series) == 0
    print("PASS: closes_series returns an empty series for a symbol with no bars yet")




if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
