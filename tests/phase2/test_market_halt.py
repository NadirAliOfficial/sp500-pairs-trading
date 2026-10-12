"""Tests for the market-wide halt: a crash or extraordinary rally in the index
excludes the whole S&P 500 for 24 hours.

This is the widest lever in the system -- one yes stops every pair trading for
a day -- so the persistence, expiry and failure behaviour all get pinned here.
The Gemini call itself is not tested (real network); run_market_wide_scan is
exercised with the scan function stubbed.
"""
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import phase2.news_blacklist as nb
from phase2.news_blacklist import MarketHalt, MarketWideEvent, run_market_wide_scan


def _halt() -> MarketHalt:
    return MarketHalt(path=Path(tempfile.mkdtemp()) / "market_halt.json")


def _event(crashed=False, rallied=False, move=0.0) -> MarketWideEvent:
    return MarketWideEvent(crashed=crashed, rallied_extraordinarily=rallied,
                           index_move_pct=move, reason="test")


def _with_scan(result, fn, *args):
    """Runs fn with scan_market_wide_event stubbed. `result` is either a
    MarketWideEvent or an exception instance to raise."""
    real = nb.scan_market_wide_event

    def stub(*_a, **_k):
        if isinstance(result, Exception):
            raise result
        return result

    nb.scan_market_wide_event = stub
    try:
        return fn(*args)
    finally:
        nb.scan_market_wide_event = real


def test_threshold_is_applied_to_the_number_not_the_models_opinion():
    """Now that "crashed" means a specific figure, the decision is arithmetic.
    The model is used to look up the index move, not to judge whether it
    counts, so a wrong verdict alongside a right number must not halt."""
    halt = _halt()
    wrong_verdict = MarketWideEvent(crashed=True, rallied_extraordinarily=False,
                                    index_move_pct=-0.4, reason="model overreacted")
    assert _with_scan(wrong_verdict, run_market_wide_scan, halt) is False
    assert not halt.is_active(), "-0.4% is inside the threshold, whatever the model says"

    halt2 = _halt()
    missed = MarketWideEvent(crashed=False, rallied_extraordinarily=False,
                             index_move_pct=-2.1, reason="model understated it")
    assert _with_scan(missed, run_market_wide_scan, halt2) is True
    assert halt2.is_active(), "-2.1% breaches the threshold, whatever the model says"
    print("PASS: the halt follows the reported number, not the model's verdict")


def test_threshold_boundaries():
    for move, should_halt in [(-1.6, True), (-1.5, True), (-1.49, False), (0.0, False),
                              (1.49, False), (1.5, True), (1.6, True)]:
        halt = _halt()
        _with_scan(_event(move=move), run_market_wide_scan, halt)
        assert halt.is_active() is should_halt, f"{move}% should {'halt' if should_halt else 'not halt'}"
    print("PASS: halts at -1.5% and +1.5% inclusive, not inside them")


def test_no_halt_on_an_ordinary_day():
    halt = _halt()
    assert _with_scan(_event(move=0.3), run_market_wide_scan, halt) is False
    assert not halt.is_active()
    print("PASS: an ordinary session raises no halt")


def test_crash_halts_everything_for_24_hours():
    halt = _halt()
    assert _with_scan(_event(crashed=True, move=-6.2), run_market_wide_scan, halt) is True
    assert halt.is_active()

    now = datetime.now(timezone.utc)
    assert halt.is_active(now + timedelta(hours=23))
    assert not halt.is_active(now + timedelta(hours=25)), "must lapse after 24 hours"
    print("PASS: a crash halts trading and the halt lapses after 24 hours")


def test_extraordinary_rally_halts_too():
    halt = _halt()
    assert _with_scan(_event(rallied=True, move=5.8), run_market_wide_scan, halt) is True
    assert halt.is_active()
    print("PASS: an extraordinary rally halts trading as well as a crash")


def test_halt_survives_a_restart():
    """Held on disk, not in memory: a restart must not resume trading into the
    same crash the halt was raised for."""
    halt = _halt()
    _with_scan(_event(crashed=True, move=-7.0), run_market_wide_scan, halt)
    assert MarketHalt(path=halt.path).is_active(), "a fresh instance must see the halt"
    print("PASS: the halt persists across a restart")


def test_failed_scan_leaves_the_state_unchanged():
    """Neither direction is safe as a default. Halting on error would let one
    Gemini outage stop trading for a day, and this project has already seen
    transient 503s. Clearing on error would silently drop a risk control."""
    halt = _halt()
    assert _with_scan(RuntimeError("gemini down"), run_market_wide_scan, halt) is False
    assert not halt.is_active(), "a failed scan must not invent a halt"

    _with_scan(_event(crashed=True, move=-6.0), run_market_wide_scan, halt)
    assert _with_scan(RuntimeError("gemini down"), run_market_wide_scan, halt) is True
    assert halt.is_active(), "a failed scan must not clear a halt already raised"
    print("PASS: a failed scan changes nothing in either direction")


def test_reason_and_move_are_recorded_for_the_dashboard():
    halt = _halt()
    _with_scan(_event(crashed=True, move=-6.25), run_market_wide_scan, halt)
    details = halt.details()
    assert details is not None
    assert "-6.25%" in details["reason"] and "crash" in details["reason"]
    assert "expires_at" in details and "raised_at" in details
    print("PASS: the halt records why it fired and what the index move was")


def test_details_are_none_once_the_halt_lapses():
    halt = _halt()
    _with_scan(_event(crashed=True, move=-6.0), run_market_wide_scan, halt)
    assert halt.details(datetime.now(timezone.utc) + timedelta(hours=25)) is None
    print("PASS: a lapsed halt reports nothing rather than stale details")


def test_missing_or_corrupt_file_is_not_a_halt():
    """Failing closed here would block trading with no way to see why."""
    halt = _halt()
    assert not halt.is_active(), "no file means no halt"
    halt.path.parent.mkdir(parents=True, exist_ok=True)
    halt.path.write_text("{ not json")
    assert not halt.is_active()
    print("PASS: a missing or unreadable halt file does not block trading")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
