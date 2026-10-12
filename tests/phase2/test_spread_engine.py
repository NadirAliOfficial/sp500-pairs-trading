"""Unit tests for spread_engine.py, using synthetic 1-minute series with
hand-constructed, known divergence events so results can be verified by hand
-- same approach as tests/test_screening.py used for Phase 1."""
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import time as dtime

from phase2.spread_engine import (passes_divergence_condition, widest_three_mean,
    divergence_gap, allowed_divergence_gap, passes_gap_condition,
    DivergenceEvent, compute_spread, equilibrium_spread, detect_divergence_events,
    divergence_trigger_level, detect_non_reverting_rth_events, reverted_after_peak,
    reverted_more_than_pct_after_peak,
)

RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)


def _build_deviation_series() -> tuple[pd.Series, pd.Series]:
    """300 one-minute bars containing exactly two qualifying divergence events
    (close within 2 hours) and two disqualified ones (one takes 130 minutes to
    close, one never closes). Equilibrium is held at a flat 0 so spread values
    ARE the deviations directly, making expected results easy to hand-verify."""
    index = pd.date_range("2026-07-27 09:30", periods=300, freq="1min")
    values = [0.0] * 300

    # Event A (qualifies): opens idx10, peaks 0.5 at idx15, closes idx20 -> 10 min
    event_a = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    values[10:21] = event_a

    # Event B (qualifies): opens idx30, peaks 0.2 at idx32, closes idx34 -> 4 min
    event_b = [0.05, 0.15, 0.2, 0.1, 0.0]
    values[30:35] = event_b

    # Event C (disqualified: too slow): opens idx50, holds at 0.4, closes idx180 -> 130 min
    values[50] = 0.05
    values[51:180] = [0.4] * 129
    values[180] = 0.0

    # Event D (disqualified: never closes): opens idx200, holds at 0.6 through end of series
    values[200] = 0.05
    values[201:300] = [0.6] * 99

    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)
    return spread, equilibrium


def test_detect_divergence_events_finds_only_the_two_qualifying_events():
    spread, equilibrium = _build_deviation_series()
    events = detect_divergence_events(spread, equilibrium, tolerance=0.01)

    assert len(events) == 2, "the too-slow (130min) and never-closes events must be excluded"

    a, b = events
    assert a.max_spread == 0.5
    assert a.max_adverse_excursion == 0.5 - 0.05
    assert a.duration == timedelta(minutes=10)
    assert a.open_time == spread.index[10]
    assert a.close_time == spread.index[20]

    assert b.max_spread == 0.2
    assert b.max_adverse_excursion == 0.2 - 0.05
    assert b.duration == timedelta(minutes=4)
    print("PASS: detect_divergence_events finds exactly the two qualifying events with correct stats")


def test_event_at_exactly_two_hours_is_included():
    index = pd.date_range("2026-07-27 09:30", periods=121, freq="1min")
    values = [0.0] * 121
    values[0] = 0.05
    values[1:120] = [0.3] * 119
    values[120] = 0.0  # closes at minute 120 -> duration exactly 120min == 2 hours
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    events = detect_divergence_events(spread, equilibrium, tolerance=0.01)
    assert len(events) == 1
    assert events[0].duration == timedelta(hours=2)
    print("PASS: event closing at exactly 2 hours is included")


def test_event_one_minute_over_two_hours_is_excluded():
    index = pd.date_range("2026-07-27 09:30", periods=122, freq="1min")
    values = [0.0] * 122
    values[0] = 0.05
    values[1:121] = [0.3] * 120
    values[121] = 0.0  # closes at minute 121 -> duration 121min > 2 hours
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    events = detect_divergence_events(spread, equilibrium, tolerance=0.01)
    assert events == []
    print("PASS: event taking 121 minutes to close is correctly excluded")


def test_two_events_are_selectable_now_the_condition_is_off():
    """The client removed the divergence condition on 2026-08-03, so any pair
    with a qualifying event is selectable and the trigger is its maximum.
    The condition itself is still exercised below via an explicit min_count,
    so reinstating it is a one-line change."""
    spread, equilibrium = _build_deviation_series()
    events = detect_divergence_events(spread, equilibrium, tolerance=0.01)
    assert len(events) == 2
    assert passes_divergence_condition(events)
    assert divergence_trigger_level(events) == 0.5  # the larger of 0.5 / 0.2

    # with the condition reinstated, two events have no three-widest mean to
    # be judged against and the pair is rejected again
    assert widest_three_mean(events) is None
    assert not passes_divergence_condition(events, min_count=2)
    print("PASS: 2 events are selectable with the condition off, rejected with it on")


def test_detect_non_reverting_rth_events_keeps_only_the_still_open_event():
    """Mirror image of detect_divergence_events on the same fixture: events A,
    B (revert quickly) and C (reverts, just slowly at 130min) are ALL
    excluded here -- they reverted, full stop, duration is irrelevant to
    this function. Only D (opens idx200, never reverts) counts."""
    spread, equilibrium = _build_deviation_series()
    events = detect_non_reverting_rth_events(spread, equilibrium, RTH_OPEN, RTH_CLOSE, tolerance=0.01)

    assert len(events) == 1
    event = events[0]
    assert event.max_spread == 0.6
    assert event.open_time == spread.index[200]
    assert event.close_time == spread.index[-1]
    print("PASS: only the never-reverting event D counts, A/B/C (all of which eventually revert) are excluded")


def test_detect_non_reverting_rth_events_excludes_an_event_opened_before_rth():
    """An event that opens during today's premarket (before rth_open) and
    never reverts must not count, even when fed a premarket-inclusive
    window -- per spec, only events that "opened during RTH" qualify."""
    index = pd.date_range("2026-07-27 08:45", periods=60, freq="1min")  # 8:45-9:44, straddles the 9:30 open
    values = [0.0] * 60
    values[0] = 0.05
    values[1:60] = [0.6] * 59  # opens at 8:45 (premarket), still open at the end -- never reverts
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    events = detect_non_reverting_rth_events(spread, equilibrium, RTH_OPEN, RTH_CLOSE, tolerance=0.01)
    assert events == []
    print("PASS: a still-open event that opened before RTH (premarket) is excluded")


def test_detect_non_reverting_rth_events_empty_when_everything_reverts():
    index = pd.date_range("2026-07-27 09:30", periods=30, freq="1min")
    values = [0.0] * 30
    values[5] = 0.05
    values[6:15] = [0.4] * 9
    values[15] = 0.0  # reverts -- nothing left open at series end
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    events = detect_non_reverting_rth_events(spread, equilibrium, RTH_OPEN, RTH_CLOSE, tolerance=0.01)
    assert events == []
    print("PASS: no events when the only divergence in the series fully reverted")


def test_reverted_after_peak_true_when_spread_crosses_zero_after_the_peak():
    """Per the client's Checklist_090926.pdf, entry condition A: the spread
    peaks at 0.5 (idx15), then fully closes back to 0.0 (idx20) -- a "100%
    mean reversion" AFTER its own defining Maximum Divergence."""
    index = pd.date_range("2026-07-27 09:30", periods=25, freq="1min")
    values = [0.0] * 25
    values[10:21] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    assert reverted_after_peak(spread, equilibrium, tolerance=0.01) is True
    print("PASS: reverted_after_peak is True once the spread crosses back through equilibrium after its peak")


def test_reverted_after_peak_false_when_spread_never_returns_to_equilibrium():
    """Same shape as Event D in _build_deviation_series -- opens, peaks at
    0.6, and stays there through the end of the series. No reversion after
    the peak, so this pair's Maximum Divergence is still live."""
    index = pd.date_range("2026-07-27 09:30", periods=25, freq="1min")
    values = [0.0] * 25
    values[10] = 0.05
    values[11:25] = [0.6] * 14
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    assert reverted_after_peak(spread, equilibrium, tolerance=0.01) is False
    print("PASS: reverted_after_peak is False when the spread never returns to equilibrium after its peak")


def test_reverted_after_peak_ignores_a_reversion_that_happened_before_the_peak():
    """A smaller, earlier event (idx5-idx9) fully reverts to 0.0, THEN a
    larger event (idx15) peaks at 0.8 and stays open through the end.
    The pre-peak reversion doesn't count -- only what happens AFTER the
    single largest deviation in the window matters."""
    index = pd.date_range("2026-07-27 09:30", periods=25, freq="1min")
    values = [0.0] * 25
    values[5:10] = [0.1, 0.2, 0.1, 0.05, 0.0]  # opens, peaks 0.2, fully reverts -- BEFORE the real peak
    values[15] = 0.05
    values[16:25] = [0.8] * 9  # the real Maximum Divergence, never reverts
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    assert reverted_after_peak(spread, equilibrium, tolerance=0.01) is False
    print("PASS: reverted_after_peak ignores a reversion that occurred before the window's real peak")


def test_reverted_after_peak_false_on_an_empty_series():
    spread = pd.Series(dtype=float)
    equilibrium = pd.Series(dtype=float)
    assert reverted_after_peak(spread, equilibrium) is False
    print("PASS: reverted_after_peak is False (fail-open, nothing to measure) on an empty series")


def _fake_event(max_spread: float) -> DivergenceEvent:
    return DivergenceEvent(
        open_time=pd.Timestamp("2026-07-31 10:00"), close_time=pd.Timestamp("2026-07-31 10:10"),
        max_spread=max_spread, duration=timedelta(minutes=10), max_adverse_excursion=0.0,
    )


def test_divergence_trigger_level_is_the_maximum_not_an_average():
    """Spec: "Use maximum divergence. Not mean of maximum divergence."
    Previously this averaged the 3 widest, which set a materially lower bar."""
    events = [_fake_event(x) for x in (0.9, 0.05, 0.7, 0.02, 0.8, 0.01)]
    assert divergence_trigger_level(events) == 0.9

    mean_of_three_widest = (0.9 + 0.8 + 0.7) / 3
    assert divergence_trigger_level(events) > mean_of_three_widest  # a stricter trigger than before
    print("PASS: trigger is the single maximum divergence (0.9), not the mean of the 3 widest "
          f"({mean_of_three_widest:.4f})")


def test_a_single_event_is_enough_with_the_condition_off():
    assert divergence_trigger_level([_fake_event(0.4)]) == 0.4
    assert divergence_trigger_level([_fake_event(0.4), _fake_event(0.6)]) == 0.6
    # still nothing to trade off when no event ever closed inside 2 hours
    assert divergence_trigger_level([]) is None
    print("PASS: one qualifying event is enough, and the level is its maximum")


def test_lone_outlier_is_rejected_when_the_condition_is_reinstated():
    """One huge divergence plus small ones drags the mean of the widest three
    up so far that only the outlier itself clears it, so the pair fails. This
    is the real work the condition does -- a single spike is not evidence of a
    repeatable pattern."""
    events = [_fake_event(x) for x in (10.0, 0.5, 0.4, 0.3, 0.2)]
    level = widest_three_mean(events)
    assert level == (10.0 + 0.5 + 0.4) / 3
    assert sum(1 for e in events if e.max_spread >= level) == 1
    assert not passes_divergence_condition(events, min_count=2)
    print("PASS: with the condition on, a lone outlier is rejected")


def test_two_divergences_condition_accepts_comparable_divergences():
    events = [_fake_event(x) for x in (1.0, 0.95, 0.9, 0.1)]
    assert passes_divergence_condition(events, min_count=2)
    assert divergence_trigger_level(events) == 1.0
    print("PASS: comparable divergences pass the condition, trigger is the maximum")


def test_three_events_can_never_reach_the_mean_of_the_widest_three():
    """The reason the count is 2 and not 3. The mean of the widest three
    always sits at or above the third of them, and nothing outside the top
    three can exceed it, so at most two events can ever clear the bar."""
    for values in [(1.0, 0.8, 0.6), (1.0, 0.99, 0.98), (5.0, 3.0, 3.0, 1.0, 0.5)]:
        events = [_fake_event(v) for v in values]
        level = widest_three_mean(events)
        assert sum(1 for e in events if e.max_spread > level) <= 2
        assert not passes_divergence_condition(events, min_count=3)
    print("PASS: no set of divergences can put 3 events above the mean of the widest 3")


def test_event_closes_when_spread_returns_to_its_equilibrium():
    """A gap that peaked $2.70 wide is closed once the spread has travelled
    $2.70 back toward the mean, which is the spread arriving back at its
    equilibrium: "the divergence gap closes back to zero / Spread -
    Equilibrium Spread = 0".

    Measured against the LIVE equilibrium, which drifts upward here. Anchoring
    to the equilibrium as it stood at the peak (10.0) instead would hold the
    event open to minute 10 and, on real data, stretch most events past the
    2-hour limit until they were discarded entirely."""
    index = pd.date_range("2026-07-27 09:30", periods=12, freq="1min")
    #                  open      peak 2.70 over eq(10.0)          back to 10.0
    spread = pd.Series([10.0, 10.5, 12.7, 12.5, 12.2, 11.8, 11.4, 11.0, 10.6, 10.2, 10.0, 10.0], index=index)
    equilibrium = pd.Series([10.0, 10.05, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.0, 11.0, 11.0, 11.0], index=index)

    events = detect_divergence_events(spread, equilibrium, tolerance=0.01)
    assert len(events) == 1
    assert abs(events[0].max_spread - 2.7) < 1e-9
    assert events[0].close_time == index[7]  # spread 11.0 meets equilibrium 11.0
    print("PASS: gap closes when the spread returns to its equilibrium (deviation back to zero)")


def test_divergence_trigger_level_none_when_no_events():
    assert divergence_trigger_level([]) is None
    print("PASS: divergence_trigger_level returns None when there are no qualifying events")


def test_compute_spread_matches_hedge_ratio_convention():
    price_a = pd.Series([100.0, 101.0, 102.0])
    price_b = pd.Series([50.0, 50.5, 51.0])
    spread = compute_spread(price_a, price_b, hedge_ratio=2.0)
    assert list(spread) == [0.0, 0.0, 0.0]
    print("PASS: compute_spread matches the P_A - hedge_ratio*P_B convention")


def test_equilibrium_spread_is_rolling_mean_with_warmup_nans():
    spread = pd.Series(range(1, 11), dtype=float)  # 1..10
    eq = equilibrium_spread(spread, window_minutes=3)
    assert eq.iloc[:2].isna().all()
    assert eq.iloc[2] == (1 + 2 + 3) / 3
    assert eq.iloc[9] == (8 + 9 + 10) / 3
    print("PASS: equilibrium_spread is a rolling mean with correct warmup NaNs")


def test_allowed_divergence_gap_matches_the_clients_worked_example():
    """His BAC/JPM example, 2026-08-04, at the rate in effect at the time
    (0.5%): ELV 29,400 -> 147, over 820 long+short shares -> 0.1793 allowed,
    D1 0.2027 - D2 0.1786 = 0.0241 -> PASS. GAP_ELV_PCT was later raised to
    1% (2026-08-11, per the client, conditional on Divergence Watch showing
    fewer than 10 pairs once properly filtered by this same test -- see
    spread_engine.py's own comment on the constant), so this reproduces the
    SAME event numbers against the CURRENT rate: 1% of 29,400 = 294, over
    820 shares -> 0.3585 allowed, comfortably still a PASS."""
    events = [_fake_event(x) for x in (0.2027, 0.1786, 0.1631)]
    assert round(divergence_gap(events), 4) == 0.0241
    assert round(allowed_divergence_gap(29400, 820), 4) == 0.3585
    assert passes_gap_condition(events, 29400, 820)
    print("PASS: reproduces the client's BAC/JPM gap example against the current 1% rate")


def test_gap_condition_rejects_a_dominant_spike():
    """The point of the rule: a widest divergence standing far above the next
    one is an outlier, and entering at it means targeting a level that
    happened once."""
    events = [_fake_event(x) for x in (20.44, 6.69, 5.41)]
    assert not passes_gap_condition(events, 29400, 820)
    print("PASS: a dominant spike fails the allowed gap")


def test_gap_scales_with_position_size():
    """Allowed gap is per share, so a bigger position tolerates less spread
    divergence between D1 and D2 -- the same dollar risk over more shares.
    Gap widened from the original 0.20 to 0.40 when GAP_ELV_PCT went from
    0.5% to 1% (2026-08-11) -- at the current rate a 0.20 gap now passes at
    both leg sizes, so this needs a wider gap to still demonstrate the
    larger size tightening the allowed threshold enough to fail."""
    events = [_fake_event(x) for x in (0.80, 0.40, 0.35)]  # gap 0.40
    assert passes_gap_condition(events, 29400, 500)        # allowed 0.588
    assert not passes_gap_condition(events, 29400, 1000)   # allowed 0.294
    print("PASS: a larger long+short leg size tightens the allowed gap")


def test_single_event_has_no_gap_to_fail():
    """One event means no second maximum to compare against. Treated as a zero
    gap rather than a failure, since requiring a second event would reinstate
    the minimum-count rule the client removed."""
    assert divergence_gap([_fake_event(3.7)]) == 0.0
    assert passes_gap_condition([_fake_event(3.7)], 29400, 820)
    print("PASS: a single divergence event passes the gap test")


def test_unknown_leg_size_fails_rather_than_passes():
    """Sizing unavailable means the test cannot be evaluated. Failing closed
    keeps an unevaluated pair out rather than letting it through untested."""
    events = [_fake_event(x) for x in (0.2027, 0.1786)]
    assert allowed_divergence_gap(29400, 0) is None
    assert not passes_gap_condition(events, 29400, 0)
    print("PASS: an unknown leg size fails the gap test rather than passing it")


def test_reverted_more_than_pct_after_peak_true_on_a_partial_pullback():
    """Per the client 2026-09-13: "After it hit the maximum divergence, it
    should have at least >50% mean reversion." Peaks at 1.0 (idx15), pulls
    back to 0.4 (idx20, the series' own last bar) -- a 60% retracement,
    well past the 50% default threshold, without ever fully reverting to
    equilibrium. Series ends exactly there, no unfilled trailing bars."""
    index = pd.date_range("2026-07-27 09:30", periods=11, freq="1min")
    values = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 0.8, 0.6, 0.5, 0.45, 0.4]
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    assert reverted_more_than_pct_after_peak(spread, equilibrium, pct=0.5) is True
    print("PASS: reverted_more_than_pct_after_peak is True on a >50% pullback that never fully reverts")


def test_reverted_more_than_pct_after_peak_false_when_pullback_stays_under_threshold():
    """Peaks at 1.0, pulls back only to 0.6 (a 40% retracement) -- stays
    above the 50% threshold through the end of the series (no unfilled
    trailing bars, which would otherwise default to 0.0 and falsely read
    as a full reversion)."""
    index = pd.date_range("2026-07-27 09:30", periods=20, freq="1min")
    values = [0.0] * 20
    values[10] = 0.2
    values[11:15] = [0.4, 0.6, 0.8, 1.0]
    values[15:20] = [0.9, 0.8, 0.7, 0.65, 0.6]
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    assert reverted_more_than_pct_after_peak(spread, equilibrium, pct=0.5) is False
    print("PASS: reverted_more_than_pct_after_peak is False when the pullback never exceeds the threshold")


def test_reverted_more_than_pct_after_peak_distinct_from_full_reversion():
    """The exact case the client called out as a separate condition: a
    pair fully (100%) reverted to equilibrium counts as MORE than 50%
    reverted too -- both conditions are True here, confirming they don't
    contradict each other, just measure different thresholds of the same
    underlying pullback."""
    index = pd.date_range("2026-07-27 09:30", periods=25, freq="1min")
    values = [0.0] * 25
    values[10:21] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    spread = pd.Series(values, index=index)
    equilibrium = pd.Series(0.0, index=index)

    assert reverted_after_peak(spread, equilibrium, tolerance=0.01) is True
    assert reverted_more_than_pct_after_peak(spread, equilibrium, pct=0.5) is True
    print("PASS: a full reversion also satisfies the >50% condition, the two are compatible, not opposites")


def test_reverted_more_than_pct_after_peak_false_on_an_empty_series():
    spread = pd.Series(dtype=float)
    equilibrium = pd.Series(dtype=float)
    assert reverted_more_than_pct_after_peak(spread, equilibrium) is False
    print("PASS: reverted_more_than_pct_after_peak is False (fail-closed) on an empty series")


def test_reverted_more_than_pct_after_peak_false_on_a_zero_peak():
    """A genuinely flat window (no real divergence at all) has nothing to
    have reverted from -- must not divide by zero."""
    index = pd.date_range("2026-07-27 09:30", periods=10, freq="1min")
    spread = pd.Series(0.0, index=index)
    equilibrium = pd.Series(0.0, index=index)
    assert reverted_more_than_pct_after_peak(spread, equilibrium) is False
    print("PASS: reverted_more_than_pct_after_peak is False on a zero-peak window, no division by zero")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
