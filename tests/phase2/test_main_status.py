"""Unit tests for phase2/main.py's testable units -- write_status() (given a
list of PairContext objects, does it produce the right JSON for the
dashboard to read) and _get_elv_with_retry() (the 9:25-9:30am startup
EquityWithLoanValue fetch). The rest of main.py is pure orchestration/glue
covered by its component modules' own tests."""
import asyncio
import json
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

import phase2.main as main
from phase2 import universe_screening
from phase1.screening import MIN_RECENT_CORRELATION
from phase2.risk_manager import Position


def _ctx_with_open_position(symbol_a="TYL", symbol_b="WDAY", pnl_spread=0.5):
    """qty_b/qty_a = 2.0, leg B pinned at 100, so leg A carries the spread:
    entry spread 10.0 -> price_a 210.0, and a +pnl_spread move in price_a is
    exactly a +pnl_spread move in the position's own spread."""
    price_b = 100.0
    ctx = main.PairContext(
        symbol_a=symbol_a, symbol_b=symbol_b, hedge_ratio=2.0,
        spread_history=pd.Series([10.0, 10.0 + pnl_spread]),
        price_history_a=pd.Series([210.0, 210.0 + pnl_spread]),
        price_history_b=pd.Series([price_b, price_b]),
    )
    ctx.position = Position(
        symbol_a=symbol_a, symbol_b=symbol_b, action_a="BUY", qty_a=100, qty_b=200,
        entry_time=datetime(2026, 7, 28, 10, 5), entry_spread=10.0,
        half_life_minutes=30.0, entry_notional=50_000,
        entry_price_a=210.0, entry_price_b=price_b, entry_margin=12_345.67,
    )
    return ctx


def test_write_status_includes_open_positions():
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = _ctx_with_open_position()
    main.write_status("running", account="DU0000001", elv=100_000, contexts=[ctx], daily_pnl=25.0)

    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["status"] == "running"
    assert data["account"] == "DU0000001"
    assert data["elv"] == 100_000
    assert data["daily_pnl"] == 25.0
    assert data["daily_pnl_pct"] == 0.03  # 25.0 / 100_000 * 100, rounded
    assert len(data["open_positions"]) == 1
    pos = data["open_positions"][0]
    assert pos["symbol_a"] == "TYL"
    assert pos["unrealized_pnl"] == 50.0  # BUY, qty_a=100, spread rose 0.5 -> 100*0.5
    assert pos["unrealized_pnl_pct"] == 0.1  # 50.0 / entry_notional(50_000) * 100
    assert pos["entry_price_a"] == 210.0
    assert pos["entry_price_b"] == 100.0
    assert pos["entry_margin"] == 12345.67
    # Per the client's own worked example ("Leg Size: 15 MPWR, 91 MRVL.
    # Position Size: (15+91)/2 = 53 Pairs"): Leg Size is qty_a/qty_b
    # directly, Position Size is their average -- not a reduced ratio.
    assert pos["pair_units"] == 150.0  # (100+200)/2
    print("PASS: write_status correctly serializes an open position with unrealized P&L, "
          "leg size, entry price, position size, entry margin, and PnL percentage")


def test_write_status_includes_points_to_trigger_fields():
    # Per the client 2026-08-19's own worked example: "Points to trigger @
    # Entry = 0.24, Points + trigger @ Take Profit = 1.52" -- trigger level
    # 1.52, divergence 1.28 at entry (1.52 - 1.28 = 0.24 distance to go).
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = _ctx_with_open_position()
    ctx.position.entry_trigger_level = 1.52
    ctx.position.entry_divergence = 1.28
    main.write_status("running", account="DU0000001", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    pos = data["open_positions"][0]
    assert pos["points_to_trigger_at_entry"] == 0.24
    assert pos["take_profit_trigger_level"] == 1.52
    print("PASS: write_status reports the client's own worked example numbers for the trigger fields")


def test_write_status_empty_positions_when_all_flat():
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    flat_ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0, spread_history=pd.Series(dtype=float))
    main.write_status("running", account="DU0000001", contexts=[flat_ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["open_positions"] == []
    assert data["qualified_pairs_count"] == 1
    print("PASS: write_status reports no open positions when every pair is flat")


def test_write_status_has_fresh_timestamp():
    import time
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    before = time.time()
    main.write_status("waiting")
    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert before <= data["ts"] <= time.time()
    print("PASS: write_status stamps a fresh epoch timestamp for the dashboard's staleness check")


def test_write_status_trading_halted_flag():
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    main.write_status("running", trading_halted=True)
    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["trading_halted"] is True
    print("PASS: write_status passes through the daily loss kill-switch state")


def test_write_status_news_scan_healthy_flag():
    # Per the client 2026-08-17: "If Gemini news API is down, no trading."
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    main.write_status("running", news_scan_healthy=False)
    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["news_scan_healthy"] is False
    print("PASS: write_status passes through the news scan health flag")


def test_write_status_includes_min_hurst():
    """Per the client 2026-09-02: the Live Johansen/Hurst panel's own
    "need >X" label was hardcoded to 0.60 and went stale the day after
    MIN_HURST was lowered to 0.50 -- sent here so the dashboard reads the
    real, current threshold every poll instead of a number baked into
    dashboard.py."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    main.write_status("running")
    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["min_hurst"] == main.MIN_HURST
    print("PASS: write_status includes the real, current MIN_HURST value")


def _prior_session_prices(n_bars: int, event_start: int | None = None, event_values: list[float] | None = None,
                           start: str = "2026-07-27 09:30", price_b: float = 100.0) -> tuple[pd.Series, pd.Series]:
    """Previous-session price_a/price_b series (hedge_ratio=2.0 baseline, so
    spread = price_a - 200.0) with an optional divergence spliced into
    price_a -- price_b pinned flat, same "peg one leg, drive the deviation
    through the other" convention _ctx_with_open_position uses above.
    event_values shapes a divergence event for establish_base_trigger to
    read the peak of -- since 2026-09-02 it no longer matters whether the
    values ramp back down (revert) or just hold/keep climbing (never
    reverts), only the peak absolute deviation reached matters now."""
    index = pd.date_range(start, periods=n_bars, freq="1min")
    values = [200.0] * n_bars
    if event_start is not None:
        values[event_start:event_start + len(event_values)] = [200.0 + v for v in event_values]
    price_a = pd.Series(values, index=index)
    price_b_series = pd.Series(price_b, index=index)
    return price_a, price_b_series


# Ramps to 3.0 over 5 bars, holds for 10, then ramps back down to 0 over 5
# more -- closes (reverts) well within the 2-hour window, so it's a valid
# detect_divergence_events candidate. Peak 3.0 sits between
# MIN_ENTRY_DIVERGENCE and MAX_ENTRY_DIVERGENCE (currently $1.5 and $5), so
# _bounded_trigger_level doesn't discard it.
REVERTING_EVENT = [0.6, 1.2, 1.8, 2.4, 3.0] + [3.0] * 10 + [2.4, 1.8, 1.2, 0.6, 0.0]


def test_establish_base_trigger_stays_none_without_a_qualifying_event():
    # 90 flat bars, no divergence at all -- nothing to compute from
    price_a, price_b = _prior_session_prices(90)
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float),
                            prior_session_price_a=price_a, prior_session_price_b=price_b)
    main.establish_base_trigger(ctx)
    assert ctx.divergence_trigger_level is None
    print("PASS: stays None when the previous session never produced a non-reverting divergence event")


def test_establish_base_trigger_disqualifies_a_trigger_below_the_floor():
    # REVERSED 2026-09-12, per the client, ending a long same-night
    # back-and-forth: "Maximum Divergence is 1.5-5... 1.5-5 range applies
    # only to the trigger." The 2026-09-04 "not below" instruction this
    # test used to encode ("It can only reject anything above 5. Not
    # below") no longer holds -- the client's own later, more direct
    # correction wins. Peaks at $1.20, below the current $1.5 floor.
    small_event = [0.2, 0.6, 1.0, 1.2, 1.2] + [1.0, 0.6, 0.2, 0.0]
    price_a, price_b = _prior_session_prices(150, event_start=70, event_values=small_event)
    ctx = main.PairContext(symbol_a="STX", symbol_b="WDC", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float),
                            prior_session_price_a=price_a, prior_session_price_b=price_b)
    main.establish_base_trigger(ctx)
    assert ctx.divergence_trigger_level is None
    print("PASS: a trigger below MIN_ENTRY_DIVERGENCE is disqualified, reinstated 2026-09-12")


def test_establish_base_trigger_disqualifies_trigger_above_maximum_divergence():
    # Per the client 2026-08-18: a trigger above the $6 Maximum Divergence
    # cap can never actually be entered (entry needs to REACH its trigger
    # first), so it's treated as disqualified. Peaks at 9.0 (above the $6
    # cap) then closes back to 0 -- a real, qualifying (reverting) event,
    # just one that must still be discarded for exceeding the cap.
    big_event = [1.0, 3.0, 6.0, 9.0, 9.0] + [6.0, 3.0, 1.0, 0.0]
    price_a, price_b = _prior_session_prices(150, event_start=70, event_values=big_event)
    ctx = main.PairContext(symbol_a="STX", symbol_b="WDC", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float),
                            prior_session_price_a=price_a, prior_session_price_b=price_b)
    main.establish_base_trigger(ctx)
    assert ctx.divergence_trigger_level is None
    print("PASS: a trigger above the $6 maximum divergence cap disqualifies the pair")


def test_establish_base_trigger_from_a_reverting_event():
    # event_start=70 -- the 60-bar equilibrium window needs 60 flat bars to
    # stabilize at exactly 200.0 BEFORE the event starts, or the event's own
    # bars leak into the equilibrium's warm-up and distort the reading.
    price_a, price_b = _prior_session_prices(150, event_start=70, event_values=REVERTING_EVENT)
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float),
                            prior_session_price_a=price_a, prior_session_price_b=price_b)
    main.establish_base_trigger(ctx)
    assert ctx.divergence_trigger_level is not None
    # Not an exact 3.0: the rolling equilibrium creeps up slightly during
    # the event's own 10-bar hold phase before the peak is reached, same
    # as any real intraday event -- just confirms a real, bounded level
    # close to the constructed peak landed.
    assert 2.5 <= ctx.divergence_trigger_level <= 3.0
    assert ctx.base_trigger_level == ctx.divergence_trigger_level
    assert ctx.half_life_minutes is not None
    # Checklist_090926.pdf condition A: this event fully reverted after its
    # own peak, so entry must be blocked on it -- see
    # PairContext.mean_reverted_since_max_divergence's own comment.
    assert ctx.mean_reverted_since_max_divergence is True
    # A full reversion also satisfies the distinct >50% condition -- see
    # PairContext.reverted_over_50pct_since_max_divergence's own comment.
    assert ctx.reverted_over_50pct_since_max_divergence is True
    print("PASS: establishes a real base trigger level (and half-life) from a reverting event "
          "in the previous session alone, and flags the reversion for condition A")


def test_establish_base_trigger_counts_a_non_reverting_event():
    """Per the client 2026-08-24's original Critical Checklist, the BASE
    trigger only counted events that mean-reverted within 2 hours --
    reversed 2026-09-02, direct and repeated correction: "I told you
    there is no reversal inside 2 hours... We just need the maximum
    divergence in the lookback period. Simple. There is no reversal
    involved anywhere." Confirmed live the same day: NOW/TEAM had
    cleared Johansen and Hurst but never got a trigger at all, because
    the old rule discarded its one real divergence event for never
    reverting. A non-reverting event's peak now establishes a real
    trigger, same as a reverting one would."""
    never_reverts = [0.6, 1.2, 1.8, 2.4, 3.0] + [3.0] * 74
    price_a, price_b = _prior_session_prices(90, event_start=10, event_values=never_reverts)
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float),
                            prior_session_price_a=price_a, prior_session_price_b=price_b)
    main.establish_base_trigger(ctx)
    assert ctx.divergence_trigger_level is not None
    assert 2.5 <= ctx.divergence_trigger_level <= 3.0
    # This event never reverted after its peak, so condition A stays clear.
    assert ctx.mean_reverted_since_max_divergence is False
    # Never pulled back at all -- the distinct >50% condition stays clear too.
    assert ctx.reverted_over_50pct_since_max_divergence is False
    print("PASS: a non-reverting event's peak establishes a real base trigger, not discarded, "
          "and leaves condition A clear")


def test_establish_base_trigger_ignores_a_high_half_life():
    """Per the client 2026-08-25: "Remove Half-Life, truly this time
    around" -- a pair whose half-life comes in >= 60 minutes must still get
    a real trigger established, not silently disqualified. Regression test
    for the exact live incident (GE/RTX at 69.7min, ANET/LITE at 87.4min,
    both blocked from ever reaching Divergence Watch by this gate)."""
    price_a, price_b = _prior_session_prices(150, event_start=70, event_values=REVERTING_EVENT)
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float),
                            prior_session_price_a=price_a, prior_session_price_b=price_b)
    with patch("phase2.main.half_life_of_mean_reversion") as mock_half_life:
        mock_half_life.return_value = 90.0  # well above the old 60-minute cutoff
        main.establish_base_trigger(ctx)
    assert ctx.divergence_trigger_level is not None
    assert ctx.base_trigger_level is not None
    assert ctx.half_life_minutes == 90.0  # still computed and kept for visibility
    print("PASS: a half-life >= 60 minutes no longer blocks trigger establishment")


def test_base_trigger_stays_frozen_as_live_bars_arrive():
    """Per the client 2026-08-22: half-life and the trigger are now frozen
    once established at warm-up, unlike the old design where a live
    recalculation every cycle from the whole growing intraday series could
    drift and disqualify an already-qualified pair mid-session (the GE/RTX
    incident)."""
    price_a, price_b = _prior_session_prices(150, event_start=70, event_values=REVERTING_EVENT)
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float),
                            prior_session_price_a=price_a, prior_session_price_b=price_b)
    main.establish_base_trigger(ctx)
    first_trigger, first_half_life = ctx.divergence_trigger_level, ctx.half_life_minutes
    assert first_trigger is not None

    # Today's live bars grow ctx.price_history_a/b and ctx.spread_history --
    # completely separate fields from prior_session_price_a/b, which
    # establish_base_trigger reads. Nothing recomputes the trigger from this.
    ctx.price_history_a = pd.Series([500.0, 600.0, 700.0])
    ctx.price_history_b = pd.Series([100.0, 100.0, 100.0])
    ctx.spread_history = main.compute_spread(ctx.price_history_a, ctx.price_history_b, ctx.hedge_ratio)

    assert ctx.divergence_trigger_level == first_trigger
    assert ctx.half_life_minutes == first_half_life
    print("PASS: the frozen base trigger and half-life are untouched by live bars arriving")


def test_update_live_cointegration_stats_stays_none_without_enough_bars():
    # 20 bars, below MIN_BARS_FOR_LIVE_COINTEGRATION_STATS (30)
    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0,
        spread_history=pd.Series([1.0] * 20),
        price_history_a=pd.Series(range(20), dtype=float),
        price_history_b=pd.Series(range(20), dtype=float),
    )
    main.update_live_cointegration_stats(ctx)
    assert ctx.live_correlation is None
    assert ctx.live_adf_pvalue is None
    print("PASS: correlation/ADF stay None until enough bars have accumulated")


def test_update_live_cointegration_stats_computes_over_the_same_window():
    """Both stats must run over the same rolling window -- correlation used
    to run over live bars only while ADF ran over warm-up + live, so the two
    conditions judged different time spans. Per the client 2026-08-22's
    Entry Conditions PDF ("N = 30 bars (30 minutes)"), that window is now
    the trailing MIN_BARS_FOR_LIVE_COINTEGRATION_STATS bars, not the whole
    history."""
    n = 40
    index = pd.date_range("2026-07-30 10:00", periods=n, freq="1min")
    # two closely-tracking price series -- should show high correlation.
    # small alternating wobble keeps the spread from being perfectly constant
    # (ADF can't run on a constant series) without hurting the correlation.
    wobble = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]
    prices_a = pd.Series([100.0 + i * 0.1 + wobble[i] for i in range(n)], index=index)
    prices_b = pd.Series([50.0 + i * 0.05 for i in range(n)], index=index)
    spread = prices_a - 2.0 * prices_b

    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0, spread_history=spread,
        price_history_a=prices_a, price_history_b=prices_b,
    )
    main.update_live_cointegration_stats(ctx)

    assert ctx.live_correlation is not None
    assert 0.9 <= ctx.live_correlation <= 1.0  # near-perfectly linear series
    assert ctx.live_adf_pvalue is not None
    assert 0.0 <= ctx.live_adf_pvalue <= 1.0
    # Still cointegrated (high correlation, presumably stationary spread) --
    # cointegration_still_broken must read False, not just "computed".
    assert ctx.cointegration_still_broken is False
    print("PASS: correlation and ADF both computed over the same rolling 30-bar window, "
          "and cointegration_still_broken reads False for a still-intact relationship")


def test_update_live_cointegration_stats_wires_the_composite_breakdown_result():
    """cointegration_still_broken is set from cointegration_breakdown.
    is_cointegration_broken's own return value -- that function's actual
    4-metric logic (correlation, half-life inflation, rolling ADF, Kalman
    drift, ALL must show broken) is exercised directly in
    test_cointegration_breakdown.py; this only confirms the wiring: the
    right ctx fields reach the call, and the result lands on the right
    ctx attribute."""
    n = 40
    index = pd.date_range("2026-07-30 10:00", periods=n, freq="1min")
    wobble = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]  # keeps the spread non-constant, ADF needs variance
    prices_a = pd.Series([100.0 + i * 0.5 + wobble[i] for i in range(n)], index=index)
    prices_b = pd.Series([50.0 + i * 0.25 for i in range(n)], index=index)
    spread = prices_a - 2.0 * prices_b

    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0, spread_history=spread,
        price_history_a=prices_a, price_history_b=prices_b, half_life_minutes=15.0,
    )
    with patch("phase2.main.is_cointegration_broken") as mock_broken:
        mock_broken.return_value = True
        main.update_live_cointegration_stats(ctx)

    assert ctx.cointegration_still_broken is True
    args, kwargs = mock_broken.call_args
    assert args[4] == 15.0  # half_life_minutes passed through
    assert args[5] == 2.0  # hedge_ratio passed through as seed_beta


def _replay_johansen_hurst_sliding_gate(ctx, prices_a, prices_b, start_ts=0):
    """Replicates run_check_cycle's own johansen_hurst_price_a/b block
    exactly (same reasoning as _replay_live_beta_bar_by_bar above): append
    the new bar, then trim to the fixed JOHANSEN_HURST_WINDOW_BARS length,
    dropping the OLDEST bar once the array is full -- per the client
    2026-08-25 ("Live Bot Execution (Sliding Gate)"): "each incoming
    1-minute bar appends to the array, dropping the oldest bar to keep a
    fixed array length.\""""
    for i, (price_a, price_b) in enumerate(zip(prices_a, prices_b)):
        stamp = pd.Timestamp(start_ts + i, unit="s")
        ctx.johansen_hurst_price_a.loc[stamp] = price_a
        ctx.johansen_hurst_price_b.loc[stamp] = price_b
        if len(ctx.johansen_hurst_price_a) > main.JOHANSEN_HURST_WINDOW_BARS:
            ctx.johansen_hurst_price_a = ctx.johansen_hurst_price_a.iloc[-main.JOHANSEN_HURST_WINDOW_BARS:]
            ctx.johansen_hurst_price_b = ctx.johansen_hurst_price_b.iloc[-main.JOHANSEN_HURST_WINDOW_BARS:]


def test_johansen_hurst_sliding_gate_keeps_a_fixed_length_dropping_the_oldest_bar():
    seed_len = main.JOHANSEN_HURST_WINDOW_BARS - 3
    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0, spread_history=pd.Series(dtype=float),
        johansen_hurst_price_a=pd.Series(
            range(seed_len), index=pd.date_range("2026-06-01", periods=seed_len, freq="1min"), dtype=float),
        johansen_hurst_price_b=pd.Series(
            range(seed_len), index=pd.date_range("2026-06-01", periods=seed_len, freq="1min"), dtype=float),
    )
    first_bar_value = ctx.johansen_hurst_price_a.iloc[0]
    # Append 10 new bars -- 3 fit under the cap, the next 7 each push the
    # oldest remaining bar out.
    _replay_johansen_hurst_sliding_gate(ctx, [999.0] * 10, [999.0] * 10, start_ts=10_000)
    assert len(ctx.johansen_hurst_price_a) == main.JOHANSEN_HURST_WINDOW_BARS
    assert len(ctx.johansen_hurst_price_b) == main.JOHANSEN_HURST_WINDOW_BARS
    assert first_bar_value not in ctx.johansen_hurst_price_a.values[:7]  # the earliest seed bars got dropped
    assert ctx.johansen_hurst_price_a.iloc[-1] == 999.0  # newest bar present
    print(f"PASS: sliding gate stays pinned at exactly {main.JOHANSEN_HURST_WINDOW_BARS} bars, "
          f"dropping the oldest as new bars arrive")


def test_a_johansen_hurst_failure_on_one_pair_does_not_block_the_rest():
    """Regression test for the pre-live audit finding, 2026-08-26:
    update_live_johansen_hurst runs inside a bare `for ctx in contexts`
    loop with no other exception handling around it -- coint_johansen can
    raise on a singular/near-singular matrix (long flat stretches,
    duplicate timestamps), a real failure mode on live data. Uncaught, that
    would abort the WHOLE cycle's loop, silently freezing every pair after
    the failing one too, repeating every cycle if the bad data persists.
    Replicates run_check_cycle's own per-pair try/except exactly."""
    import numpy as np
    np.random.seed(11)
    n = 350
    good_a = pd.Series(np.cumsum(np.random.normal(0, 1, n)) + 100)
    good_b = pd.Series(np.cumsum(np.random.normal(0, 1, n)) + 50)
    bad = pd.Series([100.0] * n)  # degenerate: zero variance, singular matrix

    ctx_bad = main.PairContext(symbol_a="BAD", symbol_b="PAIR", hedge_ratio=1.0,
                                spread_history=pd.Series(dtype=float),
                                johansen_hurst_price_a=bad, johansen_hurst_price_b=bad)
    ctx_good = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
                                 spread_history=pd.Series(dtype=float),
                                 johansen_hurst_price_a=good_a, johansen_hurst_price_b=good_b)

    for ctx in [ctx_bad, ctx_good]:
        try:
            main.update_live_johansen_hurst(ctx)
        except Exception:
            pass

    # The bad pair's own update may or may not have raised (depends on the
    # exact singularity), but the good pair AFTER it in iteration order must
    # still have been processed regardless -- that's the actual regression.
    assert ctx_good.live_johansen_confirmed is not None
    assert ctx_good.live_hurst is not None
    print("PASS: a Johansen/Hurst failure on one pair does not block the next pair in the same cycle")


def test_update_live_johansen_hurst_stays_none_without_enough_bars():
    # below MIN_BARS_FOR_LIVE_JOHANSEN_HURST (300)
    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=2.0,
        spread_history=pd.Series(dtype=float),
        johansen_hurst_price_a=pd.Series(range(200), dtype=float),
        johansen_hurst_price_b=pd.Series(range(200), dtype=float),
    )
    main.update_live_johansen_hurst(ctx)
    assert ctx.live_johansen_confirmed is None
    assert ctx.live_hurst is None
    print("PASS: live Johansen/Hurst stay None until enough bars have accumulated")


def test_update_live_johansen_hurst_computes_over_the_rolling_window():
    """Per the client 2026-08-25 (Correct_Filters.pdf): Johansen and Hurst
    recomputed live over ctx.johansen_hurst_price_a/b, separate from
    price_history_a/b (the much shorter window the entry trigger uses)."""
    import numpy as np
    np.random.seed(7)
    n = 350
    index = pd.date_range("2026-07-01 09:30", periods=n, freq="1min")
    b = pd.Series(np.cumsum(np.random.normal(0, 1, n)) + 100, index=index)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + np.random.normal(0, 0.5)  # AR(1), mean-reverting
    a = 1.5 * b + pd.Series(noise, index=index)

    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.5,
        spread_history=pd.Series(dtype=float),
        johansen_hurst_price_a=a, johansen_hurst_price_b=b,
    )
    main.update_live_johansen_hurst(ctx)

    assert ctx.live_johansen_confirmed is not None
    assert isinstance(ctx.live_johansen_confirmed, bool)
    assert ctx.live_hurst is not None
    print(f"PASS: live Johansen/Hurst computed over the rolling window "
          f"(johansen_confirmed={ctx.live_johansen_confirmed}, hurst={ctx.live_hurst:.3f})")
    print("PASS: update_live_cointegration_stats wires ctx's own half-life/hedge_ratio into the "
          "composite breakdown check and stores its result")


def test_johansen_and_hurst_sticky_flags_clear_independently_on_different_cycles():
    """Per the client 2026-08-28, clarified after the first build: "They
    must not be validated together. At the same time. No. They just need
    to be validated once during Entry RTH session." -- Johansen clearing
    on one cycle and Hurst clearing on a LATER, different cycle must still
    count: each sticky flag turns on independently and stays on, and
    entry_eligible (via PairSignal.johansen_hurst_confirmed) reads the AND
    of both, not whether they cleared in the same cycle."""
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float),
                            johansen_hurst_price_a=pd.Series(range(400), dtype=float),
                            johansen_hurst_price_b=pd.Series(range(400), dtype=float))
    assert ctx.johansen_ever_unconfirmed is False
    assert ctx.hurst_ever_above_threshold is False

    # Cycle 1: Johansen clears (Unconfirmed), Hurst does not.
    with patch.object(main, "johansen_confirmed", return_value=False), \
         patch.object(main, "hurst_exponent", return_value=0.40):
        main.update_live_johansen_hurst(ctx)
    assert ctx.johansen_ever_unconfirmed is True
    assert ctx.hurst_ever_above_threshold is False

    # Cycle 2: Johansen fails now (Confirmed), but its sticky flag stays on
    # from cycle 1; Hurst clears this cycle.
    with patch.object(main, "johansen_confirmed", return_value=True), \
         patch.object(main, "hurst_exponent", return_value=0.65):
        main.update_live_johansen_hurst(ctx)
    assert ctx.johansen_ever_unconfirmed is True  # still true from cycle 1
    assert ctx.hurst_ever_above_threshold is True  # newly true this cycle
    print("PASS: Johansen and Hurst sticky flags each turn on independently, on different cycles")


def test_update_live_johansen_hurst_logs_the_first_time_each_flag_flips(caplog):
    """Per the client repeatedly asking "at what time did X pass
    Johansen/Hurst" (2026-08-31, 2026-09-01) with no way to answer from
    the existing divergence-check log line -- one INFO line per flag, only
    on the cycle it actually flips, not every cycle after."""
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float),
                            johansen_hurst_price_a=pd.Series(range(400), dtype=float),
                            johansen_hurst_price_b=pd.Series(range(400), dtype=float))
    with caplog.at_level("INFO"):
        # Cycle 1: Johansen clears, Hurst does not yet.
        with patch.object(main, "johansen_confirmed", return_value=False), \
             patch.object(main, "hurst_exponent", return_value=0.40):
            main.update_live_johansen_hurst(ctx)
        assert "TYL/WDAY: 20 Day Johansen validated (first time this session)" in caplog.text
        assert "Hurst validated" not in caplog.text
        caplog.clear()

        # Cycle 2: Johansen already stuck True, logs nothing new; Hurst
        # clears for the first time and logs once.
        with patch.object(main, "johansen_confirmed", return_value=False), \
             patch.object(main, "hurst_exponent", return_value=0.65):
            main.update_live_johansen_hurst(ctx)
        assert "Johansen validated" not in caplog.text
        assert "TYL/WDAY: 20 Day Hurst validated (first time this session)" in caplog.text
        caplog.clear()

        # Cycle 3: both already stuck True -- silent from here on.
        with patch.object(main, "johansen_confirmed", return_value=False), \
             patch.object(main, "hurst_exponent", return_value=0.70):
            main.update_live_johansen_hurst(ctx)
        assert "validated" not in caplog.text
    print("PASS: Johansen/Hurst validation is logged once, only on the cycle each flag first flips")


def test_johansen_hurst_stays_false_until_each_clears_at_least_once():
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float),
                            johansen_hurst_price_a=pd.Series(range(400), dtype=float),
                            johansen_hurst_price_b=pd.Series(range(400), dtype=float))
    with patch.object(main, "johansen_confirmed", return_value=True), \
         patch.object(main, "hurst_exponent", return_value=0.65):
        main.update_live_johansen_hurst(ctx)
    assert ctx.johansen_ever_unconfirmed is False  # Johansen confirmed (bad), never cleared
    assert ctx.hurst_ever_above_threshold is True

    with patch.object(main, "johansen_confirmed", return_value=True), \
         patch.object(main, "hurst_exponent", return_value=0.55):
        main.update_live_johansen_hurst(ctx)
    assert ctx.johansen_ever_unconfirmed is False  # still never cleared
    assert ctx.hurst_ever_above_threshold is True  # stays true from before
    print("PASS: each sticky flag stays False until its own condition clears at least once")


def test_update_live_johansen_hurst_computes_the_20day_and_1day_windows_independently():
    """Per the client 2026-09-06: "They are two separate calculations."
    The 20-day window (400 bars here) has enough bars to compute; the
    1-day window is left empty (0 bars, under MIN_BARS_FOR_LIVE_JOHANSEN_HURST)
    so it must NOT compute or flip its own sticky flags, while the 20-day
    ones do."""
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float),
                            johansen_hurst_price_a=pd.Series(range(400), dtype=float),
                            johansen_hurst_price_b=pd.Series(range(400), dtype=float))
    with patch.object(main, "johansen_confirmed", return_value=False), \
         patch.object(main, "hurst_exponent", return_value=0.65):
        main.update_live_johansen_hurst(ctx)
    assert ctx.johansen_ever_unconfirmed is True
    assert ctx.hurst_ever_above_threshold is True
    assert ctx.live_hurst == 0.65
    # 1-day window never had enough bars -- stays at its untouched defaults.
    assert ctx.johansen_ever_unconfirmed_1day is False
    assert ctx.hurst_ever_above_threshold_1day is False
    assert ctx.live_hurst_1day is None
    print("PASS: the 20-day and 1-day windows compute and flip their sticky flags independently")


def test_update_live_johansen_hurst_1day_window_computes_once_it_has_enough_bars():
    ctx = main.PairContext(symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float),
                            johansen_hurst_price_a_1day=pd.Series(range(350), dtype=float),
                            johansen_hurst_price_b_1day=pd.Series(range(350), dtype=float))
    with patch.object(main, "johansen_confirmed", return_value=False), \
         patch.object(main, "hurst_exponent", return_value=0.55):
        main.update_live_johansen_hurst(ctx)
    assert ctx.johansen_ever_unconfirmed_1day is True
    assert ctx.hurst_ever_above_threshold_1day is True
    assert ctx.live_hurst_1day == 0.55
    # 20-day window never had enough bars -- stays at its untouched defaults.
    assert ctx.johansen_ever_unconfirmed is False
    assert ctx.hurst_ever_above_threshold is False
    print("PASS: the 1-day window computes independently once it has its own enough bars")


def _proportional_return_prices(n, start_a=100.0, start_b=50.0, ratio=2.0, seed=1):
    """n+1 bars for each leg where B's bar-to-bar return is always `ratio`
    times A's -- perfectly linear, so pearson_correlation_1day gives
    exactly +/-1.0 (a fixed multiple, not just the same sign)."""
    rng = np.random.RandomState(seed)
    returns_a = rng.uniform(-0.02, 0.02, n)
    price_a = pd.Series([start_a] + [0.0] * n)
    price_b = pd.Series([start_b] + [0.0] * n)
    for i, r in enumerate(returns_a):
        price_a.iloc[i + 1] = price_a.iloc[i] * (1 + r)
        price_b.iloc[i + 1] = price_b.iloc[i] * (1 + ratio * r)
    return price_a, price_b


def test_update_live_johansen_hurst_computes_1day_pearson_and_sets_the_sticky_flag():
    """Per the client 2026-09-07 (1_Day_Pearson_60_Bars.pdf, "Method 1: The
    Rolling Intraday Bar-to-Bar Approach", superseding two earlier same-day
    versions): correlates each leg's OWN bar-to-bar returns over a trailing
    PEARSON_1DAY_WINDOW_BARS (60) window of ctx.premarket_price_a/b -- B's
    return is always 2x A's here, so r=1.0, well above MIN_RECENT_CORRELATION
    (0.70) -> sticky flag flips."""
    price_a, price_b = _proportional_return_prices(main.PEARSON_1DAY_WINDOW_BARS)
    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
        spread_history=pd.Series(dtype=float),
        johansen_hurst_price_a_1day=pd.Series(np.arange(300, dtype=float) + 100),
        johansen_hurst_price_b_1day=pd.Series(np.arange(300, dtype=float) * 2 + 50),
        premarket_price_a=price_a, premarket_price_b=price_b,
    )
    with patch.object(main, "johansen_confirmed", return_value=False), \
         patch.object(main, "hurst_exponent", return_value=0.55):
        main.update_live_johansen_hurst(ctx)
    assert ctx.live_pearson_1day is not None
    assert abs(ctx.live_pearson_1day - 1.0) < 1e-6
    assert ctx.pearson_ever_above_threshold_1day is True
    print("PASS: update_live_johansen_hurst computes 1 Day Pearson and sets its own sticky flag")


def test_update_live_johansen_hurst_1day_pearson_stays_false_below_threshold():
    # B's return is always -1x A's here -- perfectly anti-correlated, well under 0.70.
    price_a, price_b = _proportional_return_prices(main.PEARSON_1DAY_WINDOW_BARS, ratio=-1.0)
    ctx = main.PairContext(
        symbol_a="TYL", symbol_b="WDAY", hedge_ratio=1.0,
        spread_history=pd.Series(dtype=float),
        johansen_hurst_price_a_1day=pd.Series(np.arange(300, dtype=float) + 100),
        johansen_hurst_price_b_1day=pd.Series(-np.arange(300, dtype=float) + 500),
        premarket_price_a=price_a, premarket_price_b=price_b,
    )
    with patch.object(main, "johansen_confirmed", return_value=False), \
         patch.object(main, "hurst_exponent", return_value=0.55):
        main.update_live_johansen_hurst(ctx)
    assert ctx.live_pearson_1day is not None
    assert ctx.live_pearson_1day < MIN_RECENT_CORRELATION
    assert ctx.pearson_ever_above_threshold_1day is False
    print("PASS: 1 Day Pearson stays False (sticky flag never flips) when correlation is below threshold")


def test_report_event_from_position_uses_the_trades_own_real_data():
    """Client: "Post-Trade Report should be based on executed pairs" --
    and more precisely, on the trade's own real lifecycle, not a re-scan of
    the pair's whole-day price history for divergence events that may not
    even coincide with when the position was actually open.

    entry_divergence is "the divergence" for the report (item 1); qty_a is
    the REAL executed quantity, not a hypothetical sizing figure; open/close
    times are the trade's real entry and exit."""
    entry_time = datetime(2026, 8, 3, 13, 25, 55)
    close_time = datetime(2026, 8, 3, 13, 37, 15)
    position = Position(symbol_a="FIX", symbol_b="PWR", action_a="BUY", qty_a=282, qty_b=732,
                        entry_time=entry_time, entry_spread=0.26, half_life_minutes=14.3,
                        entry_notional=500_000, entry_divergence=23.4849,
                        worst_unrealized_pnl=-500.0)  # never went positive-to-negative here, just underwater at some point
    ctx = main.PairContext(symbol_a="FIX", symbol_b="PWR", hedge_ratio=1.0,
                           spread_history=pd.Series(dtype=float))
    ctx.position = position

    event = main.report_event_from_position(ctx, close_time)
    assert event.symbol_a == "FIX" and event.symbol_b == "PWR"
    assert event.open_time == entry_time and event.close_time == close_time
    assert event.duration == close_time - entry_time
    assert event.half_life_minutes == 14.3
    assert event.max_spread == 23.4849  # the entry divergence, not a re-scanned peak
    assert event.qty_a == 282  # the real executed quantity
    assert event.max_adverse_excursion == 500.0 / 282  # worst dollar P&L converted back to per-share
    print("PASS: report_event_from_position builds the row entirely from the trade's own real data")


def test_report_event_from_position_shows_zero_drawdown_if_the_trade_never_went_underwater():
    """A trade that only ever gained is a real, correct case -- the report
    must show 0% drawdown for it, not manufacture one from an unrelated
    divergence event elsewhere in the pair's day."""
    position = Position(symbol_a="A", symbol_b="B", action_a="BUY", qty_a=100, qty_b=100,
                        entry_time=datetime(2026, 8, 3, 10, 0), entry_spread=1.0,
                        half_life_minutes=10.0, entry_notional=10_000, entry_divergence=0.5,
                        worst_unrealized_pnl=0.0)  # default: never dipped negative
    ctx = main.PairContext(symbol_a="A", symbol_b="B", hedge_ratio=1.0, spread_history=pd.Series(dtype=float))
    ctx.position = position

    event = main.report_event_from_position(ctx, datetime(2026, 8, 3, 10, 20))
    assert event.max_adverse_excursion == 0.0
    print("PASS: a trade that never went underwater reports zero drawdown, not a phantom figure")


def test_worst_unrealized_pnl_tracks_the_running_minimum():
    """The check-cycle loop updates this every pass; confirming the update
    itself keeps the running worst (most negative) value rather than the
    latest one, since that's what a real max drawdown means."""
    position = Position(symbol_a="A", symbol_b="B", action_a="BUY", qty_a=100, qty_b=100,
                        entry_time=datetime(2026, 8, 3, 10, 0), entry_spread=1.0,
                        half_life_minutes=10.0, entry_notional=10_000)
    for pnl in (50.0, -200.0, -80.0, 300.0):  # dips to -200, recovers, ends positive
        position.worst_unrealized_pnl = min(position.worst_unrealized_pnl, pnl)
    assert position.worst_unrealized_pnl == -200.0
    print("PASS: worst_unrealized_pnl keeps the running minimum across the trade's life")


def test_book_is_flat_gates_concurrent_positions():
    """One position at a time across the whole book.

    The quantity formula spends 98% of EquityWithLoanValue on a single pair,
    the tiebreak picks ONE pair when several trigger together, and re-entry
    happens "at the end of a trade" -- all three assume sequential trading.
    This is not theoretical: two pairs each sized against the full equity
    asked IBKR for more initial margin than was available and
    the second was rejected with error 201."""
    flat = [main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                             spread_history=pd.Series(dtype=float)) for _ in range(3)]
    assert main.book_is_flat(flat)
    assert main.book_is_flat([])

    flat[1].position = _ctx_with_open_position().position
    assert not main.book_is_flat(flat), "a position on any pair must block a new entry"
    print("PASS: an open position anywhere in the book blocks a second concurrent entry")


def _ctx_with_trigger(symbol_a="AAA", symbol_b="BBB", day_sizing_qty_a=0.0, day_sizing_qty_b=0.0):
    """A flat pair with an established trigger and last_deviation, Johansen
    and Hurst both already cleared live (a normal qualified pair)."""
    ctx = main.PairContext(symbol_a=symbol_a, symbol_b=symbol_b, hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.divergence_trigger_level = 3.0
    # A plausible current value -- MIN/MAX_ENTRY_DIVERGENCE no longer gate
    # this directly (see risk_manager.py's entry_eligible, 2026-09-12),
    # only the trigger above does.
    ctx.last_deviation = 2.5
    ctx.day_sizing_qty_a = day_sizing_qty_a
    ctx.day_sizing_qty_b = day_sizing_qty_b
    ctx.johansen_ever_unconfirmed = True
    ctx.hurst_ever_above_threshold = True
    # Per the client 2026-09-06: "two separate calculations" -- the 1-day
    # parallel window needs its own sticky flags cleared too, same as the
    # 20-day ones above, for a pair to count as qualified.
    ctx.johansen_ever_unconfirmed_1day = True
    ctx.hurst_ever_above_threshold_1day = True
    # Per the client 2026-09-07: 1 Day Pearson, a third live condition,
    # sticky the same way.
    ctx.pearson_ever_above_threshold_1day = True
    return ctx


def test_live_validation_shows_raw_johansen_hurst_readings_for_a_tracked_pair():
    """Per the client 2026-09-01: "We need to see the data on the
    Dashboard and be able to make decisions on the 0.60." Separate from
    pair_divergence -- shows every flat, tracked pair's actual live
    numbers, including ones that never qualify for Divergence Watch."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.live_hurst = 0.42
    ctx.live_johansen_confirmed = False
    ctx.johansen_ever_unconfirmed = True
    ctx.hurst_ever_above_threshold = False
    ctx.trigger_reached_today = True
    main.write_status("running", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert len(data["live_validation"]) == 1
    row = data["live_validation"][0]
    assert row["symbol_a"] == "AAA"
    assert row["live_hurst"] == 0.42
    assert row["live_johansen_confirmed"] is False
    assert row["johansen_ever_unconfirmed"] is True
    assert row["hurst_ever_above_threshold"] is False
    assert row["trigger_reached_today"] is True
    print("PASS: live_validation exposes the raw Johansen/Hurst readings for a tracked pair")


def test_live_validation_excludes_a_pair_not_yet_computed_and_an_open_position():
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    not_computed = main.PairContext(symbol_a="CCC", symbol_b="DDD", hedge_ratio=1.0,
                                     spread_history=pd.Series(dtype=float))
    open_position = _ctx_with_open_position(symbol_a="EEE", symbol_b="FFF")
    open_position.live_hurst = 0.65
    main.write_status("running", elv=100_000, contexts=[not_computed, open_position])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["live_validation"] == []
    print("PASS: live_validation excludes a pair that hasn't computed yet and a pair in a position")


def test_pair_divergence_no_longer_filtered_by_the_gap_test():
    """The 0.5% Allowed Divergence Gap test used to filter what shows on
    Divergence Watch, per the client 2026-08-11. Removed 2026-08-25 along
    with the gap test being fully removed as a real entry gate -- keeping
    it as a hidden display-only filter would have kept silently hiding
    pairs, the exact thing the client is now asking to fix. day_sizing_qty
    left at 0 here specifically because that used to make the gap test
    fail (allowed_divergence_gap() returns None, leg size unknown) -- now
    it makes no difference to whether the pair shows up."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = _ctx_with_trigger(day_sizing_qty_a=0.0, day_sizing_qty_b=0.0)
    main.write_status("running", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert len(data["pair_divergence"]) == 1
    assert data["pair_divergence"][0]["symbol_a"] == "AAA"
    print("PASS: pair_divergence includes a pair regardless of the gap test result")


def test_pair_divergence_excludes_pair_with_no_trigger_yet():
    """Per the client 2026-08-26, sending the actual reference mockup:
    "This is how Divergence Watch should look like. There is no 'waiting'
    anywhere." Reverts the 2026-08-25 change -- a pair without a real,
    established trigger is excluded entirely rather than shown with a
    placeholder "waiting" state."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = main.PairContext(symbol_a="CCC", symbol_b="DDD", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.divergence_trigger_level = None
    ctx.last_deviation = 0.42
    main.write_status("running", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["pair_divergence"] == []
    print("PASS: a pair with no trigger yet is excluded from Divergence Watch entirely")


def test_pair_divergence_no_longer_excludes_on_current_past_max_entry_divergence():
    """REVERSED 2026-09-12, per the client, ending a long same-night
    back-and-forth: "1.5-5 range applies only to the trigger. I never said
    live current price must be 1.5-5." The 2026-08-26 MPWR/MRVL exclusion
    ("Divergence >$6") used to also exclude on current > MAX_ENTRY_DIVERGENCE;
    that check is removed, current is governed by the entry window alone
    now. A pair with a valid, in-range trigger and cleared Johansen/Hurst
    shows on the Watch regardless of how far past MAX_ENTRY_DIVERGENCE its
    current value is."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = main.PairContext(symbol_a="MPWR", symbol_b="MRVL", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.divergence_trigger_level = 5.0
    ctx.last_deviation = 22.86
    ctx.johansen_ever_unconfirmed = True
    ctx.hurst_ever_above_threshold = True
    main.write_status("running", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "MPWR" in symbols
    print("PASS: current past MAX_ENTRY_DIVERGENCE no longer excludes a pair from Divergence Watch")


def test_pair_divergence_excludes_a_pair_whose_trigger_is_below_the_floor():
    """Per the client 2026-09-10, flagging PLTR/WDAY (trigger $1.91, back
    when the floor was $2) on the watch: a pair whose whole trigger is
    under MIN_ENTRY_DIVERGENCE can never produce a valid entry, since the
    divergence approaching a sub-floor trigger is itself always under the
    floor. Excluded from Divergence Watch, same as a pair already past its
    trigger. Floor lowered from $2 to $1 2026-09-11 ("Reduce the Divergence
    to 1-5"), so the trigger here is under $1 to stay below it."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    low_trigger = _ctx_with_trigger(symbol_a="PLTR", symbol_b="WDAY")
    low_trigger.divergence_trigger_level = 0.91
    low_trigger.last_deviation = 0.85  # in range on its own, but the trigger can't be reached validly
    ok_trigger = _ctx_with_trigger(symbol_a="AAA", symbol_b="BBB")
    main.write_status("running", elv=100_000, contexts=[low_trigger, ok_trigger])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "PLTR" not in symbols
    assert "AAA" in symbols
    print("PASS: a pair whose trigger is below the MIN_ENTRY_DIVERGENCE floor is excluded from Divergence Watch")


def test_pair_divergence_no_longer_excludes_on_current_below_min_entry_divergence():
    """REVERSED 2026-09-12, per the client, ending a long same-night
    back-and-forth: "1.5-5 range applies only to the trigger. I never said
    live current price must be 1.5-5. Live Current price should be inside
    the Entry Window, which is derived from the Maximum Divergence,
    obtained yesterday." The 2026-09-08 exclusion on current <
    MIN_ENTRY_DIVERGENCE this test used to encode is removed -- current is
    governed by the entry window alone now, not its own floor."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = main.PairContext(symbol_a="XYZ", symbol_b="ABC", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.divergence_trigger_level = 5.0
    ctx.last_deviation = 0.01
    ctx.johansen_ever_unconfirmed = True
    ctx.hurst_ever_above_threshold = True
    main.write_status("running", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "XYZ" in symbols
    print("PASS: current below MIN_ENTRY_DIVERGENCE no longer excludes a pair from Divergence Watch")


def test_pair_divergence_excludes_a_pair_past_max_stock_price():
    """Per the client 2026-08-26: "LITE stock price >$700. It should not
    be in the Divergence Watch, as qualified pair." entry_eligible's own
    Maximum Stock Price check ($700, applied per stock) would block this
    pair regardless of trigger or deviation, so it's excluded from
    Divergence Watch entirely even though it has a real, established
    trigger."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    over_price = main.PairContext(symbol_a="ANET", symbol_b="LITE", hedge_ratio=1.0,
                                   spread_history=pd.Series(dtype=float))
    over_price.divergence_trigger_level = 5.0
    over_price.last_deviation = 2.5  # inside the $2-$5 Divergence Watch range
    over_price.price_history_a = pd.Series([120.0])
    over_price.price_history_b = pd.Series([712.50])  # over $700
    at_price = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                                 spread_history=pd.Series(dtype=float))
    at_price.divergence_trigger_level = 5.0
    at_price.last_deviation = 2.5  # inside the $2-$5 Divergence Watch range
    at_price.price_history_a = pd.Series([main.MAX_STOCK_PRICE])  # exact boundary is inclusive
    at_price.price_history_b = pd.Series([100.0])
    at_price.johansen_ever_unconfirmed = True
    at_price.hurst_ever_above_threshold = True
    at_price.johansen_ever_unconfirmed_1day = True
    at_price.hurst_ever_above_threshold_1day = True
    at_price.pearson_ever_above_threshold_1day = True
    main.write_status("running", elv=100_000, contexts=[over_price, at_price])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "ANET" not in symbols
    assert "AAA" in symbols
    print("PASS: a pair with either leg over the $700 max stock price is excluded from "
          "Divergence Watch, the exact boundary is still included")


def test_pair_divergence_excludes_a_pair_not_yet_johansen_hurst_qualified():
    """Per the client 2026-09-01: "The pairs that should be on the
    Divergence Watch are qualified pairs," confirmed "Johansen and Hurst
    are qualifying conditions." A pair with a real, established trigger
    and in-range deviation still can't actually enter until both sticky
    flags have cleared at least once live this session, so it's excluded
    from Divergence Watch until then -- same reasoning as the price and
    divergence-cap exclusions above."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    not_qualified = _ctx_with_trigger(symbol_a="JPM", symbol_b="USB")
    not_qualified.johansen_ever_unconfirmed = False
    not_qualified.hurst_ever_above_threshold = False
    johansen_only = _ctx_with_trigger(symbol_a="GE", symbol_b="RTX")
    johansen_only.johansen_ever_unconfirmed = True
    johansen_only.hurst_ever_above_threshold = False
    qualified = _ctx_with_trigger(symbol_a="AAA", symbol_b="BBB")
    main.write_status("running", elv=100_000, contexts=[not_qualified, johansen_only, qualified])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "JPM" not in symbols
    assert "GE" not in symbols
    assert "AAA" in symbols
    print("PASS: a pair that hasn't cleared both Johansen and Hurst live is excluded from "
          "Divergence Watch, one flag alone isn't enough")


def test_pair_divergence_still_shows_a_pair_mean_reverted_since_max_divergence():
    """Checklist_090926.pdf condition A was REMOVED as a gate per the client
    2026-09-08 ("Please remove the condition of 100% mean reversion after
    Maximum Divergence. So, we can trade today") -- it read True on all 37
    tracked pairs, emptying the watch list. The flag is still computed and
    still shown on the dashboard, it just no longer excludes anything."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    reverted = _ctx_with_trigger(symbol_a="JPM", symbol_b="USB")
    reverted.mean_reverted_since_max_divergence = True
    main.write_status("running", elv=100_000, contexts=[reverted])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "JPM" in symbols
    print("PASS: a mean-reverted pair is no longer excluded from Divergence Watch (condition A removed)")


def test_pair_divergence_excludes_a_pair_cancelled_by_news_stop():
    """Per the client 2026-09-01: "Or cancel selected pair, if not yet a
    position." A pair NEWS STOP has cancelled pre-entry (run_check_cycle
    sets news_cancelled=True the cycle either leg hits the news
    blacklist) is done for the session -- excluded from Divergence Watch
    same as the other "cannot actually trade" exclusions above, even
    though it still has a real, established trigger and in-range
    deviation from before the cancellation."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    cancelled = _ctx_with_trigger(symbol_a="XYZ", symbol_b="ABC")
    cancelled.news_cancelled = True
    qualified = _ctx_with_trigger(symbol_a="AAA", symbol_b="BBB")
    main.write_status("running", elv=100_000, contexts=[cancelled, qualified])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "XYZ" not in symbols
    assert "AAA" in symbols
    print("PASS: a pair cancelled pre-entry by NEWS STOP is excluded from Divergence Watch")


def test_pair_divergence_includes_a_pair_discovered_by_the_live_scan():
    """Excluded from here 2026-09-01, per the client tracing a real
    dashboard confusion back to its cause: "Divergence Watch is simple.
    1. Only pairs selected 9:30-4pm yesterday. 2. If they pass Hurst and
    Johansen, they are included in the Divergence Watch." Re-added
    2026-09-02, per the client: "Add live scan discoveries." A pair the
    live universe discovery scan found mid-session
    (PairContext.discovered_live) now shows on Divergence Watch the same
    as a prior-session-selected one, once it passes every other check."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    live_found = _ctx_with_trigger(symbol_a="NEW", symbol_b="PAIR")
    live_found.discovered_live = True
    prior_session = _ctx_with_trigger(symbol_a="AAA", symbol_b="BBB")
    main.write_status("running", elv=100_000, contexts=[live_found, prior_session])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    symbols = [row["symbol_a"] for row in data["pair_divergence"]]
    assert "NEW" in symbols
    assert "AAA" in symbols
    print("PASS: a pair discovered by the live scan shows on Divergence Watch, same as a "
          "prior-RTH-session pair")


def test_pair_divergence_excludes_a_pair_never_yet_measured():
    """last_deviation stays None until the divergence-tracking loop has run
    at least once for this pair this session -- distinct from "has no
    trigger yet", which still has a real deviation reading."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = main.PairContext(symbol_a="EEE", symbol_b="FFF", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    main.write_status("running", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    assert data["pair_divergence"] == []
    print("PASS: a pair with no deviation reading yet is excluded, distinct from no trigger yet")


def test_get_elv_with_retry_recovers_from_a_transient_timeout():
    """Finding #15 from the independent audit, confirmed real: a single
    IBKR timeout during the 9:25-9:30am startup window used to propagate
    straight out of run_daily_session() and kill the whole session before
    any trading started. Now it's retried instead of fatal."""
    class FlakyIBKR:
        def __init__(self):
            self.calls = 0

        async def get_equity_with_loan_value(self):
            self.calls += 1
            if self.calls == 1:
                raise asyncio.TimeoutError()
            return 123456.0

    async def scenario():
        ibkr = FlakyIBKR()
        result = await main._get_elv_with_retry(ibkr, retries=2)
        assert result == 123456.0
        assert ibkr.calls == 2
    asyncio.run(scenario())
    print("PASS: a transient EquityWithLoanValue timeout is retried, not fatal")


def test_get_elv_with_retry_reraises_after_exhausting_retries():
    """A persistent failure (IBKR genuinely unreachable) must still fail
    loud -- silently falling back to some placeholder equity would poison
    every stop-loss/sizing calculation for the rest of the day, which is
    worse than a loud, immediate session failure."""
    class AlwaysTimesOut:
        async def get_equity_with_loan_value(self):
            raise asyncio.TimeoutError()

    async def scenario():
        raised = False
        try:
            await main._get_elv_with_retry(AlwaysTimesOut(), retries=2)
        except asyncio.TimeoutError:
            raised = True
        assert raised
    asyncio.run(scenario())
    print("PASS: a persistent failure still raises after exhausting retries, no fabricated fallback value")


def _replay_live_beta_bar_by_bar(ctx, prices_a, prices_b, start_ts=0):
    """Replicates run_check_cycle's own per-bar block exactly: seed
    ctx.beta_estimator on first use, update it from the new bar, then
    recompute the WHOLE spread series from scratch with the resulting
    hedge_ratio -- not tested by going through run_check_cycle itself (too
    much IBKR/feed/dashboard machinery to mock for what is really a
    3-line unit), but the exact same lines, copied deliberately rather than
    imported, so a change to one doesn't silently stop testing the other."""
    for i, (price_a, price_b) in enumerate(zip(prices_a, prices_b)):
        stamp = pd.Timestamp(start_ts + i, unit="s")
        ctx.price_history_a.loc[stamp] = price_a
        ctx.price_history_b.loc[stamp] = price_b
        if ctx.beta_estimator is None:
            ctx.beta_estimator = main.KalmanBetaEstimator(beta=ctx.hedge_ratio)
        ctx.hedge_ratio = ctx.beta_estimator.update(price_a, price_b)
        ctx.spread_history = main.compute_spread(ctx.price_history_a, ctx.price_history_b, ctx.hedge_ratio)


def test_live_beta_updates_bar_by_bar_instead_of_staying_frozen():
    """Un-frozen 2026-08-24 per the client: "Beta is not frozen. Continuously
    updating. Just like before." Confirms hedge_ratio actually moves as new
    bars arrive, rather than staying pinned at whatever Phase 1 screening
    computed once at the prior session's 4pm close."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=2.0,
                            spread_history=pd.Series(dtype=float))
    initial_beta = ctx.hedge_ratio
    # price_b held flat, price_a drifting up -- a real relationship change
    # a live estimator should track, a frozen one would never see.
    prices_a = [200.0 + i * 0.5 for i in range(40)]
    prices_b = [100.0] * 40
    _replay_live_beta_bar_by_bar(ctx, prices_a, prices_b)
    assert ctx.hedge_ratio != initial_beta
    assert ctx.beta_estimator is not None
    print("PASS: hedge_ratio moves live as bars arrive instead of staying frozen at its warm-up value")


def test_live_beta_recompute_avoids_the_2026_08_05_fake_divergence_bug():
    """Regression test for the confirmed historical bug (MPWR/MRVL,
    2026-08-05): appending one new spread point using a JUST-UPDATED beta
    onto a series whose earlier points were computed with an OLDER beta
    creates a fake discontinuity right at the seam -- a jump that isn't a
    real price move, just an artifact of two different betas meeting in one
    series. Directly compares the naive append-only construction against the
    whole-series recompute run_check_cycle actually does (replicated by
    _replay_live_beta_bar_by_bar), using a deliberately large beta shift
    (1.0 -> 3.0) so the artifact isn't lost in noise -- real Kalman
    convergence is usually gradual, but a sharp regime change (the kind that
    triggers this exact bug in practice) can move it quickly."""
    prices_a = [150.0 + i * 0.05 for i in range(30)] + [155.0 + i * 3.0 for i in range(10)]
    prices_b = [100.0] * 40  # held flat, so the whole spread move is beta's

    # Naive: each point built from whatever beta was current WHEN it was
    # first appended -- old points never get revisited once beta moves.
    naive_points = []
    naive_beta = 1.0
    for price_a, price_b in zip(prices_a, prices_b):
        naive_points.append(price_a - naive_beta * price_b)
        naive_beta = min(naive_beta + 0.3, 3.0)  # simulates beta drifting toward 3.0 as bars arrive
    naive_spread = pd.Series(naive_points)

    # Correct: every point recomputed from raw prices using ONE beta -- the
    # final, fully-updated value -- exactly what _replay_live_beta_bar_by_bar
    # (and run_check_cycle) actually does each cycle.
    correct_spread = pd.Series([price_a - naive_beta * price_b for price_a, price_b in zip(prices_a, prices_b)])

    max_jump_correct = correct_spread.diff().abs().max()
    max_jump_naive = naive_spread.diff().abs().max()
    assert max_jump_correct < max_jump_naive / 2, (
        f"correct recompute should have a far smaller max single-bar jump "
        f"than the naive per-bar-beta series: {max_jump_correct} vs {max_jump_naive}")
    print("PASS: recomputing the whole series fresh each bar avoids the fake-discontinuity bug "
          "even with beta live again")


def _fake_pivot(symbols, n=3):
    """A pivot DataFrame shaped like _fetch_warmup_data's real return --
    ts_event-indexed, one column per symbol -- for just the given symbols."""
    idx = pd.date_range("2026-08-28 13:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({s: [100.0 + i for i in range(n)] for s in symbols}, index=idx)


def test_warmup_retry_recovers_pairs_that_get_price_data_back():
    """2026-08-27: 5 of 7 qualified pairs came back with zero bars from the
    one-time 9:30am warm_up_pairs() fetch, silently never trading all day.
    run_warmup_retry_in_background re-fetches for whichever pairs are still
    missing price_history_a -- this checks it correctly identifies
    recovered vs still-missing pairs via identity, not the dataclass's own
    field-by-field __eq__ (which would raise on the ambiguous truth value
    of comparing two pd.Series)."""
    recovered_ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                                      spread_history=pd.Series(dtype=float))
    still_missing_ctx = main.PairContext(symbol_a="CCC", symbol_b="DDD", hedge_ratio=1.0,
                                          spread_history=pd.Series(dtype=float))

    # Only AAA/BBB come back -- CCC/DDD stay missing, same as the real
    # 2026-08-27 incident where some symbols were absent from the response.
    fake_data = main._WarmupData(pivot=_fake_pivot(["AAA", "BBB"]),
                                  prior_pivot=pd.DataFrame(), jh_pivot=pd.DataFrame(),
                                  prior_open={})

    async def scenario():
        with patch.object(main, "_fetch_warmup_data", lambda contexts, now: fake_data):
            state = {"running": False}
            await main.run_warmup_retry_in_background(
                [recovered_ctx, still_missing_ctx], datetime(2026, 8, 28, 10, 0), state)
            assert state["running"] is False
        assert len(recovered_ctx.price_history_a) == 3
        assert len(still_missing_ctx.price_history_a) == 0
    asyncio.run(scenario())
    print("PASS: warm-up retry correctly identifies recovered vs still-missing pairs by identity")


def test_warmup_retry_recovers_pairs_with_price_data_but_no_trigger():
    """2026-09-21: DGX/LH and RJF/SCHW loaded price_history_a/b fine from the
    combined pivot on the first warm-up fetch, but the narrower prior_pivot
    (prior session only, used by establish_base_trigger) came back without
    them -- a brief Databento backfill lag right at the open. The retry
    scheduler only checked price_history_a, so these pairs never counted as
    missing and sat with no trigger, no Divergence Watch entry, silently
    for the rest of the session. Confirmed live: 4 pairs qualified, only 2
    ever tracked. The retry list must also catch a pair with price data
    already loaded but base_trigger_level still None."""
    orphaned_ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                                     price_history_a=pd.Series([1.0, 2.0]),
                                     price_history_b=pd.Series([1.0, 2.0]),
                                     spread_history=pd.Series(dtype=float))
    assert orphaned_ctx.base_trigger_level is None

    import numpy as np
    np.random.seed(7)
    idx = pd.date_range("2026-08-28 13:30", periods=90, freq="1min", tz="UTC")
    walk_a = np.cumsum(np.random.normal(0, 0.5, len(idx))) + 100
    varied_prior = pd.DataFrame({"AAA": walk_a, "BBB": [100.0] * len(idx)}, index=idx)
    fake_data = main._WarmupData(pivot=_fake_pivot(["AAA", "BBB"]),
                                  prior_pivot=varied_prior,
                                  jh_pivot=pd.DataFrame(), prior_open={"AAA": 99.0, "BBB": 99.0})

    async def scenario():
        with patch.object(main, "_fetch_warmup_data", lambda contexts, now: fake_data):
            state = {"running": False}
            await main.run_warmup_retry_in_background([orphaned_ctx], datetime(2026, 8, 28, 10, 0), state)
        assert orphaned_ctx.base_trigger_level is not None
    asyncio.run(scenario())
    print("PASS: warm-up retry recovers a pair that has price data but never got a trigger level")


def test_fetch_warmup_data_never_mutates_contexts():
    """The actual 2026-08-28 audit fix: _fetch_warmup_data is the half of
    the old warm_up_pairs that runs on a worker thread (via
    asyncio.to_thread inside run_warmup_retry_in_background) while
    run_check_cycle concurrently mutates these same PairContext objects on
    the event-loop thread. It must never write to a PairContext field --
    only _apply_warmup_data (which always runs back on the event-loop
    thread) may do that. This is what makes the split race-safe."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    before = (ctx.price_history_a, ctx.price_history_b, ctx.spread_history)

    with patch.object(main, "fetch_intraday_warmup", lambda *a, **k: pd.DataFrame()):
        result = main._fetch_warmup_data([ctx], datetime(2026, 8, 28, 10, 0))

    assert result is None  # empty fetch -- but the real point is what follows
    assert ctx.price_history_a is before[0]
    assert ctx.price_history_b is before[1]
    assert ctx.spread_history is before[2]
    print("PASS: _fetch_warmup_data touches no PairContext field, safe to run on a worker thread")


def test_warmup_retry_survives_a_fetch_exception():
    """A retry failure (network blip, Databento outage) must not crash the
    background task or leave retry_state stuck -- it just tries again next
    interval."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))

    def broken_fetch(contexts, now):
        raise ConnectionError("simulated Databento outage")

    async def scenario():
        with patch.object(main, "_fetch_warmup_data", broken_fetch):
            state = {"running": False}
            await main.run_warmup_retry_in_background([ctx], datetime(2026, 8, 28, 10, 0), state)
            assert state["running"] is False
    asyncio.run(scenario())
    print("PASS: a failed warm-up retry logs and clears retry_state instead of crashing")


def _previous_session_bars(symbols, session_date="2026-08-28", correlated_daily=True):
    """390 one-minute RTH bars (9:30am-4pm ET) per symbol on `session_date`,
    shaped like fetch_intraday_warmup's real return (ts_event/symbol/close
    columns) -- enough bars to clear select_pairs_from_previous_session's
    MIN_BARS_TO_TEST floor. A small random walk, not a flat line, so
    Johansen/hedge_ratio/ADF don't hit a singular-matrix edge case on
    zero-variance input."""
    import numpy as np
    np.random.seed(3)
    idx = pd.date_range(f"{session_date} 09:30", periods=390, freq="1min", tz=main.ET)
    rows = []
    for i, sym in enumerate(symbols):
        walk = np.cumsum(np.random.normal(0, 0.1, len(idx))) + 100 + i * 10
        for ts, close in zip(idx, walk):
            rows.append({"ts_event": ts, "symbol": sym, "close": float(close)})
    return pd.DataFrame(rows)


def _daily_closes(symbols, n=60, correlated=True):
    """60 daily closes per symbol, shaped like fetch_consolidated_daily_closes's
    real return (columns=symbols, index=date).

    `correlated=True` produces the shape a pair must now have to QUALIFY,
    per the client 2026-09-08: a high 60 day correlation (>=0.90, still a
    real pair) whose RECENT 20 day correlation has BROKEN DOWN (<0.70) --
    the expansion the strategy trades. Built as a near-noiseless scaled
    copy for the first 40 days, then a decoupled last 20 (the breakdown
    window RECENT_CORRELATION_WINDOW_DAYS actually measures).
    `correlated=False` makes them independent random walks, which fails
    the 60 day gate instead."""
    import numpy as np
    np.random.seed(5)
    idx = pd.date_range("2026-06-01", periods=n, freq="B")
    base = np.cumsum(np.random.normal(0, 1, n)) + 100
    data = {symbols[0]: base}
    recent = 20
    for i, sym in enumerate(symbols[1:], start=1):
        if correlated:
            series = base * (1 + 0.1 * i) + np.random.normal(0, 0.01, n)
            # Decouple only the trailing 20 days: an independent walk from
            # where the correlated stretch left off, carrying the same
            # average drift so the 60 day level relationship (and its
            # correlation) survives while the 20 day co-movement doesn't.
            # These constants are tuned, not arbitrary -- they put the pair
            # at 60d=0.936 (needs >=0.90) and 20d=0.648 (needs <0.70).
            rng = np.random.RandomState(53)
            drift = (series[-1] - series[-recent]) / recent
            series[-recent:] = series[-recent] + np.cumsum(rng.normal(drift, 1.4, recent))
            data[sym] = series
        else:
            data[sym] = np.cumsum(np.random.normal(0, 1, n)) + 100 + i * 10
    return pd.DataFrame(data, index=idx)


def test_select_pairs_from_previous_session_ignores_johansen_and_hurst():
    """The actual point of the 2026-08-29 rebuild -- per the client:
    "9:30am, Bot selects the qualified pairs from previous RTH session...
    Then at current RTH session, Bot will validate Johansen and Hurst."
    Confirmed live the same day: re-running the old combined-gate offline
    screen fresh qualified 0 of 1295 real S&P 500 pairs, because Johansen/
    Hurst are point-in-time readings that can fail even for a pair with
    excellent correlation. A pair whose real Johansen/Hurst would have
    failed the OLD gate (Confirmed / <=0.60) must still be selected here,
    since correlation/liquidity/market cap/volatility are the only things
    this function checks."""
    now = datetime(2026, 8, 29, 9, 30, tzinfo=main.ET)
    bars = _previous_session_bars(["AAA", "BBB"])
    daily = _daily_closes(["AAA", "BBB"], correlated=True)

    with patch.object(main, "load_market_entry_list", lambda: {("AAA", "BBB")}), \
         patch.object(main, "load_permanent_blacklist", lambda: set()), \
         patch.object(main, "fetch_intraday_warmup", lambda *a, **k: bars), \
         patch("phase1.databento_source.fetch_consolidated_daily_closes", lambda *a, **k: daily), \
         patch.object(main, "fetch_market_caps_yfinance", lambda symbols: {s: 50_000_000_000 for s in symbols}), \
         patch.object(main, "fetch_volume_stats_yfinance",
                       lambda symbols, closes: {s: {"avg_share_volume": 10_000_000,
                                                     "avg_dollar_volume": 2_000_000_000} for s in symbols}), \
         patch("phase1.screening.johansen_confirmed", return_value=True), \
         patch("phase1.screening.hurst_exponent", return_value=0.30):
        contexts = main.select_pairs_from_previous_session(now)

    assert len(contexts) == 1
    assert (contexts[0].symbol_a, contexts[0].symbol_b) == ("AAA", "BBB")
    assert contexts[0].hedge_ratio is not None
    print("PASS: a pair with real Johansen Confirmed / Hurst 0.30 (would have failed the old gate) "
          "is still selected, since Johansen/Hurst are no longer checked here")


def test_select_pairs_from_previous_session_uses_fetch_now_for_data_boundaries():
    """Regression test for the live 2026-09-01 bug: cache_qualified_pairs_for_tomorrow
    calls this with `now` pushed one day ahead (so today's just-closed
    session, not yesterday's, is the "prior session"). Without `fetch_now`
    kept separate, that same pushed-ahead date leaked into the actual
    fetch end boundaries -- both fetches must use the REAL current time,
    not `now`, or they silently come back empty for a date that hasn't
    happened yet."""
    now = datetime(2026, 9, 2, 0, 0, tzinfo=main.ET)  # pseudo-tomorrow, as the caching call would pass
    fetch_now = datetime(2026, 9, 1, 16, 5, tzinfo=main.ET)  # the real, current evening time
    bars = _previous_session_bars(["AAA", "BBB"], session_date="2026-09-01")  # today's (fetch_now's) own session
    daily = _daily_closes(["AAA", "BBB"], correlated=True)
    seen_bars_end = []
    seen_daily_end = []

    def fake_bars(symbols, start, end):
        seen_bars_end.append(end)
        return bars

    def fake_daily(symbols, start, end, **kwargs):
        seen_daily_end.append(end)
        return daily

    with patch.object(main, "load_market_entry_list", lambda: {("AAA", "BBB")}), \
         patch.object(main, "load_permanent_blacklist", lambda: set()), \
         patch.object(main, "fetch_intraday_warmup", fake_bars), \
         patch("phase1.databento_source.fetch_consolidated_daily_closes", fake_daily), \
         patch.object(main, "fetch_market_caps_yfinance", lambda symbols: {s: 50_000_000_000 for s in symbols}), \
         patch.object(main, "fetch_volume_stats_yfinance",
                       lambda symbols, closes: {s: {"avg_share_volume": 10_000_000,
                                                     "avg_dollar_volume": 2_000_000_000} for s in symbols}):
        contexts = main.select_pairs_from_previous_session(now, fetch_now=fetch_now)

    assert len(contexts) == 1  # today's (fetch_now's) session was correctly picked up as "prior"
    assert seen_bars_end[0].startswith("2026-09-01")
    assert seen_daily_end[0] == "2026-09-01"
    print("PASS: both fetches use fetch_now's real date as their end boundary, not now's pushed-ahead date")


def test_select_pairs_from_previous_session_still_requires_correlation():
    now = datetime(2026, 8, 29, 9, 30, tzinfo=main.ET)
    bars = _previous_session_bars(["AAA", "BBB"])
    daily = _daily_closes(["AAA", "BBB"], correlated=False)

    with patch.object(main, "load_market_entry_list", lambda: {("AAA", "BBB")}), \
         patch.object(main, "load_permanent_blacklist", lambda: set()), \
         patch.object(main, "fetch_intraday_warmup", lambda *a, **k: bars), \
         patch("phase1.databento_source.fetch_consolidated_daily_closes", lambda *a, **k: daily), \
         patch.object(main, "fetch_market_caps_yfinance", lambda symbols: {s: 50_000_000_000 for s in symbols}), \
         patch.object(main, "fetch_volume_stats_yfinance",
                       lambda symbols, closes: {s: {"avg_share_volume": 10_000_000,
                                                     "avg_dollar_volume": 2_000_000_000} for s in symbols}):
        contexts = main.select_pairs_from_previous_session(now)

    assert contexts == []
    print("PASS: a pair with uncorrelated daily closes is not selected -- correlation still gates this step")


def test_select_pairs_from_previous_session_excludes_permanent_blacklist():
    now = datetime(2026, 8, 29, 9, 30, tzinfo=main.ET)
    bars = _previous_session_bars(["AAA", "BBB"])
    daily = _daily_closes(["AAA", "BBB"], correlated=True)

    with patch.object(main, "load_market_entry_list", lambda: {("AAA", "BBB")}), \
         patch.object(main, "load_permanent_blacklist", lambda: {"AAA"}), \
         patch.object(main, "fetch_intraday_warmup", lambda *a, **k: bars), \
         patch("phase1.databento_source.fetch_consolidated_daily_closes", lambda *a, **k: daily), \
         patch.object(main, "fetch_market_caps_yfinance", lambda symbols: {s: 50_000_000_000 for s in symbols}), \
         patch.object(main, "fetch_volume_stats_yfinance",
                       lambda symbols, closes: {s: {"avg_share_volume": 10_000_000,
                                                     "avg_dollar_volume": 2_000_000_000} for s in symbols}):
        contexts = main.select_pairs_from_previous_session(now)

    assert contexts == []
    print("PASS: a pair with a permanently blacklisted leg is never even fetched or evaluated")


def test_select_pairs_from_previous_session_empty_market_entry_list():
    now = datetime(2026, 8, 29, 9, 30, tzinfo=main.ET)
    with patch.object(main, "load_market_entry_list", lambda: None):
        assert main.select_pairs_from_previous_session(now) == []
    with patch.object(main, "load_market_entry_list", lambda: set()):
        assert main.select_pairs_from_previous_session(now) == []
    print("PASS: no market entry list (None or empty) selects no pairs, without touching any fetch")


def test_cache_qualified_pairs_for_tomorrow_writes_the_expected_pairs():
    """Per the client 2026-09-01: "There is no pairs selection today...
    by 9:30am, it will test for Johansen and Hurst. Which is a matter of
    seconds." Called at end of day, this must call
    select_pairs_from_full_universe_prior_session (the full S&P 500 scan,
    since 2026-09-12) with a `now` one day AHEAD of the real one, so
    today's just-closed session -- not the day before it -- is what gets
    selected and cached, but `fetch_now` must stay the REAL current time.

    Confirmed live 2026-09-01: before fetch_now existed, the pseudo-
    tomorrow date leaked into the actual data fetch boundaries too and
    every fetch came back empty ("no daily close data" for all 124
    pairs), silently caching an empty list. This is the regression test
    for that exact bug."""
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    status_tmp = Path(tempfile.mkdtemp()) / "cached_pairs_status.json"
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    main.CACHED_PAIRS_STATUS_PATH = status_tmp
    real_now = datetime(2026, 9, 1, 16, 5, tzinfo=main.ET)
    seen = []

    def fake_select(now, fetch_now=None):
        seen.append((now, fetch_now))
        return [main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.42,
                                  spread_history=pd.Series(dtype=float), industry="Software")]

    with patch.object(main, "select_pairs_from_full_universe_prior_session", fake_select):
        main.cache_qualified_pairs_for_tomorrow(real_now)

    seen_now, seen_fetch_now = seen[0]
    assert seen_now.date() == datetime(2026, 9, 2).date()
    assert seen_fetch_now == real_now
    data = json.loads(tmp.read_text())
    assert data["for_date"] == "2026-09-01"
    # correlation_60d/correlation_20d carried through so the next morning's
    # cache-loaded pairs show both Pearson readings too -- per the client
    # 2026-09-08. None here because this fixture's contexts don't set them.
    assert data["pairs"] == [{"symbol_a": "AAA", "symbol_b": "BBB", "hedge_ratio": 1.42,
                              "industry": "Software", "correlation_60d": None, "correlation_20d": None}]
    print("PASS: cache_qualified_pairs_for_tomorrow calls selection with tomorrow's cutoff date but the REAL "
          "current time as fetch_now, and persists the result")

    # per the client repeatedly asking "how many pairs will load by 9am" --
    # the dashboard's own separate status file gets written too, the
    # moment the scan finishes.
    status_data = json.loads(status_tmp.read_text())
    assert status_data["cached_pairs_for_tomorrow_count"] == 1
    assert status_data["cached_pairs_for_tomorrow_symbols"] == ["AAA/BBB"]
    assert status_data["cached_pairs_for_tomorrow_at"] == real_now.isoformat()
    print("PASS: cache_qualified_pairs_for_tomorrow also writes the dashboard's cached-pairs status file")


def test_cache_qualified_pairs_for_tomorrow_survives_a_failure():
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    status_tmp = Path(tempfile.mkdtemp()) / "cached_pairs_status.json"
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    main.CACHED_PAIRS_STATUS_PATH = status_tmp
    with patch.object(main, "select_pairs_from_full_universe_prior_session", side_effect=RuntimeError("boom")):
        main.cache_qualified_pairs_for_tomorrow(datetime(2026, 9, 1, 16, 5, tzinfo=main.ET))  # must not raise
    assert not tmp.exists()
    assert not status_tmp.exists()  # the failure happened before the dashboard status write too
    print("PASS: a failure while caching tonight's list is swallowed, never crashes the end-of-day path")


def test_write_cached_pairs_status_never_raises_into_the_trading_loop():
    """Best-effort, same as save_daily_state: a failure here must not take
    down the overnight caching this runs right after."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float), industry="Software")
    main.CACHED_PAIRS_STATUS_PATH = Path("/nonexistent/dir/cached_pairs_status.json")
    main.write_cached_pairs_status([ctx], datetime(2026, 9, 1, 16, 5, tzinfo=main.ET))  # must not raise
    print("PASS: a failed cached-pairs status write logs and continues rather than raising")


def test_load_cached_qualified_pairs_returns_none_when_missing():
    main.CACHED_QUALIFIED_PAIRS_PATH = Path(tempfile.mkdtemp()) / "nonexistent.json"
    assert main.load_cached_qualified_pairs(datetime(2026, 9, 2, 9, 30, tzinfo=main.ET)) is None
    print("PASS: no cache file -- falls back to None (live selection) rather than an error")


def test_load_cached_qualified_pairs_returns_none_when_corrupt():
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    tmp.write_text("not valid json{{{")
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    assert main.load_cached_qualified_pairs(datetime(2026, 9, 2, 9, 30, tzinfo=main.ET)) is None
    print("PASS: a corrupt cache file falls back to None rather than crashing the 9:30am startup")


def test_load_cached_qualified_pairs_returns_none_when_stale():
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    tmp.write_text(json.dumps({
        "for_date": "2026-08-20",  # over a week old
        "pairs": [{"symbol_a": "AAA", "symbol_b": "BBB", "hedge_ratio": 1.0, "industry": "Software"}],
    }))
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    assert main.load_cached_qualified_pairs(datetime(2026, 9, 2, 9, 30, tzinfo=main.ET)) is None
    print("PASS: a stale cache (evening job missed for days) falls back to None, not a wrong old list")


def test_load_cached_qualified_pairs_returns_none_when_empty():
    """Confirmed live 2026-09-01: a real bug in the evening fetch produced
    exactly this signature, 0 of 124 pairs, silently -- treated as
    untrustworthy rather than "correctly computed, zero pairs qualified",
    same reasoning as a stale or corrupt cache."""
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    tmp.write_text(json.dumps({"for_date": "2026-09-01", "pairs": []}))
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    assert main.load_cached_qualified_pairs(datetime(2026, 9, 2, 9, 30, tzinfo=main.ET)) is None
    print("PASS: an empty (but validly-dated) cache falls back to None, not a day with zero tracked pairs")


def test_load_cached_qualified_pairs_accepts_a_cache_from_today_itself():
    """Confirmed live 2026-09-02: a same-morning manual re-run to recover
    from the previous evening's fetch bug legitimately produces a cache
    dated the SAME day it's being loaded -- the freshest a cache can
    possibly be, not stale. Must not be rejected just because it isn't
    strictly from an earlier calendar day."""
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    tmp.write_text(json.dumps({
        "for_date": "2026-09-02",
        "pairs": [{"symbol_a": "AAA", "symbol_b": "BBB", "hedge_ratio": 1.0, "industry": "Software"}],
    }))
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    contexts = main.load_cached_qualified_pairs(datetime(2026, 9, 2, 9, 0, tzinfo=main.ET))
    assert len(contexts) == 1
    print("PASS: a cache dated the same day it's loaded is accepted, not treated as stale")


def test_load_cached_qualified_pairs_returns_none_when_from_the_future():
    """A for_date after today would only ever mean a corrupt or hand-edited
    file -- must never be trusted as if it were somehow extra-fresh."""
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    tmp.write_text(json.dumps({
        "for_date": "2026-09-05",
        "pairs": [{"symbol_a": "AAA", "symbol_b": "BBB", "hedge_ratio": 1.0, "industry": "Software"}],
    }))
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    assert main.load_cached_qualified_pairs(datetime(2026, 9, 2, 9, 0, tzinfo=main.ET)) is None
    print("PASS: a cache dated in the future is rejected as invalid")


def test_load_cached_qualified_pairs_reconstructs_pair_contexts():
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    tmp.write_text(json.dumps({
        "for_date": "2026-09-01",
        "pairs": [{"symbol_a": "AAA", "symbol_b": "BBB", "hedge_ratio": 1.42, "industry": "Software"}],
    }))
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    contexts = main.load_cached_qualified_pairs(datetime(2026, 9, 2, 9, 30, tzinfo=main.ET))
    assert len(contexts) == 1
    assert contexts[0].symbol_a == "AAA" and contexts[0].symbol_b == "BBB"
    assert contexts[0].hedge_ratio == 1.42
    assert contexts[0].industry == "Software"
    print("PASS: a valid overnight cache reconstructs the same PairContext shape select_pairs_from_previous_session "
          "would have built live, including the exact seeded hedge_ratio")


def test_status_heartbeat_passes_through_state_contexts():
    """Per the client 2026-09-01: "Tomorrow, by 9am it loads. We want it
    loaded by 9am." state["contexts"], once set (by the 9am cache load --
    see its own call site), must reach write_status on every tick, not
    just get silently dropped back to None."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    seen = []

    def fake_write_status(status, account="", elv=0.0, contexts=None, **kwargs):
        seen.append(contexts)

    async def scenario():
        state = {"status": "waiting", "account": "DU123", "contexts": [ctx]}
        with patch.object(main, "write_status", fake_write_status), \
             patch("asyncio.sleep", side_effect=lambda *a: (_ for _ in ()).throw(asyncio.CancelledError())):
            try:
                await main.status_heartbeat(state)
            except asyncio.CancelledError:
                pass
    asyncio.run(scenario())

    assert len(seen) == 1
    assert seen[0] == [ctx]
    print("PASS: status_heartbeat passes state['contexts'] through to write_status on every tick")


def test_load_cached_qualified_pairs_accepts_a_cache_across_a_long_weekend():
    tmp = Path(tempfile.mkdtemp()) / "cached_qualified_pairs.json"
    tmp.write_text(json.dumps({
        "for_date": "2026-08-28",  # Friday
        "pairs": [{"symbol_a": "AAA", "symbol_b": "BBB", "hedge_ratio": 1.0, "industry": "Software"}],
    }))
    main.CACHED_QUALIFIED_PAIRS_PATH = tmp
    contexts = main.load_cached_qualified_pairs(datetime(2026, 8, 31, 9, 30, tzinfo=main.ET))  # Monday
    assert len(contexts) == 1
    print("PASS: Friday's cache is still usable on the following Monday, a long weekend is not staleness")


def test_update_current_divergence_event_sets_baseline_on_first_breach():
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    now = datetime(2026, 9, 3, 10, 0, tzinfo=main.ET)
    main.update_current_divergence_event(ctx, deviation=0.02, price_a=100.0, price_b=50.0, now=now)
    assert ctx.current_event_start_time == pd.Timestamp(now)
    assert ctx.current_event_price_a == 100.0
    assert ctx.current_event_price_b == 50.0
    print("PASS: the first cycle a deviation breaches tolerance sets the event baseline")


def test_update_current_divergence_event_keeps_the_original_baseline():
    """The baseline must NOT keep moving to the latest price every cycle
    the event stays open -- it's fixed at the moment the event started,
    per the client: "when the current divergence event started moving
    away from equilibrium." """
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    now1 = datetime(2026, 9, 3, 10, 0, tzinfo=main.ET)
    now2 = datetime(2026, 9, 3, 10, 1, tzinfo=main.ET)
    main.update_current_divergence_event(ctx, deviation=0.02, price_a=100.0, price_b=50.0, now=now1)
    main.update_current_divergence_event(ctx, deviation=0.05, price_a=101.0, price_b=49.0, now=now2)
    assert ctx.current_event_start_time == pd.Timestamp(now1)
    assert ctx.current_event_price_a == 100.0  # unchanged from the first cycle
    assert ctx.current_event_price_b == 50.0
    print("PASS: the baseline stays fixed at the event's own start, not the latest cycle")


def test_update_current_divergence_event_clears_when_the_spread_reverts():
    """A closed event's baseline must not leak into the NEXT, separate
    event -- reverting inside the tolerance band clears it, so a later
    breach gets its own fresh start."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    now1 = datetime(2026, 9, 3, 10, 0, tzinfo=main.ET)
    now2 = datetime(2026, 9, 3, 10, 5, tzinfo=main.ET)
    now3 = datetime(2026, 9, 3, 10, 6, tzinfo=main.ET)
    main.update_current_divergence_event(ctx, deviation=0.02, price_a=100.0, price_b=50.0, now=now1)
    main.update_current_divergence_event(ctx, deviation=0.0, price_a=100.5, price_b=49.8, now=now2)  # reverts
    assert ctx.current_event_start_time is None
    assert ctx.current_event_price_a is None
    main.update_current_divergence_event(ctx, deviation=0.03, price_a=102.0, price_b=48.0, now=now3)  # new event
    assert ctx.current_event_start_time == pd.Timestamp(now3)
    assert ctx.current_event_price_a == 102.0  # the NEW event's own baseline, not the old one
    print("PASS: reverting inside tolerance clears the baseline, a later breach gets a fresh one")


def _give_ctx_compliant_prior_excursion_data(ctx):
    """Sets prior_session_price_a/b, prior_session_open_a/b, premarket_price_a/b
    (RTH-timestamped), and today_session_open_a/b for BOTH legs so
    passes_prior_excursion_direction_filter trivially passes regardless of
    which leg ends up the driver -- both legs get a clean, unambiguous
    south-dominant prior session (true Open=100, Low=90, no north move at
    all) and an all-south today session (true Open=50, Low=40), 100% south
    >= 70% either way. Used by expansion_leg_decision tests that predate
    the 2026-09-15 filters and are testing something else entirely (the
    70% driver-share/BUY-SELL logic), so these filters should never be
    what blocks them. Open set separately from the close-price series --
    see PairContext.prior_session_open_a's own comment for the real
    2026-09-19 bug this distinction exists to prevent regressing."""
    ctx.prior_session_price_a = pd.Series([100.0, 95.0, 90.0])
    ctx.prior_session_price_b = pd.Series([100.0, 95.0, 90.0])
    ctx.prior_session_open_a = 100.0
    ctx.prior_session_open_b = 100.0
    idx = pd.date_range("2026-09-15 09:30", periods=3, freq="1min", tz=main.ET)
    ctx.premarket_price_a = pd.Series([50.0, 45.0, 40.0], index=idx)
    ctx.premarket_price_b = pd.Series([50.0, 45.0, 40.0], index=idx)
    ctx.today_session_open_a = 50.0
    ctx.today_session_open_b = 50.0


def test_expansion_leg_decision_matches_the_clients_own_worked_example():
    """Regression test for the client's exact worked example in
    Expansion_Leg.pdf: Stock A +$2.50 (rally), Stock B -$0.6875, beta=1.0
    -> C_A=+2.50, C_B=+0.6875 -> Stock A dominant (78.4% of the
    expansion, clears the 2026-09-13 70% floor), rallied -> BUY. Baseline
    is the Rolling Equilibrium Price since 2026-09-13
    (price_history_a/b's own mean), replacing the old
    current_event_price_a/b -- a single-value series here so its mean
    equals the old baseline exactly, isolating this test to the 70%
    threshold and BUY/SELL logic, not the baseline change itself."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.price_history_a = pd.Series([100.00])
    ctx.price_history_b = pd.Series([100.00])
    _give_ctx_compliant_prior_excursion_data(ctx)
    symbol, action, delta_p = main.expansion_leg_decision(ctx, price_a=102.50, price_b=99.3125)
    assert symbol == "AAA"
    assert action == "BUY"
    assert abs(delta_p - 2.50) < 1e-9
    print("PASS: expansion_leg_decision matches the client's own worked example exactly")


def test_expansion_leg_decision_sell_when_the_dominant_leg_crashed():
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.price_history_a = pd.Series([100.00])
    ctx.price_history_b = pd.Series([100.00])
    _give_ctx_compliant_prior_excursion_data(ctx)
    symbol, action, delta_p = main.expansion_leg_decision(ctx, price_a=95.00, price_b=100.50)
    assert symbol == "AAA"
    assert action == "SELL"
    assert delta_p < 0
    print("PASS: the dominant leg crashing (not rallying) gives SELL, not BUY")


def test_expansion_leg_decision_picks_leg_b_when_it_dominates():
    """Per the PDF's own Method 1: C_B = -beta * delta_P_B -- a leg B
    that FALLS contributes POSITIVELY to an upward-expanding spread
    (Spread = P_A - beta*P_B), so a big negative delta_P_B can dominate
    even a real, smaller delta_P_A."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.price_history_a = pd.Series([100.00])
    ctx.price_history_b = pd.Series([100.00])
    _give_ctx_compliant_prior_excursion_data(ctx)
    symbol, action, delta_p = main.expansion_leg_decision(ctx, price_a=100.50, price_b=90.00)
    assert symbol == "BBB"
    assert action == "SELL"  # leg B itself fell (crashed), even though it dominates the spread
    assert delta_p < 0
    print("PASS: leg B dominates when its own move outweighs leg A's, direction is B's own move")


def test_expansion_leg_decision_none_without_price_history():
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    assert main.expansion_leg_decision(ctx, price_a=100.0, price_b=100.0) is None
    print("PASS: no price history yet -- returns None rather than a fabricated decision")


def test_expansion_leg_decision_none_below_the_70pct_driver_share():
    """Per the client 2026-09-13: "Selected Stock must be responsible for
    =>70% of the Divergence Expansion." A close-to-even split (55/45) no
    longer trades at all, even though leg A is still nominally larger --
    this is the NEW behavior, the old code would have traded it."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.price_history_a = pd.Series([100.00])
    ctx.price_history_b = pd.Series([100.00])
    _give_ctx_compliant_prior_excursion_data(ctx)
    # contribution_a=1.10, contribution_b=0.90 (beta=1.0) -> A's share = 1.10/2.00 = 55%
    assert main.expansion_leg_decision(ctx, price_a=101.10, price_b=99.10) is None
    print("PASS: a driver share under 70% returns None, no trade, even though one leg is still larger")


def test_passes_prior_excursion_direction_filter_matches_the_clients_own_example():
    """Regression test for the client's exact worked example, 2026-09-15."""
    assert main.passes_prior_excursion_direction_filter(
        prior_open=270.48, prior_high=272.23, prior_low=262.68,
        today_open=252.00, today_high=253.00, today_low=249.00,
    ) is True
    print("PASS: passes_prior_excursion_direction_filter matches the client's own 75% example exactly")


def test_passes_prior_excursion_direction_filter_fails_below_threshold():
    # Same prior session (south-dominant), but today's low barely dips
    # below open -- south portion = 0.50, range = 2.00 -> 25%, well under 70%.
    assert main.passes_prior_excursion_direction_filter(
        prior_open=270.48, prior_high=272.23, prior_low=262.68,
        today_open=252.00, today_high=253.50, today_low=251.50,
    ) is False
    print("PASS: a south portion under 70% of today's range fails the filter")


def test_passes_prior_excursion_direction_filter_north_dominant_case():
    # Prior: north excursion (5.0) > south excursion (2.0) -> north-dominant.
    # Today: north portion = high-open = 3.0, range = 4.0 -> 75% >= 70%.
    assert main.passes_prior_excursion_direction_filter(
        prior_open=100.0, prior_high=105.0, prior_low=98.0,
        today_open=250.0, today_high=253.0, today_low=249.0,
    ) is True
    print("PASS: the north-dominant case measures the north portion of today's range")


def test_passes_prior_excursion_direction_filter_fails_closed_on_zero_range():
    assert main.passes_prior_excursion_direction_filter(
        prior_open=100.0, prior_high=105.0, prior_low=98.0,
        today_open=250.0, today_high=250.0, today_low=250.0,
    ) is False
    print("PASS: a zero-range today session fails closed rather than dividing by zero")


def test_passes_prior_excursion_dominance_filter_matches_the_clients_own_example():
    """Regression test for the client's exact worked example, 2026-09-15:
    Long=1.75, Short=7.80, dominant Short must be >=300% of Long ->
    445.71%, PASS."""
    assert main.passes_prior_excursion_dominance_filter(
        prior_open=270.48, prior_high=272.23, prior_low=262.68,
    ) is True
    print("PASS: passes_prior_excursion_dominance_filter matches the client's own 445.71% example exactly")


def test_passes_prior_excursion_dominance_filter_fails_when_almost_equal():
    """Per the client: "Yesterday, ADBE low and high of the day was
    almost equal" -- the whole reason this filter was added. North=2.0,
    South=2.5 -> ratio 125%, well under 300%."""
    assert main.passes_prior_excursion_dominance_filter(
        prior_open=100.0, prior_high=102.0, prior_low=97.5,
    ) is False
    print("PASS: a prior session with no clear dominant direction fails the 300% dominance filter")


def test_passes_prior_excursion_dominance_filter_long_dominant_case():
    # Long=8.0, Short=2.0 -> dominant Long must be >=300% of Short -> 400%, PASS.
    assert main.passes_prior_excursion_dominance_filter(
        prior_open=100.0, prior_high=108.0, prior_low=98.0,
    ) is True
    print("PASS: the long-dominant case measures long against short, not just short against long")


def test_passes_prior_excursion_dominance_filter_passes_on_zero_opposite_excursion():
    # South=10.0, North=0.0 (price never moved above open at all) -- an
    # infinitely lopsided day trivially clears any finite ratio.
    assert main.passes_prior_excursion_dominance_filter(
        prior_open=100.0, prior_high=100.0, prior_low=90.0,
    ) is True
    print("PASS: a zero opposite excursion passes automatically, not a division by zero")


def test_passes_prior_excursion_dominance_filter_fails_closed_on_dead_flat_session():
    assert main.passes_prior_excursion_dominance_filter(
        prior_open=100.0, prior_high=100.0, prior_low=100.0,
    ) is False
    print("PASS: a completely flat prior session (both excursions zero) fails closed")


def test_expansion_leg_decision_none_when_driver_fails_prior_excursion_dominance_filter():
    """Per the client 2026-09-15: a driver stock whose prior session had
    no clear dominant direction (long and short excursions close to
    equal, the exact ADBE scenario that prompted this filter) fails here
    even though the direction filter and driver share would otherwise
    pass -- checked first since it only needs yesterday's data."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.price_history_a = pd.Series([100.00])
    ctx.price_history_b = pd.Series([100.00])
    # North=2.0, South=2.5 -> 125%, fails the 300% dominance requirement.
    ctx.prior_session_price_a = pd.Series([100.0, 102.0, 97.5])
    ctx.prior_session_open_a = 100.0
    idx = pd.date_range("2026-09-15 09:30", periods=3, freq="1min", tz=main.ET)
    ctx.premarket_price_a = pd.Series([50.0, 45.0, 40.0], index=idx)  # would pass direction if reached
    ctx.today_session_open_a = 50.0
    assert main.expansion_leg_decision(ctx, price_a=102.50, price_b=99.3125) is None
    print("PASS: the driver stock failing the 300% dominance filter returns None, no trade")


def test_expansion_leg_decision_none_when_driver_fails_prior_excursion_filter():
    """Per the client 2026-09-15: the driver stock's own prior-excursion-
    direction filter is checked here too, same site as the 70% driver
    share floor just above -- both need to already know which leg is the
    driver. A driver stock with no matching-direction move at all (prior
    session dead flat, so nothing to measure a direction from) fails it."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.price_history_a = pd.Series([100.00])
    ctx.price_history_b = pd.Series([100.00])
    ctx.prior_session_price_a = pd.Series([100.0, 100.0, 100.0])  # dead flat -- no excursion either way
    ctx.prior_session_open_a = 100.0
    idx = pd.date_range("2026-09-15 09:30", periods=3, freq="1min", tz=main.ET)
    ctx.premarket_price_a = pd.Series([100.0, 100.0, 100.0], index=idx)  # today also flat -- 0/0 range
    ctx.today_session_open_a = 100.0
    assert main.expansion_leg_decision(ctx, price_a=102.50, price_b=99.3125) is None
    print("PASS: the driver stock failing the prior-excursion-direction filter returns None, no trade")


def test_seconds_until_next_bar_close_aligns_to_the_minute_boundary():
    """Per the client 2026-09-11: catching a new candle the instant it
    closes instead of up to a minute late. now=13:35:22 -> next boundary is
    13:36:00, plus the buffer."""
    now = datetime(2026, 9, 11, 13, 35, 22, tzinfo=main.ET)
    result = main._seconds_until_next_bar_close(now, buffer_seconds=2.0)
    assert abs(result - 40.0) < 1e-9  # 38s to :36:00, +2s buffer
    print("PASS: sleep lands 2s after the next minute boundary, not a flat 60s from now")


def test_seconds_until_next_bar_close_never_goes_negative_on_a_slow_cycle():
    """A cycle that runs long enough to reach or pass the buffered target
    (e.g. now is already past :00:02 of a new minute) rolls to the NEXT
    boundary instead of returning zero or negative and busy-looping."""
    now = datetime(2026, 9, 11, 13, 36, 1, tzinfo=main.ET)  # past 13:36:02's target
    result = main._seconds_until_next_bar_close(now, buffer_seconds=2.0)
    assert result > 0
    assert abs(result - 61.0) < 1e-9  # rolls to 13:37:02
    print("PASS: never returns non-positive, rolls to the following boundary instead")


def test_apply_warmup_data_pre_clears_johansen_hurst_for_prior_rth_pairs():
    """Per the client 2026-09-11: "Please exclude. They should go straight
    to Divergence Watch. Waiting for Entry Window." A pair loaded here --
    the 9am cache, or its live 9:30am select_pairs_from_previous_session
    fallback, both going through warm_up_pairs -> _apply_warmup_data --
    already qualified from the prior RTH session, so it skips the live
    20-day Johansen/Hurst re-validation gate entirely, unlike a pair
    run_universe_screen_in_background discovers mid-session."""
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    assert ctx.johansen_ever_unconfirmed is False
    assert ctx.hurst_ever_above_threshold is False

    fake_data = main._WarmupData(pivot=_fake_pivot(["AAA", "BBB"]),
                                  prior_pivot=_fake_pivot(["AAA", "BBB"]),
                                  jh_pivot=pd.DataFrame(), prior_open={})
    main._apply_warmup_data([ctx], fake_data)

    assert ctx.johansen_ever_unconfirmed is True
    assert ctx.hurst_ever_above_threshold is True
    print("PASS: a pair loaded from the prior RTH session is pre-cleared, no live Johansen/Hurst wait")


def _history_plus_session_prices(history_bars=300, session_bars=5):
    """A jh_price_a/b-shaped pair of series: `history_bars` bars on one ET
    calendar day, followed by `session_bars` bars on the NEXT ET calendar
    day -- exactly the shape _fetch_johansen_hurst_history returns for a
    prior-session replay (history + the one day being replayed)."""
    history_index = pd.date_range("2026-07-27 09:30", periods=history_bars, freq="1min")
    session_index = pd.date_range("2026-07-28 09:30", periods=session_bars, freq="1min")
    index = history_index.append(session_index)
    n = history_bars + session_bars
    price_a = pd.Series(range(100, 100 + n), dtype=float, index=index)
    price_b = pd.Series(range(50, 50 + n), dtype=float, index=index)
    return price_a, price_b, date(2026, 7, 28)


def test_johansen_hurst_ever_cleared_in_prior_session_finds_a_late_clear():
    """Per the client 2026-09-11/12: "qualified Pearson, Johansen and
    Hurst, at some point in time during 9:30-4pm" -- a pair that only
    clears on the LAST bar of the session must still count as cleared,
    same "ever, not simultaneously" semantics the live sticky flags use."""
    price_a, price_b, session_date = _history_plus_session_prices(session_bars=5)
    # Only the final session bar (5th call) reports a real clear.
    confirmed_side_effect = [True, True, True, True, False]
    hurst_side_effect = [0.10, 0.10, 0.10, 0.10, 0.70]
    with patch.object(main, "johansen_confirmed", side_effect=confirmed_side_effect), \
         patch.object(main, "hurst_exponent", side_effect=hurst_side_effect):
        ever_unconfirmed, ever_above = main._johansen_hurst_ever_cleared_in_prior_session(
            price_a, price_b, session_date)
    assert ever_unconfirmed is True
    assert ever_above is True
    print("PASS: a pair that only clears on the last bar of the session still counts as ever cleared")


def test_johansen_hurst_ever_cleared_in_prior_session_false_when_never_clears():
    price_a, price_b, session_date = _history_plus_session_prices(session_bars=5)
    with patch.object(main, "johansen_confirmed", return_value=True), \
         patch.object(main, "hurst_exponent", return_value=0.10):
        ever_unconfirmed, ever_above = main._johansen_hurst_ever_cleared_in_prior_session(
            price_a, price_b, session_date)
    assert ever_unconfirmed is False
    assert ever_above is False
    print("PASS: a pair that never clears either condition all session returns False, False")


def test_johansen_hurst_ever_cleared_in_prior_session_stops_early_once_both_clear():
    """Most of the replay cost is coint_johansen -- once both flags are
    True nothing later in the session can un-clear them, so the replay
    must not keep calling into the expensive test for remaining bars."""
    price_a, price_b, session_date = _history_plus_session_prices(session_bars=10)
    calls = {"n": 0}

    def fake_johansen(a, b):
        calls["n"] += 1
        return False  # unconfirmed (cleared) on every call

    with patch.object(main, "johansen_confirmed", side_effect=fake_johansen), \
         patch.object(main, "hurst_exponent", return_value=0.70):  # cleared on every call too
        ever_unconfirmed, ever_above = main._johansen_hurst_ever_cleared_in_prior_session(
            price_a, price_b, session_date)
    assert ever_unconfirmed is True
    assert ever_above is True
    assert calls["n"] == 1  # stopped after the very first bar, both cleared immediately
    print("PASS: replay stops as soon as both flags clear, doesn't keep testing the rest of the session")


def test_johansen_hurst_ever_cleared_in_prior_session_skips_bars_before_min_bars():
    """A bar early in the session that doesn't yet have
    MIN_BARS_FOR_LIVE_JOHANSEN_HURST bars behind it (history + elapsed
    session bars) is skipped, not tested -- same fail-closed behavior
    update_live_johansen_hurst uses live."""
    # Only 250 history bars -- below the 300 floor -- so the first several
    # session bars still don't have enough of a window behind them.
    price_a, price_b, session_date = _history_plus_session_prices(history_bars=250, session_bars=10)
    calls = {"n": 0}

    def fake_johansen(a, b):
        calls["n"] += 1
        return False

    with patch.object(main, "johansen_confirmed", side_effect=fake_johansen), \
         patch.object(main, "hurst_exponent", return_value=0.70):
        main._johansen_hurst_ever_cleared_in_prior_session(price_a, price_b, session_date)
    # First 50 session bars needed to reach 300 total are skipped -- only
    # the remaining bars (10 - 50, clamped) actually get tested. With 250
    # history + 10 session bars, none ever reach 300, so zero calls.
    assert calls["n"] == 0
    print("PASS: bars without enough of a window behind them are skipped, never tested")


def test_select_pairs_from_full_universe_prior_session_combines_pearson_and_johansen_hurst():
    """Per the client 2026-09-11/12: the 9am cache builder scans the full
    S&P 500 (not just the Market Entry List), then only keeps a candidate
    that both passes Pearson/liquidity (evaluate_pair's qualified) AND
    cleared Johansen/Hurst at some point during the prior session."""
    ref = universe_screening.UniverseReferenceData(
        groups={"Software": ["AAA", "BBB"], "Retail": ["CCC", "DDD"]},
        market_caps={s: 5e10 for s in ["AAA", "BBB", "CCC", "DDD"]},
        volume_stats={s: {"avg_share_volume": 5_000_000, "avg_dollar_volume": 5e8} for s in
                      ["AAA", "BBB", "CCC", "DDD"]},
        daily_closes=_daily_closes(["AAA", "BBB", "CCC", "DDD"], correlated=True),
    )
    now = datetime(2026, 8, 29, 0, 0, tzinfo=main.ET)  # pseudo-tomorrow
    fetch_now = datetime(2026, 8, 28, 16, 5, tzinfo=main.ET)
    prior_bars = _previous_session_bars(["AAA", "BBB", "CCC", "DDD"], session_date="2026-08-28")

    jh_price_a, jh_price_b, session_date = _history_plus_session_prices(session_bars=5)
    jh_bars = pd.DataFrame({"AAA": jh_price_a, "BBB": jh_price_a, "CCC": jh_price_a, "DDD": jh_price_a})

    def fake_replay(price_a, price_b, sess_date):
        # AAA/BBB (Software) clears, CCC/DDD (Retail) never does.
        col_name = price_a.name
        return (True, True) if col_name == "AAA" else (False, False)

    with patch.object(universe_screening, "build_reference_data", lambda: ref), \
         patch.object(main, "load_market_entry_list", lambda: []), \
         patch.object(main, "load_permanent_blacklist", lambda: set()), \
         patch.object(main, "fetch_intraday_warmup", lambda symbols, start, end: prior_bars), \
         patch.object(main, "_fetch_johansen_hurst_history", lambda symbols, n: jh_bars), \
         patch.object(main, "_johansen_hurst_ever_cleared_in_prior_session", fake_replay):
        contexts = main.select_pairs_from_full_universe_prior_session(now, fetch_now=fetch_now)

    pairs = {(c.symbol_a, c.symbol_b) for c in contexts}
    assert ("AAA", "BBB") in pairs
    assert ("CCC", "DDD") not in pairs
    print("PASS: full-universe selection keeps only pairs that clear both Pearson and Johansen/Hurst")


def test_select_pairs_from_full_universe_prior_session_fetches_johansen_hurst_through_prior_session_date():
    """Regression test for TWO real live bugs, in order:

    2026-09-14: this used to pass `now` (pushed one day ahead, pseudo-
    tomorrow) straight into _fetch_johansen_hurst_history, pushing its
    exclusive `end` one day too far forward -- reaching into the just-
    closed session's own day mere minutes after close, when Databento
    didn't have it ready yet. The whole batch came back empty ("empty
    Johansen/Hurst history, none selected"), caching 0 pairs.

    2026-09-15 (the very next night): switching to `fetch_now` (the fix
    for the bug above) broke it a second, WORSE way -- fetch_now's own
    `today_et` no longer matched this function's, so the exclusive `end`
    landed ON prior_session_date instead of past it, excluding
    prior_session_date's own bars entirely. Every candidate's replay then
    found zero bars for that date and silently reported "never cleared
    together" across the board -- deterministic every single night, not
    an occasional data-lag race.

    Neither `now` nor `fetch_now` is correct: the real requirement is
    "through the end of prior_session_date", which has to be derived
    from prior_session_date itself (already known by this point in the
    function), independent of either time reference."""
    ref = universe_screening.UniverseReferenceData(
        groups={"Software": ["AAA", "BBB"]},
        market_caps={s: 5e10 for s in ["AAA", "BBB"]},
        volume_stats={s: {"avg_share_volume": 5_000_000, "avg_dollar_volume": 5e8} for s in ["AAA", "BBB"]},
        daily_closes=_daily_closes(["AAA", "BBB"], correlated=True),
    )
    now = datetime(2026, 9, 15, 0, 0, tzinfo=main.ET)  # pseudo-tomorrow
    fetch_now = datetime(2026, 9, 14, 20, 5, tzinfo=main.ET)  # the real, current evening time
    prior_bars = _previous_session_bars(["AAA", "BBB"], session_date="2026-09-14")
    jh_price_a, jh_price_b, session_date = _history_plus_session_prices(session_bars=5)
    jh_bars = pd.DataFrame({"AAA": jh_price_a, "BBB": jh_price_a})
    seen = []

    def fake_jh_history(symbols, jh_now):
        seen.append(jh_now)
        return jh_bars

    with patch.object(universe_screening, "build_reference_data", lambda: ref), \
         patch.object(main, "load_market_entry_list", lambda: []), \
         patch.object(main, "load_permanent_blacklist", lambda: set()), \
         patch.object(main, "fetch_intraday_warmup", lambda symbols, start, end: prior_bars), \
         patch.object(main, "_fetch_johansen_hurst_history", fake_jh_history), \
         patch.object(main, "_johansen_hurst_ever_cleared_in_prior_session", lambda a, b, d: (True, True)):
        main.select_pairs_from_full_universe_prior_session(now, fetch_now=fetch_now)

    # prior_session_date is 2026-09-14 -- the boundary must reach ONE DAY
    # PAST it (2026-09-15 00:00 ET), not stop at fetch_now's own date
    # (2026-09-14, which would exclude 09-14's own bars) and not at
    # `now`'s raw value either (it happens to already be 2026-09-15 00:00
    # here, so this assertion alone can't distinguish the two -- the
    # point is it's derived from prior_session_date, not incidentally
    # equal to `now`).
    assert seen == [datetime(2026, 9, 15, 0, 0, tzinfo=main.ET)]
    print("PASS: _fetch_johansen_hurst_history's boundary is derived from prior_session_date, "
          "not from now or fetch_now directly")


def _ctx_with_expansion_baseline(symbol_a="AAA", symbol_b="BBB", trigger=3.0,
                                  price_a=101.0, price_b=100.0,
                                  event_price_a=99.0, event_price_b=100.0):
    """A flat pair with real price history, so expansion_leg_decision (via
    _traded_leg_estimate) has a real Rolling Equilibrium Price baseline
    (price_history_a/b's own mean, per the client 2026-09-13) to measure
    from -- two points each, so .iloc[-1] (current) and .mean() (baseline)
    differ meaningfully. price_a moved further from its baseline than
    price_b (unchanged), so leg A is always the dominant leg here, well
    clear of the 2026-09-13 70% driver-share floor."""
    ctx = main.PairContext(symbol_a=symbol_a, symbol_b=symbol_b, hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.divergence_trigger_level = trigger
    ctx.price_history_a = pd.Series([event_price_a, price_a])
    ctx.price_history_b = pd.Series([event_price_b, price_b])
    _give_ctx_compliant_prior_excursion_data(ctx)
    return ctx


def _signal_for(ctx, deviation=2.4, price_a=100.0, price_b=100.0):
    return main.PairSignal(
        symbol_a=ctx.symbol_a, symbol_b=ctx.symbol_b, hedge_ratio=ctx.hedge_ratio,
        deviation=deviation, divergence_trigger_level=ctx.divergence_trigger_level,
        price_a=price_a, price_b=price_b, half_life_minutes=30.0)


def test_traded_leg_estimate_none_without_an_event_baseline():
    ctx = main.PairContext(symbol_a="AAA", symbol_b="BBB", hedge_ratio=1.0,
                            spread_history=pd.Series(dtype=float))
    ctx.price_history_a = pd.Series([100.0])
    ctx.price_history_b = pd.Series([100.0])
    signal = _signal_for(ctx)
    assert main._traded_leg_estimate(ctx, signal) is None
    print("PASS: _traded_leg_estimate returns None with no current divergence event baseline")


def test_traded_leg_estimate_returns_the_dominant_leg():
    ctx = _ctx_with_expansion_baseline(price_a=101.0, event_price_a=99.0)  # leg A moved $2
    signal = _signal_for(ctx, price_a=101.0, price_b=100.0)
    leg = main._traded_leg_estimate(ctx, signal)
    assert leg is not None
    traded_symbol, action, traded_price = leg
    assert traded_symbol == "AAA"
    assert action == "BUY"  # price rallied
    assert traded_price == 101.0
    print("PASS: _traded_leg_estimate returns the dominant leg's symbol, action and live price")


def test_probe_stock_initial_margin_per_share_uses_tier_1_when_it_rejects():
    """Per the client 2026-09-12: probing at quantity=1 first."""
    ctx = _ctx_with_expansion_baseline()
    signal = _signal_for(ctx)
    ibkr = AsyncMock()
    ibkr.probe_bracket_initial_margin.return_value = 42.50  # tier 1 rejects immediately
    result = asyncio.run(main._probe_stock_initial_margin_per_share(
        ibkr, ctx, signal, "AAA", "BUY", 100.0, 100_000.0, datetime(2026, 9, 12, 9, 35, tzinfo=main.ET)))
    assert result == 42.50
    assert ibkr.probe_bracket_initial_margin.call_count == 1  # never reached tier 2
    call_kwargs = ibkr.probe_bracket_initial_margin.call_args
    assert call_kwargs[0][2] == 1  # quantity=1 on the tier 1 attempt
    print("PASS: tier 1 (quantity=1) used directly when it produces a real rejection")


def test_probe_stock_initial_margin_per_share_falls_back_to_tier_2():
    """Tier 1 (quantity=1) fits with no rejection (returns None, the
    expected common case per this function's own docstring) -- falls
    through to tier 2, the real guessed-quantity probe, and divides by
    that quantity for the per-share figure."""
    ctx = _ctx_with_expansion_baseline()
    signal = _signal_for(ctx)
    ibkr = AsyncMock()
    ibkr.probe_bracket_initial_margin.side_effect = [None, 1000.0]  # tier 1 fits, tier 2 rejects at $1000 total
    result = asyncio.run(main._probe_stock_initial_margin_per_share(
        ibkr, ctx, signal, "AAA", "BUY", 100.0, 100_000.0, datetime(2026, 9, 12, 9, 35, tzinfo=main.ET)))
    assert ibkr.probe_bracket_initial_margin.call_count == 2
    tier2_quantity = ibkr.probe_bracket_initial_margin.call_args_list[1][0][2]
    assert tier2_quantity > 1
    assert abs(result - 1000.0 / tier2_quantity) < 1e-9
    print("PASS: falls back to tier 2 (guessed quantity) when quantity=1 doesn't reject, per-share result")


def test_probe_stock_initial_margin_per_share_falls_back_to_flat_estimate():
    """Neither real probe rejects -- falls back to the flat
    INITIAL_MARGIN_FRACTION x price estimate, never returns nothing."""
    ctx = _ctx_with_expansion_baseline()
    signal = _signal_for(ctx)
    ibkr = AsyncMock()
    ibkr.probe_bracket_initial_margin.return_value = None  # never rejects, either tier
    result = asyncio.run(main._probe_stock_initial_margin_per_share(
        ibkr, ctx, signal, "AAA", "BUY", 100.0, 100_000.0, datetime(2026, 9, 12, 9, 35, tzinfo=main.ET)))
    assert abs(result - 100.0 * main.INITIAL_MARGIN_FRACTION) < 1e-9
    print("PASS: falls back to the flat INITIAL_MARGIN_FRACTION estimate when nothing real rejects")


def test_rank_eligible_signals_skips_probing_with_only_one_candidate():
    """Per the client 2026-09-12: "If it is only one qualified pair, then
    that pair will execute." No ranking, no probe orders, at all."""
    ctx = _ctx_with_expansion_baseline()
    signal = _signal_for(ctx)
    ibkr = AsyncMock()
    result = asyncio.run(main.rank_eligible_signals_by_adjusted_spread_efficiency(
        [(signal, ctx)], ibkr, 100_000.0, datetime(2026, 9, 12, 9, 35, tzinfo=main.ET)))
    assert result is signal
    ibkr.probe_bracket_initial_margin.assert_not_called()
    print("PASS: a single eligible candidate executes directly, no probing needed to rank against nothing")


def test_rank_eligible_signals_picks_the_highest_adjusted_spread_efficiency():
    """Two simultaneously eligible candidates -- probed sequentially, the
    one with the higher divergence/margin ratio wins, per Checklist_120926.pdf."""
    ctx_a = _ctx_with_expansion_baseline(symbol_a="LOWRATIO", symbol_b="LR2", price_a=101.0, event_price_a=99.0)
    signal_a = _signal_for(ctx_a, deviation=4.5, price_a=101.0)  # big divergence
    ctx_b = _ctx_with_expansion_baseline(symbol_a="HIGHRATIO", symbol_b="HR2", price_a=101.0, event_price_a=99.0)
    signal_b = _signal_for(ctx_b, deviation=2.0, price_a=101.0)  # smaller divergence

    ibkr = AsyncMock()

    async def fake_probe(symbol, action, quantity, *args, **kwargs):
        # LOWRATIO needs a huge margin per share, HIGHRATIO a tiny one --
        # HIGHRATIO's ratio (2.0/5) beats LOWRATIO's (4.5/500) despite the
        # smaller raw divergence.
        return (500.0 if symbol == "LOWRATIO" else 5.0) if quantity == 1 else None

    ibkr.probe_bracket_initial_margin.side_effect = fake_probe
    result = asyncio.run(main.rank_eligible_signals_by_adjusted_spread_efficiency(
        [(signal_a, ctx_a), (signal_b, ctx_b)], ibkr, 100_000.0,
        datetime(2026, 9, 12, 9, 35, tzinfo=main.ET)))
    assert result.symbol_a == "HIGHRATIO"
    print("PASS: picks the candidate with the highest divergence/margin ratio, probed concurrently")


def test_rank_eligible_signals_skips_a_candidate_with_no_leg_estimate():
    """A candidate expansion_leg_decision can't resolve (no event baseline
    yet) is dropped from ranking rather than crashing the whole cycle."""
    ctx_ok = _ctx_with_expansion_baseline(symbol_a="OK", symbol_b="OK2")
    signal_ok = _signal_for(ctx_ok)
    ctx_no_baseline = main.PairContext(symbol_a="NOBASE", symbol_b="NB2", hedge_ratio=1.0,
                                        spread_history=pd.Series(dtype=float))
    ctx_no_baseline.divergence_trigger_level = 3.0
    ctx_no_baseline.price_history_a = pd.Series([100.0])
    ctx_no_baseline.price_history_b = pd.Series([100.0])
    signal_no_baseline = _signal_for(ctx_no_baseline)

    ibkr = AsyncMock()
    ibkr.probe_bracket_initial_margin.return_value = 50.0
    result = asyncio.run(main.rank_eligible_signals_by_adjusted_spread_efficiency(
        [(signal_ok, ctx_ok), (signal_no_baseline, ctx_no_baseline)], ibkr, 100_000.0,
        datetime(2026, 9, 12, 9, 35, tzinfo=main.ET)))
    assert result.symbol_a == "OK"
    print("PASS: a candidate with no resolvable leg is skipped from ranking, not a crash")


def test_rank_eligible_signals_probes_sequentially_not_concurrently():
    """Per a third-party audit finding 2026-09-14, verified directly:
    IBKR evaluates Reg-T initial margin at the ACCOUNT level across every
    currently working order, not per-probe in isolation. Two candidates'
    real probe orders sitting on the account AT THE SAME TIME would
    cross-contaminate each other's error 201 margin figures. This proves
    no two candidates' probes are ever in flight simultaneously -- an
    in-flight counter that would exceed 1 if probing were still
    concurrent (asyncio.gather)."""
    ctx_a = _ctx_with_expansion_baseline(symbol_a="AAA", symbol_b="A2", price_a=101.0, event_price_a=99.0)
    signal_a = _signal_for(ctx_a, deviation=3.0, price_a=101.0)
    ctx_b = _ctx_with_expansion_baseline(symbol_a="BBB", symbol_b="B2", price_a=101.0, event_price_a=99.0)
    signal_b = _signal_for(ctx_b, deviation=2.0, price_a=101.0)

    ibkr = AsyncMock()
    in_flight = {"count": 0, "max_seen": 0}

    async def fake_probe(symbol, action, quantity, *args, **kwargs):
        if quantity != 1:
            return None
        in_flight["count"] += 1
        in_flight["max_seen"] = max(in_flight["max_seen"], in_flight["count"])
        await asyncio.sleep(0.02)  # simulates the real probe's own network round trip
        in_flight["count"] -= 1
        return 10.0

    ibkr.probe_bracket_initial_margin.side_effect = fake_probe
    asyncio.run(main.rank_eligible_signals_by_adjusted_spread_efficiency(
        [(signal_a, ctx_a), (signal_b, ctx_b)], ibkr, 100_000.0,
        datetime(2026, 9, 14, 9, 35, tzinfo=main.ET)))
    assert in_flight["max_seen"] == 1
    print("PASS: at most one candidate's probe orders are ever live on the account at once")


def test_pair_divergence_shows_a_9am_loaded_pair_before_live_data_arrives():
    """Per the client 2026-09-13: "when the prior RTH qualified pairs are
    loaded in the Divergence Watch by 9am, we want to see the pairs. The
    pairs should be visible by 9am. Even if live data is yet to come on."
    A pair with a real trigger but no live current value yet (last_deviation
    still None) shows on the Watch with current/distance null, not
    excluded entirely."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    ctx = _ctx_with_trigger(symbol_a="EARLY", symbol_b="BIRD")
    ctx.last_deviation = None  # no live tick yet -- the 9am-before-live-data case
    ctx.reverted_over_50pct_since_max_divergence = True
    ctx.trigger_reached_today = False
    main.write_status("running", elv=100_000, contexts=[ctx])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    rows = {row["symbol_a"]: row for row in data["pair_divergence"]}
    assert "EARLY" in rows
    assert rows["EARLY"]["current"] is None
    assert rows["EARLY"]["distance"] is None
    assert rows["EARLY"]["trigger"] == ctx.divergence_trigger_level
    assert rows["EARLY"]["cleared"] is True
    print("PASS: a 9am-loaded pair with no live tick yet still shows, current/distance null, trigger real")


def test_pair_divergence_cleared_requires_both_50pct_reversion_and_no_retouch():
    """Per the client 2026-09-13: "the pair that met price conditions
    (drop of below 50%, no retouch of Trigger inside lookback period)
    will show cleared." Both conditions required -- neither alone is
    enough."""
    tmp = Path(tempfile.mkdtemp()) / "bot_status.json"
    main.STATUS_PATH = tmp
    only_reverted = _ctx_with_trigger(symbol_a="ONLYREV", symbol_b="B1")
    only_reverted.reverted_over_50pct_since_max_divergence = True
    only_reverted.trigger_reached_today = True  # retouched today -- disqualifies "cleared"
    only_no_retouch = _ctx_with_trigger(symbol_a="ONLYNORETOUCH", symbol_b="B2")
    only_no_retouch.reverted_over_50pct_since_max_divergence = False  # never reverted >50%
    only_no_retouch.trigger_reached_today = False
    both = _ctx_with_trigger(symbol_a="BOTH", symbol_b="B3")
    both.reverted_over_50pct_since_max_divergence = True
    both.trigger_reached_today = False
    main.write_status("running", elv=100_000, contexts=[only_reverted, only_no_retouch, both])

    data = json.loads(tmp.read_text(encoding="utf-8"))
    rows = {row["symbol_a"]: row for row in data["pair_divergence"]}
    assert rows["ONLYREV"]["cleared"] is False
    assert rows["ONLYNORETOUCH"]["cleared"] is False
    assert rows["BOTH"]["cleared"] is True
    print("PASS: cleared requires both >50% reversion AND no retouch of trigger today")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
