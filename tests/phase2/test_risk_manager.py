"""Unit tests for risk_manager.py's entry/exit state machine.

Entry/exit logic was replaced wholesale 2026-08-18, per the client: "I want
to change the entry... From The Divergence Watch List, select the closest
to trigger and enter. Entry minimum points to trigger <= $0.40 to go
[same-day clarified 2026-08-19 as a band: 'Entry Window: Pair must be
within $0.01 to $0.40 points away from trigger... There is a minimum and a
maximum']. Take Profit at Trigger. Stop Loss at 2% of EWLV [replaced
2026-08-19 with 'Stop Loss = 0 Divergence', then clarified same day as '0
relative to equilibrium at entry. 100% mean-reversion']. Bot is taking
profit on divergence expansion. One trade per day. No re-entry (confirmed
explicitly: no same-day re-entry regardless of whether the trade closes by
Take Profit, Stop Loss, or the 140-minute Maximum Hold Time)." Take Profit
and Stop Loss are both real OCA bracket orders at IBKR again as of
2026-08-19 (see test_bracket_order_prices_* below) since both are now
prices fixed at entry -- check_exit itself only covers what can't be a
resting order (EOD, the max hold time); reconciling native
bracket fills is main.py's job, not tested here. The opposite of the
mean-reversion logic these tests used to cover (Turnback, Dynamic Stop
Loss, Half-Life Stop Loss, same-pair-lock re-entry). No I/O, all inputs
are synthetic."""
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.risk_manager import (
    ENTRY_TRIGGER_DISTANCE_MAX_PCT, ENTRY_TRIGGER_DISTANCE_MIN_PCT,
    DailyState, PairSignal, Position,
    bracket_order_prices, check_exit, entry_eligible,
    is_early_close_day, load_daily_state, news_stop_cancel, open_position, open_single_leg_position, position_hedge_ratio, record_exit,
    reentry_allowed, resting_hard_stop_price, resting_take_profit_price, save_daily_state,
    save_pair_trigger_state, load_pair_trigger_state,
    select_closest_to_trigger, select_highest_adjusted_spread_efficiency,
    stop_loss_double_touch_exit, stop_loss_triple_touch_exit,
    pnl_stop_loss_price, take_profit_hard_stop_exit, trade_direction, unrealized_pnl, unrealized_pnl_pct,
    wide_hard_stop_price,
)

RTH_MORNING = datetime(2026, 7, 27, 10, 0)
RTH_LATE = datetime(2026, 7, 27, 15, 45)  # well after the 10:30am latest-entry cutoff
RTH_EOD = datetime(2026, 7, 27, 15, 59)


def _signal(deviation=0.5, mean_divergence=0.9, symbol_a="TYL", symbol_b="WDAY",
            price_a=100.0, price_b=100.0, half_life_minutes=30.0, trigger_reached_today=False,
            cointegration_broke_before_931=True, cointegration_still_broken=True,
            gap_ok=True, johansen_hurst_confirmed=True, mean_reverted_since_max_divergence=False,
            reverted_over_50pct_since_max_divergence=True):
    return PairSignal(symbol_a=symbol_a, symbol_b=symbol_b, hedge_ratio=2.0,
                       deviation=deviation, divergence_trigger_level=mean_divergence,
                       price_a=price_a, price_b=price_b, half_life_minutes=half_life_minutes,
                       trigger_reached_today=trigger_reached_today,
                       cointegration_broke_before_931=cointegration_broke_before_931,
                       cointegration_still_broken=cointegration_still_broken,
                       gap_ok=gap_ok,
                       johansen_hurst_confirmed=johansen_hurst_confirmed,
                       mean_reverted_since_max_divergence=mean_reverted_since_max_divergence,
                       reverted_over_50pct_since_max_divergence=reverted_over_50pct_since_max_divergence)


def test_pair_signal_distance_to_trigger():
    signal = _signal(deviation=0.6, mean_divergence=0.9)
    assert abs(signal.distance_to_trigger - 0.3) < 1e-9
    print("PASS: distance_to_trigger = trigger - abs(deviation)")


def test_pair_signal_distance_to_trigger_negative_once_crossed():
    signal = _signal(deviation=1.2, mean_divergence=0.9)
    assert signal.distance_to_trigger < 0
    print("PASS: distance_to_trigger goes negative once the divergence has already crossed the trigger")


def test_entry_eligible_happy_path():
    # deviation=2.4, trigger=2.9 -> distance=0.5, comfortably inside the
    # 13%-25% window ([0.377, 0.725] for this trigger level)
    signal = _signal(deviation=2.4, mean_divergence=2.9)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry eligible when within the 13%-25% entry window, above $2 minimum, not blacklisted, before 10:30am")


def test_entry_blocked_when_daily_trading_halted():
    signal = _signal(deviation=2.5, mean_divergence=2.9)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState(trading_halted=True)) is False
    print("PASS: entry blocked when the daily loss kill-switch has tripped")


def test_entry_blocked_at_max_trades_per_day():
    # Per the client 2026-08-19: "Maximum Trades per day = 2".
    signal = _signal(deviation=2.5, mean_divergence=2.9)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState(trades_taken_today=2)) is False
    print("PASS: entry blocked once the day's 2-trade maximum has been reached")


def test_entry_eligible_below_max_trades_per_day():
    signal = _signal(deviation=2.4, mean_divergence=2.9)  # distance = 0.5
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState(trades_taken_today=1)) is True
    print("PASS: entry still eligible after 1 of 2 allowed trades")


def test_entry_blocked_when_reentry_locked_out():
    # Per the client 2026-08-19: any close other than Take Profit before
    # the re-entry cutoff ends the day even if the 2-trade cap hasn't been
    # reached.
    signal = _signal(deviation=2.5, mean_divergence=2.9)
    assert entry_eligible(
        signal, set(), RTH_MORNING, DailyState(trades_taken_today=1, reentry_locked_out=True)) is False
    print("PASS: entry blocked when re-entry has been locked out, even with a trade slot remaining")


def test_entry_blocked_after_latest_entry_time():
    signal = _signal(deviation=2.5, mean_divergence=2.9)
    assert entry_eligible(signal, set(), RTH_LATE, DailyState()) is False
    print("PASS: entry blocked well after the 10:30am ET latest-entry cutoff")


def test_entry_eligible_at_exact_1030_latest_entry_boundary():
    # Narrowed 2026-08-20, per the client: "Instead of 3:30pm for latest
    # entry, use 11:00am ET" then corrected same day to "use 10:30am ET".
    signal = _signal(deviation=2.4, mean_divergence=2.9)  # distance = 0.5
    ten_thirty = datetime(2026, 7, 27, 10, 30)
    assert entry_eligible(signal, set(), ten_thirty, DailyState()) is True
    print("PASS: entry eligible exactly at the 10:30am ET latest-entry boundary, not just before it")


def test_entry_blocked_just_after_1030_latest_entry_boundary():
    signal = _signal(deviation=2.5, mean_divergence=2.9)
    just_after = datetime(2026, 7, 27, 10, 31)
    assert entry_eligible(signal, set(), just_after, DailyState()) is False
    print("PASS: entry blocked one minute past the 10:30am ET latest-entry boundary")


def test_entry_blocked_before_932am_floor():
    # v7 added a 10am floor, v8 walked it back as a typo, then the client
    # reinstated it 2026-07-27, reverted to 9:30am 2026-08-27 ("Market
    # Entry: 9:30am, instead of 10am."), moved to 9:31am 2026-09-12
    # (Checklist_120926.pdf: "Market Entry : 9:31-10:30am ET"), then to
    # 9:32am 2026-09-13 ("Market Entry: 9:32-10:30am ET. 9:32, not 9:31.").
    signal = _signal(deviation=2.5, mean_divergence=2.9)
    before_open = datetime(2026, 7, 27, 9, 31)
    assert entry_eligible(signal, set(), before_open, DailyState()) is False
    print("PASS: entry blocked before the 9:32am ET floor")


def test_entry_eligible_at_exact_932am_floor():
    signal = _signal(deviation=2.4, mean_divergence=2.9)  # distance = 0.5, inside the band
    market_open = datetime(2026, 7, 27, 9, 32)
    assert entry_eligible(signal, set(), market_open, DailyState()) is True
    print("PASS: entry eligible exactly at the 9:32am ET floor, not just after it")


def test_entry_blocked_when_either_leg_permanently_blacklisted():
    # Per the client 2026-08-26: entry_eligible's blacklist argument is the
    # PERMANENT blacklist only, not news exclusion -- "If a symbol is in
    # the news, it should get in... We want expansion." News only matters
    # as an exit (News Stop, main.py's run_check_cycle), entry_eligible
    # never sees news data at all anymore.
    signal = _signal(deviation=2.5, mean_divergence=2.9)
    assert entry_eligible(signal, {"WDAY"}, RTH_MORNING, DailyState()) is False
    assert entry_eligible(signal, {"TYL"}, RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked when either leg is on the permanent blacklist")


def test_entry_blocked_below_minimum_dollar_divergence():
    # trigger technically close (distance=0.1) but the deviation itself is below the $2 floor
    signal = _signal(deviation=0.8, mean_divergence=0.9)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked below the $2 minimum entry divergence even if close to trigger")


def test_entry_blocked_above_maximum_divergence():
    # Raised from $3.5 to $5 per the client 2026-09-04: "Increase Maximum
    # Divergence to 5" (itself lowered from $6 to $3.5 earlier the same day).
    signal = _signal(deviation=5.2, mean_divergence=5.3)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked above the $5 maximum divergence")


def test_entry_eligible_at_exact_maximum_divergence_boundary():
    # trigger=5.75, deviation=5.0 (the exact divergence boundary) ->
    # distance=0.75, inside the 13%-25% window ([0.7475, 1.4375]).
    signal = _signal(deviation=5.0, mean_divergence=5.75)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry eligible exactly at the $5 maximum divergence boundary")


def test_entry_blocked_when_either_leg_above_maximum_stock_price():
    # Per the client 2026-08-17: "Maximum Stock Price of $700 is for a
    # stock. Not a pair." -- checked on EACH leg independently.
    signal_a_expensive = _signal(deviation=2.5, mean_divergence=2.9, price_a=750.0, price_b=100.0)
    signal_b_expensive = _signal(deviation=2.5, mean_divergence=2.9, price_a=100.0, price_b=750.0)
    assert entry_eligible(signal_a_expensive, set(), RTH_MORNING, DailyState()) is False
    assert entry_eligible(signal_b_expensive, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked when either individual leg is above the $700 maximum stock price")


def test_entry_eligible_at_exact_maximum_stock_price_boundary():
    signal = _signal(deviation=2.4, mean_divergence=2.9, price_a=700.0, price_b=700.0)  # distance = 0.5, inside the band
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry eligible exactly at the $700 maximum stock price boundary")


def test_entry_blocked_when_already_past_the_trigger():
    # Per the client 2026-08-18: entry must be BEFORE the trigger is
    # reached, not after -- Take Profit fires exactly at the trigger, so
    # entering past it would mean the profit target is already behind entry.
    signal = _signal(deviation=3.0, mean_divergence=2.9)  # distance = -0.1, already crossed
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked once the divergence has already crossed its own trigger")


def test_entry_blocked_when_trigger_already_reached_earlier_today():
    # Per the client 2026-08-19: "Trigger has not previously been reached
    # at any point during the current RTH session." Otherwise-eligible
    # signal (distance = 0.5, well inside the $0.40-$0.60 window), but the
    # pair already touched its trigger earlier and pulled back.
    signal = _signal(deviation=2.4, mean_divergence=2.9, trigger_reached_today=True)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked when the trigger was already reached earlier in today's RTH session")


def test_entry_eligible_when_trigger_never_reached_today():
    signal = _signal(deviation=2.4, mean_divergence=2.9, trigger_reached_today=False)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry eligible when the trigger has never been reached today")


def test_entry_eligible_regardless_of_cointegration_flags():
    # Per the client 2026-08-24: rolled back to the pre-2026-08-22 "old
    # script" -- the cointegration-broke-before-931/still-broken gate added
    # by the 2026-08-22 expansion-filter update is removed entirely, not
    # just relaxed. An otherwise-eligible signal must qualify no matter what
    # these flags say, since entry_eligible no longer reads them at all.
    for broke_931, still_broken in [(True, True), (True, False), (False, True), (False, False)]:
        signal = _signal(deviation=2.4, mean_divergence=2.9,
                          cointegration_broke_before_931=broke_931,
                          cointegration_still_broken=still_broken)
        assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry_eligible ignores cointegration_broke_before_931/cointegration_still_broken entirely")


def test_entry_blocked_when_johansen_hurst_never_confirmed():
    # Per the client 2026-08-25 (Correct_Filters.pdf): Johansen Unconfirmed
    # and Hurst > 0.60, live and continuous. main.py's
    # update_live_johansen_hurst does the actual threshold checks and sets
    # ctx.johansen_ever_unconfirmed/hurst_ever_above_threshold as two
    # independent sticky flags -- entry_eligible here just reads their AND
    # (see PairSignal.johansen_hurst_confirmed).
    signal = _signal(deviation=2.4, mean_divergence=2.9, johansen_hurst_confirmed=False)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry_eligible blocks when Johansen/Hurst has never cleared this session")


def test_entry_eligible_when_johansen_hurst_confirmed():
    signal = _signal(deviation=2.4, mean_divergence=2.9, johansen_hurst_confirmed=True)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry_eligible passes when Johansen/Hurst has cleared at least once this session")


def test_entry_blocked_when_mean_reverted_since_max_divergence():
    # Per the client 2026-09-07 (Checklist_090926.pdf), entry condition A:
    # "There is no 100% mean reversion after Maximum Divergence." See
    # PairSignal.mean_reverted_since_max_divergence's own comment.
    signal = _signal(deviation=2.4, mean_divergence=2.9, mean_reverted_since_max_divergence=True)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry_eligible blocks once the spread has fully reverted to equilibrium since its own peak")


def test_entry_eligible_when_not_reverted_since_max_divergence():
    signal = _signal(deviation=2.4, mean_divergence=2.9, mean_reverted_since_max_divergence=False)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry_eligible passes when the spread has never reverted since its own peak")


def test_entry_blocked_when_not_reverted_over_50pct_since_max_divergence():
    # Per the client 2026-09-13: "After pair hit the maximum divergence, it
    # should have at least >50% mean reversion before entry." Opposite
    # polarity from Condition A above -- this one blocks when the
    # reversion HASN'T happened.
    signal = _signal(deviation=2.4, mean_divergence=2.9, reverted_over_50pct_since_max_divergence=False)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry_eligible blocks when the spread hasn't reverted more than 50% since its own peak")


def test_entry_eligible_when_reverted_over_50pct_since_max_divergence():
    signal = _signal(deviation=2.4, mean_divergence=2.9, reverted_over_50pct_since_max_divergence=True)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry_eligible passes once the spread has reverted more than 50% since its own peak")


def test_entry_eligible_regardless_of_gap_ok():
    # Per the client 2026-08-25: "Delete: Allowed Divergence Gap" -- no
    # longer a qualification gate, either value must leave an
    # otherwise-eligible signal eligible.
    assert entry_eligible(_signal(deviation=2.4, mean_divergence=2.9, gap_ok=False),
                           set(), RTH_MORNING, DailyState()) is True
    assert entry_eligible(_signal(deviation=2.4, mean_divergence=2.9, gap_ok=True),
                           set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry_eligible ignores gap_ok (Allowed Divergence Gap) entirely")


def test_entry_blocked_when_too_close_to_trigger():
    # Per the client 2026-08-24: "Entry Window: Pair must be within 13%-25%
    # of Trigger points to get to Trigger" -- replaced the fixed-dollar band
    # entirely with a percentage of the trigger level itself. trigger=2.6,
    # so the window is [0.338, 0.65]; distance=0.2 is below the minimum.
    signal = _signal(deviation=2.4, mean_divergence=2.6)  # distance = 0.2
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked when closer than 13% of the trigger level")


def test_entry_blocked_when_too_far_from_trigger():
    # Window widened to 13%-49.9% per the client 2026-09-09. trigger=5.0 ->
    # window [0.65, 2.495]; distance=2.8 is above the maximum.
    signal = _signal(deviation=2.2, mean_divergence=5.0)  # distance = 2.8, too far
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked when more than 49.9% of the trigger level away from it")


def test_entry_eligible_at_exact_13pct_distance_boundary():
    # Derives deviation from the actual constant, bit-for-bit, rather than a
    # hand-typed decimal -- 0.13 * trigger isn't exactly representable in
    # binary floating point, so a hand-typed decimal can land a hair on the
    # wrong side of what the code itself computes as the boundary.
    trigger = 3.0
    deviation = trigger - (ENTRY_TRIGGER_DISTANCE_MIN_PCT * trigger)
    signal = _signal(deviation=deviation, mean_divergence=trigger)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry eligible exactly at the 13% minimum distance-to-trigger boundary")


def test_entry_eligible_at_exact_max_distance_boundary():
    trigger = 5.0
    deviation = trigger - (ENTRY_TRIGGER_DISTANCE_MAX_PCT * trigger)
    signal = _signal(deviation=deviation, mean_divergence=trigger)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is True
    print("PASS: entry eligible exactly at the 49.9% maximum distance-to-trigger boundary")


def test_entry_blocked_when_trigger_level_is_zero_or_negative():
    # Defensive guard -- a real trigger level should never be <= 0, but the
    # percentage math can't bound a real window against one, so this must
    # not silently pass just because 0 <= 0 <= 0.
    signal = _signal(deviation=0.0, mean_divergence=0.0)
    assert entry_eligible(signal, set(), RTH_MORNING, DailyState()) is False
    print("PASS: entry blocked outright when the trigger level itself is zero or negative")


def test_select_closest_to_trigger_picks_smallest_distance():
    # Per the client 2026-08-18: "From The Divergence Watch List, select
    # the closest to trigger and enter."
    far = _signal(deviation=2.0, mean_divergence=2.9, symbol_a="FAR")  # distance 0.9
    close = _signal(deviation=2.7, mean_divergence=2.9, symbol_a="CLOSE")  # distance 0.2
    assert select_closest_to_trigger([far, close]).symbol_a == "CLOSE"
    print("PASS: select_closest_to_trigger picks the signal nearest its own trigger")


def test_select_closest_to_trigger_none_when_no_signals():
    assert select_closest_to_trigger([]) is None
    print("PASS: select_closest_to_trigger returns None with no eligible signals")


def test_select_highest_adjusted_spread_efficiency_picks_the_best_ratio():
    """Per the client 2026-09-12 (Checklist_120926.pdf): "Spread Return %
    = (Divergence $/Stock Initial Margin) x100... ranks all eligible
    pairs in descending order of Spread Return % and executes the
    highest-ranked opportunity." LOW_MARGIN has a smaller absolute
    divergence than HIGH_DIVERGENCE, but a far smaller margin per share,
    so its ratio wins -- confirms this ranks by the RATIO, not raw
    divergence $ alone."""
    high_divergence = _signal(deviation=4.5, mean_divergence=4.9, symbol_a="HIGH_DIVERGENCE")  # abs_deviation=4.5
    low_margin = _signal(deviation=2.0, mean_divergence=2.9, symbol_a="LOW_MARGIN")  # abs_deviation=2.0
    ranked = [(high_divergence, 50.0), (low_margin, 5.0)]  # ratios: 0.09 vs 0.40
    winner = select_highest_adjusted_spread_efficiency(ranked)
    assert winner.symbol_a == "LOW_MARGIN"
    print("PASS: select_highest_adjusted_spread_efficiency picks the highest divergence/margin ratio, "
          "not the highest raw divergence")


def test_select_highest_adjusted_spread_efficiency_none_when_empty():
    assert select_highest_adjusted_spread_efficiency([]) is None
    print("PASS: select_highest_adjusted_spread_efficiency returns None with no ranked candidates")


def test_trade_direction_buys_expensive_leg_when_still_expanding_above_equilibrium():
    # Per the client 2026-08-18's strategy reversal: the OPPOSITE of the old
    # mean-reversion convention -- profits from the divergence EXPANDING
    # further, not reverting.
    assert trade_direction(deviation=0.5) == ("BUY", "SELL")
    print("PASS: spread above equilibrium and expanding -> buy the (already expensive) leg A")


def test_trade_direction_sells_when_still_expanding_below_equilibrium():
    assert trade_direction(deviation=-0.5) == ("SELL", "BUY")
    print("PASS: spread below equilibrium and expanding -> sell leg A, buy leg B")


PRICE_B = 100.0  # leg B held constant across these tests; leg A carries the spread


def _open(deviation=0.5, qty_a=100, qty_b=200, entry_notional=50_000,
          half_life_minutes=30, entry_spread=10.5, now=RTH_MORNING, entry_trigger_level=12.0):
    """Opens a position whose entry PRICES actually produce entry_spread at
    the position's own executed ratio, so P&L is measured off real prices."""
    ratio = qty_b / qty_a
    return open_position(_signal(deviation=deviation), qty_a=qty_a, qty_b=qty_b,
                          entry_notional=entry_notional, half_life_minutes=half_life_minutes,
                          current_spread=entry_spread, now=now,
                          entry_price_a=entry_spread + ratio * PRICE_B, entry_price_b=PRICE_B,
                          entry_trigger_level=entry_trigger_level)


def _at(position, spread):
    """Leg prices that put this position's own fixed-ratio spread at `spread`."""
    return (spread + position_hedge_ratio(position) * PRICE_B, PRICE_B)


def test_check_exit_eod_overrides_everything():
    # Take Profit, Hard Stop, and Timed Exit are all native resting orders
    # at IBKR (as of the 2026-08-24 restructure), so check_exit only covers
    # what can't be a resting order: the 3:59pm EOD flatten.
    position = _open(entry_trigger_level=100.0)
    reason = check_exit(position, now=RTH_EOD)
    assert reason == "eod"
    print("PASS: 3:59pm ET EOD flatten overrides an otherwise-open position")


def test_check_exit_none_when_no_condition_met():
    position = _open(now=RTH_MORNING, entry_trigger_level=100.0)
    reason = check_exit(position, now=RTH_MORNING + timedelta(minutes=5))
    assert reason is None
    print("PASS: position stays open before EOD")


def test_check_exit_does_not_fire_at_the_old_max_hold_time_mark():
    # Maximum Hold Time moved out of check_exit 2026-08-24 -- it's now the
    # Timed Exit bracket order's own goodAfterTime, fired at the broker,
    # not polled here anymore.
    position = _open(now=RTH_MORNING, entry_trigger_level=100.0)
    reason = check_exit(position, now=RTH_MORNING + timedelta(minutes=60))
    assert reason is None
    print("PASS: check_exit no longer fires max_hold_time -- that's the broker-side Timed Exit's job now")


def test_news_stop_cancel_true_when_either_leg_is_blacklisted():
    # Per the client 2026-09-01: "Or cancel selected pair, if not yet a
    # position." -- the pre-entry half of NEWS STOP.
    assert news_stop_cancel("AAA", "BBB", blacklisted={"AAA"})
    assert news_stop_cancel("AAA", "BBB", blacklisted={"BBB"})
    assert news_stop_cancel("AAA", "BBB", blacklisted={"AAA", "BBB"})
    print("PASS: either leg being blacklisted cancels the pair")


def test_news_stop_cancel_false_when_neither_leg_is_blacklisted():
    assert not news_stop_cancel("AAA", "BBB", blacklisted={"CCC"})
    assert not news_stop_cancel("AAA", "BBB", blacklisted=set())
    print("PASS: an unrelated blacklist entry does not cancel the pair")


def test_bracket_order_prices_take_profit_expands_away_from_equilibrium_for_buy():
    # Per the client 2026-08-19: Take Profit is entry_equilibrium plus the
    # trigger level, in the direction a BUY entry was already betting on
    # (spread rising further above equilibrium). Hard Stop is the opposite
    # side, 2x the trigger level past equilibrium (per the client
    # 2026-08-24: "Stop Loss is the opposite of Take Profit", narrowed from
    # x3 to x2 2026-08-25).
    hard_stop_price, take_profit_price = bracket_order_prices(
        action_a="BUY", ratio_a=1, entry_equilibrium=10.0, entry_trigger_level=2.0)
    # take_profit_spread = 10.0 + 2.0 = 12.0; closing price = -ratio_a*spread*sign = -12.0
    assert take_profit_price == -12.0
    # hard_stop_spread = 10.0 - 1*(2.0*2) = 6.0; closing price = -1*6.0*1 = -6.0
    assert round(hard_stop_price, 2) == -6.0
    print("PASS: BUY bracket -- take profit expands away from equilibrium, hard stop sits on the opposite side")


def test_bracket_order_prices_take_profit_expands_away_from_equilibrium_for_sell():
    # Mirror of the BUY case: a SELL entry bets the spread keeps falling
    # further below equilibrium, so both targets land on that side instead.
    hard_stop_price, take_profit_price = bracket_order_prices(
        action_a="SELL", ratio_a=1, entry_equilibrium=10.0, entry_trigger_level=2.0)
    # take_profit_spread = 10.0 - 2.0 = 8.0; closing price = -ratio_a*spread*sign = 8.0
    assert take_profit_price == 8.0
    # hard_stop_spread = 10.0 - (-1)*(2.0*2) = 14.0; closing price = -1*14.0*(-1) = 14.0
    assert round(hard_stop_price, 2) == 14.0
    print("PASS: SELL bracket -- take profit expands away from equilibrium, hard stop sits on the opposite side")


def test_bracket_order_prices_scales_with_ratio_a():
    hard_stop_price_1x, take_profit_price_1x = bracket_order_prices(
        action_a="BUY", ratio_a=1, entry_equilibrium=10.0, entry_trigger_level=2.0)
    hard_stop_price_3x, take_profit_price_3x = bracket_order_prices(
        action_a="BUY", ratio_a=3, entry_equilibrium=10.0, entry_trigger_level=2.0)
    assert take_profit_price_3x == take_profit_price_1x * 3
    assert hard_stop_price_3x == hard_stop_price_1x * 3
    print("PASS: bracket prices scale linearly with ratio_a, the closing combo's own share ratio")


def test_bracket_order_prices_rounds_to_the_cent():
    # Confirmed live 2026-08-20: an unrounded price (many decimal places,
    # from multiplying entry_equilibrium/entry_trigger_level through by
    # ratio_a) got both bracket orders rejected outright by IBKR with error
    # 110 ("price does not conform to the minimum price variation"),
    # leaving a real position with no Take Profit or Hard Stop ever placed.
    hard_stop_price, take_profit_price = bracket_order_prices(
        action_a="BUY", ratio_a=23, entry_equilibrium=352.279999999, entry_trigger_level=2.873333333)
    assert round(hard_stop_price, 2) == hard_stop_price
    assert round(take_profit_price, 2) == take_profit_price
    print("PASS: bracket prices are rounded to the cent, never rejected by IBKR for excess precision")


def test_stop_loss_double_touch_no_exit_while_still_on_the_expansion_side():
    position = _open(deviation=0.5, entry_spread=10.0)  # action_a="BUY", entry_equilibrium=0.0
    assert stop_loss_double_touch_exit(position, *_at(position, 5.0)) is False
    assert position.touched_zero_once is False
    print("PASS: no exit signal while the spread has never touched 0 divergence")


def test_stop_loss_double_touch_first_touch_does_not_exit():
    position = _open(deviation=0.5, entry_spread=10.0)
    assert stop_loss_double_touch_exit(position, *_at(position, 0.0)) is False
    assert position.touched_zero_once is True
    assert position.rose_above_zero_after_touch is False
    print("PASS: the first touch of 0 divergence records the touch but does not exit yet")


def test_stop_loss_double_touch_rising_back_above_after_touch_does_not_exit():
    position = _open(deviation=0.5, entry_spread=10.0)
    stop_loss_double_touch_exit(position, *_at(position, 0.0))  # first touch
    assert stop_loss_double_touch_exit(position, *_at(position, 3.0)) is False  # back on expansion side
    assert position.rose_above_zero_after_touch is True
    print("PASS: rising back above 0 after the first touch does not exit, just records the rise")


def test_stop_loss_double_touch_second_touch_exits():
    position = _open(deviation=0.5, entry_spread=10.0)
    stop_loss_double_touch_exit(position, *_at(position, 0.0))   # first touch
    stop_loss_double_touch_exit(position, *_at(position, 3.0))   # rose back above
    assert stop_loss_double_touch_exit(position, *_at(position, -0.5)) is True  # second touch
    print("PASS: touch, rise above, touch again -- exits on the second touch")


def test_stop_loss_double_touch_staying_on_equilibrium_side_is_not_a_second_touch():
    # Sitting on the equilibrium side continuously after the first touch,
    # without ever rising back to the expansion side, must not count as a
    # second touch -- it has to bounce OUT and back IN.
    position = _open(deviation=0.5, entry_spread=10.0)
    stop_loss_double_touch_exit(position, *_at(position, 0.0))    # first touch
    assert stop_loss_double_touch_exit(position, *_at(position, -1.0)) is False  # still on equilibrium side
    assert stop_loss_double_touch_exit(position, *_at(position, -2.0)) is False
    print("PASS: staying on the equilibrium side after the first touch never counts as a second touch")


def test_stop_loss_double_touch_mirrors_for_sell():
    position = _open(deviation=-0.5, entry_spread=10.0)  # action_a="SELL"
    stop_loss_double_touch_exit(position, *_at(position, 0.0))    # first touch
    stop_loss_double_touch_exit(position, *_at(position, -3.0))   # rose back above (expansion side for SELL is negative)
    assert stop_loss_double_touch_exit(position, *_at(position, 0.5)) is True  # second touch
    print("PASS: the double-touch state machine mirrors correctly for a SELL entry")


def test_stop_loss_triple_touch_no_exit_while_still_on_the_expansion_side():
    position = _open(deviation=0.5, entry_spread=10.0)  # action_a="BUY", entry_equilibrium=0.0
    assert stop_loss_triple_touch_exit(position, *_at(position, 5.0)) is False
    assert position.zero_touch_count == 0
    print("PASS: no exit signal while the spread has never touched 0 divergence")


def test_stop_loss_triple_touch_first_touch_does_not_exit():
    position = _open(deviation=0.5, entry_spread=10.0)
    assert stop_loss_triple_touch_exit(position, *_at(position, 0.0)) is False
    assert position.zero_touch_count == 1
    assert position.rose_since_last_touch is False
    print("PASS: the first touch of 0 divergence records the touch but does not exit yet")


def test_stop_loss_triple_touch_second_touch_does_not_exit():
    # Per the client 2026-09-05: three complete cycles (Touch 1 -> Bounce
    # -> Touch 2 -> Bounce -> Touch 3) are required -- the second touch
    # alone must NOT exit (unlike the old double-touch rule it replaces).
    position = _open(deviation=0.5, entry_spread=10.0)
    stop_loss_triple_touch_exit(position, *_at(position, 0.0))   # touch 1
    stop_loss_triple_touch_exit(position, *_at(position, 3.0))   # bounce
    assert stop_loss_triple_touch_exit(position, *_at(position, -0.5)) is False  # touch 2
    assert position.zero_touch_count == 2
    print("PASS: touch, bounce, touch again does not exit -- that's only the second touch")


def test_stop_loss_triple_touch_third_touch_exits():
    position = _open(deviation=0.5, entry_spread=10.0)
    stop_loss_triple_touch_exit(position, *_at(position, 0.0))   # touch 1
    stop_loss_triple_touch_exit(position, *_at(position, 3.0))   # bounce
    stop_loss_triple_touch_exit(position, *_at(position, -0.5))  # touch 2
    stop_loss_triple_touch_exit(position, *_at(position, 2.0))   # bounce
    assert stop_loss_triple_touch_exit(position, *_at(position, -1.0)) is True  # touch 3
    print("PASS: touch, bounce, touch, bounce, touch a third time -- exits on the third touch")


def test_stop_loss_triple_touch_staying_on_equilibrium_side_does_not_advance_the_count():
    # Sitting on the equilibrium side continuously after a touch, without
    # ever bouncing back to the expansion side, must not count as a new
    # touch -- it has to bounce OUT and back IN each time.
    position = _open(deviation=0.5, entry_spread=10.0)
    stop_loss_triple_touch_exit(position, *_at(position, 0.0))    # touch 1
    assert stop_loss_triple_touch_exit(position, *_at(position, -1.0)) is False  # still on equilibrium side
    assert stop_loss_triple_touch_exit(position, *_at(position, -2.0)) is False
    assert position.zero_touch_count == 1
    print("PASS: staying on the equilibrium side never advances the touch count")


def test_stop_loss_triple_touch_mirrors_for_sell():
    position = _open(deviation=-0.5, entry_spread=10.0)  # action_a="SELL"
    stop_loss_triple_touch_exit(position, *_at(position, 0.0))    # touch 1
    stop_loss_triple_touch_exit(position, *_at(position, -3.0))   # bounce (expansion side for SELL is negative)
    stop_loss_triple_touch_exit(position, *_at(position, 0.5))    # touch 2
    stop_loss_triple_touch_exit(position, *_at(position, -2.0))   # bounce
    assert stop_loss_triple_touch_exit(position, *_at(position, 1.0)) is True  # touch 3
    print("PASS: the triple-touch state machine mirrors correctly for a SELL entry")


def _open_single_leg(traded_symbol="TYL", action="BUY", entry_spread=10.0,
                      entry_equilibrium=0.0, entry_trigger_level=2.0, hedge_ratio=1.0):
    return open_single_leg_position(
        symbol_a="TYL", symbol_b="WDAY", traded_symbol=traded_symbol, action=action, qty=100,
        entry_price_a=100.0, entry_price_b=100.0, entry_notional=10_000,
        half_life_minutes=30, current_spread=entry_spread, hedge_ratio=hedge_ratio, now=RTH_MORNING,
        entry_equilibrium=entry_equilibrium, entry_trigger_level=entry_trigger_level,
    )


def test_cross_leg_exit_closes_the_hedge_when_the_expansion_bracket_fires():
    """Per the client 2026-09-09: either leg's bracket firing must close
    both. The expansion leg's TP filled, so the HEDGE leg is the survivor
    that needs closing, along with its own three orphaned bracket orders."""
    from phase2.risk_manager import attach_hedge_leg, leg_still_open_after_bracket_fill
    position = _open_single_leg(traded_symbol="TYL", action="BUY")
    attach_hedge_leg(position, "WDAY", "SELL", 74, 50.0)
    position.hedge_tp_order_id, position.hedge_hard_stop_order_id, position.hedge_timed_exit_order_id = 11, 12, 13
    survivor, orders = leg_still_open_after_bracket_fill(position, fired_on_hedge=False)
    assert survivor == "WDAY"
    assert orders == (11, 12, 13)
    print("PASS: the hedge leg is the survivor to close when the expansion leg's bracket fires")


def test_cross_leg_exit_closes_the_expansion_leg_when_the_hedge_bracket_fires():
    """The mirror case, and the one most likely to be got backwards: the
    HEDGE leg's stop filled, so the EXPANSION leg is what is still open."""
    from phase2.risk_manager import attach_hedge_leg, leg_still_open_after_bracket_fill
    position = _open_single_leg(traded_symbol="TYL", action="BUY")
    attach_hedge_leg(position, "WDAY", "SELL", 74, 50.0)
    position.tp_order_id, position.hard_stop_order_id, position.timed_exit_order_id = 21, 22, 23
    survivor, orders = leg_still_open_after_bracket_fill(position, fired_on_hedge=True)
    assert survivor == "TYL"
    assert orders == (21, 22, 23)
    print("PASS: the expansion leg is the survivor to close when the hedge leg's bracket fires")


def test_cross_leg_exit_is_a_noop_with_no_opposite_leg():
    """An expansion-leg-only position (the EWLV balance funded nothing) --
    the bracket that fired closed everything, nothing survives to close."""
    from phase2.risk_manager import leg_still_open_after_bracket_fill
    position = _open_single_leg(traded_symbol="TYL", action="BUY")
    assert leg_still_open_after_bracket_fill(position, fired_on_hedge=False) is None
    print("PASS: cross leg exit does nothing when there was never an opposite leg")


def test_cross_leg_exit_survivor_is_correct_when_the_expansion_leg_is_b():
    """traded_leg=="B" is the case the 2026-09-03 rework already produced
    one silent bug in -- the survivor must be symbol_b, not symbol_a."""
    from phase2.risk_manager import attach_hedge_leg, leg_still_open_after_bracket_fill
    position = _open_single_leg(traded_symbol="WDAY", action="BUY")
    assert position.traded_leg == "B"
    attach_hedge_leg(position, "TYL", "SELL", 74, 50.0)
    survivor, _ = leg_still_open_after_bracket_fill(position, fired_on_hedge=True)
    assert survivor == "WDAY"
    print("PASS: the survivor is resolved from traded_leg, correct when the expansion leg is B")


def test_wide_hard_stop_price_matches_the_clients_at_entry_example():
    """Per the client 2026-09-09: "Hard Stop Loss @ Entry you will use
    Trigger-13% + Trigger." Entry=$100, Trigger=$5.00: Minimum Gap at Entry
    5.00x(1-0.13)=4.35, distance 4.35+5.00=9.35, Stop=$90.65 (long) /
    $109.35 (short). This is the FIRST stop placed at entry, using the
    worst-case sizing distance, not the real entry_divergence."""
    assert abs(wide_hard_stop_price("BUY", 100.0, 5.0) - 90.65) < 1e-9
    assert abs(wide_hard_stop_price("SELL", 100.0, 5.0) - 109.35) < 1e-9
    print("PASS: wide_hard_stop_price matches the client's Hard Stop Loss @ Entry example exactly")


def test_wide_hard_stop_price_is_wider_than_the_after_entry_stop():
    """The whole point of the two-stage design: the AT ENTRY stop must be
    at least as far from entry_price as the real, tighter AFTER ENTRY one
    (resting_hard_stop_price), since Minimum Gap at Entry is the worst
    case (largest) entry_divergence the entry window allows -- the real
    entry_divergence is always <= it."""
    trigger = 5.0
    wide = wide_hard_stop_price("BUY", 100.0, trigger)
    after = resting_hard_stop_price("BUY", 100.0, trigger, entry_divergence=2.5)
    assert wide < after  # further below entry price on a long
    print("PASS: the AT ENTRY stop is wider (further from entry) than the AFTER ENTRY one")


def test_pnl_stop_loss_price_matches_the_clients_five_percent_of_ewlv():
    """Per the client 2026-09-09: "Put that -5% PnL Stop Loss. It is Ibkr
    PnL. 5% of EWLV." His own numbers: 100,000 EWLV at 5% is a 5,000 budget
    over 763 shares = 6.5531/share, so a 100.00 long stops at 93.4469."""
    price = pnl_stop_loss_price("BUY", 100.0, 763, 100_000, 0.05)
    assert abs(price - 93.4469) < 1e-4, price
    print("PASS: pnl_stop_loss_price puts a 5% of EWLV loss budget onto the shares held")


def test_pnl_stop_loss_price_mirrors_for_a_short():
    price = pnl_stop_loss_price("SELL", 100.0, 763, 100_000, 0.05)
    assert abs(price - 106.5531) < 1e-4, price
    print("PASS: the PnL stop sits above entry for a short, below for a long")


def test_pnl_stop_loss_price_is_the_outer_backstop_not_a_duplicate():
    """On the client's own numbers the two stops are NOT the same price, and
    the ordering matters: the Hard Stop sits at 94.00 (its distance uses the
    REAL entry divergence, 2.50+3.50=6.00/share) while the PnL stop sits at
    93.4469 (a 5% budget over 763 shares = 6.5531/share). The gap exists
    because sizing used the worst case Minimum Gap at Entry (3.045+3.50=
    6.545) rather than this entry's actual divergence.

    So the thesis based Hard Stop fires first and the account percentage
    stop is the outer safety net behind it, which is the right way round --
    it only comes into play if the Hard Stop is jumped or never placed."""
    pnl = pnl_stop_loss_price("BUY", 100.0, 763, 100_000, 0.05)
    hard = resting_hard_stop_price("BUY", 100.0, 3.50, 2.50)
    assert pnl < hard, (pnl, hard)  # further below entry on a long
    assert abs(pnl - 93.4469) < 1e-4 and abs(hard - 94.00) < 1e-9
    print("PASS: the PnL stop sits outside the Hard Stop, as an account level backstop")


def test_pnl_stop_loss_price_rejects_zero_quantity():
    """A zero quantity would divide by zero and silently produce nonsense
    rather than a price -- refused outright."""
    try:
        pnl_stop_loss_price("BUY", 100.0, 0, 100_000, 0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero quantity must be refused")
    print("PASS: pnl_stop_loss_price refuses a zero quantity instead of dividing by zero")


def test_attach_hedge_leg_rejects_a_same_direction_second_leg():
    """A second leg in the SAME direction doubles the position instead of
    hedging it -- per the client 2026-09-09 it must be "the Opposite Leg in
    the opposite direction", so this is refused rather than executed."""
    from phase2.risk_manager import attach_hedge_leg
    position = _open_single_leg(traded_symbol="TYL", action="BUY")
    try:
        attach_hedge_leg(position, "WDAY", "BUY", 74, 50.0)
    except ValueError as e:
        assert "opposite" in str(e)
    else:
        raise AssertionError("a same-direction hedge leg must be rejected")
    print("PASS: attach_hedge_leg refuses a second leg in the same direction as the expansion leg")


def test_unrealized_pnl_includes_the_opposite_leg():
    """Per the client 2026-09-09 both legs are real money at the broker, so
    both belong in the P&L the daily loss cap reads. Expansion leg: BUY 100
    TYL at 100, now 101 -> +100. Hedge leg: SELL 74 WDAY at 50, now 49 ->
    +74. Total +174."""
    from phase2.risk_manager import attach_hedge_leg
    position = _open_single_leg(traded_symbol="TYL", action="BUY")
    attach_hedge_leg(position, "WDAY", "SELL", 74, 50.0)
    pnl = unrealized_pnl(position, 101.0, 49.0)
    assert abs(pnl - 174.0) < 1e-9, pnl
    print("PASS: unrealized_pnl adds the opposite leg's own P&L to the expansion leg's")


def test_unrealized_pnl_unchanged_when_no_opposite_leg_was_placed():
    """hedge_qty == 0 means the EWLV Balance couldn't fund a share (or the
    order never filled) -- the single-leg maths must be untouched."""
    position = _open_single_leg(traded_symbol="TYL", action="BUY")
    assert position.hedge_qty == 0
    assert abs(unrealized_pnl(position, 101.0, 49.0) - 100.0) < 1e-9
    print("PASS: a position with no opposite leg computes P&L exactly as the single-leg version did")


def test_stop_loss_double_touch_mirrors_correctly_when_traded_leg_is_b_despite_buy_action():
    """Regression guard for the bug the 2026-09-03 single-leg rework
    introduced and this same rework fixed: action_a=="BUY" no longer
    means positive divergence is the expansion side once the traded leg
    can be B. This position is entered with entry_spread BELOW
    entry_equilibrium (negative divergence is the real expansion side),
    even though action=="BUY" (buying leg B, which is what pushes the
    A-referenced spread down) -- the double-touch state machine must key
    off entry_spread vs entry_equilibrium, not action_a, or it would
    watch the WRONG side for the first touch entirely."""
    position = _open_single_leg(traded_symbol="WDAY", action="BUY", entry_spread=-10.0, hedge_ratio=1.0)
    # live spread = price_a - hedge_ratio*price_b; touch 0 divergence (spread == entry_equilibrium == 0)
    stop_loss_double_touch_exit(position, 100.0, 100.0)              # first touch (spread=0)
    stop_loss_double_touch_exit(position, 97.0, 100.0)               # rose back to the expansion side (spread=-3, still negative)
    assert stop_loss_double_touch_exit(position, 100.5, 100.0) is True  # second touch (spread=0.5, equilibrium side)
    print("PASS: double-touch keys off entry_spread vs entry_equilibrium, correct even when BUY trades leg B")


def test_take_profit_hard_stop_exit_fires_take_profit():
    position = _open_single_leg(traded_symbol="TYL", action="BUY", entry_spread=10.0,
                                 entry_equilibrium=0.0, entry_trigger_level=2.0, hedge_ratio=1.0)
    # live spread = price_a - price_b; take_profit_level = 0 + 2 = 2.0
    assert take_profit_hard_stop_exit(position, 102.0, 100.0) == "take_profit"
    print("PASS: take_profit_hard_stop_exit fires take_profit at entry_equilibrium + trigger")


def test_take_profit_hard_stop_exit_fires_hard_stop():
    position = _open_single_leg(traded_symbol="TYL", action="BUY", entry_spread=10.0,
                                 entry_equilibrium=0.0, entry_trigger_level=2.0, hedge_ratio=1.0)
    # hard_stop_level = 0 - 2*2 = -4.0
    assert take_profit_hard_stop_exit(position, 96.0, 100.0) == "hard_stop"
    print("PASS: take_profit_hard_stop_exit fires hard_stop at entry_equilibrium - 2x trigger")


def test_take_profit_hard_stop_exit_none_in_between():
    position = _open_single_leg(traded_symbol="TYL", action="BUY", entry_spread=10.0,
                                 entry_equilibrium=0.0, entry_trigger_level=2.0, hedge_ratio=1.0)
    assert take_profit_hard_stop_exit(position, 100.0, 100.0) is None  # live spread = 0, inside both levels
    print("PASS: take_profit_hard_stop_exit stays None between the two trigger levels")


def test_take_profit_hard_stop_exit_direction_correct_when_traded_leg_is_b():
    """Same regression case as the double-touch test above: BUY on leg B
    with entry_spread below entry_equilibrium means the spread FALLING
    further is the expansion side, not rising. Must fire take_profit on a
    falling live spread here, not a rising one."""
    position = _open_single_leg(traded_symbol="WDAY", action="BUY", entry_spread=-10.0,
                                 entry_equilibrium=0.0, entry_trigger_level=2.0, hedge_ratio=1.0)
    # take_profit_level = 0 + (-1)*2 = -2.0 -- fires when live spread falls to -2 or below
    assert take_profit_hard_stop_exit(position, 98.0, 100.0) == "take_profit"
    # hard_stop_level = 0 - (-1)*2*2 = 4.0 -- fires when live spread rises to 4 or above
    assert take_profit_hard_stop_exit(position, 104.0, 100.0) == "hard_stop"
    print("PASS: take_profit_hard_stop_exit direction is correct even when BUY trades leg B")


def test_resting_take_profit_price_buy_and_sell():
    """Per the client 2026-09-04's own worked example: entry=$100,
    Trigger=$3.50, Divergence_entry=$2.50 -> remaining gap $1.00."""
    assert resting_take_profit_price("BUY", 100.0, 3.50, 2.50) == 101.0
    assert resting_take_profit_price("SELL", 100.0, 3.50, 2.50) == 99.0
    print("PASS: resting_take_profit_price matches the client's own worked example, both directions")


def test_resting_hard_stop_price_buy_and_sell():
    """Per the client 2026-09-05's "100% Mean Reversion x 2" correction
    (second same-topic change in two days): "Hard Stop Loss: 100% Mean
    Reversion x 2." Per the client's own two worked examples: entry=$100,
    Trigger=$3.50, Divergence_entry=$2.50 -> Remaining Gap To Zero +
    Trigger = $2.50+$3.50=$6.00."""
    assert resting_hard_stop_price("BUY", 100.0, 3.50, 2.50) == 94.00
    assert resting_hard_stop_price("SELL", 100.0, 3.50, 2.50) == 106.00
    print("PASS: resting_hard_stop_price matches both of the client's own 100% Mean Reversion x 2 worked examples")


def test_unrealized_pnl_sign_for_sell_and_buy():
    sell_position = _open(deviation=-0.5, entry_spread=10.0)  # deviation<0 -> action_a=SELL under new convention
    buy_position = _open(deviation=0.5, entry_spread=10.0)    # deviation>0 -> action_a=BUY under new convention
    assert sell_position.action_a == "SELL" and buy_position.action_a == "BUY"
    assert unrealized_pnl(sell_position, *_at(sell_position, 9.5)) == 50.0    # short spread, spread FELL -> profit
    assert unrealized_pnl(buy_position, *_at(buy_position, 10.5)) == 50.0     # long spread, spread ROSE -> profit
    assert unrealized_pnl_pct(buy_position, *_at(buy_position, 10.5)) == 50.0 / 50_000
    print("PASS: unrealized P&L sign matches the new expansion-strategy BUY/SELL convention "
          "(profits from the spread continuing the SAME direction it entered on)")


def test_unrealized_pnl_reflects_only_the_one_real_traded_leg():
    """Rebuilt 2026-09-03 for the client's "Expansion Leg" pivot -- only ONE
    stock is ever real money now, so P&L must equal exactly that leg's own
    dollar move and must NOT pick up anything from the other (untraded)
    leg's price move, even though that leg's price still feeds the
    spread-based exit trigger. Supersedes the old two-leg version of this
    test, which summed both legs' dollar P&L -- correct back when both legs
    were real combo-order money, no longer true now that only one is."""
    position = open_single_leg_position(
        symbol_a="TYL", symbol_b="WDAY", traded_symbol="TYL", action="BUY", qty=100,
        entry_price_a=100.0, entry_price_b=100.0, entry_notional=10_000,
        half_life_minutes=30, current_spread=10.0, hedge_ratio=1.0, now=RTH_MORNING,
    )
    # traded leg (A) moves +1.30 -> +130 real P&L; untraded leg (B) also
    # moves, but it's never real money and must be ignored.
    assert abs(unrealized_pnl(position, 101.30, 105.0) - 130.0) < 1e-9
    print("PASS: unrealized_pnl reflects only the one real traded leg, ignoring the untraded leg's price move")


def test_unrealized_pnl_works_for_traded_leg_b_too():
    position = open_single_leg_position(
        symbol_a="TYL", symbol_b="WDAY", traded_symbol="WDAY", action="SELL", qty=245,
        entry_price_a=100.0, entry_price_b=100.0, entry_notional=10_000,
        half_life_minutes=30, current_spread=10.0, hedge_ratio=1.0, now=RTH_MORNING,
    )
    # short 245 WDAY, price falls 0.40 -> profit
    assert abs(unrealized_pnl(position, 999.0, 99.60) - 98.0) < 1e-9
    print("PASS: unrealized_pnl works the same way when leg B is the one actually traded")


def test_open_single_leg_position_carries_the_real_ibkr_probed_margin():
    """Per the client 2026-09-14: "I want to see the initial margin
    required by IBKR and see if it is the normal" -- distinct from
    entry_margin (the flat 35% estimate), this is the real dollar figure
    from the last real-order probe rejection main.py's resize loop hit."""
    position = open_single_leg_position(
        symbol_a="TYL", symbol_b="WDAY", traded_symbol="WDAY", action="BUY", qty=17,
        entry_price_a=100.0, entry_price_b=191.73, entry_notional=3259.41,
        half_life_minutes=30, current_spread=10.0, hedge_ratio=1.0, now=RTH_MORNING,
        entry_margin=1140.87, entry_margin_ibkr_probed=6004.47, entry_margin_ibkr_probed_qty=22,
    )
    assert position.entry_margin == 1140.87
    assert position.entry_margin_ibkr_probed == 6004.47
    assert position.entry_margin_ibkr_probed_qty == 22
    print("PASS: open_single_leg_position carries the real IBKR-probed margin separately from the flat estimate")


def test_open_single_leg_position_defaults_the_ibkr_probed_margin_to_none():
    """The probe loop can exit on its very first try with no rejection at
    all (the initial size already fit) -- there's no real figure to show
    in that case, and the field must say so explicitly rather than a
    misleading 0.0."""
    position = open_single_leg_position(
        symbol_a="TYL", symbol_b="WDAY", traded_symbol="WDAY", action="BUY", qty=17,
        entry_price_a=100.0, entry_price_b=191.73, entry_notional=3259.41,
        half_life_minutes=30, current_spread=10.0, hedge_ratio=1.0, now=RTH_MORNING,
    )
    assert position.entry_margin_ibkr_probed is None
    assert position.entry_margin_ibkr_probed_qty is None
    print("PASS: entry_margin_ibkr_probed defaults to None, not a misleading 0.0, when nothing was ever rejected")


def test_unrealized_pnl_is_immune_to_live_beta_drift():
    """Regression guard for the phantom-P&L bug: P&L used to be computed by
    differencing two spreads taken at DIFFERENT betas, so beta drift alone
    invented profit/loss. With prices unchanged, P&L must be exactly zero no
    matter how far the live beta has moved."""
    position = _open(deviation=0.5, qty_a=100, qty_b=245, entry_spread=10.0)
    flat = unrealized_pnl(position, position.entry_price_a, position.entry_price_b)
    assert flat == 0.0

    # what the old code effectively did: same prices, spread recomputed at a
    # beta that has drifted 10% -- this is the number that used to leak in
    drifted_beta = position_hedge_ratio(position) * 1.10
    old_style_spread = position.entry_price_a - drifted_beta * position.entry_price_b
    old_style_pnl = position.qty_a * 1 * (old_style_spread - position.entry_spread)
    assert abs(old_style_pnl) > 200.0  # would have tripped the 0.05%-of-ELV condition on its own
    print("PASS: P&L is zero when prices haven't moved, regardless of beta drift "
          f"(old approach would have invented ${abs(old_style_pnl):,.0f})")




def test_record_exit_does_not_halt_on_a_small_loss():
    daily_state = DailyState()
    record_exit(daily_state, cumulative_realized_pnl_today=-50.0, elv_at_925=50_000.0)  # -0.1% of ELV
    assert daily_state.trading_halted is False
    print("PASS: a small loss alone does not halt the day -- only the -2% ELV cap does")


def test_record_exit_halts_at_the_2pct_elv_cap():
    # Maximum Loss for the Day = 2% of ELV at 9:25am ET.
    daily_state = DailyState()
    record_exit(daily_state, cumulative_realized_pnl_today=-999.99, elv_at_925=50_000.0)  # -1.999998%
    assert daily_state.trading_halted is False
    record_exit(daily_state, cumulative_realized_pnl_today=-1_000.0, elv_at_925=50_000.0)  # exactly -2%
    assert daily_state.trading_halted is True
    print("PASS: trading halts once realized P&L reaches -2% of ELV at 9:25am ET, inclusive of the boundary")


def test_reentry_allowed_for_take_profit_before_1030am():
    # Per the client 2026-08-21: cutoff moved to 10:30am ET.
    assert reentry_allowed("take_profit", datetime(2026, 7, 27, 10, 29)) is True
    print("PASS: re-entry allowed after a Take Profit close before 10:30am ET")


def test_reentry_blocked_for_take_profit_at_or_after_1030am():
    # "no re-entry after 10:30am ET."
    assert reentry_allowed("take_profit", datetime(2026, 7, 27, 10, 30)) is False
    assert reentry_allowed("take_profit", datetime(2026, 7, 27, 14, 0)) is False
    print("PASS: re-entry blocked for a Take Profit close at or after 10:30am ET")


def test_reentry_blocked_for_every_other_exit_reason_regardless_of_time():
    # Stop Loss, the max hold time, EOD, and a failed bracket placement all
    # end the day outright -- Take Profit before 10:30am is the only
    # carve-out.
    morning = datetime(2026, 7, 27, 10, 0)
    for reason in ("stop_loss", "max_hold_time", "eod", "bracket_placement_failed"):
        assert reentry_allowed(reason, morning) is False
    print("PASS: every non-Take-Profit exit reason blocks re-entry, even early in the morning")


def test_is_early_close_day():
    assert is_early_close_day(dtime(13, 0)) is True   # 1pm close -> early close day
    assert is_early_close_day(dtime(16, 0)) is False  # exactly 4pm -> normal full day
    assert is_early_close_day(dtime(15, 59)) is True  # even 1 minute early counts
    print("PASS: is_early_close_day correctly flags closeTime < 16:00pm ET (v7 rule)")


def _state_path():
    import tempfile
    return Path(tempfile.mkdtemp()) / "daily_state.json"


def test_daily_state_survives_a_restart_within_the_same_day():
    """Audit finding 2026-08-27: daily_state was rebuilt fresh on every
    process start with nothing persisting it, so a mid-session restart
    silently reset trades_taken_today (defeating "Maximum Trades per day =
    2" -- take 2, restart, take 2 more) and cleared reentry_locked_out
    (set by a stop loss / news stop / hard stop, meaning "stop trading
    today")."""
    from datetime import date as _date
    path = _state_path()
    today = _date(2026, 8, 27)
    state = DailyState(trades_taken_today=2, reentry_locked_out=True,
                        realized_pnl_today=-153.20, trading_halted=True)
    save_daily_state(state, path, today=today)

    restored = load_daily_state(path, today=today)
    assert restored.trades_taken_today == 2
    assert restored.reentry_locked_out is True
    assert restored.trading_halted is True
    assert restored.realized_pnl_today == -153.20
    print("PASS: the daily risk counters survive a same-day restart")


def test_daily_state_from_a_previous_day_is_ignored():
    """The date stamp is what makes failing open safe -- yesterday's
    lockout must never carry into a new session."""
    from datetime import date as _date
    path = _state_path()
    save_daily_state(DailyState(trades_taken_today=2, reentry_locked_out=True),
                      path, today=_date(2026, 8, 26))

    restored = load_daily_state(path, today=_date(2026, 8, 27))
    assert restored.trades_taken_today == 0
    assert restored.reentry_locked_out is False
    print("PASS: a prior day's persisted state is ignored, not carried forward")


def test_daily_state_fails_open_on_a_missing_or_corrupt_file():
    """Refusing to trade because leftovers couldn't be parsed would be its
    own outage -- a fresh DailyState (normal trading) is the safe fallback,
    since the file existing at all is the exceptional case."""
    from datetime import date as _date
    today = _date(2026, 8, 27)
    missing = _state_path()  # never written
    assert load_daily_state(missing, today=today).trades_taken_today == 0

    corrupt = _state_path()
    corrupt.write_text("{ not json")
    restored = load_daily_state(corrupt, today=today)
    assert restored.trades_taken_today == 0
    assert restored.reentry_locked_out is False
    print("PASS: a missing or corrupt state file falls back to a fresh DailyState")


def test_saving_daily_state_never_raises_into_the_trading_loop():
    """Best-effort by design -- a persistence failure must not take down a
    live session."""
    save_daily_state(DailyState(), "/nonexistent/dir/daily_state.json")
    print("PASS: a failed save logs and continues rather than raising")


def test_pair_trigger_state_survives_a_restart_within_the_same_day():
    """Real incident, 2026-09-14: a mid-session restart (unrelated to this
    flag) reset trigger_reached_today back to False in memory for every
    pair. PLTR/WDAY had already reached its trigger earlier that session
    (correctly locking it out for the rest of the day per the client
    2026-08-19: "Trigger has not previously been reached at any point
    during the current RTH session"), the restart cleared that, and the
    pair went on to actually enter a real trade it should never have been
    eligible for. Same class of fix as daily_state's own 2026-08-27
    finding, applied to this per-pair sticky flag."""
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    et = _ZoneInfo("America/New_York")
    path = _state_path()
    entered_at = _datetime(2026, 9, 14, 10, 15, tzinfo=et)
    restarted_at = _datetime(2026, 9, 14, 14, 47, tzinfo=et)  # same day, before 8pm ET
    pairs = [("PLTR", "WDAY", True), ("ABT", "MDT", False), ("CVX", "XOM", True)]
    save_pair_trigger_state(pairs, path, now=entered_at)

    restored = load_pair_trigger_state(path, now=restarted_at)
    assert restored.get("PLTR/WDAY") is True
    assert restored.get("CVX/XOM") is True
    assert "ABT/MDT" not in restored  # never reached trigger, nothing to persist
    print("PASS: trigger_reached_today survives a same-day restart, per pair")


def test_pair_trigger_state_expires_at_8pm_et():
    """Per the client 2026-09-14: "If trigger is reached in current RTH
    session, pair is locked until 8pm ET." Stamped as an explicit ET
    wall-clock timestamp rather than a raw calendar date, so this holds
    even across the DST boundary where the server's own UTC calendar day
    wouldn't line up with 8pm ET."""
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    et = _ZoneInfo("America/New_York")
    path = _state_path()
    save_pair_trigger_state([("PLTR", "WDAY", True)], path, now=_datetime(2026, 9, 14, 10, 15, tzinfo=et))

    just_before = load_pair_trigger_state(path, now=_datetime(2026, 9, 14, 19, 59, tzinfo=et))
    assert just_before.get("PLTR/WDAY") is True

    at_or_after = load_pair_trigger_state(path, now=_datetime(2026, 9, 14, 20, 0, tzinfo=et))
    assert at_or_after == {}
    print("PASS: the lock expires at 8pm ET, not at a UTC calendar-day rollover")


def test_pair_trigger_state_from_a_previous_day_is_ignored():
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    et = _ZoneInfo("America/New_York")
    path = _state_path()
    save_pair_trigger_state([("PLTR", "WDAY", True)], path, now=_datetime(2026, 9, 13, 15, 0, tzinfo=et))

    restored = load_pair_trigger_state(path, now=_datetime(2026, 9, 14, 9, 30, tzinfo=et))
    assert restored == {}
    print("PASS: a prior day's persisted trigger state is ignored, not carried forward")


def test_pair_trigger_state_fails_open_on_a_missing_or_corrupt_file():
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    now = _datetime(2026, 9, 14, 10, 0, tzinfo=_ZoneInfo("America/New_York"))
    missing = _state_path()
    assert load_pair_trigger_state(missing, now=now) == {}

    corrupt = _state_path()
    corrupt.write_text("{ not json")
    assert load_pair_trigger_state(corrupt, now=now) == {}
    print("PASS: a missing or corrupt trigger state file fails open to an empty map")


def test_saving_pair_trigger_state_never_raises_into_the_trading_loop():
    save_pair_trigger_state([("AAA", "BBB", True)], "/nonexistent/dir/pair_trigger_state.json")
    print("PASS: a failed save logs and continues rather than raising")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
