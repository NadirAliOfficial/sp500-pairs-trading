"""Core spread/divergence math for Phase 2 live trading -- no I/O, pure functions
so this can be unit tested against synthetic data without Databento or IBKR.

Terminology matches the client's finalized spec: "Equilibrium Spread"
replaced "Correlation Baseline" everywhere -- correlation is only the
Phase 1 pair-selection filter, the algorithm actually trades deviations of
the spread from its own rolling mean.
"""
from dataclasses import dataclass
from datetime import time as dtime, timedelta

import pandas as pd

EQUILIBRIUM_WINDOW_MINUTES = 60  # "looks at the last 60 minutes" per the spec's intro section
MAX_EVENT_DURATION = timedelta(hours=2)
DEFAULT_DIVERGENCE_TOLERANCE = 0.01  # "small predefined tolerance" around zero -- spread rarely hits exactly 0 due to noise


@dataclass
class DivergenceEvent:
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    max_spread: float  # peak |spread - equilibrium| reached during the event
    duration: timedelta
    max_adverse_excursion: float  # how far it widened past its opening deviation before reverting


def compute_spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: float) -> pd.Series:
    """Spread_t = P_A,t - hedge_ratio * P_B,t, matching the hedge ratio convention
    already used in screening.engle_granger_hedge_ratio_and_spread."""
    return price_a - hedge_ratio * price_b


def equilibrium_spread(spread: pd.Series, window_minutes: int = EQUILIBRIUM_WINDOW_MINUTES) -> pd.Series:
    """Rolling mean of the spread -- the "trading reference" everything else
    (entry trigger, take profit) is measured against."""
    return spread.rolling(window_minutes, min_periods=window_minutes).mean()


def detect_divergence_events(
    spread: pd.Series,
    equilibrium: pd.Series,
    tolerance: float = DEFAULT_DIVERGENCE_TOLERANCE,
    max_duration: timedelta = MAX_EVENT_DURATION,
) -> list[DivergenceEvent]:
    """Finds every divergence event that opened, widened, and reverted back to
    equilibrium within max_duration -- events that never revert, or take longer
    than max_duration, are discarded (per spec: only events that "closed within
    2 hours" count toward the divergence trigger level).

    An event begins the first time |spread - equilibrium| leaves the tolerance
    band around zero, and ends the first time it returns inside that band.
    Intra-event noise (small dips back toward zero without fully closing) does
    not split one event into several -- only a full return inside the
    tolerance band closes it, matching "closed inside 2 hours" being about the
    divergence itself, not a bar-by-bar wiggle.
    """
    frame = pd.concat(
        [spread.rename("spread"), equilibrium.rename("equilibrium")], axis=1).dropna()
    events: list[DivergenceEvent] = []

    in_event = False
    open_time = None
    open_deviation = 0.0
    peak_abs_deviation = 0.0
    peak_equilibrium = 0.0
    direction = 0

    for ts, row in frame.iterrows():
        dev = row.spread - row.equilibrium
        abs_dev = abs(dev)
        if not in_event:
            if abs_dev > tolerance:
                in_event = True
                open_time = ts
                open_deviation = abs_dev
                peak_abs_deviation = abs_dev
                peak_equilibrium = row.equilibrium
                direction = 1 if dev > 0 else -1
        else:
            if abs_dev > peak_abs_deviation:
                peak_abs_deviation = abs_dev
                peak_equilibrium = row.equilibrium
            # The gap is closed once the spread has given back the full
            # divergence it opened up: a gap that peaked $2.70 wide is closed
            # when the spread has travelled $2.70 back toward the mean, which
            # is the spread arriving back at its equilibrium. Per spec, "the
            # divergence gap closes back to zero / Spread - Equilibrium
            # Spread = 0".
            #
            # Measured against the live equilibrium, so the event ends the
            # moment the spread crosses back through it. Anchoring instead to
            # the equilibrium as it stood at the peak was tried and is wrong
            # here: the 60-minute mean drifts materially within a session, so
            # requiring the spread to return to a stale level stretched
            # events past the 2-hour limit and discarded almost all of them
            # (GE/RTX went to 3 events for the whole session, 2 of them over
            # 2 hours, against 42 under this definition).
            closed = abs_dev <= tolerance or (dev > 0) != (direction == 1)
            if closed:
                duration = ts - open_time
                if duration <= max_duration:
                    events.append(DivergenceEvent(
                        open_time=open_time,
                        close_time=ts,
                        max_spread=peak_abs_deviation,
                        duration=duration,
                        max_adverse_excursion=peak_abs_deviation - open_deviation,
                    ))
                in_event = False

    return events


def detect_non_reverting_rth_events(
    spread: pd.Series,
    equilibrium: pd.Series,
    rth_open: dtime,
    rth_close: dtime,
    tolerance: float = DEFAULT_DIVERGENCE_TOLERANCE,
) -> list[DivergenceEvent]:
    """The trigger basis for the client's 2026-08-22 expansion-strategy
    rewrite: "Bot will filter off co-integration, and filter in expansion...
    The historical maximum spread divergence inside the lookback period,
    calculated using only RTH divergence events that: opened during RTH and
    did not mean-revert."

    The exact mirror image of detect_divergence_events -- that function
    keeps only events that DID close within MAX_EVENT_DURATION (the old
    mean-reversion trigger basis); this keeps only events that (a) opened
    during regular trading hours (rth_open <= open_time.time() < rth_close,
    by time-of-day only -- so an event opening in TODAY's premarket, before
    rth_open, is excluded even though the caller may feed a premarket-
    inclusive window for the wider 9:31am exception recalculation) and (b)
    were still diverged when the data ran out, i.e. never returned inside
    the tolerance band. No MAX_EVENT_DURATION cap applies here -- "did not
    mean-revert" is unbounded by construction, not bounded-then-discarded
    like the old 2-hour rule.

    close_time/duration are set to the series' own last timestamp for a
    kept (still-open) event, since a non-reverting event has no real close
    by definition -- these two fields are informational only here, unlike
    detect_divergence_events where duration is load-bearing."""
    frame = pd.concat(
        [spread.rename("spread"), equilibrium.rename("equilibrium")], axis=1).dropna()
    events: list[DivergenceEvent] = []

    in_event = False
    open_time = None
    open_deviation = 0.0
    peak_abs_deviation = 0.0
    direction = 0
    opened_during_rth = False

    for ts, row in frame.iterrows():
        dev = row.spread - row.equilibrium
        abs_dev = abs(dev)
        if not in_event:
            if abs_dev > tolerance:
                in_event = True
                open_time = ts
                open_deviation = abs_dev
                peak_abs_deviation = abs_dev
                direction = 1 if dev > 0 else -1
                opened_during_rth = rth_open <= ts.time() < rth_close
        else:
            if abs_dev > peak_abs_deviation:
                peak_abs_deviation = abs_dev
            closed = abs_dev <= tolerance or (dev > 0) != (direction == 1)
            if closed:
                # Reverted -- the opposite of what this trigger basis wants,
                # discarded regardless of how long it took.
                in_event = False

    # Only an event still open at the end of the series counts -- it opened
    # and never gave back its divergence within the data available.
    if in_event and opened_during_rth:
        last_ts = frame.index[-1]
        events.append(DivergenceEvent(
            open_time=open_time,
            close_time=last_ts,
            max_spread=peak_abs_deviation,
            duration=last_ts - open_time,
            max_adverse_excursion=peak_abs_deviation - open_deviation,
        ))

    return events


def reverted_after_peak(
    spread: pd.Series,
    equilibrium: pd.Series,
    tolerance: float = DEFAULT_DIVERGENCE_TOLERANCE,
) -> bool:
    """Per the client 2026-09-07 (Checklist_090926.pdf, entry condition A):
    "There is no 100% mean reversion after Maximum Divergence." True if,
    at any point AFTER the single largest |spread - equilibrium| point in
    this window (the "Maximum Divergence" the entry Trigger is based on),
    the spread crossed back inside the tolerance band around equilibrium --
    i.e. fully mean-reverted at least once since its own defining peak.

    Deliberately independent of detect_divergence_events/
    detect_non_reverting_rth_events above: this checks reversion relative
    to THE peak specifically (whatever point set the currently applicable
    Trigger), not any-and-all divergence events in the window. Ignores NaN
    (the equilibrium's own rolling-window warm-up)."""
    frame = pd.concat([spread.rename("spread"), equilibrium.rename("equilibrium")], axis=1).dropna()
    if frame.empty:
        return False
    deviation = frame["spread"] - frame["equilibrium"]
    peak_pos = int(deviation.abs().to_numpy().argmax())
    after_peak = deviation.iloc[peak_pos + 1:]
    return bool((after_peak.abs() <= tolerance).any())


def reverted_more_than_pct_after_peak(
    spread: pd.Series,
    equilibrium: pd.Series,
    pct: float = 0.5,
) -> bool:
    """Per the client 2026-09-13: "After it hit the maximum divergence, it
    should have at least >50% mean reversion." A DIFFERENT, distinct check
    from reverted_after_peak (Condition A, dormant) above -- confirmed by
    the client the same message: "There is no condition of 'the pair not
    having fully mean reverted since its max divergence'. This condition
    does not exist." That one looks for a FULL (100%, tolerance-band)
    reversion and disqualifies on finding one; this one looks for a
    PARTIAL (>pct, default 50%) reversion and is what has to be true for
    the Divergence Watch to show a pair as "cleared" before live data
    arrives (see write_status's pair_divergence, phase2/main.py).

    True if, at any point after the single largest |spread - equilibrium|
    point in this window (the same peak reverted_after_peak measures
    from), the deviation dropped to less than `pct` of the peak's own
    absolute value. Ignores NaN (the equilibrium's own rolling-window
    warm-up), same as reverted_after_peak."""
    frame = pd.concat([spread.rename("spread"), equilibrium.rename("equilibrium")], axis=1).dropna()
    if frame.empty:
        return False
    deviation = frame["spread"] - frame["equilibrium"]
    peak_pos = int(deviation.abs().to_numpy().argmax())
    peak_abs = abs(deviation.iloc[peak_pos])
    if peak_abs == 0:
        return False  # no real divergence to have reverted from
    after_peak = deviation.iloc[peak_pos + 1:]
    return bool((after_peak.abs() < pct * peak_abs).any())


# Removed on the client's instruction 2026-08-03: "we need to delete the
# condition of 2 events". Set back to 2 to reinstate it.
#
# What it did: took the mean of the three widest divergences and required at
# least two events to reach it, which rejected pairs whose widest gap was a
# lone spike. What removing it changes: a pair is selectable off a single
# qualifying event, and the trigger level is simply the maximum divergence.
# For a spiky pair that maximum IS the one-off, so the pair gets watched but
# the level is rarely reached -- this widens eligibility, it does not
# automatically produce more entries.
MIN_EVENTS_FOR_TRIGGER = 0


def widest_three_mean(events: list[DivergenceEvent]) -> float | None:
    """Mean of the three widest divergences -- the yardstick the selection
    condition measures against. None below three events."""
    if len(events) < 3:
        return None
    widest = sorted((e.max_spread for e in events), reverse=True)[:3]
    return sum(widest) / 3


def passes_divergence_condition(
    events: list[DivergenceEvent], min_count: int = MIN_EVENTS_FOR_TRIGGER,
) -> bool:
    """The spec's "2 Divergences Condition": take the mean of the three widest
    divergences, and require at least two events to have reached that mean and
    closed inside two hours.

    The count was 3 until the client revised it. Three is unreachable by
    construction: the mean of the widest three always sits above the third of
    them, and nothing outside the top three can exceed it, so at most two
    events can clear the bar (verified at 0 of 1431 pairs). Two is
    satisfiable, and it still does real work -- it rejects pairs whose widest
    divergence is a lone outlier, because there the mean is dragged up so far
    that only the outlier itself clears it (988 of 1431 pairs)."""
    if min_count <= 0:
        # Condition disabled: any pair that produced a qualifying divergence
        # event is selectable, and the trigger is its maximum divergence.
        return bool(events)
    level = widest_three_mean(events)
    if level is None:
        return False
    # An event sitting exactly ON the mean counts. Compare with a small
    # tolerance or binary floating point silently drops it: the widest three
    # of (0.9, 0.8, 0.7) average to 0.8000000000000002, and 0.8 >= that is
    # False, which would throw away one of the only two events that can ever
    # clear the bar and reject the pair outright.
    epsilon = abs(level) * 1e-9
    return sum(1 for e in events if e.max_spread >= level - epsilon) >= min_count


def divergence_trigger_level(
    events: list[DivergenceEvent], min_events: int = MIN_EVENTS_FOR_TRIGGER,
) -> float | None:
    """Entry trigger threshold: the single largest peak deviation among
    qualifying divergence events. Per spec, "Use maximum divergence. Not mean
    of maximum divergence" -- and deliberately not standard deviation either.

    Deliberately named for what it is rather than how it's computed: this
    has been the mean of all qualifying events, then the mean of the widest
    three, and is now the maximum.

    Two separate gates, easy to conflate: the LEVEL is the maximum, but the
    pair is only SELECTABLE once it satisfies the divergence condition (see
    passes_divergence_condition). Returns None otherwise, which makes the
    pair ineligible -- a single outlier isn't evidence of a repeatable
    pattern."""
    if not passes_divergence_condition(events, min_count=min_events):
        return None
    return max(e.max_spread for e in events)

# Allowed Divergence Gap, per the client's worked example 2026-08-04:
#
#     Allowed Gap = (0.5% x EquityWithLoanValue) / (Long + Short Leg Size)
#     D1 - D2 must be <= Allowed Gap
#
# where D1/D2 are the widest and second widest divergences in the lookback and
# "Long + Short Leg Size" is the two legs' share counts added together. His
# example: ELV 29,400 -> 147, over 820 shares -> 0.1793 allowed; BAC/JPM's
# 0.2027 - 0.1786 = 0.0241 passes.
#
# What it does: rejects a pair whose widest divergence stands too far above
# the next one, scaled so "too far" means a move worth more than 0.5% of the
# account. It restores, in account terms, the outlier protection the removed
# 2 Divergences Condition used to give.
# Raised from 0.5% to 1% per the client 2026-08-11: conditional on Divergence
# Watch, once properly filtered by this same gap test, showing fewer than 10
# pairs -- confirmed after that fix went live, so the client's own condition
# was met.
GAP_ELV_PCT = 0.01


def divergence_gap(events: list[DivergenceEvent]) -> float:
    """D1 - D2. Zero when there is only one event: with nothing to compare
    against there is no outlier to reject, and requiring a second event would
    quietly reinstate a minimum-count rule the client removed."""
    if len(events) < 2:
        return 0.0
    widest = sorted((e.max_spread for e in events), reverse=True)
    return widest[0] - widest[1]


def allowed_divergence_gap(equity_with_loan_value: float, long_plus_short_shares: float) -> float | None:
    """None when the leg size is unknown -- the test cannot be evaluated
    rather than silently passing."""
    if long_plus_short_shares <= 0:
        return None
    return (GAP_ELV_PCT * equity_with_loan_value) / long_plus_short_shares


def passes_gap_condition(events: list[DivergenceEvent], equity_with_loan_value: float,
                         long_plus_short_shares: float) -> bool:
    allowed = allowed_divergence_gap(equity_with_loan_value, long_plus_short_shares)
    if allowed is None:
        return False
    return divergence_gap(events) <= allowed
