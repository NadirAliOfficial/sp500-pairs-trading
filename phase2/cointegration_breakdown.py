"""Broken cointegration detection, per the client's 2026-08-23
"Cointegration Breakdown" document plus same-day chat corrections. "The 3
Core Detection Metrics for Broken Cointegration: All 3 must show Broken
Cointegration" -- PLUS a 4th, correlation, confirmed separately the same
day after an initial back-and-forth ("Since your new document's 3 metrics
don't include correlation at all, I'll treat this as a full replacement"
-> "No. Correlation is part of it. Broken Correlation is < 0.70"). All 4
below must show broken.

1 bar = 1 minute throughout, per the spec's own closing note.

  1. Correlation              rolling Pearson correlation < 0.70 = broken
                               (per the client 2026-08-23 chat, separate
                               from the document's own 3; the ORIGINAL
                               Entry Conditions PDF's 0.90 stays the
                               SCREENING threshold -- a different question
                               (is this pair a candidate at all) from this
                               one (has it broken down right now)).
  2. Half-Life Inflation      the pair's own frozen half-life (NOT a fixed
                              20 minutes -- per the client: "Half-Life is
                              specific for the pair... could be 20 minutes
                              or whatever value. So long as it is less
                              than 60 minutes. Now, if it is 20 minutes
                              and it has not mean reverted by 50% after 20
                              minutes, that half-life is broken"): has the
                              spread's deviation from equilibrium reverted
                              by at least 50% over a lookback of that many
                              minutes? If not, broken.
  3. Rolling ADF              Engle-Granger ADF over a rolling 60-bar
                              window; p-value > 0.10 = broken (not the
                              0.05 used for the original screening test).
  4. Kalman Innovation/Drift  a live Kalman beta estimate (diagnostic
                              only -- the pair's TRADED hedge ratio stays
                              frozen all session, per the client
                              2026-08-21) shows a sharp monotonic
                              structural shift while its innovation spikes
                              past 3 standard deviations of its own recent
                              baseline. Exact numbers for "sharp" and the
                              monotonic window aren't specified in the
                              source document -- BETA_MONOTONIC_WINDOW/
                              INNOVATION_SPIKE_SIGMA below are a
                              documented, reasonable default, flagged for
                              the client to correct if not what was meant.

Pure functions/classes only, no I/O -- fed live price/spread series by
main.py every check cycle. The Kalman monitor is replayed fresh from the
given price series each call rather than kept as persistent per-pair
state, since Kalman filtering is deterministic given the same inputs --
this lets the SAME function serve both the continuous (10am-till-now,
regular-hours) check and the 9:31am premarket-inclusive checkpoint
without needing to keep two monitors in sync, at a negligible CPU cost
(replaying up to ~480 bars is sub-millisecond)."""
import statistics
from dataclasses import dataclass, field

import pandas as pd

from phase1.screening import engle_granger_adf_pvalue, pearson_correlation

CORRELATION_BREAKDOWN_THRESHOLD = 0.70  # "Broken Correlation is < 0.70"
HALF_LIFE_REVERSION_FRACTION = 0.5  # "has not mean reverted by 50%"
ADF_ROLLING_BARS = 60
ADF_BREAKDOWN_PVALUE = 0.10
BETA_MONOTONIC_WINDOW = 20
BETA_DRIFT_MIN_SAMPLES = 30  # innovations needed before the baseline std is trustworthy
INNOVATION_SPIKE_SIGMA = 3.0
KALMAN_BASELINE_WINDOW = 60  # recent bars for the innovation baseline, matches ADF_ROLLING_BARS


def correlation_broken(price_a: pd.Series, price_b: pd.Series) -> bool | None:
    """None until there are at least 2 aligned points to correlate."""
    frame = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()
    if len(frame) < 2:
        return None
    return bool(pearson_correlation(frame["a"], frame["b"]) < CORRELATION_BREAKDOWN_THRESHOLD)


def half_life_reversion_broken(
    spread: pd.Series, equilibrium: pd.Series, half_life_minutes: float | None,
) -> bool | None:
    """None (not yet decidable) if half_life_minutes itself is unknown, or
    there isn't yet a full half-life-length lookback of data, or the
    deviation at the start of that lookback was already ~0 (nothing to
    measure reversion from -- comparing against a ~0 baseline would flag
    ordinary noise as "broken"). 1 bar = 1 minute, so the lookback is the
    half-life rounded to the nearest whole bar."""
    if half_life_minutes is None or half_life_minutes < 1:
        return None
    lookback = int(round(half_life_minutes))
    frame = pd.concat([spread.rename("s"), equilibrium.rename("e")], axis=1).dropna()
    if len(frame) <= lookback:
        return None
    deviation_start = abs(frame["s"].iloc[-lookback - 1] - frame["e"].iloc[-lookback - 1])
    deviation_now = abs(frame["s"].iloc[-1] - frame["e"].iloc[-1])
    if deviation_start == 0:
        return None
    reverted_fraction = 1 - (deviation_now / deviation_start)
    return bool(reverted_fraction < HALF_LIFE_REVERSION_FRACTION)


def rolling_adf_broken(spread: pd.Series) -> bool | None:
    """None until a full ADF_ROLLING_BARS window of spread data exists."""
    if len(spread) < ADF_ROLLING_BARS:
        return None
    p_value = engle_granger_adf_pvalue(spread.tail(ADF_ROLLING_BARS))
    return p_value > ADF_BREAKDOWN_PVALUE


@dataclass
class KalmanBetaEstimator:
    """1D random-walk Kalman filter for a pairs hedge ratio:
    price_a = beta * price_b + noise. Used here ONLY as a diagnostic input
    for cointegration-breakdown detection -- the pair's actual TRADED
    hedge ratio is frozen for the whole session (main.py's module
    docstring) and this estimate never feeds back into it.

    process_variance controls how fast beta is allowed to drift tick to
    tick; observation_variance is the assumed noise in the price
    relationship itself."""
    beta: float
    variance: float = 1.0
    process_variance: float = 1e-5
    observation_variance: float = 1e-3

    def update(self, price_a: float, price_b: float) -> float:
        predicted_variance = self.variance + self.process_variance

        innovation = price_a - self.beta * price_b
        innovation_variance = (price_b ** 2) * predicted_variance + self.observation_variance
        kalman_gain = predicted_variance * price_b / innovation_variance

        self.beta = self.beta + kalman_gain * innovation
        self.variance = (1 - kalman_gain * price_b) * predicted_variance
        return self.beta


@dataclass
class CointegrationKalmanMonitor:
    """Drives a KalmanBetaEstimator bar-by-bar and records the innovation
    (prediction error, using beta as it stood BEFORE that bar's update --
    the actual one-step-ahead prediction error) and the resulting beta at
    each step, then judges "broken" per the client's own description:
    "beta_t experiences a sharp, monotonic structural shift WHILE the
    innovation variance spikes past 3 sigma" (both conditions together,
    not either alone)."""
    estimator: KalmanBetaEstimator
    innovations: list[float] = field(default_factory=list)
    beta_history: list[float] = field(default_factory=list)

    def update(self, price_a: float, price_b: float) -> None:
        prev_beta = self.estimator.beta
        new_beta = self.estimator.update(price_a, price_b)
        self.innovations.append(price_a - prev_beta * price_b)
        self.beta_history.append(new_beta)

    def is_broken(self) -> bool:
        if len(self.innovations) < BETA_DRIFT_MIN_SAMPLES + BETA_MONOTONIC_WINDOW:
            return False
        recent_betas = self.beta_history[-BETA_MONOTONIC_WINDOW:]
        diffs = [b - a for a, b in zip(recent_betas, recent_betas[1:])]
        monotonic = bool(diffs) and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs))
        if not monotonic:
            return False
        # Rolling, not expanding -- Finding #12 from the independent audit,
        # confirmed real: an expanding-window std computed from ALL
        # innovations since 9:30 never forgets an early spike (a 10am news
        # print, say), which permanently inflates the baseline and
        # desensitizes the 3-sigma check to every later spike for the rest
        # of the session, including ones the client's own "sharp... spikes
        # past 3 sigma" language clearly means to catch.
        recent = self.innovations[-(KALMAN_BASELINE_WINDOW + 1):-1]
        baseline_std = statistics.pstdev(recent)
        if baseline_std == 0:
            return False
        return abs(self.innovations[-1]) > INNOVATION_SPIKE_SIGMA * baseline_std


def run_kalman_monitor(price_a: pd.Series, price_b: pd.Series, seed_beta: float) -> CointegrationKalmanMonitor:
    """Replays a fresh Kalman filter over the given price series from
    scratch, seeded at the pair's frozen trading beta -- see the module
    docstring for why this is replayed rather than kept as running state."""
    monitor = CointegrationKalmanMonitor(estimator=KalmanBetaEstimator(beta=seed_beta))
    frame = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()
    for _, row in frame.iterrows():
        monitor.update(row["a"], row["b"])
    return monitor


def is_cointegration_broken(
    spread: pd.Series, equilibrium: pd.Series, price_a: pd.Series, price_b: pd.Series,
    half_life_minutes: float | None, seed_beta: float,
) -> bool:
    """The composite gate: all 4 metrics (correlation, half-life inflation,
    rolling ADF, Kalman innovation/drift) must show broken. A metric that
    isn't computable yet (None -- not enough data) counts as NOT broken
    for that metric, same "absence of confirmation isn't confirmation"
    convention used throughout this project -- incomplete data should
    never itself trigger an exit-relevant signal."""
    corr_result = correlation_broken(price_a, price_b)
    half_life_result = half_life_reversion_broken(spread, equilibrium, half_life_minutes)
    adf_result = rolling_adf_broken(spread)
    kalman_broken = run_kalman_monitor(price_a, price_b, seed_beta).is_broken()
    return bool(corr_result) and bool(half_life_result) and bool(adf_result) and kalman_broken
