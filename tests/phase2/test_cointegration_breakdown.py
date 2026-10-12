"""Unit tests for cointegration_breakdown.py -- pure functions/classes,
no I/O. Synthetic series with known properties, same approach
tests/phase1/test_screening.py uses for its own ADF tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from phase2.cointegration_breakdown import (
    ADF_ROLLING_BARS, BETA_DRIFT_MIN_SAMPLES, BETA_MONOTONIC_WINDOW, KALMAN_BASELINE_WINDOW,
    CointegrationKalmanMonitor, KalmanBetaEstimator, correlation_broken,
    half_life_reversion_broken, is_cointegration_broken, rolling_adf_broken, run_kalman_monitor,
)

np.random.seed(42)


# ---------------------------------------------------------------------------
# half_life_reversion_broken
# ---------------------------------------------------------------------------

def _flat_deviation_series(n: int, start_deviation: float, final_deviation: float) -> tuple[pd.Series, pd.Series]:
    """spread/equilibrium pair where equilibrium is flat 0 and the spread
    (== deviation) ramps linearly from start_deviation down/up to
    final_deviation over the series -- deviation n-1 bars ago is exactly
    start_deviation, deviation now (last bar) is exactly final_deviation."""
    values = np.linspace(start_deviation, final_deviation, n)
    spread = pd.Series(values)
    equilibrium = pd.Series(0.0, index=spread.index)
    return spread, equilibrium


def test_half_life_reversion_broken_when_deviation_barely_shrinks():
    # 21 bars: deviation goes from 1.0 (21 bars ago... use lookback=20) to
    # 0.9 now -- only reverted 10%, well short of 50%.
    spread, equilibrium = _flat_deviation_series(21, start_deviation=1.0, final_deviation=0.9)
    result = half_life_reversion_broken(spread, equilibrium, half_life_minutes=20)
    assert result is True
    print("PASS: broken when the deviation has reverted far less than 50% over its own half-life")


def test_half_life_not_broken_when_deviation_reverts_past_50_percent():
    spread, equilibrium = _flat_deviation_series(21, start_deviation=1.0, final_deviation=0.3)
    result = half_life_reversion_broken(spread, equilibrium, half_life_minutes=20)
    assert result is False
    print("PASS: not broken when the deviation has reverted more than 50%")


def test_half_life_reversion_exact_50_percent_boundary_counts_as_not_broken():
    spread, equilibrium = _flat_deviation_series(21, start_deviation=1.0, final_deviation=0.5)
    result = half_life_reversion_broken(spread, equilibrium, half_life_minutes=20)
    assert result is False
    print("PASS: reverting exactly 50% counts as reverted, not broken")


def test_half_life_reversion_none_when_half_life_is_none():
    spread, equilibrium = _flat_deviation_series(21, 1.0, 0.5)
    assert half_life_reversion_broken(spread, equilibrium, half_life_minutes=None) is None
    print("PASS: None half-life yields None (not decidable), not a crash")


def test_half_life_reversion_none_without_enough_lookback_data():
    # Only 10 bars, but half_life_minutes=20 needs a 20-bar lookback
    spread, equilibrium = _flat_deviation_series(10, 1.0, 0.5)
    assert half_life_reversion_broken(spread, equilibrium, half_life_minutes=20) is None
    print("PASS: None when there isn't yet a full half-life-length lookback")


def test_half_life_reversion_none_when_starting_deviation_is_zero():
    spread, equilibrium = _flat_deviation_series(21, 0.0, 0.01)
    assert half_life_reversion_broken(spread, equilibrium, half_life_minutes=20) is None
    print("PASS: None when the starting deviation is ~0 -- nothing to measure reversion from")


# ---------------------------------------------------------------------------
# rolling_adf_broken
# ---------------------------------------------------------------------------

def _mean_reverting_spread(n: int, phi: float = 0.5) -> pd.Series:
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + np.random.normal(0, 1)
    return pd.Series(values)


def _random_walk_spread(n: int) -> pd.Series:
    return pd.Series(np.cumsum(np.random.normal(0, 1, n)))


def test_rolling_adf_not_broken_for_a_mean_reverting_spread():
    spread = _mean_reverting_spread(ADF_ROLLING_BARS)
    assert rolling_adf_broken(spread) is False
    print("PASS: a genuinely mean-reverting spread reads as not broken")


def test_rolling_adf_broken_for_a_random_walk_spread():
    spread = _random_walk_spread(ADF_ROLLING_BARS)
    assert rolling_adf_broken(spread) is True
    print("PASS: a random-walk (non-stationary) spread reads as broken")


def test_rolling_adf_none_without_a_full_window():
    spread = _mean_reverting_spread(ADF_ROLLING_BARS - 1)
    assert rolling_adf_broken(spread) is None
    print("PASS: None until a full 60-bar window exists")


# ---------------------------------------------------------------------------
# correlation_broken
# ---------------------------------------------------------------------------

def test_correlation_not_broken_when_highly_correlated():
    a = pd.Series(np.arange(100, dtype=float))
    b = a * 2 + 5
    assert correlation_broken(a, b) is False
    print("PASS: a near-perfectly correlated pair reads as not broken")


def test_correlation_broken_below_070():
    np.random.seed(1)
    a = pd.Series(np.random.normal(0, 1, 200))
    b = pd.Series(np.random.normal(0, 1, 200))  # independent -- correlation near 0
    assert correlation_broken(a, b) is True
    print("PASS: two independent series (correlation well below 0.70) read as broken")


def test_correlation_broken_none_without_enough_points():
    assert correlation_broken(pd.Series([1.0]), pd.Series([2.0])) is None
    print("PASS: None with fewer than 2 aligned points")


# ---------------------------------------------------------------------------
# CointegrationKalmanMonitor
# ---------------------------------------------------------------------------

def _feed(monitor: CointegrationKalmanMonitor, price_a_seq, price_b_seq) -> None:
    for a, b in zip(price_a_seq, price_b_seq):
        monitor.update(a, b)


def test_kalman_monitor_not_broken_with_too_few_samples():
    monitor = CointegrationKalmanMonitor(estimator=KalmanBetaEstimator(beta=2.0))
    n = BETA_DRIFT_MIN_SAMPLES + BETA_MONOTONIC_WINDOW - 1
    _feed(monitor, [100.0] * n, [50.0] * n)
    assert monitor.is_broken() is False
    print("PASS: not broken while there aren't yet enough samples to judge a baseline")


def test_kalman_monitor_not_broken_for_a_stable_relationship():
    monitor = CointegrationKalmanMonitor(estimator=KalmanBetaEstimator(beta=2.0))
    n = BETA_DRIFT_MIN_SAMPLES + BETA_MONOTONIC_WINDOW + 20
    price_b = 50.0 + np.cumsum(np.random.normal(0, 0.1, n))
    price_a = 2.0 * price_b + np.random.normal(0, 0.05, n)
    _feed(monitor, price_a, price_b)
    assert monitor.is_broken() is False
    print("PASS: a stable, noisy-but-consistent relationship never flags broken")


def test_kalman_monitor_not_broken_when_monotonic_without_a_variance_spike():
    """Beta creeps steadily but innovations stay small and consistent --
    monotonic alone, without the innovation spike, must not trigger."""
    monitor = CointegrationKalmanMonitor(estimator=KalmanBetaEstimator(beta=2.0, process_variance=1e-3))
    n = BETA_DRIFT_MIN_SAMPLES + BETA_MONOTONIC_WINDOW + 10
    true_beta = np.linspace(2.0, 2.0, n)  # flat true beta
    price_b = 50.0 + np.arange(n) * 0.01
    price_a = true_beta * price_b  # near-zero innovation throughout
    _feed(monitor, price_a, price_b)
    assert monitor.is_broken() is False
    print("PASS: monotonic beta drift alone, without an innovation spike, does not confirm broken")


def test_kalman_monitor_broken_with_a_real_structural_break():
    """Directly constructs the monitor's internal history rather than
    fighting the real Kalman filter's own convergence speed (which
    depends on process_variance/observation_variance in ways not worth
    coupling this test to): a monotonically ramping beta over the last
    BETA_MONOTONIC_WINDOW steps, with a final innovation far outside the
    baseline established by the rest -- exactly the "sharp, monotonic
    structural shift while the innovation variance spikes past 3 sigma"
    scenario the client described."""
    baseline_n = BETA_DRIFT_MIN_SAMPLES + 10
    monitor = CointegrationKalmanMonitor(
        estimator=KalmanBetaEstimator(beta=2.0),
        innovations=[0.01 if i % 2 == 0 else -0.01 for i in range(baseline_n)],
        beta_history=[2.0] * baseline_n,
    )
    assert monitor.is_broken() is False

    # Monotonically ramping beta across the window, ending with one huge
    # innovation (far past 3 sigma of the ~0.01 baseline).
    ramp = list(np.linspace(2.0, 5.0, BETA_MONOTONIC_WINDOW))
    monitor.beta_history.extend(ramp)
    monitor.innovations.extend([0.01] * (BETA_MONOTONIC_WINDOW - 1) + [10.0])
    assert monitor.is_broken() is True
    print("PASS: a monotonic beta ramp plus an innovation spike together confirm broken")


def test_kalman_monitor_baseline_is_rolling_not_expanding_since_session_open():
    """Finding #12 from the independent audit, confirmed real: an EXPANDING
    baseline (all innovations since 9:30) never forgets an early one-off
    spike -- e.g. a 10am news print -- which permanently inflates the std
    and can mask a genuine later structural break for the rest of the
    session. This constructs exactly that: one huge early outlier, then
    KALMAN_BASELINE_WINDOW+ quiet bars (long enough for a rolling window to
    fully roll past the outlier), then a real monotonic ramp with a spike
    that a rolling recent-window baseline correctly flags."""
    baseline_n = BETA_DRIFT_MIN_SAMPLES + 10
    quiet_n = KALMAN_BASELINE_WINDOW + 10
    monitor = CointegrationKalmanMonitor(
        estimator=KalmanBetaEstimator(beta=2.0),
        innovations=(
            [0.01 if i % 2 == 0 else -0.01 for i in range(baseline_n)]  # normal baseline
            + [50.0]  # one huge early outlier -- an old news-driven spike
            + [0.01 if i % 2 == 0 else -0.01 for i in range(quiet_n)]   # back to normal, long enough to roll past it
        ),
        beta_history=[2.0] * (baseline_n + 1 + quiet_n),
    )
    assert monitor.is_broken() is False

    # A real later spike -- clearly above the CURRENT quiet regime, but far
    # too small to clear 3-sigma of a baseline still dragged up by the old
    # 50.0 outlier.
    ramp = list(np.linspace(2.0, 5.0, BETA_MONOTONIC_WINDOW))
    monitor.beta_history.extend(ramp)
    monitor.innovations.extend([0.01] * (BETA_MONOTONIC_WINDOW - 1) + [5.0])
    assert monitor.is_broken() is True
    print("PASS: a real later spike is caught even though one big early outlier is in the innovation history")


# ---------------------------------------------------------------------------
# run_kalman_monitor / is_cointegration_broken (composite)
# ---------------------------------------------------------------------------

def test_run_kalman_monitor_replays_over_a_price_series():
    n = 50
    price_a = pd.Series(2.0 * (50.0 + np.arange(n) * 0.1))
    price_b = pd.Series(50.0 + np.arange(n) * 0.1)
    monitor = run_kalman_monitor(price_a, price_b, seed_beta=2.0)
    assert len(monitor.innovations) == n
    assert len(monitor.beta_history) == n
    print("PASS: run_kalman_monitor produces one innovation/beta per aligned price pair")


def test_is_cointegration_broken_requires_all_four_metrics():
    """A spread/price series built to be genuinely mean-reverting,
    stationary, and highly correlated (correlation not broken, ADF not
    broken, half-life not broken) must not read as broken overall even
    though the Kalman piece is never in a "confirmed broken" state
    either -- confirms the AND across all 4, not an OR."""
    n = 120
    spread = _mean_reverting_spread(n, phi=0.3)
    equilibrium = pd.Series(0.0, index=spread.index)
    price_b = pd.Series(50.0 + np.cumsum(np.random.normal(0, 0.05, n)))
    price_a = pd.Series(2.0 * price_b + np.random.normal(0, 0.02, n))
    assert is_cointegration_broken(spread, equilibrium, price_a, price_b,
                                    half_life_minutes=10, seed_beta=2.0) is False
    print("PASS: a genuinely intact relationship never reads as broken overall")


def test_is_cointegration_broken_false_when_only_correlation_is_broken():
    """Per the client: "Correlation is part of it" -- but ALL 4 must show
    broken, so decoupled prices (correlation broken) alone, with an
    otherwise intact ADF/half-life picture, must not flip the overall
    result. Reuses the mean-reverting spread from the ADF fixture but
    pairs it with independent, uncorrelated price series."""
    n = 120
    spread = _mean_reverting_spread(n, phi=0.3)
    equilibrium = pd.Series(0.0, index=spread.index)
    np.random.seed(7)
    price_a = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, n)))
    price_b = pd.Series(50.0 + np.cumsum(np.random.normal(0, 1, n)))  # independent of price_a
    assert is_cointegration_broken(spread, equilibrium, price_a, price_b,
                                    half_life_minutes=10, seed_beta=2.0) is False
    print("PASS: broken correlation alone (without the other 3 also broken) doesn't confirm overall breakdown")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
