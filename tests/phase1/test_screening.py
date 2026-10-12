"""Validates each statistical function against synthetic series with known
properties, before trusting the pipeline on real S&P 500 data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from phase1.screening import (
    MIN_HURST, PairResult,
    pearson_correlation, pearson_correlation_1day, bar_to_bar_returns, engle_granger_hedge_ratio_and_spread,
    engle_granger_adf_pvalue, johansen_confirmed, half_life_of_mean_reversion,
    hurst_exponent, volatility_is_stable, liquidity_is_high, market_cap_is_high, evaluate_pair,
)

np.random.seed(42)
N = 1000


def make_cointegrated_pair():
    """B = random walk. A = hedge_ratio*B + stationary mean-reverting noise.
    This IS a genuinely cointegrated pair by construction."""
    b = pd.Series(np.cumsum(np.random.normal(0, 1, N)) + 100)
    noise = pd.Series(np.zeros(N))
    for i in range(1, N):
        noise[i] = 0.7 * noise[i-1] + np.random.normal(0, 0.5)  # AR(1), mean-reverting
    a = 1.5 * b + noise
    return a, b


def make_independent_pair():
    """Two independent random walks -- should NOT be cointegrated."""
    a = pd.Series(np.cumsum(np.random.normal(0, 1, N)) + 100)
    b = pd.Series(np.cumsum(np.random.normal(0, 1, N)) + 50)
    return a, b


def test_pearson_correlation_perfect():
    a = pd.Series(np.arange(100, dtype=float))
    b = a * 2 + 5
    assert abs(pearson_correlation(a, b) - 1.0) < 1e-9
    print("PASS: pearson_correlation on perfectly linear series = 1.0")


def test_bar_to_bar_returns_hand_computed_example():
    """Per the client 2026-09-07 (1_Day_Pearson_60_Bars.pdf, Method 1):
    R_t = (P_t - P_{t-1}) / P_{t-1}. NaN on the first bar."""
    price = pd.Series([100.0, 110.0, 99.0])
    returns = bar_to_bar_returns(price)
    assert pd.isna(returns.iloc[0])
    assert abs(returns.iloc[1] - 0.10) < 1e-9
    assert abs(returns.iloc[2] - (-0.10)) < 1e-9
    print("PASS: bar_to_bar_returns matches R_t = (P_t - P_t-1) / P_t-1 exactly")


def test_pearson_correlation_1day_perfect_correlation():
    """Per the client 2026-09-07 (1_Day_Pearson_60_Bars.pdf, superseding
    the earlier 3pm-anchor version the same day): correlates each leg's
    OWN bar-to-bar returns against the other's, over whatever window is
    handed in -- no anchor, no cross-day alignment. B's returns here are
    exactly 2x A's every bar -> perfectly linear -> r=1.0."""
    price_a = pd.Series([100.0] + [0.0] * 9)
    returns_a = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02, -0.01, 0.005, 0.008]
    for i, r in enumerate(returns_a):
        price_a.iloc[i + 1] = price_a.iloc[i] * (1 + r)
    price_b = pd.Series([50.0] + [0.0] * 9)
    for i, r in enumerate(returns_a):
        price_b.iloc[i + 1] = price_b.iloc[i] * (1 + 2 * r)  # B's return is always 2x A's

    r = pearson_correlation_1day(price_a, price_b)
    assert r is not None
    assert abs(r - 1.0) < 1e-9
    print("PASS: pearson_correlation_1day is 1.0 when one leg's bar-to-bar returns are a fixed multiple of the other's")


def test_pearson_correlation_1day_none_with_fewer_than_two_return_pairs():
    """A single bar has no prior bar to diff against (bar_to_bar_returns'
    first value is always NaN), and two bars give exactly one return --
    still not enough to correlate. Fail-closed (None), same convention as
    the codebase's other live-window gates below MIN_BARS_FOR_LIVE_*."""
    assert pearson_correlation_1day(pd.Series([100.0]), pd.Series([50.0])) is None
    assert pearson_correlation_1day(pd.Series([100.0, 101.0]), pd.Series([50.0, 50.5])) is None
    print("PASS: pearson_correlation_1day returns None with fewer than 2 valid return pairs")


def test_normalized_hedge_ratio_matches_the_clients_ko_pep_example():
    """Per the client 2026-09-09: "KO / PEP Example (beta = 1.35). To make
    the short leg equal exactly 1 share, divide both sides of the ratio by
    1.35 -> 1 : 0.7407." The leg whose beta exceeds 1 becomes the reference,
    fixed at exactly 1."""
    from phase1.screening import normalized_hedge_ratio
    reference, other = normalized_hedge_ratio(1.35)
    assert reference == "B"
    assert abs(other - 0.7407) < 1e-4
    print("PASS: normalized_hedge_ratio reproduces the client's KO/PEP 1 : 0.7407 example")


def test_normalized_hedge_ratio_leaves_a_below_one_beta_on_leg_a():
    """beta <= 1 means leg A is already the >= leg, so A stays the reference
    at 1 and B keeps its own sub-1 ratio -- nothing to divide."""
    from phase1.screening import normalized_hedge_ratio
    reference, other = normalized_hedge_ratio(0.74)
    assert reference == "A"
    assert abs(other - 0.74) < 1e-9
    print("PASS: a hedge ratio already <= 1 leaves leg A as the reference, unscaled")


def test_normalized_hedge_ratio_at_exactly_one():
    from phase1.screening import normalized_hedge_ratio
    reference, other = normalized_hedge_ratio(1.0)
    assert reference == "A"
    assert abs(other - 1.0) < 1e-9
    print("PASS: a hedge ratio of exactly 1.0 needs no normalization either way")


def test_engle_granger_detects_real_cointegration():
    a, b = make_cointegrated_pair()
    hedge_ratio, spread = engle_granger_hedge_ratio_and_spread(a, b)
    pvalue = engle_granger_adf_pvalue(spread)
    assert abs(hedge_ratio - 1.5) < 0.1, f"hedge ratio should recover ~1.5, got {hedge_ratio}"
    assert pvalue < 0.05, f"genuinely cointegrated pair should have low ADF p-value, got {pvalue}"
    print(f"PASS: Engle-Granger correctly detects real cointegration (hedge_ratio={hedge_ratio:.3f}, p={pvalue:.4f})")


def test_engle_granger_rejects_independent_walks():
    a, b = make_independent_pair()
    _, spread = engle_granger_hedge_ratio_and_spread(a, b)
    pvalue = engle_granger_adf_pvalue(spread)
    assert pvalue > 0.05, f"independent random walks should NOT show cointegration, got p={pvalue}"
    print(f"PASS: Engle-Granger correctly rejects independent random walks (p={pvalue:.4f})")


def test_johansen_confirms_real_cointegration():
    a, b = make_cointegrated_pair()
    assert johansen_confirmed(a, b) == True
    print("PASS: Johansen test confirms genuinely cointegrated pair")


def test_johansen_rejects_independent_walks():
    a, b = make_independent_pair()
    assert johansen_confirmed(a, b) == False
    print("PASS: Johansen test rejects independent random walks")


def test_half_life_recovers_known_ar1_decay():
    # spread_t = phi * spread_{t-1} + noise, phi=0.9 -> known half-life = ln(2)/ln(1/0.9) ~= 6.58 periods
    phi = 0.9
    spread = np.zeros(2000)
    for i in range(1, 2000):
        spread[i] = phi * spread[i-1] + np.random.normal(0, 1)
    spread = pd.Series(spread)
    hl = half_life_of_mean_reversion(spread)
    expected = np.log(2) / np.log(1/phi)
    assert hl is not None
    assert abs(hl - expected) / expected < 0.25, f"expected ~{expected:.2f}, got {hl:.2f}"
    print(f"PASS: half-life recovers known AR(1) decay (expected~{expected:.2f}, got {hl:.2f})")


def test_half_life_none_or_impractically_large_for_random_walk():
    # A true random walk has lambda=0 in expectation, but a finite sample's
    # OLS estimate has sampling noise around 0 -- it can come out slightly
    # negative by chance, giving a technically-finite but practically
    # meaningless half-life (years, not minutes/days). Either None or a huge
    # value is correct; what matters is it clears the pipeline's real
    # MAX_HALF_LIFE_MINUTES=60 threshold, which either result does.
    from phase1.screening import MAX_HALF_LIFE_MINUTES
    rw = pd.Series(np.cumsum(np.random.normal(0, 1, 1000)))
    hl = half_life_of_mean_reversion(rw)
    assert hl is None or hl > MAX_HALF_LIFE_MINUTES * 5, \
        f"random walk half-life should be None or impractically large, got {hl}"
    print(f"PASS: half-life for a random walk is None or impractically large (got {hl}) -- correctly fails the <60 threshold either way")


def test_hurst_random_walk_near_half():
    rw = pd.Series(np.cumsum(np.random.normal(0, 1, 5000)))
    h = hurst_exponent(rw)
    assert 0.35 < h < 0.65, f"random walk Hurst should be near 0.5, got {h}"
    print(f"PASS: Hurst exponent for random walk near 0.5 (got {h:.3f})")


def test_hurst_mean_reverting_below_half():
    phi = 0.5
    spread = np.zeros(5000)
    for i in range(1, 5000):
        spread[i] = phi * spread[i-1] + np.random.normal(0, 1)
    h = hurst_exponent(pd.Series(spread))
    assert h < 0.5, f"strongly mean-reverting series should have Hurst < 0.5, got {h}"
    print(f"PASS: Hurst exponent for mean-reverting series < 0.5 (got {h:.3f})")


def test_volatility_is_stable_passes_a_flat_trend():
    np.random.seed(1)
    spread = pd.Series(np.random.normal(0, 1, 200))
    assert volatility_is_stable(spread) is True
    print("PASS: a flat/stable volatility trend qualifies")


def test_volatility_is_stable_passes_a_strongly_expanding_trend():
    # Per the client 2026-08-25: "Spread volatility: Stable or Expanding" --
    # an expanding trend, even a large one, must still qualify (no upper
    # bound), consistent with the divergence-EXPANSION strategy.
    np.random.seed(2)
    n = 200
    scale = np.linspace(0.2, 5.0, n)  # amplitude grows 25x over the sample
    spread = pd.Series(np.random.normal(0, 1, n) * scale)
    assert volatility_is_stable(spread) is True
    print("PASS: a strongly expanding volatility trend qualifies, no upper bound")


def test_volatility_is_stable_fails_a_contracting_trend():
    np.random.seed(3)
    n = 45
    scale = np.linspace(20.0, 0.01, n)  # amplitude collapses ~2000x over the sample
    spread = pd.Series(np.random.normal(0, 1, n) * scale)
    assert volatility_is_stable(spread) is False
    print("PASS: a contracting volatility trend fails -- the relationship is collapsing")


def test_liquidity_threshold_share_volume_alone_qualifies():
    # 15M shares clears the 10M ADV bar even with dollar volume far below $1B
    assert liquidity_is_high(15_000_000, 100_000_000, 15_000_000, 100_000_000) == True
    print("PASS: ADV > 10M shares alone satisfies the liquidity test (OR, not AND)")


def test_liquidity_threshold_dollar_volume_alone_qualifies():
    # 2M shares fails ADV, but $1.5B dollar volume alone still clears the OR
    assert liquidity_is_high(2_000_000, 1_500_000_000, 2_000_000, 1_500_000_000) == True
    print("PASS: dollar volume > $1B/day alone satisfies the liquidity test (OR, not AND)")


def test_liquidity_threshold_fails_when_both_metrics_are_low():
    assert liquidity_is_high(1_000_000, 100_000_000, 1_000_000, 100_000_000) == False
    print("PASS: liquidity fails when neither ADV nor dollar volume clears its bar")


def test_liquidity_threshold_requires_both_legs_to_pass():
    # leg A clears easily, leg B clears neither metric -- pair should still fail
    assert liquidity_is_high(50_000_000, 5_000_000_000, 1_000_000, 100_000_000) == False
    print("PASS: liquidity requires BOTH legs to individually pass, not just one")


def test_market_cap_threshold():
    assert market_cap_is_high(25_000_000_000, 30_000_000_000) == True
    assert market_cap_is_high(10_000_000_000, 30_000_000_000) == False
    print("PASS: market cap threshold ($20B) correctly applied to both legs")


def test_full_evaluate_pair_on_cointegrated_series():
    a, b = make_cointegrated_pair()
    result = evaluate_pair(
        "TEST_A", "TEST_B", "Test Industry", a, b, a, b,
        avg_share_volume_a=15_000_000, avg_dollar_volume_a=2_000_000_000, market_cap_a=25_000_000_000,
        avg_share_volume_b=15_000_000, avg_dollar_volume_b=2_000_000_000, market_cap_b=25_000_000_000,
    )
    assert result.cointegrated
    assert result.johansen_confirmed
    assert result.liquidity_ok
    assert result.market_cap_ok
    print(f"PASS: full evaluate_pair pipeline runs end-to-end, qualified={result.qualified}")


def test_evaluate_pair_uses_daily_series_for_correlation_not_intraday():
    # correlation/correlation_recent must come from daily_a/daily_b, not
    # price_a/price_b -- construct a pair where the two series disagree and
    # check the result reflects the DAILY series' correlation.
    intraday_a, intraday_b = make_independent_pair()  # near-zero correlation
    daily_a = pd.Series(np.arange(60, dtype=float))
    daily_b = daily_a * 2 + 5  # perfectly correlated
    result = evaluate_pair(
        "TEST_A", "TEST_B", "Test Industry", intraday_a, intraday_b, daily_a, daily_b,
        avg_share_volume_a=15_000_000, avg_dollar_volume_a=2_000_000_000, market_cap_a=25_000_000_000,
        avg_share_volume_b=15_000_000, avg_dollar_volume_b=2_000_000_000, market_cap_b=25_000_000_000,
    )
    assert abs(result.correlation - 1.0) < 1e-9
    assert abs(result.correlation_recent - 1.0) < 1e-9
    print("PASS: evaluate_pair computes correlation/correlation_recent from the daily series, not intraday")


def _result(**overrides):
    """A PairResult with every gate passing by default, so each test only
    has to override the one field it's actually checking."""
    defaults = dict(
        symbol_a="A", symbol_b="B", industry="Test", correlation=0.95,
        # BROKEN DOWN (< 0.70) -- per the client 2026-09-08 the 20-day
        # recent correlation is a breakdown gate, not a floor.
        correlation_recent=0.50,
        adf_pvalue=0.01, cointegrated=True, johansen_confirmed=False,
        half_life=30.0, hurst=0.70, volatility_stable=True,
        liquidity_ok=True, market_cap_ok=True, hedge_ratio=1.0,
    )
    defaults.update(overrides)
    return PairResult(**defaults)


def test_qualified_ignores_hurst():
    # Per the client 2026-08-29: "there is nothing offline, that's phase by
    # phase was only for payments, not code wise, all things should be
    # wired, nothing should be offline" -- Hurst is a point-in-time
    # technical snapshot, not the kind of structural property (correlation,
    # liquidity, market cap) this offline-data qualification step should
    # gate on. Removed as a gate here; still exclusively a live,
    # continuously-rechecked-during-RTH gate in phase2/main.py
    # (PairContext.hurst_ever_above_threshold).
    assert _result(hurst=MIN_HURST + 0.01).qualified is True
    assert _result(hurst=MIN_HURST).qualified is True  # used to fail the exact boundary, no longer does
    assert _result(hurst=0.40).qualified is True  # used to fail as mean-reverting, no longer does
    assert _result(hurst=None).qualified is True  # used to fail outright (None check), no longer does
    print("PASS: qualified no longer depends on hurst at all")


def test_qualified_ignores_johansen_confirmed():
    # Same 2026-08-29 removal as hurst above -- Johansen is now exclusively
    # a live gate (PairContext.johansen_ever_unconfirmed).
    assert _result(johansen_confirmed=False).qualified is True
    assert _result(johansen_confirmed=True).qualified is True  # used to fail, no longer does
    print("PASS: qualified no longer depends on johansen_confirmed at all")


def test_qualified_ignores_cointegrated_engle_granger_adf():
    # Per the client 2026-08-25: "Delete: Engle-Granger ADF" -- no longer a
    # qualification gate, either value must leave an otherwise-passing
    # pair qualified. Still computed and kept on PairResult for
    # visibility (see the two asserts on result.cointegrated in the full
    # pipeline test above), just not read by this property anymore.
    assert _result(cointegrated=True).qualified is True
    assert _result(cointegrated=False).qualified is True
    print("PASS: qualified no longer depends on cointegrated (Engle-Granger ADF) at all")


def test_qualified_ignores_half_life():
    # Per the client 2026-08-25: "Delete: Half-life" -- no longer a
    # qualification gate here (Phase 1 screening only; Phase 2's own live
    # half_life_reversion_broken check is untouched, not named in this
    # request). Still computed and kept on PairResult for visibility.
    assert _result(half_life=20.0).qualified is True  # was already passing under the old < 60 gate
    assert _result(half_life=90.0).qualified is True  # used to fail the old gate, no longer does
    assert _result(half_life=None).qualified is True  # used to fail outright (None check), no longer does
    print("PASS: qualified no longer depends on half_life at all")


def test_qualified_requires_recent_correlation_to_have_broken_down():
    """Per the client 2026-09-08: "Your Pearson 20 Days says >= 0.70. It
    should say < 0.70." The 20 day correlation must have BROKEN DOWN for a
    pair to qualify -- the opposite of what this gate did before today --
    while the 60 day one (>=0.90, tested separately) must still be high."""
    from phase1.screening import MIN_RECENT_CORRELATION
    assert _result(correlation_recent=0.60).qualified is True
    assert _result(correlation_recent=MIN_RECENT_CORRELATION - 0.01).qualified is True
    assert _result(correlation_recent=MIN_RECENT_CORRELATION).qualified is False  # boundary is exclusive (<)
    assert _result(correlation_recent=0.80).qualified is False
    print("PASS: qualified requires the recent (20 day) correlation to have broken down below 0.70")


def test_qualified_still_requires_every_other_gate():
    assert _result().qualified is True  # sanity check on the all-passing default
    assert _result(correlation=0.5).qualified is False
    assert _result(volatility_stable=False).qualified is False
    assert _result(liquidity_ok=False).qualified is False
    assert _result(market_cap_ok=False).qualified is False
    print("PASS: every other qualification gate still applies unchanged")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
