"""Statistical test suite for candidate pairs, per the client's spec table:

    Same industry               Required
    Pearson correlation         >= 0.90 (60 day daily, consolidated tape)
    Pearson Correl. Breakdown   >= 0.70 (20 day daily, consolidated tape)
    Engle-Granger ADF           removed (2026-08-25, see below)
    Johansen test                removed from HERE 2026-08-29, see below
    Half-life                    removed (2026-08-25, see below)
    Hurst exponent                removed from HERE 2026-08-29, see below
    Spread volatility             Stable
    ADV or Dollar Volume         ADV > 10M shares OR dollar volume > $1B/day
    Market cap                   > $20 billion

Johansen and Hurst removed from `qualified` 2026-08-29, per the client:
"there is nothing offline, that's phase by phase was only for payments, not
code wise, all things should be wired, nothing should be offline" -- both
are inherently point-in-time technical snapshots, not the kind of
structural, slow-moving property (same industry, correlation, liquidity,
market cap) that belongs in a qualification step meant to run once against
the previous session's closed data. They're now EXCLUSIVELY a live,
continuously-rechecked-during-RTH gate (phase2/main.py's
update_live_johansen_hurst, PairContext.johansen_ever_unconfirmed/
hurst_ever_above_threshold, each validated independently, sticky once
true for the session -- see that field's own comment). Confirmed live the
same day this was found: re-running the (now-removed) offline screen fresh
qualified 0 of 1295 S&P 500 candidate pairs under the OLD combined gate,
because even a pair with excellent correlation (BAC/JPM, 0.98) had a real
live Hurst of 0.48 at that moment -- exactly the kind of point-in-time
reading that shouldn't be able to permanently disqualify a pair from ever
being selected. johansen_confirmed/hurst are still computed and kept on
this dataclass for visibility (reports, logging), just not read by
`qualified` anymore.

Per Correct_Filters.pdf (2026-08-25), Pearson correlation and the new
Correlation Breakdown test both run on daily closes from the true
consolidated tape (EQUS.SUMMARY via
databento_source.fetch_consolidated_daily_closes), NOT the shared 1-minute
intraday series everything else here uses -- a 60 day window for the
original correlation test, a 20 day window (the same series' tail) for
Correlation Breakdown. Johansen, Hurst, half-life, volatility and hedge_ratio
still run on the shared intraday price_a/price_b passed into evaluate_pair,
unchanged for now -- the client's "RTH Consolidated tape, only" requirement
for THAT leg is still open pending a reply (this data plan has no
minute-level consolidated dataset, only daily; a message was sent 2026-08-25
asking whether the primary-exchange merge already in use is acceptable
there, or a different data plan is needed).

Four related changes 2026-08-25, all per the client, all pointing the
same direction: pairs should be trending, not mean-reverting, consistent
with the divergence-EXPANSION strategy this bot has traded since
2026-08-18 (enters before the trigger, profits when the spread keeps
moving FURTHER, not when it reverts). The old thresholds below were all
mean-reversion-era leftovers never updated to match that strategy:

  - Hurst exponent: "H >0.60, instead of < 0.50" -- flipped from
    mean-reverting to trending. (Removed from `qualified` entirely
    2026-08-29, see the top of this docstring -- this flip is kept as
    history since MIN_HURST/hurst_exponent are still used by phase2's live
    gate, just no longer read here.)
  - Engle-Granger ADF: "Delete." No longer a qualification gate.
    adf_pvalue/cointegrated are still computed and kept on PairResult for
    visibility, just no longer read by the qualified property -- the
    underlying OLS regression (engle_granger_hedge_ratio_and_spread) stays
    exactly as it was, since hedge_ratio/spread from it still feed
    everything else (half-life, Hurst, volatility, and the live bot's own
    frozen entry beta).
  - Johansen test: "Unconfirmed, instead of Confirmed" -- inverted at the
    time. (Removed from `qualified` entirely 2026-08-29, same as Hurst
    above -- johansen_confirmed as a FIELD NAME is unchanged, still
    computed and kept on PairResult, just no longer read here.)
  - Half-life: "Delete." Same treatment as Engle-Granger ADF -- no longer
    a qualification gate here. half_life is still computed and kept on
    PairResult for visibility. Scoped deliberately to THIS qualification
    test, not Phase 2's own separate, still-active use of half-life (the
    live bot's cointegration_breakdown.py half_life_reversion_broken
    check, part of is_cointegration_broken's 4-metric composite, and
    MAX_HOLD_MINUTES/Maximum Hold Time, which is an unrelated concept
    entirely) -- those weren't named in this request.

A fifth, unrelated change from the same PDF: a NEW gate, Pearson
Correlation Breakdown, requiring the 20 day daily correlation stay >= 0.70
even though the 60 day figure already clears 0.90 -- catches a pair whose
relationship has been drifting apart recently but hasn't dragged the longer
window down yet.

"Same industry" is handled upstream by sp500_universe.industry_groups(), which
only forms pairs within the same GICS sub-industry.

One threshold in the spec still has no numeric definition ("Stable"
volatility) -- HEDGE_STABILITY_MAX_SLOPE_PCT is a reasonable documented
default, easy to tune if the client specifies a number later. The liquidity
and market cap thresholds below are the client's own explicit numbers
(added 2026-07-28, replacing the earlier $5M/day placeholder default).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

MIN_CORRELATION = 0.90
MIN_RECENT_CORRELATION = 0.70  # "Pearson Correlation Breakdown" gate, added 2026-08-25 per Correct_Filters.pdf
MAX_ADF_PVALUE = 0.05
JOHANSEN_SIGNIFICANCE = 1  # column index into johansen critical values: 0=90%, 1=95%, 2=99%
# "Half-life < 60 minutes", measured on the 1-minute bars the spec calls for,
# so the unit here really is minutes. This previously ran on DAILY bars over a
# 2-year window with the 60 read as days -- a leftover from when Phase 1 was
# built against yfinance (daily closes only) that survived the switch to
# Databento. Daily cointegration over two years and intraday mean reversion
# are different properties, and the pair list differed materially between them.
MAX_HALF_LIFE_MINUTES = 60
# Lowered from 0.60 to 0.50 per the client 2026-09-01, after a live day
# (2026-09-01) confirmed against real data that no pair crossed 0.60 all
# session: "Hurst and Johansen is only one check... Today Hurst is >
# 0.50." Same live, continuous MIN_HURST used everywhere -- not a second
# threshold or a separate prior-session check, just this one number
# changed. See update_live_johansen_hurst (phase2/main.py) for the one
# live check this feeds.
#
# Lowered to 0.40 per the client 2026-09-09: "Use Hurst > 0.40." Only 3 of
# 37 tracked pairs cleared 0.50 that day (max reading 0.5296 20-day,
# 0.4953 1-day), which had been leaving the Divergence Watch effectively
# empty.
MIN_HURST = 0.40
HEDGE_STABILITY_MAX_SLOPE_PCT = 0.02  # rolling-vol trend must not contract more than -2% of its mean per period (no cap on expansion, see volatility_is_stable)
MIN_AVG_SHARE_VOLUME = 5_000_000  # "ADV > 5 million shares" -- lowered from 10M per client spec v31, 2026-08-05
MIN_AVG_DOLLAR_VOLUME = 500_000_000  # "Dollar volume > $500 million/day" -- lowered from $1B, OR'd with share volume, not AND'd
MIN_MARKET_CAP = 20_000_000_000  # "Market cap > $20 billion"


@dataclass
class PairResult:
    symbol_a: str
    symbol_b: str
    industry: str
    correlation: float
    correlation_recent: float
    adf_pvalue: float
    cointegrated: bool
    johansen_confirmed: bool
    half_life: float | None
    hurst: float | None
    volatility_stable: bool
    liquidity_ok: bool
    market_cap_ok: bool
    hedge_ratio: float | None

    @property
    def qualified(self) -> bool:
        # cointegrated (Engle-Granger ADF) and half_life removed as gates
        # 2026-08-25; johansen_confirmed and hurst removed as gates
        # 2026-08-29 (see the module docstring's top section) -- all four
        # still computed and kept on this dataclass for visibility, just
        # not read here anymore. Johansen/Hurst now live exclusively as a
        # continuous, live-only gate in phase2/main.py.
        return (
            self.correlation >= MIN_CORRELATION
            # BREAKDOWN, not a floor -- per the client 2026-09-08: "Your
            # Pearson 20 Days says >= 0.70. It should say < 0.70." Matches
            # the checklist's own wording ("Pearson correlation Breakdown
            # <0.70, Time Range: 20 Days") and the strategy's premise
            # ("filter off co-integration, and filter in expansion"): the
            # 60-day correlation above must still be high (>=0.90, the pair
            # is a real pair), while the RECENT 20-day one must have broken
            # down. This read >= for its whole life before today, which
            # selected still-correlated pairs -- the opposite set.
            and self.correlation_recent < MIN_RECENT_CORRELATION
            and self.volatility_stable
            and self.liquidity_ok
            and self.market_cap_ok
        )


def pearson_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    return float(series_a.corr(series_b))


def bar_to_bar_returns(price: pd.Series) -> pd.Series:
    """R_t = (P_t - P_{t-1}) / P_{t-1} for each consecutive pair of bars --
    per the client 2026-09-07 (1_Day_Pearson_60_Bars.pdf, "Method 1: The
    Rolling Intraday Bar-to-Bar Approach"), superseding the earlier
    3pm-anchor cumulative-return version the same day: "Do not compress
    yesterday into a single number... Extract 1-minute percentage
    returns." NaN on the first bar (nothing to diff against yet) --
    pandas' own pct_change, spelled out per the client's explicit R_t
    formula for direct traceability, same convention this module has
    used for every other client-specified formula."""
    return price.pct_change()


def pearson_correlation_1day(price_a: pd.Series, price_b: pd.Series) -> float | None:
    """1 Day Pearson -- correlates leg A's own bar-to-bar 1-minute return
    series against leg B's, over whatever window is handed in. Per the
    client's worked example (1_Day_Pearson_60_Bars.pdf): "Today Live"
    correlates MA's and V's OWN returns within today's rolling 60-bar
    window (0.12, reaching back into premarket to fill out 60 bars near
    the open); "Yesterday Anchor" does the same over yesterday's own
    15:00-16:00 window (0.88) as a purely informational baseline -- the
    actual entry test only reads today's live value against the 0.70
    floor ("Correlation Test: r_live < 0.70"). The caller picks the
    window (main.py's ctx.premarket_price_a/b, trailing 60 bars); this
    function doesn't care which session the prices are from.

    Returns None if there are fewer than 2 valid (non-NaN, aligned) return
    pairs to correlate -- not enough data yet, same fail-closed shape as
    every other live gate in this codebase."""
    returns_a = bar_to_bar_returns(price_a).rename("a")
    returns_b = bar_to_bar_returns(price_b).rename("b")
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    if len(aligned) < 2:
        return None
    return pearson_correlation(aligned["a"], aligned["b"])


def engle_granger_hedge_ratio_and_spread(series_a: pd.Series, series_b: pd.Series) -> tuple[float, pd.Series]:
    """OLS hedge ratio (regress A on B) and the resulting spread series."""
    x = np.column_stack([np.ones(len(series_b)), series_b.values])
    beta = np.linalg.lstsq(x, series_a.values, rcond=None)[0]
    hedge_ratio = beta[1]
    spread = series_a - hedge_ratio * series_b
    return float(hedge_ratio), spread


def normalized_hedge_ratio(hedge_ratio: float) -> tuple[str, float]:
    """Deterministic Beta Normalization -- per the client 2026-09-04, restated
    2026-09-08 and again 2026-09-09: "Find out which Leg has beta > 1. Make
    that Leg beta 1 by dividing both legs with its beta. The Leg with beta 1
    is the reference Leg."

    In this codebase's convention (Spread = P_A - hedge_ratio * P_B) leg A's
    coefficient is always 1 and leg B's is hedge_ratio, so the pair's two
    betas are (1.0, hedge_ratio) and only one of them can exceed 1. Dividing
    both by beta_max = max(1.0, hedge_ratio) therefore fixes the REFERENCE
    leg at exactly 1.0 and leaves the other at <= 1.0, which is what the
    client's KO/PEP example does by hand (beta 1.35 -> divide both by 1.35 ->
    1 : 0.7407).

    Returns (reference_leg, scaled_other_leg_ratio) where reference_leg is
    "A" or "B" -- the leg held at exactly 1 share. Per the client the same
    day, the reason this matters: "there are no fractional shares for a short
    leg", so the integer side has to be the reference, not an accident of
    which symbol happened to be regressed on which.

    Note this is a RATIO normalization, not a share count. Fractional Shares
    = False for execution, so the actual per-leg quantities are still whole
    numbers (see position_sizing) -- this only fixes which leg the ratio is
    expressed relative to."""
    beta_max = max(1.0, hedge_ratio)
    if beta_max <= 0:
        return "A", hedge_ratio
    if hedge_ratio > 1.0:
        # Leg B is the >1 leg, so B becomes the reference at 1.0 and A scales.
        return "B", 1.0 / hedge_ratio
    return "A", hedge_ratio


def engle_granger_adf_pvalue(spread: pd.Series) -> float:
    """ADF test on the OLS residual spread. p < 0.05 => stationary => cointegrated."""
    result = adfuller(spread.values, autolag="AIC")
    return float(result[1])


def johansen_confirmed(price_a: pd.Series, price_b: pd.Series) -> bool:
    """Trace statistic vs critical value at JOHANSEN_SIGNIFICANCE (95% by default).
    Confirmed if the trace stat for rank>=1 exceeds its critical value, i.e. at
    least one cointegrating relationship exists between the two price series."""
    data = np.column_stack([price_a.values, price_b.values])
    result = coint_johansen(data, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1[0]
    critical_value = result.cvt[0][JOHANSEN_SIGNIFICANCE]
    return bool(trace_stat > critical_value)


def half_life_of_mean_reversion(spread: pd.Series) -> float | None:
    """Fits the spread to an Ornstein-Uhlenbeck (AR(1)) process:
    delta_spread_t = lambda * spread_{t-1} + noise
    half_life = -ln(2) / lambda, only meaningful when lambda < 0 (mean-reverting).
    Returns None if the spread is not mean-reverting at all (lambda >= 0)."""
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lagged = lagged.loc[delta.index]
    x = np.column_stack([np.ones(len(lagged)), lagged.values])
    beta = np.linalg.lstsq(x, delta.values, rcond=None)[0]
    lam = beta[1]
    if lam >= 0:
        return None
    return float(-np.log(2) / lam)


def hurst_exponent(series: pd.Series, max_lag: int = 40) -> float:
    """Variance-based Hurst estimate: for a self-similar process,
    std(X[t+lag] - X[t]) ~ lag^H, so regressing log(std of lagged
    differences) on log(lag) recovers H directly as the slope (a random walk
    has Var(diff) ~ lag, i.e. std ~ lag^0.5, giving slope=H=0.5 -- no factor
    of 2 needed; multiplying by 2 here was a bug caught by the random-walk
    test case, which gave H=1.0 instead of the expected ~0.5).
    H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending."""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series.values[lag:], series.values[:-lag])) for lag in lags]
    tau = [t if t > 0 else 1e-10 for t in tau]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0])


def volatility_is_stable(spread: pd.Series, window: int = 20) -> bool:
    """Per the client 2026-08-25: "Spread volatility: Stable or Expanding" --
    a CONTRACTING volatility trend (the spread's swings shrinking over the
    sample) suggests the relationship is collapsing back toward equilibrium
    rather than diverging, working against this bot's divergence-EXPANSION
    strategy. An expanding trend, even a large one, is consistent with that
    strategy and qualifies with no upper bound -- only a trend meaningfully
    more negative than -HEDGE_STABILITY_MAX_SLOPE_PCT per period fails.
    Previously failed on ANY strong trend regardless of direction (using
    abs(slope)), which wrongly rejected genuinely expanding pairs -- exactly
    the kind this strategy wants."""
    rolling_std = spread.rolling(window).std().dropna()
    if len(rolling_std) < window:
        return False
    x = np.arange(len(rolling_std))
    slope, intercept = np.polyfit(x, rolling_std.values, 1)
    if intercept == 0:
        return False
    relative_slope_per_period = slope / abs(intercept)
    return bool(relative_slope_per_period > -HEDGE_STABILITY_MAX_SLOPE_PCT)


def _single_symbol_liquidity_ok(avg_daily_share_volume: float, avg_daily_dollar_volume: float) -> bool:
    """Per spec: ADV > 10M shares OR dollar volume > $1B/day -- an OR
    between the two volume metrics for ONE symbol, not between the two
    legs of the pair (both legs still individually need to pass)."""
    return avg_daily_share_volume >= MIN_AVG_SHARE_VOLUME or avg_daily_dollar_volume >= MIN_AVG_DOLLAR_VOLUME


def liquidity_is_high(
    avg_daily_share_volume_a: float, avg_daily_dollar_volume_a: float,
    avg_daily_share_volume_b: float, avg_daily_dollar_volume_b: float,
) -> bool:
    return (
        _single_symbol_liquidity_ok(avg_daily_share_volume_a, avg_daily_dollar_volume_a)
        and _single_symbol_liquidity_ok(avg_daily_share_volume_b, avg_daily_dollar_volume_b)
    )


def market_cap_is_high(market_cap_a: float, market_cap_b: float) -> bool:
    return market_cap_a >= MIN_MARKET_CAP and market_cap_b >= MIN_MARKET_CAP


RECENT_CORRELATION_WINDOW_DAYS = 20  # Correlation Breakdown test's window, sliced from the tail of daily_a/daily_b


def evaluate_pair(
    symbol_a: str, symbol_b: str, industry: str,
    price_a: pd.Series, price_b: pd.Series,
    daily_a: pd.Series, daily_b: pd.Series,
    avg_share_volume_a: float, avg_dollar_volume_a: float, market_cap_a: float,
    avg_share_volume_b: float, avg_dollar_volume_b: float, market_cap_b: float,
    size_filters_exempt: bool = False,
) -> PairResult:
    """`daily_a`/`daily_b`: 60+ days of daily closes on the consolidated
    tape, for Pearson correlation and Correlation Breakdown. `price_a`/
    `price_b`: the shared intraday series, for every other test."""
    correlation = pearson_correlation(daily_a, daily_b)
    correlation_recent = pearson_correlation(
        daily_a.tail(RECENT_CORRELATION_WINDOW_DAYS), daily_b.tail(RECENT_CORRELATION_WINDOW_DAYS))
    hedge_ratio, spread = engle_granger_hedge_ratio_and_spread(price_a, price_b)
    adf_pvalue = engle_granger_adf_pvalue(spread)
    cointegrated = adf_pvalue < MAX_ADF_PVALUE
    johansen_ok = johansen_confirmed(price_a, price_b)
    hl = half_life_of_mean_reversion(spread)
    hurst = hurst_exponent(spread)
    vol_stable = volatility_is_stable(spread)
    # Special Pairs List: exempt from Market Cap, Daily Shares Volume and
    # Daily Dollar Volume only. Every statistical test above still applies --
    # the client asked to skip all conditions for these, then reverted to
    # "we need the filters", so the exemption stays scoped to the three size
    # filters the spec names. PEP/KO is the case that makes this matter: it
    # fails correlation at 0.41, which no size exemption can rescue.
    liquidity_ok = size_filters_exempt or liquidity_is_high(
        avg_share_volume_a, avg_dollar_volume_a, avg_share_volume_b, avg_dollar_volume_b)
    market_cap_ok = size_filters_exempt or market_cap_is_high(market_cap_a, market_cap_b)

    return PairResult(
        symbol_a=symbol_a, symbol_b=symbol_b, industry=industry,
        correlation=round(correlation, 4),
        correlation_recent=round(correlation_recent, 4),
        adf_pvalue=round(adf_pvalue, 6),
        cointegrated=cointegrated,
        johansen_confirmed=johansen_ok,
        half_life=round(hl, 2) if hl is not None else None,
        hurst=round(hurst, 4),
        volatility_stable=vol_stable,
        liquidity_ok=liquidity_ok,
        market_cap_ok=market_cap_ok,
        hedge_ratio=round(hedge_ratio, 4),
    )
