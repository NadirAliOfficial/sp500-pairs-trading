"""Continuous, whole-universe pair screening for Phase 2.

Runs the same statistical tests phase1/screening.py's evaluate_pair defines
(correlation, Engle-Granger, liquidity, market cap -- see that module for
which of these actually gate qualification), on a repeating schedule inside
the live bot itself, so a newly-qualifying pair can be picked up and traded
the same day. Per the client's 2026-08-05 instruction, the Market Entry List
is additional ideas fed into this same screen, not a gate on what the screen
is allowed to consider. Per the client 2026-08-29 ("nothing should be
offline, all things should be wired"), this live screen -- not a hand-run
batch script -- is now the ONLY qualification path; phase1/screen_pairs.py
(the old offline script) is gone, and phase2/main.py's own 9:30am pair
selection (select_pairs_from_previous_session) reuses the exact same
evaluate_pair logic this module does, just scoped to the previous RTH
session and the Market Entry List's own pairs instead of the full universe.

Reference data (S&P 500 membership, market cap, average volume) barely moves
intraday, so build_reference_data() is meant to be called once per session
and reused across every screening cycle. Only the intraday price history --
which is what actually decides whether a pair newly qualifies today -- gets
refetched every cycle, via the same EQUS.MINI live-plan dataset phase2's
warm-up already uses (see phase2/live_data.py's comment on why: the
XNAS.ITCH/XNYS.PILLAR datasets the old offline screen preferred for accuracy
need Databento's $1500/mo Plus plan for same-day data, and accuracy here
only drives which pairs get a look today, not any offline export)."""
import itertools
import logging

import pandas as pd

from phase1.data_source import fetch_daily_closes, fetch_market_caps_yfinance, fetch_volume_stats_yfinance
from phase1.screening import PairResult, evaluate_pair
from phase1.sp500_universe import fetch_sp500_universe, industry_groups
from phase2.trading_lists import is_special_pair

logger = logging.getLogger("pairs_bot.universe_screening")

# ADF, Johansen and the Hurst regression (still computed on PairResult even
# though Johansen/Hurst no longer gate qualified, see phase1/screening.py)
# produce numbers on fewer bars, just not trustworthy ones -- same floor
# phase2/main.py's select_pairs_from_previous_session also uses.
MIN_BARS_TO_TEST = 300
MIN_DAILY_BARS_TO_TEST = 45  # same floor as select_pairs_from_previous_session's daily leg


class UniverseReferenceData:
    """Session-cached inputs a screening cycle needs but doesn't need to
    refetch every cycle."""

    def __init__(self, groups: dict[str, list[str]], market_caps: dict[str, float],
                 volume_stats: dict[str, dict[str, float]], daily_closes: pd.DataFrame):
        self.groups = groups
        self.market_caps = market_caps
        self.volume_stats = volume_stats
        # 3mo of daily closes (below), already fetched once per session for
        # volume stats -- reused as-is for the Pearson / Correlation
        # Breakdown tests (60/20 day) added 2026-08-25, rather than paying
        # for a second daily fetch every session for the same data.
        self.daily_closes = daily_closes
        self.all_symbols = sorted({s for symbols in groups.values() for s in symbols})
        self.industry_by_symbol = {s: industry for industry, symbols in groups.items() for s in symbols}


def build_reference_data() -> UniverseReferenceData:
    """Blocking (Wikipedia scrape + yfinance) -- call via asyncio.to_thread.
    Meant to run once per session, not once per screening cycle."""
    universe = fetch_sp500_universe()
    groups = industry_groups(universe)
    all_symbols = sorted({s for symbols in groups.values() for s in symbols})
    logger.info(f"Universe screening reference data: {len(all_symbols)} symbols, "
                f"{len(groups)} sub-industries")
    market_caps = fetch_market_caps_yfinance(all_symbols)
    # fetch_volume_stats_yfinance needs a closes frame for its avg-price term
    # -- a 3-month daily average is plenty accurate for this and matches the
    # 3-month window fetch_volume_stats_yfinance's own volume pull already
    # uses, rather than defaulting to fetch_daily_closes's normal 2-year pull.
    # ~63 trading days in 3 months, comfortably covers the 60/20 day windows
    # Pearson correlation / Correlation Breakdown need.
    daily_closes = fetch_daily_closes(all_symbols, period="3mo")
    volume_stats = fetch_volume_stats_yfinance(all_symbols, daily_closes)
    return UniverseReferenceData(groups, market_caps, volume_stats, daily_closes)


def candidate_pairs(ref: UniverseReferenceData,
                     market_entry_pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """Same-industry combinations (the spec's "same industry" test) unioned
    with the Market Entry List, per the client's instruction that the list
    is additional ideas to screen, not a filter on what can trade."""
    pairs = {tuple(sorted((a, b)))
             for symbols in ref.groups.values()
             for a, b in itertools.combinations(sorted(symbols), 2)}
    for a, b in market_entry_pairs:
        pairs.add(tuple(sorted((a, b))))
    return pairs


def screen_once(closes: pd.DataFrame, ref: UniverseReferenceData,
                 market_entry_pairs: list[tuple[str, str]],
                 already_tracked: set[frozenset], permanent_blacklisted: set[str]) -> list[PairResult]:
    """Pure/CPU-bound (correlation + ADF + Johansen + Hurst per candidate) --
    call via asyncio.to_thread so it never blocks the live trading loop.

    Only tests pairs not already tracked -- an already-tracked pair's ongoing
    eligibility is handled every check cycle by refresh_pair_statistics in
    main.py, this only needs to find pairs that aren't being watched yet.

    `permanent_blacklisted` is the client's own manually curated never-trade
    list ONLY, not the 24-hour news exclusions -- per the client 2026-08-26,
    a symbol in the news should still be discoverable/trackable ("it should
    get in... the news could be the reason for the expansion"), news only
    matters as an exit (News Stop) once a position is already open."""
    results = []
    for a, b in candidate_pairs(ref, market_entry_pairs):
        if frozenset((a, b)) in already_tracked:
            continue
        if a in permanent_blacklisted or b in permanent_blacklisted:
            continue
        if a not in closes.columns or b not in closes.columns:
            continue
        both = closes[[a, b]].dropna()
        if len(both) < MIN_BARS_TO_TEST:
            continue
        if a not in ref.daily_closes.columns or b not in ref.daily_closes.columns:
            continue
        both_daily = ref.daily_closes[[a, b]].dropna()
        if len(both_daily) < MIN_DAILY_BARS_TO_TEST:
            continue
        stats_a = ref.volume_stats.get(a, {})
        stats_b = ref.volume_stats.get(b, {})
        industry = ref.industry_by_symbol.get(a, ref.industry_by_symbol.get(b, "Market Entry List"))
        try:
            result = evaluate_pair(
                a, b, industry, both[a], both[b], both_daily[a], both_daily[b],
                avg_share_volume_a=stats_a.get("avg_share_volume", 0.0),
                avg_dollar_volume_a=stats_a.get("avg_dollar_volume", 0.0),
                market_cap_a=ref.market_caps.get(a, 0.0),
                avg_share_volume_b=stats_b.get("avg_share_volume", 0.0),
                avg_dollar_volume_b=stats_b.get("avg_dollar_volume", 0.0),
                market_cap_b=ref.market_caps.get(b, 0.0),
                size_filters_exempt=is_special_pair(a, b),
            )
        except Exception as e:
            logger.debug(f"universe screen skip {a}/{b}: {e}")
            continue
        if result.qualified:
            results.append(result)
    return results
