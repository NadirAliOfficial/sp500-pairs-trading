"""Phase 2 orchestrator -- the daily live trading session, per the client's spec:

    8:25am ET  reqTradingHours() check via SPY proxy -- skip the entire day if
               today is an early close (closeTime < 4:00pm ET)
    9:25am ET  Gemini news blacklist scan, plus a second EquityWithLoanValue
               snapshot frozen specifically for the stop loss (2% of THIS
               figure, fixed for the day -- separate from the 9:29am
               snapshot below, which sizes position quantity instead)
    9:29am ET  freeze EquityWithLoanValue for the day
    9:30am ET  morning warm-up seeds each pair's live price/spread history
               (previous RTH + today so far, for trading) and establishes
               the FROZEN base trigger level and half-life from the
               previous RTH session ALONE (see establish_base_trigger) --
               live monitoring loop starts immediately after, but
               entry_eligible() blocks any actual trade until 10:00am ET
               (client's final decision after going back and forth: v7
               added a 10am floor, v8 walked it back as a typo, then
               explicitly reinstated it -- entry window is 10:00am-10:30am
               ET, narrowed from 3:30pm 2026-08-20/21)
    9:31-10:30 recalculate_trigger_exceptions runs every cycle (not just
               once at 9:31): tests whether correlation/cointegration broke
               over a premarket-inclusive window, and if so, whether a
               larger non-reverting divergence over that same wider window
               should replace the base trigger as "the exception" -- see
               that function's own docstring.
    9:30-10:30 in addition to the entry/exit checks every 1 minute, an
               intraday Gemini/GPT news re-scan fires every 1 minute, asking
               about news "today" rather than "before today's market open" --
               catches intraday catalysts, not just pre-market ones, using the
               same blacklist as the 9:25am scan. Keeps running past 10:30am
               for as long as a position might still be open, purely to catch
               a NEWS STOP -- see risk_manager for that exit condition.
    9:00-11:30 SEC EDGAR 8-K monitor (sec_edgar_monitor.py) polls SEC's own
               real-time filings feed directly, every 1 minute, alongside
               the Gemini/GPT scan -- per the client 2026-08-23: "Supplement
               Gemini / ChatGPT news scans with direct SEC EDGAR RSS feed
               monitoring for 8-K filings... to bypass web index latency."
               A second, independent source writing into the same
               news_flagged_symbols.json raw signal (see news_blacklist.
               NEWS_FLAGGED_PATH) -- not excluded_symbols.json, which since
               2026-08-28 only records symbols that actually caused a News
               Stop exit.
    10:00am+   update_live_cointegration_stats re-verifies correlation/ADF
               every cycle over a rolling 30-bar window -- entry_eligible
               requires cointegration to have broken before 9:31am AND to
               still be broken right now ("10am till Now ensures broken
               co-integration right until entry. Not just till 10am.")
    3:59pm ET  flatten every open position
    4:00pm ET  post-trade report emailed to POST_TRADE_REPORT_TO

hedge_ratio is a frozen value for the whole session, per the client
2026-08-21: "Beta is calculated at 4:00 PM ET prior day... Beta MUST
remain frozen for the entire trade session today." It's whatever Phase 1's
screening (or, for a pair universe screening discovers live, that
discovery's own regression) computed from historical data through the
prior close -- nothing in this module mutates it intraday. There used to
be a live tick-level Kalman filter keeping it moving continuously, feeding
a Beta Destruction Stop Loss that flattened a position if its beta drifted
too far from entry -- both removed 2026-08-22 per the client ("Beta
Destruction Stop Loss is the only thing that needs to be removed... if
Kalman tracking is no longer useful, remove it"). Frozen beta makes the
live estimate pointless for its original purpose; the underlying tick
feed itself comes back in a later phase of this same update, repurposed
for Volume Delta Skew instead (RVOL was also built the same phase, then
dropped a day later per the client 2026-08-23: "Do not use RVOL. Since we
have Delta Skew.").

One important design note: qualified_pairs.csv's half_life column is
Phase 1's DAILY-bar half-life (in days -- see screening.py's comment on
MAX_HALF_LIFE_MINUTES), which is a completely different statistical quantity
from the intraday half-life the "Half-Life < 60 minutes" screening
condition needs (minutes, from 1-minute spread data). Reusing the daily
figure directly would produce a nonsensical screen, so half-life is
recomputed fresh every morning from the 1-minute warm-up spread using
screening.py's already-tested half_life_of_mean_reversion() -- the same
function, just fed intraday data, so its output is naturally in minutes
(1-min bars).

This module needs IBKR account/VPS credentials and SMTP credentials the
client hasn't provided yet -- it is written and structurally complete, but
running run_daily_session() for real requires IBKR_HOST/PORT/CLIENT_ID env
vars and (for the report) SMTP_HOST/PORT/USER/PASS. Not unit tested here
(pure orchestration/glue); every function it calls is unit tested in its own
module (spread_engine, risk_manager, position_sizing, news_blacklist,
post_trade_report, live_data, ibkr_client).
"""
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import timezone, datetime, time as dtime, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pairs_bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pairs_bot.main")

from phase1.screening import (MAX_HALF_LIFE_MINUTES, MIN_HURST, MIN_RECENT_CORRELATION,
                               engle_granger_adf_pvalue,
                               engle_granger_hedge_ratio_and_spread,
                               evaluate_pair, half_life_of_mean_reversion, hurst_exponent,
                               johansen_confirmed, normalized_hedge_ratio, pearson_correlation,
                               pearson_correlation_1day)
from phase1.data_source import fetch_market_caps_yfinance, fetch_volume_stats_yfinance
from phase2.ibkr_client import IBKRClient
from phase2.live_data import LIVE_DATASETS, LiveBarFeed, fetch_intraday_warmup
from phase2.market_entry_list import load_market_entry_list
from phase2.news_blacklist import (INTRADAY_SCAN_INTERVAL, NEWS_FLAGGED_PATH, ExcludedSymbols,
                                   MarketHalt, run_daily_scan, run_intraday_scan, run_market_wide_scan)
from phase2.sec_edgar_monitor import (MONITOR_WINDOW_START as EDGAR_WINDOW_START,
                                      MONITOR_WINDOW_END as EDGAR_WINDOW_END,
                                      SecEdgarMonitorState, run_edgar_scan)
from phase2.trading_lists import is_special_pair, load_permanent_blacklist, special_pairs_as_listed
from phase2.position_sizing import (MAX_LOSS_PCT, beta_adjusted_position_size, combo_ratio_and_units,
                                    compute_final_single_stock_quantity, compute_quantity,
                                    equity_balance_after_leg, opposite_leg_quantity,
                                     compute_single_stock_quantity, effective_sizing_equity, fractional_leg_ratio)
from phase2.post_trade_report import ReportEvent, build_report_html, send_report
from phase2.risk_manager import (
    EARLIEST_ENTRY_TIME, ENTRY_TRIGGER_DISTANCE_MIN_PCT, LATEST_ENTRY_TIME, MAX_ENTRY_DIVERGENCE,
    MAX_HOLD_MINUTES, MAX_STOCK_PRICE, MIN_ENTRY_DIVERGENCE, TRAILING_TAKE_PROFIT_AMOUNT,
    DailyState, PairSignal, bracket_order_prices, check_exit, entry_eligible,
    attach_hedge_leg, is_early_close_day, leg_still_open_after_bracket_fill, load_daily_state, news_stop_cancel, open_position, open_single_leg_position,
    record_exit, reentry_allowed, resting_hard_stop_price, resting_take_profit_price,
    save_daily_state, select_closest_to_trigger, select_highest_adjusted_spread_efficiency,
    save_pair_trigger_state, load_pair_trigger_state,
    stop_loss_double_touch_exit, stop_loss_triple_touch_exit, take_profit_hard_stop_exit,
    pnl_stop_loss_price, trade_direction, unrealized_pnl, unrealized_pnl_pct, wide_hard_stop_price,
)
from phase2 import universe_screening
from phase2.spread_engine import (DEFAULT_DIVERGENCE_TOLERANCE, compute_spread, detect_divergence_events,
                                  detect_non_reverting_rth_events, equilibrium_spread, passes_gap_condition,
                                  divergence_gap, allowed_divergence_gap, reverted_after_peak,
                                  reverted_more_than_pct_after_peak)
from phase2.cointegration_breakdown import KalmanBetaEstimator, is_cointegration_broken

ET = ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATUS_PATH = DATA_DIR / "bot_status.json"
# Today's risk counters, so a mid-session restart can't reset them -- see
# risk_manager.load_daily_state/save_daily_state.
DAILY_STATE_PATH = DATA_DIR / "daily_state.json"
# Per-pair trigger_reached_today, persisted separately -- see
# risk_manager.save_pair_trigger_state/load_pair_trigger_state's own
# comment for the 2026-09-14 incident that made this necessary.
PAIR_TRIGGER_STATE_PATH = DATA_DIR / "pair_trigger_state.json"
STOP_FLAG_PATH = DATA_DIR / "bot_stop.txt"
# Per the client 2026-09-01: "There is no pairs selection today... If
# from 9am, it has already loaded them all, by 9:30am, it will test for
# Johansen and Hurst. Which is a matter of seconds." Written once, at the
# end of each session, by cache_qualified_pairs_for_tomorrow -- see its
# own docstring and load_cached_qualified_pairs, which reads it back the
# next morning.
CACHED_QUALIFIED_PAIRS_PATH = DATA_DIR / "cached_qualified_pairs.json"
# Separate from STATUS_PATH deliberately -- see write_cached_pairs_status's
# own comment for why merging into bot_status.json doesn't work (the
# dashboard's 30s staleness TTL on that file would hide this the moment
# the bot process actually exits for the night).
CACHED_PAIRS_STATUS_PATH = DATA_DIR / "cached_pairs_status.json"
# Safety buffer on top of IBKR's own whatIf margin estimate for a real entry.
# Confirmed live 2026-08-11 and again 2026-08-12: even a whatIf probed at the
# MARGIN_ESTIMATE_SAFETY_BUFFER_PCT (a whatIf margin inflation) removed
# 2026-09-10 -- the whatIf probe it buffered is gone from the sizing path
# entirely (see INITIAL_MARGIN_FRACTION below).
# Single Stock Initial Margin as a flat fraction of the stock price -- per
# the client 2026-09-10: "remove whatIF. Calculate quantity using 35%
# Initial Margin Deposit." Replaces the IBKR whatIf probe (and its
# refinement re-probe) in the entry sizing path, after that number
# under-reported a short's real margin requirement on a live order.
INITIAL_MARGIN_FRACTION = 0.35
# Nominal cycle length -- bars are 1 minute (per the client's own "new 1
# minute candles" spec), so this is still the right number to reason about
# elsewhere (e.g. live_data.py's MAX_BAR_AGE_SECONDS staleness comment). No
# longer the literal sleep() duration below -- see
# _seconds_until_next_bar_close, added 2026-09-11 per the client: "If the
# goal is catching a new candle the instant it closes instead of up to a
# minute late... Yes of course. That is the way it is supposed to be." A
# flat sleep(60) after each cycle drifts out of alignment with real bar
# close times whenever the cycle's own compute time varies (confirmed live
# 2026-09-11: cycle-to-cycle gaps ranged 38-109s against this 60s target),
# so a slow cycle could end up checking a bar up to CHECK_INTERVAL_SECONDS
# late. Aligning the sleep to the wall-clock minute boundary instead bounds
# that lag to a couple of seconds regardless of how long the prior cycle took.
CHECK_INTERVAL_SECONDS = 60
# Grace period after each minute boundary before checking -- gives the live
# feed a moment to finish delivering and parsing that minute's bar rather
# than racing the exact boundary and reading last minute's stale value.
BAR_CLOSE_CHECK_BUFFER_SECONDS = 2.0
UNIVERSE_SCREENING_ENABLED = os.getenv("UNIVERSE_SCREENING_ENABLED") == "1"
# Narrowed from 5 minutes to 1, per the client 2026-08-31: confirmed live
# the same day that a newly-qualifying pair can have its divergence run
# straight through its own entry window in the gap between scans -- two
# pairs (CRWD/PANW, ADI/ON) were already past trigger the moment they were
# discovered and warmed up, 17-18 minutes after market open, because the
# scan hadn't caught them yet. A screen can itself run longer than this
# interval (see universe_screen_state's own comment on the overlap guard
# that already handles that), so this is a floor, not a guarantee every
# cycle actually completes every 60s.
UNIVERSE_SCREEN_INTERVAL = timedelta(minutes=1)
# Confirmed live 2026-08-27: warm_up_pairs() runs ONCE at 9:30am and a
# transient gap in that single Databento response (5 of 7 qualified pairs
# came back with zero bars that morning, re-running the identical call
# minutes later returned all of them fine) silently left those pairs with
# empty price_history_a/b for the entire session -- no trigger, no
# tracking, no way to ever trade that day, with only a debug-level "no
# data" line to show for it. Retrying on this interval for whichever pairs
# are still missing data (not a raw re-run of the whole warm-up) gives a
# one-time API hiccup a chance to self-heal during the day instead of
# taking a pair out of play until tomorrow. 1 minute, per the client
# 2026-08-28 ("5 minutes is too late for us. 1 minute, max") -- the fetch
# itself is observed to take ~30s, and retry_state["running"] already
# guards against a second attempt starting before the first finishes, so
# this is the tightest safe interval, not an arbitrary choice.
WARMUP_RETRY_INTERVAL = timedelta(minutes=1)
# Per the client 2026-09-01: "Tomorrow, by 9am it loads. We want it
# loaded by 9am." -- when the overnight cache_qualified_pairs_for_tomorrow
# result gets loaded (a local file read, not a network fetch), distinct
# from SESSION_START_TIME below, which is when actual pair SELECTION
# (the live fallback if that cache is missing) is allowed to run.
PAIR_LOAD_TIME = dtime(9, 0)
BLACKLIST_SCAN_TIME = dtime(9, 25)
STOP_LOSS_ELV_SNAPSHOT_TIME = dtime(9, 25)  # separate freeze from the 9:29am sizing snapshot below
ELV_SNAPSHOT_TIME = dtime(9, 29)
# Briefly moved to a separate 9:00am PAIR_SELECTION_TIME the same day,
# then reverted back to this -- the client 2026-09-01: "Pair Selection
# cannot start by 9am. Pair Selection starts by 9:30am. We use only RTH
# Data." See select_pairs_from_previous_session's own call site for the
# full story.
SESSION_START_TIME = dtime(9, 30)
INTRADAY_NEWS_SCAN_START_TIME = dtime(9, 25)  # "From 9:25am ET and every 5 minutes"
# Narrowed from 3:30pm to 11:30am ET, per the client 2026-08-24's Market
# Scan Timeline rollback: "SEC EDGAR, Gemini/GPT Grounded News Scan stops
# at 11:30am ET." NEWS STOP still applies to whatever the scan found before
# that cutoff -- a position still open past 11:30am simply stops getting
# fresh news coverage for the remainder of its hold, which is the client's
# own explicit timeline, not an oversight.
INTRADAY_NEWS_SCAN_END_TIME = dtime(11, 30)
MONITOR_LOOP_END_TIME = dtime(15, 59)
REPORT_SEND_TIME = dtime(16, 0)


def write_status(
    status: str, account: str = "", elv: float = 0.0,
    contexts: "list[PairContext] | None" = None,
    daily_pnl: float = 0.0, trading_halted: bool = False,
    news_scan_healthy: bool = True,
) -> None:
    """Writes data/bot_status.json every check cycle so the dashboard (a
    separate process) can show live state -- same pattern as the sibling SPY
    bot's _write_status(). The dashboard treats this file as stale (offline)
    once it's more than 30s old, so a crashed/killed bot shows as offline
    within half a minute rather than showing frozen last-known state forever."""
    open_positions = []
    for ctx in (contexts or []):
        position = ctx.position
        if position is None:
            continue
        if len(ctx.price_history_a):
            price_a = ctx.price_history_a.iloc[-1]
            price_b = ctx.price_history_b.iloc[-1]
        else:
            price_a, price_b = position.entry_price_a, position.entry_price_b
        # Position Size in "pairs" -- per the client's own worked example
        # 2026-08-17: "Leg Size: 15 MPWR, 91 MRVL. Position Size: (15+91)/2
        # = 53 Pairs. It is 53 Pairs. Not 1 Pair." The previous version used
        # combo_ratio_and_units' reduced whole-share ratio (gcd-based), which
        # only coincidentally equals qty_a/qty_b when the two share counts
        # are coprime (as they were for MPWR/MRVL) -- his own formula is
        # just the average of the two raw leg quantities, not a ratio
        # reduction. Leg Size itself is qty_a/qty_b directly, already below.
        position_size_pairs = (position.qty_a + position.qty_b) / 2
        open_positions.append({
            "symbol_a": ctx.symbol_a, "symbol_b": ctx.symbol_b,
            "action_a": position.action_a,
            # Which symbol action_a actually describes -- "A" for a real
            # two-leg position (always, by construction) or a single-leg
            # position trading symbol_a; "B" for a single-leg position
            # trading symbol_b, where action_a is repurposed to describe
            # symbol_b instead. See Position.traded_leg's own docstring.
            "traded_leg": position.traded_leg,
            "qty_a": position.qty_a, "qty_b": position.qty_b,
            "entry_price_a": position.entry_price_a, "entry_price_b": position.entry_price_b,
            "pair_units": round(position_size_pairs, 1),
            "entry_margin": round(position.entry_margin, 2),
            "entry_margin_ibkr_probed": (
                round(position.entry_margin_ibkr_probed, 2)
                if position.entry_margin_ibkr_probed is not None else None),
            "entry_margin_ibkr_probed_qty": position.entry_margin_ibkr_probed_qty,
            "entry_time": position.entry_time.isoformat(),
            "entry_spread": position.entry_spread,
            "unrealized_pnl": round(unrealized_pnl(position, price_a, price_b), 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct(position, price_a, price_b) * 100, 2),
            # Per the client 2026-08-19: "Points to trigger @ Entry = 0.24,
            # Points + trigger @ Take Profit = 1.52" -- both fixed at entry,
            # not live. The first is how close to the trigger this position
            # was when it opened (entry_trigger_level - entry_divergence,
            # the same distance_to_trigger entry_eligible() checked against
            # the $0.01-$0.40 window); the second is the trigger level
            # itself, the fixed divergence Take Profit fires at.
            "points_to_trigger_at_entry": round(position.entry_trigger_level - position.entry_divergence, 2),
            "take_profit_trigger_level": round(position.entry_trigger_level, 2),
        })
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Atomic write (temp file + os.replace) -- independent audit finding,
        # confirmed real: heartbeat_monitor.py polls this file from a
        # SEPARATE process on its own interval and already treats a
        # JSONDecodeError the same as a missing/stale file (status_is_stale),
        # so reading this file mid-write (a plain write_text is not atomic)
        # could fire a false "bot_status.json has gone stale" Telegram alert
        # purely from unlucky timing, with nothing actually wrong. Same
        # pattern this project already uses for every other JSON file two
        # processes can touch (news_blacklist.py, market_entry_list.py).
        payload = json.dumps({
            "ts": time.time(), "status": status, "account": account, "elv": elv,
            "open_positions": open_positions, "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl / elv * 100, 2) if elv else None,
            "trading_halted": trading_halted,
            "news_scan_healthy": news_scan_healthy,
            # Per the client 2026-09-02: the Live Johansen/Hurst panel's own
            # "need >X" label was hardcoded to 0.60 and went stale the
            # moment MIN_HURST was lowered to 0.50 the day before -- sent
            # here so the dashboard reads the real, current threshold
            # every poll instead of a number baked into dashboard.py that
            # has to be remembered and hand-edited every time this changes.
            "min_hurst": MIN_HURST,
            "qualified_pairs_count": len(contexts) if contexts is not None else None,
            # The pairs cleared for trading, so the dashboard can show the
            # list and edit it. Read from disk each cycle rather than cached,
            # so a dashboard edit is reflected without restarting the bot.
            "market_entry_list": sorted(f"{a}/{b}" for a, b in (load_market_entry_list() or [])),
            "permanent_blacklist": sorted(load_permanent_blacklist()),
            "market_halt": MarketHalt().details(),
            "special_pairs": [f"{a}/{b}" for a, b in special_pairs_as_listed()],
            "pair_divergence": [
                {
                    "symbol_a": ctx.symbol_a, "symbol_b": ctx.symbol_b,
                    "current": round(abs(ctx.last_deviation), 4) if ctx.last_deviation is not None else None,
                    "trigger": round(ctx.divergence_trigger_level, 4),
                    "distance": (round(ctx.divergence_trigger_level - abs(ctx.last_deviation), 4)
                                 if ctx.last_deviation is not None else None),
                    # Per the client 2026-09-13: "the pair that met price
                    # conditions (drop of below 50%, no retouch of Trigger
                    # inside lookback period) will show cleared" -- the
                    # >50% reversion since the prior session's own peak
                    # (reverted_over_50pct_since_max_divergence, a DISTINCT
                    # condition from the dormant 100%-reversion Condition
                    # A, confirmed the same message: "There is no
                    # condition of 'the pair not having fully mean
                    # reverted since its max divergence'. This condition
                    # does not exist.") AND no retouch of trigger yet
                    # this session (trigger_reached_today, sticky --
                    # confirmed lookback here is "current RTH", i.e.
                    # today, the same field entry_eligible already reads).
                    "cleared": ctx.reverted_over_50pct_since_max_divergence and not ctx.trigger_reached_today,
                }
                for ctx in (contexts or [])
                # Per the client 2026-08-26, sending the actual reference
                # mockup: "This is how Divergence Watch should look like.
                # There is no 'waiting' anywhere." Reverts the 2026-08-25
                # change that showed a pair before it had a real trigger --
                # every row here now needs a real number, not a placeholder
                # state. Only pairs with an established trigger show at all.
                #
                # RELAXED 2026-09-13, per the client: "when the prior RTH
                # qualified pairs are loaded in the Divergence Watch by
                # 9am, we want to see the pairs. The pairs should be
                # visible by 9am. Even if live data is yet to come on." A
                # pair no longer needs ctx.last_deviation (today's live
                # current) to show at all -- current/distance above go
                # null until the first live bar arrives, but trigger and
                # cleared status show immediately from the 9am cache load.
                # Only affects pairs loaded before live data starts (the
                # 9am cache): a live-discovery pair is never tracked here
                # at all until it already has a real last_deviation, so
                # this can't reopen the "waiting" placeholder state
                # 2026-08-26 removed for that path.
                #
                # MPWR/MRVL note, 2026-08-26 ("MPWR/MRVL should not be
                # there. Divergence >$6."): this used to also exclude a pair
                # whose current deviation exceeded MAX_ENTRY_DIVERGENCE, via
                # the same line removed 2026-09-12 below (see that comment).
                # abs(ctx.last_deviation) is no longer checked against this
                # range at all -- current is governed by the entry window
                # only now, per the client's own explicit 2026-09-12
                # correction. Left here as history, not a live exclusion.
                #
                # Also excludes a pair with either leg over MAX_STOCK_PRICE
                # ($700) -- per the client 2026-08-26: "LITE stock price
                # >$700. It should not be in the Divergence Watch, as
                # qualified pair." Same reasoning: entry_eligible's own
                # Maximum Stock Price check (applied per stock, not per
                # pair) would block it regardless of trigger or deviation.
                #
                # Also excludes a pair that has never cleared Johansen AND
                # Hurst live -- per the client 2026-09-01: "The pairs that
                # should be on the Divergence Watch are qualified pairs,"
                # confirmed "Johansen and Hurst are qualifying conditions."
                # Confirmed live 2026-08-31: JPM/USB showed on Divergence
                # Watch with a real, in-range distance the whole time its
                # window was open, but entry_eligible was False that whole
                # time because live_hurst never cleared 0.60 -- same
                # "cannot actually trade, so showing it is misleading"
                # reasoning as the two exclusions above, just for the live
                # gate instead of the static price/divergence ones.
                #
                # Also excludes a pair cancelled pre-entry by NEWS STOP --
                # per the client 2026-09-01: "cancel selected pair, if not
                # yet a position." See PairContext.news_cancelled's own
                # comment -- a cancelled pair is done for the session, not
                # just hidden while news is active.
                #
                # A pair the live discovery scan found mid-session was
                # excluded from here 2026-09-01, per the client tracing a
                # real dashboard confusion back to its cause -- re-added
                # 2026-09-02, per the client: "Add live scan discoveries."
                # PairContext.discovered_live is still tracked (used
                # elsewhere, e.g. the post-trade report's own pair source
                # breakdown) even though it's no longer read here.
                if ctx.divergence_trigger_level is not None
                # MIN_ENTRY_DIVERGENCE/MAX_ENTRY_DIVERGENCE gate the TRIGGER
                # only, not today's live current value -- per the client
                # 2026-09-12, ending a long same-night back-and-forth: "1.5-5
                # range applies only to the trigger. I never said live
                # current price must be 1.5-5." The line that used to check
                # abs(ctx.last_deviation) against this range directly is
                # REMOVED (was added 2026-09-08/09-10/09-11 on an
                # interpretation the client corrected here). Current is
                # governed entirely by the distance-to-trigger window check
                # further down (johansen/hurst + price/news exclusions),
                # never by its own absolute floor/ceiling.
                #
                # ctx.divergence_trigger_level itself is already clamped to
                # [MIN_ENTRY_DIVERGENCE, MAX_ENTRY_DIVERGENCE] upstream by
                # _bounded_trigger_level, so this check is defense-in-depth,
                # not the primary enforcement point -- a pair whose trigger
                # is out of range never gets a divergence_trigger_level at
                # all, so this line is normally always true by construction.
                and ctx.divergence_trigger_level >= MIN_ENTRY_DIVERGENCE
                and (len(ctx.price_history_a) == 0 or ctx.price_history_a.iloc[-1] <= MAX_STOCK_PRICE)
                and (len(ctx.price_history_b) == 0 or ctx.price_history_b.iloc[-1] <= MAX_STOCK_PRICE)
                # 1-day Hurst, 1-day Pearson, 1-day Johansen and the condition A
                # mean-reversion check all removed as gates -- 1-day Johansen
                # per the client 2026-09-09 ("Ok. Remove 1 Day Johansen"),
                # after confirming live on the Divergence Watch it was the
                # remaining blocker; the other three per the client
                # 2026-09-08. See the PairSignal construction in
                # run_check_cycle for the full reasoning. This filter tracks
                # entry_eligible exactly (a row here must be genuinely
                # enterable), so it drops the same four.
                and ctx.johansen_ever_unconfirmed and ctx.hurst_ever_above_threshold
                and not ctx.news_cancelled
            ],
            # Raw live Johansen/Hurst readings for every flat, tracked pair
            # that has computed at least once this cycle -- per the client
            # 2026-09-01: "We need to see the data on the Dashboard and be
            # able to make decisions on the 0.60." Separate from
            # pair_divergence above: this shows EVERY tracked pair's actual
            # numbers, including ones that haven't qualified for Divergence
            # Watch (or never will), not just the ones that already passed.
            "live_validation": [
                {
                    "symbol_a": ctx.symbol_a, "symbol_b": ctx.symbol_b,
                    "live_hurst": round(ctx.live_hurst, 4) if ctx.live_hurst is not None else None,
                    "live_johansen_confirmed": ctx.live_johansen_confirmed,
                    "johansen_ever_unconfirmed": ctx.johansen_ever_unconfirmed,
                    "hurst_ever_above_threshold": ctx.hurst_ever_above_threshold,
                    # The "1 Day" parallel window -- per the client
                    # 2026-09-06, confirmed "two separate calculations".
                    # See PairContext.johansen_hurst_price_a_1day's own
                    # comment.
                    "live_hurst_1day": round(ctx.live_hurst_1day, 4) if ctx.live_hurst_1day is not None else None,
                    "live_johansen_confirmed_1day": ctx.live_johansen_confirmed_1day,
                    "johansen_ever_unconfirmed_1day": ctx.johansen_ever_unconfirmed_1day,
                    "hurst_ever_above_threshold_1day": ctx.hurst_ever_above_threshold_1day,
                    # 1 Day Pearson -- per the client 2026-09-07. See
                    # PairContext.live_pearson_1day's own comment.
                    "live_pearson_1day": round(ctx.live_pearson_1day, 4) if ctx.live_pearson_1day is not None else None,
                    "pearson_ever_above_threshold_1day": ctx.pearson_ever_above_threshold_1day,
                    # The two daily-close Pearson readings this pair qualified
                    # on, so the dashboard can show the real gate values and
                    # their real directions (60 day >= 0.90, 20 day < 0.70)
                    # rather than only live intraday numbers.
                    "correlation_60d": ctx.correlation_60d,
                    "correlation_20d": ctx.correlation_20d,
                    "trigger_reached_today": ctx.trigger_reached_today,
                    # Checklist_090926.pdf condition A -- see
                    # PairContext.mean_reverted_since_max_divergence's own
                    # comment.
                    "mean_reverted_since_max_divergence": ctx.mean_reverted_since_max_divergence,
                }
                for ctx in (contexts or [])
                if ctx.position is None and (ctx.live_hurst is not None or ctx.live_johansen_confirmed is not None
                                              or ctx.live_hurst_1day is not None
                                              or ctx.live_pearson_1day is not None
                                              or ctx.live_johansen_confirmed_1day is not None)
            ],
        })
        tmp = STATUS_PATH.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, STATUS_PATH)
    except OSError:
        logger.warning("Failed to write status file", exc_info=True)


async def sleep_until_et(target_time: dtime) -> None:
    """Waits until target_time ET today (no-op if that time has already
    passed) -- lets the process be launched once at 8:25am and internally
    wait out each subsequent step at its correct clock time, rather than
    requiring an external scheduler to fire it separately for each step.

    Set PAIRS_BOT_SKIP_TIME_GATES=1 to skip every wait and run straight
    through the day's steps back to back, for testing against a live paper
    account without waiting for each real clock time. Everything still runs
    against the real current time otherwise (entry window, EOD flatten,
    etc.) -- this only removes the artificial waits between steps. Unset it
    to go back to the normal schedule."""
    if os.getenv("PAIRS_BOT_SKIP_TIME_GATES") == "1":
        return
    now = datetime.now(ET)
    target = datetime.combine(now.date(), target_time, tzinfo=ET)
    remaining = (target - now).total_seconds()
    if remaining > 0:
        await asyncio.sleep(remaining)


@dataclass
class NewsScanHealth:
    """Whether the most recent news scan (pre-market or intraday) actually
    succeeded. Per the client 2026-08-17: "If Gemini news API is down, no
    trading." -- in-memory only, not persisted, so a restart starts fresh
    optimistic rather than staying stuck on a failure from a previous
    process. Only gates NEW entries (same as market_halt/halted_market) --
    an already-open position still runs its normal exit rules regardless,
    since abandoning a live position because a news check failed is worse
    than working it out through take profit, stop loss, or EOD."""
    healthy: bool = True


@dataclass
class PairContext:
    """Per-pair running state for the day: hedge ratio (frozen for the
    whole session, per the client 2026-08-21 -- see the module docstring),
    live spread history (seeded from the morning warm-up, then appended to
    as live bars arrive), the day's divergence trigger level, and half-life.

    Per the client 2026-08-22's expansion-strategy rewrite, half_life_minutes
    is FROZEN, set once at warm-up from the previous RTH session alone
    (establish_base_trigger) and never recalculated again -- the fix for a
    pair's half-life drifting mid-session and disqualifying an
    already-qualified pair under the old live-every-cycle design (the
    GE/RTX incident). divergence_trigger_level starts the same way, but --
    per the client, restated across two later PDF revisions for emphasis --
    keeps being recalculated every cycle from 9:31am through the 10:30am
    entry cutoff (recalculate_trigger_exceptions), replaced by a larger
    non-reverting divergence over a wider, premarket-inclusive window only
    when cointegration is also confirmed broken that cycle.

    live_correlation/live_adf_pvalue are recomputed every cycle over a
    rolling N=30-bar window (update_live_cointegration_stats) and stay None
    until enough live bars have accumulated."""
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    spread_history: pd.Series
    # Persistent Kalman beta estimator, un-frozen 2026-08-24 per the client:
    # "Beta is not frozen. Continuously updating. Just like before" -- before
    # the 2026-08-21/22 freeze, per this file's own prior comment: "back when
    # beta was a live Kalman estimate that moved continuously". Lazily seeded
    # from the initial hedge_ratio the first time run_check_cycle sees this
    # ctx (not at construction, since PairContext is built before hedge_ratio
    # is necessarily final for every caller). Kept as ONE persistent instance
    # per pair across the whole session -- NOT reconstructed each cycle --
    # so it updates incrementally bar-by-bar rather than replaying history
    # from scratch (that replay-from-scratch pattern is what
    # cointegration_breakdown.run_kalman_monitor does deliberately for its
    # own diagnostic-only purpose; this is a different, stateful use).
    beta_estimator: "KalmanBetaEstimator | None" = None
    divergence_trigger_level: float | None = None
    half_life_minutes: float | None = None
    position: object | None = None
    live_correlation: float | None = None
    live_adf_pvalue: float | None = None
    # The two DAILY-CLOSE Pearson readings this pair qualified on, kept from
    # selection time so both are visible side by side -- per the client
    # 2026-09-08, who read the report's single "Correlation" column (which
    # is live_correlation above, a rolling 30-BAR intraday reading, a third
    # thing again) as the 20 day one and asked "How can 20 Day Pearson read
    # 90?". They are two different tests in opposite directions: "60 Day
    # Pearson >= 0.90, 20 Day Pearson < 0.70" (see PairResult.qualified).
    correlation_60d: float | None = None
    correlation_20d: float | None = None
    # The trigger established once at warm-up from the previous RTH session
    # alone (through its 4pm close), before any 9:31am+ exception check --
    # kept separately from divergence_trigger_level so
    # recalculate_trigger_exceptions can compare each cycle's candidate
    # exception value against this fixed baseline ("IF there is a new
    # Maximum Divergence, it is the exception. IF there is no new Maximum
    # Divergence, then the one from previous RTH applies").
    base_trigger_level: float | None = None
    # True when divergence_trigger_level is currently the recalculated
    # exception value rather than base_trigger_level -- set/cleared fresh
    # every cycle by recalculate_trigger_exceptions. Per the client
    # 2026-08-23: "Allowed Divergence Gap does not apply to Exception
    # Maximum Divergence determined at Entry" -- read by the entry signal
    # builder to skip that test only while this is True.
    using_exception_trigger: bool = False
    # Per the client 2026-09-07 (Checklist_090926.pdf), entry condition A:
    # "There is no 100% mean reversion after Maximum Divergence." Set fresh
    # every cycle -- by establish_base_trigger at warm-up (prior session
    # alone) and by recalculate_trigger_exceptions from 9:31am onward (the
    # same combined prior-session + premarket-to-now window used for the
    # trigger itself), same "re-derived fresh, not sticky" convention as
    # divergence_trigger_level -- via spread_engine.reverted_after_peak.
    # See PairSignal.mean_reverted_since_max_divergence's own comment for
    # why the other half of condition A needs no new code.
    mean_reverted_since_max_divergence: bool = False
    # Per the client 2026-09-13: "After it hit the maximum divergence, it
    # should have at least >50% mean reversion." A distinct condition from
    # the one above, confirmed the same message: "There is no condition of
    # 'the pair not having fully mean reverted since its max divergence'.
    # This condition does not exist." Same "re-derived fresh, not sticky"
    # pattern, set at the same two call sites as
    # mean_reverted_since_max_divergence, via
    # spread_engine.reverted_more_than_pct_after_peak. Feeds the
    # Divergence Watch's "cleared" label (write_status's pair_divergence)
    # together with (not signal.trigger_reached_today, "no retouch of
    # Trigger inside lookback period", the client confirming lookback here
    # is the current RTH session).
    reverted_over_50pct_since_max_divergence: bool = False
    # The previous RTH session's own price history alone (through its 4pm
    # close), set once at warm-up -- kept separately from price_history_a/b
    # (previous + today, appended to live) so recalculate_trigger_exceptions
    # can rebuild the exact "previous RTH + today's premarket-to-now" window
    # the recalculation needs, without re-fetching the previous session.
    prior_session_price_a: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    prior_session_price_b: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Today's premarket-to-now price history (from PREMARKET_FETCH_START_TIME,
    # e.g. 8:00am ET) -- refetched every cycle by recalculate_trigger_exceptions
    # while it's still due to run, separately from price_history_a/b below
    # (which stays regular-hours-only for trading).
    premarket_price_a: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    premarket_price_b: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # The TRUE session-open print for each leg -- per a real 2026-09-19
    # bug, confirmed against the client's own chart: passes_prior_excursion_dominance_filter/
    # passes_prior_excursion_direction_filter were being fed the prior
    # session's first bar's CLOSE as a stand-in for its Open (the same
    # "close series as an OHLC proxy" convention establish_base_trigger
    # uses, which is fine for High/Low but not for Open -- a bar's own
    # close can differ meaningfully from its open, and these filters
    # divide by the smaller of two excursions, so that difference gets
    # amplified into a completely wrong ratio). CME's real 2026-09-17
    # session: true Open=271.385 gives a 123% dominance ratio (fails);
    # the close-price proxy (270.19) gave 1003% (a false pass). Set once
    # at warm-up (prior) and refreshed each cycle while
    # recalculate_trigger_exceptions is still due to run (today) -- None
    # until the real bar has actually been fetched.
    prior_session_open_a: float | None = None
    prior_session_open_b: float | None = None
    today_session_open_a: float | None = None
    today_session_open_b: float | None = None
    # Per the client 2026-08-22: "Filter for pairs that were correlated and
    # co-integrated uptil close at 4pm ET, but correlation and co-integration
    # broke before 9:31am ET" -- re-verified every cycle by
    # recalculate_trigger_exceptions while it's still due to run. None until
    # that check has run at least once.
    cointegration_broke_before_931: bool | None = None
    # Per the client: "10am till Now ensures broken co-integration right
    # until entry. Not just till 10am." -- re-verified continuously by
    # update_live_cointegration_stats over a rolling N=30-bar window.
    cointegration_still_broken: bool = False
    # Last current_spread - current_equilibrium computed for this pair while
    # flat, purely for dashboard/log visibility into how close it is to
    # divergence_trigger_level -- not read by any trading decision.
    last_deviation: float | None = None
    # The CURRENT live divergence event's own start -- per the client
    # 2026-09-03, Expansion_Leg.pdf: the baseline (t0) the Expansion Leg
    # formula measures each leg's price change against is "when the
    # divergence began expanding," not the whole session. Set the moment
    # |spread - equilibrium| first leaves the tolerance band (same
    # transition detect_divergence_events' own state machine detects,
    # just tracked live here instead of after the fact), cleared back to
    # None the moment the spread returns inside the band -- see
    # update_current_divergence_event.
    current_event_start_time: pd.Timestamp | None = None
    current_event_price_a: float | None = None
    current_event_price_b: float | None = None
    # Populated at load time (qualified_pairs.csv's own column, or the
    # PairResult a universe-screening discovery produced) -- purely for the
    # post-trade report's Daily Qualifying Pairs table, not read by any
    # trading decision.
    industry: str = ""
    # True only for a pair _run_universe_screen discovered mid-session
    # (the live, full-S&P-500 discovery scan) rather than one
    # select_pairs_from_previous_session selected from the prior RTH
    # session. Per the client 2026-09-01: "Divergence Watch is simple. 1.
    # Only pairs selected 9:30-4pm yesterday. 2. If they pass Hurst and
    # Johansen, they are included in the Divergence Watch." -- a
    # live-discovered pair can still be tracked and traded (the discovery
    # scan itself is unchanged, still running every 1 minute), it just
    # never shows on Divergence Watch. See write_status's pair_divergence
    # filter.
    discovered_live: bool = False
    # Raw per-leg price history (warm-up + live), kept alongside the derived
    # spread so an open position's own fixed-ratio spread can be rebuilt at
    # any time, and so correlation and ADF run over the same window.
    price_history_a: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    price_history_b: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Position size used for this pair today -- the post-trade report needs a
    # share count to turn a spread-dollar excursion into an account-dollar %,
    # and the Allowed Divergence Gap needs both legs added together.
    day_sizing_qty_a: float = 0.0
    day_sizing_qty_b: float = 0.0
    # This cycle's qualifying divergence events, kept so the Allowed
    # Divergence Gap can be checked without recomputing them.
    divergence_events: list = field(default_factory=list)
    # Per the client 2026-08-19: "Trigger has not previously been reached
    # at any point during the current RTH session" -- sticky for the whole
    # day once set (never reset on disqualify/requalify, only ever reset by
    # a fresh session tomorrow), independent of whether a position is open.
    trigger_reached_today: bool = False
    # Per the client 2026-08-25 (Correct_Filters.pdf): "Rolling 20 Days
    # (1-Minute Intraday Bars)... Bot will continue to check with new 1
    # minute candles, right until entry," confirmed as a sliding, fixed-
    # length array in the "Live Bot Execution (Sliding Gate)" follow-up.
    # Seeded once at warm-up with the trailing 20 FULL trading days
    # (strictly before today), then today's bars from 9:30am (market open)
    # onward are appended live each cycle, dropping the oldest bar once the
    # array reaches JOHANSEN_HURST_WINDOW_BARS -- kept separate from
    # price_history_a/b, which starts at warm-up's much shorter 2-session
    # window and exists for the entry trigger/spread, not this statistical
    # gate. See update_live_johansen_hurst.
    johansen_hurst_price_a: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    johansen_hurst_price_b: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    live_johansen_confirmed: bool | None = None
    live_hurst: float | None = None
    # Per the client 2026-08-28: "Hurst and Johansen check ends IF validated
    # only once, during current RTH session," clarified same day: "They
    # must not be validated together. At the same time. No. They just need
    # to be validated once during Entry RTH session." -- two INDEPENDENT
    # sticky flags, not one flag requiring both conditions to clear in the
    # SAME cycle. Johansen can clear on cycle 5 and Hurst on cycle 12; once
    # each has cleared at least once, each stays cleared for the rest of
    # the session (same sticky convention as trigger_reached_today) even if
    # a later cycle's raw reading would fail on its own -- real, observed
    # live 2026-08-28: GE/RTX's live_hurst read 0.496 (just under 0.60) on
    # the one cycle its distance was inside the entry window.
    johansen_ever_unconfirmed: bool = False
    hurst_ever_above_threshold: bool = False
    # The "1 Day" parallel window -- per the client 2026-09-06, confirmed
    # "two separate calculations": the fields above are now specifically
    # the 20-DAY variant; these are the SAME mechanism (seed + live
    # sliding append, see JOHANSEN_HURST_WINDOW_1DAY_BARS's own comment)
    # at a 390-bar (one trading day) length instead of 7800. Both must
    # clear, independently of the 20-day pair, for entry_eligible's
    # combined gate -- see the PairSignal construction in run_check_cycle.
    johansen_hurst_price_a_1day: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    johansen_hurst_price_b_1day: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    live_johansen_confirmed_1day: bool | None = None
    live_hurst_1day: float | None = None
    johansen_ever_unconfirmed_1day: bool = False
    hurst_ever_above_threshold_1day: bool = False
    # 1 Day Pearson -- per the client 2026-09-07 (1_Day_Pearson_60_Bars.pdf,
    # "Method 1: The Rolling Intraday Bar-to-Bar Approach"), superseding
    # TWO earlier same-day versions (a yesterday-close anchor, then a
    # yesterday-3pm anchor with one averaged point) -- both confirmed by
    # the client as correct at the time, then reversed once he re-checked
    # his own worked numbers. This version needs no anchor and no seed
    # field at all: it correlates leg A's own bar-to-bar 1-minute return
    # series against leg B's, over a plain trailing PEARSON_1DAY_WINDOW_BARS
    # window of ctx.premarket_price_a/b (already fetched every cycle by
    # recalculate_trigger_exceptions, premarket-inclusive so the window is
    # full even in the first hour after the open) -- no dedicated
    # PairContext field needed for the window itself. See
    # phase1.screening.pearson_correlation_1day's own docstring for the
    # worked example (Today Live 0.12 vs Yesterday Anchor 0.88, the
    # latter purely informational, not read by the entry gate).
    live_pearson_1day: float | None = None
    pearson_ever_above_threshold_1day: bool = False

    # NEWS STOP, pre-entry half -- per the client 2026-09-01: "IF a stock
    # in a selected pair becomes an excluded symbol due to news, exit pair
    # @ MKT if a position. Or cancel selected pair, if not yet a
    # position." The exit half already existed (run_check_cycle's
    # news_stop); this flag is the "not yet a position" half -- set True
    # once, in run_check_cycle, the moment either leg is found in that
    # cycle's news blacklist while flat, and sticky for the rest of the
    # session from there (same convention as trigger_reached_today) so the
    # pair doesn't reappear on Divergence Watch or become enterable again
    # if the news listing later clears.
    news_cancelled: bool = False


WARMUP_LOOKBACK_CALENDAR_DAYS = 6
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)

# PairContext.johansen_hurst_price_a/b's window -- 20 trading days needs
# roughly 28-30 calendar days including weekends; padded further for
# holidays, same reasoning as WARMUP_LOOKBACK_CALENDAR_DAYS above.
JOHANSEN_HURST_LOOKBACK_TRADING_DAYS = 20
JOHANSEN_HURST_LOOKBACK_CALENDAR_DAYS = 34
# "Rolling Logic: 10am+Now" in the client's original PDF, then "enters the
# watch queue at 09:30 AM ET" in the "Live Bot Execution (Sliding Gate)"
# follow-up -- confirmed 9:30am by the client 2026-08-25 when asked to
# reconcile the two.
NEWS_STOP_ENABLED = False
JOHANSEN_HURST_LIVE_START_TIME = dtime(9, 30)
# NEWS STOP + Excluded List DISABLED per the client 2026-09-09: "Ok. Remove
# Excluded List and New[s] Stop" -- confirmed after ADBE/ADSK was pre-entry
# cancelled by this exact mechanism that morning. Guards both halves (the
# post-entry exit and the pre-entry cancel below) without touching any of
# the underlying computation (news_flagged_now, excluded_now, blacklisted,
# the ExcludedSymbols writes) -- same dormant-not-deleted convention as
# every other removed gate in this file. Flip back to True to restore
# either half.
MIN_BARS_FOR_LIVE_JOHANSEN_HURST = 300  # same floor select_pairs_from_previous_session uses
# Per the client 2026-08-25 ("Live Bot Execution (Sliding Gate)"): "each
# incoming 1-minute bar appends to the array, dropping the oldest bar to
# keep a fixed array length" -- 20 trading days x 390 RTH minutes/session,
# the client's own figure ("20-day 1-minute intraday arrays (7,800 bars)").
JOHANSEN_HURST_WINDOW_BARS = 7800
# Per the client 2026-09-06: Hurst and Johansen are "two separate
# calculations" -- "20 Day Hurst"/"20 Day Johansen" (JOHANSEN_HURST_WINDOW_BARS
# above) AND, independently, "1 Day Hurst"/"1 Day Johansen", each with the
# identical "Rolling ... (1-Minute Intraday Bars), Live Bot Execution
# (Sliding Gate)" description -- the SAME sliding mechanism as the 20-day
# window, just a much shorter target length: one trading day's worth of
# 1-minute bars (390 = 6.5 RTH hours x 60), not a historical-only/
# today-only split. Both windows include today's live bars as they arrive;
# they only differ in how far back each one reaches.
JOHANSEN_HURST_WINDOW_1DAY_BARS = 390
# 1 Day Pearson's own, much shorter window -- per the client 2026-09-07
# (1_Day_Pearson_60_Bars.pdf): N=60, a trailing 60 one-minute bars, NOT
# the same window as Hurst/Johansen's 390-bar one-trading-day array. See
# PairContext.live_pearson_1day's own comment.
PEARSON_1DAY_WINDOW_BARS = 60


def _fetch_johansen_hurst_history(symbols: list[str], now: datetime) -> pd.DataFrame:
    """Trailing 20 FULL trading days of 1-minute RTH closes, strictly before
    today.

    Switched to LIVE_DATASETS (EQUS.MINI, phase2/live_data.py's own
    consolidated-SIP-style feed) 2026-09-16, per the client: "Use
    consolidated SIP data and run the scan without disabling anything" --
    replacing the primary-exchange feeds (phase1/databento_source.py's
    XNAS.ITCH/XNYS.PILLAR), confirmed real that same evening to be
    returning "403 license_not_found_unauthorized... A live data license
    is required to access XNAS.ITCH data after 2026-09-15T04:00:00Z" for
    this account -- not a code bug, an account-level entitlement gap. This
    was ALSO the feed already used for today's live divergence tracking
    (fetch_intraday_warmup below), on the documented tradeoff that it's
    lower-accuracy/cheaper than the primary feeds (see live_data.py's own
    comment) -- using it here too means the whole pipeline (history AND
    today) is consistently on the lower-fidelity feed for as long as the
    account lacks XNAS.ITCH/XNYS.PILLAR live access, rather than the
    previous mixed history(accurate)+today(EQUS.MINI) split. Revert to
    PRIMARY_DATASETS (the default, by passing no `datasets` override) once
    the client's live license for XNAS.ITCH/XNYS.PILLAR is in place."""
    from phase1.databento_source import fetch_intraday_closes_databento
    today_et = now.astimezone(ET).date()
    # `end` is EXCLUSIVE on Databento's API (confirmed against the real API,
    # 2026-08-25: passing yesterday's date as `end` silently excluded
    # yesterday itself) -- today_et here correctly covers through the end of
    # yesterday while never reaching into today, matching the same
    # convention phase1/screen_pairs.py already uses (end=latest_available_date()).
    end = today_et.isoformat()
    start = (today_et - timedelta(days=JOHANSEN_HURST_LOOKBACK_CALENDAR_DAYS)).isoformat()
    return fetch_intraday_closes_databento(
        symbols, start, end, sessions=JOHANSEN_HURST_LOOKBACK_TRADING_DAYS, datasets=LIVE_DATASETS)


def _rth_only(bars: pd.DataFrame) -> pd.DataFrame:
    """Regular-hours bars only, no session-count trim -- the shared core
    _previous_and_current_rth builds on (which additionally keeps only the
    last two sessions)."""
    if bars.empty:
        return bars
    ts = pd.to_datetime(bars["ts_event"], utc=True).dt.tz_convert(ET)
    return bars[(ts.dt.time >= RTH_OPEN) & (ts.dt.time < RTH_CLOSE)].copy()


def _rth_only_series(prices: pd.Series) -> pd.Series:
    """Same RTH-hours trim as _rth_only, for a timestamp-indexed price
    Series (e.g. ctx.premarket_price_a/b, which spans PREMARKET_FETCH_START_TIME
    onward) instead of a raw bars DataFrame with its own ts_event column."""
    if prices.empty:
        return prices
    ts = pd.to_datetime(prices.index, utc=True).tz_convert(ET)
    return prices[(ts.time >= RTH_OPEN) & (ts.time < RTH_CLOSE)]


def _previous_and_current_rth(bars: pd.DataFrame) -> pd.DataFrame:
    """Trims a wider fetch down to the spec's window: the previous RTH session
    plus the current one so far, regular hours only.

    The fetch deliberately asks for several calendar days so a weekend or
    holiday can't swallow the previous session; this is what turns that back
    into "previous RTH + now" regardless of which weekday it runs on."""
    rth = _rth_only(bars)
    if rth.empty:
        return rth
    sessions = pd.to_datetime(rth["ts_event"], utc=True).dt.tz_convert(ET).dt.date
    keep = sorted(sessions.unique())[-2:]
    return rth[sessions.isin(keep)]


def _session_date_et(ts_column: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_column, utc=True).dt.tz_convert(ET).dt.date


@dataclass
class _WarmupData:
    pivot: pd.DataFrame
    prior_pivot: pd.DataFrame
    jh_pivot: pd.DataFrame
    # {symbol: true Open print of the prior session's first RTH bar} --
    # see PairContext.prior_session_open_a's own comment for why this
    # can't just be prior_pivot's own first close value.
    prior_open: "dict[str, float]"


def _fetch_warmup_data(contexts: list[PairContext], now: datetime) -> "_WarmupData | None":
    """Pure I/O + computation -- touches no PairContext field. Split out of
    warm_up_pairs 2026-08-28 so run_warmup_retry_in_background can run this
    half on a worker thread (via asyncio.to_thread) without racing
    run_check_cycle's own concurrent in-place mutation of the SAME
    PairContext objects on the event-loop thread. warm_up_pairs' initial
    call (before the loop starts, single-threaded) has no such race, but
    uses this same split for one code path instead of two. Returns None if
    no warm-up data came back at all."""
    # Reach back in CALENDAR days, then keep the last two RTH SESSIONS. The
    # window used to start at "yesterday", which on a Monday is Sunday: the
    # fetch then covered a day with no bars plus today, and Friday -- the
    # actual previous RTH session -- was missed entirely. Same on any day
    # after a holiday. Six days clears a weekend plus a holiday either side.
    # Both bounds in UTC. Databento reads a naive timestamp as UTC, and `now`
    # is ET -- so formatting it directly sent 13:18 ET as 13:18 UTC, i.e.
    # 09:18 ET, and the window ended four hours early. Before ~1:30pm ET that
    # cut off today's session completely, which is why the warm-up looked
    # empty even on days the fetch itself succeeded.
    now_utc = now.astimezone(timezone.utc)
    start = f"{(now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
    end = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
    all_symbols = sorted({s for ctx in contexts for s in (ctx.symbol_a, ctx.symbol_b)})

    bars = _previous_and_current_rth(fetch_intraday_warmup(all_symbols, start, end))
    if bars.empty:
        return None
    pivot = bars.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
    # Previous session only (strictly before today), for the frozen base
    # trigger/half-life -- "historical Trigger is only valid if computed
    # using the frozen baseline beta established at previous close."
    today_et = now.astimezone(ET).date()
    prior_bars = bars[_session_date_et(bars["ts_event"]) < today_et]
    prior_pivot = (prior_bars.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
                   if not prior_bars.empty else pd.DataFrame())
    # True Open, not the close-price proxy -- see PairContext.prior_session_open_a's
    # own comment. sort_values first: groupby.first() takes row order as given,
    # not chronological order, and bars isn't guaranteed sorted.
    prior_open = (prior_bars.sort_values("ts_event").groupby("symbol")["open"].first().to_dict()
                  if not prior_bars.empty else {})

    # Rolling 20 trading day history for the live Johansen/Hurst gate -- see
    # PairContext.johansen_hurst_price_a's own comment and
    # _fetch_johansen_hurst_history. A separate fetch/failure from the
    # divergence warm-up above: a pair still trades on price_history_a/b
    # even if this one comes back empty, it just can't pass the live
    # Johansen/Hurst gate yet (entry_eligible fails closed on that).
    try:
        jh_bars = _fetch_johansen_hurst_history(all_symbols, now)
    except Exception as e:
        logger.warning(f"could not fetch 20 day Johansen/Hurst history: {e}")
        jh_bars = pd.DataFrame()
    jh_pivot = jh_bars if not jh_bars.empty else pd.DataFrame()

    return _WarmupData(pivot=pivot, prior_pivot=prior_pivot, jh_pivot=jh_pivot, prior_open=prior_open)


def _apply_warmup_data(contexts: list[PairContext], data: _WarmupData) -> None:
    """Mutates PairContext fields from already-fetched data -- pure
    in-memory work, no I/O, so it's fast enough to always run directly on
    the event-loop thread (never inside asyncio.to_thread) and never races
    run_check_cycle's own per-cycle mutation of these same fields."""
    for ctx in contexts:
        if ctx.symbol_a not in data.pivot.columns or ctx.symbol_b not in data.pivot.columns:
            continue
        both = data.pivot[[ctx.symbol_a, ctx.symbol_b]].dropna()
        ctx.price_history_a = both[ctx.symbol_a]
        ctx.price_history_b = both[ctx.symbol_b]
        ctx.spread_history = compute_spread(ctx.price_history_a, ctx.price_history_b, ctx.hedge_ratio)

        if ctx.symbol_a in data.jh_pivot.columns and ctx.symbol_b in data.jh_pivot.columns:
            jh_full = data.jh_pivot[[ctx.symbol_a, ctx.symbol_b]].dropna()
            jh_both = jh_full.tail(JOHANSEN_HURST_WINDOW_BARS)
            ctx.johansen_hurst_price_a = jh_both[ctx.symbol_a]
            ctx.johansen_hurst_price_b = jh_both[ctx.symbol_b]
            # Same already-fetched 20 day history, just the trailing 390
            # bars (one trading day) instead of all 7800 -- see
            # JOHANSEN_HURST_WINDOW_1DAY_BARS's own comment.
            jh_1day = jh_full.tail(JOHANSEN_HURST_WINDOW_1DAY_BARS)
            ctx.johansen_hurst_price_a_1day = jh_1day[ctx.symbol_a]
            ctx.johansen_hurst_price_b_1day = jh_1day[ctx.symbol_b]
        else:
            logger.warning(f"{ctx.symbol_a}/{ctx.symbol_b}: no 20 day history for the live "
                            f"Johansen/Hurst gate -- will stay ineligible to enter until enough "
                            f"of today's live bars accumulate")

        if ctx.symbol_a in data.prior_pivot.columns and ctx.symbol_b in data.prior_pivot.columns:
            prior_both = data.prior_pivot[[ctx.symbol_a, ctx.symbol_b]].dropna()
            ctx.prior_session_price_a = prior_both[ctx.symbol_a]
            ctx.prior_session_price_b = prior_both[ctx.symbol_b]
            ctx.prior_session_open_a = data.prior_open.get(ctx.symbol_a)
            ctx.prior_session_open_b = data.prior_open.get(ctx.symbol_b)
            establish_base_trigger(ctx)
            # Per the client 2026-09-11: "Please exclude. They should go
            # straight to Divergence Watch. Waiting for Entry Window."
            # Pairs that already qualified from the prior RTH session and
            # loaded here (the 9am cache, or its live 9:30am fallback --
            # see warm_up_pairs) skip the live 20-day Johansen/Hurst
            # re-validation gate entirely, unlike a pair
            # run_universe_screen_in_background discovers mid-session
            # (that path does NOT set these, so it still earns them live).
            # Reverses the 2026-08-25/08-28/08-29 design where every pair,
            # cache-loaded or not, had to clear both live at least once
            # this session -- a deliberate spec change, not a bug fix.
            ctx.johansen_ever_unconfirmed = True
            ctx.hurst_ever_above_threshold = True
        else:
            logger.warning(f"{ctx.symbol_a}/{ctx.symbol_b}: no previous-session data -- "
                            f"no base trigger established, waiting for the 9:31am exception check")


# Same 60 trading day / 100 calendar day daily-closes window the old offline
# screen used, for the same reason: 100 calendar days comfortably covers 60
# trading days across weekends and holidays.
DAILY_LOOKBACK_TRADING_DAYS = 60
DAILY_LOOKBACK_CALENDAR_DAYS = 100


def select_pairs_from_previous_session(now: datetime, fetch_now: "datetime | None" = None) -> list[PairContext]:
    """Replaces the old load_pair_contexts()/qualified_pairs.json step --
    per the client 2026-08-29: "9:30am, Bot selects the qualified pairs from
    previous RTH session. This will include ALL the ones that already
    qualified before 10:30am, and others that may have qualified by 4pm.
    The selection is previous RTH 9:30-4pm." And, on the wider point: "there
    is nothing offline, that's phase by phase was only for payments, not
    code wise, all things should be wired, nothing should be offline."

    Called live, every session, at SESSION_START_TIME (9:30am ET --
    briefly moved to 9:00am the same day, then reverted per the client
    2026-09-01: "Pair Selection cannot start by 9am. Pair Selection
    starts by 9:30am. We use only RTH Data.") -- never a hand-run script.
    Scoped to the Market Entry List's own pairs only (not a full S&P 500
    rescan like
    universe_screening.build_reference_data/screen_once does for its own,
    separate mid-session discovery job) since the client only ever trades
    pairs already on that list; this keeps the fetch small enough to run
    in seconds, not minutes, so nothing is lost waiting at the start of
    the entry window. Runs evaluate_pair (phase1/screening.py) --
    correlation, Correlation Breakdown, liquidity, market cap, volatility --
    against the full previous RTH session (9:30am-4pm ET, the prior trading
    day) of 1-minute bars plus 60 days of daily consolidated closes.

    Deliberately does NOT check Johansen or Hurst here -- per the client the
    same day: "Then at current RTH session, Bot will validate Johansen and
    Hurst. Retry logic." Those two are exclusively a live, continuously-
    rechecked-during-RTH gate (update_live_johansen_hurst,
    PairContext.johansen_ever_unconfirmed/hurst_ever_above_threshold) that
    starts fresh every session at JOHANSEN_HURST_LIVE_START_TIME -- see
    phase1/screening.py's module docstring for the full reasoning, including
    the confirmed-live 2026-08-29 finding that qualifying on Johansen/Hurst
    here left EVERY pair disqualified, because a real-time snapshot of
    either one isn't the kind of structural property this previous-session
    selection is meant to test.

    A pair still needs to independently clear Johansen/Hurst live today to
    ever actually enter (entry_eligible) -- this function only decides which
    pairs are worth tracking and warming up for that live check in the
    first place. warm_up_pairs (called right after this, same as the old
    load_pair_contexts() -> warm_up_pairs sequence) does the actual
    price/spread/base-trigger/Johansen-Hurst-window seeding for whatever
    this returns.

    `fetch_now` (defaults to `now`) is what the actual data fetches use
    for their end boundary -- kept separate from `now` because
    cache_qualified_pairs_for_tomorrow calls this with `now` pushed one
    day ahead (so the "most recent session before `now`" cutoff below
    picks up TODAY's just-closed session, not yesterday's), while still
    needing the real, current evening timestamp for the fetches
    themselves. Confirmed live 2026-09-01: passing that same
    one-day-ahead value straight through as the fetch boundary too asked
    Databento/yfinance for daily closes and intraday bars up through a
    date that hadn't happened yet -- silently failed "no daily close
    data" for all 124 pairs, caching an empty list instead of a real
    one."""
    fetch_now = fetch_now or now
    market_entry_pairs = load_market_entry_list()
    if not market_entry_pairs:
        logger.warning("no market entry list configured -- no pairs to select from the previous session")
        return []
    permanent_blacklisted = load_permanent_blacklist()
    symbols_needed = sorted({s for pair in market_entry_pairs for s in pair} - permanent_blacklisted)
    if not symbols_needed:
        return []

    # Real GICS sub-industry per symbol, for the post-trade report's Daily
    # Qualifying Pairs table -- a single Wikipedia page fetch (no per-symbol
    # yfinance calls), cheap enough to not matter for the "no minutes lost"
    # requirement, unlike market cap/volume/dividend data below.
    try:
        industry_by_symbol = universe_screening.industry_groups(universe_screening.fetch_sp500_universe())
        industry_by_symbol = {s: industry for industry, symbols in industry_by_symbol.items() for s in symbols}
    except Exception as e:
        logger.warning(f"previous-session pair selection: could not fetch GICS industries, "
                       f"report will show a generic label: {e}")
        industry_by_symbol = {}

    fetch_now_utc = fetch_now.astimezone(timezone.utc)
    start = f"{(fetch_now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
    end = fetch_now_utc.strftime("%Y-%m-%dT%H:%M:%S")
    bars = _previous_and_current_rth(fetch_intraday_warmup(symbols_needed, start, end))
    if bars.empty:
        logger.warning("previous-session pair selection: no price data available, no pairs selected")
        return []
    today_et = now.astimezone(ET).date()
    prior_bars = bars[_session_date_et(bars["ts_event"]) < today_et]
    if prior_bars.empty:
        logger.warning("previous-session pair selection: no prior-session bars available, no pairs selected")
        return []
    prior_closes = prior_bars.set_index(["ts_event", "symbol"])["close"].unstack("symbol")

    from phase1.databento_source import fetch_consolidated_daily_closes
    daily_start = (fetch_now_utc.date() - timedelta(days=DAILY_LOOKBACK_CALENDAR_DAYS)).isoformat()
    daily_end = fetch_now_utc.date().isoformat()
    try:
        # adjust_dividends=False -- confirmed live 2026-08-29: with it on,
        # this fetch does one yfinance dividend lookup per symbol (5 workers)
        # and at the Market Entry List's real scale (~124 pairs) took several
        # minutes and started tripping Yahoo's rate limit, directly against
        # the client's "no minutes lost" requirement for this synchronous
        # 9:30am startup path. The accuracy case for dividend adjustment
        # (databento_source.py's own module docstring) was specifically
        # about a cointegration/Johansen test flipping on ex-dividend noise
        # -- neither is read by evaluate_pair's qualified property anymore
        # (both removed here 2026-08-29), so this fetch now only feeds
        # Pearson correlation / Correlation Breakdown, which a single
        # ex-dividend day's noise is far less likely to flip.
        daily_closes = fetch_consolidated_daily_closes(
            symbols_needed, daily_start, daily_end, adjust_dividends=False
        ).tail(DAILY_LOOKBACK_TRADING_DAYS)
    except Exception as e:
        logger.warning(f"previous-session pair selection: could not fetch daily closes, no pairs selected: {e}")
        return []

    market_caps = fetch_market_caps_yfinance(symbols_needed)
    volume_stats = fetch_volume_stats_yfinance(symbols_needed, daily_closes)

    contexts = []
    not_qualified = []
    for a, b in sorted(market_entry_pairs):
        if a not in prior_closes.columns or b not in prior_closes.columns:
            not_qualified.append(f"{a}/{b} (no previous-session price data)")
            continue
        both = prior_closes[[a, b]].dropna()
        if len(both) < universe_screening.MIN_BARS_TO_TEST:
            not_qualified.append(f"{a}/{b} (too few previous-session bars)")
            continue
        if a not in daily_closes.columns or b not in daily_closes.columns:
            not_qualified.append(f"{a}/{b} (no daily close data)")
            continue
        both_daily = daily_closes[[a, b]].dropna()
        if len(both_daily) < universe_screening.MIN_DAILY_BARS_TO_TEST:
            not_qualified.append(f"{a}/{b} (too few daily closes)")
            continue
        stats_a = volume_stats.get(a, {})
        stats_b = volume_stats.get(b, {})
        pair_industry = industry_by_symbol.get(a) or industry_by_symbol.get(b) or "Market Entry List"
        try:
            result = evaluate_pair(
                a, b, pair_industry, both[a], both[b], both_daily[a], both_daily[b],
                avg_share_volume_a=stats_a.get("avg_share_volume", 0.0),
                avg_dollar_volume_a=stats_a.get("avg_dollar_volume", 0.0),
                market_cap_a=market_caps.get(a, 0.0),
                avg_share_volume_b=stats_b.get("avg_share_volume", 0.0),
                avg_dollar_volume_b=stats_b.get("avg_dollar_volume", 0.0),
                market_cap_b=market_caps.get(b, 0.0),
                size_filters_exempt=is_special_pair(a, b),
            )
        except Exception as e:
            logger.warning(f"previous-session pair selection skip {a}/{b}: {e}")
            not_qualified.append(f"{a}/{b} (evaluation failed)")
            continue
        if not result.qualified:
            not_qualified.append(f"{a}/{b}")
            continue
        contexts.append(PairContext(
            symbol_a=a, symbol_b=b, hedge_ratio=result.hedge_ratio,
            spread_history=pd.Series(dtype=float), industry=result.industry,
            correlation_60d=result.correlation, correlation_20d=result.correlation_recent,
        ))

    if not_qualified:
        logger.info("on the market entry list but did not qualify from the previous session: %s",
                    ", ".join(not_qualified))
    logger.info(f"previous-session pair selection: {len(contexts)} of {len(market_entry_pairs)} "
                f"market entry list pairs qualified")
    return contexts


def _johansen_hurst_ever_cleared_in_prior_session(
    jh_price_a: pd.Series, jh_price_b: pd.Series, prior_session_date: "date",
) -> tuple[bool, bool]:
    """Replays Johansen and Hurst bar-by-bar through prior_session_date's
    own 1-minute bars, using the same sliding JOHANSEN_HURST_WINDOW_BARS
    window update_live_johansen_hurst grows live today -- just walked over
    already-known historical bars instead of a live feed. Returns
    (johansen_ever_unconfirmed, hurst_ever_above_threshold): the same
    "cleared at any point" sticky semantics the live flags use, not a
    single whole-window snapshot test -- select_pairs_from_previous_session's
    own docstring documents why a snapshot over the whole window
    disqualifies nearly every pair (confirmed live 2026-08-29).

    jh_price_a/b must already cover the trailing JOHANSEN_HURST_WINDOW_BARS
    bars before prior_session_date's first bar, plus every bar of
    prior_session_date itself, in chronological order -- exactly what
    _fetch_johansen_hurst_history(symbols, jh_as_of) returns, where
    jh_as_of is derived from prior_session_date itself (see
    select_pairs_from_full_universe_prior_session's own comment on why
    neither `now` nor `fetch_now` works here), same as today's live
    warm-up uses its own real `now`.

    Stops early the moment both flags are True -- most of the replay cost
    is coint_johansen (the expensive part), and once a pair has cleared
    both at least once nothing later in the session can un-clear it."""
    session_dates = _session_date_et(pd.Series(jh_price_a.index))
    session_positions = [i for i, d in enumerate(session_dates) if d == prior_session_date]
    ever_unconfirmed = False
    ever_above = False
    for pos in session_positions:
        window_start = max(0, pos + 1 - JOHANSEN_HURST_WINDOW_BARS)
        window_a = jh_price_a.iloc[window_start:pos + 1]
        window_b = jh_price_b.iloc[window_start:pos + 1]
        if len(window_a) < MIN_BARS_FOR_LIVE_JOHANSEN_HURST:
            continue
        try:
            confirmed, hurst = _compute_johansen_hurst(window_a, window_b)
        except Exception:
            continue  # same per-bar tolerance update_live_johansen_hurst's own try/except gives live
        if not confirmed:
            ever_unconfirmed = True
        if hurst is not None and hurst > MIN_HURST:
            ever_above = True
        if ever_unconfirmed and ever_above:
            break
    return ever_unconfirmed, ever_above


def select_pairs_from_full_universe_prior_session(
    now: datetime, fetch_now: "datetime | None" = None,
) -> list[PairContext]:
    """Full S&P 500 scan of the previous RTH session, for the 9am cache --
    per the client 2026-09-11/12, ending a long same-night back-and-forth:
    "Apart from the market entry list, the bot is supposed to do the full
    S&P 500 scan prior RTH and see which pairs qualified Pearson, Johansen
    and Hurst, at some point in time during 9:30-4pm... It will compile a
    list. And load up that list by 9am... It has nothing to do with
    Market Entry List." "I never told you to scan my Market Entry list."

    Called ONLY from cache_qualified_pairs_for_tomorrow (once daily, well
    after today's own close and report) -- select_pairs_from_previous_session
    stays completely unchanged and is still what the live 9:30am path falls
    back to if this cache is unexpectedly missing/stale, since THAT path
    has to run in seconds during the live entry window and this one does
    not (see that function's own docstring on why speed matters there).

    Candidate pairs are universe_screening.candidate_pairs's own
    same-industry combinations across the full S&P 500, unioned with the
    Market Entry List -- hundreds of pairs, not the ~11-30 the Market
    Entry List alone produces.

    Two stages, both real cost, the same shape evaluate_pair/screen_once
    already use elsewhere and already run live, continuously, against the
    same full candidate set (see run_universe_screen_in_background's own
    docstring: "runs ADF/Johansen/Hurst on every untracked candidate
    pair"). Stage 1 is NOT cheap by itself -- evaluate_pair runs its own
    ADF/Johansen/Hurst/half-life battery unconditionally, before even
    checking correlation, so it costs roughly the same per candidate as
    one bar of stage 2's replay. What stage 1 actually buys is narrowing
    hundreds of candidates down to a handful BEFORE stage 2's up-to-390-bar
    replay runs at all -- stage 2 (_johansen_hurst_ever_cleared_in_prior_session)
    is the part that would be unaffordable across the full candidate set,
    not stage 1. This only runs once daily, after hours, unlike
    screen_once's live intraday cadence, so the total cost here is lower
    even though stage 1 alone isn't cheap.

    A pair qualifies here only if BOTH Johansen and Hurst cleared at some
    point during the prior session -- see that function's own docstring."""
    fetch_now = fetch_now or now
    ref = universe_screening.build_reference_data()
    market_entry_pairs = load_market_entry_list() or []
    permanent_blacklisted = load_permanent_blacklist()
    candidates = {
        (a, b) for a, b in universe_screening.candidate_pairs(ref, market_entry_pairs)
        if a not in permanent_blacklisted and b not in permanent_blacklisted
    }
    if not candidates:
        logger.warning("full-universe prior-session selection: no candidate pairs, none selected")
        return []

    fetch_now_utc = fetch_now.astimezone(timezone.utc)
    start = f"{(fetch_now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
    end = fetch_now_utc.strftime("%Y-%m-%dT%H:%M:%S")
    bars = _previous_and_current_rth(fetch_intraday_warmup(ref.all_symbols, start, end))
    if bars.empty:
        logger.warning("full-universe prior-session selection: no price data available, none selected")
        return []
    today_et = now.astimezone(ET).date()
    prior_bars = bars[_session_date_et(bars["ts_event"]) < today_et]
    if prior_bars.empty:
        logger.warning("full-universe prior-session selection: no prior-session bars, none selected")
        return []
    prior_closes = prior_bars.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
    prior_session_date = _session_date_et(prior_bars["ts_event"]).max()

    stage1_survivors = []
    for a, b in sorted(candidates):
        if a not in prior_closes.columns or b not in prior_closes.columns:
            continue
        both = prior_closes[[a, b]].dropna()
        if len(both) < universe_screening.MIN_BARS_TO_TEST:
            continue
        if a not in ref.daily_closes.columns or b not in ref.daily_closes.columns:
            continue
        both_daily = ref.daily_closes[[a, b]].dropna()
        if len(both_daily) < universe_screening.MIN_DAILY_BARS_TO_TEST:
            continue
        stats_a = ref.volume_stats.get(a, {})
        stats_b = ref.volume_stats.get(b, {})
        pair_industry = ref.industry_by_symbol.get(a) or ref.industry_by_symbol.get(b) or "Full Universe Scan"
        try:
            result = evaluate_pair(
                a, b, pair_industry, both[a], both[b], both_daily[a], both_daily[b],
                avg_share_volume_a=stats_a.get("avg_share_volume", 0.0),
                avg_dollar_volume_a=stats_a.get("avg_dollar_volume", 0.0),
                market_cap_a=ref.market_caps.get(a, 0.0),
                avg_share_volume_b=stats_b.get("avg_share_volume", 0.0),
                avg_dollar_volume_b=stats_b.get("avg_dollar_volume", 0.0),
                market_cap_b=ref.market_caps.get(b, 0.0),
                size_filters_exempt=is_special_pair(a, b),
            )
        except Exception as e:
            logger.debug(f"full-universe prior-session selection skip {a}/{b}: {e}")
            continue
        if result.qualified:
            stage1_survivors.append((a, b, result))
    logger.info(f"full-universe prior-session selection: {len(stage1_survivors)} of "
                f"{len(candidates)} candidates cleared Pearson/liquidity, replaying Johansen/Hurst")
    if not stage1_survivors:
        return []

    jh_symbols = sorted({s for a, b, _ in stage1_survivors for s in (a, b)})
    # Neither `now` (pseudo_tomorrow) nor `fetch_now` directly -- both were
    # tried and both broke this call in opposite ways. `now` (confirmed
    # real, 2026-09-14 evening) pushed the exclusive `end` a full day past
    # prior_session_date, reaching into data Databento didn't have ready
    # yet mere minutes after close -- the whole batch came back empty
    # ("empty Johansen/Hurst history, none selected"), zeroing out that
    # night's cache. `fetch_now` (confirmed real, 2026-09-15 evening, the
    # very next night) swung the other way: its own `today_et` no longer
    # matches this function's `today_et` (which is derived from `now`), so
    # the exclusive `end` landed ON prior_session_date instead of past it
    # -- prior_session_date's own bars never made it into jh_bars at all,
    # so every single candidate's replay found zero bars for that date and
    # silently reported "never cleared together" across the board, a
    # WORSE bug than the first (deterministic every night, not just an
    # occasional data-lag race). The one boundary that's actually correct
    # for what this fetch needs -- through the end of prior_session_date,
    # whatever the real wall-clock time is when this runs -- is derived
    # directly from prior_session_date itself, already known at this
    # point, not from either time reference.
    jh_as_of = datetime.combine(prior_session_date + timedelta(days=1), dtime(0, 0), tzinfo=ET)
    try:
        jh_bars = _fetch_johansen_hurst_history(jh_symbols, jh_as_of)
    except Exception as e:
        logger.warning(f"full-universe prior-session selection: could not fetch 20 day "
                        f"Johansen/Hurst history, none selected: {e}")
        return []
    if jh_bars.empty:
        logger.warning("full-universe prior-session selection: empty Johansen/Hurst history, none selected")
        return []

    contexts = []
    not_qualified = []
    for a, b, result in stage1_survivors:
        if a not in jh_bars.columns or b not in jh_bars.columns:
            not_qualified.append(f"{a}/{b} (no 20 day Johansen/Hurst history)")
            continue
        jh_both = jh_bars[[a, b]].dropna()
        ever_unconfirmed, ever_above = _johansen_hurst_ever_cleared_in_prior_session(
            jh_both[a], jh_both[b], prior_session_date)
        if not (ever_unconfirmed and ever_above):
            not_qualified.append(f"{a}/{b} (Johansen/Hurst never cleared together prior session)")
            continue
        contexts.append(PairContext(
            symbol_a=a, symbol_b=b, hedge_ratio=result.hedge_ratio,
            spread_history=pd.Series(dtype=float), industry=result.industry,
            correlation_60d=result.correlation, correlation_20d=result.correlation_recent,
        ))

    if not_qualified:
        logger.info("cleared Pearson/liquidity but not Johansen/Hurst during the prior session: %s",
                    ", ".join(not_qualified))
    logger.info(f"full-universe prior-session selection: {len(contexts)} pairs qualified "
                f"(Pearson, Johansen and Hurst, at some point 9:30am-4pm the prior session)")
    return contexts


def cache_qualified_pairs_for_tomorrow(now: datetime) -> None:
    """Runs once, automatically, at the end of each session -- per the
    client 2026-09-01: "There is no pairs selection today... If from 9am,
    it has already loaded them all, by 9:30am, it will test for Johansen
    and Hurst. Which is a matter of seconds." Computes tomorrow's list
    right now, while today's just-closed RTH session is the freshest
    "previous session" there is, and persists it -- so tomorrow's 9am
    load gets an already-computed list instead of running a live fetch
    from scratch.

    Calls select_pairs_from_full_universe_prior_session (the full S&P 500
    scan, see its own docstring) since 2026-09-12, per the client -- NOT
    select_pairs_from_previous_session, which stays as the fast,
    Market-Entry-List-scoped live 9:30am fallback for when this cache is
    unexpectedly missing (see its own docstring on why speed matters only
    there).

    Passes a `now` one calendar day ahead so the "most recent session
    strictly before this date" logic picks up TODAY's session (which just
    closed) rather than the day before it. Works the same across a
    weekend or holiday too, since it's still just "the most recent
    session before this date" either way.

    Passes the REAL current time as `fetch_now` -- see that parameter's
    own comment on select_pairs_from_previous_session (identical
    reasoning applies here). Confirmed live 2026-09-01: without this, the
    pseudo-tomorrow date leaked into the actual data fetch boundaries too
    and every single fetch came back empty ("no daily close data" for all
    124 pairs), silently caching an empty list instead of a real one.

    Wrapped in try/except -- a failure here must never take down the live
    trading day (this runs after EOD flatten and the report, see its own
    call site). load_cached_qualified_pairs falls back to a live,
    same-morning select_pairs_from_previous_session() call if this cache
    ends up missing, stale, or empty, exactly like the pre-caching
    behavior."""
    try:
        pseudo_tomorrow = datetime.combine(now.date() + timedelta(days=1), dtime(0, 0), tzinfo=ET)
        if SINGLE_STOCK_EXCURSION_MODE:
            ref = universe_screening.build_reference_data()
            fetch_now_utc = now.astimezone(timezone.utc)
            start = f"{(fetch_now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
            end = fetch_now_utc.strftime("%Y-%m-%dT%H:%M:%S")
            bars = _previous_and_current_rth(fetch_intraday_warmup(ref.all_symbols, start, end))
            candidates = compute_single_stock_watchlist(ref.all_symbols, bars)
            save_single_stock_watchlist(candidates, for_date=now.astimezone(ET).date())
            logger.info(f"Cached {len(candidates)} single-stock candidates for tomorrow's excursion trading")
            return
        contexts = select_pairs_from_full_universe_prior_session(pseudo_tomorrow, fetch_now=now)
        payload = json.dumps({
            "for_date": now.astimezone(ET).date().isoformat(),
            "pairs": [
                {"symbol_a": ctx.symbol_a, "symbol_b": ctx.symbol_b,
                 "hedge_ratio": ctx.hedge_ratio, "industry": ctx.industry,
                 "correlation_60d": ctx.correlation_60d, "correlation_20d": ctx.correlation_20d}
                for ctx in contexts
            ],
        })
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHED_QUALIFIED_PAIRS_PATH.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, CACHED_QUALIFIED_PAIRS_PATH)
        logger.info(f"cached {len(contexts)} qualified pairs for tomorrow's 9:30am load")
        write_cached_pairs_status(contexts, now)
    except Exception:
        logger.exception("could not cache tomorrow's qualified pairs -- tomorrow will fall back "
                          "to a live select_pairs_from_previous_session() call instead")


def write_cached_pairs_status(contexts: "list[PairContext]", now: datetime) -> None:
    """Writes tonight's overnight cache result to its own small file the
    moment it's ready -- per the client repeatedly asking "how many pairs will
    load by 9am" -- so the dashboard can show tomorrow's count as soon as
    the scan finishes (this runs right after today's 4pm close, see
    cache_qualified_pairs_for_tomorrow's own call site), not only after
    tomorrow's live session actually starts and loads it.

    Deliberately NOT merged into bot_status.json: that file is treated as
    stale (the bot considered offline) once it's more than 30s old (see
    status.read_status_file's own docstring), and this write happens well
    after the process's last real write_status() call -- often much more
    than 30s later, since the full S&P 500 scan this follows is genuinely
    CPU-bound. Piggybacking on that file would make this information
    disappear again within moments of appearing, the opposite of the
    point. This file has no such staleness check; the dashboard reads it
    unconditionally, live or not.

    Best-effort, same as save_daily_state: a failure here must never take
    down the overnight caching this runs right after."""
    try:
        payload = json.dumps({
            "cached_pairs_for_tomorrow_count": len(contexts),
            "cached_pairs_for_tomorrow_symbols": [
                f"{ctx.symbol_a}/{ctx.symbol_b}" for ctx in contexts],
            "cached_pairs_for_tomorrow_at": now.astimezone(ET).isoformat(),
        })
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHED_PAIRS_STATUS_PATH.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, CACHED_PAIRS_STATUS_PATH)
    except OSError:
        logger.warning("could not write cached-pairs status update", exc_info=True)


def load_cached_qualified_pairs(now: datetime) -> "list[PairContext] | None":
    """Loads cache_qualified_pairs_for_tomorrow's result from last night --
    returns None for anything that means this cache can't be trusted:
    missing file, corrupt JSON, a for_date that isn't a recent enough
    prior trading day, or an empty pairs list. The caller
    (run_daily_session) falls back to a live
    select_pairs_from_previous_session() call in every one of those cases
    -- exactly the pre-caching behavior, so a missed or failed evening
    cache run costs ~70-90 seconds the next morning, never a silently
    wrong or empty pair list.

    Empty is treated as untrustworthy, not "correctly computed, zero
    pairs qualified" -- confirmed live 2026-09-01: a real bug in the
    evening fetch (see select_pairs_from_previous_session's `fetch_now`
    comment) produced exactly that signature, 0 of 124 pairs, silently.
    With 124 established Market Entry List pairs, a genuine zero-pair day
    would be extraordinary; treating it as suspicious and falling back to
    a live re-check costs one ~70-90s morning, which is far cheaper than
    trading nothing all day over an evening fetch glitch."""
    if not CACHED_QUALIFIED_PAIRS_PATH.exists():
        return None
    try:
        data = json.loads(CACHED_QUALIFIED_PAIRS_PATH.read_text())
        for_date = date.fromisoformat(data["for_date"])
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
        logger.warning(f"{CACHED_QUALIFIED_PAIRS_PATH} is unreadable ({e}) -- falling back to a live selection")
        return None
    today_et = now.astimezone(ET).date()
    # Invalid if for_date is in the future (a corrupt or hand-edited
    # file), or stale if it's older than a long weekend can explain.
    # for_date == today_et is valid, not stale -- confirmed live
    # 2026-09-02: a same-morning manual re-run to recover from the
    # previous evening's fetch bug (see cache_qualified_pairs_for_tomorrow's
    # `fetch_now` comment) legitimately produces this, and it's the
    # freshest a cache can possibly be, not the oldest.
    if for_date > today_et or (today_et - for_date).days > 4:
        logger.warning(f"cached qualified pairs are for {for_date}, not usable for {today_et} -- "
                        f"falling back to a live selection")
        return None
    if not data["pairs"]:
        logger.warning(f"cached qualified pairs for {for_date} is empty -- treating as untrustworthy, "
                        f"falling back to a live selection")
        return None
    return [
        PairContext(symbol_a=p["symbol_a"], symbol_b=p["symbol_b"], hedge_ratio=p["hedge_ratio"],
                    spread_history=pd.Series(dtype=float), industry=p.get("industry", ""),
                    # .get, not [] -- a cache written before these two fields
                    # existed still loads, just without them.
                    correlation_60d=p.get("correlation_60d"), correlation_20d=p.get("correlation_20d"))
        for p in data["pairs"]
    ]


def warm_up_pairs(contexts: list[PairContext], now: datetime) -> None:
    """Seeds each pair's price/spread history (previous RTH session + today
    so far, for live trading) and establishes the frozen base trigger level
    and half-life from the previous RTH session ALONE (see
    establish_base_trigger). Called once, synchronously, before the main
    loop starts -- see _fetch_warmup_data/_apply_warmup_data's own docstrings
    for why this is split into a fetch half and an apply half."""
    data = _fetch_warmup_data(contexts, now)
    if data is None:
        logger.warning("no warm-up data available -- pairs will stay inactive until live bars arrive")
        return
    _apply_warmup_data(contexts, data)


async def run_warmup_retry_in_background(missing: list[PairContext], now: datetime, retry_state: dict) -> None:
    """Fired via create_task, same reasoning as run_intraday_scan_in_background:
    the Databento historical fetch is slow enough that doing it inline would
    delay the 1-minute check_exit/entry loop.

    Only the FETCH (asyncio.to_thread(_fetch_warmup_data, ...)) runs on a
    worker thread -- the APPLY (_apply_warmup_data, mutating these
    PairContext objects) runs back on the event-loop thread after the
    await, never inside the thread. Confirmed live 2026-08-28 audit: an
    earlier version ran warm_up_pairs (fetch + apply together) inside
    asyncio.to_thread, which meant the apply half raced run_check_cycle's
    own concurrent in-place mutation of these SAME already-shared
    PairContext objects on the event-loop thread -- with the fetch
    observed taking 30+ seconds, a real window for run_check_cycle to read
    ctx.price_history_a already reassigned by the retry thread while
    ctx.price_history_b still pointed at the old series, feeding
    compute_spread a mismatched pair. _apply_warmup_data has no I/O and no
    `await` inside it, so once called on the event-loop thread it cannot be
    preempted by run_check_cycle mid-mutation.

    Only re-runs warm-up for pairs still missing price_history_a/b -- see
    WARMUP_RETRY_INTERVAL's own comment for why this exists. Isolated in
    its own try/except so a fetch failure here just gets retried next
    interval instead of ever reaching the main loop."""
    retry_state["running"] = True
    try:
        data = await asyncio.to_thread(_fetch_warmup_data, missing, now)
        if data is None:
            logger.warning("warm-up retry: no data available this attempt, will try again next interval")
            return
        _apply_warmup_data(missing, data)
        # Identity, not equality -- PairContext's dataclass-generated __eq__
        # compares every field including the pd.Series ones, which raises on
        # the ambiguous truth value of a Series comparison.
        recovered = [c for c in missing if len(c.price_history_a) > 0 and c.base_trigger_level is not None]
        still_missing = [c for c in missing if len(c.price_history_a) == 0 or c.base_trigger_level is None]
        if recovered:
            logger.info(f"warm-up retry recovered price data for: "
                        f"{', '.join(f'{c.symbol_a}/{c.symbol_b}' for c in recovered)}")
        if still_missing:
            logger.warning(f"warm-up retry still missing price data for: "
                            f"{', '.join(f'{c.symbol_a}/{c.symbol_b}' for c in still_missing)}")
    except Exception:
        logger.warning("warm-up retry failed, will try again next interval", exc_info=True)
    finally:
        retry_state["running"] = False


def _bounded_trigger_level(level: float | None) -> float | None:
    """A trigger outside [MIN_ENTRY_DIVERGENCE, MAX_ENTRY_DIVERGENCE] can
    never actually be entered, so treating it as a live candidate is
    misleading -- applied identically wherever a trigger level is derived
    (the base level here, and each cycle's exception level in
    recalculate_trigger_exceptions).

    REINSTATED the lower bound 2026-09-12, per the client, ending a long
    back-and-forth the same night: "Maximum Divergence is 1.5-5... 1.5-5
    range applies only to the trigger. I never said live current price
    must be 1.5-5. Live Current price should be inside the Entry Window,
    which is derived from the Maximum Divergence, obtained yesterday." So
    the Minimum/Maximum Entry Divergence range gates the TRIGGER alone --
    entry_eligible's separate abs_deviation < MIN_ENTRY_DIVERGENCE /
    > MAX_ENTRY_DIVERGENCE checks are REMOVED (see its own comment), not
    just left in place alongside this. This reverses the 2026-09-04
    removal that used to live in this docstring ("It can only reject
    anything above 5. Not below") -- the client's own words, on both
    dates, direct and quoted, not an inference either time."""
    if level is None:
        return None
    if level <= 0 or level < MIN_ENTRY_DIVERGENCE or level > MAX_ENTRY_DIVERGENCE:
        return None
    return level


def establish_base_trigger(ctx: PairContext) -> None:
    """The frozen base trigger and half-life, computed ONCE from the
    previous RTH session alone (through its 4pm close).

    No longer requires the divergence to have reverted within 2 hours --
    per the client 2026-09-02, direct and repeated correction: "I told
    you there is no reversal inside 2 hours. No more half-life... We
    just need the maximum divergence in the lookback period. Simple.
    There is no reversal involved anywhere." Confirmed live the same
    day: NOW/TEAM had cleared Johansen and Hurst but never got a trigger
    at all, because the old "MEAN REVERTED INSIDE 2 HOURS" rule
    (2026-08-24's "Critical Checklist") discarded every divergence event
    that didn't fully revert within the window, and this pair apparently
    had none that did. The level is now simply the largest
    |spread - equilibrium| reached anywhere in the prior RTH session,
    full stop -- no event detection, no reversion requirement.
    "historical Trigger is only valid if computed using the frozen
    baseline beta established at previous close."

    ctx.divergence_events is still populated from
    detect_divergence_events for whatever else reads it (e.g.
    passes_gap_condition, itself dormant -- see its own call site's
    comment) -- only the trigger LEVEL computation changed here.

    May be overridden upward by recalculate_trigger_exceptions's exception
    check, but is never itself recomputed again this session -- the fix for the
    GE/RTX incident, where a half-life recomputed live every cycle from the
    whole growing intraday series drifted mid-day and disqualified an
    already-qualified pair."""
    if len(ctx.prior_session_price_a) < 2:
        return
    prior_spread = compute_spread(ctx.prior_session_price_a, ctx.prior_session_price_b, ctx.hedge_ratio)
    prior_equilibrium = equilibrium_spread(prior_spread)
    ctx.divergence_events = detect_divergence_events(prior_spread, prior_equilibrium)
    deviation = (prior_spread - prior_equilibrium).dropna()
    level = _bounded_trigger_level(deviation.abs().max() if not deviation.empty else None)
    ctx.base_trigger_level = level
    ctx.divergence_trigger_level = level
    # Checklist condition A, initial value from the prior session alone --
    # see PairContext.mean_reverted_since_max_divergence's own comment.
    ctx.mean_reverted_since_max_divergence = reverted_after_peak(prior_spread, prior_equilibrium)
    # Distinct >50% condition, same initial-value pattern -- see
    # PairContext.reverted_over_50pct_since_max_divergence's own comment.
    ctx.reverted_over_50pct_since_max_divergence = reverted_more_than_pct_after_peak(
        prior_spread, prior_equilibrium)
    if level is None:
        return
    ctx.half_life_minutes = half_life_of_mean_reversion(prior_spread)
    # Half-life < 60 minutes REMOVED as a gate here 2026-08-25, per the
    # client: "Remove Half-Life, truly this time around" -- this was a
    # SEPARATE live gate from Phase 1's screening.py half-life test (removed
    # earlier the same day), sharing the name but not the code path, which
    # is why "Half-life removed" was communicated as done while this one
    # kept silently blocking trigger establishment for any pair whose
    # half-life came in >= 60 minutes (confirmed live: GE/RTX at 69.7min,
    # ANET/LITE at 87.4min, both blocked from ever reaching Divergence
    # Watch). Still computed and kept on ctx.half_life_minutes for
    # visibility, just no longer read here.
    logger.info(f"Base trigger level established for {ctx.symbol_a}/{ctx.symbol_b}: {level:.4f}, "
                f"half-life={ctx.half_life_minutes:.2f}min")


def update_current_divergence_event(
    ctx: PairContext, deviation: float, price_a: float, price_b: float,
    now: datetime, tolerance: float = DEFAULT_DIVERGENCE_TOLERANCE,
) -> None:
    """Tracks the CURRENT live divergence event's own start -- per the
    client 2026-09-03, Expansion_Leg.pdf: the Expansion Leg formula's
    baseline (t0) is "when the divergence began expanding," confirmed
    explicitly the same day as "when the current divergence event started
    moving away from equilibrium," not the whole session or some other
    reference point.

    Same open/close transition detect_divergence_events' own state
    machine uses (|spread - equilibrium| crossing the tolerance band),
    just tracked live here bar by bar instead of reconstructed after the
    fact from a historical series. Sets ctx.current_event_start_time and
    the baseline price of BOTH legs the moment the spread first leaves
    the tolerance band; clears all three back to None the moment it
    returns inside the band, so a NEW event starting later gets its own
    fresh baseline rather than reusing a stale one from a prior,
    already-closed event."""
    if abs(deviation) <= tolerance:
        ctx.current_event_start_time = None
        ctx.current_event_price_a = None
        ctx.current_event_price_b = None
        return
    if ctx.current_event_start_time is None:
        ctx.current_event_start_time = pd.Timestamp(now)
        ctx.current_event_price_a = price_a
        ctx.current_event_price_b = price_b


# Per the client 2026-09-13: "Selected Stock must be responsible for =>70%
# of the Divergence Expansion." See expansion_leg_decision's own comment.
DRIVER_STOCK_MIN_SHARE = 0.70
# Per the client 2026-09-15: "Inside 13%-49.9% Entry Window, 70% or more
# of the Entry Stock High-Low will be in the direction of the prior RTH
# highest excursion from open." A directional continuation filter on the
# driver stock's own OHLC -- see passes_prior_excursion_direction_filter's
# own docstring for the exact formula, reproduced against the client's
# own worked example (270.48/272.23/262.68 prior, 252/253/249 today ->
# 75%, PASS at this threshold).
PRIOR_EXCURSION_DIRECTION_MIN_SHARE = 0.70


from phase2.single_stock_excursion import (
    SINGLE_STOCK_EXCURSION_MODE,
    SingleStockCandidate,
    SingleStockContext,
    compute_single_stock_watchlist,
    load_single_stock_watchlist,
    save_single_stock_watchlist,
    run_single_stock_check_cycle,
    passes_prior_excursion_direction_filter,
    passes_prior_excursion_dominance_filter,
)


def expansion_leg_decision(
    ctx: PairContext, price_a: float, price_b: float,
) -> tuple[str, str, float] | None:
    """Method 1 (Mathematical Return Attribution) from the client's
    2026-09-03 Expansion_Leg.pdf, the client's own worked example
    verified exactly: decomposes the pair's spread move since the
    baseline into each leg's individual contribution, and returns which
    ONE stock is "responsible" and which direction to trade it -- this
    replaces the old two-leg hedged entry entirely ("You don't need
    Hedge Ratio anymore").

        delta_P_A = price_A_now - price_A_baseline
        delta_P_B = price_B_now - price_B_baseline
        C_A = delta_P_A
        C_B = -hedge_ratio * delta_P_B

    Whichever |C| is larger is the dominant driver (the expansion leg).
    BUY it if it rallied (delta_P > 0), SELL it if it crashed
    (delta_P < 0) -- same "ride the momentum, don't fade it" direction
    as trade_direction, just applied to one leg instead of both.

    BASELINE changed 2026-09-13, per the client (Mathematical_Return_Attribution.pdf
    and the BUY/SELL determination PDF, confirmed explicitly the same
    day): "The Rolling Equilibrium Price computed across yesterday's 390
    RTH bars plus today's current RTH bars replaces the starting-price
    baseline instruction from 2026-09-03." Each leg's OWN baseline is now
    the mean of its own price over ctx.price_history_a/b (already exactly
    "yesterday's full RTH session + today's RTH so far, joined together" --
    no separate fetch needed). REPLACES the 2026-09-03 "current divergence
    event's own start price" baseline (ctx.current_event_price_a/b) --
    that field/update_current_divergence_event are left dormant, not
    deleted, same convention as every other superseded mechanism in this
    file. A simple mean, not a literal Kalman filter, mirroring
    spread_engine.equilibrium_spread's own "rolling mean, not a Kalman
    filter" precedent for the analogous spread-level equilibrium --
    flagged as an assumption, correctable if a literal Kalman-filtered
    per-leg price was actually meant.

    Per the client 2026-09-04's "reference leg" beta-normalization
    ("Find out which Leg has beta >1. Make that Leg one by dividing both
    legs with its beta"): NOT applied here, deliberately. In this
    convention (Spread = P_A - hedge_ratio*P_B), leg A's coefficient is
    always 1 and leg B's is hedge_ratio, so leg B is the only one that
    can be the ">1" leg, and dividing both legs by hedge_ratio scales
    both contributions by the SAME positive constant -- a uniform
    rescale that changes neither which |C| is larger nor their percentage
    split (still 78.4%/21.6% for the client's own worked example either
    way). Flagged to the client 2026-09-04 as mathematically inert here;
    not yet confirmed which part of the system it's actually meant for.

    Per the client 2026-09-13 (the BUY/SELL determination PDF, confirmed
    the same day): "Selected Stock must be responsible for =>70% of the
    Divergence Expansion." A NEW threshold -- previously the larger of
    the two |C| values won outright, however close the split (a 51/49
    split still traded). Share = the dominant leg's own |contribution|
    divided by the SUM of both |contributions| -- returns None (no clear
    enough driver to trade) when that share is under
    DRIVER_STOCK_MIN_SHARE, same "nothing to measure from yet" treatment
    as the no-price-history case below.

    Returns (symbol, action, delta_p) for the dominant leg, or None if
    there's no price history yet to compute a baseline from (warm-up
    hasn't seeded price_history_a/b for this pair yet), the dominant
    leg's share is under DRIVER_STOCK_MIN_SHARE, or the driver stock
    fails the prior-excursion-dominance or prior-excursion-direction
    filters (see passes_prior_excursion_dominance_filter/
    passes_prior_excursion_direction_filter's own docstrings)."""
    if len(ctx.price_history_a) == 0 or len(ctx.price_history_b) == 0:
        return None
    baseline_a = ctx.price_history_a.mean()
    baseline_b = ctx.price_history_b.mean()
    delta_p_a = price_a - baseline_a
    delta_p_b = price_b - baseline_b
    contribution_a = delta_p_a
    contribution_b = -ctx.hedge_ratio * delta_p_b
    total_contribution = abs(contribution_a) + abs(contribution_b)
    if total_contribution == 0:
        return None
    if abs(contribution_a) >= abs(contribution_b):
        symbol, delta_p, share = ctx.symbol_a, delta_p_a, abs(contribution_a) / total_contribution
    else:
        symbol, delta_p, share = ctx.symbol_b, delta_p_b, abs(contribution_b) / total_contribution
    if share < DRIVER_STOCK_MIN_SHARE:
        return None
    action = "BUY" if delta_p > 0 else "SELL"
    # Per the client 2026-09-15: "Inside 13%-49.9% Entry Window, 70% or
    # more of the Entry Stock High-Low will be in the direction of the
    # prior RTH highest excursion from open." Checked here, not earlier
    # in entry_eligible, for the same reason DRIVER_STOCK_MIN_SHARE just
    # above is checked here -- both need to already know which stock is
    # the driver, which isn't decided until this point. See
    # passes_prior_excursion_direction_filter's own docstring for the
    # exact formula.
    driver_prior = ctx.prior_session_price_a if symbol == ctx.symbol_a else ctx.prior_session_price_b
    driver_prior_open = ctx.prior_session_open_a if symbol == ctx.symbol_a else ctx.prior_session_open_b
    if len(driver_prior) == 0 or driver_prior_open is None:
        return None
    prior_open, prior_high, prior_low = driver_prior_open, driver_prior.max(), driver_prior.min()
    # Per the client 2026-09-15: "For the stock responsible for 70% of the
    # Divergence Expansion, IF its prior RTH highest excursion is Long, it
    # must =>300% greater than the maximum Short excursion, prior RTH."
    # Checked first since it only needs yesterday's data (driver_prior),
    # not today's -- see passes_prior_excursion_dominance_filter's own
    # docstring for the exact formula.
    if not passes_prior_excursion_dominance_filter(
        prior_open=prior_open, prior_high=prior_high, prior_low=prior_low,
    ):
        return None
    driver_today_premarket = ctx.premarket_price_a if symbol == ctx.symbol_a else ctx.premarket_price_b
    driver_today_open = ctx.today_session_open_a if symbol == ctx.symbol_a else ctx.today_session_open_b
    driver_today = _rth_only_series(driver_today_premarket)
    if len(driver_today) == 0 or driver_today_open is None:
        return None
    if not passes_prior_excursion_direction_filter(
        prior_open=prior_open, prior_high=prior_high, prior_low=prior_low,
        today_open=driver_today_open, today_high=driver_today.max(), today_low=driver_today.min(),
    ):
        return None
    return symbol, action, delta_p


def _traded_leg_estimate(ctx: PairContext, signal: PairSignal) -> "tuple[str, str, float] | None":
    """expansion_leg_decision plus its traded_price lookup, factored out so
    the Adjusted Spread Efficiency ranking below (main entry flow still has
    its own separate, fresh call to expansion_leg_decision right before the
    real entry -- this is only for ranking, never what actually gets
    traded) can compute it for every eligible candidate, not just the one
    that ends up chosen. Returns None the same way expansion_leg_decision
    does (no current divergence event baseline yet) or if there's no live
    price for the dominant leg."""
    price_a, price_b = ctx.price_history_a.iloc[-1], ctx.price_history_b.iloc[-1]
    decision = expansion_leg_decision(ctx, price_a, price_b)
    if decision is None:
        return None
    traded_symbol, action, _ = decision
    traded_price = signal.price_a if traded_symbol == ctx.symbol_a else signal.price_b
    if not traded_price or traded_price <= 0:
        return None
    return traded_symbol, action, traded_price


async def _probe_stock_initial_margin_per_share(
    ibkr: IBKRClient, ctx: PairContext, signal: PairSignal,
    traded_symbol: str, action: str, traded_price: float, sizing_elv: float, now: datetime,
) -> float:
    """Real-order-probed Stock Initial Margin, per share -- feeds Adjusted
    Spread Efficiency ranking (select_highest_adjusted_spread_efficiency,
    risk_manager.py) when 2+ pairs are eligible in the same check cycle.

    Three tiers, in order, first one that actually gets rejected wins:

    1. quantity=1 -- per the client 2026-09-12: "If you use $1 to buy 1
       share, it will get rejected... you don't have to guess a
       quantity. When it gets rejected, you will see the margin required
       to buy one share of the stock." Tried first, exactly as
       instructed.
    2. The same quantity this pair's real entry would start from if it
       wins (compute_final_single_stock_quantity, the existing initial
       sizing estimate) -- probe_bracket_initial_margin's own comment
       confirms IBKR's margin calc is quantity x price x rate, independent
       of the order's limit price, so 1 share's real economic value is a
       tiny fraction of any real account's buying power and is expected
       to fit (return None, not the per-share figure) in the ordinary
       case -- this tier is what will actually produce a real number most
       of the time.
    3. INITIAL_MARGIN_FRACTION x traded_price -- the flat estimate already
       used elsewhere in this file as the pre-probe sizing guess. Only
       reached if NEITHER real probe above got rejected, so ranking always
       has a usable number rather than silently favoring whichever
       candidate happened to be probed first.

    Never raises for "no rejection" -- that's the expected, common outcome
    of tiers 1-2, not a failure. Which tier actually fires in practice on
    the live account is exactly what tier 1's own docstring note above is
    there to let the next live session settle, one way or the other,
    without this breaking either way."""
    timed_exit_activation = now + timedelta(minutes=MAX_HOLD_MINUTES)

    # Tier 1: quantity=1.
    required_margin = await ibkr.probe_bracket_initial_margin(
        traded_symbol, action, 1, traded_price, timed_exit_activation)
    if required_margin is not None:
        return required_margin  # already per-share, quantity was 1

    # Tier 2: the same quantity this pair's real entry would start from.
    margin_per_share_estimate = traded_price * INITIAL_MARGIN_FRACTION
    guessed_quantity = compute_final_single_stock_quantity(
        sizing_elv, margin_per_share_estimate, ctx.divergence_trigger_level, ENTRY_TRIGGER_DISTANCE_MIN_PCT)
    if guessed_quantity > 0:
        required_margin = await ibkr.probe_bracket_initial_margin(
            traded_symbol, action, guessed_quantity, traded_price, timed_exit_activation)
        if required_margin is not None:
            return required_margin / guessed_quantity

    # Tier 3: the flat estimate, nothing real came back.
    return margin_per_share_estimate


async def rank_eligible_signals_by_adjusted_spread_efficiency(
    signals: list[tuple[PairSignal, PairContext]], ibkr: IBKRClient, sizing_elv: float, now: datetime,
) -> PairSignal | None:
    """Per the client 2026-09-12 (Checklist_120926.pdf), confirmed and
    fully replacing select_closest_to_trigger as the tiebreak: "If it is
    only one qualified pair, then that pair will execute. But if it more
    than more qualified pair, we will use Adjusted Spread Efficiency to
    select. Yes. Fully replace trigger tiebreaker. Use initial margin as
    described." "If there are two or more opportunities to trade at the
    same time, robot will select a pair with the highest Adjusted Spread
    Efficiency."

    The single-candidate case skips probing entirely -- no ranking needed
    when there's nothing to rank against, and it keeps the common case
    exactly as fast as before this feature existed.

    For 2+ candidates: probes every eligible candidate's Stock Initial
    Margin for real, one candidate at a time (SEQUENTIAL, not
    asyncio.gather) -- real orders against the live account for every
    candidate, not just the eventual winner, per the client: "Can we not
    place multiple order at the same time? We can... place multiple
    order simultaneously. Obtain initial margins, and do the maths."

    REVERSED from concurrent (asyncio.gather) to sequential 2026-09-14,
    per a third-party audit finding verified directly: IBKR evaluates
    Reg-T initial margin at the ACCOUNT level across every currently
    working order, not per-probe in isolation. Two candidates' probe
    entries submitted at the same instant would sit on
    the account SIMULTANEOUSLY, so candidate B's error 201 would report
    the margin needed for A's orders plus B's combined, not B's alone --
    corrupting both candidates' Spread Return % (an inflated total
    divided by only B's own quantity) and risking a false rejection for
    B even when it would fit the account on its own. Sequential avoids
    this entirely: each candidate's probe orders are fully submitted,
    read, cancelled AND confirmed cancelled (probe_bracket_initial_margin
    already waits for a terminal cancel state before returning) before
    the next candidate's probe ever goes out, so at most one candidate's
    orders are ever live on the account at once. Still real orders for
    every candidate, matching the client's instruction -- just one at a
    time instead of all at once, which is what "the maths" actually
    requires to come out correct.

    A candidate whose leg/price can't be determined yet
    (_traded_leg_estimate returns None) is skipped from ranking, not
    treated as a crash."""
    only_signal = signals[0][0] if len(signals) == 1 else None
    if only_signal is not None:
        return only_signal
    if not signals:
        return None

    ranked: list[tuple[PairSignal, float]] = []
    for signal, ctx in signals:
        leg = _traded_leg_estimate(ctx, signal)
        if leg is None:
            continue
        traded_symbol, action, traded_price = leg
        margin_per_share = await _probe_stock_initial_margin_per_share(
            ibkr, ctx, signal, traded_symbol, action, traded_price, sizing_elv, now)
        if margin_per_share <= 0:
            continue
        ranked.append((signal, margin_per_share))
    return select_highest_adjusted_spread_efficiency(ranked)


PREMARKET_FETCH_START_TIME = dtime(8, 0)
TRIGGER_RECALC_START_TIME = dtime(9, 31)


async def recalculate_trigger_exceptions(contexts: list[PairContext], now: datetime) -> None:
    """Per the client 2026-08-22 (restated twice for emphasis across two PDF
    revisions): "From 9:31, recalculation continues for Maximum Divergence,
    until entry. Entry ends 10:30am. Before 10:30am, you have to use the
    applicable Maximum Divergence for Entry." NOT a one-time 9:31 check --
    called once per check cycle (from run_check_cycle) for as long as
    now.time() is within [TRIGGER_RECALC_START_TIME, LATEST_ENTRY_TIME],
    for every pair that hasn't already entered a position today.

    One batched premarket-inclusive fetch across every pair still in play
    this cycle, rather than one fetch per pair -- at a 1-minute cadence
    across up to 59 minutes (9:31-10:30), a per-pair fetch would multiply
    into hundreds of extra Databento calls a day for no benefit, since the
    window (today's premarket-to-now) is the same for every symbol.

    For each pair: tests whether cointegration has broken over the
    client's specified window ("8:00 AM to 9:31 AM ET... do not attempt an
    ADF test using only the 9:30-9:31 regular session bars" -- widened here
    to 8:00am-now as recalculation continues past 9:31), and if so,
    recomputes the non-reverting-event trigger over the previous session +
    today's premarket-to-now, replacing ctx.divergence_trigger_level only
    when a larger candidate is found this cycle -- otherwise the previous
    applicable value (base, or an earlier cycle's exception) stands
    unchanged, per "IF there is no new Maximum Divergence, then the one
    from previous RTH applies." A pair with an open position is skipped --
    entry has already happened, so recalculating its trigger no longer
    matters."""
    due = [ctx for ctx in contexts if ctx.position is None and len(ctx.prior_session_price_a) >= 2]
    if not due:
        return
    symbols = sorted({s for ctx in due for s in (ctx.symbol_a, ctx.symbol_b)})
    today = now.astimezone(ET).date()
    start_utc = datetime.combine(today, PREMARKET_FETCH_START_TIME, tzinfo=ET).astimezone(timezone.utc)
    end_utc = now.astimezone(timezone.utc)
    try:
        bars = await asyncio.to_thread(
            fetch_intraday_warmup, symbols,
            start_utc.strftime("%Y-%m-%dT%H:%M:%S"), end_utc.strftime("%Y-%m-%dT%H:%M:%S"))
    except Exception:
        logger.warning("trigger exception recalc: premarket fetch failed this cycle -- "
                        "previous applicable trigger stands for every pair", exc_info=True)
        return
    if bars.empty:
        return
    pivot = bars.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
    # True Open of today's REGULAR session (9:30am ET), not premarket's own
    # first print at PREMARKET_FETCH_START_TIME (8:00am) and not a close-price
    # proxy -- see PairContext.today_session_open_a's own comment. Filtered
    # to RTH first: `bars` here spans 8:00am-now, but "Today: 9:30am ET+Now"
    # per the client's own spec means the regular session's own open.
    today_rth_bars = _rth_only(bars)
    today_open = (today_rth_bars.sort_values("ts_event").groupby("symbol")["open"].first().to_dict()
                  if not today_rth_bars.empty else {})

    for ctx in due:
        if ctx.symbol_a not in pivot.columns or ctx.symbol_b not in pivot.columns:
            continue
        premarket_both = pivot[[ctx.symbol_a, ctx.symbol_b]].dropna()
        if premarket_both.empty:
            continue
        ctx.premarket_price_a = premarket_both[ctx.symbol_a]
        ctx.premarket_price_b = premarket_both[ctx.symbol_b]
        if ctx.symbol_a in today_open:
            ctx.today_session_open_a = today_open[ctx.symbol_a]
        if ctx.symbol_b in today_open:
            ctx.today_session_open_b = today_open[ctx.symbol_b]

        # Cointegration break test over the client's specified premarket-
        # inclusive window (not just the 9:30-9:31 regular-session bar) --
        # the 3-metric composite per the client 2026-08-23's Cointegration
        # Breakdown document.
        premarket_spread = compute_spread(ctx.premarket_price_a, ctx.premarket_price_b, ctx.hedge_ratio)
        premarket_equilibrium = equilibrium_spread(premarket_spread)
        broke = is_cointegration_broken(
            premarket_spread, premarket_equilibrium,
            ctx.premarket_price_a, ctx.premarket_price_b, ctx.half_life_minutes, ctx.hedge_ratio,
        )
        ctx.cointegration_broke_before_931 = broke

        # Checklist condition A, re-derived fresh every cycle over the full
        # "Prior RTH and Current RTH+Now" lookback -- independent of the
        # cointegration-break check below, so it still reflects the latest
        # data even on a cycle where cointegration hasn't (re)broken. See
        # PairContext.mean_reverted_since_max_divergence's own comment.
        combined_price_a = pd.concat([ctx.prior_session_price_a, ctx.premarket_price_a])
        combined_price_b = pd.concat([ctx.prior_session_price_b, ctx.premarket_price_b])
        combined_spread = compute_spread(combined_price_a, combined_price_b, ctx.hedge_ratio)
        combined_equilibrium = equilibrium_spread(combined_spread)
        ctx.mean_reverted_since_max_divergence = reverted_after_peak(combined_spread, combined_equilibrium)
        ctx.reverted_over_50pct_since_max_divergence = reverted_more_than_pct_after_peak(
            combined_spread, combined_equilibrium)

        # Re-derived fresh every cycle, not sticky -- "IF there is no new
        # Maximum Divergence, then the one from previous RTH applies" reads
        # as a live, per-cycle answer (base trigger unless an exception
        # applies RIGHT NOW), not a one-way ratchet that keeps an earlier
        # cycle's exception after cointegration has re-converged or the
        # non-reverting event it was based on has since reverted.
        if not broke:
            ctx.divergence_trigger_level = ctx.base_trigger_level
            ctx.using_exception_trigger = False
            continue

        events = detect_non_reverting_rth_events(combined_spread, combined_equilibrium, RTH_OPEN, RTH_CLOSE)
        exception_level = _bounded_trigger_level(max((e.max_spread for e in events), default=None))
        if exception_level is not None and (ctx.base_trigger_level is None or exception_level > ctx.base_trigger_level):
            if exception_level != ctx.divergence_trigger_level:
                logger.info(f"{ctx.symbol_a}/{ctx.symbol_b}: exception trigger updated to "
                            f"{exception_level:.4f} (base {ctx.base_trigger_level})")
            ctx.divergence_events = events
            ctx.divergence_trigger_level = exception_level
            ctx.using_exception_trigger = True
        else:
            ctx.divergence_trigger_level = ctx.base_trigger_level
            ctx.using_exception_trigger = False


# Minimum sample size before the informational live_correlation/
# live_adf_pvalue display (Divergence Watch / the qualifying pairs report)
# is computed at all -- ADF/correlation are noisy and unreliable on too
# few points. The actual breakdown gate's own windowing now lives inside
# cointegration_breakdown.py (60-bar ADF, half-life-length lookback,
# correlation over the full available series), per the client 2026-08-23's
# Cointegration Breakdown document.
MIN_BARS_FOR_LIVE_COINTEGRATION_STATS = 30


def report_event_from_position(ctx: "PairContext", close_time: datetime, exit_reason: str = "") -> ReportEvent:
    """Builds the report's row for a real trade directly from its own
    Position, at the moment it closes -- real entry/exit times, real
    executed quantity, real worst drawdown.

    Per the client: "Post-Trade Report should be based on executed pairs."
    This replaced re-scanning each pair's whole-day price history for
    divergence events, which had two problems: a pair that never opened a
    position could still contribute using a hypothetical share count
    computed purely for reporting (almost certainly the source of a real
    76.48% "max loss" that never happened to any position), and even for a
    pair that DID trade, other divergence events from elsewhere in its day --
    before the entry, after the exit, or while the bot was already in a
    different pair -- would show up alongside the one event that was
    actually the trade.

    max_adverse_excursion is kept in the same per-share-of-qty_a convention
    detect_divergence_events used, so elv_loss_pct/pnl_pct_at_gap_closure in
    post_trade_report.py need no change: position.worst_unrealized_pnl is
    already in real account dollars (unrealized_pnl() multiplies through by
    qty_a), so dividing back by qty_a here recovers that same per-share
    figure from real data instead of a historical re-scan.

    Divides by the TRADED leg's own qty, not qty_a unconditionally --
    added 2026-09-03 for single-leg "Expansion Leg" positions, where
    qty_a is 0 whenever traded_leg=="B" and dividing by it crashed this
    with a ZeroDivisionError on every exit of a leg-B position. For a
    two-leg position traded_leg is always "A" (the default) and qty_a is
    always nonzero, so this is unchanged from before for those."""
    position = ctx.position
    traded_qty = position.qty_a if position.traded_leg == "A" else position.qty_b
    return ReportEvent(
        symbol_a=ctx.symbol_a, symbol_b=ctx.symbol_b,
        half_life_minutes=position.half_life_minutes,
        open_time=position.entry_time, close_time=close_time,
        max_spread=position.entry_divergence,
        duration=close_time - position.entry_time,
        max_adverse_excursion=abs(position.worst_unrealized_pnl) / traded_qty,
        qty_a=position.qty_a, qty_b=position.qty_b,
        action_a=position.action_a, action_b="SELL" if position.action_a == "BUY" else "BUY",
        traded_leg=position.traded_leg,
        exit_reason=exit_reason,
        points_to_trigger_at_entry=position.entry_trigger_level - position.entry_divergence,
        take_profit_trigger_level=position.entry_trigger_level,
    )


async def flatten_rejection(ibkr: IBKRClient, order_ids: list[int]) -> str | None:
    """The rejection reason if ANY of a flatten's closing MKT orders was
    refused by the broker, else None. Confirmed live 2026-08-20 (the
    bracket price rejection incident): a flattening order is not immune to
    rejection either (a halted contract, no shares available to borrow, an
    account restriction), and every flatten_symbols()/flatten_all() caller
    used to assume it always worked and immediately clear its own tracked
    position -- if the flatten itself silently failed, the position stayed
    open at the broker with the bot now believing the book was flat and no
    longer watching it at all. Checked concurrently since a pair's flatten
    is normally two orders (one per leg)."""
    if not order_ids:
        return None
    rejections = await asyncio.gather(*(ibkr.order_was_rejected(oid) for oid in order_ids))
    return next((r for r in rejections if r), None)


def update_live_cointegration_stats(ctx: PairContext) -> None:
    """Recomputes live correlation and ADF p-value once per bar, over a
    rolling MIN_BARS_FOR_LIVE_COINTEGRATION_STATS-bar window -- informational
    only now (feeds the Divergence Watch / Daily Qualifying Pairs report
    tables), superseded as the actual breakdown gate by the 3-metric
    composite below.

    ctx.cointegration_still_broken is set from cointegration_breakdown.
    is_cointegration_broken -- per the client 2026-08-23's "Cointegration
    Breakdown" document ("The 3 Core Detection Metrics for Broken
    Cointegration: All 3 must show Broken Cointegration"), replacing the
    original correlation<0.90-OR-ADF>=0.05 test this function used to run
    directly. Re-verified every cycle rather than a one-time snapshot, so
    a pair that re-cointegrates after its 9:31am break becomes ineligible
    again immediately (see entry_eligible) -- per "10am till Now ensures
    broken co-integration right until entry. Not just till 10am."""
    if len(ctx.price_history_a) < MIN_BARS_FOR_LIVE_COINTEGRATION_STATS:
        return
    if len(ctx.spread_history) < MIN_BARS_FOR_LIVE_COINTEGRATION_STATS:
        return
    price_a = ctx.price_history_a.tail(MIN_BARS_FOR_LIVE_COINTEGRATION_STATS)
    price_b = ctx.price_history_b.tail(MIN_BARS_FOR_LIVE_COINTEGRATION_STATS)
    spread = ctx.spread_history.tail(MIN_BARS_FOR_LIVE_COINTEGRATION_STATS)
    ctx.live_correlation = pearson_correlation(price_a, price_b)
    ctx.live_adf_pvalue = engle_granger_adf_pvalue(spread)
    ctx.cointegration_still_broken = is_cointegration_broken(
        ctx.spread_history, equilibrium_spread(ctx.spread_history),
        ctx.price_history_a, ctx.price_history_b, ctx.half_life_minutes, ctx.hedge_ratio,
    )


def _compute_johansen_hurst(price_a: pd.Series, price_b: pd.Series) -> tuple[bool, float | None]:
    """The shared Johansen/Hurst computation update_live_johansen_hurst
    runs independently for both the 20-day and 1-day windows -- same
    test, just fed a different window."""
    confirmed = johansen_confirmed(price_a, price_b)
    _, spread = engle_granger_hedge_ratio_and_spread(price_a, price_b)
    hurst = hurst_exponent(spread)
    return confirmed, hurst


def update_live_johansen_hurst(ctx: PairContext) -> None:
    """Recomputes Johansen and Hurst live, once per bar, over BOTH the
    20-day window (ctx.johansen_hurst_price_a/b) and the 1-day window
    (ctx.johansen_hurst_price_a_1day/b_1day) -- per the client 2026-09-06,
    confirmed "two separate calculations", not one merged reading. Per the
    client 2026-08-25 (Correct_Filters.pdf): "Bot will continue to check
    with new 1 minute candles, right until entry" -- a hard,
    continuously-rechecked entry gate, not a one-time Phase 1 screening
    snapshot. Each window stays at its prior state (fail-closed) until it
    individually has enough bars.

    Sets ctx.johansen_ever_unconfirmed/hurst_ever_above_threshold (20-day)
    and ctx.johansen_ever_unconfirmed_1day/hurst_ever_above_threshold_1day
    (1-day) independently of each other and of their own Johansen/Hurst
    counterpart -- per the client 2026-08-28's clarification (originally
    about the single merged window, applies identically to all four now):
    "They must not be validated together. At the same time. No. They just
    need to be validated once during Entry RTH session." entry_eligible
    (via the PairSignal construction in run_check_cycle) reads the AND of
    all four sticky flags, not any single cycle's raw reading."""
    if (len(ctx.johansen_hurst_price_a) >= MIN_BARS_FOR_LIVE_JOHANSEN_HURST
            and len(ctx.johansen_hurst_price_b) >= MIN_BARS_FOR_LIVE_JOHANSEN_HURST):
        # Logged only on the cycle each sticky flag actually flips -- per
        # the client repeatedly asking "at what time did X pass
        # Johansen/Hurst" (2026-08-31, 2026-09-01) with no way to answer
        # from the existing divergence-check log line, which only ever
        # shows the combined eligible= result, not either raw condition
        # on its own.
        was_johansen_unconfirmed = ctx.johansen_ever_unconfirmed
        was_hurst_above = ctx.hurst_ever_above_threshold
        ctx.live_johansen_confirmed, ctx.live_hurst = _compute_johansen_hurst(
            ctx.johansen_hurst_price_a, ctx.johansen_hurst_price_b)
        if not ctx.live_johansen_confirmed:
            ctx.johansen_ever_unconfirmed = True
        if ctx.live_hurst is not None and ctx.live_hurst > MIN_HURST:
            ctx.hurst_ever_above_threshold = True
        if ctx.johansen_ever_unconfirmed and not was_johansen_unconfirmed:
            logger.info(f"{ctx.symbol_a}/{ctx.symbol_b}: 20 Day Johansen validated (first time this session)")
        if ctx.hurst_ever_above_threshold and not was_hurst_above:
            logger.info(f"{ctx.symbol_a}/{ctx.symbol_b}: 20 Day Hurst validated (first time this session), "
                        f"live_hurst={ctx.live_hurst:.4f}")

    if (len(ctx.johansen_hurst_price_a_1day) >= MIN_BARS_FOR_LIVE_JOHANSEN_HURST
            and len(ctx.johansen_hurst_price_b_1day) >= MIN_BARS_FOR_LIVE_JOHANSEN_HURST):
        was_johansen_unconfirmed_1day = ctx.johansen_ever_unconfirmed_1day
        was_hurst_above_1day = ctx.hurst_ever_above_threshold_1day
        ctx.live_johansen_confirmed_1day, ctx.live_hurst_1day = _compute_johansen_hurst(
            ctx.johansen_hurst_price_a_1day, ctx.johansen_hurst_price_b_1day)
        if not ctx.live_johansen_confirmed_1day:
            ctx.johansen_ever_unconfirmed_1day = True
        if ctx.live_hurst_1day is not None and ctx.live_hurst_1day > MIN_HURST:
            ctx.hurst_ever_above_threshold_1day = True
        if ctx.johansen_ever_unconfirmed_1day and not was_johansen_unconfirmed_1day:
            logger.info(f"{ctx.symbol_a}/{ctx.symbol_b}: 1 Day Johansen validated (first time this session)")
        if ctx.hurst_ever_above_threshold_1day and not was_hurst_above_1day:
            logger.info(f"{ctx.symbol_a}/{ctx.symbol_b}: 1 Day Hurst validated (first time this session), "
                        f"live_hurst={ctx.live_hurst_1day:.4f}")
        # 1 Day Pearson -- per the client's Method 1 (1_Day_Pearson_60_Bars.pdf):
        # each leg's OWN bar-to-bar returns over a trailing 60-bar window,
        # correlated against the other leg's. ctx.premarket_price_a/b is
        # already fetched every cycle by recalculate_trigger_exceptions
        # (premarket-inclusive, so the window is a full 60 bars even in
        # the first hour after the open) -- no separate fetch or seed
        # field needed here.
        if len(ctx.premarket_price_a) >= PEARSON_1DAY_WINDOW_BARS + 1 and len(ctx.premarket_price_b) >= PEARSON_1DAY_WINDOW_BARS + 1:
            was_pearson_above_1day = ctx.pearson_ever_above_threshold_1day
            ctx.live_pearson_1day = pearson_correlation_1day(
                ctx.premarket_price_a.tail(PEARSON_1DAY_WINDOW_BARS + 1),
                ctx.premarket_price_b.tail(PEARSON_1DAY_WINDOW_BARS + 1))
            if ctx.live_pearson_1day is not None and ctx.live_pearson_1day >= MIN_RECENT_CORRELATION:
                ctx.pearson_ever_above_threshold_1day = True
            if ctx.pearson_ever_above_threshold_1day and not was_pearson_above_1day:
                logger.info(f"{ctx.symbol_a}/{ctx.symbol_b}: 1 Day Pearson validated (first time this session), "
                            f"live_pearson={ctx.live_pearson_1day:.4f}")


def book_is_flat(contexts: "list[PairContext]") -> bool:
    """True when no pair holds a position anywhere in the book.

    The spec runs one position at a time: the quantity formula spends 98% of
    EquityWithLoanValue on a single pair, the tiebreak picks ONE pair when
    several trigger together, and re-entry happens "at the end of a trade".
    Concurrent entries also break sizing outright -- two pairs each sized
    against the full equity asked IBKR for more initial margin than was
    available, and the second was rejected with error 201."""
    return all(c.position is None for c in contexts)


async def run_check_cycle(
    contexts: list[PairContext], feed: LiveBarFeed, ibkr: IBKRClient,
    excluded: ExcludedSymbols, news_flagged: ExcludedSymbols, daily_state: DailyState,
    equity_with_loan_value: float, elv_at_925: float,
    market_halt: "MarketHalt | None" = None,
    news_scan_health: "NewsScanHealth | None" = None,
) -> None:
    now = datetime.now(ET)
    if SINGLE_STOCK_EXCURSION_MODE:
        await run_single_stock_check_cycle(
            contexts, feed, ibkr, daily_state, equity_with_loan_value, elv_at_925, now=now
        )
        return
    # Per the client 2026-08-26: news exclusion does NOT block entry or
    # candidate discovery -- "If a symbol is in the news, it should get in.
    # The news could be the reason for the expansion. We want expansion."
    # News only ever matters as an EXIT: "After we have already entered,
    # and a symbol comes in the news, we exit. News Stop." So the raw news
    # signal feeds ONLY the News Stop check below, never entry eligibility
    # or screening. The PERMANENT blacklist (the client's own manually
    # curated never-trade list, unrelated to news) still blocks both, same
    # as before -- see permanent_blacklisted below.
    #
    # news_flagged_now is the raw "currently in the news" signal (both
    # scans write here every cycle); excluded_now is the client-visible
    # Excluded Symbols list, which after the 2026-08-28 restructure only
    # ever gets a symbol added BELOW, at the moment a News Stop actually
    # fires over it -- included here too so a symbol the client manually
    # adds via the dashboard still forces an exit, same as before. Both
    # read from disk each cycle so a dashboard edit takes effect without a
    # restart.
    news_flagged_now = news_flagged.active_symbols(now=now)
    excluded_now = excluded.active_symbols(now=now)
    permanent_blacklisted = load_permanent_blacklist()
    # This combined set feeds ONLY the News Stop exit check below --
    # screening and entry eligibility use permanent_blacklisted alone (see
    # its own comment above), not this one.
    blacklisted = news_flagged_now | excluded_now | permanent_blacklisted

    # A market-wide crash or extraordinary rally excludes the entire S&P 500
    # for 24 hours, so no pair may open. Checked here rather than inside
    # entry_eligible because it is not a property of any one signal, and only
    # new entries are blocked -- open positions still run their normal exit
    # rules, since abandoning a live position mid-crash is worse than working
    # it out through take profit, stop loss or the end of day flatten.
    halted_market = market_halt is not None and market_halt.is_active()
    # Per the client 2026-08-17: "If Gemini news API is down, no trading." --
    # same treatment as halted_market: blocks new entries only, an already
    # open position still runs its normal exit rules regardless.
    news_scan_down = news_scan_health is not None and not news_scan_health.healthy

    # append the latest live bar (if any) to each pair's running history
    for ctx in contexts:
        price_a = feed.book.latest_close(ctx.symbol_a)
        price_b = feed.book.latest_close(ctx.symbol_b)
        if price_a is None or price_b is None:
            logger.info(f"{ctx.symbol_a}/{ctx.symbol_b} no live price yet -- "
                        f"price_a={price_a} price_b={price_b}")
            continue
        stamp = pd.Timestamp(now)
        ctx.price_history_a.loc[stamp] = price_a
        ctx.price_history_b.loc[stamp] = price_b
        # Beta un-frozen 2026-08-24 -- see PairContext.beta_estimator's own
        # comment. Lazily seeded on the first live bar this ctx sees, then
        # updated incrementally every bar after that.
        if ctx.beta_estimator is None:
            ctx.beta_estimator = KalmanBetaEstimator(beta=ctx.hedge_ratio)
        ctx.hedge_ratio = ctx.beta_estimator.update(price_a, price_b)
        # Recompute the WHOLE series from raw prices with the current hedge
        # ratio, rather than appending one point onto a series whose earlier
        # points were computed with a different beta. This is the exact fix
        # for the 2026-08-05 bug from back when beta was ALSO a live Kalman
        # estimate that moved continuously: appending inconsistently created
        # a fake discontinuity in the series exactly where the beta had
        # moved, which the old divergence-event detector read as a huge,
        # entirely fictional divergence event -- confirmed against MPWR/MRVL,
        # a "closed event" of $1052.50 where a fresh, consistent recompute
        # over the same data found nothing above $51.77. Now that beta is
        # live again, this recompute is what keeps that bug from coming
        # back, not a leftover precaution for a scenario that can't happen
        # anymore. compute_spread is a cheap vectorized op even over a full
        # session's bars, so recomputing every cycle costs nothing real.
        ctx.spread_history = compute_spread(ctx.price_history_a, ctx.price_history_b, ctx.hedge_ratio)
        update_live_cointegration_stats(ctx)
        # Confirmed 9:30am (market open), not the PDF's original "10am+Now"
        # -- see JOHANSEN_HURST_LIVE_START_TIME's own comment. Today's bars
        # join the 20 day Johansen/Hurst window from here onward.
        if now.astimezone(ET).time() >= JOHANSEN_HURST_LIVE_START_TIME:
            ctx.johansen_hurst_price_a.loc[stamp] = price_a
            ctx.johansen_hurst_price_b.loc[stamp] = price_b
            # Sliding Gate, per the client: each new bar drops the oldest,
            # keeping the array at a fixed JOHANSEN_HURST_WINDOW_BARS length.
            if len(ctx.johansen_hurst_price_a) > JOHANSEN_HURST_WINDOW_BARS:
                ctx.johansen_hurst_price_a = ctx.johansen_hurst_price_a.iloc[-JOHANSEN_HURST_WINDOW_BARS:]
                ctx.johansen_hurst_price_b = ctx.johansen_hurst_price_b.iloc[-JOHANSEN_HURST_WINDOW_BARS:]
            # Same new bar, same Sliding Gate mechanism, the "1 Day"
            # parallel window -- see JOHANSEN_HURST_WINDOW_1DAY_BARS's own
            # comment.
            ctx.johansen_hurst_price_a_1day.loc[stamp] = price_a
            ctx.johansen_hurst_price_b_1day.loc[stamp] = price_b
            if len(ctx.johansen_hurst_price_a_1day) > JOHANSEN_HURST_WINDOW_1DAY_BARS:
                ctx.johansen_hurst_price_a_1day = ctx.johansen_hurst_price_a_1day.iloc[-JOHANSEN_HURST_WINDOW_1DAY_BARS:]
                ctx.johansen_hurst_price_b_1day = ctx.johansen_hurst_price_b_1day.iloc[-JOHANSEN_HURST_WINDOW_1DAY_BARS:]
            # Per-pair guard: coint_johansen (inside johansen_confirmed) can
            # raise on a singular/near-singular matrix -- a real failure
            # mode on live data (long flat stretches, duplicate timestamps).
            # Uncaught here, that would abort this bare for-loop entirely,
            # silently freezing price/spread/divergence tracking for every
            # OTHER pair still left to process this cycle too, not just this
            # one -- and if the bad data persists in the sliding window, it
            # repeats every cycle. Isolated to this pair only instead.
            try:
                update_live_johansen_hurst(ctx)
            except Exception:
                logger.exception(f"{ctx.symbol_a}/{ctx.symbol_b}: update_live_johansen_hurst failed, "
                                  f"skipping this pair's Johansen/Hurst update this cycle")

    # Maximum Divergence trigger exception recalculation -- REMOVED
    # 2026-08-24, per the client's rollback to the pre-2026-08-22 "old
    # script": "Delete broken co-integration and broken co-relation
    # condition. Together with the 4pm-9:29:59am ET exception trigger."
    # recalculate_trigger_exceptions is no longer called, so
    # ctx.divergence_trigger_level stays permanently equal to
    # ctx.base_trigger_level (set once at warm-up from the prior RTH
    # session) and ctx.using_exception_trigger stays permanently False for
    # the rest of the session -- entries now trigger purely off the base,
    # prior-session Maximum Divergence, with no live exception override.
    # The function itself, and the PairContext fields it alone populated
    # (premarket_price_a/b, cointegration_broke_before_931), are left in
    # place but dormant rather than deleted outright, to avoid a sprawling
    # cross-file removal the night before going live.

    # check exits first -- an open position always takes priority over new entries
    for ctx in contexts:
        if ctx.position is None or len(ctx.price_history_a) == 0:
            continue

        # Confirmed live 2026-08-07: order_was_rejected only watches a
        # 3-second window right after placement, so a LATE cancel (an
        # external reqGlobalCancel, IBKR discarding a stale working order,
        # anything after that window) left ctx.position pointing at a
        # position that was never actually opened at the broker, computing
        # phantom P&L against it indefinitely. Reconciled here every cycle,
        # same spirit as the native-fill check below.
        # Checked on BOTH legs' order ids since 2026-09-03 -- each leg is
        # now its own independent order (no more single combo id covering
        # both).
        if (ibkr.entry_order_abandoned(ctx.position.entry_order_id)
                or ibkr.entry_order_abandoned(ctx.position.entry_order_id_b)):
            logger.warning(f"{ctx.symbol_a}/{ctx.symbol_b}: entry order "
                           f"{ctx.position.entry_order_id}/{ctx.position.entry_order_id_b} "
                           f"was cancelled without ever filling -- clearing the phantom position, "
                           f"no trade actually happened")
            ctx.position = None
            continue

        price_a = ctx.price_history_a.iloc[-1]
        price_b = ctx.price_history_b.iloc[-1]
        # Tracks the worst point this specific trade has actually reached, so
        # the report can use real data instead of re-scanning the pair's
        # whole-day price history for a divergence event that may not even
        # coincide with this position's holding period.
        live_pnl = unrealized_pnl(ctx.position, price_a, price_b)
        ctx.position.worst_unrealized_pnl = min(ctx.position.worst_unrealized_pnl, live_pnl)

        # Hard Stop Loss AFTER ENTRY re-set -- per the client 2026-09-09:
        # "you move the Hard Stop Loss... Remaining Gap to Zero + Trigger...
        # No limit to continuous re-try. It is until it is Done." Retried
        # every cycle until it succeeds once; applies ONLY to Hard Stop
        # Loss (his own words), not Take Profit/Timed Exit/the PnL Stop.
        # See Position.hard_stop_reset_done's own comment for why a
        # never-succeeding retry is safe -- the wide @ ENTRY price keeps
        # resting and protecting the whole time.
        if not ctx.position.hard_stop_reset_done and ctx.position.hard_stop_order_id is not None:
            traded_symbol = ctx.symbol_a if ctx.position.traded_leg == "A" else ctx.symbol_b
            entry_price = ctx.position.entry_price_a if ctx.position.traded_leg == "A" else ctx.position.entry_price_b
            quantity = int(ctx.position.qty_a if ctx.position.traded_leg == "A" else ctx.position.qty_b)
            closing_action = "SELL" if ctx.position.action_a == "BUY" else "BUY"
            tight_price = resting_hard_stop_price(
                ctx.position.action_a, entry_price, ctx.position.entry_trigger_level,
                ctx.position.entry_divergence)
            oca_group = f"single_{ctx.position.tp_order_id}"
            succeeded = await ibkr.modify_hard_stop(
                traded_symbol, closing_action, quantity, tight_price,
                ctx.position.hard_stop_order_id, oca_group)
            if succeeded:
                ctx.position.hard_stop_reset_done = True
                logger.info(f"{ctx.symbol_a}/{ctx.symbol_b}: Hard Stop Loss re-set to "
                           f"{tight_price:.2f} (AFTER ENTRY, was {wide_hard_stop_price(ctx.position.action_a, entry_price, ctx.position.entry_trigger_level):.2f} @ ENTRY)")

        # NEWS STOP: exit @ MKT immediately if either leg has become newly
        # excluded by the news scan AFTER entry -- per the client 2026-08-21:
        # "NEWS STOP applies to symbols that are now in news after Entry.
        # These symbols were not in the News before Entry." entry_eligible
        # already guarantees neither leg was blacklisted at entry, so simply
        # finding either leg in the live blacklist here means it became
        # excluded sometime after -- checked ahead of native bracket
        # reconciliation below, so a real news risk pre-empts a still-open
        # Take Profit/Stop Loss rather than waiting for one of those to fill.
        if NEWS_STOP_ENABLED and (ctx.symbol_a in blacklisted or ctx.symbol_b in blacklisted):
            # Per the client 2026-08-28: "The Symbol responsible for the
            # News Stop Loss will become Excluded." Records ONLY the leg(s)
            # that came from the raw news signal -- one already in excluded
            # (e.g. a manual dashboard addition) or only in
            # permanent_blacklisted needs no new record here.
            for symbol in (ctx.symbol_a, ctx.symbol_b):
                if symbol in news_flagged_now:
                    excluded.add(symbol, now=now)
            await ibkr.cancel_orders_and_confirm([
                ctx.position.tp_order_id, ctx.position.hard_stop_order_id,
                ctx.position.timed_exit_order_id, ctx.position.pnl_stop_order_id, ctx.position.trail_order_id])
            cancel_hedge_bracket(ibkr, ctx.position)
            flatten_ids = await ibkr.flatten_symbols({ctx.symbol_a, ctx.symbol_b})
            rejection = await flatten_rejection(ibkr, flatten_ids)
            if rejection:
                logger.error(f"{ctx.symbol_a}/{ctx.symbol_b}: flatten for news_stop was REJECTED "
                             f"({rejection}) -- position may still be open at the broker, "
                             f"keeping it tracked and retrying next cycle")
                continue
            realized_pnl = unrealized_pnl(ctx.position, price_a, price_b)
            logger.warning(f"EXIT {ctx.symbol_a}/{ctx.symbol_b}: news_stop, pnl={realized_pnl:.2f}")
            daily_state.realized_pnl_today += realized_pnl
            record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
            if not reentry_allowed("news_stop", now):
                daily_state.reentry_locked_out = True
            daily_state.trade_report_events.append(report_event_from_position(ctx, now, exit_reason="news_stop"))
            ctx.position = None
            continue

        # Stop Loss (bot-monitored) REINSTATED 2026-09-05, one day after
        # being deleted -- per the client: "Re-instate: API Monitored
        # Stop Loss", now as a TRIPLE-touch rule instead of the old
        # double-touch one ("API-Monitored Soft Exit, Triple-Touch Rule,
        # High Precision Control... Noise Immunity"). Runs as a second,
        # independent layer alongside the resting Hard Stop Loss below --
        # this one bot-monitored and tighter (fires at true zero
        # divergence), the resting one wider and broker-side, a backstop
        # in case this check somehow doesn't catch a real blowup in
        # time. See risk_manager.stop_loss_triple_touch_exit's own
        # docstring.
        if stop_loss_triple_touch_exit(ctx.position, price_a, price_b):
            await ibkr.cancel_orders_and_confirm([
                ctx.position.tp_order_id, ctx.position.hard_stop_order_id,
                ctx.position.timed_exit_order_id, ctx.position.pnl_stop_order_id, ctx.position.trail_order_id])
            cancel_hedge_bracket(ibkr, ctx.position)
            flatten_ids = await ibkr.flatten_symbols({ctx.symbol_a, ctx.symbol_b})
            rejection = await flatten_rejection(ibkr, flatten_ids)
            if rejection:
                logger.error(f"{ctx.symbol_a}/{ctx.symbol_b}: flatten for stop_loss was REJECTED "
                             f"({rejection}) -- position may still be open at the broker, "
                             f"keeping it tracked and retrying next cycle")
                continue
            realized_pnl = unrealized_pnl(ctx.position, price_a, price_b)
            logger.warning(f"EXIT {ctx.symbol_a}/{ctx.symbol_b}: stop_loss (triple touch), pnl={realized_pnl:.2f}")
            daily_state.realized_pnl_today += realized_pnl
            record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
            daily_state.reentry_locked_out = True  # not a take_profit close -- never eligible for re-entry
            daily_state.trade_report_events.append(report_event_from_position(ctx, now, exit_reason="stop_loss"))
            ctx.position = None
            continue

        # Take Profit, Hard Stop, and Timed Exit are all real resting
        # orders again as of 2026-09-04 (less than a day after the
        # 2026-09-03 restructure moved TP/Hard Stop to bot-monitored logic
        # -- the client restored them: "Why can it not be a resting
        # order... You simply close a single stock instead of two
        # stocks." See risk_manager.resting_take_profit_price/
        # resting_hard_stop_price and ibkr_client.place_single_stock_bracket).
        # Reconciled here every cycle, same pattern as the original
        # two-leg 3-way bracket reconciliation this replaces (just scoped
        # to the one real stock now).
        tp_filled = ibkr.bracket_order_filled(ctx.position.tp_order_id)
        hard_stop_filled = ibkr.bracket_order_filled(ctx.position.hard_stop_order_id)
        timed_exit_filled = ibkr.bracket_order_filled(ctx.position.timed_exit_order_id)
        # The fourth OCA leg -- per the client 2026-09-09's "-5% PnL Stop
        # Loss". Reported as "hard_stop" below rather than its own reason:
        # it is a stop loss by any reading, and reentry_allowed already
        # treats every non take_profit exit identically, so a separate
        # reason would change nothing downstream while adding a new value
        # every report and state check would have to learn.
        pnl_stop_filled = ibkr.bracket_order_filled(ctx.position.pnl_stop_order_id)
        # Trailing Take Profit, the FIFTH OCA leg -- per the client
        # 2026-09-15. Reported as "take_profit" below, same reasoning as
        # the resting MIT take profit: it only ever fires to lock in a
        # favorable move, never a loss.
        trail_filled = ibkr.bracket_order_filled(ctx.position.trail_order_id)
        # Per the client 2026-09-09: "Exit both Stock A and Stock B IF Hard
        # Stop Loss triggers in either Stock A or Stock B", and the same for
        # Take Profit. Each leg carries its own resting bracket, so EITHER
        # leg's bracket firing has to close the whole position -- the broker
        # cannot do that cross-leg step itself (an OCA group only cancels
        # sibling orders on the same stock, it cannot flatten a different
        # symbol), so it is done here.
        hedge_tp_filled = ibkr.bracket_order_filled(ctx.position.hedge_tp_order_id)
        hedge_stop_filled = ibkr.bracket_order_filled(ctx.position.hedge_hard_stop_order_id)
        hedge_timed_filled = ibkr.bracket_order_filled(ctx.position.hedge_timed_exit_order_id)
        if (tp_filled is not None or hard_stop_filled is not None or timed_exit_filled is not None
                or pnl_stop_filled is not None or trail_filled is not None
                or hedge_tp_filled is not None or hedge_stop_filled is not None
                or hedge_timed_filled is not None):
            if tp_filled is not None or trail_filled is not None or hedge_tp_filled is not None:
                reason = "take_profit"
            elif (hard_stop_filled is not None or pnl_stop_filled is not None
                  or hedge_stop_filled is not None):
                reason = "hard_stop"
            else:
                reason = "timed_exit"
            # Whichever leg did NOT fire is still open at the broker: cancel
            # its now-orphaned bracket orders and close it at market, BEFORE
            # booking P&L, so the recorded exit reflects a genuinely flat
            # pair rather than a half-closed one.
            fired_on_hedge = (hedge_tp_filled is not None or hedge_stop_filled is not None
                              or hedge_timed_filled is not None)
            survivor = leg_still_open_after_bracket_fill(ctx.position, fired_on_hedge)
            if survivor is not None:
                still_open, its_orders = survivor
                for oid in its_orders:
                    ibkr.cancel_order_id(oid)
                try:
                    flatten_ids = await ibkr.flatten_symbols({still_open})
                    leg_rejection = await flatten_rejection(ibkr, flatten_ids)
                    if leg_rejection:
                        logger.critical(f"{still_open}: cross leg close was REJECTED ({leg_rejection}) "
                                        f"after the other leg's {reason} -- this leg is still OPEN and "
                                        f"now unhedged, needs manual attention")
                    else:
                        logger.info(f"{still_open}: closed at market to follow the other leg's {reason}")
                except Exception as e:
                    logger.critical(f"{still_open}: cross leg close FAILED ({type(e).__name__}: {e}) "
                                    f"after the other leg's {reason} -- this leg is still OPEN and "
                                    f"now unhedged, needs manual attention")
            realized_pnl = unrealized_pnl(ctx.position, price_a, price_b)
            logger.info(f"EXIT {ctx.symbol_a}/{ctx.symbol_b}: {reason} (native bracket), pnl={realized_pnl:.2f}")
            await ibkr.cancel_orders_and_confirm([
                ctx.position.tp_order_id, ctx.position.hard_stop_order_id,
                ctx.position.timed_exit_order_id, ctx.position.pnl_stop_order_id, ctx.position.trail_order_id])
            ibkr.cancel_order_id(ctx.position.hedge_tp_order_id)
            ibkr.cancel_order_id(ctx.position.hedge_hard_stop_order_id)
            ibkr.cancel_order_id(ctx.position.hedge_timed_exit_order_id)
            daily_state.realized_pnl_today += realized_pnl
            record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
            if not reentry_allowed(reason, now):
                daily_state.reentry_locked_out = True
            daily_state.trade_report_events.append(report_event_from_position(ctx, now, exit_reason=reason))
            ctx.position = None
            continue

        # Per the independent audit's Finding #8, verified real 2026-08-24:
        # the bracket-placement flow already flattens immediately if a
        # bracket is REJECTED at submission, but nothing was watching for
        # one that was accepted and later died (Cancelled/Inactive) on its
        # own without ever filling -- a connectivity blip, an IBKR-side
        # risk check, or a stale-order sweep could silently kill a working
        # order sometime after entry. Checked here, AFTER the fill check
        # above, so this only fires when NONE of the three bracket orders
        # filled.
        tp_abandoned = ibkr.bracket_order_abandoned(ctx.position.tp_order_id)
        hard_stop_abandoned = ibkr.bracket_order_abandoned(ctx.position.hard_stop_order_id)
        timed_exit_abandoned = ibkr.bracket_order_abandoned(ctx.position.timed_exit_order_id)
        pnl_stop_abandoned = ibkr.bracket_order_abandoned(ctx.position.pnl_stop_order_id)
        trail_abandoned = ibkr.bracket_order_abandoned(ctx.position.trail_order_id)
        if (tp_abandoned or hard_stop_abandoned or timed_exit_abandoned or pnl_stop_abandoned
                or trail_abandoned):
            logger.error(f"{ctx.symbol_a}/{ctx.symbol_b}: bracket order(s) died without filling "
                         f"(TP abandoned={tp_abandoned}, HardStop abandoned={hard_stop_abandoned}, "
                         f"TimedExit abandoned={timed_exit_abandoned}, "
                         f"PnLStop abandoned={pnl_stop_abandoned}, "
                         f"TrailingTP abandoned={trail_abandoned}) -- flattening immediately, "
                         f"this position cannot be left without full protection")
            await ibkr.cancel_orders_and_confirm([
                ctx.position.tp_order_id, ctx.position.hard_stop_order_id,
                ctx.position.timed_exit_order_id, ctx.position.pnl_stop_order_id, ctx.position.trail_order_id])
            cancel_hedge_bracket(ibkr, ctx.position)
            flatten_ids = await ibkr.flatten_symbols({ctx.symbol_a, ctx.symbol_b})
            rejection = await flatten_rejection(ibkr, flatten_ids)
            if rejection:
                logger.critical(f"{ctx.symbol_a}/{ctx.symbol_b}: flatten after bracket abandonment "
                                 f"was ALSO REJECTED ({rejection}) -- position may still be open at "
                                 f"the broker with reduced/no protection, keeping it tracked")
                continue
            realized_pnl = unrealized_pnl(ctx.position, price_a, price_b)
            logger.warning(f"EXIT {ctx.symbol_a}/{ctx.symbol_b}: bracket_abandoned, pnl={realized_pnl:.2f}")
            daily_state.realized_pnl_today += realized_pnl
            record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
            daily_state.reentry_locked_out = True  # not a take_profit close -- never eligible for re-entry
            daily_state.trade_report_events.append(
                report_event_from_position(ctx, now, exit_reason="bracket_abandoned"))
            ctx.position = None
            continue

        # Time-based exits the native bracket can't cover: the 3:59pm EOD
        # flatten -- the Maximum Hold Time is the Timed Exit bracket
        # order's own job now, not checked here anymore.
        reason = check_exit(ctx.position, now)
        if reason is None:
            continue
        await ibkr.cancel_orders_and_confirm([
            ctx.position.tp_order_id, ctx.position.hard_stop_order_id,
            ctx.position.timed_exit_order_id, ctx.position.pnl_stop_order_id, ctx.position.trail_order_id])
        cancel_hedge_bracket(ibkr, ctx.position)
        flatten_ids = await ibkr.flatten_symbols({ctx.symbol_a, ctx.symbol_b})
        rejection = await flatten_rejection(ibkr, flatten_ids)
        if rejection:
            logger.error(f"{ctx.symbol_a}/{ctx.symbol_b}: flatten for {reason} was REJECTED "
                         f"({rejection}) -- position may still be open at the broker, "
                         f"keeping it tracked and retrying next cycle")
            continue
        realized_pnl = unrealized_pnl(ctx.position, price_a, price_b)
        logger.info(f"EXIT {ctx.symbol_a}/{ctx.symbol_b}: {reason}, pnl={realized_pnl:.2f}")
        daily_state.realized_pnl_today += realized_pnl
        record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
        if not reentry_allowed(reason, now):
            daily_state.reentry_locked_out = True
        daily_state.trade_report_events.append(report_event_from_position(ctx, now, exit_reason=reason))
        ctx.position = None

    # Tracks divergence for every flat, qualified pair EVERY cycle,
    # regardless of whether some OTHER pair currently holds a position --
    # confirmed live 2026-08-17: this used to sit behind "if not
    # book_is_flat(contexts): return" (below), which froze ctx.last_deviation
    # (Divergence Watch's own data source) for every other pair the entire
    # time any one position was open. Only the actual entry DECISION below
    # needs to wait for the book to be flat -- tracking candidates in the
    # background does not.
    signals = []
    for ctx in contexts:
        if ctx.position is not None or len(ctx.spread_history) < 2:
            continue
        # NEWS STOP, pre-entry half -- per the client 2026-09-01: "Or
        # cancel selected pair, if not yet a position." Sticky once set
        # (see PairContext.news_cancelled's own comment), so a pair
        # already cancelled this session skips tracking entirely -- it
        # must not reappear on Divergence Watch or become enterable again.
        if ctx.news_cancelled:
            continue
        if NEWS_STOP_ENABLED and news_stop_cancel(ctx.symbol_a, ctx.symbol_b, blacklisted):
            # Same "only the leg that came from the raw news signal gets a
            # new Excluded Symbols record" reasoning as the exit-side News
            # Stop above.
            for symbol in (ctx.symbol_a, ctx.symbol_b):
                if symbol in news_flagged_now:
                    excluded.add(symbol, now=now)
            ctx.news_cancelled = True
            logger.warning(f"{ctx.symbol_a}/{ctx.symbol_b}: cancelled, news_stop "
                            f"(no position yet)")
            continue
        current_spread = ctx.spread_history.iloc[-1]
        current_equilibrium = equilibrium_spread(ctx.spread_history).iloc[-1]
        if pd.isna(current_equilibrium):
            continue
        price_a = feed.book.latest_close(ctx.symbol_a)
        price_b = feed.book.latest_close(ctx.symbol_b)
        if price_a is None or price_b is None:
            continue
        deviation = current_spread - current_equilibrium
        # Per the client 2026-08-25: Divergence Watch should show pairs that
        # passed all the filters even before they have a trigger, not just
        # the ones already waiting on one -- update ctx.last_deviation for
        # EVERY flat, tracked pair, not only ones with a real trigger level.
        # This does NOT affect the entry decision itself, which still needs
        # a real trigger and stays gated below.
        ctx.last_deviation = deviation
        # Per the client 2026-09-03, Expansion_Leg.pdf: the baseline the
        # Expansion Leg formula measures from. See its own docstring.
        update_current_divergence_event(ctx, deviation, price_a, price_b, now)
        if ctx.divergence_trigger_level is None:
            logger.debug(f"{ctx.symbol_a}/{ctx.symbol_b}: tracked, current=${abs(deviation):.4f}, "
                         f"no trigger established yet")
            continue
        # Allowed Divergence Gap -- per the client 2026-08-23: "Allowed
        # Divergence Gap does not apply to Exception Maximum Divergence
        # determined at Entry." Skipped (always True) while
        # using_exception_trigger; otherwise the real D1-D2 test against
        # ctx.divergence_events (the base trigger's own reverting-within-
        # 2-hours events -- see establish_base_trigger).
        gap_ok = ctx.using_exception_trigger or passes_gap_condition(
            ctx.divergence_events, equity_with_loan_value, ctx.day_sizing_qty_a + ctx.day_sizing_qty_b)
        signal = PairSignal(
            symbol_a=ctx.symbol_a, symbol_b=ctx.symbol_b, hedge_ratio=ctx.hedge_ratio,
            deviation=deviation,
            divergence_trigger_level=ctx.divergence_trigger_level,
            price_a=price_a, price_b=price_b, half_life_minutes=ctx.half_life_minutes or 30.0,
            trigger_reached_today=ctx.trigger_reached_today,
            # bool(...) -- ctx.cointegration_broke_before_931 is None until
            # recalculate_trigger_exceptions has run at least once for this
            # pair, which must read as "not yet confirmed broken", not the
            # PairSignal field's own True default (that default exists only
            # so old tests that don't exercise this condition keep passing).
            cointegration_broke_before_931=bool(ctx.cointegration_broke_before_931),
            cointegration_still_broken=ctx.cointegration_still_broken,
            gap_ok=gap_ok,
            # Each of the five sticky flags independent for the session --
            # see PairContext.johansen_ever_unconfirmed/hurst_ever_above_threshold's
            # own comment. Requires 20-day and 1-day Johansen, 20-day and
            # 1-day Hurst (2026-09-06: "They are two separate
            # calculations"), and now 1 Day Pearson too (2026-09-07),
            # confirmed sticky the same way: "Validation is once. Not
            # continuous."
            # Per the client 2026-09-08, mid-session, with nothing having
            # qualified all day: "Also remove the 1 Day: Pearson and Hurst.
            # Keep 1 Day Johansen." 1-day Hurst was under 0.50 on all 37
            # tracked pairs (max 0.4953) and 1-day Pearson never produced a
            # value at all, so between them they blocked every pair. The
            # 20-day pair and 1-day Johansen still gate normally.
            # hurst_ever_above_threshold_1day/pearson_ever_above_threshold_1day
            # are still computed and shown on the dashboard, just not read
            # here -- same dormant-not-deleted convention as every other
            # removed gate in this file.
            # 1-day Johansen REMOVED as a gate per the client 2026-09-09:
            # "Ok. Remove 1 Day Johansen" -- confirmed after watching the
            # live Divergence Watch, which the two 20-day gates plus 1-day
            # Hurst had already narrowed enough on their own.
            johansen_hurst_confirmed=(
                ctx.johansen_ever_unconfirmed and ctx.hurst_ever_above_threshold),
            # Checklist_090926.pdf condition A REMOVED as a gate per the
            # client 2026-09-08: "Please remove the condition of 100% mean
            # reversion after Maximum Divergence. So, we can trade today."
            # It read True on all 37 tracked pairs, blocking the entire
            # watch list. Still computed (see
            # PairContext.mean_reverted_since_max_divergence) and shown on
            # the dashboard, just no longer passed to entry_eligible.
            #
            # The DISTINCT >50% condition, per the client 2026-09-13: "it
            # should have at least >50% mean reversion before entry." This
            # one IS passed through -- see
            # PairSignal.reverted_over_50pct_since_max_divergence's own
            # comment for why it isn't dormant like the line above.
            reverted_over_50pct_since_max_divergence=ctx.reverted_over_50pct_since_max_divergence,
        )
        # ctx.last_deviation already set above, before the trigger-gated block.
        # Per the client 2026-08-19: "Trigger has not previously been
        # reached at any point during the current RTH session" -- sticky
        # for the rest of today the moment it's ever touched, checked
        # AFTER building this cycle's signal (so the signal above reflects
        # whether it had been reached BEFORE now, not including this tick).
        if signal.abs_deviation >= ctx.divergence_trigger_level:
            ctx.trigger_reached_today = True
        logger.info(f"{ctx.symbol_a}/{ctx.symbol_b} divergence check: current=${signal.abs_deviation:.4f} "
                    f"trigger=${ctx.divergence_trigger_level:.4f} distance=${signal.distance_to_trigger:.4f} "
                    f"trigger_reached_today={signal.trigger_reached_today} "
                    f"eligible={entry_eligible(signal, permanent_blacklisted, now, daily_state) if not (halted_market or news_scan_down) else 'halted'}")
        if not (halted_market or news_scan_down) and entry_eligible(signal, permanent_blacklisted, now, daily_state):
            signals.append((signal, ctx))

    # One position at a time, across the whole book -- and per the client
    # 2026-08-19, "Maximum Trades per day = 2" (with the Take-Profit-before-
    # 11am re-entry carve-out) caps the whole day regardless, but this still
    # matters for the brief window between an entry order being placed and
    # its fill actually confirming. The quantity formula also
    # spends 98% of EquityWithLoanValue on a single pair, so a second
    # concurrent entry breaks sizing outright: two pairs each sized against
    # the full equity asked IBKR for more initial margin than was
    # available (error 201). Deliberately checked HERE, after
    # tracking above has already run for every candidate this cycle -- not
    # before it -- so this gate only ever blocks placing a new trade, never
    # the background divergence tracking those candidates need regardless
    # of whether the book is flat.
    if not book_is_flat(contexts):
        return

    # Per the client 2026-08-18: "From The Divergence Watch List, select the
    # closest to trigger and enter." SUPERSEDED 2026-09-12 -- Adjusted
    # Spread Efficiency (see rank_eligible_signals_by_adjusted_spread_efficiency's
    # own docstring) fully replaces this as the tiebreak, per the client:
    # "Yes. Fully replace trigger tiebreaker. Use initial margin as
    # described." Ranking against the 9:29am equity snapshot, not a fresh
    # live read -- this only needs to be good enough to RANK candidates
    # against each other, the chosen pair's own entry below still does the
    # careful, fresh effective_sizing_equity read before it actually sizes.
    chosen = await rank_eligible_signals_by_adjusted_spread_efficiency(
        signals, ibkr, equity_with_loan_value, now)
    if chosen is None:
        return
    ctx = next(c for s, c in signals if s is chosen)

    # Method 1 (Mathematical Return Attribution) -- per the client
    # 2026-09-03's "Expansion Leg" pivot ("You don't need Hedge Ratio
    # anymore... We will buy only 1 leg. The expansion leg"): decides
    # which ONE stock to trade and which direction. See
    # expansion_leg_decision's own docstring for the formula. Prices read
    # once here and reused through the rest of entry, same reasoning as
    # entry_beta below -- there are awaits ahead and the tick thread keeps
    # moving both series.
    price_a = ctx.price_history_a.iloc[-1]
    price_b = ctx.price_history_b.iloc[-1]
    decision = expansion_leg_decision(ctx, price_a, price_b)
    if decision is None:
        # Per a third-party audit finding 2026-09-14: this used to always
        # say "no current divergence event baseline yet", which stopped
        # being the only real reason once the 2026-09-13 70% driver-share
        # floor (DRIVER_STOCK_MIN_SHARE) was added -- decision is also
        # None when there's real price history but neither leg clears
        # 70% of the divergence expansion, or (2026-09-15) the driver
        # stock fails the prior-excursion-dominance or prior-excursion-
        # direction filters. None of these are distinguishable from each
        # other from here, so all are named.
        logger.warning(f"Skipping {ctx.symbol_a}/{ctx.symbol_b}: expansion_leg_decision has nothing "
                        f"to trade -- no price history yet, neither leg reached the "
                        f"{DRIVER_STOCK_MIN_SHARE:.0%} driver share floor, or the driver stock failed "
                        f"the prior-excursion-dominance or prior-excursion-direction filter")
        return
    traded_symbol, action, delta_p = decision
    # Size against the SAME beta the signal was built from -- entry_beta
    # feeds Method 1's C_B term and the ongoing spread-based exit trigger
    # (see open_single_leg_position). ctx.hedge_ratio is mutated
    # continuously by the tick thread, and there are awaits below, so
    # re-reading it here would let the exit trigger drift against a
    # different beta than the one this entry was actually decided on.
    entry_beta = chosen.hedge_ratio

    # Single Stock Initial Margin -- per the client 2026-09-10: "remove
    # whatIF. Calculate quantity using 35% Initial Margin Deposit."
    # A flat INITIAL_MARGIN_FRACTION of the stock's own price, no IBKR
    # whatIf probe, no refinement re-probe. IBKR's whatIf initMarginChange
    # had under-reported a short's real requirement on 2026-09-10 (quoted
    # ~36%, the live order needed ~66%) and, on 2026-08-11, had been 2-4%
    # optimistic for combos -- the client's own fixed figure replaces it.
    # traded_symbol's current price comes from the same PairSignal the
    # entry was decided on.
    traded_price = chosen.price_a if traded_symbol == ctx.symbol_a else chosen.price_b
    if not traded_price or traded_price <= 0:
        logger.error(f"{traded_symbol}: no live price to size against -- skipping this entry")
        return
    margin_per_share = traded_price * INITIAL_MARGIN_FRACTION
    single_stock_margin = margin_per_share

    # Spec: "EquityWithLoanValue <= 9:29am ET EquityWithLoanValue" -- cap at
    # the morning snapshot, but follow the account down if it has lost ground
    # since. Falls back to the snapshot if the live read fails.
    try:
        live_elv = await ibkr.get_equity_with_loan_value()
        sizing_elv = effective_sizing_equity(equity_with_loan_value, live_elv)
        if sizing_elv < equity_with_loan_value:
            logger.info(f"Sizing off live equity {sizing_elv:,.2f} "
                        f"(below the 9:29am snapshot {equity_with_loan_value:,.2f})")
    except (asyncio.TimeoutError, ValueError) as e:
        # Per the independent audit's Finding #6, verified real 2026-08-24:
        # the whole point of "EquityWithLoanValue <= 9:29am snapshot" is a
        # DAILY-LOSS safety cap -- silently sizing off the fixed 9:29am
        # value on a data FAILURE defeats that, since a live query timeout
        # says nothing about whether the account has actually lost ground
        # since. Skip the entry instead, matching the margin-probe failure
        # a few lines above this -- no safe quantity without a real number.
        logger.error(f"{traded_symbol}: live equity read failed "
                     f"({type(e).__name__}) -- skipping this entry rather than sizing off a "
                     f"possibly-stale snapshot")
        return

    quantity = compute_final_single_stock_quantity(
        sizing_elv, single_stock_margin, ctx.divergence_trigger_level, ENTRY_TRIGGER_DISTANCE_MIN_PCT)
    if quantity <= 0:
        logger.warning(f"Skipping {traded_symbol}: computed quantity is zero")
        return
    logger.info(f"{traded_symbol}: initial size {quantity} shares ({action} at ${traded_price:.2f}, "
                f"{INITIAL_MARGIN_FRACTION:.0%} initial margin ${margin_per_share:.2f}/share, "
                f"sizing equity ${sizing_elv:,.2f})")

    # Real-order margin probe -- per the client 2026-09-10: "Use $1, execute
    # the order, full entry plus bracket submitted as one real set. So it
    # rejects on 201 and we read the margin, then resize and submit for
    # real." The 35% figure above is only the STARTING size; this confirms
    # it against IBKR's real order engine (not whatIf) by submitting the
    # entry as a non-fillable LMT plus all four bracket legs, reading the
    # required margin off the 201 rejection, and shrinking to fit. Briefly
    # simplified to entry-only 2026-09-15, then REVERTED the same day after
    # a real live incident (see probe_bracket_initial_margin's own comment):
    # INTU's entry-only probe found 29 shares fit easily, but the real
    # bracket placed right after was rejected needing nearly double that --
    # the bracket legs DO change the true required margin, this just
    # doesn't show up unless they're in the probe too.
    timed_exit_activation = now + timedelta(minutes=MAX_HOLD_MINUTES)
    # 6 iterations, not 3 -- live-verified 2026-09-11 against the real
    # account: an oversized ADBE short converged 139 -> 84 -> 56 -> 42
    # (FITS) needing four full probe rounds, because required margin does
    # NOT scale proportionally with quantity here (139->84 cut share count
    # 40% but only cut required margin 9%, a large near-fixed component
    # unrelated to size). A 3-iteration cap silently aborted this exact,
    # real, fitting trade -- correctness matters far more than the ~1.5s
    # per extra round costs against a multi-minute entry window.
    # Per the client 2026-09-14: "I want to see the initial margin required
    # by IBKR and see if it is the normal" -- the LAST real 201 rejection
    # this loop hits is the closest real IBKR-confirmed figure available
    # (see Position.entry_margin_ibkr_probed's own comment for why it
    # isn't an exact match to the final traded quantity).
    last_probed_margin: float | None = None
    last_probed_margin_qty: int | None = None
    for _probe_iter in range(6):
        required_margin = await ibkr.probe_bracket_initial_margin(
            traded_symbol, action, quantity, traded_price, timed_exit_activation)
        if required_margin is None:
            break  # this size already fits -- submit it for real
        last_probed_margin, last_probed_margin_qty = required_margin, quantity
        fitted = int(quantity * (sizing_elv * 0.98) / required_margin)
        if fitted >= quantity or fitted <= 0:
            logger.warning(f"Skipping {traded_symbol}: real-order margin probe wants "
                           f"${required_margin:,.2f} for {quantity} shares, cannot shrink to fit "
                           f"${sizing_elv * 0.98:,.2f}")
            return
        logger.info(f"{traded_symbol}: margin probe -> ${required_margin:,.2f} needed for {quantity}, "
                    f"resizing to {fitted}")
        quantity = fitted
    else:
        logger.warning(f"Skipping {traded_symbol}: margin probe still rejecting after 6 resizes")
        return
    logger.info(f"{traded_symbol}: final size {quantity} shares after margin probe"
                + (f" (last IBKR-confirmed figure: ${last_probed_margin:,.2f} for {last_probed_margin_qty} shares)"
                   if last_probed_margin is not None else ""))

    current_spread = ctx.spread_history.iloc[-1]
    order_id, filled_qty, avg_price = await ibkr.place_single_stock_trade(traded_symbol, action, quantity)

    # place_single_stock_trade IS the Leg Risk Safeguard for this design
    # (2-second timeout, cancels the unfilled remainder, keeps whatever
    # partial quantity did fill) -- filled_qty==0 covers both an outright
    # rejection and a fully-timed-out zero fill, so both are the same
    # "don't track a position" case here.
    if filled_qty <= 0:
        logger.warning(f"{traded_symbol}: entry order {order_id} did not fill at all -- not tracking a position")
        return

    current_equilibrium = equilibrium_spread(ctx.spread_history).iloc[-1]
    # Real gross notional of the one real leg actually held, not
    # sizing_elv*0.98 (the INTENDED order size) -- same reasoning as the
    # old two-leg version: a partial fill would otherwise understate/
    # overstate unrealized_pnl_pct's denominator.
    entry_notional = filled_qty * avg_price
    # IBKR's own whatIf-quoted margin for the real position actually held --
    # per the client 2026-08-17: "Pair Entry Initial Margin (the figure
    # returned by IBKR what IF)". margin_per_share x actual shares traded.
    entry_margin = margin_per_share * filled_qty

    # The untraded leg's own live price is recorded too (not just the
    # traded leg's real fill price) -- purely informational, since the
    # exit trigger (position_spread) reads entry_equilibrium/
    # entry_trigger_level, not these, but it's what the OLD entry_price_a/
    # entry_price_b fields have always held for the report.
    ctx.position = open_single_leg_position(
        ctx.symbol_a, ctx.symbol_b, traded_symbol, action, filled_qty,
        entry_price_a=avg_price if traded_symbol == ctx.symbol_a else price_a,
        entry_price_b=avg_price if traded_symbol == ctx.symbol_b else price_b,
        entry_notional=entry_notional, half_life_minutes=ctx.half_life_minutes or 30.0,
        current_spread=current_spread, hedge_ratio=entry_beta, now=now,
        entry_equilibrium=current_equilibrium, entry_margin=entry_margin,
        entry_trigger_level=ctx.divergence_trigger_level,
        entry_margin_ibkr_probed=last_probed_margin, entry_margin_ibkr_probed_qty=last_probed_margin_qty,
    )
    ctx.position.entry_order_id = order_id
    ctx.day_sizing_qty_a = quantity
    # Per the client 2026-08-19: "Maximum Trades per day = 2" -- set the
    # moment a trade is confirmed real (BEFORE the Timed Exit placement
    # below), not after it. Confirmed live 2026-08-19: an unhandled
    # exception in bracket placement left this line unreached when it came
    # after, which would have let the book accept an extra entry on top of
    # an already unprotected position if another pair had qualified before EOD.
    daily_state.trades_taken_today += 1
    # Persisted immediately, for the same reason the increment itself comes
    # before Timed Exit placement: this is the counter enforcing "Maximum
    # Trades per day = 2", and a restart between here and the end-of-cycle
    # save would forget this trade ever happened.
    save_daily_state(daily_state, DAILY_STATE_PATH)
    save_pair_trigger_state(
        [(c.symbol_a, c.symbol_b, c.trigger_reached_today) for c in contexts], PAIR_TRIGGER_STATE_PATH, now=now)

    # Take Profit, Hard Stop, and Timed Exit, placed as a real 3-way OCA
    # bracket on the single traded stock -- restored 2026-09-04, less than
    # a day after TP/Hard Stop moved to bot-monitored logic: the client
    # sent worked Resting Take Profit/Hard Stop Loss formulas (How To Set
    # Resting Take Profit.pdf) that translate the spread-trigger condition
    # onto this one stock's own price. If placement fails for ANY reason
    # the position must not be left unprotected -- flatten it immediately
    # rather than let it sit until EOD.
    closing_action = "SELL" if action == "BUY" else "BUY"
    hard_stop_order_id = tp_order_id = timed_exit_order_id = None
    try:
        # Hard Stop Loss @ ENTRY -- per the client 2026-09-09: "Hard Stop
        # Loss @ Entry you will use Trigger-13% + Trigger", the SAME
        # worst-case distance sizing itself was built from, deliberately
        # wider (safer) than the real entry_divergence -- placed immediately
        # since it needs no re-try, being the exact distance the position's
        # own quantity was already sized to tolerate.
        #
        # Re-setting this to the tighter, real resting_hard_stop_price
        # (entry_divergence + trigger, his "AFTER ENTRY" step, with
        # "Continuous re-try logic until DONE") is NOT yet built -- retry
        # bounds are still an open question sent to the client 2026-09-09,
        # unanswered: how many attempts/how long, what happens to the
        # position if it never succeeds, and whether this applies to
        # Take Profit/Timed Exit/the PnL Stop too or Hard Stop Loss only.
        hard_stop_price = wide_hard_stop_price(action, avg_price, ctx.divergence_trigger_level)
        take_profit_price = resting_take_profit_price(
            action, avg_price, ctx.divergence_trigger_level, ctx.position.entry_divergence)
        timed_exit_activation = now + timedelta(minutes=MAX_HOLD_MINUTES)
        # -5% PnL Stop Loss, the fourth OCA leg -- per the client
        # 2026-09-09: "It is Ibkr PnL. 5% of EWLV." Priced off the SAME
        # sizing_elv the quantity was derived from, so the stop reflects the
        # equity the position was actually sized against.
        pnl_stop_price = pnl_stop_loss_price(
            action, avg_price, filled_qty, sizing_elv, MAX_LOSS_PCT)
        hard_stop_order_id, tp_order_id, timed_exit_order_id, pnl_stop_order_id, trail_order_id = \
            await ibkr.place_single_stock_bracket(
                traded_symbol, closing_action, filled_qty, hard_stop_price, take_profit_price,
                timed_exit_activation, pnl_stop_price=pnl_stop_price,
                trailing_take_profit_amount=TRAILING_TAKE_PROFIT_AMOUNT)
        # placeOrder is fire-and-forget -- a broker-side rejection lands on
        # the error() callback moments later, same as the entry order's
        # own rejection check above.
        # Checked concurrently -- order_was_rejected waits its full 3s
        # timeout when an order is NOT rejected (the normal case), so a
        # sequential `or` chain of five would idle 15s on every clean
        # entry. gather runs them together: 3s total. Same pattern
        # flatten_rejection already uses.
        _brk = await asyncio.gather(
            ibkr.order_was_rejected(hard_stop_order_id),
            ibkr.order_was_rejected(tp_order_id),
            ibkr.order_was_rejected(timed_exit_order_id),
            ibkr.order_was_rejected(pnl_stop_order_id),
            ibkr.order_was_rejected(trail_order_id),
        )
        bracket_rejection = next((r for r in _brk if r), None)
        if bracket_rejection:
            raise RuntimeError(f"bracket order rejected by IBKR -- {bracket_rejection}")
        ctx.position.hard_stop_order_id = hard_stop_order_id
        ctx.position.tp_order_id = tp_order_id
        ctx.position.timed_exit_order_id = timed_exit_order_id
        ctx.position.pnl_stop_order_id = pnl_stop_order_id
        ctx.position.trail_order_id = trail_order_id
    except Exception as e:
        logger.error(f"{traded_symbol}: bracket order placement failed "
                     f"({type(e).__name__}: {e}) -- flattening immediately, this position "
                     f"cannot be left without Take Profit/Hard Stop/Timed Exit protection")
        await ibkr.cancel_orders_and_confirm(
            [hard_stop_order_id, tp_order_id, timed_exit_order_id, pnl_stop_order_id, trail_order_id])
        flatten_ids = await ibkr.flatten_symbols({ctx.symbol_a, ctx.symbol_b})
        rejection = await flatten_rejection(ibkr, flatten_ids)
        if rejection:
            # Can't clear the position as if it's flat when the flatten
            # itself was refused -- it's still open at the broker. Leave
            # ctx.position tracked (its bracket order id fields stay None/
            # rejected) so the exit loop's own check_exit fallback -- EOD --
            # still eventually catches it.
            logger.critical(f"{traded_symbol}: fail-safe flatten after bracket "
                             f"rejection was ALSO REJECTED ({rejection}) -- position genuinely has "
                             f"no protection right now, still tracked, waiting on EOD")
            return
        realized_pnl = unrealized_pnl(ctx.position, price_a, price_b)
        daily_state.realized_pnl_today += realized_pnl
        record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
        daily_state.reentry_locked_out = True  # not a take_profit close -- never eligible for re-entry
        daily_state.trade_report_events.append(
            report_event_from_position(ctx, now, exit_reason="bracket_placement_failed"))
        ctx.position = None
        return
    logger.info(f"ENTER {traded_symbol}: {action} {filled_qty} (delta_p={delta_p:+.4f}, "
                f"entry orderId={order_id}, trigger={ctx.divergence_trigger_level:.4f})")

    # OPPOSITE LEG DISABLED, per the client 2026-09-09, hours after asking
    # for it: "Please, ignore buying two legs. Let's stick to single leg. As
    # it was before these update requests... Ignore all these instructions
    # today. I will stick with single stock." That withdrew (E), (F), (G),
    # the cross-leg exits and the Leg B direction rules together.
    #
    # place_opposite_leg and everything it depends on (position_sizing's
    # (E)/(F)/(G), risk_manager's hedge_* fields and attach_hedge_leg,
    # screening.normalized_hedge_ratio, the cross-leg exit reconciliation)
    # are all left in place and tested but UNREACHED -- the same
    # dormant-not-deleted convention this file uses for every other
    # withdrawn mechanism. Re-enabling is this one call, nothing else.
    #
    # Everything downstream already treats hedge_qty == 0 as "there is no
    # second leg", so with this call gone the entry, exit, P&L and report
    # paths behave exactly as the single-stock version did before today.
    # await place_opposite_leg(ibkr, ctx, traded_symbol, action, filled_qty, avg_price,
    #                          price_a, price_b, sizing_elv, now)


def cancel_hedge_bracket(ibkr, position) -> None:
    """Cancels the opposite leg's three resting bracket orders.

    Called on EVERY exit path that flattens the pair, not just the bracket
    reconciliation one. Those paths already flatten both symbols (they pass
    {symbol_a, symbol_b}), so the POSITION closes correctly either way --
    but a working TP/Hard Stop/Timed Exit left behind on a now-flat leg is
    live at the broker, and price touching it later would execute it and
    open a brand new naked position out of nothing. No-ops safely when
    there is no opposite leg (the ids are None)."""
    ibkr.cancel_order_id(position.hedge_tp_order_id)
    ibkr.cancel_order_id(position.hedge_hard_stop_order_id)
    ibkr.cancel_order_id(position.hedge_timed_exit_order_id)


async def place_opposite_leg(
    ibkr, ctx: PairContext, traded_symbol: str, action: str, filled_qty: float,
    avg_price: float, price_a: float, price_b: float, sizing_elv: float, now: datetime,
) -> None:
    """(E)/(F) -- the OPPOSITE leg, per the client 2026-09-09: "Execute the
    Opposite Leg in the opposite direction... Apply Beta to determine
    opposite stock shares quantity. Determine how many Stock B hedge ratio
    shares Balance EWLV can execute."

    Deliberately runs AFTER the expansion leg is open AND bracketed, and
    never raises: the expansion leg is the position, and it is already
    fully protected by its own bracket by the time this is called. If the
    opposite leg cannot be sized, funded, filled, or bracketed, the trade
    continues as an expansion-leg-only position (exactly what shipped
    before today) rather than unwinding a good, protected entry. Every
    failure path here therefore logs and returns instead of flattening.

    The reverse is NOT true -- see the cross-leg exit in run_check_cycle:
    once this leg IS on, either leg's bracket firing closes both."""
    opposite_symbol = ctx.symbol_b if traded_symbol == ctx.symbol_a else ctx.symbol_a
    opposite_price = price_b if opposite_symbol == ctx.symbol_b else price_a
    opposite_action = "SELL" if action == "BUY" else "BUY"

    # Which leg is the reference fixed at 1 share, and what the other leg
    # scales to -- per the client: "there are no fractional shares for a
    # short leg." See screening.normalized_hedge_ratio.
    reference_leg, scaled_ratio = normalized_hedge_ratio(ctx.hedge_ratio)
    traded_is_reference = (reference_leg == "A") == (traded_symbol == ctx.symbol_a)
    opposite_ratio = scaled_ratio if traded_is_reference else 1.0

    # (E) EWLV less the expansion leg's own COST, per his clarification.
    balance = equity_balance_after_leg(sizing_elv, int(filled_qty), avg_price)
    hedge_qty = opposite_leg_quantity(balance, opposite_price, int(filled_qty), opposite_ratio)
    if hedge_qty <= 0:
        logger.warning(f"{opposite_symbol}: no opposite leg placed -- EWLV balance {balance:,.2f} "
                       f"funds {int(balance / opposite_price) if opposite_price > 0 else 0} shares "
                       f"at {opposite_price:.2f} (beta implied {int(filled_qty * opposite_ratio)}). "
                       f"Continuing as an expansion-leg-only position.")
        return

    try:
        hedge_order_id, hedge_filled, hedge_avg = await ibkr.place_single_stock_trade(
            opposite_symbol, opposite_action, hedge_qty)
    except Exception as e:
        logger.error(f"{opposite_symbol}: opposite leg order failed ({type(e).__name__}: {e}) -- "
                     f"continuing as an expansion-leg-only position")
        return
    if hedge_filled <= 0:
        logger.warning(f"{opposite_symbol}: opposite leg order {hedge_order_id} did not fill -- "
                       f"continuing as an expansion-leg-only position")
        return

    attach_hedge_leg(ctx.position, opposite_symbol, opposite_action,
                     hedge_filled, hedge_avg, hedge_order_id)
    logger.info(f"ENTER opposite leg {opposite_symbol}: {opposite_action} {hedge_filled} "
                f"@ {hedge_avg:.4f} (beta ratio {opposite_ratio:.4f}, reference leg {reference_leg}, "
                f"EWLV balance {balance:,.2f})")

    # This leg gets its own resting bracket too -- per the client: "Exit both
    # Stock A and Stock B IF Hard Stop Loss triggers in either", same for
    # Take Profit. Prices come from the SAME formulas as the expansion leg,
    # measured against this leg's own fill price and its own direction.
    hedge_closing_action = "BUY" if opposite_action == "SELL" else "SELL"
    try:
        hedge_stop = resting_hard_stop_price(
            opposite_action, hedge_avg, ctx.divergence_trigger_level, ctx.position.entry_divergence)
        hedge_tp = resting_take_profit_price(
            opposite_action, hedge_avg, ctx.divergence_trigger_level, ctx.position.entry_divergence)
        h_stop_id, h_tp_id, h_timed_id, _, _ = await ibkr.place_single_stock_bracket(
            opposite_symbol, hedge_closing_action, hedge_filled, hedge_stop, hedge_tp,
            now + timedelta(minutes=MAX_HOLD_MINUTES))
        _hbrk = await asyncio.gather(
            ibkr.order_was_rejected(h_stop_id),
            ibkr.order_was_rejected(h_tp_id),
            ibkr.order_was_rejected(h_timed_id),
        )
        rejection = next((r for r in _hbrk if r), None)
        if rejection:
            raise RuntimeError(f"opposite leg bracket rejected by IBKR -- {rejection}")
        ctx.position.hedge_hard_stop_order_id = h_stop_id
        ctx.position.hedge_tp_order_id = h_tp_id
        ctx.position.hedge_timed_exit_order_id = h_timed_id
    except Exception as e:
        # The opposite leg is filled but unprotected. Close THIS leg only --
        # the expansion leg keeps its own working bracket and stays open, so
        # the outcome is the expansion-leg-only position that shipped before
        # today rather than an unprotected hedge sitting until EOD.
        logger.error(f"{opposite_symbol}: opposite leg bracket failed ({type(e).__name__}: {e}) -- "
                     f"closing this leg only, expansion leg stays open and protected")
        ibkr.cancel_order_id(ctx.position.hedge_hard_stop_order_id)
        ibkr.cancel_order_id(ctx.position.hedge_tp_order_id)
        ibkr.cancel_order_id(ctx.position.hedge_timed_exit_order_id)
        try:
            await ibkr.flatten_symbols({opposite_symbol})
        except Exception:
            logger.critical(f"{opposite_symbol}: opposite leg is filled, unbracketed, and the "
                            f"unwind ALSO failed -- it is exposed until EOD")
        ctx.position.hedge_qty = 0.0
        ctx.position.hedge_symbol = ""


async def run_intraday_scan_in_background(
    news_flagged: ExcludedSymbols, market_halt: MarketHalt, news_scan_health: "NewsScanHealth",
) -> None:
    """Gemini's search-grounded calls occasionally spike to 2-5+ seconds of
    latency (longer with our own retry/backoff on a transient 503). Fired via
    create_task rather than awaited inline, and run_intraday_scan itself
    pushed onto a thread via to_thread, so a slow scan never delays or blocks
    the primary 1-minute check_exit/entry loop -- it just updates the
    blacklist file whenever it finishes, which the next check cycle picks up
    naturally.

    The scan is isolated in its own try/except -- confirmed live 2026-08-07:
    a Gemini read timeout (the network hanging past the 30s HttpOptions
    timeout, not a 503 _generate_with_retry already handles) propagated out
    of run_intraday_scan uncaught, which killed this whole task before it
    reached run_internal_circuit_breaker below. Since this task is fired
    with create_task and never awaited, that failure surfaced only as
    asyncio's generic "Task exception was never retrieved" and silently
    skipped that cycle's circuit breaker check -- a safety-critical feature
    with no dependency on Gemini being up."""
    try:
        newly_flagged = await asyncio.to_thread(run_intraday_scan, news_flagged)
        if newly_flagged:
            logger.info(f"Intraday news scan flagged: {newly_flagged}")
        news_scan_health.healthy = True
    except Exception:
        logger.warning("Intraday news scan failed, skipping this cycle", exc_info=True)
        # Per the client 2026-08-17: "If Gemini news API is down, no
        # trading." -- clears automatically the next time a scan succeeds.
        news_scan_health.healthy = False
    # Internal circuit breaker (real SPY price data) replaces the disabled
    # Gemini-based scan -- see run_internal_circuit_breaker's docstring.
    await run_internal_circuit_breaker(market_halt, datetime.now(ET))


async def run_edgar_scan_in_background(edgar_state: SecEdgarMonitorState, news_flagged: ExcludedSymbols) -> None:
    """Fired via create_task, same reasoning as run_intraday_scan_in_background:
    isolated in its own try/except so a SEC EDGAR outage or a 403 (a bad
    User-Agent, or the 10 req/sec rate limit) never kills the task silently
    or delays the primary check-exit/entry loop. A second, independent
    source alongside the Gemini/GPT scan above -- both write into the same
    news_flagged (ExcludedSymbols), so a failure here doesn't need its own
    NewsScanHealth-style circuit breaker; the Gemini/GPT side covers "no
    trading if news scanning is down" on its own."""
    try:
        newly_flagged = await asyncio.to_thread(run_edgar_scan, edgar_state, news_flagged)
        if newly_flagged:
            logger.info(f"SEC EDGAR 8-K monitor flagged: {newly_flagged}")
    except Exception:
        logger.warning("SEC EDGAR 8-K scan failed, skipping this cycle", exc_info=True)


CIRCUIT_BREAKER_THRESHOLD_PCT = 2.0  # crash or rally, either direction -- raised from 1.0 per client 2026-09-08


async def run_internal_circuit_breaker(market_halt: MarketHalt, now: datetime) -> None:
    """Real-price-data circuit breaker, replacing the Gemini search-grounded
    scan disabled 2026-08-05 -- successive calls the same evening reported
    the S&P 500 move anywhere from +1.25% to +1.91%, and the client's own
    independent recheck against real data found the actual peak move that
    day never even reached 1%. Uses SPY as a real-data index proxy, same
    technique get_todays_market_hours already uses for the RTH calendar.

    2% threshold, crash or rally, expires at 4pm ET the SAME day rather than
    a rolling 24h window -- raised from 1% per the client 2026-09-08
    ("Has the S&P 500 Index rallied or crashed today by 2%?"), the original
    1% having been set 2026-08-06.

    Baseline changed from the previous session's close to TODAY's own 9:30am
    RTH open, per the client 2026-09-04: "You calculate 1% from 9:30am+Now.
    Not from previous close." Confirmed live the same day: this fired a
    false +1.06% reading vs the previous close (773.22) while SPY's real
    intraday move from today's own open was flat (+0.05%) -- a single bad
    or incomplete price at the open skews either baseline the same way, so
    this change alone would not have caught that specific incident, but
    it's the baseline the client explicitly asked for going forward."""
    now_utc = now.astimezone(timezone.utc)
    start = f"{(now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
    end = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
    bars = _previous_and_current_rth(await asyncio.to_thread(fetch_intraday_warmup, ["SPY"], start, end))
    if bars.empty:
        return
    ts = pd.to_datetime(bars["ts_event"], utc=True).dt.tz_convert(ET)
    today_bars = bars.loc[ts.dt.date == now.date()]
    if today_bars.empty:
        return  # no bars for today's own session yet -- nothing to compare against
    today_open = today_bars["open"].iloc[0]
    current = today_bars["close"].iloc[-1]
    move_pct = (current - today_open) / today_open * 100
    if abs(move_pct) < CIRCUIT_BREAKER_THRESHOLD_PCT:
        return
    which = "crash" if move_pct < 0 else "rally"
    expires_at = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET).astimezone(timezone.utc)
    reason = f"internal circuit breaker: {which}, SPY {move_pct:+.2f}% vs today's 9:30am open (real price data)"
    market_halt.raise_halt(reason, now=now_utc, expires_at=expires_at)
    logger.warning(f"INTERNAL CIRCUIT BREAKER raised, all S&P 500 symbols excluded until 4pm ET -- {reason}")


async def compute_opening_30min_breakdown(now: datetime) -> dict | None:
    """Real-price-data version of the report's Opening 30-Minute Breakdown,
    per client 2026-08-06 -- same SPY-as-index-proxy technique as the
    internal circuit breaker, not an LLM search. Returns None if there's not
    yet a full previous + current session in the window (e.g. very first
    session ever, or fetch came back empty)."""
    now_utc = now.astimezone(timezone.utc)
    start = f"{(now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
    end = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
    bars = _previous_and_current_rth(await asyncio.to_thread(fetch_intraday_warmup, ["SPY"], start, end))
    if bars.empty:
        return None
    ts = pd.to_datetime(bars["ts_event"], utc=True).dt.tz_convert(ET)
    sessions = sorted(ts.dt.date.unique())
    if len(sessions) < 2:
        return None
    previous_session, today_session = sessions[-2], sessions[-1]
    today_mask = ts.dt.date == today_session
    today = bars.loc[today_mask]
    today_ts = ts.loc[today_mask]
    if today.empty:
        return None
    prev_close = bars.loc[ts.dt.date == previous_session, "close"].iloc[-1]
    open_price = today["open"].iloc[0]
    thirty_min_cutoff = datetime.combine(today_session, dtime(10, 0), tzinfo=ET)
    thirty_min_bars = today.loc[today_ts <= thirty_min_cutoff]
    thirty_min_price = thirty_min_bars["close"].iloc[-1] if not thirty_min_bars.empty else today["close"].iloc[-1]
    intraday_high = today["high"].max()
    return {
        "previous_close": prev_close, "open": open_price,
        "thirty_min_price": thirty_min_price, "intraday_high": intraday_high,
        "open_pct": (open_price - prev_close) / prev_close * 100,
        "thirty_min_pct_prev_close": (thirty_min_price - prev_close) / prev_close * 100,
        "thirty_min_pct_open": (thirty_min_price - open_price) / open_price * 100,
        "intraday_high_pct": (intraday_high - prev_close) / prev_close * 100,
    }


async def run_universe_screen_in_background(
    contexts: list[PairContext],
    feed: LiveBarFeed, ref: universe_screening.UniverseReferenceData,
    excluded: ExcludedSymbols, ibkr: IBKRClient, equity_with_loan_value: float, now: datetime,
    screen_state: dict,
) -> None:
    """Fired via create_task, same reasoning as run_intraday_scan_in_background:
    a full universe screen touches every S&P 500 symbol's price history and runs
    ADF/Johansen/Hurst on every untracked candidate pair, both slow enough that
    doing it inline would delay the 1-minute check_exit/entry loop.

    Only ADDS newly-qualifying pairs -- an already-tracked pair's ongoing
    eligibility (including one that stops qualifying) is already handled every
    cycle by update_live_cointegration_stats, and a pair with an open position
    is never touched here regardless.

    screen_state["running"] guards against overlapping invocations --
    confirmed live 2026-08-20: this screen can run longer than
    UNIVERSE_SCREEN_INTERVAL, so a second screen starting before the first
    finished each took its own stale `already_tracked` snapshot and both
    appended the SAME newly-qualifying pair as a separate PairContext,
    doubling every pair on the dashboard and firing duplicate entry orders
    for the same pair seconds apart. The caller checks this flag before
    creating a new task; set/cleared here so it stays accurate regardless
    of which return path or exception this run takes."""
    screen_state["running"] = True
    try:
        await _run_universe_screen(
            contexts, feed, ref, excluded, ibkr, equity_with_loan_value, now)
    finally:
        screen_state["running"] = False


async def _run_universe_screen(
    contexts: list[PairContext],
    feed: LiveBarFeed, ref: universe_screening.UniverseReferenceData,
    excluded: ExcludedSymbols, ibkr: IBKRClient, equity_with_loan_value: float, now: datetime,
) -> None:
    """The actual screen -- split out from run_universe_screen_in_background
    purely so that function's try/finally guard doesn't have to track every
    early-return path in here separately."""
    market_entry_pairs = load_market_entry_list() or []
    already_tracked = {frozenset((c.symbol_a, c.symbol_b)) for c in contexts}
    # Per the client 2026-08-26: news exclusion doesn't block discovery
    # either, only the permanent blacklist does -- see screen_once's own
    # docstring.
    permanent_blacklisted = load_permanent_blacklist()
    symbols_needed = sorted(set(ref.all_symbols) | {s for pair in market_entry_pairs for s in pair})

    now_utc = now.astimezone(timezone.utc)
    start = f"{(now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
    end = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
    bars = _previous_and_current_rth(await asyncio.to_thread(fetch_intraday_warmup, symbols_needed, start, end))
    if bars.empty:
        logger.warning("universe screen: no price data returned this cycle, skipping")
        return
    closes = bars.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
    # Previous session only, for establish_base_trigger -- same split
    # warm_up_pairs uses. A pair discovered here only ever runs between
    # 10:00-10:30am ET (this task's own caller), well after
    # TRIGGER_RECALC_START_TIME, so run_check_cycle's existing per-cycle
    # recalculate_trigger_exceptions call picks up this fresh ctx (it has
    # position=None and real prior_session_price_a/b) on its very next
    # cycle -- no separate handling needed here.
    today_et = now.astimezone(ET).date()
    prior_bars = bars[_session_date_et(bars["ts_event"]) < today_et]
    prior_closes = (prior_bars.set_index(["ts_event", "symbol"])["close"].unstack("symbol")
                     if not prior_bars.empty else pd.DataFrame())

    results = await asyncio.to_thread(
        universe_screening.screen_once, closes, ref, market_entry_pairs, already_tracked, permanent_blacklisted)
    if not results:
        logger.info(f"universe screen: {len(already_tracked)} pairs already tracked, none new qualified this cycle")
        return

    for result in results:
        a, b = result.symbol_a, result.symbol_b
        if a not in closes.columns or b not in closes.columns:
            continue
        both = closes[[a, b]].dropna()
        ctx = PairContext(
            symbol_a=a, symbol_b=b, hedge_ratio=result.hedge_ratio,
            spread_history=pd.Series(dtype=float), industry=result.industry,
            discovered_live=True,
            correlation_60d=result.correlation, correlation_20d=result.correlation_recent,
        )
        ctx.price_history_a, ctx.price_history_b = both[a], both[b]
        ctx.spread_history = compute_spread(ctx.price_history_a, ctx.price_history_b, ctx.hedge_ratio)
        # Same 20 trading day Johansen/Hurst seed warm_up_pairs does for the
        # session-start pairs -- without this, a pair discovered mid-session
        # here would never accumulate MIN_BARS_FOR_LIVE_JOHANSEN_HURST bars
        # in the ~30 minutes left before the entry window closes, silently
        # blocking it from ever entering (same class of bug day_sizing_qty's
        # own comment above already flags for a different field).
        try:
            jh_bars = await asyncio.to_thread(_fetch_johansen_hurst_history, [a, b], now)
            if a in jh_bars.columns and b in jh_bars.columns:
                jh_full = jh_bars[[a, b]].dropna()
                jh_both = jh_full.tail(JOHANSEN_HURST_WINDOW_BARS)
                ctx.johansen_hurst_price_a, ctx.johansen_hurst_price_b = jh_both[a], jh_both[b]
                jh_1day = jh_full.tail(JOHANSEN_HURST_WINDOW_1DAY_BARS)
                ctx.johansen_hurst_price_a_1day, ctx.johansen_hurst_price_b_1day = jh_1day[a], jh_1day[b]
        except Exception as e:
            logger.warning(f"{a}/{b}: could not fetch 20 day Johansen/Hurst history: {e}")
        if a in prior_closes.columns and b in prior_closes.columns:
            prior_both = prior_closes[[a, b]].dropna()
            ctx.prior_session_price_a, ctx.prior_session_price_b = prior_both[a], prior_both[b]
            establish_base_trigger(ctx)
        # Reference-sizing whatIf REMOVED here 2026-08-26, per the client:
        # "WhatIF should not be by 9:30am or any other time or way, apart
        # from" the real entry-time one -- "WhatIF is performed in the same
        # millisecond of Entry... when the qualified pair first appeared
        # inside 13%-25% of Trigger to get to Trigger." This used to probe
        # a placeholder "BUY a / SELL b" whatIf just to populate
        # day_sizing_qty for the report's adverse-excursion-as-%-of-equity
        # figure on pairs that never actually trade -- day_sizing_qty now
        # simply stays at its 0.0 default until (if) this pair gets a real
        # entry, at which point the real entry-time whatIf (main.py's
        # entry flow, gated on the 13%-25% window like every other pair)
        # sets it from the actual fill.
        contexts.append(ctx)
        feed.add_symbols([a, b])
        logger.info(f"universe screen: new pair qualified {a}/{b} ({result.industry}) "
                    f"corr={result.correlation} adf_p={result.adf_pvalue} half_life={result.half_life}min")


async def _get_elv_with_retry(ibkr: IBKRClient, retries: int = 3) -> float:
    """Finding #15 from the independent audit, confirmed real: the two
    EquityWithLoanValue snapshots taken during the 9:25-9:30am startup
    window (elv_at_925 for stop loss, equity_with_loan_value for sizing)
    had no protection at all -- a single transient IBKR timeout during
    market open's own busiest reconnect window propagated straight out of
    run_daily_session() and killed the whole session before any trading
    started, exactly the "crash here means zero trading for the whole day"
    risk already called out for the pre-market news scan just above this.
    Retried with backoff (same shape as news_blacklist._generate_with_retry
    and data_source's yfinance retries) rather than falling back to some
    placeholder equity figure -- a wrong number silently poisons every
    stop-loss/sizing calculation for the rest of the day, which is worse
    than a loud failure after genuinely exhausting retries."""
    for attempt in range(retries):
        try:
            return await ibkr.get_equity_with_loan_value()
        except asyncio.TimeoutError:
            if attempt == retries - 1:
                raise
            logger.warning(f"EquityWithLoanValue fetch timed out (attempt {attempt + 1}/{retries}), retrying")
            await asyncio.sleep(2 ** attempt)


async def status_heartbeat(state: dict) -> None:
    """Keeps data/bot_status.json fresh during the long pre-market waits.

    write_status() used to be called once at startup and then not again until
    the 9:30am monitoring loop -- well past the dashboard's staleness TTL, so
    the client would see "Offline" for over an hour every morning while the
    bot was in fact running and waiting correctly.

    state["contexts"] -- per the client 2026-09-01: "Tomorrow, by 9am it
    loads. We want it loaded by 9am." Set once PAIR_LOAD_TIME's overnight
    cache load succeeds (see its own call site), so every heartbeat tick
    from 9am onward keeps showing the loaded qualified pairs count/list
    instead of the next tick silently reverting it back to None."""
    while True:
        write_status(state.get("status", "waiting"), account=state.get("account", ""),
                      contexts=state.get("contexts"))
        await asyncio.sleep(30)


def _seconds_until_next_bar_close(now: datetime, buffer_seconds: float = BAR_CLOSE_CHECK_BUFFER_SECONDS) -> float:
    """Seconds to sleep so the next check cycle lands just after the next
    1-minute bar actually closes, instead of CHECK_INTERVAL_SECONDS after
    whenever THIS cycle happened to finish -- see CHECK_INTERVAL_SECONDS's
    own comment for why. Never returns a non-positive number: if a slow
    cycle already ran past the next boundary, the boundary after that is
    targeted instead, so this can't come back negative or cause a
    zero-second busy loop."""
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    target = next_minute + timedelta(seconds=buffer_seconds)
    if target <= now:
        target += timedelta(minutes=1)
    return (target - now).total_seconds()


async def run_daily_session() -> None:
    ibkr = IBKRClient(account=os.getenv("IBKR_ACCOUNT"))
    ibkr.connect_and_start(
        os.getenv("IBKR_HOST", "127.0.0.1"),
        int(os.getenv("IBKR_PORT", "7497")),
        int(os.getenv("IBKR_CLIENT_ID", "1")),
    )
    await ibkr.wait_ready()
    # excluded is the client-visible Excluded Symbols list -- written to
    # ONLY when a News Stop actually flattens a position (see
    # run_check_cycle). news_flagged is the raw "currently in the news"
    # signal both scans write into every cycle -- see news_blacklist.
    # NEWS_FLAGGED_PATH's own comment.
    excluded = ExcludedSymbols()
    news_flagged = ExcludedSymbols(path=NEWS_FLAGGED_PATH)
    edgar_state = SecEdgarMonitorState()
    market_halt = MarketHalt()
    news_scan_health = NewsScanHealth()
    heartbeat_state = {"status": "waiting", "account": ibkr.account or ""}
    heartbeat = asyncio.create_task(status_heartbeat(heartbeat_state))

    now = datetime.now(ET)
    logger.info(f"=== Session start {now} ===")

    market_hours = await ibkr.get_todays_market_hours()
    if market_hours is None:
        logger.info("Today is a market holiday (SPY reqContractDetails reports CLOSED) -- skipping the session entirely")
        heartbeat.cancel()
        write_status("holiday", account=ibkr.account or "")
        return
    _, close_dt = market_hours
    if is_early_close_day(close_dt.time()):
        logger.info(f"Today is an early close day (close={close_dt.time()}) -- v7 spec says no trading, skipping the session entirely")
        heartbeat.cancel()
        write_status("holiday", account=ibkr.account or "")
        return

    # Per the client 2026-09-01: "Tomorrow, by 9am it loads. We want it
    # loaded by 9am." -- distinct from actual pair SELECTION, which he was
    # explicit stays at 9:30am ("Pair Selection cannot start by 9am...
    # We use only RTH Data"). This only ever attempts the overnight
    # CACHE (a local file read, not a network fetch, so there's nothing
    # for the "RTH data only" objection to apply to) -- if it's missing
    # or stale, contexts stays None here and the live fallback still only
    # runs at SESSION_START_TIME below, never before 9:30.
    await sleep_until_et(PAIR_LOAD_TIME)
    cached_contexts = load_cached_qualified_pairs(datetime.now(ET))
    if cached_contexts is not None:
        logger.info(f"loaded {len(cached_contexts)} qualified pairs from last night's cache at 9am")
        # Keeps the dashboard showing this from 9am on -- see
        # status_heartbeat's own comment on state["contexts"].
        heartbeat_state["contexts"] = cached_contexts

    await sleep_until_et(BLACKLIST_SCAN_TIME)
    # Backgrounded via to_thread for consistency with the intraday scan's
    # own pattern -- the Gemini client now has a 30s timeout (2026-08-07
    # fix, see news_blacklist._client), so this can no longer hang
    # indefinitely, but staying off the event loop directly is still safer.
    #
    # Wrapped in try/except -- confirmed live 2026-08-10: a Gemini API
    # billing failure (429 RESOURCE_EXHAUSTED, "prepayment credits are
    # depleted") propagated out of this call uncaught and crashed the
    # ENTIRE session before it ever started trading, taking down a full
    # paper-testing day over a news-scan failure that has nothing to do
    # with the trading logic itself. The intraday scan already had this
    # same protection (2026-08-07 fix); this, the pre-market scan, is even
    # more safety-critical since it runs before anything else and a crash
    # here means zero trading for the whole day, yet it had no protection
    # at all until now.
    try:
        newly_flagged = await asyncio.to_thread(run_daily_scan, news_flagged)
        logger.info(f"Flagged today: {newly_flagged}")
        news_scan_health.healthy = True
    except Exception:
        logger.error("Pre-market news scan failed, continuing the session without it "
                     "(no symbols flagged from this scan)", exc_info=True)
        # Per the client 2026-08-17: "If Gemini news API is down, no
        # trading." -- blocks new entries (not exits) until a later scan,
        # pre-market or intraday, actually succeeds.
        news_scan_health.healthy = False
    elv_at_925 = await _get_elv_with_retry(ibkr)
    logger.info(f"EquityWithLoanValue snapshot for stop loss (frozen for the day): {elv_at_925}")

    await sleep_until_et(ELV_SNAPSHOT_TIME)
    equity_with_loan_value = await _get_elv_with_retry(ibkr)
    logger.info(f"EquityWithLoanValue snapshot (frozen for the day): {equity_with_loan_value}")

    await sleep_until_et(SESSION_START_TIME)
    # heartbeat stays alive through pair selection/warm-up/reference-data
    # below, NOT cancelled here -- confirmed live 2026-08-31: cancelling it
    # at 9:30am sharp, before select_pairs_from_previous_session (itself a
    # real network fetch taking 1-2+ minutes) and warm_up_pairs had a
    # chance to run, reopened the exact gap this heartbeat was built to
    # close (see its own docstring) -- the dashboard's 30s staleness TTL
    # showed "Offline" for several minutes of a perfectly healthy startup.
    # Cancelled below, right before the monitoring loop starts writing its
    # own richer status.
    #
    # Briefly moved to a 9:00am PAIR_SELECTION_TIME the same day, then
    # reverted -- the client 2026-09-01: "Pair Selection cannot start by
    # 9am. Pair Selection starts by 9:30am. We use only RTH Data." Back to
    # exactly the original 2026-08-29 timing.
    now = datetime.now(ET)
    # Per the client 2026-09-01: "There is no pairs selection today...
    # by 9:30am, it will test for Johansen and Hurst. Which is a matter
    # of seconds." Reuses the 9am cache load above if it succeeded (see
    # PAIR_LOAD_TIME's own comment) -- only re-checked here, not
    # re-loaded, since the cache can't have changed in the last 30
    # minutes. A fresh None here (cache was missing or stale at 9am too)
    # falls back to a live selection, now correctly at 9:30am.
    if SINGLE_STOCK_EXCURSION_MODE:
        candidates = load_single_stock_watchlist(today=now.date())
        if candidates is None:
            logger.info("no usable single-stock watchlist -- computing from prior session")
            ref = universe_screening.build_reference_data()
            fetch_now_utc = now.astimezone(timezone.utc)
            start = f"{(fetch_now_utc - timedelta(days=WARMUP_LOOKBACK_CALENDAR_DAYS)).date()}T00:00:00"
            end = fetch_now_utc.strftime("%Y-%m-%dT%H:%M:%S")
            bars = _previous_and_current_rth(fetch_intraday_warmup(ref.all_symbols, start, end))
            candidates = compute_single_stock_watchlist(ref.all_symbols, bars)
            save_single_stock_watchlist(candidates, for_date=now.date())
        contexts = [SingleStockContext(c) for c in candidates]
        logger.info(f"SINGLE_STOCK_EXCURSION_MODE active: loaded {len(contexts)} candidates")
    else:
        contexts = cached_contexts
        if contexts is None:
            logger.info("no usable overnight cache -- running a live previous-session pair selection")
            contexts = select_pairs_from_previous_session(now)
        else:
            logger.info(f"loaded {len(contexts)} qualified pairs from last night's cache")
        warm_up_pairs(contexts, now)

        # Restored, not reset -- same reasoning as daily_state just below, per
        # a real 2026-09-14 incident: a mid-session restart (unrelated to this
        # flag) reset trigger_reached_today to False in memory for every pair,
        # and a pair that had legitimately already reached its trigger earlier
        # that session went on to actually enter a trade it should never have
        # been eligible for. See risk_manager.save_pair_trigger_state's own
        # comment.
        restored_trigger_state = load_pair_trigger_state(PAIR_TRIGGER_STATE_PATH, now=now)
        for ctx in contexts:
            if restored_trigger_state.get(f"{ctx.symbol_a}/{ctx.symbol_b}"):
                ctx.trigger_reached_today = True

    all_symbols = sorted({s for ctx in contexts for s in (ctx.symbol_a, ctx.symbol_b) if s})
    feed = LiveBarFeed(all_symbols)
    feed.start()

    # Restored, not reset -- see load_daily_state. A mid-session restart
    # used to silently zero trades_taken_today and reentry_locked_out.
    daily_state = load_daily_state(DAILY_STATE_PATH)
    last_intraday_scan: datetime | None = None
    last_edgar_scan: datetime | None = None
    last_universe_screen: datetime | None = None
    last_warmup_retry: datetime | None = None
    # Guards against overlapping universe screens -- confirmed live
    # 2026-08-20: a full screen (ADF/Johansen/Hurst across the whole S&P
    # 500) can take longer than UNIVERSE_SCREEN_INTERVAL (5 minutes), so a
    # second screen started before the first finished each took its own
    # stale "already tracked" snapshot and both appended the SAME
    # newly-qualifying pair as a separate PairContext -- every pair showing
    # doubled on the dashboard with different prices, and two duplicate
    # entry orders firing for the same pair seconds apart at slightly
    # different sizes. A plain dict (not a bool local) since it has to be
    # mutated from inside run_universe_screen_in_background itself.
    universe_screen_state = {"running": False}
    warmup_retry_state = {"running": False}

    # Clear a stale stop flag before the loop starts. The dashboard's STOP
    # button can write this at ANY time, but it's only checked (and
    # unlinked) from inside the main loop below -- so pressing STOP after
    # the loop has already exited (e.g. 2pm, while the process is still
    # alive waiting to send the 4pm report) left the file sitting there
    # with nothing to consume it. The next morning's session would then
    # break out on its very first iteration and trade nothing all day,
    # explained only by a single log line. Audit finding 2026-08-27.
    if STOP_FLAG_PATH.exists():
        logger.warning("Clearing a stale stop flag left over from a previous session")
        STOP_FLAG_PATH.unlink()

    universe_ref: "universe_screening.UniverseReferenceData | None" = None
    if UNIVERSE_SCREENING_ENABLED:
        try:
            universe_ref = await asyncio.to_thread(universe_screening.build_reference_data)
        except Exception:
            # Wikipedia scrape + yfinance across ~500 symbols, neither of which
            # this bot depends on for anything else -- a failure here should
            # disable universe screening for the day, not the whole session.
            logger.exception("universe screen: could not build reference data, disabled for today")

    try:
        while datetime.now(ET).time() < MONITOR_LOOP_END_TIME:
            if STOP_FLAG_PATH.exists():
                logger.info("Stop flag detected (dashboard STOP button) -- flattening and ending session early")
                STOP_FLAG_PATH.unlink()
                break

            now = datetime.now(ET)
            # Per the client 2026-08-25: "If by 10:30am there is no entry,
            # Dashboard should shut down." No open position guarantees
            # trades_taken_today == 0 already covers "no entry happened",
            # but checked explicitly too since a stray position with
            # trades_taken_today somehow still 0 (shouldn't happen, but
            # this is the difference between a cosmetic bug and a
            # position getting abandoned mid-session) must never be walked
            # away from. Ends the session the same way the loop's normal
            # end-of-day exit does -- falls through to the EOD block below,
            # which is a safe no-op with nothing open, and the dashboard's
            # own "ended" status write already happens there.
            if (now.time() > LATEST_ENTRY_TIME and daily_state.trades_taken_today == 0
                    and book_is_flat(contexts)):
                logger.info("No entry by 10:30am ET -- ending the session early, per the client's "
                            "'Dashboard should shut down' instruction")
                break
            if (INTRADAY_NEWS_SCAN_START_TIME <= now.time() < INTRADAY_NEWS_SCAN_END_TIME
                    and (last_intraday_scan is None or now - last_intraday_scan >= INTRADAY_SCAN_INTERVAL)):
                asyncio.create_task(run_intraday_scan_in_background(news_flagged, market_halt, news_scan_health))
                last_intraday_scan = now

            if (EDGAR_WINDOW_START <= now.time() <= EDGAR_WINDOW_END
                    and (last_edgar_scan is None or now - last_edgar_scan >= INTRADAY_SCAN_INTERVAL)):
                asyncio.create_task(run_edgar_scan_in_background(edgar_state, news_flagged))
                last_edgar_scan = now

            if (universe_ref is not None and not universe_screen_state["running"]
                    and EARLIEST_ENTRY_TIME <= now.time() <= LATEST_ENTRY_TIME
                    and (last_universe_screen is None or now - last_universe_screen >= UNIVERSE_SCREEN_INTERVAL)):
                asyncio.create_task(run_universe_screen_in_background(
                    contexts, feed, universe_ref,
                    excluded, ibkr, equity_with_loan_value, now, universe_screen_state))
                last_universe_screen = now

            still_missing = [c for c in contexts
                              if len(c.price_history_a) == 0 or c.base_trigger_level is None]
            if (still_missing and not warmup_retry_state["running"]
                    and (last_warmup_retry is None or now - last_warmup_retry >= WARMUP_RETRY_INTERVAL)):
                asyncio.create_task(run_warmup_retry_in_background(still_missing, now, warmup_retry_state))
                last_warmup_retry = now

            try:
                await run_check_cycle(contexts, feed, ibkr, excluded, news_flagged, daily_state,
                                       equity_with_loan_value, elv_at_925, market_halt,
                                       news_scan_health)
            except Exception:
                # An unhandled error here used to escape the loop entirely,
                # which skipped the 3:59pm flatten below and could strand an
                # open position overnight. Log it and carry on to the next
                # cycle instead -- staying alive to flatten matters more than
                # any single cycle succeeding.
                logger.exception("Check cycle failed -- continuing to the next cycle")
            # Catch-all persist, at the cycle boundary rather than at each
            # of the ~10 sites that set reentry_locked_out/trading_halted
            # inside the cycle -- one call here can't miss one of them, and
            # runs even when the cycle above raised partway through. Same
            # catch-all reasoning covers trigger_reached_today now too --
            # see save_pair_trigger_state's own comment for the 2026-09-14
            # incident this closes.
            save_daily_state(daily_state, DAILY_STATE_PATH)
            save_pair_trigger_state(
                [(c.symbol_a, c.symbol_b, c.trigger_reached_today) for c in contexts], PAIR_TRIGGER_STATE_PATH, now=now)
            write_status(
                "running", account=ibkr.account or "", elv=equity_with_loan_value,
                contexts=contexts, daily_pnl=daily_state.realized_pnl_today,
                trading_halted=daily_state.trading_halted,
                news_scan_healthy=news_scan_health.healthy,
            )
            if not heartbeat.cancelled():
                # Cancelled here, after this loop's OWN first status write,
                # not any earlier -- see heartbeat's own comment above on why
                # cancelling it before this point reopens the exact staleness
                # gap it exists to close. The first run_check_cycle can itself
                # take ~2 minutes (Johansen/Hurst over the full 20-day window
                # for every newly tracked pair, confirmed live 2026-08-28),
                # so select_pairs_from_previous_session finishing isn't late
                # enough either -- only this line guarantees the dashboard
                # never goes dark between the two.
                heartbeat.cancel()
            await asyncio.sleep(_seconds_until_next_bar_close(datetime.now(ET)))
    finally:
        feed.stop()

    # A position still open at the hard flatten never goes through the normal
    # exit path -- flatten_all() below closes it directly at the broker
    # without touching the bot's own per-pair state -- so it has to be
    # recorded into the report here, or the trade that was open right up to
    # the close would be the one missing from it.
    eod_now = datetime.now(ET)
    for ctx in contexts:
        if ctx.position is None:
            continue
        if len(ctx.price_history_a):
            price_a, price_b = ctx.price_history_a.iloc[-1], ctx.price_history_b.iloc[-1]
            live_pnl = unrealized_pnl(ctx.position, price_a, price_b)
            ctx.position.worst_unrealized_pnl = min(ctx.position.worst_unrealized_pnl, live_pnl)
            # Independent audit finding, confirmed real: every OTHER exit
            # path in this file adds its realized_pnl into
            # daily_state.realized_pnl_today (and calls record_exit) --
            # this EOD path was the one exception, so the dashboard's final
            # "ended" status daily_pnl figure silently omitted whatever was
            # still open going into the 3:59pm flatten. The emailed report
            # itself is unaffected (build_report_html sums per-trade
            # figures straight from trade_report_events, appended correctly
            # just below), but the dashboard's own total was wrong.
            daily_state.realized_pnl_today += live_pnl
            record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
        daily_state.trade_report_events.append(report_event_from_position(ctx, eod_now, exit_reason="eod"))

    logger.info("=== 3:59pm flatten ===")
    # Retried once if rejected -- this is the LAST chance to flatten before
    # the session ends and this process disconnects, no next cycle exists
    # to retry from the way every other flatten call site can. If it's
    # still rejected after a retry, there's nothing more this process can
    # do -- logged as loudly as possible since a position could otherwise
    # be left open overnight with nothing watching it at all.
    eod_flatten_ids = await ibkr.flatten_all()  # whole book flat at EOD, so flatten_all is correct here
    eod_rejection = await flatten_rejection(ibkr, eod_flatten_ids)
    if eod_rejection:
        logger.error(f"EOD flatten was REJECTED ({eod_rejection}) -- retrying once")
        await asyncio.sleep(2)
        retry_ids = await ibkr.flatten_all()
        retry_rejection = await flatten_rejection(ibkr, retry_ids)
        if retry_rejection:
            logger.critical(f"EOD flatten REJECTED again on retry ({retry_rejection}) -- "
                             f"a position may be left open overnight with nothing watching it")
    write_status(
        "ended", account=ibkr.account or "", elv=equity_with_loan_value,
        contexts=contexts, daily_pnl=daily_state.realized_pnl_today,
        trading_halted=daily_state.trading_halted,
    )

    await sleep_until_et(REPORT_SEND_TIME)  # spec: report goes out at 4:00pm ET
    breakdown = await compute_opening_30min_breakdown(datetime.now(ET))
    html = build_report_html(daily_state.trade_report_events, equity_with_loan_value, contexts, breakdown)
    try:
        send_report(html)
        logger.info("Post-trade report sent")
    except RuntimeError as e:
        logger.error(f"Could not send post-trade report: {e}")

    ibkr.disconnect()  # releases ibapi's socket reader thread

    # Per the client 2026-09-01: "Today's session closes by 4pm. Selection
    # is 9:30-4pm." -- today's own RTH session is now fully closed, so
    # this is the earliest point where computing tomorrow's list is valid.
    # After disconnect(), not before -- guarantees zero interference with
    # today's actual trading or the report send above, both already done.
    #
    # asyncio.to_thread since 2026-09-12: this now runs the full S&P 500
    # scan (select_pairs_from_full_universe_prior_session), genuinely
    # CPU-bound (hundreds of candidates' correlation math, plus a
    # per-bar Johansen/Hurst replay for whichever survive that), not the
    # small Market-Entry-List-only fetch this used to be -- no reason to
    # block the event loop for it even though nothing else needs it
    # before the process exits.
    await asyncio.to_thread(cache_qualified_pairs_for_tomorrow, datetime.now(ET))


if __name__ == "__main__":
    # Hard exit, in a finally so it happens on the error path too.
    # Databento's live client runs its own asyncio loop in a NON-daemon
    # thread, so once the session finishes the interpreter parks in
    # threading._shutdown waiting on it and the process never dies
    # (confirmed with py-spy: MainThread idle in _shutdown, databento_live
    # still in run_forever). A process that never exits leaves the systemd
    # unit "active", which makes the next morning's `systemctl start` a
    # silent no-op -- the bot would simply skip that trading day.
    exit_code = 0
    try:
        asyncio.run(run_daily_session())
    except BaseException:
        logger.exception("Session terminated abnormally")
        exit_code = 1
    finally:
        logging.shutdown()
        os._exit(exit_code)
