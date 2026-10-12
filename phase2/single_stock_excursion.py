"""Temporary Single-Stock Excursion Trading Mode for Phase 2.

Built per the client's directive (2026-09-16) when Databento historical access
to XNAS.ITCH and XNYS.PILLAR expired, blocking Johansen, Pearson, Hurst, and
Divergence calculations across pairs.

Summary of rules:
1. Selection: No Johansen, Pearson, Hurst, or Divergence. Full S&P 500 universe,
   single stocks only, no pairing.
2. Watchlist Filter: Computed from yesterday's prior RTH session (9:30am - 4:00pm ET).
   Passes if dominant excursion from that day's open (north or south) is at least
   300% greater than opposite excursion (passes_prior_excursion_dominance_filter).
3. Entry Trigger: Evaluated live between 9:32:00am ET and 10:30:00am ET.
   For each watchlist stock, 70% or more of today's high-to-low range so far
   must sit in the same direction as yesterday's dominant excursion
   (passes_prior_excursion_direction_filter). The first stock in the universe
   to qualify is entered. Direction: BUY if yesterday was upward, SELL if downward.
4. Bracket Order: Three legs in ONE OCA group:
   - Trailing Take Profit: Native IBKR TRAIL order at $1.00 trail.
   - Stop Loss: -2% PnL Stop Loss resting STP order.
   - Timed Exit: MKT order after 20 minutes maximum hold.
5. Risk & Position Sizing:
   - Sizing off effective_sizing_equity (9:29am snapshot capped, follows down).
   - 35% initial margin per share, probed against IBKR.
   - MAX_TRADES_PER_DAY = 2, with Take Profit before 10:30am ET unlocking 1 re-entry.
     All other exits (Stop Loss, Timed Exit, EOD) end the day.
6. Toggle:
   - SINGLE_STOCK_EXCURSION_MODE env var, strictly defaults to False (0).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from phase2.position_sizing import effective_sizing_equity
from phase2.risk_manager import (
    DailyState,
    Position,
    record_exit,
    reentry_allowed,
)

logger = logging.getLogger("pairs_bot.single_stock_excursion")

ET = ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SINGLE_STOCK_WATCHLIST_PATH = DATA_DIR / "single_stock_watchlist.json"

# Master Toggle: Defaults to False. Must be explicitly set to '1' or 'true' to activate.
SINGLE_STOCK_EXCURSION_MODE: bool = os.getenv("SINGLE_STOCK_EXCURSION_MODE", "0").lower() in ("1", "true")

# Strategy constants
PRIOR_EXCURSION_DOMINANCE_MIN_RATIO: float = 3.00  # >= 300% greater than opposite excursion
TODAY_EXCURSION_DIRECTION_MIN_SHARE: float = 0.70  # >= 70% of today's High-Low in excursion direction
ENTRY_WINDOW_START: time = time(9, 32, 0)
ENTRY_WINDOW_END: time = time(10, 30, 0)
MAX_HOLD_MINUTES: int = 20
TRAILING_TAKE_PROFIT_AMOUNT: float = 1.00  # $1.00 native IBKR TRAIL
PNL_STOP_LOSS_PCT: float = 0.02           # -2% of EquityWithLoanValue
INITIAL_MARGIN_FRACTION: float = 0.35      # 35% margin per share estimate
MAX_TRADES_PER_DAY: int = 2


def passes_prior_excursion_direction_filter(
    prior_open: float, prior_high: float, prior_low: float,
    today_open: float, today_high: float, today_low: float,
    threshold: float = TODAY_EXCURSION_DIRECTION_MIN_SHARE,
) -> bool:
    """Per the client 2026-09-15, worked example reproduced exactly:

        Prior RTH: Open=270.48, High=272.23, Low=262.68
        North excursion = High - Open = 1.75
        South excursion = Open - Low = 7.80
        Dominant = South (7.80 > 1.75)

        Today (as of the entry check): Open=252.00, High=253.00, Low=249.00
        Total range = High - Low = 4.00
        South portion (matching the dominant direction) = Open - Low = 3.00
        3.00 / 4.00 = 75% >= 70% -> PASS

    Open/High/Low here are each series' first/max/min close price over
    the relevant session -- same "1-minute close series stands in for
    true OHLC" convention establish_base_trigger already uses, no
    separate tick-level open field tracked anywhere in this file.

    A tie (north excursion exactly equals south) is arbitrarily treated
    as north-dominant -- unconfirmed with the client, this exact case
    wasn't in the worked example and is expected to be rare.

    Fails closed (False) if today's range is zero or negative -- nothing
    to measure a portion of yet, same "no signal" treatment used
    elsewhere in this file rather than a division by zero."""
    today_range = today_high - today_low
    if today_range <= 0:
        return False
    north_excursion = prior_high - prior_open
    south_excursion = prior_open - prior_low
    if north_excursion >= south_excursion:
        portion = max(today_high - today_open, 0.0)
    else:
        portion = max(today_open - today_low, 0.0)
    return (portion / today_range) >= threshold


def passes_prior_excursion_dominance_filter(
    prior_open: float, prior_high: float, prior_low: float,
    min_ratio: float = PRIOR_EXCURSION_DOMINANCE_MIN_RATIO,
) -> bool:
    """Per the client 2026-09-15, worked example reproduced exactly:

        Prior RTH: Open=270.48, High=272.23, Low=262.68
        Long excursion = High - Open = 1.75
        Short excursion = Open - Low = 7.80
        Dominant = Short (7.80 > 1.75)
        Short must be >=300% of Long: 7.80 / 1.75 x 100 = 445.71% -> PASS

    Symmetric the other way (dominant Long must be >=300% of Short) when
    the prior session's dominant excursion is Long instead. Same
    "1-minute close series as an OHLC proxy" convention as
    passes_prior_excursion_direction_filter -- no separate today
    component here at all, this reads ONLY the prior session's own Open/
    High/Low, checking how DECISIVE that day's move was, not whether
    today is continuing it (that's the other filter's job).

    A zero opposite excursion (price never moved at all the other way)
    passes automatically -- an infinitely lopsided day trivially clears
    any finite ratio requirement -- unless the dominant excursion is ALSO
    zero (a completely flat prior session), which fails closed."""
    north_excursion = prior_high - prior_open
    south_excursion = prior_open - prior_low
    dominant, opposite = (
        (north_excursion, south_excursion) if north_excursion >= south_excursion
        else (south_excursion, north_excursion))
    if opposite <= 0:
        return dominant > 0
    return (dominant / opposite) >= min_ratio


@dataclass
class SingleStockCandidate:
    symbol: str
    prior_open: float
    prior_high: float
    prior_low: float
    prior_close: float
    dominant_direction: str  # "BUY" if North excursion >= South excursion, else "SELL"
    north_excursion: float
    south_excursion: float
    dominance_ratio: float


def build_candidate_from_prior_bars(
    symbol: str, prior_bars: pd.DataFrame,
    min_ratio: float = PRIOR_EXCURSION_DOMINANCE_MIN_RATIO,
) -> SingleStockCandidate | None:
    """Evaluates a single symbol's prior session bars. Returns a SingleStockCandidate
    if it satisfies the 300% dominance filter, or None otherwise.
    """
    if prior_bars.empty:
        return None
    if "open" in prior_bars.columns:
        prior_open = float(prior_bars.iloc[0]["open"])
        prior_high = float(prior_bars["high"].max())
        prior_low = float(prior_bars["low"].min())
        prior_close = float(prior_bars.iloc[-1]["close"])
    else:
        prior_open = float(prior_bars.iloc[0]["close"])
        prior_high = float(prior_bars["close"].max())
        prior_low = float(prior_bars["close"].min())
        prior_close = float(prior_bars.iloc[-1]["close"])

    if not passes_prior_excursion_dominance_filter(prior_open, prior_high, prior_low, min_ratio):
        return None

    north = prior_high - prior_open
    south = prior_open - prior_low
    dominant_direction = "BUY" if north >= south else "SELL"
    dominant, opposite = (north, south) if north >= south else (south, north)
    ratio = (dominant / opposite) if opposite > 0 else float("inf")

    return SingleStockCandidate(
        symbol=symbol,
        prior_open=prior_open,
        prior_high=prior_high,
        prior_low=prior_low,
        prior_close=prior_close,
        dominant_direction=dominant_direction,
        north_excursion=north,
        south_excursion=south,
        dominance_ratio=ratio,
    )


def compute_single_stock_watchlist(
    all_symbols: list[str],
    prior_bars_by_symbol: dict[str, pd.DataFrame] | pd.DataFrame,
    min_ratio: float = PRIOR_EXCURSION_DOMINANCE_MIN_RATIO,
) -> list[SingleStockCandidate]:
    """Scans all symbols across the prior session and returns the list of
    candidates that passed the 300% dominance filter.
    """
    watchlist: list[SingleStockCandidate] = []
    if isinstance(prior_bars_by_symbol, pd.DataFrame):
        grouped = prior_bars_by_symbol.groupby("symbol")
        for symbol in sorted(all_symbols):
            if symbol not in grouped.groups:
                continue
            df = grouped.get_group(symbol)
            candidate = build_candidate_from_prior_bars(symbol, df, min_ratio)
            if candidate is not None:
                watchlist.append(candidate)
    else:
        for symbol in sorted(all_symbols):
            bars = prior_bars_by_symbol.get(symbol)
            if bars is None or bars.empty:
                continue
            candidate = build_candidate_from_prior_bars(symbol, bars, min_ratio)
            if candidate is not None:
                watchlist.append(candidate)
    return watchlist


def save_single_stock_watchlist(
    candidates: list[SingleStockCandidate],
    path: Path = SINGLE_STOCK_WATCHLIST_PATH,
    for_date: date | None = None,
) -> None:
    """Persists the watchlist to disk for tomorrow's session."""
    payload = {
        "for_date": (for_date or date.today()).isoformat(),
        "candidates": [
            {
                "symbol": c.symbol,
                "prior_open": c.prior_open,
                "prior_high": c.prior_high,
                "prior_low": c.prior_low,
                "prior_close": c.prior_close,
                "dominant_direction": c.dominant_direction,
                "north_excursion": c.north_excursion,
                "south_excursion": c.south_excursion,
                "dominance_ratio": c.dominance_ratio,
            }
            for c in candidates
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
    logger.info(f"Saved {len(candidates)} single-stock candidates to {path}")


def load_single_stock_watchlist(
    path: Path = SINGLE_STOCK_WATCHLIST_PATH,
    today: date | None = None,
) -> list[SingleStockCandidate] | None:
    """Loads today's single-stock watchlist from disk. Returns None if missing,
    corrupt, or from a different date.
    """
    today_iso = (today or date.today()).isoformat()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("for_date") != today_iso:
            return None
        candidates = [
            SingleStockCandidate(
                symbol=item["symbol"],
                prior_open=item["prior_open"],
                prior_high=item["prior_high"],
                prior_low=item["prior_low"],
                prior_close=item["prior_close"],
                dominant_direction=item["dominant_direction"],
                north_excursion=item["north_excursion"],
                south_excursion=item["south_excursion"],
                dominance_ratio=item["dominance_ratio"],
            )
            for item in data.get("candidates", [])
        ]
        return candidates
    except Exception as e:
        logger.warning(f"Could not load single-stock watchlist from {path}: {e}")
        return None


def is_entry_window_active(now: datetime) -> bool:
    """Checked live only between 9:32:00am ET and 10:30:00am ET."""
    t = now.astimezone(ET).time()
    return ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END


def check_single_stock_trigger(
    candidate: SingleStockCandidate,
    today_open: float,
    today_high: float,
    today_low: float,
    now: datetime,
) -> bool:
    """Evaluates whether the candidate meets the live entry trigger.
    Must be within 9:32:00am - 10:30:00am ET and satisfy the 70% High-Low rule.
    """
    if not is_entry_window_active(now):
        return False
    return passes_prior_excursion_direction_filter(
        candidate.prior_open, candidate.prior_high, candidate.prior_low,
        today_open, today_high, today_low,
        threshold=TODAY_EXCURSION_DIRECTION_MIN_SHARE,
    )


def compute_single_stock_quantity(
    sizing_elv: float,
    stock_price: float,
    initial_margin_fraction: float = INITIAL_MARGIN_FRACTION,
) -> int:
    """Position sizing for single stock:
    floor((sizing_elv * 0.98) / (stock_price * initial_margin_fraction))
    """
    if stock_price <= 0:
        return 0
    margin_per_share = stock_price * initial_margin_fraction
    return math.floor((sizing_elv * 0.98) / margin_per_share)


def compute_pnl_stop_price(
    action: str,
    entry_price: float,
    quantity: int,
    sizing_elv: float,
    max_loss_pct: float = PNL_STOP_LOSS_PCT,
) -> float:
    """Calculates the -2% PnL Stop Loss price level:
    per_share = (sizing_elv * max_loss_pct) / quantity
    stop_price = entry_price - per_share (BUY) or entry_price + per_share (SELL)
    Clamped to >= $0.01.
    """
    if quantity <= 0:
        raise ValueError("Cannot calculate PnL stop price for zero or negative quantity")
    per_share = (sizing_elv * max_loss_pct) / quantity
    stop = (entry_price - per_share) if action == "BUY" else (entry_price + per_share)
    return max(round(stop, 2), 0.01)


class SingleStockContext:
    """Context object representing a single stock candidate during the trading day.
    Duck-types PairContext so write_status, dashboard, and daily reconciliation
    work seamlessly without modification.
    """

    def __init__(self, candidate: SingleStockCandidate):
        self.candidate = candidate
        self.symbol_a = candidate.symbol
        self.symbol_b = ""  # Single stock mode has no second leg
        self.hedge_ratio = 1.0
        self.spread_history = pd.Series(dtype=float)
        self.price_history_a = pd.Series(dtype=float)
        self.price_history_b = pd.Series(dtype=float)
        self.prior_session_price_a = pd.Series(dtype=float)
        self.prior_session_price_b = pd.Series(dtype=float)
        self.position: Position | None = None
        self.base_trigger_level = 0.0
        self.divergence_trigger_level = 0.0
        self.half_life_minutes = 0.0
        self.trigger_reached_today = False
        self.using_exception_trigger = False
        self.mean_reverted_since_max_divergence = False
        self.reverted_over_50pct_since_max_divergence = False
        self.correlation_60d = None
        self.correlation_20d = None
        self.live_correlation = None
        self.live_adf_pvalue = None
        self.beta_estimator = None
        self.day_sizing_qty_a = 0.0
        self.day_sizing_qty_b = 0.0
        self.industry = "S&P 500"
        self.today_open: float | None = None
        self.today_high: float | None = None
        self.today_low: float | None = None
        self.today_close: float | None = None

    def record_today_bar(self, close_price: float, timestamp: datetime) -> None:
        if self.today_open is None:
            self.today_open = close_price
            self.today_high = close_price
            self.today_low = close_price
        else:
            self.today_high = max(self.today_high, close_price)
            self.today_low = min(self.today_low, close_price)
        self.today_close = close_price
        stamp = pd.Timestamp(timestamp)
        self.price_history_a.loc[stamp] = close_price
        self.spread_history.loc[stamp] = close_price

    def is_triggered(self, now: datetime) -> bool:
        if self.today_open is None or self.today_high is None or self.today_low is None:
            return False
        return check_single_stock_trigger(
            self.candidate, self.today_open, self.today_high, self.today_low, now
        )


def report_event_from_single_stock_position(
    ctx: SingleStockContext, now: datetime, exit_reason: str
) -> dict:
    """Builds a post-trade report event dictionary for a closed single-stock position."""
    pos = ctx.position
    if pos is None:
        return {}
    return {
        "symbol_a": ctx.symbol_a,
        "symbol_b": "",
        "action_a": pos.action_a,
        "traded_leg": "A",
        "qty_a": pos.qty_a,
        "qty_b": 0.0,
        "entry_price_a": pos.entry_price_a,
        "entry_price_b": 0.0,
        "entry_time": pos.entry_time.isoformat(),
        "exit_time": now.isoformat(),
        "exit_reason": exit_reason,
        "entry_notional": pos.entry_notional,
        "worst_unrealized_pnl": pos.worst_unrealized_pnl,
        "mode": "single_stock_excursion",
    }


async def run_single_stock_check_cycle(
    contexts: list[SingleStockContext],
    feed,
    ibkr,
    daily_state: DailyState,
    equity_with_loan_value: float,
    elv_at_925: float,
    now: datetime | None = None,
) -> None:
    """Executes one 60-second cycle for single-stock excursion trading."""
    now = now or datetime.now(ET)

    # 1. Update live prices for all watchlist candidates
    if feed is not None and hasattr(feed, "book"):
        for ctx in contexts:
            price = feed.book.latest_close(ctx.symbol_a)
            if price is not None:
                ctx.record_today_bar(price, now)

    # 2. Monitor and reconcile exit for open position
    for ctx in contexts:
        if ctx.position is None:
            continue
        pos = ctx.position

        # Abandoned order fail-safe
        if ibkr.entry_order_abandoned(pos.entry_order_id):
            logger.warning(f"{ctx.symbol_a}: entry order {pos.entry_order_id} abandoned -- clearing position")
            ctx.position = None
            continue

        price = ctx.today_close or pos.entry_price_a
        sign = 1 if pos.action_a == "BUY" else -1
        live_pnl = pos.qty_a * sign * (price - pos.entry_price_a)
        pos.worst_unrealized_pnl = min(pos.worst_unrealized_pnl, live_pnl)

        # Check bracket fills (Trailing TP, PnL Stop, Timed Exit)
        trail_filled = ibkr.bracket_order_filled(pos.trail_order_id)
        pnl_stop_filled = ibkr.bracket_order_filled(pos.pnl_stop_order_id)
        timed_exit_filled = ibkr.bracket_order_filled(pos.timed_exit_order_id)

        if trail_filled is not None or pnl_stop_filled is not None or timed_exit_filled is not None:
            if trail_filled is not None:
                reason = "take_profit"
            elif pnl_stop_filled is not None:
                reason = "hard_stop"
            else:
                reason = "timed_exit"

            # Cancel remaining sibling bracket orders in the OCA group
            for oid in (pos.trail_order_id, pos.pnl_stop_order_id, pos.timed_exit_order_id):
                if oid is not None:
                    ibkr.cancel_order_id(oid)

            daily_state.realized_pnl_today += live_pnl
            record_exit(daily_state, daily_state.realized_pnl_today, elv_at_925)
            daily_state.trade_report_events.append(
                report_event_from_single_stock_position(ctx, now, exit_reason=reason)
            )

            # Check if re-entry is unlocked (Take Profit before 10:30am ET)
            if not reentry_allowed(reason, now):
                daily_state.reentry_locked_out = True
                logger.info(f"{ctx.symbol_a}: position closed ({reason}). Day done -- no further entries today.")
            else:
                logger.info(f"{ctx.symbol_a}: Take profit before 10:30am ET! Re-entry unlocked (trade {daily_state.trades_taken_today}/{MAX_TRADES_PER_DAY}).")

            ctx.position = None

    # 3. Evaluate live entry trigger if no position is open
    book_is_flat = all(c.position is None for c in contexts)
    if (book_is_flat
            and not daily_state.reentry_locked_out
            and not daily_state.trading_halted
            and daily_state.trades_taken_today < MAX_TRADES_PER_DAY
            and is_entry_window_active(now)):

        for ctx in contexts:
            if not ctx.is_triggered(now):
                continue

            symbol = ctx.symbol_a
            action = ctx.candidate.dominant_direction
            traded_price = ctx.today_close
            if not traded_price or traded_price <= 0:
                continue

            try:
                live_elv = await ibkr.get_equity_with_loan_value()
                sizing_elv = effective_sizing_equity(equity_with_loan_value, live_elv)
            except Exception as e:
                logger.error(f"{symbol}: equity read failed ({e}) -- skipping entry")
                return

            quantity = compute_single_stock_quantity(sizing_elv, traded_price, INITIAL_MARGIN_FRACTION)
            if quantity <= 0:
                continue

            # Real-order margin probe with IBKR
            timed_exit_activation = now + timedelta(minutes=MAX_HOLD_MINUTES)
            for _probe_iter in range(6):
                required_margin = await ibkr.probe_excursion_bracket_initial_margin(
                    symbol, action, quantity, traded_price, timed_exit_activation
                )
                if required_margin is None:
                    break
                fitted = int(quantity * (sizing_elv * 0.98) / required_margin)
                if fitted >= quantity or fitted <= 0:
                    logger.warning(f"{symbol}: margin probe cannot fit {quantity} shares -- skipping")
                    return
                quantity = fitted
            else:
                logger.warning(f"{symbol}: margin probe still rejecting after 6 resizes")
                return

            # Submit Entry Order
            order_id, filled_qty, avg_price = await ibkr.place_single_stock_trade(symbol, action, quantity)
            if filled_qty <= 0:
                logger.warning(f"{symbol}: entry order {order_id} did not fill -- skipping")
                return

            closing_action = "SELL" if action == "BUY" else "BUY"
            pnl_stop_price = compute_pnl_stop_price(action, avg_price, filled_qty, sizing_elv, PNL_STOP_LOSS_PCT)

            # Submit 3-Leg OCA Bracket
            trail_id, pnl_id, te_id = await ibkr.place_excursion_bracket(
                symbol, closing_action, filled_qty, pnl_stop_price,
                TRAILING_TAKE_PROFIT_AMOUNT, timed_exit_activation
            )

            _brk = await asyncio.gather(
                ibkr.order_was_rejected(trail_id),
                ibkr.order_was_rejected(pnl_id),
                ibkr.order_was_rejected(te_id),
            )
            bracket_rejection = next((r for r in _brk if r), None)
            if bracket_rejection:
                logger.critical(f"{symbol}: excursion bracket rejected ({bracket_rejection}) -- emergency flatten")
                await ibkr.flatten_symbols({symbol})
                daily_state.reentry_locked_out = True
                return

            entry_notional = filled_qty * avg_price
            pos = Position(
                symbol_a=symbol,
                symbol_b="",
                action_a=action,
                qty_a=filled_qty,
                qty_b=0.0,
                entry_time=now,
                entry_spread=0.0,
                half_life_minutes=0.0,
                entry_notional=entry_notional,
                entry_price_a=avg_price,
                entry_price_b=0.0,
                entry_margin=avg_price * INITIAL_MARGIN_FRACTION * filled_qty,
                traded_leg="A",
                entry_order_id=order_id,
                trail_order_id=trail_id,
                hard_stop_order_id=None,
                pnl_stop_order_id=pnl_id,
                timed_exit_order_id=te_id,
            )
            ctx.position = pos
            daily_state.trades_taken_today += 1
            logger.info(f"Entered single-stock excursion trade: {filled_qty}x {symbol} {action} @ ${avg_price:.2f}, "
                        f"TrailingTP=${TRAILING_TAKE_PROFIT_AMOUNT:.2f}, PnLStop=${pnl_stop_price:.2f}, "
                        f"TimedExit after {MAX_HOLD_MINUTES}m (trade {daily_state.trades_taken_today}/{MAX_TRADES_PER_DAY})")
            break  # First qualifying stock enters
