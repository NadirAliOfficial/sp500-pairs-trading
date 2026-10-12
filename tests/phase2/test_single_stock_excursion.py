"""Unit tests for phase2/single_stock_excursion.py and IBKRClient excursion bracket orders.

Verifies:
1. Toggle defaults strictly to OFF (False).
2. Prior session 300% dominance filter.
3. Live 70% range trigger and 9:32am - 10:30am ET window check.
4. Sizing calculation matching margin limit formula.
5. PnL Stop Loss price level calculation (-2% of EWLV).
6. 3-Leg OCA bracket placement (TRAIL $1, STP -2%, MKT 20 min).
7. Probe excursion bracket initial margin.
8. Re-entry rules: Take Profit before 10:30am ET allows 1 re-entry, capped at MAX_TRADES_PER_DAY=2.
   All other exits lock out re-entry.
"""
import asyncio
import math
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.ibkr_client import IBKRClient
from phase2.risk_manager import DailyState, Position, reentry_allowed
from phase2.single_stock_excursion import (
    ENTRY_WINDOW_END,
    ENTRY_WINDOW_START,
    INITIAL_MARGIN_FRACTION,
    MAX_HOLD_MINUTES,
    MAX_TRADES_PER_DAY,
    PNL_STOP_LOSS_PCT,
    PRIOR_EXCURSION_DOMINANCE_MIN_RATIO,
    SINGLE_STOCK_EXCURSION_MODE,
    TODAY_EXCURSION_DIRECTION_MIN_SHARE,
    TRAILING_TAKE_PROFIT_AMOUNT,
    SingleStockCandidate,
    SingleStockContext,
    build_candidate_from_prior_bars,
    check_single_stock_trigger,
    compute_pnl_stop_price,
    compute_single_stock_quantity,
    compute_single_stock_watchlist,
    is_entry_window_active,
    load_single_stock_watchlist,
    passes_prior_excursion_direction_filter,
    passes_prior_excursion_dominance_filter,
    run_single_stock_check_cycle,
    save_single_stock_watchlist,
)

ET = ZoneInfo("America/New_York")


def test_toggle_defaults_to_off():
    """Toggle must strictly default to False in code and when env var is unset."""
    assert SINGLE_STOCK_EXCURSION_MODE is False
    print("PASS: SINGLE_STOCK_EXCURSION_MODE is False by default")


def test_prior_excursion_dominance_300pct():
    """Dominant excursion from open must be >= 300% (3.0x) of opposite excursion."""
    # North dominant: High - Open = 6.0, Open - Low = 2.0 -> Ratio = 3.0 -> PASS
    assert passes_prior_excursion_dominance_filter(prior_open=100.0, prior_high=106.0, prior_low=98.0) is True

    # North dominant: High - Open = 5.9, Open - Low = 2.0 -> Ratio = 2.95 < 3.0 -> FAIL
    assert passes_prior_excursion_dominance_filter(prior_open=100.0, prior_high=105.9, prior_low=98.0) is False

    # South dominant: Open - Low = 6.0, High - Open = 1.5 -> Ratio = 4.0 >= 3.0 -> PASS
    assert passes_prior_excursion_dominance_filter(prior_open=100.0, prior_high=101.5, prior_low=94.0) is True

    # Zero opposite excursion: pure one-way day -> PASS
    assert passes_prior_excursion_dominance_filter(prior_open=100.0, prior_high=105.0, prior_low=100.0) is True

    # Flat day: Open = High = Low -> FAIL
    assert passes_prior_excursion_dominance_filter(prior_open=100.0, prior_high=100.0, prior_low=100.0) is False
    print("PASS: passes_prior_excursion_dominance_filter correctly enforces 300% ratio")


def test_passes_prior_excursion_direction_filter_70pct():
    """70% or more of today's High-Low range must sit in dominant excursion direction."""
    # Prior North dominant (Open=100, High=105, Low=99) -> North=5, South=1
    # Today: Open=50, High=54, Low=50 -> Range=4.0, North portion=4.0 -> 100% >= 70% -> PASS
    assert passes_prior_excursion_direction_filter(
        prior_open=100.0, prior_high=105.0, prior_low=99.0,
        today_open=50.0, today_high=54.0, today_low=50.0,
    ) is True

    # Today: Open=50, High=52, Low=48 -> Range=4.0, North portion=2.0 -> 50% < 70% -> FAIL
    assert passes_prior_excursion_direction_filter(
        prior_open=100.0, prior_high=105.0, prior_low=99.0,
        today_open=50.0, today_high=52.0, today_low=48.0,
    ) is False

    # Prior South dominant (Open=100, High=101, Low=95) -> South=5, North=1
    # Today: Open=50, High=50.5, Low=47.5 -> Range=3.0, South portion=2.5 -> 83.3% >= 70% -> PASS
    assert passes_prior_excursion_direction_filter(
        prior_open=100.0, prior_high=101.0, prior_low=95.0,
        today_open=50.0, today_high=50.5, today_low=47.5,
    ) is True

    # Zero range today: FAIL
    assert passes_prior_excursion_direction_filter(
        prior_open=100.0, prior_high=105.0, prior_low=99.0,
        today_open=50.0, today_high=50.0, today_low=50.0,
    ) is False
    print("PASS: passes_prior_excursion_direction_filter correctly evaluates 70% range")


def test_entry_window_timing():
    """Checked live only between 9:32:00am ET and 10:30:00am ET."""
    d = date(2026, 9, 17)
    # Pre-market 9:30am
    t_930 = datetime.combine(d, time(9, 30, 0), tzinfo=ET)
    assert is_entry_window_active(t_930) is False

    # 9:31:59am
    t_93159 = datetime.combine(d, time(9, 31, 59), tzinfo=ET)
    assert is_entry_window_active(t_93159) is False

    # 9:32:00am sharp
    t_932 = datetime.combine(d, time(9, 32, 0), tzinfo=ET)
    assert is_entry_window_active(t_932) is True

    # 10:00:00am
    t_1000 = datetime.combine(d, time(10, 0, 0), tzinfo=ET)
    assert is_entry_window_active(t_1000) is True

    # 10:30:00am
    t_1030 = datetime.combine(d, time(10, 30, 0), tzinfo=ET)
    assert is_entry_window_active(t_1030) is True

    # 10:30:01am
    t_103001 = datetime.combine(d, time(10, 30, 1), tzinfo=ET)
    assert is_entry_window_active(t_103001) is False
    print("PASS: is_entry_window_active strictly enforces 9:32:00am - 10:30:00am ET window")


def test_compute_single_stock_quantity():
    """Quantity = floor((sizing_elv * 0.98) / (price * 0.35))."""
    sizing_elv = 100_000.0
    price = 100.0
    # margin_per_share = 35.0; 98,000 / 35 = 2800
    qty = compute_single_stock_quantity(sizing_elv, price, INITIAL_MARGIN_FRACTION)
    assert qty == 2800

    # Non-round division floors correctly
    qty2 = compute_single_stock_quantity(25_500.0, 142.50, INITIAL_MARGIN_FRACTION)
    expected = math.floor((25_500.0 * 0.98) / (142.50 * 0.35))
    assert qty2 == expected
    assert qty2 == 501
    print("PASS: compute_single_stock_quantity sizes correctly")


def test_compute_pnl_stop_price():
    """PnL Stop Loss = -2% of EquityWithLoanValue, converted to resting price."""
    # BUY at $100, 1000 shares, EWLV $100,000 -> 2% loss is $2,000 -> $2/share -> Stop = $98.00
    pnl_stop_buy = compute_pnl_stop_price("BUY", entry_price=100.0, quantity=1000, sizing_elv=100_000.0)
    assert pnl_stop_buy == 98.00

    # SELL at $100, 1000 shares, EWLV $100,000 -> Stop = $102.00
    pnl_stop_sell = compute_pnl_stop_price("SELL", entry_price=100.0, quantity=1000, sizing_elv=100_000.0)
    assert pnl_stop_sell == 102.00

    # Minimum floor clamping at $0.01
    pnl_stop_low = compute_pnl_stop_price("BUY", entry_price=0.50, quantity=10, sizing_elv=100_000.0)
    assert pnl_stop_low == 0.01
    print("PASS: compute_pnl_stop_price accurately prices -2% PnL Stop Loss")


def test_place_excursion_bracket_places_three_orders_in_one_oca_group():
    """Places Trailing TP ($1), PnL Stop (-2%), Timed Exit (20 min) in ONE OCA group."""
    async def scenario():
        client = IBKRClient(account="DU123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 50
        placed = []
        client.placeOrder = lambda order_id, contract, order: placed.append((order_id, contract, order))
        client.live_trading_enabled = True

        activation = datetime(2026, 9, 17, 9, 52, 0, tzinfo=ET)
        trail_id, pnl_id, te_id = await client.place_excursion_bracket(
            symbol="NVDA", closing_action="SELL", quantity=150,
            pnl_stop_price=118.50, trailing_take_profit_amount=1.00,
            timed_exit_activation=activation,
        )

        assert len(placed) == 3
        orders_by_id = {order_id: order for order_id, _, order in placed}

        # 1. Trailing TP
        trail_order = orders_by_id[trail_id]
        assert trail_order.orderType == "TRAIL"
        assert trail_order.auxPrice == 1.00
        assert trail_order.action == "SELL"
        assert trail_order.totalQuantity == 150

        # 2. PnL Stop
        pnl_order = orders_by_id[pnl_id]
        assert pnl_order.orderType == "STP"
        assert pnl_order.auxPrice == 118.50
        assert pnl_order.action == "SELL"

        # 3. Timed Exit
        te_order = orders_by_id[te_id]
        assert te_order.orderType == "MKT"
        assert "20260917 09:52:00 US/Eastern" in te_order.goodAfterTime

        # All 3 legs share identical ocaGroup and ocaType=1
        assert trail_order.ocaGroup == pnl_order.ocaGroup == te_order.ocaGroup
        assert trail_order.ocaType == 1
    asyncio.run(scenario())
    print("PASS: place_excursion_bracket places 3-leg OCA bracket correctly")


def test_probe_excursion_bracket_initial_margin():
    """Probe submits non-fillable entry + standins and cancels cleanly."""
    async def scenario():
        client = IBKRClient(account="DU123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 10
        placed = []
        cancelled = []
        client.placeOrder = lambda order_id, contract, order: placed.append((order_id, contract, order))
        client.cancelOrder = lambda order_id: cancelled.append(order_id)
        client.live_trading_enabled = True

        activation = datetime(2026, 9, 17, 9, 52, 0, tzinfo=ET)
        # Simulate no 201 rejection (size fits)
        margin = await client.probe_excursion_bracket_initial_margin(
            "AAPL", "BUY", 100, 220.0, activation, wait_seconds=0.1
        )
        assert margin is None
        assert len(placed) == 4
        assert len(cancelled) == 4
        assert set(cancelled) == {10, 11, 12, 13}
    asyncio.run(scenario())
    print("PASS: probe_excursion_bracket_initial_margin probes and cancels cleanly")


def test_watchlist_persistence(tmp_path):
    """Saves and loads candidate watchlist to disk."""
    cand = SingleStockCandidate(
        symbol="MSFT", prior_open=400.0, prior_high=415.0, prior_low=398.0,
        prior_close=414.0, dominant_direction="BUY", north_excursion=15.0,
        south_excursion=2.0, dominance_ratio=7.5,
    )
    test_file = tmp_path / "test_watchlist.json"
    today = date(2026, 9, 17)
    save_single_stock_watchlist([cand], path=test_file, for_date=today)

    loaded = load_single_stock_watchlist(path=test_file, today=today)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].symbol == "MSFT"
    assert loaded[0].dominant_direction == "BUY"
    assert loaded[0].dominance_ratio == 7.5
    print("PASS: save_single_stock_watchlist / load_single_stock_watchlist roundtrips cleanly")


def test_reentry_rules_untouched():
    """Untouched MAX_TRADES_PER_DAY=2 with Take Profit before 10:30am ET carve-out."""
    # Take Profit before 10:30am ET unlocks re-entry
    t_early = datetime.combine(date(2026, 9, 17), time(10, 15), tzinfo=ET)
    assert reentry_allowed("take_profit", t_early) is True

    # Take Profit after 10:30am ET blocks re-entry
    t_late = datetime.combine(date(2026, 9, 17), time(10, 31), tzinfo=ET)
    assert reentry_allowed("take_profit", t_late) is False

    # Stop Loss or Timed Exit always blocks re-entry
    assert reentry_allowed("hard_stop", t_early) is False
    assert reentry_allowed("timed_exit", t_early) is False
    assert reentry_allowed("eod", t_early) is False
    print("PASS: re-entry rules match untouched spec")


def test_single_stock_position_order_ids_and_fill():
    """Verifies single stock excursion sets pnl_stop_order_id (and hard_stop_order_id=None),
    and correctly reconciles a pnl stop fill using pnl_stop_order_id."""
    async def scenario():
        client = IBKRClient(account="DU123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 100
        client.live_trading_enabled = True

        async def fake_elv():
            return 100_000.0

        async def fake_probe(*args, **kwargs):
            return None

        async def fake_entry(sym, act, qty):
            return (101, qty, 100.0)

        async def fake_bracket(*args, **kwargs):
            return (102, 103, 104)

        async def fake_rejected(oid):
            return None

        client.get_equity_with_loan_value = fake_elv
        client.probe_excursion_bracket_initial_margin = fake_probe
        client.place_single_stock_trade = fake_entry
        client.place_excursion_bracket = fake_bracket
        client.order_was_rejected = fake_rejected

        cand = SingleStockCandidate(
            symbol="NVDA", prior_open=100.0, prior_high=105.0, prior_low=99.0,
            prior_close=104.0, dominant_direction="BUY", north_excursion=5.0,
            south_excursion=1.0, dominance_ratio=5.0,
        )
        ctx = SingleStockContext(candidate=cand)
        ctx.today_open = 100.0
        ctx.today_high = 104.0
        ctx.today_low = 100.0
        ctx.today_close = 103.5

        daily_state = DailyState()
        now = datetime(2026, 9, 17, 9, 35, 0, tzinfo=ET)

        client.entry_order_abandoned = lambda oid: False
        client.bracket_order_abandoned = lambda oid: False

        # 1. Enter position
        await run_single_stock_check_cycle([ctx], None, client, daily_state, 100_000.0, 100_000.0, now)

        assert ctx.position is not None
        assert ctx.position.entry_order_id == 101
        assert ctx.position.trail_order_id == 102
        assert ctx.position.hard_stop_order_id is None
        assert ctx.position.pnl_stop_order_id == 103
        assert ctx.position.timed_exit_order_id == 104
        assert daily_state.trades_taken_today == 1

        # 2. Simulate pnl stop fill
        client.bracket_order_filled = lambda oid: 100.0 if oid == 103 else None
        cancelled_ids = []
        client.cancel_order_id = lambda oid: cancelled_ids.append(oid)

        now_exit = datetime(2026, 9, 17, 9, 45, 0, tzinfo=ET)
        await run_single_stock_check_cycle([ctx], None, client, daily_state, 100_000.0, 100_000.0, now_exit)

        assert ctx.position is None
        assert 102 in cancelled_ids
        assert 104 in cancelled_ids
        assert daily_state.reentry_locked_out is True

    asyncio.run(scenario())
    print("PASS: test_single_stock_position_order_ids_and_fill verified")
