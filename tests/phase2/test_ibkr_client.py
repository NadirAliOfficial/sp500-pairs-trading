"""Unit tests for ibkr_client.py order-construction helpers and the dry-run
safety gate -- no real IBKR connection is made (IBKRClient.connect_and_start
is never called), matching how the SPY bot's margin/order logic was verified
against hand-built expectations before ever touching a live gateway."""
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ibapi.common import UNSET_DOUBLE
from ibapi.contract import Contract
from ibapi.execution import Execution

from phase2.ibkr_client import (IBKRClient, _INITIAL_MARGIN_RE, _resolve_future, combo_bracket_child_order,
                                 limit_order, market_order, parse_todays_liquid_hours,
                                 stock_bracket_child_order, stock_contract)


def _leg_contract(symbol):
    c = Contract()
    c.symbol = symbol
    return c


def _execution(order_id, cum_qty, avg_price, side="BOT"):
    e = Execution()
    e.orderId = order_id
    e.cumQty = cum_qty
    e.avgPrice = avg_price
    e.side = side
    return e


def test_market_order_always_sets_safety_flags():
    o = market_order("SELL", 42, account=None)
    assert o.eTradeOnly is False
    assert o.firmQuoteOnly is False
    assert o.orderType == "MKT"
    print("PASS: market_order always sets eTradeOnly=False, firmQuoteOnly=False")


def test_stock_contract_is_smart_usd():
    c = stock_contract("AAPL")
    assert c.secType == "STK"
    assert c.exchange == "SMART"
    assert c.currency == "USD"
    print("PASS: stock_contract builds a SMART/USD stock contract")


def test_place_single_stock_trade_is_dry_run_by_default():
    async def scenario():
        client = IBKRClient(account="U123")
        client.live_trading_enabled = False
        client._next_order_id = 1
        order_id, qty, price = await client.place_single_stock_trade("AAPL", "BUY", 100)
        assert order_id == 1 and qty == 0.0 and price == 0.0
        assert client._next_order_id == 2
    asyncio.run(scenario())
    print("PASS: place_single_stock_trade dry-runs unless IBKR_LIVE_TRADING_ENABLED=1")


def test_place_single_stock_trade_places_one_mkt_order():
    async def scenario():
        import os
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 5

        placed = []

        def fake_place(order_id, contract, order):
            placed.append((order_id, contract, order))
            client._order_status[order_id] = "Filled"
            client.execDetails(-1, _leg_contract("AAPL"), _execution(order_id, cum_qty=100, avg_price=180.0))

        client.placeOrder = fake_place
        os.environ["IBKR_LIVE_TRADING_ENABLED"] = "1"
        client.live_trading_enabled = True
        try:
            order_id, qty, price = await client.place_single_stock_trade("AAPL", "BUY", 100, wait_seconds=0.3)
        finally:
            del os.environ["IBKR_LIVE_TRADING_ENABLED"]

        assert order_id == 5 and qty == 100 and price == 180.0
        assert len(placed) == 1
        contract, order = placed[0][1], placed[0][2]
        assert contract.symbol == "AAPL" and contract.secType == "STK"
        assert order.action == "BUY" and order.totalQuantity == 100 and order.orderType == "MKT"
    asyncio.run(scenario())
    print("PASS: place_single_stock_trade places one MKT order and reconciles the real fill")


def test_place_single_stock_trade_cancels_after_timeout_keeps_partial_fill():
    """Per the client 2026-09-03: "Partial Fill Allowed. Leg Risk
    Safeguard: A 2-second timeout monitors fill status and cancels
    unfilled portions." A partial fill is kept, not discarded -- only
    the still-working remainder gets cancelled."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 7
        cancelled = []
        client.cancelOrder = lambda order_id: cancelled.append(order_id)

        def fake_place(order_id, contract, order):
            # Only a partial fill arrives -- status never reaches Filled.
            client.execDetails(-1, _leg_contract("AAPL"), _execution(order_id, cum_qty=40, avg_price=180.0))

        client.placeOrder = fake_place
        client.live_trading_enabled = True

        order_id, qty, price = await client.place_single_stock_trade("AAPL", "BUY", 100, wait_seconds=0.1)

        assert qty == 40 and price == 180.0  # the partial fill is kept
        assert cancelled == [order_id]  # the unfilled remainder was cancelled
    asyncio.run(scenario())
    print("PASS: a partial fill after the timeout is kept, the unfilled remainder is cancelled")


def test_place_single_stock_timed_exit_is_dry_run_by_default():
    async def scenario():
        client = IBKRClient(account="U123")
        client.live_trading_enabled = False
        client._next_order_id = 9
        order_id = await client.place_single_stock_timed_exit(
            "AAPL", "SELL", 100, datetime(2026, 9, 3, 11, 0))
        assert order_id == 9
        assert client._next_order_id == 10
    asyncio.run(scenario())
    print("PASS: place_single_stock_timed_exit dry-runs unless IBKR_LIVE_TRADING_ENABLED=1")


def test_place_single_stock_timed_exit_places_one_mkt_order_with_good_after_time():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 11
        placed = []
        client.placeOrder = lambda order_id, contract, order: placed.append((order_id, contract, order))
        client.live_trading_enabled = True

        order_id = await client.place_single_stock_timed_exit(
            "AAPL", "SELL", 100, datetime(2026, 9, 3, 11, 0))

        assert order_id == 11
        assert len(placed) == 1
        contract, order = placed[0][1], placed[0][2]
        assert contract.symbol == "AAPL" and contract.secType == "STK"
        assert order.action == "SELL" and order.totalQuantity == 100 and order.orderType == "MKT"
        assert order.goodAfterTime == "20260903 11:00:00 US/Eastern"
    asyncio.run(scenario())
    print("PASS: place_single_stock_timed_exit places one MKT order held until goodAfterTime")


def test_stock_bracket_child_order_uses_the_real_closing_action():
    o = stock_bracket_child_order("SELL", "STP", 95.50, 100, account=None, oca_group="single_5")
    assert o.action == "SELL"  # not hardcoded "BUY" like the combo version
    assert o.orderType == "STP" and o.auxPrice == 95.50
    assert o.ocaGroup == "single_5" and o.ocaType == 1
    assert not hasattr(o, "smartComboRoutingParams") or not o.smartComboRoutingParams
    print("PASS: stock_bracket_child_order carries a real action, no combo routing params")


def test_initial_margin_regex_parses_the_201_rejection_text():
    """The real 201 text from the 2026-09-10 incident, verbatim (IBKR wraps
    it in <br> tags)."""
    txt = ("201: Order rejected - reason:We are unable to accept your order. Your Available "
           "Funds are in sufficient to cover the change in the <br>account's margin requirements "
           "if this order executes. In order to obtain the desired position your Equity with<br> "
           "Loan Value [13740.51 USD] must exceed the new total Initial Margin of  [23006.62 USD].")
    m = _INITIAL_MARGIN_RE.search(txt)
    assert m is not None
    assert float(m.group(1).replace(",", "")) == 23006.62
    print("PASS: the 201 rejection text yields the real required initial margin")


def test_initial_margin_regex_handles_thousands_separators():
    m = _INITIAL_MARGIN_RE.search("must exceed the new total Initial Margin of [1,234,567.89 USD]")
    assert float(m.group(1).replace(",", "")) == 1_234_567.89
    print("PASS: the initial-margin regex handles comma thousands separators")


def test_limit_order_is_non_marketable_and_safe_flagged():
    o = limit_order("BUY", 100, 1.00, account="U1")
    assert o.orderType == "LMT" and o.lmtPrice == 1.00
    assert o.eTradeOnly is False and o.firmQuoteOnly is False
    print("PASS: limit_order builds a $1 non-fillable probe order with the safety flags")


def test_probe_bracket_initial_margin_reads_the_201_and_cancels_everything():
    """Per the client 2026-09-10: submit entry (non-fillable LMT) + 4
    bracket legs as real orders, read the required margin off the 201,
    cancel every probe order.

    Briefly simplified to entry-only 2026-09-15, then REVERTED the same
    day: a real live incident showed the bracket legs DO change the true
    required margin (INTU's entry-only probe found 29 shares fit, but the
    real bracket right after was rejected needing nearly double) -- see
    probe_bracket_initial_margin's own comment.

    A SIXTH leg was added the same day too -- a stand-in for the new
    Trailing Take Profit leg (see place_single_stock_bracket's own
    comment)."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 100
        client.live_trading_enabled = True
        placed, cancelled = [], []
        client.placeOrder = lambda oid, c, o: placed.append((oid, o))
        client.cancelOrder = lambda oid: cancelled.append(oid)

        async def drive():
            await asyncio.sleep(0.05)
            # IBKR rejects the trail-standin leg (6th id) on 201
            client.error(105, 201, "must exceed the new total Initial Margin of [23006.62 USD]")
            client._rejected_orders[105] = "201: must exceed the new total Initial Margin of [23006.62 USD]"
            # ...and acknowledges the cancels moments later
            await asyncio.sleep(0.05)
            for oid in (100, 101, 102, 103, 104, 105):
                client._order_status[oid] = "Cancelled"

        _, margin = await asyncio.gather(
            drive(),
            client.probe_bracket_initial_margin(
                "ADBE", "SELL", 139, 251.0, datetime(2026, 9, 10, 11, 0), wait_seconds=1.0),
        )
        assert margin == 23006.62
        assert len(placed) == 6  # entry LMT + 4 bracket legs + trail standin
        assert placed[0][1].orderType == "LMT"  # entry is the non-fillable probe
        assert sorted(cancelled) == [100, 101, 102, 103, 104, 105]  # all probe orders cancelled
        # Bracket legs go in at un-triggerable prices: for a SELL entry the
        # closing side is BUY, so the STP legs sit far ABOVE and the MIT far
        # BELOW -- neither can fill in the probe window.
        by_id = {oid: o for oid, o in placed}
        assert by_id[101].auxPrice == round(251.0 * 5.0, 2)  # hard-stop STP, far above, rounded
        assert by_id[104].auxPrice == round(251.0 * 5.0, 2)  # pnl STP, far above, rounded
        assert by_id[102].auxPrice == round(251.0 * 0.01, 2)  # take-profit MIT, far below
        assert by_id[105].orderType == "STP"  # trail standin is a plain STP, not a real TRAIL
        assert by_id[105].auxPrice == round(251.0 * 5.0, 2)  # same far price as the hard stop
    asyncio.run(scenario())
    print("PASS: probe reads the 201 margin, uses un-triggerable leg prices, and waits out the cancels")


def test_probe_leg_prices_conform_to_the_cent_for_a_fractional_price():
    """far_hi must be rounded, or IBKR rejects the probe leg on error 110
    (not 201), the probe misses it, and falsely reports the size fits."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 300
        client.live_trading_enabled = True
        placed = []
        client.placeOrder = lambda oid, c, o: placed.append((oid, o))
        client.cancelOrder = lambda oid: [client._order_status.__setitem__(oid, "Cancelled")]
        await client.probe_bracket_initial_margin(
            "XYZ", "SELL", 10, 42.155, datetime(2026, 9, 10, 11, 0), wait_seconds=0.2)
        for oid, o in placed[1:]:  # skip the entry LMT
            if o.orderType in ("STP", "MIT"):
                assert round(o.auxPrice, 2) == o.auxPrice, (oid, o.auxPrice)
    asyncio.run(scenario())
    print("PASS: probe leg prices are cent-conforming even from a fractional current price")


def test_probe_bracket_initial_margin_returns_none_when_nothing_rejects():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 200
        client.live_trading_enabled = True
        client.placeOrder = lambda oid, c, o: None
        client.cancelOrder = lambda oid: None
        margin = await client.probe_bracket_initial_margin(
            "ADBE", "SELL", 50, 251.0, datetime(2026, 9, 10, 11, 0), wait_seconds=0.3)
        assert margin is None  # fits -> submit for real as-is
    asyncio.run(scenario())
    print("PASS: probe returns None (size already fits) when no 201 comes back")


def test_modify_hard_stop_reuses_the_same_order_id_and_rounds():
    """Per the client 2026-09-09: modifying the Hard Stop Loss must reuse
    the same order id (IBKR treats this as a modification, never leaving
    the position without a resting stop) and round to the cent, same
    reasoning as place_single_stock_bracket's own rounding."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        placed = []
        client.placeOrder = lambda oid, contract, order: placed.append((oid, order))
        client.live_trading_enabled = True

        ok = await client.modify_hard_stop("ADSK", "SELL", 100, 92.503333, order_id=42, oca_group="single_43")

        assert ok is True
        assert len(placed) == 1
        oid, order = placed[0]
        assert oid == 42  # same id -- a modification, not a new order
        assert order.orderType == "STP" and round(order.auxPrice, 2) == order.auxPrice
        assert order.auxPrice == 92.50
        assert order.ocaGroup == "single_43"
    asyncio.run(scenario())
    print("PASS: modify_hard_stop reuses the same order id and rounds the price to the cent")


def test_modify_hard_stop_is_dry_run_by_default():
    async def scenario():
        client = IBKRClient(account="U123")
        client.live_trading_enabled = False
        ok = await client.modify_hard_stop("ADSK", "SELL", 100, 92.50, order_id=42, oca_group="single_43")
        assert ok is True
    asyncio.run(scenario())
    print("PASS: modify_hard_stop dry-runs unless IBKR_LIVE_TRADING_ENABLED=1")


def test_place_single_stock_bracket_rounds_prices_to_the_cent():
    """Confirmed live 2026-09-09: an ADSK entry filled and then had all three
    of its price-bearing bracket legs rejected with IBKR error 110 ("price
    does not conform to the minimum price variation"), because the resting
    price formulas return raw arithmetic with many decimals. The fail-safe
    flattened the unprotected position correctly, but the trade was lost."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 60
        placed = []
        client.placeOrder = lambda oid, contract, order: placed.append((oid, order))
        client.live_trading_enabled = True

        await client.place_single_stock_bracket(
            "ADSK", "SELL", 100, 93.455123, 101.0079, datetime(2026, 9, 9, 11, 0),
            pnl_stop_price=93.44691)

        by_id = {oid: o for oid, o in placed}
        prices = sorted(o.auxPrice for o in by_id.values() if o.orderType in ("STP", "MIT"))
        assert prices == [93.45, 93.46, 101.01], prices
        for p in prices:
            assert round(p, 2) == p
    asyncio.run(scenario())
    print("PASS: every price-bearing bracket leg is rounded to the cent before it reaches IBKR")


def test_place_single_stock_bracket_is_dry_run_by_default():
    async def scenario():
        client = IBKRClient(account="U123")
        client.live_trading_enabled = False
        client._next_order_id = 20
        hard_stop_id, tp_id, timed_exit_id, pnl_stop_id, trail_id = await client.place_single_stock_bracket(
            "AAPL", "SELL", 100, 95.50, 101.0, datetime(2026, 9, 4, 11, 0))
        assert (hard_stop_id, tp_id, timed_exit_id) == (20, 21, 22)
        assert pnl_stop_id is None  # no pnl_stop_price given -> three legs, as before
        assert trail_id is None  # no trailing_take_profit_amount given
        assert client._next_order_id == 23
    asyncio.run(scenario())
    print("PASS: place_single_stock_bracket dry-runs unless IBKR_LIVE_TRADING_ENABLED=1")


def test_place_single_stock_bracket_places_three_orders_one_oca_group():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 30
        placed = []
        client.placeOrder = lambda order_id, contract, order: placed.append((order_id, contract, order))
        client.live_trading_enabled = True

        hard_stop_id, tp_id, timed_exit_id, pnl_stop_id, trail_id = await client.place_single_stock_bracket(
            "AAPL", "SELL", 100, 95.50, 101.0, datetime(2026, 9, 4, 11, 0))

        assert len(placed) == 3
        assert pnl_stop_id is None and trail_id is None
        orders_by_id = {order_id: order for order_id, _, order in placed}
        assert orders_by_id[hard_stop_id].orderType == "STP" and orders_by_id[hard_stop_id].auxPrice == 95.50
        assert orders_by_id[tp_id].orderType == "MIT" and orders_by_id[tp_id].auxPrice == 101.0
        assert orders_by_id[timed_exit_id].orderType == "MKT"
        for _, contract, order in placed:
            assert contract.symbol == "AAPL" and contract.secType == "STK"
            assert order.action == "SELL"
            assert order.ocaGroup == orders_by_id[tp_id].ocaGroup  # all three share one OCA group
    asyncio.run(scenario())
    print("PASS: place_single_stock_bracket places Hard Stop/Take Profit/Timed Exit as one OCA group")


def test_place_single_stock_bracket_adds_trailing_take_profit_as_a_fifth_leg():
    """Per the client 2026-09-15: "Trailing Take Profit = $1... if INTU
    rises by $1, stop loss will trail by $1, protecting that profit."
    Confirmed as IBKR's own native TRAIL order type, running alongside
    the other four legs in the same OCA group, not replacing any of
    them."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 40
        placed = []
        client.placeOrder = lambda order_id, contract, order: placed.append((order_id, contract, order))
        client.live_trading_enabled = True

        hard_stop_id, tp_id, timed_exit_id, pnl_stop_id, trail_id = await client.place_single_stock_bracket(
            "INTU", "SELL", 29, 327.14, 337.29, datetime(2026, 9, 15, 11, 0),
            pnl_stop_price=326.90, trailing_take_profit_amount=1.00)

        assert len(placed) == 5
        assert trail_id is not None
        orders_by_id = {order_id: order for order_id, _, order in placed}
        trail_order = orders_by_id[trail_id]
        # auxPrice here is the TRAIL AMOUNT ($1), not a price level like
        # every other leg in this bracket.
        assert trail_order.orderType == "TRAIL" and trail_order.auxPrice == 1.00
        assert trail_order.action == "SELL"
        assert trail_order.ocaGroup == orders_by_id[tp_id].ocaGroup  # same OCA group as the rest
    asyncio.run(scenario())
    print("PASS: trailing_take_profit_amount adds a TRAIL order as a fifth leg of the same OCA group")


def test_place_pair_trade_is_dry_run_by_default():
    async def scenario():
        client = IBKRClient(account="U123")
        client.live_trading_enabled = False  # explicit, not relying on the ambient env var being unset
        client._next_order_id = 1
        order_id_a, order_id_b = await client.place_pair_trade("TYL", 262.0, "BUY", "WDAY", 393.0, "SELL")
        assert (order_id_a, order_id_b) == (1, 2)
        assert client._next_order_id == 3
    asyncio.run(scenario())
    print("PASS: place_pair_trade dry-runs (no live order sent, no IBKR calls at all) unless IBKR_LIVE_TRADING_ENABLED=1")


def test_place_pair_trade_places_two_independent_orders():
    """Rebuilt 2026-09-03 -- Combo/BAG orders removed per the client:
    "Delete Combo/Bag Order." Each leg is now its own independent MKT
    order with the real fractional quantity, not a whole-number combo
    ratio."""
    async def scenario():
        import os
        client = IBKRClient(account="U123")
        client._next_order_id = 1

        placed = []
        client.placeOrder = lambda order_id, contract, order: placed.append((order_id, contract, order))
        os.environ["IBKR_LIVE_TRADING_ENABLED"] = "1"
        client.live_trading_enabled = True
        try:
            order_id_a, order_id_b = await client.place_pair_trade(
                "TYL", 262.0, "BUY", "WDAY", 353.7, "SELL")
        finally:
            del os.environ["IBKR_LIVE_TRADING_ENABLED"]

        assert (order_id_a, order_id_b) == (1, 2)
        assert len(placed) == 2
        order_id_1, contract_1, order_1 = placed[0]
        order_id_2, contract_2, order_2 = placed[1]
        assert order_id_1 == 1 and contract_1.symbol == "TYL" and contract_1.secType == "STK"
        assert order_1.action == "BUY" and order_1.totalQuantity == 262.0 and order_1.orderType == "MKT"
        assert order_id_2 == 2 and contract_2.symbol == "WDAY" and contract_2.secType == "STK"
        assert order_2.action == "SELL" and order_2.totalQuantity == 353.7 and order_2.orderType == "MKT"
        assert order_1.eTradeOnly is False and order_1.firmQuoteOnly is False
    asyncio.run(scenario())
    print("PASS: place_pair_trade places both legs as two independent MKT orders, "
          "the fractional leg keeping its real fractional quantity")


def test_combo_bracket_child_order_mit_uses_aux_price():
    # MIT (Market-If-Touched), not LMT, per the client 2026-08-19/20:
    # "Take Profit = MKT @ Trigger" -- a resting LMT could simply never
    # fill if the market gaps past it.
    o = combo_bracket_child_order("MIT", 12.34, 131, account="U123", oca_group="pair_5")
    assert o.action == "BUY"
    assert o.orderType == "MIT"
    assert o.auxPrice == 12.34
    assert o.ocaGroup == "pair_5"
    assert o.ocaType == 1
    assert o.transmit is True
    assert o.smartComboRoutingParams[0].tag == "NonGuaranteed"
    print("PASS: combo_bracket_child_order builds a MIT closing order with the take-profit price")


def test_combo_bracket_child_order_stp_uses_aux_price():
    o = combo_bracket_child_order("STP", 9.99, 131, account="U123", oca_group="pair_5")
    assert o.orderType == "STP"
    assert o.auxPrice == 9.99
    assert o.ocaGroup == "pair_5"
    print("PASS: combo_bracket_child_order builds an STP closing order with the hard-stop price")


def test_combo_bracket_child_order_mkt_has_no_aux_price_and_sets_good_after_time():
    # The Timed Exit, added 2026-08-24: a plain MKT order has no trigger
    # price at all, and goodAfterTime is what makes IBKR itself hold it
    # inactive until the 60-minute mark instead of firing at submission.
    activation = datetime(2026, 8, 24, 10, 45, 0)
    o = combo_bracket_child_order("MKT", 0.0, 131, account="U123", oca_group="pair_5",
                                   good_after_time=activation)
    assert o.orderType == "MKT"
    assert o.auxPrice == UNSET_DOUBLE  # ibapi's own sentinel for an unset numeric field, never explicitly set
    assert o.goodAfterTime == "20260824 10:45:00 US/Eastern"
    assert o.ocaGroup == "pair_5"
    print("PASS: combo_bracket_child_order builds an MKT Timed Exit with goodAfterTime, no auxPrice")


def test_place_pair_bracket_is_dry_run_by_default():
    async def scenario():
        client = IBKRClient(account="U123")
        client.live_trading_enabled = False
        client._next_order_id = 1
        activation = datetime(2026, 8, 24, 10, 45, 0)
        hard_stop_id, tp_id, timed_exit_id = await client.place_pair_bracket(
            "TYL", "BUY", "WDAY", "SELL", 262, 393, hard_stop_price=-4.0, take_profit_price=-12.0,
            timed_exit_activation=activation)
        assert (hard_stop_id, tp_id, timed_exit_id) == (1, 2, 3)
        assert client._next_order_id == 4
    asyncio.run(scenario())
    print("PASS: place_pair_bracket dry-runs (no live order sent) unless IBKR_LIVE_TRADING_ENABLED=1")


def test_place_pair_bracket_places_an_oca_group_of_three_on_the_closing_combo():
    """The closing combo must have BOTH legs reversed vs. the entry (BUY
    TYL/SELL WDAY entry -> SELL TYL/BUY WDAY closing), sized to the actual
    filled quantity passed in, and all three orders share one OCA group."""
    async def scenario():
        import os
        client = IBKRClient(account="U123")
        client._next_order_id = 1
        con_ids = {"TYL": 111, "WDAY": 222}
        client.get_contract_id = lambda symbol: asyncio.sleep(0, result=con_ids[symbol])

        placed = []
        client.placeOrder = lambda order_id, contract, order: placed.append((order_id, contract, order))
        os.environ["IBKR_LIVE_TRADING_ENABLED"] = "1"
        client.live_trading_enabled = True
        activation = datetime(2026, 8, 24, 10, 45, 0)
        try:
            hard_stop_id, tp_id, timed_exit_id = await client.place_pair_bracket(
                "TYL", "BUY", "WDAY", "SELL", 262, 393, hard_stop_price=-4.0, take_profit_price=-12.0,
                timed_exit_activation=activation)
        finally:
            del os.environ["IBKR_LIVE_TRADING_ENABLED"]

        assert len(placed) == 3
        hs_id, hs_contract, hs_order = placed[0]
        tp_id_placed, tp_contract, tp_order = placed[1]
        te_id, te_contract, te_order = placed[2]
        assert hs_id == hard_stop_id and tp_id_placed == tp_id and te_id == timed_exit_id
        for contract in (hs_contract, tp_contract, te_contract):
            leg_a, leg_b = contract.comboLegs
            assert leg_a.conId == 111 and leg_a.action == "SELL" and leg_a.ratio == 2
            assert leg_b.conId == 222 and leg_b.action == "BUY" and leg_b.ratio == 3
        assert hs_order.orderType == "STP" and hs_order.auxPrice == -4.0
        assert tp_order.orderType == "MIT" and tp_order.auxPrice == -12.0
        assert te_order.orderType == "MKT" and te_order.goodAfterTime == "20260824 10:45:00 US/Eastern"
        assert hs_order.ocaGroup == tp_order.ocaGroup == te_order.ocaGroup
        assert hs_order.ocaType == 1 and tp_order.ocaType == 1 and te_order.ocaType == 1
    asyncio.run(scenario())
    print("PASS: place_pair_bracket places TP+HardStop+TimedExit on a reversed-leg closing combo, "
          "sized to the actual fill, sharing one OCA group")


def test_bracket_order_filled_returns_none_until_status_is_filled():
    client = IBKRClient(account="U123")
    client._order_status[7] = "Submitted"
    assert client.bracket_order_filled(7) is None
    print("PASS: bracket_order_filled returns None while still working")


def test_bracket_order_filled_returns_the_fill_price_once_filled():
    client = IBKRClient(account="U123")
    client._order_status[7] = "Filled"
    client._order_fill_price[7] = 11.98
    assert client.bracket_order_filled(7) == 11.98
    print("PASS: bracket_order_filled returns the recorded fill price once the order status is Filled")


def test_bracket_order_filled_none_for_none_order_id():
    client = IBKRClient(account="U123")
    assert client.bracket_order_filled(None) is None
    print("PASS: bracket_order_filled is a no-op for a None order id (bracket never placed)")


def test_bracket_order_abandoned_true_for_terminal_statuses_without_a_fill():
    client = IBKRClient(account="U123")
    for status in ("Cancelled", "ApiCancelled", "Inactive"):
        client._order_status[7] = status
        assert client.bracket_order_abandoned(7) is True, f"status={status} should count as abandoned"
    print("PASS: bracket_order_abandoned is True for Cancelled/ApiCancelled/Inactive")


def test_bracket_order_abandoned_false_while_still_working_or_filled():
    client = IBKRClient(account="U123")
    for status in ("Submitted", "PreSubmitted", "Filled"):
        client._order_status[7] = status
        assert client.bracket_order_abandoned(7) is False, f"status={status} should not count as abandoned"
    print("PASS: bracket_order_abandoned is False while working or once actually filled")


def test_bracket_order_abandoned_none_for_none_order_id():
    client = IBKRClient(account="U123")
    assert client.bracket_order_abandoned(None) is False
    print("PASS: bracket_order_abandoned is False (not abandoned) for a None order id")


def test_margin_probe_uses_a_priceable_order_type():
    """Regression guard. The probe used a STP with auxPrice=0, which IBKR
    will not price -- confirmed live: the whatIf never returns and the call
    just times out after 15s. That silently broke report sizing and would
    have killed the session on the first real entry."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_order_id = 1
        placed = []

        def fake_place(order_id, contract, order):
            placed.append(order)
            # resolve the way IBKR does for a priceable whatIf
            client._margin_futures[order_id].set_result(128.17)

        client.placeOrder = fake_place
        margin = await client.get_initial_margin("AVGO", "BUY", 1)

        assert margin == 128.17
        order = placed[0]
        assert order.whatIf is True
        assert order.orderType == "MKT", "STP with auxPrice=0 is unpriceable and times out"
    asyncio.run(scenario())
    print("PASS: margin probe uses a MKT whatIf, not an unpriceable STP at auxPrice 0")


def test_get_pair_initial_margin_sums_two_independent_leg_probes():
    """Rebuilt 2026-09-03 -- Combo/BAG orders removed, so there is no more
    combo-level margin treatment to probe as one whatIf. Per
    get_pair_initial_margin's own docstring: with two independent orders,
    summing each leg's own single-leg margin IS the accurate estimate now
    (the opposite of the pre-2026-09-03 combo case, where summing
    understated the real combo requirement)."""
    async def scenario():
        client = IBKRClient(account="U123")

        async def fake_get_initial_margin(symbol, action, quantity):
            return {"BAC": 70.0, "JPM": 49.0}[symbol]

        client.get_initial_margin = fake_get_initial_margin
        margin = await client.get_pair_initial_margin("BAC", "BUY", 10.0, "JPM", "SELL", 9.0)
        assert margin == 119.0  # 70 + 49, at the default units=1 (no-op division)
    asyncio.run(scenario())
    print("PASS: get_pair_initial_margin sums two independent single-leg whatIf probes")


def test_get_pair_initial_margin_divides_by_units_for_a_real_candidate_size():
    """Confirmed live 2026-08-11: over 70 entry rejections in one day, every
    one needing 2-4% more margin than a 1-unit-scaled estimate predicted --
    probing at the real candidate size and dividing by the unit count gives
    the true per-unit figure at that scale, not an extrapolation from a
    single share."""
    async def scenario():
        client = IBKRClient(account="U123")
        seen_quantities = []

        async def fake_get_initial_margin(symbol, action, quantity):
            seen_quantities.append((symbol, quantity))
            return {"BAC": 35_000.0, "JPM": 30_000.0}[symbol]

        client.get_initial_margin = fake_get_initial_margin
        # 500 units of a 10:9 per-unit ratio -> real candidate size 5000/4500
        margin_per_unit = await client.get_pair_initial_margin(
            "BAC", "BUY", 5000.0, "JPM", "SELL", 4500.0, units=500)

        assert margin_per_unit == 130.0  # (35,000 + 30,000) / 500
        assert ("BAC", 5000.0) in seen_quantities
        assert ("JPM", 4500.0) in seen_quantities
    asyncio.run(scenario())
    print("PASS: get_pair_initial_margin probes at the real candidate size and returns margin per unit at that scale")


def test_position_callback_ignores_other_accounts_on_the_same_login():
    """Confirmed live 2026-08-26: reqPositions() reports every sub-account
    on the login, not just self.account -- same multi-account gotcha as
    accountSummary. Without this filter, a stray position on a SIBLING
    project's account (shared login) got summed into THIS account's
    close-out list and closed with a real order against this account: a
    SELL 3 against an already-flat/smaller real position on this account
    that overshot to a naked short (-3)."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._positions = []
        client._positions_future = None
        client.position("U123", _leg_contract("AAPL"), 100.0, 150.0)
        client.position("U999", _leg_contract("AAPL"), 3.0, 200.0)  # sibling account, must be ignored
        client.position("U123", _leg_contract("MSFT"), -50.0, 300.0)
        client.position("U999", _leg_contract("IBKR"), 3.0, 90.0)  # sibling account, must be ignored
        assert client._positions == [("AAPL", 100.0), ("MSFT", -50.0)]
    asyncio.run(scenario())
    print("PASS: position() only collects this account's own positions, ignoring any other "
          "account reported on the same multi-account login")


def test_flatten_symbols_only_touches_the_named_pair():
    """A single pair exiting used to call flatten_all(), which closed EVERY
    open position at the broker while the bot's own state still believed the
    other pairs were open."""
    async def scenario():
        import os
        client = IBKRClient(account="U123")
        client._next_order_id = 1
        client._current_positions = lambda: asyncio.sleep(0, result=[
            ("AVGO", 100.0), ("NVDA", -245.0), ("INTC", 500.0), ("MU", -54.0),
        ])
        placed = []
        client.placeOrder = lambda oid, contract, order: placed.append((contract.symbol, order))
        cancelled_all = []
        client.reqGlobalCancel = lambda: cancelled_all.append(True)

        os.environ["IBKR_LIVE_TRADING_ENABLED"] = "1"
        client.live_trading_enabled = True
        try:
            order_ids = await client.flatten_symbols({"AVGO", "NVDA"})
        finally:
            del os.environ["IBKR_LIVE_TRADING_ENABLED"]

        # NVDA (the short leg) closes before AVGO (the long leg) -- see
        # _close_out's own docstring: closing the short first releases
        # margin instead of consuming more of it, at 98% margin utilization.
        assert [s for s, _ in placed] == ["NVDA", "AVGO"]  # INTC/MU left alone
        assert placed[0][1].action == "BUY" and placed[0][1].totalQuantity == 245.0   # short -> buy, first
        assert placed[1][1].action == "SELL" and placed[1][1].totalQuantity == 100.0  # long -> sell, second
        assert cancelled_all == []  # must NOT globally cancel another pair's orders
        # Confirmed live 2026-08-20 (the bracket price rejection incident):
        # a flattening MKT order can be rejected too, and every caller used
        # to assume flatten always worked and clear its own tracked state
        # immediately after. Returning the order ids lets the caller check
        # order_was_rejected on each before trusting the position is flat.
        assert order_ids == [1, 2]
    asyncio.run(scenario())
    print("PASS: flatten_symbols closes only the named pair's legs, never issues a global cancel, "
          "and returns the closing order ids so the caller can verify they weren't rejected")


def test_flatten_symbols_noop_when_pair_has_no_position():
    async def scenario():
        client = IBKRClient(account="U123")
        client.live_trading_enabled = True
        client._current_positions = lambda: asyncio.sleep(0, result=[("INTC", 500.0)])
        placed = []
        client.placeOrder = lambda oid, contract, order: placed.append(contract.symbol)
        await client.flatten_symbols({"AVGO", "NVDA"})
        assert placed == []
    asyncio.run(scenario())
    print("PASS: flatten_symbols is a no-op when the pair holds nothing")


def test_parse_todays_liquid_hours_normal_day():
    liquid_hours = "20260727:0930-20260727:1600;20260728:0930-20260728:1600"
    result = parse_todays_liquid_hours(liquid_hours, today=date(2026, 7, 27))
    assert result == (datetime(2026, 7, 27, 9, 30), datetime(2026, 7, 27, 16, 0))
    print("PASS: parse_todays_liquid_hours parses a normal full trading day")


def test_parse_todays_liquid_hours_early_close_day():
    liquid_hours = "20261127:0930-20261127:1300;20261128:0930-20261128:1600"
    result = parse_todays_liquid_hours(liquid_hours, today=date(2026, 11, 27))
    assert result == (datetime(2026, 11, 27, 9, 30), datetime(2026, 11, 27, 13, 0))
    print("PASS: parse_todays_liquid_hours parses an early-close day (1pm close)")


def test_parse_todays_liquid_hours_market_holiday():
    liquid_hours = "20260904:CLOSED;20260905:0930-20260905:1600"
    result = parse_todays_liquid_hours(liquid_hours, today=date(2026, 9, 4))
    assert result is None
    print("PASS: parse_todays_liquid_hours returns None on a market holiday")


def test_wait_ready_resolves_once_next_valid_id_arrives():
    """Regression guard for the bug that made every session die on the first
    request regardless of memory: requests must not fire until nextValidId
    actually arrives."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_valid_id_future = client._loop.create_future()
        sent = []
        client.reqMarketDataType = lambda kind: sent.append(kind)

        client.nextValidId(5)
        await client.wait_ready(timeout=1)

        assert client._next_order_id == 5
        assert sent == [1]
    asyncio.run(scenario())
    print("PASS: wait_ready resolves once nextValidId fires, then requests live market data")


def test_wait_ready_times_out_if_next_valid_id_never_arrives():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._next_valid_id_future = client._loop.create_future()
        try:
            await client.wait_ready(timeout=0.05)
            assert False, "expected a timeout"
        except asyncio.TimeoutError:
            pass
    asyncio.run(scenario())
    print("PASS: wait_ready times out instead of hanging forever if nextValidId never arrives")


def test_parse_todays_liquid_hours_date_not_found():
    liquid_hours = "20260728:0930-20260728:1600"
    result = parse_todays_liquid_hours(liquid_hours, today=date(2026, 7, 27))
    assert result is None
    print("PASS: parse_todays_liquid_hours returns None when today isn't in the liquidHours string")

def test_rejected_order_is_recorded_against_its_order_id():
    """placeOrder returns immediately and the rejection lands on error() a
    moment later. Without tracking it, a refused order looked identical to a
    working one -- which is how the bot came to hold a GE/RTX position IBKR
    had rejected for margin, show it on the dashboard, and book $543 of
    imaginary take-profit on it."""
    import asyncio

    client = IBKRClient(account="DU0000001")
    client._loop = asyncio.new_event_loop()

    client.error(44, 201, "Order rejected - Available Funds are insufficient")
    assert 44 in client._rejected_orders
    assert "201" in client._rejected_orders[44]

    # 110 -- confirmed live 2026-08-20 as the exact code behind the
    # unrounded-bracket-price incident; added to ORDER_REJECTED_CODES
    # 2026-08-24 so a recurrence (e.g. a different contract's tick size)
    # is actually detected, not just prevented for this bot's own prices.
    client.error(47, 110, "The price does not conform to the minimum price variation for this contract.")
    assert 47 in client._rejected_orders

    # 404 -- the bot's SELL entries are real short sales; a hard-to-borrow
    # rejection on one must be recognized, not treated as still working.
    client.error(48, 404, "Shares for this order are not immediately available for short sale.")
    assert 48 in client._rejected_orders

    client.error(45, 2104, "Market data farm connection is OK")
    client.error(46, 10349, "Order TIF was set to DAY based on order preset.")
    assert 45 not in client._rejected_orders, "an informational notice is not a rejection"
    assert 46 not in client._rejected_orders, "a TIF adjustment warning is not a rejection"

    async def check():
        assert await client.order_was_rejected(44, wait_seconds=0.1) is not None
        assert await client.order_was_rejected(99, wait_seconds=0.1) is None

    client._loop.run_until_complete(check())
    client._loop.close()
    print("PASS: a rejected order is recorded, notices and TIF warnings are not")


def test_second_audit_pass_codes_are_recorded_as_rejections():
    """103/200/434/449/10268, added from a second independent audit pass,
    2026-08-24 -- each individually verified against IBKR's own published
    message codes as unambiguous. Deliberately NOT testing 399 or 10197
    here: 399 is a generic catch-all that can also carry a benign warning,
    and 10197 turned out (per IBKR's own docs) to be about a competing
    market-data session, not order rejection at all -- neither was added."""
    client = IBKRClient(account="DU0000001")
    for order_id, code in [(1, 103), (2, 200), (3, 434), (4, 449), (5, 10268)]:
        client.error(order_id, code, f"message for {code}")
        assert order_id in client._rejected_orders, f"code {code} should be recorded as a rejection"
    print("PASS: the second audit pass's verified-unambiguous codes are all recorded as rejections")


def test_order_was_rejected_detects_inactive_status_with_no_matching_error_code():
    """Independent audit finding, confirmed real: IBKR can reject an order
    by moving its status straight to Inactive with no error() callback that
    ORDER_REJECTED_CODES would ever recognize. Checking _rejected_orders
    alone reported this dead order as still working."""
    client = IBKRClient(account="DU0000001")
    client._loop = asyncio.new_event_loop()
    client.orderStatus(55, "Inactive", 0, 10, 0.0, 0, 0, 0.0, 0, "", 0.0)

    async def check():
        reason = await client.order_was_rejected(55, wait_seconds=0.1)
        assert reason is not None and "Inactive" in reason

    client._loop.run_until_complete(check())
    client._loop.close()
    print("PASS: an order that goes Inactive with no matching error code is still detected as rejected")


def test_resolve_future_sets_result_normally():
    async def scenario():
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        _resolve_future(future, 42)
        assert future.result() == 42
    asyncio.run(scenario())
    print("PASS: _resolve_future sets a still-pending future's result normally")


def test_resolve_future_is_a_noop_on_an_already_done_future():
    """Independent audit finding: every EWrapper callback pre-checks
    `not future.done()` on the IBAPI reader thread before scheduling
    set_result via call_soon_threadsafe, but that pre-check and the actual
    set don't happen atomically -- if the awaiting coroutine's own
    asyncio.wait_for times out (cancelling the future ON THE LOOP THREAD)
    in between, the scheduled set_result could still run against an
    already-done future and raise InvalidStateError. Re-checking done()
    here, in the same callback as the set, closes that window completely."""
    async def scenario():
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.cancel()
        _resolve_future(future, 42)  # must not raise InvalidStateError
    asyncio.run(scenario())
    print("PASS: _resolve_future silently no-ops against an already-cancelled/done future")


def test_account_summary_ignores_other_accounts_on_a_multi_account_login():
    """reqAccountSummary("All", ...) reports every sub-account on the login,
    not just self.account. Without filtering, the pending future resolved on
    whichever account replied first -- on a real 3-account login this
    silently returned a different sub-account's equity, once as $2,000
    against the configured account's real $25,000."""
    client = IBKRClient(account="U1000001")
    client._loop = asyncio.new_event_loop()
    client._elv_future = client._loop.create_future()

    async def scenario():
        # a different sub-account answers first
        client.accountSummary(1, "U1000002", "EquityWithLoanValue", "2000.00", "USD")
        assert not client._elv_future.done(), "must not resolve on an account it wasn't asked about"

        # the configured account answers next
        client.accountSummary(1, "U1000001", "EquityWithLoanValue", "25000.00", "USD")
        await asyncio.sleep(0)
        assert client._elv_future.done()
        assert client._elv_future.result() == 25000.00

    client._loop.run_until_complete(scenario())
    client._loop.close()
    print("PASS: EquityWithLoanValue only resolves from the account actually configured")


def test_exec_details_accumulates_latest_cumulative_fill_per_leg():
    """execDetails fires once per leg per fill report on a combo order --
    contract is the individual leg's own contract (e.g. real BAC/JPM
    stock), not the BAG. cumQty/avgPrice are IBKR's own running totals per
    order+contract, so recording the latest report is correct."""
    client = IBKRClient(account="U123")
    client.execDetails(-1, _leg_contract("BAC"), _execution(order_id=100, cum_qty=50, avg_price=30.0))
    client.execDetails(-1, _leg_contract("JPM"), _execution(order_id=100, cum_qty=40, avg_price=150.0, side="SLD"))
    assert client._leg_fills[100]["BAC"] == {"qty": 50, "avg_price": 30.0, "side": "BOT"}
    assert client._leg_fills[100]["JPM"] == {"qty": 40, "avg_price": 150.0, "side": "SLD"}
    print("PASS: execDetails records each leg's own contract and fill separately, keyed by order id")


def test_exec_details_overwrites_with_latest_cumulative_not_summing():
    """cumQty is already the running total IBKR maintains -- a second
    partial-fill report must REPLACE the stored quantity, not add to it."""
    client = IBKRClient(account="U123")
    client.execDetails(-1, _leg_contract("BAC"), _execution(100, cum_qty=30, avg_price=30.0))
    client.execDetails(-1, _leg_contract("BAC"), _execution(100, cum_qty=50, avg_price=30.2))
    assert client._leg_fills[100]["BAC"]["qty"] == 50  # not 30+50=80
    assert client._leg_fills[100]["BAC"]["avg_price"] == 30.2
    print("PASS: a later execution report overwrites with IBKR's own cumulative total, doesn't get re-summed")


def test_refresh_executions_from_broker_requests_and_waits():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        requested = []

        def fake_req_executions(req_id, filt):
            requested.append((req_id, filt.acctCode))
            client.execDetails(req_id, _leg_contract("BAC"), _execution(200, cum_qty=100, avg_price=30.0))
            client.execDetailsEnd(req_id)

        client.reqExecutions = fake_req_executions
        await client.refresh_executions_from_broker()

        assert len(requested) == 1
        assert requested[0][1] == "U123"
        assert client._leg_fills[200]["BAC"]["qty"] == 100
    asyncio.run(scenario())
    print("PASS: refresh_executions_from_broker requests for the configured account and populates _leg_fills from the response")


def test_get_pair_actual_fill_returns_reconciled_quantities_and_prices():
    """The client's own worked example: target 100 BAC / 80 JPM, actual
    100 BAC / 76 JPM -- get_pair_actual_fill must surface the REAL 76, not
    the originally ordered 80. Rebuilt 2026-09-03 for two independent
    order ids (one per leg) instead of one shared combo order id -- this
    is the Leg Risk Safeguard's own reconciliation."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._order_status[100] = "Filled"
        client._order_status[200] = "Filled"
        client.execDetails(-1, _leg_contract("BAC"), _execution(100, cum_qty=100, avg_price=30.0))
        client.execDetails(-1, _leg_contract("JPM"), _execution(200, cum_qty=76, avg_price=150.0, side="SLD"))
        result = await client.get_pair_actual_fill(100, "BAC", 200, "JPM")
        assert result == (100, 30.0, 76, 150.0)
    asyncio.run(scenario())
    print("PASS: get_pair_actual_fill reconciles the real per-leg fill across two independent orders")


def test_get_pair_actual_fill_falls_back_to_broker_records_before_giving_up():
    """If nothing was captured live (a dropped execDetails callback), it
    must check the broker's own execution records via
    refresh_executions_from_broker before concluding there was no fill --
    confirmed here without a real IBKR connection by stubbing that method
    to populate _leg_fills the way a real reqExecutions response would."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._order_status[101] = "Filled"
        client._order_status[201] = "Filled"

        async def fake_refresh():
            client._leg_fills[101] = {"BAC": {"qty": 100, "avg_price": 30.0, "side": "BOT"}}
            client._leg_fills[201] = {"JPM": {"qty": 76, "avg_price": 150.0, "side": "SLD"}}
        client.refresh_executions_from_broker = fake_refresh

        result = await client.get_pair_actual_fill(101, "BAC", 201, "JPM", wait_seconds=0.1)
        assert result == (100, 30.0, 76, 150.0)
    asyncio.run(scenario())
    print("PASS: falls back to the broker's own execution records when nothing was captured live")


def test_get_pair_actual_fill_returns_none_when_truly_nothing_filled():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._order_status[102] = "Cancelled"
        client._order_status[202] = "Cancelled"
        client.refresh_executions_from_broker = lambda: asyncio.sleep(0)  # broker confirms nothing either

        result = await client.get_pair_actual_fill(102, "BAC", 202, "JPM", wait_seconds=0.1)
        assert result is None
    asyncio.run(scenario())
    print("PASS: returns None (not a fabricated fill) when neither order genuinely executed")


def test_get_pair_actual_fill_waits_out_a_partial_fill_instead_of_grabbing_it_early():
    """Per the 2026-08-24 fix: the wait loop no longer breaks just because
    both legs have produced SOME execDetails report -- a large or
    less-liquid order can fill across several partial reports (cumQty
    growing with each one), and reading _leg_fills the instant both legs
    show any report at all risked capturing a still-partial quantity and
    sizing the Take Profit/Stop Loss bracket for only a fraction of the
    real position. Only BOTH orders' own status reaching a terminal state
    (or the deadline) should end the wait -- two independent orders now,
    not one combo."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        # Only a partial fill so far -- order status stays non-terminal.
        client.execDetails(-1, _leg_contract("BAC"), _execution(104, cum_qty=30, avg_price=30.0))
        client.execDetails(-1, _leg_contract("JPM"), _execution(204, cum_qty=25, avg_price=150.0, side="SLD"))

        async def deliver_the_rest_of_the_fill():
            await asyncio.sleep(0.15)
            client.execDetails(-1, _leg_contract("BAC"), _execution(104, cum_qty=100, avg_price=30.0))
            client.execDetails(-1, _leg_contract("JPM"), _execution(204, cum_qty=76, avg_price=150.0, side="SLD"))
            client._order_status[104] = "Filled"
            client._order_status[204] = "Filled"

        asyncio.create_task(deliver_the_rest_of_the_fill())
        result = await client.get_pair_actual_fill(104, "BAC", 204, "JPM", wait_seconds=0.3)
        assert result == (100, 30.0, 76, 150.0)
    asyncio.run(scenario())
    print("PASS: waits out the deadline for the full fill instead of returning an early partial report")


def test_get_pair_actual_fill_none_on_one_sided_fill():
    """Real leg risk: BAC fills, JPM never does at all -- exactly the
    scenario the Leg Risk Safeguard exists for now that there is no combo
    order at all. Must not synthesize a two-leg position out of a real
    one-sided fill -- the caller is responsible for then checking IBKR's
    live positions and flattening the stray leg, this just must not
    silently pretend it's a matched pair."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client._order_status[103] = "Filled"
        client._order_status[203] = "Cancelled"
        client.execDetails(-1, _leg_contract("BAC"), _execution(103, cum_qty=100, avg_price=30.0))
        client.refresh_executions_from_broker = lambda: asyncio.sleep(0)  # broker agrees: JPM never filled

        result = await client.get_pair_actual_fill(103, "BAC", 203, "JPM", wait_seconds=0.1)
        assert result is None
    asyncio.run(scenario())
    print("PASS: a one-sided fill (real leg risk) returns None rather than a fabricated matched pair")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")


def test_cancel_orders_and_confirm_waits_for_terminal_state():
    """Per a third-party audit finding 2026-09-14: cancel_order_id (and
    the raw cancelOrder() it wraps) is fire-and-forget -- a caller acting
    immediately after can race the broker's own processing of the
    cancel. This waits for every id to reach a real terminal state."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client.live_trading_enabled = True
        cancelled = []
        client.cancelOrder = lambda oid: cancelled.append(oid)

        async def drive():
            await asyncio.sleep(0.05)
            # Not yet terminal -- PendingCancel does NOT count.
            client._order_status[10] = "PendingCancel"
            client._order_status[11] = "Cancelled"
            await asyncio.sleep(0.05)
            client._order_status[10] = "Cancelled"

        before = asyncio.get_event_loop().time()
        await asyncio.gather(drive(), client.cancel_orders_and_confirm([10, 11], timeout_seconds=1.0))
        elapsed = asyncio.get_event_loop().time() - before
        assert sorted(cancelled) == [10, 11]
        assert elapsed >= 0.1  # actually waited for order 10 to reach a real terminal state
    asyncio.run(scenario())
    print("PASS: cancel_orders_and_confirm cancels every id and waits for all to go terminal, "
          "PendingCancel does not count")


def test_cancel_orders_and_confirm_skips_none_entries():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client.live_trading_enabled = True
        cancelled = []
        client.cancelOrder = lambda oid: cancelled.append(oid)
        client._order_status[5] = "Cancelled"
        await client.cancel_orders_and_confirm([5, None, None], timeout_seconds=0.5)
        assert cancelled == [5]
    asyncio.run(scenario())
    print("PASS: cancel_orders_and_confirm skips None order ids, same convention as cancel_order_id")


def test_cancel_orders_and_confirm_no_op_when_all_none():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client.live_trading_enabled = True
        client.cancelOrder = lambda oid: (_ for _ in ()).throw(AssertionError("should never be called"))
        await client.cancel_orders_and_confirm([None, None], timeout_seconds=0.5)
    asyncio.run(scenario())
    print("PASS: cancel_orders_and_confirm is a clean no-op with no real ids to cancel")


def test_cancel_orders_and_confirm_dry_run_never_calls_cancelorder():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client.live_trading_enabled = False
        client.cancelOrder = lambda oid: (_ for _ in ()).throw(AssertionError("should never be called"))
        await client.cancel_orders_and_confirm([1, 2, 3])
    asyncio.run(scenario())
    print("PASS: dry run never touches the real cancelOrder API call")


def test_place_single_stock_bracket_clamps_prices_to_a_cent_floor():
    """Per a third-party audit finding 2026-09-14, verified directly:
    wide_hard_stop_price('BUY', 8.50, 5.00) == -0.85 -- a real, ordinary
    S&P 500 stock priced under a $5 trigger's distance produces a
    negative hard stop. round() alone never fixes a negative number, and
    IBKR rejects any non-positive auxPrice outright."""
    async def scenario():
        client = IBKRClient(account="U123")
        client._next_order_id = 200
        client.live_trading_enabled = True
        placed = []
        client.placeOrder = lambda oid, c, o: placed.append((oid, o))
        await client.place_single_stock_bracket(
            "XYZ", "SELL", 10, hard_stop_price=-0.85, take_profit_price=-0.02,
            timed_exit_activation=datetime(2026, 9, 14, 15, 0), pnl_stop_price=-1.5)
        by_id = {oid: o for oid, o in placed}
        # hard_stop_order_id=200, tp=201, timed_exit=202, pnl_stop=203
        assert by_id[200].auxPrice == 0.01
        assert by_id[201].auxPrice == 0.01
        assert by_id[203].auxPrice == 0.01
    asyncio.run(scenario())
    print("PASS: place_single_stock_bracket clamps negative hard stop/TP/PnL prices to a $0.01 floor")


def test_modify_hard_stop_clamps_price_to_a_cent_floor():
    async def scenario():
        client = IBKRClient(account="U123")
        client._loop = asyncio.get_event_loop()
        client.live_trading_enabled = True
        placed = []
        client.placeOrder = lambda oid, c, o: placed.append((oid, o))
        client.order_was_rejected = lambda order_id: _immediate_none()  # isolate the price clamp only
        await client.modify_hard_stop("XYZ", "SELL", 10, -2.5, order_id=50, oca_group="oca1")
        assert placed[0][1].auxPrice == 0.01
    asyncio.run(scenario())
    print("PASS: modify_hard_stop clamps a negative resting stop price to a $0.01 floor")


async def _immediate_none():
    return None
