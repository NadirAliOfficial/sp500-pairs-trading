"""IBKR order execution, following the same ibapi conventions proven on the
sibling SPY bot project:
  - EWrapper/EClient subclass, blocking run() in its own daemon thread
  - callbacks bridged to asyncio via loop.call_soon_threadsafe
  - reqMarketDataType(1) for live data (type 3 is 15-min delayed, useless intraday)
  - every Order() sets eTradeOnly=False, firmQuoteOnly=False (paper accounts
    silently reject orders without these)
  - order.account set explicitly on every order for multi-account logins

Combo/BAG orders removed 2026-09-03 per the client: "Delete Combo/Bag
Order." Each leg is now placed as its own independent order -- per the
client the same day: "buying fractional shares = True, selling fractional
shares = False." IBKR combo legs can't carry a fractional ratio
(ComboLeg.ratio is typed as a whole number in ibapi itself), which was
the whole reason position_sizing.py's Fractional Shares was False in the
first place; two independent orders have no such restriction, since each
leg's own Order.totalQuantity accepts a Decimal directly.

Two independent orders means no more atomic fill guarantee -- see
get_pair_actual_fill's own docstring for the Leg Risk Safeguard (a short
timeout after both orders are placed, checking both filled; if only one
did, flatten whatever's actually at the broker rather than leave a real,
unhedged single-leg position). A version of this same reconciliation
already existed for the Combo order too, since it was placed
NonGuaranteed (not the "Guaranteed by default" the client's original
spec assumed) -- this rebuild adapts that existing logic to two order
ids instead of one, it isn't new from nothing.

No live connection is opened by importing this module -- IBKRClient.connect_and_start()
must be called explicitly, and real order placement additionally requires
IBKR_LIVE_TRADING_ENABLED=1 in the environment as a deliberate safety gate
until the client confirms which account/VPS this runs against.
"""
import asyncio
import logging
import os
import re
import threading
from datetime import date, datetime

from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.contract import ComboLeg, Contract
from ibapi.execution import ExecutionFilter
from ibapi.order import Order
from ibapi.tag_value import TagValue
from ibapi.wrapper import EWrapper

from phase2.position_sizing import combo_ratio_and_units

load_dotenv()

logger = logging.getLogger("pairs_bot.ibkr_client")


def parse_todays_liquid_hours(liquid_hours: str, today: date) -> tuple[datetime, datetime] | None:
    """Parses IBKR's ContractDetails.liquidHours format
    ("YYYYMMDD:HHMM-YYYYMMDD:HHMM;YYYYMMDD:CLOSED;...") and returns today's
    (open, close) datetimes, or None if today is a market holiday. Pure
    string parsing -- no IBKR connection needed to test it."""
    today_str = today.strftime("%Y%m%d")
    for segment in liquid_hours.split(";"):
        if not segment.startswith(today_str):
            continue
        if segment.endswith("CLOSED"):
            return None
        open_part, close_part = segment.split("-")
        open_date, open_time = open_part.split(":")
        close_date, close_time = close_part.split(":")
        open_dt = datetime.strptime(open_date + open_time, "%Y%m%d%H%M")
        close_dt = datetime.strptime(close_date + close_time, "%Y%m%d%H%M")
        return open_dt, close_dt
    return None


def stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def market_order(action: str, quantity: float, account: str | None) -> Order:
    o = Order()
    o.action = action
    o.orderType = "MKT"
    o.totalQuantity = quantity
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    if account:
        o.account = account
    return o


def limit_order(action: str, quantity: float, limit_price: float, account: str | None) -> Order:
    """A plain LMT order -- used only as a deliberately NON-FILLABLE margin
    probe (see IBKRClient.probe_bracket_initial_margin): a BUY at \$1 or a
    SELL far above the market can never execute, but IBKR still evaluates
    its full margin impact and rejects the set on 201 if the account can't
    carry it, with the real required figure in the rejection text."""
    o = Order()
    o.action = action
    o.orderType = "LMT"
    o.totalQuantity = quantity
    o.lmtPrice = round(limit_price, 2)
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    if account:
        o.account = account
    return o


_INITIAL_MARGIN_RE = re.compile(r"[Ii]nitial [Mm]argin of\s*\[?\s*([\d,]+(?:\.\d+)?)")


def combo_contract(
    symbol_a: str, con_id_a: int, action_a: str, ratio_a: int,
    con_id_b: int, action_b: str, ratio_b: int,
) -> Contract:
    """BAG contract for a Guaranteed combo order -- per spec, IBKR
    SmartRouting treats stock-for-stock combos as Guaranteed by default,
    filling both legs together or not at all."""
    c = Contract()
    c.symbol = symbol_a
    c.secType = "BAG"
    c.currency = "USD"
    c.exchange = "SMART"

    leg_a = ComboLeg()
    leg_a.conId = con_id_a
    leg_a.ratio = ratio_a
    leg_a.action = action_a
    leg_a.exchange = "SMART"

    leg_b = ComboLeg()
    leg_b.conId = con_id_b
    leg_b.ratio = ratio_b
    leg_b.action = action_b
    leg_b.exchange = "SMART"

    c.comboLegs = [leg_a, leg_b]
    return c


def combo_market_order(total_quantity: int, account: str | None) -> Order:
    """Direction is set per leg on each ComboLeg; the parent combo order's
    own action is conventionally BUY, with totalQuantity being the number of
    combo units (one unit = ratio_a shares of leg A + ratio_b shares of leg B).

    Confirmed live against IBKR (paper): a MKT+MKT combo is rejected
    (error 10043, "Missing or invalid NonGuaranteed value") without this --
    contrary to the client's spec assumption that stock-for-stock combos are
    Guaranteed by default, IBKR only allows that for certain order type
    combinations. MKT+MKT (needed here for immediate execution when a
    divergence trigger fires) must be explicitly routed non-guaranteed."""
    o = Order()
    o.action = "BUY"
    o.orderType = "MKT"
    o.totalQuantity = total_quantity
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    o.smartComboRoutingParams = [TagValue("NonGuaranteed", "1")]
    if account:
        o.account = account
    return o


def combo_bracket_child_order(
    order_type: str, price: float, total_quantity: int, account: str | None,
    oca_group: str, good_after_time: datetime | None = None,
) -> Order:
    """A closing bracket order on the CLOSING combo -- STP for the Hard
    Stop, MIT (Market-If-Touched) for Take Profit, MKT for the Timed Exit
    -- same "action is always BUY, direction lives in the legs" convention
    as combo_market_order, since the closing BAG's own legs are already
    reversed vs. the entry.

    Take profit is MIT, not LMT, per the client 2026-08-19/20: "Take
    Profit = MKT @ Trigger" -- a resting LMT only fills at or better than
    its price and can simply never fill if the market gaps past it, which
    isn't what "MKT" means. MIT triggers an actual market order the
    instant price touches auxPrice, same as STP but for the FAVORABLE
    direction instead of the adverse one -- confirmed algebraically via
    risk_manager.bracket_order_prices' own derivation (take profit's
    closing price falls as the position gains, the opposite direction a
    plain STP triggers on).

    The Hard Stop (STP), added 2026-08-24 as a broker-side backstop once
    the native "0 Divergence" Stop Loss moved to bot-monitored logic, uses
    the same auxPrice mechanics as the old Stop Loss did -- see
    risk_manager.bracket_order_prices for the new "opposite side of Take
    Profit, 3x the trigger" price derivation.

    The Timed Exit (MKT), also added 2026-08-24 per the client's "Now bot
    will calculate 60 minutes Maximum Hold Time and attach it as a MKT
    Timed Exit order... sits with the broker as part of Bracket Order":
    good_after_time holds the order INACTIVE at IBKR until that timestamp,
    then IBKR itself transmits it as a live market order -- this is what
    makes Maximum Hold Time survive a bot crash, unlike the old
    script-checked exit in risk_manager.check_exit. No auxPrice for this
    one; a plain MKT order has no trigger price at all.

    Placed AFTER the entry's actual fill is confirmed (see
    place_pair_bracket), not bracketed to the entry order at submission
    time -- so there's no parentId here, unlike the mean-reversion-era
    version of this function. ocaGroup links all three orders together:
    whichever fills first cancels the other two at IBKR itself.

    price is auxPrice for STP/MIT, ignored for MKT."""
    o = Order()
    o.action = "BUY"
    o.orderType = order_type
    o.totalQuantity = total_quantity
    if order_type != "MKT":
        o.auxPrice = price
    if good_after_time is not None:
        o.goodAfterTime = good_after_time.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    o.smartComboRoutingParams = [TagValue("NonGuaranteed", "1")]
    o.ocaGroup = oca_group
    o.ocaType = 1  # cancel remaining orders in the group once one fills
    o.transmit = True
    if account:
        o.account = account
    return o


def stock_bracket_child_order(
    action: str, order_type: str, price: float, total_quantity: int, account: str | None,
    oca_group: str, good_after_time: datetime | None = None,
) -> Order:
    """The single-stock version of combo_bracket_child_order -- per the
    client 2026-09-04 (How To Set Resting Take Profit.pdf), restoring
    Take Profit and Hard Stop Loss as real resting orders less than a day
    after they moved to bot-monitored logic ("Why can it not be a resting
    order... You simply close a single stock instead of two stocks").

    Unlike the combo version, action is a REAL parameter here (the
    closing side -- SELL to close a BUY entry, BUY to close a SELL entry),
    not hardcoded to "BUY" -- there's no combo leg to carry the real
    direction on a plain stock order. No smartComboRoutingParams either;
    that tag is combo-specific.

    Same STP (Hard Stop) / MIT (Take Profit) / MKT (Timed Exit) order-type
    convention as combo_bracket_child_order, same ocaGroup mechanics
    (whichever fills first cancels the other two), same price=auxPrice
    for STP/MIT, ignored for MKT. See risk_manager.resting_take_profit_price/
    resting_hard_stop_price for the price levels themselves."""
    o = Order()
    o.action = action
    o.orderType = order_type
    o.totalQuantity = total_quantity
    if order_type != "MKT":
        o.auxPrice = price
    if good_after_time is not None:
        o.goodAfterTime = good_after_time.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    o.ocaGroup = oca_group
    o.ocaType = 1
    o.transmit = True
    if account:
        o.account = account
    return o


def _resolve_future(future: asyncio.Future, value) -> None:
    """set_result, re-checking done() on the event loop thread itself right
    before setting it -- independent audit finding, confirmed real as a
    genuine (if rare) race, though not the process crash the audit
    described: every EWrapper callback below pre-checks `not future.done()`
    on the IBAPI READER thread before scheduling this via
    call_soon_threadsafe, but that check and this call don't happen
    atomically. If the awaiting coroutine's own asyncio.wait_for times out
    in between -- which cancels the future ON THE LOOP THREAD -- this can
    still run afterward and call set_result on an already-done future,
    raising InvalidStateError. Asyncio's default handler just logs that
    (it does not crash the loop), so the previous code was not the
    "critical, crashes the bot" bug the audit rated it as -- but it's a
    real, needless race, and it's log noise that could bury a genuinely
    important error during a live session. Redoing the check in here, on
    the loop thread, in the same scheduled callback as the set, closes the
    window completely rather than narrowing it."""
    if not future.done():
        future.set_result(value)


class IBKRClient(EWrapper, EClient):
    def __init__(self, account: str | None = None):
        EClient.__init__(self, self)
        self.account = account
        self.live_trading_enabled = os.getenv("IBKR_LIVE_TRADING_ENABLED") == "1"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._next_order_id: int | None = None
        self._next_req_id = 9000
        self._next_valid_id_future: asyncio.Future | None = None
        self._elv_future: asyncio.Future | None = None
        self._margin_futures: dict[int, asyncio.Future] = {}
        self._order_status: dict[int, str] = {}
        self._positions_future: asyncio.Future | None = None
        self._positions: list[tuple[str, float]] = []
        self._contract_details_futures: dict[int, asyncio.Future] = {}
        self._rejected_orders: dict[int, str] = {}
        self._order_fill_price: dict[int, float] = {}
        # Per-leg actual fills, keyed by orderId -> symbol -> {"qty", "avg_price"}.
        # Populated from execDetails(), which IBKR fires once per leg per fill
        # report for a combo order, NOT from the combo-level order status --
        # per the client 2026-08-08: "Do not infer leg quantities solely from
        # the Combo/BAG-level fill confirmation." qty/avg_price are each
        # execution's cumQty/avgPrice, IBKR's own running totals for that
        # order+contract, so a later report simply overwrites with the latest
        # cumulative state rather than needing to be manually re-summed here.
        self._leg_fills: dict[int, dict[str, dict]] = {}
        self._exec_details_futures: dict[int, asyncio.Future] = {}

    # -- connection lifecycle --------------------------------------------

    def connect_and_start(self, host: str, port: int, client_id: int) -> None:
        self._loop = asyncio.get_event_loop()
        self._next_valid_id_future = self._loop.create_future()
        self.connect(host, port, client_id)
        threading.Thread(target=self.run, daemon=True).start()

    async def wait_ready(self, timeout: float = 15) -> None:
        """Waits for nextValidId -- the callback that signals the API
        handshake has actually completed. Sending requests before this
        arrives is a known TWS API pitfall: they can be silently dropped and
        the gateway just closes the connection, with no error callback and
        nothing useful in its own logs -- which is exactly what was
        happening every single session regardless of available memory, on
        both this bot and the sibling SPY bot. Call this right after
        connect_and_start(), before anything else."""
        await asyncio.wait_for(self._next_valid_id_future, timeout=timeout)
        self.reqMarketDataType(1)  # live data, never delayed, for intraday trading

    def _new_req_id(self) -> int:
        self._next_req_id += 1
        return self._next_req_id

    # -- EWrapper callbacks ------------------------------------------------

    def nextValidId(self, orderId: int) -> None:
        self._next_order_id = orderId
        if self._next_valid_id_future and not self._next_valid_id_future.done():
            self._loop.call_soon_threadsafe(_resolve_future, self._next_valid_id_future, orderId)

    def managedAccounts(self, accountsList: str) -> None:
        if not self.account:
            self.account = accountsList.split(",")[0]

    # Codes that mean the order will never execute. Recorded per order id so
    # the caller can tell a placed order from an accepted one -- placeOrder
    # returns immediately and the rejection arrives later on this callback,
    # so without this a rejected order looked identical to a working one and
    # the bot tracked a position the broker had refused.
    #
    # 110 ("price does not conform to the minimum price variation") added
    # 2026-08-24 -- confirmed live 2026-08-20 as the exact code behind the
    # unrounded-bracket-price incident (see bracket_order_prices), but that
    # fix only prevented THIS bot's own prices from tripping it. The code
    # itself was never added here, so order_was_rejected (used for the
    # bracket rejection check in main.py) still could not detect it if it
    # ever fired again for a different reason -- e.g. a contract whose tick
    # size doesn't match this bot's rounding assumption.
    #
    # 103/200/434/449/10268 added from a second independent audit pass,
    # 2026-08-24 -- each verified individually against IBKR's own published
    # message codes as an unambiguous "this order can never execute" signal
    # (duplicate order id, no security definition found, zero order size, a
    # malformed scale-order parameter, a deprecated order attribute).
    # Deliberately NOT adding the audit's other two suggestions: 10197 is
    # about a competing paper/live market-data session, unrelated to order
    # rejection at all (the audit's own description of it was wrong); 399
    # ("Order message error") is IBKR's own generic catch-all and can carry
    # a benign hold-until-market-open WARNING under the same code, not only
    # hard rejections -- adding it risked treating a perfectly good working
    # order as dead.
    # 404 ("Shares for this order are not immediately available for short
    # sale") added 2026-09-11 -- the bot's own SELL entries are real short
    # sales (the expansion leg direction), and a hard-to-borrow rejection
    # on one was never recognized as a rejection at all before this, so
    # order_was_rejected would have reported a dead SELL entry as still
    # working.
    ORDER_REJECTED_CODES = frozenset({
        103, 110, 200, 201, 202, 203, 321, 382, 383, 404, 434, 449, 10147, 10148, 10268,
    })

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # 2100-2169 are informational notices (market data farm connection
        # status etc.), not real errors -- logging them as warnings would
        # bury genuine problems in noise every time the gateway connects.
        if 2100 <= errorCode < 2170:
            logger.info(f"IBKR notice {errorCode} (reqId={reqId}): {errorString}")
        else:
            logger.warning(f"IBKR error {errorCode} (reqId={reqId}): {errorString}")
        if errorCode in self.ORDER_REJECTED_CODES:
            self._rejected_orders[reqId] = f"{errorCode}: {errorString}"

    def accountSummary(self, reqId, account, tag, value, currency):
        # reqAccountSummary("All", ...) reports every sub-account on the
        # login, not just self.account. Without this check, the future
        # resolved on whichever account answered first -- on a 3-account
        # login that is a coin flip between $0, $2k and $25k, and it would
        # have sized the 9:29am snapshot off the wrong account silently.
        if (tag == "EquityWithLoanValue" and account == self.account
                and self._elv_future and not self._elv_future.done()):
            self._loop.call_soon_threadsafe(_resolve_future, self._elv_future, float(value))

    def openOrder(self, orderId, contract, order, orderState):
        future = self._margin_futures.get(orderId)
        if future and not future.done():
            margin = float(orderState.initMarginChange)
            self._loop.call_soon_threadsafe(_resolve_future, future, margin)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                     permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self._order_status[orderId] = status
        if status == "Filled" and avgFillPrice:
            self._order_fill_price[orderId] = avgFillPrice

    def execDetails(self, reqId, contract, execution):
        # Fires once per leg per fill report for a combo order (contract is
        # the INDIVIDUAL leg's contract, e.g. the real BAC or JPM stock, not
        # the BAG) -- this is the per-leg ground truth the client asked for,
        # separate from orderStatus's combo-level filled/avgFillPrice.
        # cumQty/avgPrice are IBKR's own running totals for this
        # order+contract, so simply overwriting with the latest report is
        # correct and handles multiple partial fills without needing to
        # manually re-sum shares*price here.
        self._leg_fills.setdefault(execution.orderId, {})[contract.symbol] = {
            "qty": execution.cumQty, "avg_price": execution.avgPrice, "side": execution.side,
        }
        future = self._exec_details_futures.get(reqId)
        if future and not future.done():
            self._loop.call_soon_threadsafe(_resolve_future, future, None)

    def execDetailsEnd(self, reqId):
        future = self._exec_details_futures.get(reqId)
        if future and not future.done():
            self._loop.call_soon_threadsafe(_resolve_future, future, None)

    def contractDetails(self, reqId, contractDetails):
        future = self._contract_details_futures.get(reqId)
        if future and not future.done():
            self._loop.call_soon_threadsafe(_resolve_future, future, contractDetails)

    def position(self, account, contract, position, avgCost):
        # reqPositions() reports every sub-account on the login, not just
        # self.account -- same reasoning as accountSummary's own account
        # check above. Without this, flatten_all()/flatten_symbols() would
        # append another account's stray position into this account's own
        # close-out list and place a real order against self.account for a
        # position it never actually held. Confirmed live 2026-08-26: a
        # 3-share IBKR stock position on the sibling project's account
        # (shared login) got summed in here and closed against THIS
        # account, a SELL 3 against an already-flat/smaller real position
        # that overshot to a naked short (-3).
        if position != 0 and account == self.account:
            self._positions.append((contract.symbol, position))

    def positionEnd(self):
        if self._positions_future and not self._positions_future.done():
            self._loop.call_soon_threadsafe(_resolve_future, self._positions_future, list(self._positions))

    # -- account / margin queries -------------------------------------------

    async def get_equity_with_loan_value(self) -> float:
        """Live account value snapshot -- caller freezes this at 9:29am ET
        per spec ("EquityWithLoanValue <= 9:29am ET EquityWithLoanValue")."""
        self._elv_future = self._loop.create_future()
        req_id = self._new_req_id()
        self.reqAccountSummary(req_id, "All", "EquityWithLoanValue")
        try:
            return await asyncio.wait_for(self._elv_future, timeout=15)
        finally:
            self.cancelAccountSummary(req_id)

    async def get_todays_market_hours(self) -> tuple[datetime, datetime] | None:
        """Uses SPY as a market-wide RTH calendar proxy via
        reqContractDetails() -- all US equity exchanges share the same
        holiday/early-close schedule, same technique as the sibling SPY
        bot's own 8:25am early-close check. Returns None if today is a
        market holiday."""
        req_id = self._new_req_id()
        future = self._loop.create_future()
        self._contract_details_futures[req_id] = future
        self.reqContractDetails(req_id, stock_contract("SPY"))
        try:
            details = await asyncio.wait_for(future, timeout=15)
        finally:
            self._contract_details_futures.pop(req_id, None)
        return parse_todays_liquid_hours(details.liquidHours, date.today())

    async def get_contract_id(self, symbol: str) -> int:
        """Resolves a stock's IBKR conId -- needed to build a BAG combo's
        ComboLegs, which reference legs by conId rather than symbol. Same
        reqContractDetails pattern as get_todays_market_hours()."""
        req_id = self._new_req_id()
        future = self._loop.create_future()
        self._contract_details_futures[req_id] = future
        self.reqContractDetails(req_id, stock_contract(symbol))
        try:
            details = await asyncio.wait_for(future, timeout=15)
        finally:
            self._contract_details_futures.pop(req_id, None)
        return details.contract.conId

    async def get_initial_margin(self, symbol: str, action: str, quantity: float) -> float:
        """whatIf=True probe -- never actually submitted, just priced.

        Uses a MKT order rather than a STP. Confirmed live against the paper
        account: a STP with auxPrice=0 is not something IBKR will price, so
        the whatIf silently never comes back and the request just times out
        after 15s. MKT returns initMarginChange immediately, and margin
        doesn't depend on the order type anyway."""
        order_id = self._next_order_id
        self._next_order_id += 1
        order = market_order(action, quantity, account=self.account)
        order.whatIf = True
        future = self._loop.create_future()
        self._margin_futures[order_id] = future
        self.placeOrder(order_id, stock_contract(symbol), order)
        try:
            return await asyncio.wait_for(future, timeout=15)
        finally:
            self._margin_futures.pop(order_id, None)

    async def get_pair_initial_margin(
        self, symbol_a: str, action_a: str, qty_a: float,
        symbol_b: str, action_b: str, qty_b: float, units: int = 1,
    ) -> float:
        """Sum of two independent single-leg whatIf probes (get_initial_margin),
        returned as margin PER UNIT (divided by `units`).

        Rebuilt 2026-09-03 -- Combo/BAG orders removed per the client:
        "Delete Combo/Bag Order." Per the client 2026-08-08 (when this WAS
        a combo probe): "Your required calculation is the margin of the
        entire paired position, not simply margin(long) + margin(short)."
        That reasoning doesn't carry over here -- it was specifically about
        a COMBO's margin treatment not being a linear sum of its legs
        (confirmed live 2026-08-07, EME/PWR rejected because summed
        single-leg margins UNDERSTATED the real combo requirement, which
        got margin relief a combo specifically grants). With no combo at
        all, two independent orders really do each carry their own full
        Reg-T requirement -- summing them is now the accurate estimate,
        not the understating one.

        qty_a/qty_b are the REAL fractional quantities for this probe
        (fractional_leg_ratio's per-unit values times `units`), not a
        whole-number combo ratio -- IBKR's fractional share orders take a
        Decimal totalQuantity directly, no ratio/units split needed at
        the order level the way a combo required.

        `units` defaults to 1 (a cheap, fast estimate) for callers that
        just need a reference figure. But probing at 1 unit and linearly
        scaling that up to size a real position ASSUMES margin scales
        perfectly linearly with quantity, which real accounts don't
        always honor -- confirmed live 2026-08-11: over 70 entry
        rejections in a single day, every one needing 2-4% more margin
        than the 1-unit-scaled estimate predicted. Callers sizing a real
        entry should probe again at the actual candidate size (see
        main.py's entry flow) for an accurate per-unit figure at that
        real scale, not extrapolate from a single share."""
        margin_a, margin_b = await asyncio.gather(
            self.get_initial_margin(symbol_a, action_a, qty_a),
            self.get_initial_margin(symbol_b, action_b, qty_b),
        )
        return (margin_a + margin_b) / units

    async def refresh_executions_from_broker(self) -> None:
        """Backfills self._leg_fills from IBKR's own execution records via
        reqExecutions(), rather than trusting only the live execDetails()
        callbacks accumulated so far -- per the client 2026-08-08: fill
        tracking must be "robust to... reconnects". This bot connects once
        at session start and has no in-process auto-reconnect, so the real
        gap this covers is a brief network blip during the handshake that
        drops a live execDetails report without dropping the whole session;
        reqExecutions asks the broker directly for what actually happened,
        which is authoritative regardless of what this process's own memory
        currently holds. Filtered to today only (IBKR's default execution
        lookback), scoped to this client's own account.

        Safe to call defensively before reading back a fill -- it only ever
        adds/overwrites _leg_fills entries with the broker's own records, it
        doesn't invent anything and doesn't clear entries a live callback
        already correctly populated."""
        filt = ExecutionFilter()
        if self.account:
            filt.acctCode = self.account
        req_id = self._new_req_id()
        future = self._loop.create_future()
        self._exec_details_futures[req_id] = future
        self.reqExecutions(req_id, filt)
        try:
            await asyncio.wait_for(future, timeout=15)
        except asyncio.TimeoutError:
            logger.warning("refresh_executions_from_broker: reqExecutions timed out, "
                            "proceeding with whatever live execDetails already captured")
        finally:
            self._exec_details_futures.pop(req_id, None)

    # -- order placement -----------------------------------------------------

    async def place_single_stock_trade(
        self, symbol: str, action: str, quantity: int, wait_seconds: float = 2.0,
    ) -> tuple[int, float, float]:
        """Places ONE MKT order for the Expansion Leg -- per the client
        2026-09-03's pivot away from a hedged two-leg trade: "You don't
        need Hedge Ratio anymore." No Combo/BAG order, no second leg at
        all.

        This IS the Leg Risk Safeguard for the single-stock design,
        confirmed the same day: "Partial Fill Allowed. Leg Risk
        Safeguard: A 2-second timeout monitors fill status and cancels
        unfilled portions." Waits up to wait_seconds for the order to
        reach a terminal state; if it hasn't fully filled by then,
        cancels whatever's still working -- but keeps whatever quantity
        DID fill rather than treating a partial fill as a problem. With
        only one leg, there's no second leg to desynchronize from the
        way the old two-leg safeguard had to guard against.

        Returns (order_id, filled_qty, avg_price) -- filled_qty is 0.0 and
        avg_price is 0.0 if nothing filled at all. No-ops (logs only, no
        IBKR calls at all) unless IBKR_LIVE_TRADING_ENABLED=1."""
        order_id = self._next_order_id
        self._next_order_id += 1

        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would place {action} {quantity}x{symbol} (orderId={order_id})")
            return order_id, 0.0, 0.0

        order = market_order(action, quantity, account=self.account)
        order.transmit = True
        self._rejected_orders.pop(order_id, None)
        self.placeOrder(order_id, stock_contract(symbol), order)

        deadline = self._loop.time() + wait_seconds
        while self._loop.time() < deadline:
            if self._order_status.get(order_id) in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                break
            await asyncio.sleep(0.1)

        if self._order_status.get(order_id) != "Filled":
            # Cancels whatever's still working -- "Partial Fill Allowed"
            # means keeping whatever DID fill by now, not that the order
            # itself should keep working past the timeout. Waits for the
            # cancel to actually reach a terminal state before reading
            # fills below -- added 2026-09-14 per a third-party audit
            # finding verified directly: a bare cancelOrder() is
            # fire-and-forget, so a fill landing in the ~100ms+ window
            # before the exchange processes the cancel would never make
            # it into filled_qty, leaving those extra shares open at the
            # broker with no bracket protection sized for them.
            await self.cancel_orders_and_confirm([order_id], timeout_seconds=1.0)

        legs = self._leg_fills.get(order_id, {})
        if symbol not in legs:
            await self.refresh_executions_from_broker()
            legs = self._leg_fills.get(order_id, {})
        fill = legs.get(symbol)
        if fill is None or fill["qty"] <= 0:
            return order_id, 0.0, 0.0
        return order_id, fill["qty"], fill["avg_price"]

    async def place_single_stock_timed_exit(
        self, symbol: str, closing_action: str, quantity: int, timed_exit_activation: datetime,
    ) -> int:
        """Places the Timed Exit as a plain single-stock MKT order, held
        inactive at IBKR until timed_exit_activation via goodAfterTime --
        same mechanism combo_bracket_child_order's Timed Exit leg always
        used, just on one real stock instead of a closing combo.

        Take Profit and Hard Stop are no longer resting orders at all as
        of the 2026-09-03 "Expansion Leg" pivot (see
        risk_manager.take_profit_hard_stop_exit -- there's no combo left
        to rest a two-symbol spread-level order on), so this is the only
        bracket leg still placed at entry; no ocaGroup needed since
        there's nothing left in the group to race against.

        closing_action is the OPPOSITE of the entry action (SELL to close
        a BUY, BUY to close a SELL) -- the caller's job, same convention
        as place_pair_bracket's closing_action_a/b.

        Returns the order id. No-ops (logs only, no IBKR calls at all)
        unless IBKR_LIVE_TRADING_ENABLED=1."""
        order_id = self._next_order_id
        self._next_order_id += 1

        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would place TimedExit {closing_action} {quantity}x{symbol} "
                        f"@{timed_exit_activation} (orderId={order_id})")
            return order_id

        order = market_order(closing_action, quantity, account=self.account)
        order.goodAfterTime = timed_exit_activation.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"
        order.transmit = True
        self._rejected_orders.pop(order_id, None)
        self.placeOrder(order_id, stock_contract(symbol), order)
        return order_id

    async def modify_hard_stop(
        self, symbol: str, closing_action: str, quantity: int, new_price: float,
        order_id: int, oca_group: str,
    ) -> bool:
        """Re-submits the Hard Stop Loss leg at a new price, reusing the
        SAME order_id -- IBKR's own semantics treat placeOrder with an
        existing working order id as a MODIFICATION, not a new order, so
        the position is never left without a resting stop mid-retry.

        Per the client 2026-09-09: "After Entry, you move the Hard Stop
        Loss... No limit to continuous re-try. It is until it is Done. If
        it is not done, then the hard stop loss will not reset" -- the
        caller (run_check_cycle) is expected to call this again next cycle
        for as long as it returns False; the WIDE @ ENTRY price (see
        risk_manager.wide_hard_stop_price) keeps working the whole time,
        so there is no gap in protection while this retries.

        Rounded to the cent, same reason as place_single_stock_bracket's
        own rounding (confirmed live 2026-09-09: unrounded prices got a
        bracket leg rejected outright). Clamped to a $0.01 floor too,
        same reason as place_single_stock_bracket's own floor added
        2026-09-14 -- resting_hard_stop_price's distance
        (entry_divergence + trigger) can exceed a low-priced stock's own
        entry price just as easily as wide_hard_stop_price's can.

        Returns True once NOT rejected (accepted/working), False if
        rejected and still needing a retry next cycle. No-ops (returns
        True immediately, nothing sent to IBKR) unless
        IBKR_LIVE_TRADING_ENABLED=1."""
        new_price = max(round(new_price, 2), 0.01)
        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would modify Hard Stop {quantity}x{symbol} to "
                        f"{new_price:.4f} (orderId={order_id})")
            return True
        self._rejected_orders.pop(order_id, None)
        order = stock_bracket_child_order(closing_action, "STP", new_price, quantity, self.account, oca_group)
        self.placeOrder(order_id, stock_contract(symbol), order)
        rejection = await self.order_was_rejected(order_id)
        return rejection is None

    async def probe_bracket_initial_margin(
        self, symbol: str, entry_action: str, quantity: int, current_price: float,
        timed_exit_activation: datetime, wait_seconds: float = 1.5,
    ) -> "float | None":
        """Per the client 2026-09-10: "Use \$1, execute the order, full entry
        plus bracket submitted as one real set. So it rejects on 201 and we
        read the margin, then resize and submit for real."

        Submits the intended entry PLUS all four bracket legs as REAL orders
        (not whatIf), but with the entry as a non-fillable LMT -- a BUY at
        \$1, or a SELL far above the market -- so nothing can actually
        execute during the probe. IBKR still margin-checks the whole set;
        if the account can't carry it, error 201 comes back carrying the
        real figure ("...must exceed the new total Initial Margin of
        [23006.62 USD]"). Every probe order is cancelled before returning.

        Briefly simplified to entry-only 2026-09-15, per the client's own
        review that the bracket legs never carried real stop/target prices
        (always priced off far, deliberately non-triggering levels derived
        from current_price) so "shouldn't" change the 201 figure -- REVERTED
        the same day after a real live incident proved that assumption
        wrong: INTU entered clean (the entry-only probe found 29 shares fit
        easily against a $4,781 account), but the real bracket placed right
        after was REJECTED on its own 201 needing $6,920.14 total, nearly
        double the entry-alone figure, and the position had to be flattened
        immediately by the no-naked-position fail-safe. IBKR is evidently
        reserving margin for the bracket's own resting SELL orders on top
        of the position itself, not just the net position quantity x price x
        rate as this function's own comment used to claim -- the same class
        of "margin double-counts simultaneous working orders" behavior
        Critical Finding #1 (sequential candidate probing) already
        established ACROSS pairs, just not previously known to also apply
        WITHIN one pair's own bracket. Probing the full set again is what
        actually reflects the total IBKR will check.

        Includes a SIXTH probe leg too, standing in for the real bracket's
        Trailing Take Profit (see place_single_stock_bracket's own
        comment, added 2026-09-15) -- a plain STP at the same far,
        non-triggering price as the Hard Stop, NOT an actual TRAIL order.
        A real TRAIL order's own stop level starts right next to the
        current market price by design (current price minus the trail
        amount), so there's no way to place one "safely far" the way
        STP/MIT can be -- probing with the real order type would risk an
        actual fill in the ~2s window. Since today's incident already
        showed IBKR reserves margin per RESTING ORDER at the account
        level rather than by order type, a stand-in STP of the same
        quantity/side gives the same margin reservation without that
        risk.

        Returns that dollar figure (the TOTAL initial margin the full set
        needs at `quantity`), or None if no 201 came back -- meaning
        `quantity` already fits and can be submitted for real as-is.

        No-ops (returns None) unless IBKR_LIVE_TRADING_ENABLED=1."""
        if not self.live_trading_enabled:
            return None
        ids = [self._next_order_id + i for i in range(6)]
        self._next_order_id += 6
        entry_id, hs_id, tp_id, te_id, pnl_id, trail_standin_id = ids
        for oid in ids:
            self._rejected_orders.pop(oid, None)

        contract = stock_contract(symbol)
        oca_group = f"probe_{entry_id}"
        # Non-fillable: BUY at $1, SELL far above the market -- never marketable.
        probe_limit = 1.00 if entry_action == "BUY" else current_price * 2.0
        closing_action = "SELL" if entry_action == "BUY" else "BUY"
        # The bracket legs go in at prices that CANNOT trigger during the
        # probe -- deliberately un-triggerable prices give the same 201
        # figure with zero chance a leg fills into a naked position in the
        # ~2s window (STP put far the wrong way for its side, MIT the
        # same). The real bracket, with the real prices, is placed only
        # after the real entry fills.
        far_hi, far_lo = round(current_price * 5.0, 2), max(round(current_price * 0.01, 2), 0.01)
        if closing_action == "BUY":
            stp_probe, mit_probe = far_hi, far_lo
        else:
            stp_probe, mit_probe = far_lo, far_hi
        entry = limit_order(entry_action, quantity, probe_limit, self.account)
        self.placeOrder(entry_id, contract, entry)
        self.placeOrder(hs_id, contract, stock_bracket_child_order(
            closing_action, "STP", stp_probe, quantity, self.account, oca_group))
        self.placeOrder(tp_id, contract, stock_bracket_child_order(
            closing_action, "MIT", mit_probe, quantity, self.account, oca_group))
        self.placeOrder(te_id, contract, stock_bracket_child_order(
            closing_action, "MKT", 0.0, quantity, self.account, oca_group,
            good_after_time=timed_exit_activation))
        self.placeOrder(pnl_id, contract, stock_bracket_child_order(
            closing_action, "STP", stp_probe, quantity, self.account, oca_group))
        self.placeOrder(trail_standin_id, contract, stock_bracket_child_order(
            closing_action, "STP", stp_probe, quantity, self.account, oca_group))

        margin = None
        deadline = self._loop.time() + wait_seconds
        while self._loop.time() < deadline and margin is None:
            for oid in ids:
                reason = self._rejected_orders.get(oid)
                if reason and "201" in reason:
                    m = _INITIAL_MARGIN_RE.search(reason)
                    if m:
                        margin = float(m.group(1).replace(",", ""))
                        break
            if margin is None:
                await asyncio.sleep(0.2)

        # Cancel every probe order AND wait for the broker to acknowledge --
        # a bare cancelOrder() is fire-and-forget, so returning before the
        # cancels land leaves the 6 probe orders still pending at IBKR, and
        # the real entry submitted moments later gets margin-checked against
        # them too and rejects on its own 201. Poll each to a terminal
        # state, up to 2s.
        for oid in ids:
            self.cancelOrder(oid)
        # NOT "PendingCancel" -- that is only TWS acknowledging the cancel was
        # sent on; the order is still live at the exchange and its margin still
        # reserved until it reaches a real terminal state. The 2s ceiling below
        # is the backstop if it never gets there.
        _terminal = ("Cancelled", "ApiCancelled", "Inactive", "Filled")
        clear_deadline = self._loop.time() + 2.0
        while self._loop.time() < clear_deadline:
            if all(self._order_status.get(oid) in _terminal for oid in ids):
                break
            await asyncio.sleep(0.1)
        for oid in ids:
            self._rejected_orders.pop(oid, None)
        return margin

    async def place_single_stock_bracket(
        self, symbol: str, closing_action: str, quantity: int,
        hard_stop_price: float, take_profit_price: float, timed_exit_activation: datetime,
        pnl_stop_price: float | None = None, trailing_take_profit_amount: float | None = None,
    ) -> tuple[int, int, int, int | None, int | None]:
        """SUPERSEDES place_single_stock_timed_exit as of 2026-09-04, less
        than a day after it was built: the client restored Take Profit and
        Hard Stop Loss as real resting orders (How To Set Resting Take
        Profit.pdf) -- "Why can it not be a resting order... You simply
        close a single stock instead of two stocks." This places all
        three bracket legs (Hard Stop STP, Take Profit MIT, Timed Exit
        MKT) as one OCA group on the single traded stock, the direct
        single-stock analog of place_pair_bracket.

        hard_stop_price/take_profit_price come from
        risk_manager.resting_hard_stop_price/resting_take_profit_price.
        closing_action is the OPPOSITE of the entry action (SELL to close
        a BUY, BUY to close a SELL) -- the caller's job, same convention
        place_pair_bracket and place_single_stock_timed_exit both use.

        pnl_stop_price adds a FOURTH leg to the same OCA group -- the -5%
        PnL Stop Loss, per the client 2026-09-09 ("It is Ibkr PnL. 5% of
        EWLV"), priced by risk_manager.pnl_stop_loss_price. Optional so
        existing callers and tests that place the original three-leg
        bracket keep working unchanged; None places three legs exactly as
        before.

        trailing_take_profit_amount adds a FIFTH leg -- IBKR's own native
        TRAIL order type (auxPrice = the trail amount, not a price level),
        per the client 2026-09-15: "Trailing Take Profit = $1... From the
        time you enter INTU, if INTU rises by $1, stop loss will trail by
        $1, protecting that profit." Confirmed as running ALONGSIDE the
        other four legs, not replacing any of them -- whichever of the
        five fills first still cancels the rest via the same OCA group.

        Returns (hard_stop_order_id, tp_order_id, timed_exit_order_id,
        pnl_stop_order_id, trail_order_id) -- either of the last two is
        None when its corresponding price/amount wasn't given. No-ops
        (logs only, no IBKR calls at all) unless
        IBKR_LIVE_TRADING_ENABLED=1."""
        hard_stop_order_id = self._next_order_id
        self._next_order_id += 1
        tp_order_id = self._next_order_id
        self._next_order_id += 1
        timed_exit_order_id = self._next_order_id
        self._next_order_id += 1
        pnl_stop_order_id = None
        if pnl_stop_price is not None:
            pnl_stop_order_id = self._next_order_id
            self._next_order_id += 1
        trail_order_id = None
        if trailing_take_profit_amount is not None:
            trail_order_id = self._next_order_id
            self._next_order_id += 1

        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would place bracket {quantity}x{symbol}: "
                        f"TP={take_profit_price:.4f} (orderId={tp_order_id}) "
                        f"HardStop={hard_stop_price:.4f} (orderId={hard_stop_order_id}) "
                        f"TimedExit@{timed_exit_activation} (orderId={timed_exit_order_id})"
                        + (f" PnLStop={pnl_stop_price:.4f} (orderId={pnl_stop_order_id})"
                           if pnl_stop_price is not None else "")
                        + (f" TrailingTP=${trailing_take_profit_amount:.2f} (orderId={trail_order_id})"
                           if trailing_take_profit_amount is not None else ""))
            return hard_stop_order_id, tp_order_id, timed_exit_order_id, pnl_stop_order_id, trail_order_id

        contract = stock_contract(symbol)
        # Rounded to the cent before anything is sent -- confirmed live
        # 2026-09-09: an entry on ADSK filled and then had ALL THREE of its
        # price-bearing bracket legs rejected with error 110 ("price does not
        # conform to the minimum price variation"), because
        # resting_hard_stop_price/resting_take_profit_price/pnl_stop_loss_price
        # all return raw arithmetic with many decimals. The bot's own
        # fail-safe flattened the unprotected position correctly, but the
        # trade was lost. bracket_order_prices has rounded for this exact
        # reason since 2026-08-20; the single-stock path never did. Rounded
        # here, at the single point where every caller's prices reach IBKR,
        # rather than in each formula.
        #
        # Clamped to a $0.01 floor too, added 2026-09-14 per a third-party
        # audit finding verified directly: wide_hard_stop_price's distance
        # is trigger*1.87, which on a $5 trigger is $9.35 -- any stock
        # entered below that (real, ordinary S&P 500 names trade under
        # $9.35) produces a negative hard stop price (confirmed:
        # wide_hard_stop_price('BUY', 8.50, 5.00) == -0.85). round() alone
        # never fixes a negative number, and IBKR rejects any
        # non-positive auxPrice outright -- exactly the same class of
        # instant-bracket-rejection-then-emergency-flatten this ADSK
        # rounding fix was for, just from a different cause. resting_take_profit_price
        # (short side) and pnl_stop_loss_price (small quantity) can hit
        # the same failure mode, so all three are clamped here.
        hard_stop_price = max(round(hard_stop_price, 2), 0.01)
        take_profit_price = max(round(take_profit_price, 2), 0.01)
        if pnl_stop_price is not None:
            pnl_stop_price = max(round(pnl_stop_price, 2), 0.01)
        oca_group = f"single_{tp_order_id}"
        hard_stop_order = stock_bracket_child_order(
            closing_action, "STP", hard_stop_price, quantity, self.account, oca_group)
        tp_order = stock_bracket_child_order(
            closing_action, "MIT", take_profit_price, quantity, self.account, oca_group)
        timed_exit_order = stock_bracket_child_order(
            closing_action, "MKT", 0.0, quantity, self.account, oca_group,
            good_after_time=timed_exit_activation)
        self.placeOrder(hard_stop_order_id, contract, hard_stop_order)
        self.placeOrder(tp_order_id, contract, tp_order)
        self.placeOrder(timed_exit_order_id, contract, timed_exit_order)
        if pnl_stop_order_id is not None:
            # Same STP type and same OCA group as the Hard Stop -- whichever
            # of the four fills first cancels the other three at the broker.
            pnl_stop_order = stock_bracket_child_order(
                closing_action, "STP", pnl_stop_price, quantity, self.account, oca_group)
            self.placeOrder(pnl_stop_order_id, contract, pnl_stop_order)
        if trail_order_id is not None:
            # TRAIL, not STP/MIT -- auxPrice here is the trail AMOUNT (a
            # dollar distance IBKR itself ratchets as price moves
            # favorably), not a fixed price level like every other leg in
            # this bracket. Same OCA group, same "whichever fills first
            # cancels the rest" mechanics.
            trail_order = stock_bracket_child_order(
                closing_action, "TRAIL", trailing_take_profit_amount, quantity, self.account, oca_group)
            self.placeOrder(trail_order_id, contract, trail_order)
        return hard_stop_order_id, tp_order_id, timed_exit_order_id, pnl_stop_order_id, trail_order_id

    async def place_excursion_bracket(
        self, symbol: str, closing_action: str, quantity: int,
        pnl_stop_price: float, trailing_take_profit_amount: float,
        timed_exit_activation: datetime,
    ) -> tuple[int, int, int]:
        """Places the 3-leg OCA bracket for temporary single-stock excursion mode:
        1. Trailing Take Profit: Native IBKR TRAIL order at trailing_take_profit_amount ($1.00)
        2. PnL Stop Loss: STP order at pnl_stop_price (-2% PnL)
        3. Timed Exit: MKT order at timed_exit_activation (20 min max hold)
        All 3 legs share a single OCA group with ocaType = 1 (Cancel with Block).
        Returns (trail_order_id, pnl_stop_order_id, timed_exit_order_id).
        """
        trail_order_id = self._next_order_id
        self._next_order_id += 1
        pnl_stop_order_id = self._next_order_id
        self._next_order_id += 1
        timed_exit_order_id = self._next_order_id
        self._next_order_id += 1

        pnl_stop_price = max(round(pnl_stop_price, 2), 0.01)
        trailing_take_profit_amount = round(trailing_take_profit_amount, 2)

        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would place excursion bracket {quantity}x{symbol}: "
                        f"TrailingTP=${trailing_take_profit_amount:.2f} (orderId={trail_order_id}) "
                        f"PnLStop={pnl_stop_price:.2f} (orderId={pnl_stop_order_id}) "
                        f"TimedExit@{timed_exit_activation} (orderId={timed_exit_order_id})")
            return trail_order_id, pnl_stop_order_id, timed_exit_order_id

        contract = stock_contract(symbol)
        oca_group = f"single_{trail_order_id}"
        trail_order = stock_bracket_child_order(
            closing_action, "TRAIL", trailing_take_profit_amount, quantity, self.account, oca_group)
        pnl_stop_order = stock_bracket_child_order(
            closing_action, "STP", pnl_stop_price, quantity, self.account, oca_group)
        timed_exit_order = stock_bracket_child_order(
            closing_action, "MKT", 0.0, quantity, self.account, oca_group,
            good_after_time=timed_exit_activation)

        self.placeOrder(trail_order_id, contract, trail_order)
        self.placeOrder(pnl_stop_order_id, contract, pnl_stop_order)
        self.placeOrder(timed_exit_order_id, contract, timed_exit_order)

        return trail_order_id, pnl_stop_order_id, timed_exit_order_id

    async def probe_excursion_bracket_initial_margin(
        self, symbol: str, entry_action: str, quantity: int, current_price: float,
        timed_exit_activation: datetime, wait_seconds: float = 1.5,
    ) -> "float | None":
        """Probes margin for the 3-leg excursion bracket:
        - Entry order (non-fillable limit: BUY at $1 or SELL at 2x market)
        - Standin STP for Trailing TP
        - STP for -2% PnL Stop
        - MKT for 20-min Timed Exit
        All in one probe OCA group. Returns required margin if 201 received, else None.
        """
        if not self.live_trading_enabled:
            return None
        ids = [self._next_order_id + i for i in range(4)]
        self._next_order_id += 4
        entry_id, trail_standin_id, pnl_id, te_id = ids
        for oid in ids:
            self._rejected_orders.pop(oid, None)

        contract = stock_contract(symbol)
        oca_group = f"probe_{entry_id}"
        probe_limit = 1.00 if entry_action == "BUY" else current_price * 2.0
        closing_action = "SELL" if entry_action == "BUY" else "BUY"
        far_hi, far_lo = round(current_price * 5.0, 2), max(round(current_price * 0.01, 2), 0.01)
        stp_probe = far_hi if closing_action == "BUY" else far_lo

        entry = limit_order(entry_action, quantity, probe_limit, self.account)
        self.placeOrder(entry_id, contract, entry)
        self.placeOrder(trail_standin_id, contract, stock_bracket_child_order(
            closing_action, "STP", stp_probe, quantity, self.account, oca_group))
        self.placeOrder(pnl_id, contract, stock_bracket_child_order(
            closing_action, "STP", stp_probe, quantity, self.account, oca_group))
        self.placeOrder(te_id, contract, stock_bracket_child_order(
            closing_action, "MKT", 0.0, quantity, self.account, oca_group,
            good_after_time=timed_exit_activation))

        margin = None
        deadline = self._loop.time() + wait_seconds
        while self._loop.time() < deadline and margin is None:
            for oid in ids:
                reason = self._rejected_orders.get(oid)
                if reason and "201" in reason:
                    m = _INITIAL_MARGIN_RE.search(reason)
                    if m:
                        margin = float(m.group(1).replace(",", ""))
                        break
            if margin is None:
                await asyncio.sleep(0.2)

        for oid in ids:
            self.cancelOrder(oid)
        _terminal = ("Cancelled", "ApiCancelled", "Inactive", "Filled")
        clear_deadline = self._loop.time() + 2.0
        while self._loop.time() < clear_deadline:
            if all(self._order_status.get(oid) in _terminal for oid in ids):
                break
            await asyncio.sleep(0.1)
        for oid in ids:
            self._rejected_orders.pop(oid, None)
        return margin

    async def place_pair_trade(
        self,
        symbol_a: str, qty_a: float, action_a: str,
        symbol_b: str, qty_b: float, action_b: str,
    ) -> tuple[int, int]:
        """Places both legs as two independent MKT orders (no Combo/BAG --
        see this module's own docstring). Take Profit and Stop Loss are
        placed separately, AFTER the actual fill is confirmed -- see
        place_pair_bracket -- rather than bracketed to this order at
        submission time, since their real size has to match what actually
        filled, not the originally requested quantity.

        Both orders are placed back to back, without awaiting either
        fill in between -- placing them sequentially with a wait would
        only widen the window one leg could move against the other
        before the second is even submitted. get_pair_actual_fill (the
        Leg Risk Safeguard) is what handles the two orders filling
        independently after this.

        Returns (order_id_a, order_id_b). No-ops (logs only, no IBKR
        calls at all) unless IBKR_LIVE_TRADING_ENABLED=1, so this is safe
        to wire up before a live account is confirmed."""
        order_id_a = self._next_order_id
        self._next_order_id += 1
        order_id_b = self._next_order_id
        self._next_order_id += 1

        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would place two orders: "
                        f"{action_a} {qty_a}x{symbol_a} (orderId={order_id_a}) / "
                        f"{action_b} {qty_b}x{symbol_b} (orderId={order_id_b})")
            return order_id_a, order_id_b

        contract_a = stock_contract(symbol_a)
        contract_b = stock_contract(symbol_b)
        order_a = market_order(action_a, qty_a, account=self.account)
        order_a.transmit = True
        order_b = market_order(action_b, qty_b, account=self.account)
        order_b.transmit = True

        self._rejected_orders.pop(order_id_a, None)
        self._rejected_orders.pop(order_id_b, None)
        self.placeOrder(order_id_a, contract_a, order_a)
        self.placeOrder(order_id_b, contract_b, order_b)
        return order_id_a, order_id_b

    async def place_pair_bracket(
        self,
        symbol_a: str, action_a: str, symbol_b: str, action_b: str,
        qty_a: int, qty_b: int, hard_stop_price: float, take_profit_price: float,
        timed_exit_activation: datetime,
    ) -> tuple[int, int, int]:
        """Places the closing Take Profit (MIT), Hard Stop (STP), and Timed
        Exit (MKT) combo orders as one OCA group, sized to the position's
        REAL filled quantity (qty_a/qty_b here must be the actual fill from
        get_pair_actual_fill, not the originally requested quantity -- a
        partial fill would otherwise leave the bracket trying to close
        more shares than the position actually holds).

        Restructured 2026-08-24, per the client: the native "0 Divergence"
        Stop Loss moved out of the bracket into bot-monitored logic
        (risk_manager.stop_loss_double_touch_exit), replaced here by a
        Hard Stop backstop (Trigger x 2, opposite side from Take Profit)
        and a Timed Exit that fires automatically at timed_exit_activation
        (entry_time + Maximum Hold Time -- the caller computes this, this
        function only formats and places it) instead of Maximum Hold Time
        being a script-checked exit.

        hard_stop_price/take_profit_price come from
        risk_manager.bracket_order_prices, already in the closing combo's
        own price convention. timed_exit_activation must be ET-aware --
        see combo_bracket_child_order's own goodAfterTime formatting.

        Returns (hard_stop_order_id, tp_order_id, timed_exit_order_id).
        No-ops (logs only, no IBKR calls at all) unless
        IBKR_LIVE_TRADING_ENABLED=1."""
        ratio_a, ratio_b, units = combo_ratio_and_units(qty_a, qty_b)
        hard_stop_order_id = self._next_order_id
        self._next_order_id += 1
        tp_order_id = self._next_order_id
        self._next_order_id += 1
        timed_exit_order_id = self._next_order_id
        self._next_order_id += 1

        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would place bracket {units} units: "
                        f"TP={take_profit_price:.4f} (orderId={tp_order_id}) "
                        f"HardStop={hard_stop_price:.4f} (orderId={hard_stop_order_id}) "
                        f"TimedExit@{timed_exit_activation} (orderId={timed_exit_order_id})")
            return hard_stop_order_id, tp_order_id, timed_exit_order_id

        con_id_a = await self.get_contract_id(symbol_a)
        con_id_b = await self.get_contract_id(symbol_b)
        closing_action_a = "SELL" if action_a == "BUY" else "BUY"
        closing_action_b = "SELL" if action_b == "BUY" else "BUY"
        closing_contract = combo_contract(
            symbol_a, con_id_a, closing_action_a, ratio_a, con_id_b, closing_action_b, ratio_b)

        oca_group = f"pair_{tp_order_id}"
        hard_stop_order = combo_bracket_child_order(
            "STP", hard_stop_price, units, self.account, oca_group)
        tp_order = combo_bracket_child_order(
            "MIT", take_profit_price, units, self.account, oca_group)
        timed_exit_order = combo_bracket_child_order(
            "MKT", 0.0, units, self.account, oca_group, good_after_time=timed_exit_activation)
        self.placeOrder(hard_stop_order_id, closing_contract, hard_stop_order)
        self.placeOrder(tp_order_id, closing_contract, tp_order)
        self.placeOrder(timed_exit_order_id, closing_contract, timed_exit_order)
        return hard_stop_order_id, tp_order_id, timed_exit_order_id

    async def get_pair_actual_fill(
        self, order_id_a: int, symbol_a: str, order_id_b: int, symbol_b: str,
        wait_seconds: float = 2.0,
    ) -> tuple[float, float, float, float] | None:
        """The Leg Risk Safeguard. The REAL filled quantity and average
        price for each leg, reconciled from IBKR's own per-leg execution
        reports (execDetails) against the two independent orders placed
        by place_pair_trade -- not inferred from any single combined
        confirmation, since there is no combo order tying the two
        together anymore.

        Rebuilt 2026-09-03 -- Combo/BAG orders removed, two independent
        orders (order_id_a, order_id_b) replace the single entry_order_id
        this used to reconcile against. Per the client the same day: "A
        2-second timeout monitors dual-leg fill status. If an asymmetric
        fill occurs, the orphan position is immediately flattened at
        market." wait_seconds default lowered from 5.0 to 2.0
        accordingly -- the underlying wait-then-reconcile mechanism
        already existed (per the client 2026-08-08, "My VPS latency is
        only 4 milliseconds. Orders will fill inside 1 second"), because
        even the Combo order was placed NonGuaranteed and could leg risk
        the same way; this is an adaptation of that existing logic to two
        order ids, not new from nothing. The caller (main.py's entry
        flow) is what actually does the flattening on a None result here,
        via flatten_symbols() -- unchanged from before this rebuild.

        Waits for BOTH orders to reach a terminal state (Filled,
        Cancelled, ApiCancelled, Inactive) or wait_seconds to elapse,
        whichever comes first. If nothing has filled by then, refreshes
        from the broker's own execution records once (covers a dropped
        execDetails callback) before giving up.

        Only breaks early once BOTH orders' own status reaches a genuine
        terminal state -- NOT merely "both legs have produced at least one
        execDetails report" (removed 2026-08-24, same reasoning applies
        with two orders now). execution.cumQty is IBKR's own running
        CUMULATIVE total for that order, so a large or less-liquid order
        that fills across several partial reports keeps growing after the
        first one arrives; reading _leg_fills the instant a report
        arrives -- which could be the first of several -- risked
        capturing a still-partial qty and sizing the Take Profit/Stop
        Loss bracket for only a fraction of the real position, leaving
        the rest unprotected.

        Returns (qty_a, avg_price_a, qty_b, avg_price_b), or None if
        either leg never received a real fill -- the caller must not
        construct a Position from a None result, and must check for (and
        flatten) whichever leg DID fill, since a None here can mean one
        real, unhedged position is sitting at the broker."""
        deadline = self._loop.time() + wait_seconds
        while self._loop.time() < deadline:
            status_a = self._order_status.get(order_id_a)
            status_b = self._order_status.get(order_id_b)
            terminal = ("Filled", "Cancelled", "ApiCancelled", "Inactive")
            if status_a in terminal and status_b in terminal:
                break
            await asyncio.sleep(0.1)

        legs_a = self._leg_fills.get(order_id_a, {})
        legs_b = self._leg_fills.get(order_id_b, {})
        if symbol_a not in legs_a or symbol_b not in legs_b:
            # Nothing captured live -- one more check against the broker's
            # own execution records before concluding this really never filled.
            await self.refresh_executions_from_broker()
            legs_a = self._leg_fills.get(order_id_a, {})
            legs_b = self._leg_fills.get(order_id_b, {})

        fill_a = legs_a.get(symbol_a)
        fill_b = legs_b.get(symbol_b)
        if fill_a is None or fill_b is None or fill_a["qty"] <= 0 or fill_b["qty"] <= 0:
            return None
        return fill_a["qty"], fill_a["avg_price"], fill_b["qty"], fill_b["avg_price"]

    def cancel_order_id(self, order_id: int | None) -> None:
        """Cancels a single working order -- used to clear a still-working
        entry order that got rejected or never produced a real two-leg
        fill, or a still-resting Take Profit / Stop Loss bracket child when
        the position exits through a different path (beta breakdown, the
        max hold time, the 3:59pm flatten) so it doesn't linger and fire
        later against a position that no longer exists.

        None is a genuine, expected input, not a caller bug -- callers that
        conditionally have an order id to cancel can pass it through
        unconditionally rather than checking first. A no-op here keeps
        those call sites simple."""
        if order_id is None:
            return
        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would cancel orderId={order_id}")
            return
        self.cancelOrder(order_id)

    async def cancel_orders_and_confirm(
        self, order_ids: "list[int | None]", timeout_seconds: float = 1.5,
    ) -> None:
        """Cancels every given order id and WAITS for each one to reach a
        real terminal state before returning -- added 2026-09-14 per a
        third-party audit finding verified directly: cancel_order_id (and
        the raw cancelOrder() it wraps) is fire-and-forget, so a caller
        that acts immediately after calling it can race the broker's own
        processing of the cancel. Two real incidents this pattern already
        caused elsewhere: probe_bracket_initial_margin's own real entry
        getting margin-checked against its own still-pending probe
        orders, and this same risk at every "cancel the other bracket
        legs, then immediately flatten" emergency handler in main.py --
        IBKR can reject the flatten MKT order ("shares already allocated
        to a working order") if a leg hasn't actually cleared yet,
        leaving the position naked until EOD.

        NOT "PendingCancel" -- that's only TWS acknowledging the cancel
        was sent, the order is still live at the exchange and its
        margin/shares still reserved until a real terminal state. None
        entries are skipped (same convention as cancel_order_id itself).
        Gives up after timeout_seconds regardless -- a backstop, not a
        guarantee, same ceiling probe_bracket_initial_margin's own
        cancel-wait already uses."""
        ids = [oid for oid in order_ids if oid is not None]
        if not ids:
            return
        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] would cancel and confirm orderIds={ids}")
            return
        for oid in ids:
            self.cancelOrder(oid)
        _terminal = ("Cancelled", "ApiCancelled", "Inactive", "Filled")
        deadline = self._loop.time() + timeout_seconds
        while self._loop.time() < deadline:
            if all(self._order_status.get(oid) in _terminal for oid in ids):
                break
            await asyncio.sleep(0.1)

    def entry_order_abandoned(self, order_id: int | None) -> bool:
        """True if the entry order reached a terminal state (cancelled,
        discarded, inactive) without ever filling. order_was_rejected only
        watches a 3-second window right after placement -- a LATE failure
        (an external cancel, IBKR discarding a stale working order, anything
        that happens after that window closes) would otherwise leave
        ctx.position pointing at a position that was never actually opened
        at the broker, computing phantom P&L against it indefinitely.
        Confirmed live 2026-08-07: a global cancel issued for unrelated
        orders swept up a still-pending entry 5 minutes after placement,
        well past the rejection check's window."""
        if order_id is None:
            return False
        return self._order_status.get(order_id) in ("Cancelled", "ApiCancelled", "Inactive")

    def bracket_order_filled(self, order_id: int | None) -> float | None:
        """Fill price if this bracket order (take profit or stop loss) has
        filled at the broker, else None. The native bracket can fire at any
        moment, independent of the bot's own 60-second check cycle, so this
        is polled every cycle for every open position to catch a fill the
        script itself never decided on -- see main.py's run_check_cycle."""
        if order_id is None:
            return None
        if self._order_status.get(order_id) != "Filled":
            return None
        return self._order_fill_price.get(order_id)

    def bracket_order_abandoned(self, order_id: int | None) -> bool:
        """True if this bracket child (take profit or stop loss) reached a
        terminal state without ever filling -- same status check as
        entry_order_abandoned, applied to a bracket instead of the entry.

        Per the independent audit's Finding #8, verified real 2026-08-24:
        the bracket-placement flow already flattens immediately if a
        bracket is REJECTED at submission (see main.py's entry flow), but
        nothing was watching for one that was accepted and later went
        Cancelled/Inactive on its own -- a connectivity blip, an IBKR-side
        risk check, or a stale-order sweep could silently kill a working
        stop loss or take profit sometime after entry, leaving the
        position with no real protection until the 3:59pm flatten or
        Maximum Hold Time eventually caught it. Polled every cycle
        alongside bracket_order_filled -- see main.py's run_check_cycle."""
        if order_id is None:
            return False
        return self._order_status.get(order_id) in ("Cancelled", "ApiCancelled", "Inactive")

    async def order_was_rejected(self, order_id: int, wait_seconds: float = 3.0) -> str | None:
        """The rejection reason if IBKR refused this order, else None.

        placeOrder is fire-and-forget and the rejection lands on error()
        milliseconds to seconds later, so the caller has to give it a moment
        before deciding whether a position actually exists. Three seconds is
        well clear of the ~0.5s observed for a margin rejection while still
        being short against a one-minute check cycle.

        Also treats orderStatus reaching "Inactive" as a rejection -- per
        the independent audit, confirmed real: IBKR sometimes rejects an
        order by only moving its status to Inactive, with no matching entry
        in ORDER_REJECTED_CODES at all (no error() callback this class
        recognizes as a rejection). Checking _rejected_orders alone missed
        that case entirely, reporting a dead order as still working."""
        deadline = self._loop.time() + wait_seconds
        while self._loop.time() < deadline:
            if order_id in self._rejected_orders:
                return self._rejected_orders[order_id]
            if self._order_status.get(order_id) == "Inactive":
                return "OrderStatus: Inactive"
            await asyncio.sleep(0.25)
        if order_id in self._rejected_orders:
            return self._rejected_orders[order_id]
        if self._order_status.get(order_id) == "Inactive":
            return "OrderStatus: Inactive"
        return None

    async def _current_positions(self) -> list[tuple[str, float]]:
        self._positions = []
        self._positions_future = self._loop.create_future()
        self.reqPositions()
        try:
            return await asyncio.wait_for(self._positions_future, timeout=15)
        finally:
            self.cancelPositions()

    def _close_out(self, positions: list[tuple[str, float]]) -> list[int]:
        """Returns the order ids actually placed (empty in dry-run) so the
        caller can check order_was_rejected on each -- a flattening MKT
        order is not immune to rejection (a halted contract, no shares
        available to borrow, an account restriction), and every caller of
        flatten_symbols/flatten_all used to assume it always worked and
        clear its own tracked position immediately after, with nothing
        checking whether the position was actually still open at the
        broker.

        Short (BUY-to-cover) legs are submitted before long (SELL) legs --
        independent audit finding, confirmed real: this bot sizes at 98%
        margin utilization (position_sizing.py), so a pair's long and short
        legs are relying on each other's Reg-T offset for most of their
        margin headroom. If the long leg's SELL fills first, the short leg
        is briefly held alone -- a naked short needs 150%+ of notional
        (project's own confirmed prior incident), which this account has no
        spare margin for, and IBKR can reject the second leg's close. Buying
        back the short first releases margin instead of consuming more of
        it, so the SELL that follows never hits that shortfall."""
        positions = sorted(positions, key=lambda p: p[1] > 0)  # short (qty<0) legs first
        if not self.live_trading_enabled:
            for symbol, qty in positions:
                side = "SELL" if qty > 0 else "BUY"
                logger.info(f"[DRY RUN] would {side} {abs(qty)} {symbol} MKT to flatten")
            return []
        order_ids = []
        for symbol, qty in positions:
            side = "SELL" if qty > 0 else "BUY"
            order_id = self._next_order_id
            self._next_order_id += 1
            self.placeOrder(order_id, stock_contract(symbol),
                             market_order(side, abs(qty), self.account))
            order_ids.append(order_id)
        return order_ids

    async def flatten_all(self) -> list[int]:
        """Cancels ALL open orders and closes EVERY open position -- correct
        only for the 3:59pm hard exit, where the whole book should go flat.
        Use flatten_symbols() for a single pair's exit. Returns the closing
        order ids placed -- see _close_out."""
        self.reqGlobalCancel()
        return self._close_out(await self._current_positions())

    async def flatten_symbols(self, symbols: set[str]) -> list[int]:
        """Closes only the given symbols' positions, leaving any other pair's
        position untouched. A single pair exiting used to call flatten_all(),
        which silently closed every other open pair at the broker while the
        bot's own state still believed those pairs were open. Returns the
        closing order ids placed (empty if there was nothing to flatten) --
        see _close_out."""
        targets = [(s, q) for s, q in await self._current_positions() if s in symbols]
        if not targets:
            logger.info(f"Nothing to flatten for {sorted(symbols)}")
            return []
        return self._close_out(targets)
