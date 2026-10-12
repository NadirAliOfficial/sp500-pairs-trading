"""The entry/exit state machine, per the client's spec. Pure logic, no I/O --
risk_manager consumes live spread/equilibrium numbers (from spread_engine +
live_data) and the news blacklist (news_blacklist.ExcludedSymbols) and
decides what to do; ibkr_client.py is what actually places the orders.

The DAILY loss cap (record_exit) is still a fixed dollar limit: 2% of
EquityWithLoanValue measured at 09:25:00 ET, not a percentage of any one
trade's own notional. The PER-TRADE stop loss is different -- see
check_exit's own docstring for "Stop Loss = 0 Divergence", per the client
2026-08-19.

Entry/exit logic replaced wholesale 2026-08-18, per the client: "I want to
change the entry... From The Divergence Watch List, select the closest to
trigger and enter. Entry minimum points to trigger <= $0.40 to go. Take
Profit at Trigger. Stop Loss at 2% of EWLV. Take Profit is MKT. Bot is
taking profit on divergence expansion. One trade per day. No re-entry."
This is the OPPOSITE of every mean-reversion mechanism this module used to
implement (Turnback pullback timing, Dynamic Stop Loss, Half-Life Stop
Loss, same-pair-lock re-entry, profit-on-reversion-to-equilibrium) -- the
bot enters BEFORE the divergence reaches its trigger and profits when it
EXPANDS further past that trigger, not when it reverts.

Same-day, 2026-08-19: Stop Loss replaced with "0 Divergence", then
clarified as "0 relative to equilibrium at entry. 100% mean-reversion" --
both Take Profit and Stop Loss are prices fixed the moment a position
opens (entry_equilibrium, not a live-recomputed one), which is what makes
a real resting bracket order at IBKR possible again (see
bracket_order_prices / ibkr_client.place_pair_bracket), unlike the
several-hours-earlier version of this same day where both depended on a
live-drifting equilibrium and had to be script-fired market orders. Also
added that day: the entry-distance window became a $0.01-$0.40 band
instead of a single ceiling, and a 140-minute Maximum Hold Time exit (the
third way the day's one trade can end, alongside Take Profit and Stop
Loss -- confirmed explicitly there is NO same-day re-entry regardless of
which of the three closes it).
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from phase2.position_sizing import minimum_gap_at_entry

logger = logging.getLogger("pairs_bot.risk_manager")

MIN_ENTRY_DIVERGENCE = 1.50  # Per the client 2026-09-11: "Please use 1.5" -- was $1 briefly, $2 before that (raised 2026-08-18)
# Maximum Stock Price applies PER STOCK (each leg checked independently),
# not to the pair as a whole -- "Maximum Stock Price of $700 is for a
# stock. Not a pair." (2026-08-17)
MAX_STOCK_PRICE = 700.0
MAX_ENTRY_DIVERGENCE = 5.0  # Raised from $3.5 per the client 2026-09-04: "Increase Maximum Divergence to 5"
# Entry timing, per the client 2026-08-18: "From The Divergence Watch List,
# select the closest to trigger and enter." -- enter BEFORE the divergence
# reaches its own trigger level, only once it's within this distance of it,
# not after crossing it (a distance of 0 or less means it has already
# crossed, which is now too late to enter -- Take Profit is AT the trigger,
# so entering past it would mean the profit target is already behind the
# entry). Per the client 2026-08-19: "Entry Window: Pair must be within
# $0.01 to $0.40 points away from trigger... There is a minimum and a
# maximum" -- this is a band, not a single ceiling. The earlier same-day
# "$0.01 to go" instruction was the floor of this band, not the whole rule.
# Widened to $0.40-$0.60 per the client 2026-08-21's Entry Conditions PDF,
# then to $0.40-$0.80 (2026-08-24: "0.60 is too tight"), then replaced
# entirely with a percentage-of-trigger band the same day: "Pair must be
# within 13%-25% of Trigger points to get to Trigger" -- confirmed by the
# client as measuring distance_to_trigger against the trigger level itself,
# not a fixed dollar range. A fixed-dollar band was a poor fit across pairs
# whose trigger levels differ widely (a $6 max-divergence trigger and a
# $2.50 one need very different entry windows in absolute terms).
# Upper bound widened from 25% to 49.9% per the client 2026-09-09: "Entry
# Window. Pair must be between 13%-49.9% of Trigger points to go, to get to
# Trigger", confirmed after a first version that read ">49.9% remaining"
# ("Typo"). The floor stays 13%. His point 2 the same day ("Maximum
# Divergence dropped below 50% of Trigger, inside lookback period") is this
# same band read from the other side, not a separate gate -- entering while
# 13%-49.9% of the trigger remains to go IS the divergence having pulled
# back past the 50% mark, exactly as the old 25%/75% pair worked.
ENTRY_TRIGGER_DISTANCE_MIN_PCT = 0.13
ENTRY_TRIGGER_DISTANCE_MAX_PCT = 0.499
# v7 added a 10am floor, v8 walked it back as a "typo" (trading starts 9:30),
# then the client explicitly reinstated it 2026-07-27 20:28 ("Yes. I have
# decided to use 10am ... MARKET ENTRY : 10:00am ET - 3:30pm ET"). Reverted
# back to 9:30am 2026-08-27, per the client: "Market Entry: 9:30am, instead
# of 10am." Moved to 9:31am 2026-09-12 (Checklist_120926.pdf: "Market Entry
# : 9:31-10:30am ET"), then to 9:32am 2026-09-13: "Market Entry: 9:32-10:30am
# ET. 9:32, not 9:31."
EARLIEST_ENTRY_TIME = time(9, 32)
# Latest entry cut narrowed 2026-08-20, per the client: "Instead of 3:30pm
# for latest entry, use 11:00am ET" then corrected same day to "use 10:30am
# ET" -- the second message supersedes the first. Entry window is now only
# 10:00-10:30am ET (30 minutes), not the whole trading day.
LATEST_ENTRY_TIME = time(10, 30)  # 10:30am ET
EOD_FLATTEN_TIME = time(15, 59)  # 3:59pm ET
EARLY_CLOSE_THRESHOLD = time(16, 0)  # v7: "No trading if today is an early close day: closeTime < 16:00pm ET"
# Per the client 2026-08-14: "Maximum Loss for the Day = 2% of Equity With
# Loan Value at 9:25am ET" -- the DAILY halt cap (record_exit), unrelated to
# the per-trade stop loss below.
STOP_LOSS_PCT = 0.02
# Per-trade stop loss, replaced 2026-08-19: "Instead of 2% EWLV Stop Loss,
# use Stop Loss = 0 Divergence." Fires once the live divergence has fully
# reverted back to equilibrium -- the opposite of what this trade bet on
# (the divergence expanding to the trigger), so a full round-trip back to 0
# is the clearest possible sign the thesis failed. Exact 0.0 is unrealistic
# with continuous price data, so this tolerance treats anything within a
# cent of it as "reached", same convention the old mean-reversion
# take-profit-to-equilibrium check used.
STOP_LOSS_DIVERGENCE_TOLERANCE = 0.01
# Hard Stop, added 2026-08-24 as a broker-side backstop once the native
# "0 Divergence" Stop Loss moved out of the bracket into bot-monitored
# logic (see risk_manager.stop_loss_double_touch_exit) -- per the client:
# "Add a Hard Stop Stop Loss of Trigger x 3. This will be part of the
# Bracket Order, OCA." Direction confirmed: "Stop Loss is the opposite of
# Take Profit" -- see bracket_order_prices' own docstring. Narrowed from
# x3 to x2 2026-08-25, per the client: "HARD STOP LOSS = Trigger x2 @ MKT.
# Use x2, instead of x3." @ MKT matches the existing STP order type
# unchanged -- an STP triggers a market order the instant price reaches
# auxPrice, which is already what this is; only the multiplier changed.
HARD_STOP_TRIGGER_MULTIPLIER = 2
# "Maximum Hold Time" -- per the client 2026-08-19, the third way (besides
# Take Profit and Stop Loss) a trade can close. Unlike TP/SL this can't be
# a resting order at IBKR (it's a clock, not a price), so it stays a
# script-checked exit in check_exit(). Narrowed from 140 to 30 minutes,
# then corrected same day to 60 minutes, per the client 2026-08-20.
# Narrowed again to 20 minutes per the client 2026-09-15 (Checklist_15092026.pdf).
MAX_HOLD_MINUTES = 20
# Per the client 2026-09-15: "Trailing Take Profit = $1. Include it in the
# bracket order... From the time you enter INTU, if INTU rises by $1,
# stop loss will trail by $1, protecting that profit." Confirmed as
# IBKR's own native TRAIL order type (trailAmount = this constant), not
# something the bot computes and resubmits itself -- a FIFTH OCA leg
# alongside the existing Hard Stop/Take Profit/Timed Exit/PnL Stop, not a
# replacement for any of them.
TRAILING_TAKE_PROFIT_AMOUNT = 1.00
# Per the client 2026-08-19: "IF Take Profit is Done, no re-entry after
# 11am ET. If Take Profit is Done, we can re-enter, until 11am. Maximum
# Trades per day = 2." This replaces the flat "one trade per day, no
# re-entry" from 2026-08-18 with a single carve-out: closing via Take
# Profit before the cutoff unlocks ONE re-entry (see reentry_allowed
# below), capped at MAX_TRADES_PER_DAY regardless. Every other close --
# Stop Loss, the Maximum Hold Time, EOD, a failed bracket placement --
# still ends the day outright, same as before. Cutoff moved to 10:30am ET
# per the client 2026-08-21's Entry Conditions PDF (matches the same
# narrowed LATEST_ENTRY_TIME).
MAX_TRADES_PER_DAY = 2
REENTRY_CUTOFF_TIME = time(10, 30)  # "no re-entry after 10:30am ET"


def is_early_close_day(close_time: time) -> bool:
    """close_time comes from IBKR's reqContractDetails() liquidHours for
    today, same technique as the sibling SPY bot's own early-close check
    (using SPY as a market-wide RTH calendar proxy -- all US equities share
    the same holiday/early-close schedule)."""
    return close_time < EARLY_CLOSE_THRESHOLD


@dataclass
class PairSignal:
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    deviation: float  # signed: current_spread - current_equilibrium
    divergence_trigger_level: float
    price_a: float  # current price of leg A
    half_life_minutes: float  # this pair's freshly-computed intraday half-life -- still needed for screening
    # Leg B's current price -- needed for the Maximum Stock Price entry
    # filter, which applies PER STOCK, not per pair, so both legs have to be
    # checked independently. Defaults to 0.0 (never exceeds the cap) so
    # existing callers/tests that don't care about this filter don't have
    # to change.
    price_b: float = 0.0
    # Per the client 2026-08-19: "Trigger has not previously been reached
    # at any point during the current RTH session" -- True if this pair's
    # divergence has ever equalled or exceeded its trigger level at any
    # earlier point today, tracked by the caller (main.py's PairContext)
    # across the whole session, not just this one check. Defaults to False
    # so existing callers/tests that don't track this don't have to change.
    trigger_reached_today: bool = False
    # Per the client 2026-08-22's expansion-strategy rewrite: "Bot will
    # filter off co-integration, and filter in expansion." A pair must have
    # been correlated/cointegrated through the prior session's 4pm close
    # (Phase 1 screening, already gates qualified_pairs.csv), then broken
    # (correlation or ADF) SPECIFICALLY before 9:31am ET -- a break that
    # happens later doesn't satisfy this checkpoint. Computed once per
    # session by main.py's finalize_trigger_at_931. Defaults to True so
    # existing callers/tests that don't track this don't have to change,
    # same convention as trigger_reached_today above.
    cointegration_broke_before_931: bool = True
    # Per the client: "10am till Now ensures broken co-integration right
    # until entry. Not just till 10am." -- re-verified every check cycle
    # from 10am onward over a rolling N=30-bar window (main.py's
    # update_live_cointegration_stats), separate from the 9:31-10:30am
    # recalculation above (main.py's recalculate_trigger_exceptions -- also
    # continuous, not a one-time 9:31 checkpoint, per the client's later
    # clarification). Same True default/reasoning.
    cointegration_still_broken: bool = True
    # Allowed Divergence Gap (D1 - D2 <= 0.5% of ELV / total leg size, see
    # spread_engine.passes_gap_condition), one of the PDF's "trade must
    # also satisfy" conditions. Per the client 2026-08-23: "Allowed
    # Divergence Gap does not apply to Exception Maximum Divergence
    # determined at Entry" -- main.py sets this True unconditionally
    # whenever ctx.using_exception_trigger, and to the real
    # passes_gap_condition() result otherwise. Same True default/reasoning
    # as the fields above.
    gap_ok: bool = True
    # Per the client 2026-08-25 (Correct_Filters.pdf): Johansen must be
    # Unconfirmed and Hurst must clear the trending threshold, continuously
    # rechecked "with new 1 minute candles, right until entry" over a
    # rolling 20 trading day window (plus today from 9:30am forward) -- not
    # just Phase 1's one-time screening snapshot. Computed every cycle by
    # main.py's update_live_johansen_hurst.
    #
    # Sticky once true, per the client 2026-08-28: "Hurst and Johansen ends
    # IF validated only once, during current RTH session," clarified same
    # day: "They must not be validated together. At the same time. No.
    # They just need to be validated once during Entry RTH session." --
    # main.py tracks Johansen-Unconfirmed and Hurst-above-threshold as two
    # INDEPENDENT sticky flags (each may clear in a different cycle) and
    # sets this field from their AND. Confirmed live 2026-08-28: GE/RTX
    # crossed its entry window on a single 60-second check where live_hurst
    # read 0.496 (just under the 0.60 floor), missed entry, then never got
    # a second chance since price moved straight through and touched the
    # trigger. Fail-closed default (False) so a pair that hasn't cleared
    # both conditions at least once this session (not necessarily
    # together) cannot enter.
    johansen_hurst_confirmed: bool = False
    # Per the client 2026-09-07 (Checklist_090926.pdf, entry condition A):
    # "There is no 100% mean reversion after Maximum Divergence." True if
    # the spread has fully closed back to equilibrium at any point since
    # the peak that set the currently applicable divergence_trigger_level,
    # over the lookback window (Prior RTH + Current RTH+Now) -- computed by
    # main.py's establish_base_trigger/recalculate_trigger_exceptions via
    # spread_engine.reverted_after_peak. The OTHER half of condition A
    # ("Maximum Divergence dropped below 25% of Trigger") is already the
    # existing 13%-25%-of-Trigger entry window below -- per the client,
    # direct: "If you are going to join 13%-25% of Trigger to get to
    # Trigger, you have to make sure that divergence have dropped below
    # 25% from Maximum Divergence... The only 'new scope' is that there is
    # no 100% mean reversion after Maximum Divergence." Fail-open default
    # (False) so existing callers/tests that don't track this don't have
    # to change, same convention as trigger_reached_today above.
    #
    # DORMANT in practice, 2026-09-08 onward: main.py's PairSignal
    # construction (run_check_cycle) never passes this keyword, so every
    # real signal gets the dataclass default (False) regardless of the
    # pair's actual reversion state -- the field/check stay in the code
    # (dormant-not-deleted), but nothing live can ever trip it. Per the
    # client the same day: "Please remove the condition of 100% mean
    # reversion after Maximum Divergence. So, we can trade today." It read
    # True on all 37 tracked pairs, blocking the entire watch list.
    mean_reverted_since_max_divergence: bool = False
    # Per the client 2026-09-13: "After pair hit the maximum divergence, it
    # should have at least >50% mean reversion before entry." A DISTINCT
    # condition from the one above (confirmed the same night: "There is no
    # condition of 'the pair not having fully mean reverted since its max
    # divergence'. This condition does not exist.") -- opposite polarity
    # too: blocks entry when this is FALSE (not reverted enough yet), not
    # when True. Computed the same way as mean_reverted_since_max_divergence
    # (main.py's establish_base_trigger/recalculate_trigger_exceptions, via
    # spread_engine.reverted_more_than_pct_after_peak), and IS wired into
    # the real PairSignal construction -- this one is live, not dormant.
    reverted_over_50pct_since_max_divergence: bool = False

    @property
    def abs_deviation(self) -> float:
        return abs(self.deviation)

    @property
    def distance_to_trigger(self) -> float:
        """How far this signal's divergence still has to move to reach its
        own trigger level -- 0 means it's right at the trigger, negative
        means it has already crossed it."""
        return self.divergence_trigger_level - self.abs_deviation


@dataclass
class Position:
    symbol_a: str
    symbol_b: str
    action_a: str  # "BUY" or "SELL" -- action_b is always the opposite
    qty_a: float
    qty_b: float
    entry_time: datetime
    entry_spread: float
    half_life_minutes: float
    entry_notional: float
    entry_beta: float = 0.0  # frozen hedge ratio for the day (see main.py's module docstring)
    # Raw leg prices at entry. P&L has to be measured against these fixed
    # prices and the position's OWN executed ratio, not entry_beta directly --
    # the BAG combo fills whole-number leg ratios, so differencing spreads
    # computed at mismatched ratios fabricates P&L the position never
    # actually experienced.
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    # The divergence at the moment this position opened -- purely
    # informational for the report ("size of the gap" at entry), not read
    # by any exit decision under the new expansion strategy.
    entry_divergence: float = 0.0
    entry_equilibrium: float = 0.0
    # The divergence trigger level in effect when this position opened,
    # fixed for the life of the position -- per the client 2026-08-18:
    # "Take Profit at Trigger." Exits when the live divergence reaches or
    # exceeds this, not a continuously-recalculated live trigger, same
    # "fixed once at entry" convention this module has always used for
    # entry-time reference values.
    entry_trigger_level: float = 0.0
    # A flat INITIAL_MARGIN_FRACTION (35%) estimate of the position actually
    # entered (fraction x price x units traded), per the client 2026-09-10:
    # "remove whatIF. Calculate quantity using 35% Initial Margin Deposit" --
    # replaced the original whatIf-quoted figure this field used to hold
    # (per the client 2026-08-17: "Pair Entry Initial Margin (the figure
    # returned by IBKR what IF)") after IBKR's whatIf was found to
    # under-report a short's real requirement (~36% quoted vs ~66% actual).
    # Purely informational for the dashboard, not read by any trading
    # decision. See entry_margin_ibkr_probed below for the REAL IBKR-
    # confirmed figure from the real-order margin probe.
    entry_margin: float = 0.0
    # The real IBKR-confirmed initial margin (error 201's own dollar
    # figure, not an estimate) from the LAST real-order probe rejection
    # main.py's resize loop hit before landing on the quantity actually
    # traded -- per the client 2026-09-14: "I want to see the initial
    # margin required by IBKR and see if it is the normal." None if the
    # very first size probed already fit (no rejection ever came back, so
    # there's no real figure to show -- entry_margin above is the only
    # number available in that case). entry_margin_ibkr_probed_qty is the
    # share count that figure applied to -- usually somewhat above the
    # quantity actually traded, since the loop exits the probe the moment
    # a smaller size fits, per the same live-verified 2026-09-11 finding
    # noted on the resize loop itself: required margin does not scale
    # proportionally with quantity, so this is the closest real IBKR
    # number available, not a bit-for-bit match to the traded size.
    entry_margin_ibkr_probed: float | None = None
    entry_margin_ibkr_probed_qty: int | None = None
    # Worst (most negative) unrealized_pnl observed since entry, in real
    # account dollars. Updated every check cycle while the position is open,
    # so the report's Max/Mean ELV loss % reflects what this specific trade
    # actually experienced.
    worst_unrealized_pnl: float = 0.0
    entry_order_id: int | None = None
    # The second leg's own entry order id -- added 2026-09-03 when
    # Combo/BAG orders were removed and each leg became its own
    # independent order (previously one combo order covered both legs
    # under a single id). entry_order_id is leg A's, this is leg B's.
    entry_order_id_b: int | None = None
    # Take Profit (MIT) and Stop Loss (STP) bracket child order ids, placed
    # as an OCA pair at IBKR right after the entry's actual fill is
    # confirmed -- per the client 2026-08-19: "0 relative to equilibrium at
    # entry. 100% mean-reversion" confirmed Stop Loss = 0 Divergence is
    # measured against entry_equilibrium (fixed), not the live-drifting
    # one, which means both TP and SL are prices known at entry and CAN be
    # resting orders again (unlike the same-day-earlier version, where both
    # depended on a live-recomputed equilibrium and had to be script-fired
    # market orders). See ibkr_client.place_pair_bracket and
    # risk_manager.bracket_order_prices.
    #
    # Restructured 2026-08-24, per the client: Stop Loss ("0 Divergence")
    # is no longer part of the bracket at all -- it's bot-monitored now
    # (see stop_loss_double_touch_exit below). The bracket is Take Profit
    # (MIT, unchanged) + Hard Stop (STP @ Trigger x 2, new -- a broker-side
    # backstop) + Timed Exit (MKT, fires automatically at the 60-minute
    # mark via goodAfterTime, replacing the old script-checked
    # max_hold_time exit in check_exit), all three in one OCA group.
    tp_order_id: int | None = None
    hard_stop_order_id: int | None = None
    timed_exit_order_id: int | None = None
    # The -5% PnL Stop Loss, a FOURTH leg of the same OCA group -- per the
    # client 2026-09-09: "It is Ibkr PnL. 5% of EWLV." Priced by
    # pnl_stop_loss_price, which converts that account percentage into a
    # per-share price for the shares actually held. Sits very close to
    # hard_stop_order_id by construction; see that function's own comment
    # for when the two genuinely diverge.
    pnl_stop_order_id: int | None = None
    # IBKR's own native TRAIL order, a FIFTH leg of the same OCA group --
    # per the client 2026-09-15: "Trailing Take Profit = $1... if INTU
    # rises by $1, stop loss will trail by $1, protecting that profit."
    # Runs alongside every other leg here, not a replacement for any of
    # them.
    trail_order_id: int | None = None
    # Whether the Hard Stop Loss has been successfully re-set from the wide
    # @ ENTRY price to the tight, real AFTER ENTRY price yet -- per the
    # client 2026-09-09: "No limit to continuous re-try. It is until it is
    # Done. If it is not done, then the hard stop loss will not reset...
    # This applies only to Hard Stop Loss. Original level @ entry will
    # apply" (i.e. if it never succeeds, the wide stop just keeps working,
    # not a gap in protection). Retried every cycle in run_check_cycle
    # while this stays False; set True the first cycle the re-submission
    # is not rejected, and never touched again.
    hard_stop_reset_done: bool = False
    # Double-touch Stop Loss state, per the client 2026-08-24: "IF market
    # touch 0 Divergence, rise above 0 divergence, and then touch 0
    # Divergence again, Exit Pair @ MKT." No time limit on the second touch
    # beyond the position's own life (bounded by the Timed Exit and EOD
    # regardless). touched_zero_once tracks the FIRST touch; rose_above_
    # zero_after_touch tracks whether it's since moved back to the
    # expansion side -- the second touch only counts once both of those
    # are true, per stop_loss_double_touch_exit's own docstring.
    #
    # SUPERSEDED 2026-09-05 by the triple-touch fields below -- per the
    # client: "API Monitored Stop Loss is touch 0, rise, touch 0, rise
    # and touch 0 before exit" (API-Monitored Soft Exit, Triple-Touch
    # Rule, "Noise Immunity"). Left in place, unused, same convention
    # this codebase uses for every superseded mechanism.
    touched_zero_once: bool = False
    rose_above_zero_after_touch: bool = False
    # Triple-touch Stop Loss state -- per the client 2026-09-05: "three
    # complete cycles (Touch 1 -> Bounce -> Touch 2 -> Bounce -> Touch 3)
    # before sending a Market Exit order." zero_touch_count is how many
    # times equilibrium has been touched so far (0, 1, 2 -- the exit
    # fires the moment it would become 3); rose_since_last_touch tracks
    # whether the spread has moved back to the expansion side since the
    # MOST RECENT touch -- a touch only counts as a new one once it has,
    # same "must bounce out and back in" rule stop_loss_double_touch_exit
    # always used, just carried through a third cycle now instead of
    # stopping after the second. See stop_loss_triple_touch_exit's own
    # docstring.
    zero_touch_count: int = 0
    rose_since_last_touch: bool = False
    # Which symbol is the real, only leg actually held at the broker --
    # added 2026-09-03 for the client's "Expansion Leg" pivot away from a
    # hedged two-leg trade to a single-stock directional one ("You don't
    # need Hedge Ratio anymore"). "A" or "B". The OTHER leg's qty is 0
    # (never ordered). action_a is REPURPOSED to always hold the TRADED
    # leg's own action regardless of which symbol that is -- when
    # traded_leg="B", action_a holds symbol_b's action, not symbol_a's,
    # despite the field name. Every reader that needs the sign of the
    # real position (unrealized_pnl) goes through action_a + traded_leg
    # together, never action_a alone. Dashboard/report display, which
    # DOES assume action_a describes symbol_a specifically, is not yet
    # updated for this -- a known display-only gap, not a trading-logic
    # one. See unrealized_pnl's own comment for why P&L only ever reads
    # the traded leg now, not the full pair spread.
    traded_leg: str = "A"
    # THE OPPOSITE LEG -- per the client 2026-09-09, reversing the 2026-09-03
    # single-stock pivot: "(F) Calculate Final Position Size: Opposite Stock.
    # Apply Beta to determine opposite stock shares quantity", and "Execute
    # the Opposite Leg in the opposite direction."
    #
    # Deliberately ADDITIVE fields rather than repurposing qty_a/qty_b into a
    # symmetric two-leg position: every reader of traded_leg/action_a above
    # (unrealized_pnl, expansion_direction_sign, position_spread, the exit
    # state machine, the report) is written and tested against "one real leg
    # plus one zero leg", and quietly making both legs non-zero would change
    # all of their behaviour at once. The expansion leg stays the primary in
    # every one of those paths; this hedge leg is tracked alongside it and
    # folded into P&L explicitly (see unrealized_pnl).
    #
    # hedge_qty of 0 means no opposite leg was placed -- either the EWLV
    # Balance from (E) could not fund a single share, or its order did not
    # fill. Every hedge-leg code path must treat 0 as "there is no second
    # leg" rather than assuming one always exists.
    hedge_symbol: str = ""
    hedge_action: str = ""  # always the opposite of action_a
    hedge_qty: float = 0.0
    hedge_entry_price: float = 0.0
    hedge_entry_order_id: int | None = None
    # The opposite leg's own resting bracket. Per the client 2026-09-09:
    # "Exit both Stock A and Stock B IF Hard Stop Loss triggers in either
    # Stock A or Stock B", and the same for Take Profit -- so each leg
    # carries its own bracket, and whichever fires first closes the other
    # leg at market (the broker cannot do that cross-leg step itself; see
    # run_check_cycle's bracket reconciliation).
    hedge_tp_order_id: int | None = None
    hedge_hard_stop_order_id: int | None = None
    hedge_timed_exit_order_id: int | None = None


def position_hedge_ratio(position: Position) -> float:
    """The ratio the position was ACTUALLY executed at (qty_b / qty_a), not
    entry_beta -- the BAG combo fills whole-number leg ratios, so the real
    economic hedge is whatever quantities got filled. This is the only ratio
    for which the spread's dollar change equals the position's real dollar
    P&L per unit of qty_a."""
    if position.qty_a == 0:
        return 0.0
    return position.qty_b / position.qty_a


def position_spread(position: Position, price_a: float, price_b: float) -> float:
    """The position's own spread: P_A - ratio * P_B. Fixed ratio, so changes
    in it are pure P&L rather than a mix of P&L and beta drift.

    ratio is qty_b/qty_a (the position's own executed fill ratio) for a
    real two-leg position, but that's 0 for a single-leg "Expansion Leg"
    position (one leg's qty is always 0 by construction -- see
    open_single_leg_position), which would silently collapse this to the
    raw, unweighted price of whichever leg is nonzero. Single-leg
    positions use entry_beta instead -- the model hedge ratio Method 1
    was computed against, the only ratio available once there's no real
    second-leg fill to derive one from."""
    if position.qty_a == 0 or position.qty_b == 0:
        return price_a - position.entry_beta * price_b
    return price_a - position_hedge_ratio(position) * price_b


def expansion_direction_sign(position: Position) -> int:
    """+1 if positive divergence (spread above entry_equilibrium) is the
    side this trade profits from expanding further into, -1 if negative
    divergence is.

    Two-leg positions: action_a=="BUY" is this by construction --
    trade_direction sets action_a and the divergence sign together off
    the same value, always in agreement. Kept as the source here (rather
    than switching everything to entry_spread-vs-equilibrium) so existing
    two-leg behavior is untouched.

    Single-leg "Expansion Leg" positions (qty_a or qty_b == 0): action_a
    is repurposed to describe whichever leg actually got traded, and
    BUYING leg B pushes the A-referenced spread DOWN, not up -- action_a
    alone no longer says which side is the expansion side. Falls back to
    sign(entry_spread - entry_equilibrium) instead, the model-spread-
    native fact that stays correct regardless of which leg was traded."""
    if position.qty_a == 0 or position.qty_b == 0:
        return 1 if position.entry_spread >= position.entry_equilibrium else -1
    return 1 if position.action_a == "BUY" else -1


@dataclass
class DailyState:
    trading_halted: bool = False
    realized_pnl_today: float = 0.0  # accumulated by main.py on each exit, for the dashboard's daily P&L card
    # Hard ceiling per the client 2026-08-19: "Maximum Trades per day = 2".
    # Incremented the moment a trade is confirmed real (main.py), not per
    # signal or per attempt.
    trades_taken_today: int = 0
    # Set True on any close that does NOT qualify for the Take Profit
    # before-11am re-entry carve-out (see reentry_allowed) -- once True, no
    # further entries today regardless of trades_taken_today.
    reentry_locked_out: bool = False
    # One entry per real trade, built directly from its own Position at the
    # moment it closes -- real entry/exit times, real executed quantity, real
    # worst drawdown. Per the client: "Post-Trade Report should be based on
    # executed pairs" -- rather than filtering divergence events after the
    # fact, the report is populated only from trades that actually happened.
    trade_report_events: list = field(default_factory=list)


# The daily risk counters that MUST survive a mid-session process restart.
# Audit finding 2026-08-27: daily_state was rebuilt fresh on every start
# with nothing persisting it, so a restart silently reset trades_taken_today
# (defeating "Maximum Trades per day = 2" -- take 2, restart, take 2 more)
# and cleared reentry_locked_out (which is set by a stop loss, news stop,
# hard stop or a one-sided-fill failure and means "stop trading today").
# Yesterday's session included four restarts, one of them inside the entry
# window -- it only failed to bite because no trade had been taken yet.
#
# trade_report_events is deliberately NOT persisted: it holds ReportEvent
# objects (not JSON-serialisable without a converter), and losing it costs
# report completeness, not capital. The counters here are the safety-
# critical ones.
_PERSISTED_FIELDS = ("trading_halted", "realized_pnl_today",
                     "trades_taken_today", "reentry_locked_out")


def save_daily_state(state: DailyState, path, today: date | None = None) -> None:
    """Writes the risk counters, stamped with today's date. Best-effort: a
    failure here must never take down a live session, so it logs and
    continues rather than raising into the trading loop."""
    import json
    from pathlib import Path
    payload = {f: getattr(state, f) for f in _PERSISTED_FIELDS}
    payload["date"] = (today or date.today()).isoformat()
    try:
        p = Path(path)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, p)
    except OSError:
        logger.warning("could not persist daily state", exc_info=True)


def load_daily_state(path, today: date | None = None) -> DailyState:
    """Restores today's risk counters, or a fresh DailyState if there's
    nothing to restore.

    Fails OPEN to a fresh state (normal trading) on a missing, corrupt, or
    stale-dated file -- the file existing at all is the exceptional case,
    and refusing to trade because yesterday's leftovers couldn't be parsed
    would be its own outage. The date stamp is what makes that safe: a file
    from any prior day is ignored outright rather than carrying a stale
    lockout into a new session."""
    import json
    from pathlib import Path
    today = today or date.today()
    try:
        data = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return DailyState()
    if data.get("date") != today.isoformat():
        return DailyState()  # yesterday's (or older) -- start clean
    state = DailyState()
    for f in _PERSISTED_FIELDS:
        if f in data:
            setattr(state, f, data[f])
    logger.info(f"Restored today's daily state after a restart: "
                f"trades_taken_today={state.trades_taken_today}, "
                f"reentry_locked_out={state.reentry_locked_out}, "
                f"trading_halted={state.trading_halted}")
    return state


_PAIR_TRIGGER_LOCK_TZ = ZoneInfo("America/New_York")
_PAIR_TRIGGER_LOCK_END_TIME = time(20, 0)  # 8pm ET


def save_pair_trigger_state(
    pairs: "list[tuple[str, str, bool]]", path, now: datetime | None = None,
) -> None:
    """Persists trigger_reached_today per pair, keyed by "symbol_a/symbol_b",
    stamped with an explicit lock-expiry timestamp -- the SAME class of gap
    save_daily_state/load_daily_state fixed for the account-wide counters
    on 2026-08-27, just never applied to this per-pair sticky flag.

    Confirmed live 2026-09-14: a mid-session restart (to fix an unrelated
    warm-up issue) reset trigger_reached_today back to False in memory for
    every tracked pair -- PLTR/WDAY had already reached its trigger earlier
    that session (correctly locking it out for the rest of the day per
    the client 2026-08-19: "Trigger has not previously been reached at
    any point during the current RTH session"), the restart cleared that,
    and the pair went on to actually enter a real trade it should never
    have been eligible for.

    The expiry is stamped as an explicit ET wall-clock timestamp (today's
    date at 8pm ET) rather than a raw calendar date, per the client
    2026-09-14: "If trigger is reached in current RTH session, pair is
    locked until 8pm ET." A raw date.today() rollover isn't equivalent --
    the server runs in UTC, so a same-calendar-day check happens to lapse
    at 8pm ET only while EDT is in effect (UTC-4) and would silently drift
    to 7pm ET once EST resumes.

    `pairs` is a plain list of (symbol_a, symbol_b, trigger_reached_today)
    tuples rather than PairContext objects, so this module stays free of
    any dependency on main.py (same reasoning as position_sizing.py's own
    "stays pure/no I/O" convention). Best-effort, same as save_daily_state:
    a failure here must never take down a live session."""
    import json
    from pathlib import Path
    now = (now or datetime.now(_PAIR_TRIGGER_LOCK_TZ)).astimezone(_PAIR_TRIGGER_LOCK_TZ)
    payload = {
        f"{a}/{b}": reached for a, b, reached in pairs if reached
    }  # only pairs that actually reached trigger need persisting at all
    locked_until = datetime.combine(now.date(), _PAIR_TRIGGER_LOCK_END_TIME, tzinfo=_PAIR_TRIGGER_LOCK_TZ)
    data = {"locked_until": locked_until.isoformat(), "pairs": payload}
    try:
        p = Path(path)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, p)
    except OSError:
        logger.warning("could not persist pair trigger state", exc_info=True)


def load_pair_trigger_state(path, now: datetime | None = None) -> dict[str, bool]:
    """Restores the trigger_reached_today-by-pair map, or an empty one if
    there's nothing to restore -- same fail-open semantics as
    load_daily_state, but keyed off the 8pm ET lock expiry (see
    save_pair_trigger_state) rather than a raw calendar date: a file whose
    locked_until has already passed is ignored outright, never carrying a
    stale lockout into a new session. Returns a dict keyed by
    "symbol_a/symbol_b" -> True for every pair that had already reached
    its trigger; the caller applies this to its own PairContext objects by
    matching keys."""
    import json
    from pathlib import Path
    now = (now or datetime.now(_PAIR_TRIGGER_LOCK_TZ)).astimezone(_PAIR_TRIGGER_LOCK_TZ)
    try:
        data = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    try:
        locked_until = datetime.fromisoformat(data["locked_until"])
    except (KeyError, TypeError, ValueError):
        return {}
    if now >= locked_until:
        return {}
    restored = data.get("pairs", {})
    if restored:
        logger.info(f"Restored today's trigger_reached_today state after a restart "
                    f"for: {', '.join(sorted(restored))}")
    return restored


def trade_direction(deviation: float) -> tuple[str, str]:
    """Per the client's 2026-08-18 strategy change, this is the OPPOSITE of
    what it used to be. The old mean-reversion convention sold the
    expensive leg expecting the spread to revert back toward equilibrium.
    The new strategy profits from the divergence EXPANDING further past
    its current level (entry only happens close to, but before, the
    trigger -- Take Profit fires when it reaches/exceeds the trigger), so
    the position has to be positioned to gain when the spread keeps moving
    the SAME direction it's already moving, not reverses:

    Spread above equilibrium and still rising (deviation > 0): BUY the
    (already relatively expensive) leg A, SELL B -- profits further as A
    keeps outpacing B. Spread below equilibrium and still falling
    (deviation < 0): SELL A, BUY B -- profits further as A keeps
    underperforming B.

    Verified algebraically against unrealized_pnl = qty_a * sign *
    (current_spread - entry_spread), sign=+1 for BUY A: if deviation > 0
    and the spread keeps rising (current_spread grows even more positive
    relative to entry), a BUY-A position's unrealized_pnl grows positive
    right along with it."""
    return ("BUY", "SELL") if deviation > 0 else ("SELL", "BUY")


def entry_eligible(
    signal: PairSignal,
    permanent_blacklisted_symbols: set[str],
    now: datetime,
    daily_state: DailyState,
) -> bool:
    if daily_state.trading_halted:
        return False
    # Per the client 2026-08-19: "Maximum Trades per day = 2" plus the
    # Take-Profit-before-11am re-entry carve-out (see reentry_allowed) --
    # checked before any other entry condition.
    if daily_state.trades_taken_today >= MAX_TRADES_PER_DAY:
        return False
    if daily_state.reentry_locked_out:
        return False
    if now.time() < EARLIEST_ENTRY_TIME:
        return False
    if now.time() > LATEST_ENTRY_TIME:
        return False
    # Per the client 2026-08-26: news exclusion does NOT gate entry --
    # "If a symbol is in the news, it should get in. The news could be the
    # reason for the expansion. We want expansion." Only the PERMANENT
    # blacklist (the client's own manually curated never-trade list,
    # unrelated to news) blocks entry here. News exclusion is handled
    # entirely as an exit (News Stop, main.py's run_check_cycle), not here.
    if signal.symbol_a in permanent_blacklisted_symbols or signal.symbol_b in permanent_blacklisted_symbols:
        return False
    # Per the client 2026-08-19: "Trigger has not previously been reached
    # at any point during the current RTH session" -- a pair that already
    # touched or crossed its trigger earlier today and pulled back is
    # disqualified for the rest of the day, even if it's now back inside
    # the $0.01-$0.40 entry window on a second approach.
    if signal.trigger_reached_today:
        return False
    # Per the client 2026-08-24: rolled back to the pre-2026-08-22 "old
    # script" -- pairs qualify off correlation/cointegration through the
    # prior day's 4pm close (Phase 1 screening, unchanged) and entry
    # triggers purely off approaching the prior RTH session's maximum
    # divergence. The cointegration-broke-before-931/still-broken gate
    # added 2026-08-22's expansion-filter update is removed entirely, not
    # just relaxed -- signal.cointegration_broke_before_931/
    # cointegration_still_broken are no longer read here.
    #
    # Volume Delta Skew (CVD Skew) removed as a gate 2026-08-27, per the
    # client: "Delete CVD Skew."
    #
    # Per the client 2026-08-25 (Correct_Filters.pdf): Johansen Unconfirmed
    # and Hurst > 0.60, live and continuous -- but sticky once validated
    # once this session, per the client 2026-08-28. See
    # PairSignal.johansen_hurst_confirmed above.
    if not signal.johansen_hurst_confirmed:
        return False
    # Per the client 2026-09-07 (Checklist_090926.pdf), entry condition A:
    # "There is no 100% mean reversion after Maximum Divergence." See
    # PairSignal.mean_reverted_since_max_divergence's own comment -- DORMANT
    # in practice, never actually trips on a real signal.
    if signal.mean_reverted_since_max_divergence:
        return False
    # Per the client 2026-09-13: "After pair hit the maximum divergence, it
    # should have at least >50% mean reversion before entry." Opposite
    # polarity from the dormant check above -- blocks when this HASN'T
    # happened yet, not when it has. See
    # PairSignal.reverted_over_50pct_since_max_divergence's own comment.
    if not signal.reverted_over_50pct_since_max_divergence:
        return False
    # Allowed Divergence Gap removed as a gate 2026-08-25, per the client:
    # "Delete: Allowed Divergence Gap." signal.gap_ok is no longer read
    # here. Left on PairSignal and still computed in main.py (dormant,
    # same pattern as the other removals tonight) rather than deleted
    # outright, to limit the blast radius of a change to the live,
    # currently-running bot.
    #
    # abs_deviation < MIN_ENTRY_DIVERGENCE / > MAX_ENTRY_DIVERGENCE checks
    # REMOVED here 2026-09-12, per the client, ending a long same-night
    # back-and-forth: "1.5-5 range applies only to the trigger. I never
    # said live current price must be 1.5-5. Live Current price should
    # be inside the Entry Window, which is derived from the Maximum
    # Divergence, obtained yesterday." MIN_ENTRY_DIVERGENCE/
    # MAX_ENTRY_DIVERGENCE now gate ONLY the trigger level itself (see
    # _bounded_trigger_level, phase2/main.py) -- today's live abs_deviation
    # is governed exclusively by the distance-to-trigger window check
    # below (ENTRY_TRIGGER_DISTANCE_MIN_PCT/MAX_PCT of the trigger), which
    # was already the real Entry Window this whole time.
    # Maximum Stock Price, per the client 2026-08-17: "Maximum Stock Price of
    # $700 is for a stock. Not a pair." -- checked on each leg independently,
    # not the pair's combined notional. price_b defaults to 0.0 for callers
    # that don't track it, so this never blocks a signal that only carries
    # price_a.
    if signal.price_a > MAX_STOCK_PRICE or signal.price_b > MAX_STOCK_PRICE:
        return False
    # Entry timing, per the client 2026-08-24: "Entry Window: Pair must be
    # within 13%-25% of Trigger points to get to Trigger" -- must still be
    # short of the trigger, and within the band (as a fraction of the
    # trigger level itself), not a fixed dollar range and not just under a
    # single ceiling. A zero or negative trigger level never qualifies
    # (min/max against zero can't bound a real window).
    distance = signal.distance_to_trigger
    if signal.divergence_trigger_level <= 0:
        return False
    min_distance = ENTRY_TRIGGER_DISTANCE_MIN_PCT * signal.divergence_trigger_level
    max_distance = ENTRY_TRIGGER_DISTANCE_MAX_PCT * signal.divergence_trigger_level
    # A tiny epsilon, not just a test-authoring convenience -- distance is
    # trigger - abs_deviation while min/max are PCT * trigger, two different
    # floating-point paths that don't always round identically even for
    # values that are mathematically exactly at the boundary. Without this,
    # a real trade sitting genuinely at 13% or 25% of its trigger could be
    # incorrectly excluded by a rounding error smaller than a thousandth of
    # a cent.
    epsilon = 1e-9
    if distance < min_distance - epsilon or distance > max_distance + epsilon:
        return False
    return True


def select_closest_to_trigger(signals: list[PairSignal]) -> PairSignal | None:
    """Per the client 2026-08-18: "From The Divergence Watch List, select
    the closest to trigger and enter." When 2+ pairs are eligible in the
    same check cycle, the one with the smallest remaining distance to its
    own trigger wins.

    SUPERSEDED as the tiebreak 2026-09-12, per the client (Checklist_120926.pdf,
    "Yes. Fully replace trigger tiebreaker. Use initial margin as
    described."): select_highest_adjusted_spread_efficiency (below) is
    what main.py's entry flow actually calls now when 2+ pairs are
    eligible at once. Left here, not deleted, same dormant-not-deleted
    convention as every other superseded gate in this codebase -- it's
    exactly what this function's own docstring predicted might happen
    someday ("superseding the old... Adjusted Spread Efficiency ranking")
    now reversed."""
    if not signals:
        return None
    return min(signals, key=lambda s: s.distance_to_trigger)


def select_highest_adjusted_spread_efficiency(
    ranked: list[tuple[PairSignal, float]],
) -> PairSignal | None:
    """Per the client 2026-09-12 (Checklist_120926.pdf): "If there are two
    or more opportunities to trade at the same time, robot will select a
    pair with the highest Adjusted Spread Efficiency... Spread Return % =
    (Divergence $/Stock Initial Margin) x100... The robot ranks all
    eligible pairs in descending order of Spread Return % and executes
    the highest-ranked opportunity." Confirmed the same day: "If it is
    only one qualified pair, then that pair will execute. But if it more
    than more qualified pair, we will use Adjusted Spread Efficiency to
    select. Yes. Fully replace trigger tiebreaker."

    Takes (signal, stock_initial_margin_per_share) pairs -- the margin
    figure is real-order-probed per candidate in main.py (I/O, so it
    can't live in this pure module), this function is just the ranking
    math: divergence $ / margin per share, highest wins. The x100 in the
    client's formula is a pure scale factor on every candidate equally,
    so it's dropped here without changing which one ranks highest --
    kept out to avoid a meaningless "spread_return_pct" field nothing
    else reads.

    Single-candidate callers don't need this at all (see main.py's own
    "only one qualified pair" skip), but it degenerates correctly here
    too -- one candidate's own ratio trivially wins against nothing."""
    if not ranked:
        return None
    return max(ranked, key=lambda pair: pair[0].abs_deviation / pair[1])[0]


def open_position(signal: PairSignal, qty_a: float, qty_b: float, entry_notional: float,
                   half_life_minutes: float, current_spread: float, now: datetime,
                   entry_price_a: float = 0.0, entry_price_b: float = 0.0,
                   entry_equilibrium: float = 0.0, entry_margin: float = 0.0,
                   entry_trigger_level: float = 0.0) -> Position:
    """entry_divergence is recomputed here from current_spread and
    entry_equilibrium (abs(current_spread - entry_equilibrium)), not read
    from signal.abs_deviation -- the signal's own deviation is the MODEL
    spread at signal-detection time, but current_spread here is (per the
    real caller, main.py) the ACTUAL fill spread, which can differ from the
    model (confirmed live 2026-08-13: MCD/SBUX's real fill ratio differed
    from what was ordered)."""
    action_a, _ = trade_direction(signal.deviation)
    return Position(
        symbol_a=signal.symbol_a, symbol_b=signal.symbol_b,
        action_a=action_a, qty_a=qty_a, qty_b=qty_b,
        entry_time=now, entry_spread=current_spread,
        half_life_minutes=half_life_minutes, entry_notional=entry_notional,
        entry_beta=signal.hedge_ratio,
        entry_price_a=entry_price_a, entry_price_b=entry_price_b,
        entry_divergence=abs(current_spread - entry_equilibrium),
        entry_equilibrium=entry_equilibrium,
        entry_margin=entry_margin,
        entry_trigger_level=entry_trigger_level,
    )


def open_single_leg_position(
    symbol_a: str, symbol_b: str, traded_symbol: str, action: str, qty: float,
    entry_price_a: float, entry_price_b: float, entry_notional: float,
    half_life_minutes: float, current_spread: float, hedge_ratio: float, now: datetime,
    entry_equilibrium: float = 0.0, entry_margin: float = 0.0, entry_trigger_level: float = 0.0,
    entry_margin_ibkr_probed: float | None = None, entry_margin_ibkr_probed_qty: int | None = None,
) -> Position:
    """The Expansion Leg entry -- per the client 2026-09-03: only ONE
    stock ever gets a real order. traded_symbol/action/qty describe that
    one real leg (from expansion_leg_decision + a real fill); the OTHER
    leg's quantity is 0 (never ordered), but its entry price is still
    recorded, since Take Profit/Stop Loss stay tied to the PAIR's spread
    reaching a level (unchanged, confirmed the same day) -- the spread
    needs both legs' prices to evaluate, even though only one is real
    money.

    entry_divergence/entry_spread here are the MODEL spread at the
    moment of entry (current_spread, entry_price_a/b as given), not
    re-derived from a real two-leg fill the way open_position's
    docstring describes -- there is no such fill to derive it from
    anymore, only the one real leg's own fill price feeds entry_price_a
    or entry_price_b, whichever traded_symbol names."""
    is_a = traded_symbol == symbol_a
    return Position(
        symbol_a=symbol_a, symbol_b=symbol_b,
        action_a=action, qty_a=qty if is_a else 0.0, qty_b=qty if not is_a else 0.0,
        traded_leg="A" if is_a else "B",
        entry_time=now, entry_spread=current_spread,
        half_life_minutes=half_life_minutes, entry_notional=entry_notional,
        entry_beta=hedge_ratio,
        entry_price_a=entry_price_a, entry_price_b=entry_price_b,
        entry_divergence=abs(current_spread - entry_equilibrium),
        entry_equilibrium=entry_equilibrium,
        entry_margin=entry_margin,
        entry_trigger_level=entry_trigger_level,
        entry_margin_ibkr_probed=entry_margin_ibkr_probed,
        entry_margin_ibkr_probed_qty=entry_margin_ibkr_probed_qty,
    )


def attach_hedge_leg(
    position: Position, hedge_symbol: str, hedge_action: str,
    hedge_qty: float, hedge_entry_price: float, hedge_entry_order_id: int | None = None,
) -> Position:
    """Records the OPPOSITE leg on an already-open expansion-leg position --
    per the client 2026-09-09: "Execute the Opposite Leg in the opposite
    direction."

    Called only after that leg's own order has actually FILLED, so hedge_qty
    is a real executed quantity and hedge_entry_price a real average fill,
    never an intended size -- the same convention the expansion leg itself
    follows. The position is mutated in place and returned for convenience.

    hedge_action must be the opposite of position.action_a: a caller passing
    the same direction would be DOUBLING the position rather than hedging
    it, so that is rejected outright rather than silently executed."""
    expected = "SELL" if position.action_a == "BUY" else "BUY"
    if hedge_action != expected:
        raise ValueError(
            f"hedge leg must be the opposite direction: expansion leg is {position.action_a}, "
            f"so the hedge must be {expected}, got {hedge_action}")
    position.hedge_symbol = hedge_symbol
    position.hedge_action = hedge_action
    position.hedge_qty = hedge_qty
    position.hedge_entry_price = hedge_entry_price
    position.hedge_entry_order_id = hedge_entry_order_id
    return position


def leg_still_open_after_bracket_fill(position: Position, fired_on_hedge: bool) -> "tuple[str, tuple] | None":
    """Cross-leg exit bookkeeping -- per the client 2026-09-09: "Exit both
    Stock A and Stock B IF Hard Stop Loss triggers in either Stock A or
    Stock B" (same for Take Profit).

    Given which leg's bracket just fired, returns the OTHER leg's
    (symbol, its three bracket order ids) so the caller can cancel those
    orphaned orders and close that leg at market. Returns None when there
    is nothing to do -- no opposite leg was ever placed (hedge_qty == 0),
    so the bracket that fired closed the entire position by itself.

    Pulled out of run_check_cycle as a pure function specifically because
    it is the one piece of the two-leg exit that can be got wrong silently:
    closing the leg that just closed itself would be a no-op, while
    leaving the true survivor open would leave real money unhedged and
    unprotected."""
    if position.hedge_qty <= 0:
        return None
    if fired_on_hedge:
        survivor = position.symbol_a if position.traded_leg == "A" else position.symbol_b
        return survivor, (position.tp_order_id, position.hard_stop_order_id,
                          position.timed_exit_order_id)
    return position.hedge_symbol, (position.hedge_tp_order_id, position.hedge_hard_stop_order_id,
                                   position.hedge_timed_exit_order_id)


def unrealized_pnl(position: Position, price_a: float, price_b: float) -> float:
    """Exact dollar P&L of the ONE real leg actually held --
        sign * qty * (P_now - P_entry)
    for whichever symbol position.traded_leg names.

    Rebuilt 2026-09-03 for the client's "Expansion Leg" pivot: Take Profit
    and Stop Loss are still tied to the PAIR's spread reaching a level
    (unchanged, confirmed the same day -- see check_exit/
    bracket_order_prices, still spread-based), but P&L itself can only
    ever reflect real money, and only one leg is ever real now. The old
    formula (spread P&L across both legs) would show a number that
    doesn't match what the account's own balance actually did, which
    would poison the daily loss cap along with everything else that
    reads this.

    Extended 2026-09-09 for the client's return to two legs: the OPPOSITE
    leg's own P&L is added whenever one was actually placed (hedge_qty > 0).
    Both legs are real money at the broker now, so both belong in the number
    the daily loss cap reads. A position with no opposite leg (hedge_qty ==
    0 -- the EWLV Balance couldn't fund a share, or its order never filled)
    computes exactly as it did before, unchanged."""
    sign = 1 if position.action_a == "BUY" else -1
    if position.traded_leg == "A":
        pnl = position.qty_a * sign * (price_a - position.entry_price_a)
    else:
        pnl = position.qty_b * sign * (price_b - position.entry_price_b)
    if position.hedge_qty > 0:
        hedge_sign = 1 if position.hedge_action == "BUY" else -1
        hedge_price_now = price_a if position.hedge_symbol == position.symbol_a else price_b
        pnl += position.hedge_qty * hedge_sign * (hedge_price_now - position.hedge_entry_price)
    return pnl


def unrealized_pnl_pct(position: Position, price_a: float, price_b: float) -> float:
    return unrealized_pnl(position, price_a, price_b) / position.entry_notional


def stop_loss_double_touch_exit(position: Position, price_a: float, price_b: float) -> bool:
    """SUPERSEDED 2026-09-05 by stop_loss_triple_touch_exit below -- per
    the client: "API Monitored Stop Loss is touch 0, rise, touch 0, rise
    and touch 0 before exit" (a third touch/bounce cycle added for
    "Noise Immunity" -- see that function's own docstring). Left in
    place, unused, same convention this codebase uses for every
    superseded mechanism.

    Per the client 2026-08-24, replacing the native STP "0 Divergence"
    bracket leg entirely: "IF market touch 0 Divergence, rise above 0
    divergence, and then touch 0 Divergence again, Exit Pair @ MKT. 0 is
    relative to equilibrium at entry." Confirmed: "touch" means crossing
    exactly through equilibrium, and there is no time limit on the second
    touch beyond the position's own life.

    Mutates position.touched_zero_once / rose_above_zero_after_touch as a
    3-state machine (not touched -> touched once -> rose back above ->
    exit on the second touch), checked every cycle against the position's
    OWN executed ratio (position_spread), same convention unrealized_pnl
    uses -- not entry_beta, which the live Kalman estimator keeps moving
    post-entry and is deliberately never read here.

    "0 divergence" is relative to entry_equilibrium in the direction the
    trade was originally betting on -- see expansion_direction_sign for
    which side that is and why action_a alone stopped being reliable for
    single-leg "Expansion Leg" positions (2026-09-03). A live divergence
    that is still on the expansion side (the side the trade was betting
    on) has NOT touched 0 -- only crossing to the equilibrium side, or
    landing exactly on it, counts as a touch.

    Deliberately does NOT reset the state machine if the position later
    moves back toward expansion without a genuine second touch -- once
    touched, staying touched (rose_above_zero_after_touch can flip back and
    forth, but touched_zero_once never un-sets) matches the client's own
    "and THEN touch again" wording: the first touch is a standing fact
    about this trade's history, not a live condition that can un-happen."""
    sign = expansion_direction_sign(position)
    live_divergence = position_spread(position, price_a, price_b) - position.entry_equilibrium
    on_equilibrium_side = (live_divergence * sign) <= 0

    if not position.touched_zero_once:
        if on_equilibrium_side:
            position.touched_zero_once = True
        return False

    if not position.rose_above_zero_after_touch:
        if not on_equilibrium_side:
            position.rose_above_zero_after_touch = True
        return False

    # Already touched once, then rose back above -- this touch is the second.
    return on_equilibrium_side


def stop_loss_triple_touch_exit(position: Position, price_a: float, price_b: float) -> bool:
    """Per the client 2026-09-05, reinstating the bot-monitored Stop Loss
    (deleted the day before) with an added third cycle: "API-Monitored
    Soft Exit (Triple-Touch Rule), High Precision Control... Noise
    Immunity: Instead of getting stopped out on a single random
    micro-tick that touches 0 Divergence, your API monitor requires
    three complete cycles (Touch 1 -> Bounce -> Touch 2 -> Bounce ->
    Touch 3) before sending a Market Exit order." Confirmed directly:
    "API Monitored Stop Loss is touch 0, rise, touch 0, rise and touch 0
    before exit."

    Same "0 divergence relative to entry_equilibrium, in the direction
    the trade was betting on" convention as stop_loss_double_touch_exit
    (see expansion_direction_sign), and the same "must bounce OUT to the
    expansion side and back IN before a touch counts as a new one" rule
    -- just generalized from 2 touches to 3. Mutates
    position.zero_touch_count / rose_since_last_touch; exits the moment
    the count would reach 3, not before.

    Runs alongside the resting Hard Stop Loss (resting_hard_stop_price)
    as a second, independent layer -- this one bot-monitored and tighter
    (fires at true zero divergence), the resting one wider and
    broker-side, a backstop in case this check somehow doesn't catch a
    real blowup in time. Restores the original two-layer design from
    before the 2026-09-04 consolidation."""
    sign = expansion_direction_sign(position)
    live_divergence = position_spread(position, price_a, price_b) - position.entry_equilibrium
    on_equilibrium_side = (live_divergence * sign) <= 0

    if position.zero_touch_count == 0:
        if on_equilibrium_side:
            position.zero_touch_count = 1
        return False

    if not position.rose_since_last_touch:
        if not on_equilibrium_side:
            position.rose_since_last_touch = True
        return False

    # Bounced out since the last touch -- this touch counts as a new one.
    if on_equilibrium_side:
        position.zero_touch_count += 1
        position.rose_since_last_touch = False
        if position.zero_touch_count >= 3:
            return True
    return False


def take_profit_hard_stop_exit(position: Position, price_a: float, price_b: float) -> str | None:
    """SUPERSEDED 2026-09-04, less than a day after being built: the client
    sent worked Resting Take Profit/Hard Stop Loss formulas (How To Set
    Resting Take Profit.pdf) that translate the spread-trigger condition
    into a real price level on the single traded stock -- see
    resting_take_profit_price/resting_hard_stop_price below, which the
    entry flow now uses to place real resting STP/MIT orders again,
    instead of this bot-monitored check. Left in place, unused, rather
    than deleted -- same convention this codebase uses for every other
    superseded mechanism.

    Take Profit and Hard Stop, bot-monitored -- per the client
    2026-09-03: "Take Profit is market at Trigger... calculate when the
    spread gets to trigger, as we are currently doing... You simply close
    a single stock instead of two stocks." Same for Hard Stop.

    Rebuilt for the "Expansion Leg" pivot: with no Combo/BAG order left,
    there is no single instrument IBKR can rest a native STP/MIT order
    on for a two-symbol spread level, so this is checked live every
    cycle instead (same pattern as stop_loss_double_touch_exit, which
    made the identical move for the same reason on 2026-08-24) and the
    caller fires a plain MKT order on the one real leg when it returns.

    Same price levels bracket_order_prices has always used --
    take_profit_spread = entry_equilibrium + trigger (expansion side),
    hard_stop_spread = entry_equilibrium - 2*trigger (opposite side) --
    just evaluated against the live spread directly instead of translated
    into a resting combo order's price. Direction comes from
    expansion_direction_sign, not action_a alone -- see that function's
    own docstring for why action_a stopped being reliable on its own for
    single-leg positions.

    Returns "take_profit", "hard_stop", or None."""
    sign = expansion_direction_sign(position)
    live_spread = position_spread(position, price_a, price_b)
    take_profit_level = position.entry_equilibrium + sign * position.entry_trigger_level
    hard_stop_level = position.entry_equilibrium - sign * 2 * position.entry_trigger_level
    if (live_spread - take_profit_level) * sign >= 0:
        return "take_profit"
    if (live_spread - hard_stop_level) * sign <= 0:
        return "hard_stop"
    return None


def resting_take_profit_price(action: str, entry_price: float, trigger: float, entry_divergence: float) -> float:
    """Resting Take Profit price for the single traded stock -- per the
    client 2026-09-04 (How To Set Resting Take Profit.pdf), restoring a
    real resting order less than a day after take_profit_hard_stop_exit
    (bot-monitored) was built: "Why can it not be a resting order, when we
    can calculate when the spread gets to trigger... You simply close a
    single stock instead of two stocks."

    Translates the remaining gap to Trigger directly onto the stock's own
    price, entry_price +/- (Trigger - Divergence_entry) -- an approximation
    that assumes the REST of the expansion lands entirely on this one
    stock (the only way to express a two-symbol spread condition as a
    single-stock resting order price at all). Verified against the
    client's own worked example: entry=$100, Trigger=$3.50,
    Divergence_entry=$2.50 -> remaining gap $1.00 -> TP=$101 (BUY) /
    $99 (SELL)."""
    remaining_gap = trigger - entry_divergence
    return entry_price + remaining_gap if action == "BUY" else entry_price - remaining_gap


def wide_hard_stop_price(action: str, entry_price: float, trigger: float) -> float:
    """Hard Stop Loss @ ENTRY -- the FIRST stop placed the moment a position
    opens, per the client 2026-09-09: "Hard Stop Loss @ Entry you will use
    Trigger-13% + Trigger." Uses the SAME worst-case sizing distance
    position_sizing.risk_per_share was built from (minimum_gap_at_entry,
    Trigger reduced by the entry window's own 13% floor), not the real
    entry_divergence of this specific trade -- deliberately the widest
    (safest, furthest from entry_price) stop the sizing itself already
    assumed the position could tolerate, so it can be placed immediately
    without waiting on anything.

    Superseded moments later by resting_hard_stop_price (the "AFTER ENTRY"
    re-set, using the real entry_divergence) -- see that function's own
    docstring for why the two differ and the client's worked example.
    Verified against his own numbers: entry=$100, Trigger=$5.00 -> Minimum
    Gap at Entry 5.00x(1-0.13)=4.35 -> distance 4.35+5.00=9.35 ->
    Stop=$90.65 (BUY/long)."""
    gap = minimum_gap_at_entry(trigger, ENTRY_TRIGGER_DISTANCE_MIN_PCT)
    distance = gap + trigger
    return entry_price - distance if action == "BUY" else entry_price + distance


def resting_hard_stop_price(action: str, entry_price: float, trigger: float, entry_divergence: float) -> float:
    """Resting Hard Stop Loss price for the single traded stock @ 100%
    Mean Reversion x 2 -- per the client 2026-09-05 (second same-topic
    correction in two days): "Hard Stop Loss: 100% Mean Reversion x 2",
    with a full worked example. Reinstates the bot-monitored Stop Loss
    (stop_loss_triple_touch_exit) as a SEPARATE, independent layer
    alongside this one -- per the client the same day: "Re-instate: API
    Monitored Stop Loss" -- restoring the original two-layer design
    (tight bot-monitored primary + wide resting backstop) from before
    the 2026-09-04 consolidation that briefly made this the only
    mechanism.

    entry_price -/+ (Divergence_entry + Trigger) -- the "Remaining Gap To
    Zero + Trigger", a SUM of the two, not the earlier x2/-2x-Trigger
    formulas this superseded. Verified against both of the client's own
    worked examples: entry=$100, Trigger=$3.50, Divergence_entry=$2.50 ->
    distance $2.50+$3.50=$6.00 -> Stop=$94.00 (BUY/long) / $106.00
    (SELL/short)."""
    distance = entry_divergence + trigger
    return entry_price - distance if action == "BUY" else entry_price + distance


def pnl_stop_loss_price(
    action: str, entry_price: float, quantity: float,
    equity_with_loan_value: float, max_loss_pct: float,
) -> float:
    """The -5% PnL Stop Loss, as a resting price -- per the client
    2026-09-09: "Put that -5% PnL Stop Loss. It is Ibkr PnL. 5% of EWLV."

    A P&L threshold has to become a PRICE to sit in the OCA bracket
    alongside Take Profit, Hard Stop and Timed Exit, so the dollar loss
    budget is divided across the shares actually held:

        loss_budget   = EquityWithLoanValue x max_loss_pct
        per_share     = loss_budget / quantity
        stop price    = entry_price -/+ per_share

    This lands very close to resting_hard_stop_price by construction, and
    that is expected rather than redundant: the sizing formula picks the
    quantity so that hitting the Hard Stop costs about 5% of EWLV (the
    client's own worked example lands at 4.994%). The two only diverge
    when the quantity was NOT set by the risk limit -- a margin-limited
    size, a partial fill, or an equity figure that has moved since entry --
    and those are exactly the cases where a hard account-percentage
    backstop is worth having.

    Verified against the client's own numbers: EWLV 100,000 at 5% is a
    5,000 budget over 763 shares = 6.5531/share, so a 100.00 long stops at
    93.4469, against the Hard Stop's own 93.455."""
    if quantity <= 0:
        raise ValueError("cannot price a PnL stop for a zero or negative quantity")
    per_share = (equity_with_loan_value * max_loss_pct) / quantity
    return entry_price - per_share if action == "BUY" else entry_price + per_share


def check_exit(position: Position, now: datetime) -> str | None:
    """Returns the exit reason, or None if the position should stay open.
    Only covers the 3:59pm EOD flatten now. Take Profit and Hard Stop are
    native resting orders again as of 2026-09-04 (see
    resting_take_profit_price/resting_hard_stop_price and
    ibkr_client.place_single_stock_bracket), reconciled every cycle in
    main.py via bracket_order_filled() rather than recomputed here.
    take_profit_hard_stop_exit (bot-monitored, built 2026-09-03) is
    superseded -- see its own docstring. Maximum Hold Time
    used to be checked here as a script-fired exit (it can't be a resting
    order the way a price-based exit can -- it's a clock, not a price),
    but per the client 2026-08-24 it is now attached to the order itself
    as the Timed Exit's own goodAfterTime, so it fires at the broker even
    if this process is
    down -- MAX_HOLD_MINUTES is still the deadline used to compute that
    order's activation time (see ibkr_client.place_pair_bracket), just no
    longer polled here."""
    if now.time() >= EOD_FLATTEN_TIME:
        return "eod"
    return None


def news_stop_cancel(symbol_a: str, symbol_b: str, blacklisted: set[str]) -> bool:
    """NEWS STOP, pre-entry half -- per the client 2026-09-01: "IF a stock
    in a selected pair becomes an excluded symbol due to news, exit pair @
    MKT if a position. Or cancel selected pair, if not yet a position."
    True when either leg of a still-flat pair is in the live news/excluded
    blacklist. The exit-side twin of this same rule lives inline in
    main.py's run_check_cycle -- an open position needs to actually
    flatten at the broker, not just flip a flag, so it isn't a pure
    function the same way this half is."""
    return symbol_a in blacklisted or symbol_b in blacklisted


def bracket_order_prices(
    action_a: str, ratio_a: int, entry_equilibrium: float, entry_trigger_level: float,
) -> tuple[float, float]:
    """Native combo bracket price levels for the CLOSING order (the BAG
    contract with reversed leg actions vs. the entry). Returns
    (hard_stop_price, take_profit_price), both expressed in the closing
    combo's own price convention -- the number to hand IBKR as auxPrice on
    that reversed-leg BAG contract (both STP and MIT use auxPrice, not
    lmtPrice -- see ibkr_client.combo_bracket_child_order).

    Hard Stop ("Trigger x 2", narrowed from x3 2026-08-25) direction
    confirmed 2026-08-24 by the client directly: "Stop Loss is the opposite
    of Take Profit" -- the opposite side from Take Profit,
    HARD_STOP_TRIGGER_MULTIPLIER times the trigger level further than the
    old "0 Divergence" stop ever sat, i.e. a genuine reversal past
    equilibrium into real loss territory, not a backstop on the expansion
    side.

    Derivation: unrealized_pnl = qty_a * sign * (spread - entry_spread),
    sign = +1 for a "BUY A" entry, -1 for "SELL A" (this module's own
    unrealized_pnl()). The CLOSING combo's own "BUY" price for a
    reversed-leg BAG is -ratio_a * spread * sign (same convention this
    project has always used to make a combo order's own price track
    unrealized_pnl exactly).

    Per the client 2026-08-19 ("0 relative to equilibrium at entry. 100%
    mean-reversion"): both targets are measured from entry_equilibrium, the
    equilibrium AS OF ENTRY (fixed for the life of the position), not a
    live-recomputed one -- which is what makes a real resting bracket
    order possible again. Take Profit is entry_equilibrium plus the
    trigger level in the direction the spread was already expanding
    (sign*entry_trigger_level); Stop Loss is entry_equilibrium plus a
    STOP_LOSS_DIVERGENCE_TOLERANCE cushion in that same direction -- the
    spread having round-tripped back to within a cent of where it started.

    As a position loses money the closing price RISES (confirmed against
    unrealized_pnl algebraically), so the stop loss is a BUY-action STP
    (triggers a market order once price rises to auxPrice). As it gains,
    the closing price FALLS, so take profit is a BUY-action MIT (per the
    client 2026-08-19/20: "Take Profit = MKT @ Trigger" -- triggers a
    market order once price falls to touch auxPrice, not a resting LMT
    that could simply never fill if the market gaps past it) -- both
    consistent with combo_market_order's "action is always BUY, direction
    lives in the legs" convention."""
    sign = 1 if action_a == "BUY" else -1
    take_profit_spread = entry_equilibrium + sign * entry_trigger_level
    # Reversal past equilibrium into real loss territory --
    # HARD_STOP_TRIGGER_MULTIPLIER (2) times the trigger level on the
    # OPPOSITE side from Take Profit, not further out on the expansion
    # side. This is what makes it behave like a stop loss: the old "0
    # Divergence" stop fired the moment the spread merely reverted BACK TO
    # equilibrium; this fires only if it keeps going, past equilibrium,
    # another HARD_STOP_TRIGGER_MULTIPLIER trigger-lengths into the loss
    # direction.
    hard_stop_spread = entry_equilibrium - sign * (entry_trigger_level * HARD_STOP_TRIGGER_MULTIPLIER)
    take_profit_price = -ratio_a * take_profit_spread * sign
    hard_stop_price = -ratio_a * hard_stop_spread * sign
    # Rounded to the cent -- confirmed live 2026-08-20: an unrounded price
    # (many decimal places, from multiplying entry_equilibrium/
    # entry_trigger_level through by ratio_a) got both bracket orders
    # rejected outright by IBKR with error 110 ("price does not conform to
    # the minimum price variation for this contract"), leaving the
    # position with neither a Take Profit nor a Stop Loss ever actually
    # placed, and nothing in the code checking that they'd been rejected.
    return round(hard_stop_price, 2), round(take_profit_price, 2)


def record_exit(daily_state: DailyState, cumulative_realized_pnl_today: float, elv_at_925: float) -> None:
    """Halts trading for the day once cumulative realized P&L reaches the
    daily loss cap. Per the client: "Maximum Loss for the Day = 2% of
    Equity With Loan Value at 9:25am ET... IF 2% PnL Stop Loss triggers, no
    more trading for the day." Independent of trades_taken_today/
    reentry_locked_out -- this can halt the day even mid-way through a
    still-allowed re-entry, e.g. a losing second trade after an early
    Take Profit."""
    if cumulative_realized_pnl_today <= -STOP_LOSS_PCT * elv_at_925:
        daily_state.trading_halted = True


def reentry_allowed(exit_reason: str, now: datetime) -> bool:
    """Per the client 2026-08-19: "IF Take Profit is Done, no re-entry
    after 11am ET. If Take Profit is Done, we can re-enter, until 11am.
    Maximum Trades per day = 2." Closing via Take Profit before the cutoff
    is the ONLY exit that keeps the door open for another entry -- Stop
    Loss, the Maximum Hold Time, News Stop, EOD, and a failed bracket
    placement all end the day outright, same as the original "one trade
    per day, no re-entry." The MAX_TRADES_PER_DAY cap in
    entry_eligible still applies on top of this regardless."""
    return exit_reason == "take_profit" and now.time() < REENTRY_CUTOFF_TIME


# Real-Time Beta Destruction Stop Loss (BetaBreakdownMonitor,
# beta_drift_exceeded, and their five-condition drift threshold) removed
# 2026-08-22 per the client: "Beta Destruction Stop Loss is the only thing
# that needs to be removed" -- stays removed even now that beta is live
# again (2026-08-24 rollback, see main.py's PairContext.beta_estimator).
# The client asked specifically for THIS mechanism to go, distinct from
# beta being frozen or not; nothing since has asked for it back.
