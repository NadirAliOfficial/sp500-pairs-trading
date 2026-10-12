"""Quantity formula from the client's spec (corrected 2026-08-08, halved
2026-08-28 -- "(EquityWithLoanValue -2%) /2) / Pair Initial Margin ...
Confirmed. Use /2"):

    (EquityWithLoanValue -2%) / 2 / Pair Initial Margin
    Round decimals down to zero
    EquityWithLoanValue <= 9:29am ET EquityWithLoanValue
    Capital Per Leg is determined by Beta (Hedge Ratio)
    Fractional Shares = False

"Pair Initial Margin" is IBKR's own whatIf-quoted margin for the COMPLETE
combo position at a given ratio (ratio_a shares of leg A + ratio_b shares
of leg B) -- not margin(long) + margin(short) computed from two independent
single-leg whatIf probes and summed. The client was explicit about this
2026-08-08: "the margin of the entire paired position, not simply
margin(long) + margin(short)." Confirmed live 2026-08-07 this distinction
is real, not academic -- EME/PWR was sized off two summed single-leg
margins, IBKR then rejected the real combo order for needing $411,601 of
margin against $393,317 available, because the combo's real margin
treatment isn't a linear sum of its legs' standalone margins.

Beta determines the hedge ratio (and so the two legs' integer quantity
ratio, via integer_leg_ratio below); IBKR's own whatIf on that exact combo
then determines the real initial margin for the pair -- see
ibkr_client.get_pair_initial_margin for the whatIf call itself. This module
stays pure/no I/O; the margin figure is supplied by the caller.

Fractional Shares was flipped from True to False once the client confirmed
both legs are placed as a single IBKR Guaranteed BAG combo order -- combo
legs only accept whole-number ratios (IBKR's own ComboLeg.ratio field is
typed int), so an arbitrary hedge ratio like 2.4478 can't be carried exactly.
Instead it's approximated as 1 share of the smaller-coefficient leg against
the hedge ratio rounded UP for the other (see integer_leg_ratio's own
2026-09-02 history), and both legs' quantities come out as whole-number
multiples of that ratio --
"Capital Per Leg is determined by Beta" now means the ratio between the two
integer leg sizes, not a literal fractional share count.
"""
import math

# Superseded 2026-09-02 as the precision cap on a Fraction search (see
# integer_leg_ratio's own history below) -- kept now only as the sanity
# bound past which a hedge ratio is too extreme to represent as "1 share
# of one leg, N of the other" at all, matching the existing entry-path
# zero-leg guard's own cutoff.
MAX_RATIO_DENOMINATOR = 20


def effective_sizing_equity(snapshot_at_929: float, live_equity: float) -> float:
    """Per the spec line "EquityWithLoanValue <= 9:29am ET EquityWithLoanValue":
    the equity used for sizing is capped at the 9:29am snapshot but follows
    the live value down if the account loses ground during the session.

    So intraday gains are never leveraged into bigger positions, while
    intraday losses do shrink them. Freezing the snapshot outright (the
    previous behaviour) would keep sizing off the morning's equity even
    after the account had fallen well below it."""
    return min(snapshot_at_929, live_equity)


def compute_single_stock_quantity(equity_with_loan_value: float, single_stock_initial_margin: float) -> int:
    """SUPERSEDED 2026-09-05 by margin_limit_position_size/
    risk_limit_position_size/final_position_size below -- the client
    replaced this single-formula sizing with a two-step "smaller of a
    margin limit and a risk limit" system the same day ("Quantity
    Formula" rebuild, sections A-D). Left in place, unused, same
    convention this codebase uses for every superseded mechanism.

    (EquityWithLoanValue - 2%) / 2 / Single Stock Initial Margin, floored
    to a whole share -- per the client 2026-09-03's "Expansion Leg" pivot
    away from a hedged two-leg trade to a single-stock directional one
    ("You don't need Hedge Ratio anymore"). No hedge ratio, no capital
    split between two legs -- confirmed the same day: "Fractional Shares =
    False," so this is a plain whole-share floor division, same -2%/÷2
    formula as the pair version (compute_quantity) with a single-leg
    whatIf margin in place of the pair margin.

    Briefly changed to /3 the same evening ("I am dividing by 3"), reverted
    back to /2 the next day, per the client 2026-09-04: "I am dividing by 2
    as usual (EquityWithLoanValue -2%) /2) / Single Stock Initial Margin.\""""
    total_capital = equity_with_loan_value * 0.98 / 2
    return math.floor(total_capital / single_stock_initial_margin)


# Per the client 2026-09-05's sizing rebuild -- "Calculate Max Loss IF Hard
# Stop Loss is fixed at EWLV x 5%".
MAX_LOSS_PCT = 0.05


def margin_limit_position_size(equity_with_loan_value: float, single_stock_initial_margin: float) -> int:
    """(A) Margin Limit Position Size = EquityWithLoanValue / Single Stock
    Initial Margin, floored to a whole share.

    The -2% haircut DOES apply -- per the client 2026-09-05, correcting
    an earlier reading of his own worked example (100,000/29.48=3,392.13,
    which has no visible 2% deduction, initially taken at face value and
    confirmed when asked directly): "-2% is there. Look at (A) above."
    The example itself was evidently just illustrative/rounded rather
    than the exact computation; the formula's own written text
    ("(EquityWithLoanValue -2%) / Single Stock Initial Margin") is
    authoritative. With the haircut applied, 100,000 -> 98,000/29.48=
    3,324 (floored), not 3,392 -- the Final Position Size in the client's
    full example is unaffected either way (still 763, since the Risk
    Limit is the binding constraint there regardless of which Margin
    Limit value is used).

    Only the /2 halving from the old sizing formula is gone -- the -2%
    haircut survives. The other real risk control is
    risk_limit_position_size below, not a blanket halving of the
    margin-based figure."""
    return math.floor((equity_with_loan_value * 0.98) / single_stock_initial_margin)


def minimum_gap_at_entry(trigger: float, entry_trigger_distance_min_pct: float) -> float:
    """(B)(1) Trigger reduced by the entry window's own minimum distance
    (13%) -- the worst-case (largest) entry_divergence the 13%-25% entry
    window allows for this trigger, per the client 2026-09-05: "Calculate
    Minimum Gap at Entry (Trigger - 13%)." entry_trigger_distance_min_pct
    is a parameter rather than imported directly, to keep this module
    free of any dependency on risk_manager.py (its own docstring: "This
    module stays pure/no I/O") -- callers should pass
    risk_manager.ENTRY_TRIGGER_DISTANCE_MIN_PCT so the two stay in sync.

    Verified against the client's own worked example: 3.50-13%=3.045."""
    return trigger * (1 - entry_trigger_distance_min_pct)


def risk_per_share(trigger: float, minimum_gap_at_entry_value: float) -> float:
    """(B)(2) "Calculate Risk Per Share: Remaining Gap To Zero + Trigger"
    -- the same sum resting_hard_stop_price's distance uses for the real
    Hard Stop Loss price, but computed AHEAD of the actual entry using
    the worst-case Minimum Gap at Entry (above) rather than the real,
    not-yet-known entry_divergence of the specific trade about to be
    placed -- a conservative sizing assumption, not the real stop price
    itself. Verified: 3.045+3.50=6.545."""
    return minimum_gap_at_entry_value + trigger


def risk_limit_position_size(equity_with_loan_value: float, risk_per_share_value: float) -> int:
    """(C) Risk Limit Position Size = Max Loss / Risk Per Share, floored,
    where Max Loss = EquityWithLoanValue x 5% (MAX_LOSS_PCT). Verified
    against the client's own worked example: Max Loss=100,000x5%=5,000;
    5,000/6.545=763.942 -> 763 shares."""
    max_loss = equity_with_loan_value * MAX_LOSS_PCT
    return math.floor(max_loss / risk_per_share_value)


def final_position_size(margin_limit: int, risk_limit: int) -> int:
    """(D) Final Position Size = the SMALLER of the Margin Limit and Risk
    Limit position sizes. Verified: min(3392, 763)=763."""
    return min(margin_limit, risk_limit)


def compute_final_single_stock_quantity(
    equity_with_loan_value: float, single_stock_initial_margin: float,
    trigger: float, entry_trigger_distance_min_pct: float,
) -> int:
    """Ties sections (A)-(D) together into the one call main.py's entry
    flow actually needs -- see each of the individual functions above
    for their own worked-example verification. Pass
    risk_manager.ENTRY_TRIGGER_DISTANCE_MIN_PCT for
    entry_trigger_distance_min_pct."""
    margin_limit = margin_limit_position_size(equity_with_loan_value, single_stock_initial_margin)
    gap = minimum_gap_at_entry(trigger, entry_trigger_distance_min_pct)
    risk_limit = risk_limit_position_size(equity_with_loan_value, risk_per_share(trigger, gap))
    return final_position_size(margin_limit, risk_limit)


def beta_adjusted_position_size(risk_limit: int, normalized_leg_ratio: float) -> int:
    """(G) Entry Position Limit = Beta Adjusted Position Size -- per the
    client 2026-09-09: "Script will use Beta to determine Beta Adjusted
    Position Limit. Beta Adjusted Position Limit will be inside the Risk
    Position Limit... Entry Quantity = Beta Adjusted Position Limit. Not
    the Risk Limit."

    normalized_leg_ratio is the traded leg's own share of one normalized
    hedged unit, from screening.normalized_hedge_ratio -- 1.0 when the
    traded leg IS the reference leg, <= 1.0 otherwise. Scaling the Risk
    Limit by it therefore always lands inside the Risk Limit, which is the
    property the client states twice.

    Floored, per "Fractional Shares = False" -- the client's own example
    quotes 762.94 shares, which cannot actually be sent to IBKR; the
    floor is what the account can hold, and erring down keeps the result
    inside the Risk Limit rather than a hair over it."""
    if normalized_leg_ratio <= 0:
        return 0
    return min(risk_limit, math.floor(risk_limit * normalized_leg_ratio))


def equity_balance_after_leg(
    equity_with_loan_value: float, leg_quantity: int, leg_price: float,
) -> float:
    """(E) "EWLV less Stock A Quantity Cost = EWLV Balance" -- per the
    client's own 2026-09-09 clarification, which resolved whether "Quantity"
    meant the share count's COST or its margin: it is the cost. What is left
    is what leg B can be funded from (see opposite_leg_quantity below).
    Never negative."""
    return max(0.0, equity_with_loan_value - leg_quantity * leg_price)


def opposite_leg_quantity(
    equity_balance: float, opposite_price: float,
    traded_leg_quantity: int, opposite_leg_ratio: float,
) -> int:
    """(F) "Apply Beta to determine opposite stock shares quantity. Determine
    how many Stock B hedge ratio shares Balance EWLV can execute. Enter the
    applicable quantity."

    Two independent caps, take the smaller:
      * the BETA-implied size -- the traded leg's quantity carried across the
        normalized hedge ratio, i.e. what actually hedges the position; and
      * what the remaining EWLV Balance from (E) can actually pay for.

    The second cap is the point of (E) existing at all: leg A is sized first
    and consumes most of the account, so leg B gets whatever is genuinely
    left, not its full beta-implied size. Floored (Fractional Shares =
    False), and 0 if the balance cannot fund even one share -- the caller
    must treat 0 as "do not place the second leg" rather than sending a
    zero-quantity order."""
    if opposite_price <= 0 or opposite_leg_ratio <= 0:
        return 0
    beta_implied = math.floor(traded_leg_quantity * opposite_leg_ratio)
    affordable = math.floor(equity_balance / opposite_price)
    return max(0, min(beta_implied, affordable))


def capital_per_leg(equity_with_loan_value: float, hedge_ratio: float) -> tuple[float, float]:
    """Notional capital split between the two legs, proportional to the hedge
    ratio (1 share of A hedges hedge_ratio shares of B per the regression
    Spread = P_A - hedge_ratio*P_B, so leg B's dollar allocation is
    hedge_ratio times leg A's)."""
    total_capital = equity_with_loan_value * 0.98
    capital_a = total_capital / (1 + hedge_ratio)
    capital_b = total_capital - capital_a
    return capital_a, capital_b


def fractional_leg_ratio(
    hedge_ratio: float, action_a: str, max_ratio: float = MAX_RATIO_DENOMINATOR,
) -> tuple[float, float]:
    """One hedged unit's per-leg share counts: the SHORT leg is always
    exactly 1 whole share, the BUY leg carries the fractional remainder.

    Per the client 2026-09-03: "buying fractional shares = True, selling
    fractional shares = False... Hedge ratio for the minimum unit is
    always 1 share of the reference leg. Beta directly dictates which leg
    is fixed at 1 share (reference leg) and which leg scales." Confirmed
    the same day as a priority rule over the earlier
    smaller-coefficient definition of the reference leg: "If the short is
    1, then no problem. But if the short is 1.3 or something, then short
    has to become beta 1. That's the priority."

    Fractional shares cannot be sold short at IBKR, which is what forces
    the short side to be the whole-share leg rather than a free choice.

    Both of the client's own worked examples resolve through this:

      KO/PEP, beta 1.35, KO short (leg A):
        A is short -> A = 1 KO, B = 1.35 PEP.
      PEP/KO, KO short (leg B), natural ratio 1.35 short per 1 long:
        B is short -> divide both sides by 1.35 -> B = 1 KO,
        A = 1/1.35 = 0.7407 PEP.

    Returns (per_unit_a, per_unit_b). Collapses to (1.0, 0.0) -- an
    unhedged single leg the entry path refuses outright -- when the
    fractional side would exceed max_ratio, the same sanity bound the old
    whole-number version used for a pathologically extreme beta."""
    if hedge_ratio <= 0:
        return 1.0, 0.0
    if action_a.upper() == "SELL":  # leg A is the short -> A must be whole
        return (1.0, hedge_ratio) if hedge_ratio <= max_ratio else (1.0, 0.0)
    per_unit_a = 1.0 / hedge_ratio  # leg B is the short -> B must be whole
    return (per_unit_a, 1.0) if per_unit_a <= max_ratio else (1.0, 0.0)


def compute_quantity(
    equity_with_loan_value: float,
    per_unit_a: float,
    per_unit_b: float,
    pair_initial_margin: float,
) -> tuple[float, float]:
    """Returns (qty_leg_a, qty_leg_b) for the whole position -- the SHORT
    leg a whole number, the BUY leg fractional, both scaled from the same
    whole number of hedged units.

    per_unit_a/per_unit_b come from fractional_leg_ratio(hedge_ratio,
    action_a) -- exactly one of them is 1.0 (the short leg), the other
    carries the fraction. pair_initial_margin is IBKR's real whatIf-quoted
    margin for ONE such unit.

    "Round decimals down to zero" applies to the UNIT count, which is what
    keeps the short leg whole (units * 1.0). The buy leg is deliberately
    left fractional (units * the fractional side) -- rounding it would
    reintroduce exactly the hedge error this rebuild removed.

    Halved 2026-08-28, per the client: "(EquityWithLoanValue -2%) /2) /
    Pair Initial Margin ... Confirmed. Use /2." -- the /2 is the last step
    before dividing by margin, same place the -2% adjustment already sits."""
    total_capital = equity_with_loan_value * 0.98 / 2
    units = math.floor(total_capital / pair_initial_margin)
    return units * per_unit_a, units * per_unit_b


def combo_ratio_and_units(qty_a: int, qty_b: int) -> tuple[int, int, int]:
    """Recovers (ratio_a, ratio_b, units) from the two leg quantities compute_quantity
    produced -- ibkr_client.py needs this to build the BAG order's ComboLegs
    (whole-number ratios) and the parent order's totalQuantity (units)."""
    units = math.gcd(qty_a, qty_b)
    return qty_a // units, qty_b // units, units
