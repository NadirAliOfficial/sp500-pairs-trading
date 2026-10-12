"""Unit tests for position_sizing.py against hand-computed examples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.position_sizing import (
    beta_adjusted_position_size, capital_per_leg, compute_final_single_stock_quantity,
    compute_quantity, equity_balance_after_leg, opposite_leg_quantity,
    compute_single_stock_quantity, effective_sizing_equity, final_position_size,
    fractional_leg_ratio, margin_limit_position_size, minimum_gap_at_entry,
    risk_limit_position_size, risk_per_share,
)

ENTRY_TRIGGER_DISTANCE_MIN_PCT = 0.13  # matches risk_manager's own constant


def test_sizing_equity_is_capped_at_the_929_snapshot():
    """Spec: "EquityWithLoanValue <= 9:29am ET EquityWithLoanValue" -- gains
    after 9:29 must not increase position size."""
    assert effective_sizing_equity(snapshot_at_929=400_000, live_equity=450_000) == 400_000
    print("PASS: intraday gains never raise the sizing equity above the 9:29am snapshot")


def test_sizing_equity_follows_the_account_down():
    """It can reduce -- sizing off the morning snapshot after the account has
    fallen would over-allocate against equity that no longer exists."""
    assert effective_sizing_equity(snapshot_at_929=400_000, live_equity=310_000) == 310_000
    print("PASS: intraday losses do reduce the sizing equity below the snapshot")


def test_sizing_equity_equal_values():
    assert effective_sizing_equity(snapshot_at_929=400_000, live_equity=400_000) == 400_000
    print("PASS: unchanged equity sizes off the snapshot exactly")


def test_fractional_leg_ratio_short_a_above_1():
    """Per the client 2026-09-03's KO/PEP example: "Base Ratio: Short 1 KO
    (Integer) : Buy 1.35 PEP (Fractional)." Leg A is short (action_a=SELL),
    hedge_ratio (B relative to A) is 1.35 -- A stays whole at 1, B carries
    the fraction."""
    per_unit_a, per_unit_b = fractional_leg_ratio(1.35, action_a="SELL")
    assert per_unit_a == 1.0
    assert abs(per_unit_b - 1.35) < 1e-9
    print("PASS: short leg A, beta 1.35 above 1 -> (1.0, 1.35), matching the KO/PEP example exactly")


def test_fractional_leg_ratio_short_b_below_1():
    """The mirror case from the same client example, PEP/KO: "If the Short
    is 1.35 shares to 1 long share... divide both sides of the ratio by
    1.35... Short 1 KO (Integer) : Buy 0.7407 PEP (Fractional)." Leg B is
    short here (action_a=BUY, so A is the buy leg) -- B stays whole at 1,
    A carries 1/1.35."""
    per_unit_a, per_unit_b = fractional_leg_ratio(1.35, action_a="BUY")
    assert abs(per_unit_a - 1 / 1.35) < 1e-4
    assert abs(per_unit_a - 0.7407) < 1e-3
    assert per_unit_b == 1.0
    print("PASS: short leg B mirrors to (0.7407, 1.0), matching the PEP/KO example exactly")


def test_fractional_leg_ratio_short_leg_always_whole_priority_rule():
    """Per the client 2026-09-03, explicit priority over the earlier
    smaller-coefficient rule: "If the short is 1, then no problem. But if
    the short is 1.3 or something, then short has to become beta 1.
    That's the priority." Confirms the short side is ALWAYS forced to
    1.0 regardless of which leg has the smaller coefficient -- here A is
    short with the LARGER coefficient (2.0), and A still ends up whole."""
    per_unit_a, per_unit_b = fractional_leg_ratio(2.0, action_a="SELL")
    assert per_unit_a == 1.0  # short leg, whole, even though it's the larger coefficient
    assert per_unit_b == 2.0
    print("PASS: the short leg is always forced whole, overriding which side has the smaller coefficient")


def test_fractional_leg_ratio_exact_whole_number():
    assert fractional_leg_ratio(2.0, action_a="SELL") == (1.0, 2.0)
    print("PASS: a whole-number hedge ratio needs no fractional remainder on the buy leg")


def test_fractional_leg_ratio_no_rounding():
    """Unlike the rejected whole-share rebuild ("6:7 is wrong... You can
    buy 1 share and hedge it"), the buy leg carries the EXACT hedge ratio,
    no rounding in either direction."""
    per_unit_a, per_unit_b = fractional_leg_ratio(1.1632653, action_a="SELL")  # 57/49
    assert per_unit_a == 1.0
    assert abs(per_unit_b - 1.1632653) < 1e-6  # exact, not rounded to 1.16 or 2
    print("PASS: the buy leg carries the hedge ratio exactly, no rounding")


def test_fractional_leg_ratio_collapses_to_a_zero_leg_at_extreme_beta():
    """Same sanity bound as the whole-share version this replaces --
    audit finding 2026-08-27, preserved through the 2026-09-03 rebuild.
    A pathologically extreme beta (one leg priced far more than the
    other) still collapses to a zero leg rather than trading a wildly
    wrong hedge. Extreme in EITHER direction: a tiny hedge_ratio with the
    short on the buy side (1/hedge_ratio blows up), or a huge hedge_ratio
    with the short on the sell side (hedge_ratio itself blows up)."""
    per_unit_a, per_unit_b = fractional_leg_ratio(0.0099, action_a="BUY")  # 1/0.0099 = 101 > 20
    assert (per_unit_a, per_unit_b) == (1.0, 0.0)
    per_unit_a, per_unit_b = fractional_leg_ratio(30.0, action_a="SELL")  # 30 > 20
    assert (per_unit_a, per_unit_b) == (1.0, 0.0)
    print("PASS: an extreme beta still collapses to a zero leg, refused at the entry path")


def test_fractional_leg_ratio_zero_or_negative_beta():
    assert fractional_leg_ratio(0.0, action_a="SELL") == (1.0, 0.0)
    assert fractional_leg_ratio(-0.5, action_a="SELL") == (1.0, 0.0)
    print("PASS: zero or negative beta collapses to a zero leg, same as before")


def test_compute_quantity_propagates_a_zero_leg_ratio_to_a_zero_quantity():
    """The other half of the same finding: compute_quantity faithfully
    multiplies through a zero ratio, so (1.0, 0.0) becomes (N, 0) -- N
    shares of leg A and nothing on leg B. main.py guards both the ratio
    and the resulting quantities before this can be traded."""
    qty_a, qty_b = compute_quantity(20_000, 1.0, 0.0, 500)
    assert qty_a > 0 and qty_b == 0
    print("PASS: a zero leg ratio produces a zero leg quantity, which the entry path must reject")


def test_compute_quantity_short_leg_whole_buy_leg_fractional():
    """Per the client 2026-09-03: fractional shares on the buy leg,
    whole numbers on the short leg. ELV=100,000 -> capital=98,000/2=49,000.
    per_unit_a=1.0 (short), per_unit_b=1.35 (buy) -> units=floor(49000/110)=445.
    qty_a=445 (whole), qty_b=445*1.35=600.75 (fractional, not rounded)."""
    qty_a, qty_b = compute_quantity(
        equity_with_loan_value=100_000, per_unit_a=1.0, per_unit_b=1.35,
        pair_initial_margin=110,
    )
    assert qty_a == 445.0
    assert qty_a == int(qty_a)  # short leg lands on a whole number
    assert abs(qty_b - 600.75) < 1e-9  # buy leg stays fractional, not rounded
    print("PASS: compute_quantity keeps the short leg whole and the buy leg exactly fractional")


def test_compute_quantity_units_still_floors():
    """"Round decimals down to zero" applies to the UNIT count -- confirms
    a non-exact division still floors the units (not the final fractional
    buy-leg quantity, which is a separate thing)."""
    qty_a, qty_b = compute_quantity(
        equity_with_loan_value=100_000, per_unit_a=1.0, per_unit_b=1.5,
        pair_initial_margin=747,
    )
    # capital=49,000 -> units=floor(49000/747)=65
    assert qty_a == 65.0
    assert qty_b == 97.5
    print("PASS: the unit count floors on an inexact division, same as before")


def test_capital_per_leg_splits_proportional_to_hedge_ratio():
    capital_a, capital_b = capital_per_leg(equity_with_loan_value=100_000, hedge_ratio=2.0)
    assert abs(capital_a - 98_000 / 3) < 1e-9
    assert abs(capital_b - 98_000 * 2 / 3) < 1e-9
    assert abs((capital_a + capital_b) - 98_000) < 1e-9
    print("PASS: capital_per_leg splits total capital 1:hedge_ratio and sums back to the full amount")


def test_compute_single_stock_quantity_hand_computed_example():
    """Per the client 2026-09-03's Expansion Leg pivot: single-stock
    sizing, no hedge ratio. Divisor briefly changed /2 -> /3 the same
    evening, reverted back to /2 the next day 2026-09-04: "I am dividing
    by 2 as usual." ELV=100,000 -> capital=98,000/2=49,000.
    units=floor(49000/110)=445."""
    qty = compute_single_stock_quantity(equity_with_loan_value=100_000, single_stock_initial_margin=110)
    assert qty == 445
    assert isinstance(qty, int)
    print("PASS: compute_single_stock_quantity matches hand-computed example")


def test_compute_single_stock_quantity_floors_on_an_inexact_division():
    qty = compute_single_stock_quantity(equity_with_loan_value=10_000, single_stock_initial_margin=98)
    # capital = 10,000 * 0.98 / 2 = 4,900; floor(4900/98) = 50
    assert qty == 50
    print("PASS: rounds decimals down to zero, same as the pair formula")


def test_two_percent_haircut_is_applied():
    qty_a, qty_b = compute_quantity(
        equity_with_loan_value=10_000, per_unit_a=1.0, per_unit_b=1.0,
        pair_initial_margin=49,
    )
    # capital = 10,000 * 0.98 / 2 = 4,900; ratio 1:1 -> units=floor(4900/49)=100
    assert qty_a == 100.0
    assert qty_b == 100.0
    print("PASS: the -2% EquityWithLoanValue haircut is applied before dividing by margin")


def test_half_sizing_is_applied():
    """2026-08-28, per the client: "(EquityWithLoanValue -2%) /2) / Pair
    Initial Margin ... Confirmed. Use /2." -- halves the previous formula's
    output, on top of (not instead of) the existing -2% haircut."""
    qty_a, qty_b = compute_quantity(
        equity_with_loan_value=10_000, per_unit_a=1.0, per_unit_b=1.0,
        pair_initial_margin=98,
    )
    # capital = 10,000 * 0.98 / 2 = 4,900; units=floor(4900/98)=50
    assert qty_a == 50.0
    assert qty_b == 50.0
    print("PASS: the /2 halving is applied on top of the -2% haircut")


def test_margin_limit_position_size_applies_the_2pct_haircut():
    """Per the client 2026-09-05: "-2% is there. Look at (A) above" --
    correcting an earlier reading of his own worked example, which
    computed 100,000/29.48=3,392.13 with no visible 2% deduction. The
    formula's own written text is authoritative: 100,000*0.98/29.48=
    3,324 (floored)."""
    assert margin_limit_position_size(100_000, 29.48) == 3324
    print("PASS: margin_limit_position_size applies the -2% haircut per the client's correction")


def test_minimum_gap_at_entry_hand_computed_example():
    gap = minimum_gap_at_entry(3.50, ENTRY_TRIGGER_DISTANCE_MIN_PCT)
    assert abs(gap - 3.045) < 1e-9
    print("PASS: minimum_gap_at_entry matches the client's own worked example (3.50-13%=3.045)")


def test_risk_per_share_hand_computed_example():
    assert abs(risk_per_share(3.50, 3.045) - 6.545) < 1e-9
    print("PASS: risk_per_share matches the client's own worked example (3.045+3.50=6.545)")


def test_risk_limit_position_size_hand_computed_example():
    # Max Loss = 100,000 x 5% = 5,000; 5,000/6.545=763.942 -> 763
    assert risk_limit_position_size(100_000, 6.545) == 763
    print("PASS: risk_limit_position_size matches the client's own worked example")


def test_final_position_size_takes_the_smaller_limit():
    assert final_position_size(margin_limit=3392, risk_limit=763) == 763
    assert final_position_size(margin_limit=100, risk_limit=763) == 100
    print("PASS: final_position_size takes whichever of the two limits is smaller")


def test_compute_final_single_stock_quantity_matches_the_clients_own_full_worked_example():
    """End-to-end: the client's own full worked example, sections A-D
    tied together, matching his stated Final Position Size of 763."""
    quantity = compute_final_single_stock_quantity(
        equity_with_loan_value=100_000, single_stock_initial_margin=29.48,
        trigger=3.50, entry_trigger_distance_min_pct=ENTRY_TRIGGER_DISTANCE_MIN_PCT,
    )
    assert quantity == 763
    print("PASS: compute_final_single_stock_quantity matches the client's own full worked example end to end")


def test_beta_adjusted_position_size_stays_inside_the_risk_limit():
    """(G), per the client 2026-09-09: "Beta Adjusted Position Limit will be
    inside the Risk Position Limit... Entry Quantity = Beta Adjusted Position
    Limit. Not the Risk Limit." His example has them nearly equal (763 ->
    762.94), which is a normalized ratio of essentially 1."""
    # Traded leg IS the reference leg -- ratio 1.0, so nothing is given up.
    assert beta_adjusted_position_size(763, 1.0) == 763
    # Traded leg is the scaling leg -- strictly inside the Risk Limit.
    assert beta_adjusted_position_size(763, 0.7407) == 565
    assert beta_adjusted_position_size(763, 0.7407) < 763
    print("PASS: beta adjusted position size is the risk limit scaled by the leg's normalized ratio, never above it")


def test_beta_adjusted_position_size_floors_the_clients_fractional_example():
    """The client quotes 762.94 shares, but Fractional Shares = False --
    762.94 cannot be sent to IBKR, so it floors to 762."""
    ratio = 762.94 / 763
    assert beta_adjusted_position_size(763, ratio) == 762
    print("PASS: a fractional beta adjusted size floors to whole shares (762.94 -> 762)")


def test_equity_balance_after_leg_uses_cost_not_margin():
    """(E), per the client's own 2026-09-09 clarification: "EWLV less Stock A
    Quantity Cost = EWLV Balance" -- the share count's COST."""
    assert equity_balance_after_leg(100_000, leg_quantity=763, leg_price=100.0) == 100_000 - 76_300
    print("PASS: EWLV Balance subtracts the traded leg's full cost")


def test_equity_balance_after_leg_never_goes_negative():
    assert equity_balance_after_leg(10_000, leg_quantity=500, leg_price=100.0) == 0.0
    print("PASS: EWLV Balance floors at zero rather than going negative")


def test_opposite_leg_quantity_takes_the_smaller_of_beta_and_affordable():
    """(F): the beta implied size, capped by what the (E) balance can pay for."""
    # Beta implied 500 x 0.7407 = 370; balance funds 23_700/50 = 474 -> beta binds.
    assert opposite_leg_quantity(23_700, opposite_price=50.0,
                                  traded_leg_quantity=500, opposite_leg_ratio=0.7407) == 370
    # Same beta size, but only 5_000 left -> 100 affordable -> the balance binds.
    assert opposite_leg_quantity(5_000, opposite_price=50.0,
                                  traded_leg_quantity=500, opposite_leg_ratio=0.7407) == 100
    print("PASS: opposite leg takes the smaller of its beta implied size and what the EWLV balance funds")


def test_opposite_leg_quantity_is_zero_when_the_balance_cannot_fund_one_share():
    """The caller must read 0 as "do not place the second leg" -- never send
    a zero quantity order."""
    assert opposite_leg_quantity(30.0, opposite_price=50.0,
                                  traded_leg_quantity=500, opposite_leg_ratio=0.74) == 0
    print("PASS: opposite leg quantity is 0 when the remaining balance cannot fund a single share")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
