"""Unit tests for post_trade_report.py's pure report-building functions.
send_report() needs real SMTP credentials the client hasn't provided yet and
is not exercised here."""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.post_trade_report import (
    ReportEvent, build_report_html, elv_loss_pct, filter_by_half_life, format_duration, max_elv_loss_pct,
    mean_divergence, mean_duration, mean_elv_loss_pct, pnl_pct_at_gap_closure,
)


def test_pnl_pct_at_gap_closure_scales_by_position_size():
    """Item 7. The gap captured (max_spread) in account terms, same scaling
    as the ELV loss so the upside and the drawdown are comparable."""
    elv = 100_000
    e = _event(30, max_spread=0.4, duration_minutes=10, mae=0.1, qty_a=250.0)
    assert pnl_pct_at_gap_closure(e, elv) == 0.1  # 0.4 * 250 / 100000 * 100
    print("PASS: PnL %% at gap closure scales max divergence by the position's share count")


def test_pnl_pct_at_gap_closure_handles_zero_equity():
    e = _event(30, max_spread=0.4, duration_minutes=10, mae=0.1, qty_a=250.0)
    assert pnl_pct_at_gap_closure(e, 0) == 0.0
    print("PASS: PnL %% returns 0 rather than dividing by zero equity")


def test_format_duration_states_minutes():
    """Item 8. A raw timedelta prints as "0 days 00:39:00"; the client asked
    for it stated in minutes."""
    assert format_duration(timedelta(minutes=39)) == "39 min"
    assert format_duration(timedelta(minutes=72)) == "72 min (1h 12m)"
    assert format_duration(timedelta(seconds=90)) == "1 min"
    print("PASS: durations state minutes, with the h/m breakdown once past an hour")


def test_gap_times_are_converted_from_utc_to_et():
    """The bar index is UTC (Databento), so gap times printed as
    '2026-07-30 14:18:00+00:00' -- a 10:18am ET event shown as 14:18."""
    from datetime import datetime, timezone
    from phase2.post_trade_report import _time_only
    utc_1418 = datetime(2026, 7, 30, 14, 18, tzinfo=timezone.utc)
    assert _time_only(utc_1418) == "10:18:00"  # EDT is UTC-4
    print("PASS: gap open/close times are converted from UTC to ET")


def _event(half_life, max_spread, duration_minutes, mae, a="TYL", b="WDAY", qty_a=1.0):
    return ReportEvent(
        symbol_a=a, symbol_b=b, half_life_minutes=half_life,
        open_time="09:35", close_time="09:45",
        max_spread=max_spread, duration=timedelta(minutes=duration_minutes),
        max_adverse_excursion=mae, qty_a=qty_a,
    )


def test_filter_by_half_life_excludes_slow_pairs():
    events = [_event(half_life=30, max_spread=0.5, duration_minutes=10, mae=100),
              _event(half_life=90, max_spread=0.5, duration_minutes=10, mae=100)]
    filtered = filter_by_half_life(events)
    assert len(filtered) == 1
    assert filtered[0].half_life_minutes == 30
    print("PASS: filter_by_half_life excludes pairs with half-life >= 60 minutes")


def test_mean_divergence_and_duration():
    events = [_event(30, 0.4, 10, 50), _event(30, 0.6, 20, 50)]
    assert mean_divergence(events) == 0.5
    assert mean_duration(events) == timedelta(minutes=15)
    print("PASS: mean_divergence and mean_duration average correctly across events")


def test_elv_loss_pct_calculations():
    events = [_event(30, 0.4, 10, mae=500), _event(30, 0.6, 20, mae=1500)]
    elv = 100_000
    assert max_elv_loss_pct(events, elv) == 1.5  # 1500*1 / 100000 * 100
    assert mean_elv_loss_pct(events, elv) == 1.0  # (0.5 + 1.5)/2
    print("PASS: max/mean EquityWithLoanValue loss %% computed correctly from max_adverse_excursion")


def test_elv_loss_pct_scales_by_position_size():
    """max_adverse_excursion is spread dollars PER SHARE of leg A -- dividing
    it straight into account equity (as this used to) understated the loss by
    the whole share count."""
    elv = 100_000
    per_share = _event(30, 0.4, 10, mae=2.0, qty_a=1.0)
    real_size = _event(30, 0.4, 10, mae=2.0, qty_a=250.0)
    assert max_elv_loss_pct([per_share], elv) == 0.002       # 2*1/100000*100
    assert max_elv_loss_pct([real_size], elv) == 0.5         # 2*250/100000*100
    print("PASS: ELV loss %% scales by the position's share count, not raw per-share spread dollars")


def test_traded_qty_for_single_leg_event_trading_symbol_b():
    """Regression guard: elv_loss_pct/pnl_pct_at_gap_closure used to
    multiply by qty_a unconditionally, which silently evaluated to $0 for
    a single-leg "Expansion Leg" event trading symbol_b (qty_a is 0
    there) -- always reporting zero drawdown/upside for those trades."""
    event = ReportEvent(
        symbol_a="TYL", symbol_b="WDAY", half_life_minutes=30,
        open_time="09:35", close_time="09:45",
        max_spread=0.4, duration=timedelta(minutes=10),
        max_adverse_excursion=2.0, qty_a=0.0, qty_b=250.0,
        action_a="BUY", traded_leg="B",
    )
    assert event.traded_qty == 250.0
    elv = 100_000
    assert elv_loss_pct(event, elv) == 0.5          # 2.0 * 250 / 100000 * 100
    assert pnl_pct_at_gap_closure(event, elv) == 0.1  # 0.4 * 250 / 100000 * 100
    print("PASS: elv_loss_pct/pnl_pct_at_gap_closure scale by the traded leg's own qty, not qty_a")


def test_buy_sell_leg_labels_for_single_leg_event():
    buy_event = ReportEvent(
        symbol_a="TYL", symbol_b="WDAY", half_life_minutes=30,
        open_time="09:35", close_time="09:45",
        max_spread=0.4, duration=timedelta(minutes=10),
        max_adverse_excursion=2.0, qty_a=0.0, qty_b=250.0,
        action_a="BUY", traded_leg="B",
    )
    assert buy_event.buy_leg == "WDAY x250"
    assert buy_event.sell_leg == "(none)"

    sell_event = ReportEvent(
        symbol_a="TYL", symbol_b="WDAY", half_life_minutes=30,
        open_time="09:35", close_time="09:45",
        max_spread=0.4, duration=timedelta(minutes=10),
        max_adverse_excursion=2.0, qty_a=100.0, qty_b=0.0,
        action_a="SELL", traded_leg="A",
    )
    assert sell_event.sell_leg == "TYL x100"
    assert sell_event.buy_leg == "(none)"
    print("PASS: buy_leg/sell_leg correctly label a single-leg event regardless of which symbol traded")


def test_empty_events_return_none_not_crash():
    assert mean_divergence([]) is None
    assert mean_duration([]) is None
    assert max_elv_loss_pct([], 100_000) is None
    assert mean_elv_loss_pct([], 100_000) is None
    print("PASS: report stats return None instead of crashing on a day with zero events")


def test_build_report_html_includes_all_six_items():
    events = [_event(30, 0.5, 10, mae=1000), _event(90, 0.9, 300, mae=5000)]  # second excluded by half-life filter
    html = build_report_html(events, equity_with_loan_value=100_000)
    assert "Mean spread divergence" in html
    assert "Mean duration" in html
    # Renamed from "Loss before mean-reversion" per the client: "You should
    # exclude 'before mean-reversion' Just say Intratrade Loss."
    assert "Maximum Intratrade Loss" in html
    assert "Mean Intratrade Loss" in html
    assert "before mean-reversion" not in html
    assert "TYL/WDAY" in html
    # events table: header + the one event that survives the half-life filter
    events_table = html.split("Daily Qualifying Pairs")[0]
    assert events_table.count("<tr>") == 2
    print("PASS: build_report_html includes all 6 spec items and applies the half-life filter to the table")


def test_build_report_html_shows_loss_with_negative_sign():
    """Per the client: "Loss should have a negative sign: -0.01%.\""""
    events = [_event(30, 0.5, 10, mae=1000)]
    html = build_report_html(events, equity_with_loan_value=100_000)
    assert "-1.00%" in html  # 1000*1/100_000*100
    print("PASS: Intratrade Loss is shown with a leading negative sign")


def test_build_report_html_shows_outcome():
    """Per the client 2026-08-18's entry/exit rewrite, Take Profit now fires
    on divergence EXPANSION reaching the trigger, not reversion -- the
    Outcome column states this explicitly rather than implying the old
    mean-reversion meaning."""
    take_profit = ReportEvent(
        symbol_a="TYL", symbol_b="WDAY", half_life_minutes=30,
        open_time="09:35", close_time="09:45", max_spread=0.5,
        duration=timedelta(minutes=10), max_adverse_excursion=100, qty_a=1.0,
        exit_reason="take_profit",
    )
    stop_loss = ReportEvent(
        symbol_a="GE", symbol_b="RTX", half_life_minutes=30,
        open_time="09:35", close_time="10:15", max_spread=0.5,
        duration=timedelta(minutes=40), max_adverse_excursion=100, qty_a=1.0,
        exit_reason="stop_loss",
    )
    html = build_report_html([take_profit, stop_loss], equity_with_loan_value=100_000)
    assert "Take Profit (Divergence Expansion)" in html
    assert "Stop Loss" in html
    print("PASS: build_report_html states how each trade actually closed")


def test_build_report_html_shows_points_to_trigger_fields():
    # Per the client 2026-08-19/20's own worked example: "Points to
    # trigger @ Entry = 0.24, Points past trigger @ Take Profit = 1.52".
    event = ReportEvent(
        symbol_a="TYL", symbol_b="WDAY", half_life_minutes=30,
        open_time="09:35", close_time="09:45", max_spread=0.5,
        duration=timedelta(minutes=10), max_adverse_excursion=100, qty_a=1.0,
        exit_reason="take_profit",
        points_to_trigger_at_entry=0.24, take_profit_trigger_level=1.52,
    )
    html = build_report_html([event], equity_with_loan_value=100_000)
    assert "Points to Trigger @ Entry" in html
    assert "Points past Trigger @ Take Profit" in html
    assert "0.24" in html
    assert "1.52" in html
    print("PASS: build_report_html shows the client's own worked example numbers for the trigger fields")


def _ctx(symbol_a, symbol_b, industry, half_life_minutes=11.74, day_sizing_qty=100.0):
    """A PairContext that clears qualifying_pairs_rows' own two filters
    (half-life under the limit, and a passing Allowed Divergence Gap --
    which needs a non-zero day_sizing_qty to be evaluable at all)."""
    import pandas as pd
    from phase2.main import PairContext
    ctx = PairContext(symbol_a=symbol_a, symbol_b=symbol_b, hedge_ratio=1.0,
                       spread_history=pd.Series(dtype=float), industry=industry)
    ctx.half_life_minutes = half_life_minutes
    ctx.day_sizing_qty_a = day_sizing_qty
    ctx.day_sizing_qty_b = day_sizing_qty
    ctx.live_correlation = 0.9417
    ctx.live_adf_pvalue = 0.000001
    ctx.divergence_trigger_level = 2.5
    return ctx


def test_qualifying_pairs_section_flags_what_is_actually_traded(tmp_csv=None):
    """Item 9. The client reviews this to decide what to add to the Market
    Entry List, so pairs the bot is NOT allowed to trade have to appear, and
    be visibly distinguishable from the ones that are.

    Rewritten 2026-08-27: this test (and the one below) had been calling
    the pre-2026-08-06 signature, which took a qualified_pairs.csv path.
    Commit 0ffd9a7 changed it to (contexts, equity_with_loan_value) --
    sourced from the day's real tracked contexts rather than the static
    one-time screen file -- without updating the tests, so both had been
    failing on a TypeError ever since. Audited 2026-08-27 and confirmed
    they concealed no defect: the only real call site, build_report_html,
    passes the current signature correctly.
    """
    from phase2 import post_trade_report as ptr

    ctx_on_list = _ctx(symbol_a="GE", symbol_b="RTX", industry="Aerospace & Defense")
    ctx_off_list = _ctx(symbol_a="ANET", symbol_b="LITE", industry="Communications Equipment")

    real = ptr.load_market_entry_list
    ptr.load_market_entry_list = lambda: {ptr.pair_key("GE", "RTX")}
    try:
        rows = ptr.qualifying_pairs_rows([ctx_on_list, ctx_off_list], 100_000.0)
    finally:
        ptr.load_market_entry_list = real

    assert "GE/RTX" in rows and "ANET/LITE" in rows, "both qualifying pairs must be listed"
    assert "on Market Entry List" in rows and "not on list" in rows
    assert rows.index("GE/RTX") < rows.index("on Market Entry List") < rows.index("ANET/LITE")
    print("PASS: qualifying pairs section lists every tracked pair and flags Market Entry List membership")


def test_qualifying_pairs_section_handles_no_tracked_pairs():
    from phase2.post_trade_report import qualifying_pairs_rows
    rows = qualifying_pairs_rows([], 100_000.0)
    assert "No pairs qualified in today's screen" in rows
    print("PASS: an empty context list degrades to a message instead of crashing the report")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
