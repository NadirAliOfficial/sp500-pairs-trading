"""Daily 4pm ET post-trade report, per the client's 6-item spec:

    1. Mean spread divergence from the equilibrium spread, before the
       divergence gap closed inside 2 hours
    2. Mean duration for divergence gap closure
    3. Table of divergence gap open/close times
    4. Filter data with Half-Life < 60 minutes
    5. Maximum EquityWithLoanValue loss % before divergence gap closure
    6. Mean EquityWithLoanValue loss % before divergence gap closure
    7. PnL % at gap closure for each pair trade
    8. Duration for divergence gap closure for each pair trade
    9. Daily Qualifying Pairs -- everything that passed the screen, flagged
       with whether it is on the Market Entry List, so the client can review
       and decide what to trade next time

Report-building is pure and unit tested. send_report() uses the
Resend API (same service as the sibling SPY bot, but a separate account/key
for this project per the client's explicit instruction not to share the SPY
bot's key across projects) -- set RESEND_API_KEY in .env. RESEND_FROM
defaults to Resend's sandbox sender (onboarding@resend.dev), which only
reliably delivers to the Resend account's own verified email; once the
client verifies a domain in Resend, set RESEND_FROM to an address on that
domain so delivery to the recipient works from any account state. The
recipient itself is POST_TRADE_REPORT_TO in .env, not a literal here.
"""
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from phase1.screening import MAX_HALF_LIFE_MINUTES
import resend
from dotenv import load_dotenv

from phase2.market_entry_list import load_market_entry_list, pair_key
from phase2.spread_engine import passes_gap_condition

ET = ZoneInfo("America/New_York")
QUALIFIED_PAIRS_PATH = str(Path(__file__).resolve().parent.parent / "data" / "qualified_pairs.csv")

load_dotenv()

POST_TRADE_REPORT_TO = os.getenv("POST_TRADE_REPORT_TO", "")
HALF_LIFE_FILTER_MINUTES = 60


@dataclass
class ReportEvent:
    """One divergence event, annotated with its pair's half-life (a Phase 1
    per-pair property) so the report can apply the "Half-Life < 60 minutes"
    filter.

    qty_a is the position size the bot uses for this pair. It's required to
    make items 5/6 dimensionally correct: max_adverse_excursion is in spread
    dollars PER SHARE of leg A, so dividing it straight into account equity
    (as this used to) understated the loss % by a factor of the share count."""
    symbol_a: str
    symbol_b: str
    half_life_minutes: float
    open_time: object
    close_time: object
    max_spread: float
    duration: timedelta
    max_adverse_excursion: float
    qty_a: float = 0.0
    qty_b: float = 0.0
    action_a: str = ""
    action_b: str = ""
    # Which symbol action_a/qty actually describe -- "A" for a real
    # two-leg position (always, by construction) or a single-leg
    # "Expansion Leg" position (2026-09-03) trading symbol_a; "B" for a
    # single-leg position trading symbol_b, where qty_a is 0 and action_a
    # is repurposed to describe symbol_b instead. See
    # risk_manager.Position.traded_leg's own docstring -- mirrors it here
    # so the report can tell which leg is real without guessing from
    # qty_a/qty_b alone.
    traded_leg: str = "A"
    # How the trade actually closed -- "take_profit", "stop_loss", "eod", or
    # "beta_breakdown". Per the client 2026-08-18's entry/exit rewrite, Take
    # Profit now fires when the divergence EXPANDS to reach the trigger
    # ("Bot is taking profit on divergence expansion"), not on reversion --
    # outcome_label() below turns this into the report's own wording.
    exit_reason: str = ""
    # Per the client 2026-08-19/20: "Points to trigger @ Entry = 0.24,
    # Points past trigger @ Take Profit = 1.52" -- both fixed at entry, not
    # live. The first is how close to the trigger this trade was when it
    # opened (entry_trigger_level - entry_divergence); the second is the
    # trigger level itself, the fixed divergence Take Profit fires at.
    # Same fields already shown on the dashboard's open-position card.
    points_to_trigger_at_entry: float = 0.0
    take_profit_trigger_level: float = 0.0

    @property
    def traded_qty(self) -> float:
        """The TRADED leg's own quantity -- qty_a for a real two-leg
        position (traded_leg is always "A" there) or a single-leg
        position trading symbol_a; qty_b for a single-leg position
        trading symbol_b. See elv_loss_pct/pnl_pct_at_gap_closure, which
        both scale max_adverse_excursion/max_spread (dollars PER SHARE OF
        THE TRADED LEG) back to real account dollars by this -- using
        qty_a unconditionally there always evaluated to $0 for a
        traded_leg="B" event, silently zeroing out its reported drawdown
        and upside."""
        return self.qty_a if self.traded_leg == "A" else self.qty_b

    @property
    def buy_leg(self) -> str:
        if self.qty_a == 0 or self.qty_b == 0:  # single-leg position -- only one real leg
            symbol = self.symbol_a if self.traded_leg == "A" else self.symbol_b
            return f"{symbol} x{self.traded_qty:g}" if self.action_a == "BUY" else "(none)"
        symbol, qty = (self.symbol_a, self.qty_a) if self.action_a == "BUY" else (self.symbol_b, self.qty_b)
        return f"{symbol} x{qty:g}"

    @property
    def sell_leg(self) -> str:
        if self.qty_a == 0 or self.qty_b == 0:
            symbol = self.symbol_a if self.traded_leg == "A" else self.symbol_b
            return f"{symbol} x{self.traded_qty:g}" if self.action_a == "SELL" else "(none)"
        symbol, qty = (self.symbol_a, self.qty_a) if self.action_a == "SELL" else (self.symbol_b, self.qty_b)
        return f"{symbol} x{qty:g}"


def filter_by_half_life(events: list[ReportEvent], max_minutes: float = HALF_LIFE_FILTER_MINUTES) -> list[ReportEvent]:
    return [e for e in events if e.half_life_minutes < max_minutes]


def mean_divergence(events: list[ReportEvent]) -> float | None:
    if not events:
        return None
    return sum(e.max_spread for e in events) / len(events)


def mean_duration(events: list[ReportEvent]) -> timedelta | None:
    if not events:
        return None
    return sum((e.duration for e in events), timedelta()) / len(events)


def elv_loss_pct(event: ReportEvent, equity_with_loan_value: float) -> float:
    """Adverse excursion scaled to real account dollars (spread dollars per
    share x share count) before expressing it as a % of equity.

    Scaled by event.traded_qty, not qty_a unconditionally -- qty_a is 0
    for a single-leg "Expansion Leg" event trading symbol_b, which used
    to zero this out entirely regardless of the real drawdown."""
    if equity_with_loan_value == 0:
        return 0.0
    account_dollars = event.max_adverse_excursion * event.traded_qty
    return (account_dollars / equity_with_loan_value) * 100


def pnl_pct_at_gap_closure(event: ReportEvent, equity_with_loan_value: float) -> float:
    """Item 7 -- the gap the trade captures, as a % of account equity.

    max_spread is the peak divergence in spread dollars per share of the
    traded leg, so a position held from that peak until the gap closes
    back to equilibrium captures it in full. Scaled by event.traded_qty,
    same as elv_loss_pct and for the same reason (qty_a alone is 0 for a
    single-leg event trading symbol_b), so the two stay directly
    comparable: item 7 is the upside of the event, items 5/6 are the
    drawdown endured to get it."""
    if equity_with_loan_value == 0:
        return 0.0
    return (event.max_spread * event.traded_qty / equity_with_loan_value) * 100


EXIT_REASON_LABELS = {
    "take_profit": "Take Profit (Divergence Expansion)",
    # Restructured 2026-08-24: Stop Loss is now bot-monitored double-touch
    # logic (no longer a native bracket leg), Hard Stop and Timed Exit are
    # the two new native bracket legs that replaced its broker-side role --
    # see risk_manager.stop_loss_double_touch_exit / bracket_order_prices.
    "stop_loss": "Stop Loss (double touch)",
    "hard_stop": "Hard Stop (Trigger x 2)",
    "timed_exit": "Timed Exit (60 min)",
    "max_hold_time": "Maximum Hold Time (60 min)",  # kept for old report rows from before the 2026-08-24 restructure
    "eod": "End of Day Flatten",
    "news_stop": "News Stop",
    "bracket_placement_failed": "Bracket Order Failed (flattened immediately)",
    "bracket_abandoned": "Bracket Order Died Mid-Session (flattened immediately)",
}


def outcome_label(event: ReportEvent) -> str:
    """Item 10 -- states how the trade actually closed. Per the client
    2026-08-18's entry/exit rewrite, Take Profit fires when the divergence
    EXPANDS to reach the trigger level ("Bot is taking profit on divergence
    expansion"), the opposite of the old mean-reversion take-profit -- the
    label makes that explicit rather than leaving the reader to guess which
    direction "Take Profit" means here."""
    return EXIT_REASON_LABELS.get(event.exit_reason, event.exit_reason or "--")


def format_duration(duration: timedelta) -> str:
    """Item 8 -- a raw timedelta prints as "0 days 00:39:00", which is noise.
    Client asked for it stated in minutes, so minutes is always the headline
    figure, with the h/m breakdown alongside once it runs past an hour."""
    total_minutes = int(duration.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{total_minutes} min ({hours}h {minutes}m)"
    return f"{total_minutes} min"


def _neg_pct(value: float) -> str:
    """elv_loss_pct/max_elv_loss_pct/mean_elv_loss_pct all return a positive
    magnitude -- signed here, once, at display time, per the client: "Loss
    should have a negative sign: -0.01%." A true zero stays unsigned rather
    than printing a misleading "-0.00%"."""
    return f"{'-' if value else ''}{value:.2f}%"


def max_elv_loss_pct(events: list[ReportEvent], equity_with_loan_value: float) -> float | None:
    if not events:
        return None
    return max(elv_loss_pct(e, equity_with_loan_value) for e in events)


def mean_elv_loss_pct(events: list[ReportEvent], equity_with_loan_value: float) -> float | None:
    if not events:
        return None
    return sum(elv_loss_pct(e, equity_with_loan_value) for e in events) / len(events)


def _time_only(value: object) -> str:
    """Gap open/close as an ET clock time.

    The underlying bar index is UTC (that's what Databento returns), so these
    printed as "2026-07-30 14:18:00+00:00" -- a 10:18am ET event shown as
    14:18. Converted to ET here so the times match the trading day the rest
    of the report is describing."""
    if not hasattr(value, "strftime"):
        return str(value)
    if getattr(value, "tzinfo", None) is not None:
        value = value.astimezone(ET)
    return value.strftime("%H:%M:%S")


def qualifying_pairs_rows(contexts: list, equity_with_loan_value: float) -> str:
    """Item 9 -- every pair tracked today, flagged with whether it is already
    cleared for trading.

    Sourced from the actual day's tracked contexts (the statically-loaded
    pairs plus anything universe screening found live), not the static
    qualified_pairs.csv -- that file is a one-time offline snapshot and never
    reflects pairs discovered during the session, which used to make this
    table show a fraction of the day's real qualified-pair count (confirmed
    2026-08-06: 8 static rows vs. 46+ actually tracked that day).

    This is a review list, not a trading record: the client uses it to decide
    what to add to the Market Entry List next time, so it deliberately shows
    pairs the bot did NOT trade, including ones it is currently blocked from
    trading. Hurst isn't tracked live (only computed once, offline, in Phase
    1's screen) so it shows as "--" here rather than a stale number."""
    if not contexts:
        return ('<tr><td colspan="8" style="text-align:center;color:#777">'
                "No pairs qualified in today's screen</td></tr>")

    trading = load_market_entry_list() or set()
    rows = []
    for ctx in contexts:
        # Half-life < 60 minutes is a mandatory screening test, per the
        # client 2026-08-07: a pair whose live half-life has drifted past
        # that bar (confirmed same day: GE/RTX showed 4851 min against the
        # spec limit of 60) is excluded from this table entirely, not
        # flagged -- it did not qualify.
        if ctx.half_life_minutes is None or ctx.half_life_minutes >= MAX_HALF_LIFE_MINUTES:
            continue
        # 0.5% Allowed Divergence Gap test, per the client 2026-08-11:
        # "Divergence Watch should show only pairs that passed all the
        # tests, including" the gap condition -- same requirement applies
        # here, this table had the identical gap. That condition was
        # disabled from blocking entries per the client 2026-08-05 (a
        # separate, still-valid decision about the entry trigger itself),
        # but was never applied to what gets listed as "qualifying" here.
        if not passes_gap_condition(ctx.divergence_events, equity_with_loan_value,
                                     ctx.day_sizing_qty_a + ctx.day_sizing_qty_b):
            continue
        on_list = pair_key(ctx.symbol_a, ctx.symbol_b) in trading
        # "not on list" describes Market Entry List membership only, not
        # trading eligibility -- per the client 2026-08-05, any qualifying
        # pair executes whether it's on the list or not, universe screening
        # can and does trade pairs that were never manually added. The label
        # says so explicitly now rather than reading like a trading gate.
        mark = ('<span style="color:#1a7f37">&#10003; on Market Entry List</span>' if on_list
                else '<span style="color:#777">not on list (still eligible to trade)</span>')
        corr = f"{ctx.live_correlation:.4f}" if ctx.live_correlation is not None else "--"
        # The two daily-close Pearson readings the pair qualified on, shown
        # separately from the live intraday one above -- per the client
        # 2026-09-08, who read that single "Correlation" column as the 20 day
        # test and asked "How can 20 Day Pearson read 90?". They run in
        # OPPOSITE directions: 60 day >= 0.90, 20 day < 0.70.
        corr60 = f"{ctx.correlation_60d:.4f}" if ctx.correlation_60d is not None else "--"
        corr20 = f"{ctx.correlation_20d:.4f}" if ctx.correlation_20d is not None else "--"
        adf = f"{ctx.live_adf_pvalue:.4g}" if ctx.live_adf_pvalue is not None else "--"
        # excluded above, so by this point half_life_minutes is always a
        # real number under the 60-minute limit.
        hl = f"{ctx.half_life_minutes:.2f} min"
        # Per client 2026-08-07: shown so it is visible whether a listed
        # pair actually clears the current MIN_ENTRY_DIVERGENCE floor
        # ($1.50 as of 2026-09-12), rather than just trusting the filter
        # silently. Gates the trigger shown here directly, not a separate
        # check on live current -- see _bounded_trigger_level.
        max_div = f"${ctx.divergence_trigger_level:.4f}" if ctx.divergence_trigger_level is not None else "--"
        rows.append(
            f"<tr>"
            f"<td>{ctx.symbol_a}/{ctx.symbol_b}</td>"
            f"<td>{ctx.industry}</td>"
            f"<td>{corr60}</td>"
            f"<td>{corr20}</td>"
            f"<td>{corr}</td>"
            f"<td>{adf}</td>"
            f"<td>{hl}</td>"
            f"<td>--</td>"
            f"<td>{max_div}</td>"
            f"<td>{mark}</td>"
            f"</tr>"
        )
    return "".join(rows)


def divergence_working_html(contexts: list) -> str:
    """Item 11 -- "Show working": the 3 widest divergences in each tracked
    pair's lookback period, their mean, and the selected maximum (the
    pair's current trigger level). Per client 2026-08-06, shown separately
    from the Daily Qualifying Pairs table since it's about how the trigger
    number itself was derived, not the pair's screening status."""
    rows = []
    for ctx in contexts:
        events = ctx.divergence_events
        if not events:
            continue
        widest = sorted((e.max_spread for e in events), reverse=True)[:3]
        mean_widest = sum(widest) / len(widest)
        widest_str = ", ".join(f"{w:.4f}" for w in widest)
        selected_max = f"{ctx.divergence_trigger_level:.4f}" if ctx.divergence_trigger_level is not None else "--"
        rows.append(
            f"<tr><td>{ctx.symbol_a}/{ctx.symbol_b}</td>"
            f"<td>{widest_str}</td>"
            f"<td>{mean_widest:.4f}</td>"
            f"<td>{selected_max}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
    <h3 style="margin-top:28px">Show Working -- Divergence Trigger Calculation</h3>
    <p style="color:#777;font-size:13px">The 3 widest divergences found in each pair's lookback period,
    their mean, and the selected maximum (the pair's current entry trigger).</p>
    <table border="1" cellpadding="6" style="border-collapse:collapse">
      <tr><th>Pair</th><th>3 Widest Divergences</th><th>Mean of 3 Widest</th><th>Selected Maximum</th></tr>
      {"".join(rows)}
    </table>
    """


def opening_30min_breakdown_html(breakdown: dict | None) -> str:
    """Real SPY price data (previous close, open, 30-minute mark, intraday
    high), per client 2026-08-06 -- built to avoid a repeat of the market
    halt's Gemini-search inaccuracy, so this section is real data end to end,
    not a model's account of the morning."""
    if breakdown is None:
        return ""
    b = breakdown
    return f"""
    <h3 style="margin-top:28px">Opening 30-Minute Breakdown</h3>
    <p style="color:#777;font-size:13px">SPY used as the index proxy, real price data.</p>
    <table border="1" cellpadding="6" style="border-collapse:collapse">
      <tr><td>Previous Close</td><td>{b['previous_close']:.2f}</td></tr>
      <tr><td>Open</td><td>{b['open']:.2f} ({b['open_pct']:+.2f}% from previous close)</td></tr>
      <tr><td>At 30 Minutes (10:00 AM ET)</td>
          <td>{b['thirty_min_price']:.2f} ({b['thirty_min_pct_prev_close']:+.2f}% from previous close / """ \
          f"""{b['thirty_min_pct_open']:+.2f}% from open)</td></tr>
      <tr><td>Intraday High</td><td>{b['intraday_high']:.2f} ({b['intraday_high_pct']:+.2f}% from previous close)</td></tr>
    </table>
    """


def build_report_html(events: list[ReportEvent], equity_with_loan_value: float, contexts: list = (),
                      breakdown: dict | None = None) -> str:
    filtered = filter_by_half_life(events)
    mean_div = mean_divergence(filtered)
    mean_dur = mean_duration(filtered)
    max_loss = max_elv_loss_pct(filtered, equity_with_loan_value)
    mean_loss = mean_elv_loss_pct(filtered, equity_with_loan_value)

    if filtered:
        rows = "".join(
            f"<tr>"
            f"<td>{e.symbol_a}/{e.symbol_b}</td>"
            f"<td>{_time_only(e.open_time)}</td>"
            f"<td>{_time_only(e.close_time)}</td>"
            f"<td>{format_duration(e.duration)}</td>"
            f"<td>{e.max_spread:.4f}</td>"
            f"<td>{e.points_to_trigger_at_entry:.2f}</td>"
            f"<td>{e.take_profit_trigger_level:.2f}</td>"
            f"<td>{pnl_pct_at_gap_closure(e, equity_with_loan_value):.2f}%</td>"
            f"<td>{_neg_pct(elv_loss_pct(e, equity_with_loan_value))}</td>"
            f"<td>{outcome_label(e)}</td>"
            f"<td>{e.buy_leg}</td>"
            f"<td>{e.sell_leg}</td>"
            f"</tr>"
            for e in filtered
        )
    else:
        rows = ('<tr><td colspan="12" style="text-align:center;color:#777">'
                "No trades executed today with a half-life under 60 minutes"
                "</td></tr>")

    return f"""
    <h2>Post-Trade Report</h2>
    <p>Mean spread divergence from equilibrium: {f'{mean_div:.4f}' if mean_div is not None else 'N/A'}</p>
    <p>Mean duration for divergence gap closure: {format_duration(mean_dur) if mean_dur is not None else 'N/A'}</p>
    <p>Maximum Intratrade Loss: {_neg_pct(max_loss) if max_loss is not None else 'N/A'}</p>
    <p>Mean Intratrade Loss: {_neg_pct(mean_loss) if mean_loss is not None else 'N/A'}</p>
    <p style="color:#777;font-size:13px">Rows shown are trades executed today, filtered to pairs
    with a half-life under 60 minutes. Times are ET. Outcome shows how the trade actually closed --
    Take Profit now fires on divergence EXPANSION reaching the trigger, not on reversion.</p>
    <table border="1" cellpadding="6" style="border-collapse:collapse">
      <tr>
        <th>Pair</th><th>Gap Open</th><th>Gap Close</th><th>Duration</th>
        <th>Entry Divergence</th><th>Points to Trigger @ Entry</th><th>Points past Trigger @ Take Profit</th>
        <th>PnL % at Closure</th><th>Intratrade Loss %</th>
        <th>Outcome</th><th>Buy Leg</th><th>Sell Leg</th>
      </tr>{rows}
    </table>

    <h3 style="margin-top:28px">Daily Qualifying Pairs</h3>
    <p style="color:#777;font-size:13px">Every pair that passed today's screen. Only pairs on the
    Market Entry List are traded; the rest are shown so they can be reviewed and added if wanted.</p>
    <table border="1" cellpadding="6" style="border-collapse:collapse">
      <tr>
        <th>Pair</th><th>Industry</th>
        <th>60 Day Pearson<br><span style="font-weight:normal;color:#777">needs &ge; 0.90</span></th>
        <th>20 Day Pearson<br><span style="font-weight:normal;color:#777">needs &lt; 0.70</span></th>
        <th>Live Correlation<br><span style="font-weight:normal;color:#777">intraday, 30 bar</span></th>
        <th>ADF p</th>
        <th>Half-Life</th><th>Hurst</th><th>Maximum Divergence</th><th>Status</th>
      </tr>{qualifying_pairs_rows(contexts, equity_with_loan_value)}
    </table>
    {divergence_working_html(contexts)}
    {opening_30min_breakdown_html(breakdown)}
    """


def send_report(html: str, to_addr: str = POST_TRADE_REPORT_TO,
                 subject: str = "Pairs Bot Post-Trade Report") -> None:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        raise RuntimeError("RESEND_API_KEY not set in .env -- need the client's Resend API key before this can send")

    resend.api_key = api_key
    return resend.Emails.send({
        "from": os.getenv("RESEND_FROM", "Pairs Bot <onboarding@resend.dev>"),
        "to": [to_addr],
        "subject": subject,
        "html": html,
    })
