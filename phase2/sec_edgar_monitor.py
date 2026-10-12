"""Direct SEC EDGAR 8-K filing monitor, per the client 2026-08-23: "Direct
SEC EDGAR Integration: Supplement Gemini / ChatGPT news scans with direct
SEC EDGAR RSS feed monitoring for 8-K filings between 9:00 AM and 10:30 AM
ET to bypass web index latency" -- window corrected the same day to
9:00-11:30am ("Monitoring should be from 9-11:30am"), covering the full
possible holding period (entries end 10:30am, Maximum Hold Time is 60
minutes, so a position entered at 10:29 could still be open until 11:29).

A second, independent source alongside news_blacklist.py's Gemini/GPT
scan -- both write into the same ExcludedSymbols-backed "currently in the
news" store (news_flagged_symbols.json, see news_blacklist.
NEWS_FLAGGED_PATH), so nothing downstream (main.py's News Stop check)
needs to know which source flagged a symbol.

SEC EDGAR is public and free, no API key -- but strictly enforces a custom
User-Agent header (format "Company Name AdminContact@domain.com") and a 10
requests/second rate limit; requests without a valid User-Agent are
rejected with 403. Polled once per INTRADAY_SCAN_INTERVAL alongside the
existing news scan (one request per poll, nowhere near the rate limit, so
no separate throttling logic is needed here).

Both endpoints below were fetched live and inspected 2026-08-23 to confirm
the parsers match the real response shape: the "getcurrent" atom feed's
entry titles read "8-K - Company Name (0001234567) (Filer)", and
company_tickers.json maps {"0": {"cik_str": 1045810, "ticker": "NVDA",
"title": "NVIDIA CORP"}, ...}."""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from xml.etree import ElementTree as XmlET
from zoneinfo import ZoneInfo

import requests

from phase2.news_blacklist import ExcludedSymbols

logger = logging.getLogger("pairs_bot.sec_edgar_monitor")

ET = ZoneInfo("America/New_York")

# Required by SEC -- "the format Sample Company Name
# AdminContact@<sample company domain>.com" -- requests without a valid
# User-Agent are blocked with a 403. Client-provided identity, 2026-08-23.
SEC_USER_AGENT = "Example Capital admin@example.com"
EDGAR_CURRENT_8K_FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=100&output=atom"
)
EDGAR_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
REQUEST_TIMEOUT_SECONDS = 15

# Per the client 2026-08-23 (corrected same day from an initial 10:30am
# end): "Monitoring should be from 9-11:30am".
MONITOR_WINDOW_START = dtime(9, 0)
MONITOR_WINDOW_END = dtime(11, 30)

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
# The "getcurrent" feed's entry title is the only place carrying the
# filer's CIK: "8-K  - Company Name (0001234567) (Filer)".
_TITLE_CIK_RE = re.compile(r"\((\d{10})\)")


@dataclass
class FilingEntry:
    accession_number: str
    cik: str | None
    filed_at: datetime  # tz-aware, from the feed's own <updated>
    title: str


def parse_current_8k_feed(atom_xml: str) -> list[FilingEntry]:
    """Pure parser for the "getcurrent" atom feed -- no I/O, so this is
    unit tested against a real captured sample rather than a live fetch.
    Entries with no parseable <updated> timestamp are dropped outright
    (filings_in_window can't place them); a missing/unparseable CIK is
    kept (cik=None) since that's still logged, just never resolves to a
    ticker in flagged_tickers."""
    root = XmlET.fromstring(atom_xml)
    entries = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = entry.findtext("atom:title", default="", namespaces=_ATOM_NS)
        updated = entry.findtext("atom:updated", default="", namespaces=_ATOM_NS)
        entry_id = entry.findtext("atom:id", default="", namespaces=_ATOM_NS)
        if not updated:
            continue
        try:
            filed_at = datetime.fromisoformat(updated)
        except ValueError:
            continue
        cik_match = _TITLE_CIK_RE.search(title)
        entries.append(FilingEntry(
            accession_number=entry_id or title, cik=cik_match.group(1) if cik_match else None,
            filed_at=filed_at, title=title,
        ))
    return entries


def build_cik_to_ticker_map(tickers_json: dict) -> dict[str, str]:
    """SEC's company_tickers.json is keyed by an arbitrary row index, not
    CIK -- {"0": {"cik_str": 1045810, "ticker": "NVDA", ...}, ...}.
    Re-keyed here by CIK as a zero-padded 10-digit string, matching
    parse_current_8k_feed's own CIK format, so the two join directly."""
    return {
        str(row["cik_str"]).zfill(10): row["ticker"]
        for row in tickers_json.values()
        if "cik_str" in row and "ticker" in row
    }


def filings_in_window(
    entries: list[FilingEntry], now: datetime,
    window_start: dtime = MONITOR_WINDOW_START, window_end: dtime = MONITOR_WINDOW_END,
) -> list[FilingEntry]:
    """Keeps only TODAY's filings whose own filed_at time-of-day (ET)
    falls within the monitoring window -- the feed returns the most
    recent ~100 filings regardless of date/time, so this is the actual
    "9:00-11:30 only" gate."""
    today_et = now.astimezone(ET).date()
    kept = []
    for entry in entries:
        filed_et = entry.filed_at.astimezone(ET)
        if filed_et.date() == today_et and window_start <= filed_et.time() <= window_end:
            kept.append(entry)
    return kept


def flagged_tickers(entries: list[FilingEntry], cik_to_ticker: dict[str, str]) -> list[str]:
    """Maps each filing's CIK to a ticker, skipping filings with no CIK
    parsed or no match in the ticker map (an OTC/foreign filer not in
    SEC's own ticker file, or a parsing miss) -- these can't be excluded
    by symbol regardless."""
    return [
        cik_to_ticker[entry.cik] for entry in entries
        if entry.cik is not None and entry.cik in cik_to_ticker
    ]


def _headers() -> dict:
    return {"User-Agent": SEC_USER_AGENT}


def fetch_current_8k_filings() -> list[FilingEntry]:
    response = requests.get(EDGAR_CURRENT_8K_FEED_URL, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_current_8k_feed(response.text)


def fetch_cik_to_ticker_map() -> dict[str, str]:
    response = requests.get(EDGAR_TICKER_MAP_URL, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return build_cik_to_ticker_map(response.json())


@dataclass
class SecEdgarMonitorState:
    """Per-session state, held by the caller (main.py) across polls.

    cik_to_ticker is fetched once and reused all day -- it barely changes
    intraday and re-fetching a ~10MB JSON file every 1-minute poll would
    be wasteful. seen_accession_numbers holds every filing already
    processed (whether or not it fell in the monitoring window), so a
    filing still sitting in the feed's rolling ~100-entry window on a
    later poll isn't re-checked or re-flagged every single cycle."""
    cik_to_ticker: dict[str, str] = field(default_factory=dict)
    seen_accession_numbers: set = field(default_factory=set)


def run_edgar_scan(
    state: SecEdgarMonitorState, news_flagged: ExcludedSymbols, now: datetime | None = None,
) -> list[str]:
    """Entry point for the background poll (main.py), mirroring
    news_blacklist.run_intraday_scan's shape: returns the tickers newly
    flagged THIS call. Writes into news_flagged (the raw "currently in the
    news" signal, NEWS_FLAGGED_PATH by default) -- restructured 2026-08-28,
    per the client: "Excluded Symbols should only be populated AFTER a real
    News Stop Loss exit fires... The Symbol responsible for the News Stop
    Loss will become Excluded." main.py's News Stop check reads this
    against open positions and only THEN writes the client-visible Excluded
    Symbols list (excluded_symbols.json) -- see news_blacklist.
    NEWS_FLAGGED_PATH's own comment."""
    now = now or datetime.now(timezone.utc)
    if not state.cik_to_ticker:
        state.cik_to_ticker = fetch_cik_to_ticker_map()
    entries = fetch_current_8k_filings()
    new_entries = [e for e in entries if e.accession_number not in state.seen_accession_numbers]
    for entry in new_entries:
        state.seen_accession_numbers.add(entry.accession_number)
    tickers = flagged_tickers(filings_in_window(new_entries, now), state.cik_to_ticker)
    for ticker in tickers:
        news_flagged.add(ticker, now=now)
    return tickers
