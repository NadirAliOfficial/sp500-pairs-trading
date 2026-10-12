"""Unit tests for sec_edgar_monitor.py. parse_current_8k_feed is tested
against a real sample captured live from SEC EDGAR's "getcurrent" atom
feed 2026-08-23 (https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent
&type=8-K&...&output=atom), so the parser is verified against the real
response shape, not a guessed one. fetch_current_8k_filings/
fetch_cik_to_ticker_map make real network calls and are not exercised
here -- run_edgar_scan's orchestration is tested with those two mocked,
same pattern test_news_blacklist.py uses for run_daily_scan/
run_intraday_scan."""
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.news_blacklist import ExcludedSymbols
from phase2.sec_edgar_monitor import (
    FilingEntry, SecEdgarMonitorState, build_cik_to_ticker_map, filings_in_window,
    flagged_tickers, parse_current_8k_feed, run_edgar_scan,
)

# Trimmed to 3 entries from a real capture (2026-08-23) of SEC EDGAR's
# "getcurrent" atom feed for type=8-K -- the exact tags/attributes/escaping
# the live endpoint returns, not a hand-built approximation.
REAL_SAMPLE_FEED = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Latest Filings - Sun, 23 Aug 2026 03:10:06 EDT</title>
<link rel="alternate" href="/cgi-bin/browse-edgar?action=getcurrent"/>
<link rel="self" href="/cgi-bin/browse-edgar?action=getcurrent"/>
<id>https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent</id>
<author><name>Webmaster</name><email>webmaster@sec.gov</email></author>
<updated>2026-08-23T03:10:06-04:00</updated>
<entry>
<title>8-K - Peraso Inc. (0000890394) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/890394/000121390026092719/0001213900-26-092719-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-08-21 &lt;b&gt;AccNo:&lt;/b&gt; 0001213900-26-092719 &lt;b&gt;Size:&lt;/b&gt; 189 KB
&lt;br&gt;Item 5.02: Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers: Compensatory Arrangements of Certain Officers
</summary>
<updated>2026-08-21T17:30:52-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0001213900-26-092719</id>
</entry>
<entry>
<title>8-K - Flash Sports &amp; Media Holdings, Inc. (0001706524) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1706524/000121390026092714/0001213900-26-092714-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-08-21 &lt;b&gt;AccNo:&lt;/b&gt; 0001213900-26-092714 &lt;b&gt;Size:&lt;/b&gt; 195 KB
&lt;br&gt;Item 3.01: Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing
</summary>
<updated>2026-08-21T17:28:21-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0001213900-26-092714</id>
</entry>
<entry>
<title>8-K - K&amp;F GROWTH ACQUISITION CORP. II (0002029976) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/2029976/000121390026092693/0001213900-26-092693-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-08-21 &lt;b&gt;AccNo:&lt;/b&gt; 0001213900-26-092693 &lt;b&gt;Size:&lt;/b&gt; 255 KB
&lt;br&gt;Item 3.01: Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing
</summary>
<updated>2026-08-21T17:20:28-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0001213900-26-092693</id>
</entry>
</feed>
"""

# Trimmed real sample of company_tickers.json (also captured 2026-08-23).
REAL_SAMPLE_TICKERS_JSON = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "2": {"cik_str": 890394, "ticker": "PRSO", "title": "Peraso Inc."},
}


def test_parse_current_8k_feed_extracts_all_entries():
    entries = parse_current_8k_feed(REAL_SAMPLE_FEED)
    assert len(entries) == 3
    print("PASS: all three entries in the real sample feed are parsed")


def test_parse_current_8k_feed_extracts_cik_from_title():
    entries = parse_current_8k_feed(REAL_SAMPLE_FEED)
    ciks = {e.cik for e in entries}
    assert ciks == {"0000890394", "0001706524", "0002029976"}
    print("PASS: CIK correctly extracted from each entry's title, zero-padded to 10 digits")


def test_parse_current_8k_feed_extracts_accession_number_and_filed_at():
    entries = parse_current_8k_feed(REAL_SAMPLE_FEED)
    peraso = next(e for e in entries if e.cik == "0000890394")
    assert peraso.accession_number == "urn:tag:sec.gov,2008:accession-number=0001213900-26-092719"
    assert peraso.filed_at == datetime.fromisoformat("2026-08-21T17:30:52-04:00")
    print("PASS: accession number (from <id>) and filed_at (from <updated>) both parsed correctly")


def test_parse_current_8k_feed_handles_ampersand_escaping():
    """"K&F GROWTH..." and "Flash Sports & Media..." both contain a raw
    XML entity (&amp;) in the real feed -- confirms the parser doesn't
    choke on it and the title comes back unescaped."""
    entries = parse_current_8k_feed(REAL_SAMPLE_FEED)
    titles = {e.title for e in entries}
    assert "8-K - K&F GROWTH ACQUISITION CORP. II (0002029976) (Filer)" in titles
    assert "8-K - Flash Sports & Media Holdings, Inc. (0001706524) (Filer)" in titles
    print("PASS: ampersand-escaped titles are correctly unescaped")


def test_parse_current_8k_feed_empty_feed():
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_current_8k_feed(empty) == []
    print("PASS: a feed with no entries returns an empty list, not an error")


def test_build_cik_to_ticker_map_from_real_sample():
    mapping = build_cik_to_ticker_map(REAL_SAMPLE_TICKERS_JSON)
    assert mapping["0001045810"] == "NVDA"
    assert mapping["0000320193"] == "AAPL"
    assert mapping["0000890394"] == "PRSO"
    print("PASS: CIK-to-ticker map correctly re-keyed by zero-padded CIK")


def test_flagged_tickers_joins_filings_to_the_ticker_map():
    entries = parse_current_8k_feed(REAL_SAMPLE_FEED)
    mapping = build_cik_to_ticker_map(REAL_SAMPLE_TICKERS_JSON)
    tickers = flagged_tickers(entries, mapping)
    assert tickers == ["PRSO"]  # only Peraso's CIK is in the trimmed ticker map sample
    print("PASS: only filings whose CIK resolves in the ticker map are flagged")


def test_flagged_tickers_skips_entries_with_no_cik():
    unparseable = FilingEntry(accession_number="x", cik=None,
                               filed_at=datetime.now(timezone.utc), title="8-K - No CIK Here")
    assert flagged_tickers([unparseable], {"0000890394": "PRSO"}) == []
    print("PASS: an entry with no parsed CIK is skipped, not KeyError'd")


def test_filings_in_window_keeps_only_todays_filings_within_the_clock_window():
    in_window = FilingEntry(accession_number="a", cik="1", title="t",
                             filed_at=datetime.fromisoformat("2026-08-23T09:15:00-04:00"))
    before_window = FilingEntry(accession_number="b", cik="2", title="t",
                                 filed_at=datetime.fromisoformat("2026-08-23T08:59:00-04:00"))
    after_window = FilingEntry(accession_number="c", cik="3", title="t",
                                filed_at=datetime.fromisoformat("2026-08-23T11:31:00-04:00"))
    yesterday = FilingEntry(accession_number="d", cik="4", title="t",
                             filed_at=datetime.fromisoformat("2026-08-22T09:15:00-04:00"))
    now = datetime.fromisoformat("2026-08-23T09:20:00-04:00")
    kept = filings_in_window([in_window, before_window, after_window, yesterday], now)
    assert kept == [in_window]
    print("PASS: only today's filings inside the 9:00-11:30am ET window are kept")


def test_filings_in_window_boundaries_are_inclusive():
    at_start = FilingEntry(accession_number="a", cik="1", title="t",
                            filed_at=datetime.fromisoformat("2026-08-23T09:00:00-04:00"))
    at_end = FilingEntry(accession_number="b", cik="2", title="t",
                          filed_at=datetime.fromisoformat("2026-08-23T11:30:00-04:00"))
    now = datetime.fromisoformat("2026-08-23T12:00:00-04:00")
    kept = filings_in_window([at_start, at_end], now)
    assert kept == [at_start, at_end]
    print("PASS: exactly 9:00am and exactly 11:30am both count as inside the window")


def _fresh_excluded() -> ExcludedSymbols:
    path = Path(tempfile.mkdtemp()) / "excluded_symbols.json"
    return ExcludedSymbols(path=path)


def _filing_at(clock_time: str, cik: str = "0000890394", accession: str = "acc-1") -> FilingEntry:
    return FilingEntry(accession_number=accession, cik=cik, title="8-K - Peraso Inc.",
                        filed_at=datetime.fromisoformat(f"2026-08-21T{clock_time}-04:00"))


def test_run_edgar_scan_flags_and_excludes_new_tickers():
    excluded = _fresh_excluded()
    state = SecEdgarMonitorState(cik_to_ticker={"0000890394": "PRSO"})
    now = datetime.fromisoformat("2026-08-21T09:15:00-04:00")  # inside the 9:00-11:30am window
    with patch("phase2.sec_edgar_monitor.fetch_current_8k_filings") as mock_fetch:
        mock_fetch.return_value = [_filing_at("09:10:00")]
        newly_flagged = run_edgar_scan(state, excluded, now=now)
    assert newly_flagged == ["PRSO"]
    assert excluded.is_excluded("PRSO", now=now)
    print("PASS: run_edgar_scan flags a real filing and adds it to the shared ExcludedSymbols blacklist")


def test_run_edgar_scan_does_not_refetch_ticker_map_when_already_cached():
    excluded = _fresh_excluded()
    state = SecEdgarMonitorState(cik_to_ticker={"0000890394": "PRSO"})
    now = datetime.fromisoformat("2026-08-21T17:31:00-04:00")
    with patch("phase2.sec_edgar_monitor.fetch_current_8k_filings") as mock_fetch, \
         patch("phase2.sec_edgar_monitor.fetch_cik_to_ticker_map") as mock_map:
        mock_fetch.return_value = []
        run_edgar_scan(state, excluded, now=now)
    mock_map.assert_not_called()
    print("PASS: the ticker map is not re-fetched once state already has one cached")


def test_run_edgar_scan_fetches_ticker_map_when_state_is_empty():
    excluded = _fresh_excluded()
    state = SecEdgarMonitorState()  # no cached ticker map yet
    now = datetime.fromisoformat("2026-08-21T17:31:00-04:00")
    with patch("phase2.sec_edgar_monitor.fetch_current_8k_filings") as mock_fetch, \
         patch("phase2.sec_edgar_monitor.fetch_cik_to_ticker_map") as mock_map:
        mock_fetch.return_value = []
        mock_map.return_value = {"0000890394": "PRSO"}
        run_edgar_scan(state, excluded, now=now)
    mock_map.assert_called_once()
    assert state.cik_to_ticker == {"0000890394": "PRSO"}
    print("PASS: the ticker map is fetched once and cached onto state when missing")


def test_run_edgar_scan_does_not_reprocess_already_seen_filings():
    """Confirmed live 2026-08-23: the "getcurrent" feed returns the same
    ~100 most-recent filings on every poll until they age out, so without
    dedup the same filing would be re-flagged (and re-logged) every single
    1-minute cycle it stays in that window."""
    excluded = _fresh_excluded()
    state = SecEdgarMonitorState(cik_to_ticker={"0000890394": "PRSO"})
    now = datetime.fromisoformat("2026-08-21T09:15:00-04:00")
    with patch("phase2.sec_edgar_monitor.fetch_current_8k_filings") as mock_fetch:
        mock_fetch.return_value = [_filing_at("09:10:00")]
        first = run_edgar_scan(state, excluded, now=now)
        second = run_edgar_scan(state, excluded, now=now)
    assert first == ["PRSO"]
    assert second == []
    print("PASS: the same filing seen on a later poll is not re-flagged")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
