"""Unit tests for news_blacklist.py. scan_market_news() itself makes real
Gemini network calls and is not exercised here -- these tests cover the pure
filtering logic (symbols_to_exclude) and the ExcludedSymbols TTL/persistence
logic using an injectable clock, mirroring how databento_source.py's
dividend-adjustment tests avoid hitting real yfinance/Databento."""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.genai import errors as genai_errors

from phase2.news_blacklist import (
    INTRADAY_NEWS_QUESTION, NEWS_QUESTION, ExcludedSymbols, NewsItem,
    _generate_with_retry, _is_fabricated_placeholder_response, run_daily_scan, run_intraday_scan,
    scan_market_news_openai, symbols_to_exclude,
)


def test_symbols_to_exclude_only_flags_positive_and_negative():
    items = [
        NewsItem(ticker="AAPL", sentiment="Positive", news_summary="beat earnings"),
        NewsItem(ticker="TSLA", sentiment="Negative", news_summary="recall announced"),
        NewsItem(ticker="MSFT", sentiment="Neutral", news_summary="no major news"),
    ]
    excluded = symbols_to_exclude(items)
    assert excluded == ["AAPL", "TSLA"]
    print("PASS: symbols_to_exclude flags Positive and Negative, skips Neutral")


def test_symbols_to_exclude_drops_malformed_tickers():
    """Audit finding 2026-08-27: these come straight from an LLM response,
    and since the 2026-08-26 rescoping news exclusion drives the News Stop
    EXIT -- a hallucinated ticker matching an open position's leg would
    market-flatten a real live trade."""
    items = [
        NewsItem(ticker="NVDA", sentiment="Negative", news_summary="real"),
        NewsItem(ticker="$$FAKE!!", sentiment="Negative", news_summary="hallucinated"),
        NewsItem(ticker="", sentiment="Negative", news_summary="empty"),
        NewsItem(ticker="   ", sentiment="Negative", news_summary="whitespace"),
        NewsItem(ticker="TOOLONGSYM", sentiment="Negative", news_summary="not a ticker"),
    ]
    assert symbols_to_exclude(items) == ["NVDA"]
    print("PASS: malformed/hallucinated tickers are dropped before reaching the blacklist")


def test_symbols_to_exclude_deduplicates_and_normalises():
    """Production logs regularly show the same name repeated in one scan."""
    items = [
        NewsItem(ticker="NVDA", sentiment="Negative", news_summary="a"),
        NewsItem(ticker="nvda", sentiment="Negative", news_summary="b"),
        NewsItem(ticker=" NVDA ", sentiment="Positive", news_summary="c"),
        NewsItem(ticker="BRK.B", sentiment="Negative", news_summary="dotted ticker is valid"),
    ]
    assert symbols_to_exclude(items) == ["NVDA", "BRK.B"]
    print("PASS: tickers are upper-cased, trimmed, and de-duplicated")


def test_symbols_to_exclude_empty_when_all_neutral():
    items = [NewsItem(ticker="KO", sentiment="Neutral", news_summary="quiet day")]
    assert symbols_to_exclude(items) == []
    print("PASS: symbols_to_exclude returns empty list when nothing is directional")


def _fresh_blacklist() -> ExcludedSymbols:
    tmp = Path(tempfile.mkdtemp()) / "excluded_symbols.json"
    return ExcludedSymbols(path=tmp)


def test_add_and_is_excluded_within_ttl():
    bl = _fresh_blacklist()
    now = datetime(2026, 7, 27, 9, 25, tzinfo=timezone.utc)
    bl.add("NVDA", now=now)
    assert bl.is_excluded("NVDA", now=now + timedelta(hours=23)) is True
    print("PASS: symbol stays excluded within the 24-hour TTL")


def test_self_purges_after_24_hours():
    bl = _fresh_blacklist()
    now = datetime(2026, 7, 27, 9, 25, tzinfo=timezone.utc)
    bl.add("NVDA", now=now)
    assert bl.is_excluded("NVDA", now=now + timedelta(hours=24, minutes=1)) is False
    print("PASS: symbol self-purges once the 24-hour TTL expires, no manual cleanup needed")


def test_purge_persists_to_disk():
    bl = _fresh_blacklist()
    now = datetime(2026, 7, 27, 9, 25, tzinfo=timezone.utc)
    bl.add("NVDA", now=now)
    bl.add("PFE", now=now)
    later = now + timedelta(hours=25)
    # touching the blacklist after expiry should purge NVDA/PFE from the file itself
    assert bl.active_symbols(now=later) == set()
    fresh_view = ExcludedSymbols(path=bl.path)
    assert fresh_view.active_symbols(now=later) == set()
    print("PASS: expired entries are purged from disk, not just filtered in memory")


def test_active_symbols_reflects_multiple_entries_with_different_ages():
    bl = _fresh_blacklist()
    now = datetime(2026, 7, 27, 9, 25, tzinfo=timezone.utc)
    bl.add("OLD", now=now - timedelta(hours=23))  # added yesterday, about to expire
    bl.add("NEW", now=now)
    assert bl.active_symbols(now=now) == {"OLD", "NEW"}
    assert bl.active_symbols(now=now + timedelta(hours=2)) == {"NEW"}
    print("PASS: active_symbols correctly reflects mixed-age entries as time passes")


def test_all_entries_exposes_expiry_datetimes_for_the_dashboard():
    bl = _fresh_blacklist()
    now = datetime(2026, 7, 28, 9, 25, tzinfo=timezone.utc)
    bl.add("NVDA", now=now)
    entries = bl.all_entries(now=now)
    assert entries == {"NVDA": now + timedelta(hours=24)}
    print("PASS: all_entries exposes symbol -> expires_at datetimes for the dashboard's table view")


def test_corrupted_blacklist_file_reads_as_empty_not_a_crash():
    """A corrupted/truncated excluded_symbols.json (e.g. a crash mid-write)
    used to raise json.JSONDecodeError straight out of is_excluded()/
    active_symbols(), with no handling at this layer -- every call site
    depends on an OUTER try/except to survive it, silently disabling the
    whole news blacklist for the rest of the session rather than failing
    loudly and self-healing on the next write."""
    bl = _fresh_blacklist()
    bl.path.parent.mkdir(parents=True, exist_ok=True)
    bl.path.write_text("{not valid json")
    now = datetime(2026, 7, 27, 9, 25, tzinfo=timezone.utc)
    assert bl.is_excluded("NVDA", now=now) is False
    assert bl.active_symbols(now=now) == set()
    print("PASS: a corrupted blacklist file reads as empty instead of raising")


def test_corrupted_blacklist_file_self_heals_on_next_write():
    bl = _fresh_blacklist()
    bl.path.parent.mkdir(parents=True, exist_ok=True)
    bl.path.write_text("{not valid json")
    now = datetime(2026, 7, 27, 9, 25, tzinfo=timezone.utc)
    bl.add("NVDA", now=now)
    assert bl.is_excluded("NVDA", now=now) is True
    print("PASS: the next successful add() overwrites the corrupted file with valid JSON")


def test_concurrent_add_from_two_threads_does_not_lose_either_update():
    """Independent audit finding, confirmed real: excluded_symbols.json is
    written by two separate OS processes in production (the dashboard's
    manual add/remove, and the bot's own automated Gemini/EDGAR scans) --
    an unlocked read-modify-write let a losing writer's save silently
    clobber a winning writer's just-added symbol. Simulated here with two
    real threads (a real fcntl lock is what matters in production since
    those are separate processes, not just separate threads, but the same
    lock serializes threads within one process too). A short sleep is
    injected between the read and the write to widen the race window
    enough to force a genuine interleave attempt -- without the fix this
    deterministically loses one add(); with the fix, both survive."""
    import threading
    import time as time_module

    bl = _fresh_blacklist()
    now = datetime(2026, 7, 27, 9, 25, tzinfo=timezone.utc)

    real_load = bl._load

    def slow_load(load_now):
        data = real_load(load_now)
        time_module.sleep(0.05)
        return data

    bl._load = slow_load

    t1 = threading.Thread(target=bl.add, args=("AAA",), kwargs={"now": now})
    t2 = threading.Thread(target=bl.add, args=("BBB",), kwargs={"now": now})
    t1.start()
    time_module.sleep(0.01)  # ensure t1 grabs the lock first
    t2.start()
    t1.join()
    t2.join()

    bl._load = real_load
    assert bl.active_symbols(now=now) == {"AAA", "BBB"}
    print("PASS: two concurrent add() calls from separate threads both survive under the file lock")


def test_generate_with_retry_recovers_from_a_transient_503():
    # simulates Gemini's own "currently experiencing high demand" error --
    # this exact 503 crashed the whole session before this retry existed
    attempts = {"count": 0}

    def flaky_generate_content(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise genai_errors.ServerError(503, {"error": {"message": "high demand"}})
        return "ok"

    fake_client = type("FakeClient", (), {"models": type("M", (), {"generate_content": staticmethod(flaky_generate_content)})()})()
    with patch("time.sleep"):
        result = _generate_with_retry(fake_client, model="x", contents="y")
    assert result == "ok"
    assert attempts["count"] == 3
    print("PASS: _generate_with_retry recovers from transient 503s instead of crashing the session")


def test_generate_with_retry_gives_up_after_max_attempts():
    def always_fails(**kwargs):
        raise genai_errors.ServerError(503, {"error": {"message": "high demand"}})

    fake_client = type("FakeClient", (), {"models": type("M", (), {"generate_content": staticmethod(always_fails)})()})()
    try:
        with patch("time.sleep"):
            _generate_with_retry(fake_client, retries=3, model="x", contents="y")
        assert False, "expected ServerError to propagate after exhausting retries"
    except genai_errors.ServerError:
        pass
    print("PASS: _generate_with_retry re-raises once retries are exhausted, doesn't retry forever")


def test_both_questions_cover_the_previous_premarket_session_to_now():
    """Spec: "Market News Scan will cover the previous pre-market session +
    now" -- stated for BOTH the 9:25am scan and the 5-minute intraday rescan.
    News from any point in that span is still inside the rolling dataset the
    entry trigger is built from, so those symbols must stay excluded. The
    originals ("before today's market open" / "today") each left most of that
    window uncovered."""
    for label, q in (("9:25am", NEWS_QUESTION), ("intraday", INTRADAY_NEWS_QUESTION)):
        low = q.lower()
        assert "previous pre-market session" in low, f"{label} question misses the start of the window"
        assert "yesterday" in low, f"{label} question doesn't reach back to yesterday"
        assert "now" in low, f"{label} question doesn't run up to now"
    print("PASS: both scans cover the previous pre-market session through to now")


def test_run_daily_scan_uses_the_premarket_question():
    bl = _fresh_blacklist()
    with patch("phase2.news_blacklist.scan_market_news") as mock_scan:
        mock_scan.return_value = [NewsItem(ticker="XOM", sentiment="Negative", news_summary="oil drop")]
        newly_excluded = run_daily_scan(bl)
    mock_scan.assert_called_once_with(NEWS_QUESTION)
    assert newly_excluded == ["XOM"]
    print("PASS: run_daily_scan asks the pre-market question")


def test_run_intraday_scan_uses_the_intraday_question():
    bl = _fresh_blacklist()
    with patch("phase2.news_blacklist.scan_market_news") as mock_scan:
        mock_scan.return_value = [NewsItem(ticker="AAPL", sentiment="Positive", news_summary="rally")]
        newly_excluded = run_intraday_scan(bl)
    mock_scan.assert_called_once_with(INTRADAY_NEWS_QUESTION)
    assert newly_excluded == ["AAPL"]
    print("PASS: run_intraday_scan asks the 'today' question, added 2026-07-28 for the every-5-minute rescan")


def test_scan_falls_back_to_openai_when_gemini_fails():
    # Per the client 2026-08-18: "If Gemini API is down, use ChatGPT API."
    bl = _fresh_blacklist()
    with patch("phase2.news_blacklist.scan_market_news") as mock_gemini, \
         patch("phase2.news_blacklist.scan_market_news_openai") as mock_openai:
        mock_gemini.side_effect = genai_errors.ServerError(503, {"error": {"message": "down"}})
        mock_openai.return_value = [NewsItem(ticker="TSLA", sentiment="Negative", news_summary="recall")]
        newly_excluded = run_daily_scan(bl)
    mock_gemini.assert_called_once_with(NEWS_QUESTION)
    mock_openai.assert_called_once_with(NEWS_QUESTION)
    assert newly_excluded == ["TSLA"]
    print("PASS: falls back to OpenAI when Gemini raises")


def test_scan_does_not_call_openai_when_gemini_succeeds():
    # "Don't use both at the same time."
    bl = _fresh_blacklist()
    with patch("phase2.news_blacklist.scan_market_news") as mock_gemini, \
         patch("phase2.news_blacklist.scan_market_news_openai") as mock_openai:
        mock_gemini.return_value = [NewsItem(ticker="XOM", sentiment="Negative", news_summary="oil drop")]
        run_daily_scan(bl)
    mock_openai.assert_not_called()
    print("PASS: OpenAI is never called while Gemini is working")


def test_is_fabricated_placeholder_response_detects_the_exact_pattern():
    # Confirmed live 2026-08-24/25, three separate times in one evening:
    # OpenAI's fallback returned this exact sequential AAA/BBB/CCC/...
    # pattern with a clean 200 OK on both calls, no error to catch.
    items = [NewsItem(ticker=t, sentiment="Neutral", news_summary="x")
             for t in ["AAA", "BBB", "CCC", "DDD"]]
    assert _is_fabricated_placeholder_response(items) is True
    print("PASS: detects the exact fabricated AAA/BBB/CCC/... sequence")


def test_is_fabricated_placeholder_response_false_for_real_tickers():
    items = [NewsItem(ticker=t, sentiment="Negative", news_summary="x")
             for t in ["TSLA", "NVDA", "AAPL"]]
    assert _is_fabricated_placeholder_response(items) is False
    print("PASS: real tickers never look like the fabricated placeholder sequence")


def test_is_fabricated_placeholder_response_false_when_empty():
    assert _is_fabricated_placeholder_response([]) is False
    print("PASS: an empty (genuinely no news) result is not treated as fabricated")


def test_is_fabricated_placeholder_response_false_for_a_near_miss():
    # A real ticker landing in AAA/BBB order by coincidence is essentially
    # impossible, but this proves the check is exact, not fuzzy -- narrow
    # by design so it can never false-positive on a real day's real news.
    items = [NewsItem(ticker=t, sentiment="Neutral", news_summary="x")
             for t in ["AAA", "CCC", "BBB"]]  # right letters, wrong order
    assert _is_fabricated_placeholder_response(items) is False
    print("PASS: an out-of-order near-miss is not flagged -- the check is exact, not fuzzy")


def test_is_fabricated_placeholder_response_detects_single_letter_cluster():
    # Confirmed live 2026-08-27/28: a second fabrication shape -- bare
    # single-letter "tickers" (['A', 'C', 'D', 'E', 'F', 'G', 'I', 'J',
    # 'K', 'L']), not in the strict AAA/BBB/CCC order the first check
    # requires, so that check alone let this through and polluted
    # ExcludedSymbols with 10 fake entries.
    items = [NewsItem(ticker=t, sentiment="Neutral", news_summary="x")
             for t in ["A", "C", "D", "E", "F", "G", "I", "J", "K", "L"]]
    assert _is_fabricated_placeholder_response(items) is True
    print("PASS: detects the confirmed live single-letter fabrication cluster")


def test_is_fabricated_placeholder_response_false_for_one_or_two_real_single_letter_tickers():
    # Real single-letter tickers exist (C=Citigroup, F=Ford, ...) -- a
    # couple showing up in real news must not be flagged, only a cluster
    # of MIN_SINGLE_LETTER_TICKERS_TO_FLAG or more.
    items = [NewsItem(ticker=t, sentiment="Negative", news_summary="x")
             for t in ["C", "F", "TSLA", "NVDA"]]
    assert _is_fabricated_placeholder_response(items) is False
    print("PASS: one or two genuine single-letter tickers alongside real news is not flagged")


def test_scan_market_news_openai_raises_on_a_fabricated_response():
    """Confirmed real 2026-08-24/25: both OpenAI calls return a clean 200
    OK, but the second one fabricates a placeholder ticker sequence
    instead of admitting it had nothing real to extract. This must raise,
    not silently return fake data as if it were real news."""
    fake_client = type("FakeClient", (), {})()
    fake_client.responses = type("R", (), {
        "create": lambda self, **kwargs: type("Resp", (), {"output_text": "no relevant news found"})()
    })()
    fake_client.chat = type("C", (), {"completions": type("CC", (), {
        "create": lambda self, **kwargs: type("Structured", (), {
            "choices": [type("Choice", (), {
                "message": type("Msg", (), {
                    "content": json.dumps({"items": [
                        {"ticker": "AAA", "sentiment": "Neutral", "news_summary": "x"},
                        {"ticker": "BBB", "sentiment": "Neutral", "news_summary": "x"},
                    ]})
                })()
            })()]
        })()
    })()})()

    with patch("phase2.news_blacklist._openai_client", return_value=fake_client):
        raised = False
        try:
            scan_market_news_openai("some question")
        except RuntimeError as e:
            raised = True
            assert "fabricated" in str(e)
        assert raised
    print("PASS: scan_market_news_openai raises instead of returning a fabricated placeholder list")


def test_scan_raises_when_both_gemini_and_openai_fail():
    # "If both are down, no trading" -- this propagates up to main.py's
    # NewsScanHealth, which is what actually pauses new entries.
    bl = _fresh_blacklist()
    with patch("phase2.news_blacklist.scan_market_news") as mock_gemini, \
         patch("phase2.news_blacklist.scan_market_news_openai") as mock_openai:
        mock_gemini.side_effect = genai_errors.ServerError(503, {"error": {"message": "down"}})
        mock_openai.side_effect = RuntimeError("OpenAI down too")
        try:
            run_daily_scan(bl)
            assert False, "expected an exception"
        except RuntimeError:
            pass
    print("PASS: propagates a failure when both Gemini and OpenAI fail")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed")
