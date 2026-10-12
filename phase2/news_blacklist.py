"""Gemini news blacklist, per the client's Market News Scan spec. Two scan
cadences:
  - once at 9:25am ET: "List US Equities with huge market moving news before
    today's market open"
  - every 1 minute from 9:30am-3:30pm ET (added 2026-07-28, narrowed from 5
    minutes 2026-08-21 for NEWS STOP): "List US Equities with huge market
    moving news today"
Both exclude any symbol with a directional (Positive or Negative) catalyst
for 1 day (24hr TTL) -- only Neutral symbols stay tradeable. The execution
engine (risk_manager.py) checks this blacklist before submitting any order.

Two-call design for the Gemini request: the Gemini API does not allow
combining the GoogleSearch grounding tool with structured JSON output
(response_schema) in the same call -- grounding needs freeform text. So this
first asks with GoogleSearch grounding for the live pre-market news as plain
text, then makes a second, tool-free call asking Gemini to structure that
text into the Pydantic schema. This still satisfies the spec's three
requirements (live search grounding, 100%-parseable structured output,
automated scheduling) -- it just takes two model calls to get both.
"""
import fcntl
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

BLACKLIST_PATH = Path(__file__).resolve().parent.parent / "data" / "excluded_symbols.json"
BLACKLIST_TTL = timedelta(hours=24)  # "Script will not trade those symbols for 1 day"
MARKET_HALT_PATH = Path(__file__).resolve().parent.parent / "data" / "market_halt.json"
# Restructured 2026-08-28, per the client: "Excluded Symbols should only be
# populated AFTER a real News Stop Loss exit fires on an open position. The
# Symbol responsible for the News Stop Loss will become Excluded." Before
# this, every symbol a news/EDGAR scan flagged went straight into
# excluded_symbols.json (BLACKLIST_PATH) regardless of whether any pair
# held a position in it -- confirmed live 2026-08-28: 24+ real symbols
# (NVDA, TSLA, JPM, ...) showed up there with zero News Stops having
# actually fired that day. This separate, otherwise-identical store holds
# the RAW scan signal ("currently in the news") that main.py's News Stop
# check reads against open positions; BLACKLIST_PATH now only gets written
# to when a News Stop actually flattens a position over one of these.
NEWS_FLAGGED_PATH = Path(__file__).resolve().parent.parent / "data" / "news_flagged_symbols.json"

logger = logging.getLogger("pairs_bot.news_blacklist")
# Both scans cover the same window per spec: "the previous pre-market
# session + now". That spans yesterday's pre-market, yesterday's regular
# session, the overnight/after-hours gap, and today up to the moment of the
# scan. It has to reach back that far because a stock that moved on news in
# any of those windows is still inside the rolling dataset the entry trigger
# is computed from, so it must stay excluded. Asking only about news "before
# today's market open" (or just "today") left most of that span uncovered.
_LOOKBACK_WINDOW = (
    "from the previous pre-market session up to now, covering yesterday's "
    "pre-market, yesterday's regular trading session, overnight and "
    "after-hours, and today so far"
)
NEWS_QUESTION = (
    f"List US Equities with huge market moving news {_LOOKBACK_WINDOW}."
)
INTRADAY_NEWS_QUESTION = (
    f"List US Equities with huge market moving news today, and {_LOOKBACK_WINDOW}."
)
# Narrowed from 5 to 1 minute per the client 2026-08-21's Entry Conditions
# PDF, so NEWS STOP (exit @ MKT the instant a held symbol becomes newly
# excluded) can react fast -- the scan keeps running at this pace through
# the existing window end time (see INTRADAY_NEWS_SCAN_END_TIME in
# main.py), which already covers the post-entry holding period, not just
# the pre-10:30am entry window.
INTRADAY_SCAN_INTERVAL = timedelta(minutes=1)

# Market-wide event check. Breaching either threshold excludes the entire S&P
# 500 for 24 hours, so this is a kill switch for the whole universe rather
# than a per-symbol exclusion.
CRASH_THRESHOLD_PCT = -1.5   # "crashed today by <= -1.5%"
RALLY_THRESHOLD_PCT = 1.5    # "rallied extraordinarily today => 1.5%"

# Measured from the previous session's closing level to now, per spec ("the
# measurement starts from Market Close Level + now"). Stated explicitly rather
# than left as "so far today", which a model can just as reasonably read as
# today's open to now -- on a gap day those two differ by the whole gap, which
# is exactly the sort of day this check exists for.
MARKET_WIDE_QUESTION = (
    f"What is the S&P 500 Index percentage move measured from the previous "
    f"session's closing level to its level right now? Has it fallen by "
    f"{CRASH_THRESHOLD_PCT}% or worse against that close? Has it risen by "
    f"{RALLY_THRESHOLD_PCT}% or more against that close?"
)


class MarketWideEvent(BaseModel):
    crashed: bool
    rallied_extraordinarily: bool
    index_move_pct: float
    reason: str


# "gemini-2.5-flash" is restricted to legacy accounts only ("no longer
# available to new users" per the API's own error) -- gemini-flash-latest is
# an alias Google keeps pointed at their current recommended flash model, so
# this won't silently break again as models get deprecated.
GEMINI_MODEL = "gemini-flash-latest"


class NewsItem(BaseModel):
    ticker: str
    sentiment: Literal["Positive", "Negative", "Neutral"]
    news_summary: str


class NewsItemsResponse(BaseModel):
    """Wrapper around list[NewsItem] -- OpenAI's structured outputs require
    an object at the schema root, unlike Gemini's response_schema which
    accepts a bare array directly."""
    items: list[NewsItem]


# Per the client 2026-08-18: "Use GPT-5.6 Sol" -- not a real OpenAI model
# identifier (their naming has never used an "X.Y Name" pattern), and a
# wrong string here would 404 silently, defeating the whole point of a
# fallback for the exact scenario ("if both are down, no trading") this
# exists to avoid. Using a real, current model with live web-search
# capability instead -- swap this constant once he confirms the exact
# model he means.
OPENAI_MODEL = "gpt-4o"


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (check .env)")
    # Confirmed live 2026-08-07: an unbounded Gemini call left the whole bot
    # frozen for 7+ minutes (an ESTABLISHED TCP connection to Google that
    # never returned), and since run_daily_scan is called synchronously
    # (not backgrounded like the intraday scan), that hang blocked the
    # entire event loop -- session start never reached warm-up or the main
    # check-cycle loop at all. 30s is generous against the "occasionally
    # spike to 2-5+ seconds" already documented for this call.
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (check .env)")
    return OpenAI(api_key=api_key, timeout=30.0)


def _generate_with_retry(client: genai.Client, retries: int = 3, **kwargs):
    """Gemini's own servers occasionally return a transient 503 ("currently
    experiencing high demand") -- retried with backoff since that's the
    server telling us to try again, not a problem with the request. A 4xx
    ClientError (bad key, bad request) means retrying won't help, so those
    propagate immediately."""
    for attempt in range(retries):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ServerError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def scan_market_news(question: str = NEWS_QUESTION) -> list[NewsItem]:
    """Live two-call Gemini scan. Not unit tested directly (real network call)
    -- symbols_to_exclude() below is the pure, tested part of this pipeline."""
    client = _client()

    grounded = _generate_with_retry(
        client, model=GEMINI_MODEL, contents=question,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )

    structured = _generate_with_retry(
        client, model=GEMINI_MODEL,
        contents=(
            "Extract every US equity ticker mentioned in the following market "
            "news summary, with its sentiment (Positive, Negative, or Neutral) and "
            "a one-sentence news_summary for each.\n\n" + grounded.text
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[NewsItem],
        ),
    )
    return [NewsItem.model_validate(item) for item in json.loads(structured.text)]


MIN_SINGLE_LETTER_TICKERS_TO_FLAG = 3


def _is_fabricated_placeholder_response(items: list[NewsItem]) -> bool:
    """Confirmed real 2026-08-24/25: when Gemini fails over to this OpenAI
    fallback and the web_search_preview call itself doesn't turn up real
    usable content, the second (structured JSON) call doesn't fail or
    return empty -- forced to emit valid JSON matching the schema, GPT-4o
    instead fabricates a plausible-looking example: tickers AAA, BBB, CCC,
    DDD... in exact alphabetical sequence. Both calls return a clean 200
    OK, so nothing else catches this -- confirmed live three separate
    times in one evening, always this exact sequential pattern. Detected
    narrowly (an exact prefix of the tripled-letter alphabet sequence, not
    just "a ticker with a repeated letter") specifically so this can never
    false-positive on a real day's real tickers, which would essentially
    never come back in perfect A, B, C, D... order.

    Confirmed a SECOND fabrication shape live 2026-08-27/28: the same
    empty-search failure mode instead produced bare single-letter
    "tickers" -- ['A', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L'], not in
    strict alphabetical order (so the first check alone doesn't catch it),
    most likely the model echoing back a lettered list (A. ... B. ...) from
    its own grounded search text as if each letter were a ticker. Real
    single-letter tickers exist (C=Citigroup, F=Ford, ...), but a real
    day's news naming MIN_SINGLE_LETTER_TICKERS_TO_FLAG or more of them in
    one scan is not a thing that happens -- flagged as fabricated on count
    alone, same "if OpenAI also fails, no new trading" outcome as the first
    check."""
    if not items:
        return False
    placeholder_sequence = [chr(ord("A") + i) * 3 for i in range(26)]
    if [item.ticker for item in items] == placeholder_sequence[:len(items)]:
        return True
    single_letter_count = sum(1 for item in items if len(item.ticker) == 1)
    return single_letter_count >= MIN_SINGLE_LETTER_TICKERS_TO_FLAG


def scan_market_news_openai(question: str = NEWS_QUESTION) -> list[NewsItem]:
    """Per the client 2026-08-18: "Use GPT-5.6 Sol. It will only come on
    when Gemini fails. Don't use both at the same time." -- ChatGPT/OpenAI
    fallback for scan_market_news, only ever called after Gemini has
    already failed (see run_daily_scan/run_intraday_scan below), never
    alongside it.

    Same two-call shape as the Gemini version and for the same reason:
    OpenAI's web_search tool and its structured-output mode don't mix
    reliably in one call either, so a first call gets grounded free text
    and a second, tool-free call structures that into JSON.

    Raises RuntimeError if the response looks fabricated (see
    _is_fabricated_placeholder_response) -- confirmed real 2026-08-24/25,
    three separate times in one evening: both API calls return a clean 200
    OK, but the "extracted" tickers are a made-up AAA/BBB/CCC/... sequence,
    not real news data, most likely because the web_search_preview call
    itself didn't turn up anything real and the second call fabricated a
    plausible-shaped example rather than admitting it had nothing to
    extract. Raising here routes this through the exact same
    already-tested "if OpenAI also fails, no new trading" path
    _scan_market_news_with_fallback already has for a genuine API
    failure -- silently accepting fabricated data would be worse than that,
    since it defeats the whole point of this fallback existing."""
    client = _openai_client()

    grounded = client.responses.create(
        model=OPENAI_MODEL, input=question, tools=[{"type": "web_search_preview"}],
    )

    structured = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Extract every US equity ticker mentioned in the following market "
                "news summary, with its sentiment (Positive, Negative, or Neutral) and "
                "a one-sentence news_summary for each. Respond as JSON matching "
                '{"items": [{"ticker": str, "sentiment": "Positive"|"Negative"|"Neutral", '
                '"news_summary": str}]}.\n\n' + grounded.output_text
            ),
        }],
        response_format={"type": "json_object"},
    )
    parsed = NewsItemsResponse.model_validate(json.loads(structured.choices[0].message.content))
    if _is_fabricated_placeholder_response(parsed.items):
        raise RuntimeError(
            f"OpenAI fallback returned a fabricated placeholder sequence "
            f"({[i.ticker for i in parsed.items]}), not real news data -- "
            f"treating this scan as failed rather than trusting it")
    return parsed.items


def scan_market_wide_event(question: str = MARKET_WIDE_QUESTION) -> MarketWideEvent:
    """Asks whether the index has crashed or rallied extraordinarily today.
    Same two call shape as scan_market_news: grounding cannot be combined with
    a response schema, so one grounded call answers and a second extracts it
    into a structured verdict."""
    client = _client()

    grounded = _generate_with_retry(
        client, model=GEMINI_MODEL, contents=question,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )

    structured = _generate_with_retry(
        client, model=GEMINI_MODEL,
        contents=(
            f"From the following, state the S&P 500's percentage move from the previous "
            f"session's close to now as a number, whether that is a fall of "
            f"{abs(CRASH_THRESHOLD_PCT)}% or worse, whether it is a rise of "
            f"{RALLY_THRESHOLD_PCT}% or more, and a one-sentence reason.\n\n" + grounded.text
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MarketWideEvent,
        ),
    )
    return MarketWideEvent.model_validate(json.loads(structured.text))


@dataclass
class MarketHalt:
    """A 24-hour halt on the whole S&P 500 after a market-wide event.

    Persisted rather than held in memory so a restart cannot resume trading
    into the same crash the halt was raised for, and self-expiring on read for
    the same reason the symbol exclusions are."""
    path: Path = MARKET_HALT_PATH

    def _read(self, now: datetime) -> datetime | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path) as f:
                expires_at = datetime.fromisoformat(json.load(f)["expires_at"])
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            return None
        return expires_at if expires_at > now else None

    def raise_halt(self, reason: str, now: datetime | None = None,
                   expires_at: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write (temp file + os.replace) -- per the independent
        # audit's Finding #7, verified real 2026-08-24: a crash mid-write
        # (power loss, OOM kill) could leave this file truncated, and
        # _read() already treats an unreadable file as "no halt active" --
        # silently dropping a real 24-hour market-wide crash halt on
        # restart. Same pattern trading_lists.py already uses for its own
        # JSON files.
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"expires_at": (expires_at or (now + BLACKLIST_TTL)).isoformat(),
                       "raised_at": now.isoformat(), "reason": reason}, f, indent=2)
        os.replace(tmp, self.path)

    def is_active(self, now: datetime | None = None) -> bool:
        return self._read(now or datetime.now(timezone.utc)) is not None

    def expires_at(self, now: datetime | None = None) -> datetime | None:
        return self._read(now or datetime.now(timezone.utc))

    def details(self, now: datetime | None = None) -> dict | None:
        """Reason and expiry for the dashboard, or None once it has lapsed."""
        if not self.is_active(now):
            return None
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None


def run_market_wide_scan(halt: "MarketHalt | None" = None) -> bool:
    """Raises a 24-hour market-wide halt if the index crashed or rallied
    extraordinarily. Returns whether a halt is active afterwards.

    A failed scan deliberately changes nothing. Defaulting to halt would let a
    single Gemini outage stop trading for a day, and this project has already
    seen transient 503s; defaulting to clear would silently drop a risk
    control. No answer means no new information, so the previous state
    stands."""
    halt = halt or MarketHalt()
    try:
        event = scan_market_wide_event()
    except Exception:
        logger.warning("market-wide scan failed, leaving the halt state unchanged", exc_info=True)
        return halt.is_active()

    # The threshold is applied here, to the number, rather than trusting the
    # model's own yes/no. Now that "crashed" means a specific figure, the
    # decision is arithmetic and belongs in code: the model is used to look up
    # the index move, not to judge whether it counts. Its booleans are still
    # logged when they disagree, since a mismatch means the reported figure
    # and the reported verdict came apart and the number is worth checking.
    move = event.index_move_pct
    breached = move <= CRASH_THRESHOLD_PCT or move >= RALLY_THRESHOLD_PCT
    if breached != (event.crashed or event.rallied_extraordinarily):
        logger.info(
            f"market-wide scan: reported move {move:+.2f}% and the model's own verdict "
            f"(crashed={event.crashed}, rallied={event.rallied_extraordinarily}) disagree; "
            f"going with the number")

    if breached:
        which = "crash" if move <= CRASH_THRESHOLD_PCT else "extraordinary rally"
        reason = f"{which}: index move {move:+.2f}% -- {event.reason}"
        halt.raise_halt(reason)
        logger.warning(f"MARKET WIDE HALT raised, all S&P 500 symbols excluded for 24h -- {reason}")
        return True
    return halt.is_active()


_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,6}$")


def symbols_to_exclude(items: list[NewsItem]) -> list[str]:
    """Any symbol with a directional catalyst (Positive or Negative) gets
    excluded -- only Neutral stays tradeable, per spec: news catalysts mean
    the spread move is fundamental divergence, not mean-reverting noise.

    Tickers are validated and de-duplicated before going any further. These
    come straight out of an LLM response, and since the 2026-08-26 rescoping
    news exclusion no longer merely blocks entry -- it drives the News Stop
    EXIT, which market-flattens a real open position the moment either leg
    matches. A malformed or hallucinated ticker reaching that check could
    force-close a live trade, so anything that isn't ticker-shaped is
    dropped with a warning rather than trusted. (Contrast EDGAR, which
    resolves through SEC's own official CIK-to-ticker file and needs none
    of this.) Duplicates are collapsed too -- production logs regularly
    show the same name repeated several times in one scan."""
    out: list[str] = []
    for item in items:
        if item.sentiment not in ("Positive", "Negative"):
            continue
        ticker = (item.ticker or "").strip().upper()
        if not _TICKER_RE.match(ticker):
            logger.warning(f"Dropping malformed ticker from the news scan: {item.ticker!r}")
            continue
        if ticker not in out:
            out.append(ticker)
    return out


@contextmanager
def _file_lock(path: Path):
    """Cross-PROCESS advisory lock (fcntl.flock) around a read-modify-write
    critical section -- independent audit finding, confirmed real: this
    file is written by two separate OS processes (the dashboard, for a
    human's manual add/remove, and the bot's own Gemini/EDGAR scans), each
    doing its own unlocked load-then-save. Two writes landing close enough
    together let the second one's save silently clobber the first's --
    e.g. the dashboard's read-modify-write overwriting a real news
    exclusion the bot had just added, based on a read that happened before
    that add. Every OTHER dashboard-editable list (market_entry_list.json,
    trading_lists.json's files) is written ONLY by the dashboard's own
    single-threaded Flask process, so this exact race doesn't apply there.
    Locks a dedicated sibling .lock file rather than the JSON file itself,
    so this works the same whether the JSON file exists yet or not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@dataclass
class ExcludedSymbols:
    """Persistent 1-day blacklist backed by excluded_symbols.json. Self-purges
    expired entries on every read, per spec's "no manual cleanup required."""
    path: Path = BLACKLIST_PATH

    def _load(self, now: datetime) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A corrupted/unreadable file (e.g. a crash mid-write) would
            # otherwise raise here every single read -- add()/is_excluded()
            # have no other error handling above them, so this silently
            # disabled the whole news blacklist for the rest of the session
            # (every call failing the same way, never healing) with nothing
            # louder than an unhandled exception buried in the caller's own
            # try/except. Same recovery pattern trading_lists.py already
            # uses for its own JSON files -- treat it as empty and move on,
            # loudly, rather than let the safety feature go dark quietly.
            logger.error(f"{self.path} is unreadable ({e}) -- treating the blacklist as empty "
                         f"this read; it will self-heal on the next successful write")
            return {}
        return {
            symbol: expires_at for symbol, expires_at in raw.items()
            if datetime.fromisoformat(expires_at) > now
        }

    def _save(self, data: dict[str, str]) -> None:
        # Atomic write -- see MarketHalt.raise_halt's own comment on why
        # (the independent audit's Finding #7): a crash mid-write could
        # otherwise leave this file truncated, and _load already treats an
        # unreadable file as an empty blacklist -- a corrupted write would
        # both lose every active exclusion AND immediately overwrite the
        # good copy that was there before it, with no recovery.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def add(self, symbol: str, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        with _file_lock(self.path):
            data = self._load(now)
            data[symbol] = (now + BLACKLIST_TTL).isoformat()
            self._save(data)

    def remove(self, symbol: str, now: datetime | None = None) -> None:
        """Lifts an exclusion before its 24 hours are up -- the dashboard's
        delete button. A symbol the news scan flags again will simply be
        re-added on the next scan, so this is a manual override for the
        current window, not a permanent allow."""
        now = now or datetime.now(timezone.utc)
        with _file_lock(self.path):
            data = self._load(now)
            data.pop(symbol, None)
            self._save(data)

    def is_excluded(self, symbol: str, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        with _file_lock(self.path):
            data = self._load(now)
            self._save(data)  # persist the purge so expired entries don't linger on disk
        return symbol in data

    def active_symbols(self, now: datetime | None = None) -> set[str]:
        now = now or datetime.now(timezone.utc)
        with _file_lock(self.path):
            data = self._load(now)
            self._save(data)
        return set(data.keys())

    def all_entries(self, now: datetime | None = None) -> dict[str, datetime]:
        """Symbol -> expires_at datetime, for UI display (dashboard) --
        active_symbols() alone doesn't expose when each entry expires."""
        now = now or datetime.now(timezone.utc)
        data = self._load(now)
        self._save(data)
        return {symbol: datetime.fromisoformat(expires_at) for symbol, expires_at in data.items()}


def _scan_market_news_with_fallback(question: str) -> list[NewsItem]:
    """Per the client 2026-08-18: "Use GPT-5.6 Sol. It will only come on
    when Gemini fails... If Gemini API is down, use ChatGPT API. If both
    are down, no trading." -- Gemini tried first; OpenAI only ever runs
    after Gemini has already raised. If OpenAI also fails, this itself
    raises and the caller (run_daily_scan/run_intraday_scan, and above
    that main.py's NewsScanHealth) treats it exactly like a Gemini-only
    failure always has -- new entries pause until a later scan succeeds."""
    try:
        return scan_market_news(question)
    except Exception as e:
        logger.warning(f"Gemini scan failed ({type(e).__name__}), falling back to OpenAI")
        return scan_market_news_openai(question)


def run_daily_scan(news_flagged: ExcludedSymbols | None = None) -> list[str]:
    """Entry point called once at 9:25am ET. Returns the list of
    newly-flagged symbols. Writes into news_flagged (NEWS_FLAGGED_PATH by
    default), the raw "currently in the news" signal -- NOT the
    client-visible Excluded Symbols list (BLACKLIST_PATH/excluded_symbols.
    json), which main.py's News Stop check populates only for a symbol that
    actually causes a position to flatten. See NEWS_FLAGGED_PATH's own
    comment."""
    news_flagged = news_flagged or ExcludedSymbols(path=NEWS_FLAGGED_PATH)
    items = _scan_market_news_with_fallback(NEWS_QUESTION)
    newly_flagged = symbols_to_exclude(items)
    for symbol in newly_flagged:
        news_flagged.add(symbol)
    return newly_flagged


def run_intraday_scan(news_flagged: ExcludedSymbols | None = None) -> list[str]:
    """Entry point called every INTRADAY_SCAN_INTERVAL (1 minute, narrowed
    from 5 2026-08-21 for NEWS STOP) from 9:30am-3:30pm ET. Same logic as
    run_daily_scan, just asking about news "today" instead of "before
    today's market open" -- see run_daily_scan's own comment on
    news_flagged vs the client-visible Excluded Symbols list."""
    news_flagged = news_flagged or ExcludedSymbols(path=NEWS_FLAGGED_PATH)
    items = _scan_market_news_with_fallback(INTRADAY_NEWS_QUESTION)
    newly_flagged = symbols_to_exclude(items)
    for symbol in newly_flagged:
        news_flagged.add(symbol)
    return newly_flagged
