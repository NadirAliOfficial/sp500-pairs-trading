"""Two hand-maintained lists from the client's spec, both editable from the
dashboard:

  Permanent Blacklist  -- symbols the bot must never trade, whatever the
                          screen says. Unlike the Gemini news blacklist in
                          news_blacklist.py, these never expire.

  Special Pairs List   -- pairs exempted from the Market Cap, Daily Shares
                          Volume and Daily Dollar Volume conditions. They
                          still have to pass every statistical test; the
                          exemption only covers those three size filters.

Kept separate from market_entry_list.py because they answer different
questions: this file is "may these symbols trade at all / do the size filters
apply", the other is "which of the qualified pairs did the client pick".
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("pairs_bot.trading_lists")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERMANENT_BLACKLIST_PATH = str(DATA_DIR / "permanent_blacklist.json")
SPECIAL_PAIRS_PATH = str(DATA_DIR / "special_pairs.json")

DEFAULT_PERMANENT_BLACKLIST = ["NVDA", "TSLA", "SPCX", "GOOG", "META"]
DEFAULT_SPECIAL_PAIRS = [
    ("PEP", "KO"), ("V", "MA"), ("XOM", "CVX"),
    ("JPM", "BAC"), ("CAT", "DE"), ("MDT", "ABT"),
]


def pair_key(symbol_a: str, symbol_b: str) -> tuple[str, str]:
    """Order-independent pair identity -- JPM/BAC and BAC/JPM are one pair."""
    return tuple(sorted((symbol_a.strip().upper(), symbol_b.strip().upper())))


def _read(path: str, key: str, default):
    """Falls back to the spec defaults when the file is missing, so a fresh
    deployment starts with the client's lists rather than empty ones. A file
    that exists but is unreadable is a different matter -- that returns the
    defaults too, but loudly, because silently trading a blacklisted symbol
    is the failure that matters here."""
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f).get(key, default)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"{path} is unreadable ({e}) -- falling back to the spec defaults")
        return default


def _write(path: str, key: str, value) -> None:
    Path(path).parent.mkdir(exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({key: value}, f, indent=2)
    os.replace(tmp, path)  # atomic: a dashboard edit can't leave a half-written file


def load_permanent_blacklist(path: str = PERMANENT_BLACKLIST_PATH) -> set[str]:
    return {s.strip().upper() for s in _read(path, "symbols", DEFAULT_PERMANENT_BLACKLIST)}


def save_permanent_blacklist(symbols, path: str = PERMANENT_BLACKLIST_PATH) -> None:
    _write(path, "symbols", sorted({s.strip().upper() for s in symbols}))


def add_to_blacklist(symbol: str, path: str = PERMANENT_BLACKLIST_PATH) -> None:
    save_permanent_blacklist(load_permanent_blacklist(path) | {symbol}, path)


def remove_from_blacklist(symbol: str, path: str = PERMANENT_BLACKLIST_PATH) -> None:
    save_permanent_blacklist(load_permanent_blacklist(path) - {symbol.strip().upper()}, path)


def special_pairs_as_listed(path: str = SPECIAL_PAIRS_PATH) -> list[tuple[str, str]]:
    """As the client wrote them, for display. Use load_special_pairs() to
    test membership."""
    return [(a.strip().upper(), b.strip().upper())
            for a, b in _read(path, "pairs", DEFAULT_SPECIAL_PAIRS)]


def load_special_pairs(path: str = SPECIAL_PAIRS_PATH) -> set[tuple[str, str]]:
    """Normalised for matching -- see special_pairs_as_listed() for display."""
    return {pair_key(a, b) for a, b in special_pairs_as_listed(path)}


def save_special_pairs(pairs, path: str = SPECIAL_PAIRS_PATH) -> None:
    """Stores each pair in the order it was given, deduplicating on the
    order-independent key. The client writes these as PEP/KO, XOM/CVX,
    MDT/ABT; normalising the stored order would show them back as KO/PEP,
    CVX/XOM, ABT/MDT, which reads as a mistake on a list he maintains by
    hand. Matching stays order-independent regardless."""
    seen, ordered = set(), []
    for a, b in pairs:
        key = pair_key(a, b)
        if key not in seen:
            seen.add(key)
            ordered.append([a.strip().upper(), b.strip().upper()])
    _write(path, "pairs", ordered)


def add_special_pair(symbol_a: str, symbol_b: str, path: str = SPECIAL_PAIRS_PATH) -> None:
    save_special_pairs([*special_pairs_as_listed(path), (symbol_a, symbol_b)], path)


def remove_special_pair(symbol_a: str, symbol_b: str, path: str = SPECIAL_PAIRS_PATH) -> None:
    target = pair_key(symbol_a, symbol_b)
    save_special_pairs([p for p in special_pairs_as_listed(path) if pair_key(*p) != target], path)


def is_blacklisted(symbol: str, path: str = PERMANENT_BLACKLIST_PATH) -> bool:
    return symbol.strip().upper() in load_permanent_blacklist(path)


def pair_is_blacklisted(symbol_a: str, symbol_b: str, path: str = PERMANENT_BLACKLIST_PATH) -> bool:
    """One blacklisted leg disqualifies the whole pair -- the bot cannot trade
    one side of a pair on its own."""
    blacklist = load_permanent_blacklist(path)
    return symbol_a.strip().upper() in blacklist or symbol_b.strip().upper() in blacklist


def is_special_pair(symbol_a: str, symbol_b: str, path: str = SPECIAL_PAIRS_PATH) -> bool:
    return pair_key(symbol_a, symbol_b) in load_special_pairs(path)
