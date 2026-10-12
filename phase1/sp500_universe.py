"""S&P 500 constituent list with GICS sector / sub-industry classification.

Source: Wikipedia's maintained "List of S&P 500 companies" table. This is the
standard, widely-used source for this exact data in quant tooling — it's kept
current by editors tracking index reconstitutions, which is why it's preferred
over a static hardcoded list.
"""
import re
import ssl
import urllib.request
from io import StringIO

import certifi
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_html(url: str) -> str:
    # some Python installs (notably python.org macOS builds) don't wire up
    # the system certificate store by default, causing SSL_CERT_FILE-less
    # requests to fail with CERTIFICATE_VERIFY_FAILED. Explicitly using
    # certifi's bundle here keeps the fix self-contained to this project's
    # venv instead of requiring a machine-wide Python config change.
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_sp500_universe() -> pd.DataFrame:
    """Returns a DataFrame with columns: symbol, security, gics_sector, gics_sub_industry."""
    html = _fetch_html(WIKI_URL)
    tables = pd.read_html(StringIO(html))
    df = tables[0]
    df = df.rename(columns={
        "Symbol": "symbol",
        "Security": "security",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
    })
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)  # yfinance/Databento format (e.g. BRK.B -> BRK-B)
    return df[["symbol", "security", "gics_sector", "gics_sub_industry"]]


CLASS_SUFFIX = re.compile(r"\s*\(Class [A-Z]\)\s*$")


def drop_duplicate_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps one ticker per company where the index lists several share
    classes (Alphabet GOOGL/GOOG, Fox FOXA/FOX, News Corp NWSA/NWS).

    Two classes of the same company are the same underlying business and
    track each other almost exactly, so pairing each against the same
    counterpart produces what look like two independent pairs but is really
    one bet held twice -- the bot would carry double the intended exposure
    while believing it was diversified. Seen for real: a 1-minute screen
    returned GOOGL/META and GOOG/META as its only two "pairs".

    Keeps the first ticker alphabetically. The classes are economically
    equivalent for spread purposes, so this only needs to be deterministic."""
    base = df["security"].str.replace(CLASS_SUFFIX, "", regex=True).str.strip()
    return df.assign(_base=base).sort_values("symbol").drop_duplicates("_base", keep="first").drop(columns="_base")


def industry_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Groups symbols by GICS Sub-Industry (the spec's "same industry" test —
    sub-industry is the tightest, most economically meaningful grouping;
    e.g. Visa/Mastercard share "Transaction & Payment Processing Services",
    not just the broader "Financials" sector)."""
    df = drop_duplicate_share_classes(df)
    groups: dict[str, list[str]] = {}
    for sub_industry, group in df.groupby("gics_sub_industry"):
        symbols = group["symbol"].tolist()
        if len(symbols) >= 2:
            groups[sub_industry] = symbols
    return groups


if __name__ == "__main__":
    df = fetch_sp500_universe()
    print(f"Fetched {len(df)} S&P 500 constituents")
    groups = industry_groups(df)
    print(f"{len(groups)} sub-industries with 2+ members")
    total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in groups.values())
    print(f"{total_pairs} candidate pairs to screen")
