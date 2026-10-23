"""Yahoo sector and industry labels per ticker, cached to CSV and resumable.

These are today's labels, applied to the whole history (a mild lookahead that the
correlation clusters do not share).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

COLUMNS = ["ticker", "sector", "industry"]


def _lookup(ticker: str, retries: int = 3) -> tuple[str | None, str | None] | None:
    """Labels for one ticker, or None if Yahoo could not be reached (not cached, so retried)."""
    for attempt in range(retries):
        try:
            info = yf.Ticker(ticker).info
            return info.get("sector"), info.get("industry")
        except Exception as exc:  # yfinance raises many transport/rate-limit types
            wait = 5 * (attempt + 1)
            log.warning("info %s failed (%s), retrying in %ds", ticker, exc, wait)
            time.sleep(wait)
    log.error("info %s failed %d times; left out of the cache", ticker, retries)
    return None


def fetch_classification(tickers: Iterable[str], cache: Path, pause: float = 0.2) -> pd.DataFrame:
    """Sector and industry for each ticker. Labels Yahoo does not have stay NaN."""
    tickers = sorted(set(tickers))
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=COLUMNS)
    todo = [t for t in tickers if t not in set(have["ticker"])]
    rows = []
    for n, t in enumerate(todo, 1):
        labels = _lookup(t)
        if labels is not None:
            rows.append((t, *labels))
        time.sleep(pause)
        if n % 50 == 0 or n == len(todo):
            log.info("classification %d of %d", n, len(todo))
            have = pd.concat([have, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
            rows = []
            cache.parent.mkdir(parents=True, exist_ok=True)
            have.to_csv(cache, index=False)
    return have[have["ticker"].isin(tickers)].set_index("ticker").sort_index()
