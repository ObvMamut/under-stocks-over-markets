"""Daily adjusted closes from Yahoo (via yfinance), cached to parquet.

Yahoo has no history for most delisted companies and reuses tickers, so prices are only
trusted inside a membership spell whose Yahoo history already exists when the spell
starts. A later-starting history usually belongs to a different company that took the
symbol over (e.g. "FB" today is not Meta).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _download_chunk(tickers: list[str], start: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers, start=start, auto_adjust=True, progress=False, threads=True, group_by="column"
        )
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close


def fetch_prices(
    tickers: Iterable[str], start: str, cache: Path, chunk_size: int = 100
) -> pd.DataFrame:
    """Adjusted closes (dates x tickers). Only tickers missing from the cache are fetched.

    Tickers Yahoo cannot serve are recorded next to the cache so reruns skip them.
    """
    tickers = sorted(set(tickers))
    failed_path = cache.with_suffix(".failed.txt")
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    failed = set(failed_path.read_text().split()) if failed_path.exists() else set()
    todo = [t for t in tickers if t not in have.columns and t not in failed]

    for i in range(0, len(todo), chunk_size):
        chunk = todo[i : i + chunk_size]
        log.info("prices %d-%d of %d", i + 1, i + len(chunk), len(todo))
        got = _download_chunk(chunk, start).dropna(axis=1, how="all")
        failed |= set(chunk) - set(got.columns)
        have = got if have.empty else have.join(got, how="outer")
        cache.parent.mkdir(parents=True, exist_ok=True)
        have.sort_index(axis=1).to_parquet(cache)
        failed_path.write_text("\n".join(sorted(failed)))

    have.index = pd.DatetimeIndex(have.index).tz_localize(None).normalize()
    return have.reindex(columns=[t for t in tickers if t in have.columns])


def drop_spikes(prices: pd.DataFrame, jump: float = 1.0, crash: float = -0.5) -> pd.DataFrame:
    """Blank out one-day bad ticks: a quote that jumps by `jump` and reverts by `crash`
    the next day, or the reverse. Real moves (a bailout crash, a trial result) persist.
    """
    r = prices.pct_change(fill_method=None)
    nxt = r.shift(-1)
    bad = ((r > jump) & (nxt < crash)) | ((r < crash) & (nxt > jump))
    return prices.mask(bad)


def valid_spells(spells: pd.DataFrame, prices: pd.DataFrame, tolerance_days: int = 10) -> pd.Series:
    """Flag each membership spell whose Yahoo history covers the spell's start.

    Spells starting before the price download are judged against the download start.
    """
    first = prices.apply(pd.Series.first_valid_index)
    data_start = prices.index[0]
    tol = pd.Timedelta(days=tolerance_days)

    def ok(row: pd.Series) -> bool:
        f = first.get(row["ticker"])
        if f is None or pd.isna(f):
            return False
        return f <= max(row["start"], data_start) + tol

    return spells.apply(ok, axis=1).rename("valid")


def eligibility(
    member: pd.DataFrame, spells: pd.DataFrame, prices: pd.DataFrame, valid: pd.Series
) -> pd.DataFrame:
    """Boolean frame (member.index x tickers): member, in a trusted spell, and priced that day.

    `prices` must not be forward-filled, so a missing quote counts as unavailable.
    """
    dates = member.index
    trusted = pd.DataFrame(False, index=dates, columns=member.columns)
    for row in spells[valid].itertuples():
        if row.ticker not in trusted.columns:
            continue
        end = row.end if pd.notna(row.end) else dates[-1] + pd.Timedelta(days=1)
        trusted.loc[(dates >= row.start) & (dates < end), row.ticker] = True
    priced = prices.reindex(index=dates, columns=member.columns).notna()
    return member & trusted & priced


def coverage(member: pd.DataFrame, eligible: pd.DataFrame) -> pd.Series:
    """Share of index members that are usable on each date."""
    n_members = member.sum(axis=1)
    return (eligible.sum(axis=1) / n_members.replace(0, np.nan)).rename("coverage")
