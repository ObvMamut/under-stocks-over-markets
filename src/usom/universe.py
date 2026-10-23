"""Point-in-time S&P 500 membership.

The source file has one row per date on which the index changed, listing every member
on that date. Membership on any other day is the most recent row on or before it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from usom.config import MEMBERSHIP_URL, TICKER_ALIASES


def normalize_ticker(ticker: str) -> str:
    """Map an index-file symbol to the symbol Yahoo uses (BRK.B -> BRK-B, FB -> META)."""
    ticker = ticker.strip().upper()
    ticker = TICKER_ALIASES.get(ticker, ticker)
    return ticker.replace(".", "-")


def download_membership(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(MEMBERSHIP_URL, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_membership(path: Path) -> pd.Series:
    """Return change dates -> frozenset of normalized member tickers."""
    df = pd.read_csv(path, parse_dates=["date"])
    members = df["tickers"].map(
        lambda s: frozenset(normalize_ticker(t) for t in s.split(",") if t.strip())
    )
    return pd.Series(members.to_numpy(), index=pd.DatetimeIndex(df["date"]), name="members")


def members_on(membership: pd.Series, date: pd.Timestamp) -> frozenset[str]:
    """Members on `date`: the latest change row on or before it."""
    pos = membership.index.searchsorted(date, side="right") - 1
    if pos < 0:
        return frozenset()
    return membership.iloc[pos]


def membership_matrix(membership: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean frame (dates x tickers): was the ticker an index member on that date?"""
    tickers = sorted(set().union(*membership))
    rows = [[t in members_on(membership, d) for t in tickers] for d in dates]
    return pd.DataFrame(rows, index=dates, columns=tickers, dtype=bool)


def membership_spells(membership: pd.Series) -> pd.DataFrame:
    """Contiguous stretches of membership: columns ticker, start, end.

    `end` is the first change date on which the ticker was no longer listed, or NaT if
    it is still a member in the last row.
    """
    spells = []
    open_since: dict[str, pd.Timestamp] = {}
    previous: frozenset[str] = frozenset()
    for date, members in membership.items():
        for t in members - previous:
            open_since[t] = date
        for t in previous - members:
            spells.append((t, open_since.pop(t), date))
        previous = members
    spells.extend((t, start, pd.NaT) for t, start in open_since.items())
    out = pd.DataFrame(spells, columns=["ticker", "start", "end"])
    return out.sort_values(["ticker", "start"], ignore_index=True)
