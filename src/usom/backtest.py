"""Monthly overlapping-portfolio backtest (Jegadeesh-Titman style).

A new equal-weight cohort is formed at every month end and held for H months, so in any
month the portfolio is the average of the last H cohorts. Trades execute at the close of
the first trading day after the signal date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each calendar month in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def execution_dates(index: pd.DatetimeIndex, signal_dates: pd.DatetimeIndex) -> pd.Series:
    """Signal date -> the next trading day in `index` (NaT when there is none)."""
    pos = index.searchsorted(signal_dates, side="right")
    out = [index[p] if p < len(index) else pd.NaT for p in pos]
    return pd.Series(out, index=signal_dates)


def forward_returns(close: pd.DataFrame, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return earned between consecutive executions, indexed by the signal date it follows.

    `close` is forward-filled first, so a stock that stops trading earns 0 from its last
    quote onward (Yahoo has no delisting returns).
    """
    filled = close.ffill()
    exe = execution_dates(close.index, signal_dates).dropna()
    px = filled.loc[exe.to_numpy()]
    px.index = exe.index
    return (px.shift(-1) / px - 1).iloc[:-1]


def cohort_weights(picks: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Equal weights per formation date from rows (date, ticker); empty dates are all zero."""
    w = pd.DataFrame(0.0, index=dates, columns=tickers)
    if picks.empty:
        return w
    counts = picks.groupby("date")["ticker"].transform("size")
    flat = pd.DataFrame({"date": picks["date"], "ticker": picks["ticker"], "w": 1.0 / counts})
    wide = flat.pivot_table(index="date", columns="ticker", values="w", aggfunc="sum")
    w.update(wide.reindex(index=dates, columns=tickers))
    return w.fillna(0.0)


def run_portfolio(
    cohorts: pd.DataFrame, fwd: pd.DataFrame, holding: int, cost_bps: float
) -> pd.DataFrame:
    """Monthly gross/net returns, one-way turnover and holding count.

    Rows are indexed by the month end in which the return is earned. Costs are
    `cost_bps` per unit of weight bought or sold.
    """
    dates = fwd.index.intersection(cohorts.index)
    c = cohorts.loc[dates, fwd.columns.intersection(cohorts.columns)]
    r = fwd.loc[dates, c.columns].fillna(0.0)
    w = c.rolling(holding, min_periods=holding).mean().dropna()
    r = r.loc[w.index]

    gross = (w * r).sum(axis=1)
    drifted = (w * (1 + r)).div(1 + gross, axis=0)
    prev = drifted.shift(1).fillna(0.0)
    traded = (w - prev).abs().sum(axis=1)
    net = gross - traded * cost_bps / 1e4

    out = pd.DataFrame(
        {
            "gross": gross,
            "net": net,
            "turnover": traded / 2,
            "n_holdings": (w > 0).sum(axis=1),
        }
    )
    # A return that follows signal date t is earned over the next calendar month.
    out.index = next_month_end(out.index)
    return out


def next_month_end(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar end of the month after each date (2020-02-28 -> 2020-03-31)."""
    return (dates.to_period("M") + 1).to_timestamp(how="end").normalize().rename(None)


def holding_returns(fwd: pd.DataFrame, holding: int) -> pd.DataFrame:
    """Compounded H-month return from each formation date (NaN where H months are not known)."""
    log = np.log1p(fwd)
    ahead = log[::-1].rolling(holding, min_periods=holding).sum()[::-1]
    return np.expm1(ahead)


def pick_outcomes(
    picks: pd.DataFrame, fwd: pd.DataFrame, holding: int, basket: str, peers: str
) -> pd.DataFrame:
    """Each pick's H-month return beside its group's and the universe's equal-weight return.

    `picks` holds rows (date, basket, ticker, group); the group return averages the
    `peers` basket rows of the same date and group (e.g. hot_all for hot_laggards).
    """
    h = holding_returns(fwd, holding).stack().rename("ret")
    h.index.names = ["date", "ticker"]
    tagged = picks.join(h, on=["date", "ticker"])
    group_ret = (
        tagged[tagged["basket"] == peers]
        .groupby(["date", "group"])["ret"]
        .mean()
        .rename("group_ret")
    )
    universe_ret = (
        tagged[tagged["basket"] == "universe"].groupby("date")["ret"].mean().rename("universe_ret")
    )
    out = tagged[tagged["basket"] == basket].join(group_ret, on=["date", "group"])
    out = out.join(universe_ret, on="date")
    return out.dropna(subset=["ret", "group_ret", "universe_ret"]).reset_index(drop=True)
