"""Point-in-time features used by the hypothesis tests. Each value at a signal date uses
only prices up to and including that date."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def market_returns(daily: pd.DataFrame, eligible: pd.DataFrame) -> pd.Series:
    """Equal-weight daily return of the stocks eligible at the latest signal date."""
    elig = eligible.reindex(daily.index, method="ffill").fillna(False).astype(bool)
    return daily[elig.columns].where(elig).mean(axis=1).rename("market")


def market_state(
    market: pd.Series, signal_dates: pd.DatetimeIndex, vol_days: int = 63, min_history: int = 36
) -> pd.DataFrame:
    """Volatility and trend of the market at each signal date.

    `high_vol` compares trailing realised volatility with the median of all earlier
    month-end readings (an expanding median, so no future data sets the bar); `down` marks
    a negative trailing 12-month market return.
    """
    vol = market.rolling(vol_days, min_periods=vol_days).std() * np.sqrt(TRADING_DAYS)
    ret12 = np.expm1(np.log1p(market).rolling(TRADING_DAYS, min_periods=TRADING_DAYS).sum())
    out = pd.DataFrame({"vol": vol, "ret12": ret12}).reindex(signal_dates)
    out["vol_median"] = out["vol"].expanding(min_periods=min_history).median()
    # Nullable booleans: unknown (too little history) stays <NA> rather than False.
    out["high_vol"] = (
        (out["vol"] > out["vol_median"]).astype("boolean").mask(out["vol_median"].isna())
    )
    out["down"] = (out["ret12"] < 0).astype("boolean").mask(out["ret12"].isna())
    return out


def abnormal_volatility(
    daily: pd.DataFrame, market: pd.Series, signal_dates: pd.DatetimeIndex, window: int
) -> pd.DataFrame:
    """Market-residual volatility over the last `window` days divided by the residual
    volatility over the year before that window. A high ratio suggests the move came
    with news rather than drifting without it."""
    resid = daily.sub(market, axis=0)
    recent = resid.rolling(window, min_periods=int(0.8 * window)).std()
    base = resid.shift(window).rolling(TRADING_DAYS, min_periods=int(0.8 * TRADING_DAYS)).std()
    return (recent / base).reindex(signal_dates)


def group_breadth(groups: pd.Series, rets: pd.Series) -> pd.Series:
    """Per group: share of its members whose return beat the median stock's."""
    r = rets.dropna()
    g = groups.reindex(r.index).dropna()
    beat = (r[g.index] > r.median()).astype(float)
    return beat.groupby(g).mean()
