"""Performance statistics for monthly return series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev"]


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    return float(np.prod(1 + r) ** (12 / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(12))


def sharpe(r: pd.Series, rf: pd.Series | None = None) -> float:
    x = (r - rf.reindex(r.index)) if rf is not None else r
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(12))


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def nw_tstat(x: pd.Series, lags: int) -> tuple[float, float]:
    """Mean of x and its Newey-West t-statistic."""
    x = x.dropna()
    fit = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])


def factor_regression(
    excess: pd.Series, factors: pd.DataFrame, lags: int, cols: list[str] = FACTORS
) -> pd.DataFrame:
    """OLS of monthly excess returns on factors with Newey-West errors.

    Returns one row per coefficient (alpha first) with coef, t, and the fit's R^2.
    """
    df = pd.concat([excess.rename("y"), factors[cols]], axis=1, join="inner").dropna()
    X = sm.add_constant(df[cols]).rename(columns={"const": "alpha"})
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = pd.DataFrame({"coef": fit.params, "t": fit.tvalues})
    out.attrs.update(r2=float(fit.rsquared), n=int(fit.nobs))
    return out


def summarize(net: pd.Series, rf: pd.Series, bench: pd.Series, lags: int) -> dict[str, float]:
    """Headline numbers for one strategy against a benchmark return series."""
    excess = (net - bench.reindex(net.index)).dropna()
    mean_x, t_x = nw_tstat(excess, lags)
    return {
        "cagr": cagr(net),
        "vol": ann_vol(net),
        "sharpe": sharpe(net, rf),
        "max_dd": max_drawdown(net),
        "excess_ann": mean_x * 12,
        "excess_t": t_x,
        "months": int(net.notna().sum()),
    }
