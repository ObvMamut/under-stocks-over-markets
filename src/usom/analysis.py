"""Tables built from the saved results (data/results/*.parquet)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from usom import config
from usom.config import StudyConfig
from usom.metrics import factor_regression, nw_tstat, summarize

# The idea, split into the pieces that could each be doing the work.
SPREADS = {
    "hot sector vs market": ("hot_all", "universe"),
    "laggards vs rest of hot sector": ("hot_laggards", "hot_all"),
    "hot laggards vs hot leaders": ("hot_laggards", "hot_leaders"),
    "hot laggards vs laggards anywhere": ("hot_laggards", "all_laggards"),
    "hot laggards vs market": ("hot_laggards", "universe"),
}


def load_results() -> dict[str, pd.DataFrame]:
    names = ["groups", "picks", "returns", "outcomes", "monthly"]
    return {n: pd.read_parquet(config.RESULTS / f"{n}.parquet") for n in names}


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __call__(self, s: pd.Series | pd.DataFrame):
        return s.loc[self.start : self.end]


def windows(monthly: pd.DataFrame, cfg: StudyConfig) -> dict[str, Window]:
    """In-sample (thin early coverage) and out-of-sample (coverage >= cfg.min_coverage).

    The returns for a month count toward the main sample once the portfolio formed at
    the previous month end had enough of the index priced, and every month after.
    """
    cov = monthly["coverage_at_formation"].dropna()
    good = cov >= cfg.min_coverage
    # first month after which coverage never drops below the bar again
    main_start = good[~good].index.max() + pd.offsets.MonthEnd(1)
    end = monthly["spy"].dropna().index.max()
    ext_start = pd.Timestamp(cfg.extended_start)
    return {
        "in_sample": Window("in-sample", ext_start, main_start - pd.offsets.MonthEnd(1)),
        "out_of_sample": Window("out-of-sample", main_start, end),
        "full": Window("full", ext_start, end),
    }


def net_at_cost(df: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Net return for any one-way cost, from gross return and one-way turnover."""
    return df["gross"] - 2 * df["turnover"] * cost_bps / 1e4


def series(
    returns: pd.DataFrame,
    scheme: str,
    lookback: int,
    holding: int,
    basket: str,
    cost_bps: float | None = None,
) -> pd.Series:
    """One basket's monthly net return series."""
    sel = returns[
        (returns["scheme"] == scheme)
        & (returns["lookback"] == lookback)
        & (returns["holding"] == holding)
        & (returns["basket"] == basket)
    ].set_index("month")
    r = sel["net"] if cost_bps is None else net_at_cost(sel, cost_bps)
    return r.rename(basket)


def _wide(returns: pd.DataFrame, cost_bps: float | None) -> pd.DataFrame:
    """Net returns with one column per basket, indexed by (scheme, lookback, holding, month)."""
    r = returns.copy()
    if cost_bps is not None:
        r["net"] = net_at_cost(r, cost_bps)
    return r.pivot_table(
        index=["scheme", "lookback", "holding", "month"], columns="basket", values="net"
    )


def grid_table(
    returns: pd.DataFrame,
    monthly: pd.DataFrame,
    window: Window,
    cost_bps: float | None = None,
) -> pd.DataFrame:
    """Headline stats for every (scheme, lookback, holding, basket) against the EW universe."""
    wide = _wide(returns, cost_bps)
    rf = monthly["RF"]
    rows = []
    for (scheme, lookback, holding), block in wide.groupby(level=[0, 1, 2]):
        block = window(block.droplevel([0, 1, 2]))
        for basket in block.columns:
            stats = summarize(block[basket], rf, block["universe"], lags=holding)
            rows.append(
                {
                    "scheme": scheme,
                    "lookback": lookback,
                    "holding": holding,
                    "basket": basket,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def spread_table(
    returns: pd.DataFrame, window: Window, cost_bps: float | None = None
) -> pd.DataFrame:
    """Annualised mean and Newey-West t of each SPREADS long/short, for every grid cell."""
    wide = _wide(returns, cost_bps)
    rows = []
    for (scheme, lookback, holding), block in wide.groupby(level=[0, 1, 2]):
        block = window(block.droplevel([0, 1, 2]))
        for name, (long, short) in SPREADS.items():
            mean, t = nw_tstat(block[long] - block[short], lags=holding)
            rows.append(
                {
                    "scheme": scheme,
                    "lookback": lookback,
                    "holding": holding,
                    "spread": name,
                    "ann": mean * 12,
                    "t": t,
                }
            )
    return pd.DataFrame(rows)


def hit_rates(outcomes: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Share of picks that beat their own group and the universe over the holding period."""
    o = outcomes.copy()
    # A pick formed at date t is judged on the months after t.
    o = o[(o["date"] >= window.start - pd.offsets.MonthEnd(1)) & (o["date"] < window.end)]
    o["beat_group"] = o["ret"] > o["group_ret"]
    o["beat_universe"] = o["ret"] > o["universe_ret"]
    o["vs_group"] = o["ret"] - o["group_ret"]
    return (
        o.groupby(["scheme", "lookback", "holding", "basket"])
        .agg(
            picks=("ret", "size"),
            beat_group=("beat_group", "mean"),
            beat_universe=("beat_universe", "mean"),
            median_vs_group=("vs_group", "median"),
        )
        .reset_index()
    )


def factor_table(r: pd.Series, monthly: pd.DataFrame, lags: int) -> pd.DataFrame:
    """Factor regression of a basket's excess-of-T-bill returns, alpha annualised."""
    excess = (r - monthly["RF"].reindex(r.index)).dropna()
    out = factor_regression(excess, monthly, lags=lags)
    out.loc["alpha", "coef"] *= 12
    return out


def choose_in_sample(spreads: pd.DataFrame, spread: str) -> pd.Series:
    """The in-sample grid cell with the best t-stat for one spread: the one to test OOS."""
    s = spreads[spreads["spread"] == spread]
    return s.loc[s["t"].idxmax()]


def growth(r: pd.Series) -> pd.Series:
    """Value of $1 invested at the start of the series."""
    r = r.dropna()
    first = r.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.Series({first: 1.0}), (1 + r).cumprod()])


def drawdown(r: pd.Series) -> pd.Series:
    g = growth(r)
    return g / g.cummax() - 1


def rolling_annual(r: pd.Series, months: int = 36) -> pd.Series:
    """Rolling annualised mean."""
    return r.rolling(months).mean() * 12


def cost_sensitivity(
    returns: pd.DataFrame,
    monthly: pd.DataFrame,
    window: Window,
    cell: dict,
    costs: tuple[float, ...],
    baskets: tuple[str, ...] = ("hot_laggards", "hot_leaders", "hot_all", "universe"),
) -> pd.DataFrame:
    rows = []
    for cost in costs:
        for b in baskets:
            r = window(series(returns, **cell, basket=b, cost_bps=cost))
            bench = window(series(returns, **cell, basket="universe", cost_bps=cost))
            stats = summarize(r, monthly["RF"], bench, lags=cell["holding"])
            rows.append({"cost_bps": cost, "basket": b, **stats})
    return pd.DataFrame(rows)


def ann_turnover(returns: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Average annual one-way turnover per basket, scheme, lookback and holding."""
    r = returns[returns["month"].between(window.start, window.end)]
    t = r.groupby(["scheme", "lookback", "holding", "basket"])["turnover"].mean() * 12
    return t.rename("turnover_per_year").reset_index()
