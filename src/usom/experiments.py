"""Five hypotheses for improving the hot-sector laggard idea: `uv run python -m usom.experiments`.

Run after `usom.pipeline`; it reuses that run's groups, picks and returns. Writes
data/results/experiments.parquet (monthly returns per hypothesis, variant, scheme,
holding and basket) and data/results/market_state.parquet.

Protocol, fixed before any of these results were seen:
- Each hypothesis names one long/short spread and a family of cells
  (variant x scheme x holding).
- In-sample (the early, thin-coverage years) picks the cell with the highest Newey-West
  t-stat. The hypothesis passes in-sample if that t >= 2.
- That one cell is then run out-of-sample. A hypothesis is confirmed if its
  Holm-adjusted one-sided p-value across the five tests is below 5%.
- Every cell is reported, not just the chosen one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from usom import config
from usom.analysis import Window, load_results, windows
from usom.backtest import forward_returns
from usom.config import StudyConfig
from usom.features import (
    abnormal_volatility,
    daily_returns,
    group_breadth,
    market_returns,
    market_state,
)
from usom.metrics import nw_tstat
from usom.pipeline import Dataset, basket_returns, form_all, load_dataset, lookback_returns

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hypothesis:
    key: str
    title: str
    claim: str
    long: str
    short: str


HYPOTHESES = [
    Hypothesis(
        "H1",
        "Horizon mismatch",
        "Rank groups on 3-12 month momentum but pick laggards on the last month only: "
        "within-industry reversal is a one-month effect, industry momentum a longer one.",
        long="hot_laggards",
        short="universe",
    ),
    Hypothesis(
        "H2",
        "News vs noise",
        "Laggards that fell quietly (no jump in stock-specific volatility) rebound; "
        "laggards that fell on news keep drifting.",
        long="hot_laggards_quiet",
        short="hot_laggards_loud",
    ),
    Hypothesis(
        "H3",
        "Rally breadth",
        "Laggards catch up when their group's rally is broad, and stay behind when a few "
        "names carry it.",
        long="hot_laggards_broad",
        short="hot_laggards_narrow",
    ),
    Hypothesis(
        "H4",
        "Leaders, not laggards",
        "Hot-group leaders beat plain stock momentum (the strongest names in the whole index).",
        long="hot_leaders",
        short="universe_leaders",
    ),
    Hypothesis(
        "H5",
        "Market regime",
        "Hold hot-group laggards when the market is volatile (reversal pays for providing "
        "liquidity) and hot-group leaders otherwise.",
        long="switch",
        short="hot_leaders",
    ),
]
BY_KEY = {h.key: h for h in HYPOTHESES}


def formation_in_sample(dates: pd.Series | pd.DatetimeIndex, oos: Window) -> np.ndarray:
    """True for formation dates whose following month falls before the out-of-sample start."""
    periods = pd.DatetimeIndex(dates).to_period("M")
    return np.asarray(periods < oos.start.to_period("M") - 1)


def quiet_loud_tags(ratio: pd.DataFrame) -> pd.DataFrame:
    """'quiet' at or below each date's cross-sectional median abnormal volatility, else 'loud'."""
    median = ratio.median(axis=1)
    tags = pd.DataFrame(
        np.where(ratio.le(median, axis=0), "quiet", "loud"), ratio.index, ratio.columns
    )
    return tags.where(ratio.notna())


def breadth_table(groups: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Rows (date, group, breadth) for every group on every date."""
    frames = []
    for date, g in groups.groupby("date"):
        b = group_breadth(g.set_index("ticker")["group"], rets.loc[date])
        frames.append(pd.DataFrame({"date": date, "group": b.index, "breadth": b.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def breadth_tags(
    groups: pd.DataFrame, breadth: pd.DataFrame, threshold: float, like: pd.DataFrame
) -> pd.DataFrame:
    """'broad' for members of groups at or above the breadth threshold, else 'narrow'."""
    m = groups.merge(breadth, on=["date", "group"])
    m["tag"] = np.where(m["breadth"] >= threshold, "broad", "narrow")
    wide = m.pivot(index="date", columns="ticker", values="tag")
    return wide.reindex(index=like.index, columns=like.columns)


def switch_picks(picks: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    """Hot-group laggards on dates where `state` is True, hot-group leaders otherwise
    (including dates with too little history to judge)."""
    on = state.fillna(False).astype(bool)
    use_laggards = picks["date"].map(on).fillna(False).astype(bool)
    chosen = np.where(use_laggards, "hot_laggards", "hot_leaders")
    out = picks[picks["basket"] == chosen].copy()
    out["basket"] = "switch"
    return out


def run(ds: Dataset, cfg: StudyConfig) -> dict[str, pd.DataFrame]:
    main = load_results()
    oos = windows(main["monthly"], cfg)["out_of_sample"]
    fwd = forward_returns(ds.close.reindex(columns=ds.eligible.columns), ds.signal_dates)
    daily = daily_returns(ds.close.reindex(columns=ds.eligible.columns))
    market = market_returns(daily, ds.eligible)
    state = market_state(market, ds.signal_dates)
    rets = {lb: lookback_returns(ds, lb) for lb in sorted({1, *cfg.lookbacks})}
    frames = []

    def record(picks: pd.DataFrame, hyp: str, variant: str, scheme: str, baskets: tuple):
        res = basket_returns(picks, fwd, cfg.holdings, cfg.cost_bps, baskets)
        frames.append(res.assign(hypothesis=hyp, variant=variant, scheme=scheme))

    for scheme in cfg.schemes:
        groups = main["groups"][main["groups"]["scheme"] == scheme].drop(columns="scheme")
        main_picks = main["picks"][main["picks"]["scheme"] == scheme]

        for lg in (3, 6, 12):
            log.info("H1 %s: groups %dm, stocks 1m", scheme, lg)
            picks = form_all(ds, groups, 1, cfg, group_lookback=lg)
            record(
                picks,
                "H1",
                f"groups {lg}m / stocks 1m",
                scheme,
                ("hot_laggards", "hot_leaders", "hot_all", "universe"),
            )

        for lb in cfg.lookbacks:
            log.info("H2 %s: %dm", scheme, lb)
            ratio = abnormal_volatility(daily, market, ds.signal_dates, window=min(21 * lb, 63))
            tags = quiet_loud_tags(ratio.where(ds.eligible))
            picks = form_all(ds, groups, lb, cfg, tags=tags)
            record(
                picks,
                "H2",
                f"lookback {lb}m",
                scheme,
                ("hot_laggards_quiet", "hot_laggards_loud", "hot_laggards", "universe"),
            )

            log.info("H3 %s: %dm", scheme, lb)
            breadth = breadth_table(groups, rets[lb])
            mp = main_picks[main_picks["lookback"] == lb]
            hot = mp.loc[mp["basket"] == "hot_all", ["date", "group"]].drop_duplicates()
            hot = hot.merge(breadth, on=["date", "group"])
            threshold = float(hot.loc[formation_in_sample(hot["date"], oos), "breadth"].median())
            tags = breadth_tags(groups, breadth, threshold, ds.eligible)
            picks = form_all(ds, groups, lb, cfg, tags=tags)
            record(
                picks,
                "H3",
                f"lookback {lb}m",
                scheme,
                ("hot_laggards_broad", "hot_laggards_narrow", "hot_laggards", "universe"),
            )

            log.info("H5 %s: %dm", scheme, lb)
            for flag, label in (("high_vol", "volatile market"), ("down", "falling market")):
                switched = pd.concat([switch_picks(mp, state[flag]), mp])
                record(
                    switched,
                    "H5",
                    f"lookback {lb}m, laggards in {label}",
                    scheme,
                    ("switch", "hot_laggards", "hot_leaders", "universe"),
                )

    h4 = main["returns"][
        main["returns"]["basket"].isin(["hot_leaders", "universe_leaders", "universe"])
    ]
    h4 = h4.assign(hypothesis="H4", variant="lookback " + h4["lookback"].astype(str) + "m")
    frames.append(h4.drop(columns="lookback"))
    return {"experiments": pd.concat(frames, ignore_index=True), "market_state": state}


def load_experiments() -> dict[str, pd.DataFrame]:
    return {
        n: pd.read_parquet(config.RESULTS / f"{n}.parquet") for n in ("experiments", "market_state")
    }


def spread_cells(exp: pd.DataFrame, hyp: Hypothesis, window: Window) -> pd.DataFrame:
    """Annualised mean and Newey-West t of the hypothesis spread for every cell."""
    e = exp[(exp["hypothesis"] == hyp.key) & exp["month"].between(window.start, window.end)]
    wide = e.pivot_table(
        index=["variant", "scheme", "holding", "month"], columns="basket", values="net"
    )
    rows = []
    for (variant, scheme, holding), block in wide.groupby(level=[0, 1, 2]):
        diff = (block[hyp.long] - block[hyp.short]).dropna()
        mean, t = nw_tstat(diff, lags=int(holding))
        rows.append(
            {
                "variant": variant,
                "scheme": scheme,
                "holding": int(holding),
                "ann": mean * 12,
                "t": t,
                "months": len(diff),
            }
        )
    return pd.DataFrame(rows)


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    order = p.sort_values()
    m = len(order)
    adj = (order * (m - np.arange(m))).cummax().clip(upper=1.0)
    return adj.reindex(p.index)


def scoreboard(exp: pd.DataFrame, wins: dict[str, Window], alpha: float = 0.05) -> pd.DataFrame:
    """The pre-registered test of every hypothesis, one row each."""
    rows = []
    for hyp in HYPOTHESES:
        is_cells = spread_cells(exp, hyp, wins["in_sample"])
        oos_cells = spread_cells(exp, hyp, wins["out_of_sample"])
        best = is_cells.loc[is_cells["t"].idxmax()]
        key = ["variant", "scheme", "holding"]
        chosen = oos_cells.set_index(key).loc[tuple(best[key])]
        rows.append(
            {
                "hypothesis": hyp.key,
                "title": hyp.title,
                "spread": f"{hyp.long} − {hyp.short}",
                "variant": best["variant"],
                "scheme": best["scheme"],
                "holding": int(best["holding"]),
                "is_ann": best["ann"],
                "is_t": best["t"],
                "passes_is": bool(best["t"] >= 2),
                "oos_ann": chosen["ann"],
                "oos_t": chosen["t"],
                "oos_p": float(norm.sf(chosen["t"])),
                "oos_median_all_cells": oos_cells["ann"].median(),
                "oos_share_positive": (oos_cells["ann"] > 0).mean(),
                "cells": len(oos_cells),
            }
        )
    board = pd.DataFrame(rows).set_index("hypothesis")
    board["oos_p_holm"] = holm(board["oos_p"])
    board["confirmed"] = board["oos_p_holm"] < alpha
    return board


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = StudyConfig()
    out = run(load_dataset(), cfg)
    for name, frame in out.items():
        frame.to_parquet(config.RESULTS / f"{name}.parquet")
        log.info("wrote %s: %d rows", name, len(frame))


if __name__ == "__main__":
    main()
