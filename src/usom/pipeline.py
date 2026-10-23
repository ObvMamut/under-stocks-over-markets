"""Build the dataset and run the full grid: `uv run python -m usom.pipeline`.

Outputs (parquet, under data/results/):
- groups.parquet    each month's groups per scheme
- picks.parquet     every basket's members at every formation date, per scheme and lookback
- returns.parquet   monthly portfolio returns per scheme, lookback, holding and basket
- outcomes.parquet  each scored pick's H-month return beside its group's and the universe's
- monthly.parquet   coverage, SPY and factor data on the same monthly calendar
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import pandas as pd

from usom import config
from usom.backtest import (
    cohort_weights,
    forward_returns,
    month_end_dates,
    next_month_end,
    pick_outcomes,
    run_portfolio,
)
from usom.classify import fetch_classification
from usom.config import StudyConfig
from usom.factors import load_factors
from usom.groups import correlation_clusters, label_groups
from usom.prices import coverage, drop_spikes, eligibility, fetch_prices, valid_spells
from usom.signals import form_baskets
from usom.universe import (
    download_membership,
    load_membership,
    membership_matrix,
    membership_spells,
)

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    close: pd.DataFrame  # daily adjusted closes, bad ticks removed, not forward-filled
    signal_dates: pd.DatetimeIndex  # last trading day of each month
    member: pd.DataFrame  # signal_dates x tickers: index member that day
    eligible: pd.DataFrame  # member, trusted spell, and quoted that day
    labels: pd.DataFrame  # ticker -> sector, industry
    factors: pd.DataFrame  # monthly French factors and RF


def load_dataset(refresh_membership: bool = False) -> Dataset:
    path = config.RAW / "sp500_history.csv"
    if refresh_membership or not path.exists():
        download_membership(path)
    membership = load_membership(path)
    tickers = sorted(set().union(*membership))

    close = fetch_prices(
        [*tickers, config.BENCHMARK], config.PRICE_START, config.CACHE / "prices.parquet"
    )
    close = drop_spikes(close)
    # Membership is only known up to its last change row; stop at the end of that month.
    last = membership.index[-1] + pd.offsets.MonthEnd(1)
    signal_dates = month_end_dates(close.index[close.index <= last])

    member = membership_matrix(membership, signal_dates)
    spells = membership_spells(membership)
    stocks = close.drop(columns=config.BENCHMARK)
    eligible = eligibility(member, spells, stocks, valid_spells(spells, stocks))

    labels = fetch_classification(stocks.columns, config.CACHE / "classification.csv")
    factors = load_factors(config.RAW / "french")
    return Dataset(close, signal_dates, member, eligible, labels, factors)


def lookback_returns(ds: Dataset, months: int) -> pd.DataFrame:
    """Trailing `months` return at each signal date, NaN where the stock is not eligible."""
    px = ds.close.reindex(columns=ds.eligible.columns).ffill(limit=5).loc[ds.signal_dates]
    ret = px / px.shift(months) - 1
    return ret.where(ds.eligible)


def group_panel(ds: Dataset, scheme: str, cfg: StudyConfig) -> pd.DataFrame:
    """Long frame (date, ticker, group) of each month's groups for one scheme."""
    frames = []
    daily = ds.close.reindex(columns=ds.eligible.columns).pct_change(fill_method=None)
    for date in ds.signal_dates:
        members = ds.eligible.columns[ds.eligible.loc[date]]
        if len(members) == 0:
            continue
        if scheme == "cluster":
            window = daily.loc[:date, members].iloc[-cfg.cluster_window :]
            if len(window) < cfg.cluster_window:
                continue
            g = correlation_clusters(window, cfg.cluster_avg_size, cfg.min_group_size)
        else:
            g = label_groups(members, ds.labels[scheme], cfg.min_group_size)
        frames.append(pd.DataFrame({"date": date, "ticker": g.index, "group": g.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def form_all(ds: Dataset, groups: pd.DataFrame, lookback: int, cfg: StudyConfig) -> pd.DataFrame:
    """Every basket's members at every formation date for one scheme and lookback."""
    rets = lookback_returns(ds, lookback)
    by_date = dict(tuple(groups.groupby("date")))
    frames = []
    for date in ds.signal_dates:
        if date not in by_date:
            continue
        g = by_date[date].set_index("ticker")["group"]
        b = form_baskets(g, rets.loc[date], cfg.hot_frac, cfg.tail_frac, cfg.min_group_size)
        frames.append(b.assign(date=date))
    return pd.concat(frames, ignore_index=True)


# Baskets whose individual picks are scored against their own group, and the basket
# holding that whole group.
SCORED = {"hot_laggards": "hot_all", "hot_leaders": "hot_all", "cold_laggards": "cold_all"}


def run_grid(ds: Dataset, cfg: StudyConfig) -> dict[str, pd.DataFrame]:
    """All results of the study, keyed by the name they are saved under."""
    fwd = forward_returns(ds.close.reindex(columns=ds.eligible.columns), ds.signal_dates)
    out: dict[str, list[pd.DataFrame]] = {"groups": [], "picks": [], "returns": [], "outcomes": []}
    for scheme in cfg.schemes:
        log.info("groups: %s", scheme)
        groups = group_panel(ds, scheme, cfg)
        out["groups"].append(groups.assign(scheme=scheme))
        for lookback in cfg.lookbacks:
            log.info("baskets: %s, %d-month lookback", scheme, lookback)
            picks = form_all(ds, groups, lookback, cfg)
            keys = {"scheme": scheme, "lookback": lookback}
            out["picks"].append(picks.assign(**keys))
            for basket, p in picks.groupby("basket"):
                w = cohort_weights(p, fwd.index, fwd.columns)
                for holding in cfg.holdings:
                    res = run_portfolio(w, fwd, holding, cfg.cost_bps)
                    out["returns"].append(
                        res.rename_axis("month")
                        .reset_index()
                        .assign(**keys, holding=holding, basket=basket)
                    )
            for holding in cfg.holdings:
                for basket, peers in SCORED.items():
                    o = pick_outcomes(picks, fwd, holding, basket, peers)
                    out["outcomes"].append(o.assign(**keys, holding=holding))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in out.items()}


def monthly_frame(ds: Dataset) -> pd.DataFrame:
    """Coverage, SPY return and factors on the month-end calendar the returns use."""
    spy = forward_returns(ds.close[[config.BENCHMARK]], ds.signal_dates)[config.BENCHMARK]
    spy.index = next_month_end(spy.index)
    at_signal = pd.DataFrame(
        {
            "coverage": coverage(ds.member, ds.eligible),
            "n_members": ds.member.sum(axis=1),
            "n_eligible": ds.eligible.sum(axis=1),
        }
    )
    # Label each row by the month whose return it governs (formed at the previous month end).
    at_signal.index = next_month_end(at_signal.index)
    out = at_signal.add_suffix("_at_formation").join(spy.rename("spy"), how="outer")
    return out.join(ds.factors, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-membership", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = StudyConfig()
    ds = load_dataset(args.refresh_membership)
    results = run_grid(ds, cfg)
    results["monthly"] = monthly_frame(ds)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_parquet(config.RESULTS / f"{name}.parquet")
        log.info("wrote %s: %d rows", name, len(frame))


if __name__ == "__main__":
    main()
