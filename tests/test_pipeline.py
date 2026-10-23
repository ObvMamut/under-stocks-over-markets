import numpy as np
import pandas as pd
import pytest

from usom.backtest import month_end_dates
from usom.config import StudyConfig
from usom.pipeline import Dataset, form_all, group_panel, lookback_returns


def _dataset(seed=0, days=700, n=40):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=days)
    tickers = [f"T{i:02d}" for i in range(n)]
    # Four themes of ten stocks each so clusters and labels line up.
    theme = rng.normal(0, 0.01, (days, 4))
    r = np.repeat(theme, 10, axis=1) + rng.normal(0, 0.01, (days, n))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), idx, tickers)
    signal_dates = month_end_dates(idx)
    member = pd.DataFrame(True, index=signal_dates, columns=tickers)
    labels = pd.DataFrame(
        {
            "sector": [f"S{i // 20}" for i in range(n)],
            "industry": [f"I{i // 10}" for i in range(n)],
        },
        index=tickers,
    )
    return Dataset(close, signal_dates, member, member.copy(), labels, pd.DataFrame())


def _picks_on(ds, date, scheme, cfg):
    groups = group_panel(ds, scheme, cfg)
    picks = form_all(ds, groups, lookback=3, cfg=cfg)
    return picks[picks.date == date].reset_index(drop=True)


@pytest.mark.parametrize("scheme", ["industry", "cluster"])
def test_changing_future_prices_does_not_change_todays_baskets(scheme):
    cfg = StudyConfig(cluster_avg_size=10)
    ds = _dataset()
    date = ds.signal_dates[16]
    before = _picks_on(ds, date, scheme, cfg)

    shocked = ds.close.copy()
    after_date = shocked.index > date
    shocked.loc[after_date] *= np.random.default_rng(9).uniform(0.2, 5.0, shocked.shape[1])
    ds_shocked = Dataset(shocked, ds.signal_dates, ds.member, ds.eligible, ds.labels, ds.factors)
    after = _picks_on(ds_shocked, date, scheme, cfg)

    assert len(before) > 0
    pd.testing.assert_frame_equal(before, after)


def test_lookback_returns_respect_eligibility():
    ds = _dataset()
    date = ds.signal_dates[10]
    ds.eligible.loc[date, "T00"] = False
    rets = lookback_returns(ds, 3)
    assert pd.isna(rets.loc[date, "T00"])
    px = ds.close.loc[ds.signal_dates]
    assert rets.loc[date, "T01"] == pytest.approx(px.loc[date, "T01"] / px["T01"].iloc[7] - 1)
