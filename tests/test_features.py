import numpy as np
import pandas as pd
import pytest

from usom.features import abnormal_volatility, group_breadth, market_returns, market_state


def test_market_returns_average_only_eligible_names():
    idx = pd.bdate_range("2020-01-01", periods=4)
    daily = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04], "B": [0.03, 0.04, 0.05, 0.06]}, idx)
    eligible = pd.DataFrame({"A": [True, True], "B": [False, True]}, index=idx[[0, 2]])
    m = market_returns(daily, eligible)
    assert m.tolist() == pytest.approx([0.01, 0.02, 0.04, 0.05])


def test_market_state_uses_only_past_volatility():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=2000)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), idx)
    market.iloc[1500:] *= 4  # a volatile regime late in the sample
    dates = pd.DatetimeIndex(pd.Series(idx, idx).groupby(idx.to_period("M")).max().to_numpy())
    state = market_state(market, dates)
    assert state["high_vol"].iloc[:30].isna().all()  # too little history to judge
    assert state["high_vol"].loc[: idx[1400]].mean() < 0.7
    assert state["high_vol"].loc[idx[1600] :].all()
    # Changing returns after a date leaves that date's state untouched.
    shocked = market.copy()
    shocked.loc[idx[1000] :] += 0.05
    cut = dates[dates < idx[1000]]
    pd.testing.assert_frame_equal(state.loc[cut], market_state(shocked, dates).loc[cut])


def test_abnormal_volatility_flags_a_recent_burst():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    daily = pd.DataFrame(rng.normal(0, 0.01, (400, 2)), idx, columns=["CALM", "NEWS"])
    daily.iloc[-21:, 1] *= 5
    market = pd.Series(0.0, idx)
    ratio = abnormal_volatility(daily, market, idx[[-1]], window=21).iloc[0]
    assert ratio["NEWS"] > 3 > ratio["CALM"]


def test_group_breadth_counts_members_beating_the_median_stock():
    rets = pd.Series({"A1": 0.5, "A2": 0.4, "A3": -0.1, "B1": 0.0, "B2": 0.1, "B3": 0.2})
    groups = pd.Series({t: t[0] for t in rets.index})
    b = group_breadth(groups, rets)  # median is 0.15
    assert b["A"] == pytest.approx(2 / 3)
    assert b["B"] == pytest.approx(1 / 3)
