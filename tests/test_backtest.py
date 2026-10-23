import numpy as np
import pandas as pd
import pytest

from usom.backtest import (
    cohort_weights,
    execution_dates,
    forward_returns,
    holding_returns,
    month_end_dates,
    next_month_end,
    run_portfolio,
)
from usom.metrics import cagr


def _close(days=400, growth=None):
    idx = pd.bdate_range("2020-01-01", periods=days)
    growth = growth or {"A": 0.001, "B": 0.0}
    return pd.DataFrame({t: 100 * (1 + g) ** np.arange(days) for t, g in growth.items()}, idx)


def test_month_end_and_execution_dates():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    me = month_end_dates(idx)
    assert list(me) == list(pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"]))
    exe = execution_dates(idx, me)
    assert exe.iloc[0] == pd.Timestamp("2020-02-03")
    assert pd.isna(exe.iloc[-1])


def test_next_month_end_handles_months_ending_on_weekends():
    got = next_month_end(pd.DatetimeIndex(["2020-02-28", "2020-01-31"]))
    assert list(got) == list(pd.to_datetime(["2020-03-31", "2020-02-29"]))


def test_forward_returns_measure_execution_to_execution_and_zero_after_delisting():
    close = _close()
    close.loc["2020-06-15":, "B"] = np.nan  # delisted mid-June
    me = month_end_dates(close.index)
    fwd = forward_returns(close, me)
    exe = execution_dates(close.index, me)
    t = me[0]
    expected = close.loc[exe[me[1]], "A"] / close.loc[exe[t], "A"] - 1
    assert fwd.loc[t, "A"] == pytest.approx(expected)
    assert (fwd.loc["2020-06-30":, "B"] == 0).all()


def test_constant_growth_portfolio_has_expected_cagr_and_turnover():
    close = _close(days=800)
    me = month_end_dates(close.index)
    fwd = forward_returns(close, me)
    picks = pd.DataFrame({"date": fwd.index, "ticker": "A"})
    w = cohort_weights(picks, fwd.index, close.columns)
    res = run_portfolio(w, fwd, holding=3, cost_bps=0.0)
    per_day = 1.001
    days = close.index.to_series().groupby(close.index.to_period("M")).size()
    assert cagr(res["gross"]) == pytest.approx(per_day ** (days.mean() * 12) - 1, rel=0.02)
    assert res["turnover"].iloc[0] == pytest.approx(0.5)  # initial buy, one-way
    assert res["turnover"].iloc[1:].max() == pytest.approx(0.0, abs=1e-12)


def test_overlapping_cohorts_average_and_costs_follow_turnover():
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    fwd = pd.DataFrame({"A": [0.1, 0.1, 0.1, 0.1], "B": [0.0, 0.0, 0.0, 0.0]}, idx)
    picks = pd.DataFrame({"date": idx, "ticker": ["A", "B", "A", "B"]})
    w = cohort_weights(picks, idx, fwd.columns)
    res = run_portfolio(w, fwd, holding=2, cost_bps=100.0)
    # Month 2 onward holds half A, half B.
    assert res["gross"].tolist() == pytest.approx([0.05, 0.05, 0.05])
    first_trade = 1.0
    assert res["net"].iloc[0] == pytest.approx(0.05 - first_trade * 0.01)
    # Drift moves A from 0.5 to 0.55/1.05; rebalancing back costs |Δ| on both legs.
    drift_a = 0.55 / 1.05
    assert res["turnover"].iloc[1] == pytest.approx(drift_a - 0.5)
    assert res["net"].iloc[1] == pytest.approx(0.05 - 2 * (drift_a - 0.5) * 0.01)
    assert list(res.index) == list(pd.to_datetime(["2020-03-31", "2020-04-30", "2020-05-31"]))


def test_holding_returns_compound_forward():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    fwd = pd.DataFrame({"A": [0.1, 0.2, -0.1, 0.0]}, idx)
    h = holding_returns(fwd, 2)["A"]
    assert h.iloc[0] == pytest.approx(1.1 * 1.2 - 1)
    assert h.iloc[2] == pytest.approx(0.9 - 1)
    assert pd.isna(h.iloc[3])
