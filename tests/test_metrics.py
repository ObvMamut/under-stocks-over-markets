import numpy as np
import pandas as pd
import pytest

from usom.metrics import ann_vol, cagr, factor_regression, max_drawdown, nw_tstat, sharpe


def test_basic_stats_match_hand_computation():
    r = pd.Series([0.10, -0.05, 0.02, 0.03])
    assert cagr(r) == pytest.approx((1.1 * 0.95 * 1.02 * 1.03) ** 3 - 1)
    assert ann_vol(r) == pytest.approx(r.std() * np.sqrt(12))
    rf = pd.Series([0.01] * 4)
    assert sharpe(r, rf) == pytest.approx((r - 0.01).mean() / r.std() * np.sqrt(12))
    assert max_drawdown(r) == pytest.approx(0.95 - 1)


def test_newey_west_matches_plain_t_without_lags():
    x = pd.Series(np.random.default_rng(0).normal(0.01, 0.05, 240))
    mean, t = nw_tstat(x, lags=0)
    assert mean == pytest.approx(x.mean())
    # HAC with zero lags is White's estimator: n rather than n-1 in the variance.
    assert t == pytest.approx(x.mean() / (x.std(ddof=0) / np.sqrt(len(x))))


def test_factor_regression_recovers_alpha_and_beta():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=300, freq="ME")
    f = pd.DataFrame(rng.normal(0, 0.04, (300, 2)), idx, columns=["Mkt-RF", "Mom"])
    y = 0.002 + 1.2 * f["Mkt-RF"] - 0.3 * f["Mom"] + rng.normal(0, 0.001, 300)
    out = factor_regression(y, f, lags=3, cols=["Mkt-RF", "Mom"])
    assert out.loc["alpha", "coef"] == pytest.approx(0.002, abs=2e-4)
    assert out.loc["Mkt-RF", "coef"] == pytest.approx(1.2, abs=0.01)
    assert out.loc["Mom", "coef"] == pytest.approx(-0.3, abs=0.01)
    assert out.attrs["n"] == 300
