import numpy as np
import pandas as pd
import pytest

from usom.analysis import Window
from usom.experiments import (
    breadth_tags,
    formation_in_sample,
    holm,
    quiet_loud_tags,
    switch_picks,
)


def test_formation_in_sample_excludes_the_month_before_the_test_window():
    oos = Window("oos", pd.Timestamp("2017-10-31"), pd.Timestamp("2026-08-31"))
    dates = pd.to_datetime(["2017-08-31", "2017-09-29", "2017-10-31"])
    # A September formation earns its return in October, which is already out-of-sample.
    assert formation_in_sample(dates, oos).tolist() == [True, False, False]


def test_quiet_loud_tags_split_at_each_dates_median():
    idx = pd.to_datetime(["2020-01-31", "2020-02-28"])
    ratio = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 1.0], "C": [3.0, np.nan]}, idx)
    tags = quiet_loud_tags(ratio)
    assert tags.loc["2020-01-31"].tolist() == ["quiet", "quiet", "loud"]
    assert tags.loc["2020-02-28", "A"] == "loud" and tags.loc["2020-02-28", "B"] == "quiet"
    assert pd.isna(tags.loc["2020-02-28", "C"])


def test_breadth_tags_follow_each_tickers_group():
    d = pd.Timestamp("2020-01-31")
    groups = pd.DataFrame({"date": d, "ticker": ["A", "B", "C"], "group": ["X", "X", "Y"]})
    breadth = pd.DataFrame({"date": d, "group": ["X", "Y"], "breadth": [0.8, 0.4]})
    like = pd.DataFrame(index=[d], columns=["A", "B", "C", "D"])
    tags = breadth_tags(groups, breadth, threshold=0.6, like=like)
    assert tags.loc[d, ["A", "B", "C"]].tolist() == ["broad", "broad", "narrow"]
    assert pd.isna(tags.loc[d, "D"])


def test_switch_picks_choose_laggards_only_when_state_is_true():
    d1, d2, d3 = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    picks = pd.DataFrame(
        {
            "date": [d1, d1, d2, d2, d3, d3],
            "basket": ["hot_laggards", "hot_leaders"] * 3,
            "ticker": ["L1", "W1", "L2", "W2", "L3", "W3"],
            "group": "G",
        }
    )
    state = pd.Series([True, False, pd.NA], index=[d1, d2, d3], dtype="boolean")
    out = switch_picks(picks, state)
    assert out["ticker"].tolist() == ["L1", "W2", "W3"]
    assert set(out["basket"]) == {"switch"}


def test_holm_adjustment():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    adj = holm(p)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # never below a smaller p's adjusted value
