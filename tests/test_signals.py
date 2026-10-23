import numpy as np
import pandas as pd

from usom.signals import form_baskets


def _setup():
    # Five groups of six; group strength rises with the group number.
    tickers = [f"G{g}_{i}" for g in range(5) for i in range(6)]
    groups = pd.Series({t: t.split("_")[0] for t in tickers})
    rets = pd.Series({t: int(t[1]) * 0.10 + int(t[-1]) * 0.01 for t in tickers})
    return groups, rets


def _basket(df, name):
    return sorted(df.loc[df.basket == name, "ticker"])


def test_hot_laggards_are_the_weakest_names_in_the_strongest_group():
    groups, rets = _setup()
    b = form_baskets(groups, rets, hot_frac=0.2, tail_frac=1 / 3, min_size=5)
    assert _basket(b, "hot_laggards") == ["G4_0", "G4_1"]
    assert _basket(b, "hot_leaders") == ["G4_4", "G4_5"]
    assert _basket(b, "hot_all") == [f"G4_{i}" for i in range(6)]
    assert _basket(b, "cold_laggards") == ["G0_0", "G0_1"]
    assert _basket(b, "cold_all") == [f"G0_{i}" for i in range(6)]
    assert len(_basket(b, "all_laggards")) == 10
    assert _basket(b, "universe_laggards") == sorted(rets.nsmallest(10).index)
    assert len(_basket(b, "universe")) == 30
    assert set(b.loc[b.basket == "hot_laggards", "group"]) == {"G4"}


def test_ineligible_tickers_and_small_groups_are_ignored():
    groups, rets = _setup()
    rets["G4_0"] = np.nan  # not eligible this month
    groups = groups.drop(["G3_0", "G3_1"])  # G3 shrinks below min_size
    b = form_baskets(groups, rets, hot_frac=0.2, tail_frac=1 / 3, min_size=5)
    assert _basket(b, "hot_laggards") == ["G4_1", "G4_2"]  # round(5/3) = 2 names
    assert "G4_0" not in _basket(b, "universe")
    assert not b.loc[b.basket == "all_laggards", "group"].eq("G3").any()


def test_group_that_falls_below_min_size_after_eligibility_is_dropped():
    groups, rets = _setup()
    rets[["G4_0", "G4_1"]] = np.nan  # G4 keeps only 4 eligible names
    b = form_baskets(groups, rets, hot_frac=0.2, tail_frac=1 / 3, min_size=5)
    assert _basket(b, "hot_laggards") == ["G3_0", "G3_1"]
