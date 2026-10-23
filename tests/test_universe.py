import pandas as pd

from usom.universe import (
    load_membership,
    members_on,
    membership_matrix,
    membership_spells,
    normalize_ticker,
)

CSV = """date,tickers
2000-01-03,"AAA,BRK.B,FB"
2000-03-01,"AAA,BRK.B,CCC"
2000-06-01,"AAA,CCC,FB"
"""


def _membership(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(CSV)
    return load_membership(path)


def test_normalize_ticker_maps_dots_and_renames():
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker(" fb ") == "META"
    assert normalize_ticker("AAPL") == "AAPL"


def test_members_on_uses_latest_change_on_or_before_date(tmp_path):
    m = _membership(tmp_path)
    assert members_on(m, pd.Timestamp("1999-12-31")) == frozenset()
    assert members_on(m, pd.Timestamp("2000-01-03")) == {"AAA", "BRK-B", "META"}
    assert members_on(m, pd.Timestamp("2000-02-29")) == {"AAA", "BRK-B", "META"}
    assert members_on(m, pd.Timestamp("2000-03-01")) == {"AAA", "BRK-B", "CCC"}
    assert members_on(m, pd.Timestamp("2030-01-01")) == {"AAA", "CCC", "META"}


def test_membership_matrix(tmp_path):
    m = _membership(tmp_path)
    dates = pd.DatetimeIndex(["2000-01-31", "2000-04-28", "2000-06-30"])
    mat = membership_matrix(m, dates)
    assert list(mat.columns) == ["AAA", "BRK-B", "CCC", "META"]
    assert mat["BRK-B"].tolist() == [True, True, False]
    assert mat["META"].tolist() == [True, False, True]


def test_membership_spells_split_on_exit_and_reentry(tmp_path):
    spells = membership_spells(_membership(tmp_path))
    meta = spells[spells.ticker == "META"]
    assert meta["start"].tolist() == [pd.Timestamp("2000-01-03"), pd.Timestamp("2000-06-01")]
    assert meta["end"].iloc[0] == pd.Timestamp("2000-03-01")
    assert pd.isna(meta["end"].iloc[1])
    brk = spells[spells.ticker == "BRK-B"].iloc[0]
    assert (brk.start, brk.end) == (pd.Timestamp("2000-01-03"), pd.Timestamp("2000-06-01"))
