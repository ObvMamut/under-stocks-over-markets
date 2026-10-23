import numpy as np
import pandas as pd

from usom.prices import coverage, drop_spikes, eligibility, valid_spells


def _prices():
    idx = pd.bdate_range("2000-01-03", "2000-12-29")
    p = pd.DataFrame(100.0, index=idx, columns=["OLD", "REUSED", "LATE_IPO"])
    p.loc[:"2000-06-30", "REUSED"] = np.nan  # Yahoo history starts mid-year
    p.loc["2000-09-01":, "OLD"] = np.nan  # delisted
    p.loc[:"2000-03-31", "LATE_IPO"] = np.nan
    return p


def _spells():
    return pd.DataFrame(
        {
            "ticker": ["OLD", "REUSED", "LATE_IPO"],
            "start": pd.to_datetime(["1999-01-01", "2000-01-03", "2000-04-03"]),
            "end": [pd.NaT, pd.NaT, pd.NaT],
        }
    )


def test_spell_is_untrusted_when_yahoo_history_starts_after_it():
    valid = valid_spells(_spells(), _prices())
    # OLD predates the download: judged against the download start, so trusted.
    # REUSED: the index spell began in January but Yahoo's series only in July,
    # which is the signature of a ticker taken over by another company.
    # LATE_IPO: joined the index the day its quotes begin.
    assert valid.tolist() == [True, False, True]


def test_eligibility_requires_member_trusted_spell_and_quote():
    prices = _prices()
    dates = pd.DatetimeIndex(["2000-02-29", "2000-05-31", "2000-08-31", "2000-10-31"])
    member = pd.DataFrame(True, index=dates, columns=prices.columns)
    member.loc["2000-02-29", "LATE_IPO"] = False
    spells = _spells()
    elig = eligibility(member, spells, prices, valid_spells(spells, prices))
    assert elig["OLD"].tolist() == [True, True, True, False]  # no quote after delisting
    assert not elig["REUSED"].any()
    assert elig["LATE_IPO"].tolist() == [False, True, True, True]
    assert coverage(member, elig).iloc[-1] == 1 / 3


def test_drop_spikes_removes_reverting_ticks_but_keeps_real_moves():
    idx = pd.bdate_range("2020-01-01", periods=6)
    p = pd.DataFrame(
        {
            "BADTICK": [10.0, 10.0, 0.1, 10.0, 10.0, 10.0],
            "CRASH": [10.0, 10.0, 1.0, 1.0, 1.1, 1.0],
            "JUMP": [10.0, 10.0, 30.0, 28.0, 29.0, 30.0],
        },
        idx,
    )
    clean = drop_spikes(p)
    assert clean["BADTICK"].isna().tolist() == [False, False, True, False, False, False]
    pd.testing.assert_frame_equal(clean[["CRASH", "JUMP"]], p[["CRASH", "JUMP"]])
