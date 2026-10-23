import pandas as pd
import pytest

from usom.factors import parse_french_csv

TEXT = """This file was created using the 202608 CRSP database.
Missing data are indicated by -99.99 or -999.

,Mkt-RF,SMB,RF
202601,    1.50,   -0.25,    0.30
202602,   -2.00,  -99.99,    0.31

 Annual Factors: January-December
,Mkt-RF,SMB,RF
2026,   10.00,    1.00,    3.00
"""


def test_parse_french_csv_reads_first_monthly_block_as_decimals():
    df = parse_french_csv(TEXT)
    assert list(df.columns) == ["Mkt-RF", "SMB", "RF"]
    assert list(df.index) == [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-28")]
    assert df.loc["2026-01-31", "Mkt-RF"] == pytest.approx(0.015)
    assert pd.isna(df.loc["2026-02-28", "SMB"])
    assert len(df) == 2
