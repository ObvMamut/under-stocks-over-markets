"""Monthly factors and the T-bill rate from the Ken French Data Library (public CSV zips)."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from usom.config import FRENCH_BASE, FRENCH_FILES

_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")


def parse_french_csv(text: str) -> pd.DataFrame:
    """The first monthly block of a French CSV, as decimals indexed by month end.

    Each file has a free-text preamble, a header line starting with a comma, then
    `YYYYMM, v1, v2, ...` rows; later blocks (annual figures) are ignored.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if _MONTH_ROW.match(line))
    header = [h.strip() for h in lines[start - 1].split(",")[1:]]
    rows = []
    for line in lines[start:]:
        if not _MONTH_ROW.match(line):
            break
        cells = [c.strip() for c in line.split(",")]
        rows.append([cells[0], *map(float, cells[1:])])
    df = pd.DataFrame(rows, columns=["month", *header])
    df.index = pd.to_datetime(df.pop("month"), format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = None
    return df.mask(df <= -99.99) / 100


def load_factors(raw_dir: Path) -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RMW, CMA, RF, Mom, ST_Rev (monthly, decimals). Downloads if absent."""
    frames = []
    for name in FRENCH_FILES.values():
        path = raw_dir / name
        if not path.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)
            resp = requests.get(FRENCH_BASE + name, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        with zipfile.ZipFile(path) as zf:
            text = io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="latin-1").read()
        frames.append(parse_french_csv(text))
    return pd.concat(frames, axis=1, join="inner")
