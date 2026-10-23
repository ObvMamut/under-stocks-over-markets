"""Paths, data sources and the parameter grid for the study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
RESULTS = DATA / "results"

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
    "st_rev": "F-F_ST_Reversal_Factor_CSV.zip",
}

# Old symbols that Yahoo only serves under the company's current ticker.
TICKER_ALIASES = {"FB": "META"}

BENCHMARK = "SPY"
PRICE_START = "1998-01-01"


@dataclass(frozen=True)
class StudyConfig:
    """Everything that defines one run of the grid."""

    schemes: tuple[str, ...] = ("sector", "industry", "cluster")
    lookbacks: tuple[int, ...] = (1, 3, 6, 12)  # months
    holdings: tuple[int, ...] = (1, 3, 6)  # months
    hot_frac: float = 0.2  # share of groups called hot (and cold)
    tail_frac: float = 1 / 3  # share of a group called laggards (and leaders)
    min_group_size: int = 5
    cluster_window: int = 252  # trading days of history used to form clusters
    cluster_avg_size: int = 12  # target members per cluster
    cost_bps: float = 10.0  # one-way, per unit of traded weight
    cost_grid: tuple[float, ...] = (0.0, 10.0, 25.0)
    # Main sample: formation dates on which at least this share of index members has
    # usable prices. The earlier, thinner years from `extended_start` serve as the
    # in-sample period for choosing parameters; the main sample is the out-of-sample test.
    min_coverage: float = 0.8
    extended_start: str = "2005-01-31"
    baskets: tuple[str, ...] = field(
        default=(
            "hot_laggards",
            "hot_leaders",
            "hot_all",
            "cold_laggards",
            "cold_all",
            "all_laggards",
            "universe_laggards",
            "universe",
        )
    )
