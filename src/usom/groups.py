"""The three ways of cutting the universe into groups ("sectors").

- sector:   Yahoo's 11 broad sectors
- industry: Yahoo's ~145 industries
- cluster:  stocks that moved together over the trailing window, re-formed every month
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage


def label_groups(tickers: pd.Index, labels: pd.Series, min_size: int) -> pd.Series:
    """Ticker -> group label, keeping only labelled tickers in groups of at least min_size."""
    g = labels.reindex(tickers).dropna()
    sizes = g.map(g.value_counts())
    return g[sizes >= min_size].rename("group")


def correlation_clusters(
    returns: pd.DataFrame, avg_size: int, min_size: int, min_obs: float = 0.9
) -> pd.Series:
    """Ticker -> cluster id from a trailing window of daily returns (rows = days).

    Market moves are removed first (each day's equal-weight mean), so stocks cluster on
    what they share beyond the market. On z-scored residuals, Euclidean distance is
    sqrt(2T(1 - rho)), so Ward linkage clusters on correlation distance while keeping
    cluster sizes balanced.
    """
    enough = returns.notna().mean() >= min_obs
    r = returns.loc[:, enough]
    resid = r.sub(r.mean(axis=1), axis=0).fillna(0.0)
    std = resid.std()
    resid = resid.loc[:, std > 0]
    if resid.shape[1] < 2 * min_size:
        return pd.Series(dtype=object, name="group")
    z = (resid - resid.mean()) / resid.std()
    k = max(2, round(z.shape[1] / avg_size))
    tree = linkage(z.T.to_numpy() / np.sqrt(len(z)), method="ward")
    ids = pd.Series(fcluster(tree, t=k, criterion="maxclust"), index=z.columns)
    sizes = ids.map(ids.value_counts())
    return ("C" + ids[sizes >= min_size].astype(str)).rename("group")
