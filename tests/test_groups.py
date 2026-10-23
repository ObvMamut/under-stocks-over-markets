import numpy as np
import pandas as pd

from usom.groups import correlation_clusters, label_groups


def test_label_groups_drops_unlabelled_and_small_groups():
    labels = pd.Series({"A": "Tech", "B": "Tech", "C": "Tech", "D": "Banks", "E": None})
    g = label_groups(pd.Index(["A", "B", "C", "D", "E", "F"]), labels, min_size=2)
    assert g.to_dict() == {"A": "Tech", "B": "Tech", "C": "Tech"}


def _block_returns(n_blocks=4, per_block=10, days=252, seed=0):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, days)
    cols, data = [], []
    for b in range(n_blocks):
        theme = rng.normal(0, 0.01, days)
        for i in range(per_block):
            cols.append(f"B{b}_{i}")
            data.append(market + theme + rng.normal(0, 0.005, days))
    return pd.DataFrame(np.array(data).T, columns=cols)


def test_correlation_clusters_recover_planted_blocks():
    rets = _block_returns()
    clusters = correlation_clusters(rets, avg_size=10, min_size=5)
    assert len(clusters) == 40
    block = clusters.index.str.split("_").str[0]
    # Every planted block maps to exactly one cluster, and blocks never share one.
    assert clusters.groupby(block).nunique().eq(1).all()
    assert clusters.nunique() == 4


def test_correlation_clusters_skip_sparse_tickers_and_small_clusters():
    rets = _block_returns(n_blocks=3, per_block=10)
    rets["SPARSE"] = np.nan
    rets.loc[:200, "SPARSE"] = 0.01
    rets[["LONE_0", "LONE_1"]] = np.random.default_rng(1).normal(0, 0.03, (252, 2))
    clusters = correlation_clusters(rets, avg_size=8, min_size=5)
    assert "SPARSE" not in clusters.index
    assert clusters.map(clusters.value_counts()).min() >= 5
