"""Turn one month's groups and trailing returns into the baskets being compared."""

from __future__ import annotations

import pandas as pd


def _tail(rets: pd.Series, frac: float, lowest: bool) -> list[str]:
    """The `frac` share (at least one) of names with the lowest or highest return.

    Ties break on ticker so the selection is deterministic.
    """
    k = max(1, round(frac * len(rets)))
    frame = pd.DataFrame({"ret": rets, "ticker": rets.index})
    ordered = frame.sort_values(["ret", "ticker"], ascending=[lowest, True])
    return ordered["ticker"].iloc[:k].tolist()


def form_baskets(
    groups: pd.Series,
    lookback_ret: pd.Series,
    hot_frac: float,
    tail_frac: float,
    min_size: int,
    group_ret: pd.Series | None = None,
    tags: pd.Series | None = None,
) -> pd.DataFrame:
    """One formation date's picks, as rows (basket, ticker, group).

    `groups` maps ticker -> group; `lookback_ret` maps every eligible ticker to its return
    over the lookback (NaN = not eligible). Groups are ranked by the equal-weight mean of
    their members' returns; the top `hot_frac` are hot and the bottom `hot_frac` cold.

    `group_ret`, if given, ranks the groups instead of `lookback_ret`, so groups and the
    stocks inside them can be judged over different horizons. `tags`, if given, maps
    tickers to a label, and every hot-group laggard is also added to
    `hot_laggards_<label>`.
    """
    rets = lookback_ret.dropna()
    if group_ret is not None:
        rets = rets[group_ret.reindex(rets.index).notna()]
    gret = rets if group_ret is None else group_ret.reindex(rets.index)
    g = groups.reindex(rets.index).dropna()
    g = g[g.map(g.value_counts()) >= min_size]

    rows: list[tuple[str, str, str]] = []

    def add(basket: str, tickers: list[str]) -> None:
        rows.extend((basket, t, g.get(t, "")) for t in tickers)

    add("universe", sorted(rets.index))
    add("universe_laggards", _tail(rets, tail_frac, lowest=True))
    add("universe_leaders", _tail(rets, tail_frac, lowest=False))

    if g.empty:
        return pd.DataFrame(rows, columns=["basket", "ticker", "group"])

    members = {name: rets[idx.index] for name, idx in g.groupby(g)}
    strength = pd.Series({name: gret[r.index].mean() for name, r in members.items()})
    n_hot = max(1, round(hot_frac * len(strength)))
    ranked = strength.sort_values(ascending=False, kind="stable")
    hot, cold = ranked.index[:n_hot], ranked.index[-n_hot:]

    for name, r in members.items():
        laggards = _tail(r, tail_frac, lowest=True)
        add("all_laggards", laggards)
        if name in hot:
            add("hot_laggards", laggards)
            add("hot_leaders", _tail(r, tail_frac, lowest=False))
            add("hot_all", sorted(r.index))
            if tags is not None:
                for t in laggards:
                    if pd.notna(tags.get(t)):
                        add(f"hot_laggards_{tags[t]}", [t])
        if name in cold:
            add("cold_laggards", laggards)
            add("cold_all", sorted(r.index))

    return pd.DataFrame(rows, columns=["basket", "ticker", "group"])
