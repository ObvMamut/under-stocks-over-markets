"""Static charts for the notebook (matplotlib), one function per figure.

Colours follow the entity, never its rank: each basket and scheme keeps one hue in every
chart. Palette and chrome come from the dataviz reference palette (light surface); the
categorical slots used here were run through its validator.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BASKET_COLORS = {
    "hot_laggards": "#2a78d6",
    "hot_leaders": "#eb6834",
    "hot_all": "#1baf7a",
    "cold_laggards": "#eda100",
    "universe": MUTED,
    "spy": INK_2,
}
BASKET_NAMES = {
    "hot_laggards": "Hot-sector laggards",
    "hot_leaders": "Hot-sector leaders",
    "hot_all": "Whole hot sector",
    "cold_laggards": "Cold-sector laggards",
    "cold_all": "Whole cold sector",
    "all_laggards": "Laggards, every sector",
    "universe_laggards": "Laggards, whole index",
    "universe": "Equal-weight S&P 500",
    "spy": "SPY",
}
SCHEME_COLORS = {"sector": "#2a78d6", "industry": "#eb6834", "cluster": "#1baf7a"}
SCHEME_NAMES = {
    "sector": "Broad sectors",
    "industry": "Industries",
    "cluster": "Co-movement clusters",
}
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#a8302f", "#e34948", "#f0efec", "#2a78d6", "#104281"]
)


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
        }
    )


def _pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _end_labels(
    ax, lines: dict[str, pd.Series], min_gap: float, priority: tuple[str, ...] = ()
) -> None:
    """Label line ends with the series name; drop a label that would collide.

    Names in `priority` are placed first, so they are never the ones dropped.
    """
    placed: list[float] = []
    ends = sorted(((name not in priority, -s.dropna().iloc[-1], name) for name, s in lines.items()))
    x = max(s.dropna().index[-1] for s in lines.values())
    for _, neg_y, name in ends:
        y = -neg_y
        pos = np.log10(y) if ax.get_yscale() == "log" else y
        if all(abs(pos - p) >= min_gap for p in placed):
            ax.annotate(
                f" {name}",
                (x, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK_2,
                annotation_clip=False,
            )
            placed.append(pos)


def coverage_chart(coverage: pd.Series, threshold: float, main_start: pd.Timestamp):
    """Share of index members that have usable Yahoo prices, by month."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(coverage.index, coverage.to_numpy(), color=BASKET_COLORS["hot_laggards"])
    ax.axhline(threshold, color=AXIS, linewidth=1)
    ax.axvline(main_start, color=AXIS, linewidth=1)
    ax.annotate(
        f"main sample starts {main_start:%b %Y}\n(coverage ≥ {threshold:.0%})",
        (main_start, threshold),
        xytext=(-8, 18),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_2,
    )
    ax.set_ylim(0, 1)
    _pct_axis(ax)
    ax.set_title("How much of the historical S&P 500 has usable price data")
    ax.set_ylabel("share of index members")
    fig.tight_layout()
    return fig


def growth_chart(returns: dict[str, pd.Series], title: str):
    """Value of $1 (log scale) above drawdowns, one line per basket."""
    from usom.analysis import drawdown, growth

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    curves = {}
    for key, r in returns.items():
        g = growth(r)
        color = BASKET_COLORS.get(key, MUTED)
        main = key == "hot_laggards"
        top.plot(
            g.index,
            g.to_numpy(),
            color=color,
            linewidth=2.4 if main else 1.6,
            zorder=4 if main else 2,
            label=BASKET_NAMES[key],
        )
        dd = drawdown(r)
        bottom.plot(dd.index, dd.to_numpy(), color=color, linewidth=1.2)
        curves[BASKET_NAMES[key]] = g
    top.set_yscale("log")
    top.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    top.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    top.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    _end_labels(top, curves, min_gap=0.035, priority=(BASKET_NAMES["hot_laggards"],))
    top.set_title(title)
    top.set_ylabel("value of $1")
    top.legend(loc="upper left", ncols=2, fontsize=9)
    bottom.set_ylabel("drawdown")
    _pct_axis(bottom)
    fig.tight_layout()
    return fig


def spread_heatmaps(spreads: pd.DataFrame, spread: str, title: str):
    """Lookback x holding grid of one long/short spread, one panel per scheme."""
    s = spreads[spreads["spread"] == spread]
    schemes = [k for k in SCHEME_NAMES if k in set(s["scheme"])]
    vmax = max(0.02, s["ann"].abs().max())
    norm = TwoSlopeNorm(0, -vmax, vmax)
    fig, axes = plt.subplots(1, len(schemes), figsize=(3.6 * len(schemes), 3.6))
    for ax, scheme in zip(np.atleast_1d(axes), schemes, strict=True):
        cell = s[s["scheme"] == scheme]
        ann = cell.pivot(index="lookback", columns="holding", values="ann")
        t = cell.pivot(index="lookback", columns="holding", values="t")
        ax.imshow(ann.to_numpy(), cmap=DIVERGING, norm=norm, aspect="auto")
        for i in range(ann.shape[0]):
            for j in range(ann.shape[1]):
                v = ann.iat[i, j]
                ink = "#ffffff" if abs(v) > 0.6 * vmax else INK
                ax.text(
                    j,
                    i,
                    f"{v:+.1%}\nt={t.iat[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=ink,
                )
        ax.set_xticks(range(ann.shape[1]), [f"{h}m" for h in ann.columns])
        ax.set_yticks(range(ann.shape[0]), [f"{lb}m" for lb in ann.index])
        ax.set_xlabel("holding period")
        ax.set_ylabel("lookback")
        ax.set_title(SCHEME_NAMES[scheme], fontsize=10)
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def spread_strips(spreads: pd.DataFrame, title: str):
    """Every grid cell's annualised spread as a dot, per spread and scheme."""
    from usom.analysis import SPREADS

    names = list(SPREADS)
    schemes = [k for k in SCHEME_NAMES if k in set(spreads["scheme"])]
    fig, ax = plt.subplots(figsize=(9.5, 0.75 * len(names) + 1.4))
    offsets = np.linspace(-0.22, 0.22, len(schemes))
    for off, scheme in zip(offsets, schemes, strict=True):
        s = spreads[spreads["scheme"] == scheme]
        for i, name in enumerate(names):
            v = s.loc[s["spread"] == name, "ann"].to_numpy()
            ax.scatter(
                v,
                np.full(len(v), i + off),
                s=30,
                color=SCHEME_COLORS[scheme],
                edgecolor=SURFACE,
                linewidth=1.5,
                alpha=0.9,
                zorder=3,
                label=SCHEME_NAMES[scheme] if i == 0 else None,
            )
            ax.plot(
                [np.median(v)] * 2,
                [i + off - 0.09, i + off + 0.09],
                color=INK,
                linewidth=2,
                zorder=4,
            )
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _pct_axis(ax, "x")
    ax.set_xlabel("annualised long/short return  (dot = one lookback × holding cell, bar = median)")
    ax.set_title(title, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, fontsize=9, borderaxespad=0.2)
    fig.tight_layout()
    return fig


def is_vs_oos(is_spreads: pd.DataFrame, oos_spreads: pd.DataFrame, spread: str, title: str):
    """Does a grid cell's in-sample spread predict its out-of-sample spread?"""
    keys = ["scheme", "lookback", "holding"]
    a = is_spreads[is_spreads["spread"] == spread].set_index(keys)["ann"]
    b = oos_spreads[oos_spreads["spread"] == spread].set_index(keys)["ann"]
    both = pd.concat([a.rename("is"), b.rename("oos")], axis=1).dropna().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for scheme, g in both.groupby("scheme"):
        ax.scatter(
            g["is"],
            g["oos"],
            s=48,
            color=SCHEME_COLORS[scheme],
            edgecolor=SURFACE,
            linewidth=2,
            label=SCHEME_NAMES[scheme],
            zorder=3,
        )
    lim = max(both[["is", "oos"]].abs().max().max() * 1.1, 0.02)
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, linewidth=1)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _pct_axis(ax, "x")
    _pct_axis(ax, "y")
    ax.set_xlabel("in-sample, annualised")
    ax.set_ylabel("out-of-sample, annualised")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def outcome_histogram(outcomes: pd.DataFrame, title: str):
    """Distribution of each pick's return minus its own group's, laggards vs leaders."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    diffs = {b: g["ret"] - g["group_ret"] for b, g in outcomes.groupby("basket")}
    bins = np.linspace(-0.6, 0.6, 61)
    for basket in ("hot_laggards", "hot_leaders"):
        if basket not in diffs:
            continue
        d = diffs[basket]
        color = BASKET_COLORS[basket]
        ax.hist(
            d,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{BASKET_NAMES[basket]} (median {d.median():+.1%}, "
            f"beat group {np.mean(d > 0):.0%})",
        )
        ax.axvline(d.median(), color=color, linewidth=1)
    ax.axvline(0, color=AXIS, linewidth=1)
    _pct_axis(ax, "x")
    ax.set_yticks([])
    ax.set_xlabel(
        "pick's return minus its sector's equal-weight return over the holding period"
        "  (tails beyond ±60% not drawn)"
    )
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def rolling_chart(r: pd.Series, months: int, title: str):
    """Rolling annualised mean of one spread."""
    from usom.analysis import rolling_annual

    roll = rolling_annual(r, months).dropna()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.plot(roll.index, roll.to_numpy(), color=BASKET_COLORS["hot_laggards"])
    ax.fill_between(roll.index, 0, roll.to_numpy(), color=BASKET_COLORS["hot_laggards"], alpha=0.1)
    _pct_axis(ax)
    ax.set_title(title)
    ax.set_ylabel(f"{months}-month rolling, annualised")
    fig.tight_layout()
    return fig


def case_study_chart(paths: pd.DataFrame, roles: pd.Series, title: str):
    """Price paths indexed to 100 at formation; laggards and leaders highlighted."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    role_color = {"laggard": BASKET_COLORS["hot_laggards"], "leader": BASKET_COLORS["hot_leaders"]}
    ends = {}
    for ticker in paths.columns:
        role = roles.get(ticker, "middle")
        color = role_color.get(role, AXIS)
        ax.plot(
            paths.index,
            paths[ticker].to_numpy(),
            color=color,
            linewidth=2 if role != "middle" else 1.2,
            zorder=3 if role != "middle" else 2,
        )
        if role != "middle":
            ends[ticker] = paths[ticker]
    _end_labels(ax, ends, min_gap=3)
    ax.axhline(100, color=AXIS, linewidth=1)
    ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    handles = [
        mpl.lines.Line2D([], [], color=role_color["laggard"], label="laggard (bought)"),
        mpl.lines.Line2D([], [], color=role_color["leader"], label="leader"),
        mpl.lines.Line2D([], [], color=AXIS, linewidth=1.2, label="rest of the cluster"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_ylabel("price, 100 = formation day")
    ax.set_title(title)
    fig.tight_layout()
    return fig
