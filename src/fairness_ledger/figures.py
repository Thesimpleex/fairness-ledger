"""Exhibit builders (matplotlib). Each takes computed data and a palette and only draws.

Every builder returns a finished exhibit: kicker, the finding as a sentence, a unit line,
the plot and a source line, in the house style of :mod:`fairness_ledger.viz`.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

mpl.use("Agg")
import matplotlib.pyplot as plt

from .viz import callout, end_label, exhibit, style

FIGSIZE = (8.0, 5.0)


def fig_rates_by_year(
    by_year: pd.DataFrame, p: dict[str, str], *, kicker: str, title: str, source: str
) -> Figure:
    """Deal-median discount rate and terminal growth by filing year next to the 10-year yield."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    years = by_year["year"].to_numpy()
    ax.fill_between(
        years, by_year["discount_q25"], by_year["discount_q75"], color=p["band"], lw=0, zorder=1
    )
    ax.plot(years, by_year["y10_mean"], color=p["ink_secondary"], lw=1.4, zorder=3)
    ax.plot(years, by_year["discount_median"], color=p["accent"], lw=2.4, zorder=4)
    grown = by_year[by_year["growth_n"] >= 3]
    ax.plot(grown["year"], grown["growth_median"], color=p["context_strong"], lw=1.4, zorder=3)
    last = by_year.iloc[-1]
    end_label(ax, last["year"], last["discount_median"], "Discount rate", p["accent"], bold=True)
    end_label(ax, last["year"], last["y10_mean"] - 0.35, "10-year yield", p["ink_secondary"])
    end_label(
        ax,
        grown.iloc[-1]["year"],
        grown.iloc[-1]["growth_median"],
        "Terminal growth",
        p["context_strong"],
    )
    top = int(max(14.0, float(by_year["discount_q75"].max()) + 1.0))
    top += top % 2
    ax.set_ylim(0, top)
    ax.set_xlim(years.min() - 0.3, years.max() + 0.3)
    ax.set_xticks(years)
    ticks = list(range(0, top + 1, 2))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" if t == ticks[-1] else str(t) for t in ticks])
    ax.set_xlabel("Proxy filing year")
    exhibit(
        fig,
        p,
        kicker=kicker,
        title=title,
        subtitle="Deal medians of target-standalone DCF midpoints, % per year; band: interquartile range",
        source=source,
    )
    return fig


def fig_disagreement(
    pairs: pd.DataFrame,
    p: dict[str, str],
    *,
    kicker: str,
    title: str,
    source: str,
    highlight: float = 1.0,
) -> Figure:
    """Histogram of absolute differences between two advisors' discount-rate midpoints."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    d = pairs["abs_diff"].to_numpy()
    top = 5.0
    edges = np.arange(0.0, top + 0.5, 0.5)
    counts, _ = np.histogram(np.clip(d, 0, top - 1e-9), bins=edges)
    left = edges[:-1]
    colours = [p["accent"] if x >= highlight else p["context"] for x in left]
    ax.bar(left, counts, width=0.46, align="edge", color=colours, zorder=3)
    ax.set_xlim(0, top)
    ax.set_xticks(np.arange(0, top + 0.1, 1.0))
    ax.set_xticklabels([f"{t:g}" for t in np.arange(0, top, 1.0)] + [f"{top:g}+"])
    ax.set_xlabel(
        "Absolute difference of the two advisors' discount-rate midpoints, percentage points"
    )
    ax.set_ylim(0, counts.max() * 1.18)
    share = float((d >= highlight).mean())
    peak = int(np.argmax(left >= highlight))
    callout(
        ax,
        p,
        (left[peak] + 0.23, counts[peak]),
        f"{100 * share:.0f}%",
        f"of advisor pairs differ by\n{highlight:g} pp or more (n = {len(d)})",
        dx=40,
        dy=46,
    )
    exhibit(
        fig,
        p,
        kicker=kicker,
        title=title,
        subtitle="Advisor pairs within one proxy, by absolute difference of target-standalone DCF discount rates",
        source=source,
    )
    return fig


def fig_implied_growth(
    units: pd.DataFrame,
    bound: float,
    p: dict[str, str],
    *,
    kicker: str,
    title: str,
    source: str,
) -> Figure:
    """Empirical distribution of the disclosed implied perpetuity growth (high end)."""
    values = np.sort(units.loc[units["covered"], "implied_high"].to_numpy())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    y = np.arange(1, len(values) + 1) / len(values)
    lo, hi = min(-2.0, np.floor(values.min())), min(14.0, np.ceil(values.max()))
    ax.step(
        np.r_[lo, values, hi],
        np.r_[0, y, 1],
        where="post",
        color=p["ink_secondary"],
        lw=1.8,
        zorder=4,
    )
    ax.axvline(bound, color=p["accent"], lw=1.4, zorder=3)
    share = float((values > bound).mean())
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Implied perpetuity growth, high end of the disclosed range, % per year")
    at = float((values <= bound).mean())
    callout(
        ax,
        p,
        (bound, at),
        f"{100 * share:.0f}%",
        f"of the {len(values)} analyses imply growth\nabove the {bound:g}% bound",
        dx=-46,
        dy=64,
        ha="right",
    )
    exhibit(
        fig,
        p,
        kicker=kicker,
        title=title,
        subtitle="Cumulative share of exit-multiple DCFs that disclose the growth rate their multiple implies, %",
        source=source,
    )
    return fig


GOLD_FIELDS = (
    "discount_rate",
    "terminal_growth",
    "exit_multiple",
    "implied_growth",
    "value_per_share",
    "offer_price",
    "fee_total",
)


def fig_gold_scores(
    dev: dict[str, Any],
    holdout: dict[str, Any],
    p: dict[str, str],
    *,
    kicker: str,
    title: str,
    source: str,
) -> Figure:
    """Precision and recall by field on the dev and holdout filings, with Wilson intervals."""
    fig, (a, b) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    y = np.arange(len(GOLD_FIELDS))[::-1].astype(float)
    for ax, metric, name in ((a, "precision", "Precision, %"), (b, "recall", "Recall, %")):
        for offset, res, colour, label in (
            (0.17, dev, p["context_strong"], "Development filings"),
            (-0.17, holdout, p["accent"], "Holdout filings"),
        ):
            for yi, field in zip(y, GOLD_FIELDS, strict=True):
                s = res["by_field"].get(field)
                if s is None or s[metric] != s[metric]:
                    continue
                v, lo, hi = (100 * s[metric], 100 * s[f"{metric}_lo"], 100 * s[f"{metric}_hi"])
                ax.plot([lo, hi], [yi + offset] * 2, color=colour, lw=1.3, zorder=3)
                ax.plot(
                    v,
                    yi + offset,
                    "o",
                    color=colour,
                    ms=5.5,
                    mec=p["surface"],
                    zorder=4,
                    label=label,
                )
        ax.set_xlim(20, 100)
        ax.set_ylim(-0.7, len(GOLD_FIELDS) - 0.3)
        ax.set_xlabel(name)
        ax.grid(axis="x", color=p["grid"])
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    a.set_yticks(y)
    a.set_yticklabels([f.replace("_", " ") for f in GOLD_FIELDS], color=p["ink_secondary"])
    a.axvline(95, color=p["context"], lw=0.9, ls=(0, (1, 2)), zorder=2)
    handles, labels = a.get_legend_handles_labels()
    keep = dict(zip(labels, handles, strict=False))
    a.legend(keep.values(), keep.keys(), loc="lower left", fontsize=8.5, borderaxespad=0.2)
    exhibit(
        fig,
        p,
        kicker=kicker,
        title=title,
        subtitle="Share of extracted (precision) and annotated (recall) facts that match; 95% Wilson intervals; dotted: 95% precision target",
        source=source,
        right_in=0.5,
    )
    fig.subplots_adjust(left=1.45 / FIGSIZE[0], wspace=0.08)
    return fig


def render_png(
    build: Callable[[dict[str, str]], Figure], theme: str = "light", dpi: int = 160
) -> bytes:
    """Render ``build(palette)`` to PNG bytes in one theme (used by the HTML report)."""
    buffer = io.BytesIO()
    with style(theme) as p:
        fig = build(p)
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
    return buffer.getvalue()
