"""House style for figures: consulting-exhibit layout in black, grey and one red.

Modelled on the exhibit format of the large strategy consultancies and on The
Economist's charts: the title states the finding, grey carries context, a single red
accent carries the message, labels sit on the data, and source and date sit under a
hairline. Every figure is rendered on a light and a dark surface so the README can
serve the variant that matches the reader's GitHub theme.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#FFFFFF",
        "ink": "#111111",
        "ink_secondary": "#3D3D3D",
        "muted": "#6B6B6B",
        "grid": "#E8E8E8",
        "rule": "#D9D9D9",
        "context": "#BDBDBD",
        "context_strong": "#8C8C8C",
        "accent": "#C8102E",
        "accent_soft": "#F3D0D6",
        "band": "#F2F2F2",
    },
    "dark": {
        "surface": "#0F0F0F",
        "ink": "#F5F5F5",
        "ink_secondary": "#CFCFCF",
        "muted": "#9E9E9E",
        "grid": "#262626",
        "rule": "#333333",
        "context": "#555555",
        "context_strong": "#808080",
        "accent": "#FF4D5E",
        "accent_soft": "#4D1A21",
        "band": "#1C1C1C",
    },
}

for _p in THEMES.values():
    # Aliases kept for scripts written against the earlier palette.
    _p["axis"] = _p["rule"]
    _p["neutral"] = _p["band"]
    _p["series"] = (_p["accent"], _p["ink_secondary"], _p["context_strong"], _p["context"])  # type: ignore[assignment]

FONT = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]

# Layout in inches, independent of figure size.
_LEFT, _RIGHT = 0.30, 0.30
_KICKER_H, _TITLE_LINE_H, _SUBTITLE_H, _GAP = 0.26, 0.30, 0.26, 0.16
_SOURCE_H = 0.42


def sequential_cmap(theme: str = "light") -> LinearSegmentedColormap:
    """Single-hue magnitude map from the surface to the accent red."""
    p = THEMES[theme]
    stops = (
        p["surface"],
        p["accent_soft"],
        p["accent"],
        "#6E0918" if theme == "light" else "#FFB3BB",
    )
    return LinearSegmentedColormap.from_list(f"seq-{theme}", stops)


def diverging_cmap(theme: str = "light") -> LinearSegmentedColormap:
    """Charcoal (negative) to red (positive) around a neutral grey midpoint."""
    p = THEMES[theme]
    low = "#3D3D3D" if theme == "light" else "#BDBDBD"
    return LinearSegmentedColormap.from_list(f"div-{theme}", (low, p["band"], p["accent"]))


@contextmanager
def style(theme: str = "light") -> Iterator[dict[str, str]]:
    """Apply the house style for one theme and yield its palette."""
    p = THEMES[theme]
    rc = {
        "figure.facecolor": p["surface"],
        "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"],
        "text.color": p["ink"],
        "axes.labelcolor": p["ink_secondary"],
        "axes.edgecolor": p["ink"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": p["grid"],
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": p["ink"],
        "ytick.color": p["grid"],
        "xtick.labelcolor": p["muted"],
        "ytick.labelcolor": p["muted"],
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "xtick.minor.width": 0,
        "ytick.minor.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 9.5,
        "axes.labelpad": 8,
        "axes.prop_cycle": mpl.cycler(color=[p["accent"], p["context_strong"], p["context"]]),
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "patch.edgecolor": p["surface"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": p["ink_secondary"],
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": 10,
        "font.weight": "regular",
        "savefig.dpi": 220,
        "savefig.bbox": "standard",
    }
    with mpl.rc_context(rc):
        yield p


def exhibit(
    fig: Figure,
    p: dict[str, str],
    title: str,
    subtitle: str | None = None,
    source: str | None = None,
    kicker: str | None = None,
    wrap: int = 72,
    right_in: float = 1.05,
) -> None:
    """Lay out an exhibit: kicker, finding as title, unit line, plot area, source.

    Call after creating the axes; this sets the subplot margins from fixed inch
    measurements so that every figure in a repository shares the same rhythm.
    """
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(title, wrap) or [""]
    top_in = 0.28 + (_KICKER_H if kicker else 0) + _TITLE_LINE_H * len(lines)
    top_in += (_SUBTITLE_H if subtitle else 0) + _GAP + 0.10
    has_xlabel = any(ax.get_xlabel() for ax in fig.axes)
    bottom_in = (_SOURCE_H if source else 0.18) + 0.30 + 0.30 + (0.30 if has_xlabel else 0.0)
    fig.subplots_adjust(
        left=0.62 / w + _LEFT / w, right=1 - right_in / w, top=1 - top_in / h, bottom=bottom_in / h
    )
    x = _LEFT / w
    y = 1 - 0.28 / h
    if kicker:
        fig.text(
            x, y, kicker.upper(), color=p["muted"], fontsize=8.5, fontweight="medium", va="top"
        )
        y -= _KICKER_H / h
    for line in lines:
        fig.text(x, y, line, color=p["ink"], fontsize=15.5, fontweight="regular", va="top")
        y -= _TITLE_LINE_H / h
    if subtitle:
        fig.text(x, y - 0.02 / h, subtitle, color=p["ink_secondary"], fontsize=10, va="top")
    if source:
        ys = 0.36 / h
        fig.add_artist(
            Line2D(
                [x, 1 - _RIGHT / w],
                [ys + 0.1 / h] * 2,
                transform=fig.transFigure,
                color=p["rule"],
                linewidth=0.7,
            )
        )
        fig.text(x, ys, source, color=p["muted"], fontsize=8, va="top")


def end_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    bold: bool = False,
    dx: float = 6,
    dy: float = 0,
) -> None:
    """Label a series at its last point instead of using a legend."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        color=color,
        fontsize=9,
        fontweight="medium" if bold else "regular",
        annotation_clip=False,
    )


def callout(
    ax: Axes,
    p: dict[str, str],
    xy: tuple[float, float],
    value: str,
    note: str,
    *,
    dx: float = 34,
    dy: float = 0,
    ha: str = "left",
) -> None:
    """Large red figure with a short note, tied to a data point by a hairline."""
    sign = 1 if ha == "left" else -1
    ax.plot(*xy, "o", ms=6.5, color=p["accent"], mec=p["surface"], mew=1.6, zorder=6, clip_on=False)
    ax.annotate(
        "",
        xy=xy,
        xytext=(sign * dx, dy),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": p["accent"], "lw": 0.9, "shrinkA": 0, "shrinkB": 5},
        annotation_clip=False,
    )
    halo = {"boxstyle": "square,pad=0.12", "fc": p["surface"], "ec": "none"}
    ax.annotate(
        value,
        xy,
        xytext=(sign * (dx + 4), dy + 2),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        color=p["accent"],
        fontsize=19,
        fontweight="light",
        annotation_clip=False,
        bbox=halo,
        zorder=7,
    )
    ax.annotate(
        note,
        xy,
        xytext=(sign * (dx + 4), dy - 2),
        textcoords="offset points",
        ha=ha,
        va="top",
        color=p["ink_secondary"],
        fontsize=8.5,
        annotation_clip=False,
        linespacing=1.25,
        bbox=halo,
        zorder=7,
    )


def save_figure(
    build: Callable[[dict[str, str]], Figure],
    stem: str | Path,
    themes: Sequence[str] = ("light", "dark"),
) -> list[Path]:
    """Render ``build(palette)`` once per theme and save ``<stem>-<theme>.png``.

    ``build`` must only draw: compute all data beforehand so that both variants
    show identical content.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in themes:
        with style(theme) as p:
            fig = build(p)
            path = stem.with_name(f"{stem.name}-{theme}.png")
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def picture_html(stem: str, alt: str, width: int | None = None) -> str:
    """README snippet that serves the dark variant to dark-theme readers."""
    size = f' width="{width}"' if width else ""
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{stem}-dark.png">\n'
        f'  <img alt="{alt}" src="{stem}-light.png"{size}>\n'
        "</picture>"
    )
