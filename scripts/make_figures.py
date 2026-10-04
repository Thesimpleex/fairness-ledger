"""Render every exhibit, the banner and the explanatory animation into ``docs/``.

python scripts/make_figures.py

Numbers in titles and callouts are computed from the same objects that
``scripts/run_analyses.py`` writes to ``docs/results.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from animate import save_animation
from banner import render_banner
from run_analyses import compute

from fairness_ledger import figures, load
from fairness_ledger.analyses import NOMINAL_GROWTH_BOUND_PCT
from fairness_ledger.market import load_dgs10
from fairness_ledger.viz import end_label, save_figure, style

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def sources(results: dict) -> dict[str, str]:
    corpus = results["corpus"]
    first, last = corpus["filing_years"]
    base = f"{corpus['n_filings']:,} DEFM14A proxies filed {first} to {last} (SEC EDGAR)"
    ana = results["analyses"]
    dis = ana["bank_disagreement"]["discount_rate"]
    imp = ana["implied_growth"]
    mkt = ana["market_practice"]
    return {
        "disagreement": (
            f"Source: {base}; {dis['n_pairs']} advisor pairs in {dis['n_deals']} deals. "
            "Automatic extraction, see README for precision."
        ),
        "rates": (
            f"Source: {base}; {mkt['n_deals']:,} non-financial deals with a target DCF; "
            "FRED DGS10 on the filing date."
        ),
        "implied": (
            f"Source: {base}; {imp['n_with_disclosed_implied_growth']} of "
            f"{imp['n_exit_multiple_dcfs']:,} exit-multiple DCFs disclose implied growth."
        ),
        "gold": (
            "Source: gold set of 55 filings (25 development, 30 holdout; holdout labels "
            "hash-frozen before scoring)."
        ),
    }


def banner_chart(by_year: pd.DataFrame, p: dict[str, str]):
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    years = by_year["year"].to_numpy()
    ax.fill_between(
        years, by_year["discount_q25"], by_year["discount_q75"], color=p["band"], lw=0, zorder=1
    )
    ax.plot(years, by_year["y10_mean"], color=p["ink_secondary"], lw=1.4, zorder=3)
    ax.plot(years, by_year["discount_median"], color=p["accent"], lw=2.4, zorder=4)
    end_label(ax, years[-1], by_year["discount_median"].iloc[-1], "Discount rate", p["accent"])
    end_label(ax, years[-1], by_year["y10_mean"].iloc[-1], "10-yr yield", p["ink_secondary"])
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_yticklabels(["0", "4", "8", "12", "16%"])
    ax.set_xticks([2012, 2016, 2020, 2025])
    ax.set_xlim(2011.7, 2025.3)
    ax.set_title(
        "Disclosed discount rates by filing year", loc="left", color=p["muted"], fontsize=10
    )
    fig.subplots_adjust(left=0.1, right=0.74, top=0.9, bottom=0.12)
    return fig


def main() -> int:
    release = ROOT / "data" / "release"
    results = json.loads((DOCS / "results.json").read_text())
    facts, filings = load(release), load(release, kind="filings")
    computed = compute(facts, filings, load_dgs10(release / "dgs10.csv"))
    src = sources(results)
    by_year = computed["market"]["_by_year"]
    dis = results["analyses"]["bank_disagreement"]["discount_rate"]
    imp = results["analyses"]["implied_growth"]
    bound = f"{NOMINAL_GROWTH_BOUND_PCT:g}"
    share = imp["bounds"][bound]["high_end_share"]
    results["analyses"]["market_practice"]["pass_through"]["pooled"]

    y = by_year.set_index("year")
    first_year, mid_year = 2021, 2023
    save_figure(
        lambda p: figures.fig_disagreement(
            computed["disagreement"]["discount_rate"]["_pairs"],
            p,
            kicker="Exhibit 1 · Advisors in the same proxy",
            title=(
                f"Two banks valuing the same company differ by a median {dis['median_pp']:.1f} "
                f"percentage points in discount rate, and by 1 point or more in "
                f"{100 * dis['share_ge_1pp']:.0f}% of pairs"
            ),
            source=src["disagreement"],
        ),
        DOCS / "figures" / "exhibit1-disagreement",
    )
    save_figure(
        lambda p: figures.fig_rates_by_year(
            by_year,
            p,
            kicker="Exhibit 2 · Market practice over time",
            title=(
                f"Median disclosed discount rates rose from {y.loc[first_year, 'discount_median']:.1f}% "
                f"in {first_year} to {y.loc[mid_year, 'discount_median']:.1f}% in {mid_year} while the "
                f"ten-year yield rose from {y.loc[first_year, 'y10_mean']:.1f}% to "
                f"{y.loc[mid_year, 'y10_mean']:.1f}%"
            ),
            source=src["rates"],
        ),
        DOCS / "figures" / "exhibit2-market-practice",
    )
    save_figure(
        lambda p: figures.fig_implied_growth(
            computed["implied"]["_units"],
            NOMINAL_GROWTH_BOUND_PCT,
            p,
            kicker="Exhibit 3 · Exit multiples and perpetual growth",
            title=(
                f"{100 * share:.0f}% of exit-multiple DCFs that disclose implied growth reach more "
                f"than {bound}% a year, but only {100 * imp['coverage']:.0f}% of them disclose it"
            ),
            source=src["implied"],
        ),
        DOCS / "figures" / "exhibit3-implied-growth",
    )
    gold = results["gold"]
    key_h = gold["holdout"]["key_fields"]
    save_figure(
        lambda p: figures.fig_gold_scores(
            gold["dev"],
            gold["holdout"],
            p,
            kicker="Exhibit 4 · Extraction quality",
            title=(
                f"On held-out filings, {100 * key_h['precision']:.1f}% of extracted discount rates, "
                "growth rates and exit multiples are correct; fees and per-share values are weaker"
            ),
            source=src["gold"],
        ),
        DOCS / "figures" / "exhibit4-extraction-quality",
    )

    save_figure(lambda p: banner_chart(by_year, p), DOCS / "figures" / "banner-chart")
    precision = key_h["precision"]
    lo, hi = key_h["precision_lo"], key_h["precision_hi"]
    render_banner(
        DOCS / "banner",
        name="fairness-ledger",
        claim=(
            "An open, provenance-linked dataset of the valuation work banks disclose in "
            "U.S. merger proxies."
        ),
        kicker="Open-source Python library · SEC EDGAR · valuation research",
        value=f"{100 * precision:.1f}%",
        label=(
            f"of {key_h['tp'] + key_h['fp']} discount rates, growth rates and exit multiples "
            f"extracted from {gold['holdout']['n_filings']} held-out proxies match the filing "
            f"(95% interval {100 * lo:.1f} to {100 * hi:.1f}%)."
        ),
        chart_light=DOCS / "figures" / "banner-chart-light.png",
        chart_dark=DOCS / "figures" / "banner-chart-dark.png",
        footer_left="github.com/Thesimpleex/fairness-ledger",
        footer_right="Python 3.11+ · MIT licence",
    )

    deals = computed["market"]["_deals"]
    deals = deals[~deals["financial"]].copy()
    deals["year"] = deals["year"].astype(int)
    years = sorted(deals["year"].unique())
    rng = np.random.default_rng(0)
    jitter = {yr: rng.uniform(-0.3, 0.3, (deals["year"] == yr).sum()) for yr in years}
    per_year = {yr: deals.loc[deals["year"] == yr, "discount"].to_numpy() for yr in years}
    steps = 3

    def draw(ax, p, i: int) -> None:
        upto = min(len(years) - 1, i // steps)
        frac = (i % steps + 1) / steps if upto < len(years) - 1 or i // steps < len(years) else 1
        shown = years[: upto + 1]
        for yr in shown:
            alpha = frac if yr == shown[-1] else 1.0
            ax.scatter(
                yr + jitter[yr],
                per_year[yr],
                s=7,
                color=p["context"],
                alpha=0.55 * alpha,
                lw=0,
                zorder=2,
            )
        sub = by_year[by_year["year"].isin(shown)]
        ax.plot(sub["year"], sub["y10_mean"], color=p["ink_secondary"], lw=1.4, zorder=3)
        ax.plot(sub["year"], sub["discount_median"], color=p["accent"], lw=2.4, zorder=4)
        last = sub.iloc[-1]
        end_label(ax, last["year"], last["discount_median"], "Median discount rate", p["accent"])
        end_label(ax, last["year"], last["y10_mean"], "10-year yield", p["ink_secondary"])
        ax.set_xlim(2011.5, 2025.5)
        ax.set_ylim(0, 24)
        ax.set_xticks(years)
        ax.set_xticklabels([str(yr)[2:] for yr in years])
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_yticklabels(["0", "6", "12", "18", "24%"])
        ax.set_title(
            f"Each dot is one deal's target-DCF discount rate; filing years to 20{str(shown[-1])[2:]}",
            loc="left",
            color=p["muted"],
            fontsize=9.5,
        )

    save_animation(
        draw,
        steps * len(years),
        DOCS / "animations" / "discount-rates-by-year",
        style=style,
        figsize=(8.0, 4.5),
        fps=14,
        hold_last=28,
    )
    print("figures, banner and animation written to docs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
