"""Self-contained HTML market-practice report (inline CSS, figures embedded as PNG)."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import pandas as pd

from .analyses import (
    FINANCIAL_SIC,
    NOMINAL_GROWTH_BOUND_PCT,
    UNNAMED,
    bank_disagreement,
    implied_growth_check,
    market_practice,
    midpoints,
)
from .figures import fig_disagreement, fig_implied_growth, fig_rates_by_year, render_png
from .market import load_dgs10

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--rule:#e1e0d9;--surface:#fcfcfb;--blue:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font:15px/1.55 "Helvetica Neue",Helvetica,Arial,sans-serif}
.page{max-width:900px;margin:0 auto;padding:56px 40px 72px;background:var(--surface)}
.kicker{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);font-weight:600}
h1{font-size:30px;line-height:1.2;margin:10px 0 8px;font-weight:700;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:16px;margin:0 0 28px;max-width:680px}
h2{font-size:18px;margin:44px 0 10px;padding-top:18px;border-top:1px solid var(--rule)}
.kpis{display:flex;gap:0;margin:8px 0 6px;border-top:1px solid var(--ink);border-bottom:1px solid var(--rule)}
.kpi{flex:1;padding:16px 18px 14px 0}
.kpi+.kpi{padding-left:18px;border-left:1px solid var(--rule)}
.kpi b{display:block;font-size:30px;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.kpi span{color:var(--ink2);font-size:12.5px;display:block;margin-top:4px}
figure{margin:18px 0 6px}
figure img{width:100%;height:auto;display:block}
figcaption{color:var(--ink2);font-size:12.5px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th{text-align:right;font-weight:600;border-bottom:1px solid var(--ink);padding:6px 8px}
td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
.note{color:var(--ink2);font-size:12.5px}
.foot{margin-top:48px;padding-top:14px;border-top:1px solid var(--rule);color:var(--muted);font-size:11.5px}
"""


def _png(build: Any) -> str:
    return base64.b64encode(render_png(build, "light")).decode("ascii")


def _fmt(x: float, nd: int = 1) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def advisor_table(facts: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Per-advisor summary of target standalone DCFs."""
    sub = facts[
        (facts["field"] == "discount_rate")
        & (facts["subject"] == "target")
        & (facts["analysis"] == "dcf")
        & ~facts["advisor"].isin(UNNAMED)
        & ~facts["sic"].astype(str).str[:2].isin(FINANCIAL_SIC)
    ].copy()
    sub["mid"] = midpoints(sub)
    sub["width"] = sub["high"] - sub["low"]
    growth = facts[
        (facts["field"] == "terminal_growth")
        & (facts["subject"] == "target")
        & (facts["analysis"] == "dcf")
        & ~facts["sic"].astype(str).str[:2].isin(FINANCIAL_SIC)
    ].copy()
    growth["mid"] = midpoints(growth)
    out = (
        sub.groupby("advisor")
        .agg(
            deals=("deal_id", "nunique"), discount_mid=("mid", "median"), width=("width", "median")
        )
        .join(growth.groupby("advisor")["mid"].median().rename("growth_mid"))
        .sort_values("deals", ascending=False)
        .head(top)
        .reset_index()
    )
    return out


def write_report(
    facts: pd.DataFrame,
    filings: pd.DataFrame,
    out: str | Path = "docs/report.html",
    dgs10_path: str | Path | None = None,
    seed: int = 0,
) -> Path:
    """Write the report and return its path."""
    dgs10_path = (
        Path(dgs10_path) if dgs10_path else Path(out).parent.parent / "data/release/dgs10.csv"
    )
    yields = load_dgs10(dgs10_path)
    market = market_practice(facts, yields, draws=1000, seed=seed)
    dis = bank_disagreement(facts, draws=500, seed=seed)
    imp = implied_growth_check(facts)
    by_year = market["_by_year"]
    pt = market["pass_through"]["pooled"]
    d_disc = dis["discount_rate"]
    figures: list[tuple[str, str, str]] = []
    figures.append(
        (
            _png(
                lambda p: fig_rates_by_year(
                    by_year,
                    f"Disclosed discount rates track Treasury yields with a slope of {pt['slope_pp_per_pp']:.2f}",
                    p,
                )
            ),
            "Discount rates and terminal growth by filing year",
            f"Pooled slope of deal-level discount-rate midpoints on the ten-year yield at filing: "
            f"{pt['slope_pp_per_pp']:.2f} (95% year-block bootstrap interval "
            f"{pt['ci_95_year_block_bootstrap'][0]:.2f} to {pt['ci_95_year_block_bootstrap'][1]:.2f}). "
            "Descriptive; not a causal estimate.",
        )
    )
    if d_disc.get("n_pairs"):
        figures.append(
            (
                _png(
                    lambda p: fig_disagreement(
                        d_disc["_pairs"],
                        dis["terminal_growth"].get("_pairs", d_disc["_pairs"].iloc[0:0]),
                        f"Two banks valuing one target differ by a median {d_disc['median_pp']:.1f} pp in discount rate",
                        p,
                    )
                ),
                "Disagreement between advisors in the same proxy",
                "Absolute difference of the advisors' target-standalone DCF midpoints, one observation "
                "per advisor pair within a deal.",
            )
        )
    if imp["n_with_disclosed_implied_growth"]:
        b = imp["bounds"][f"{NOMINAL_GROWTH_BOUND_PCT:g}"]
        figures.append(
            (
                _png(
                    lambda p: fig_implied_growth(
                        imp["_units"],
                        NOMINAL_GROWTH_BOUND_PCT,
                        f"{100 * b['high_end_share']:.0f}% of disclosed implied growth rates exceed a "
                        f"{NOMINAL_GROWTH_BOUND_PCT:g}% nominal bound",
                        p,
                    )
                ),
                "Implied perpetuity growth of exit-multiple DCFs",
                f"Coverage: {imp['n_with_disclosed_implied_growth']} of {imp['n_exit_multiple_dcfs']} "
                "exit-multiple DCFs disclose the growth rate their multiple implies.",
            )
        )
    kpis = [
        (f"{facts['deal_id'].nunique():,}", "deals with extracted valuation facts"),
        (f"{len(facts):,}", "facts, each linked to its sentence"),
        (f"{market['n_deals']:,}", "non-financial deals with a target DCF"),
        (f"{pt['slope_pp_per_pp']:.2f}", "discount-rate pass-through from the 10-year yield"),
    ]
    rows = "".join(
        f"<tr><td>{int(r.year)}</td><td>{int(r.n)}</td><td>{_fmt(r.discount_median)}</td>"
        f"<td>{_fmt(r.discount_q25)} to {_fmt(r.discount_q75)}</td><td>{_fmt(r.growth_median)}</td>"
        f"<td>{_fmt(r.y10_mean, 2)}</td></tr>"
        for r in by_year.itertuples()
    )
    adv = advisor_table(facts)
    adv_rows = "".join(
        f"<tr><td>{html.escape(r.advisor)}</td><td>{int(r.deals)}</td><td>{_fmt(r.discount_mid)}</td>"
        f"<td>{_fmt(r.width)}</td><td>{_fmt(r.growth_mid)}</td></tr>"
        for r in adv.itertuples()
    )
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Fairness opinion valuation practice</title>",
        f"<style>{CSS}</style></head><body><div class='page'>",
        "<div class='kicker'>fairness-ledger · market practice</div>",
        "<h1>What investment banks disclose about the value of the companies they opine on</h1>",
        "<p class='sub'>Discount rates, terminal growth, exit multiples and fees extracted from "
        f"{filings['accession'].nunique():,} U.S. merger proxies ({int(filings['year'].min())} to "
        f"{int(filings['year'].max())}). Every number links to its sentence in the filing.</p>",
        "<div class='kpis'>"
        + "".join(
            f"<div class='kpi'><b>{a}</b><span>{html.escape(b)}</span></div>" for a, b in kpis
        )
        + "</div>",
    ]
    for image, heading, caption in figures:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(
            f"<figure><img alt='{html.escape(heading)}' src='data:image/png;base64,{image}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    parts += [
        "<h2>By filing year</h2>",
        "<table><tr><th>Year</th><th>Deals</th><th>Discount rate, median %</th>"
        "<th>Interquartile range %</th><th>Terminal growth, median %</th><th>10-year yield, mean %</th></tr>"
        f"{rows}</table>",
        "<p class='note'>Deal-level medians of target-standalone DCF midpoints; non-financial "
        "targets only (banks and insurers are valued with dividend models).</p>",
        "<h2>By advisor</h2>",
        "<table><tr><th>Advisor</th><th>Deals</th><th>Discount rate, median %</th>"
        "<th>Median range width, pp</th><th>Terminal growth, median %</th></tr>"
        f"{adv_rows}</table>",
        "<p class='note'>Advisors named by a lexicon of house names; analyses of target "
        "standalone DCFs only.</p>",
        "<div class='foot'>Descriptive research on public filings, not investment advice. "
        "Extraction is automatic: see the precision and recall figures in the repository "
        "README before relying on any single value, and follow its provenance link to the "
        "filing text.</div>",
        "</div></body></html>",
    ]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out
