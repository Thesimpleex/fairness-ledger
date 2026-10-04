"""Rich terminal view of one deal: every advisor's analyses side by side, with sources."""

from __future__ import annotations

import re

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

ROWS = (
    ("discount_rate", "Discount rate"),
    ("terminal_growth", "Terminal growth"),
    ("exit_multiple", "Exit multiple"),
    ("implied_growth", "Implied perp. growth"),
    ("implied_multiple", "Implied multiple"),
    ("value_per_share", "Value per share"),
)
SUBJECTS = {
    "target": "Target standalone",
    "acquirer": "Acquirer standalone",
    "pro_forma": "Pro forma combined",
    "unspecified": "Subject not stated",
}
ANALYSES = {
    "dcf": "discounted cash flow",
    "ddm": "dividend discount",
    "pv_future_price": "present value of future price",
    "nav": "net asset value",
    "other": "other",
}
SUBJECT_ORDER = ["target", "acquirer", "pro_forma", "unspecified"]
ANALYSIS_ORDER = ["dcf", "ddm", "pv_future_price", "nav", "other", "unspecified", ""]


def find_deal(filings: pd.DataFrame, query: str, facts: pd.DataFrame | None = None) -> str | None:
    """Resolve an accession number, ticker or company-name fragment to a ``deal_id``.

    Ambiguous matches resolve to the deal with the most extracted facts (or the most recent
    filing if no fact table is given).
    """
    q = query.strip()
    exact = filings[(filings["accession"] == q) | (filings["deal_id"] == q)]
    if not exact.empty:
        return str(exact["deal_id"].iloc[0])
    tickers = filings["tickers"].fillna("").astype(str)
    by_ticker = filings[tickers.str.upper().str.split(",").apply(lambda t: q.upper() in t)]
    names = (
        filings["company"].fillna("").astype(str)
        + " "
        + filings["current_name"].fillna("").astype(str)
    )
    by_name = filings[names.str.contains(re.escape(q), case=False, regex=True)]
    candidates = by_ticker if not by_ticker.empty else by_name
    if candidates.empty:
        return None
    ids = candidates["deal_id"].unique()
    if facts is not None:
        counts = facts[facts["deal_id"].isin(ids)].groupby("deal_id").size()
        if not counts.empty:
            return str(counts.idxmax())
    latest = candidates.sort_values("file_date").iloc[-1]
    return str(latest["deal_id"])


def fmt_range(low: float, high: float, unit: str) -> str:
    """Compact range such as ``7.5–9.5%``, ``9.5–11.5x`` or ``$21.00–$31.00``."""
    if unit == "usd":
        a, b = f"${low:,.2f}", f"${high:,.2f}"
        return a if low == high else f"{a}–{b}"
    suffix = {"pct": "%", "x": "x"}.get(unit, "")
    a, b = f"{low:g}", f"{high:g}"
    return f"{a}{suffix}" if low == high else f"{a}–{b}{suffix}"


def _cell(group: pd.DataFrame) -> str:
    if group.empty:
        return "—"
    group = group.sort_values("span_start")
    first = group.iloc[0]
    text = fmt_range(first["low"], first["high"], first["unit"])
    extra = len(group) - 1
    return text if extra == 0 else f"{text} +{extra}"


def _snippet(text: str, raw: str, width: int) -> Text:
    """Trim ``text`` around the value ``raw`` to ``width`` characters, value in bold."""
    k = text.find(raw)
    if k < 0:
        k = 0
    start = max(0, k - width // 2)
    end = min(len(text), start + width)
    start = max(0, end - width)
    piece = text[start:end]
    out = Text(("…" if start > 0 else "") + piece + ("…" if end < len(text) else ""), style="dim")
    needle = out.plain.find(raw)
    if needle >= 0:
        out.stylize("bold", needle, needle + len(raw))
    return out


def _money(value: float) -> str:
    return f"${value:,.1f}m"


def render_deal(
    console: Console,
    facts: pd.DataFrame,
    filings: pd.DataFrame,
    deal_id: str,
    width: int = 64,
) -> None:
    """Print the deal header, advisor fees, and one table per subject and analysis type."""
    deal_filings = filings[filings["deal_id"] == deal_id].sort_values("file_date")
    head = deal_filings.iloc[0]
    sub = facts[facts["deal_id"] == deal_id]
    title = Text()
    title.append(
        str(head["company"]).title() if str(head["company"]).isupper() else str(head["company"]),
        "bold",
    )
    console.print(title)
    advisors = sorted(set(sub["advisor"]) - {"unattributed", "joint", ""})
    console.print(
        Text(
            f"{head['form']} {head['accession']} · filed {head['file_date']} · {len(sub)} facts · "
            f"advisors: {', '.join(advisors) or 'none identified'}",
            style="dim",
        )
    )
    offer = sub[sub["field"] == "offer_price"]
    if not offer.empty:
        console.print(f"Cash offer  ${offer['low'].iloc[0]:,.2f} per share")
    fees = sub[sub["field"].isin(["fee_total", "fee_opinion", "fee_contingent"])]
    if not fees.empty:
        parts = []
        for advisor, group in fees.groupby("advisor"):
            bits = []
            for field, label in (
                ("fee_total", "total"),
                ("fee_opinion", "opinion"),
                ("fee_contingent", "contingent"),
            ):
                row = group[group["field"] == field]
                if not row.empty:
                    bits.append(f"{_money(row['low'].iloc[0])} {label}")
            parts.append(f"{advisor}: {', '.join(bits)}")
        console.print(Text("Fees  " + "  |  ".join(parts), style="dim"))
    console.print()
    wanted = sub[sub["field"].isin([f for f, _ in ROWS])]
    for subject in SUBJECT_ORDER:
        for analysis in ANALYSIS_ORDER:
            block = wanted[(wanted["subject"] == subject) & (wanted["analysis"] == analysis)]
            if block.empty:
                continue
            if (
                analysis in ("other", "unspecified", "")
                and not (block["field"] == "discount_rate").any()
            ):
                continue
            names = sorted(block["advisor"].unique(), key=lambda a: (a == "unattributed", a))
            table = Table(
                title=f"{SUBJECTS[subject]} · {ANALYSES.get(analysis, analysis or 'unspecified')}",
                title_justify="left",
                title_style="bold",
                box=box.SIMPLE_HEAD,
                header_style="bold",
                pad_edge=False,
            )
            table.add_column("")
            for name in names:
                table.add_column(name, justify="right")
            for field, label in ROWS:
                if not (block["field"] == field).any():
                    continue
                table.add_row(
                    label,
                    *[
                        _cell(block[(block["field"] == field) & (block["advisor"] == n)])
                        for n in names
                    ],
                )
            console.print(table)
            for name in names:
                rate = block[(block["field"] == "discount_rate") & (block["advisor"] == name)]
                pick = rate if not rate.empty else block[block["advisor"] == name]
                row = pick.sort_values("span_start").iloc[0]
                line = Text(f"  {name}: ", style="dim")
                line.append_text(_snippet(str(row["snippet"]), str(row["raw_low"]), width))
                console.print(line)
            console.print()
