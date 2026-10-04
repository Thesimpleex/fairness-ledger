"""Build, write and load the released dataset of extracted facts.

One row per admitted fact. Every row carries its provenance: accession number, character
span in the normalised text, a verbatim snippet, and the SHA-256 of that text so a reader
can re-derive the span from the SEC filing (``fairness-ledger verify``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .consistency import add_flags
from .deals import Deal, Filing
from .document import html_to_document
from .edgar import read_cached_filing
from .extract import Extraction, Fact, extract_document

CORE_FIELDS = (
    "discount_rate",
    "terminal_growth",
    "exit_multiple",
    "implied_growth",
    "implied_multiple",
)

COLUMNS: dict[str, str] = {
    "fact_id": "Stable identifier: accession number, field and span start.",
    "deal_id": "Accession number of the first filing of the transaction (amendments share it).",
    "accession": "SEC accession number of the filing the value was read from.",
    "cik": "CIK of the filing company (the target in most proxies).",
    "company": "Company name at the time of the filing.",
    "sic": "SIC code from the SEC submissions record.",
    "filing_date": "Filing date (ISO).",
    "year": "Filing year.",
    "field": (
        "discount_rate, terminal_growth, exit_multiple, implied_growth (perpetuity growth implied "
        "by an exit multiple), implied_multiple (terminal multiple implied by a perpetuity "
        "growth rate), value_per_share (implied value per share of a DCF-type analysis), "
        "offer_price (cash per share), fee_total, fee_opinion, fee_contingent."
    ),
    "low": "Lower end of the stated range (equal to ``high`` for a point value).",
    "high": "Upper end of the stated range.",
    "unit": "pct (percent), x (multiple), usd (dollars per share) or usd_million.",
    "tier": (
        "core (discount rate, terminal growth, exit multiple and their implied counterparts: "
        "precision-validated) or auxiliary (per-share values, fees, offer price: lower measured "
        "precision, see README)."
    ),
    "is_point": "True if the filing states a single value.",
    "advisor": "Financial advisor performing the analysis (lexicon) or 'unattributed'.",
    "subject": "target, acquirer, pro_forma or unspecified (relative to the transaction).",
    "analysis": "dcf, ddm, pv_future_price, nav, other (valuation parameters only).",
    "region": "Section of the proxy: opinion, background or context.",
    "metric": "Metric of an exit multiple: EBITDA, P/E, book value, FCF, ... (if stated).",
    "valuation_date": "Discounting date stated in the same sentence (ISO), if any.",
    "flags": "Semicolon-separated soft consistency flags (see docs/methodology.md).",
    "n_filings": "Number of filings of the deal that contain the same statement.",
    "span_start": "Start of the value phrase in the normalised filing text.",
    "span_end": "End (exclusive) of the value phrase.",
    "snippet_start": "Start of the verbatim snippet in the normalised text.",
    "snippet_end": "End (exclusive) of the snippet window.",
    "snippet": "Verbatim sentence-level snippet (whitespace collapsed, at most 300 characters).",
    "raw_low": "Lower endpoint as written in the filing.",
    "raw_high": "Upper endpoint as written in the filing.",
    "doc_sha256": "SHA-256 of the normalised text the span refers to.",
}


def _process(args: tuple[Filing, str, str]) -> tuple[str, str, list[Fact], dict[str, int]]:
    filing, raw_dir, deal_id = args
    raw = read_cached_filing(Path(raw_dir) / "filings" / filing.accession / f"{filing.filename}.gz")
    doc = html_to_document(filing.accession, raw)
    result: Extraction = extract_document(doc, filing.company)
    reasons: dict[str, int] = {}
    for r in result.rejections:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    return filing.accession, deal_id, result.facts, reasons


def build_dataset(
    deals: Iterable[Deal],
    raw_dir: str | Path = "data/raw",
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    only: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract facts from every cached filing of ``deals`` and de-duplicate within deals.

    Returns the fact table and the total rejection counts by reason.
    """
    deals = list(deals)
    jobs = [
        (f, str(raw_dir), d.deal_id)
        for d in deals
        for f in d.filings
        if only is None or f.accession in only
    ]
    meta = {f.accession: f for d in deals for f in d.filings}
    rows: list[dict[str, object]] = []
    totals: dict[str, int] = {}
    with ProcessPoolExecutor(workers) as pool:
        for i, (accession, deal_id, facts, reasons) in enumerate(
            pool.map(_process, jobs, chunksize=4), 1
        ):
            f = meta[accession]
            for fact in facts:
                row = fact.to_row()
                row.update(
                    deal_id=deal_id,
                    cik=f.cik,
                    company=f.company,
                    sic=f.sic,
                    filing_date=f.file_date,
                    year=f.year,
                )
                rows.append(row)
            for k, v in reasons.items():
                totals[k] = totals.get(k, 0) + v
            if progress:
                progress(i, len(jobs))
    return finalize(pd.DataFrame(rows)), totals


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Add identifiers and flags, drop duplicates across filings of one deal."""
    if frame.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    frame = frame.copy()
    frame["is_point"] = frame["low"] == frame["high"]
    frame["tier"] = frame["field"].map(lambda f: "core" if f in CORE_FIELDS else "auxiliary")
    key = [
        "deal_id",
        "field",
        "low",
        "high",
        "advisor",
        "analysis",
        "subject",
        "metric",
        "valuation_date",
        "snippet",
    ]
    frame = frame.sort_values(["filing_date", "accession", "span_start"], kind="mergesort")
    frame["n_filings"] = frame.groupby(key)["accession"].transform("nunique")
    frame = frame.drop_duplicates(key, keep="first").reset_index(drop=True)
    frame["fact_id"] = (
        frame["accession"] + ":" + frame["field"] + ":" + frame["span_start"].astype(str)
    )
    frame = add_flags(frame)
    return frame[list(COLUMNS)]


def write_dataset(frame: pd.DataFrame, directory: str | Path = "data/release") -> list[Path]:
    """Write ``facts.parquet`` and ``facts.csv`` into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parquet, csv = directory / "facts.parquet", directory / "facts.csv"
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return [parquet, csv]


def default_data_dir() -> Path:
    """``$FAIRNESS_LEDGER_DATA`` or ``data/release`` in the working directory or repository."""
    env = os.environ.get("FAIRNESS_LEDGER_DATA")
    if env:
        return Path(env)
    here = Path("data/release")
    if here.exists():
        return here
    return Path(__file__).resolve().parents[2] / "data" / "release"


def load(path: str | Path | None = None, kind: str = "facts") -> pd.DataFrame:
    """Load the released dataset.

    Parameters
    ----------
    path
        Directory or file; defaults to :func:`default_data_dir`.
    kind
        ``"facts"`` (extracted values) or ``"filings"`` (the filing catalog).
    """
    if kind not in ("facts", "filings"):
        raise ValueError("kind must be 'facts' or 'filings'")
    base = Path(path) if path is not None else default_data_dir()
    if base.is_file():
        return pd.read_parquet(base) if base.suffix == ".parquet" else pd.read_csv(base)
    parquet = base / f"{kind}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = base / ("filings.csv" if kind == "filings" else "facts.csv")
    if csv.exists():
        return pd.read_csv(csv, dtype={"cik": str, "sic": str})
    raise FileNotFoundError(f"no {kind} dataset found in {base}")
