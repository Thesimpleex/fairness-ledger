"""Re-verify a released dataset against the cached SEC filings."""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .document import html_to_document
from .edgar import read_cached_filing
from .provenance import verify_span


@dataclass
class VerifyReport:
    """Outcome of :func:`verify_dataset`."""

    checked: int = 0
    verified: int = 0
    missing_filings: int = 0
    hash_mismatch: int = 0
    span_mismatch: int = 0

    @property
    def ok(self) -> bool:
        return self.checked > 0 and self.verified == self.checked


def _verify_filing(args: tuple[str, str, list[dict]]) -> tuple[int, int, int, int, int]:
    raw_dir, accession, rows = args
    folder = Path(raw_dir) / "filings" / accession
    files = sorted(folder.glob("*.gz")) if folder.exists() else []
    if not files:
        return len(rows), 0, len(rows), 0, 0
    text = html_to_document(accession, read_cached_filing(files[0])).text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if any(r["doc_sha256"] != digest for r in rows):
        return len(rows), 0, 0, len(rows), 0
    verified = sum(1 for r in rows if not verify_span(SimpleNamespace(**r), text))
    return len(rows), verified, 0, 0, len(rows) - verified


def verify_dataset(
    frame: pd.DataFrame, raw_dir: str | Path = "data/raw", limit: int | None = None
) -> VerifyReport:
    """Check every fact: text hash, span, snippet and value reproduced from the filing."""
    if limit is not None:
        frame = frame.head(limit)
    cols = [
        "field",
        "low",
        "high",
        "raw_low",
        "raw_high",
        "span_start",
        "span_end",
        "snippet_start",
        "snippet_end",
        "snippet",
        "doc_sha256",
    ]
    jobs = [
        (str(raw_dir), accession, group[cols].to_dict("records"))
        for accession, group in frame.groupby("accession")
    ]
    report = VerifyReport()
    with ProcessPoolExecutor() as pool:
        for checked, verified, missing, mismatch, bad in pool.map(_verify_filing, jobs):
            report.checked += checked
            report.verified += verified
            report.missing_filings += missing
            report.hash_mismatch += mismatch
            report.span_mismatch += bad
    return report
