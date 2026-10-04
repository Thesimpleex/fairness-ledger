"""Filings, companies and deals.

A merger is often filed more than once (a revised definitive proxy, a supplement
filed under the same form type). Filings are therefore resolved to *deals*:
filings that share a CIK and lie within ``window_days`` of each other form one deal.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from .edgar import Hit


@dataclass(frozen=True)
class Filing:
    """One definitive proxy document of a target company."""

    accession: str
    filename: str
    form: str
    file_date: str
    cik: str
    company: str
    sic: str = ""
    sic_description: str = ""
    tickers: tuple[str, ...] = ()
    current_name: str = ""

    @property
    def year(self) -> int:
        return int(self.file_date[:4])

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["tickers"] = ",".join(self.tickers)
        return row


@dataclass
class Deal:
    """Filings of one transaction, ordered by filing date."""

    deal_id: str
    filings: list[Filing] = field(default_factory=list)

    @property
    def target(self) -> Filing:
        return self.filings[0]

    @property
    def accessions(self) -> list[str]:
        return [f.accession for f in self.filings]


def clean_company_name(display_name: str) -> str:
    """Strip the ``(CIK ...)`` suffix and ticker annotations from an EDGAR display name.

    The display name of a search hit is the company name at the time of the filing.
    """
    name = re.sub(r"\s*\([^)]*\)", "", display_name.split("(CIK")[0])
    return " ".join(name.split()).strip(" ,")


def filing_from_hit(hit: Hit, company: dict[str, Any] | None = None) -> Filing:
    """Build a :class:`Filing`, preferring the submissions record over search metadata."""
    company = company or {}
    cik = hit.ciks[0] if hit.ciks else ""
    current = str(company.get("name") or "")
    name = clean_company_name(hit.names[0]) if hit.names else current
    sic = str(company.get("sic") or (hit.sics[0] if hit.sics else ""))
    return Filing(
        accession=hit.accession,
        filename=hit.filename,
        form=hit.form,
        file_date=hit.file_date,
        cik=cik,
        company=name,
        sic=sic,
        sic_description=str(company.get("sicDescription") or ""),
        tickers=tuple(company.get("tickers") or ()),
        current_name=current,
    )


def group_deals(filings: Iterable[Filing], window_days: int = 270) -> list[Deal]:
    """Group filings of the same company filed within ``window_days`` into deals.

    Parameters
    ----------
    filings
        Filings of any number of companies.
    window_days
        Maximum gap between consecutive filings of one company that still counts
        as the same transaction.

    Returns
    -------
    list of Deal
        Deals ordered by first filing date; ``deal_id`` is the first accession number.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    ordered = sorted(filings, key=lambda f: (f.file_date, f.accession))
    by_cik: dict[str, list[Deal]] = {}
    deals: list[Deal] = []
    for filing in ordered:
        candidates = by_cik.setdefault(filing.cik, [])
        current = candidates[-1] if candidates else None
        if current is not None and _gap_days(current.filings[-1].file_date, filing.file_date) <= (
            window_days
        ):
            current.filings.append(filing)
            continue
        deal = Deal(deal_id=filing.accession, filings=[filing])
        candidates.append(deal)
        deals.append(deal)
    return deals


def _gap_days(earlier: str, later: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def deal_index(deals: Sequence[Deal]) -> dict[str, str]:
    """Map every accession number to its deal id."""
    return {f.accession: d.deal_id for d in deals for f in d.filings}
