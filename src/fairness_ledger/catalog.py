"""Build and read the filing catalog: discovery, company metadata, deal grouping."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .deals import Deal, Filing, filing_from_hit, group_deals
from .edgar import DEFAULT_QUERIES, Hit, SecClient

CATALOG_FIELDS = [
    "deal_id",
    "accession",
    "filename",
    "form",
    "file_date",
    "cik",
    "company",
    "sic",
    "sic_description",
    "tickers",
    "current_name",
]


def build_catalog(
    client: SecClient,
    years: Iterable[int],
    queries: Sequence[str] = DEFAULT_QUERIES,
    download: bool = True,
    workers: int = 4,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[Deal]:
    """Discover filings, attach company metadata, group into deals, optionally download.

    Parameters
    ----------
    client
        Cached SEC client.
    years
        Filing years to search.
    queries
        Full-text-search phrases; a filing is kept if any of them matches.
    download
        Fetch every discovered document into the cache.
    workers
        Download threads (the shared rate limiter still bounds requests per second).
    progress
        Optional callback ``(stage, done, total)``.
    """
    say = progress or (lambda stage, done, total: None)
    hits = client.discover(years, queries=queries, progress=lambda y, n: say(f"search {y}", n, n))
    ciks = sorted({h.ciks[0] for h in hits if h.ciks})
    companies: dict[str, dict[str, Any]] = {}

    def company(cik: str) -> tuple[str, dict[str, Any]]:
        try:
            return cik, client.submissions(cik)
        except Exception:
            return cik, {}

    with ThreadPoolExecutor(workers) as pool:
        for i, (cik, record) in enumerate(pool.map(company, ciks), 1):
            companies[cik] = record
            say("company metadata", i, len(ciks))
    filings = [filing_from_hit(h, companies.get(h.ciks[0] if h.ciks else "")) for h in hits]
    deals = group_deals(filings)
    if download:
        by_key = {(h.accession, h.filename): h for h in hits}

        def fetch(filing: Filing) -> str:
            client.fetch_filing(by_key[(filing.accession, filing.filename)])
            return filing.accession

        todo = [f for d in deals for f in d.filings]
        with ThreadPoolExecutor(workers) as pool:
            for i, _ in enumerate(pool.map(fetch, todo), 1):
                say("download", i, len(todo))
    return deals


def write_catalog(deals: Iterable[Deal], path: str | Path) -> None:
    """Write the filing catalog as CSV (one row per filing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        for deal in deals:
            for filing in deal.filings:
                writer.writerow({"deal_id": deal.deal_id, **filing.to_row()})


def read_catalog(path: str | Path) -> list[Deal]:
    """Read a catalog written by :func:`write_catalog`."""
    deals: dict[str, Deal] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filing = Filing(
                accession=row["accession"],
                filename=row["filename"],
                form=row["form"],
                file_date=row["file_date"],
                cik=row["cik"],
                company=row["company"],
                sic=row["sic"],
                sic_description=row["sic_description"],
                tickers=tuple(t for t in row["tickers"].split(",") if t),
                current_name=row.get("current_name", ""),
            )
            deals.setdefault(row["deal_id"], Deal(row["deal_id"])).filings.append(filing)
    return list(deals.values())


def catalog_hits(deals: Iterable[Deal]) -> list[Hit]:
    """Rebuild minimal :class:`Hit` objects (for downloads) from catalog deals."""
    return [
        Hit(f.accession, f.filename, f.form, f.form, f.file_date, (f.cik,), (f.company,), (f.sic,))
        for d in deals
        for f in d.filings
    ]
