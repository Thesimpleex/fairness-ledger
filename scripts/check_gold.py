"""Validate label files: every quote must resolve to a span in its filing."""

from __future__ import annotations

import sys
from pathlib import Path

from fairness_ledger.catalog import read_catalog
from fairness_ledger.document import html_to_document
from fairness_ledger.edgar import read_cached_filing
from fairness_ledger.gold import parse_labels

ROOT = Path(__file__).resolve().parents[1]


def main(accessions: list[str]) -> int:
    deals = read_catalog(ROOT / "data" / "release" / "filings.csv")
    by_acc = {f.accession: f for d in deals for f in d.filings}
    labels = sorted((ROOT / "gold" / "labels").glob("*.txt"))
    bad = 0
    for path in labels:
        acc = path.stem
        if accessions and acc not in accessions:
            continue
        filing = by_acc[acc]
        raw = read_cached_filing(ROOT / "data" / "raw" / "filings" / acc / f"{filing.filename}.gz")
        doc = html_to_document(acc, raw)
        try:
            facts = parse_labels(acc, doc.text, path.read_text(encoding="utf-8"))
            print(f"{acc}: {len(facts)} labels ok")
        except ValueError as exc:
            bad += 1
            print(f"{acc}: ERROR {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
