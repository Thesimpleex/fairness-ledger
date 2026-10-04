"""Run the extractor on a gold split and score it against the labels."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import read_catalog
from .document import html_to_document
from .edgar import read_cached_filing
from .extract import extract_document
from .gold import GoldFact, PredictedFact, evaluate, labels_digest, parse_labels


@dataclass
class SplitRun:
    """Predictions, gold labels and the scored result of one split."""

    split: str
    accessions: list[str]
    gold: list[GoldFact]
    predicted: list[PredictedFact]
    texts: dict[str, str]
    rejected: dict[str, list[dict[str, Any]]]
    result: dict[str, Any]


def _run_one(args: tuple[str, str, str, str]) -> tuple[str, list[PredictedFact], str, list[dict]]:
    root, accession, filename, company = args
    raw = read_cached_filing(Path(root) / "data" / "raw" / "filings" / accession / f"{filename}.gz")
    doc = html_to_document(accession, raw)
    result = extract_document(doc, company)
    predicted = [
        PredictedFact(
            accession,
            f.field,
            f.low,
            f.high,
            f.advisor,
            f.analysis,
            f.subject,
            f.metric,
            f.valuation_date,
            f.span_start,
            f.span_end,
        )
        for f in result.facts
    ]
    return accession, predicted, doc.text, [r.__dict__ for r in result.rejections]


def load_manifest(root: Path, split: str) -> list[dict[str, str]]:
    manifest = json.loads((root / "gold" / "manifest.json").read_text())
    return [m for m in manifest if m["split"] == split]


def verify_frozen(root: Path) -> bool:
    """Do the holdout label files still match the frozen digest?"""
    frozen = (root / "gold" / "holdout.sha256").read_text().split()[0]
    paths = [
        root / "gold" / "labels" / f"{m['accession']}.txt" for m in load_manifest(root, "holdout")
    ]
    return labels_digest(paths) == frozen


def run_split(root: str | Path, split: str, workers: int | None = None) -> SplitRun:
    """Extract from every filing of ``split`` and score against ``gold/labels``.

    Raises ``RuntimeError`` for the holdout split if its labels differ from the frozen hash.
    """
    root = Path(root)
    if split == "holdout" and not verify_frozen(root):
        raise RuntimeError("holdout labels do not match gold/holdout.sha256")
    manifest = load_manifest(root, split)
    by_acc = {
        f.accession: f
        for d in read_catalog(root / "data" / "release" / "filings.csv")
        for f in d.filings
    }
    jobs = [
        (str(root), m["accession"], by_acc[m["accession"]].filename, by_acc[m["accession"]].company)
        for m in manifest
    ]
    with ProcessPoolExecutor(workers) as pool:
        outputs = list(pool.map(_run_one, jobs))
    texts = {acc: text for acc, _, text, _ in outputs}
    predicted = [p for _, preds, _, _ in outputs for p in preds]
    rejected = {acc: rej for acc, _, _, rej in outputs}
    gold: list[GoldFact] = []
    for m in manifest:
        label_text = (root / "gold" / "labels" / f"{m['accession']}.txt").read_text(
            encoding="utf-8"
        )
        gold += parse_labels(m["accession"], texts[m["accession"]], label_text)
    groups = {
        "era": {m["accession"]: m["era"] for m in manifest},
        "advisor": {m["accession"]: m["dominant_advisor"] for m in manifest},
    }
    result = evaluate(gold, predicted, groups)
    result["split"] = split
    result["n_filings"] = len(manifest)
    return SplitRun(
        split, [m["accession"] for m in manifest], gold, predicted, texts, rejected, result
    )
