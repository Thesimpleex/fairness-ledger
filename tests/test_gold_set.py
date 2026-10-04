"""Integrity of the committed gold set (offline: the filings themselves are not committed)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fairness_ledger.evaluation import load_manifest, verify_frozen
from fairness_ledger.gold import ANALYSIS_CODES, FIELDS, SUBJECT_CODES

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "gold" / "manifest.json").read_text())


def test_manifest_size_and_split_are_as_documented():
    assert 50 <= len(MANIFEST) <= 60
    splits = {m["split"] for m in MANIFEST}
    assert splits == {"dev", "holdout"}
    holdout = [m for m in MANIFEST if m["split"] == "holdout"]
    assert len(holdout) == 30
    assert len({m["accession"] for m in MANIFEST}) == len(MANIFEST)


def test_every_manifest_entry_has_a_label_file_and_vice_versa():
    files = {p.stem for p in (ROOT / "gold" / "labels").glob("*.txt")}
    assert files == {m["accession"] for m in MANIFEST}


def test_holdout_labels_match_the_frozen_digest():
    assert verify_frozen(ROOT)
    assert len(load_manifest(ROOT, "holdout")) == 30


def test_label_lines_are_well_formed():
    n = 0
    for path in (ROOT / "gold" / "labels").glob("*.txt"):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 8)
            assert len(parts) == 9, (path.name, line)
            field, low, high, _advisor, analysis, subject, _metric, vdate, quote = parts
            assert field in FIELDS
            assert float(low) <= float(high)
            assert analysis in ANALYSIS_CODES
            assert subject in SUBJECT_CODES
            assert vdate == "-" or re.fullmatch(r"\d{4}-\d{2}-\d{2}", vdate)
            assert quote.strip()
            n += 1
    assert n > 300


def test_dev_pool_excludes_holdout():
    pool = set((ROOT / "gold" / "dev_pool.txt").read_text().split())
    holdout = {m["accession"] for m in MANIFEST if m["split"] == "holdout"}
    assert pool and not pool & holdout


@pytest.mark.parametrize("name", ["manifest.json", "holdout.sha256"])
def test_gold_files_exist(name):
    assert (ROOT / "gold" / name).exists()
