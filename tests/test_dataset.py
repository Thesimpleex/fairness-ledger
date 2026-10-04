from __future__ import annotations

import gzip

import pandas as pd
import pytest

from conftest import excerpt_html
from fairness_ledger import load
from fairness_ledger.catalog import read_catalog, write_catalog
from fairness_ledger.dataset import COLUMNS, build_dataset, finalize, write_dataset
from fairness_ledger.deals import Deal, Filing
from fairness_ledger.document import html_to_document
from fairness_ledger.provenance import verify_span
from fairness_ledger.verify import verify_dataset


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    specs = [
        ("0000000001-20-000001", "goldman_altra", "Altra Industrial Motion Corp.", "2020-03-01"),
        ("0000000001-20-000002", "goldman_altra", "Altra Industrial Motion Corp.", "2020-04-01"),
        ("0000000002-21-000001", "lazard_chenergy", "CH Energy Group, Inc.", "2021-05-01"),
    ]
    filings = []
    for accession, name, company, day in specs:
        path = raw / "filings" / accession / "p.htm.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(gzip.compress(excerpt_html(name).encode()))
        filings.append(
            Filing(accession, "p.htm", "DEFM14A", day, accession[:10], company, sic="3560")
        )
    deals = [Deal("0000000001-20-000001", filings[:2]), Deal("0000000002-21-000001", filings[2:])]
    return raw, deals


def test_build_dataset_deduplicates_within_deal_and_counts_filings(corpus):
    raw, deals = corpus
    frame, rejections = build_dataset(deals, raw, workers=1)
    assert list(frame.columns) == list(COLUMNS)
    altra = frame[frame["deal_id"] == "0000000001-20-000001"]
    assert altra["accession"].eq("0000000001-20-000001").all()  # first filing is kept
    assert altra["n_filings"].eq(2).all()
    assert len(altra) == 6
    assert frame["fact_id"].is_unique
    assert set(frame["tier"]) == {"core", "auxiliary"}
    assert isinstance(rejections, dict)


def test_release_round_trip_and_verification_against_cache(corpus, tmp_path):
    raw, deals = corpus
    frame, _ = build_dataset(deals, raw, workers=1)
    write_dataset(frame, tmp_path)
    loaded = load(tmp_path)
    pd.testing.assert_frame_equal(loaded, frame)
    csv = pd.read_csv(tmp_path / "facts.csv", dtype={"cik": str, "sic": str})
    assert len(csv) == len(frame)
    write_catalog(deals, tmp_path / "filings.csv")
    assert [d.accessions for d in read_catalog(tmp_path / "filings.csv")] == [
        d.accessions for d in deals
    ]
    report = verify_dataset(loaded, raw)
    assert report.ok and report.verified == report.checked == len(frame)


def test_verification_detects_tampering(corpus, tmp_path):
    raw, deals = corpus
    frame, _ = build_dataset(deals, raw, workers=1)
    bad = frame.copy()
    bad.loc[0, "low"] = bad.loc[0, "low"] + 1
    assert not verify_dataset(bad, raw).ok
    stale = frame.copy()
    stale["doc_sha256"] = "0" * 64
    report = verify_dataset(stale, raw)
    assert report.hash_mismatch > 0 and not report.ok


def test_every_released_fact_passes_the_span_check(corpus):
    raw, deals = corpus
    frame, _ = build_dataset(deals, raw, workers=1)
    doc = html_to_document("x", excerpt_html("lazard_chenergy"))
    for row in frame[frame["deal_id"] == "0000000002-21-000001"].itertuples():
        assert verify_span(row, doc.text) == []


def test_finalize_of_empty_frame_has_all_columns():
    assert list(finalize(pd.DataFrame()).columns) == list(COLUMNS)


def test_load_validates_kind_and_missing_directory(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, kind="nope")
    with pytest.raises(FileNotFoundError):
        load(tmp_path)
