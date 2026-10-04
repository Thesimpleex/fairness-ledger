"""End-to-end extraction on verbatim excerpts of five real proxies (one per house style)."""

from __future__ import annotations

import pytest

from conftest import excerpt_html
from fairness_ledger.document import html_to_document
from fairness_ledger.extract import Fact, extract_document
from fairness_ledger.provenance import verify_span

COMPANIES = {
    "goldman_altra": "Altra Industrial Motion Corp.",
    "centerview_anixter": "Anixter International Inc.",
    "sandler_poage": "Poage Bankshares, Inc.",
    "kbw_peoples": "Peoples Bancorp of North Carolina",
    "lazard_chenergy": "CH Energy Group, Inc.",
}


def run(name: str):
    doc = html_to_document(name, excerpt_html(name))
    return doc, extract_document(doc, COMPANIES[name])


def pick(facts: list[Fact], field: str) -> list[tuple[float, float]]:
    return [(f.low, f.high) for f in facts if f.field == field]


@pytest.mark.parametrize("name", sorted(COMPANIES))
def test_every_fact_is_proven_by_its_span(name):
    doc, result = run(name)
    assert result.facts
    for fact in result.facts:
        assert verify_span(fact, doc.text) == [], fact
        assert fact.doc_sha256 == doc.sha256
        assert len(fact.snippet) <= 300


def test_goldman_exit_multiple_dcf_with_implied_multiple_and_valuation_date():
    _, result = run("goldman_altra")
    by = {f.field: f for f in result.facts}
    assert (by["discount_rate"].low, by["discount_rate"].high) == (10.0, 11.0)
    assert by["discount_rate"].valuation_date == "2022-09-30"
    assert (by["terminal_growth"].low, by["terminal_growth"].high) == (2.5, 3.5)
    assert (by["implied_multiple"].low, by["implied_multiple"].high) == (8.0, 10.5)
    assert by["implied_multiple"].metric == "EBITDA"
    assert (by["value_per_share"].low, by["value_per_share"].high) == (48.0, 64.0)
    assert by["fee_total"].low == 45.0 and by["fee_total"].unit == "usd_million"
    assert by["offer_price"].low == 62.0
    for field in ("discount_rate", "terminal_growth", "implied_multiple"):
        assert by[field].advisor == "Goldman Sachs"
        assert by[field].subject == "target"
        assert by[field].analysis == "dcf"


def test_centerview_fee_components_add_up():
    _, result = run("centerview_anixter")
    facts = {f.field: f.low for f in result.facts if f.field.startswith("fee_")}
    assert facts == {"fee_total": 21.8, "fee_opinion": 4.5, "fee_contingent": 17.3}
    assert facts["fee_opinion"] + facts["fee_contingent"] == pytest.approx(facts["fee_total"])
    assert pick(result.facts, "value_per_share") == [(80.5, 124.75)]


def test_bank_dividend_discount_model_and_enumerated_rates():
    _, result = run("sandler_poage")
    assert pick(result.facts, "discount_rate") == [(11.0, 15.0), (12.68, 12.68)]
    assert pick(result.facts, "exit_multiple") == [(13.0, 18.0), (13.0, 18.0)]
    assert {f.analysis for f in result.facts if f.field != "fee_total"} == {"ddm"}
    assert {f.metric for f in result.facts if f.field == "exit_multiple"} == {"P/E"}


def test_acquirer_pro_forma_subject_is_separated_from_the_target():
    _, result = run("kbw_peoples")
    subjects = [f.subject for f in result.facts if f.field == "discount_rate"]
    assert "pro_forma" in subjects
    assert all(f.advisor == "KBW" for f in result.facts if f.field != "offer_price")


def test_two_analyses_in_one_opinion_stay_separate_with_their_own_metric():
    _, result = run("lazard_chenergy")
    assert pick(result.facts, "discount_rate") == [(6.0, 6.5), (6.75, 7.75)]
    metrics = [f.metric for f in result.facts if f.field == "exit_multiple"]
    assert metrics == ["P/E", "EBITDA"]


def test_text_without_valuation_content_yields_nothing():
    doc = html_to_document(
        "none",
        "<p>The Board approved a discount rate of 9.0% on the Company's employee stock "
        "purchase plan.</p><p>Revenue grew 4.5% in fiscal 2019.</p>",
    )
    result = extract_document(doc, "Acme Corp")
    assert result.facts == []


def test_extraction_is_deterministic():
    _, a = run("goldman_altra")
    _, b = run("goldman_altra")
    assert [f.to_row() for f in a.facts] == [f.to_row() for f in b.facts]
