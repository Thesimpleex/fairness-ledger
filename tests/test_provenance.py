from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import excerpt_html
from fairness_ledger.document import html_to_document
from fairness_ledger.extract import extract_document
from fairness_ledger.provenance import verify_span


@pytest.fixture(scope="module")
def sample():
    doc = html_to_document("g", excerpt_html("goldman_altra"))
    facts = extract_document(doc, "Altra Industrial Motion Corp.").facts
    fact = next(f for f in facts if f.field == "discount_rate")
    return doc.text, fact


def test_clean_fact_passes(sample):
    text, fact = sample
    assert verify_span(fact, text) == []


def test_span_outside_text_is_rejected(sample):
    text, fact = sample
    assert verify_span(replace(fact, span_end=len(text) + 5), text) == ["span_out_of_range"]
    assert verify_span(replace(fact, span_start=fact.span_end), text) == ["span_out_of_range"]


def test_altered_value_is_not_reproduced(sample):
    text, fact = sample
    problems = verify_span(replace(fact, low=fact.low + 1.0), text)
    assert "value_not_reproduced_from_span" in problems


def test_value_text_must_be_inside_span(sample):
    text, fact = sample
    assert "value_text_not_in_span" in verify_span(replace(fact, raw_low="99.9%"), text)


def test_snippet_must_be_verbatim_and_contain_the_span(sample):
    text, fact = sample
    assert "snippet_not_verbatim" in verify_span(replace(fact, snippet="made up"), text)
    moved = replace(fact, snippet_start=fact.span_start + 1)
    assert "snippet_does_not_contain_span" in verify_span(moved, text)


def test_snippet_length_cap(sample):
    text, fact = sample
    assert "snippet_too_long" in verify_span(replace(fact, snippet="x " * 200), text)
