from __future__ import annotations

import pytest

from fairness_ledger.advisors import MentionIndex
from fairness_ledger.deal_terms import (
    find_fee_terms,
    find_offer_price,
    find_per_share_ranges,
    find_valuation_date,
    money_value,
    reparse_money,
)


def test_money_value_scales():
    assert money_value("1,250", "thousand", "usd_million") == pytest.approx(1.25)
    assert money_value("2.5", "million", "usd_million") == 2.5
    assert money_value("1.2", "billion", "usd_million") == pytest.approx(1200.0)
    assert money_value("62.00", None, "usd") == 62.0
    assert reparse_money("$21.8 million, of which $4.5 million", "usd_million") == [21.8, 4.5]


def test_offer_price_needs_cash_consideration_and_repetition():
    text = (
        "each share will be converted into the right to receive $62.00 in cash, without interest. "
        "Stockholders will receive $62.00 in cash for each share. A break-up fee of $15.00 applies."
    )
    term = find_offer_price(text)
    assert term is not None and term.low == 62.0 and term.field == "offer_price"


def test_offer_price_is_skipped_for_mixed_consideration():
    text = (
        "right to receive $20.00 in cash and 0.776 of a share of Parent common stock. "
        "right to receive $20.00 in cash and 0.776 of a share of Parent common stock."
    )
    assert find_offer_price(text) is None


def test_fee_total_opinion_and_contingent():
    text = (
        "Evercore will receive an aggregate fee of approximately $21.8 million, of which "
        "$4.5 million became payable upon delivery of its opinion and $17.3 million is "
        "contingent upon the closing of the merger."
    )
    terms = find_fee_terms(text, MentionIndex.build(text))
    assert {t.field: t.low for t in terms} == {
        "fee_total": 21.8,
        "fee_opinion": 4.5,
        "fee_contingent": 17.3,
    }
    assert all(t.house == "Evercore" for t in terms)


def test_termination_fee_is_not_an_advisor_fee():
    text = "Goldman Sachs advised Parent. The Company must pay a termination fee of $120 million."
    assert find_fee_terms(text, MentionIndex.build(text)) == []


def test_per_share_ranges_need_a_derived_value_cue():
    good = "This analysis implied a range of values per share of $48.00 to $64.00 for Altra common stock."
    bad = "The stock traded between $48.00 and $64.00 per share during the period."
    assert [(t.low, t.high) for t in find_per_share_ranges(good)] == [(48.0, 64.0)]
    assert find_per_share_ranges(bad) == []


def test_valuation_date_in_both_notations():
    mdy = "discounted to present value as of September 30, 2022 at discount rates"
    dmy = "discounted to present value as of 30 September 2022 at discount rates"
    assert find_valuation_date(mdy, 60) == "2022-09-30"
    assert find_valuation_date(dmy, 60) == "2022-09-30"
    assert find_valuation_date("The meeting was held on May 3, 2021 to vote", 5) == ""
