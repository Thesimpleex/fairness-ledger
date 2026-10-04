from __future__ import annotations

import math

import pytest

from fairness_ledger.gold import (
    PredictedFact,
    bootstrap_precision,
    evaluate,
    labels_digest,
    locate_quote,
    match_facts,
    parse_labels,
    wilson_interval,
)

TEXT = (
    "Perella used discount rates ranging from 8.0% to 10.0%. "
    "It also applied terminal multiples of 7.0x to 9.0x. the discount rates ranging from 8.0% to 10.0% again."
)
LABELS = """
# comment
discount_rate|8.0|10.0|Perella Weinberg|dcf|T|-|-|discount rates ranging from 8.0% to 10.0%#1
exit_multiple|7.0|9.0|Perella Weinberg|dcf|T|EBITDA|-|terminal multiples of 7.0x to 9.0x
"""


def predicted(field, low, high, start, end, accession="x"):
    return PredictedFact(accession, field, low, high, "", "", "", "", "", start, end)


def test_locate_quote_whitespace_tolerant_occurrence_and_ambiguity():
    s, e = locate_quote(TEXT, "terminal  multiples of 7.0x", None)
    assert TEXT[s:e] == "terminal multiples of 7.0x"
    with pytest.raises(ValueError, match="ambiguous"):
        locate_quote(TEXT, "discount rates ranging from 8.0% to 10.0%", None)
    s2, _ = locate_quote(TEXT, "discount rates ranging from 8.0% to 10.0%", 2)
    assert s2 > 50
    with pytest.raises(ValueError, match="not found"):
        locate_quote(TEXT, "nowhere", None)


def test_parse_labels():
    gold = parse_labels("x", TEXT, LABELS)
    assert [g.field for g in gold] == ["discount_rate", "exit_multiple"]
    assert gold[0].advisor == "Perella Weinberg"
    assert gold[0].valuation_date == "" and gold[1].metric == "EBITDA"
    assert TEXT[gold[1].span_start : gold[1].span_end] == "terminal multiples of 7.0x to 9.0x"


@pytest.mark.parametrize(
    "line",
    ["discount_rate|8|10|A|dcf|T|-|-", "bogus|8|10|A|dcf|T|-|-|x"],
)
def test_parse_labels_rejects_malformed_lines(line):
    with pytest.raises(ValueError):
        parse_labels("x", TEXT, line)


def test_matching_is_one_to_one_on_values_and_span_overlap():
    gold = parse_labels("x", TEXT, LABELS)
    g0, g1 = gold
    hits = [
        predicted("discount_rate", 8.0, 10.0, g0.span_start + 5, g0.span_end),
        predicted("discount_rate", 8.0, 10.0, g0.span_start, g0.span_end),  # duplicate
        predicted("exit_multiple", 7.0, 9.5, g1.span_start, g1.span_end),  # wrong value
    ]
    pairs, missed, extra = match_facts(gold, hits)
    assert len(pairs) == 1 and pairs[0][0] is g0
    assert missed == [g1]
    assert len(extra) == 2


def test_evaluate_scores_and_attributes():
    gold = parse_labels("x", TEXT, LABELS)
    pred = [
        PredictedFact(
            "x",
            "discount_rate",
            8.0,
            10.0,
            "Perella Weinberg",
            "dcf",
            "target",
            "",
            "",
            gold[0].span_start,
            gold[0].span_end,
        ),
        predicted("terminal_growth", 2.0, 3.0, 0, 5),
    ]
    result = evaluate(gold, pred)
    dr = result["by_field"]["discount_rate"]
    assert (dr["tp"], dr["fp"], dr["fn"]) == (1, 0, 0)
    assert result["by_field"]["terminal_growth"]["precision"] == 0.0
    assert result["by_field"]["exit_multiple"]["recall"] == 0.0
    assert result["key_fields"]["tp"] == 1 and result["key_fields"]["fn"] == 1
    assert result["attributes"]["advisor"]["accuracy"] == 1.0


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == (pytest.approx(0.4038, abs=1e-3), pytest.approx(0.5962, abs=1e-3))
    assert math.isnan(wilson_interval(0, 0)[0])


def test_bootstrap_precision_interval_contains_point_estimate():
    data = {f"f{i}": (9, 1) for i in range(30)}
    lo, hi = bootstrap_precision(data, draws=500, seed=1)
    assert lo <= 0.9 <= hi


def test_labels_digest_depends_on_content_and_name(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("one")
    b.write_text("two")
    d1 = labels_digest([a, b])
    assert d1 == labels_digest([b, a])
    b.write_text("three")
    assert d1 != labels_digest([a, b])
