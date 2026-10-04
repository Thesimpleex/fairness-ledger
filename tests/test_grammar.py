"""Grammar tests on sentences taken from real merger proxies (house styles) and negatives."""

from __future__ import annotations

import pytest

from fairness_ledger.grammar import find_candidates, parse_number


def values(text: str, field: str | None = None):
    return [
        (c.field, c.low, c.high) for c in find_candidates(text) if field is None or c.field == field
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("using discount rates ranging from 7.5% to 9.5%, which were", (7.5, 9.5)),
        ("a range of discount rates from 8.5% to 9.5%, chosen by", (8.5, 9.5)),
        ("a discount rate range of 5.00% to 7.00%. This analysis", (5.0, 7.0)),
        ("discount rates of 8.0% - 10.0% were selected", (8.0, 10.0)),
        ("discount rates of 8.0%–10.0% were selected", (8.0, 10.0)),
        ("discount rates between 9.0% and 11.0%", (9.0, 11.0)),
        ("discount rates of 8.0% through 10.0% were selected", (8.0, 10.0)),
        ("discount rates ranging from 8.0% through 10.0%", (8.0, 10.0)),
        ("discount rates ranging from 9.0 percent to 11.0 percent", (9.0, 11.0)),
        ("a weighted average cost of capital of 11.5% was used", (11.5, 11.5)),
        ("a cost of equity range from 10.75% to 12.75%, which", (10.75, 12.75)),
        ("an 11.0% discount rate and 3.5% perpetuity growth", (11.0, 11.0)),
        ("estimated 2022 earnings, 10.0% to 14.0% discount rates, using", (10.0, 14.0)),
        ("a discount rate range of 10.0%-12.0% for its analysis", (10.0, 12.0)),
    ],
)
def test_discount_rate_styles(text, expected):
    found = values(text, "discount_rate")
    assert found and found[0][1:] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("perpetuity growth rates ranging from (1.5)% to 1.5% to a terminal", (-1.5, 1.5)),
        ("perpetuity growth rates of (0.5%) to 0.5%", (-0.5, 0.5)),
        ("perpetuity growth rates of negative 1.0% to 1.0%", (-1.0, 1.0)),
        ("a perpetual growth rate between 3.0% and 3.5%.", (3.0, 3.5)),
        ("a terminal value growth rate of 2.5% to 3.5%", (2.5, 3.5)),
        ("a 3.25% terminal growth rate", (3.25, 3.25)),
        ("a dividend growth rate range of 1.50% to 3.00%.", (1.5, 3.0)),
    ],
)
def test_terminal_growth_styles_and_negatives(text, expected):
    found = values(text, "terminal_growth")
    assert found and found[0][1:] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("terminal EBITDA multiples of 7.0x to 8.0x. The", (7.0, 8.0)),
        ("an exit multiple range of 10.0x – 12.0x", (10.0, 12.0)),
        ("terminal value multiples ranging from 9.5x to 11.5x", (9.5, 11.5)),
        ("a terminal multiple of 9.5 times", (9.5, 9.5)),
        ("by applying terminal multiples ranging from 6.0 times to 8.0 times", (6.0, 8.0)),
        ("applied a range of 12.0x to 16.0x to terminal value", None),
    ],
)
def test_exit_multiple_styles(text, expected):
    found = values(text, "exit_multiple")
    if expected is None:
        assert not found
    else:
        assert found and found[0][1:] == expected


def test_terminal_value_sentence_with_distant_cue():
    text = (
        "In calculating the terminal value of NewBridge, KBW applied a range of 12.0x to 16.0x "
        "estimated 2021 net income."
    )
    assert values(text, "exit_multiple")[0][1:] == (12.0, 16.0)


def test_implied_fields_are_separate():
    text = (
        "calculated by applying perpetuity growth rates ranging from 2.0% to 3.0% (which analysis "
        "implied exit terminal year EBITDA multiples ranging from 12.0x to 20.0x) and a terminal "
        "EBITDA multiple range of 8.5x to 10.0x which implied perpetuity growth rates ranging "
        "from 0.5% to 2.3%."
    )
    found = values(text)
    assert ("terminal_growth", 2.0, 3.0) in found
    assert ("implied_multiple", 12.0, 20.0) in found
    assert ("exit_multiple", 8.5, 10.0) in found
    assert ("implied_growth", 0.5, 2.3) in found


def test_enumerations_become_separate_candidates():
    text = "perpetuity growth rates of 1.00%, 2.00% and 3.00% and discount rates of 9.00%, 8.13% and 7.25% to"
    found = values(text)
    assert [v for f, *v in found if f == "terminal_growth"] == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert [v for f, *v in found if f == "discount_rate"] == [
        [9.0, 9.0],
        [8.13, 8.13],
        [7.25, 7.25],
    ]
    ranges = "terminal EBITDA multiples of 9.0x to 10.0x, 4.0x to 6.0x, 7.0x to 9.0x and 8.5x to 9.5x for"
    assert [v for f, *v in values(ranges) if f == "exit_multiple"] == [
        [9.0, 10.0],
        [4.0, 6.0],
        [7.0, 9.0],
        [8.5, 9.5],
    ]


def test_federal_reserve_rate_and_table_rows_rejected():
    assert not values("5% over the Federal Reserve discount rate (including any surcharge)")
    # a parenthesised bare number is never a percentage point value
    assert not values("Perpetuity Growth Rate Method | $ | (2.44) - $(1.45 | )")


def test_guard_against_next_cue():
    found = values("a discount rate and 3.5% perpetuity growth rate", "discount_rate")
    assert not found


def test_candidate_spans_reproduce_text():
    text = "The analysis applied discount rates ranging from 7.5% to 9.5%, which were chosen."
    (c,) = find_candidates(text)
    assert text[c.start : c.end] == "discount rates ranging from 7.5% to 9.5%"
    assert (c.raw_low, c.raw_high) == ("7.5%", "9.5%")


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("9.0%", 9.0),
        ("(1.0)%", -1.0),
        ("(1.0%)", -1.0),
        ("-2.5%", -2.5),
        ("negative 1.5 percent", -1.5),
        ("$1,200.50", 1200.5),
    ],
)
def test_parse_number(raw, value):
    assert parse_number(raw) == value


def test_parse_number_rejects_text():
    with pytest.raises(ValueError):
        parse_number("none")
