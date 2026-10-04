from __future__ import annotations

import math

import pandas as pd
import pytest

from fairness_ledger.consistency import add_flags, growth_matches, implied_perpetuity_growth


def test_implied_growth_formula_end_of_year():
    # TV = 10 * 100 = 1000, UFCF = 60, r = 10%: g = (1000*0.1 - 60) / (1000 + 60)
    g = implied_perpetuity_growth(10.0, 100.0, 60.0, 0.10)
    assert g == pytest.approx(40.0 / 1060.0)


def test_implied_growth_inverts_gordon_growth():
    r, g_true, ufcf = 0.09, 0.025, 80.0
    tv = ufcf * (1 + g_true) / (r - g_true)
    assert implied_perpetuity_growth(tv / 120.0, 120.0, ufcf, r) == pytest.approx(g_true)


def test_mid_year_convention_lowers_implied_growth_for_same_multiple():
    end = implied_perpetuity_growth(9.0, 100.0, 55.0, 0.10, "end")
    mid = implied_perpetuity_growth(9.0, 100.0, 55.0, 0.10, "mid")
    assert mid < end
    tv, cash = 900.0, 55.0 * math.sqrt(1.10)
    assert mid == pytest.approx((tv * 0.10 - cash) / (tv + cash))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"multiple": 0.0},
        {"terminal_metric": -1.0},
        {"terminal_ufcf": 0.0},
        {"discount_rate": 9.0},
        {"timing": "quarterly"},
    ],
)
def test_implied_growth_input_validation(kwargs):
    base = {
        "multiple": 9.0,
        "terminal_metric": 100.0,
        "terminal_ufcf": 50.0,
        "discount_rate": 0.1,
        "timing": "end",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        implied_perpetuity_growth(**base)


def test_growth_matches_tolerance():
    assert growth_matches(3.0, 3.1)
    assert not growth_matches(3.0, 3.4)


def _row(acc, field, low, high, start, advisor="A", subject="target"):
    return {
        "accession": acc,
        "field": field,
        "low": low,
        "high": high,
        "span_start": start,
        "advisor": advisor,
        "subject": subject,
    }


def test_soft_flags():
    frame = pd.DataFrame(
        [
            _row("a", "discount_rate", 8.0, 10.0, 100),
            _row("a", "terminal_growth", 9.5, 10.5, 200),
            _row("a", "discount_rate", 30.0, 35.0, 90_000),
            _row("a", "implied_growth", 4.5, 6.0, 500, advisor="B"),
            _row("a", "exit_multiple", 45.0, 50.0, 700, advisor="B"),
            _row("b", "terminal_growth", 2.0, 3.0, 100),
            _row("b", "discount_rate", 8.0, 10.0, 150, subject="acquirer"),
        ]
    )
    flags = add_flags(frame)["flags"].tolist()
    assert flags[0] == "growth_ge_discount"
    assert flags[1] == "growth_atypical;growth_ge_discount"
    assert flags[2] == "rate_atypical"
    assert flags[3] == "implied_growth_high"
    assert flags[4] == "multiple_atypical"
    assert flags[5] == "" and flags[6] == ""
