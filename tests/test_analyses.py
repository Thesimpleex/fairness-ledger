from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fairness_ledger.analyses import (
    advisor_values,
    bank_disagreement,
    block_bootstrap_slope,
    implied_growth_check,
    market_practice,
    pairwise_differences,
    strip_private,
)
from fairness_ledger.market import load_dgs10, yield_on


def fact(deal, field, low, high, advisor="Lazard", subject="target", analysis="dcf", **kw):
    row = {
        "deal_id": deal,
        "accession": deal,
        "field": field,
        "low": low,
        "high": high,
        "advisor": advisor,
        "subject": subject,
        "analysis": analysis,
        "span_start": 1000,
        "filing_date": "2019-06-01",
        "year": 2019,
        "sic": "3560",
        "company": deal,
    }
    row.update(kw)
    return row


def test_bank_disagreement_pairs_advisors_within_a_deal():
    facts = pd.DataFrame(
        [
            fact("d1", "discount_rate", 8.0, 10.0, advisor="Lazard"),  # midpoint 9
            fact("d1", "discount_rate", 10.0, 12.0, advisor="Evercore"),  # midpoint 11
            fact("d1", "discount_rate", 9.0, 9.0, advisor="Evercore", span_start=2000),  # median 10
            fact("d2", "discount_rate", 7.0, 7.0, advisor="Lazard"),
            fact("d2", "discount_rate", 7.5, 7.5, advisor="unattributed"),  # excluded
            fact("d3", "discount_rate", 6.0, 8.0, advisor="Lazard", subject="acquirer"),  # excluded
        ]
    )
    values = advisor_values(facts, "discount_rate")
    assert len(values) == 3
    pairs = pairwise_differences(values)
    assert len(pairs) == 1
    assert pairs["abs_diff"].iloc[0] == pytest.approx(1.0)
    result = bank_disagreement(facts, draws=50, seed=1)
    assert result["discount_rate"]["n_pairs"] == 1
    assert result["terminal_growth"]["n_pairs"] == 0


def exit_dcf(accession, advisor, implied_high=None):
    rows = [fact(accession, "exit_multiple", 8.0, 10.0, advisor=advisor)]
    if implied_high is not None:
        rows.append(
            fact(accession, "implied_growth", 1.0, implied_high, advisor=advisor, span_start=1500)
        )
    return rows


def test_implied_growth_coverage_and_bound_shares():
    rows = []
    rows += exit_dcf("a1", "Lazard", 3.0)
    rows += exit_dcf("a2", "Lazard", 5.0)
    rows += exit_dcf("a3", "Lazard", 7.5)
    rows += exit_dcf("a4", "Lazard")  # no implied growth disclosed
    rows.append(fact("a5", "implied_growth", 2.0, 9.0, span_start=90_000))
    rows += exit_dcf("a5", "Lazard")  # implied growth too far away to be paired
    result = implied_growth_check(pd.DataFrame(rows), bounds=(4.0,))
    assert result["n_exit_multiple_dcfs"] == 5
    assert result["n_with_disclosed_implied_growth"] == 3
    assert result["coverage"] == pytest.approx(0.6)
    bound = result["bounds"]["4"]
    assert bound["high_end_above"] == 2
    assert bound["high_end_share"] == pytest.approx(2 / 3)
    lo, hi = bound["high_end_ci_95"]
    assert lo < 2 / 3 < hi


def test_market_practice_recovers_a_known_slope(tmp_path):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2012-01-02", "2021-12-31", freq="D")
    y10 = pd.Series(1.5 + 0.5 * np.sin(np.arange(len(dates)) / 400.0) + 1.0, index=dates)
    rows = []
    for i in range(160):
        filed = dates[int(rng.integers(0, len(dates)))]
        base = float(y10.loc[filed])
        rate = 6.0 + 1.0 * base + rng.normal(0, 0.2)
        rows.append(
            fact(
                f"m{i}",
                "discount_rate",
                rate - 0.5,
                rate + 0.5,
                filing_date=filed.strftime("%Y-%m-%d"),
                year=filed.year,
                sic=str(3500 + i % 7),
            )
        )
        rows.append(
            fact(
                f"m{i}",
                "terminal_growth",
                2.0,
                3.0,
                filing_date=filed.strftime("%Y-%m-%d"),
                year=filed.year,
                sic=str(3500 + i % 7),
            )
        )
    rows.append(fact("bank", "discount_rate", 9.0, 11.0, sic="6022"))
    result = market_practice(pd.DataFrame(rows), y10, draws=200, seed=0)
    pooled = result["pass_through"]["pooled"]
    assert pooled["slope_pp_per_pp"] == pytest.approx(1.0, abs=0.15)
    assert result["financials_excluded"] == 1
    assert result["n_deals"] == 160
    lo, hi = pooled["ci_95_year_block_bootstrap"]
    assert lo <= pooled["slope_pp_per_pp"] <= hi
    assert set(result["_by_year"]["year"]) <= set(range(2012, 2022))


def test_block_bootstrap_resamples_years():
    frame = pd.DataFrame(
        {
            "year": np.repeat([2015, 2016, 2017, 2018], 10),
            "y10": np.tile(np.linspace(1, 3, 10), 4),
            "discount": np.tile(np.linspace(1, 3, 10), 4) * 2.0,
            "sic2": "35",
        }
    )
    lo, hi = block_bootstrap_slope(frame, "discount", False, draws=100, seed=0)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_yield_lookup_uses_last_observation_on_or_before(tmp_path):
    path = tmp_path / "dgs10.csv"
    path.write_text("date,dgs10\n2020-01-02,1.8\n2020-01-03,1.9\n2020-01-06,2.0\n")
    series = load_dgs10(path)
    got = yield_on(series, pd.Series(["2020-01-01", "2020-01-04", "2020-01-06", "2020-02-01"]))
    assert math.isnan(got[0])
    assert list(got[1:]) == [1.9, 2.0, 2.0]


def test_strip_private_removes_frames_and_nan():
    tree = {"a": 1.0, "_f": pd.DataFrame(), "b": {"c": math.nan, "_g": 3}, "d": [{"_h": 1, "e": 2}]}
    assert strip_private(tree) == {"a": 1.0, "b": {"c": None}, "d": [{"e": 2}]}
