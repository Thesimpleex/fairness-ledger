"""Descriptive analyses of the extracted dataset.

Three questions, all descriptive and none causal:

* how far apart are two advisors' discount rates and terminal growth rates when they value
  the same target in the same proxy (:func:`bank_disagreement`);
* how often do exit-multiple DCFs imply perpetual growth above a nominal-growth bound
  (:func:`implied_growth_check`);
* how do disclosed discount rates and terminal growth move with the ten-year Treasury yield
  (:func:`market_practice`).

Unit of observation: ``(filing, advisor, subject)`` for the first two, the deal for the third.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .gold import wilson_interval
from .market import yield_on

NOMINAL_GROWTH_BOUND_PCT = 4.0
FINANCIAL_SIC = tuple(str(s) for s in range(60, 68))
UNNAMED = {"unattributed", "joint"}


def midpoints(frame: pd.DataFrame) -> pd.Series:
    """Midpoint of each fact's range (equal to the value for a point)."""
    return (frame["low"] + frame["high"]) / 2.0


def _target_dcf(facts: pd.DataFrame, field: str) -> pd.DataFrame:
    sel = (
        (facts["field"] == field)
        & (facts["subject"] == "target")
        & (facts["analysis"] == "dcf")
        & ~facts["advisor"].isin(UNNAMED)
    )
    out = facts.loc[sel].copy()
    out["mid"] = midpoints(out)
    return out


def advisor_values(facts: pd.DataFrame, field: str, single_only: bool = False) -> pd.DataFrame:
    """One value per (deal, advisor): the median midpoint of the advisor's target DCF facts.

    With ``single_only`` only advisors that state exactly one such range in the deal are
    kept, which removes ambiguity about scenarios and repeated statements.
    """
    sub = _target_dcf(facts, field)
    if single_only and not sub.empty:
        sub = sub.drop_duplicates(["deal_id", "advisor", "low", "high"])
        size = sub.groupby(["deal_id", "advisor"])["mid"].transform("size")
        sub = sub[size == 1]
    if sub.empty:
        return pd.DataFrame(columns=["deal_id", "advisor", "value"])
    grouped = sub.groupby(["deal_id", "advisor"])["mid"].median().rename("value").reset_index()
    return grouped


def pairwise_differences(values: pd.DataFrame) -> pd.DataFrame:
    """Absolute differences between advisors within a deal (one row per advisor pair)."""
    rows: list[dict[str, Any]] = []
    for deal_id, group in values.groupby("deal_id"):
        if len(group) < 2:
            continue
        for (_, a), (_, b) in combinations(group.iterrows(), 2):
            rows.append(
                {
                    "deal_id": deal_id,
                    "advisor_a": a["advisor"],
                    "advisor_b": b["advisor"],
                    "value_a": a["value"],
                    "value_b": b["value"],
                    "abs_diff": abs(a["value"] - b["value"]),
                }
            )
    return pd.DataFrame(
        rows, columns=["deal_id", "advisor_a", "advisor_b", "value_a", "value_b", "abs_diff"]
    )


def _cluster_bootstrap_median(pairs: pd.DataFrame, draws: int, seed: int) -> tuple[float, float]:
    deals = pairs["deal_id"].unique()
    by_deal = {d: g["abs_diff"].to_numpy() for d, g in pairs.groupby("deal_id")}
    rng = np.random.default_rng(seed)
    meds = np.empty(draws)
    for i in range(draws):
        pick = rng.choice(deals, size=len(deals), replace=True)
        meds[i] = np.median(np.concatenate([by_deal[d] for d in pick]))
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def bank_disagreement(
    facts: pd.DataFrame, draws: int = 2000, seed: int = 0, single_only: bool = False
) -> dict[str, Any]:
    """Distribution of differences between two advisors' target-DCF midpoints, per deal."""
    result: dict[str, Any] = {}
    for field, label in (
        ("discount_rate", "discount_rate"),
        ("terminal_growth", "terminal_growth"),
    ):
        values = advisor_values(facts, field, single_only)
        pairs = pairwise_differences(values)
        if pairs.empty:
            result[label] = {"n_pairs": 0, "n_deals": 0}
            continue
        d = pairs["abs_diff"].to_numpy()
        lo, hi = _cluster_bootstrap_median(pairs, draws, seed)
        result[label] = {
            "n_pairs": len(pairs),
            "n_deals": int(pairs["deal_id"].nunique()),
            "median_pp": float(np.median(d)),
            "median_ci_95": [lo, hi],
            "p25_pp": float(np.quantile(d, 0.25)),
            "p75_pp": float(np.quantile(d, 0.75)),
            "mean_pp": float(d.mean()),
            "share_ge_1pp": float((d >= 1.0).mean()),
            "share_ge_2pp": float((d >= 2.0).mean()),
            "max_pp": float(d.max()),
        }
        result[label]["_pairs"] = pairs
    return result


def implied_growth_check(
    facts: pd.DataFrame, bounds: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0), window: int = 6000
) -> dict[str, Any]:
    """Share of exit-multiple DCFs whose disclosed implied perpetuity growth exceeds a bound.

    An analysis unit is ``(filing, advisor, subject)`` with at least one exit-multiple fact in
    a DCF. It is *covered* if an implied-growth fact of the same filing and advisor lies within
    ``window`` characters of one of its exit-multiple facts. Shares are computed only over
    covered units; coverage itself is reported.
    """
    exits = facts[
        (facts["field"] == "exit_multiple")
        & (facts["analysis"] == "dcf")
        & ~facts["advisor"].isin(UNNAMED)
    ]
    implied = facts[facts["field"] == "implied_growth"]
    units: list[dict[str, Any]] = []
    for (accession, advisor, subject), group in exits.groupby(["accession", "advisor", "subject"]):
        near = implied[(implied["accession"] == accession) & (implied["advisor"] == advisor)]
        positions = group["span_start"].to_numpy()
        close = [bool(np.any(np.abs(positions - s) <= window)) for s in near["span_start"]]
        hit = near[close]
        units.append(
            {
                "accession": accession,
                "deal_id": group["deal_id"].iloc[0],
                "advisor": advisor,
                "subject": subject,
                "covered": not hit.empty,
                "implied_high": float(hit["high"].max()) if not hit.empty else math.nan,
                "implied_mid": float(midpoints(hit).median()) if not hit.empty else math.nan,
            }
        )
    frame = pd.DataFrame(
        units,
        columns=[
            "accession",
            "deal_id",
            "advisor",
            "subject",
            "covered",
            "implied_high",
            "implied_mid",
        ],
    )
    covered = frame[frame["covered"]]
    n, k = len(frame), len(covered)
    lo, hi = wilson_interval(k, n)
    result: dict[str, Any] = {
        "n_exit_multiple_dcfs": n,
        "n_with_disclosed_implied_growth": k,
        "coverage": k / n if n else math.nan,
        "coverage_ci_95": [lo, hi],
        "implied_growth_median_high_pct": float(covered["implied_high"].median())
        if k
        else math.nan,
        "bounds": {},
    }
    for bound in bounds:
        above_high = int((covered["implied_high"] > bound).sum())
        above_mid = int((covered["implied_mid"] > bound).sum())
        h_lo, h_hi = wilson_interval(above_high, k)
        m_lo, m_hi = wilson_interval(above_mid, k)
        result["bounds"][f"{bound:g}"] = {
            "n_covered": k,
            "high_end_above": above_high,
            "high_end_share": above_high / k if k else math.nan,
            "high_end_ci_95": [h_lo, h_hi],
            "midpoint_above": above_mid,
            "midpoint_share": above_mid / k if k else math.nan,
            "midpoint_ci_95": [m_lo, m_hi],
        }
    result["_units"] = frame
    return result


def deal_level_rates(facts: pd.DataFrame, yields: pd.Series) -> pd.DataFrame:
    """One row per deal: median target-DCF discount and growth midpoints, yield at filing date."""
    disc = _target_dcf(facts, "discount_rate").groupby("deal_id")["mid"].median().rename("discount")
    growth = (
        _target_dcf(facts, "terminal_growth").groupby("deal_id")["mid"].median().rename("growth")
    )
    meta = (
        facts.sort_values("filing_date")
        .groupby("deal_id")[["filing_date", "year", "sic", "company"]]
        .first()
    )
    frame = meta.join(disc, how="inner").join(growth, how="left").reset_index()
    frame["y10"] = yield_on(yields, frame["filing_date"])
    frame["sic2"] = frame["sic"].astype(str).str[:2]
    frame["financial"] = frame["sic2"].isin(FINANCIAL_SIC)
    return frame.dropna(subset=["y10"])


def _slope(frame: pd.DataFrame, y: str, fixed_effects: bool) -> float:
    x = frame["y10"].to_numpy(dtype=float)
    v = frame[y].to_numpy(dtype=float)
    if fixed_effects:
        key = frame["sic2"].to_numpy()
        x = x - pd.Series(x).groupby(key).transform("mean").to_numpy()
        v = v - pd.Series(v).groupby(key).transform("mean").to_numpy()
    else:
        x, v = x - x.mean(), v - v.mean()
    denom = float((x * x).sum())
    return float((x * v).sum() / denom) if denom > 0 else math.nan


def block_bootstrap_slope(
    frame: pd.DataFrame, y: str, fixed_effects: bool, draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval of the slope under a bootstrap that resamples filing years."""
    years = sorted(frame["year"].unique())
    by_year = {yr: frame[frame["year"] == yr] for yr in years}
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(draws):
        pick = rng.choice(years, size=len(years), replace=True)
        sample = pd.concat([by_year[yr] for yr in pick], ignore_index=True)
        s = _slope(sample, y, fixed_effects)
        if not math.isnan(s):
            slopes.append(s)
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def market_practice(
    facts: pd.DataFrame, yields: pd.Series, draws: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Discount rate and terminal growth midpoints by year, and descriptive pass-through."""
    frame = deal_level_rates(facts, yields)
    industrial = frame[~frame["financial"]]
    by_year = (
        industrial.groupby("year")
        .agg(
            n=("discount", "size"),
            discount_median=("discount", "median"),
            discount_q25=("discount", lambda s: s.quantile(0.25)),
            discount_q75=("discount", lambda s: s.quantile(0.75)),
            growth_median=("growth", "median"),
            growth_n=("growth", "count"),
            y10_mean=("y10", "mean"),
        )
        .reset_index()
    )
    out: dict[str, Any] = {
        "n_deals": len(industrial),
        "n_deals_with_growth": int(industrial["growth"].notna().sum()),
        "financials_excluded": int(frame["financial"].sum()),
        "pass_through": {},
    }
    for label, fe in (("pooled", False), ("sic2_fixed_effects", True)):
        sub = (
            industrial
            if not fe
            else industrial[industrial.groupby("sic2")["sic2"].transform("size") > 1]
        )
        slope = _slope(sub, "discount", fe)
        lo, hi = block_bootstrap_slope(sub, "discount", fe, draws, seed)
        out["pass_through"][label] = {
            "n": len(sub),
            "slope_pp_per_pp": slope,
            "ci_95_year_block_bootstrap": [lo, hi],
        }
    growth = industrial.dropna(subset=["growth"])
    if len(growth) > 30:
        slope = _slope(growth, "growth", False)
        lo, hi = block_bootstrap_slope(growth, "growth", False, draws, seed)
        out["growth_pass_through"] = {
            "n": len(growth),
            "slope_pp_per_pp": slope,
            "ci_95_year_block_bootstrap": [lo, hi],
        }
    out["_by_year"] = by_year
    out["_deals"] = frame
    return out


def strip_private(obj: Any) -> Any:
    """Copy of a result tree without the DataFrames stored under ``_``-prefixed keys."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
