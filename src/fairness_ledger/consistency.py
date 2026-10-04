"""Consistency algebra for extracted valuation parameters.

Hard checks (``low <= high``, units, plausibility bounds) are enforced at extraction.
This module adds *soft* flags to admitted facts, never silently accepting a violation:

* ``rate_atypical`` / ``growth_atypical`` / ``multiple_atypical``: outside the range
  practitioners treat as typical (stated below);
* ``growth_ge_discount``: a terminal growth rate at or above the discount rate used with it
  (the Gordon-growth terminal value is then undefined or negative);
* ``implied_growth_high``: an implied perpetuity growth rate above the nominal-growth bound.

It also provides :func:`implied_perpetuity_growth`, the recomputation of the perpetuity growth
rate implied by an exit multiple, with an explicit timing convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

TYPICAL_BOUNDS_PCT = {
    "discount_rate": (3.0, 25.0),
    "terminal_growth": (-3.0, 6.0),
    "implied_growth": (-5.0, 8.0),
}
TYPICAL_MULTIPLE_BOUNDS = (1.0, 40.0)
NOMINAL_GROWTH_BOUND_PCT = 4.0
PAIRING_DISTANCE = 3000


def implied_perpetuity_growth(
    multiple: float,
    terminal_metric: float,
    terminal_ufcf: float,
    discount_rate: float,
    timing: str = "end",
) -> float:
    """Perpetuity growth rate implied by an exit multiple.

    The exit method values the business at the end of the terminal year ``T`` as
    ``TV = multiple * terminal_metric``. Equating this with the Gordon-growth value of
    the cash flow of year ``T + 1`` gives, with end-of-year discounting,

    .. math:: TV = \\frac{UFCF_T (1 + g)}{r - g}
       \\quad\\Longrightarrow\\quad g = \\frac{TV\\, r - UFCF_T}{TV + UFCF_T}.

    With mid-year discounting the perpetuity is worth ``(1 + r)^{1/2}`` times more, so
    ``UFCF_T`` is replaced by ``UFCF_T (1 + r)^{1/2}`` (Goldman-style disclosures use
    the end-of-year form unless they state a mid-year convention).

    Parameters
    ----------
    multiple
        Exit multiple of the terminal metric (for example 10.0 for 10.0x EBITDA).
    terminal_metric, terminal_ufcf
        Terminal-year metric and unlevered free cash flow in the same currency units.
    discount_rate
        Discount rate as a decimal (0.09 for 9%).
    timing
        ``"end"`` or ``"mid"`` year discounting.

    Returns
    -------
    float
        Implied growth as a decimal.
    """
    if multiple <= 0 or terminal_metric <= 0 or terminal_ufcf <= 0:
        raise ValueError("multiple, terminal_metric and terminal_ufcf must be positive")
    if not 0 < discount_rate < 1:
        raise ValueError("discount_rate must be a decimal in (0, 1)")
    if timing not in ("end", "mid"):
        raise ValueError("timing must be 'end' or 'mid'")
    tv = multiple * terminal_metric
    cash = terminal_ufcf * (math.sqrt(1.0 + discount_rate) if timing == "mid" else 1.0)
    return (tv * discount_rate - cash) / (tv + cash)


def growth_matches(disclosed: float, recomputed: float, rounding: float = 0.15) -> bool:
    """Do a disclosed and a recomputed growth rate (in percent) agree up to rounding?

    ``rounding`` is the tolerance in percentage points: disclosed multiples, rates and
    cash flows are rounded, so exact equality is not expected.
    """
    return abs(disclosed - recomputed) <= rounding


def add_flags(facts: pd.DataFrame) -> pd.DataFrame:
    """Return ``facts`` with a ``flags`` column of ``;``-joined soft-check names."""
    out = facts.copy()
    flags: list[set[str]] = [set() for _ in range(len(out))]
    pos = {idx: i for i, idx in enumerate(out.index)}
    for field, (lo, hi) in TYPICAL_BOUNDS_PCT.items():
        sel = out["field"] == field
        bad = sel & ((out["low"] < lo) | (out["high"] > hi))
        name = {"discount_rate": "rate", "terminal_growth": "growth", "implied_growth": "growth"}[
            field
        ]
        for idx in out.index[bad]:
            flags[pos[idx]].add(f"{name}_atypical")
    sel = out["field"].isin(["exit_multiple", "implied_multiple"])
    bad = sel & (
        (out["low"] < TYPICAL_MULTIPLE_BOUNDS[0]) | (out["high"] > TYPICAL_MULTIPLE_BOUNDS[1])
    )
    for idx in out.index[bad]:
        flags[pos[idx]].add("multiple_atypical")
    sel = out["field"] == "implied_growth"
    for idx in out.index[sel & (out["high"] > NOMINAL_GROWTH_BOUND_PCT)]:
        flags[pos[idx]].add("implied_growth_high")
    _flag_growth_vs_discount(out, flags, pos)
    out["flags"] = [";".join(sorted(f)) for f in flags]
    return out


def _flag_growth_vs_discount(
    out: pd.DataFrame, flags: Sequence[set[str]], pos: dict[object, int]
) -> None:
    growth = out[out["field"] == "terminal_growth"]
    rates = out[out["field"] == "discount_rate"]
    if growth.empty or rates.empty:
        return
    for acc, g_rows in growth.groupby("accession"):
        r_rows = rates[rates["accession"] == acc]
        for gi, g in g_rows.iterrows():
            near = r_rows[
                (r_rows["advisor"] == g["advisor"])
                & (r_rows["subject"] == g["subject"])
                & ((r_rows["span_start"] - g["span_start"]).abs() <= PAIRING_DISTANCE)
            ]
            for ri, r in near.iterrows():
                if g["high"] >= r["low"]:
                    flags[pos[gi]].add("growth_ge_discount")
                    flags[pos[ri]].add("growth_ge_discount")
