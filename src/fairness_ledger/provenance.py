"""Provenance check: a value is admitted only if its span proves it.

For a fact with span ``[start, end)`` in document text ``T`` the check requires

* the span lies inside ``T``, the snippet window contains the span and, whitespace
  collapsed, equals the stored snippet (at most 300 characters);
* both stated endpoints appear verbatim inside ``T[start:end]``;
* re-reading ``T[start:end]`` alone reproduces the same field and values (the valuation
  grammar for percentages and multiples, the dollar parser for per-share values, fees
  and the offer price).

The same function validates a released dataset against the cached filings
(``fairness-ledger verify``), so a value cannot drift away from its source.
"""

from __future__ import annotations

import math
from typing import Protocol

from .deal_terms import reparse_money
from .grammar import find_candidates

MONEY_FIELDS = {
    "value_per_share": "usd",
    "offer_price": "usd",
    "fee_total": "usd_million",
    "fee_opinion": "usd_million",
    "fee_contingent": "usd_million",
}


class SpanFact(Protocol):
    field: str
    low: float
    high: float
    raw_low: str
    raw_high: str
    span_start: int
    span_end: int
    snippet_start: int
    snippet_end: int
    snippet: str


def _collapse(text: str) -> str:
    return " ".join(text.split())


def verify_span(fact: SpanFact, text: str) -> list[str]:
    """Return the list of problems (empty if the fact is proven by its span)."""
    problems: list[str] = []
    n = len(text)
    if not (0 <= fact.span_start < fact.span_end <= n):
        return ["span_out_of_range"]
    if not (0 <= fact.snippet_start <= fact.span_start and fact.span_end <= fact.snippet_end <= n):
        problems.append("snippet_does_not_contain_span")
    elif _collapse(text[fact.snippet_start : fact.snippet_end]) != fact.snippet:
        problems.append("snippet_not_verbatim")
    if len(fact.snippet) > 300:
        problems.append("snippet_too_long")
    span_text = text[fact.span_start : fact.span_end]
    if fact.raw_low not in span_text or fact.raw_high not in span_text:
        problems.append("value_text_not_in_span")
    unit = MONEY_FIELDS.get(fact.field)
    if unit is not None:
        values = reparse_money(span_text, unit)
        wanted = [fact.low] if fact.low == fact.high else [fact.low, fact.high]
        if len(values) < len(wanted) or not all(
            math.isclose(v, w, rel_tol=1e-9, abs_tol=1e-9)
            for v, w in zip(values[: len(wanted)], wanted, strict=False)
        ):
            problems.append("value_not_reproduced_from_span")
    else:
        reparsed = find_candidates(span_text)
        if not any(
            c.field == fact.field and c.low == fact.low and c.high == fact.high for c in reparsed
        ):
            problems.append("value_not_reproduced_from_span")
    return problems
