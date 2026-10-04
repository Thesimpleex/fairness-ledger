# Annotation guide for the gold set

Labels are written from the filing text without looking at extractor output. The reading
pack of `scripts/annotation_pack.py` lists headings and every sentence with a valuation cue
and a number; it never calls the extractor. Labels live in `gold/labels/<accession>.txt`,
one fact per line:

```
field|low|high|advisor|analysis|subject|metric|valuation_date|quote[#k]
```

`quote` is a verbatim phrase of the filing (whitespace-insensitive) that contains the
value; `#k` picks the k-th occurrence if the phrase repeats. `scripts/check_gold.py`
verifies that every quote resolves to a span of its filing.

## Fields

- `discount_rate`: a discount rate, weighted average cost of capital or cost of equity used
  to discount cash flows in a DCF or dividend discount analysis. A range is one fact. An
  enumeration of several rates is one fact per rate.
- `terminal_growth`: a perpetuity (terminal) growth rate used to compute a terminal value.
- `exit_multiple`: a multiple applied to a terminal-year metric.
- `implied_growth`, `implied_multiple`: the growth rate or multiple that the other terminal
  method implies, as stated by the advisor.
- `value_per_share`: the implied value-per-share range that a DCF-type analysis produces.
- `offer_price`: the all-cash per-share price (not stated for mixed or election deals).
- `fee_total`, `fee_opinion`, `fee_contingent`: advisor compensation in millions of dollars,
  for the advisor's engagement on this transaction.

Not labelled: discount rates, growth rates or multiples that appear in tables, in lists of
inputs of other analyses (comparable companies, precedent transactions, sum of the parts),
in descriptions of the Federal Reserve discount rate, or in a different sense (for example
interest rates of financing).

## Attributes

- `advisor`: the house that performed the analysis; `-` if the filing does not say, `joint`
  if two houses are named together. Joint labels are excluded from advisor accuracy.
- `analysis`: `dcf`, `ddm`, `pvfp` (present value of future share price), `nav`, `other`.
- `subject`: `T` target standalone, `A` acquirer standalone, `P` pro forma combined, `U` not
  stated.
- `metric`: the terminal metric of an exit multiple (EBITDA, P/E, book value, ...).
- `valuation_date`: ISO date the cash flows are discounted to, if stated in the sentence.

## Procedure

1. Development filings were annotated while the grammar was written, so their scores are
   diagnostic.
2. Holdout filings were selected after the grammar was frozen, from filings not in the
   development pool, annotated from their text and hashed (`scripts/freeze_holdout.py`)
   before the first evaluation.
3. Annotation was done once by the author of the extractor, so the labels share the author's
   reading of the field definitions; no second annotator checked them, and no agreement
   statistic exists. The holdout audit (`gold/holdout_audit.md`) shows where definitions and
   labels diverge.
