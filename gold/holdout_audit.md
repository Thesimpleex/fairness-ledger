# Holdout audit

The extractor was frozen before the holdout was scored, and the holdout was scored once
(`python scripts/evaluate_gold.py holdout`). The scores in the README are those first-run
numbers. They were not changed afterwards; the extractor code was not edited in response
to the holdout.

This note classifies the 27 extracted facts that matched no gold label (false positives
in the first run). The classification is the author's post-hoc reading of each sentence,
made by the same person who wrote the extractor, so it is an explanation of the errors
and not a corrected score.

| class | count | what it covers |
|---|---|---|
| extractor error | 6 | the fact is wrong for its field |
| valid value, absent from the gold labels | 13 | the sentence supports the value; the gold labels omit it, or the field definition (value per share of a DCF-type analysis) was applied more narrowly in the gold set |
| definitional ambiguity | 8 | reasonable people could label it either way |

## Extractor errors (6)

- `0001193125-12-212341` Morgan Stanley, `fee_contingent` $0.25 million: the sentence is a
  non-refundable fee paid on engagement, not a contingent fee.
- `0001193125-15-025716` `offer_price` $66.00: the cash amount of a cash-or-stock election,
  not a single per-share price.
- `0000905729-16-000606` `offer_price` $1.61: the cash component of mixed consideration in a
  table.
- `0001193125-18-347556` KBW, `fee_total` $0.145 million: a fee for an earlier, different
  merger of the same client.
- `0001104659-21-086839` Moelis, `fee_opinion` $5.0 million: a fee for completing work, not
  for the opinion.
- `0001047469-21-001235` `offer_price` $10.00: a footnote about a per-share price in a
  sources-and-uses table.

## Valid value, absent from the gold labels (13)

- `value_per_share` (12): Goldman Sachs ranges of illustrative present values per share for
  Cooper (one) and Eaton (one); Goldman Sachs, BMO and Citi ranges for Newmont and Goldcorp
  (seven, among them net-asset-value and future-price analyses); Goldman Sachs equity values
  per share in `0001140361-25-029741` (two); Goldman Sachs' dividend-discount range for
  Platinum (one).
- `offer_price` $38.75 in `0001193125-19-026360`: the cash price of the merger; the gold
  set carries no `offer_price` label for this filing.

## Definitional ambiguity (8)

- Platinum: two `value_per_share` ranges stated "with respect to the RenaissanceRe offer"
  (pro forma value of the offer rather than of the target).
- `fee_total` of $3.5 million (FairPoint) and $1.5 million (Nutrisystem), described as
  payable on signing the engagement letter, and Morgan Stanley's $3 million announcement
  fee: a total fee or an advance on it.
- `fee_contingent` of $2.5 million (BofA, "discretionary fee") and $12.5 million (Morgan
  Stanley, transaction fee against which an announcement fee is credited).
- `discount_rate` 19.58% in `0001193125-25-155514`: the cost of equity inside a stated WACC of
  18.0%. It is the only key-field false positive of the holdout.

## Reading

All three precision-target fields are at or near target on the holdout (see the README
for the intervals). The weak spots are the auxiliary fields: fees (sentences that mix
several amounts and payment events) and per-share values (several analyses report per-share
ranges and the field definition is narrow). These fields are released with `tier =
"auxiliary"` for this reason.
