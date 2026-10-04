# Methodology

This note describes how a filing becomes rows of the dataset, what is checked on the way,
and how the evaluation is set up. All numbers (counts, precision, recall, analysis
results) are in `docs/results.json`; none are repeated here.

## 1. Discovery

EDGAR full-text search is queried for form DEFM14A, one calendar year at a time, with
seven phrase queries whose union is taken (the search matches exact tokens, so
"discounted cash flow" does not match "discounted cash flows"):

`"discounted cash flow"`, `"discounted cash flows"`, `"discount rate"`, `"discount rates"`,
`"discount rates ranging from"`, `"weighted average cost of capital"`, `"perpetuity growth"`.

Search is capped at 1,000 hits per request window. A window whose reported total reaches
the cap is bisected by date until every window is below it, and the number of hits
collected must equal the reported total, otherwise discovery stops with an error. Only
main documents of the form type are kept. Every response is cached under `data/raw`
(gzip, gitignored) and requests are limited to eight per second with the declared
`SEC_USER_AGENT`.

Filings are resolved to deals: filings of the same CIK within 270 days of each other form
one deal (a revised definitive proxy repeats the same analyses). The deal id is the first
accession number. Facts identical in field, values, advisor, analysis, subject, metric,
valuation date and snippet are stored once per deal, with `n_filings` counting the filings
that contain them. CIK, SIC and tickers come from the SEC submissions API; the company
name is the name at the time of the filing.

## 2. Document model

The HTML is converted to plain text by a small parser that keeps one block per paragraph
or table row (cells joined by ` | `), collapses whitespace and records, for every output
character, the offset of its source character in the original HTML (`Document.html_offsets`).
Charrefs in the Windows-1252 range are mapped to the characters a browser shows. Lines
carry two flags used downstream, bold and linked. Page numbers, "Table of Contents"
links and similar page-break artifacts that interrupt a sentence are removed so that
sentences stay contiguous.

The SHA-256 of the normalised text is stored with every fact. Spans refer to this text;
`fairness-ledger verify` rebuilds it from the cached filing and checks the hash first.

### Sections

Headings are lines that look like "Opinion of X", "Background of the Merger" or the
heading that closes a section. Table-of-contents entries (page numbers, hyperlinks,
"page N" references) are rejected. The text between headings becomes an opinion,
background or other segment. A value is admitted if it lies in an opinion or background
segment, or if a house name occurs within 3,500 characters before it and a valuation
marker (discounted cash flow, perpetuity, terminal, dividend discount, ...) within 6,000
characters before it. Values that satisfy neither are recorded as rejections
(`outside_analysis_context`).

### Analysis type, advisor, subject

- Analysis type: the nearest preceding marker within 12,000 characters and within the
  segment, heading-like markers ranking above markers inside running sentences. Classes are
  `dcf`, `ddm`, `pv_future_price`, `nav`, `other`. A discount rate, growth rate or multiple
  with no marker is rejected (`no_analysis_marker`).
- Advisor: the nearest preceding house mention that is followed by an analysis verb
  (performed, applied, calculated, ...), otherwise the advisor of the opinion segment,
  otherwise `unattributed`. The lexicon (`advisors.py`) lists about a hundred houses and
  excludes publication titles that contain a house name (for example valuation handbooks).
- Subject: target, acquirer, pro forma or unspecified. The role of the filing company is
  inferred from the evidence of share issuance in the merger consideration; the entity
  nearest to the value in its sentence (company aliases, counterparty names, "pro forma",
  "combined company") decides.

## 3. Extraction grammar

Three rule families (`grammar.py`) find discount rates, terminal growth and exit
multiples. Each combines a cue ("discount rates", "weighted average cost of capital",
"perpetuity growth rates", "terminal multiples", ...), a short lead, and a value
expression: a range ("7.5% to 9.5%", "7.5% - 9.5%", "between ... and ...", "from ... through
..."), a point, a parenthetical negative "(1.5)%", a worded negative ("negative 1.5%") or an
enumeration ("1.0%, 2.0% and 3.0%": every member becomes its own candidate, with a span
that runs from the head of the enumeration). A reverse rule handles "10.0% to 12.0% discount
rates". Candidates followed by another cue (as in "a discount rate and 3.5% perpetuity growth")
are guarded against, and table rows (a point value followed by a pipe or a second percentage)
are dropped. Implied perpetuity growth rates and implied terminal multiples are recognised
from the words "implied", "implying", "which implied" in front of the range and stored as
separate fields.

`deal_terms.py` adds per-share value ranges ("implied value per share of $48.00 to
$64.00"), the cash offer price (a consistent cash amount that must be stated at least twice
and not be followed by further share consideration), the valuation date (dates introduced by
"as of", "to" or "at" in a sentence about discounting) and advisor fees (total, opinion
portion, contingent portion, classified from the wording of each amount; termination fees,
expense reimbursements, financing roles and earlier engagements are excluded).

Hard bounds apply to every field (for example discount rate 0 to 60%, exit multiple 0.05x
to 150x); a violation is a rejection, never a silently corrected value.

## 4. Provenance

Every candidate becomes a fact only if `verify_span` passes:

1. the span `[start, end)` lies inside the text;
2. the stored snippet is the whitespace-collapsed text of its window, at most 300
   characters, and the window contains the span;
3. both endpoints as written (`raw_low`, `raw_high`) occur verbatim inside the span;
4. re-reading the span alone reproduces the same field and values.

The same function validates a released dataset against the cached filings. Candidates that
fail are listed in the rejection counts (`extraction_summary.json`), never in the data.

## 5. Consistency algebra

Hard checks (`low <= high`, units, bounds) are applied at extraction. Soft flags are added
to admitted facts and kept in the `flags` column:

- `rate_atypical`: discount rate outside 3 to 25%; `growth_atypical`: growth outside -3 to
  6% (implied growth: -5 to 8%); `multiple_atypical`: multiple outside 1x to 40x;
- `growth_ge_discount`: a terminal growth rate at or above the discount rate of the same
  advisor and subject within 3,000 characters (the Gordon value is then undefined);
- `implied_growth_high`: a disclosed implied growth above the nominal-growth bound of 4%.

### Implied perpetuity growth

An exit multiple $m$ of a terminal metric $M_T$ values the business at the end of year
$T$ at $TV = m\,M_T$. Equating this with the Gordon value of the cash flow of year $T+1$
with end-of-year discounting,

$$
TV = \frac{UFCF_T\,(1+g)}{r-g}
\qquad\Longrightarrow\qquad
g = \frac{TV\,r - UFCF_T}{TV + UFCF_T}.
$$

With mid-year discounting the perpetuity is worth a factor $(1+r)^{1/2}$ more, so
$UFCF_T$ is replaced by $UFCF_T\,(1+r)^{1/2}$ in the formula. The function
`implied_perpetuity_growth` implements both conventions; end-of-year is the default
because it is the form most disclosures state or imply. `growth_matches` compares a
disclosed and a recomputed rate up to a rounding tolerance of 0.15 percentage points.
The recomputation needs terminal-year UFCF and metric, which proxies rarely disclose in
a parseable table; it is therefore tested against closed-form inverses rather than run
over the corpus (see Limitations in the README).

## 6. Gold set and evaluation

Gold labels are written from the filing text only. Annotation uses a reading pack
(`scripts/annotation_pack.py`) that lists heading-like lines and every sentence with a
valuation cue and a number, using the text conversion and generic regular expressions but
not the extractor. A label is `field|low|high|advisor|analysis|subject|metric|date|quote`,
where `quote` is a verbatim phrase of the filing that contains the value (its location in
the text is the label's span). `docs/annotation_guide.md` lists the rules.

The set has 55 filings: 25 development filings (the pool used while the grammar was
written, scored for diagnostics and optimistic by construction) and 30 holdout filings
drawn from filings never opened during development, stratified by five filing-year eras and
by dominant advisor with a fixed seed (`scripts/select_gold.py`). The holdout labels were
hashed (`gold/holdout.sha256`) before the first holdout run; the evaluation refuses to run
if the labels differ from the hash. `gold/holdout_audit.md` classifies the first-run errors.

A predicted fact matches a gold fact if field and both values are equal and the spans
overlap (offer price: values only), one-to-one. Precision is $TP/(TP+FP)$ and recall
$TP/(TP+FN)$, both with Wilson intervals; precision on the three key fields also has a
filing-level bootstrap interval. Attribute accuracy (advisor, analysis, subject, metric,
valuation date) is computed on matched facts.

## 7. Analyses

All three are descriptive.

**Bank disagreement.** The unit is (deal, advisor). Each advisor's value is the median
midpoint of its target-standalone DCF facts; the statistic is the absolute difference of
two advisors' values within a deal, one row per advisor pair. The interval for the median
resamples deals.

**Implied growth.** An analysis unit is a (filing, advisor, subject) with an exit-multiple
DCF. It is covered if an implied-growth fact of the same filing and advisor lies within
6,000 characters of one of its exit multiples. The share above a bound is computed over
covered units only and reported next to the coverage. The reference bound of 4% is
a nominal long-run growth ceiling (long-run real growth of about 2% plus 2% inflation);
3, 5 and 6% are reported for sensitivity.

**Market practice.** One observation per non-financial deal: the median midpoint of its
target-standalone DCF discount rates, and of its terminal growth rates, matched to the
last DGS10 observation on or before the first filing date. The pass-through slope is
the OLS slope of the discount rate on the yield, pooled and with SIC two-digit fixed
effects. The confidence interval is a year-block bootstrap: filing years are resampled
with replacement and the slope is recomputed on the resampled deals. The estimate is
descriptive: deal selection, composition and the timing of disclosure all vary with
rates.
