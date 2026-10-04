"""Gold-set labels: format, span location, freezing and evaluation.

A label file ``gold/labels/<accession>.txt`` has one fact per line::

    field|low|high|advisor|analysis|subject|metric|valuation_date|quote[#k]

``quote`` is a verbatim phrase of the filing text (whitespace-insensitive) that contains
the value; ``#k`` selects the k-th occurrence when the phrase repeats. The span of a
label is the location of its quote, so every label is anchored in the document.
Labels are written from the filing text alone, never from extractor output.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

ANALYSIS_CODES = {
    "dcf": "dcf",
    "ddm": "ddm",
    "pvfp": "pv_future_price",
    "nav": "nav",
    "other": "other",
    "-": "",
}
SUBJECT_CODES = {"T": "target", "A": "acquirer", "P": "pro_forma", "U": "unspecified", "-": ""}
FIELDS = (
    "discount_rate",
    "terminal_growth",
    "exit_multiple",
    "implied_growth",
    "implied_multiple",
    "value_per_share",
    "offer_price",
    "fee_total",
    "fee_opinion",
    "fee_contingent",
)
JOINT = "joint"


@dataclass(frozen=True)
class GoldFact:
    """One annotated fact with the span of its quote."""

    accession: str
    field: str
    low: float
    high: float
    advisor: str
    analysis: str
    subject: str
    metric: str
    valuation_date: str
    quote: str
    span_start: int
    span_end: int


def _quote_regex(quote: str) -> re.Pattern[str]:
    words = quote.split()
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def locate_quote(text: str, quote: str, occurrence: int | None) -> tuple[int, int]:
    """Span of ``quote`` in ``text``; ``occurrence`` is 1-based, ``None`` requires uniqueness."""
    matches = list(_quote_regex(quote).finditer(text))
    if not matches:
        raise ValueError(f"quote not found: {quote!r}")
    if occurrence is None:
        if len(matches) > 1:
            raise ValueError(f"quote is ambiguous ({len(matches)} matches): {quote!r}")
        match = matches[0]
    else:
        if occurrence > len(matches):
            raise ValueError(f"only {len(matches)} matches for {quote!r}, wanted #{occurrence}")
        match = matches[occurrence - 1]
    return match.start(), match.end()


def parse_labels(accession: str, text: str, label_text: str) -> list[GoldFact]:
    """Parse one label file against its document text."""
    facts: list[GoldFact] = []
    for number, raw in enumerate(label_text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 8)
        if len(parts) != 9:
            raise ValueError(f"{accession}:{number}: expected 9 fields, got {len(parts)}")
        field, low, high, advisor, analysis, subject, metric, vdate, quote = parts
        if field not in FIELDS:
            raise ValueError(f"{accession}:{number}: unknown field {field!r}")
        occurrence = None
        match = re.search(r"#(\d+)$", quote)
        if match:
            occurrence = int(match.group(1))
            quote = quote[: match.start()]
        try:
            start, end = locate_quote(text, quote, occurrence)
        except ValueError as exc:
            if field == "offer_price" and "ambiguous" in str(exc):
                start, end = locate_quote(text, quote, 1)
            else:
                raise ValueError(f"{accession}:{number}: {exc}") from exc
        facts.append(
            GoldFact(
                accession=accession,
                field=field,
                low=float(low),
                high=float(high),
                advisor="" if advisor == "-" else advisor,
                analysis=ANALYSIS_CODES[analysis],
                subject=SUBJECT_CODES[subject],
                metric="" if metric == "-" else metric,
                valuation_date="" if vdate == "-" else vdate,
                quote=quote,
                span_start=start,
                span_end=end,
            )
        )
    return facts


def labels_digest(paths: Iterable[Path]) -> str:
    """SHA-256 over the sorted (name, bytes) pairs of the label files."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# -- evaluation -----------------------------------------------------------------


@dataclass(frozen=True)
class PredictedFact:
    """Minimal view of an extracted fact needed for matching."""

    accession: str
    field: str
    low: float
    high: float
    advisor: str
    analysis: str
    subject: str
    metric: str
    valuation_date: str
    span_start: int
    span_end: int


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)


def match_facts(
    gold: Sequence[GoldFact], predicted: Sequence[PredictedFact]
) -> tuple[list[tuple[GoldFact, PredictedFact]], list[GoldFact], list[PredictedFact]]:
    """One-to-one matching on field, values and span overlap (offer price: values only)."""
    pairs: list[tuple[GoldFact, PredictedFact]] = []
    used: set[int] = set()
    missed: list[GoldFact] = []
    for g in gold:
        found = None
        for i, p in enumerate(predicted):
            if i in used or p.accession != g.accession or p.field != g.field:
                continue
            if not (_same(p.low, g.low) and _same(p.high, g.high)):
                continue
            if g.field == "offer_price" or (
                p.span_start < g.span_end and g.span_start < p.span_end
            ):
                found = i
                break
        if found is None:
            missed.append(g)
        else:
            used.add(found)
            pairs.append((g, predicted[found]))
    extra = [p for i, p in enumerate(predicted) if i not in used]
    return pairs, missed, extra


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (NaN bounds for ``total == 0``)."""
    if total == 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Score:
    """Precision and recall with Wilson intervals."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else math.nan

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else math.nan

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or p + r == 0:
            return math.nan
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict[str, float | int]:
        p_lo, p_hi = wilson_interval(self.tp, self.tp + self.fp)
        r_lo, r_hi = wilson_interval(self.tp, self.tp + self.fn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "precision_lo": p_lo,
            "precision_hi": p_hi,
            "recall": self.recall,
            "recall_lo": r_lo,
            "recall_hi": r_hi,
            "f1": self.f1,
        }


def evaluate(
    gold: Sequence[GoldFact],
    predicted: Sequence[PredictedFact],
    group_of: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    """Field-level scores overall and by era/advisor groups.

    Parameters
    ----------
    gold, predicted
        Facts of the evaluated filings only.
    group_of
        Mapping ``{"era": {accession: label}, "advisor": {accession: label}}`` used for
        the stratified tables.
    """
    pairs, missed, extra = match_facts(gold, predicted)
    result: dict[str, object] = {"n_gold": len(gold), "n_predicted": len(predicted)}

    def score(sel: set[str] | None, keep_acc: set[str] | None = None) -> Score:
        def ok_field(f: str) -> bool:
            return sel is None or f in sel

        def ok_acc(a: str) -> bool:
            return keep_acc is None or a in keep_acc

        tp = sum(1 for g, _ in pairs if ok_field(g.field) and ok_acc(g.accession))
        fn = sum(1 for g in missed if ok_field(g.field) and ok_acc(g.accession))
        fp = sum(1 for p in extra if ok_field(p.field) and ok_acc(p.accession))
        return Score(tp, fp, fn)

    fields_present = sorted({g.field for g in gold} | {p.field for p in predicted})
    result["by_field"] = {f: score({f}).to_dict() for f in fields_present}
    result["overall"] = score(None).to_dict()
    key_fields = {"discount_rate", "terminal_growth", "exit_multiple"}
    result["key_fields"] = score(key_fields).to_dict()
    strata: dict[str, dict[str, dict]] = {}
    for name, mapping in (group_of or {}).items():
        groups: dict[str, set[str]] = defaultdict(set)
        for acc, label in mapping.items():
            groups[label].add(acc)
        strata[name] = {
            label: score(key_fields, accs).to_dict() for label, accs in sorted(groups.items())
        }
    result["by_group"] = strata
    attributes: dict[str, dict[str, float | int]] = {}
    for attr in ("advisor", "analysis", "subject", "metric", "valuation_date"):
        correct = total = 0
        for g, p in pairs:
            want = getattr(g, attr)
            if not want or (attr == "advisor" and want == JOINT):
                continue
            total += 1
            correct += int(getattr(p, attr) == want)
        lo, hi = wilson_interval(correct, total)
        attributes[attr] = {
            "n": total,
            "correct": correct,
            "accuracy": correct / total if total else math.nan,
            "lo": lo,
            "hi": hi,
        }
    result["attributes"] = attributes
    result["missed"] = [g.__dict__ for g in missed]
    result["extra"] = [p.__dict__ for p in extra]
    return result


def bootstrap_precision(
    pairs_by_filing: dict[str, tuple[int, int]], draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Filing-level bootstrap 95% interval of precision from ``{acc: (tp, fp)}``."""
    rng = np.random.default_rng(seed)
    keys = sorted(pairs_by_filing)
    tp = np.array([pairs_by_filing[k][0] for k in keys], dtype=float)
    fp = np.array([pairs_by_filing[k][1] for k in keys], dtype=float)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    t, f = tp[idx].sum(axis=1), fp[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = t / (t + f)
    prec = prec[~np.isnan(prec)]
    return float(np.quantile(prec, 0.025)), float(np.quantile(prec, 0.975))


def dump_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")
