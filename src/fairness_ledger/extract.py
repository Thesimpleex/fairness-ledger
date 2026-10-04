"""Turn a filing into provenance-linked facts.

``extract_document`` runs the grammars over the text, assigns each candidate to a section,
advisor, analysis type and subject, and admits it only if

1. valuation parameters lie in an opinion or background section (or have a local
   advisor-plus-analysis context) and per-share ranges belong to a DCF-type analysis,
2. its numbers are inside hard plausibility bounds and ordered ``low <= high``,
3. its span provably contains the stated numbers (:mod:`fairness_ledger.provenance`).

Everything else is returned in a rejection list with the reason, never dropped silently.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .advisors import UNATTRIBUTED, MentionIndex
from .deal_terms import (
    Term,
    find_fee_terms,
    find_offer_price,
    find_per_share_ranges,
    find_valuation_date,
    sentence_bounds,
)
from .document import Document
from .grammar import Candidate, find_candidates
from .sections import SegmentIndex, segment_document

SNIPPET_MAX = 300
VALUATION_FIELDS = (
    "discount_rate",
    "terminal_growth",
    "exit_multiple",
    "implied_growth",
    "implied_multiple",
)
FEE_FIELDS = ("fee_total", "fee_opinion", "fee_contingent")
DCF_LIKE = ("dcf", "ddm", "pv_future_price", "nav")

HARD_BOUNDS: dict[str, tuple[float, float]] = {
    "discount_rate": (0.0, 60.0),
    "terminal_growth": (-20.0, 15.0),
    "implied_growth": (-60.0, 60.0),
    "exit_multiple": (0.05, 150.0),
    "implied_multiple": (0.05, 150.0),
    "value_per_share": (0.0, 20000.0),
    "offer_price": (0.5, 5000.0),
    "fee_total": (0.01, 500.0),
    "fee_opinion": (0.01, 500.0),
    "fee_contingent": (0.01, 500.0),
}

_ANALYSIS_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ddm",
        re.compile(
            r"dividend\s+discount|\bDDM\b|discounted\s+dividend|"
            r"net\s+present\s+value\s+(?:analys[ie]s|per\s+share)",
            re.I,
        ),
    ),
    (
        "pv_future_price",
        re.compile(
            r"present\s+values?\s+of\s+(?:an?\s+|the\s+)?(?:illustrative\s+)?(?:projected\s+)?"
            r"future\s+(?:stock|share|equity)|future\s+(?:stock|share)\s+price|"
            r"discount(?:ed|ing)\s+(?:the\s+)?(?:analyst\s+)?(?:stock\s+|share\s+)?price\s+targets?|"
            r"price\s+targets?[^.]{0,80}?discounted|discounted\s+equity\s+value",
            re.I,
        ),
    ),
    ("nav", re.compile(r"net\s+asset\s+value|\bNAV\b", re.I)),
    (
        "dcf",
        re.compile(
            r"discounted\s+cash\s+flow|\bDCF\b|sum[-\s]of[-\s]the[-\s]parts|"
            r"unlevered\s+free\s+cash\s+flow",
            re.I,
        ),
    ),
    (
        "other",
        re.compile(
            r"selected\s+(?:[\w-]+\s+){0,3}compan(?:y|ies)|comparable\s+compan|public\s+trading|"
            r"precedent\s+transactions?|selected\s+transactions?|premiums?\s+paid|"
            r"contribution\s+analysis|leveraged\s+buy|\bLBO\b|accretion|"
            r"historical\s+(?:stock|share)\s+(?:price|trading)|trading\s+multiples|"
            r"returns\s+analysis",
            re.I,
        ),
    ),
]

_PRO_FORMA = re.compile(r"pro\s+forma|combined\s+(?:company|entity)|\bcombined\b", re.I)
_ACQUIRER = re.compile(
    r"\b(?:Parent|Buyer|Acquiror|Acquirer|Purchaser|Merger\s+Sub|Acquisition\s+Corp)\b"
)
_COMPANY = re.compile(r"\b(?:the\s+)?Company\b['’]?")
_WS = re.compile(r"\s+")
_GENERIC_NAME_WORDS = {
    "american",
    "first",
    "united",
    "national",
    "general",
    "global",
    "international",
    "new",
    "the",
    "bank",
    "financial",
}
_VERB = re.compile(
    r"\s*(?:\(|and\b|,)?[^.]{0,80}?\b(?:perform|conduct|calculat|appl|deriv|us(?:e|ed|ing)|"
    r"assum|utiliz|select|discount|review|estimat|valu|prepar|also|then|analy|noted|"
    r"considered|determin|compar)",
    re.I,
)
_VALUATION_MARKER = re.compile(
    "|".join(p.pattern for name, p in _ANALYSIS_MARKERS if name != "other"), re.I
)
_TABLE_FOLLOW = re.compile(r"\s*\||\s+\d{1,3}(?:\.\d+)?\s*%")


@dataclass(frozen=True)
class Fact:
    """One extracted value with its provenance."""

    accession: str
    field: str
    low: float
    high: float
    unit: str
    raw_low: str
    raw_high: str
    advisor: str
    subject: str
    analysis: str
    region: str
    metric: str
    valuation_date: str
    span_start: int
    span_end: int
    snippet_start: int
    snippet_end: int
    snippet: str
    doc_sha256: str
    flags: tuple[str, ...] = ()

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["flags"] = ";".join(self.flags)
        return row


@dataclass(frozen=True)
class Rejection:
    """A candidate that was found but not admitted, with the reason."""

    accession: str
    field: str
    start: int
    end: int
    low: float
    high: float
    reason: str


@dataclass
class Extraction:
    """Result of running the extractor on one document."""

    accession: str
    doc_sha256: str
    facts: list[Fact] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)


_LEGAL = {
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "plc",
    "llc",
    "lp",
    "l.p",
    "nv",
    "n.v",
    "sa",
    "s.a",
    "ag",
    "se",
    "de",
    "the",
    "trust",
    "incorporated",
    "lllp",
}


def target_aliases(company: str) -> list[str]:
    """Names under which a company is likely to be referred to in its own proxy.

    Legal suffixes are stripped, then trailing words are dropped one at a time
    (``Evoqua Water Technologies`` -> ``Evoqua Water`` -> ``Evoqua``). Single-word
    aliases are returned capitalised so that they only match proper names.
    """
    raw = company.replace(",", " ").replace(".", ". ").split()
    tokens = [t for t in (w.strip() for w in raw) if t]
    if len(tokens) == 1:
        m = re.match(r"(?i)^(.{3,}?)(ltd|inc|corp|plc|llc)$", tokens[0])
        if m:
            tokens = [m.group(1)]
    while tokens and tokens[-1].lower().strip(".") in _LEGAL:
        tokens.pop()
    aliases: list[str] = []
    while tokens:
        name = " ".join(tokens)
        if len(tokens) == 1:
            if name.isupper() and len(name) <= 4:
                single = name
            else:
                single = name.title() if name.isupper() else name
            if len(single) >= 3 and single.lower() not in _GENERIC_NAME_WORDS:
                aliases.append(single)
            break
        aliases.append(name)
        tokens.pop()
    return aliases


def collapse(text: str) -> str:
    """Collapse runs of whitespace (including line breaks) to single spaces."""
    return _WS.sub(" ", text).strip()


def classify_analysis(text: str, position: int, floor: int) -> str:
    """Analysis type from the nearest preceding marker within the segment.

    Markers at the start of a line or in a short line (headings, run-in headings) rank
    above markers inside running sentences; weak ``other`` markers are ignored because
    comparable-company language is routinely mentioned inside DCF paragraphs.
    """
    window_start = max(floor, position - 12000)
    strong: tuple[int, str] | None = None
    weak: tuple[int, str] | None = None
    for name, pattern in _ANALYSIS_MARKERS:
        for m in pattern.finditer(text, window_start, position):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line_end = len(text) if line_end == -1 else line_end
            is_strong = line_end - line_start <= 160 or m.start() - line_start <= 40
            if is_strong and (strong is None or m.start() > strong[0]):
                strong = (m.start(), name)
            elif not is_strong and name != "other" and (weak is None or m.start() > weak[0]):
                weak = (m.start(), name)
    if strong is not None:
        return strong[1]
    if weak is not None:
        return weak[1]
    return "unspecified"


_COMMON_NAME = re.compile(
    r"((?:[A-Z][\w&.'-]*)(?:\s+[A-Z][\w&.'-]*){0,3})(?:['\u2019]s)?\s+(?:Class\s+[A-Z]\s+)?"
    r"(?i:common\s+(?:stock|shares)|ordinary\s+shares|shares|shareholders|stockholders)\b"
)
_NAME_STOP = {
    "Company",
    "Parent",
    "Merger",
    "Sub",
    "The",
    "Series",
    "Surviving",
    "Holders",
    "Class",
    "Each",
    "Our",
    "Such",
    "New",
    "No",
    "Any",
    "Of",
    "In",
    "For",
    "Common",
    "Preferred",
    "Treasury",
    "Acquiror",
    "Acquirer",
    "Buyer",
    "Purchaser",
}


def infer_filer_role(text: str, filer_aliases: list[str]) -> str:
    """``acquirer`` if the filer's own stock is the one being issued, else ``target``.

    Counts statements in the first 300,000 characters that holders of the filer's shares
    receive consideration (target evidence) against share-issuance proposals for the
    filer's stock (acquirer evidence).
    """
    if not filer_aliases:
        return "target"
    alias = "|".join(re.escape(a) for a in filer_aliases[:3])
    head = text[:300_000]
    target = re.compile(
        rf"(?:each|every)\s+(?:outstanding\s+)?share\s+of\s+(?:the\s+)?(?i:{alias})"
        rf"(?:['\u2019]s)?\s+(?:Class\s+[A-Z]\s+)?(?i:common|ordinary)[^.]{{0,300}}?"
        rf"(?i:right\s+to\s+receive|entitled\s+to\s+receive|converted\s+into)"
    )
    acquirer = re.compile(
        rf"issuance\s+of\s+(?:up\s+to\s+)?[\d,.]+\s+(?:million\s+)?(?:additional\s+)?"
        rf"(?:shares|ordinary\s+shares)\s+of\s+(?:the\s+)?(?i:{alias})\b|"
        rf"(?i:{alias})\s+(?:common\s+stock\s+|share\s+)?issuance\s+proposal"
    )
    return "acquirer" if len(acquirer.findall(head)) > len(target.findall(head)) else "target"


def counterparty_aliases(text: str, filer_aliases: list[str]) -> list[str]:
    """Most frequent other company named before ``common stock`` in the first 400,000 characters."""
    counts: dict[str, int] = {}
    for m in _COMMON_NAME.finditer(text, 0, 400_000):
        words = [w for w in m.group(1).split() if w not in _NAME_STOP]
        if not words:
            continue
        name = " ".join(words)
        if any(name.startswith(a) or a.startswith(name) for a in filer_aliases if a):
            continue
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [n for n, c in ranked[:2] if c >= 6]


_PRO_FORMA_STRICT = re.compile(r"pro\s+forma|combined\s+(?:company|entity|business)", re.I)


def _alias_regex(alias: str) -> re.Pattern[str]:
    """Word-bounded matcher: multi-word names ignore case, single words keep a capital initial."""
    if " " in alias:
        return re.compile(rf"\b{re.escape(alias)}\b", re.I)
    if alias.isupper():
        return re.compile(rf"\b{re.escape(alias)}\b")
    return re.compile(rf"\b(?-i:{re.escape(alias[0].upper())})(?i:{re.escape(alias[1:])})\b")


def _entity_hits(
    context: str, base: int, filer_aliases: list[str], other_aliases: list[str]
) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for alias in filer_aliases:
        hits += [(base + m.start(), "filer") for m in _alias_regex(alias).finditer(context)]
    hits += [(base + m.start(), "filer") for m in _COMPANY.finditer(context)]
    for alias in other_aliases:
        hits += [(base + m.start(), "counterparty") for m in _alias_regex(alias).finditer(context)]
    hits += [(base + m.start(), "counterparty") for m in _ACQUIRER.finditer(context)]
    return hits


def classify_subject(
    text: str,
    start: int,
    end: int,
    filer_aliases: list[str],
    other_aliases: list[str],
    filer_role: str = "target",
    floor: int = 0,
) -> str:
    """Target, acquirer, pro forma or unspecified.

    ``pro_forma`` if the sentence of the value, the first words of its paragraph or the
    heading line above speak of a pro forma or combined company. Otherwise the entity
    named closest to the value inside its sentence decides (before or after); if the
    sentence names none, the closest earlier mention in the paragraph or heading above
    decides. The filer maps to ``target`` or ``acquirer`` through ``filer_role``.
    """
    s_start, s_end = sentence_bounds(text, start, 700)
    line_start = text.rfind("\n", 0, start) + 1
    prev_start = text.rfind("\n", 0, max(0, line_start - 1)) + 1
    heading = text[prev_start : max(prev_start, line_start - 1)]
    sentence = text[s_start:s_end]
    head_of_paragraph = text[line_start : line_start + 90]
    if (
        _PRO_FORMA_STRICT.search(sentence)
        or _PRO_FORMA_STRICT.search(head_of_paragraph)
        or (len(heading) < 140 and _PRO_FORMA_STRICT.search(heading))
    ):
        return "pro_forma"
    side = ""
    hits = _entity_hits(sentence, s_start, filer_aliases, other_aliases)
    if hits:
        side = min(hits, key=lambda h: abs(h[0] - start))[1]
    else:
        lo = max(floor, start - 2500)
        earlier = _entity_hits(text[lo:start], lo, filer_aliases, other_aliases)
        if earlier:
            side = max(earlier, key=lambda h: h[0])[1]
    if not side:
        return "unspecified"
    if side == "filer":
        return "target" if filer_role == "target" else "acquirer"
    return "acquirer" if filer_role == "target" else "target"


def attribute_advisor(
    text: str, position: int, seg_start: int, seg_advisor: str, index: MentionIndex
) -> str:
    """House that performs the analysis at ``position``.

    The nearest preceding house mention that acts as the subject of the clause (followed
    by an analysis verb) within 3,500 characters and the same segment wins; otherwise
    the advisor of the segment; otherwise ``unattributed``.
    """
    floor = max(seg_start, position - 3500)
    for mention in reversed(index.between(floor, position)):
        tail = text[mention.end : mention.end + 120]
        if _VERB.match(tail):
            return mention.house
    return seg_advisor


_METRICS = (
    ("EBITDA", r"EBITDA|EBITDAR|EBITDAX"),
    ("EBIT", r"\bEBIT\b"),
    ("FCF", r"free\s+cash\s+flow|\bFCF\b|distributable\s+cash"),
    ("P/E", r"earnings|\bEPS\b|price[-\s]to[-\s]earnings|net\s+income|\bP/E\b"),
    ("revenue", r"revenue|sales"),
    ("book value", r"book\s+value|\bNAV\b|net\s+asset"),
    ("FFO", r"\bA?FFO\b|funds\s+from\s+operations"),
)


def _metric_of(text: str, cand: Candidate) -> str:
    if cand.unit != "x":
        return ""
    window = text[max(0, cand.start - 60) : cand.end + 140]
    for label, pattern in _METRICS:
        if re.search(pattern, window, re.I):
            return label
    return ""


def make_snippet(text: str, start: int, end: int) -> tuple[int, int, str]:
    """Verbatim sentence-level snippet around ``[start, end)``, whitespace-collapsed.

    The window is the sentence containing the span, trimmed symmetrically to at most
    ``SNIPPET_MAX`` characters after whitespace collapsing.
    """
    s, e = sentence_bounds(text, start, SNIPPET_MAX * 2)
    s, e = min(s, start), max(e, end)
    nl = text.rfind("\n", s, start)
    if nl >= 0:
        s = nl + 1
    nl = text.find("\n", end, e)
    if nl >= 0:
        e = nl
    snippet = collapse(text[s:e])
    while len(snippet) > SNIPPET_MAX:
        excess = max(1, len(snippet) - SNIPPET_MAX)
        before, after = start - s, e - end
        if before >= after and before > 0:
            s += min(excess, before)
        elif after > 0:
            e -= min(excess, after)
        else:
            break
        snippet = collapse(text[s:e])
    return s, e, snippet


def _bounds_problem(field_name: str, low: float, high: float) -> str | None:
    lo, hi = HARD_BOUNDS[field_name]
    if low > high:
        return "reversed_range"
    if not (lo <= low and high <= hi):
        return "outside_hard_bounds"
    return None


def _local_context(text: str, position: int, index: MentionIndex) -> bool:
    """A house mention within 3,500 and a valuation-analysis marker within 6,000 characters."""
    if not index.between(max(0, position - 3500), position):
        return False
    return bool(_VALUATION_MARKER.search(text, max(0, position - 6000), position))


def _term_candidate(term: Term) -> Candidate:
    return Candidate(
        term.field,
        term.start,
        term.end,
        term.low,
        term.high,
        term.unit,
        term.raw_low,
        term.raw_high,
        "",
    )


def extract_document(doc: Document, company: str = "") -> Extraction:
    """Extract all admitted facts of one document, with rejections listed separately."""
    from .provenance import verify_span

    text = doc.text
    result = Extraction(doc.accession, doc.sha256)
    index = MentionIndex.build(text)
    segments = SegmentIndex(segment_document(doc, index))
    aliases = target_aliases(company)
    filer_role = infer_filer_role(text, aliases)
    others = counterparty_aliases(text, aliases)

    items: list[tuple[Candidate, str]] = [(c, "") for c in find_candidates(text)]
    items += [(_term_candidate(t), "") for t in find_per_share_ranges(text)]
    items += [(_term_candidate(t), t.house) for t in find_fee_terms(text, index)]
    offer = find_offer_price(text)
    if offer is not None:
        items.append((_term_candidate(offer), ""))
    items.sort(key=lambda p: (p[0].start, p[0].end, p[0].field))

    for cand, house in items:
        reject = _bounds_problem(cand.field, cand.low, cand.high)
        if (
            reject is None
            and cand.field in VALUATION_FIELDS
            and cand.is_point
            and _TABLE_FOLLOW.match(text[cand.end : cand.end + 14])
        ):
            reject = "table_row"
        seg = segments.at(cand.start)
        region = seg.kind if seg else "other"
        in_section = region in ("opinion", "background")
        needs_context = cand.field in VALUATION_FIELDS or cand.field == "value_per_share"
        if reject is None and needs_context and not in_section:
            if _local_context(text, cand.start, index):
                region = "context"
            else:
                reject = "outside_analysis_context"
        if reject:
            result.rejections.append(
                Rejection(
                    doc.accession, cand.field, cand.start, cand.end, cand.low, cand.high, reject
                )
            )
            continue
        use_seg = seg is not None and region != "context"
        seg_start = seg.start if use_seg and seg else cand.start
        seg_advisor = seg.advisor if use_seg and seg else UNATTRIBUTED
        start = cand.start
        if cand.field.startswith("implied") and not cand.reversed_order:
            m = re.search(
                r"impl\w*\s+(?:(?:a|an|the|range|of|illustrative)\s+){0,4}$", text[:start], re.I
            )
            if m:
                start = m.start()
        analysis = ""
        subject = ""
        advisor = house
        if cand.field in VALUATION_FIELDS or cand.field == "value_per_share":
            analysis = classify_analysis(text, start, seg_start)
            if cand.field in VALUATION_FIELDS and analysis == "unspecified":
                result.rejections.append(
                    Rejection(
                        doc.accession,
                        cand.field,
                        cand.start,
                        cand.end,
                        cand.low,
                        cand.high,
                        "no_analysis_marker",
                    )
                )
                continue
            if cand.field == "value_per_share" and analysis not in DCF_LIKE:
                result.rejections.append(
                    Rejection(
                        doc.accession,
                        cand.field,
                        cand.start,
                        cand.end,
                        cand.low,
                        cand.high,
                        "not_dcf_analysis",
                    )
                )
                continue
            advisor = attribute_advisor(text, start, seg_start, seg_advisor, index)
            subject = classify_subject(
                text,
                start,
                cand.end,
                aliases,
                others,
                filer_role,
                seg_start if use_seg else start - 2500,
            )
        s0, s1, snippet = make_snippet(text, start, cand.end)
        valuation_date = ""
        if cand.field == "discount_rate":
            ss, se = sentence_bounds(text, start, 900)
            valuation_date = find_valuation_date(text[ss:se], start - ss)
        fact = Fact(
            accession=doc.accession,
            field=cand.field,
            low=cand.low,
            high=cand.high,
            unit=cand.unit,
            raw_low=cand.raw_low,
            raw_high=cand.raw_high,
            advisor=advisor or UNATTRIBUTED,
            subject=subject,
            analysis=analysis,
            region=region,
            metric=_metric_of(text, cand),
            valuation_date=valuation_date,
            span_start=start,
            span_end=cand.end,
            snippet_start=s0,
            snippet_end=s1,
            snippet=snippet,
            doc_sha256=doc.sha256,
        )
        problems = verify_span(fact, text)
        if problems:
            result.rejections.append(
                Rejection(
                    doc.accession,
                    cand.field,
                    start,
                    cand.end,
                    cand.low,
                    cand.high,
                    "provenance:" + problems[0],
                )
            )
            continue
        result.facts.append(fact)
    return result
