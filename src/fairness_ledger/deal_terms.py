"""Deal-level grammar: offer price, advisor fees, implied per-share ranges, valuation dates.

These finders complement :mod:`fairness_ledger.grammar` (valuation parameters). Each
returns :class:`Term` objects whose span is the dollar phrase itself (``$2.5 million``,
``$44.76 to $51.30``); contextual decisions (section, analysis type) are made in
:mod:`fairness_ledger.extract`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

from .advisors import MentionIndex
from .grammar import DASHES

_MONEY = (
    r"(?<![\w.])(?:U\.S\.)?\$\s?(?P<n>\d{1,4}(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?P<s>million|billion|thousand))?"
)
_MONEY_RE = re.compile(_MONEY, re.I)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_LIST = [m.lower() for m in _MONTHS.split("|")]
_DATE_MDY = re.compile(rf"(?P<m>{_MONTHS})\s+(?P<d>\d{{1,2}}),?\s+(?P<y>(?:19|20)\d{{2}})", re.I)
_DATE_DMY = re.compile(rf"(?P<d>\d{{1,2}})\s+(?P<m>{_MONTHS}),?\s+(?P<y>(?:19|20)\d{{2}})", re.I)
_SENT_SPLIT = re.compile(
    r"(?<!\b[A-Z]\.)(?<!Inc\.)(?<!Co\.)(?<!Corp\.)(?<!Ltd\.)(?<!No\.)"
    r"(?<=[.;!?])\s+(?=[A-Z\"'(\[$])|\n"
)


@dataclass(frozen=True)
class Term:
    """A deal-level value; ``house`` is set for fees."""

    field: str
    start: int
    end: int
    low: float
    high: float
    unit: str
    raw_low: str
    raw_high: str
    house: str = ""


def money_value(amount: str, scale: str | None, unit: str) -> float:
    """Numeric value of a dollar phrase in dollars (``usd``) or millions (``usd_million``)."""
    value = float(amount.replace(",", ""))
    factor = {"thousand": 1e3, "million": 1e6, "billion": 1e9}.get((scale or "").lower(), 1.0)
    dollars = value * factor
    return dollars if unit == "usd" else dollars / 1e6


def reparse_money(span_text: str, unit: str) -> list[float]:
    """All dollar values in ``span_text`` in the given unit (used by the provenance check)."""
    return [money_value(m.group("n"), m.group("s"), unit) for m in _MONEY_RE.finditer(span_text)]


def sentence_bounds(text: str, position: int, window: int = 900) -> tuple[int, int]:
    """Start and end of the sentence around ``position`` (bounded by ``window``)."""
    lo = max(0, position - window)
    starts = [lo]
    for m in _SENT_SPLIT.finditer(text, lo, position):
        starts.append(m.end())
    s = starts[-1]
    hi = min(len(text), position + window)
    m = _SENT_SPLIT.search(text, position, hi)
    e = m.start() if m else hi
    return s, e


# -- offer price ---------------------------------------------------------------------

_OFFER = re.compile(
    r"(?:entitled\s+to\s+receive|right\s+to\s+receive|will\s+receive|receive|"
    r"merger\s+consideration\s+of|consideration\s+of|offer\s+price\s+of|purchase\s+price\s+of|"
    r"price\s+of|(?:will|to)\s+pay|amount\s+in\s+cash\s+equal\s+to|cash\s+equal\s+to)\s+"
    r"(?:(?:an\s+amount\s+)?(?:in\s+cash\s+)?(?:equal\s+to\s+)?|approximately\s+|\(a\)\s+)?"
    + _MONEY,
    re.I,
)
_CASH_AFTER = re.compile(
    r"\s*(?:per\s+(?:share|ordinary\s+share|common\s+share|unit)\s+)?(?:in\s+cash|cash)|"
    r"\s*per\s+(?:share|ordinary\s+share|common\s+share|unit)\b|"
    r"\s*(?:,\s*)?(?:without\s+interest|net\s+to\s+the\s+seller)",
    re.I,
)
_MIXED_AFTER = re.compile(
    r"\s*,?\s*(?:and|plus|together\s+with)\s+(?:\(?[a-z0-9]\)\s+)?"
    r"(?:\d[\d.,]*|one|a\s+fraction|the\s+(?:remainder|balance)|[\d.,]+\s+of\s+a)"
    r"[^;$]{0,60}?(?:shares?|common\s+stock|stock|units)",
    re.I,
)
_NOT_OFFER_BEFORE = re.compile(
    r"premium|closing\s+(?:stock\s+)?price|trading\s+price|\bhigh\b|\blow\b|average|VWAP|"
    r"52-week|termination|break-?up|exercise\s+price|strike|option|warrant|liquidation|fee|"
    r"book\s+value|reference\s+price|unaffected|value\s+per\s+share|proposal|between|"
    r"increase|increased|reduc|initial|revised|preferred|series\s+[a-z]\b|CVR|implied|"
    r"aggregate|equity\s+value|transaction\s+value",
    re.I,
)


def find_offer_price(text: str, limit: int = 1_500_000) -> Term | None:
    """Per-share cash consideration, if one consistent cash price is stated at least twice.

    Matches of the consideration pattern vote; the most frequent price wins (ties:
    earliest). Statements followed by additional share consideration are skipped.
    """
    votes: Counter[float] = Counter()
    first: dict[float, re.Match[str]] = {}
    for m in _OFFER.finditer(text, 0, min(len(text), limit)):
        after = text[m.end() : m.end() + 160]
        cash_at = after[:70].find("cash")
        if cash_at < 0 and "cash" not in text[max(0, m.start() - 40) : m.start()]:
            continue
        tail = after[cash_at + 4 :] if cash_at >= 0 else after
        if _MIXED_AFTER.match(tail) or (
            re.match(r"\s*,\s*\(?[a-z]{1,3}\)\s+[\d.,]+\s+(?:of\s+a\s+)?shares?", tail, re.I)
            and not re.search(r"\bor\b", tail[:200])
        ):
            continue
        before = text[max(0, m.start() - 70) : m.start()]
        if _NOT_OFFER_BEFORE.search(before):
            continue
        price = money_value(m.group("n"), m.group("s"), "usd")
        if not 0.5 <= price <= 5000:
            continue
        votes[price] += 1
        first.setdefault(price, m)
    if not votes:
        return None
    best = max(votes, key=lambda p: (votes[p], -first[p].start()))
    if votes[best] < 2:
        return None
    m = first[best]
    raw = text[m.start("n") - 1 : m.end()]
    raw = raw[raw.index("$") :]
    return Term("offer_price", m.start("n") - 1, m.end(), best, best, "usd", raw, raw)


# -- advisor fees --------------------------------------------------------------------

_FEE_WORD = re.compile(r"\bfees?\b|\bcompensation\b", re.I)
_FEE_EXCLUDE = re.compile(
    r"termination|break-?up|unrelated|preceding|prior\s+two|previous\s+(?:two|three|years?)|"
    r"past\s+(?:two|three)|"
    r"(?:two|three)[- ]year|during\s+(?:such|the|that)\s+(?:past\s+)?(?:two|three)|reimburs|"
    r"expenses|solicit|proxy\s+solicit|received\s+(?:an\s+)?(?:aggregate\s+)?(?:fees?|compensation)|"
    r"for\s+which\s+(?:\w+\s+){1,4}received|have\s+received|recogni[sz]ed|portfolio\s+companies|"
    r"may\s+pay|additional\s+fee|capital\s+rais|placement\s+agent|book\s*runner|"
    r"book\s+manager|lead\s+arranger|co-manager|underwrit|financing|tendered|director",
    re.I,
)
_NOT_SPLIT = r"(?:(?!and\s+the\s+(?:balance|remainder|rest)\b|another\s+opinion)[^;$])"
_FEE_OPINION_RIGHT = re.compile(
    rf"^{_NOT_SPLIT}{{0,140}}?(?:upon|on|at\s+the\s+time|concurrently\s+with|in\s+connection\s+with|"
    rf"for|following|when|as\s+a\s+result\s+of)\s+(?:{_NOT_SPLIT}{{0,40}}?)(?:delivery|delivering|rendering|rendered|"
    rf"delivered|issuance|issuing|its|such)\b{_NOT_SPLIT}{{0,40}}?opinion",
    re.I,
)
_FEE_OPINION_LEFT = re.compile(r"(?:opinion|fairness\s+opinion)\s+fee\s+(?:of\s+)?$", re.I)
_FEE_CONTINGENT_RIGHT = re.compile(
    rf"^{_NOT_SPLIT}{{0,200}}?(?:contingent|payable|due|will\s+be\s+payable|becomes?\s+payable)"
    rf"{_NOT_SPLIT}{{0,60}}?(?:upon|on|following)\s+(?:the\s+)?(?:successful\s+)?(?:consummation|"
    rf"closing|completion|effectiveness)",
    re.I,
)
_FEE_CONTINGENT_LEFT = re.compile(
    r"(?:contingent|completion|success|additional)\s+(?:fee|compensation)\s+(?:of\s+)?$", re.I
)
_FEE_ON_CLOSING = re.compile(
    r"^\s*(?:upon|at)\s+(?:the\s+)?(?:closing|consummation|completion)", re.I
)
_PARTITION = re.compile(
    r"^\s*,?\s*(?:(?:all|substantially\s+all|a\s+significant\s+portion|a\s+portion|a\s+part|"
    r"the\s+(?:large\s+)?majority)\s+)?of\s+which",
    re.I,
)
_FEE_LABEL = re.compile(r"\bfees?\b|\bcompensation\b", re.I)


def _total_left(left: str) -> bool:
    """Is the amount introduced as *the* fee (fee word shortly before, no other amount between)?"""
    last = None
    for last in _FEE_LABEL.finditer(left):  # noqa: B007 - want the final match
        pass
    if last is None:
        return False
    between = left[last.end() :]
    return (
        "$" not in between
        and "%" not in between
        and len(between) <= 130
        and not re.search(
            r"\b(?:received|credit|reimburs|of\s+which|portion|paid)\b", between, re.I
        )
    )


def find_fee_terms(
    text: str, index: MentionIndex, start: int = 0, end: int | None = None
) -> list[Term]:
    """Fee amounts disclosed for a named house: total, opinion portion, contingent portion.

    A sentence qualifies if it mentions a fee, names a lexicon house within the sentence
    or the 250 characters before it, and is not about termination fees, expenses,
    financing roles or earlier engagements. Each dollar amount is classified from its own
    wording: the amount introduced as the fee is the total (also the opinion fee when it
    is stated as paid on delivery of the opinion and not credited against a larger fee);
    later amounts are the opinion or contingent portion if their wording says so.
    """
    stop = len(text) if end is None else end
    out: list[Term] = []
    seen_sentence: set[int] = set()
    for m in _MONEY_RE.finditer(text, start, stop):
        s, e = sentence_bounds(text, m.start(), 900)
        if s in seen_sentence:
            continue
        sentence = text[s:e]
        if not _FEE_WORD.search(sentence) or _FEE_EXCLUDE.search(sentence):
            continue
        mentions = index.between(max(0, s - 250), e)
        if not mentions:
            continue
        seen_sentence.add(s)
        first_total_done = False
        credited = bool(re.search(r"\bcredit(?:ed|able)\b", text[s : e + 260], re.I))
        for money in _MONEY_RE.finditer(text, s, e):
            house_m = [mm for mm in mentions if mm.start < money.start()] or mentions
            house = house_m[-1].house
            left = text[max(s, money.start() - 170) : money.start()]
            right = text[money.end() : e]
            raw = text[money.start() : money.end()]
            value = money_value(money.group("n"), money.group("s"), "usd_million")
            if money.group("s") is None and value > 100:
                continue
            if re.search(r"between\s*$", left, re.I) or re.match(r"\s+and\s+\$", right):
                continue
            total_left = _total_left(left)
            partition = bool(_PARTITION.match(right)) and not first_total_done
            kind = None
            if not partition:
                if _FEE_OPINION_LEFT.search(left) or _FEE_OPINION_RIGHT.match(right[:260]):
                    kind = "fee_opinion"
                elif (
                    _FEE_CONTINGENT_LEFT.search(left)
                    or _FEE_CONTINGENT_RIGHT.match(right[:300])
                    or (total_left and _FEE_ON_CLOSING.match(right))
                ):
                    kind = "fee_contingent"
            additional = bool(re.search(r"\badditional\b", left[-60:], re.I))
            emit: list[str] = []
            if total_left and not first_total_done and not additional:
                if not (kind == "fee_opinion" and credited):
                    emit.append("fee_total")
                    first_total_done = True
                if kind == "fee_opinion":
                    emit.append("fee_opinion")
            elif kind is not None:
                emit.append(kind)
            for field in emit:
                out.append(
                    Term(
                        field,
                        money.start(),
                        money.end(),
                        value,
                        value,
                        "usd_million",
                        raw,
                        raw,
                        house,
                    )
                )
    out.sort(key=lambda t: (t.start, t.field))
    return out


# -- implied value per share ------------------------------------------------------------

_PS_RANGE = re.compile(
    rf"(?<![\w.])\$\s?(?P<lo>\d{{1,4}}(?:,\d{{3}})*(?:\.\d+)?)(?:\s+per\s+(?:fully\s+diluted\s+)?"
    rf"(?:common\s+)?(?:share|unit))?(?:\s+(?:to|through|and)\s+|\s*[{DASHES}]\s*)"
    rf"\$?\s?(?P<hi>\d{{1,4}}(?:,\d{{3}})*(?:\.\d+)?)(?![\d,]*\d%)",
    re.I,
)
_PS_CUE = re.compile(
    r"\b(?:implied|indicated|illustrative|imputed|reference|resulting|resulted|yield(?:ed|ing)|"
    r"produc(?:ed|ing)|deriv(?:ed|ing)|range\s+of\s+(?:values|prices)|values?\s+per|"
    r"present\s+values?)\b",
    re.I,
)
_PS_SHARE = re.compile(
    r"per\s+(?:[A-Z][\w&.'-]*\s+){0,3}(?:fully\s+diluted\s+)?(?:common\s+)?(?:share|unit)|"
    r"\bper\s+Share\b",
    re.I,
)
_PS_NEGATIVE = re.compile(
    r"closing\s+(?:stock\s+)?price|trading\s+(?:range|price)|52-week|price\s+targets?|target\s+prices?|"
    r"low\s+to\s+high|high\s+and\s+low|consideration\s+value|implied\s+consideration|"
    r"per\s+share\s+consideration|refers\s+to|stock\s+price\s+targets",
    re.I,
)


def find_per_share_ranges(text: str, start: int = 0, end: int | None = None) -> list[Term]:
    """Dollar ranges introduced as implied/derived values per share."""
    stop = len(text) if end is None else end
    out: list[Term] = []
    for m in _PS_RANGE.finditer(text, start, stop):
        lo = float(m.group("lo").replace(",", ""))
        hi = float(m.group("hi").replace(",", ""))
        if lo > hi or lo <= 0 or hi > 20000:
            continue
        s, e = sentence_bounds(text, m.start(), 600)
        head = text[s : m.start()]
        tail = text[m.end() : e]
        scope = head + " " + text[m.start() : m.end()] + " " + tail[:60]
        if not (_PS_SHARE.search(scope) and _PS_CUE.search(head[-260:] + tail[:40])):
            continue
        if _PS_NEGATIVE.search(head[-220:]) or "|" in text[max(0, m.start() - 14) : m.start()]:
            continue
        if "|" in text[m.end() : m.end() + 14] or re.match(r"\s+in\s+cash", tail):
            continue
        raw_lo = "$" + m.group("lo")
        raw_hi = m.group("hi")
        out.append(
            Term(
                "value_per_share",
                m.start(),
                m.end(),
                lo,
                hi,
                "usd",
                raw_lo,
                raw_hi,
            )
        )
    return out


# -- valuation date ----------------------------------------------------------------------


def find_valuation_date(sentence: str, anchor: int) -> str:
    """ISO date of the discounting date stated in ``sentence`` closest to ``anchor``.

    Only dates introduced by ``as of`` / ``to`` / ``back to`` / ``at`` in a sentence about
    discounting are considered; returns an empty string if none is stated.
    """
    if not re.search(r"present|discount|valu", sentence, re.I):
        return ""
    best, best_distance = "", 10**9
    for pattern in (_DATE_MDY, _DATE_DMY):
        for m in pattern.finditer(sentence):
            lead = sentence[max(0, m.start() - 40) : m.start()].lower()
            if not re.search(r"(?:as\s+of|\bto|back\s+to|\bat|\()\s*$", lead):
                continue
            month = _MONTH_LIST.index(m.group("m").lower()) + 1
            try:
                iso = date(int(m.group("y")), month, int(m.group("d"))).isoformat()
            except ValueError:
                continue
            distance = abs(m.start() - anchor)
            if distance < best_distance:
                best, best_distance = iso, distance
    return best
