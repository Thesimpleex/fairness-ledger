"""Extraction grammar for valuation parameters in fairness-opinion summaries.

The grammar is a set of regular expressions of the form ``cue + connector + value``
(and the reversed ``value + cue``) where a *value* is a range or a point value with an
explicit unit. It recognises

* percentages: ``9.0%``, ``9.0 percent``, ``(1.0)%``, ``(1.0%)``, ``negative 1.0%``,
  ``-1.0%``; ranges joined by ``to``, ``-``, en/em dashes, ``and``, ``through``;
* multiples: ``9.5x``, ``9.5 times``, ranges as above;
* enumerations (``9.0%, 8.13% and 7.25%``; ``10.0x to 12.0x, 4.0x to 6.0x``), each member
  becoming its own candidate.

Every candidate carries the span of the phrase it was read from. Nothing here knows
about documents, advisors or sections; see :mod:`fairness_ledger.extract`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DASHES = "-‐‑‒–—―−"
_NUM = r"\d{1,3}(?:\.\d+)?"
_NEG = r"(?:(?:negative|minus)\s+)?"
_PCT_UNIT = r"(?:%|percent\b|per\s?cent\b)"
_X_UNIT = r"(?:x\b|×|times\b)"
_SEP = rf"(?:\s+(?:to|through|and)\s+|\s*[{DASHES}]\s*)"

_PCT_END = (
    rf"{_NEG}(?:\(\s*[{DASHES}]?{_NUM}\s*%\s*\)|\(\s*[{DASHES}]?{_NUM}\s*\)\s*{_PCT_UNIT}"
    rf"|[{DASHES}]?{_NUM}\s*{_PCT_UNIT})(?![\d.]\d)"
)
_PCT_LO = (
    rf"(?<![\w.,$]){_NEG}(?:\(\s*[{DASHES}]?{_NUM}\s*%?\s*\)\s*%?"
    rf"|[{DASHES}]?{_NUM}(?:\s*{_PCT_UNIT})?)"
)
_X_END = rf"{_NUM}\s*{_X_UNIT}"
_X_LO = rf"(?<![\w.,$]){_NUM}(?:\s*{_X_UNIT})?"

_PCT_RANGE = rf"(?P<lo>{_PCT_LO}){_SEP}(?P<hi>{_PCT_END})"
_PCT_POINT = rf"(?P<pt>{_PCT_END})"
_X_RANGE = rf"(?P<lo>{_X_LO}){_SEP}(?P<hi>{_X_END})"
_X_POINT = rf"(?P<pt>{_X_END})"

_LEAD_WORD = (
    r"(?:ranging|range|ranges|ranged|from|of|between|equal(?:ed|s)?\s+to|to|at|selected|"
    r"illustrative|assumed|approximately|about|around|in|the|a|an|using|and|or|applied|used|"
    r"estimated|midpoint|per\s+annum|annual|annualized|rate|rates|was|were|is|are|"
    r"after-tax|pre-tax|nominal)"
)
_LEAD = rf"(?:\s*[,:(\[]?\s*{_LEAD_WORD}){{0,8}}\s*[,:(\[]?\s*"
_UNIT_NUMBER = r"\d+(?:\.\d+)?\s*(?:x\b|%)"
_OTHER_CUE = r"discount\s+rate|cost\s+of\s+(?:equity|capital)|WACC|multiples?|exit"
_LOOSE_GAP = rf"(?:(?!{_UNIT_NUMBER}|{_OTHER_CUE})[^.;%]){{0,150}}?"
_AFTER_GUARD = (
    r"(?!\s*(?:%\s*)?(?:(?:unlevered|levered|pre-tax|after-tax|nominal|real|equity|illustrative)\s+)*"
    r"(?:perpetuity|perpetual|terminal|growth|discount\s+rate|cost\s+of|WACC|multiples?|exit))"
)

_DISCOUNT_CUE = (
    r"(?:discount\s+(?:rates?|range)|weighted[\s-]+average\s+cost\s+of\s+capital|WACC"
    r"|cost\s+of\s+(?:equity|capital))"
)
_GROWTH_CUE = (
    r"(?:(?:perpetuity|perpetual|terminal)(?:\s+value)?\s+(?:(?:unlevered\s+)?(?:free\s+)?"
    r"cash\s+flow\s+|dividend\s+)?growth(?:\s+rates?)?|growth\s+rates?\s+in\s+perpetuity"
    r"|perpetuity\s+rates?|(?:long[-\s]term\s+)?dividend\s+growth\s+rates?)"
)
_IMPLIED = r"impl(?:ied|ying|ies|y)\b"
_EXIT_CUE = (
    r"(?:(?:exit|terminal)(?:\s+[A-Za-z][A-Za-z/&\-]*){0,8}?\s+(?:multiples?|ratios?)"
    r"(?:\s+method)?)"
)
_TERMINAL_VALUE_CUE = (
    r"(?:terminal\s+(?:values?|year)|exit\s+(?:\w+\s+){0,3}multiples?\s+method)"
    + _LOOSE_GAP
    + r"(?:appl(?:ied|ying)|using|selected|assum\w+|"
    r"based\s+on|utiliz\w+|multipl\w+)"
    + _LOOSE_GAP
    + r"(?:multiples?\s*,?\s*)?(?:range\s+of|ranging\s+from|of|from|"
    r"between)\s*"
)


@dataclass(frozen=True)
class Candidate:
    """A value phrase found by the grammar, before any context is assigned."""

    field: str
    start: int
    end: int
    low: float
    high: float
    unit: str
    raw_low: str
    raw_high: str
    cue: str
    reversed_order: bool = False

    @property
    def is_point(self) -> bool:
        return self.low == self.high and self.raw_low == self.raw_high


def parse_number(text: str) -> float:
    """Parse one endpoint (``9.0%``, ``(1.0)%``, ``negative 1.5 percent``, ``$1,200.50``)."""
    s = text.strip()
    negative = bool(re.match(r"(?i)(negative|minus)\b", s))
    s = re.sub(r"(?i)^(negative|minus)\s+", "", s)
    if s.startswith("(") or s.startswith(tuple(DASHES)):
        negative = True
    match = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if match is None:
        raise ValueError(f"no number in {text!r}")
    value = float(match.group().replace(",", ""))
    return -value if negative else value


@dataclass(frozen=True)
class _Rule:
    field: str
    unit: str
    pattern: re.Pattern[str]
    direction: str
    implied_ok: bool = False


_RULES: list[_Rule] = []


def _add(
    field: str, unit: str, pattern: str, direction: str = "fwd", implied_ok: bool = False
) -> None:
    _RULES.append(_Rule(field, unit, re.compile(pattern, re.I), direction, implied_ok))


def _fwd(cue: str, value: str, lead: str = _LEAD) -> str:
    return rf"(?P<cue>{cue}){lead}{value}{_AFTER_GUARD}"


_ADJ = (
    r"(?:(?:pre-tax|after-tax|nominal|annual|equity|estimated|selected|illustrative|assumed|"
    r"applied|pro\s+forma)\s+){0,2}"
)


def _rev(value: str, cue: str) -> str:
    return rf"{value}\s*,?\s*{_ADJ}(?P<cue>{cue})"


_DISC_REV_CUE = (
    r"discount\s+rates?|WACC|weighted[\s-]+average\s+cost\s+of\s+capital|"
    r"cost\s+of\s+(?:equity|capital)"
)
_POINT_START = r"(?<![\w.,$])"

for _value in (_PCT_RANGE, _PCT_POINT):
    _add("discount_rate", "pct", _fwd(_DISCOUNT_CUE, _value))
_add("discount_rate", "pct", _rev(_PCT_RANGE, _DISC_REV_CUE), "rev")
_add("discount_rate", "pct", _rev(_POINT_START + _PCT_POINT, _DISC_REV_CUE), "rev")

for _value in (_PCT_RANGE, _PCT_POINT):
    _add("terminal_growth", "pct", _fwd(_GROWTH_CUE, _value), implied_ok=True)
for _value in (_PCT_RANGE, _PCT_POINT):
    _add("terminal_growth", "pct", _fwd(_GROWTH_CUE, _value, _LOOSE_GAP), "loose")
_add("terminal_growth", "pct", _rev(_PCT_RANGE, _GROWTH_CUE), "rev")
_add("terminal_growth", "pct", _rev(_POINT_START + _PCT_POINT, _GROWTH_CUE), "rev")

for _value in (_X_RANGE, _X_POINT):
    _add("exit_multiple", "x", _fwd(_EXIT_CUE, _value), implied_ok=True)
_add("exit_multiple", "x", rf"(?P<cue>{_TERMINAL_VALUE_CUE}){_X_RANGE}", "loose")
_add(
    "exit_multiple",
    "x",
    _rev(_X_RANGE, r"(?:exit|terminal)(?:\s+[A-Za-z][A-Za-z/&\-]*){0,4}?\s+multiples?"),
    "rev",
)

_ENTITY = r"[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,3}"
_add(
    "exit_multiple",
    "x",
    rf"(?P<cue>multiples?\s*,?\s*(?:ranging\s+from|range\s+of|of)\s*){_X_RANGE}"
    rf"[^.;]{{0,220}}?\bterminal\s+values?",
    "loose",
)
_add(
    "exit_multiple",
    "x",
    rf"(?P<cue>(?:earnings|EPS|price)(?:\s+per\s+share)?\s+multiples?\s+range\s+of\s*){_X_RANGE}"
    r"\s+referred\s+to\s+above",
    "loose",
)
_ENUM_RANGE = rf"\s*(?:\s+for\s+{_ENTITY})?(?:,\s*|\s+and\s+|\s+or\s+)(?:and\s+)?"
_ENUM_POINT = r"\s*(?:,\s*|\s+and\s+|\s+or\s+)(?:and\s+)?"
_IMPLIED_FILLER = re.compile(_IMPLIED + r"\s+(?:(?:a|an|the|range|of|illustrative)\s+){0,4}$", re.I)
_NOT_DISCOUNT_RATE = re.compile(r"(?:federal\s+reserve|reserve\s+bank|\bprime\b|window)\W*$", re.I)

_ANCHOR = re.compile(
    r"discount\s+(?:rate|range)|WACC|weighted[\s-]+average\s+cost|cost\s+of\s+(?:equity|capital)|growth"
    r"|multiple|ratio|terminal\s+value",
    re.I,
)
_WINDOW_BEFORE = 160
_WINDOW_AFTER = 520
_ENUM_PCT_RANGE = re.compile(_ENUM_RANGE + rf"(?P<lo>{_PCT_LO}){_SEP}(?P<hi>{_PCT_END})", re.I)
_ENUM_X_RANGE = re.compile(_ENUM_RANGE + rf"(?P<lo>{_X_LO}){_SEP}(?P<hi>{_X_END})", re.I)
_ENUM_PCT_POINT = re.compile(_ENUM_POINT + rf"(?P<pt>{_PCT_END})", re.I)
_ENUM_X_POINT = re.compile(_ENUM_POINT + rf"(?P<pt>{_X_END})", re.I)
_UNIT_TOKEN = re.compile(r"%|percent|per\s?cent|x\b|×|times", re.I)


def _windows(text: str, start: int, stop: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _ANCHOR.finditer(text, start, stop):
        lo, hi = max(start, m.start() - _WINDOW_BEFORE), min(stop, m.end() + _WINDOW_AFTER)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return spans


def _make(
    rule: _Rule, cue: str, name: str, lo_text: str, hi_text: str, start: int, end: int
) -> Candidate | None:
    try:
        low, high = parse_number(lo_text), parse_number(hi_text)
    except ValueError:
        return None
    return Candidate(
        field=name,
        start=start,
        end=end,
        low=low,
        high=high,
        unit=rule.unit,
        raw_low=lo_text.strip(),
        raw_high=hi_text.strip(),
        cue=cue,
        reversed_order=rule.direction == "rev",
    )


def _continuations(
    text: str, rule: _Rule, cue: str, first: Candidate, is_range: bool
) -> list[Candidate]:
    """Further members of an enumeration that follows ``first`` directly."""
    if rule.direction == "rev":
        return []
    if rule.unit == "pct":
        pattern = _ENUM_PCT_RANGE if is_range else _ENUM_PCT_POINT
    else:
        pattern = _ENUM_X_RANGE if is_range else _ENUM_X_POINT
    out: list[Candidate] = []
    pos = first.end
    while len(out) < 6:
        nxt = pattern.match(text, pos, min(len(text), pos + 160))
        if nxt is None:
            break
        if is_range:
            lo_text, hi_text = nxt.group("lo"), nxt.group("hi")
            if not _UNIT_TOKEN.search(hi_text):
                break
        else:
            lo_text = hi_text = nxt.group("pt")
        cand = _make(rule, cue, first.field, lo_text, hi_text, first.start, nxt.end())
        if cand is None:
            break
        out.append(cand)
        pos = cand.end
    return out


def find_candidates(text: str, start: int = 0, end: int | None = None) -> list[Candidate]:
    """Find all value phrases for the supported fields inside ``text[start:end]``.

    Overlapping matches of the same field are resolved in favour of earlier rules
    (tight before loose patterns, ranges before points). A growth rate or multiple
    introduced by *implied* (the perpetuity growth rate an exit multiple implies, or
    vice versa) is reported as ``implied_growth`` / ``implied_multiple``.
    """
    stop = len(text) if end is None else end
    found: list[Candidate] = []
    windows = _windows(text, start, stop)
    taken: dict[str, list[tuple[int, int]]] = {}
    for rule in _RULES:
        spans = taken.setdefault(rule.field, [])
        for w_start, w_stop in windows:
            for m in rule.pattern.finditer(text, w_start, w_stop):
                s, e = m.start(), m.end()
                if any(s < b and a < e for a, b in spans):
                    continue
                cue_start = m.start("cue")
                if rule.field == "discount_rate" and _NOT_DISCOUNT_RATE.search(
                    text[max(0, cue_start - 30) : cue_start]
                ):
                    continue
                gd = m.groupdict()
                is_range = bool(gd.get("lo"))
                lo_text = gd["lo"] if is_range else gd["pt"]
                hi_text = gd["hi"] if is_range else lo_text
                name = rule.field
                if (
                    rule.implied_ok
                    and rule.direction == "fwd"
                    and _IMPLIED_FILLER.search(text[max(0, cue_start - 90) : cue_start])
                ):
                    name = (
                        "implied_growth" if rule.field == "terminal_growth" else "implied_multiple"
                    )
                cand = _make(rule, m.group("cue"), name, lo_text, hi_text, s, e)
                if cand is None:
                    continue
                for member in (cand, *_continuations(text, rule, m.group("cue"), cand, is_range)):
                    spans.append((member.start, member.end))
                    found.append(member)
    found.sort(key=lambda c: (c.start, c.end))
    return found
