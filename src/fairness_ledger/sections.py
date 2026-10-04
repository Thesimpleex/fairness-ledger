"""Section segmentation of merger proxies.

Headings are recognised from short block-level lines. Table-of-contents entries and
summary cross-references are rejected (page numbers, hyperlinks, ``(page 87)``). The
document is then cut into *opinion* segments (an advisor's opinion and analyses),
*background* segments (the narrative of the negotiations) and everything else.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from .advisors import UNATTRIBUTED, MentionIndex
from .document import Document

_MAX_HEADING = 260
_PAGE_REF = re.compile(r"(?:\bpages?\s+\d+|\(\s*see\s+page|\bpage\s+[ivx]+\b)", re.I)
_TRAILING_PAGE = re.compile(r"(?:\|\s*|\s)(?:\d{1,3}|[ivxl]{1,5})\s*\|?\s*$")
_LEADING_NOISE = re.compile(r"^[\W_]*(?:\d{1,2}[.)]\s+)?")

_OPINION = re.compile(
    r"^(?:the\s+|our\s+|summary\s+of\s+(?:the\s+)?)?(?:fairness\s+)?opinions?\s+(?:of|from)\b"
    r"|^.{3,80}?['’]s\s+(?:fairness\s+)?opinion\b|^fairness\s+opinions?\b",
    re.I,
)
_OPINION_EXCLUDE = re.compile(
    r"financial\s+statements|tax\s+(?:counsel|opinion)|legal|counsel|accountant|auditor|"
    r"independent\s+registered|opinion\s+on\b|will\s+not\b|may\s+not\b|not\s+reflect|"
    r"opinions?\s+of\s+(?:the\s+)?(?:board|directors)\b|certain\s+of\s+the\s+company",
    re.I,
)
_BACKGROUND = re.compile(
    r"^(?:the\s+)?background\s+(?:of|to)\s+the\s+(?:merger|offer|transaction|proposed|acquisition|"
    r"sale|business\s+combination|asset|arrangement|going|spin|tender|mergers|transactions)",
    re.I,
)
_END = re.compile(
    r"^(?:certain\s+)?(?:unaudited\s+)?(?:management\s+)?(?:prospective|projected)\s+"
    r"(?:financial\s+)?(?:information|projections|forecasts|results)\b|"
    r"^certain\s+(?:unaudited\s+)?(?:financial\s+)?(?:projections|forecasts)\b|"
    r"^interests\s+of\s+(?:certain\s+)?(?:the\s+)?(?:\w+['\u2019]?s?\s+)?"
    r"(?:directors|officers|executive|persons|management|certain)|"
    r"^(?:the\s+)?financing(?:\s+of\s+the\s+(?:merger|transaction|offer))?\s*$|"
    r"^regulatory\s+(?:approvals?|matters|clearances?)|"
    r"^(?:the\s+)?(?:merger|purchase|asset\s+purchase|transaction)\s+agreement\s*$|"
    r"^(?:appraisal|dissenters['\u2019]?)\s+rights|^litigation\s+relating|"
    r"^(?:certain\s+)?(?:material\s+)?(?:u\.s\.\s+)?federal\s+income\s+tax\s+"
    r"(?:consequences|considerations)|^accounting\s+treatment|^delisting\s+and|"
    r"^recommendation\s+of\s+(?:the|our)\s+board|^reasons\s+for\s+the\s+(?:merger|"
    r"offer|transaction)|^(?:certain\s+)?effects\s+of\s+the\s+(?:merger|offer)|"
    r"^(?:security|stock|beneficial)\s+ownership|^where\s+you\s+can\s+find|^annex\s+[a-z]\b|"
    r"^the\s+special\s+meeting\s*$|^risk\s+factors\s*$|^cautionary\s+statement|"
    r"^conditions\s+to\s+the\s+(?:merger|offer|completion)|^termination\s+of\s+the\s+"
    r"(?:merger|agreement)",
    re.I,
)
_STOPWORDS = frozenset(
    ["of", "the", "and", "to", "for", "in", "on", "by", "a", "an", "or", "as", "at", "with"]
)


def _is_title_case(text: str) -> bool:
    words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z'\u2019.&-]*", text) if w.lower() not in _STOPWORDS
    ]
    if not words:
        return False
    upper = sum(1 for w in words if w[0].isupper())
    return upper / len(words) >= 0.7


@dataclass(frozen=True)
class Heading:
    start: int
    end: int
    text: str
    kind: str


@dataclass(frozen=True)
class Segment:
    """A contiguous stretch of text with a kind and, for opinions, an advisor."""

    kind: str
    start: int
    end: int
    heading: str
    advisor: str = UNATTRIBUTED


def _is_toc_like(text: str, linked: float) -> bool:
    return linked >= 0.5 or bool(_PAGE_REF.search(text)) or bool(_TRAILING_PAGE.search(text))


def find_headings(doc: Document) -> list[Heading]:
    """Opinion, background and section-end headings with table-of-contents entries removed."""
    out: list[Heading] = []
    text = doc.text
    for line in doc.lines:
        if line.end - line.start > _MAX_HEADING:
            continue
        raw = text[line.start : line.end].strip()
        if not raw or _is_toc_like(raw, line.linked):
            continue
        cleaned = _LEADING_NOISE.sub("", raw)
        kind = None
        if _OPINION.search(cleaned) and not _OPINION_EXCLUDE.search(cleaned):
            kind = "opinion"
        elif _BACKGROUND.search(cleaned):
            kind = "background"
        elif (
            len(cleaned) <= 110
            and "|" not in cleaned
            and not cleaned.endswith((";", ":", ","))
            and _is_title_case(cleaned)
            and _END.search(cleaned)
        ):
            kind = "end"
        if kind:
            out.append(Heading(line.start, line.end, cleaned, kind))
    return out


def _majority_house(index: MentionIndex, start: int, end: int) -> str:
    counts: dict[str, int] = {}
    for m in index.between(start, end):
        counts[m.house] = counts.get(m.house, 0) + 1
    if not counts:
        return UNATTRIBUTED
    return max(
        counts,
        key=lambda h: (counts[h], -min(m.start for m in index.between(start, end) if m.house == h)),
    )


def segment_document(doc: Document, index: MentionIndex) -> list[Segment]:
    """Cut ``doc`` into opinion / background / other segments (non-overlapping, ordered)."""
    headings = find_headings(doc)
    segments: list[Segment] = []
    cursor = 0
    for i, h in enumerate(headings):
        nxt = headings[i + 1].start if i + 1 < len(headings) else len(doc.text)
        if h.kind == "end":
            kind, advisor = "other", UNATTRIBUTED
        elif h.kind == "background":
            kind, advisor = "background", UNATTRIBUTED
        else:
            kind = "opinion"
            in_heading = {m.house for m in index.between(h.start, h.end)}
            if len(in_heading) == 1:
                advisor = next(iter(in_heading))
            else:
                advisor = _majority_house(index, h.start, min(nxt, h.start + 6000))
        if h.start > cursor:
            segments.append(Segment("other", cursor, h.start, "", UNATTRIBUTED))
        segments.append(Segment(kind, h.start, nxt, h.text, advisor))
        cursor = nxt
    if cursor < len(doc.text):
        segments.append(Segment("other", cursor, len(doc.text), "", UNATTRIBUTED))
    return segments


class SegmentIndex:
    """Position look-up over ordered segments."""

    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        self._starts = [s.start for s in segments]

    def at(self, position: int) -> Segment | None:
        i = bisect_right(self._starts, position) - 1
        if 0 <= i < len(self.segments) and position < self.segments[i].end:
            return self.segments[i]
        return None
