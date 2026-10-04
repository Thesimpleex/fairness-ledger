"""Lexicon of financial-advisor house names and a mention finder.

Attribution is deliberately lexicon-based and conservative: a house that is not in the
lexicon is never guessed, the value goes to the explicit ``UNATTRIBUTED`` bucket.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

UNATTRIBUTED = "unattributed"

# canonical name -> surface forms (matched case-sensitively with word boundaries)
_HOUSES: dict[str, tuple[str, ...]] = {
    "Goldman Sachs": ("Goldman Sachs", "Goldman, Sachs", "Goldman Sach", "GS&Co"),
    "J.P. Morgan": ("J.P. Morgan", "J.P.Morgan", "JPMorgan", "JP Morgan", "J. P. Morgan", "JPM"),
    "Morgan Stanley": ("Morgan Stanley",),
    "BofA Securities": (
        "BofA Securities",
        "BofA Merrill Lynch",
        "BofA",
        "Merrill Lynch",
        "Bank of America Merrill Lynch",
        "Bank of America Securities",
        "Bank of America",
        "BAML",
    ),
    "Citi": ("Citigroup", "Citi", "Citibank"),
    "Credit Suisse": ("Credit Suisse",),
    "Barclays": ("Barclays",),
    "UBS": ("UBS",),
    "Deutsche Bank": ("Deutsche Bank",),
    "Lazard": ("Lazard",),
    "Evercore": ("Evercore",),
    "Centerview": ("Centerview",),
    "Moelis": ("Moelis",),
    "PJT Partners": ("PJT Partners", "PJT"),
    "Perella Weinberg": ("Perella Weinberg", "PWP"),
    "Guggenheim": ("Guggenheim",),
    "Jefferies": ("Jefferies",),
    "Houlihan Lokey": ("Houlihan Lokey", "Houlihan"),
    "RBC Capital Markets": ("RBC Capital Markets", "RBCCM", "RBC"),
    "Wells Fargo": ("Wells Fargo Securities", "Wells Fargo"),
    "Piper Sandler": ("Piper Sandler", "Piper Jaffray"),
    "Sandler O'Neill": ("Sandler O'Neill", "Sandler O’Neill"),
    "Raymond James": ("Raymond James",),
    "KBW": ("Keefe, Bruyette & Woods", "Keefe, Bruyette and Woods", "KBW"),
    "Stifel": ("Stifel",),
    "Cowen": ("Cowen",),
    "Leerink": ("Leerink",),
    "Qatalyst": ("Qatalyst",),
    "Allen & Company": ("Allen & Company", "Allen & Co."),
    "Duff & Phelps": ("Duff & Phelps",),
    "Kroll": ("Kroll",),
    "Greenhill": ("Greenhill",),
    "William Blair": ("William Blair",),
    "KeyBanc": ("KeyBanc", "KBCM"),
    "Truist": ("Truist",),
    "SunTrust Robinson Humphrey": ("SunTrust Robinson Humphrey", "SunTrust"),
    "Janney": ("Janney",),
    "Baird": ("Robert W. Baird", "Baird"),
    "Needham": ("Needham",),
    "Lincoln International": ("Lincoln International",),
    "Macquarie": ("Macquarie",),
    "BMO": ("BMO Capital Markets", "BMO"),
    "TD Securities": ("TD Securities",),
    "Mizuho": ("Mizuho",),
    "MUFG": ("MUFG",),
    "Nomura": ("Nomura",),
    "B. Riley": ("B. Riley",),
    "Oppenheimer": ("Oppenheimer",),
    "Canaccord": ("Canaccord",),
    "Craig-Hallum": ("Craig-Hallum",),
    "Roth": ("Roth Capital", "ROTH"),
    "Ladenburg Thalmann": ("Ladenburg",),
    "Stephens": ("Stephens Inc", "Stephens"),
    "LionTree": ("LionTree",),
    "Ducera": ("Ducera",),
    "FT Partners": ("Financial Technology Partners", "FT Partners"),
    "Hovde": ("Hovde",),
    "Boenning & Scattergood": ("Boenning",),
    "Wedbush": ("Wedbush",),
    "Northland": ("Northland",),
    "Imperial Capital": ("Imperial Capital",),
    "Cantor Fitzgerald": ("Cantor Fitzgerald",),
    "Tudor Pickering Holt": ("Tudor, Pickering, Holt", "Tudor Pickering Holt", "TPH"),
    "Petrie Partners": ("Petrie Partners",),
    "Peter J. Solomon": ("Peter J. Solomon",),
    "Rothschild": ("Rothschild",),
    "Santander": ("Santander",),
    "HSBC": ("HSBC",),
    "BNP Paribas": ("BNP Paribas",),
    "Societe Generale": ("Société Générale", "Societe Generale"),
    "Marathon Capital": ("Marathon Capital",),
    "Gordon Dyal": ("Gordon Dyal",),
    "Mooreland Partners": ("Mooreland",),
    "JMP Securities": ("JMP Securities",),
    "D.A. Davidson": ("D.A. Davidson",),
    "Dougherty": ("Dougherty",),
    "Chardan": ("Chardan",),
    "H.C. Wainwright": ("H.C. Wainwright",),
    "Maxim Group": ("Maxim Group",),
    "Alantra": ("Alantra",),
    "Blackstone": ("Blackstone Advisory",),
    "Union Square Advisors": ("Union Square",),
    "Gleacher": ("Gleacher",),
    "Simmons": ("Simmons & Company", "Simmons"),
    "Brean Capital": ("Brean",),
    "Seaport": ("Seaport",),
    "Berkshire Capital": ("Berkshire Capital",),
    "Capital One Securities": ("Capital One Securities",),
    "Fifth Third": ("Fifth Third Securities",),
    "Huntington": ("Huntington Capital Markets",),
    "ING": ("ING Bank",),
    "Natixis": ("Natixis",),
    "Scotiabank": ("Scotia Capital", "Scotiabank"),
    "CIBC": ("CIBC",),
    "National Bank": ("National Bank Financial",),
}

_SURFACE: dict[str, str] = {}
for _canonical, _forms in _HOUSES.items():
    for _form in _forms:
        _SURFACE[_form] = _canonical

_MENTION = re.compile(
    "|".join(
        (r"\b" if f[0].isalnum() else "") + re.escape(f) + (r"\b" if f[-1].isalnum() else "")
        for f in sorted(_SURFACE, key=len, reverse=True)
    )
)


_PUBLICATION = re.compile(
    r"\s+(?:Valuation\s+Handbook|Cost\s+of\s+Capital|Risk\s+Premium|Size\s+Study|Navigator)"
)


@dataclass(frozen=True)
class Mention:
    """One occurrence of a house name in the text."""

    start: int
    end: int
    house: str


def canonical_house(surface: str) -> str | None:
    """Canonical house name for a surface form, or ``None``."""
    return _SURFACE.get(surface)


def find_mentions(text: str, start: int = 0, end: int | None = None) -> list[Mention]:
    """All house-name mentions in ``text[start:end]`` in order of appearance."""
    stop = len(text) if end is None else end
    return [
        Mention(m.start(), m.end(), _SURFACE[m.group()])
        for m in _MENTION.finditer(text, start, stop)
        if m.group() in _SURFACE and not _PUBLICATION.match(text, m.end())
    ]


class MentionIndex:
    """Sorted mention list with fast look-ups by position."""

    def __init__(self, mentions: list[Mention]) -> None:
        self.mentions = mentions
        self._starts = [m.start for m in mentions]

    @classmethod
    def build(cls, text: str) -> MentionIndex:
        return cls(find_mentions(text))

    def between(self, start: int, end: int) -> list[Mention]:
        lo = bisect_right(self._starts, start - 1)
        hi = bisect_right(self._starts, end - 1)
        return self.mentions[lo:hi]

    def last_before(self, position: int, floor: int = 0) -> Mention | None:
        i = bisect_right(self._starts, position - 1) - 1
        if i >= 0 and self.mentions[i].start >= floor:
            return self.mentions[i]
        return None
