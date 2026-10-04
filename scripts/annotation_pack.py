"""Print a reading pack for annotating one filing.

The pack lists heading-like lines and every sentence that mentions a valuation cue
together with a number. It uses only the text conversion and generic regular
expressions, never the extractor, so labels written from it stay independent of the
extractor's output.

    python scripts/annotation_pack.py <accession> [--all-sentences]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from fairness_ledger.catalog import read_catalog
from fairness_ledger.document import html_to_document
from fairness_ledger.edgar import read_cached_filing

ROOT = Path(__file__).resolve().parents[1]
CUE = re.compile(
    r"discount\s+rate|cost\s+of\s+(?:equity|capital)|WACC|weighted[\s-]+average\s+cost|"
    r"perpetuity|perpetual|terminal|exit\s+(?:\w+\s+){0,3}multiple|growth\s+rates?|"
    r"discounted\s+cash|dividend\s+discount|present\s+value",
    re.I,
)
NUMBER = re.compile(r"\d")
FEE = re.compile(r"\bfees?\b", re.I)
SENT = re.compile(
    r"(?<!\b[A-Z]\.)(?<!Inc\.)(?<!Co\.)(?<!Corp\.)(?<!Ltd\.)(?<!No\.)(?<!Mr\.)"
    r"(?<=[.;!?])\s+(?=[A-Z\"'(\[$])|\n"
)
HEAD_HINT = re.compile(
    r"^(?:the\s+)?(?:opinions?\s+of|financial\s+analys|summary\s+of|discounted|dividend\s+discount|"
    r"illustrative|present\s+value|net\s+asset|sum-of|selected|precedent|leveraged|background\s+of)",
    re.I,
)


MONEY_RANGE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s+per\s+[Ss]hare)?\s*(?:to|through|and|[-\u2013\u2014])\s*\$?\s?\d[\d,]*(?:\.\d+)?"
)
HOUSE = re.compile(
    r"Goldman|Morgan|BofA|Merrill|Citi|Credit Suisse|Barclays|UBS|Deutsche|Lazard|Evercore|"
    r"Centerview|"
    r"Moelis|PJT|Perella|Guggenheim|Jefferies|Houlihan|RBC|Wells Fargo|Piper|Sandler|Raymond James|"
    r"KBW|Keefe|Stifel|Cowen|Leerink|Qatalyst|Allen|Duff|Greenhill|Blair|KeyBanc|Imperial|Stephens|"
    r"Baird|Needham|Macquarie|BMO|Mizuho|Nomura|Riley|Oppenheimer|Rothschild|LionTree|Ducera|Truist",
)


def print_money(text: str, accession: str) -> None:
    label_path = ROOT / "gold" / "labels" / f"{accession}.txt"
    spans: list[tuple[int, int, str]] = []
    if label_path.exists():
        from fairness_ledger.gold import parse_labels

        spans = [
            (g.span_start, g.span_end, g.field)
            for g in parse_labels(accession, text, label_path.read_text(encoding="utf-8"))
        ]
    pos = 0
    last = None
    for part in SENT.split(text):
        start = text.find(part, pos)
        pos = start + len(part)
        end = start + len(part)
        ps = MONEY_RANGE.search(part) and "per share" in part.lower()
        fee = (
            re.search(r"\bfees?\b|compensation|\bpaid\b", part, re.I)
            and "$" in part
            and HOUSE.search(part)
            and not re.search(
                r"termination|Innisfree|Georgeson|MacKenzie|Okapi|Regan|solicit", part
            )
        )
        if not (ps or fee):
            continue
        tags = sorted({f for a, b, f in spans if start <= a and b <= end})
        mark = ("LABELLED " + ",".join(tags)) if tags else "NEW"
        line = f"[{start}] ({mark}) " + re.sub(r"\s+", " ", part)[:900]
        if line != last:
            print(line)
        last = line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accession")
    parser.add_argument("--max-sentence", type=int, default=2400)
    parser.add_argument(
        "--money", action="store_true", help="only per-share ranges and fee sentences"
    )
    parser.add_argument(
        "--window", type=int, nargs=2, metavar=("START", "END"), help="print raw text slice"
    )
    args = parser.parse_args()
    deals = read_catalog(ROOT / "data" / "release" / "filings.csv")
    filing = next(f for d in deals for f in d.filings if f.accession == args.accession)
    raw = read_cached_filing(
        ROOT / "data" / "raw" / "filings" / filing.accession / f"{filing.filename}.gz"
    )
    doc = html_to_document(filing.accession, raw)
    text = doc.text
    print(f"# {filing.accession} | {filing.company} | filed {filing.file_date} | {len(text)} chars")
    if args.window:
        print(text[args.window[0] : args.window[1]])
        return
    if args.money:
        print_money(text, filing.accession)
        return
    items: list[tuple[int, str]] = []
    for line in doc.lines:
        s = text[line.start : line.end].strip()
        if (
            4 <= len(s) <= 150
            and line.linked < 0.5
            and "|" not in s
            and HEAD_HINT.search(s)
            and not re.search(r"(?:\s|\|)\d{1,3}\s*$|\(page|\(see", s, re.I)
        ):
            items.append((line.start, "## " + s))
    pos = 0
    cash_seen = 0
    for part in SENT.split(text):
        start = text.find(part, pos)
        pos = start + len(part)
        if "|" in part and part.count("|") > 6:
            continue
        if re.search(r"preferred\s+(?:stock|units|shares)", part, re.I) and not re.search(
            r"discount\s+rate|growth|multiple|cost\s+of\s+equity", part, re.I
        ):
            continue
        has_cue = (
            CUE.search(part)
            and NUMBER.search(part)
            and re.search(r"%|\d\s*x\b|times|percent", part)
        )
        has_fee = (
            FEE.search(part)
            and "$" in part
            and re.search(r"advis|opinion|engag|retain", part, re.I)
        )
        has_ps = (
            "per share" in part
            and part.count("$") >= 2
            and re.search(r"discount|DCF|present value|perpetu|terminal|dividend", part, re.I)
        )
        has_cash = re.search(
            r"\$\d[\d,.]*\s+(?:in\s+cash|per\s+share\s+in\s+cash)|cash\s+(?:consideration|price)\s+of\s+\$",
            part,
            re.I,
        )
        if has_cash and start < 0.5 * len(text):
            cash_seen += 1
        if has_cue or has_fee or has_ps or (has_cash and cash_seen <= 6):
            items.append((start, f"[{start}] " + re.sub(r"\s+", " ", part)[: args.max_sentence]))
    items.sort(key=lambda p: p[0])
    last = None
    for _, s in items:
        if s != last:
            print(s)
        last = s


if __name__ == "__main__":
    sys.exit(main())
