"""Evaluate the extractor on the gold set (dev or holdout split).

python scripts/evaluate_gold.py dev [--errors] [--confusion subject]
python scripts/evaluate_gold.py holdout     # verifies the frozen hash first
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from fairness_ledger.evaluation import run_split
from fairness_ledger.gold import match_facts

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float) -> str:
    return "  n/a" if x != x else f"{100 * x:5.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "holdout"])
    parser.add_argument("--errors", action="store_true", help="print missed and extra facts")
    parser.add_argument("--confusion", help="attribute: print gold/predicted pairs per filing")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        run = run_split(ROOT, args.split)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    res = run.result
    print(
        f"split={args.split} filings={res['n_filings']} gold={res['n_gold']} "
        f"predicted={res['n_predicted']}"
    )
    for name, s in sorted(res["by_field"].items()):
        print(
            f"  {name:17s} tp={s['tp']:4d} fp={s['fp']:3d} fn={s['fn']:3d}  "
            f"P={pct(s['precision'])} R={pct(s['recall'])}"
        )
    for label, key in (("KEY(disc,growth,exit)", "key_fields"), ("OVERALL", "overall")):
        o = res[key]
        print(
            f"  {label:21s} tp={o['tp']} fp={o['fp']} fn={o['fn']} "
            f"P={pct(o['precision'])} [{pct(o['precision_lo'])},{pct(o['precision_hi'])}] "
            f"R={pct(o['recall'])}"
        )
    for k, v in res["attributes"].items():
        print(f"  attr {k:15s} {v['correct']}/{v['n']} = {pct(v['accuracy'])}")
    if args.errors:
        print("--- MISSED")
        for g in res["missed"]:
            why = [
                r["reason"]
                for r in run.rejected[g["accession"]]
                if r["field"] == g["field"]
                and r["start"] < g["span_end"]
                and g["span_start"] < r["end"]
            ]
            print(
                f"{g['accession']} {g['field']} {g['low']}-{g['high']} | {g['quote'][:80]} | {why}"
            )
        print("--- EXTRA")
        for p in res["extra"]:
            text = run.texts[p["accession"]]
            context = text[max(0, p["span_start"] - 80) : p["span_end"] + 40]
            tags = f"{p['advisor']}/{p['analysis']}/{p['subject']}"
            print(f"{p['accession']} {p['field']} {p['low']}-{p['high']} [{tags}] | {context!r}")
    if args.confusion:
        pairs, _, _ = match_facts(run.gold, run.predicted)
        table: dict[str, Counter] = defaultdict(Counter)
        for g, p in pairs:
            want = getattr(g, args.confusion)
            if want:
                table[g.accession][(want, getattr(p, args.confusion))] += 1
        for acc, counts in table.items():
            print(acc, dict(counts))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
