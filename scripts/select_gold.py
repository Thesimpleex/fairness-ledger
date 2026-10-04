"""Select the gold-set filings: stratified by era and dominant advisor, seeded.

Dev filings come from the development pool listed in ``gold/dev_pool.txt`` (the filings
used while the grammar was written); holdout filings are drawn from all other filings and
were never opened during development. The result is ``gold/manifest.json``.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fairness_ledger.advisors import find_mentions
from fairness_ledger.catalog import read_catalog
from fairness_ledger.edgar import read_cached_filing

ROOT = Path(__file__).resolve().parents[1]
ERAS = [(2012, 2014), (2015, 2017), (2018, 2020), (2021, 2022), (2023, 2025)]
PER_ERA = {"dev": 5, "holdout": 6}
SEED = 20261004


def dominant_advisor(args: tuple[str, str]) -> tuple[str, str]:
    accession, filename = args
    path = ROOT / "data" / "raw" / "filings" / accession / f"{filename}.gz"
    text = read_cached_filing(path).decode("latin-1")
    counts = Counter(m.house for m in find_mentions(text))
    return accession, counts.most_common(1)[0][0] if counts else "unattributed"


def pick(candidates: list[dict], quota: int, rng: random.Random) -> list[dict]:
    by_advisor: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_advisor[c["dominant_advisor"]].append(c)
    for group in by_advisor.values():
        rng.shuffle(group)
    order = sorted(by_advisor, key=lambda a: (-len(by_advisor[a]), a))
    chosen: list[dict] = []
    while len(chosen) < quota and any(by_advisor.values()):
        for advisor in order:
            if by_advisor[advisor] and len(chosen) < quota:
                chosen.append(by_advisor[advisor].pop())
    return chosen


def main() -> None:
    deals = read_catalog(ROOT / "data" / "release" / "filings.csv")
    filings = [f for d in deals for f in d.filings]
    pool = set((ROOT / "gold" / "dev_pool.txt").read_text().split())
    with ProcessPoolExecutor() as ex:
        dom = dict(
            ex.map(dominant_advisor, [(f.accession, f.filename) for f in filings], chunksize=16)
        )
    rng = random.Random(SEED)
    manifest = []
    for split in ("dev", "holdout"):
        for lo, hi in ERAS:
            cands = [
                {
                    "accession": f.accession,
                    "company": f.company,
                    "file_date": f.file_date,
                    "era": f"{lo}-{hi}",
                    "dominant_advisor": dom[f.accession],
                    "split": split,
                }
                for f in filings
                if lo <= f.year <= hi and ((f.accession in pool) == (split == "dev"))
            ]
            manifest += pick(sorted(cands, key=lambda c: c["accession"]), PER_ERA[split], rng)
    (ROOT / "gold" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(manifest), Counter(m["split"] for m in manifest))
    print(Counter(m["dominant_advisor"] for m in manifest).most_common())


if __name__ == "__main__":
    main()
