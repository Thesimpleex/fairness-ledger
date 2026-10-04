"""Freeze the holdout labels: write their combined SHA-256 to ``gold/holdout.sha256``.

Run once, after all holdout labels are written and before the first evaluation on them.
``scripts/evaluate_gold.py holdout`` refuses to run if the labels no longer match.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from fairness_ledger.gold import labels_digest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "gold" / "manifest.json").read_text())
    paths = [
        ROOT / "gold" / "labels" / f"{m['accession']}.txt"
        for m in manifest
        if m["split"] == "holdout"
    ]
    digest = labels_digest(paths)
    lines = [f"{digest}  holdout labels ({len(paths)} files), frozen {date.today().isoformat()}"]
    (ROOT / "gold" / "holdout.sha256").write_text("\n".join(lines) + "\n")
    print(lines[0])


if __name__ == "__main__":
    main()
