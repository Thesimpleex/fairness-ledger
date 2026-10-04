"""Compute every number quoted in the README and write ``docs/results.json``.

python scripts/run_analyses.py [--skip-verify] [--skip-gold]

Needs ``data/release/facts.parquet`` (``fairness-ledger extract``). Gold scores need the
cached filings in ``data/raw``; provenance verification re-reads all of them.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fairness_ledger import __version__, load
from fairness_ledger.analyses import (
    NOMINAL_GROWTH_BOUND_PCT,
    bank_disagreement,
    implied_growth_check,
    market_practice,
    strip_private,
)
from fairness_ledger.dataset import CORE_FIELDS
from fairness_ledger.evaluation import run_split
from fairness_ledger.gold import bootstrap_precision, match_facts
from fairness_ledger.market import load_dgs10
from fairness_ledger.report import advisor_table

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
DRAWS = 2000
KEY_FIELDS = ("discount_rate", "terminal_growth", "exit_multiple")

GORMSEN_HUBER = {
    "citation": (
        "Gormsen, N. J. and Huber, K. (2024), Corporate Discount Rates, NBER Working Paper "
        "31329 (June 2023, revised June 2024)."
    ),
    "statement": (
        "Within firms, a 1 percentage point increase in the perceived cost of capital raises "
        "discount rates by about 0.25 percentage points over 3 to 4 years and by about 0.7 "
        "percentage points over 7 to 11 years (paper, introduction)."
    ),
    "short_run_pp": 0.25,
    "long_run_pp": 0.7,
    "comparability": (
        "Different object: managers' within-firm discount rates against their perceived cost "
        "of capital on earnings calls, at multi-year horizons. This repository relates "
        "deal-level fairness-opinion discount rates to the ten-year Treasury yield at filing."
    ),
}


def gold_block(split: str) -> dict[str, Any]:
    run = run_split(ROOT, split)
    result = {k: v for k, v in run.result.items() if k not in ("missed", "extra")}
    pairs, _, extra = match_facts(run.gold, run.predicted)
    per_filing: dict[str, list[int]] = {a: [0, 0] for a in run.accessions}
    for g, _ in pairs:
        if g.field in KEY_FIELDS:
            per_filing[g.accession][0] += 1
    for p in extra:
        if p.field in KEY_FIELDS:
            per_filing[p.accession][1] += 1
    lo, hi = bootstrap_precision({a: (t, f) for a, (t, f) in per_filing.items()}, DRAWS, SEED)
    result["key_fields_precision_filing_bootstrap_95"] = [lo, hi]
    result["n_missed"] = len(run.result["missed"])
    result["n_extra"] = len(run.result["extra"])
    return result


def corpus_block(facts: pd.DataFrame, filings: pd.DataFrame) -> dict[str, Any]:
    core = facts[facts["field"].isin(CORE_FIELDS)]
    deals_with_core = core["deal_id"].nunique()
    summary_path = ROOT / "data" / "release" / "extraction_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    flags = Counter(f for cell in facts["flags"].fillna("") for f in cell.split(";") if f)
    return {
        "n_filings": int(filings["accession"].nunique()),
        "n_deals": int(filings["deal_id"].nunique()),
        "filing_years": [
            int(filings["file_date"].str[:4].min()),
            int(filings["file_date"].str[:4].max()),
        ],
        "n_facts": len(facts),
        "n_core_facts": len(core),
        "n_deals_with_facts": int(facts["deal_id"].nunique()),
        "n_deals_with_core_facts": int(deals_with_core),
        "facts_by_field": {k: int(v) for k, v in facts.groupby("field").size().items()},
        "facts_by_tier": {k: int(v) for k, v in facts.groupby("tier").size().items()},
        "facts_by_year": {int(k): int(v) for k, v in facts.groupby("year").size().items()},
        "advisor_attribution_share_unattributed_core": float(
            (core["advisor"] == "unattributed").mean()
        ),
        "soft_flag_counts": dict(flags),
        "extraction_rejections_by_reason": summary.get("rejections_by_reason", {}),
    }


def nuvasive_example(facts: pd.DataFrame) -> dict[str, Any]:
    sub = facts[facts["company"].str.contains("NUVASIVE", case=False)]
    if sub.empty:
        return {}
    rows = sub[
        sub["field"].isin(["discount_rate", "terminal_growth", "exit_multiple", "implied_growth"])
        & (sub["analysis"] == "dcf")
    ].sort_values(["advisor", "span_start"])
    return {
        "deal_id": str(sub["deal_id"].iloc[0]),
        "facts": [
            {
                "advisor": r.advisor,
                "field": r.field,
                "low": r.low,
                "high": r.high,
                "subject": r.subject,
            }
            for r in rows.itertuples()
        ],
    }


def compute(facts: pd.DataFrame, filings: pd.DataFrame, yields: pd.Series) -> dict[str, Any]:
    """All analysis objects, including the private data frames used by the figures."""
    return {
        "disagreement": bank_disagreement(facts, DRAWS, SEED),
        "disagreement_single": bank_disagreement(facts, DRAWS, SEED, single_only=True),
        "implied": implied_growth_check(facts),
        "market": market_practice(facts, yields, DRAWS, SEED),
        "advisors": advisor_table(facts),
    }


def versions() -> dict[str, str]:
    names = ("numpy", "scipy", "pandas", "pyarrow", "matplotlib", "rich")
    out = {n: metadata.version(n) for n in names}
    out["python"] = platform.python_version()
    out["fairness_ledger"] = __version__
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "results.json")
    args = parser.parse_args()

    release = ROOT / "data" / "release"
    facts, filings = load(release), load(release, kind="filings")
    yields = load_dgs10(release / "dgs10.csv")
    computed = compute(facts, filings, yields)
    market = computed["market"]
    by_year = market["_by_year"]

    old = json.loads(args.out.read_text()) if args.out.exists() else {}
    results: dict[str, Any] = {
        "generated": date.today().isoformat(),
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "versions": versions(),
        "data_vintage": {
            "edgar_catalog_built": old.get("data_vintage", {}).get("edgar_catalog_built")
            or date.today().isoformat(),
            "dgs10_last_observation": str(yields.index.max().date()),
            "dgs10_source": "FRED series DGS10",
        },
        "corpus": corpus_block(facts, filings),
        "analyses": {
            "nominal_growth_bound_pct": NOMINAL_GROWTH_BOUND_PCT,
            "bank_disagreement": strip_private(computed["disagreement"]),
            "bank_disagreement_single_range_only": strip_private(computed["disagreement_single"]),
            "implied_growth": strip_private(computed["implied"]),
            "market_practice": strip_private(market),
            "market_practice_by_year": json.loads(by_year.to_json(orient="records")),
            "advisor_table": json.loads(computed["advisors"].to_json(orient="records")),
            "nuvasive_example": nuvasive_example(facts),
            "gormsen_huber_reference": GORMSEN_HUBER,
        },
    }
    if not args.skip_gold:
        results["gold"] = {
            "dev": gold_block("dev"),
            "holdout": gold_block("holdout"),
            "holdout_frozen_digest": (ROOT / "gold" / "holdout.sha256").read_text().split()[0],
        }
    elif "gold" in old:
        results["gold"] = old["gold"]
    if not args.skip_verify:
        from fairness_ledger.verify import verify_dataset

        report = verify_dataset(facts, ROOT / "data" / "raw")
        results["verification"] = {
            "checked": report.checked,
            "verified": report.verified,
            "missing_filings": report.missing_filings,
            "hash_mismatch": report.hash_mismatch,
            "span_mismatch": report.span_mismatch,
        }
    elif "verification" in old:
        results["verification"] = old["verification"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True, default=float) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
