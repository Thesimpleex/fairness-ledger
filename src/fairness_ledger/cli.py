"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

DEFAULT_RAW = "data/raw"
DEFAULT_RELEASE = "data/release"
DEFAULT_CATALOG = f"{DEFAULT_RELEASE}/filings.csv"


def _years(text: str) -> range:
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


def _progress(console: Console) -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def _cmd_fetch(args: argparse.Namespace, console: Console) -> int:
    from .catalog import build_catalog, write_catalog
    from .edgar import DEFAULT_QUERIES, SecClient

    client = SecClient(args.raw_dir)
    with _progress(console) as bar:
        tasks: dict[str, int] = {}

        def progress(stage: str, done: int, total: int) -> None:
            key = stage.split(" ")[0]
            if key not in tasks:
                tasks[key] = bar.add_task(stage, total=total)
            bar.update(tasks[key], completed=done, total=total, description=stage)

        deals = build_catalog(
            client,
            _years(args.years),
            queries=args.query or DEFAULT_QUERIES,
            download=not args.no_download,
            workers=args.workers,
            progress=progress,
        )
    write_catalog(deals, args.catalog)
    n_filings = sum(len(d.filings) for d in deals)
    console.print(f"{n_filings} filings in {len(deals)} deals, catalog written to {args.catalog}")
    return 0


def _cmd_extract(args: argparse.Namespace, console: Console) -> int:
    from .catalog import read_catalog
    from .dataset import build_dataset, write_dataset

    deals = read_catalog(args.catalog)
    with _progress(console) as bar:
        task = bar.add_task("extracting", total=sum(len(d.filings) for d in deals))

        def progress(done: int, total: int) -> None:
            bar.update(task, completed=done, total=total)

        frame, rejections = build_dataset(
            deals, args.raw_dir, workers=args.workers, progress=progress
        )
    paths = write_dataset(frame, args.out)
    summary = {
        "n_filings": sum(len(d.filings) for d in deals),
        "n_deals": len(deals),
        "n_facts": len(frame),
        "facts_by_field": {k: int(v) for k, v in frame.groupby("field").size().items()},
        "rejections_by_reason": rejections,
    }
    (Path(args.out) / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    console.print(f"{len(frame)} facts from {summary['n_filings']} filings -> {paths[0].parent}")
    return 0


def _cmd_verify(args: argparse.Namespace, console: Console) -> int:
    from .dataset import load
    from .verify import verify_dataset

    frame = load(args.data_dir)
    report = verify_dataset(frame, args.raw_dir, limit=args.limit)
    table = Table(box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("check")
    table.add_column("facts", justify="right")
    table.add_row("facts checked", f"{report.checked:,}")
    table.add_row("provenance verified", f"{report.verified:,}")
    table.add_row("filings missing from cache", f"{report.missing_filings:,}")
    table.add_row("text hash mismatch", f"{report.hash_mismatch:,}")
    table.add_row("span / value mismatch", f"{report.span_mismatch:,}")
    console.print(table)
    return 0 if report.ok else 1


def _cmd_show(args: argparse.Namespace, console: Console) -> int:
    from .dataset import load
    from .show import find_deal, render_deal

    facts = load(args.data_dir)
    filings = load(args.data_dir, kind="filings")
    deal_id = find_deal(filings, args.query, facts)
    if deal_id is None:
        console.print(f"[red]no deal matches {args.query!r}[/red]")
        return 1
    render_deal(console, facts, filings, deal_id, width=args.snippet_width)
    return 0


def _cmd_report(args: argparse.Namespace, console: Console) -> int:
    from .dataset import load
    from .report import write_report

    path = write_report(load(args.data_dir), load(args.data_dir, kind="filings"), args.out)
    console.print(f"report written to {path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace, console: Console) -> int:
    from .evaluation import run_split

    try:
        run = run_split(Path(args.root), args.split)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    res = run.result
    console.print(
        Text(
            f"{args.split}: {res['n_filings']} filings, {res['n_gold']} gold facts, "
            f"{res['n_predicted']} predicted",
            style="dim",
        )
    )
    table = Table(box=box.SIMPLE_HEAD, header_style="bold")
    for col in ("field", "TP", "FP", "FN", "precision", "recall"):
        table.add_column(col, justify="right" if col != "field" else "left")
    for name, s in sorted(res["by_field"].items()):
        table.add_row(
            name,
            str(s["tp"]),
            str(s["fp"]),
            str(s["fn"]),
            f"{100 * s['precision']:.1f}%",
            f"{100 * s['recall']:.1f}%",
        )
    console.print(table)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fairness-ledger",
        description="Provenance-linked dataset of valuation work disclosed in merger proxies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="discover DEFM14A filings and cache them (SEC EDGAR)")
    fetch.add_argument("--years", default="2012-2025", help="filing years, e.g. 2012-2025")
    fetch.add_argument(
        "--query", action="append", help="full-text query (repeatable; default: built-in set)"
    )
    fetch.add_argument("--raw-dir", default=DEFAULT_RAW)
    fetch.add_argument("--catalog", default=DEFAULT_CATALOG)
    fetch.add_argument("--workers", type=int, default=4)
    fetch.add_argument("--no-download", action="store_true", help="only build the catalog")
    fetch.set_defaults(func=_cmd_fetch)

    extract = sub.add_parser("extract", help="extract facts from cached filings")
    extract.add_argument("--catalog", default=DEFAULT_CATALOG)
    extract.add_argument("--raw-dir", default=DEFAULT_RAW)
    extract.add_argument("--out", default=DEFAULT_RELEASE)
    extract.add_argument("--workers", type=int, default=None)
    extract.set_defaults(func=_cmd_extract)

    show = sub.add_parser("show", help="side-by-side analyses of one deal, with sources")
    show.add_argument("query", help="accession number, ticker or company name fragment")
    show.add_argument("--data-dir", default=DEFAULT_RELEASE)
    show.add_argument("--snippet-width", type=int, default=64)
    show.set_defaults(func=_cmd_show)

    report = sub.add_parser("report", help="self-contained HTML market-practice report")
    report.add_argument("--data-dir", default=DEFAULT_RELEASE)
    report.add_argument("--out", default="docs/report.html")
    report.set_defaults(func=_cmd_report)

    verify = sub.add_parser("verify", help="re-check every fact's span against the cached filing")
    verify.add_argument("--data-dir", default=DEFAULT_RELEASE)
    verify.add_argument("--raw-dir", default=DEFAULT_RAW)
    verify.add_argument("--limit", type=int, default=None, help="check only the first N facts")
    verify.set_defaults(func=_cmd_verify)

    evaluate = sub.add_parser("evaluate", help="score the extractor on the gold set")
    evaluate.add_argument("split", choices=["dev", "holdout"])
    evaluate.add_argument("--root", default=".", help="repository root (with gold/ and data/)")
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(highlight=False)
    try:
        return int(args.func(args, console))
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
