from __future__ import annotations

import gzip
import io

import pytest
from rich.console import Console

from conftest import excerpt_html
from fairness_ledger.catalog import write_catalog
from fairness_ledger.cli import build_parser, main
from fairness_ledger.dataset import build_dataset, write_dataset
from fairness_ledger.deals import Deal, Filing
from fairness_ledger.show import find_deal, fmt_range, render_deal


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("rel")
    raw = root / "raw"
    filings, deals = [], []
    for accession, name, company, ticker in [
        ("0001374535-22-000001", "goldman_altra", "Altra Industrial Motion Corp.", "AIMC"),
        ("0000049615-12-000001", "lazard_chenergy", "CH Energy Group, Inc.", "CHG"),
    ]:
        path = raw / "filings" / accession / "p.htm.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(gzip.compress(excerpt_html(name).encode()))
        f = Filing(
            accession,
            "p.htm",
            "DEFM14A",
            "2022-10-31",
            accession[:10],
            company,
            "3560",
            "",
            (ticker,),
            company,
        )
        filings.append(f)
        deals.append(Deal(accession, [f]))
    frame, _ = build_dataset(deals, raw, workers=1)
    out = root / "release"
    write_dataset(frame, out)
    write_catalog(deals, out / "filings.csv")
    return out, raw


def test_fmt_range():
    assert fmt_range(7.5, 9.5, "pct") == "7.5–9.5%"
    assert fmt_range(9.0, 9.0, "pct") == "9%"
    assert fmt_range(10.0, 12.0, "x") == "10–12x"
    assert fmt_range(21.0, 31.5, "usd") == "$21.00–$31.50"


def test_find_deal_by_accession_ticker_and_name(release):
    from fairness_ledger import load

    out, _ = release
    filings = load(out, kind="filings")
    facts = load(out)
    assert find_deal(filings, "0001374535-22-000001", facts) == "0001374535-22-000001"
    assert find_deal(filings, "aimc", facts) == "0001374535-22-000001"
    assert find_deal(filings, "ch energy", facts) == "0000049615-12-000001"
    assert find_deal(filings, "no such company", facts) is None


def test_render_deal_shows_every_analysis_and_its_source_sentence(release):
    from fairness_ledger import load

    out, _ = release
    console = Console(file=io.StringIO(), width=110, highlight=False, force_terminal=False)
    render_deal(console, load(out), load(out, kind="filings"), "0001374535-22-000001")
    text = console.file.getvalue()
    assert "Altra Industrial Motion Corp." in text
    assert "Target standalone" in text
    assert "10–11%" in text
    assert "Goldman Sachs" in text
    assert "discount rates" in text  # the verbatim sentence under the table


def test_cli_show_extract_verify_end_to_end(release, capsys):
    out, raw = release
    assert main(["show", "AIMC", "--data-dir", str(out)]) == 0
    assert "Goldman Sachs" in capsys.readouterr().out
    assert main(["show", "zzz", "--data-dir", str(out)]) == 1
    assert main(["verify", "--data-dir", str(out), "--raw-dir", str(raw)]) == 0
    assert "provenance verified" in capsys.readouterr().out


def test_cli_extract_writes_release_files(release, tmp_path, capsys):
    out, raw = release
    target = tmp_path / "again"
    code = main(
        [
            "extract",
            "--catalog",
            str(out / "filings.csv"),
            "--raw-dir",
            str(raw),
            "--out",
            str(target),
            "--workers",
            "1",
        ]
    )
    assert code == 0
    assert (target / "facts.parquet").exists()
    assert (target / "facts.csv").exists()
    assert (target / "extraction_summary.json").exists()


def test_cli_missing_sec_user_agent_is_a_clean_error(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    code = main(
        [
            "fetch",
            "--years",
            "2020",
            "--raw-dir",
            str(tmp_path),
            "--catalog",
            str(tmp_path / "c.csv"),
        ]
    )
    assert code == 1
    assert "SEC_USER_AGENT" in capsys.readouterr().out


def test_parser_knows_the_documented_commands():
    parser = build_parser()
    extra = {"show": ["x"], "evaluate": ["dev"]}
    for command in ("fetch", "extract", "show", "report", "verify", "evaluate"):
        assert parser.parse_args([command, *extra.get(command, [])]).command == command
