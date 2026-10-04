"""Discovery, caching and rate limiting, tested against a stubbed HTTP layer (no network)."""

from __future__ import annotations

import gzip
import json
import time
from datetime import date

import pytest

from fairness_ledger import edgar
from fairness_ledger.deals import Filing, clean_company_name, group_deals
from fairness_ledger.edgar import Hit, RateLimiter, SecClient, user_agent_from_env


def _filing(acc, cik, day):
    return Filing(acc, "p.htm", "DEFM14A", day, cik, "Co")


def test_group_deals_merges_amendments_within_window_only():
    filings = [
        _filing("1", "100", "2020-01-10"),
        _filing("2", "100", "2020-02-20"),  # revised proxy, same deal
        _filing("3", "100", "2021-06-01"),  # a different transaction of the same company
        _filing("4", "200", "2020-01-12"),
    ]
    deals = group_deals(filings)
    assert [d.accessions for d in deals] == [["1", "2"], ["4"], ["3"]]
    assert deals[0].deal_id == "1"
    with pytest.raises(ValueError):
        group_deals(filings, window_days=0)


def test_clean_company_name():
    assert clean_company_name("ALTRA INDUSTRIAL MOTION CORP.  (AIMC)  (CIK 0001374535)") == (
        "ALTRA INDUSTRIAL MOTION CORP."
    )


def test_user_agent_comes_from_environment(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        user_agent_from_env()
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.org")
    assert user_agent_from_env() == "Jane Doe jane@example.org"


def test_rate_limiter_spaces_calls_and_caps_rate():
    with pytest.raises(ValueError):
        RateLimiter(11)
    limiter = RateLimiter(10)
    start = time.monotonic()
    for _ in range(6):
        limiter.wait()
    assert time.monotonic() - start >= 5 / 10 - 0.02


def _source(accession, form="DEFM14A", file_type="DEFM14A", name="a.htm"):
    return {
        "_id": f"{accession}:{name}",
        "_source": {
            "form": form,
            "file_type": file_type,
            "file_date": "2020-03-01",
            "ciks": ["0000000100"],
            "display_names": ["Co (CIK 0000000100)"],
            "sics": ["3560"],
        },
    }


class FakeSec(SecClient):
    """Serves search pages from memory; counts downloads."""

    def __init__(self, cache_dir, hits_for):
        super().__init__(cache_dir, user_agent="test test@example.org")
        self.hits_for = hits_for
        self.calls = 0

    def _download(self, url, attempts=6):
        self.calls += 1
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(url).query)
        start, end = date.fromisoformat(q["startdt"][0]), date.fromisoformat(q["enddt"][0])
        offset = int(q["from"][0])
        pool = [h for h in self.hits_for if start <= date.fromisoformat(h[0]) <= end]
        page = [_source(acc) for _, acc in pool][offset : offset + 100]
        return json.dumps({"hits": {"total": {"value": len(pool)}, "hits": page}}).encode()


def test_search_window_paginates_caches_and_discovers(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "WINDOW_HIT_CAP", 150)
    data = [("2020-03-01", f"0000-20-{i:06d}") for i in range(130)]
    data += [("2020-09-01", f"0000-20-{i:06d}") for i in range(130, 140)]
    sec = FakeSec(tmp_path, data)
    hits = sec.search_window("q", "DEFM14A", date(2020, 1, 1), date(2020, 12, 31))
    assert len(hits) == 140  # 130 in the first half needed a second page, window split by cap
    before = sec.calls
    again = sec.search_window("q", "DEFM14A", date(2020, 1, 1), date(2020, 12, 31))
    assert again == hits and sec.calls == before  # everything came from the gzip cache
    found = sec.discover([2020], queries=["q", "q2"])
    assert len({h.accession for h in found}) == 140


def test_incomplete_window_raises(tmp_path):
    class Short(FakeSec):
        def _download(self, url, attempts=6):
            body = json.loads(super()._download(url))
            body["hits"]["total"]["value"] += 5
            return json.dumps(body).encode()

    sec = Short(tmp_path, [("2020-03-01", "0000-20-000001")])
    with pytest.raises(RuntimeError, match="incomplete search window"):
        sec.search_window("q", "DEFM14A", date(2020, 1, 1), date(2020, 12, 31))


def test_fetch_filing_is_cached_and_stored_compressed(tmp_path):
    class One(SecClient):
        calls = 0

        def _download(self, url, attempts=6):
            One.calls += 1
            return b"<html>" + b"x" * 2000 + b"</html>"

    sec = One(tmp_path, user_agent="t t@example.org")
    hit = Hit(
        "0000-20-000001", "a.htm", "DEFM14A", "DEFM14A", "2020-03-01", ("0000000100",), (), ()
    )
    path = sec.fetch_filing(hit)
    assert path.suffix == ".gz" and gzip.decompress(path.read_bytes()).startswith(b"<html>")
    sec.fetch_filing(hit)
    assert One.calls == 1
    assert edgar.read_cached_filing(path).endswith(b"</html>")


def test_hit_main_document_detection():
    main = Hit("a", "x.htm", "DEFM14A", "DEFM14A", "d", (), (), ())
    exhibit = Hit("a", "ex.htm", "DEFM14A", "EX-99", "d", (), (), ())
    assert main.is_main_document and not exhibit.is_main_document
