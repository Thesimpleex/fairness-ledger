"""Minimal SEC EDGAR client: rate-limited, cached, with full-text-search discovery.

All network access of the package lives here and in :mod:`fairness_ledger.market`.
Responses are cached gzip-compressed below ``data/raw`` so every later step runs
offline. The SEC requires a descriptive ``User-Agent``; it is read from the
environment variable ``SEC_USER_AGENT`` (``"Name email"``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{filename}"
MAX_REQUESTS_PER_SECOND = 8.0
PAGE_SIZE = 100
WINDOW_HIT_CAP = 1000
DEFAULT_QUERIES = (
    '"discounted cash flow"',
    '"discounted cash flows"',
    '"discount rate"',
    '"discount rates"',
    '"discount rates ranging from"',
    '"weighted average cost of capital"',
    '"perpetuity growth"',
)


class RateLimiter:
    """Thread-safe limiter that spaces calls by at least ``1 / rate`` seconds."""

    def __init__(self, rate: float = MAX_REQUESTS_PER_SECOND) -> None:
        if not 0 < rate <= 10:
            raise ValueError("rate must be in (0, 10] requests per second (SEC fair-access limit)")
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._interval
        if delay > 0:
            time.sleep(delay)


def user_agent_from_env() -> str:
    """Return the declared SEC user agent or raise with instructions."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError(
            "Set SEC_USER_AGENT to 'Name email' (the SEC requires a declared User-Agent), "
            "for example: export SEC_USER_AGENT='Jane Doe jane@example.org'"
        )
    return ua


@dataclass(frozen=True)
class Hit:
    """One full-text-search hit: a document inside a filing."""

    accession: str
    filename: str
    form: str
    file_type: str
    file_date: str
    ciks: tuple[str, ...]
    names: tuple[str, ...]
    sics: tuple[str, ...]

    @property
    def is_main_document(self) -> bool:
        return self.file_type == self.form and self.filename.lower().endswith((".htm", ".html"))


def _hit_from_source(doc_id: str, source: dict[str, Any]) -> Hit:
    accession, filename = doc_id.split(":", 1)
    return Hit(
        accession=accession,
        filename=filename,
        form=source.get("form", ""),
        file_type=source.get("file_type", ""),
        file_date=source.get("file_date", ""),
        ciks=tuple(source.get("ciks") or ()),
        names=tuple(source.get("display_names") or ()),
        sics=tuple(source.get("sics") or ()),
    )


class SecClient:
    """Cached, rate-limited HTTP access to SEC endpoints.

    Parameters
    ----------
    cache_dir
        Root of the gitignored download cache (default ``data/raw``).
    user_agent
        Declared user agent; defaults to ``SEC_USER_AGENT``.
    rate
        Maximum requests per second (hard-capped at 10 by the SEC).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/raw",
        user_agent: str | None = None,
        rate: float = MAX_REQUESTS_PER_SECOND,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._ua = user_agent
        self._limiter = RateLimiter(rate)

    @property
    def user_agent(self) -> str:
        if self._ua is None:
            self._ua = user_agent_from_env()
        return self._ua

    def _download(self, url: str, attempts: int = 6) -> bytes:
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip"}
        last: Exception | None = None
        for attempt in range(attempts):
            self._limiter.wait()
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (403, 429, 500, 502, 503, 504):
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"giving up on {url}: {last}")

    def cached_get(self, url: str, path: Path, validate: Callable[[bytes], bool] | None = None):
        """Return the body of ``url``, downloading into ``path`` (gzip) if absent."""
        if path.exists():
            return gzip.decompress(path.read_bytes())
        for _ in range(4):
            body = self._download(url)
            if validate is None or validate(body):
                break
            time.sleep(2.0)
        else:
            raise RuntimeError(f"response for {url} failed validation")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(gzip.compress(body, compresslevel=6))
        tmp.replace(path)
        return body

    # -- full-text search -------------------------------------------------

    def _efts_page(self, query: str, form: str, start: date, end: date, offset: int) -> dict:
        params = {
            "q": query,
            "forms": form,
            "dateRange": "custom",
            "startdt": start.isoformat(),
            "enddt": end.isoformat(),
            "from": offset,
        }
        url = EFTS_URL + "?" + urllib.parse.urlencode(params)
        key = hashlib.sha1(url.encode()).hexdigest()[:20]
        path = self.cache_dir / "efts" / f"{key}.json.gz"

        def valid(body: bytes) -> bool:
            try:
                return "hits" in json.loads(body, strict=False)
            except ValueError:
                return False

        return json.loads(self.cached_get(url, path, valid), strict=False)

    def search_window(self, query: str, form: str, start: date, end: date) -> list[Hit]:
        """All hits of ``query`` for one form in ``[start, end]``, split to stay below the cap.

        The window is bisected while the reported total reaches ``WINDOW_HIT_CAP``;
        the number of hits actually collected must equal the reported total.
        """
        first = self._efts_page(query, form, start, end, 0)
        total = int(first["hits"]["total"]["value"])
        if total >= WINDOW_HIT_CAP and start < end:
            mid = start + (end - start) // 2
            return self.search_window(query, form, start, mid) + self.search_window(
                query, form, mid + timedelta(days=1), end
            )
        raw = list(first["hits"]["hits"])
        offset = len(raw)
        while offset < total:
            page = self._efts_page(query, form, start, end, offset)["hits"]["hits"]
            if not page:
                break
            raw.extend(page)
            offset += len(page)
        if len(raw) != total:
            raise RuntimeError(
                f"incomplete search window {start}..{end}: collected {len(raw)} of {total} hits"
            )
        return [_hit_from_source(h["_id"], h["_source"]) for h in raw]

    def discover(
        self,
        years: Iterable[int],
        queries: Sequence[str] = DEFAULT_QUERIES,
        form: str = "DEFM14A",
        progress: Callable[[int, int], None] | None = None,
    ) -> list[Hit]:
        """Main-document hits of ``form`` filings matching any of ``queries``, per year.

        The union over several overlapping valuation phrases is used because the search
        matches exact tokens (``discounted cash flow`` does not match ``flows``).
        """
        hits: dict[tuple[str, str], Hit] = {}
        for year in years:
            for query in queries:
                for hit in self.search_window(query, form, date(year, 1, 1), date(year, 12, 31)):
                    if hit.form == form and hit.is_main_document:
                        hits[(hit.accession, hit.filename)] = hit
            if progress:
                progress(year, len(hits))
        return sorted(hits.values(), key=lambda h: (h.file_date, h.accession, h.filename))

    # -- filings and company metadata ------------------------------------

    def filing_path(self, accession: str, filename: str) -> Path:
        return self.cache_dir / "filings" / accession / f"{filename}.gz"

    def fetch_filing(self, hit: Hit) -> Path:
        """Download one filing document into the cache and return its path."""
        path = self.filing_path(hit.accession, hit.filename)
        if not path.exists():
            cik = str(int(hit.ciks[0])) if hit.ciks else "0"
            url = ARCHIVE_URL.format(
                cik=cik, acc_nodash=hit.accession.replace("-", ""), filename=hit.filename
            )
            self.cached_get(url, path, lambda body: len(body) > 1000)
        return path

    def submissions(self, cik: str | int) -> dict[str, Any]:
        """Company record from the submissions API (name, SIC, tickers, exchanges)."""
        number = int(cik)
        path = self.cache_dir / "submissions" / f"CIK{number:010d}.json.gz"
        body = self.cached_get(SUBMISSIONS_URL.format(cik=number), path)
        record = json.loads(body, strict=False)
        record.pop("filings", None)
        return record


def read_cached_filing(path: str | Path) -> bytes:
    """Return the decompressed bytes of a cached filing."""
    return gzip.decompress(Path(path).read_bytes())
