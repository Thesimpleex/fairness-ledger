"""Ten-year Treasury yield from FRED (series ``DGS10``) for the market-practice analysis."""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"


def fetch_dgs10(path: str | Path, user_agent: str | None = None) -> Path:
    """Download the DGS10 series into ``path`` (CSV with ``date,dgs10``)."""
    request = urllib.request.Request(
        FRED_URL, headers={"User-Agent": user_agent or "fairness-ledger"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8")
    frame = pd.read_csv(io.StringIO(raw))
    frame.columns = ["date", "dgs10"]
    frame["dgs10"] = pd.to_numeric(frame["dgs10"], errors="coerce")
    frame = frame.dropna()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_dgs10(path: str | Path) -> pd.Series:
    """Daily 10-year yield in percent, indexed by date."""
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date")["dgs10"].sort_index()


def yield_on(series: pd.Series, dates: pd.Series) -> np.ndarray:
    """Last observed yield on or before each date (NaN before the first observation)."""
    idx = series.index.searchsorted(pd.to_datetime(dates).to_numpy(), side="right") - 1
    values = series.to_numpy()
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out
