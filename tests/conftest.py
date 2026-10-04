"""Shared helpers: build a small HTML filing from a committed text excerpt."""

from __future__ import annotations

import html
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


def excerpt_html(name: str) -> str:
    """HTML for ``tests/data/<name>.txt``: ``## `` lines become bold headings, others paragraphs."""
    parts: list[str] = []
    for line in (DATA / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("# source"):
            continue
        if line.startswith("## "):
            parts.append(f"<p><b>{html.escape(line[3:])}</b></p>")
        else:
            parts.append(f"<p>{html.escape(line)}</p>")
    return "<html><body>" + "".join(parts) + "</body></html>"


@pytest.fixture
def excerpt():
    return excerpt_html
