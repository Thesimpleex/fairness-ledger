"""fairness-ledger: provenance-linked valuation facts from merger proxies."""

from .consistency import implied_perpetuity_growth
from .dataset import load
from .document import html_to_document
from .extract import extract_document
from .provenance import verify_span

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "extract_document",
    "html_to_document",
    "implied_perpetuity_growth",
    "load",
    "verify_span",
]
