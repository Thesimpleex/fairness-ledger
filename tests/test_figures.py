from __future__ import annotations

import numpy as np
import pandas as pd

from fairness_ledger import figures
from fairness_ledger.viz import save_figure

TEXT = {"kicker": "Exhibit 1", "title": "A finding stated as a sentence", "source": "Source: test"}


def _by_year():
    return pd.DataFrame(
        {
            "year": np.arange(2012, 2020),
            "discount_median": np.linspace(10, 8, 8),
            "discount_q25": np.linspace(9, 7, 8),
            "discount_q75": np.linspace(11, 9, 8),
            "growth_median": np.full(8, 2.5),
            "growth_n": np.full(8, 5),
            "y10_mean": np.linspace(2, 2.5, 8),
        }
    )


def test_rates_exhibit_renders_in_both_themes(tmp_path):
    by_year = _by_year()
    paths = save_figure(lambda p: figures.fig_rates_by_year(by_year, p, **TEXT), tmp_path / "r")
    assert sorted(path.name for path in paths) == ["r-dark.png", "r-light.png"]
    assert all(path.stat().st_size > 5_000 for path in paths)


def test_disagreement_and_implied_growth_exhibits_render():
    rng = np.random.default_rng(0)
    pairs = pd.DataFrame({"abs_diff": rng.gamma(2.0, 0.5, 80)})
    units = pd.DataFrame({"covered": [True] * 40, "implied_high": rng.normal(3.0, 1.5, 40)})
    png = figures.render_png(lambda p: figures.fig_disagreement(pairs, p, **TEXT), "dark", dpi=40)
    assert png.startswith(b"\x89PNG")
    png = figures.render_png(
        lambda p: figures.fig_implied_growth(units, 4.0, p, **TEXT), "light", dpi=40
    )
    assert png.startswith(b"\x89PNG")


def test_gold_exhibit_renders_with_missing_fields():
    row = {
        "precision": 0.9,
        "precision_lo": 0.8,
        "precision_hi": 0.95,
        "recall": 0.8,
        "recall_lo": 0.7,
        "recall_hi": 0.9,
    }
    res = {"by_field": {"discount_rate": row, "fee_total": dict(row, precision=float("nan"))}}
    png = figures.render_png(
        lambda p: figures.fig_gold_scores(res, res, p, **TEXT), "light", dpi=40
    )
    assert png.startswith(b"\x89PNG")
