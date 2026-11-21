"""Exploratory trees stay outside the pre-specified Route A and Route B outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from macao_equity.config import OUTPUTS_METRICS_DIR
from macao_equity.phase_b.exploratory_trees import (
    MODEL_NAMES,
    run_exploratory_trees,
    write_exploratory_artifacts,
)

FEATURES = ["pop_total", "elderly_share"]


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for block in range(5):
        for i in range(8):
            rows.append(
                {
                    "pop_total": float(rng.normal()),
                    "elderly_share": float(rng.random()),
                    "spatial_block": block,
                    "has_ev_charger": int(i < 2),
                    "cell_id": f"{block}-{i}",
                }
            )
    return pd.DataFrame(rows)


def test_inner_search_does_not_see_the_outer_block(monkeypatch) -> None:
    seen: list[set[int]] = []
    import macao_equity.phase_b.exploratory_trees as trees

    real = trees._fit_one

    def spy(name, x_train, y_train, blocks_train, seed, **kwargs):
        seen.append(set(np.unique(blocks_train).tolist()))
        return real(name, x_train, y_train, blocks_train, seed, **kwargs)

    monkeypatch.setattr(trees, "_fit_one", spy)
    tiny = {
        name: {key: (values[0],) for key, values in grid.items()}
        for name, grid in trees.GRIDS.items()
    }
    run_exploratory_trees(
        _frame(),
        feature_columns=FEATURES,
        seed=42,
        n_estimators=5,
        grids=tiny,
    )
    assert len(seen) == 15
    for index, blocks in enumerate(seen):
        assert index % 5 not in blocks


def test_writer_refuses_protected_outputs(tmp_path: Path) -> None:
    summary = {
        "rows": [],
        "label": "exploratory",
        "outside_prespecified_plan": True,
        "seed": 42,
        "feature_columns": FEATURES,
        "inner_tuning": "test",
        "published_random_forest": {},
        "models": {},
    }
    try:
        write_exploratory_artifacts(
            summary,
            metrics_path=tmp_path / "phase_b_comparison.csv",
            markdown_path=tmp_path / "ok.md",
        )
    except RuntimeError as exc:
        assert "comparison" in str(exc)
    else:
        raise AssertionError("comparison table should be protected")


def test_committed_exploratory_table_is_labelled() -> None:
    path = OUTPUTS_METRICS_DIR / "phase_b_exploratory_trees.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["outside_prespecified_plan"] is True
    assert set(payload["models"]) == set(MODEL_NAMES)
    for name in MODEL_NAMES:
        assert payload["models"][name]["label"] == "exploratory"
    forest = payload["published_random_forest"]["mean_roc_auc"]
    assert forest == 0.77283316
