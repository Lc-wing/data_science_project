#!/usr/bin/env python3
"""Exploratory XGBoost, LightGBM, and histogram gradient boosting.

Outside the pre-specified plan. Uses the frozen spatial blocks and ml_features.
Does not rewrite Route A, Route B, run 1, or run 2 outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from macao_equity.config import DATA_FROZEN_DIR, OUTPUTS_METRICS_DIR  # noqa: E402
from macao_equity.ml.baselines import load_baseline_inputs  # noqa: E402
from macao_equity.phase_b.exploratory_trees import (  # noqa: E402
    assert_exploratory_features,
    run_exploratory_trees,
    summarize,
    write_exploratory_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    del argv
    frame, features, manifest = load_baseline_inputs(
        DATA_FROZEN_DIR / "cells.gpkg",
        DATA_FROZEN_DIR / "spatial_folds.csv",
        DATA_FROZEN_DIR / "feature_manifest.json",
    )
    assert_exploratory_features(features, manifest)
    report = run_exploratory_trees(frame, feature_columns=features, seed=42)
    phase_a = json.loads(
        (OUTPUTS_METRICS_DIR / "phase_a_ml_baselines.json").read_text(encoding="utf-8")
    )
    summary = summarize(report, phase_a)
    paths = write_exploratory_artifacts(
        summary,
        metrics_path=OUTPUTS_METRICS_DIR / "phase_b_exploratory_trees.json",
        markdown_path=OUTPUTS_METRICS_DIR.parent / "tables" / "phase_b_exploratory_trees.md",
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
