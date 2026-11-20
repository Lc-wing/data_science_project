#!/usr/bin/env python3
"""Milestone M3 CLI: leak-free spatial-block baselines for H3 r9.

Reads ``data/frozen/cells_base_res9.gpkg``, ``spatial_folds.csv``, and
``feature_manifest.json``. Writes ``outputs/metrics/phase_a_ml_baselines.json``,
``outputs/tables/ml_cv_results.csv``, and ``outputs/tables/ml_oof_predictions.csv``.

The target is ``has_ev_charger``. Features are the manifest ``ml_features`` only.
Metrics describe association under the frozen geographic blocks. They do not
identify a causal effect and they are not a recommendation for where to site
chargers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from macao_equity.config import (  # noqa: E402
    DATA_FROZEN_DIR,
    OUTPUTS_METRICS_DIR,
    RANDOM_SEED,
)
from macao_equity.ml.baselines import (  # noqa: E402
    MODEL_NAMES,
    load_baseline_inputs,
    run_spatial_block_baselines,
    write_baseline_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit dummy, logistic, and random-forest baselines on H3 r9 "
            "with the frozen 5 spatial blocks."
        )
    )
    parser.add_argument(
        "--cells",
        type=Path,
        default=DATA_FROZEN_DIR / "cells_base_res9.gpkg",
        help="Populated H3 resolution-9 cell table.",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=DATA_FROZEN_DIR / "spatial_folds.csv",
        help="Frozen cell_id to spatial_block map.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_FROZEN_DIR / "feature_manifest.json",
        help="Leak-free ml_features whitelist.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=OUTPUTS_METRICS_DIR / "phase_a_ml_baselines.json",
        help="JSON metric contract path.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=OUTPUTS_METRICS_DIR.parent / "tables",
        help="Directory for ml_cv_results.csv and ml_oof_predictions.csv.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    frame, features, _manifest = load_baseline_inputs(args.cells, args.folds, args.manifest)
    report = run_spatial_block_baselines(frame, feature_columns=features, seed=args.seed)
    paths = write_baseline_artifacts(report, args.metrics_out, args.tables_dir)
    payload = report.to_json_dict()
    print(
        f"n_cells={payload['n_cells']} n_pos={payload['n_pos']} "
        f"prevalence={payload['prevalence']:.6f}"
    )
    for name in MODEL_NAMES:
        block = payload["models"][name]
        print(
            f"{name}: oof_roc_auc={block['oof_roc_auc']:.4f} "
            f"oof_pr_auc={block['oof_pr_auc']:.4f} "
            f"oof_brier={block['oof_brier']:.4f} "
            f"oof_top20_hit_rate={block['oof_top20_hit_rate']:.4f}"
        )
    print(f"wrote {paths['metrics']}")
    print(f"wrote {paths['cv']}")
    print(f"wrote {paths['oof']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
