#!/usr/bin/env python3
"""Train the pre-specified Route A GNNs on the frozen spatial blocks.

Reads ``data/frozen/cells.gpkg``, ``graph_edges.csv``, ``spatial_folds.csv``,
``feature_manifest.json``, and ``cell_poi_texts.csv``. Writes Route A metrics
and the comparison table. Route B rows stay marked not-run. This script does
not download Qwen3-Embedding-4B or BGE-M3.

Scores describe association under the frozen blocks. They are not a causal
effect and they are not a charger-siting recommendation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from macao_equity.config import DATA_FROZEN_DIR, OUTPUTS_METRICS_DIR  # noqa: E402
from macao_equity.phase_b.data import load_phase_b_bundle  # noqa: E402
from macao_equity.phase_b.route_a import run_route_a, write_route_a_artifacts  # noqa: E402
from macao_equity.phase_b.spec import PROTOCOLS  # noqa: E402
from macao_equity.phase_b.splits import make_fold_split  # noqa: E402


def _assembly(bundle) -> dict:
    summary = bundle.assembly_summary()
    fold_edges = []
    for fold in range(5):
        for protocol in PROTOCOLS:
            split = make_fold_split(
                bundle.blocks,
                bundle.src,
                bundle.dst,
                fold=fold,
                protocol=protocol,
                edge_weight=bundle.edge_weight,
            )
            fold_edges.append(
                {
                    "fold": fold,
                    "protocol": protocol,
                    "n_train_edges": int(split.train_src.shape[0]),
                    "n_eval_edges": int(split.eval_src.shape[0]),
                    "n_loss": split.n_loss,
                    "n_test": split.n_test,
                    "train_edges_touch_test": _touches_test(
                        split.train_src, split.train_dst, split.test_mask
                    ),
                }
            )
    summary["fold_edges"] = fold_edges
    return summary


def _touches_test(src, dst, test_mask) -> bool:
    if src.size == 0:
        return False
    return bool(test_mask[src].any() or test_mask[dst].any())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B Route A from-scratch GNNs.")
    parser.add_argument("--cells", type=Path, default=DATA_FROZEN_DIR / "cells.gpkg")
    parser.add_argument("--edges", type=Path, default=DATA_FROZEN_DIR / "graph_edges.csv")
    parser.add_argument("--folds", type=Path, default=DATA_FROZEN_DIR / "spatial_folds.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_FROZEN_DIR / "feature_manifest.json")
    parser.add_argument("--poi", type=Path, default=DATA_FROZEN_DIR / "cell_poi_texts.csv")
    parser.add_argument(
        "--phase-a-metrics",
        type=Path,
        default=OUTPUTS_METRICS_DIR / "phase_a_ml_baselines.json",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=OUTPUTS_METRICS_DIR / "phase_b_route_a.json",
    )
    parser.add_argument(
        "--assembly-out",
        type=Path,
        default=OUTPUTS_METRICS_DIR / "phase_b_graph_assembly.json",
    )
    parser.add_argument("--tables-dir", type=Path, default=OUTPUTS_METRICS_DIR.parent / "tables")
    args = parser.parse_args(argv)

    bundle = load_phase_b_bundle(args.cells, args.edges, args.folds, args.manifest, args.poi)
    assembly = _assembly(bundle)
    args.assembly_out.parent.mkdir(parents=True, exist_ok=True)
    args.assembly_out.write_text(json.dumps(assembly, indent=2) + "\n", encoding="utf-8")
    result = run_route_a(bundle, phase_a_metrics=args.phase_a_metrics)
    paths = write_route_a_artifacts(result, args.metrics_out, args.tables_dir)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
