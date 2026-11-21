#!/usr/bin/env python3
"""Colab entry for Route B. Does not download weights unless CUDA is present.

Typical Colab L4/A100 use, from the repository root:

    python scripts/colab_phase_b.py --allow-weight-download

The weight path writes run 2 only. It does not replace ``outputs/colab_runs/run1``
or the first comparison table.

Without that flag, or without a CUDA device, the process exits before any call
to the Hugging Face hub. ``--assemble-only`` checks the frozen bundle and
writes the graph-assembly audit on CPU.
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
from macao_equity.phase_b.route_b import REFUSAL, run_route_b, write_route_b_run2  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B Route B Colab runner.")
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
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
        "--route-a-metrics",
        type=Path,
        default=OUTPUTS_METRICS_DIR / "phase_b_route_a.json",
    )
    parser.add_argument(
        "--assembly-out",
        type=Path,
        default=OUTPUTS_METRICS_DIR / "phase_b_graph_assembly.json",
    )
    args = parser.parse_args(argv)

    bundle = load_phase_b_bundle(args.cells, args.edges, args.folds, args.manifest, args.poi)
    if args.assemble_only:
        payload = bundle.assembly_summary()
        args.assembly_out.parent.mkdir(parents=True, exist_ok=True)
        args.assembly_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"assembly: {args.assembly_out}")
        return 0
    if not args.allow_weight_download:
        print(REFUSAL, file=sys.stderr)
        return 2
    if not args.route_a_metrics.exists():
        print(
            "Route A metrics are missing. Run scripts/run_phase_b_route_a.py first "
            "so the comparison table can copy the numeric GATv2 arm.",
            file=sys.stderr,
        )
        return 2
    result = run_route_b(bundle, allow_weight_download=True)
    paths = write_route_b_run2(
        result,
        project_root=PROJECT_ROOT,
        phase_a_metrics=args.phase_a_metrics,
        route_a_metrics=args.route_a_metrics,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
