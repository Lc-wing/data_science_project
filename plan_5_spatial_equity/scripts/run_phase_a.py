#!/usr/bin/env python3
"""Unified Phase A replay.

``--skip-download`` does not call the ingest downloader and does not rewrite
the M1 text contracts. It rebuilds the M4 freeze, figures, Table 1, and report
from the frozen base grids, the interim 2SFCA layers, and the committed M2/M3
tables. If those interim layers or tables are missing, the M2 or M3 scripts
are run locally. They read the frozen grids and do not download sources.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(script: Path, extra: list[str]) -> None:
    command = [sys.executable, str(script), *extra]
    print("running", " ".join(command))
    proc = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _require(paths: list[Path]) -> None:
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Phase A replay is missing frozen or interim inputs: {joined}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the MacaoSpatialEquity Phase A pipeline through the M4 report."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=(
            "Replay from frozen and interim inputs without downloading raw sources. "
            "M1 text contracts are left untouched."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Passed through to the M1 ingest script when downloads are enabled.",
    )
    args = parser.parse_args(argv)

    scripts = PROJECT_ROOT / "scripts"
    frozen = PROJECT_ROOT / "data" / "frozen"
    interim = PROJECT_ROOT / "data" / "interim"
    tables = PROJECT_ROOT / "outputs" / "tables"
    metrics = PROJECT_ROOT / "outputs" / "metrics"

    if args.skip_download and args.force_download:
        raise SystemExit("Use either --skip-download or --force-download, not both")

    if not args.skip_download:
        ingest_args = ["--force-download"] if args.force_download else []
        _run(scripts / "01_ingest_and_prepare.py", ingest_args)
        _run(scripts / "02_run_spatial_analysis.py", [])
        _run(scripts / "03_run_ml_baselines.py", [])
    else:
        base_inputs = [
            frozen / "cells_base_res9.gpkg",
            frozen / "cells_base_res8.gpkg",
            frozen / "facilities_clean.gpkg",
            frozen / "parishes_7.gpkg",
            frozen / "graph_edges.csv",
            frozen / "spatial_folds.csv",
            frozen / "feature_manifest.json",
        ]
        _require(base_inputs)
        interim_r9 = interim / "cells_r9_with_2sfca_lisa.gpkg"
        interim_r8 = interim / "cells_r8_with_2sfca.gpkg"
        h1 = tables / "h1_inequality_lisa.csv"
        if not (interim_r9.is_file() and interim_r8.is_file() and h1.is_file()):
            _run(scripts / "02_run_spatial_analysis.py", [])
        ml_json = metrics / "phase_a_ml_baselines.json"
        if not ml_json.is_file():
            _run(scripts / "03_run_ml_baselines.py", [])
        _require(
            [
                interim_r9,
                interim_r8,
                h1,
                tables / "h2_gini_diff_bootstrap.csv",
                tables / "h3_spatial_regression.csv",
                tables / "h3_regression_diagnostics.json",
                tables / "lorenz_curves.csv",
                tables / "maup_r8_vs_r9.csv",
                ml_json,
                tables / "ml_cv_results.csv",
            ]
        )

    _run(scripts / "04_freeze_and_report.py", [])
    print(f"Phase A replay complete (skip_download={args.skip_download})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
