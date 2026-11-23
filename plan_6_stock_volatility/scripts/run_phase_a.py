"""One-command deterministic Phase A replay CLI (Steps 01 -> 06)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from stock_volatility import config
from stock_volatility.eda import run_eda_pipeline
from stock_volatility.ingest.audit import (
    run_download_and_audit_pipeline,
    run_skip_download_pipeline,
)
from stock_volatility.panel.freeze import build_and_freeze_panel, verify_freeze_manifest
from stock_volatility.regression import run_h1_regression_pipeline
from stock_volatility.report import run_report_build_and_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for single-command Phase A replay."""
    parser = argparse.ArgumentParser(
        description="Plan 6 Phase A: Deterministic end-to-end replay CLI."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Replay offline from committed data/raw/ and frozen snapshots (zero network calls).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Verify frozen manifest and rebuild only M5 figures, Table 1, CS482 PDF & audit.",
    )
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help="Rebuild data/frozen/ during full replay instead of read-only --check verification.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional Plan 6 project root override (defaults to plan_6_stock_volatility/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute full Phase A replay or `--report-only` rebuild."""
    args = parse_args(argv)
    root = args.project_root.resolve() if args.project_root is not None else config.PROJECT_ROOT

    if args.report_only:
        report_summary = run_report_build_and_audit(project_root=root)
        print(json.dumps(report_summary, indent=2))
        return 0

    results: dict[str, Any] = {}
    if args.skip_download:
        results["step_01_ingest"] = run_skip_download_pipeline(project_root=root)
    else:
        results["step_01_ingest"] = run_download_and_audit_pipeline(force_download=False)

    if args.refreeze:
        results["step_02_freeze"] = build_and_freeze_panel(project_root=root, refreeze=True)
    else:
        results["step_02_freeze"] = verify_freeze_manifest(
            frozen_dir=root / "data" / "frozen", project_root=root
        )

    results["step_03_eda"] = {
        "modeling_total_rows": run_eda_pipeline(project_root=root)["modeling_total_rows"],
    }
    h1_res = run_h1_regression_pipeline(project_root=root)
    results["step_04_h1"] = {
        "n_obs": h1_res["main_model"]["n_obs"],
        "within_r2": h1_res["main_model"]["within_r2"],
    }

    # Load Step 05 M4 backtest runner dynamically from sibling script
    step05_path = Path(__file__).resolve().parent / "05_forecast_backtest.py"
    spec = importlib.util.spec_from_file_location("step05_forecast_backtest", step05_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {step05_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results["step_05_h2"] = mod.run_h2_forecast_backtest_pipeline(project_root=root)
    results["step_06_report"] = run_report_build_and_audit(project_root=root)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
