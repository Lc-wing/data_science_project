"""Unified one-command deterministic offline replay CLI (Steps 01 -> 05 -> 07)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from stock_volatility import config
from stock_volatility.analysis.eda import run_eda_pipeline
from stock_volatility.analysis.lgbm import run_h2_forecast_backtest_pipeline
from stock_volatility.analysis.panel_har_rv import run_h1_regression_pipeline
from stock_volatility.ingest.audit import (
    run_download_and_audit_pipeline,
    run_skip_download_pipeline,
)
from stock_volatility.panel.freeze import build_and_freeze_panel, verify_freeze_manifest
from stock_volatility.report import run_report_build_and_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for single-command offline pipeline replay."""
    parser = argparse.ArgumentParser(
        description="Plan 6 Stock Volatility: Unified deterministic offline replay CLI."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Replay offline from committed data/raw/ and frozen snapshots (zero network calls).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Verify frozen manifest and rebuild only 300-DPI figures, Table 1, CS482 PDF & audit.",
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
        help="Optional project root override (defaults to plan_6_stock_volatility/).",
    )
    args = parser.parse_args(argv)
    if args.report_only and args.refreeze:
        parser.error("Cannot combine --report-only with --refreeze")
    return args


def main(argv: list[str] | None = None) -> int:
    """Execute full CPU pipeline replay (preserving committed H3 GPU outputs) or `--report-only`."""
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
        results["step_01_ingest"] = run_download_and_audit_pipeline(
            force_download=False,
            project_root=root,
        )

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
    results["step_05_h2"] = run_h2_forecast_backtest_pipeline(project_root=root)
    results["step_07_report"] = run_report_build_and_audit(project_root=root)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
