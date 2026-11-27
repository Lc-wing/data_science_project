"""CLI entry point for Step 05: H2 Out-of-Sample Baselines, Global LightGBM, HLN-DM & TreeSHAP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_volatility.analysis.lgbm import run_h2_forecast_backtest_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Step 05 H2 Out-of-Sample Backtest."""
    parser = argparse.ArgumentParser(
        description="Step 05: H2 Out-of-sample baselines, Global LightGBM, HLN-DM & TreeSHAP."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional project root override (defaults to plan_6_stock_volatility/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run deterministic H2 forecasting, evaluation & TreeSHAP pipeline."""
    args = parse_args(argv)
    summary = run_h2_forecast_backtest_pipeline(project_root=args.project_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
