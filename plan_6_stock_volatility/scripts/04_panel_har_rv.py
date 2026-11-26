"""CLI entry point for Milestone M3 Step 04: H1 Stock-FE Extended Panel HAR-RV Regression."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_volatility.regression import run_h1_regression_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Step 04 H1 Panel HAR-RV Regression."""
    parser = argparse.ArgumentParser(
        description="Plan 6 M3 Step 04: H1 Stock-FE Extended Panel HAR-RV with Driscoll-Kraay SE."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional Plan 6 project root override (defaults to plan_6_stock_volatility/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run deterministic M3 H1 panel regression pipeline from frozen data/frozen/ bundle."""
    args = parse_args(argv)
    payload = run_h1_regression_pipeline(project_root=args.project_root)
    compact = {
        "milestone": payload["milestone"],
        "generated_at_utc": payload["generated_at_utc"],
        "n_obs": payload["main_model"]["n_obs"],
        "n_dates": payload["main_model"]["n_dates"],
        "bartlett_bandwidth_L": payload["main_model"]["bartlett_bandwidth_L"],
        "within_r2": round(payload["main_model"]["within_r2"], 4),
        "confirmatory_holm_significant": {
            row["test_id"]: row["significant_holm_05"]
            for row in payload["confirmatory_family_holm"]
        },
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
