"""CLI entry point for Milestone M5 Step 06: 300-DPI Figures, Table 1, CS482 PDF & Audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_volatility.report import run_report_build_and_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Step 06 Report Build & Audit."""
    parser = argparse.ArgumentParser(
        description="Plan 6 M5 Step 06: Build 300-DPI figures, Table 1, Phase A PDF & audit."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional Plan 6 project root override (defaults to plan_6_stock_volatility/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build 300-DPI figures, Table 1, CS482 Phase A Draft PDF, and automated audit."""
    args = parse_args(argv)
    summary = run_report_build_and_audit(project_root=args.project_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
