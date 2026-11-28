"""CLI entry point for Step 07: Build 300-DPI Figures, Table 1/2, Unified CS482 PDF & Audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_volatility.report import run_report_build_and_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Step 07 Unified CS482 Report Build & Audit."""
    parser = argparse.ArgumentParser(
        description="Step 07: Build 300-DPI figures, Table 1, unified CS482 PDF & 61-claim audit."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional project root override (defaults to plan_6_stock_volatility/).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip regenerating 300-DPI Figure 1 & Figure 2 and reuse existing outputs/figures/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and audit `report/CS482_GroupX_StockVolatilityLLM.pdf` from frozen & H1/H2/H3 data."""
    args = parse_args(argv)
    summary = run_report_build_and_audit(
        project_root=args.project_root,
        regenerate_figures=not args.no_figures,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
