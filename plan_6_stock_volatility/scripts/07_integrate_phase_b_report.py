"""Post-Colab CLI: integrate verified Phase B Colab GPU (L4/A100) outputs into final CS482 PDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_volatility.phase_b.report_integration import build_and_audit_phase_b_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for post-Colab Phase B report integration."""
    parser = argparse.ArgumentParser(
        description="Integrate verified Phase B Colab outputs into final CS482 PDF & audit."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional Plan 6 project root override (defaults to plan_6_stock_volatility/).",
    )
    parser.add_argument(
        "--phase-b-dir",
        type=Path,
        default=None,
        help="Optional directory containing phase_b_summary.json & phase_b_comparison.tex.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and audit `report/CS482_GroupX_StockVolatilityLLM.pdf` from real Phase B results."""
    args = parse_args(argv)
    summary = build_and_audit_phase_b_report(
        project_root=args.project_root,
        phase_b_dir=args.phase_b_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
