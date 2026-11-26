"""CLI entry point for Milestone M3 Step 03: Cross-Cycle EDA & Coverage Consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_volatility.eda import run_eda_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Step 03 Cross-Cycle EDA."""
    parser = argparse.ArgumentParser(
        description="Plan 6 M3 Step 03: Cross-cycle EDA & corporate-event coverage consistency."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional Plan 6 project root override (defaults to plan_6_stock_volatility/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run deterministic M3 EDA pipeline from frozen data/frozen/ bundle."""
    args = parse_args(argv)
    summary = run_eda_pipeline(project_root=args.project_root)
    compact = {
        "milestone": summary["milestone"],
        "generated_at_utc": summary["generated_at_utc"],
        "modeling_total_rows": summary["modeling_total_rows"],
        "modeling_total_days": summary["modeling_total_days"],
        "vix_train_median_threshold": round(summary["vix_train_median_threshold"], 4),
        "leverage_ratio_neg_over_pos": round(
            summary["leverage_overall"]["ratio_neg_over_pos"], 4
        ),
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
