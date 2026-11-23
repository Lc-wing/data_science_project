"""CLI entry point for Milestone M3 Step 03: Cross-Cycle EDA & Coverage Consistency."""

from __future__ import annotations

import json
import sys

from stock_volatility.eda import run_eda_pipeline


def main() -> int:
    """Run deterministic M3 EDA pipeline from frozen data/frozen/ bundle."""
    summary = run_eda_pipeline()
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
