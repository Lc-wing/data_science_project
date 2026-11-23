"""CLI entry point for Milestone M5 Step 06: 300-DPI Figures, Table 1, CS482 PDF & Audit."""

from __future__ import annotations

import json
import sys

from stock_volatility.report import run_report_build_and_audit


def main() -> int:
    """Build 300-DPI figures, Table 1, CS482 Phase A Draft PDF, and automated audit."""
    summary = run_report_build_and_audit()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
