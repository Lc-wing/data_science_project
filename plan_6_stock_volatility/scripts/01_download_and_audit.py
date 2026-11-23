"""CLI entry point for Milestone M1: Data Ingest & Day-1 Availability / Coverage Audit."""

from __future__ import annotations

import argparse
import json
import sys

from stock_volatility.ingest.audit import (
    run_download_and_audit_pipeline,
    run_skip_download_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for Step 01 Data Ingest & Day-1 Audit."""
    parser = argparse.ArgumentParser(
        description=(
            "Plan 6 M1: Download 60-stock OHLCV, macro, and paginated SEC EDGAR data "
            "and run Day-1 audit."
        )
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=(
            "Verify committed data/raw/ SHA-256 hashes and rebuild interim & Day-1 "
            "audit outputs offline."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Force fresh network downloads ignoring existing data/raw/ and .cache/ "
            "snapshots."
        ),
    )
    args = parser.parse_args(argv)
    if args.skip_download and args.force_download:
        parser.error("Cannot specify both --skip-download and --force-download")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run M1 download+audit or offline `--skip-download` audit replay."""
    args = parse_args(argv)
    if args.skip_download:
        summary = run_skip_download_pipeline()
    else:
        summary = run_download_and_audit_pipeline(force_download=bool(args.force_download))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
