"""CLI entry point for Milestone M2: Panel Build, Purged Splits & Phase A -> B SHA-256 Freeze."""

from __future__ import annotations

import argparse
import json
import sys

from stock_volatility.panel.freeze import (
    build_and_freeze_panel,
    verify_freeze_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for Step 02 Panel Build & Freeze."""
    parser = argparse.ArgumentParser(
        description=(
            "Plan 6 M2: Build aligned 16:00 ET events, <=t features, t+1..t+5 targets, "
            "purged+embargoed splits, and frozen SHA-256 manifest."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Read-only SHA-256 verification of data/frozen/FREEZE_MANIFEST.json.",
    )
    group.add_argument(
        "--refreeze",
        action="store_true",
        help="Deterministically rebuild interim/frozen artifacts and rewrite FREEZE_MANIFEST.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run M2 freeze build, `--refreeze` deterministic rebuild, or `--check` audit."""
    args = parse_args(argv)
    if args.check:
        summary = verify_freeze_manifest()
    else:
        summary = build_and_freeze_panel(refreeze=bool(args.refreeze))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
