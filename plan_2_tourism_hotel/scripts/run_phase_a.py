"""Replay Phase A from the saved snapshots.

    uv run python scripts/run_phase_a.py --skip-download

Runs the audit, checks the freeze, refits nothing that the specification left to the
test window's second look, and writes the report. --skip-download reads data/raw and
does not contact the sources. The validation stage refits every SARIMAX candidate, so
the command takes several minutes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from macao_tourism.config import PROJECT_ROOT


def run(script: str, *args: str) -> None:
    command = [sys.executable, f"scripts/{script}", *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="audit the saved snapshots instead of downloading them",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="write the report from the outputs already on disk",
    )
    args = parser.parse_args()
    if args.report_only:
        run("06_write_report.py")
        return 0
    if not args.skip_download:
        parser.error("pass --skip-download to replay, or --report-only to rebuild the draft")
    run("01_download_and_audit.py", "--skip-download")
    run("02_build_panel_and_freeze.py", "--check")
    run("03_eda.py")
    run("04_panel_regression.py")
    run("05_forecast_backtest.py")
    run("06_write_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
