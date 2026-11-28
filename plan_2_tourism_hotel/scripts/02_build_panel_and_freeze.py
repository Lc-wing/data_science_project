"""Phase A step 2: build the tier × month panel and freeze it with the splits and the spec.

    uv run python scripts/02_build_panel_and_freeze.py             # build; write the freeze
    uv run python scripts/02_build_panel_and_freeze.py --check     # verify only, write nothing
    uv run python scripts/02_build_panel_and_freeze.py --refreeze  # replace a changed freeze

A second run that rebuilds the same content leaves the freeze, and its timestamp, as it is.
A run that would change a frozen file stops unless --refreeze is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from macao_tourism import freeze
from macao_tourism.config import (
    DATA_FROZEN_DIR,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_TABLES_DIR,
    PROJECT_ROOT,
    sha256_file,
)
from macao_tourism.panel import variable_summary


def write_summary(manifest: dict) -> None:
    months = pd.read_csv(DATA_FROZEN_DIR / "months.csv")
    panel = pd.read_csv(DATA_FROZEN_DIR / "panel_tier_month.csv")
    origins = pd.read_csv(DATA_FROZEN_DIR / "rolling_origins.csv")
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    variable_summary(months, panel).to_csv(
        OUTPUTS_TABLES_DIR / "panel_variable_summary.csv", index=False, lineterminator="\n"
    )
    blocks = (
        origins.groupby("block", sort=False)
        .agg(
            split=("split", "first"),
            origins=("origin", "size"),
            first_origin=("origin", "min"),
            last_origin=("origin", "max"),
            first_target=("target", "min"),
            last_target=("target", "max"),
            min_train_months_main=("n_train_target_months_main", "min"),
            max_train_months_main=("n_train_target_months_main", "max"),
        )
        .reset_index()
    )
    blocks.to_csv(
        OUTPUTS_TABLES_DIR / "rolling_origin_blocks.csv", index=False, lineterminator="\n"
    )
    summary = {
        "schema_version": "1.0.0",
        "step": "Phase A step 2: panel build and freeze",
        "frozen_at_utc8": manifest["frozen_at_utc8"],
        "counts": manifest["counts"],
        "analysis_spec_sha256": manifest["files"]["data/frozen/analysis_spec.json"],
        "panel_sha256": manifest["files"]["data/frozen/panel_tier_month.csv"],
        "rolling_origins_sha256": manifest["files"]["data/frozen/rolling_origins.csv"],
        "manifest_sha256": sha256_file(DATA_FROZEN_DIR / freeze.MANIFEST_NAME),
        "blocks": blocks.to_dict("records"),
    }
    (OUTPUTS_METRICS_DIR / "panel_freeze_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the freeze, write nothing")
    parser.add_argument("--refreeze", action="store_true", help="replace a freeze that changed")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        rebuilt_dir = Path(scratch)
        counts = freeze.build_frozen_files(rebuilt_dir)
        exists = (DATA_FROZEN_DIR / freeze.MANIFEST_NAME).is_file()

        if args.check:
            problems = freeze.verify_freeze() if exists else ["no freeze has been written"]
            if not problems:
                changed = freeze.differences_from_rebuild(rebuilt_dir)
                problems += [f"differs from a rebuild out of data/raw: {name}" for name in changed]
            for problem in problems:
                print(f"FREEZE CHECK FAILED: {problem}")
            if not problems:
                print(f"Freeze verified: {len(freeze.FROZEN_FILES)} files, 0 mismatches.")
            return 1 if problems else 0

        frozen_at = None
        if exists:
            changed = freeze.verify_freeze() + freeze.differences_from_rebuild(rebuilt_dir)
            if changed and not args.refreeze:
                print("The rebuild differs from the existing freeze:")
                for item in changed:
                    print(f"  {item}")
                print("Nothing was written. Use --refreeze to replace the freeze.")
                return 1
            if not changed:
                frozen_at = freeze.load_manifest()["frozen_at_utc8"]

        DATA_FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        for name in freeze.FROZEN_FILES:
            shutil.copyfile(rebuilt_dir / name, DATA_FROZEN_DIR / name)
        manifest = freeze.write_manifest(DATA_FROZEN_DIR, counts, frozen_at=frozen_at)

    write_summary(manifest)
    print(json.dumps(manifest["counts"], indent=2))
    state = "unchanged" if frozen_at else "written"
    print(f"Freeze {state}: {DATA_FROZEN_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
