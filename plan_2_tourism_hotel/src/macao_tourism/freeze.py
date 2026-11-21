"""Write and verify the frozen Phase A inputs that Phase B will share."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from macao_tourism.config import (
    DATA_FROZEN_DIR,
    PERIODS,
    PROVENANCE_PATH,
    RANDOM_SEED,
    TIERS,
    sha256_file,
    utc8_now_iso,
)
from macao_tourism.ingest import holidays
from macao_tourism.panel import build_months, build_panel, validate_panel
from macao_tourism.spec import build_analysis_spec
from macao_tourism.splits import build_rolling_origins, split_metadata, validate_rolling_origins

MANIFEST_NAME = "FREEZE_MANIFEST.json"
FROZEN_FILES: tuple[str, ...] = (
    "panel_tier_month.csv",
    "months.csv",
    "holiday_periods.csv",
    "rolling_origins.csv",
    "split_metadata.json",
    "analysis_spec.json",
    "data_dictionary.md",
)

DICTIONARY: tuple[tuple[str, str, str, str], ...] = (
    (
        "tier",
        "tier-month",
        "",
        "Star class: 5star, 4star, 3star, 2star (DSEC classes as published)",
    ),
    ("period", "month", "YYYY-MM", "Reference month"),
    ("occupancy_pct", "tier-month", "%", "Average occupancy rate of guest rooms (DSEC)"),
    ("rooms", "tier-month", "rooms", "Guest rooms of operating establishments at month end (DSEC)"),
    ("log_rooms", "tier-month", "log rooms", "Natural log of rooms"),
    ("establishments", "tier-month", "count", "Operating establishments at month end (DSEC)"),
    ("guests", "tier-month", "persons", "Guests of the star class during the month (DSEC)"),
    ("year", "month", "", "Calendar year"),
    ("month", "month", "1-12", "Calendar month"),
    ("days_in_month", "month", "days", "Days in the month"),
    (
        "period_label",
        "month",
        "",
        "pre_pandemic (2010-01 to 2019-12), transition_2020_01, pandemic (2020-02 to 2022-12), "
        "post_pandemic (2023-01 on)",
    ),
    ("post", "month", "0/1", "1 in the post-pandemic period"),
    ("in_main_sample", "month", "0/1", "1 in the pre-pandemic and post-pandemic periods"),
    ("visitors_total", "month", "persons", "Visitor arrivals (DSEC)"),
    ("visitors_overnight", "month", "persons", "Overnight visitor arrivals (DSEC)"),
    ("visitors_same_day", "month", "persons", "Same-day visitor arrivals (DSEC)"),
    ("visitors_mainland", "month", "persons", "Arrivals with a Chinese mainland travel document"),
    ("visitors_hong_kong", "month", "persons", "Arrivals with a Hong Kong SAR travel document"),
    ("visitors_taiwan", "month", "persons", "Arrivals with a Taiwan region travel document"),
    ("package_tour_visitors", "month", "persons", "Visitors on package tours (DSEC)"),
    ("log_visitors_overnight", "month", "log persons", "Natural log of visitors_overnight"),
    ("same_day_share_pct", "month", "%", "100 × visitors_same_day / visitors_total"),
    (
        "international_share_pct",
        "month",
        "%",
        "100 × (1 − (mainland + Hong Kong + Taiwan) / visitors_total)",
    ),
    ("package_tour_share_pct", "month", "%", "100 × package_tour_visitors / visitors_total"),
    (
        "cn_holiday_days",
        "month",
        "days",
        "Days inside a mainland public-holiday period, final schedule (State Council notices). "
        "Used in the ex-post regression",
    ),
    (
        "cn_golden_week_days",
        "month",
        "days",
        "Days inside a Spring Festival or National Day holiday period",
    ),
    (
        "cn_holiday_days_at_origin",
        "month",
        "days",
        "cn_holiday_days as scheduled by the notices published up to the end of the month "
        "before. This, not cn_holiday_days, is the known calendar feature for forecasts",
    ),
    (
        "cn_golden_week_days_at_origin",
        "month",
        "days",
        "cn_golden_week_days as scheduled by the end of the month before",
    ),
    (
        "typhoon_days",
        "month",
        "days",
        "Days with tropical cyclone signal No. 8 or higher in force (SMG). Known only after "
        "the month: used in the ex-post regression and as a lagged forecast feature, never "
        "for the target month",
    ),
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def holiday_periods_frame() -> pd.DataFrame:
    """Holiday periods with the day each schedule was published."""
    return holidays.versions_frame(holidays.load_schedule_versions())


def render_dictionary() -> str:
    lines = [
        "# Frozen data dictionary (Plan 2)",
        "",
        "`panel_tier_month.csv` has one row per star class and month (804 rows) and carries "
        "every column below. `months.csv` has one row per month (201 rows) and carries the "
        "month-level columns. Shares are in percent. Derived values are rounded to 10 decimals.",
        "",
        "| Column | Level | Unit | Definition |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| `{name}` | {level} | {unit} | {text} |" for name, level, unit, text in DICTIONARY]
    lines += [
        "",
        "Other files:",
        "",
        "- `holiday_periods.csv`: the 115 mainland holiday periods with start, end, and the "
        "day the schedule was published, plus the 2 schedules a later notice replaced "
        "(`superseded`).",
        "- `rolling_origins.csv`: validation and test origins. The origin is the last month "
        "whose data may be used; the target is the next month.",
        "- `split_metadata.json`: the split rules in words and counts.",
        "- `analysis_spec.json`: the pre-registered analysis specification.",
        "- `FREEZE_MANIFEST.json`: SHA-256 of every file above.",
        "",
    ]
    return "\n".join(lines)


def build_frozen_files(target_dir: Path) -> dict[str, Any]:
    """Write every frozen file except the manifest into ``target_dir``; return counts."""
    target_dir.mkdir(parents=True, exist_ok=True)
    months = build_months()
    panel = build_panel(months)
    validate_panel(months, panel)
    origins = build_rolling_origins()
    validate_rolling_origins(origins)
    holiday_periods = holiday_periods_frame()

    _write_csv(panel, target_dir / "panel_tier_month.csv")
    _write_csv(months, target_dir / "months.csv")
    _write_csv(holiday_periods, target_dir / "holiday_periods.csv")
    _write_csv(origins, target_dir / "rolling_origins.csv")
    _write_json(split_metadata(origins), target_dir / "split_metadata.json")
    _write_json(build_analysis_spec(), target_dir / "analysis_spec.json")
    (target_dir / "data_dictionary.md").write_text(render_dictionary(), encoding="utf-8")

    by_label = months["period_label"].value_counts()
    differs = months[months["cn_holiday_days_at_origin"] != months["cn_holiday_days"]]
    role = origins.set_index("target")["split"].map("{} target".format).to_dict()
    late = {
        row.period: role.get(
            row.period, "training target" if row.in_main_sample else "not in the main sample"
        )
        for row in differs.itertuples()
    }
    return {
        "panel_rows": int(len(panel)),
        "panel_columns": int(panel.shape[1]),
        "months": int(len(months)),
        "tiers": len(TIERS),
        "months_by_period": {label: int(by_label[label]) for label in PERIODS},
        "main_sample_months": int(months["in_main_sample"].sum()),
        "main_sample_rows": int(panel["in_main_sample"].sum()),
        "post_pandemic_rows": int(panel["post"].sum()),
        "missing_values": int(panel.isna().sum().sum()),
        "holiday_periods": int((~holiday_periods["superseded"]).sum()),
        "superseded_holiday_schedules": int(holiday_periods["superseded"].sum()),
        "months_where_schedule_at_origin_differs_from_final": late,
        "validation_origins": int((origins["split"] == "validation").sum()),
        "test_origins": int((origins["split"] == "test").sum()),
        "test_forecasts": int((origins["split"] == "test").sum() * len(TIERS)),
    }


def file_hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in FROZEN_FILES}


def write_manifest(
    frozen_dir: Path, counts: dict[str, Any], frozen_at: str | None = None
) -> dict[str, Any]:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["files"]
    downloaded = sorted(entry["downloaded_at_utc8"] for entry in provenance.values())
    manifest = {
        "schema_version": "1.0.0",
        "frozen_at_utc8": frozen_at or utc8_now_iso(),
        "seed": RANDOM_SEED,
        "phase": "A_to_B",
        "note": (
            "SHA-256 of every file in data/frozen except this manifest. Phase B reads the "
            "same panel and the same rolling origins. The files are rebuilt from data/raw "
            "by scripts/02_build_panel_and_freeze.py; --check verifies them."
        ),
        "inputs": {
            "data/raw/data_provenance.json": sha256_file(PROVENANCE_PATH),
            "raw_files": len(provenance),
            "downloaded_between_utc8": [downloaded[0], downloaded[-1]],
        },
        "counts": counts,
        "files": {
            f"data/frozen/{name}": digest for name, digest in file_hashes(frozen_dir).items()
        },
    }
    _write_json(manifest, frozen_dir / MANIFEST_NAME)
    return manifest


def load_manifest(frozen_dir: Path = DATA_FROZEN_DIR) -> dict[str, Any]:
    return json.loads((frozen_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def verify_freeze(frozen_dir: Path = DATA_FROZEN_DIR) -> list[str]:
    """Problems found when comparing the files on disk with the manifest; empty if none."""
    if not (frozen_dir / MANIFEST_NAME).is_file():
        return [f"{MANIFEST_NAME} is missing"]
    recorded = {
        Path(rel).name: digest for rel, digest in load_manifest(frozen_dir)["files"].items()
    }
    problems = []
    on_disk = {path.name for path in frozen_dir.iterdir() if path.is_file()} - {MANIFEST_NAME}
    problems += [f"not in manifest: {name}" for name in sorted(on_disk - set(recorded))]
    for name, digest in sorted(recorded.items()):
        path = frozen_dir / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256_file(path) != digest:
            problems.append(f"hash mismatch: {name}")
    if set(recorded) != set(FROZEN_FILES):
        problems.append("manifest does not list exactly the expected frozen files")
    return problems


def differences_from_rebuild(rebuilt_dir: Path, frozen_dir: Path = DATA_FROZEN_DIR) -> list[str]:
    """Frozen files whose content differs from a fresh build out of data/raw."""
    recorded = load_manifest(frozen_dir)["files"]
    rebuilt = file_hashes(rebuilt_dir)
    return [name for name in FROZEN_FILES if recorded.get(f"data/frozen/{name}") != rebuilt[name]]


@dataclass(frozen=True)
class Frozen:
    """The frozen tables, read after their hashes were checked against the manifest."""

    months: pd.DataFrame
    panel: pd.DataFrame
    origins: pd.DataFrame
    spec: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def spec_sha256(self) -> str:
        return self.manifest["files"]["data/frozen/analysis_spec.json"]

    @property
    def panel_sha256(self) -> str:
        return self.manifest["files"]["data/frozen/panel_tier_month.csv"]


def load_frozen(frozen_dir: Path = DATA_FROZEN_DIR) -> Frozen:
    """Read the frozen inputs of steps 3 to 6; refuse if any file differs from the manifest."""
    problems = verify_freeze(frozen_dir)
    if problems:
        raise RuntimeError("The freeze does not match its manifest: " + "; ".join(problems))
    return Frozen(
        months=pd.read_csv(frozen_dir / "months.csv"),
        panel=pd.read_csv(frozen_dir / "panel_tier_month.csv"),
        origins=pd.read_csv(frozen_dir / "rolling_origins.csv"),
        spec=json.loads((frozen_dir / "analysis_spec.json").read_text(encoding="utf-8")),
        manifest=load_manifest(frozen_dir),
    )
