from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from macao_tourism import freeze
from macao_tourism.config import DATA_FROZEN_DIR, PROJECT_ROOT, PROVENANCE_PATH, sha256_file
from macao_tourism.spec import (
    KNOWN_CALENDAR_FEATURES,
    LAGGED_VARIABLES,
    build_analysis_spec,
    forecast_feature_names,
)


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("rebuilt")
    freeze.build_frozen_files(directory)
    return directory


@pytest.fixture()
def copy_of_freeze(tmp_path: Path) -> Path:
    target = tmp_path / "frozen"
    shutil.copytree(DATA_FROZEN_DIR, target)
    return target


def test_committed_freeze_matches_its_manifest() -> None:
    assert freeze.verify_freeze() == []
    manifest = freeze.load_manifest()
    assert set(manifest["files"]) == {f"data/frozen/{name}" for name in freeze.FROZEN_FILES}
    assert manifest["inputs"]["data/raw/data_provenance.json"] == sha256_file(PROVENANCE_PATH)
    assert manifest["counts"]["panel_rows"] == 804
    assert manifest["counts"]["main_sample_rows"] == 660
    assert manifest["counts"]["post_pandemic_rows"] == 180
    assert manifest["counts"]["missing_values"] == 0
    assert manifest["counts"]["test_origins"] == 26
    assert manifest["counts"]["test_forecasts"] == 104


def test_committed_freeze_is_what_the_raw_files_rebuild_to(rebuilt: Path) -> None:
    assert freeze.differences_from_rebuild(rebuilt) == []
    for name in freeze.FROZEN_FILES:
        assert (rebuilt / name).read_bytes() == (DATA_FROZEN_DIR / name).read_bytes(), name


def test_a_changed_file_is_detected(copy_of_freeze: Path) -> None:
    assert freeze.verify_freeze(copy_of_freeze) == []
    panel = copy_of_freeze / "panel_tier_month.csv"
    panel.write_text(panel.read_text(encoding="utf-8").replace("80.5", "80.6", 1), encoding="utf-8")
    assert freeze.verify_freeze(copy_of_freeze) == ["hash mismatch: panel_tier_month.csv"]


def test_a_missing_or_extra_file_is_detected(copy_of_freeze: Path) -> None:
    (copy_of_freeze / "months.csv").unlink()
    (copy_of_freeze / "notes.csv").write_text("x\n", encoding="utf-8")
    assert freeze.verify_freeze(copy_of_freeze) == [
        "not in manifest: notes.csv",
        "missing: months.csv",
    ]
    (copy_of_freeze / freeze.MANIFEST_NAME).unlink()
    assert freeze.verify_freeze(copy_of_freeze) == [f"{freeze.MANIFEST_NAME} is missing"]


def test_check_command_passes_and_writes_nothing() -> None:
    before = {path.name: path.stat().st_mtime_ns for path in DATA_FROZEN_DIR.iterdir()}
    result = subprocess.run(
        [sys.executable, "scripts/02_build_panel_and_freeze.py", "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 mismatches" in result.stdout
    assert before == {path.name: path.stat().st_mtime_ns for path in DATA_FROZEN_DIR.iterdir()}


def test_frozen_tables_read_back_with_the_expected_shape() -> None:
    panel = pd.read_csv(DATA_FROZEN_DIR / "panel_tier_month.csv")
    months = pd.read_csv(DATA_FROZEN_DIR / "months.csv")
    origins = pd.read_csv(DATA_FROZEN_DIR / "rolling_origins.csv")
    periods = pd.read_csv(DATA_FROZEN_DIR / "holiday_periods.csv")
    assert panel.shape == (804, 29)
    assert months.shape == (201, 23)
    assert origins["split"].value_counts().to_dict() == {"validation": 36, "test": 26}
    assert len(periods) == 117
    assert periods["superseded"].sum() == 2
    assert set(panel["period"]) == set(months["period"])
    assert set(origins["target"]) <= set(months["period"])
    dictionary = (DATA_FROZEN_DIR / "data_dictionary.md").read_text(encoding="utf-8")
    for column in panel.columns:
        assert f"| `{column}` |" in dictionary, column


def test_spec_on_disk_is_the_spec_in_code() -> None:
    on_disk = json.loads((DATA_FROZEN_DIR / "analysis_spec.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(build_analysis_spec()))


def test_forecast_features_cannot_see_the_target_month() -> None:
    features = forecast_feature_names()
    assert len(features) == 1 + 4 + 7 * 4
    assert len(set(features)) == len(features)
    lagged = [name for name in features if "_lag" in name]
    assert all(int(name.rsplit("_lag", 1)[1]) >= 1 for name in lagged)
    undated = set(features) - set(lagged)
    assert undated == {"tier", *KNOWN_CALENDAR_FEATURES}
    assert "cn_holiday_days" not in features
    assert "cn_golden_week_days" not in features
    assert "typhoon_days" not in features
    assert "typhoon_days" in LAGGED_VARIABLES
    panel = pd.read_csv(DATA_FROZEN_DIR / "panel_tier_month.csv", nrows=1)
    sources = set(LAGGED_VARIABLES) | set(KNOWN_CALENDAR_FEATURES) | {"tier"}
    assert sources <= set(panel.columns)


def test_spec_fixes_the_confirmatory_choices() -> None:
    spec = build_analysis_spec()
    regression = spec["regression_h1_h2"]
    assert regression["standard_errors"]["type"] == "Driscoll–Kraay"
    assert set(regression["confirmatory_tests"]) == {"H1", "H2", "multiplicity"}
    assert "not changed on the basis of a diagnostic p-value" in regression["diagnostics"]["rule"]
    assert set(regression["terms"]["controls"]) <= set(
        pd.read_csv(DATA_FROZEN_DIR / "panel_tier_month.csv", nrows=1).columns
    )
    forecast = spec["forecast_h3"]
    assert forecast["test_origins"] == ["2024-07", "2026-08"]
    assert forecast["horizon_months"] == 1
    assert set(forecast["models"]) == {"seasonal_naive", "sarimax", "lightgbm"}
    assert forecast["models"]["lightgbm"]["lags_months"] == [1, 2, 3, 12]
    assert forecast["tests"]["multiplicity"].startswith("Holm")
    assert max(max(window) for window in forecast["validation_targets"]) < "2024-07"
    assert "no causal claim" in spec["scope"]["claims"]
    assert spec["data"]["left_out_of_main_sample"] == ["transition_2020_01", "pandemic"]
