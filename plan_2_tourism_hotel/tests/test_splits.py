from __future__ import annotations

import pandas as pd
import pytest

from macao_tourism import splits
from macao_tourism.config import MAIN_SAMPLE_PERIODS, period_label


@pytest.fixture(scope="module")
def origins() -> pd.DataFrame:
    return splits.build_rolling_origins()


def test_shift_month_crosses_years() -> None:
    assert splits.shift_month("2024-12", 1) == "2025-01"
    assert splits.shift_month("2010-01", 12) == "2011-01"
    assert splits.shift_month("2024-01", -1) == "2023-12"
    assert splits.first_trainable_target() == "2011-01"


def test_test_origins_are_the_26_months_of_the_plan(origins: pd.DataFrame) -> None:
    test = origins[origins["split"] == "test"]
    assert len(test) == 26
    assert test["origin"].tolist() == splits.month_range("2024-07", "2026-08")
    assert test["target"].tolist() == splits.month_range("2024-08", "2026-09")
    assert (test["target_period_label"] == "post_pandemic").all()


def test_validation_origins_end_before_the_first_test_origin(origins: pd.DataFrame) -> None:
    validation = origins[origins["split"] == "validation"]
    assert validation["block"].value_counts().to_dict() == {"validation_1": 24, "validation_2": 12}
    assert validation["target"].min() == "2018-01"
    assert validation["target"].max() == "2024-06"
    assert validation["target"].max() < origins.loc[origins["split"] == "test", "origin"].min()
    assert validation["target_period_label"].isin(MAIN_SAMPLE_PERIODS).all()


def test_every_origin_is_one_month_before_its_target(origins: pd.DataFrame) -> None:
    assert (origins["horizon_months"] == 1).all()
    assert [splits.shift_month(o, 1) for o in origins["origin"]] == origins["target"].tolist()
    assert origins["target"].is_unique
    assert (origins["train_last_target"] == origins["origin"]).all()


@pytest.mark.parametrize("include_pandemic", [False, True])
def test_training_targets_never_reach_past_the_origin(
    origins: pd.DataFrame, include_pandemic: bool
) -> None:
    for row in origins.itertuples():
        training = splits.training_targets(row.origin, include_pandemic=include_pandemic)
        assert max(training) <= row.origin < row.target
        assert row.target not in training
        assert min(training) == "2011-01"


def test_main_training_rule_leaves_out_the_pandemic(origins: pd.DataFrame) -> None:
    main = splits.training_targets("2024-07")
    assert len(main) == 108 + 19
    assert {period_label(month) for month in main} == set(MAIN_SAMPLE_PERIODS)
    assert "2020-01" not in main
    everything = splits.training_targets("2024-07", include_pandemic=True)
    assert len(everything) == len(main) + 36
    first_test = origins[origins["origin"] == "2024-07"].iloc[0]
    assert first_test["n_train_target_months_main"] == 127
    assert first_test["n_train_target_months_all"] == 163
    assert origins["n_train_target_months_main"].is_monotonic_increasing


def test_validation_rejects_a_tampered_split(origins: pd.DataFrame) -> None:
    splits.validate_rolling_origins(origins)
    shortened = origins[origins["origin"] != "2025-03"]
    with pytest.raises(ValueError, match="one per month"):
        splits.validate_rolling_origins(shortened)
    leaking = origins.copy()
    row = leaking.index[leaking["split"] == "validation"][-1]
    leaking.loc[row, "target"] = "2024-07"
    with pytest.raises(ValueError, match="not earlier than the first test origin"):
        splits.validate_rolling_origins(leaking)
    backwards = origins.copy()
    backwards.loc[backwards.index[0], "origin"] = "2018-02"
    with pytest.raises(ValueError, match="not earlier than its target"):
        splits.validate_rolling_origins(backwards)


def test_split_metadata_counts(origins: pd.DataFrame) -> None:
    meta = splits.split_metadata(origins)
    assert meta["test_origins"] == 26
    assert meta["test_forecasts"] == 104
    assert meta["validation_origins"] == 36
    assert meta["blocks"]["test"] == {
        "first_origin": "2024-07",
        "last_origin": "2026-08",
        "first_target": "2024-08",
        "last_target": "2026-09",
        "origins": 26,
    }
