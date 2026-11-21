"""Rolling-origin splits shared by Phase A and Phase B (plan section 2.3)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from macao_tourism.config import (
    FORECAST_LAGS,
    MAIN_SAMPLE_PERIODS,
    PANEL_END,
    PANEL_START,
    TEST_ORIGIN_FIRST,
    TEST_ORIGIN_LAST,
    TIERS,
    VALIDATION_TARGET_WINDOWS,
    period_label,
)

HORIZON_MONTHS = 1


def shift_month(month: str, by: int) -> str:
    return str(pd.Period(month, freq="M") + by)


def month_range(start: str, end: str) -> list[str]:
    return [str(m) for m in pd.period_range(start, end, freq="M")]


def first_trainable_target() -> str:
    """Earliest target month whose longest lag is still inside the panel window."""
    return shift_month(PANEL_START, max(FORECAST_LAGS))


def build_rolling_origins() -> pd.DataFrame:
    """One row per forecast origin: validation origins first, then the test origins.

    The origin is the last month whose data may be used; the target is the next month.
    """
    rows: list[dict[str, Any]] = []
    for window, (start, end) in enumerate(VALIDATION_TARGET_WINDOWS, start=1):
        for target in month_range(start, end):
            rows.append({"split": "validation", "block": f"validation_{window}", "target": target})
    test_targets = month_range(
        shift_month(TEST_ORIGIN_FIRST, HORIZON_MONTHS),
        shift_month(TEST_ORIGIN_LAST, HORIZON_MONTHS),
    )
    rows += [{"split": "test", "block": "test", "target": target} for target in test_targets]

    frame = pd.DataFrame(rows)
    frame["origin"] = [shift_month(target, -HORIZON_MONTHS) for target in frame["target"]]
    frame["horizon_months"] = HORIZON_MONTHS
    frame["target_period_label"] = [period_label(target) for target in frame["target"]]
    frame["train_first_target"] = first_trainable_target()
    frame["train_last_target"] = frame["origin"]
    frame["n_train_target_months_main"] = [
        len(training_targets(origin, include_pandemic=False)) for origin in frame["origin"]
    ]
    frame["n_train_target_months_all"] = [
        len(training_targets(origin, include_pandemic=True)) for origin in frame["origin"]
    ]
    columns = [
        "split",
        "block",
        "origin",
        "target",
        "horizon_months",
        "target_period_label",
        "train_first_target",
        "train_last_target",
        "n_train_target_months_main",
        "n_train_target_months_all",
    ]
    return frame[columns]


def training_targets(origin: str, include_pandemic: bool = False) -> list[str]:
    """Target months a model fit at ``origin`` may learn from: none later than the origin.

    The main analysis leaves out target months in the pandemic period and January 2020;
    the sensitivity analysis keeps them.
    """
    months = month_range(first_trainable_target(), origin)
    if include_pandemic:
        return months
    return [month for month in months if period_label(month) in MAIN_SAMPLE_PERIODS]


def validate_rolling_origins(origins: pd.DataFrame) -> None:
    test = origins[origins["split"] == "test"]
    validation = origins[origins["split"] == "validation"]
    if test["origin"].tolist() != month_range(TEST_ORIGIN_FIRST, TEST_ORIGIN_LAST):
        raise ValueError("Test origins are not one per month over the planned range")
    if origins["target"].duplicated().any():
        raise ValueError("A target month appears in more than one split")
    if not (origins["origin"] < origins["target"]).all():
        raise ValueError("An origin is not earlier than its target")
    if validation["target"].max() >= TEST_ORIGIN_FIRST:
        raise ValueError("A validation target is not earlier than the first test origin")
    if test["target"].max() > PANEL_END or origins["target"].min() < first_trainable_target():
        raise ValueError("A target month is outside the usable panel window")
    if not origins["target_period_label"].isin(MAIN_SAMPLE_PERIODS).all():
        raise ValueError("A validation or test target lies in the pandemic period")


def split_metadata(origins: pd.DataFrame) -> dict[str, Any]:
    test = origins[origins["split"] == "test"]
    validation = origins[origins["split"] == "validation"]
    blocks = {
        block: {
            "first_origin": group["origin"].min(),
            "last_origin": group["origin"].max(),
            "first_target": group["target"].min(),
            "last_target": group["target"].max(),
            "origins": int(len(group)),
        }
        for block, group in origins.groupby("block", sort=False)
    }
    return {
        "schema_version": "1.0.0",
        "horizon_months": HORIZON_MONTHS,
        "tiers": list(TIERS),
        "definition": (
            "An origin is the last month whose data may be used. The target is the next "
            "month. A model for an origin is fit on target months up to and including the "
            "origin, so no value from the target month or later enters the fit."
        ),
        "blocks": blocks,
        "test_origins": int(len(test)),
        "test_forecasts": int(len(test) * len(TIERS)),
        "validation_origins": int(len(validation)),
        "validation_forecasts": int(len(validation) * len(TIERS)),
        "first_trainable_target": first_trainable_target(),
        "training_rule_main": (
            "target months in the pre-pandemic and post-pandemic periods, from the first "
            "trainable target up to the origin"
        ),
        "training_rule_sensitivity": "the same, with pandemic-period target months kept",
        "use_of_validation": (
            "SARIMAX orders and LightGBM hyperparameters are chosen on the validation "
            "origins only and are then fixed for every test origin"
        ),
        "use_of_test": "each test origin is forecast once, after the choices above are fixed",
    }
