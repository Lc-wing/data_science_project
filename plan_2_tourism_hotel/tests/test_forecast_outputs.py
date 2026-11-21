"""The committed backtest is what the frozen panel and the scoring rules produce."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from macao_tourism import forecast as fc
from macao_tourism.config import OUTPUTS_METRICS_DIR, OUTPUTS_TABLES_DIR, TIERS
from macao_tourism.freeze import Frozen, load_frozen, load_manifest
from macao_tourism.splits import shift_month

ATOL = 1e-8


@pytest.fixture(scope="module")
def frozen() -> Frozen:
    return load_frozen()


@pytest.fixture(scope="module")
def forecasts() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_test_forecasts.csv")


def _same_numbers(committed: pd.DataFrame, computed: pd.DataFrame, columns: list[str]) -> None:
    assert len(committed) == len(computed)
    for column in columns:
        np.testing.assert_allclose(committed[column], computed[column], atol=ATOL, err_msg=column)


def test_test_forecasts_cover_the_frozen_test_origins(
    frozen: Frozen, forecasts: pd.DataFrame
) -> None:
    origins = frozen.origins.loc[frozen.origins["split"] == "test", "origin"].tolist()
    assert origins[0] == "2024-07" and origins[-1] == "2026-08" and len(origins) == 26
    assert forecasts.groupby("run")["origin"].nunique().to_dict() == {
        setting.name: 26 for setting in fc.SETTINGS
    }
    main = forecasts[forecasts["run"] == "main"]
    assert len(main) == 104
    assert main.groupby("origin")["tier"].apply(list).map(lambda tiers: tiers == list(TIERS)).all()
    lookup = frozen.panel.set_index(["tier", "period"])["occupancy_pct"]
    for row in main.itertuples():
        assert row.target == shift_month(row.origin, 1)
        assert row.actual == lookup[(row.tier, row.target)]
        assert row.seasonal_naive == lookup[(row.tier, shift_month(row.target, -12))]
    assert forecasts[["sarimax", "lightgbm"]].stack().between(0, 100).all()
    assert forecasts["sarimax_converged"].all()


def test_committed_scores_follow_from_the_saved_forecasts(
    frozen: Frozen, forecasts: pd.DataFrame
) -> None:
    scales = fc.mase_scales(frozen.panel)
    metrics = pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_test_metrics.csv")
    tests = pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_dm_tests.csv")
    summary = json.loads(
        (OUTPUTS_METRICS_DIR / "forecast_summary.json").read_text(encoding="utf-8")
    )
    manifest = load_manifest()
    assert summary["panel_sha256"] == manifest["files"]["data/frozen/panel_tier_month.csv"]
    assert summary["analysis_spec_sha256"] == manifest["files"]["data/frozen/analysis_spec.json"]
    assert summary["deviations_from_spec"] == []
    assert summary["mase_scales"] == pytest.approx(scales)
    assert len(tests) == 3 * 10
    for setting in fc.SETTINGS:
        part = forecasts[forecasts["run"] == setting.name].reset_index(drop=True)
        scored = fc.metrics_table(part, scales)
        recorded = metrics[metrics["run"] == setting.name].reset_index(drop=True)
        assert recorded[["model", "unit"]].to_dict("list") == scored[["model", "unit"]].to_dict(
            "list"
        )
        _same_numbers(recorded, scored, ["mae", "rmse", "mase"])
        compared = fc.dm_table(part)
        recorded_tests = tests[tests["run"] == setting.name].reset_index(drop=True)
        assert recorded_tests["comparison"].tolist() == compared["comparison"].tolist()
        _same_numbers(
            recorded_tests,
            compared,
            ["mean_loss_difference", "ci_low", "ci_high", "dm_hln", "p_value", "p_holm"],
        )
        verdict = fc.h3_verdict(scored, compared)
        assert summary["h3"][setting.name]["h3_supported"] is verdict["h3_supported"] is False
        assert summary["h3"][setting.name]["pooled_mae"] == pytest.approx(verdict["pooled_mae"])


def test_selection_is_the_lowest_validation_error() -> None:
    selection = json.loads(
        (OUTPUTS_METRICS_DIR / "forecast_selection.json").read_text(encoding="utf-8")
    )
    assert selection["test_months_read"] == 0
    assert selection["validation_origins"] == 36
    sarimax = pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_validation_sarimax.csv")
    for setting in fc.SETTINGS:
        part = sarimax[sarimax["run"] == setting.name]
        chosen = part[part["selected"]]
        assert chosen["tier"].tolist() == list(TIERS)
        for row in chosen.itertuples():
            eligible = part[(part["tier"] == row.tier) & part["eligible"]]
            assert row.validation_mae == pytest.approx(eligible["validation_mae"].min())
            recorded = selection["selection"][setting.name]["sarimax"][row.tier]
            assert str(tuple(recorded["order"])) == row.order
            assert str(tuple(recorded["seasonal_order"])) == row.seasonal_order
            assert recorded["fallback_used"] is False
        grid = pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_validation_lightgbm.csv")
        grid = grid[grid["run"] == setting.name]
        best = grid[grid["selected"]].iloc[0]
        assert best["validation_mae"] == pytest.approx(grid["validation_mae"].min())
        assert selection["selection"][setting.name]["lightgbm"] == {
            "num_leaves": int(best["num_leaves"]),
            "min_data_in_leaf": int(best["min_data_in_leaf"]),
            "n_estimators": int(best["n_estimators"]),
        }


def test_shap_values_add_up_and_rank_the_saved_importance() -> None:
    shap = pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_shap_values.csv")
    assert shap.groupby(["origin", "tier"]).ngroups == 104
    totals = shap.groupby(["origin", "tier"]).agg(
        base=("base_value", "first"),
        contribution=("shap_value", "sum"),
        forecast=("forecast", "first"),
    )
    np.testing.assert_allclose(
        totals["base"] + totals["contribution"], totals["forecast"], atol=1e-6
    )
    importance = pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_shap_importance.csv")
    recomputed = fc.shap_importance(shap)
    assert importance["feature"].tolist() == recomputed["feature"].tolist()
    _same_numbers(importance, recomputed, ["all_classes", *TIERS])
    assert importance["feature"].iloc[0] == "occupancy_pct_lag1"


def test_one_saved_forecast_is_what_a_refit_gives(frozen: Frozen) -> None:
    """Refit the chosen models at the first test origin and compare with the saved forecast."""
    spec = frozen.spec["forecast_h3"]
    selection = json.loads(
        (OUTPUTS_METRICS_DIR / "forecast_selection.json").read_text(encoding="utf-8")
    )["selection"]["main"]
    boosted = spec["models"]["lightgbm"]
    frame = fc.feature_frame(frozen.panel, boosted["lagged_variables"], boosted["lags_months"])
    features = list(boosted["features"])
    origin = "2024-07"
    choice = selection["lightgbm"]
    params = fc.lightgbm_params(
        spec["models"]["lightgbm"]["fixed"], choice["num_leaves"], choice["min_data_in_leaf"]
    )
    booster = fc.lightgbm_fit(frame, features, origin, fc.MAIN, params, choice["n_estimators"])
    rows = fc.target_rows(frame, origin)
    again = fc.lightgbm_forecast(booster, rows, features, fc.MAIN)
    saved = pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_test_forecasts.csv")
    saved = (
        saved[(saved["run"] == "main") & (saved["origin"] == origin)]
        .set_index("tier")
        .loc[list(TIERS)]
    )
    np.testing.assert_allclose(again, saved["lightgbm"], atol=1e-6)

    order = tuple(selection["sarimax"]["5star"]["order"])
    seasonal = tuple(selection["sarimax"]["5star"]["seasonal_order"])
    inputs = fc.sarimax_inputs(
        frozen.panel, "5star", origin, spec["models"]["sarimax"]["exogenous"], fc.MAIN
    )
    forecast, converged = fc.sarimax_one_step(*inputs, order, seasonal)
    assert converged
    assert forecast == pytest.approx(saved.loc["5star", "sarimax"], abs=1e-6)
