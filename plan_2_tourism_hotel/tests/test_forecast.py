from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from macao_tourism import forecast as fc
from macao_tourism.config import TIERS
from macao_tourism.freeze import Frozen, load_frozen
from macao_tourism.splits import shift_month

FALLBACK = ((1, 0, 0), (0, 1, 1, 12))
FIXED_CHOICE = {
    "sarimax": {tier: {"order": [1, 0, 0], "seasonal_order": [0, 1, 1, 12]} for tier in TIERS},
    "lightgbm": {"num_leaves": 4, "min_data_in_leaf": 20, "n_estimators": 100},
}
PSEUDO_TEST = ["2019-10", "2024-04"]


@pytest.fixture(scope="module")
def frozen() -> Frozen:
    return load_frozen()


@pytest.fixture(scope="module")
def spec(frozen: Frozen) -> dict:
    return frozen.spec["forecast_h3"]


@pytest.fixture(scope="module")
def frame(frozen: Frozen, spec: dict) -> pd.DataFrame:
    boosted = spec["models"]["lightgbm"]
    return fc.feature_frame(frozen.panel, boosted["lagged_variables"], boosted["lags_months"])


@pytest.fixture(scope="module")
def features(spec: dict) -> list[str]:
    return list(spec["models"]["lightgbm"]["features"])


@pytest.fixture(scope="module")
def pseudo(frozen: Frozen, spec: dict, frame: pd.DataFrame, features: list[str]):
    """The test-stage code run on two validation origins with a fixed choice."""
    return fc.forecast_origins(
        frozen.panel,
        frame,
        features,
        spec["models"]["sarimax"]["exogenous"],
        PSEUDO_TEST,
        fc.MAIN,
        FIXED_CHOICE,
        dict(spec["models"]["lightgbm"]["fixed"]),
        FALLBACK,
        with_shap=True,
    )


def test_feature_frame_lags_are_values_of_earlier_months(
    frozen: Frozen, frame: pd.DataFrame, features: list[str]
) -> None:
    assert set(features) <= set(frame.columns)
    assert frame["period"].min() == "2011-01"
    assert len(frame) == 4 * (201 - 12)
    assert not frame[features].isna().any().any()
    lookup = frozen.panel.set_index(["tier", "period"])
    row = frame[(frame["tier"] == "3star") & (frame["period"] == "2024-08")].iloc[0]
    for name in ("occupancy_pct", "log_rooms", "typhoon_days", "same_day_share_pct"):
        for lag in (1, 2, 3, 12):
            earlier = shift_month("2024-08", -lag)
            assert row[f"{name}_lag{lag}"] == lookup.loc[("3star", earlier), name]
    assert (
        row["cn_holiday_days_at_origin"]
        == lookup.loc[("3star", "2024-08"), "cn_holiday_days_at_origin"]
    )


def test_features_of_a_target_month_do_not_move_when_that_month_changes(
    frozen: Frozen, spec: dict, frame: pd.DataFrame, features: list[str]
) -> None:
    boosted = spec["models"]["lightgbm"]
    changed = frozen.panel.copy()
    later = changed["period"] >= "2024-08"
    for name in boosted["lagged_variables"]:
        changed.loc[later, name] = changed.loc[later, name] + 7.0
    for name in ("cn_holiday_days", "cn_golden_week_days", "visitors_total", "guests"):
        changed.loc[later, name] = 0
    again = fc.feature_frame(changed, boosted["lagged_variables"], boosted["lags_months"])
    before = frame[frame["period"] <= "2024-08"].reset_index(drop=True)
    after = again[again["period"] <= "2024-08"].reset_index(drop=True)
    pd.testing.assert_frame_equal(before[features], after[features])
    assert not before["occupancy_pct"].equals(after["occupancy_pct"])


def test_training_rows_stop_at_the_origin(
    frame: pd.DataFrame, features: list[str], spec: dict
) -> None:
    fixed = dict(spec["models"]["lightgbm"]["fixed"])
    params = fc.lightgbm_params(fixed, 4, 20)
    origin = "2024-04"
    base = fc.lightgbm_fit(frame, features, origin, fc.MAIN, params, 50)
    changed = frame.copy()
    after = changed["period"] > origin
    changed.loc[after, "occupancy_pct"] = 1.0
    changed.loc[after, "occupancy_pct_lag1"] = 1.0
    again = fc.lightgbm_fit(changed, features, origin, fc.MAIN, params, 50)
    rows = fc.target_rows(frame, origin)
    np.testing.assert_array_equal(
        fc.lightgbm_forecast(base, rows, features, fc.MAIN),
        fc.lightgbm_forecast(again, rows, features, fc.MAIN),
    )
    assert base.num_trees() == 50
    assert fc.lightgbm_fit(frame, features, origin, fc.MAIN, params, 50).model_to_string() == (
        base.model_to_string()
    )


def test_sarimax_inputs_hide_the_pandemic_and_everything_after_the_origin(
    frozen: Frozen, spec: dict
) -> None:
    exogenous = spec["models"]["sarimax"]["exogenous"]
    endog, exog, exog_next = fc.sarimax_inputs(frozen.panel, "4star", "2024-04", exogenous, fc.MAIN)
    assert len(endog) == len(exog) == 172
    assert np.isnan(endog).sum() == 36
    assert np.isnan(endog[120:156]).all() and not np.isnan(endog[156:]).any()
    lookup = frozen.panel.set_index(["tier", "period"])
    assert endog[-1] == lookup.loc[("4star", "2024-04"), "occupancy_pct"]
    assert exog_next.tolist() == [lookup.loc[("4star", "2024-05"), exogenous].tolist()]
    kept, _, _ = fc.sarimax_inputs(
        frozen.panel, "4star", "2024-04", exogenous, fc.Setting("x", include_pandemic=True)
    )
    assert not np.isnan(kept).any()
    logit, _, _ = fc.sarimax_inputs(
        frozen.panel, "4star", "2019-10", exogenous, fc.Setting("x", logit=True)
    )
    assert logit[-1] == pytest.approx(math.log(kept[117] / (100 - kept[117])))
    assert fc.sarimax_trend((1, 0, 0), (1, 0, 0, 12)) == "c"
    assert fc.sarimax_trend((1, 0, 0), (0, 1, 1, 12)) == "n"
    assert fc.sarimax_trend((0, 1, 1), (1, 0, 1, 12)) == "n"


def test_pseudo_test_forecasts_have_the_expected_shape(pseudo, frozen: Frozen) -> None:
    forecasts, shap = pseudo
    assert len(forecasts) == 8
    assert forecasts["target"].tolist() == ["2019-11"] * 4 + ["2024-05"] * 4
    assert forecasts["tier"].tolist() == list(TIERS) * 2
    lookup = frozen.panel.set_index(["tier", "period"])["occupancy_pct"]
    for row in forecasts.itertuples():
        assert row.actual == lookup[(row.tier, row.target)]
        assert row.seasonal_naive == lookup[(row.tier, shift_month(row.target, -12))]
        assert 0 < row.sarimax < 100 and 0 < row.lightgbm < 100
    assert forecasts["sarimax_converged"].all()
    assert len(shap) == 8 * 33


def test_shap_values_add_up_to_the_forecast(pseudo) -> None:
    forecasts, shap = pseudo
    totals = shap.groupby(["origin", "tier"], sort=False).agg(
        base=("base_value", "first"),
        contribution=("shap_value", "sum"),
        forecast=("forecast", "first"),
    )
    np.testing.assert_allclose(
        totals["base"] + totals["contribution"], totals["forecast"], atol=1e-6
    )
    np.testing.assert_allclose(totals["forecast"].to_numpy(), forecasts["lightgbm"].to_numpy())
    importance = fc.shap_importance(shap)
    assert len(importance) == 33
    assert importance["all_classes"].is_monotonic_decreasing
    np.testing.assert_allclose(importance[list(TIERS)].mean(axis=1), importance["all_classes"])


def test_logit_setting_returns_percent(frozen: Frozen, spec: dict, frame, features) -> None:
    forecasts, _ = fc.forecast_origins(
        frozen.panel,
        frame,
        features,
        spec["models"]["sarimax"]["exogenous"],
        ["2019-10"],
        fc.Setting("F1_logit", logit=True),
        FIXED_CHOICE,
        dict(spec["models"]["lightgbm"]["fixed"]),
        FALLBACK,
    )
    assert forecasts[["sarimax", "lightgbm"]].stack().between(50, 100).all()


def test_selection_takes_the_lowest_mae_among_converging_candidates() -> None:
    actual = pd.Series(
        [80.0, 80.0], index=pd.MultiIndex.from_tuples([("5star", "2018-01"), ("5star", "2018-02")])
    )
    rows = []
    candidates = {"(1, 0, 0)": [81, 81], "(1, 0, 1)": [80.5, 80.5], "(0, 1, 1)": [80, 80]}
    for order, values in candidates.items():
        for target, value in zip(["2018-01", "2018-02"], values, strict=True):
            rows.append(
                {
                    "tier": "5star",
                    "origin": shift_month(target, -1),
                    "target": target,
                    "order": order,
                    "seasonal_order": "(0, 1, 1, 12)",
                    "forecast": value,
                    "converged": not (order == "(0, 1, 1)" and target == "2018-02"),
                }
            )
    table, chosen = fc.select_sarimax(pd.DataFrame(rows), actual, FALLBACK)
    assert chosen["5star"]["order"] == [1, 0, 1]
    assert chosen["5star"]["validation_mae"] == pytest.approx(0.5)
    assert table.set_index("order")["eligible"].to_dict() == {
        "(1, 0, 0)": True,
        "(1, 0, 1)": True,
        "(0, 1, 1)": False,
    }
    assert chosen["4star"] == {
        "order": [1, 0, 0],
        "seasonal_order": [0, 1, 1, 12],
        "validation_mae": None,
        "fallback_used": True,
    }


def test_mase_scale_uses_main_sample_pairs_before_the_test(frozen: Frozen) -> None:
    scales = fc.mase_scales(frozen.panel)
    rows = frozen.panel[frozen.panel["tier"] == "2star"].set_index("period")["occupancy_pct"]
    months = pd.period_range("2011-01", "2019-12", freq="M").astype(str).tolist()
    months += pd.period_range("2024-01", "2024-06", freq="M").astype(str).tolist()
    expected = np.mean([abs(rows[m] - rows[shift_month(m, -12)]) for m in months])
    assert len(months) == 114
    assert scales["2star"] == pytest.approx(expected)


def invented_forecasts() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    origins = pd.period_range("2030-01", periods=26, freq="M").astype(str)
    rows = []
    for origin in origins:
        for tier in TIERS:
            rows.append(
                {
                    "origin": origin,
                    "tier": tier,
                    "actual": 80.0,
                    "seasonal_naive": 80 + rng.normal(0, 4),
                    "sarimax": 80 + rng.normal(0, 3),
                    "lightgbm": 80 + rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


def test_metrics_on_invented_forecasts() -> None:
    forecasts = invented_forecasts()
    scales = {tier: 2.0 for tier in TIERS}
    table = fc.metrics_table(forecasts, scales).set_index(["model", "unit"])
    assert len(table) == 3 * 5
    error = forecasts["sarimax"] - forecasts["actual"]
    assert table.loc[("sarimax", "pooled"), "mae"] == pytest.approx(error.abs().mean())
    assert table.loc[("sarimax", "pooled"), "rmse"] == pytest.approx(math.sqrt((error**2).mean()))
    assert table.loc[("sarimax", "pooled"), "mase"] == pytest.approx(error.abs().mean() / 2)
    assert table.loc[("sarimax", "pooled"), "forecasts"] == 104
    five = error[forecasts["tier"] == "5star"]
    assert table.loc[("sarimax", "5star"), "mae"] == pytest.approx(five.abs().mean())
    assert table.loc[("sarimax", "pooled"), "mae"] == pytest.approx(
        table.loc["sarimax"].loc[list(TIERS), "mae"].mean()
    )


def test_dm_hln_at_horizon_one_is_the_paired_t_test() -> None:
    rng = np.random.default_rng(6)
    a, b = rng.gamma(2, 1, 26), rng.gamma(2, 1.3, 26)
    result = fc.dm_hln(a, b)
    reference = stats.ttest_rel(a, b)
    assert result["dm_hln"] == pytest.approx(reference.statistic)
    assert result["p_value"] == pytest.approx(reference.pvalue)
    assert result["df"] == 25
    low, high = reference.confidence_interval(0.95)
    assert (result["ci_low"], result["ci_high"]) == pytest.approx((low, high))
    with pytest.raises(ValueError, match="identical losses"):
        fc.dm_hln(a, a)


def test_dm_table_and_h3_rule_on_invented_forecasts() -> None:
    forecasts = invented_forecasts()
    tests = fc.dm_table(forecasts)
    assert len(tests) == 10
    assert tests["comparison"].value_counts().to_dict() == {
        "lightgbm vs seasonal_naive": 5,
        "lightgbm vs sarimax": 5,
    }
    assert (tests["origins"] == 26).all()
    assert (tests["p_holm"] >= tests["p_value"]).all()
    assert tests["p_holm"].min() == pytest.approx(min(1, 10 * tests["p_value"].min()))
    pooled = tests[tests["unit"] == "pooled"].iloc[0]
    loss = forecasts.assign(
        a=(forecasts["lightgbm"] - 80).abs(), b=(forecasts["seasonal_naive"] - 80).abs()
    )
    per_origin = loss.groupby("origin")[["a", "b"]].mean()
    assert pooled["mean_loss_difference"] == pytest.approx(
        (per_origin["a"] - per_origin["b"]).mean()
    )
    metrics = fc.metrics_table(forecasts, {tier: 1.0 for tier in TIERS})
    assert fc.h3_verdict(metrics, tests)["h3_supported"] is True

    swapped = forecasts.rename(columns={"lightgbm": "sarimax", "sarimax": "lightgbm"})
    verdict = fc.h3_verdict(
        fc.metrics_table(swapped, {tier: 1.0 for tier in TIERS}), fc.dm_table(swapped)
    )
    assert verdict["lightgbm_has_lower_pooled_mae"] == {"seasonal_naive": True, "sarimax": False}
    assert verdict["h3_supported"] is False

    weak = forecasts.copy()
    weak["lightgbm"] = weak["sarimax"] + np.where(np.arange(len(weak)) % 2 == 0, -0.01, 0.0099)
    verdict = fc.h3_verdict(
        fc.metrics_table(weak, {tier: 1.0 for tier in TIERS}), fc.dm_table(weak)
    )
    assert verdict["h3_supported"] is False
