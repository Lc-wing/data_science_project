"""Rolling-origin forecasts for H3: seasonal naive, SARIMAX per star class, global LightGBM.

Every function takes the origins it is to forecast. Orders and hyperparameters are chosen
on validation origins and written to a selection file; the test origins are forecast from
that file.
"""

from __future__ import annotations

import os

# The fits run in several processes. Each one must keep to a single thread, or they
# spend the time fighting over the cores.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import ast  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

from macao_tourism.config import TEST_ORIGIN_FIRST, TIERS  # noqa: E402
from macao_tourism.panel import inv_logit_pct, logit_pct  # noqa: E402
from macao_tourism.regression import holm, month_index  # noqa: E402
from macao_tourism.splits import first_trainable_target, shift_month, training_targets  # noqa: E402

ALPHA = 0.05
MODELS: tuple[str, ...] = ("seasonal_naive", "sarimax", "lightgbm")
BASELINES: tuple[str, ...] = ("seasonal_naive", "sarimax")
POOLED = "pooled"
PANDEMIC_MISSING: tuple[str, str] = ("2020-01", "2022-12")
SARIMAX_MAXITER = 200
SHAP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Setting:
    """One run of the backtest: the main analysis or a sensitivity analysis."""

    name: str
    logit: bool = False
    include_pandemic: bool = False


MAIN = Setting("main")
SETTINGS: tuple[Setting, ...] = (
    MAIN,
    Setting("F1_logit", logit=True),
    Setting("F2_pandemic_included", include_pandemic=True),
)


def feature_frame(panel: pd.DataFrame, lagged: list[str], lags: list[int]) -> pd.DataFrame:
    """One row per star class and target month, with lagged values as features.

    A lag-k feature of target month t is the value of month t − k. The calendar columns of
    the target month are kept as they are; no other column of month t becomes a feature.
    """
    parts = []
    for tier in TIERS:
        rows = panel[panel["tier"] == tier].sort_values("period").reset_index(drop=True)
        if (np.diff(month_index(rows["period"])) != 1).any():
            raise ValueError(f"Months of {tier} are not consecutive")
        for name in lagged:
            for lag in lags:
                rows[f"{name}_lag{lag}"] = rows[name].shift(lag)
        parts.append(rows[rows["period"] >= first_trainable_target()])
    frame = pd.concat(parts, ignore_index=True)
    frame["tier"] = pd.Categorical(frame["tier"], categories=list(TIERS))
    return frame


def target_rows(frame: pd.DataFrame, origin: str) -> pd.DataFrame:
    """The four rows forecast from an origin, in the fixed star-class order."""
    rows = frame[frame["period"] == shift_month(origin, 1)].sort_values("tier")
    if rows["tier"].tolist() != list(TIERS):
        raise ValueError(f"Target month of origin {origin} is not complete")
    return rows


def lightgbm_params(fixed: dict[str, Any], num_leaves: int, min_data_in_leaf: int) -> dict:
    return {
        **fixed,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "force_col_wise": True,
    }


def lightgbm_fit(
    frame: pd.DataFrame,
    features: list[str],
    origin: str,
    setting: Setting,
    params: dict[str, Any],
    n_estimators: int,
) -> lgb.Booster:
    """Global model fit on target months no later than the origin."""
    months = training_targets(origin, include_pandemic=setting.include_pandemic)
    train = frame[frame["period"].isin(months)]
    y = train["occupancy_pct"].to_numpy(dtype=float)
    data = lgb.Dataset(
        train[features], logit_pct(y) if setting.logit else y, categorical_feature=["tier"]
    )
    return lgb.train(params, data, num_boost_round=n_estimators)


def lightgbm_forecast(
    booster: lgb.Booster, rows: pd.DataFrame, features: list[str], setting: Setting
) -> np.ndarray:
    raw = booster.predict(rows[features])
    return np.asarray(inv_logit_pct(raw) if setting.logit else raw, dtype=float)


def sarimax_trend(order: tuple[int, int, int], seasonal: tuple[int, int, int, int]) -> str:
    """A constant when the model has no differencing, no trend otherwise."""
    return "c" if order[1] == 0 and seasonal[1] == 0 else "n"


def sarimax_inputs(
    panel: pd.DataFrame, tier: str, origin: str, exogenous: list[str], setting: Setting
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Series up to the origin, its exogenous columns, and those of the target month."""
    rows = panel[panel["tier"] == tier].sort_values("period")
    target = shift_month(origin, 1)
    history = rows[rows["period"] <= origin]
    endog = history["occupancy_pct"].to_numpy(dtype=float)
    if setting.logit:
        endog = np.asarray(logit_pct(endog), dtype=float)
    if not setting.include_pandemic:
        first, last = PANDEMIC_MISSING
        endog = np.where(history["period"].between(first, last), np.nan, endog)
    exog = history[exogenous].to_numpy(dtype=float)
    exog_next = rows.loc[rows["period"] == target, exogenous].to_numpy(dtype=float)
    if exog_next.shape != (1, len(exogenous)):
        raise ValueError(f"No calendar row for target month {target}")
    return endog, exog, exog_next


def sarimax_one_step(
    endog: np.ndarray,
    exog: np.ndarray,
    exog_next: np.ndarray,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int, int],
) -> tuple[float, bool]:
    """One-month-ahead forecast and whether the optimiser reported convergence."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog,
                exog=exog,
                order=order,
                seasonal_order=seasonal,
                trend=sarimax_trend(order, seasonal),
            )
            result = model.fit(disp=False, maxiter=SARIMAX_MAXITER)
            forecast = float(result.forecast(steps=1, exog=exog_next)[0])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return math.nan, False
    converged = bool(result.mle_retvals.get("converged", False)) and math.isfinite(forecast)
    return forecast, converged


def _sarimax_task(task: tuple) -> tuple[float, bool]:
    return sarimax_one_step(*task)


def sarimax_forecasts(
    panel: pd.DataFrame,
    origins: list[str],
    exogenous: list[str],
    setting: Setting,
    candidates: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]],
    workers: int = 1,
) -> pd.DataFrame:
    """Forecasts of every listed candidate, per star class and origin, in percentage points."""
    keys, tasks = [], []
    for tier in TIERS:
        for origin in origins:
            inputs = sarimax_inputs(panel, tier, origin, exogenous, setting)
            for order, seasonal in candidates[tier]:
                keys.append((tier, origin, order, seasonal))
                tasks.append((*inputs, order, seasonal))
    if workers > 1:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(pool.map(_sarimax_task, tasks, chunksize=8))
    else:
        results = [_sarimax_task(task) for task in tasks]
    rows = []
    for (tier, origin, order, seasonal), (forecast, converged) in zip(keys, results, strict=True):
        if setting.logit and math.isfinite(forecast):
            forecast = float(inv_logit_pct(forecast))
        rows.append(
            {
                "tier": tier,
                "origin": origin,
                "target": shift_month(origin, 1),
                "order": str(tuple(order)),
                "seasonal_order": str(tuple(seasonal)),
                "forecast": forecast,
                "converged": converged,
            }
        )
    return pd.DataFrame(rows)


def actuals(panel: pd.DataFrame) -> pd.Series:
    return panel.set_index(["tier", "period"])["occupancy_pct"]


def select_sarimax(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    fallback: tuple[tuple[int, int, int], tuple[int, int, int, int]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Per star class: the converging candidate with the lowest validation MAE."""
    scored = forecasts.copy()
    index = pd.MultiIndex.from_frame(scored[["tier", "target"]])
    scored["abs_error"] = (scored["forecast"] - actual.reindex(index).to_numpy()).abs()
    table = (
        scored.groupby(["tier", "order", "seasonal_order"], sort=False)
        .agg(
            origins=("origin", "size"),
            not_converged=("converged", lambda flags: int((~flags).sum())),
            validation_mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    table["eligible"] = table["not_converged"] == 0
    table["selected"] = False
    chosen: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        eligible = table[(table["tier"] == tier) & table["eligible"]]
        if eligible.empty:
            chosen[tier] = {
                "order": list(fallback[0]),
                "seasonal_order": list(fallback[1]),
                "validation_mae": None,
                "fallback_used": True,
            }
            continue
        best = eligible["validation_mae"].idxmin()
        table.loc[best, "selected"] = True
        chosen[tier] = {
            "order": list(ast.literal_eval(table.loc[best, "order"])),
            "seasonal_order": list(ast.literal_eval(table.loc[best, "seasonal_order"])),
            "validation_mae": float(table.loc[best, "validation_mae"]),
            "fallback_used": False,
        }
    return table, chosen


def lightgbm_validation(
    frame: pd.DataFrame,
    features: list[str],
    origins: list[str],
    setting: Setting,
    fixed: dict[str, Any],
    grid: dict[str, list[int]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pooled validation MAE of every grid point; the lowest is chosen, ties to the earlier."""
    rows = []
    points = itertools.product(grid["num_leaves"], grid["min_data_in_leaf"], grid["n_estimators"])
    for num_leaves, min_data_in_leaf, n_estimators in points:
        params = lightgbm_params(fixed, num_leaves, min_data_in_leaf)
        errors = []
        for origin in origins:
            booster = lightgbm_fit(frame, features, origin, setting, params, n_estimators)
            target = target_rows(frame, origin)
            forecast = lightgbm_forecast(booster, target, features, setting)
            errors.append(np.abs(forecast - target["occupancy_pct"].to_numpy()))
        rows.append(
            {
                "num_leaves": num_leaves,
                "min_data_in_leaf": min_data_in_leaf,
                "n_estimators": n_estimators,
                "forecasts": int(np.size(errors)),
                "validation_mae": float(np.mean(errors)),
            }
        )
    table = pd.DataFrame(rows)
    best = table["validation_mae"].idxmin()
    table["selected"] = table.index == best
    chosen = {
        name: int(table.loc[best, name])
        for name in ("num_leaves", "min_data_in_leaf", "n_estimators")
    }
    return table, chosen


def shap_rows(
    booster: lgb.Booster, rows: pd.DataFrame, features: list[str], origin: str
) -> pd.DataFrame:
    """TreeSHAP (tree-path-dependent) values of the four forecasts of one origin.

    The background is the training window of the origin: the values come from the cover
    counts stored in the trees. Base value + sum of the values must equal the forecast.
    """
    contributions = booster.predict(rows[features], pred_contrib=True)
    forecast = booster.predict(rows[features])
    gap = np.abs(contributions.sum(axis=1) - forecast).max()
    if gap > SHAP_TOLERANCE:
        raise ValueError(f"SHAP values do not add up to the forecast at {origin}: gap {gap}")
    out = []
    for position, tier in enumerate(TIERS):
        for column, feature in enumerate(features):
            value = rows[feature].iloc[position]
            out.append(
                {
                    "origin": origin,
                    "target": shift_month(origin, 1),
                    "tier": tier,
                    "feature": feature,
                    "feature_value": str(value) if feature == "tier" else float(value),
                    "shap_value": float(contributions[position, column]),
                    "base_value": float(contributions[position, -1]),
                    "forecast": float(forecast[position]),
                }
            )
    return pd.DataFrame(out)


def forecast_origins(
    panel: pd.DataFrame,
    frame: pd.DataFrame,
    features: list[str],
    exogenous: list[str],
    origins: list[str],
    setting: Setting,
    selection: dict[str, Any],
    fixed: dict[str, Any],
    fallback: tuple[tuple[int, int, int], tuple[int, int, int, int]],
    workers: int = 1,
    with_shap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts of the three models at the given origins, from an already fixed selection."""
    chosen = {
        tier: [(tuple(spec["order"]), tuple(spec["seasonal_order"]))]
        for tier, spec in selection["sarimax"].items()
    }
    sarimax = sarimax_forecasts(panel, origins, exogenous, setting, chosen, workers)
    for row in sarimax[~np.isfinite(sarimax["forecast"])].itertuples():
        inputs = sarimax_inputs(panel, row.tier, row.origin, exogenous, setting)
        forecast, converged = sarimax_one_step(*inputs, *fallback)
        if setting.logit and math.isfinite(forecast):
            forecast = float(inv_logit_pct(forecast))
        sarimax.loc[row.Index, "order"] = str(tuple(fallback[0]))
        sarimax.loc[row.Index, "seasonal_order"] = str(tuple(fallback[1]))
        sarimax.loc[row.Index, "forecast"] = forecast
        sarimax.loc[row.Index, "converged"] = converged
    sarimax = sarimax.set_index(["tier", "origin"])

    choice = selection["lightgbm"]
    params = lightgbm_params(fixed, choice["num_leaves"], choice["min_data_in_leaf"])
    out, shap = [], []
    for origin in origins:
        rows = target_rows(frame, origin)
        booster = lightgbm_fit(frame, features, origin, setting, params, choice["n_estimators"])
        boosted = lightgbm_forecast(booster, rows, features, setting)
        if with_shap:
            shap.append(shap_rows(booster, rows, features, origin))
        for position, tier in enumerate(TIERS):
            row = rows.iloc[position]
            out.append(
                {
                    "run": setting.name,
                    "origin": origin,
                    "target": row["period"],
                    "tier": tier,
                    "actual": float(row["occupancy_pct"]),
                    "seasonal_naive": float(row["occupancy_pct_lag12"]),
                    "sarimax": float(sarimax.loc[(tier, origin), "forecast"]),
                    "lightgbm": float(boosted[position]),
                    "sarimax_order": sarimax.loc[(tier, origin), "order"],
                    "sarimax_seasonal_order": sarimax.loc[(tier, origin), "seasonal_order"],
                    "sarimax_converged": bool(sarimax.loc[(tier, origin), "converged"]),
                }
            )
    return pd.DataFrame(out), (pd.concat(shap, ignore_index=True) if shap else pd.DataFrame())


def mase_scales(panel: pd.DataFrame) -> dict[str, float]:
    """Mean absolute 12-month difference per star class over main-sample pairs before the test."""
    last = shift_month(TEST_ORIGIN_FIRST, -1)
    scales = {}
    for tier in TIERS:
        rows = panel[panel["tier"] == tier].sort_values("period").reset_index(drop=True)
        difference = (rows["occupancy_pct"] - rows["occupancy_pct"].shift(12)).abs()
        both = (rows["in_main_sample"] == 1) & (rows["in_main_sample"].shift(12) == 1)
        scales[tier] = float(difference[both & (rows["period"] <= last)].mean())
    return scales


def metrics_table(forecasts: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    """MAE, RMSE, and MASE per star class and pooled over the four classes."""
    rows = []
    scale = forecasts["tier"].map(scales).to_numpy()
    for model in MODELS:
        error = (forecasts[model] - forecasts["actual"]).to_numpy()
        for unit in (*TIERS, POOLED):
            inside = np.ones(len(forecasts), bool) if unit == POOLED else forecasts["tier"] == unit
            rows.append(
                {
                    "model": model,
                    "unit": unit,
                    "forecasts": int(np.sum(inside)),
                    "mae": float(np.abs(error[inside]).mean()),
                    "rmse": float(math.sqrt((error[inside] ** 2).mean())),
                    "mase": float((np.abs(error[inside]) / scale[inside]).mean()),
                }
            )
    return pd.DataFrame(rows)


def dm_hln(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float]:
    """Diebold–Mariano test of equal one-step loss with the Harvey–Leybourne–Newbold correction.

    For horizon 1 the long-run variance is the variance of the loss differences, and the
    corrected statistic is compared with Student's t with n − 1 degrees of freedom.
    """
    difference = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(difference)
    mean = float(difference.mean())
    gamma0 = float(((difference - mean) ** 2).mean())
    if gamma0 == 0:
        raise ValueError("The two forecasts have identical losses at every origin")
    dm = mean / math.sqrt(gamma0 / n)
    horizon = 1
    correction = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = dm * correction
    half = stats.t.ppf(1 - ALPHA / 2, n - 1) * math.sqrt(gamma0 / n) / correction
    return {
        "origins": n,
        "mean_loss_difference": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "dm_hln": statistic,
        "df": n - 1,
        "p_value": float(2 * stats.t.sf(abs(statistic), n - 1)),
    }


def dm_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """LightGBM against each baseline, per star class and pooled, with Holm over all tests.

    The loss difference is LightGBM minus baseline: negative means LightGBM has the lower
    absolute error. The pooled loss of an origin is the mean over the four classes.
    """
    loss = forecasts[["origin", "tier"]].copy()
    for model in MODELS:
        loss[model] = (forecasts[model] - forecasts["actual"]).abs()
    rows = []
    for baseline in BASELINES:
        for unit in (*TIERS, POOLED):
            if unit == POOLED:
                per_origin = loss.groupby("origin", sort=True)[list(MODELS)].mean()
            else:
                per_origin = loss[loss["tier"] == unit].sort_values("origin").set_index("origin")
            rows.append(
                {
                    "comparison": f"lightgbm vs {baseline}",
                    "unit": unit,
                    **dm_hln(per_origin["lightgbm"].to_numpy(), per_origin[baseline].to_numpy()),
                }
            )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_value"].tolist())
    return table


def h3_verdict(metrics: pd.DataFrame, tests: pd.DataFrame) -> dict[str, Any]:
    """The decision rule of the specification, applied to the pooled results."""
    pooled = metrics[metrics["unit"] == POOLED].set_index("model")["mae"]
    pooled_tests = tests[tests["unit"] == POOLED].set_index("comparison")
    lower = {b: bool(pooled["lightgbm"] < pooled[b]) for b in BASELINES}
    significant = {
        b: bool(pooled_tests.loc[f"lightgbm vs {b}", "p_holm"] < ALPHA) for b in BASELINES
    }
    return {
        "pooled_mae": {model: float(pooled[model]) for model in MODELS},
        "lightgbm_has_lower_pooled_mae": lower,
        "pooled_holm_p_below_0.05": significant,
        "h3_supported": all(lower.values()) and all(significant.values()),
    }


def shap_importance(shap: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, over all test forecasts and per star class."""
    overall = shap.groupby("feature", sort=False)["shap_value"].apply(lambda v: v.abs().mean())
    by_tier = (
        shap.assign(abs_value=shap["shap_value"].abs())
        .pivot_table(index="feature", columns="tier", values="abs_value", aggfunc="mean")
        .reindex(columns=list(TIERS))
    )
    table = by_tier.assign(all_classes=overall).sort_values("all_classes", ascending=False)
    return table.reset_index()[["feature", "all_classes", *TIERS]]
