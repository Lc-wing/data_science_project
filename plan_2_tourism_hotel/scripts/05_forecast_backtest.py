"""Phase A step 5: rolling-origin backtest for H3.

    uv run python scripts/05_forecast_backtest.py                     # both stages
    uv run python scripts/05_forecast_backtest.py --stage validation  # choose orders and grid
    uv run python scripts/05_forecast_backtest.py --stage test        # forecast the test origins

The validation stage chooses the SARIMAX orders and the LightGBM hyperparameters on the
validation origins and writes them to outputs/metrics/forecast_selection.json. The test
stage reads that file, forecasts the 26 test origins, and scores them. Nothing is chosen
in the test stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from macao_tourism import forecast as fc
from macao_tourism import plots
from macao_tourism.config import (
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_TABLES_DIR,
    PROJECT_ROOT,
    TIERS,
)
from macao_tourism.freeze import Frozen, load_frozen
from macao_tourism.tables import markdown_table, write_csv

DECIMALS = 10
SELECTION_PATH = OUTPUTS_METRICS_DIR / "forecast_selection.json"
SUMMARY_PATH = OUTPUTS_METRICS_DIR / "forecast_summary.json"
FIGURE_FORECASTS = "fig_forecast_test_window"
FIGURE_SHAP = "fig_shap_importance_by_class"
LEFT_OPEN = [
    "SARIMAX is statsmodels SARIMAX with the exogenous columns as regressors and SARIMA "
    "errors, default starting values, L-BFGS with at most 200 iterations, and stationarity "
    "and invertibility enforced. A fit counts as converged when the optimiser says so and "
    "the forecast is finite.",
    "Candidates are ordered with the non-seasonal order outside and the seasonal order "
    "inside, both as listed in the specification.",
    "At a test origin the forecast of the chosen order is used as it comes; fits that did "
    "not converge are counted. If a fit fails or gives no finite forecast, the fallback "
    "order is used for that origin and counted.",
    "LightGBM gets force_col_wise = true, which its deterministic mode asks for. Every "
    "other parameter not in the specification is the library default.",
    "F1 and F2 choose their own orders and hyperparameters on the validation origins, "
    "under their own setting. F1 back-transforms with the inverse logit, without a bias "
    "correction; the lagged occupancy features stay in percent.",
    "Pooled MASE is the mean of the scaled absolute errors, each divided by the scale of "
    "its star class.",
    "The 95% CI of a loss difference uses the same corrected variance and t distribution "
    "as the test.",
    "SHAP values are LightGBM's built-in TreeSHAP (tree-path-dependent), for the main run.",
    "Seasonal-naive validation forecasts for 2023-07 to 2023-12 use pandemic months as "
    "their lag. No test forecast does.",
]


def context(frozen: Frozen) -> dict:
    spec = frozen.spec["forecast_h3"]
    sarimax, boosted = spec["models"]["sarimax"], spec["models"]["lightgbm"]
    features = list(boosted["features"])
    frame = fc.feature_frame(frozen.panel, boosted["lagged_variables"], boosted["lags_months"])
    origins = frozen.origins
    return {
        "spec": spec,
        "frame": frame,
        "features": features,
        "exogenous": list(sarimax["exogenous"]),
        "candidates": [
            (tuple(order), tuple(seasonal))
            for order in sarimax["orders"]
            for seasonal in sarimax["seasonal_orders"]
        ],
        "fallback": (
            tuple(sarimax["fallback"]["order"]),
            tuple(sarimax["fallback"]["seasonal_order"]),
        ),
        "fixed": dict(boosted["fixed"]),
        "grid": {name: list(values) for name, values in boosted["grid"].items()},
        "validation_origins": origins.loc[origins["split"] == "validation", "origin"].tolist(),
        "test_origins": origins.loc[origins["split"] == "test", "origin"].tolist(),
    }


def forecast_with(frozen: Frozen, ctx: dict, origins: list[str], setting, chosen, workers, shap):
    return fc.forecast_origins(
        frozen.panel,
        ctx["frame"],
        ctx["features"],
        ctx["exogenous"],
        origins,
        setting,
        chosen,
        ctx["fixed"],
        ctx["fallback"],
        workers=workers,
        with_shap=shap,
    )


def run_validation(frozen: Frozen, ctx: dict, workers: int) -> dict:
    """Choose orders and hyperparameters on the validation origins; no test month is read."""
    actual = fc.actuals(frozen.panel)
    scales = fc.mase_scales(frozen.panel)
    origins = ctx["validation_origins"]
    sarimax_tables, boosted_tables, metric_tables, chosen = [], [], [], {}
    for setting in fc.SETTINGS:
        print(f"[validation] {setting.name}: fitting SARIMAX candidates", flush=True)
        candidates = {tier: ctx["candidates"] for tier in TIERS}
        forecasts = fc.sarimax_forecasts(
            frozen.panel, origins, ctx["exogenous"], setting, candidates, workers
        )
        table, orders = fc.select_sarimax(forecasts, actual, ctx["fallback"])
        grid, point = fc.lightgbm_validation(
            ctx["frame"], ctx["features"], origins, setting, ctx["fixed"], ctx["grid"]
        )
        chosen[setting.name] = {"sarimax": orders, "lightgbm": point}
        sarimax_tables.append(table.assign(run=setting.name))
        boosted_tables.append(grid.assign(run=setting.name))
        validation, _ = forecast_with(
            frozen, ctx, origins, setting, chosen[setting.name], workers, False
        )
        metric_tables.append(fc.metrics_table(validation, scales).assign(run=setting.name))
        print(f"[validation] {setting.name}: {json.dumps(chosen[setting.name])}", flush=True)

    def stacked(parts: list[pd.DataFrame]) -> pd.DataFrame:
        frame = pd.concat(parts, ignore_index=True)
        return frame[["run", *[c for c in frame.columns if c != "run"]]].round(DECIMALS)

    write_csv(stacked(sarimax_tables), "forecast_validation_sarimax.csv")
    write_csv(stacked(boosted_tables), "forecast_validation_lightgbm.csv")
    write_csv(stacked(metric_tables), "forecast_validation_metrics.csv")
    selection = {
        "schema_version": "1.0.0",
        "step": "Phase A step 5: choices made on the validation origins",
        "panel_sha256": frozen.panel_sha256,
        "analysis_spec_sha256": frozen.spec_sha256,
        "validation_origins": len(origins),
        "validation_targets": ctx["spec"]["validation_targets"],
        "test_months_read": 0,
        "selection": chosen,
    }
    OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def load_selection(frozen: Frozen) -> dict:
    if not SELECTION_PATH.is_file():
        raise SystemExit("No selection file. Run the validation stage first.")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if (
        selection["panel_sha256"] != frozen.panel_sha256
        or selection["analysis_spec_sha256"] != frozen.spec_sha256
    ):
        raise SystemExit("The selection file was made from a different freeze.")
    return selection


def p_text(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def shown(frame: pd.DataFrame, decimals: int = 3) -> str:
    out = frame.copy()
    for column in ("p_value", "p_holm"):
        if column in out:
            out[column] = out[column].map(p_text)
    return markdown_table(out, decimals=decimals)


def verdict_text(verdict: dict) -> str:
    mae = verdict["pooled_mae"]
    lower, significant = (
        verdict["lightgbm_has_lower_pooled_mae"],
        verdict["pooled_holm_p_below_0.05"],
    )
    parts = [
        f"Pooled test MAE: LightGBM {mae['lightgbm']:.2f}, SARIMAX {mae['sarimax']:.2f}, "
        f"seasonal naive {mae['seasonal_naive']:.2f} percentage points."
    ]
    for baseline in fc.BASELINES:
        parts.append(
            f"Against {baseline}: LightGBM MAE is {'lower' if lower[baseline] else 'not lower'}; "
            f"pooled Holm-adjusted p is {'below' if significant[baseline] else 'not below'} 0.05."
        )
    parts.append(
        "By the rule of the specification, H3 is "
        + ("supported." if verdict["h3_supported"] else "not supported.")
    )
    return " ".join(parts)


def render_markdown(summary: dict, tables: dict[str, pd.DataFrame]) -> str:
    spec = summary["specification"]
    selection = pd.DataFrame(
        [
            {
                "run": run,
                **{
                    tier: f"{c['order']} × {c['seasonal_order']}"
                    for tier, c in s["sarimax"].items()
                },
                "lightgbm": json.dumps(s["lightgbm"]),
            }
            for run, s in summary["selection"].items()
        ]
    )

    def of(table: str, run: str) -> pd.DataFrame:
        frame = tables[table]
        return frame[frame["run"] == run].drop(columns="run")

    sections = [
        "# Rolling-origin backtest for H3 (Plan 2, Phase A step 5)",
        "One-month-ahead forecasts of occupancy per star class. The comparison is predictive "
        "only. SHAP values describe how the fitted model behaves; they are not causal effects.",
        f"Frozen panel SHA-256: `{summary['panel_sha256']}`  \n"
        f"Analysis specification SHA-256: `{summary['analysis_spec_sha256']}` "
        f"(version {summary['spec_version']})  \n"
        f"Deviations from the specification: {len(summary['deviations_from_spec'])}",
        "## Design",
        f"- Test: {summary['test']['origins']} origins, {summary['test']['first_origin']} to "
        f"{summary['test']['last_origin']}, {summary['test']['forecasts']} forecasts. Every "
        "model is refit at every origin on months up to the origin.\n"
        f"- Validation: {summary['validation_origins']} origins, all before the first test "
        "origin. Orders and hyperparameters were chosen there and written to "
        "`outputs/metrics/forecast_selection.json` before any test month was forecast.\n"
        f"- {spec['information_set']}",
        "## Choices made on the validation origins",
        markdown_table(selection),
        "Validation metrics of the chosen models (the chosen models were picked on these "
        "months, so their validation errors flatter them). The seasonal-naive error is "
        "large because the targets 2023-07 to 2023-12 use a pandemic month as the "
        "forecast; no test forecast does.",
        shown(of("validation_metrics", "main")),
        "## Test metrics (main run)",
        f"MASE scale: {spec['metrics']['mase_scale']}.",
        shown(of("metrics", "main")),
        "## Tests of equal accuracy (main run)",
        f"{spec['tests']['method']}; loss: {spec['tests']['loss']}; {spec['tests']['sides']}; "
        f"{spec['tests']['multiplicity']}. The loss difference is LightGBM minus baseline, in "
        "percentage points; negative means LightGBM has the lower error.",
        shown(of("dm_tests", "main")),
        "## H3",
        f"Rule: {spec['tests']['h3_rule']}",
        verdict_text(summary["h3"]["main"]),
        "## Sensitivity runs",
    ]
    for run in [s.name for s in fc.SETTINGS[1:]]:
        pooled = of("metrics", run)
        sections += [
            f"### {run}",
            spec["sensitivity_analyses"].get(run, ""),
            shown(pooled[pooled["unit"] == fc.POOLED]),
            shown(of("dm_tests", run).loc[lambda f: f["unit"] == fc.POOLED]),
            verdict_text(summary["h3"][run]),
        ]
    sections += [
        "## SARIMAX fits at the test origins",
        markdown_table(tables["sarimax_fits"]),
        "## SHAP: mean absolute contribution to the forecast (main run, percentage points)",
        f"{spec['shap']['method']}; background: {spec['shap']['background']}; rows: "
        f"{spec['shap']['rows_explained']}. Largest gap between base value plus SHAP values "
        f"and the forecast: {summary['shap']['max_additivity_gap']:.2e}.",
        shown(tables["shap_importance"].head(12)),
        "## Choices the specification left open",
        "\n".join(f"- {item}" for item in summary["left_open_by_the_specification"]),
        "## Figures",
        f"- `outputs/figures/{FIGURE_FORECASTS}.png`: occupancy and the three forecasts over "
        "the test window.\n"
        f"- `outputs/figures/{FIGURE_SHAP}.png`: mean absolute SHAP value by star class.",
    ]
    return "\n\n".join(section for section in sections if section)


def run_test(frozen: Frozen, ctx: dict, selection: dict, workers: int) -> None:
    """Forecast the test origins from the fixed selection and score them."""
    scales = fc.mase_scales(frozen.panel)
    origins = ctx["test_origins"]
    forecasts, metrics, tests, verdicts = [], [], [], {}
    shap = pd.DataFrame()
    for setting in fc.SETTINGS:
        chosen = selection["selection"][setting.name]
        result, values = forecast_with(
            frozen, ctx, origins, setting, chosen, workers, setting == fc.MAIN
        )
        if setting == fc.MAIN:
            shap = values
        run_metrics, run_tests = fc.metrics_table(result, scales), fc.dm_table(result)
        verdicts[setting.name] = fc.h3_verdict(run_metrics, run_tests)
        forecasts.append(result)
        metrics.append(run_metrics.assign(run=setting.name))
        tests.append(run_tests.assign(run=setting.name))
        print(f"[test] {setting.name}: forecasts written", flush=True)

    def stacked(parts: list[pd.DataFrame]) -> pd.DataFrame:
        frame = pd.concat(parts, ignore_index=True)
        return frame[["run", *[c for c in frame.columns if c != "run"]]].round(DECIMALS)

    all_forecasts = stacked(forecasts)
    fits = (
        all_forecasts.groupby(
            ["run", "tier", "sarimax_order", "sarimax_seasonal_order"], sort=False
        )
        .agg(
            origins=("origin", "size"),
            not_converged=("sarimax_converged", lambda f: int((~f).sum())),
        )
        .reset_index()
    )
    tables = {
        "forecasts": all_forecasts,
        "metrics": stacked(metrics),
        "dm_tests": stacked(tests),
        "sarimax_fits": fits,
        "shap_values": shap.round(DECIMALS),
        "shap_importance": fc.shap_importance(shap).round(DECIMALS),
        "validation_metrics": pd.read_csv(OUTPUTS_TABLES_DIR / "forecast_validation_metrics.csv"),
    }
    for name in (
        "forecasts",
        "metrics",
        "dm_tests",
        "sarimax_fits",
        "shap_values",
        "shap_importance",
    ):
        prefix = (
            "forecast_test_" if name in ("forecasts", "metrics", "sarimax_fits") else "forecast_"
        )
        write_csv(tables[name], f"{prefix}{name}.csv")

    main_forecasts = all_forecasts[all_forecasts["run"] == "main"]
    figures = plots.save(
        plots.figure_test_window(main_forecasts), OUTPUTS_FIGURES_DIR, FIGURE_FORECASTS
    )
    figures += plots.save(
        plots.figure_shap_importance(tables["shap_importance"]), OUTPUTS_FIGURES_DIR, FIGURE_SHAP
    )
    gap = max(
        abs(group["base_value"].iloc[0] + group["shap_value"].sum() - group["forecast"].iloc[0])
        for _, group in shap.groupby(["origin", "tier"], sort=False)
    )
    summary = {
        "schema_version": "1.0.0",
        "step": "Phase A step 5: rolling-origin backtest for H3",
        "frozen_at_utc8": frozen.manifest["frozen_at_utc8"],
        "panel_sha256": frozen.panel_sha256,
        "analysis_spec_sha256": frozen.spec_sha256,
        "spec_version": frozen.spec["spec_version"],
        "specification": ctx["spec"],
        "deviations_from_spec": [],
        "left_open_by_the_specification": LEFT_OPEN,
        "validation_origins": selection["validation_origins"],
        "selection": selection["selection"],
        "test": {
            "origins": len(origins),
            "first_origin": origins[0],
            "last_origin": origins[-1],
            "forecasts": int(len(main_forecasts)),
        },
        "mase_scales": scales,
        "metrics": tables["metrics"].to_dict("records"),
        "dm_tests": tables["dm_tests"].to_dict("records"),
        "h3": verdicts,
        "sarimax_fits": fits.to_dict("records"),
        "shap": {
            "forecasts_explained": int(shap.groupby(["origin", "tier"]).ngroups),
            "max_additivity_gap": float(gap),
            "importance": tables["shap_importance"].to_dict("records"),
        },
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figures],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUTS_TABLES_DIR / "forecast_summary.md").write_text(
        render_markdown(summary, tables) + "\n", encoding="utf-8"
    )
    main_metrics = tables["metrics"][tables["metrics"]["run"] == "main"].drop(columns="run")
    print(shown(main_metrics))
    print(shown(tables["dm_tests"][tables["dm_tests"]["run"] == "main"].drop(columns="run")))
    for run, verdict in verdicts.items():
        print(f"{run}: {verdict_text(verdict)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "test", "all"), default="all")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    frozen = load_frozen()
    ctx = context(frozen)
    if args.stage in ("validation", "all"):
        run_validation(frozen, ctx, args.workers)
    if args.stage in ("test", "all"):
        run_test(frozen, ctx, load_selection(frozen), args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
