"""CLI entry point for Milestone M4 Step 05: H2 Out-of-Sample Backtest, HLN-DM & TreeSHAP."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from stock_volatility import config
from stock_volatility.baselines import run_baselines_all_folds
from stock_volatility.evaluation import evaluate_holdout_predictions
from stock_volatility.explain import compute_train_background_treeshap
from stock_volatility.forecast import run_lgbm_val_tuning_and_test_backtest
from stock_volatility.panel.freeze import verify_freeze_manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_h2_forecast_backtest_pipeline(
    *,
    project_root: Path | None = None,
    outputs_tables_dir: Path | None = None,
    outputs_metrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify frozen manifest and run complete M4 baselines, LightGBM, evaluation & TreeSHAP."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    frozen_dir = root / "data" / "frozen"
    manifest = verify_freeze_manifest(frozen_dir=frozen_dir, project_root=root)
    generated_at_utc = str(manifest["generated_at_utc"])

    tables_dir = Path(outputs_tables_dir) if outputs_tables_dir else (root / "outputs" / "tables")
    metrics_dir = (
        Path(outputs_metrics_dir) if outputs_metrics_dir else (root / "outputs" / "metrics")
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    panel_df = pd.read_parquet(frozen_dir / "panel_daily.parquet")

    # 1. Baselines (Persistence, GJR-GARCH(1,1), Predictive HAR-RV) on Val + 2025 Q1..Q4 Test
    base_res = run_baselines_all_folds(panel_df)
    base_res["baselines_summary"]["generated_at_utc"] = generated_at_utc

    # 2. Global LightGBM tuning + 2024 Val OOF calibrator selection + 2025 quarterly Test refits
    lgbm_res = run_lgbm_val_tuning_and_test_backtest(
        panel_df,
        val_fold=base_res["val_fold"],
        test_folds=base_res["test_folds"],
    )
    lgbm_res["selection_payload"]["generated_at_utc"] = generated_at_utc

    # Merge LightGBM predictions with baseline predictions
    val_pred_df = base_res["val_pred_df"].merge(
        lgbm_res["val_lgbm_df"], on=["ticker", "date"], how="left", validate="one_to_one"
    )
    test_pred_df = base_res["test_pred_df"].merge(
        lgbm_res["test_lgbm_df"], on=["ticker", "date"], how="left", validate="one_to_one"
    )

    # 3. Formal 2025 Holdout evaluation (metrics + date-block bootstrap, HLN-DM + Holm, ECE)
    eval_res = evaluate_holdout_predictions(
        test_pred_df,
        val_pred_df=val_pred_df,
        block_len=5,
        n_boot=1000,
        seed=config.RANDOM_SEED,
    )
    eval_res["summary_json"]["generated_at_utc"] = generated_at_utc

    # 4. Train-background exact TreeSHAP explainability audit
    shap_res = compute_train_background_treeshap(
        panel_df=panel_df,
        reg_model=lgbm_res["stage1_reg_model"],
        clf_model=lgbm_res["stage1_clf_model"],
        background_n=1000,
        eval_n=2000,
        seed=config.RANDOM_SEED,
    )
    shap_res["audit_payload"]["generated_at_utc"] = generated_at_utc

    # 5. Persist all 14 M4 artifacts deterministically
    val_pred_df.to_parquet(
        tables_dir / "h2_val_predictions.parquet", index=False, compression="zstd"
    )
    test_pred_df.to_parquet(
        tables_dir / "h2_test_predictions.parquet", index=False, compression="zstd"
    )
    base_res["garch_convergence_df"].to_csv(tables_dir / "h2_garch_convergence.csv", index=False)
    eval_res["metrics_df"].to_csv(tables_dir / "h2_test_metrics.csv", index=False)
    eval_res["by_sector_df"].to_csv(tables_dir / "h2_test_by_sector.csv", index=False)
    eval_res["by_event_df"].to_csv(tables_dir / "h2_test_by_event_day.csv", index=False)
    eval_res["dm_df"].to_csv(tables_dir / "h2_dm_tests.csv", index=False)
    eval_res["cal_curve_df"].to_csv(tables_dir / "h2_calibration_curve.csv", index=False)
    shap_res["importance_df"].to_csv(tables_dir / "shap_importance.csv", index=False)
    shap_res["regime_df"].to_csv(tables_dir / "shap_regime_summary.csv", index=False)

    _write_json(metrics_dir / "h2_baselines.json", base_res["baselines_summary"])
    _write_json(metrics_dir / "h2_lgbm_selection.json", lgbm_res["selection_payload"])
    _write_json(metrics_dir / "h2_evaluation.json", eval_res["summary_json"])
    _write_json(metrics_dir / "shap_audit.json", shap_res["audit_payload"])

    return {
        "milestone": "M4",
        "generated_at_utc": generated_at_utc,
        "val_rows": int(len(val_pred_df)),
        "test_rows": int(len(test_pred_df)),
        "garch_convergence_rate": base_res["baselines_summary"]["garch_convergence_rate"],
        "selected_calibrator": lgbm_res["selection_payload"]["selected_calibrator"]["method"],
        "shap_clf_max_err": shap_res["audit_payload"]["classifier_max_additivity_error"],
        "shap_reg_max_err": shap_res["audit_payload"]["regressor_max_additivity_error"],
    }


def main() -> int:
    """Run deterministic M4 forecasting, evaluation & TreeSHAP pipeline."""
    summary = run_h2_forecast_backtest_pipeline()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
