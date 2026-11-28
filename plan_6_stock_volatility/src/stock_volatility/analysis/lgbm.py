"""Milestone M4: Global LightGBM regressor/classifier, 2024 Val tuning & probability calibration."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from stock_volatility import config
from stock_volatility.analysis.baselines import (
    MODEL_FEATURES,
    duan_smearing_backtransform,
    run_baselines_all_folds,
    slice_fold_data,
)
from stock_volatility.analysis.treeshap import compute_train_background_treeshap
from stock_volatility.evaluation import (
    evaluate_holdout_predictions,
    expected_calibration_error,
    qlike_loss_array,
)
from stock_volatility.panel.freeze import verify_freeze_manifest

PROB_CLIP: float = 1e-6

REGRESSOR_GRID: tuple[dict[str, Any], ...] = (
    {
        "config_id": "reg_l15_m100_n250",
        "num_leaves": 15,
        "min_child_samples": 100,
        "n_estimators": 250,
        "reg_lambda": 5.0,
    },
    {
        "config_id": "reg_l15_m50_n300",
        "num_leaves": 15,
        "min_child_samples": 50,
        "n_estimators": 300,
        "reg_lambda": 1.0,
    },
    {
        "config_id": "reg_l31_m100_n250",
        "num_leaves": 31,
        "min_child_samples": 100,
        "n_estimators": 250,
        "reg_lambda": 5.0,
    },
    {
        "config_id": "reg_l31_m50_n300",
        "num_leaves": 31,
        "min_child_samples": 50,
        "n_estimators": 300,
        "reg_lambda": 1.0,
    },
    {
        "config_id": "reg_l15_m200_n200",
        "num_leaves": 15,
        "min_child_samples": 200,
        "n_estimators": 200,
        "reg_lambda": 10.0,
    },
    {
        "config_id": "reg_l31_m200_n200",
        "num_leaves": 31,
        "min_child_samples": 200,
        "n_estimators": 200,
        "reg_lambda": 10.0,
    },
)

CLASSIFIER_GRID: tuple[dict[str, Any], ...] = (
    {
        "config_id": "clf_l15_m100_w1",
        "num_leaves": 15,
        "min_child_samples": 100,
        "n_estimators": 200,
        "scale_pos_weight": 1.0,
        "reg_lambda": 5.0,
    },
    {
        "config_id": "clf_l15_m100_w3",
        "num_leaves": 15,
        "min_child_samples": 100,
        "n_estimators": 200,
        "scale_pos_weight": 3.0,
        "reg_lambda": 5.0,
    },
    {
        "config_id": "clf_l15_m100_wbal",
        "num_leaves": 15,
        "min_child_samples": 100,
        "n_estimators": 200,
        "scale_pos_weight": "balanced",
        "reg_lambda": 5.0,
    },
    {
        "config_id": "clf_l31_m100_w1",
        "num_leaves": 31,
        "min_child_samples": 100,
        "n_estimators": 200,
        "scale_pos_weight": 1.0,
        "reg_lambda": 5.0,
    },
    {
        "config_id": "clf_l31_m100_w3",
        "num_leaves": 31,
        "min_child_samples": 100,
        "n_estimators": 200,
        "scale_pos_weight": 3.0,
        "reg_lambda": 5.0,
    },
    {
        "config_id": "clf_l31_m200_wbal",
        "num_leaves": 31,
        "min_child_samples": 200,
        "n_estimators": 200,
        "scale_pos_weight": "balanced",
        "reg_lambda": 10.0,
    },
)


def fit_and_apply_calibrator(
    method: str,
    raw_probs: np.ndarray,
    y_true: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit post-hoc probability calibrator (isotonic, platt_sigmoid, or temperature_scaling)."""
    p_clip = np.clip(np.asarray(raw_probs, dtype=float), PROB_CLIP, 1.0 - PROB_CLIP)
    y = np.asarray(y_true, dtype=int)

    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p_clip, y)

        def _apply_iso(p_new: np.ndarray) -> np.ndarray:
            return np.clip(
                iso.predict(np.clip(np.asarray(p_new, dtype=float), PROB_CLIP, 1.0 - PROB_CLIP)),
                0.0,
                1.0,
            )

        return _apply_iso

    z_fit = logit(p_clip)
    if method == "platt_sigmoid":
        lr = LogisticRegression(C=1e6, max_iter=1000, random_state=config.RANDOM_SEED)
        lr.fit(z_fit.reshape(-1, 1), y)

        def _apply_platt(p_new: np.ndarray) -> np.ndarray:
            z_new = logit(np.clip(np.asarray(p_new, dtype=float), PROB_CLIP, 1.0 - PROB_CLIP))
            return np.clip(lr.predict_proba(z_new.reshape(-1, 1))[:, 1], 0.0, 1.0)

        return _apply_platt

    if method == "temperature_scaling":

        def _nll(temp: float) -> float:
            p_t = np.clip(expit(z_fit / temp), PROB_CLIP, 1.0 - PROB_CLIP)
            return float(log_loss(y, p_t, labels=[0, 1]))

        opt = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded")
        best_temp = float(opt.x)

        def _apply_temp(p_new: np.ndarray) -> np.ndarray:
            z_new = logit(np.clip(np.asarray(p_new, dtype=float), PROB_CLIP, 1.0 - PROB_CLIP))
            return np.clip(expit(z_new / best_temp), 0.0, 1.0)

        return _apply_temp

    raise ValueError(f"Unsupported calibration method: {method!r}")


def select_calibrator_on_val_oof(
    val_dates: np.ndarray,
    val_raw_probs: np.ndarray,
    val_y: np.ndarray,
) -> tuple[str, Callable[[np.ndarray], np.ndarray], list[dict[str, Any]]]:
    """Select calibrator via 4-quarter out-of-fold Val CV, then refit winner on full 2024 Val."""
    if len(val_y) < 4:
        raise ValueError(f"Need at least 4 validation rows for OOF calibration, got {len(val_y)}")
    quarters = pd.to_datetime(pd.Series(val_dates)).dt.to_period("Q").astype(str).to_numpy()
    unique_qs = sorted(np.unique(quarters).tolist())
    if len(unique_qs) < 2:
        mid = len(val_y) // 2
        quarters = np.where(np.arange(len(val_y)) < mid, "H1", "H2")
        unique_qs = ["H1", "H2"]
    candidates = ("isotonic", "platt_sigmoid", "temperature_scaling")

    raw_brier = float(brier_score_loss(val_y, val_raw_probs))
    raw_ece = expected_calibration_error(val_y, val_raw_probs, n_bins=10)

    results: list[dict[str, Any]] = []
    for method in candidates:
        oof_probs = np.empty_like(val_raw_probs, dtype=float)
        for q in unique_qs:
            val_m = quarters == q
            fit_m = ~val_m
            cal_fn = fit_and_apply_calibrator(method, val_raw_probs[fit_m], val_y[fit_m])
            oof_probs[val_m] = cal_fn(val_raw_probs[val_m])
        results.append(
            {
                "method": method,
                "oof_val_brier": float(brier_score_loss(val_y, oof_probs)),
                "oof_val_ece": expected_calibration_error(val_y, oof_probs, n_bins=10),
                "raw_val_brier": raw_brier,
                "raw_val_ece": raw_ece,
            }
        )

    best_entry = min(results, key=lambda r: (r["oof_val_brier"], r["oof_val_ece"]))
    best_method = str(best_entry["method"])
    frozen_calibrator = fit_and_apply_calibrator(best_method, val_raw_probs, val_y)
    return best_method, frozen_calibrator, results


def _make_lgbm_regressor(cfg: dict[str, Any]) -> lgb.LGBMRegressor:
    """Instantiate a deterministic single-threaded LightGBM regressor."""
    return lgb.LGBMRegressor(
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=0.03,
        num_leaves=int(cfg["num_leaves"]),
        min_child_samples=int(cfg["min_child_samples"]),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=float(cfg["reg_lambda"]),
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        deterministic=True,
        verbose=-1,
    )


def _make_lgbm_classifier(cfg: dict[str, Any], y_fit: np.ndarray) -> lgb.LGBMClassifier:
    """Instantiate a deterministic single-threaded LightGBM classifier."""
    spw_cfg = cfg["scale_pos_weight"]
    if spw_cfg == "balanced":
        pos = max(1, int(np.sum(y_fit == 1)))
        neg = max(1, int(np.sum(y_fit == 0)))
        spw = float(neg / pos)
    else:
        spw = float(spw_cfg)
    return lgb.LGBMClassifier(
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=0.03,
        num_leaves=int(cfg["num_leaves"]),
        min_child_samples=int(cfg["min_child_samples"]),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=float(cfg["reg_lambda"]),
        scale_pos_weight=spw,
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        deterministic=True,
        verbose=-1,
    )


def run_lgbm_val_tuning_and_test_backtest(
    panel_df: pd.DataFrame,
    *,
    val_fold: dict[str, Any],
    test_folds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tune LightGBM & calibrator strictly on 2024 Val, then refit quarterly on 2025 Test."""
    feat_cols = list(MODEL_FEATURES)
    train_fit_df, val_eval_df = slice_fold_data(panel_df, val_fold)
    X_train = train_fit_df[feat_cols]
    y_train_log = train_fit_df["y_log_rv"].to_numpy(dtype=float)
    y_train_tail = train_fit_df["y_tail"].to_numpy(dtype=int)

    X_val = val_eval_df[feat_cols]
    rv_val_true = val_eval_df["rv_5d_fwd"].to_numpy(dtype=float)
    y_val_tail = val_eval_df["y_tail"].to_numpy(dtype=int)

    # 1. Stage 1 Regressor tuning strictly on 2024 Val QLIKE
    reg_grid_results: list[dict[str, Any]] = []
    best_reg_model: lgb.LGBMRegressor | None = None
    best_reg_cfg: dict[str, Any] = {}
    best_val_qlike = float("inf")
    best_val_rv_hat: np.ndarray | None = None

    for cfg in REGRESSOR_GRID:
        model = _make_lgbm_regressor(cfg)
        model.fit(X_train, y_train_log)
        resid_fit = y_train_log - model.predict(X_train)
        smear, rv_val_hat = duan_smearing_backtransform(model.predict(X_val), resid_fit)
        qlike_val = float(np.mean(qlike_loss_array(rv_val_true, rv_val_hat)))
        rmse_val = float(np.sqrt(np.mean((rv_val_true - rv_val_hat) ** 2)))
        reg_grid_results.append(
            {
                **cfg,
                "train_duan_smearing": smear,
                "val_qlike": qlike_val,
                "val_rmse": rmse_val,
            }
        )
        if qlike_val < best_val_qlike:
            best_val_qlike = qlike_val
            best_reg_cfg = dict(cfg)
            best_reg_model = model
            best_val_rv_hat = rv_val_hat

    # 2. Stage 1 Classifier tuning strictly on 2024 Val PR-AUC
    clf_grid_results: list[dict[str, Any]] = []
    best_clf_model: lgb.LGBMClassifier | None = None
    best_clf_cfg: dict[str, Any] = {}
    best_val_pr = -1.0
    best_val_p_raw: np.ndarray | None = None

    for cfg in CLASSIFIER_GRID:
        model = _make_lgbm_classifier(cfg, y_train_tail)
        model.fit(X_train, y_train_tail)
        p_val_raw = model.predict_proba(X_val)[:, 1]
        pr_val = float(average_precision_score(y_val_tail, p_val_raw))
        brier_val = float(brier_score_loss(y_val_tail, p_val_raw))
        ece_val = expected_calibration_error(y_val_tail, p_val_raw, n_bins=10)
        clf_grid_results.append(
            {
                **cfg,
                "val_pr_auc": pr_val,
                "val_brier_raw": brier_val,
                "val_ece_raw": ece_val,
            }
        )
        if pr_val > best_val_pr:
            best_val_pr = pr_val
            best_clf_cfg = dict(cfg)
            best_clf_model = model
            best_val_p_raw = p_val_raw

    assert best_reg_model is not None and best_val_rv_hat is not None
    assert best_clf_model is not None and best_val_p_raw is not None

    # 3. Stage 1 Probability Calibration selection strictly via 4-quarter OOF CV inside 2024 Val
    best_cal_method, frozen_calibrator, cal_cv_results = select_calibrator_on_val_oof(
        val_eval_df["date"].to_numpy(), best_val_p_raw, y_val_tail
    )
    val_p_cal = frozen_calibrator(best_val_p_raw)

    # 4. Stage 2 Single-Shot 2025 Test via quarterly expanding refits with frozen hyperparams
    test_parts: list[pd.DataFrame] = []
    test_fold_smearing: list[dict[str, Any]] = []
    for fold in test_folds:
        fit_df, eval_df = slice_fold_data(panel_df, fold)
        X_f = fit_df[feat_cols]
        y_f_log = fit_df["y_log_rv"].to_numpy(dtype=float)
        y_f_tail = fit_df["y_tail"].to_numpy(dtype=int)
        X_e = eval_df[feat_cols]

        reg_f = _make_lgbm_regressor(best_reg_cfg)
        reg_f.fit(X_f, y_f_log)
        smear_f, rv_e_hat = duan_smearing_backtransform(
            reg_f.predict(X_e), y_f_log - reg_f.predict(X_f)
        )

        clf_f = _make_lgbm_classifier(best_clf_cfg, y_f_tail)
        clf_f.fit(X_f, y_f_tail)
        p_e_raw = clf_f.predict_proba(X_e)[:, 1]
        p_e_cal = frozen_calibrator(p_e_raw)

        test_fold_smearing.append(
            {
                "fold_id": fold["fold_id"],
                "fit_end_date": fold["fit_end_date"],
                "n_fit_rows": int(len(fit_df)),
                "duan_smearing_factor": smear_f,
            }
        )
        part = eval_df[["ticker", "date"]].copy()
        part["rv_hat_lgbm"] = rv_e_hat
        part["p_tail_lgbm_raw"] = p_e_raw
        part["p_tail_lgbm_cal"] = p_e_cal
        test_parts.append(part)

    val_lgbm_df = val_eval_df[["ticker", "date"]].copy()
    val_lgbm_df["rv_hat_lgbm"] = best_val_rv_hat
    val_lgbm_df["p_tail_lgbm_raw"] = best_val_p_raw
    val_lgbm_df["p_tail_lgbm_cal"] = val_p_cal

    test_lgbm_df = (
        pd.concat(test_parts, ignore_index=True)
        .sort_values(["date", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )

    selection_payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "milestone": "M4",
        "selection_split": "val_2024_only",
        "random_seed": config.RANDOM_SEED,
        "feature_columns": feat_cols,
        "selected_regressor": best_reg_cfg,
        "regressor_val_grid": reg_grid_results,
        "selected_classifier": best_clf_cfg,
        "classifier_val_grid": clf_grid_results,
        "selected_calibrator": {
            "method": best_cal_method,
            "selection_protocol": "4_quarter_oof_cv_inside_2024_val_then_refit_full_val",
            "candidates_oof_val": cal_cv_results,
        },
        "test_quarterly_refits": test_fold_smearing,
    }

    return {
        "val_lgbm_df": val_lgbm_df,
        "test_lgbm_df": test_lgbm_df,
        "stage1_reg_model": best_reg_model,
        "stage1_clf_model": best_clf_model,
        "selection_payload": selection_payload,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_h2_forecast_backtest_pipeline(
    *,
    project_root: Path | None = None,
    outputs_tables_dir: Path | None = None,
    outputs_metrics_dir: Path | None = None,
    outputs_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify frozen manifest and run complete H2 baselines, LightGBM, evaluation & TreeSHAP."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    frozen_dir = root / "data" / "frozen"
    manifest = verify_freeze_manifest(frozen_dir=frozen_dir, project_root=root)
    generated_at_utc = str(manifest["generated_at_utc"])

    tables_dir = Path(outputs_tables_dir) if outputs_tables_dir else (root / "outputs" / "tables")
    metrics_dir = (
        Path(outputs_metrics_dir) if outputs_metrics_dir else (root / "outputs" / "metrics")
    )
    preds_dir = (
        Path(outputs_predictions_dir)
        if outputs_predictions_dir
        else (root / "outputs" / "predictions")
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

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

    # 5. Persist all 14 H2 artifacts deterministically across predictions/, tables/, and metrics/
    val_pred_df.to_parquet(
        preds_dir / "h2_val_predictions.parquet", index=False, compression="zstd"
    )
    test_pred_df.to_parquet(
        preds_dir / "h2_test_predictions.parquet", index=False, compression="zstd"
    )
    base_res["garch_convergence_df"].to_csv(
        tables_dir / "h2_garch_convergence.csv", index=False, lineterminator="\n"
    )
    eval_res["metrics_df"].to_csv(
        tables_dir / "h2_test_metrics.csv", index=False, lineterminator="\n"
    )
    eval_res["by_sector_df"].to_csv(
        tables_dir / "h2_test_by_sector.csv", index=False, lineterminator="\n"
    )
    eval_res["by_event_df"].to_csv(
        tables_dir / "h2_test_by_event_day.csv", index=False, lineterminator="\n"
    )
    eval_res["dm_df"].to_csv(tables_dir / "h2_dm_tests.csv", index=False, lineterminator="\n")
    eval_res["cal_curve_df"].to_csv(
        tables_dir / "h2_calibration_curve.csv", index=False, lineterminator="\n"
    )
    shap_res["importance_df"].to_csv(
        tables_dir / "shap_importance.csv", index=False, lineterminator="\n"
    )
    shap_res["regime_df"].to_csv(
        tables_dir / "shap_regime_summary.csv", index=False, lineterminator="\n"
    )

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
