"""Milestone M4: Train-background exact TreeSHAP explainability & additivity/regime audit."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from stock_volatility import config
from stock_volatility.baselines import MODEL_FEATURES


def _extract_positive_class_shap(
    shap_values: Any,
    expected_value: Any,
) -> tuple[np.ndarray, float]:
    """Extract positive-class (index 1) SHAP matrix and scalar base value robustly."""
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1], dtype=float)
        ev = float(np.asarray(expected_value, dtype=float).ravel()[1])
        return sv, ev
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        sv = arr[:, :, 1]
        ev_arr = np.asarray(expected_value, dtype=float).ravel()
        ev = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
        return sv, ev
    ev_arr = np.asarray(expected_value, dtype=float).ravel()
    return arr, float(ev_arr[0])


def compute_train_background_treeshap(
    *,
    panel_df: pd.DataFrame,
    reg_model: lgb.LGBMRegressor,
    clf_model: lgb.LGBMClassifier,
    background_n: int = 1000,
    eval_n: int = 2000,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Compute exact interventional TreeSHAP on 2025 Test using strictly 2019-2023 Train bg."""
    feat_cols = list(MODEL_FEATURES)
    classes = [int(c) for c in clf_model.classes_.tolist()]
    if classes != [0, 1]:
        raise ValueError(f"Expected classifier classes_ == [0, 1], got {classes}")

    train_df = panel_df[
        (panel_df["split"] == "train")
        & panel_df["has_full_features"].astype(bool)
        & panel_df["has_full_target"].astype(bool)
    ].sort_values(["date", "ticker"], kind="mergesort")
    test_df = panel_df[
        (panel_df["split"] == "test")
        & panel_df["has_full_features"].astype(bool)
        & panel_df["has_full_target"].astype(bool)
    ].sort_values(["date", "ticker"], kind="mergesort")

    # Train-only VIX median threshold for High/Low VIX SHAP regime audit (zero lookahead)
    vix_train_median = float(train_df.groupby("date")["vix_close"].first().median())

    bg_df = train_df.sample(n=min(background_n, len(train_df)), random_state=seed)
    eval_df = test_df.sample(n=min(eval_n, len(test_df)), random_state=seed).reset_index(drop=True)

    X_bg = bg_df[feat_cols]
    X_eval = eval_df[feat_cols]

    # 1. Classifier TreeSHAP in probability space (positive class 1 = downside tail risk)
    clf_explainer = shap.TreeExplainer(
        clf_model,
        data=X_bg,
        feature_perturbation="interventional",
        model_output="probability",
    )
    clf_sv_raw = clf_explainer.shap_values(X_eval, check_additivity=False)
    clf_sv, clf_ev = _extract_positive_class_shap(clf_sv_raw, clf_explainer.expected_value)
    clf_pred_p = clf_model.predict_proba(X_eval)[:, 1]
    clf_max_err = float(np.max(np.abs((clf_ev + clf_sv.sum(axis=1)) - clf_pred_p)))
    if clf_max_err >= 1e-6:
        raise ValueError(f"Classifier TreeSHAP additivity violation: {clf_max_err:.3e}")

    # 2. Regressor TreeSHAP in raw log-RV space (y_log_rv)
    reg_explainer = shap.TreeExplainer(
        reg_model,
        data=X_bg,
        feature_perturbation="interventional",
        model_output="raw",
    )
    reg_sv = np.asarray(reg_explainer.shap_values(X_eval, check_additivity=False), dtype=float)
    reg_ev = float(np.asarray(reg_explainer.expected_value, dtype=float).ravel()[0])
    reg_pred = reg_model.predict(X_eval)
    reg_max_err = float(np.max(np.abs((reg_ev + reg_sv.sum(axis=1)) - reg_pred)))
    if reg_max_err >= 1e-6:
        raise ValueError(f"Regressor TreeSHAP additivity violation: {reg_max_err:.3e}")

    # 3. Global feature importance table
    imp_rows: list[dict[str, Any]] = []
    for model_type, sv_mat in (("classifier_prob", clf_sv), ("regressor_log_rv", reg_sv)):
        mean_abs = np.mean(np.abs(sv_mat), axis=0)
        mean_signed = np.mean(sv_mat, axis=0)
        std_signed = np.std(sv_mat, axis=0)
        total_abs = float(np.sum(mean_abs))
        order = np.argsort(-mean_abs)
        rank_map = {int(feat_idx): int(r + 1) for r, feat_idx in enumerate(order)}
        for idx, feat in enumerate(feat_cols):
            imp_rows.append(
                {
                    "model_type": model_type,
                    "feature": feat,
                    "rank": rank_map[idx],
                    "mean_abs_shap": float(mean_abs[idx]),
                    "share_pct": float(100.0 * mean_abs[idx] / total_abs) if total_abs > 0 else 0.0,
                    "mean_shap": float(mean_signed[idx]),
                    "std_shap": float(std_signed[idx]),
                }
            )
    importance_df = (
        pd.DataFrame(imp_rows)
        .sort_values(["model_type", "rank"], kind="mergesort")
        .reset_index(drop=True)
    )

    # 4. Regime summary (Low_VIX vs High_VIX, No_Event vs Has_Event)
    regime_masks = [
        ("Low_VIX", (eval_df["vix_close"] < vix_train_median).to_numpy()),
        ("High_VIX", (eval_df["vix_close"] >= vix_train_median).to_numpy()),
        ("No_Event", (eval_df["has_news"] == 0).to_numpy()),
        ("Has_Event", (eval_df["has_news"] == 1).to_numpy()),
    ]
    regime_rows: list[dict[str, Any]] = []
    for model_type, sv_mat in (("classifier_prob", clf_sv), ("regressor_log_rv", reg_sv)):
        for reg_name, mask in regime_masks:
            n_grp = int(mask.sum())
            if n_grp == 0:
                continue
            sub_sv = sv_mat[mask]
            mean_abs = np.mean(np.abs(sub_sv), axis=0)
            mean_signed = np.mean(sub_sv, axis=0)
            total_abs = float(np.sum(mean_abs))
            for idx, feat in enumerate(feat_cols):
                regime_rows.append(
                    {
                        "model_type": model_type,
                        "regime_group": reg_name,
                        "n_obs": n_grp,
                        "feature": feat,
                        "mean_abs_shap": float(mean_abs[idx]),
                        "share_pct": (
                            float(100.0 * mean_abs[idx] / total_abs) if total_abs > 0 else 0.0
                        ),
                        "mean_shap": float(mean_signed[idx]),
                    }
                )
    regime_df = pd.DataFrame(regime_rows)

    top_clf = (
        importance_df[importance_df["model_type"] == "classifier_prob"]
        .head(8)[["feature", "rank", "mean_abs_shap", "share_pct", "mean_shap"]]
        .to_dict(orient="records")
    )
    top_reg = (
        importance_df[importance_df["model_type"] == "regressor_log_rv"]
        .head(8)[["feature", "rank", "mean_abs_shap", "share_pct", "mean_shap"]]
        .to_dict(orient="records")
    )

    audit_payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "milestone": "M4",
        "method": "TreeExplainer (interventional, exact)",
        "background_split": "train",
        "background_date_range": [str(bg_df["date"].min()), str(bg_df["date"].max())],
        "background_n": int(len(bg_df)),
        "eval_split": "test",
        "eval_date_range": [str(eval_df["date"].min()), str(eval_df["date"].max())],
        "eval_n": int(len(eval_df)),
        "random_seed": seed,
        "vix_train_median_threshold": vix_train_median,
        "classifier_classes": classes,
        "classifier_positive_class_index": 1,
        "classifier_output_space": "probability_class_1_uncalibrated_lgbm",
        "classifier_expected_value": clf_ev,
        "classifier_max_additivity_error": clf_max_err,
        "regressor_output_space": "y_log_rv",
        "regressor_expected_value": reg_ev,
        "regressor_max_additivity_error": reg_max_err,
        "additivity_tolerance": 1e-6,
        "non_causal_framing": (
            "TreeSHAP quantifies feature contributions to LightGBM model behavior only; "
            "attributions are strictly non-causal and do not imply economic causation."
        ),
        "top_features_classifier_prob": top_clf,
        "top_features_regressor_log_rv": top_reg,
    }
    return {
        "importance_df": importance_df,
        "regime_df": regime_df,
        "audit_payload": audit_payload,
    }
