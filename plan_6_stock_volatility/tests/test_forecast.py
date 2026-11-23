"""Unit, anti-leakage fold, calibration, and committed-output tests for M4 forecasting."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from stock_volatility import config

# ---------------------------------------------------------------------------
# Layer 1: Pure Offline Unit & Anti-Leakage Schedule Tests
# ---------------------------------------------------------------------------


def test_quarterly_expanding_folds_enforce_strict_5d_purge(block_network: None) -> None:
    """Verify Val + 2025 Q1..Q4 expanding folds maintain >= 5d purge and exclude embargoes."""
    from stock_volatility.baselines import build_quarterly_expanding_folds

    panel = pd.read_parquet(config.DATA_FROZEN_DIR / "panel_daily.parquet")
    val_fold, test_folds = build_quarterly_expanding_folds(panel)

    assert val_fold["fold_id"] == "val_2024"
    assert val_fold["fit_end_date"] == config.TRAIN_END
    assert val_fold["eval_start_date"] == config.VAL_START
    assert val_fold["eval_end_date"] == config.VAL_END
    assert val_fold["purge_trading_days"] >= config.HORIZON_DAYS

    assert len(test_folds) == 4
    assert [f["fold_id"] for f in test_folds] == ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]

    all_dates = sorted(panel["date"].unique().tolist())
    date_idx = {d: i for i, d in enumerate(all_dates)}
    total_test_days = 0
    for fold in test_folds:
        fit_end_idx = date_idx[fold["fit_end_date"]]
        eval_start_idx = date_idx[fold["eval_start_date"]]
        # Target at fit_end_date spans [fit_end_idx+1 .. fit_end_idx+5]; must end < eval_start_idx
        assert fit_end_idx + config.HORIZON_DAYS < eval_start_idx, (
            f"Target leakage in {fold['fold_id']}: fit_end={fold['fit_end_date']} "
            f"eval_start={fold['eval_start_date']}"
        )
        assert fold["purge_trading_days"] >= config.HORIZON_DAYS
        total_test_days += fold["n_eval_days"]

    assert total_test_days == 245


def test_duan_smearing_and_gjr_garch_5d_forecast_formulas(block_network: None) -> None:
    """Verify Duan smearing back-transform and closed-form 5-day GJR-GARCH recursion."""
    from stock_volatility.baselines import (
        duan_smearing_backtransform,
        gjr_garch_5d_annualized_vol,
    )

    # Zero residuals -> smearing factor == 1.0
    smear_1, rv_hat = duan_smearing_backtransform(
        y_log_pred=np.array([math.log(0.25 + config.RV_EPS)]),
        train_residuals=np.zeros(50),
    )
    assert math.isclose(smear_1, 1.0, rel_tol=1e-12)
    assert math.isclose(rv_hat[0], 0.25, rel_tol=1e-6)

    # Positive variance in residuals -> smearing factor > 1.0 (Jensen's inequality)
    smear_2, rv_hat_2 = duan_smearing_backtransform(
        y_log_pred=np.array([-10.0]),
        train_residuals=np.array([-0.2, 0.2, -0.2, 0.2]),
    )
    assert smear_2 > 1.0
    assert rv_hat_2[0] >= 1e-4  # floor protects QLIKE from division by zero

    # GJR-GARCH recursion with stationary params: negative return yields higher vol than positive
    rv_after_neg, sig5_neg = gjr_garch_5d_annualized_vol(
        omega=0.05, alpha=0.05, gamma=0.10, beta=0.85, last_ret_pct=-3.0, last_var_pct2=1.5
    )
    rv_after_pos, sig5_pos = gjr_garch_5d_annualized_vol(
        omega=0.05, alpha=0.05, gamma=0.10, beta=0.85, last_ret_pct=+3.0, last_var_pct2=1.5
    )
    assert rv_after_neg > rv_after_pos > 0.0
    assert math.isclose(rv_after_neg, sig5_neg * math.sqrt(252.0 / 5.0), rel_tol=1e-12)


def test_probability_calibrators_reduce_ece_on_miscalibrated_probs(block_network: None) -> None:
    """Verify Platt, Temperature Scaling, and Isotonic calibrators improve calibration."""
    from stock_volatility.evaluation import expected_calibration_error
    from stock_volatility.forecast import fit_and_apply_calibrator

    rng = np.random.default_rng(42)
    logits = rng.normal(-2.2, 1.0, size=2000)
    true_p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, true_p)
    # Artificially under-confident / mis-scaled probabilities (pure temperature distortion)
    raw_p = 1.0 / (1.0 + np.exp(-(logits * 0.4)))
    raw_ece = expected_calibration_error(y, raw_p, n_bins=10)

    for method in ("isotonic", "platt_sigmoid", "temperature_scaling"):
        cal = fit_and_apply_calibrator(method, raw_p[:1200], y[:1200])
        cal_p = cal(raw_p[1200:])
        assert np.all((cal_p >= 0.0) & (cal_p <= 1.0))
        cal_ece = expected_calibration_error(y[1200:], cal_p, n_bins=10)
        assert cal_ece < raw_ece * 0.5, f"{method} failed to cut ECE: {cal_ece} vs {raw_ece}"


# ---------------------------------------------------------------------------
# Layer 2: Committed M4 Prediction & Selection Artifact Contract Tests
# ---------------------------------------------------------------------------


def test_committed_m4_predictions_garch_and_selection_artifacts(block_network: None) -> None:
    """Verify committed M4 prediction parquets, GARCH convergence log, and Val selection JSON."""
    val_pred_path = config.OUTPUTS_TABLES_DIR / "h2_val_predictions.parquet"
    test_pred_path = config.OUTPUTS_TABLES_DIR / "h2_test_predictions.parquet"
    garch_conv_path = config.OUTPUTS_TABLES_DIR / "h2_garch_convergence.csv"
    baselines_json_path = config.OUTPUTS_METRICS_DIR / "h2_baselines.json"
    selection_json_path = config.OUTPUTS_METRICS_DIR / "h2_lgbm_selection.json"

    for p in (
        val_pred_path,
        test_pred_path,
        garch_conv_path,
        baselines_json_path,
        selection_json_path,
    ):
        assert p.is_file(), f"Missing M4 artifact: {p}"

    val_pred = pd.read_parquet(val_pred_path)
    test_pred = pd.read_parquet(test_pred_path)
    assert len(val_pred) == 14_760
    assert len(test_pred) == 14_700

    required_pred_cols = {
        "ticker",
        "sector",
        "date",
        "fold_id",
        "has_news",
        "has_8k_earnings",
        "rv_5d_fwd",
        "y_tail",
        "rv_hat_persistence",
        "rv_hat_gjr_garch",
        "rv_hat_har_rv",
        "rv_hat_lgbm",
        "p_tail_persistence",
        "p_tail_gjr_garch",
        "p_tail_har_rv",
        "p_tail_lgbm_raw",
        "p_tail_lgbm_cal",
    }
    assert required_pred_cols.issubset(set(val_pred.columns))
    assert required_pred_cols.issubset(set(test_pred.columns))
    assert int(test_pred[list(required_pred_cols)].isna().sum().sum()) == 0
    for col in ("rv_hat_persistence", "rv_hat_gjr_garch", "rv_hat_har_rv", "rv_hat_lgbm"):
        assert (test_pred[col] > 0).all()
    for col in (
        "p_tail_persistence",
        "p_tail_gjr_garch",
        "p_tail_har_rv",
        "p_tail_lgbm_raw",
        "p_tail_lgbm_cal",
    ):
        assert ((test_pred[col] >= 0.0) & (test_pred[col] <= 1.0)).all()

    garch_conv = pd.read_csv(garch_conv_path)
    assert len(garch_conv) == 300  # 5 folds (val + Q1..Q4) x 60 tickers
    assert garch_conv["converged"].mean() >= 0.95

    sel = json.loads(selection_json_path.read_text(encoding="utf-8"))
    assert sel["selection_split"] == "val_2024_only"
    assert sel["selected_calibrator"]["method"] in {
        "isotonic",
        "platt_sigmoid",
        "temperature_scaling",
    }
    assert len(sel["regressor_val_grid"]) >= 4
    assert len(sel["classifier_val_grid"]) >= 4
