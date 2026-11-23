"""Unit, HLN-DM, block-bootstrap, TreeSHAP additivity, and M4 contract tests."""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np
import pandas as pd

from stock_volatility import config

# ---------------------------------------------------------------------------
# Layer 1: Pure Offline Metric, HLN-DM & Bootstrap Unit Tests
# ---------------------------------------------------------------------------


def test_qlike_and_ece_exact_mathematical_properties(block_network: None) -> None:
    """Verify QLIKE is minimized uniquely at rv_hat == rv (0 loss) and ECE matches bin formula."""
    from stock_volatility.evaluation import (
        expected_calibration_error,
        qlike_loss_array,
    )

    rv = np.array([0.20, 0.35, 0.50])
    # Exact forecast -> QLIKE == 0
    assert np.allclose(qlike_loss_array(rv, rv), 0.0, atol=1e-14)
    # Any perturbation -> strictly positive QLIKE
    assert (qlike_loss_array(rv, rv * 1.2) > 0.0).all()
    assert (qlike_loss_array(rv, rv * 0.8) > 0.0).all()
    # Under-prediction penalized more heavily than equal-ratio over-prediction in QLIKE
    assert qlike_loss_array(np.array([0.30]), np.array([0.15]))[0] > (
        qlike_loss_array(np.array([0.30]), np.array([0.60]))[0]
    )

    # Exact ECE check: 50 obs at p=0.2 (10 pos -> err 0), 50 obs at p=0.8 (30 pos -> err 0.2)
    y = np.array([1] * 10 + [0] * 40 + [1] * 30 + [0] * 20)
    p = np.array([0.2] * 50 + [0.8] * 50)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert math.isclose(ece, 0.10, abs_tol=1e-12)


def test_hln_dm_h5_bartlett_psd_and_date_block_bootstrap(block_network: None) -> None:
    """Verify h=5 HLN-DM detects genuine loss improvement and remains PSD with serial corr."""
    from stock_volatility.evaluation import date_block_bootstrap_ci, hln_dm_test

    rng = np.random.default_rng(42)
    # Create autocorrelated MA(4) loss differential with negative mean (-0.05 => model A better)
    eps = rng.normal(0.0, 0.04, size=260)
    d = np.array([-0.05 + np.mean(eps[i : i + 5]) for i in range(245)])
    res = hln_dm_test(d, horizon=5)
    assert res["mean_diff"] < 0.0
    assert res["long_run_var"] > 0.0
    assert res["dm_hln_stat"] < -3.0
    assert res["p_value"] < 0.01
    assert res["df"] == 244

    # Date-block bootstrap determinism
    dates = np.repeat(np.arange(50), 4)
    vals = rng.normal(0.25, 0.05, size=200)
    ci_a = date_block_bootstrap_ci(
        dates, lambda idx: float(np.mean(vals[idx])), block_len=5, n_boot=200, seed=42
    )
    ci_b = date_block_bootstrap_ci(
        dates, lambda idx: float(np.mean(vals[idx])), block_len=5, n_boot=200, seed=42
    )
    assert ci_a == ci_b
    assert ci_a[0] < float(np.mean(vals)) < ci_a[1]


# ---------------------------------------------------------------------------
# Layer 2: Committed M4 Evaluation, TreeSHAP & Milestone Doc Contract Tests
# ---------------------------------------------------------------------------


def test_committed_m4_evaluation_dm_and_calibration_contracts(block_network: None) -> None:
    """Verify committed 2025 Holdout metrics, sector/event breakdowns, HLN-DM & calibration."""
    req_files = (
        config.OUTPUTS_TABLES_DIR / "h2_test_metrics.csv",
        config.OUTPUTS_TABLES_DIR / "h2_test_by_sector.csv",
        config.OUTPUTS_TABLES_DIR / "h2_test_by_event_day.csv",
        config.OUTPUTS_TABLES_DIR / "h2_dm_tests.csv",
        config.OUTPUTS_TABLES_DIR / "h2_calibration_curve.csv",
        config.OUTPUTS_METRICS_DIR / "h2_evaluation.json",
    )
    for p in req_files:
        assert p.is_file(), f"Missing M4 evaluation artifact: {p}"

    metrics_df = pd.read_csv(config.OUTPUTS_TABLES_DIR / "h2_test_metrics.csv")
    assert set(metrics_df["model"]) == {"persistence", "gjr_garch", "har_rv", "lgbm"}
    for col in (
        "qlike",
        "qlike_ci95_low",
        "qlike_ci95_high",
        "rmse",
        "mae",
        "pr_auc",
        "roc_auc",
        "brier",
        "ece",
    ):
        assert col in metrics_df.columns
        assert metrics_df[col].notna().all()

    by_sec = pd.read_csv(config.OUTPUTS_TABLES_DIR / "h2_test_by_sector.csv")
    assert set(by_sec["sector"]) == set(config.SECTORS)
    assert set(by_sec["model"]) == {"persistence", "gjr_garch", "har_rv", "lgbm"}

    by_ev = pd.read_csv(config.OUTPUTS_TABLES_DIR / "h2_test_by_event_day.csv")
    assert {"No_Event", "Has_Event", "Has_8K_Earnings"}.issubset(set(by_ev["event_group"]))

    dm_df = pd.read_csv(config.OUTPUTS_TABLES_DIR / "h2_dm_tests.csv")
    assert {"loss_metric", "baseline_model", "dm_hln_stat", "p_value", "p_holm"}.issubset(
        set(dm_df.columns)
    )
    assert set(dm_df["baseline_model"]) == {"persistence", "gjr_garch", "har_rv"}
    assert {"qlike", "se", "brier"}.issubset(set(dm_df["loss_metric"]))

    cal_df = pd.read_csv(config.OUTPUTS_TABLES_DIR / "h2_calibration_curve.csv")
    assert {"lgbm_raw", "lgbm_cal"}.issubset(set(cal_df["model_variant"]))

    eval_json = json.loads(
        (config.OUTPUTS_METRICS_DIR / "h2_evaluation.json").read_text(encoding="utf-8")
    )
    assert eval_json["bootstrap_config"] == {
        "type": "date_block",
        "block_len": 5,
        "B": 1000,
        "seed": 42,
    }
    assert eval_json["test_scorable_rows"] == 14_700
    assert eval_json["test_scorable_days"] == 245


def test_committed_m4_treeshap_audit_and_additivity(block_network: None) -> None:
    """Verify Train-background TreeSHAP additivity < 1e-6, classes_ check, and regime tables."""
    imp_path = config.OUTPUTS_TABLES_DIR / "shap_importance.csv"
    reg_path = config.OUTPUTS_TABLES_DIR / "shap_regime_summary.csv"
    audit_path = config.OUTPUTS_METRICS_DIR / "shap_audit.json"
    for p in (imp_path, reg_path, audit_path):
        assert p.is_file(), f"Missing M4 SHAP artifact: {p}"

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["background_split"] == "train"
    assert audit["background_n"] == 1000
    assert audit["classifier_classes"] == [0, 1]
    assert audit["classifier_positive_class_index"] == 1
    assert audit["classifier_max_additivity_error"] < 1e-6
    assert audit["regressor_max_additivity_error"] < 1e-6
    assert "model behavior" in audit["non_causal_framing"].lower()

    imp_df = pd.read_csv(imp_path)
    assert set(imp_df["model_type"]) == {"classifier_prob", "regressor_log_rv"}
    assert len(imp_df[imp_df["model_type"] == "classifier_prob"]) == 22
    assert len(imp_df[imp_df["model_type"] == "regressor_log_rv"]) == 22

    reg_df = pd.read_csv(reg_path)
    assert {"Low_VIX", "High_VIX", "No_Event", "Has_Event"}.issubset(set(reg_df["regime_group"]))


def test_m4_milestone_delivery_doc_complete_linked_and_not_gitignored() -> None:
    """Verify M4 milestone delivery doc exists, records deliverables + AI log, & is tracked."""
    repo_root = config.PROJECT_ROOT.parent
    milestones_dir = config.PROJECT_ROOT / "docs" / "milestones"
    m4_doc = milestones_dir / "M4_h2_forecasting_calibration_and_shap.md"
    assert m4_doc.is_file(), f"Missing M4 milestone doc: {m4_doc}"
    assert m4_doc.stat().st_size > 2000

    m4_text = m4_doc.read_text(encoding="utf-8")
    for token in (
        "d7a35b611e53366b4d658f0ccaaa8713d3fed267",
        "h2_test_metrics.csv",
        "h2_dm_tests.csv",
        "h2_calibration_curve.csv",
        "h2_garch_convergence.csv",
        "h2_evaluation.json",
        "shap_importance.csv",
        "shap_regime_summary.csv",
        "shap_audit.json",
        "05_forecast_backtest.py",
        "HLN",
        "TreeSHAP",
    ):
        assert token in m4_text, f"Missing {token!r} in M4_h2_forecasting_calibration_and_shap.md"

    ledger_text = (milestones_dir / "README.md").read_text(encoding="utf-8")
    assert "d7a35b611e53366b4d658f0ccaaa8713d3fed267" in ledger_text
    assert "M4_h2_forecasting_calibration_and_shap.md" in ledger_text

    plan6_readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/milestones/M4_h2_forecasting_calibration_and_shap.md" in plan6_readme

    proc = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "--no-index",
            "plan_6_stock_volatility/docs/milestones/M4_h2_forecasting_calibration_and_shap.md",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 1, f"M4 milestone doc must NOT be gitignored:\n{proc.stdout}"
    assert proc.stdout.strip() == ""
