"""Tier 1 Opaque-Box E2E Suite: Complete Feature Coverage across Plan 6 Phase A (M0-M5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stock_volatility import config
from stock_volatility.ingest import ProvenanceLog
from stock_volatility.panel.freeze import FREEZE_ARTIFACT_NAMES, verify_freeze_manifest


def test_t1_01_repo_scaffold_and_gitignore_hygiene(project_root: Path) -> None:
    """F1-F2: Verify Plan 6 project structure, pyproject, README, and .gitignore rules."""
    for rel in (
        "README.md",
        "pyproject.toml",
        ".gitignore",
        "docs/milestones/README.md",
        "docs/milestones/M0_repo_cleanup_and_scaffold.md",
        "docs/milestones/M1_data_ingest_and_day1_audit.md",
        "docs/milestones/M2_panel_splits_and_freeze.md",
        "docs/milestones/M3_eda_and_h1_panel_har_rv.md",
        "docs/milestones/M4_h2_forecasting_calibration_and_shap.md",
        "docs/milestones/M5_figures_table1_and_cs482_report.md",
    ):
        p = project_root / rel
        assert p.is_file() and p.stat().st_size > 200, f"Missing required file: {rel}"

    gi = (project_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".cache/", "*.aux", "*.log", "*.bbl"):
        assert pattern in gi, f"Missing ignore pattern {pattern!r} in plan_6 .gitignore"


def test_t1_02_m1_raw_provenance_universe_ohlcv_macro_edgar(project_root: Path) -> None:
    """F4-F7: Verify M1 raw SHA-256 provenance, 60-stock universe, OHLCV, macro, and SEC EDGAR."""
    prov_path = project_root / "data" / "raw" / "data_provenance.json"
    prov = ProvenanceLog.load(prov_path, project_root=project_root)
    prov.verify_files()

    uni_df = pd.read_csv(project_root / "data" / "raw" / "universe_60.csv")
    assert len(uni_df) == 60
    assert set(uni_df["ticker"]) == set(config.UNIVERSE_TICKERS)
    assert uni_df["sector"].value_counts().to_dict() == {s: 12 for s in config.SECTORS}
    assert (uni_df["n_trading_days"] == 1760).all()

    ohlcv = pd.read_parquet(project_root / "data" / "raw" / "prices" / "ohlcv_daily.parquet")
    assert len(ohlcv) == 60 * 1760
    assert ohlcv[["open", "high", "low", "close", "adj_close", "volume"]].isna().sum().sum() == 0
    assert (ohlcv["volume"] > 0).all()
    assert (ohlcv["high"] >= ohlcv["low"]).all() and (ohlcv["low"] > 0).all()

    macro = pd.read_parquet(project_root / "data" / "interim" / "macro_daily.parquet")
    assert len(macro) == 1760
    assert macro[["spy_close", "vix_close", "vix_fred", "t10y2y"]].isna().sum().sum() == 0
    assert int(macro["t10y2y_ffilled_flag"].sum()) == 13

    day1 = json.loads(
        (project_root / "outputs" / "metrics" / "day1_data_audit.json").read_text(encoding="utf-8")
    )
    assert day1["sec_edgar"]["total_unique_filings"] == 7703
    assert day1["sec_edgar"]["overlapping_archives_fetched"] == 195
    assert day1["sec_edgar"]["active_tickers_2019_2023"] == 60
    assert day1["sec_edgar"]["active_tickers_2024_2025"] == 60


def test_t1_03_m2_frozen_bundle_manifest_and_schemas(
    project_root: Path,
    panel_df: pd.DataFrame,
    events_phase_b_df: pd.DataFrame,
    analysis_spec: dict[str, Any],
    split_metadata: dict[str, Any],
    feature_manifest: dict[str, Any],
) -> None:
    """F8-F10: Verify M2 SHA-256 freeze manifest and exact schemas for Phase A -> Phase B."""
    verified = verify_freeze_manifest(
        frozen_dir=project_root / "data" / "frozen", project_root=project_root
    )
    assert verified["status"] == "verified"
    assert verified["artifact_count"] == len(FREEZE_ARTIFACT_NAMES)

    assert len(panel_df) == 105_600
    req_panel_cols = {
        "ticker",
        "sector",
        "sub_group",
        "date",
        "split",
        "has_full_features",
        "has_full_target",
        "rv_5d_fwd",
        "y_log_rv",
        "rv_5d_cc_fwd",
        "cum_ret_5d_fwd",
        "y_tail",
        "y_tail_rv90",
        "rv_d",
        "rv_w",
        "rv_m",
        "log_rv_d",
        "log_rv_w",
        "log_rv_m",
        "ret_1d",
        "neg_ret_1d",
        "ret_5d",
        "vol_ratio_20d",
        "daily_range_pct",
        "vix_close",
        "t10y2y",
        "spy_ret_1d",
        "has_news",
        "news_count",
        "has_8k_earnings",
        "news_source",
    }
    assert req_panel_cols.issubset(set(panel_df.columns))

    assert len(events_phase_b_df) == 105_600
    assert int((events_phase_b_df["has_news"] == 1).sum()) == 6822
    assert analysis_spec["schema_version"]
    assert split_metadata["total_canonical_trading_days"] == 1760
    assert feature_manifest["train_preprocessing"]["fitted_split"] == "train"
    assert feature_manifest["train_preprocessing"]["fitted_date_max"] == config.TRAIN_END


def test_t1_04_m3_eda_and_h1_panel_har_rv_artifacts(project_root: Path) -> None:
    """F11-F12: Verify all M3 EDA & H1 Stock-FE Panel HAR-RV outputs and confirmatory family."""
    for rel in (
        "outputs/tables/eda_regimes_by_sector.csv",
        "outputs/tables/eda_leverage_asymmetry.csv",
        "outputs/tables/eda_coverage_consistency.csv",
        "outputs/tables/eda_vix_news_tail_incidence.csv",
        "outputs/tables/eda_summary.md",
        "outputs/metrics/eda_summary.json",
        "outputs/tables/h1_panel_har_rv_main.csv",
        "outputs/tables/h1_panel_har_rv_sensitivities.csv",
        "outputs/tables/h1_summary.md",
        "outputs/metrics/h1_regression.json",
    ):
        p = project_root / rel
        assert p.is_file() and p.stat().st_size > 150, f"Missing M3 output: {rel}"

    vix_news = pd.read_csv(project_root / "outputs" / "tables" / "eda_vix_news_tail_incidence.csv")
    assert {"n_obs", "y_tail_pos", "y_tail_rate", "ci95_low", "ci95_high"}.issubset(
        set(vix_news.columns)
    )

    h1 = json.loads(
        (project_root / "outputs" / "metrics" / "h1_regression.json").read_text(encoding="utf-8")
    )
    assert h1["main_model"]["n_obs"] == 104_040
    assert h1["main_model"]["bartlett_bandwidth_L"] == 7
    assert len(h1["confirmatory_family_holm"]) == 5
    assert len(h1["sensitivities_summary"]) >= 5


def test_t1_05_m4_h2_backtest_calibration_dm_and_shap_artifacts(
    project_root: Path,
    test_pred_df: pd.DataFrame,
) -> None:
    """F13-F16: Verify all 14 M4 forecasting, calibration, HLN-DM, and TreeSHAP artifacts."""
    for rel in (
        "outputs/predictions/h2_val_predictions.parquet",
        "outputs/predictions/h2_test_predictions.parquet",
        "outputs/tables/h2_garch_convergence.csv",
        "outputs/tables/h2_test_metrics.csv",
        "outputs/tables/h2_test_by_sector.csv",
        "outputs/tables/h2_test_by_event_day.csv",
        "outputs/tables/h2_dm_tests.csv",
        "outputs/tables/h2_calibration_curve.csv",
        "outputs/tables/shap_importance.csv",
        "outputs/tables/shap_regime_summary.csv",
        "outputs/metrics/h2_baselines.json",
        "outputs/metrics/h2_lgbm_selection.json",
        "outputs/metrics/h2_evaluation.json",
        "outputs/metrics/shap_audit.json",
    ):
        p = project_root / rel
        assert p.is_file() and p.stat().st_size > 100, f"Missing M4 output: {rel}"

    val_pred = pd.read_parquet(
        project_root / "outputs" / "predictions" / "h2_val_predictions.parquet"
    )
    assert len(val_pred) == 14_760
    assert len(test_pred_df) == 14_700

    base_m = json.loads(
        (project_root / "outputs" / "metrics" / "h2_baselines.json").read_text(encoding="utf-8")
    )
    assert base_m["garch_converged_fits"] == 300
    assert base_m["total_garch_fits"] == 300

    sel_m = json.loads(
        (project_root / "outputs" / "metrics" / "h2_lgbm_selection.json").read_text("utf-8")
    )
    assert sel_m["selected_calibrator"]["method"] in {
        "isotonic",
        "platt_sigmoid",
        "temperature_scaling",
    }

    shap_m = json.loads(
        (project_root / "outputs" / "metrics" / "shap_audit.json").read_text("utf-8")
    )
    assert shap_m["background_split"] == "train"
    assert shap_m["background_n"] == 1000
    assert shap_m["classifier_classes"] == [0, 1]


def test_t1_06_figures_tables_and_unified_cs482_report_artifacts(project_root: Path) -> None:
    """F17-F20: Verify 300-DPI Figure 1 & 2, Table 1 & 2, BibTeX, LaTeX, PDF, and 61-claim audit."""
    for rel in (
        "outputs/figures/fig1_volatility_regimes_and_forecasts.png",
        "outputs/figures/fig1_volatility_regimes_and_forecasts.pdf",
        "outputs/figures/fig2_shap_and_calibration.png",
        "outputs/figures/fig2_shap_and_calibration.pdf",
        "outputs/tables/table1_summary.csv",
        "outputs/tables/table1_summary.md",
        "outputs/tables/table1_summary.tex",
        "outputs/tables/table2_h3_ablation.csv",
        "outputs/tables/table2_h3_ablation.md",
        "outputs/tables/table2_h3_ablation.tex",
        "outputs/metrics/h3_dl_summary.json",
        "report/references.bib",
        "report/report.tex",
        f"report/{config.REPORT_PDF_NAME}",
        "report/report_audit.md",
    ):
        p = project_root / rel
        assert p.is_file() and p.stat().st_size > 500, f"Missing report/table artifact: {rel}"

    audit_md = (project_root / "report" / "report_audit.md").read_text(encoding="utf-8")
    assert "Overall Audit Verdict: **PASS**" in audit_md
    assert "Numeric Claims Verified: `61/61`" in audit_md
    assert "Forbidden Wording / Artifact Hits: `0`" in audit_md
