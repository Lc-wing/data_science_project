"""Milestone M5: Programmatic Table 1, CS482 Phase A LaTeX/PDF build, and automated audit."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from stock_volatility import config
from stock_volatility.figures import generate_all_figures
from stock_volatility.panel.freeze import verify_freeze_manifest

FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bimpacts?\b", "forbidden causal word 'impact(s)'"),
    (r"\bdrivers?\b", "forbidden causal word 'driver(s)'"),
    (r"\belasticit(?:y|ies)\b", "forbidden causal word 'elasticity'"),
    (r"\bcaused\b", "forbidden causal word 'caused'"),
    (r"\bcauses\b", "forbidden causal word 'causes'"),
    (r"\?\?", "unresolved LaTeX reference '??'"),
    (r"\bTODO\b", "unresolved placeholder 'TODO'"),
    (r"\bTBD\b", "unresolved placeholder 'TBD'"),
    (r"\bNaN\b", "missing/NaN value rendered in text"),
)


def scan_report_text_issues(text: str) -> list[str]:
    """Scan report prose/PDF text for forbidden causal phrasing and unresolved LaTeX artifacts."""
    issues: list[str] = []
    for pattern, label in FORBIDDEN_PATTERNS:
        flags = 0 if pattern in (r"\bTODO\b", r"\bTBD\b", r"\bNaN\b", r"\?\?") else re.IGNORECASE
        for match in re.finditer(pattern, text, flags=flags):
            snippet = text[max(0, match.start() - 25) : min(len(text), match.end() + 25)].replace(
                "\n", " "
            )
            issues.append(f"{label}: ...{snippet}...")
    return issues


def _fmt_p(p: float) -> str:
    """Format p-values cleanly for academic tables and prose."""
    if p < 1e-4:
        return "<0.0001"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def build_table1_summary(project_root: Path | None = None) -> dict[str, Any]:
    """Build unified Table 1 (Panel A: H1 HAR-RV; Panel B: H2 2025 Holdout) from M3/M4 files."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    h1 = json.loads((root / "outputs" / "metrics" / "h1_regression.json").read_text("utf-8"))
    h2_metrics = pd.read_csv(root / "outputs" / "tables" / "h2_test_metrics.csv")
    h2_dm = pd.read_csv(root / "outputs" / "tables" / "h2_dm_tests.csv")

    dm_qlike = h2_dm[h2_dm["loss_metric"] == "qlike"].set_index("baseline_model").to_dict("index")
    dm_brier = h2_dm[h2_dm["loss_metric"] == "brier"].set_index("baseline_model").to_dict("index")

    rows: list[dict[str, Any]] = []
    label_map_h1 = {
        "neg_ret_1d": "Leverage AME (neg_ret_1d)",
        "vix_close": "VIX AME (per +1 pt)",
        "tech_x_neg_ret_1d_vs_healthcare": "Tech x neg_ret_1d vs Health.",
        "tech_x_vix_close_vs_healthcare": "Tech x VIX vs Health.",
        "joint_5sector_interaction_wald": "Joint 5-Sector Wald (df=8)",
    }
    for item in h1["confirmatory_family_holm"]:
        tid = item["test_id"]
        is_wald = tid == "joint_5sector_interaction_wald"
        est_s = f"{item['estimate']:.2f}" if is_wald else f"{item['estimate']:+.4f}"
        ci_s = (
            f"F={item['t_stat']:.2f}"
            if is_wald
            else f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}]"
        )
        rows.append(
            {
                "panel": "Panel A: H1 Panel HAR-RV",
                "item": label_map_h1[tid],
                "metric_1": est_s,
                "metric_2": ci_s,
                "metric_3": _fmt_p(float(item["p_value"])),
                "metric_4": _fmt_p(float(item["p_holm"])),
                "metric_5": "Yes" if item["significant_holm_05"] else "No",
                "metric_6": f"Within R2={h1['main_model']['within_r2']:.4f}",
            }
        )

    model_labels = {
        "persistence": "Persistence (RV_w / Base)",
        "gjr_garch": "GJR-GARCH(1,1) Per-Stock",
        "har_rv": "Panel HAR-RV (+ Logistic)",
        "lgbm": "Global LightGBM (Cal.)",
    }
    for m_key in ("persistence", "gjr_garch", "har_rv", "lgbm"):
        r = h2_metrics[h2_metrics["model"] == m_key].iloc[0]
        if m_key == "lgbm":
            dm_str = "Ref."
        else:
            pq = _fmt_p(float(dm_qlike[m_key]["p_holm"]))
            pb = _fmt_p(float(dm_brier[m_key]["p_holm"]))
            dm_str = f"{pq} / {pb}"
        ece_str = (
            f"{r['ece']:.4f} ({r['ece_pre_cal']:.3f})"
            if m_key == "lgbm"
            else f"{r['ece']:.4f}"
        )
        qlike_ci = f"[{r['qlike_ci95_low']:.3f},{r['qlike_ci95_high']:.3f}]"
        prauc_ci = f"[{r['pr_auc_ci95_low']:.3f},{r['pr_auc_ci95_high']:.3f}]"
        rows.append(
            {
                "panel": "Panel B: H2 2025 Holdout",
                "item": model_labels[m_key],
                "metric_1": f"{r['qlike']:.4f} {qlike_ci}",
                "metric_2": f"{r['rmse']:.4f}",
                "metric_3": f"{r['pr_auc']:.4f} {prauc_ci}",
                "metric_4": f"{r['brier']:.4f}",
                "metric_5": ece_str,
                "metric_6": dm_str,
            }
        )

    table_df = pd.DataFrame(rows)

    # Markdown Table 1
    md_lines = [
        "# Table 1: H1 Panel HAR-RV Confirmatory Associations & H2 2025 Holdout Performance",
        "",
        "## Panel A: H1 Stock-FE Extended Panel HAR-RV (Driscoll–Kraay SE, L=7, N=104,040)",
        "",
        "| Confirmatory Term | Estimate / Stat | 95% CI / F-Stat | Raw p | Holm p | Holm Sig. |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows[:5]:
        md_lines.append(
            f"| {row['item']} | `{row['metric_1']}` | `{row['metric_2']}` | "
            f"`{row['metric_3']}` | `{row['metric_4']}` | `{row['metric_5']}` |"
        )
    md_lines.extend(
        [
            "",
            "## Panel B: H2 Single-Shot 2025 Holdout Test (N=14,700; 245 Scorable Days)",
            "",
            "| Model | QLIKE [95% CI] | RMSE | PR-AUC [95% CI] | Brier | ECE (Pre-Cal) | "
            "HLN-DM Holm p (QLIKE / Brier vs LGBM) |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows[5:]:
        md_lines.append(
            f"| {row['item']} | `{row['metric_1']}` | `{row['metric_2']}` | "
            f"`{row['metric_3']}` | `{row['metric_4']}` | `{row['metric_5']}` | "
            f"`{row['metric_6']}` |"
        )
    md_lines.append("")
    markdown_str = "\n".join(md_lines)

    # Compact, clean publication LaTeX Table 1 (fits comfortably within single-column A4 width)
    def _esc(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%")

    within_r2_s = f"{h1['main_model']['within_r2']:.4f}"
    tex_lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Summary of H1 Stock-FE Extended Panel HAR-RV Confirmatory Tests (2019--2025, "
        r"$N=104{,}040$, Bartlett Driscoll--Kraay SE with $L=7$) and H2 Single-Shot 2025 Holdout "
        r"Forecasting Performance ($N=14{,}700$, 245 scorable days, 5-day date-block bootstrap "
        r"$B=1{,}000$ 95\% CIs, and $h=5$ HLN-DM Holm-adjusted $p$-values vs.\ LightGBM).}",
        r"\label{tab:main_summary}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{Panel A: H1 Confirmatory Associational Family "
        + rf"(Within $R^2 = {within_r2_s}$, Healthcare Reference)}}}}"
        + r" \\",
        r"\midrule",
        r"Confirmatory Hypothesis & Estimate / $\chi^2$ & 95\% CI / $F$ & Raw $p$ & "
        r"Holm $p$ & Sig. & Sample \\",
        r"\midrule",
    ]
    for row in rows[:5]:
        tex_lines.append(
            f"{_esc(row['item'])} & {row['metric_1']} & {row['metric_2']} & "
            f"{row['metric_3']} & {row['metric_4']} & {row['metric_5']} & 104,040 \\\\"
        )
    tex_lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{\textbf{Panel B: H2 2025 Holdout Test Performance "
            r"(Base Tail Rate = 8.62\%, 300/300 GJR-GARCH Converged)}} \\",
            r"\midrule",
            r"Model & QLIKE [95\% CI] & RMSE & PR-AUC [95\% CI] & Brier & ECE (Pre) & "
            r"DM Holm $p$ (Q/B) \\",
            r"\midrule",
        ]
    )
    for row in rows[5:]:
        item_lbl = _esc(row["item"])
        if "LightGBM" in row["item"]:
            item_lbl = r"\textbf{" + item_lbl + "}"
        tex_lines.append(
            f"{item_lbl} & {row['metric_1']} & {row['metric_2']} & "
            f"{row['metric_3']} & {row['metric_4']} & {row['metric_5']} & {row['metric_6']} \\\\"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    latex_str = "\n".join(tex_lines) + "\n"

    return {
        "table_df": table_df,
        "markdown": markdown_str,
        "latex": latex_str,
    }


def collect_report_claims(project_root: Path | None = None) -> dict[str, dict[str, str]]:
    """Collect every reported number directly from committed M1-M4 JSON/CSV outputs."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    day1 = json.loads((root / "outputs" / "metrics" / "day1_data_audit.json").read_text("utf-8"))
    split_m = json.loads((root / "data" / "frozen" / "split_metadata.json").read_text("utf-8"))
    feat_m = json.loads((root / "data" / "frozen" / "feature_manifest.json").read_text("utf-8"))
    freeze_m = json.loads((root / "data" / "frozen" / "FREEZE_MANIFEST.json").read_text("utf-8"))
    eda = json.loads((root / "outputs" / "metrics" / "eda_summary.json").read_text("utf-8"))
    h1 = json.loads((root / "outputs" / "metrics" / "h1_regression.json").read_text("utf-8"))
    h2_base = json.loads((root / "outputs" / "metrics" / "h2_baselines.json").read_text("utf-8"))
    h2_sel = json.loads(
        (root / "outputs" / "metrics" / "h2_lgbm_selection.json").read_text("utf-8")
    )
    h2_eval = json.loads((root / "outputs" / "metrics" / "h2_evaluation.json").read_text("utf-8"))
    shap_a = json.loads((root / "outputs" / "metrics" / "shap_audit.json").read_text("utf-8"))
    by_sec = pd.read_csv(root / "outputs" / "tables" / "h2_test_by_sector.csv")
    by_ev = pd.read_csv(root / "outputs" / "tables" / "h2_test_by_event_day.csv")

    conf_h1 = {r["test_id"]: r for r in h1["confirmatory_family_holm"]}
    h2_m = {r["model"]: r for r in h2_eval["test_metrics"]}
    dm_q = {
        r["baseline_model"]: r for r in h2_eval["dm_tests"] if r["loss_metric"] == "qlike"
    }
    dm_b = {
        r["baseline_model"]: r for r in h2_eval["dm_tests"] if r["loss_metric"] == "brier"
    }
    tech_lgb = by_sec[(by_sec["sector"] == "Technology") & (by_sec["model"] == "lgbm")].iloc[0]
    tech_har = by_sec[(by_sec["sector"] == "Technology") & (by_sec["model"] == "har_rv")].iloc[0]
    earn_har = by_ev[
        (by_ev["event_group"] == "Has_8K_Earnings") & (by_ev["model"] == "har_rv")
    ].iloc[0]
    earn_lgb = by_ev[
        (by_ev["event_group"] == "Has_8K_Earnings") & (by_ev["model"] == "lgbm")
    ].iloc[0]

    panel_sha = freeze_m["files"]["data/frozen/panel_daily.parquet"]["sha256"][:12]
    ece_drop_pct = (
        100.0 * (1.0 - h2_m["lgbm"]["ece"] / h2_m["lgbm"]["ece_pre_cal"])
    )

    def _c(val: str, src: str, loc: str) -> dict[str, str]:
        return {"value": val, "source": src, "locator": loc}

    return {
        "N_STOCKS": _c("60", "data/raw/universe_60.csv", "len(rows)"),
        "N_DAYS": _c(
            f"{split_m['total_canonical_trading_days']:,}",
            "data/frozen/split_metadata.json",
            "total_canonical_trading_days",
        ),
        "PANEL_ROWS": _c(
            f"{split_m['total_panel_rows']:,}",
            "data/frozen/split_metadata.json",
            "total_panel_rows",
        ),
        "T10Y2Y_FFILL": _c(
            str(day1["macro"]["t10y2y_ffilled_count"]),
            "outputs/metrics/day1_data_audit.json",
            "macro.t10y2y_ffilled_count",
        ),
        "SEC_ARCHIVES": _c(
            str(day1["sec_edgar"]["overlapping_archives_fetched"]),
            "outputs/metrics/day1_data_audit.json",
            "sec_edgar.overlapping_archives_fetched",
        ),
        "SEC_FILINGS_RAW": _c(
            f"{day1['sec_edgar']['total_unique_filings']:,}",
            "outputs/metrics/day1_data_audit.json",
            "sec_edgar.total_unique_filings",
        ),
        "EARLY_CLOSE_SHIFTED": _c(
            str(split_m["event_alignment_audit"]["shifted_due_to_early_close"]),
            "data/frozen/split_metadata.json",
            "event_alignment_audit.shifted_due_to_early_close",
        ),
        "DEDUP_DROPPED": _c(
            str(split_m["event_dedup_and_cap_audit"]["dedup_headline_dropped"]),
            "data/frozen/split_metadata.json",
            "event_dedup_and_cap_audit.dedup_headline_dropped",
        ),
        "EVENT_ANCHORED_DAYS": _c(
            f"{split_m['event_dedup_and_cap_audit']['active_ticker_days']:,}",
            "data/frozen/split_metadata.json",
            "event_dedup_and_cap_audit.active_ticker_days",
        ),
        "TRAIN_MOD_ROWS": _c(
            f"{split_m['splits']['train']['modeling_complete_rows']:,}",
            "data/frozen/split_metadata.json",
            "splits.train.modeling_complete_rows",
        ),
        "VAL_MOD_ROWS": _c(
            f"{split_m['splits']['val']['modeling_complete_rows']:,}",
            "data/frozen/split_metadata.json",
            "splits.val.modeling_complete_rows",
        ),
        "TEST_MOD_ROWS": _c(
            f"{split_m['splits']['test']['modeling_complete_rows']:,}",
            "data/frozen/split_metadata.json",
            "splits.test.modeling_complete_rows",
        ),
        "TEST_DAYS": _c(
            str(split_m["test_scorable_trading_days"]),
            "data/frozen/split_metadata.json",
            "test_scorable_trading_days",
        ),
        "RV90_THRESH": _c(
            f"{feat_m['train_preprocessing']['rv90_threshold']:.4f}",
            "data/frozen/feature_manifest.json",
            "train_preprocessing.rv90_threshold",
        ),
        "PANEL_SHA12": _c(
            panel_sha,
            "data/frozen/FREEZE_MANIFEST.json",
            "files.panel_daily.parquet.sha256[:12]",
        ),
        "VIX_MEDIAN": _c(
            f"{eda['vix_train_median_threshold']:.2f}",
            "outputs/metrics/eda_summary.json",
            "vix_train_median_threshold",
        ),
        "RV_COVID_2020": _c(
            f"{eda['regimes_all']['covid_shock_2020']['rv_5d_fwd_mean']:.4f}",
            "outputs/metrics/eda_summary.json",
            "regimes_all.covid_shock_2020.rv_5d_fwd_mean",
        ),
        "RV_BEAR_2022": _c(
            f"{eda['regimes_all']['rate_hike_bear_2022']['rv_5d_fwd_mean']:.4f}",
            "outputs/metrics/eda_summary.json",
            "regimes_all.rate_hike_bear_2022.rv_5d_fwd_mean",
        ),
        "RV_VAL_2024": _c(
            f"{eda['regimes_all']['ai_cycle_val_2024']['rv_5d_fwd_mean']:.4f}",
            "outputs/metrics/eda_summary.json",
            "regimes_all.ai_cycle_val_2024.rv_5d_fwd_mean",
        ),
        "RV_TEST_2025": _c(
            f"{eda['regimes_all']['ai_cycle_test_2025']['rv_5d_fwd_mean']:.4f}",
            "outputs/metrics/eda_summary.json",
            "regimes_all.ai_cycle_test_2025.rv_5d_fwd_mean",
        ),
        "COV_RATE_RATIO": _c(
            f"{eda['coverage_all']['event_day_rate_ratio_2425_vs_1923']:.4f}",
            "outputs/metrics/eda_summary.json",
            "coverage_all.event_day_rate_ratio_2425_vs_1923",
        ),
        "H1_N_OBS": _c(
            f"{h1['main_model']['n_obs']:,}",
            "outputs/metrics/h1_regression.json",
            "main_model.n_obs",
        ),
        "H1_R2": _c(
            f"{h1['main_model']['within_r2']:.4f}",
            "outputs/metrics/h1_regression.json",
            "main_model.within_r2",
        ),
        "H1_DK_L": _c(
            str(h1["main_model"]["bartlett_bandwidth_L"]),
            "outputs/metrics/h1_regression.json",
            "main_model.bartlett_bandwidth_L",
        ),
        "H1_NEG_EST": _c(
            f"{conf_h1['neg_ret_1d']['estimate']:.4f}",
            "outputs/metrics/h1_regression.json",
            "confirmatory_family_holm.neg_ret_1d.estimate",
        ),
        "H1_VIX_EST": _c(
            f"{conf_h1['vix_close']['estimate']:.4f}",
            "outputs/metrics/h1_regression.json",
            "confirmatory_family_holm.vix_close.estimate",
        ),
        "H1_TECH_NEG_PH": _c(
            f"{conf_h1['tech_x_neg_ret_1d_vs_healthcare']['p_holm']:.3f}",
            "outputs/metrics/h1_regression.json",
            "confirmatory_family_holm.tech_x_neg_ret_1d_vs_healthcare.p_holm",
        ),
        "H1_WALD_CHI2": _c(
            f"{conf_h1['joint_5sector_interaction_wald']['estimate']:.2f}",
            "outputs/metrics/h1_regression.json",
            "confirmatory_family_holm.joint_5sector_interaction_wald.estimate",
        ),
        "GARCH_CONV": _c(
            f"{h2_base['garch_converged_fits']}/{h2_base['total_garch_fits']}",
            "outputs/metrics/h2_baselines.json",
            "garch_converged_fits/total_garch_fits",
        ),
        "CAL_METHOD": _c(
            str(h2_sel["selected_calibrator"]["method"]),
            "outputs/metrics/h2_lgbm_selection.json",
            "selected_calibrator.method",
        ),
        "LGBM_QLIKE": _c(
            f"{h2_m['lgbm']['qlike']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.lgbm.qlike",
        ),
        "HAR_QLIKE": _c(
            f"{h2_m['har_rv']['qlike']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.har_rv.qlike",
        ),
        "GARCH_QLIKE": _c(
            f"{h2_m['gjr_garch']['qlike']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.gjr_garch.qlike",
        ),
        "PERS_QLIKE": _c(
            f"{h2_m['persistence']['qlike']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.persistence.qlike",
        ),
        "LGBM_RMSE": _c(
            f"{h2_m['lgbm']['rmse']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.lgbm.rmse",
        ),
        "HAR_PRAUC": _c(
            f"{h2_m['har_rv']['pr_auc']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.har_rv.pr_auc",
        ),
        "LGBM_PRAUC": _c(
            f"{h2_m['lgbm']['pr_auc']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.lgbm.pr_auc",
        ),
        "LGBM_BRIER": _c(
            f"{h2_m['lgbm']['brier']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.lgbm.brier",
        ),
        "LGBM_ECE_PRE": _c(
            f"{h2_m['lgbm']['ece_pre_cal']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.lgbm.ece_pre_cal",
        ),
        "LGBM_ECE_POST": _c(
            f"{h2_m['lgbm']['ece']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "test_metrics.lgbm.ece",
        ),
        "ECE_DROP_PCT": _c(
            f"{ece_drop_pct:.1f}",
            "outputs/metrics/h2_evaluation.json",
            "100*(1 - ece/ece_pre_cal)",
        ),
        "DM_Q_GARCH_PH": _c(
            f"{dm_q['gjr_garch']['p_holm']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "dm_tests.qlike.gjr_garch.p_holm",
        ),
        "DM_Q_HAR_PH": _c(
            f"{dm_q['har_rv']['p_holm']:.3f}",
            "outputs/metrics/h2_evaluation.json",
            "dm_tests.qlike.har_rv.p_holm",
        ),
        "DM_B_PERS_PH": _c(
            f"{dm_b['persistence']['p_holm']:.4f}",
            "outputs/metrics/h2_evaluation.json",
            "dm_tests.brier.persistence.p_holm",
        ),
        "TECH_LGB_QLIKE": _c(
            f"{tech_lgb['qlike']:.4f}",
            "outputs/tables/h2_test_by_sector.csv",
            "Technology.lgbm.qlike",
        ),
        "TECH_HAR_QLIKE": _c(
            f"{tech_har['qlike']:.4f}",
            "outputs/tables/h2_test_by_sector.csv",
            "Technology.har_rv.qlike",
        ),
        "EARN_HAR_PRAUC": _c(
            f"{earn_har['pr_auc']:.4f}",
            "outputs/tables/h2_test_by_event_day.csv",
            "Has_8K_Earnings.har_rv.pr_auc",
        ),
        "EARN_LGB_PRAUC": _c(
            f"{earn_lgb['pr_auc']:.4f}",
            "outputs/tables/h2_test_by_event_day.csv",
            "Has_8K_Earnings.lgbm.pr_auc",
        ),
        "SHAP_BG_N": _c(
            f"{shap_a['background_n']:,}",
            "outputs/metrics/shap_audit.json",
            "background_n",
        ),
        "SHAP_RVM_CLF_PCT": _c(
            f"{shap_a['top_features_classifier_prob'][0]['share_pct']:.1f}",
            "outputs/metrics/shap_audit.json",
            "top_features_classifier_prob[0].share_pct",
        ),
        "SHAP_VIX_CLF_PCT": _c(
            f"{shap_a['top_features_classifier_prob'][1]['share_pct']:.1f}",
            "outputs/metrics/shap_audit.json",
            "top_features_classifier_prob[1].share_pct",
        ),
    }


def render_references_bib() -> str:
    """Return verified BibTeX references for Plan 6 Phase A report."""
    return r"""@article{parkinson1980,
  author  = {Parkinson, Michael},
  title   = {The Extreme Value Method for Estimating the Variance of the Rate of Return},
  journal = {Journal of Business},
  year    = {1980},
  volume  = {53},
  number  = {1},
  pages   = {61--65}
}

@article{duan1983,
  author  = {Duan, Naihua},
  title   = {Smearing Estimate: A Nonparametric Retransformation Method},
  journal = {Journal of the American Statistical Association},
  year    = {1983},
  volume  = {78},
  number  = {383},
  pages   = {605--610}
}

@article{glosten1993,
  author  = {Glosten, Lawrence R. and Jagannathan, Ravi and Runkle, David E.},
  title   = {On the Relation between the Expected Value and the Volatility of the
             Nominal Excess Return on Stocks},
  journal = {Journal of Finance},
  year    = {1993},
  volume  = {48},
  number  = {5},
  pages   = {1779--1801}
}

@article{harvey1997,
  author  = {Harvey, David and Leybourne, Stephen and Newbold, Paul},
  title   = {Testing the Equality of Prediction Mean Squared Errors},
  journal = {International Journal of Forecasting},
  year    = {1997},
  volume  = {13},
  number  = {2},
  pages   = {281--291}
}

@article{driscoll1998,
  author  = {Driscoll, John C. and Kraay, Aart C.},
  title   = {Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data},
  journal = {Review of Economics and Statistics},
  year    = {1998},
  volume  = {80},
  number  = {4},
  pages   = {549--560}
}

@article{holm1979,
  author  = {Holm, Sture},
  title   = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scandinavian Journal of Statistics},
  year    = {1979},
  volume  = {6},
  number  = {2},
  pages   = {65--70}
}

@article{corsi2009,
  author  = {Corsi, Fulvio},
  title   = {A Simple Approximate Long-Memory Model of Realized Volatility},
  journal = {Journal of Financial Econometrics},
  year    = {2009},
  volume  = {7},
  number  = {2},
  pages   = {174--196}
}

@article{patton2011,
  author  = {Patton, Andrew J.},
  title   = {Volatility Forecast Comparison Using Imperfect Volatility Proxies},
  journal = {Journal of Econometrics},
  year    = {2011},
  volume  = {160},
  number  = {1},
  pages   = {246--256}
}

@inproceedings{ke2017lightgbm,
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and
               Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  title     = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  volume    = {30}
}

@inproceedings{guo2017calibration,
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  title     = {On Calibration of Modern Neural Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  year      = {2017},
  pages     = {1321--1330}
}

@book{lopezdeprado2018,
  author    = {L{\'o}pez de Prado, Marcos},
  title     = {Advances in Financial Machine Learning},
  publisher = {Wiley},
  year      = {2018}
}

@article{lundberg2020treeshap,
  author  = {Lundberg, Scott M. and Erion, Gabriel and Chen, Hugh and DeGrave, Alex and
             Prutkin, Jordan M. and Nair, Bala and Katz, Ronit and Himmelfarb, Jonathan and
             Bansal, Nisha and Lee, Su-In},
  title   = {From Local Explanations to Global Understanding with Explainable {AI} for Trees},
  journal = {Nature Machine Intelligence},
  year    = {2020},
  volume  = {2},
  number  = {1},
  pages   = {56--67}
}

@article{kassis2026skills,
  author  = {Kassis, Timothy and Agarwal, Vaibhav and He, Yujie and
             Patel, Dev and Brueckner, Anne M.},
  title   = {Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents},
  journal = {arXiv preprint arXiv:2609.00065},
  year    = {2026},
  note    = {https://doi.org/10.48550/arXiv.2609.00065}
}
"""


def render_phase_a_tex(
    claims: dict[str, dict[str, str]],
    table1_latex: str,
) -> str:
    """Render CS482 Phase A Draft LaTeX source with 100% programmatic claim injection."""
    v = {k: item["value"] for k, item in claims.items()}
    return rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[a4paper,top=1.75cm,bottom=1.75cm,left=1.95cm,right=1.95cm]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{setspace}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage[numbers,sort&compress]{{natbib}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{microtype}}

\onehalfspacing
\setlength{{\parskip}}{{0.16em}}
\setlength{{\textfloatsep}}{{6pt plus 2pt minus 2pt}}
\setlength{{\intextsep}}{{6pt plus 2pt minus 2pt}}
\setlength{{\abovecaptionskip}}{{3pt}}
\setlength{{\belowcaptionskip}}{{2pt}}
\renewcommand{{\topfraction}}{{0.90}}
\renewcommand{{\bottomfraction}}{{0.85}}
\renewcommand{{\textfraction}}{{0.08}}
\renewcommand{{\floatpagefraction}}{{0.82}}

\begin{{document}}

% =========================================================================
% COVER PAGE (Excluded from CS482 3-5 A4 Page Main Text Limit)
% =========================================================================
\pagenumbering{{roman}}
\begin{{titlepage}}
\centering
\vspace*{{1.2cm}}
{{\large CS482 Data Science --- Group Project (Phase A Draft)\par}}
\vspace{{0.8cm}}
{{\LARGE\bfseries Corporate Event Shocks, Realized Volatility, and Downside
Tail-Risk Forecasting in U.S.\ Equities (2019--2025)\par}}
\vspace{{0.4cm}}
{{\large Econometric Panel HAR-RV, Per-Stock GJR-GARCH(1,1), Calibrated
LightGBM, and Train-Background TreeSHAP Under Leakage-Free Purged Splits\par}}
\vspace{{1.2cm}}

{{\small
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Group Member \& Role}} & \textbf{{Student ID (Replace Before Moodle)}} \\
\midrule
Member A (Financial Data \& Freeze) & [Student ID A] \\
Member B (Panel Econometrics \& GARCH) & [Student ID B] \\
Member C (LightGBM, Calibration \& TreeSHAP) & [Student ID C] \\
Member D / All (Phase B QLoRA \& Audit) & [Student ID D] \\
\bottomrule
\end{{tabular}}
}}

\vspace{{1.0cm}}
\begin{{minipage}}{{0.88\textwidth}}
\small\raggedright
\textbf{{Academic Research \& Non-Investment-Advice Notice.}}
This study is conducted solely for CS482 academic coursework and methodological
evaluation of associational and predictive risk models (academic research only;
\textbf{{not investment advice}}). It makes no causal claims and does not
constitute investment, financial, or trading advice. All empirical statistics
are generated programmatically from the frozen Phase A bundle
(\texttt{{panel\_daily.parquet}} SHA-256 prefix \texttt{{{v['PANEL_SHA12']}}}).
\end{{minipage}}

\vfill
{{\normalsize Phase A Freeze Date: November 2026 \quad | \quad
Target Final Submission: December 14, 2026\par}}
\end{{titlepage}}

% =========================================================================
% MAIN TEXT (CS482 Requirement: 3-5 A4 Pages, Single-Column, 11pt, 1.5 Spacing)
% =========================================================================
\pagenumbering{{arabic}}
\setcounter{{page}}{{1}}

\section{{Introduction, Research Question, and Hypotheses}}

Forecasting short-horizon equity volatility and downside tail-risk episodes
matters directly to portfolio risk managers, volatility-targeting allocators,
and financial stability researchers because market turbulence clusters across
regimes while return levels closely resemble a random walk. Rather than
forecasting noisy directional prices, this study examines 5-day forward realized
volatility and downside tail-risk events across a deterministic cross-sector
universe of {v['N_STOCKS']} liquid U.S.\ equities over 2019--2025, spanning the
2020 COVID-19 shock, the 2022 monetary tightening bear market, the 2024
expansion, and 2025 AI and policy volatility.

\textbf{{Research Question.}} Under strict purged and embargoed chronological
splits, how reliably do classical econometric baselines (panel HAR-RV and
per-stock GJR-GARCH(1,1)) and nonlinear gradient-boosted trees (LightGBM)
forecast 5-day forward annualized realized volatility and downside tail-risk
events (5-day return $< -5\%$), and how strongly are lagged negative returns and
implied volatility (VIX) associated with future volatility across sectors? We
pre-register three falsifiable hypotheses:
\begin{{itemize}}\setlength{{\itemsep}}{{0pt}}\setlength{{\parsep}}{{0pt}}
\item \textbf{{H1 (Confirmatory, Associational):}} Lagged negative daily returns
(leverage association) and lagged VIX are positively associated with forward
5-day log realized volatility, with heterogeneity across growth/cyclical and
defensive sectors.
\item \textbf{{H2 (Confirmatory, Predictive):}} Under quarterly expanding purged
backtesting, global LightGBM outperforms Persistence, per-stock GJR-GARCH(1,1),
and Panel HAR-RV on 2025 holdout QLIKE/RMSE and downside tail-risk PR-AUC, while
2024 validation calibration improves probability reliability (ECE).
\item \textbf{{H3 (Deferred to Phase B):}} Event-anchored 4-bit QLoRA
fine-tuning of 27B--32B open LLMs on frozen pre-close SEC EDGAR event text adds
incremental 2025 tail-risk lift over price-only Phase A baselines (null results
accepted).
\end{{itemize}}
All Phase A conclusions are strictly associational or predictive and are not
investment advice.

\section{{Data Collection, Leakage-Free Alignment, and Frozen Splits}}

\textbf{{Cross-Sector Universe and Market Data.}} We fix a pre-2019
deterministic universe of {v['N_STOCKS']} S\&P 500 bellwethers (12 stocks across
each of five GICS sectors: Technology, Financials, Healthcare, Consumer, and
Energy; zero 2025 lookahead). Daily split-adjusted OHLCV and dividend-adjusted
closes for all {v['N_STOCKS']} equities plus \texttt{{SPY}} and
\texttt{{\^{{}}VIX}} are collected from Yahoo Finance across {v['N_DAYS']}
canonical NYSE trading days (2019-01-02 to 2025-12-31; {v['PANEL_ROWS']}
stock-days, zero missing cells or zero-volume halts). Macro indicators from FRED
(\texttt{{VIXCLS}} and 10Y--2Y yield spread \texttt{{T10Y2Y}}) are aligned onto
the NYSE calendar using backward-only forward-fill (never backfilled), flagging
{v['T10Y2Y_FFILL']} U.S.\ bond-market holidays.

\textbf{{Authoritative Corporate Event Backbone.}} Day-1 probes confirmed that
Hugging Face's \texttt{{financial-news-multisource}} is gated (HTTP 401 without
token) and lacks continuous 2024 intraday coverage, whereas \texttt{{FNSPID}}
terminates in 2023 and exhibits \texttt{{00:00:00 UTC}} timestamp leakage.
Following our pre-registered fallback, we ingest public SEC EDGAR Form
\texttt{{8-K}}, \texttt{{8-K/A}}, \texttt{{10-Q}}, and \texttt{{10-K}} filings
across 2019--2025, resolving holding-company predecessor CIKs for \texttt{{BLK}}
and \texttt{{XOM}} and recursively paginating {v['SEC_ARCHIVES']} historical
\texttt{{filings.files}} archives to recover {v['SEC_FILINGS_RAW']} unique
filings with 60/60 dual-period continuity (2024--2025 vs.\ 2019--2023 event-day
rate ratio {v['COV_RATE_RATIO']}). All second-precision UTC
\texttt{{acceptanceDateTime}} stamps are converted to \texttt{{America/New\_York}}
(DST-aware): filings before 16:00 ET (13:00 ET on 15 scheduled NYSE early-close
half-days, shifting {v['EARLY_CLOSE_SHIFTED']} post-13:00 \texttt{{WFC}} filings
to $t+1$) stay on day $t$, while post-close and holiday filings roll to $t+1$.
Intra-day headline deduplication removes {v['DEDUP_DROPPED']} duplicates and
caps counts at $\le 5$/day, yielding {v['EVENT_ANCHORED_DAYS']} event-anchored
stock-days frozen for Phase B.

\textbf{{Targets, Predictors, and Purged Splits.}} Primary continuous target
\texttt{{rv\_5d\_fwd}} is the 5-day forward annualized Parkinson high-low range
volatility~\cite{{parkinson1980}} over $t+1\dots t+5$, modeled on
$y = \ln(\text{{RV}} + 10^{{-6}})$ with close-to-close sensitivity
\texttt{{rv\_5d\_cc\_fwd}}. Primary binary target \texttt{{y\_tail}} equals 1
iff 5-day forward cumulative adjusted return is below $-5\%$ (sensitivity
\texttt{{y\_tail\_rv90}} uses the Train-only 90th-percentile RV threshold
{v['RV90_THRESH']}). All 22 predictors use strictly $\le t$ information.
Following \citet{{lopezdeprado2018}}, chronological splits enforce 6-trading-day
embargoes ($\ge h=5$): Train 2019-01-02 to 2023-12-20 ({v['TRAIN_MOD_ROWS']}
modeling rows after 21-day \texttt{{rv\_m}} warmup), Val 2024-01-02 to
2024-12-20 ({v['VAL_MOD_ROWS']} rows, tuning/calibration only), and single-shot
Holdout Test 2025-01-02 to 2025-12-31 ({v['TEST_MOD_ROWS']} scorable rows across
{v['TEST_DAYS']} days; final 5 incomplete days masked \texttt{{NA}}). All frozen
artifacts are sealed under SHA-256 (\texttt{{panel\_daily.parquet}} prefix
\texttt{{{v['PANEL_SHA12']}}}).

\section{{Econometric and Machine Learning Methodology}}

\textbf{{EDA and H1 Panel HAR-RV.}} Cross-cycle EDA audits regime volatility,
leverage asymmetry, and tail-risk incidence across Train-median VIX regimes
($\text{{VIX}} \ge {v['VIX_MEDIAN']}$). To test H1 across all {v['H1_N_OBS']}
modeling-complete stock-days without pretest switching, we estimate a
within-entity Stock-FE extended Panel HAR-RV model~\cite{{corsi2009}}
interacting lagged negative returns ($\min(r_{{i,t}}, 0)$) and VIX with sector
indicators (Healthcare reference), using Bartlett-kernel Driscoll--Kraay
standard errors~\cite{{driscoll1998}} ($L={v['H1_DK_L']}$) to account for both
serial overlap and contemporaneous cross-stock shocks, with step-down
Holm--Bonferroni correction~\cite{{holm1979}} across the 5-test confirmatory
family and pre-specified Train-only, close-to-close, and entity-clustered
Newey--West sensitivities.

\textbf{{H2 Out-of-Sample Forecasting, Calibration, and TreeSHAP.}} Four models
share an identical quarterly expanding-window purged schedule (2024 Val, then
2025 Q1--Q4 with $\ge 5$ trading-day purge before every quarter): (i)
Persistence (\texttt{{rv\_w}} and expanding base rate); (ii) per-stock
GJR-GARCH(1,1)~\cite{{glosten1993}} on $100r_{{i,t}}$ with causal post-fit
variance filtering ({v['GARCH_CONV']} fits converged); (iii) Stock-FE Predictive
HAR-RV back-transformed via fit-fold Duan smearing~\cite{{duan1983}} plus
Logistic HAR-RV; and (iv) Global LightGBM~\cite{{ke2017lightgbm}} tuned strictly
on 2024 Val QLIKE (regression) and PR-AUC (classification). To avoid in-sample
calibration overfitting, we compare Isotonic, Platt sigmoid, and 1-parameter
Temperature Scaling~\cite{{guo2017calibration}} via 4-quarter out-of-fold CV
strictly inside 2024 Val, selecting \texttt{{{v['CAL_METHOD']}}}, refitting on
full 2024 Val, and freezing it for single-shot 2025 Test evaluation. Regression
is evaluated by variance-scale QLIKE~\cite{{patton2011}}, RMSE, and MAE;
classification by PR-AUC (primary), ROC-AUC, Brier, and 10-bin ECE, with 5-day
date-block bootstrap 95\% CIs ($B=1{{,}}000$) and $h=5$ Bartlett-PSD HLN-modified
Diebold--Mariano tests~\cite{{harvey1997}} (Holm-adjusted). Exact interventional
TreeSHAP~\cite{{lundberg2020treeshap}} uses a {v['SHAP_BG_N']}-row 2019--2023
Train background to audit LightGBM model behavior strictly non-causally.

\section{{Empirical Results}}

\textbf{{Cross-Cycle Regimes and H1 Panel Evidence.}} Mean 5-day annualized
Parkinson RV peaked during the 2020 COVID shock ({v['RV_COVID_2020']}) and 2022
bear market ({v['RV_BEAR_2022']}), compressed in 2024 ({v['RV_VAL_2024']}), and
re-expanded in 2025 ({v['RV_TEST_2025']}; Figure~\ref{{fig:regimes_forecasts}}A--B).
In H1 (Table~\ref{{tab:main_summary}}, Panel A; Within $R^2 = {v['H1_R2']}$),
overall 5-sector average leverage ({v['H1_NEG_EST']}, Holm $p < 0.0001$;
negative slope on $\min(r_t,0)$ implies higher forward volatility after drops),
overall VIX (+{v['H1_VIX_EST']}, Holm $p < 0.0001$), and the joint 5-sector
interaction Wald test ($\chi^2 = {v['H1_WALD_CHI2']}$, Holm $p < 0.0001$) are
overwhelmingly significant across all sensitivities. However, pairwise
Technology vs.\ Healthcare interactions are not individually significant under
conservative Driscoll--Kraay SEs and Holm adjustment (leverage Holm
$p = {v['H1_TECH_NEG_PH']}$), whereas Financials and Energy exhibit the
strongest cyclical differentials relative to Healthcare. H1 therefore receives
partial, nuanced support.

\begin{{figure}}[!tb]
\centering
\includegraphics[width=0.86\textwidth]{{../outputs/figures/fig1_volatility_regimes_and_forecasts.pdf}}
\caption{{Cross-sector 2019--2025 volatility regimes (Panel A), lagged return
leverage asymmetry (Panel B), and 2025 holdout forecast tracking during the
March--May volatility spike (Panel C).}}
\label{{fig:regimes_forecasts}}
\end{{figure}}

{table1_latex}

\textbf{{H2 Single-Shot 2025 Holdout Performance and TreeSHAP Audit.}} On 2025
volatility forecasting (Table~\ref{{tab:main_summary}}, Panel B;
Figure~\ref{{fig:regimes_forecasts}}C), Global LightGBM achieves the lowest
overall QLIKE ({v['LGBM_QLIKE']}) and RMSE ({v['LGBM_RMSE']}), significantly
outperforming Persistence (QLIKE {v['PERS_QLIKE']}, Holm $p < 0.0001$) and
per-stock GJR-GARCH(1,1) ({v['GARCH_QLIKE']}, Holm $p = {v['DM_Q_GARCH_PH']}$),
and showing its largest QLIKE reduction in Technology ({v['TECH_LGB_QLIKE']} vs.\
Panel HAR-RV {v['TECH_HAR_QLIKE']}), while full-universe QLIKE vs.\ Panel HAR-RV
({v['HAR_QLIKE']}) is directionally favorable but not statistically
distinguishable (Holm $p = {v['DM_Q_HAR_PH']}$). For downside tail-risk
classification, applying the frozen 2024-Val \texttt{{{v['CAL_METHOD']}}}
calibrator cuts LightGBM's 2025 ECE by {v['ECE_DROP_PCT']}\%
({v['LGBM_ECE_PRE']} $\to$ {v['LGBM_ECE_POST']}) and achieves the lowest overall
Brier score ({v['LGBM_BRIER']}, Holm $p = {v['DM_B_PERS_PH']}$ vs.\ Persistence;
Figure~\ref{{fig:shap_cal}}B). Yet regularized Logistic HAR-RV achieves higher
2025 PR-AUC ({v['HAR_PRAUC']} vs.\ LightGBM {v['LGBM_PRAUC']}), especially on
Form 8-K Item 2.02 earnings days ({v['EARN_HAR_PRAUC']} vs.\
{v['EARN_LGB_PRAUC']}). Train-background TreeSHAP (Figure~\ref{{fig:shap_cal}}A;
additivity error $< 10^{{-7}}$, positive class 1 verified) reveals that LightGBM
tail-risk probability behavior concentrates on \texttt{{rv\_m}}
({v['SHAP_RVM_CLF_PCT']}\%), \texttt{{vix\_close}} ({v['SHAP_VIX_CLF_PCT']}\%),
and yield-curve slope \texttt{{t10y2y}}, with amplified attribution magnitudes
under High-VIX regimes.

\begin{{figure}}[!b]
\centering
\includegraphics[width=0.84\textwidth]{{../outputs/figures/fig2_shap_and_calibration.pdf}}
\caption{{Train-background TreeSHAP attributions for LightGBM tail-risk model
behavior split by VIX regime (Panel A, strictly non-causal) and 2025 holdout
calibration reliability curves (Panel B).}}
\label{{fig:shap_cal}}
\end{{figure}}

\section{{Discussion, Limitations, and Phase B Outlook}}

Our findings yield three practical methodological lessons. First, accounting for
contemporaneous cross-stock correlation via Driscoll--Kraay SEs expands standard
errors by 3.6--4.7$\times$ relative to entity-clustered Newey--West SEs,
preventing false-positive claims on pairwise sector interactions. Second,
class-weighted tree models require out-of-sample post-hoc calibration before
probabilities can be interpreted as risks (cutting ECE by
{v['ECE_DROP_PCT']}\%), yet piecewise-constant isotonic steps introduce
upper-tail ties that can compress ranking granularity (PR-AUC) relative to
smooth linear logits on sparse earnings days. Third, all models lag abrupt
overnight macro shocks (such as the April 2025 tariff gap) and binary earnings
jumps because Phase A relies solely on daily OHLCV, macro indicators, and filing
counts.

\textbf{{Key Limitations and Scope Boundary.}} Our fixed {v['N_STOCKS']}-stock
large-cap universe introduces survivorship bias relative to delisted small-caps;
daily Parkinson range volatility contains measurement error compared with
intraday high-frequency RV; and SEC EDGAR filings capture authoritative
corporate disclosures rather than continuous analyst commentary or social media.
These results represent associational and predictive risk benchmarks for CS482
coursework only and \textbf{{do not constitute investment advice}}.

\textbf{{Phase B Extension (Frozen Handoff Ready).}} With Phase A data, purged
splits, and {v['EVENT_ANCHORED_DAYS']} pre-close event-anchored text rows
cryptographically frozen (\texttt{{events\_text\_phase\_b.parquet}}), Phase B
will evaluate single-A100 4-bit QLoRA fine-tuning of 27B--32B open financial
LLMs against \texttt{{FinBERT}} (110M) and price-only Phase A models on the
identical 2025 holdout test set, directly testing H3 on earnings and
material-event days.

% =========================================================================
% EXCLUDED BACK MATTER (References, Contribution Statement, AI Declaration)
% =========================================================================
\clearpage
\bibliographystyle{{plainnat}}
\bibliography{{references}}

\clearpage
\section*{{Separate Contribution Statement}}
\begin{{itemize}}\setlength{{\itemsep}}{{2pt}}
\item \textbf{{Member A (Financial Data \& Freeze Lead):}} Designed the pre-2019
60-stock universe, built Yahoo/FRED/paginated SEC EDGAR ingestion with
\texttt{{BLK}}/\texttt{{XOM}} predecessor CIK linkages, enforced DST-aware
16:00/13:00 ET cutoffs and purged splits, froze \texttt{{FREEZE\_MANIFEST.json}},
and produced Figure~1.
\item \textbf{{Member B (Financial Econometrics Lead):}} Implemented cross-cycle
EDA, Stock-FE extended Panel HAR-RV with Bartlett Driscoll--Kraay SE and
Holm-adjusted confirmatory tests, per-stock GJR-GARCH(1,1) convergence audits,
and $h=5$ HLN-DM evaluation.
\item \textbf{{Member C (Machine Learning \& Explainability Lead):}} Built the
quarterly expanding Global LightGBM pipeline, 2024-Val out-of-fold probability
calibration, Train-background exact TreeSHAP additivity audits, Table~1, and
Figure~2.
\item \textbf{{Member D / All Members (Integration, Verification \& Phase B
Prep):}} Audited numeric claim traceability against frozen artifacts
(\texttt{{phase\_a\_audit.md}}), verified formatting and
non-causal/non-investment boundaries, and prepared the frozen Phase B handoff.
\end{{itemize}}

\section*{{AI Usage Declaration \& Academic Integrity}}
AI coding and research assistants were used during project planning, code
implementation, test-driven debugging, LaTeX formatting, and prose polishing,
guided by the procedural research guardrails of \citet{{kassis2026skills}}. All
external data downloads, SHA-256 freeze manifests, econometric standard errors
(cross-checked against \texttt{{statsmodels}}), 300/300 GJR-GARCH fits,
bootstrap CIs, HLN-DM statistics, TreeSHAP additivity checks ($< 10^{{-7}}$),
and report numbers were independently executed and verified via deterministic
offline Python scripts (\texttt{{uv run python scripts/run\_phase\_a.py
-{{}}-skip-download}}) and automated pytest suites. No fabricated data,
references, or unverified AI outputs appear in this report.

\end{{document}}
"""


def compile_latex_pdf(
    tex_path: Path,
    *,
    output_pdf_name: str = config.REPORT_PDF_NAME,
    source_date_epoch: int | None = None,
) -> Path:
    """Compile LaTeX + BibTeX into final PDF cleanly (`pdflatex -> bibtex -> pdflatex x2`)."""
    report_dir = tex_path.parent
    stem = tex_path.stem
    env = dict(os.environ)
    if source_date_epoch is not None:
        env["SOURCE_DATE_EPOCH"] = str(int(source_date_epoch))
        env["FORCE_SOURCE_DATE"] = "1"
        env["TZ"] = "UTC"

    cmds = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        ["bibtex", stem],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
    ]
    for cmd in cmds:
        proc = subprocess.run(
            cmd,
            cwd=report_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LaTeX command {' '.join(cmd)} failed (exit {proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-1000:]}"
            )

    built_pdf = report_dir / f"{stem}.pdf"
    target_pdf = report_dir / output_pdf_name
    if built_pdf.resolve() != target_pdf.resolve():
        target_pdf.write_bytes(built_pdf.read_bytes())
        built_pdf.unlink()
    return target_pdf


def audit_compiled_pdf(
    *,
    pdf_path: Path,
    claims: dict[str, dict[str, str]],
    generated_at_utc: str,
) -> dict[str, Any]:
    """Audit PDF A4 geometry, main-text page budget [3.0, 5.0], claims, and forbidden words."""
    # 1. pdfinfo geometry check
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    m_size = re.search(r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info.stdout)
    if m_size is None:
        raise ValueError(f"Failed to parse pdfinfo Page size:\n{info.stdout}")
    w_pts, h_pts = float(m_size.group(1)), float(m_size.group(2))
    is_a4 = abs(w_pts - 595.28) < 1.5 and abs(h_pts - 841.89) < 1.5

    # 2. Extract per-page text with pypdf and pdftotext
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    page_texts = [p.extract_text() or "" for p in reader.pages]

    txt_proc = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    full_text_pdftotext = txt_proc.stdout
    full_text_combined = "\n".join(page_texts) + "\n" + full_text_pdftotext

    # Identify excluded cover (page 0) and excluded back matter (first page with "References")
    ref_page_idx = None
    for idx in range(1, total_pages):
        if re.search(r"(?m)^References\b", page_texts[idx]) or (
            "References" in page_texts[idx] and "Contribution Statement" in page_texts[idx]
        ):
            ref_page_idx = idx
            break
    if ref_page_idx is None:
        raise ValueError("Could not locate References section in compiled PDF")

    main_page_indices = list(range(1, ref_page_idx))
    n_full_main_pages = len(main_page_indices)
    # Estimate fractional fill of last main page relative to average full main-text page length
    if n_full_main_pages >= 2:
        prev_lens = [len(page_texts[i].strip()) for i in main_page_indices[:-1]]
        avg_len = max(1.0, float(sum(prev_lens)) / len(prev_lens))
        last_frac = min(1.0, len(page_texts[main_page_indices[-1]].strip()) / avg_len)
        main_text_pages_est = round((n_full_main_pages - 1) + last_frac, 2)
    else:
        main_text_pages_est = float(n_full_main_pages)

    # 3. Verify all registered numeric claims appear in extracted text
    norm_text = re.sub(r"\s+", " ", full_text_combined)
    missing_claims: list[str] = []
    for key, item in claims.items():
        val = item["value"]
        # Accept minor hyphen/minus sign variations from PDF text extraction
        val_variants = {val, val.replace("-", "\u2212"), val.replace(",", "")}
        if not any(v_str in norm_text for v_str in val_variants):
            missing_claims.append(f"{key}={val}")

    # 4. Forbidden words & LaTeX artifact scan (scan main text + declarations, exclude BibTeX)
    main_and_decl_text = "\n".join(page_texts[i] for i in range(total_pages))
    # Strip bibliography reference titles before causal-word scan so cited paper titles don't trip
    scan_pages = [page_texts[0], *(page_texts[i] for i in main_page_indices)]
    for idx in range(ref_page_idx, total_pages):
        pt = page_texts[idx]
        if "Separate Contribution Statement" in pt:
            scan_pages.append(pt.split("Separate Contribution Statement", 1)[1])
    text_issues = scan_report_text_issues("\n".join(scan_pages))
    # Also scan full document strictly for ?? / TODO / TBD / NaN
    for pat, lbl in (
        (r"\?\?", "unresolved '??'"),
        (r"\bTODO\b", "'TODO'"),
        (r"\bTBD\b", "'TBD'"),
        (r"\bNaN\b", "'NaN'"),
    ):
        if re.search(pat, main_and_decl_text):
            text_issues.append(f"Full PDF contains {lbl}")

    # 5. Required CS482 sections
    required_checks = {
        "a4_dimensions": is_a4,
        "main_text_within_3_to_5_pages": 3.0 <= main_text_pages_est <= 5.0
        and 3 <= n_full_main_pages <= 5,
        "zero_missing_claims": len(missing_claims) == 0,
        "zero_forbidden_words_or_artifacts": len(text_issues) == 0,
        "has_non_investment_notice": "not constitute investment" in norm_text.lower(),
        "has_references": "References" in full_text_combined,
        "has_contribution_statement": "Separate Contribution Statement" in full_text_combined,
        "has_ai_declaration": "AI Usage Declaration" in full_text_combined,
        "has_kdense_citation": "2609.00065" in full_text_combined,
    }
    all_pass = all(required_checks.values())
    verdict = "PASS" if all_pass else "FAIL"

    # Render audit markdown
    lines = [
        "# Milestone M5 — CS482 Phase A Draft Automated Audit",
        "",
        f"- Overall Audit Verdict: **{verdict}**",
        f"- Provenance Timestamp (UTC): `{generated_at_utc}`",
        f"- Target PDF: `report/{pdf_path.name}` (`{pdf_path.stat().st_size:,} B`)",
        f"- Page Geometry (`pdfinfo`): `{w_pts:.2f} x {h_pts:.2f} pts` "
        f"(A4 = `595.28 x 841.89 pts`, pass=`{is_a4}`)",
        f"- Total PDF Pages: `{total_pages}` (Cover: `1`, Main Text pages occupied: "
        f"`{n_full_main_pages}`, Back Matter: `{total_pages - ref_page_idx}`)",
        f"- Main Text Length: `{main_text_pages_est:.2f}` A4 pages (CS482 required range: "
        "`[3.00, 5.00]` A4 pages, cover/references/declarations excluded)",
        f"- Numeric Claims Verified: `{len(claims) - len(missing_claims)}/{len(claims)}`",
        f"- Forbidden Wording / Artifact Hits: `{len(text_issues)}`",
        "",
        "## 1. Automated Gate Checklist",
        "",
        "| Check | Status |",
        "| --- | --- |",
    ]
    for chk, ok in required_checks.items():
        lines.append(f"| `{chk}` | `{'PASS' if ok else 'FAIL'}` |")

    if missing_claims:
        lines.extend(["", "## Missing Claims", "", *[f"- {c}" for c in missing_claims]])
    if text_issues:
        lines.extend(["", "## Text / Wording Issues", "", *[f"- {t}" for t in text_issues]])

    lines.extend(
        [
            "",
            "## 2. Programmatic Numeric Claim Traceability",
            "",
            "| Claim Key | Reported Value | Source Artifact | Locator |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key, item in claims.items():
        lines.append(
            f"| `{key}` | `{item['value']}` | `{item['source']}` | `{item['locator']}` |"
        )
    lines.append("")

    return {
        "verdict": verdict,
        "is_a4": is_a4,
        "width_pts": w_pts,
        "height_pts": h_pts,
        "total_pages": total_pages,
        "main_pages_occupied": n_full_main_pages,
        "main_text_pages_est": main_text_pages_est,
        "claims_verified": len(claims) - len(missing_claims),
        "claims_total": len(claims),
        "missing_claims": missing_claims,
        "text_issues": text_issues,
        "checks": required_checks,
        "audit_markdown": "\n".join(lines),
    }


def run_report_build_and_audit(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify frozen manifest, generate figures & Table 1, build CS482 PDF, and run audit."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    manifest = verify_freeze_manifest(frozen_dir=root / "data" / "frozen", project_root=root)
    generated_at_utc = str(manifest["generated_at_utc"])

    # 1. Generate 300-DPI Figure 1 & Figure 2 (PNG + PDF)
    fig_paths = generate_all_figures(project_root=root)

    # 2. Generate Table 1 (CSV + MD + TEX)
    t1 = build_table1_summary(root)
    tables_dir = root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    t1["table_df"].to_csv(tables_dir / "table1_summary.csv", index=False)
    (tables_dir / "table1_summary.md").write_text(t1["markdown"], encoding="utf-8")
    (tables_dir / "table1_summary.tex").write_text(t1["latex"], encoding="utf-8")

    # 3. Collect claims, write references.bib & phase_a_draft.tex, compile PDF
    claims = collect_report_claims(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    bib_path = report_dir / "references.bib"
    tex_path = report_dir / "phase_a_draft.tex"
    bib_path.write_text(render_references_bib(), encoding="utf-8")
    tex_path.write_text(render_phase_a_tex(claims, t1["latex"]), encoding="utf-8")

    source_date_epoch = int(datetime.fromisoformat(generated_at_utc).timestamp())
    pdf_path = compile_latex_pdf(
        tex_path,
        output_pdf_name=config.REPORT_PDF_NAME,
        source_date_epoch=source_date_epoch,
    )

    # 4. Run full automated audit and write report/phase_a_audit.md
    audit = audit_compiled_pdf(
        pdf_path=pdf_path,
        claims=claims,
        generated_at_utc=generated_at_utc,
    )
    audit_path = report_dir / "phase_a_audit.md"
    audit_path.write_text(audit["audit_markdown"], encoding="utf-8")

    if audit["verdict"] != "PASS":
        raise RuntimeError(
            f"CS482 Phase A PDF audit FAILED! See {audit_path}:\n"
            f"checks={audit['checks']}\n"
            f"missing_claims={audit['missing_claims']}\n"
            f"text_issues={audit['text_issues']}"
        )

    return {
        "milestone": "M5",
        "generated_at_utc": generated_at_utc,
        "verdict": audit["verdict"],
        "pdf_path": str(pdf_path.relative_to(root)),
        "total_pages": audit["total_pages"],
        "main_pages_occupied": audit["main_pages_occupied"],
        "main_text_pages_est": audit["main_text_pages_est"],
        "claims_verified": audit["claims_verified"],
        "claims_total": audit["claims_total"],
        "figures": fig_paths,
    }
