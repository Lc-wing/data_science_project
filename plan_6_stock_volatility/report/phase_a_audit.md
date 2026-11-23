# Milestone M5 — CS482 Phase A Draft Automated Audit

- Overall Audit Verdict: **PASS**
- Provenance Timestamp (UTC): `2026-11-22T14:04:28+00:00`
- Target PDF: `report/CS482_GroupX_StockVolatilityLLM_PhaseA_Draft.pdf` (`429,991 B`)
- Page Geometry (`pdfinfo`): `595.28 x 841.89 pts` (A4 = `595.28 x 841.89 pts`, pass=`True`)
- Total PDF Pages: `7` (Cover: `1`, Main Text pages occupied: `4`, Back Matter: `2`)
- Main Text Length: `4.00` A4 pages (CS482 required range: `[3.00, 5.00]` A4 pages, cover/references/declarations excluded)
- Numeric Claims Verified: `51/51`
- Forbidden Wording / Artifact Hits: `0`

## 1. Automated Gate Checklist

| Check | Status |
| --- | --- |
| `a4_dimensions` | `PASS` |
| `main_text_within_3_to_5_pages` | `PASS` |
| `zero_missing_claims` | `PASS` |
| `zero_forbidden_words_or_artifacts` | `PASS` |
| `has_non_investment_notice` | `PASS` |
| `has_references` | `PASS` |
| `has_contribution_statement` | `PASS` |
| `has_ai_declaration` | `PASS` |
| `has_kdense_citation` | `PASS` |

## 2. Programmatic Numeric Claim Traceability

| Claim Key | Reported Value | Source Artifact | Locator |
| --- | --- | --- | --- |
| `N_STOCKS` | `60` | `data/raw/universe_60.csv` | `len(rows)` |
| `N_DAYS` | `1,760` | `data/frozen/split_metadata.json` | `total_canonical_trading_days` |
| `PANEL_ROWS` | `105,600` | `data/frozen/split_metadata.json` | `total_panel_rows` |
| `T10Y2Y_FFILL` | `13` | `outputs/metrics/day1_data_audit.json` | `macro.t10y2y_ffilled_count` |
| `SEC_ARCHIVES` | `195` | `outputs/metrics/day1_data_audit.json` | `sec_edgar.overlapping_archives_fetched` |
| `SEC_FILINGS_RAW` | `7,703` | `outputs/metrics/day1_data_audit.json` | `sec_edgar.total_unique_filings` |
| `EARLY_CLOSE_SHIFTED` | `2` | `data/frozen/split_metadata.json` | `event_alignment_audit.shifted_due_to_early_close` |
| `DEDUP_DROPPED` | `29` | `data/frozen/split_metadata.json` | `event_dedup_and_cap_audit.dedup_headline_dropped` |
| `EVENT_ANCHORED_DAYS` | `6,822` | `data/frozen/split_metadata.json` | `event_dedup_and_cap_audit.active_ticker_days` |
| `TRAIN_MOD_ROWS` | `73,860` | `data/frozen/split_metadata.json` | `splits.train.modeling_complete_rows` |
| `VAL_MOD_ROWS` | `14,760` | `data/frozen/split_metadata.json` | `splits.val.modeling_complete_rows` |
| `TEST_MOD_ROWS` | `14,700` | `data/frozen/split_metadata.json` | `splits.test.modeling_complete_rows` |
| `TEST_DAYS` | `245` | `data/frozen/split_metadata.json` | `test_scorable_trading_days` |
| `RV90_THRESH` | `0.3742` | `data/frozen/feature_manifest.json` | `train_preprocessing.rv90_threshold` |
| `PANEL_SHA12` | `3515fcd12cab` | `data/frozen/FREEZE_MANIFEST.json` | `files.panel_daily.parquet.sha256[:12]` |
| `VIX_MEDIAN` | `19.61` | `outputs/metrics/eda_summary.json` | `vix_train_median_threshold` |
| `RV_COVID_2020` | `0.3241` | `outputs/metrics/eda_summary.json` | `regimes_all.covid_shock_2020.rv_5d_fwd_mean` |
| `RV_BEAR_2022` | `0.2687` | `outputs/metrics/eda_summary.json` | `regimes_all.rate_hike_bear_2022.rv_5d_fwd_mean` |
| `RV_VAL_2024` | `0.1998` | `outputs/metrics/eda_summary.json` | `regimes_all.ai_cycle_val_2024.rv_5d_fwd_mean` |
| `RV_TEST_2025` | `0.2411` | `outputs/metrics/eda_summary.json` | `regimes_all.ai_cycle_test_2025.rv_5d_fwd_mean` |
| `COV_RATE_RATIO` | `0.8860` | `outputs/metrics/eda_summary.json` | `coverage_all.event_day_rate_ratio_2425_vs_1923` |
| `H1_N_OBS` | `104,040` | `outputs/metrics/h1_regression.json` | `main_model.n_obs` |
| `H1_R2` | `0.4736` | `outputs/metrics/h1_regression.json` | `main_model.within_r2` |
| `H1_DK_L` | `7` | `outputs/metrics/h1_regression.json` | `main_model.bartlett_bandwidth_L` |
| `H1_NEG_EST` | `-1.4458` | `outputs/metrics/h1_regression.json` | `confirmatory_family_holm.neg_ret_1d.estimate` |
| `H1_VIX_EST` | `0.0141` | `outputs/metrics/h1_regression.json` | `confirmatory_family_holm.vix_close.estimate` |
| `H1_TECH_NEG_PH` | `0.171` | `outputs/metrics/h1_regression.json` | `confirmatory_family_holm.tech_x_neg_ret_1d_vs_healthcare.p_holm` |
| `H1_WALD_CHI2` | `84.84` | `outputs/metrics/h1_regression.json` | `confirmatory_family_holm.joint_5sector_interaction_wald.estimate` |
| `GARCH_CONV` | `300/300` | `outputs/metrics/h2_baselines.json` | `garch_converged_fits/total_garch_fits` |
| `CAL_METHOD` | `isotonic` | `outputs/metrics/h2_lgbm_selection.json` | `selected_calibrator.method` |
| `LGBM_QLIKE` | `0.2306` | `outputs/metrics/h2_evaluation.json` | `test_metrics.lgbm.qlike` |
| `HAR_QLIKE` | `0.2383` | `outputs/metrics/h2_evaluation.json` | `test_metrics.har_rv.qlike` |
| `GARCH_QLIKE` | `0.3007` | `outputs/metrics/h2_evaluation.json` | `test_metrics.gjr_garch.qlike` |
| `PERS_QLIKE` | `0.3144` | `outputs/metrics/h2_evaluation.json` | `test_metrics.persistence.qlike` |
| `LGBM_RMSE` | `0.0951` | `outputs/metrics/h2_evaluation.json` | `test_metrics.lgbm.rmse` |
| `HAR_PRAUC` | `0.1422` | `outputs/metrics/h2_evaluation.json` | `test_metrics.har_rv.pr_auc` |
| `LGBM_PRAUC` | `0.1231` | `outputs/metrics/h2_evaluation.json` | `test_metrics.lgbm.pr_auc` |
| `LGBM_BRIER` | `0.0773` | `outputs/metrics/h2_evaluation.json` | `test_metrics.lgbm.brier` |
| `LGBM_ECE_PRE` | `0.3804` | `outputs/metrics/h2_evaluation.json` | `test_metrics.lgbm.ece_pre_cal` |
| `LGBM_ECE_POST` | `0.0102` | `outputs/metrics/h2_evaluation.json` | `test_metrics.lgbm.ece` |
| `ECE_DROP_PCT` | `97.3` | `outputs/metrics/h2_evaluation.json` | `100*(1 - ece/ece_pre_cal)` |
| `DM_Q_GARCH_PH` | `0.0207` | `outputs/metrics/h2_evaluation.json` | `dm_tests.qlike.gjr_garch.p_holm` |
| `DM_Q_HAR_PH` | `0.134` | `outputs/metrics/h2_evaluation.json` | `dm_tests.qlike.har_rv.p_holm` |
| `DM_B_PERS_PH` | `0.0470` | `outputs/metrics/h2_evaluation.json` | `dm_tests.brier.persistence.p_holm` |
| `TECH_LGB_QLIKE` | `0.2862` | `outputs/tables/h2_test_by_sector.csv` | `Technology.lgbm.qlike` |
| `TECH_HAR_QLIKE` | `0.3369` | `outputs/tables/h2_test_by_sector.csv` | `Technology.har_rv.qlike` |
| `EARN_HAR_PRAUC` | `0.2861` | `outputs/tables/h2_test_by_event_day.csv` | `Has_8K_Earnings.har_rv.pr_auc` |
| `EARN_LGB_PRAUC` | `0.0948` | `outputs/tables/h2_test_by_event_day.csv` | `Has_8K_Earnings.lgbm.pr_auc` |
| `SHAP_BG_N` | `1,000` | `outputs/metrics/shap_audit.json` | `background_n` |
| `SHAP_RVM_CLF_PCT` | `24.4` | `outputs/metrics/shap_audit.json` | `top_features_classifier_prob[0].share_pct` |
| `SHAP_VIX_CLF_PCT` | `11.9` | `outputs/metrics/shap_audit.json` | `top_features_classifier_prob[1].share_pct` |
