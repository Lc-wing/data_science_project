# Phase A numeric audit

Numeric audit: PASS.
Every cited number matches its source within rounding tolerance.
Main text length: 5.00 A4 pages (cover, references, and declarations excluded).

## Minimum requirements

- Research question: H1, H2, and H3, with their nulls and the H3 rule.
- Data: DSEC, State Council notices, and SMG, with the frozen panel counts.
- Analysis: STL, PELT (exploratory only), the regression, and the forecasts.
- Evaluation: Driscoll--Kraay intervals, Holm, HLN-DM, MAE, RMSE, and MASE.
- Results: Figure 1, Table 1, Figure 2, and the three answers.

Phase B is not estimated in this draft.

## Claim trace

| Claim | Reported | Source | Locator |
| --- | --- | --- | --- |
| WINDOW | 2010-01 to 2026-09 | outputs/metrics/day1_data_audit.json | panel_window |
| PANEL_ROWS | 804 | outputs/metrics/day1_data_audit.json | tier_month_observations |
| N_TIERS | 4 | outputs/metrics/day1_data_audit.json | panel_window.tiers |
| N_MONTHS | 201 | outputs/metrics/day1_data_audit.json | months_by_period.all_months |
| PRE_MONTHS | 120 | outputs/metrics/day1_data_audit.json | pre_pandemic |
| PANDEMIC_MONTHS | 35 | outputs/metrics/day1_data_audit.json | pandemic |
| POST_MONTHS | 45 | outputs/metrics/day1_data_audit.json | post_pandemic |
| MAIN_MONTHS | 165 | outputs/metrics/regression_summary.json | main_model.months |
| MAIN_ROWS | 660 | outputs/metrics/regression_summary.json | main_model.n |
| POST_ROWS | 180 | data/frozen/FREEZE_MANIFEST.json | post_pandemic_rows |
| OCC_MIN | 5.0 | outputs/tables/panel_variable_summary.csv | occupancy_pct.min |
| OCC_MAX | 97.2 | outputs/tables/panel_variable_summary.csv | occupancy_pct.max |
| HOLIDAY_PERIODS | 115 | outputs/metrics/day1_data_audit.json | cn_holidays.periods |
| HOLIDAY_DAYS | 492 | outputs/metrics/day1_data_audit.json | holiday_days |
| TYPHOON_DAYS | 44 | outputs/metrics/day1_data_audit.json | typhoon_days |
| EST_CHANGES | 87 | outputs/metrics/day1_data_audit.json | establishment changes |
| ROOM_JUMPS | 22 | outputs/metrics/day1_data_audit.json | room change of 10 percent |
| ROOMS_BEFORE | 2,528 | outputs/metrics/day1_data_audit.json | largest_room_change.rooms_before |
| ROOMS_AFTER | 5,477 | outputs/metrics/day1_data_audit.json | largest_room_change.rooms_after |
| VAL_ORIGINS | 36 | data/frozen/FREEZE_MANIFEST.json | validation_origins |
| TEST_ORIGINS | 26 | data/frozen/FREEZE_MANIFEST.json | test_origins |
| TEST_FORECASTS | 104 | data/frozen/FREEZE_MANIFEST.json | test_forecasts |
| PANEL_SHA | 9c9547d653c5 | data/frozen/FREEZE_MANIFEST.json | panel sha256 |
| PRE_SAME | 51.35 | outputs/tables/eda_visitor_mix_by_period.csv | pre_pandemic.mean_same_day_share_pct |
| POST_SAME | 55.34 | outputs/tables/eda_visitor_mix_by_period.csv | post_pandemic.mean_same_day_share_pct |
| PRE_INTL | 9.82 | outputs/tables/eda_visitor_mix_by_period.csv | pre_pandemic.mean_international_share_pct |
| POST_INTL | 6.26 | outputs/tables/eda_visitor_mix_by_period.csv | post_pandemic.mean_international_share_pct |
| PRE_TOUR | 28.05 | outputs/tables/eda_visitor_mix_by_period.csv | pre_pandemic.mean_package_tour_share_pct |
| POST_TOUR | 5.36 | outputs/tables/eda_visitor_mix_by_period.csv | post_pandemic.mean_package_tour_share_pct |
| PRE_ON | 1,263,786 | outputs/tables/eda_visitor_mix_by_period.csv | pre_pandemic.mean_visitors_overnight |
| POST_ON | 1,315,841 | outputs/tables/eda_visitor_mix_by_period.csv | post_pandemic.mean_visitors_overnight |
| PAN_INTL | 0.5 | outputs/tables/eda_visitor_mix_by_period.csv | pandemic.mean_international_share_pct |
| PAN_TOUR | 0.5 | outputs/tables/eda_visitor_mix_by_period.csv | pandemic.mean_package_tour_share_pct |
| CHANGE_POINTS | Change points that remain at every reported penalty are 2021-08 and 2022-08 for the same-day share; 2020-03 for the international share; 2020-01 and 2023-03 for the package-tour share. | outputs/tables/eda_change_point_runs.csv | all penalties |
| OCC5_PRE | 85.6 | outputs/tables/eda_occupancy_by_period.csv | 5star.pre_pandemic |
| OCC5_POST | 89.0 | outputs/tables/eda_occupancy_by_period.csv | 5star.post_pandemic |
| OCC4_PRE | 86.9 | outputs/tables/eda_occupancy_by_period.csv | 4star.pre_pandemic |
| OCC4_POST | 83.0 | outputs/tables/eda_occupancy_by_period.csv | 4star.post_pandemic |
| OCC3_PRE | 85.6 | outputs/tables/eda_occupancy_by_period.csv | 3star.pre_pandemic |
| OCC3_POST | 83.6 | outputs/tables/eda_occupancy_by_period.csv | 3star.post_pandemic |
| OCC2_PRE | 73.7 | outputs/tables/eda_occupancy_by_period.csv | 2star.pre_pandemic |
| OCC2_POST | 88.0 | outputs/tables/eda_occupancy_by_period.csv | 2star.post_pandemic |
| BANDWIDTH | 4 | outputs/metrics/regression_summary.json | bandwidth |
| N_COEF | 34 | outputs/metrics/regression_summary.json | parameters |
| R2 | 0.755 | outputs/metrics/regression_summary.json | r_squared |
| H1_EST | 18.33 | outputs/tables/regression_confirmatory.csv | H1.estimate |
| H1_SE | 6.94 | outputs/tables/regression_confirmatory.csv | H1.se |
| H1_LO | 4.73 | outputs/tables/regression_confirmatory.csv | H1.ci_low |
| H1_HI | 31.93 | outputs/tables/regression_confirmatory.csv | H1.ci_high |
| H1_HOLM | 0.016 | outputs/tables/regression_confirmatory.csv | H1.p_holm |
| H2_EST | 0.04 | outputs/tables/regression_confirmatory.csv | H2.estimate |
| H2_SE | 0.15 | outputs/tables/regression_confirmatory.csv | H2.se |
| H2_LO | -0.26 | outputs/tables/regression_confirmatory.csv | H2.ci_low |
| H2_HI | 0.34 | outputs/tables/regression_confirmatory.csv | H2.ci_high |
| H2_HOLM | 0.804 | outputs/tables/regression_confirmatory.csv | H2.p_holm |
| B5_POST | 21.23 | outputs/tables/regression_slopes.csv | 5star.post |
| B5_LO | 14.28 | outputs/tables/regression_slopes.csv | 5star.ci_low |
| B5_HI | 28.17 | outputs/tables/regression_slopes.csv | 5star.ci_high |
| B3_POST | 2.89 | outputs/tables/regression_slopes.csv | 3star.post |
| B3_LO | -8.83 | outputs/tables/regression_slopes.csv | 3star.ci_low |
| B3_HI | 14.62 | outputs/tables/regression_slopes.csv | 3star.ci_high |
| TEN_5 | 2.02 | outputs/tables/regression_slopes.csv | 5star post times ln(1.1) |
| TEN_3 | 0.28 | outputs/tables/regression_slopes.csv | 3star post times ln(1.1) |
| DF_EQ | 3 | outputs/tables/regression_secondary.csv | equal slopes df |
| CHI_EQ | 15.3 | outputs/tables/regression_secondary.csv | equal slopes chi2 |
| P_EQ | 0.002 | outputs/tables/regression_secondary.csv | equal slopes p |
| DF_CHOW | 5 | outputs/tables/regression_secondary.csv | chow df |
| CHI_CHOW | 36.4 | outputs/tables/regression_secondary.csv | chow chi2 |
| P_CHOW | < 0.001 | outputs/tables/regression_secondary.csv | chow p |
| S5_HOLM | 0.060 | outputs/tables/regression_sensitivity.csv | S5 H1 holm |
| S4_EST | 0.39 | outputs/tables/regression_sensitivity.csv | S4 H2 estimate |
| S4_HOLM | 0.020 | outputs/tables/regression_sensitivity.csv | S4 H2 holm |
| LB_LOW | 14 | outputs/metrics/regression_summary.json | ljung_box_below_0.05 |
| LB_N | 16 | outputs/metrics/regression_summary.json | ljung_box_cells |
| BP_P | < 0.001 | outputs/metrics/regression_summary.json | breusch_pagan_p |
| MAE_LGB | 2.06 | outputs/tables/forecast_test_metrics.csv | main.lightgbm.mae |
| MAE_SAR | 2.36 | outputs/tables/forecast_test_metrics.csv | main.sarimax.mae |
| MAE_NAIVE | 3.26 | outputs/tables/forecast_test_metrics.csv | main.seasonal_naive.mae |
| RMSE_LGB | 2.83 | outputs/tables/forecast_test_metrics.csv | main.lightgbm.rmse |
| RMSE_SAR | 3.42 | outputs/tables/forecast_test_metrics.csv | main.sarimax.rmse |
| RMSE_NAIVE | 4.11 | outputs/tables/forecast_test_metrics.csv | main.seasonal_naive.rmse |
| MASE_LGB | 0.47 | outputs/tables/forecast_test_metrics.csv | main.lightgbm.mase |
| MASE_SAR | 0.53 | outputs/tables/forecast_test_metrics.csv | main.sarimax.mase |
| MASE_NAIVE | 0.75 | outputs/tables/forecast_test_metrics.csv | main.seasonal_naive.mase |
| MAE5 | 1.66 | outputs/tables/forecast_test_metrics.csv | lightgbm.5star.mae |
| MAE4 | 2.16 | outputs/tables/forecast_test_metrics.csv | lightgbm.4star.mae |
| MAE3 | 2.45 | outputs/tables/forecast_test_metrics.csv | lightgbm.3star.mae |
| MAE2 | 1.99 | outputs/tables/forecast_test_metrics.csv | lightgbm.2star.mae |
| DM_N_EST | -1.19 | outputs/tables/forecast_dm_tests.csv | lightgbm vs seasonal_naive.difference |
| DM_N_LO | -1.92 | outputs/tables/forecast_dm_tests.csv | lightgbm vs seasonal_naive.ci_low |
| DM_N_HI | -0.46 | outputs/tables/forecast_dm_tests.csv | lightgbm vs seasonal_naive.ci_high |
| DM_N_HOLM | 0.022 | outputs/tables/forecast_dm_tests.csv | lightgbm vs seasonal_naive.p_holm |
| DM_S_EST | -0.30 | outputs/tables/forecast_dm_tests.csv | lightgbm vs sarimax.difference |
| DM_S_LO | -1.04 | outputs/tables/forecast_dm_tests.csv | lightgbm vs sarimax.ci_low |
| DM_S_HI | 0.44 | outputs/tables/forecast_dm_tests.csv | lightgbm vs sarimax.ci_high |
| DM_S_HOLM | 1.000 | outputs/tables/forecast_dm_tests.csv | lightgbm vs sarimax.p_holm |
| F1_LGB | 1.98 | outputs/tables/forecast_test_metrics.csv | F1_logit.lightgbm.mae |
| F1_SAR | 2.25 | outputs/tables/forecast_test_metrics.csv | F1_logit.sarimax.mae |
| F2_LGB | 4.52 | outputs/tables/forecast_test_metrics.csv | F2_pandemic_included.lightgbm.mae |
| F2_SAR | 2.85 | outputs/tables/forecast_test_metrics.csv | F2_pandemic_included.sarimax.mae |
| F2_NAIVE | 3.26 | outputs/tables/forecast_test_metrics.csv | F2_pandemic_included.seasonal_naive.mae |
| F2_BIAS | -3.00 | outputs/tables/forecast_test_forecasts.csv | F2 mean forecast minus actual |
| N_FITS | 312 | outputs/tables/forecast_test_forecasts.csv | all test fits |
| SHAP1 | 1.82 | outputs/tables/forecast_shap_importance.csv | row 0 all_classes |
| SHAP2 | 1.23 | outputs/tables/forecast_shap_importance.csv | row 1 all_classes |
| SHAP3 | 1.22 | outputs/tables/forecast_shap_importance.csv | row 2 all_classes |
