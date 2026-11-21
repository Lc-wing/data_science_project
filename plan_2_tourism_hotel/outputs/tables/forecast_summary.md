# Rolling-origin backtest for H3 (Plan 2, Phase A step 5)

One-month-ahead forecasts of occupancy per star class. The comparison is predictive only. SHAP values describe how the fitted model behaves; they are not causal effects.

Frozen panel SHA-256: `9c9547d653c52a168f5b7d8b632458b7d62eece2c0e66c46fcbd8934de479729`  
Analysis specification SHA-256: `347c14d671f53f9ba3fc472cb1eba972b4519c146a920144d9c356a71ca1a4d5` (version 1.0.0)  
Deviations from the specification: 0

## Design

- Test: 26 origins, 2024-07 to 2026-08, 104 forecasts. Every model is refit at every origin on months up to the origin.
- Validation: 36 origins, all before the first test origin. Orders and hyperparameters were chosen there and written to `outputs/metrics/forecast_selection.json` before any test month was forecast.
- pseudo-out-of-sample: the forecast of month t+1 uses the DSEC values of months up to t as published on the download date. DSEC's publication lag and later revisions are not reproduced, so the errors are not those of a forecast made in real time. The holiday features are the exception: they follow the notices published by the end of month t.

## Choices made on the validation origins

| run | 5star | 4star | 3star | 2star | lightgbm |
| --- | --- | --- | --- | --- | --- |
| main | [1, 0, 1] × [0, 1, 1, 12] | [1, 1, 1] × [0, 1, 1, 12] | [1, 1, 1] × [0, 1, 1, 12] | [1, 1, 1] × [0, 1, 1, 12] | {"num_leaves": 8, "min_data_in_leaf": 10, "n_estimators": 400} |
| F1_logit | [1, 0, 1] × [0, 1, 1, 12] | [0, 1, 1] × [0, 1, 1, 12] | [0, 1, 1] × [0, 1, 1, 12] | [1, 1, 1] × [0, 1, 1, 12] | {"num_leaves": 8, "min_data_in_leaf": 10, "n_estimators": 400} |
| F2_pandemic_included | [1, 1, 1] × [1, 0, 1, 12] | [1, 1, 1] × [0, 1, 1, 12] | [1, 1, 1] × [1, 0, 1, 12] | [1, 0, 0] × [0, 1, 1, 12] | {"num_leaves": 4, "min_data_in_leaf": 10, "n_estimators": 400} |

Validation metrics of the chosen models (the chosen models were picked on these months, so their validation errors flatter them). The seasonal-naive error is large because the targets 2023-07 to 2023-12 use a pandemic month as the forecast; no test forecast does.

| model | unit | forecasts | mae | rmse | mase |
| --- | --- | --- | --- | --- | --- |
| seasonal_naive | 5star | 36 | 12.236 | 22.487 | 3.118 |
| seasonal_naive | 4star | 36 | 8.561 | 16.101 | 1.941 |
| seasonal_naive | 3star | 36 | 8.283 | 14.707 | 2.115 |
| seasonal_naive | 2star | 36 | 9.317 | 14.469 | 1.568 |
| seasonal_naive | pooled | 144 | 9.599 | 17.252 | 2.186 |
| sarimax | 5star | 36 | 1.552 | 2.346 | 0.396 |
| sarimax | 4star | 36 | 2.102 | 2.677 | 0.477 |
| sarimax | 3star | 36 | 2.022 | 2.471 | 0.516 |
| sarimax | 2star | 36 | 3.055 | 3.877 | 0.514 |
| sarimax | pooled | 144 | 2.183 | 2.907 | 0.476 |
| lightgbm | 5star | 36 | 2.229 | 3.108 | 0.568 |
| lightgbm | 4star | 36 | 2.274 | 3.268 | 0.516 |
| lightgbm | 3star | 36 | 2.657 | 3.181 | 0.678 |
| lightgbm | 2star | 36 | 3.540 | 4.723 | 0.596 |
| lightgbm | pooled | 144 | 2.675 | 3.632 | 0.590 |

## Test metrics (main run)

MASE scale: per star class, mean absolute 12-month difference of occupancy over months up to 2024-06 for which both the month and the month one year earlier are in the main sample; fixed for all origins.

| model | unit | forecasts | mae | rmse | mase |
| --- | --- | --- | --- | --- | --- |
| seasonal_naive | 5star | 26 | 3.600 | 4.380 | 0.917 |
| seasonal_naive | 4star | 26 | 3.215 | 4.033 | 0.729 |
| seasonal_naive | 3star | 26 | 3.735 | 4.721 | 0.954 |
| seasonal_naive | 2star | 26 | 2.477 | 3.128 | 0.417 |
| seasonal_naive | pooled | 104 | 3.257 | 4.108 | 0.754 |
| sarimax | 5star | 26 | 1.645 | 2.223 | 0.419 |
| sarimax | 4star | 26 | 2.363 | 3.757 | 0.536 |
| sarimax | 3star | 26 | 3.088 | 4.123 | 0.788 |
| sarimax | 2star | 26 | 2.356 | 3.290 | 0.396 |
| sarimax | pooled | 104 | 2.363 | 3.424 | 0.535 |
| lightgbm | 5star | 26 | 1.657 | 2.037 | 0.422 |
| lightgbm | 4star | 26 | 2.159 | 3.185 | 0.490 |
| lightgbm | 3star | 26 | 2.450 | 3.359 | 0.626 |
| lightgbm | 2star | 26 | 1.990 | 2.524 | 0.335 |
| lightgbm | pooled | 104 | 2.064 | 2.826 | 0.468 |

## Tests of equal accuracy (main run)

Diebold–Mariano with the Harvey–Leybourne–Newbold correction; loss: absolute error; two-sided; Holm over all ten tests (2 comparisons × 5 units). The loss difference is LightGBM minus baseline, in percentage points; negative means LightGBM has the lower error.

| comparison | unit | origins | mean_loss_difference | ci_low | ci_high | dm_hln | df | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm vs seasonal_naive | 5star | 26 | -1.943 | -3.000 | -0.886 | -3.785 | 25 | < 0.001 | 0.009 |
| lightgbm vs seasonal_naive | 4star | 26 | -1.056 | -2.132 | 0.019 | -2.023 | 25 | 0.054 | 0.377 |
| lightgbm vs seasonal_naive | 3star | 26 | -1.285 | -2.454 | -0.116 | -2.263 | 25 | 0.033 | 0.260 |
| lightgbm vs seasonal_naive | 2star | 26 | -0.486 | -1.505 | 0.532 | -0.984 | 25 | 0.334 | 1.000 |
| lightgbm vs seasonal_naive | pooled | 26 | -1.193 | -1.923 | -0.462 | -3.364 | 25 | 0.002 | 0.022 |
| lightgbm vs sarimax | 5star | 26 | 0.012 | -0.613 | 0.638 | 0.040 | 25 | 0.968 | 1.000 |
| lightgbm vs sarimax | 4star | 26 | -0.204 | -1.127 | 0.719 | -0.455 | 25 | 0.653 | 1.000 |
| lightgbm vs sarimax | 3star | 26 | -0.638 | -1.718 | 0.442 | -1.216 | 25 | 0.235 | 1.000 |
| lightgbm vs sarimax | 2star | 26 | -0.365 | -1.325 | 0.595 | -0.784 | 25 | 0.441 | 1.000 |
| lightgbm vs sarimax | pooled | 26 | -0.299 | -1.040 | 0.442 | -0.830 | 25 | 0.414 | 1.000 |

## H3

Rule: H3 is supported if LightGBM has the lower pooled test MAE against both baselines and both pooled Holm-adjusted p-values are below 0.05. Otherwise H3 is reported as not supported, with the MAE differences and their 95% CIs.

Pooled test MAE: LightGBM 2.06, SARIMAX 2.36, seasonal naive 3.26 percentage points. Against seasonal_naive: LightGBM MAE is lower; pooled Holm-adjusted p is below 0.05. Against sarimax: LightGBM MAE is lower; pooled Holm-adjusted p is not below 0.05. By the rule of the specification, H3 is not supported.

## Sensitivity runs

### F1_logit

SARIMAX and LightGBM fit on logit(occupancy_pct / 100); forecasts are back-transformed and all metrics are computed in percentage points

| model | unit | forecasts | mae | rmse | mase |
| --- | --- | --- | --- | --- | --- |
| seasonal_naive | pooled | 104 | 3.257 | 4.108 | 0.754 |
| sarimax | pooled | 104 | 2.248 | 3.278 | 0.513 |
| lightgbm | pooled | 104 | 1.977 | 2.658 | 0.447 |

| comparison | unit | origins | mean_loss_difference | ci_low | ci_high | dm_hln | df | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm vs seasonal_naive | pooled | 26 | -1.279 | -2.015 | -0.544 | -3.583 | 25 | 0.001 | 0.014 |
| lightgbm vs sarimax | pooled | 26 | -0.270 | -0.923 | 0.383 | -0.853 | 25 | 0.402 | 1.000 |

Pooled test MAE: LightGBM 1.98, SARIMAX 2.25, seasonal naive 3.26 percentage points. Against seasonal_naive: LightGBM MAE is lower; pooled Holm-adjusted p is below 0.05. Against sarimax: LightGBM MAE is lower; pooled Holm-adjusted p is not below 0.05. By the rule of the specification, H3 is not supported.

### F2_pandemic_included

pandemic-period target months kept in LightGBM training and observed in SARIMAX

| model | unit | forecasts | mae | rmse | mase |
| --- | --- | --- | --- | --- | --- |
| seasonal_naive | pooled | 104 | 3.257 | 4.108 | 0.754 |
| sarimax | pooled | 104 | 2.850 | 3.841 | 0.636 |
| lightgbm | pooled | 104 | 4.517 | 6.733 | 1.020 |

| comparison | unit | origins | mean_loss_difference | ci_low | ci_high | dm_hln | df | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm vs seasonal_naive | pooled | 26 | 1.260 | -0.755 | 3.275 | 1.288 | 25 | 0.209 | 1.000 |
| lightgbm vs sarimax | pooled | 26 | 1.667 | -0.089 | 3.424 | 1.955 | 25 | 0.062 | 0.495 |

Pooled test MAE: LightGBM 4.52, SARIMAX 2.85, seasonal naive 3.26 percentage points. Against seasonal_naive: LightGBM MAE is not lower; pooled Holm-adjusted p is not below 0.05. Against sarimax: LightGBM MAE is not lower; pooled Holm-adjusted p is not below 0.05. By the rule of the specification, H3 is not supported.

## SARIMAX fits at the test origins

| run | tier | sarimax_order | sarimax_seasonal_order | origins | not_converged |
| --- | --- | --- | --- | --- | --- |
| main | 5star | (1, 0, 1) | (0, 1, 1, 12) | 26 | 0 |
| main | 4star | (1, 1, 1) | (0, 1, 1, 12) | 26 | 0 |
| main | 3star | (1, 1, 1) | (0, 1, 1, 12) | 26 | 0 |
| main | 2star | (1, 1, 1) | (0, 1, 1, 12) | 26 | 0 |
| F1_logit | 5star | (1, 0, 1) | (0, 1, 1, 12) | 26 | 0 |
| F1_logit | 4star | (0, 1, 1) | (0, 1, 1, 12) | 26 | 0 |
| F1_logit | 3star | (0, 1, 1) | (0, 1, 1, 12) | 26 | 0 |
| F1_logit | 2star | (1, 1, 1) | (0, 1, 1, 12) | 26 | 0 |
| F2_pandemic_included | 5star | (1, 1, 1) | (1, 0, 1, 12) | 26 | 0 |
| F2_pandemic_included | 4star | (1, 1, 1) | (0, 1, 1, 12) | 26 | 0 |
| F2_pandemic_included | 3star | (1, 1, 1) | (1, 0, 1, 12) | 26 | 0 |
| F2_pandemic_included | 2star | (1, 0, 0) | (0, 1, 1, 12) | 26 | 0 |

## SHAP: mean absolute contribution to the forecast (main run, percentage points)

TreeSHAP, tree-path-dependent; background: the training window of that origin only; rows: that origin's four test forecasts. Largest gap between base value plus SHAP values and the forecast: 2.84e-13.

| feature | all_classes | 5star | 4star | 3star | 2star |
| --- | --- | --- | --- | --- | --- |
| occupancy_pct_lag1 | 1.821 | 2.813 | 1.327 | 1.294 | 1.851 |
| log_visitors_overnight_lag12 | 1.232 | 1.147 | 1.279 | 1.284 | 1.216 |
| log_rooms_lag1 | 1.225 | 1.626 | 0.994 | 1.025 | 1.254 |
| occupancy_pct_lag3 | 0.698 | 1.141 | 0.369 | 0.392 | 0.888 |
| cn_holiday_days_at_origin | 0.635 | 0.587 | 0.640 | 0.641 | 0.674 |
| occupancy_pct_lag12 | 0.575 | 0.790 | 0.379 | 0.401 | 0.729 |
| international_share_pct_lag1 | 0.485 | 0.411 | 0.550 | 0.572 | 0.408 |
| log_rooms_lag2 | 0.372 | 0.825 | 0.083 | 0.052 | 0.529 |
| same_day_share_pct_lag12 | 0.345 | 0.300 | 0.368 | 0.371 | 0.341 |
| log_visitors_overnight_lag1 | 0.304 | 0.231 | 0.340 | 0.341 | 0.305 |
| month | 0.299 | 0.201 | 0.372 | 0.361 | 0.264 |
| package_tour_share_pct_lag3 | 0.291 | 0.246 | 0.313 | 0.299 | 0.307 |

## Choices the specification left open

- SARIMAX is statsmodels SARIMAX with the exogenous columns as regressors and SARIMA errors, default starting values, L-BFGS with at most 200 iterations, and stationarity and invertibility enforced. A fit counts as converged when the optimiser says so and the forecast is finite.
- Candidates are ordered with the non-seasonal order outside and the seasonal order inside, both as listed in the specification.
- At a test origin the forecast of the chosen order is used as it comes; fits that did not converge are counted. If a fit fails or gives no finite forecast, the fallback order is used for that origin and counted.
- LightGBM gets force_col_wise = true, which its deterministic mode asks for. Every other parameter not in the specification is the library default.
- F1 and F2 choose their own orders and hyperparameters on the validation origins, under their own setting. F1 back-transforms with the inverse logit, without a bias correction; the lagged occupancy features stay in percent.
- Pooled MASE is the mean of the scaled absolute errors, each divided by the scale of its star class.
- The 95% CI of a loss difference uses the same corrected variance and t distribution as the test.
- SHAP values are LightGBM's built-in TreeSHAP (tree-path-dependent), for the main run.
- Seasonal-naive validation forecasts for 2023-07 to 2023-12 use pandemic months as their lag. No test forecast does.

## Figures

- `outputs/figures/fig_forecast_test_window.png`: occupancy and the three forecasts over the test window.
- `outputs/figures/fig_shap_importance_by_class.png`: mean absolute SHAP value by star class.
