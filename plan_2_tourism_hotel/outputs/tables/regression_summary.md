# Panel regression for H1 and H2 (Plan 2, Phase A step 4)

Ex-post description of the same-month association between visitor variables and occupancy. The coefficients are conditional associations. They are not causal effects, and they are not forecasts.

Frozen panel SHA-256: `9c9547d653c52a168f5b7d8b632458b7d62eece2c0e66c46fcbd8934de479729`  
Analysis specification SHA-256: `347c14d671f53f9ba3fc472cb1eba972b4519c146a920144d9c356a71ca1a4d5` (version 1.0.0)  
Deviations from the specification: 0

## Model

`occupancy_pct[i,t] = a[i] + a_post[i]·post[t] + month-of-year[t] + (b[i] + b_post[i]·post[t])·log_visitors_overnight[t] + (c + c_post·post[t])·same_day_share_pct[t] + controls + e[i,t]`

- Sample: main sample: pre-pandemic and post-pandemic months, all four classes: 660 rows, 165 months.
- 34 coefficients. R² = 0.755.
- Standard errors: Driscoll–Kraay, Bartlett kernel, bandwidth 4 (floor(4 × (T / 100)^(2/9)), T = months in the sample); lags counted in calendar months, so no autocovariance is taken across the gap between December 2019 and January 2023.
- Inference: two-sided, alpha 0.05, normal reference; 95% CI for every coefficient.

## Confirmatory tests

- H1 null: `(b[5star] + b_post[5star]) − (b[3star] + b_post[3star]) = 0`. Unit: percentage points of occupancy per unit of log overnight visitors.
- H2 null: `c_post = 0`. Unit: percentage points of occupancy per percentage point of same-day share.
- Multiplicity: Holm over the two tests above.

| hypothesis | estimate | se | z | p_value | ci_low | ci_high | p_holm | reject_at_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | 18.334 | 6.940 | 2.642 | 0.008 | 4.732 | 31.935 | 0.016 | True |
| H2 | 0.038 | 0.152 | 0.248 | 0.804 | -0.260 | 0.336 | 0.804 | False |

- H1: estimate 18.33 (SE 6.94, 95% CI 4.73 to 31.93), p = 0.008, Holm-adjusted p = 0.016. The null is rejected at the 0.05 level.
- H2: estimate 0.04 (SE 0.15, 95% CI -0.26 to 0.34), p = 0.804, Holm-adjusted p = 0.804. The null is not rejected at the 0.05 level. This is not evidence that the difference is zero; the interval gives the differences the data are compatible with.

## Slopes before and after the pandemic

A slope b on log overnight visitors means: a month with 10% more overnight visitors has b × ln(1.1) ≈ 0.095 × b points higher occupancy, the other regressors being equal.

| regressor | tier | quantity | estimate | se | z | p_value | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log_visitors_overnight | 5star | pre_pandemic | 36.125 | 2.935 | 12.307 | < 0.001 | 30.371 | 41.878 |
| log_visitors_overnight | 5star | post_pandemic | 21.228 | 3.543 | 5.992 | < 0.001 | 14.284 | 28.172 |
| log_visitors_overnight | 5star | change | -14.896 | 4.812 | -3.096 | 0.002 | -24.327 | -5.465 |
| log_visitors_overnight | 4star | pre_pandemic | 27.311 | 3.979 | 6.863 | < 0.001 | 19.511 | 35.110 |
| log_visitors_overnight | 4star | post_pandemic | 15.024 | 4.055 | 3.705 | < 0.001 | 7.077 | 22.970 |
| log_visitors_overnight | 4star | change | -12.287 | 5.491 | -2.238 | 0.025 | -23.049 | -1.526 |
| log_visitors_overnight | 3star | pre_pandemic | 30.685 | 4.522 | 6.786 | < 0.001 | 21.822 | 39.548 |
| log_visitors_overnight | 3star | post_pandemic | 2.895 | 5.984 | 0.484 | 0.629 | -8.834 | 14.624 |
| log_visitors_overnight | 3star | change | -27.790 | 7.065 | -3.934 | < 0.001 | -41.637 | -13.943 |
| log_visitors_overnight | 2star | pre_pandemic | 11.121 | 4.421 | 2.515 | 0.012 | 2.456 | 19.786 |
| log_visitors_overnight | 2star | post_pandemic | 6.047 | 5.097 | 1.187 | 0.235 | -3.942 | 16.036 |
| log_visitors_overnight | 2star | change | -5.073 | 6.797 | -0.746 | 0.455 | -18.395 | 8.248 |
| same_day_share_pct | all | pre_pandemic | 0.261 | 0.110 | 2.367 | 0.018 | 0.045 | 0.477 |
| same_day_share_pct | all | post_pandemic | 0.298 | 0.091 | 3.269 | 0.001 | 0.119 | 0.477 |
| same_day_share_pct | all | change | 0.038 | 0.152 | 0.248 | 0.804 | -0.260 | 0.336 |

## Secondary tests

| test | chi2 | df | p_value |
| --- | --- | --- | --- |
| equal_post_pandemic_slopes | 15.341 | 3 | 0.002 |
| chow_post_interactions_on_focal_regressors | 36.415 | 5 | < 0.001 |

## Sensitivity runs

S2 is on the logit scale, so its estimates are not comparable in size with the other rows; signs and tests are. AIC and BIC are not compared across scales.

| run | n | months | parameters | bandwidth | r_squared | H1_estimate | H1_se | H1_ci_low | H1_ci_high | H1_p_value | H1_p_holm | H2_estimate | H2_se | H2_ci_low | H2_ci_high | H2_p_value | H2_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | 660 | 165 | 34 | 4 | 0.755 | 18.334 | 6.940 | 4.732 | 31.935 | 0.008 | 0.016 | 0.038 | 0.152 | -0.260 | 0.336 | 0.804 | 0.804 |
| S1_pandemic_included | 804 | 201 | 43 | 4 | 0.881 | 18.417 | 6.933 | 4.828 | 32.005 | 0.008 | 0.016 | 0.294 | 0.195 | -0.090 | 0.677 | 0.133 | 0.133 |
| S2_logit_outcome | 660 | 165 | 34 | 4 | 0.769 | 1.872 | 0.384 | 1.118 | 2.625 | < 0.001 | < 0.001 | 0.008 | 0.013 | -0.018 | 0.034 | 0.534 | 0.534 |
| S3_bandwidth_2 | 660 | 165 | 34 | 2 | 0.755 | 18.334 | 6.130 | 6.319 | 30.349 | 0.003 | 0.006 | 0.038 | 0.153 | -0.261 | 0.337 | 0.805 | 0.805 |
| S3_bandwidth_8 | 660 | 165 | 34 | 8 | 0.755 | 18.334 | 7.554 | 3.527 | 33.140 | 0.015 | 0.030 | 0.038 | 0.155 | -0.266 | 0.341 | 0.807 | 0.807 |
| S4_trends | 660 | 165 | 38 | 4 | 0.818 | 15.131 | 6.678 | 2.043 | 28.219 | 0.023 | 0.023 | 0.390 | 0.152 | 0.093 | 0.687 | 0.010 | 0.020 |
| S5_fully_interacted | 660 | 165 | 50 | 4 | 0.781 | 16.861 | 7.773 | 1.626 | 32.096 | 0.030 | 0.060 | -0.210 | 0.165 | -0.534 | 0.114 | 0.203 | 0.203 |

- H1: the null is rejected after Holm in 6 of 7 runs; not in S5_fully_interacted. Rejected in main, S1_pandemic_included, S2_logit_outcome, S3_bandwidth_2, S3_bandwidth_8, S4_trends.
- H2: the null is rejected after Holm in 1 of 7 runs; not in main, S1_pandemic_included, S2_logit_outcome, S3_bandwidth_2, S3_bandwidth_8, S5_fully_interacted. Rejected in S4_trends.

## Diagnostics (reported only)

Reported only; the model, the standard errors, and the tests are not changed on the basis of a diagnostic p-value.

| test | tier | block | lag | n | statistic | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| ljung_box | 5star | pre_pandemic | 12 | 120 | 96.780 | < 0.001 |
| ljung_box | 5star | pre_pandemic | 24 | 120 | 158.956 | < 0.001 |
| ljung_box | 5star | post_pandemic | 12 | 45 | 33.140 | < 0.001 |
| ljung_box | 5star | post_pandemic | 24 | 45 | 50.597 | 0.001 |
| ljung_box | 4star | pre_pandemic | 12 | 120 | 110.459 | < 0.001 |
| ljung_box | 4star | pre_pandemic | 24 | 120 | 134.908 | < 0.001 |
| ljung_box | 4star | post_pandemic | 12 | 45 | 30.149 | 0.003 |
| ljung_box | 4star | post_pandemic | 24 | 45 | 41.531 | 0.015 |
| ljung_box | 3star | pre_pandemic | 12 | 120 | 322.000 | < 0.001 |
| ljung_box | 3star | pre_pandemic | 24 | 120 | 358.607 | < 0.001 |
| ljung_box | 3star | post_pandemic | 12 | 45 | 22.613 | 0.031 |
| ljung_box | 3star | post_pandemic | 24 | 45 | 39.680 | 0.023 |
| ljung_box | 2star | pre_pandemic | 12 | 120 | 126.643 | < 0.001 |
| ljung_box | 2star | pre_pandemic | 24 | 120 | 133.236 | < 0.001 |
| ljung_box | 2star | post_pandemic | 12 | 45 | 15.769 | 0.202 |
| ljung_box | 2star | post_pandemic | 24 | 45 | 25.486 | 0.380 |
| breusch_pagan | all | main_sample | 0 | 660 | 125.004 | < 0.001 |

Ljung–Box has p < 0.05 in 14 of 16 cells; Breusch–Pagan has p < 0.001. The Driscoll–Kraay standard errors allow for heteroskedasticity, for correlation between star classes in a month, and for serial correlation up to the bandwidth; the S3 rows show the tests at bandwidth 2 and 8. The normal reference rests on 165 months and may be optimistic.

## All coefficients of the main model

| term | estimate | se | z | p_value | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| a[5star] | -416.386 | 43.335 | -9.609 | < 0.001 | -501.320 | -331.451 |
| a[4star] | -294.568 | 55.686 | -5.290 | < 0.001 | -403.710 | -185.427 |
| a[3star] | -345.150 | 61.081 | -5.651 | < 0.001 | -464.866 | -225.434 |
| a[2star] | -86.377 | 67.818 | -1.274 | 0.203 | -219.298 | 46.544 |
| a_post[5star] | 212.731 | 70.348 | 3.024 | 0.002 | 74.851 | 350.610 |
| a_post[4star] | 169.248 | 80.325 | 2.107 | 0.035 | 11.814 | 326.682 |
| a_post[3star] | 390.239 | 102.189 | 3.819 | < 0.001 | 189.953 | 590.525 |
| a_post[2star] | 87.953 | 98.099 | 0.897 | 0.370 | -104.318 | 280.224 |
| month[2] | 3.207 | 1.249 | 2.568 | 0.010 | 0.760 | 5.655 |
| month[3] | -0.300 | 1.088 | -0.276 | 0.782 | -2.432 | 1.831 |
| month[4] | 1.509 | 1.000 | 1.509 | 0.131 | -0.451 | 3.468 |
| month[5] | -0.068 | 0.986 | -0.069 | 0.945 | -2.000 | 1.864 |
| month[6] | 0.614 | 1.074 | 0.571 | 0.568 | -1.492 | 2.719 |
| month[7] | 2.825 | 1.237 | 2.284 | 0.022 | 0.401 | 5.249 |
| month[8] | 3.471 | 1.361 | 2.551 | 0.011 | 0.804 | 6.138 |
| month[9] | 0.228 | 1.291 | 0.177 | 0.860 | -2.302 | 2.758 |
| month[10] | 2.100 | 1.083 | 1.939 | 0.053 | -0.023 | 4.223 |
| month[11] | 3.153 | 0.937 | 3.364 | < 0.001 | 1.316 | 4.989 |
| month[12] | 1.237 | 1.026 | 1.205 | 0.228 | -0.775 | 3.248 |
| b[5star] | 36.125 | 2.935 | 12.307 | < 0.001 | 30.371 | 41.878 |
| b[4star] | 27.311 | 3.979 | 6.863 | < 0.001 | 19.511 | 35.110 |
| b[3star] | 30.685 | 4.522 | 6.786 | < 0.001 | 21.822 | 39.548 |
| b[2star] | 11.121 | 4.421 | 2.515 | 0.012 | 2.456 | 19.786 |
| b_post[5star] | -14.896 | 4.812 | -3.096 | 0.002 | -24.327 | -5.465 |
| b_post[4star] | -12.287 | 5.491 | -2.238 | 0.025 | -23.049 | -1.526 |
| b_post[3star] | -27.790 | 7.065 | -3.934 | < 0.001 | -41.637 | -13.943 |
| b_post[2star] | -5.073 | 6.797 | -0.746 | 0.455 | -18.395 | 8.248 |
| c | 0.261 | 0.110 | 2.367 | 0.018 | 0.045 | 0.477 |
| c_post | 0.038 | 0.152 | 0.248 | 0.804 | -0.260 | 0.336 |
| international_share_pct | 0.924 | 0.344 | 2.689 | 0.007 | 0.251 | 1.597 |
| package_tour_share_pct | -0.003 | 0.072 | -0.037 | 0.971 | -0.143 | 0.138 |
| log_rooms | -2.863 | 1.569 | -1.825 | 0.068 | -5.939 | 0.212 |
| cn_holiday_days | -0.310 | 0.141 | -2.199 | 0.028 | -0.586 | -0.034 |
| typhoon_days | -0.421 | 0.378 | -1.114 | 0.265 | -1.161 | 0.319 |

## Choices the specification left open

Fixed before the first fit on the real outcome:

- Bandwidth is the highest lag with a non-zero Bartlett weight; lag j has weight 1 − j / (bandwidth + 1).
- The covariance is scaled by n / (n − k).
- Joint tests are Wald tests with a chi-square reference.
- S1 uses the bandwidth rule with its own number of months.
- S4 trends are linear in calendar time, one per star class.
- Ljung–Box is computed per star class and per contiguous block (pre-pandemic, post-pandemic), because the residual series has a 36-month gap.
- Breusch–Pagan is the studentised (Koenker) LM test on the regressors of the model.

## Figure

`outputs/figures/fig_regression_slopes_by_class.png`: slopes before and after the pandemic with 95% confidence intervals.
