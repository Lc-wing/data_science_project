# Exploratory data analysis (Plan 2, Phase A step 3)

Exploratory description only. Nothing here is a test of H1, H2, or H3, and nothing here is an input to the regression or the forecasts. The period boundaries are those of the plan; they were not taken from the change points.

Frozen panel SHA-256: `9c9547d653c52a168f5b7d8b632458b7d62eece2c0e66c46fcbd8934de479729`  
Analysis specification SHA-256: `347c14d671f53f9ba3fc472cb1eba972b4519c146a920144d9c356a71ca1a4d5`

## Settings

- STL on occupancy: period 12, robust, one fit per star class and block (pre-pandemic, post-pandemic); seasonal pattern periodic (constant inside a block). Pandemic months are not decomposed.
- PELT input: share in percent minus its STL seasonal component (one fit over all months, period 12, robust, seasonal window 7). Cost l2 (piecewise-constant mean), segments of at least 12 months, penalty = multiplier × 2 × noise_sd² × ln(n), noise_sd = 1.4826 × MAD of the 12-month differences / √2. Main multiplier 1.0.
- The specification names STL (period 12, robust) on occupancy and PELT on the three shares. The months of each STL fit, the seasonal window, the seasonal adjustment before PELT, the cost, the minimum segment length, and the penalty are choices of this step.

### Changed after the first run

- STL on occupancy was first fitted once per star class over all 201 months with seasonal window 7. That fit pushed the pandemic collapse into the seasonal component and the remainder of the post-pandemic months (see eda_stl_full_series_check.csv). It was replaced by one fit per block with a periodic seasonal pattern.
- The PELT settings were not changed. A run on the shares without seasonal adjustment was added to the sensitivity table.

## Visitor mix by period (means of monthly values)

| period_label | months | mean_same_day_share_pct | mean_international_share_pct | mean_package_tour_share_pct | mean_visitors_total | mean_visitors_overnight |
| --- | --- | --- | --- | --- | --- | --- |
| pre_pandemic | 120 | 51.35 | 9.82 | 28.05 | 2,594,881.25 | 1,263,786.22 |
| pandemic | 35 | 54.00 | 0.50 | 0.47 | 470,076.14 | 219,520.91 |
| post_pandemic | 45 | 55.34 | 6.26 | 5.36 | 3,008,717.44 | 1,315,840.67 |

## Change points of the visitor-mix shares (main run)

A change point is the first month of a new segment. `mean_share_pct` is the mean of the published share over the segment.

| series | segment | first_month | last_month | months | mean_share_pct | starts_in_period |
| --- | --- | --- | --- | --- | --- | --- |
| same_day_share_pct | 1 | 2010-01 | 2016-05 | 77 | 52.68 | pre_pandemic |
| same_day_share_pct | 2 | 2016-06 | 2018-10 | 29 | 47.27 | pre_pandemic |
| same_day_share_pct | 3 | 2018-11 | 2021-07 | 33 | 51.49 | pre_pandemic |
| same_day_share_pct | 4 | 2021-08 | 2022-07 | 12 | 62.42 | pandemic |
| same_day_share_pct | 5 | 2022-08 | 2023-07 | 12 | 46.90 | pandemic |
| same_day_share_pct | 6 | 2023-08 | 2024-11 | 16 | 52.91 | post_pandemic |
| same_day_share_pct | 7 | 2024-12 | 2026-09 | 22 | 59.58 | post_pandemic |
| international_share_pct | 1 | 2010-01 | 2011-02 | 14 | 11.78 | pre_pandemic |
| international_share_pct | 2 | 2011-03 | 2013-04 | 26 | 10.72 | pre_pandemic |
| international_share_pct | 3 | 2013-05 | 2018-06 | 62 | 9.54 | pre_pandemic |
| international_share_pct | 4 | 2018-07 | 2020-02 | 20 | 7.95 | pre_pandemic |
| international_share_pct | 5 | 2020-03 | 2023-01 | 35 | 0.27 | pandemic |
| international_share_pct | 6 | 2023-02 | 2024-01 | 12 | 5.33 | post_pandemic |
| international_share_pct | 7 | 2024-02 | 2026-09 | 32 | 6.74 | post_pandemic |
| package_tour_share_pct | 1 | 2010-01 | 2011-06 | 18 | 23.51 | pre_pandemic |
| package_tour_share_pct | 2 | 2011-07 | 2012-06 | 12 | 29.64 | pre_pandemic |
| package_tour_share_pct | 3 | 2012-07 | 2015-07 | 37 | 34.26 | pre_pandemic |
| package_tour_share_pct | 4 | 2015-08 | 2018-12 | 41 | 26.03 | pre_pandemic |
| package_tour_share_pct | 5 | 2019-01 | 2019-12 | 12 | 21.01 | pre_pandemic |
| package_tour_share_pct | 6 | 2020-01 | 2023-02 | 38 | 0.72 | transition_2020_01 |
| package_tour_share_pct | 7 | 2023-03 | 2026-09 | 43 | 5.55 | post_pandemic |

### Sensitivity to the penalty and to the seasonal adjustment

| series | input | penalty_multiplier | n_change_points | change_points |
| --- | --- | --- | --- | --- |
| same_day_share_pct | seasonally_adjusted | 0.50 | 6 | 2016-06;2018-11;2021-08;2022-08;2023-08;2024-12 |
| same_day_share_pct | seasonally_adjusted | 1.00 | 6 | 2016-06;2018-11;2021-08;2022-08;2023-08;2024-12 |
| same_day_share_pct | seasonally_adjusted | 2.00 | 4 | 2016-03;2021-08;2022-08;2024-07 |
| same_day_share_pct | seasonally_adjusted | 4.00 | 3 | 2021-08;2022-08;2024-07 |
| same_day_share_pct | unadjusted | 1.00 | 6 | 2016-06;2018-10;2021-08;2022-08;2023-08;2025-01 |
| international_share_pct | seasonally_adjusted | 0.50 | 9 | 2011-03;2012-09;2014-03;2015-12;2018-02;2019-02;2020-03;2023-02;2024-02 |
| international_share_pct | seasonally_adjusted | 1.00 | 6 | 2011-03;2013-05;2018-07;2020-03;2023-02;2024-02 |
| international_share_pct | seasonally_adjusted | 2.00 | 5 | 2012-09;2018-07;2020-03;2023-02;2024-02 |
| international_share_pct | seasonally_adjusted | 4.00 | 4 | 2012-09;2018-07;2020-03;2023-04 |
| international_share_pct | unadjusted | 1.00 | 6 | 2011-01;2013-02;2018-07;2020-03;2023-03;2024-03 |
| package_tour_share_pct | seasonally_adjusted | 0.50 | 7 | 2011-07;2012-07;2014-06;2015-08;2019-01;2020-01;2023-03 |
| package_tour_share_pct | seasonally_adjusted | 1.00 | 6 | 2011-07;2012-07;2015-08;2019-01;2020-01;2023-03 |
| package_tour_share_pct | seasonally_adjusted | 2.00 | 4 | 2011-09;2015-11;2020-01;2023-03 |
| package_tour_share_pct | seasonally_adjusted | 4.00 | 4 | 2011-09;2015-11;2020-01;2023-03 |
| package_tour_share_pct | unadjusted | 1.00 | 6 | 2011-06;2012-07;2015-12;2019-01;2020-01;2023-03 |

## STL decomposition of occupancy

Strength = 1 − Var(remainder) / Var(component + remainder). The post-pandemic block has 45 months, under four yearly cycles, so its seasonal profile rests on three or four values per calendar month.

| tier | block | months | trend_strength | seasonal_strength | seasonal_peak_month | seasonal_trough_month | seasonal_range_pp | remainder_sd_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5star | pre_pandemic | 120 | 0.815 | 0.612 | 8 | 9 | 7.737 | 1.973 |
| 5star | post_pandemic | 45 | 0.797 | 0.254 | 8 | 6 | 5.764 | 2.661 |
| 4star | pre_pandemic | 120 | 0.605 | 0.519 | 11 | 1 | 7.735 | 2.651 |
| 4star | post_pandemic | 45 | 0.686 | 0.518 | 8 | 9 | 10.619 | 2.475 |
| 3star | pre_pandemic | 120 | 0.678 | 0.331 | 8 | 1 | 7.917 | 2.670 |
| 3star | post_pandemic | 45 | 0.370 | 0.382 | 8 | 9 | 9.768 | 2.685 |
| 2star | pre_pandemic | 120 | 0.506 | 0.468 | 11 | 1 | 12.736 | 3.820 |
| 2star | post_pandemic | 45 | 0.524 | 0.634 | 8 | 9 | 9.659 | 2.137 |

### One fit over all months (first run, kept as a check)

| tier | max_abs_seasonal_pre_pandemic_pp | max_abs_seasonal_post_pandemic_pp | month_of_post_pandemic_max | occupancy_pct_that_month | seasonal_that_month_pp | remainder_that_month_pp |
| --- | --- | --- | --- | --- | --- | --- |
| 5star | 6.57 | 34.81 | 2023-08 | 90.60 | -34.81 | 50.50 |
| 4star | 9.68 | 9.44 | 2023-03 | 74.10 | -9.44 | 10.36 |
| 3star | 8.08 | 13.72 | 2023-09 | 73.60 | -13.72 | 10.02 |
| 2star | 9.43 | 10.97 | 2023-09 | 76.20 | -10.97 | 1.90 |

## Occupancy by star class and period

`pearson_r_log_overnight` describes the scatter plot. No test is attached to it; it is a same-month correlation of two series that share a seasonal pattern.

| tier | period_label | months | occupancy_mean | occupancy_sd | occupancy_min | occupancy_max | pearson_r_log_overnight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5star | pre_pandemic | 120 | 85.65 | 5.30 | 75.30 | 96.20 | 0.84 |
| 5star | pandemic | 35 | 33.94 | 16.49 | 5.00 | 63.60 | 0.79 |
| 5star | post_pandemic | 45 | 88.98 | 6.03 | 70.50 | 96.50 | 0.83 |
| 4star | pre_pandemic | 120 | 86.92 | 5.09 | 73.20 | 94.60 | 0.61 |
| 4star | pandemic | 35 | 42.83 | 15.04 | 16.80 | 67.50 | 0.52 |
| 4star | post_pandemic | 45 | 83.00 | 5.10 | 70.00 | 94.40 | 0.81 |
| 3star | pre_pandemic | 120 | 85.58 | 5.12 | 72.80 | 97.20 | 0.66 |
| 3star | pandemic | 35 | 45.87 | 16.03 | 18.40 | 81.30 | 0.41 |
| 3star | post_pandemic | 45 | 83.56 | 3.96 | 73.60 | 91.10 | 0.60 |
| 2star | pre_pandemic | 120 | 73.69 | 6.39 | 55.80 | 85.00 | 0.14 |
| 2star | pandemic | 35 | 40.37 | 15.41 | 11.70 | 86.30 | -0.14 |
| 2star | post_pandemic | 45 | 87.97 | 4.18 | 76.20 | 93.60 | 0.66 |

## Figures

- `outputs/figures/fig1_visitor_mix_and_occupancy.png`: visitor-mix shares with change points, and occupancy by star class (Figure 1).
- `outputs/figures/eda_stl_occupancy.png`: trend, seasonal profile, and remainder.
- `outputs/figures/eda_scatter_occupancy_overnight.png`: occupancy against log overnight visitors by star class and period.
