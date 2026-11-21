"""Pre-registered analysis specification for Plan 2 Phase A.

Written to ``data/frozen/analysis_spec.json`` and hashed in the freeze manifest before
any model is fit on the panel. Steps 3 to 6 read their settings from that file and
record its SHA-256 next to their results.
"""

from __future__ import annotations

from typing import Any

from macao_tourism.config import (
    FORECAST_LAGS,
    MAIN_SAMPLE_PERIODS,
    PANEL_END,
    PANEL_START,
    PERIODS,
    RANDOM_SEED,
    TEST_ORIGIN_FIRST,
    TEST_ORIGIN_LAST,
    TIERS,
    TYPHOON_DAY_MIN_SIGNAL,
    VALIDATION_TARGET_WINDOWS,
)

SPEC_VERSION = "1.0.0"

LAGGED_VARIABLES: tuple[str, ...] = (
    "occupancy_pct",
    "log_visitors_overnight",
    "same_day_share_pct",
    "international_share_pct",
    "package_tour_share_pct",
    "log_rooms",
    "typhoon_days",
)
KNOWN_CALENDAR_FEATURES: tuple[str, ...] = (
    "month",
    "days_in_month",
    "cn_holiday_days_at_origin",
    "cn_golden_week_days_at_origin",
)
REGRESSION_CONTROLS: tuple[str, ...] = (
    "international_share_pct",
    "package_tour_share_pct",
    "log_rooms",
    "cn_holiday_days",
    "typhoon_days",
)
SARIMAX_ORDERS: tuple[tuple[int, int, int], ...] = ((1, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
SARIMAX_SEASONAL_ORDERS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 1, 12),
    (1, 0, 0, 12),
    (1, 0, 1, 12),
)
LIGHTGBM_GRID: dict[str, tuple[int, ...]] = {
    "num_leaves": (4, 8),
    "min_data_in_leaf": (10, 20),
    "n_estimators": (100, 200, 400),
}
LIGHTGBM_FIXED: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "seed": RANDOM_SEED,
    "deterministic": True,
    "num_threads": 1,
    "verbosity": -1,
}


def forecast_feature_names() -> list[str]:
    lagged = [f"{name}_lag{lag}" for name in LAGGED_VARIABLES for lag in FORECAST_LAGS]
    return ["tier", *KNOWN_CALENDAR_FEATURES, *lagged]


def build_analysis_spec() -> dict[str, Any]:
    return {
        "spec_version": SPEC_VERSION,
        "plan": "plans/plan_2_macao_tourism_hotel_multivariate_dl.md",
        "constraints": "plans/00_review_report.md (C1, C3, C6, C7, C11, C15 and the Plan 2 items)",
        "status": (
            "Fixed before any regression, forecast, decomposition, or comparison between "
            "periods or star classes was run on the panel. What had been looked at: the "
            "Day-1 audit (coverage, value ranges, supply changes, latest-month values)."
        ),
        "deviation_rule": (
            "Any departure from this file is listed, with its reason, in the report and in "
            "the output of the step where it happens. The file is not edited after results "
            "are seen; a correction gets a new spec_version and a new freeze."
        ),
        "scope": {
            "claims": "descriptive, associational, and predictive; no causal claim",
            "wording": (
                "'associated with' and 'conditional association'; not 'effect', 'impact', "
                "'driver', or 'elasticity'. SHAP values describe model behaviour only."
            ),
            "phase_b": "not part of this specification",
        },
        "data": {
            "unit": "star class × month",
            "tiers": list(TIERS),
            "window": [PANEL_START, PANEL_END],
            "periods": {label: list(span) for label, span in PERIODS.items()},
            "main_sample": list(MAIN_SAMPLE_PERIODS),
            "left_out_of_main_sample": ["transition_2020_01", "pandemic"],
            "outcome": "occupancy_pct (average occupancy rate of guest rooms, percent)",
            "variables": {
                "log_visitors_overnight": "natural log of overnight visitor arrivals",
                "same_day_share_pct": "100 × same-day visitors / all visitors",
                "international_share_pct": (
                    "100 × (1 − (Chinese mainland + Hong Kong + Taiwan) / all visitors), "
                    "by place of issue of the travel document"
                ),
                "package_tour_share_pct": "100 × visitors on package tours / all visitors",
                "log_rooms": "natural log of guest rooms of the star class at month end",
                "cn_holiday_days": (
                    "days of the month inside a mainland public-holiday period (State "
                    "Council notices; declared days off plus adjoining free weekend days)"
                ),
                "cn_golden_week_days": (
                    "the same, counting Spring Festival and National Day periods only"
                ),
                "cn_holiday_days_at_origin, cn_golden_week_days_at_origin": (
                    "the two counts above as scheduled by the notices published up to the "
                    "end of the month before; these are the calendar features of the "
                    "forecasts, the final counts are used in the ex-post regression"
                ),
                "typhoon_days": (
                    f"days of the month with tropical cyclone signal No. "
                    f"{TYPHOON_DAY_MIN_SIGNAL} or higher in force"
                ),
            },
            "star_classes": (
                "DSEC's four classes as published. Establishments move in and out of a "
                "class over time; no class is merged, dropped, or re-labelled, and rooms "
                "per class enter as the supply variable."
            ),
            "revisions": "values are those DSEC published on the download date in the manifest",
        },
        "eda": {
            "role": "exploratory description only (Figure 1)",
            "stl": {"series": "occupancy_pct per star class", "period": 12, "robust": True},
            "change_points": {
                "method": "PELT",
                "series": [
                    "same_day_share_pct",
                    "international_share_pct",
                    "package_tour_share_pct",
                ],
                "use": (
                    "drawn on the timeline; never an input to a regression or a forecast. "
                    "The period boundaries above come from the plan, not from PELT."
                ),
            },
            "scatter": "occupancy_pct against log_visitors_overnight, by star class and period",
        },
        "regression_h1_h2": {
            "role": "ex-post description of the same-month association; not a forecast",
            "sample": "main sample: pre-pandemic and post-pandemic months, all four classes",
            "model": (
                "occupancy_pct[i,t] = a[i] + a_post[i]·post[t] + month-of-year[t] "
                "+ (b[i] + b_post[i]·post[t])·log_visitors_overnight[t] "
                "+ (c + c_post·post[t])·same_day_share_pct[t] + controls + e[i,t]"
            ),
            "terms": {
                "a[i], a_post[i]": "star-class intercepts and their post-pandemic shifts",
                "month-of-year": "eleven calendar-month indicators, common to all classes",
                "b[i], b_post[i]": "class-specific slopes on log overnight visitors",
                "c, c_post": "common slope on the same-day share",
                "controls": list(REGRESSION_CONTROLS),
            },
            "controls_note": "controls have one coefficient each, common to classes and periods",
            "estimator": "least squares with indicator variables",
            "standard_errors": {
                "type": "Driscoll–Kraay",
                "kernel": "Bartlett",
                "bandwidth_rule": "floor(4 × (T / 100)^(2/9)), T = months in the sample",
                "lag_counting": (
                    "in calendar months, so no autocovariance is taken across the gap "
                    "between December 2019 and January 2023"
                ),
            },
            "inference": "two-sided, alpha 0.05, normal reference; 95% CI for every coefficient",
            "confirmatory_tests": {
                "H1": {
                    "statement": (
                        "after the pandemic, the association between log overnight "
                        "visitors and occupancy differs between 5-star and 3-star hotels"
                    ),
                    "null": "(b[5star] + b_post[5star]) − (b[3star] + b_post[3star]) = 0",
                },
                "H2": {
                    "statement": (
                        "the association between the same-day share and occupancy differs "
                        "between the post-pandemic and the pre-pandemic period"
                    ),
                    "null": "c_post = 0",
                },
                "multiplicity": "Holm over the two tests above",
            },
            "secondary_tests": [
                "equality of the four post-pandemic slopes b[i] + b_post[i] (3 df)",
                "Chow-type test: the five post interactions on the two focal regressors "
                "are jointly zero (5 df)",
            ],
            "sensitivity_analyses": {
                "S1_pandemic_included": (
                    "add 2020-01 to 2022-12 with their own period indicator, interacted like post"
                ),
                "S2_logit_outcome": (
                    "outcome logit(occupancy_pct / 100); signs and tests are compared, "
                    "AIC/BIC are not compared across the two scales"
                ),
                "S3_bandwidth": "Driscoll–Kraay bandwidth 2 and 8",
                "S4_trends": "class-specific linear time trends added",
                "S5_fully_interacted": "controls and month indicators also interacted with post",
            },
            "diagnostics": {
                "reported": [
                    "Ljung–Box on residuals per star class, lags 12 and 24",
                    "Breusch–Pagan test for heteroskedasticity",
                ],
                "rule": (
                    "reported only; the model, the standard errors, and the tests are not "
                    "changed on the basis of a diagnostic p-value"
                ),
            },
        },
        "forecast_h3": {
            "task": "forecast occupancy_pct of month t+1 for each star class from data up to t",
            "horizon_months": 1,
            "test_origins": [TEST_ORIGIN_FIRST, TEST_ORIGIN_LAST],
            "validation_targets": [list(window) for window in VALIDATION_TARGET_WINDOWS],
            "training_rows": (
                "target months up to the origin in the pre-pandemic and post-pandemic "
                "periods; lagged features keep their observed values even when the lagged "
                "month is in the pandemic period"
            ),
            "leakage_rules": [
                "no variable dated at the target month enters a forecast, except the "
                "known calendar features",
                "typhoon days of the target month are never a feature",
                "nothing is scaled, encoded, or selected using rows after the origin",
                "orders and hyperparameters are chosen on validation origins only",
                "each test origin is forecast once, after those choices are fixed",
            ],
            "information_set": (
                "pseudo-out-of-sample: the forecast of month t+1 uses the DSEC values of "
                "months up to t as published on the download date. DSEC's publication lag "
                "and later revisions are not reproduced, so the errors are not those of a "
                "forecast made in real time. The holiday features are the exception: they "
                "follow the notices published by the end of month t."
            ),
            "models": {
                "seasonal_naive": "forecast = occupancy of the same month one year earlier",
                "sarimax": {
                    "fit": "one model per star class, refit at every origin",
                    "pandemic_months": "occupancy of 2020-01 to 2022-12 treated as missing",
                    "exogenous": ["cn_holiday_days_at_origin", "cn_golden_week_days_at_origin"],
                    "orders": [list(order) for order in SARIMAX_ORDERS],
                    "seasonal_orders": [list(order) for order in SARIMAX_SEASONAL_ORDERS],
                    "trend": "constant when no differencing is used, none otherwise",
                    "selection": (
                        "per star class, lowest validation MAE; a candidate that fails to "
                        "converge at any validation origin is dropped; ties go to the "
                        "earlier candidate in the lists above; AIC/BIC are not used "
                        "because the candidates differ in differencing"
                    ),
                    "fallback": {"order": [1, 0, 0], "seasonal_order": [0, 1, 1, 12]},
                },
                "lightgbm": {
                    "fit": "one global model for the four classes, refit at every origin",
                    "features": forecast_feature_names(),
                    "categorical": ["tier"],
                    "lags_months": list(FORECAST_LAGS),
                    "lagged_variables": list(LAGGED_VARIABLES),
                    "known_calendar_features": list(KNOWN_CALENDAR_FEATURES),
                    "grid": {name: list(values) for name, values in LIGHTGBM_GRID.items()},
                    "fixed": LIGHTGBM_FIXED,
                    "selection": (
                        "lowest validation MAE pooled over classes; ties go to the "
                        "earlier grid point"
                    ),
                },
            },
            "metrics": {
                "scale": "percentage points of occupancy",
                "reported": ["MAE", "RMSE", "MASE"],
                "by": ["each star class", "pooled over the four classes"],
                "mase_scale": (
                    "per star class, mean absolute 12-month difference of occupancy over "
                    "months up to 2024-06 for which both the month and the month one year "
                    "earlier are in the main sample; fixed for all origins"
                ),
            },
            "tests": {
                "method": "Diebold–Mariano with the Harvey–Leybourne–Newbold correction",
                "loss": "absolute error",
                "horizon": 1,
                "sides": "two-sided",
                "comparisons": ["lightgbm vs seasonal_naive", "lightgbm vs sarimax"],
                "units": ["each star class", "pooled: mean loss over the four classes per origin"],
                "multiplicity": "Holm over all ten tests (2 comparisons × 5 units)",
                "h3_rule": (
                    "H3 is supported if LightGBM has the lower pooled test MAE against "
                    "both baselines and both pooled Holm-adjusted p-values are below "
                    "0.05. Otherwise H3 is reported as not supported, with the MAE "
                    "differences and their 95% CIs."
                ),
            },
            "sensitivity_analyses": {
                "F1_logit": (
                    "SARIMAX and LightGBM fit on logit(occupancy_pct / 100); forecasts "
                    "are back-transformed and all metrics are computed in percentage points"
                ),
                "F2_pandemic_included": (
                    "pandemic-period target months kept in LightGBM training and observed "
                    "in SARIMAX"
                ),
            },
            "shap": {
                "method": "TreeSHAP, tree-path-dependent",
                "model_explained": "the LightGBM model fit at each test origin",
                "rows_explained": "that origin's four test forecasts",
                "background": "the training window of that origin only",
                "output": "the forecast in percentage points",
                "check": "base value + sum of SHAP values equals the forecast within 1e-6",
                "reading": "model behaviour only",
            },
        },
        "reproducibility": {"seed": RANDOM_SEED, "python": ">=3.11", "backend": "MPLBACKEND=Agg"},
    }
