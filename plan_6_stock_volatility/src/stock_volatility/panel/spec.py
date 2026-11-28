"""Pre-registered analysis_spec.json, feature_manifest.json, and data_dictionary.md builders."""

from __future__ import annotations

from typing import Any

from stock_volatility import config
from stock_volatility.panel.events import (
    EARLY_CLOSE_HOUR,
    EVENTS_PHASE_B_COLUMNS,
    NYSE_EARLY_CLOSE_DATES,
)
from stock_volatility.panel.features import (
    CONTINUOUS_FEATURE_COLUMNS,
    EVENT_FEATURE_COLUMNS,
    PANEL_DAILY_COLUMNS,
    SECTOR_DUMMY_COLUMNS,
    TARGET_COLUMNS,
)


def build_analysis_spec(
    *,
    generated_at_utc: str,
    split_metadata: dict[str, Any],
    train_params: dict[str, Any],
) -> dict[str, Any]:
    """Build pre-registered Phase A & Phase B analysis specification locked before M3/M4."""
    return {
        "schema_version": "1.0.0",
        "project": "plan_6_stock_volatility",
        "phase": "Phase A (CPU-reproducible; frozen interface for Phase B A100 QLoRA)",
        "generated_at_utc": generated_at_utc,
        "random_seed": config.RANDOM_SEED,
        "date_window": {"start": config.START_DATE, "end": config.END_DATE},
        "horizon_days": config.HORIZON_DAYS,
        "rv_eps": config.RV_EPS,
        "tail_return_threshold": config.TAIL_RETURN_THRESHOLD,
        "rv90_train_threshold": round(float(train_params["rv90_threshold"]), 8),
        "cutoff_tz": config.CUTOFF_TZ,
        "cutoff_hour": config.CUTOFF_HOUR,
        "early_close_hour": EARLY_CLOSE_HOUR,
        "early_close_dates": sorted(NYSE_EARLY_CLOSE_DATES),
        "max_events_per_day": config.MAX_EVENTS_PER_DAY,
        "universe": {
            "n_tickers": len(config.UNIVERSE_TICKERS),
            "n_sectors": len(config.SECTORS),
            "sectors": list(config.SECTORS),
            "tickers": list(config.UNIVERSE_TICKERS),
            "zero_2025_lookahead": True,
        },
        "research_question": (
            "Across 60 liquid cross-sector US equities (2019-2025), how accurately do "
            "econometric volatility models (HAR-RV, GJR-GARCH) and nonlinear ML (LightGBM) "
            "forecast 5-day forward realized volatility and downside tail-risk events under "
            "strict purged+embargoed time-series splits, and does pre-close corporate event "
            "representation via 27B-64B QLoRA LLM fine-tuning (Phase B) add incremental lift?"
        ),
        "scope_and_disclaimer": (
            "Associational and predictive risk-modeling research for CS482; no causal claims "
            "and strictly not investment or trading advice."
        ),
        "hypotheses": {
            "H1": {
                "type": "confirmatory_associational",
                "statement": (
                    "Lagged negative return (leverage effect) and lagged VIX are positively "
                    "associated with 5-day forward log realized volatility, with heterogeneity "
                    "between growth/Technology and defensive sectors."
                ),
                "estimator": "Stock-FE extended Panel HAR-RV with Bartlett Driscoll-Kraay SE",
                "confirmatory_family": [
                    "neg_ret_1d",
                    "vix_close",
                    "tech_x_neg_ret_1d_vs_healthcare",
                    "tech_x_vix_close_vs_healthcare",
                    "joint_5sector_interaction_wald",
                ],
                "multiple_testing_correction": "Holm-Bonferroni",
                "sensitivities": [
                    "train_only_2019_2023",
                    "close_to_close_rv_5d_cc_fwd",
                    "newey_west_entity_clustered_se",
                ],
            },
            "H2": {
                "type": "confirmatory_predictive",
                "statement": (
                    "Under purged+embargoed expanding-window backtesting, global LightGBM "
                    "outperforms Persistence, per-stock GJR-GARCH(1,1), and Panel HAR-RV on "
                    "2025 holdout QLIKE/RMSE and downside tail-risk PR-AUC."
                ),
                "baselines": [
                    "persistence_rv_w",
                    "gjr_garch_11_per_stock",
                    "panel_har_rv_and_logistic",
                ],
                "ml_models": [
                    "global_lightgbm_regressor_duan_smearing",
                    "global_lightgbm_classifier_val_calibrated",
                ],
                "calibration_candidates_on_val_2024": [
                    "isotonic",
                    "platt_sigmoid",
                    "temperature_scaling",
                ],
                "test_refit_schedule": "quarterly_expanding_2025_Q1_Q4_with_5d_purge",
                "regression_metrics": ["QLIKE_variance_scale", "RMSE", "MAE"],
                "classification_metrics": ["PR_AUC_primary", "ROC_AUC", "Brier", "ECE"],
                "inference": {
                    "dm_test": "HLN_modified_DM_h5_Bartlett_PSD_max_lag_4",
                    "multiple_testing": "Holm-Bonferroni",
                    "bootstrap": {"type": "date_block", "block_len": 5, "B": 1000, "seed": 42},
                },
                "explainability": {
                    "method": "TreeSHAP",
                    "background_split": "train_2019_2023_only",
                    "background_n": 1000,
                    "seed": 42,
                    "additivity_tol": 1e-6,
                },
            },
            "H3": {
                "type": "phase_b_extension_frozen_handoff",
                "statement": (
                    "Event-anchored 4-bit QLoRA fine-tuning of 27B-64B open LLMs on frozen "
                    "pre-close SEC EDGAR event text provides incremental 2025 tail-risk lift "
                    "over price-only LightGBM and FinBERT (110M); null results accepted."
                ),
                "frozen_text_input": "data/frozen/events_text_phase_b.parquet",
                "event_anchored_only": True,
                "max_events_per_day": config.MAX_EVENTS_PER_DAY,
            },
        },
        "targets": {
            "primary_regression": {
                "raw_scale": "rv_5d_fwd",
                "model_scale": "y_log_rv",
                "definition": (
                    "sqrt((252 / (5 * 4 * ln(2))) * sum_{k=1..5} (ln(H_{t+k}/L_{t+k}))^2)"
                ),
                "back_transform": "Train-fold Duan smearing: max(exp(y_hat)*smear - 1e-6, 1e-4)",
            },
            "secondary_regression": {
                "raw_scale": "rv_5d_cc_fwd",
                "definition": (
                    "sqrt((252 / 5) * sum_{k=1..5} (ln(AdjClose_{t+k}/AdjClose_{t+k-1}))^2)"
                ),
            },
            "primary_classification": {
                "column": "y_tail",
                "definition": "1 iff (AdjClose_{t+5}/AdjClose_t - 1) < -0.05 else 0",
            },
            "secondary_classification": {
                "column": "y_tail_rv90",
                "definition": "1 iff rv_5d_fwd >= Train 90th percentile threshold else 0",
                "train_rv90_threshold": round(float(train_params["rv90_threshold"]), 8),
            },
        },
        "splits_frozen": {
            "train": {"start": config.START_DATE, "end": config.TRAIN_END},
            "embargo_1_days": split_metadata["embargo_1_trading_days"],
            "val": {"start": config.VAL_START, "end": config.VAL_END},
            "embargo_2_days": split_metadata["embargo_2_trading_days"],
            "test": {"start": config.TEST_START, "end": config.TEST_END},
            "test_scorable_days": split_metadata["test_scorable_trading_days"],
            "test_incomplete_tail_days": split_metadata["test_incomplete_tail_days"],
            "zero_target_overlap_verified": split_metadata["zero_target_overlap_verified"],
        },
    }


def build_feature_manifest(
    *,
    generated_at_utc: str,
    train_params: dict[str, Any],
    event_audit: dict[str, Any],
    dedup_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build machine-readable feature schema, lag guarantees, and Train-only preprocessing stats."""
    return {
        "schema_version": "1.0.0",
        "generated_at_utc": generated_at_utc,
        "panel_daily_columns": list(PANEL_DAILY_COLUMNS),
        "events_text_phase_b_columns": list(EVENTS_PHASE_B_COLUMNS),
        "primary_key": ["ticker", "date"],
        "identifiers_and_flags": [
            "ticker",
            "sector",
            "sub_group",
            "date",
            "split",
            "has_full_features",
            "has_full_target",
        ],
        "targets_future_t_plus_1_to_t_plus_5": list(TARGET_COLUMNS),
        "predictors_strictly_le_t": {
            "continuous": list(CONTINUOUS_FEATURE_COLUMNS),
            "event": list(EVENT_FEATURE_COLUMNS),
            "sector_dummies": list(SECTOR_DUMMY_COLUMNS),
        },
        "anti_leakage_invariants": {
            "price_features_max_timestamp": "date t NYSE close",
            "macro_features_fill_policy": "backward-only ffill on full FRED history, never bfill",
            "event_cutoff_rule": (
                "strictly < 16:00:00 America/New_York on normal NYSE days and "
                "< 13:00:00 America/New_York on 15 NYSE scheduled early-close days; "
                "at/after close or non-trading days shifted forward to next NYSE trading day"
            ),
            "event_capping_rule": (
                "intra-day normalized headline dedup (keep latest) + cap at <= 5 latest items/day"
            ),
            "preprocessing_fit_split": "train_2019_2023_only",
            "titles_capped_5_encoding": "JSON array string of up to 5 ordered headline strings",
        },
        "train_preprocessing": train_params,
        "event_alignment_summary": event_audit,
        "event_dedup_and_cap_summary": dedup_audit,
    }


def render_data_dictionary_markdown(
    *,
    generated_at_utc: str,
    split_metadata: dict[str, Any],
    train_params: dict[str, Any],
    event_audit: dict[str, Any],
    dedup_audit: dict[str, Any],
    target_prevalence: dict[str, dict[str, Any]],
) -> str:
    """Render human-readable frozen data dictionary Markdown for Phase A & Phase B handoff."""
    splits = split_metadata["splits"]
    lines: list[str] = [
        "# Plan 6 Phase A → Phase B Frozen Data Dictionary",
        "",
        f"- **Generated (Provenance UTC)**: `{generated_at_utc}`",
        "- **Universe**: `60` pre-2019 S&P 500 equities (`5` GICS sectors × `12` stocks)",
        f"- **Canonical NYSE Calendar**: `{config.START_DATE}` to `{config.END_DATE}` "
        f"(`{split_metadata['total_canonical_trading_days']}` trading days, "
        f"`{split_metadata['total_panel_rows']}` panel rows)",
        "- **Freeze Verification**: Call `stock_volatility.panel.verify_freeze_manifest()` "
        "before any M3–M5 or Phase B execution.",
        "",
        "---",
        "",
        "## 1. Purged & Embargoed Chronological Splits (`split_metadata.json`)",
        "",
        f"- **Horizon**: `h = {config.HORIZON_DAYS}` trading days (`t+1..t+5`).",
        f"- **Zero-Overlap Proof**: Last Train date `{split_metadata['train_last_date']}` "
        f"target ends on `{split_metadata['train_last_target_date']}` `<` Val start "
        f"`{split_metadata['val_first_date']}`; last Val date `{split_metadata['val_last_date']}` "
        f"target ends on `{split_metadata['val_last_target_date']}` `<` Test start "
        f"`{split_metadata['test_first_date']}`.",
        "",
        "| Split | Date Range | Trading Days | Total Rows | Modeling-Complete Days | "
        "Modeling-Complete Rows | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    notes_map = {
        "train": "First 21 days warm up 22-day `rv_m`; Train-only thresholds/scaling fit here",
        "embargo_1": "6-day purge/embargo gap (>= 5d); kept for lags, excluded from fitting",
        "val": "Hyperparameter tuning & probability calibration ONLY",
        "embargo_2": "6-day purge/embargo gap (>= 5d); kept for lags, excluded from fitting",
        "test": "Evaluated ONCE; final 5 days (`has_full_target=False`) excluded from scoring",
    }
    for name in ("train", "embargo_1", "val", "embargo_2", "test"):
        s = splits[name]
        lines.append(
            f"| `{name}` | `{s['start_date']}`..`{s['end_date']}` | "
            f"`{s['trading_days']}` | `{s['total_rows']}` | "
            f"`{s['modeling_complete_trading_days']}` | `{s['modeling_complete_rows']}` | "
            f"{notes_map[name]} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Event Cutoff (`16:00 ET` & `13:00 ET` Early-Close), Dedup & `<= 5`/Day Cap",
            "",
            "| Audit Metric | Value |",
            "| --- | ---: |",
            f"| Input SEC EDGAR filings (`events_raw.parquet`) | "
            f"`{event_audit['total_input_events']}` |",
            f"| Same-day pre-close (`< 16:00 ET` normal / `< 13:00 ET` early close) | "
            f"`{event_audit['on_trading_day_before_cutoff']}` |",
            f"| Shifted to `t+1` (normal day `>= 16:00 ET`) | "
            f"`{event_audit['shifted_after_1600_et']}` |",
            f"| Shifted to `t+1` (NYSE early-close day `13:00..15:59 ET`) | "
            f"`{event_audit['shifted_due_to_early_close']}` |",
            f"| Shifted to `t+1` (weekend / NYSE holiday) | "
            f"`{event_audit['shifted_non_trading_day']}` |",
            f"| Dropped after final calendar day cutoff (`2025-12-31 16:00 ET`) | "
            f"`{event_audit['dropped_after_end_date']}` |",
            f"| Duplicate intra-day headlines removed (kept latest) | "
            f"`{dedup_audit['dedup_headline_dropped']}` |",
            f"| Excess events removed by `<= 5`/day cap | "
            f"`{dedup_audit['capped_over_5_dropped']}` |",
            f"| Final retained events (`events_aligned.parquet`) | "
            f"`{dedup_audit['output_rows']}` |",
            f"| Event-anchored ticker-days (`has_news == 1`) | "
            f"`{dedup_audit['active_ticker_days']}` |",
            f"| Max events per ticker-day after cap | "
            f"`{dedup_audit['max_events_per_day_after_cap']}` |",
            "",
            "---",
            "",
            "## 3. Target Distributions & Train-Only Thresholds",
            "",
            f"- **Primary Tail-Risk Threshold**: `cum_ret_5d_fwd < {config.TAIL_RETURN_THRESHOLD}`",
            f"- **Secondary RV-90 Threshold (fitted strictly on Train)**: "
            f"`rv_5d_fwd >= {train_params['rv90_threshold']:.6f}`",
            "",
            "| Split | Modeling Rows | Median `rv_5d_fwd` | Mean `rv_5d_fwd` | "
            "`y_tail` Positives (%) | `y_tail_rv90` Positives (%) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("train", "val", "test"):
        tp = target_prevalence[name]
        lines.append(
            f"| `{name}` | `{tp['n_rows']}` | `{tp['rv_5d_fwd_median']:.4f}` | "
            f"`{tp['rv_5d_fwd_mean']:.4f}` | "
            f"`{tp['y_tail_pos']}` (`{tp['y_tail_rate']*100:.2f}%`) | "
            f"`{tp['y_tail_rv90_pos']}` (`{tp['y_tail_rv90_rate']*100:.2f}%`) |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. `data/frozen/panel_daily.parquet` Column Schema",
            "",
            "| Column | Role | Time Window | Definition / Formula |",
            "| --- | --- | --- | --- |",
            "| `ticker`, `sector`, `sub_group` | Identifier | Pre-2019 static | "
            "Deterministic 60-stock GICS metadata |",
            "| `date`, `split` | Key / Split | Day `t` | Canonical NYSE trading date & "
            "`train`/`embargo_1`/`val`/`embargo_2`/`test` |",
            "| `has_full_features`, `has_full_target` | Mask | `<= t` / `t+1..t+5` | "
            "Boolean completeness masks (warmup & tail-5 exclusion) |",
            "| `rv_5d_fwd` | Primary Target | `t+1..t+5` | "
            "Annualized 5-day Parkinson high-low volatility |",
            "| `y_log_rv` | Primary Target (Model) | `t+1..t+5` | `ln(rv_5d_fwd + 1e-6)` |",
            "| `rv_5d_cc_fwd` | Sensitivity Target | `t+1..t+5` | "
            "Annualized 5-day close-to-close log-return volatility |",
            "| `cum_ret_5d_fwd` | Return Target | `t+1..t+5` | `AdjClose_{t+5}/AdjClose_t - 1` |",
            "| `y_tail` | Primary Binary Target | `t+1..t+5` | "
            "Nullable `Int8`: `1` iff `cum_ret_5d_fwd < -0.05` (`NA` on last 5 days) |",
            "| `y_tail_rv90` | Sensitivity Binary Target | `t+1..t+5` | "
            "Nullable `Int8`: `1` iff `rv_5d_fwd >= Train p90` (`NA` on last 5 days) |",
            "| `rv_d`, `rv_w`, `rv_m` | HAR Predictors | `<= t` | "
            "1-day, 5-day, and 22-day annualized Parkinson RV |",
            "| `log_rv_d`, `log_rv_w`, `log_rv_m` | HAR Predictors | `<= t` | "
            "`ln(rv_* + 1e-6)` |",
            "| `ret_1d`, `neg_ret_1d`, `ret_5d` | Return / Leverage | `<= t` | "
            "1-day adj-close return, `min(ret_1d, 0)`, 5-day cumulative adj-close return |",
            "| `vol_ratio_20d`, `daily_range_pct` | Microstructure | `<= t` | "
            "`volume_t / mean_{20d}(volume)` and `(high_t - low_t) / close_t` |",
            "| `vix_close`, `t10y2y`, `spy_ret_1d` | Macro | `<= t` | "
            "Yahoo `^VIX`, backward-only `ffill` FRED `T10Y2Y`, `SPY` 1-day adj return |",
            "| `has_news`, `news_count`, `has_8k_earnings`, `news_source` | Events | "
            "`< 16:00 ET` (`< 13:00 ET` early close) | "
            "Pre-close indicator, deduplicated capped count (`0..5`), Form 8-K Item 2.02 flag, "
            "and source (`sec_edgar`/`none`) |",
            "| `sector_technology`..`sector_energy` | Sector Dummies | Pre-2019 static | "
            "One-hot binary indicators for the 5 GICS sectors |",
            "",
            "---",
            "",
            "## 5. `data/frozen/events_text_phase_b.parquet` Column Schema",
            "",
            "- Primary key `(ticker, date)` aligned 1-to-1 with `panel_daily.parquet` "
            "(`105,600` rows; filter `has_news == 1` for Phase B event-anchored QLoRA).",
            "- `titles_capped_5`: JSON-encoded array of up to `5` chronologically ordered, "
            "deduplicated pre-close corporate event headlines.",
            "- `text_concat_capped_5`: ` || `-delimited concatenation of `titles_capped_5` "
            "(empty string when `has_news == 0`).",
            "",
        ]
    )
    return "\n".join(lines)
