# Plan 6 Phase A → Phase B Frozen Data Dictionary

- **Generated (Provenance UTC)**: `2026-11-22T14:04:28+00:00`
- **Universe**: `60` pre-2019 S&P 500 equities (`5` GICS sectors × `12` stocks)
- **Canonical NYSE Calendar**: `2019-01-02` to `2025-12-31` (`1760` trading days, `105600` panel rows)
- **Freeze Verification**: Call `stock_volatility.panel.verify_freeze_manifest()` before any M3–M5 or Phase B execution.

---

## 1. Purged & Embargoed Chronological Splits (`split_metadata.json`)

- **Horizon**: `h = 5` trading days (`t+1..t+5`).
- **Zero-Overlap Proof**: Last Train date `2023-12-20` target ends on `2023-12-28` `<` Val start `2024-01-02`; last Val date `2024-12-20` target ends on `2024-12-30` `<` Test start `2025-01-02`.

| Split | Date Range | Trading Days | Total Rows | Modeling-Complete Days | Modeling-Complete Rows | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `train` | `2019-01-02`..`2023-12-20` | `1252` | `75120` | `1231` | `73860` | First 21 days warm up 22-day `rv_m`; Train-only thresholds/scaling fit here |
| `embargo_1` | `2023-12-21`..`2023-12-29` | `6` | `360` | `6` | `360` | 6-day purge/embargo gap (>= 5d); kept for lags, excluded from fitting |
| `val` | `2024-01-02`..`2024-12-20` | `246` | `14760` | `246` | `14760` | Hyperparameter tuning & probability calibration ONLY |
| `embargo_2` | `2024-12-23`..`2024-12-31` | `6` | `360` | `6` | `360` | 6-day purge/embargo gap (>= 5d); kept for lags, excluded from fitting |
| `test` | `2025-01-02`..`2025-12-31` | `250` | `15000` | `245` | `14700` | Evaluated ONCE; final 5 days (`has_full_target=False`) excluded from scoring |

---

## 2. Event Cutoff (`16:00 ET` & `13:00 ET` Early-Close), Dedup & `<= 5`/Day Cap

| Audit Metric | Value |
| --- | ---: |
| Input SEC EDGAR filings (`events_raw.parquet`) | `7703` |
| Same-day pre-close (`< 16:00 ET` normal / `< 13:00 ET` early close) | `3123` |
| Shifted to `t+1` (normal day `>= 16:00 ET`) | `4570` |
| Shifted to `t+1` (NYSE early-close day `13:00..15:59 ET`) | `2` |
| Shifted to `t+1` (weekend / NYSE holiday) | `8` |
| Dropped after final calendar day cutoff (`2025-12-31 16:00 ET`) | `0` |
| Duplicate intra-day headlines removed (kept latest) | `29` |
| Excess events removed by `<= 5`/day cap | `0` |
| Final retained events (`events_aligned.parquet`) | `7674` |
| Event-anchored ticker-days (`has_news == 1`) | `6822` |
| Max events per ticker-day after cap | `4` |

---

## 3. Target Distributions & Train-Only Thresholds

- **Primary Tail-Risk Threshold**: `cum_ret_5d_fwd < -0.05`
- **Secondary RV-90 Threshold (fitted strictly on Train)**: `rv_5d_fwd >= 0.374196`

| Split | Modeling Rows | Median `rv_5d_fwd` | Mean `rv_5d_fwd` | `y_tail` Positives (%) | `y_tail_rv90` Positives (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | `73860` | `0.2070` | `0.2390` | `6416` (`8.69%`) | `7386` (`10.00%`) |
| `val` | `14760` | `0.1835` | `0.2003` | `911` (`6.17%`) | `502` (`3.40%`) |
| `test` | `14700` | `0.2147` | `0.2411` | `1267` (`8.62%`) | `1219` (`8.29%`) |

---

## 4. `data/frozen/panel_daily.parquet` Column Schema

| Column | Role | Time Window | Definition / Formula |
| --- | --- | --- | --- |
| `ticker`, `sector`, `sub_group` | Identifier | Pre-2019 static | Deterministic 60-stock GICS metadata |
| `date`, `split` | Key / Split | Day `t` | Canonical NYSE trading date & `train`/`embargo_1`/`val`/`embargo_2`/`test` |
| `has_full_features`, `has_full_target` | Mask | `<= t` / `t+1..t+5` | Boolean completeness masks (warmup & tail-5 exclusion) |
| `rv_5d_fwd` | Primary Target | `t+1..t+5` | Annualized 5-day Parkinson high-low volatility |
| `y_log_rv` | Primary Target (Model) | `t+1..t+5` | `ln(rv_5d_fwd + 1e-6)` |
| `rv_5d_cc_fwd` | Sensitivity Target | `t+1..t+5` | Annualized 5-day close-to-close log-return volatility |
| `cum_ret_5d_fwd` | Return Target | `t+1..t+5` | `AdjClose_{t+5}/AdjClose_t - 1` |
| `y_tail` | Primary Binary Target | `t+1..t+5` | Nullable `Int8`: `1` iff `cum_ret_5d_fwd < -0.05` (`NA` on last 5 days) |
| `y_tail_rv90` | Sensitivity Binary Target | `t+1..t+5` | Nullable `Int8`: `1` iff `rv_5d_fwd >= Train p90` (`NA` on last 5 days) |
| `rv_d`, `rv_w`, `rv_m` | HAR Predictors | `<= t` | 1-day, 5-day, and 22-day annualized Parkinson RV |
| `log_rv_d`, `log_rv_w`, `log_rv_m` | HAR Predictors | `<= t` | `ln(rv_* + 1e-6)` |
| `ret_1d`, `neg_ret_1d`, `ret_5d` | Return / Leverage | `<= t` | 1-day adj-close return, `min(ret_1d, 0)`, 5-day cumulative adj-close return |
| `vol_ratio_20d`, `daily_range_pct` | Microstructure | `<= t` | `volume_t / mean_{20d}(volume)` and `(high_t - low_t) / close_t` |
| `vix_close`, `t10y2y`, `spy_ret_1d` | Macro | `<= t` | Yahoo `^VIX`, backward-only `ffill` FRED `T10Y2Y`, `SPY` 1-day adj return |
| `has_news`, `news_count`, `has_8k_earnings`, `news_source` | Events | `< 16:00 ET` (`< 13:00 ET` early close) | Pre-close indicator, deduplicated capped count (`0..5`), Form 8-K Item 2.02 flag, and source (`sec_edgar`/`none`) |
| `sector_technology`..`sector_energy` | Sector Dummies | Pre-2019 static | One-hot binary indicators for the 5 GICS sectors |

---

## 5. `data/frozen/events_text_phase_b.parquet` Column Schema

- Primary key `(ticker, date)` aligned 1-to-1 with `panel_daily.parquet` (`105,600` rows; filter `has_news == 1` for Phase B event-anchored QLoRA).
- `titles_capped_5`: JSON-encoded array of up to `5` chronologically ordered, deduplicated pre-close corporate event headlines.
- `text_concat_capped_5`: ` || `-delimited concatenation of `titles_capped_5` (empty string when `has_news == 0`).
