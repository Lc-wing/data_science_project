# Milestone M2: Panel Build, Purged Splits & Phase A→B SHA-256 Freeze (里程碑 M2：面板构建、防泄漏切分与 Phase A→B 冻结清单)

- **Milestone**: `M2 — Panel Build, Purged Splits & Phase A→B SHA-256 Freeze (面板构建、防泄漏切分与 Phase A→B 冻结清单)`
- **Status**: `COMPLETED & PUSHED`
- **Parent / Baseline Commit SHA**: `5370d1835215f5463b38f3855bffa96211ceba69` (`M1`)
- **Date (UTC)**: `2026-11-23`
- **Source Provenance Timestamp (UTC)**: `2026-11-22T14:04:28+00:00`
- **Verification Verdict**: `INDEPENDENT VERIFICATION PASS` (`ruff check .`: 0 errors; `pytest -v`: `43 passed` [26 M0 + 9 M1 + 8 M2]; `02_build_panel_and_freeze.py --check` verified `6/6` frozen SHA-256 digests; offline `--refreeze` bit-for-bit identical; `git check-ignore` exit `1`; `0` sibling diffs; `0` secrets)

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

Milestone M2 builds the strictly zero-leakage, CPU-reproducible modeling panel and locks the cryptographic Phase A → Phase B handoff contract on top of M1's verified raw/interim snapshots (`5370d1835215f5463b38f3855bffa96211ceba69`):

1. **Strict Pre-Close Event Alignment (`America/New_York`), Dedup & `<= 5`/Day Cap**: Convert all 7,703 SEC EDGAR UTC `acceptanceDateTime` timestamps to `America/New_York` (DST-aware), enforce `< 16:00:00 ET` on normal NYSE trading days and `< 13:00:00 ET` on all 15 scheduled NYSE early-close days (shifting at/after-close and weekend/holiday filings to the next NYSE trading day `t+1`), deduplicate identical intra-day headlines keeping the latest timestamp, cap each `(ticker, date)` at `<= 5` latest items, and emit both `data/interim/events_aligned.parquet` and frozen `data/frozen/events_text_phase_b.parquet`.
2. **Zero-Leakage Targets (`t+1..t+5`) & Strictly-Past Predictors (`<= t`)**: Construct annualized 5-day forward Parkinson RV (`rv_5d_fwd`, `y_log_rv = ln(rv_5d_fwd + 1e-6)`), close-to-close sensitivity target (`rv_5d_cc_fwd`), 5-day cumulative return (`cum_ret_5d_fwd`), primary binary tail target (`y_tail`: `cum_ret_5d_fwd < -0.05`), and Train-only 90th-percentile RV tail sensitivity (`y_tail_rv90`); construct strictly `<= t` HAR-RV, return/leverage, volume/range, macro, event, and 5-sector dummy features; fit all Winsorization (`p005`/`p995`), z-score (`mean`/`std`), and `rv90_threshold` parameters strictly on `2019–2023 Train`.
3. **Purged + 5-Day Embargo Chronological Splits & Pre-Registered Freeze**: Assign immutable `train / embargo_1 / val / embargo_2 / test` splits (`1,252 / 6 / 246 / 6 / 250` trading days), formally prove zero `t+1..t+5` target overlap across boundaries, lock `analysis_spec.json`, `split_metadata.json`, `feature_manifest.json`, and `data_dictionary.md`, and seal all 6 frozen artifacts with `FREEZE_MANIFEST.json` + `verify_freeze_manifest()`.

---

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

### 2.1 Source Code, CLI, Tests & Documentation

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Role & Interface Contract |
| --- | --- | --- |
| `src/stock_volatility/panel/__init__.py` | Source | Public M2 exports (`align_events_to_trading_days`, `dedup_and_cap_events`, `build_panel_features_and_targets`, `assign_purged_splits`, `validate_purged_embargo_splits`, `build_and_freeze_panel`, `verify_freeze_manifest`) |
| `src/stock_volatility/panel/events.py` | Source | `NYSE_EARLY_CLOSE_DATES` (15 half-days), `EARLY_CLOSE_HOUR = 13`, `align_events_to_trading_days()`, `dedup_and_cap_events()`, `build_daily_event_features_and_phase_b_text()` |
| `src/stock_volatility/panel/splits.py` | Source | `SPLIT_ORDER`, `assign_split_label()`, `assign_purged_splits()`, `validate_purged_embargo_splits()` (formal zero-overlap proof) |
| `src/stock_volatility/panel/features.py` | Source | `PANEL_DAILY_COLUMNS` (`36` cols), `fit_train_preprocessing_params()`, `apply_train_winsor_and_scale()`, `build_panel_features_and_targets()` |
| `src/stock_volatility/panel/spec.py` | Source | Deterministic builders for `analysis_spec.json`, `feature_manifest.json`, and `data_dictionary.md` |
| `src/stock_volatility/panel/freeze.py` | Source | `FREEZE_ARTIFACT_NAMES` (`6` files), `write_freeze_manifest()`, `verify_freeze_manifest()` (SHA-256 checked first, raises `ValueError` on tamper), `build_and_freeze_panel()` |
| `scripts/02_build_panel_and_freeze.py` | CLI | `uv run python scripts/02_build_panel_and_freeze.py [--check | --refreeze]` |
| `tests/test_panel_and_freeze.py` | Test | 8 comprehensive tests covering DST/16:00/13:00 early-close alignment, dedup & `<= 5`/day cap, analytical Parkinson formulas & future-shock anti-leakage perturbation, purged split overlap checks, freeze tamper detection, committed artifact contracts, offline bit-identical `--refreeze`, and M2 milestone doc checks |
| `docs/milestones/M2_panel_splits_and_freeze.md` | Doc | This milestone delivery record & AI execution log (plus ledger updates in `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md`) |

### 2.2 Committed Interim & Frozen Artifacts (All `< 15 MB`, zstd Parquet + SHA-256 Sealed)

| File Path (Relative to `plan_6_stock_volatility/`) | Rows / Items | Byte Size | SHA-256 Digest | Schema / Contract Summary |
| --- | ---: | ---: | --- | --- |
| `data/interim/events_aligned.parquet` | `7,674` | `345,464 B` | `8c0f4863ff4df0ec1d01684adc6752ed67332f3fe663735bf5fa98a2e08ec9cb` | Deduplicated, DST/early-close-aligned SEC EDGAR events (`<= 5`/ticker-day, max observed `4`) |
| `data/frozen/panel_daily.parquet` | `105,600` | `14,637,611 B` | `3515fcd12cabdbb0cfa2671010224e57a6a8d7d7ad498c51029537c5a225e051` | `36` columns: keys/splits/masks, `6` `t+1..t+5` targets, `14` continuous `<= t` predictors, `4` event features, `5` sector dummies |
| `data/frozen/events_text_phase_b.parquet` | `105,600` (`6,822` active) | `275,799 B` | `f07b663f0719359eb470c224ab7c9686dc580b3380b9e92ce16ca61d14c39045` | `[ticker, date, split, has_news, news_count, news_source, forms, item_codes, titles_capped_5, text_concat_capped_5]` |
| `data/frozen/analysis_spec.json` | `1` | `6,286 B` | `ac7dfb0155139440bbea86fa5ed2bb3c471798da9d3b7a236bb72f1c1862c604` | Pre-registered RQ, H1–H3, target/feature formulas, H1 DK+Holm & H2 backtest/HLN-DM/TreeSHAP specs |
| `data/frozen/split_metadata.json` | `1` | `4,832 B` | `c0e21ca3570a16867ba9817e2a4a1a4abc8b872654c66f437913def97f5bcecf` | Split boundaries, zero-overlap proof, early-close & dedup audits, per-split target prevalence |
| `data/frozen/feature_manifest.json` | `1` | `8,696 B` | `bfb13efc7a66dca209c8aa5a6632950c7a947edc753b5dc28c20178ee4b9fb0e` | Column roles, anti-leakage invariants, Train-only `rv90_threshold` (`0.374196`), Train `p005`/`p995`/`mean`/`std` |
| `data/frozen/data_dictionary.md` | `1` | `5,578 B` | `e7c6b6b6e28d13bb77a9791f9b04f5ce57cb52c3c921e0c8267a3254bb50aff3` | Human-readable Phase A → Phase B frozen data dictionary |
| `data/frozen/FREEZE_MANIFEST.json` | `6` files | `1,465 B` | `8aa3b379b8beef2b077d5c3e6651733be089eea45e0ac56b738de07a30aa3b1b` | Cryptographic SHA-256 & byte-size manifest verified by `verify_freeze_manifest()` |

---

## 3. Technical Implementation & Key Metrics (核心技术实现与关键指标)

### 3.1 Purged + 5-Day Embargo Split Audit (`split_metadata.json`)

- **Zero-Overlap Verification**:
  - Last `train` date `2023-12-20` → `t+5` target ends on `2023-12-28` `<` `val` start `2024-01-02` (`embargo_1` = `6` trading days, 1-day extra buffer beyond `h = 5`).
  - Last `val` date `2024-12-20` → `t+5` target ends on `2024-12-30` `<` `test` start `2025-01-02` (`embargo_2` = `6` trading days, 1-day extra buffer beyond `h = 5`).

| Split | Date Range | Trading Days | Total Rows | Modeling-Complete Days | Modeling-Complete Rows | Median `rv_5d_fwd` | `y_tail` (`< -5%`) | `y_tail_rv90` (`>= 0.374196`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `2019-01-02`..`2023-12-20` | `1,252` | `75,120` | `1,231` (`2019-02-01`..) | `73,860` | `0.2070` | `6,416` (`8.69%`) | `7,386` (`10.00%`) |
| `embargo_1` | `2023-12-21`..`2023-12-29` | `6` | `360` | `6` *(excluded from fit)* | `360` | — | — | — |
| `val` | `2024-01-02`..`2024-12-20` | `246` | `14,760` | `246` | `14,760` | `0.1835` | `911` (`6.17%`) | `502` (`3.40%`) |
| `embargo_2` | `2024-12-23`..`2024-12-31` | `6` | `360` | `6` *(excluded from fit)* | `360` | — | — | — |
| `test` | `2025-01-02`..`2025-12-31` | `250` | `15,000` | `245` (`..2025-12-23`; tail `5` `NA`) | `14,700` | `0.2147` | `1,267` (`8.62%`) | `1,219` (`8.29%`) |

### 3.2 Event Cutoff, Early-Close & Capping Summary

- **Input**: `7,703` SEC EDGAR filings (`5,089` EDT + `2,614` EST).
- **Pre-Close Same-Day (`< 16:00 ET` normal / `< 13:00 ET` early close)**: `3,123`.
- **Shifted to `t+1`**: `4,570` (`>= 16:00 ET` normal trading days) + `2` (`13:00..15:59 ET` on NYSE early-close days: `WFC` `0001387131-19-004828` at `2019-07-03T14:55:29-04:00` → `2019-07-05` and `WFC` `0001387131-19-009231` at `2019-11-29T13:02:31-05:00` → `2019-12-02`) + `8` (weekends/holidays); `0` dropped past `2025-12-31`.
- **Dedup & Cap**: `29` duplicate intra-day headlines removed (keeping latest timestamp); `0` rows exceeded `5`/day (max observed `4` after dedup); **`7,674`** retained filings across **`6,822`** event-anchored `(ticker, date)` rows.

---

## 4. What the AI Agents Did in Milestone M2 (本里程碑内 AI 完成的工作全记录)

### 4.1 M1 Pre-Check & Empirical Data Audit (M1 复核与实测探查)

Before writing any M2 code, the AI audited M1's committed Parquet/CSV files (`universe_60.csv`, `ohlcv_daily.parquet`, `macro_daily.parquet`, `events_raw.parquet`) against the Plan 6 v2.2 specification:
- Confirmed all 60 tickers × 1,760 NYSE trading days (`105,600` rows) have `0` `high <= low` anomalies and confirmed Yahoo Finance `auto_adjust=False` OHLC columns are already split-adjusted across `AAPL` (`4x`), `NVDA` (`10x`), and `AMZN` (`20x`) splits so `ln(H/L)` is split-invariant while `adj_close` captures total return including dividends.
- Verified canonical NYSE calendar counts (`1,252 / 6 / 246 / 6 / 250 = 1,760` days) and verified M1 provenance SHA-256 integrity (`0` mismatches).

### 4.2 Four Subtle Traps Discovered & Resolved by AI (发现的 4 个细微陷阱与工程解决)

1. **Trap 1 — NYSE 13:00 ET Early-Close Half-Day Leakage (`2` Live WFC Filings):**
   - *Discovery*: Auditing `events_raw.parquet` against the 15 NYSE scheduled 13:00 ET early-close dates (2019–2025) caught **2 real filings** filed after the 13:00 ET closing bell yet before 16:00 ET (`WFC` on `2019-07-03` at `14:55:29 EDT` and `2019-11-29` at `13:02:31 EST`). A naive `16:00 ET`-only rule would have leaked post-close filings onto the same day.
   - *Resolution*: Encoded `NYSE_EARLY_CLOSE_DATES` (all 15 half-days) and enforced `< 13:00:00 ET` on early-close days alongside `< 16:00:00 ET` on full days, shifting both WFC filings to `t+1` and logging them explicitly in `events_aligned.parquet` and `split_metadata.json`.
2. **Trap 2 — 29 Duplicate Intra-Day Boilerplate Form 8-K Headlines:**
   - *Discovery*: Inspection revealed 29 `(ticker, aligned_date, headline_text)` duplicates (e.g., `C` filing 3 consecutive `Items 9.01: Financial Statements and Exhibits` shells minutes apart after hours).
   - *Resolution*: Added whitespace/case-normalized intra-day headline deduplication keeping the latest `(acceptance_datetime_utc, accession_number)` prior to the `<= 5`/day cap.
3. **Trap 3 — Exact `rv_m` 22-Day Rolling Warmup Count (`21` Warmup Days → `1,231` Train Days) & Nullable Tail Labels:**
   - *Discovery*: A 22-day inclusive rolling Parkinson window `{t-21..t}` becomes defined on index `21` (`2019-02-01`), giving exactly `21` warmup days and `1,231` feature-complete Train days (`73,860` rows) rather than an informal `1,230` estimate. Meanwhile, casting `y_tail` and `y_tail_rv90` with nullable pandas `Int8` prevents the final 5 incomplete Test days (`2025-12-24`..`2025-12-31`) from silently turning into false `0` negatives.
   - *Resolution*: Documented exact `1,231`-day (`2019-02-01`..`2023-12-20`) modeling bounds in `split_metadata.json` and stored `y_tail`/`y_tail_rv90` as nullable `Int8` with verified `pd.NA` on all 5 incomplete tail days.
4. **Trap 4 — `verify_freeze_manifest()` Check Ordering & Compact zstd Parquet Compression:**
   - *Discovery*: During TDD GREEN verification, `test_freeze_manifest_verify_and_tamper_detection` caught that checking `byte_size` before `sha256` raised a byte-size error instead of a `SHA-256` mismatch error on different-length tampering; additionally, Snappy compression produced a `15.65 MB` `panel_daily.parquet`.
   - *Resolution*: Reordered `verify_freeze_manifest()` to verify `sha256` before `byte_size` and switched M2 Parquet writes to deterministic `zstd` compression (`panel_daily.parquet` reduced to `14.64 MB`, `events_text_phase_b.parquet` to `276 KB`, both bit-identical across `--refreeze`).

### 4.3 Implementation & TDD Workflow (`test-driven-development`)

1. **RED Phase**: Authored `tests/test_panel_and_freeze.py` (`8` tests) first before writing any code under `src/stock_volatility/panel/` and executed `pytest -v tests/test_panel_and_freeze.py`, confirming `8 failed in 0.25s` (`ModuleNotFoundError: No module named 'stock_volatility.panel'` and missing M2 milestone doc).
2. **GREEN Phase**: Implemented `src/stock_volatility/panel/{__init__,events,splits,features,spec,freeze}.py` and `scripts/02_build_panel_and_freeze.py`; fixed 12 ruff lint items and the `verify_freeze_manifest()` SHA-256 check order caught by TDD; executed `scripts/02_build_panel_and_freeze.py --refreeze` and `--check`.
3. **Documentation & Full-Suite Verification**: Authored `docs/milestones/M2_panel_splits_and_freeze.md` and updated `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md`, then re-ran `ruff check .`, full `pytest -v` (`43 passed`), and CLI `--check` / `--refreeze`.

### 4.4 Independent Verification Audit (`verification-before-completion`)

Executed freshly after final edits before committing to GitHub:
- `ruff check .`: `All checks passed!` (`0` errors).
- `pytest -v`: `43 passed` (`26` M0 + `9` M1 + `8` M2, `0` failures).
- `scripts/02_build_panel_and_freeze.py --check`: `status: "verified"`, `6/6` frozen SHA-256 digests matched; `--refreeze` bit-for-bit deterministic offline under `block_network`.
- `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/M2_panel_splits_and_freeze.md`: exit `1` (empty stdout, tracked by Git).
- Sibling isolation & hygiene: `0` diffs in `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`, and M0/M1 frozen files; max M2 file `13.96 MB` `< 25 MB`; `0` secret hits.

---

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd /home/antigravity/Lc-wing/data_science_project/plan_6_stock_volatility

# 1. Read-only cryptographic verification of frozen Phase A -> Phase B bundle
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/02_build_panel_and_freeze.py --check

# 2. Deterministic offline rebuild & hash verification (zero network calls)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/02_build_panel_and_freeze.py --refreeze

# 3. Ruff lint check (exits 0: "All checks passed!")
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .

# 4. Full pytest suite (26 M0 + 9 M1 + 8 M2 = 43 passed)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v

# 5. Confirm M2 milestone doc is tracked by Git (exits 1 with empty output)
cd /home/antigravity/Lc-wing/data_science_project && \
  git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/M2_panel_splits_and_freeze.md
```

---

## 6. Git Hygiene & Delivery Checklist (Git 交付检查清单)

- [x] Dual-content record complete (M2 deliverables, schemas, SHA-256 digests & metrics + chronological AI execution log & 4 traps resolved)
- [x] `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md` updated and linked
- [x] `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` exited `1` with empty output
- [x] Zero sibling diffs in `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, and `plans/`
- [x] All staged files `< 25 MB` (max `panel_daily.parquet` = `13.96 MB`), zero `.agents/` or secret leakage
