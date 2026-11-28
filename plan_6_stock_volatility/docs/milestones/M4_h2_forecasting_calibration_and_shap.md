# Milestone M4: H2 Out-of-Sample Forecasting, 2024 Val Calibration, HLN-DM/Holm & Train-Background TreeSHAP (里程碑 M4：H2 样本外预测回测、验证集概率校准、HLN-DM/Holm 检验与 TreeSHAP 解释)

- **Milestone**: `M4 — H2 Out-of-Sample Forecasting, 2024 Val Calibration, HLN-DM/Holm & Train-Background TreeSHAP (H2 样本外预测回测、2024 验证集概率校准、HLN-DM/Holm 检验与 TreeSHAP 解释)`
- **Status**: `COMPLETED & PUSHED` *(Current Session Boundary — Stop After M4)*
- **Parent / Baseline Commit SHA**: `d7a35b611e53366b4d658f0ccaaa8713d3fed267` (`M3`)
- **Date (UTC)**: `2026-11-23`
- **Frozen Source Provenance Timestamp (UTC)**: `2026-11-22T14:04:28+00:00`
- **Verification Verdict**: `INDEPENDENT VERIFICATION PASS` (`ruff check .`: 0 errors; `pytest -v`: `60 passed` [26 M0 + 9 M1 + 8 M2 + 8 M3 + 9 M4]; `300/300` GJR-GARCH(1,1) fits converged; TreeSHAP additivity error `4.07e-8` [clf] and `8.35e-8` [reg] `< 1e-6`; `git check-ignore` exit `1`; `0` sibling diffs; `0` secrets)

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

Milestone M4 implements and audits the complete H2 out-of-sample forecasting, probability calibration, statistical hypothesis testing, and TreeSHAP explainability pipeline strictly on M2's verified `data/frozen/` bundle (`verify_freeze_manifest()` called at startup):

1. **Purged Quarterly Expanding-Window Schedule (`src/stock_volatility/baselines.py`)**:
   - **Stage 1 (`val_2024`)**: Fit strictly on `Train (2019-02-01..2023-12-20)`; evaluate on `2024 Val (2024-01-02..2024-12-20, 246 days / 14,760 rows)` for hyperparameter tuning and calibrator selection ONLY (`embargo_1` excluded, `6` purge days `>= 5`).
   - **Stage 2 (`2025 Holdout Test`, evaluated ONCE)**: Quarterly expanding refits (`2025Q1`, `2025Q2`, `2025Q3`, `2025Q4`, totaling `245` scorable days / `14,700` rows; final 5 incomplete tail days excluded) with strict `>= 5` trading-day purge before every quarter start (`fit_end_idx + 5 < eval_start_idx`) and both embargoes excluded from fitting.
2. **Econometric Baselines (`baselines.py`)**: Persistence (`rv_w` & expanding base rate), per-stock `GJR-GARCH(1,1)` (`arch`, `100*r` scaling, causal post-fit variance filtering, closed-form 5-day volatility & analytical tail probability, explicit convergence log across all 300 fits), and Predictive Stock-FE Panel HAR-RV (+ fit-fold Duan smearing) & Logistic HAR-RV.
3. **Global LightGBM & 2024 Val OOF Calibration (`forecast.py`)**: Tune 6-config LightGBM Regressor (by 2024 Val QLIKE) and 6-config LightGBM Classifier (by 2024 Val PR-AUC); select calibrator among `{isotonic, platt_sigmoid, temperature_scaling}` strictly via **4-quarter out-of-fold CV inside 2024 Val** to prevent in-sample Isotonic overfitting, refit the winning calibrator on full 2024 Val, and apply it frozen to 2025 Test.
4. **Formal Inference & Train-Background TreeSHAP (`evaluation.py`, `explain.py`, `scripts/05_forecast_backtest.py`)**: Variance-scale QLIKE, RMSE, MAE, PR-AUC (primary), ROC-AUC, Brier, 10-bin ECE (pre & post calibration), date-block bootstrap 95% CIs (`block_len=5, B=1000, seed=42`), multi-step `h=5` HLN-modified Diebold–Mariano tests with Bartlett PSD variance (`max_lag=4`) + Holm–Bonferroni correction, sector & event-day breakdowns, and exact interventional Train-background (`N=1000`) TreeSHAP.

---

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

### 2.1 Source Code, CLI, Tests & Documentation

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Role & Interface Contract |
| --- | --- | --- |
| `src/stock_volatility/baselines.py` | Source | `MODEL_FEATURES` (`22` cols), `duan_smearing_backtransform()`, `gjr_garch_5d_annualized_vol()`, `build_quarterly_expanding_folds()`, `slice_fold_data()`, `fit_predict_har_rv_fold()`, `fit_predict_gjr_garch_fold()`, `run_baselines_all_folds()` |
| `src/stock_volatility/forecast.py` | Source | `REGRESSOR_GRID` (`6`), `CLASSIFIER_GRID` (`6`), `fit_and_apply_calibrator()`, `select_calibrator_on_val_oof()`, `run_lgbm_val_tuning_and_test_backtest()` |
| `src/stock_volatility/evaluation.py` | Source | `qlike_loss_array()`, `regression_metrics()`, `calibration_bins_table()`, `expected_calibration_error()`, `classification_metrics()`, `date_block_bootstrap_ci()`, `hln_dm_test()`, `evaluate_holdout_predictions()` |
| `src/stock_volatility/explain.py` | Source | `compute_train_background_treeshap()` (`bg=1000` strictly from `train`, `eval=2000` from `test`, `classes_ == [0, 1]` check, additivity `< 1e-6`, High/Low VIX & event regime breakdown) |
| `scripts/05_forecast_backtest.py` | CLI | Verifies `FREEZE_MANIFEST.json` and runs full deterministic M4 pipeline emitting all 14 artifacts |
| `tests/test_forecast.py` | Test | 4 tests verifying `>= 5d` purged quarterly folds, Duan smearing & GJR-GARCH formulas, all 3 probability calibrators, and committed prediction/convergence/selection contracts |
| `tests/test_evaluation_and_shap.py` | Test | 5 tests verifying exact QLIKE/ECE math, `h=5` HLN-DM & date-block bootstrap determinism, committed 2025 Holdout metrics/DM/calibration, Train-background TreeSHAP additivity, and M4 milestone doc tracking |
| `docs/milestones/M4_h2_forecasting_calibration_and_shap.md` | Doc | This milestone delivery record & AI execution log (plus ledger updates in `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md`) |

### 2.2 Committed M4 Outputs & Cryptographic SHA-256 Digests

| File Path (Relative to `plan_6_stock_volatility/`) | Byte Size | SHA-256 Digest | Content Summary |
| --- | ---: | --- | --- |
| `outputs/tables/h2_val_predictions.parquet` | `1,110,913 B` | `0dd164ddb6be1b78161be2ba5e540956580ab2ffaa56fb95d018b2032295255d` | `14,760` rows (`2024 Val`) predictions for all 4 models (`zstd`) |
| `outputs/tables/h2_test_predictions.parquet` | `1,105,498 B` | `a8c2b6413a84badbf9b7946f281cc18b0202e043e0469bb45a406f5d1184b7ff` | `14,700` rows (`2025 Holdout Test`) predictions for all 4 models (`zstd`) |
| `outputs/tables/h2_test_metrics.csv` | `2,205 B` | `382af55fa84659b60ab45ea90bc28ae857638044de6923945c55b5e3af67af75` | 4 models × 7 metrics with date-block bootstrap 95% CIs + pre-cal Brier/ECE |
| `outputs/tables/h2_test_by_sector.csv` | `3,393 B` | `a1eb3c29af823b4034bbbcc04ec33ae8b849273035321775467f2b5a6697bc84` | 5 sectors × 4 models 2025 Test breakdown |
| `outputs/tables/h2_test_by_event_day.csv` | `2,084 B` | `bc14e5ef177b5e217ee830bbfbdbe7393905b1112fda52b18b87ac9be2c2520c` | `No_Event` / `Has_Event` / `Has_8K_Earnings` × 4 models breakdown |
| `outputs/tables/h2_dm_tests.csv` | `2,109 B` | `b2374ac59cc270727d91955543c10bc27b13ed2505d2c399f161c2a3a5b68da5` | `h=5` HLN-DM + Holm tests (`lgbm` vs 3 baselines across QLIKE, SE, Brier) |
| `outputs/tables/h2_calibration_curve.csv` | `5,148 B` | `b15d12cce1e48a1602bda5bf48d0a6af1d569dc1f2fb8c84a6baba47d10b5a80` | 10-bin reliability tables for Val & Test (`lgbm_raw`, `lgbm_cal`, `har_rv`, `gjr_garch`) |
| `outputs/tables/h2_garch_convergence.csv` | `41,568 B` | `6302ce7faead78d6ad5b04c28103c1a8220d9f6b1cbd35f26be44f201dadf890` | Full audit of all `300` per-stock GJR-GARCH(1,1) fits (`300/300` converged) |
| `outputs/tables/shap_importance.csv` | `4,838 B` | `92e340a076193be7ab514d81fcbfb33d33ae03ad6dad10c004008996de4d09ea` | Global `mean_abs_shap`, `share_pct`, `mean_shap` for 22 features (clf & reg) |
| `outputs/tables/shap_regime_summary.csv` | `17,565 B` | `c6ee243dfb338283f3bdf0be1c0567669b2cda864e18dfb85f3f4fbce96d8247` | TreeSHAP breakdown by `Low_VIX`/`High_VIX` and `No_Event`/`Has_Event` |
| `outputs/metrics/h2_baselines.json` | `2,572 B` | `0ffbd8cb34dcea4eab9c599a3ff18d8f8c6a9433df79fce3417cf7bf30f2460e` | Expanding-fold purge & Duan smearing & GJR-GARCH convergence audit |
| `outputs/metrics/h2_lgbm_selection.json` | `6,168 B` | `622124a065ecc41e0dd93f905c49a9df9f013a1eb9a302ff85d1db66d91a64f6` | Full 2024 Val tuning grid, 4-quarter OOF calibrator CV & 2025 refit log |
| `outputs/metrics/h2_evaluation.json` | `10,039 B` | `f303d29275b2ee5ffcd906076f444f61b4a537d1c91fb2988ce9404822f73b73` | Machine-readable 2025 Holdout metrics, bootstrap CIs & HLN-DM results |
| `outputs/metrics/shap_audit.json` | `4,101 B` | `43d6bcd6d4e9cb983e59d1cf0ee712506ba1b45ca412c7f17bbc55d4cc5e972c` | Train-background TreeSHAP provenance, `classes_`, additivity & top features |

---

## 3. Technical Implementation & Key 2025 Holdout Findings (核心技术实现与 2025 独立测试集实证结果)

### 3.1 Single-Shot 2025 Holdout Test Leaderboard (`14,700` rows, `245` days, base rate `8.62%`)

| Model | QLIKE [95% CI] | RMSE [95% CI] | MAE | PR-AUC [95% CI] | ROC-AUC | Brier | ECE (Post / Pre Cal) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `persistence` | `0.3144` [`0.2130`, `0.4841`] | `0.1196` [`0.0815`, `0.1581`] | `0.0730` | `0.0721` [`0.0515`, `0.1037`] | `0.3922` | `0.0788` | `0.0023` / `0.0023` |
| `gjr_garch` | `0.3007` [`0.2412`, `0.3865`] | `0.1362` [`0.1176`, `0.1580`] | `0.0986` | `0.1271` [`0.0982`, `0.1687`] | `0.6309` | `0.0796` | `0.0264` / `0.0264` |
| `har_rv` | `0.2383` [`0.1584`, `0.3796`] | `0.0957` [`0.0691`, `0.1300`] | `0.0587` | **`0.1422`** [`0.1042`, `0.1905`] | **`0.6483`** | `0.0774` | `0.0070` / `0.0070` |
| `lgbm` *(cal)* | **`0.2306`** [`0.1543`, `0.3716`] | **`0.0951`** [`0.0687`, `0.1286`] | `0.0594` | `0.1231` [`0.0829`, `0.1770`] | `0.6413` | **`0.0773`** | `0.0102` / `0.3804` |

### 3.2 Multi-Step `h = 5` HLN-Modified Diebold–Mariano Tests (`lgbm` vs Baselines, Holm-Adjusted)

| Loss Metric | Baseline | Mean Loss Diff (`lgbm - base`) | `DM_HLN` (`df=244`) | Raw `p` | Holm `p` | Holm Sig (`α=0.05`) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `qlike` | `persistence` | `-0.08380` | `-5.11` | `6.51e-07` | `1.95e-06` | **YES** (`lgbm` better) |
| `qlike` | `gjr_garch` | `-0.07012` | `-2.58` | `1.04e-02` | `2.07e-02` | **YES** (`lgbm` better) |
| `qlike` | `har_rv` | `-0.00776` | `-1.50` | `1.34e-01` | `1.34e-01` | **NO** (directionally better) |
| `se` | `persistence` | `-0.00526` | `-2.02` | `4.42e-02` | `8.83e-02` | **NO** |
| `se` | `gjr_garch` | `-0.00952` | `-5.34` | `2.15e-07` | `6.46e-07` | **YES** (`lgbm` better) |
| `se` | `har_rv` | `-0.00011` | `-0.20` | `8.39e-01` | `8.39e-01` | **NO** |
| `brier` | `persistence` | `-0.00156` | `-2.43` | `1.57e-02` | `4.70e-02` | **YES** (`lgbm` better) |
| `brier` | `gjr_garch` | `-0.00234` | `-1.88` | `6.13e-02` | `1.23e-01` | **NO** |
| `brier` | `har_rv` | `-0.00017` | `-0.36` | `7.16e-01` | `7.16e-01` | **NO** |

### 3.3 Honest H2 Verdict, Calibration Audit & TreeSHAP Summary

- **Honest H2 Verdict (Mixed / Nuanced Support — Zero Post-Hoc Tuning on Test)**:
  - On **5-day volatility forecasting**, `lgbm` achieves the best overall **QLIKE (`0.2306`)** and **RMSE (`0.0951`)**, significantly beating Persistence (`p_holm = 1.95e-6`) and per-stock GJR-GARCH(1,1) (`p_holm = 0.0207` QLIKE, `6.46e-7` SE), while marginally outperforming Stock-FE Panel HAR-RV (`0.2383` QLIKE, `p_holm = 0.134`).
  - On **downside tail-risk classification**, `2024 Val` Isotonic calibration reduced `lgbm` 2025 Test **ECE by 97.3%** (`0.3804` → `0.0102`) and **Brier by 66.7%** (`0.2322` → `0.0773`, best overall Brier, `p_holm = 0.0470` vs Persistence). However, linear regularized Logistic HAR-RV achieved higher 2025 Test **PR-AUC (`0.1422` vs `0.1231`)**, particularly on Form 8-K earnings days (`0.2861` vs `0.0948`), highlighting both the upper-tail tie compression of piecewise-constant isotonic steps and the strong motivation for Phase B's 27B–32B QLoRA corporate-event text channel.
- **Train-Background Exact TreeSHAP (`N_bg = 1000` strictly from `2019–2023 Train`, `N_eval = 2000` from `2025 Test`)**:
  - Verified `classes_ == [0, 1]`, positive class `1` = downside tail risk, max additivity error `4.07e-8` (classifier) and `8.35e-8` (regressor), both `< 1e-6`.
  - Top model-behavior contributors (strictly non-causal): `rv_m` (`24.4%` clf / `28.6%` reg), `vix_close` (`11.9%` / `11.3%`), `t10y2y` (`11.5%` clf), `log_rv_m` (`8.2%` / `9.5%`), and `log_rv_w` (`8.5%` reg).

---

## 4. What the AI Agents Did in Milestone M4 (本里程碑内 AI 完成的工作全记录)

### 4.1 Pre-Implementation Benchmarks & Anti-Leakage Architecture

- Benchmarked per-stock `GJR-GARCH(1,1)` (`arch==9.0.0`) on 60 stocks (`60/60` converged in `2.39s`), LightGBM Regressor/Classifier (`~2.3s` per fit under `OMP_NUM_THREADS=1`), and interventional probability-space TreeSHAP (`max_additivity_err = 4.1e-8`).
- Engineered **4-quarter out-of-fold CV strictly inside 2024 Val** for calibrator selection (`isotonic` vs `platt_sigmoid` vs `temperature_scaling`) so flexible isotonic calibration is judged strictly out-of-fold before being refit on full 2024 Val and frozen for single-shot 2025 Holdout Test.

### 4.2 Design & Engineering Traps Resolved

1. **Pure 1-Parameter Temperature Scaling vs 2-Parameter Platt Scaling**: Kept `temperature_scaling` strictly 1-parameter (`sigmoid(logit(p)/T)`) distinct from 2-parameter `platt_sigmoid` (`sigmoid(a*logit(p)+b)`). Because `clf_l15_m100_wbal` uses balanced class weights (`scale_pos_weight ≈ 10.5`), raw probabilities are shifted upward in intercept, which 2024 Val OOF CV accurately detected (`temperature_scaling` OOF Brier `0.1439` vs `isotonic` `0.0573` and `platt_sigmoid` `0.0576`), automatically choosing `isotonic` and dropping 2025 Test ECE from `0.3804` to `0.0102`.
2. **Causal Post-Fit GJR-GARCH Variance Filtering Across Purge Gaps**: Ensured that after estimating `(ω, α, γ, β)` strictly on `date <= fit_end_date`, `fit_predict_gjr_garch_fold()` filters conditional variance causally day-by-day across the 5–6 purge days and evaluation quarter using ONLY realized returns `<= t`, never touching `t+1`.

### 4.3 TDD & Independent Verification (`test-driven-development`, `verification-before-completion`)

1. **RED Phase**: Authored `tests/test_forecast.py` (`4` tests) and `tests/test_evaluation_and_shap.py` (`5` tests) first; verified `9 failed in 0.20s`.
2. **GREEN Phase**: Implemented `evaluation.py`, `baselines.py`, `forecast.py`, `explain.py`, and `scripts/05_forecast_backtest.py`; resolved 16 `ruff` E501 line-length items; verified `5/5` pure unit tests passed (`3.22s`) and executed full `05_forecast_backtest.py` (`300/300` GJR-GARCH converged, TreeSHAP additivity `< 1e-7`).
3. **Final Verification**: Verified `ruff check .` (`0` errors), full `pytest -v` (`60 passed`), `git check-ignore` exit `1`, zero sibling diffs, all files `< 25 MB`, and zero secrets prior to GitHub push.

---

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd /home/antigravity/Lc-wing/data_science_project/plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/05_forecast_backtest.py
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
```

---

## 6. Git Hygiene & Delivery Checklist (Git 交付检查清单)

- [x] Dual-content record complete (M4 deliverables, SHA-256 digests & 2025 Holdout findings + AI execution log & traps resolved)
- [x] `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md` updated and linked
- [x] `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` exited `1`
- [x] Zero sibling diffs in `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`, and `data/`
- [x] All staged files `< 25 MB` (max `h2_val_predictions.parquet` = `1.06 MB`), zero `.agents/` or secret leakage
